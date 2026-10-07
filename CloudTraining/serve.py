"""
Inference endpoint for the spiral classifier.

A small FastAPI app that loads the ONNX model exported by train.py and serves
predictions over HTTP. It deliberately does not import torch or the spiralnet
package, the ONNX file is the only thing it needs from training.

Run it locally with

    MODEL_DIR=bucket/runs/<run_id> uv run --with fastapi --with uvicorn uvicorn serve:app --port 8080
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))

# a point is exactly two numbers, and we cap the batch so one request can't eat the server
Point = Annotated[list[float], Field(min_length=2, max_length=2)]


class PredictRequest(BaseModel):
    points: Annotated[list[Point], Field(min_length=1, max_length=10_000)]


class PredictResponse(BaseModel):
    classes: list[int]
    probabilities: list[list[float]]
    model_version: str


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load once at start up, not per request. If the model is missing we want the
    # container to fail straight away rather than return errors later
    state["session"] = ort.InferenceSession(
        str(MODEL_DIR / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    metrics_file = MODEL_DIR / "metrics.json"
    state["version"] = (
        json.loads(metrics_file.read_text())["run_id"]
        if metrics_file.exists()
        else "unknown"
    )
    yield
    state.clear()


app = FastAPI(title="Spiral classifier", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": state["version"]}


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    points = np.asarray(request.points, dtype=np.float32)
    (logits,) = state["session"].run(["logits"], {"points": points})
    # softmax, subtracting the max first for numerical stability
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    return PredictResponse(
        classes=probabilities.argmax(axis=1).tolist(),
        probabilities=probabilities.astype(np.float64).round(4).tolist(),
        model_version=state["version"],
    )
