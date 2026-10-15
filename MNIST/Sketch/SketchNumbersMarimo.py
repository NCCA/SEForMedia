#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sketch Numbers

    This is the marimo version of the Qt `SketchNumbers.py` demo in this folder. Draw a digit in the black box
    below and when you let go of the mouse the image is shrunk down to 28x28 pixels and passed to the model we
    trained in `ReadDigitsTraining`. It uses exactly the same `mnist_model.pth` file as the Qt version so the
    predictions should match.

    Run it as an app (code hidden) or as a notebook with

    ```bash
    uv run marimo run MNIST/Sketch/SketchNumbersMarimo.py
    uv run marimo edit MNIST/Sketch/SketchNumbersMarimo.py
    ```

    The pen size slider and clear button live in the sketch pad itself (you can also press space to clear once you
    have clicked on it). The processing mode and border size are ordinary marimo widgets, so unlike the Qt version
    changing them re-runs the prediction straight away without having to draw again.
    """)
    return


@app.cell(hide_code=True)
def _(border_size, mo, prediction, processing, sketch):
    _controls = mo.vstack(
        [
            processing,
            border_size,
            mo.md("**What the model sees**"),
            prediction["preview"],
            prediction["stat"],
        ],
        gap=1,
    )
    mo.hstack(
        [sketch, _controls, prediction["plot"]], justify="start", gap=2, align="start"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The sketch pad

    marimo doesn't have a drawing widget built in, however it will happily host any
    [anywidget](https://anywidget.dev/), which is a Python class with a small chunk of JavaScript attached. The
    JavaScript draws white strokes onto a black HTML canvas (the same way round as the MNIST data) and when the mouse
    is released it encodes the canvas as a PNG [data URL](https://developer.mozilla.org/en-US/docs/Web/URI/Schemes/data)
    and sets the `image` trait. marimo sees the trait change and re-runs any cell that reads `sketch.value`, which is
    the equivalent of the `mouse_release` signal in the Qt `SketchWidget`.

    The canvas is 280x280 which is exactly 10 times the MNIST image size. The default pen is 20 pixels so the strokes
    end up roughly 2 pixels wide once shrunk down, which is about what the MNIST digits look like.
    """)
    return


@app.cell
def _(anywidget, traitlets):
    class SketchPad(anywidget.AnyWidget):
        """
        A simple black and white sketch pad for drawing digits.

        Attributes
        ----------
            image : str
                PNG data URL of the canvas, set each time the mouse is released ("" when cleared)
            pen_width : int
                width of the pen in canvas pixels
            size : int
                width and height of the square canvas in pixels
        """

        _esm = r"""
        function render({ model, el }) {
          const size = model.get("size");
          const canvas = document.createElement("canvas");
          canvas.width = size;
          canvas.height = size;
          canvas.tabIndex = 0;
          canvas.className = "sketch-canvas";
          const ctx = canvas.getContext("2d");

          const pen = document.createElement("input");
          pen.type = "range";
          pen.min = 1;
          pen.max = 30;
          pen.value = model.get("pen_width");
          const penLabel = document.createElement("span");
          penLabel.textContent = `Pen size ${pen.value}`;

          const clearButton = document.createElement("button");
          clearButton.textContent = "Clear";

          const controls = document.createElement("div");
          controls.className = "sketch-controls";
          controls.append(penLabel, pen, clearButton);
          el.append(canvas, controls);

          function clear() {
            ctx.fillStyle = "black";
            ctx.fillRect(0, 0, size, size);
            model.set("image", "");
            model.save_changes();
          }

          // the canvas may be scaled by CSS so map the mouse back to canvas pixels
          function position(event) {
            const rect = canvas.getBoundingClientRect();
            return [
              (event.clientX - rect.left) * (canvas.width / rect.width),
              (event.clientY - rect.top) * (canvas.height / rect.height),
            ];
          }

          let drawing = false;
          let last = [0, 0];

          function drawTo(point) {
            ctx.strokeStyle = "white";
            ctx.lineWidth = model.get("pen_width");
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            ctx.beginPath();
            ctx.moveTo(last[0], last[1]);
            ctx.lineTo(point[0], point[1]);
            ctx.stroke();
            last = point;
          }

          canvas.addEventListener("pointerdown", (event) => {
            if (event.button !== 0) return;
            canvas.setPointerCapture(event.pointerId);
            drawing = true;
            last = position(event);
            drawTo(last); // so a single click leaves a dot
          });
          canvas.addEventListener("pointermove", (event) => {
            if (drawing) drawTo(position(event));
          });
          function finish(event) {
            if (!drawing) return;
            drawing = false;
            model.set("image", canvas.toDataURL("image/png"));
            model.save_changes();
          }
          canvas.addEventListener("pointerup", finish);
          canvas.addEventListener("pointercancel", finish);
          canvas.addEventListener("keydown", (event) => {
            if (event.code === "Space") {
              event.preventDefault();
              clear();
            }
          });

          pen.addEventListener("input", () => {
            penLabel.textContent = `Pen size ${pen.value}`;
            model.set("pen_width", Number(pen.value));
            model.save_changes();
          });
          clearButton.addEventListener("click", clear);

          ctx.fillStyle = "black";
          ctx.fillRect(0, 0, size, size);
        }
        export default { render };
        """

        _css = r"""
        .sketch-canvas {
          border: 1px solid #888;
          cursor: crosshair;
          touch-action: none;
        }
        .sketch-controls {
          display: flex;
          align-items: center;
          gap: 0.5em;
          margin-top: 0.5em;
        }
        .sketch-controls button {
          border: 1px solid #888;
          border-radius: 4px;
          padding: 0.1em 0.8em;
          cursor: pointer;
        }
        """

        image = traitlets.Unicode("").tag(sync=True)
        pen_width = traitlets.Int(20).tag(sync=True)
        size = traitlets.Int(280).tag(sync=True)

    return (SketchPad,)


@app.cell
def _(SketchPad, mo):
    sketch = mo.ui.anywidget(SketchPad())
    processing = mo.ui.dropdown(
        options=["Bounding Box", "Centre of Mass"],
        value="Centre of Mass",
        label="Image processing",
    )
    border_size = mo.ui.slider(
        start=0, stop=15, value=4, label="Border size", show_value=True
    )
    return border_size, processing, sketch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading the model

    The model was saved with `torch.save(model, ...)` rather than just the `state_dict`, so it is a complete
    `nn.Sequential` (Flatten, then three Linear layers with ReLU between them) and we don't need to re-create the
    layers first. This needs `weights_only=False` as newer versions of PyTorch refuse to unpickle whole models by
    default. I use `mo.notebook_dir()` to find the file so it works no matter which folder you launch marimo from.
    """)
    return


@app.cell
def _(mo, torch):
    def get_device() -> torch.device:
        """
        Returns the appropriate device for the current environment.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():  # mac metal backend
            return torch.device("mps")
        else:
            return torch.device("cpu")

    device = get_device()
    model = torch.load(
        mo.notebook_dir() / "mnist_model.pth", map_location=device, weights_only=False
    )
    model.to(device)
    model.eval()
    return device, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Processing the image

    These do the same job as `get_image_tensor` and `center_image_by_mass` in `SketchWidget.py` but work on a NumPy
    array rather than a `QImage`, which makes the bounding box search a couple of `np.any` calls rather than looping
    over every pixel.

    - **Bounding Box** crops the drawing to the smallest box around the ink plus a border, then squashes that to
      28x28. As with the Qt version the aspect ratio is ignored, so a tall thin "1" gets stretched sideways.
    - **Centre of Mass** shrinks the whole canvas to 28x28 then shifts it so the centre of mass of the ink is in the
      middle of the image, which is closer to how the original MNIST digits were prepared.

    Both use nearest neighbour sampling to match Qt's `FastTransformation`. The model was trained on raw 0-255 pixel
    values (no normalisation) so that is what we pass in.
    """)
    return


@app.cell
def _(Image, base64, io, ndimage, np):
    def decode_image(data_url: str) -> np.ndarray:
        """
        Convert the PNG data URL from the sketch pad into a greyscale array.

        Parameters
        ----------
            data_url : str
                the "data:image/png;base64,..." string from the widget

        Returns
        -------
            np.ndarray
                uint8 array of shape (size, size)
        """
        _, encoded = data_url.split(",", 1)
        image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("L")
        return np.asarray(image)

    def bounding_box(image: np.ndarray, border: int) -> np.ndarray:
        """
        Crop to the drawn pixels plus a border and resize to 28x28.

        Parameters
        ----------
            image : np.ndarray
                full size greyscale sketch
            border : int
                number of canvas pixels to leave around the drawing

        Returns
        -------
            np.ndarray
                uint8 array of shape (28, 28)
        """
        rows = np.flatnonzero(np.any(image > 0, axis=1))
        cols = np.flatnonzero(np.any(image > 0, axis=0))
        # pad first so a border that runs off the edge of the canvas is just black
        padded = np.pad(image, border)
        cropped = padded[
            rows[0] : rows[-1] + 2 * border + 1, cols[0] : cols[-1] + 2 * border + 1
        ]
        return np.asarray(
            Image.fromarray(cropped).resize((28, 28), Image.Resampling.NEAREST)
        )

    def centre_of_mass(image: np.ndarray) -> np.ndarray:
        """
        Resize to 28x28 and shift so the centre of mass is in the middle.

        Parameters
        ----------
            image : np.ndarray
                full size greyscale sketch

        Returns
        -------
            np.ndarray
                uint8 array of shape (28, 28)
        """
        small = np.asarray(
            Image.fromarray(image).resize((28, 28), Image.Resampling.NEAREST)
        )
        com_y, com_x = ndimage.center_of_mass(small)
        centre_y, centre_x = np.array(small.shape) / 2
        # whole pixel shifts (as the Qt version does) so we don't blur the image
        shift = (int(centre_y - com_y), int(centre_x - com_x))
        return ndimage.shift(small, shift, order=0, cval=0)

    return bounding_box, centre_of_mass, decode_image


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Making the prediction

    The model outputs ten raw scores (logits), one per digit, and the biggest one is the prediction. The Qt version
    plots these raw values, here I pass them through a
    [softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html) first so the graph shows
    probabilities that add up to 1, which is a lot easier to read. The argmax is the same either way.

    This cell reads `sketch.value`, `processing.value` and `border_size.value`, so marimo re-runs it whenever any of
    them change.
    """)
    return


@app.cell
def _(
    Image,
    border_size,
    bounding_box,
    centre_of_mass,
    decode_image,
    device,
    io,
    mo,
    model,
    np,
    plt,
    processing,
    sketch,
    torch,
):
    def _plot(probabilities: np.ndarray):
        fig, ax = plt.subplots(figsize=(4, 3.5))
        digits = np.arange(10)
        ax.stem(digits, probabilities)
        ax.set_xticks(digits)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("digit")
        ax.set_ylabel("probability")
        ax.set_title("Prediction")
        fig.tight_layout()
        plt.close(fig)
        return mo.as_html(fig)

    def _preview(small: np.ndarray):
        # scale up with nearest so you can see the individual pixels
        buffer = io.BytesIO()
        Image.fromarray(small).resize((140, 140), Image.Resampling.NEAREST).save(
            buffer, format="PNG"
        )
        return mo.image(buffer.getvalue(), width=140, height=140)

    _full = decode_image(sketch.value["image"]) if sketch.value["image"] else None

    if _full is None or not _full.any():
        prediction = {
            "preview": _preview(np.zeros((28, 28), dtype=np.uint8)),
            "stat": mo.stat(value="-", label="Prediction", caption="draw a digit"),
            "plot": _plot(np.zeros(10)),
        }
    else:
        if processing.value == "Bounding Box":
            _small = bounding_box(_full, border_size.value)
        else:
            _small = centre_of_mass(_full)
        # add the batch dimension, Flatten in the model turns (1, 28, 28) into (1, 784)
        _tensor = torch.from_numpy(_small.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            _logits = model(_tensor)
        _probabilities = torch.softmax(_logits, dim=1)[0].cpu().numpy()
        _digit = int(_probabilities.argmax())
        prediction = {
            "preview": _preview(_small),
            "stat": mo.stat(
                value=str(_digit),
                label="Prediction",
                caption=f"{_probabilities[_digit]:.0%} confident",
            ),
            "plot": _plot(_probabilities),
        }
    return (prediction,)


@app.cell
def _():
    import base64
    import io

    import anywidget
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import traitlets
    from PIL import Image
    from scipy import ndimage

    return Image, anywidget, base64, io, mo, ndimage, np, plt, torch, traitlets


if __name__ == "__main__":
    app.run()
