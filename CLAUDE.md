# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Lecture and lab code for the [Software Engineering for Media](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/) unit on MSc AIM at the NCCA, Bournemouth University. It is teaching material, not a library — nothing here is imported by anything outside the repo, and most folders are a self-contained demo for one lecture or lab. Students read and run this code, so clarity beats cleverness, and prose (notebook markdown, docstrings, READMEs) is part of the deliverable rather than an afterthought.

## Running things

Everything goes through [uv](https://docs.astral.sh/uv/). There is a `.envrc` for direnv which sets `VIRTUAL_ENV` to `.venv` and runs `uv sync` if the venv is missing, so in a normal shell the environment is already active.

```bash
uv sync                                    # create / update .venv from uv.lock
uv run marimo edit Lecture2/Lecture2Marimo.py   # open a marimo notebook
uv run --with jupyter jupyter lab          # Jupyter (not installed in .venv; see zsh_functions.sh)
uv run Lecture3/rothko.py                  # a plain demo script
```

Most `.py` demos carry a shebang (`#!/usr/bin/env -S uv run --script` for scripts, `#!/usr/bin/env -S uv run marimo edit` for marimo notebooks) so students can run them directly. Note that `uv run --script` deliberately ignores the project environment, which is fine because those demos are standard library only (turtle, keyword, random). Anything needing numpy, torch or Qt must be run with plain `uv run` so it picks up `.venv`.

Torch comes from a CUDA 12.4 index on Linux and Windows and from PyPI (MPS/CPU) on macOS — see `[tool.uv.sources]` in `pyproject.toml`. Don't pin or reorder those without checking it still resolves on all three.

## Linting

`ruff` via pre-commit, run as a global tool rather than a project dependency:

```bash
pre-commit run --all-files    # ruff check --fix and ruff format
ruff check Lecture3/          # one folder
ruff format Lecture3/
```

Notebooks are excluded from ruff (`[tool.ruff] exclude = ["*.ipynb"]`). The `.flake8` file is a leftover — flake8 is not installed and not used.

## Tests

Tests use [pytest](https://docs.pytest.org/), which is in the `dev` dependency group. Write plain test functions with `assert`, fixtures and `parametrize`, not `unittest.TestCase` classes. There is no single top-level `tests/` folder. Each lesson that has tests keeps them next to its notebooks:

| Folder | Tests |
| --- | --- |
| `ImageToImage/` | `test_image_models.py`, `test_notebook.py` |
| `PointCloudsAndMeshNet/tests/` | model code and lesson notebooks, training tests marked `slow` |
| `TorchAudioForML/tests/`, `TorchCodecForML/tests/` | run each numbered notebook and check its outputs |
| `Lecture6/tests/` | the polars notebook against a small fake `MetObjects.csv` |
| `ExperimentTracking/tests/` | TensorBoard and MLflow demos, needs `--with mlflow --with tensorboard` |

```bash
MPLBACKEND=Agg uv run --with mlflow --with tensorboard pytest   # everything
uv run pytest TorchAudioForML/tests                             # one lesson
uv run pytest -m "not slow"                                     # skip the training tests
```

Most of these tests are notebook tests. They load a marimo notebook with `importlib.util.spec_from_file_location` and call `app.run()`, which returns every name the notebook defines. Pass `defs={...}` to swap a cell's outputs for test data, for example a `SimpleNamespace(value=...)` standing in for a submitted training form. marimo needs every name the replaced cell defines, not only the one you care about, or it raises `IncompleteRefsError`. Load notebooks by path rather than `import`, because numbered names like `01IntroductionMarimo.py` aren't valid module names. When a notebook renames a variable or file, its test breaks, so run the folder's tests after editing a notebook that has them.

`[tool.pytest.ini_options]` in `pyproject.toml` sets `--import-mode=importlib`, because several folders each have a `tests/test_notebooks.py`, and `pythonpath = ["."]`, so tests can import `ImageToImage.image_models`. The `Lecture4/*/colour_demo.py` files are driver scripts that print, not tests. Don't name a script `test_*.py` unless it really is a pytest file, or pytest will try to collect it.

## Layout

Three kinds of thing live side by side:

- **Slides** in `Slides/<Topic>/` — reveal.js, one `slides.md` per deck split on `\n---\n` (horizontal) and `\n--\n` (vertical), with a fixed `index.html` that loads it from Jon's web space. Edit `slides.md`; `index.html` is boilerplate. Deck titles carry the lecture number (`Slides/Numpy/slides.md` is Lecture 5), which does _not_ line up with the `LectureN/` folder numbering — the folders stop at 7 while the decks run to 10.
- **Notebooks** in `LectureN/`, `ASL/`, `MNIST/`, `Classification/` and friends. Most exist twice: a Jupyter `X.ipynb` and a marimo `XMarimo.py`. The marimo version is the one being actively maintained, so when changing notebook content, change both or say which you skipped. `IntroToMarimo/` explains marimo to students and is the reference for house style: markdown cells are `@app.cell(hide_code=True)` wrapping `mo.md(r"""...""")`.
- **Standalone apps and script sets** — `MNIST/Sketch/` (Qt sketchpad that feeds a trained model), `ASL/RealTimeCapture/` (webcam ASL demo, needs OpenCV installed separately), `Seminars/Arguments` and `Seminars/Files` (argparse/click and file IO exercises as starter and solution pairs), `NumPyForML/` and `PyTorchForML/` (numbered marimo notebook series with their own READMEs), `Neuron/nn_from_scratch.py`.

Qt code PySide6

## Utils and the sys.path convention

`Utils/` is the one shared package: `in_lab()` (hostname check for NCCA lab machines), `download`, `unzip_file`, `get_device`, `accuracy`, `get_batch_accuracy`, and in `Utils/training.py` the shared `Metrics`, `train_epoch`, `evaluate` and `copy_weights` used by the training notebooks. Notebooks reach it with `sys.path.append("../")` before `from Utils import ...`, because they run from their own folder rather than the repo root. Keep that pattern when adding a notebook that needs it — `Packages/PackagesMarimo.py` teaches `sys.path` using exactly this example, so it is deliberate rather than an accident to tidy up.

Code that has to work both in the labs and at home should branch on `in_lab()` (dataset paths, cache locations) rather than hardcoding one machine's layout.

## Training loops

Notebooks that train a model use a `mo.ui.dictionary(...).form(submit_button_label="Train")` for epochs and learning rate, `mo.stop` until it is submitted, build a fresh model and optimiser inside the training cell (so models are made by a `build_model()` function, not a module-level instance), show `mo.status.progress_bar`, return a `history` list for a separate plot cell, and put any `torch.save` behind a `mo.ui.run_button`. `FreeSpokenDigits/FSDDMarimoPt3.py` is the reference. `MNIST/ReadDigitsTrainingMarimo.py`, `ASL/ASLPart1Marimo.py` and `FSDDMarimoPt3.py` write the loop out in the notebook because they teach it; later notebooks import `train_epoch` / `evaluate` from `Utils`. `device` is a `torch.device`, so compare `device.type == "cuda"`, never `device == "cuda"` (always False).

## Data and models

Trained weights (`*.pth`) and `ScratchCode/` are gitignored, so notebooks must be able to regenerate anything they need. ONNX exports (`ASL/asl_model.onnx`) and small datasets are committed. `.gitattributes` sets `*.ipynb merge=theirs`, which means notebook merge conflicts are resolved by taking the incoming version wholesale — don't rely on git to merge notebook changes, coordinate instead.

## Branches

Teaching happens on the main branch,

## Writing style

Notebook markdown, READMEs, docstrings and slide bullets are written in Jon's voice: first person, British English, plain and unfussy, links out to the official docs rather than re-explaining them. Use the `jon-writing-style` skill for anything longer than a line. Python uses type hints on signatures and numpydoc-style docstrings, with trivial methods left undocumented.
