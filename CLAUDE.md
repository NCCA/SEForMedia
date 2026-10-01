# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Lecture and lab code for the [Software Engineering for Media](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/) unit on MSc AIM at the NCCA, Bournemouth University. It is teaching material, not a library — nothing here is imported by anything outside the repo, and most folders are a self-contained demo for one lecture or lab. Students read and run this code, so clarity beats cleverness, and prose (notebook markdown, docstrings, READMEs) is part of the deliverable rather than an afterthought.

`TODO.md` and `UnitImprovementPlan.md` hold the current plans for the material — read them before proposing new content, since they already record what is missing and in what order it should be filled.

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

There is no test suite. `Lecture4/Colour/test_colour.py` and `Lecture4/ClassMethod/test_colour.py` are driver scripts that exercise the `Colour` class by printing, not pytest tests, and pytest is not installed. Don't claim tests pass, and don't run `pytest` expecting anything to be collected. Adding a real test suite is the top item in `UnitImprovementPlan.md`, so if you are asked to write tests, add pytest to the dev dependency group first and rename those two files so they stop shadowing real test discovery.

## Layout

Three kinds of thing live side by side:

- **Slides** in `Slides/<Topic>/` — reveal.js, one `slides.md` per deck split on `\n---\n` (horizontal) and `\n--\n` (vertical), with a fixed `index.html` that loads it from Jon's web space. Edit `slides.md`; `index.html` is boilerplate. Deck titles carry the lecture number (`Slides/Numpy/slides.md` is Lecture 5), which does _not_ line up with the `LectureN/` folder numbering — the folders stop at 7 while the decks run to 10.
- **Notebooks** in `LectureN/`, `ASL/`, `MNIST/`, `Classification/` and friends. Most exist twice: a Jupyter `X.ipynb` and a marimo `XMarimo.py`. The marimo version is the one being actively maintained (see `TODO.md`), so when changing notebook content, change both or say which you skipped. `IntroToMarimo/` explains marimo to students and is the reference for house style: markdown cells are `@app.cell(hide_code=True)` wrapping `mo.md(r"""...""")`.
- **Standalone apps and script sets** — `MNIST/Sketch/` (Qt sketchpad that feeds a trained model), `ASL/RealTimeCapture/` (webcam ASL demo, needs OpenCV installed separately), `Seminars/Arguments` and `Seminars/Files` (argparse/click and file IO exercises as starter and solution pairs), `NumPyForML/` and `PyTorchForML/` (numbered marimo notebook series with their own READMEs), `Neuron/nn_from_scratch.py`.

Qt code uses `qtpy` for PyQt5/PySide compatibility. `MainWindow.py` files are generated from `MainWindow.ui` by `pyuic5` — edit the `.ui` in Designer and regenerate, never hand-edit the generated file.

## Utils and the sys.path convention

`Utils/` is the one shared package: `in_lab()` (hostname check for NCCA lab machines), `download`, `unzip_file`, `get_device`, `accuracy`, `get_batch_accuracy`. Notebooks reach it with `sys.path.append("../")` before `from Utils import ...`, because they run from their own folder rather than the repo root. Keep that pattern when adding a notebook that needs it — `Packages/PackagesMarimo.py` teaches `sys.path` using exactly this example, so it is deliberate rather than an accident to tidy up.

Code that has to work both in the labs and at home should branch on `in_lab()` (dataset paths, cache locations) rather than hardcoding one machine's layout.

## Data and models

Trained weights (`*.pth`) and `ScratchCode/` are gitignored, so notebooks must be able to regenerate anything they need. ONNX exports (`ASL/asl_model.onnx`) and small datasets are committed. `.gitattributes` sets `*.ipynb merge=theirs`, which means notebook merge conflicts are resolved by taking the incoming version wholesale — don't rely on git to merge notebook changes, coordinate instead.

## Branches

Teaching happens on the main branch,

## Writing style

Notebook markdown, READMEs, docstrings and slide bullets are written in Jon's voice: first person, British English, plain and unfussy, links out to the official docs rather than re-explaining them. Use the `jon-writing-style` skill for anything longer than a line. Python uses type hints on signatures and numpydoc-style docstrings, with trivial methods left undocumented.
