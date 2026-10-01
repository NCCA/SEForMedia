# TorchVision Part 2 visual examples

We wanted to see what each transform does to the image, following the visual changes to Part 1.

I added eleven comparison figures covering the source image, conversion pairs, float scaling, normalisation and its inverse, resize proportions, greyscale channels, the preprocessing stages, redundant conversion, and image/mask rotation. The resize and crop controls show the crop boundary. A shared rotation control lets us compare functional transforms with a jointly transformed Image and Mask.

Files changed:

- `TorchVisionForML/TorchVisionForMLPart2Transforms.py`
- This summary and the accompanying JSONL session export.

Commands run:

- `git status --short --branch`
- `git worktree add .worktrees/torchvision-transforms-visuals -b agent/torchvision-transforms-visuals`
- `MPLBACKEND=Agg python /private/tmp/check_torchvision_transforms.py`
- `python -m marimo check TorchVisionForML/TorchVisionForMLPart2Transforms.py`
- `uv run --active --with ruff ruff format TorchVisionForML/TorchVisionForMLPart2Transforms.py`
- `uv run --active --with ruff ruff check --ignore EXE001,PLR1711,B018 --fix TorchVisionForML/TorchVisionForMLPart2Transforms.py`
- `MPLBACKEND=Agg python -m marimo export html TorchVisionForML/TorchVisionForMLPart2Transforms.py -o /private/tmp/torchvision-transforms-visuals.html`
- `uv build --out-dir /private/tmp/torchvision-transforms-dist`
- `git diff --check`

The initial execution check failed because the original notebook produced no comparison figures. The updated notebook produced eleven. Checks also passed at the slider extremes and at zero rotation, including crop dimensions, matching functional and joint rotations, binary mask labels and inverse normalisation. I inspected the resize and rotation figures.

Marimo validation, lint and the executed HTML export passed. Ruff exclusions cover the existing shebang and Marimo cell returns and display expressions. The package build still fails because setuptools finds multiple top-level packages without package-discovery configuration.

The session export is a snapshot taken before the final commit and merge.
