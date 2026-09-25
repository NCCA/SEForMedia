# Matplotlib for Machine Learning

Four marimo notebooks covering the matplotlib used in the machine learning demos in this repository, following `NumPyForML/`, `PyTorchForML/` and `TorchVisionForML/`.

Matplotlib turns out to be the most widely used library in the unit — 73 demos call it, against 32 for PyTorch and 21 for NumPy, across 629 call sites. This follows on from `Lecture6/Matplotlib.ipynb`, which introduces plot types, subplots and images; these four are about the plots you make in the machine learning material and the decisions that make a figure worth putting in a report.

| Notebook | Covers |
| --- | --- |
| `MatplotlibForMLPart1Figures.py` | pyplot vs the object-oriented API, `figure`/`subplots`/`subplot`, labelling, layout, `show` |
| `MatplotlibForMLPart2Plots.py` | `plot`, `scatter`, `hist`, `bar`, legends, log scale, the colour cycle, why not to use a second y-axis |
| `MatplotlibForMLPart3Images.py` | `imshow`, colormaps, `colorbar`, confusion matrices, `contourf` decision boundaries, image grids |
| `MatplotlibForMLPart4Output.py` | `savefig`, sizing for print, `rcParams` and styles, colour vision checks, backends |

Run them with uv:

```
uv run marimo edit MatplotlibForML/MatplotlibForMLPart3Images.py
```

Two things in here are measured rather than asserted, which is the point of putting them in a notebook rather than a slide.

Part 3 converts each colormap to CIE L\* and reports how monotonic and how evenly spaced it is. `viridis` scores 0 reversals and 0.06 unevenness; `jet` scores 1 reversal and 0.65, ten times less uniform, which is where its false banding comes from. The same cell shows why `coolwarm`'s single reversal is correct rather than a fault.

Part 4 simulates deuteranopia and measures the perceptual distance between palette colours. Matplotlib's default C2 and C3 — the green and red you get for free as the third and fourth series — are 120 apart to normal vision and about 8 apart to a colourblind reader. A four-series plot drawn with the defaults has two lines one reader in twelve cannot separate. The notebook measures the alternatives and lands on Okabe-Ito, which stays above 17 for all seven colours under both common deficiencies.

Part 4 also ends with a checklist to run a figure through before it goes in a report.
