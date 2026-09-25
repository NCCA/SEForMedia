#!/usr/bin/env uv run marimo edit

import marimo

__generated_with = "0.14.17"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Matplotlib for Machine Learning, Part 4: getting figures out

    The last of the four, and the one covering the least-travelled ground in this unit.

    Some of it is already here. `Lecture5/IntroductionToNumpy.py:856` and `Lecture6/IntroductionToPandas.ipynb:193` both set `plt.rcParams["figure.figsize"]` and `figure.autolayout`, and `Lecture1/test.py` has a `style.use("ggplot")`. So the idea of setting defaults once is established.

    What is missing entirely is **`savefig`** — there is not one call to it anywhere in the repository. Every figure in the unit is looked at in a notebook and then left there.

    That matters because every student on this unit eventually has to put a figure in a report, and a figure that looked fine in a notebook usually does not survive the trip. The text comes out too small, the labels are cut off, the lines vanish when it is printed, and two of the series turn out to be the same colour for one reader in twelve.

    Four topics: saving, sizing, setting defaults, and checking the result is readable.
    """
    )
    return


@app.cell
def _():
    import tempfile
    from pathlib import Path

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(42)

    epochs = np.arange(80)
    train = 2.3 * np.exp(-epochs / 14) + 0.05 + rng.normal(0, 0.015, 80)
    valid = 2.3 * np.exp(-epochs / 16) + 0.22 + rng.normal(0, 0.03, 80)

    print("matplotlib", matplotlib.__version__, "| writing to", out_dir)
    return epochs, matplotlib, np, out_dir, plt, rng, train, valid


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## [savefig](https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.savefig)

    ```python
    fig.savefig(fname, *, dpi='figure', format=None, bbox_inches=None,
                pad_inches=0.1, facecolor='auto', transparent=False)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `fname` | required | the format is taken from the extension |
    | `dpi` | the figure's own | dots per inch for raster formats |
    | `bbox_inches` | `None` | `'tight'` crops to the content, rescuing cut-off labels |
    | `pad_inches` | `0.1` | margin left when cropping tight |
    | `transparent` | `False` | transparent background, for slides |

    Two habits worth forming immediately. **`bbox_inches="tight"`** fixes the cut-off y-label, which is the most common fault in a submitted figure. And **save the `fig`, not `plt`** — `plt.savefig()` saves whatever the current figure happens to be, which is the state-machine trap from Part 1 with your coursework attached.
    """
    )
    return


@app.cell
def _(epochs, out_dir, plt, train, valid):
    fig_save, ax_save = plt.subplots(figsize=(6, 3.5))
    ax_save.plot(epochs, train, label="train", linewidth=2)
    ax_save.plot(epochs, valid, label="validation", linewidth=2)
    ax_save.set(
        xlabel="epoch", ylabel="cross-entropy loss (nats)", title="Training curve"
    )
    ax_save.legend()
    ax_save.grid(alpha=0.3)

    for _name, _kw in [
        ("default.png", {}),
        ("tight.png", {"bbox_inches": "tight"}),
        ("print.png", {"bbox_inches": "tight", "dpi": 300}),
        ("vector.pdf", {"bbox_inches": "tight"}),
        ("vector.svg", {"bbox_inches": "tight"}),
    ]:
        fig_save.savefig(out_dir / _name, **_kw)

    for _f in sorted(out_dir.iterdir()):
        print(f"  {_f.name:14} {_f.stat().st_size / 1024:8.1f} KB")
    plt.show()
    return (fig_save,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Which format

    | Format | Use it for | Why |
    | --- | --- | --- |
    | **PDF** or **SVG** | line plots, bar charts, anything in a written report | vector — scales to any size without going fuzzy, and the text stays selectable |
    | **PNG** | images, dense scatter plots, anything with thousands of marks | raster — a fixed grid of pixels, so complexity does not inflate the file |
    | **JPEG** | nothing here | lossy compression puts artefacts around sharp lines and text |

    The rule is about what is in the figure. A training curve is a few hundred line segments and a vector file stays tiny and perfectly sharp at any zoom. A scatter of 50,000 points as a PDF is 50,000 individual objects, and will produce a file that takes ten seconds to render in a PDF viewer — that one wants PNG at 300 dpi.

    `dpi=300` is the usual requirement for print. The screen default of 100 looks soft on paper.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Size, and why the text ends up too small

    `figsize` is in **inches** and `dpi` converts to pixels. A `figsize=(6, 4)` figure at `dpi=100` is 600x400 pixels; the same figure at `dpi=300` is 1800x1200 pixels of the same drawing.

    The thing that trips people up is what happens next. You save a 10-inch-wide figure, drop it into a report, and drag it down to fit a 3-inch column. Everything scales — including the text, which is now a third of the size you set it. That is why so many figures in dissertations have unreadable axis labels.

    **Make the figure the size it will be printed at, and leave it alone.** If the column is 3.2 inches wide, set `figsize=(3.2, 2.4)` and raise `dpi` for quality rather than scaling afterwards. The font sizes then mean what they say.
    """
    )
    return


@app.cell
def _(epochs, out_dir, plt, train, valid):
    def curve_on(ax):
        ax.plot(epochs, train, label="train", linewidth=2)
        ax.plot(epochs, valid, label="validation", linewidth=2)
        ax.set(xlabel="epoch", ylabel="loss (nats)")
        ax.legend(fontsize=8)

    for _w, _h, _dpi, _label in [
        (10, 6.5, 100, "big_then_shrunk"),
        (3.2, 2.1, 300, "sized_for_column"),
    ]:
        _f, _a = plt.subplots(figsize=(_w, _h), dpi=_dpi)
        curve_on(_a)
        _a.set_title(f"figsize=({_w}, {_h}) dpi={_dpi}", fontsize=9)
        _f.savefig(out_dir / f"{_label}.png", bbox_inches="tight")
        _px = (_w * _dpi, _h * _dpi)
        print(f"{_label:18} {_w}x{_h} in at {_dpi} dpi = {_px[0]:.0f}x{_px[1]:.0f} px")
        plt.close(_f)

    print()
    print("both end up about 3 inches wide on the page.")
    print("in the first, every label has been scaled down by a factor of three.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## [rcParams](https://matplotlib.org/stable/users/explain/customizing.html) and style sheets

    ```python
    plt.rcParams["figure.dpi"] = 150          # one setting
    plt.rcParams.update({...})                # several
    plt.style.use("ggplot")                   # a whole preset
    with plt.style.context("ggplot"):         # ...temporarily
        ...
    ```

    `rcParams` is the dictionary of every default matplotlib uses. Setting things there once at the top of a notebook beats repeating `fontsize=` on every call, and it means every figure in a report matches.

    A set worth putting at the top of any notebook you will take figures out of:
    """
    )
    return


@app.cell
def _(plt):
    SENSIBLE_DEFAULTS = {
        "figure.figsize": (6, 3.5),
        "figure.dpi": 110,  # on screen
        "savefig.dpi": 300,  # on disk
        "savefig.bbox": "tight",  # never cut off a label again
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,  # recessive
        "axes.spines.top": False,  # less frame, more data
        "axes.spines.right": False,
        "lines.linewidth": 2,
        "legend.frameon": False,
    }

    print("a few of the defaults these change:")
    for _k in ["figure.figsize", "savefig.dpi", "lines.linewidth", "axes.spines.top"]:
        print(f"  {_k:22} {str(plt.rcParams[_k]):12} -> {SENSIBLE_DEFAULTS[_k]}")
    print()
    print("total rcParams available:", len(plt.rcParams))
    return (SENSIBLE_DEFAULTS,)


@app.cell
def _(SENSIBLE_DEFAULTS, epochs, plt, train, valid):
    with plt.rc_context(SENSIBLE_DEFAULTS):
        _f, _a = plt.subplots(layout="constrained")
        _a.plot(epochs, train, label="train")
        _a.plot(epochs, valid, label="validation")
        _a.set(
            xlabel="epoch",
            ylabel="cross-entropy loss (nats)",
            title="With the defaults applied",
        )
        _a.legend()
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    `plt.rc_context(...)` as a context manager is the tidy way to do it — the settings apply inside the block and revert afterwards, so one figure with different rules does not leak into the rest of the notebook.

    Style sheets are the same idea packaged. `plt.style.available` lists what is installed; `'ggplot'`, `'bmh'` and the `'seaborn-v0_8-*'` set are the common ones. A style changes appearance only, never data, so it is always safe to try.
    """
    )
    return


@app.cell
def _(plt):
    styles = plt.style.available
    print(len(styles), "styles available, including:")
    for _s in [s for s in styles if not s.startswith("_")][:12]:
        print("  ", _s)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Checking a figure is actually readable

    Here is the part I would most like students to take away, because it is checkable rather than a matter of taste.

    Around **1 in 12 men and 1 in 200 women** have some form of colour vision deficiency, the commonest being deuteranopia — reduced sensitivity to green, which compresses the red-green axis. If two series in your figure are distinguished by colour alone, and those colours collapse together under that condition, the figure does not work for those readers. They will not tell you; they will just misread it.

    You do not have to guess at this. Simulating it is about fifteen lines of arithmetic, and comparing the results is a distance in a perceptual colour space. The cell below does both, then runs it on matplotlib's default cycle.
    """
    )
    return


@app.cell
def _(np, plt):
    def _to_linear(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def _to_srgb(c):
        c = np.clip(c, 0, 1)
        return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)

    RGB_TO_LMS = np.array(
        [
            [17.8824, 43.5161, 4.11935],
            [3.45565, 27.1554, 3.86714],
            [0.0299566, 0.184309, 1.46709],
        ]
    )
    LMS_TO_RGB = np.linalg.inv(RGB_TO_LMS)
    CVD = {
        "deuteranopia": np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]]),
        "protanopia": np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]]),
    }

    def simulate_cvd(rgb, kind):
        """What this colour looks like to someone with that colour vision deficiency."""
        lms = _to_linear(rgb) @ RGB_TO_LMS.T
        return _to_srgb((lms @ CVD[kind].T) @ LMS_TO_RGB.T)

    def to_lab(rgb):
        M = np.array(
            [
                [0.4124, 0.3576, 0.1805],
                [0.2126, 0.7152, 0.0722],
                [0.0193, 0.1192, 0.9505],
            ]
        )
        xyz = _to_linear(rgb) @ M.T / np.array([0.95047, 1.0, 1.08883])
        f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
        return np.stack(
            [
                116 * f[..., 1] - 16,
                500 * (f[..., 0] - f[..., 1]),
                200 * (f[..., 1] - f[..., 2]),
            ],
            -1,
        )

    def delta_e(a, b):
        """Perceptual distance. Below about 10, two colours are hard to tell apart."""
        return float(np.linalg.norm(to_lab(a) - to_lab(b)))

    cycle_rgb = np.array(
        [
            plt.matplotlib.colors.to_rgb(c)
            for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]
        ]
    )
    print("simulation ready. The default cycle has", len(cycle_rgb), "colours.")
    return cycle_rgb, delta_e, simulate_cvd


@app.cell
def _(cycle_rgb, delta_e, simulate_cvd):
    print("A four-series plot using the defaults (C0 to C3):")
    print(f"{'pair':8} {'normal':>8} {'deuteranopia':>14}")
    for _i in range(4):
        for _j in range(_i + 1, 4):
            _n = delta_e(cycle_rgb[_i], cycle_rgb[_j])
            _d = delta_e(
                simulate_cvd(cycle_rgb[_i], "deuteranopia"),
                simulate_cvd(cycle_rgb[_j], "deuteranopia"),
            )
            _flag = "  <-- collapses" if _d < 10 else ""
            print(f"C{_i}-C{_j}    {_n:8.1f} {_d:14.1f}{_flag}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    There it is. **C2 and C3 — matplotlib's green and red, the third and fourth colours you get for free — are 120 apart normally and about 8 apart under deuteranopia.** A four-series plot drawn with the defaults has two lines that a colourblind reader cannot reliably separate, and nothing in the notebook hints at it.

    Three ways to deal with it, in order of preference:

    1. **Pick a palette designed for it.** The Okabe-Ito set is the standard one and it is eight colours that stay separated under both common deficiencies.
    2. **Do not rely on colour alone.** Vary the line style or marker too, so identity is carried twice. This also survives a black-and-white printer, which is the other reason to do it.
    3. **Reorder the default cycle** so the colours you actually use are the ones that separate — `C0, C1, C5, C6` scores far better than `C0` to `C3`.

    The cell below measures all three against the default.
    """
    )
    return


@app.cell
def _(cycle_rgb, delta_e, np, plt, simulate_cvd):
    import itertools

    OKABE_ITO = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ]
    okabe_rgb = np.array([plt.matplotlib.colors.to_rgb(c) for c in OKABE_ITO])

    def worst_pair(colours, kind="deuteranopia"):
        return min(
            delta_e(simulate_cvd(colours[i], kind), simulate_cvd(colours[j], kind))
            for i, j in itertools.combinations(range(len(colours)), 2)
        )

    print(f"{'palette':34} {'worst pair under deuteranopia':>30}")
    print(f"{'default cycle, first 4':34} {worst_pair(cycle_rgb[:4]):>30.1f}")
    print(f"{'default cycle, all 10':34} {worst_pair(cycle_rgb):>30.1f}")
    print(
        f"{'default reordered C0,C1,C5,C6':34} {worst_pair(cycle_rgb[[0, 1, 5, 6]]):>30.1f}"
    )
    print(f"{'Okabe-Ito, first 4':34} {worst_pair(okabe_rgb[:4]):>30.1f}")
    print(f"{'Okabe-Ito, all 7':34} {worst_pair(okabe_rgb):>30.1f}")
    print(
        f"{'Okabe-Ito, all 7 (protanopia)':34} {worst_pair(okabe_rgb, 'protanopia'):>30.1f}"
    )
    print()
    print("Below about 10 is a collision. Okabe-Ito stays above 17 for all seven,")
    print("under both deficiencies - which is why it is the one to reach for.")
    return (OKABE_ITO,)


@app.cell
def _(OKABE_ITO, epochs, np, plt, rng, train, valid):
    # making it the default for a notebook is one line
    with plt.rc_context(
        {"axes.prop_cycle": plt.cycler(color=OKABE_ITO), "lines.linewidth": 2}
    ):
        _f, _a = plt.subplots(figsize=(6, 3.2), layout="constrained")
        _a.plot(epochs, train, label="train")
        _a.plot(epochs, valid, label="validation")
        _a.plot(epochs, valid + 0.3 + rng.normal(0, 0.02, 80), label="test")
        _a.plot(
            epochs, 2.3 * np.exp(-epochs / 10) + 0.4, label="baseline", linestyle="--"
        )
        _a.set(
            xlabel="epoch",
            ylabel="cross-entropy loss (nats)",
            title="Okabe-Ito cycle, one dashed",
        )
        _a.legend()
        _a.grid(alpha=0.3)
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Note the dashed baseline in that figure. That is point 2 — the line style distinguishes it whatever happens to the colour, and it is the single cheapest robustness measure available.

    ## Backends

    ```python
    import matplotlib
    matplotlib.use("Agg")      # before importing pyplot
    ```

    A backend is what matplotlib draws onto. Interactive ones open a window; `Agg` renders to a memory buffer and is what you want in a script, on a headless machine, or anywhere there is no display — a lab batch job, or a training run on a remote GPU.

    There are three `matplotlib.use` calls in this repository, which is the right instinct. The rule is that it must come **before** `import matplotlib.pyplot`, because the backend is chosen when pyplot is first imported.

    Note the output of the cell below: these notebooks are themselves running under `Agg` when exported to a script, which is why `plt.show()` produces nothing there and the figures only appear in marimo.
    """
    )
    return


@app.cell
def _(matplotlib):
    print("current backend:", matplotlib.get_backend())
    print()
    print(
        "some that exist here:", sorted(set(matplotlib.rcsetup.all_backends))[:8], "..."
    )
    print()
    print("Agg           - headless, writes files. The safe choice in a script.")
    print("module://...  - what a notebook front end installs.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## A checklist

    Before a figure goes in a report:

    - [ ] every axis has a label, **with units**
    - [ ] the figure is the size it will be printed at, not scaled afterwards
    - [ ] font sizes are readable at that size — 8pt minimum, and check it on paper
    - [ ] saved with `bbox_inches="tight"`, so nothing is cut off
    - [ ] vector (PDF/SVG) unless it has thousands of marks in it
    - [ ] two or more series means a legend
    - [ ] colour is not the only thing distinguishing series — line style or marker too
    - [ ] a sequential colormap for magnitude, a qualitative one for categories, never `jet`
    - [ ] a colorbar wherever colour encodes a number, with a label
    - [ ] bar charts start at zero
    - [ ] no second y-axis
    - [ ] compared images share `vmin` and `vmax`
    - [ ] the caption says what the reader should conclude, not just what is plotted

    ## Exercises

    1. Save the same training curve as PDF and as PNG at 72, 150 and 300 dpi. Compare the file sizes, then zoom to 400% on each.
    2. Make a scatter of 50,000 points and save it both ways. How long does the PDF take to open?
    3. Build an rcParams dictionary for your dissertation template — measure the column width first.
    4. Run the CVD check on a five-series figure of your own. If any pair is below 10, fix it and re-run.
    5. Take a figure you have already submitted for something and put it through the checklist. How many items does it fail?

    That is the matplotlib set, and the end of the four groups. Between `NumPyForML/`, `PyTorchForML/`, `TorchVisionForML/` and these, you have the libraries the machine learning demos in this repository are built from.
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
