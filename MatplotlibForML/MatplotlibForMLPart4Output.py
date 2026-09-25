#!/usr/bin/env uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Matplotlib for Machine Learning, Part 4: saving figures

    In this final notebook we will save figures for reports and slides. We will look at file formats, figure size, shared settings and readability.

    A plot that works in a notebook may need some changes before we use it in a report. I check the saved file at its final size, paying attention to the labels, lines and colours.
    """)
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
    mo.md(r"""
    ## [savefig](https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.savefig)

    ```python
    fig.savefig(fname, *, dpi='figure', format=None, bbox_inches=None,
                pad_inches=0.1, facecolor='auto', transparent=False)
    ```

    These are the parameters we will use:

    | Parameter | What it does |
    | --- | --- |
    | `fname` | output path; the extension normally selects the format |
    | `dpi` | resolution for raster output |
    | `bbox_inches` | use `'tight'` to fit the saved bounds around the content |
    | `pad_inches` | padding in inches when using tight bounds |
    | `transparent` | makes the background transparent, useful for slides |

    I use `fig.savefig(...)` so it is clear which figure we are saving. `plt.savefig(...)` uses the current figure, which can be easy to lose track of when we have several plots.

    The example saves the same training curve in several formats. The files go into a temporary directory, printed by the first code cell. Check the saved files for clipped labels, even when using `bbox_inches="tight"`.
    """)
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Choosing a format

    | Format | Use | Notes |
    | --- | --- | --- |
    | PDF or SVG | line plots and bar charts | vector lines and text remain sharp when scaled |
    | PNG | images and dense scatter plots | stores a fixed grid of pixels |
    | JPEG | photographs | lossy compression can add artefacts around plot lines and text |

    A training curve usually works well as a vector file. A dense scatter plot can produce a large vector file because it stores the individual marks. We will compare the output sizes in the exercises.

    For raster figures intended for print, 300 dpi is a useful starting point.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure size

    `figsize` is measured in inches. Multiplying by `dpi` gives the pixel dimensions: a `(6, 4)` figure at 100 dpi is 600 by 400 pixels, whilst at 300 dpi it is 1800 by 1200. Cropping with `bbox_inches="tight"` can change the saved dimensions.

    Resizing a figure in a report also resizes its text. A 10-inch figure reduced to a 3-inch column will have labels at roughly a third of their original size.

    I set the figure width to match the space in the report before saving it. For a 3.2-inch column we could start with `figsize=(3.2, 2.4)`, then adjust the font sizes and spacing. Increasing the dpi adds pixels to raster output; it does not make the labels larger.
    """)
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
    mo.md(r"""
    ## [rcParams](https://matplotlib.org/stable/users/explain/customizing.html) and style sheets

    ```python
    plt.rcParams["figure.dpi"] = 150       # change one setting
    plt.rcParams.update({...})           # change several settings
    plt.style.use("ggplot")              # apply a style sheet
    with plt.style.context("ggplot"):    # apply it within this block
        ...
    ```

    `rcParams` holds Matplotlib's default settings. We can use it to set font sizes, line widths and figure sizes once, keeping the figures in a report consistent.

    The dictionary below is a starting point for these examples. Adjust it to suit the size and layout of your report.
    """)
    return


@app.cell
def _(plt):
    SENSIBLE_DEFAULTS = {
        "figure.figsize": (6, 3.5),
        "figure.dpi": 110,  # on screen
        "savefig.dpi": 300,  # on disk
        "savefig.bbox": "tight",  # fit the saved bounds to the content
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,  # keep the grid faint
        "axes.spines.top": False,  # hide the top border
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
    mo.md(r"""
    We use `plt.rc_context(...)` to apply the settings within a block. When we leave the block, the previous settings are restored.

    Style sheets collect settings in the same way. `plt.style.available` lists the installed styles. Try one with `plt.style.context(...)`, then check the labels, contrast and spacing in the resulting figure.
    """)
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
    mo.md(r"""
    ## Checking readability

    We need to be able to identify each series without relying on colour alone. Line styles, markers and labels also help when a figure is printed in greyscale.

    The next cells use a simple colour vision simulation and compare colours in CIE Lab space. We will use the results to explore which pairs become more similar under the simulation.

    This is an approximate demonstration. The distances are useful for comparison, but a single threshold cannot tell us whether a complete figure is readable. Line width, background and the size of the marks also matter.
    """)
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
        """Approximate an RGB colour using the selected colour vision simulation."""
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
        """Return the Euclidean distance between two colours in CIE Lab space."""
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
            _flag = "  <-- inspect this pair" if _d < 10 else ""
            print(f"C{_i}-C{_j}    {_n:8.1f} {_d:14.1f}{_flag}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare the distances for the original colours with those from the simulation. Smaller distances mean the colours are closer together in this calculation.

    We can try a different palette, reorder the colours, or distinguish the lines using markers and line styles. I would use line styles or markers alongside colour when the series need to be easy to identify.

    The next cell compares the default cycle, a reordered selection and seven colours from the Okabe-Ito palette. It measures colour distances only; we will add a dashed line in the plot afterwards. The threshold of 10 used above is an example flag for inspection, not a pass or fail test.
    """)
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
    print("Compare the smallest colour distances for each palette above.")
    print(
        "Check the plotted lines as well, using markers or line styles to identify them."
    )
    return (OKABE_ITO,)


@app.cell
def _(OKABE_ITO, epochs, np, plt, rng, train, valid):
    # apply the colour cycle within this block
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
    mo.md(r"""
    The dashed baseline gives us another way to identify that series. Try giving each line a different style or marker, then check the figure in greyscale.

    ## Backends

    ```python
    import matplotlib
    matplotlib.use("Agg")      # select the backend before importing pyplot
    ```

    The backend handles rendering. Interactive backends can open plot windows. `Agg` renders raster images without a display, which is useful for batch jobs and training scripts on a remote machine.

    When selecting a backend explicitly, put `matplotlib.use(...)` before importing `matplotlib.pyplot`. With `Agg`, use `savefig` to write the output; `plt.show()` will not open a plot window.

    Notebook tools can handle figure display themselves. The next cell prints the backend used by this session.
    """)
    return


@app.cell
def _(matplotlib):
    print("current backend:", matplotlib.get_backend())
    print()
    print(
        "some built-in backends:",
        sorted(matplotlib.backends.backend_registry.list_builtin())[:8],
        "...",
    )
    print()
    print("Agg           - headless, writes files. The safe choice in a script.")
    print("module://...  - what a notebook front end installs.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checking the saved figure

    Before adding a figure to a report, I check the following:

    - Label the axes and include units where they apply.
    - Inspect the figure at its final size, including the smallest text.
    - Check that labels and legends are inside the saved bounds.
    - Choose a file format and resolution suited to the content.
    - Identify each series with a legend or direct labels.
    - Use line styles or markers as well as colour where needed.
    - Choose a colourmap that matches the data, and label the colourbar for numerical values.
    - Start bar chart value axes at zero so bar lengths represent the values.
    - Check that axes and scales make comparisons clear; separate plots may be easier to read than a second y-axis.
    - Share `vmin` and `vmax` when comparing images on the same numerical scale.
    - Write a caption that explains what the figure shows and why it matters.

    ## Exercises

    1. Save the training curve as PDF and as PNG at 72, 150 and 300 dpi. Compare file sizes and view each at 400% zoom.
    2. Save a scatter plot of 50,000 points as PDF and PNG. Compare the file sizes and the time taken to open them.
    3. Build an `rcParams` dictionary for your report template. Measure the column width first.
    4. Run the colour simulation on a five-series plot. Inspect the closest pairs, then add line styles or markers and check the result in greyscale.
    5. Review a figure from an earlier piece of work using the checks above. Save a revised version and compare the two at their final size.

    This completes the Matplotlib notebooks. We use these plotting tools alongside `NumPyForML/`, `PyTorchForML/` and `TorchVisionForML/` in the machine learning examples.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
