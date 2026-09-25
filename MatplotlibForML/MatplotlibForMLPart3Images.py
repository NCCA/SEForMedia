#!/usr/bin/env uv run marimo edit

import marimo

__generated_with = "0.14.17"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Matplotlib for Machine Learning, Part 3: images and 2D fields

    `plt.imshow` is called 66 times across 33 demos here, which makes it the busiest drawing function in the unit after `plot`. Anything laid out on a grid goes through it: a digit, a feature map, a confusion matrix, a decision boundary.

    Most of this notebook is about **colour**, because that is where the decisions are. When colour encodes a number rather than decorating a line, the colormap is doing the work of an axis — and a badly chosen one invents structure that is not in the data. It is the one topic in these notebooks where the wrong choice produces a figure that is confidently, legibly wrong.
    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(42)
    return np, plt, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## [plt.imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)

    ```python
    plt.imshow(X, cmap=None, norm=None, aspect=None, interpolation=None,
               vmin=None, vmax=None, origin='upper', extent=None)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `X` | required | see the layouts below |
    | `cmap` | `'viridis'` | colormap — used **only** for 2D scalar input |
    | `vmin`, `vmax` | `None` | the values mapped to the ends of the colormap |
    | `interpolation` | `'antialiased'` | how pixels are resampled when drawn |
    | `origin` | `'upper'` | whether row 0 is at the top |
    | `aspect` | `'equal'` | `'auto'` lets pixels be non-square |

    It accepts three layouts, and which one you have determines whether `cmap` does anything at all:

    | Input shape | Interpreted as | `cmap` used? |
    | --- | --- | --- |
    | `(M, N)` | scalar values, mapped through the colormap | yes |
    | `(M, N, 3)` | RGB | no |
    | `(M, N, 4)` | RGBA | no |

    That last column catches people. Passing an RGB image and a `cmap` is not an error — the colormap is simply ignored, silently.

    And note the channel position: matplotlib wants `(H, W, C)`, while torchvision gives you `(C, H, W)`. That is the `permute(1, 2, 0)` from `TorchVisionForML` Part 1, and it is the most common reason an image comes out as an unreadable smear.
    """
    )
    return


@app.cell
def _(np, plt, rng):
    digit = np.zeros((16, 16))
    digit[3:13, 6:9] = 1.0  # a crude vertical stroke
    digit[3:6, 4:9] = 1.0
    digit += rng.normal(0, 0.06, digit.shape)

    rgb = rng.random((16, 16, 3))

    fig_layouts, axes_layouts = plt.subplots(
        1, 3, figsize=(7.5, 2.6), layout="constrained"
    )

    axes_layouts[0].imshow(digit, cmap="gray")
    axes_layouts[0].set_title("(M, N) + cmap", fontsize=9)

    axes_layouts[1].imshow(rgb)
    axes_layouts[1].set_title("(M, N, 3), cmap ignored", fontsize=9)

    axes_layouts[2].imshow(rgb, cmap="gray")  # cmap silently does nothing
    axes_layouts[2].set_title(
        "(M, N, 3) + cmap='gray'\nno error, no effect", fontsize=9
    )

    for _ax in axes_layouts:
        _ax.axis("off")
    plt.show()
    return (digit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### axis("off")

    `plt.axis("off")` appears 78 times in this repository, essentially always alongside `imshow`. Tick marks measured in pixel indices tell a reader nothing about a photograph, so turning them off is right for an image.

    It is *not* right for a confusion matrix or a feature map where the row and column indices mean something. Turn the axes off when the axes carry no information, not as a reflex.

    ### interpolation

    The default resampling smooths when an image is drawn larger than its pixel grid. For a photograph that is what you want. For a 28x28 digit or a small feature map, smoothing invents intermediate values that are not in the data — set `interpolation="nearest"` so you see the actual pixels.
    """
    )
    return


@app.cell
def _(digit, plt):
    fig_interp, axes_interp = plt.subplots(1, 2, figsize=(6, 3), layout="constrained")

    axes_interp[0].imshow(digit, cmap="gray")
    axes_interp[0].set_title("default - smoothed", fontsize=9)

    axes_interp[1].imshow(digit, cmap="gray", interpolation="nearest")
    axes_interp[1].set_title("interpolation='nearest'\nthe actual pixels", fontsize=9)

    for _ax in axes_interp:
        _ax.axis("off")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### vmin and vmax

    By default `imshow` scales the colormap to the range of *that array*. Draw four feature maps side by side and each gets its own scale, so identical colours mean different numbers in different panels and the comparison you are trying to make is meaningless.

    Whenever you put two images next to each other to compare them, fix `vmin` and `vmax` across both.
    """
    )
    return


@app.cell
def _(np, plt, rng):
    weak = rng.random((12, 12)) * 0.2
    strong = rng.random((12, 12)) * 1.0

    fig_v, axes_v = plt.subplots(1, 4, figsize=(9, 2.4), layout="constrained")

    axes_v[0].imshow(weak, cmap="viridis")
    axes_v[0].set_title("weak, autoscaled", fontsize=8)
    axes_v[1].imshow(strong, cmap="viridis")
    axes_v[1].set_title("strong, autoscaled", fontsize=8)

    axes_v[2].imshow(weak, cmap="viridis", vmin=0, vmax=1)
    axes_v[2].set_title("weak, vmin=0 vmax=1", fontsize=8)
    axes_v[3].imshow(strong, cmap="viridis", vmin=0, vmax=1)
    axes_v[3].set_title("strong, vmin=0 vmax=1", fontsize=8)

    for _ax in axes_v:
        _ax.axis("off")
    plt.show()

    print(f"weak   range {weak.min():.2f} to {weak.max():.2f}")
    print(f"strong range {strong.min():.2f} to {strong.max():.2f}")
    print("the first pair look equally bright. Only the second pair can be compared.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Colormaps

    A [colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html) maps a number to a colour. There are three kinds, they are not interchangeable, and picking by appearance rather than by kind is where figures go wrong.

    | Kind | For | Examples |
    | --- | --- | --- |
    | **sequential** | magnitude — low to high | `viridis`, `plasma`, `gray`, `Blues` |
    | **diverging** | deviation either side of a meaningful middle | `coolwarm`, `RdBu`, `bwr` |
    | **qualitative** | identity — categories with no order | `tab10`, `Set2` |

    The rule that picks between them is a question about the data: **does the number have an order, and does it have a meaningful midpoint?**

    - loss values, pixel intensity, confidence: ordered, no special middle -> sequential
    - a difference, an error that can be positive or negative, a correlation: ordered, zero is meaningful -> diverging, with the neutral colour pinned at zero
    - class labels: no order at all -> qualitative
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Why `jet` is the wrong answer

    `jet` — the rainbow — was matplotlib's default until version 2.0 and is still what a lot of older code and a lot of papers use. It is a bad colormap and the reason is measurable rather than a matter of taste.

    A colormap encodes a number, and the eye reads magnitude mostly through **lightness**. Two properties follow from that, and both can be measured rather than argued about:

    - lightness should be **monotonic** — rising steadily from one end to the other, so one lightness means one value
    - the steps should be **even** — equal steps in the data should look like equal steps in the colour, or the map invents edges where the data is smooth

    The cell below converts each colormap to CIE L\*, the perceptual lightness axis, and measures both. `viridis` was designed to score well on exactly these and became matplotlib's default in version 2.0 for that reason.
    """
    )
    return


@app.cell
def _(np, plt):
    def perceptual_lightness(cmap_name, n=256):
        """CIE L* for each step of a colormap."""
        srgb = plt.get_cmap(cmap_name)(np.linspace(0, 1, n))[:, :3]
        lin = np.where(
            srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4
        )  # sRGB -> linear
        Y = lin @ np.array([0.2126, 0.7152, 0.0722])  # -> relative luminance
        f = np.where(Y > 0.008856, np.cbrt(Y), 7.787 * Y + 16 / 116)
        return 116 * f - 16

    def reversals(L, frac=0.001):
        """How many times does lightness change direction?"""
        d = np.diff(L)
        d = d[np.abs(d) > frac * (L.max() - L.min())]  # ignore flat noise
        return int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))

    def evenness(L):
        """Spread of step sizes, relative to the mean. Lower is more uniform."""
        d = np.abs(np.diff(L))
        return d.std() / d.mean()

    print(
        f"{'colormap':10} {'kind':12} {'L* range':>9} {'reversals':>10} {'unevenness':>11}"
    )
    for name, kind in [
        ("viridis", "sequential"),
        ("plasma", "sequential"),
        ("gray", "sequential"),
        ("coolwarm", "diverging"),
        ("jet", "'sequential'"),
        ("rainbow", "'sequential'"),
        ("turbo", "'sequential'"),
    ]:
        L = perceptual_lightness(name)
        print(
            f"{name:10} {kind:12} {L.max() - L.min():9.1f} {reversals(L):>10} {evenness(L):>11.2f}"
        )
    return (perceptual_lightness,)


@app.cell
def _(np, perceptual_lightness, plt):
    fig_L, ax_L = plt.subplots(figsize=(6, 3), layout="constrained")

    _steps = np.linspace(0, 1, 256)
    for _name, _style in [
        ("viridis", "-"),
        ("plasma", "-"),
        ("jet", "--"),
        ("rainbow", ":"),
    ]:
        ax_L.plot(_steps, perceptual_lightness(_name), _style, linewidth=2, label=_name)

    ax_L.set(
        xlabel="position along the colormap",
        ylabel="perceptual lightness L*",
        title="A sequential colormap should be a straight line here",
    )
    ax_L.legend()
    ax_L.grid(alpha=0.3)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Read those two columns together, because on its own the reversal count is misleading.

    **`coolwarm` has one reversal and that is correct.** A diverging map is meant to be dark at both ends and light in the middle — lightness encodes distance from the midpoint and hue encodes which side you are on. Used for the job it is for, that reversal is the design.

    **`jet` also has one reversal, and there it is a defect**, because `jet` gets used as a sequential map. Lightness climbs to the yellow band and falls away to dark red, so a dark cell might be a low value or a high one and the reader cannot tell which.

    The `unevenness` column is the sharper indictment. `viridis` scores about 0.06 — its steps are nearly all the same perceptual size, so equal steps in your data look equal. `jet` scores around 0.65, ten times less uniform: some stretches barely change and others lurch. Those lurches are the bands you see in the right-hand panel below, and they look exactly like edges in the data.

    There is a second reason, which matters for the same figure printed or read by a colourblind reader: a rainbow relies on hue to carry magnitude, and hue is exactly what red-green colour blindness compresses. `viridis` carries magnitude in lightness, which survives both greyscale printing and every common form of colour vision deficiency.

    **The short version to give students: use `viridis` unless you have a reason, `gray` for images, `coolwarm` for signed differences, and never `jet`.**
    """
    )
    return


@app.cell
def _(np, plt, rng):
    # the same smooth field under both, so the invented banding is visible
    _yy, _xx = np.mgrid[0:120, 0:160]
    field = np.sin(_xx / 22) + np.cos(_yy / 18) + 0.3 * rng.normal(0, 0.05, (120, 160))

    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(8, 2.8), layout="constrained")
    for _ax, _cmap in zip(axes_cm, ["viridis", "jet"]):
        _im = _ax.imshow(field, cmap=_cmap)
        _ax.set_title(_cmap, fontsize=9)
        _ax.axis("off")
        fig_cm.colorbar(_im, ax=_ax, fraction=0.046)
    plt.show()
    print("the data is smooth. Any band or edge you see in the right-hand panel")
    print("is the colormap, not the data.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Colour for class identity

    This one is worth flagging because it appears in this repository. `Classification/MultiClassificationMarimo.py:79` colours a multi-class scatter with a sequential map:

    ```python
    plt.scatter(x=x[:, 0], y=x[:, 1], c=y, cmap=plt.cm.plasma)
    ```

    For the **binary** demos this is fine — with two classes any two distinguishable colours work, and `coolwarm` even reads sensibly as two poles.

    With three or more it starts saying something untrue. A sequential map is ordered, so class 0 and class 1 come out similar and class 0 and class 3 come out very different — implying that class 3 is further from class 0 than class 1 is. Class labels are nominal; "further" is not a thing they do. The fix is a qualitative colormap, and a legend, since with categories a colourbar makes no sense either.
    """
    )
    return


@app.cell
def _(np, plt, rng):
    centres = np.array([[-2.0, -1.5], [2.0, -1.0], [0.0, 2.0], [3.0, 2.5]])
    labels = np.repeat(np.arange(4), 80)
    points = np.vstack([rng.normal(c, 0.55, (80, 2)) for c in centres])

    fig_cls, axes_cls = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")

    axes_cls[0].scatter(points[:, 0], points[:, 1], c=labels, cmap=plt.cm.plasma, s=18)
    axes_cls[0].set_title(
        "sequential map on class labels\nimplies 0 < 1 < 2 < 3", fontsize=9
    )

    for _k in range(4):
        _m = labels == _k
        axes_cls[1].scatter(points[_m, 0], points[_m, 1], s=18, label=f"class {_k}")
    axes_cls[1].set_title("one colour per class, with a legend", fontsize=9)
    axes_cls[1].legend(fontsize=8)

    for _ax in axes_cls:
        _ax.set(xlabel="feature 1", ylabel="feature 2")
    plt.show()
    return centres, labels, points


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## [colorbar](https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.colorbar)

    ```python
    fig.colorbar(mappable, ax=..., label=None, fraction=0.15, shrink=1.0)
    ```

    **If colour encodes a number, the figure needs a colorbar.** It is the legend for the colour axis, and without it the reader can see the pattern but cannot read a single value off it.

    The argument is the *mappable* — the object `imshow` or `contourf` returned — which is why you have to keep that return value rather than discarding it. `ax=` tells matplotlib which axes to steal space from.

    A colorbar needs a label as much as any axis does. `fig.colorbar(im, ax=ax, label="activation")` costs nothing.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The confusion matrix

    The standard way to see *how* a classifier is wrong rather than just how often. `imshow` plus text annotations, and it is worth writing once properly.

    Two decisions in the version below. The colour is sequential, because a count has an order and no meaningful midpoint. And the counts are printed in each cell, because a reader wants the numbers — with a grid this small, colour is orientation and the text is the data.
    """
    )
    return


@app.cell
def _(np, plt, rng):
    n_cls = 5
    class_names = ["cat", "dog", "bird", "fish", "frog"]
    confusion = np.zeros((n_cls, n_cls), dtype=int)
    for _t in range(n_cls):
        confusion[_t, _t] = rng.integers(60, 95)
        for _p in range(n_cls):
            if _p != _t:
                confusion[_t, _p] = rng.integers(0, 12)
    confusion[0, 1] = 24  # cats mistaken for dogs, a real-looking confusion

    fig_cm2, ax_cm2 = plt.subplots(figsize=(5, 4.2), layout="constrained")
    im_cm = ax_cm2.imshow(confusion, cmap="Blues")

    ax_cm2.set(
        xticks=np.arange(n_cls),
        yticks=np.arange(n_cls),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="predicted",
        ylabel="true",
        title="Confusion matrix",
    )

    _threshold = confusion.max() / 2
    for _i in range(n_cls):
        for _j in range(n_cls):
            ax_cm2.text(
                _j,
                _i,
                confusion[_i, _j],
                ha="center",
                va="center",
                color="white" if confusion[_i, _j] > _threshold else "black",
                fontsize=9,
            )

    fig_cm2.colorbar(im_cm, ax=ax_cm2, label="samples", fraction=0.046)
    plt.show()

    print("per-class recall:", (confusion.diagonal() / confusion.sum(axis=1)).round(2))
    print(
        "worst confusion: true",
        class_names[0],
        "predicted",
        class_names[1],
        "->",
        confusion[0, 1],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The `color="white" if value > threshold else "black"` is doing real work. Dark text on a dark cell is unreadable, and a confusion matrix has both extremes by construction. Switching at the midpoint of the scale is the cheap fix.

    ## [contourf](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html) and decision boundaries

    ```python
    plt.contourf(X, Y, Z, levels=None, cmap=None, alpha=None)
    ```

    9 calls across 9 demos — every classification notebook here ends with one.

    This is the payoff for the `meshgrid` and `np.c_` material in `NumPyForML` Part 2. The recipe:

    1. `meshgrid` over the feature ranges to build a lattice
    2. `ravel` and `np.c_` to turn it into a list of points
    3. run the model over all of them at once
    4. `reshape` the predictions back to the lattice
    5. `contourf` the result, with the training points scattered on top

    `Classification/BinaryClassificationMarimo.py:314` does exactly this. The `alpha=0.2` in that code matters — the regions are context and the data points are the subject, so the fill has to sit back.
    """
    )
    return


@app.cell
def _(centres, labels, np, plt, points):
    # a stand-in for a trained model: nearest centre wins
    def predict(grid_points):
        d = np.linalg.norm(grid_points[:, None, :] - centres[None, :, :], axis=2)
        return d.argmin(axis=1)

    x_min, x_max = points[:, 0].min() - 0.5, points[:, 0].max() + 0.5
    y_min, y_max = points[:, 1].min() - 0.5, points[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    zz = predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig_db, ax_db = plt.subplots(figsize=(5.5, 4), layout="constrained")
    ax_db.contourf(xx, yy, zz, levels=np.arange(-0.5, 4.5), cmap="tab10", alpha=0.18)
    for _k in range(4):
        _m = labels == _k
        ax_db.scatter(
            points[_m, 0],
            points[_m, 1],
            s=18,
            label=f"class {_k}",
            edgecolors="white",
            linewidths=0.4,
        )

    ax_db.set(
        xlabel="feature 1",
        ylabel="feature 2",
        title="Decision regions",
        xlim=(x_min, x_max),
        ylim=(y_min, y_max),
    )
    ax_db.legend(fontsize=8)
    plt.show()

    print("grid points evaluated:", zz.size)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Two details there. `levels=np.arange(-0.5, 4.5)` puts one band per class with the boundaries between integers — without it `contourf` picks its own levels and you get bands that do not correspond to classes. And `edgecolors="white"` on the scatter separates points from the fill behind them, which is the 2px surface ring idea: a thin light outline keeps a mark readable over anything.

    ## Grids of images

    The other common use of `imshow`: a batch of samples at once, to check a data pipeline or to look at what a model got wrong.
    """
    )
    return


@app.cell
def _(np, plt, rng):
    samples = rng.random((12, 20, 20))
    truth = rng.integers(0, 10, 12)
    predicted = truth.copy()
    predicted[[2, 7, 9]] = (truth[[2, 7, 9]] + 1) % 10  # three wrong

    fig_grid, axes_grid = plt.subplots(3, 4, figsize=(7, 5.4), layout="constrained")

    for _i, _ax in enumerate(axes_grid.ravel()):
        _ax.imshow(samples[_i], cmap="gray", interpolation="nearest")
        _ok = truth[_i] == predicted[_i]
        _ax.set_title(
            f"true {truth[_i]} / pred {predicted[_i]}",
            fontsize=8,
            color="black" if _ok else "C3",
        )
        _ax.axis("off")

    fig_grid.suptitle(
        "Predictions, with errors marked in the title colour", fontsize=10
    )
    plt.show()

    print("errors:", int(np.sum(truth != predicted)), "of", len(truth))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Marking the errors by title colour alone is worth a caveat — colour alone is exactly what a colourblind reader cannot use. In a figure that mattered I would add a marker to the text as well, so identity is carried by two channels. Part 4 comes back to this.

    ## Exercises

    1. Load an MNIST digit and display it four ways: `gray`, `gray_r`, `viridis`, `jet`. Which lets you judge stroke thickness most reliably, and why?
    2. Run the lightness calculation on `hot`, `magma` and `Blues`. One of them ends darker than it starts — does that make it a bad sequential map, or just a reversed one?
    3. Display eight feature maps from one layer with and without shared `vmin`/`vmax`. Which channel is genuinely the strongest?
    4. Build a confusion matrix for a badly imbalanced problem. Then normalise each row and display that instead. Which shows the failure more clearly?
    5. Take the decision-region plot and remove `alpha`. Then remove `edgecolors`. Which of the two mattered more for readability?

    Part 4 is about getting these out of the notebook and into something somebody else can read.
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
