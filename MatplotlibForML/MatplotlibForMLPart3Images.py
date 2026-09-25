#!/usr/bin/env uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Matplotlib for Machine Learning, Part 3: images and 2D fields

    In this notebook we will use `plt.imshow` to display images and data stored on a grid. This includes digits, feature maps and confusion matrices.

    We will also look at choosing colours. When a colour represents a value, we need to choose a colourmap that helps us read the data.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(42)
    return np, plt, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [plt.imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)

    ```python
    plt.imshow(X, cmap=None, norm=None, aspect=None, interpolation=None,
               vmin=None, vmax=None, origin='upper', extent=None)
    ```

    These are the parameters we will use in the examples:

    | Parameter | What it does |
    | --- | --- |
    | `X` | the array to display |
    | `cmap` | maps scalar values to colours |
    | `vmin`, `vmax` | set the values at either end of the colour scale |
    | `interpolation` | controls resampling when the image is drawn |
    | `origin` | places row 0 at the top (`'upper'`) or bottom (`'lower'`) |
    | `aspect` | use `'equal'` for square pixels, or `'auto'` to fill the axes |

    `imshow` accepts the following array layouts:

    | Input shape | Contents | Uses `cmap` |
    | --- | --- | --- |
    | `(M, N)` | scalar values | yes |
    | `(M, N, 3)` | RGB colours | no |
    | `(M, N, 4)` | RGBA colours | no |

    Passing `cmap="gray"` with an RGB image does not convert it to greyscale. The colourmap is ignored, as we can see below.

    Matplotlib expects channels last, `(H, W, C)`. For a torchvision tensor stored as `(C, H, W)`, we use `permute(1, 2, 0)` before displaying it. We used this in `TorchVisionForML` Part 1.
    """)
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

    axes_layouts[2].imshow(rgb, cmap="gray")  # RGB values already specify the colours
    axes_layouts[2].set_title(
        "(M, N, 3) + cmap='gray'\nno error, no effect", fontsize=9
    )

    for _ax in axes_layouts:
        _ax.axis("off")
    plt.show()
    return (digit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### axis("off")

    We can use `plt.axis("off")` to hide the axes when displaying a photograph. Keep them when the row and column labels help explain the data, such as in a confusion matrix.

    ### interpolation

    Resampling can smooth an image when it is enlarged. For a small digit or feature map, I use `interpolation="nearest"` to show the individual pixels. Compare the two versions below.
    """)
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
    mo.md(r"""
    ### vmin and vmax

    By default, `imshow` scales each scalar array separately. This means the same colour can represent different values in two plots.

    When comparing feature maps on the same scale, we set the same `vmin` and `vmax` for each. In the example below the first pair is scaled separately, whilst the second pair shares a range of 0 to 1.
    """)
    return


@app.cell
def _(plt, rng):
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
    mo.md(r"""
    ## Colormaps

    A [colourmap](https://matplotlib.org/stable/users/explain/colors/colormaps.html) maps values to colours. We will use three types here:

    | Type | Use | Examples |
    | --- | --- | --- |
    | sequential | values ordered from low to high | `viridis`, `plasma`, `gray`, `Blues` |
    | diverging | values either side of a meaningful midpoint | `coolwarm`, `RdBu`, `bwr` |
    | qualitative | categories with no numerical order | `tab10`, `Set2` |

    For pixel intensity or confidence, we can use a sequential map. For signed differences, we use a diverging map centred on zero. Class labels need separate colours without implying an order.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparing lightness

    I use `viridis` for sequential data in these examples. We can compare it with `jet` by looking at how lightness changes along each colourmap.

    For a sequential map, lightness should increase or decrease steadily. Fairly even changes also help us see small changes across the data range.

    The code below converts the colours to CIE L\*, a measure of perceptual lightness. We count changes in direction and calculate the variation in step size. This checks lightness only; it does not measure all aspects of colour perception.
    """)
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
        title="Lightness along each colourmap",
    )
    ax_L.legend()
    ax_L.grid(alpha=0.3)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We need to read the reversal count alongside the type of colourmap.

    `coolwarm` becomes lighter towards the middle and darker towards either end. This is useful for a diverging map: the hue distinguishes the two sides of the midpoint.

    `jet` also changes lightness direction. When we use it for sequential data, both low and high values can appear dark. Its uneven lightness steps can also emphasise bands in a smooth field.

    Compare the lightness step sizes in the printed results, then look at the same field drawn with both maps below. I use `viridis` for sequential values, `gray` for greyscale images and `coolwarm` for signed differences. We should also check whether the figure remains readable in greyscale.
    """)
    return


@app.cell
def _(np, plt, rng):
    # use the same field to compare the colourmaps
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
    mo.md(r"""
    ### Colour for class identity

    The multi-class example in `Classification/MultiClassificationMarimo.py` uses a sequential map:

    ```python
    plt.scatter(x=x[:, 0], y=x[:, 1], c=y, cmap=plt.cm.plasma)
    ```

    For two classes we need two distinguishable colours. With several classes, a sequential map can suggest an ordering that the labels do not have.

    Here we draw each class separately and add a legend. This lets us identify the classes by name without treating their labels as measured values.
    """)
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
    mo.md(r"""
    ## [colorbar](https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.colorbar)

    ```python
    fig.colorbar(mappable, ax=..., label=None, fraction=0.15, shrink=1.0)
    ```

    When colour represents a number, a colourbar shows how to read the scale.

    We pass the object returned by `imshow` or `contourf` as `mappable`. The `ax` argument specifies which axes make room for the colourbar.

    Add a label to explain the values, for example `fig.colorbar(im, ax=ax, label="activation")`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The confusion matrix

    A confusion matrix shows which classes a model confuses. We can draw it with `imshow` and add the counts as text.

    I use a sequential colourmap here because the values are counts. Printing each count also lets us read the exact values. The data below is generated for this example.
    """)
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
    confusion[0, 1] = 24  # add some cats predicted as dogs

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
    mo.md(r"""
    We switch between white and black text to keep the counts readable against the cell colours. The threshold below is a simple starting point; check the result with the colourmap you use.

    ## [contourf](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html) and decision boundaries

    ```python
    plt.contourf(X, Y, Z, levels=None, cmap=None, alpha=None)
    ```

    We can use `contourf` to display a model's predictions over a grid. This builds on `meshgrid` and `np.c_` from `NumPyForML` Part 2:

    1. Use `meshgrid` to cover the feature ranges.
    2. Use `ravel` and `np.c_` to make an array of points.
    3. Predict a class for each point.
    4. Reshape the predictions to match the grid.
    5. Draw the regions with `contourf` and scatter the training points on top.

    We use this approach in `Classification/BinaryClassificationMarimo.py`. A low `alpha` makes the fill transparent so we can still see the points.
    """)
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
    mo.md(r"""
    `levels=np.arange(-0.5, 4.5)` places the boundaries between integer class labels, giving us one band per class. The white outlines help separate the sample points from the background.

    ## Grids of images

    We can also display a batch of images to check a data pipeline or inspect predictions. Here we use random images and labels to demonstrate the layout, with three deliberately incorrect predictions.
    """)
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
    mo.md(r"""
    The example marks errors using the title colour. I would also add a text marker when sharing the figure, so readers do not need to distinguish the colours to find the errors. We will return to this in Part 4.

    ## Exercises

    1. Display an MNIST digit using `gray`, `gray_r`, `viridis` and `jet`. Which helps you judge stroke thickness, and why?
    2. Run the lightness calculation on `hot`, `magma` and `Blues`. Does a sequential map need to become lighter as values increase, or can it become darker?
    3. Display eight feature maps from one layer, first with separate scales and then with shared `vmin` and `vmax`. Which has the largest activations?
    4. Build a confusion matrix for an imbalanced dataset. Compare the counts with a version where each row is normalised. What does each show?
    5. Remove `alpha` from the decision-region plot, then remove `edgecolors`. How does each change affect readability?

    In Part 4 we will look at preparing figures to share outside the notebook.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
