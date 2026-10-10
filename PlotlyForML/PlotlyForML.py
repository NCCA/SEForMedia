#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Intoduction to Plotly

    [Plotly](https://plotly.com/python/) is a more modern version of matplotlib, with the main difference being that is is interactive by default. This notebook takes the same format as the [Matplotlib notebooks](../MatplotlibForML/MatplotlibForMLPart1Figures.py) where I built figures for training curves, results and images. Here we will use the
    same kinds of data here, but this time we can hover over points, zoom into
    a region and hide a series by clicking its legend entry.

    I mainly use Matplotlib for figures in reports. Plotly is useful whilst
    exploring results, particularly when I want to inspect individual samples
    without adding a label to every point. We will start with a small line plot,
    then look at training curves, distributions, subplots and saving our work.

    Plotly is already in the project dependencies, if you need to add it to your own use

    ```bash
    uv add plotly
    ```
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    print("Plotly", plotly.__version__)
    return go, make_subplots, mo, np, pd, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two ways to build a figure

    [Plotly Express](https://plotly.com/python/plotly-express/) builds a complete
    figure from arrays or a table in one call. We use `plotly.express` as `px`.
    [Graph Objects](https://plotly.com/python/graph-objects/) lets us build the
    figure from individual traces, using `plotly.graph_objects` as `go`.

    Both produce a `go.Figure`. Express is where I would start for a quick plot;
    we can still change the resulting figure using Graph Objects methods.

    | Name | What it represents |
    | --- | --- |
    | Figure | The whole plot, including data and layout. |
    | Trace | One set of marks, such as a line, scatter points or histogram. |
    | Layout | Titles, axes, legend, margins and other figure settings. |

    A line is a `Scatter` trace with its mode set to `"lines"` or
    `"lines+markers"`. There is no separate `go.Line` trace.
    """)
    return


@app.cell
def _(np, px):
    x = np.linspace(0, 2 * np.pi, 41)
    y = np.sin(x)
    express_figure = px.line(
        x=x,
        y=y,
        markers=True,
        title="A sine wave with Plotly Express",
        labels={"x": "Angle (radians)", "y": "sin(x)"},
        template="plotly_white",
    )
    express_figure
    return x, y


@app.cell
def _(go, x, y):
    _figure = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers", name="sin(x)"))
    _figure.update_layout(
        title="The same data with Graph Objects",
        xaxis_title="Angle (radians)",
        yaxis_title="sin(x)",
        template="plotly_white",
    )
    print("Figure:", type(_figure).__name__)
    print("Traces:", len(_figure.data), "| first trace:", _figure.data[0].type)
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In marimo we put the figure at the end of the cell to display it. In a Python
    script, `fig.show()` uses Plotly's configured renderer to display the figure.

    Hover over a marker to read its coordinates. Drag a rectangle to zoom in and
    double-click to reset the view. The toolbar also has zoom and pan controls.
    This interaction changes the browser's view; it does not change our arrays.

    ## [px.line](https://plotly.com/python-api-reference/generated/plotly.express.line.html)

    ```python
    px.line(data_frame, x="epoch", y="loss", color="split", markers=True)
    ```

    These are the options we will use:

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `data_frame` | `None` | Table containing the named columns. |
    | `x`, `y` | `None` | Column names or arrays for coordinates. |
    | `color` | `None` | Groups rows into coloured series. |
    | `markers` | `False` | Adds a marker at each point. |
    | `labels` | `None` | Replaces column names in axes, legends and hover text. |
    | `template` | `None` | Selects a style; otherwise the configured default applies. |

    We will make a table with one row per epoch and split. This is often called
    long-form data. It lets us use the `split` column to group the lines.
    """)
    return


@app.cell
def _(np, pd):
    epochs = np.arange(1, 61)
    train_loss = 2.0 * np.exp(-epochs / 12) + 0.06
    valid_loss = 2.0 * np.exp(-epochs / 14) + 0.18
    valid_loss = valid_loss + np.maximum(epochs - 30, 0) * 0.012
    accuracy = 0.5 + 0.42 * (1 - np.exp(-epochs / 15))
    history = pd.DataFrame(
        {
            "epoch": np.tile(epochs, 2),
            "loss": np.concatenate([train_loss, valid_loss]),
            "split": np.repeat(["Training", "Validation"], len(epochs)),
        }
    )
    print("History shape:", history.shape)
    print(history.head())
    return accuracy, epochs, history, train_loss, valid_loss


@app.cell
def _(history, px):
    loss_figure = px.line(
        history,
        x="epoch",
        y="loss",
        color="split",
        line_dash="split",
        labels={
            "epoch": "Epoch",
            "loss": "Cross-entropy loss (nats)",
            "split": "Split",
        },
        color_discrete_map={"Training": "#0072B2", "Validation": "#D55E00"},
        title="Synthetic training and validation loss",
        template="plotly_white",
    )
    loss_figure.update_layout(hovermode="x unified", height=400)
    loss_figure
    return (loss_figure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The validation loss starts rising whilst training loss continues to fall.
    In a real run this would suggest overfitting. These are invented curves to
    practise plotting; the accuracy values below are also generated separately.

    `hovermode="x unified"` puts the losses for an epoch in one hover panel.
    Click a legend entry to hide that trace, then click it again to restore it.
    We have used both colour and line style to distinguish the splits.

    ## Updating a figure

    `update_layout()` changes the figure's settings, `update_traces()` changes
    its marks, and `add_annotation()` adds text at a chosen position. These
    methods modify the figure. In a reactive notebook I make a copy before
    customising a figure from another cell, so rerunning this cell does not
    keep adding annotations to the original.
    """)
    return


@app.cell
def _(epochs, go, loss_figure, np, valid_loss):
    best_index = int(np.argmin(valid_loss))
    best_epoch = int(epochs[best_index])
    annotated_figure = go.Figure(loss_figure)
    annotated_figure.update_traces(line_width=3)
    annotated_figure.add_annotation(
        x=best_epoch,
        y=float(valid_loss[best_index]),
        text=f"Lowest validation loss: epoch {best_epoch}",
        showarrow=True,
        arrowhead=2,
        ax=70,
        ay=-55,
    )
    print(
        "Best epoch:",
        best_epoch,
        "| validation loss:",
        round(valid_loss[best_index], 3),
    )
    annotated_figure
    return (annotated_figure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A line follows the input order

    Plotly joins line points in the order we supply them. It does not sort by
    the x coordinate. A shuffled training log can therefore produce a line
    which doubles back on itself without raising an error.

    Compare the figures below. We sort within each split and epoch before
    drawing the corrected line. Use a scatter plot when there is no meaningful
    order between samples.
    """)
    return


@app.cell
def _(history, mo, px):
    _shuffled = history.sample(frac=1, random_state=42)
    _wrong = px.line(
        _shuffled, x="epoch", y="loss", color="split", title="Shuffled rows"
    )
    _correct = px.line(
        _shuffled.sort_values(["split", "epoch"]),
        x="epoch",
        y="loss",
        color="split",
        title="Sorted within each split",
    )
    mo.hstack([_wrong, _correct], widths="equal")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [px.scatter](https://plotly.com/python-api-reference/generated/plotly.express.scatter.html)

    We can put a sample identifier in `hover_name` and extra columns in
    `hover_data`. This is useful when investigating an outlier in a feature
    plot: we can find the sample without printing hundreds of labels.

    Our class labels are strings so Express treats them as categories. If we
    pass integer class IDs as `color`, Express treats those values as a
    continuous quantity. Convert IDs to strings when they represent classes.
    The coordinates here are generated features, not a trained embedding.
    """)
    return


@app.cell
def _(np, pd, px):
    _rng = np.random.default_rng(42)
    _points = np.concatenate(
        [
            _rng.normal([-1, -0.5], 0.55, (80, 2)),
            _rng.normal([1, 0.5], 0.55, (80, 2)),
        ]
    )
    samples = pd.DataFrame(
        {
            "feature_1": _points[:, 0],
            "feature_2": _points[:, 1],
            "class": np.repeat(["Class 0", "Class 1"], 80),
            "sample": [f"sample_{i:03d}" for i in range(160)],
        }
    )
    scatter_figure = px.scatter(
        samples,
        x="feature_1",
        y="feature_2",
        color="class",
        symbol="class",
        hover_name="sample",
        hover_data={"feature_1": ":.2f", "feature_2": ":.2f"},
        opacity=0.65,
        labels={"feature_1": "Feature 1", "feature_2": "Feature 2"},
        title="Inspecting individual samples",
        template="plotly_white",
    )
    scatter_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Histograms and bin width

    A histogram groups numerical values into intervals and counts the values
    in each. We will use the same generated weights whilst changing the bin
    width. Watch how a wide interval hides detail and a narrow one makes
    small fluctuations more visible.

    Express has `nbins`, but it is a target maximum rather than an exact bin
    count. Here we use `go.Histogram` with explicit `xbins` to choose the
    interval edges. Histogram counts are computed in the browser; the trace's
    `y` property does not contain the resulting counts.
    """)
    return


@app.cell
def _(mo):
    bin_width = mo.ui.slider(
        start=0.05, stop=0.5, step=0.05, value=0.15, label="Bin width"
    )
    bin_width
    return (bin_width,)


@app.cell
def _(bin_width, go, np):
    _weights = np.random.default_rng(7).normal(0, 0.35, 1000)
    histogram_figure = go.Figure(
        go.Histogram(
            x=_weights,
            xbins=dict(start=-1.5, end=1.5, size=bin_width.value),
            marker_color="#0072B2",
        )
    )
    histogram_figure.update_layout(
        title="Generated weight distribution",
        xaxis_title="Weight",
        yaxis_title="Count",
        template="plotly_white",
        bargap=0.05,
    )
    print("Samples:", len(_weights), "| bin width:", bin_width.value)
    histogram_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [Subplots](https://plotly.com/python/subplots/)

    Loss and accuracy measure different things. We will give each its own
    y-axis in stacked plots and share the epoch axis. `make_subplots()` creates
    the grid, then `add_trace(..., row=..., col=...)` places each trace.
    Row and column numbers start at 1.
    """)
    return


@app.cell
def _(accuracy, epochs, go, make_subplots, train_loss, valid_loss):
    metrics_figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12
    )
    metrics_figure.add_trace(
        go.Scatter(x=epochs, y=train_loss, name="Training", mode="lines"),
        row=1,
        col=1,
    )
    metrics_figure.add_trace(
        go.Scatter(
            x=epochs,
            y=valid_loss,
            name="Validation",
            mode="lines",
            line=dict(dash="dash"),
        ),
        row=1,
        col=1,
    )
    metrics_figure.add_trace(
        go.Scatter(x=epochs, y=accuracy, name="Validation accuracy", mode="lines"),
        row=2,
        col=1,
    )
    metrics_figure.update_yaxes(title_text="Loss (nats)", row=1, col=1)
    metrics_figure.update_yaxes(title_text="Accuracy", range=[0, 1], row=2, col=1)
    metrics_figure.update_xaxes(title_text="Epoch", row=2, col=1)
    metrics_figure.update_layout(
        title="Synthetic training metrics", height=600, template="plotly_white"
    )
    metrics_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Images and 2D arrays

    [`px.imshow`](https://plotly.com/python/imshow/) displays an image or a
    numerical field. Here we use a small matrix so we can check the values by
    hand. Hover over each cell and compare it with the printed array.

    `origin="upper"` puts row zero at the top, matching the usual image layout.
    We set the colour range explicitly; when comparing several fields, use
    the same range so equal colours represent equal values.
    """)
    return


@app.cell
def _(np, px):
    _field = np.array([[0.0, 0.2, 0.4], [0.2, 0.5, 0.7], [0.4, 0.7, 1.0]])
    print("Field shape:", _field.shape)
    print(_field)
    image_figure = px.imshow(
        _field,
        origin="upper",
        zmin=0,
        zmax=1,
        text_auto=".1f",
        color_continuous_scale="Viridis",
        labels={"x": "Column", "y": "Row", "color": "Value"},
        title="A small numerical field",
    )
    image_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saving an interactive figure

    [HTML export](https://plotly.com/python/interactive-html-export/) keeps
    hover, zoom and legend interaction. A self-contained file includes the
    Plotly JavaScript library, so it is larger but works without a network
    connection. Using `include_plotlyjs="cdn"` makes a smaller file which
    needs access to the hosted library.

    In a script we can save directly:

    ```python
    annotated_figure.write_html("training_loss.html", include_plotlyjs=True)
    ```

    Here we offer a download so running the notebook does not write files into
    the source directory. Open the downloaded HTML in a browser and check the
    labels and interaction.
    """)
    return


@app.cell
def _(annotated_figure, mo):
    html_bytes = annotated_figure.to_html(full_html=True, include_plotlyjs=True).encode(
        "utf-8"
    )
    mo.download(
        data=html_bytes,
        filename="training_loss.html",
        mimetype="text/html",
        label="Download training curve",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For reports we can use `fig.write_image("training_loss.png")` or select
    SVG or PDF by changing the extension. [Static image export](https://plotly.com/python/static-image-export/)
    requires Kaleido and a compatible Chrome installation; Kaleido is not a
    dependency of this project, so we do not run that example here. A static
    image cannot preserve hover information. Put important values in labels
    or a caption if the reader needs them.

    Plotly layout width and height use pixels rather than Matplotlib's inches.
    Check text at the final displayed size, and keep bar chart value axes
    starting at zero. Interactivity still needs readable labels and sensible
    scales.

    ## Exercises

    1. Add markers to the training curves and use a log y-axis. Why must the
       loss values be positive for that scale?
    2. Add an annotation to the final validation point. Keep the original
       `loss_figure` unchanged when the annotation cell is rerun.
    3. Give one scatter sample an unusual feature value. Find its identifier
       by hovering over it, then try colouring points with integer class IDs.
       What changes when you convert the IDs to strings?
    4. Compare three histogram bin widths. Which features of the distribution
       stay visible? Which appear only with narrow bins?
    5. Download the training curve, open it in a browser and hide the training
       trace. Explain what a static screenshot would lose.

    We can now use these figures alongside the NumPy and PyTorch examples.
    Try replacing the generated training history with values from one of your
    own runs, keeping the split names, units and epoch order clear.
    """)
    return


if __name__ == "__main__":
    app.run()
