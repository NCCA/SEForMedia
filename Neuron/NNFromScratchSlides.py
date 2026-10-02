# /// script
# requires-python = ">=3.10,<3.15"
#
# [tool.marimo-studio]
# default = "dashboard"
#
# [tool.marimo-studio.cells]
# cell-2 = {ref = "cell:v1:c399e080786ffb2716a26488f94a2c7e6a94acb8c34a07b789aca0d08eb54d8b:7d4c6eb0ddec7101c57f0909e36a24121b93ddaf3830db7ff41ed297b4fa3be1:0"}
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import inspect

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy.typing import NDArray

    return NDArray, inspect, mo, np, plt


@app.cell(hide_code=True)
def katex_fix(mo):
    # marimo 0.25 renders maths with KaTeX 0.18.5 but ships the 0.16.9 stylesheet, which
    # misses renamed classes like katex-sizing, so subscripts come out full size.
    # This loads the matching stylesheet for the notebook view, the slides bundle their own copy.
    # Remove once marimo ships a matching stylesheet.
    mo.Html(
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.18.5/dist/katex.min.css">'
    )
    return


@app.cell
def slide_order():
    # The slide order for the Studio deck. Each inner list is one topic: the first
    # name is the slide you reach with the right arrow, the rest stack under it and
    # are reached with the down arrow. Add a cell, give it a name, and list it here.
    deck = [  # noqa: F841 read by the Studio view, not by another cell
        ["master_slide"],
        [
            "fixed_neuron",
            "diagram1",
            "neuron_live",
            "input_stage",
            "activation_function",
            "activation_python",
        ],
        [
            "sigmoid_table",
            "sigmoid_explorer",
            "neuron_example",
            "sigmoid_derivative",
            "limitations",
        ],
        ["single_neuron", "neuron_demo", "and_gate_plot"],
        [
            "stage2_intro",
            "bce_explained",
            "bce_code",
            "gradient_explained",
            "gradient_weights",
            "compute_gradient_code",
            "numerical_gradient_code",
            "gradient_check",
        ],
        ["train_neuron_code", "train_neuron_live"],
        ["stage3_xor", "xor_training", "xor_idea"],
        [
            "stage4_network",
            "forward_mlp_code",
            "backprop_explained",
            "backprop_code",
            "mlp_gradient_check",
        ],
        ["train_mlp_code", "train_mlp_live"],
        ["wrap_up", "extensions"],
    ]
    return


@app.cell(hide_code=True)
def master_slide(mo):
    mo.md(r"""
    ## Introduction

    - In this notebook we are going to generate a simple neural network from scratch.
    - We will only use NumPy for our maths
    - We will look at how we can build a network to "solve" a simple gate
    """)
    return


@app.cell(hide_code=True)
def fixed_neuron(mo):
    mo.md(r"""
    ### Stage 1 a fixed weight neuron

    - An artificial neuron takes some numbers as inputs, combines them, and produces one number as its output.
    - It is a small mathematical model loosely inspired by a biological neuron.
    - The following diagram demonstrates the basic function.
    """)
    return


@app.cell
def diagram1(mo):
    mo.vstack(
        [
            mo.md("### Stage 1 A single fixed Neuron (fixed weights)"),
            mo.mermaid(r"""flowchart LR
        x1(("x₁")) -->|w₁| sum(("Σ"))
        x2(("x₂")) -->|w₂| sum
        bias(("1")) -->|b| sum
        sum -->|"z = x₁w₁ + x₂w₂ + b"| sigmoid(("σ"))
        sigmoid --> output["$$\sigma(z) = \frac{1}{1 + e^{-z}}$$"]

        classDef input fill:#55c1ed,stroke:#55c1ed,color:#000
        classDef sumNode fill:#ffff33,stroke:#ffff33,color:#000
        classDef activation fill:#b2dd85,stroke:#b2dd85,color:#000

        class x1,x2,bias input
        class sum sumNode
        class sigmoid activation"""),
        ]
    )
    return


@app.cell(hide_code=True)
def neuron_live(mo, neuron_inputs, sigmoid):
    # round to tidy up the float steps from the sliders
    _v = {_k: round(_val, 2) for _k, _val in neuron_inputs.value.items()}
    _z = _v["x1"] * _v["w1"] + _v["x2"] * _v["w2"] + _v["b"]
    _out = float(sigmoid(_z))

    def _colour(weight: float) -> str:
        # blue pushes z up, red pulls it down, grey does nothing
        if weight > 0:
            return "#2b7bb9"
        if weight < 0:
            return "#d64545"
        return "#999999"

    # white outline behind text so lines passing underneath don't make it unreadable
    _HALO = (
        'paint-order="stroke" stroke="white" stroke-width="5" stroke-linejoin="round"'
    )

    def _edge(
        x: float,
        y: float,
        weight: float,
        name: str,
        value: float,
        below: bool = False,
    ) -> str:
        # an input-to-sum edge, thicker for a bigger weight, labelled with the weight and its contribution
        _mx, _my = (x + 300) / 2, (y + 160) / 2
        # put the label beside the line rather than on it
        _ty = _my + 28 if below else _my - 26
        return f"""
        <line x1="{x}" y1="{y}" x2="300" y2="160" stroke="{_colour(weight)}"
              stroke-width="{1.5 + 2.5 * abs(weight)}" marker-end="url(#neuron-arrow)"/>
        <text x="{_mx}" y="{_ty}" text-anchor="middle" font-size="15" {_HALO}>{name} = {weight:g}
          <tspan x="{_mx}" dy="17" font-size="13" fill="#555">adds {value * weight:+.2f}</tspan></text>
        """

    def _node(x: float, y: float, fill: str, label: str) -> str:
        return f"""
        <circle cx="{x}" cy="{y}" r="26" fill="{fill}"/>
        <text x="{x}" y="{y + 6}" text-anchor="middle" font-size="18">{label}</text>
        """

    _svg = f"""
    <svg viewBox="0 0 820 340" width="100%" font-family="sans-serif" style="user-select: none; -webkit-user-select: none">
      <defs>
        <marker id="neuron-arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
        </marker>
      </defs>
      {_edge(86, 50, _v["w1"], "w₁", _v["x1"])}
      {_edge(86, 160, _v["w2"], "w₂", _v["x2"])}
      {_edge(86, 270, _v["b"], "b", 1.0, below=True)}
      {_node(60, 50, "#55c1ed", f"{_v['x1']:g}")}
      {_node(60, 160, "#55c1ed", f"{_v['x2']:g}")}
      {_node(60, 270, "#55c1ed", "1")}
      <text x="60" y="15" text-anchor="middle" font-size="14" fill="#555">x₁</text>
      <text x="60" y="125" text-anchor="middle" font-size="14" fill="#555">x₂</text>
      <text x="60" y="235" text-anchor="middle" font-size="14" fill="#555">bias</text>

      {_node(326, 160, "#ffff33", "Σ")}
      <line x1="352" y1="160" x2="474" y2="160" stroke="#333" stroke-width="1.5" marker-end="url(#neuron-arrow)"/>
      <text x="413" y="148" text-anchor="middle" font-size="16" {_HALO}>z = {_z:.2f}</text>

      {_node(500, 160, "#b2dd85", "σ")}
      <line x1="526" y1="160" x2="600" y2="160" stroke="#333" stroke-width="1.5" marker-end="url(#neuron-arrow)"/>

      <rect x="604" y="120" width="200" height="80" rx="6" fill="#ffffff" stroke="#333"/>
      <text x="704" y="152" text-anchor="middle" font-size="18">σ(z) = {_out:.3f}</text>
      <rect x="624" y="168" width="160" height="14" fill="#eeeeee"/>
      <rect x="624" y="168" width="{160 * _out:.1f}" height="14" fill="#b2dd85"/>
    </svg>
    """

    mo.vstack(
        [
            mo.md("## A single neuron, live"),
            mo.Html(_svg),
            mo.hstack(
                [
                    mo.vstack([neuron_inputs["x1"], neuron_inputs["x2"]]),
                    mo.vstack([neuron_inputs["w1"], neuron_inputs["w2"]]),
                    neuron_inputs["b"],
                ],
                justify="space-around",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def input_stage(mo):
    mo.md(r"""
    ## Input Stage

    - First, we multiply each input \(x_i\) by a weight \(w_i\), add the results, then add a bias \(b\):
    - The weights control how each input affects the result.
    - A positive weight makes increasing that input raise the result; a negative weight makes it lower the result.
    - The bias shifts the result up or down, independently of the inputs.
    """)
    return


@app.cell(hide_code=True)
def activation_function(mo):
    mo.md(r"""
    ## Activation Function

    - Next, we pass \(z\) through an activation function. One option is the sigmoid function:

    $$
    \sigma(z) = \frac{1}{1 + e^{-z}}
    $$

    - Here, \(e\) is the mathematical constant approximately equal to \(2.718\). Sigmoid squeezes any real number into a value between 0 and 1, following a smooth S shaped curve
    """)
    return


@app.cell
def activation_python(mo, np):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    mo.vstack(
        [
            mo.md("## Python Version"),
            mo.md(r"""
    ```python
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    ```
    """),
        ]
    )
    return (sigmoid,)


@app.cell(hide_code=True)
def sigmoid_table(mo, sigmoid):
    _rows = "\n".join(f"| {z} | {sigmoid(z):.3f} |" for z in (-6, -2, 0, 2, 6))

    mo.vstack(
        [
            mo.md("## Squashing values with sigmoid"),
            mo.hstack(
                [
                    mo.md(rf"""
    | Input \(z\) | Sigmoid output |
    | ---: | ---: |
    {_rows}
    """),
                    mo.md("""
    - Large negative inputs give outputs close to 0.
    - Large positive inputs give outputs close to 1.
    - At zero the output is exactly 0.5.
    """),
                ],
                widths=[1, 1],
                align="center",
            ),
        ]
    )
    return


@app.cell
def z_input(mo):
    z_slider = mo.ui.slider(
        -8.0,
        8.0,
        step=0.1,
        value=0.0,
        label=r"Input \(z\)",
        show_value=True,
        full_width=True,
    )
    return (z_slider,)


@app.cell(hide_code=True)
def sigmoid_explorer(mo, np, plt, sigmoid, z_slider):
    _z = np.linspace(-8, 8, 400)
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(_z, sigmoid(_z), label=r"$\sigma(z)$")
    for _level in (0.0, 0.5, 1.0):
        _ax.axhline(_level, color="grey", linestyle=":", linewidth=0.8)
    _ax.scatter([z_slider.value], [sigmoid(z_slider.value)], color="C1", s=60, zorder=3)
    _ax.set_xlabel("z")
    _ax.set_ylabel(r"$\sigma(z)$")
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md("## Exploring sigmoid"),
            z_slider,
            mo.md(rf"$$\sigma({z_slider.value:.1f}) = {sigmoid(z_slider.value):.3f}$$"),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell
def neuron_controls(mo):
    neuron_inputs = mo.ui.dictionary(
        {
            "x1": mo.ui.slider(
                -3.0,
                3.0,
                step=0.1,
                value=1.0,
                label=r"\(x_1\)",
                show_value=True,
            ),
            "x2": mo.ui.slider(
                -3.0,
                3.0,
                step=0.1,
                value=2.0,
                label=r"\(x_2\)",
                show_value=True,
            ),
            "w1": mo.ui.slider(
                -2.0,
                2.0,
                step=0.1,
                value=0.8,
                label=r"\(w_1\)",
                show_value=True,
            ),
            "w2": mo.ui.slider(
                -2.0,
                2.0,
                step=0.1,
                value=-0.4,
                label=r"\(w_2\)",
                show_value=True,
            ),
            "b": mo.ui.slider(
                -2.0, 2.0, step=0.1, value=0.5, label=r"\(b\)", show_value=True
            ),
        }
    )
    return (neuron_inputs,)


@app.cell(hide_code=True)
def neuron_example(mo, neuron_inputs, sigmoid):
    # round to tidy up the float steps from the sliders (0.8 can come back as 0.7999...)
    _v = {_k: round(_val, 2) for _k, _val in neuron_inputs.value.items()}
    _z = _v["x1"] * _v["w1"] + _v["x2"] * _v["w2"] + _v["b"]
    _out = sigmoid(_z)

    mo.vstack(
        [
            mo.md("## A worked example"),
            mo.hstack(
                [
                    mo.vstack([neuron_inputs["x1"], neuron_inputs["x2"]]),
                    mo.vstack([neuron_inputs["w1"], neuron_inputs["w2"]]),
                    neuron_inputs["b"],
                ],
                justify="start",
                gap=2,
            ),
            mo.md(rf"""
    $$z = ({_v["x1"]:g} \times {_v["w1"]:g}) + ({_v["x2"]:g} \times {_v["w2"]:g}) + {_v["b"]:g} = {_z:.3f}$$

    $$\text{{output}} = \sigma({_z:.3f}) \approx {_out:.3f}$$

    - If doing binary classification this output reads as an estimated probability, about {_out:.1%} here. Sigmoid alone doesn't make that estimate accurate, the neuron has to learn its weights and bias from data.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def sigmoid_derivative(mo, np, plt, sigmoid, z_slider):
    _z = np.linspace(-8, 8, 400)
    _s = sigmoid(_z)
    _here = sigmoid(z_slider.value)
    _slope = _here * (1 - _here)

    _fig, _ax = plt.subplots(figsize=(4.5, 2.6))
    _ax.plot(_z, _s, label=r"$\sigma(z)$")
    _ax.plot(_z, _s * (1 - _s), label=r"$\sigma'(z)$")
    _ax.scatter([z_slider.value], [_slope], color="C1", s=50, zorder=3)
    _ax.set_xlabel("z")
    _ax.legend(loc="upper left")
    _fig.tight_layout()

    # slider straight under the heading so it is always on screen
    mo.vstack(
        [
            mo.md("## Learning needs a slope"),
            z_slider,
            mo.hstack(
                [
                    mo.md(rf"""
    - Training nudges the weights and bias to reduce the error.
    - Sigmoid is smooth, so we can work out how a nudge changes the output.

    $$\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr)$$

    $$\sigma'({z_slider.value:.1f}) = {_slope:.3f}$$
    """),
                    mo.as_html(_fig),
                ],
                widths=[1, 1],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def limitations(mo):
    mo.md(r"""
    ## Limitations

    - One limitation is that the curve becomes very flat near 0 and 1.
    - The resulting small gradients can make learning slow.
    - Sigmoid is one choice of activation function, neurons can use other functions as we will learn later.
    """)
    return


@app.cell(hide_code=True)
def single_neuron(NDArray, mo, np, sigmoid):
    def neuron(
        X: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
    ) -> NDArray[np.float64]:
        z = X @ w + b
        return sigmoid(z)

    mo.vstack(
        [
            mo.md("## A single neuron"),
            mo.md(
                "- we can generate a neuron as follows, note the inputs for X and weights are NumPy arrays"
            ),
            mo.md(r"""
    ```python
    def neuron(X: NDArray[np.float64],
               w: NDArray[np.float64],
               b: float,) -> NDArray[np.float64]:
        z = X @ w + b
        return sigmoid(z)
    ```
    """),
        ]
    )
    return (neuron,)


@app.cell
def gate_controls(mo):
    gate_weights = mo.ui.dictionary(
        {
            "w1": mo.ui.slider(
                -20.0,
                20.0,
                step=0.5,
                value=1.0,
                label=r"\(w_1\)",
                show_value=True,
            ),
            "w2": mo.ui.slider(
                -20.0,
                20.0,
                step=0.5,
                value=1.0,
                label=r"\(w_2\)",
                show_value=True,
            ),
            "b": mo.ui.slider(
                -30.0,
                30.0,
                step=0.5,
                value=0.0,
                label=r"\(b\)",
                show_value=True,
            ),
        }
    )
    return (gate_weights,)


@app.cell(hide_code=True)
def neuron_demo(gate_weights, mo, neuron, np):
    # every combination of two binary inputs, one row per sample
    _X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    _w = np.array([gate_weights.value["w1"], gate_weights.value["w2"]])
    _b = gate_weights.value["b"]
    _out = neuron(_X, _w, _b)

    _rows = "\n".join(
        f"| {int(_a)} | {int(_c)} | {int(_a and _c)} | {_o:.3f} |"
        for (_a, _c), _o in zip(_X, _out)
    )
    _correct = sum(int(_o > 0.5) == int(_a and _c) for (_a, _c), _o in zip(_X, _out))

    mo.vstack(
        [
            mo.md("## Trying out the neuron"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            gate_weights["w1"],
                            gate_weights["w2"],
                            gate_weights["b"],
                            mo.md(rf"""
    ```python
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    neuron(X, np.array([{_w[0]:g}, {_w[1]:g}]), {_b:g})
    ```
    """),
                        ]
                    ),
                    mo.md(rf"""
    | \(x_1\) | \(x_2\) | AND | output |
    | :---: | :---: | :---: | ---: |
    {_rows}

    **{_correct} / 4** match AND (output above 0.5 counts as 1)
    """),
                ],
                widths=[1, 1],
                align="center",
            ),
            mo.md(
                "- All four samples go through in one call, `X @ w + b` does every row at once. Can you find weights and a bias that make an AND gate?"
            ),
        ]
    )
    return


@app.cell
def and_gate_plot(gate_weights, mo, neuron, np, plt, sigmoid):
    def plot_boundary(w, b, X, y, title, ax=None):
        """Shade the region where the neuron predicts class 1 vs class 0."""
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4))
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = sigmoid(grid @ w + b).reshape(xx.shape)
        ax.contourf(
            xx,
            yy,
            zz,
            levels=[0, 0.5, 1],
            colors=["#fde0dd", "#deebf7"],
            alpha=0.8,
        )
        for (px, py), target in zip(X, y):
            is_true = target > 0.5
            ax.text(
                px,
                py,
                "T" if is_true else "F",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="white",
                bbox={
                    "boxstyle": "circle",
                    "facecolor": "#d6604d" if is_true else "#4393c3",
                    "edgecolor": "k",
                },
                zorder=3,
            )
        ax.set_title(title)
        return ax

    # the AND truth table, the weights come from the shared gate_weights sliders
    _X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    _y = np.array([0.0, 0.0, 0.0, 1.0])
    _w = np.array([gate_weights.value["w1"], gate_weights.value["w2"]])
    _b = gate_weights.value["b"]
    _preds = neuron(_X, _w, _b)
    _accuracy = np.mean((_preds > 0.5) == _y) * 100

    _rows = "\n".join(
        f"| {_x[0]:.0f} | {_x[1]:.0f} | {_t:.0f} | {_p:.3f} |"
        for _x, _t, _p in zip(_X, _y, _preds, strict=True)
    )
    _table = mo.md(
        f"""
    **Accuracy:** {_accuracy:.0f}%

    | x1 | x2 | target | neuron output |
    |----|----|--------|----------------|
    {_rows}
    """
    )

    _ax = plot_boundary(_w, _b, _X, _y, "AND")

    mo.vstack(
        [
            mo.md("## Where the AND neuron draws the line"),
            mo.hstack(
                [_table, _ax.figure],
                justify="center",
                align="center",
                widths="equal",
            ),
            mo.hstack(
                [gate_weights["w1"], gate_weights["w2"], gate_weights["b"]],
                justify="center",
            ),
        ]
    )
    return (plot_boundary,)


@app.cell
def code_slide_helper(inspect, mo):
    def code_slide(title: str, *functions) -> mo.Html:
        """
        A slide heading followed by the source of one or more functions.

        Parameters
        ----------
            title : str
                markdown for the slide heading
            functions :
                the functions to show, their source comes from the running code so it never drifts
        """
        _blocks = "\n\n".join(
            f"```python\n{inspect.getsource(f).rstrip()}\n```" for f in functions
        )
        # longer functions overflow a slide at the deck's text size, so shrink the code a little
        return mo.vstack(
            [
                mo.md(f"## {title}"),
                mo.md(_blocks).style({"font-size": "0.7em"}),
            ]
        )

    return (code_slide,)


@app.cell
def logic_data(np):
    # the four input pairs for a two-input logic gate, and the target output for each gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    gates = {
        "AND": np.array([0, 0, 0, 1], dtype=float),
        "OR": np.array([0, 1, 1, 1], dtype=float),
        "XOR": np.array([0, 1, 1, 0], dtype=float),
    }
    return X, gates


@app.cell(hide_code=True)
def stage2_intro(mo):
    mo.md(r"""
    ## Stage 2 learning the weights

    - Finding weights by hand is fine for AND, but not for a network with thousands of weights. Instead we let the neuron learn them by repeating four steps:
        1. **Forward pass**, run the inputs through the neuron to get predictions \(a\).
        2. **Loss**, measure how wrong those predictions are.
        3. **Gradient**, work out which way each weight and the bias should move to reduce the loss.
        4. **Update**, take a small step in that direction.
    - This loop is called **gradient descent**.
    """)
    return


@app.cell
def prediction_input(mo):
    prediction_input = mo.ui.slider(
        0.01,
        0.99,
        step=0.01,
        value=0.8,
        label=r"Prediction \(a\)",
        show_value=True,
        full_width=True,
    )
    return (prediction_input,)


@app.cell(hide_code=True)
def bce_explained(mo, np, plt, prediction_input):
    _a = np.linspace(0.01, 0.99, 200)
    _p = prediction_input.value
    _fig, _ax = plt.subplots(figsize=(4.5, 2.6))
    _ax.plot(_a, -np.log(_a), label="target y = 1")
    _ax.plot(_a, -np.log(1 - _a), label="target y = 0")
    _ax.scatter(
        [_p, _p],
        [-np.log(_p), -np.log(1 - _p)],
        color=["C0", "C1"],
        s=50,
        zorder=3,
    )
    _ax.set_xlabel("prediction a")
    _ax.set_ylabel("loss")
    _ax.legend(loc="upper center")
    _fig.tight_layout()

    # the equation gets the full width, the half-width column is too narrow for it
    mo.vstack(
        [
            mo.md("## Measuring how wrong we are"),
            prediction_input,
            mo.md(r"""
    For a yes / no output we use **binary cross-entropy**:

    $$L = -\text{mean}\big(y \log a + (1 - y)\log(1 - a)\big)$$
    """),
            mo.hstack(
                [
                    mo.md(rf"""
    - \(y = 1\): loss is \(-\log a\), here **{-np.log(_p):.3f}**
    - \(y = 0\): loss is \(-\log(1 - a)\), here **{-np.log(1 - _p):.3f}**
    - Confident and wrong costs a lot.
    """),
                    mo.as_html(_fig),
                ],
                widths=[1, 1],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def bce_code(code_slide, np):
    def bce_loss(a, y):
        """Binary cross-entropy loss, averaged over examples."""
        # clip so we never take log(0)
        eps = 1e-9
        a = np.clip(a, eps, 1 - eps)
        return -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))

    code_slide("Binary cross-entropy in NumPy", bce_loss)
    return (bce_loss,)


@app.cell(hide_code=True)
def gradient_explained(mo):
    mo.md(r"""
    ## Which way should the weights move?

    The chain rule takes us from the loss back to \(z\):

    $$\frac{\partial L}{\partial a} = \frac{a - y}{a(1 - a)} \qquad \frac{\partial a}{\partial z} = a(1 - a)$$

    $$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a}\,\frac{\partial a}{\partial z} = a - y$$

    - The sigmoid derivative from earlier cancels out, leaving just the **error**.
    """)
    return


@app.cell(hide_code=True)
def gradient_weights(mo):
    mo.md(r"""
    ## Gradients for the weights and bias

    Our neuron computes

    $$z = x_1 w_1 + x_2 w_2 + b$$

    so for a batch of examples:

    $$\frac{\partial L}{\partial w} = \text{mean}\big((a - y)\,x\big) \qquad \frac{\partial L}{\partial b} = \text{mean}(a - y)$$

    - Each weight moves by the error scaled by the input that caused it.
    """)
    return


@app.cell(hide_code=True)
def compute_gradient_code(code_slide, neuron, np):
    def compute_gradient(X, y, w, b):
        """Gradient of the binary cross-entropy loss with respect to w and b."""
        a = neuron(X, w, b)
        error = a - y
        dw = np.mean(error[:, None] * X, axis=0)
        db = np.mean(error)
        return dw, db

    code_slide("`compute_gradient` ", compute_gradient)
    return (compute_gradient,)


@app.cell(hide_code=True)
def numerical_gradient_code(code_slide, np):
    def numerical_gradient(f, param, eps=1e-5):
        """Finite-difference gradient of scalar function f w.r.t. array param."""
        grad = np.zeros_like(param, dtype=float)
        it = np.nditer(param, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            original = param[idx]
            param[idx] = original + eps
            plus = f(param)
            param[idx] = original - eps
            minus = f(param)
            param[idx] = original
            grad[idx] = (plus - minus) / (2 * eps)
        return grad

    code_slide("Checking a gradient with finite differences", numerical_gradient)
    return (numerical_gradient,)


@app.cell
def eps_input(mo):
    eps_input = mo.ui.slider(
        steps=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12],
        value=1e-5,
        label=r"Step size \(\varepsilon\)",
        show_value=True,
    )
    return (eps_input,)


@app.cell(hide_code=True)
def gradient_check(
    X,
    bce_loss,
    compute_gradient,
    eps_input,
    gates,
    mo,
    neuron,
    np,
    numerical_gradient,
):
    # the same test point the lab notebook uses
    _y = gates["OR"]
    _w = np.array([0.3, -0.7])
    _b = 0.1
    _eps = eps_input.value

    def _loss(w, b):
        return bce_loss(neuron(X, w, b), _y)

    _dw_num = numerical_gradient(lambda w: _loss(w, _b), _w.copy(), eps=_eps)
    _db_num = numerical_gradient(lambda b: _loss(_w, b[0]), np.array([_b]), eps=_eps)[0]
    _dw, _db = compute_gradient(X, _y, _w, _b)

    _pairs = [
        ("w1", _dw[0], _dw_num[0]),
        ("w2", _dw[1], _dw_num[1]),
        ("b", _db, _db_num),
    ]
    _rows = "\n".join(
        f"| {_n} | {_an:.6f} | {_nu:.6f} | {abs(_an - _nu):.1e} |"
        for _n, _an, _nu in _pairs
    )
    _worst = max(abs(_an - _nu) for _, _an, _nu in _pairs)
    _verdict = "matches" if _worst < 1e-4 else "does **not** match"

    mo.vstack(
        [
            mo.md("## Does our gradient match?"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md(r"""
    Nudge a parameter by \(\pm\varepsilon\) and watch the loss:

    $$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \varepsilon) - L(\theta - \varepsilon)}{2\varepsilon}$$

    - Slow but simple, a good test for hand-written backprop.
    - Too big an \(\varepsilon\) is inaccurate, too small hits rounding error.
    """),
                            eps_input,
                        ]
                    ),
                    mo.md(f"""
    | | analytic | numerical | difference |
    | :---: | ---: | ---: | ---: |
    {_rows}

    `compute_gradient` {_verdict} the numerical gradient.
    """),
                ],
                widths=[1, 1],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def train_neuron_code(bce_loss, code_slide, compute_gradient, neuron, np):
    def train_neuron(X, y, lr, epochs, seed=0):
        rng = np.random.default_rng(seed)
        w = rng.normal(size=X.shape[1]) * 0.5
        b = 0.0
        history = []
        for _ in range(epochs):
            a = neuron(X, w, b)
            history.append(bce_loss(a, y))
            dw, db = compute_gradient(X, y, w, b)
            w = w - lr * dw
            b = b - lr * db
        return w, b, history

    code_slide(
        r"The training loop, \(w \leftarrow w - \eta \, \partial L / \partial w\)",
        train_neuron,
    )
    return (train_neuron,)


@app.cell
def train_controls(mo):
    train_controls = mo.ui.dictionary(
        {
            "gate": mo.ui.dropdown(
                options=["AND", "OR", "XOR"], value="OR", label="Gate"
            ),
            "lr": mo.ui.slider(
                0.01,
                3.0,
                step=0.01,
                value=0.5,
                label=r"Learning rate \(\eta\)",
                show_value=True,
            ),
            "epochs": mo.ui.slider(
                50, 3000, step=50, value=500, label="Epochs", show_value=True
            ),
        }
    )
    return (train_controls,)


@app.cell
def epoch_view(mo, train_controls):
    # rebuilt whenever the training settings change, so it always covers the full run
    epoch_view = mo.ui.slider(
        0,
        train_controls.value["epochs"],
        step=10,
        value=train_controls.value["epochs"],
        label="Show epoch",
        show_value=True,
        full_width=True,
    )
    return (epoch_view,)


@app.cell(hide_code=True)
def train_neuron_live(
    X,
    epoch_view,
    gates,
    mo,
    neuron,
    np,
    plot_boundary,
    plt,
    train_controls,
    train_neuron,
):
    _gate = train_controls.value["gate"]
    _y = gates[_gate]
    _lr = train_controls.value["lr"]
    _, _, _history = train_neuron(X, _y, _lr, train_controls.value["epochs"])
    # training is deterministic, so re-running to the chosen epoch replays the same run
    _w, _b, _ = train_neuron(X, _y, _lr, epoch_view.value)
    _accuracy = np.mean((neuron(X, _w, _b) > 0.5) == _y) * 100

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(8, 3.2))
    _ax1.plot(_history)
    _ax1.axvline(epoch_view.value, color="C1", linestyle="--")
    _ax1.set_xlabel("epoch")
    _ax1.set_ylabel("loss")
    _ax1.set_title("Training loss")
    plot_boundary(_w, _b, X, _y, f"{_gate} after {epoch_view.value} epochs", ax=_ax2)
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md("## Watching the neuron learn"),
            mo.hstack(
                [
                    train_controls["gate"],
                    train_controls["lr"],
                    train_controls["epochs"],
                ],
                justify="space-around",
            ),
            epoch_view,
            mo.as_html(_fig),
            mo.md(
                rf"\(w = [{_w[0]:.2f}, {_w[1]:.2f}]\), \(b = {_b:.2f}\), accuracy **{_accuracy:.0f}%**"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def stage3_xor(X, gate_weights, gates, mo, neuron, np, plot_boundary, plt):
    _y = gates["XOR"]
    _w = np.array([gate_weights.value["w1"], gate_weights.value["w2"]])
    _b = gate_weights.value["b"]
    _accuracy = np.mean((neuron(X, _w, _b) > 0.5) == _y) * 100

    _fig, _ax = plt.subplots(figsize=(4, 3.2))
    plot_boundary(_w, _b, X, _y, f"XOR by hand, {_accuracy:.0f}% correct", ax=_ax)
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md("## Stage 3 where a single neuron breaks"),
            mo.hstack(
                [gate_weights["w1"], gate_weights["w2"], gate_weights["b"]],
                justify="space-around",
            ),
            mo.hstack(
                [
                    mo.as_html(_fig),
                    mo.md("""
    - A single neuron draws one straight line.
    - No straight line separates XOR's Ts from its Fs, the best you can do is 3 out of 4.
    - So can training find something better?
    """),
                ],
                widths=[1, 1],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def xor_training(X, gates, mo, np, plt, train_neuron):
    _, _, _history = train_neuron(X, gates["XOR"], 0.5, 2000)

    _fig, _ax = plt.subplots(figsize=(4.5, 2.6))
    _ax.plot(_history)
    _ax.axhline(np.log(2), color="grey", linestyle=":")
    _ax.text(
        len(_history) * 0.5,
        np.log(2) + 0.03,
        "ln 2 = 0.693",
        ha="center",
        va="bottom",
        color="grey",
    )
    # a fixed 0 to 1 scale, otherwise matplotlib zooms in on the tiny change and the plateau looks like progress
    _ax.set_ylim(0, 1)
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("loss")
    _ax.set_title("Training on XOR")
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md("## Training doesn't help"),
            mo.hstack(
                [
                    mo.as_html(_fig),
                    mo.md(rf"""
    - Loss stalls at **{_history[-1]:.3f}** = \(\ln 2\), the same as always answering 0.5.
    - More epochs or another learning rate won't help, the model is the problem.
    """),
                ],
                widths=[1, 1],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def xor_idea(mo):
    mo.vstack(
        [
            mo.md("## Combining straight lines"),
            mo.hstack(
                [
                    mo.md(r"""
    | \(x_1\) | \(x_2\) | OR | NAND | XOR |
    | :---: | :---: | :---: | :---: | :---: |
    | 0 | 0 | 0 | 1 | 0 |
    | 0 | 1 | 1 | 1 | 1 |
    | 1 | 0 | 1 | 1 | 1 |
    | 1 | 1 | 1 | 0 | 0 |
    """),
                    mo.md("""
    - XOR is **OR** AND **NAND**, at least one input on but not both.
    - OR and NAND are each one straight line, so one neuron can learn each.
    - A third neuron ANDs their outputs together.
    - Neurons feeding other neurons form a **hidden layer**.
    """),
                ],
                widths=[1, 1],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def stage4_network(mo):
    mo.vstack(
        [
            mo.md("## Stage 4 a two-layer network"),
            mo.hstack(
                [
                    mo.mermaid("""flowchart LR
        x1(("x₁")) --> h1(("h₁"))
        x1 --> h2(("h₂"))
        x2(("x₂")) --> h1
        x2 --> h2
        h1 --> out(("a₂"))
        h2 --> out

        classDef input fill:#55c1ed,stroke:#55c1ed,color:#000
        classDef hidden fill:#ffff33,stroke:#ffff33,color:#000
        classDef output fill:#b2dd85,stroke:#b2dd85,color:#000

        class x1,x2 input
        class h1,h2 hidden
        class out output"""),
                    mo.md(r"""
    $$z_1 = X W_1 + b_1 \qquad a_1 = \sigma(z_1)$$

    $$z_2 = a_1 W_2 + b_2 \qquad a_2 = \sigma(z_2)$$

    Shapes: \(X\) is \((n, 2)\), \(W_1\) is \((2, h)\), \(b_1\) is \((h,)\), \(W_2\) is \((h, 1)\), \(b_2\) is \((1,)\).

    - Each column of \(W_1\) is one hidden neuron, just like our single neuron.
    """),
                ],
                widths=[1, 1],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def forward_mlp_code(code_slide, sigmoid):
    def forward_mlp(X, W1, b1, W2, b2):
        """Forward pass through one hidden layer and a sigmoid output."""
        z1 = X @ W1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)
        return a1, a2

    code_slide("The forward pass", forward_mlp)
    return (forward_mlp,)


@app.cell(hide_code=True)
def backprop_explained(mo):
    mo.vstack(
        [
            mo.md(r"""
    ## Backpropagation

    The same chain rule, one layer at a time from the output back, with \(y\) as a column \((n, 1)\).
    """),
            mo.hstack(
                [
                    mo.md(r"""
    **Output layer**

    $$dz_2 = a_2 - y$$

    $$dW_2 = \tfrac{1}{n} a_1^T dz_2$$

    $$db_2 = \text{mean}(dz_2)$$

    - Exactly Stage 2, with \(a_1\) in place of \(X\).
    """),
                    mo.md(r"""
    **Hidden layer**

    $$dz_1 = (dz_2\, W_2^T) \odot a_1 \odot (1 - a_1)$$

    $$dW_1 = \tfrac{1}{n} X^T dz_1$$

    $$db_1 = \text{mean}(dz_1,\ \text{axis}=0)$$

    - \(a_1(1 - a_1)\) is the sigmoid derivative again.
    """),
                ],
                widths=[1, 1],
                align="start",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def backprop_code(code_slide, forward_mlp, np):
    def compute_gradients_mlp(X, y, W1, b1, W2, b2):
        n = X.shape[0]
        y_col = y.reshape(-1, 1)
        a1, a2 = forward_mlp(X, W1, b1, W2, b2)
        # output layer
        dz2 = a2 - y_col  # (n, 1)
        dW2 = a1.T @ dz2 / n  # (h, 1)
        db2 = np.mean(dz2, axis=0)  # (1,)
        # hidden layer, chain rule back through sigmoid
        da1 = dz2 @ W2.T  # (n, h)
        dz1 = da1 * a1 * (1 - a1)  # (n, h)
        dW1 = X.T @ dz1 / n  # (2, h)
        db1 = np.mean(dz1, axis=0)  # (h,)
        return dW1, db1, dW2, db2

    code_slide("`compute_gradients_mlp`", compute_gradients_mlp)
    return (compute_gradients_mlp,)


@app.cell(hide_code=True)
def mlp_gradient_check(
    X,
    bce_loss,
    compute_gradients_mlp,
    forward_mlp,
    gates,
    mo,
    np,
    numerical_gradient,
):
    # the same test point the lab notebook uses
    _rng = np.random.default_rng(1)
    _X = X[:3]
    _y = gates["XOR"][:3]
    _W1 = _rng.normal(size=(2, 3)) * 0.5
    _b1 = _rng.normal(size=3) * 0.5
    _W2 = _rng.normal(size=(3, 1)) * 0.5
    _b2 = _rng.normal(size=1) * 0.5

    def _loss(W1=_W1, b1=_b1, W2=_W2, b2=_b2):
        _, _a2 = forward_mlp(_X, W1, b1, W2, b2)
        return bce_loss(_a2, _y.reshape(-1, 1))

    _analytic = compute_gradients_mlp(_X, _y, _W1, _b1, _W2, _b2)
    _numeric = (
        numerical_gradient(lambda w: _loss(W1=w), _W1.copy()),
        numerical_gradient(lambda b: _loss(b1=b), _b1.copy()),
        numerical_gradient(lambda w: _loss(W2=w), _W2.copy()),
        numerical_gradient(lambda b: _loss(b2=b), _b2.copy()),
    )
    _names = (r"\(W_1\)", r"\(b_1\)", r"\(W_2\)", r"\(b_2\)")
    _diffs = [float(np.max(np.abs(_an - _nu))) for _an, _nu in zip(_analytic, _numeric)]
    _rows = "\n".join(
        f"| {_n} | {_an.shape} | {_d:.1e} |"
        for _n, _an, _d in zip(_names, _analytic, _diffs)
    )
    _verdict = "matches" if max(_diffs) < 1e-4 else "does **not** match"

    mo.md(f"""
    ## Checking backprop

    The same finite-difference check, on every parameter of a network with 3 hidden neurons.

    | parameter | shape | largest difference |
    | :---: | :---: | ---: |
    {_rows}

    `compute_gradients_mlp` {_verdict} the numerical gradient.
    """)
    return


@app.cell(hide_code=True)
def train_mlp_code(
    bce_loss,
    code_slide,
    compute_gradients_mlp,
    forward_mlp,
    np,
):
    def train_mlp(X, y, hidden_units, lr, epochs, seed=0):
        rng = np.random.default_rng(seed)
        W1 = rng.normal(size=(X.shape[1], hidden_units)) * 0.5
        b1 = np.zeros(hidden_units)
        W2 = rng.normal(size=(hidden_units, 1)) * 0.5
        b2 = np.zeros(1)
        y_col = y.reshape(-1, 1)
        history = []
        for _ in range(epochs):
            _, a2 = forward_mlp(X, W1, b1, W2, b2)
            history.append(bce_loss(a2, y_col))
            dW1, db1, dW2, db2 = compute_gradients_mlp(X, y, W1, b1, W2, b2)
            W1 = W1 - lr * dW1
            b1 = b1 - lr * db1
            W2 = W2 - lr * dW2
            b2 = b2 - lr * db2
        return W1, b1, W2, b2, history

    code_slide("Training the network", train_mlp)
    return (train_mlp,)


@app.cell
def mlp_plot_helpers(forward_mlp, np, plt):
    def plot_boundary_mlp(W1, b1, W2, b2, X, y, title, ax=None):
        """Shade the region where the network predicts class 1 vs class 0."""
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4))
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        _, a2 = forward_mlp(grid, W1, b1, W2, b2)
        zz = a2.reshape(xx.shape)
        ax.contourf(
            xx,
            yy,
            zz,
            levels=[0, 0.5, 1],
            colors=["#fde0dd", "#deebf7"],
            alpha=0.8,
        )
        for (px, py), target in zip(X, y):
            is_true = target > 0.5
            ax.text(
                px,
                py,
                "T" if is_true else "F",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="white",
                bbox={
                    "boxstyle": "circle",
                    "facecolor": "#d6604d" if is_true else "#4393c3",
                    "edgecolor": "k",
                },
                zorder=3,
            )
        ax.set_title(title)
        return ax

    return (plot_boundary_mlp,)


@app.cell
def mlp_controls(mo):
    mlp_controls = mo.ui.dictionary(
        {
            "hidden": mo.ui.slider(
                1, 8, step=1, value=2, label="Hidden units", show_value=True
            ),
            "lr": mo.ui.slider(
                0.01,
                3.0,
                step=0.01,
                value=1.0,
                label=r"Learning rate \(\eta\)",
                show_value=True,
            ),
            "epochs": mo.ui.slider(
                200,
                8000,
                step=200,
                value=3000,
                label="Epochs",
                show_value=True,
            ),
            "seed": mo.ui.slider(0, 20, step=1, value=0, label="Seed", show_value=True),
        }
    )
    return (mlp_controls,)


@app.cell(hide_code=True)
def train_mlp_live(
    X,
    forward_mlp,
    gates,
    mlp_controls,
    mo,
    np,
    plot_boundary_mlp,
    plt,
    train_mlp,
):
    _v = mlp_controls.value
    _y = gates["XOR"]
    _W1, _b1, _W2, _b2, _history = train_mlp(
        X, _y, _v["hidden"], _v["lr"], _v["epochs"], seed=_v["seed"]
    )
    _, _a2 = forward_mlp(X, _W1, _b1, _W2, _b2)
    _accuracy = np.mean((_a2.ravel() > 0.5) == _y) * 100

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(7.5, 2.6))
    _ax1.plot(_history)
    _ax1.set_xlabel("epoch")
    _ax1.set_ylabel("loss")
    _ax1.set_title("Training loss (XOR)")
    plot_boundary_mlp(
        _W1, _b1, _W2, _b2, X, _y, f"XOR, {_accuracy:.0f}% correct", ax=_ax2
    )
    # each hidden neuron switches over along the line W1[0] x + W1[1] y + b1 = 0
    _xs = np.array([-0.5, 1.5])
    for _j in range(_W1.shape[1]):
        _wx, _wy = _W1[:, _j]
        if abs(_wy) > 1e-6:
            _ax2.plot(
                _xs,
                -(_wx * _xs + _b1[_j]) / _wy,
                "k--",
                linewidth=1,
                alpha=0.6,
            )
        elif abs(_wx) > 1e-6:
            _ax2.axvline(
                -_b1[_j] / _wx,
                color="k",
                linestyle="--",
                linewidth=1,
                alpha=0.6,
            )
    _ax2.set_xlim(-0.5, 1.5)
    _ax2.set_ylim(-0.5, 1.5)
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md("## A hidden layer solves XOR"),
            mo.hstack(
                [
                    mlp_controls["hidden"],
                    mlp_controls["lr"],
                    mlp_controls["epochs"],
                    mlp_controls["seed"],
                ],
                justify="space-around",
            ),
            mo.as_html(_fig),
            mo.md(rf"""
    Dashed lines: where each hidden neuron switches over. Final loss **{_history[-1]:.3f}**, try seed 7 with 2 hidden units!
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def wrap_up(mo):
    mo.md(r"""
    ## Conclusion

    - A single neuron is a linear classifier, it can only separate data with one straight cut.
    - Training is the same loop every time: forward pass, loss, gradient, update.
    - A hidden layer combines several straight cuts into a curved boundary, which is exactly what XOR needs.
    - Everything that follows in deep learning (more layers, more units, different activations) is the same idea scaled up.
    """)
    return


@app.cell(hide_code=True)
def extensions(mo):
    mo.md(r"""
    ## Things to try

    - Increase the hidden units and see how much the boundary shape changes.
    - Use a stricter `atol` in the gradient checks.
    - Swap the hidden layer's sigmoid for `tanh` or ReLU, only one line of backprop changes.
    - Generalise `forward_mlp` / `compute_gradients_mlp` into a `Layer` class that can be stacked to any depth.
    - Build the same network in [PyTorch](https://pytorch.org/docs/stable/nn.html) and compare your gradients with autograd.
    """)
    return


if __name__ == "__main__":
    app.run()
