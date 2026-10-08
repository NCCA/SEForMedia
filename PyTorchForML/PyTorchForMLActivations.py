#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PyTorch for Machine Learning: activation functions

    In Part 3 we saw that stacking `nn.Linear` layers without anything in between collapses to a single linear layer, and that putting an `nn.ReLU` between them stops this happening. The function we put between the layers is called an *activation function* (or just a non-linearity), and PyTorch ships with quite a lot of them.

    This notebook looks at the activations listed in the [`torch.nn` documentation](https://docs.pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity). The same set is available in the C++ frontend, and the [C++ activation page](https://docs.pytorch.org/cppdocs/api/nn/activation.html) groups them in a useful way, so I have followed that grouping here:

    1. Sigmoid and Tanh, and why they cause trouble in deep networks.
    2. ReLU and its relatives.
    3. The smooth alternatives such as ELU, GELU and SiLU.
    4. The clamping and shrinking functions.
    5. Softmax and the other functions that work across a whole dimension.

    Most of the sections have sliders. Move them and watch both the function and its derivative, as the derivative is what the optimiser actually sees during training. We finish by training a tiny network with each activation so we can see the effect on a real fit.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from torch import nn

    torch.manual_seed(42)
    return F, nn, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Measuring the derivative with autograd

    Rather than writing out the derivative of every activation by hand, we can ask autograd for it, as we did in Part 2. An element-wise activation only uses its own input to produce each output, so if we sum the outputs and call backward, the gradient at each position is the derivative of the activation at that input. One backward pass gives us the whole curve.

    The two helpers below do this and draw the function next to its derivative. Every plot in the notebook uses them, so all the derivatives you see come from PyTorch, not from formulas I have typed in.
    """)
    return


@app.cell
def _(nn, plt, torch):
    def evaluate(
        activation: nn.Module, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies an element-wise activation and uses autograd to find its derivative.

        Parameters
        ----------
            activation : nn.Module
                the activation to evaluate, for example nn.ReLU()
            x : torch.Tensor
                the inputs to evaluate the activation at

        Returns
        -------
            tuple[torch.Tensor, torch.Tensor]
                the outputs f(x) and the derivatives f'(x), detached from the graph
        """
        x = x.detach().clone().requires_grad_(True)
        y = activation(x)
        # each output only depends on its own input, so d(sum)/dx_i is f'(x_i)
        (dy,) = torch.autograd.grad(y.sum(), x)
        return y.detach(), dy

    def plot_curves(
        x: torch.Tensor,
        curves: dict[str, nn.Module],
        title: str,
        dashed: set[str] | None = None,
    ) -> plt.Figure:
        """
        Plots each activation and its derivative side by side.

        Parameters
        ----------
            x : torch.Tensor
                the inputs to evaluate the activations at
            curves : dict[str, nn.Module]
                legend label mapped to the activation to draw
            title : str
                title for the function plot
            dashed : set[str] | None
                labels to draw dashed, used for reference curves
        """
        dashed = dashed or set()
        fig, (ax_f, ax_g) = plt.subplots(1, 2, figsize=(14, 4.5))
        for name, activation in curves.items():
            y, dy = evaluate(activation, x)
            style = "--" if name in dashed else "-"
            ax_f.plot(x, y, style, label=name)
            ax_g.plot(x, dy, style, label=name)
        for ax, label in ((ax_f, "f(x)"), (ax_g, "f'(x)")):
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.axvline(0, color="grey", linewidth=0.5)
            ax.set_xlabel("x")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            ax.legend()
        ax_f.set_title(title)
        ax_g.set_title("derivative (from autograd)")
        fig.tight_layout()
        return fig

    return evaluate, plot_curves


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sigmoid and Tanh

    These are the classic activations. [`nn.Sigmoid`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html) squashes any input into \((0, 1)\) and [`nn.Tanh`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Tanh.html) squashes it into \((-1, 1)\):

    \[
    \sigma(x) = \frac{1}{1 + e^{-x}} \qquad \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} = 2\sigma(2x) - 1
    \]

    Neither has any parameters. Their derivatives are

    \[
    \sigma'(x) = \sigma(x)\,(1 - \sigma(x)) \qquad \tanh'(x) = 1 - \tanh^2(x)
    \]

    so the largest gradient Sigmoid can ever pass back is \(0.25\) (at \(x = 0\)), and Tanh manages \(1\). Move the slider below to change the input \(z\). The red line is the tangent at \(z\); its slope is the gradient. Watch what happens once \(|z|\) gets past about 4.
    """)
    return


@app.cell
def _(mo):
    squash_choice = mo.ui.dropdown(
        options=["Sigmoid", "Tanh"], value="Sigmoid", label="activation"
    )
    squash_z = mo.ui.slider(
        start=-8, stop=8, step=0.1, value=0.0, label="input z", show_value=True
    )
    mo.hstack([squash_choice, squash_z], justify="start", gap=2)
    return squash_choice, squash_z


@app.cell
def _(evaluate, mo, nn, plt, squash_choice, squash_z, torch):
    _activation = nn.Sigmoid() if squash_choice.value == "Sigmoid" else nn.Tanh()
    _peak = 0.25 if squash_choice.value == "Sigmoid" else 1.0
    _x = torch.linspace(-8, 8, 400)
    _y, _dy = evaluate(_activation, _x)
    _fz, _dfz = evaluate(_activation, torch.tensor([squash_z.value]))
    _fz, _dfz = _fz.item(), _dfz.item()

    _fig, (_ax_f, _ax_g) = plt.subplots(1, 2, figsize=(14, 4.5))
    _ax_f.plot(_x, _y, label=squash_choice.value)
    _tx = torch.linspace(squash_z.value - 2, squash_z.value + 2, 20)
    _ax_f.plot(_tx, _fz + _dfz * (_tx - squash_z.value), "r", label="tangent at z")
    _ax_f.plot(squash_z.value, _fz, "ro")
    _ax_f.set_ylim(-1.3, 1.3)
    _ax_f.set_title(f"{squash_choice.value}(z)")
    _ax_g.plot(_x, _dy, label=f"{squash_choice.value}'")
    _ax_g.plot(squash_z.value, _dfz, "ro")
    _ax_g.set_title("derivative (from autograd)")
    for _ax, _label in ((_ax_f, "f(x)"), (_ax_g, "f'(x)")):
        _ax.set_xlabel("x")
        _ax.set_ylabel(_label)
        _ax.grid(alpha=0.3)
        _ax.legend()
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"f({squash_z.value:.1f}) = **{_fz:.5f}**, "
                f"f'({squash_z.value:.1f}) = **{_dfz:.6f}**, "
                f"which is **{100 * _dfz / _peak:.2f}%** of the largest gradient "
                f"{squash_choice.value} can produce."
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once the input is large in either direction the curve is flat and the gradient is close to zero. We say the unit has *saturated*. A saturated unit has stopped learning, as the weights feeding it receive almost no gradient however wrong the output is.

    ### Vanishing gradients through depth

    In a deep network the chain rule multiplies the local derivatives of every layer together on the way back. If each Sigmoid contributes at most 0.25, ten layers contribute at most \(0.25^{10} \approx 10^{-6}\), and the early layers barely move. This is the *vanishing gradient* problem, and it is the main reason ReLU replaced Sigmoid in hidden layers.

    The plot below builds a stack of `Linear` + activation blocks (32 units wide) for every depth from 1 to 30, pushes a random batch through, and measures the size of the gradient on the **first** layer's weights. Note the log scale; if a line stops early, the gradient has underflowed to exactly zero. The checkbox swaps PyTorch's default `Linear` initialisation for one chosen to match the activation ([`kaiming_normal_`](https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_) for ReLU and [`xavier_normal_`](https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_normal_) with the matching gain for the others), because the activation is only half the story.
    """)
    return


@app.cell
def _(nn, torch):
    def first_layer_gradient(
        activation: type[nn.Module], depth: int, matched_init: bool, width: int = 32
    ) -> float:
        """
        Builds a deep Linear + activation stack and returns the norm of the
        gradient on the first layer's weights.

        Parameters
        ----------
            activation : type[nn.Module]
                the activation class to put after each Linear
            depth : int
                number of Linear + activation blocks
            matched_init : bool
                use an initialisation matched to the activation rather than the default
            width : int
                number of units in each hidden layer
        """
        torch.manual_seed(0)
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(width, width), activation()]
        layers.append(nn.Linear(width, 1))
        model = nn.Sequential(*layers)
        if matched_init:
            name = activation.__name__.lower()
            for layer in model:
                if isinstance(layer, nn.Linear):
                    if name == "relu":
                        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    else:
                        gain = nn.init.calculate_gain(name)
                        nn.init.xavier_normal_(layer.weight, gain=gain)
                    nn.init.zeros_(layer.bias)
        x = torch.randn(64, width)
        model(x).pow(2).mean().backward()
        return model[0].weight.grad.norm().item()

    # this is cheap (about half a second) so I work out every depth up front and
    # the sliders below only choose what to highlight
    depths = list(range(1, 31))
    depth_results = {
        matched: {
            act.__name__: [first_layer_gradient(act, d, matched) for d in depths]
            for act in (nn.Sigmoid, nn.Tanh, nn.ReLU)
        }
        for matched in (False, True)
    }
    return depth_results, depths


@app.cell
def _(mo):
    depth_slider = mo.ui.slider(
        start=1, stop=30, step=1, value=10, label="depth", show_value=True
    )
    depth_matched = mo.ui.checkbox(
        value=False, label="initialisation matched to activation"
    )
    mo.hstack([depth_slider, depth_matched], justify="start", gap=2)
    return depth_matched, depth_slider


@app.cell
def _(depth_matched, depth_results, depth_slider, depths, mo, plt):
    _results = depth_results[depth_matched.value]
    _fig, _ax = plt.subplots(figsize=(10, 4.5))
    for _name, _norms in _results.items():
        # a gradient that underflows to exactly 0 can't go on a log scale, so it is left out
        _ax.semilogy(
            depths,
            [n if n > 0 else float("nan") for n in _norms],
            marker=".",
            label=_name,
        )
    _ax.axvline(depth_slider.value, color="red", linestyle="--")
    _ax.set_xlabel("number of hidden layers")
    _ax.set_ylabel("first layer gradient norm (log scale)")
    _ax.set_title("How much gradient reaches the first layer?")
    _ax.grid(alpha=0.3, which="both")
    _ax.legend()
    _fig.tight_layout()

    _rows = "\n".join(
        f"| {_name} | {_norms[depth_slider.value - 1]:.3e} |"
        for _name, _norms in _results.items()
    )
    mo.hstack(
        [
            _fig,
            mo.md(
                f"At depth **{depth_slider.value}**:\n\n"
                "| Activation | Gradient norm |\n| --- | --- |\n" + _rows
            ),
        ],
        widths=[3, 1],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the default initialisation all three shrink, ReLU included. PyTorch's default `Linear` initialisation is not tuned for any particular activation, so it is worth looking at the checkbox before blaming the activation. With a matched initialisation Tanh and ReLU hold their gradient through all 30 layers. Sigmoid still collapses: no choice of weight scale fixes a derivative that never gets above 0.25 and an output that is never centred on zero.

    ### In machine learning

    Sigmoid is still used, just not in hidden layers. It is the right choice for an output that is a probability of a yes / no answer, and it appears inside LSTM and GRU gates (see the RNN notebooks), where the 0 to 1 range acts as a "how much to let through" switch. Tanh is used inside the same recurrent cells for the cell values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ReLU and its relatives

    [`nn.ReLU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html) is simply \(\max(0, x)\). Its derivative is 1 for positive inputs and 0 for negative ones (PyTorch uses 0 at exactly \(x = 0\)), so a positive signal passes back through any number of ReLUs without shrinking. It is also very cheap to compute. The relatives change what happens on the negative side.

    | Module | Parameters used here | What it does |
    | --- | --- | --- |
    | [`ReLU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html) | | \(\max(0, x)\) |
    | [`LeakyReLU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html) | `negative_slope=0.01` | \(x\) for positive inputs, `negative_slope * x` otherwise, so the gradient is never exactly zero. |
    | [`ReLU6`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU6.html) | | \(\min(\max(0, x), 6)\). Bounded output, used in MobileNet and other networks aimed at low-precision hardware. |
    | [`PReLU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.PReLU.html) | `num_parameters=1`, `init=0.25` | A LeakyReLU where the slope is a learnable parameter. |
    | [`RReLU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RReLU.html) | `lower=1/8`, `upper=1/3` | A LeakyReLU whose slope is picked at random during training and fixed to the average in evaluation. |

    Change the negative slope and watch the left half of each plot. PReLU is created with the same starting slope so it sits on top of LeakyReLU; the difference is that it can change during training.
    """)
    return


@app.cell
def _(mo):
    leaky_slope = mo.ui.slider(
        start=0.0,
        stop=0.5,
        step=0.01,
        value=0.1,
        label="negative slope",
        show_value=True,
    )
    leaky_slope
    return (leaky_slope,)


@app.cell
def _(leaky_slope, nn, plot_curves, torch):
    plot_curves(
        torch.linspace(-8, 8, 801),
        {
            "ReLU": nn.ReLU(),
            f"LeakyReLU({leaky_slope.value:.2f})": nn.LeakyReLU(leaky_slope.value),
            f"PReLU(init={leaky_slope.value:.2f})": nn.PReLU(init=leaky_slope.value),
            "ReLU6": nn.ReLU6(),
        },
        "ReLU family",
        dashed={f"PReLU(init={leaky_slope.value:.2f})"},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dead ReLUs

    The flat left half of ReLU has a cost. A unit computes \(z = wx + b\) and then \(\text{ReLU}(z)\). If \(z\) is negative for every sample, the derivative is zero for every sample, so \(w\) and \(b\) receive no gradient at all and can never move back. The unit is *dead*. This can happen after a large update pushes the bias too far negative, and it is one reason a network with too high a learning rate can quietly lose capacity.

    The demo below feeds 1000 samples from a standard normal through a single unit. Pull the bias down and watch the red part of the histogram grow. The table shows the gradients a loss of \(\text{mean}((f(z) - 1)^2)\) produces on \(w\) and \(b\); once every sample is red, ReLU's gradients become exactly zero whilst LeakyReLU (using the slope slider above) keeps a small one.
    """)
    return


@app.cell
def _(mo):
    dead_weight = mo.ui.slider(
        start=-3, stop=3, step=0.1, value=1.0, label="weight w", show_value=True
    )
    dead_bias = mo.ui.slider(
        start=-5, stop=2, step=0.1, value=0.0, label="bias b", show_value=True
    )
    mo.hstack([dead_weight, dead_bias], justify="start", gap=2)
    return dead_bias, dead_weight


@app.cell
def _(dead_bias, dead_weight, leaky_slope, mo, nn, plt, torch):
    _gen = torch.Generator().manual_seed(1)
    _x = torch.randn(1000, generator=_gen)

    def _unit_gradients(activation: nn.Module) -> tuple[float, float, float]:
        w = torch.tensor(dead_weight.value, requires_grad=True)
        b = torch.tensor(dead_bias.value, requires_grad=True)
        z = w * _x + b
        out = activation(z)
        loss = (out - 1.0).pow(2).mean()
        loss.backward()
        return w.grad.item(), b.grad.item(), (z <= 0).float().mean().item()

    _relu = _unit_gradients(nn.ReLU())
    _leaky = _unit_gradients(nn.LeakyReLU(leaky_slope.value))

    _z = dead_weight.value * _x + dead_bias.value
    _fig, _ax = plt.subplots(figsize=(10, 4))
    _bins = torch.linspace(-9, 9, 73)
    _ax.hist(_z[_z <= 0], bins=_bins, color="tab:red", label="z <= 0 (ReLU gradient 0)")
    _ax.hist(_z[_z > 0], bins=_bins, color="tab:green", label="z > 0 (ReLU gradient 1)")
    _ax.set_xlabel("pre-activation z = w x + b")
    _ax.set_ylabel("samples")
    _ax.legend()
    _ax.grid(alpha=0.3)
    _fig.tight_layout()

    mo.hstack(
        [
            _fig,
            mo.md(
                f"**{100 * _relu[2]:.1f}%** of samples are in the flat region.\n\n"
                "| | dL/dw | dL/db |\n| --- | --- | --- |\n"
                f"| ReLU | {_relu[0]:.6f} | {_relu[1]:.6f} |\n"
                f"| LeakyReLU({leaky_slope.value:.2f}) | {_leaky[0]:.6f} | {_leaky[1]:.6f} |"
            ),
        ],
        widths=[3, 2],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PReLU learns its slope

    `PReLU` stores its slope in a `weight` parameter, so it shows up in `parameters()` and the optimiser updates it like any other weight. Here I give it inputs where the target is half the input on the negative side and let SGD find the slope. With `num_parameters` set to the number of channels each channel gets its own slope.
    """)
    return


@app.cell
def _(nn, torch):
    prelu = nn.PReLU()
    print("parameters :", [(n, p.data) for n, p in prelu.named_parameters()])

    _x = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    _target = torch.where(_x < 0, 0.5 * _x, _x)
    _optimiser = torch.optim.SGD(prelu.parameters(), lr=0.1)
    for _step in range(50):
        _optimiser.zero_grad()
        _loss = (prelu(_x) - _target).pow(2).mean()
        _loss.backward()
        _optimiser.step()
        if _step % 10 == 0:
            print(
                f"step {_step:2d}  slope {prelu.weight.item():.4f}  loss {_loss.item():.6f}"
            )
    print(f"final slope {prelu.weight.item():.4f}")

    print("per channel:", nn.PReLU(num_parameters=3).weight.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RReLU in training and evaluation

    `RReLU` picks a random slope from \(U(\text{lower}, \text{upper})\) for every element whilst the module is in training mode, which acts as a mild regulariser in the same spirit as dropout (Part 4). In evaluation mode the slope is fixed at \((\text{lower} + \text{upper}) / 2\). Toggle the switch and change the range to see the scatter collapse onto a line.
    """)
    return


@app.cell
def _(mo):
    rrelu_training = mo.ui.switch(value=True, label="training mode")
    rrelu_range = mo.ui.range_slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=[0.125, 0.33],
        label="slope range",
        show_value=True,
    )
    mo.hstack([rrelu_training, rrelu_range], justify="start", gap=2)
    return rrelu_range, rrelu_training


@app.cell
def _(nn, plt, rrelu_range, rrelu_training, torch):
    _lower, _upper = rrelu_range.value
    _rrelu = nn.RReLU(lower=_lower, upper=_upper)
    _rrelu.train(rrelu_training.value)
    _x = torch.linspace(-5, 5, 400)
    with torch.no_grad():
        _y = _rrelu(_x)

    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.scatter(_x, _y, s=6, label="RReLU output")
    _ax.plot(
        _x, torch.where(_x < 0, _lower * _x, _x), "--", label=f"slope {_lower:.2f}"
    )
    _ax.plot(
        _x, torch.where(_x < 0, _upper * _x, _x), "--", label=f"slope {_upper:.2f}"
    )
    _ax.set_title(
        f"RReLU in {'training' if rrelu_training.value else 'evaluation'} mode"
    )
    _ax.set_xlabel("x")
    _ax.set_ylabel("f(x)")
    _ax.grid(alpha=0.3)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Smooth alternatives

    ReLU has a sharp corner at zero and a completely flat negative side. A family of newer activations keeps the ReLU-like shape for positive inputs but smooths the corner and lets a little negative signal through. Most modern architectures use one of these.

    | Module | Parameters used here | What it does |
    | --- | --- | --- |
    | [`ELU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ELU.html) | `alpha=1.0` | \(x\) for positive inputs, \(\alpha(e^{x} - 1)\) otherwise, so it levels off at \(-\alpha\). |
    | [`CELU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CELU.html) | `alpha=1.0` | \(\alpha(e^{x/\alpha} - 1)\) on the negative side. The same as ELU when \(\alpha = 1\), but its derivative stays continuous for any \(\alpha\). |
    | [`SELU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.SELU.html) | | A scaled ELU with fixed constants \(\lambda \approx 1.0507\), \(\alpha \approx 1.6733\), chosen so activations keep zero mean and unit variance through deep networks (with the right initialisation). |
    | [`GELU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html) | `approximate="none"` | \(x\,\Phi(x)\) where \(\Phi\) is the normal CDF. Used in BERT, GPT and vision transformers. `approximate="tanh"` uses a faster formula. |
    | [`SiLU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html) | | \(x\,\sigma(x)\), also called Swish. Used in EfficientNet and many diffusion U-Nets. |
    | [`Mish`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Mish.html) | | \(x \tanh(\text{softplus}(x))\). A similar shape to SiLU, with a slightly deeper dip. |
    | [`Softplus`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softplus.html) | `beta=1.0`, `threshold=20.0` | \(\frac{1}{\beta}\log(1 + e^{\beta x})\), a smooth ReLU that is always positive. Above `beta * x > threshold` it returns \(x\) to avoid overflow. |

    Pick which curves to overlay and adjust the parameters. Things to look for: GELU, SiLU and Mish dip slightly below zero before rising (they are not monotonic), and Softplus gets closer and closer to ReLU as \(\beta\) increases. Look at the derivative plot too; these functions have no flat region, so there is always some gradient.
    """)
    return


@app.cell
def _(mo):
    smooth_pick = mo.ui.multiselect(
        options=[
            "ReLU",
            "ELU",
            "CELU",
            "SELU",
            "GELU",
            "GELU (tanh)",
            "SiLU",
            "Mish",
            "Softplus",
        ],
        value=["ReLU", "ELU", "GELU", "SiLU", "Softplus"],
        label="activations",
    )
    smooth_alpha = mo.ui.slider(
        start=0.1,
        stop=3.0,
        step=0.1,
        value=1.0,
        label="ELU / CELU alpha",
        show_value=True,
    )
    softplus_beta = mo.ui.slider(
        start=0.25,
        stop=10.0,
        step=0.25,
        value=1.0,
        label="Softplus beta",
        show_value=True,
    )
    smooth_range = mo.ui.range_slider(
        start=-10, stop=10, step=0.5, value=[-5, 5], label="x range", show_value=True
    )
    mo.vstack(
        [
            smooth_pick,
            mo.hstack(
                [smooth_alpha, softplus_beta, smooth_range], justify="start", gap=2
            ),
        ]
    )
    return smooth_alpha, smooth_pick, smooth_range, softplus_beta


@app.cell
def _(
    mo,
    nn,
    plot_curves,
    smooth_alpha,
    smooth_pick,
    smooth_range,
    softplus_beta,
    torch,
):
    _available = {
        "ReLU": nn.ReLU(),
        "ELU": nn.ELU(alpha=smooth_alpha.value),
        "CELU": nn.CELU(alpha=smooth_alpha.value),
        "SELU": nn.SELU(),
        "GELU": nn.GELU(),
        "GELU (tanh)": nn.GELU(approximate="tanh"),
        "SiLU": nn.SiLU(),
        "Mish": nn.Mish(),
        "Softplus": nn.Softplus(beta=softplus_beta.value),
    }
    _chosen = {name: _available[name] for name in smooth_pick.value}
    if _chosen:
        _output = plot_curves(
            torch.linspace(*smooth_range.value, 801),
            _chosen,
            "Smooth activations",
            dashed={"ReLU"},
        )
    else:
        _output = mo.md("Pick at least one activation to plot.")
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two GELU curves are almost indistinguishable. The printout below measures the largest difference between them, and between SiLU and Mish (which are not as close as they look), and also checks the SELU constants by pushing a large negative and a positive value through.
    """)
    return


@app.cell
def _(F, torch):
    _x = torch.linspace(-6, 6, 2001)
    print(
        f"max |GELU - GELU(tanh)| = {(F.gelu(_x) - F.gelu(_x, approximate='tanh')).abs().max():.2e}"
    )
    print(f"max |SiLU - Mish|       = {(F.silu(_x) - F.mish(_x)).abs().max():.4f}")
    print(f"SELU(1)    = {F.selu(torch.tensor(1.0)):.4f}  (lambda)")
    print(f"SELU(-100) = {F.selu(torch.tensor(-100.0)):.4f}  (-lambda * alpha)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Clamping and shrinking

    The remaining element-wise activations fall into two groups.

    The **clamping** functions are cheap piecewise-linear versions of the smooth ones. They avoid `exp`, which matters on mobile and quantised hardware. [`Hardsigmoid`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html) and [`Hardswish`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Hardswish.html) were introduced for MobileNetV3 as stand-ins for Sigmoid and SiLU, and [`Hardtanh`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html) clips to `[min_val, max_val]` (ReLU6 is `Hardtanh(0, 6)`).

    The **shrinking** functions push small values to zero. [`Hardshrink`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html) zeroes anything with \(|x| \le \lambda\) and leaves the rest alone, [`Softshrink`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softshrink.html) zeroes the same range but also moves everything else \(\lambda\) towards zero, and [`Threshold`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Threshold.html) replaces anything at or below `threshold` with a fixed `value`. [`Tanhshrink`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Tanhshrink.html) is \(x - \tanh(x)\). The dashed curves are the smooth functions the hard ones approximate.
    """)
    return


@app.cell
def _(mo):
    hardtanh_range = mo.ui.range_slider(
        start=-3,
        stop=3,
        step=0.1,
        value=[-1, 1],
        label="Hardtanh min / max",
        show_value=True,
    )
    shrink_lambda = mo.ui.slider(
        start=0.0,
        stop=2.0,
        step=0.05,
        value=0.5,
        label="shrink lambda / threshold",
        show_value=True,
    )
    mo.hstack([hardtanh_range, shrink_lambda], justify="start", gap=2)
    return hardtanh_range, shrink_lambda


@app.cell
def _(hardtanh_range, mo, nn, plot_curves, shrink_lambda, torch):
    _x = torch.linspace(-5, 5, 801)
    _low, _high = hardtanh_range.value
    _lam = shrink_lambda.value
    mo.vstack(
        [
            plot_curves(
                _x,
                {
                    f"Hardtanh({_low:.1f}, {_high:.1f})": nn.Hardtanh(_low, _high),
                    "Hardsigmoid": nn.Hardsigmoid(),
                    "Sigmoid": nn.Sigmoid(),
                    "Hardswish": nn.Hardswish(),
                    "SiLU": nn.SiLU(),
                },
                "Clamping",
                dashed={"Sigmoid", "SiLU"},
            ),
            plot_curves(
                _x,
                {
                    f"Hardshrink({_lam:.2f})": nn.Hardshrink(_lam),
                    f"Softshrink({_lam:.2f})": nn.Softshrink(_lam),
                    f"Threshold({_lam:.2f}, 0)": nn.Threshold(_lam, 0.0),
                    "Tanhshrink": nn.Tanhshrink(),
                },
                "Shrinking",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### In media: shrinkage as denoising

    The shrink functions are rarely used as hidden-layer activations, but soft thresholding is a standard denoising tool: it is the classic way of cleaning up wavelet coefficients in image and audio denoising, and it appears in "unrolled" networks that learn \(\lambda\). It works when the signal we care about is a few large values and the noise is many small ones.

    Below is a generated signal of a few spikes with Gaussian noise added. The same \(\lambda\) slider as above controls both shrink functions. Too small and the noise survives; too large and the spikes are removed too. Softshrink also reduces the height of the spikes it keeps, which is why its error behaves differently from Hardshrink as you increase \(\lambda\).
    """)
    return


@app.cell
def _(mo, nn, plt, shrink_lambda, torch):
    _gen = torch.Generator().manual_seed(3)
    _clean = torch.zeros(200)
    _positions = torch.randperm(200, generator=_gen)[:10]
    _clean[_positions] = (torch.rand(10, generator=_gen) * 2 + 1) * torch.sign(
        torch.randn(10, generator=_gen)
    )
    _noisy = _clean + 0.3 * torch.randn(200, generator=_gen)

    _lam = shrink_lambda.value
    with torch.no_grad():
        _results = {
            "noisy": _noisy,
            f"Hardshrink({_lam:.2f})": nn.Hardshrink(_lam)(_noisy),
            f"Softshrink({_lam:.2f})": nn.Softshrink(_lam)(_noisy),
        }

    _fig, _axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True, sharey=True)
    _errors = []
    for _ax, (_name, _signal) in zip(_axes, _results.items()):
        _error = (_signal - _clean).pow(2).mean().item()
        _errors.append(f"| {_name} | {_error:.4f} |")
        _ax.plot(_clean, color="lightgrey", linewidth=4, label="clean")
        _ax.plot(_signal, linewidth=1, label=_name)
        _ax.legend(loc="upper right")
        _ax.grid(alpha=0.3)
    _axes[-1].set_xlabel("sample")
    _fig.tight_layout()

    _table = "\n".join(_errors)
    mo.hstack(
        [
            _fig,
            mo.md(
                "Mean squared error against the clean signal:\n\n"
                "| Signal | MSE |\n| --- | --- |\n" + _table
            ),
        ],
        widths=[3, 1],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Softmax and functions across a dimension

    Everything so far works on each element on its own. [`nn.Softmax`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html) is different: it turns a vector of logits into probabilities that sum to 1, so each output depends on **every** input along the chosen dimension.

    \[
    \text{softmax}(z)_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
    \]

    PyTorch's `Softmax` has no temperature \(T\) (it is always 1), but dividing the logits by \(T\) before the softmax is common when sampling from generative models, so I have added a slider for it. Move the logits and the temperature. Low temperatures push all the probability onto the largest logit; high temperatures spread it out towards uniform. Adding the same amount to every logit changes nothing, as only the differences matter.
    """)
    return


@app.cell
def _(mo):
    class_names = ["cat", "dog", "bird", "fish", "frog"]
    logit_sliders = mo.ui.array(
        [
            mo.ui.slider(
                start=-5, stop=5, step=0.1, value=v, label=name, show_value=True
            )
            for name, v in zip(class_names, [2.0, 1.0, 0.5, -1.0, -2.0])
        ],
        label="logits",
    )
    temperature = mo.ui.slider(
        start=0.1, stop=5.0, step=0.1, value=1.0, label="temperature T", show_value=True
    )
    mo.hstack([logit_sliders, temperature], justify="start", gap=4)
    return class_names, logit_sliders, temperature


@app.cell
def _(class_names, logit_sliders, mo, nn, plt, temperature, torch):
    _logits = torch.tensor(logit_sliders.value) / temperature.value
    _probs = nn.Softmax(dim=0)(_logits)
    _log_probs = nn.LogSoftmax(dim=0)(_logits)

    _fig, (_ax_l, _ax_p) = plt.subplots(1, 2, figsize=(12, 4))
    _ax_l.bar(class_names, torch.tensor(logit_sliders.value), color="tab:grey")
    _ax_l.set_title("logits")
    _ax_l.axhline(0, color="black", linewidth=0.5)
    _ax_p.bar(class_names, _probs, color="tab:blue")
    _ax_p.set_ylim(0, 1)
    _ax_p.set_title(f"softmax(logits / {temperature.value:.1f})")
    for _ax in (_ax_l, _ax_p):
        _ax.grid(alpha=0.3, axis="y")
    _fig.tight_layout()

    _rows = "\n".join(
        f"| {n} | {p:.4f} | {lp:.4f} |"
        for n, p, lp in zip(class_names, _probs.tolist(), _log_probs.tolist())
    )
    mo.hstack(
        [
            _fig,
            mo.md(
                "| Class | Softmax | LogSoftmax |\n| --- | --- | --- |\n"
                + _rows
                + f"\n\nSum of probabilities: **{_probs.sum():.6f}**"
            ),
        ],
        widths=[3, 2],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Choosing the dimension

    With a batch the logits are usually shaped `(batch, classes)`, and we want each **row** to sum to 1, so `dim=1` (or `dim=-1`). Using `dim=0` is a silent mistake: it produces a perfectly valid-looking tensor, but normalises each class across the batch instead. Leaving `dim` out still works for now but gives a deprecation warning, so I always pass it.
    """)
    return


@app.cell
def _(nn, torch):
    import warnings

    batch_logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
    print("shape          :", tuple(batch_logits.shape))
    print("dim=1 row sums :", nn.Softmax(dim=1)(batch_logits).sum(dim=1))
    print(
        "dim=0 row sums :",
        nn.Softmax(dim=0)(batch_logits).sum(dim=1),
        "<- wrong, but no error",
    )
    print("dim=0 col sums :", nn.Softmax(dim=0)(batch_logits).sum(dim=0))

    with warnings.catch_warnings(record=True) as _caught:
        warnings.simplefilter("always")
        nn.Softmax()(batch_logits)
    for _w in _caught:
        print("warning        :", _w.message)
    return (batch_logits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why not just write it ourselves?

    Writing softmax straight from the formula works for small logits, but \(e^{1000}\) overflows a 32-bit float to `inf`, and `inf / inf` is `nan`. PyTorch subtracts the largest logit before taking the exponential, which gives the same answer (only differences matter) without overflowing. The same applies to `LogSoftmax`, which is more accurate than `torch.log(softmax(x))` for very small probabilities.
    """)
    return


@app.cell
def _(torch):
    _big = torch.tensor([1000.0, 1001.0, 1002.0])
    _naive = torch.exp(_big) / torch.exp(_big).sum()
    print("naive softmax  :", _naive)
    print("torch.softmax  :", torch.softmax(_big, dim=0))

    _spread = torch.tensor([0.0, 120.0])
    print("log(softmax)   :", torch.log(torch.softmax(_spread, dim=0)))
    print("log_softmax    :", torch.log_softmax(_spread, dim=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### In machine learning

    In Part 3 we saw that `nn.CrossEntropyLoss` expects raw logits. That is because it applies `LogSoftmax` and then [`NLLLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html) internally, using the stable version above. So we do **not** put a `Softmax` at the end of a classifier we train with `CrossEntropyLoss`; we only apply it afterwards when we want probabilities to show. The check below confirms the two routes give the same loss.

    The rest of this family, briefly:

    - [`Softmin`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmin.html) is `Softmax` of the negated input, so the smallest value gets the most weight.
    - [`Softmax2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax2d.html) applies softmax across the channels of every pixel of an `(N, C, H, W)` image, which is what we want for per-pixel segmentation.
    - [`LogSigmoid`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LogSigmoid.html) is the stable \(\log \sigma(x)\), the two-class counterpart of `LogSoftmax`.
    - [`GLU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GLU.html) splits its input in half along `dim` and returns \(a \otimes \sigma(b)\): one half gated by the other. It needs an even size along that dimension.
    """)
    return


@app.cell
def _(batch_logits, nn, torch):
    _targets = torch.tensor([0, 1])
    print(
        "CrossEntropyLoss          :",
        nn.CrossEntropyLoss()(batch_logits, _targets).item(),
    )
    print(
        "NLLLoss(LogSoftmax(x))    :",
        nn.NLLLoss()(nn.LogSoftmax(dim=1)(batch_logits), _targets).item(),
    )

    print("Softmin                   :", nn.Softmin(dim=1)(batch_logits[0:1]))

    _image = torch.randn(1, 4, 2, 3)
    _per_pixel = nn.Softmax2d()(_image)
    print("Softmax2d shape           :", tuple(_per_pixel.shape))
    print("sum over channels         :\n", _per_pixel.sum(dim=1))

    _features = torch.randn(2, 6)
    print("GLU (2, 6) ->", tuple(nn.GLU(dim=1)(_features).shape))
    try:
        nn.GLU(dim=1)(torch.randn(2, 5))
    except RuntimeError as error:
        print("GLU (2, 5) -> RuntimeError:", error)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modules, functions and `inplace`

    Every activation is available three ways: as a module in `torch.nn`, as a function in [`torch.nn.functional`](https://docs.pytorch.org/docs/stable/nn.functional.html), and for the most common ones as a tensor function such as `torch.relu`. They compute the same thing. I use the module inside `nn.Sequential` and the functional form inside a `forward` method when there are no parameters to store. `PReLU` is the exception, as it has a parameter, so it needs to be a module.

    Several modules take `inplace=True`, which overwrites the input rather than allocating a new tensor. This saves memory but can break autograd. Some operations, such as `sigmoid`, save their output to compute the gradient later; if a following in-place ReLU overwrites it, backward fails. In-place operations are also not allowed on a leaf tensor that requires a gradient.
    """)
    return


@app.cell
def _(F, nn, torch):
    _x = torch.randn(5)
    print(
        "all equal:",
        torch.equal(nn.ReLU()(_x), F.relu(_x))
        and torch.equal(F.relu(_x), torch.relu(_x)),
    )

    try:
        _a = torch.randn(4, requires_grad=True)
        _out = nn.ReLU(inplace=True)(torch.sigmoid(_a))
        _out.sum().backward()
    except RuntimeError as error:
        print("\nsigmoid then in-place ReLU:\n ", str(error).split(". Hint")[0])

    try:
        _leaf = torch.randn(4, requires_grad=True)
        nn.ReLU(inplace=True)(_leaf)
    except RuntimeError as error:
        print("\nin-place ReLU on a leaf:\n ", error)

    _b = torch.randn(4, requires_grad=True)
    _ok = nn.ReLU(inplace=True)(_b * 2)
    _ok.sum().backward()
    print("\nafter a multiply it works, gradient:", _b.grad)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The last case works because `_b * 2` creates a new intermediate tensor and the multiply does not need its own output to compute the gradient. Whether an in-place activation is safe depends on what is either side of it, which is why I leave `inplace` at its default unless memory is actually a problem.

    ## In machine learning: what the activation does to a fit

    To finish, let's see the effect on a real model. The network below has one hidden layer, `Linear(1, hidden) -> activation -> Linear(hidden, 1)`, and is trained with Adam for 1000 steps to fit \(\sin(2x)\). The left plot shows the fit in blue and, faintly, each hidden unit's contribution to the output (its activation multiplied by its output weight). The prediction is the sum of those curves plus a bias.

    Try `Identity` first: with no activation the network is linear and the best it can do is a straight line (Part 3 again). Then compare ReLU, which builds the curve out of straight-line hinges, with Tanh or SiLU, which build it out of smooth bumps. With this setup ReLU does noticeably worse than the smooth activations, and the main reason is the dead unit problem from earlier: units whose hinge ends up outside the data contribute a flat line (or nothing at all) and stop learning. The count is shown under the plot; compare it with LeakyReLU. Training takes about half a second, so the hidden units slider only updates when you let go.
    """)
    return


@app.cell
def _(mo):
    fit_activation = mo.ui.dropdown(
        options=["Identity", "ReLU", "LeakyReLU", "Tanh", "Sigmoid", "GELU", "SiLU"],
        value="ReLU",
        label="activation",
    )
    fit_hidden = mo.ui.slider(
        start=1,
        stop=32,
        step=1,
        value=8,
        label="hidden units",
        show_value=True,
        debounce=True,
    )
    mo.hstack([fit_activation, fit_hidden], justify="start", gap=2)
    return fit_activation, fit_hidden


@app.cell
def _(F, nn, torch):
    fit_x = torch.linspace(-3, 3, 200).unsqueeze(1)
    fit_y = torch.sin(2 * fit_x)

    def train_curve(
        activation_name: str, hidden: int, steps: int = 1000
    ) -> tuple[nn.Sequential, list[float]]:
        """
        Trains a one hidden layer network to fit sin(2x).

        Parameters
        ----------
            activation_name : str
                name of the torch.nn activation class to use, e.g. "ReLU"
            hidden : int
                number of hidden units
            steps : int
                number of Adam steps

        Returns
        -------
            tuple[nn.Sequential, list[float]]
                the trained model and the loss at every step
        """
        # the same seed every time so a change in the plot is down to the activation
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Linear(1, hidden), getattr(nn, activation_name)(), nn.Linear(hidden, 1)
        )
        optimiser = torch.optim.Adam(model.parameters(), lr=0.03)
        losses = []
        for _ in range(steps):
            optimiser.zero_grad()
            loss = F.mse_loss(model(fit_x), fit_y)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        return model, losses

    return fit_x, fit_y, train_curve


@app.cell
def _(fit_activation, fit_hidden, train_curve):
    fit_model, fit_losses = train_curve(fit_activation.value, fit_hidden.value)
    return fit_losses, fit_model


@app.cell
def _(fit_activation, fit_losses, fit_model, fit_x, fit_y, mo, plt, torch):
    with torch.no_grad():
        _hidden = fit_model[1](fit_model[0](fit_x))
        # each unit's contribution is its activation times its output weight
        _contributions = _hidden * fit_model[2].weight[0]
        _prediction = fit_model(fit_x)
    _dead = int((_hidden.abs().amax(dim=0) == 0).sum())

    _fig, (_ax_fit, _ax_loss) = plt.subplots(
        1, 2, figsize=(14, 4.5), width_ratios=[2, 1]
    )
    _ax_fit.plot(fit_x, _contributions, color="tab:orange", alpha=0.35, linewidth=1)
    _ax_fit.plot(fit_x, fit_y, "k--", label="target sin(2x)")
    _ax_fit.plot(fit_x, _prediction, "tab:blue", linewidth=2, label="prediction")
    _ax_fit.set_ylim(-2.5, 2.5)
    _ax_fit.set_xlabel("x")
    _ax_fit.set_ylabel("y")
    _ax_fit.set_title(
        f"{fit_activation.value}: fit and hidden unit contributions (orange)"
    )
    _ax_fit.grid(alpha=0.3)
    _ax_fit.legend()
    _ax_loss.semilogy(fit_losses)
    _ax_loss.set_xlabel("step")
    _ax_loss.set_ylabel("MSE (log scale)")
    _ax_loss.set_title("training loss")
    _ax_loss.grid(alpha=0.3, which="both")
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.md(
                f"Final loss **{fit_losses[-1]:.5f}**, "
                f"dead hidden units (output zero for every input): **{_dead} / {_hidden.shape[1]}**"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quick reference

    This last plot lets you look at any of the element-wise activations with their default settings. `RReLU` is shown in evaluation mode so it is not random.
    """)
    return


@app.cell
def _(mo, nn):
    reference_modules = {
        name: getattr(nn, name)()
        for name in [
            "CELU",
            "ELU",
            "GELU",
            "Hardshrink",
            "Hardsigmoid",
            "Hardswish",
            "Hardtanh",
            "Identity",
            "LeakyReLU",
            "LogSigmoid",
            "Mish",
            "PReLU",
            "ReLU",
            "ReLU6",
            "RReLU",
            "SELU",
            "SiLU",
            "Sigmoid",
            "Softplus",
            "Softshrink",
            "Softsign",
            "Tanh",
            "Tanhshrink",
        ]
    }
    reference_modules["RReLU"].eval()
    reference_pick = mo.ui.multiselect(
        options=list(reference_modules), value=["Softsign", "Tanh"], label="activations"
    )
    reference_pick
    return reference_modules, reference_pick


@app.cell
def _(mo, plot_curves, reference_modules, reference_pick, torch):
    if reference_pick.value:
        _output = plot_curves(
            torch.linspace(-6, 6, 801),
            {name: reference_modules[name] for name in reference_pick.value},
            "Default settings",
        )
    else:
        _output = mo.md("Pick at least one activation to plot.")
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - Without an activation, stacked linear layers are just one linear layer.
    - The derivative matters as much as the function. Flat regions (saturated Sigmoid / Tanh, negative ReLU) pass back little or no gradient.
    - Sigmoid in hidden layers makes gradients vanish with depth. Use it for probability outputs and gates.
    - ReLU is the cheap default. Initialise to match it, and watch for dead units; LeakyReLU or a smooth alternative avoids the flat side.
    - GELU and SiLU are the usual choice in transformers and diffusion models.
    - Softmax works across a dimension, so always pass `dim`, and do not put it before `CrossEntropyLoss`.

    ## Exercises

    1. In the Sigmoid / Tanh section, find the input \(z\) where Sigmoid's gradient falls to 1% of its peak. Predict the value for Tanh before you move the slider.
    2. Add `nn.GELU` and `nn.SiLU` to the depth experiment. `calculate_gain` has no entry for them; what happens if you use the ReLU initialisation? Is it a good match?
    3. Using the dead ReLU demo, find a weight and bias where ReLU's gradients are exactly zero. Then set the negative slope to 0 and explain why LeakyReLU now behaves the same.
    4. In the curve fit, find the smallest number of hidden units that gets Tanh's final loss below 0.01. ReLU does not get there even with 32 units. Use the dead unit count to explain why, then try a lower learning rate (`lr=0.01`) and a different `torch.manual_seed` in `train_curve`. Which helps more?
    5. Replace `sin(2x)` with `torch.abs(fit_x)`. Which activation now needs the fewest units, and why?
    6. Write `softmax` yourself in a stable way (subtract the maximum first) and check it against `torch.softmax` on the `[1000, 1001, 1002]` example.

    The next part of the PyTorch series, Part 4, covers what to do once a model is trained: evaluation, inference and turning logits into predictions.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
