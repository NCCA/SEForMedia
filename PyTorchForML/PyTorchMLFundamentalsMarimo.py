#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Machine learning fundamentals through PyTorch

    In this notebook I look under the hood of PyTorch and see what each code element actually does.  We will start with a prediction we can calculate by hand, work backwards to its weights, then build the same process using PyTorch.

    The question is: **given a measured input, can we predict a continuous target?** Later we will change the output to a class and see what changes in the loss and its gradient.

    We will cover the model, loss, derivatives, backpropagation, gradient descent, mini-batches and generalisation. Along the way we will inspect the graph, parameters, gradients and optimiser state that PyTorch manages for us. This is a companion to [Part 1: tensors](PyTorchForMLPart1Tensors.py), [Part 2: autograd](PyTorchForMLPart2Autograd.py) and [Part 3: modules](PyTorchForMLPart3Modules.py).

    Whilst this does have maths in it you don't really need to fully understand it, this is just an attempt to contextualize how PyTorch works with respect to the maths.
    """)
    return


@app.cell
def _():
    import copy
    import matplotlib.pyplot as plt
    import marimo as mo
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    print("PyTorch:", torch.__version__)
    print("Device: CPU")
    return DataLoader, TensorDataset, copy, mo, nn, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. What does learning mean?

    In supervised learning, we train a model using examples that pair inputs with known targets. The model uses adjustable values, called parameters, to turn each input into a prediction. During training, we compare these predictions with the targets and adjust the parameters to reduce the error. The aim is to make useful predictions for examples the model has not seen before.

    We can start with a model that fits a straight line:

    $$\hat{y}=f_\theta(x)=wx+b, \qquad \theta=(w,b).$$

    Here $x$ is an input, $y$ is its target, $\hat{y}$ is a prediction, $w$ is a weight and $b$ is a bias. The hat means “predicted”. $\theta$ is shorthand for all the learnable parameters. A bias lets the line move away from the origin.

    | ML idea | Meaning | PyTorch representation |
    | --- | --- | --- |
    | Inputs | Information available when predicting | Usually floating-point tensors |
    | Targets | Answers supplied during training | Float values for regression; often integer IDs for classes |
    | Model | Family of functions we are willing to fit | Tensor operations or `nn.Module` |
    | Parameters | Numbers adjusted by learning | Tensors, normally `nn.Parameter` inside a module |
    | Loss | A scalar measure of prediction error | Tensor expression or a loss module |
    | Gradient | Local sensitivity of loss to each parameter | Parameter-shaped tensors in `.grad` |
    | Optimiser | Rule for updating parameters | For example `torch.optim.SGD` |
    | Hyperparameters | Choices we make about training | Learning rate, batch size, architecture |

    I use `inputs` and `targets` in the code. Textbooks often call these $X$ and $y$; the maths is the same.
    """)
    return


@app.cell
def _(torch):
    example_inputs = torch.tensor([[1.0], [2.0], [3.0]])
    example_targets = torch.tensor([[3.0], [5.0], [7.0]])
    _example_predictions = 0.5 * example_inputs + 0.0
    print("input, target, prediction")
    print(torch.cat([example_inputs, example_targets, _example_predictions], dim=1))
    print("Shapes:", example_inputs.shape, example_targets.shape)
    return example_inputs, example_targets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Loss function

    The loss function us used to convert how "wrong" we are into a quantifiable number.

    For [regression](https://en.wikipedia.org/wiki/Regression_analysis), mean squared error (MSE) is a common starting point:

    $$L(\theta)=\frac{1}{B}\sum_{i=1}^{B}(\hat{y}_i-y_i)^2.$$

    In this equation $B$ is the number of examples in this batch. Squaring prevents positive and negative errors cancelling and gives larger errors more influence. The mean makes the scale easier to compare across batch sizes. With several output values, PyTorch's default MSE averages over **all output elements**, not just the batch axis.

    MSE also has a statistical interpretation. If we assume the targets differ from the model’s predictions by independent, normally distributed noise with the same fixed variance, minimising MSE gives the maximum likelihood fit: the parameter values that make the observed targets most likely under those assumptions.

    We make those assumptions; PyTorch does not check them. Squaring the errors also gives outliers a large influence, so MSE may be a poor choice when the data contains unusually large errors.

    We need to check the shapes before calculating the loss. Below, predictions and targets both have shape (B,1), so each prediction is compared with its corresponding target.

    If the targets have shape (B,), broadcasting produces a (B,B) array of errors instead. This compares every prediction with every target. The calculation still runs and returns a loss, so the mistake can be easy to miss.
    """)
    return


@app.cell
def _(example_inputs, example_targets, nn, torch):
    _predictions = 0.5 * example_inputs
    _errors = _predictions - example_targets
    _explicit_loss = (_errors**2).mean()
    _library_loss = nn.MSELoss()(_predictions, example_targets)
    print("Errors:", _errors.flatten().tolist())
    print("Explicit MSE:", _explicit_loss.item())
    print("nn.MSELoss:", _library_loss.item())
    print(
        "Accidental broadcast shape:",
        (_predictions - example_targets[:, 0]).shape,
    )
    torch.testing.assert_close(_explicit_loss, _library_loss)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. A derivative measures local change

    The dirivitave is what we use to change the values. For $L(w)$, the derivative $dL/dw$ tells us how rapidly loss changes as we change $w$. Near the current value,

    $$L(w+\epsilon)\approx L(w)+\epsilon\frac{dL}{dw}.$$

    A positive derivative suggests decreasing $w$ will reduce loss for a small enough step. A negative derivative suggests increasing it. With many parameters we collect the partial derivatives into the **gradient**, $\nabla_\theta L$. Each partial derivative varies one parameter whilst holding the others fixed.

    This is local information. A zero gradient need not mean we have found a useful global minimum. It could be a local minimum, a saddle point or a flat region.

    Change the weight below. Watch the tangent and the sign of the derivative for $L(w)=(2w-4)^2$.
    """)
    return


@app.cell
def _(mo):
    weight_control = mo.ui.slider(
        -1.0, 4.0, step=0.1, value=0.5, label="Weight w", show_value=True
    )
    weight_control
    return (weight_control,)


@app.cell
def _(plt, torch, weight_control):
    _weight = weight_control.value
    _grid = torch.linspace(-1, 4, 151)
    _loss_here = (2 * _weight - 4) ** 2
    _slope = 4 * (2 * _weight - 4)
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(_grid, (2 * _grid - 4) ** 2, label="Loss")
    _ax.plot(
        _grid,
        _loss_here + _slope * (_grid - _weight),
        "--",
        label="Local tangent",
    )
    _ax.scatter([_weight], [_loss_here], color="black")
    _ax.set(
        xlabel="Weight w",
        ylabel="Squared error",
        ylim=(-5, 40),
        title=f"At w={_weight:.1f}, dL/dw={_slope:.1f}",
    )
    _ax.legend()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Backpropagation for one neuron, by hand

    Take one input $x=2$, target $y=4$, weight $w=0.5$ and bias $b=0$. Break the forward calculation into operations:

    $$a=wx=1, \quad \hat{y}=a+b=1, \quad e=\hat{y}-y=-3, \quad L=e^2=9.$$

    The **chain rule** combines local derivatives by multiplication:

    $$\frac{\partial L}{\partial w}
    =\frac{\partial L}{\partial e}
     \frac{\partial e}{\partial\hat{y}}
     \frac{\partial\hat{y}}{\partial a}
     \frac{\partial a}{\partial w}
    =(2e)(1)(1)(x)=-12.$$

    The bias has a shorter path:

    $$\frac{\partial L}{\partial b}=(2e)(1)(1)=-6.$$

    Backpropagation works backwards from the loss to calculate how each parameter affects it. At each step, we use the chain rule to multiply the incoming gradient by the local derivative. Where a value contributes to the loss through several paths, we add the gradient contributions from those paths. We start with \(\partial L/\partial L=1\): the derivative of the loss with respect to itself.

    PyTorch has derivative rules for the differentiable operations we use. During the forward pass, it records how those operations connect. When we call loss.backward(), it follows those connections backwards and applies the rules to calculate the gradients.
    """)
    return


@app.cell
def _(torch):
    _scalar_weight = torch.tensor(0.5, requires_grad=True)
    _scalar_bias = torch.tensor(0.0, requires_grad=True)
    _product = _scalar_weight * 2.0
    _prediction = _product + _scalar_bias
    _error = _prediction - 4.0
    _scalar_loss = _error.square()
    _prediction.retain_grad()
    _error.retain_grad()
    _scalar_loss.backward()
    print("Prediction:", _prediction.item(), "loss:", _scalar_loss.item())
    print("dL/de:", _error.grad.item())
    print("dL/d prediction:", _prediction.grad.item())
    print("dL/dw:", _scalar_weight.grad.item(), "hand calculation:", 2 * (-3) * 2)
    print("dL/db:", _scalar_bias.grad.item(), "hand calculation:", 2 * (-3))
    print("Weight after backward:", _scalar_weight.item())
    torch.testing.assert_close(_scalar_weight.grad, torch.tensor(-12.0))
    torch.testing.assert_close(_scalar_bias.grad, torch.tensor(-6.0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The computation graph

    During the forward pass, PyTorch records tracked operations in a computation graph, connecting results to the operations that produced them.

    A leaf tensor is a starting point in this graph: it was not produced by an operation that autograd recorded. Model parameters are normally leaves because we create them first, then use them to calculate predictions and loss.

    For example, if we create weight with ```requires_grad=True``` and calculate ```prediction = weight * 3```, then weight is a leaf with no ```grad_fn```, whilst prediction is a non-leaf whose ```grad_fn``` points to its backward operation.

    “Leaf” describes a tensor’s place in the graph, not its shape or whether it is learnable, ordinary inputs can also be leaves without requiring gradients.

    When we call ```loss.backward()```, PyTorch follows the graph backwards and, by default, adds gradients to the ```.grad``` attributes of participating leaves that require them. It calculates intermediate gradients too, but we need to call ```retain_grad()``` before backward if we want to inspect them afterwards.

    Some backward calculations also need values saved during the forward pass, which is one reason training needs more memory than prediction.

    After a normal backward pass, PyTorch releases those saved values, and the next forward pass builds a fresh graph. We use retain_graph=True when we deliberately need another backward pass through the same graph.

    [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) explains the recording and saved tensors. The exact names printed for backward nodes are implementation details and can change between versions.
    """)
    return


@app.cell
def _(torch):
    _leaf = torch.tensor(2.0, requires_grad=True)
    _shared = _leaf * 3
    _branched_loss = _shared.square() + _shared
    print("Leaf:", _leaf.is_leaf, "grad_fn:", _leaf.grad_fn)
    print("Shared result:", _shared.is_leaf, "grad_fn:", _shared.grad_fn)
    print("Loss grad_fn:", _branched_loss.grad_fn)
    _branched_loss.backward()
    print("Gradient from both paths:", _leaf.grad.item())
    print("By hand: (2 * 6 + 1) * 3 =", (2 * 6 + 1) * 3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. How automatic differentiation works

    We can check a derivative by evaluating the function on either side of a point:

    $$\frac{dL}{dw}\approx\frac{L(w+\epsilon)-L(w-\epsilon)}{2\epsilon}.$$

    This is a numerical approximation. It is useful for checking small examples, but choosing $\epsilon$ is awkward: large values introduce approximation error; very small values lose precision through subtraction. Checking every parameter also needs many forward evaluations.

    Reverse mode [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) instead evaluates derivative rules at the current values and composes them backwards. A scalar loss lets us calculate gradients for many parameters in one backward traversal. There will still be floating point rounding error, but there is no finite difference step size.

    Below we use double precision for the check. Normal training examples later use float32.
    """)
    return


@app.cell
def _(torch):
    _checked_weight = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    _checked_loss = (2 * _checked_weight - 4).square()
    _checked_loss.backward()
    _epsilon = 1e-5
    _plus = (2 * (_checked_weight.detach() + _epsilon) - 4).square()
    _minus = (2 * (_checked_weight.detach() - _epsilon) - 4).square()
    _finite_difference = (_plus - _minus) / (2 * _epsilon)
    print("Autograd:", _checked_weight.grad.item())
    print("Finite difference:", _finite_difference.item())
    torch.testing.assert_close(_checked_weight.grad, _finite_difference)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Gradient descent actually changes the parameters

    For plain gradient descent the update is

    $$\theta_{\mathrm{new}}=\theta-\eta\nabla_\theta L,$$

    where $\eta$ is the **learning rate**. Backpropagation calculates the gradient, the optimiser uses it. These are separate jobs.

    With our scalar example and $\eta=0.1$:

    $$w_{\mathrm{new}}=0.5-0.1(-12)=1.7,\qquad b_{\mathrm{new}}=0-0.1(-6)=0.6.$$

    The new prediction is $1.7(2)+0.6=4$, so this particular step reaches zero loss. That convenient result is specific to these numbers (and one example does not uniquely determine both a weight and a bias).

    We update under `torch.no_grad()` so the update itself is not recorded as part of the model's next differentiation problem. `.grad` is a stored tensor, not an instruction to change a weight.
    """)
    return


@app.cell
def _(torch):
    _manual_weight = torch.tensor(0.5, requires_grad=True)
    _manual_bias = torch.tensor(0.0, requires_grad=True)
    _before = (2 * _manual_weight + _manual_bias - 4).square()
    _before.backward()
    with torch.no_grad():
        _manual_weight -= 0.1 * _manual_weight.grad
        _manual_bias -= 0.1 * _manual_bias.grad
        _after = (2 * _manual_weight + _manual_bias - 4).square()
    print("Updated w, b:", _manual_weight.item(), _manual_bias.item())
    print("Loss before and after:", _before.item(), _after.item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Learning rate

    The learning rate, $\eta$, controls how far we move the parameters in the direction given by the gradient. For $L(w)=w^2$, the gradient is $2w$, so gradient descent gives

    $$w_{t+1}=w_t-\eta(2w_t)=(1-2\eta)w_t.$$

    This lets us see how the learning rate affects each update:

    - For $0<\eta<0.5$, the weight moves towards zero without crossing it.
    - At $\eta=0.5$, it reaches zero in one update.
    - For $0.5<\eta<1$, it crosses zero on each update, but gets closer each time.
    - At $\eta=1$, it alternates between two values and the loss stays the same.
    - For $\eta>1$, its magnitude grows and the loss increases.

    These ranges apply to this particular loss. In a neural network, the loss can change at different rates along different parameter directions, so a suitable learning rate depends on the problem. This is where our knowledge comes into play in setting it.

    The plot shows three learning rates from the same starting weight. We can compare how quickly the loss decreases and see what happens when the updates become too large.
    """)
    return


@app.cell
def _(plt):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    for _rate in (0.05, 0.4, 1.05):
        _weight = 2.0
        _history = [_weight**2]
        for _step in range(25):
            _weight -= _rate * 2 * _weight
            _history.append(_weight**2)
        _ax.semilogy(_history, label=f"Learning rate {_rate}")
    _ax.set(
        xlabel="Update",
        ylabel="Loss (log scale)",
        title="The same derivative, different step sizes",
    )
    _ax.legend()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Batch calculations.

    For $B$ samples, $F$ input features and $O$ outputs, write the inputs as a matrix $X$ of shape `(B,F)`. [A PyTorch linear layer](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) stores its weight $W$ as `(O,F)` and bias $b$ as `(O,)`:

    $$P=XW^T+b, \qquad E=P-Y, \qquad L=\frac{1}{BO}\sum_{i,j}E_{ij}^2.$$

    The bias broadcasts across rows. For this loss define $G=\partial L/\partial P=2E/(BO)$. Matrix backpropagation gives

    $$\frac{\partial L}{\partial W}=G^TX,\qquad
    \frac{\partial L}{\partial b}=\sum_iG_{i,:},\qquad
    \frac{\partial L}{\partial X}=GW.$$

    These are the scalar chain rule applied to all entries together. The bias gradient sums because the same bias contributes to every row. We do not divide again after forming $G$; its entries already include the mean reduction.

    We compare all three results to autograd. Inputs only need `requires_grad=True` here because we want to inspect their gradient.
    """)
    return


@app.cell
def _(torch):
    _matrix_inputs = torch.tensor([[1.0, 2.0], [3.0, -1.0]], requires_grad=True)
    _matrix_targets = torch.tensor([[1.0, 0.0], [2.0, 1.0]])
    _matrix_weight = torch.tensor([[0.5, -0.2], [0.1, 0.3]], requires_grad=True)
    _matrix_bias = torch.tensor([0.0, 0.2], requires_grad=True)
    _matrix_predictions = _matrix_inputs @ _matrix_weight.T + _matrix_bias
    _matrix_error = _matrix_predictions - _matrix_targets
    _matrix_loss = _matrix_error.square().mean()
    with torch.no_grad():
        _upstream = 2 * _matrix_error / _matrix_error.numel()
        _hand_weight = _upstream.T @ _matrix_inputs
        _hand_bias = _upstream.sum(dim=0)
        _hand_inputs = _upstream @ _matrix_weight
    _matrix_loss.backward()
    for _name, _hand, _automatic in (
        ("weight", _hand_weight, _matrix_weight.grad),
        ("bias", _hand_bias, _matrix_bias.grad),
        ("inputs", _hand_inputs, _matrix_inputs.grad),
    ):
        print(_name, "gradient:\n", _automatic)
        torch.testing.assert_close(_hand, _automatic)
    print("All matrix derivatives agree.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Hidden layer gradients

    A hidden layer calculates intermediate values that the next layer uses to make its predictions. We apply a nonlinear activation between linear layers so the network can learn nonlinear relationships. Without an activation, the layers could be combined into one linear layer.

    For a network with one hidden layer and a linear output layer, the forward calculation is

    $$Z=XW_1^T+b_1,\qquad H=\operatorname{ReLU}(Z),\qquad P=HW_2^T+b_2.$$

    Here, $X$ contains the inputs, $Z$ contains the hidden layer’s values before activation, $H$ contains its activated values and $P$ contains the predictions. The weights and biases belong to the first and second layers respectively.

    During backpropagation, we start with the prediction gradient, $G_P=\partial L/\partial P$. We pass it backwards through the output layer to calculate the gradient with respect to the hidden values:

    $$G_H=G_PW_2.$$

    Next, we pass through ReLU. ReLU returns $\max(0,z)$, so its derivative is 1 where $z>0$ and 0 where $z<0$. PyTorch also uses a derivative of 0 at zero. We therefore calculate

    $$G_Z=G_H\odot\mathbf{1}_{Z>0}.$$

    The symbol $\odot$ means elementwise multiplication. The expression $\mathbf{1}_{Z>0}$ is a mask containing 1 for positive entries in $Z$ and 0 elsewhere. It allows gradients through positive pre-activations and blocks them at the others.

    We can then calculate the first layer’s parameter gradients:

    $$\frac{\partial L}{\partial W_1}=G_Z^TX,\qquad
    \frac{\partial L}{\partial b_1}=\sum_i(G_Z)_{i,:}.$$

    PyTorch performs these calculations when we call `loss.backward()`. Each layer receives a gradient from the following operation and applies its own derivative using the chain rule.

    The example below shows how the ReLU mask affects the first layer’s gradients. In deeper networks, repeatedly multiplying derivatives can make gradients very small or very large. These behaviours are called **vanishing gradients** and **exploding gradients**. Initialisation, activation choice, normalisation and residual connections can all affect how gradients pass through the network.
    """)
    return


@app.cell
def _(torch):
    _hidden_inputs = torch.tensor([[1.0, -1.0]])
    _first_weight = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], requires_grad=True)
    _second_weight = torch.tensor([[2.0, 3.0]], requires_grad=True)
    _pre_activation = _hidden_inputs @ _first_weight.T
    _hidden_values = torch.relu(_pre_activation)
    _hidden_prediction = _hidden_values @ _second_weight.T
    _hidden_loss = (_hidden_prediction - 4.0).square().mean()
    with torch.no_grad():
        _prediction_gradient = (
            2 * (_hidden_prediction - 4.0) / _hidden_prediction.numel()
        )
        _hidden_gradient = _prediction_gradient @ _second_weight
        _masked_gradient = _hidden_gradient * (_pre_activation > 0)
        _first_hand = _masked_gradient.T @ _hidden_inputs
    _hidden_loss.backward()
    print("Pre-activation:", _pre_activation.detach())
    print("ReLU mask:", _pre_activation.detach() > 0)
    print("Gradient arriving at hidden values:", _hidden_gradient)
    print("Gradient after the mask:", _masked_gradient)
    print("First layer weight gradient:", _first_weight.grad)
    torch.testing.assert_close(_first_hand, _first_weight.grad)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Gradients for vector outputs

    When a loss contains one element, `.backward()` starts with a gradient of 1. For a vector output, each element has its own derivatives with respect to the parameters. We need to tell PyTorch how to combine their contributions into one gradient per parameter.

    We do this by passing a tensor of weights to `.backward()`. For a vector $v=f(\theta)$ and supplied weights $s$, the result is

    $$J^Ts=\nabla_\theta\sum_i s_iv_i,$$

    where $J$ is the [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant): a matrix containing the derivative of each output element with respect to each parameter. This calculation is called a **vector–Jacobian product**. PyTorch calculates the weighted contributions directly, without constructing the full Jacobian.

    For example, supplying a weight of 1 for every output gives the gradient of their **sum**. Supplying $1/B$ for each of $B$ per-example scalar losses gives the gradient of their **mean**. Calling `.sum().backward()` or `.mean().backward()` performs those combinations for us.

    The reduction therefore affects the size of the gradients. For the same $B$ losses, the gradient of the sum is $B$ times the gradient of the mean.
    """)
    return


@app.cell
def _(torch):
    _vector_parameter = torch.tensor(2.0, requires_grad=True)
    _vector_output = torch.stack([_vector_parameter.square(), 3 * _vector_parameter])
    try:
        _vector_output.backward()
    except RuntimeError as _error:
        print("Expected error:", str(_error).split("\n")[0])
    _vector_output.backward(gradient=torch.tensor([0.25, 0.75]))
    print("Weighted gradient:", _vector_parameter.grad.item())
    print("By hand: 0.25 * 4 + 0.75 * 3 =", 0.25 * 4 + 0.75 * 3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Using nn.Module

    [`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) is the base class we use to organise a PyTorch model. It keeps track of learnable parameters and any layers we assign as submodules. It also manages registered **buffers**: tensors that belong to the model’s state but are not normally updated by the optimiser. BatchNorm’s running mean and variance are examples.

    We define the model’s calculation in its `forward` method, then run it by calling `model(inputs)`. This also runs the module’s surrounding behaviour, including registered hooks (functions attached to particular stages of a module call). Calling `model.forward(inputs)` directly bypasses that surrounding behaviour. Autograd records the tracked operations performed during the calculation and uses them to calculate gradients.

    When we assign an `nn.Parameter` to a module attribute, PyTorch registers it so that `model.parameters()` can find it. Assigning an ordinary tensor does not register it as a parameter. For tensors we want to keep as model state without treating them as parameters, we can use `register_buffer()`.

    Layers such as `nn.Linear` create and initialise their own parameters. This saves us writing the setup code, although the initial values can affect training. In the example below, we use a local random-seed context to make the initialisation repeatable without changing the random state used by other cells.

    The next cell compares `nn.Linear` with the explicit calculation `inputs @ weight.T + bias`. Both produce the same predictions.



    `nn.Module` organises the model’s state and layers. Calling `model(inputs)` runs the hooks and `forward()` calculation. Autograd records the tracked operations, allowing a later backward pass to calculate gradients.
    """)
    return


@app.cell
def _(mo):
    mo.image(
        "images/nn-module.svg",
        width=600,
    )
    return


@app.cell
def _(example_inputs, nn, torch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        _layer = nn.Linear(1, 1)
    _module_output = _layer(example_inputs)
    _explicit_output = example_inputs @ _layer.weight.T + _layer.bias
    print("Registered parameters:")
    for _name, _parameter in _layer.named_parameters():
        print(
            _name,
            tuple(_parameter.shape),
            "requires_grad:",
            _parameter.requires_grad,
        )
    print("Registered buffers:", list(_layer.named_buffers()))
    print("State dictionary keys:", list(_layer.state_dict()))
    torch.testing.assert_close(_module_output, _explicit_output)
    print("Module and matrix outputs agree.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. Gradient accumulation

    Each call to `backward()` adds the calculated gradients to the values already stored in a parameter’s `.grad` attribute. Running another forward pass creates a fresh computation graph, but leaves these stored gradients in place.

    For a normal training update, we clear the previous gradients before calculating new ones:

    ```python
    optimiser.zero_grad(set_to_none=True)
    ```

    This sets `.grad` to `None` for the parameters managed by that optimiser. The next backward pass stores their newly calculated gradients.

    A gradient of `None` means no gradient is currently stored. A zero tensor means a gradient is stored, but its values are zero. This distinction matters: an optimiser normally skips parameters whose gradient is `None`, whilst a stored zero gradient can still allow an update through momentum or weight decay.

    The example below repeats the calculation $L=w^2$ with $w=3$. Each forward pass creates a fresh graph, and each backward pass calculates the same derivative, $2w=6$. Because we deliberately leave the gradients in place, `.grad` accumulates values of 6, 12 and 18. We do not call `optimiser.step()`, so the weight stays at 3 throughout.
    """)
    return


@app.cell
def _(torch):
    _accumulated_weight = torch.tensor(3.0, requires_grad=True)
    for _call in range(3):
        (_accumulated_weight.square()).backward()
        print(
            f"Backward {_call + 1}: stored gradient {_accumulated_weight.grad.item()}"
        )
    _accumulated_weight.grad = None
    (_accumulated_weight.square()).backward()
    print("After clearing and recomputing:", _accumulated_weight.grad.item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14. Mini batches

    The training objective is usually the average loss over all training examples. Rather than calculating its gradient using the entire dataset at once, we can use a smaller group of examples called a **mini batch**. A randomly sampled mini-batch gives an estimate of the full training gradient. This estimate varies between batches, so individual updates may increase or decrease the loss.

    An **update** is one optimiser step. An **epoch** is one pass through the training data. With 128 examples and a batch size of 32, one epoch contains four batches and, if we update after each batch, four optimiser steps.

    In PyTorch, `TensorDataset` pairs corresponding rows from our input and target tensors. `DataLoader` gathers those pairs into batches and can shuffle their order for each epoch. We still need to choose the data split, apply suitable normalisation and check for leakage between training and evaluation data.

    We can also accumulate gradients from several smaller **micro-batches** before making one update. To match the gradient of their combined batch, we keep the parameters unchanged and weight each micro-batch’s mean loss by its share of the total examples. For example, micro-batches containing two examples and one example need weights of $2/3$ and $1/3$.

    This gives the same gradient, apart from floating-point rounding, when each example’s calculation is independent of the others. BatchNorm in training mode depends on batch statistics, so dividing a batch changes its calculation. Dropout can also produce different random masks, meaning separate runs may give different gradients.
    """)
    return


@app.cell
def _(example_inputs, example_targets, torch):
    _full_weight = torch.tensor(0.5, requires_grad=True)
    _full_loss = (_full_weight * example_inputs - example_targets).square().mean()
    _full_loss.backward()
    _micro_weight = torch.tensor(0.5, requires_grad=True)
    for _indices in (slice(0, 2), slice(2, 3)):
        _micro_inputs = example_inputs[_indices]
        _micro_targets = example_targets[_indices]
        _micro_loss = (_micro_weight * _micro_inputs - _micro_targets).square().mean()
        _weighted_loss = _micro_loss * (len(_micro_inputs) / len(example_inputs))
        _weighted_loss.backward()
    print("Full batch gradient:", _full_weight.grad.item())
    print("Accumulated gradient:", _micro_weight.grad.item())
    torch.testing.assert_close(_full_weight.grad, _micro_weight.grad)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15. Fitting a noisy line

    We can now bring these steps together to train a model. We generate input and target pairs using

    $$y=2x+1+\epsilon,$$

    where $\epsilon$ is random noise added to each target. The model receives the training pairs and learns a weight and bias from them. It has no access to the formula we used to generate the targets.

    I split the data into 128 training examples, 32 validation examples and 32 test examples. We use the training set to calculate gradients and update the parameters. After each epoch, we calculate the validation loss and keep a copy of the model state if it improves on the best result so far. Once training finishes, we restore that checkpoint and evaluate it on the test set.

    A random split is suitable here because the examples were generated independently. With related examples, such as overlapping windows from one recording, we would need a split that keeps those related examples together to avoid leakage.

    The input and target tensors both have shape `(B,1)`. The first axis contains the examples, the second contains one input feature or one target value. Predictions have the same shape as the targets. We only need gradients for the model’s parameters, so the inputs and targets keep the default `requires_grad=False`.
    """)
    return


@app.cell
def _(torch):
    _data_generator = torch.Generator().manual_seed(42)
    _all_inputs = 2 * torch.rand(192, 1, generator=_data_generator) - 1
    _all_targets = (
        2 * _all_inputs + 1 + 0.15 * torch.randn(192, 1, generator=_data_generator)
    )
    _order = torch.randperm(192, generator=_data_generator)
    _train_indices, _valid_indices, _test_indices = (
        _order[:128],
        _order[128:160],
        _order[160:],
    )
    train_inputs, train_targets = (
        _all_inputs[_train_indices],
        _all_targets[_train_indices],
    )
    valid_inputs, valid_targets = (
        _all_inputs[_valid_indices],
        _all_targets[_valid_indices],
    )
    test_inputs, test_targets = (
        _all_inputs[_test_indices],
        _all_targets[_test_indices],
    )
    print("Training:", train_inputs.shape, train_targets.shape)
    print("Validation:", valid_inputs.shape, valid_targets.shape)
    print("Test:", test_inputs.shape, test_targets.shape)
    return (
        test_inputs,
        test_targets,
        train_inputs,
        train_targets,
        valid_inputs,
        valid_targets,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The training loop

    Each part of the training loop has a specific job:

    | Code | Operation |
    | --- | --- |
    | `model.train()` | Put the model’s layers into training mode |
    | `optimiser.zero_grad(set_to_none=True)` | Clear the stored gradients |
    | `predictions = model(batch_inputs)` | Calculate predictions and record tracked operations |
    | `loss = loss_function(predictions, batch_targets)` | Compare predictions with targets and calculate a scalar loss |
    | `loss.backward()` | Calculate the loss gradients with respect to the parameters, $\nabla_\theta L$ |
    | `optimiser.step()` | Use the stored gradients to update the parameters |

    We call `model.train()` before processing the training batches. It sets the behaviour of layers such as Dropout and BatchNorm. Our linear model produces the same results in training and evaluation modes, but including the mode changes gives us a pattern we can also use with those layers.

    For each batch, we clear the gradients, calculate predictions and loss, run backward, then update the parameters. The parameter values carry over to the next batch, where we repeat the process.

    The training cell creates the model, data loader and optimiser together. Rerunning the cell therefore starts a new experiment with freshly initialised parameters and optimiser state.
    """)
    return


@app.cell
def _(
    DataLoader,
    TensorDataset,
    copy,
    nn,
    torch,
    train_inputs,
    train_targets,
    valid_inputs,
    valid_targets,
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        fitted_model = nn.Linear(1, 1)
    _loss_function = nn.MSELoss()
    _optimiser = torch.optim.SGD(fitted_model.parameters(), lr=0.1)
    _loader = DataLoader(
        TensorDataset(train_inputs, train_targets),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(8),
        num_workers=0,
    )
    training_history = []
    validation_history = []
    _best_validation = float("inf")
    _best_state = copy.deepcopy(fitted_model.state_dict())
    best_epoch = 0
    for _epoch in range(10):
        fitted_model.train()
        for _batch_inputs, _batch_targets in _loader:
            _optimiser.zero_grad(set_to_none=True)
            _predictions = fitted_model(_batch_inputs)
            _loss = _loss_function(_predictions, _batch_targets)
            _loss.backward()
            _optimiser.step()
        fitted_model.eval()
        with torch.no_grad():
            _train_loss = _loss_function(
                fitted_model(train_inputs), train_targets
            ).item()
            _valid_loss = _loss_function(
                fitted_model(valid_inputs), valid_targets
            ).item()
        training_history.append(_train_loss)
        validation_history.append(_valid_loss)
        if _valid_loss < _best_validation:
            _best_validation = _valid_loss
            best_epoch = _epoch + 1
            _best_state = copy.deepcopy(fitted_model.state_dict())
    fitted_model.load_state_dict(_best_state)
    fitted_model.eval()
    print("Selected epoch:", best_epoch)
    print("Selected validation MSE:", _best_validation)
    print(
        "Learned weight and bias:",
        fitted_model.weight.item(),
        fitted_model.bias.item(),
    )
    return best_epoch, fitted_model, training_history, validation_history


@app.cell
def _(
    best_epoch,
    fitted_model,
    plt,
    torch,
    train_inputs,
    train_targets,
    training_history,
    valid_inputs,
    valid_targets,
    validation_history,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _axes[0].semilogy(
        range(1, len(training_history) + 1), training_history, label="Training"
    )
    _axes[0].semilogy(
        range(1, len(validation_history) + 1),
        validation_history,
        label="Validation",
    )
    _axes[0].axvline(best_epoch, color="grey", linestyle="--", label="Selected epoch")
    _axes[0].set(
        xlabel="Epoch",
        ylabel="MSE (log scale)",
        title="Full-split loss after each epoch",
    )
    _axes[0].legend()
    _axes[1].scatter(
        train_inputs[:, 0],
        train_targets[:, 0],
        s=14,
        alpha=0.5,
        label="Training",
    )
    _axes[1].scatter(valid_inputs[:, 0], valid_targets[:, 0], s=16, label="Validation")
    _grid = torch.linspace(-1, 1, 100).reshape(-1, 1)
    with torch.no_grad():
        _fitted_line = fitted_model(_grid)
    _axes[1].plot(
        _grid[:, 0], _fitted_line[:, 0], color="black", label="Selected model"
    )
    _axes[1].set(
        xlabel="Input",
        ylabel="Target / prediction",
        title="What the fitted function does",
    )
    _axes[1].legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16. Generalisation and baselines

    Training loss measures how well the model fits the training examples. We also want useful predictions for examples it has not seen before. This ability is called **generalisation**.

    A model can reduce its training loss by fitting noise or using information that would be unavailable when making real predictions. Its performance can also change if new inputs differ from the training data. Comparing training and validation results helps us assess whether improvements carry over to unseen examples.

    We use validation results to choose model settings and select a checkpoint. The test set provides a final evaluation after those decisions. If we repeatedly adjust the model based on test results, we are using that set for model selection too.

    A **baseline** gives us a simple result to compare against. For this regression problem, we predict the mean of the training targets for every example. We then compare the selected model and this baseline on the same test set. The mean must come from the training set so we do not use test targets to construct the predictor.

    Our generated targets contain random noise, so even a model that recovers the original line will still make errors. We should expect it to improve on the baseline, but not to achieve zero loss on unseen examples.
    """)
    return


@app.cell
def _(fitted_model, test_inputs, test_targets, torch, train_targets):
    with torch.no_grad():
        _test_predictions = fitted_model(test_inputs)
        _test_mse = (_test_predictions - test_targets).square().mean()
        _baseline_mse = (train_targets.mean() - test_targets).square().mean()
    print("Test MSE:", _test_mse.item())
    print("Training-mean baseline test MSE:", _baseline_mse.item())
    print("Test RMSE (target units):", _test_mse.sqrt().item())
    assert _test_mse < _baseline_mse, (
        "The fitted line should beat the mean baseline here."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 17. Classification outputs and loss

    For classification, we want to predict which class an example belongs to. A common approach is for the model to produce one **logit** per class. A logit is an unnormalised score: it can be positive or negative, and the scores do not need to sum to one.

    Softmax converts these scores into probabilities:

    $$p_k=\frac{\exp(z_k)}{\sum_j\exp(z_j)},$$

    where $z_k$ is the logit for class $k$ and $p_k$ is its predicted probability. The denominator sums over all classes, so the resulting probabilities sum to one.

    For an example whose target class is $c$, the cross-entropy loss is

    $$L=-\log p_c.$$

    This loss becomes smaller as the probability assigned to the correct class increases. For this unweighted, single-example calculation, its derivative with respect to each logit is

    $$\frac{\partial L}{\partial z_k}=p_k-\mathbf{1}_{k=c}.$$

    The expression $\mathbf{1}_{k=c}$ is 1 for the correct class and 0 for every other class. The correct class therefore receives a gradient of $p_c-1$, whilst the other classes receive gradients of $p_k$. These gradients pass backwards through the model using the same chain rule we used for regression. Averaging over a batch of $B$ examples adds a factor of $1/B$.

    In PyTorch, [`CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) takes logits directly and combines log-softmax with the negative log-likelihood calculation. For a batch with one target class per example:

    | Tensor | Shape | Contents |
    | --- | --- | --- |
    | Model output | `(B,K)` | Floating-point logits for $K$ classes |
    | Targets | `(B,)` | Class indices from 0 to $K-1`, stored as `torch.long` |

    We pass the model’s logits directly to the loss. Applying softmax first makes `CrossEntropyLoss` treat the resulting probabilities as logits, changing the calculation. We can apply softmax separately when we want probabilities for displaying or interpreting predictions.

    For binary classification, we can instead produce one logit per example and use `BCEWithLogitsLoss`. This combines sigmoid with binary cross entropy and expects floating-point targets, usually 0 or 1, with the same shape as the logits.
    """)
    return


@app.cell
def _(nn, torch):
    _logits = torch.tensor([[1.0, 2.0, -1.0]], requires_grad=True)
    _class_targets = torch.tensor([1], dtype=torch.long)
    _classification_loss = nn.CrossEntropyLoss()(_logits, _class_targets)
    _classification_loss.backward()
    _probabilities = _logits.detach().softmax(dim=1)
    _expected_gradient = _probabilities.clone()
    _expected_gradient[0, _class_targets.item()] -= 1
    _stable_manual_loss = -_logits.detach().log_softmax(dim=1)[0, _class_targets.item()]
    print("Probabilities:", _probabilities)
    print("Cross entropy:", _classification_loss.item())
    print("Logit gradient:", _logits.grad)
    torch.testing.assert_close(_classification_loss.detach(), _stable_manual_loss)
    torch.testing.assert_close(_logits.grad, _expected_gradient)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 18. Optimisers

    Once backpropagation has calculated the gradients, the optimiser uses them to update the parameters. [`torch.optim`](https://docs.pytorch.org/docs/stable/optim.html) provides several update rules. For the same parameters and loss, the gradients are the same; the chosen rule determines how we use them.

    Plain **stochastic gradient descent (SGD)**, without momentum or weight decay, updates each parameter using

    $$\theta_{\text{new}}=\theta-\eta g,$$

    where $\eta$ is the learning rate and $g$ is the gradient. During mini-batch training, we calculate $g$ from the current batch.

    **Momentum** keeps a running combination of current and previous gradients. This carries information from earlier updates into the next step and can reduce oscillation whilst maintaining progress along a consistent direction.

    **Adam** keeps moving averages of both gradients and squared gradients. It corrects for the initial bias in these averages and uses them to scale updates separately for each parameter. We still need to choose a suitable learning rate and evaluate the resulting model.

    The next cell compares plain SGD with our hand-calculated update, then inspects Adam’s stored state after one step. Adam creates additional tensors for its moving averages, so our training memory budget must include parameters, gradients, saved activations and optimiser state.

    ### Regularisation

    We can use regularisation to influence which parameter values training favours. For example, adding an L2 penalty gives the objective

    $$L_{\text{total}}=L_{\text{data}}+\lambda\lVert\theta\rVert^2.$$

    The penalty contributes $2\lambda\theta$ to the gradient, encouraging smaller parameter magnitudes. The value of $\lambda$ controls its strength.

    `AdamW` applies **decoupled weight decay**, shrinking parameters separately from Adam’s gradient-based update. This generally produces different updates from adding an L2 penalty to the loss when using Adam. We choose the regularisation strength and assess its effect using validation results.
    """)
    return


@app.cell
def _(nn, torch):
    _sgd_parameter = nn.Parameter(torch.tensor(0.5))
    _sgd = torch.optim.SGD([_sgd_parameter], lr=0.1, momentum=0.0, weight_decay=0.0)
    _sgd.zero_grad(set_to_none=True)
    ((2 * _sgd_parameter - 4) ** 2).backward()
    _expected_parameter = _sgd_parameter.detach().clone() - 0.1 * _sgd_parameter.grad
    _sgd.step()
    torch.testing.assert_close(_sgd_parameter.detach(), _expected_parameter)
    print(
        "SGD parameter:",
        _sgd_parameter.item(),
        "hand update:",
        _expected_parameter.item(),
    )
    _adam_parameter = nn.Parameter(torch.tensor([0.5, -0.5]))
    _adam = torch.optim.Adam([_adam_parameter], lr=0.1)
    print("Adam state before a step:", dict(_adam.state))
    _adam_parameter.square().sum().backward()
    _adam.step()
    print("Adam state after a step:")
    for _name, _value in _adam.state[_adam_parameter].items():
        print(_name, _value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 19. Evaluation mode and gradient recording

    During validation and prediction, we usually set the model to evaluation mode and disable gradient recording:

    ```python
    model.eval()

    with torch.no_grad():
        predictions = model(inputs)
    ```

    `model.eval()` puts the model and its submodules into evaluation mode. This changes the behaviour of layers such as Dropout, which stops randomly dropping activations, and BatchNorm, which normally uses its stored running mean and variance. Layers such as `nn.Linear` perform the same calculation in either mode. Autograd remains available, so we can still calculate gradients in evaluation mode when we need them.

    `torch.no_grad()` disables gradient recording for calculations inside its context. This reduces the memory needed when we only want predictions. It leaves the model’s training or evaluation mode unchanged, so a Dropout layer still in training mode will continue dropping activations.

    The example below demonstrates both behaviours. We use `eval()` and inspect whether the output requires gradients, then use `no_grad()` whilst the model is in training mode and inspect the Dropout output.

    PyTorch also provides `torch.inference_mode()`, which removes additional autograd bookkeeping. Tensors created in that mode have restrictions on their later use in gradient-tracked calculations. We use `no_grad()` here because it is sufficient for validation and prediction in this notebook.
    """)
    return


@app.cell
def _(nn, torch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        _mode_model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(p=0.5))
        _mode_inputs = torch.ones(8, 2)
        _mode_model.eval()
        _eval_output = _mode_model(_mode_inputs)
        print(
            "eval(): training flag",
            _mode_model.training,
            "output requires_grad",
            _eval_output.requires_grad,
        )
        _mode_model.train()
        with torch.no_grad():
            _no_grad_output = _mode_model(_mode_inputs)
        print(
            "no_grad(): training flag",
            _mode_model.training,
            "output requires_grad",
            _no_grad_output.requires_grad,
        )
        print("Dropout output whilst still training:\n", _no_grad_output)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 20. Gradient paths

    To calculate a parameter’s gradient, autograd needs a differentiable path connecting that parameter to the loss. We can deliberately disconnect a tensor from the graph or stop recording calculations when gradients are unnecessary.

    | Operation | What happens | Typical use |
    | --- | --- | --- |
    | `tensor.detach()` | Returns a tensor disconnected from the current graph, sharing the original tensor’s storage | Treating a prediction as a fixed target or preparing values for display |
    | `tensor.item()` | Converts a one-element tensor into a Python number | Recording scalar metrics |
    | `torch.no_grad()` | Disables gradient recording for calculations inside the context | Updating parameters or calculating validation predictions |

    These operations also affect subsequent calculations. For example, if we detach a prediction before calculating the loss, that loss can no longer send gradients through the prediction to the model’s parameters.

    A later expression involving the detached tensor may still require gradients if another tensor in that expression requires them. Those gradients follow the remaining tracked paths; they do not reconnect the detached tensor to its original calculation.

    Similarly, converting a tensor to NumPy, performing calculations and converting the result back does not preserve the original autograd history. Setting `requires_grad=True` on the new tensor starts tracking from that point.

    For normal training, we keep gradient recording enabled during the forward pass and loss calculation. We use `no_grad()` when deliberately updating parameters or evaluating the model. Avoid using `.data` to work around autograd errors, as it can bypass checks that detect changes to values needed for backward.
    """)
    return


@app.cell
def _(torch):
    _detached_weight = torch.tensor(2.0, requires_grad=True)
    _detached_loss = (_detached_weight * 3).detach().square()
    try:
        _detached_loss.backward()
    except RuntimeError as _error:
        print("Detached loss:", str(_error).split("\n")[0])
    print("Original weight gradient:", _detached_weight.grad)
    _logged_weight = torch.tensor(2.0, requires_grad=True)
    _python_number = (_logged_weight * 3).item()
    print("item() returns:", type(_python_number).__name__)
    print("Weight gradient after logging:", _logged_weight.grad)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 21. In place operations

    An **in place operation** changes an existing tensor rather than creating a new one. Many PyTorch methods use a trailing underscore to indicate this, such as `add_()` and `mul_()`.

    Some backward calculations need tensor values saved during the forward pass. If we modify those values before running backward, the saved information may no longer match the original calculation. PyTorch tracks tensor versions and can raise an error when it detects such a change.

    For example, the derivative of $L=w^2$ is $2w$, which requires the value of $w$ used to calculate the loss. In the example below, we change `w` after calculating the loss but before calling `backward()`. PyTorch detects the modification and raises an error, which we catch so the notebook can continue.

    Using `no_grad()` prevents the modification from being recorded in the graph, but the modification still happens. We therefore keep the normal training order: calculate predictions and loss, run backward, then update the parameters. The next forward pass uses the updated values.
    """)
    return


@app.cell
def _(torch):
    _modified_weight = torch.tensor(2.0, requires_grad=True)
    _saved_loss = _modified_weight.square()
    with torch.no_grad():
        _modified_weight.add_(1.0)
    try:
        _saved_loss.backward()
    except RuntimeError as _error:
        print("Expected version error:", str(_error).split("\n")[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 22. PyTorch and experiment design

    PyTorch provides the machinery for tensor calculations, automatic differentiation and parameter updates. We choose how to use that machinery to represent our problem and train a model.

    | PyTorch provides | We choose |
    | --- | --- |
    | Tensor storage and execution on supported devices | Shapes, dtypes, device and memory budget |
    | Derivative rules for supported operations | Model structure and loss function |
    | Recording of tracked operations | Where gradients should flow and where to stop them |
    | Values saved for backward calculations | Model size and batch size |
    | Registration of parameters, submodules and buffers | Layers, activations and initialisation |
    | Gradient accumulation | When to clear gradients and how to scale losses |
    | Optimiser update rules | Learning rate, regularisation and stopping criteria |
    | Batch loading through `DataLoader` | Data splits, sampling, normalisation and leakage checks |

    Most tensor arithmetic runs in compiled routines called **kernels**, which process many elements without a Python loop over each one. Moving a model to a GPU changes where these calculations run. The loss and gradient calculations follow the same principles, although floating-point results can differ between devices. Model parameters, inputs and targets need compatible devices and suitable dtypes.

    ### Saving model state

    A model’s `state_dict()` contains its parameters and persistent buffers. This lets us save learned weights and stored values such as BatchNorm’s running statistics.

    Resuming training also requires the state of the training process. Depending on the experiment, we may need optimiser and learning-rate scheduler state, random-generator states, progress through the data and the configuration used to construct the model.

    The tensors returned by `state_dict()` reference the model’s current state. If we keep that dictionary in memory whilst training continues, its values can change with the model. I used `copy.deepcopy()` when selecting the best checkpoint so that later updates would not alter the saved copy.

    We can explore further details later, including custom backward calculations with `autograd.Function`, convolution kernels, mixed precision, activation checkpointing and `torch.compile`. Each builds on the forward calculation, backward calculation and parameter update we have examined here.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 23. Checking a training loop

    Before running a training experiment, I would check the following against the code:

    1. What does one example contain, and which inputs will be available when making predictions?
    2. What are the shapes of the inputs, predictions and targets? Could broadcasting allow an incorrect calculation to run?
    3. Does the loss function match the model’s output representation and the targets’ shape and dtype?
    4. Which tensors are learnable parameters? Does the optimiser hold references to the parameters used by the model?
    5. Do tracked operations connect the loss to every parameter we intend to train?
    6. Where do we clear gradients, calculate new ones and update the parameters?
    7. How are training, validation and test examples separated? Could related examples leak between these sets?
    8. Which baseline will we use to judge whether the model gives a useful result?

    During training, inspect the values behind the loss curve. If training loss stays unchanged, check whether gradients are being calculated and whether the intended parameters change after `optimiser.step()`.

    If training loss decreases whilst validation loss rises, the model may be overfitting. Check the data split and preprocessing, then consider model capacity, regularisation and when to stop training. These checks help us connect the observed behaviour to a specific part of the experiment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Further reading

    These official references connect the calculations to the APIs used here. The examples were checked with the installed PyTorch version printed at the start; the online stable documentation can move ahead of it.

    - [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html): graph construction, saved tensors and gradient modes.
    - [Tensor.backward](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html): accumulation and the supplied output gradient.
    - [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) and [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html): parameter storage and module behaviour.
    - [MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) and [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html): reductions and target conventions.
    - [Optimisation tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) and [optimisers](https://docs.pytorch.org/docs/stable/optim.html): putting the loop together.

    For more API examples, return to [the autograd notebook](PyTorchForMLPart2Autograd.py). For prediction after training, continue to [Part 4: inference](PyTorchForMLPart4Inference.py).
    """)
    return


if __name__ == "__main__":
    app.run()
