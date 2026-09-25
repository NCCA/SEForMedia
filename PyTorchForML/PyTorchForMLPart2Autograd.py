#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PyTorch for Machine Learning, Part 2: autograd and the training loop

    As mentioned in the previous notebook autograd is the main thing that sets PyTorch apart from NumPy.

    In `NumPyForML/NumPyForMLPart4Random.py` we derived the gradient of a logistic neuron by hand, then checked it numerically to make sure the algebra was right. This works, and for one neuron it is a reasonable effort. For a network with ten layers it is not, and every time you change the architecture you start again.

    Autograd removes that job entirely. PyTorch records every operation you perform on a tensor, and when you ask for the gradient it walks the recording backwards applying the chain rule. You write the forward pass; the backward pass is derived for you.

    Four calls appear in virtually all model training demos, `model.train()`, `optimizer.zero_grad()`, `loss.backward()` and `optimizer.step()`. By the end of this notebook you should know what each of them does and, more to the point, what goes wrong when one is missing.
    """)
    return


@app.cell
def _():
    import torch

    return (torch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Chain Rule

    The chain rule tells us how a change in an input affects an output through a sequence of calculations. We multiply the derivatives at each step.
    For example, if \(y = 3x\) and \(z = y^2\), then:
    \[
    \frac{dz}{dx}
    =
    \frac{dz}{dy}\frac{dy}{dx}
    =
    2y \times 3
    \]At \(x = 2\), we have \(y = 6\), so the derivative is \(12 \times 3 = 36\). This means a small change in \(x\) produces approximately 36 times that change in \(z\), near this point.
    PyTorch’s autograd records the operations used to calculate the output. Calling .backward() works backwards through these operations, applying the chain rule automatically:
    """)
    return


@app.cell
def _(torch):
    _x = torch.tensor(2.0, requires_grad=True)
    _y = 3 * _x
    _z = _y**2

    _z.backward()

    print(_x.grad)  # tensor(36.)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, requires_grad=True tells PyTorch to track calculations involving x. After z.backward(), x.grad contains the derivative of z with respect to x.
    In a neural network, the same process works backwards from the loss through each layer to calculate how each weight affects the loss. When several paths contribute, their gradient contributions are added together. The optimiser then uses these gradients to update the weights; .backward() calculates the gradients but does not update the weights itself.

    The following sections will explain this in more detail.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## requires_grad, and the recording

    A tensor with `requires_grad=True` is one PyTorch watches. Any tensor computed from it inherits the watching, and carries a `grad_fn` — a reference to the operation that made it, which is the link in the chain.

    ```python
    torch.tensor(data, requires_grad=False)   # the flag, off by default
    t.requires_grad_(True)                    # turn it on in place, note the underscore
    t.grad_fn                                 # the operation that produced t, or None
    t.grad                                    # where the gradient lands after backward()
    t.is_leaf                                 # True if you made it, False if it was computed
    ```

    Start with something you can differentiate in your head. If `y = x²` then `dy/dx = 2x`, so at `x = 3` the gradient should be 6.
    """)
    return


@app.cell
def _(torch):
    x = torch.tensor(3.0, requires_grad=True)
    y = x**2

    print("x        ", x, "is_leaf", x.is_leaf)
    print("y        ", y)
    print("y.grad_fn", y.grad_fn, "<- y knows it came from a power operation")

    y.backward()
    print()
    print("x.grad   ", x.grad, "and 2x at x=3 is", 2 * 3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use a longer chain to see how the derivative is built up from individual operations. Take

    $$
    y = (3x + 1)^2
    $$

    and evaluate its derivative at \(x = 2\). Introducing an intermediate value,

    $$
    u = 3x + 1,
    \qquad
    y = u^2.
    $$

    Applying the chain rule gives

    $$
    \frac{dy}{dx}
    =
    \frac{dy}{du}\frac{du}{dx}
    =
    2u \cdot 3
    =
    2(3x + 1)\cdot 3.
    $$

    At \(x = 2\), we have \(u = 7\), so

    $$
    \left.\frac{dy}{dx}\right|_{x=2}
    =
    2 \cdot 7 \cdot 3
    =
    42.
    $$

    PyTorch builds up this result using the derivative of each operation and the chain rule.
    """)
    return


@app.cell
def _(torch):
    x2 = torch.tensor(2.0, requires_grad=True)

    step_a = 3 * x2
    step_b = step_a + 1
    out = step_b**2

    print("the recorded chain, read backwards from the output:")
    print("  ", out.grad_fn)
    print("  ", out.grad_fn.next_functions[0][0])
    print("  ", out.grad_fn.next_functions[0][0].next_functions[0][0])

    out.backward()
    print()
    print("x2.grad  ", x2.grad.item())
    print("by hand  ", 2 * (3 * 2 + 1) * 3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That chain printed backwards from the output is the graph. It is built fresh on every forward pass, which is what people mean when they call PyTorch a [*define-by-run*](https://www.shadecoder.com/topics/define-by-run-a-comprehensive-guide-for-2025) framework, there is no separate compilation step, the graph is whatever your Python actually did this time round. It is also why you can put an `if` or a loop in `forward` and it simply works.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [loss.backward()](https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html)

    `loss.backward()` works backwards through the computation graph, applying the chain rule to calculate gradients.

    ```python
    loss.backward(gradient=None, retain_graph=None, create_graph=False, inputs=None)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `gradient` | `None` | Supplies the starting gradient; required when the loss has more than one element. |
    | `retain_graph` | `None` | Keeps the graph for another backward pass when `True`. Defaults to the value of `create_graph`. |
    | `create_graph` | `False` | Records the gradient calculation so we can calculate higher derivatives. |
    | `inputs` | `None` | Selects which tensors receive gradients. By default, these are the leaf tensors involved in the calculation that require gradients. |

    During training, gradients accumulate in each participating parameter’s `.grad`. The weights stay unchanged until we call `optimiser.step()`. Existing gradients are added to, so we use `optimiser.zero_grad()` when we want to start a fresh calculation.

    We usually combine the per-sample losses into one scalar using `mean()`:

    ```python
    loss = per_sample_losses.mean()
    loss.backward()
    ```

    A vector of losses also has derivatives, but we must supply a matching `gradient` tensor to specify how their contributions are weighted.
    """)
    return


@app.cell
def _(torch):
    vec = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    not_scalar = vec * 2

    try:
        not_scalar.backward()
    except RuntimeError as e:
        print("RuntimeError:", str(e).split("\n")[0])

    not_scalar.mean().backward()  # reduce first, then it works
    print("after reducing with mean, vec.grad =", vec.grad)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why [zero_grad](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html) exists

    **PyTorch accumulates gradients.** Each call to `backward()` adds the calculated gradient to `.grad`. In the cell below, we run the same calculation three times without clearing the gradients between calls. The derivative is the same each time, but `.grad` stores the running total.
    """)
    return


@app.cell
def _(torch):
    w = torch.tensor(3.0, requires_grad=True)

    for call in range(1, 4):
        loss_demo = w**2  # the same computation every time
        loss_demo.backward()
        print(
            f"after backward() #{call}: w.grad = {w.grad.item():5.1f}  (true gradient is 6.0)"
        )

    print()
    w.grad.zero_()  # this is what optimizer.zero_grad() does for you
    (w**2).backward()
    print(f"after zeroing first:       w.grad = {w.grad.item():5.1f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So a training loop that forgets `zero_grad` does not fail, it just takes steps proportional to the sum of every gradient it has ever seen, which grows without bound. The loss usually goes haywire within a few epochs, but not immediately, which makes it an annoying bug to track down.

    The accumulation default is deliberate rather than an oversight. It lets you split a batch too large for memory into several smaller passes, call `backward()` on each, and take one step on the total, *gradient accumulation*, and a standard technique when training big models on small GPUs. You get that for free, at the cost of having to remember one line.

    ```python
    optimizer.zero_grad(set_to_none=True)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `set_to_none` | `True` | set `.grad` to `None` rather than a tensor of zeros |

    The default changed to `True` in PyTorch 2.0. Setting to `None` is slightly faster and frees the memory; the only thing to watch is that `param.grad` is then `None` rather than zeros if you go looking at it before the first `backward()`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ,## Optimisers and [step()](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)

    An optimiser holds a reference to the parameters and the rule for updating them from their gradients.

    ```pyth,
    torch.optim.SGD(params, lr=0.001, momentum=0, weight_decay=0, nesterov=False)
    torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `params` | required | usually `model.parameters()` |
    | `lr` | `0.001` | learning rate — how far to move per step |
    | `momentum` (SGD) | `0` | carry a fraction of the previous step, smooths the path |
    | `betas` (Adam) | `(0.9, 0.999)` | decay rates for Adam's running averages |
    | `weight_decay` | `0` | L2 penalty pulling weights towards zero |

    For plain SGD, `step()` is just `w -= lr * w.grad` applied to every parameter. Adam keeps a running average of the gradient and of its square, and uses them to scale each parameter's step individually, which is why `optim.Adam(model.parameters())` with no learning rate at all often works, and why `TransferLearning.py:39` does exactly that.

    The split between `backward()` and `step()` is what lets you swap SGD for Adam by changing one line. One computes gradients, the other decides what to do with them.
    """)
    return


@app.cell
def _(torch):
    # step() by hand, so there is no magic in it
    manual_w = torch.tensor(5.0, requires_grad=True)
    manual_lr = 0.1

    for _ in range(3):
        manual_loss = manual_w**2  # minimum is at w = 0
        manual_loss.backward()
        with torch.no_grad():  # do not record the update itself
            manual_w -= manual_lr * manual_w.grad
        manual_w.grad.zero_()
        print(f"w = {manual_w.item():.3f}")

    print()
    print("and the same thing with an optimiser:")
    auto_w = torch.tensor(5.0, requires_grad=True)
    opt = torch.optim.SGD([auto_w], lr=0.1)

    for _ in range(3):
        opt.zero_grad()
        (auto_w**2).backward()
        opt.step()
        print(f"w = {auto_w.item():.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Swapping the rule really is one line, so here are both on the same problem with the same learning rate and the same seed. Adam is not doing anything SGD cannot; it is choosing a different step size per parameter from the running averages it keeps, and on a problem this small that shows up as getting closer in the same number of steps.
    """)
    return


@app.cell
def _(torch):
    def fit_line(make_optimiser, steps: int = 100) -> tuple[float, float, float]:
        """Fit y = 2x + 1 with one optimiser and report where it got to."""
        torch.manual_seed(7)
        fit_x = torch.arange(0.0, 1.0, 0.02).unsqueeze(1)
        fit_y = 2.0 * fit_x + 1.0
        fit_model = torch.nn.Linear(1, 1)
        fit_opt = make_optimiser(fit_model.parameters())
        fit_loss_fn = torch.nn.MSELoss()
        for _ in range(steps):
            step_loss = fit_loss_fn(fit_model(fit_x), fit_y)
            fit_opt.zero_grad()
            step_loss.backward()
            fit_opt.step()
        return step_loss.item(), fit_model.weight.item(), fit_model.bias.item()

    sgd_result = fit_line(lambda params: torch.optim.SGD(params, lr=0.1))
    adam_result = fit_line(lambda params: torch.optim.Adam(params, lr=0.1))

    print("after 100 steps at lr=0.1, aiming for weight 2.0 and bias 1.0")
    print(
        f"  SGD : loss {sgd_result[0]:.6f}  weight {sgd_result[1]:.4f}  bias {sgd_result[2]:.4f}"
    )
    print(
        f"  Adam: loss {adam_result[0]:.6f}  weight {adam_result[1]:.4f}  bias {adam_result[2]:.4f}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Do not read that as "Adam is better". It is better here, on a two-parameter problem with a fixed budget of steps. Plenty of image classifiers still train to a better final accuracy with SGD and momentum, and the reason people reach for Adam first is that it is forgiving about the learning rate rather than that it always wins.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note the `with torch.no_grad():` around the manual update. Without it, subtracting from `manual_w` would itself be recorded as an operation, the graph would grow every iteration, and `manual_w` would stop being a leaf. The optimiser does the same thing internally. This is the first hint of why Part 4 spends time on `no_grad` and `inference_mode`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The loop

    Here it is in full, in the order `LinearModel/LinearModel.py:70` writes it. I have fitted `y = 2x + 1` so you can check the answer against something known.
    """)
    return


@app.cell
def _(torch):
    torch.manual_seed(42)

    X_train = torch.arange(0.0, 1.0, 0.02).unsqueeze(dim=1)  # (50, 1)
    y_train = 2.0 * X_train + 1.0  # the answer we are hoping to recover

    model = torch.nn.Linear(in_features=1, out_features=1)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

    print(
        "before training:",
        {k: round(v.item(), 3) for k, v in model.state_dict().items()},
    )
    return X_train, loss_fn, model, optimizer, y_train


@app.cell
def _(X_train, loss_fn, model, optimizer, y_train):
    losses = []

    for epoch in range(2000):
        model.train()  # 1. training mode
        y_pred = model(X_train)  # 2. forward pass
        loss = loss_fn(y_pred, y_train)  # 3. how wrong are we
        optimizer.zero_grad()  # 4. clear the old gradients
        loss.backward()  # 5. compute the new ones
        optimizer.step()  # 6. update the weights

        if epoch % 400 == 0:
            losses.append((epoch, loss.item()))

    for e, v in losses:
        print(f"epoch {e:5d}  loss {v:.6f}")

    print()
    print(
        "learned:",
        {k: round(v.item(), 3) for k, v in model.state_dict().items()},
    )
    print("wanted:  weight 2.0, bias 1.0")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These six lines form the training loop used in our examples. Each has a separate job:

    | Line | What it does |
    | --- | --- |
    | `model.train()` | Sets training mode, affecting layers such as dropout and batch normalisation. See Part 4. |
    | `model(X)` | Runs the forward pass and, with gradient tracking enabled, records the computation graph. |
    | `loss_fn(pred, y)` | Calculates the loss. Here, the loss function reduces it to one scalar. |
    | `optimizer.zero_grad()` | Clears the stored gradients, ready for the next backward pass. |
    | `loss.backward()` | Calculates and accumulates gradients in `.grad`. |
    | `optimizer.step()` | Uses the stored gradients to update the model parameters. |

    We can place `zero_grad()` anywhere after the previous `step()` and before the next `backward()` in this loop. `LinearModel.py` calls it after calculating the loss; another common approach is to call it at the start of each iteration. Both ensure that gradients from the previous iteration are cleared before we calculate the next ones.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [.item()](https://pytorch.org/docs/stable/generated/torch.Tensor.item.html) and [.detach()](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html)

    When recording losses or plotting results, we usually want the values without their gradient history. `.item()` returns a Python number from a tensor containing one element. `.detach()` returns a tensor disconnected from the computation graph.

    ```python
    loss.item()                # Python number for logging
    t.detach()                 # tensor without gradient history; shares memory
    t.detach().cpu().numpy()   # NumPy array for plotting with matplotlib
    ```

    For a loss history, we can store the number directly:

    ```python
    loss_history.append(loss.item())
    ```

    Storing `loss` itself keeps references to its graph. Although a normal `backward()` frees the saved intermediate tensors, keeping graph references can still consume unnecessary memory.

    Use `.item()` when we need a single Python number and `.detach()` when we want to keep a tensor, whatever its size. Detaching does not copy the data, so changes to the detached tensor also affect the original. Use `.detach().clone()` if we need an independent copy.
    """)
    return


@app.cell
def _(X_train, loss_fn, model, y_train):
    live_loss = loss_fn(model(X_train), y_train)

    print("the loss tensor   ", live_loss)
    print("grad_fn           ", live_loss.grad_fn, "<- the graph is attached")
    print()
    print("item()            ", live_loss.item(), type(live_loss.item()))
    print("detach().grad_fn  ", live_loss.detach().grad_fn, "<- cut loose")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bringing it back to NumPy Part 4

    We can now check our earlier work using PyTorch. Below, we recreate the logistic neuron from NumPy Part 4 and compare the gradient we derived by hand with the one calculated by autograd.

    If the gradients agree within numerical precision, we have a useful check that our derivation and implementation are consistent for this example. From here, we can use `backward()` to calculate the gradients automatically.
    """)
    return


@app.cell
def _(torch):
    torch.manual_seed(0)

    X_check = torch.randn(20, 3, dtype=torch.float64)
    y_check = (X_check[:, 0] + X_check[:, 1] > 0).to(torch.float64)
    w_check = (torch.randn(3, dtype=torch.float64) * 0.5).requires_grad_(True)
    b_check = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)

    # the hand-derived gradient from NumPy Part 4: X.T @ (sigmoid(Xw + b) - y) / n
    with torch.no_grad():
        p_hand = torch.sigmoid(X_check @ w_check + b_check)
        dw_by_hand = X_check.T @ (p_hand - y_check) / len(y_check)

    # and now autograd, from the loss alone
    p_auto = torch.sigmoid(X_check @ w_check + b_check)
    bce = -torch.mean(
        y_check * torch.log(p_auto) + (1 - y_check) * torch.log(1 - p_auto)
    )
    bce.backward()

    print("derived by hand:", dw_by_hand.numpy().round(8))
    print("from autograd:  ", w_check.grad.numpy().round(8))
    print()
    print(
        "do they agree?  ",
        torch.allclose(dw_by_hand, w_check.grad, atol=1e-10),
    )
    print("largest difference:", (dw_by_hand - w_check.grad).abs().max().item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The largest difference is around \(5 \times 10^{-17}\), consistent with rounding error in `float64` arithmetic for values of this size. The results agree to numerical precision, though they are not necessarily identical bit for bit.

    This is much closer than the `atol=1e-4` tolerance we used for the numerical check in NumPy Part 4. Finite differences estimate a derivative using a small change in the input. Autograd applies the chain rule to the recorded operations, with rounding errors from floating-point arithmetic.

    We have already done the same differentiation by hand. PyTorch automates that process using the computation graph recorded during the forward pass, saving us from deriving and implementing every gradient ourselves.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it

    Exercise 3 below asks you to set the learning rate to 1.0 and explain what you see. Here it is as a slider, minimising `f(w) = w²` from `w = 5`, because that is the one case where the maths is simple enough to predict exactly what the slider will do before you drag it.

    The gradient is `2w`, so a step is `w - lr·2w`, which is `w(1 - 2·lr)`. Every step multiplies `w` by the same constant, and that constant is all you need:

    - **`lr < 0.5`** — the multiplier is between 0 and 1, so `w` shrinks towards the minimum
    - **`lr = 0.5`** — the multiplier is exactly 0. One step lands on the answer
    - **`0.5 < lr < 1.0`** — the multiplier is negative but small, so `w` overshoots, changes sign, and still converges
    - **`lr = 1.0`** — the multiplier is exactly −1. `w` flips between 5 and −5 forever and the loss never moves
    - **`lr > 1.0`** — the multiplier is worse than −1 and it diverges, fast. At `lr = 2.0` twelve steps is enough to reach the hundreds of billions

    That last one is what a learning rate set too high looks like in a real training run, and it is why a loss of `nan` after a handful of epochs is nearly always the learning rate rather than the data.
    """)
    return


@app.cell
def _(mo):
    lr_slider = mo.ui.slider(
        0.05, 2.0, step=0.05, value=0.1, label="Learning rate", show_value=True
    )
    lr_slider
    return (lr_slider,)


@app.cell
def _(lr_slider, torch):
    lr_w = torch.tensor(5.0, requires_grad=True)
    lr_opt = torch.optim.SGD([lr_w], lr=lr_slider.value)
    lr_history = []

    for _ in range(12):
        lr_loss = lr_w**2
        lr_opt.zero_grad()
        lr_loss.backward()
        lr_opt.step()
        lr_history.append(lr_w.item())

    print(f"lr = {lr_slider.value}, starting from w = 5.0")
    print("  w after each step:", [f"{v:.3g}" for v in lr_history])
    print(f"  final w    {lr_history[-1]:.6g}")
    print(f"  final loss {lr_history[-1] ** 2:.6g}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Take `y = x³` at `x = 2`. Predict `x.grad` before running it.
    2. Delete `optimizer.zero_grad()` from the training loop above and run it. How many epochs before the loss becomes `nan`? Now put it back but move it after `step()` — does that still work?
    3. Set the learning rate to 1.0 and explain what you see. Then try 0.00001.
    4. Build a tensor with `requires_grad=True`, do some arithmetic, and find a way to make `.grad` stay `None` after `backward()`. What does that tell you about leaves?
    5. Append `loss` rather than `loss.item()` to a list for 1,000 epochs and watch the memory. (`torch.cuda.memory_allocated()` if you have a GPU, otherwise `psutil` or just Activity Monitor.)

    Part 3 builds the models properly, with `torch.nn`.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
