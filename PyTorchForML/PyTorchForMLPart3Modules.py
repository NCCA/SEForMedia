#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PyTorch for Machine Learning, Part 3: building models with torch.nn

    [`torch.nn`](https://pytorch.org/docs/stable/nn.html) provides the building blocks for our models. These include layers that transform inputs and manage any parameters they need, such as weights and biases.

    In Part 2, we saw how autograd calculates gradients through supported tensor operations. We can build models using those operations directly, but `torch.nn` handles much of the repeated work: creating parameters, keeping track of them, and combining layers into a model.

    This notebook introduces the layers used in our demos and the loss functions we use to train them. We will also look at [logits](https://en.wikipedia.org/wiki/Logit): the raw scores produced by a model, and why some loss functions expect these rather than probabilities.
    """)
    return


@app.cell
def _():
    import torch
    from torch import nn

    torch.manual_seed(42)
    return nn, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)

    `nn.Module` is the base class we use to build models. In `__init__`, we initialise the base class and create our layers. In `forward`, we describe how the input passes through them.

    ```python
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(4, 2)

        def forward(self, x):
            return self.layer(x)
    ```

    Using `nn.Module` gives us a consistent way to manage the model:

    - **Parameters:** assigning a layer to an attribute such as `self.layer` registers it as a submodule. `model.parameters()` then includes its parameters, along with those of any nested layers.
    - **Devices:** `model.to(device)` moves registered parameters and buffers throughout the model to the chosen device.
    - **Training mode:** `model.train()` and `model.eval()` set the mode throughout the model. Layers such as dropout and batch normalisation use this to change their behaviour.

    We define `forward()`, but call the model using `model(x)`. This lets PyTorch run its registered hooks around the forward pass. Calling `model.forward(x)` directly bypasses that handling.
    """)
    return


@app.cell
def _(nn, torch):
    class TinyModel(nn.Module):
        def __init__(self, in_features: int, hidden: int, out_features: int):
            super().__init__()
            self.stack = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_features),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.stack(x)

    tiny = TinyModel(4, 8, 2)
    print(tiny)
    return (tiny,)


@app.cell
def _(tiny):
    print("every parameter the module found:")
    total = 0
    for pname, param in tiny.named_parameters():
        print(f"  {pname:20} {tuple(param.shape)}  {param.numel():>3} values")
        total += param.numel()
    print("total trainable parameters:", total)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can check the parameter count by hand. Each linear layer has one weight per input–output connection and, by default, one bias per output:

    | Layer | Weights | Biases | Total |
    | --- | --- | --- | --- |
    | 4 inputs → 8 outputs | \(4 \times 8 = 32\) | 8 | 40 |
    | 8 inputs → 2 outputs | \(8 \times 2 = 16\) | 2 | 18 |
    | **Total** | **48** | **10** | **58** |

    Doing this once helps us see where the parameters come from and check that the model matches our intended design.

    `model.parameters()` iterates over the registered parameters without their names. We usually pass it directly to the optimiser:

    ```python
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    ```

    This tells the optimiser which parameters to update. A layer must be registered within the model for its parameters to appear here.

    Assigning a layer directly to `self.layer` registers it, but storing layers only in a plain Python list does not. For a collection of layers, use `nn.ModuleList` or `nn.Sequential` and assign that container to the model. Otherwise, their parameters will be missing from `model.parameters()`, and this optimiser will not update them.
    """)
    return


@app.cell
def _(nn, torch):
    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(4, 4), nn.Linear(4, 2)]  # a plain list

        def forward(self, x):
            for layer_item in self.layers:
                x = layer_item(x)
            return x

    class Fixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 2)])

        def forward(self, x):
            for layer_item in self.layers:
                x = layer_item(x)
            return x

    print("plain list  :", len(list(Broken().parameters())), "parameters found")
    print("ModuleList  :", len(list(Fixed().parameters())), "parameters found")
    print()
    print(
        "both forward passes work, so nothing tells you:",
        Broken()(torch.randn(2, 4)).shape,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [nn.Parameter](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html)

    We use `nn.Parameter` to register a tensor as a model parameter.

    ```python
    nn.Parameter(data, requires_grad=True)
    ```

    Assigning a `Parameter` to a module attribute makes it available through `model.parameters()` and includes it in `model.state_dict()`:

    ```python
    self.scale = nn.Parameter(torch.tensor(1.0))
    ```

    We can then use `self.scale` in `forward()`. An optimiser constructed from `model.parameters()` will receive it and can update it during training.

    An ordinary tensor assigned to `self` is not registered automatically. It can still participate in gradient calculations if configured to do so, but it will not appear in `model.parameters()` or `model.state_dict()`.

    Standard layers create their own parameters, so we mainly use `nn.Parameter` when adding something custom, such as a learned temperature, a scale for each channel, or a mixing weight between two branches. For a tensor that should move and be saved with the model without being optimised, we can use `register_buffer()` instead.
    """)
    return


@app.cell
def _(nn, torch):
    class Scale(nn.Module):
        """Multiplies by one learned number. Wrap it or not, to see the difference."""

        def __init__(self, wrap: bool):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0)) if wrap else torch.tensor(1.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.scale

    for wrapped in (False, True):
        probe = Scale(wrapped)
        found = len(list(probe.parameters()))
        print(f"nn.Parameter({wrapped}):  parameters() finds {found}", end="")
        print(f"   state_dict keys {list(probe.state_dict())}")
    return (Scale,)


@app.cell
def _(Scale, torch):
    # the wrapped one trains: x * scale should learn scale = 3 from 2 -> 6
    trainable = Scale(wrap=True)
    scale_opt = torch.optim.SGD(trainable.parameters(), lr=0.1)

    for _ in range(20):
        scale_loss = (trainable(torch.tensor([2.0])) - 6.0) ** 2
        scale_opt.zero_grad()
        scale_loss.mean().backward()
        scale_opt.step()

    print("learned scale:", round(trainable.scale.item(), 4), "- wanted 3.0")
    print()

    # the unwrapped one has nothing to hand the optimiser at all
    try:
        torch.optim.SGD(Scale(wrap=False).parameters(), lr=0.1)
    except ValueError as e:
        print("unwrapped:", type(e).__name__, "-", e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, the `ValueError` tells us that the optimiser received no parameters. If the model also contained an `nn.Linear` layer, its parameters would reach the optimiser and no error would be raised. The ordinary tensor would still be missing, so that optimiser would never update it.

    This is the same registration problem we saw with a plain Python list of layers. Check the names and number of parameters the model exposes against what you intended to build.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

    `nn.Linear` is a fully connected layer. It applies a learned weight matrix and an optional bias:

    $$
    y = xW^{\mathsf{T}} + b
    $$

    ```python
    nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `in_features` | required | Number of input features. |
    | `out_features` | required | Number of output features. |
    | `bias` | `True` | Includes a learnable bias for each output feature. |

    When connecting linear layers directly, the output size of one must match the input size of the next. Using keyword arguments makes this easier to check:

    ```python
    self.hidden = nn.Linear(in_features=4, out_features=8)
    self.output = nn.Linear(in_features=8, out_features=2)
    ```

    The first layer produces eight features, which is what the second layer expects.

    `nn.Linear` transforms only the last dimension and preserves all preceding dimensions. For example, passing an input of shape `(32, 10, 4)` through `nn.Linear(4, 2)` produces an output of shape `(32, 10, 2)`. The same weights and biases are applied to each vector of four features.
    """)
    return


@app.cell
def _(nn, torch):
    lin = nn.Linear(in_features=4, out_features=2)

    print(
        "weight",
        tuple(lin.weight.shape),
        " note it is (out, in), not (in, out)",
    )
    print("bias  ", tuple(lin.bias.shape))
    print()
    print("a batch of 5:      ", tuple(lin(torch.randn(5, 4)).shape))
    print("only the last axis:", tuple(lin(torch.randn(32, 10, 4)).shape))

    try:
        lin(torch.randn(5, 7))
    except RuntimeError as e:
        print()
        print("RuntimeError:", str(e).split("\n")[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

    `nn.Linear` is a fully connected layer. It applies a learned weight matrix and an optional bias:

    $$
    y = xW^{\mathsf{T}} + b
    $$

    ```python
    nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `in_features` | required | Number of input features. |
    | `out_features` | required | Number of output features. |
    | `bias` | `True` | Includes a learnable bias for each output feature. |

    When connecting linear layers directly, the output size of one must match the input size of the next. Using keyword arguments makes this easier to check:

    ```python
    self.hidden = nn.Linear(in_features=4, out_features=8)
    self.output = nn.Linear(in_features=8, out_features=2)
    ```

    The first layer produces eight features, which is what the second layer expects.

    `nn.Linear` transforms only the last dimension and preserves all preceding dimensions. For example, passing an input of shape `(32, 10, 4)` through `nn.Linear(4, 2)` produces an output of shape `(32, 10, 2)`. The same weights and biases are applied to each vector of four features.
    """)
    return


@app.cell
def _(nn, torch):
    no_activation = nn.Sequential(
        nn.Linear(3, 5, bias=False),
        nn.Linear(5, 4, bias=False),
        nn.Linear(4, 2, bias=False),
    )

    sample = torch.randn(6, 3)

    # the three layers, collapsed into one matrix by multiplying them out
    collapsed = (
        no_activation[0].weight.T
        @ no_activation[1].weight.T
        @ no_activation[2].weight.T
    )

    print("three layers :", no_activation(sample)[0].detach().numpy().round(5))
    print("one matrix   :", (sample @ collapsed)[0].detach().numpy().round(5))
    print()
    print(
        "identical?",
        torch.allclose(no_activation(sample), sample @ collapsed, atol=1e-6),
    )
    print("so the depth bought us nothing at all")
    return (sample,)


@app.cell
def _(nn, sample, torch):
    with_activation = nn.Sequential(
        nn.Linear(3, 5, bias=False),
        nn.ReLU(),
        nn.Linear(5, 4, bias=False),
        nn.ReLU(),
        nn.Linear(4, 2, bias=False),
    )

    flat = (
        with_activation[0].weight.T
        @ with_activation[2].weight.T
        @ with_activation[4].weight.T
    )

    print("with ReLU between the layers, no single matrix reproduces it:")
    print("  network :", with_activation(sample)[0].detach().numpy().round(5))
    print("  matrix  :", (sample @ flat)[0].detach().numpy().round(5))
    print(
        "  equal?  ",
        torch.allclose(with_activation(sample), sample @ flat, atol=1e-6),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)

    `nn.Sequential` chains modules together, passing the output of each to the next. It registers the modules and handles the forward pass through the sequence.

    ```python
    nn.Sequential(*modules)
    ```

    We can index and slice it to inspect individual layers or select part of a model. This is useful in transfer learning, where we may keep the early layers and replace the final ones.

    Use `nn.Sequential` when the data follows a single path through the layers. For skip connections, multiple inputs or conditional behaviour, we can describe the operations ourselves in `forward()`.

    A common pattern, used in `Checkpoints/CheckPoints.py`, is to assign a `Sequential` to an attribute inside an `nn.Module`. Our `forward()` then calls that sequence, with room to add other operations later.
    """)
    return


@app.cell
def _(nn, torch):
    stack = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )

    print("index it:", stack[0])
    print("slice it:", len(stack[:2]), "layers in the slice")
    print("run it:  ", tuple(stack(torch.randn(4, 8)).shape))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [nn.Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html)

    `nn.Flatten` combines a range of dimensions into one. By default, it flattens each sample into a vector while preserving the first dimension, which we normally use for the batch.

    ```python
    nn.Flatten(start_dim=1, end_dim=-1)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `start_dim` | `1` | First dimension to include. |
    | `end_dim` | `-1` | Last dimension to include; `-1` means the final dimension. |

    For example, a batch of 64 single-channel images with shape `(64, 1, 28, 28)` becomes `(64, 784)`, since each image contains \(1 \times 28 \times 28 = 784\) values.

    We often use this between convolutional layers and a fully connected layer, turning each sample’s feature maps into the feature vector expected by `nn.Linear`.
    """)
    return


@app.cell
def _(nn, torch):
    images = torch.randn(64, 1, 28, 28)
    print("a batch of MNIST digits:", tuple(images.shape))
    print(
        "after nn.Flatten():     ",
        tuple(nn.Flatten()(images).shape),
        "= 1*28*28",
    )
    return (images,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The convolutional block

    The ASL and transfer-learning demos combine several layers to extract image features, reduce their spatial dimensions and regularise training.

    ```python
    nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
    nn.MaxPool2d(kernel_size, stride=None, padding=0)
    nn.BatchNorm2d(num_features)
    nn.Dropout(p=0.5, inplace=False)
    ```

    | Layer | Key parameters | What it does |
    | --- | --- | --- |
    | [`Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) | `in_channels`, `out_channels`, `kernel_size` | Applies learned filters across the input, reusing the same weights at each spatial position. |
    | [`MaxPool2d`](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) | `kernel_size`, `stride` | Takes the maximum value in each window, usually reducing the height and width. |
    | [`BatchNorm2d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) | `num_features` = channel count | During training, normalises each channel using statistics across the batch and spatial dimensions, then applies a learned scale and offset. |
    | [`Dropout`](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) | `p` = probability of dropping an activation | During training, randomly sets activations to zero and scales the remaining values to preserve their expected value. |

    For `MaxPool2d`, `stride` defaults to `kernel_size`. This means `MaxPool2d(2)` uses non-overlapping \(2 \times 2\) windows, halving the height and width when both are even.

    `Dropout` defaults to `p=0.5`, so each activation has a 50% chance of being zeroed during training. `PreTrainedModels/FruitExample/Train.py` uses `0.3`. In evaluation mode, dropout leaves its input unchanged, whilst batch normalisation uses its running statistics by default.

    Convolution also keeps the parameter count manageable. Each filter connects to a local region and reuses its weights across the image, so increasing the image size does not increase the number of learned parameters. The cell below compares this with a fully connected layer.
    """)
    return


@app.cell
def _(nn):
    conv = nn.Conv2d(1, 25, kernel_size=3, stride=1, padding=1)
    dense = nn.Linear(28 * 28, 25 * 28 * 28)

    conv_params = sum(p.numel() for p in conv.parameters())
    dense_params = sum(p.numel() for p in dense.parameters())

    print(f"Conv2d(1, 25, 3)   {conv_params:>12,} parameters")
    print(f"the Linear version {dense_params:>12,} parameters")
    print(f"ratio              {dense_params / conv_params:>12,.0f}x")
    return (conv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is weight sharing. The convolution learns one small kernel and applies it everywhere, on the assumption that a feature worth detecting in one corner is worth detecting in the others. The fully connected version learns a separate weight for every pixel-to-pixel pair and has no idea the image has a spatial layout at all.

    Below is the first block of the ASL CNN, with the shapes printed at each step. `ASL/ASLPart2CNN.py:91` annotates each line with the resulting shape in a comment.
    """)
    return


@app.cell
def _(conv, images, nn, torch):
    block = nn.Sequential(
        nn.Conv2d(1, 25, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(25),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Dropout(0.2),
    )

    activation = images
    print(f"{'input':22} {tuple(activation.shape)}")
    for module in block:
        activation = module(activation)
        print(f"{module.__class__.__name__:22} {tuple(activation.shape)}")

    print()
    print(
        "the conv kernel itself:",
        tuple(conv.weight.shape),
        "= (out_ch, in_ch, kh, kw)",
    )
    print("flattened for a Linear:", tuple(torch.flatten(activation, 1).shape))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note the padding of 1 with a 3x3 kernel keeps the image at 28x28 — that pairing is worth remembering, since `padding = (kernel_size - 1) // 2` preserves the size for any odd kernel. Only the `MaxPool2d` changes the spatial dimensions, from 28 down to 14.

    `Dropout` is in there too, and it does nothing at all in the printout above because the module has not been put in training mode. That behaviour is the subject of Part 4.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loss functions

    A loss function measures how predictions differ from their targets. By default, these four losses reduce the result to one scalar:

    ```python
    nn.CrossEntropyLoss()
    nn.BCEWithLogitsLoss()
    nn.MSELoss()
    nn.L1Loss()
    ```

    | Loss | Used for | Expects |
    | --- | --- | --- |
    | `CrossEntropyLoss` | Multi-class classification | Raw logits and, in our examples, integer class indices. |
    | `BCEWithLogitsLoss` | Binary or multi-label classification | Raw logits and floating-point targets, usually 0 or 1. |
    | `MSELoss` | Regression | Predicted values and matching targets. |
    | `L1Loss` | Regression | Predicted values and matching targets. |

    All four accept `reduction='mean'`, `'sum'` or `'none'`. With `'none'`, we get the individual loss values, which may be per sample or per element depending on the loss and input shape. This lets us apply our own weighting before reducing them to a scalar. As we saw in Part 2, calling `backward()` on a result with multiple elements requires an explicit `gradient` argument.

    For regression, MSE and L1 differ in how they penalise errors:

    $$
    \text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2,
    \qquad
    \text{L1} = \frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i-y_i|.
    $$

    Squaring the error gives large errors more influence. L1 is less sensitive to outliers: away from zero, its derivative with respect to each prediction has a constant magnitude, subject to the reduction used.

    For classification, we need to pay attention to what the loss expects. **Both classification losses above take logits: the model’s raw scores before conversion to probabilities.**

    `CrossEntropyLoss` combines log-softmax with negative log-likelihood. `BCEWithLogitsLoss` combines sigmoid with binary cross-entropy. These combined calculations improve numerical stability, avoiding the need to calculate probabilities and then take their logarithms separately.

    We should therefore pass the model’s logits directly to the loss:

    ```python
    logits = model(X)
    loss = loss_fn(logits, targets)
    ```

    Applying softmax before `CrossEntropyLoss` makes it interpret those probabilities as logits and apply log-softmax to them. This usually raises no error, but changes the loss and gradients and can make learning harder. The same mistake occurs when we apply sigmoid before `BCEWithLogitsLoss`.

    We can convert logits to probabilities afterwards when we need to interpret or display the predictions.
    """)
    return


@app.cell
def _(nn, torch):
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 3.0, 0.2], [0.1, 0.2, 4.0]])
    targets = torch.tensor([0, 1, 2])  # integer class indices, not one-hot

    ce = nn.CrossEntropyLoss()

    correct_use = ce(logits, targets)
    double_softmax = ce(torch.softmax(logits, dim=1), targets)

    print("logits straight in     :", round(correct_use.item(), 4), " <- right")
    print(
        "softmax applied first  :",
        round(double_softmax.item(), 4),
        " <- wrong, and silent",
    )
    print()
    print(
        "the loss is",
        round((double_softmax / correct_use).item(), 1),
        "times larger,",
    )
    print("but nothing raised and the shapes were fine")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also need to match the target format to the loss. For our multi-class examples, `CrossEntropyLoss` takes logits of shape `(batch, classes)` and class indices of shape `(batch,)`, stored as `torch.long`.

    It also supports floating-point targets of shape `(batch, classes)` representing class probabilities, including one-hot labels. Both formats are valid, but integer indices are usually simpler when each sample belongs to one class.

    For binary classification, `BCEWithLogitsLoss` expects floating-point targets with the same shape as the logits, usually containing 0 or 1. `PreTrainedModels/TransferLearning.py` uses this loss to distinguish dogs from non-dogs.
    """)
    return


@app.cell
def _(nn, torch):
    binary_logits = torch.tensor([2.5, -1.0, 0.3])
    binary_targets = torch.tensor([1.0, 0.0, 1.0])  # floats, same shape

    bce = nn.BCEWithLogitsLoss()
    print(
        "BCEWithLogitsLoss:",
        round(bce(binary_logits, binary_targets).item(), 4),
    )

    # the same thing spelled out, to show the sigmoid really is folded in
    manual = nn.BCELoss()(torch.sigmoid(binary_logits), binary_targets)
    print(
        "sigmoid + BCELoss:",
        round(manual.item(), 4),
        " <- same number, worse stability",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And the regression pair, on four predictions where three are close and one is badly wrong. Look at the gradients rather than the loss values, the gradient is what the optimiser acts on, so it is where the difference between the two actually bites.
    """)
    return


@app.cell
def _(nn, torch):
    for loss_name, loss_obj in [
        ("MSELoss", nn.MSELoss()),
        ("L1Loss", nn.L1Loss()),
    ]:
        reg_pred = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        reg_target = torch.tensor([1.2, 2.1, 2.9, 20.0])  # the last one is the outlier
        reg_loss = loss_obj(reg_pred, reg_target)
        reg_loss.backward()
        grads = reg_pred.grad
        share = grads[3].abs().item() / grads.abs().sum().item()
        print(
            f"{loss_name:8} loss {reg_loss.item():7.3f}  gradients {grads.numpy().round(3)}"
        )
        print(f"{'':8} the outlier accounts for {share:.1%} of the total gradient")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The outlier takes 97.6% of MSE's gradient and 25% of L1's — and 25% is exactly one quarter of four samples, because L1's gradient is `±1/n` for every sample no matter how wrong it is. That is the whole difference in one number. With MSE the model will spend its effort chasing that one point; with L1 it will fit the other three and leave it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [torch.compile](https://pytorch.org/docs/stable/generated/torch.compile.html)


    ```python
    torch.compile(model, mode='default', fullgraph=False, dynamic=None, backend='inductor')
    ```

    Traces the model and emits fused kernels, usually a solid speed-up on CUDA with no change in what the model computes. `MNIST/PyTorchDataLoaders.py:62` applies it straight after construction.

    Two practical notes. The first call is slow because that is when compilation happens, so a one-epoch benchmark will show it as a loss rather than a win. And it wraps the model, so `compiled_model.state_dict()` keys gain an `_orig_mod.` prefix — which will bite you when you try to load those weights into an uncompiled model. Part 5 deals with that.

    Compile may not always work on other non CUDA architetures such as mps.
    """)
    return


@app.cell
def _(nn, torch):
    plain = nn.Linear(4, 3)
    compiled = torch.compile(plain)

    print("plain state_dict keys   :", list(plain.state_dict().keys()))
    print("compiled state_dict keys:", list(compiled.state_dict().keys()))
    print()
    print("that prefix is what breaks load_state_dict later")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it

    Above we counted `TinyModel(4, 8, 2)` by hand and got 58. The general form for that shape is `(4h + h) + (2h + 2)`, or `7h + 2` — so the parameter count is linear in the hidden width, and the slider is the quickest way to convince yourself of that.

    Worth noticing where the values live. Widening the hidden layer adds to both `Linear` layers at once, because it is the `out_features` of one and the `in_features` of the next. That is the chaining rule from earlier, restated as arithmetic.
    """)
    return


@app.cell
def _(mo):
    width_slider = mo.ui.slider(
        2, 64, step=2, value=8, label="Hidden width", show_value=True
    )
    width_slider
    return (width_slider,)


@app.cell
def _(nn, width_slider):
    hidden = width_slider.value
    width_model = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    counted = sum(param.numel() for param in width_model.parameters())
    predicted = 7 * hidden + 2

    for layer_name, layer_param in width_model.named_parameters():
        print(
            f"  {layer_name:10} {str(tuple(layer_param.shape)):12} {layer_param.numel():>5}"
        )

    print()
    print(f"counted   {counted}")
    print(f"7h + 2    {predicted} at h = {hidden}")
    print("agree?   ", counted == predicted)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Build a model for MNIST: flatten, then 784 to 128, ReLU, 128 to 10. How many parameters? Check your arithmetic against `named_parameters()`.
    2. Take the collapse demo and add a single ReLU in only the middle position. Does the collapse still happen? Why not?
    3. Write the `TinyModel` above twice — once with `nn.Sequential` and once with a hand-written `forward` — and confirm they give the same output for the same seed.
    4. Take the conv block and work out on paper what a second identical block would produce, starting from `(64, 25, 14, 14)`. Then check.
    5. Train a three-class model twice, once passing logits to `CrossEntropyLoss` and once passing softmax output. Plot both loss curves.

    Part 4 covers what to do once the model is trained — evaluation, inference and turning logits into predictions.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
