import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction

    In this notebook we are going to train a simple [Recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network) (RNN) to convert text into a different format, in particular we will train out network to mimic text written by [William Shakespear](https://en.wikipedia.org/wiki/William_Shakespeare). This demo is based on Chapter 14 of [Hands On Machine Learning with Scikit Learn and Pytorch](https://www.oreilly.com/library/view/hands-on-machine-learning/9798341607972/) which is avaliable digitially in the library. This is in turn inspired by [this](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) blogpost.

    ## Dataset

    There is a simple dataset we can use for our training avaliable online, it contains about 25% of Shakespeare's work and is part of the original blog post. In this example we will download it via the Hugging Face api in this repository it is already installed, however if you need to use it in your own project you need to add it

    ```bash
    uv add datasets
    ```

    You will also need to have a hugging face loging and authorize it via the cli once.

    ```bash
    hf auth login
    ```

    And then you can load a dataset from the Hugging Face Hub using

    ```python
    from datasets import load_dataset

    dataset = load_dataset("username/my_dataset")

    # or load the separate splits if the dataset has train/validation/test splits
    train_dataset = load_dataset("username/my_dataset", split="train")
    valid_dataset = load_dataset("username/my_dataset", split="validation")
    test_dataset  = load_dataset("username/my_dataset", split="test")
    ```

    In our case the data set is "Trelis/tiny-shakespeare" so we can download using the following
    """)
    return


@app.cell
def _():
    from datasets import load_dataset

    data_set_name = "Trelis/tiny-shakespeare"
    train_dataset = load_dataset(data_set_name, split="train")
    test_dataset = load_dataset(data_set_name, split="test")

    train_dataset[:10]
    return test_dataset, train_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We as you can see the data is in the form of a json dictionary, index values with elements of text.

    ## How neural networks process text

    As we have seen in previous examples all a neural network wants to see is text. So we need to encode it into numbers. In general, this is done by splitting text into tokens. This can be done on different boundaries such as words, characters etc. We then assign an integer id to each token.
    """)
    return


@app.cell
def _(test_dataset, train_dataset):
    _train_text = "\n".join(train_dataset["Text"]).lower()
    _test_text = "\n".join(test_dataset["Text"]).lower()
    vocab = sorted(set(_train_text) | set(_test_text))
    "".join(vocab)
    return (vocab,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The call of ```lower()``` makes it easier to process the data as we don't need to worry about upper / lower case. I have also combined both the test and train data just in case there are extra tokens in one and not the other (in this case not!).

    We can now write a function to assign a token id to each character then use it to encode / decode our text.
    """)
    return


@app.cell
def _(vocab):
    char_to_id = {char: index for index, char in enumerate(vocab)}
    id_to_char = {index: char for index, char in enumerate(vocab)}
    print(f"{char_to_id['a']=} {id_to_char[13]=}")
    return char_to_id, id_to_char


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can use this to create tensors for our training.
    """)
    return


@app.cell
def _(char_to_id, id_to_char):
    import torch

    def encode_text(text: str) -> torch.tensor:
        return torch.tensor([char_to_id[char] for char in text.lower()])

    def decode_text(char_ids: torch.tensor) -> str:
        return "".join([id_to_char[char_id.item()] for char_id in char_ids])

    _encoded = encode_text("To be or Not to Be")
    print(_encoded)
    print(decode_text(_encoded))
    return decode_text, encode_text, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note the result is now lower case, be you can see how the data is converted into a tensor and back again.

    ## Building a Data loader.

    Now we have our encode function we can build a dataloader to process the input text, the main difference between this and previous dataloaders is that this is going to generate a moving window of text which the sequence to sequence RNN requires to do analysis of the data. The targets will be similar to the inputs, but shifted by one time step into the “future”.

    For example, one sample in the dataset may be a sequence of character IDs representing the text “to be or not to b” (without the final “e”), and the corresponding target—a sequence of character IDs representing the text “o be or not to be” (with the final “e”, but without the leading “t”).
    """)
    return


@app.cell
def _(encode_text):
    from torch.utils.data import Dataset, DataLoader

    class CharDataset(Dataset):
        def __init__(self, text, window_length):
            self.encoded_text = encode_text(text)
            self.window_length = window_length

        def __len__(self):
            return len(self.encoded_text) - self.window_length

        def __getitem__(self, idx):
            if idx >= len(self):
                raise IndexError("dataset index out of range")
            end = idx + self.window_length
            window = self.encoded_text[idx:end]
            target = self.encoded_text[idx + 1 : end + 1]
            return window, target

    return CharDataset, DataLoader


@app.cell
def _(CharDataset, decode_text):
    to_be_dataset = CharDataset("To be or not to be", window_length=10)
    for x, y in to_be_dataset:
        print(f"x={x}, y={y}")
        print(f"    decoded: x={decode_text(x)!r}, y={decode_text(y)!r}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can create some datasets from our actualy data and use it.
    """)
    return


@app.cell
def _(CharDataset, DataLoader, test_dataset, train_dataset):
    window_length = 50
    batch_size = 512  # reduce if your GPU cannot handle such a large batch size
    train_set = CharDataset("\n".join(train_dataset["Text"]), window_length)
    valid_set = CharDataset("\n".join(test_dataset["Text"]), window_length)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    return train_loader, valid_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [Embeddings](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html)

    We use nn.Embedding in a text RNN to turn each word or character ID into a vector of learnable numbers. These vectors become the inputs the RNN processes at each step. Suppose our vocabulary assigns these IDs:

    ```
    "cat" -> 0
    "dog" -> 1
    "car" -> 2
    ```

    The IDs are just labels. Feeding them directly as numerical values would introduce an arbitrary ordering: "car" is not twice "dog"!
    An embedding gives each token its own vector instead:

    ```
    "cat" -> 0 -> [ 0.2, -0.4,  0.7]
    "dog" -> 1 -> [ 0.3, -0.3,  0.6]
    "car" -> 2 -> [-0.8,  0.5, -0.1]
    ```

    These numbers are illustrative. By default, PyTorch initialises the embedding table randomly, and training updates its values alongside the RNN’s weights. The layer retrieves a row using the token ID. [See the PyTorch Embedding documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html).

    There are three reasons this is useful:
    - Learned features. Training can give tokens used in similar ways similar representations, helping the model share what it learns.
    - Compact inputs. With 10,000 tokens, a one hot representation needs 10,000 values per token. We might use an embedding with just 128.
    - Efficient lookup. We retrieve the token’s vector without constructing a large one-hot input.

    For example

    ```python
    import torch
    from torch import nn

    embedding = nn.Embedding(num_embeddings=10_000, embedding_dim=128)
    rnn = nn.RNN(input_size=128, hidden_size=256, batch_first=True)

    token_ids = torch.tensor([[12, 45, 9, 31]])

    vectors = embedding(token_ids)
    output, hidden = rnn(vectors)
    ```

    This will change the shape of the data to

    ```
    token IDs       [1, 4]         one sequence containing four tokens
    embeddings      [1, 4, 128]    128 values for each token
    RNN output      [1, 4, 256]    256 hidden-state values at each step
    ```


    For text generation, we usually add a linear layer that converts the RNN output into scores for the next token. We choose a token, pass its ID through the embedding, and continue generating.
    The embedding represents each token; the RNN builds up context from the sequence. An embedding is optional: an RNN can use one-hot inputs, and numerical time-series data can often go straight into the RNN.

    For our model we will use the following.
    """)
    return


@app.cell
def _(torch):
    import torch.nn as nn

    torch.manual_seed(42)
    embed = nn.Embedding(5, 3)  # 5 categories × 3D embeddings
    embed(torch.tensor([[3, 2], [0, 2]]))
    return (nn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Building the RNN

    We can now build our model as follows.
    """)
    return


@app.cell
def _(nn, torch, vocab):
    def get_device() -> torch.device:
        """
        Returns the appropriate device for the current environment.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():  # mac metal backend
            return torch.device("mps")
        else:
            return torch.device("cpu")

    class ShakespeareModel(nn.Module):
        def __init__(
            self,
            vocab_size,
            n_layers=2,
            embed_dim=10,
            hidden_dim=128,
            dropout=0.1,
        ):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.gru = nn.GRU(
                embed_dim,
                hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout,
            )
            self.output = nn.Linear(hidden_dim, vocab_size)

        def forward(self, X):
            embeddings = self.embed(X)
            outputs, _states = self.gru(embeddings)
            return self.output(outputs).permute(0, 2, 1)

    torch.manual_seed(42)
    _model = ShakespeareModel(len(vocab)).to(get_device())
    _model
    return ShakespeareModel, get_device


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This model learns to predict the next token in Shakespeare’s text. If vocab contains characters, it predicts the next character; if it contains words, it predicts the next word.
    The data passes through three layers:

    ```text
    Token IDs -> Embedding -> GRU -> Scores for each possible next token
    ```

    The model inputs are as follows

    | Argument | Meaning |
    |---|---|
    | `vocab_size` | Number of distinct tokens |
    | `n_layers=2` | Two GRU layers stacked on top of each other |
    | `embed_dim=10` | Represent each token using 10 learned values |
    | `hidden_dim=128` | Each GRU layer maintains a hidden state with 128 values |
    | `dropout=0.1` | During training, drop 10% of the values passed between GRU layers |

    A [GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit) (Gated Recurrent Unit) processes the sequence one token at a time. At each step, it combines the current input with its previous hidden state. Its learned gates control how much previous information to retain and how much to update.
    With two layers, the first GRU layer processes the embeddings, and the second processes the first layer’s outputs. Dropout applies between these layers, so it has no effect if n_layers=1.
    batch_first=True means the input dimensions are:

    ```text
    [batch size, sequence length, features]
    ```

    The linear layer then converts each 128-value GRU output into one score per vocabulary token (logits), higher score means the model favours that token, then permute(0, 2, 1) swaps the last two dimensions so that nn.CrossEntropyLoss can process it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training and Evaluation

    The training loop is almost the same as the one I used for the [Free Spoken Digits](../FreeSpokenDigits/FSDDMarimoPt3.py) CNN. The main difference is what counts as a "sample". The digits model made one prediction per recording. This model makes one prediction for *every character* in the window, so a batch of 512 windows of 50 characters gives 25,600 predictions. The `_Metrics` class therefore counts with `targets.numel()` rather than the batch size.

    The logits come out of the model as `[batch, vocab, sequence]` (this is why we did the `permute` in `forward`), and the targets are `[batch, sequence]`. [`CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) expects the class scores in dimension 1, so `argmax(1)` gives us the predicted character at each position.

    Accuracy here means "how often did the model guess the next character exactly". Don't expect it to get anywhere near the digits model. Even a person reading Shakespeare can't reliably guess the next letter!
    """)
    return


@app.cell
def _(mo, torch):
    class _Metrics:
        """Running totals for character-weighted loss and accuracy."""

        def __init__(self) -> None:
            self.total_loss, self.correct, self.count = 0.0, 0, 0

        def update(
            self,
            loss: torch.Tensor,
            logits: torch.Tensor,
            targets: torch.Tensor,
        ) -> None:
            n = targets.numel()
            self.count += n
            self.total_loss += loss.item() * n
            self.correct += (logits.argmax(1) == targets).sum().item()

        def result(self) -> tuple[float, float]:
            return self.total_loss / self.count, self.correct / self.count

    def train_epoch(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        device: torch.device,
    ) -> tuple[float, float]:
        """One pass over the data with weight updates; returns (loss, accuracy)."""
        model.train()
        metrics = _Metrics()
        for windows, targets in mo.status.progress_bar(
            loader, title="Batches", remove_on_exit=True
        ):
            windows, targets = windows.to(device), targets.to(device)
            optimiser.zero_grad()
            logits = model(windows)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimiser.step()
            metrics.update(loss, logits, targets)
        return metrics.result()

    @torch.inference_mode()
    def evaluate(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device,
    ) -> tuple[float, float]:
        """One pass over the data without gradients; returns (loss, accuracy)."""
        model.eval()
        metrics = _Metrics()
        for windows, targets in loader:
            windows, targets = windows.to(device), targets.to(device)
            logits = model(windows)
            metrics.update(loss_fn(logits, targets), logits, targets)
        return metrics.result()

    return evaluate, train_epoch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training the RNN

    The training text is about a million characters long, and every position is the start of a window, so one epoch is around 2,000 batches. On a lab GPU each epoch takes a few seconds; on a laptop CPU it will be *much* slower, so start with a couple of epochs to see if it's working. There is a progress bar for the batches in each epoch so you can tell it hasn't hung.

    As before, I keep the weights from the epoch with the lowest validation loss. Choose the settings and press **Train**; submitting again starts a fresh model.
    """)
    return


@app.cell
def _(mo):
    training_settings = mo.ui.dictionary(
        {
            "epochs": mo.ui.number(start=1, stop=100, value=10, label="Epochs"),
            "learning_rate": mo.ui.number(
                start=0.0001,
                stop=0.1,
                step=0.0001,
                value=0.001,
                label="Learning rate",
            ),
        }
    ).form(submit_button_label="Train")
    training_settings
    return (training_settings,)


@app.cell
def _(
    ShakespeareModel,
    evaluate,
    get_device,
    mo,
    torch,
    train_epoch,
    train_loader,
    training_settings,
    valid_loader,
    vocab,
):
    mo.stop(training_settings.value is None, mo.md("Press Train above to begin."))

    device = get_device()
    torch.manual_seed(42)
    model = ShakespeareModel(len(vocab)).to(device)

    _loss_fn = torch.nn.CrossEntropyLoss()
    _optimiser = torch.optim.Adam(
        model.parameters(), lr=training_settings.value["learning_rate"]
    )
    history = []
    _best_loss = float("inf")
    best_epoch = 0
    for _epoch in mo.status.progress_bar(
        range(training_settings.value["epochs"]), title="Training"
    ):
        _train_loss, _train_accuracy = train_epoch(
            model, train_loader, _loss_fn, _optimiser, device
        )
        _val_loss, _val_accuracy = evaluate(model, valid_loader, _loss_fn, device)
        history.append((_train_loss, _val_loss, _train_accuracy, _val_accuracy))
        if _val_loss < _best_loss:
            _best_loss = _val_loss
            best_epoch = _epoch + 1
            _best_weights = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(
            f"Epoch {_epoch + 1}: train loss {_train_loss:.3f}, validation loss {_val_loss:.3f}, validation accuracy {_val_accuracy:.1%}"
        )
    model.load_state_dict(_best_weights)
    mo.md(
        f"Training finished on **{device}**. Restored weights from epoch **{best_epoch}**."
    )
    return best_epoch, device, history, model


@app.cell
def _(history):
    import matplotlib.pyplot as plt

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3), layout="constrained")
    _epochs = range(1, len(history) + 1)
    for _index, _name in enumerate(["Loss", "Accuracy"]):
        for _offset, _label in enumerate(["Training", "Validation"]):
            _axes[_index].plot(
                _epochs,
                [_row[2 * _index + _offset] for _row in history],
                label=_label,
            )
        _axes[_index].set(xlabel="Epoch", ylabel=_name)
        _axes[_index].legend()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generating Text

    The dataset only has a train and test split, and we have already used the test split for validation, so a separate test score wouldn't tell us much. The real test of this model is to ask it to write some Shakespeare.

    We give it a prompt, take the scores for the *last* position, turn them into probabilities with softmax and sample the next character with [`torch.multinomial`](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html). That character is added to the text and the process repeats.

    The temperature divides the logits before the softmax. Low values (around 0.2) make the model pick the most likely character almost every time, which tends to get stuck repeating itself. High values (above 1.5) flatten the distribution and you get gibberish. Somewhere around 0.5 to 1.0 usually looks most like the real thing. The prompt can only use characters that are in `vocab`.
    """)
    return


@app.cell
def _(decode_text, encode_text, torch):
    @torch.inference_mode()
    def generate_text(
        model: torch.nn.Module,
        prompt: str,
        length: int,
        temperature: float,
        device: torch.device,
    ) -> str:
        """
        Extend a prompt one sampled character at a time.

        Parameters
        ----------
            model : torch.nn.Module
                the trained ShakespeareModel
            prompt : str
                the starting text, every character must be in vocab
            length : int
                how many characters to add
            temperature : float
                values below 1 sharpen the distribution, above 1 flatten it
            device : torch.device
                the device the model is on
        """
        model.eval()
        text = prompt.lower()
        for _ in range(length):
            ids = encode_text(text).unsqueeze(0).to(device)
            logits = model(ids)[0, :, -1]
            probabilities = torch.softmax(logits / temperature, dim=0)
            next_id = torch.multinomial(probabilities, num_samples=1)
            text += decode_text(next_id)
        return text

    return (generate_text,)


@app.cell
def _(mo):
    generation_settings = mo.ui.dictionary(
        {
            "prompt": mo.ui.text(value="To be or not to b", label="Prompt"),
            "length": mo.ui.number(start=10, stop=1000, value=300, label="Characters"),
            "temperature": mo.ui.slider(
                start=0.1, stop=2.0, step=0.1, value=0.8, label="Temperature"
            ),
        }
    ).form(submit_button_label="Generate")
    generation_settings
    return (generation_settings,)


@app.cell
def _(device, generate_text, generation_settings, mo, model):
    mo.stop(generation_settings.value is None, mo.md("Press Generate above."))
    _text = generate_text(model, device=device, **generation_settings.value)
    mo.md(f"```text\n{_text}\n```")
    return


@app.cell
def _(mo):
    ckpt_path = mo.ui.text(value="shakespeare.pt", label="checkpoint file")
    save_btn = mo.ui.run_button(label="Save model")
    mo.hstack([ckpt_path, save_btn], justify="start")
    return ckpt_path, save_btn


@app.cell
def _(
    best_epoch,
    ckpt_path,
    history,
    mo,
    model,
    save_btn,
    torch,
    training_settings,
    vocab,
):
    mo.stop(not save_btn.value)
    _path = mo.notebook_dir() / ckpt_path.value
    torch.save(
        {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "vocab": vocab,
            "best_epoch": best_epoch,
            "history": history,
            "settings": dict(training_settings.value),
            "torch_version": torch.__version__,
        },
        _path,
    )
    mo.md(f"Saved to `{_path}` (weights from epoch {best_epoch}).")
    return


if __name__ == "__main__":
    app.run()
