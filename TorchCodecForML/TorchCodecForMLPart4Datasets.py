import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TorchCodec for Machine Learning, Part 4: datasets and model inputs

    We will build a small manifest of video paths and labels, decode fixed-size clips on demand and pass a batch through a temporal model. Each video is generated locally, so no weights or dataset downloads are required.

    I use two motion directions as labels. These drawings are a way to check the pipeline, not a useful action-recognition benchmark. We will check gradients and a short training run without claiming that it generalises to real video.
    """)
    return


@app.cell
def _():
    import tempfile
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torchcodec
    from torchcodec.decoders import VideoDecoder
    from torchcodec.encoders import Encoder

    print("torch", torch.__version__, "torchcodec", torchcodec.__version__)
    return Encoder, Path, VideoDecoder, mo, plt, tempfile, torch


@app.cell
def _(Encoder, Path, plt, torch):
    def moving_frames(
        count: int = 24, reverse: bool = False, offset: int = 0
    ) -> torch.Tensor:
        """Draw a moving square so we can recognise the frame order."""
        frames = torch.full((count, 3, 64, 96), 24, dtype=torch.uint8)
        frames[:, :, ::8, :] = 42
        frames[:, :, :, ::8] = 42
        for index in range(count):
            x = 4 + (index * 3 + offset) % 72
            frames[index, 0, 22:38, x : x + 16] = 235
            frames[index, 1, 22:38, x : x + 16] = 155
            frames[index, 2, 22:38, x : x + 16] = 65
        return frames.flip(0) if reverse else frames

    def write_video(path: Path, frames: torch.Tensor, fps: int = 12) -> None:
        """Store RGB frames losslessly for the decoding experiments."""
        encoder = Encoder()
        stream = encoder.add_video(
            height=frames.shape[-2],
            width=frames.shape[-1],
            frame_rate=fps,
            codec="ffv1",
            pixel_format="bgr0",
            device="cpu",
        )
        with encoder.open_file(path):
            stream.add_frames(frames[: len(frames) // 2])
            stream.add_frames(frames[len(frames) // 2 :])

    def frame_grid(frames: torch.Tensor, titles: list[str]) -> plt.Figure:
        """Show CHW frames in their original colour range."""
        columns = min(4, len(frames))
        rows = (len(frames) + columns - 1) // columns
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(3 * columns, 2.5 * rows),
            squeeze=False,
            layout="constrained",
        )
        for axis in axes.flat:
            axis.axis("off")
        for axis, frame, title in zip(axes.flat, frames, titles, strict=False):
            axis.imshow(frame.cpu().permute(1, 2, 0).numpy(), interpolation="nearest")
            axis.set_title(title)
        plt.close(figure)
        return figure

    return frame_grid, moving_frames, write_video


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Split recordings before sampling clips

    If overlapping clips from one recording appear in both training and test sets, the test score can be misleading. We assign whole source files to a split first. Real datasets may need a stricter split by person, scene or recording session.

    Our manifest records `(path, label)`. Save the class mapping with a trained model so that output index zero keeps its meaning. Each file has a slightly different starting position and length.
    """)
    return


@app.cell
def _(Path, moving_frames, tempfile, write_video):
    media_directory = tempfile.TemporaryDirectory(prefix="torchcodec-dataset-")
    class_names = ["right", "left"]
    train_records = []
    test_records = []
    for _label, _class_name in enumerate(class_names):
        for _recording in range(3):
            _path = Path(media_directory.name) / f"{_class_name}-{_recording}.mkv"
            _frames = moving_frames(
                count=18 + 2 * _recording, reverse=bool(_label), offset=2 * _recording
            )
            write_video(_path, _frames)
            _record = (_path, _label)
            if _recording < 2:
                train_records.append(_record)
            else:
                test_records.append(_record)

    print(
        "training recordings:",
        len(train_records),
        "test recordings:",
        len(test_records),
    )
    print("class mapping:", dict(enumerate(class_names)))
    return class_names, test_records, train_records


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A Dataset decodes one item

    `__getitem__` constructs a decoder and requests four frames. It returns one `(channels, time, height, width)` clip and an integer label. Fixed temporal length and spatial size allow the default DataLoader to stack the items.

    We keep paths in the dataset rather than live decoder objects. This makes ownership clear and avoids sharing a decoder across workers. Reopening a decoder has a cost; for a larger application we would profile this and consider a bounded cache owned by each worker.

    The first regular clip is deterministic. Training could use random starts; validation should use a fixed policy. Very short videos use `repeat_last`, which is an explicit padding choice rather than extra motion evidence.
    """)
    return


@app.cell
def _(Path, VideoDecoder, torch):
    from torch.utils.data import DataLoader, Dataset
    from torchcodec.samplers import clips_at_regular_indices

    class ClipDataset(Dataset):
        """
        Decode one fixed-size clip from each manifest entry.

        Attributes
        ----------
        records : list[tuple[Path, int]]
            Source files and their class indices.
        """

        def __init__(self, records: list[tuple[Path, int]]) -> None:
            self.records = list(records)

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            path, label = self.records[index]
            decoder = VideoDecoder(path, device="cpu", num_ffmpeg_threads=1)
            clip = clips_at_regular_indices(
                decoder,
                num_clips=1,
                num_frames_per_clip=4,
                num_indices_between_frames=2,
                sampling_range_start=0,
                sampling_range_end=1,
                policy="repeat_last",
            ).data[0]
            frames = torch.nn.functional.interpolate(
                clip.float() / 255.0,
                size=(32, 48),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            return frames.permute(1, 0, 2, 3).contiguous(), label

    return ClipDataset, DataLoader


@app.cell
def _(ClipDataset, DataLoader, train_records):
    dataset = ClipDataset(train_records)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch_clips, batch_labels = next(iter(loader))
    print("batch:", tuple(batch_clips.shape), batch_clips.dtype)
    print("labels:", batch_labels.tolist())
    return batch_clips, batch_labels


@app.cell
def _(batch_clips, batch_labels, class_names, frame_grid):
    frame_grid(
        batch_clips.permute(0, 2, 1, 3, 4).flatten(0, 1),
        [
            f"{class_names[int(label)]}: frame {frame}"
            for label in batch_labels
            for frame in range(4)
        ],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A small temporal model

    A 3D convolution reads local neighbourhoods in time and space. The input axes are `(batch, channels, time, height, width)`; swapping channels and time can silently change the task when their sizes happen to match.

    We use random initial weights, a cross-entropy loss and a few optimisation steps on this one batch. This is an **overfit check**: can the model reduce training loss through our pipeline? It is not an evaluation. The plot and gradient norm help catch disconnected tensors or an incorrect training loop.
    """)
    return


@app.cell
def _(batch_clips, batch_labels, torch):
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = torch.nn.Sequential(
            torch.nn.Conv3d(3, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool3d((2, 2, 3)),
            torch.nn.Flatten(),
            torch.nn.Linear(8 * 2 * 2 * 3, 2),
        )
    model.train()
    _optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_history = []
    gradient_norm = 0.0
    for _step in range(30):
        _optimizer.zero_grad()
        _scores = model(batch_clips)
        _loss = torch.nn.functional.cross_entropy(_scores, batch_labels)
        _loss.backward()
        if _step == 0:
            gradient_norm = sum(
                float(p.grad.norm()) for p in model.parameters() if p.grad is not None
            )
        _optimizer.step()
        loss_history.append(float(_loss.detach()))
    model.eval()
    with torch.inference_mode():
        logits = model(batch_clips)
        final_loss = float(torch.nn.functional.cross_entropy(logits, batch_labels))
    initial_loss = loss_history[0]
    print("initial loss:", initial_loss, "final loss:", final_loss)
    print("first-step gradient norm:", gradient_norm)
    return loss_history, model


@app.cell
def _(loss_history, plt):
    _fig, _axis = plt.subplots(figsize=(8, 3), layout="constrained")
    _axis.plot(loss_history)
    _axis.set(
        xlabel="Optimisation step",
        ylabel="Cross-entropy",
        title="Overfitting one generated batch",
    )
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run the held-out recordings through the same preprocessing

    `eval()` switches model behaviour for evaluation, whilst `inference_mode()` disables gradient bookkeeping. They serve different purposes. We show predictions for two held-out files to check the path from source to output label. Two synthetic examples cannot establish generalisation.
    """)
    return


@app.cell
def _(ClipDataset, DataLoader, class_names, mo, model, test_records, torch):
    _test_loader = DataLoader(ClipDataset(test_records), batch_size=2, num_workers=0)
    _test_clips, _test_labels = next(iter(_test_loader))
    with torch.inference_mode():
        test_predictions = model(_test_clips).argmax(dim=1)
    mo.ui.table(
        [
            {
                "file": path.name,
                "target": class_names[label],
                "prediction": class_names[int(prediction)],
            }
            for (path, label), prediction in zip(
                test_records, test_predictions, strict=False
            )
        ],
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Moving beyond the notebook

    `num_workers=0` keeps execution in the notebook process. For worker processes, move the dataset class into an importable module and use a normal script entry point. Tune worker count and FFmpeg thread count together so that they do not compete for all CPU cores.

    CPU decoding followed by tensor transfer is a useful starting point. CUDA decoding requires a compatible TorchCodec build, FFmpeg and supported hardware; a PyTorch MPS device on a Mac is not a CUDA decoder. Measure on the deployment machine before choosing a different decoding path.

    ## Try it

    1. Switch training to random clip starts whilst keeping test sampling fixed.
    2. Add one two-frame recording. Inspect the repeated timestamps before trusting its motion label.
    3. Replace the generated files with your own labelled recordings, splitting by source before selecting clips.
    4. Reverse each clip in time and swap its label. Would a horizontal flip also require changing the label for this particular task?
    5. Extend the manifest with audio and use the same time interval for both decoders. Check stream start times before assuming that time zero aligns.

    See the [TorchCodec tutorials](https://meta-pytorch.org/torchcodec/stable/) for decoding transforms, CUDA and larger input pipelines.
    """)
    return


if __name__ == "__main__":
    app.run()
