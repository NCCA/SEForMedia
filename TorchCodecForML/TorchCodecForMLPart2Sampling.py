import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TorchCodec for Machine Learning, Part 2: sampling clips

    A model rarely needs every frame in a recording. We will choose short sequences, inspect their timestamps and convert the resulting tensors into a model input. This notebook generates its own two-second clip, so Part 1 does not need to be running.

    I use regular samples for a repeatable view of the clip and random samples to show how training examples can vary. Sampling is a choice about which evidence the model sees.
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
        for axis, frame, title in zip(axes.flat, frames, titles):
            axis.imshow(frame.cpu().permute(1, 2, 0).numpy(), interpolation="nearest")
            axis.set_title(title)
        plt.close(figure)
        return figure

    return frame_grid, moving_frames, write_video


@app.cell
def _(Path, VideoDecoder, moving_frames, tempfile, write_video):
    media_directory = tempfile.TemporaryDirectory(prefix="torchcodec-video-")
    video_path = Path(media_directory.name) / "moving-square.mkv"
    source_frames = moving_frames()
    write_video(video_path, source_frames)
    decoder = VideoDecoder(video_path, device="cpu", seek_mode="exact")
    print(decoder.metadata)
    return (decoder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [Index samplers](https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.samplers.clips_at_regular_indices.html)

    ```python
    clips_at_regular_indices(decoder, num_clips=3, num_frames_per_clip=4,
                             num_indices_between_frames=2)
    ```

    `num_clips` counts sequences; `num_frames_per_clip` counts images within each sequence. A stride of two reads every other frame. Four such frames cover six index intervals from first to last, rather than eight.

    The result has shape `(clips, time, channels, height, width)`. Its timestamps have shape `(clips, time)`. Flattening the first two axes is convenient for a contact sheet, but a temporal model needs us to preserve which frames belong together.
    """)
    return


@app.cell
def _(decoder):
    from torchcodec.samplers import (
        clips_at_random_indices,
        clips_at_regular_indices,
        clips_at_regular_timestamps,
    )

    regular = clips_at_regular_indices(
        decoder,
        num_clips=3,
        num_frames_per_clip=4,
        num_indices_between_frames=2,
    )
    print("clips", tuple(regular.data.shape))
    print("timestamps", regular.pts_seconds)
    return (
        clips_at_random_indices,
        clips_at_regular_indices,
        clips_at_regular_timestamps,
        regular,
    )


@app.cell
def _(frame_grid, regular):
    frame_grid(
        regular.data.flatten(0, 1),
        [
            f"Clip {clip + 1}: {float(t):.3f} s"
            for clip, row in enumerate(regular.pts_seconds)
            for t in row
        ],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random sampling and repeatability

    A fixed seed allows us to repeat an experiment. `fork_rng` restores the surrounding random state afterwards, so this demonstration does not reset every random operation in the notebook. During training we would normally let sampling vary between epochs.

    Regular sampling is a useful validation policy. Changing the validation clips randomly makes it harder to tell whether the model improved.
    """)
    return


@app.cell
def _(clips_at_random_indices, decoder, torch):
    with torch.random.fork_rng():
        torch.manual_seed(7)
        random_a = clips_at_random_indices(decoder, num_clips=3, num_frames_per_clip=4)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        random_b = clips_at_random_indices(decoder, num_clips=3, num_frames_per_clip=4)
    print("same seed, same clips:", torch.equal(random_a.data, random_b.data))
    print(random_a.pts_seconds)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [Sampling in seconds](https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.samplers.clips_at_regular_timestamps.html)

    Index spacing changes its meaning when frame rates differ. For example, six frames are half a second at 12 fps but a quarter-second at 24 fps. Timestamp sampling expresses the gap in seconds instead.

    This API takes `seconds_between_clip_starts`, rather than a clip count. The requested sampling times and the returned frame start times need not match exactly: a frame covers an interval. Closely spaced requests can even select the same image.
    """)
    return


@app.cell
def _(clips_at_regular_timestamps, decoder, plt, regular):
    timed = clips_at_regular_timestamps(
        decoder,
        seconds_between_clip_starts=0.5,
        num_frames_per_clip=4,
        seconds_between_frames=0.2,
        sampling_range_start=0.0,
        sampling_range_end=1.1,
    )
    _fig, _axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True, layout="constrained")
    for _axis, _batch, _title in zip(
        _axes, [regular, timed], ["Index spacing", "Time spacing"]
    ):
        for _row, _pts in enumerate(_batch.pts_seconds):
            _axis.plot(_pts.numpy(), [_row + 1] * len(_pts), "o-")
        _axis.set(ylabel="Clip", title=_title, yticks=[1, 2, 3])
    _axes[-1].set_xlabel("Presentation time (s)")
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What happens at the end?

    We deliberately start at frame 22 of a 24-frame video and ask for four frames. The valid part is `[22, 23]`.

    | Policy | Resulting indices |
    | --- | --- |
    | `repeat_last` | 22, 23, 23, 23 |
    | `wrap` | 22, 23, 22, 23 |
    | `error` | Reject the request |

    Wrapping repeats the valid portion of the sampled clip. Padding preserves shape but changes the motion a model sees. With the default sampling range, the sampler normally avoids overrunning; we set the range explicitly to explore the policies.
    """)
    return


@app.cell
def _(clips_at_regular_indices, decoder):
    _tail = dict(
        num_clips=1,
        num_frames_per_clip=4,
        sampling_range_start=22,
        sampling_range_end=23,
    )
    repeat = clips_at_regular_indices(decoder, **_tail, policy="repeat_last")
    wrapped = clips_at_regular_indices(decoder, **_tail, policy="wrap")
    policy_error = ""
    try:
        clips_at_regular_indices(decoder, **_tail, policy="error")
    except ValueError as _error:
        policy_error = str(_error)
    print("repeat:", repeat.pts_seconds)
    print("wrap:", wrapped.pts_seconds)
    print("error:", policy_error)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From clips to model inputs

    Our model in Part 4 expects `(batch, channels, time, height, width)`. We first convert the byte pixels to floating point in `[0, 1]`, resize each frame, then move the time axis. `reshape` changes how dimensions are grouped; `permute` changes their order.

    All frames receive the same resize. A random spatial crop should also use the same crop across a clip, otherwise we introduce artificial camera motion. A pre-trained model may additionally require a specific size, frame sampling policy and channel normalisation; follow its weights' preprocessing instructions.
    """)
    return


@app.cell
def _(regular, torch):
    _float_frames = regular.data.float() / 255.0
    _small = torch.nn.functional.interpolate(
        _float_frames.flatten(0, 1),
        size=(32, 48),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    model_input = _small.reshape(3, 4, 3, 32, 48).permute(0, 2, 1, 3, 4).contiguous()
    print("model input:", tuple(model_input.shape), model_input.dtype)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it

    1. Increase the index stride from two to four. Predict the first-to-last time span.
    2. Set the timestamp gap to 0.02 seconds. Why do timestamps repeat at 12 fps?
    3. Sample a clip longer than the video and compare the three end policies.
    4. Reverse the time axis of one clip. Does its average image change? Would a motion classifier lose information if it averaged frames first?

    Next: [audio decoding and encoding](TorchCodecForMLPart3Audio.py).
    """)
    return


if __name__ == "__main__":
    app.run()
