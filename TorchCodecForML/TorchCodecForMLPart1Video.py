import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TorchCodec for Machine Learning, Part 1: video and tensors

    These four notebooks follow [TorchVisionForML](../TorchVisionForML/) and [TorchAudioForML](../TorchAudioForML/). We will turn media files into tensors, select clips and prepare batches for a model. I use a moving square and generated tones so we can check what came back from the decoder.

    A video file is more than a stack of images. It contains compressed data and timing information, and may contain several streams. [TorchCodec](https://meta-pytorch.org/torchcodec/stable/) handles the boundary between that file and PyTorch. We will keep the timestamps alongside the pixels.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running this notebook

    For ease each notebook makes its own media in a temporary directory and runs on the CPU. Nothing is downloaded by the notebook. If importing the decoders fails, check the [installation instructions](https://github.com/meta-pytorch/torchcodec#installing-torchcodec); installing the Python package alone may not provide the FFmpeg libraries. You may need to export the DYLD / LD library path to point to the correct libraries needed.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Containers, codecs and frames

    | Term | Meaning here |
    | --- | --- |
    | Container | A file structure, such as Matroska (`.mkv`), holding streams and metadata |
    | Codec | The method used to encode and decode a stream; we use FFV1 for lossless video |
    | Frame | One RGB image, with three colour channels |
    | Frame rate | Frames shown per second; our generated clip uses 12 fps |
    | PTS | Presentation timestamp: when a frame begins to appear |
    | Duration | How long that frame is displayed |

    A `.mp4` extension alone does not tell us which codec is inside. Some codecs store changes between frames, so reading one frame can require decoding earlier frames too.

    The helper below makes 24 frames, giving two seconds at 12 fps. [Encoder](https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.encoders.Encoder.html) lets us configure a stream and add frames in batches. Closing its context flushes the remaining data. FFV1 with an RGB-compatible pixel format lets us check an exact pixel round trip; lossy video would need a different comparison.
    """)
    return


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
    return decoder, source_frames, video_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [VideoDecoder](https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.decoders.VideoDecoder.html)

    ```python
    VideoDecoder(source, device="cpu", dimension_order="NCHW", seek_mode="exact")
    ```

    | Argument | Why we use it |
    | --- | --- |
    | `source` | A path here; encoded bytes are useful when data is already in memory |
    | `device` | CPU keeps this example independent of a CUDA installation |
    | `dimension_order` | The default puts colour channels before height and width |
    | `seek_mode` | Exact indexing scans the file to establish frame positions |

    `decoder[0]` returns a tensor. `get_frames_at` returns pixels **and** timing information in a `FrameBatch`. The four axes below are `(frames, channels, height, width)`. A `uint8` pixel is an integer from 0 to 255.
    """)
    return


@app.cell
def _(decoder, frame_grid, source_frames, torch):
    frames = decoder.get_frames_at(indices=[0, 6, 12, 18])
    decoded = decoder[:]
    print("batch", tuple(frames.data.shape), frames.data.dtype)
    print("exact pixel round trip", torch.equal(decoded, source_frames))
    frame_grid(frames.data, [f"PTS {float(t):.3f} s" for t in frames.pts_seconds])
    return (decoded,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which frame is on screen at a given time?

    The slider selects a presentation time. We ask for the frame visible then, which can begin slightly earlier. Frame intervals are half open: they include their start and exclude their end. We therefore keep the slider below the two-second endpoint.

    For this constant-rate clip we could estimate an index from `time * fps`. Real recordings may have variable frame rates or non-zero start times, so use the timestamp methods when the question is about time.
    """)
    return


@app.cell
def _(mo):
    play_time = mo.ui.slider(
        start=0.0,
        stop=1.9,
        step=0.1,
        value=0.7,
        label="Time (s)",
        show_value=True,
    )
    play_time
    return (play_time,)


@app.cell
def _(decoder, frame_grid, play_time):
    _selected = decoder.get_frame_played_at(seconds=play_time.value)
    frame_grid(
        _selected.data.unsqueeze(0),
        [f"Requested {play_time.value:.2f} s / PTS {_selected.pts_seconds:.3f} s"],
    )
    return


@app.cell
def _(decoder, mo):
    requested_times = [0.03, 0.55, 1.08, 1.77]
    timed_frames = decoder.get_frames_played_at(seconds=requested_times)
    mo.ui.table(
        [
            {
                "requested (s)": requested,
                "frame starts (s)": float(pts),
                "duration (s)": float(duration),
            }
            for requested, pts, duration in zip(
                requested_times,
                timed_frames.pts_seconds,
                timed_frames.duration_seconds,
            )
        ],
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Encoded bytes are not image pixels

    We can read the file into memory and construct another decoder. The encoded byte sequence has no image axes; it must be decoded before we can treat it as a picture. This is useful with an object store, but reading a whole large video into RAM defeats lazy file access.

    Our tiny clip is safe to decode in full. For a long recording, request only the frames needed for the current batch.
    """)
    return


@app.cell
def _(VideoDecoder, decoded, torch, video_path):
    _encoded_bytes = video_path.read_bytes()
    from_bytes = VideoDecoder(_encoded_bytes, device="cpu")[0]
    print("encoded file:", len(_encoded_bytes), "bytes")
    print(
        "all decoded pixels:",
        decoded.numel() * decoded.element_size(),
        "bytes",
    )
    print("same first frame:", torch.equal(from_bytes, decoded[0]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it

    1. Request the same index twice with `get_frames_at`. Check the two tensors and timestamps.
    2. Change `frame_rate` in `write_video` to 24. Predict the duration before rerunning. Adjust the time controls to remain inside the shorter clip.
    3. Use `dimension_order="NHWC"` and inspect the shape. Which permutation would the plotting code now need?
    4. Compare `decoder[0:12:3]` with `get_frames_in_range(start=0, stop=12, step=3)`.

    Next: [sampling clips](TorchCodecForMLPart2Sampling.py).
    """)
    return


if __name__ == "__main__":
    app.run()
