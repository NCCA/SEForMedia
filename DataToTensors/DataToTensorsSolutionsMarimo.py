#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # From media data to tensors: worked solutions

    These are worked answers to the nine exercises in
    [From media data to tensors](DataToTensorsMarimo.py). I have kept the
    examples small so we can inspect individual values as well as shapes.
    They run independently of the original notebook, on the CPU, without
    downloading data or training a model.

    For each answer we will identify the axes, keep targets separate from
    inputs and check that the values still mean what we intended. The last
    two exercises have several valid answers; I use one facial rig contract
    and the MNIST dataset to make the choices concrete.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.nn.utils.rnn import pad_sequence
    from torchaudio.functional import resample

    print("marimo", mo.__version__)
    print("PyTorch", torch.__version__)
    return F, mo, nn, pad_sequence, resample, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Read the same RGB pixel in two layouts

    The index expressions are `image_hwc[row, column, :]` and
    `image_chw[:, row, column]`. In HWC the last axis selects colour; in CHW
    the first axis does. Row is height and column is width.

    I have reproduced the original 3-row, 5-column image. At row 1, column 2
    we expect the 8-bit RGB values `[80, 80, 64]`. Both tensors below retain
    `uint8` values so we can compare them directly. If we also convert to
    floating point and divide by 255, we must account for that change in
    values when comparing with the source.
    """)
    return


@app.cell
def _(torch):
    image_hwc = torch.zeros(3, 5, 3, dtype=torch.uint8)
    image_hwc[..., 0] = torch.arange(5) * 40
    image_hwc[..., 1] = torch.arange(3)[:, None] * 80
    image_hwc[..., 2] = 64
    image_chw = image_hwc.permute(2, 0, 1)
    print("HWC pixel:", image_hwc[1, 2, :])
    print("CHW pixel:", image_chw[:, 1, 2])
    assert torch.equal(image_hwc[1, 2, :], torch.tensor([80, 80, 64]))
    assert torch.equal(image_hwc[1, 2, :], image_chw[:, 1, 2])

    _own_hwc = torch.arange(2 * 7 * 3, dtype=torch.uint8).reshape(2, 7, 3)
    _own_chw = _own_hwc.permute(2, 0, 1)
    print("Own image HWC / CHW:", tuple(_own_hwc.shape), tuple(_own_chw.shape))
    print("Own pixel at row 1, column 4:", _own_chw[:, 1, 4])
    assert torch.equal(_own_hwc[1, 4, :], _own_chw[:, 1, 4])
    assert _own_chw[:, 1, 4].tolist() == [33, 34, 35]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Greyscale classification and segmentation

    For this example `B=2` images, `C=1` intensity channel, `H=3` rows,
    `W=5` columns and `K=3` output classes.

    | Task | Inputs | Model output | Target |
    | --- | --- | --- | --- |
    | Classification | `(B,1,H,W)` float32 | `(B,K)` logits | `(B,)` int64 class IDs |
    | Segmentation | `(B,1,H,W)` float32 | `(B,K,H,W)` logits | `(B,H,W)` int64 class IDs |

    The input can stay the same. What changes is the question: one class
    for the image becomes one class for each pixel. A greyscale image can
    still contain several semantic classes. Its one input channel does
    not limit the number of output classes.

    The heads below are untrained examples that produce the required
    shapes. Cross entropy takes the logits directly; we do not apply a
    softmax first. The segmentation labels are invented for this exercise.
    """)
    return


@app.cell
def _(F, nn, torch):
    grey_images = torch.linspace(0, 1, 30).reshape(2, 1, 3, 5)
    _class_targets = torch.tensor([0, 2], dtype=torch.int64)
    _pixel_targets = torch.arange(30).reshape(2, 3, 5) % 3
    with torch.random.fork_rng():
        torch.manual_seed(2)
        _classifier = nn.Sequential(nn.Flatten(), nn.Linear(15, 3))
        _segmenter = nn.Conv2d(1, 3, kernel_size=1)
    _class_logits = _classifier(grey_images)
    _pixel_logits = _segmenter(grey_images)
    assert _class_logits.shape == (2, 3)
    assert _pixel_logits.shape == (2, 3, 3, 5)
    print("Inputs:", tuple(grey_images.shape), grey_images.dtype)
    print("Classification output / target:", _class_logits.shape, _class_targets.shape)
    print("Segmentation output / target:", _pixel_logits.shape, _pixel_targets.shape)
    print("Classification loss:", F.cross_entropy(_class_logits, _class_targets).item())
    print("Segmentation loss:", F.cross_entropy(_pixel_logits, _pixel_targets).item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. RGB, depth and a validity mask

    I assume registered RGB and depth images: the same row and column refer
    to the same camera ray. RGB is float32 in `[0,1]`, depth is float32 in
    metres, and validity is boolean. Here depth means distance along the
    camera's forward axis; a sensor reporting radial range would need a
    different contract. The class target remains separate.

    For a model accepting five channels, we can concatenate RGB, scaled
    depth and a floating-point copy of the mask as `(B,5,H,W)`. Keeping
    them in a dictionary is also valid. The mask carries no physical unit;
    depth has units before scaling. RGB divided by 255 is a dimensionless
    encoded colour value, and is still sRGB if that was the source encoding.

    I divide valid depth by a fixed 10-metre reference distance. This makes
    it dimensionless; it only lies in `[0,1]` if depths are within 10 metres.
    Invalid depth is filled with zero **after** checking validity. The mask
    distinguishes that fill from a real measurement. For standardisation,
    fit statistics on valid training pixels and save them for inference.
    """)
    return


@app.cell
def _(torch):
    _rgb = torch.linspace(0, 1, 90).reshape(2, 3, 3, 5)
    _depth_metres = torch.linspace(0.5, 5, 30).reshape(2, 1, 3, 5)
    _depth_metres[0, 0, 1, 2] = float("nan")
    _depth_valid = torch.isfinite(_depth_metres) & (_depth_metres > 0)
    _scaled_depth = torch.where(_depth_valid, _depth_metres / 10, 0.0)
    _inputs = torch.cat((_rgb, _scaled_depth, _depth_valid.float()), dim=1)
    _targets = torch.tensor([0, 2], dtype=torch.int64)
    assert _inputs.shape == (2, 5, 3, 5)
    assert torch.isfinite(_inputs).all()
    assert _inputs[0, 3, 1, 2] == 0 and _inputs[0, 4, 1, 2] == 0
    print(
        "RGB / scaled depth / mask:",
        _rgb.shape,
        _scaled_depth.shape,
        _depth_valid.shape,
    )
    print("Combined inputs:", _inputs.shape, "separate targets:", _targets.shape)
    print("Pixel with absent depth:", _inputs[0, :, 1, 2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Encode video frames, then restore time

    We start with `(B,T,C,H,W)` where `B=2` clips, `T=4` frames, `C=3`
    channels, `H=6` rows and `W=8` columns. Combining the batch and time
    axes gives `(B*T,C,H,W)`, so the same 2D encoder processes each frame.
    After spatial pooling we have `(B*T,F)` with `F=5` features per frame.
    Restoring the first two axes gives `(B,T,F)`.

    This preserves frame order, but the 2D encoder has not modelled any
    relationships between frames. A temporal classifier needs to combine
    information across time. Here a small GRU consumes the sequence and
    its final hidden state produces one set of class logits per clip.
    A temporal convolution or a Transformer with time information is
    another choice. Mean pooling alone loses order.

    Unequal clip lengths also need masks or packed sequences; taking the
    final padded frame would otherwise use padding as if it were a pose
    or image. These clips all have four valid frames.
    """)
    return


@app.cell
def _(nn, torch):
    _b, _t, _c, _h, _w = 2, 4, 3, 6, 8
    _video = torch.linspace(0, 1, _b * _t * _c * _h * _w).reshape(_b, _t, _c, _h, _w)
    with torch.random.fork_rng():
        torch.manual_seed(4)
        _encoder = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
        _temporal = nn.GRU(input_size=5, hidden_size=7, batch_first=True)
        _head = nn.Linear(7, 3)
    _frames = _video.reshape(_b * _t, _c, _h, _w)
    _flat_features = _encoder(_frames)
    _sequence = _flat_features.reshape(_b, _t, 5)
    _temporal_output, _hidden = _temporal(_sequence)
    _logits = _head(_hidden[-1])
    assert torch.equal(_frames[4], _video[1, 0])
    torch.testing.assert_close(_sequence[1, 0], _encoder(_video[1, :1])[0])
    assert _logits.shape == (2, 3)
    print("Frames:", _frames.shape, "encoded frames:", _flat_features.shape)
    print("Restored (B,T,F):", _sequence.shape)
    print("Temporal output:", _temporal_output.shape, "clip logits:", _logits.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Double the audio sample rate

    The original stereo example lasts 0.25 seconds at 8000 Hz, giving
    `(C,S) = (2,2000)`. At 16000 Hz the same duration needs `(2,4000)`.
    A batch would add a leading batch axis.

    Resampling estimates new samples on the new time grid. The bandlimited
    resampler below keeps approximately the same sound, duration and pitch
    (small boundary differences are expected). Upsampling does not recover
    frequencies missing from the original recording. Downsampling also
    requires low-pass filtering to avoid aliasing.

    Changing only the metadata leaves 2000 samples. Playing these at
    16000 Hz takes 0.125 seconds and doubles the apparent frequencies:
    440 Hz becomes 880 Hz. That is a playback-rate change.
    """)
    return


@app.cell
def _(resample, torch):
    _source_rate, _target_rate = 8000, 16000
    _times = torch.arange(2000, dtype=torch.float32) / _source_rate
    _waveform = torch.stack(
        (
            0.5 * torch.sin(2 * torch.pi * 440 * _times),
            0.5 * torch.sin(2 * torch.pi * 660 * _times),
        )
    )
    _resampled = resample(_waveform, orig_freq=_source_rate, new_freq=_target_rate)
    assert _resampled.shape == (2, 4000)
    assert _waveform.shape[-1] / _source_rate == _resampled.shape[-1] / _target_rate
    print(
        "Original shape / seconds:", _waveform.shape, _waveform.shape[-1] / _source_rate
    )
    print(
        "Resampled shape / seconds:",
        _resampled.shape,
        _resampled.shape[-1] / _target_rate,
    )
    print(
        "Metadata-only shape / seconds:",
        _waveform.shape,
        _waveform.shape[-1] / _target_rate,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Pad motion clips and exclude missing coordinates

    I use two clips of three and five frames, with two joints and three
    coordinates per joint. Padding produces `(B,Tmax,J,3) = (2,5,2,3)`.
    Frame validity is `(B,Tmax)`; joint validity is `(B,Tmax,J)`.
    Here `True` means an observation is available.

    For a mean squared error **per observed coordinate**, the denominator
    counts observed scalar coordinates:

    $$L = \frac{\sum_{b,t,j,c} m_{btjc}(\hat y_{btjc}-y_{btjc})^2}
                    {\sum_{b,t,j,c}m_{btjc}}.$$

    Expanding the combined frame/joint mask along XYZ makes the denominator
    explicit. We have eight valid frames, two joints and three coordinates:
    48 scalar observations. Removing one joint at one valid frame leaves
    45. This weights each observed coordinate equally; averaging clip means
    would instead give each clip equal weight regardless of its length.

    Select valid entries before subtracting so missing targets stored as
    NaN never enter the loss. Multiplying a NaN loss by zero would still
    give NaN. A batch with no observed coordinates needs an explicit policy;
    I skip its loss calculation here.
    """)
    return


@app.cell
def _(pad_sequence, torch):
    _clips = [torch.zeros(3, 2, 3), torch.zeros(5, 2, 3)]
    _lengths = torch.tensor([len(_clip) for _clip in _clips])
    _targets = pad_sequence(_clips, batch_first=True)
    _frame_valid = torch.arange(_targets.shape[1])[None, :] < _lengths[:, None]
    _predictions = torch.ones_like(_targets)
    _predictions[~_frame_valid] = 100.0
    _frame_coordinates = _frame_valid[:, :, None, None].expand_as(_targets)
    _frame_loss = (
        (_predictions[_frame_coordinates] - _targets[_frame_coordinates])
        .square()
        .mean()
    )
    assert _frame_coordinates.sum() == 48
    torch.testing.assert_close(_frame_loss, torch.tensor(1.0))

    _joint_valid = torch.ones(_targets.shape[:3], dtype=torch.bool)
    _joint_valid[1, 2, 1] = False
    _targets[1, 2, 1] = float("nan")
    _observed = (_frame_valid[:, :, None] & _joint_valid)[..., None].expand_as(_targets)
    _count = _observed.sum()
    if _count > 0:
        _loss = (_predictions[_observed] - _targets[_observed]).square().sum() / _count
        torch.testing.assert_close(_loss, torch.tensor(1.0))
        print("Observed coordinates:", _count.item(), "masked MSE:", _loss.item())
    else:
        print("No observed coordinates: skip this batch.")
    assert _count == 45
    print("Padded targets:", _targets.shape, "frame validity:\n", _frame_valid)
    print(
        "MSE without a padding mask:",
        (_predictions - torch.nan_to_num(_targets)).square().mean().item(),
    )
    print("MSE excluding padded frames:", _frame_loss.item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Pack meshes with their vertex labels

    I use the tetrahedron and triangle from the lesson. Each mesh has its
    own per-vertex class labels drawn from a shared class vocabulary. These
    are targets, so they stay separate from the position inputs.

    Concatenate positions and labels in the same mesh order. Add the number
    of preceding vertices to each mesh's face indices. Labels are class
    IDs, so we do not add a vertex offset to them. Keep vertex and face
    boundaries to recover individual meshes, including their local indices.

    Checking only the largest index is insufficient: a face might point
    into the wrong mesh whilst staying within the packed tensor. We check
    each mesh's permitted vertex interval, triangle positions and labels.
    """)
    return


@app.cell
def _(torch):
    _vertices = [
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        torch.tensor([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]]),
    ]
    _faces = [
        torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
        torch.tensor([[0, 1, 2]]),
    ]
    _labels = [torch.tensor([0, 0, 1, 1]), torch.tensor([2, 1, 2])]
    _vertex_offsets = torch.tensor([0, 4, 7])
    _face_offsets = torch.tensor([0, 4, 5])
    _packed_vertices = torch.cat(_vertices)
    _packed_faces = torch.cat(
        [_f + _vertex_offsets[_i] for _i, _f in enumerate(_faces)]
    )
    _packed_labels = torch.cat(_labels)
    _vertex_mesh_ids = torch.repeat_interleave(torch.arange(2), torch.tensor([4, 3]))
    for _i in range(2):
        _start, _end = _vertex_offsets[_i : _i + 2].tolist()
        _face_start, _face_end = _face_offsets[_i : _i + 2].tolist()
        _global_faces = _packed_faces[_face_start:_face_end]
        assert ((_global_faces >= _start) & (_global_faces < _end)).all()
        assert torch.equal(
            _vertex_mesh_ids[_global_faces], torch.full_like(_global_faces, _i)
        )
        torch.testing.assert_close(
            _packed_vertices[_global_faces], _vertices[_i][_faces[_i]]
        )
        assert torch.equal(_packed_labels[_global_faces], _labels[_i][_faces[_i]])
        assert torch.equal(_global_faces - _start, _faces[_i])
        print(
            f"Mesh {_i}: vertices [{_start}:{_end}], faces [{_face_start}:{_face_end}]"
        )
        print("Triangle positions:\n", _packed_vertices[_global_faces])
        print("Triangle vertex labels:\n", _packed_labels[_global_faces])
    print("Packed vertex labels:", _packed_labels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. A contract for speech-driven facial animation

    I will predict facial blendshape weights at 60 Hz from mono speech at
    16 kHz. This version is causal: an output at time `t` uses only audio
    available up to `t`. An offline version could use future context, but
    that lookahead would need to be part of its declared latency.

    | Field | Contract |
    | --- | --- |
    | Sample | One identified take and a half-open interval `[start,end)` on a shared clock |
    | Audio input | `(B,1,S)` float32 waveform; amplitude is dimensionless, with saved decoding/gain convention; 16000 samples per second |
    | Optional initial state | `(B,A)` blendshape weights measured before the first output, only if the same state is available at inference |
    | Model output and target | `(B,T,A)` float32 blendshape weights at 60 Hz; targets are separate from audio inputs |
    | Blendshape order | `A=4`: `jawOpen`, `mouthClose`, `mouthSmileLeft`, `mouthSmileRight`; the real rig must save its full named order and version |
    | Weight units | Dimensionless; for this rig `[0,1]` means neutral to full activation; other rigs can use different ranges |
    | Rig metadata | Neutral mesh, blendshape deltas, vertex correspondence, topology, rig version and coordinate convention; mesh deltas in metres |
    | Time metadata | Audio sample rate, output timestamps, clock origin, measured audio/animation offset and any drift correction |
    | Masks | Audio validity `(B,S)` and coefficient validity `(B,T,A)`; lengths distinguish padding, masks also mark internal dropouts |
    | Scaling | Save training-only statistics if used; never normalise using a future portion of the current take for this causal task |
    | Split | Hold speakers out before extracting windows to test new-speaker performance; keep each source take and its windows in one split |

    At 60 Hz there are `16000/60` audio samples per animation frame, which
    is not an integer. We use timestamps, rather than repeatedly advancing
    by 267 samples and accumulating drift. In the example, each frame time
    maps to the latest available audio sample using `floor(t*sample_rate)`.
    A model still needs an audio encoder and a causal alignment policy;
    these indices only demonstrate the clock mapping.

    Future leakage includes centred audio windows, bidirectional encoders,
    unmasked attention over later audio and normalisation over the complete
    take. Ground-truth blendshape weights or velocities derived from a
    later animation frame also reveal unavailable information if fed into
    the model. Whole-take alignment estimated from later motion would need
    replacing with calibration available at inference. Training targets
    can contain later observations; the restriction concerns what reaches
    the prediction inputs. Overlapping windows crossing splits cause a
    separate train/test leak.
    """)
    return


@app.cell
def _(torch):
    _blendshape_names = ("jawOpen", "mouthClose", "mouthSmileLeft", "mouthSmileRight")
    _audio_rate, _animation_rate = 16000, 60
    _audio = torch.zeros(2, 1, 16000)
    _targets = torch.zeros(2, 60, len(_blendshape_names))
    _frame_times = torch.arange(60, dtype=torch.float64) / _animation_rate
    _latest_sample = torch.floor(_frame_times * _audio_rate).long()
    _audio_times = torch.arange(16000, dtype=torch.float64) / _audio_rate
    _allowed_audio = _audio_times[None, :] <= _frame_times[:, None]
    assert _allowed_audio.shape == (60, 16000)
    assert _allowed_audio[0].sum() == 1
    assert (_audio_times[_latest_sample] <= _frame_times).all()
    print("Audio / animation target:", _audio.shape, _targets.shape)
    print("Blendshape order:", _blendshape_names)
    print("First frame times:", _frame_times[:6])
    print("Latest available sample indices:", _latest_sample[:6])
    print("Sample advances:", _latest_sample.diff()[:6])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. A real dataset contract: MNIST

    This answer depends on the dataset you choose. I use the MNIST digit
    classification task from the [existing MNIST lesson](../MNIST/TheMNISTDataSetMarimo.py).
    We write the contract first, then inspect the real data when building
    the dataset. The executable example below uses a synthetic record to
    check the contract without downloading MNIST; it is not a real digit.

    | Item | Shape and meaning |
    | --- | --- |
    | Decoded source image | `(28,28)` uint8; rows then columns; intensity `[0,255]` |
    | One model input | `(1,28,28)` float32; channel, row, column; intensity `[0,1]` |
    | One target | Scalar int64 class ID in `0..9`; separate from the image |
    | Batched inputs | `(B,1,28,28)`; `B` independent digit images |
    | Batched logits | `(B,10)`; ten unnormalised scores in digit order `0..9` |
    | Batched targets | `(B,)` int64; one digit label per image |

    Keep source split, original record index, source file/version, digit
    vocabulary and preprocessing settings as metadata. The record index is
    only unique when combined with its source split. Preserve intensity
    polarity (bright strokes on a dark background), resize/orientation
    rules and any fitted normalisation statistics with the model. MNIST
    has no audio clock, physical pixel size or mesh topology to invent.

    Keep the official test split for final evaluation. Create validation
    data from the training split with a recorded seed and index lists,
    then apply random augmentation only to training inputs. Fit any
    mean/std statistics on the resulting training subset. Dividing by 255
    is a fixed conversion and requires no fitted statistics. Do not claim
    a new-writer evaluation unless writer identities support that split.

    Before implementing `Dataset`, I would verify a few decoded records
    against their labels and check that batching adds just the leading
    axis. For a different dataset, replace this contract with its actual
    axes, target and metadata rather than forcing it into image shapes.
    """)
    return


@app.cell
def _(torch):
    _source = torch.zeros(28, 28, dtype=torch.uint8)
    _source[5:23, 12:16] = 255
    _input = _source.unsqueeze(0).float() / 255
    _target = torch.tensor(1, dtype=torch.int64)
    _metadata = {
        "sample_id": "synthetic_contract_check",
        "source_split": "synthetic",
        "record_index": 0,
        "axes": "C,H,W",
        "class_names": [str(_i) for _i in range(10)],
        "preprocessing": "uint8 to float32; divide by 255; add channel axis",
        "polarity": "bright strokes on dark background",
    }
    _batch_x = torch.stack((_input, _input))
    _batch_y = torch.stack((_target, _target))
    assert _batch_x.shape == (2, 1, 28, 28)
    assert _batch_x.dtype == torch.float32 and torch.isfinite(_batch_x).all()
    assert 0 <= _batch_x.min() and _batch_x.max() <= 1
    assert _batch_y.shape == (2,) and _batch_y.dtype == torch.int64
    assert ((_batch_y >= 0) & (_batch_y < 10)).all()
    print("Contract inputs / targets:", _batch_x.shape, _batch_y.shape)
    print("Metadata:", _metadata)
    return


if __name__ == "__main__":
    app.run()
