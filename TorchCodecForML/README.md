# TorchCodec for Machine Learning

Four Marimo notebooks for the SE for Media classes, following [TorchVisionForML](../TorchVisionForML/) and [TorchAudioForML](../TorchAudioForML/). I use generated video and audio so we can inspect the decoded tensors without downloading recordings or model weights.

| Notebook                                            | Topics                                                                     |
| --------------------------------------------------- | -------------------------------------------------------------------------- |
| [Part 1: video](TorchCodecForMLPart1Video.py)       | Containers, encoding, frame tensors, metadata, timestamps and byte sources |
| [Part 2: sampling](TorchCodecForMLPart2Sampling.py) | Regular and random clips, time sampling, end policies and tensor layouts   |
| [Part 3: audio](TorchCodecForMLPart3Audio.py)       | WAV and FLAC, audio ranges, resampling, channel mixing and playback        |
| [Part 4: datasets](TorchCodecForMLPart4Datasets.py) | Recording splits, lazy decoding, DataLoader and a small temporal model     |

Each notebook runs independently and includes plots and exercises. The video examples use lossless FFV1 in a Matroska container, whilst the audio example uses WAV and FLAC. Temporary files are cleaned up when their directory objects are released. Part 4 overfits one small generated batch to check the training pipeline; it is not an action-recognition benchmark.
