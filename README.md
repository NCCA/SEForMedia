# MSc AIM Software Engineering for Media

This code is used in the lectures for the [SE for Media Unit](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/) for MSc AIM. The lecture pages, labs and seminars all live on the unit website and the code for them lives here.

It is expected you will use [uv](https://docs.astral.sh/uv/) to run everything. Most of the notebooks are [marimo](https://marimo.io/) notebooks, so from the root of the repo

Note this README.md is autogenerate by claude on update and commit so may not be 100% accurate.

```bash
uv sync
uv run marimo edit Lecture2/Lecture2Marimo.py
```

## Contents

- [Lectures](#lectures)
- [Machine learning notebooks](#machine-learning-notebooks)
- [Labs and seminars](#labs-and-seminars)
- [Support code](#support-code)

## Lectures

The lecture numbers below follow the website, which doesn't always match the folder names (the `LectureN` folders stop at 7).

| Lecture                                                                         | Slides                                                                                               | Code                                                                        | Description                                                                               |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [1](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/lectures/Lecture1/)   | [Slides](https://nccastaff.bournemouth.ac.uk/jmacey/Lectures/SEForMedia/IntroToSEforMedia/)          | [Lecture1](Lecture1/), [IntroToMarimo](IntroToMarimo/)                      | Introduction to the unit, uv, and a first look at marimo                                  |
| [2](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/lectures/Lecture2/)   | [Slides](https://nccastaff.bournemouth.ac.uk/jmacey/Lectures/SEForMedia/IntroToPython/)              | [Lecture2](Lecture2/)                                                       | Introduction to Python, numbers, strings, lists, tuples and dictionaries                  |
| [3](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/lectures/Lecture3/)   | [Slides](https://nccastaff.bournemouth.ac.uk/jmacey/Lectures/SEForMedia/SequenceSelectionIteration/) | [Lecture3](Lecture3/)                                                       | Sequence, selection and iteration, functions and exceptions                               |
| [4](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/lectures/Lecture4/)   | [Slides](https://nccastaff.bournemouth.ac.uk/jmacey/Lectures/SEForMedia/OOInPython/)                 | [Lecture4](Lecture4/)                                                       | Object oriented programming, accessors, aggregation, inheritance and operator overloading |
| [5](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/lectures/Lecture5/)   | [Slides](https://nccastaff.bournemouth.ac.uk/jmacey/Lectures/SEForMedia/Numpy/)                      | [Lecture5](Lecture5/), [NumPyForML](NumPyForML/)                            | Using NumPy                                                                               |
| [6](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/lectures/Lecture6/)   | [Slides](https://nccastaff.bournemouth.ac.uk/jmacey/Lectures/SEForMedia/Pandas/)                     | [Lecture6](Lecture6/), [MatplotlibForML](MatplotlibForML/)                  | Matplotlib, Pandas and Polars                                                             |
| [7](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/lectures/Lecture7/)   | [Slides](https://nccastaff.bournemouth.ac.uk/jmacey/Lectures/SEForMedia/PyTorch/)                    | [Lecture7](Lecture7/), [PyTorchForML](PyTorchForML/)                        | PyTorch and tensors                                                                       |
| [8](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/lectures/Lecture8/)   | [Slides](https://nccastaff.bournemouth.ac.uk/jmacey/Lectures/SEForMedia/MLWorkflows/)                | [LinearModel](LinearModel/), [Neuron](Neuron/), [Checkpoints](Checkpoints/) | ML workflows with PyTorch                                                                 |
| [9](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/lectures/Lecture9/)   | [Slides](https://nccastaff.bournemouth.ac.uk/jmacey/Lectures/SEForMedia/Classification/)             | [Classification](Classification/)                                           | Simple, binary and multi-class classification                                             |
| [10](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/lectures/Lecture10/) | [Slides](https://nccastaff.bournemouth.ac.uk/jmacey/Lectures/SEForMedia/DataLoaders/)                | [MNIST](MNIST/), [Packages](Packages/)                                      | Data loaders, the python path and the MNIST dataset                                       |

## Machine learning notebooks

These build on the lectures and are the ones I use in the later machine learning sessions.

| Folder                                | Description                                                                                                                                               |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [NumPyForML](NumPyForML/)             | Five notebooks on the NumPy you actually need for the ML demos                                                                                            |
| [MatplotlibForML](MatplotlibForML/)   | Figures, plots, images and saving output                                                                                                                  |
| [PyTorchForML](PyTorchForML/)         | Tensors, autograd, modules, inference, data and saving models                                                                                             |
| [TorchVisionForML](TorchVisionForML/) | Images, transforms, augmentation and datasets with torchvision                                                                                            |
| [TorchAudioForML](TorchAudioForML/)   | Waveforms, features, augmentation and audio datasets                                                                                                      |
| [Neuron](Neuron/)                     | A neural network from scratch in plain Python                                                                                                             |
| [LinearModel](LinearModel/)           | Fitting a linear model with PyTorch                                                                                                                       |
| [Checkpoints](Checkpoints/)           | Saving and restoring training checkpoints                                                                                                                 |
| [Classification](Classification/)     | Simple, binary and multi-class classification                                                                                                             |
| [MNIST](MNIST/)                       | The MNIST dataset, data loaders, training, and a Qt sketch pad to test the model ([Sketch](MNIST/Sketch/))                                                |
| [ASL](ASL/)                           | American Sign Language recognition in three parts (dense, CNN, data augmentation) with a real-time webcam demo in [RealTimeCapture](ASL/RealTimeCapture/) |
| [PreTrainedModels](PreTrainedModels/) | Using pre-trained ImageNet models and transfer learning                                                                                                   |

## Labs and seminars

The lab and seminar write-ups are on the [unit website](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/), this is the code that goes with them.

| Folder                                    | Description                                                                                                                                                                                   |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Lab1](Lab1/)                             | A small maths quiz program and images for the first lab                                                                                                                                       |
| [ImageLab](ImageLab/)                     | Solution for [Lab 4 An Image Class](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/labs/lab4/lab4/)                                                                                    |
| [Seminars/Arguments](Seminars/Arguments/) | Starter and solution code for the [Command Line Arguments](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/seminars/Arguments/Arguments/) seminar, using `sys.argv`, argparse and click |
| [Seminars/Files](Seminars/Files/)         | Code for the [File IO](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/seminars/Files/Files/) seminar, reading and writing text and CSV files                                           |

## Support code

| Folder                          | Description                                                                                                        |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| [Utils](Utils/)                 | Shared helpers used by the notebooks (downloading data, picking a device, accuracy, checking if we are in the lab) |
| [IntroToMarimo](IntroToMarimo/) | An introduction to marimo notebooks and their UI elements                                                          |
| [Packages](Packages/)           | How Python finds packages and why the notebooks use `sys.path.append("../")` to get at `Utils`                     |
| [scripts](scripts/)             | A pre-commit hook I use to export the marimo notebooks to markdown                                                 |
