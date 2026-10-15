# Notebook glossary

I have grouped these terms around the notebooks so we can look up a word whilst working through an example. Each entry gives a short explanation and a source for further reading. The notebook links show where to try the ideas in this repository. Use your editor's search to find a term, or start with a topic below.

For a first training example, read **feature**, **target**, **logit**, **loss**, **gradient**, **optimiser**, **epoch**, **validation** and **inference**. “Train” means fit the parameters; “validate” means check development choices; “test” means assess the final choice on reserved data.

The glossary covers the Python, array and machine learning vocabulary in the marimo notebook collection, including the image, audio, video, text and point-cloud examples. It is a reading companion rather than an API reference. The paper links give background on named methods; our small teaching models may differ from the published architectures.

## Contents

- [Data and the learning task](#data-and-the-learning-task)
- [Training, validation and testing](#training-validation-and-testing)
- [Model outputs and losses](#model-outputs-and-losses)
- [Gradients and optimisation](#gradients-and-optimisation)
- [Arrays, tensors and model layers](#arrays-tensors-and-model-layers)
- [Inference, saving and experiments](#inference-saving-and-experiments)
- [Evaluation and comparing results](#evaluation-and-comparing-results)
- [Images and image-to-image models](#images-and-image-to-image-models)
- [Audio and video](#audio-and-video)
- [Text and recurrent models](#text-and-recurrent-models)
- [Point clouds and 3D learning](#point-clouds-and-3d-learning)
- [Notebook and programming vocabulary](#notebook-and-programming-vocabulary)

## Data and the learning task

Try it in [Classification](Classification), [Data leakage](DataLeakage/DataLeakageMarimo.py), [Data pipelines](PyTorchForML/PyTorchForMLPart5DataAndSaving.py).

| Term | Meaning in these notebooks | Further reading |
| --- | --- | --- |
| Sample / example | One item presented to a model: an image, audio clip, text window or point cloud. In audio, “sample” can also mean one waveform measurement; check the context. | [Source](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) |
| Dataset | A collection of examples. A PyTorch `Dataset` defines how we retrieve an item and, where available, its target. | [Source](https://docs.pytorch.org/docs/stable/data.html) |
| Feature | A numerical property used as input, such as a coordinate, pixel value or frequency-band energy. A network can also learn intermediate features. | [Source](https://developers.google.com/machine-learning/glossary#feature) |
| Label / target / ground truth | The answer we train against: a class ID, next token, clean image or segmentation mask. Ground truth can contain annotation errors. | [Source](https://developers.google.com/machine-learning/glossary#label) |
| Supervised learning | Learning a mapping from inputs to supplied targets. Classification, regression and the paired image tasks here use this approach. | [Source](https://developers.google.com/machine-learning/glossary#supervised-machine-learning) |
| Classification | Predicting a category, such as a digit or object type. | [Source](https://developers.google.com/machine-learning/glossary#classification-model) |
| Binary classification | Choosing between two classes. Our single-output models use one logit per example and a threshold to choose a class. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) |
| Multi-class classification | Choosing one class from several mutually exclusive classes. Recognising one digit from 0–9 is multi-class classification. | [Source](https://scikit-learn.org/stable/modules/multiclass.html) |
| Multi-label classification | Allowing several labels for one example, such as both “person” and “bicycle”. Independent sigmoid outputs suit this task; a softmax forces competition between classes. | [Source](https://scikit-learn.org/stable/modules/multiclass.html) |
| Regression | Predicting numerical values, such as the output of a fitted line or reconstructed pixel intensities. | [Source](https://developers.google.com/machine-learning/glossary#regression-model) |
| Preprocessing | Preparing raw data for the model: decoding, resizing, scaling or tokenising. Keep the required steps consistent during training and inference. | [Source](https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html) |
| Normalisation / standardisation | Normalisation can mean several kinds of rescaling. Standardisation commonly subtracts a mean and divides by a standard deviation. State the formula; dividing pixels by 255 differs from standardising them. | [Source](https://scikit-learn.org/stable/modules/preprocessing.html) |
| Data augmentation | Creating varied training inputs, for example rotations, crops or added noise. The transformed input must still agree with its target; rotate a segmentation mask with its image. | [Source](https://docs.pytorch.org/vision/stable/transforms.html) |
| Batch / mini-batch | A group of examples processed together. Batch size controls how many examples contribute to a usual training update. | [Source](https://docs.pytorch.org/docs/stable/data.html) |
| DataLoader / collate | A `DataLoader` retrieves and batches dataset items. Its collate function combines individual items, sometimes padding variable-length sequences. | [Source](https://docs.pytorch.org/docs/stable/data.html) |
| Shuffle | Changing example order, usually each training epoch. Shuffling is different from creating a train/validation/test split. | [Source](https://docs.pytorch.org/docs/stable/data.html) |

## Training, validation and testing

Try it in [Linear model](LinearModel/LinearModelMarimo.py), [Group splits](DataLeakage/DataLeakageMarimo.py), [Point-cloud classification](PointCloudsAndMeshNet/PointCloudForMLPart4Classification.py).

| Term | Meaning in these notebooks | Further reading |
| --- | --- | --- |
| Training / train | Adjusting model parameters to reduce a loss on training examples. A “trained model” has gone through this fitting process; this alone does not tell us how well it generalises. | [Source](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) |
| Training set | The examples used to fit model parameters. Fit learned preprocessing statistics, such as a feature mean, using this set too. | [Source](https://scikit-learn.org/stable/common_pitfalls.html) |
| Validation / validate | Checking performance on held-out examples while developing a model. We use this feedback to choose settings or a checkpoint, without ordinary gradient updates on these examples. | [Source](https://scikit-learn.org/stable/modules/cross_validation.html) |
| Test set / held-out test | Examples reserved for the final assessment after model choices are settled. Repeatedly tuning against test results makes them part of model selection. | [Source](https://scikit-learn.org/stable/modules/cross_validation.html) |
| Group split | Keeping related examples in the same partition: all clips from one recording, patches from one photograph, or points sampled from one object. Choose the group to match the kind of unseen data you want to handle. | [Source](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data) |
| Stratified split | A split that approximately preserves class proportions. Stratification alone does not keep related recordings or subjects apart. | [Source](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) |
| Data leakage | Information unavailable at genuine prediction time entering training or model selection. Shared source recordings across splits and scaling fitted before splitting can make results misleading. | [Source](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) |
| Generalisation | How well a model works on unseen examples from the intended use case. | [Source](https://developers.google.com/machine-learning/glossary#generalization) |
| Overfitting | Fitting training-specific detail that does not transfer well. Falling training loss alongside rising validation loss is a warning sign. | [Source](https://developers.google.com/machine-learning/glossary#overfitting) |
| Underfitting | Failing to capture useful structure even in the training data, for example when a straight line cannot describe a curved relationship. | [Source](https://developers.google.com/machine-learning/glossary#underfitting) |
| Epoch | One pass through the training dataset. With 1,000 examples and batches of 100, a usual epoch contains ten updates. | [Source](https://developers.google.com/machine-learning/glossary#epoch) |
| Step / iteration | One iteration of a loop. An optimiser step updates parameters; an epoch usually contains many such steps. | [Source](https://docs.pytorch.org/docs/stable/optim.html) |
| Hyperparameter | A setting chosen outside ordinary parameter fitting, such as learning rate, batch size or hidden-layer width. We select these using validation feedback. | [Source](https://developers.google.com/machine-learning/glossary#hyperparameter) |
| Learning curve / loss curve | A plot of performance against training progress. Compare training and validation curves to see whether extra training is helping. | [Source](https://developers.google.com/machine-learning/glossary#loss-curve) |
| Early stopping / best checkpoint | Stopping when validation performance ceases to improve, or restoring the checkpoint with the best validation score. The last epoch need not be the best one. | [Source](https://developers.google.com/machine-learning/glossary#early-stopping) |
| Random seed / reproducibility | A seed controls a pseudorandom generator so we can repeat choices such as initial weights or splits. It does not guarantee identical results across devices and software versions. | [Source](https://docs.pytorch.org/docs/stable/notes/randomness.html) |

## Model outputs and losses

Try it in [Inference](PyTorchForML/PyTorchForMLPart4Inference.py), [Loss functions](PyTorchForML/PyTorchForMLPart3Modules.py), [Activations](PyTorchForML/PyTorchForMLActivations.py).

| Term | Meaning in these notebooks | Further reading |
| --- | --- | --- |
| Logit / logits | Raw classification scores before sigmoid or softmax. They can be negative or greater than one. For a binary probability p, the corresponding logit is log(p/(1−p)). | [Source](https://developers.google.com/machine-learning/glossary#logits) |
| Sigmoid | Maps a real number z to 1/(1+exp(−z)), between zero and one. A binary logit of 0 gives 0.5; a logit of about 2.20 gives 0.90. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html) |
| Softmax | Converts scores into positive values that sum to one along a chosen axis: exp(zᵢ)/sum(exp(zⱼ)). For batched classification, apply it across classes, not examples. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html) |
| Probability / confidence | Sigmoid or softmax outputs are model probability estimates. A value of 0.9 does not establish that the model is correct 90% of the time; that requires checking calibration. | [Source](https://scikit-learn.org/stable/modules/calibration.html) |
| Argmax / top-k | Argmax returns the index of the largest value. Top-k returns several highest values and their indices. For one class choice, logits and their softmax have the same argmax. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.topk.html) |
| Threshold | A cut-off used to turn a score into a decision. For sigmoid outputs, 0.5 corresponds to logit 0; another threshold may suit the costs of different errors. | [Source](https://scikit-learn.org/stable/modules/classification_threshold.html) |
| Decision boundary | The locations where the model changes its predicted class. The classification notebooks draw this boundary over a grid of input points. | [Source](https://developers.google.com/machine-learning/glossary#decision-boundary) |
| Loss / objective | A numerical penalty used to train the model. Lower loss means a better fit according to that particular objective. | [Source](https://docs.pytorch.org/docs/stable/nn.html#loss-functions) |
| Cross-entropy | A classification loss that penalises assigning low probability to the target class. With class-index targets, PyTorch `CrossEntropyLoss` takes raw logits and integer class IDs; do not apply softmax first. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) |
| Binary cross-entropy / BCEWithLogitsLoss | A loss for binary decisions, including per-pixel binary masks. `BCEWithLogitsLoss` combines sigmoid and BCE in a numerically stable operation and takes raw logits with floating-point targets. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) |
| Mean absolute error / MAE / L1 loss | The average absolute prediction error. Its units match the predicted quantity, and large errors grow linearly. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html) |
| Mean squared error / MSE / L2 loss | The average squared prediction error. Squaring gives large errors more influence. “L2 loss” sometimes means a sum rather than a mean, so check the reduction. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) |
| Reduction | Combining values using a sum, mean, maximum or another operation. A loss reduction determines how per-example or per-pixel errors become one value. | [Source](https://numpy.org/doc/stable/reference/ufuncs.html#methods) |
| Numerical stability | Arranging a calculation to avoid overflow, underflow or loss of precision. Built-in logits losses avoid several problems caused by separately taking exponentials and logarithms. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) |

## Gradients and optimisation

Try it in [Autograd](PyTorchForML/PyTorchForMLPart2Autograd.py), [Network from scratch](Neuron/nn_from_scratch.py), [NumPy gradient check](NumPyForML/NumPyForMLPart4Random.py).

| Term | Meaning in these notebooks | Further reading |
| --- | --- | --- |
| Parameter / weight / bias | A learned value. In a dense layer y=xWᵀ+b, weights W control input contributions and bias b adds an offset. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) |
| Forward pass | Computing model outputs from inputs using the current parameters. | [Source](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) |
| Derivative / gradient | A derivative measures local change. A gradient collects derivatives with respect to several values, telling us how the loss changes as parameters change. | [Source](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) |
| Chain rule / backpropagation | The chain rule combines derivatives through composed operations. Backpropagation applies it backwards through the computation to obtain parameter gradients. | [Source](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) |
| Autograd / computational graph | PyTorch records differentiable operations in a graph and uses it to calculate derivatives. `requires_grad=True` requests gradient tracking for a tensor. | [Source](https://docs.pytorch.org/docs/stable/notes/autograd.html) |
| Optimiser / gradient descent / SGD | An optimiser updates parameters using gradients. Gradient descent moves against the gradient; mini-batch stochastic gradient descent uses an estimate from a batch. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html) |
| Learning rate | The scale of an optimiser update. Too large can make training unstable; too small can make progress slow. | [Source](https://developers.google.com/machine-learning/glossary#learning-rate) |
| Momentum | Using information from previous updates to influence the next one, rather than responding only to the current gradient. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html) |
| Adam | An optimiser that maintains moving estimates of gradients and squared gradients to adapt updates for each parameter. | [Adam (Kingma and Ba)](https://arxiv.org/abs/1412.6980) |
| Zero gradients / gradient accumulation | PyTorch adds new gradients to existing `.grad` values. Clear them before an ordinary update; retaining them deliberately allows accumulation across batches. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html) |
| Detach | Disconnecting a tensor from its gradient history. Useful when recording predictions or converting them to NumPy without retaining the training graph. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html) |
| Finite differences / gradient checking | Estimating a derivative by making a small change to an input or parameter. Comparing this estimate with autograd helps check a gradient calculation. | [Source](https://docs.pytorch.org/docs/stable/notes/gradcheck.html) |
| Vanishing / exploding gradients | Gradients becoming extremely small or large through repeated operations. This can prevent effective learning, especially in deep or recurrent networks. | [Training RNNs (Pascanu et al.)](https://arxiv.org/abs/1211.5063) |

## Arrays, tensors and model layers

Try it in [NumPy](NumPyForML), [PyTorch tensors](PyTorchForML/PyTorchForMLPart1Tensors.py), [Modules](PyTorchForML/PyTorchForMLPart3Modules.py), [ASL CNN](ASL/ASLPart2CNNMarimo.py).

| Term | Meaning in these notebooks | Further reading |
| --- | --- | --- |
| Scalar / vector / matrix / tensor | A scalar is one value, a vector a one-axis array, and a matrix a two-axis array. In these notebooks, tensor means the array-like object used by PyTorch, with any number of axes. | [Source](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) |
| Shape / axis / dimension | Shape lists the lengths of the axes. A batch of RGB images often has shape (N, C, H, W): examples, channels, height and width. “Dimension” can mean an axis or its length. | [Source](https://numpy.org/doc/stable/user/absolute_beginners.html) |
| Dtype / floating point | The representation of each value, such as `float32`, `int64` or Boolean. Floating-point values have limited precision; decimal arithmetic may not be exact. | [Source](https://numpy.org/doc/stable/user/basics.types.html) |
| Broadcasting | Rules allowing compatible shapes to participate in array operations without explicitly copying values. Compare trailing axes: lengths must match or one must be 1. | [Source](https://numpy.org/doc/stable/user/basics.broadcasting.html) |
| Reshape / flatten / squeeze / unsqueeze | Reshape changes how values are arranged into axes; flatten collapses axes. Squeeze removes axes of length 1, whilst unsqueeze adds them. None means swapping axis order. | [Source](https://docs.pytorch.org/docs/stable/torch.html#tensor-manipulation-ops) |
| Transpose / permute | Changing axis order, for example HWC images to CHW. Reshaping to the desired lengths does not perform this rearrangement. | [Source](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html) |
| View / copy / contiguous | A view shares underlying storage; a copy owns separate storage. Contiguous tensors follow the expected dense memory layout, which some view operations require. | [Source](https://docs.pytorch.org/docs/stable/tensor_view.html) |
| Element-wise / matrix multiplication | Element-wise multiplication pairs corresponding values, with broadcasting where allowed. Matrix multiplication combines rows and columns using sums of products; NumPy and PyTorch use `@` for this. | [Source](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) |
| Boolean mask | An array of true/false values used to select or alter elements. This differs from a semantic segmentation mask, which supplies spatial class labels. | [Source](https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing) |
| Device / CPU / GPU / CUDA / MPS | The hardware location of a tensor or model. CUDA supports NVIDIA GPUs; MPS supports GPU computation on suitable Macs. Inputs and model parameters must be on compatible devices. | [Source](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) |
| Module / layer / Sequential | An `nn.Module` groups computation and registered state. Layers are modules; `nn.Sequential` connects modules in order. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) |
| Dense / fully connected / MLP | A dense layer combines all input features for each output. An MLP (multilayer perceptron) stacks dense layers with nonlinear activations. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) |
| Activation function | A function applied to layer outputs. Nonlinear activations let stacked dense layers represent more than a single affine mapping. | [Source](https://docs.pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) |
| ReLU / LeakyReLU / dead ReLU | ReLU returns max(0,x); LeakyReLU keeps a small slope for negative inputs. A dead ReLU remains on the zero-gradient side for the inputs it sees. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html) |
| Tanh / GELU / SiLU | Tanh squashes values into (−1,1). GELU and SiLU are smooth activations with different shapes; the activation notebook compares their outputs and derivatives. | [Source](https://docs.pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) |
| CNN / convolution / kernel | A convolutional neural network applies learned filters across a grid. The kernel is the filter's small window of weights; shared weights detect patterns at different locations. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) |
| Channel / feature map | A channel is one component of a spatial input or output, such as red, green or blue. A feature map is a spatial array of learned responses. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) |
| Stride / padding / receptive field | Stride controls how far a filter moves; padding extends boundaries. A receptive field is the input region that can affect an output value. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) |
| Pooling / global pooling | Combining local or whole-input features using a maximum or average. Global pooling can turn variable spatial sizes or point counts into a fixed-length vector. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool1d.html) |
| Batch normalisation | Normalising activations with batch statistics during training, followed by learned scale and offset. With the usual running-statistics setting, evaluation uses stored statistics. | [Batch Normalization (Ioffe and Szegedy)](https://arxiv.org/abs/1502.03167) |
| Dropout / regularisation | Dropout randomly zeros activations during training and rescales the retained values. It is a regularisation method: a way to discourage fitting training-specific detail. | [Dropout (Srivastava et al.)](https://jmlr.org/papers/v15/srivastava14a.html) |

## Inference, saving and experiments

Try it in [Inference modes](PyTorchForML/PyTorchForMLPart4Inference.py), [Checkpoints](Checkpoints/CheckPointsMarimo.py), [Transfer learning](PreTrainedModels/TransferLearningMarimo.py), [Tracking](ExperimentTracking), [Cloud training](CloudTraining/CloudTrainingMarimo.py).

| Term | Meaning in these notebooks | Further reading |
| --- | --- | --- |
| Inference / prediction | Using a model to compute outputs for inputs. Inference usually leaves learned parameters unchanged. | [Source](https://docs.pytorch.org/docs/stable/notes/autograd.html#evaluation-mode-nn-module-eval) |
| Training mode / evaluation mode | `model.train()` and `model.eval()` select behaviour for layers such as dropout and batch normalisation. Calling `train()` does not perform training; calling `eval()` does not disable gradients. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) |
| no_grad / inference_mode | Contexts that disable recording operations for backward differentiation. Inference mode removes extra bookkeeping too, with restrictions on later use in autograd. Use evaluation mode as well when appropriate. | [Source](https://docs.pytorch.org/docs/stable/notes/autograd.html#grad-modes) |
| Checkpoint / state_dict | A `state_dict` stores named parameters and persistent buffers. A resumable checkpoint may also include optimiser state, progress, random-generator state and preprocessing settings. | [Source](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) |
| Pre-trained model / transfer learning / fine-tuning | A pre-trained model has weights learned on earlier data. Transfer learning reuses this knowledge; fine-tuning updates some or all weights for a new task. | [Source](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) |
| Freezing | Disabling gradient computation for selected parameters so ordinary optimisation leaves them unchanged. Batch-normalisation buffers can still change in training mode. | [Source](https://docs.pytorch.org/docs/stable/notes/autograd.html#setting-requires-grad) |
| Backbone / classifier head | The backbone extracts features; the head maps them to task outputs. Transfer learning often replaces the original classifier head. | [Source](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) |
| ONNX / export | ONNX is a model interchange format. Export represents a model for another runtime; check supported operations, expected input shapes and agreement with the original outputs. | [Source](https://docs.pytorch.org/docs/stable/onnx.html) |
| Run / metric / artifact | A run is one recorded experiment. Metrics are measured values such as validation loss; artifacts are saved outputs such as weights, plots and configuration files. | [Source](https://mlflow.org/docs/latest/ml/tracking/) |
| TensorBoard / MLflow | Tools used here to record and compare experiments. TensorBoard visualises summaries; MLflow records runs, settings, metrics and artifacts. | [Source](https://docs.pytorch.org/docs/stable/tensorboard.html) |
| Container / image / volume | A container image packages a program and its environment. A container runs that image; a volume or bind mount keeps data accessible outside its writable layer. | [Source](https://docs.podman.io/en/latest/markdown/podman-run.1.html) |
| Endpoint / deployment | Deployment makes a model available for use. An inference endpoint accepts inputs and returns predictions, using the same required preprocessing as training. | [Source](https://fastapi.tiangolo.com/deployment/) |

## Evaluation and comparing results

Try it in [Image evaluation](EvaluationOfResults/BaselinesEvaluationMarimo.py), [Pet segmentation](ImageToImage/04PetSegmentationMarimo.py), [Point segmentation](PointCloudsAndMeshNet/PointCloudForMLPart6Segmentation.py).

| Term | Meaning in these notebooks | Further reading |
| --- | --- | --- |
| Accuracy | The fraction of predictions matching their targets. High accuracy can hide poor performance on rare classes or small foreground regions. | [Source](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) |
| Confusion matrix | A table counting true and predicted class combinations. It shows which classes the model confuses; check which axis represents the truth. | [Source](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) |
| Intersection over union / IoU / Jaccard | Overlap divided by union for predicted and target regions or point sets. Perfect agreement gives 1. State the class averaging and treatment of empty regions. | [Source](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html) |
| Baseline | A comparison method that gives a result we should try to improve on, such as bicubic enlargement or predicting the majority class. | [Source](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators) |
| PSNR / decibel / dB | Peak signal-to-noise ratio: 10 log10(L²/MSE), with L the permitted signal peak. Higher is better for this pixel-error measure. State the pixel range; a PSNR gain does not directly measure visual improvement. | [Source](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.peak_signal_noise_ratio) |
| Mean / standard deviation | The mean summarises the centre; standard deviation describes spread around it. Mean ± standard deviation is not a confidence interval for the mean. | [Source](https://numpy.org/doc/stable/reference/generated/numpy.std.html) |
| Paired comparison | Comparing methods on the same examples, then examining their per-example differences. This helps separate method differences from differences in example difficulty. | [Source](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html) |
| Bootstrap / paired bootstrap | Repeatedly sampling observed examples with replacement to estimate variability of a statistic. In a paired bootstrap, resample the same example indices for both methods. | [Source](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) |
| Confidence interval | An interval from a procedure designed to cover a population quantity at a stated long-run rate, under its assumptions. A 95% interval is not a range containing 95% of individual image scores. | [Source](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) |
| Error map / failure gallery | An error map shows where predictions differ from a reference. A failure gallery displays difficult examples so we can inspect problems that an average score hides. | [Source](https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html) |
| Robustness / distribution shift | Robustness concerns performance under changes such as noise, missing points or rotation. Distribution shift means the data encountered differs from the training distribution. | [Source](https://developers.google.com/machine-learning/glossary#distribution-skew) |

## Images and image-to-image models

Try it in [Image transforms](TorchVisionForML), [Image-to-image introduction](ImageToImage/01IntroductionMarimo.py), [Denoising](ImageToImage/02DenoisingMarimo.py), [Super resolution](ImageToImage/03SuperResolutionMarimo.py).

| Term | Meaning in these notebooks | Further reading |
| --- | --- | --- |
| RGB / greyscale / image layout | RGB uses red, green and blue channels; greyscale uses intensity values. Image files and libraries differ in axis order, dtype and range, so inspect these before feeding a model. | [Source](https://docs.pytorch.org/vision/stable/transforms.html) |
| Interpolation / nearest / bilinear / bicubic | Estimating pixels when resizing. Nearest selects a nearby value; bilinear blends nearby values; bicubic uses a larger cubic neighbourhood. Use nearest for discrete class masks to avoid inventing label values. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html) |
| Image-to-image learning | Predicting an output image or spatial map from an input image. Our examples restore noisy images, enlarge images and predict foreground masks. | [U-Net (Ronneberger et al.)](https://arxiv.org/abs/1505.04597) |
| Denoising / residual noise | Denoising estimates a clean signal from a noisy one. In our residual approach the network predicts noise, which we subtract from the input. | [DnCNN (Zhang et al.)](https://arxiv.org/abs/1608.03981) |
| Super resolution | Estimating a higher-resolution image from lower-resolution input. Missing detail is inferred from learned patterns and may differ from the original scene. | [Sub-pixel CNN (Shi et al.)](https://arxiv.org/abs/1609.05158) |
| Segmentation / mask | Assigning a class to each pixel or point. A binary mask identifies foreground and background; a multi-class mask assigns one of several class IDs. | [U-Net (Ronneberger et al.)](https://arxiv.org/abs/1505.04597) |
| Encoder / decoder / U-Net | An encoder builds features whilst reducing spatial resolution; a decoder builds a spatial output. U-Net connects matching resolutions so the decoder can use earlier detail. | [U-Net (Ronneberger et al.)](https://arxiv.org/abs/1505.04597) |
| Skip connection / concatenation | A skip connection carries features past intermediate layers. U-Net concatenates earlier and later features along channels; residual networks commonly add compatible features instead. | [Deep Residual Learning (He et al.)](https://arxiv.org/abs/1512.03385) |
| Residual learning | Learning a difference relative to a reference. The reference may be a feature tensor or an input image; the precise addition or subtraction depends on the model. | [Deep Residual Learning (He et al.)](https://arxiv.org/abs/1512.03385) |
| Upsampling / transposed convolution | Upsampling increases spatial size. A transposed convolution uses learned weights and can increase resolution; it is not generally an inverse convolution. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html) |
| Pixel shuffle / sub-pixel convolution | Rearranging channels into spatial positions. For scale r, r² channel groups supply the additional pixels in each spatial neighbourhood. | [Sub-pixel CNN (Shi et al.)](https://arxiv.org/abs/1609.05158) |
| Patch / tile / overlap blending | A patch is a small image region used for training or processing. Tiled inference processes large images in pieces; overlap blending combines shared areas to reduce visible seams. | [U-Net (Ronneberger et al.)](https://arxiv.org/abs/1505.04597) |
| Checkerboard artefact | An alternating pattern that can arise from uneven overlap during learned upsampling. Inspect outputs as well as aggregate errors. | [Deconvolution and Checkerboard Artifacts (Odena et al.)](https://distill.pub/2016/deconv-checkerboard/) |

## Audio and video

Try it in [Waveforms and features](TorchAudioForML), [Spoken digits](FreeSpokenDigits), [Decoding and clip sampling](TorchCodecForML).

| Term | Meaning in these notebooks | Further reading |
| --- | --- | --- |
| Waveform / amplitude / audio channel | A waveform stores signal amplitude over time. Mono has one channel, whilst stereo has two; audio channels differ from spectrogram frequency bands. | [Source](https://docs.pytorch.org/audio/stable/tutorials/audio_io_tutorial.html) |
| Sample rate / resampling | Sample rate is the number of waveform measurements per second, in Hz. Resampling recalculates the waveform at another rate; changing only the rate metadata changes playback speed and pitch. | [Source](https://docs.pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html) |
| FFT / frequency bin / phase | The fast Fourier transform efficiently computes a discrete Fourier transform. Its bins represent frequencies; complex values encode magnitude and phase. | [Source](https://numpy.org/doc/stable/reference/routines.fft.html) |
| STFT / spectrogram / window / hop | An STFT transforms successive overlapping windows of a signal. A spectrogram displays magnitude or power over time and frequency. Hop length sets the spacing between window starts. | [Source](https://docs.pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html) |
| Mel spectrogram / log mel | A mel spectrogram combines spectral energy into perceptually motivated frequency bands. Taking logarithms compresses its dynamic range; the result is not a waveform. | [Source](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html) |
| MFCC | Mel-frequency cepstral coefficients summarise the shape of a log mel spectrum using a cosine transform. The chosen coefficient count controls how much detail is retained. | [Source](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MFCC.html) |
| Signal-to-noise ratio / SNR | A ratio comparing signal and noise power, often expressed in dB. The audio augmentation notebook controls added noise using a chosen SNR. | [Source](https://docs.pytorch.org/audio/stable/generated/torchaudio.functional.add_noise.html) |
| SpecAugment | Augmentation of time-frequency features, including masking time or frequency regions. Our notebook uses these masks to remove parts of a spectrogram during training. | [SpecAugment (Park et al.)](https://arxiv.org/abs/1904.08779) |
| Padding / sequence length | Adding values to make examples fit a common batch shape. Keep track of real lengths so padding is not mistaken for meaningful signal. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html) |
| Codec / container / decoding | A codec encodes and decodes media; a container holds streams and metadata. Decoding turns encoded bytes into usable frames or audio samples. | [Source](https://meta-pytorch.org/torchcodec/stable/index.html) |
| Frame rate / timestamp / clip | Frame rate describes frames per second; timestamps locate frames in time. A clip is a sampled sequence, and timestamp-based sampling matters when frames are not evenly spaced. | [Source](https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.decoders.VideoDecoder.html) |
| Temporal model | A model that combines information across time, such as a 3D convolutional model processing a video clip. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html) |

## Text and recurrent models

Try it in [Text processing and generation](TextProcessingRNN/RNNPart1.py).

| Term | Meaning in these notebooks | Further reading |
| --- | --- | --- |
| Token / tokenisation / vocabulary | A token is a unit such as a character or word. Tokenisation splits text into units; a vocabulary maps recognised tokens to IDs. Our text example uses character tokens. | [Source](https://huggingface.co/docs/tokenizers/index) |
| One-hot encoding | Representing a category with a vector containing one 1 and zeros elsewhere. The position identifies the category without implying an ordering between categories. | [Source](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) |
| Embedding | A learned vector retrieved using a token ID. The embedding table is trained with the model; the ID itself is an index rather than a numerical measure of meaning. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) |
| RNN / hidden state | A recurrent neural network processes a sequence using a state carried between time steps. Its hidden state summarises context learned from the preceding inputs. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html) |
| GRU / gate | A gated recurrent unit uses learned gates to control how state is retained and updated. The text notebook stacks GRU layers after its embedding. | [RNN Encoder–Decoder (Cho et al.)](https://arxiv.org/abs/1406.1078) |
| Next-token prediction / autoregressive generation | Training predicts the following token at each position. During generation, we feed generated tokens back into the model to continue the sequence. | [Source](https://docs.pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html) |
| Temperature / sampling | Temperature rescales logits before converting them into probabilities. Lower values sharpen the distribution; higher values flatten it. Sampling draws a token rather than always taking the largest score. | [Source](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html) |

## Point clouds and 3D learning

Try it in [Point-cloud series](PointCloudsAndMeshNet).

| Term | Meaning in these notebooks | Further reading |
| --- | --- | --- |
| Mesh / vertex / face / point cloud | A mesh connects vertices using faces, often triangles. A point cloud stores sampled positions without requiring those surface connections. | [Source](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html) |
| Surface sampling / area weighting | Drawing points on a mesh surface. Choosing triangles in proportion to their area avoids giving small triangles disproportionate influence. | [Source](https://trimesh.org/trimesh.sample.html#trimesh.sample.sample_surface) |
| Centring / unit-sphere scaling | Translating points around a chosen centre and scaling their extent. This reduces translation and size variation but does not remove rotation differences. | [Source](https://github.com/charlesq34/pointnet) |
| Permutation invariance | An output stays unchanged when the input points are reordered. Classification of an unordered point set should not depend on storage order. | [PointNet (Qi et al.)](https://arxiv.org/abs/1612.00593) |
| Permutation equivariance | Reordering input points reorders their per-point outputs in the same way. Segmentation needs predictions to stay attached to the corresponding points. | [PointNet (Qi et al.)](https://arxiv.org/abs/1612.00593) |
| Shared point mapping / PointNet | PointNet applies the same learned mapping to each point, then combines features with a symmetric operation such as maximum. Our notebooks implement a small PointNet-style model. | [PointNet (Qi et al.)](https://arxiv.org/abs/1612.00593) |
| Global feature / critical points | A pooled feature summarises a cloud. With max pooling, particular points supply the maximum for each feature; the notebook highlights those points. | [PointNet (Qi et al.)](https://arxiv.org/abs/1612.00593) |
| Rotation augmentation / invariance | Rotation augmentation exposes training to more orientations. It can improve robustness but does not guarantee that every rotation produces the same prediction. | [PointNet (Qi et al.)](https://arxiv.org/abs/1612.00593) |
| ModelNet / ShapeNet Part | ModelNet supplies labelled 3D object models for classification. ShapeNet Part supplies part-labelled shapes for segmentation; object labels and point-level part labels answer different questions. | [Source](https://modelnet.cs.princeton.edu/), [Related source](https://shapenet.cs.stanford.edu/iccv17/) |

## Notebook and programming vocabulary

Try it in [marimo introduction](IntroToMarimo), [Python basics](Lecture2/Lecture2Marimo.py), [Control flow](Lecture3/Lecture3Marimo.py), [DataFrames](Lecture6).

| Term | Meaning in these notebooks | Further reading |
| --- | --- | --- |
| Reactive notebook / cell / widget | A notebook is divided into cells. In marimo, changing a value reruns dependent cells; widgets supply interactive values such as slider settings. | [Source](https://docs.marimo.io/guides/reactivity/) |
| Function / argument / return value | A function groups reusable operations. Arguments supply inputs; a return value passes a result to the caller. | [Source](https://docs.python.org/3/tutorial/controlflow.html#defining-functions) |
| Class / instance / method / inheritance | A class defines a type; an instance is one object of that type. Methods provide behaviour. Inheritance lets a class extend another, as our models extend `nn.Module`. | [Source](https://docs.python.org/3/tutorial/classes.html) |
| List / tuple / dictionary | Lists and tuples store ordered items; lists are mutable. Dictionaries map keys to values. A checkpoint dictionary can hold weights alongside configuration. | [Source](https://docs.python.org/3/tutorial/datastructures.html) |
| Index / slice | An index selects an item; a slice selects a range using start, stop and optional step. Python excludes the stop position. | [Source](https://docs.python.org/3/tutorial/introduction.html#lists) |
| Iteration / generator | Iteration visits items in turn. A generator produces items as needed, which is useful when processing data without constructing every item in advance. | [Source](https://docs.python.org/3/tutorial/classes.html#generators) |
| Exception | An object signalling an error or exceptional situation. `try` and `except` allow us to handle expected failures such as a missing file. | [Source](https://docs.python.org/3/tutorial/errors.html) |
| Module / package / virtual environment | A module organises Python code; packages organise modules. A virtual environment isolates installed dependencies for a project. | [Source](https://docs.python.org/3/tutorial/modules.html), [Related source](https://docs.python.org/3/tutorial/venv.html) |
| DataFrame / Series / group-by | A DataFrame is a labelled table; a Series is a labelled one-dimensional collection. Group-by gathers rows by keys so we can summarise each group. | [Source](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html) |
| Figure / axes / subplot | A Matplotlib figure holds the visual output; axes hold plotted data. Subplots arrange several axes within one figure. | [Source](https://matplotlib.org/stable/users/explain/quick_start.html) |

## A few distinctions to keep beside the training loop

For three mutually exclusive classes, a model might output logits `[2, 1, 0]`. Softmax gives approximately `[0.665, 0.245, 0.090]`, and argmax selects class index `0`. We pass the original logits to `CrossEntropyLoss`. For one binary output, logit `0` becomes probability `0.5` through sigmoid; we pass the original logit to `BCEWithLogitsLoss`.

A usual training batch follows this order: forward pass, loss, clear old gradients, backward pass, optimiser step. For validation, we switch to evaluation mode and disable gradient recording, compute the chosen metrics and leave out the optimiser step. Evaluation mode controls layer behaviour; disabling gradients controls autograd. [PyTorch explains these modes separately](https://docs.pytorch.org/docs/stable/notes/autograd.html#evaluation-mode-nn-module-eval).

The introductory [linear model](LinearModel/LinearModelMarimo.py) and [classification](Classification/SimpleClassificationMarimo.py) notebooks use training and validation sets. They check validation performance during training and omit a separate final test set to keep the examples small. For a final assessment after choosing settings, reserve another set. The [point-cloud classification notebook](PointCloudsAndMeshNet/PointCloudForMLPart4Classification.py) demonstrates training, validation and testing as separate stages.

The [multi-class notebook](Classification/MultiClassificationMarimo.py) chooses one class per example using one logit per class, `CrossEntropyLoss` for training and argmax for predictions. For a task where several labels may apply to one example, use the multi-label definition above instead.
