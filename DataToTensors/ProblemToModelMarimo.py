#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # From a problem to a first machine learning model

    When starting a machine learning project it is tempting to pick a network and then work out how to fit our data into it. I usually start with the question I want to answer. This helps us decide what one sample contains, what we want to predict and how we will check the result. We can then look at the structure of the data and choose a suitable representation and first model.

    In this notebook we will work through ten steps, using the circles from the [classification example](../Classification/SimpleClassificationMarimo.py). We will also look at how the same decisions apply to images, sound, motion and 3D data. All the code runs on the CPU using generated data, so there is nothing to download.

    I use *classic models* here to include linear regression, logistic regression, nearest neighbours and trees, alongside established neural network families such as multilayer perceptrons (MLPs), convolutional networks (CNNs) and recurrent networks (RNNs). Each makes different assumptions about the data and how it is organised.

    ## The ten rules

    | Step | Rule | What to write down |
    | --- | --- | --- |
    | 1 | State the prediction question | What information do we have, what do we want to predict, and when will we make the prediction? |
    | 2 | Inspect the source data | What do the values mean, what units do they use, what is missing, and where do the data and labels come from? |
    | 3 | Define one sample | What input, target and supporting information belong to one prediction? |
    | 4 | Decide how to split the data | What should the model encounter for the first time during evaluation? |
    | 5 | Choose a representation | What information and relationships do we need to preserve? |
    | 6 | Choose a first model | What simple baseline can we use, and which model family suits the structure of the data? |
    | 7 | Specify the output | What are the output shape, target type, loss function and evaluation metric? |
    | 8 | Build the smallest complete experiment | How will we split, preprocess, train and validate? |
    | 9 | Inspect the errors | Where does the model fail, and does it improve on the baseline? |
    | 10 | Save what we need to use the model | Save the model and preprocessing, document what each input and output axis means, and record its limitations. |

    Keep [From media data to tensors](DataToTensorsMarimo.py) alongside this guide for more detail on data layouts. There is no need to memorise every shape we use, but you should be able to explain what each axis means in your own data.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from sklearn.datasets import make_circles
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return (
        DummyClassifier,
        LogisticRegression,
        RandomForestClassifier,
        StandardScaler,
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        make_circles,
        make_pipeline,
        mo,
        nn,
        np,
        pd,
        plt,
        torch,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. State the prediction question

    I find it useful to start by completing this sentence:

    **Given [information available at prediction time], predict [answer] for [one sample].**

    This makes us think about what the model will actually receive when we use it, and what we expect it to produce.

    | Question | Task | Target for one sample |
    | --- | --- | --- |
    | Given an object's measurements, predict its mass | Regression: predict a numerical value | Mass in kilograms |
    | Given a digit image, predict which digit it shows | Classification: choose a category | An integer identifying the digit |
    | Given an image, predict the object class at each pixel | Segmentation: classify each location | A grid of class IDs |
    | Given a noisy recording, predict the clean waveform | Regression with multiple outputs | The clean audio samples |
    | Given earlier poses, predict the next ten poses | Forecasting: predict future observations | Future joint positions |
    | Given unlabelled assets, group similar assets | Clustering: find groups using a chosen measure of similarity | No supplied target |

    Clustering is slightly different here as we have no supplied answers. We need to decide what makes two assets similar and how we will judge whether the groups are useful.

    The same source data can be used for several tasks. For example, we might give a whole video clip one action label, label each frame separately, or predict a future frame. Each choice changes what one sample contains and what the model needs to output.

    For our circles example, the question is:

    **Given the two coordinates of one point, predict whether it belongs to the inner or outer circle.**

    Circle membership is what we want the model to learn, so we must keep it out of the input. We will use class IDs `0` and `1` to identify the two circles. These are labels; they do not describe a distance or imply a ranking.

    Before training anything, I would also check whether we can solve the problem with a known calculation. For ideal circles centred on the origin, we could calculate each point's distance from the origin and compare it with a fixed threshold. With real data, we would need to check whether that rule still works.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Inspect a few records before writing a model

    Open images, listen to recordings, plot measurements and count labels.
    Ask what produced each record and whether the collection represents
    the conditions in which we intend to use the model.

    | Check | Example problem it can reveal |
    | --- | --- |
    | Units and ranges | Centimetres mixed with metres |
    | Missing and invalid values | NaN depth, absent audio or untracked joints |
    | Label definitions and counts | Inconsistent annotation or a rare class |
    | Source identity | Duplicate images or clips from the same take |
    | Timing and alignment | Sound and motion offset by several frames |
    | Inference availability | A feature calculated using the answer or a later frame |

    A filename or a speaker ID can be useful metadata without being an
    appropriate input feature. If labels are missing, decide whether to
    annotate data, use a suitable pretrained representation, or change
    the question. Clustering does not automatically discover the semantic
    classes you wanted to predict.
    """)
    return


@app.cell
def _(make_circles, np, pd):
    coordinates, circle_labels = make_circles(
        n_samples=600,
        factor=0.45,
        noise=0.09,
        random_state=12,
    )
    feature_names = ("horizontal_position", "vertical_position")
    circle_records = pd.DataFrame(coordinates, columns=feature_names)
    circle_records["class_id"] = circle_labels
    assert np.isfinite(coordinates).all()
    print("Inputs (all samples, features):", coordinates.shape, coordinates.dtype)
    print("Separate targets:", circle_labels.shape, circle_labels.dtype)
    print("Class counts:", np.bincount(circle_labels))
    print("Toy coordinates are dimensionless; both features describe the same point.")
    circle_records.head(6)
    return circle_labels, coordinates, feature_names


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Define one sample, then add a batch

    A sample is one example for the prediction question. For circles it is
    one point with two features. For action recognition it might be a whole
    clip; its frames are time steps within the sample.

    | Part | Circles example | Spoken-digit example |
    | --- | --- | --- |
    | Input `X` | Two coordinates `(2,)` | One mono waveform `(1,S)` |
    | Target `y` | One scalar class ID | One scalar digit ID |
    | Metadata | Record ID, source, feature order | Recording ID, speaker, sample rate, length |

    A **feature** is a measurement supplied to the model. A **target** is
    what we ask it to predict. A **batch** is a group of samples processed
    together. Metadata tells us how to interpret or group the measurements;
    some metadata may also become an explicit model input if justified.

    Adding a batch axis gives point inputs `(B,2)` and labels `(B,)`, where
    `B` is the number of points in that batch. In scikit-learn we normally
    pass a whole training array `(n_samples,n_features)` to `fit`; the first
    dimension there is the training set size, rather than a chosen PyTorch
    minibatch size.
    """)
    return


@app.cell
def _(circle_labels, coordinates):
    _sample = {
        "inputs": coordinates[0],
        "target": circle_labels[0],
        "metadata": {"record_id": "generated_000", "axes": "features"},
    }
    print("One input:", _sample["inputs"], "shape:", _sample["inputs"].shape)
    print("One target:", _sample["target"], "shape:", _sample["target"].shape)
    print(
        "Four inputs:",
        coordinates[:4].shape,
        "four targets:",
        circle_labels[:4].shape,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Split according to what must be new

    We use training data to fit the model's parameters, validation data to choose between models and settings, and test data for a final assessment once we have made those choices.

    I choose the split by asking what the model will encounter for the first time when we use it. We should decide this before fitting preprocessing steps or generating related examples such as image crops and audio windows.

    | Intended use | How to split |
    | --- | --- |
    | New independent records from the same population | Use a random split. For classification, preserve the class proportions where possible (a stratified split). |
    | New speakers, actors or source objects | Keep all examples from each identity together, and reserve some identities for evaluation. |
    | New recordings from known performers | Keep complete recordings or sessions together, and reserve some for evaluation. |
    | Future observations | Train on earlier periods and evaluate on later ones. Keep windows from overlapping across the split boundaries. |

    For example, randomly splitting adjacent video frames can put almost identical images into training and test data. This tells us little about how the model will perform on a new recording.

    Keep augmented examples and duplicates in the same split as their source. If a dataset already has an evaluation protocol, check what it is designed to measure before changing it. The [data leakage notebook](../DataLeakage/DataLeakageMarimo.py) shows how these choices can affect the result.

    Our generated points are independent draws, so I use a stratified split with 60% for training, 20% for validation and 20% for testing. These proportions are a starting point. With a small dataset, we may need cross-validation within the development data, whilst keeping a suitable test set for the final assessment.
    """)
    return


@app.cell
def _(circle_labels, coordinates, np, train_test_split):
    _indices = np.arange(len(coordinates))
    train_ids, _remaining = train_test_split(
        _indices,
        test_size=0.4,
        stratify=circle_labels,
        random_state=3,
    )
    validation_ids, test_ids = train_test_split(
        _remaining,
        test_size=0.5,
        stratify=circle_labels[_remaining],
        random_state=4,
    )
    assert not (set(train_ids) & set(validation_ids))
    assert not (set(train_ids) & set(test_ids))
    assert not (set(validation_ids) & set(test_ids))
    print(
        "Training / validation / test:",
        len(train_ids),
        len(validation_ids),
        len(test_ids),
    )
    return test_ids, train_ids, validation_ids


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Choose what information the representation keeps

    Before choosing a tensor shape, write down what each axis means. The shape should reflect the information we want the model to use. Adding a dimension because a layer expects one does not give that dimension any meaning.

    For example, neighbouring pixels have a spatial relationship in an image. Two neighbouring rows in a point cloud array may contain points from opposite sides of an object. We need to preserve the relationships that matter for our task.

    | Source | Simple features | Representation that keeps more structure | What to consider |
    | --- | --- | --- | --- |
    | Table | A row of measurements | `(B,F)` features | Feature order, units and how categories are encoded |
    | Image | Flattened pixels or colour and texture measurements | `(B,C,H,W)` image grid | Flattening removes the explicit grid layout, although the pixel values remain |
    | Audio | Duration, energy or averaged spectral features | `(B,C,S)` waveform or `(B,C,Q,L)` spectrogram | Averaging can remove timing information and detail |
    | Video | Frame features averaged over a clip | `(B,T,F)` frame features or `(B,C,T,H,W)` pixels | Averaging can hide the order of actions |
    | Motion | A fixed history flattened into one vector | `(B,T,J,3)` joint positions | Joint identities, coordinate frame and time order |
    | Point cloud | Size, extent or aggregated point features | `(B,N,F)` point attributes | Changing the row order should not change the geometry |
    | Triangle mesh | Geometric measurements | Vertex arrays and integer face indices | Connectivity and which vertices and faces belong to each mesh |
    | Text | Word counts or TF-IDF weights | `(B,L)` token IDs with masks identifying valid positions | Vocabulary, word order and tokenisation |

    The symbols used here are:

    | Symbol | Meaning |
    | --- | --- |
    | `B` | Samples in a batch |
    | `F` | Features |
    | `C` | Channels |
    | `H`, `W` | Image height and width |
    | `S` | Audio samples |
    | `Q` | Frequency bins |
    | `L` | Analysis frames for a spectrogram, or token positions for text |
    | `T` | Time steps |
    | `J` | Joints |
    | `N` | Points |

    These are conventions for this guide. Record what each axis means beside your own data, particularly where the same letter could mean different things.

    We also need to decide how to prepare the values. Models such as logistic regression and nearest neighbours usually benefit from scaling numerical features to comparable magnitudes. Fit the scaler using the training data, then use those same settings for validation, testing and new predictions. Trees generally do not need this scaling.

    Categories need some care too. If we give materials integer IDs, the difference between IDs `2` and `3` does not describe a physical difference between those materials. Choose an encoding that suits the model, and decide how to handle missing values and categories that were absent from training.

    For our circles example, I keep the two original coordinates. We could also calculate the radius and add it as a feature, provided distance from the origin is meaningful for the problem. This is *feature engineering*: we are using our knowledge of the data to provide another measurement. We still need to check whether it helps on new data. Reshaping the coordinates alone would not calculate that feature.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Match the first model to the structure

    I usually start with a simple baseline and one model that suits the data. The baseline gives us something to compare against. For regression, this might mean always predicting the mean training target; for classification, always choosing the most common training class.

    The table gives us some starting experiments. Our choice also depends on how much data we have, the available labels, how long the model takes to run and where we intend to use it.

    | Data and question | Simple baseline | Model to try | Model input layout |
    | --- | --- | --- | --- |
    | Table → a measured quantity | Mean training target | Linear or ridge regression, then trees if needed | `(B,F)` |
    | Table → a class | Most common training class | Logistic regression, k-nearest neighbours, random forest or an MLP | `(B,F)` |
    | Image → a class | Most common training class | Small CNN, or a pretrained image encoder with a classifier | `(B,C,H,W)` for a CNN |
    | Image → pixel labels | Most common training pixel class at every location | Fully convolutional network, such as U-Net | `(B,C,H,W)` |
    | Audio → a class | Logistic regression on averaged spectral features | CNN on a spectrogram, or a 1D CNN on the waveform | `(B,C,Q,L)` or `(B,C,S)` |
    | Video → an action | Average frame features over the clip and use a linear classifier | Frame CNN followed by a GRU/LSTM, or a 3D CNN | `(B,T,F)` for a GRU with `batch_first=True`; `(B,C,T,H,W)` for a 3D CNN |
    | Earlier motion → future motion | Repeat the last observed pose | Linear predictor using a fixed history, or a GRU/LSTM | `(B,T,J*3)` for a basic GRU |
    | Point cloud → a class | Classifier using geometric measurements | PointNet-style model with a shared mapping for each point and symmetric pooling | `(B,N,F)` conceptually; often `(B,F,N)` in `Conv1d` code |
    | Mesh → a deformation | Rest pose or a known rig deformation | MLP with fixed vertex correspondence, or a graph/mesh network using connectivity | Positions and controls, plus topology where needed |
    | Text → a class | Most common training class | TF-IDF with logistic regression or a linear SVM | Sparse `(n_samples,vocabulary_size)` matrix |

    It helps to know what each model assumes about its input. Logistic regression is a classifier despite its name (slightly confusing!). It learns a linear decision boundary in the features we supply. Nearest neighbours compares examples using a distance, whilst trees make decisions by splitting feature values.

    An MLP learns combinations of features through layers and nonlinearities. A CNN shares filters across a grid, such as an image or spectrogram. An RNN carries a state through an ordered sequence; GRUs and LSTMs are common variants. A PointNet-style model processes each point using the same mapping, then combines the results with an operation such as maximum pooling. This makes the whole-cloud prediction independent of the point row order.

    If one sample contains several types of data, such as audio and video, I would keep them as separate entries in a dictionary. Keep the inputs aligned and retain their identities, timestamps and masks indicating which values are valid. A first model could summarise each input separately, then combine those features in a linear classifier or MLP. We still need to establish how the inputs line up in time; concatenating raw audio samples and video pixels does not do this for us.

    Some assumptions need particular care. An MLP trained on meshes with matching vertex IDs relies on that correspondence when we give it a new mesh. Accepting arbitrary topology requires a different approach.

    Axis order matters too. A video stored as `(B,T,C,H,W)` needs its axes reordered before we pass it to a PyTorch 3D convolution expecting `(B,C,T,H,W)`.

    ### Explore four layouts

    Change the domain below and look at how the input shape and model operations change. Each example classifies a sample into one of three classes, so the output shape is always `(B,3)`.

    The models are untrained: we are checking how data moves through them and what shapes they produce. We are not comparing prediction quality. All sequences in these examples have the same valid length.
    """)
    return


@app.cell
def _(mo):
    layout_choice = mo.ui.dropdown(
        options=[
            "Table / MLP",
            "Image / CNN",
            "Motion / GRU",
            "Points / shared MLP",
        ],
        value="Table / MLP",
        label="Input structure",
    )
    layout_choice
    return (layout_choice,)


@app.cell
def _(layout_choice, nn, torch):
    with torch.random.fork_rng():
        torch.manual_seed(6)
        if layout_choice.value == "Table / MLP":
            _x = torch.randn(2, 4)
            _model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
            _output = _model(_x)
            _meaning = "(B,F): 2 records, 4 measurements each"
        elif layout_choice.value == "Image / CNN":
            _x = torch.randn(2, 3, 6, 8)
            _model = nn.Sequential(
                nn.Conv2d(3, 5, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(5, 3),
            )
            _output = _model(_x)
            _meaning = "(B,C,H,W): 2 RGB images, 6 rows, 8 columns"
        elif layout_choice.value == "Motion / GRU":
            _poses = torch.randn(2, 5, 4, 3)
            _x = _poses.flatten(start_dim=2)
            _rnn = nn.GRU(12, 8, batch_first=True)
            _sequence_output, _hidden = _rnn(_x)
            _output = nn.Linear(8, 3)(_hidden[-1])
            _meaning = "(B,T,J*3): 2 clips, 5 frames, 4 joints with XYZ; classify the whole clip"
        else:
            _x = torch.randn(2, 7, 3)
            _shared = nn.Sequential(nn.Linear(3, 8), nn.ReLU())
            _head = nn.Linear(8, 3)
            _output = _head(_shared(_x).amax(dim=1))
            _shuffled = _head(
                _shared(_x[:, torch.tensor([6, 2, 4, 0, 5, 1, 3])]).amax(dim=1)
            )
            torch.testing.assert_close(_output, _shuffled)
            _meaning = "(B,N,F): 2 clouds, 7 points, XYZ features; pooling is unchanged by shuffling"
    assert _output.shape == (2, 3)
    print(_meaning)
    print("Input:", tuple(_x.shape), "output logits:", tuple(_output.shape))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Specify the output, target and scoring rule

    Work backwards from the answer you need. Input channels do not determine
    output classes. An RGB image has three colour channels whether we want
    to predict two classes or twenty.

    | Task | Model output | Training target | Typical loss | Evaluation to inspect |
    | --- | --- | --- | --- | --- |
    | One numerical value | `(B,1)` float | `(B,1)` float in stated units | Mean squared or absolute error | MAE in those units |
    | One of `K` classes | `(B,K)` logits | `(B,)` int64 IDs | Cross entropy | Accuracy, per-class recall, confusion matrix |
    | Several independent labels | `(B,K)` logits | `(B,K)` float 0/1 values | Binary cross entropy with logits | Per-label precision/recall and chosen thresholds |
    | Class at every image pixel | `(B,K,H,W)` logits | `(B,H,W)` int64 IDs | Cross entropy, masking unlabelled pixels | Per-class IoU and visual overlays |
    | Future joint positions | `(B,Tout,J,3)` float | Same shape, same units and joint order | Masked regression loss | Position error and plotted motion |

    `K` is the number of classes or labels, `Tout` the forecast length. A
    logit is an unnormalised model score. PyTorch cross entropy takes
    logits directly. A binary problem can alternatively use one logit per
    sample and binary cross entropy with matching floating-point targets;
    do not mix that convention with the two-logit class-ID convention.

    A loss supplies the training signal; a metric reports something useful
    about the result. Mean absolute error in metres is easier to interpret
    than squared metres. With rare classes, inspect per-class results:
    predicting the common class every time can give high accuracy whilst
    missing every rare example. No single metric describes all errors.

    In scikit-learn, classifier `predict` returns class IDs, not logits.
    `predict_proba` returns probabilities when supported. For a single
    regression target scikit-learn often uses `(n_samples,)`, whereas the
    PyTorch example above keeps an explicit output axis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Run one small, complete experiment

    We will compare three fixed choices on validation data: a constant
    baseline, logistic regression and a random forest. This follows the
    circles classification lesson using traditional estimators, so we can
    focus on the experiment rather than a neural network training loop.

    The scaler and logistic regression live in one pipeline. Calling
    `fit` on training rows fits both steps; validation data is transformed
    using those saved training statistics. A pipeline does not fix a bad
    split or an input that reveals the target.

    Logistic regression cannot draw a circular boundary from the original
    two coordinates. A forest can approximate a nonlinear boundary with
    many rectangular splits. If the linear model fails here, adding
    training epochs would not change the shape of boundary it can express.

    We select the highest validation balanced accuracy from the two fitted
    candidate models. The constant model supplies the baseline. Settings
    and the selection rule are fixed before we inspect the test results.
    Balanced accuracy averages recall across the classes.
    """)
    return


@app.cell
def _(
    DummyClassifier,
    LogisticRegression,
    RandomForestClassifier,
    StandardScaler,
    balanced_accuracy_score,
    circle_labels,
    coordinates,
    make_pipeline,
    pd,
    train_ids,
    validation_ids,
):
    fitted_models = {
        "Constant baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=300)
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=80, max_depth=6, random_state=8
        ),
    }
    _validation_scores = {}
    _rows = []
    for _name, _model in fitted_models.items():
        _model.fit(coordinates[train_ids], circle_labels[train_ids])
        _train_prediction = _model.predict(coordinates[train_ids])
        _validation_prediction = _model.predict(coordinates[validation_ids])
        _score = balanced_accuracy_score(
            circle_labels[validation_ids], _validation_prediction
        )
        _validation_scores[_name] = _score
        _rows.append(
            {
                "Model": _name,
                "Training balanced accuracy": balanced_accuracy_score(
                    circle_labels[train_ids], _train_prediction
                ),
                "Validation balanced accuracy": _score,
            }
        )
    chosen_name = max(
        ("Logistic regression", "Random forest"),
        key=lambda _name: _validation_scores[_name],
    )
    print("Selected using validation data:", chosen_name)
    pd.DataFrame(_rows)
    return chosen_name, fitted_models


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Inspect errors and assess the selected model

    Plot predictions as well as a score. A confusion matrix counts true
    classes against predicted classes. Here rows are true IDs and columns
    are predicted IDs. Off-diagonal entries are mistakes.

    We now evaluate the selected model on the reserved test set and compare
    it with the constant baseline. The plot shows only test points, marking
    errors with crosses. Because there are only 120 test points, the score
    is an estimate from this sample, rather than a guarantee of performance.

    | Observation | First thing I would check |
    | --- | --- |
    | Both training and validation scores are poor | Labels, input information, preprocessing and model capacity |
    | Training is strong, validation is weak | Overfitting, split differences and too few independent examples |
    | Score is unexpectedly perfect | Leakage, duplicates, trivial cues, or a genuinely easy task |
    | One class fails | Label counts, annotation quality and per-class recall |
    | Synthetic success, real-world failure | Whether generated data represents actual conditions |

    Change one justified choice at a time and record it. If we use these
    test errors to choose the next feature or model, this set has become
    development data. We would need fresh held-out data for a final claim.
    """)
    return


@app.cell
def _(
    accuracy_score,
    balanced_accuracy_score,
    chosen_name,
    circle_labels,
    confusion_matrix,
    coordinates,
    fitted_models,
    np,
    plt,
    test_ids,
):
    _truth = circle_labels[test_ids]
    _prediction = fitted_models[chosen_name].predict(coordinates[test_ids])
    _baseline = fitted_models["Constant baseline"].predict(coordinates[test_ids])
    _matrix = confusion_matrix(_truth, _prediction, labels=[0, 1])
    assert _prediction.shape == _truth.shape
    assert _matrix.sum() == len(test_ids)
    print("Selected model:", chosen_name)
    print("Test accuracy:", accuracy_score(_truth, _prediction))
    print("Test balanced accuracy:", balanced_accuracy_score(_truth, _prediction))
    print(
        "Baseline test balanced accuracy:",
        balanced_accuracy_score(_truth, _baseline),
    )
    print("Confusion matrix (true rows, predicted columns):\n", _matrix)
    _mistakes = _prediction != _truth
    print("Misclassified test record IDs:", test_ids[_mistakes].tolist())
    evaluation_figure, _axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for _class_id in (0, 1):
        _selected = _prediction == _class_id
        _axes[0].scatter(
            *coordinates[test_ids][_selected].T,
            s=20,
            label=f"Predicted {_class_id}",
        )
    _axes[0].scatter(
        *coordinates[test_ids][_mistakes].T,
        marker="x",
        s=65,
        color="black",
        label="Error",
    )
    _axes[0].set(
        xlabel="Horizontal position",
        ylabel="Vertical position",
        title="Reserved test points",
    )
    _axes[0].set_aspect("equal")
    _axes[0].legend(fontsize=8)
    _axes[1].imshow(_matrix, cmap="Blues", vmin=0)
    for _row, _column in np.ndindex(_matrix.shape):
        _text_colour = (
            "white" if _matrix[_row, _column] > _matrix.max() / 2 else "black"
        )
        _axes[1].text(
            _column,
            _row,
            str(_matrix[_row, _column]),
            ha="center",
            va="center",
            color=_text_colour,
        )
    _axes[1].set(
        xticks=[0, 1],
        yticks=[0, 1],
        xlabel="Predicted class",
        ylabel="True class",
        title="Confusion matrix",
    )
    evaluation_figure.tight_layout()
    plt.close(evaluation_figure)
    evaluation_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Save the contract, then try an unfamiliar sample

    Saving weights alone is not enough. Save the feature or class order,
    preprocessing, source version, split membership and evaluation settings.
    For media, also retain timing, coordinate conventions, rig or skeleton
    definitions, and masks where they are needed to interpret the data.

    The trained model should receive the same representation at inference
    that it saw during training. An input with the correct dimensions can
    still be wrong if its columns have been swapped or its units changed.
    The next cell creates an in-memory contract; a real experiment would
    save it alongside the fitted pipeline and its dependency versions.

    ### A worksheet for your own project

    Complete these sentences before implementing a dataset:

    1. Given **...**, available at **...**, I want to predict **...**.
    2. One independent sample is **...**, identified by **...**.
    3. Its input is **...**, with axes **...**, units **...** and missing values handled by **...**.
    4. Its target is **...**, with dtype **...** and output shape **...**.
    5. I will hold out **...** because I need to evaluate performance on new **...**.
    6. My baseline is **...**; my first learned model is **...** because **...**.
    7. I will fit preprocessing on **...** and judge the result using **...**.
    8. I will inspect errors by **...** and retain this metadata: **...**.

    If you cannot fill in a sentence yet, that identifies the next thing
    to investigate. Choosing a larger model will not fill it in for you.
    """)
    return


@app.cell
def _(
    chosen_name,
    coordinates,
    feature_names,
    fitted_models,
    np,
    test_ids,
    train_ids,
    validation_ids,
):
    problem_contract = {
        "task": "classify one generated point as outer or inner circle",
        "input_axes": "samples, features",
        "feature_order": feature_names,
        "units": "dimensionless toy coordinates",
        "class_names_in_id_order": ["outer", "inner"],
        "source": {
            "generator": "make_circles",
            "seed": 12,
            "factor": 0.45,
            "noise": 0.09,
        },
        "split_record_ids": {
            "train": train_ids.tolist(),
            "validation": validation_ids.tolist(),
            "test": test_ids.tolist(),
        },
        "model": chosen_name,
        "selection_metric": "validation balanced accuracy",
        "limitation": "evaluated only on generated circles from this distribution",
    }
    _new_records = np.array([[0.0, 0.45], [0.0, 1.0]])
    assert _new_records.shape[1] == coordinates.shape[1] == len(feature_names)
    _new_ids = fitted_models[chosen_name].predict(_new_records)
    print("Feature order:", problem_contract["feature_order"])
    print("New inputs:\n", _new_records)
    print("Predicted IDs:", _new_ids)
    print(
        "Predicted names:",
        [problem_contract["class_names_in_id_order"][_id] for _id in _new_ids],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where to go next

    | Your next question | Existing example |
    | --- | --- |
    | How do I organise media into tensors? | [Data to tensors](./DataToTensorsMarimo.py) |
    | How does a numerical prediction get fitted? | [Linear model](../LinearModel/LinearModelMarimo.py) |
    | How do I train a nonlinear classifier? | [Binary classification](../Classification/BinaryClassificationMarimo.py) |
    | How do I build a dataset and inspect batches? | [PyTorch data pipelines](../PyTorchForML/PyTorchForMLPart5DataAndSaving.py) |
    | What changes for spoken digits? | [Free Spoken Digits](../FreeSpokenDigits/FSDDMarimoPt1.py) |
    | Can I reuse an image model? | [Transfer learning](../PreTrainedModels/TransferLearningMarimo.py) |
    | How should I process unordered 3D points? | [A small PointNet](../PointCloudsAndMeshNet/PointCloudForMLPart3PointNet.py) |
    | Why do recording groups matter? | [Data leakage and group splits](../DataLeakage/DataLeakageMarimo.py) |

    For reference, see scikit-learn's [estimator guide](https://scikit-learn.org/stable/tutorial/machine_learning_map/),
    [preprocessing pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
    and [pipelines](https://scikit-learn.org/stable/modules/compose.html).
    The input requirements for the layout examples are documented in
    PyTorch's [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html),
    [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
    and [GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) pages.

    ## Exercises

    1. Write the worksheet for a dataset you intend to use. Explain what
       will be new at inference: records, subjects, recordings or time periods.
    2. Change digit classification into digit segmentation. Rewrite the
       sample, output, target and evaluation parts of the contract.
    3. Propose two representations of an audio recording: pooled features
       for logistic regression and a spectrogram for a CNN. What does each retain?
    4. Using only development data, compare logistic regression on XY
       coordinates with logistic regression on distance from the origin.
       Explain which domain assumption you added. Reserve a new test set
       if the displayed test results influenced your choices.
    5. For motion forecasting, specify a last-pose baseline and identify
       an input feature that would accidentally use a future frame.
    """)
    return


if __name__ == "__main__":
    app.run()
