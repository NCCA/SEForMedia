import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Free Spoken Digits Dataset

    A simple audio/speech dataset consisting of recordings of spoken digits in wav files at 8kHz. The recordings are trimmed so that they have near minimal silence at the beginnings and ends. This is basically the audio equivilent of the MINST data set and fairly easy to use.

    FSDD is an open dataset, which means it will grow over time as data is contributed. In order to enable reproducibility and accurate citation the dataset is versioned using Zenodo DOI as well as git tags. In this example we will use the data set

    ## Data Download

    This dataset is very common and you can find it on GitHub, HugginFace and Kaggle along with other locations. For this demo I have decided to use Kaggle hub and also introduce their API for downloading the data.

    ## Kaggle Hug

    KaggleHub is a Python library for accessing resources hosted on Kaggle. We can use it to download pre-trained models for our own programs, saving us from downloading and organising the files by hand.

    We can add it to our project by using uv as follows (this is already in the pyproject.toml for this repo)

    ```bash
    uv add kagglehub
    ```

    ## Naming cells

    Just to note, I have right clicked on the cell below and given it a name. This is so I can use the same code again in another notebook. I have named this one "download_digits" and in the next notebooks I will use this again, so I can use the variable dataset_path.
    """)
    return


@app.cell
def download_digits():
    import sys

    import kagglehub

    sys.path.append("../")
    from Utils import in_lab

    # in the lab use /transfer, otherwise keep the data beside the notebook
    if in_lab():
        output_dir = "/transfer/spoken_digits"
    else:
        output_dir = "./data/spoken_digits"

    dataset_path = kagglehub.dataset_download(
        "jackvial/freespokendigitsdataset",
        output_dir=output_dir,
    )

    print(f"Dataset downloaded to: {dataset_path}")
    return (dataset_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lets have a look at what we downloaded and see how the data is formatted, we can use the marimo file ui to do this easily.
    """)
    return


@app.cell
def _(dataset_path, mo):
    from pathlib import Path

    file_browser = mo.ui.file_browser(
        initial_path=Path(dataset_path),
        selection_mode="all",
        ignore_empty_dirs=True,
    )
    file_browser
    return (Path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After a little exploring, you can see that there is a file called "train.csv" and a folder called recordings. Each of the recordings are in the name format ```{Number}_{Name}_{Take/Version}.wav```

    So we can use this data to help build our classification, the first number is the label. The rest is the variance we need to use to split our test / train data. You will also notice there is a "train.csv" file in the downloaded file and we can have a look at this as well.

    To do this we can use the pandas library and the marimo data explorer.
    """)
    return


@app.cell
def _(dataset_path, mo):
    import pandas as pd

    df = pd.read_csv(f"{dataset_path}/train.csv")
    mo.ui.data_explorer(df)
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Looking at the data.

    Let's dig a little deeper into the data and see what we have. I'm going to scan the folder with pathlib then build a data frame with information. As the files are .wav files we can also investigate what we have using the standard python library "wave"
    """)
    return


@app.cell
def _(Path, dataset_path, pd):
    import wave

    root = Path(f"{dataset_path}/recordings")
    rows = []
    for path in sorted(root.glob("*.wav")):
        digit, speaker, take = path.stem.split("_")
        with wave.open(str(path)) as w:
            frames, rate = w.getnframes(), w.getframerate()
        rows.append(
            {
                "file_name": path.name,
                "digit": int(digit),
                "speaker": speaker,
                "take": int(take),
                "sample_rate": rate,
                "frames": frames,
                "duration_s": frames / rate,
                "bytes": path.stat().st_size,
            }
        )

    file_df = pd.DataFrame(rows)
    return file_df, root


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can dig in a little more now. The following shows the username and how many didgets the have each generated.
    """)
    return


@app.cell
def _(file_df, pd):
    pd.crosstab(file_df.speaker, file_df.digit)  # recordings per speaker per digit
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can filter the durations to get an idea of how long the data clips are
    """)
    return


@app.cell
def _(file_df):
    file_df.groupby("speaker").duration_s.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So there are 3,000 records in total. That's 3,000 .wav files in recordings/: 6 speakers × 10 digits × 50 takes.

    We can looks at the sample rates and see if we have any odd data
    """)
    return


@app.cell
def _(file_df):
    file_df.sample_rate.value_counts()  # check the audio format is consistent
    return


@app.cell
def _(file_df):
    file_df.nlargest(5, "duration_s")  # find outliers
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Audio Playback

    Marimo can playback audio using the ```mo.audio(audio_path.read_bytes())``` function.  I use a vstack to place them all together and grab some of the files so you can hear them.
    """)
    return


@app.cell
def _(mo, root):
    mo.vstack(
        [
            mo.audio(_path.read_bytes())
            for _path in sorted(root.glob("*.wav"))[0:3000:500]
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decision Time

    So we can split out test / train data in two main ways. If we look at the typical 80/20 split we could  either

    - 4 people for train with all the numbers 2 people for validation with all the numbers
    - Split randomly on number rather than speaker

    Either is a valid approach, and it is worth trying both at some stage to see what happens.

    In the next notebooks we will investigate how we can use a Dataloader to pad or trim each clip to 1 s, compute a 64-band log-mel spectrogram, and then train our networks.
    """)
    return


if __name__ == "__main__":
    app.run()
