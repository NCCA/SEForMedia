#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Using Polars

    In this notebook we will use [Polars](https://docs.pola.rs/user-guide/) to load, display and clean data. It follows the same examples as `IntroductionToPandasMarimo.py`, using the MetObjects dataset from the [Metropolitan Museum of Art](https://github.com/metmuseum/openaccess).

    I will use a copy of the dataset from my website. We will work through the same questions, whilst looking at how Polars uses expressions to describe operations on columns.

    Run this notebook from the repository with `uv run marimo edit Lecture6/IntroductionToPolarsMarimo.py`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Downloading the dataset

    We use [pathlib](https://docs.python.org/3/library/pathlib.html) to store the data beside the notebook, or in /transfer if we are in the lab. `parents=True` creates intermediate folders and `exist_ok=True` allows us to reuse a folder. The Pandas notebook's downloaded files can be reused here.
    """)
    return


@app.cell
def _(mo):
    import sys
    from pathlib import Path

    sys.path.append("../")
    from Utils import in_lab

    if in_lab():
        data_dir = Path("/transfer/met_objects")
    else:
        data_dir = Path(mo.notebook_dir()) / "data" / "met_objects"
    data_dir.mkdir(parents=True, exist_ok=True)
    return (
        Path,
        data_dir,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use `requests` to download the zip and `tqdm` to show progress. `raise_for_status()` reports HTTP errors before we try to unpack the response.
    """)
    return


@app.cell
def _(Path):
    import requests
    from tqdm import tqdm

    def download(url: str, fname: Path) -> None:
        """Download an archive whilst displaying progress.

        Parameters
        ----------
        url : str
            Address of the archive.
        fname : Path
            Local destination.
        """
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with (
                fname.open("wb") as file,
                tqdm(total=total, unit="B", unit_scale=True) as progress,
            ):
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    progress.update(file.write(chunk))

    return (download,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first run downloads and extracts `MetObjects.csv`. Later runs reuse the CSV. If we already have the zip, we only need to extract it.
    """)
    return


@app.cell
def _(data_dir, download):
    import zipfile

    url = (
        "https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/DataSets/MetObjects.zip"
    )
    zip_file = data_dir / "MetObjects.zip"
    csv_file = data_dir / "MetObjects.csv"
    if not csv_file.exists():
        if not zip_file.exists():
            download(url, zip_file)
        with zipfile.ZipFile(zip_file) as archive:
            archive.extract("MetObjects.csv", data_dir)
    print(f"Using {csv_file}")
    return (csv_file,)


@app.cell
def _(data_dir):
    for child in data_dir.iterdir():
        print(child)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setting up the environment

    We import Polars as `pl` and use Matplotlib for the pie chart exercise. `pl.Config` controls the text representation of a DataFrame; Marimo also provides an interactive table view.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import polars as pl

    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_cols(20)
    plt.rcParams["figure.figsize"] = [8, 7]
    plt.rcParams["figure.autolayout"] = True
    return (
        pl,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The DataFrame

    A DataFrame is a table of records. Each column has a name and one data type. We use [`pl.read_csv`](https://docs.pola.rs/api/python/stable/reference/api/polars.read_csv.html) to load this comma separated file; a tab separated file would use `separator="\t"`.

    This dataset has mixed formats in some columns. I use `infer_schema=False` to load the columns as strings, then convert the years explicitly later. Empty CSV fields become `null`, which is Polars' missing-value marker. A floating-point `NaN` is a separate value; `fill_null` and `drop_nulls` do not handle it.
    """)
    return


@app.cell
def _(csv_file, pl):
    dataset = pl.read_csv(csv_file, separator=",", infer_schema=False)
    dataset
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accessing and displaying data

    ### Integer indexing

    Polars has no Pandas-style row index or `iloc` accessor. `slice(offset, length)` selects rows by position. To select positions 29 through 34, we start at 29 and request six rows.
    """)
    return


@app.cell
def _(dataset):
    int_indexing = dataset.slice(29, 6)
    int_indexing
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Boolean Series and expressions

    A Series is a single column of values. Comparing a Series with a value produces booleans, which we can pass to `filter`. Combine conditions with `&` (and), `|` (or) and `~` (not).

    We can also describe a column operation using `pl.col("Department")`. This creates an expression, which Polars evaluates when we pass it to `filter`, `select` or `with_columns`.
    """)
    return


@app.cell
def _(dataset):
    medieval_art_bool_series = dataset["Department"] == "Medieval Art"
    dataset.filter(medieval_art_bool_series)
    return (medieval_art_bool_series,)


@app.cell
def _(dataset, pl):
    both_departments = dataset.filter(
        (pl.col("Department") == "Medieval Art")
        | (pl.col("Department") == "European Sculpture and Decorative Arts")
    )
    both_departments
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Grouping by column names

    `group_by` collects records with matching values. Calling `len(name="Counts")` counts the rows in each group. Unlike Pandas' default grouping, Polars includes null keys. Here we drop missing keys explicitly to match the original exercise, then sort the result for display.
    """)
    return


@app.cell
def _(dataset):
    g = (
        dataset.drop_nulls(["Object Name", "Culture"])
        .group_by(["Object Name", "Culture"])
        .len(name="Counts")
        .sort(["Object Name", "Culture"])
    )
    g
    return (g,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conditional values and filtering

    Pandas' `where` can preserve the table shape whilst replacing values that fail a condition. In Polars we use `when(...).then(...).otherwise(...)` for this. Here groups containing one object become null across all columns.
    """)
    return


@app.cell
def _(g, pl):
    masked_groups = g.select(
        pl.when(pl.col("Counts") > 1).then(pl.all()).otherwise(None)
    )
    masked_groups
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we want to remove those rows completely, `filter` expresses that directly. Neither operation changes `g`.
    """)
    return


@app.cell
def _(g, pl):
    g.filter(pl.col("Counts") > 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data cleaning

    Before using the data we need to inspect missing values, irrelevant columns and inconsistent formats. We will keep a smaller set of columns, supply missing titles and convert the years.

    ### Deleting rows by position

    `dataset.slice(1)` returns all rows after the first. This demonstrates the operation in the Pandas notebook, but the first row of this CSV is a valid coin record. We will retain it in the cleaning steps below. Always inspect a record before deleting it!
    """)
    return


@app.cell
def _(dataset):
    dataset_without_first = dataset.slice(1)
    dataset_without_first
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Removing columns

    We can remove columns containing only null values by comparing their null count with the number of rows.
    """)
    return


@app.cell
def _(dataset):
    empty_cols = [
        name
        for name, count in dataset.null_count().row(0, named=True).items()
        if count == dataset.height
    ]
    dataset_2 = dataset.drop(empty_cols)
    dataset_2
    return (dataset_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `columns` property lists the column names. `drop` removes columns by name; it does not need an `axis` argument.
    """)
    return


@app.cell
def _(dataset_2):
    dataset_2.columns
    return


@app.cell
def _(dataset_2):
    keep = [
        "Object Number",
        "Is Public Domain",
        "Department",
        "AccessionYear",
        "Object Name",
        "Title",
        "Object Begin Date",
        "Medium",
        "Dimensions",
        "Tags",
    ]
    exclude_cols = [name for name in dataset_2.columns if name not in keep]
    print(exclude_cols)
    dataset_3 = dataset_2.drop(exclude_cols)
    dataset_3
    return (dataset_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Default values

    We replace missing titles with "Untitled" using `fill_null`. `with_columns` returns a new DataFrame with the updated column. Assigning each cleaning stage to a new name also makes the dependencies clear to Marimo.
    """)
    return


@app.cell
def _(dataset_3, pl):
    dataset_titled = dataset_3.with_columns(pl.col("Title").fill_null("Untitled"))
    dataset_titled
    return (dataset_titled,)


@app.cell
def _(dataset_titled, pl):
    untitled_rows = dataset_titled.filter(pl.col("Title") == "Untitled")
    untitled_rows
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Removing rows

    For this exercise we require tags, dimensions and both year fields. `drop_nulls(subset=...)` removes a row if any of those fields is missing.

    This produces a subset of the collection. Our later answers describe this subset, so they should not be taken as statistics for the whole museum. The Dimensions strings still use different formats; parsing them is beyond this exercise.
    """)
    return


@app.cell
def _(dataset_titled):
    important_cols = ["Tags", "Dimensions", "Object Begin Date", "AccessionYear"]
    dataset_4 = dataset_titled.drop_nulls(subset=important_cols)
    dataset_4
    return (dataset_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data types

    We only need years for sorting, so I will store them as signed integers. This also preserves negative years for ancient artworks. Each column is converted independently.

    `str.strip_chars()` removes surrounding whitespace. `cast(pl.Int32, strict=False)` turns invalid values, such as date ranges or words, into nulls. For complete calendar dates we could instead use [`str.to_date`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.to_date.html).
    """)
    return


@app.cell
def _(dataset_4, pl):
    dataset_years = dataset_4.with_columns(
        pl.col("AccessionYear", "Object Begin Date")
        .str.strip_chars()
        .cast(pl.Int32, strict=False)
    )
    dataset_years
    return (dataset_years,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now remove rows whose years could not be converted. This is our final cleaned dataset.
    """)
    return


@app.cell
def _(dataset_years):
    dataset_5 = dataset_years.drop_nulls(["AccessionYear", "Object Begin Date"])
    dataset_5
    return (dataset_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DataFrame interrogation

    Using `dataset_5`, try these questions:

    1. Which department houses the oldest artwork in our cleaned subset?
    2. What proportion of these artworks belongs to each department? Display this as a pie chart.
    3. Which tag occurs most often? This needs several operations.

    There is space for your code below each question, followed by a solution.
    """)
    return


@app.cell(hide_code=True)
def _():
    solutions = [
        'dataset_sorted = dataset_5.sort("Object Begin Date")\n'
        "oldest_record = dataset_sorted.row(0, named=True)\n"
        "print(f\"The oldest record is from the {oldest_record['Department']} department\")",
        'departments = dataset_5.group_by("Department").len(name="Counts").sort("Department")\n'
        "departments",
        "fig, ax = plt.subplots(figsize=(12, 7))\n"
        'wedges, _ = ax.pie(departments["Counts"].to_list(),\n'
        "                   colors=plt.get_cmap('tab20').colors)\n"
        'ax.legend(wedges, departments["Department"].fill_null("Unknown").to_list(),\n'
        '          loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)\n'
        'ax.set_title("Artworks by department (cleaned subset)")\n'
        "fig",
        "tags_df = (\n"
        '    dataset_5.select(pl.col("Tags").str.split("|").alias("Tag"))\n'
        '    .explode("Tag")\n'
        '    .with_columns(pl.col("Tag").str.strip_chars())\n'
        '    .filter(pl.col("Tag").is_not_null() & (pl.col("Tag") != ""))\n'
        '    .group_by("Tag").len(name="Counts")\n'
        '    .sort(["Counts", "Tag"], descending=[True, False])\n'
        ")\n"
        "tags_df.head(10)",
    ]
    return (solutions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The oldest artwork

    Sort by `Object Begin Date`, then use `row(0, named=True)` to retrieve a dictionary for the oldest record. This assumes our cleaned subset contains at least one row.
    """)
    return


@app.cell
def _():
    # Write your code here.
    return


@app.cell(hide_code=True)
def _(mo, solutions):
    mo.md(
        "<details><summary>Solution</summary>\n\n```python\n"
        + solutions[0]
        + "\n```\n\n</details>"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Proportion of artworks by department

    Group by `Department` and count the records in each group.
    """)
    return


@app.cell
def _():
    # Write your code here.
    return


@app.cell(hide_code=True)
def _(mo, solutions):
    mo.md(
        "<details><summary>Solution</summary>\n\n```python\n"
        + solutions[1]
        + "\n```\n\n</details>"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Drawing the pie chart

    Pass the counts and department names to [Matplotlib pie](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pie.html). We can use `to_list()` at the plotting boundary without converting the DataFrame to Pandas.
    """)
    return


@app.cell
def _():
    # Write your code here.
    return


@app.cell(hide_code=True)
def _(mo, solutions):
    mo.md(
        "<details><summary>Solution</summary>\n\n```python\n"
        + solutions[2]
        + "\n```\n\n</details>"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Most common theme

    The tags are separated by `|`. Use `str.split` to create a list in each row, then `explode` to make one row per tag. Strip whitespace and remove empty tags before grouping, counting and sorting. This counts tag occurrences across the cleaned subset; filter to paintings first if that is the question you want to answer.
    """)
    return


@app.cell
def _():
    # Write your code here.
    return


@app.cell(hide_code=True)
def _(mo, solutions):
    mo.md(
        "<details><summary>Solution</summary>\n\n```python\n"
        + solutions[3]
        + "\n```\n\n</details>"
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
