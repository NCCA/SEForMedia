#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Using Pandas""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this notebook, we will be looking at pandas, a Python library that provides many useful tools for loading, displaying, and cleaning data. Please have a look at the [official Pandas documentation](https://pandas.pydata.org/docs/reference/index.html) to learn more about any of the functions you encounter in this notebook.

    To aid us in showing off the functionality of this library, we will be looking at the MetObjects dataset, which comes courtesy of the [Metropolitan Museum of Art in New York](https://www.metmuseum.org/). This data can be found on GitHub [here](https://github.com/metmuseum/openaccess/tree/master) as well as on [Kaggle](https://www.kaggle.com/metmuseum/the-metropolitan-museum-of-art-open-access).

    For this lecture we will download the dataset from my website as a zip file then extract that into a new folder in this current directory.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Downloading the dataset
    You can use a leading `!` in a line of Jupyter notebook code to specify that the rest of the line should be interpreted as a shell command. This is convenient for modifying files or running scripts that live on your filesystem without having to switch between the browser and terminal. Let's use this syntax to create a directory for the Met Museum dataset
    """
    )
    return


@app.cell
def _():
    data_dir = "./data/met_objects"  # Normal python code

    from pathlib import Path

    Path.mkdir(data_dir, exist_ok=True)
    # note the exits_ok parameter in mkdir, which allows us to create a new directory without throwing an error if it already exists
    # it will also create any intermediate directories that don't exist
    return Path, data_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Now we are going to download the dataset from my website and extract it into the data_dir folder. There are many different methods to do this, in this case we are going to use the ```requests``` library to download the file and ```zipfile``` to extract it."""
    )
    return


@app.cell
def _():
    import requests
    from tqdm import tqdm

    def download(url: str, fname: str):
        resp = requests.get(url, stream=True, verify=False)
        total = int(resp.headers.get("content-length", 0))
        # Can also replace 'file' with a io.BytesIO object
        with (
            open(fname, "wb") as file,
            tqdm(
                desc=fname,
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar,
        ):
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)

    return (download,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""The above function takes two parameters, one is the url of the file to download and the other is the path to save the file. We can use it as follows to download the zip file."""
    )
    return


@app.cell
def _(Path, data_dir, download):
    url = (
        "https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/DataSets/MetObjects.zip"
    )
    zip_file = f"{data_dir}/MetObjects.zip"
    csv_file = f"{data_dir}/MetObjects.csv"
    # check to see if we have downloaded the data
    if not Path(zip_file).exists() or not Path(csv_file).exists():
        download(url, f"{data_dir}/MetObjects.zip")
    else:
        print(f"{zip_file} already downloaded")
    return csv_file, zip_file


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Now we will unzip the dowlnoaded file into the data_dir folder. We can use the ```zipfile``` library to do this."""
    )
    return


@app.cell
def _(data_dir, zip_file):
    import zipfile

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        print(f"[INFO] Unzipping {zip_file} data...")
        zip_ref.extractall(data_dir)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We should now be able to see the file MetObject.csv in the data_dir folder."""
    )
    return


@app.cell
def _(Path, data_dir):
    folder = Path(data_dir)
    for child in folder.iterdir():
        print(child)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Setting up the environment
    Now that we have the data, we can start by importing the necessary libraries and loading the data into a pandas DataFrame. To make displaying the data easier, we can set some options for pandas to display more rows and columns before we create our first data frame.
    """
    )
    return


@app.cell
def _():
    # Module imports and plot settings
    import matplotlib.pyplot as plt
    import pandas as pd

    pd.set_option("display.max_rows", 20)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)
    pd.options.mode.chained_assignment = None
    plt.rcParams["figure.figsize"] = [8, 7]
    plt.rcParams["figure.autolayout"] = True
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The DataFrame

    The DataFrame is the central data structure provided by Pandas, and it is this structure that we need to interrogate when we want to ask questions about our data. You can think of a DataFrame as a table with rows of records and columns that describe the fields of those records. Pandas provides built in functions for loading text files and automatically puts their contents into a DataFrame. The dataset we just downloaded (`MetObjects.csv`) is a CSV (comma separated value) file, so we need to use the `load_csv` function provided by Pandas.
    """
    )
    return


@app.cell
def _(csv_file, pd):
    # Loading the dataset into a DataFrame
    # The `sep` argument is the delimiter character, which
    # tells pandas how to separate columns. Be careful: always inspect your CSV
    # files to check what the delimiter character is: sometimes people use
    # tabs (\t). Make a note of what the delimiter is and pass it into
    # read_csv. In this case, the separator is a comma (,) so that's what
    # we'll use.
    dataset = pd.read_csv(csv_file, sep=",")
    dataset  # <- this is a Pandas DataFrame
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Accessing and displaying data

    ### Integer indexing
    Similar to Python list slices, uses 0-indexed start and end positions to return a subset of the dataframe. With a Padas dataframe, this is done via the `iloc` indexer.
    """
    )
    return


@app.cell
def _(dataset):
    # Get rows number 29 to 35
    int_indexing = dataset.iloc[29:35]
    int_indexing
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Boolean Series
    A series is a 1-D array - a boolean series is one that is filled with boolean (i.e., `True` or `False`) values. We can pass boolean series into a Dataframe's `loc` indexer to keep only the values that align with `True`. Different boolean series of the same length can be combined using the following logical operators: `&` (and), `|` (or), `~` (not).
    """
    )
    return


@app.cell
def _(dataset):
    # getting all rows from the 'Medieval Art' department
    medieval_art_bool_series = dataset["Department"] == "Medieval Art"
    dataset.loc[medieval_art_bool_series]
    return (medieval_art_bool_series,)


@app.cell
def _(dataset, medieval_art_bool_series):
    # getting rows from the 'Medieval Art' OR 'European Sculpture and Decorative Arts' departments
    esada_bool_series = (
        dataset["Department"] == "European Sculpture and Decorative Arts"
    )
    both = esada_bool_series | medieval_art_bool_series
    dataset.loc[both]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Grouping by column name(s)
    We can also group the data by a list of columns. This returns a Pandas GroupBy object, which contains a dictionary of mappings from each group name to a Series of its elements
    """
    )
    return


@app.cell
def _(dataset):
    g = dataset.groupby(["Object Name", "Culture"])
    g = g.size().reset_index(name="Counts")
    g
    return (g,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Using ```where``` to filter data

    Similar to Numpy arrays, Pandas dataframes also make use of the ```where``` function to conditionally modify its elements based on some criteria. ```where``` takes a dataframe condition as an argument and returns the modified dataframe - if the condition is fulfilled, it keeps the value of the field, if not, it replaces it with `NaN`. We can use this function to remove objects that do not
    """
    )
    return


@app.cell
def _(g):
    g.where(g["Counts"] > 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""As you can see, most of the records are now NaN. We will look at how these types of situations can be fixed in the upcoming cells. Let's  print out the original dataset again so we can use it as a visual reference for the future:"""
    )
    return


@app.cell
def _(dataset):
    dataset
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data cleaning
    Before data can be fed into your application, it needs to be verified and checked for consistency. We can see that there are several problems with the dataset right off the bat:
    1. First row seems to contain garbage: none of the column names match up with the data types, and many are NaN
    2. It looks like many of the columns are completely empty - they add nothing to the dataset but clutter it
    3. Too many columns! This depends on what your needs are, but we don't need all of them for this exercise
    4. Inconsistent formatting in the Dimensions column - makes it difficult to use them downstream
    5. Mixed datatypes in Year fields

    Let's address all of these issues one by one

    ### Deleting rows by index
    We can get rid of the first row (index 0) by taking a slice of the dataframe beginning at index 1 and going all the way to the end.
    """
    )
    return


@app.cell
def _(dataset):
    dataset_1 = dataset.iloc[1:]
    dataset_1
    return (dataset_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Removing columns

    Columns can be removed conditionally by checking their contents to see if they meet a certain criteria, or simply by name
    """
    )
    return


@app.cell
def _(dataset_1):
    dataset_2 = dataset_1.dropna(how="all", axis=1)
    dataset_2
    return (dataset_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We can also remove columns by name. This is done by passing a list of column names to the `drop` function, along with the `axis=1` argument to specify that we are dropping columns, not rows. We can see all of the keys in the dataframe by calling the `keys` function on it."""
    )
    return


@app.cell
def _(dataset_2):
    dataset_2.keys()
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
    exclude_cols = [col for col in dataset_2.keys() if col not in keep]
    print(exclude_cols)
    dataset_3 = dataset_2.drop(exclude_cols, axis=1)
    dataset_3
    return (dataset_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Default values
    Artists are known to often leave their work untitled. In our dataset, this is not handled very gracefully - the titles of such artworks are simple NaN. Fortunately, we have another way of dealing with missing data: assigning a default value. We can replace any instance of a NaN title with the string "Untitled"
    """
    )
    return


@app.cell
def _(dataset_3):
    dataset_3["Title"].fillna("Untitled", inplace=True)
    dataset_3
    return


@app.cell
def _(dataset_3):
    untitled_rows = dataset_3[dataset_3["Title"] == "Untitled"]
    untitled_rows
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Removing rows
    We can also get rid of rows that do not meet certain criteria. For example, given a subset of fields that we deem very important, we can drop all rows are NaN in any of these fields
    """
    )
    return


@app.cell
def _(dataset_3):
    important_cols = ["Tags", "Dimensions", "Object Begin Date", "AccessionYear"]
    dataset_4 = dataset_3.dropna(subset=important_cols)
    dataset_4
    return (dataset_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Data types
    We change some data types that don't really make sense: `AccessionYear` and `Object Begin Date` were originally loaded in with mixed datatypes (some are strings, some are numbers), which makes it difficult to sort correctly.

    We can convert this data using the pd.to_datetime function, which will convert the data to a datetime object. This will allow us to sort the data correctly. In this case I will convert the `AccessionYear` and `Object Begin Date` columns to datetime objects. I will also ignore any errors that may arise from this conversion and set the `coerce` parameter to True. This will replace any errors with NaT values. I am only intereseed in the year, so I will extract that from the datetime object using the format string '%Y'.
    """
    )
    return


@app.cell
def _(dataset_4, pd):
    dataset_4["AccessionYear"] = pd.to_datetime(
        dataset_4["AccessionYear"], errors="coerce", format="%Y"
    )
    dataset_4["Object Begin Date"] = pd.to_datetime(
        dataset_4["AccessionYear"], errors="coerce", format="%Y"
    )
    dataset_4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""You can see some of the data has NaT values. We can drop these for our final dataset as follows"""
    )
    return


@app.cell
def _(dataset_4):
    dataset_5 = dataset_4.dropna(subset=["AccessionYear"])
    dataset_5 = dataset_5.dropna(subset=["Object Begin Date"])
    dataset_5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Dataframe interrogation
    We can now begin to ask some interesting questions about this dataset:
    1. Which department houses the oldest artwork in the museum? Use `Object Begin Date` for this task.
    2. What is the proportion of artworks from each department? Display this graphically using [`pd.DataFrame.plot.pie`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.pie.html)
    3. _What is the most common theme across all the paintings? Or, which tag is most common?**\*\***_

    _**\*\* Slightly more difficult, multi-step problem**_
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The oldest artwork

    To do this we can sort the data by the "Object Begin Date" column and then take the first row. We can then print out the department that houses this artwork.
    """
    )
    return


@app.cell
def _():
    # Put code in here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <details>

    <summary>Solution</summary>

    ```python
    dataset_sorted = dataset.sort_values('Object Begin Date')
    oldest_record = dataset_sorted.iloc[0] # access the oldest record
    department_name = oldest_record['Department'] # get the name of the Department from the record
    print(f"The oldest record in the dataset is from the {department_name} department")
    ```

    </details>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Proportion of artworks by department

    In this case we can use the groupby function to grab the departments and then count the number of records in each department. We can then use the plot function to display this data as a pie chart.
    """
    )
    return


@app.cell
def _():
    # put code here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <details>

    <summary>Solution</summary>

    ```python

    departments = dataset.groupby(['Department']).size().reset_index(name='Counts')
    departments

    ```

    </details>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We can use the build in function plot.pie to display this data as a pie chart"""
    )
    return


@app.cell
def _():
    # put code  here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <details>

    <summary>Solution</summary>

    ```python
    # The DataFrame has a built-in plotting function that you can
    # call like so:
    departments.plot.pie(y="Counts",
                         explode=[0.125 for _ in range(len(departments))],
                         labels=departments['Department'],
                         legend=None, ylabel="")
    ```

    </details>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Most common theme

    This is a multi stage process,  first we are going to generate an empty list to store all the tags in.
    New we will generate a function to split the tags into individual tags and add them to the list. We can then apply this function to the tags column of the dataset.

    Note the tags are separated by a pipe character `|`. We can use the `str.split` function to split the tags into a list of tags.

    Now we can use the power of pandas to generate a new dataframe from the tags list and run some analysis on it.
    """
    )
    return


@app.cell
def _():
    # write code here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <details>

    <summary>Solution</summary>

    ```python
    # declare an empty list to collect all the separated tags in the database
    tags_list = []
    # get the Tags series out of the dataset and assign it to a variable
    # called tags_series
    tags_series = dataset['Tags']


    # Write a function that accepts a Tag string and a list, and updates
    # the list with the individual tags found in the string.
    def update_list(t, l):
        separated = t.split("|")
        l += separated

    tags_series.apply(update_list, args=(tags_list,))

    # turn the tags list into a dataframe called tags_df
    tags_df = pd.DataFrame(tags_list, columns=["Tag"])

    # group the dataframe by Tag and count the occurences of each tag
    tags_df = tags_df.groupby("Tag").size().reset_index(name='Counts')

    # sort the list by count to find the one with the tag with the
    # highest number of occurences
    tags_df = tags_df.sort_values('Counts', ascending=False)
    tags_df.iloc[0]['Tag']
    tags_df
    ```

    </details>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
