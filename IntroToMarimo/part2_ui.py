#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # marimo UI widgets

    In this notebook we will use `mo.ui` to add controls to our examples.
    Each widget has a `.value` attribute which we can read from another cell.
    Changing the control causes marimo to re-run the cells which use it.

    I have grouped the examples by the type of input we need. Run the notebook
    from the repository root using:

    ```bash
    uv run marimo edit IntroToMarimo/part2_ui.py
    ```

    The [inputs reference](https://docs.marimo.io/api/inputs/) lists the
    available widgets and their options.

    ## Creating and reading widgets

    We create a widget in one cell and read its `.value` in another. marimo
    does not allow us to read a widget's value in the cell which creates it.
    Each example below has a cell for the controls followed by a cell which
    uses their values.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Numbers: `slider`, `range_slider`, `number`
    """)
    return


@app.cell
def _(mo):
    num_slider = mo.ui.slider(
        start=0, stop=100, value=25, step=5, label="slider", show_value=True
    )
    num_range = mo.ui.range_slider(
        start=0, stop=100, value=[20, 60], step=5, label="range", show_value=True
    )
    num_input = mo.ui.number(start=0, stop=100, value=42, label="number")
    mo.vstack([num_slider, num_range, num_input])
    return num_input, num_range, num_slider


@app.cell
def _(mo, num_input, num_range, num_slider):
    mo.md(f"""
    `slider` = {num_slider.value},
    `range_slider` = {num_range.value},
    `number` = {num_input.value}

    I use a `slider` when I want to experiment with a value, such as an opacity
    or a threshold. A `range_slider` selects two endpoints, which we could use
    to filter a dataset. With `number` we can type a value directly.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Boolean values: `checkbox`, `switch`
    """)
    return


@app.cell
def _(mo):
    flag_check = mo.ui.checkbox(value=True, label="checkbox")
    flag_switch = mo.ui.switch(value=False, label="switch")
    mo.hstack([flag_check, flag_switch], justify="start", gap=2)
    return flag_check, flag_switch


@app.cell
def _(flag_check, flag_switch, mo):
    mo.md(f"""
    `checkbox` = {flag_check.value}, `switch` = {flag_switch.value}

    Both controls return a `bool`. I tend to use a checkbox for an option in
    a list and a switch for turning something on or off. Try changing them
    and watch the values above.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Choosing options: `radio`, `dropdown`, `multiselect`
    """)
    return


@app.cell
def _(mo):
    pick_radio = mo.ui.radio(
        options=["Linux", "macOS", "Windows"], value="Linux", label="radio"
    )
    # map the colour names to the hex values used by our code
    pick_drop = mo.ui.dropdown(
        options={"Red": "#ff0000", "Green": "#00ff00", "Blue": "#0000ff"},
        value="Green",
        label="dropdown",
    )
    pick_multi = mo.ui.multiselect(
        options=["numpy", "pandas", "polars", "torch"],
        value=["numpy"],
        label="multiselect",
    )
    mo.vstack([pick_radio, pick_drop, pick_multi])
    return pick_drop, pick_multi, pick_radio


@app.cell
def _(mo, pick_drop, pick_multi, pick_radio):
    mo.md(f"""
    `radio` = {pick_radio.value}, `dropdown` = {pick_drop.value},
    `multiselect` = {pick_multi.value}

    A `radio` control displays all the options, whilst a `dropdown` keeps
    them in a menu. For a longer list we can add `searchable=True` to the
    dropdown. Use `multiselect` when we need more than one selection.

    Notice the dictionary used for the colours above. We see `Green` in the
    menu, but the value returned to our code is `#00ff00`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Text: `text`, `text_area`, `code_editor`
    """)
    return


@app.cell
def _(mo):
    txt_line = mo.ui.text(
        value="", placeholder="one line", label="text", full_width=True
    )
    txt_area = mo.ui.text_area(
        value="",
        placeholder="several lines",
        label="text_area",
        rows=3,
        full_width=True,
    )
    txt_code = mo.ui.code_editor(
        value="print('hello')", language="python", label="code_editor"
    )
    mo.vstack([txt_line, txt_area, txt_code])
    return txt_area, txt_code, txt_line


@app.cell
def _(mo, txt_area, txt_code, txt_line):
    mo.md(f"""
    `text` = {txt_line.value!r}, `text_area` has
    {len(txt_area.value)} characters, `code_editor` has
    {len(txt_code.value.splitlines())} lines.

    Use `text` for a single line and `text_area` for several lines. The
    `code_editor` adds syntax highlighting, which is useful when we want to
    edit a code example in a notebook. Editing the text does not execute it.

    Try typing into each control and watch the output above. The text controls
    can delay updates until we finish typing; their `debounce` option controls
    this behaviour.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Dates: `date`, `date_range`, `datetime`
    """)
    return


@app.cell
def _(mo):
    dt_day = mo.ui.date(label="date")
    dt_range = mo.ui.date_range(label="date_range")
    dt_stamp = mo.ui.datetime(label="datetime")
    mo.vstack([dt_day, dt_range, dt_stamp])
    return dt_day, dt_range, dt_stamp


@app.cell
def _(dt_day, dt_range, dt_stamp, mo):
    mo.md(f"""
    `date` = {dt_day.value}, `date_range` = {dt_range.value},
    `datetime` = {dt_stamp.value}

    Use `date` to select a day, `date_range` for a pair of dates and `datetime`
    when we also need a time. The selected values use Python's date and
    datetime types, so we can work with them without parsing a text field.
    We can set `start` and `stop` to limit the available dates.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Buttons and timers: `button`, `run_button`, `refresh`
    """)
    return


@app.cell
def _(mo):
    # add one to the current count each time we click
    act_button = mo.ui.button(
        value=0, on_click=lambda count: count + 1, label="click me", kind="success"
    )
    act_run = mo.ui.run_button(label="run", kind="warn")
    act_refresh = mo.ui.refresh(options=["1s", "5s", "10s"], label="refresh every")
    mo.hstack([act_button, act_run, act_refresh], justify="start", gap=2)
    return act_button, act_run


@app.cell
def _(act_button, act_run, mo):
    mo.md(f"""
    Button clicked {act_button.value} times; run button pressed:
    {act_run.value}.

    The `button` above uses `on_click` to add one to its current value.
    A `run_button` is useful when I want to control when a slow calculation
    runs. In the calculation cell we can use `mo.stop(not act_run.value)`
    to stop execution until the button is pressed.

    A `refresh` control updates on a timer. We could use its value in a cell
    which reads a file or requests data from an API. Here we only display
    the control; there is no polling cell.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Files: `file`, `file_browser`
    """)
    return


@app.cell
def _(mo):
    file_upload = mo.ui.file(kind="area", label="drop a file")
    file_pick = mo.ui.file_browser(label="file_browser", multiple=False)
    mo.vstack([file_upload, file_pick])
    return file_pick, file_upload


@app.cell
def _(file_pick, file_upload, mo):
    _uploaded = file_upload.value[0].name if file_upload.value else "nothing yet"
    _picked = file_pick.value[0].path if file_pick.value else "nothing yet"
    mo.md(
        f"""
    Uploaded: {_uploaded}, selected path: {_picked}

    The `file` widget uploads file contents. Its value is a list of results
    with a `.name` and `.contents` containing the bytes. This is useful when
    the browser and notebook are running on different machines.

    The `file_browser` selects a path on the machine running the notebook.
    I would use this when the data is already available on disk.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Tables: `table`
    """)
    return


@app.cell
def _(mo):
    people = [
        {"name": "Ada", "language": "Analytical Engine", "year": 1843},
        {"name": "Grace", "language": "COBOL", "year": 1959},
        {"name": "Dennis", "language": "C", "year": 1972},
        {"name": "Guido", "language": "Python", "year": 1991},
    ]
    data_table = mo.ui.table(people, selection="multi", label="pick rows")
    data_table
    return (data_table,)


@app.cell
def _(data_table, mo):
    mo.md(f"""
    We have selected {len(data_table.value)} rows.

    The `table` widget displays data with controls for searching, sorting and
    moving between pages. This example uses a list of dictionaries; we can
    also pass a pandas or polars dataframe.

    Select some rows and look at the count above. With `selection="multi"`,
    `.value` contains the selected rows. Use `selection="single"` to limit
    this to one row. For editable data see `mo.ui.data_editor`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Combining widgets: `array`, `dictionary`, `form`, `batch`

    We can combine controls and read their values together. The examples
    below show four ways of doing this.
    """)
    return


@app.cell
def _(mo):
    # array: a variable number of widgets, value comes back as a list
    comp_array = mo.ui.array([mo.ui.slider(0, 10, value=i) for i in range(3)])
    # dictionary: named widgets, value comes back as a dict
    comp_dict = mo.ui.dictionary(
        {
            "first": mo.ui.text(placeholder="first"),
            "last": mo.ui.text(placeholder="last"),
        }
    )
    # form: defer updates until Submit is pressed
    comp_form = mo.ui.dictionary(
        {"name": mo.ui.text(placeholder="name"), "age": mo.ui.number(0, 120)}
    ).form()
    # batch: embed widgets inside markdown, referenced by name
    comp_batch = mo.md("Serve {drink} at {temp}°C").batch(
        drink=mo.ui.dropdown(["tea", "coffee"], value="tea"),
        temp=mo.ui.slider(0, 100, value=80),
    )
    mo.vstack(
        [
            mo.md("`array`"),
            comp_array,
            mo.md("`dictionary`"),
            comp_dict,
            mo.md("`form` (press Submit to update its value)"),
            comp_form,
            mo.md("`batch`"),
            comp_batch,
        ]
    )
    return comp_array, comp_batch, comp_dict, comp_form


@app.cell
def _(comp_array, comp_batch, comp_dict, comp_form, mo):
    mo.md(f"""
    `array` = {comp_array.value}, `dictionary` = {comp_dict.value},
    `form` = {comp_form.value}, `batch` = {comp_batch.value}

    An `array` gives us a list of values. Here I have created three sliders
    using a list comprehension. A `dictionary` gives us named values, which
    is useful for a group of related settings.

    A `form` holds the changes until we press Submit. Its value starts as
    `None`. Try entering a name and age, then submitting the form and
    checking the output. I would use this before a slow calculation so
    editing each field does not start it again.

    The `batch` example places controls inside Markdown using named
    placeholders. Its value is a dictionary containing the drink and
    temperature.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Layout: `tabs`
    """)
    return


@app.cell
def _(mo):
    layout_tabs = mo.ui.tabs(
        {
            "Overview": mo.md("We can group related content into tabs."),
            "Details": mo.md("Use this tab for the details of an example."),
            "Notes": mo.md("I use a separate tab for notes when an example gets long."),
        }
    )
    layout_tabs
    return (layout_tabs,)


@app.cell
def _(layout_tabs, mo):
    mo.md(f"""
    Open tab: {layout_tabs.value}

    The `.value` of `tabs` tells us which tab is open. I use tabs to group
    related content when a notebook is getting long. We can also arrange
    controls with `mo.hstack`, `mo.vstack` and `mo.accordion`.

    For content which should only be evaluated when its tab is opened, see
    `mo.lazy` in the [layout reference](https://docs.marimo.io/api/layouts/).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Try it yourself

    Combine some of these controls in a small example. We could use a dropdown
    to select a colour and a slider to set the radius of a circle, then draw
    it in another cell.

    There are more specialised controls in the
    [inputs reference](https://docs.marimo.io/api/inputs/), including chart
    selections, audio input and chat. We can also use
    [anywidget](https://anywidget.dev) widgets through `mo.ui.anywidget`.

    The same pattern applies: create the widget in one cell and use its
    `.value` in another.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
