#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # marimo UI widgets — what they are and when to reach for them

    marimo ships its interactive widgets under `mo.ui`. Every one of them is a
    Python object with a `.value` attribute, and whenever you change it in the
    browser marimo re-runs the cells that read that `.value`. That reactive
    loop is the whole point — you get an interface for free without writing any
    callbacks.

    This notebook walks through the lot, grouped by the job they do, with a
    live example of each and a short note on when I would actually use it. Run
    it with:

    ```bash
    uvx marimo edit marimo_ui_widgets.py
    ```

    The full reference is in the [inputs docs](https://docs.marimo.io/api/inputs/).

    ## The one rule worth learning first

    Define a widget in one cell and read its `.value` in a *different* cell. If
    you create the widget and read its value in the same cell, interacting with
    it re-runs that cell, rebuilds the widget from scratch and throws your input
    away. Every example below follows that pattern: a cell that builds the
    widgets, then a cell that reacts to them.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Numbers — `slider`, `range_slider`, `number`
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
    `slider` = **{num_slider.value}**,
    `range_slider` = **{num_range.value}**,
    `number` = **{num_input.value}**

    Reach for a **slider** when the exact figure doesn't matter and you want
    people to explore a range by feel — a threshold, an opacity, a year. Use
    **range_slider** when you're picking a *window* rather than a point, like a
    min/max filter. Use **number** when the precise value does matter and typing
    `42` is quicker than dragging to it.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Yes/no — `checkbox`, `switch`
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
    `checkbox` = **{flag_check.value}**, `switch` = **{flag_switch.value}**

    They return the same thing — a `bool` — so the choice is purely about
    tone. A **checkbox** reads as "tick this option" in a list of settings; a
    **switch** reads as "turn this feature on/off". Pick whichever matches the
    mental model and stay consistent.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Picking from a set — `radio`, `dropdown`, `multiselect`
    """)
    return


@app.cell
def _(mo):
    pick_radio = mo.ui.radio(
        options=["Linux", "macOS", "Windows"], value="Linux", label="radio"
    )
    # a dict maps a nice label to the value your code actually gets back
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
    `radio` = **{pick_radio.value}**, `dropdown` = **{pick_drop.value}**,
    `multiselect` = **{pick_multi.value}**

    All three choose from a fixed list. **radio** shows every option at once —
    good for two to five choices where seeing them all helps. **dropdown**
    hides them until clicked, so it's the one for long lists (pass
    `searchable=True` when there are lots), and note the dict trick above: the
    user sees `Green`, your code gets `#00ff00`. **multiselect** is the same
    idea when more than one answer is allowed.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Text — `text`, `text_area`, `code_editor`
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
    `text` = **{txt_line.value!r}**, `text_area` has
    **{len(txt_area.value)}** characters, `code_editor` has
    **{len(txt_code.value.splitlines())}** line(s).

    **text** is a single line — names, search boxes, a URL; it also does
    `kind="password"` and `"email"`. **text_area** is the multi-line version
    for prompts or free-form notes. **code_editor** adds syntax highlighting,
    which is handy in teaching material when you want people to edit a snippet
    and feed it back into the notebook. Note text inputs debounce by default,
    so the value updates when you stop typing rather than on every keystroke.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Dates — `date`, `date_range`, `datetime`
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
    `date` = **{dt_day.value}**, `date_range` = **{dt_range.value}**,
    `datetime` = **{dt_stamp.value}**

    Each hands back a real `datetime.date`/`datetime.datetime`, not a string,
    so no parsing at your end. Use **date** for a single day, **date_range**
    for a from/to filter over a dataframe, and **datetime** when the time of
    day matters. All three take `start`/`stop` to fence off the allowed range.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Actions — `button`, `run_button`, `refresh`
    """)
    return


@app.cell
def _(mo):
    # on_click gets the current value and returns the next one — a click counter
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
    button clicked **{act_button.value}** times; run_button pressed:
    **{act_run.value}**.

    These are for *actions* rather than values. A plain **button** carries
    whatever `value` you compute in `on_click` — above it counts clicks.
    **run_button** is the escape hatch from reactivity: gate an expensive cell
    behind `if not run.value: mo.stop()` so it only fires when asked.
    **refresh** re-runs its dependents on a timer, which is what you want for
    polling a file or an API. (There's also `mo.ui.microphone` for audio and
    `mo.ui.file`/`file_browser` next.)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Files — `file`, `file_browser`
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
        uploaded: **{_uploaded}**, browsed: **{_picked}**

        **file** uploads a file's *contents* into the notebook — the value is a
        list of results with `.name` and `.contents` (bytes). That's the one for a
        deployed app where the user is on another machine. **file_browser** picks a
        *path* on the machine the notebook is running on, which is more useful for
        local work where the data is already on disk.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Tabular data — `table`
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
    you selected **{len(data_table.value)}** row(s).

    **table** renders a list of dicts, a dataframe (pandas or polars) or a
    dict of columns, with search, sorting and paging built in. Set
    `selection="single"` or `"multi"` and `.value` gives you back the chosen
    rows — a clean way to let someone pick records and drive the next cell off
    them. For editing cells in place look at `mo.ui.data_editor`, and for a
    no-code column explorer there's `mo.ui.dataframe`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Composing widgets — `array`, `dictionary`, `form`, `batch`

    The four below don't add new controls; they combine the ones you've seen.
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
            mo.md("**array**"),
            comp_array,
            mo.md("**dictionary**"),
            comp_dict,
            mo.md("**form** (nothing updates until you Submit)"),
            comp_form,
            mo.md("**batch**"),
            comp_batch,
        ]
    )
    return comp_array, comp_batch, comp_dict, comp_form


@app.cell
def _(comp_array, comp_batch, comp_dict, comp_form, mo):
    mo.md(f"""
    `array` = **{comp_array.value}**, `dictionary` = **{comp_dict.value}**,
    `form` = **{comp_form.value}**, `batch` = **{comp_batch.value}**

    Use **array** when the *number* of widgets isn't known ahead of time —
    one slider per column of a dataframe, say. Use **dictionary** for the same
    idea with meaningful keys instead of positions. Wrap anything in **form**
    when reacting on every keystroke would be wasteful (a slow query, a model
    run): the value stays `None` until Submit, then updates in one go. **batch**
    is for laying controls out inside a sentence of markdown rather than
    stacking them.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Layout — `tabs`
    """)
    return


@app.cell
def _(mo):
    layout_tabs = mo.ui.tabs(
        {
            "Overview": mo.md("Tabs organise content — the value is the open tab."),
            "Details": mo.md("Put a heavy widget in a lazy tab to defer its work."),
            "Notes": mo.md("Handy for keeping a dense notebook navigable."),
        }
    )
    layout_tabs
    return (layout_tabs,)


@app.cell
def _(layout_tabs, mo):
    mo.md(f"""
    open tab: **{layout_tabs.value}**

    **tabs** is layout rather than input, but it still reports which tab is
    open via `.value`. Pair it with `mo.hstack`/`mo.vstack` and `mo.accordion`
    to keep a widget-heavy notebook from turning into one long scroll. Set
    `lazy=True` so a tab's contents only compute when it's first opened.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Where to go next

    That covers the everyday widgets. A few specialised ones I've skipped
    because they need extra libraries or a real device: `mo.ui.altair_chart`,
    `mo.ui.plotly` and `mo.ui.matplotlib` make plots that report selections
    back as `.value`; `mo.ui.microphone` and `mo.ui.chat` are for audio and LLM
    chat; `mo.ui.anywidget` lets you drop in any
    [anywidget](https://anywidget.dev). The full list is in the
    [inputs reference](https://docs.marimo.io/api/inputs/).

    The pattern is always the same: build the widget in one cell, read
    `.value` in another, and let marimo work out what re-runs.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
