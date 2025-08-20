#!/usr/bin/env -S uvx marimo run
#

import marimo

__generated_with = "0.14.17"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Lecture 1 Introduction to Python

    This notebook contains the code for the Lecture 1 Introduction to Python. These demos are to be used in conjunction with the lecture slides to explore the concepts of Python programming language.

    ## Keywords

    Run the following example to see the keywords this version of python uses.
    """
    )
    return


@app.cell
def _():
    import keyword
    import pprint

    # here I am using the pprint module to print the list of keywords in a more readable format than print would do
    pprint.pprint(keyword.kwlist, width=60, compact=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Identifiers

    Identifiers are the names given to entities like classes, functions, variables etc. in Python. It helps differentiate one entity from another. It is important to note that Python is case-sensitive.
    >Programs are meant to be read by humans and only incidentally for computers to execute.
        Donald Knuth

    We should try to make our names easy to read and understand, this may be specific to a problem domain or using the "traditional" or "accepted" names for things (more on this when we talk about machine learning etc). Try to be clear and concise with your names.
    """
    )
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Identifier rules
    - Must Begin with a Letter or an Underscore (****): Identifiers cannot start with a number. They must start with a letter (A-Z or a-z) or an underscore ().
    - Can Contain Letters, Digits, and Underscores:
      - After the first character, identifiers can include letters, digits (0-9), and underscores.
    - Case Sensitive:
        - Identifiers are case-sensitive, meaning var, Var, and VAR would be treated as different identifiers.
    - No Reserved Keywords: Identifiers cannot be the same as Python’s reserved keywords
    """
    )
    return


@app.cell
def _():
    _a = "hello"
    prompt = "hello"
    print(prompt)
    print(_a)
    return


@app.cell
def _():
    # Other examples
    magic_number = 3
    print(f"{magic_number} is a magic number")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We will discuss more examples of naming and identifiers as we progress through the course.

    ## Data Types

    Python is a dynamically typed language, this means that variable values are checked at run-time (sometimes known as “lazy binding”). All variables in Python hold references to objects, and these references are passed to functions by value.

    Python has 5 standard data types
      - numbers
      -  string
      -  list
      -  tuple
      -  dictionary


    ## Numbers

    Python supports 3 different numerical types:
    - int (signed integers)
    - float (floating point real values)
    - complex (complex numbers)
    """
    )
    return


@app.cell
def _():
    integer_variable = 1  # note the names are not reserved words
    float_variable = 1.2
    complex_variable = 4 + 5j

    print(f"{integer_variable} is of type {type(integer_variable)}")
    print(f"{float_variable} is of type {type(float_variable)}")
    print(f"{complex_variable} is of type {type(complex_variable)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## [operations on numbers](https://docs.python.org/3/library/stdtypes.html#typesnumeric)

    We can use the typical arithmetic operators to perform operations on these numbers (and most python number types). The simple operations are shown below (and there are more complex operations available in the python documentation).


    | operation | Result |
    |--------|-------|
    | ``` x+y``` | sum of x and y |
    | ``` x-y``` | difference of x and y |
    | ``` x*y``` | product of x and y |
    | ``` x/y``` | quotient of x and y |
    | ``` x//y``` | floored quotient of x and y |
    | ``` -x``` | x negated |
    | ``` +x``` | x unchanged |
    | ``` x**y``` | x to the power y |
    """
    )
    return


@app.cell
def _():
    # modify x and y for different numeric types
    x = 2
    y = 5
    print(f"{x + y=}")
    print(f"{x - y=}")
    print(f"{x * y=}")
    print(f"{x / y=}")
    print(f"{x // y=}")
    print(f"{-x=}")
    print(f"{+x=}")
    print(f"{x ** y=}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In python integers are of unlimited size, so you can perform operations on very large numbers without worrying about overflow (unlike python 2 or other programming languages).

    Python 3 division will result in a float, to get the integer division you can use the // operator.
    """
    )
    return


@app.cell
def _():
    print(f"type(3/3)= {type(3 / 3)}")
    print(f"type(3//3)= {type(3 // 3)}")
    print(3 / 3)
    print(3 // 3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Floating Point (Real) numbers

    To represent fractions we use floating point numbers we need to be explicit as to the type of the number. We can use the float() function to convert integers to floats if required.
    """
    )
    return


@app.cell
def _():
    _a = 1
    b = 1.0
    c = float(1)
    print(f"a={_a!r} b={b!r} c={c!r}")
    print(f"type(a)={type(_a)!r}")
    print(f"type(b)={type(b)!r}")
    print(f"type(c)={type(c)!r}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Floats can be problematic, as they are actually approximations (following the IEEE 754 standard). This can lead to some strange behavior when comparing floats."""
    )
    return


@app.cell
def _():

    print(0.2 + 0.2 == 0.4)
    print(0.1 + 0.2 == 0.3)

    print(f"{0.2 + 0.2=}")
    print(f"{0.1 + 0.2=}")

    print(f"{(0.1 + 0.2).hex()=}")
    print(f"{(0.3).hex()=}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""If we really need to check floating point numbers for equality we can use the math.isclose() function. This function takes two numbers and an optional relative tolerance and an optional absolute tolerance. The function returns True if the two numbers are close enough to be considered equal."""
    )
    return


@app.cell
def _():
    import math

    print(f"{math.isclose(0.1 + 0.2, 0.3)=}")
    print(f"{math.isclose(0.1 + 0.2, 0.3, rel_tol=1e-9)=}")
    print(f"{math.isclose(0.1 + 0.2, 0.3, rel_tol=1e-15)=}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This website has a good explanation of the issues with floating point numbers in python: https://docs.python.org/3/tutorial/floatingpoint.html you can also play with this interactive demo to see how floating point numbers are represented in python: https://evanw.github.io/float-toy/

    <div class="stretch">
    <iframe src="https://evanw.github.io/float-toy/" width=1000 height=600></iframe>
    </div>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Strings

    Strings are sequences of characters, using the str type. Strings are immutable, meaning that once they are created they cannot be changed. Strings can be created using single quotes, double quotes, or triple quotes. Triple quotes are used for multi-line strings.
    """
    )
    return


@app.cell
def _():
    _hello = "hello world"
    print(_hello)
    text = 'We can use single "quotes"'
    print(text)
    text = "Triple single or double quotes\ncan be used for longer strings. These will\nBe\nprinted\nverbatim"
    print(text)
    print("Unicode strings start with u but not all terminals can print these chars 𝛑 ")
    print("Omega:  Ω")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Subsets of strings can be taken using the slice operator  (```[ ]``` and ```[ : ]``` ) with indexes starting at 0 in the beginning of the string and working their way from -1 at the end

    The plus ( ```+``` ) sign is the string concatenation operator, and the asterisk ( ```*``` ) is the repetition operator.
    """
    )
    return


@app.cell
def _():
    str = "Hello python"

    # Prints complete string
    print(f"{str=}")
    # Prints first character of the string
    print(f"{str[0]=}")
    # Prints characters starting from 3rd to 6th
    print(f"{str[2:5]=}")
    # Prints string starting from 3rd character
    print(f"{str[2:]=}")
    # Prints string two times
    print(f"{str * 2=}")
    # Prints concatenated string
    print(str + " with added text")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Lists
    A list is the most common of the Python data containers / types. It can hold mixed data, include lists of lists. A list is contained within the [] brackets and is analogous to C arrays

    Like a string data is accessed using the slice operator ( ```[ ]``` and ```[ : ]``` ) with indexes starting at 0 in the beginning of the list and working their way to end-1. The + operator concatenates and the * duplicates
    """
    )
    return


@app.cell
def _():
    list_1 = [123, "hello", 2.45, 3 + 2j]
    list_2 = [" ", "world"]
    print(f"list_1={list_1!r}")
    print(f"list_1[1]={list_1[1]!r}")
    print(f"list_1[2:]={list_1[2:]!r}")
    _hello = list_1[1] + list_2[0] + list_2[1]
    print(f"hello={_hello!r}")
    for i in range(0, len(list_1)):
        print(f"list_1[i]={list_1[i]!r}")
    for ch in list_1:
        print(f"ch={ch!r}")
    print("Using enumerate which gives us the index and value")
    for num, ch in enumerate(list_1):
        print(f"{num} {ch}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Tuples

    A tuple is a sequence of immutable Python objects. Tuples are sequences, just like lists. The differences between tuples and lists are, the tuples cannot be changed unlike lists and tuples use parentheses, whereas lists use square brackets.
    """
    )
    return


@app.cell
def _():
    tuple_1 = (123, "hello", 2.45, 3 + 2j)
    tuple_2 = (" ", "world")
    print(f"tuple_1={tuple_1!r}")
    print(f"tuple_1[1]={tuple_1[1]!r}")
    print(f"tuple_1[2:]={tuple_1[2:]!r}")
    _hello = tuple_1[1] + tuple_2[0] + tuple_2[1]
    print(f"hello={_hello!r}")
    tuple_1[0] = 3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Slice operations

     In Python, slice operators are used to extract parts of sequences like strings, lists, tuples, and other iterable objects. Slicing allows you to access a subset of the elements in these sequences using a specific syntax.

    The basic syntax for slicing is:

    ```sequence[start:stop:step]```

    - start: The index where the slice starts (inclusive). If omitted, it defaults to the beginning of the sequence (0).
    - stop: The index where the slice ends (exclusive). If omitted, it defaults to the end of the sequence.
    - step: The interval between elements in the slice. If omitted, it defaults to 1.
    """
    )
    return


@app.cell
def _():
    _a = list(range(10))
    print(f"a[::1]={_a[::1]!r}")
    print(f"a[::-1]={_a[::-1]!r}")
    print(f"a[1:10:2]={_a[1:10:2]!r}")
    print(f"a[:-1:1]={_a[:-1:1]!r}")
    del _a[::2]
    print(f"a={_a!r}")
    print(f"list(range(10))[slice(0, 5, 2)]={list(range(10))[slice(0, 5, 2)]!r}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Dictionaries

    Python dictionaries are a powerful key / value data structure which allows the storing of different data types in the same data set, they are also the core foundation on which the whole python language is built. Dictionaries are enclosed by curly braces ( ```{}``` ) and values can be assigned and accessed using square braces ( ```[]``` ).

    We use the key to access the value in the dictionary. The key must be unique and immutable (meaning it cannot be changed). The value can be any data type (including lists, tuples, and other dictionaries).

    It may take time to understand when and how to use dictionaries, but they are a powerful tool in python and are used extensively in the language.
    """
    )
    return


@app.cell
def _():
    colours = {
        "red": [1, 0, 0],
        "green": [0, 1, 0],
        "blue": [0, 0, 1],
        "white": [1, 1, 1],
        "black": [0, 0, 0],
    }

    print(f"{colours.get('red')=}")
    print(f"{colours.get('green')=}")
    print(f"{colours.get('purple')=}")
    # can also use
    print(f"{colours['white']=}")
    # but
    print(f"{colours['purple']=}")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
