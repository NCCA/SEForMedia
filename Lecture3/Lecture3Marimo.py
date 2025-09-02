#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Lecture 3 Sequence, Selection, and Iteration

    In this lecture we are going to look at the three basic control structures in programming: sequence, selection, and iteration. We will also look at how to write functions in Python.

    ## Sequences

    As it's name suggests, a sequence is a series of steps that are executed in order. In Python, the sequence is the default control structure. When you write a series of statements in a Python script, they are executed in order from top to bottom.  This is also true when typing in the REPL.
    """
    )
    return


@app.cell
def _():
    _i = 1
    print(f"Sequence i={_i!r}")
    _i = _i + 1
    print(f"Sequence i={_i!r}")
    _i = _i + 1
    print(f"Sequence i={_i!r}")
    _i = _i + 1
    print(f"Sequence i={_i!r}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    As you can see from the code block above, there is a lot of repitition in the code. Whilst this is not really an issue for small amounts of code, it can become a problem as the code base grows.

    This is where the idea of functions come into play, functions allow us to encapsulate a block of code that can be reused multiple times.

    ## Functions

    A function is a block of code which only runs when it is called. It needs to be defined before it is called. You can pass data, known as parameters, into a function. A function can return data as a result. The syntax for defining a function in Python is as follows:

    ```python
    def function_name(parameters):
        # code block
        return value
    ```

    The `def` keyword is used to define a function. The function name is the name of the function. The parameters are the values that are passed into the function. The code block is the code that is executed when the function is called. The `return` keyword is used to return a value from the function.

    Not all functions require parameters, some functions require multiple parameters.

    Not all functions need to return a value, some functions are used to perform an action and do not need to return a value.

    |Component	 | Meaning |
    |------------|---------|
    |```def```	| The keyword that informs Python that a function is being defined |
    | ```<function_name>``` |	A valid Python identifier that names the function |
    | ```<parameters>``` |	An optional, comma-separated list of parameters that may be passed to the function |
    | ```:``` |	Punctuation that denotes the end of the Python function header (the name and parameter list) |
    | ```<statement(s)>``` |	A block of valid Python statements |

    When defining functions, we need to indent the code block that makes up the function. This happens after the colon `:` at the end of the function definition. All code that is indented after the colon is part of the function. Care must be taken to ensure that the indentation is consistent throughout the function, and follows the [PE8 style guide](https://peps.python.org/pep-0008/#function-and-variable-names).

    In Python notebooks we can use the `def` keyword to define a function and then call the function in the same cell, this will also be available in other cells in the notebook after it has been defined. The same is true for scripts and the REPL.

    ### [indentation](https://www.python.org/dev/peps/pep-0008/)

    - Python uses indentation to block code
      - convention states we use 4 spaces for indentation (see PEP-8)
    - This is unusual as most programming languages use {}
    - This can lead to problem, especially when mixing tabs and spaces (python 3 doesn't allow this)
    - I will show different examples of this as we go
    - usually this will follow a statement and the ```:``` operator to indicate the start of the block

    ## Example 1 a simple add function
    """
    )
    return


@app.cell
def _():
    def add(a, b):
        return _a + _b

    _a = 1
    _b = 2
    c = add(_a, _b)
    print(f"c={c!r}")
    str1 = "hello"
    str2 = " python"
    result = add(str1, str2)
    print(f"result={result!r}")
    print(f"add(1.0,2)={add(1.0, 2)!r}")
    return (add,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""You will note that the function add can take any type of input and return the result. This is because Python is a dynamically typed language. This means that the type of the variable is determined at runtime. This is different from statically typed languages like C++ or Java where the type of the variable is determined at compile time. This is a powerful feature of Python but can also lead to bugs if you are not careful. For example if you try to pass a string and a number to the add function it will throw an error."""
    )
    return


@app.cell
def _():
    add("hello", 2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Functions in practice

    A function needs to be declared before it is called, however they can be placed in external files / modules (more on this  in a future lecture). Python 3.5 also added a feature called type hints which allows you to specify the type of the input and output of a function. This is not enforced by the interpreter but can be used by IDEs to provide better code completion and error checking.

    For example if we wished the ```add``` function to only use numbers we could use the following type hint.

    `
    """
    )
    return


@app.function
def add(a: int, b: int) -> int:
    return a + b


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Whilst this does take more time type hints are part of modern python and should be used whenever possible.  There is a good [cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html) for type hints on the mypy website."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Let's design a function

    In one of the earlier examples we used the turtle to draw two squares, This is idea for a function so we can design one but what do we need?
      - a [name](https://en.wikipedia.org/wiki/Naming_convention_%28programming%29 )
      - useful parameters (and perhaps useful defaults)
      - possible return values

    One of the things to consider when designing function is the problem domain, in this case drawing a square, what are the key components of a square?
      - position
      - size
      - color

    Position / shape is a complex one, how do we define the position of a square?

      1. Top (x,y) Bottom (x,y)
      2. Start Pos (x,y) Width Height
      3. Center (x,y) Width / Height

    Which one to choose is a matter of design (the hard part), it ays to be consistent with other functions etc.

    ##
    """
    )
    return


@app.cell
def _():
    import turtle
    from typing import Type

    def square(
        turtle: Type[turtle], x: float, y: float, width: float, height: float
    ) -> None:
        """
        Draws a square using the given turtle object.

        Args:
            turtle (Turtle): The turtle object used to draw the square.
            x (float): The x-coordinate of the starting position.
            y (float): The y-coordinate of the starting position.
            width (float): The width of the square.
            height (float): The height of the square.

        Returns:
            None
        """
        turtle.penup()
        turtle.goto(x, y)
        turtle.pendown()
        turtle.forward(width)
        turtle.left(90)
        turtle.forward(height)
        turtle.left(90)
        turtle.forward(width)
        turtle.left(90)
        turtle.forward(height)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""In the cell above we have defined the square function, it can now be called from our code below. It is important that the above cell has been run before the cell below."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    You will now  notice if you hover over the square function you will see the signature of the function and the type hints. This is a useful feature of Jupyter notebooks and can be used to check the type hints of a function.

    This is generated from the [docstring](https://www.python.org/dev/peps/pep-0257/) which is a way of documenting your code and is very useful for others (and yourself) to understand what the function does. It helps to do this when you are designing the function as it helps to clarify what the function should do. AI-Tools are also very good at generating typehints and docstrings from code, however it is always best to write these yourself.

    ## Functions in practice

    It is important to note that functions can be defined in any order in a script, however they must be defined before they are called. This is because Python is an interpreted language and reads the script from top to bottom. If a function is called before it is defined, Python will throw an error.

    We can also put function into different files, packages and modules. We will look in depth at generating more complex python packages in a future lecture, but for now we will see how we can add a function to a different file.

    The file [jupyter_square.py](jupyter_square.py) contains the square function, we can import this function into our script using the `import` keyword. This will allow us to use the square function in our script.

    The following example is going to generate a series of squares using the square function and random module to make a very bad attempt at generating Rothko style art.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    As you can see I have introduced some new concepts in the above code, iteration (looping) and selection (if statements). We will start to look at these in more details and how we can combine them into more complex programs.


    # Selection

    Most programming languages have a way of making decisions based on the value of a variable. This is known as selection. In Python, the `if` statement is used to make decisions based on the value of a variable. The syntax for the `if` statement is as follows:

    ```python
    if condition:
        # code block
    ```

    The same rules for code indetation apply to the `if` statement as they do for functions. The code block that is executed if the condition is true must be indented. The `if` statement can also be followed by an `else` statement. The `else` statement is executed if the condition is false. The syntax for the `else` statement is as follows:

    if statements need work on boolean values, these are values that are either true or false. In Python, the following values are considered false:

    - `False`
    - `None`
    - `0`
    - `0.0`
    - `''`
    - `[]`
    - `()`
    - `{}`
    - `set()`
    - `range(0)`
    - `0j`
    - `Decimal(0)`
    - `Fraction(0, 1)`
    - objects for which `bool(obj)` returns `False`

    All other values are considered true. This is important to remember when writing `if` statements.

    The following example show a simple if statement in action
    """
    )
    return


@app.cell
def _():
    _value = input("Enter some text and press enter : ")
    if len(_value) > 10:
        print("the length of the string is over 10")
    else:
        print("the length of the string is under 10")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## elif

    in the previous example we had an if followed by an else, it is also possible to have an `elif` statement. This is short for else if and allows for multiple conditions to be checked. The syntax for the `elif` statement is as follows:
    """
    )
    return


@app.cell
def _():
    try:
        number = int(input("please enter a number between 1 and 100 : "))
    except ValueError:
        print("you did not enter a number")

    if number < 1 or number > 100:
        print("the number is not between 1 and 100")
    elif number < 50:
        print("the number is less than 50")
    else:
        print("the number is greater than 50")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## try / except

    Note in this example we use a try / except block to catch the exception if the user enters a non number, this is a common pattern in python to catch exceptions we will look at this in more detail in a later lecture / lab.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Python Comparison Operators
    <small>given ```a=10 b=20```</small>

    | <small>Operators    </small>              | <small>Description               </small>                                                                 | <small>Example  </small>          |
    |----------------------------|--------------------------------------------------------------------------------------------|--------------------|
    | <small>```==```    </small>                | <small>equality operator returns true if values are the same </small>                                     | <small>(a==b) is not true  </small>|
    | <small>```!=```  </small>                  | <small>not equal operator             </small>                                                            | <small>(a!=b) is true   </small>   |
    | <small>```>```    </small>                 | <small>Checks if the value of left operand is greater than the value of right operand  </small>           |  <small>(a>b) is not true  </small> |
    | <small>```<```    </small>                 |  <small>Checks if the value of left operand is less than the value of right operand   </small>              |  <small>(a>b) is true     </small>  |
    | <small>```>=```    </small>                |  <small>Checks if the value of left operand is greater than or equal to the value of right operand  </small>| <small> (a>=b) is not true </small> |
    | <small>```<=```    </small>                |  <small>Checks if the value of left operand is less than or equal to the value of right operand   </small>  | <small> (a<=) is true   </small>    |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _():
    _a = 10
    _b = 20
    print(f"a={_a!r}, b={_b!r}")
    print(f"a==b={_a == _b!r}")
    print(f"a!=b={_a != _b!r}")
    print(f"a>b={_a > _b!r}")
    print(f"a<b={_a < _b!r}")
    print(f"a>=b={_a >= _b!r}")
    print(f"a<=b={_a <= _b!r}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Python Logical Operators

    <small>given ```a=true b=False```</small>

    | <small>Operators    </small>              | <small>Description               </small>                                                                 | <small>Example  </small>          |
    |----------------------------|--------------------------------------------------------------------------------------------|--------------------|
    | <small>```and```    </small>                | <small>Logical and </small>                                     | <small>a and b is False  </small>|
    | <small>```or```  </small>                  | <small>Logical or             </small>                                                            | <small>a or b is True   </small>   |
    | <small>```not```   </small>| <small>Logical not   </small>                                    | <small>not (a and b) is True  </small>   |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _():
    _a = True
    _b = False
    print(f"a={_a!r}, b={_b!r}")
    print(f"a and b={_a and _b!r}")
    print(f"a or b={_a or _b!r}")
    print(f"not(a and b)={not (_a and _b)!r}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## A note on style

    The following code whilst correct is not considered good style in Python.

    ```python
    if a == True:
        print("a is True")
    ```

    flake8 will throw a warning if you use this code. The correct way to write this is as follows:

    ```bash
    E712 comparison to True should be 'if cond is True:' or 'if cond:'
    ```

    Typically you would write the code as follows:

    ```python
    if a:
        print("a is True")
    ```

    ## Complex selections

    Selections can be embedded to create quite complex hierarchies of “questions”, this can sometimes make reading code and maintenance hard especially with the python white space rules as code quite quickly becomes complex to read

    We usually prefer to put complex sequences in functions to make the code easier to read / maintain, we can also  simplify these  using set operators such as ```in```.

    Python 3.10 and above also has a new feature called [match](https://docs.python.org/3/library/match.html) which is a more powerful version of the switch statement in other languages.
    """
    )
    return


@app.cell
def _():
    format = "png"

    match format:
        case "png":
            print("PNG format selected")
        case "jpeg":
            print("JPEG format selected")
        case "gif":
            print("GIF format selected")
        case _:
            print("Unknown format selected")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # iteration

    Iteration is the ability to repeat sections of code. Python has two main looping constructs:
      - ```for``` (each)
      - ```while```

    ```for``` each loops operate on ranges of data, ```while``` loops repeat while a condition is met

    ## [for](https://docs.python.org/3/tutorial/controlflow.html#for-statements)

    A for loop is used for iterating over a sequence built in types such as ```list```, ```tuple```, ```dictionary```, ```set``` and  ```string``` will work by default.

    Also any iterable  object that can return one of its elements at a time can be used in a for loop. This is known as [iterable](https://docs.python.org/3/glossary.html#term-iterable).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _():
    list_of_ints = [1, 2, 3, 4, 5]
    tuple_of_strings = ("a", "b", "c")
    a_string = "hello loops"
    for _i in list_of_ints:
        print(f"i={_i!r}")
    print()
    for _i in tuple_of_strings:
        print(f"i={_i!r}")
    print()
    for _i in a_string:
        print(f"i={_i!r}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## range

    The range function is a useful function for generating a sequence of numbers. It is common to use this in conjunction with the for loop to generate a sequence of numbers. The range function can take up to three arguments, the start, stop, and step. The syntax for the range function is as follows:

    ```python
    range(start,stop,step)
    ```

    start is the first number in the sequence, stop is the last number in the sequence, and step is the difference between each number in the sequence. If the start and step arguments are not provided, they default to 0 and 1 respectively. The range function generates a sequence of numbers from start to stop - 1. The range function is a generator function, this means that it does not generate all the numbers at once, but generates them one at a time. This is useful when working with large sequences of numbers.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _():
    for _i in range(5):
        print(f"i={_i!r}")
    for _x in range(1, 6):
        print(f"x={_x!r}")
    for j in range(1, 10, 2):
        print(f"j={j!r}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## [```break```](https://docs.python.org/3/tutorial/controlflow.html#break-and-continue-statements-and-else-clauses-on-loops)

    The ```break``` clause allows us to jump out of a loop, tt is usually used in conjunction with an if statement and will break out of the loop if the condition is met.
    """
    )
    return


@app.cell
def _(random):
    numbers = [random.uniform(-1, 10) for _i in range(0, 10)]
    print(numbers)
    for n in numbers:
        print(f"n={n!r}")
        if n < 0.0:
            print("found a negative exiting {}".format(n))
            break
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## [```continue```](https://docs.python.org/3/tutorial/controlflow.html#break-and-continue-statements-and-else-clauses-on-loops)

    ```continue``` will stop the current loop and jump to the next item, this is useful if you want to skip an item in a loop.
    """
    )
    return


@app.cell
def _():
    _a = range(0, 10)
    even = []
    for _i in _a:
        if _i % 2:
            continue
        even.append(_i)
    print(f"even={even!r}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##  [```-```](https://docs.python.org/3/tutorial/controlflow.html#break-and-continue-statements-and-else-clauses-on-loops)

    It is convention in Python to use ```_```  as a general purpose "throwaway" variable name, it is very common to use this multiple times in a file or module.
    """
    )
    return


@app.cell
def _():
    sum = 2
    for _ in range(0, 10):
        sum = sum + sum
    print(f"sum={sum!r}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## dictionary iteration

    The dictionary ```items()``` method returns the key and the value, we could use just ```keys()``` or ```values()``` to get the individual elements.
    """
    )
    return


@app.cell
def _():
    colours = {"red": [1, 0, 0], "green": [0, 1, 0], "blue": [0, 0, 1]}
    print(f"colours.items()={colours.items()!r}")
    for colour, _value in colours.items():
        print(f"colour={colour!r} := value={_value!r}")
    for colour in colours.keys():
        print(f"colour={colour!r}")
    for values in colours.values():
        print(f"values={values!r}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## the ```while``` statement

    The while loop is used to iterate over a block of code as long as the condition is true. The syntax for the while loop is as follows:

    ```python
    while condition:
        # code block
    ```
    The same indentation rules apply to the while loop as they do to the for loop and the if statement. The code block that is executed if the condition is true must be indented. Both the break and continue statements can be used in the while loop. The break statement will break out of the loop if the condition is met, and the continue statement will skip the current iteration of the loop.
    """
    )
    return


@app.cell
def _():
    _i = 10
    while _i >= 0:
        print(f"i={_i!r}")
        _i = _i - 1
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
