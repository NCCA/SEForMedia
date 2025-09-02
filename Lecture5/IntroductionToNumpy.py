#!/usr/bin/env uv run marimo edit

import marimo

__generated_with = "0.14.17"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # [Numpy](https://numpy.org/)

    This notebook is a companion to the lecture slides [here](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/lectures/Lecture5/), it will be used to introduce numpy as well as other concepts. Note that we are still using Numpy 1.26.x so some of the newer features may not be available. For full documentation see [here](https://numpy.org/doc/1.26/index.html).

    ## Getting Started

    numpy is already installed as part of the Anaconda distribution, so you should be able to import it directly. If you are using a different Python distribution you may need to install it using pip. The following code cell will import numpy and check the version you are using.
    """
    )
    return


@app.cell
def _():
    import numpy as np

    print(np.__version__)
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""In the above example we import ```numpy as np``` this is a common convention as it makes it easier to type and read the code. We are creating an alias called ```np``` that contains all the numpy functions. This is a common convention in python programming."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Arrays

    The core data type in numpy is the array, this is similar to a list but can be multi-dimensional. They can be created in a number of ways for example

    1. Instantiated using values from a normal Python list
    2. Filled with sequential values upto a target number
    3. Filled with a specific value
    4. Filled with random values (various random distributions [available](https://numpy.org/doc/1.26/reference/random/index.html))


    For a full list of array creation routines, please see [here](https://numpy.org/doc/1.26/reference/routines.array-creation.html)
    """
    )
    return


@app.cell
def _(np):
    # Array creation from python list
    python_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    np_from_list = np.array(python_list)
    print(np_from_list, type(np_from_list))
    return (np_from_list,)


@app.cell
def _(np):
    # Array creation with sequential values
    # arange always returns a 1D array, will need to reshape it separately if
    # we wish to do so
    seq_array = np.arange(20)
    print(seq_array)
    return (seq_array,)


@app.cell
def _(np):
    # Array creation with random values
    random_array = np.random.rand(
        10
    )  # returns an array filled with 10 random floats between 0 and 1
    print(random_array)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _(np):
    # Array filled with the same specific value
    filled_array = np.full(10, 255)  # returns an array filled with 10 instances of 255
    print(filled_array)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Array Shape

    In NumPy, an array’s shape refers to the dimensions of the array, specifying how many elements it contains along each axis. The shape of a NumPy array is represented as a tuple of integers, where each integer represents the size of the array along that dimension.

    This is a key concept in numpy as it allows you to create multi-dimensional arrays. For example, a 2D array has a shape of (rows, columns) and a 3D array has a shape of (depth, rows, columns). We use this to represent our data when we input it into a neural network.

    So far we have only created 1D arrayas, but we can create multi-dimensional arrays by passing in a tuple of values to the array function.

    **1D Array (Vector):**
    - Shape: (n,) where n is the number of elements.
    """
    )
    return


@app.cell
def _(np):
    arr = np.array([1, 2, 3])
    print(arr.shape)  # Output: (3,)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **2D Array (Matrix):**

    - Shape: (n, m) where n is the number of rows and m is the number of columns.
    """
    )
    return


@app.cell
def _(np):
    arr_1 = np.array([[1, 2], [3, 4], [5, 6]])
    print(arr_1.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **3D Array:**
    - Shape: (n, m, p) where n is the number of matrices, m is the number of rows, and p is the number of columns in each matrix.
    """
    )
    return


@app.cell
def _(np):
    arr_2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(arr_2.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Why Do We Need Array Shapes?

    1.	Understanding Data Structure: Knowing the shape of an array tells you how the data is organized. For example, whether the data is a single vector, a 2D matrix, or a multi-dimensional dataset is important for operations like indexing, reshaping, or applying functions.
    2.	Efficient Computation: Many mathematical operations in NumPy, like matrix multiplication, element-wise operations, or broadcasting, depend on the array’s shape. Mismatched shapes can cause errors, so understanding the shape helps ensure proper operation.

    ## Reshaping Arrays

    Reshaping an array means changing the shape of the array without changing the data it contains. This is a common operation when working with neural networks as we often need to change the shape of the data to fit the input layer of the network.

    One of the many things that set Numpy arrays apart from Python lists is their `reshape` method. This allows you to define the number of dimensions that the structure represented by the array has. For example, the array `np_from_list` that we have just defined is just a flat list of numbers at the present. `reshape` helps us represent it as a 2D data structure (a matrix). Keep in mind that the shape that you put into this method must agree with the number of elements in the array. In our example, the array has 10 elements, so acceptable shapes include:
    * 5, 2
    * 2, 5
    * 1, 10
    * 10, 1
    """
    )
    return


@app.cell
def _(np_from_list):
    matrix_5_2 = np_from_list.reshape(5, 2)
    print(matrix_5_2)
    print(f"{matrix_5_2.shape=}")
    return (matrix_5_2,)


@app.cell
def _(seq_array):
    matrix_2_10 = seq_array.reshape(2, 10)
    print(matrix_2_10)
    print(f"matrix_2_10 shape: {matrix_2_10.shape}")
    return


@app.cell
def _(matrix_5_2, np):
    # You can also pass a desired shape into the array creation function to instantiate
    # an array with a pre-defined shape
    array_from_shape = np.full(matrix_5_2.shape, 32)
    array_from_shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Exercises

    The following cells have been set up to help you learn how to use numpy. The comments in the cells will ask you to complete a simple numpy task. Experiment with the code and see what happens.

    ### Task 1
    """
    )
    return


@app.cell
def _():
    # Create a Numpy array A that contains the integers from 0 - 31 (32 elements long)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <details>

    <summary>Solution</summary>

    ```python
    A = np.arange(32)
    print(A)
    ```

    </details>

    ### Task 2
    """
    )
    return


@app.cell
def _():
    # Create a Numpy array B that contains 12 random numbers between 0 and 1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <details>

    <summary>Solution</summary>

    ```python
    B = np.random.rand(12)
    print(B)
    ```

    </details>

    ## Task 3
    """
    )
    return


@app.cell
def _():
    # Turn A into a 2d matrix of shape (8, 4) - 8 rows, 4 columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <details>

    <summary>Solution</summary>

    ```python
    A = A.reshape(8, 4)
    print(A)
    ```

    </details>

    ## Task 4
    """
    )
    return


@app.cell
def _():
    # Turn B into a 2d matrix of shape (4, 3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <details>

    <summary>Solution</summary>

    ```python
    B = B.reshape(4, 3)
    print(B)
    ```

    </details>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Array operations

    Once you have created an array, there are various operations you may want to perform on them. We'll go through some of the most common ones in the next few cells.

    ### Binary operators

    Similar to normal counting numbers, Numpy arrays can be used as arguments to binary arithmetic operators like ```*``` (multiplication), ```+``` (addition), ```-```(subtraction) and ```/```(division). However, these can only be applied to arrays under certain conditions:

    1. The arrays have the same shape, OR
    2. The operation is between an array and a scalar, OR
    3. The operations is between 2 arrays of different shapes that can be "broadcast" together - more on this later

    These operations are applied **element-wise** meaning each element in the array is combined with its corresponding element at the same position in the other array
    """
    )
    return


@app.cell
def _(np):
    arr_3 = np.random.rand(32).reshape((8, 4))
    print(arr_3.shape)
    print(arr_3)
    scalar_add_array = 11 + arr_3
    print(scalar_add_array)
    return (arr_3,)


@app.cell
def _(arr_3, np):
    A = np.full(32, 5).reshape((8, 4))
    sub_array = arr_3 - A
    print(sub_array)
    return (A,)


@app.cell
def _(A, arr_3):
    mul_array = arr_3 * A
    print(mul_array)
    return


@app.cell
def _(arr_3):
    _div_array = 11 / arr_3
    print(_div_array)
    return


@app.cell
def _(arr_3):
    _div_array = arr_3 / 0
    print(_div_array)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Broadcasting

    As long as an array's dimensions are compatible with their counterparts in the other array, they can be "broadcasted" together. This means that numpy will 'strectch' the smaller dimension along its axis so that element-wise operations can be applied between them. In general, 2 arrays can be broadcast together if each dimension in one array is compatible with its counterpart in the other array. Dimensions are compatible if they are equal or one of them is equal to 1. The dimension of size 1 is expanded to fit the size of its counterpart see [broadcasting rules](https://numpy.org/doc/1.26/user/basics.broadcasting.html) for more information.
    """
    )
    return


@app.cell
def _(np):
    _a = np.array([1.0, 2.0, 3.0])
    _b = np.array([2.0, 2.0, 2.0])
    _a * _b
    return


@app.cell
def _(np):
    _a = np.array([1.0, 2.0, 3.0])
    _b = 2.0
    _a * _b
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""![broadcasting](images/broadcast.png) (source: [numpy.org](https://numpy.org/doc/1.26/user/basics.broadcasting.html))"""
    )
    return


@app.cell
def _(np):
    array_4_1 = np.arange(4).reshape(4, 1)
    print(array_4_1, array_4_1.shape)
    B = np.random.rand(12).reshape(4, 3)
    _broadcasted = array_4_1 + B
    print(_broadcasted, _broadcasted.shape)
    return (B,)


@app.cell
def _(B, np):
    array_1_3 = np.arange(3).reshape(1, 3)
    print(array_1_3, array_1_3.shape)
    _broadcasted = array_1_3 * B
    print(_broadcasted, _broadcasted.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Array Products

    Contrary to the `*` operator, which performs element-wise multiplication between the arrays, `np.matmul` is the matrix product of the arrays.
    """
    )
    return


@app.cell
def _(np):
    A_1 = np.random.rand(32).reshape(8, 4)
    B_1 = np.random.rand(12).reshape(4, 3)
    _C = np.matmul(A_1, B_1)
    print(f"A shape: {A_1.shape}, B shape: {B_1.shape}, AB shape: {_C.shape}")
    print(_C)
    return A_1, B_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Newer versions of python introduced the `@` operator which is equivalent to `np.matmul` for 2D arrays (it is an overloaded operator). For 1D arrays, it is equivalent to the dot product."""
    )
    return


@app.cell
def _(A_1, B_1):
    _C = A_1 @ B_1
    print(_C, _C.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Exercises

    ### Task 1
    """
    )
    return


@app.cell
def _(np):
    B_2 = np.random.rand(4, 3)  # noqa F841
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <details>

    <summary>Solution</summary>

    ```python
    C = B / 2.5
    print(C)
    ```

    </details>

    ### Task 2
    """
    )
    return


@app.cell
def _():
    # Create a new array filled with -1. The array must have the same shape as B. Store the result in a variable D
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <details>

    <summary>Solution</summary>

    ```python
    D = np.full(12, -1).reshape(B.shape)
    print(D)
    ```

    </details>

    ### Task 3
    """
    )
    return


@app.cell
def _():
    # Multiply this new array ELEMENT-WISE with B and store the result in a variable called E.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <details>

    <summary>Solution</summary>

    ```python
    E = D * C
    print(E)
    ```

    </details>

    ### Task 4
    """
    )
    return


@app.cell
def _(np):
    # Consider the intermediate array below. Try and multiply it with D element-wise. What happens?
    intermediate_arr = np.arange(4).reshape(2, 2)
    print(intermediate_arr)
    return


@app.cell
def _():
    # What could you do to make this multiplication work? (hint: try and make the intermediate array broadcastable to E)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <details>

    <summary>Solution</summary>

    ```python
    intermediate_arr = intermediate_arr.reshape(4, 1)
    F = E * intermediate_arr
    print(F)
    ```

    </details>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Functions

    Another useful thing we might want to do with an array is to apply a function to each element. Generally speaking, it is best to use the built-in Numpy functions to work on numpy arrays instead of defining your own. This is because Numpy functions are highly optimised for speed, which is important when working with large amounts of data. Chances are, whatever kind of mathematical operation you want to perform on a Numpy array is already built in.  A full list can be found [here](https://numpy.org/doc/1.26/reference/routines.math.html)

    ### Element-wise functions
    Common uses are applying trig functions to an array of values:
    """
    )
    return


@app.cell
def _(np):
    import matplotlib.pyplot as plt

    def simple_plot(x, y, label):
        plt.plot(x, y)
        plt.suptitle(label)
        plt.show()

    # Element-wise functions:
    X = np.arange(0, 2 * np.pi, 0.2)
    Y = np.sin(X)
    simple_plot(X, Y, "sin(X)")

    Y_1 = np.exp(X)
    simple_plot(X, Y_1, "exp(X)")
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Reduce functions
    These are applied along the selected axis, consume all the elements and reduce the axis down to 1 element. If no axis is specified, the function is applied to the whole array, and reduces it to a scalar
    """
    )
    return


@app.cell
def _(np):
    A_2 = np.random.rand(32).reshape(8, 4)
    print(A_2)
    max_a2 = A_2.max()
    print(f"Maximum value in A: \n{max_a2}")
    I_2 = np.mean(A_2, axis=0, keepdims=True)
    print(f"Average of each column in A: \n{I_2}")
    I_3 = A_2.min(axis=1, keepdims=True)
    print(f"Minimum value in each row in A: \n{I_3}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Exercises

    ### Task 1
    """
    )
    return


@app.cell
def _(np):
    # Have a look at the trigonometric functions listed in the official numpy documentation: https://numpy.org/doc/stable/reference/routines.math.html
    # Pick any element-wise function that takes 1 argument and apply it to the following array. store the result in a variable called Ys
    Xs = np.arange(-2 * np.pi, 2 * np.pi, 0.1)  # noqa F841
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <details>

    <summary>Solution</summary>

    ```python
    Ys = np.arctan(Xs)
    print(Ys)
    ```

    </details>

    ### Task 2
    """
    )
    return


@app.cell
def _():
    # Call our previously-defined simple plot function and display your result! Give the plot a suitable title
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <details>

    <summary>Solution</summary>

    ```python
    simple_plot(Xs, Ys, "arctan(Ys)")
    ```

    </details>

    ### Task 3
    """
    )
    return


@app.cell
def _():
    # Use a suitable Numpy reduce function to find the maximum value of your function's result
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <details>

    <summary>Solution</summary>

    ```python
    max_val = np.max(Ys)
    print(max_val)
    ```

    </details>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ```numpy.where```

    This allows us to conditionally modify the elements of an array, which can be useful for tasks like thresholding or masking.
    """
    )
    return


@app.cell
def _(np):
    big_image = np.arange(10 * 10).reshape(10, 10)
    big_image_thresh = np.where(big_image < 50, 0, big_image)
    big_image_thresh
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Exercise""")
    return


@app.cell
def _():
    # Create a Numpy array of shape (10, 5) and fill it with random values between 0 and 1. Store it in a variable called J
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <details>

    <summary>Solution</summary>

    ```python
    J = np.random.rand(10*5).reshape(10, 5)
    print(J)
    ```

    </details>
    """
    )
    return


@app.cell
def _():
    # Conditionally modify the array such that numbers below 0.5 are replaced with 0 and numbers above or equal to 0.5 are replaced with 1. Store the result in a variable called K
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <details>

    <summary>Solution</summary>

    ```python
    K = np.where(J<0.5, 0, 1)
    print(K)
    ```

    </details>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Practical Example

    The following function will plot pairs of x and y values on a graph. We will use this to plot the results of the following exercises.
    """
    )
    return


@app.cell
def _(plt):
    def plot_points(point):
        plt.rcParams["figure.figsize"] = [3.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        x = point[0::2]  # even indices for x
        y = point[1::2]  # odd indices for y
        lims = 3
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xlim(-lims, lims)
        plt.ylim(-lims, lims)
        plt.grid()
        ax.plot(
            x,
            y,
            marker="o",
            markersize=7,
            markeredgecolor="black",
            markerfacecolor="red",
        )
        ax.spines["left"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["bottom"].set_position("zero")
        ax.spines["top"].set_color("none")
        plt.show()

    return (plot_points,)


@app.cell
def _(np, plot_points):
    points = np.array([[0, 1], [1, 2], [2, 1], [0, 1]])
    print(points.shape)
    triangle = points.flatten()
    print(triangle.shape)
    plot_points(triangle)
    return (points,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now we will define a function to generate a 2D rotation matrix. This will be used to rotate a set of points in 2D space. The rotation matrix is defined as:

    $$  \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix} $$

    Where $\theta$ is the angle of rotation in radians.
    """
    )
    return


@app.cell
def _(np):
    import math

    def get_rotation_matrix(angle):
        angle = math.radians(angle)
        return np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )

    return (get_rotation_matrix,)


@app.cell
def _(get_rotation_matrix):
    print(get_rotation_matrix(90))
    return


@app.cell
def _(get_rotation_matrix, plot_points, points):
    rot_matrix = get_rotation_matrix(45)
    _rotated_triangle = points @ rot_matrix
    print(_rotated_triangle)
    plot_points(_rotated_triangle.flatten())
    return (rot_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can add a translation to the rotation matrix to move the points to a new location. The translation matrix is defined as:

    $$  \begin{bmatrix} x \\ y \end{bmatrix} $$

    Where x and y are the amount to move the points in the x and y directions respectively.
    """
    )
    return


@app.cell
def _(np, plot_points, points, rot_matrix):
    translate = np.array([-2, 0])
    _rotated_triangle = points @ rot_matrix + translate
    plot_points(_rotated_triangle.flatten())
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
