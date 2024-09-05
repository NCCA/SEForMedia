#!/usr/bin/env python
import turtle
from typing import Type


def square(turtle: Type[turtle], x: float, y: float, width: float, height: float) -> None:
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


turtle = turtle.Turtle()

square(turtle, 20, 20, 100, 100)
