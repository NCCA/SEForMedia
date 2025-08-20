#!/usr/bin/env -S uv run --script
import ipyturtle3 as turtle

from turtle import colormode
from typing import Type


def square(
    turtle: Type[turtle],
    x: float,
    y: float,
    width: float,
    height: float,
    colour: tuple[int, int, int] = (0, 0, 0),
    fill_colour: tuple[int, int, int] = (0, 0, 0),
    fill: bool = True,
) -> None:
    """
    Draws a square using the given turtle object.

    Args:
        turtle (Turtle): The turtle object used to draw the square.
        x (float): The x-coordinate of the starting position.
        y (float): The y-coordinate of the starting position.
        width (float): The width of the square.
        height (float): The height of the square.
        colour (Tuple[int, int, int], optional): The color of the square's outline. Defaults to (0, 0, 0).
        fill_colour (Tuple[int, int, int], optional): The fill color of the square. Defaults to (0, 0, 0).
        fill (bool, optional): Whether to fill the square with the fill color. Defaults to True.

    Returns:
        None
    """
    colormode(255)
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()
    turtle.pencolor(colour)
    turtle.fillcolor(fill_colour)
    if fill:
        turtle.begin_fill()
    turtle.forward(width)
    turtle.left(90)
    turtle.forward(height)
    turtle.left(90)
    turtle.forward(width)
    turtle.left(90)
    turtle.forward(height)
    if fill:
        turtle.end_fill()


if __name__ == "__main__":
    print("this is to be run in a jupyter notebook")
