#!/usr/bin/env -S uv run --script

import turtle
from turtle import colormode
import random


def colour_function(increment: int = 25) -> tuple:
    """
    Generates a color based on the given increment.

    Args:
    increment (int, optional): The increment value used to generate the color. Defaults to 25.

    Returns:
    tuple: A tuple representing an RGB color, where each value is an integer between 0 and 255.
    """
    red = 0
    green = 0
    blue = 0
    while True:
        colour = (red, green, blue)
        red = red + increment
        if red >= 255:
            red = 0
            green = green + increment
        if green >= 255:
            green = 0
            blue = blue + increment
        if blue >= 255:
            blue = 0
            red = 0
            green = 0
        yield colour


colour = colour_function()
turtle = turtle.Turtle()
turtle.speed(0)
colormode(255)
for i in range(0, 100000):
    current_colour = next(colour)
    print(f"{current_colour=}")
    turtle.color(current_colour)
    turtle.goto(random.uniform(-100, 100), random.uniform(-100, 100))
