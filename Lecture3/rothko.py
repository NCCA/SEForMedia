#!/usr/bin/env python

import random
import turtle

from Square import square


def random_colour() -> tuple[int, int, int]:
    """
    Generates a random color.

    Returns:
        tuple[int, int, int]: A tuple representing an RGB color,
                              where each value is an integer between 0 and 255.
    """
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


my_turtle = turtle.Turtle()
my_turtle.speed(0)

for i in range(0, 200):
    x = random.uniform(-200, 200)
    y = random.uniform(-200, 200)
    width = random.uniform(50, 300)
    height = random.uniform(50, 300)

    square(my_turtle, x, y, width, height, random_colour(), random_colour())
