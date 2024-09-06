#!/usr/bin/env python
import turtle

turtle = turtle.Turtle()
turtle.speed(0)


def spiral(n):
    if n < 300:
        turtle.forward(n)
        turtle.right(91)
        spiral(n + 2)


spiral(2)
