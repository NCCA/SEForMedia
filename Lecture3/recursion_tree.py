#!/usr/bin/env -S uv run --script
import turtle


def tree(turtle, length):
    if length > 5:
        turtle.forward(length)
        turtle.right(20)
        tree(turtle, length - 15)
        turtle.left(40)
        tree(turtle, length - 15)
        turtle.right(20)
        turtle.backward(length)


turtle = turtle.Turtle()
turtle.penup()
turtle.goto(0, -150)
turtle.left(90)
turtle.pendown()
tree(turtle, 75)
