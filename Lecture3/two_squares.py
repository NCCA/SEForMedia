#!/usr/bin/env -S uv run --script

import turtle

# create an instance of the turtle object and call it turtle
turtle = turtle.Turtle()
# now we set the shape (default is an arrow)
turtle.shape("turtle")
# now we will run a sequence moving forward then turn left by 90 degrees
turtle.penup()
turtle.goto(-100, 0)

turtle.pendown()
turtle.forward(100)
turtle.left(90)
turtle.forward(100)
turtle.left(90)
turtle.forward(100)
turtle.left(90)
turtle.forward(100)

turtle.penup()
turtle.goto(100, 0)

turtle.pendown()
turtle.forward(100)
turtle.left(90)
turtle.forward(100)
turtle.left(90)
turtle.forward(100)
turtle.left(90)
turtle.forward(100)
