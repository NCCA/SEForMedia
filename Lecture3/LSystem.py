#!/usr/bin/env -S uv run --script
import turtle
from turtle import colormode
from typing import Type
import random


def generate_rule_string(axiom: str, rules: dict[str, str], iterations: int) -> str:
    """
    Generates a rule string based on the given axiom, rules, and number of iterations.

    Args:
        axiom (str): The initial string to start the generation process.
        rules (dict[str, str]): A dictionary where keys are characters in the axiom and values are the replacement strings.
        iterations (int): The number of iterations to apply the rules.

    Returns:
        str: The generated rule string after applying the rules for the specified number of iterations.
    """
    derived = [axiom]  # this is the first seed
    for _ in range(iterations):  # now loop for each iteration
        next_sequence = derived[-1]  # grab the last rule
        next_axiom = [
            rule(char, rules) for char in next_sequence
        ]  # for each element in the rule expand
        derived.append(
            "".join(next_axiom)
        )  # append to the list, we will only need the last element
    return derived


def rule(sequence: str, rules: dict[str, str]) -> str:
    """
    Applies the given rules to the sequence.

    Args:
        sequence (str): The initial string to which the rules will be applied.
        rules (dict[str, str]): A dictionary where keys are characters in the sequence and values are the replacement strings.

    Returns:
        str: The resulting string after applying the rules to the sequence.
    """
    if sequence in rules:
        return rules[sequence]
    return sequence


def draw_lsystem(
    turtle: Type[turtle], commands: str, length: float, angle: float
) -> None:
    """
    Draws an L-system based on the given commands.

    Args:
        turtle (Turtle): The turtle object used to draw the L-system.
        commands (str): The string of commands to control the turtle.
        length (float): The length of each step the turtle takes.
        angle (float): The angle by which the turtle turns.

    Returns:
        None
    """
    stack = []
    for command in commands:
        turtle.pendown()
        if command in [
            "F",
            "G",
            "R",
            "L",
            "A",
        ]:  # forward rules for some l system grammars
            turtle.forward(length)
        elif command in ["f", "B"]:
            turtle.penup()
            turtle.forward(length)
        elif command == "+":
            turtle.right(angle)
        elif command == "-":
            turtle.left(angle)
        elif command == "[":
            stack.append((turtle.position(), turtle.heading()))  # save turtle values
        elif command == "]":
            turtle.penup()  # were moving back to save pos
            position, heading = stack.pop()
            turtle.goto(position)
            turtle.setheading(heading)


# F -> Forward
# X -> A place holder for movements
# [ push position and direction onto stack
# ] pop position and direction back to turtle
# + Turn Left
# - Turn Right

axiom = "X"  # start
rules = {"X": "F+[[X]-X]-F[-FX]+X", "F": "FF"}  # fern
iterations = 4  # lower is quicker
length = 10  # lower this if more iterations
angle = 25  # change this to make different shapes

colormode(255)
g = generate_rule_string(axiom, rules, iterations)
print(g[-1])

turtle = turtle.Turtle()
turtle.speed(100)
turtle.penup()
turtle.goto(0, -200)
turtle.left(90)

turtle.color((0, 128, 0))
draw_lsystem(turtle, g[-1], 10, 25)
