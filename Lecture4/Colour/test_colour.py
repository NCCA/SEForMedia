#!/usr/bin/env -S uv run --script

from Colour import Colour


red = Colour()
red.r = 1.0
print(red)
print(repr(red))
print(type(red))

c1 = Colour(0.2, 1.0, 2.0, 1)
print(c1)
newColour = c1.mix(red, 0.2)
mixed = c1.mix(red, 0.5)
print("mixing {} {} {} {}".format(mixed.r, mixed.g, mixed.b, mixed.a))
# Problem
c1 = Colour("red", "green", "blue", "alpha")
print(c1)
newColour = c1.mix(red, 0.2)
