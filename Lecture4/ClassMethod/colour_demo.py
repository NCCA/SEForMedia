#!/usr/bin/env -S uv run --script

from Colour import Colour

c = Colour()
print(c)

f = Colour.fromFloat(0.1, 2.0, 1.0, 2.0)
print(f)

f = Colour.fromInt(255, 127, 12)
print(f)
try:
    f = Colour.fromFloat("1", "hello", 2, 3)
    print(f)
except AttributeError:
    print("error")
