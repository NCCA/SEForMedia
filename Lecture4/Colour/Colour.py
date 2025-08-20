#!/usr/bin/env -S uv run --script


class Colour:
    "a very simple colour container"

    def __init__(self, r=0.0, g=0.0, b=0.0, a=1.0):
        "constructor to set default values"
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    def __str__(self):
        "method to print out the colour data for debug"
        return "[{},{},{},{}]".format(self.r, self.g, self.b, self.a)

    def __repr__(self):
        "method to print out the colour data for debug"
        return "Colour[r={},g={},b={},a={}]".format(self.r, self.g, self.b, self.a)

    def mix(self, colour, t):
        """method to mix current colour with another by t
        will catch the attribute error and pass back black if
        wrong values are passed
        """
        c = Colour()
        try:
            c.r = self.r + (colour.r - self.r) * t
            c.g = self.g + (colour.g - self.g) * t
            c.b = self.b + (colour.b - self.b) * t
            c.a = self.a + (colour.a - self.a) * t
        except AttributeError:
            pass
        return c
