#!/usr/bin/env -S uv run --script


class Colour:
    "a very simple colour container"

    def __init__(self):
        "constructor to set default values"
        self.r = 0
        self.g = 0
        self.b = 0
        self.a = 0

    def __str__(self):
        "method to print out the colour data for debug"
        return "[{},{},{},{}]".format(self.r, self.g, self.b, self.a)

    def __repr__(self):
        "method to print out the colour data for debug"
        return "Colour[r={},g={},b={},a={}]".format(self.r, self.g, self.b, self.a)

    @classmethod
    def fromFloat(cls, r, g, b, a=1.0):
        c = Colour()
        try:
            c.r = float(r)
            c.g = float(g)
            c.b = float(b)
            c.a = float(a)
        except ValueError:
            raise ValueError
        return c

    @classmethod
    def fromInt(cls, r, g, b, a=255):
        c = Colour()
        try:
            c.r = int(r)
            c.g = int(g)
            c.b = int(b)
            c.a = int(a)
        except ValueError:
            pass
        return c

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
