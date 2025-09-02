from Point3 import Point3
from Colour import Colour


class Sphere:
    # ctor to assign values
    def __init__(self, pos=Point3(), colour=Colour(), radius=1, name=""):
        self._pos = pos
        self._colour = colour
        self._radius = radius
        self._name = name

    def debug(self):
        print(
            f"{self._name} pos:{self._pos} colour:{self._colour} radius:{self._radius}"
        )
