class Colour:
    # ctor to assign values
    def __init__(self, r=0.0, g=0.0, b=0.0, a=1.0):
        self._r = r
        self._g = g
        self._b = b
        self._a = a

    # debug print function to print vector values
    def __str__(self):
        return f"[{self._r},{self._g},{self._b},{self._a}]"
