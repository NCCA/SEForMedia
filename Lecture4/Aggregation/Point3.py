class Point3:
    # ctor to assign values
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self._x = x
        self._y = y
        self._z = z

    # debug print function to print vector values
    def __str__(self):
        return f"[{self._x},{self._y},{self._z}]"
