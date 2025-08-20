#!/usr/bin/env -S uv run --script


class Vec3:
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        if isinstance(x, (int, float)):
            self._x = x
        else:
            raise ValueError("need float or int")

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        if isinstance(y, (int, float)):
            self._y = y
        else:
            raise ValueError("need float or int")

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, z):
        if isinstance(z, (int, float)):
            self._z = z
        else:
            raise ValueError("need float or int")

    def __str__(self):
        return f"Vec3({self._x}, {self._y}, {self._z})"

    def __repr__(self):
        return f"Vec3({self._x}, {self._y}, {self._z})"

    def __add__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
        raise TypeError("Operand must be of type Vec3")

    def __sub__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
        raise TypeError("Operand must be of type Vec3")

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
        raise TypeError("Operand must be a number")

    def __rmul__(self, scalar):
        return self.__mul__(scalar)  # Just reuse __mul__

    def __eq__(self, other: "Vec3") -> bool:
        if isinstance(other, Vec3):
            return self.x == other.x and self.y == other.y and self.z == other.z
        return False

    def __ne__(self, other: "Vec3") -> bool:
        return not self.__eq__(other)


if __name__ == "__main__":
    # lets test the class
    v1 = Vec3(1.0, 2.0, 3.0)
    v2 = Vec3(4.0, 5.0, 6.0)
    print(v1)
    print(v2)
    v3 = v1 + v2
    print(v3)
    v1.x = 10
    print(v1)

    v3 = v1 - v2
    print(v3)
    v3 = v1 * 2
    print(v3)
    v3 = 2 * v1
    print(v3)
    print(v1 == v2)
    print(v1 != v2)
    print(v1 == 2)
    try:
        v1.x = "hello"
    except ValueError as e:
        print(e)
