#!/usr/bin/env -S uv run --script

a = 10
b = 0
try:
    print("Doing Division {}".format(a / b))
except ZeroDivisionError:
    print("Cant divide by zero")

print("now do something else")
