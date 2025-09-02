#!/usr/bin/env -S uv run --script

a = True
b = False

print(f"{a=}, {b=}")
print(f"{a and b=}")
print(f"{a or b=}")
print(f"{not(a and b)=}")
