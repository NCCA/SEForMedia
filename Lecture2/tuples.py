#!/usr/bin/env -S uv run --script
tuple_1 = (123, "hello", 2.45, 3 + 2j)
tuple_2 = (" ", "world")

print(f"{tuple_1=}")
print(f"{tuple_1[1]=}")
print(f"{tuple_1[2:]=}")

hello = tuple_1[1] + tuple_2[0] + tuple_2[1]
print(f"{hello=}")
tuple_1[0] = 3
