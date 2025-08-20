#!/usr/bin/env -S uv run --script

a = range(0, 10)
even = []
for i in a:
    if i % 2:  # is it even?
        continue
    even.append(i)
print(f"{even=}")
