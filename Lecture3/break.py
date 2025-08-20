#!/usr/bin/env -S uv run --script
import random

# note this is called a list comprehension
# it is a way to create a list in one line of code
numbers = [random.uniform(-1, 10) for i in range(0, 10)]
print(numbers)
for n in numbers:
    print(f"{n=}")
    if n < 0.0:
        print(f"found a negative exiting {n}")
        break
