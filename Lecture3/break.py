#!/usr/bin/env python
import random


numbers = [random.uniform(-1, 10) for i in range(0, 10)]
print(numbers)
for n in numbers:
    print(f"{n=}")
    if n < 0.0:
        print("found a negative exiting {}".format(n))
        break
