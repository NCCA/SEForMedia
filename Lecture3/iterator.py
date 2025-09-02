#!/usr/bin/env -S uv run --script

fruit = ("apple", "banana", "cherry")
fruit_it = iter(fruit)

print(f"{next(fruit_it)=}")
print(f"{next(fruit_it)=}")
print(f"{next(fruit_it)=}")

try:
    print(next(fruit_it))
except StopIteration:
    print("none left")
