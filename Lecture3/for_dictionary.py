#!/usr/bin/env python

colours = {"red": [1, 0, 0], "green": [0, 1, 0], "blue": [0, 0, 1]}

print(f"{colours.items()=}")

# note for can unpack the key and value from the dictionary
for colour, value in colours.items():
    print(f"{colour=} := {value=}")

for colour in colours.keys():
    print(f"{colour=}")

for values in colours.values():
    print(f"{values=}")
