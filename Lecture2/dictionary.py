#!/usr/bin/env -S uv run --script
colours = {
    "red": [1, 0, 0],
    "green": [0, 1, 0],
    "blue": [0, 0, 1],
    "white": [1, 1, 1],
    "black": [0, 0, 0],
}

print(f"{colours.get('red')=}")
print(f"{colours.get('green')=}")
print(f"{colours.get('purple')=}")
# can also use
print(f"{colours['white']=}")
# but
print(f"{colours['purple']=}")
