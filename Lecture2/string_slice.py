#!/usr/bin/env -S uv run --script
str = "Hello python"

# Prints complete string
print(f"{str=}")
# Prints first character of the string
print(f"{str[0]=}")
# Prints characters starting from 3rd to 6th
print(f"{str[2:5]=}")
# Prints string starting from 3rd character
print(f"{str[2:]=}")
# Prints string two times
print(f"{str * 2=}")
# Prints concatenated string
print(str + " with added text")
