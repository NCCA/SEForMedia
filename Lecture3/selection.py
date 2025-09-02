#!/usr/bin/env -S uv run --script

value = input("Enter some text and press enter : ")

if len(value) > 10:
    print("the length of the string is over 10")
else:
    print("the length of the string is under 10")
