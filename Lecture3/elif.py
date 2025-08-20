#!/usr/bin/env -S uv run --script

try:
    number = int(input("please enter a number between 1 and 100 : "))
except ValueError:
    print("you did not enter a number")

if number < 1 or number > 100:
    print("the number is not between 1 and 100")
elif number < 50:
    print("the number is less than 50")
else:
    print("the number is greater than 50")
