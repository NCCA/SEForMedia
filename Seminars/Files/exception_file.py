#!/usr/bin/env python


try:
    with open("nothere", "r") as file:
        contents = file.read()
        print(contents)
except FileNotFoundError:
    print("File not found")

# throw a permission denied exception
try:
    with open("/etc/passwd", "w") as file:
        ...
except PermissionError:
    print("Permission denied")
