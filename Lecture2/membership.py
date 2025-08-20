#!/usr/bin/env -S uv run --script
data = (123, "hello", 2.45, 3 + 2j)
numbers = [1, 2, 3, 4, 5]

print(f"{data=}")
print(f"{numbers=}")

print(f"{'world' in data=}")
print(f"{'text' not in numbers=}")
print(f"{99 in numbers=}")
print(f"{2 in numbers=}")


image_types = ["png", "tiff", "tif", "jpg", "gif"]
# note use of lower to ensure we check against correct values
if "TIFF".lower() or "jpg".lower() in image_types:
    print("Have image")
