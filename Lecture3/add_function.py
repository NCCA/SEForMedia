#!/usr/bin/env -S uv run --script


def add(a: int, b: int) -> int:
    return a + b


a = 1
b = 2
c = add(a, b)
print(f"{c=}")
str1 = "hello"
str2 = " python"
result = add(str1, str2)
print(f"{result=}")

print(f" {add(1.0, 2)=}")
