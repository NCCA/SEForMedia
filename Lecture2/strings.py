#!/usr/bin/env -S uv run --script
hello = "hello world"
print(hello)

text = 'We can use single "quotes"'

print(text)

text = """Triple single or double quotes
can be used for longer strings. These will
Be
printed
verbatim"""

print(text)

print(
    "Unicode strings start with u but not all terminals can print these chars \U0001d6d1 "
)
print("Omega:  Ω")
