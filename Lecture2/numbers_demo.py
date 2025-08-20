#!/usr/bin/env -S uv run --script
integer_variable = 1
float_variable = 1.2
complex_variable = 4 + 5j

print(f"{integer_variable} is of type {type(integer_variable)}")
print(f"{float_variable} is of type {type(float_variable)}")
print(f"{complex_variable} is of type {type(complex_variable)}")
