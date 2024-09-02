#!/usr/bin/env python

int_text = "12"
float_text = "0.23123"
int_data = 123

a = int(int_text)
b = float(float_text)
text = str(int_data)

print(f"{a} type {str(type(a))}")
print(f"{b} type {str(type(b))}")
print(f"{text} type {str(type(text))}")
