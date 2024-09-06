#!/usr/bin/env python

y = -10.0

while y <= 2.0:
    x = -2.0
    while x <= 2.0:
        print(f"{x=},{y=}")
        x += 0.5
    y += 0.5
