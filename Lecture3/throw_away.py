#!/usr/bin/env -S uv run --script

sum = 2
for _ in range(0, 10):
    sum += sum
print(f"{sum=}")
