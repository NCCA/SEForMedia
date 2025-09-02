#!/usr/bin/env -S uv run --script


n = ((a, b) for a in range(0, 5) for b in range(0, 5))
for i in n:
    print(i)
