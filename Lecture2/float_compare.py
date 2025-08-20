#!/usr/bin/env -S uv run --script
print(0.2 + 0.2 == 0.4)
print(0.1 + 0.2 == 0.3)

print(f"{0.2 + 0.2=}")
print(f"{0.1 + 0.2=}")

print(f"{(0.1 + 0.2).hex()=}")
print(f"{(0.3).hex()=}")
