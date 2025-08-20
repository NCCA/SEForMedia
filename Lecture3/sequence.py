#!/usr/bin/env -S uv run --script

i = 1
print(f"Sequence {i=}")
i = i + 1
print(f"Sequence {i=}")
i += 1  # note this is the same as i=i+1
print(f"Sequence {i=}")
i += 1
print(f"Sequence {i=}")
