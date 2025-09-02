#!/usr/bin/env -S uv run --script

# create a list of 10 elements
a = list(range(10))

print(f"{a[::1]=}")
print(f"{a[::-1]=}")
print(f"{a[1:10:2]=}")
print(f"{a[:-1:1]=}")
del a[::2]
print(f"{a=}")
print(f"{list(range(10))[slice(0, 5, 2)]=}")
