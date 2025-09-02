#!/usr/bin/env -S uv run --script
list_1 = [123, "hello", 2.45, 3 + 2j]
list_2 = [" ", "world"]

print(f"{list_1=}")
print(f"{list_1[1]=}")
print(f"{list_1[2:]=}")

hello = list_1[1] + list_2[0] + list_2[1]
print(f"{hello=}")

# this is very much python2 but still common in python3
for i in range(0, len(list_1)):
    print(f"{list_1[i]=}")
# lists are iterable so we can do this which is preferred in python3
for ch in list_1:
    print(f"{ch=}")

print("Using enumerate which gives us the index and value")
for num, ch in enumerate(list_1):
    print(f"{num} {ch}")
