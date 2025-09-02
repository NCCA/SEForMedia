#!/usr/bin/env -S uv run --script
int_text = "12"
float_text = "0.23123"
int_data = 123

a = int(int_text)
b = float(float_text)
text = str(int_data)


print(f"{int_text=} type {type(int_text)}")
print(f"{float_text=} type {type(float_text)}")
print(f"{int_data=} type {type(int_data)}")

# we get a ValueError here we will see how to handle this
# in a future lecture
err = float("12.3.4")
