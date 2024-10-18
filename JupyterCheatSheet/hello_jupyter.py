import numpy as np
import matplotlib.pyplot as plt

print("This is an external python file")
print("I am going to run some code from here")

# create a sine wave and plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.show()
