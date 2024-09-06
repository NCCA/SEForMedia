#!/usr/bin/env python

import numpy as np

cords = ((x, y) for x in np.arange(-1.0, 1.0, 0.5) for y in np.arange(-1.0, 1.0, 0.5))
for x, y in cords:
    print(x, y)
