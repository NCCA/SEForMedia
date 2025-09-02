#!/usr/bin/env python3

from PIL import Image as PILImage
import numpy as np
from matplotlib import pyplot as plt


## copy the previous class and add the method to load image
class Image:
    def __init__(self, width, height, colour=(255, 255, 255)):
        self.width = width
        self.height = height
        self.clear(colour)

    def clear(self, colour):
        self.pixels = np.full((self.height, self.width, 3), colour, dtype=np.uint8)

    def set_pixel(self, x, y, color):
        self.pixels[y, x] = color

    def get_pixel(self, x, y):
        return self.pixels[y, x]

    def save(self, filename):
        img = PILImage.fromarray(self.pixels)
        img.save(filename)

    def display(self):
        plt.axis("off")
        img = PILImage.fromarray(self.pixels)
        plt.imshow(img)
        plt.show(img)

    def load(self, filename):
        img = PILImage.open(filename)
        self.pixels = np.array(img)
        self.width = self.pixels.shape[1]
        self.height = self.pixels.shape[0]

    @classmethod
    def from_file(cls, filename):
        img = PILImage.open(filename)
        pixels = np.array(img)
        height = pixels.shape[0]
        width = pixels.shape[1]
        instance = cls(width, height)
        instance.pixels = pixels
        return instance

    def draw_line(self, x0, y0, x1, y1, color):
        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                self.set_pixel(x, y, color)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                self.set_pixel(x, y, color)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy


if __name__ == "__main__":
    img = Image(100, 100)
    img.draw_line(10, 10, 90, 90, (255, 0, 0))
    img.draw_line(10, 90, 90, 10, (0, 255, 0))
    img.save("output.png")

    img.clear((255, 255, 255))
    img.draw_line(0, 50, 100, 50, (0, 0, 255))
    img.draw_line(50, 0, 50, 100, (0, 0, 255))
    img.save("output2.png")

    ## random lines

    img.clear((255, 255, 255))
    for i in range(2000):
        x0 = np.random.randint(0, 100)
        y0 = np.random.randint(0, 100)
        x1 = np.random.randint(0, 100)
        y1 = np.random.randint(0, 100)
        color = np.random.randint(0, 255, 3)
        img.draw_line(x0, y0, x1, y1, color)
    img.save("output3.png")
