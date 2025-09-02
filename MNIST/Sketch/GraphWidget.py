"""A Simple Graph Widget for plotting the prediction data into a Qt application."""

from typing import List

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

matplotlib.use("Qt5Agg")


class GraphWidget(FigureCanvasQTAgg):
    def __init__(
        self, parent=None, width: int = 5, height: int = 4, dpi: int = 100
    ) -> None:
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_title("Prediction")
        super().__init__(self.fig)

    def plot_data(self, data: List) -> None:
        """
        Plot the data into the graph widget
        """
        x = list(range(0, 10))
        self.axes.cla()
        self.axes.stem(x, data)
        self.axes.set_title("Prediction")
        self.axes.set_xticks(x)
        self.draw()
