import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

matplotlib.use("Qt5Agg")


class GraphWidget(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_title("Prediction")
        super().__init__(self.fig)

    def plot_data(self, data):
        x = list(range(0, 10))
        self.axes.cla()
        self.axes.stem(x, data)
        self.axes.set_title("Prediction")
        self.axes.set_xticks(x)
        self.draw()
