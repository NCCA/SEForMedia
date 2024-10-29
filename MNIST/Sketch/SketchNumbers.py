#!/usr/bin/env python3

import sys

import torch
import torch.nn as nn
from GraphWidget import GraphWidget
from MainWindow import Ui_MainWindow
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QMainWindow
from SketchWidget import SketchWidget


# Define a simple MainWindow class
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.sketch_widget = SketchWidget(self)
        self.main_layout.addWidget(self.sketch_widget, 0, 1, 1, 1)
        self.plot = GraphWidget(self, width=4, height=4, dpi=100)

        self.main_layout.addWidget(self.plot, 0, 2, 1, 1)

        self.clear_button.clicked.connect(self.clear_sketch)
        self.device = self.get_device()
        self.build_model()
        self.count = 0
        self.statusBar().showMessage("Ready")

        self.sketch_widget.mouse_release.connect(self.predict)
        self.pen_size.valueChanged.connect(
            lambda value: setattr(self.sketch_widget, "pen_width", value)
        )
        self.border_size.valueChanged.connect(
            lambda value: setattr(self.sketch_widget, "border", value)
        )

    def build_model(self) -> None:
        """
        build the pytorch model to predict the numbers
        """
        n_classes = 10
        input_size = 28 * 28
        layers = [
            nn.Flatten(),
            nn.Linear(input_size, 512),  # Input
            nn.ReLU(),  # Activation for input
            nn.Linear(512, 512),  # Hidden
            nn.ReLU(),  # Activation for hidden
            nn.Linear(512, n_classes),  # Output
        ]
        self.model = nn.Sequential(*layers)
        # load the model weights, we need to say what device we are using
        # and that we are only loading the weights
        self.model.load_state_dict(
            torch.load("mnist_model.pth", map_location=self.device, weights_only=True)
        )
        # Now move to our device
        self.model.to(self.device)
        self.model.eval()

    def get_device(self) -> torch.device:
        """
        Returns the appropriate device for the current environment.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():  # mac metal backend
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def predict(self) -> None:
        """
        Predict the number from the sketch
        """

        # first get the image an process it
        if self.proc_type.currentIndex() == 0:
            image_tensor, q_image = self.sketch_widget.get_image_tensor()
        else:
            image_tensor, q_image = self.sketch_widget.center_image_by_mass()

        # send to our model
        prediction = self.model(image_tensor.to(self.device))
        # set the number display to the prediction
        self.numbers.display(prediction.argmax().item())
        # now plot a graph to show the prediction tensor.
        pred_data = list(prediction[0].detach().cpu().numpy())
        self.plot.plot_data(pred_data)
        self.image_label.setPixmap(q_image)

    def clear_sketch(self) -> None:
        """
        Clear the sketch widget
        """
        self.sketch_widget.image.fill(Qt.black)
        self.sketch_widget.modified = True
        self.sketch_widget.update()

    def keyPressEvent(self, event) -> None:
        """
        Handle key press events
        """
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Space:
            self.clear_sketch()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
