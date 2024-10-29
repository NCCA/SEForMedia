#!/usr/bin/env python3

import sys

import torch
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
        # controls are in added at 0,0 and spans 1 row and 1 columns
        self.main_layout.addWidget(self.sketch_widget, 0, 1, 1, 1)
        self.plot = GraphWidget(self, width=4, height=4, dpi=100)

        self.main_layout.addWidget(self.plot, 0, 2, 1, 1)

        self.clear_button.clicked.connect(self.clear_sketch)
        self.device = self.get_device()
        self.load_model()
        self.count = 0
        self.statusBar().showMessage("Ready")

        self.sketch_widget.mouse_release.connect(self.predict)
        self.pen_size.valueChanged.connect(
            lambda value: setattr(self.sketch_widget, "pen_width", value)
        )

    # def build_model(self):
    #     n_classes = 10
    #     input_size = 28 * 28
    #     layers = [
    #         nn.Flatten(),
    #         nn.Linear(input_size, 512),  # Input
    #         nn.ReLU(),  # Activation for input
    #         nn.Linear(512, 512),  # Hidden
    #         nn.ReLU(),  # Activation for hidden
    #         nn.Linear(512, n_classes),  # Output
    #     ]
    #     self.model = nn.Sequential(*layers)
    #     self.model.load_state_dict(torch.load("mnist_model.pth"))

    #     self.model.to(self.device)
    #     self.model.eval()

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

    def load_model(self):
        self.model = torch.load("minst_model_full.pth")
        self.model.to(self.device)
        self.model.eval()

    def predict(self):
        if self.proc_type.currentIndex() == 0:
            image_tensor, q_image = self.sketch_widget.get_image_tensor()
        else:
            image_tensor, q_image = self.sketch_widget.center_image_by_mass()

        image_tensor.to(self.device)

        prediction = self.model(image_tensor.to(self.device))
        self.numbers.display(prediction.argmax().item())
        # print the prediction on the info bar

        print(f"prediction {prediction.argmax().item()}")
        pred = str(prediction)
        pred = pred[0 : pred.find("]]")]
        pred = pred.replace("\t", "")
        self.statusBar().showMessage(pred)
        pred_data = list(prediction[0].detach().cpu().numpy())
        self.plot.plot_data(pred_data)
        self.image_label.setPixmap(q_image)

    def clear_sketch(self):
        self.sketch_widget.image.fill(Qt.black)
        self.sketch_widget.modified = True
        self.sketch_widget.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Space:
            self.clear_sketch()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
