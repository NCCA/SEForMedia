#!/usr/bin/env python3
# Import the necessary Qt modules through QtPy
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from MainWindow import Ui_MainWindow
from qtpy.QtCore import QPoint, QRect, QSize, Qt, Signal
from qtpy.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from qtpy.QtWidgets import QApplication, QFileDialog, QLabel, QMainWindow, QWidget


class SketchWidget(QWidget):
    mouse_release = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StaticContents)
        self.modified = False
        self.scribbling = False
        self.myPenWidth = 24
        self.myPenColor = Qt.white
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)
        self.lastPoint = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.scribbling = True

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.scribbling:
            self.drawLineTo(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.scribbling:
            self.drawLineTo(event.pos())
            self.scribbling = False
            self.mouse_release.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        dirtyRect = event.rect()
        painter.drawImage(dirtyRect, self.image, dirtyRect)

    def resizeEvent(self, event):
        if self.width() > self.image.width() or self.height() > self.image.height():
            newWidth = max(self.width() + 128, self.image.width())
            newHeight = max(self.height() + 128, self.image.height())
            self.resizeImage(self.image, QSize(newWidth, newHeight))
        super().resizeEvent(event)

    def drawLineTo(self, endPoint):
        painter = QPainter(self.image)
        # add some variance to the pen color

        painter.setPen(
            QPen(self.myPenColor, self.myPenWidth, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        )
        painter.drawLine(self.lastPoint, endPoint)
        self.modified = True

        rad = self.myPenWidth // 2 + 2
        self.update(QRect(self.lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad))
        self.lastPoint = QPoint(endPoint)

    def resizeImage(self, image, newSize):
        if image.size() == newSize:
            return

        newImage = QImage(newSize, QImage.Format_RGB32)
        newImage.fill(Qt.white)
        painter = QPainter(newImage)
        painter.drawImage(QPoint(0, 0), image)
        self.image = newImage

    def _crop_image(self, image):
        # find the first and last non black black image row
        first_black_row = -1
        last_black_row = -1
        for row in range(image.height()):
            for col in range(image.width()):
                color = image.pixelColor(col, row)
                if color != Qt.black:
                    first_black_row = row
                    break
            if first_black_row != -1:
                break
        # now find the last non black row
        for row in range(image.height() - 1, -1, -1):
            for col in range(image.width()):
                color = image.pixelColor(col, row)
                if color != Qt.black:
                    last_black_row = row
                    break
            if last_black_row != -1:
                break
        # now find the first and last black column
        first_black_col = -1
        last_black_col = -1
        for col in range(image.width()):
            for row in range(image.height()):
                color = image.pixelColor(col, row)
                if color != Qt.black:
                    first_black_col = col
                    break
            if first_black_col != -1:
                break
        # now find the last black column
        for col in range(image.width() - 1, -1, -1):
            for row in range(image.height()):
                color = image.pixelColor(col, row)
                if color != Qt.black:
                    last_black_col = col
                    break
            if last_black_col != -1:
                break
        # now crop the image to the bounding box
        border = 5
        bounding_box = QRect(
            first_black_col - border,
            first_black_row - border,
            (last_black_col - first_black_col) + border,
            (last_black_row - first_black_row) + border,
        )
        print(bounding_box)
        image = image.copy(bounding_box)
        return image

    def get_image_tensor(self):
        # Step 1: Capture the current image
        image = self.image
        image = self._crop_image(image)
        resize_image = image.mirrored(horizontal=False, vertical=False)
        # Step 2: Resize the image to 28x28 pixels
        resized_image = resize_image.scaled(28, 28, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        # Step 3: Convert the resized image to a grayscale numpy array
        grayscale_image = resized_image.convertToFormat(QImage.Format_Grayscale8)
        buffer = grayscale_image.bits()
        buffer.setsize(grayscale_image.byteCount())
        np_image = np.array(buffer).reshape((28, 28))

        # Step 4: Normalize the numpy array and convert to PyTorch tensor
        np_image = np_image.astype(np.float32)
        tensor_image = torch.tensor(np_image).unsqueeze(0)  # Add batch and channel dimensions

        return tensor_image

    def get_image_pixmap(self):
        # Step 1: Capture the current image
        image = self.image
        image = self._crop_image(image)
        resize_image = image.mirrored(horizontal=False, vertical=False)
        # Step 2: Resize the image to 28x28 pixels
        resized_image = resize_image.scaled(28, 28, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        # Step 3: Convert the resized image to a grayscale numpy array
        grayscale_image = resized_image.convertToFormat(QImage.Format_Grayscale8)
        return QPixmap.fromImage(grayscale_image)


# Define a simple MainWindow class
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.sketch_widget = SketchWidget()
        self.horizontalLayout.addWidget(self.sketch_widget, 0)
        self.clear_button.clicked.connect(self.clear_sketch)
        self.device = self.get_device()
        # self.load_model()
        self.build_model()
        self.count = 0

        self.sketch_widget.mouse_release.connect(self.predict)

    def build_model(self):
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
        self.model.load_state_dict(torch.load("mnist_model.pth"))

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

    def load_model(self):
        self.model = torch.load("minst_model_full.pth")
        self.model.to(self.device)
        self.model.eval()

    def predict(self):
        image_tensor = self.sketch_widget.get_image_tensor().to(self.device)
        # print(image_tensor)
        print(image_tensor.shape)

        # torch.save(image_tensor, f"image_{self.count}.pth")
        # self.count+=1
        prediction = self.model(image_tensor.to(self.device))
        self.numbers.display(prediction.argmax().item())
        print(f"prediction {prediction.argmax().item()}")
        print(prediction)
        self.image_label.setPixmap(self.sketch_widget.get_image_pixmap())

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
