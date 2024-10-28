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
from scipy.ndimage import center_of_mass
from torchvision.transforms import ToPILImage, ToTensor, functional as F


class SketchWidget(QWidget):
    mouse_release = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StaticContents)
        self.modified = False
        self.scribbling = False
        self.pen_width = 10
        self.pen_colour = Qt.white
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)
        self.last_point = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.pos()
            self.scribbling = True

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.scribbling:
            self._draw_to(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.scribbling:
            self._draw_to(event.pos())
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
            self._resize_image(self.image, QSize(newWidth, newHeight))
        super().resizeEvent(event)

    def _draw_to(self, endPoint):
        painter = QPainter(self.image)

        painter.setPen(
            QPen(self.pen_colour, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        )
        painter.drawLine(self.last_point, endPoint)
        self.modified = True

        rad = self.pen_width // 2 + 2
        self.update(QRect(self.last_point, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad))
        self.last_point = QPoint(endPoint)

    def _resize_image(self, image, newSize):
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
        # now find the first and last non black column
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

    def center_image_by_mass(self):
        # Convert the tensor to a numpy array
        image = self.image
        # convert to a numpy array
        grayscale_image = image.convertToFormat(QImage.Format_Grayscale8)
        buffer = grayscale_image.bits()
        buffer.setsize(grayscale_image.byteCount())
        image_array = np.array(buffer).reshape((image.height(), image.width()))
        # Calculate the center of mass
        com_y, com_x = center_of_mass(image_array)  # Assuming a (1, H, W) shape for grayscale

        # Calculate the center of the image
        center_y, center_x = np.array(image_array.shape) / 2  # Center of the image (H/2, W/2)

        # Calculate the shifts needed to align the center of mass to the center
        shift_y, shift_x = center_y - com_y, center_x - com_x

        # Convert the tensor to PIL image for transformation
        image_pil = ToPILImage()(image_array)  # Convert to PIL format

        # Apply the affine transformation with calculated shifts
        centered_image_pil = F.affine(
            image_pil, angle=0, translate=(int(shift_x), int(shift_y)), scale=1, shear=0
        )
        # now resize the image to 28x28
        centered_image_pil = centered_image_pil.resize((28, 28))
        print(f"size is {centered_image_pil.size=}")
        # Convert the transformed image back to a tensor
        centered_image_tensor = ToTensor()(centered_image_pil)

        print(f"size is {centered_image_tensor.shape=}")
        return centered_image_tensor

    def get_image_tensor(self):
        # Step 1: Capture the current image
        image = self.image
        resize_image = self._crop_image(image)
        # resize_image = image.mirrored(horizontal=False, vertical=False)
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

        # np_image = np.ascontiguousarray(np_image)
        # # Calculate the stride
        # stride = np_image.strides[0]
        # # Create QImage from the numpy array
        # q_image = QImage(np_image.data, 28, 28, stride, QImage.Format_Grayscale8)
        return tensor_image, QPixmap.fromImage(grayscale_image)


# Define a simple MainWindow class
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.sketch_widget = SketchWidget(self)
        self.gridLayout.addWidget(self.sketch_widget, 0, 2, 1, 2)
        # self.horizontalLayout.addWidget(self.sketch_widget, 0)

        self.clear_button.clicked.connect(self.clear_sketch)
        self.device = self.get_device()
        # self.load_model()
        self.build_model()
        self.count = 0

        self.sketch_widget.mouse_release.connect(self.predict)
        self.pen_size.valueChanged.connect(
            lambda value: setattr(self.sketch_widget, "pen_width", value)
        )

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
        image_tensor, q_image = self.sketch_widget.get_image_tensor()
        image_tensor.to(self.device)
        print("input tensor info")
        print(image_tensor.shape)
        print(image_tensor.shape)
        print(image_tensor.dtype)
        print(image_tensor.min(), image_tensor.max())
        print(image_tensor.device)

        # torch.save(image_tensor, f"image_{self.count}.pth")
        # self.count+=1
        prediction = self.model(image_tensor.to(self.device))
        self.numbers.display(prediction.argmax().item())
        print(f"prediction {prediction.argmax().item()}")
        print(prediction)
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
