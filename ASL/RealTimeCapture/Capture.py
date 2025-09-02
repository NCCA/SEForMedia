#!/usr/bin/env python3
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

# Visualization tools
import torchvision.transforms.v2 as transforms
from MainWindow import Ui_MainWindow
from PIL import Image
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QImage, QPainter, QPen, QPixmap
from qtpy.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p):
        kernel_size = 3
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool2d(2, stride=2),
        )

    def forward(self, x):
        return self.model(x)


class WebcamWidget(QWidget):
    updated_frame = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.image_label = QLabel(self)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.image_label)
        self.cropped_frame = None
        # Set up a QTimer to update the webcam feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms
        self.raw_frame = None

        # Open the webcam
        self.cap = cv2.VideoCapture(0)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to QImage
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # now draw a rectangle in the image for the area we are going to send to the model

            x, y, width, height = 200, 100, 400, 400
            painter = QPainter(qt_image)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawRect(x, y, width, height)
            painter.end()
            self.cropped_frame = qt_image.copy(x, y, width, height)
            # Display the QImage in the QLabel
            qt_image = qt_image.mirrored(True, False)
            # turn all pixels outside the bounding box black
            qt_image = qt_image.scaled(640, 480, Qt.KeepAspectRatio)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))
            # grab the raw frame data from the region of interest
            frame = frame[x : x + width, y : y + height]
            self.raw_frame = frame
            print(type(self.raw_frame))
            print(self.raw_frame.shape)
            self.updated_frame.emit()

    def closeEvent(self, event):
        # Release the webcam when the application is closed
        self.cap.release()
        event.accept()


class WebcamApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setGeometry(0, 0, 400, 400)  # Set initial size

        self.setWindowTitle("Webcam Viewer")
        self.webcam_widget = WebcamWidget(self)
        self.grid_layout.addWidget(self.webcam_widget, 0, 0, 1, 1)
        self.device = self.get_device()
        self.webcam_widget.updated_frame.connect(self.update_frame)
        # load the full model
        self.model = torch.load("asl_model_full.pth", map_location=self.device)
        # load Qimage from file
        image = QImage("amer_sign2.png")
        image = image.scaled(600, 400)
        self.asl_label.setPixmap(QPixmap.fromImage(image))
        # Open the webcam
        self.cap = cv2.VideoCapture(0)

    def update_frame(self):
        # convert this to a tensor and send to our model
        # Define the preprocessing transformations
        frame = self.webcam_widget.raw_frame
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        preprocess_trans = transforms.Compose(
            [
                transforms.Grayscale(),  # Convert to grayscale
                transforms.Resize((28, 28)),  # Resize to 28x28
                transforms.ToDtype(
                    torch.float32, scale=True
                ),  # Converts PIL image to tensor
                transforms.ToTensor(),
            ]
        )
        # Apply the transformations
        img = preprocess_trans(frame_pil)
        img = transforms.functional.adjust_brightness(img, 1.5)
        img = transforms.functional.adjust_contrast(img, 1.4)
        # img=transforms.functional.adjust_saturation(img,1.0)
        img = transforms.functional.adjust_sharpness(img, 2.0)
        pred = self.model(img.to(self.device).unsqueeze(0))
        print("Predicted class:", torch.argmax(pred).item())
        alphabet = "abcdefghijklmnopqrstuvwxy"
        print("Predicted letter:", alphabet[torch.argmax(pred).item()])
        self.predicted_letter.setText(f"{alphabet[torch.argmax(pred).item()]}")
        # display the image we are sending to the model
        img = self.tensor_to_qimage(img)
        img = img.scaled(200, 200)

        self.input_image.setPixmap(QPixmap.fromImage(img))
        self.cropped_image.setPixmap(
            QPixmap.fromImage(self.webcam_widget.cropped_frame)
        )

    def tensor_to_qimage(self, tensor: torch.Tensor) -> QImage:
        """
        Convert a PyTorch tensor to a QImage.

        Args:
            tensor (torch.Tensor): The input tensor with shape (C, H, W) and values in [0, 1].

        Returns:
            QImage: The resulting QImage.
        """
        # Ensure the tensor is on the CPU and detach it from the computation graph
        tensor = tensor.cpu().detach()

        # Convert the tensor to a NumPy array
        array = tensor.numpy()

        # Rescale the array to [0, 255] and convert to uint8
        array = (array * 255).astype(np.uint8)

        # Handle grayscale and RGB images
        if array.shape[0] == 1:  # Grayscale
            array = array.squeeze(0)  # Remove the channel dimension
            h, w = array.shape
            qimage = QImage(array.data, w, h, w, QImage.Format_Grayscale8)
        elif array.shape[0] == 3:  # RGB
            array = np.transpose(array, (1, 2, 0))  # Convert to (H, W, C)
            h, w, c = array.shape
            bytes_per_line = c * w
            qimage = QImage(array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            raise ValueError("Unsupported tensor shape: {}".format(array.shape))

        return qimage

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

    def closeEvent(self, event):
        # Release the webcam when the application is closed
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WebcamApp()
    window.show()
    sys.exit(app.exec_())
