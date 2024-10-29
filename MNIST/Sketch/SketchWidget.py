import numpy as np
import torch
from qtpy.QtCore import QPoint, QRect, Qt, Signal
from qtpy.QtGui import QImage, QPainter, QPen, QPixmap
from qtpy.QtWidgets import QWidget
from scipy.ndimage import center_of_mass
from torchvision.transforms import ToPILImage, functional as F


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
        self._resize_image(self.image, self.size())
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
        newImage.fill(Qt.black)
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
        # Step 1: Capture the current image
        image = self.image
        # resized_image = image.scaled(28, 28, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        # # Step 3: Convert the resized image to a grayscale numpy array
        # grayscale_image = resized_image.convertToFormat(QImage.Format_Grayscale8)
        # buffer = grayscale_image.bits()
        # buffer.setsize(grayscale_image.byteCount())

        grayscale_image = image.convertToFormat(QImage.Format_Grayscale8)
        grayscale_image = grayscale_image.scaled(
            28, 28, Qt.IgnoreAspectRatio, Qt.FastTransformation
        )
        buffer = grayscale_image.bits()
        buffer.setsize(grayscale_image.byteCount())
        image_array = np.array(buffer).reshape((grayscale_image.height(), grayscale_image.width()))
        # Calculate the center of mass
        com_y, com_x = center_of_mass(image_array)
        center_y, center_x = np.array(image_array.shape) / 2
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

        np_image = np.array(centered_image_pil)

        # Step 4: Normalize the numpy array and convert to PyTorch tensor
        np_image = np_image.astype(np.float32)
        tensor_image = torch.tensor(np_image).unsqueeze(0)  # Add batch and channel dimensions

        return tensor_image, QPixmap.fromImage(grayscale_image)

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
