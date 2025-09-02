import numpy as np
import torch
from qtpy.QtCore import QPoint, QRect, Qt, Signal
from qtpy.QtGui import QImage, QPainter, QPen, QPixmap, QColor
from qtpy.QtWidgets import QWidget
from scipy.ndimage import center_of_mass
from torchvision.transforms import ToPILImage, functional as F


class SketchWidget(QWidget):
    # Define a signal for the mouse release event in the widget
    mouse_release = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_StaticContents)
        self.modified = False
        self.scribbling = False
        self.pen_width = 10
        self.border = 5
        self.pen_colour = Qt.white
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)
        self.last_point = QPoint()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.last_point = event.pos()
            self.scribbling = True

    def mouseMoveEvent(self, event) -> None:
        if (event.buttons() & Qt.LeftButton) and self.scribbling:
            self._draw_to(event.pos())

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.scribbling:
            self._draw_to(event.pos())
            self.scribbling = False
            self.mouse_release.emit()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        dirtyRect = event.rect()
        painter.drawImage(dirtyRect, self.image, dirtyRect)

    def resizeEvent(self, event) -> None:
        self._resize_image(self.image, self.size())
        super().resizeEvent(event)

    def _draw_to(self, endPoint) -> None:
        painter = QPainter(self.image)

        painter.setPen(
            QPen(
                self.pen_colour, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin
            )
        )
        painter.drawLine(self.last_point, endPoint)
        self.modified = True

        rad = self.pen_width // 2 + 2
        self.update(
            QRect(self.last_point, endPoint)
            .normalized()
            .adjusted(-rad, -rad, +rad, +rad)
        )
        self.last_point = QPoint(endPoint)

    def _resize_image(self, image, newSize) -> None:
        if image.size() == newSize:
            return
        newImage = QImage(newSize, QImage.Format_RGB32)
        newImage.fill(Qt.black)
        painter = QPainter(newImage)
        painter.drawImage(QPoint(0, 0), image)
        self.image = newImage

    def _crop_image(self, image) -> None:
        def is_non_black(pixel):
            return pixel != QColor(Qt.black)

        # Find the first and last non-black rows
        first_black_row = next(
            (
                row
                for row in range(image.height())
                if any(
                    is_non_black(image.pixelColor(col, row))
                    for col in range(image.width())
                )
            ),
            -1,
        )
        last_black_row = next(
            (
                row
                for row in range(image.height() - 1, -1, -1)
                if any(
                    is_non_black(image.pixelColor(col, row))
                    for col in range(image.width())
                )
            ),
            -1,
        )

        # Find the first and last non-black columns
        first_black_col = next(
            (
                col
                for col in range(image.width())
                if any(
                    is_non_black(image.pixelColor(col, row))
                    for row in range(image.height())
                )
            ),
            -1,
        )
        last_black_col = next(
            (
                col
                for col in range(image.width() - 1, -1, -1)
                if any(
                    is_non_black(image.pixelColor(col, row))
                    for row in range(image.height())
                )
            ),
            -1,
        )

        # Crop the image to the bounding box
        bounding_box = QRect(
            first_black_col - self.border,
            first_black_row - self.border,
            (last_black_col - first_black_col) + 2 * self.border,
            (last_black_row - first_black_row) + 2 * self.border,
        )
        return image.copy(bounding_box)

    def center_image_by_mass(self) -> None:
        """process the image by centering it using the center of mass"""

        image = self.image
        # convert to grey scale and smaller size
        grayscale_image = image.convertToFormat(QImage.Format_Grayscale8)
        grayscale_image = grayscale_image.scaled(
            28, 28, Qt.IgnoreAspectRatio, Qt.FastTransformation
        )
        buffer = grayscale_image.bits()
        buffer.setsize(grayscale_image.byteCount())
        # we need a numpy array for the center of mass
        image_array = np.array(buffer).reshape(
            (grayscale_image.height(), grayscale_image.width())
        )
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
        centered_image_pil = centered_image_pil.resize((28, 28))

        np_image = np.array(centered_image_pil)
        # now convert to tensor and return
        np_image = np_image.astype(np.float32)
        tensor_image = torch.tensor(np_image).unsqueeze(
            0
        )  # Add batch and channel dimensions

        return tensor_image, QPixmap.fromImage(grayscale_image)

    def get_image_tensor(self):
        image = self.image
        # crop the image to the bounding box (+ boarder)
        resize_image = self._crop_image(image)
        # Step 2: Resize the image to 28x28 pixels
        resized_image = resize_image.scaled(
            28, 28, Qt.IgnoreAspectRatio, Qt.FastTransformation
        )
        # Step 3: Convert the resized image to a grayscale numpy array
        grayscale_image = resized_image.convertToFormat(QImage.Format_Grayscale8)
        buffer = grayscale_image.bits()
        buffer.setsize(grayscale_image.byteCount())
        np_image = np.array(buffer).reshape((28, 28))
        # convert to tensor
        np_image = np_image.astype(np.float32)
        tensor_image = torch.tensor(np_image).unsqueeze(
            0
        )  # Add batch and channel dimensions
        return tensor_image, QPixmap.fromImage(grayscale_image)
