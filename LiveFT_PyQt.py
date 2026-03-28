# test with PyQt as UI:

import sys
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PIL import Image


class CameraViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize camera
        self.cap = cv2.VideoCapture(0)  # Use 0 for default camera

        # Set up the PyQt window layout
        self.setWindowTitle("Live Camera Feed with PyTorch Processing")
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # QLabel to display images
        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        # Timer to update the image
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms (around 33 FPS)

    def update_frame(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert frame to PyTorch tensor, process, and convert back to displayable format
        frame_tensor = self.process_frame_with_pytorch(frame)
        image = Image.fromarray((frame_tensor * 255).astype(np.uint8))  # Scale to 0-255 for display
        qimage = self.pil2qpixmap(image)

        # Display the image on the QLabel
        self.image_label.setPixmap(qimage)

    def process_frame_with_pytorch(self, frame: np.ndarray) -> np.ndarray:
        """Process frame using PyTorch and return as NumPy array for display."""
        # Convert OpenCV frame (BGR) to RGB and PyTorch tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0  # [H, W, C] -> [C, H, W]

        # Example processing: Convert to grayscale and return to NumPy format
        gray_tensor = tensor.mean(dim=0, keepdim=True)  # Simple grayscale conversion
        processed_tensor = gray_tensor.repeat(3, 1, 1)  # Convert back to [C, H, W] for RGB display

        # Convert processed tensor to NumPy and reshape to [H, W, C]
        processed_np = processed_tensor.permute(1, 2, 0).cpu().numpy()
        return processed_np

    def pil2qpixmap(self, image: Image) -> QPixmap:
        """Convert PIL Image to QPixmap for display."""
        qimage = QImage(image.tobytes(), image.width, image.height, QImage.Format_RGB888)
        return QPixmap.fromImage(qimage)

    def closeEvent(self, event):
        """Close event handler to release the camera."""
        self.cap.release()
        event.accept()


# Main application entry point
app = QApplication(sys.argv)
viewer = CameraViewer()
viewer.show()
sys.exit(app.exec_())
