from typing import Any, Tuple
import sys
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PIL import Image
from attrs import define, field, validators

@define
class LiveFT:
    """Handles live Fourier Transform display of camera feed."""
    camDevice: int = field(default=0, metadata={"help": "Camera device ID"})
    vScale: float = field(default=1.0, metadata={"help": "Vertical video scale"})
    hScale: float = field(default=1.0, metadata={"help": "Horizontal video scale"})
    killCenterLines: bool = field(default=False, metadata={"help": "Remove central lines from FFT image"})
    rows: int = field(default=400, metadata={"help": "Use center N rows of video"})
    columns: int = field(default=400, metadata={"help": "Use center N columns of video"})
    device: torch.device = field(init=False, validator=validators.instance_of(torch.device))
    vc: cv2.VideoCapture = field(init=False, validator=validators.instance_of(cv2.VideoCapture))



    def __attrs_post_init__(self) -> None:
        # Initialize device and capture object
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vc = cv2.VideoCapture(self.camDevice)

    def capture_frame(self) -> Tuple[bool, np.ndarray]:
        """Captures a frame from the camera."""
        return self.vc.read()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with FFT and return processed image."""
        # Ensure the frame is cropped to the specified rows and columns
        height, width = frame.shape[:2]
        row_start = max((height - self.rows) // 2, 0)
        col_start = max((width - self.columns) // 2, 0)
        frame_cropped = frame[row_start:row_start + self.rows, col_start:col_start + self.columns]
        
        # Convert to grayscale for FFT
        frame_gray = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2GRAY)
        frame_tensor = torch.tensor(frame_gray, device=self.device, dtype=torch.float32) / 255.0
        
        # Perform FFT and shift
        fft_tensor = torch.fft.fftshift(torch.abs(torch.fft.fft2(frame_tensor)) ** 2)
        fft_tensor = torch.log1p(fft_tensor)
        
        # Optionally remove central lines
        if self.killCenterLines:
            h, w = fft_tensor.shape
            fft_tensor[h // 2 - 1:h // 2 + 1, :] = fft_tensor[h // 2 + 1:h // 2 + 3, :]
            fft_tensor[:, w // 2 - 1:w // 2 + 1] = fft_tensor[:, w // 2 + 1:w // 2 + 3]
        
        # Normalize and convert to numpy for display
        processed_np = (fft_tensor / fft_tensor.max()).cpu().numpy().clip(0, 1)
        return (frame_gray / 255.0).astype(np.uint8), (processed_np * 255).astype(np.uint8)

    def release(self):
        """Releases the camera device."""
        self.vc.release()


class CameraViewer(QMainWindow):
    def __init__(self, live_ft: LiveFT, width: int = 1024, height: int = 400):
        super().__init__()
        self.live_ft = live_ft

        # Set up PyQt window layout
        self.resize(width, height)  # Set initial window size (width and height in pixels)
        self.setWindowTitle("Live Camera Feed with PyTorch FFT Processing")
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # QLabel to display images
        self.image_label = QLabel(self)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Allow QLabel to resize with window
        self.image_label.setScaledContents(True)  # Scale image to fit label size dynamically
        self.layout.addWidget(self.image_label)

        # Timer to update the image
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms

    def update_frame(self):
        # Capture and process frame
        ret, frame = self.live_ft.capture_frame()
        if not ret:
            self.close()  # Close window if frame capture fails
            return
        
        # Process frame using FFT
        frame_image, fft_image = self.live_ft.process_frame(frame)

        # Concatenate original and FFT images side-by-side for display
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # combined_image = np.concatenate((frame_rgb, cv2.cvtColor(fft_image, cv2.COLOR_GRAY2RGB)), axis=1)
        combined_image = np.concatenate((frame_image, fft_image), axis=1)

        # Convert combined image to QImage for PyQt display
        image = Image.fromarray(combined_image)
        qimage = self.pil2qpixmap(image)
        
        # Update the QLabel image in the PyQt window
        self.image_label.setPixmap(qimage)
        # self.image_label.setFixedSize(800, 400)  # Set a fixed size for the QLabel to keep window size under control
        self.image_label.setScaledContents(True)  # Scale image to fit label size

    def pil2qpixmap(self, image: Image) -> QPixmap:
        """Convert PIL Image to QPixmap for display."""
        qimage = QImage(image.tobytes(), image.width, image.height, QImage.Format_RGB888)
        return QPixmap.fromImage(qimage)

    def closeEvent(self, event):
        """Close event handler to release camera resources."""
        self.live_ft.release()
        event.accept()


# Main application entry point
app = QApplication(sys.argv)
live_ft = LiveFT(camDevice=0, vScale=0.5, hScale=0.5, killCenterLines=False, rows=400, columns=400)
viewer = CameraViewer(live_ft=live_ft)
viewer.show()
sys.exit(app.exec_())
