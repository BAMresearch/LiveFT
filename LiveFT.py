#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Overview
========
This program captures images from the camera, applies a Fourier transform using PyTorch,
and displays the transformed image alongside the original on the screen.

To run:
    $ python3 liveFT.py
For command-line options, use:
    $ python3 liveFT.py --help

press "q" to exit the application
    
Author: Brian R. Pauw with some suggestions from AI
Contact: brian@stack.nl
License: Apache-2.0
"""

from typing import Any, Tuple
import time
import numpy as np
import torch
import cv2
import argparse
from attrs import define, field, validators

# Function to parse arguments for the script
def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Live Fourier Transform of camera feed.")
    parser.add_argument("-n", "--numShots", type=int, default=1e5, help="Max. number of images before exit")
    # parser.add_argument("-N", "--numBins", type=int, default=200, help="Number of integration bins")
    # parser.add_argument("-o", "--nContrIms", type=int, default=30, help="Calculate average contrast over N images")
    parser.add_argument("-d", "--camDevice", type=int, default=0, help="Camera device ID")
    parser.add_argument("-i", "--imAvgs", type=int, default=1, help="Average N images for display and FFT")
    parser.add_argument("-y", "--vScale", type=float, default=1.2, help="Vertical video scale")
    parser.add_argument("-x", "--hScale", type=float, default=1.2, help="Horizontal video scale")
    parser.add_argument("-p", "--downScale", action="store_true", help="Enable pyramidal downscaling")
    parser.add_argument("-k", "--killCenterLines", action="store_true", help="Remove central lines from FFT image")
    parser.add_argument("-f", "--figid", type=str, default="liveFFT by Brian R. Pauw - press 'q' to exit.", help="Image window name")
    parser.add_argument("-r", "--rows", type=int, default=500, help="Use center N rows of video")
    parser.add_argument("-c", "--columns", type=int, default=500, help="Use center N columns of video")
    # parser.add_argument("-P", "--plot", action="store_true", help="Enable a separate 1D plot window")

    return parser.parse_args()

@define
class LiveFT:
    """Handles live Fourier Transform display of camera feed."""

    # Core attributes with default values from command-line arguments
    numShots: int = field(default=1e5, metadata={"help": "Max number of images before program exits"})
    # numBins: int = field(default=200, metadata={"help": "Number of integration bins"})
    # nContrIms: int = field(default=30, metadata={"help": "Average contrast over N images"})
    camDevice: int = field(default=0, metadata={"help": "Camera device ID"})
    imAvgs: int = field(default=1, metadata={"help": "Average N images for display and FFT"})
    vScale: float = field(default=1.0, metadata={"help": "Vertical video scale"})
    hScale: float = field(default=1.0, metadata={"help": "Horizontal video scale"})
    downScale: bool = field(default=False, metadata={"help": "Enable pyramidal downscaling"})
    killCenterLines: bool = field(default=False, metadata={"help": "Remove central lines from FFT image"})
    figid: str = field(default="liveFFT by Brian R. Pauw - press 'q' to exit.", metadata={"help": "Image window name"})
    rows: int = field(default=400, metadata={"help": "Use center N rows of video"})
    columns: int = field(default=400, metadata={"help": "Use center N columns of video"})

    # Derived attributes initialized post-instantiation
    device: torch.device = field(init=False, validator=validators.instance_of(torch.device))
    vc: cv2.VideoCapture = field(init=False, validator=validators.instance_of(cv2.VideoCapture))
    frame_shape: Tuple[int, int, int] = field(init=False)
    v_crop: Tuple[int, int] = field(init=False)
    h_crop: Tuple[int, int] = field(init=False)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize video capture and plotting after attribute setup."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for kw, value in kwargs.items():
            setattr(self, kw, value)
        # Open camera device
        self.vc = cv2.VideoCapture(self.camDevice)
        if not self.vc.isOpened():
            raise ValueError("Could not open video device.")
        
        # Initialize display window
        cv2.namedWindow(self.figid, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.figid, 1024, 768)

        # Capture first frame to determine frame shape
        success, frame = self.vc.read()
        if not success:
            raise ValueError("Failed to capture initial frame.")
        self.frame_shape = frame.shape  # Frame shape for cropping setup

        # Setup cropping and plotting
        self._setup_cropping()

        # Start main loop for capturing and processing frames
        self.run()

    def _setup_cropping(self) -> None:
        """Configure cropping boundaries for the center region of the frame."""
        rows, columns = self.rows, self.columns
        height, width = self.frame_shape[:2]

        # Ensure crop dimensions are within frame limits
        if rows > height:
            rows = height
        if columns > width:
            columns = width

        self.v_crop = (height // 2 - rows // 2, height // 2 + rows // 2)
        self.h_crop = (width // 2 - columns // 2, width // 2 + columns // 2)

    def run(self) -> None:
        """Main loop to capture and process frames from the camera."""
        num_frames = 0
        while num_frames < self.numShots:
            num_frames += 1
            start_time = time.time()
            self.process_frame()

            # Capture key press to close window (e.g., 'q' key)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting on user request.")
                break

            # Check if the window is still open, break if closed
            if not cv2.getWindowProperty(self.figid, cv2.WND_PROP_VISIBLE):
                print("Window closed by user.")
                break

            fps = 1 / (time.time() - start_time)
            print(f"Frame {num_frames}, FPS: {fps:.2f}", end="\r")

        self.vc.release()
        cv2.destroyAllWindows()

    def process_frame(self) -> np.ndarray:
        """Capture, process, and display a single frame."""

        success, iframe = self.vc.read()
        if not success:
            return np.array([])
        frame = iframe.astype(np.float32) / self.imAvgs
        nframes = 1
        while nframes < self.imAvgs:
            nframes += 1
            success, iframe = self.vc.read()
            frame += iframe.astype(np.float32) / self.imAvgs
            if not success:
                return np.array([])
        
        # Prepare and compute FFT on the frame
        frame_tensor = self._process_image(frame)
        # output is numpy array
        fft_image = self._compute_fft(frame_tensor)
        # normalize and convert to numpy array
        frame = (frame_tensor / frame_tensor.max()).cpu().numpy().clip(0, 1)

        cv2.imshow(self.figid, np.concatenate((frame, fft_image), axis=1))
        return
        # return self.contrast
    
    def _process_image(self, frame: np.ndarray) -> np.ndarray:
        """Crop, scale, and normalize the captured frame."""
        # Crop the frame to the specified center region
        frame = frame[self.v_crop[0]:self.v_crop[1], self.h_crop[0]:self.h_crop[1]]
        
        # Scale frame dimensions if necessary
        if self.vScale != 1 or self.hScale != 1:
            frame = cv2.resize(frame, None, fx=self.hScale, fy=self.vScale)
 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # convert to torch tensor
        frame_tensor = torch.tensor(frame, device=self.device)

        # Apply an error function window
        h, w = frame_tensor.shape
        y = torch.linspace(-1.0, 1.0, h, device=self.device)
        x = torch.linspace(-1.0, 1.0, w, device=self.device)
        x, y = torch.meshgrid(x, y, indexing='xy')
        
        # Create a window using the error function
        taper_width = 0.2  # Adjust the taper width as necessary
        window_x = torch.erf((x + 1) / taper_width) * torch.erf((1 - x) / taper_width)
        window_y = torch.erf((y + 1) / taper_width) * torch.erf((1 - y) / taper_width)
        window = window_x * window_y
        
        # Apply the window to the frame
        frame_tensor *= window
        
        # expand range:
        frame_tensor = (frame_tensor - frame_tensor.min()) / (frame_tensor.max() - frame_tensor.min()).cpu()

        return frame_tensor

    def _compute_fft(self, frame_tensor) -> np.ndarray:
        """Perform FFT on the frame using PyTorch, with optional line removal."""
        # Convert frame to PyTorch tensor on the designated device (GPU/CPU)
        # frame_tensor = torch.tensor(frame, device=self.device)
        
        # Convert to grayscale if color display is disabled
        # if not self.color:
        #     frame_tensor = frame_tensor.mean(dim=2)
        
        # Compute the FFT and take the magnitude (squared)
        fft_tensor = torch.fft.fftshift(torch.abs(torch.fft.fft2(frame_tensor)) ** 2)
        
        # Apply logarithmic scaling for better contrast in visualization
        fft_tensor = torch.log1p(fft_tensor)

        # Optionally remove central lines to enhance dynamic range in display
        if self.killCenterLines:
            h, w = fft_tensor.shape[:2]
            fft_tensor[h // 2 - 1:h // 2 + 1, :] = fft_tensor[h // 2 + 1:h // 2 + 3, :]
            fft_tensor[:, w // 2 - 1:w // 2 + 1] = fft_tensor[:, w // 2 + 1:w // 2 + 3]
        
        # Normalize and convert back to NumPy array for display
        fft_image = (fft_tensor / fft_tensor.max()).cpu().numpy().clip(0, 1)
        return fft_image

if __name__ == "__main__":
    args = parse_args()
    live_ft = LiveFT(**vars(args))
