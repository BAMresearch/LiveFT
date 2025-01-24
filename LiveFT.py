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

@define
class LiveFT:
    """Handles live Fourier Transform display of camera feed."""

    # Core attributes with default values from command-line arguments
    # Note: This order affects parse_args() below, all attrs until device become cmdline args
    numShots: int = field(default=1e5,
                          metadata={"help": "Max number of images before program exits", "short": "n"})
    # numBins: int = field(default=200, metadata={"help": "Number of integration bins", "short": "N"})
    # nContrIms: int = field(default=30, metadata={"help": "Average contrast over N images", "short": "o"})
    camDevice: int = field(default=0,
                           metadata={"help": "Camera device ID", "short": "d"})
    imAvgs: int = field(default=1,
                        metadata={"help": "Average N images for display and FFT", "short": "a"})
    vScale: float = field(default=1.2,
                          metadata={"help": "Vertical video scale", "short": "y"})
    hScale: float = field(default=1.2,
                          metadata={"help": "Horizontal video scale", "short": "x"})
    downScale: bool = field(default=False,
                            metadata={"help": "Enable pyramidal downscaling", "short": "p"})
    killCenterLines: bool = field(default=False,
                                  metadata={"help": "Remove central lines from FFT image", "short": "k"})
    figid: str = field(default="liveFFT by Brian R. Pauw - press 'q' to exit.",
                       metadata={"help": "Image window name", "short": "f"})
    rows: int = field(default=500, metadata={"help": "Use center N rows of video", "short": "r"})
    columns: int = field(default=500, metadata={"help": "Use center N columns of video", "short": "c"})
    showInfo: bool = field(default=False, metadata={"help": "Show FPS info text overlay", "short": "i"})
    noGPU: bool = field(default=True,
                        metadata={"help": "Switch between CPU or GPU for Fourier Transform", "short": "g"})

    # Derived attributes initialized post-instantiation
    device: torch.device = field(init=False, validator=validators.instance_of(torch.device))
    vc: cv2.VideoCapture = field(init=False, validator=validators.instance_of(cv2.VideoCapture))
    frame_shape: Tuple[int, int, int] = field(init=False)
    v_crop: Tuple[int, int] = field(init=False)
    h_crop: Tuple[int, int] = field(init=False)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize video capture and plotting after attribute setup."""

        for kw, value in kwargs.items():
            # print(f"{kw=} {value=}") # show actual configuration for debugging
            setattr(self, kw, value)
        # Open camera device
        self.vc = cv2.VideoCapture(self.camDevice)
        if not self.vc.isOpened():
            raise ValueError("Could not open video device.")
        # init torch calculation device for fourier transform
        self.device = torch.device("cuda"
            if self.noGPU and torch.cuda.is_available() else "cpu")
        
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

    def drawInfoText(self, frame, data) -> None:
        org = (50, 50)  # Coordinates of the bottom-left corner of the text string
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = .7
        color = (255, 255, 255)  # White color in BGR
        thickness = 2
        cv2.putText(frame, ", ".join([f"{k}: {v}" for k,v in data.items()]),
                    org, font, font_scale, color, thickness)

    def run(self) -> None:
        """Main loop to capture and process frames from the camera."""
        num_frames = 0
        frames_counted = 0
        start_time = time.time()
        infoData = {"#Frame": 0, "fps": "", "torch": str(self.device)}
        if infoData["torch"].lower() != "cpu":
            # get GPU name if enabled
            infoData["torch"] = torch.cuda.get_device_name(torch.cuda.current_device())
        while num_frames < self.numShots:
            num_frames += 1

            # Capture key press to close window (e.g., 'q' key)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                print("Exiting on user request.")
                break
            elif key & 0xFF == ord('i'):
                self.showInfo = not self.showInfo

            # Check if the window is still open, break if closed
            if not cv2.getWindowProperty(self.figid, cv2.WND_PROP_VISIBLE):
                print("Window closed by user.")
                break

            frame_final = self.process_frame()
            # gather some info
            elapsed = time.time() - start_time
            fps = (num_frames - frames_counted) / elapsed
            if elapsed > 2: # duration of FPS measurement window
                start_time = time.time()
                frames_counted = num_frames
            # print(f"Frame {num_frames}, FPS: {fps:.2f}", end="\r")

            # Show info text on request
            if self.showInfo:
                infoData.update({"#Frame": num_frames, "fps": f"{fps:.2f}"})
                self.drawInfoText(frame_final, infoData)

            if frame_final.size: # show the frame if there is any
                (wx, wy, ww, wh) = cv2.getWindowImageRect(self.figid)
                (fh, fw) = frame_final.shape
                if num_frames == 1 and (ww != fw or wh != fh):
                    # resize appropriately only once initially
                    cv2.resizeWindow(self.figid, fw, fh)
                cv2.imshow(self.figid, frame_final)

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
        frames_combined = np.concatenate((frame, fft_image), axis=1)
        return frames_combined
    
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

# Function to parse arguments for the script
def parse_args(liveftCls: LiveFT) -> argparse.Namespace:
    """Parses command-line arguments.
    Uses the LiveFT class for some options configuration."""
    parser = argparse.ArgumentParser(description="Live Fourier Transform of camera feed.")
    for attr in liveftCls.__attrs_attrs__:
        if attr.name == "device":
            break
        # print(f"{attr=}") # class config for debugging
        pkwargs = dict(help=attr.metadata["help"])
        if attr.type is bool:
            pkwargs["action"] = "store_true" if not attr.default else "store_false"
        else:
            pkwargs.update(type=attr.type, default=attr.default)
        # print(f"{pkwargs}") # show parser config for debugging
        parser.add_argument("-"+attr.metadata["short"], "--"+attr.name, **pkwargs)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args(LiveFT)
    live_ft = LiveFT(**vars(args))
