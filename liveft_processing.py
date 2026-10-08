"""Numerical image processing shared by LiveFT and headless tools.

The module deliberately separates the centred numerical Fourier spectrum from
the transformations used to display that spectrum.  LiveFT's historical
display is a log-scaled power spectrum (``|FFT|**2``), normalized to [0, 1].
"""

from dataclasses import dataclass
from functools import lru_cache
from math import erf
from typing import Literal

import cv2
import numpy as np

SpectrumRepresentation = Literal["magnitude", "power"]
DisplayScaling = Literal["linear", "log"]


@dataclass(frozen=True)
class ProcessingConfig:
    """Configuration for :func:`process_image`.

    Crop coordinates are ``(x, y, width, height)`` in input pixels and output
    size is ``(width, height)``.  Defaults reproduce FrameProcessor's numerical
    operations for an uncropped, unscaled input.
    """

    crop: tuple[int, int, int, int] | None = None
    output_size: tuple[int, int] | None = None
    window: bool = True
    taper_width: float = 0.2
    representation: SpectrumRepresentation = "power"
    display_scaling: DisplayScaling = "log"
    gamma: float = 1.0
    suppress_center_lines: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.taper_width <= 1.0:
            raise ValueError("taper_width must be between 0 and 1")
        if self.gamma <= 0:
            raise ValueError("gamma must be greater than zero")
        if self.representation not in {"magnitude", "power"}:
            raise ValueError("representation must be 'magnitude' or 'power'")
        if self.display_scaling not in {"linear", "log"}:
            raise ValueError("display_scaling must be 'linear' or 'log'")
        if self.crop is not None and (self.crop[2] <= 0 or self.crop[3] <= 0):
            raise ValueError("crop width and height must be positive")
        if self.output_size is not None and (self.output_size[0] <= 0 or self.output_size[1] <= 0):
            raise ValueError("output width and height must be positive")


@dataclass(frozen=True)
class ProcessingResult:
    """Physical and display products for one processed image."""

    prepared_image: np.ndarray
    processed_image: np.ndarray
    spectrum: np.ndarray
    display_spectrum: np.ndarray


def normalize_unit(image: np.ndarray) -> np.ndarray:
    """Normalize by the finite positive maximum, matching LiveFT."""
    max_value = image.max()
    if not np.isfinite(max_value) or max_value <= 0:
        return np.zeros_like(image, dtype=np.float32)
    return (image / max_value).astype(np.float32, copy=False)


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply display gamma to a normalized image."""
    if gamma <= 0:
        raise ValueError("Gamma must be greater than zero.")
    clipped = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
    if gamma == 1.0:
        return clipped
    return np.power(clipped, gamma).astype(np.float32, copy=False)


@lru_cache(maxsize=16)
def error_function_window(shape: tuple[int, int], taper_width: float = 0.2) -> np.ndarray:
    """Return LiveFT's separable error-function edge window."""
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("window dimensions must be positive")
    if not 0.0 <= taper_width <= 1.0:
        raise ValueError("taper_width must be between 0 and 1")
    if taper_width == 0:
        return np.ones(shape, dtype=np.float32)

    x = np.linspace(-1.0, 1.0, width)
    y = np.linspace(-1.0, 1.0, height)
    window_x = np.array([erf((value + 1) / taper_width) * erf((1 - value) / taper_width) for value in x])
    window_y = np.array([erf((value + 1) / taper_width) * erf((1 - value) / taper_width) for value in y])
    # Keep float64 here to preserve FrameProcessor's historical arithmetic:
    # NumPy casts the in-place product back to the float32 camera frame.
    return np.multiply.outer(window_y, window_x)


def prepare_image(
    image: np.ndarray,
    crop: tuple[int, int, int, int] | None = None,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Crop, resize and convert a BGR/BGRA or grayscale array to float32 gray."""
    if image.ndim not in {2, 3}:
        raise ValueError("image must be a 2D grayscale or 3D BGR/BGRA array")
    prepared = image
    if crop is not None:
        x, y, width, height = crop
        if x < 0 or y < 0 or x + width > image.shape[1] or y + height > image.shape[0]:
            raise ValueError("crop lies outside the input image")
        prepared = prepared[y : y + height, x : x + width]
    if output_size is not None:
        prepared = cv2.resize(prepared, output_size)
    if prepared.ndim == 3:
        channels = prepared.shape[2]
        if channels == 3:
            prepared = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            prepared = cv2.cvtColor(prepared, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("color image must have three (BGR) or four (BGRA) channels")
    return prepared.astype(np.float32, copy=False)


def apply_window(image: np.ndarray, taper_width: float = 0.2) -> np.ndarray:
    """Apply LiveFT's vignette, subtract the minimum and normalize."""
    windowed = image.astype(np.float32, copy=True)
    windowed *= error_function_window(windowed.shape, taper_width)
    windowed -= windowed.min()
    return normalize_unit(windowed)


def suppress_center_lines(image: np.ndarray) -> np.ndarray:
    """Apply LiveFT's display-only central row/column replacement."""
    suppressed = image.copy()
    height, width = suppressed.shape
    if height < 5 or width < 5:
        raise ValueError("center-line suppression requires dimensions of at least 5 pixels")
    suppressed[height // 2 - 1 : height // 2 + 1, :] = suppressed[height // 2 + 1 : height // 2 + 3, :]
    suppressed[:, width // 2 - 1 : width // 2 + 1] = suppressed[:, width // 2 + 1 : width // 2 + 3]
    return suppressed


def compute_fourier_spectrum(
    image: np.ndarray,
    representation: SpectrumRepresentation = "power",
) -> np.ndarray:
    """Return a centred magnitude or power spectrum without display scaling."""
    if image.ndim != 2:
        raise ValueError("Fourier processing expects a 2D image")
    dft = cv2.dft(image.astype(np.float32, copy=False), flags=cv2.DFT_COMPLEX_OUTPUT)
    power = dft[:, :, 0] ** 2 + dft[:, :, 1] ** 2
    spectrum = np.sqrt(power) if representation == "magnitude" else power
    if representation not in {"magnitude", "power"}:
        raise ValueError("representation must be 'magnitude' or 'power'")
    return np.fft.fftshift(spectrum).astype(np.float32, copy=False)


def spectrum_for_display(
    spectrum: np.ndarray,
    scaling: DisplayScaling = "log",
    gamma: float = 1.0,
    suppress_lines: bool = False,
) -> np.ndarray:
    """Turn a numerical spectrum into a normalized display image."""
    if scaling == "log":
        displayed = np.log1p(spectrum)
    elif scaling == "linear":
        displayed = spectrum.copy()
    else:
        raise ValueError("scaling must be 'linear' or 'log'")
    if suppress_lines:
        displayed = suppress_center_lines(displayed)
    return apply_gamma(normalize_unit(displayed), gamma)


def process_image(image: np.ndarray, config: ProcessingConfig | None = None) -> ProcessingResult:
    """Process an array using the same numerical path as the LiveFT display."""
    config = config or ProcessingConfig()
    prepared = prepare_image(image, config.crop, config.output_size)
    if config.window:
        processed = apply_window(prepared, config.taper_width)
    else:
        processed = normalize_unit(prepared - prepared.min())
    spectrum = compute_fourier_spectrum(processed, config.representation)
    display = spectrum_for_display(
        spectrum,
        scaling=config.display_scaling,
        gamma=config.gamma,
        suppress_lines=config.suppress_center_lines,
    )
    return ProcessingResult(prepared, processed, spectrum, display)
