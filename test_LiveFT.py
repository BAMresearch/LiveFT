import pymupdf
import io
from PIL import Image
import cv2
import numpy as np
import sys
from pathlib import Path
import pytest

from LiveFT import LiveFT

inputDir  = "images"
outputDir = "testdata"
subdirs = ("singleObjects", "arrays")

def pixmapToCV(pix):
    # Get the width and height of the pixmap
    width = pix.width
    height = pix.height

    # Extract pixel data as a byte array
    img_data = pix.samples

    # Create a NumPy array from the byte data
    img_array = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, pix.n))

    # Convert from RGB to BGR if necessary (OpenCV uses BGR format)
    if pix.n == 3:  # RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif pix.n == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

    return img_array

def pdfToCV(pdf_path, page_number=0):
    print(f"Reading '{pdf_path}'")
    # Open the PDF file
    pdf_document = pymupdf.open(pdf_path)

    # Get the specified page
    page = pdf_document.load_page(page_number)

    # Convert PIL image to OpenCV format (BGR)
    cv2_image = pixmapToCV(page.get_pixmap(dpi=90))

    return cv2_image  # Return the OpenCV frame

def showArray(frame, title=""):
    cv2.imshow(title, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def examplePDFs():
    return [fn for sub in subdirs for fn in (Path(inputDir)/Path(sub)).iterdir()
            if fn.suffix.lower() == ".pdf"]

def writeFTImage(dest_path, src_image):
    if not dest_path.parent.is_dir():
        dest_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(dest_path, src_image)
    print(f"Wrote '{dest_path}'")

def readFTImage(src_path):
    assert src_path.is_file()
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    return img

def pytest_generate_tests(metafunc):
    src_paths = examplePDFs()
    fft_paths = [Path(outputDir) / ((fn.with_suffix(".png")).relative_to(inputDir)) for fn in src_paths]
    # there argument names relate to the following function definition below
    if "src_path" in metafunc.fixturenames and "fft_path" in metafunc.fixturenames:
        metafunc.parametrize(("src_path", "fft_path"), zip(src_paths, fft_paths))

def testFT(src_path, fft_path, showImages=False, writeImages=False):
    print(f"{src_path=} {fft_path=}")
    frame_in = pdfToCV(src_path)
    assert frame_in is not None
    if showImages:
        showArray(frame_in, f"Extracted Image {src_path}")
    frame_prepared = LiveFT._process_image(frame_in.astype(np.float32), h_crop=None, v_crop=None, h_scale=1, v_scale=1, device=None)
    # output is numpy array
    fft_image = LiveFT._compute_fft(frame_prepared, False)
    assert fft_image is not None
    fft_image = (fft_image * 255).astype(np.uint8)
    # print(f"{type(fft_image)=}")
    if showImages:
        showArray(fft_image, "FFT Image")
    if writeImages:
        writeFTImage(fft_path, fft_image)
        return
    fft_expected = readFTImage(fft_path)
    assert fft_expected.dtype == fft_image.dtype
    assert fft_expected.shape == fft_image.shape
    def printRange(arr, title, *args):
        print(f"{title} range: [{arr.min():.12e}, {arr.max():.12e}], mean: {arr.mean()}", *args)
    printRange(fft_expected, "fft_expected")
    printRange(fft_image, "fft_image")
    fft_diff = np.abs(fft_expected.astype(np.float32) - fft_image.astype(np.float32))
    printRange(fft_diff, "fft_diff", fft_diff.max() <= 1., fft_diff.mean() < 1e-3)
    # float comparison of 2d FT arrays, there as assumed to be equivalent in case:
    assert fft_diff.max() <= 1. and fft_diff.mean() < 1e-3

if __name__ == "__main__":
    # calc the fft for every image in each path given above
    for src_path in examplePDFs():
        fft_path = Path(outputDir) / ((src_path.with_suffix(".png")).relative_to(inputDir))
        testFT(src_path, fft_path, showImages=False, writeImages=False)
