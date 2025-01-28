import pymupdf
import io
from PIL import Image
import cv2
import numpy as np
import sys
from pathlib import Path

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

def writeFTImages(showImages=False):
    # calc the fft for every image in each path given above
    for pdf_path in [pdf for sub in subdirs for pdf in (Path(inputDir)/Path(sub)).iterdir()]:
        frame = pdfToCV(pdf_path)
        if frame is None:
            continue
        if showImages:
            showArray(frame, f"Extracted Image {pdf_path}")

        frame_prepared = LiveFT._process_image(frame.astype(np.float32), h_crop=None, v_crop=None, h_scale=1, v_scale=1, device=None)
        # output is numpy array
        fft_image = LiveFT._compute_fft(frame_prepared, False)
        # print(f"{type(fft_image)=}")
        if showImages:
            showArray(fft_image, "FFT Image")

        out_path = Path(outputDir) / ((pdf_path.with_suffix(".png")).relative_to(inputDir))
        if not out_path.parent.is_dir():
            out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, (fft_image * 255).astype(np.uint8))
        print(f"Wrote '{out_path}'")

def testFT():
    """Verifies that the current implementation generates the expected Fourier transforms from a set of example images."""
    pass

#writeFTImages()
