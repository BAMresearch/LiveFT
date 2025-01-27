import pymupdf
import io
from PIL import Image
import cv2
import numpy as np
import sys
from pathlib import Path

paths = ("images/singleObjects", "images/arrays")

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
    # Open the PDF file
    pdf_document = pymupdf.open(pdf_path)

    # Get the specified page
    page = pdf_document.load_page(page_number)

    # Convert PIL image to OpenCV format (BGR)
    cv2_image = pixmapToCV(page.get_pixmap())

    return cv2_image  # Return the OpenCV frame

# calc the fft for every image in each path given above
for pdf_path in [pdf for p in paths for pdf in Path(p).iterdir()]:
    print(pdf_path)
    frame = pdfToCV(pdf_path)
    if frame is None:
        continue
    cv2.imshow("Extracted Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
