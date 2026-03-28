# Live Fourier Transform of Camera Feed

This Python program captures images from a camera feed, applies a Fourier transform (FFT) with OpenCV and NumPy, and
displays the transformed image alongside the original.

This can be used to explain concepts such as the Fourier Transform, but also scattering and diffraction effects, 
if the lecturer shows printouts of the library of shapes and arrays in front of the camera. 

<img width="806" alt="image" src="https://github.com/user-attachments/assets/1a6b2f9b-a5c9-4ac3-900e-31de1161e004">


## Prerequisites
Ensure you have Python 3.12 or above installed.

## Installation
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage
Run the program with:
```bash
python LiveFT.py
```

Press "q" to exit the application

## Options
- `-n, --numShots`: Maximum number of images before program exits.
- `-d, --camDevice`: Camera device ID to use.
- `-i, --imAvgs`: Number of images to average for display and FFT processing.
- `-y, --vScale`: Vertical video scaling factor.
- `-x, --hScale`: Horizontal video scaling factor.
- `-k, --killCenterLines`: Remove central lines from FFT image to improve dynamic range.
- `-f, --figid`: Name for the display window.
- `-r, --rows`: Number of central rows to use in video frame cropping.
- `-c, --columns`: Number of central columns to use in video frame cropping.
- `-o, --showRadialProfile`: Show the FFT radial distribution panel.
- `-e, --fftGamma`: Set the FFT display gamma.
- `-m, --maxFPS`: Limit processing speed to N FPS. Use `0` to disable the limit.

### Notes
- This program uses OpenCV and NumPy for FFT calculations.
- Tested on MacOS and Linux environments.

## License
This project is licensed under the Apache-2.0 license.
