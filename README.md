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

## Packaging
Build a distributable bundle with:
```bash
pip install -r requirements_for_testing.txt
sh scripts/build_dist.sh
```

The build script auto-detects `.venv/bin/python`, `python`, or `python3`. Override it with
`LIVEFT_PYTHON=/path/to/python` when needed.

Useful packaging environment variables:
- `LIVEFT_BUNDLE_ID`
- `LIVEFT_CAMERA_USAGE`
- `LIVEFT_CODESIGN_IDENTITY`
- `LIVEFT_ENTITLEMENTS_FILE`
- `LIVEFT_ICON_PATH`
- `PYINSTALLER_CONFIG_DIR`

On macOS, create a DMG after the app bundle is built:
```bash
sh scripts/create_dmg.sh dist
```

Set `LIVEFT_KEEP_APP_AFTER_DMG=1` if you want to keep the `.app` bundle in `dist/` after the DMG is created.

For a local macOS release build:
```bash
LIVEFT_CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)" sh scripts/release_macos.sh
```

Additional macOS release scripts:
- `sh scripts/sign_macos.sh dist`
- `sh scripts/notarize_macos.sh dist`
- `sh scripts/verify_macos.sh dist`

The GitHub workflow uses these optional secrets for signed/notarized macOS releases:
- `MACOS_CODESIGN_IDENTITY`
- `MACOS_CERTIFICATE_P12_BASE64`
- `MACOS_CERTIFICATE_PASSWORD`
- `MACOS_NOTARY_KEY_ID`
- `MACOS_NOTARY_ISSUER`
- `MACOS_NOTARY_KEY_P8`

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
