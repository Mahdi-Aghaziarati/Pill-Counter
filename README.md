# Pill-Counter
A professional real-time pill counting application using OpenCV, Tkinter and Python. Features auto/manual thresholding, watershed separation, HSV color filtering, ROI selection, CSV logging and full GUI control.

## Features

- Real-time camera feed with original and processed mask views
- Automatic (Otsu) and manual thresholding
- HSV color filtering for colored pills
- Watershed separation for touching pills
- Circularity and area-based filtering
- Adjustable ROI (rectangle or circle)
- Live pill count, FPS, and processing time
- Snapshot export (PNG)
- CSV logging with timestamps
- Save / Load / Reset settings (JSON)
- Multi-camera support with friendly device names
- Adjustable resolution and target FPS

## Screenshot

<p align="center">
  <img src="image/gui.PNG" width="700">
</p>

## Requirements

- Python 3.8+
- opencv-python
- numpy
- Pillow
- tkinter
