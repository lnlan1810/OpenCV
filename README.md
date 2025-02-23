# OpenCV Face and Hand Gesture Recognition

## Overview
This project implements a facial recognition system with hand gesture detection using OpenCV and MediaPipe. It includes functionalities to create a dataset, train a model, and test it in real-time with a webcam.

## Features
- Face detection using OpenCV's Haar cascades.
- Hand gesture recognition using MediaPipe.
- Training a face recognition model with LBPH (Local Binary Pattern Histogram).
- Real-time face and hand gesture detection.

## Installation
### Prerequisites
Ensure you have Python installed along with the following dependencies:

```bash
pip install opencv-python numpy mediapipe pillow
```

## Usage

### 1. Create Dataset
To capture face images and create a dataset, uncomment the `create_dataset()` function in `opencv.py` and run:
```bash
python opencv.py
```
Press `q` to stop capturing after collecting sufficient images.

### 2. Train Model
Train the face recognition model by running:
```bash
python opencv.py
```
This will generate a `model.yml` file containing the trained model.

### 3. Test Model
Test real-time face recognition with hand gesture-based identification by running:
```bash
python opencv.py
```
Press `q` to exit the test mode.

## File Structure
```
OpenCV/
│── faceDataset/       # Folder containing captured face images
│── model.yml          # Trained face recognition model
│── opencv.py          # Main script for dataset creation, training, and testing
│── README.md          # Documentation
```

## Dependencies
- Python
- OpenCV
- NumPy
- Pillow
- MediaPipe


