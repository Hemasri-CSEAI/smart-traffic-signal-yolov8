# AI-Powered Smart Traffic Signal Control System

An adaptive traffic signal control system built using OpenCV and Object Detection (YOLOv8). The application analyzes real-time video feeds to count vehicles dynamically and adjusts green signal duration based on traffic density.

## Key Features
- **Real-Time Vehicle Detection:** Utilizes YOLOv8 trained model targeting cars, buses, trucks, and motorcycles (`classes=[2, 3, 5, 7]`).
- **Dynamic Signal Allocation:** Dynamically calculates green light durations based on live vehicle density.
- **Interactive Dashboard:** Overlay visualization built with OpenCV showing remaining signal time, current lane status, and live density metrics.

## Tech Stack
- **Language:** Python
- **Computer Vision:** OpenCV (`cv2`)
- **Object Detection:** Ultralytics YOLOv8
- **GUI & File Management:** Tkinter

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Hemasri-CSEAI/smart-traffic-signal-yolov8.git
   
  2. Install dependencies:
   ```bash
   pip install opencv-python ultralytics
```
   ```bash
   python main.py
```
