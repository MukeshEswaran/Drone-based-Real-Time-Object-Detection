# Drone-based-Real-Time-Object-Detection

This project implements a real-time drone surveillance system that detects, tracks, and follows objects (e.g., cars) using a combination of YOLOv8, DeepSORT, and an ONNX-based re-identification (ReID) model. It allows users to select a target object from the screen, and the drone autonomously follows the selected object. This application is tailored for security, traffic monitoring, and urban surveillance use cases.

---

## 🚀 Features

- Real-time object detection with **YOLOv8**
- Multi-object tracking using **DeepSORT**
- Re-identification via **ONNX model**
- User-interactive target selection via **OpenCV GUI**
- Follows selected object using drone positioning logic
- Trained and evaluated on the **VisDrone dataset**

---

## 🧠 Technologies Used

- [Ultralytics YOLOv8](https://docs.ultralytics.com)
- [DeepSORT Tracking](https://github.com/nwojke/deep_sort)
- [ONNX Runtime](https://onnxruntime.ai)
- [OpenCV](https://opencv.org/)
- Python, PyTorch

---

## 📁 Project Structure

```bash
📦drone-object-tracker
 ┣ 📂models/
 ┃ ┣ yolov8_weights.pt
 ┃ ┗ reid_model.onnx
 ┣ 📂data/
 ┃ ┣ visdrone/
 ┃ ┗ ...
 ┣ 📂scripts/
 ┃ ┣ detect_and_track.py
 ┃ ┣ yolo_inference.py
 ┃ ┗ deep_sort_tracker.py
 ┣ main.py
 ┣ README.md
 ┗ requirements.txt
