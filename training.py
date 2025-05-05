from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load YOLO model

# Train the model using the correct dataset path
model.train(data="C:/Users/mukie/Desktop/Compuuter_Vision_Drone/datasets/VisDrone/data.yaml", epochs=50, batch=16, imgsz=640)
