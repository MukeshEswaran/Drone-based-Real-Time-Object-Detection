import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load trained YOLOv8 model
model = YOLO(r"C:\Users\mukie\Desktop\Computer_Vision_Drone\yolov8n.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Load drone footage
video_path = "video1.mp4"
cap = cv2.VideoCapture(video_path)

# Video frame size
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter("output_tracking.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracking results
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    out.write(frame)
    cv2.imshow("YOLOv8 Object Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
