from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Load video file
video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Save output video
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{model.names[cls]} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write frame to output file
    out.write(frame)

    # Show the video frame with detections
    cv2.imshow("YOLOv8 Video Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
