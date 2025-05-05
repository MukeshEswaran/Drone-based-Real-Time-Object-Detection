import os

# Define paths
visdrone_root = "VisDrone"
annotations_folder = os.path.join(visdrone_root, "annotations")
labels_folder = os.path.join(visdrone_root, "labels")

# Ensure the labels folder exists
os.makedirs(labels_folder, exist_ok=True)
for subset in ["train", "val", "test"]:
    os.makedirs(os.path.join(labels_folder, subset), exist_ok=True)

# VisDrone class mapping (excluding "ignored regions" and "other" category)
class_mapping = {
    0: "ignored", 1: "pedestrian", 2: "people", 3: "bicycle",
    4: "car", 5: "van", 6: "truck", 7: "tricycle",
    8: "awning-tricycle", 9: "bus", 10: "motor",
    11: "others"
}
valid_classes = list(range(1, 11))  # Only keep classes 1-10 for YOLO

# Image dimensions (VisDrone images are usually 1920x1080 but verify)
image_w, image_h = 1920, 1080  # Adjust if needed

# Convert each annotation file
for subset in ["train", "val", "test"]:
    ann_path = os.path.join(annotations_folder, subset)
    lbl_path = os.path.join(labels_folder, subset)
    
    for file in os.listdir(ann_path):
        if not file.endswith(".txt"):
            continue
        
        with open(os.path.join(ann_path, file), "r") as f:
            lines = f.readlines()
        
        yolo_annotations = []
        for line in lines:
            data = line.strip().split(",")  # VisDrone uses comma-separated values
            x, y, w, h = map(int, data[:4])  # Bounding box
            class_id = int(data[5])  # Object class

            if class_id not in valid_classes:
                continue  # Skip unwanted classes

            # Convert to YOLO format (normalized)
            x_center = (x + w / 2) / image_w
            y_center = (y + h / 2) / image_h
            w /= image_w
            h /= image_h

            # Save in YOLO format
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        # Save converted annotations in labels folder
        with open(os.path.join(lbl_path, file), "w") as f:
            f.writelines(yolo_annotations)

print("Conversion complete! YOLO labels are saved in the 'labels/' folder.")
