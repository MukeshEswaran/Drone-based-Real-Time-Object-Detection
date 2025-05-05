import os
import cv2
import shutil

# Define paths (Update these paths)
visdrone_root = r"C:\Users\mukie\Desktop\Computer_Vision_Drone\VisDrone_dataset"
yolo_root = r"C:\Users\mukie\Desktop\Computer_Vision_Drone\yolo_dataset"

# Define dataset splits
datasets = {
    "train": os.path.join(visdrone_root, "trainset"),
    "val": os.path.join(visdrone_root, "valset"),
}

# Create YOLO dataset directories
for split in ["train", "val"]:
    os.makedirs(os.path.join(yolo_root, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(yolo_root, split, "labels"), exist_ok=True)

# Step 1: Match annotation files with corresponding sequence folders
def get_sequence_mapping(visdrone_path):
    ann_folder = os.path.join(visdrone_path, "annotations")
    seq_folder = os.path.join(visdrone_path, "sequences")

    annotation_files = os.listdir(ann_folder)
    sequence_folders = os.listdir(seq_folder)

    mapping = {}  # Store correct matches

    for ann_file in annotation_files:
        ann_name = ann_file.replace(".txt", "")
        for seq_folder_name in sequence_folders:
            if ann_name in seq_folder_name:  # Match substring
                mapping[ann_name] = seq_folder_name
                break

    return mapping

# Step 2: Convert Dataset Using Mapping
def convert_annotations(visdrone_path, yolo_img_path, yolo_label_path):
    ann_folder = os.path.join(visdrone_path, "annotations")
    seq_folder = os.path.join(visdrone_path, "sequences")
    
    mapping = get_sequence_mapping(visdrone_path)

    for ann_name, seq_folder_name in mapping.items():
        seq_path = os.path.join(seq_folder, seq_folder_name)
        ann_path = os.path.join(ann_folder, f"{ann_name}.txt")

        print(f"Processing: {ann_name} -> {seq_folder_name}")

        if not os.path.exists(ann_path):
            print(f"Skipping {ann_name}: Annotation file not found!")
            continue

        # Read annotation file
        with open(ann_path, "r") as f:
            lines = f.readlines()

        img_annotations = {}

        # Step 3: Detect if images are 6-digit (000001.jpg) or 7-digit (0000001.jpg)
        sample_img = sorted(os.listdir(seq_path))[0]  # Get first image filename
        digit_length = len(sample_img.split('.')[0])  # Check filename length
        print(f"Detected {digit_length}-digit image format in {seq_folder_name}")

        for line in lines:
            data = line.strip().split(",")
            if len(data) < 8:
                continue

            frame_index = int(data[0])  # Frame number
            x, y, box_w, box_h = map(int, data[2:6])  # Bounding box
            class_id = int(data[7]) - 1  # Convert to zero-based index

            # Generate correct image filename based on detected format
            img_name = f"{frame_index:0{digit_length}d}.jpg"
            img_path = os.path.join(seq_path, img_name)

            if not os.path.exists(img_path):
                print(f"Warning: Image {img_name} not found in {seq_folder_name}")
                continue

            # Read image to get dimensions
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Could not read {img_path}")
                continue
            h, w, _ = img.shape

            # Normalize bounding box for YOLO format
            x_center = (x + box_w / 2) / w
            y_center = (y + box_h / 2) / h
            box_w = box_w / w
            box_h = box_h / h

            if img_name not in img_annotations:
                img_annotations[img_name] = []

            img_annotations[img_name].append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

        # Process images
        for img_file in os.listdir(seq_path):
            img_path = os.path.join(seq_path, img_file)
            new_img_path = os.path.join(yolo_img_path, f"{seq_folder_name}_{img_file}")

            shutil.copy(img_path, new_img_path)

            if img_file in img_annotations:
                new_label_file = os.path.join(yolo_label_path, f"{seq_folder_name}_{img_file.replace('.jpg', '.txt')}")
                with open(new_label_file, "w") as label_file:
                    label_file.writelines(img_annotations[img_file])
                    print(f"Saved label: {new_label_file}")

# Convert train and validation datasets
convert_annotations(datasets["train"], os.path.join(yolo_root, "train/images"), os.path.join(yolo_root, "train/labels"))
convert_annotations(datasets["val"], os.path.join(yolo_root, "val/images"), os.path.join(yolo_root, "val/labels"))

print("Dataset conversion complete!")
