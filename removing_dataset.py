import os
import cv2

image_dir = "C:/Users/mukie/Desktop/Computer_Vision_Drone/yolo_dataset/train/images"

for img_file in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_file)
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Corrupt image found: {img_path}")
            os.remove(img_path)
    except:
        print(f"Error processing: {img_path}")
        os.remove(img_path)
