import os
import json
import cv2
import numpy as np
import random
import shutil
from glob import glob
from ultralytics import YOLO

# ==== CONFIGURATION ====
images_dir = "images"                # Folder containing original images
json_dir = "annotations"             # Folder with JSON annotations
output_base_dir = "yolo_dataset"     # Output YOLO dataset directory
train_split = 0.8                    # Train/validation split ratio
pretrained_model = "yolov8n.pt"      # Pretrained YOLOv8 model variant
epochs = 200                         # Training epochs
img_size = 640                       # YOLO input image size
random.seed(42)                      # Reproducibility
l_rate = 0.001                       # Learning rate

# Define class names
CLASS_NAMES = ["T_Crack", "L_Crack", "P_Crack"]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# ==== FUNCTIONS ====
def get_bbox_from_json(json_path):
    """
    Extract bounding boxes and class IDs from LabelMe JSON annotations.
    Supports rectangles and polygons.
    """
    with open(json_path) as f:
        data = json.load(f)

    boxes = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "").strip()
        if label not in CLASS_MAP:
            continue  # Skip unknown labels

        class_id = CLASS_MAP[label]

        if shape["shape_type"] == "rectangle":
            (x1, y1), (x2, y2) = shape["points"]
            boxes.append((class_id, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))

        elif shape["shape_type"] == "polygon":
            pts = np.array(shape["points"])
            xmin, ymin = np.min(pts, axis=0)
            xmax, ymax = np.max(pts, axis=0)
            boxes.append((class_id, xmin, ymin, xmax, ymax))

    return boxes


def save_yolo_label(txt_path, boxes, img_width, img_height):
    """
    Save bounding boxes in YOLO format (class_id x_center y_center width height).
    Creates an empty file even if there are no boxes.
    """
    with open(txt_path, "w") as f:
        for (class_id, xmin, ymin, xmax, ymax) in boxes:
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            bw = (xmax - xmin) / img_width
            bh = (ymax - ymin) / img_height

            # Skip invalid/tiny boxes
            if bw > 0.001 and bh > 0.001:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")


def prepare_yolo_dataset():
    """
    Convert images + JSON annotations into YOLOv8 dataset format.
    Splits into train/val folders and creates YOLO labels.
    """
    all_files = [f for f in glob(os.path.join(images_dir, "*")) if os.path.isfile(f)]
    random.shuffle(all_files)

    split_idx = int(len(all_files) * train_split)
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]

    for phase, files in [("train", train_files), ("val", val_files)]:
        img_out_dir = os.path.join(output_base_dir, "images", phase)
        lbl_out_dir = os.path.join(output_base_dir, "labels", phase)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)

        for img_path in files:
            filename = os.path.splitext(os.path.basename(img_path))[0]
            json_path = os.path.join(json_dir, filename + ".json")

            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Skipping unreadable image: {img_path}")
                continue
            h, w = img.shape[:2]

            # Get bounding boxes from JSON
            boxes = []
            if os.path.exists(json_path):
                boxes.extend(get_bbox_from_json(json_path))

            # Save YOLO label file
            txt_path = os.path.join(lbl_out_dir, filename + ".txt")
            if boxes:
                save_yolo_label(txt_path, boxes, w, h)

            # Copy image to YOLO dataset
            shutil.copy(img_path, os.path.join(img_out_dir, os.path.basename(img_path)))

    print(f"✅ YOLO dataset prepared: {len(train_files)} train, {len(val_files)} val images")


def create_data_yaml():
    """
    Generate YOLOv8-compatible data.yaml file.
    """
    data_yaml = os.path.join(output_base_dir, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"path: {os.path.abspath(output_base_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for idx, name in enumerate(CLASS_NAMES):
            f.write(f"  {idx}: {name}\n")
    return data_yaml


def train_yolo():
    """
    Train YOLOv8 model using prepared dataset.
    """
    data_yaml = create_data_yaml()
    model = YOLO(pretrained_model)
    results = model.train(data=data_yaml, epochs=epochs, imgsz=img_size, lr0=l_rate)
    print(f"✅ Training complete! Best model saved to: runs/detect/train/weights/best.pt")


# ==== MAIN EXECUTION ====
if __name__ == "__main__":
    prepare_yolo_dataset()   # Step 1: Create YOLO dataset
    train_yolo()            # Step 2: Train YOLOv8 model
