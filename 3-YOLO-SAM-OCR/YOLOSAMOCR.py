import cv2
import numpy as np
import torch
import os
import pytesseract
import pandas as pd
from glob import glob
from PIL import Image
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from skimage import measure
import re

# ==== CONFIGURATION ====
model_path = "runs/detect/train/weights/best.pt"
test_folder = "test_images"
output_dir = "output"
detections_dir = os.path.join(output_dir, "detections")
visualizations_dir = os.path.join(output_dir, "visualizations")
detect_segment_dir = os.path.join(output_dir, "detect_segment")
conf_threshold = 0.4
iou_threshold = 0.4
img_size = 640
pavement_top = 172
pavement_height = 2485 - pavement_top
total_pavement_pixels = 2313 * 2313

HEADER_WIDTH = 2313
HEADER_HEIGHT = 172
LON_BOX = (0, 74, 182, 98)
LAT_BOX = (759, 74, 941, 98)

YOLO_COLORS = {"P_Crack": (0, 0, 255), "L_Crack": (0, 255, 0), "T_Crack": (255, 0, 0)}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM_CHECKPOINT = "segment-anything/pavesam_vit_h.pth"
CLASS_MAPPING = {"P_Crack": 1, "L_Crack": 2, "T_Crack": 3}
CLASS_COLORS = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0)}

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(detections_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)
os.makedirs(detect_segment_dir, exist_ok=True)

# ==== SETUP TESSERACT ====
tesseract_path = os.getenv('TESSERACT_PATH', r'C:\Program Files\Tesseract-OCR\tesseract.exe')
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    print("Warning: Tesseract not found at specified path")

# ==== LOAD MODELS ====
print("Loading YOLO model...")
model = YOLO(model_path)
class_names = model.names
print(f"✅ Loaded YOLO model with {len(class_names)} classes")

print("Loading SAM model...")
sam = sam_model_registry["vit_h"]()
checkpoint = torch.load(SAM_CHECKPOINT, map_location=DEVICE)
sam.load_state_dict(checkpoint)
sam.to(DEVICE)
predictor = SamPredictor(sam)


# ==== CUSTOM NMS ====
def custom_nms(boxes, scores, iou_threshold=0.4):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        if len(idxs) == 1:
            break
        xx1 = np.maximum(boxes[current][0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[current][1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[current][2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[current][3], boxes[idxs[1:], 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        area_current = (boxes[current][2] - boxes[current][0]) * (boxes[current][3] - boxes[current][1])
        area_others = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
        union = area_current + area_others - intersection
        iou = intersection / (union + 1e-6)
        idxs = idxs[1:][iou < iou_threshold]
    return keep


# ==== SAM UTILS ====
def expand_box(box, img_w, img_h, pct=0.7):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    dx = int(w * pct)
    dy = int(h * pct)
    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(img_w, x2 + dx)
    ny2 = min(img_h, y2 + dy)
    return [nx1, ny1, nx2, ny2]


def get_connected_components(binary_mask, min_area=300):
    labeled = measure.label(binary_mask, connectivity=2)
    props = measure.regionprops(labeled)
    boxes = []
    for p in props:
        if p.area < min_area:
            continue
        y1, x1, y2, x2 = p.bbox
        boxes.append([x1, y1, x2, y2])
    return boxes


def refine_with_sam(predictor, image, coarse_mask, expansion_pct=0.7):
    h, w = image.shape[:2]
    refined_mask = np.zeros_like(coarse_mask, dtype=np.uint8)
    for class_id in np.unique(coarse_mask):
        if class_id == 0:
            continue
        class_mask = (coarse_mask == class_id).astype(np.uint8)
        boxes = get_connected_components(class_mask)
        for box in boxes:
            exp_box = expand_box(box, w, h, pct=expansion_pct)
            x1, y1, x2, y2 = exp_box
            patch = image[y1:y2, x1:x2]
            predictor.set_image(patch)
            tx1, ty1, tx2, ty2 = box
            prompt_box = [tx1 - x1, ty1 - y1, tx2 - x1, ty2 - y1]
            masks, _, _ = predictor.predict(box=np.array(prompt_box), multimask_output=False)
            refined_patch_mask = masks[0].astype(np.uint8)
            refined_mask[y1:y2, x1:x2][refined_patch_mask == 1] = class_id
    return refined_mask


# ==== OCR ====
def extract_lat_lon(image_path):
    image = Image.open(image_path)
    header = image.crop((0, 0, HEADER_WIDTH, HEADER_HEIGHT))
    lon_crop = header.crop(LON_BOX)
    lat_crop = header.crop(LAT_BOX)

    lon_text = pytesseract.image_to_string(lon_crop, config='--psm 7').strip()
    lat_text = pytesseract.image_to_string(lat_crop, config='--psm 7').strip()

    lon_match = re.search(r"[-+]?[0-9]*\.?[0-9]+", lon_text)
    lat_match = re.search(r"[-+]?[0-9]*\.?[0-9]+", lat_text)

    lon_value = lon_match.group(0) if lon_match else ''
    lat_value = lat_match.group(0) if lat_match else ''
    return lat_value, lon_value


# ==== DRAW YOLO DETECTIONS ====
def draw_yolo_detections(image, boxes, scores, classes, class_names):
    """Draw YOLO bounding boxes with labels and confidence scores"""
    det_image = image.copy()
    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        x1, y1, x2, y2 = map(int, box)
        label = class_names[cls]
        color = YOLO_COLORS.get(label, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(det_image, (x1, y1), (x2, y2), color, 4)

        # Create label text
        label_text = f"{label}: {score:.2f}"

        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
        cv2.rectangle(det_image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)

        # Draw label text
        cv2.putText(det_image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

    return det_image


# ==== COMBINE DETECTION AND SEGMENTATION ====
def combine_detection_segmentation(detection_img, segmentation_img):
    """Combine YOLO detection image with SAM segmentation image"""
    if segmentation_img is None:
        return detection_img

    # Blend the two images (50% detection, 50% segmentation)
    combined_img = cv2.addWeighted(detection_img, 0.5, segmentation_img, 0.5, 0)
    return combined_img

# ==== PROCESS IMAGE ====
def process_image_with_yolo_sam(img_path):
    original_image = cv2.imread(img_path)
    if original_image is None:
        return None, None, {}, ('', '')

    lat, lon = extract_lat_lon(img_path)

    pavement_img = original_image[pavement_top:, :]
    rgb_image = cv2.cvtColor(pavement_img, cv2.COLOR_BGR2RGB)
    processed_image = rgb_image.copy()
    h, w = processed_image.shape[:2]

    # YOLO detection
    results = model.predict(pavement_img, imgsz=img_size, conf=conf_threshold, iou=0.6, verbose=False)
    boxes, scores, classes = [], [], []
    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())
        boxes.append(xyxy)
        scores.append(conf)
        classes.append(cls)

    # Create YOLO detection image
    yolo_detection_img = draw_yolo_detections(pavement_img, boxes, scores, classes, class_names)

    keep_idxs = custom_nms(boxes, scores, iou_threshold=iou_threshold)
    if not keep_idxs:
        return yolo_detection_img, None, {}, (lat, lon)

    # SAM segmentation
    predictor.set_image(processed_image)
    coarse_mask = np.zeros((h, w), dtype=np.uint8)
    for i in keep_idxs:
        xyxy = boxes[i]
        cls = classes[i]
        label = class_names[cls]
        if label not in CLASS_MAPPING:
            continue
        class_id = CLASS_MAPPING[label]
        masks, _, _ = predictor.predict(box=xyxy, multimask_output=False)
        coarse_mask[masks[0]] = class_id

    refined_mask = refine_with_sam(predictor, processed_image, coarse_mask)

    # Create segmentation visualization
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_value, color in CLASS_COLORS.items():
        color_mask[refined_mask == class_value] = color

    color_mask_resized = cv2.resize(color_mask, (pavement_img.shape[1], pavement_img.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
    segmentation_img = cv2.addWeighted(pavement_img, 0.5, color_mask_resized, 0.5, 0)

    # Calculate statistics
    total_pixels = np.count_nonzero(refined_mask)
    p_pixels = np.count_nonzero(refined_mask == 1)
    l_pixels = np.count_nonzero(refined_mask == 2)
    t_pixels = np.count_nonzero(refined_mask == 3)

    stats = {
        "total_cracks": total_pixels,
        "p_cracks": p_pixels,
        "l_cracks": l_pixels,
        "t_cracks": t_pixels,
        "total_cracks_ratio": total_pixels / total_pavement_pixels,
        "p_cracks_ratio": p_pixels / total_pavement_pixels,
        "l_cracks_ratio": l_pixels / total_pavement_pixels,
        "t_cracks_ratio": t_pixels / total_pavement_pixels,
        "latitude": lat,
        "longitude": lon
    }

    return yolo_detection_img, segmentation_img, stats, (lat, lon)


# ==== MAIN EXECUTION ====
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob(os.path.join(test_folder, ext)))

csv_data = []
for i, img_path in enumerate(image_paths):
    print(f"Processing {i + 1}/{len(image_paths)}: {os.path.basename(img_path)}")

    try:
        yolo_img, seg_img, stats, (lat, lon) = process_image_with_yolo_sam(img_path)

        if not stats:
            print(f"No detections found in {os.path.basename(img_path)}")
            continue

        base_name = os.path.basename(img_path)
        name_wo_ext = os.path.splitext(base_name)[0]

        # Save YOLO detection image
        cv2.imwrite(os.path.join(detections_dir, f"{name_wo_ext}_detection.png"), yolo_img)

        # Save segmentation visualization
        if seg_img is not None:
            cv2.imwrite(os.path.join(visualizations_dir, f"{name_wo_ext}_segmentation.png"), seg_img)

            # NEW: Save combined detection + segmentation image
            combined_img = combine_detection_segmentation(yolo_img, seg_img)
            cv2.imwrite(os.path.join(detect_segment_dir, f"{name_wo_ext}_combined.png"), combined_img)

        csv_data.append({"filename": base_name, **stats})

        # Clean up memory
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        continue

csv_path = os.path.join(output_dir, "crack_summary_with_latlon.csv")
df = pd.DataFrame(csv_data)
df.to_csv(csv_path, index=False)
print(f"\n🎉 All images processed successfully!")
print(f"📊 CSV saved at: {csv_path}")
print(f"📁 YOLO detections saved in: {detections_dir}")
print(f"🎨 Segmentations saved in: {visualizations_dir}")
print(f"🔗 Combined detections+segmentations saved in: {detect_segment_dir}")  # NEW MESSAGE