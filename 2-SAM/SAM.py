import os
import json
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM_CHECKPOINT = "segment-anything/sam_vit_h_4b8939.pth"
NEW_CHECKPOINT = "pavesam_vit_h.pth"
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)  # Create directory if it doesn't exist

# CSV file path for training metrics
METRICS_CSV = os.path.join(SAVE_DIR, "training_metrics.csv")

IMAGE_DIR = "data/images"
ANNOTATION_DIR = "data/annotations"
BATCH_SIZE = 1
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4

# ----------------------------------------

# ---- 1. Dataset ----
class PavementDataset(Dataset):
    def __init__(self, img_dir, ann_dir):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.img_names = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".png"))
        ]
        self.transform = ResizeLongestSide(1024)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        json_path = os.path.join(
            self.ann_dir, os.path.splitext(img_name)[0] + ".json"
        )

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        with open(json_path) as f:
            annotations = json.load(f)

        # Load ground truth mask from polygon or rectangle annotations
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for shape in annotations["shapes"]:
            if shape["shape_type"] == "rectangle":
                [[x1, y1], [x2, y2]] = shape["points"]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                mask[y1:y2, x1:x2] = 1

        # Resize image and mask
        image_resized = self.transform.apply_image(image)
        mask_resized = cv2.resize(mask, (image_resized.shape[1], image_resized.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        # Convert to tensors
        image_tensor = torch.as_tensor(image_resized, dtype=torch.float32).permute(2, 0, 1) / 255.
        mask_tensor = torch.as_tensor(mask_resized, dtype=torch.float32).unsqueeze(0)

        # Bounding box from mask
        ys, xs = np.where(mask_resized > 0)
        if len(xs) > 0 and len(ys) > 0:
            box = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], dtype=torch.float32)
        else:
            box = torch.tensor([0, 0, 1, 1], dtype=torch.float32)

        return image_tensor, mask_tensor, box

# ---- 2. Loss Functions ----
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))

bce_loss = nn.BCEWithLogitsLoss()

# ---- 3. Load SAM and Freeze Encoders ----
sam = sam_model_registry["vit_h"](checkpoint=None)
sam.load_state_dict(torch.load(SAM_CHECKPOINT, weights_only=True))
sam.to(DEVICE)

for param in sam.image_encoder.parameters():
    param.requires_grad = False
for param in sam.prompt_encoder.parameters():
    param.requires_grad = False

# Mask decoder will be trained
optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=LEARNING_RATE)

# ---- 4. DataLoader ----
dataset = PavementDataset(IMAGE_DIR, ANNOTATION_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---- 5. Training Loop with Metrics ----
def compute_metrics(preds, masks):
    preds_bin = (torch.sigmoid(preds) > 0.5).int()
    masks_bin = (masks > 0.5).int()

    tp = ((preds_bin == 1) & (masks_bin == 1)).sum().item()
    fp = ((preds_bin == 1) & (masks_bin == 0)).sum().item()
    fn = ((preds_bin == 0) & (masks_bin == 1)).sum().item()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    intersection = ((preds_bin == 1) & (masks_bin == 1)).sum().item()
    union = ((preds_bin == 1) | (masks_bin == 1)).sum().item()
    iou = intersection / (union + 1e-6)

    return precision, recall, f1, iou

# Initialize CSV file with header
with open(METRICS_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'Loss', 'Precision', 'Recall', 'F1_Score', 'IoU'])

sam.train()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    epoch_prec, epoch_rec, epoch_f1, epoch_iou = 0, 0, 0, 0
    num_batches = 0

    for images, masks, boxes in loader:
        images, masks, boxes = images.to(DEVICE), masks.to(DEVICE), boxes.to(DEVICE)

        # Encode image
        image_embeddings = sam.image_encoder(images)

        # Encode bounding box prompt
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=None,
            boxes=boxes.unsqueeze(1),  # shape: [B, 1, 4]
            masks=None,
        )

        # Decode mask
        low_res_masks, _ = sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale to original image size
        upscaled_masks = sam.postprocess_masks(
            low_res_masks,
            input_size=(images.shape[2], images.shape[3]),
            original_size=(images.shape[2], images.shape[3])
        )
        pred_mask = upscaled_masks.squeeze(1)  # [B, H, W]

        # Compute loss
        loss = bce_loss(pred_mask, masks.squeeze(1)) + dice_loss(pred_mask, masks.squeeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        epoch_loss += loss.item()

        # Track metrics
        prec, rec, f1, iou = compute_metrics(pred_mask, masks.squeeze(1))
        epoch_prec += prec
        epoch_rec += rec
        epoch_f1 += f1
        epoch_iou += iou
        num_batches += 1

    # Average metrics
    avg_loss = epoch_loss / num_batches
    avg_prec = epoch_prec / num_batches
    avg_rec = epoch_rec / num_batches
    avg_f1 = epoch_f1 / num_batches
    avg_iou = epoch_iou / num_batches

    # Print metrics
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"- Loss: {avg_loss:.4f} "
          f"- Precision: {avg_prec:.4f} "
          f"- Recall: {avg_rec:.4f} "
          f"- F1: {avg_f1:.4f} "
          f"- IoU: {avg_iou:.4f}")

    # Save metrics to CSV
    with open(METRICS_CSV, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch+1, avg_loss, avg_prec, avg_rec, avg_f1, avg_iou])

# ---- 6. Save Fine-Tuned Model ----
save_path = os.path.join(SAVE_DIR, NEW_CHECKPOINT)
torch.save(sam.state_dict(), save_path)
print(f"Fine-tuned checkpoint saved to {save_path}")
print(f"Training metrics saved to {METRICS_CSV}")