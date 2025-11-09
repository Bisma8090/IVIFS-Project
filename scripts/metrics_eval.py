import cv2
import os
import numpy as np
from tqdm import tqdm

def calc_sd(img):
    return np.std(img)

def calc_entropy(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 255))
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def calc_cc(img1, img2):
    return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]

def calc_miou(pred, gt):
    pred_bin = (pred > 127).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return intersection / union if union != 0 else 0

# ---- Paths ----
visible_folder = "../datasets/visible"
fused_folder = "../datasets/fused"
seg_folder = "../outputs/segmentation_maps"
gt_folder = "../datasets/segmentation_gt"

# ---- Lists to store metrics ----
sd_list, en_list, cc_list, miou_list = [], [], [], []

# ---- Main Evaluation Loop ----
for fname in tqdm(os.listdir(fused_folder)):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    fused_path = os.path.join(fused_folder, fname)
    fused = cv2.imread(fused_path, cv2.IMREAD_GRAYSCALE)

    # ✅ Fix: remove 'fused_' only
    vis_name = fname.replace("fused_", "")
    vis_path = os.path.join(visible_folder, vis_name)
    vis = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)

    sd_list.append(calc_sd(fused))
    en_list.append(calc_entropy(fused))

    if vis is not None:
        cc_list.append(calc_cc(fused, vis))
    else:
        print(f"[WARN] Missing visible image for {fname}")

    # ---- Optional: segmentation ----
    seg_path = os.path.join(seg_folder, fname)
    gt_path = os.path.join(gt_folder, fname)

    if os.path.exists(seg_path) and os.path.exists(gt_path):
        seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        miou_list.append(calc_miou(seg, gt))

# ---- Print Results ----
print("\n📊 Evaluation Results:")
print(f"• Average SD   : {np.mean(sd_list):.4f}")
print(f"• Average EN   : {np.mean(en_list):.4f}")
print(f"• Average CC   : {np.mean(cc_list):.4f}" if cc_list else "• Average CC   : N/A (no visible images matched)")
print(f"• Average mIoU : {np.mean(miou_list):.4f}" if miou_list else "• mIoU skipped (no GT segmentation maps found)")
