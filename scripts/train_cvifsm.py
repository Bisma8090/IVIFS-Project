import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------- Basic setup ----------
visible_folder = "../datasets/visible"
infrared_folder = "../datasets/infrared"
output_folder = "../outputs/fused_stage1"
os.makedirs(output_folder, exist_ok=True)

# ---------- Helper Functions ----------
def load_images(vis_path, ir_path):
    vis = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    return vis, ir

def simple_fusion(vis, ir, alpha=0.5):
    return cv2.addWeighted(vis, alpha, ir, 1 - alpha, 0)

def dummy_segmentation(fused):
    # simple threshold based segmentation
    _, seg = cv2.threshold(fused, 0, 255, cv2.THRESH_OTSU)
    return seg

def compute_loss(vis, ir, fused):
    # simulate reconstruction-like loss
    return np.mean(np.abs(fused - vis)) + np.mean(np.abs(fused - ir))

# ---------- Training Loop (simulated) ----------
EPOCHS = 3   # keep small; this is just a CPU-demo
alpha = 0.5

for epoch in range(EPOCHS):
    print(f"\n🌀 Epoch {epoch+1}/{EPOCHS}")
    total_loss = 0

    for filename in tqdm(os.listdir(visible_folder)):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        vis_path = os.path.join(visible_folder, filename)
        ir_path = os.path.join(infrared_folder, f"ir_{filename}")
        if not os.path.exists(ir_path):
            print(f" Missing IR for {filename}")
            continue

        vis, ir = load_images(vis_path, ir_path)
        fused = simple_fusion(vis, ir, alpha)
        seg = dummy_segmentation(fused)

        loss = compute_loss(vis, ir, fused)
        total_loss += loss

        # save example
        save_path = os.path.join(output_folder, f"epoch{epoch+1}_{filename}")
        cv2.imwrite(save_path, fused)

    avg_loss = total_loss / (len(os.listdir(visible_folder)) + 1)
    print(f"Average Loss: {avg_loss:.4f}")

print(" CVIFSM training (simulated) completed!")

# ---------- Visualization ----------
sample = cv2.imread(os.path.join(output_folder, os.listdir(output_folder)[0]), 0)
seg = dummy_segmentation(sample)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(sample, cmap='gray'); plt.title("Sample Fused Image"); plt.axis('off')
plt.subplot(1,2,2); plt.imshow(seg, cmap='gray'); plt.title("Dummy Segmentation Map"); plt.axis('off')
plt.show()
