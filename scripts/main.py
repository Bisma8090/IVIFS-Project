import cv2
import numpy as np
import os
from tqdm import tqdm

# Function: Visible aur Infrared images load karta hai
def load_images(vis_path, ir_path):
    vis = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    return vis, ir

# Function: Simple weighted fusion (visible + infrared)
def simple_fusion(vis, ir, alpha=0.5):
    fused = cv2.addWeighted(vis, alpha, ir, 1 - alpha, 0)
    return fused

if __name__ == "__main__":
    visible_folder = "../datasets/visible"
    infrared_folder = "../datasets/infrared"
    output_folder = "../datasets/fused"

    os.makedirs(output_folder, exist_ok=True)

    for filename in tqdm(os.listdir(visible_folder)):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            vis_path = os.path.join(visible_folder, filename)
            ir_path = os.path.join(infrared_folder, f"ir_{filename}")

            if not os.path.exists(ir_path):
                print(f" Missing infrared file for {filename}")
                continue

            vis, ir = load_images(vis_path, ir_path)

            vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX)
            ir  = cv2.normalize(ir,  None, 0, 255, cv2.NORM_MINMAX)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            vis = clahe.apply(vis)
            ir  = clahe.apply(ir)

            alpha = 0.7  # (Try also 0.3 or 0.5 for experiments)
            fused = simple_fusion(vis, ir, alpha=alpha)

            save_path = os.path.join(output_folder, f"fused_{filename}")
            cv2.imwrite(save_path, fused)

    print(" All images fused successfully!")
