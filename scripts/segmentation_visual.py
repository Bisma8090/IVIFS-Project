import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

input_folder = "../datasets/fused"
output_folder = "../outputs/segmentation_stage3"
os.makedirs(output_folder, exist_ok=True)

# List all fused images
fused_images = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

if not fused_images:
    raise FileNotFoundError(" No fused images found in ../datasets/fused")

# Loop through all fused images
for fname in fused_images:
    input_path = os.path.join(input_folder, fname)
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"= Skipping unreadable file: {fname}")
        continue

    # Step 1: Simple threshold segmentation
    _, mask = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

    # Step 2: Region enhancement (apply mask)
    enhanced = cv2.bitwise_and(img, mask)

    # Step 3: Display preview (optional)
    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1); plt.imshow(img, cmap='gray'); plt.title("Fused"); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(mask, cmap='gray'); plt.title("Mask"); plt.axis('off')
    plt.subplot(1,3,3); plt.imshow(enhanced, cmap='gray'); plt.title("Enhanced"); plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Step 4: Save output
    save_path = os.path.join(output_folder, f"segmented_{fname}")
    cv2.imwrite(save_path, enhanced)
    print(f" Saved segmented result: {save_path}")

print(" All fused images processed successfully!")
