import cv2
import os

# Input aur output folders ke path
visible_folder = "../datasets/visible"
infrared_folder = "../datasets/infrared"

os.makedirs(infrared_folder, exist_ok=True)

for filename in os.listdir(visible_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(visible_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Image not loaded: {filename}")
            continue

        # Step 1: Convert to grayscale (IR-like)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 2: Contrast enhancement
        gray = cv2.equalizeHist(gray)

        # Step 3: Invert colors (white-hot effect)
        ir_like = cv2.bitwise_not(gray)

        # Step 4: Save grayscale IR-style image
        save_path = os.path.join(infrared_folder, f"ir_{filename}")
        cv2.imwrite(save_path, ir_like)
        print(f" Converted: {filename} → {save_path}")

print(" All visible images converted to infrared-style")
