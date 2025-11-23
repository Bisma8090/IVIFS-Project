# generate_masks.py
import os
import cv2
import numpy as np
from glob import glob

# adjust these paths if your dataset path is different
IR_DIR = "data/custom_dataset/TRAIN/Infrared"
MASK_DIR = "data/custom_dataset/MASKS"
os.makedirs(MASK_DIR, exist_ok=True)

# parameters (tune if needed)
THRESH = None         # if None we will use Otsu automatic thresholding
GAUSSIAN_KERNEL = (5,5)
SOBEL_KERNEL = 3
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

def make_mask(ir_img):
    # ir_img is single channel numpy array, uint8
    # denoise a bit
    img = cv2.GaussianBlur(ir_img, GAUSSIAN_KERNEL, 0)
    # threshold: Otsu if THRESH None
    if THRESH is None:
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, th = cv2.threshold(img, THRESH, 255, cv2.THRESH_BINARY)
    # edges
    gradx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=SOBEL_KERNEL)
    grady = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=SOBEL_KERNEL)
    grad = cv2.magnitude(gradx.astype(np.float32), grady.astype(np.float32))
    grad = np.uint8(np.clip((grad / (grad.max()+1e-9)) * 255, 0, 255))
    _, grad_th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # combine threshold and edges
    combined = cv2.bitwise_or(th, grad_th)
    # morphological open/close to clean
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
    # final smoothing and ensure binary
    _, final = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)
    return final

def main():
    ir_files = sorted(glob(os.path.join(IR_DIR, "*.*")))
    if len(ir_files) == 0:
        print("No infrared images found in", IR_DIR)
        return
    for p in ir_files:
        fname = os.path.basename(p)
        ir = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if ir is None:
            print("Could not read", p)
            continue
        if len(ir.shape) == 3:
            # if color, convert to gray
            ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        mask = make_mask(ir)
        outp = os.path.join(MASK_DIR, fname)
        cv2.imwrite(outp, mask)
        print("Saved mask:", outp)

if __name__ == "__main__":
    main()
