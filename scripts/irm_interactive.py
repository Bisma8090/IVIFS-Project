import cv2
import numpy as np
import os

visible_folder = "../datasets/visible"
infrared_folder = "../datasets/infrared"
output_folder = "../outputs/fused_stage2"
os.makedirs(output_folder, exist_ok=True)

def load_images(vis_path, ir_path):
    vis = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    return vis, ir

def simple_fusion(vis, ir, alpha):
    return cv2.addWeighted(vis, alpha, ir, 1 - alpha, 0)

# ---------- Text-based adjustment ----------
def interpret_text(text):
    text = text.lower()
    if "infrared" in text or "heat" in text:
        return 0.2   
    elif "visible" in text or "details" in text:
        return 0.8  
    elif "balanced" in text:
        return 0.5
    else:
        return 0.5   

# ---------- Interactive loop ----------
if __name__ == "__main__":
    print("🗣 Interactive Fusion Mode (IRM Simulation)")
    print("Try commands like:")
    print("  - 'more infrared focus'")
    print("  - 'increase visible details'")
    print("  - 'balanced fusion'\n")

    text = input("Enter your command: ")
    alpha = interpret_text(text)
    print(f" Using alpha = {alpha:.2f}")

    sample_vis = os.listdir(visible_folder)[0]
    vis_path = os.path.join(visible_folder, sample_vis)
    ir_path = os.path.join(infrared_folder, f"ir_{sample_vis}")

    vis, ir = load_images(vis_path, ir_path)
    fused = simple_fusion(vis, ir, alpha)

    save_path = os.path.join(output_folder, f"fused_{alpha:.2f}_{sample_vis}")
    cv2.imwrite(save_path, fused)
    print(f" Saved: {save_path}")

    cv2.imshow("Fused Image", fused)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
