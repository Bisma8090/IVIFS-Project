import os
import torch
from PIL import Image
import torchvision.transforms as T

from src.models.cvifsm import CVIFSM


def load_pair(vis_path, ir_path, img_size=(480,640)):
    to_tensor = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
    ])
    vis = Image.open(vis_path).convert("RGB")
    ir  = Image.open(ir_path).convert("L")

    vis_t = to_tensor(vis).unsqueeze(0)  # [1,3,H,W]
    ir_t  = to_tensor(ir).unsqueeze(0)   # [1,1,H,W]

    return vis_t, ir_t


def run_alpha_sweep(vis_path, ir_path, ckpt_path, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    print("Loading model...")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # FUSION-ONLY Model (no num_classes argument)
    model = CVIFSM().cpu()

    # Load weights
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    vis, ir = load_pair(vis_path, ir_path)

    alpha_values = [i/10 for i in range(11)]  # 0.0 → 1.0

    print("Running alpha sweep...")
    for a in alpha_values:
        with torch.no_grad():
            fused = model(vis, ir, alpha=a, mask=None)   # FIXED

        fused_img = fused[0].permute(1,2,0).numpy()
        fused_img = (fused_img * 255).clip(0,255).astype("uint8")
        fused_img = Image.fromarray(fused_img)

        save_path = os.path.join(save_dir, f"alpha_{a:.1f}.png")
        fused_img.save(save_path)

        print(f"Saved: {save_path}")

    print("Done!")


if __name__ == "__main__":
    vis = "data/custom_dataset/TEST/Visible/260523.jpg"
    ir  = "data/custom_dataset/TEST/Infrared/260523.jpg"

    ckpt = "experiments/outputs/run1/model_cvifsm.pth"
    out = "experiments/outputs/run1/alpha_sweep"

    run_alpha_sweep(vis, ir, ckpt, out)
