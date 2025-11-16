# src/text_fusion.py

import torch
import argparse
import os
from PIL import Image
from torchvision import transforms

from src.models.cvifsm import CVIFSM
from src.models.ifam import SimpleIFAM
from src.utils_tokenizer import SimpleTokenizer


# ------------------------------------------------------------
# Load image as tensor
# ------------------------------------------------------------
def load_image(path):
    img = Image.open(path).convert("RGB")
    t = transforms.Compose([
        transforms.ToTensor()
    ])
    return t(img).unsqueeze(0)   # [1,3,H,W]


def load_ir(path):
    img = Image.open(path).convert("L")
    t = transforms.Compose([
        transforms.ToTensor()
    ])
    return t(img).unsqueeze(0)   # [1,1,H,W]


# ------------------------------------------------------------
# MAIN TEXT → FUSION PIPELINE
# ------------------------------------------------------------
def run_text_fusion(args):

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading CVIFSM:", args.ckpt_cvifsm)
    model = CVIFSM(base_ch=32, att_enable=True, use_mask=False).to(device)

    ckpt = torch.load(args.ckpt_cvifsm, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("Loading IFAM:", args.ckpt_ifam)
    tokenizer = SimpleTokenizer()
    ifam = SimpleIFAM().to(device)
    ifam.load_state_dict(torch.load(args.ckpt_ifam, map_location=device))
    ifam.eval()

    # Load inputs
    vi = load_image(args.vis).to(device)
    ir = load_ir(args.ir).to(device)

    # Encode text → α
    text_ids = tokenizer.encode(args.prompt)
    text_ids = text_ids.to(device)

    predicted_alpha = ifam(text_ids).item()
    print(f"\nPrompt: {args.prompt}")
    print(f"Predicted α = {predicted_alpha:.3f}")

    # Fusion
    with torch.no_grad():
      fused = model(vi, ir, alpha=predicted_alpha, mask=None)

    # Save output
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "fused_text_controlled.png")

    save_img = transforms.ToPILImage()(fused.squeeze(0).cpu())
    save_img.save(out_path)

    print("Saved:", out_path)


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--vis", type=str, required=True)
    parser.add_argument("--ir", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)

    parser.add_argument("--ckpt_cvifsm", type=str, default="experiments/outputs/run1/model_cvifsm.pth")
    parser.add_argument("--ckpt_ifam", type=str, default="experiments/outputs/ifam_model.pth")
    parser.add_argument("--output", type=str, default="experiments/outputs/text_fusion")

    args = parser.parse_args()
    run_text_fusion(args)
