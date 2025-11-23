# src/inference.py  (fusion + segmentation)
import os
import argparse
import torch
from torchvision.utils import save_image
from src.datasets import VIFDataset
from src.models.cvifsm import CVIFSM
import numpy as np
from PIL import Image
import matplotlib.cm as cm  # For applying colormap

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = VIFDataset(os.path.join(args.data_root, "TEST"), mode='test', img_size=args.img_size)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    model = CVIFSM(base_ch=args.base_ch, att_enable=not args.disable_amim, use_mask=not args.disable_mask).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # handle multiple alphas
    alphas = [float(x) for x in args.alpha.split(',')] if isinstance(args.alpha, str) else [float(args.alpha)]
    out_root = args.save_dir
    ensure_dir(out_root)

    for a in alphas:
        sub = os.path.join(out_root, f"alpha_{a:.2f}")
        ensure_dir(sub)
        fusion_dir = os.path.join(sub, "fusion")
        seg_dir = os.path.join(sub, "segmentation")
        ensure_dir(fusion_dir)
        ensure_dir(seg_dir)

        with torch.no_grad():
            for vi, ir, label, mask, fname in loader:
                vi, ir, mask = vi.to(device), ir.to(device), mask.to(device)
                alpha_tensor = torch.full((vi.size(0),1,1,1), a, device=device)

                fused, seg_logits = model(vi, ir, alpha=alpha_tensor, mask=mask)

                # ------------------------
                # Save fusion image
                # ------------------------
                fusion_out = fused[0].cpu()
                save_image(fusion_out, os.path.join(fusion_dir, fname[0]))

                # ------------------------
                # Save segmentation as colorful image (fixed)
                # ------------------------
                seg_out = torch.sigmoid(seg_logits[0,1:2]).cpu().numpy()  # shape: (1,H,W)
                seg_out = np.squeeze(seg_out)  # remove singleton dimensions -> H x W

                # Normalize 0-1
                seg_out_norm = (seg_out - seg_out.min()) / (seg_out.max() - seg_out.min() + 1e-8)

                # Apply colormap safely
                seg_rgba = cm.jet(seg_out_norm)  # H x W x 4
                seg_rgb = seg_rgba[:, :, :3]     # drop alpha
                seg_rgb_uint8 = (seg_rgb * 255).astype(np.uint8)

                # Save using PIL
                Image.fromarray(seg_rgb_uint8).save(os.path.join(seg_dir, fname[0]))

                print(f"Saved {fname[0]} alpha={a:.2f} (fusion + segmentation)")

    print("Inference complete. Outputs:", out_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/custom_dataset')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='experiments/outputs/run1')
    parser.add_argument('--alpha', type=str, default='0.1,0.5,0.9', help='comma list e.g. 0.1,0.5,0.9')
    parser.add_argument('--img_size', type=int, nargs=2, default=[480,640])
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--disable_amim', action='store_true')
    parser.add_argument('--disable_mask', action='store_true')
    args = parser.parse_args()
    run_inference(args)
