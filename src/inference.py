# src/inference.py  (fusion-only inference)
import os
import argparse
import torch
from torchvision.utils import save_image
from src.datasets import VIFDataset
from src.models.cvifsm import CVIFSM

def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = VIFDataset(os.path.join(args.data_root, "TEST"), mode='test', img_size=args.img_size)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    model = CVIFSM(base_ch=args.base_ch, att_enable=not args.disable_amim, use_mask=not args.disable_mask).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # support multiple alphas
    alphas = [float(x) for x in args.alpha.split(',')] if isinstance(args.alpha, str) else [float(args.alpha)]
    out_root = args.save_dir
    ensure_dir(out_root)
    for a in alphas:
        sub = os.path.join(out_root, f"alpha_{a:.2f}")
        ensure_dir(sub)
        fusion_dir = os.path.join(sub, "fusion")
        ensure_dir(fusion_dir)

        with torch.no_grad():
            for vi, ir, label, mask, fname in loader:
                vi = vi.to(device); ir = ir.to(device); mask = mask.to(device)
                alpha_tensor = torch.full((vi.size(0),1,1,1), a, device=device)
                fused = model(vi, ir, alpha=alpha_tensor, mask=mask)
                out = fused[0].cpu()
                save_image(out, os.path.join(fusion_dir, fname[0]))
                print(f"Saved {fname[0]} alpha={a:.2f}")

    print("Inference complete. Outputs:", out_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/custom_dataset')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='experiments/outputs/run1')
    parser.add_argument('--alpha', type=str, default='0.1,0.5,0.9', help='comma list e.g. 0.1,0.5,0.9')
    parser.add_argument('--img_size', type=tuple, default=(480,640))
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--disable_amim', action='store_true')
    parser.add_argument('--disable_mask', action='store_true')
    args = parser.parse_args()
    run_inference(args)
