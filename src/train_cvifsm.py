# src/train_cvifsm.py  (fusion-only training)
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from src.datasets import VIFDataset
from src.models.cvifsm import CVIFSM
from src.losses import FusionLoss

def image_to_save(tensor):
    return tensor.clamp(0,1)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs(args.save_dir, exist_ok=True)
    sample_dir = os.path.join(args.save_dir, "fusion_samples")
    os.makedirs(sample_dir, exist_ok=True)

    train_ds = VIFDataset(os.path.join(args.data_root, 'TRAIN'), mode='train', img_size=args.img_size)
    test_ds  = VIFDataset(os.path.join(args.data_root, 'TEST'), mode='test', img_size=args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    model = CVIFSM(base_ch=args.base_ch, att_enable=not args.disable_amim, use_mask=not args.disable_mask).to(device)
    loss_fn = FusionLoss(w_l1=args.w_l1, w_grad=args.w_grad)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for vi, ir, label, mask, fname in pbar:
            vi = vi.to(device); ir = ir.to(device); mask = mask.to(device)
            alpha = torch.rand(vi.size(0), device=device).view(vi.size(0),1,1,1) if args.random_alpha else torch.full((vi.size(0),1,1,1), args.alpha, device=device)

            fused = model(vi, ir, alpha=alpha, mask=mask)
            loss = loss_fn(fused, vi, ir, alpha)

            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss.item())
            pbar.set_postfix({'loss': f"{running/((pbar.n+1)):.4f}"})

        # save checkpoint
        ckpt = {'epoch': epoch, 'model': model.state_dict(), 'opt': opt.state_dict()}
        torch.save(ckpt, os.path.join(args.save_dir, f"ckpt_epoch_{epoch:03d}.pth"))

        # save a few sample fused images from test set
        model.eval()
        with torch.no_grad():
            saved = 0
            for vi, ir, label, mask, fname in test_loader:
                vi = vi.to(device); ir = ir.to(device); mask = mask.to(device)
                alpha_test = torch.full((vi.size(0),1,1,1), args.alpha, device=device)
                fused = model(vi, ir, alpha=alpha_test, mask=mask)
                out = image_to_save(fused[0].cpu())
                vutils.save_image(out, os.path.join(sample_dir, f"epoch{epoch:03d}_{fname[0]}"))
                saved += 1
                if saved >= args.num_samples: break

    # save final model (as model field for compatibility)
    torch.save({'epoch': args.epochs-1, 'model': model.state_dict()}, os.path.join(args.save_dir, "model_cvifsm.pth"))
    print("Training finished. Outputs saved to", args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/custom_dataset')
    parser.add_argument('--save_dir', type=str, default='experiments/outputs/run1')
    parser.add_argument('--img_size', type=tuple, default=(480,640))
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--random_alpha', action='store_true')
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--disable_amim', action='store_true')
    parser.add_argument('--disable_mask', action='store_true')
    parser.add_argument('--w_l1', type=float, default=1.0)
    parser.add_argument('--w_grad', type=float, default=1.0)
    parser.add_argument('--num_samples', type=int, default=6)
    args = parser.parse_args()
    train(args)
