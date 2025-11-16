# src/datasets.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import random

class VIFDataset(Dataset):
    """
    Paired Visible-Infrared dataset loader.
    """
    def __init__(self, root, mode='train', label_dir=None, img_size=(240,320), transform=None):
        super().__init__()
        assert mode in ('train','test')
        self.root = root
        self.mode = mode
        self.label_dir = label_dir
        self.img_size = img_size

        vis_dir = os.path.join(root, "Visible")
        ir_dir  = os.path.join(root, "Infrared")

        if not os.path.isdir(vis_dir) or not os.path.isdir(ir_dir):
            raise RuntimeError(f"Visible or Infrared folder not found under {root}")

        vis_list = sorted([f for f in os.listdir(vis_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        ir_list  = sorted([f for f in os.listdir(ir_dir)  if f.lower().endswith(('.png','.jpg','.jpeg'))])

        self.names = vis_list[:min(len(vis_list), len(ir_list))]
        self.vis_dir = vis_dir
        self.ir_dir = ir_dir

        # labels (not used in your dataset)
        self.lbl_dir = None

        # transforms
        self.to_tensor = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.names)

    def random_binary_mask(self, H, W, max_regions=4):
        mask = np.zeros((H, W), dtype=np.uint8)
        num = random.randint(1, max_regions)
        for _ in range(num):
            x1 = random.randint(0, W-1)
            y1 = random.randint(0, H-1)
            x2 = random.randint(x1, min(W-1, x1 + random.randint(20, W//2)))
            y2 = random.randint(y1, min(H-1, y1 + random.randint(20, H//2)))
            mask[y1:y2, x1:x2] = 1
        return mask

    def __getitem__(self, idx):
        fname = self.names[idx]

        vis_path = os.path.join(self.vis_dir, fname)
        ir_path  = os.path.join(self.ir_dir, fname)

        vis = Image.open(vis_path).convert('RGB')
        ir  = Image.open(ir_path).convert('L')

        vis_t = self.to_tensor(vis)     # [3,H,W]
        ir_t  = self.to_tensor(ir)      # [1,H,W]

        if ir_t.dim() == 2:
            ir_t = ir_t.unsqueeze(0)

        H, W = vis_t.shape[1], vis_t.shape[2]

        # ---------------------------
        # 🔥 FIX: ALWAYS return a valid segmentation label
        # ---------------------------
        label = torch.zeros((H, W), dtype=torch.long)

        # mask
        if self.mode == "train":
            mask_np = self.random_binary_mask(H, W)
        else:
            mask_np = np.ones((H, W), dtype=np.uint8)

        mask = torch.from_numpy(mask_np).float()

        return vis_t, ir_t, label, mask, fname


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/custom_dataset/TRAIN"
    ds = VIFDataset(root, mode='train', img_size=(480,640))
    print("Dataset size:", len(ds))
    v,i,lab,m,name = ds[0]
    print("sample shapes:", v.shape, i.shape, "label:", lab.shape, "mask:", m.shape, "name:", name)
