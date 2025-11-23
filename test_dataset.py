from src.datasets import VIFDataset
from src.models.cvifsm import CVIFSM
import torch

# Dataset
ds = VIFDataset("data/custom_dataset/TRAIN", mode='train', label_dir="data/custom_dataset/MASKS", img_size=(240,320))
print("Dataset size:", len(ds))

vi, ir, label, mask, name = ds[0]
print("Shapes:", vi.shape, ir.shape, label.shape, mask.shape)

# Model
model = CVIFSM()
fused = model(vi.unsqueeze(0), ir.unsqueeze(0), alpha=0.5, mask=mask.unsqueeze(0))
print("Fused shape:", fused.shape)
