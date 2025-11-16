# src/train_ifam.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.models.ifam import SimpleIFAM
from src.utils_tokenizer import SimpleTokenizer
import random
import argparse
import os

# -------------------------------------------------------------------
# Fake tiny dataset of <text, alpha> pairs (demo for professor test)
# -------------------------------------------------------------------
class IFAMTrainingSet(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # Simple mapping of text -> target alpha
        self.samples = [
            ("increase visible features", 0.9),
            ("more visible", 0.85),
            ("increase brightness", 0.8),
            ("make infrared weaker", 0.7),
            ("balanced fusion", 0.5),
            ("equal fusion", 0.5),
            ("increase infrared features", 0.2),
            ("more infrared", 0.25),
            ("dark infrared boost", 0.1),
            ("less visible", 0.3),
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, alpha = self.samples[idx]
        ids = self.tokenizer.encode(text)  # tokenized text
        return ids, torch.tensor([alpha], dtype=torch.float)


# -------------------------------------------------------------------
# Training Function
# -------------------------------------------------------------------
def train_ifam(save_path="experiments/outputs/ifam_model.pth", epochs=50):
    tokenizer = SimpleTokenizer()
    dataset = IFAMTrainingSet(tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = SimpleIFAM()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("Training IFAM...")
    for ep in range(epochs):
        for text_ids, target_alpha in loader:
            pred_alpha = model(text_ids)
            loss = loss_fn(pred_alpha, target_alpha)

            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f"Epoch {ep+1}/{epochs} | Loss = {loss.item():.4f}")

    # save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"IFAM saved to: {save_path}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default="experiments/outputs/ifam_model.pth")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    train_ifam(args.save, args.epochs)
