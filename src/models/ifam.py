# src/models/ifam.py

import torch
import torch.nn as nn


class SimpleIFAM(nn.Module):
    """
    A simplified Interactive Fusion Adjustment Module (IFAM).
    Takes a text prompt -> predicts alpha (0 to 1).
    """

    def __init__(self, vocab_size=5000, embed_dim=64, hidden_dim=128):
        super().__init__()

        # 1. Simple word embedding (not CLIP to keep it easy)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 2. Text encoder (GRU)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        # 3. Final layer to output alpha (0–1)
        self.fc_alpha = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()       # alpha in [0,1]
        )

    def forward(self, text_indices):
        """
        text_indices: tensor [B, seq_len] with token IDs
        """
        embedded = self.embedding(text_indices)      # [B, seq_len, embed_dim]
        _, h = self.gru(embedded)                    # h: [1, B, hidden_dim]
        h = h.squeeze(0)                             # [B, hidden_dim]

        alpha = self.fc_alpha(h)                     # [B,1]
        return alpha
