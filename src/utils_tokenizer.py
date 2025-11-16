# src/utils_tokenizer.py

import torch
import re

class SimpleTokenizer:
    """
    A tiny tokenizer for converting text into token IDs.
    Only for demonstration (NOT real NLP).
    """

    def __init__(self, vocab=None):
        if vocab is None:
            # Basic words useful for IFAM training
            self.vocab = {
                "<pad>": 0, "<unk>": 1,
                "increase": 2, "decrease": 3,
                "visible": 4, "infrared": 5,
                "more": 6, "less": 7,
                "contrast": 8, "bright": 9,
                "dark": 10, "feature": 11, "features": 12,
            }
        else:
            self.vocab = vocab

    def encode(self, text, max_len=10):
        text = text.lower()
        words = re.findall(r"\w+", text)
        ids = []

        for w in words:
            if w in self.vocab:
                ids.append(self.vocab[w])
            else:
                ids.append(1)  # <unk>

        # padding / cut
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        else:
            ids = ids[:max_len]

        return torch.tensor(ids, dtype=torch.long)
