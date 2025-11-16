# src/utils.py

import os
import torch
import numpy as np

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
    return path
