# utils.py

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    Note: exact bit-level reproducibility may still vary across hardware,
    but this makes training highly stable and consistent.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Ensures repeatable hashing in Python
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CUDNN flags for deterministic behaviour
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
