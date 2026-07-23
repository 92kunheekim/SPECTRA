"""Global seeding for reproducibility."""
import os, random
import numpy as np


def seed_everything(seed: int = 42) -> int:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return seed
