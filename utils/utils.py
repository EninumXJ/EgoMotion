import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)
import hypertools as hyp
import torch.nn.functional as F

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def simp(tensor: torch.Tensor) -> np.array:
    """Simplify a tensor in cuda to a simple numpy array."""
    return torch.squeeze(tensor).cpu().detach().numpy()

def norm(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor in cuda to [0,1]."""
    d_max, _ = torch.max(tensor, dim=-1)
    d_min, _ = torch.min(tensor, dim=-1)
    tensor_norm = (tensor - d_min.unsqueeze(-1)).true_divide(d_max.unsqueeze(-1) - d_min.unsqueeze(-1))
    return tensor_norm
