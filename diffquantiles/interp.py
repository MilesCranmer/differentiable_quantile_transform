"""1D Interpolation in PyTorch"""

import torch
import numpy as np

def interp(x, xp, fp):
    x = x.reshape(-1)
    xp = xp.reshape(-1)
    fp = fp.reshape(-1)
    mask = ((x - xp[:-1, None] >= 0)*(xp[1:, None] - x > 0)).detach().type(x.type())
    output = (mask * fp[:-1, None] + mask * (x - xp[:-1, None]) * (fp[1:, None] - fp[:-1, None])/(xp[1:, None] - xp[:-1, None])).sum(dim=0)
    return output

