"""1D Interpolation in PyTorch"""

import torch

def interp(x, xp, fp):
    return (((x - xp[:-1, None] > 0)*(xp[1:, None] - x > 0)).type(torch.FloatTensor) * \
        (fp[:-1, None] + (x - xp[:-1, None]) * (fp[1:, None] - fp[:-1, None])/(xp[1:, None] - xp[:-1, None]))).sum(dim=0)
