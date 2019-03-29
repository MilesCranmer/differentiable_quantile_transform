"""Contains a differentiable transform for PyTorch"""

import torch
import numpy as np
from .interp import interp as interp


BOUNDS_THRESHOLD = 1e-7
SQRT2 = float(np.sqrt(2))


def inverse_transform_col(X_col, quantiles, references):
    """Private function to transform a single feature"""

    lower_bound_y = quantiles[0]
    upper_bound_y = quantiles[-1]

    X_col = 0.5 + 0.5*torch.erf(X_col/SQRT2)
    X_col = interp(X_col, references, quantiles)

    lower_bounds_mask = (X_col - BOUNDS_THRESHOLD <
                        lower_bound_y).type(X_col.type())
    upper_bounds_mask = (X_col + BOUNDS_THRESHOLD >
                        upper_bound_y).type(X_col.type())
    
    X_col = X_col + upper_bounds_mask * (-X_col + upper_bound_y)
    X_col = X_col + lower_bounds_mask * (-X_col + lower_bound_y)

    return X_col

def inverse_transform(X, quantiles, references):
    """Pass through fitted quantiles from scipy"""
    all_features = []
    for feature_idx in range(X.shape[1]):
        all_features.append(inverse_transform_col(
            X[:, feature_idx], quantiles[:, feature_idx],
            references))
    if len(all_features) > 1:
        X = torch.cat(all_features, dim=1)
    else:
        X = all_features[0]

    return X
