"""Contains a differentiable transform for PyTorch"""

import torch
from interp import interp

BOUNDS_THRESHOLD = 1e-7
SQRT2 = float(np.sqrt(2))


def inverse_transform_col(X_col, quantiles, references):
    """Private function to transform a single feature"""

    lower_bound_y = quantiles[0]
    upper_bound_y = quantiles[-1]

    X_col = 0.5 + 0.5*torch.erf(X_col/SQRT2)
    X_col = interp(X_col, references, quantiles)

    lower_bounds_mask = (X_col - BOUNDS_THRESHOLD <
                        lower_bound_y).type(torch.FloatTensor)
    upper_bounds_mask = (X_col + BOUNDS_THRESHOLD >
                        upper_bound_y).type(torch.FloatTensor)
    
    X_col = X_col + upper_bounds_mask * (-X_col + upper_bound_y)
    X_col = X_col + lower_bounds_mask * (-X_col + lower_bound_y)

    return X_col

def inverse_transform(X, quantiles, references):
    """Pass through fitted quantiles from scipy"""
    for feature_idx in range(X.shape[1]):
        X[:, feature_idx] = inverse_transform_col(
            X[:, feature_idx], quantiles[:, feature_idx],
            references)

    return X
