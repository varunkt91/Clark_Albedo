# 🔹 Define Z-score normalization function

import numpy as np
def z_score_normalize(train, val, test):
    """
    Apply Z-score normalization using training data statistics.

    Parameters:
        train, val, test : np.ndarray
            Arrays to normalize.

    Returns:
        train_norm, val_norm, test_norm, mean, std : np.ndarray
    """
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    std[std == 0] = 1e-8  # avoid division by zero

    train_norm = (train - mean) / std
    val_norm = (val - mean) / std
    test_norm = (test - mean) / std

    return train_norm, val_norm, test_norm, mean, std


def inverse_z_score(normalized, mean, std):
    """
    Reverse Z-score normalization to get back original values.

    Parameters:
        normalized : np.ndarray
            Normalized array to invert.
        mean : float or np.ndarray
            Mean used during normalization.
        std : float or np.ndarray
            Standard deviation used during normalization.

    Returns:
        np.ndarray : Original scale values
    """
    return normalized * std + mean