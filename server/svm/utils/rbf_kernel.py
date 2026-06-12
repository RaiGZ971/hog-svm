import numpy as np

def rbf_kernel(X, Z, gamma):
    """
    Measures similarity between two sets of data points.
    Closer points get a score near 1, distant points near 0.
    This is what allows the SVM to find non-linear boundaries
    between the gesture classes.
    """
    X_norm  = np.sum(X ** 2, axis=1, keepdims=True)
    Z_norm  = np.sum(Z ** 2, axis=1, keepdims=True)
    sq_dist = X_norm + Z_norm.T - 2.0 * X @ Z.T
    return np.exp(-gamma * sq_dist)
