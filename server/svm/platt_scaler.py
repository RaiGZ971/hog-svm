import numpy as np


class PlattScaler:
    """
    Converts the SVM's raw scores into actual confidence percentages.
    Without this, the SVM only tells you which class wins — not how
    confident it is. run_svm_model needs a confidence value, so this
    is required.
    """

    def __init__(self):
        self.A = self.B = 0.0

    def fit(self, scores, y):
        """
        Learns how to map raw SVM scores to probabilities by fitting
        a sigmoid curve to the training scores.
        Uses smoothed targets so the model doesn't become overconfident.
        """
        n_pos, n_neg = np.sum(y == 1), np.sum(y == -1)
        t  = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1.0 / (n_neg + 2))
        A, B = 0.0, np.log((n_neg + 1) / (n_pos + 1))
        for _ in range(1000):
            fApB = scores * A + B
            p    = np.where(fApB >= 0,
                            np.exp(-fApB) / (1 + np.exp(-fApB)),
                            1.0 / (1 + np.exp(fApB)))
            A -= 0.01 * np.mean(scores * (p - t))
            B -= 0.01 * np.mean(p - t)
        self.A, self.B = A, B

    def predict_proba(self, scores):
        """
        Outputs the probability of belonging to each class for a sample.
        e.g. [0.05, 0.95] means 95% confident it is the positive class.
        run_svm_model uses this to get the confidence score it prints.
        """
        fApB = scores * self.A + self.B
        p1   = np.where(fApB >= 0,
                        np.exp(-fApB) / (1 + np.exp(-fApB)),
                        1.0 / (1 + np.exp(fApB)))
        return np.column_stack([1 - p1, p1])
