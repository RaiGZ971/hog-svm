import numpy as np
from scipy.optimize import minimize

class PlattScaler:
    def __init__(self):
        self.A = self.B = 0.0

    def fit(self, scores, y):
        n_pos, n_neg = np.sum(y == 1), np.sum(y == -1)
        t = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1.0 / (n_neg + 2))

        def loss(params):
            A, B = params
            fApB = np.clip(scores * A + B, -500, 500)
            p = np.where(fApB >= 0,
                         np.exp(-fApB) / (1 + np.exp(-fApB)),
                         1.0 / (1 + np.exp(fApB)))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))

        B0 = np.log((n_neg + 1) / (n_pos + 1))
        result = minimize(loss, x0=[0.0, B0], method='L-BFGS-B')
        self.A, self.B = result.x

    def predict_proba(self, scores):
        """
        confidence = (sum of sigmoid(score * A + B) across length pairs) / length pairs
             ──────────────────────────────────────────────────────
             sum of all 23 class averages
        """
        scores = np.array(scores).flatten()
        fApB = np.clip(scores * self.A + self.B, -500, 500)
        p1 = np.where(fApB >= 0,
                      np.exp(-fApB) / (1 + np.exp(-fApB)),
                      1.0 / (1 + np.exp(fApB)))
        return np.column_stack([1 - p1, p1])