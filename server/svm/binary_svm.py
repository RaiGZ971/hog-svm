import numpy as np
from svm.utils.rbf_kernel import rbf_kernel


class BinarySVM:
    """
    Learns a decision boundary between exactly two gesture classes.
    Finds the widest possible margin separating them using only
    the most important training samples (support vectors).
    C controls the trade-off between a wider margin and fewer misclassifications.
    """

    def __init__(self, C=1.0, gamma="scale", tol=1e-3, max_passes=10):
        self.C, self.gamma, self.tol, self.max_passes = C, gamma, tol, max_passes
        self.sv_X = self.sv_y = self.sv_alpha = self.b = None
        self._gamma_val = None

    def _resolve_gamma(self, X):
        # Converts gamma="scale" into a concrete number based on
        # your data's spread. Avoids having to tune it manually.
        if self.gamma == "scale":
            var = X.var()
            return 1.0 / (X.shape[1] * var) if var != 0 else 1.0
        return float(self.gamma)

    def _K(self, A, B):
        # Shorthand to compute the RBF kernel with the resolved gamma.
        return rbf_kernel(A, B, self._gamma_val)

    def _decision(self, X):
        # Computes the raw score for each sample.
        # Positive = belongs to class +1, negative = class -1.
        return (self.sv_alpha * self.sv_y) @ self._K(self.sv_X, X) + self.b

    def fit(self, X, y):
        """
        Trains the binary SVM by finding the best alpha values
        (one per training sample) that define the decision boundary.
        Uses the SMO algorithm — solves the optimisation by updating
        two alphas at a time until no more improvements can be made.
        After training, only support vectors (alphas > 0) are kept.
        """
        n = X.shape[0]
        self._gamma_val = self._resolve_gamma(X)
        alphas, b = np.zeros(n), 0.0
        K = self._K(X, X)
        passes = 0
        while passes < self.max_passes:
            num_changed = 0
            for i in range(n):
                Ei = float((alphas * y) @ K[:, i]) + b - y[i]
                if not ((y[i] * Ei < -self.tol and alphas[i] < self.C) or
                        (y[i] * Ei >  self.tol and alphas[i] > 0)):
                    continue
                j = i
                while j == i:
                    j = np.random.randint(0, n)
                Ej = float((alphas * y) @ K[:, j]) + b - y[j]
                ai_old, aj_old = alphas[i], alphas[j]
                if y[i] != y[j]:
                    L, H = max(0.0, alphas[j] - alphas[i]), min(self.C, self.C + alphas[j] - alphas[i])
                else:
                    L, H = max(0.0, alphas[i] + alphas[j] - self.C), min(self.C, alphas[i] + alphas[j])
                if L >= H:
                    continue
                eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue
                alphas[j] = np.clip(alphas[j] - y[j] * (Ei - Ej) / eta, L, H)
                if abs(alphas[j] - aj_old) < 1e-5:
                    continue
                alphas[i] += y[i] * y[j] * (aj_old - alphas[j])
                b1 = b - Ei - y[i] * (alphas[i] - ai_old) * K[i, i] - y[j] * (alphas[j] - aj_old) * K[i, j]
                b2 = b - Ej - y[i] * (alphas[i] - ai_old) * K[i, j] - y[j] * (alphas[j] - aj_old) * K[j, j]
                b  = b1 if 0 < alphas[i] < self.C else b2 if 0 < alphas[j] < self.C else (b1 + b2) / 2.0
                num_changed += 1
            passes = passes + 1 if num_changed == 0 else 0
        sv = alphas > 1e-5
        self.sv_X, self.sv_y, self.sv_alpha, self.b = X[sv], y[sv], alphas[sv], b

    def predict(self, X):
        # Returns the predicted class label (-1 or +1) for each sample.
        return np.sign(self._decision(X)).astype(int)

    def decision_function(self, X):
        # Returns raw float scores before converting to labels.
        # PlattScaler needs these scores to learn the probability mapping.
        return self._decision(X)
