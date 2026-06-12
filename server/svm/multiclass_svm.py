import numpy as np
from svm.binary_svm import BinarySVM
from svm.platt_scaler import PlattScaler


class MulticlassSVM:
    """
    Extends BinarySVM to recognise more than two gesture classes.
    Trains one BinarySVM for every pair of classes (one-vs-one).
    At prediction time, each classifier votes for its preferred class
    and the class with the most votes wins.
    Example: 5 gesture classes → 10 binary classifiers trained.
    """

    def __init__(self, C=1.0, gamma="scale", tol=1e-3, max_passes=10, **_):
        self.C, self.gamma, self.tol, self.max_passes = C, gamma, tol, max_passes
        self.classes_ = None
        self._classifiers, self._scalers = {}, {}

    def fit(self, X, y):
        """
        Trains a BinarySVM and a PlattScaler for every class pair.
        Each classifier only sees the two classes it is responsible for.
        """
        self.classes_ = np.unique(y)
        pairs = [(i, j) for i in range(len(self.classes_))
                        for j in range(len(self.classes_)) if i < j]
        for i, j in pairs:
            ci, cj = self.classes_[i], self.classes_[j]
            mask   = (y == ci) | (y == cj)
            X_bin  = X[mask]
            y_bin  = np.where(y[mask] == ci, 1, -1)
            svm    = BinarySVM(C=self.C, gamma=self.gamma,
                               tol=self.tol, max_passes=self.max_passes)
            svm.fit(X_bin, y_bin)
            ps = PlattScaler()
            ps.fit(svm.decision_function(X_bin), y_bin)
            self._classifiers[(i, j)] = svm
            self._scalers[(i, j)]     = ps
        return self

    def predict(self, X):
        """
        Predicts the gesture class for each sample by majority vote.
        Every binary classifier votes for one of its two classes,
        and the class with the highest total votes is the prediction.
        """
        votes = np.zeros((len(X), len(self.classes_)), dtype=int)
        for (i, j), svm in self._classifiers.items():
            preds = svm.predict(X)
            for k, p in enumerate(preds):
                if p == 1: votes[k, i] += 1
                else:      votes[k, j] += 1
        return self.classes_[np.argmax(votes, axis=1)]

    def predict_proba(self, X):
        """
        Returns a confidence score for each gesture class per sample.
        Averages the probability outputs from all binary Platt scalers
        that involve each class, then normalises to sum to 1.
        run_svm_model uses this to print the confidence of its prediction.
        """
        n_cls  = len(self.classes_)
        proba  = np.zeros((len(X), n_cls))
        counts = np.zeros(n_cls)
        for (i, j), svm in self._classifiers.items():
            p = self._scalers[(i, j)].predict_proba(svm.decision_function(X))
            proba[:, i] += p[:, 1]
            proba[:, j] += p[:, 0]
            counts[i]   += 1
            counts[j]   += 1
        proba /= np.maximum(counts, 1)
        row_sums = proba.sum(axis=1, keepdims=True)
        return proba / np.where(row_sums == 0, 1, row_sums)

    def score(self, X, y):
        # Returns the fraction of correctly predicted gesture classes.
        # Used in evaluate_svm_model to print train and test accuracy.
        return float(np.mean(self.predict(X) == y))
