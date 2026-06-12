import numpy as np

class StandardScaler:
    """
    Normalises each feature to have zero mean and unit variance.
    Prevents features with large numeric ranges from dominating
    the SVM kernel, which improves accuracy and training stability.
    Must always be fit on training data only, then applied to
    test and inference data to prevent data leakage.
    """

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        # Applies the learned mean/std to new data (test or inference).
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        # Fits and transforms in one step — used on training data only.
        return self.fit(X).transform(X)
