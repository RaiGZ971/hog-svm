import numpy as np

class StratifiedKFold:
    """
    Splits your gesture dataset into K folds for cross-validation
    while keeping the same class ratio in every fold.
    This matters because if one fold has no samples of a gesture class,
    that fold's accuracy score will be misleading.
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits, self.shuffle, self.random_state = n_splits, shuffle, random_state

    def split(self, X, y):
        """
        Yields (train_indices, val_indices) for each fold.
        Distributes samples from each class evenly so every fold
        gets a fair share of every gesture type.
        """
        rng = np.random.default_rng(self.random_state)
        class_idx = {c: np.where(y == c)[0] for c in np.unique(y)}
        if self.shuffle:
            for idx in class_idx.values():
                rng.shuffle(idx)
        folds = [[] for _ in range(self.n_splits)]
        for idx in class_idx.values():
            for k, part in enumerate(np.array_split(idx, self.n_splits)):
                folds[k].extend(part)
        for k in range(self.n_splits):
            test  = np.array(folds[k], dtype=int)
            train = np.concatenate([np.array(folds[f], dtype=int)
                                    for f in range(self.n_splits) if f != k])
            yield train, test
