import numpy as np
from itertools import product
from svm.standard_scaler import StandardScaler
from svm.multiclass_svm import MulticlassSVM


class GridSearchCV:
    """
    Automatically finds the best C and gamma values for your SVM
    by trying every combination and measuring accuracy via cross-validation.
    Prevents you from manually tuning hyperparameters.
    After the best combination is found, re-trains the final model
    on your full training set.
    """

    def __init__(self, param_grid, cv, verbose=0):
        self.param_grid, self.cv, self.verbose = param_grid, cv, verbose
        self.best_params_    = None
        self.best_score_     = -np.inf
        self.best_estimator_ = None
        self._best_scaler    = None

    def fit(self, X, y):
        """
        Runs the full grid search. For every parameter combination:
        scales the data within each fold, trains a MulticlassSVM,
        and records the validation accuracy. The best combination
        is then used to train the final model on all training data.
        """
        keys, values = list(self.param_grid.keys()), list(self.param_grid.values())
        for combo in product(*values):
            params = dict(zip(keys, combo))
            scores = []
            for train_idx, val_idx in self.cv.split(X, y):
                sc    = StandardScaler()
                X_tr  = sc.fit_transform(X[train_idx])
                X_val = sc.transform(X[val_idx])
                model = MulticlassSVM(**params)
                model.fit(X_tr, y[train_idx])
                scores.append(model.score(X_val, y[val_idx]))
            mean_score = float(np.mean(scores))
            if self.verbose:
                print(f"  params={params}  cv_acc={mean_score:.4f}")
            if mean_score > self.best_score_:
                self.best_score_, self.best_params_ = mean_score, params
        sc = StandardScaler()
        self._best_scaler    = sc
        self.best_estimator_ = MulticlassSVM(**self.best_params_)
        self.best_estimator_.fit(sc.fit_transform(X), y)
        return self
