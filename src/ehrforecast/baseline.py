"""One L2-regularised logistic regression per target, on standardised features."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _fit_one(
    x: np.ndarray, y: np.ndarray, c: float, max_iter: int
) -> tuple[LogisticRegression, bool]:
    model = LogisticRegression(C=c, max_iter=max_iter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, converged


@dataclass
class MultiLabelLogistic:
    """Independent binary logistic regressions that share one feature scaler."""

    C: float
    max_iter: int = 5000
    n_jobs: int = -1

    def fit(self, x: np.ndarray, y: np.ndarray) -> MultiLabelLogistic:
        if y.ndim != 2 or len(x) != len(y):
            raise ValueError("expected x of shape (n, d) and y of shape (n, k)")
        single_class = [j for j in range(y.shape[1]) if np.unique(y[:, j]).size < 2]
        if single_class:
            raise ValueError(f"targets with only one class in training data: {single_class}")
        self.scaler_ = StandardScaler().fit(x)
        xs = self.scaler_.transform(x)
        fitted = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(xs, y[:, j], self.C, self.max_iter) for j in range(y.shape[1])
        )
        self.estimators_ = [model for model, _ in fitted]
        self.n_not_converged_ = sum(not ok for _, ok in fitted)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return the positive-class probability, shape (n, k)."""
        xs = self.scaler_.transform(x)
        return np.column_stack([m.predict_proba(xs)[:, 1] for m in self.estimators_])

    def coefficients(self) -> np.ndarray:
        """Coefficients on the standardised features, shape (k, d)."""
        return np.vstack([m.coef_[0] for m in self.estimators_])
