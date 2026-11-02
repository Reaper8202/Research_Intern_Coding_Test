from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ehrforecast.baseline import MultiLabelLogistic


def test_learns_separable_signal_per_target() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 5))
    y = np.column_stack([(x[:, 0] > 0), (x[:, 1] + x[:, 2] > 0)]).astype(int)
    model = MultiLabelLogistic(C=1.0, n_jobs=1).fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (400, 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert roc_auc_score(y[:, 0], proba[:, 0]) > 0.95
    assert roc_auc_score(y[:, 1], proba[:, 1]) > 0.95
    assert model.coefficients().shape == (2, 5)


def test_rejects_target_with_one_class() -> None:
    x = np.zeros((10, 2))
    y = np.column_stack([np.r_[np.ones(5), np.zeros(5)], np.zeros(10)]).astype(int)
    with pytest.raises(ValueError, match="only one class"):
        MultiLabelLogistic(C=1.0, n_jobs=1).fit(x, y)
