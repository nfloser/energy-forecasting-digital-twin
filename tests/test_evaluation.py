import numpy as np
import pandas as pd
import pytest

from energy_twin.evaluation import regression_metrics, walk_forward_splits


def test_regression_metrics_are_numerically_correct() -> None:
    metrics = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0]))
    assert metrics.mae == pytest.approx(2 / 3)
    assert metrics.rmse == pytest.approx((2 / 3) ** 0.5)


def test_walk_forward_splits_preserve_time_order() -> None:
    frame = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")})
    splits = list(walk_forward_splits(frame, min_train_size=10, test_size=3, step=3))
    assert splits
    for train_idx, test_idx in splits:
        assert max(train_idx) < min(test_idx)
