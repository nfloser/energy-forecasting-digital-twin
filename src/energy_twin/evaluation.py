from dataclasses import asdict, dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    mape: float | None
    r2: float | None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    nonzero = np.abs(true) > 1e-12
    mape = None
    if nonzero.any():
        mape = float(np.mean(np.abs((true[nonzero] - pred[nonzero]) / true[nonzero])) * 100)
    r2 = float(r2_score(true, pred)) if len(true) > 1 else None
    return RegressionMetrics(
        mae=float(mean_absolute_error(true, pred)),
        rmse=float(mean_squared_error(true, pred) ** 0.5),
        mape=mape,
        r2=r2,
    )


def walk_forward_splits(
    frame: pd.DataFrame,
    *,
    min_train_size: int,
    test_size: int,
    step: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if min_train_size < 1 or test_size < 1 or step < 1:
        raise ValueError("split sizes must be positive")
    n = len(frame)
    train_end = min_train_size
    while train_end + test_size <= n:
        yield np.arange(0, train_end), np.arange(train_end, train_end + test_size)
        train_end += step


def relative_improvement(baseline_mae: float, model_mae: float) -> float:
    if baseline_mae <= 0:
        raise ValueError("baseline MAE must be positive")
    return (baseline_mae - model_mae) / baseline_mae * 100.0
