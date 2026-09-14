from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import clone

from energy_twin.evaluation import regression_metrics, relative_improvement, walk_forward_splits
from energy_twin.features import FEATURE_COLUMNS, modelling_rows
from energy_twin.models import default_model_specs


def _baseline_predictions(y: np.ndarray, test_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    persistence: list[float] = []
    seasonal: list[float] = []
    for idx in test_indices:
        persistence.append(float(y[idx - 1]))
        seasonal.append(float(y[idx - 24]))
    return np.asarray(persistence), np.asarray(seasonal)


def evaluate_forecasters(
    frame: pd.DataFrame,
    *,
    horizon_hours: int = 1,
    test_size: int = 24 * 7,
    min_train_size: int = 24 * 30,
    seed: int = 42,
) -> dict[str, object]:
    """Run deterministic expanding-window evaluation for one-hour-ahead forecasting."""
    if horizon_hours != 1:
        raise ValueError("v1 evaluation currently supports a one-hour forecast horizon")

    rows = modelling_rows(frame)
    if len(rows) < min_train_size + test_size:
        raise ValueError("not enough complete modelling rows for requested walk-forward evaluation")

    x = rows[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = rows["consumption_kwh"].to_numpy(dtype=float)
    truth: list[float] = []
    predictions: dict[str, list[float]] = defaultdict(list)
    folds = 0

    for train_idx, test_idx in walk_forward_splits(
        rows, min_train_size=min_train_size, test_size=test_size, step=test_size
    ):
        if test_idx[0] < 24:
            continue
        folds += 1
        y_true = y[test_idx]
        persistence, seasonal = _baseline_predictions(y, test_idx)
        truth.extend(y_true.tolist())
        predictions["persistence"].extend(persistence.tolist())
        predictions["seasonal_naive_24h"].extend(seasonal.tolist())

        for spec in default_model_specs(seed=seed):
            estimator = clone(spec.estimator)
            estimator.fit(x[train_idx], y[train_idx])
            predictions[spec.name].extend(estimator.predict(x[test_idx]).tolist())

    if folds == 0:
        raise ValueError("walk-forward configuration produced no evaluation folds")

    truth_array = np.asarray(truth, dtype=float)
    metrics_by_model: dict[str, dict[str, float | None]] = {}
    for name, values in predictions.items():
        metrics_by_model[name] = regression_metrics(
            truth_array, np.asarray(values, dtype=float)
        ).as_dict()

    persistence_mae = float(metrics_by_model["persistence"]["mae"] or 0.0)
    models: list[dict[str, object]] = []
    for name in ["persistence", "seasonal_naive_24h", "ridge", "hist_gradient_boosting"]:
        metrics = metrics_by_model[name]
        improvement = 0.0
        if name != "persistence" and persistence_mae > 0:
            improvement = relative_improvement(persistence_mae, float(metrics["mae"] or 0.0))
        models.append(
            {
                "model": name,
                **metrics,
                "relative_improvement_vs_persistence_pct": improvement,
            }
        )

    return {
        "forecast_horizon_hours": horizon_hours,
        "validation": "expanding_window",
        "folds": folds,
        "observations_evaluated": len(truth),
        "baseline_reference": "persistence",
        "models": models,
    }
