from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


class PersistenceBaseline:
    def fit(self, values: np.ndarray) -> "PersistenceBaseline":
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            raise ValueError("history must not be empty")
        self.last_value = float(values[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        return np.full(horizon, self.last_value, dtype=float)


class SeasonalNaiveBaseline:
    def __init__(self, period: int = 24) -> None:
        if period < 1:
            raise ValueError("period must be positive")
        self.period = period

    def fit(self, values: np.ndarray) -> "SeasonalNaiveBaseline":
        values = np.asarray(values, dtype=float)
        if values.size < self.period:
            raise ValueError("history shorter than seasonal period")
        self.season = values[-self.period :].copy()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        return np.resize(self.season, horizon).astype(float)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object


def default_model_specs(seed: int = 42) -> list[ModelSpec]:
    return [
        ModelSpec("ridge", Ridge(alpha=1.0)),
        ModelSpec(
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(
                learning_rate=0.08,
                max_iter=200,
                max_leaf_nodes=31,
                random_state=seed,
            ),
        ),
    ]
