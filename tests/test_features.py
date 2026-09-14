import numpy as np
import pandas as pd

from energy_twin.features import build_features


def _frame() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "consumption_kwh": np.arange(72, dtype=float),
            "temperature_c": np.linspace(5, 10, 72),
            "humidity_pct": np.linspace(80, 60, 72),
        }
    )


def test_lag_features_use_only_past_values() -> None:
    features = build_features(_frame())
    row = features.loc[features["timestamp"] == pd.Timestamp("2024-01-02 06:00Z")].iloc[0]
    assert row["lag_1h"] == 29.0
    assert row["lag_24h"] == 6.0
    assert row["rolling_mean_6h"] == np.mean([24, 25, 26, 27, 28, 29])


def test_future_target_change_does_not_change_past_features() -> None:
    original = _frame()
    mutated = original.copy()
    mutated.loc[mutated.index[-1], "consumption_kwh"] = 999999
    a = build_features(original).iloc[:-1].reset_index(drop=True)
    b = build_features(mutated).iloc[:-1].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
