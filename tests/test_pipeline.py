import numpy as np
import pandas as pd

from energy_twin.pipeline import evaluate_forecasters


def test_evaluation_compares_all_required_model_families() -> None:
    n = 24 * 20
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    hour = ts.hour.to_numpy()
    demand = 2.0 + np.sin(2 * np.pi * hour / 24) + np.arange(n) * 0.001
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "consumption_kwh": demand,
            "temperature_c": 10 + 5 * np.sin(2 * np.pi * hour / 24),
            "humidity_pct": np.full(n, 70.0),
        }
    )
    result = evaluate_forecasters(frame, horizon_hours=1, test_size=24, min_train_size=200)
    assert {row["model"] for row in result["models"]} == {
        "persistence",
        "seasonal_naive_24h",
        "ridge",
        "hist_gradient_boosting",
    }
    assert all(row["mae"] >= 0 for row in result["models"])
    assert result["baseline_reference"] == "persistence"
