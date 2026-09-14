from pathlib import Path

import numpy as np
import pandas as pd

from energy_twin.training import publish_training_run


def test_training_run_persists_reproducible_research_artefacts(tmp_path: Path) -> None:
    n = 24 * 20
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    hour = ts.hour.to_numpy()
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "consumption_kwh": 2.5 + np.sin(2 * np.pi * hour / 24) + np.arange(n) * 0.0005,
            "temperature_c": 8 + 4 * np.sin(2 * np.pi * hour / 24),
            "humidity_pct": np.full(n, 70.0),
        }
    )

    result = publish_training_run(
        frame,
        tmp_path,
        source_metadata={"energy": "test fixture", "weather": "test fixture"},
        evaluation_test_size=24,
        evaluation_min_train_size=200,
    )

    assert (tmp_path / "evaluation.json").exists()
    assert (tmp_path / "provenance.json").exists()
    assert (tmp_path / "forecast.json").exists()
    assert (tmp_path / "models" / "ridge-v1.json").exists()
    assert (tmp_path / "models" / "hist-gradient-boosting-v1.json").exists()
    assert result["selected_model"] in {"ridge", "hist_gradient_boosting"}
