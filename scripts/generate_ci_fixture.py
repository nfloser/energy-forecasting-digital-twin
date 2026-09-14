from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from energy_twin.training import publish_training_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()

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
    publish_training_run(
        frame,
        args.output,
        source_metadata={"energy": "deterministic CI fixture", "weather": "deterministic CI fixture"},
        evaluation_test_size=24,
        evaluation_min_train_size=200,
    )


if __name__ == "__main__":
    main()
