from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone

from energy_twin.features import FEATURE_COLUMNS, modelling_rows
from energy_twin.models import default_model_specs
from energy_twin.pipeline import evaluate_forecasters
from energy_twin.provenance import dataframe_sha256, write_json
from energy_twin.quality import assess_hourly_series
from energy_twin.registry import ModelMetadata, ModelRegistry


def _model_id(name: str) -> str:
    return f"{name.replace('_', '-')}-v1"


def publish_training_run(
    frame: pd.DataFrame,
    output_dir: Path | str,
    *,
    source_metadata: dict[str, Any],
    evaluation_test_size: int = 24 * 7,
    evaluation_min_train_size: int = 24 * 30,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate, fit and persist a traceable one-hour-ahead research forecast run."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "processed").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)

    canonical = frame.copy().sort_values("timestamp").reset_index(drop=True)
    canonical["timestamp"] = pd.to_datetime(canonical["timestamp"], utc=True)
    canonical.to_csv(root / "processed" / "hourly.csv", index=False)

    quality = assess_hourly_series(canonical[["timestamp", "consumption_kwh"]])
    write_json(root / "quality.json", quality.as_dict())

    evaluation = evaluate_forecasters(
        canonical,
        horizon_hours=1,
        test_size=evaluation_test_size,
        min_train_size=evaluation_min_train_size,
        seed=seed,
    )
    write_json(root / "evaluation.json", evaluation)

    rows = modelling_rows(canonical)
    if len(rows) < 2:
        raise ValueError("at least two complete modelling rows are required")
    train = rows.iloc[:-1]
    target = rows.iloc[-1]
    x_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train["consumption_kwh"].to_numpy(dtype=float)
    x_target = target[FEATURE_COLUMNS].to_numpy(dtype=float).reshape(1, -1)

    evaluation_rows = {item["model"]: item for item in evaluation["models"]}
    ml_names = ["ridge", "hist_gradient_boosting"]
    selected_model = min(ml_names, key=lambda name: float(evaluation_rows[name]["mae"]))

    dataset_hash = dataframe_sha256(canonical)
    trained_at = datetime.now(UTC).isoformat()
    registry = ModelRegistry(root / "models")
    predictions: dict[str, float] = {}

    for spec in default_model_specs(seed=seed):
        estimator = clone(spec.estimator)
        estimator.fit(x_train, y_train)
        prediction = float(estimator.predict(x_target)[0])
        predictions[spec.name] = prediction
        model_id = _model_id(spec.name)
        joblib.dump(estimator, root / "models" / f"{model_id}.joblib")

        row = evaluation_rows[spec.name]
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=spec.name,
            training_start=pd.Timestamp(train.iloc[0]["timestamp"]).isoformat(),
            training_end=pd.Timestamp(train.iloc[-1]["timestamp"]).isoformat(),
            feature_set=list(FEATURE_COLUMNS),
            forecast_horizon_hours=1,
            metrics={
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "mape": None if row["mape"] is None else float(row["mape"]),
                "r2": None if row["r2"] is None else float(row["r2"]),
            },
            dataset_hash=dataset_hash,
            trained_at=trained_at,
            baseline_comparison={
                "persistence_mae": float(evaluation_rows["persistence"]["mae"]),
                "relative_improvement_pct": float(
                    row["relative_improvement_vs_persistence_pct"]
                ),
            },
        )
        registry.save_metadata(metadata)

    forecast = {
        "mode": "historical_holdout",
        "methodological_note": (
            "The demo forecast is a one-step historical holdout. The target observation is "
            "excluded from model fitting and retained only for transparent retrospective "
            "comparison."
        ),
        "forecast_horizon_hours": 1,
        "target_timestamp": pd.Timestamp(target["timestamp"]).isoformat(),
        "selected_model": selected_model,
        "model_id": _model_id(selected_model),
        "predicted_consumption_kwh": predictions[selected_model],
        "actual_consumption_kwh": float(target["consumption_kwh"]),
        "generated_at": trained_at,
    }
    write_json(root / "forecast.json", forecast)

    provenance = {
        "dataset_hash_sha256": dataset_hash,
        "sources": source_metadata,
        "training_period": {
            "start": pd.Timestamp(train.iloc[0]["timestamp"]).isoformat(),
            "end": pd.Timestamp(train.iloc[-1]["timestamp"]).isoformat(),
        },
        "forecast_target": pd.Timestamp(target["timestamp"]).isoformat(),
        "weather_features": ["temperature_c", "humidity_pct"],
        "feature_set": list(FEATURE_COLUMNS),
        "forecast_horizon_hours": 1,
        "random_seed": seed,
        "validation": evaluation["validation"],
        "selected_model": selected_model,
    }
    write_json(root / "provenance.json", provenance)

    return {
        "selected_model": selected_model,
        "dataset_hash": dataset_hash,
        "evaluation": evaluation,
        "forecast": forecast,
    }
