from pathlib import Path

from energy_twin.registry import ModelMetadata, ModelRegistry


def test_registry_round_trip(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    metadata = ModelMetadata(
        model_id="ridge-v1",
        model_type="ridge",
        training_start="2024-01-01T00:00:00+00:00",
        training_end="2024-02-01T00:00:00+00:00",
        feature_set=["hour", "lag_1h"],
        forecast_horizon_hours=1,
        metrics={"mae": 0.4, "rmse": 0.5},
        dataset_hash="abc123",
        trained_at="2026-09-14T10:00:00+00:00",
        baseline_comparison={"persistence_mae": 0.6, "relative_improvement_pct": 33.3},
    )
    registry.save_metadata(metadata)
    assert registry.get("ridge-v1") == metadata
