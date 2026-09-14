from pathlib import Path

from fastapi.testclient import TestClient

from energy_twin.api import create_app


def test_health_and_readiness_are_separate(tmp_path: Path) -> None:
    client = TestClient(create_app(data_dir=tmp_path))
    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["detail"] == "forecast artefacts are not ready"


def test_models_endpoint_returns_persisted_metadata(tmp_path: Path) -> None:
    registry = tmp_path / "models"
    registry.mkdir()
    (registry / "demo.json").write_text(
        '{"model_id":"demo","model_type":"ridge","training_start":"2024-01-01T00:00:00+00:00",'
        '"training_end":"2024-02-01T00:00:00+00:00","feature_set":["hour"],'
        '"forecast_horizon_hours":1,"metrics":{"mae":0.5},"dataset_hash":"abc",'
        '"trained_at":"2026-09-14T10:00:00+00:00","baseline_comparison":{"relative_improvement_pct":1.0}}',
        encoding="utf-8",
    )
    client = TestClient(create_app(data_dir=tmp_path))
    response = client.get("/models")
    assert response.status_code == 200
    assert response.json()[0]["model_id"] == "demo"
