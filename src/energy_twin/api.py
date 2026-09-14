from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from energy_twin.registry import ModelMetadata, ModelRegistry


def _read_json(path: Path) -> object:
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"missing artefact: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def create_app(data_dir: Path | str = "data") -> FastAPI:
    root = Path(data_dir)
    registry = ModelRegistry(root / "models")
    app = FastAPI(
        title="Energy Forecasting Digital Twin API",
        version="1.0.0",
        description=(
            "Traceable short-term energy forecasts with baseline comparison, "
            "model metadata and provenance."
        ),
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        if not (root / "forecast.json").exists() or not (root / "evaluation.json").exists():
            raise HTTPException(status_code=503, detail="forecast artefacts are not ready")
        return {"status": "ready"}

    @app.get("/current")
    def current() -> dict[str, object]:
        path = root / "processed" / "hourly.csv"
        if not path.exists():
            raise HTTPException(status_code=503, detail="processed energy data are not ready")
        frame = pd.read_csv(path)
        if frame.empty:
            raise HTTPException(status_code=503, detail="processed energy data are empty")
        return frame.iloc[-1].to_dict()

    @app.get("/history")
    def history(limit: int = Query(default=168, ge=1, le=5000)) -> list[dict[str, object]]:
        path = root / "processed" / "hourly.csv"
        if not path.exists():
            raise HTTPException(status_code=503, detail="processed energy data are not ready")
        frame = pd.read_csv(path).tail(limit)
        return frame.to_dict(orient="records")

    @app.get("/forecast")
    def forecast() -> object:
        return _read_json(root / "forecast.json")

    @app.get("/models", response_model=list[ModelMetadata])
    def models() -> list[ModelMetadata]:
        return registry.list()

    @app.get("/models/{model_id}", response_model=ModelMetadata)
    def model(model_id: str) -> ModelMetadata:
        try:
            return registry.get(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model not found") from exc

    @app.get("/metrics")
    def metrics() -> object:
        return _read_json(root / "evaluation.json")

    @app.get("/provenance")
    def provenance() -> object:
        return _read_json(root / "provenance.json")

    return app


app = create_app()
