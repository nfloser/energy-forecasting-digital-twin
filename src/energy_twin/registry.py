import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ModelMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    model_type: str
    training_start: str
    training_end: str
    feature_set: list[str]
    forecast_horizon_hours: int
    metrics: dict[str, float | None]
    dataset_hash: str
    trained_at: str
    baseline_comparison: dict[str, float]


class ModelRegistry:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, metadata: ModelMetadata) -> Path:
        path = self.root / f"{metadata.model_id}.json"
        payload = json.dumps(metadata.model_dump(), indent=2, sort_keys=True)
        path.write_text(payload, encoding="utf-8")
        return path

    def get(self, model_id: str) -> ModelMetadata:
        path = self.root / f"{model_id}.json"
        if not path.exists():
            raise KeyError(model_id)
        return ModelMetadata.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[ModelMetadata]:
        return [
            ModelMetadata.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.root.glob("*.json"))
        ]
