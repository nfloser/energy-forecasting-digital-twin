from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _TimestampedModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value


class EnergyObservation(_TimestampedModel):
    consumption_kwh: float = Field(ge=0)


class WeatherObservation(_TimestampedModel):
    temperature_c: float
    humidity_pct: float = Field(ge=0, le=100)


class BuildingMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    building_id: str
    building_type: str = "residential"
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
