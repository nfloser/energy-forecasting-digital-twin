from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from energy_twin.domain import EnergyObservation, WeatherObservation


def test_energy_observation_rejects_negative_consumption() -> None:
    with pytest.raises(ValidationError):
        EnergyObservation(timestamp=datetime.now(UTC), consumption_kwh=-0.1)


def test_weather_observation_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValidationError):
        WeatherObservation(timestamp=datetime(2026, 1, 1), temperature_c=8.0, humidity_pct=80.0)
