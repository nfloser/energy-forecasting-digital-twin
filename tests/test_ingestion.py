from io import StringIO

import pandas as pd

from energy_twin.ingestion import parse_uci_power, parse_weather_payload


def test_uci_parser_aggregates_minute_power_to_hourly_energy() -> None:
    text = "Date;Time;Global_active_power\n01/01/2009;00:00:00;1.2\n01/01/2009;00:01:00;1.8\n"
    frame = parse_uci_power(StringIO(text), min_hourly_coverage=0.0)
    assert len(frame) == 1
    assert frame.iloc[0]["consumption_kwh"] == 0.05
    assert str(frame.iloc[0]["timestamp"].tz) == "UTC"


def test_weather_payload_is_normalised_to_utc() -> None:
    payload = {
        "hourly": {
            "time": ["2009-01-01T00:00", "2009-01-01T01:00"],
            "temperature_2m": [4.0, 3.5],
            "relative_humidity_2m": [80.0, 82.0],
        }
    }
    frame = parse_weather_payload(payload, timezone="Europe/Paris")
    assert list(frame.columns) == ["timestamp", "temperature_c", "humidity_pct"]
    assert frame["timestamp"].dt.tz is not None
