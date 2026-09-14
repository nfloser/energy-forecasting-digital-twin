from __future__ import annotations

import json
from io import TextIOBase
from pathlib import Path
from typing import IO, Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

UCI_POWER_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def parse_uci_power(
    source: str | Path | IO[str], *, min_hourly_coverage: float = 0.8
) -> pd.DataFrame:
    """Parse UCI minute-level active power and aggregate it to hourly kWh.

    Global_active_power is an average kW value for each minute. Dividing by 60 and
    summing therefore yields kWh. Hours below the requested observed-minute coverage
    are discarded rather than silently imputed.
    """
    frame = pd.read_csv(
        source,
        sep=";",
        usecols=["Date", "Time", "Global_active_power"],
        na_values=["?", ""],
        low_memory=False,
    )
    local = pd.to_datetime(
        frame["Date"] + " " + frame["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    timestamps = local.dt.tz_localize(
        "Europe/Paris", ambiguous="infer", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")
    power_kw = pd.to_numeric(frame["Global_active_power"], errors="coerce")
    minute = pd.DataFrame({"timestamp": timestamps, "power_kw": power_kw}).dropna(
        subset=["timestamp"]
    )
    minute = minute.set_index("timestamp")
    hourly = minute.resample("h").agg(power_kw_sum=("power_kw", "sum"), samples=("power_kw", "count"))
    hourly["coverage"] = hourly["samples"] / 60.0
    hourly["consumption_kwh"] = hourly["power_kw_sum"] / 60.0
    hourly = hourly[hourly["coverage"] >= min_hourly_coverage]
    return hourly.reset_index()[["timestamp", "consumption_kwh"]]


def parse_weather_payload(payload: dict[str, Any], *, timezone: str) -> pd.DataFrame:
    hourly = payload.get("hourly") or {}
    required = {"time", "temperature_2m", "relative_humidity_2m"}
    if not required.issubset(hourly):
        raise ValueError("weather payload is missing required hourly fields")
    local = pd.to_datetime(hourly["time"], errors="raise")
    timestamps = local.tz_localize(
        timezone, ambiguous="infer", nonexistent="shift_forward"
    ).tz_convert("UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_c": pd.to_numeric(hourly["temperature_2m"], errors="coerce"),
            "humidity_pct": pd.to_numeric(hourly["relative_humidity_2m"], errors="coerce"),
        }
    )


def fetch_historical_weather(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str = "Europe/Paris",
    timeout_seconds: int = 60,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": timezone,
    }
    url = f"{OPEN_METEO_ARCHIVE_URL}?{urlencode(params)}"
    with urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310 - fixed HTTPS endpoint
        payload = json.load(response)
    return parse_weather_payload(payload, timezone=timezone)


def merge_energy_weather(energy: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    merged = energy.merge(weather, on="timestamp", how="left", validate="one_to_one")
    return merged.sort_values("timestamp").reset_index(drop=True)
