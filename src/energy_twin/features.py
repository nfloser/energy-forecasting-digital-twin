import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "temperature_c",
    "humidity_pct",
    "heating_degree_c",
    "cooling_degree_c",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_6h",
    "rolling_std_6h",
]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe features; every demand-derived feature is shifted into the past."""
    required = {"timestamp", "consumption_kwh", "temperature_c", "humidity_pct"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    result = frame.copy().sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(result["timestamp"], utc=True)
    result["timestamp"] = ts
    result["hour"] = ts.dt.hour
    result["day_of_week"] = ts.dt.dayofweek
    result["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    result["month"] = ts.dt.month
    result["heating_degree_c"] = np.maximum(18.0 - result["temperature_c"], 0.0)
    result["cooling_degree_c"] = np.maximum(result["temperature_c"] - 22.0, 0.0)

    demand = result["consumption_kwh"]
    result["lag_1h"] = demand.shift(1)
    result["lag_24h"] = demand.shift(24)
    result["lag_168h"] = demand.shift(168)
    past = demand.shift(1)
    result["rolling_mean_6h"] = past.rolling(6, min_periods=6).mean()
    result["rolling_std_6h"] = past.rolling(6, min_periods=6).std(ddof=0)
    return result


def modelling_rows(frame: pd.DataFrame) -> pd.DataFrame:
    featured = build_features(frame)
    return featured.dropna(subset=FEATURE_COLUMNS + ["consumption_kwh"]).reset_index(drop=True)
