from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    row_count: int
    duplicate_timestamps: int
    missing_consumption: int
    irregular_intervals: int
    timezone_aware: bool

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "row_count": self.row_count,
            "duplicate_timestamps": self.duplicate_timestamps,
            "missing_consumption": self.missing_consumption,
            "irregular_intervals": self.irregular_intervals,
            "timezone_aware": self.timezone_aware,
        }


def assess_hourly_series(frame: pd.DataFrame) -> DataQualityReport:
    required = {"timestamp", "consumption_kwh"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    duplicate_count = int(timestamps.duplicated().sum())
    unique = timestamps.dropna().drop_duplicates().sort_values()
    deltas = unique.diff().dropna()
    irregular = int((deltas != pd.Timedelta(hours=1)).sum())
    tz = getattr(timestamps.dt, "tz", None)

    return DataQualityReport(
        row_count=len(frame),
        duplicate_timestamps=duplicate_count,
        missing_consumption=int(frame["consumption_kwh"].isna().sum()),
        irregular_intervals=irregular,
        timezone_aware=tz is not None,
    )


def normalise_hourly_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Return sorted, de-duplicated observations without inventing missing demand values."""
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result = result.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return result.reset_index(drop=True)
