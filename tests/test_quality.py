import pandas as pd

from energy_twin.quality import assess_hourly_series


def test_quality_report_detects_duplicate_missing_and_irregular_rows() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 00:00Z", "2024-01-01 01:00Z", "2024-01-01 01:00Z", "2024-01-01 03:00Z"]
            ),
            "consumption_kwh": [1.0, None, 2.0, 4.0],
        }
    )
    report = assess_hourly_series(frame)
    assert report.duplicate_timestamps == 1
    assert report.missing_consumption == 1
    assert report.irregular_intervals >= 1
