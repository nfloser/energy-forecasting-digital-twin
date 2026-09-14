import pandas as pd

from energy_twin.provenance import dataframe_sha256


def test_dataset_hash_changes_when_values_change() -> None:
    a = pd.DataFrame({"timestamp": ["2024-01-01T00:00:00Z"], "consumption_kwh": [1.0]})
    b = pd.DataFrame({"timestamp": ["2024-01-01T00:00:00Z"], "consumption_kwh": [2.0]})
    assert dataframe_sha256(a) != dataframe_sha256(b)
