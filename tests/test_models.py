import numpy as np

from energy_twin.models import PersistenceBaseline, SeasonalNaiveBaseline


def test_persistence_predicts_last_observation() -> None:
    model = PersistenceBaseline().fit(np.array([1.0, 2.0, 3.0]))
    assert model.predict(3).tolist() == [3.0, 3.0, 3.0]


def test_seasonal_naive_replays_last_period() -> None:
    history = np.arange(48, dtype=float)
    model = SeasonalNaiveBaseline(period=24).fit(history)
    assert model.predict(3).tolist() == [24.0, 25.0, 26.0]
