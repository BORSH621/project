import numpy as np
import pandas as pd
import pytest

from src.metrics import (compare_with_baseline, evaluate, mae,
                         persistence_forecast, rmse)


def test_metrics_known_values():
    assert mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    assert set(evaluate([1, 2], [1, 3])) == {"MAE", "MSE", "RMSE"}


def test_persistence_uses_current_temperature():
    df = pd.DataFrame({"температура": [1.0, 2.0, 3.0]})
    assert np.array_equal(persistence_forecast(df), [1.0, 2.0, 3.0])


def test_compare_with_baseline_flags_better_and_worse_model():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    good = compare_with_baseline(y, y + 0.1, y + 1.0)
    bad = compare_with_baseline(y, y + 2.0, y + 1.0)
    assert good["better_than_baseline"] and good["rmse_gain_pct"] > 0
    assert not bad["better_than_baseline"]
