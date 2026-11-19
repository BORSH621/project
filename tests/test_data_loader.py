import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_csv


def make_raw(n=200):
    idx = pd.date_range("2024-01-01", periods=n, freq="3h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "дата": idx,
        "температура": np.round(10 + 8 * np.sin(np.arange(n) / 8) + rng.normal(0, 1, n), 1),
        "осадки": 0.0, "ветер": 3.0, "влажность": 70, "давление": 740,
    })


def test_load_csv_sets_datetime_index_and_sorts(tmp_path):
    p = tmp_path / "shuffled.csv"
    make_raw().sample(frac=1, random_state=1).to_csv(p, index=False)
    df = load_csv(p)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert len(df) == 200


def test_load_csv_reports_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    make_raw().drop(columns=["давление", "ветер"]).to_csv(p, index=False)
    with pytest.raises(KeyError, match="давление"):
        load_csv(p)


def test_load_csv_saves_copy_to_raw_folder(tmp_path):
    p = tmp_path / "raw.csv"
    make_raw().to_csv(p, index=False)
    out = tmp_path / "data" / "raw"
    load_csv(p, save_copy_to=out)
    assert (out / "raw.csv").exists()
