import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (FEATURES, TARGET, Normalizer, make_xy,
                               prepare_features, split_by_time)


def make_df(n=200):
    idx = pd.date_range("2024-01-01", periods=n, freq="3h", name="дата")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "температура": np.round(10 + 8 * np.sin(np.arange(n) / 8) + rng.normal(0, 1, n), 1),
        "осадки": 0.0, "ветер": 3.0, "влажность": 70, "давление": 740,
    }, index=idx)


def test_prepare_features_adds_all_columns():
    df = prepare_features(make_df())
    assert set(FEATURES + [TARGET]) <= set(df.columns)
    assert df["час_sin"].between(-1, 1).all()


def test_make_xy_drops_nan_rows():
    X, y = make_xy(make_df())
    assert X.shape[1] == len(FEATURES)
    assert len(X) == len(y) == 200 - 1 - 8   # 1 строка без лага, 8 без цели
    assert not np.isnan(X).any() and not np.isnan(y).any()


def test_split_by_time_has_no_overlap_and_keeps_order():
    train, test = split_by_time(make_df(), 0.7)
    assert len(train) + len(test) == 200
    assert train.index.max() < test.index.min()


def test_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        split_by_time(make_df(), 1.5)


def test_normalizer_fits_on_train_and_roundtrips(tmp_path):
    X, y = make_xy(make_df())
    norm = Normalizer.fit(X[:100], y[:100])
    assert np.allclose(norm.transform_x(X[:100]).mean(axis=0), 0, atol=1e-9)
    assert np.allclose(y[:100], norm.inverse_y(norm.transform_y(y[:100])))
    norm.save(tmp_path / "norm.json")
    again = Normalizer.load(tmp_path / "norm.json")
    assert np.allclose(again.x_mean, norm.x_mean) and again.y_std == norm.y_std


def test_normalizer_handles_constant_feature():
    X, y = make_xy(make_df())                    # колонка 'осадки' константна
    assert np.isfinite(Normalizer.fit(X, y).transform_x(X)).all()
