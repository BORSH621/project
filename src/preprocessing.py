"""Признаки, разбиение по времени и нормализация (задача T-04).

Код перенесён из ноутбука Synoptic.ipynb. Отличие от ноутбука: разбиение на
train/test выполняется по времени, а не случайно (случайное разбиение даёт утечку:
соседние 3-часовые строки почти одинаковы, и тест оказывается не независимым).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "температура", "осадки", "ветер", "влажность", "давление",
    "час_sin", "час_cos", "день_года_sin", "день_года_cos", "темп_3ч_назад",
]
TARGET = "темп_завтра"


# ------------------- T-04: признаки, разбиение, нормализация ----------------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Достраивает циклические признаки часа и дня года, если их нет."""
    df = df.copy()
    idx = df.index
    if "час_sin" not in df.columns or "час_cos" not in df.columns:
        df["час_sin"] = np.sin(2 * np.pi * idx.hour / 24)
        df["час_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    if "день_года_sin" not in df.columns or "день_года_cos" not in df.columns:
        days_in_year = 365 + idx.is_leap_year.astype(int)
        df["день_года_sin"] = np.sin(2 * np.pi * idx.dayofyear / days_in_year)
        df["день_года_cos"] = np.cos(2 * np.pi * idx.dayofyear / days_in_year)
    if "темп_3ч_назад" not in df.columns:
        df["темп_3ч_назад"] = df["температура"].shift(1)
    if TARGET not in df.columns:
        df[TARGET] = df["температура"].shift(-8)  # температура через 24 ч (8 шагов по 3 ч)
    return df


def make_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Матрица признаков X и вектор цели y без строк с пропусками."""
    df = prepare_features(df).dropna(subset=FEATURES + [TARGET])
    return df[FEATURES].to_numpy(dtype=float), df[TARGET].to_numpy(dtype=float)


def split_by_time(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит таблицу по времени: первые train_frac строк - обучение, остальные - тест."""
    if not 0 < train_frac < 1:
        raise ValueError("train_frac должен быть между 0 и 1")
    df = df.sort_index()
    cut = int(len(df) * train_frac)
    return df.iloc[:cut], df.iloc[cut:]


@dataclass
class Normalizer:
    """Стандартизация (z-score). Параметры считаются ТОЛЬКО по обучающей части."""
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, X_train: np.ndarray, y_train: np.ndarray) -> "Normalizer":
        x_std = X_train.std(axis=0)
        x_std[x_std == 0] = 1.0                     # константный признак: не делим на ноль
        y_std = float(y_train.std()) or 1.0
        return cls(X_train.mean(axis=0), x_std, float(y_train.mean()), y_std)

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / self.x_std

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self.y_std + self.y_mean

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(),
                "y_mean": self.y_mean, "y_std": self.y_std, "features": FEATURES}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.array(d["x_mean"]), np.array(d["x_std"]), d["y_mean"], d["y_std"])
