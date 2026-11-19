"""Метрики качества и базовая модель персистентности (задача T-08)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, MSE и RMSE в исходных единицах (°C)."""
    return {"MAE": mae(y_true, y_pred), "MSE": mse(y_true, y_pred), "RMSE": rmse(y_true, y_pred)}


def persistence_forecast(df: pd.DataFrame, current_col: str = "температура") -> np.ndarray:
    """Базовая модель: «завтра как сегодня», то есть прогноз = текущая температура."""
    return df[current_col].to_numpy(dtype=float)


def compare_with_baseline(y_true: np.ndarray, y_pred: np.ndarray, y_baseline: np.ndarray) -> dict:
    """Метрики модели и базовой модели и выигрыш по RMSE в процентах."""
    model, base = evaluate(y_true, y_pred), evaluate(y_true, y_baseline)
    gain = (base["RMSE"] - model["RMSE"]) / base["RMSE"] * 100 if base["RMSE"] else 0.0
    return {"model": model, "baseline": base, "rmse_gain_pct": gain, "better_than_baseline": gain > 0}
