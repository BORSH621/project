"""Загрузка исторических данных из CSV (задача T-02).

Код перенесён из ноутбука Synoptic.ipynb.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATE_COL = "дата"
RAW_COLUMNS = ["температура", "осадки", "ветер", "влажность", "давление"]


def load_csv(path: str | Path, save_copy_to: str | Path | None = None) -> pd.DataFrame:
    """Читает CSV, проверяет колонки, делает 'дата' индексом, сортирует по времени.

    save_copy_to - папка (например, data/raw), куда сохранить копию файла.
    """
    path = Path(path)
    df = pd.read_csv(path)
    missing = [c for c in [DATE_COL, *RAW_COLUMNS] if c not in df.columns]
    if missing:
        raise KeyError(f"В файле {path.name} не хватает колонок: {', '.join(missing)}")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).set_index(DATE_COL)
    if save_copy_to is not None:
        out = Path(save_copy_to)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / path.name)
    print(f"Загружено {len(df)} строк: {df.index.min()} .. {df.index.max()}")
    return df
