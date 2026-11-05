from __future__ import annotations

from datetime import datetime, timedelta

HORIZON_HOURS = 72
STEP_HOURS = 3
POINTS = HORIZON_HOURS // STEP_HOURS  # 24


def build_timeline(start: datetime,
                   horizon_hours: int = HORIZON_HOURS,
                   step_hours: int = STEP_HOURS) -> list[datetime]:
    """Возвращает моменты прогноза: start + step, start + 2*step, ... start + horizon."""
    if step_hours <= 0 or horizon_hours <= 0:
        raise ValueError("horizon_hours и step_hours должны быть больше 0")
    if horizon_hours % step_hours != 0:
        raise ValueError("horizon_hours должен делиться на step_hours")
    count = horizon_hours // step_hours
    return [start + timedelta(hours=step_hours * i) for i in range(1, count + 1)]


def stub_forecast(timeline: list[datetime]) -> list[dict]:
    """Заглушка: постоянные значения. Реальную модель подключит T-10."""
    return [
        {"time": t, "temperature_c": 15.0, "humidity_pct": 60.0, "pressure_hpa": 1013.0}
        for t in timeline
    ]


def format_table(rows: list[dict]) -> str:
    """Текстовая таблица прогноза."""
    header = f"{'Время':<17} {'Темп., °C':>10} {'Влажн., %':>10} {'Давл., гПа':>11}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['time']:%Y-%m-%d %H:%M} "
            f"{r['temperature_c']:>10.1f} {r['humidity_pct']:>10.0f} {r['pressure_hpa']:>11.0f}"
        )
    return "\n".join(lines)
