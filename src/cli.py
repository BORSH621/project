from __future__ import annotations

import argparse
from datetime import datetime

from src.forecast_grid import build_timeline, format_table, stub_forecast


def _parse_start(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("формат даты: ГГГГ-ММ-ДД ЧЧ:ММ") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="weatherlstm", description="WeatherLSTM: прогноз на 72 часа")
    sub = parser.add_subparsers(dest="command", required=True)
    fc = sub.add_parser("forecast", help="показать прогноз на 72 часа (шаг 3 часа)")
    fc.add_argument("--start", type=_parse_start, default=None,
                    help="время старта, ГГГГ-ММ-ДД ЧЧ:ММ (по умолчанию: сейчас)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "forecast":
        start = args.start or datetime.now().replace(minute=0, second=0, microsecond=0)
        rows = stub_forecast(build_timeline(start))
        print("ВНИМАНИЕ: значения — заглушка, модель LSTM ещё не подключена.")
        print(format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
