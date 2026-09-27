"""Command-line interface for JSONL evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .evaluation import evaluate_cases


def read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            yield item


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Arabic AI predictions stored as JSONL."
    )
    parser.add_argument("input", type=Path, help="JSONL file containing reference and prediction")
    parser.add_argument("--details", action="store_true", help="include per-case metrics")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate_cases(read_jsonl(args.input))
    print(json.dumps(summary.to_dict(args.details), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
