"""Evaluation summary helpers."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def read_metric_rows(path: str | Path) -> list[dict]:
    metric_path = Path(path)
    if not metric_path.exists():
        return []
    with metric_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def group_metric_rows(rows: list[dict], keys: list[str]) -> list[dict]:
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    summary = []
    for key_values, group_rows in grouped.items():
        out = {key: value for key, value in zip(keys, key_values)}
        out["runs"] = str(len(group_rows))
        statuses = sorted({row.get("status", "") for row in group_rows})
        out["statuses"] = "|".join(statuses)
        summary.append(out)
    return summary
