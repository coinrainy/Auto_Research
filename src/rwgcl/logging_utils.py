"""Logging and result-file helpers."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_run_id(method: str, dataset: str, seed: int, split_index: int | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = f"seed{seed}"
    if split_index is not None:
        suffix = f"{suffix}_split{split_index}"
    return f"{stamp}_{method}_{dataset}_{suffix}"


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_yaml(path: str | Path, data: dict) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def write_json(path: str | Path, data: dict) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: str | Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv(path: str | Path, row: dict, fieldnames: list[str]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    exists = target.exists()
    if exists:
        with target.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            old_fieldnames = list(reader.fieldnames or [])
            if old_fieldnames and any(field not in old_fieldnames for field in fieldnames):
                merged_fieldnames = old_fieldnames + [
                    field for field in fieldnames if field not in old_fieldnames
                ]
                old_rows = list(reader)
                with target.open("w", newline="", encoding="utf-8") as rewrite_handle:
                    writer = csv.DictWriter(rewrite_handle, fieldnames=merged_fieldnames)
                    writer.writeheader()
                    for old_row in old_rows:
                        writer.writerow({field: old_row.get(field, "") for field in merged_fieldnames})
                fieldnames = merged_fieldnames
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
