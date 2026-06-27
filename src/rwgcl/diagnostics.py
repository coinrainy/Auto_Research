"""Diagnostic scaffold writers."""

from __future__ import annotations

from pathlib import Path

from .logging_utils import ensure_dir, write_csv


AVAILABLE_DIAGNOSTICS = {
    "shuffled_reliability",
    "false_negative_mass",
    "view_consistency",
}


def write_placeholder_diagnostic(
    output_dir: str | Path,
    run_id: str,
    diagnostic: str,
) -> Path:
    if diagnostic not in AVAILABLE_DIAGNOSTICS:
        raise ValueError(f"Unknown diagnostic: {diagnostic}")
    out_dir = ensure_dir(output_dir)
    out_path = out_dir / f"{diagnostic}.csv"
    rows = [
        {
            "run_id": run_id,
            "diagnostic": diagnostic,
            "status": "scaffold_only",
            "value": "",
            "notes": "Diagnostic schema placeholder; compute logic not implemented yet.",
        }
    ]
    write_csv(out_path, rows, ["run_id", "diagnostic", "status", "value", "notes"])
    return out_path
