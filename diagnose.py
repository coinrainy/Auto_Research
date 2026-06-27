#!/usr/bin/env python3
"""Diagnostic entry point for RW-GCL experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rwgcl.diagnostics import AVAILABLE_DIAGNOSTICS, write_placeholder_diagnostic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scaffold diagnostics.")
    parser.add_argument("--run_id", "--run-id", dest="run_id", required=True, help="Run identifier.")
    parser.add_argument(
        "--diagnostics",
        nargs="+",
        default=sorted(AVAILABLE_DIAGNOSTICS),
        help="Diagnostics to run.",
    )
    parser.add_argument("--out-dir", default="results/diagnostics", help="Diagnostic output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for diagnostic in args.diagnostics:
        path = write_placeholder_diagnostic(args.out_dir, args.run_id, diagnostic)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
