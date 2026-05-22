#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cache_outage.experiments import run_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all reproducible Python experiments for the cache-aided relaying paper."
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--samples",
        type=int,
        default=200_000,
        help="Monte Carlo samples per mode for the analytical verification figures.",
    )
    args = parser.parse_args()
    run_all(args.out_dir, samples=args.samples)
    print(f"Wrote figures and CSV files to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
