#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import DEFAULT_OUT_DIR
from cache_outage.experiments import analysis_power_sweep


parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
parser.add_argument("--samples", type=int, default=200_000)
args = parser.parse_args()

analysis_power_sweep(args.out_dir, "bs", samples=args.samples)
