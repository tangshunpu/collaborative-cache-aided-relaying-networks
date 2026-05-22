#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import DEFAULT_OUT_DIR
from cache_outage.experiments import strategy_power_sweep


parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
parser.add_argument("--paper-data", action="store_true")
args = parser.parse_args()

strategy_power_sweep(args.out_dir, "relay", use_paper_data=args.paper_data)
