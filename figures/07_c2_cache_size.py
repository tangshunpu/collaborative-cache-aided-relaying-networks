#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import DEFAULT_OUT_DIR
from cache_outage.experiments import parameter_sweep


parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
parser.add_argument("--paper-data", action="store_true")
args = parser.parse_args()

parameter_sweep(args.out_dir, "c2", use_paper_data=args.paper_data)
