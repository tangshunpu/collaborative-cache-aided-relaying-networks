#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import DEFAULT_OUT_DIR
from cache_outage.experiments import parameter_sweep


parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
args = parser.parse_args()

parameter_sweep(args.out_dir, "tau")
