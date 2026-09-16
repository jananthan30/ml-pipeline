#!/usr/bin/env python3
"""Regenerate canary.csv deterministically. Usage: python3 make_canary.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
import mlpipeline_canary as canary  # noqa: E402

out = Path(__file__).with_name("canary.csv")
canary.write_csv(out)
print(f"wrote {out} ({sum(1 for _ in open(out)) - 1} rows, honest ceiling {canary.CEILING:.2f})")
