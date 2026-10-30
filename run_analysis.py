#!/usr/bin/env python3
"""Run the complete numerical analysis without generating figures or tables."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis"

STAGES = (
    ("model.py", "structural smoke tests"),
    ("run_basecase.py", "base-case analysis"),
    ("run_threshold.py", "risk, price, and relative-risk thresholds"),
    ("run_dsa.py", "deterministic sensitivity analysis"),
    ("run_psa.py", "probabilistic sensitivity analysis (10,000 draws)"),
    ("run_scenarios.py", "scenario analyses and budget impact"),
    ("validation.py", "internal and external validation"),
)


def main() -> int:
    (ROOT / "results").mkdir(exist_ok=True)
    started = time.monotonic()

    for script, label in STAGES:
        print(f"\n[{script}] {label}", flush=True)
        result = subprocess.run(
            [sys.executable, script],
            cwd=ANALYSIS_DIR,
            check=False,
        )
        if result.returncode:
            print(f"FAILED: {script} exited with status {result.returncode}", file=sys.stderr)
            return result.returncode

    elapsed = time.monotonic() - started
    print(f"\nAll numerical analyses completed in {elapsed:.1f} seconds.")
    print(f"Results: {ROOT / 'results'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
