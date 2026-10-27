#!/usr/bin/env python3
"""Run the harmonized ablation experiment and render Figure 3 in one step.

This is a thin orchestration wrapper. It keeps the two stages separate so
either one can be re-run in isolation:

1. ``scripts/harmonize_ablation_experiment.py`` retrains the ablation ladder
   and writes ``intermediate_results/ablation_results.npz``.
2. ``scripts/plot_figure3.py`` renders the figure from that archive.

Usage:
    python scripts/generate_ablation_data_and_plot.py
"""

from __future__ import annotations

import os
import subprocess
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

STAGES = [
    ("Harmonized ablation experiment", "harmonize_ablation_experiment.py"),
    ("Figure 3 (feature ablation ladder)", "plot_figure3.py"),
]


def run_stage(label: str, script: str) -> None:
    """Execute one pipeline stage in a child interpreter.

    Raises:
        subprocess.CalledProcessError: When the stage exits non-zero.
    """
    path = os.path.join(SCRIPTS_DIR, script)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline stage not found: {path}")

    print(f"\n---> {label}: {script}", flush=True)
    subprocess.run([sys.executable, path], check=True)


def main() -> int:
    """Entry point."""
    print("=" * 70)
    print("ABLATION PIPELINE: EXPERIMENT + FIGURE 3")
    print("=" * 70)

    for label, script in STAGES:
        run_stage(label, script)

    print("\n" + "=" * 70)
    print("[OK] Ablation results harmonized and Figure 3 rendered.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
