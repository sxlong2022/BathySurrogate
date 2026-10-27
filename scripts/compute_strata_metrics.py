#!/usr/bin/env python3
"""Error stratification by depth and distance-to-nearest-sounding.

Produces the strata tables reported in the manuscript. Each stratum lists the
sample count and the RMSE / MAE of every model.

Usage:
    python scripts/compute_strata_metrics.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (calc_metrics, distance_km, load_config, model_style,
                     require_file, resolve_input)

#: Models reported in both stratification tables.
MODEL_KEYS = ["rf_enh", "idw", "rf_old", "lyzenga"]


def print_table(title: str, strata: list, table: pd.DataFrame,
                config: dict) -> None:
    """Print one stratification table.

    Args:
        title: Section heading.
        strata: List of ``(label, boolean_mask)`` pairs.
        table: OOF prediction table.
        config: Parsed pipeline configuration.
    """
    true_depth = table["true_depth"].values

    header = f"{'stratum':<24} {'n':>5}"
    for key in MODEL_KEYS:
        header += f" | {model_style(config, key)['short_label']:<18}"
    print(title)
    print(header)
    print("-" * len(header))

    for label, mask in strata:
        count = int(np.count_nonzero(mask))
        line = f"{label:<24} {count:>5}"
        for key in MODEL_KEYS:
            stats = calc_metrics(true_depth[mask], table[key].values[mask])
            line += f" | MAE {stats['mae']:5.2f} / RMSE {stats['rmse']:5.2f}"
        print(line)
    print()


def main() -> int:
    """Print both stratification tables."""
    config = load_config()

    table = pd.read_csv(require_file(
        resolve_input(config, "oof_predictions"),
        "Run: python scripts/build_oof_predictions.py",
    ))

    true_depth = table["true_depth"].values
    distance = distance_km(config, table["nn_dist"].values)

    strata_config = config["strata"]
    shallow_max = float(strata_config["shallow_max"])
    intermediate_max = float(strata_config["intermediate_max"])
    near_max = float(strata_config["near_max_km"])
    transition_max = float(strata_config["transition_max_km"])

    depth_strata = [
        (f"Shallow (0-{shallow_max:.0f} m)", true_depth <= shallow_max),
        (f"Intermediate ({shallow_max:.0f}-{intermediate_max:.0f} m)",
         (true_depth > shallow_max) & (true_depth <= intermediate_max)),
        (f"Deep (>{intermediate_max:.0f} m)", true_depth > intermediate_max),
    ]

    distance_strata = [
        (f"Near (<{near_max:.1f} km)", distance < near_max),
        (f"Transition ({near_max:.1f}-{transition_max:.1f} km)",
         (distance >= near_max) & (distance <= transition_max)),
        (f"Far (>{transition_max:.1f} km)", distance > transition_max),
    ]

    print_table("=== DEPTH STRATA (MAE / RMSE in metres) ===",
                depth_strata, table, config)
    print_table("=== DISTANCE STRATA (operational regimes) ===",
                distance_strata, table, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
