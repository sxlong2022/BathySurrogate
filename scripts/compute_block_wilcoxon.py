#!/usr/bin/env python3
"""Block-level statistical inference for every model comparison.

The independent unit is the spatial block, not the pixel. A pixel-level test
would inflate significance through spatial autocorrelation, so each
comparison is reduced to one paired difference per block before testing.

Reported per comparison:
    mean and median paired MAE difference, Wilcoxon signed-rank statistic and
    p-value, and a percentile bootstrap confidence interval over blocks.

Usage:
    python scripts/compute_block_wilcoxon.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (block_mae, bootstrap_mean_ci, load_config, model_style,
                     require_file, resolve_input, resolve_output)

#: Reference model that every comparison is measured against.
REFERENCE_MODEL = "rf_enh"

#: Baselines compared against the reference model.
BASELINE_MODELS = ["rf_old", "lyzenga", "stumpf", "idw"]


def main() -> int:
    """Print the block-level comparison table."""
    from scipy.stats import wilcoxon

    config = load_config()

    table = pd.read_csv(require_file(
        resolve_input(config, "oof_predictions"),
        "Run: python scripts/build_oof_predictions.py",
    ))

    n_resamples = int(config["analysis"]["bootstrap_resamples"])
    seed = int(config["analysis"]["bootstrap_seed"])

    reference_blocks, reference_mae = block_mae(table, REFERENCE_MODEL)
    n_blocks = reference_blocks.size
    n_points = len(table)

    print(f"Soundings: {n_points} | spatial blocks: {n_blocks}")
    print("=" * 108)
    print(f"{'comparison':<40} {'mean dMAE':>10} {'median':>9} "
          f"{'W':>8} {'p':>12} {'95% bootstrap CI':>22}")
    print("-" * 108)

    rows = []
    for baseline in BASELINE_MODELS:
        blocks, values = block_mae(table, baseline)
        if not np.array_equal(blocks, reference_blocks):
            raise ValueError(f"Block identifiers differ for model '{baseline}'.")

        difference = values - reference_mae
        statistic, p_value = wilcoxon(difference, alternative="greater")
        low, high = bootstrap_mean_ci(difference, n_resamples=n_resamples, seed=seed)

        label = (f"{model_style(config, REFERENCE_MODEL)['short_label']} vs "
                 f"{model_style(config, baseline)['short_label']}")
        print(f"{label:<40} {difference.mean():>10.2f} {np.median(difference):>9.2f} "
              f"{statistic:>8.1f} {p_value:>12.4e} "
              f"{'[' + f'{low:.2f}, {high:.2f}' + ']':>22}")

        rows.append({
            "comparison": label,
            "mean_dmae": float(difference.mean()),
            "median_dmae": float(np.median(difference)),
            "wilcoxon_w": float(statistic),
            "p_value": float(p_value),
            "ci_low": low,
            "ci_high": high,
        })

    print("-" * 108)
    print("Positive dMAE means the reference model has the lower error.")

    output_path = os.path.join(
        resolve_output(config, "validation_dir"),
        "block_wilcoxon_summary.csv",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\n[OK] Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
