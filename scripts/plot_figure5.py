#!/usr/bin/env python3
"""Figure 5: Block-level statistical inference across 16 spatial blocks.

Panels:
    (a) Paired per-block MAE for each model, so the reader can see whether a
        model wins uniformly or only in a subset of blocks.
    (b) Bootstrap distribution of the mean paired MAE difference, resampled
        over spatial blocks. The 95% interval replaces a pixel-level test,
        which would inflate significance through spatial autocorrelation.

Usage:
    python scripts/plot_figure5.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (apply_style, block_mae, bootstrap_mean_ci, load_config,
                     model_style, report, require_file, resolve_input,
                     save_figure)

#: Reference model that every difference is measured against.
REFERENCE_MODEL = "rf_enh"

#: Models drawn in panel (a), in marker order.
DUMBBELL_MODELS = ["rf_old", "idw", "rf_enh"]

#: Models compared against the reference in panel (b).
COMPARISON_MODELS = ["rf_old", "lyzenga", "idw"]


def main() -> int:
    """Render Figure 5."""
    import matplotlib.pyplot as plt

    config = load_config()
    apply_style(config)

    table = pd.read_csv(require_file(
        resolve_input(config, "oof_predictions"),
        "Run: python scripts/build_oof_predictions.py",
    ))

    analysis = config["analysis"]
    n_resamples = int(analysis["bootstrap_resamples"])
    seed = int(analysis["bootstrap_seed"])

    reference_blocks, reference_mae = block_mae(table, REFERENCE_MODEL)
    n_blocks = reference_blocks.size

    block_curves = {}
    for key in set(DUMBBELL_MODELS) | set(COMPARISON_MODELS):
        blocks, values = block_mae(table, key)
        if not np.array_equal(blocks, reference_blocks):
            raise ValueError(f"Block identifiers differ for model '{key}'.")
        block_curves[key] = values

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.88, wspace=0.32)

    # --- Panel (a): Paired per-block MAE ---
    ax = axes[0]
    positions = np.arange(1, n_blocks + 1)

    for index in range(n_blocks):
        ax.plot([positions[index], positions[index]],
                [block_curves["rf_enh"][index], block_curves["rf_old"][index]],
                color="#95a5a6", linewidth=1.0, linestyle=":", zorder=1)

    marker_sizes = {"rf_old": 32, "idw": 30, "rf_enh": 36}
    for key in DUMBBELL_MODELS:
        entry = model_style(config, key)
        ax.scatter(positions, block_curves[key], color=entry["color"],
                   s=marker_sizes.get(key, 30), marker=entry["marker"],
                   label=entry["short_label"], zorder=4,
                   edgecolors="black", linewidth=0.5)

    ax.set_xlabel(f"Spatial block index (1-{n_blocks})")
    ax.set_ylabel(r"Block-level $\mathrm{MAE}$ (m)")
    ax.set_xticks(positions)
    ax.set_xlim(0.4, n_blocks + 0.6)
    ax.set_ylim(0, float(max(curve.max() for curve in block_curves.values())) * 1.15)
    ax.grid(axis="y", linestyle=":", zorder=1)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95,
              edgecolor="#bdc3c7", fancybox=False)
    ax.set_title("(a)", loc="left", fontsize=10, fontweight="bold", pad=8)

    # --- Panel (b): Bootstrap distribution of paired differences ---
    ax = axes[1]
    distributions = []
    colors = []
    labels = []

    for key in COMPARISON_MODELS:
        entry = model_style(config, key)
        difference = block_curves[key] - reference_mae
        rng = np.random.default_rng(seed)
        samples = rng.choice(difference, size=(n_resamples, difference.size), replace=True)
        distributions.append(samples.mean(axis=1))
        colors.append(entry["color"])
        labels.append(r"$\Delta\mathrm{MAE}$" + "\n" + f"({entry['short_label']})")

        mean_ci = bootstrap_mean_ci(difference, n_resamples=n_resamples, seed=seed)
        print(f"{key:<10} mean dMAE {difference.mean():+.2f} m | "
              f"95% CI [{mean_ci[0]:+.2f}, {mean_ci[1]:+.2f}]")

    parts = ax.violinplot(distributions, positions=range(1, len(distributions) + 1),
                          widths=0.55, showmeans=False, showmedians=False,
                          showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_alpha(0.65)

    for index, distribution in enumerate(distributions):
        median = float(np.median(distribution))
        low, high = np.percentile(distribution, [2.5, 97.5])
        ax.errorbar(index + 1, median, yerr=[[median - low], [high - median]],
                    fmt="o", color="black", ecolor="black", elinewidth=1.2,
                    capsize=3.5, capthick=1.0, markersize=4.5, zorder=5)
        ax.text(index + 1, high, f"[{low:.2f}, {high:.2f}]", ha="center",
                va="bottom", fontsize=7.2, fontweight="bold")

    ax.axhline(0, color="black", linestyle="--", linewidth=0.9, zorder=2, alpha=0.8)
    ax.set_ylabel(r"Mean paired $\mathrm{MAE}$ difference $\Delta_b$ (m)")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.4, len(labels) + 0.6)
    ax.grid(axis="y", linestyle=":", zorder=1)
    ax.set_title("(b)", loc="left", fontsize=10, fontweight="bold", pad=8)

    png_path, pdf_path = save_figure(fig, config, "Figure_5")
    report(png_path, pdf_path, f"blocks={n_blocks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
