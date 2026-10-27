#!/usr/bin/env python3
"""Figure 3: Feature ablation ladder for the Enhanced Random Forest surrogate.

Panels:
    (a) Coefficient of determination across the five ablation tiers.
    (b) RMSE and MAE decay along the same ladder.
    (c) Distribution of per-block MAE, showing that the improvement is not
        driven by a single spatial block.

Reads ``intermediate_results/ablation_results.npz`` produced by
``scripts/harmonize_ablation_experiment.py``.

Usage:
    python scripts/plot_figure3.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (apply_style, calc_metrics, load_config, report,
                     require_file, resolve_input, save_figure)


def main() -> int:
    """Render Figure 3."""
    import matplotlib.pyplot as plt

    config = load_config()
    apply_style(config)

    archive_path = require_file(
        resolve_input(config, "ablation_results"),
        "Run: python scripts/harmonize_ablation_experiment.py",
    )
    archive = np.load(archive_path)

    true_depth = archive["true_depth"]
    block_ids = archive["block_ids"]
    unique_blocks = np.unique(block_ids)

    tiers = config["ablation"]["tiers"]
    names = [tier["name"] for tier in tiers]
    labels = [f"{tier['name']} ({tier['n_features']})" for tier in tiers]

    missing = [name for name in names if name not in archive.files]
    if missing:
        raise KeyError(f"Ablation archive is missing tiers: {missing}")

    r2_values, rmse_values, mae_values, block_mae_values = [], [], [], []
    for name in names:
        prediction = archive[name]
        stats = calc_metrics(true_depth, prediction)
        r2_values.append(stats["r2"])
        rmse_values.append(stats["rmse"])
        mae_values.append(stats["mae"])

        per_block = np.empty(unique_blocks.size, dtype=float)
        for index, block in enumerate(unique_blocks):
            mask = block_ids == block
            per_block[index] = float(np.mean(np.abs(prediction[mask] - true_depth[mask])))
        block_mae_values.append(per_block)

    print("Ablation tiers")
    for name, r2, rmse, mae in zip(names, r2_values, rmse_values, mae_values):
        print(f"  {name:<12} R2 {r2:+.3f} | RMSE {rmse:5.2f} m | MAE {mae:5.2f} m")

    # A muted-to-dark ramp conveys the progression along the ladder.
    palette = ["#7f8c8d", "#2980b9", "#16a085", "#d35400", "#2c3e50"]
    bar_colors = palette[: len(names)]

    positions = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.26, top=0.88, wspace=0.36)

    # --- Panel (a): R2 progression ---
    ax = axes[0]
    bars = ax.bar(positions, r2_values, color=bar_colors, width=0.55,
                  edgecolor="black", linewidth=0.6, zorder=3)
    ax.axhline(0, color="#c0392b", linestyle="--", linewidth=0.8,
               zorder=2, alpha=0.85)
    ax.set_ylabel(r"Coefficient of determination ($\mathrm{R}^2$)")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylim(-0.40, 0.75)
    ax.grid(axis="y", linestyle=":", zorder=1)

    for bar, value in zip(bars, r2_values):
        offset = 0.035 if value >= 0 else -0.09
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.2f}",
                ha="center", va="bottom" if value >= 0 else "top",
                fontsize=7.2, fontweight="bold")

    ax.set_title("(a)", loc="left", fontsize=10, fontweight="bold", pad=8)

    # --- Panel (b): RMSE and MAE decay ---
    ax = axes[1]
    width = 0.34
    rmse_bars = ax.bar(positions - width / 2, rmse_values, width,
                       label=r"$\mathrm{RMSE}$", color="#2980b9",
                       edgecolor="black", linewidth=0.6, zorder=3)
    mae_bars = ax.bar(positions + width / 2, mae_values, width,
                      label=r"$\mathrm{MAE}$", color="#e74c3c",
                      edgecolor="black", linewidth=0.6, zorder=3)

    ax.set_ylabel(r"Error metrics ($\mathrm{RMSE}$ and $\mathrm{MAE}$) (m)")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, max(rmse_values + mae_values) * 1.35)
    ax.grid(axis="y", linestyle=":", zorder=1)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95,
              edgecolor="#bdc3c7", fancybox=False)

    for bar in list(rmse_bars) + list(mae_bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=6.8)

    ax.set_title("(b)", loc="left", fontsize=10, fontweight="bold", pad=8)

    # --- Panel (c): Block-level MAE distribution ---
    ax = axes[2]
    box = ax.boxplot(block_mae_values, positions=positions, widths=0.48,
                     patch_artist=True,
                     boxprops=dict(facecolor="#ecf0f1", edgecolor="black", linewidth=0.7),
                     medianprops=dict(color="black", linewidth=1.2),
                     whiskerprops=dict(color="black", linewidth=0.7),
                     capprops=dict(color="black", linewidth=0.7),
                     flierprops=dict(marker="o", markerfacecolor="#e74c3c",
                                     markersize=3, markeredgecolor="none", alpha=0.6),
                     zorder=3)

    for patch, color in zip(box["boxes"], bar_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.set_ylabel(r"Block-level $\mathrm{MAE}$ ($n_{\mathrm{blocks}}=%d$) (m)"
                  % unique_blocks.size)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, float(np.max(np.concatenate(block_mae_values))) * 1.25)
    ax.grid(axis="y", linestyle=":", zorder=1)

    ax.set_title("(c)", loc="left", fontsize=10, fontweight="bold", pad=8)

    png_path, pdf_path = save_figure(fig, config, "Figure_3")
    report(png_path, pdf_path, f"tiers={len(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
