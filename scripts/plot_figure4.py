#!/usr/bin/env python3
"""Figure 4: Multi-model out-of-fold prediction scatter comparison.

Four panels compare the classical baselines and the two Random Forest
configurations against chart soundings. Statistics are computed from the
SSOT table at run time, so the panels cannot drift from the reported
validation numbers.

Usage:
    python scripts/plot_figure4.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (apply_style, calc_metrics, load_config, model_style,
                     report, require_file, resolve_input, save_figure)

#: Models shown in Figure 4, in panel order.
PANEL_MODELS = ["lyzenga", "idw", "rf_old", "rf_enh"]
PANEL_TAGS = ["(a)", "(b)", "(c)", "(d)"]


def main() -> int:
    """Render Figure 4."""
    import matplotlib.pyplot as plt

    config = load_config()
    apply_style(config)

    table = pd.read_csv(require_file(
        resolve_input(config, "oof_predictions"),
        "Run: python scripts/build_oof_predictions.py",
    ))

    limit = float(config["domain"]["scatter_limit"])
    true_depth = table["true_depth"].values

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.0))
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.09, top=0.95,
                        wspace=0.25, hspace=0.30)

    reference = np.linspace(0, limit, 100)

    for ax, key, tag in zip(axes.flat, PANEL_MODELS, PANEL_TAGS):
        entry = model_style(config, key)
        prediction = table[key].values

        stats = calc_metrics(true_depth, prediction)

        ax.scatter(true_depth, prediction, color=entry["color"], alpha=0.55,
                   s=18, edgecolors="black", linewidth=0.3, zorder=3)
        ax.plot([0, limit], [0, limit], color="black", linestyle="--",
                linewidth=0.9, zorder=2, label="1:1 reference")

        finite = np.isfinite(true_depth) & np.isfinite(prediction)
        if np.count_nonzero(finite) > 1:
            slope, intercept = np.polyfit(true_depth[finite], prediction[finite], 1)
            ax.plot(reference, slope * reference + intercept, color=entry["color"],
                    linestyle="-", linewidth=1.3, zorder=4, label="Linear fit")

        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", zorder=1)

        ax.set_xlabel("Nautical chart sounding depth (m)")
        ax.set_ylabel("Predicted water depth (m)")
        ax.set_title(f"{tag} {entry['label']}", loc="left", fontsize=9.5,
                     fontweight="bold", pad=7)

        annotation = (
            f"$\\mathrm{{R}}^2 = {stats['r2']:+.3f}$\n"
            f"$\\mathrm{{r}} = {stats['corr']:+.3f}$\n"
            f"$\\mathrm{{RMSE}} = {stats['rmse']:.2f}\\,\\mathrm{{m}}$\n"
            f"$\\mathrm{{MAE}} = {stats['mae']:.2f}\\,\\mathrm{{m}}$"
        )
        ax.text(0.05, 0.95, annotation, transform=ax.transAxes, fontsize=7.5,
                verticalalignment="top", horizontalalignment="left",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          alpha=0.88, edgecolor="#bdc3c7", linewidth=0.6),
                zorder=5)

        print(f"{key:<10} R2 {stats['r2']:+.3f} | r {stats['corr']:+.3f} | "
              f"RMSE {stats['rmse']:5.2f} m | MAE {stats['mae']:5.2f} m")

    png_path, pdf_path = save_figure(fig, config, "Figure_4")
    report(png_path, pdf_path, f"models={len(PANEL_MODELS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
