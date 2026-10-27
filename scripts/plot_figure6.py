#!/usr/bin/env python3
"""Figure 6: Error stratification by depth and distance-to-nearest-sounding.

Panels:
    (a) Mean absolute error within three depth strata.
    (b) Binned mean absolute error against the distance to the nearest
        training sounding, with the operational regimes annotated.

All values are computed from the SSOT table, so the reported strata match
the manuscript tables exactly.

Usage:
    python scripts/plot_figure6.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (apply_style, distance_km, load_config, model_style,
                     report, require_file, resolve_input, save_figure)

#: Models drawn in both panels, in legend order.
PANEL_MODELS = ["rf_enh", "idw", "rf_old", "lyzenga"]


def main() -> int:
    """Render Figure 6."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    config = load_config()
    apply_style(config)

    table = pd.read_csv(require_file(
        resolve_input(config, "oof_predictions"),
        "Run: python scripts/build_oof_predictions.py",
    ))

    strata = config["strata"]
    true_depth = table["true_depth"].values
    distance = distance_km(config, table["nn_dist"].values)

    shallow_max = float(strata["shallow_max"])
    intermediate_max = float(strata["intermediate_max"])
    near_max = float(strata["near_max_km"])
    transition_max = float(strata["transition_max_km"])

    depth_masks = [
        (true_depth <= shallow_max, f"Shallow\n(0-{shallow_max:.0f} m)"),
        ((true_depth > shallow_max) & (true_depth <= intermediate_max),
         f"Intermediate\n({shallow_max:.0f}-{intermediate_max:.0f} m)"),
        (true_depth > intermediate_max, f"Deep\n(>{intermediate_max:.0f} m)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.78, wspace=0.28)

    # --- Panel (a): Depth strata ---
    ax = axes[0]
    positions = np.arange(len(depth_masks))
    width = 0.20
    offsets = np.linspace(-1.5, 1.5, len(PANEL_MODELS)) * width

    for offset, key in zip(offsets, PANEL_MODELS):
        entry = model_style(config, key)
        values = [float(np.mean(np.abs(table[key].values[mask] - true_depth[mask])))
                  for mask, _ in depth_masks]
        bars = ax.bar(positions + offset, values, width, color=entry["color"],
                      edgecolor="black", linewidth=0.6, zorder=3)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.1f}", ha="center", va="bottom",
                    fontsize=6.2)
        print(f"{key:<10} depth strata MAE (m): "
              + ", ".join(f"{value:.2f}" for value in values))

    ax.set_ylabel(r"Mean absolute error ($\mathrm{MAE}$) (m)")
    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label in depth_masks])
    ax.set_ylim(0, float(max(
        np.mean(np.abs(table[key].values[mask] - true_depth[mask]))
        for key in PANEL_MODELS for mask, _ in depth_masks
    )) * 1.20)
    ax.grid(axis="y", linestyle=":", zorder=1)
    ax.text(0.03, 0.94, "(a)", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")

    # --- Panel (b): Distance to nearest training sounding ---
    ax = axes[1]
    n_bins = int(config["analysis"]["distance_bins"])
    min_samples = int(config["analysis"]["min_samples_per_bin"])
    edges = np.linspace(0, distance.max(), n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    for key in PANEL_MODELS:
        entry = model_style(config, key)
        bin_centers, bin_errors = [], []
        for index in range(n_bins):
            mask = (distance >= edges[index]) & (distance < edges[index + 1])
            if np.count_nonzero(mask) >= min_samples:
                bin_centers.append(centers[index])
                bin_errors.append(float(np.mean(
                    np.abs(table[key].values[mask] - true_depth[mask]))))
        ax.plot(bin_centers, bin_errors, color=entry["color"],
                marker=entry["marker"], markersize=4.5, linewidth=1.5,
                label=entry["short_label"], zorder=4)

    # Operational regime shading
    upper = float(distance.max())
    ax.axvspan(0, near_max, color="#2ecc71", alpha=0.10, zorder=1)
    ax.axvspan(near_max, transition_max, color="#f1c40f", alpha=0.10, zorder=1)
    ax.axvspan(transition_max, upper, color="#3498db", alpha=0.10, zorder=1)
    ax.axvline(near_max, color="#27ae60", linestyle="--", linewidth=0.9,
               alpha=0.85, zorder=2)
    ax.axvline(transition_max, color="#e67e22", linestyle="--", linewidth=0.9,
               alpha=0.85, zorder=2)

    ax.set_xlabel(r"Distance to nearest training sounding $d_{\mathrm{nn}}$ (km)")
    ax.set_ylabel(r"Binned $\mathrm{MAE}$ (m)")
    ax.set_xlim(0, upper)
    ax.grid(axis="both", linestyle=":", zorder=1)
    ax.text(0.03, 0.94, "(b)", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")

    regime_headers = [
        (near_max / 2, "Near regime\nIDW favored", "#1e8449"),
        ((near_max + transition_max) / 2, "Transition regime\nComparable", "#b7950b"),
        ((transition_max + upper) / 2, "Far regime\nSurrogate dominance", "#1b4f72"),
    ]
    for position, text, color in regime_headers:
        ax.text(position / upper, 1.02, text, ha="center", va="bottom",
                fontsize=7.0, color=color, fontweight="bold",
                transform=ax.transAxes)

    # Unified legend above both panels
    handles = [mpatches.Patch(facecolor=model_style(config, key)["color"],
                              edgecolor="black", linewidth=0.6,
                              label=model_style(config, key)["short_label"])
               for key in PANEL_MODELS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.50, 0.98),
               ncol=len(PANEL_MODELS), frameon=True, framealpha=0.95,
               edgecolor="#bdc3c7", fancybox=False)

    png_path, pdf_path = save_figure(fig, config, "Figure_6")
    report(png_path, pdf_path, f"strata={len(depth_masks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
