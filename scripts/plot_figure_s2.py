#!/usr/bin/env python3
"""Figure S2: S3GM performance dichotomy.

Two panels contrast the two evaluation regimes that the manuscript reports:

    (a) Diagnostic out-of-fold spatial holdout (fold 0). Soundings are held
        out together with their spatial block, so this panel measures spatial
        generalization to unseen water.
    (b) In-sample conditional reconstruction. Every sounding was part of the
        conditioning set, so this panel measures how faithfully the diffusion
        posterior honours the observations it was given. It is not a
        generalization metric.

Both panels are computed from stored artefacts. No synthetic noise is added
to any prediction: the contrast between panels is a property of the
evaluation protocol, not of the data.

Usage:
    python scripts/plot_figure_s2.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (apply_style, calc_metrics, load_config, report,
                     require_file, resolve_input, save_figure)


def sample_raster(raster: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Nearest-pixel lookup of a raster at normalized coordinates."""
    height, width = raster.shape
    values = np.empty(coords.shape[0], dtype=float)
    for index, coord in enumerate(coords):
        row = int(min(max(coord[0] * (height - 1), 0), height - 1))
        column = int(min(max(coord[1] * (width - 1), 0), width - 1))
        values[index] = float(raster[row, column])
    return values


def draw_panel(ax, observed, predicted, color, title, x_label, y_label,
               stats, limit, tag):
    """Draw one scatter panel with the 1:1 reference, fit line and statistics."""
    ax.scatter(observed, predicted, color=color, alpha=0.55, s=18,
               edgecolors="black", linewidth=0.3, zorder=3)
    ax.plot([0, limit], [0, limit], color="black", linestyle="--",
            linewidth=0.9, zorder=2, label="1:1 reference")

    finite = np.isfinite(observed) & np.isfinite(predicted)
    if np.count_nonzero(finite) > 1:
        slope, intercept = np.polyfit(observed[finite], predicted[finite], 1)
        grid = np.linspace(0, limit, 100)
        ax.plot(grid, slope * grid + intercept, color=color, linestyle="-",
                linewidth=1.3, zorder=4, label="Linear fit")

    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", zorder=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{tag} {title}", loc="left", fontsize=9.2,
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
                      alpha=0.90, edgecolor="#bdc3c7", linewidth=0.6),
            zorder=5)


def main() -> int:
    """Render Figure S2."""
    import matplotlib.pyplot as plt

    config = load_config()
    apply_style(config)

    table = pd.read_csv(require_file(
        resolve_input(config, "oof_predictions"),
        "Run: python scripts/build_oof_predictions.py",
    ))

    limit = float(config["domain"]["scatter_limit"])
    true_depth = table["true_depth"].values
    coords = table[["coord_y", "coord_x"]].values

    # --- Panel (a): diagnostic out-of-fold holdout, fold 0 ---
    fold0 = table["fold_id"].values == 0
    observed_fold0 = true_depth[fold0]
    predicted_fold0 = table["s3gm"].values[fold0]

    valid_fold0 = np.isfinite(predicted_fold0)
    if not valid_fold0.any():
        raise ValueError(
            "The s3gm column is empty. Run stage 2 for fold 0 and rebuild "
            "the OOF table with scripts/build_oof_predictions.py."
        )
    stats_fold0 = calc_metrics(observed_fold0[valid_fold0], predicted_fold0[valid_fold0])
    n_fold0 = int(np.count_nonzero(valid_fold0))

    # --- Panel (b): in-sample conditional reconstruction ---
    reconstruction_path = require_file(
        resolve_input(config, "s3gm_year", year=2023),
        "Run: python run_bathymetry.py --stage 2",
    )
    reconstruction = np.load(reconstruction_path)
    predicted_insample = sample_raster(reconstruction, coords)

    # Every sounding is part of the conditioning set, so all finite values
    # contribute. Dropping shallow zero-metre pixels would inflate the fit.
    valid_insample = np.isfinite(predicted_insample)
    stats_insample = calc_metrics(true_depth[valid_insample],
                                  predicted_insample[valid_insample])
    n_insample = int(np.count_nonzero(valid_insample))

    print(f"Panel (a) out-of-fold holdout : n = {n_fold0:3d} | "
          f"R2 {stats_fold0['r2']:+.3f} | RMSE {stats_fold0['rmse']:5.2f} m")
    print(f"Panel (b) in-sample anchors   : n = {n_insample:3d} | "
          f"R2 {stats_insample['r2']:+.3f} | RMSE {stats_insample['rmse']:5.2f} m")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.14, top=0.88, wspace=0.28)

    draw_panel(
        axes[0],
        observed_fold0[valid_fold0],
        predicted_fold0[valid_fold0],
        color="#8e44ad",
        title=f"Diagnostic out-of-fold holdout (fold 0, n = {n_fold0})",
        x_label="Nautical chart sounding depth (m)",
        y_label="Predicted water depth (m)",
        stats=stats_fold0,
        limit=limit,
        tag="(a)",
    )

    draw_panel(
        axes[1],
        true_depth[valid_insample],
        predicted_insample[valid_insample],
        color="#27ae60",
        title=f"In-sample conditional reconstruction (n = {n_insample})",
        x_label="Nautical chart sounding depth (m)",
        y_label="Reconstructed water depth (m)",
        stats=stats_insample,
        limit=limit,
        tag="(b)",
    )

    png_path, pdf_path = save_figure(fig, config, "Figure_S2")
    report(png_path, pdf_path, "regimes=2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
