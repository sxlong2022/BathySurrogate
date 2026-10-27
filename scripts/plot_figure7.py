#!/usr/bin/env python3
"""Figure 7: Six-year decadal bathymetric mapping (2018-2023).

A 2x3 panel grid shows the annual Enhanced Random Forest products on a shared
colour scale with geographic coordinates. Where a JRC land-water mask is
available it is applied so land pixels read as neutral grey rather than as
zero-metre water.

Inputs:
    ``results/classic_models/rf_time_series_enh.npy`` produced by
    ``scripts/build_rf_time_series.py``.

Usage:
    python scripts/plot_figure7.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (apply_style, load_config, report, require_file,
                     resolve_input, save_figure)


def load_land_mask(config: dict, shape: tuple) -> np.ndarray:
    """Load and nearest-resample the JRC land-water mask.

    Returns:
        Boolean array of the requested shape where ``True`` marks water.
        When the mask is unavailable the function returns ``None`` and the
        caller falls back to optical validity only.
    """
    from PIL import Image

    path = resolve_input(config, "land_water_mask")
    if not os.path.exists(path):
        print(f"[WARN] Land-water mask not found: {path}")
        print("       Land pixels will not be masked in this render.")
        return None

    image = Image.open(path)
    height, width = shape
    resized = np.array(image.resize((width, height), Image.NEAREST))
    return resized != 0


def main() -> int:
    """Render Figure 7."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker

    config = load_config()
    apply_style(config)

    series_path = require_file(
        resolve_input(config, "rf_time_series"),
        "Run: python scripts/build_rf_time_series.py",
    )
    series = np.load(series_path)

    years = list(config["domain"]["years"])
    if series.shape[0] != len(years):
        raise ValueError(
            f"Series holds {series.shape[0]} frames but {len(years)} years "
            "are configured. Rebuild with scripts/build_rf_time_series.py."
        )

    land_mask = load_land_mask(config, series.shape[1:])

    west, south, east, north = [float(value) for value in config["domain"]["extent"]]
    extent = [west, east, south, north]
    depth_max = float(config["domain"]["depth_max"])

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#bdc3c7")

    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.85))
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.10, top=0.94,
                        wspace=0.18, hspace=0.22)

    image = None
    for index, ax in enumerate(axes.flat):
        frame = series[index].astype(float).copy()
        if land_mask is not None:
            frame[~land_mask] = np.nan

        image = ax.imshow(frame, cmap=cmap, vmin=0.0, vmax=depth_max,
                          extent=extent, origin="upper", aspect="auto")

        ax.set_title(f"({chr(ord('a') + index)}) {years[index]}", loc="left",
                     fontsize=9.0, fontweight="bold", pad=5)

        if index in (0, 3):
            ax.set_ylabel("Latitude (deg N)")
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda value, _: f"{value:.2f}N"))
        else:
            ax.set_yticklabels([])

        if index in (3, 4, 5):
            ax.set_xlabel("Longitude (deg E)")
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda value, _: f"{value:.2f}E"))
        else:
            ax.set_xticklabels([])

    colorbar_ax = fig.add_axes([0.90, 0.12, 0.022, 0.74])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Water depth (m)", fontsize=9.0)
    colorbar.ax.tick_params(labelsize=8.0)

    if land_mask is not None:
        land_patch = mpatches.Patch(facecolor="#bdc3c7", edgecolor="black",
                                    linewidth=0.5, label="Land (JRC)")
        fig.legend(handles=[land_patch], loc="lower right",
                   bbox_to_anchor=(0.99, 0.02), fontsize=7.5, frameon=False)

    png_path, pdf_path = save_figure(fig, config, "Figure_7")
    report(png_path, pdf_path, f"years={len(years)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
