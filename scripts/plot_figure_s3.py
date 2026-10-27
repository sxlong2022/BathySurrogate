#!/usr/bin/env python3
"""Figure S3: Spatial block geometry sensitivity analysis.

Compares a 4x4 and a 6x6 spatial block grid under the same blocked 5-fold
protocol. The point of the panel is that model ranking is stable across block
sizes, so the reported conclusions do not depend on an arbitrary grid choice.

Input:
    ``results/validation/block_sensitivity.csv`` produced by
    ``scripts/compute_block_sensitivity.py``. When that file is absent the
    published Table S3 values from ``configs/figures.yaml`` are plotted
    instead, and the fallback is reported on stdout.

Usage:
    python scripts/plot_figure_s3.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (apply_style, load_config, model_style, report, repo_path,
                     save_figure)


def load_from_csv(path: str) -> dict:
    """Read the sensitivity CSV into a nested mapping.

    Returns:
        ``{model: {protocol: {r2, corr, rmse, mae}}}``
    """
    import pandas as pd

    table = pd.read_csv(path)
    required = {"protocol", "model", "r2", "corr", "rmse", "mae"}
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"Sensitivity CSV is missing columns: {sorted(missing)}")

    data = {}
    for _, row in table.iterrows():
        data.setdefault(row["model"], {})[row["protocol"]] = {
            "r2": float(row["r2"]),
            "corr": float(row["corr"]),
            "rmse": float(row["rmse"]),
            "mae": float(row["mae"]),
        }
    return data


def load_from_config(config: dict) -> dict:
    """Read the published Table S3 values from the configuration."""
    return config["block_sensitivity"]["table"]


def main() -> int:
    """Render Figure S3."""
    import matplotlib.pyplot as plt

    config = load_config()
    apply_style(config)

    settings = config["block_sensitivity"]
    csv_path = repo_path(settings["csv_path"])

    if os.path.exists(csv_path):
        data = load_from_csv(csv_path)
        print(f"Source: {csv_path}")
    else:
        data = load_from_config(config)
        print(f"[WARN] Sensitivity CSV not found: {csv_path}")
        print("       Plotting the published Table S3 values from "
              "configs/figures.yaml. Regenerate with "
              "scripts/compute_block_sensitivity.py.")

    model_keys = list(settings["models"])
    protocols = [entry["key"] for entry in settings["protocols"]]
    protocol_labels = [entry["label"] for entry in settings["protocols"]]

    labels = [model_style(config, key)["short_label"] for key in model_keys]
    positions = np.arange(len(model_keys))
    width = 0.36

    fig, (ax_error, ax_score) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.20, top=0.90, wspace=0.28)

    # --- Panel (a): MAE and RMSE per protocol ---
    error_specs = [
        ("mae", 0, "#2980b9", "#5dade2"),
        ("rmse", 1, "#c0392b", "#e74c3c"),
    ]
    group_offsets = np.linspace(-0.30, 0.30, 2 * len(protocols))

    for metric, metric_index, dark, light in error_specs:
        for protocol_index, protocol in enumerate(protocols):
            offset = group_offsets[metric_index * len(protocols) + protocol_index]
            values = [data[key][protocol][metric] for key in model_keys]
            color = dark if protocol_index == 0 else light
            bars = ax_error.bar(positions + offset, values, width / 2,
                                label=f"{metric.upper()} ({protocol_labels[protocol_index]})",
                                color=color, alpha=0.9, edgecolor="black",
                                linewidth=0.5, zorder=3)
            for bar in bars:
                ax_error.annotate(
                    f"{bar.get_height():.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.2,
                )

    ax_error.set_ylabel("Error metric (m)")
    ax_error.set_xticks(positions)
    ax_error.set_xticklabels(labels, rotation=15, ha="right")
    ax_error.set_ylim(0, float(max(
        data[key][protocol][metric]
        for key in model_keys for protocol in protocols
        for metric, _, _, _ in error_specs
    )) * 1.22)
    ax_error.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax_error.legend(loc="upper right", framealpha=0.92, ncol=2,
                    handlelength=1.1, columnspacing=0.6, fontsize=7.2)
    ax_error.text(-0.12, 1.03, "(a)", transform=ax_error.transAxes,
                  fontsize=9.5, fontweight="bold", va="bottom")
    ax_error.set_title("MAE and RMSE comparison", fontsize=8.5,
                       fontweight="bold", pad=6)

    # --- Panel (b): R2 and Pearson r per protocol ---
    score_specs = [
        ("r2", 0, "#27ae60", "#58d68d"),
        ("corr", 1, "#8e44ad", "#bb8fce"),
    ]

    for metric, metric_index, dark, light in score_specs:
        for protocol_index, protocol in enumerate(protocols):
            offset = group_offsets[metric_index * len(protocols) + protocol_index]
            values = [data[key][protocol][metric] for key in model_keys]
            color = dark if protocol_index == 0 else light
            label = r"$\mathrm{R}^2$" if metric == "r2" else "Pearson r"
            bars = ax_score.bar(positions + offset, values, width / 2,
                                label=f"{label} ({protocol_labels[protocol_index]})",
                                color=color, alpha=0.9, edgecolor="black",
                                linewidth=0.5, zorder=3)
            for bar in bars:
                height = bar.get_height()
                ax_score.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2 if height >= 0 else -6),
                    textcoords="offset points", ha="center",
                    va="bottom" if height >= 0 else "top", fontsize=6.2,
                )

    ax_score.axhline(0, color="black", linewidth=0.7, zorder=2)
    ax_score.set_ylabel("Metric value (dimensionless)")
    ax_score.set_xticks(positions)
    ax_score.set_xticklabels(labels, rotation=15, ha="right")
    ax_score.set_ylim(-0.52, 1.12)
    ax_score.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax_score.legend(loc="upper right", framealpha=0.92, ncol=2,
                    handlelength=1.1, columnspacing=0.6, fontsize=7.2)
    ax_score.text(-0.12, 1.03, "(b)", transform=ax_score.transAxes,
                  fontsize=9.5, fontweight="bold", va="bottom")
    ax_score.set_title(r"Correlation and $\mathrm{R}^2$ comparison",
                       fontsize=8.5, fontweight="bold", pad=6)

    png_path, pdf_path = save_figure(fig, config, "Figure_S3")
    report(png_path, pdf_path, f"protocols={len(protocols)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
