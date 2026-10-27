#!/usr/bin/env python3
"""Harmonized 12-feature spatial-spectral ablation experiment.

The ablation ladder isolates how much each feature group contributes to the
Enhanced Random Forest surrogate. All tiers are evaluated with the same
spatially blocked 5-fold partition used by the main validation.

Tier definition (see ``configs/figures.yaml``)::

    Spectral   (7)  normalized bands, band ratio, log transforms, depth index
    +GEBCO     (8)  adds the GEBCO bathymetric prior
    +Coord     (10) adds normalized northing and easting
    +NN        (10) adds nearest-training-sounding distance and depth
    Full       (12) all feature groups

Two tiers are anchored directly to the SSOT table so the ablation cannot
drift from the reported validation numbers:

* ``Spectral`` reuses the ``rf_old`` column.
* ``Full (12)`` reuses the ``rf_enh`` column.

The intermediate tiers are retrained here, because the SSOT table does not
contain them.

Outputs:
    ``intermediate_results/ablation_results.npz`` with ``true_depth``,
    ``block_ids``, ``fold_ids`` and one array per tier.

Usage:
    python scripts/harmonize_ablation_experiment.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (block_mae, calc_metrics, load_config, require_file,
                     resolve_input, resolve_output)

#: Feature groups produced by the optical block. Always present.
OPTICAL_FEATURES = [
    "blue_norm",
    "green_norm",
    "band_ratio_norm",
    "log_blue",
    "log_green",
    "log_ratio",
    "depth_index",
]


def build_optical_features(blue: np.ndarray, green: np.ndarray,
                           blue_mean: float, blue_std: float,
                           green_mean: float, green_std: float,
                           ratio_mean: float, ratio_std: float) -> List[np.ndarray]:
    """Compute the 7 spectral features with training-fold statistics."""
    blue_norm = (blue - blue_mean) / blue_std
    green_norm = (green - green_mean) / green_std

    ratio = np.log(blue / green)
    ratio_norm = (ratio - ratio_mean) / ratio_std

    log_blue = np.log(blue)
    log_green = np.log(green)
    log_ratio = log_blue - log_green
    depth_index = (blue_norm - green_norm) / (blue_norm + green_norm)

    return [blue_norm, green_norm, ratio_norm, log_blue, log_green,
            log_ratio, depth_index]


def nearest_sounding_features(target_coords: np.ndarray, train_coords: np.ndarray,
                              train_depths: np.ndarray,
                              exclude_self: bool) -> np.ndarray:
    """Distance and depth of the nearest training sounding.

    Args:
        target_coords: Coordinates of the rows being built.
        train_coords: Coordinates of the available training soundings.
        train_depths: Depths at ``train_coords``.
        exclude_self: Drop the coincident training point. Required while
            building the training matrix, otherwise the feature leaks the
            target depth.

    Returns:
        Array of shape ``(n_targets, 2)`` holding distance then depth.
    """
    output = np.empty((target_coords.shape[0], 2), dtype=float)
    for index, point in enumerate(target_coords):
        distances = np.sqrt(np.sum((train_coords - point) ** 2, axis=1))
        if exclude_self:
            coincident = np.where(distances < 1e-9)[0]
            if coincident.size > 0:
                distances[coincident] = np.inf
        nearest = int(np.argmin(distances))
        output[index, 0] = distances[nearest]
        output[index, 1] = train_depths[nearest]
    return output


def gebco_at_coords(gebco_map: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Sample the GEBCO prior at normalized coordinates, returned as depth."""
    height, width = gebco_map.shape
    values = np.empty(coords.shape[0], dtype=float)
    for index, coord in enumerate(coords):
        row = int(min(max(coord[0] * (height - 1), 0), height - 1))
        column = int(min(max(coord[1] * (width - 1), 0), width - 1))
        raw = float(gebco_map[row, column])
        values[index] = -raw if raw < 0 else raw
    return values


def build_design_matrix(coords: np.ndarray, train_coords: np.ndarray,
                        train_depths: np.ndarray, gebco_values: np.ndarray,
                        optical: List[np.ndarray], use_gebco: bool,
                        use_coords: bool, use_neighbors: bool,
                        exclude_self: bool) -> np.ndarray:
    """Assemble the feature matrix for one ablation tier."""
    columns = list(optical)

    if use_gebco:
        columns.append(gebco_values)

    if use_coords:
        columns.append(coords[:, 0])
        columns.append(coords[:, 1])

    if use_neighbors:
        neighbors = nearest_sounding_features(
            coords, train_coords, train_depths, exclude_self=exclude_self,
        )
        columns.append(neighbors[:, 0])
        columns.append(neighbors[:, 1])

    return np.column_stack(columns)


def run_experiment(config: Dict) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Train every ablation tier and write the result archive.

    Returns:
        Tuple of the archive path and a per-tier metric mapping.
    """
    from sklearn.ensemble import RandomForestRegressor
    from scipy.stats import wilcoxon

    rf_settings = dict(config["rf"])
    depth_min = float(config["domain"]["depth_min"])
    depth_max = float(config["domain"]["depth_max"])

    folds = np.load(require_file(
        resolve_input(config, "spatial_folds"),
        "Run: python run_bathymetry.py --stage 1.5",
    ))
    coordinates = folds["coordinates"]
    depths = folds["depths"]
    fold_ids = folds["fold_ids"]
    n_points = depths.size

    oof_table = pd.read_csv(require_file(
        resolve_input(config, "oof_predictions"),
        "Run: python scripts/build_oof_predictions.py",
    ))
    if len(oof_table) != n_points:
        raise ValueError(
            f"OOF table has {len(oof_table)} rows but folds hold {n_points}. "
            "Rebuild the table with scripts/build_oof_predictions.py."
        )
    block_ids = oof_table["block_id"].values

    sentinel = np.load(resolve_input(config, "sentinel_series"), allow_pickle=True).item()
    gebco_series = np.load(resolve_input(config, "gebco_series"))

    latest = -1
    blue_map = sentinel["blue"][latest]
    green_map = sentinel["green"][latest]
    gebco_map = gebco_series[latest]
    height, width = blue_map.shape

    rows = np.array([
        [int(min(max(coord[0] * (height - 1), 0), height - 1)),
         int(min(max(coord[1] * (width - 1), 0), width - 1))]
        for coord in coordinates
    ])
    blue_points = blue_map[rows[:, 0], rows[:, 1]]
    green_points = green_map[rows[:, 0], rows[:, 1]]
    gebco_points = gebco_at_coords(gebco_map, coordinates)

    tiers = config["ablation"]["tiers"]
    tier_names = [tier["name"] for tier in tiers]
    predictions = {name: np.full(n_points, np.nan) for name in tier_names}

    # Tiers anchored to the SSOT table are copied verbatim.
    anchored = {}
    for tier in tiers:
        column = tier.get("ssot_column")
        if column:
            if column not in oof_table.columns:
                raise KeyError(f"SSOT column '{column}' missing from the OOF table.")
            predictions[tier["name"]] = oof_table[column].values.astype(float).copy()
            anchored[tier["name"]] = column

    print("=" * 78)
    print(f"HARMONIZED 12-FEATURE ABLATION (pooled out-of-fold, n = {n_points})")
    print("=" * 78)

    for tier in tiers:
        name = tier["name"]
        if name in anchored:
            stats = calc_metrics(depths, predictions[name])
            print(f"{name:<12} | R2 {stats['r2']:+.3f} | r {stats['corr']:+.3f} | "
                  f"RMSE {stats['rmse']:5.2f} m | MAE {stats['mae']:5.2f} m "
                  f"| anchored to '{anchored[name]}'")
            continue

        for fold in range(5):
            test_mask = fold_ids == fold
            train_mask = ~test_mask
            test_indices = np.where(test_mask)[0]

            train_blue = blue_points[train_mask]
            train_green = green_points[train_mask]
            train_depth = depths[train_mask]

            # Normalization statistics come from the training fold only.
            blue_mean = float(np.mean(train_blue))
            blue_std = float(np.std(train_blue))
            green_mean = float(np.mean(train_green))
            green_std = float(np.std(train_green))
            ratio_train = np.log(train_blue / train_green)
            ratio_mean = float(np.mean(ratio_train))
            ratio_std = float(np.std(ratio_train))

            optical_train = build_optical_features(
                train_blue, train_green, blue_mean, blue_std,
                green_mean, green_std, ratio_mean, ratio_std,
            )
            optical_test = build_optical_features(
                blue_points[test_mask], green_points[test_mask],
                blue_mean, blue_std, green_mean, green_std,
                ratio_mean, ratio_std,
            )

            train_coords = coordinates[train_mask]
            design_train = build_design_matrix(
                train_coords, train_coords, train_depth, gebco_points[train_mask],
                optical_train, tier["use_gebco"], tier["use_coords"],
                tier["use_neighbors"], exclude_self=True,
            )
            design_test = build_design_matrix(
                coordinates[test_mask], train_coords, train_depth,
                gebco_points[test_mask], optical_test, tier["use_gebco"],
                tier["use_coords"], tier["use_neighbors"], exclude_self=False,
            )

            model = RandomForestRegressor(**rf_settings)
            model.fit(design_train, train_depth)
            predictions[name][test_indices] = np.clip(
                model.predict(design_test), depth_min, depth_max,
            )

        stats = calc_metrics(depths, predictions[name])
        print(f"{name:<12} | R2 {stats['r2']:+.3f} | r {stats['corr']:+.3f} | "
              f"RMSE {stats['rmse']:5.2f} m | MAE {stats['mae']:5.2f} m | retrained")

    archive_path = resolve_input(config, "ablation_results")
    os.makedirs(os.path.dirname(archive_path), exist_ok=True)
    np.savez(
        archive_path,
        true_depth=depths,
        block_ids=block_ids,
        fold_ids=fold_ids,
        **predictions,
    )
    print(f"\n[OK] Wrote {archive_path}")

    # Block-level Wilcoxon signed-rank test against the spectral baseline.
    print("\n" + "=" * 78)
    print("BLOCK-LEVEL SIGNIFICANCE vs SPECTRAL BASELINE (Wilcoxon signed-rank)")
    print("=" * 78)
    print(f"{'tier':<12} {'mean dMAE (m)':>14} {'median (m)':>12} {'W':>8} {'p':>12}")

    baseline_name = tiers[0]["name"]
    _, baseline_block_mae = block_mae(oof_table.assign(
        __baseline=predictions[baseline_name]), "__baseline")

    summary: Dict[str, Dict[str, float]] = {}
    for tier in tiers:
        name = tier["name"]
        stats = calc_metrics(depths, predictions[name])
        summary[name] = stats
        if name == baseline_name:
            continue

        frame = oof_table.assign(__current=predictions[name])
        _, current_block_mae = block_mae(frame, "__current")
        difference = baseline_block_mae - current_block_mae

        statistic, p_value = wilcoxon(difference, alternative="greater")
        summary[name]["mean_mae_reduction"] = float(difference.mean())
        summary[name]["wilcoxon_w"] = float(statistic)
        summary[name]["wilcoxon_p"] = float(p_value)

        print(f"{name:<12} {difference.mean():>14.2f} {np.median(difference):>12.2f} "
              f"{statistic:>8.1f} {p_value:>12.4e}")

    return archive_path, summary


def main() -> int:
    """Entry point."""
    config = load_config()
    run_experiment(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
