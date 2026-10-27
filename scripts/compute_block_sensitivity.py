#!/usr/bin/env python3
"""Spatial block geometry sensitivity check (manuscript Table S3).

Re-runs the blocked 5-fold protocol with a 4x4 and a 6x6 block grid. The
check shows that the ranking of models does not depend on the block size,
which is the claim Table S3 supports.

Outputs:
    ``results/validation/block_sensitivity.csv`` with one row per model and
    protocol.

Usage:
    python scripts/compute_block_sensitivity.py
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import calc_metrics, load_config, repo_path, require_file, resolve_input

#: Models included in the sensitivity table.
MODEL_KEYS = ["lyzenga", "idw", "rf_enh"]

#: Number of cross-validation folds.
N_FOLDS = 5


def assign_blocks(coords: np.ndarray, n_blocks_per_axis: int) -> np.ndarray:
    """Row-major spatial block identifier for every coordinate."""
    edges = np.linspace(0.0, 1.0, n_blocks_per_axis + 1)
    rows = np.clip(np.digitize(coords[:, 0], edges) - 1, 0, n_blocks_per_axis - 1)
    columns = np.clip(np.digitize(coords[:, 1], edges) - 1, 0, n_blocks_per_axis - 1)
    return rows * n_blocks_per_axis + columns


def assign_folds(block_ids: np.ndarray, n_folds: int) -> np.ndarray:
    """Greedy balanced assignment of blocks to folds.

    Blocks are visited from largest to smallest and each one is placed in the
    currently smallest fold. This keeps fold sizes comparable while never
    splitting a block across folds.
    """
    unique, counts = np.unique(block_ids, return_counts=True)
    order = np.argsort(-counts)
    fold_sizes = np.zeros(n_folds, dtype=int)
    block_to_fold = {}
    for index in order:
        target = int(np.argmin(fold_sizes))
        block_to_fold[int(unique[index])] = target
        fold_sizes[target] += int(counts[index])
    return np.array([block_to_fold[int(block)] for block in block_ids])


def spectral_features(blue: np.ndarray, green: np.ndarray) -> np.ndarray:
    """The 7 spectral features without any spatial term."""
    return np.column_stack([
        blue, green,
        np.log(blue / green),
        np.log(blue),
        np.log(green),
        np.log(blue) - np.log(green),
        (blue - green) / (blue + green),
    ])


def idw_predict(test_coords: np.ndarray, train_coords: np.ndarray,
                train_depth: np.ndarray, power: int = 4) -> np.ndarray:
    """Inverse Distance Weighting with power = 4."""
    predictions = np.empty(test_coords.shape[0], dtype=float)
    for index, point in enumerate(test_coords):
        distances = np.maximum(
            np.sqrt(np.sum((train_coords - point) ** 2, axis=1)), 1e-6)
        weights = 1.0 / distances ** power
        predictions[index] = float(np.sum(weights * train_depth) / np.sum(weights))
    return predictions


def run_protocol(config: dict, n_blocks_per_axis: int) -> list:
    """Evaluate every model under one block geometry.

    Returns:
        List of result rows ready for CSV serialisation.
    """
    from sklearn.ensemble import RandomForestRegressor

    folds = np.load(require_file(
        resolve_input(config, "spatial_folds"),
        "Run: python run_bathymetry.py --stage 1.5",
    ))
    coordinates = folds["coordinates"]
    depths = folds["depths"]

    sentinel = np.load(resolve_input(config, "sentinel_series"), allow_pickle=True).item()
    gebco_series = np.load(resolve_input(config, "gebco_series"))

    latest = -1
    blue_map = sentinel["blue"][latest]
    green_map = sentinel["green"][latest]
    gebco_map = gebco_series[latest]
    height, width = blue_map.shape

    pixels = np.array([
        [int(min(max(coord[0] * (height - 1), 0), height - 1)),
         int(min(max(coord[1] * (width - 1), 0), width - 1))]
        for coord in coordinates
    ])
    blue_points = blue_map[pixels[:, 0], pixels[:, 1]]
    green_points = green_map[pixels[:, 0], pixels[:, 1]]
    gebco_points = np.array([
        -gebco_map[row, column] if gebco_map[row, column] < 0 else gebco_map[row, column]
        for row, column in pixels
    ])

    block_ids = assign_blocks(coordinates, n_blocks_per_axis)
    fold_ids = assign_folds(block_ids, N_FOLDS)

    rf_settings = dict(config["rf"])
    depth_min = float(config["domain"]["depth_min"])
    depth_max = float(config["domain"]["depth_max"])

    pooled = {key: np.full(depths.size, np.nan) for key in MODEL_KEYS}

    for fold in range(N_FOLDS):
        test_mask = fold_ids == fold
        train_mask = ~test_mask
        test_indices = np.where(test_mask)[0]

        train_coords = coordinates[train_mask]
        train_depth = depths[train_mask]

        pooled["idw"][test_indices] = idw_predict(
            coordinates[test_mask], train_coords, train_depth)

        # Optical-only Random Forest with 7 features.
        optical = RandomForestRegressor(**rf_settings)
        optical.fit(spectral_features(blue_points[train_mask], green_points[train_mask]),
                    train_depth)
        pooled["rf_enh"][test_indices] = np.clip(
            optical.predict(spectral_features(blue_points[test_mask],
                                              green_points[test_mask])),
            depth_min, depth_max,
        )

        # Enhanced Random Forest with 12 features.
        neighbors_train = np.empty((train_coords.shape[0], 2), dtype=float)
        for index, point in enumerate(train_coords):
            distances = np.sqrt(np.sum((train_coords - point) ** 2, axis=1))
            distances[distances < 1e-9] = np.inf
            nearest = int(np.argmin(distances))
            neighbors_train[index, 0] = distances[nearest]
            neighbors_train[index, 1] = train_depth[nearest]

        test_coords = coordinates[test_mask]
        neighbors_test = np.empty((test_coords.shape[0], 2), dtype=float)
        for index, point in enumerate(test_coords):
            distances = np.sqrt(np.sum((train_coords - point) ** 2, axis=1))
            nearest = int(np.argmin(distances))
            neighbors_test[index, 0] = distances[nearest]
            neighbors_test[index, 1] = train_depth[nearest]

        design_train = np.column_stack([
            spectral_features(blue_points[train_mask], green_points[train_mask]),
            gebco_points[train_mask],
            train_coords[:, 0], train_coords[:, 1],
            neighbors_train,
        ])
        design_test = np.column_stack([
            spectral_features(blue_points[test_mask], green_points[test_mask]),
            gebco_points[test_mask],
            test_coords[:, 0], test_coords[:, 1],
            neighbors_test,
        ])

        enhanced = RandomForestRegressor(**rf_settings)
        enhanced.fit(design_train, train_depth)
        pooled["rf_enh"][test_indices] = np.clip(
            enhanced.predict(design_test), depth_min, depth_max)

        # Lyzenga log-linear model, refit per fold.
        design_train = np.column_stack([
            np.ones(train_depth.size),
            np.log(blue_points[train_mask]),
            np.log(green_points[train_mask]),
        ])
        coefficients, *_ = np.linalg.lstsq(design_train, train_depth, rcond=None)
        design_test = np.column_stack([
            np.ones(int(test_mask.sum())),
            np.log(blue_points[test_mask]),
            np.log(green_points[test_mask]),
        ])
        pooled["lyzenga"][test_indices] = np.clip(
            design_test @ coefficients, depth_min, depth_max)

    rows = []
    for key in MODEL_KEYS:
        stats = calc_metrics(depths, pooled[key])
        rows.append({
            "protocol": f"{n_blocks_per_axis}x{n_blocks_per_axis}",
            "n_blocks": int(n_blocks_per_axis ** 2),
            "model": key,
            "r2": round(stats["r2"], 4),
            "corr": round(stats["corr"], 4),
            "rmse": round(stats["rmse"], 4),
            "mae": round(stats["mae"], 4),
        })
        print(f"  {n_blocks_per_axis}x{n_blocks_per_axis} {key:<10} "
              f"R2 {stats['r2']:+.3f} | r {stats['corr']:+.3f} | "
              f"RMSE {stats['rmse']:5.2f} m | MAE {stats['mae']:5.2f} m")

    return rows


def main() -> int:
    """Entry point."""
    config = load_config()

    print("=" * 70)
    print("SPATIAL BLOCK GEOMETRY SENSITIVITY")
    print("=" * 70)

    all_rows = []
    for n_blocks_per_axis in (4, 6):
        print(f"\nProtocol {n_blocks_per_axis}x{n_blocks_per_axis}")
        all_rows.extend(run_protocol(config, n_blocks_per_axis))

    output_path = repo_path(config["block_sensitivity"]["csv_path"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = ["protocol", "n_blocks", "model", "r2", "corr", "rmse", "mae"]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[OK] Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
