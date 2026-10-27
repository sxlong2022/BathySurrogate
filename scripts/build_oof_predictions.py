#!/usr/bin/env python3
"""Build the single source of truth (SSOT) out-of-fold prediction table.

Every model reported in the manuscript is evaluated with the same spatially
blocked 5-fold partition. This script writes one row per chart sounding and
one column per model, which downstream analysis and figure scripts consume.

Output schema::

    point_id, coord_y, coord_x, fold_id, block_id, true_depth,
    lyzenga, stumpf, idw, rf_old, rf_enh, s3gm, nn_dist

Definitions:

* ``rf_old``  - Random Forest with the 7 spectral features only.
* ``rf_enh``  - Random Forest with the 12 features (spectral + GEBCO +
  normalized coordinates + nearest-sounding distance and depth).
* ``idw``     - Inverse Distance Weighting with power = 4.
* ``lyzenga`` - Log-linear model after Lyzenga (1978).
* ``stumpf``  - Band-ratio model after Stumpf et al. (2003).
* ``s3gm``    - S3GM diffusion prediction for diagnostic fold 0 only.
* ``nn_dist`` - Distance to the nearest sounding in a *different* fold,
  normalized by the study area side length.

Usage:
    python scripts/build_oof_predictions.py
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (calc_metrics, load_config, require_file,
                     resolve_input, resolve_output)

#: Models written to the SSOT table, in column order.
MODEL_KEYS: List[str] = ["lyzenga", "stumpf", "idw", "rf_old", "rf_enh", "s3gm"]


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def lyzenga_predict(train_blue: np.ndarray, train_green: np.ndarray,
                    train_depth: np.ndarray, test_blue: np.ndarray,
                    test_green: np.ndarray) -> np.ndarray:
    """Log-linear inversion after Lyzenga (1978).

    Fits ``depth = a0 + a1 * ln(blue) + a2 * ln(green)`` on the training fold.
    """
    design_train = np.column_stack([np.ones(train_depth.size),
                                    np.log(train_blue), np.log(train_green)])
    coefficients, *_ = np.linalg.lstsq(design_train, train_depth, rcond=None)
    design_test = np.column_stack([np.ones(test_blue.size),
                                   np.log(test_blue), np.log(test_green)])
    return design_test @ coefficients


def stumpf_predict(train_blue: np.ndarray, train_green: np.ndarray,
                   train_depth: np.ndarray, test_blue: np.ndarray,
                   test_green: np.ndarray) -> np.ndarray:
    """Band-ratio inversion after Stumpf et al. (2003).

    Uses the ratio ``ln(n * blue) / ln(m * green)`` with the published
    constants ``n = 1000`` and ``m = 10000``.
    """
    n_scale, m_scale = 1000.0, 10000.0
    design_train = np.column_stack([
        np.ones(train_depth.size),
        np.log(n_scale * train_blue) / np.log(m_scale * train_green),
    ])
    coefficients, *_ = np.linalg.lstsq(design_train, train_depth, rcond=None)
    design_test = np.column_stack([
        np.ones(test_blue.size),
        np.log(n_scale * test_blue) / np.log(m_scale * test_green),
    ])
    return design_test @ coefficients


def idw_predict(test_coords: np.ndarray, train_coords: np.ndarray,
                train_depth: np.ndarray, power: int = 4) -> np.ndarray:
    """Inverse Distance Weighting with a fixed power exponent.

    Args:
        test_coords: Normalized coordinates of the evaluation soundings.
        train_coords: Normalized coordinates of the training soundings.
        train_depth: Observed depths at the training soundings.
        power: Distance exponent. The manuscript reports ``power = 4``.

    Returns:
        Interpolated depths for ``test_coords``.
    """
    predictions = np.empty(test_coords.shape[0], dtype=float)
    for index, point in enumerate(test_coords):
        distances = np.sqrt(np.sum((train_coords - point) ** 2, axis=1))
        distances = np.maximum(distances, 1e-6)
        weights = 1.0 / distances ** power
        predictions[index] = float(np.sum(weights * train_depth) / np.sum(weights))
    return predictions


# --------------------------------------------------------------------------
# Spatial bookkeeping
# --------------------------------------------------------------------------


def block_id_of(coord: np.ndarray, n_blocks_per_axis: int) -> int:
    """Map a normalized coordinate to a row-major spatial block identifier."""
    edges = np.linspace(0.0, 1.0, n_blocks_per_axis + 1)
    row = min(max(0, np.digitize(coord[0], edges) - 1), n_blocks_per_axis - 1)
    column = min(max(0, np.digitize(coord[1], edges) - 1), n_blocks_per_axis - 1)
    return int(row * n_blocks_per_axis + column)


def nearest_other_fold_distance(coords: np.ndarray, fold_ids: np.ndarray) -> np.ndarray:
    """Distance to the nearest sounding held out in a different fold.

    The value represents how far each evaluation sounding sits from the
    training data that was actually available for its fold.
    """
    distances = np.empty(coords.shape[0], dtype=float)
    for index in range(coords.shape[0]):
        other_fold = fold_ids != fold_ids[index]
        delta = coords[other_fold] - coords[index]
        distances[index] = float(np.sqrt(np.sum(delta ** 2, axis=1)).min())
    return distances


def pixel_index(coord: np.ndarray, height: int, width: int) -> tuple:
    """Convert a normalized coordinate to a clipped raster index pair."""
    row = int(min(max(coord[0] * (height - 1), 0), height - 1))
    column = int(min(max(coord[1] * (width - 1), 0), width - 1))
    return row, column


# --------------------------------------------------------------------------
# Main routine
# --------------------------------------------------------------------------


def build_table(config: Dict) -> str:
    """Compute every model prediction and write the SSOT CSV.

    Returns:
        Absolute path to the written CSV file.
    """
    from joblib import load

    from bathysurrogate.classic_models import ClassicModels

    depth_min = float(config["domain"]["depth_min"])
    depth_max = float(config["domain"]["depth_max"])

    folds = np.load(require_file(
        resolve_input(config, "spatial_folds"),
        "Run: python run_bathymetry.py --stage 1.5",
    ))
    coordinates = folds["coordinates"]
    depths = folds["depths"]
    fold_ids = folds["fold_ids"]

    sentinel = np.load(resolve_input(config, "sentinel_series"), allow_pickle=True).item()
    gebco_series = np.load(resolve_input(config, "gebco_series"))

    # Latest acquisition year drives the single-date comparison.
    latest = -1
    blue_map = sentinel["blue"][latest]
    green_map = sentinel["green"][latest]
    gebco_map = gebco_series[latest]
    height, width = blue_map.shape

    pixels = np.array([pixel_index(coord, height, width) for coord in coordinates])
    blue_points = blue_map[pixels[:, 0], pixels[:, 1]]
    green_points = green_map[pixels[:, 0], pixels[:, 1]]

    n_points = depths.size
    predictions = {key: np.full(n_points, np.nan) for key in MODEL_KEYS}

    # S3GM is available for the diagnostic fold only.
    s3gm_fold0_path = resolve_input(config, "s3gm_fold_year", fold=0, year=2023)
    s3gm_fold0 = np.load(s3gm_fold0_path) if os.path.exists(s3gm_fold0_path) else None
    if s3gm_fold0 is None:
        print(f"[WARN] S3GM fold 0 map not found: {s3gm_fold0_path}")
        print("       The s3gm column will be empty.")

    for fold in range(5):
        test_mask = fold_ids == fold
        train_mask = ~test_mask
        test_indices = np.where(test_mask)[0]

        train_blue = blue_points[train_mask]
        train_green = green_points[train_mask]
        train_depth = depths[train_mask]

        predictions["lyzenga"][test_indices] = lyzenga_predict(
            train_blue, train_green, train_depth,
            blue_points[test_mask], green_points[test_mask],
        )
        predictions["stumpf"][test_indices] = stumpf_predict(
            train_blue, train_green, train_depth,
            blue_points[test_mask], green_points[test_mask],
        )
        predictions["idw"][test_indices] = idw_predict(
            coordinates[test_mask], coordinates[train_mask], train_depth,
        )

        # Optical-only Random Forest: refit inside the fold with 7 features.
        optical = ClassicModels()
        optical.train_rf(blue_points[train_mask], green_points[train_mask],
                         train_depth, use_enhanced=False)
        optical_map = optical.predict_rf(blue_map, green_map)
        for point_index in test_indices:
            row, column = pixels[point_index]
            predictions["rf_old"][point_index] = optical_map[row, column]

        # Enhanced Random Forest: reuse the fold model produced by stage 1.5.
        enhanced = ClassicModels()
        enhanced.rf_params = np.load(
            resolve_input(config, "rf_params_fold", fold=fold), allow_pickle=True,
        ).item()
        enhanced.rf_model = load(resolve_input(config, "rf_model_fold", fold=fold))
        enhanced_map = enhanced.predict_rf(blue_map, green_map, gebco_band=gebco_map)
        for point_index in test_indices:
            row, column = pixels[point_index]
            predictions["rf_enh"][point_index] = enhanced_map[row, column]

        if fold == 0 and s3gm_fold0 is not None:
            for point_index in test_indices:
                row, column = pixels[point_index]
                value = float(s3gm_fold0[row, column])
                if np.isfinite(value) and value > 0:
                    predictions["s3gm"][point_index] = value

        print(f"[OK] Fold {fold}: {test_indices.size} evaluation soundings")

    # Clip predictions to the physical depth range.
    for key in ("lyzenga", "stumpf", "idw", "rf_old", "rf_enh"):
        predictions[key] = np.clip(predictions[key], depth_min, depth_max)

    block_ids = np.array([
        block_id_of(coord, int(config["domain"]["n_blocks_per_axis"]))
        for coord in coordinates
    ])
    nn_dist = nearest_other_fold_distance(coordinates, fold_ids)

    output_path = os.path.join(
        resolve_output(config, "validation_dir"),
        "oof_predictions_all_models.csv",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    header = ["point_id", "coord_y", "coord_x", "fold_id", "block_id",
              "true_depth"] + MODEL_KEYS + ["nn_dist"]

    def format_value(value: float, digits: int) -> str:
        return f"{value:.{digits}f}" if np.isfinite(value) else ""

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(n_points):
            row = [
                index,
                f"{coordinates[index][0]:.6f}",
                f"{coordinates[index][1]:.6f}",
                int(fold_ids[index]),
                int(block_ids[index]),
                f"{depths[index]:.3f}",
            ]
            row += [format_value(predictions[key][index], 3) for key in MODEL_KEYS]
            row.append(f"{nn_dist[index]:.8f}")
            writer.writerow(row)

    print(f"[OK] Wrote {output_path}")

    # Report pooled metrics so the run is self-documenting.
    print("\nPooled out-of-fold metrics")
    print(f"{'model':<10} {'n':>5} {'R2':>8} {'r':>8} {'RMSE':>8} {'MAE':>8}")
    for key in MODEL_KEYS:
        valid = np.isfinite(predictions[key])
        if not valid.any():
            print(f"{key:<10} {0:>5} {'-':>8} {'-':>8} {'-':>8} {'-':>8}")
            continue
        stats = calc_metrics(depths[valid], predictions[key][valid])
        print(f"{key:<10} {int(valid.sum()):>5} {stats['r2']:>8.3f} "
              f"{stats['corr']:>8.3f} {stats['rmse']:>8.2f} {stats['mae']:>8.2f}")

    return output_path


def main() -> int:
    """Entry point."""
    config = load_config()
    build_table(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
