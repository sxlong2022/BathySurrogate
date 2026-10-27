#!/usr/bin/env python3
"""Build the Enhanced Random Forest annual bathymetry series (2018-2023).

The figure pipeline needs a six-year raster stack for the decadal mapping
panel. Stage 1.5 saves the full-dataset surrogate model; this script applies
that model to every annual optical composite to produce the stack.

Outputs:
    ``results/classic_models/rf_time_series_enh.npy`` with shape
    ``(n_years, H, W)`` in metres.

Usage:
    python scripts/build_rf_time_series.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import load_config, repo_path, require_file, resolve_input


def main() -> int:
    """Apply the full-dataset surrogate to every annual composite."""
    from joblib import load

    from bathysurrogate.classic_models import ClassicModels

    config = load_config()

    sentinel = np.load(require_file(
        resolve_input(config, "sentinel_series"),
        "Run: python run_bathymetry.py --stage 1",
    ), allow_pickle=True).item()
    gebco_series = np.load(require_file(
        resolve_input(config, "gebco_series"),
        "Run: python run_bathymetry.py --stage 1",
    ))

    model = ClassicModels()
    model.rf_params = np.load(require_file(
        resolve_input(config, "rf_params_full"),
        "Run: python run_bathymetry.py --stage 1.5",
    ), allow_pickle=True).item()
    model.rf_model = load(require_file(
        resolve_input(config, "rf_model_full"),
        "Run: python run_bathymetry.py --stage 1.5",
    ))

    # A stale checkpoint would silently produce a 7-feature product that no
    # longer matches the reported validation. Fail loudly instead.
    n_expected = int(config["ablation"]["tiers"][-1]["n_features"])
    n_model = getattr(model.rf_model, "n_features_in_", None)
    if n_model != n_expected:
        raise ValueError(
            f"Checkpoint expects {n_model} features but the operational "
            f"surrogate uses {n_expected}. Retrain with: "
            "python run_bathymetry.py --stage 1.5"
        )

    years = list(config["domain"]["years"])
    if len(years) > sentinel["blue"].shape[0]:
        raise ValueError(
            f"Configuration lists {len(years)} years but the optical series "
            f"holds {sentinel['blue'].shape[0]} frames."
        )

    series = []
    for index, year in enumerate(years):
        depth = model.predict_rf(
            sentinel["blue"][index],
            sentinel["green"][index],
            gebco_band=gebco_series[index],
        )
        series.append(depth)
        valid = depth[depth > 0]
        mean_depth = float(valid.mean()) if valid.size else float("nan")
        print(f"  {year}: valid pixels = {valid.size:5d} | mean depth = {mean_depth:5.2f} m")

    stack = np.stack(series)
    output_path = repo_path(config["inputs"]["rf_time_series"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, stack)

    print(f"[OK] Wrote {output_path} with shape {stack.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
