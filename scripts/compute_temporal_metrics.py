#!/usr/bin/env python3
"""Multi-temporal consistency metrics for the annual bathymetry series.

For each consecutive year pair the script reports the Pearson correlation and
the mean absolute difference between the two maps. A stable surrogate should
produce high year-to-year correlation and low mean absolute difference.

Usage:
    python scripts/compute_temporal_metrics.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import load_config, require_file, resolve_input


def adjacent_year_metrics(stack: np.ndarray) -> tuple:
    """Correlation and mean absolute difference for each consecutive pair.

    Args:
        stack: Array of shape ``(n_years, H, W)``.

    Returns:
        Tuple of ``(correlations, mean_abs_differences, coefficient_of_variation)``.
    """
    correlations, mean_abs_differences = [], []
    for index in range(stack.shape[0] - 1):
        first = stack[index].ravel()
        second = stack[index + 1].ravel()

        first_centered = first - first.mean()
        second_centered = second - second.mean()
        denominator = np.sqrt(np.sum(first_centered ** 2)) * np.sqrt(np.sum(second_centered ** 2))
        correlations.append(float(np.sum(first_centered * second_centered) / denominator))
        mean_abs_differences.append(float(np.mean(np.abs(second - first))))

    mean_map = stack.mean(axis=0)
    std_map = stack.std(axis=0)
    coefficient = float(np.mean(std_map / np.maximum(mean_map, 0.1)))

    return correlations, mean_abs_differences, coefficient


def main() -> int:
    """Print the temporal consistency table."""
    config = load_config()

    years = list(config["domain"]["years"])
    rf_series = np.load(require_file(
        resolve_input(config, "rf_time_series"),
        "Run: python scripts/build_rf_time_series.py",
    ))
    print(f"Enhanced RF series: shape {rf_series.shape} | "
          f"range [{rf_series.min():.2f}, {rf_series.max():.2f}] m")

    # Only consecutive years present on disk can be compared, so a partial
    # archive must not be padded or silently re-indexed.
    s3gm_series, s3gm_years = [], []
    for year in years:
        path = resolve_input(config, "s3gm_year", year=year)
        if os.path.exists(path):
            s3gm_series.append(np.load(path))
            s3gm_years.append(year)

    if len(s3gm_series) >= 2:
        s3gm_stack = np.stack(s3gm_series)
        print(f"S3GM series: shape {s3gm_stack.shape} | "
              f"range [{s3gm_stack.min():.2f}, {s3gm_stack.max():.2f}] m")
    else:
        s3gm_stack = None
        s3gm_years = []
        print(f"[WARN] Only {len(s3gm_series)} S3GM annual map(s) found. "
              "At least two are needed. Only the surrogate is reported.")

    rf_correlations, rf_differences, rf_cv = adjacent_year_metrics(rf_series)
    if s3gm_stack is not None:
        s3gm_correlations, s3gm_differences, s3gm_cv = adjacent_year_metrics(s3gm_stack)

    print("\n=== MULTI-TEMPORAL CONSISTENCY ===")
    header = f"{'year pair':<14} | {'Enhanced RF (r / dMAE)':<26}"
    if s3gm_stack is not None:
        header += f" | {'S3GM (r / dMAE)':<24}"
    print(header)
    print("-" * len(header))

    rf_pairs = list(zip(years[:-1], years[1:]))
    for index, (first_year, second_year) in enumerate(rf_pairs):
        line = (f"{first_year}-{second_year:<9} | "
                f"{rf_correlations[index]:6.3f} / {rf_differences[index]:5.2f} m")
        line = line.ljust(43)
        if s3gm_stack is not None and index < len(s3gm_correlations):
            s3gm_pair = (s3gm_years[index], s3gm_years[index + 1])
            line += (f" | {s3gm_correlations[index]:6.3f} / "
                     f"{s3gm_differences[index]:5.2f} m")
            if s3gm_pair != (first_year, second_year):
                line += f"  [{s3gm_pair[0]}-{s3gm_pair[1]}]"
        print(line)

    print("-" * len(header))
    summary = (f"{'mean':<14} | "
               f"{np.mean(rf_correlations):6.3f} / {np.mean(rf_differences):5.2f} m "
               f"(CV {rf_cv:.2f})")
    if s3gm_stack is not None:
        summary += (f" | {np.mean(s3gm_correlations):6.3f} / "
                    f"{np.mean(s3gm_differences):5.2f} m (CV {s3gm_cv:.2f})")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
