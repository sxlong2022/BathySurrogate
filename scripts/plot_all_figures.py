#!/usr/bin/env python3
"""Render every manuscript figure in one run.

Figure order follows the manuscript numbering:

    Figure 3  Feature ablation ladder
    Figure 4  Multi-model out-of-fold scatter
    Figure 5  Block-level statistical inference
    Figure 6  Depth and distance stratification
    Figure 7  Decadal bathymetric mapping
    Figure S2 S3GM performance dichotomy
    Figure S3 Block geometry sensitivity

Figure 1 (study area) and Figure 2 (validation protocol) are produced
elsewhere: Figure 1 by the cartography stage and Figure 2 as a vector
diagram. Figure S1 (software architecture) is also a vector diagram.

Usage:
    python scripts/plot_all_figures.py
"""

from __future__ import annotations

import os
import subprocess
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

#: Ordered (label, script) pairs.
FIGURES = [
    ("Figure 3  Feature ablation ladder", "plot_figure3.py"),
    ("Figure 4  Multi-model OOF scatter", "plot_figure4.py"),
    ("Figure 5  Block-level inference", "plot_figure5.py"),
    ("Figure 6  Depth and distance strata", "plot_figure6.py"),
    ("Figure 7  Decadal mapping", "plot_figure7.py"),
    ("Figure S2 S3GM dichotomy", "plot_figure_s2.py"),
    ("Figure S3 Block sensitivity", "plot_figure_s3.py"),
]


def main() -> int:
    """Render every figure sequentially."""
    print("=" * 70)
    print("RENDERING MANUSCRIPT FIGURES 3-7 AND S2-S3")
    print("=" * 70)

    for label, script in FIGURES:
        path = os.path.join(SCRIPTS_DIR, script)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Figure script not found: {path}")

        print(f"\n---> {label}: {script}", flush=True)
        subprocess.run([sys.executable, path], check=True)

    print("\n" + "=" * 70)
    print(f"[OK] Rendered {len(FIGURES)} figures.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
