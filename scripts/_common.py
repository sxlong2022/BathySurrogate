"""Shared helpers for the BathySurrogate figure and analysis pipeline.

All scripts under ``scripts/`` import from this module so that path
resolution, metric definitions, and publication styling stay identical
across figures.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Repository layout
# --------------------------------------------------------------------------

#: Absolute path to the repository root (parent of ``scripts/``).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Absolute path to the configuration directory.
CONFIG_DIR = REPO_ROOT / "configs"

#: Default configuration file consumed by every pipeline script.
DEFAULT_CONFIG_PATH = CONFIG_DIR / "figures.yaml"

#: Environment variable that points at a directory holding ``results/``
#: and ``intermediate_results/``. Datasets and derived arrays are excluded
#: from version control, so a checkout normally keeps them elsewhere.
#: When the variable is unset, the repository root is used.
DATA_ROOT_ENV = "BATHYSURROGATE_DATA_ROOT"

#: Prefixes that resolve against the data root instead of the code root.
#: Prefixes that resolve against the data root instead of the code root.
DATA_PREFIXES = ("results", "intermediate_results", "logging")

# Make the package importable when a script is run directly from scripts/.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def data_root() -> Path:
    """Return the directory that holds the derived data tree."""
    override = os.environ.get(DATA_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT


def repo_path(relative: str) -> str:
    """Resolve a repository-relative path to an absolute path string.

    Paths under ``results/``, ``intermediate_results/`` or ``logging/``
    resolve against :func:`data_root`; everything else resolves against the
    code root.
    """
    normalized = relative.replace("\\", "/")
    base = REPO_ROOT
    if normalized.split("/", 1)[0] in DATA_PREFIXES:
        base = data_root()
    return str((base / normalized).resolve())


def ensure_parent(path: str) -> str:
    """Create the parent directory of ``path`` when it does not exist."""
    parent = Path(path).resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def load_config(path: Optional[str] = None) -> Dict:
    """Load the figure pipeline configuration.

    Args:
        path: Optional override for the YAML configuration file.

    Returns:
        Parsed configuration mapping.
    """
    import yaml

    target = Path(path).resolve() if path else DEFAULT_CONFIG_PATH
    if not target.exists():
        raise FileNotFoundError(f"Configuration file not found: {target}")
    with open(target, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {target}")
    return config


def resolve_input(config: Dict, key: str, **fmt: object) -> str:
    """Resolve an entry of ``inputs`` to an absolute path.

    Args:
        config: Parsed pipeline configuration.
        key: Key inside the ``inputs`` section.
        **fmt: Optional ``str.format`` fields such as ``fold`` or ``year``.

    Returns:
        Absolute path string.
    """
    template = config["inputs"][key]
    if fmt:
        template = template.format(**fmt)
    return repo_path(template)


def resolve_output(config: Dict, key: str, **fmt: object) -> str:
    """Resolve an entry of ``outputs`` to an absolute path."""
    template = config["outputs"][key]
    if fmt:
        template = template.format(**fmt)
    return repo_path(template)


def require_file(path: str, hint: str = "") -> str:
    """Raise a descriptive error when a required artefact is missing."""
    if not os.path.exists(path):
        message = f"Required input not found: {path}"
        if hint:
            message = f"{message}\n  Hint: {hint}"
        raise FileNotFoundError(message)
    return path


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute R2, Pearson r, RMSE and MAE over finite paired samples.

    Args:
        y_true: Observed depths.
        y_pred: Predicted depths.

    Returns:
        Mapping with keys ``r2``, ``corr``, ``rmse`` and ``mae``.
        Undefined statistics return ``nan``.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]

    if y_true.size == 0:
        return {"r2": float("nan"), "corr": float("nan"),
                "rmse": float("nan"), "mae": float("nan")}

    residual = y_pred - y_true
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))

    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual ** 2) / ss_tot) if ss_tot > 0 else float("nan")

    centered_true = y_true - y_true.mean()
    centered_pred = y_pred - y_pred.mean()
    denom = float(np.sqrt(np.sum(centered_true ** 2)) * np.sqrt(np.sum(centered_pred ** 2)))
    corr = float(np.sum(centered_true * centered_pred) / denom) if denom > 0 else float("nan")

    return {"r2": r2, "corr": corr, "rmse": rmse, "mae": mae}


def block_mae(df, model: str, block_column: str = "block_id",
              true_column: str = "true_depth") -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-block MAE for one model column.

    Args:
        df: OOF prediction table.
        model: Column name holding the model predictions.
        block_column: Column holding spatial block identifiers.
        true_column: Column holding observed depths.

    Returns:
        Tuple of ``(block_ids, mae_values)`` ordered by ascending block id.
    """
    block_ids = np.unique(df[block_column].values)
    values = np.empty(block_ids.size, dtype=float)
    for index, block in enumerate(block_ids):
        subset = df[df[block_column] == block]
        values[index] = float(np.mean(np.abs(subset[model].values - subset[true_column].values)))
    return block_ids, values


def bootstrap_mean_ci(differences: np.ndarray, n_resamples: int = 10000,
                      seed: int = 42, alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``differences``.

    Resampling is performed over spatial blocks, which are the independent
    units of the validation protocol.

    Returns:
        Tuple ``(lower, upper)`` bounds of the ``1 - alpha`` interval.
    """
    differences = np.asarray(differences, dtype=float).ravel()
    if differences.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = differences.size
    means = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample = rng.integers(0, n, size=n)
        means[index] = float(np.mean(differences[sample]))

    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------


def apply_style(config: Dict) -> None:
    """Apply the Elsevier EMS publication style defined in the configuration."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = config["style"]
    plt.rcParams.update({
        "font.family": style["font_family"],
        "font.serif": list(style["font_serif"]),
        "mathtext.fontset": style["mathtext_fontset"],
        "font.size": style["font_size"],
        "axes.labelsize": style["axes_labelsize"],
        "xtick.labelsize": style["xtick_labelsize"],
        "ytick.labelsize": style["ytick_labelsize"],
        "legend.fontsize": style["legend_fontsize"],
        "figure.dpi": style["figure_dpi"],
        "axes.linewidth": style["axes_linewidth"],
        "grid.linewidth": style["grid_linewidth"],
        "grid.alpha": style["grid_alpha"],
        "axes.unicode_minus": False,
    })


def model_style(config: Dict, key: str) -> Dict[str, str]:
    """Return the registry entry for one model key."""
    models = config["models"]
    if key not in models:
        raise KeyError(f"Unknown model key '{key}'. Available: {sorted(models)}")
    return models[key]


def ordered_models(config: Dict, keys: Optional[Sequence[str]] = None) -> List[str]:
    """Return model keys in configuration order, optionally filtered."""
    available = list(config["models"].keys())
    if keys is None:
        return available
    missing = [key for key in keys if key not in available]
    if missing:
        raise KeyError(f"Unknown model keys: {missing}")
    return list(keys)


def save_figure(fig, config: Dict, name: str) -> Tuple[str, str]:
    """Save one figure as 300 dpi PNG and vector PDF.

    Args:
        fig: Matplotlib figure.
        config: Parsed pipeline configuration.
        name: Base file name without extension.

    Returns:
        Tuple ``(png_path, pdf_path)`` of absolute paths.
    """
    import matplotlib.pyplot as plt

    dpi = config["style"]["savefig_dpi"]
    root = resolve_output(config, "figures_root")

    directory = os.path.join(root, name)
    os.makedirs(directory, exist_ok=True)

    png_path = os.path.join(directory, f"{name}_300dpi.png")
    pdf_path = os.path.join(directory, f"{name}.pdf")

    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    return png_path, pdf_path


def panel_tag(ax, text: str, config: Dict, x: float = 0.03, y: float = 0.94) -> None:
    """Place an (a)/(b) panel tag inside the axes coordinate system."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=config["style"]["panel_tag_size"],
            fontweight="bold", va="top", ha="left")


def distance_km(config: Dict, nn_dist: np.ndarray) -> np.ndarray:
    """Convert normalized nearest-neighbour distances to kilometres."""
    return np.asarray(nn_dist, dtype=float) * float(config["domain"]["roi_side_km"])


def report(png_path: str, pdf_path: str, extra: str = "") -> None:
    """Print a uniform completion line for every figure script."""
    suffix = f" | {extra}" if extra else ""
    print(f"[OK] Wrote {png_path} and {pdf_path}{suffix}", flush=True)
