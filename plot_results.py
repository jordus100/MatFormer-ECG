#!/usr/bin/env python3
"""
Plot ML training and validation results from Matryoshka model training.

Usage:
    python plot_training.py <name>
    python plot_training.py <name> --output-dir ./plots
    python plot_training.py <name> --no-show

Expects:
    <name>_train.pickle      - list of dicts with keys: mat_gran, train_loss
    <name>_validation.pickle - list of dicts with keys: mat_gran, val_loss, val_accuracy
"""

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── Styling ──────────────────────────────────────────────────────────────────

STYLE = {
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "axes.titlecolor":   "#f0f6fc",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linewidth":    0.8,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "legend.labelcolor": "#c9d1d9",
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
}

# Distinct colours for each Matryoshka granularity level
GRAN_COLOURS = [
    "#58a6ff",  # 0 – blue
    "#3fb950",  # 1 – green
    "#f78166",  # 2 – red/salmon
    "#d2a8ff",  # 3 – purple
    "#ffa657",  # 4 – orange
    "#79c0ff",  # 5 – light blue
    "#56d364",  # 6 – bright green
    "#ff7b72",  # 7 – coral
]


def colour_for(gran: int) -> str:
    return GRAN_COLOURS[gran % len(GRAN_COLOURS)]


def moving_average(values: list[float], window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, smoothed_values) using a centred moving average.
    Valid output starts at index (window//2) and ends (window//2) before the end."""
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    # align to the centre of each window
    half = window // 2
    indices = np.arange(half, half + len(smoothed))  # 0-based
    return indices, smoothed


# ── Data loading ──────────────────────────────────────────────────────────────

def load_pickle(path: Path) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


def organise_train(records: list[list[dict]]) -> dict[int, list[float]]:
    """
    Returns {mat_gran: [loss_epoch0, loss_epoch1, …]}
    Each element of `records` is one epoch (a list of per-granularity dicts).
    """
    gran_losses: dict[int, list[float]] = defaultdict(list)
    for epoch_data in records:
        for entry in epoch_data:
            gran_losses[entry["mat_gran"]].append(entry["train_loss"])
    return dict(gran_losses)


def organise_val(records: list[list[dict]]) -> tuple[dict, dict]:
    """
    Returns ({mat_gran: [val_loss…]}, {mat_gran: [val_acc…]})
    """
    gran_loss: dict[int, list[float]] = defaultdict(list)
    gran_acc:  dict[int, list[float]] = defaultdict(list)
    for epoch_data in records:
        for entry in epoch_data:
            g = entry["mat_gran"]
            gran_loss[g].append(entry["val_loss"])
            gran_acc[g].append(entry["val_accuracy"])
    return dict(gran_loss), dict(gran_acc)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _epoch_axis(n: int) -> np.ndarray:
    return np.arange(1, n + 1)


def _style_ax(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(
        title="Granularity",
        title_fontsize=9,
        fontsize=9,
        loc="best",
        framealpha=0.6,
    )


def _plot_series(ax, epochs, values, color, label, smooth_window: int | None):
    """Plot a raw series and, if requested, overlay a smoothed version."""
    raw_alpha = 0.00 if smooth_window else 1.0
    ax.plot(epochs, values, linewidth=1.5, marker="o", markersize=2,
            color=color, alpha=raw_alpha, label=None if smooth_window else label)
    if smooth_window and smooth_window > 1 and len(values) >= smooth_window:
        idx, smoothed = moving_average(values, smooth_window)
        sm_epochs = idx + 1  # convert 0-based to 1-based epoch numbers
        ax.plot(sm_epochs, smoothed, linewidth=2.5, color=color, label=label)


def plot_train(name: str, train_data: dict[int, list[float]], output_dir: Path, show: bool,
               smooth_window: int | None = None):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        smooth_note = f"  (smoothed w={smooth_window})" if smooth_window else ""
        fig.suptitle(f"{name}  ·  Training Loss{smooth_note}", fontsize=15, fontweight="bold",
                     color="#f0f6fc", y=1.01)

        for gran in sorted(train_data):
            losses = train_data[gran]
            epochs = _epoch_axis(len(losses))
            _plot_series(ax, epochs, losses, colour_for(gran), f"gran {gran}", smooth_window)

        _style_ax(ax, "", "Epoch", "Loss")
        fig.tight_layout()

        out = output_dir / f"{name}_train_loss.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  saved → {out}")
        if show:
            plt.show()
        plt.close(fig)


def plot_val(name: str,
             val_loss: dict[int, list[float]],
             val_acc:  dict[int, list[float]],
             output_dir: Path,
             show: bool,
             smooth_window: int | None = None):
    with plt.rc_context(STYLE):
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(16, 5))
        smooth_note = f"  (smoothed w={smooth_window})" if smooth_window else ""
        fig.suptitle(f"{name}  ·  Validation Metrics{smooth_note}", fontsize=15, fontweight="bold",
                     color="#f0f6fc", y=1.01)

        for gran in sorted(val_loss):
            c, lbl = colour_for(gran), f"gran {gran}"
            epochs_l = _epoch_axis(len(val_loss[gran]))
            _plot_series(ax_loss, epochs_l, val_loss[gran], c, lbl, smooth_window)

        for gran in sorted(val_acc):
            c, lbl = colour_for(gran), f"gran {gran}"
            epochs_a = _epoch_axis(len(val_acc[gran]))
            _plot_series(ax_acc, epochs_a, [v * 100 for v in val_acc[gran]], c, lbl, smooth_window)

        _style_ax(ax_loss, "Validation Loss",     "Epoch", "Loss")
        _style_ax(ax_acc,  "Validation Accuracy", "Epoch", "Accuracy (%)")
        ax_acc.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))

        fig.tight_layout()

        out = output_dir / f"{name}_validation.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  saved → {out}")
        if show:
            plt.show()
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot Matryoshka model training & validation results."
    )
    parser.add_argument("name", help="Base name used in <name>_train.pickle / <name>_validation.pickle")
    parser.add_argument("--output-dir", default=".", metavar="DIR",
                        help="Directory to save plots (default: current dir)")
    parser.add_argument("--smooth-window", type=int, default=None, metavar="N",
                        help="Rolling average window size (odd number recommended). "
                             "Raw data shown faintly underneath.")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip interactive display; only save files")
    args = parser.parse_args()

    name         = args.name
    output_dir   = Path(args.output_dir)
    show         = not args.no_show
    smooth_window = args.smooth_window

    if smooth_window is not None and smooth_window < 2:
        print("ERROR: --smooth-window must be at least 2", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(f"{name}_train.pickle")
    val_path   = Path(f"{name}_validation.pickle")

    missing = [p for p in (train_path, val_path) if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: file not found: {p}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {train_path} …")
    raw_train = load_pickle(train_path)
    print(f"Loading {val_path} …")
    raw_val   = load_pickle(val_path)

    train_data          = organise_train(raw_train)
    val_loss, val_acc   = organise_val(raw_val)

    n_epochs = len(next(iter(train_data.values())))
    grans    = sorted(set(train_data) | set(val_loss))
    print(f"  epochs={n_epochs}  granularities={grans}")

    print("Plotting …")
    plot_train(name, train_data, output_dir, show, smooth_window)
    plot_val(name, val_loss, val_acc, output_dir, show, smooth_window)

    print("Done.")


if __name__ == "__main__":
    main()