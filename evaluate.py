import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

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

GRAN_COLOURS = [
    "#58a6ff", "#3fb950", "#f78166", "#d2a8ff",
    "#ffa657", "#79c0ff", "#56d364", "#ff7b72",
]

def colour_for(gran: int) -> str:
    return GRAN_COLOURS[gran % len(GRAN_COLOURS)]


def plot_evaluation(name: str, results: dict, output_dir: Path, show: bool):
    granularities = sorted(results.keys())
    n = len(granularities)

    # ── derive class list from the report keys (skip macro/weighted avg etc.) ──
    skip = {"accuracy", "macro avg", "weighted avg"}
    sample_report = next(iter(results.values()))["report"]
    classes = [k for k in sample_report if k not in skip]

    with plt.rc_context(STYLE):

        # ── 1. Confusion matrices — one per granularity (individual files) ──
        for g in granularities:
            cm = results[g]["cm"]
            _plot_single_cm(name, g, cm, classes, output_dir, show)

        # ── 2. Confusion matrices — all in a grid (single file) ──
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols * 4.5, rows * 4.2))
        axes = np.array(axes).flatten()
        fig.suptitle(f"{name}  ·  Confusion Matrices (all granularities)",
                     fontsize=14, fontweight="bold", color="#f0f6fc", y=1.01)
        for i, g in enumerate(granularities):
            _draw_cm(axes[i], results[g]["cm"], classes, f"Granularity {g}")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.tight_layout()
        _save(fig, output_dir / f"{name}_confusion_matrices_grid.png", show)

        # ── 3. Per-class bar charts: F1, Precision, Recall ──
        metrics = [
            ("f1-score",  "F1 Score"),
            ("precision", "Precision"),
            ("recall",    "Recall"),
        ]
        for metric_key, metric_label in metrics:
            fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.8 * n * 0.35 + 2), 5))
            _plot_per_class_bars(ax, results, granularities, classes,
                                 metric_key, metric_label)
            fig.suptitle(f"{name}  ·  {metric_label} per Class",
                         fontsize=14, fontweight="bold", color="#f0f6fc", y=1.01)
            fig.tight_layout()
            _save(fig, output_dir / f"{name}_{metric_key.replace('-','_')}_per_class.png", show)

        # ── 4. Accuracy per granularity ──
        fig, ax = plt.subplots(figsize=(max(5, n * 1.4), 5))
        accuracies = [results[g]["report"]["accuracy"] * 100 for g in granularities]
        bars = ax.bar(
            [f"gran {g}" for g in granularities],
            accuracies,
            color=[colour_for(g) for g in granularities],
            width=0.5, zorder=3,
        )
        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.2f}%", ha="center", va="bottom",
                    fontsize=9, color="#c9d1d9")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Accuracy (%)", fontsize=10, labelpad=8)
        ax.set_title("", fontsize=13)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
        fig.suptitle(f"{name}  ·  Accuracy per Granularity",
                     fontsize=14, fontweight="bold", color="#f0f6fc", y=1.01)
        fig.tight_layout()
        _save(fig, output_dir / f"{name}_accuracy_per_granularity.png", show)


def _draw_cm(ax, cm: np.ndarray, classes: list[str], title: str):
    """Draw a single confusion matrix onto an axes object."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks); ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(ticks); ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9, labelpad=6)
    ax.set_ylabel("True",      fontsize=9, labelpad=6)
    thresh = 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.0%})",
                    ha="center", va="center", fontsize=7,
                    color="white" if cm_norm[i, j] > thresh else "#c9d1d9")


def _plot_single_cm(name, g, cm, classes, output_dir, show):
    fig, ax = plt.subplots(figsize=(max(5, len(classes) * 1.1 + 1),
                                    max(4, len(classes) * 1.0 + 1)))
    fig.suptitle(f"{name}  ·  Confusion Matrix  —  Granularity {g}",
                 fontsize=13, fontweight="bold", color="#f0f6fc", y=1.01)
    _draw_cm(ax, cm, classes, "")
    fig.tight_layout()
    _save(fig, output_dir / f"{name}_confusion_matrix_gran{g}.png", show)


def _plot_per_class_bars(ax, results, granularities, classes,
                         metric_key, metric_label):
    n_gran   = len(granularities)
    n_class  = len(classes)
    width    = 0.8 / n_gran
    x        = np.arange(n_class)

    for i, g in enumerate(granularities):
        vals = [results[g]["report"].get(cls, {}).get(metric_key, 0.0)
                for cls in classes]
        offset = (i - n_gran / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      color=colour_for(g), label=f"gran {g}", zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color="#8b949e")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylabel(metric_label, fontsize=10, labelpad=8)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(title="Granularity", title_fontsize=9, fontsize=9,
              loc="upper right", framealpha=0.6)


def _save(fig, path: Path, show: bool):
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  saved → {path}")
    if show:
        plt.show()
    plt.close(fig)

