from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from tthcc_an.definitions import process_color


AXIS_LABEL_SIZE = 13
TITLE_SIZE = 12
TICK_LABEL_SIZE = 11
LEGEND_FONT_SIZE = 9
CMS_LABEL_SIZE = 15



def _setup_style() -> None:
    plt.style.use(hep.style.CMS)
    plt.rcParams.update(
        {
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
        }
    )



def _cms_label(ax: plt.Axes) -> None:
    hep.cms.label(
        label="Private Work",
        data=False,
        ax=ax,
        rlabel="2024 (13.6 TeV)",
        fontsize=CMS_LABEL_SIZE,
    )



def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.6)
    fig.savefig(path, dpi=220)
    pdf_path = path.with_suffix(".pdf")
    if pdf_path.exists():
        pdf_path.unlink()
    plt.close(fig)



def _weighted_density(values: np.ndarray, weights: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values, bins=bins, weights=weights)
    total = float(np.sum(hist))
    if total <= 0:
        return np.zeros_like(hist, dtype=np.float64), edges
    widths = np.diff(edges)
    density = hist.astype(np.float64) / total
    positive = widths > 0
    density[positive] = density[positive] / widths[positive]
    return density, edges



def _class_color(index: int) -> tuple[float, float, float, float]:
    return plt.cm.tab10(index % 10)



def plot_roc_curve(
    outpath: Path,
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    auc_value: float,
) -> None:
    from sklearn.metrics import roc_curve

    _setup_style()
    fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.plot(tpr, 1.0 - fpr, linewidth=2.0, label=f"Event BDT (AUC = {auc_value:.4f})")
    ax.plot([0.0, 1.0], [1.0, 0.0], linestyle="--", color="black", linewidth=1.1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Signal efficiency", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Background rejection", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best", fontsize=LEGEND_FONT_SIZE)
    _cms_label(ax)
    _save(fig, outpath)



def plot_ovr_roc_curves(
    outpath: Path,
    score_by_class: dict[str, np.ndarray],
    labels: np.ndarray,
    weights: np.ndarray,
    class_names: list[str],
    class_labels: dict[str, str],
    auc_by_class: dict[str, float],
    macro_auc: float,
) -> None:
    from sklearn.metrics import roc_curve

    _setup_style()
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    for class_index, class_name in enumerate(class_names):
        one_vs_rest = (labels == class_index).astype(np.int8)
        scores = np.asarray(score_by_class[class_name], dtype=np.float64)
        fpr, tpr, _ = roc_curve(one_vs_rest, scores, sample_weight=weights)
        label = class_labels.get(class_name, class_name)
        auc_value = float(auc_by_class[class_name])
        ax.plot(
            tpr,
            1.0 - fpr,
            linewidth=1.8,
            color=_class_color(class_index),
            label=f"{label} (AUC = {auc_value:.4f})",
        )

    ax.plot([0.0, 1.0], [1.0, 0.0], linestyle="--", color="black", linewidth=1.1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Class efficiency", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Rest rejection", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best", fontsize=LEGEND_FONT_SIZE)
    _cms_label(ax)
    _save(fig, outpath)



def plot_class_score_shapes(
    outpath: Path,
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> None:
    _setup_style()
    bins = np.linspace(0.0, 1.0, 31, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    signal_mask = labels == 1
    background_mask = labels == 0
    for mask, label, color in [
        (signal_mask, "Signal: ttHcc + ttHbb", "#d62728"),
        (background_mask, "Background", "#1f77b4"),
    ]:
        density, edges = _weighted_density(scores[mask], weights[mask], bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.step(centers, density, where="mid", linewidth=1.8, label=label, color=color)

    ax.set_xlabel("Event BDT score", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Unit-normalized weighted density", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best", fontsize=LEGEND_FONT_SIZE)
    _cms_label(ax)
    _save(fig, outpath)



def plot_training_class_score_shapes(
    outpath: Path,
    score_name: str,
    score_label: str,
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    class_names: list[str],
    class_labels: dict[str, str],
) -> None:
    _setup_style()
    bins = np.linspace(0.0, 1.0, 31, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))

    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        if not np.any(mask):
            continue
        density, edges = _weighted_density(scores[mask], weights[mask], bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.step(
            centers,
            density,
            where="mid",
            linewidth=1.6,
            color=_class_color(class_index),
            label=class_labels.get(class_name, class_name),
        )

    ax.set_xlabel(f"{score_label} score", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Unit-normalized weighted density", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best", fontsize=LEGEND_FONT_SIZE)
    _cms_label(ax)
    _save(fig, outpath)



def plot_process_score_shapes(
    outpath: Path,
    scores: np.ndarray,
    weights: np.ndarray,
    processes: np.ndarray,
    process_order: list[str],
    process_labels: dict[str, str],
    x_label: str = "Event BDT score",
) -> None:
    _setup_style()
    bins = np.linspace(0.0, 1.0, 31, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.0, 6.4))

    for index, process in enumerate(process_order):
        mask = processes == process
        if not np.any(mask):
            continue
        density, edges = _weighted_density(scores[mask], weights[mask], bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.step(
            centers,
            density,
            where="mid",
            linewidth=1.5,
            label=process_labels.get(process, process),
            color=process_color(process, index),
        )

    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Unit-normalized weighted density", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=LEGEND_FONT_SIZE)
    _cms_label(ax)
    _save(fig, outpath)
