from __future__ import annotations

from pathlib import Path
import warnings

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
VARIABLE_DISPLAY_LABELS = {
    "TargetFatJet_mass": r"TargetFatJet mass [GeV]",
    "TargetFatJet_msoftdrop": r"TargetFatJet msoftdrop [GeV]",
    "CleanedJet_mass": r"Leading CleanedJet mass [GeV]",
}



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
    # Some EOS-backed environments can fail when savefig overwrites an existing file.
    if path.exists() or path.is_symlink():
        path.unlink()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
            category=UserWarning,
        )
        fig.tight_layout(pad=0.6)
    fig.savefig(path, dpi=220)
    pdf_path = path.with_suffix(".pdf")
    if pdf_path.exists() or pdf_path.is_symlink():
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



def _weighted_counts_and_errors(
    values: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts, edges = np.histogram(values, bins=bins, weights=weights)
    variances, _ = np.histogram(values, bins=bins, weights=np.square(weights))
    return (
        counts.astype(np.float64),
        np.sqrt(variances.astype(np.float64)),
        edges,
    )



def _set_weighted_events_axis_scale(ax: plt.Axes, positive_min: float | None, *, log_y: bool) -> None:
    if not log_y:
        ax.set_ylim(bottom=0.0)
        return
    if positive_min is None or positive_min <= 0.0:
        return
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(positive_min * 0.5, 1e-12))



def _class_color(index: int) -> tuple[float, float, float, float]:
    return plt.cm.tab10(index % 10)



def _display_label(name: str) -> str:
    return VARIABLE_DISPLAY_LABELS.get(name, name)



def _score_branch_name(score_name: str) -> str:
    return f"bdt_score_{score_name}"



def _score_column_title(score_name: str, score_labels: dict[str, str]) -> str:
    score_label = score_labels.get(score_name, score_name)
    return f"{score_label}\nclass={score_name}, {_score_branch_name(score_name)}"



def _score_axis_label(score_name: str) -> str:
    return f"BDT score ({_score_branch_name(score_name)})"



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



def plot_training_metric_curves(
    outpath: Path,
    fold_summaries: list[dict[str, object]],
    metric_name: str,
) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    has_curve = False

    for index, fold_summary in enumerate(fold_summaries):
        evals_result = fold_summary.get("evals_result", {})
        if not isinstance(evals_result, dict):
            continue
        fold_number = int(fold_summary.get("fold", index)) + 1
        color = _class_color(index)
        dataset_styles = {
            "train": ("--", 0.82),
            "eval_balanced": ("-", 0.90),
            "eval_physics": (":", 0.90),
            "eval": ("-", 0.90),
        }
        preferred_order = ["train", "eval_balanced", "eval_physics", "eval"]
        dataset_order = [name for name in preferred_order if name in evals_result]
        dataset_order.extend(name for name in evals_result if name not in dataset_order)
        for dataset_name in dataset_order:
            dataset_metrics = evals_result.get(dataset_name, {})
            if not isinstance(dataset_metrics, dict) or metric_name not in dataset_metrics:
                continue
            values = np.asarray(dataset_metrics[metric_name], dtype=np.float64)
            valid = np.isfinite(values)
            if not np.any(valid):
                continue
            rounds = np.arange(1, values.size + 1, dtype=np.int32)
            linestyle, alpha = dataset_styles.get(dataset_name, ("-.", 0.78))
            ax.plot(
                rounds[valid],
                values[valid],
                color=color,
                linestyle=linestyle,
                linewidth=1.2,
                alpha=alpha,
                label=f"fold {fold_number} {dataset_name}",
            )
            has_curve = True

    ax.set_xlabel("Boosting round", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(metric_name, fontsize=AXIS_LABEL_SIZE)
    ax.set_title(f"Training curve: {metric_name}", fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.grid(alpha=0.25)
    if has_curve:
        ax.legend(frameon=False, loc="best", ncol=2, fontsize=max(7, LEGEND_FONT_SIZE - 1))
    else:
        ax.text(0.5, 0.5, "No metric history", ha="center", va="center", transform=ax.transAxes)
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



def plot_pairwise_roc_curve(
    outpath: Path,
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    positive_label: str,
    negative_label: str,
    auc_value: float,
) -> None:
    from sklearn.metrics import roc_curve

    _setup_style()
    fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.plot(
        tpr,
        1.0 - fpr,
        linewidth=2.0,
        label=f"{positive_label} vs {negative_label} (AUC = {auc_value:.4f})",
    )
    ax.plot([0.0, 1.0], [1.0, 0.0], linestyle="--", color="black", linewidth=1.1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(f"{positive_label} efficiency", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(f"{negative_label} rejection", fontsize=AXIS_LABEL_SIZE)
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



def plot_training_class_score_weighted_events(
    outpath: Path,
    score_name: str,
    score_label: str,
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    class_names: list[str],
    class_labels: dict[str, str],
    log_y: bool = False,
) -> None:
    _setup_style()
    bins = np.linspace(0.0, 1.0, 31, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    positive_min: float | None = None

    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        if not np.any(mask):
            continue
        counts, errors, edges = _weighted_counts_and_errors(scores[mask], weights[mask], bins)
        positive_counts = counts[counts > 0.0]
        if positive_counts.size:
            current_min = float(np.min(positive_counts))
            positive_min = current_min if positive_min is None else min(positive_min, current_min)
        centers = 0.5 * (edges[:-1] + edges[1:])
        color = _class_color(class_index)
        ax.step(
            centers,
            counts,
            where="mid",
            linewidth=1.6,
            color=color,
            label=class_labels.get(class_name, class_name),
        )
        ax.errorbar(
            centers,
            counts,
            yerr=errors,
            fmt="none",
            ecolor=color,
            elinewidth=0.9,
            capsize=0.0,
            alpha=0.85,
        )

    ax.set_xlabel(f"{score_label} score", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Weighted events / bin", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlim(0.0, 1.0)
    _set_weighted_events_axis_scale(ax, positive_min, log_y=log_y)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.grid(alpha=0.25, which="both")
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



def plot_process_score_weighted_events(
    outpath: Path,
    scores: np.ndarray,
    weights: np.ndarray,
    processes: np.ndarray,
    process_order: list[str],
    process_labels: dict[str, str],
    x_label: str = "Event BDT score",
    log_y: bool = False,
) -> None:
    _setup_style()
    bins = np.linspace(0.0, 1.0, 31, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    positive_min: float | None = None

    for index, process in enumerate(process_order):
        mask = processes == process
        if not np.any(mask):
            continue
        counts, errors, edges = _weighted_counts_and_errors(scores[mask], weights[mask], bins)
        positive_counts = counts[counts > 0.0]
        if positive_counts.size:
            current_min = float(np.min(positive_counts))
            positive_min = current_min if positive_min is None else min(positive_min, current_min)
        centers = 0.5 * (edges[:-1] + edges[1:])
        color = process_color(process, index)
        ax.step(
            centers,
            counts,
            where="mid",
            linewidth=1.5,
            label=process_labels.get(process, process),
            color=color,
        )
        ax.errorbar(
            centers,
            counts,
            yerr=errors,
            fmt="none",
            ecolor=color,
            elinewidth=0.9,
            capsize=0.0,
            alpha=0.85,
        )

    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Weighted events / bin", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlim(0.0, 1.0)
    _set_weighted_events_axis_scale(ax, positive_min, log_y=log_y)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.grid(alpha=0.25, which="both")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=LEGEND_FONT_SIZE)
    _cms_label(ax)
    _save(fig, outpath)



def plot_feature_mass_correlation_heatmap(
    outpath: Path,
    matrix: np.ndarray,
    variable_names: list[str],
    process_label: str,
) -> None:
    _setup_style()
    n_variables = len(variable_names)
    figsize = (
        max(8.5, 0.54 * n_variables + 2.6),
        max(7.5, 0.54 * n_variables + 2.0),
    )
    fig, ax = plt.subplots(figsize=figsize)

    masked = np.ma.masked_invalid(np.asarray(matrix, dtype=np.float64))
    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="#f0f0f0")
    image = ax.imshow(masked, vmin=-1.0, vmax=1.0, cmap=cmap, aspect="auto")

    tick_labels = [_display_label(name) for name in variable_names]
    positions = np.arange(n_variables, dtype=int)
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(tick_labels, rotation=60, ha="right")
    ax.set_yticklabels(tick_labels)
    ax.set_title(f"{process_label}: feature/mass weighted correlation", fontsize=TITLE_SIZE)
    ax.set_xlim(-0.5, n_variables - 0.5)
    ax.set_ylim(n_variables - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, n_variables, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, n_variables, 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label("Weighted Pearson correlation", fontsize=AXIS_LABEL_SIZE)
    _cms_label(ax)
    _save(fig, outpath)



def plot_score_vs_mass_grid(
    outpath: Path,
    process_label: str,
    score_by_name: dict[str, np.ndarray],
    score_labels: dict[str, str],
    mass_by_name: dict[str, np.ndarray],
    mass_names: list[str],
    mass_ranges: dict[str, tuple[float, float]],
    weights: np.ndarray,
) -> None:
    _setup_style()
    score_names = list(score_by_name)
    fig, axes = plt.subplots(
        nrows=len(mass_names),
        ncols=len(score_names),
        figsize=(13.8, 10.4),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(
        f"{process_label}: per-class BDT score vs mass",
        fontsize=TITLE_SIZE + 1,
        y=0.995,
    )

    histogram_payloads: list[tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]] = []
    global_min: float | None = None
    global_max: float | None = None
    for mass_name in mass_names:
        y_min, y_max = mass_ranges[mass_name]
        mass_values = np.asarray(mass_by_name[mass_name], dtype=np.float64)
        for score_name in score_names:
            score_values = np.asarray(score_by_name[score_name], dtype=np.float64)
            valid = (
                np.isfinite(score_values)
                & np.isfinite(mass_values)
                & np.isfinite(weights)
                & (weights > 0.0)
            )
            if not np.any(valid):
                histogram_payloads.append((None, None, None))
                continue
            hist, x_edges, y_edges = np.histogram2d(
                score_values[valid],
                mass_values[valid],
                bins=(40, 40),
                range=((0.0, 1.0), (y_min, y_max)),
                weights=weights[valid],
            )
            positive = hist[hist > 0.0]
            if positive.size:
                current_min = float(np.min(positive))
                current_max = float(np.max(positive))
                global_min = current_min if global_min is None else min(global_min, current_min)
                global_max = current_max if global_max is None else max(global_max, current_max)
            histogram_payloads.append((hist, x_edges, y_edges))

    norm = None
    if global_min is not None and global_max is not None and global_max > 0.0:
        norm = matplotlib.colors.LogNorm(vmin=max(global_min, 1e-12), vmax=global_max)

    color_mesh = None
    payload_index = 0
    for row_index, mass_name in enumerate(mass_names):
        y_min, y_max = mass_ranges[mass_name]
        for column_index, score_name in enumerate(score_names):
            ax = axes[row_index, column_index]
            hist, x_edges, y_edges = histogram_payloads[payload_index]
            payload_index += 1
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(y_min, y_max)
            if row_index == 0:
                ax.set_title(_score_column_title(score_name, score_labels), fontsize=TITLE_SIZE - 1)
            if column_index == 0:
                ax.set_ylabel(_display_label(mass_name), fontsize=AXIS_LABEL_SIZE)
            if row_index == len(mass_names) - 1:
                ax.set_xlabel(_score_axis_label(score_name), fontsize=AXIS_LABEL_SIZE)

            if hist is None or x_edges is None or y_edges is None or not np.any(hist > 0.0):
                ax.text(0.5, 0.5, "No finite entries", ha="center", va="center", transform=ax.transAxes)
                ax.grid(alpha=0.25)
                continue

            color_mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                hist.T,
                shading="auto",
                cmap="viridis",
                norm=norm,
            )
            ax.grid(alpha=0.15)

    if color_mesh is not None:
        colorbar = fig.colorbar(color_mesh, ax=axes, fraction=0.03, pad=0.02)
        colorbar.set_label("Weighted events / bin", fontsize=AXIS_LABEL_SIZE)
    _cms_label(axes[0, 0])
    _save(fig, outpath)
