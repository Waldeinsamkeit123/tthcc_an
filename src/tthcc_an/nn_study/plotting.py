from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from tthcc_an.nn_study.config import (
    NnQcdScoreGroup,
    NnScoreClass,
    NnStudyConfig,
    NnTruthCategory,
)
from tthcc_an.nn_study.metrics import PairwiseRoc


def _setup_style() -> None:
    plt.style.use("default")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _histograms_by_truth(
    values: np.ndarray,
    truth_index: np.ndarray,
    weights: np.ndarray,
    truths: list[NnTruthCategory],
    bins: np.ndarray,
    *,
    normalize: bool,
) -> list[np.ndarray]:
    histograms: list[np.ndarray] = []
    for truth_number, _ in enumerate(truths):
        mask = (
            (truth_index == truth_number)
            & np.isfinite(values)
            & np.isfinite(weights)
            & (weights > 0)
        )
        hist, _ = np.histogram(values[mask], bins=bins, weights=weights[mask])
        hist = hist.astype(np.float64)
        if normalize and np.sum(hist) > 0:
            hist /= np.sum(hist)
        histograms.append(hist)
    return histograms


def plot_score_distribution(
    *,
    outpath: Path,
    score: NnScoreClass,
    values: np.ndarray,
    truth_index: np.ndarray,
    weights: np.ndarray,
    truths: list[NnTruthCategory],
    config: NnStudyConfig,
    normalize: bool,
    figsize: tuple[float, float] | None = None,
    legend_columns: int | None = None,
    legend_fontsize: float | None = None,
    bin_edges: np.ndarray | None = None,
    display_range: tuple[float, float] | None = None,
    cms_com: float | None = None,
    reference_style: bool = False,
    selection_label: str | None = None,
    stack: bool = False,
) -> None:
    if stack and normalize:
        raise ValueError("Stacked score distributions require weighted yields.")
    _setup_style()
    if reference_style:
        plt.style.use(hep.style.CMS)
        plt.rcParams.update({"font.size": 13})
    if bin_edges is None:
        n_bins = int(config.plot_options.get("score_bins", 20))
        score_range = config.plot_options.get("score_range", [0.0, 1.0])
        bins = np.linspace(float(score_range[0]), float(score_range[1]), n_bins + 1)
    else:
        bins = np.asarray(bin_edges, dtype=np.float64)
        if bins.ndim != 1 or bins.size < 2 or not np.all(np.diff(bins) > 0.0):
            raise ValueError("Score histogram bin edges must be strictly increasing.")
    histograms = _histograms_by_truth(
        values, truth_index, weights, truths, bins, normalize=normalize
    )
    fig, ax = plt.subplots(
        dpi=int(config.plot_options.get("score_dpi", 150)),
        figsize=figsize or (5, 4),
    )
    bottom = np.zeros(bins.size - 1, dtype=np.float64)
    for truth, hist in zip(truths, histograms):
        if np.sum(hist) <= 0:
            continue
        if stack:
            ax.bar(bins[:-1], hist, width=np.diff(bins), bottom=bottom,
                   align="edge", color=truth.color, label=truth.label,
                   linewidth=0.25, edgecolor="white")
            bottom += hist
        else:
            hep.histplot(hist, bins=bins, ax=ax, color=truth.color, label=truth.label)
    x_min, x_max = display_range or (float(bins[0]), float(bins[-1]))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min >= x_max:
        raise ValueError("Score display range must contain two ascending finite values.")
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(score.label if reference_style else f"{score.label} score")
    ax.set_ylabel(
        "Fraction of events" if normalize else ("Events" if reference_style else "Total expected events")
    )
    use_log = bool(
        config.plot_options.get("shape_log_y", True)
        if normalize
        else config.plot_options.get("yield_log_y", True)
    )
    if use_log and (normalize or not reference_style):
        ax.set_yscale("log")
    if normalize:
        ax.set_ylim(float(config.plot_options.get("shape_y_min", 1e-4)), None)
    if reference_style:
        peak = (float(np.max(bottom)) if stack else
                max((float(np.max(hist)) for hist in histograms), default=0.0))
        if peak > 0.0:
            ax.set_ylim(top=peak * (8.0 if normalize else 1.8))
        hep.cms.label(data=False, llabel="", ax=ax, year=config.year,
                      lumi=f"{config.lumi_fb:.1f}", com=config.energy_tev, fontsize=15)
        ax.text(0.13, 1.005, "Simulation", transform=ax.transAxes,
                va="bottom", fontsize=15, fontstyle="italic")
        if selection_label:
            ax.text(0.02, 0.97, selection_label, transform=ax.transAxes,
                    va="top", fontsize="small")
    else:
        cms_kwargs = {} if cms_com is None else {"com": cms_com}
        hep.cms.label(data=False, ax=ax, **cms_kwargs)
    ax.legend(
        ncol=2 if reference_style else legend_columns or 2,
        frameon=False,
        columnspacing=0.6,
        handlelength=1.0,
        fontsize=9 if reference_style else legend_fontsize,
        loc="upper left" if reference_style else "best",
        bbox_to_anchor=(0.02, 0.93) if reference_style else None,
    )
    if reference_style:
        fig.tight_layout()
    _save(fig, outpath)


def plot_mass_sculpting(
    *,
    outpath: Path,
    bins: np.ndarray,
    base_density: np.ndarray,
    curves: list[dict[str, object]],
    variable_label: str,
    population_label: str,
    score_branch: str,
) -> None:
    _setup_style()
    fig, (ax, ratio_ax) = plt.subplots(
        nrows=2,
        ncols=1,
        dpi=150,
        figsize=(7.6, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.08},
    )
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax.step(
        centers,
        base_density,
        where="mid",
        linewidth=2.3,
        color="black",
        label="Uncut",
        zorder=10,
    )
    ratio_ax.axhline(1.0, color="black", linewidth=1.1, linestyle="--")
    colors = plt.get_cmap("tab10").colors
    for index, curve in enumerate(curves):
        if not bool(curve["has_yield"]):
            continue
        color = colors[index % len(colors)]
        ax.step(
            centers,
            np.asarray(curve["density"], dtype=np.float64),
            where="mid",
            linewidth=1.55,
            color=color,
            label=str(curve["label"]),
        )
        ratio_ax.step(
            centers,
            np.asarray(curve["retention"], dtype=np.float64),
            where="mid",
            linewidth=1.3,
            color=color,
        )
    ax.set_xlim(float(bins[0]), float(bins[-1]))
    ax.set_ylabel("Unit-normalized weighted density")
    ax.grid(alpha=0.25)
    ax.legend(
        frameon=False,
        loc="upper right",
        ncol=2 if len(curves) >= 6 else 1,
        fontsize=7.5,
        columnspacing=0.8,
        handlelength=1.8,
    )
    ax.text(
        0.02,
        0.96,
        f"{population_label}\nCuts on {score_branch}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2.5},
    )
    ratio_ax.set_xlabel(variable_label)
    ratio_ax.set_ylabel("Kept / uncut")
    ratio_ax.set_ylim(0.0, 1.08)
    ratio_ax.grid(alpha=0.25)
    hep.cms.label(data=False, ax=ax)
    _save(fig, outpath)


def _threshold_edges(thresholds: np.ndarray) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.size == 1:
        width = max(abs(float(thresholds[0])) * 0.1, 0.05)
        return np.array([thresholds[0] - width, thresholds[0] + width])
    midpoints = 0.5 * (thresholds[:-1] + thresholds[1:])
    return np.concatenate(
        (
            [thresholds[0] - (midpoints[0] - thresholds[0])],
            midpoints,
            [thresholds[-1] + (thresholds[-1] - midpoints[-1])],
        )
    )


def plot_score_significance_2d_heatmap(
    *,
    outpath: Path,
    values: np.ndarray,
    x_thresholds: np.ndarray,
    y_thresholds: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    colorbar_label: str,
    best_point: tuple[float, float] | None = None,
    value_format: str = ".3f",
) -> None:
    values = np.asarray(values, dtype=np.float64)
    x_thresholds = np.asarray(x_thresholds, dtype=np.float64)
    y_thresholds = np.asarray(y_thresholds, dtype=np.float64)
    if values.shape != (x_thresholds.size, y_thresholds.size):
        raise ValueError("2D heatmap values must have shape (n_x, n_y).")

    _setup_style()
    fig, ax = plt.subplots(dpi=150, figsize=(8.4, 6.8))
    mesh = ax.pcolormesh(
        _threshold_edges(x_thresholds),
        _threshold_edges(y_thresholds),
        values.T,
        shading="flat",
        cmap="viridis",
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label(colorbar_label)
    for x_index, x_value in enumerate(x_thresholds):
        for y_index, y_value in enumerate(y_thresholds):
            value = values[x_index, y_index]
            if np.isfinite(value):
                ax.text(
                    x_value,
                    y_value,
                    format(float(value), value_format),
                    ha="center",
                    va="center",
                    fontsize=5.4,
                    color="white" if value < 0.45 * np.nanmax(values) else "black",
                )
    if best_point is not None:
        ax.scatter(
            [best_point[0]],
            [best_point[1]],
            marker="*",
            s=150,
            facecolor="none",
            edgecolor="red",
            linewidth=1.5,
            label="Grid maximum",
            zorder=5,
        )
        ax.legend(frameon=False, loc="lower left")
    ax.set_xticks(x_thresholds)
    ax.set_yticks(y_thresholds)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    hep.cms.label(data=False, ax=ax)
    _save(fig, outpath)


def _qcd_group_style(
    groups: list[NnQcdScoreGroup], name: str
) -> tuple[str, str | None]:
    group = next(group for group in groups if group.name == name)
    return group.label, group.color


def plot_qcd_score_distribution(
    *,
    outpath: Path,
    bins: np.ndarray,
    histograms: dict[str, np.ndarray],
    groups: list[NnQcdScoreGroup],
    xscale: str = "linear",
) -> None:
    _setup_style()
    fig, ax = plt.subplots(dpi=150, figsize=(6.4, 4.8))
    for name in ("ttHcc", "qcd", "ttX"):
        label, color = _qcd_group_style(groups, name)
        ax.stairs(
            histograms[name],
            bins,
            color=color,
            linewidth=1.8,
            label=label,
        )
    ax.set_xlim(float(bins[0]), float(bins[-1]))
    ax.set_yscale("log")
    ax.set_xscale(xscale)
    ax.set_xlabel(
        r"$\log_{10}(\mathrm{score}_{\mathrm{QCD}})$"
        if xscale == "linear" and float(bins[0]) < 0.0
        else r"$\mathrm{score}_{\mathrm{QCD}}$"
    )
    ax.set_ylabel("Total expected events")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False)
    hep.cms.label(data=False, ax=ax)
    _save(fig, outpath)


def plot_qcd_working_point_distribution(
    *,
    outpath: Path,
    bins: np.ndarray,
    histograms: dict[str, np.ndarray],
    groups: list[NnQcdScoreGroup],
    reference_threshold: float | None,
    normalize: bool,
    xscale: str = "log",
) -> None:
    _setup_style()
    fig, ax = plt.subplots(dpi=150, figsize=(7.0, 5.2))
    for group in groups:
        values = np.asarray(histograms[group.name], dtype=np.float64)
        if np.sum(values) <= 0.0:
            continue
        ax.stairs(
            values,
            bins,
            color=group.color,
            linewidth=1.8,
            label=group.label,
        )
    if reference_threshold is not None:
        ax.axvspan(
            reference_threshold,
            float(bins[-1]),
            color="tab:red",
            alpha=0.10,
            linewidth=0.0,
            zorder=0,
        )
        ax.axvline(
            reference_threshold,
            color="tab:red",
            linestyle="--",
            linewidth=1.6,
            label=rf"reference cut $={reference_threshold:g}$",
        )
        rejected_label_x = (
            np.sqrt(reference_threshold * float(bins[-1]))
            if xscale == "log"
            else 0.5 * (reference_threshold + float(bins[-1]))
        )
        ax.text(
            rejected_label_x,
            0.58,
            "rejected",
            color="firebrick",
            fontsize=9,
            ha="center",
            va="center",
            transform=ax.get_xaxis_transform(),
        )
    ax.set_xscale(xscale)
    ax.set_yscale("log")
    ax.set_xlim(float(bins[0]), float(bins[-1]))
    ax.set_xlabel(r"$\mathrm{score}_{\mathrm{QCD}}$")
    ax.set_ylabel("Fraction of events" if normalize else "Total expected events")
    ax.grid(alpha=0.18, which="both")
    ax.legend(
        frameon=False,
        ncol=2,
        fontsize=8.5,
        columnspacing=1.0,
        handlelength=1.8,
    )
    hep.cms.label(data=False, ax=ax)
    _save(fig, outpath)


def plot_qcd_logx_distribution(
    *,
    outpath: Path,
    bins: np.ndarray,
    histograms: dict[str, np.ndarray],
    groups: list[NnQcdScoreGroup],
    reference_lines: list[float],
) -> None:
    _setup_style()
    fig, ax = plt.subplots(dpi=150, figsize=(7.2, 5.2))
    for group in groups:
        values = np.asarray(histograms[group.name], dtype=np.float64)
        if np.sum(values) <= 0.0:
            continue
        ax.stairs(
            values,
            bins,
            color=group.color,
            linewidth=1.8,
            label=group.label,
        )
    for value, linestyle, color, label, height in (
        (0.05, ":", "darkorange", "candidate 0.05", 0.79),
        (0.075, "-.", "steelblue", "nominal 0.075", 0.87),
        (0.10, "--", "firebrick", "reference 0.10", 0.95),
    ):
        if value not in reference_lines:
            continue
        ax.axvline(value, color=color, linestyle=linestyle, linewidth=1.1)
        ax.text(
            value,
            height,
            label,
            color=color,
            rotation=90,
            ha="right",
            va="top",
            fontsize=7.5,
            transform=ax.get_xaxis_transform(),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(bins[0]), float(bins[-1]))
    ax.set_xlabel(r"$\mathrm{score}_{\mathrm{qcd}}$")
    ax.set_ylabel("Fraction of events / log bin")
    ax.grid(alpha=0.18, which="both")
    ax.legend(frameon=False, ncol=2, fontsize=8.0)
    hep.cms.label(data=False, com=13, ax=ax)
    _save(fig, outpath)


def plot_qcd_analysis_region_distribution(
    *,
    outpath: Path,
    bins: np.ndarray,
    histograms: dict[str, np.ndarray],
    groups: list[NnQcdScoreGroup],
    ar_upper: float = 0.10,
    vr_upper: float = 0.20,
) -> None:
    bins = np.asarray(bins, dtype=np.float64)
    if (
        bins.ndim != 1
        or bins.size < 2
        or not np.all(np.diff(bins) > 0.0)
        or not bins[0] <= 0.0 < ar_upper < vr_upper <= bins[-1]
    ):
        raise ValueError("Invalid bins or QCD analysis-region boundaries.")
    for group in groups:
        values = np.asarray(histograms[group.name], dtype=np.float64)
        if values.shape != (bins.size - 1,):
            raise ValueError(
                f"QCD analysis-region histogram '{group.name}' has invalid shape."
            )

    _setup_style()
    fig, (left_ax, right_ax) = plt.subplots(
        ncols=2,
        sharey=True,
        dpi=150,
        figsize=(8.4, 5.4),
        gridspec_kw={"width_ratios": [3.0, 1.0], "wspace": 0.05},
    )
    for ax in (left_ax, right_ax):
        for group in groups:
            values = np.asarray(histograms[group.name], dtype=np.float64)
            if np.sum(values) <= 0.0:
                continue
            ax.stairs(
                values,
                bins,
                color=group.color,
                linewidth=1.8,
                label=group.label,
                zorder=3,
            )
        ax.set_yscale("log")
        ax.grid(alpha=0.18, which="both", zorder=1)

    left_ax.axvspan(
        0.0, ar_upper, color="seagreen", alpha=0.09, linewidth=0.0, zorder=0
    )
    left_ax.axvspan(
        ar_upper,
        vr_upper,
        color="darkorange",
        alpha=0.10,
        linewidth=0.0,
        zorder=0,
    )
    right_ax.axvspan(
        vr_upper,
        float(bins[-1]),
        color="firebrick",
        alpha=0.09,
        linewidth=0.0,
        zorder=0,
    )
    left_ax.axvline(
        ar_upper, color="darkgoldenrod", linestyle="--", linewidth=1.35, zorder=4
    )
    left_ax.axvline(
        vr_upper, color="firebrick", linestyle="--", linewidth=1.35, zorder=4
    )
    right_ax.axvline(
        vr_upper, color="firebrick", linestyle="--", linewidth=1.35, zorder=4
    )
    left_ax.text(
        0.5 * ar_upper,
        0.76,
        "AR",
        color="seagreen",
        fontweight="bold",
        ha="center",
        transform=left_ax.get_xaxis_transform(),
    )
    left_ax.text(
        0.5 * (ar_upper + vr_upper),
        0.76,
        "VR",
        color="darkgoldenrod",
        fontweight="bold",
        ha="center",
        transform=left_ax.get_xaxis_transform(),
    )
    right_ax.text(
        0.5 * (vr_upper + float(bins[-1])),
        0.66,
        "Rejected",
        color="firebrick",
        fontweight="bold",
        ha="center",
        transform=right_ax.get_xaxis_transform(),
    )

    left_ax.set_xlim(0.0, vr_upper)
    right_ax.set_xlim(vr_upper, float(bins[-1]))
    left_ax.set_xticks(np.linspace(0.0, vr_upper, 5))
    right_ax.set_xticks(np.linspace(0.4, float(bins[-1]), 4))
    left_ax.set_ylabel("Fraction of events / score unit")
    right_ax.tick_params(axis="y", left=False, labelleft=False)
    left_ax.spines["right"].set_visible(False)
    right_ax.spines["left"].set_visible(False)

    break_size = 0.012
    break_style = {
        "color": "black",
        "clip_on": False,
        "linewidth": 1.0,
    }
    left_ax.plot(
        (1.0 - break_size, 1.0 + break_size),
        (-break_size, break_size),
        transform=left_ax.transAxes,
        **break_style,
    )
    left_ax.plot(
        (1.0 - break_size, 1.0 + break_size),
        (1.0 - break_size, 1.0 + break_size),
        transform=left_ax.transAxes,
        **break_style,
    )
    right_ax.plot(
        (-break_size, break_size),
        (-break_size, break_size),
        transform=right_ax.transAxes,
        **break_style,
    )
    right_ax.plot(
        (-break_size, break_size),
        (1.0 - break_size, 1.0 + break_size),
        transform=right_ax.transAxes,
        **break_style,
    )

    right_ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=7.8,
        handlelength=1.8,
    )
    fig.supxlabel(r"$\mathrm{score}_{\mathrm{qcd}}$", y=0.02)
    hep.cms.label(data=False, com=13, ax=left_ax)
    fig.subplots_adjust(bottom=0.13)
    _save(fig, outpath)


def _qcd_veto_axis_ticks(
    veto_threshold: float, maximum: float
) -> tuple[np.ndarray, np.ndarray]:
    left_ticks = np.asarray([0.025, 0.050, veto_threshold])
    right_ticks = np.linspace(max(0.20, 2.0 * veto_threshold), maximum, 5)
    if right_ticks[0] <= veto_threshold:
        raise ValueError("Right QCD-veto ticks must start above the broken-axis boundary.")
    return left_ticks, right_ticks


def plot_qcd_veto_distribution(
    *,
    outpath: Path,
    bins: np.ndarray,
    histograms: dict[str, np.ndarray],
    groups: list[NnQcdScoreGroup],
    veto_threshold: float,
) -> None:
    bins = np.asarray(bins, dtype=np.float64)
    if (
        bins.ndim != 1
        or bins.size < 2
        or not np.all(np.diff(bins) > 0.0)
        or not bins[0] <= 0.0 < veto_threshold < bins[-1]
        or np.count_nonzero(np.isclose(bins, veto_threshold, rtol=0.0, atol=1e-12)) != 1
    ):
        raise ValueError("Invalid bins or standalone QCD-veto threshold.")
    for group in groups:
        values = np.asarray(histograms[group.name], dtype=np.float64)
        if values.shape != (bins.size - 1,):
            raise ValueError(
                f"QCD-veto histogram '{group.name}' has invalid shape."
            )

    _setup_style()
    fig, (left_ax, right_ax) = plt.subplots(
        ncols=2,
        sharey=True,
        dpi=150,
        figsize=(8.4, 5.4),
        gridspec_kw={"width_ratios": [2.4, 1.0], "wspace": 0.05},
    )
    for ax in (left_ax, right_ax):
        for group in groups:
            values = np.asarray(histograms[group.name], dtype=np.float64)
            if np.sum(values) <= 0.0:
                continue
            ax.stairs(
                values,
                bins,
                color=group.color,
                linewidth=1.8,
                label=group.label,
                zorder=3,
            )
        ax.set_yscale("log")
        ax.grid(alpha=0.18, which="both", zorder=1)

    left_ax.axvspan(
        float(bins[0]),
        veto_threshold,
        color="seagreen",
        alpha=0.09,
        linewidth=0.0,
        zorder=0,
    )
    right_ax.axvspan(
        veto_threshold,
        float(bins[-1]),
        color="firebrick",
        alpha=0.09,
        linewidth=0.0,
        zorder=0,
    )
    for ax in (left_ax, right_ax):
        ax.axvline(
            veto_threshold,
            color="firebrick",
            linestyle="--",
            linewidth=1.35,
            zorder=4,
        )
    left_ax.text(
        0.5 * veto_threshold,
        0.76,
        "QCD-veto pass",
        color="seagreen",
        fontweight="bold",
        ha="center",
        transform=left_ax.get_xaxis_transform(),
    )
    right_ax.text(
        0.5 * (veto_threshold + float(bins[-1])),
        0.66,
        "Rejected",
        color="firebrick",
        fontweight="bold",
        ha="center",
        transform=right_ax.get_xaxis_transform(),
    )

    left_ax.set_xlim(float(bins[0]), veto_threshold)
    right_ax.set_xlim(veto_threshold, float(bins[-1]))
    left_ticks, right_ticks = _qcd_veto_axis_ticks(veto_threshold, float(bins[-1]))
    left_ax.set_xticks(left_ticks, [f"{value:.3f}" for value in left_ticks])
    right_ax.set_xticks(right_ticks, [f"{value:.3g}" for value in right_ticks])
    left_ax.set_ylabel("Normalized density")
    right_ax.tick_params(axis="y", left=False, labelleft=False)
    left_ax.spines["right"].set_visible(False)
    right_ax.spines["left"].set_visible(False)

    break_size = 0.012
    break_style = {"color": "black", "clip_on": False, "linewidth": 1.0}
    for y0, y1 in ((-break_size, break_size), (1.0 - break_size, 1.0 + break_size)):
        left_ax.plot(
            (1.0 - break_size, 1.0 + break_size),
            (y0, y1),
            transform=left_ax.transAxes,
            **break_style,
        )
        right_ax.plot(
            (-break_size, break_size),
            (y0, y1),
            transform=right_ax.transAxes,
            **break_style,
        )

    right_ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=7.8,
        handlelength=1.8,
    )
    fig.supxlabel(r"$\mathrm{score}_{\mathrm{qcd}}$", y=0.02)
    hep.cms.label(data=False, com=13, ax=left_ax)
    fig.subplots_adjust(bottom=0.13)
    _save(fig, outpath)


def plot_qcd_expected_yields(
    *,
    outpath: Path,
    thresholds: np.ndarray,
    yields: dict[str, np.ndarray],
    groups: list[NnQcdScoreGroup],
    xscale: str = "log",
) -> None:
    _setup_style()
    fig, ax = plt.subplots(dpi=150, figsize=(6.4, 4.8))
    for name in ("qcd", "ttX", "ttHcc"):
        label, color = _qcd_group_style(groups, name)
        values = np.asarray(yields[name], dtype=np.float64)
        valid = np.isfinite(values) & (values > 0.0)
        ax.plot(thresholds[valid], values[valid], color=color, label=label)
    ax.set_xscale(xscale)
    ax.set_yscale("log")
    ax.set_xlabel(r"Cut on $\mathrm{score}_{\mathrm{QCD}}$ ($<$ cut)")
    ax.set_ylabel("Remaining expected events")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False)
    hep.cms.label(data=False, ax=ax)
    _save(fig, outpath)


def plot_qcd_efficiencies(
    *,
    outpath: Path,
    thresholds: np.ndarray,
    tthcc_efficiency: np.ndarray,
    qcd_efficiency: np.ndarray,
    groups: list[NnQcdScoreGroup],
    xscale: str = "log",
) -> None:
    _setup_style()
    fig, ax = plt.subplots(dpi=150, figsize=(6.4, 4.8))
    tthcc_label, tthcc_color = _qcd_group_style(groups, "ttHcc")
    qcd_label, qcd_color = _qcd_group_style(groups, "qcd")
    ax.plot(
        thresholds,
        tthcc_efficiency,
        color=tthcc_color,
        label=f"{tthcc_label} efficiency",
    )
    ax.plot(
        thresholds,
        qcd_efficiency,
        color=qcd_color,
        label=f"{qcd_label} efficiency",
    )
    ax.plot(
        thresholds,
        1.0 - qcd_efficiency,
        color=qcd_color,
        linestyle="--",
        label=f"{qcd_label} rejection",
    )
    ax.set_xscale(xscale)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Cut on $\mathrm{score}_{\mathrm{QCD}}$ ($<$ cut)")
    ax.set_ylabel("Weighted efficiency / rejection")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False)
    hep.cms.label(data=False, ax=ax)
    _save(fig, outpath)


def plot_qcd_roc_like(
    *,
    outpath: Path,
    tthcc_efficiency: np.ndarray,
    qcd_efficiency: np.ndarray,
    candidate_points: list[dict[str, float | None]],
) -> None:
    _setup_style()
    fig, ax = plt.subplots(dpi=150, figsize=(6.4, 5.2))
    valid = (
        np.isfinite(tthcc_efficiency)
        & np.isfinite(qcd_efficiency)
        & (qcd_efficiency > 0.0)
    )
    ax.plot(tthcc_efficiency[valid], qcd_efficiency[valid], color="black")
    positive_efficiencies = qcd_efficiency[np.isfinite(qcd_efficiency) & (qcd_efficiency > 0.0)]
    display_floor = (
        max(float(np.min(positive_efficiencies)) * 0.35, 1e-8)
        if positive_efficiencies.size
        else 1e-8
    )
    for point in candidate_points:
        x = point["ttHcc_efficiency"]
        y = point["qcd_efficiency"]
        if x is None or y is None:
            continue
        display_y = y if y > 0.0 else display_floor
        marker = "o" if y > 0.0 else "v"
        ax.scatter(x, display_y, s=24, marker=marker, color="tab:red", zorder=3)
        ax.annotate(
            f"{point['cut']:g}" if y > 0.0 else f"{point['cut']:g} (0)",
            (x, display_y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlim(0.0, 1.05)
    ax.set_yscale("log")
    ax.set_ylim(display_floor * 0.7, None)
    ax.set_xlabel("ttHcc weighted efficiency")
    ax.set_ylabel("QCD weighted efficiency")
    ax.grid(alpha=0.2, which="both")
    hep.cms.label(data=False, ax=ax)
    _save(fig, outpath)


def plot_qcd_to_ttx_ratio(
    *, outpath: Path, thresholds: np.ndarray, ratio: np.ndarray, xscale: str = "log"
) -> None:
    _setup_style()
    fig, ax = plt.subplots(dpi=150, figsize=(6.4, 4.8))
    valid = np.isfinite(ratio) & (ratio > 0.0)
    ax.plot(thresholds[valid], ratio[valid], color="tab:purple")
    ax.set_xscale(xscale)
    ax.set_yscale("log")
    ax.set_xlabel(r"Cut on $\mathrm{score}_{\mathrm{QCD}}$ ($<$ cut)")
    ax.set_ylabel("Remaining QCD / tt+X yield")
    ax.grid(alpha=0.2, which="both")
    hep.cms.label(data=False, ax=ax)
    _save(fig, outpath)


def plot_qcd_relative_significance(
    *,
    outpath: Path,
    thresholds: np.ndarray,
    qcd_only: np.ndarray,
    qcd_plus_ttx: np.ndarray,
    xscale: str = "log",
) -> None:
    _setup_style()
    fig, ax = plt.subplots(dpi=150, figsize=(6.4, 4.8))
    for values, label, color in [
        (qcd_only, r"$B=\mathrm{QCD}$", "tab:green"),
        (qcd_plus_ttx, r"$B=\mathrm{QCD}+\mathrm{tt+X}$", "tab:blue"),
    ]:
        valid = np.isfinite(values)
        ax.plot(thresholds[valid], values[valid], color=color, label=label)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xscale(xscale)
    ax.set_xlabel(r"Cut on $\mathrm{score}_{\mathrm{QCD}}$ ($<$ cut)")
    ax.set_ylabel(r"Relative $S/\sqrt{B}$")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False)
    hep.cms.label(data=False, ax=ax)
    _save(fig, outpath)


def plot_score_significance(
    *,
    outpath: Path,
    thresholds: np.ndarray,
    signal_results: dict[str, dict[str, object]],
    signal_styles: dict[str, dict[str, str | None]],
    score_branch: str,
    direction: str,
    xscale: str,
    candidate_thresholds: list[float] | None = None,
    selection_label: str | None = None,
) -> None:
    _setup_style()
    fig, ax = plt.subplots(dpi=150, figsize=(7.4, 5.4))
    fallback_colors = plt.get_cmap("tab10").colors
    for index, (signal_name, result) in enumerate(signal_results.items()):
        points = list(result["points"])
        significance = np.asarray(
            [point["significance"] for point in points], dtype=np.float64
        )
        style = signal_styles[signal_name]
        color = style.get("color") or fallback_colors[index % len(fallback_colors)]
        label = str(style.get("label") or signal_name)
        ax.plot(
            thresholds,
            significance,
            color=color,
            linewidth=2.0,
            label=rf"{label}: $S/\sqrt{{S+B}}$",
        )
        best_index = int(result["best_scan_index"])
        best_cut = float(result["best_threshold"])
        best_significance = float(result["best_significance"])
        ax.scatter(
            [best_cut],
            [best_significance],
            color=color,
            edgecolor="white",
            linewidth=0.7,
            s=42,
            zorder=4,
        )
        ax.annotate(
            f"{label}: best cut = {best_cut:.4g}\nmax Z = {best_significance:.4g}",
            (thresholds[best_index], significance[best_index]),
            xytext=(8, 10 - 28 * index),
            textcoords="offset points",
            color=color,
            fontsize=8,
            bbox={
                "facecolor": "white",
                "alpha": 0.78,
                "edgecolor": "none",
                "pad": 1.5,
            },
        )
    for threshold in candidate_thresholds or []:
        ax.axvline(
            threshold,
            color="gray",
            linestyle=":",
            linewidth=0.8,
            alpha=0.38,
            zorder=0,
        )
    ax.set_xscale(xscale)
    ax.set_xlim(float(thresholds[0]), float(thresholds[-1]))
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(f"Cut on {score_branch}: keep {score_branch} {direction} cut")
    ax.set_ylabel(r"Expected significance $S/\sqrt{S+B}$")
    if selection_label:
        ax.text(
            0.98,
            0.96,
            selection_label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={
                "facecolor": "white",
                "alpha": 0.78,
                "edgecolor": "none",
                "pad": 1.5,
            },
        )
    ax.grid(alpha=0.22, which="both")
    ax.legend(frameon=False, ncol=2, fontsize=8.5, loc="best")
    hep.cms.label(data=False, ax=ax)
    _save(fig, outpath)


def plot_score_significance_comparison(
    *,
    outpath: Path,
    raw_result: dict[str, object],
    discriminant_result: dict[str, object],
    signal_label: str,
    selection_label: str | None = None,
    raw_label: str = r"raw $score_{ttHZ}$",
    discriminant_label: str = r"persisted $D_{ttHZ}$",
    title: str | None = None,
    cms_com: float | None = None,
) -> None:
    _setup_style()
    fig, ax = plt.subplots(dpi=150, figsize=(7.0, 5.0))
    for result, label, color in (
        (raw_result, raw_label, "royalblue"),
        (discriminant_result, discriminant_label, "darkorange"),
    ):
        points = list(result["points"])
        cuts = np.asarray([point["cut"] for point in points], dtype=np.float64)
        significance = np.asarray(
            [point["significance"] for point in points], dtype=np.float64
        )
        ax.plot(cuts, significance, color=color, linewidth=2.0, label=label)
        best_index = int(result["best_scan_index"])
        ax.scatter(
            cuts[best_index],
            significance[best_index],
            color=color,
            edgecolor="white",
            linewidth=0.7,
            s=42,
            zorder=4,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Cut value (keep score > cut)")
    ax.set_ylabel(r"Expected significance $S/\sqrt{S+B}$")
    ax.set_title(title or f"{signal_label}: raw score vs persisted discriminant")
    if selection_label:
        ax.text(
            0.98,
            0.96,
            selection_label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
        )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    cms_kwargs = {} if cms_com is None else {"com": cms_com}
    hep.cms.label(data=False, ax=ax, **cms_kwargs)
    _save(fig, outpath)


def plot_confusion_matrix(
    *,
    outpath: Path,
    matrix: np.ndarray,
    truths: list[NnTruthCategory],
    scores: list[NnScoreClass],
    normalization: str,
    config: NnStudyConfig,
    title: str | None = None,
) -> None:
    _setup_style()
    fig, ax = plt.subplots(
        dpi=int(config.plot_options.get("confusion_dpi", 150)),
        figsize=(8, 8),
    )
    cmap = "Blues" if normalization == "truth" else "Greens"
    display = matrix.T[:, ::-1]
    artists = hep.hist2dplot(
        display,
        ax=ax,
        labels=np.round(display, 4),
        cmap=cmap,
        flow=None,
    )
    for label in artists.text:
        label.set_fontsize(8)
    ax.set_xticks(
        np.arange(len(scores)) + 0.5,
        [score.label for score in scores],
        rotation=90,
    )
    ax.set_yticks(
        np.arange(len(truths)) + 0.5,
        [truth.label for truth in truths][::-1],
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Truth label")
    default_title = (
        "Confusion matrix per truth class (row normalized)"
        if normalization == "truth"
        else "Confusion matrix per predicted class (column normalized)"
    )
    ax.set_title(title or default_title)
    if config.uses_distinct_prediction_truth:
        hep.cms.label(data=False, ax=ax)
    fig.tight_layout()
    _save(fig, outpath)


def plot_pairwise_roc(
    *,
    outpath: Path,
    signal: NnScoreClass,
    background_results: dict[str, PairwiseRoc],
    truths_by_name: dict[str, NnTruthCategory],
    config: NnStudyConfig,
) -> None:
    _setup_style()
    fig, ax = plt.subplots(dpi=int(config.plot_options.get("roc_dpi", 100)))
    for background_name, result in background_results.items():
        truth = truths_by_name[background_name]
        ax.plot(
            result.signal_efficiency,
            result.background_efficiency,
            label=f"{truth.label}: AUC {result.auc:.3f}",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(
        float(config.plot_options.get("roc_y_min", 1e-5)),
        float(config.plot_options.get("roc_y_max", 2.0)),
    )
    ax.set_yscale("log")
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background efficiency")
    ax.set_title(f"ROC curve for {signal.label}")
    ax.legend(title=f"{signal.label} vs..")
    if config.uses_distinct_prediction_truth:
        hep.cms.label(data=False, ax=ax)
    _save(fig, outpath)


def plot_auc_matrix(
    *,
    outpath: Path,
    auc_matrix: np.ndarray,
    scores: list[NnScoreClass],
    config: NnStudyConfig,
    title: str | None = None,
) -> None:
    _setup_style()
    fig, ax = plt.subplots(
        dpi=int(config.plot_options.get("auc_dpi", 150)),
        figsize=(10, 10),
    )
    display = np.nan_to_num(auc_matrix[:, ::-1], nan=0.0)
    display_labels = np.where(display >= 0.5, np.round(display, 3), np.nan)
    hep.hist2dplot(
        display,
        ax=ax,
        labels=display_labels,
        cmin=0.5,
        cmax=1.0,
        flow=None,
    )
    labels = [score.label for score in scores]
    ax.set_xticks(np.arange(len(scores)) + 0.5, labels, rotation=90)
    ax.set_yticks(np.arange(len(scores)) + 0.5, labels[::-1])
    ax.set_title(title or "AUC for all signal vs background pairs")
    if config.uses_distinct_prediction_truth:
        hep.cms.label(data=False, ax=ax)
    _save(fig, outpath)


def build_auc_matrix(
    score_names: list[str], pairwise: dict[str, dict[str, PairwiseRoc]]
) -> np.ndarray:
    matrix = np.full((len(score_names), len(score_names)), np.nan, dtype=np.float64)
    for row, signal in enumerate(score_names):
        for column, background in enumerate(score_names):
            result = pairwise.get(signal, {}).get(background)
            if result is not None:
                matrix[row, column] = result.auc
    return matrix
