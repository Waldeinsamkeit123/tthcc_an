from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


PROCESS_COLORS = {
    "ttHbb": "#1f77b4",
    "ttHcc": "#d62728",
}


def _setup_style() -> None:
    plt.style.use(hep.style.CMS)


def _cms_label(fig: plt.Figure) -> None:
    fig.text(0.015, 0.995, "CMS", ha="left", va="top", fontsize=16, fontweight="bold")
    fig.text(0.090, 0.995, "Private Work", ha="left", va="top", fontsize=15)
    fig.text(0.985, 0.995, "2024 (13.6 TeV)", ha="right", va="top", fontsize=15)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _efficiency_error(efficiency: np.ndarray, denominator: np.ndarray, denominator_w2: np.ndarray) -> np.ndarray:
    neff = np.divide(
        denominator * denominator,
        denominator_w2,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator_w2 > 0,
    )
    variance = np.divide(
        efficiency * (1.0 - efficiency),
        neff,
        out=np.zeros_like(efficiency, dtype=np.float64),
        where=neff > 0,
    )
    return np.sqrt(np.clip(variance, 0.0, None))


def _draw_efficiency_panel(
    ax: plt.Axes,
    *,
    trigger: str,
    variable_label: str,
    bin_edges: np.ndarray,
    processes: list[str],
    process_labels: dict[str, str],
    denominator: dict[str, np.ndarray],
    denominator_w2: dict[str, np.ndarray],
    numerator: dict[str, dict[str, np.ndarray]],
) -> None:
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = 0.5 * np.diff(bin_edges)
    for process in processes:
        den = denominator[process]
        num = numerator[trigger][process]
        eff = np.divide(num, den, out=np.full_like(den, np.nan, dtype=np.float64), where=den > 0)
        err = _efficiency_error(eff, den, denominator_w2[process])
        finite = np.isfinite(eff)
        if not np.any(finite):
            continue
        ax.errorbar(
            centers[finite],
            eff[finite],
            xerr=widths[finite],
            yerr=err[finite],
            marker="o",
            markersize=3.2,
            linewidth=1.4,
            capsize=1.5,
            color=PROCESS_COLORS.get(process, None),
            label=process_labels.get(process, process),
        )
    ax.set_title(trigger.replace("HLT_", ""), fontsize=9.2)
    ax.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel(variable_label, fontsize=10)
    ax.set_ylabel("Trigger efficiency", fontsize=10)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.grid(alpha=0.25)


def plot_trigger_group(
    *,
    outpath: Path,
    group_label: str,
    triggers: list[str],
    variable_label: str,
    bin_edges: np.ndarray,
    processes: list[str],
    process_labels: dict[str, str],
    denominator: dict[str, np.ndarray],
    denominator_w2: dict[str, np.ndarray],
    numerator: dict[str, dict[str, np.ndarray]],
) -> None:
    _setup_style()
    n_panels = len(triggers)
    ncols = 3 if n_panels > 4 else 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.25 * nrows), squeeze=False)
    for ax, trigger in zip(axes.ravel(), triggers):
        _draw_efficiency_panel(
            ax,
            trigger=trigger,
            variable_label=variable_label,
            bin_edges=bin_edges,
            processes=processes,
            process_labels=process_labels,
            denominator=denominator,
            denominator_w2=denominator_w2,
            numerator=numerator,
        )
    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(group_label, fontsize=15, y=0.998)
    _cms_label(fig)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    _save(fig, outpath)
