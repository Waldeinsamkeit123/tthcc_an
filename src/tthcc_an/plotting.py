from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, to_rgba
from scipy.ndimage import gaussian_filter

from tthcc_an.definitions import (
    GLOBALPART3_CONTOUR_CATEGORIES,
    GLOBALPART3_CONTOUR_CLIP_EPS,
    GLOBALPART3_CONTOUR_ENCLOSED_FRACTIONS,
    GLOBALPART3_CONTOUR_HIST_BINS,
    GLOBALPART3_CONTOUR_PLOT_BY_KEY,
    GLOBALPART3_CONTOUR_SMOOTH_SIGMA,
    GLOBALPART3_FIXED_OTHER_EFF_TARGETS,
    GLOBALPART3_FIXED_X_CUT,
    SCORE_LABELS,
    TARGET_DEFINITIONS,
    TRUTH_LABEL_COLORS,
    TRUTH_LABEL_TITLES,
    TRUTH_LABEL_TO_CODE,
)
from tthcc_an.metrics import (
    build_target_masks,
    codes_for_labels,
    pass_from_hist,
    truth_mask,
    weighted_quantile,
    weighted_sum,
    working_point_bin_index,
)


DISPLAY_SCORE_BINS = 50
DISPLAY_PROCESS_SCORE_BINS = 50
CONTOUR_REGION_LINE_COLOR = "#2f2f2f"
CONTOUR_REGION_LINEWIDTH = 1.1


@dataclass
class PlotStyle:
    title_size: float
    label_size: float
    tick_size: float
    legend_size: float
    cms_size: float
    dpi: int


def build_plot_style(args: argparse.Namespace) -> PlotStyle:
    return PlotStyle(
        title_size=args.plot_title_size,
        label_size=args.plot_label_size,
        tick_size=args.plot_tick_size,
        legend_size=args.plot_legend_size,
        cms_size=args.plot_cms_size,
        dpi=args.plot_dpi,
    )


def _cms_label(ax: plt.Axes, plot_style: PlotStyle) -> None:
    hep.cms.label(
        data=False,
        ax=ax,
        fontsize=plot_style.cms_size,
        rlabel="2024 (13.6 TeV)",
    )


def private_cms_label(ax: plt.Axes, plot_style: PlotStyle) -> None:
    hep.cms.label(
        label="Private Work",
        data=False,
        ax=ax,
        fontsize=plot_style.cms_size,
        rlabel="2024 (13.6 TeV)",
    )


def _inside_legend(
    ax: plt.Axes,
    plot_style: PlotStyle,
    *,
    loc: str = "upper right",
    ncol: int = 1,
) -> None:
    legend = ax.legend(
        fontsize=plot_style.legend_size,
        loc=loc,
        bbox_to_anchor=(0.975, 0.985),
        ncol=ncol,
        frameon=False,
        borderpad=0.25,
        borderaxespad=0.0,
        labelspacing=0.3,
        handlelength=1.6,
        columnspacing=0.8,
    )
    if legend is not None:
        legend.set_zorder(5)


def _scientific_tick_label(value: float, _position: float) -> str:
    if not np.isfinite(value):
        return ""
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    mantissa_text = f"{mantissa:.1f}".rstrip("0").rstrip(".")
    if mantissa_text == "1":
        return rf"$10^{{{exponent}}}$"
    return rf"${mantissa_text}\times10^{{{exponent}}}$"


def _scientific_yaxis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_scientific_tick_label))
    ax.yaxis.offsetText.set_visible(False)


def _rebin_histogram_for_display(
    hist: np.ndarray,
    edges: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    display_edges = np.linspace(float(edges[0]), float(edges[-1]), n_bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rebinned, _ = np.histogram(centers, bins=display_edges, weights=hist)
    return rebinned.astype(np.float64), display_edges


def _histogram_density(hist: np.ndarray, edges: np.ndarray) -> np.ndarray:
    total = float(np.sum(hist))
    if total <= 0:
        return np.zeros_like(hist, dtype=np.float64)
    widths = np.diff(edges)
    density = hist.astype(np.float64) / total
    positive = widths > 0
    density[positive] = density[positive] / widths[positive]
    return density


def _save_plot(
    fig: plt.Figure,
    outpath: Path,
    plot_style: PlotStyle,
    *,
    left: float = 0.125,
    right: float = 0.97,
    bottom: float = 0.14,
    top: float = 0.875,
    save_pdf: bool = False,
) -> None:
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    fig.savefig(outpath, dpi=plot_style.dpi)
    if save_pdf and outpath.suffix.lower() != ".pdf":
        fig.savefig(outpath.with_suffix(".pdf"))
    plt.close(fig)


def _smoothed_normalized_histogram(hist2d: np.ndarray) -> np.ndarray:
    hist_smooth = gaussian_filter(
        np.asarray(hist2d, dtype=np.float64),
        sigma=GLOBALPART3_CONTOUR_SMOOTH_SIGMA,
        mode="nearest",
    )
    total = float(np.sum(hist_smooth))
    if total <= 0:
        return np.zeros_like(hist_smooth, dtype=np.float64)
    return hist_smooth / total


def _enclosed_fraction_contour_levels(hist_norm: np.ndarray) -> np.ndarray:
    flat = np.asarray(hist_norm, dtype=np.float64).ravel()
    positive = np.isfinite(flat) & (flat > 0)
    if not np.any(positive):
        return np.array([], dtype=np.float64)
    sorted_density = flat[positive][np.argsort(flat[positive])[::-1]]
    cumsum = np.cumsum(sorted_density)
    thresholds: list[float] = []
    for fraction in GLOBALPART3_CONTOUR_ENCLOSED_FRACTIONS:
        clipped_fraction = min(max(float(fraction), 0.0), 1.0)
        threshold_index = int(np.searchsorted(cumsum, clipped_fraction, side="left"))
        threshold_index = min(threshold_index, len(sorted_density) - 1)
        thresholds.append(float(sorted_density[threshold_index]))
    contour_levels = np.sort(np.asarray(thresholds, dtype=np.float64))
    contour_levels = contour_levels[np.isfinite(contour_levels) & (contour_levels > 0)]
    contour_levels = np.unique(contour_levels)
    return contour_levels


def _filled_contour_levels(contour_levels: np.ndarray, hist_norm: np.ndarray) -> np.ndarray:
    if contour_levels.size == 0:
        return np.array([], dtype=np.float64)
    max_density = float(np.max(hist_norm))
    if not np.isfinite(max_density) or max_density <= 0:
        return np.array([], dtype=np.float64)
    if max_density <= contour_levels[-1]:
        max_density = float(np.nextafter(contour_levels[-1], np.inf))
    return np.concatenate([contour_levels, np.array([max_density], dtype=np.float64)])


def _make_alpha_cmap(base_color: str, name: str, n_intervals: int) -> ListedColormap:
    rgba = np.asarray(to_rgba(base_color), dtype=np.float64)
    n_steps = max(int(n_intervals), 1)
    if n_steps == 1:
        alpha_values = np.array([0.68], dtype=np.float64)
    else:
        # Keep the outermost 80%-90% shell light, while making each inner shell
        # progressively darker and visually distinct.
        alpha_values = np.linspace(0.18, 0.92, n_steps, dtype=np.float64)
    colors = [
        (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(alpha))
        for alpha in alpha_values
    ]
    return ListedColormap(colors, name=name)


def _sorted_contour_histograms(
    contour_histograms: list[tuple[dict[str, Any], np.ndarray]],
) -> list[tuple[dict[str, Any], np.ndarray]]:
    draw_order = {
        "others": 0,
        "hcc_pure": 1,
        "hbb_pure": 2,
    }
    return sorted(contour_histograms, key=lambda item: draw_order.get(str(item[0]["key"]), 99))


def _globalpart3_contour_handles(categories: list[dict[str, Any]]) -> list[Line2D]:
    legend_order = {
        "hbb_pure": 0,
        "hcc_pure": 1,
        "others": 2,
    }
    legend_label_map = {
        "hbb_pure": "hbb_pure jets & efficiencies (%)",
        "hcc_pure": "hcc_pure jets & efficiencies (%)",
        "others": "Others jets & efficiencies (%)",
    }
    return [
        Line2D(
            [0],
            [0],
            color=category["color"],
            linewidth=1.7,
            alpha=0.95,
            label=legend_label_map.get(str(category["key"]), str(category.get("legend_label", category["key"]))),
        )
        for category in sorted(categories, key=lambda item: legend_order.get(str(item["key"]), 99))
    ]


def _resolve_contour_plot_def(plot_def: dict[str, Any] | None = None, *, plot_key: str | None = None) -> dict[str, Any]:
    if plot_def is not None:
        fallback_key = plot_key or str(plot_def.get("key", ""))
        resolved = dict(GLOBALPART3_CONTOUR_PLOT_BY_KEY.get(fallback_key, {}))
        if not resolved:
            return dict(plot_def)
        for key, value in plot_def.items():
            if key == "region_definitions":
                resolved[key] = dict(value)
            elif key == "boundary_segments":
                resolved[key] = list(value)
            else:
                resolved[key] = value
        return resolved
    if plot_key is None:
        raise ValueError("A contour plot definition or plot key is required.")
    if plot_key not in GLOBALPART3_CONTOUR_PLOT_BY_KEY:
        raise KeyError(f"Unknown contour plot key: {plot_key}")
    return dict(GLOBALPART3_CONTOUR_PLOT_BY_KEY[plot_key])


def _rectangular_region_mask(x_values: np.ndarray, y_values: np.ndarray, region_def: dict[str, Any]) -> np.ndarray:
    mask = np.ones_like(x_values, dtype=bool)
    if "x_min_exclusive" in region_def:
        mask &= x_values > float(region_def["x_min_exclusive"])
    if "x_min_inclusive" in region_def:
        mask &= x_values >= float(region_def["x_min_inclusive"])
    if "x_max_exclusive" in region_def:
        mask &= x_values < float(region_def["x_max_exclusive"])
    if "x_max_inclusive" in region_def:
        mask &= x_values <= float(region_def["x_max_inclusive"])
    if "y_min_exclusive" in region_def:
        mask &= y_values > float(region_def["y_min_exclusive"])
    if "y_min_inclusive" in region_def:
        mask &= y_values >= float(region_def["y_min_inclusive"])
    if "y_max_exclusive" in region_def:
        mask &= y_values < float(region_def["y_max_exclusive"])
    if "y_max_inclusive" in region_def:
        mask &= y_values <= float(region_def["y_max_inclusive"])
    return mask


def _globalpart3_region_masks(
    x_values: np.ndarray,
    y_values: np.ndarray,
    plot_def: dict[str, Any],
) -> dict[str, np.ndarray]:
    region_definitions = dict(plot_def.get("region_definitions", {}))
    hcc_region = _rectangular_region_mask(x_values, y_values, dict(region_definitions.get("hcc", {})))
    hbb_region = _rectangular_region_mask(x_values, y_values, dict(region_definitions.get("hbb", {})))
    qcd_region = ~(hcc_region | hbb_region)
    return {
        "qcd_others": qcd_region,
        "hcc": hcc_region,
        "hbb": hbb_region,
    }


def _compute_contour_region_efficiencies_from_raw(
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    truth_codes: np.ndarray,
    weights: np.ndarray,
    plot_def: dict[str, Any],
) -> dict[str, dict[str, float]]:
    positive_weights = np.isfinite(weights) & (weights > 0)
    use_weights = bool(np.any(positive_weights))
    effective_weights = np.asarray(weights, dtype=np.float64) if use_weights else np.ones_like(x_scores, dtype=np.float64)
    valid = np.isfinite(x_scores) & np.isfinite(y_scores) & np.isfinite(effective_weights)
    if use_weights:
        valid &= effective_weights > 0
    if not np.any(valid):
        return {}

    x_valid = np.clip(np.asarray(x_scores[valid], dtype=np.float64), GLOBALPART3_CONTOUR_CLIP_EPS, 1.0 - GLOBALPART3_CONTOUR_CLIP_EPS)
    y_valid = np.clip(np.asarray(y_scores[valid], dtype=np.float64), GLOBALPART3_CONTOUR_CLIP_EPS, 1.0 - GLOBALPART3_CONTOUR_CLIP_EPS)
    truth_valid = np.asarray(truth_codes[valid], dtype=np.int32)
    weights_valid = np.asarray(effective_weights[valid], dtype=np.float64)
    region_masks = _globalpart3_region_masks(x_valid, y_valid, plot_def)

    category_code_map = {
        "hbb_pure": TRUTH_LABEL_TO_CODE["hbb_pure"],
        "hcc_pure": TRUTH_LABEL_TO_CODE["hcc_pure"],
    }
    category_masks = {
        "hbb_pure": truth_valid == category_code_map["hbb_pure"],
        "hcc_pure": truth_valid == category_code_map["hcc_pure"],
    }
    category_masks["others"] = ~(category_masks["hbb_pure"] | category_masks["hcc_pure"])

    region_efficiencies: dict[str, dict[str, float]] = {}
    for region_key, region_mask in region_masks.items():
        region_efficiencies[region_key] = {}
        for category_key, category_mask in category_masks.items():
            total = float(np.sum(weights_valid[category_mask]))
            if total <= 0:
                region_efficiencies[region_key][category_key] = float("nan")
                continue
            inside = float(np.sum(weights_valid[category_mask & region_mask]))
            region_efficiencies[region_key][category_key] = inside / total
    return region_efficiencies


def _compute_contour_region_efficiencies_from_hist(
    categories: list[dict[str, Any]],
    weight_category: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    plot_def: dict[str, Any],
) -> dict[str, dict[str, float]]:
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    mesh_x, mesh_y = np.meshgrid(x_centers, y_centers, indexing="ij")
    region_masks = _globalpart3_region_masks(mesh_x, mesh_y, plot_def)

    category_index_map = {str(category["key"]): index for index, category in enumerate(categories)}
    region_efficiencies: dict[str, dict[str, float]] = {}
    for region_key, region_mask in region_masks.items():
        region_efficiencies[region_key] = {}
        for category_key in ["hbb_pure", "hcc_pure", "others"]:
            if category_key not in category_index_map:
                region_efficiencies[region_key][category_key] = float("nan")
                continue
            hist = np.asarray(weight_category[category_index_map[category_key]], dtype=np.float64)
            total = float(np.sum(hist))
            if total <= 0:
                region_efficiencies[region_key][category_key] = float("nan")
                continue
            inside = float(np.sum(hist[region_mask]))
            region_efficiencies[region_key][category_key] = inside / total
    return region_efficiencies


def _draw_globalpart3_region_boundaries(ax: plt.Axes, plot_def: dict[str, Any]) -> None:
    for segment in plot_def.get("boundary_segments", []):
        ax.plot(
            list(segment["x"]),
            list(segment["y"]),
            color=CONTOUR_REGION_LINE_COLOR,
            linewidth=CONTOUR_REGION_LINEWIDTH,
            zorder=8.0,
        )


def _format_region_efficiency(efficiency: float) -> str:
    if not np.isfinite(efficiency):
        return "n/a"
    return f"{efficiency * 100.0:.1f}%"


def _annotate_globalpart3_regions(
    ax: plt.Axes,
    region_efficiencies: dict[str, dict[str, float]],
    plot_style: PlotStyle,
    plot_def: dict[str, Any],
) -> None:
    region_definitions = dict(plot_def.get("region_definitions", {}))
    annotation_specs = [
        (
            "qcd_others",
            str(region_definitions.get("qcd_others", {}).get("label", "QCD&Others region")),
            float(region_definitions.get("qcd_others", {}).get("annotation", {}).get("x", 0.15)),
            float(region_definitions.get("qcd_others", {}).get("annotation", {}).get("y", 0.80)),
        ),
        (
            "hcc",
            str(region_definitions.get("hcc", {}).get("label", "Hcc region")),
            float(region_definitions.get("hcc", {}).get("annotation", {}).get("x", 0.55)),
            float(region_definitions.get("hcc", {}).get("annotation", {}).get("y", 0.65)),
        ),
        (
            "hbb",
            str(region_definitions.get("hbb", {}).get("label", "Hbb region")),
            float(region_definitions.get("hbb", {}).get("annotation", {}).get("x", 0.78)),
            float(region_definitions.get("hbb", {}).get("annotation", {}).get("y", 0.88)),
        ),
    ]
    category_styles = [
        ("hbb_pure", "#ff7f0e"),
        ("hcc_pure", "#d62728"),
        ("others", "#1f77b4"),
    ]

    title_fontsize = max(plot_style.tick_size + 0.3, 9.0)
    value_fontsize = max(plot_style.tick_size - 0.2, 8.5)
    line_spacing = 0.045

    for region_key, title, x_pos, y_pos in annotation_specs:
        ax.text(
            x_pos,
            y_pos,
            title,
            color=CONTOUR_REGION_LINE_COLOR,
            fontsize=title_fontsize,
            ha="left",
            va="top",
            zorder=9.0,
        )
        region_map = region_efficiencies.get(region_key, {})
        for index, (category_key, color) in enumerate(category_styles, start=1):
            ax.text(
                x_pos,
                y_pos - line_spacing * index,
                _format_region_efficiency(float(region_map.get(category_key, float("nan")))),
                color=color,
                fontsize=value_fontsize,
                ha="left",
                va="top",
                zorder=9.0,
            )


def _plot_globalpart3_contours_from_histograms(
    contour_histograms: list[tuple[dict[str, Any], np.ndarray]],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    region_efficiencies: dict[str, dict[str, float]],
    outpath: Path,
    plot_style: PlotStyle,
    *,
    plot_def: dict[str, Any],
) -> None:
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    plotted_categories: list[dict[str, Any]] = []
    for zorder, (category, hist2d) in enumerate(_sorted_contour_histograms(contour_histograms), start=1):
        hist_norm = _smoothed_normalized_histogram(hist2d)
        contour_levels = _enclosed_fraction_contour_levels(hist_norm)
        filled_levels = _filled_contour_levels(contour_levels, hist_norm)
        if contour_levels.size == 0 or filled_levels.size <= 1:
            continue
        plotted_categories.append(category)
        cmap = _make_alpha_cmap(
            category["color"],
            f"{category['key']}_alpha_cmap",
            len(filled_levels) - 1,
        )
        ax.contourf(
            x_centers,
            y_centers,
            hist_norm.T,
            levels=filled_levels,
            cmap=cmap,
            antialiased=True,
            extend="max",
            zorder=float(zorder),
        )
        ax.contour(
            x_centers,
            y_centers,
            hist_norm.T,
            levels=contour_levels,
            colors=[category["color"]],
            linewidths=0.7,
            alpha=0.35,
            zorder=float(zorder) + 0.1,
        )

    if not plotted_categories:
        plt.close(fig)
        return

    ax.set_xlabel(SCORE_LABELS[str(plot_def["x_score"])], fontsize=plot_style.label_size)
    ax.set_ylabel(SCORE_LABELS[str(plot_def["y_score"])], fontsize=plot_style.label_size)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="both", labelsize=plot_style.tick_size)
    ax.grid(True, alpha=0.2)
    _draw_globalpart3_region_boundaries(ax, plot_def)
    ax.legend(
        handles=_globalpart3_contour_handles(plotted_categories),
        fontsize=plot_style.legend_size,
        loc="upper left",
        bbox_to_anchor=(0.11, 0.975),
        frameon=False,
        borderpad=0.2,
        labelspacing=0.3,
        handlelength=1.8,
    )
    _annotate_globalpart3_regions(ax, region_efficiencies, plot_style, plot_def)
    private_cms_label(ax, plot_style)
    _save_plot(fig, outpath, plot_style, left=0.14, right=0.97, bottom=0.14, top=0.87, save_pdf=False)


def plot_globalpart3_contours(
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    truth_codes: np.ndarray,
    weights: np.ndarray,
    plot_def: dict[str, Any],
    outpath: Path,
    plot_style: PlotStyle,
) -> None:
    plot_def = _resolve_contour_plot_def(plot_def)
    x_edges = np.linspace(0.0, 1.0, GLOBALPART3_CONTOUR_HIST_BINS + 1, dtype=np.float64)
    y_edges = np.linspace(0.0, 1.0, GLOBALPART3_CONTOUR_HIST_BINS + 1, dtype=np.float64)
    valid = np.isfinite(x_scores) & np.isfinite(y_scores) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return

    contour_histograms: list[tuple[dict[str, Any], np.ndarray]] = []
    x_valid = np.clip(np.asarray(x_scores[valid], dtype=np.float64), GLOBALPART3_CONTOUR_CLIP_EPS, 1.0 - GLOBALPART3_CONTOUR_CLIP_EPS)
    y_valid = np.clip(np.asarray(y_scores[valid], dtype=np.float64), GLOBALPART3_CONTOUR_CLIP_EPS, 1.0 - GLOBALPART3_CONTOUR_CLIP_EPS)
    truth_valid = np.asarray(truth_codes[valid], dtype=np.int32)
    weights_valid = np.asarray(weights[valid], dtype=np.float64)
    for category in GLOBALPART3_CONTOUR_CATEGORIES:
        category_mask = np.isin(truth_valid, np.asarray(category["truth_codes"], dtype=np.int32))
        if not np.any(category_mask):
            continue
        hist2d, _, _ = np.histogram2d(
            x_valid[category_mask],
            y_valid[category_mask],
            bins=(x_edges, y_edges),
            weights=weights_valid[category_mask],
        )
        contour_histograms.append((category, np.asarray(hist2d, dtype=np.float64)))
    region_efficiencies = _compute_contour_region_efficiencies_from_raw(
        x_scores=x_scores,
        y_scores=y_scores,
        truth_codes=truth_codes,
        weights=weights,
        plot_def=plot_def,
    )

    _plot_globalpart3_contours_from_histograms(
        contour_histograms,
        x_edges,
        y_edges,
        region_efficiencies,
        outpath,
        plot_style,
        plot_def=plot_def,
    )


def plot_globalpart3_contours_from_hist(
    contour_payload: dict[str, Any],
    outpath: Path,
    plot_style: PlotStyle,
    plot_def: dict[str, Any] | None = None,
) -> None:
    plot_def = _resolve_contour_plot_def(plot_def or contour_payload, plot_key=str(contour_payload.get("key", "")))
    categories = list(contour_payload.get("categories", []))
    weight_category = np.asarray(contour_payload["weight_category"], dtype=np.float64)
    contour_histograms: list[tuple[dict[str, Any], np.ndarray]] = []
    for index, category in enumerate(categories):
        if index >= weight_category.shape[0]:
            break
        contour_histograms.append((category, np.asarray(weight_category[index], dtype=np.float64)))
    x_edges = np.asarray(contour_payload["x_edges"], dtype=np.float64)
    y_edges = np.asarray(contour_payload["y_edges"], dtype=np.float64)
    region_efficiencies = _compute_contour_region_efficiencies_from_hist(
        categories=categories,
        weight_category=weight_category,
        x_edges=x_edges,
        y_edges=y_edges,
        plot_def=plot_def,
    )

    _plot_globalpart3_contours_from_histograms(
        contour_histograms,
        x_edges,
        y_edges,
        region_efficiencies,
        outpath,
        plot_style,
        plot_def=plot_def,
    )


def compute_globalpart3_region_efficiencies_from_raw(
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    truth_codes: np.ndarray,
    weights: np.ndarray,
    plot_def: dict[str, Any],
) -> dict[str, dict[str, float]]:
    return _compute_contour_region_efficiencies_from_raw(
        x_scores=x_scores,
        y_scores=y_scores,
        truth_codes=truth_codes,
        weights=weights,
        plot_def=_resolve_contour_plot_def(plot_def),
    )


def compute_globalpart3_region_efficiencies_from_hist_payload(
    contour_payload: dict[str, Any],
    plot_def: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    plot_def = _resolve_contour_plot_def(plot_def or contour_payload, plot_key=str(contour_payload.get("key", "")))
    return _compute_contour_region_efficiencies_from_hist(
        categories=list(contour_payload.get("categories", [])),
        weight_category=np.asarray(contour_payload["weight_category"], dtype=np.float64),
        x_edges=np.asarray(contour_payload["x_edges"], dtype=np.float64),
        y_edges=np.asarray(contour_payload["y_edges"], dtype=np.float64),
        plot_def=plot_def,
    )


def _axis_interval_mask(values: np.ndarray, region_def: dict[str, Any], axis_name: str) -> np.ndarray:
    mask = np.ones_like(values, dtype=bool)
    min_exclusive = region_def.get(f"{axis_name}_min_exclusive")
    min_inclusive = region_def.get(f"{axis_name}_min_inclusive")
    max_exclusive = region_def.get(f"{axis_name}_max_exclusive")
    max_inclusive = region_def.get(f"{axis_name}_max_inclusive")
    if min_exclusive is not None:
        mask &= values > float(min_exclusive)
    if min_inclusive is not None:
        mask &= values >= float(min_inclusive)
    if max_exclusive is not None:
        mask &= values < float(max_exclusive)
    if max_inclusive is not None:
        mask &= values <= float(max_inclusive)
    return mask


def _compute_fixed_other_efficiency_scan_from_hist(
    categories: list[dict[str, Any]],
    weight_category: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    plot_def: dict[str, Any],
    other_eff_targets: list[float],
) -> dict[str, Any]:
    x_edges = np.asarray(x_edges, dtype=np.float64)
    y_centers = 0.5 * (np.asarray(y_edges[:-1], dtype=np.float64) + np.asarray(y_edges[1:], dtype=np.float64))
    category_index_map = {str(category["key"]): index for index, category in enumerate(categories)}
    region_definitions = dict(plot_def.get("region_definitions", {}))
    category_keys = ["hbb_pure", "hcc_pure", "others"]

    total_by_category: dict[str, float] = {}
    for category_key in category_keys:
        if category_key not in category_index_map:
            total_by_category[category_key] = 0.0
            continue
        total_by_category[category_key] = float(
            np.sum(np.asarray(weight_category[category_index_map[category_key]], dtype=np.float64))
        )

    region_rows: dict[str, list[dict[str, Any]]] = {}
    for region_key in ["hcc", "hbb"]:
        region_def = dict(region_definitions.get(region_key, {}))
        y_mask = _axis_interval_mask(y_centers, region_def, "y")
        category_eff_curves: dict[str, np.ndarray] = {}
        for category_key in category_keys:
            if category_key not in category_index_map or total_by_category[category_key] <= 0:
                category_eff_curves[category_key] = np.full(len(x_edges) - 1, np.nan, dtype=np.float64)
                continue
            hist = np.asarray(weight_category[category_index_map[category_key]], dtype=np.float64)
            band_weights_x = np.sum(hist[:, y_mask], axis=1)
            pass_weights_x = np.cumsum(band_weights_x[::-1])[::-1]
            category_eff_curves[category_key] = pass_weights_x / total_by_category[category_key]

        others_curve = category_eff_curves["others"]
        rows: list[dict[str, Any]] = []
        for target_eff in other_eff_targets:
            finite_curve = np.isfinite(others_curve)
            if not np.any(finite_curve):
                x_cut = float("nan")
                efficiencies = {category_key: float("nan") for category_key in category_keys}
            else:
                first_index = int(np.argmax(finite_curve))
                if float(target_eff) >= float(others_curve[first_index]):
                    cut_index = first_index
                else:
                    meeting = np.flatnonzero(finite_curve & (others_curve <= float(target_eff)))
                    cut_index = int(meeting[0]) if meeting.size > 0 else int(np.flatnonzero(finite_curve)[-1])
                x_cut = float(x_edges[cut_index])
                efficiencies = {
                    category_key: float(category_eff_curves[category_key][cut_index])
                    if np.isfinite(category_eff_curves[category_key][cut_index])
                    else float("nan")
                    for category_key in category_keys
                }
            rows.append(
                {
                    "target_other_eff": float(target_eff),
                    "x_cut": x_cut,
                    "efficiencies": efficiencies,
                }
            )
        region_rows[region_key] = rows

    return {
        "key": plot_def["key"],
        "x_score": plot_def["x_score"],
        "y_score": plot_def["y_score"],
        "other_eff_targets": [float(value) for value in other_eff_targets],
        "regions": region_rows,
        "scan_note": "x-cut chosen from histogram bin edges as the smallest threshold with Others efficiency <= target",
    }


def compute_fixed_other_efficiency_scan_from_hist_payload(
    contour_payload: dict[str, Any],
    other_eff_targets: list[float] | None = None,
    plot_def: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plot_def = _resolve_contour_plot_def(plot_def or contour_payload, plot_key=str(contour_payload.get("key", "")))
    targets = list(other_eff_targets or GLOBALPART3_FIXED_OTHER_EFF_TARGETS)
    return _compute_fixed_other_efficiency_scan_from_hist(
        categories=list(contour_payload.get("categories", [])),
        weight_category=np.asarray(contour_payload["weight_category"], dtype=np.float64),
        x_edges=np.asarray(contour_payload["x_edges"], dtype=np.float64),
        y_edges=np.asarray(contour_payload["y_edges"], dtype=np.float64),
        plot_def=plot_def,
        other_eff_targets=targets,
    )


def compute_fixed_other_efficiency_scan_from_raw(
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    truth_codes: np.ndarray,
    weights: np.ndarray,
    plot_def: dict[str, Any],
    other_eff_targets: list[float] | None = None,
) -> dict[str, Any]:
    plot_def = _resolve_contour_plot_def(plot_def)
    x_edges = np.linspace(0.0, 1.0, GLOBALPART3_CONTOUR_HIST_BINS + 1, dtype=np.float64)
    y_edges = np.linspace(0.0, 1.0, GLOBALPART3_CONTOUR_HIST_BINS + 1, dtype=np.float64)
    categories = list(GLOBALPART3_CONTOUR_CATEGORIES)
    weight_category = np.zeros((len(categories), GLOBALPART3_CONTOUR_HIST_BINS, GLOBALPART3_CONTOUR_HIST_BINS), dtype=np.float64)
    valid = np.isfinite(x_scores) & np.isfinite(y_scores) & np.isfinite(weights) & (weights > 0)
    if np.any(valid):
        x_valid = np.clip(np.asarray(x_scores[valid], dtype=np.float64), GLOBALPART3_CONTOUR_CLIP_EPS, 1.0 - GLOBALPART3_CONTOUR_CLIP_EPS)
        y_valid = np.clip(np.asarray(y_scores[valid], dtype=np.float64), GLOBALPART3_CONTOUR_CLIP_EPS, 1.0 - GLOBALPART3_CONTOUR_CLIP_EPS)
        truth_valid = np.asarray(truth_codes[valid], dtype=np.int32)
        weights_valid = np.asarray(weights[valid], dtype=np.float64)
        for index, category in enumerate(categories):
            category_mask = np.isin(truth_valid, np.asarray(category["truth_codes"], dtype=np.int32))
            if not np.any(category_mask):
                continue
            hist2d, _, _ = np.histogram2d(
                x_valid[category_mask],
                y_valid[category_mask],
                bins=(x_edges, y_edges),
                weights=weights_valid[category_mask],
            )
            weight_category[index] = np.asarray(hist2d, dtype=np.float64)
    targets = list(other_eff_targets or GLOBALPART3_FIXED_OTHER_EFF_TARGETS)
    return _compute_fixed_other_efficiency_scan_from_hist(
        categories=categories,
        weight_category=weight_category,
        x_edges=x_edges,
        y_edges=y_edges,
        plot_def=plot_def,
        other_eff_targets=targets,
    )


def _compute_fixed_x_ycut_scan_from_hist(
    categories: list[dict[str, Any]],
    weight_category: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    plot_def: dict[str, Any],
    fixed_x_cut: float,
) -> dict[str, Any]:
    x_edges = np.asarray(x_edges, dtype=np.float64)
    y_edges = np.asarray(y_edges, dtype=np.float64)
    y_cuts = y_edges[1:-1]
    region_definitions = dict(plot_def.get("region_definitions", {}))
    hcc_region_def = dict(region_definitions.get("hcc", {}))
    hbb_region_def = dict(region_definitions.get("hbb", {}))
    y_min = float(hcc_region_def.get("y_min_exclusive", 0.0))
    y_max = float(hbb_region_def.get("y_max_inclusive", 1.0))
    valid_y = (y_cuts > y_min) & (y_cuts < y_max)
    y_cuts = y_cuts[valid_y]

    category_index_map = {str(category["key"]): index for index, category in enumerate(categories)}
    required = {"hbb_pure", "hcc_pure"}
    if not required.issubset(category_index_map):
        return {
            "key": plot_def["key"],
            "x_score": plot_def["x_score"],
            "y_score": plot_def["y_score"],
            "fixed_x_cut": float(fixed_x_cut),
            "rows": [],
            "best": {},
            "y_region": {"min": y_min, "max": y_max},
            "scan_note": "Missing required categories for fixed-x y-cut scan.",
        }

    x_bin_start = int(np.argmin(np.abs(x_edges[:-1] - float(fixed_x_cut))))
    x_mask = np.zeros(len(x_edges) - 1, dtype=bool)
    x_mask[x_bin_start:] = True
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    hist_hbb = np.asarray(weight_category[category_index_map["hbb_pure"]], dtype=np.float64)
    hist_hcc = np.asarray(weight_category[category_index_map["hcc_pure"]], dtype=np.float64)
    total_hbb = float(np.sum(hist_hbb))
    total_hcc = float(np.sum(hist_hcc))

    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_merit = -np.inf

    for y_cut in y_cuts:
        hcc_y_mask = (y_centers > y_min) & (y_centers <= float(y_cut))
        hbb_y_mask = (y_centers > float(y_cut)) & (y_centers <= y_max)
        hcc_to_hcc = float(np.sum(hist_hcc[np.ix_(x_mask, hcc_y_mask)]) / total_hcc) if total_hcc > 0 else float("nan")
        hbb_to_hcc = float(np.sum(hist_hbb[np.ix_(x_mask, hcc_y_mask)]) / total_hbb) if total_hbb > 0 else float("nan")
        hbb_to_hbb = float(np.sum(hist_hbb[np.ix_(x_mask, hbb_y_mask)]) / total_hbb) if total_hbb > 0 else float("nan")
        hcc_to_hbb = float(np.sum(hist_hcc[np.ix_(x_mask, hbb_y_mask)]) / total_hcc) if total_hcc > 0 else float("nan")

        denom = (hbb_to_hcc + 1.0e-6) * (hcc_to_hbb + 1.0e-6)
        merit = float(np.sqrt((hcc_to_hcc * hbb_to_hbb) / denom)) if denom > 0 else float("nan")
        row = {
            "y_cut": float(y_cut),
            "efficiencies": {
                "hcc_to_hcc": hcc_to_hcc,
                "hbb_to_hcc": hbb_to_hcc,
                "hbb_to_hbb": hbb_to_hbb,
                "hcc_to_hbb": hcc_to_hbb,
            },
            "merit": merit,
        }
        rows.append(row)
        if np.isfinite(merit) and merit > best_merit:
            best_merit = merit
            best_row = row

    return {
        "key": plot_def["key"],
        "x_score": plot_def["x_score"],
        "y_score": plot_def["y_score"],
        "fixed_x_cut": float(fixed_x_cut),
        "x_cut_bin_edge": float(x_edges[x_bin_start]),
        "rows": rows,
        "best": best_row or {},
        "y_region": {"min": y_min, "max": y_max},
        "scan_note": "y-cut scanned over histogram bin edges with fixed x threshold",
    }


def compute_fixed_x_ycut_scan_from_hist_payload(
    contour_payload: dict[str, Any],
    fixed_x_cut: float | None = None,
    plot_def: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plot_def = _resolve_contour_plot_def(plot_def or contour_payload, plot_key=str(contour_payload.get("key", "")))
    return _compute_fixed_x_ycut_scan_from_hist(
        categories=list(contour_payload.get("categories", [])),
        weight_category=np.asarray(contour_payload["weight_category"], dtype=np.float64),
        x_edges=np.asarray(contour_payload["x_edges"], dtype=np.float64),
        y_edges=np.asarray(contour_payload["y_edges"], dtype=np.float64),
        plot_def=plot_def,
        fixed_x_cut=float(
            fixed_x_cut if fixed_x_cut is not None else plot_def.get("fixed_x_cut", GLOBALPART3_FIXED_X_CUT)
        ),
    )


def compute_fixed_x_ycut_scan_from_raw(
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    truth_codes: np.ndarray,
    weights: np.ndarray,
    plot_def: dict[str, Any],
    fixed_x_cut: float | None = None,
) -> dict[str, Any]:
    plot_def = _resolve_contour_plot_def(plot_def)
    x_edges = np.linspace(0.0, 1.0, GLOBALPART3_CONTOUR_HIST_BINS + 1, dtype=np.float64)
    y_edges = np.linspace(0.0, 1.0, GLOBALPART3_CONTOUR_HIST_BINS + 1, dtype=np.float64)
    categories = list(GLOBALPART3_CONTOUR_CATEGORIES)
    weight_category = np.zeros((len(categories), GLOBALPART3_CONTOUR_HIST_BINS, GLOBALPART3_CONTOUR_HIST_BINS), dtype=np.float64)
    valid = np.isfinite(x_scores) & np.isfinite(y_scores) & np.isfinite(weights) & (weights > 0)
    if np.any(valid):
        x_valid = np.clip(np.asarray(x_scores[valid], dtype=np.float64), GLOBALPART3_CONTOUR_CLIP_EPS, 1.0 - GLOBALPART3_CONTOUR_CLIP_EPS)
        y_valid = np.clip(np.asarray(y_scores[valid], dtype=np.float64), GLOBALPART3_CONTOUR_CLIP_EPS, 1.0 - GLOBALPART3_CONTOUR_CLIP_EPS)
        truth_valid = np.asarray(truth_codes[valid], dtype=np.int32)
        weights_valid = np.asarray(weights[valid], dtype=np.float64)
        for index, category in enumerate(categories):
            category_mask = np.isin(truth_valid, np.asarray(category["truth_codes"], dtype=np.int32))
            if not np.any(category_mask):
                continue
            hist2d, _, _ = np.histogram2d(
                x_valid[category_mask],
                y_valid[category_mask],
                bins=(x_edges, y_edges),
                weights=weights_valid[category_mask],
            )
            weight_category[index] = np.asarray(hist2d, dtype=np.float64)
    return _compute_fixed_x_ycut_scan_from_hist(
        categories=categories,
        weight_category=weight_category,
        x_edges=x_edges,
        y_edges=y_edges,
        plot_def=plot_def,
        fixed_x_cut=float(
            fixed_x_cut if fixed_x_cut is not None else plot_def.get("fixed_x_cut", GLOBALPART3_FIXED_X_CUT)
        ),
    )


def plot_fixed_x_ycut_scan(
    scan_payload: dict[str, Any],
    plot_def: dict[str, Any],
    outpath: Path,
    plot_style: PlotStyle,
) -> None:
    rows = list(scan_payload.get("rows", []))
    if not rows:
        return
    y_cut = np.asarray([row["y_cut"] for row in rows], dtype=np.float64)
    hcc_to_hcc = np.asarray([row["efficiencies"]["hcc_to_hcc"] for row in rows], dtype=np.float64)
    hbb_to_hcc = np.asarray([row["efficiencies"]["hbb_to_hcc"] for row in rows], dtype=np.float64)
    hbb_to_hbb = np.asarray([row["efficiencies"]["hbb_to_hbb"] for row in rows], dtype=np.float64)
    hcc_to_hbb = np.asarray([row["efficiencies"]["hcc_to_hbb"] for row in rows], dtype=np.float64)
    merit = np.asarray([row["merit"] for row in rows], dtype=np.float64)
    valid = np.isfinite(y_cut)
    if not np.any(valid):
        return

    plt.style.use(hep.style.CMS)
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(9.0, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
    )

    ax_top.plot(y_cut, hcc_to_hcc, color="#d62728", linewidth=2.0, label=r"$hcc\to Hcc$")
    ax_top.plot(y_cut, hbb_to_hbb, color="#ff7f0e", linewidth=2.0, label=r"$hbb\to Hbb$")
    ax_top.plot(y_cut, hbb_to_hcc, color="#ff7f0e", linewidth=1.7, linestyle="--", label=r"$hbb\to Hcc$")
    ax_top.plot(y_cut, hcc_to_hbb, color="#d62728", linewidth=1.7, linestyle="--", label=r"$hcc\to Hbb$")
    ax_top.set_ylabel("Weighted efficiency", fontsize=plot_style.label_size)
    finite_eff = np.concatenate(
        [
            hcc_to_hcc[np.isfinite(hcc_to_hcc)],
            hbb_to_hbb[np.isfinite(hbb_to_hbb)],
            hbb_to_hcc[np.isfinite(hbb_to_hcc)],
            hcc_to_hbb[np.isfinite(hcc_to_hbb)],
        ]
    )
    eff_ymax = float(np.max(finite_eff)) * 1.1 if finite_eff.size > 0 else 1.0
    ax_top.set_ylim(0.0, min(1.0, eff_ymax))
    ax_top.tick_params(axis="both", labelsize=plot_style.tick_size)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(
        fontsize=plot_style.legend_size,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=False,
    )
    plot_title = (
        f"{SCORE_LABELS[str(plot_def['y_score'])]} y-cut scan at "
        f"{SCORE_LABELS[str(plot_def['x_score'])]} > {float(scan_payload.get('fixed_x_cut', GLOBALPART3_FIXED_X_CUT)):.4f}"
    )
    ax_top.set_title(plot_title, fontsize=plot_style.title_size)

    ax_bottom.plot(y_cut, merit, color="#1f77b4", linewidth=2.0, label="Merit")
    best = dict(scan_payload.get("best", {}))
    best_y = float(best.get("y_cut", float("nan")))
    best_merit = float(best.get("merit", float("nan")))
    if np.isfinite(best_y) and np.isfinite(best_merit):
        ax_top.axvline(best_y, color="#2f2f2f", linewidth=1.0, linestyle=":")
        ax_bottom.axvline(best_y, color="#2f2f2f", linewidth=1.0, linestyle=":")
        ax_bottom.scatter([best_y], [best_merit], color="#1f77b4", s=35, zorder=4)
        ax_bottom.annotate(
            f"best y={best_y:.4f}\nmerit={best_merit:.3g}",
            xy=(best_y, best_merit),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=max(plot_style.tick_size - 0.5, 7.5),
            color="#1f77b4",
        )
    ax_bottom.set_xlabel(SCORE_LABELS[str(plot_def["y_score"])], fontsize=plot_style.label_size)
    ax_bottom.set_ylabel("Merit", fontsize=plot_style.label_size)
    ax_bottom.tick_params(axis="both", labelsize=plot_style.tick_size)
    ax_bottom.grid(True, alpha=0.3)
    private_cms_label(ax_top, plot_style)
    _save_plot(fig, outpath, plot_style, left=0.13, right=0.97, bottom=0.11, top=0.89, save_pdf=False)


def plot_significance_scan(
    rows: list[dict[str, Any]],
    target: str,
    score_name: str,
    outpath: Path,
    plot_style: PlotStyle,
) -> None:
    x = np.asarray([row["target_sig_eff"] for row in rows], dtype=float)
    y = np.asarray([row["s_over_sqrt_s_plus_b"] for row in rows], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return

    x = x[valid]
    y = y[valid]
    best_index = int(np.argmax(y))
    best_x = float(x[best_index])
    best_y = float(y[best_index])

    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax.plot(
        x,
        y,
        color="#d62728",
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        label=SCORE_LABELS[score_name],
    )
    ax.scatter([best_x], [best_y], color="#1f77b4", s=42, zorder=4, label=f"Best: {best_x*100:.1f}%")
    ax.set_xlabel("Target signal efficiency", fontsize=plot_style.label_size)
    ax.set_ylabel(r"$S/\sqrt{S+B}$", fontsize=plot_style.label_size)
    ax.set_title(
        f"{TARGET_DEFINITIONS[target]['title']} significance scan",
        fontsize=plot_style.title_size,
    )
    ax.set_xlim(max(0.0, float(np.min(x)) - 0.02), min(1.0, float(np.max(x)) + 0.02))
    if np.any(y > 0):
        ax.set_ylim(0.0, float(np.max(y)) * 1.18)
    ax.tick_params(axis="both", labelsize=plot_style.tick_size)
    ax.grid(True, alpha=0.3)
    ax.annotate(
        f"max={best_y:.3f}\nTargetEff={best_x*100:.1f}%",
        xy=(best_x, best_y),
        xytext=(10, 8),
        textcoords="offset points",
        fontsize=max(plot_style.tick_size - 0.5, 7.5),
        color="#1f77b4",
    )
    _inside_legend(ax, plot_style, loc="upper right", ncol=1)
    _cms_label(ax, plot_style)
    _save_plot(fig, outpath, plot_style, left=0.125, right=0.97, bottom=0.14, top=0.875)


def plot_score_distribution(
    scores: np.ndarray,
    truth_codes: np.ndarray,
    weights: np.ndarray,
    target: str,
    score_name: str,
    outpath: Path,
    plot_style: PlotStyle,
) -> None:
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ordered_labels = TARGET_DEFINITIONS[target]["signal_labels"] + TARGET_DEFINITIONS[target]["background_labels"]
    positive_contents: list[np.ndarray] = []
    for label in ordered_labels:
        mask = truth_mask(truth_codes, label) & np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
        values = scores[mask]
        label_weights = weights[mask]
        if values.size == 0 or np.sum(label_weights) <= 0:
            continue
        density_hist, _, _ = ax.hist(
            values,
            bins=DISPLAY_SCORE_BINS,
            range=(0.0, 1.0),
            density=True,
            weights=label_weights,
            histtype="step",
            linewidth=1.8,
            label=f"{TRUTH_LABEL_TITLES[label]} (Y={np.sum(label_weights):.2f})",
            color=TRUTH_LABEL_COLORS[label],
        )
        positive_contents.append(np.asarray(density_hist)[np.asarray(density_hist) > 0])
    ax.set_xlabel(SCORE_LABELS[score_name], fontsize=plot_style.label_size)
    ax.set_ylabel("Arbitrary units", fontsize=plot_style.label_size)
    ax.set_title(
        f"{TARGET_DEFINITIONS[target]['title']} score distribution",
        fontsize=plot_style.title_size,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_yscale("log")
    if positive_contents:
        positive = np.concatenate(positive_contents)
        if positive.size > 0:
            ax.set_ylim(max(float(np.min(positive)) * 0.7, 1e-6), float(np.max(positive)) * 1.6)
    ax.tick_params(axis="both", labelsize=plot_style.tick_size)
    ax.grid(True, which="both", alpha=0.3)
    _inside_legend(ax, plot_style, loc="upper right", ncol=2)
    _cms_label(ax, plot_style)
    _save_plot(fig, outpath, plot_style, left=0.125, right=0.97, bottom=0.14, top=0.875)


def plot_background_process_score_distribution(
    scores: np.ndarray,
    truth_codes: np.ndarray,
    process_codes: np.ndarray,
    weights: np.ndarray,
    target: str,
    score_name: str,
    process_entries: list[dict[str, Any]],
    outpath: Path,
    plot_style: PlotStyle,
) -> None:
    _, background_mask, _ = build_target_masks(truth_codes, target)
    valid = background_mask & np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return

    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    plotted = 0
    positive_contents: list[np.ndarray] = []
    for entry in process_entries:
        mask = valid & (process_codes == int(entry["code"]))
        values = scores[mask]
        label_weights = weights[mask]
        if values.size == 0 or np.sum(label_weights) <= 0:
            continue
        plotted += 1
        density_hist, _, _ = ax.hist(
            values,
            bins=DISPLAY_PROCESS_SCORE_BINS,
            range=(0.0, 1.0),
            density=True,
            weights=label_weights,
            histtype="step",
            linewidth=1.8,
            label=f"{entry['label']} (Y={np.sum(label_weights):.2f})",
            color=entry["color"],
        )
        positive_contents.append(np.asarray(density_hist)[np.asarray(density_hist) > 0])

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_xlabel(SCORE_LABELS[score_name], fontsize=plot_style.label_size)
    ax.set_ylabel("Arbitrary units", fontsize=plot_style.label_size)
    ax.set_title(
        f"{TARGET_DEFINITIONS[target]['title']} background by process",
        fontsize=plot_style.title_size,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_yscale("log")
    if positive_contents:
        positive = np.concatenate(positive_contents)
        if positive.size > 0:
            ax.set_ylim(max(float(np.min(positive)) * 0.7, 1e-6), float(np.max(positive)) * 1.6)
    ax.tick_params(axis="both", labelsize=plot_style.tick_size)
    ax.grid(True, which="both", alpha=0.3)
    _inside_legend(ax, plot_style, loc="upper right", ncol=2)
    _cms_label(ax, plot_style)
    _save_plot(fig, outpath, plot_style, left=0.125, right=0.97, bottom=0.14, top=0.875)


def plot_background_process_working_points(
    scores: np.ndarray,
    truth_codes: np.ndarray,
    process_codes: np.ndarray,
    weights: np.ndarray,
    target: str,
    score_name: str,
    sig_effs: list[float],
    process_entries: list[dict[str, Any]],
    outpath: Path,
    plot_style: PlotStyle,
) -> None:
    signal_mask, background_mask, _ = build_target_masks(truth_codes, target)
    valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    signal_valid = signal_mask & valid
    background_valid = background_mask & valid
    if not np.any(signal_valid) or not np.any(background_valid):
        return

    signal_scores = scores[signal_valid]
    signal_weights = weights[signal_valid]
    if np.sum(signal_weights) <= 0:
        return

    x_positions = np.arange(len(sig_effs), dtype=float)
    cuts = [weighted_quantile(signal_scores, signal_weights, 1.0 - sig_eff) for sig_eff in sig_effs]
    process_yields: list[tuple[dict[str, Any], np.ndarray]] = []
    for entry in process_entries:
        process_mask = background_valid & (process_codes == int(entry["code"]))
        yields = []
        for cut in cuts:
            yields.append(weighted_sum(weights[process_mask & (scores >= cut)]))
        yields_array = np.asarray(yields, dtype=float)
        if np.sum(yields_array) <= 0:
            continue
        process_yields.append((entry, yields_array))

    if not process_yields:
        return

    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    bottom = np.zeros(len(sig_effs), dtype=float)
    for entry, yields_array in process_yields:
        ax.bar(
            x_positions,
            yields_array,
            bottom=bottom,
            width=0.75,
            color=entry["color"],
            edgecolor="black",
            linewidth=0.4,
            label=entry["label"],
        )
        bottom += yields_array

    ax.set_xticks(x_positions, [f"{sig_eff * 100:.0f}%" for sig_eff in sig_effs])
    ax.set_xlabel("Target signal efficiency working point", fontsize=plot_style.label_size)
    ax.set_ylabel("Weighted background yield", fontsize=plot_style.label_size)
    if np.any(bottom > 0):
        ax.set_ylim(0.0, float(np.max(bottom)) * 1.25)
    ax.set_title(
        f"{TARGET_DEFINITIONS[target]['title']}: background vs WP",
        fontsize=plot_style.title_size,
        pad=10,
    )
    _scientific_yaxis(ax)
    ax.tick_params(axis="both", labelsize=plot_style.tick_size)
    ax.grid(True, axis="y", alpha=0.3)
    _inside_legend(ax, plot_style, loc="upper right", ncol=2)
    _cms_label(ax, plot_style)
    _save_plot(fig, outpath, plot_style, left=0.17, right=0.97, bottom=0.14, top=0.85)


def plot_roc_curve(
    roc_payload: dict[str, np.ndarray | float],
    target: str,
    score_name: str,
    outpath: Path,
    plot_style: PlotStyle,
) -> None:
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(8, 6))
    sig_eff = np.asarray(roc_payload["sig_eff"])
    bkg_eff = np.asarray(roc_payload["bkg_eff"])
    auc = float(roc_payload["auc"])
    ax.plot(
        sig_eff,
        np.clip(bkg_eff, 1e-6, None),
        linewidth=2.0,
        color="#d62728",
        label=f"{SCORE_LABELS[score_name]} (AUC={auc:.4f})",
    )
    ax.set_xlabel(
        f"{TARGET_DEFINITIONS[target]['title']} efficiency",
        fontsize=plot_style.label_size,
    )
    ax.set_ylabel("Background efficiency", fontsize=plot_style.label_size)
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1e-4, 1.0)
    ax.set_title(f"{TARGET_DEFINITIONS[target]['title']} ROC", fontsize=plot_style.title_size)
    ax.tick_params(axis="both", labelsize=plot_style.tick_size)
    ax.grid(True, which="both", alpha=0.3)
    _inside_legend(ax, plot_style, loc="upper right", ncol=1)
    _cms_label(ax, plot_style)
    _save_plot(fig, outpath, plot_style, left=0.125, right=0.97, bottom=0.14, top=0.875)


def plot_score_distribution_from_hist(
    score_payload: dict[str, np.ndarray],
    hist_edges: np.ndarray,
    target: str,
    score_name: str,
    outpath: Path,
    plot_style: PlotStyle,
) -> None:
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ordered_labels = TARGET_DEFINITIONS[target]["signal_labels"] + TARGET_DEFINITIONS[target]["background_labels"]
    weight_truth = np.asarray(score_payload["weight_truth"], dtype=np.float64)
    positive_contents: list[np.ndarray] = []
    for label in ordered_labels:
        hist = weight_truth[TRUTH_LABEL_TO_CODE[label]]
        total = float(np.sum(hist))
        if total <= 0:
            continue
        rebinned_hist, display_edges = _rebin_histogram_for_display(hist, hist_edges, DISPLAY_SCORE_BINS)
        density_hist = _histogram_density(rebinned_hist, display_edges)
        if np.sum(density_hist) <= 0:
            continue
        ax.stairs(
            density_hist,
            display_edges,
            linewidth=1.8,
            label=f"{TRUTH_LABEL_TITLES[label]} (Y={total:.2f})",
            color=TRUTH_LABEL_COLORS[label],
        )
        positive_contents.append(density_hist[density_hist > 0])
    ax.set_xlabel(SCORE_LABELS[score_name], fontsize=plot_style.label_size)
    ax.set_ylabel("A.U.", fontsize=plot_style.label_size)
    ax.set_title(f"{TARGET_DEFINITIONS[target]['title']} score distribution", fontsize=plot_style.title_size)
    ax.set_xlim(0.0, 1.0)
    ax.set_yscale("log")
    if positive_contents:
        positive = np.concatenate(positive_contents)
        if positive.size > 0:
            ax.set_ylim(max(float(np.min(positive)) * 0.7, 1e-6), float(np.max(positive)) * 1.6)
    ax.tick_params(axis="both", labelsize=plot_style.tick_size)
    ax.grid(True, which="both", alpha=0.3)
    _inside_legend(ax, plot_style, loc="upper right", ncol=2)
    _cms_label(ax, plot_style)
    _save_plot(fig, outpath, plot_style, left=0.125, right=0.97, bottom=0.14, top=0.875)


def plot_background_process_score_distribution_from_hist(
    score_payload: dict[str, np.ndarray],
    hist_edges: np.ndarray,
    target: str,
    score_name: str,
    process_entries: list[dict[str, Any]],
    outpath: Path,
    plot_style: PlotStyle,
) -> None:
    background_codes = codes_for_labels(TARGET_DEFINITIONS[target]["background_labels"])
    process_hist = np.asarray(score_payload["weight_process_truth"], dtype=np.float64)
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    plotted = 0
    positive_contents: list[np.ndarray] = []
    for entry in process_entries:
        hist = np.sum(process_hist[int(entry["code"]), background_codes], axis=0)
        total = float(np.sum(hist))
        if total <= 0:
            continue
        rebinned_hist, display_edges = _rebin_histogram_for_display(
            hist,
            hist_edges,
            DISPLAY_PROCESS_SCORE_BINS,
        )
        density_hist = _histogram_density(rebinned_hist, display_edges)
        if np.sum(density_hist) <= 0:
            continue
        plotted += 1
        ax.stairs(
            density_hist,
            display_edges,
            linewidth=1.8,
            label=f"{entry['label']} (Y={total:.2f})",
            color=entry["color"],
        )
        positive_contents.append(density_hist[density_hist > 0])
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel(SCORE_LABELS[score_name], fontsize=plot_style.label_size)
    ax.set_ylabel("A.U.", fontsize=plot_style.label_size)
    ax.set_title(f"{TARGET_DEFINITIONS[target]['title']} background by process", fontsize=plot_style.title_size)
    ax.set_xlim(0.0, 1.0)
    ax.set_yscale("log")
    if positive_contents:
        positive = np.concatenate(positive_contents)
        if positive.size > 0:
            ax.set_ylim(max(float(np.min(positive)) * 0.7, 1e-6), float(np.max(positive)) * 1.6)
    ax.tick_params(axis="both", labelsize=plot_style.tick_size)
    ax.grid(True, which="both", alpha=0.3)
    _inside_legend(ax, plot_style, loc="upper right", ncol=2)
    _cms_label(ax, plot_style)
    _save_plot(fig, outpath, plot_style, left=0.125, right=0.97, bottom=0.14, top=0.875)


def plot_background_process_working_points_from_hist(
    score_payload: dict[str, np.ndarray],
    hist_edges: np.ndarray,
    target: str,
    score_name: str,
    sig_effs: list[float],
    process_entries: list[dict[str, Any]],
    outpath: Path,
    plot_style: PlotStyle,
) -> None:
    weight_truth = np.asarray(score_payload["weight_truth"], dtype=np.float64)
    process_hist = np.asarray(score_payload["weight_process_truth"], dtype=np.float64)
    signal_codes = codes_for_labels(TARGET_DEFINITIONS[target]["signal_labels"])
    background_codes = codes_for_labels(TARGET_DEFINITIONS[target]["background_labels"])
    signal_pass = pass_from_hist(np.sum(weight_truth[signal_codes], axis=0))
    total_signal = float(np.sum(weight_truth[signal_codes]))
    if total_signal <= 0:
        return

    x_positions = np.arange(len(sig_effs), dtype=float)
    process_background_hist = np.sum(process_hist[:, background_codes], axis=1)
    process_background_pass = pass_from_hist(process_background_hist)
    process_yields: list[tuple[dict[str, Any], np.ndarray]] = []
    for entry in process_entries:
        process_index = int(entry["code"])
        yields = []
        for eff in sig_effs:
            bin_index = working_point_bin_index(signal_pass, total_signal, eff)
            yields.append(float(process_background_pass[process_index, bin_index]))
        yields_array = np.asarray(yields, dtype=float)
        if np.sum(yields_array) <= 0:
            continue
        process_yields.append((entry, yields_array))
    if not process_yields:
        return

    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    bottom = np.zeros(len(sig_effs), dtype=float)
    for entry, yields_array in process_yields:
        ax.bar(
            x_positions,
            yields_array,
            bottom=bottom,
            width=0.75,
            color=entry["color"],
            edgecolor="black",
            linewidth=0.4,
            label=entry["label"],
        )
        bottom += yields_array
    ax.set_xticks(x_positions, [f"{sig_eff * 100:.0f}%" for sig_eff in sig_effs])
    ax.set_xlabel("Target signal efficiency working point", fontsize=plot_style.label_size)
    ax.set_ylabel("Weighted background yield", fontsize=plot_style.label_size)
    ax.set_title(
        f"{TARGET_DEFINITIONS[target]['title']}: background vs WP",
        fontsize=plot_style.title_size,
        pad=10,
    )
    _scientific_yaxis(ax)
    ax.tick_params(axis="both", labelsize=plot_style.tick_size)
    ax.grid(True, axis="y", alpha=0.3)
    _inside_legend(ax, plot_style, loc="upper right", ncol=2)
    _cms_label(ax, plot_style)
    _save_plot(fig, outpath, plot_style, left=0.17, right=0.97, bottom=0.14, top=0.85)
