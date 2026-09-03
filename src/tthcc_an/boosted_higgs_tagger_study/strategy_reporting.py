from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplhep as hep
import numpy as np

from tthcc_an.boosted_higgs_tagger_study.definitions import (
    SCORE_LABELS,
    TRUTH_LABEL_ORDER,
    TRUTH_LABEL_TO_CODE,
)
from tthcc_an.boosted_higgs_tagger_study.plotting import (
    CONTOUR_REGION_LINE_COLOR,
    CONTOUR_REGION_LINEWIDTH,
    PlotStyle,
    _annotate_globalpart3_regions,
    _enclosed_fraction_contour_levels,
    _filled_contour_levels,
    _globalpart3_contour_handles,
    _make_alpha_cmap,
    _save_plot,
    _smoothed_normalized_histogram,
    _sorted_contour_histograms,
    private_cms_label,
)
from tthcc_an.boosted_higgs_tagger_study.reporting import write_csv, write_json
from tthcc_an.boosted_higgs_tagger_study.y_split_reporting import derive_y_split_rows


STRATEGY_CONTOUR_KEY = "gpart_higgs_vs_qcd__vs__gpart_xbb_vs_xcc"
NOMINAL_TARGET = 0.02
TIGHTER_TARGET = 0.01
STRATEGY_Y_SPLIT = 0.80
HISTORICAL_X_CUT = 0.9467
HISTORICAL_Y_SPLIT = 0.85
NON_HIGGS_TRUTH_CODES = [
    code
    for label, code in TRUTH_LABEL_TO_CODE.items()
    if label not in {"hcc_pure", "hbb_pure"}
]


def _ratio(value: float, denominator: float) -> float:
    return float(value / denominator) if denominator > 0.0 else 0.0


def _process_codes(entries: list[dict[str, Any]]) -> dict[str, int]:
    return {entry["process"]: int(entry["code"]) for entry in entries}


def _dedicated_row(
    aggregate: dict[str, Any],
    *,
    target: float,
    label: str,
    role: str,
) -> dict[str, Any]:
    wp_index = next(
        index
        for index, wp in enumerate(aggregate["working_points"])
        if np.isclose(wp["non_higgs_target_efficiency"], target)
    )
    wp = aggregate["working_points"][wp_index]
    y_index = int(
        np.flatnonzero(np.isclose(aggregate["y_splits"], STRATEGY_Y_SPLIT))[0]
    )
    truth_regions = np.asarray(aggregate["jet_truth_region_yields"])[wp_index, :, y_index]
    truth_totals = np.asarray(aggregate["jet_truth_candidate_totals"])
    process_regions = np.asarray(aggregate["jet_process_region_yields"])[wp_index, :, y_index]
    process_totals = np.asarray(aggregate["jet_process_candidate_totals"])
    process_codes = _process_codes(aggregate["process_entries"])
    qcd_code = process_codes["qcd"]

    def truth_efficiency(codes: list[int], region: int) -> float:
        numerator = float(np.sum(truth_regions[codes, region]))
        denominator = float(np.sum(truth_totals[codes]))
        return _ratio(numerator, denominator)

    hcc_code = TRUTH_LABEL_TO_CODE["hcc_pure"]
    hbb_code = TRUTH_LABEL_TO_CODE["hbb_pure"]
    row = {
        "wp_label": label,
        "role": role,
        "inclusive_non_higgs_target": float(target),
        "x_cut": float(wp["x_cut"]),
        "x_cut_evaluated": float(wp["x_cut_evaluated"]),
        "x_convention": ">=",
        "y_split": STRATEGY_Y_SPLIT,
        "hcc_pure_to_Hcc": truth_efficiency([hcc_code], 0),
        "hcc_pure_to_Hbb": truth_efficiency([hcc_code], 1),
        "hbb_pure_to_Hcc": truth_efficiency([hbb_code], 0),
        "hbb_pure_to_Hbb": truth_efficiency([hbb_code], 1),
        "non_higgs_to_Hcc": truth_efficiency(NON_HIGGS_TRUTH_CODES, 0),
        "non_higgs_to_Hbb": truth_efficiency(NON_HIGGS_TRUTH_CODES, 1),
        "qcd_process_to_Hcc": _ratio(
            float(process_regions[qcd_code, 0]), float(process_totals[qcd_code])
        ),
        "qcd_process_to_Hbb": _ratio(
            float(process_regions[qcd_code, 1]), float(process_totals[qcd_code])
        ),
        "event_level_available": True,
    }
    for category in ("hcc_pure", "hbb_pure", "non_higgs", "qcd_process"):
        row[f"{category}_rejected"] = max(
            1.0 - row[f"{category}_to_Hcc"] - row[f"{category}_to_Hbb"], 0.0
        )
    row["qcd_process_inclusive_Higgs_efficiency"] = (
        row["qcd_process_to_Hcc"] + row["qcd_process_to_Hbb"]
    )
    row["achieved_non_higgs_jet_efficiency"] = (
        row["non_higgs_to_Hcc"] + row["non_higgs_to_Hbb"]
    )
    expected_non_higgs = wp.get("achieved_non_higgs_jet_efficiency")
    if expected_non_higgs is not None and not np.isclose(
        row["achieved_non_higgs_jet_efficiency"],
        float(expected_non_higgs),
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError(
            "y-split non-Higgs Hcc+Hbb efficiency does not match the "
            "Inclusive-Higgs WP aggregate."
        )

    event_rows = derive_y_split_rows(aggregate)
    event = next(
        item
        for item in event_rows
        if np.isclose(item["non_higgs_target_wp"], target)
        and np.isclose(item["y_split"], STRATEGY_Y_SPLIT)
    )
    for process in ("ttHcc", "ttHbb", "qcd"):
        row[f"{process}_Hcc_pass_weighted_yield"] = event[
            f"{process}_Hcc_pass_weighted_yield"
        ]
        row[f"{process}_Hcc_pass_weighted_efficiency"] = event[
            f"{process}_Hcc_pass_absolute_efficiency"
        ]
        row[f"{process}_Hcc_pass_raw_event_count"] = event[
            f"{process}_Hcc_pass_raw_event_count"
        ]
    return row


def _historical_row(hcc: dict[str, Any]) -> dict[str, Any]:
    x_cuts = np.asarray(hcc["x_cuts"], dtype=np.float64)
    y_cuts = np.asarray(hcc["y_cuts"], dtype=np.float64)
    x_index = int(np.clip(np.searchsorted(x_cuts, HISTORICAL_X_CUT, side="left"), 0, len(x_cuts) - 1))
    y_index = int(np.clip(np.searchsorted(y_cuts, HISTORICAL_Y_SPLIT, side="right") - 1, 0, len(y_cuts) - 1))
    truth_hcc = np.asarray(hcc["truth_yields"])[:, x_index, y_index]
    truth_inclusive = np.asarray(hcc["truth_yields"])[:, x_index, -1]
    truth_hbb = truth_inclusive - truth_hcc
    truth_totals = np.asarray(hcc["truth_totals"])
    process_codes = _process_codes(hcc["process_entries"])
    qcd_code = process_codes["qcd"]
    process_hcc = np.asarray(hcc["process_yields"])[qcd_code, x_index, y_index]
    process_inclusive = np.asarray(hcc["process_yields"])[qcd_code, x_index, -1]
    qcd_total = float(np.asarray(hcc["process_totals"])[qcd_code])

    def truth_efficiency(values: np.ndarray, codes: list[int]) -> float:
        return _ratio(float(np.sum(values[codes])), float(np.sum(truth_totals[codes])))

    hcc_code = TRUTH_LABEL_TO_CODE["hcc_pure"]
    hbb_code = TRUTH_LABEL_TO_CODE["hbb_pure"]
    row = {
        "wp_label": "Historical tight region reference",
        "role": "historical_region_reference",
        "inclusive_non_higgs_target": None,
        "x_cut": HISTORICAL_X_CUT,
        "x_cut_evaluated": float(x_cuts[x_index]),
        "x_convention": ">",
        "y_split": HISTORICAL_Y_SPLIT,
        "hcc_pure_to_Hcc": truth_efficiency(truth_hcc, [hcc_code]),
        "hcc_pure_to_Hbb": truth_efficiency(truth_hbb, [hcc_code]),
        "hbb_pure_to_Hcc": truth_efficiency(truth_hcc, [hbb_code]),
        "hbb_pure_to_Hbb": truth_efficiency(truth_hbb, [hbb_code]),
        "non_higgs_to_Hcc": truth_efficiency(truth_hcc, NON_HIGGS_TRUTH_CODES),
        "non_higgs_to_Hbb": truth_efficiency(truth_hbb, NON_HIGGS_TRUTH_CODES),
        "qcd_process_to_Hcc": _ratio(float(process_hcc), qcd_total),
        "qcd_process_to_Hbb": _ratio(float(process_inclusive - process_hcc), qcd_total),
        "event_level_available": False,
    }
    for category in ("hcc_pure", "hbb_pure", "non_higgs", "qcd_process"):
        row[f"{category}_rejected"] = max(
            1.0 - row[f"{category}_to_Hcc"] - row[f"{category}_to_Hbb"], 0.0
        )
    row["qcd_process_inclusive_Higgs_efficiency"] = row["qcd_process_to_Hcc"] + row["qcd_process_to_Hbb"]
    for process in ("ttHcc", "ttHbb", "qcd"):
        row[f"{process}_Hcc_pass_weighted_yield"] = None
        row[f"{process}_Hcc_pass_weighted_efficiency"] = None
        row[f"{process}_Hcc_pass_raw_event_count"] = None
    return row


def derive_strategy_comparison(
    y_split: dict[str, Any],
    hcc: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _dedicated_row(
            y_split,
            target=NOMINAL_TARGET,
            label="Nominal loose candidate",
            role="nominal_candidate_for_downstream_validation",
        ),
        _dedicated_row(
            y_split,
            target=TIGHTER_TARGET,
            label="Tighter alternative",
            role="reference_alternative",
        ),
        _historical_row(hcc),
    ]


def _summary_text(payload: dict[str, Any]) -> str:
    lines = [
        "=== Boosted Higgs Tagger WP Strategy Comparison ===",
        "Current candidate for downstream validation; not a significance-ranked final WP.",
        "Region annotations: jet-level weighted efficiencies; no event deduplication.",
    ]
    for row in payload["working_points"]:
        target = (
            "N/A"
            if row["inclusive_non_higgs_target"] is None
            else f"{100 * row['inclusive_non_higgs_target']:g}%"
        )
        lines.extend(
            [
                "",
                row["wp_label"],
                f"  role = {row['role']}",
                f"  inclusive non-Higgs target = {target}",
                f"  x cut requested/evaluated = {row['x_cut']:.6f} / {row['x_cut_evaluated']:.6f}",
                f"  x convention = {row['x_convention']}",
                f"  y split = {row['y_split']:.2f}",
                f"  hcc_pure -> Hcc/Hbb/rejected = {row['hcc_pure_to_Hcc']:.6g} / {row['hcc_pure_to_Hbb']:.6g} / {row['hcc_pure_rejected']:.6g}",
                f"  hbb_pure -> Hcc/Hbb/rejected = {row['hbb_pure_to_Hcc']:.6g} / {row['hbb_pure_to_Hbb']:.6g} / {row['hbb_pure_rejected']:.6g}",
                f"  non-Higgs -> Hcc/Hbb/rejected = {row['non_higgs_to_Hcc']:.6g} / {row['non_higgs_to_Hbb']:.6g} / {row['non_higgs_rejected']:.6g}",
                f"  process-QCD (diagnostic) -> Hcc/Hbb/inclusive = {row['qcd_process_to_Hcc']:.6g} / {row['qcd_process_to_Hbb']:.6g} / {row['qcd_process_inclusive_Higgs_efficiency']:.6g}",
            ]
        )
        if row["event_level_available"]:
            for process in ("ttHcc", "ttHbb", "qcd"):
                lines.append(
                    f"  {process} Hcc-pass yield/eff/raw = "
                    f"{row[f'{process}_Hcc_pass_weighted_yield']:.8g} / "
                    f"{row[f'{process}_Hcc_pass_weighted_efficiency']:.6g} / "
                    f"{row[f'{process}_Hcc_pass_raw_event_count']}"
                )
        else:
            lines.append("  event-level fields = N/A (not reconstructed for historical tight)")
    return "\n".join(lines) + "\n"


def write_strategy_comparison_outputs(
    rows: list[dict[str, Any]],
    outdirs: dict[str, Path],
) -> dict[str, Any]:
    payload = {
        "status": "available",
        "source": "unified boosted-tagger merged aggregates",
        "new_root_or_condor_processing": False,
        "region_efficiency_level": "candidate_fatjet",
        "region_efficiency_weighting": "analysis_weight = sample_norm * abs(event_weight_raw)",
        "region_efficiency_normalization": (
            "weighted jet yield in region / total weighted jet yield of truth category"
        ),
        "nominal_is_not_final_wp": True,
        "working_points": rows,
    }
    write_csv(outdirs["tables"] / "boosted_higgs_tagger_wp_comparison.csv", rows)
    write_json(outdirs["summaries"] / "boosted_higgs_tagger_wp_comparison.json", payload)
    (outdirs["summaries"] / "boosted_higgs_tagger_wp_comparison.txt").write_text(
        _summary_text(payload), encoding="utf-8"
    )
    return payload


def _draw_contours(ax: plt.Axes, contour_payload: dict[str, Any]) -> list[dict[str, Any]]:
    x_edges = np.asarray(contour_payload["x_edges"], dtype=np.float64)
    y_edges = np.asarray(contour_payload["y_edges"], dtype=np.float64)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    categories = list(contour_payload["categories"])
    weights = np.asarray(contour_payload["weight_category"], dtype=np.float64)
    histograms = [
        (category, weights[index])
        for index, category in enumerate(categories)
        if index < weights.shape[0]
    ]
    plotted: list[dict[str, Any]] = []
    for zorder, (category, histogram) in enumerate(
        _sorted_contour_histograms(histograms), start=1
    ):
        normalized = _smoothed_normalized_histogram(histogram)
        levels = _enclosed_fraction_contour_levels(normalized)
        filled = _filled_contour_levels(levels, normalized)
        if levels.size == 0 or filled.size <= 1:
            continue
        plotted.append(category)
        cmap = _make_alpha_cmap(category["color"], f"strategy_{category['key']}", len(filled) - 1)
        ax.contourf(
            x_centers, y_centers, normalized.T, levels=filled, cmap=cmap,
            antialiased=True, extend="max", zorder=float(zorder),
        )
        ax.contour(
            x_centers, y_centers, normalized.T, levels=levels,
            colors=[category["color"]], linewidths=0.7, alpha=0.35,
            zorder=float(zorder) + 0.1,
        )
    return plotted


def _base_strategy_axes(
    contour_payload: dict[str, Any], style: PlotStyle
) -> tuple[plt.Figure, plt.Axes, list[dict[str, Any]]]:
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(9.0, 7.3))
    plotted = _draw_contours(ax, contour_payload)
    ax.set_xlabel(SCORE_LABELS["gpart_higgs_vs_qcd"], fontsize=style.label_size)
    ax.set_ylabel(SCORE_LABELS["gpart_xbb_vs_xcc"], fontsize=style.label_size)
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    ax.tick_params(axis="both", labelsize=style.tick_size)
    ax.grid(True, alpha=0.16)
    return fig, ax, plotted


def _nominal_region_plot_definition() -> dict[str, Any]:
    return {
        "region_definitions": {
            "qcd_others": {
                "label": "non-Higgs / rejected region",
                "annotation": {"x": 0.13, "y": 0.56, "ha": "left"},
            },
            "hcc": {
                "label": "Hcc region",
                "annotation": {"x": 0.98, "y": 0.66, "ha": "right"},
            },
            "hbb": {
                "label": "Hbb region",
                "annotation": {"x": 0.98, "y": 0.97, "ha": "right"},
            },
        }
    }


def _nominal_region_efficiencies(row: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        "qcd_others": {
            "hbb_pure": float(row["hbb_pure_rejected"]),
            "hcc_pure": float(row["hcc_pure_rejected"]),
            "others": float(row["non_higgs_rejected"]),
        },
        "hcc": {
            "hbb_pure": float(row["hbb_pure_to_Hcc"]),
            "hcc_pure": float(row["hcc_pure_to_Hcc"]),
            "others": float(row["non_higgs_to_Hcc"]),
        },
        "hbb": {
            "hbb_pure": float(row["hbb_pure_to_Hbb"]),
            "hcc_pure": float(row["hcc_pure_to_Hbb"]),
            "others": float(row["non_higgs_to_Hbb"]),
        },
    }


def _region_annotations(ax: plt.Axes, x_cut: float, y_split: float) -> None:
    ax.text(0.89, 0.56, "Hcc region", ha="center", va="center", fontsize=11, color="#303030")
    ax.text(0.89, 0.91, "Hbb region", ha="center", va="center", fontsize=11, color="#303030")
    ax.text(0.35, 0.09, "non-Higgs / rejected", ha="center", fontsize=10, color="#555555")


def plot_strategy_v3(
    contour_payload: dict[str, Any],
    nominal_row: dict[str, Any],
    outpath: Path,
    style: PlotStyle,
) -> None:
    fig, ax, plotted = _base_strategy_axes(contour_payload, style)
    color = CONTOUR_REGION_LINE_COLOR
    x_cut = float(nominal_row["x_cut"])
    ax.axvline(
        x_cut, color=color, lw=CONTOUR_REGION_LINEWIDTH, zorder=10
    )
    ax.plot(
        [x_cut, 1.0],
        [0.80, 0.80],
        color=color,
        lw=CONTOUR_REGION_LINEWIDTH,
        zorder=10,
    )
    _annotate_globalpart3_regions(
        ax,
        _nominal_region_efficiencies(nominal_row),
        style,
        _nominal_region_plot_definition(),
    )
    handles = _globalpart3_contour_handles(plotted)
    handles.append(
        Line2D([], [], color=color, lw=CONTOUR_REGION_LINEWIDTH, label="Nominal loose candidate: non-Higgs mistag = 2%")
    )
    ax.legend(handles=handles, frameon=False, fontsize=style.legend_size, loc="upper left")
    private_cms_label(ax, style)
    _save_plot(fig, outpath, style, left=0.14, right=0.97, bottom=0.14, top=0.87, save_pdf=False)


def plot_strategy_v3_comparison(
    contour_payload: dict[str, Any],
    rows: list[dict[str, Any]],
    outpath: Path,
    style: PlotStyle,
) -> None:
    fig, ax, plotted = _base_strategy_axes(contour_payload, style)
    nominal = CONTOUR_REGION_LINE_COLOR
    nominal_x = float(rows[0]["x_cut"])
    tighter_x = float(rows[1]["x_cut"])
    tighter = "#2b6f9f"
    historical = "#777777"
    ax.axvline(nominal_x, color=nominal, lw=CONTOUR_REGION_LINEWIDTH, zorder=10)
    ax.plot(
        [nominal_x, 1.0], [0.80, 0.80],
        color=nominal, lw=CONTOUR_REGION_LINEWIDTH, zorder=10,
    )
    ax.axvline(tighter_x, color=tighter, lw=CONTOUR_REGION_LINEWIDTH, ls="-.", zorder=9)
    ax.axvline(HISTORICAL_X_CUT, color=historical, lw=CONTOUR_REGION_LINEWIDTH, ls="--", zorder=8)
    ax.plot(
        [HISTORICAL_X_CUT, 1.0], [HISTORICAL_Y_SPLIT, HISTORICAL_Y_SPLIT],
        color=historical, lw=CONTOUR_REGION_LINEWIDTH, ls="--", zorder=8,
    )
    _region_annotations(ax, nominal_x, 0.80)
    handles = _globalpart3_contour_handles(plotted)
    handles.extend(
        [
            Line2D([], [], color=nominal, lw=CONTOUR_REGION_LINEWIDTH, label="Nominal loose candidate: non-Higgs mistag = 2%"),
            Line2D([], [], color=tighter, lw=CONTOUR_REGION_LINEWIDTH, ls="-.", label="Tighter alternative: non-Higgs mistag = 1%"),
            Line2D([], [], color=historical, lw=CONTOUR_REGION_LINEWIDTH, ls="--", label="Historical tight region reference"),
        ]
    )
    ax.legend(
        handles=handles, frameon=False, fontsize=max(style.legend_size - 0.4, 7.5),
        loc="upper left", labelspacing=0.35,
    )
    private_cms_label(ax, style)
    _save_plot(fig, outpath, style, left=0.14, right=0.97, bottom=0.14, top=0.87, save_pdf=False)


def write_and_plot_strategy_comparison(
    histogram_payload: dict[str, Any],
    outdirs: dict[str, Path],
    style: PlotStyle,
    *,
    skip_plots: bool,
) -> dict[str, Any]:
    y_split = histogram_payload.get("y_split_study")
    hcc = histogram_payload.get("hcc_wp_scan")
    contour = histogram_payload.get("contour_payloads", {}).get(STRATEGY_CONTOUR_KEY)
    if y_split is None or hcc is None or contour is None:
        raise ValueError(
            "Strategy comparison requires dedicated y-split, Hcc scan, and x-vs-y contour payloads."
        )
    rows = derive_strategy_comparison(y_split, hcc)
    summary = write_strategy_comparison_outputs(rows, outdirs)
    if not skip_plots:
        plot_strategy_v3(
            contour,
            rows[0],
            outdirs["plots"] / "gpart_higgs_vs_qcd__vs__gpart_xbb_vs_xcc__strategy_v3.png",
            style,
        )
        plot_strategy_v3_comparison(
            contour,
            rows,
            outdirs["plots"] / "gpart_higgs_vs_qcd__vs__gpart_xbb_vs_xcc__strategy_v3_comparison.png",
            style,
        )
    return summary
