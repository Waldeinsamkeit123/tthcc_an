from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from tthcc_an.boosted_higgs_tagger_study.definitions import TRUTH_LABEL_TO_CODE
from tthcc_an.boosted_higgs_tagger_study.plotting import PlotStyle, private_cms_label
from tthcc_an.boosted_higgs_tagger_study.reporting import write_csv, write_json
from tthcc_an.boosted_higgs_tagger_study.y_split_study import EVENT_STATES


COLORS = {
    "hcc_hcc": "#c43c39", "hcc_hbb": "#e89c90",
    "hbb_hcc": "#78a7d3", "hbb_hbb": "#2364aa",
    "ttHcc": "#c43c39", "ttHbb": "#2364aa", "qcd": "#343a40",
}


def _ratio(value: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    output = np.zeros(np.broadcast_shapes(value.shape, denominator.shape), dtype=np.float64)
    np.divide(value, denominator, out=output, where=denominator > 0.0)
    return output


def _none_or_float(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def hcc_vs_hbb_purity_scan(
    aggregate: dict[str, Any], target_non_higgs_efficiency: float = 0.02
) -> np.ndarray:
    wp_index = next(
        index
        for index, wp in enumerate(aggregate["working_points"])
        if np.isclose(wp["non_higgs_target_efficiency"], target_non_higgs_efficiency)
    )
    truth_regions = np.asarray(
        aggregate["jet_truth_region_yields"][wp_index], dtype=np.float64
    )
    signal = truth_regions[TRUTH_LABEL_TO_CODE["hcc_pure"], :, 0]
    background = truth_regions[TRUTH_LABEL_TO_CODE["hbb_pure"], :, 0]
    return _ratio(signal, signal + background)


def hcc_vs_hbb_significance_scan(
    aggregate: dict[str, Any], target_non_higgs_efficiency: float = 0.02
) -> np.ndarray:
    wp_index = next(
        index
        for index, wp in enumerate(aggregate["working_points"])
        if np.isclose(wp["non_higgs_target_efficiency"], target_non_higgs_efficiency)
    )
    truth_regions = np.asarray(
        aggregate["jet_truth_region_yields"][wp_index], dtype=np.float64
    )
    signal = truth_regions[TRUTH_LABEL_TO_CODE["hcc_pure"], :, 0]
    background = truth_regions[TRUTH_LABEL_TO_CODE["hbb_pure"], :, 0]
    denominator = signal + background
    return np.divide(
        signal,
        np.sqrt(np.maximum(denominator, 0.0)),
        out=np.zeros_like(signal),
        where=denominator > 0.0,
    )


def derive_y_split_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    process_codes = {
        entry["process"]: int(entry["code"]) for entry in aggregate["process_entries"]
    }
    for process in ("ttHcc", "ttHbb", "qcd"):
        if process not in process_codes:
            raise ValueError(f"y-split study is missing required process {process}.")
    hcc_truth = TRUTH_LABEL_TO_CODE["hcc_pure"]
    hbb_truth = TRUTH_LABEL_TO_CODE["hbb_pure"]
    state_index = {state: index for index, state in enumerate(EVENT_STATES)}
    complete = aggregate["availability"]["event_weighted_complete"]
    raw_available = aggregate["availability"]["event_raw_counts"]
    rows: list[dict[str, Any]] = []

    for wp_index, wp in enumerate(aggregate["working_points"]):
        truth_regions = aggregate["jet_truth_region_yields"][wp_index]
        truth_x = aggregate["jet_truth_x_totals"][wp_index]
        truth_base = aggregate["jet_truth_candidate_totals"]
        process_regions = aggregate["jet_process_region_yields"][wp_index]
        process_x = aggregate["jet_process_x_totals"][wp_index]
        for y_index, y_split in enumerate(aggregate["y_splits"]):
            row: dict[str, Any] = {
                "non_higgs_target_wp": float(wp["non_higgs_target_efficiency"]),
                "x_cut": float(wp["x_cut"]),
                "x_cut_evaluated": float(wp["x_cut_evaluated"]),
                "y_split": float(y_split),
            }
            for truth_name, truth_code in (("hcc", hcc_truth), ("hbb", hbb_truth)):
                for region_index, region in enumerate(("Hcc", "Hbb")):
                    value = truth_regions[truth_code, y_index, region_index]
                    row[f"jet_absolute_{truth_name}_to_{region}"] = float(
                        _ratio(value, truth_base[truth_code])
                    )
                    row[f"jet_conditional_{truth_name}_to_{region}"] = float(
                        _ratio(value, truth_x[truth_code])
                    )
            qcd_code = process_codes["qcd"]
            for region_index, region in enumerate(("Hcc", "Hbb")):
                value = process_regions[qcd_code, y_index, region_index]
                row[f"jet_absolute_qcd_to_{region}"] = float(
                    _ratio(value, aggregate["jet_process_candidate_totals"][qcd_code])
                )
                row[f"jet_conditional_qcd_to_{region}"] = float(
                    _ratio(value, process_x[qcd_code])
                )

            for process, code in process_codes.items():
                baseline_weighted = float(aggregate["baseline_weighted_yields"][code])
                inclusive_weighted = float(
                    aggregate["inclusive_weighted_yields"][wp_index, code]
                )
                row[f"{process}_candidate_baseline_weighted_yield"] = baseline_weighted
                row[f"{process}_inclusive_x_weighted_yield"] = inclusive_weighted
                row[f"{process}_inclusive_x_absolute_efficiency"] = float(
                    _ratio(inclusive_weighted, baseline_weighted)
                )
                if raw_available:
                    row[f"{process}_candidate_baseline_raw_event_count"] = int(
                        aggregate["baseline_raw_counts"][code]
                    )
                    row[f"{process}_inclusive_x_pass_raw_event_count"] = int(
                        aggregate["inclusive_raw_counts"][wp_index, code]
                    )
                else:
                    row[f"{process}_candidate_baseline_raw_event_count"] = None
                    row[f"{process}_inclusive_x_pass_raw_event_count"] = None

                if complete:
                    states = aggregate["state_weighted_yields"][wp_index, code, y_index]
                    raw_states = aggregate["state_raw_counts"][wp_index, code, y_index]
                    hcc_pass = states[state_index["hcc_only"]] + states[state_index["both"]]
                    hbb_pass = states[state_index["hbb_only"]] + states[state_index["both"]]
                    for region, value in (("Hcc_pass", hcc_pass), ("Hbb_pass", hbb_pass)):
                        row[f"{process}_{region}_weighted_yield"] = float(value)
                        row[f"{process}_{region}_absolute_efficiency"] = float(
                            _ratio(value, baseline_weighted)
                        )
                        row[f"{process}_{region}_conditional_efficiency"] = float(
                            _ratio(value, inclusive_weighted)
                        )
                    for state, index in state_index.items():
                        value = float(states[index])
                        row[f"{process}_{state}_weighted_yield"] = value
                        row[f"{process}_{state}_absolute_efficiency"] = float(
                            _ratio(value, baseline_weighted)
                        )
                        conditional_value = value
                        if state == "neither":
                            conditional_value = max(
                                value - (baseline_weighted - inclusive_weighted), 0.0
                            )
                        row[f"{process}_{state}_conditional_fraction"] = float(
                            _ratio(conditional_value, inclusive_weighted)
                        )
                        row[f"{process}_{state}_raw_event_count"] = int(raw_states[index])
                    row[f"{process}_Hcc_pass_raw_event_count"] = int(
                        raw_states[state_index["hcc_only"]] + raw_states[state_index["both"]]
                    )
                    row[f"{process}_Hbb_pass_raw_event_count"] = int(
                        raw_states[state_index["hbb_only"]] + raw_states[state_index["both"]]
                    )
                else:
                    legacy_hcc = float(
                        aggregate["legacy_event_hcc_pass_yields"][wp_index, code, y_index]
                    )
                    row[f"{process}_Hcc_pass_weighted_yield"] = legacy_hcc
                    row[f"{process}_Hcc_pass_absolute_efficiency"] = float(
                        _ratio(legacy_hcc, baseline_weighted)
                    )
                    row[f"{process}_Hcc_pass_conditional_efficiency"] = float(
                        _ratio(legacy_hcc, inclusive_weighted)
                    )
                    for key in (
                        "Hbb_pass_weighted_yield", "Hbb_pass_absolute_efficiency",
                        "Hbb_pass_conditional_efficiency",
                    ):
                        row[f"{process}_{key}"] = None
                    for state in EVENT_STATES:
                        row[f"{process}_{state}_weighted_yield"] = None
                        row[f"{process}_{state}_conditional_fraction"] = None
                        row[f"{process}_{state}_raw_event_count"] = None
                    row[f"{process}_Hcc_pass_raw_event_count"] = None
                    row[f"{process}_Hbb_pass_raw_event_count"] = None
            rows.append(row)
    return rows


def _reference_rows(rows: list[dict[str, Any]], references: list[float]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if any(np.isclose(row["y_split"], reference) for reference in references)
    ]


def _summary_text(payload: dict[str, Any]) -> str:
    lines = [
        "=== Hcc/Hbb y-split study ===",
        "Hcc: x >= x_Higgs and y <= y_split",
        "Hbb: x >= x_Higgs and y > y_split",
        "No significance ranking and no final y_split selection.",
        "Diagnostic candidates:",
        *[
            f"  y={item['y_split']:.2f}: {item['role']}"
            for item in payload["diagnostic_candidates"]
        ],
        f"Source: {payload['precision']['source']}",
        f"Complete weighted event states: {payload['availability']['event_weighted_complete']}",
        f"Raw event counts available: {payload['availability']['event_raw_counts']}",
    ]
    if not payload["availability"]["event_weighted_complete"]:
        lines.extend([
            "Legacy limitation: only weighted Hcc-pass event yields are recoverable.",
            "Hbb-pass and overlap states require a new production.",
        ])
    if not payload["availability"]["event_raw_counts"]:
        lines.append("Raw counts are unavailable and were not inferred from weighted yields.")
    for wp in payload["working_points"]:
        lines.extend([
            "",
            "=" * 60,
            f"Inclusive Higgs WP: non-Higgs jet mistag = {100 * wp['non_higgs_target_efficiency']:g}%",
            f"x_cut = {wp['x_cut']:.6f}",
            f"x_cut evaluated = {wp['x_cut_evaluated']:.6f}",
            "=" * 60,
        ])
        selected = [
            row for row in payload["reference_rows"]
            if np.isclose(row["non_higgs_target_wp"], wp["non_higgs_target_efficiency"])
        ]
        for row in selected:
            lines.extend([
                "",
                f"y_split = {row['y_split']:.2f}",
                "Jet conditional migration:",
                f"  hcc -> Hcc/Hbb = {row['jet_conditional_hcc_to_Hcc']:.6g} / {row['jet_conditional_hcc_to_Hbb']:.6g}",
                f"  hbb -> Hcc/Hbb = {row['jet_conditional_hbb_to_Hcc']:.6g} / {row['jet_conditional_hbb_to_Hbb']:.6g}",
                f"  QCD -> Hcc/Hbb = {row['jet_conditional_qcd_to_Hcc']:.6g} / {row['jet_conditional_qcd_to_Hbb']:.6g}",
            ])
            for process in ("ttHcc", "ttHbb", "qcd"):
                lines.append(
                    f"  {process} Hcc-pass weighted yield/absolute eff = "
                    f"{row[f'{process}_Hcc_pass_weighted_yield']:.8g} / "
                    f"{row[f'{process}_Hcc_pass_absolute_efficiency']:.6g}"
                )
                if payload["availability"]["event_weighted_complete"]:
                    lines.append(
                        f"  {process} Hbb-pass/both weighted yield = "
                        f"{row[f'{process}_Hbb_pass_weighted_yield']:.8g} / "
                        f"{row[f'{process}_both_weighted_yield']:.8g}"
                    )
    return "\n".join(lines) + "\n"


def write_y_split_outputs(
    aggregate: dict[str, Any],
    outdirs: dict[str, Path],
    candidate_selection: dict[str, Any],
) -> dict[str, Any]:
    rows = derive_y_split_rows(aggregate)
    references = _reference_rows(rows, aggregate["config"]["reference_points"])
    payload = {
        "status": "available",
        "definitions": {
            "x": "gpart_higgs_vs_qcd",
            "y": "gpart_xbb_vs_xcc = Xbb/(Xbb+Xcc)",
            "Hcc": "x >= x_Higgs and y <= y_split",
            "Hbb": "x >= x_Higgs and y > y_split",
            "event_regions_are_mutually_exclusive": False,
            "jet_regions_are_mutually_exclusive": True,
        },
        "candidate_selection": candidate_selection,
        "config": aggregate["config"],
        "working_points": aggregate["working_points"],
        "availability": aggregate["availability"],
        "precision": aggregate["precision"],
        "final_y_split_selected": False,
        "diagnostic_candidates": aggregate["config"]["diagnostic_candidates"],
        "scan": rows,
        "reference_rows": references,
    }
    write_csv(outdirs["tables"] / "y_split_scan.csv", rows)
    write_csv(outdirs["tables"] / "y_split_reference_points.csv", references)
    write_json(outdirs["summaries"] / "y_split_study.json", payload)
    (outdirs["summaries"] / "y_split_study.txt").write_text(
        _summary_text(payload), encoding="utf-8"
    )
    return payload


def _finish(fig: plt.Figure, ax: plt.Axes, path: Path, style: PlotStyle) -> None:
    ax.tick_params(labelsize=style.tick_size)
    ax.legend(frameon=False, fontsize=style.legend_size)
    private_cms_label(ax, style)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.88)
    fig.savefig(path, dpi=style.dpi)
    plt.close(fig)


def plot_y_split_study(
    aggregate: dict[str, Any],
    rows: list[dict[str, Any]],
    outdir: Path,
    style: PlotStyle,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    plt.style.use(hep.style.CMS)
    references = aggregate["config"]["reference_points"]

    nominal_target = 0.02
    nominal_wp = next(
        wp
        for wp in aggregate["working_points"]
        if np.isclose(wp["non_higgs_target_efficiency"], nominal_target)
    )
    y_splits = np.asarray(aggregate["y_splits"], dtype=np.float64)
    purity = hcc_vs_hbb_purity_scan(aggregate, nominal_target)
    nominal_y = 0.80
    nominal_y_index = int(np.argmin(np.abs(y_splits - nominal_y)))
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    ax.plot(
        y_splits,
        purity,
        color="#2a6f6b",
        linewidth=2.0,
        label="Hcc pure vs Hbb pure",
    )
    ax.scatter(
        [y_splits[nominal_y_index]],
        [purity[nominal_y_index]],
        color="#d62728",
        s=42,
        zorder=4,
        label="candidate: x={:.3f}, y={:.2f}".format(
            nominal_wp["x_cut"], y_splits[nominal_y_index]
        ),
    )
    ax.set_xlabel(
        "gParT3 Xbb vs Xcc upper cut (Hcc: y <= cut)", fontsize=style.label_size
    )
    ax.set_ylabel("Weighted jet purity S/(S+B)", fontsize=style.label_size)
    ax.set_xlim(float(y_splits[0]), float(y_splits[-1]))
    ax.set_ylim(0.0, 1.05)
    _finish(fig, ax, outdir / "hcc_vs_hbb__s_over_s_plus_b_scan.png", style)

    significance = hcc_vs_hbb_significance_scan(aggregate, nominal_target)
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    ax.plot(
        y_splits,
        significance,
        color="#2a6f6b",
        linewidth=2.0,
        label="Hcc pure vs Hbb pure",
    )
    ax.scatter(
        [y_splits[nominal_y_index]],
        [significance[nominal_y_index]],
        color="#d62728",
        s=42,
        zorder=4,
        label="candidate: x={:.3f}, y={:.2f}".format(
            nominal_wp["x_cut"], y_splits[nominal_y_index]
        ),
    )
    ax.set_xlabel(
        "gParT3 Xbb vs Xcc upper cut (Hcc: y <= cut)", fontsize=style.label_size
    )
    ax.set_ylabel("Weighted jet significance S/sqrt(S+B)", fontsize=style.label_size)
    ax.set_xlim(float(y_splits[0]), float(y_splits[-1]))
    ax.set_ylim(bottom=0.0)
    _finish(fig, ax, outdir / "hcc_vs_hbb__significance_scan.png", style)

    for wp in aggregate["working_points"]:
        target = wp["non_higgs_target_efficiency"]
        selected = [row for row in rows if np.isclose(row["non_higgs_target_wp"], target)]
        y = np.asarray([row["y_split"] for row in selected])
        tag = f"qcd_{100 * target:g}pct".replace(".", "p")

        fig, ax = plt.subplots(figsize=(9.2, 7.0))
        for key, label, color in (
            ("jet_conditional_hcc_to_Hcc", "hcc -> Hcc", COLORS["hcc_hcc"]),
            ("jet_conditional_hcc_to_Hbb", "hcc -> Hbb", COLORS["hcc_hbb"]),
            ("jet_conditional_hbb_to_Hcc", "hbb -> Hcc", COLORS["hbb_hcc"]),
            ("jet_conditional_hbb_to_Hbb", "hbb -> Hbb", COLORS["hbb_hbb"]),
        ):
            ax.plot(y, [row[key] for row in selected], label=label, color=color, lw=2)
        ax.axvline(0.85, color="#555555", ls=":", label="historical y=0.85")
        ax.set(xlabel="y split", ylabel="Conditional jet efficiency after x", ylim=(0, 1.05))
        _finish(fig, ax, outdir / f"{tag}__flavor_migration.png", style)

        fig, ax = plt.subplots(figsize=(9.2, 7.0))
        ax.plot(y, [row["jet_absolute_hcc_to_Hcc"] for row in selected], label="hcc -> Hcc", color=COLORS["hcc_hcc"], lw=2)
        ax.plot(y, [row["jet_absolute_hbb_to_Hbb"] for row in selected], label="hbb -> Hbb", color=COLORS["hbb_hbb"], lw=2)
        ax.axvline(0.85, color="#555555", ls=":", label="historical y=0.85")
        ax.set(xlabel="y split", ylabel="Absolute candidate-baseline jet efficiency", ylim=(0, 1.05))
        _finish(fig, ax, outdir / f"{tag}__absolute_flavor_efficiency.png", style)

        if aggregate["availability"]["event_weighted_complete"]:
            for process_set, suffix in ((("ttHcc", "ttHbb"), "signal"), (("qcd",), "qcd")):
                fig, ax = plt.subplots(figsize=(9.2, 7.0))
                for process in process_set:
                    for region, ls in (("Hcc_pass", "-"), ("Hbb_pass", "--"), ("both", ":")):
                        key = f"{process}_{region}_absolute_efficiency" if region != "both" else f"{process}_both_conditional_fraction"
                        ax.plot(y, [row[key] for row in selected], label=f"{process} {region}", ls=ls, lw=2)
                ax.set(xlabel="y split", ylabel="Weighted event efficiency / fraction", ylim=(0, 1.05))
                _finish(fig, ax, outdir / f"{tag}__event_efficiency_{suffix}.png", style)

            fig, ax = plt.subplots(figsize=(9.2, 7.0))
            for process in ("ttHcc", "ttHbb", "qcd"):
                ax.plot(y, [row[f"{process}_both_conditional_fraction"] for row in selected], label=f"{process} both", lw=2)
            ax.set(xlabel="y split", ylabel="Both fraction among inclusive-x events", ylim=(0, 1.05))
            _finish(fig, ax, outdir / f"{tag}__event_overlap.png", style)

        if aggregate["availability"]["event_raw_counts"]:
            for process_set, suffix in ((("ttHcc", "ttHbb"), "signal"), (("qcd",), "qcd")):
                fig, ax = plt.subplots(figsize=(9.2, 7.0))
                for process in process_set:
                    preferred = "Hcc_pass" if process == "ttHcc" else "Hbb_pass"
                    ax.plot(y, [row[f"{process}_{preferred}_raw_event_count"] for row in selected], label=f"{process} {preferred}", lw=2)
                ax.set(xlabel="y split", ylabel="Unweighted event count")
                _finish(fig, ax, outdir / f"{tag}__raw_event_counts_{suffix}.png", style)

    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    for wp in aggregate["working_points"]:
        selected = [row for row in rows if np.isclose(row["non_higgs_target_wp"], wp["non_higgs_target_efficiency"])]
        x = [row["jet_conditional_hbb_to_Hcc"] for row in selected]
        y = [row["jet_conditional_hcc_to_Hcc"] for row in selected]
        ax.plot(x, y, lw=2, label=f"{100 * wp['non_higgs_target_efficiency']:g}% non-Higgs WP")
        for ref in references:
            if ref >= 0.70:
                point = min(selected, key=lambda row: abs(row["y_split"] - ref))
                ax.scatter(point["jet_conditional_hbb_to_Hcc"], point["jet_conditional_hcc_to_Hcc"], s=22)
    ax.set(xlabel="hbb -> Hcc conditional leakage", ylabel="hcc -> Hcc conditional efficiency", xlim=(0, 1), ylim=(0, 1))
    _finish(fig, ax, outdir / "hcc_efficiency_vs_hbb_leakage.png", style)

    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    for wp in aggregate["working_points"]:
        selected = [row for row in rows if np.isclose(row["non_higgs_target_wp"], wp["non_higgs_target_efficiency"])]
        ax.plot(
            [row["jet_conditional_hcc_to_Hbb"] for row in selected],
            [row["jet_conditional_hbb_to_Hbb"] for row in selected],
            lw=2, label=f"{100 * wp['non_higgs_target_efficiency']:g}% non-Higgs WP",
        )
    ax.set(xlabel="hcc -> Hbb conditional leakage", ylabel="hbb -> Hbb conditional efficiency", xlim=(0, 1), ylim=(0, 1))
    _finish(fig, ax, outdir / "hbb_efficiency_vs_hcc_leakage.png", style)
