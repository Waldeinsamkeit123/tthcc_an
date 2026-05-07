from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from tthcc_an.definitions import (
    GLOBALPART3_FIXED_OTHER_EFF_TARGETS,
    GLOBALPART3_FIXED_X_CUT,
    SCORE_LABELS,
    TARGET_DEFINITIONS,
    TRUTH_LABEL_ORDER,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def format_summary_text(
    target: str,
    score_name: str,
    rows: list[dict[str, Any]],
    counts_by_label: dict[str, dict[str, float]],
    sample_summaries: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    lines: list[str] = []
    lines.append("=== Boosted Higgs Tagger Study ===")
    lines.append(f"Target: {TARGET_DEFINITIONS[target]['title']}")
    lines.append(f"Score: {SCORE_LABELS[score_name]}")
    lines.append(f"Candidate strategy: {args.candidate_strategy}")
    lines.append(f"AK8 selection: pt >= {args.pt_min:.1f} GeV, |eta| <= {args.eta_max:.1f}")
    lines.append(
        f"Weighting: lumi={args.lumi_fb if args.lumi_fb is not None else 'none'} /fb, "
        f"xsec/gen_sumw normalization, weight branch={args.weight_branch}, analysis uses abs(event weight)"
    )
    lines.append("")
    lines.append("Loaded samples:")
    for summary in sample_summaries:
        lines.append(
            f"  - {summary['dataset']}: files={summary['n_files']}, events={summary['n_events']}, "
            f"selected_jets={summary['n_selected_jets']}, xsec_fb={summary['xsec_fb']}, "
            f"gen_sumw={summary['gen_sumw']}, sample_norm={summary['sample_norm']:.8g}"
        )
    lines.append("")
    lines.append("Truth-category content after finite-score selection:")
    for label in TRUTH_LABEL_ORDER:
        entry = counts_by_label[label]
        lines.append(
            f"  - {label}: N={entry['n_jets']}, weight_sum={entry['weight_sum']:.6f}, "
            f"signed_weight_sum={entry['signed_weight_sum']:.6f}"
        )
    lines.append("")
    lines.append(
        f"{'TargetEff':>10}  {'Cut':>10}  {'SigEff':>8}  {'BkgEff':>8}  "
        f"{'Y_sig':>12}  {'Y_bkg':>12}  {'S/B':>10}  {'S/sqrt(S+B)':>12}  "
        f"{'S/sqrt(B)':>10}  {'Purity':>8}"
    )
    lines.append("-" * 136)
    for row in rows:
        purity = row["purity"]
        purity_str = f"{purity*100:7.2f}%" if np.isfinite(purity) else "   nan%"
        s_over_b = row["s_over_b"]
        s_over_b_str = "inf" if np.isinf(s_over_b) else f"{s_over_b:.6f}"
        s_over_sqrt_s_plus_b = row["s_over_sqrt_s_plus_b"]
        s_over_sqrt_s_plus_b_str = (
            f"{s_over_sqrt_s_plus_b:.6f}" if np.isfinite(s_over_sqrt_s_plus_b) else "nan"
        )
        s_over_sqrt_b = row["s_over_sqrt_b"]
        s_over_sqrt_b_str = "inf" if np.isinf(s_over_sqrt_b) else f"{s_over_sqrt_b:.6f}"
        lines.append(
            f"{row['target_sig_eff']*100:8.1f}%  {row['score_cut']:10.6f}  "
            f"{row['actual_sig_eff']*100:7.2f}%  {row['bkg_eff']*100:7.2f}%  "
            f"{row['yield_sig_pass']:12.6f}  {row['yield_bkg_pass']:12.6f}  "
            f"{s_over_b_str:>10}  {s_over_sqrt_s_plus_b_str:>12}  "
            f"{s_over_sqrt_b_str:>10}  {purity_str:>8}"
        )
    return "\n".join(lines) + "\n"


def format_contour_region_efficiency_text(
    plot_def: dict[str, Any],
    region_efficiencies: dict[str, dict[str, float]],
) -> str:
    x_score = str(plot_def["x_score"])
    y_score = str(plot_def["y_score"])
    region_definitions = dict(plot_def.get("region_definitions", {}))
    region_rows = [
        ("qcd_others", str(region_definitions.get("qcd_others", {}).get("label", "QCD&Others region"))),
        ("hcc", str(region_definitions.get("hcc", {}).get("label", "Hcc region"))),
        ("hbb", str(region_definitions.get("hbb", {}).get("label", "Hbb region"))),
    ]
    category_columns = [
        ("hbb_pure", "hbb_pure"),
        ("hcc_pure", "hcc_pure"),
        ("others", "Others"),
    ]

    def _format_percent(value: float) -> str:
        if not np.isfinite(value):
            return "n/a"
        return f"{value * 100.0:.1f}%"

    lines: list[str] = []
    lines.append("=== GlobalParT3 Contour Region Efficiencies ===")
    lines.append(f"x-axis: {SCORE_LABELS[x_score]}")
    lines.append(f"y-axis: {SCORE_LABELS[y_score]}")
    if plot_def.get("region_preset") not in {None, "", "default"}:
        preset_label = str(plot_def.get("region_preset_label", plot_def["region_preset"]))
        preset_description = str(plot_def.get("region_preset_description", "")).strip()
        preset_line = f"Region preset: {preset_label}"
        if preset_description:
            preset_line += f" ({preset_description})"
        lines.append(preset_line)
    lines.append("Weighting: analysis_weight = sample_norm * abs(event_weight_raw)")
    lines.append("Normalization: weighted efficiency = weighted yield in region / total weighted yield of that category")
    lines.append("")
    lines.append("Region definitions:")
    hcc_region = region_definitions.get("hcc", {})
    hbb_region = region_definitions.get("hbb", {})
    qcd_region = region_definitions.get("qcd_others", {})
    lines.append(
        f"  - {hcc_region.get('label', 'Hcc region')}: "
        f"(x > {hcc_region.get('x_min_exclusive', 'n/a')}) and "
        f"({hcc_region.get('y_min_exclusive', 'n/a')} < y <= {hcc_region.get('y_max_inclusive', 'n/a')})"
    )
    lines.append(
        f"  - {hbb_region.get('label', 'Hbb region')}: "
        f"(x > {hbb_region.get('x_min_exclusive', 'n/a')}) and "
        f"({hbb_region.get('y_min_exclusive', 'n/a')} < y <= {hbb_region.get('y_max_inclusive', 'n/a')})"
    )
    lines.append(f"  - {qcd_region.get('label', 'QCD&Others region')}: not(Hcc region or Hbb region)")
    lines.append("")
    lines.append(f"{'Region':<20}  {'hbb_pure':>12}  {'hcc_pure':>12}  {'Others':>12}")
    lines.append("-" * 62)
    for region_key, region_title in region_rows:
        row = region_efficiencies.get(region_key, {})
        values = [_format_percent(float(row.get(category_key, float('nan')))) for category_key, _ in category_columns]
        lines.append(f"{region_title:<20}  {values[0]:>12}  {values[1]:>12}  {values[2]:>12}")
    return "\n".join(lines) + "\n"


def format_fixed_other_efficiency_scan_text(
    plot_def: dict[str, Any],
    scan_payload: dict[str, Any],
) -> str:
    x_score = str(plot_def["x_score"])
    y_score = str(plot_def["y_score"])
    region_definitions = dict(plot_def.get("region_definitions", {}))

    def _format_percent(value: float) -> str:
        if not np.isfinite(value):
            return "n/a"
        return f"{value * 100.0:.3g}%"

    def _format_cut(value: float) -> str:
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.4f}"

    lines: list[str] = []
    lines.append("=== Fixed-Other-Efficiency X-Cut Scan ===")
    lines.append(f"x-axis: {SCORE_LABELS[x_score]}")
    lines.append(f"y-axis: {SCORE_LABELS[y_score]}")
    if plot_def.get("region_preset") not in {None, "", "default"}:
        preset_label = str(plot_def.get("region_preset_label", plot_def["region_preset"]))
        preset_description = str(plot_def.get("region_preset_description", "")).strip()
        preset_line = f"Region preset: {preset_label}"
        if preset_description:
            preset_line += f" ({preset_description})"
        lines.append(preset_line)
    lines.append("Goal: keep the y-region definition fixed and scan only the x-threshold.")
    lines.append("Normalization: each efficiency is normalized to the total weighted yield of that category.")
    lines.append(
        "Requested Others efficiencies: "
        + ", ".join(
            _format_percent(float(value))
            for value in scan_payload.get("other_eff_targets", GLOBALPART3_FIXED_OTHER_EFF_TARGETS)
        )
    )
    lines.append("")

    for region_key in ["hcc", "hbb"]:
        region_def = dict(region_definitions.get(region_key, {}))
        region_label = str(region_def.get("label", region_key))
        lines.append(
            f"{region_label}: "
            f"({region_def.get('y_min_exclusive', 'n/a')} < y <= {region_def.get('y_max_inclusive', 'n/a')}) "
            "with scanned cut x > threshold"
        )
        lines.append(
            f"{'Target Others':>14}  {'x-cut':>10}  {'hbb_pure':>12}  {'hcc_pure':>12}  {'Others':>12}"
        )
        lines.append("-" * 70)
        for row in scan_payload.get("regions", {}).get(region_key, []):
            efficiencies = dict(row.get("efficiencies", {}))
            lines.append(
                f"{_format_percent(float(row.get('target_other_eff', float('nan')))):>14}  "
                f"{_format_cut(float(row.get('x_cut', float('nan')))):>10}  "
                f"{_format_percent(float(efficiencies.get('hbb_pure', float('nan')))):>12}  "
                f"{_format_percent(float(efficiencies.get('hcc_pure', float('nan')))):>12}  "
                f"{_format_percent(float(efficiencies.get('others', float('nan')))):>12}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_fixed_x_ycut_scan_text(
    plot_def: dict[str, Any],
    scan_payload: dict[str, Any],
) -> str:
    x_score = str(plot_def["x_score"])
    y_score = str(plot_def["y_score"])
    fixed_x_cut = float(scan_payload.get("fixed_x_cut", GLOBALPART3_FIXED_X_CUT))
    y_region = dict(scan_payload.get("y_region", {}))

    def _format_percent(value: float) -> str:
        if not np.isfinite(value):
            return "n/a"
        return f"{value * 100.0:.3g}%"

    def _format_cut(value: float) -> str:
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.4f}"

    def _format_merit(value: float) -> str:
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.4g}"

    best = dict(scan_payload.get("best", {}))

    lines: list[str] = []
    lines.append("=== Fixed-X Y-Cut Scan ===")
    lines.append(f"x-axis: {SCORE_LABELS[x_score]}")
    lines.append(f"y-axis: {SCORE_LABELS[y_score]}")
    if plot_def.get("region_preset") not in {None, "", "default"}:
        preset_label = str(plot_def.get("region_preset_label", plot_def["region_preset"]))
        preset_description = str(plot_def.get("region_preset_description", "")).strip()
        preset_line = f"Region preset: {preset_label}"
        if preset_description:
            preset_line += f" ({preset_description})"
        lines.append(preset_line)
    lines.append(f"Fixed x-cut: x > {fixed_x_cut:.4f}")
    lines.append(
        "Hcc region: x > x_cut and y <= y_cut; "
        "Hbb region: x > x_cut and y > y_cut"
    )
    lines.append("Normalization: each efficiency is normalized to the total weighted yield of that category.")
    lines.append(
        "Merit = sqrt( eps(hcc->Hcc) * eps(hbb->Hbb) / "
        "[(eps(hbb->Hcc)+1e-6) * (eps(hcc->Hbb)+1e-6)] )"
    )
    lines.append("")
    lines.append(
        f"Scanned y-cut range: {_format_cut(float(y_region.get('min', float('nan'))))} "
        f"to {_format_cut(float(y_region.get('max', float('nan'))))}"
    )
    lines.append(f"Best y-cut by merit: {_format_cut(float(best.get('y_cut', float('nan'))))}")
    lines.append(f"Best merit: {_format_merit(float(best.get('merit', float('nan'))))}")
    if best:
        best_eff = dict(best.get("efficiencies", {}))
        lines.append(
            "Best point efficiencies: "
            f"hcc->Hcc={_format_percent(float(best_eff.get('hcc_to_hcc', float('nan'))))}, "
            f"hbb->Hcc={_format_percent(float(best_eff.get('hbb_to_hcc', float('nan'))))}, "
            f"hbb->Hbb={_format_percent(float(best_eff.get('hbb_to_hbb', float('nan'))))}, "
            f"hcc->Hbb={_format_percent(float(best_eff.get('hcc_to_hbb', float('nan'))))}"
        )
    lines.append("")
    lines.append(
        f"{'y-cut':>10}  {'hcc->Hcc':>12}  {'hbb->Hcc':>12}  {'hbb->Hbb':>12}  {'hcc->Hbb':>12}  {'Merit':>10}"
    )
    lines.append("-" * 82)
    for row in scan_payload.get("rows", []):
        eff = dict(row.get("efficiencies", {}))
        lines.append(
            f"{_format_cut(float(row.get('y_cut', float('nan')))):>10}  "
            f"{_format_percent(float(eff.get('hcc_to_hcc', float('nan')))):>12}  "
            f"{_format_percent(float(eff.get('hbb_to_hcc', float('nan')))):>12}  "
            f"{_format_percent(float(eff.get('hbb_to_hbb', float('nan')))):>12}  "
            f"{_format_percent(float(eff.get('hcc_to_hbb', float('nan')))):>12}  "
            f"{_format_merit(float(row.get('merit', float('nan')))):>10}"
        )
    return "\n".join(lines).rstrip() + "\n"
