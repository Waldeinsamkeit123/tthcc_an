from __future__ import annotations

from pathlib import Path
from typing import Any

from tthcc_an.boosted_higgs_tagger_study.reporting import write_csv, write_json


def _scan_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    truth_efficiencies = result.get("truth_efficiencies", {})
    process_efficiencies = result.get("process_efficiencies", {})
    for ix, x_cut in enumerate(result["x_cuts"]):
        for iy, y_cut in enumerate(result["y_cuts"]):
            row = {
                "x_cut": float(x_cut),
                "y_cut": float(y_cut),
                "S": float(result["signal"][ix, iy]),
                "B": float(result["background"][ix, iy]),
                "S_over_B": float(result["signal_over_background"][ix, iy]),
                "S_over_sqrt_S_plus_B": float(result["significance"][ix, iy]),
                "S_over_sqrt_B": float(result["signal_over_sqrt_background"][ix, iy]),
                "relative_significance": float(result["relative_significance"][ix, iy]),
                result["signal_efficiency_name"]: float(result["signal_efficiency"][ix, iy]),
                "total_background_efficiency": float(
                    result["background_efficiency"][ix, iy]
                ),
            }
            for label, efficiency in truth_efficiencies.items():
                row[f"{label}_efficiency"] = float(efficiency[ix, iy])
            for process, efficiency in process_efficiencies.items():
                suffix = "acceptance" if result["level"] == "event" else "efficiency"
                row[f"{process}_{suffix}"] = float(efficiency[ix, iy])
            rows.append(row)
    return rows


def _summary_payload(
    result: dict[str, Any],
    scan_config: dict[str, Any],
    candidate_selection: dict[str, Any],
    current_references: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "level": result["level"],
        "status": "available",
        "ranking_metric": "S/sqrt(S+B)",
        "score_definition": {
            "x": "(Xbb + Xcc) / (Xbb + Xcc + QCD)",
            "y": "Xbb / (Xbb + Xcc)",
        },
        "new_scan_region": "x >= x_cut and y <= y_cut",
        "candidate_selection": candidate_selection,
        "scan_config": scan_config,
        "baseline": result["baseline"],
        "recommendations_are_diagnostics_not_final_wp": True,
        "recommendations": result["recommendations"],
        "current_wp_references": current_references,
        "available_process_efficiencies": sorted(result["process_efficiencies"]),
    }


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.6g}%"


def _summary_text(payload: dict[str, Any]) -> str:
    candidate = payload["candidate_selection"]
    lines = [
        "=== Hcc Working-Point Optimization ===",
        f"Level: {payload['level']}",
        "Ranking metric: S/sqrt(S+B)",
        "New scan convention: x >= x_cut and y <= y_cut",
        "x = (Xbb + Xcc) / (Xbb + Xcc + QCD)",
        "y = Xbb / (Xbb + Xcc)",
        (
            "Candidate preselection: "
            f"pt {candidate['pt_convention']} {candidate['pt_min']:.1f} GeV, "
            f"|eta| {candidate['eta_convention']} {candidate['eta_abs_max']:.2f}, "
            f"{candidate['mass_window_low']:.1f} "
            f"{'<' if candidate['mass_window_convention'] == 'open' else '<='} "
            f"{candidate['mass_variable']} "
            f"{'<' if candidate['mass_window_convention'] == 'open' else '<='} "
            f"{candidate['mass_window_high']:.1f} GeV"
        ),
        "",
    ]
    if payload["level"] == "jet":
        lines.extend(
            [
                "Signal: hcc_pure jets",
                (
                    "Background: hbb_pure + hcc_contaminated + hcc_partial + "
                    "hbb_contaminated + hbb_partial + top + other jets"
                ),
            ]
        )
    else:
        lines.extend(
            [
                "Signal: configured ttHcc MC events with >=1 candidate jet passing the region",
                "Background: all other configured MC events; each event is weighted once",
            ]
        )
    baseline = payload["baseline"]
    lines.extend(
        [
            (
                f"Candidate-preselection baseline: S={baseline['S']:.8g}, "
                f"B={baseline['B']:.8g}, "
                f"S/sqrt(S+B)={baseline['S_over_sqrt_S_plus_B']:.8g}"
            ),
            "",
            "Current preset references (historical convention is strict x > and 0 < y <=):",
        ]
    )
    for name, reference in payload["current_wp_references"].items():
        lines.append(
            f"  {name}: x={reference['x_cut']:.4f}, y={reference['y_cut']:.4f}; "
            f"{reference['description']}"
        )
    lines.extend(
        [
            "",
            "Optimization diagnostics (not automatically selected as a final WP):",
            (
                f"{'label':<31} {'x_cut':>8} {'y_cut':>8} {'S':>12} {'B':>12} "
                f"{'S/B':>11} {'Z':>11} {'Z/Z0':>11} {'sig eff':>11} {'bkg eff':>11}"
            ),
            "-" * 132,
        ]
    )
    for row in payload["recommendations"]:
        signal_eff_key = (
            "hcc_pure_efficiency"
            if "hcc_pure_efficiency" in row
            else "ttHcc_event_efficiency"
        )
        lines.append(
            f"{row['label']:<31} {row['x_cut']:8.4f} {row['y_cut']:8.4f} "
            f"{row['S']:12.5g} {row['B']:12.5g} {row['S_over_B']:11.5g} "
            f"{row['S_over_sqrt_S_plus_B']:11.5g} {row['relative_significance']:11.5g} "
            f"{_format_percent(row[signal_eff_key]):>11} "
            f"{_format_percent(row['total_background_efficiency']):>11}"
        )
    lines.extend(
        [
            "",
            "Constrained points are diagnostics only; the analysis owner chooses the final WP.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_hcc_wp_outputs(
    result: dict[str, Any],
    outdirs: dict[str, Path],
    scan_config: dict[str, Any],
    candidate_selection: dict[str, Any],
    current_references: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    is_event = result["level"] == "event"
    stem = "hcc_wp_scan_event_level" if is_event else "hcc_wp_scan"
    payload = _summary_payload(
        result,
        scan_config,
        candidate_selection,
        current_references,
    )
    write_csv(outdirs["tables"] / f"{stem}.csv", _scan_rows(result))
    write_json(outdirs["summaries"] / f"{stem}.json", payload)
    (outdirs["summaries"] / f"{stem}.txt").write_text(
        _summary_text(payload),
        encoding="utf-8",
    )
    if not is_event:
        write_csv(
            outdirs["tables"] / "hcc_wp_recommendations.csv",
            result["recommendations"],
        )
    return payload

