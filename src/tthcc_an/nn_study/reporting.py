from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def format_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "=== Event-level NN Study ===",
        f"Channel: {summary['channel']}",
        f"Input: {summary['input_location']}",
        f"Selection: {summary['selection'] or 'all events in the produced ntuple'}",
        f"Files: {summary['number_of_files_processed']} processed, "
        f"{summary['number_of_files_missing_tree']} without Events",
        f"Events: {summary['number_of_events_read']} read, "
        f"{summary['number_of_events_selected']} selected, "
        f"{summary['number_of_events_classified']} classified",
        "Weighting: analysis_weight = lumi_fb * xsec / gen_sumw * abs(weight)",
        "ROC weighting: analysis_weight; AUC is integral of signal efficiency over background efficiency",
        "",
        "Truth-category yields:",
    ]
    for name, entry in summary["truth_yields"].items():
        lines.append(
            f"  {name:8s} N={entry['events']:8d}  "
            f"yield={entry['analysis_weight_sum']:.8g}  "
            f"signed={entry['signed_weight_sum']:.8g}"
        )
    validation = summary["auc_validation"]
    lines.extend(
        [
            "",
            "AUC validation:",
            f"  sklearn available: {validation['sklearn_available']}",
            f"  compared pairs: {validation['compared_pairs']}",
            f"  maximum absolute difference: {validation['max_abs_difference']}",
            "",
            "Pairwise AUC:",
        ]
    )
    for signal, backgrounds in summary["pairwise_auc"].items():
        values = ", ".join(f"{background}={value:.5f}" for background, value in backgrounds.items())
        lines.append(f"  {signal}: {values or 'no available background classes'}")
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

