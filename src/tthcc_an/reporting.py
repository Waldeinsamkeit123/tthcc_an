from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from tthcc_an.definitions import SCORE_LABELS, TARGET_DEFINITIONS, TRUTH_LABEL_ORDER


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
