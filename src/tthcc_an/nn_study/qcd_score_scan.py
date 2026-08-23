from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tthcc_an.nn_study.config import NnQcdScoreGroup, NnStudyConfig
from tthcc_an.nn_study.dataset import NnDataset
from tthcc_an.nn_study.plotting import (
    plot_qcd_efficiencies,
    plot_qcd_expected_yields,
    plot_qcd_relative_significance,
    plot_qcd_roc_like,
    plot_qcd_score_distribution,
    plot_qcd_to_ttx_ratio,
    plot_qcd_working_point_distribution,
)
from tthcc_an.nn_study.reporting import write_json, write_text


def _group_mask(
    config: NnStudyConfig, dataset: NnDataset, group: NnQcdScoreGroup
) -> np.ndarray:
    index_by_name = {
        truth.name: index for index, truth in enumerate(config.truth_categories)
    }
    return np.isin(
        dataset.truth_index,
        [index_by_name[name] for name in group.truth_categories],
    )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return None
    value = numerator / denominator
    return float(value) if np.isfinite(value) else None


def _significance(signal: float, background: float) -> float | None:
    if signal < 0.0 or background <= 0.0:
        return None
    value = signal / np.sqrt(background)
    return float(value) if np.isfinite(value) else None


def _as_plot_array(values: list[float | None]) -> np.ndarray:
    return np.asarray([np.nan if value is None else value for value in values])


def _nondecreasing(values: list[float]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    tolerance = 1e-12 * max(float(np.max(np.abs(array))), 1.0)
    return bool(np.all(np.diff(array) >= -tolerance))


def _format_summary(payload: dict[str, Any]) -> str:
    lines = [
        "=== 0L QCD Score Scan ===",
        f"Channel: {payload['channel']}",
        f"Selection: {payload['selection'] or '<none after upstream Pepper selection>'}",
        f"Cut: {payload['score_branch']} {payload['direction']} threshold",
        "Weight: sample_norm * abs(weight)",
        "",
        "Groups:",
    ]
    for group in payload["groups"]:
        lines.append(
            f"  {group['name']}: {', '.join(group['truth_categories'])}"
        )
    working_point = payload["working_point_distribution"]
    lines.extend(
        [
            "",
            "Working-point distribution:",
            f"  score range: {working_point['score_range'][0]:g} to "
            f"{working_point['score_range'][1]:g}",
            f"  logarithmic bins: {working_point['bins']}",
            f"  reference threshold: {working_point['reference_threshold']:g}",
            f"  rejected side: {working_point['rejected_side']}",
        ]
    )
    for group in working_point["groups"]:
        integral = group["normalized_histogram_integral"]
        integral_text = "empty" if integral is None else f"{integral:.9g}"
        lines.append(
            f"  {group['label']}: {', '.join(group['truth_categories'])}; "
            f"normalized integral={integral_text}"
        )
    score = payload["score_diagnostics"]
    lines.extend(
        [
            "",
            "Score diagnostics:",
            f"  finite range: {score['finite_min']:.9g} to {score['finite_max']:.9g}",
            f"  finite/nonfinite/nonpositive: {score['finite_count']}/"
            f"{score['nonfinite_count']}/{score['nonpositive_count']}",
            "",
            "Candidate working points:",
            "cut         ttHcc yield before/after  ttHcc eff   "
            "QCD yield before/after    QCD eff   QCD rej   "
            "tt+X after   QCD/tt+X   N(ttHcc)  N(QCD)",
        ]
    )
    for row in payload["candidate_working_points"]:
        ratio = "n/a" if row["qcd_to_ttX_after_cut"] is None else f"{row['qcd_to_ttX_after_cut']:.5g}"
        lines.append(
            f"{row['cut']:<11.5g} "
            f"{row['ttHcc_weighted_yield_before_cut']:.6g}/"
            f"{row['ttHcc_weighted_yield_after_cut']:<11.6g} "
            f"{row['ttHcc_efficiency']:<10.5g} "
            f"{row['qcd_weighted_yield_before_cut']:.6g}/"
            f"{row['qcd_weighted_yield_after_cut']:<11.6g} "
            f"{row['qcd_efficiency']:<9.5g} "
            f"{row['qcd_rejection']:<9.5g} "
            f"{row['ttX_weighted_yield_after_cut']:<11.6g} "
            f"{ratio:<10s} "
            f"{row['ttHcc_event_count_after_cut']:<9d} "
            f"{row['qcd_event_count_after_cut']:<7d}"
        )
    validation = payload["validation"]
    lines.extend(
        [
            "",
            "Validation:",
            f"  candidate thresholds exact: {validation['candidate_thresholds_exact']}",
            f"  event counts monotonic: {validation['event_counts_monotonic']}",
            f"  weighted yields monotonic: {validation['weighted_yields_monotonic']}",
            f"  efficiencies in [0, 1]: {validation['efficiencies_in_unit_interval']}",
            f"  finite JSON metrics: {validation['no_nonfinite_serialized_metrics']}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_qcd_score_scan(
    *,
    config: NnStudyConfig,
    dataset: NnDataset,
    plot_dir: Path,
    summary_dir: Path,
    plot_suffix: str,
) -> tuple[list[str], dict[str, Any]]:
    settings = config.qcd_score_scan
    score = np.asarray(dataset.scores[settings.score_name], dtype=np.float64)
    weights = np.asarray(dataset.analysis_weight, dtype=np.float64)
    finite_score = np.isfinite(score)
    finite_weight = np.isfinite(weights) & (weights >= 0.0)
    eligible = finite_score & finite_weight
    if not np.any(eligible):
        raise ValueError("QCD-score scan has no finite-score, finite-weight events.")

    thresholds = np.unique(
        np.concatenate(
            [
                np.logspace(
                    np.log10(settings.scan_min),
                    np.log10(settings.scan_max),
                    settings.scan_points,
                ),
                np.asarray(settings.candidate_thresholds, dtype=np.float64),
            ]
        )
    )
    group_masks = {
        group.name: _group_mask(config, dataset, group) & eligible
        for group in settings.groups
    }
    group_results: dict[str, dict[str, Any]] = {}
    for group in settings.groups:
        base = group_masks[group.name]
        before_yield = float(np.sum(weights[base]))
        before_count = int(np.sum(base))
        if before_yield <= 0.0:
            raise ValueError(
                f"QCD-score scan group '{group.name}' has zero weighted yield."
            )
        after_yields: list[float] = []
        after_counts: list[int] = []
        efficiencies: list[float] = []
        for threshold in thresholds:
            selected = base & (score < threshold)
            after_yield = float(np.sum(weights[selected]))
            efficiency = after_yield / before_yield
            if not -1e-12 <= efficiency <= 1.0 + 1e-12:
                raise ValueError(
                    f"QCD-score efficiency outside [0, 1] for group '{group.name}'."
                )
            after_yields.append(after_yield)
            after_counts.append(int(np.sum(selected)))
            efficiencies.append(float(np.clip(efficiency, 0.0, 1.0)))
        group_results[group.name] = {
            "weighted_yield_before_cut": before_yield,
            "event_count_before_cut": before_count,
            "weighted_yield_after_cut": after_yields,
            "event_count_after_cut": after_counts,
            "weighted_efficiency": efficiencies,
        }

    tthcc_yields = group_results["ttHcc"]["weighted_yield_after_cut"]
    qcd_yields = group_results["qcd"]["weighted_yield_after_cut"]
    ttx_yields = group_results["ttX"]["weighted_yield_after_cut"]
    qcd_to_ttx = [
        _safe_ratio(qcd_yield, ttx_yield)
        for qcd_yield, ttx_yield in zip(qcd_yields, ttx_yields)
    ]
    significance_qcd = [
        _significance(signal, background)
        for signal, background in zip(tthcc_yields, qcd_yields)
    ]
    significance_qcd_ttx = [
        _significance(signal, qcd + ttx)
        for signal, qcd, ttx in zip(tthcc_yields, qcd_yields, ttx_yields)
    ]
    baseline_qcd = _significance(
        group_results["ttHcc"]["weighted_yield_before_cut"],
        group_results["qcd"]["weighted_yield_before_cut"],
    )
    baseline_qcd_ttx = _significance(
        group_results["ttHcc"]["weighted_yield_before_cut"],
        group_results["qcd"]["weighted_yield_before_cut"]
        + group_results["ttX"]["weighted_yield_before_cut"],
    )
    relative_qcd = [
        _safe_ratio(value, baseline_qcd) if value is not None and baseline_qcd is not None else None
        for value in significance_qcd
    ]
    relative_qcd_ttx = [
        _safe_ratio(value, baseline_qcd_ttx)
        if value is not None and baseline_qcd_ttx is not None
        else None
        for value in significance_qcd_ttx
    ]

    candidate_rows: list[dict[str, Any]] = []
    candidate_indices: list[int] = []
    for cut in settings.candidate_thresholds:
        matches = np.flatnonzero(thresholds == cut)
        if len(matches) != 1:
            raise ValueError(f"Candidate threshold {cut:g} is not exact in the scan grid.")
        index = int(matches[0])
        candidate_indices.append(index)
        tthcc_eff = group_results["ttHcc"]["weighted_efficiency"][index]
        qcd_eff = group_results["qcd"]["weighted_efficiency"][index]
        candidate_rows.append(
            {
                "cut": cut,
                "ttHcc_weighted_yield_before_cut": group_results["ttHcc"]["weighted_yield_before_cut"],
                "ttHcc_weighted_yield_after_cut": tthcc_yields[index],
                "ttHcc_efficiency": tthcc_eff,
                "qcd_weighted_yield_before_cut": group_results["qcd"]["weighted_yield_before_cut"],
                "qcd_weighted_yield_after_cut": qcd_yields[index],
                "qcd_efficiency": qcd_eff,
                "qcd_rejection": 1.0 - qcd_eff,
                "ttX_weighted_yield_after_cut": ttx_yields[index],
                "qcd_to_ttX_after_cut": qcd_to_ttx[index],
                "ttHcc_event_count_after_cut": group_results["ttHcc"]["event_count_after_cut"][index],
                "qcd_event_count_after_cut": group_results["qcd"]["event_count_after_cut"][index],
                "significance_s_over_sqrt_qcd": significance_qcd[index],
                "significance_s_over_sqrt_qcd_plus_ttX": significance_qcd_ttx[index],
                "relative_significance_qcd": relative_qcd[index],
                "relative_significance_qcd_plus_ttX": relative_qcd_ttx[index],
            }
        )

    log_bins = np.linspace(
        settings.distribution_log10_range[0],
        settings.distribution_log10_range[1],
        settings.distribution_bins + 1,
    )
    histograms: dict[str, np.ndarray] = {}
    for name in ("ttHcc", "qcd", "ttX"):
        positive = group_masks[name] & (score > 0.0)
        histograms[name] = np.histogram(
            np.log10(score[positive]), bins=log_bins, weights=weights[positive]
        )[0].astype(np.float64)

    working_point_bins = np.geomspace(
        settings.working_point_score_range[0],
        settings.working_point_score_range[1],
        settings.working_point_bins + 1,
    )
    working_point_yields: dict[str, np.ndarray] = {}
    working_point_shapes: dict[str, np.ndarray] = {}
    normalized_integrals: dict[str, float | None] = {}
    for group in settings.distribution_groups:
        mask = _group_mask(config, dataset, group) & eligible
        histogram = np.histogram(
            score[mask], bins=working_point_bins, weights=weights[mask]
        )[0].astype(np.float64)
        integral = float(np.sum(histogram))
        working_point_yields[group.name] = histogram
        if integral > 0.0:
            shape = histogram / integral
            working_point_shapes[group.name] = shape
            normalized_integrals[group.name] = float(np.sum(shape))
        else:
            working_point_shapes[group.name] = np.zeros_like(histogram)
            normalized_integrals[group.name] = None

    qcd_plot_dir = plot_dir / "qcd_score_scan"
    plot_paths = {
        "score_distribution": qcd_plot_dir / f"log10_score_qcd{plot_suffix}",
        "expected_yields": qcd_plot_dir / f"expected_yield_vs_cut{plot_suffix}",
        "efficiencies": qcd_plot_dir / f"efficiency_vs_cut{plot_suffix}",
        "roc_like": qcd_plot_dir / f"qcd_efficiency_vs_tthcc_efficiency{plot_suffix}",
        "qcd_to_ttX": qcd_plot_dir / f"qcd_to_ttx_ratio_vs_cut{plot_suffix}",
        "relative_significance": qcd_plot_dir / f"relative_significance_improvement_vs_cut{plot_suffix}",
        "working_point_distribution": qcd_plot_dir / f"qcd_score_distribution{plot_suffix}",
        "working_point_distribution_yield": qcd_plot_dir / f"qcd_score_distribution__yield{plot_suffix}",
    }
    plot_qcd_score_distribution(
        outpath=plot_paths["score_distribution"],
        bins=log_bins,
        histograms=histograms,
        groups=settings.groups,
    )
    plot_qcd_expected_yields(
        outpath=plot_paths["expected_yields"],
        thresholds=thresholds,
        yields={name: np.asarray(group_results[name]["weighted_yield_after_cut"]) for name in ("ttHcc", "qcd", "ttX")},
        groups=settings.groups,
    )
    tthcc_efficiency = np.asarray(group_results["ttHcc"]["weighted_efficiency"])
    qcd_efficiency = np.asarray(group_results["qcd"]["weighted_efficiency"])
    plot_qcd_efficiencies(
        outpath=plot_paths["efficiencies"],
        thresholds=thresholds,
        tthcc_efficiency=tthcc_efficiency,
        qcd_efficiency=qcd_efficiency,
        groups=settings.groups,
    )
    plot_qcd_roc_like(
        outpath=plot_paths["roc_like"],
        tthcc_efficiency=tthcc_efficiency,
        qcd_efficiency=qcd_efficiency,
        candidate_points=candidate_rows,
    )
    plot_qcd_to_ttx_ratio(
        outpath=plot_paths["qcd_to_ttX"],
        thresholds=thresholds,
        ratio=_as_plot_array(qcd_to_ttx),
    )
    plot_qcd_relative_significance(
        outpath=plot_paths["relative_significance"],
        thresholds=thresholds,
        qcd_only=_as_plot_array(relative_qcd),
        qcd_plus_ttx=_as_plot_array(relative_qcd_ttx),
    )
    plot_qcd_working_point_distribution(
        outpath=plot_paths["working_point_distribution"],
        bins=working_point_bins,
        histograms=working_point_shapes,
        groups=settings.distribution_groups,
        reference_threshold=settings.reference_threshold,
        normalize=True,
    )
    plot_qcd_working_point_distribution(
        outpath=plot_paths["working_point_distribution_yield"],
        bins=working_point_bins,
        histograms=working_point_yields,
        groups=settings.distribution_groups,
        reference_threshold=settings.reference_threshold,
        normalize=False,
    )

    weighted_monotonic = all(
        _nondecreasing(group_results[name]["weighted_yield_after_cut"])
        for name in group_results
    )
    count_monotonic = all(
        np.all(np.diff(group_results[name]["event_count_after_cut"]) >= 0)
        for name in group_results
    )
    efficiencies_valid = all(
        np.all(
            (np.asarray(group_results[name]["weighted_efficiency"]) >= 0.0)
            & (np.asarray(group_results[name]["weighted_efficiency"]) <= 1.0)
        )
        for name in group_results
    )
    normalized_shapes_valid = all(
        integral is None or np.isclose(integral, 1.0, rtol=0.0, atol=1e-12)
        for integral in normalized_integrals.values()
    )
    if not weighted_monotonic or not count_monotonic or not efficiencies_valid:
        raise ValueError("QCD-score scan failed monotonicity or efficiency validation.")

    finite_values = score[finite_score]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "channel": config.channel,
        "score_name": settings.score_name,
        "score_branch": settings.score_branch,
        "direction": settings.direction,
        "selection": config.selection,
        "event_population": (
            "Configured MC samples after per-sample stitching and the NN-study selection; "
            "no additional QCD-scan physics selection. Scan metrics require finite score_qcd "
            "and finite non-negative analysis weight."
        ),
        "weighting": "sample_norm * abs(raw event weight)",
        "data_comparison": {
            "included": False,
            "reason": "Data normalization and trigger compatibility are not configured; this expected-yield scan is MC-only.",
        },
        "groups": [
            {
                "name": group.name,
                "label": group.label,
                "truth_categories": group.truth_categories,
            }
            for group in settings.groups
        ],
        "score_diagnostics": {
            "total_count": int(score.size),
            "finite_count": int(np.sum(finite_score)),
            "nonfinite_count": int(np.sum(~finite_score)),
            "nonpositive_count": int(np.sum(finite_score & (score <= 0.0))),
            "finite_min": float(np.min(finite_values)),
            "finite_max": float(np.max(finite_values)),
            "log10_distribution_range": list(settings.distribution_log10_range),
            "log10_distribution_bins": settings.distribution_bins,
        },
        "working_point_distribution": {
            "score_range": list(settings.working_point_score_range),
            "bins": settings.working_point_bins,
            "bin_edges": working_point_bins.tolist(),
            "binning": "Logarithmic edges in linear score_qcd.",
            "normalization": (
                "Each non-empty displayed group is independently normalized to unit "
                "weighted integral over the plotted score range."
            ),
            "weighting": "sample_norm * abs(raw event weight)",
            "reference_threshold": settings.reference_threshold,
            "reference_threshold_semantics": (
                "Visualization setting only; not a selected final working point."
            ),
            "kept_side": "score_qcd < reference_threshold",
            "rejected_side": "score_qcd > reference_threshold",
            "groups": [
                {
                    "name": group.name,
                    "label": group.label,
                    "truth_categories": group.truth_categories,
                    "weighted_yield_in_plot_range": float(
                        np.sum(working_point_yields[group.name])
                    ),
                    "normalized_histogram_integral": normalized_integrals[group.name],
                }
                for group in settings.distribution_groups
            ],
        },
        "candidate_thresholds": settings.candidate_thresholds,
        "scan": {
            "configured_min": settings.scan_min,
            "configured_max": settings.scan_max,
            "configured_log_points": settings.scan_points,
            "thresholds": thresholds.tolist(),
            "groups": group_results,
            "qcd_to_ttX": qcd_to_ttx,
            "significance": {
                "definition": "S/sqrt(B), with S=ttHcc; diagnostic only.",
                "baseline_no_cut_qcd": baseline_qcd,
                "baseline_no_cut_qcd_plus_ttX": baseline_qcd_ttx,
                "qcd": significance_qcd,
                "qcd_plus_ttX": significance_qcd_ttx,
                "relative_qcd": relative_qcd,
                "relative_qcd_plus_ttX": relative_qcd_ttx,
            },
        },
        "candidate_working_points": candidate_rows,
        "validation": {
            "candidate_thresholds_exact": [thresholds[index] for index in candidate_indices] == settings.candidate_thresholds,
            "event_counts_monotonic": count_monotonic,
            "weighted_yields_monotonic": weighted_monotonic,
            "efficiencies_in_unit_interval": efficiencies_valid,
            "working_point_bin_edges_positive": bool(
                np.all(working_point_bins > 0.0)
            ),
            "working_point_histograms_normalized": normalized_shapes_valid,
            "working_point_rejected_side_is_high_score": True,
            "no_nonfinite_serialized_metrics": True,
        },
        "plots": {name: str(path) for name, path in plot_paths.items()},
    }
    json_path = summary_dir / "qcd_score_scan.json"
    text_path = summary_dir / "qcd_score_scan.txt"
    write_json(json_path, payload)
    write_text(text_path, _format_summary(payload))
    return [str(path) for path in plot_paths.values()], payload
