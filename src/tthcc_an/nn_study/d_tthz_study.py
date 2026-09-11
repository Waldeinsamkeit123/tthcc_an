from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tthcc_an.nn_study.config import NnStudyConfig
from tthcc_an.nn_study.dataset import NnDataset
from tthcc_an.nn_study.metrics import pairwise_discriminator, weighted_roc
from tthcc_an.nn_study.plotting import (
    plot_score_distribution,
    plot_score_significance_comparison,
)
from tthcc_an.nn_study.reporting import write_json, write_text


def validate_persisted_d_tthz(dataset: NnDataset) -> dict[str, Any]:
    required = ("D_ttHZ", "ttHZ", "ttjets", "qcd")
    missing = [name for name in required if name not in dataset.scores]
    if missing:
        raise KeyError("Missing D_ttHZ validation scores: " + ", ".join(missing))
    persisted = np.asarray(dataset.scores["D_ttHZ"], dtype=np.float64)
    score_tthz = np.asarray(dataset.scores["ttHZ"], dtype=np.float64)
    score_ttjets = np.asarray(dataset.scores["ttjets"], dtype=np.float64)
    denominator = score_tthz + score_ttjets
    comparable = (
        np.isfinite(persisted)
        & np.isfinite(score_tthz)
        & np.isfinite(score_ttjets)
        & (denominator > 0.0)
    )
    differences = np.abs(
        persisted[comparable]
        - score_tthz[comparable] / denominator[comparable]
    )
    nonfinite = int(np.sum(~np.isfinite(persisted)))
    out_of_range = int(np.sum(np.isfinite(persisted) & ((persisted < 0) | (persisted > 1))))
    if nonfinite or out_of_range or not np.any(comparable):
        raise ValueError("Persisted D_ttHZ failed finite/range/comparability validation.")
    return {
        "persisted_branch": "D_ttHZ",
        "derived_definition": "score_ttHZ / (score_ttHZ + score_ttjets)",
        "selected_event_count": int(persisted.size),
        "finite_event_count": int(np.sum(np.isfinite(persisted))),
        "nonfinite_event_count": nonfinite,
        "out_of_range_event_count": out_of_range,
        "minimum": float(np.min(persisted)),
        "maximum": float(np.max(persisted)),
        "positive_denominator_event_count": int(np.sum(comparable)),
        "zero_or_invalid_denominator_event_count": int(persisted.size - np.sum(comparable)),
        "max_absolute_difference": float(np.max(differences)),
        "mean_absolute_difference": float(np.mean(differences)),
    }


def _remap(index: np.ndarray, source_names: list[str], selected: list[str]) -> np.ndarray:
    result = np.full(index.shape, -1, dtype=np.int16)
    source = {name: number for number, name in enumerate(source_names)}
    for number, name in enumerate(selected):
        result[index == source[name]] = number
    return result


def _distribution_payload(
    *,
    values: np.ndarray,
    truth_index: np.ndarray,
    truth_names: list[str],
    weights: np.ndarray,
    bins: np.ndarray,
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for number, name in enumerate(truth_names):
        mask = (
            (truth_index == number)
            & np.isfinite(values)
            & np.isfinite(weights)
            & (weights > 0.0)
        )
        hist, _ = np.histogram(values[mask], bins=bins, weights=weights[mask])
        total = float(np.sum(hist))
        groups[name] = {
            "event_count": int(np.sum(mask)),
            "weighted_yield": total,
            "bin_yields": [float(value) for value in hist],
            "normalized_bin_fractions": [
                float(value / total) if total > 0.0 else 0.0 for value in hist
            ],
            "normalized_integral": 1.0 if total > 0.0 else 0.0,
        }
    return {"bin_edges": [float(value) for value in bins], "groups": groups}


def _find_scan(payload: dict[str, Any] | None, score_name: str, key: str) -> dict[str, Any] | None:
    if payload is None:
        return None
    for scan in payload.get(key, []):
        if scan["score_name"] == score_name:
            return scan
    return None


def _plateaus(signal: dict[str, Any]) -> dict[str, Any]:
    points = signal["points"]
    maximum = float(signal["best_significance"])
    output: dict[str, Any] = {}
    for fraction in (0.95, 0.98):
        selected = [point for point in points if point["significance"] >= fraction * maximum]
        output[f"{int(fraction * 100)}_percent"] = {
            "fraction_of_maximum": fraction,
            "minimum_significance": fraction * maximum,
            "point_count": len(selected),
            "cut_range": (
                [float(selected[0]["cut"]), float(selected[-1]["cut"])]
                if selected
                else None
            ),
            "points": selected,
        }
    return output


def _one_dimensional_comparison(
    score_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, label in (("scans", "inclusive"), ("mass_window_scans", "mass_window")):
        raw = _find_scan(score_summary, "ttHZ", key)
        discriminant = _find_scan(score_summary, "D_ttHZ", key)
        if raw is None or discriminant is None:
            output[label] = None
            continue
        comparisons: dict[str, Any] = {}
        for signal_name, d_signal in discriminant["signals"].items():
            raw_signal = raw["signals"][signal_name]
            raw_z = np.asarray([p["significance"] for p in raw_signal["points"]])
            d_z = np.asarray([p["significance"] for p in d_signal["points"]])
            best = d_signal["points"][int(d_signal["best_scan_index"])]
            comparisons[signal_name] = {
                "raw_best_threshold": raw_signal["best_threshold"],
                "raw_best_significance": raw_signal["best_significance"],
                "D_best_threshold": d_signal["best_threshold"],
                "D_best_significance": d_signal["best_significance"],
                "D_best_signal_efficiency": (
                    best["signal_weighted_yield"]
                    / d_signal["baseline_signal_weighted_yield"]
                    if d_signal["baseline_signal_weighted_yield"] > 0.0
                    else 0.0
                ),
                "max_absolute_significance_difference_on_common_cut_grid": float(
                    np.max(np.abs(d_z - raw_z))
                ),
                "D_plateaus": _plateaus(d_signal),
            }
        output[label] = {"raw": raw, "D": discriminant, "signals": comparisons}
    return output


def _roc_validation(config: NnStudyConfig, dataset: NnDataset) -> dict[str, Any]:
    names = config.prediction_truth_names
    signal_index = names.index("ttHZ")
    background_index = names.index("ttjets")
    mask = (dataset.prediction_truth_index == signal_index) | (
        dataset.prediction_truth_index == background_index
    )
    labels = dataset.prediction_truth_index[mask] == signal_index
    raw, zero_count = pairwise_discriminator(
        dataset.scores["ttHZ"][mask], dataset.scores["ttjets"][mask]
    )
    persisted = np.asarray(dataset.scores["D_ttHZ"][mask], dtype=np.float64)
    raw_tpr, raw_fpr, _, raw_auc = weighted_roc(raw, labels, dataset.analysis_weight[mask])
    d_tpr, d_fpr, _, d_auc = weighted_roc(
        persisted, labels, dataset.analysis_weight[mask]
    )
    same_shape = raw_tpr.shape == d_tpr.shape and raw_fpr.shape == d_fpr.shape
    tolerance = 1e-6
    eventwise_agreement = np.abs(raw - persisted) <= tolerance
    canonical = 0.5 * (raw + persisted)
    raw_coalesced = np.where(eventwise_agreement, canonical, raw)
    d_coalesced = np.where(eventwise_agreement, canonical, persisted)
    raw_coalesced_tpr, raw_coalesced_fpr, _, raw_coalesced_auc = weighted_roc(
        raw_coalesced, labels, dataset.analysis_weight[mask]
    )
    d_coalesced_tpr, d_coalesced_fpr, _, d_coalesced_auc = weighted_roc(
        d_coalesced, labels, dataset.analysis_weight[mask]
    )
    coalesced_same_shape = (
        raw_coalesced_tpr.shape == d_coalesced_tpr.shape
        and raw_coalesced_fpr.shape == d_coalesced_fpr.shape
    )
    return {
        "population": "coarse ttHZ truth vs coarse ttjets truth",
        "weighting": "sample_norm * abs(weight)",
        "raw_discriminator": "score_ttHZ / (score_ttHZ + score_ttjets)",
        "persisted_discriminator": "D_ttHZ",
        "zero_denominator_event_count": zero_count,
        "raw_auc": raw_auc,
        "persisted_D_auc": d_auc,
        "auc_absolute_difference": abs(raw_auc - d_auc),
        "roc_array_shapes_identical": same_shape,
        "signal_efficiency_max_absolute_difference": (
            float(np.max(np.abs(raw_tpr - d_tpr))) if same_shape else None
        ),
        "background_efficiency_max_absolute_difference": (
            float(np.max(np.abs(raw_fpr - d_fpr))) if same_shape else None
        ),
        "coalescing_tolerance": tolerance,
        "all_eventwise_discriminators_agree_within_tolerance": bool(
            np.all(eventwise_agreement)
        ),
        "coalescing_reason": (
            "After validating eventwise agreement, pairs within tolerance use their "
            "shared midpoint. This prevents float32 persistence from artificially "
            "splitting physically identical ROC thresholds."
        ),
        "coalesced_raw_auc": raw_coalesced_auc,
        "coalesced_persisted_D_auc": d_coalesced_auc,
        "coalesced_auc_absolute_difference": abs(
            raw_coalesced_auc - d_coalesced_auc
        ),
        "coalesced_roc_array_shapes_identical": coalesced_same_shape,
        "coalesced_signal_efficiency_max_absolute_difference": (
            float(np.max(np.abs(raw_coalesced_tpr - d_coalesced_tpr)))
            if coalesced_same_shape
            else None
        ),
        "coalesced_background_efficiency_max_absolute_difference": (
            float(np.max(np.abs(raw_coalesced_fpr - d_coalesced_fpr)))
            if coalesced_same_shape
            else None
        ),
    }


def _two_dimensional_comparison(
    raw: dict[str, Any] | None, hierarchical: dict[str, Any] | None
) -> dict[str, Any] | None:
    if raw is None or hierarchical is None:
        return None
    raw_x = raw["axes"]["x"]["thresholds"]
    raw_y = raw["axes"]["y"]["thresholds"]
    d_x = hierarchical["axes"]["x"]["thresholds"]
    d_y = hierarchical["axes"]["y"]["thresholds"]
    common_x = [value for value in raw_x if value in d_x]
    common_y = [value for value in raw_y if value in d_y]
    differences: dict[str, float] = {}
    for signal in hierarchical["signals"]:
        raw_matrix = np.asarray(raw["signals"][signal]["significance_matrix"])
        d_matrix = np.asarray(hierarchical["signals"][signal]["significance_matrix"])
        values = [
            abs(raw_matrix[raw_x.index(x), raw_y.index(y)] - d_matrix[d_x.index(x), d_y.index(y)])
            for x in common_x
            for y in common_y
        ]
        differences[signal] = float(max(values, default=0.0))
    return {
        "raw_axes": raw["axes"],
        "hierarchical_axes": hierarchical["axes"],
        "common_grid": {"x_thresholds": common_x, "y_thresholds": common_y},
        "max_absolute_significance_difference_on_common_grid": differences,
        "raw_best_grid_points": {
            name: entry["best_grid_point"] for name, entry in raw["signals"].items()
        },
        "hierarchical_best_grid_points": {
            name: entry["best_grid_point"]
            for name, entry in hierarchical["signals"].items()
        },
    }


def _format_summary(payload: dict[str, Any]) -> str:
    validation = payload["branch_validation"]
    lines = [
        "=== Persisted D_ttHZ Study ===",
        f"Branch: {validation['persisted_branch']}",
        f"Range: [{validation['minimum']:.8g}, {validation['maximum']:.8g}]",
        f"Nonfinite / out of range: {validation['nonfinite_event_count']} / {validation['out_of_range_event_count']}",
        f"Derived max / mean abs diff: {validation['max_absolute_difference']:.8g} / {validation['mean_absolute_difference']:.8g}",
        "No final D_ttHZ working point is selected.",
    ]
    roc = payload["pairwise_roc_validation"]
    lines.append(
        f"ROC AUC raw / D: {roc['raw_auc']:.8g} / {roc['persisted_D_auc']:.8g}"
    )
    for region, comparison in payload["one_dimensional_comparison"].items():
        if comparison is None:
            continue
        lines.append(f"{region} 1D significance:")
        for signal, values in comparison["signals"].items():
            lines.append(
                f"  {signal}: D best cut={values['D_best_threshold']:.6g}, "
                f"Z={values['D_best_significance']:.8g}, "
                f"eff={values['D_best_signal_efficiency']:.6g}"
            )
    return "\n".join(lines) + "\n"


def run_d_tthz_study(
    *,
    config: NnStudyConfig,
    dataset: NnDataset,
    plot_dir: Path,
    summary_dir: Path,
    plot_suffix: str,
    score_significance_summary: dict[str, Any] | None,
    score_significance_2d_summary: dict[str, Any] | None,
    hierarchical_summary: dict[str, Any] | None,
    mass_sculpting_summary: dict[str, Any] | None,
) -> tuple[list[str], dict[str, Any]]:
    score = next(item for item in config.auxiliary_scores if item.name == "D_ttHZ")
    plots: list[str] = []
    fine_dir = plot_dir / "score_fine_truth"
    for normalize, suffix in ((True, ""), (False, "__yield")):
        path = fine_dir / f"D_ttHZ__fine_truth{suffix}{plot_suffix}"
        plot_score_distribution(
            outpath=path,
            score=score,
            values=dataset.scores["D_ttHZ"],
            truth_index=dataset.truth_index,
            weights=dataset.analysis_weight,
            truths=config.truth_categories,
            config=config,
            normalize=normalize,
            figsize=(7.2, 5.2),
            legend_columns=3,
            legend_fontsize=8.0,
        )
        plots.append(str(path))

    top_names = ["ttHZ", "ttjets"]
    top_truths = [
        next(item for item in config.prediction_truth_categories if item.name == name)
        for name in top_names
    ]
    top_index = _remap(dataset.prediction_truth_index, config.prediction_truth_names, top_names)
    composition_dir = plot_dir / "score_composition"
    for normalize, suffix in ((True, ""), (False, "__yield")):
        path = composition_dir / f"D_ttHZ__ttHZ_vs_ttjets{suffix}{plot_suffix}"
        plot_score_distribution(
            outpath=path,
            score=score,
            values=dataset.scores["D_ttHZ"],
            truth_index=top_index,
            weights=dataset.analysis_weight,
            truths=top_truths,
            config=config,
            normalize=normalize,
        )
        plots.append(str(path))

    comparison = _one_dimensional_comparison(score_significance_summary)
    truth_by_name = {truth.name: truth for truth in config.truth_categories}
    for signal_name in ("ttHcc", "ttHbb"):
        inclusive = comparison.get("inclusive")
        if inclusive is not None:
            path = plot_dir / "score_significance" / (
                f"D_ttHZ_vs_score_ttHZ__{signal_name}{plot_suffix}"
            )
            plot_score_significance_comparison(
                outpath=path,
                raw_result=inclusive["raw"]["signals"][signal_name],
                discriminant_result=inclusive["D"]["signals"][signal_name],
                signal_label=truth_by_name[signal_name].label,
            )
            plots.append(str(path))

    n_bins = int(config.plot_options.get("score_bins", 20))
    score_range = config.plot_options.get("score_range", [0.0, 1.0])
    bins = np.linspace(float(score_range[0]), float(score_range[1]), n_bins + 1)
    top_distribution = _distribution_payload(
        values=dataset.scores["D_ttHZ"],
        truth_index=top_index,
        truth_names=top_names,
        weights=dataset.analysis_weight,
        bins=bins,
    )
    fine_distribution = _distribution_payload(
        values=dataset.scores["D_ttHZ"],
        truth_index=dataset.truth_index,
        truth_names=config.truth_names,
        weights=dataset.analysis_weight,
        bins=bins,
    )
    mass_scans = [] if mass_sculpting_summary is None else [
        scan for scan in mass_sculpting_summary.get("scans", [])
        if scan["score_name"] == "D_ttHZ"
    ]
    payload = {
        "schema_version": 1,
        "channel": config.channel,
        "weighting": "sample_norm * abs(raw event weight)",
        "branch_validation": validate_persisted_d_tthz(dataset),
        "fine_truth_distribution": fine_distribution,
        "top_only_distribution": top_distribution,
        "pairwise_roc_validation": _roc_validation(config, dataset),
        "one_dimensional_comparison": comparison,
        "hierarchical_score_significance_2d": hierarchical_summary,
        "raw_vs_hierarchical_2d_comparison": _two_dimensional_comparison(
            score_significance_2d_summary, hierarchical_summary
        ),
        "mass_sculpting": {
            "thresholds": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            "population_count": len(mass_scans),
            "scans": mass_scans,
        },
        "candidate_operating_points": (
            [] if hierarchical_summary is None else hierarchical_summary["representative_points"]
        ),
        "interpretation": (
            "Candidate points are diagnostics for balancing signal efficiency, background yield, "
            "and QCD/tt+jets rejection. No final D_ttHZ threshold is selected."
        ),
        "plots": plots,
    }
    write_json(summary_dir / "D_ttHZ_study.json", payload)
    write_text(summary_dir / "D_ttHZ_study.txt", _format_summary(payload))
    return plots, payload
