from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tthcc_an.nn_study.config import NnStudyConfig
from tthcc_an.nn_study.dataset import NnDataset
from tthcc_an.nn_study.plotting import plot_score_significance
from tthcc_an.nn_study.reporting import write_json, write_text


METRIC_NAME = "s_over_sqrt_s_plus_b"


def _significance(signal: np.ndarray, background: np.ndarray) -> np.ndarray:
    total = signal + background
    return np.divide(
        signal,
        np.sqrt(total),
        out=np.zeros_like(signal, dtype=np.float64),
        where=total > 0.0,
    )


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))


def calculate_score_significance_scan(
    *,
    score: np.ndarray,
    weights: np.ndarray,
    truth_index: np.ndarray,
    truth_names: list[str],
    signals: list[str],
    thresholds: np.ndarray,
    direction: str,
) -> dict[str, Any]:
    if direction not in {"<", ">"}:
        raise ValueError("Score-significance direction must be '<' or '>'.")
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if (
        thresholds.ndim != 1
        or thresholds.size < 2
        or not np.all(np.isfinite(thresholds))
        or np.any(np.diff(thresholds) <= 0.0)
    ):
        raise ValueError(
            "Score-significance thresholds must be finite, unique, and ascending."
        )

    score = np.asarray(score, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    truth_index = np.asarray(truth_index, dtype=np.int64)
    if not (score.shape == weights.shape == truth_index.shape):
        raise ValueError("Score-significance input arrays must have identical shapes.")
    index_by_truth = {name: index for index, name in enumerate(truth_names)}
    unknown_signals = sorted(set(signals) - set(index_by_truth))
    if unknown_signals:
        raise ValueError(
            "Unknown score-significance signals: " + ", ".join(unknown_signals)
        )

    valid_weight = np.isfinite(weights) & (weights >= 0.0)
    eligible = valid_weight & np.isfinite(score)
    if not np.any(valid_weight):
        raise ValueError("Score-significance scan has no finite non-negative weights.")
    if not np.any(eligible):
        raise ValueError("Score-significance scan has no finite score values.")

    order = np.argsort(score[eligible], kind="mergesort")
    sorted_score = score[eligible][order]
    sorted_weight = weights[eligible][order]
    sorted_truth = truth_index[eligible][order]
    total_weight_prefix = _prefix(sorted_weight)
    total_count_prefix = np.arange(sorted_score.size + 1, dtype=np.int64)
    if direction == "<":
        split_indices = np.searchsorted(sorted_score, thresholds, side="left")
        total_yield_after = total_weight_prefix[split_indices]
        total_count_after = total_count_prefix[split_indices]
    else:
        split_indices = np.searchsorted(sorted_score, thresholds, side="right")
        total_yield_after = total_weight_prefix[-1] - total_weight_prefix[split_indices]
        total_count_after = total_count_prefix[-1] - total_count_prefix[split_indices]

    signal_results: dict[str, dict[str, Any]] = {}
    complement_valid = True
    nonnegative_valid = True
    finite_valid = True
    monotonic_valid = True
    for signal_name in signals:
        is_signal_sorted = sorted_truth == index_by_truth[signal_name]
        signal_weight_prefix = _prefix(
            np.where(is_signal_sorted, sorted_weight, 0.0)
        )
        signal_count_prefix = np.concatenate(
            ([0], np.cumsum(is_signal_sorted, dtype=np.int64))
        )
        if direction == "<":
            signal_yield = signal_weight_prefix[split_indices]
            signal_count = signal_count_prefix[split_indices]
        else:
            signal_yield = (
                signal_weight_prefix[-1] - signal_weight_prefix[split_indices]
            )
            signal_count = (
                signal_count_prefix[-1] - signal_count_prefix[split_indices]
            )
        background_yield = np.maximum(total_yield_after - signal_yield, 0.0)
        background_count = total_count_after - signal_count
        significance = _significance(signal_yield, background_yield)

        baseline_signal_mask = (
            valid_weight & (truth_index == index_by_truth[signal_name])
        )
        baseline_signal = float(np.sum(weights[baseline_signal_mask]))
        baseline_background = float(
            np.sum(weights[valid_weight & ~baseline_signal_mask])
        )
        baseline_total = baseline_signal + baseline_background
        baseline_significance = (
            baseline_signal / np.sqrt(baseline_total)
            if baseline_total > 0.0
            else 0.0
        )
        relative = (
            significance / baseline_significance
            if baseline_significance > 0.0
            else np.zeros_like(significance)
        )
        best_index = int(np.argmax(significance))

        complement_valid &= bool(
            np.allclose(
                signal_yield + background_yield,
                total_yield_after,
                rtol=1e-12,
                atol=1e-12,
            )
            and np.array_equal(signal_count + background_count, total_count_after)
        )
        nonnegative_valid &= bool(
            np.all(signal_yield >= 0.0)
            and np.all(background_yield >= 0.0)
            and np.all(significance >= 0.0)
        )
        finite_valid &= bool(
            np.all(np.isfinite(signal_yield))
            and np.all(np.isfinite(background_yield))
            and np.all(np.isfinite(significance))
            and np.all(np.isfinite(relative))
        )
        expected_diff_sign = 1.0 if direction == "<" else -1.0
        tolerance = 1e-12 * max(float(np.max(total_yield_after)), 1.0)
        monotonic_valid &= bool(
            np.all(expected_diff_sign * np.diff(signal_yield) >= -tolerance)
            and np.all(
                expected_diff_sign * np.diff(background_yield) >= -tolerance
            )
        )
        points = [
            {
                "cut": float(threshold),
                "signal_weighted_yield": float(signal_value),
                "background_weighted_yield": float(background_value),
                "significance": float(z_value),
                "relative_significance": float(relative_value),
                "signal_event_count": int(signal_events),
                "background_event_count": int(background_events),
            }
            for threshold, signal_value, background_value, z_value,
            relative_value, signal_events, background_events in zip(
                thresholds,
                signal_yield,
                background_yield,
                significance,
                relative,
                signal_count,
                background_count,
            )
        ]
        signal_results[signal_name] = {
            "signal_definition": f"truth category == {signal_name}",
            "background_definition": (
                f"all other selected MC events (truth category != {signal_name})"
            ),
            "baseline_signal_weighted_yield": baseline_signal,
            "baseline_background_weighted_yield": baseline_background,
            "baseline_significance": float(baseline_significance),
            "baseline_signal_event_count": int(np.sum(baseline_signal_mask)),
            "baseline_background_event_count": int(
                np.sum(valid_weight & ~baseline_signal_mask)
            ),
            "best_threshold": float(thresholds[best_index]),
            "best_significance": float(significance[best_index]),
            "best_scan_index": best_index,
            "points": points,
        }

    other_signals_in_background = True
    for signal_name, result in signal_results.items():
        background_yield = np.asarray(
            [point["background_weighted_yield"] for point in result["points"]],
            dtype=np.float64,
        )
        background_count = np.asarray(
            [point["background_event_count"] for point in result["points"]],
            dtype=np.int64,
        )
        for other_name in signals:
            if other_name == signal_name:
                continue
            is_other = sorted_truth == index_by_truth[other_name]
            other_weight_prefix = _prefix(np.where(is_other, sorted_weight, 0.0))
            other_count_prefix = np.concatenate(
                ([0], np.cumsum(is_other, dtype=np.int64))
            )
            if direction == "<":
                other_yield = other_weight_prefix[split_indices]
                other_count = other_count_prefix[split_indices]
            else:
                other_yield = (
                    other_weight_prefix[-1] - other_weight_prefix[split_indices]
                )
                other_count = (
                    other_count_prefix[-1] - other_count_prefix[split_indices]
                )
            tolerance = 1e-12 * max(float(np.max(background_yield)), 1.0)
            other_signals_in_background &= bool(
                np.all(other_yield <= background_yield + tolerance)
                and np.all(other_count <= background_count)
            )

    if (
        not complement_valid
        or not nonnegative_valid
        or not finite_valid
        or not monotonic_valid
        or not other_signals_in_background
    ):
        raise ValueError("Score-significance scan failed numerical validation.")
    return {
        "metric": METRIC_NAME,
        "metric_definition": "Z = S / sqrt(S + B)",
        "direction": direction,
        "thresholds": thresholds.tolist(),
        "weighting": "sample_norm * abs(raw event weight)",
        "signal_semantics": "One configured NN-study truth category.",
        "background_semantics": (
            "Every other selected MC event, including all other signal and "
            "background truth categories."
        ),
        "additional_mass_window": False,
        "baseline_semantics": (
            "Full selected MC population before any score cut; finite non-negative "
            "analysis weight required, but no finite-score requirement."
        ),
        "scan_population": (
            "Full selected MC population with finite score and finite non-negative "
            "analysis weight."
        ),
        "selected_event_count_for_baseline": int(np.sum(valid_weight)),
        "selected_event_count_for_scan": int(np.sum(eligible)),
        "nonfinite_score_count": int(np.sum(valid_weight & ~np.isfinite(score))),
        "signals": signal_results,
        "validation": {
            "thresholds_strictly_ascending": True,
            "signal_background_are_complements": complement_valid,
            "nonnegative_yields_and_significance": nonnegative_valid,
            "finite_scan_metrics": finite_valid,
            "weighted_yields_monotonic_for_direction": monotonic_valid,
            "background_includes_other_requested_signals": (
                other_signals_in_background
            ),
            "baseline_uses_full_uncut_population": True,
        },
    }


def build_significance_mass_window_selection(
    *,
    config: NnStudyConfig,
    dataset: NnDataset,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    settings = config.significance_mass_window
    if not settings.enabled:
        return None
    if (
        settings.branch is None
        or settings.minimum is None
        or settings.maximum is None
    ):
        raise ValueError("Enabled significance mass window is incomplete.")
    if settings.branch not in dataset.analysis_columns:
        raise ValueError(
            f"Significance mass-window branch '{settings.branch}' was not loaded."
        )
    mass = np.asarray(dataset.analysis_columns[settings.branch], dtype=np.float64)
    if mass.shape != dataset.truth_index.shape:
        raise ValueError(
            "Significance mass branch and event arrays must have identical shapes."
        )
    finite = np.isfinite(mass)
    selected = finite & (mass >= settings.minimum) & (mass <= settings.maximum)
    metadata = {
        "enabled": True,
        "branch": settings.branch,
        "range": [settings.minimum, settings.maximum],
        "boundary_semantics": (
            f"{settings.minimum:g} <= {settings.branch} <= {settings.maximum:g}"
        ),
        "lower_bound_inclusive": True,
        "upper_bound_inclusive": True,
        "total_event_count": int(mass.size),
        "finite_mass_event_count": int(np.sum(finite)),
        "nonfinite_mass_event_count": int(np.sum(~finite)),
        "selected_event_count": int(np.sum(selected)),
    }
    return selected, metadata


def mass_window_suffix(metadata: dict[str, Any]) -> str:
    def token(value: float) -> str:
        return f"{value:g}".replace("-", "m").replace(".", "p")

    minimum, maximum = metadata["range"]
    return f"mass_window_{token(minimum)}_{token(maximum)}"


def mark_mass_window_result(
    result: dict[str, Any], metadata: dict[str, Any]
) -> None:
    result["additional_mass_window"] = True
    result["mass_window"] = metadata
    result["baseline_semantics"] = (
        "Mass-window selected MC population before any score cut; finite "
        "non-negative analysis weight required, but no finite-score requirement."
    )
    result["scan_population"] = (
        "Mass-window selected MC population with finite score and finite "
        "non-negative analysis weight."
    )
    for signal_name, signal_result in result["signals"].items():
        signal_result["signal_definition"] = (
            f"mass-window selected truth category == {signal_name}"
        )
        signal_result["background_definition"] = (
            "all other mass-window selected MC events "
            f"(truth category != {signal_name})"
        )


def _format_summary(payload: dict[str, Any]) -> str:
    mass_window = payload["mass_window"]
    lines = [
        "=== NN Score Significance Scans ===",
        f"Channel: {payload['channel']}",
        "Metric: Z = S / sqrt(S + B)",
        "Weight: sample_norm * abs(weight)",
        "Inclusive scans: yes",
        "Additional mass-window scans: "
        + (
            mass_window["boundary_semantics"]
            if mass_window is not None
            else "no"
        ),
        "Signal: exactly one configured NN-study truth category",
        "Background: every other selected MC event",
    ]
    scan_groups = [("Inclusive", payload["scans"])]
    if payload["mass_window_scans"]:
        scan_groups.append(("Mass window", payload["mass_window_scans"]))
    for group_label, scans in scan_groups:
        lines.extend(["", f"--- {group_label} ---"])
        for scan in scans:
            lines.extend(
                [
                    "",
                    f"Score: {scan['score_branch']}",
                    f"Cut: keep {scan['score_branch']} {scan['direction']} cut",
                    f"Range/points: {scan['scan_min']:g} to {scan['scan_max']:g} / "
                    f"{scan['scan_points']}",
                ]
            )
            for signal_name, result in scan["signals"].items():
                lines.append(
                    f"  {signal_name}: baseline Z={result['baseline_significance']:.9g}; "
                    f"scan maximum cut={result['best_threshold']:.9g}, "
                    f"Z={result['best_significance']:.9g}"
                )
    return "\n".join(lines) + "\n"


def run_score_significance(
    *,
    config: NnStudyConfig,
    dataset: NnDataset,
    plot_dir: Path,
    summary_dir: Path,
    plot_suffix: str,
) -> tuple[list[str], dict[str, Any]]:
    settings = config.score_significance
    truths_by_name = {truth.name: truth for truth in config.truth_categories}
    scan_payloads: list[dict[str, Any]] = []
    mass_window_scan_payloads: list[dict[str, Any]] = []
    plots: list[str] = []
    significance_plot_dir = plot_dir / "score_significance"
    mass_window_selection = build_significance_mass_window_selection(
        config=config,
        dataset=dataset,
    )
    for scan in settings.scans:
        thresholds = np.linspace(
            scan.scan_min,
            scan.scan_max,
            scan.scan_points,
            dtype=np.float64,
        )
        result = calculate_score_significance_scan(
            score=dataset.scores[scan.score_name],
            weights=dataset.analysis_weight,
            truth_index=dataset.truth_index,
            truth_names=config.truth_names,
            signals=settings.signals,
            thresholds=thresholds,
            direction=scan.direction,
        )
        result.update(
            {
                "score_name": scan.score_name,
                "score_branch": scan.score_branch,
                "scan_min": scan.scan_min,
                "scan_max": scan.scan_max,
                "scan_points": scan.scan_points,
            }
        )
        path = significance_plot_dir / (
            f"{scan.score_branch}__significance_s_over_sqrt_s_plus_b{plot_suffix}"
        )
        plot_score_significance(
            outpath=path,
            thresholds=thresholds,
            signal_results=result["signals"],
            signal_styles={
                name: {
                    "label": truths_by_name[name].label,
                    "color": truths_by_name[name].color,
                }
                for name in settings.signals
            },
            score_branch=scan.score_branch,
            direction=scan.direction,
            xscale="linear",
        )
        result["plot"] = str(path)
        scan_payloads.append(result)
        plots.append(str(path))

        if mass_window_selection is not None:
            mass_mask, mass_window_metadata = mass_window_selection
            mass_result = calculate_score_significance_scan(
                score=dataset.scores[scan.score_name][mass_mask],
                weights=dataset.analysis_weight[mass_mask],
                truth_index=dataset.truth_index[mass_mask],
                truth_names=config.truth_names,
                signals=settings.signals,
                thresholds=thresholds,
                direction=scan.direction,
            )
            mark_mass_window_result(mass_result, mass_window_metadata)
            mass_result.update(
                {
                    "score_name": scan.score_name,
                    "score_branch": scan.score_branch,
                    "scan_min": scan.scan_min,
                    "scan_max": scan.scan_max,
                    "scan_points": scan.scan_points,
                }
            )
            mass_path = significance_plot_dir / (
                f"{scan.score_branch}__significance_s_over_sqrt_s_plus_b__"
                f"{mass_window_suffix(mass_window_metadata)}{plot_suffix}"
            )
            plot_score_significance(
                outpath=mass_path,
                thresholds=thresholds,
                signal_results=mass_result["signals"],
                signal_styles={
                    name: {
                        "label": truths_by_name[name].label,
                        "color": truths_by_name[name].color,
                    }
                    for name in settings.signals
                },
                score_branch=scan.score_branch,
                direction=scan.direction,
                xscale="linear",
                selection_label=mass_window_metadata["boundary_semantics"],
            )
            mass_result["plot"] = str(mass_path)
            mass_window_scan_payloads.append(mass_result)
            plots.append(str(mass_path))

    payload: dict[str, Any] = {
        "schema_version": 2,
        "channel": config.channel,
        "metric": settings.metric,
        "signals": settings.signals,
        "weighting": "sample_norm * abs(raw event weight)",
        "additional_mass_window": mass_window_selection is not None,
        "mass_window": (
            None if mass_window_selection is None else mass_window_selection[1]
        ),
        "scans": scan_payloads,
        "mass_window_scans": mass_window_scan_payloads,
        "plots": plots,
    }
    write_json(summary_dir / "score_significance.json", payload)
    write_text(summary_dir / "score_significance.txt", _format_summary(payload))
    return plots, payload
