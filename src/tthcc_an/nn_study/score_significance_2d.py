from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tthcc_an.nn_study.config import (
    NnScoreSignificance2DAxis,
    NnScoreSignificance2DConfig,
    NnStudyConfig,
)
from tthcc_an.nn_study.dataset import NnDataset
from tthcc_an.nn_study.plotting import plot_score_significance_2d_heatmap
from tthcc_an.nn_study.reporting import write_json, write_text


def _cut_ranges(
    score: np.ndarray,
    axis: NnScoreSignificance2DAxis,
) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.asarray(axis.thresholds, dtype=np.float64)
    if axis.direction == ">":
        return (
            np.zeros(score.size, dtype=np.int64),
            np.searchsorted(thresholds, score, side="left"),
        )
    return (
        np.searchsorted(thresholds, score, side="right"),
        np.full(score.size, thresholds.size, dtype=np.int64),
    )


def _category_rectangle_yields(
    *,
    score_x: np.ndarray,
    score_y: np.ndarray,
    weights: np.ndarray,
    truth_index: np.ndarray,
    number_of_truths: int,
    x_axis: NnScoreSignificance2DAxis,
    y_axis: NnScoreSignificance2DAxis,
) -> tuple[np.ndarray, np.ndarray]:
    valid_weight = np.isfinite(weights) & (weights >= 0.0)
    eligible = valid_weight & np.isfinite(score_x) & np.isfinite(score_y)
    if not np.any(valid_weight):
        raise ValueError("2D significance scan has no finite non-negative weights.")
    if not np.any(eligible):
        raise ValueError("2D significance scan has no events with two finite scores.")

    selected_truth = truth_index[eligible].astype(np.int64, copy=False)
    category = np.where(
        (selected_truth >= 0) & (selected_truth < number_of_truths),
        selected_truth + 1,
        0,
    )
    selected_weight = weights[eligible]
    x_lo, x_hi = _cut_ranges(score_x[eligible], x_axis)
    y_lo, y_hi = _cut_ranges(score_y[eligible], y_axis)
    nx = len(x_axis.thresholds)
    ny = len(y_axis.thresholds)
    shape = (number_of_truths + 1, nx + 1, ny + 1)
    difference = np.zeros(shape, dtype=np.float64)
    flat_size = int(np.prod(shape))

    def update(x_index: np.ndarray, y_index: np.ndarray, sign: float) -> None:
        flat_index = (category * (nx + 1) + x_index) * (ny + 1) + y_index
        difference.flat[:] += sign * np.bincount(
            flat_index,
            weights=selected_weight,
            minlength=flat_size,
        )

    update(x_lo, y_lo, 1.0)
    update(x_hi, y_lo, -1.0)
    update(x_lo, y_hi, -1.0)
    update(x_hi, y_hi, 1.0)
    yields = np.cumsum(
        np.cumsum(difference, axis=1, dtype=np.float64),
        axis=2,
        dtype=np.float64,
    )[:, :nx, :ny]
    tolerance = 1e-11 * max(float(np.sum(weights[valid_weight])), 1.0)
    if float(np.min(yields)) < -tolerance:
        raise ValueError("2D significance rectangle accumulation produced negative yields.")
    return np.maximum(yields, 0.0), eligible


def _matrix(values: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in values]


def _safe_efficiency(numerator: np.ndarray, denominator: float) -> np.ndarray:
    if denominator <= 0.0:
        return np.zeros_like(numerator, dtype=np.float64)
    return numerator / denominator


def _significance(signal: np.ndarray, background: np.ndarray) -> np.ndarray:
    total = signal + background
    return np.divide(
        signal,
        np.sqrt(total),
        out=np.zeros_like(signal, dtype=np.float64),
        where=total > 0.0,
    )


def _plateau(
    values: np.ndarray,
    x_thresholds: list[float],
    y_thresholds: list[float],
    fraction: float,
) -> dict[str, Any]:
    maximum = float(np.max(values))
    selected = values >= fraction * maximum
    x_indices, y_indices = np.nonzero(selected)
    points = [
        {
            "x_threshold": float(x_thresholds[x_index]),
            "y_threshold": float(y_thresholds[y_index]),
            "significance": float(values[x_index, y_index]),
        }
        for x_index, y_index in zip(x_indices, y_indices)
    ]
    return {
        "fraction_of_maximum": fraction,
        "minimum_significance": fraction * maximum,
        "point_count": len(points),
        "x_threshold_range": [
            float(x_thresholds[int(np.min(x_indices))]),
            float(x_thresholds[int(np.max(x_indices))]),
        ],
        "y_threshold_range": [
            float(y_thresholds[int(np.min(y_indices))]),
            float(y_thresholds[int(np.max(y_indices))]),
        ],
        "points": points,
    }


def _monotonic_for_direction(
    values: np.ndarray,
    *,
    axis: int,
    direction: str,
) -> bool:
    differences = np.diff(values, axis=axis)
    tolerance = 1e-11 * max(float(np.max(values)), 1.0)
    if direction == ">":
        return bool(np.all(differences <= tolerance))
    return bool(np.all(differences >= -tolerance))


def _point_indices(
    point: tuple[float, float],
    x_thresholds: list[float],
    y_thresholds: list[float],
) -> tuple[int, int]:
    return x_thresholds.index(point[0]), y_thresholds.index(point[1])


def _group_vector(
    category_yields: np.ndarray,
    category_indices: list[int],
) -> np.ndarray:
    return np.sum(category_yields[np.asarray(category_indices) + 1], axis=0)


def _third_cut_diagnostic(
    *,
    settings: NnScoreSignificance2DConfig,
    dataset: NnDataset,
    truth_index_by_name: dict[str, int],
    group_indices: dict[str, list[int]],
    group_baselines: dict[str, float],
    valid_weight: np.ndarray,
) -> dict[str, Any] | None:
    third = settings.third_cut
    if not third.enabled:
        return None
    if (
        settings.x_axis is None
        or settings.y_axis is None
        or third.score_name is None
        or third.score_branch is None
        or third.direction is None
    ):
        raise ValueError("Enabled third-score diagnostic configuration is incomplete.")

    score_x = np.asarray(dataset.scores[settings.x_axis.score_name], dtype=np.float64)
    score_y = np.asarray(dataset.scores[settings.y_axis.score_name], dtype=np.float64)
    score_third = np.asarray(dataset.scores[third.score_name], dtype=np.float64)
    weights = np.asarray(dataset.analysis_weight, dtype=np.float64)
    truth_index = np.asarray(dataset.truth_index, dtype=np.int64)
    thresholds = np.asarray(third.thresholds, dtype=np.float64)
    number_of_categories = len(truth_index_by_name) + 1
    results: list[dict[str, Any]] = []

    for x_threshold, y_threshold in third.base_points:
        x_keep = (
            score_x > x_threshold
            if settings.x_axis.direction == ">"
            else score_x < x_threshold
        )
        y_keep = (
            score_y > y_threshold
            if settings.y_axis.direction == ">"
            else score_y < y_threshold
        )
        base_mask = (
            valid_weight
            & np.isfinite(score_x)
            & np.isfinite(score_y)
            & x_keep
            & y_keep
        )
        category = np.where(
            (truth_index >= 0) & (truth_index < len(truth_index_by_name)),
            truth_index + 1,
            0,
        )
        base_category_yields = np.bincount(
            category[base_mask],
            weights=weights[base_mask],
            minlength=number_of_categories,
        )

        eligible = base_mask & np.isfinite(score_third)
        selected_category = category[eligible]
        lo, hi = _cut_ranges(
            score_third[eligible],
            NnScoreSignificance2DAxis(
                score_name=third.score_name,
                score_branch=third.score_branch,
                direction=third.direction,
                thresholds=third.thresholds,
            ),
        )
        difference = np.zeros(
            (number_of_categories, thresholds.size + 1), dtype=np.float64
        )
        flat_size = difference.size
        start = selected_category * (thresholds.size + 1) + lo
        stop = selected_category * (thresholds.size + 1) + hi
        difference.flat[:] += np.bincount(
            start, weights=weights[eligible], minlength=flat_size
        )
        difference.flat[:] -= np.bincount(
            stop, weights=weights[eligible], minlength=flat_size
        )
        cut_category_yields = np.cumsum(
            difference, axis=1, dtype=np.float64
        )[:, : thresholds.size]
        cut_category_yields = np.maximum(cut_category_yields, 0.0)

        rows: list[dict[str, Any]] = []
        all_category_yields = [
            (None, base_category_yields),
            *[
                (float(threshold), cut_category_yields[:, index])
                for index, threshold in enumerate(thresholds)
            ],
        ]
        base_total = float(np.sum(base_category_yields))
        base_significance: dict[str, float] = {}
        for signal_name in settings.signals:
            signal_yield = float(
                base_category_yields[truth_index_by_name[signal_name] + 1]
            )
            base_significance[signal_name] = float(
                _significance(
                    np.asarray(signal_yield),
                    np.asarray(max(base_total - signal_yield, 0.0)),
                )
            )

        for threshold, category_yields in all_category_yields:
            total_yield = float(np.sum(category_yields))
            group_efficiencies = {
                name: (
                    float(
                        np.sum(category_yields[np.asarray(indices) + 1])
                        / group_baselines[name]
                    )
                    if group_baselines[name] > 0.0
                    else 0.0
                )
                for name, indices in group_indices.items()
            }
            signal_metrics: dict[str, Any] = {}
            for signal_name in settings.signals:
                signal_yield = float(
                    category_yields[truth_index_by_name[signal_name] + 1]
                )
                background_yield = max(total_yield - signal_yield, 0.0)
                z_value = float(
                    _significance(
                        np.asarray(signal_yield), np.asarray(background_yield)
                    )
                )
                base_z = base_significance[signal_name]
                base_efficiency = (
                    float(
                        base_category_yields[
                            truth_index_by_name[signal_name] + 1
                        ]
                        / group_baselines[signal_name]
                    )
                    if group_baselines[signal_name] > 0.0
                    else 0.0
                )
                efficiency = group_efficiencies[signal_name]
                relative_significance = (
                    z_value / base_z if base_z > 0.0 else 0.0
                )
                relative_efficiency = (
                    efficiency / base_efficiency
                    if base_efficiency > 0.0
                    else 0.0
                )
                if abs(relative_significance - 1.0) < 1e-12:
                    relative_significance = 1.0
                if abs(relative_efficiency - 1.0) < 1e-12:
                    relative_efficiency = 1.0
                signal_metrics[signal_name] = {
                    "signal_weighted_yield": signal_yield,
                    "background_weighted_yield": background_yield,
                    "significance": z_value,
                    "relative_significance_to_no_third_cut": relative_significance,
                    "signal_efficiency": efficiency,
                    "relative_signal_efficiency_to_no_third_cut": relative_efficiency,
                    "signal_efficiency_loss": max(1.0 - relative_efficiency, 0.0),
                }
            rows.append(
                {
                    "third_threshold": threshold,
                    "selection": (
                        "no additional third-score cut"
                        if threshold is None
                        else f"{third.score_branch} {third.direction} {threshold:g}"
                    ),
                    "total_selected_weighted_yield": total_yield,
                    "group_efficiencies": group_efficiencies,
                    "signals": signal_metrics,
                }
            )

        best_by_signal: dict[str, Any] = {}
        for signal_name in settings.signals:
            best_row = rows[0]
            for row in rows[1:]:
                current = row["signals"][signal_name]["significance"]
                best = best_row["signals"][signal_name]["significance"]
                if current > best * (1.0 + 1e-12):
                    best_row = row
            best_by_signal[signal_name] = {
                "third_threshold": best_row["third_threshold"],
                **best_row["signals"][signal_name],
            }
        results.append(
            {
                "x_threshold": x_threshold,
                "y_threshold": y_threshold,
                "rows": rows,
                "best_including_no_third_cut": best_by_signal,
            }
        )

    return {
        "score_name": third.score_name,
        "score_branch": third.score_branch,
        "direction": third.direction,
        "thresholds": third.thresholds,
        "base_points": [
            {"x": point[0], "y": point[1]} for point in third.base_points
        ],
        "comparison_baseline": (
            "The same two-score selection without an explicit third-score cut."
        ),
        "score_constraint_note": (
            "The three native scores sum approximately to one, so only two score "
            "dimensions are independent."
        ),
        "points": results,
    }


def _format_summary(payload: dict[str, Any]) -> str:
    x_axis = payload["axes"]["x"]
    y_axis = payload["axes"]["y"]
    lines = [
        "=== NN Two-Dimensional Score Significance Scan ===",
        f"Channel: {payload['channel']}",
        "Metric: Z = S / sqrt(S + B)",
        "Weight: sample_norm * abs(weight)",
        f"Selection: {x_axis['score_branch']} {x_axis['direction']} A AND "
        f"{y_axis['score_branch']} {y_axis['direction']} B",
        f"A grid: {x_axis['thresholds']}",
        f"B grid: {y_axis['thresholds']}",
        "Signal: one fine truth category; background: every other selected MC event",
    ]
    for signal_name, result in payload["signals"].items():
        best = result["best_grid_point"]
        lines.extend(
            [
                "",
                f"{signal_name}: best A={best['x_threshold']:g}, "
                f"B={best['y_threshold']:g}, Z={best['significance']:.9g}",
            ]
        )
        for key, plateau in result["plateaus"].items():
            lines.append(
                f"  {key}: A={plateau['x_threshold_range']}, "
                f"B={plateau['y_threshold_range']}, N={plateau['point_count']}"
            )
    lines.extend(["", "Representative working points:"])
    for row in payload["representative_points"]:
        lines.append(
            f"  A={row['x_threshold']:g}, B={row['y_threshold']:g}: "
            f"eff(ttHcc/ttHbb/ttZ/ttjets/qcd)="
            f"{row['group_efficiencies'].get('ttHcc', 0.0):.5g}/"
            f"{row['group_efficiencies'].get('ttHbb', 0.0):.5g}/"
            f"{row['group_efficiencies'].get('ttZ', 0.0):.5g}/"
            f"{row['group_efficiencies'].get('ttjets', 0.0):.5g}/"
            f"{row['group_efficiencies'].get('qcd', 0.0):.5g}; "
            f"Z(ttHcc/ttHbb)="
            f"{row['signals'].get('ttHcc', {}).get('significance', 0.0):.6g}/"
            f"{row['signals'].get('ttHbb', {}).get('significance', 0.0):.6g}"
        )
    third = payload["third_score_diagnostic"]
    if third is not None:
        lines.extend(["", "Third-score diagnostic:"])
        for point in third["points"]:
            best_text = ", ".join(
                f"{signal}: C={entry['third_threshold']}, "
                f"Z ratio={entry['relative_significance_to_no_third_cut']:.6g}, "
                f"signal loss={entry['signal_efficiency_loss']:.6g}"
                for signal, entry in point["best_including_no_third_cut"].items()
            )
            lines.append(
                f"  A={point['x_threshold']:g}, B={point['y_threshold']:g}: "
                + best_text
            )
    return "\n".join(lines) + "\n"


def run_score_significance_2d(
    *,
    config: NnStudyConfig,
    dataset: NnDataset,
    plot_dir: Path,
    summary_dir: Path,
    plot_suffix: str,
    settings: NnScoreSignificance2DConfig | None = None,
    summary_stem: str = "score_significance_2d",
) -> tuple[list[str], dict[str, Any]]:
    settings = config.score_significance_2d if settings is None else settings
    if not settings.enabled or settings.x_axis is None or settings.y_axis is None:
        raise ValueError("score_significance_2d is not enabled or is incomplete.")

    x_axis = settings.x_axis
    y_axis = settings.y_axis
    score_x = np.asarray(dataset.scores[x_axis.score_name], dtype=np.float64)
    score_y = np.asarray(dataset.scores[y_axis.score_name], dtype=np.float64)
    weights = np.asarray(dataset.analysis_weight, dtype=np.float64)
    truth_index = np.asarray(dataset.truth_index, dtype=np.int64)
    valid_weight = np.isfinite(weights) & (weights >= 0.0)
    category_yields, eligible = _category_rectangle_yields(
        score_x=score_x,
        score_y=score_y,
        weights=weights,
        truth_index=truth_index,
        number_of_truths=len(config.truth_categories),
        x_axis=x_axis,
        y_axis=y_axis,
    )
    total_yields = np.sum(category_yields, axis=0)
    truth_index_by_name = {
        truth.name: index for index, truth in enumerate(config.truth_categories)
    }
    group_indices = {
        group.name: [
            truth_index_by_name[name] for name in group.truth_categories
        ]
        for group in settings.groups
    }
    group_baselines = {
        name: float(
            np.sum(weights[valid_weight & np.isin(truth_index, indices)])
        )
        for name, indices in group_indices.items()
    }
    group_yields = {
        name: _group_vector(category_yields, indices)
        for name, indices in group_indices.items()
    }
    group_efficiencies = {
        name: _safe_efficiency(values, group_baselines[name])
        for name, values in group_yields.items()
    }
    total_baseline = float(np.sum(weights[valid_weight]))

    signal_payloads: dict[str, Any] = {}
    signal_arrays: dict[str, dict[str, np.ndarray]] = {}
    for signal_name in settings.signals:
        signal_index = truth_index_by_name[signal_name]
        signal_yields = category_yields[signal_index + 1]
        background_yields = np.maximum(total_yields - signal_yields, 0.0)
        significance = _significance(signal_yields, background_yields)
        baseline_signal = float(
            np.sum(weights[valid_weight & (truth_index == signal_index)])
        )
        baseline_background = max(total_baseline - baseline_signal, 0.0)
        signal_efficiency = _safe_efficiency(signal_yields, baseline_signal)
        background_efficiency = _safe_efficiency(
            background_yields, baseline_background
        )
        best_flat = int(np.argmax(significance))
        best_x, best_y = np.unravel_index(best_flat, significance.shape)
        plateaus = {
            f"{int(round(fraction * 100))}_percent": _plateau(
                significance,
                x_axis.thresholds,
                y_axis.thresholds,
                fraction,
            )
            for fraction in settings.plateau_fractions
        }
        signal_payloads[signal_name] = {
            "signal_definition": f"fine truth category == {signal_name}",
            "background_definition": (
                f"every other selected MC event (fine truth category != {signal_name})"
            ),
            "baseline_signal_weighted_yield": baseline_signal,
            "baseline_background_weighted_yield": baseline_background,
            "signal_weighted_yield_matrix": _matrix(signal_yields),
            "background_weighted_yield_matrix": _matrix(background_yields),
            "significance_matrix": _matrix(significance),
            "signal_efficiency_matrix": _matrix(signal_efficiency),
            "background_efficiency_matrix": _matrix(background_efficiency),
            "best_grid_point": {
                "x_threshold": x_axis.thresholds[best_x],
                "y_threshold": y_axis.thresholds[best_y],
                "signal_weighted_yield": float(signal_yields[best_x, best_y]),
                "background_weighted_yield": float(
                    background_yields[best_x, best_y]
                ),
                "signal_efficiency": float(signal_efficiency[best_x, best_y]),
                "background_efficiency": float(
                    background_efficiency[best_x, best_y]
                ),
                "significance": float(significance[best_x, best_y]),
            },
            "plateaus": plateaus,
        }
        signal_arrays[signal_name] = {
            "signal": signal_yields,
            "background": background_yields,
            "significance": significance,
            "signal_efficiency": signal_efficiency,
            "background_efficiency": background_efficiency,
        }

    representative_rows: list[dict[str, Any]] = []
    for point in settings.representative_points:
        x_index, y_index = _point_indices(
            point, x_axis.thresholds, y_axis.thresholds
        )
        representative_rows.append(
            {
                "x_threshold": point[0],
                "y_threshold": point[1],
                "total_selected_weighted_yield": float(
                    total_yields[x_index, y_index]
                ),
                "group_efficiencies": {
                    name: float(values[x_index, y_index])
                    for name, values in group_efficiencies.items()
                },
                "signals": {
                    signal_name: {
                        "background_weighted_yield": float(
                            arrays["background"][x_index, y_index]
                        ),
                        "signal_weighted_yield": float(
                            arrays["signal"][x_index, y_index]
                        ),
                        "signal_efficiency": float(
                            arrays["signal_efficiency"][x_index, y_index]
                        ),
                        "background_efficiency": float(
                            arrays["background_efficiency"][x_index, y_index]
                        ),
                        "significance": float(
                            arrays["significance"][x_index, y_index]
                        ),
                    }
                    for signal_name, arrays in signal_arrays.items()
                },
            }
        )

    third_diagnostic = _third_cut_diagnostic(
        settings=settings,
        dataset=dataset,
        truth_index_by_name=truth_index_by_name,
        group_indices=group_indices,
        group_baselines=group_baselines,
        valid_weight=valid_weight,
    )
    validation = {
        "all_significance_values_finite_nonnegative": all(
            bool(
                np.all(np.isfinite(arrays["significance"]))
                and np.all(arrays["significance"] >= 0.0)
            )
            for arrays in signal_arrays.values()
        ),
        "signal_background_are_complements": all(
            bool(
                np.allclose(
                    arrays["signal"] + arrays["background"],
                    total_yields,
                    rtol=1e-11,
                    atol=1e-11,
                )
            )
            for arrays in signal_arrays.values()
        ),
        "x_threshold_selection_is_nested": _monotonic_for_direction(
            total_yields, axis=0, direction=x_axis.direction
        ),
        "y_threshold_selection_is_nested": _monotonic_for_direction(
            total_yields, axis=1, direction=y_axis.direction
        ),
        "background_includes_other_requested_signals": all(
            bool(
                np.all(
                    signal_arrays[other]["signal"]
                    <= signal_arrays[signal]["background"]
                    + 1e-11 * np.maximum(total_yields, 1.0)
                )
            )
            for signal in settings.signals
            for other in settings.signals
            if signal != other
        ),
        "baseline_uses_full_uncut_population": True,
        "scan_uses_two_finite_scores": True,
    }
    if not all(validation.values()):
        raise ValueError("Two-dimensional significance scan failed validation.")

    output_dir = plot_dir / "score_significance_2d"
    stem = (
        f"{x_axis.score_branch}_{'gt' if x_axis.direction == '>' else 'lt'}__"
        f"{y_axis.score_branch}_{'gt' if y_axis.direction == '>' else 'lt'}"
    )
    plots: list[str] = []
    x_label = f"Minimum {x_axis.score_branch} (keep {x_axis.direction} A)"
    y_label = f"Maximum {y_axis.score_branch} (keep {y_axis.direction} B)"
    for signal_name, arrays in signal_arrays.items():
        best = signal_payloads[signal_name]["best_grid_point"]
        path = output_dir / (
            f"{stem}__{signal_name}__s_over_sqrt_s_plus_b{plot_suffix}"
        )
        plot_score_significance_2d_heatmap(
            outpath=path,
            values=arrays["significance"],
            x_thresholds=np.asarray(x_axis.thresholds),
            y_thresholds=np.asarray(y_axis.thresholds),
            x_label=x_label,
            y_label=y_label,
            title=f"{signal_name}: two-score significance",
            colorbar_label=r"$S/\sqrt{S+B}$",
            best_point=(best["x_threshold"], best["y_threshold"]),
            value_format=".3f",
        )
        plots.append(str(path))
    for group_name in ("ttHcc", "qcd", "ttjets"):
        if group_name not in group_efficiencies:
            continue
        path = output_dir / f"{stem}__{group_name}__efficiency{plot_suffix}"
        plot_score_significance_2d_heatmap(
            outpath=path,
            values=group_efficiencies[group_name],
            x_thresholds=np.asarray(x_axis.thresholds),
            y_thresholds=np.asarray(y_axis.thresholds),
            x_label=x_label,
            y_label=y_label,
            title=f"{group_name}: weighted efficiency",
            colorbar_label="Weighted efficiency",
            value_format=".2f",
        )
        plots.append(str(path))

    payload: dict[str, Any] = {
        "schema_version": 1,
        "channel": config.channel,
        "metric": settings.metric,
        "metric_definition": "Z = S / sqrt(S + B)",
        "weighting": "sample_norm * abs(raw event weight)",
        "mass_window": None,
        "axes": {
            "x": {
                "score_name": x_axis.score_name,
                "score_branch": x_axis.score_branch,
                "direction": x_axis.direction,
                "thresholds": x_axis.thresholds,
            },
            "y": {
                "score_name": y_axis.score_name,
                "score_branch": y_axis.score_branch,
                "direction": y_axis.direction,
                "thresholds": y_axis.thresholds,
            },
        },
        "baseline_semantics": (
            "Full selected MC population before score cuts; finite non-negative "
            "analysis weight required, with no finite-score requirement."
        ),
        "scan_population": (
            "Selected MC events with both scan scores finite and finite non-negative "
            "analysis weight."
        ),
        "selected_event_count_for_baseline": int(np.sum(valid_weight)),
        "selected_event_count_for_scan": int(np.sum(eligible)),
        "nonfinite_two_score_count": int(np.sum(valid_weight & ~eligible)),
        "total_baseline_weighted_yield": total_baseline,
        "total_selected_weighted_yield_matrix": _matrix(total_yields),
        "groups": {
            group.name: {
                "label": group.label,
                "truth_categories": group.truth_categories,
                "baseline_weighted_yield": group_baselines[group.name],
                "weighted_yield_matrix": _matrix(group_yields[group.name]),
                "efficiency_matrix": _matrix(group_efficiencies[group.name]),
            }
            for group in settings.groups
        },
        "signals": signal_payloads,
        "representative_points": representative_rows,
        "third_score_diagnostic": third_diagnostic,
        "validation": validation,
        "plots": plots,
    }
    write_json(summary_dir / f"{summary_stem}.json", payload)
    write_text(
        summary_dir / f"{summary_stem}.txt", _format_summary(payload)
    )
    return plots, payload
