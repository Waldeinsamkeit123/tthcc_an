from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tthcc_an.nn_study.config import (
    NnAnalysisRegionCut,
    NnArScorePlotRange,
    NnStudyConfig,
)
from tthcc_an.nn_study.dataset import NnDataset
from tthcc_an.nn_study.plotting import plot_score_distribution
from tthcc_an.nn_study.reporting import write_json, write_text


def score_cut_mask(values: np.ndarray, cut: NnAnalysisRegionCut) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if cut.direction == "<":
        return finite & (values < cut.threshold)
    if cut.direction == "<=":
        return finite & (values <= cut.threshold)
    if cut.direction == ">":
        return finite & (values > cut.threshold)
    if cut.direction == ">=":
        return finite & (values >= cut.threshold)
    raise ValueError(f"Unsupported analysis-region cut direction '{cut.direction}'.")


def build_ar_mask(config: NnStudyConfig, dataset: NnDataset) -> np.ndarray:
    if not config.analysis_regions.enabled:
        raise ValueError("analysis_regions is not enabled in this config.")
    mask = np.ones(dataset.truth_index.shape, dtype=bool)
    for cut in config.analysis_regions.ar_cuts:
        if cut.score_name not in dataset.scores:
            raise KeyError(f"AR score '{cut.score_name}' was not loaded.")
        mask &= score_cut_mask(dataset.scores[cut.score_name], cut)
    return mask


def qcd_veto_threshold(config: NnStudyConfig) -> float | None:
    matching = [
        cut
        for cut in config.analysis_regions.ar_cuts
        if cut.score_branch == "score_qcd" and cut.direction == "<"
    ]
    if not matching:
        return None
    if len(matching) != 1:
        raise ValueError("Expected exactly one score_qcd '<' cut in the AR definition.")
    return matching[0].threshold


def _ar_histogram_edges_and_coverage(
    *,
    plot_range: NnArScorePlotRange,
    values: np.ndarray,
    truth_index: np.ndarray,
    weights: np.ndarray,
    number_of_truths: int,
    floating_tolerance: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(values, dtype=np.float64)
    truth_index = np.asarray(truth_index)
    weights = np.asarray(weights, dtype=np.float64)
    valid_truth = (truth_index >= 0) & (truth_index < number_of_truths)
    valid_weight = np.isfinite(weights) & (weights > 0.0)
    finite_score = np.isfinite(values)
    eligible = valid_truth & valid_weight
    nonfinite = eligible & ~finite_score
    if np.any(nonfinite):
        raise ValueError(
            f"AR score '{plot_range.score_branch}' has "
            f"{int(np.sum(nonfinite))} non-finite positive-weight events."
        )

    nominal_minimum = plot_range.minimum
    nominal_maximum = plot_range.maximum
    histogram_minimum = nominal_minimum
    histogram_maximum = nominal_maximum
    finite_eligible = eligible & finite_score
    observed_minimum = (
        float(np.min(values[finite_eligible])) if np.any(finite_eligible) else None
    )
    observed_maximum = (
        float(np.max(values[finite_eligible])) if np.any(finite_eligible) else None
    )
    below = finite_eligible & (values < nominal_minimum)
    above = finite_eligible & (values > nominal_maximum)
    if np.any(below):
        excursion = nominal_minimum - float(np.min(values[below]))
        if excursion > floating_tolerance:
            raise ValueError(
                f"AR score '{plot_range.score_branch}' extends {excursion:g} below "
                f"its configured plotting minimum {nominal_minimum:g}."
            )
        histogram_minimum = float(np.nextafter(np.min(values[below]), -np.inf))
    if np.any(above):
        excursion = float(np.max(values[above])) - nominal_maximum
        if excursion > floating_tolerance:
            raise ValueError(
                f"AR score '{plot_range.score_branch}' extends {excursion:g} above "
                f"its configured plotting maximum {nominal_maximum:g}."
            )
        histogram_maximum = float(np.nextafter(np.max(values[above]), np.inf))

    edges = np.linspace(
        histogram_minimum, histogram_maximum, plot_range.bins + 1, dtype=np.float64
    )
    histogram = np.histogram(
        values[finite_eligible], bins=edges, weights=weights[finite_eligible]
    )[0]
    expected_yield = float(np.sum(weights[finite_eligible]))
    histogram_yield = float(np.sum(histogram))
    if not np.isclose(
        histogram_yield,
        expected_yield,
        rtol=1e-12,
        atol=1e-9 * max(expected_yield, 1.0),
    ):
        raise ValueError(
            f"AR score '{plot_range.score_branch}' histogram loses weighted yield: "
            f"{histogram_yield:g} != {expected_yield:g}."
        )
    return edges, {
        "score_name": plot_range.score_name,
        "score_branch": plot_range.score_branch,
        "display_range": [nominal_minimum, nominal_maximum],
        "histogram_range": [histogram_minimum, histogram_maximum],
        "bins": plot_range.bins,
        "nominal_bin_width": (
            nominal_maximum - nominal_minimum
        ) / plot_range.bins,
        "observed_finite_minimum": observed_minimum,
        "observed_finite_maximum": observed_maximum,
        "below_nominal_event_count": int(np.sum(below)),
        "above_nominal_event_count": int(np.sum(above)),
        "histogram_weighted_yield": histogram_yield,
        "expected_weighted_yield": expected_yield,
        "weighted_yield_preserved": True,
    }


def _category_composition(
    *,
    index: np.ndarray,
    names: list[str],
    ar_mask: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    valid_weight = np.isfinite(weights) & (weights >= 0.0)
    total_ar_yield = float(np.sum(weights[ar_mask & valid_weight]))
    categories: dict[str, Any] = {}
    for number, name in enumerate(names):
        category = index == number
        baseline_yield = float(np.sum(weights[category & valid_weight]))
        selected = category & ar_mask
        selected_yield = float(np.sum(weights[selected & valid_weight]))
        categories[name] = {
            "pre_ar_event_count": int(np.sum(category)),
            "ar_event_count": int(np.sum(selected)),
            "pre_ar_weighted_yield": baseline_yield,
            "ar_weighted_yield": selected_yield,
            "fraction_of_total_ar_weighted_yield": (
                selected_yield / total_ar_yield if total_ar_yield > 0.0 else 0.0
            ),
            "weighted_efficiency_relative_to_pre_ar": (
                selected_yield / baseline_yield if baseline_yield > 0.0 else 0.0
            ),
        }
    return {
        "total_ar_event_count": int(np.sum(ar_mask)),
        "total_ar_weighted_yield": total_ar_yield,
        "categories": categories,
    }


def _format_ar_summary(payload: dict[str, Any]) -> str:
    lines = [
        "=== NN-level AR Score Composition ===",
        f"Channel: {payload['channel']}",
        f"AR: {payload['ar_definition']}",
        "Weight: sample_norm * abs(weight)",
        "",
        "Fine truth category | N(AR) | Yield(AR) | AR fraction | Efficiency",
    ]
    for name, entry in payload["fine_truth"]["categories"].items():
        lines.append(
            f"{name} | {entry['ar_event_count']} | "
            f"{entry['ar_weighted_yield']:.8g} | "
            f"{entry['fraction_of_total_ar_weighted_yield']:.6g} | "
            f"{entry['weighted_efficiency_relative_to_pre_ar']:.6g}"
        )
    lines.extend(["", "Coarse category | N(AR) | Yield(AR) | AR fraction | Efficiency"])
    for name, entry in payload["coarse_truth"]["categories"].items():
        lines.append(
            f"{name} | {entry['ar_event_count']} | "
            f"{entry['ar_weighted_yield']:.8g} | "
            f"{entry['fraction_of_total_ar_weighted_yield']:.6g} | "
            f"{entry['weighted_efficiency_relative_to_pre_ar']:.6g}"
        )
    return "\n".join(lines) + "\n"


def run_ar_score_distributions(
    *,
    config: NnStudyConfig,
    dataset: NnDataset,
    plot_dir: Path,
    summary_dir: Path,
    plot_suffix: str,
) -> tuple[list[str], dict[str, Any]]:
    ar_mask = build_ar_mask(config, dataset)
    score_by_name = {score.name: score for score in config.all_scores}
    plot_scores = [*config.scores]
    plot_scores.extend(
        score_by_name[name]
        for name in config.analysis_discriminants
        if name not in config.score_names
    )
    output_dir = plot_dir / "score_fine_truth_ar"
    plots: list[str] = []
    plot_range_metadata: dict[str, Any] = {}
    for score in plot_scores:
        if score.name not in config.analysis_regions.ar_score_plot_ranges:
            raise ValueError(
                f"Missing configured AR score plot range for '{score.branch}'."
            )
        plot_range = config.analysis_regions.ar_score_plot_ranges[score.name]
        bin_edges, coverage = _ar_histogram_edges_and_coverage(
            plot_range=plot_range,
            values=dataset.scores[score.name][ar_mask],
            truth_index=dataset.truth_index[ar_mask],
            weights=dataset.analysis_weight[ar_mask],
            number_of_truths=len(config.truth_categories),
        )
        plot_range_metadata[score.branch] = coverage
        for normalize, suffix in ((True, ""), (False, "__yield")):
            path = output_dir / (
                f"{score.branch}__fine_truth__AR{suffix}{plot_suffix}"
            )
            plot_score_distribution(
                outpath=path,
                score=score,
                values=dataset.scores[score.name][ar_mask],
                truth_index=dataset.truth_index[ar_mask],
                weights=dataset.analysis_weight[ar_mask],
                truths=config.truth_categories,
                config=config,
                normalize=normalize,
                figsize=(7.2, 5.2),
                legend_columns=3,
                legend_fontsize=8.0,
                bin_edges=bin_edges,
                display_range=(plot_range.minimum, plot_range.maximum),
            )
            plots.append(str(path))

    cuts = [
        {
            "score_name": cut.score_name,
            "score_branch": cut.score_branch,
            "direction": cut.direction,
            "threshold": cut.threshold,
        }
        for cut in config.analysis_regions.ar_cuts
    ]
    payload = {
        "schema_version": 1,
        "channel": config.channel,
        "ar_definition": " AND ".join(
            f"{cut.score_branch} {cut.direction} {cut.threshold:g}"
            for cut in config.analysis_regions.ar_cuts
        ),
        "cuts": cuts,
        "weighting": "sample_norm * abs(raw event weight)",
        "mass_window_applied": False,
        "flavour_tagging_applied": False,
        "pre_ar_selected_event_count": int(dataset.truth_index.size),
        "ar_selected_event_count": int(np.sum(ar_mask)),
        "fine_truth": _category_composition(
            index=dataset.truth_index,
            names=config.truth_names,
            ar_mask=ar_mask,
            weights=dataset.analysis_weight,
        ),
        "coarse_truth": _category_composition(
            index=dataset.prediction_truth_index,
            names=config.prediction_truth_names,
            ar_mask=ar_mask,
            weights=dataset.analysis_weight,
        ),
        "plots": plots,
        "score_plot_ranges": plot_range_metadata,
        "validation": {
            "mask_recomputed_from_configured_cuts": True,
            "all_plots_use_same_ar_mask": True,
            "all_histogram_weighted_yields_preserved": all(
                entry["weighted_yield_preserved"]
                for entry in plot_range_metadata.values()
            ),
        },
    }
    write_json(summary_dir / "ar_score_composition.json", payload)
    write_text(
        summary_dir / "ar_score_composition.txt", _format_ar_summary(payload)
    )
    return plots, payload


def _truth_union_mask(
    config: NnStudyConfig, dataset: NnDataset, names: list[str]
) -> np.ndarray:
    indices = [config.truth_names.index(name) for name in names]
    return np.isin(dataset.truth_index, indices)


def _significance(signal: float, background: float) -> float:
    total = signal + background
    return signal / np.sqrt(total) if total > 0.0 else 0.0


def _format_conditional_summary(payload: dict[str, Any]) -> str:
    lines = [
        "=== Conditional QCD-Veto Study ===",
        f"Channel: {payload['channel']}",
        "Weight: sample_norm * abs(weight)",
        "No final QCD-veto threshold is selected.",
        "",
        "ttHZ cut | qcd cut | eff ttHcc | eff ttHbb | eff ttZ | eff ttjets | eff qcd | Z ttHcc | Z ttHbb",
    ]
    for row in payload["rows"]:
        qcd_cut = "none" if row["qcd_threshold"] is None else f"<{row['qcd_threshold']:g}"
        efficiencies = row["global_efficiencies"]
        lines.append(
            f">={row['tthz_threshold']:g} | {qcd_cut} | "
            f"{efficiencies['ttHcc']:.6g} | {efficiencies['ttHbb']:.6g} | "
            f"{efficiencies['ttZ']:.6g} | {efficiencies['ttjets']:.6g} | "
            f"{efficiencies['qcd']:.6g} | "
            f"{row['signals']['ttHcc']['significance']:.8g} | "
            f"{row['signals']['ttHbb']['significance']:.8g}"
        )
    return "\n".join(lines) + "\n"


def run_conditional_qcd_veto(
    *,
    config: NnStudyConfig,
    dataset: NnDataset,
    summary_dir: Path,
) -> tuple[list[str], dict[str, Any]]:
    settings = config.analysis_regions.conditional_qcd_veto
    if not settings.enabled:
        raise ValueError("conditional_qcd_veto is not enabled in this config.")
    assert settings.tthz_score_name is not None
    assert settings.qcd_score_name is not None
    score_tthz = np.asarray(dataset.scores[settings.tthz_score_name], dtype=np.float64)
    score_qcd = np.asarray(dataset.scores[settings.qcd_score_name], dtype=np.float64)
    weights = np.asarray(dataset.analysis_weight, dtype=np.float64)
    valid = np.isfinite(weights) & (weights >= 0.0)
    group_masks = {
        "ttHcc": _truth_union_mask(config, dataset, ["ttHcc"]),
        "ttHbb": _truth_union_mask(config, dataset, ["ttHbb"]),
        "ttZ": _truth_union_mask(config, dataset, ["ttZqq", "ttZcc", "ttZbb"]),
        "ttjets": _truth_union_mask(
            config,
            dataset,
            ["ttLF", "ttcj", "ttcc", "tt2c", "ttbj", "ttbb", "tt2b"],
        ),
        "qcd": _truth_union_mask(config, dataset, ["qcd"]),
    }
    global_baselines = {
        name: float(np.sum(weights[mask & valid])) for name, mask in group_masks.items()
    }
    signal_union = group_masks["ttHcc"] | group_masks["ttHbb"]
    rows: list[dict[str, Any]] = []
    for tthz_threshold in settings.tthz_thresholds:
        preselection = valid & np.isfinite(score_tthz) & (score_tthz >= tthz_threshold)
        preselection_yields = {
            name: float(np.sum(weights[preselection & mask]))
            for name, mask in group_masks.items()
        }
        reference_row: dict[str, Any] | None = None
        for qcd_threshold in [None, *settings.qcd_thresholds]:
            selected = preselection.copy()
            if qcd_threshold is not None:
                selected &= np.isfinite(score_qcd) & (score_qcd < qcd_threshold)
            yields = {
                name: float(np.sum(weights[selected & mask]))
                for name, mask in group_masks.items()
            }
            total_yield = float(np.sum(weights[selected]))
            all_background = float(np.sum(weights[selected & ~signal_union]))
            signals = {}
            for signal_name in ("ttHcc", "ttHbb"):
                signal_yield = yields[signal_name]
                background_yield = max(total_yield - signal_yield, 0.0)
                signals[signal_name] = {
                    "signal_weighted_yield": signal_yield,
                    "background_weighted_yield": background_yield,
                    "significance": _significance(signal_yield, background_yield),
                }
            row = {
                "tthz_threshold": tthz_threshold,
                "qcd_threshold": qcd_threshold,
                "selected_event_count": int(np.sum(selected)),
                "total_selected_weighted_yield": total_yield,
                "all_background_expected_yield_excluding_ttHcc_ttHbb": all_background,
                "weighted_yields": yields,
                "global_efficiencies": {
                    name: yields[name] / global_baselines[name]
                    if global_baselines[name] > 0.0
                    else 0.0
                    for name in group_masks
                },
                "conditional_efficiencies_relative_to_tthz_preselection": {
                    name: yields[name] / preselection_yields[name]
                    if preselection_yields[name] > 0.0
                    else 0.0
                    for name in group_masks
                },
                "signals": signals,
            }
            if reference_row is None:
                reference_row = row
                row["changes_vs_no_qcd_cut"] = {
                    "signal_efficiency_loss": {"ttHcc": 0.0, "ttHbb": 0.0, "ttZ": 0.0},
                    "qcd_rejection_gain": 0.0,
                    "significance_change": {
                        "ttHcc": {"absolute": 0.0, "relative": 0.0},
                        "ttHbb": {"absolute": 0.0, "relative": 0.0},
                    },
                }
            else:
                conditional = row[
                    "conditional_efficiencies_relative_to_tthz_preselection"
                ]
                significance_change = {}
                for signal_name in ("ttHcc", "ttHbb"):
                    current = row["signals"][signal_name]["significance"]
                    reference = reference_row["signals"][signal_name]["significance"]
                    significance_change[signal_name] = {
                        "absolute": current - reference,
                        "relative": current / reference - 1.0 if reference > 0.0 else 0.0,
                    }
                row["changes_vs_no_qcd_cut"] = {
                    "signal_efficiency_loss": {
                        name: 1.0 - conditional[name]
                        for name in ("ttHcc", "ttHbb", "ttZ")
                    },
                    "qcd_rejection_gain": 1.0 - conditional["qcd"],
                    "significance_change": significance_change,
                }
            rows.append(row)

    payload = {
        "schema_version": 1,
        "channel": config.channel,
        "weighting": "sample_norm * abs(raw event weight)",
        "tthz_score_branch": settings.tthz_score_branch,
        "qcd_score_branch": settings.qcd_score_branch,
        "tthz_thresholds": settings.tthz_thresholds,
        "qcd_thresholds": settings.qcd_thresholds,
        "efficiency_denominators": {
            "global": "Original selected sample yield for the same truth group.",
            "conditional": "Yield after the corresponding score_ttHZ preselection and before score_qcd cut.",
        },
        "all_background_definition": "Every selected event except fine-truth ttHcc and ttHbb.",
        "signal_background_definition": "For each signal, B is every other selected MC event.",
        "global_baseline_weighted_yields": global_baselines,
        "rows": rows,
        "interpretation": (
            "Diagnostic comparison only. The code does not rank or select a final "
            "standalone QCD-veto threshold."
        ),
    }
    write_json(summary_dir / "qcd_veto_conditional.json", payload)
    write_text(
        summary_dir / "qcd_veto_conditional.txt",
        _format_conditional_summary(payload),
    )
    return [], payload
