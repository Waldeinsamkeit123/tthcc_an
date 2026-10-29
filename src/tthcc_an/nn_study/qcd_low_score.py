from __future__ import annotations

from typing import Any

import numpy as np

from tthcc_an.nn_study.config import NnStudyConfig
from tthcc_an.nn_study.dataset import NnDataset


LOW_SCORE_THRESHOLDS = (
    0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040,
    0.050, 0.060, 0.075, 0.080, 0.090, 0.100,
)
CANDIDATE_THRESHOLDS = (0.030, 0.040, 0.050, 0.075, 0.100)
GROUP_TRUTHS = {
    "ttHcc": ["ttHcc"],
    "ttHbb": ["ttHbb"],
    "ttZ": ["ttZqq", "ttZcc", "ttZbb"],
    "tt_light": ["ttLF"],
    "tt_ge1c": ["ttcj", "ttcc", "tt2c"],
    "tt_ge1b": ["ttbj", "ttbb", "tt2b"],
    "ttjets": ["ttLF", "ttcj", "ttcc", "tt2c", "ttbj", "ttbb", "tt2b"],
    "qcd": ["qcd"],
    "ttHZ": ["ttHcc", "ttHbb", "ttZqq", "ttZcc", "ttZbb"],
}


def _group_masks(config: NnStudyConfig, dataset: NnDataset) -> dict[str, np.ndarray]:
    missing = sorted(
        {name for names in GROUP_TRUTHS.values() for name in names}
        - set(config.truth_names)
    )
    if missing:
        raise ValueError(
            "Detailed low-score QCD scan is missing truth categories: "
            + ", ".join(missing)
        )
    indices = {name: config.truth_names.index(name) for name in config.truth_names}
    return {
        group: np.isin(dataset.truth_index, [indices[name] for name in names])
        for group, names in GROUP_TRUTHS.items()
    }


def _significance(signal: float, background: float) -> float:
    total = signal + background
    return float(signal / np.sqrt(total)) if total > 0.0 else 0.0


def _relative_change(value: float, baseline: float) -> float:
    return value / baseline - 1.0 if baseline > 0.0 else 0.0


def build_detailed_low_score_scan(
    *,
    config: NnStudyConfig,
    dataset: NnDataset,
    score: np.ndarray,
    weights: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, Any]:
    tthz_cuts = [
        cut
        for cut in config.analysis_regions.ar_cuts
        if cut.score_branch == "score_ttHZ" and cut.direction == ">="
    ]
    if len(tthz_cuts) != 1:
        raise ValueError(
            "Detailed low-score QCD scan requires one score_ttHZ >= AR cut."
        )
    mass_window = config.significance_mass_window
    if (
        not mass_window.enabled
        or mass_window.branch is None
        or mass_window.minimum is None
        or mass_window.maximum is None
    ):
        raise ValueError("Detailed low-score QCD scan requires a mass window.")

    tthz_cut = tthz_cuts[0]
    if tthz_cut.score_name not in dataset.scores:
        raise KeyError("Detailed low-score QCD scan requires score_ttHZ.")
    if mass_window.branch not in dataset.analysis_columns:
        raise KeyError(
            f"Detailed low-score QCD scan requires '{mass_window.branch}'."
        )
    score_tthz = np.asarray(dataset.scores[tthz_cut.score_name], dtype=np.float64)
    mass = np.asarray(dataset.analysis_columns[mass_window.branch], dtype=np.float64)
    ar = eligible & np.isfinite(score_tthz) & (score_tthz >= tthz_cut.threshold)
    ar_mass = (
        ar
        & np.isfinite(mass)
        & (mass >= mass_window.minimum)
        & (mass <= mass_window.maximum)
    )
    stage_specs = (
        ("inclusive", "selected sample", eligible, False),
        ("ar", f"score_ttHZ >= {tthz_cut.threshold:g}", ar, False),
        (
            "ar_mass_window",
            f"score_ttHZ >= {tthz_cut.threshold:g} AND "
            f"{mass_window.minimum:g} <= {mass_window.branch} <= "
            f"{mass_window.maximum:g}",
            ar_mass,
            True,
        ),
    )
    masks = _group_masks(config, dataset)
    global_yields = {
        name: float(np.sum(weights[eligible & mask]))
        for name, mask in masks.items()
    }
    stages: dict[str, Any] = {}
    for stage_name, selection, stage_mask, mass_applied in stage_specs:
        stage_yields = {
            name: float(np.sum(weights[stage_mask & mask]))
            for name, mask in masks.items()
        }
        rows = []
        for threshold in (None, *LOW_SCORE_THRESHOLDS):
            selected = stage_mask if threshold is None else stage_mask & (score < threshold)
            total_yield = float(np.sum(weights[selected]))
            groups = {}
            for name, mask in masks.items():
                selected_group = selected & mask
                weighted_yield = float(np.sum(weights[selected_group]))
                groups[name] = {
                    "event_count": int(np.sum(selected_group)),
                    "weighted_yield": weighted_yield,
                    "stage_efficiency": (
                        weighted_yield / stage_yields[name]
                        if stage_yields[name] > 0.0
                        else 0.0
                    ),
                    "global_efficiency": (
                        weighted_yield / global_yields[name]
                        if global_yields[name] > 0.0
                        else 0.0
                    ),
                }
            signals = {}
            for signal_name in ("ttHcc", "ttHbb"):
                signal_yield = groups[signal_name]["weighted_yield"]
                background_yield = max(total_yield - signal_yield, 0.0)
                signals[signal_name] = {
                    "signal_weighted_yield": signal_yield,
                    "background_weighted_yield": background_yield,
                    "significance": _significance(signal_yield, background_yield),
                }
            qcd_yield = groups["qcd"]["weighted_yield"]
            qcd_background_fractions = {
                f"{signal_name}_as_signal": (
                    qcd_yield / signals[signal_name]["background_weighted_yield"]
                    if signals[signal_name]["background_weighted_yield"] > 0.0
                    else 0.0
                )
                for signal_name in ("ttHcc", "ttHbb")
            }
            rows.append(
                {
                    "threshold": threshold,
                    "cut": (
                        "no score_qcd cut"
                        if threshold is None
                        else f"score_qcd < {threshold:g}"
                    ),
                    "event_count": int(np.sum(selected)),
                    "weighted_yield": total_yield,
                    "groups": groups,
                    "signals": signals,
                    "qcd_fraction_of_total_background": qcd_background_fractions,
                    "qcd_fraction_of_total_selected_yield": (
                        qcd_yield / total_yield if total_yield > 0.0 else 0.0
                    ),
                    "composition_fractions": {
                        name: (
                            groups[name]["weighted_yield"] / total_yield
                            if total_yield > 0.0
                            else 0.0
                        )
                        for name in ("ttHcc", "ttHbb", "ttHZ", "ttjets", "qcd")
                    },
                    "relative_to_0p10": None,
                }
            )

        baseline = next(row for row in rows if row["threshold"] == 0.1)
        if stage_name in {"ar", "ar_mass_window"}:
            for row in rows[1:]:
                changes = {
                    f"relative_{name}_{'loss' if name in {'ttHcc', 'ttHbb', 'ttZ'} else 'reduction'}": (
                        -_relative_change(
                            row["groups"][name]["weighted_yield"],
                            baseline["groups"][name]["weighted_yield"],
                        )
                    )
                    for name in ("ttHcc", "ttHbb", "ttZ", "ttjets", "qcd")
                }
                changes["significance_change"] = {
                    signal_name: {
                        "absolute": (
                            row["signals"][signal_name]["significance"]
                            - baseline["signals"][signal_name]["significance"]
                        ),
                        "relative": _relative_change(
                            row["signals"][signal_name]["significance"],
                            baseline["signals"][signal_name]["significance"],
                        ),
                    }
                    for signal_name in ("ttHcc", "ttHbb")
                }
                changes["qcd_background_fraction_change"] = {
                    signal_name: {
                        "absolute": (
                            row["qcd_fraction_of_total_background"][
                                f"{signal_name}_as_signal"
                            ]
                            - baseline["qcd_fraction_of_total_background"][
                                f"{signal_name}_as_signal"
                            ]
                        ),
                        "relative": _relative_change(
                            row["qcd_fraction_of_total_background"][
                                f"{signal_name}_as_signal"
                            ],
                            baseline["qcd_fraction_of_total_background"][
                                f"{signal_name}_as_signal"
                            ],
                        ),
                    }
                    for signal_name in ("ttHcc", "ttHbb")
                }
                row["relative_to_0p10"] = changes

        stages[stage_name] = {
            "selection": selection,
            "mass_window_applied": mass_applied,
            "flavour_tagging_applied": False,
            "before_qcd_cut_event_count": int(np.sum(stage_mask)),
            "before_qcd_cut_weighted_yield": float(np.sum(weights[stage_mask])),
            "rows": rows,
        }

    monotonic = all(
        np.all(
            np.diff([row["groups"][name]["weighted_yield"] for row in stage["rows"][1:]])
            >= -1e-10
        )
        and np.all(
            np.diff([row["groups"][name]["event_count"] for row in stage["rows"][1:]])
            >= 0
        )
        for stage in stages.values()
        for name in masks
    )
    significances = np.asarray(
        [
            row["signals"][signal]["significance"]
            for stage in stages.values()
            for row in stage["rows"]
            for signal in ("ttHcc", "ttHbb")
        ]
    )
    fractions = np.asarray(
        [
            value
            for stage in stages.values()
            for row in stage["rows"]
            for value in (
                row["qcd_fraction_of_total_selected_yield"],
                *row["qcd_fraction_of_total_background"].values(),
                *row["composition_fractions"].values(),
            )
        ]
    )
    validation = {
        "thresholds_strictly_ordered": bool(
            np.all(np.diff(LOW_SCORE_THRESHOLDS) > 0.0)
        ),
        "event_counts_and_weighted_yields_monotonic": bool(monotonic),
        "significances_finite_and_nonnegative": bool(
            np.all(np.isfinite(significances)) and np.all(significances >= 0.0)
        ),
        "fractions_in_unit_interval": bool(
            np.all(np.isfinite(fractions))
            and np.all(fractions >= 0.0)
            and np.all(fractions <= 1.0)
        ),
        "stage_c_subset_of_stage_b": bool(np.all(~ar_mass | ar)),
        "mass_window_applied_only_to_stage_c": bool(
            not stages["inclusive"]["mass_window_applied"]
            and not stages["ar"]["mass_window_applied"]
            and stages["ar_mass_window"]["mass_window_applied"]
        ),
        "flavour_tagging_applied": False,
    }
    if not all(
        value is False if key == "flavour_tagging_applied" else value
        for key, value in validation.items()
    ):
        raise ValueError("Detailed low-score QCD scan validation failed.")

    return {
        "thresholds": list(LOW_SCORE_THRESHOLDS),
        "cut_semantics": "keep score_qcd < threshold",
        "weighting": "sample_norm * abs(raw event weight)",
        "global_efficiency_reference": (
            "Original selected sample with finite score_qcd and finite "
            "non-negative analysis weight."
        ),
        "stage_efficiency_reference": "Each stage before the score_qcd cut.",
        "significance": "S/sqrt(S+B); B is every other selected MC event.",
        "group_truth_categories": GROUP_TRUTHS,
        "stages": stages,
        "candidate_thresholds": list(CANDIDATE_THRESHOLDS),
        "candidate_comparison": {
            stage_name: [
                next(
                    row
                    for row in stages[stage_name]["rows"]
                    if row["threshold"] == threshold
                )
                for threshold in CANDIDATE_THRESHOLDS
            ]
            for stage_name in ("ar", "ar_mass_window")
        },
        "validation": validation,
    }


def build_logx_distribution(
    *,
    config: NnStudyConfig,
    dataset: NnDataset,
    score: np.ndarray,
    weights: np.ndarray,
    eligible: np.ndarray,
    minimum: float = 1e-5,
    maximum: float = 1.0,
    bins_count: int = 50,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    bins = np.geomspace(minimum, maximum, bins_count + 1)
    histograms = {}
    groups = []
    in_range = eligible & (score >= minimum) & (score <= maximum)
    for group in config.qcd_score_scan.distribution_groups:
        mask = np.isin(
            dataset.truth_index,
            [config.truth_names.index(name) for name in group.truth_categories],
        ) & in_range
        histogram = np.histogram(
            score[mask], bins=bins, weights=weights[mask]
        )[0].astype(np.float64)
        integral = float(np.sum(histogram))
        shape = histogram / integral if integral > 0.0 else np.zeros_like(histogram)
        histograms[group.name] = shape
        groups.append(
            {
                "name": group.name,
                "label": group.label,
                "truth_categories": group.truth_categories,
                "weighted_yield_in_plot_range": integral,
                "normalized_histogram_integral": (
                    float(np.sum(shape)) if integral > 0.0 else None
                ),
            }
        )

    nonpositive = eligible & (score <= 0.0)
    underflow = eligible & (score > 0.0) & (score < minimum)
    overflow = eligible & (score > maximum)
    metadata = {
        "plot_range": [minimum, maximum],
        "bins": bins_count,
        "bin_edges": bins.tolist(),
        "binning": "Geometric score_qcd bin edges.",
        "xscale": "log",
        "yscale": "log",
        "normalization": (
            "Each displayed group is independently normalized to unit weighted "
            "integral over the plotted score range; values are fractions per "
            "logarithmic bin."
        ),
        "weighting": "sample_norm * abs(raw event weight)",
        "reference_lines": [0.05, 0.075, 0.10],
        "score_qcd_nonpositive_event_count": int(np.sum(nonpositive)),
        "score_qcd_nonpositive_weighted_yield": float(np.sum(weights[nonpositive])),
        "positive_underflow_event_count": int(np.sum(underflow)),
        "positive_underflow_weighted_yield": float(np.sum(weights[underflow])),
        "overflow_event_count": int(np.sum(overflow)),
        "overflow_weighted_yield": float(np.sum(weights[overflow])),
        "groups": groups,
    }
    return bins, histograms, metadata


def format_detailed_low_score_scan(scan: dict[str, Any]) -> list[str]:
    lines = []
    titles = {
        "inclusive": "inclusive",
        "ar": "score_ttHZ >= 0.70",
        "ar_mass_window": "score_ttHZ >= 0.70 + mass window",
    }
    for stage_name, stage in scan["stages"].items():
        lines.extend(
            [
                "",
                f"=== Detailed low-score scan: {titles[stage_name]} ===",
                f"Selection before qcd cut: {stage['selection']}",
                "Efficiencies are relative to this stage before the qcd cut.",
                "cut     eff_cc eff_bb eff_Z  eff_ttj eff_qcd qcd/Bcc qcd/all "
                "Z_cc      Z_bb",
            ]
        )
        for row in stage["rows"]:
            cut = "none" if row["threshold"] is None else f"{row['threshold']:.3f}"
            group = row["groups"]
            lines.append(
                f"{cut:<8s} {group['ttHcc']['stage_efficiency']:<6.4f} "
                f"{group['ttHbb']['stage_efficiency']:<6.4f} "
                f"{group['ttZ']['stage_efficiency']:<6.4f} "
                f"{group['ttjets']['stage_efficiency']:<8.4f} "
                f"{group['qcd']['stage_efficiency']:<7.4f} "
                f"{row['qcd_fraction_of_total_background']['ttHcc_as_signal']:<7.4f} "
                f"{row['qcd_fraction_of_total_selected_yield']:<7.4f} "
                f"{row['signals']['ttHcc']['significance']:<9.6g} "
                f"{row['signals']['ttHbb']['significance']:<9.6g}"
            )
        lines.extend(
            [
                "",
                "cut     Y_cc       Y_bb       Y_Z        Y_ttj      Y_qcd      "
                "N_cc    N_bb    N_ttj   N_qcd",
            ]
        )
        for row in stage["rows"]:
            cut = "none" if row["threshold"] is None else f"{row['threshold']:.3f}"
            group = row["groups"]
            lines.append(
                f"{cut:<8s} {group['ttHcc']['weighted_yield']:<10.6g} "
                f"{group['ttHbb']['weighted_yield']:<10.6g} "
                f"{group['ttZ']['weighted_yield']:<10.6g} "
                f"{group['ttjets']['weighted_yield']:<10.6g} "
                f"{group['qcd']['weighted_yield']:<10.6g} "
                f"{group['ttHcc']['event_count']:<7d} "
                f"{group['ttHbb']['event_count']:<7d} "
                f"{group['ttjets']['event_count']:<7d} "
                f"{group['qcd']['event_count']:<7d}"
            )
        if stage_name == "ar_mass_window":
            lines.extend(
                [
                    "",
                    "AR_mass_window composition:",
                    "cut     ttHcc/all ttHbb/all ttHZ/all ttjets/all qcd/all",
                ]
            )
            for row in stage["rows"]:
                cut = "none" if row["threshold"] is None else f"{row['threshold']:.3f}"
                comp = row["composition_fractions"]
                lines.append(
                    f"{cut:<8s} {comp['ttHcc']:<9.5f} {comp['ttHbb']:<9.5f} "
                    f"{comp['ttHZ']:<8.5f} {comp['ttjets']:<10.5f} "
                    f"{comp['qcd']:<7.5f}"
                )

    lines.extend(
        [
            "",
            "=== Detailed low-score candidate comparison vs 0.10 ===",
            "stage          cut   loss_cc loss_bb red_ttj red_qcd dZ_cc    "
            "dZ_bb    dqcd/Bcc",
        ]
    )
    for stage_name, rows in scan["candidate_comparison"].items():
        for row in rows:
            change = row["relative_to_0p10"]
            lines.append(
                f"{stage_name:<14s} {row['threshold']:<5.3f} "
                f"{change['relative_ttHcc_loss']:<7.4f} "
                f"{change['relative_ttHbb_loss']:<7.4f} "
                f"{change['relative_ttjets_reduction']:<7.4f} "
                f"{change['relative_qcd_reduction']:<7.4f} "
                f"{change['significance_change']['ttHcc']['relative']:<8.4f} "
                f"{change['significance_change']['ttHbb']['relative']:<8.4f} "
                f"{change['qcd_background_fraction_change']['ttHcc']['absolute']:<9.5f}"
            )
    return lines
