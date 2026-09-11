from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tthcc_an.metrics import weighted_quantile
from tthcc_an.nn_study.config import (
    NnMassPopulation,
    NnMassScan,
    NnMassSculptingConfig,
    NnMassVariable,
    NnStudyConfig,
)
from tthcc_an.nn_study.dataset import NnDataset
from tthcc_an.nn_study.plotting import plot_mass_sculpting
from tthcc_an.nn_study.reporting import write_json, write_text


def _population_mask(
    config: NnStudyConfig,
    dataset: NnDataset,
    population: NnMassPopulation,
) -> np.ndarray:
    mask = np.ones(dataset.truth_index.shape, dtype=bool)
    if population.truth_categories is not None:
        index_by_name = {
            truth.name: index for index, truth in enumerate(config.truth_categories)
        }
        indices = [index_by_name[name] for name in population.truth_categories]
        mask &= np.isin(dataset.truth_index, indices)
    if population.samples is not None:
        index_by_name = {
            sample.name: index for index, sample in enumerate(config.samples)
        }
        indices = [index_by_name[name] for name in population.samples]
        mask &= np.isin(dataset.sample_index, indices)
    if population.exclude_samples:
        index_by_name = {
            sample.name: index for index, sample in enumerate(config.samples)
        }
        excluded_indices = [
            index_by_name[name] for name in population.exclude_samples
        ]
        mask &= ~np.isin(dataset.sample_index, excluded_indices)
    if population.sample_labels is not None:
        included_indices = [
            index
            for index, sample in enumerate(config.samples)
            if sample.label in population.sample_labels
        ]
        mask &= np.isin(dataset.sample_index, included_indices)
    if population.exclude_sample_labels:
        excluded_indices = [
            index
            for index, sample in enumerate(config.samples)
            if sample.label in population.exclude_sample_labels
        ]
        mask &= ~np.isin(dataset.sample_index, excluded_indices)
    return mask


def _mass_bins(
    variable: NnMassVariable,
    values: np.ndarray,
    weights: np.ndarray,
    population_mask: np.ndarray,
) -> np.ndarray:
    valid = (
        population_mask
        & np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if not np.any(valid):
        raise ValueError(
            f"No finite positive-weight events for mass variable '{variable.branch}'."
        )
    lower = float(
        weighted_quantile(
            values[valid], weights[valid], variable.range_quantiles[0]
        )
    )
    upper = float(
        weighted_quantile(
            values[valid], weights[valid], variable.range_quantiles[1]
        )
    )
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        lower = float(np.min(values[valid]))
        upper = float(np.max(values[valid]))
    if lower >= upper:
        padding = max(1.0, abs(lower) * 0.02)
        lower -= padding
        upper += padding
    else:
        padding = max(
            (upper - lower) * variable.padding_fraction,
            1.0,
        )
        lower -= padding
        upper += padding
    return np.linspace(lower, upper, variable.bins + 1, dtype=np.float64)


def _cut_mask(values: np.ndarray, direction: str, threshold: float) -> np.ndarray:
    if direction == ">":
        return values > threshold
    if direction == ">=":
        return values >= threshold
    if direction == "<":
        return values < threshold
    if direction == "<=":
        return values <= threshold
    raise ValueError(f"Unsupported mass-sculpting cut direction '{direction}'.")


def _shape_payload(
    values: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    counts, _ = np.histogram(values[mask], bins=bins, weights=weights[mask])
    counts = counts.astype(np.float64)
    total = float(np.sum(counts))
    probabilities = np.zeros_like(counts)
    density = np.zeros_like(counts)
    if total > 0.0:
        probabilities = counts / total
        density = probabilities / np.diff(bins)
    integral = float(np.sum(density * np.diff(bins)))
    return counts, density, integral


def _nested_selection_valid(masks: list[np.ndarray], direction: str) -> bool:
    if len(masks) < 2:
        return True
    if direction in {">", ">="}:
        return all(
            not np.any(current & ~previous)
            for previous, current in zip(masks[:-1], masks[1:])
        )
    return all(
        not np.any(previous & ~current)
        for previous, current in zip(masks[:-1], masks[1:])
    )


def _scan_payload(
    *,
    config: NnStudyConfig,
    dataset: NnDataset,
    variable: NnMassVariable,
    scan: NnMassScan,
    population: NnMassPopulation,
    bins: np.ndarray,
    population_mask: np.ndarray,
    outpath: Path | None,
    make_plot: bool = True,
) -> dict[str, Any]:
    mass_values = np.asarray(dataset.analysis_columns[variable.branch], dtype=np.float64)
    score_values = np.asarray(dataset.scores[scan.score_name], dtype=np.float64)
    weights = np.asarray(dataset.analysis_weight, dtype=np.float64)
    in_range = (mass_values >= bins[0]) & (mass_values <= bins[-1])
    base_mask = (
        population_mask
        & in_range
        & np.isfinite(mass_values)
        & np.isfinite(score_values)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    before_count = int(np.sum(base_mask))
    before_yield = float(np.sum(weights[base_mask]))
    if before_count == 0 or before_yield <= 0.0:
        raise ValueError(
            f"Mass-sculpting scan '{scan.score_branch}' has no events in the plotted range."
        )

    base_counts, base_density, base_integral = _shape_payload(
        mass_values, weights, base_mask, bins
    )
    base_probabilities = base_counts / np.sum(base_counts)
    selected_masks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for threshold in scan.thresholds:
        selected = base_mask & _cut_mask(score_values, scan.direction, threshold)
        selected_masks.append(selected)
        after_count = int(np.sum(selected))
        after_yield = float(np.sum(weights[selected]))
        counts, density, integral = _shape_payload(
            mass_values, weights, selected, bins
        )
        if after_yield > 0.0:
            probabilities = counts / np.sum(counts)
            shape_change = float(
                np.max(np.abs(probabilities - base_probabilities))
            )
        else:
            shape_change = None
        efficiency = after_yield / before_yield
        if not 0.0 <= efficiency <= 1.0 + 1e-12:
            raise ValueError(
                f"Mass-sculpting weighted efficiency outside [0, 1]: {efficiency}."
            )
        efficiency = min(max(efficiency, 0.0), 1.0)
        retention = np.divide(
            counts,
            base_counts,
            out=np.zeros_like(counts),
            where=base_counts > 0.0,
        )
        rows.append(
            {
                "score_name": scan.score_name,
                "score_branch": scan.score_branch,
                "direction": scan.direction,
                "threshold": threshold,
                "weighted_yield_before_cut": before_yield,
                "weighted_yield_after_cut": after_yield,
                "weighted_efficiency": efficiency,
                "event_count_before_cut": before_count,
                "event_count_after_cut": after_count,
                "normalized_shape_integral": integral,
                "max_abs_bin_probability_difference": shape_change,
            }
        )
        curves.append(
            {
                "threshold": threshold,
                "label": f"{scan.score_branch} {scan.direction} {threshold:g}",
                "density": density,
                "retention": retention,
                "has_yield": after_yield > 0.0,
            }
        )

    nested_valid = _nested_selection_valid(selected_masks, scan.direction)
    if not nested_valid:
        raise ValueError(
            f"Mass-sculpting selections are not nested for '{scan.score_branch}'."
        )
    if make_plot:
        if outpath is None:
            raise ValueError("A plot path is required for a plotted mass scan.")
        plot_mass_sculpting(
            outpath=outpath,
            bins=bins,
            base_density=base_density,
            curves=curves,
            variable_label=variable.label,
            population_label=population.label,
            score_branch=scan.score_branch,
        )
    return {
        "variable": variable.branch,
        "variable_label": variable.label,
        "score_name": scan.score_name,
        "score_branch": scan.score_branch,
        "population": population.name,
        "population_label": population.label,
        "direction": scan.direction,
        "thresholds": scan.thresholds,
        "plot": None if outpath is None else str(outpath),
        "bins": variable.bins,
        "plot_range": [float(bins[0]), float(bins[-1])],
        "range_quantiles": list(variable.range_quantiles),
        "padding_fraction": variable.padding_fraction,
        "uncut_normalized_shape_integral": base_integral,
        "nested_selection_valid": nested_valid,
        "points": rows,
    }


def _binary_complementarity_validation(
    *,
    config: NnStudyConfig,
    dataset: NnDataset,
    settings: NnMassSculptingConfig,
    scans: list[NnMassScan] | None = None,
) -> dict[str, Any]:
    if len(config.scores) != 2:
        return {
            "applicable": False,
            "reason": "Native prediction model is not binary.",
        }
    selected_scans = settings.scans if scans is None else scans
    upper_scans = [scan for scan in selected_scans if scan.direction == ">"]
    lower_scans = [scan for scan in selected_scans if scan.direction == "<"]
    if len(upper_scans) != 1 or len(lower_scans) != 1:
        return {
            "applicable": False,
            "reason": "Exactly one '>' scan and one '<' scan are required.",
        }

    upper_scan = upper_scans[0]
    lower_scan = lower_scans[0]
    upper_score = np.asarray(dataset.scores[upper_scan.score_name], dtype=np.float64)
    lower_score = np.asarray(dataset.scores[lower_scan.score_name], dtype=np.float64)
    weights = np.asarray(dataset.analysis_weight, dtype=np.float64)
    finite = (
        np.isfinite(upper_score)
        & np.isfinite(lower_score)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    score_sum = upper_score[finite] + lower_score[finite]
    max_sum_deviation = (
        float(np.max(np.abs(score_sum - 1.0))) if score_sum.size else None
    )
    boundary_tolerance = max(
        1e-6,
        2.0 * (max_sum_deviation or 0.0),
    )

    rows: list[dict[str, Any]] = []
    for upper_cut in upper_scan.thresholds:
        target_lower_cut = 1.0 - upper_cut
        matches = [
            cut
            for cut in lower_scan.thresholds
            if np.isclose(cut, target_lower_cut, rtol=0.0, atol=1e-12)
        ]
        if len(matches) != 1:
            continue
        lower_cut = matches[0]
        upper_selected = finite & (upper_score > upper_cut)
        lower_selected = finite & (lower_score < lower_cut)
        mismatch = upper_selected ^ lower_selected
        boundary = (
            (np.abs(upper_score - upper_cut) <= boundary_tolerance)
            | (np.abs(lower_score - lower_cut) <= boundary_tolerance)
        )
        nonboundary_mismatch = mismatch & ~boundary
        populations: dict[str, Any] = {}
        for population in settings.populations:
            population_mask = _population_mask(config, dataset, population)
            upper_mask = population_mask & upper_selected
            lower_mask = population_mask & lower_selected
            upper_yield = float(np.sum(weights[upper_mask]))
            lower_yield = float(np.sum(weights[lower_mask]))
            absolute_difference = abs(upper_yield - lower_yield)
            scale = max(abs(upper_yield), abs(lower_yield), 1.0)
            populations[population.name] = {
                "upper_score_event_count": int(np.sum(upper_mask)),
                "lower_score_event_count": int(np.sum(lower_mask)),
                "upper_score_weighted_yield": upper_yield,
                "lower_score_weighted_yield": lower_yield,
                "weighted_yield_absolute_difference": absolute_difference,
                "weighted_yield_relative_difference": absolute_difference / scale,
                "exact_selected_mask_match": bool(
                    not np.any(population_mask & mismatch)
                ),
                "nonboundary_mismatch_count": int(
                    np.sum(population_mask & nonboundary_mismatch)
                ),
                "weighted_yields_agree_with_boundary_tolerance": bool(
                    not np.any(population_mask & nonboundary_mismatch)
                ),
            }
        rows.append(
            {
                "upper_score": upper_scan.score_branch,
                "upper_direction": upper_scan.direction,
                "upper_threshold": upper_cut,
                "lower_score": lower_scan.score_branch,
                "lower_direction": lower_scan.direction,
                "lower_threshold": lower_cut,
                "mismatched_event_count": int(np.sum(mismatch)),
                "nonboundary_mismatch_count": int(np.sum(nonboundary_mismatch)),
                "populations": populations,
            }
        )

    all_pairs_present = len(rows) == len(upper_scan.thresholds)
    all_pairs_agree = all_pairs_present and all(
        row["nonboundary_mismatch_count"] == 0 for row in rows
    )
    score_sum_compatible = (
        max_sum_deviation is not None and max_sum_deviation <= 1e-5
    )
    return {
        "applicable": True,
        "semantics": (
            "For a binary normalized output, score_A > t is compared with "
            "score_B < 1-t. Differences are accepted only at floating-point "
            "boundaries."
        ),
        "score_sum_max_abs_deviation_from_one": max_sum_deviation,
        "boundary_tolerance": boundary_tolerance,
        "score_sum_compatible": score_sum_compatible,
        "all_complementary_pairs_present": all_pairs_present,
        "all_selected_masks_and_weighted_yields_agree": all_pairs_agree,
        "pairs": rows,
    }


def _format_summary(payload: dict[str, Any]) -> str:
    lines = [
        "=== NN Mass-Sculpting Study ===",
        f"Channel: {payload['channel']}",
        "Weight: sample_norm * abs(weight)",
        "Shapes: unit-normalized weighted density over the plotted mass range",
    ]
    for population in payload["populations"]:
        lines.extend(
            [
                f"Population: {population['label']} ({population['name']})",
                f"  Samples: {population['samples']}",
                f"  Excluded samples: {population['exclude_samples']}",
                f"  Sample labels: {population['sample_labels']}",
                f"  Excluded sample labels: {population['exclude_sample_labels']}",
                f"  Truth categories: {population['truth_categories']}",
            ]
        )
    for scan in payload["scans"]:
        lines.extend(
            [
                "",
                f"{scan['population']}: {scan['variable']} vs "
                f"{scan['score_branch']} {scan['direction']} threshold",
                f"  bins: {scan['bins']}",
                f"  range: {scan['plot_range'][0]:.6g} to {scan['plot_range'][1]:.6g} GeV",
                f"  nested selections valid: {scan['nested_selection_valid']}",
            ]
        )
        for row in scan["points"]:
            shape_change = row["max_abs_bin_probability_difference"]
            shape_text = "none" if shape_change is None else f"{shape_change:.6g}"
            lines.append(
                f"  cut {scan['direction']} {row['threshold']:g}: "
                f"N={row['event_count_after_cut']}/{row['event_count_before_cut']}, "
                f"yield={row['weighted_yield_after_cut']:.8g}/"
                f"{row['weighted_yield_before_cut']:.8g}, "
                f"eff={row['weighted_efficiency']:.6g}, shape_diff={shape_text}"
            )
    if payload["summary_scans"]:
        lines.extend(["", "Detailed summary-only scans (not added to plots):"])
        for scan in payload["summary_scans"]:
            lines.extend(
                [
                    "",
                    f"{scan['population']}: {scan['variable']} vs "
                    f"{scan['score_branch']} {scan['direction']} threshold",
                    f"  bins: {scan['bins']}",
                    f"  range: {scan['plot_range'][0]:.6g} to {scan['plot_range'][1]:.6g} GeV",
                    f"  nested selections valid: {scan['nested_selection_valid']}",
                ]
            )
            for row in scan["points"]:
                shape_change = row["max_abs_bin_probability_difference"]
                shape_text = "none" if shape_change is None else f"{shape_change:.6g}"
                lines.append(
                    f"  cut {scan['direction']} {row['threshold']:g}: "
                    f"N={row['event_count_after_cut']}/{row['event_count_before_cut']}, "
                    f"yield={row['weighted_yield_after_cut']:.8g}/"
                    f"{row['weighted_yield_before_cut']:.8g}, "
                    f"eff={row['weighted_efficiency']:.6g}, shape_diff={shape_text}"
                )
    complementarity = payload["binary_complementarity_validation"]
    lines.extend(["", "Binary complementarity validation:"])
    if not complementarity["applicable"]:
        lines.append(f"  not applicable: {complementarity['reason']}")
    else:
        lines.append(
            "  all pairs agree outside floating boundaries: "
            f"{complementarity['all_selected_masks_and_weighted_yields_agree']}"
        )
        for row in complementarity["pairs"]:
            all_selected = row["populations"].get("all_selected_mc", {})
            lines.append(
                f"  {row['upper_score']} > {row['upper_threshold']:g} vs "
                f"{row['lower_score']} < {row['lower_threshold']:g}: "
                f"mismatch={row['mismatched_event_count']}, "
                f"nonboundary={row['nonboundary_mismatch_count']}, "
                f"yield={all_selected.get('upper_score_weighted_yield')} vs "
                f"{all_selected.get('lower_score_weighted_yield')}"
            )
    return "\n".join(lines) + "\n"


def run_mass_sculpting(
    *,
    config: NnStudyConfig,
    dataset: NnDataset,
    plot_dir: Path,
    summary_dir: Path,
    plot_suffix: str,
    selected_score_names: set[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    settings: NnMassSculptingConfig = config.mass_sculpting
    configured_scans = [
        scan
        for scan in settings.scans
        if selected_score_names is None or scan.score_name in selected_score_names
    ]
    configured_summary_scans = [
        scan
        for scan in settings.summary_scans
        if selected_score_names is None or scan.score_name in selected_score_names
    ]
    if selected_score_names is not None and not (
        configured_scans or configured_summary_scans
    ):
        raise ValueError("No mass-sculpting scans match the requested score filter.")
    scans: list[dict[str, Any]] = []
    summary_scans: list[dict[str, Any]] = []
    plots: list[str] = []
    populations: list[dict[str, Any]] = []
    for population in settings.populations:
        population_mask = _population_mask(config, dataset, population)
        populations.append(
            {
                "name": population.name,
                "label": population.label,
                "samples": "all" if population.samples is None else population.samples,
                "exclude_samples": population.exclude_samples,
                "sample_labels": (
                    "all"
                    if population.sample_labels is None
                    else population.sample_labels
                ),
                "exclude_sample_labels": population.exclude_sample_labels,
                "truth_categories": (
                    "all"
                    if population.truth_categories is None
                    else population.truth_categories
                ),
                "selected_event_count": int(np.sum(population_mask)),
            }
        )
        if not configured_scans and not configured_summary_scans:
            continue
        for variable in settings.variables:
            bins = _mass_bins(
                variable,
                np.asarray(
                    dataset.analysis_columns[variable.branch], dtype=np.float64
                ),
                dataset.analysis_weight,
                population_mask,
            )
            for scan in configured_scans:
                if (
                    scan.populations is not None
                    and population.name not in scan.populations
                ):
                    continue
                population_suffix = (
                    "" if population.name == "all_selected_mc" else f"__{population.name}"
                )
                outpath = plot_dir / (
                    f"mass_sculpting__{variable.branch}__{scan.score_branch}"
                    f"{population_suffix}{plot_suffix}"
                )
                scans.append(
                    _scan_payload(
                        config=config,
                        dataset=dataset,
                        variable=variable,
                        scan=scan,
                        population=population,
                        bins=bins,
                        population_mask=population_mask,
                        outpath=outpath,
                    )
                )
                plots.append(str(outpath))
            for scan in configured_summary_scans:
                if (
                    scan.populations is not None
                    and population.name not in scan.populations
                ):
                    continue
                summary_scans.append(
                    _scan_payload(
                        config=config,
                        dataset=dataset,
                        variable=variable,
                        scan=scan,
                        population=population,
                        bins=bins,
                        population_mask=population_mask,
                        outpath=None,
                        make_plot=False,
                    )
                )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "channel": config.channel,
        "populations": populations,
        "population_definition": (
            "Events passing configured sample stitching, NN-study selection, and each "
            "population's optional included/excluded sample and truth filters; finite "
            "score/mass and positive finite analysis weight; restricted to the plotted "
            "mass range."
        ),
        "weighting": "sample_norm * abs(raw event weight)",
        "shape_normalization": (
            "Each curve is independently normalized so its weighted density integrates to 1 "
            "over the plotted mass range when its yield is nonzero."
        ),
        "shape_change_metric": (
            "Maximum absolute bin-by-bin difference between uncut and cut normalized bin probabilities."
        ),
        "scan_status": (
            "configured"
            if configured_scans
            else "disabled: no plotted working points configured"
        ),
        "selected_score_names": (
            "all" if selected_score_names is None else sorted(selected_score_names)
        ),
        "binary_complementarity_validation": _binary_complementarity_validation(
            config=config,
            dataset=dataset,
            settings=settings,
            scans=configured_scans,
        ),
        "scans": scans,
        "summary_scans": summary_scans,
        "plots": plots,
    }
    json_path = summary_dir / "mass_sculpting_summary.json"
    text_path = summary_dir / "mass_sculpting_summary.txt"
    write_json(json_path, payload)
    write_text(text_path, _format_summary(payload))
    return plots, payload
