from __future__ import annotations

from typing import Any

import numpy as np

from tthcc_an.boosted_higgs_tagger_study.definitions import (
    TRUTH_LABEL_ORDER,
    TRUTH_LABEL_TO_CODE,
    XBB_VS_XCC_REGION_PRESETS,
    build_process_entries_from_pairs,
)


INCLUSIVE_HIGGS_SCHEMA_VERSION = 1
INCLUSIVE_HIGGS_X_SCORE = "gpart_higgs_vs_qcd"
NON_HIGGS_TRUTH_LABELS = tuple(
    label for label in TRUTH_LABEL_ORDER
    if label not in {"hcc_pure", "hbb_pure"}
)
NON_HIGGS_TRUTH_CODES = tuple(TRUTH_LABEL_TO_CODE[label] for label in NON_HIGGS_TRUTH_LABELS)


class InclusiveHiggsCacheIncompatibleError(ValueError):
    pass


def normalize_inclusive_higgs_wp_config(raw: Any) -> dict[str, Any]:
    payload = {} if raw is None else dict(raw)
    legacy_targets = payload.get("qcd_target_efficiencies")
    targets = [
        float(value)
        for value in payload.get(
            "target_efficiencies",
            legacy_targets if legacy_targets is not None else [0.001, 0.005, 0.01, 0.02, 0.05],
        )
    ]
    default_background = (
        "qcd_process_legacy"
        if legacy_targets is not None and "target_efficiencies" not in payload
        else "non_higgs_truth"
    )
    config = {
        "enabled": bool(payload.get("enabled", False)),
        "x_score": str(payload.get("x_score", INCLUSIVE_HIGGS_X_SCORE)),
        "x_min": float(payload.get("x_min", 0.0)),
        "x_max": float(payload.get("x_max", 1.0)),
        "scan_points": int(payload.get("scan_points", 1001)),
        "background_definition": str(
            payload.get("background_definition", default_background)
        ),
        "target_efficiencies": targets,
    }
    if config["x_score"] != INCLUSIVE_HIGGS_X_SCORE:
        raise ValueError(
            "Inclusive Higgs WP currently requires x_score=\'gpart_higgs_vs_qcd\'."
        )
    if config["background_definition"] not in {
        "non_higgs_truth", "qcd_process_legacy"
    }:
        raise ValueError(
            "Inclusive Higgs background_definition must be non_higgs_truth; "
            "qcd_process_legacy is accepted only for historical configs."
        )
    if config["scan_points"] < 2:
        raise ValueError("Inclusive Higgs WP scan_points must be at least 2.")
    if config["x_min"] >= config["x_max"]:
        raise ValueError("Inclusive Higgs WP requires x_min < x_max.")
    if not targets or any(value <= 0.0 or value > 1.0 for value in targets):
        raise ValueError("Inclusive Higgs target efficiencies must lie in (0, 1].")
    if targets != sorted(set(targets)):
        raise ValueError("Inclusive Higgs target efficiencies must be unique and sorted.")
    return config


def inclusive_x_grid(config: dict[str, Any]) -> np.ndarray:
    return np.linspace(
        config["x_min"],
        config["x_max"],
        config["scan_points"],
        dtype=np.float64,
    )


def _weighted_group_x_scan(
    group_codes: np.ndarray,
    weights: np.ndarray,
    x_scores: np.ndarray,
    x_cuts: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    histogram = np.zeros((n_groups, len(x_cuts)), dtype=np.float64)
    valid = (
        np.isfinite(x_scores)
        & np.isfinite(weights)
        & (weights >= 0.0)
        & (group_codes >= 0)
        & (group_codes < n_groups)
    )
    if np.any(valid):
        max_cut_index = np.searchsorted(
            x_cuts,
            x_scores[valid],
            side="right",
        ) - 1
        in_grid = max_cut_index >= 0
        groups = group_codes[valid][in_grid].astype(np.int64, copy=False)
        indices = max_cut_index[in_grid]
        flat = groups * len(x_cuts) + indices
        histogram = np.bincount(
            flat,
            weights=weights[valid][in_grid],
            minlength=n_groups * len(x_cuts),
        ).reshape(n_groups, len(x_cuts))
    return np.cumsum(histogram[:, ::-1], axis=1)[:, ::-1]


def _weighted_group_totals(
    group_codes: np.ndarray,
    weights: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    valid = (
        np.isfinite(weights)
        & (weights >= 0.0)
        & (group_codes >= 0)
        & (group_codes < n_groups)
    )
    return np.bincount(
        group_codes[valid].astype(np.int64, copy=False),
        weights=weights[valid],
        minlength=n_groups,
    ).astype(np.float64, copy=False)


def _event_process_x_scan(
    data: dict[str, np.ndarray],
    x_cuts: np.ndarray,
    n_processes: int,
) -> tuple[np.ndarray, np.ndarray]:
    if "event_index" not in data:
        raise ValueError(
            "event-level information unavailable in legacy cache: event_index is missing"
        )
    event_ids = np.asarray(data["event_index"], dtype=np.int64)
    weights = np.asarray(data["weight"], dtype=np.float64)
    if event_ids.shape != weights.shape:
        raise ValueError("event_index shape does not match jet weights.")

    yields = np.zeros((n_processes, len(x_cuts)), dtype=np.float64)
    totals = np.zeros(n_processes, dtype=np.float64)
    if event_ids.size == 0:
        return yields, totals

    order = np.argsort(event_ids, kind="stable")
    event_ids = event_ids[order]
    scores = np.asarray(data[INCLUSIVE_HIGGS_X_SCORE], dtype=np.float64)[order]
    weights = weights[order]
    process_codes = np.asarray(data["process_code"], dtype=np.int32)[order]
    boundaries = np.r_[0, np.flatnonzero(event_ids[1:] != event_ids[:-1]) + 1, len(event_ids)]
    endpoint_histogram = np.zeros_like(yields)

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        process = int(process_codes[start])
        weight = float(weights[start])
        if process < 0 or process >= n_processes or not np.isfinite(weight) or weight < 0.0:
            continue
        if np.any(process_codes[start:stop] != process):
            raise ValueError("One event maps to multiple process codes.")
        if not np.allclose(weights[start:stop], weight, rtol=1.0e-6, atol=1.0e-12):
            raise ValueError("One event has inconsistent repeated jet weights.")
        totals[process] += weight

        finite_scores = scores[start:stop][np.isfinite(scores[start:stop])]
        if finite_scores.size == 0:
            continue
        max_cut_index = int(
            np.searchsorted(x_cuts, float(np.max(finite_scores)), side="right") - 1
        )
        if max_cut_index >= 0:
            endpoint_histogram[process, max_cut_index] += weight

    return np.cumsum(endpoint_histogram[:, ::-1], axis=1)[:, ::-1], totals


def build_inclusive_higgs_wp_aggregate(
    data: dict[str, np.ndarray],
    process_entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    required = {"truth_code", "process_code", "weight", INCLUSIVE_HIGGS_X_SCORE}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(
            "Inclusive Higgs WP scan is missing required arrays: " + ", ".join(missing)
        )

    x_cuts = inclusive_x_grid(config)
    truth_codes = np.asarray(data["truth_code"], dtype=np.int32)
    process_codes = np.asarray(data["process_code"], dtype=np.int32)
    weights = np.asarray(data["weight"], dtype=np.float64)
    scores = np.asarray(data[INCLUSIVE_HIGGS_X_SCORE], dtype=np.float64)
    if len({len(value) for value in (truth_codes, process_codes, weights, scores)}) != 1:
        raise ValueError("Inclusive Higgs WP input arrays have inconsistent lengths.")

    n_truth = len(TRUTH_LABEL_ORDER)
    n_processes = max((int(entry["code"]) for entry in process_entries), default=-1) + 1
    event_yields, event_totals = _event_process_x_scan(data, x_cuts, n_processes)
    return {
        "schema_version": INCLUSIVE_HIGGS_SCHEMA_VERSION,
        "config": dict(config),
        "x_cuts": x_cuts,
        "event_x_cuts": x_cuts.copy(),
        "truth_yields": _weighted_group_x_scan(
            truth_codes, weights, scores, x_cuts, n_truth
        ),
        "truth_totals": _weighted_group_totals(truth_codes, weights, n_truth),
        "process_entries": [dict(entry) for entry in process_entries],
        "process_yields": _weighted_group_x_scan(
            process_codes, weights, scores, x_cuts, n_processes
        ),
        "process_totals": _weighted_group_totals(
            process_codes, weights, n_processes
        ),
        "event_process_yields": event_yields,
        "event_process_totals": event_totals,
        "precision": {
            "source": "dedicated_x_only_chunk_aggregate",
            "jet_scan": "exact_on_configured_grid",
            "event_scan": "exact_on_configured_grid",
            "event_values_are_coarse": False,
        },
    }


def merge_inclusive_higgs_wp_aggregates(
    aggregates: list[dict[str, Any]],
) -> dict[str, Any]:
    if not aggregates:
        raise ValueError("No Inclusive Higgs WP aggregates were supplied.")
    reference = aggregates[0]
    for aggregate in aggregates[1:]:
        if aggregate["config"] != reference["config"]:
            raise ValueError("Cannot merge Inclusive Higgs WP scans with different configs.")
        if not np.array_equal(aggregate["x_cuts"], reference["x_cuts"]):
            raise ValueError("Cannot merge Inclusive Higgs WP scans with different x grids.")
        if not np.array_equal(aggregate["event_x_cuts"], reference["event_x_cuts"]):
            raise ValueError("Cannot merge Inclusive Higgs WP scans with different event x grids.")

    labels = {
        entry["process"]: entry.get("label", entry["process"])
        for aggregate in aggregates
        for entry in aggregate["process_entries"]
    }
    process_entries = build_process_entries_from_pairs(labels)
    code_by_process = {entry["process"]: int(entry["code"]) for entry in process_entries}
    jet_shape = (len(process_entries), len(reference["x_cuts"]))
    event_shape = (len(process_entries), len(reference["event_x_cuts"]))
    process_yields = np.zeros(jet_shape, dtype=np.float64)
    process_totals = np.zeros(len(process_entries), dtype=np.float64)
    event_yields = np.zeros(event_shape, dtype=np.float64)
    event_totals = np.zeros(len(process_entries), dtype=np.float64)
    truth_yields = np.zeros_like(reference["truth_yields"], dtype=np.float64)
    truth_totals = np.zeros_like(reference["truth_totals"], dtype=np.float64)

    for aggregate in aggregates:
        truth_yields += np.asarray(aggregate["truth_yields"], dtype=np.float64)
        truth_totals += np.asarray(aggregate["truth_totals"], dtype=np.float64)
        for entry in aggregate["process_entries"]:
            local_code = int(entry["code"])
            global_code = code_by_process[entry["process"]]
            process_yields[global_code] += aggregate["process_yields"][local_code]
            process_totals[global_code] += aggregate["process_totals"][local_code]
            event_yields[global_code] += aggregate["event_process_yields"][local_code]
            event_totals[global_code] += aggregate["event_process_totals"][local_code]

    return {
        "schema_version": INCLUSIVE_HIGGS_SCHEMA_VERSION,
        "config": dict(reference["config"]),
        "x_cuts": np.asarray(reference["x_cuts"]),
        "event_x_cuts": np.asarray(reference["event_x_cuts"]),
        "truth_yields": truth_yields,
        "truth_totals": truth_totals,
        "process_entries": process_entries,
        "process_yields": process_yields,
        "process_totals": process_totals,
        "event_process_yields": event_yields,
        "event_process_totals": event_totals,
        "precision": dict(reference["precision"]),
    }


def _histogram_scan(
    histogram: np.ndarray,
    edges: np.ndarray,
    x_cuts: np.ndarray,
) -> np.ndarray:
    cumulative = np.cumsum(np.asarray(histogram)[..., ::-1], axis=-1)[..., ::-1]
    indices = np.searchsorted(edges, x_cuts, side="right") - 1
    indices = np.clip(indices, 0, cumulative.shape[-1] - 1)
    return cumulative[..., indices]


def build_inclusive_higgs_wp_from_existing_histogram(
    histogram_payload: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    hcc_aggregate = histogram_payload.get("hcc_wp_scan")
    score_payload = histogram_payload.get("score_histograms", {}).get(
        INCLUSIVE_HIGGS_X_SCORE
    )
    if hcc_aggregate is None or score_payload is None:
        raise InclusiveHiggsCacheIncompatibleError(
            "Existing cache lacks the x score histogram or Hcc event aggregate needed "
            "for the Inclusive Higgs coarse diagnostic."
        )
    if not hcc_aggregate.get("event_level_available", False):
        raise InclusiveHiggsCacheIncompatibleError(
            "event-level information unavailable in legacy cache"
        )

    x_cuts = inclusive_x_grid(config)
    edges = np.asarray(histogram_payload["hist_edges"], dtype=np.float64)
    truth_yields = _histogram_scan(
        np.asarray(score_payload["weight_truth"], dtype=np.float64),
        edges,
        x_cuts,
    )
    process_truth_hist = np.asarray(
        score_payload["weight_process_truth"], dtype=np.float64
    )
    process_yields = _histogram_scan(
        np.sum(process_truth_hist, axis=1),
        edges,
        x_cuts,
    )
    event_x_cuts = np.asarray(hcc_aggregate["x_cuts"], dtype=np.float64)
    y_cuts = np.asarray(hcc_aggregate["y_cuts"], dtype=np.float64)
    inclusive_y_index = int(np.argmax(y_cuts))

    return {
        "schema_version": INCLUSIVE_HIGGS_SCHEMA_VERSION,
        "config": dict(config),
        "x_cuts": x_cuts,
        "event_x_cuts": event_x_cuts,
        "truth_yields": truth_yields,
        "truth_totals": np.asarray(hcc_aggregate["truth_totals"], dtype=np.float64),
        "process_entries": [dict(entry) for entry in hcc_aggregate["process_entries"]],
        "process_yields": process_yields,
        "process_totals": np.asarray(hcc_aggregate["process_totals"], dtype=np.float64),
        "event_process_yields": np.asarray(
            hcc_aggregate["event_process_yields"][:, :, inclusive_y_index],
            dtype=np.float64,
        ),
        "event_process_totals": np.asarray(
            hcc_aggregate["event_process_totals"], dtype=np.float64
        ),
        "precision": {
            "source": "existing_2000_bin_x_histogram_plus_201_point_hcc_event_grid_at_y_1",
            "jet_scan": f"histogram_bin_width_{float(np.max(np.diff(edges))):.8g}",
            "event_scan": f"coarse_grid_step_{float(np.max(np.diff(event_x_cuts))):.8g}",
            "event_values_are_coarse": True,
            "no_y_requirement": (
                "The y=1 slice is used only for x>0 passing yields; no y split is applied. "
                "Events with undefined y necessarily have Xbb+Xcc=0 and cannot pass x>0."
            ),
        },
    }


def _safe_ratio(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    result = np.zeros(
        np.broadcast_shapes(numerator_array.shape, denominator_array.shape),
        dtype=np.float64,
    )
    np.divide(
        numerator_array,
        denominator_array,
        out=result,
        where=denominator_array > 0.0,
    )
    return result


def _assert_nonincreasing(name: str, values: np.ndarray) -> None:
    scale = max(float(np.max(np.abs(values))), 1.0)
    if np.any(np.diff(values) > 1.0e-10 * scale):
        raise ValueError(f"{name} is not monotonically non-increasing with x_cut.")


def historical_x_references() -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for preset, payload in XBB_VS_XCC_REGION_PRESETS.items():
        for flavor in ("hcc", "hbb"):
            references.append(
                {
                    "label": f"historical {preset} {flavor.upper()} region",
                    "preset": preset,
                    "flavor": flavor,
                    "x_cut": float(payload[f"{flavor}_x_cut"]),
                    "not_an_inclusive_wp": True,
                }
            )
    return references


def derive_inclusive_higgs_wp_results(
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    process_code = {
        entry["process"]: int(entry["code"]) for entry in aggregate["process_entries"]
    }
    required_processes = {"ttHcc", "ttHbb", "qcd"}
    missing = sorted(required_processes - process_code.keys())
    if missing:
        raise ValueError(
            "Inclusive Higgs WP is missing required processes: " + ", ".join(missing)
        )

    x_cuts = np.asarray(aggregate["x_cuts"], dtype=np.float64)
    truth_efficiencies = {
        label: _safe_ratio(
            aggregate["truth_yields"][TRUTH_LABEL_TO_CODE[label]],
            aggregate["truth_totals"][TRUTH_LABEL_TO_CODE[label]],
        )
        for label in TRUTH_LABEL_ORDER
    }
    hcc_code = TRUTH_LABEL_TO_CODE["hcc_pure"]
    hbb_code = TRUTH_LABEL_TO_CODE["hbb_pure"]
    pure_higgs_yield = (
        aggregate["truth_yields"][hcc_code] + aggregate["truth_yields"][hbb_code]
    )
    pure_higgs_total = float(
        aggregate["truth_totals"][hcc_code] + aggregate["truth_totals"][hbb_code]
    )
    inclusive_pure_higgs_efficiency = _safe_ratio(
        pure_higgs_yield, pure_higgs_total
    )
    process_efficiencies = {
        entry["process"]: _safe_ratio(
            aggregate["process_yields"][int(entry["code"])],
            aggregate["process_totals"][int(entry["code"])],
        )
        for entry in aggregate["process_entries"]
    }
    qcd_efficiency = process_efficiencies["qcd"]
    non_higgs_yield = np.sum(
        np.asarray(aggregate["truth_yields"])[list(NON_HIGGS_TRUTH_CODES)], axis=0
    )
    non_higgs_total = float(
        np.sum(np.asarray(aggregate["truth_totals"])[list(NON_HIGGS_TRUTH_CODES)])
    )
    non_higgs_efficiency = _safe_ratio(non_higgs_yield, non_higgs_total)

    event_x_cuts = np.asarray(aggregate["event_x_cuts"], dtype=np.float64)
    event_indices = np.searchsorted(event_x_cuts, x_cuts, side="left")
    event_indices = np.clip(event_indices, 0, len(event_x_cuts) - 1)
    event_evaluated_x_cuts = event_x_cuts[event_indices]
    event_yields = {
        entry["process"]: np.asarray(
            aggregate["event_process_yields"][int(entry["code"])][event_indices],
            dtype=np.float64,
        )
        for entry in aggregate["process_entries"]
    }
    event_baselines = {
        entry["process"]: float(
            aggregate["event_process_totals"][int(entry["code"])]
        )
        for entry in aggregate["process_entries"]
    }
    event_efficiencies = {
        process: _safe_ratio(yields, event_baselines[process])
        for process, yields in event_yields.items()
    }
    background_processes = [
        process for process in process_code if process not in {"ttHcc", "ttHbb"}
    ]
    total_background_yield = np.sum(
        [event_yields[process] for process in background_processes],
        axis=0,
    )
    total_background_baseline = float(
        sum(event_baselines[process] for process in background_processes)
    )

    for name, values in {
        "non-Higgs truth jet efficiency": non_higgs_efficiency,
        "QCD process jet efficiency": qcd_efficiency,
        "ttHcc event efficiency": event_efficiencies["ttHcc"],
        "ttHbb event efficiency": event_efficiencies["ttHbb"],
    }.items():
        _assert_nonincreasing(name, np.asarray(values))

    result = {
        "x_cuts": x_cuts,
        "event_evaluated_x_cuts": event_evaluated_x_cuts,
        "truth_efficiencies": truth_efficiencies,
        "truth_yields": np.asarray(aggregate["truth_yields"], dtype=np.float64),
        "truth_totals": np.asarray(aggregate["truth_totals"], dtype=np.float64),
        "inclusive_pure_higgs_efficiency": inclusive_pure_higgs_efficiency,
        "process_efficiencies": process_efficiencies,
        "non_higgs_truth_labels": list(NON_HIGGS_TRUTH_LABELS),
        "non_higgs_jet_efficiency": non_higgs_efficiency,
        "non_higgs_jet_rejection_fraction": 1.0 - non_higgs_efficiency,
        "non_higgs_jet_inverse_efficiency": np.divide(
            1.0,
            non_higgs_efficiency,
            out=np.full_like(non_higgs_efficiency, np.inf),
            where=non_higgs_efficiency > 0.0,
        ),
        "qcd_process_jet_efficiency": qcd_efficiency,
        "qcd_process_jet_rejection_fraction": 1.0 - qcd_efficiency,
        "qcd_process_jet_inverse_efficiency": np.divide(
            1.0,
            qcd_efficiency,
            out=np.full_like(qcd_efficiency, np.inf),
            where=qcd_efficiency > 0.0,
        ),
        "event_yields": event_yields,
        "event_efficiencies": event_efficiencies,
        "event_baselines": event_baselines,
        "total_background_event_yield": total_background_yield,
        "total_background_event_efficiency": _safe_ratio(
            total_background_yield, total_background_baseline
        ),
        "total_background_event_baseline": total_background_baseline,
        "precision": dict(aggregate["precision"]),
        "config": dict(aggregate["config"]),
    }

    background_definition = str(aggregate["config"]["background_definition"])
    selection_efficiency = (
        non_higgs_efficiency
        if background_definition == "non_higgs_truth"
        else qcd_efficiency
    )
    result["wp_background_definition"] = background_definition
    result["wp_background_efficiency"] = selection_efficiency

    recommendations: list[dict[str, Any]] = []
    for target in aggregate["config"]["target_efficiencies"]:
        allowed = np.flatnonzero(selection_efficiency <= float(target))
        if allowed.size == 0:
            raise ValueError(
                f"No x cut satisfies requested background efficiency {target:g}."
            )
        index = int(allowed[0])
        row: dict[str, Any] = {
            "background_definition": background_definition,
            "target_non_higgs_jet_efficiency": float(target),
            "achieved_non_higgs_jet_efficiency": float(non_higgs_efficiency[index]),
            "achieved_minus_requested_non_higgs_efficiency": float(
                non_higgs_efficiency[index] - target
            ),
            "qcd_process_jet_efficiency": float(qcd_efficiency[index]),
            "x_cut": float(x_cuts[index]),
            "event_x_cut_evaluated": float(event_evaluated_x_cuts[index]),
            "event_values_are_coarse": bool(
                aggregate["precision"]["event_values_are_coarse"]
            ),
            "hcc_pure_jet_efficiency": float(
                truth_efficiencies["hcc_pure"][index]
            ),
            "hbb_pure_jet_efficiency": float(
                truth_efficiencies["hbb_pure"][index]
            ),
            "inclusive_pure_higgs_jet_efficiency": float(
                inclusive_pure_higgs_efficiency[index]
            ),
            "total_background_event_yield": float(total_background_yield[index]),
            "total_background_event_efficiency": float(
                result["total_background_event_efficiency"][index]
            ),
        }
        for process in process_code:
            row[f"{process}_event_yield"] = float(event_yields[process][index])
            row[f"{process}_event_efficiency"] = float(
                event_efficiencies[process][index]
            )
        for process in ("ttHcc", "ttHbb", "qcd"):
            row[f"relative_{process}_yield_vs_candidate_baseline"] = float(
                event_efficiencies[process][index]
            )
        recommendations.append(row)
    result["recommendations"] = recommendations
    return result
