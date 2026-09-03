from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tthcc_an.boosted_higgs_tagger_study.definitions import (
    TRUTH_LABEL_ORDER,
    build_process_entries_from_pairs,
)


Y_SPLIT_SCHEMA_VERSION = 1
EVENT_STATES = ("hcc_only", "hbb_only", "both", "neither")


class YSplitCacheIncompatibleError(ValueError):
    pass


def normalize_y_split_config(raw: Any) -> dict[str, Any]:
    payload = {} if raw is None else dict(raw)
    scan = dict(payload.get("scan", {}))
    config = {
        "enabled": bool(payload.get("enabled", False)),
        "inclusive_higgs_non_higgs_targets": [
            float(value)
            for value in payload.get(
                "inclusive_higgs_non_higgs_targets",
                payload.get("inclusive_higgs_qcd_targets", [0.01, 0.02]),
            )
        ],
        "y_min": float(scan.get("min", 0.0)),
        "y_max": float(scan.get("max", 1.0)),
        "y_points": int(scan.get("points", 1001)),
        "reference_points": [
            float(value)
            for value in payload.get(
                "reference_points",
                [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
            )
        ],
        "diagnostic_candidates": [
            {
                "y_split": float(item["y_split"]),
                "role": str(item["role"]),
            }
            for item in payload.get(
                "diagnostic_candidates",
                [
                    {"y_split": 0.80, "role": "nominal_candidate_for_further_study"},
                    {"y_split": 0.85, "role": "historical_reference_candidate"},
                ],
            )
        ],
        "inclusive_higgs_result": str(
            payload.get(
                "inclusive_higgs_result",
                "outputs/boosted_higgs_tagger_study_2024_hcc_wp_v1/"
                "summaries/inclusive_higgs_wp.json",
            )
        ),
    }
    targets = config["inclusive_higgs_non_higgs_targets"]
    if not targets or targets != sorted(set(targets)):
        raise ValueError("y_split_study targets must be unique and sorted.")
    if any(value <= 0.0 or value > 1.0 for value in targets):
        raise ValueError("y_split_study targets must lie in (0, 1].")
    if config["y_points"] < 2 or config["y_min"] >= config["y_max"]:
        raise ValueError("y_split_study requires a valid y grid with at least 2 points.")
    references = config["reference_points"]
    if references != sorted(set(references)):
        raise ValueError("y_split_study reference_points must be unique and sorted.")
    if any(value < config["y_min"] or value > config["y_max"] for value in references):
        raise ValueError("y_split_study reference_points must lie inside the y grid.")
    candidate_values = [item["y_split"] for item in config["diagnostic_candidates"]]
    if len(candidate_values) != len(set(candidate_values)):
        raise ValueError("y_split_study diagnostic candidates must be unique.")
    if any(value < config["y_min"] or value > config["y_max"] for value in candidate_values):
        raise ValueError("y_split_study diagnostic candidates must lie inside the y grid.")
    return config


def y_split_grid(config: dict[str, Any]) -> np.ndarray:
    return np.linspace(
        config["y_min"], config["y_max"], config["y_points"], dtype=np.float64
    )


def resolve_inclusive_x_working_points(
    config: dict[str, Any],
    *,
    repo_root: Path,
) -> list[dict[str, float]]:
    path = Path(config["inclusive_higgs_result"])
    if not path.is_absolute():
        path = repo_root / path
    if not path.exists():
        raise YSplitCacheIncompatibleError(
            f"Inclusive Higgs result does not exist: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    recommendations = {
        float(
            row.get(
                "target_non_higgs_jet_efficiency",
                row.get("target_qcd_jet_efficiency"),
            )
        ): row
        for row in payload.get("recommendations", [])
    }
    resolved: list[dict[str, float]] = []
    for target in config["inclusive_higgs_non_higgs_targets"]:
        matches = [
            row for value, row in recommendations.items() if np.isclose(value, target)
        ]
        if len(matches) != 1:
            raise YSplitCacheIncompatibleError(
                f"Inclusive Higgs result has no unique recommendation for target {target:g}."
            )
        resolved.append(
            {
                "non_higgs_target_efficiency": float(target),
                "achieved_non_higgs_jet_efficiency": float(
                    matches[0]["achieved_non_higgs_jet_efficiency"]
                ),
                "x_cut": float(matches[0]["x_cut"]),
            }
        )
    return resolved


def with_resolved_x_working_points(
    config: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    resolved = dict(config)
    resolved["resolved_x_working_points"] = resolve_inclusive_x_working_points(
        config, repo_root=repo_root
    )
    return resolved


def _group_hcc_yields(
    group_codes: np.ndarray,
    weights: np.ndarray,
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    x_cut: float,
    y_splits: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = (
        np.isfinite(x_scores)
        & np.isfinite(y_scores)
        & np.isfinite(weights)
        & (weights >= 0.0)
        & (x_scores >= x_cut)
        & (group_codes >= 0)
        & (group_codes < n_groups)
    )
    endpoint = np.zeros((n_groups, len(y_splits)), dtype=np.float64)
    if np.any(valid):
        indices = np.searchsorted(y_splits, y_scores[valid], side="left")
        in_grid = indices < len(y_splits)
        groups = group_codes[valid][in_grid].astype(np.int64, copy=False)
        flat = groups * len(y_splits) + indices[in_grid]
        endpoint = np.bincount(
            flat,
            weights=weights[valid][in_grid],
            minlength=n_groups * len(y_splits),
        ).reshape(n_groups, len(y_splits))
    hcc = np.cumsum(endpoint, axis=1)
    x_totals = np.sum(endpoint, axis=1)
    return hcc, x_totals


def _group_totals(
    group_codes: np.ndarray, weights: np.ndarray, n_groups: int
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


def _add_range(
    difference: np.ndarray,
    wp: int,
    process: int,
    state: int,
    start: int,
    stop: int,
    value: float,
) -> None:
    if start >= stop:
        return
    difference[wp, process, start, state] += value
    difference[wp, process, stop, state] -= value


def _event_state_scan(
    data: dict[str, np.ndarray],
    working_points: list[dict[str, float]],
    y_splits: np.ndarray,
    n_processes: int,
) -> dict[str, np.ndarray]:
    if "event_index" not in data:
        raise ValueError("y-split event aggregation requires event_index.")
    event_ids = np.asarray(data["event_index"], dtype=np.int64)
    weights = np.asarray(data["weight"], dtype=np.float64)
    process_codes = np.asarray(data["process_code"], dtype=np.int32)
    x_scores = np.asarray(data["gpart_higgs_vs_qcd"], dtype=np.float64)
    y_scores = np.asarray(data["gpart_xbb_vs_xcc"], dtype=np.float64)
    if len({len(event_ids), len(weights), len(process_codes), len(x_scores), len(y_scores)}) != 1:
        raise ValueError("y-split event input arrays have inconsistent lengths.")

    n_wp = len(working_points)
    ny = len(y_splits)
    weighted_diff = np.zeros((n_wp, n_processes, ny + 1, len(EVENT_STATES)))
    raw_diff = np.zeros((n_wp, n_processes, ny + 1, len(EVENT_STATES)), dtype=np.int64)
    baseline_weighted = np.zeros(n_processes)
    baseline_raw = np.zeros(n_processes, dtype=np.int64)
    inclusive_weighted = np.zeros((n_wp, n_processes))
    inclusive_raw = np.zeros((n_wp, n_processes), dtype=np.int64)

    if event_ids.size == 0:
        return {
            "state_weighted_yields": weighted_diff[:, :, :ny],
            "state_raw_counts": raw_diff[:, :, :ny],
            "baseline_weighted_yields": baseline_weighted,
            "baseline_raw_counts": baseline_raw,
            "inclusive_weighted_yields": inclusive_weighted,
            "inclusive_raw_counts": inclusive_raw,
        }

    order = np.argsort(event_ids, kind="stable")
    event_ids = event_ids[order]
    weights = weights[order]
    process_codes = process_codes[order]
    x_scores = x_scores[order]
    y_scores = y_scores[order]
    boundaries = np.r_[0, np.flatnonzero(event_ids[1:] != event_ids[:-1]) + 1, len(event_ids)]

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        process = int(process_codes[start])
        weight = float(weights[start])
        if process < 0 or process >= n_processes or not np.isfinite(weight) or weight < 0.0:
            continue
        if np.any(process_codes[start:stop] != process):
            raise ValueError("One event maps to multiple process codes.")
        if not np.allclose(weights[start:stop], weight, rtol=1.0e-6, atol=1.0e-12):
            raise ValueError("One event has inconsistent repeated jet weights.")
        baseline_weighted[process] += weight
        baseline_raw[process] += 1

        event_x = x_scores[start:stop]
        event_y = y_scores[start:stop]
        for wp_index, wp in enumerate(working_points):
            x_passing = np.isfinite(event_x) & (event_x >= float(wp["x_cut"]))
            if not np.any(x_passing):
                _add_range(weighted_diff, wp_index, process, 3, 0, ny, weight)
                _add_range(raw_diff, wp_index, process, 3, 0, ny, 1)
                continue
            inclusive_weighted[wp_index, process] += weight
            inclusive_raw[wp_index, process] += 1
            passing = x_passing & np.isfinite(event_y)
            if not np.any(passing):
                _add_range(weighted_diff, wp_index, process, 3, 0, ny, weight)
                _add_range(raw_diff, wp_index, process, 3, 0, ny, 1)
                continue
            values = event_y[passing]
            first_hcc = int(np.searchsorted(y_splits, np.min(values), side="left"))
            first_not_hbb = int(np.searchsorted(y_splits, np.max(values), side="left"))
            if first_hcc > 0:
                _add_range(weighted_diff, wp_index, process, 1, 0, first_hcc, weight)
                _add_range(raw_diff, wp_index, process, 1, 0, first_hcc, 1)
            if first_hcc < first_not_hbb:
                _add_range(
                    weighted_diff, wp_index, process, 2, first_hcc, first_not_hbb, weight
                )
                _add_range(raw_diff, wp_index, process, 2, first_hcc, first_not_hbb, 1)
            if first_not_hbb < ny:
                _add_range(
                    weighted_diff, wp_index, process, 0, first_not_hbb, ny, weight
                )
                _add_range(raw_diff, wp_index, process, 0, first_not_hbb, ny, 1)

    return {
        "state_weighted_yields": np.cumsum(weighted_diff, axis=2)[:, :, :ny],
        "state_raw_counts": np.cumsum(raw_diff, axis=2)[:, :, :ny],
        "baseline_weighted_yields": baseline_weighted,
        "baseline_raw_counts": baseline_raw,
        "inclusive_weighted_yields": inclusive_weighted,
        "inclusive_raw_counts": inclusive_raw,
    }


def build_y_split_aggregate(
    data: dict[str, np.ndarray],
    process_entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "truth_code", "process_code", "weight", "event_index",
        "gpart_higgs_vs_qcd", "gpart_xbb_vs_xcc",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError("y-split study is missing arrays: " + ", ".join(missing))
    working_points = [
        {**dict(wp), "x_cut_evaluated": float(wp["x_cut"])}
        for wp in config["resolved_x_working_points"]
    ]
    y_splits = y_split_grid(config)
    truth_codes = np.asarray(data["truth_code"], dtype=np.int32)
    process_codes = np.asarray(data["process_code"], dtype=np.int32)
    weights = np.asarray(data["weight"], dtype=np.float64)
    x_scores = np.asarray(data["gpart_higgs_vs_qcd"], dtype=np.float64)
    y_scores = np.asarray(data["gpart_xbb_vs_xcc"], dtype=np.float64)
    n_truth = len(TRUTH_LABEL_ORDER)
    n_processes = max((int(entry["code"]) for entry in process_entries), default=-1) + 1
    jet_truth = np.zeros((len(working_points), n_truth, len(y_splits), 2))
    jet_process = np.zeros((len(working_points), n_processes, len(y_splits), 2))
    truth_x_totals = np.zeros((len(working_points), n_truth))
    process_x_totals = np.zeros((len(working_points), n_processes))
    for wp_index, wp in enumerate(working_points):
        hcc, truth_x_totals[wp_index] = _group_hcc_yields(
            truth_codes, weights, x_scores, y_scores, wp["x_cut"], y_splits, n_truth
        )
        jet_truth[wp_index, :, :, 0] = hcc
        jet_truth[wp_index, :, :, 1] = truth_x_totals[wp_index, :, None] - hcc
        hcc, process_x_totals[wp_index] = _group_hcc_yields(
            process_codes, weights, x_scores, y_scores, wp["x_cut"], y_splits, n_processes
        )
        jet_process[wp_index, :, :, 0] = hcc
        jet_process[wp_index, :, :, 1] = process_x_totals[wp_index, :, None] - hcc

    event = _event_state_scan(data, working_points, y_splits, n_processes)
    return {
        "schema_version": Y_SPLIT_SCHEMA_VERSION,
        "config": dict(config),
        "working_points": working_points,
        "y_splits": y_splits,
        "process_entries": [dict(entry) for entry in process_entries],
        "jet_truth_region_yields": jet_truth,
        "jet_truth_candidate_totals": _group_totals(truth_codes, weights, n_truth),
        "jet_truth_x_totals": truth_x_totals,
        "jet_process_region_yields": jet_process,
        "jet_process_candidate_totals": _group_totals(process_codes, weights, n_processes),
        "jet_process_x_totals": process_x_totals,
        **event,
        "availability": {
            "jet_weighted": True,
            "event_weighted_complete": True,
            "event_raw_counts": True,
        },
        "precision": {
            "source": "dedicated_y_split_chunk_aggregate",
            "x": "resolved_from_inclusive_higgs_result",
            "y": "exact_on_configured_grid",
        },
    }


def build_y_split_from_legacy_hcc(
    hcc: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    working_points = [
        {**dict(wp), "x_cut_evaluated": float(wp["x_cut"])}
        for wp in config["resolved_x_working_points"]
    ]
    y_splits = y_split_grid(config)
    old_x = np.asarray(hcc["x_cuts"], dtype=np.float64)
    old_y = np.asarray(hcc["y_cuts"], dtype=np.float64)
    x_indices = np.clip(
        np.searchsorted(old_x, [wp["x_cut"] for wp in working_points], side="left"),
        0, len(old_x) - 1,
    )
    working_points = [
        {**wp, "x_cut_evaluated": float(old_x[index])}
        for wp, index in zip(working_points, x_indices)
    ]
    y_indices = np.clip(np.searchsorted(old_y, y_splits, side="right") - 1, 0, len(old_y) - 1)
    truth_hcc = np.asarray(hcc["truth_yields"])[:, x_indices, :]
    truth_hcc = np.transpose(truth_hcc, (1, 0, 2))[:, :, y_indices]
    process_hcc = np.asarray(hcc["process_yields"])[:, x_indices, :]
    process_hcc = np.transpose(process_hcc, (1, 0, 2))[:, :, y_indices]
    truth_x = np.asarray(hcc["truth_yields"])[:, x_indices, -1].T
    process_x = np.asarray(hcc["process_yields"])[:, x_indices, -1].T
    jet_truth = np.stack((truth_hcc, truth_x[:, :, None] - truth_hcc), axis=-1)
    jet_process = np.stack((process_hcc, process_x[:, :, None] - process_hcc), axis=-1)

    n_wp = len(working_points)
    n_process = len(hcc["process_entries"])
    ny = len(y_splits)
    states = np.full((n_wp, n_process, ny, len(EVENT_STATES)), np.nan)
    event_hcc = np.asarray(hcc["event_process_yields"])[:, x_indices, :]
    event_hcc = np.transpose(event_hcc, (1, 0, 2))[:, :, y_indices]
    # Hcc-pass is recoverable, but Hbb and overlap are not; keep states unavailable.
    return {
        "schema_version": Y_SPLIT_SCHEMA_VERSION,
        "config": dict(config),
        "working_points": working_points,
        "y_splits": y_splits,
        "process_entries": [dict(entry) for entry in hcc["process_entries"]],
        "jet_truth_region_yields": jet_truth,
        "jet_truth_candidate_totals": np.asarray(hcc["truth_totals"]),
        "jet_truth_x_totals": truth_x,
        "jet_process_region_yields": jet_process,
        "jet_process_candidate_totals": np.asarray(hcc["process_totals"]),
        "jet_process_x_totals": process_x,
        "state_weighted_yields": states,
        "legacy_event_hcc_pass_yields": event_hcc,
        "state_raw_counts": np.full(states.shape, -1, dtype=np.int64),
        "baseline_weighted_yields": np.asarray(hcc["event_process_totals"]),
        "baseline_raw_counts": np.full(n_process, -1, dtype=np.int64),
        "inclusive_weighted_yields": np.asarray(hcc["event_process_yields"])[:, x_indices, -1].T,
        "inclusive_raw_counts": np.full((n_wp, n_process), -1, dtype=np.int64),
        "availability": {
            "jet_weighted": True,
            "event_weighted_complete": False,
            "event_raw_counts": False,
            "legacy_event_hcc_pass_only": True,
        },
        "precision": {
            "source": "legacy_201x201_hcc_scan",
            "x": "conservative_next_grid_point_step_0.005",
            "y": "conservative_previous_grid_point_step_0.005",
        },
    }


def merge_y_split_aggregates(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    if not aggregates:
        raise ValueError("No y-split aggregates supplied.")
    reference = aggregates[0]
    for aggregate in aggregates[1:]:
        if aggregate["config"] != reference["config"]:
            raise ValueError("Cannot merge y-split aggregates with different config.")
        if not np.array_equal(aggregate["y_splits"], reference["y_splits"]):
            raise ValueError("Cannot merge y-split aggregates with different y grids.")
    labels = {
        entry["process"]: entry.get("label", entry["process"])
        for aggregate in aggregates for entry in aggregate["process_entries"]
    }
    entries = build_process_entries_from_pairs(labels)
    code = {entry["process"]: int(entry["code"]) for entry in entries}
    nwp, _, ny, _ = reference["jet_process_region_yields"].shape
    merged = {
        key: np.sum([np.asarray(a[key]) for a in aggregates], axis=0)
        for key in ("jet_truth_region_yields", "jet_truth_candidate_totals", "jet_truth_x_totals")
    }
    for key, shape in {
        "jet_process_region_yields": (nwp, len(entries), ny, 2),
        "jet_process_candidate_totals": (len(entries),),
        "jet_process_x_totals": (nwp, len(entries)),
        "state_weighted_yields": (nwp, len(entries), ny, len(EVENT_STATES)),
        "state_raw_counts": (nwp, len(entries), ny, len(EVENT_STATES)),
        "baseline_weighted_yields": (len(entries),),
        "baseline_raw_counts": (len(entries),),
        "inclusive_weighted_yields": (nwp, len(entries)),
        "inclusive_raw_counts": (nwp, len(entries)),
    }.items():
        merged[key] = np.zeros(shape, dtype=np.int64 if "raw" in key else np.float64)
    for aggregate in aggregates:
        for entry in aggregate["process_entries"]:
            local = int(entry["code"])
            target = code[entry["process"]]
            for key in (
                "jet_process_region_yields", "jet_process_candidate_totals",
                "jet_process_x_totals", "state_weighted_yields", "state_raw_counts",
                "baseline_weighted_yields", "baseline_raw_counts",
                "inclusive_weighted_yields", "inclusive_raw_counts",
            ):
                axis = 1 if np.asarray(aggregate[key]).ndim >= 2 and key not in (
                    "jet_process_candidate_totals", "baseline_weighted_yields",
                    "baseline_raw_counts",
                ) else 0
                source = np.take(aggregate[key], local, axis=axis)
                index = [slice(None)] * merged[key].ndim
                index[axis] = target
                merged[key][tuple(index)] += source
    return {
        "schema_version": Y_SPLIT_SCHEMA_VERSION,
        "config": dict(reference["config"]),
        "working_points": list(reference["working_points"]),
        "y_splits": np.asarray(reference["y_splits"]),
        "process_entries": entries,
        **merged,
        "availability": dict(reference["availability"]),
        "precision": dict(reference["precision"]),
    }
