from __future__ import annotations

from typing import Any

import numpy as np


PREFIX = "y_split_study__"
ARRAY_KEYS = (
    "y_splits",
    "jet_truth_region_yields",
    "jet_truth_candidate_totals",
    "jet_truth_x_totals",
    "jet_process_region_yields",
    "jet_process_candidate_totals",
    "jet_process_x_totals",
    "state_weighted_yields",
    "state_raw_counts",
    "baseline_weighted_yields",
    "baseline_raw_counts",
    "inclusive_weighted_yields",
    "inclusive_raw_counts",
)


def export_y_split_aggregate(
    arrays: dict[str, np.ndarray], aggregate: dict[str, Any]
) -> dict[str, Any]:
    for key in ARRAY_KEYS:
        arrays[f"{PREFIX}{key}"] = np.asarray(aggregate[key])
    return {
        "schema_version": int(aggregate["schema_version"]),
        "config": aggregate["config"],
        "working_points": aggregate["working_points"],
        "process_entries": aggregate["process_entries"],
        "availability": aggregate["availability"],
        "precision": aggregate["precision"],
    }


def load_y_split_aggregate(
    payload: Any, metadata: dict[str, Any]
) -> dict[str, Any] | None:
    scan = metadata.get("y_split_study")
    if scan is None:
        return None
    aggregate = {
        "schema_version": int(scan["schema_version"]),
        "config": dict(scan["config"]),
        "working_points": list(scan["working_points"]),
        "process_entries": list(scan["process_entries"]),
        "availability": dict(scan["availability"]),
        "precision": dict(scan["precision"]),
    }
    for key in ARRAY_KEYS:
        aggregate[key] = np.asarray(payload[f"{PREFIX}{key}"])
    return aggregate
