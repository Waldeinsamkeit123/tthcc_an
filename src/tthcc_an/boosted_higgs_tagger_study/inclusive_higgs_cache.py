from __future__ import annotations

from typing import Any

import numpy as np


INCLUSIVE_HIGGS_PREFIX = "inclusive_higgs_wp__"


def export_inclusive_higgs_wp_aggregate(
    arrays: dict[str, np.ndarray],
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    for key in (
        "x_cuts",
        "event_x_cuts",
        "truth_yields",
        "truth_totals",
        "process_yields",
        "process_totals",
        "event_process_yields",
        "event_process_totals",
    ):
        arrays[f"{INCLUSIVE_HIGGS_PREFIX}{key}"] = np.asarray(aggregate[key])
    return {
        "schema_version": int(aggregate["schema_version"]),
        "config": aggregate["config"],
        "process_entries": aggregate["process_entries"],
        "precision": aggregate["precision"],
    }


def load_inclusive_higgs_wp_aggregate(
    payload: Any,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    scan_metadata = metadata.get("inclusive_higgs_wp")
    if scan_metadata is None:
        return None
    aggregate = {
        "schema_version": int(scan_metadata["schema_version"]),
        "config": dict(scan_metadata["config"]),
        "process_entries": list(scan_metadata["process_entries"]),
        "precision": dict(scan_metadata["precision"]),
    }
    for key in (
        "x_cuts",
        "event_x_cuts",
        "truth_yields",
        "truth_totals",
        "process_yields",
        "process_totals",
        "event_process_yields",
        "event_process_totals",
    ):
        aggregate[key] = np.asarray(payload[f"{INCLUSIVE_HIGGS_PREFIX}{key}"])
    return aggregate
