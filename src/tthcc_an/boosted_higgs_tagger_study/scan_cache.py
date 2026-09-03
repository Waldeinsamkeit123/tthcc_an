from __future__ import annotations

from typing import Any

import numpy as np


HCC_SCAN_PREFIX = "hcc_wp_scan__"


def export_hcc_scan_aggregate(
    arrays: dict[str, np.ndarray],
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    for key in (
        "x_cuts",
        "y_cuts",
        "truth_yields",
        "truth_totals",
        "process_yields",
        "process_totals",
    ):
        arrays[f"{HCC_SCAN_PREFIX}{key}"] = np.asarray(aggregate[key])
    if aggregate.get("event_level_available", False):
        arrays[f"{HCC_SCAN_PREFIX}event_process_yields"] = np.asarray(
            aggregate["event_process_yields"]
        )
        arrays[f"{HCC_SCAN_PREFIX}event_process_totals"] = np.asarray(
            aggregate["event_process_totals"]
        )
    return {
        "schema_version": int(aggregate["schema_version"]),
        "config": aggregate["config"],
        "process_entries": aggregate["process_entries"],
        "event_level_available": bool(aggregate.get("event_level_available", False)),
    }


def load_hcc_scan_aggregate(
    payload: Any,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    scan_metadata = metadata.get("hcc_wp_scan")
    if scan_metadata is None:
        return None
    aggregate = {
        "schema_version": int(scan_metadata["schema_version"]),
        "config": dict(scan_metadata["config"]),
        "process_entries": list(scan_metadata["process_entries"]),
        "event_level_available": bool(scan_metadata.get("event_level_available", False)),
    }
    for key in (
        "x_cuts",
        "y_cuts",
        "truth_yields",
        "truth_totals",
        "process_yields",
        "process_totals",
    ):
        aggregate[key] = np.asarray(payload[f"{HCC_SCAN_PREFIX}{key}"])
    if aggregate["event_level_available"]:
        aggregate["event_process_yields"] = np.asarray(payload[f"{HCC_SCAN_PREFIX}event_process_yields"])
        aggregate["event_process_totals"] = np.asarray(payload[f"{HCC_SCAN_PREFIX}event_process_totals"])
    else:
        aggregate["event_process_yields"] = None
        aggregate["event_process_totals"] = None
    return aggregate
