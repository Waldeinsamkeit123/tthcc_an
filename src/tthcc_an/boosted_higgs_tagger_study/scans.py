from __future__ import annotations

from typing import Any

import numpy as np

from tthcc_an.boosted_higgs_tagger_study.definitions import (
    TRUTH_LABEL_ORDER,
    TRUTH_LABEL_TO_CODE,
    XBB_VS_XCC_REGION_PRESETS,
    build_process_entries_from_pairs,
)


HCC_SIGNAL_LABEL = "hcc_pure"
HCC_BACKGROUND_LABELS = [
    "hbb_pure",
    "hcc_contaminated",
    "hcc_partial",
    "hbb_contaminated",
    "hbb_partial",
    "top",
    "other",
]
HCC_SCAN_SCHEMA_VERSION = 1


class EventLevelUnavailableError(ValueError):
    pass


class HccScanCacheIncompatibleError(ValueError):
    pass


def normalize_hcc_wp_scan_config(raw: Any) -> dict[str, Any]:
    payload = {} if raw is None else dict(raw)
    enabled = bool(payload.get("enabled", False))
    config = {
        "enabled": enabled,
        "x_min": float(payload.get("x_min", 0.0)),
        "x_max": float(payload.get("x_max", 1.0)),
        "x_points": int(payload.get("x_points", 201)),
        "y_min": float(payload.get("y_min", 0.0)),
        "y_max": float(payload.get("y_max", 1.0)),
        "y_points": int(payload.get("y_points", 201)),
        "background_efficiency_constraints": [
            float(value)
            for value in payload.get(
                "background_efficiency_constraints",
                [0.001, 0.005, 0.01, 0.02, 0.05],
            )
        ],
        "event_level_enabled": bool(payload.get("event_level_enabled", False)),
        "event_signal_processes": [
            str(value) for value in payload.get("event_signal_processes", ["ttHcc"])
        ],
    }
    for axis in ("x", "y"):
        if config[f"{axis}_points"] < 2:
            raise ValueError(f"Hcc WP scan {axis}_points must be at least 2.")
        if config[f"{axis}_min"] >= config[f"{axis}_max"]:
            raise ValueError(f"Hcc WP scan requires {axis}_min < {axis}_max.")
    constraints = config["background_efficiency_constraints"]
    if not constraints or any(value < 0.0 or value > 1.0 for value in constraints):
        raise ValueError("Hcc WP background-efficiency constraints must lie in [0, 1].")
    if config["event_level_enabled"] and not config["event_signal_processes"]:
        raise ValueError("event_signal_processes cannot be empty when event-level scan is enabled.")
    return config


def scan_grid(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.linspace(config["x_min"], config["x_max"], config["x_points"], dtype=np.float64),
        np.linspace(config["y_min"], config["y_max"], config["y_points"], dtype=np.float64),
    )


def candidate_selection_metadata(args: Any) -> dict[str, Any]:
    uses_mass_window = args.candidate_strategy in {
        "mass_window_all_jets",
        "mass_window_leading_pt",
    }
    return {
        "candidate_strategy": str(args.candidate_strategy),
        "pt_min": float(args.pt_min),
        "pt_convention": ">=",
        "eta_abs_max": float(args.eta_max),
        "eta_convention": "<=",
        "mass_variable": "msoftdrop" if uses_mass_window else None,
        "mass_window_low": float(args.msd_window_low) if uses_mass_window else None,
        "mass_window_high": float(args.msd_window_high) if uses_mass_window else None,
        "mass_window_convention": (
            "closed" if bool(args.msd_window_inclusive) else "open"
        )
        if uses_mass_window
        else None,
    }


def validate_hcc_scan_provenance(
    weighting_info: dict[str, Any],
    args: Any,
) -> None:
    expected = candidate_selection_metadata(args)
    actual = weighting_info.get("candidate_selection")
    if actual is None:
        raise HccScanCacheIncompatibleError(
            "Hcc WP scan unavailable: legacy cache has no candidate-selection metadata. "
            "The historical 100-150 GeV cache cannot recover the required 50-200 GeV jets; "
            "run a new ROOT scan or merge new chunk payloads."
        )
    mismatches = [
        key for key, value in expected.items() if actual.get(key) != value
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: cache={actual.get(key)!r}, requested={expected[key]!r}"
            for key in mismatches
        )
        raise HccScanCacheIncompatibleError(
            "Hcc WP scan candidate preselection does not match the cache: " + details
        )


def _score_cells(
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    x_cuts: np.ndarray,
    y_cuts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(x_scores) & np.isfinite(y_scores)
    ix_max = np.searchsorted(x_cuts, x_scores[valid], side="right") - 1
    iy_min = np.searchsorted(y_cuts, y_scores[valid], side="left")
    in_grid = (ix_max >= 0) & (iy_min < len(y_cuts))
    valid_indices = np.flatnonzero(valid)[in_grid]
    return (
        valid_indices,
        ix_max[in_grid].astype(np.int32, copy=False),
        iy_min[in_grid].astype(np.int32, copy=False),
    )


def _weighted_group_scan(
    group_codes: np.ndarray,
    weights: np.ndarray,
    valid_indices: np.ndarray,
    ix_max: np.ndarray,
    iy_min: np.ndarray,
    n_groups: int,
    nx: int,
    ny: int,
) -> np.ndarray:
    hist = np.zeros((n_groups, nx, ny), dtype=np.float64)
    if valid_indices.size:
        groups = group_codes[valid_indices].astype(np.int64, copy=False)
        valid_group = (
            (groups >= 0)
            & (groups < n_groups)
            & np.isfinite(weights[valid_indices])
            & (weights[valid_indices] >= 0.0)
        )
        flat = (groups[valid_group] * nx + ix_max[valid_group]) * ny + iy_min[valid_group]
        hist = np.bincount(
            flat,
            weights=weights[valid_indices][valid_group],
            minlength=n_groups * nx * ny,
        ).reshape(n_groups, nx, ny)
    x_integrated = np.cumsum(hist[:, ::-1, :], axis=1)[:, ::-1, :]
    return np.cumsum(x_integrated, axis=2)


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


def _event_process_scan(
    data: dict[str, np.ndarray],
    x_cuts: np.ndarray,
    y_cuts: np.ndarray,
    n_processes: int,
) -> tuple[np.ndarray, np.ndarray]:
    if "event_index" not in data:
        raise EventLevelUnavailableError(
            "event-level information unavailable in legacy cache: event_index is missing"
        )
    event_ids = np.asarray(data["event_index"], dtype=np.int64)
    if event_ids.shape != np.asarray(data["weight"]).shape:
        raise EventLevelUnavailableError(
            "event-level information unavailable: event_index shape does not match jet arrays"
        )
    if event_ids.size == 0:
        return (
            np.zeros((n_processes, len(x_cuts), len(y_cuts)), dtype=np.float64),
            np.zeros(n_processes, dtype=np.float64),
        )

    order = np.argsort(event_ids, kind="stable")
    event_ids = event_ids[order]
    x_scores = np.asarray(data["gpart_higgs_vs_qcd"], dtype=np.float64)[order]
    y_scores = np.asarray(data["gpart_xbb_vs_xcc"], dtype=np.float64)[order]
    weights = np.asarray(data["weight"], dtype=np.float64)[order]
    process_codes = np.asarray(data["process_code"], dtype=np.int32)[order]
    boundaries = np.r_[0, np.flatnonzero(event_ids[1:] != event_ids[:-1]) + 1, len(event_ids)]

    nx = len(x_cuts)
    ny = len(y_cuts)
    difference = np.zeros((n_processes, nx + 1, ny + 1), dtype=np.float64)
    totals = np.zeros(n_processes, dtype=np.float64)

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        process = int(process_codes[start])
        weight = float(weights[start])
        if process < 0 or process >= n_processes or not np.isfinite(weight) or weight < 0.0:
            continue
        if np.any(process_codes[start:stop] != process):
            raise EventLevelUnavailableError("One event maps to multiple process codes.")
        if not np.allclose(weights[start:stop], weight, rtol=1.0e-6, atol=1.0e-12):
            raise EventLevelUnavailableError("One event has inconsistent repeated jet weights.")
        totals[process] += weight

        valid = np.isfinite(x_scores[start:stop]) & np.isfinite(y_scores[start:stop])
        if not np.any(valid):
            continue
        event_x = x_scores[start:stop][valid]
        event_y = y_scores[start:stop][valid]
        ix = np.searchsorted(x_cuts, event_x, side="right") - 1
        iy = np.searchsorted(y_cuts, event_y, side="left")
        in_grid = (ix >= 0) & (iy < ny)
        ix = ix[in_grid]
        iy = iy[in_grid]
        if ix.size == 0:
            continue

        jet_order = np.argsort(iy, kind="stable")
        ix = ix[jet_order]
        iy = iy[jet_order]
        running_max = -1
        cursor = 0
        while cursor < len(ix):
            y_start = int(iy[cursor])
            next_cursor = cursor + 1
            while next_cursor < len(ix) and iy[next_cursor] == y_start:
                next_cursor += 1
            new_max = max(running_max, int(np.max(ix[cursor:next_cursor])))
            if new_max > running_max:
                x_low = running_max + 1
                x_high = new_max
                difference[process, x_low, y_start] += weight
                difference[process, x_high + 1, y_start] -= weight
                difference[process, x_low, ny] -= weight
                difference[process, x_high + 1, ny] += weight
                running_max = new_max
            cursor = next_cursor

    yields = np.cumsum(np.cumsum(difference, axis=1), axis=2)
    return yields[:, :nx, :ny], totals


def build_hcc_wp_scan_aggregate(
    data: dict[str, np.ndarray],
    process_entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "truth_code",
        "process_code",
        "weight",
        "gpart_higgs_vs_qcd",
        "gpart_xbb_vs_xcc",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError("Hcc WP scan is missing required arrays: " + ", ".join(missing))

    x_cuts, y_cuts = scan_grid(config)
    truth_codes = np.asarray(data["truth_code"], dtype=np.int32)
    process_codes = np.asarray(data["process_code"], dtype=np.int32)
    weights = np.asarray(data["weight"], dtype=np.float64)
    x_scores = np.asarray(data["gpart_higgs_vs_qcd"], dtype=np.float64)
    y_scores = np.asarray(data["gpart_xbb_vs_xcc"], dtype=np.float64)
    sizes = {len(value) for value in (truth_codes, process_codes, weights, x_scores, y_scores)}
    if len(sizes) != 1:
        raise ValueError("Hcc WP scan input arrays have inconsistent lengths.")

    n_truth = len(TRUTH_LABEL_ORDER)
    n_processes = max((int(entry["code"]) for entry in process_entries), default=-1) + 1
    valid_indices, ix_max, iy_min = _score_cells(x_scores, y_scores, x_cuts, y_cuts)
    truth_yields = _weighted_group_scan(
        truth_codes,
        weights,
        valid_indices,
        ix_max,
        iy_min,
        n_truth,
        len(x_cuts),
        len(y_cuts),
    )
    process_yields = _weighted_group_scan(
        process_codes,
        weights,
        valid_indices,
        ix_max,
        iy_min,
        n_processes,
        len(x_cuts),
        len(y_cuts),
    )
    event_yields: np.ndarray | None = None
    event_totals: np.ndarray | None = None
    if config["event_level_enabled"]:
        event_yields, event_totals = _event_process_scan(
            data,
            x_cuts,
            y_cuts,
            n_processes,
        )

    return {
        "schema_version": HCC_SCAN_SCHEMA_VERSION,
        "config": dict(config),
        "x_cuts": x_cuts,
        "y_cuts": y_cuts,
        "truth_yields": truth_yields,
        "truth_totals": _weighted_group_totals(truth_codes, weights, n_truth),
        "process_entries": [dict(entry) for entry in process_entries],
        "process_yields": process_yields,
        "process_totals": _weighted_group_totals(process_codes, weights, n_processes),
        "event_process_yields": event_yields,
        "event_process_totals": event_totals,
        "event_level_available": event_yields is not None,
    }


def merge_hcc_wp_scan_aggregates(
    aggregates: list[dict[str, Any]],
) -> dict[str, Any]:
    if not aggregates:
        raise ValueError("No Hcc WP scan aggregates were supplied.")
    reference = aggregates[0]
    x_cuts = np.asarray(reference["x_cuts"], dtype=np.float64)
    y_cuts = np.asarray(reference["y_cuts"], dtype=np.float64)
    for aggregate in aggregates[1:]:
        if not np.array_equal(x_cuts, aggregate["x_cuts"]) or not np.array_equal(
            y_cuts, aggregate["y_cuts"]
        ):
            raise ValueError("Cannot merge Hcc WP scans with different grids.")

    labels = {
        entry["process"]: entry.get("label", entry["process"])
        for aggregate in aggregates
        for entry in aggregate["process_entries"]
    }
    process_entries = build_process_entries_from_pairs(labels)
    code_by_process = {entry["process"]: int(entry["code"]) for entry in process_entries}
    shape = (len(process_entries), len(x_cuts), len(y_cuts))
    process_yields = np.zeros(shape, dtype=np.float64)
    process_totals = np.zeros(len(process_entries), dtype=np.float64)
    event_requested = bool(reference["config"]["event_level_enabled"])
    event_yields = np.zeros(shape, dtype=np.float64) if event_requested else None
    event_totals = np.zeros(len(process_entries), dtype=np.float64) if event_requested else None
    truth_yields = np.zeros_like(reference["truth_yields"], dtype=np.float64)
    truth_totals = np.zeros_like(reference["truth_totals"], dtype=np.float64)

    for aggregate in aggregates:
        truth_yields += np.asarray(aggregate["truth_yields"], dtype=np.float64)
        truth_totals += np.asarray(aggregate["truth_totals"], dtype=np.float64)
        local_by_code = {
            int(entry["code"]): entry["process"] for entry in aggregate["process_entries"]
        }
        for local_code, process in local_by_code.items():
            global_code = code_by_process[process]
            process_yields[global_code] += aggregate["process_yields"][local_code]
            process_totals[global_code] += aggregate["process_totals"][local_code]
            if event_requested:
                if not aggregate.get("event_level_available", False):
                    raise EventLevelUnavailableError(
                        "event-level information unavailable in one or more chunk payloads"
                    )
                event_yields[global_code] += aggregate["event_process_yields"][local_code]
                event_totals[global_code] += aggregate["event_process_totals"][local_code]

    return {
        "schema_version": HCC_SCAN_SCHEMA_VERSION,
        "config": dict(reference["config"]),
        "x_cuts": x_cuts,
        "y_cuts": y_cuts,
        "truth_yields": truth_yields,
        "truth_totals": truth_totals,
        "process_entries": process_entries,
        "process_yields": process_yields,
        "process_totals": process_totals,
        "event_process_yields": event_yields,
        "event_process_totals": event_totals,
        "event_level_available": event_requested,
    }


def _safe_ratio(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    result = np.zeros(np.broadcast_shapes(numerator_array.shape, denominator_array.shape), dtype=np.float64)
    np.divide(numerator_array, denominator_array, out=result, where=denominator_array > 0.0)
    return result


def _significance(signal: np.ndarray, background: np.ndarray) -> np.ndarray:
    return _safe_ratio(signal, np.sqrt(np.maximum(signal + background, 0.0)))


def _process_efficiencies(aggregate: dict[str, Any], *, event: bool) -> dict[str, np.ndarray]:
    yields_key = "event_process_yields" if event else "process_yields"
    totals_key = "event_process_totals" if event else "process_totals"
    yields = aggregate[yields_key]
    totals = aggregate[totals_key]
    if yields is None or totals is None:
        return {}
    return {
        entry["process"]: _safe_ratio(yields[int(entry["code"])], totals[int(entry["code"])])
        for entry in aggregate["process_entries"]
    }


def _best_index(metric: np.ndarray, allowed: np.ndarray | None = None) -> tuple[int, int] | None:
    mask = np.ones(metric.shape, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
    if not np.any(mask):
        return None
    ranked = np.where(mask, metric, -np.inf)
    return tuple(int(value) for value in np.unravel_index(np.argmax(ranked), ranked.shape))


def _point_payload(
    label: str,
    index: tuple[int, int],
    result: dict[str, Any],
) -> dict[str, Any]:
    ix, iy = index
    point = {
        "label": label,
        "x_cut": float(result["x_cuts"][ix]),
        "y_cut": float(result["y_cuts"][iy]),
        "S": float(result["signal"][ix, iy]),
        "B": float(result["background"][ix, iy]),
        "S_over_B": float(result["signal_over_background"][ix, iy]),
        "S_over_sqrt_S_plus_B": float(result["significance"][ix, iy]),
        "S_over_sqrt_B": float(result["signal_over_sqrt_background"][ix, iy]),
        "relative_significance": float(result["relative_significance"][ix, iy]),
        result["signal_efficiency_name"]: float(result["signal_efficiency"][ix, iy]),
        "total_background_efficiency": float(result["background_efficiency"][ix, iy]),
    }
    for key in (
        "hbb_pure",
        "hcc_contaminated",
        "hcc_partial",
        "hbb_contaminated",
        "hbb_partial",
        "top",
        "other",
    ):
        if key in result.get("truth_efficiencies", {}):
            output_key = "hbb_pure_leakage" if key == "hbb_pure" else f"{key}_efficiency"
            point[output_key] = float(result["truth_efficiencies"][key][ix, iy])
    for process in ("qcd", "ttbar", "ttbb", "ttv", "ttHbb"):
        if process in result["process_efficiencies"]:
            suffix = "acceptance" if result["level"] == "event" else "efficiency"
            point[f"{process}_{suffix}"] = float(result["process_efficiencies"][process][ix, iy])
    return point


def _recommendations(result: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    global_index = _best_index(result["significance"])
    if global_index is not None:
        recommendations.append(_point_payload("global_max", global_index, result))
    for constraint in config["background_efficiency_constraints"]:
        index = _best_index(
            result["significance"],
            result["background_efficiency"] <= float(constraint),
        )
        if index is not None:
            recommendations.append(
                _point_payload(
                    f"background_eff_le_{100.0 * constraint:g}pct",
                    index,
                    result,
                )
            )
    return recommendations



def derive_jet_level_results(aggregate: dict[str, Any]) -> dict[str, Any]:
    signal_code = TRUTH_LABEL_TO_CODE[HCC_SIGNAL_LABEL]
    background_codes = [TRUTH_LABEL_TO_CODE[label] for label in HCC_BACKGROUND_LABELS]
    signal = aggregate["truth_yields"][signal_code]
    background = np.sum(aggregate["truth_yields"][background_codes], axis=0)
    signal_total = float(aggregate["truth_totals"][signal_code])
    background_total = float(np.sum(aggregate["truth_totals"][background_codes]))
    significance = _significance(signal, background)
    baseline_significance = float(
        _significance(np.array(signal_total), np.array(background_total))
    )
    truth_efficiencies = {
        label: _safe_ratio(
            aggregate["truth_yields"][TRUTH_LABEL_TO_CODE[label]],
            aggregate["truth_totals"][TRUTH_LABEL_TO_CODE[label]],
        )
        for label in TRUTH_LABEL_ORDER
    }
    result = {
        "level": "jet",
        "x_cuts": aggregate["x_cuts"],
        "y_cuts": aggregate["y_cuts"],
        "signal": signal,
        "background": background,
        "signal_over_background": _safe_ratio(signal, background),
        "significance": significance,
        "signal_over_sqrt_background": _safe_ratio(
            signal, np.sqrt(np.maximum(background, 0.0))
        ),
        "relative_significance": _safe_ratio(significance, baseline_significance),
        "signal_efficiency": _safe_ratio(signal, signal_total),
        "signal_efficiency_name": "hcc_pure_efficiency",
        "background_efficiency": _safe_ratio(background, background_total),
        "truth_efficiencies": truth_efficiencies,
        "process_efficiencies": _process_efficiencies(aggregate, event=False),
        "baseline": {
            "S": signal_total,
            "B": background_total,
            "S_over_sqrt_S_plus_B": baseline_significance,
        },
    }
    result["recommendations"] = _recommendations(result, aggregate["config"])
    return result


def derive_event_level_results(aggregate: dict[str, Any]) -> dict[str, Any]:
    if not aggregate.get("event_level_available", False):
        raise EventLevelUnavailableError(
            "event-level information unavailable in legacy cache"
        )
    signal_processes = set(aggregate["config"]["event_signal_processes"])
    signal_codes = [
        int(entry["code"])
        for entry in aggregate["process_entries"]
        if entry["process"] in signal_processes
    ]
    if not signal_codes:
        raise ValueError(
            "No configured event-level signal process was found: "
            + ", ".join(sorted(signal_processes))
        )
    background_codes = [
        int(entry["code"])
        for entry in aggregate["process_entries"]
        if entry["process"] not in signal_processes
    ]
    signal = np.sum(aggregate["event_process_yields"][signal_codes], axis=0)
    background = np.sum(aggregate["event_process_yields"][background_codes], axis=0)
    signal_total = float(np.sum(aggregate["event_process_totals"][signal_codes]))
    background_total = float(np.sum(aggregate["event_process_totals"][background_codes]))
    significance = _significance(signal, background)
    baseline_significance = float(
        _significance(np.array(signal_total), np.array(background_total))
    )
    result = {
        "level": "event",
        "x_cuts": aggregate["x_cuts"],
        "y_cuts": aggregate["y_cuts"],
        "signal": signal,
        "background": background,
        "signal_over_background": _safe_ratio(signal, background),
        "significance": significance,
        "signal_over_sqrt_background": _safe_ratio(
            signal, np.sqrt(np.maximum(background, 0.0))
        ),
        "relative_significance": _safe_ratio(significance, baseline_significance),
        "signal_efficiency": _safe_ratio(signal, signal_total),
        "signal_efficiency_name": "ttHcc_event_efficiency",
        "background_efficiency": _safe_ratio(background, background_total),
        "truth_efficiencies": {},
        "process_efficiencies": _process_efficiencies(aggregate, event=True),
        "baseline": {
            "S": signal_total,
            "B": background_total,
            "S_over_sqrt_S_plus_B": baseline_significance,
        },
    }
    result["recommendations"] = _recommendations(result, aggregate["config"])
    return result


def current_hcc_wp_references() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "label": f"current_{name}",
            "x_cut": float(payload["hcc_x_cut"]),
            "y_cut": 0.85,
            "legacy_convention": "x > x_cut and 0 < y <= y_cut",
            "description": str(payload["description"]),
        }
        for name, payload in XBB_VS_XCC_REGION_PRESETS.items()
    }
