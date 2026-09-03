from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tthcc_an.config_loader import expand_file_patterns
from tthcc_an.boosted_higgs_tagger_study.definitions import (
    GLOBALPART3_CONTOUR_CATEGORIES,
    GLOBALPART3_CONTOUR_CLIP_EPS,
    GLOBALPART3_CONTOUR_HIST_BINS,
    GLOBALPART3_CONTOUR_PLOT_BY_KEY,
    GLOBALPART3_CONTOUR_PLOTS,
    SCORE_LABELS,
    TRUTH_LABEL_ORDER,
    build_process_entries_from_pairs,
    build_process_entries_from_summaries,
    process_color,
)
from tthcc_an.boosted_higgs_tagger_study.inclusive_higgs_cache import (
    export_inclusive_higgs_wp_aggregate,
    load_inclusive_higgs_wp_aggregate,
)
from tthcc_an.boosted_higgs_tagger_study.inclusive_higgs_wp import (
    build_inclusive_higgs_wp_aggregate,
    merge_inclusive_higgs_wp_aggregates,
)
from tthcc_an.boosted_higgs_tagger_study.y_split_cache import (
    export_y_split_aggregate,
    load_y_split_aggregate,
)
from tthcc_an.boosted_higgs_tagger_study.y_split_study import (
    build_y_split_aggregate,
    merge_y_split_aggregates,
)
from tthcc_an.boosted_higgs_tagger_study.scan_cache import (
    export_hcc_scan_aggregate,
    load_hcc_scan_aggregate,
)
from tthcc_an.boosted_higgs_tagger_study.scans import (
    build_hcc_wp_scan_aggregate,
    merge_hcc_wp_scan_aggregates,
)


RAW_PAYLOAD_MODE = "raw_v1"
HISTOGRAM_PAYLOAD_MODE = "histogram_v1"


def _available_scores_from_data(data: dict[str, np.ndarray]) -> list[str]:
    available: list[str] = []
    for score_name in SCORE_LABELS:
        if score_name in data and data[score_name].size > 0 and np.any(np.isfinite(data[score_name])):
            available.append(score_name)
    return available


def score_hist_edges(n_bins: int) -> np.ndarray:
    if n_bins <= 0:
        raise ValueError("--score-hist-bins must be positive.")
    return np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)


def score_to_hist_bins(scores: np.ndarray, n_bins: int) -> np.ndarray:
    clipped = np.clip(scores, 0.0, 1.0)
    return np.minimum((clipped * n_bins).astype(np.int32), n_bins - 1)


def payload_array_key(score_name: str, kind: str) -> str:
    return f"{score_name}__{kind}"


def contour_payload_array_key(plot_key: str, kind: str) -> str:
    return f"{plot_key}__{kind}"


def payload_mode_from_metadata(metadata: dict[str, Any]) -> str:
    return str(metadata.get("payload_mode", RAW_PAYLOAD_MODE))


def load_payload_metadata(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as payload:
        return json.loads(np.asarray(payload["metadata_json"]).item())


def detect_payload_mode(chunk_patterns: list[str]) -> str:
    chunk_paths = [Path(path) for path in expand_file_patterns(chunk_patterns)]
    if not chunk_paths:
        raise ValueError("No chunk NPZ files were found.")
    return payload_mode_from_metadata(load_payload_metadata(chunk_paths[0]))


def _score_histograms_from_arrays(
    scores: np.ndarray,
    truth_codes: np.ndarray,
    weights: np.ndarray,
    signed_weights: np.ndarray,
    process_codes: np.ndarray | None,
    n_bins: int,
    n_processes: int,
) -> dict[str, np.ndarray]:
    valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    count_truth = np.zeros((len(TRUTH_LABEL_ORDER), n_bins), dtype=np.int64)
    weight_truth = np.zeros((len(TRUTH_LABEL_ORDER), n_bins), dtype=np.float64)
    signed_truth = np.zeros((len(TRUTH_LABEL_ORDER), n_bins), dtype=np.float64)
    weight_process_truth = np.zeros((n_processes, len(TRUTH_LABEL_ORDER), n_bins), dtype=np.float64)
    if not np.any(valid):
        return {
            "count_truth": count_truth,
            "weight_truth": weight_truth,
            "signed_truth": signed_truth,
            "weight_process_truth": weight_process_truth,
        }

    truth_valid = truth_codes[valid].astype(np.int32, copy=False)
    bins_valid = score_to_hist_bins(scores[valid], n_bins)
    weights_valid = weights[valid].astype(np.float64, copy=False)
    signed_valid = signed_weights[valid].astype(np.float64, copy=False)
    np.add.at(count_truth, (truth_valid, bins_valid), 1)
    np.add.at(weight_truth, (truth_valid, bins_valid), weights_valid)
    np.add.at(signed_truth, (truth_valid, bins_valid), signed_valid)

    if process_codes is not None and n_processes > 0:
        process_valid = process_codes[valid].astype(np.int32, copy=False)
        np.add.at(weight_process_truth, (process_valid, truth_valid, bins_valid), weights_valid)

    return {
        "count_truth": count_truth,
        "weight_truth": weight_truth,
        "signed_truth": signed_truth,
        "weight_process_truth": weight_process_truth,
    }


def _contour_histogram_from_arrays(
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    truth_codes: np.ndarray,
    weights: np.ndarray,
    n_bins: int,
    plot_def: dict[str, Any],
) -> dict[str, Any]:
    x_edges = score_hist_edges(n_bins)
    y_edges = score_hist_edges(n_bins)
    weight_category = np.zeros((len(GLOBALPART3_CONTOUR_CATEGORIES), n_bins, n_bins), dtype=np.float64)
    valid = np.isfinite(x_scores) & np.isfinite(y_scores) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return {
            "key": plot_def["key"],
            "x_score": plot_def["x_score"],
            "y_score": plot_def["y_score"],
            "filename_stem": plot_def["filename_stem"],
            "fixed_x_cut": plot_def.get("fixed_x_cut"),
            "region_preset": plot_def.get("region_preset"),
            "region_preset_label": plot_def.get("region_preset_label"),
            "region_preset_description": plot_def.get("region_preset_description"),
            "region_definitions": plot_def["region_definitions"],
            "boundary_segments": plot_def["boundary_segments"],
            "categories": list(GLOBALPART3_CONTOUR_CATEGORIES),
            "x_edges": x_edges,
            "y_edges": y_edges,
            "weight_category": weight_category,
        }

    x_valid = np.clip(np.asarray(x_scores[valid], dtype=np.float64), GLOBALPART3_CONTOUR_CLIP_EPS, 1.0 - GLOBALPART3_CONTOUR_CLIP_EPS)
    y_valid = np.clip(np.asarray(y_scores[valid], dtype=np.float64), GLOBALPART3_CONTOUR_CLIP_EPS, 1.0 - GLOBALPART3_CONTOUR_CLIP_EPS)
    truth_valid = np.asarray(truth_codes[valid], dtype=np.int32)
    weights_valid = np.asarray(weights[valid], dtype=np.float64)

    for index, category in enumerate(GLOBALPART3_CONTOUR_CATEGORIES):
        category_codes = np.asarray(category["truth_codes"], dtype=np.int32)
        category_mask = np.isin(truth_valid, category_codes)
        if not np.any(category_mask):
            continue
        hist2d, _, _ = np.histogram2d(
            x_valid[category_mask],
            y_valid[category_mask],
            bins=(x_edges, y_edges),
            weights=weights_valid[category_mask],
        )
        weight_category[index] = np.asarray(hist2d, dtype=np.float64)

    return {
        "key": plot_def["key"],
        "x_score": plot_def["x_score"],
        "y_score": plot_def["y_score"],
        "filename_stem": plot_def["filename_stem"],
        "fixed_x_cut": plot_def.get("fixed_x_cut"),
        "region_preset": plot_def.get("region_preset"),
        "region_preset_label": plot_def.get("region_preset_label"),
        "region_preset_description": plot_def.get("region_preset_description"),
        "region_definitions": plot_def["region_definitions"],
        "boundary_segments": plot_def["boundary_segments"],
        "categories": list(GLOBALPART3_CONTOUR_CATEGORIES),
        "x_edges": x_edges,
        "y_edges": y_edges,
        "weight_category": weight_category,
    }


def build_histogram_payload_from_raw_data(
    data: dict[str, np.ndarray],
    sample_summaries: list[dict[str, Any]],
    score_names: list[str],
    n_bins: int,
    contour_plot_defs: list[dict[str, Any]] | None = None,
    hcc_wp_scan_config: dict[str, Any] | None = None,
    inclusive_higgs_wp_config: dict[str, Any] | None = None,
    y_split_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    process_entries = build_process_entries_from_summaries(sample_summaries)
    n_processes = len(process_entries)
    process_codes = data.get("process_code")
    score_histograms: dict[str, dict[str, np.ndarray]] = {}
    contour_payloads: dict[str, dict[str, Any]] = {}
    for score_name in score_names:
        score_histograms[score_name] = _score_histograms_from_arrays(
            scores=np.asarray(data[score_name], dtype=np.float64),
            truth_codes=np.asarray(data["truth_code"], dtype=np.int8),
            weights=np.asarray(data["weight"], dtype=np.float64),
            signed_weights=np.asarray(data["weight_signed"], dtype=np.float64),
            process_codes=None if process_codes is None else np.asarray(process_codes, dtype=np.int16),
            n_bins=n_bins,
            n_processes=n_processes,
        )
    selected_contour_plot_defs = list(contour_plot_defs or GLOBALPART3_CONTOUR_PLOTS)
    for plot_def in selected_contour_plot_defs:
        contour_x_score = str(plot_def["x_score"])
        contour_y_score = str(plot_def["y_score"])
        if contour_x_score in data and contour_y_score in data:
            contour_payloads[str(plot_def["key"])] = _contour_histogram_from_arrays(
                x_scores=np.asarray(data[contour_x_score], dtype=np.float64),
                y_scores=np.asarray(data[contour_y_score], dtype=np.float64),
                truth_codes=np.asarray(data["truth_code"], dtype=np.int8),
                weights=np.asarray(data["weight"], dtype=np.float64),
                n_bins=GLOBALPART3_CONTOUR_HIST_BINS,
                plot_def=plot_def,
            )
    hcc_wp_scan = None
    if hcc_wp_scan_config is not None and hcc_wp_scan_config.get("enabled", False):
        hcc_wp_scan = build_hcc_wp_scan_aggregate(
            data,
            process_entries,
            hcc_wp_scan_config,
        )
    inclusive_higgs_wp = None
    if (
        inclusive_higgs_wp_config is not None
        and inclusive_higgs_wp_config.get("enabled", False)
    ):
        inclusive_higgs_wp = build_inclusive_higgs_wp_aggregate(
            data, process_entries, inclusive_higgs_wp_config
        )
    y_split_study = None
    if y_split_config is not None and y_split_config.get("enabled", False):
        y_split_study = build_y_split_aggregate(
            data, process_entries, y_split_config
        )
    return {
        "payload_mode": HISTOGRAM_PAYLOAD_MODE,
        "hist_edges": score_hist_edges(n_bins),
        "available_scores": list(score_names),
        "process_entries": process_entries,
        "score_histograms": score_histograms,
        "contour_payloads": contour_payloads,
        "hcc_wp_scan": hcc_wp_scan,
        "inclusive_higgs_wp": inclusive_higgs_wp,
        "y_split_study": y_split_study,
    }


def export_histogram_payload(
    outpath: Path,
    histogram_payload: dict[str, Any],
    sample_summaries: list[dict[str, Any]],
    weighting_info: dict[str, Any],
) -> list[str]:
    arrays: dict[str, np.ndarray] = {
        "hist_edges": np.asarray(histogram_payload["hist_edges"], dtype=np.float64),
    }
    for score_name, score_payload in histogram_payload["score_histograms"].items():
        for kind, array in score_payload.items():
            arrays[payload_array_key(score_name, kind)] = array
    contour_plot_defs: dict[str, Any] = {}
    for plot_key, contour_payload in histogram_payload.get("contour_payloads", {}).items():
        arrays[contour_payload_array_key(plot_key, "x_edges")] = np.asarray(contour_payload["x_edges"], dtype=np.float64)
        arrays[contour_payload_array_key(plot_key, "y_edges")] = np.asarray(contour_payload["y_edges"], dtype=np.float64)
        arrays[contour_payload_array_key(plot_key, "weight_category")] = np.asarray(
            contour_payload["weight_category"],
            dtype=np.float64,
        )
        contour_plot_defs[plot_key] = {
            "key": contour_payload["key"],
            "x_score": contour_payload["x_score"],
            "y_score": contour_payload["y_score"],
            "filename_stem": contour_payload["filename_stem"],
            "fixed_x_cut": contour_payload.get("fixed_x_cut"),
            "region_preset": contour_payload.get("region_preset"),
            "region_preset_label": contour_payload.get("region_preset_label"),
            "region_preset_description": contour_payload.get("region_preset_description"),
            "region_definitions": contour_payload["region_definitions"],
            "boundary_segments": contour_payload["boundary_segments"],
            "categories": contour_payload["categories"],
        }

    hcc_scan_metadata = None
    if histogram_payload.get("hcc_wp_scan") is not None:
        hcc_scan_metadata = export_hcc_scan_aggregate(
            arrays, histogram_payload["hcc_wp_scan"]
        )

    inclusive_higgs_metadata = None
    if histogram_payload.get("inclusive_higgs_wp") is not None:
        inclusive_higgs_metadata = export_inclusive_higgs_wp_aggregate(
            arrays,
            histogram_payload["inclusive_higgs_wp"],
        )

    y_split_metadata = None
    if histogram_payload.get("y_split_study") is not None:
        y_split_metadata = export_y_split_aggregate(
            arrays, histogram_payload["y_split_study"]
        )

    arrays["metadata_json"] = np.array(
        json.dumps(
            {
                "payload_mode": HISTOGRAM_PAYLOAD_MODE,
                "available_scores": histogram_payload["available_scores"],
                "available_contours": list(histogram_payload.get("contour_payloads", {}).keys()),
                "contour_plot_defs": contour_plot_defs,
                "process_entries": histogram_payload["process_entries"],
                "sample_summaries": sample_summaries,
                "weighting": weighting_info,
                "score_hist_bins": int(len(np.asarray(histogram_payload["hist_edges"])) - 1),
                "hcc_wp_scan": hcc_scan_metadata,
                "inclusive_higgs_wp": inclusive_higgs_metadata,
                "y_split_study": y_split_metadata,
            },
            sort_keys=True,
        )
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outpath, **arrays)
    return list(histogram_payload["available_scores"])


def export_chunk_payload(
    outpath: Path,
    data: dict[str, np.ndarray],
    sample_summaries: list[dict[str, Any]],
    weighting_info: dict[str, Any],
) -> list[str]:
    available_scores = _available_scores_from_data(data)
    payload = {key: data[key] for key in ["truth_code", "weight", "weight_signed", *available_scores] if key in data}
    if "process_code" in data:
        payload["process_code"] = data["process_code"]
    if "event_index" in data:
        payload["event_index"] = data["event_index"]
    payload["metadata_json"] = np.array(
        json.dumps(
            {
                "payload_mode": RAW_PAYLOAD_MODE,
                "available_scores": available_scores,
                "sample_summaries": sample_summaries,
                "process_entries": build_process_entries_from_summaries(sample_summaries),
                "weighting": weighting_info,
            },
            sort_keys=True,
        )
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outpath, **payload)
    return available_scores


def _merge_sample_summaries(summary_groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    fields_to_sum = [
        "n_files",
        "n_skipped_files_missing_tree",
        "n_events",
        "n_selected_jets",
        "analysis_weight_sum",
        "signed_weight_sum",
    ]

    for summaries in summary_groups:
        for summary in summaries:
            key = (summary["name"], summary["dataset"], summary["process"], summary["label"])
            if key not in merged:
                merged[key] = dict(summary)
                for field in fields_to_sum:
                    merged[key].setdefault(field, 0)
                continue
            for field in fields_to_sum:
                merged[key][field] += summary.get(field, 0)
            if merged[key].get("weight_branch") is None:
                merged[key]["weight_branch"] = summary.get("weight_branch")

    return sorted(merged.values(), key=lambda entry: (entry["process"], entry["dataset"], entry["name"]))


def load_merged_chunk_payloads(chunk_patterns: list[str]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    chunk_paths = [Path(path) for path in expand_file_patterns(chunk_patterns)]
    if not chunk_paths:
        raise ValueError("No chunk NPZ files were found for merge.")

    arrays_by_key: dict[str, list[np.ndarray]] = {"truth_code": [], "weight": [], "weight_signed": []}
    sample_summary_groups: list[list[dict[str, Any]]] = []
    expected_scores: list[str] | None = None
    weighting_info: dict[str, Any] | None = None
    process_chunk_payloads: list[tuple[np.ndarray, list[dict[str, Any]]]] = []
    process_payload_complete = True
    saw_process_payload = False
    event_index_chunks: list[np.ndarray] = []
    event_payload_complete = True
    next_event_offset = 0

    for chunk_path in chunk_paths:
        with np.load(chunk_path, allow_pickle=False) as payload:
            metadata = json.loads(np.asarray(payload["metadata_json"]).item())
            chunk_scores = list(metadata.get("available_scores", []))
            if expected_scores is None:
                expected_scores = chunk_scores
                for score_name in expected_scores:
                    arrays_by_key[score_name] = []
            elif chunk_scores != expected_scores:
                raise ValueError(
                    f"Inconsistent available scores across chunk payloads. "
                    f"Expected {expected_scores}, got {chunk_scores} from {chunk_path}"
                )

            if weighting_info is None:
                weighting_info = metadata.get("weighting", {})

            for key in ["truth_code", "weight", "weight_signed", *(expected_scores or [])]:
                arrays_by_key[key].append(np.asarray(payload[key]))

            chunk_process_entries = metadata.get("process_entries", [])
            if "process_code" in payload.files and chunk_process_entries:
                saw_process_payload = True
                process_chunk_payloads.append(
                    (
                        np.asarray(payload["process_code"], dtype=np.int16),
                        list(chunk_process_entries),
                    )
                )
            else:
                process_payload_complete = False
            if "event_index" in payload.files:
                event_ids = np.asarray(payload["event_index"], dtype=np.int64)
                if event_ids.size:
                    event_ids = event_ids - int(np.min(event_ids)) + next_event_offset
                    next_event_offset = int(np.max(event_ids)) + 1
                event_index_chunks.append(event_ids)
            else:
                event_payload_complete = False
            sample_summary_groups.append(metadata.get("sample_summaries", []))

    merged_data = {key: np.concatenate(value_chunks) for key, value_chunks in arrays_by_key.items() if value_chunks}
    if event_payload_complete and len(event_index_chunks) == len(chunk_paths):
        merged_data["event_index"] = np.concatenate(event_index_chunks)
    if saw_process_payload and process_payload_complete and len(process_chunk_payloads) == len(chunk_paths):
        process_to_label: dict[str, str] = {}
        for _, chunk_entries in process_chunk_payloads:
            for entry in chunk_entries:
                process_to_label.setdefault(entry["process"], entry.get("label", entry["process"]))

        global_entries = build_process_entries_from_pairs(process_to_label)
        global_code_by_process = {entry["process"]: int(entry["code"]) for entry in global_entries}
        remapped_chunks: list[np.ndarray] = []
        for codes, chunk_entries in process_chunk_payloads:
            local_process_by_code = {int(entry["code"]): entry["process"] for entry in chunk_entries}
            remapped = np.full(codes.shape, -1, dtype=np.int16)
            for local_code, process in local_process_by_code.items():
                remapped[codes == local_code] = global_code_by_process[process]
            if np.any(remapped < 0):
                raise ValueError("Failed to remap process codes while merging chunk payloads.")
            remapped_chunks.append(remapped)
        merged_data["process_code"] = np.concatenate(remapped_chunks)

    sample_summaries = _merge_sample_summaries(sample_summary_groups)
    return merged_data, sample_summaries, (weighting_info or {})


def _expand_process_axis(array: np.ndarray, new_size: int) -> np.ndarray:
    if array.shape[0] >= new_size:
        return array
    expanded = np.zeros((new_size, *array.shape[1:]), dtype=array.dtype)
    expanded[: array.shape[0]] = array
    return expanded


def load_merged_histogram_payloads(
    chunk_patterns: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    chunk_paths = [Path(path) for path in expand_file_patterns(chunk_patterns)]
    if not chunk_paths:
        raise ValueError("No chunk NPZ files were found for merge.")

    score_histograms: dict[str, dict[str, np.ndarray]] = {}
    expected_scores: list[str] | None = None
    hist_edges: np.ndarray | None = None
    weighting_info: dict[str, Any] | None = None
    sample_summary_groups: list[list[dict[str, Any]]] = []
    global_process_entries: list[dict[str, Any]] = []
    process_to_global_code: dict[str, int] = {}
    contour_payloads: dict[str, dict[str, Any]] = {}
    expected_contours: list[str] | None = None
    hcc_scan_aggregates: list[dict[str, Any]] = []
    inclusive_higgs_aggregates: list[dict[str, Any]] = []
    y_split_aggregates: list[dict[str, Any]] = []

    for chunk_path in chunk_paths:
        with np.load(chunk_path, allow_pickle=False) as payload:
            metadata = json.loads(np.asarray(payload["metadata_json"]).item())
            hcc_scan = load_hcc_scan_aggregate(payload, metadata)
            if hcc_scan is not None:
                hcc_scan_aggregates.append(hcc_scan)
            inclusive_higgs = load_inclusive_higgs_wp_aggregate(payload, metadata)
            if inclusive_higgs is not None:
                inclusive_higgs_aggregates.append(inclusive_higgs)
            y_split = load_y_split_aggregate(payload, metadata)
            if y_split is not None:
                y_split_aggregates.append(y_split)
            if payload_mode_from_metadata(metadata) != HISTOGRAM_PAYLOAD_MODE:
                raise ValueError(f"Chunk payload {chunk_path} is not a histogram payload.")

            chunk_scores = list(metadata.get("available_scores", []))
            if expected_scores is None:
                expected_scores = chunk_scores
            elif chunk_scores != expected_scores:
                raise ValueError(
                    f"Inconsistent available scores across chunk payloads. "
                    f"Expected {expected_scores}, got {chunk_scores} from {chunk_path}"
                )

            chunk_contours = list(metadata.get("available_contours", []))
            if expected_contours is None:
                expected_contours = chunk_contours
            elif chunk_contours != expected_contours:
                raise ValueError(
                    f"Inconsistent available contour payloads across chunk payloads. "
                    f"Expected {expected_contours}, got {chunk_contours} from {chunk_path}"
                )
            contour_plot_defs = dict(metadata.get("contour_plot_defs", {}))

            chunk_edges = np.asarray(payload["hist_edges"], dtype=np.float64)
            if hist_edges is None:
                hist_edges = chunk_edges
            elif not np.array_equal(hist_edges, chunk_edges):
                raise ValueError(f"Inconsistent histogram edges found in {chunk_path}.")

            if weighting_info is None:
                weighting_info = metadata.get("weighting", {})

            chunk_process_entries = list(metadata.get("process_entries", []))
            for entry in chunk_process_entries:
                process = entry["process"]
                if process not in process_to_global_code:
                    new_code = len(global_process_entries)
                    process_to_global_code[process] = new_code
                    global_process_entries.append(
                        {
                            "code": new_code,
                            "process": process,
                            "label": entry.get("label", process),
                            "color": entry.get("color", process_color(process, new_code)),
                        }
                    )
                    for score_payload in score_histograms.values():
                        score_payload["weight_process_truth"] = _expand_process_axis(
                            score_payload["weight_process_truth"],
                            len(global_process_entries),
                        )

            local_process_map = {int(entry["code"]): entry["process"] for entry in chunk_process_entries}
            for score_name in expected_scores or []:
                count_truth = np.asarray(payload[payload_array_key(score_name, "count_truth")], dtype=np.int64)
                weight_truth = np.asarray(payload[payload_array_key(score_name, "weight_truth")], dtype=np.float64)
                signed_truth = np.asarray(payload[payload_array_key(score_name, "signed_truth")], dtype=np.float64)
                chunk_process_hist = np.asarray(
                    payload[payload_array_key(score_name, "weight_process_truth")],
                    dtype=np.float64,
                )
                if score_name not in score_histograms:
                    score_histograms[score_name] = {
                        "count_truth": np.zeros_like(count_truth),
                        "weight_truth": np.zeros_like(weight_truth),
                        "signed_truth": np.zeros_like(signed_truth),
                        "weight_process_truth": np.zeros(
                            (len(global_process_entries), *chunk_process_hist.shape[1:]),
                            dtype=np.float64,
                        ),
                    }

                score_histograms[score_name]["count_truth"] += count_truth
                score_histograms[score_name]["weight_truth"] += weight_truth
                score_histograms[score_name]["signed_truth"] += signed_truth
                if score_histograms[score_name]["weight_process_truth"].shape[0] < len(global_process_entries):
                    score_histograms[score_name]["weight_process_truth"] = _expand_process_axis(
                        score_histograms[score_name]["weight_process_truth"],
                        len(global_process_entries),
                    )
                for local_code, process in local_process_map.items():
                    global_code = process_to_global_code[process]
                    score_histograms[score_name]["weight_process_truth"][global_code] += chunk_process_hist[local_code]

            for plot_key in expected_contours or []:
                plot_def = contour_plot_defs.get(plot_key)
                if plot_def is None:
                    raise ValueError(f"Missing contour plot definition for {plot_key} in {chunk_path}")
                canonical_plot_def = dict(GLOBALPART3_CONTOUR_PLOT_BY_KEY.get(plot_key, {}))
                if canonical_plot_def:
                    canonical_plot_def["key"] = plot_def.get("key", canonical_plot_def.get("key", plot_key))
                    canonical_plot_def["x_score"] = plot_def.get("x_score", canonical_plot_def["x_score"])
                    canonical_plot_def["y_score"] = plot_def.get("y_score", canonical_plot_def["y_score"])
                    canonical_plot_def["filename_stem"] = plot_def.get(
                        "filename_stem",
                        canonical_plot_def["filename_stem"],
                    )
                    canonical_plot_def["fixed_x_cut"] = plot_def.get(
                        "fixed_x_cut",
                        canonical_plot_def.get("fixed_x_cut"),
                    )
                    canonical_plot_def["region_preset"] = plot_def.get(
                        "region_preset",
                        canonical_plot_def.get("region_preset"),
                    )
                    canonical_plot_def["region_preset_label"] = plot_def.get(
                        "region_preset_label",
                        canonical_plot_def.get("region_preset_label"),
                    )
                    canonical_plot_def["region_preset_description"] = plot_def.get(
                        "region_preset_description",
                        canonical_plot_def.get("region_preset_description"),
                    )
                else:
                    canonical_plot_def = dict(plot_def)
                x_edges = np.asarray(payload[contour_payload_array_key(plot_key, "x_edges")], dtype=np.float64)
                y_edges = np.asarray(payload[contour_payload_array_key(plot_key, "y_edges")], dtype=np.float64)
                weight_category = np.asarray(
                    payload[contour_payload_array_key(plot_key, "weight_category")],
                    dtype=np.float64,
                )
                if plot_key not in contour_payloads:
                    contour_payloads[plot_key] = {
                        "key": canonical_plot_def.get("key", plot_key),
                        "x_score": canonical_plot_def["x_score"],
                        "y_score": canonical_plot_def["y_score"],
                        "filename_stem": canonical_plot_def["filename_stem"],
                        "fixed_x_cut": canonical_plot_def.get("fixed_x_cut"),
                        "region_preset": canonical_plot_def.get("region_preset"),
                        "region_preset_label": canonical_plot_def.get("region_preset_label"),
                        "region_preset_description": canonical_plot_def.get("region_preset_description"),
                        "region_definitions": canonical_plot_def.get("region_definitions", {}),
                        "boundary_segments": canonical_plot_def.get("boundary_segments", []),
                        "categories": list(plot_def["categories"]),
                        "x_edges": x_edges,
                        "y_edges": y_edges,
                        "weight_category": np.zeros_like(weight_category),
                    }
                else:
                    if not np.array_equal(contour_payloads[plot_key]["x_edges"], x_edges):
                        raise ValueError(f"Inconsistent contour x-edges found in {chunk_path} for {plot_key}.")
                    if not np.array_equal(contour_payloads[plot_key]["y_edges"], y_edges):
                        raise ValueError(f"Inconsistent contour y-edges found in {chunk_path} for {plot_key}.")
                contour_payloads[plot_key]["weight_category"] += weight_category

            sample_summary_groups.append(metadata.get("sample_summaries", []))

    if hist_edges is None:
        raise ValueError("Histogram payload merge found no histogram edges.")

    ordered_process_entries = build_process_entries_from_pairs(
        {entry["process"]: entry["label"] for entry in global_process_entries}
    )
    if ordered_process_entries:
        reorder = [process_to_global_code[entry["process"]] for entry in ordered_process_entries]
        for score_payload in score_histograms.values():
            score_payload["weight_process_truth"] = score_payload["weight_process_truth"][reorder]

    histogram_payload = {
        "payload_mode": HISTOGRAM_PAYLOAD_MODE,
        "hist_edges": hist_edges,
        "available_scores": list(expected_scores or []),
        "process_entries": ordered_process_entries,
        "score_histograms": score_histograms,
        "contour_payloads": contour_payloads,
    }
    sample_summaries = _merge_sample_summaries(sample_summary_groups)
    if hcc_scan_aggregates:
        if len(hcc_scan_aggregates) != len(chunk_paths):
            raise ValueError("Inconsistent Hcc WP scan payload availability across chunks.")
        histogram_payload["hcc_wp_scan"] = merge_hcc_wp_scan_aggregates(
            hcc_scan_aggregates
        )
    if inclusive_higgs_aggregates:
        if len(inclusive_higgs_aggregates) != len(chunk_paths):
            raise ValueError(
                "Inconsistent Inclusive Higgs WP payload availability across chunks."
            )
        histogram_payload["inclusive_higgs_wp"] = (
            merge_inclusive_higgs_wp_aggregates(inclusive_higgs_aggregates)
        )
    if y_split_aggregates:
        if len(y_split_aggregates) != len(chunk_paths):
            raise ValueError("Inconsistent y-split payload availability across chunks.")
        histogram_payload["y_split_study"] = merge_y_split_aggregates(
            y_split_aggregates
        )
    return histogram_payload, sample_summaries, (weighting_info or {})
