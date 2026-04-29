from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tthcc_an.config_loader import expand_file_patterns
from tthcc_an.definitions import (
    GLOBALPART3_CONTOUR_CATEGORIES,
    GLOBALPART3_CONTOUR_CLIP_EPS,
    GLOBALPART3_CONTOUR_HIST_BINS,
    GLOBALPART3_CONTOUR_PLOT,
    SCORE_LABELS,
    TRUTH_LABEL_ORDER,
    build_process_entries_from_pairs,
    build_process_entries_from_summaries,
    process_color,
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
) -> dict[str, Any]:
    x_edges = score_hist_edges(n_bins)
    y_edges = score_hist_edges(n_bins)
    weight_category = np.zeros((len(GLOBALPART3_CONTOUR_CATEGORIES), n_bins, n_bins), dtype=np.float64)
    valid = np.isfinite(x_scores) & np.isfinite(y_scores) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return {
            "x_score": GLOBALPART3_CONTOUR_PLOT["x_score"],
            "y_score": GLOBALPART3_CONTOUR_PLOT["y_score"],
            "filename_stem": GLOBALPART3_CONTOUR_PLOT["filename_stem"],
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
        "x_score": GLOBALPART3_CONTOUR_PLOT["x_score"],
        "y_score": GLOBALPART3_CONTOUR_PLOT["y_score"],
        "filename_stem": GLOBALPART3_CONTOUR_PLOT["filename_stem"],
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
    contour_x_score = GLOBALPART3_CONTOUR_PLOT["x_score"]
    contour_y_score = GLOBALPART3_CONTOUR_PLOT["y_score"]
    if contour_x_score in data and contour_y_score in data:
        contour_payloads[GLOBALPART3_CONTOUR_PLOT["key"]] = _contour_histogram_from_arrays(
            x_scores=np.asarray(data[contour_x_score], dtype=np.float64),
            y_scores=np.asarray(data[contour_y_score], dtype=np.float64),
            truth_codes=np.asarray(data["truth_code"], dtype=np.int8),
            weights=np.asarray(data["weight"], dtype=np.float64),
            n_bins=GLOBALPART3_CONTOUR_HIST_BINS,
        )
    return {
        "payload_mode": HISTOGRAM_PAYLOAD_MODE,
        "hist_edges": score_hist_edges(n_bins),
        "available_scores": list(score_names),
        "process_entries": process_entries,
        "score_histograms": score_histograms,
        "contour_payloads": contour_payloads,
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
            "x_score": contour_payload["x_score"],
            "y_score": contour_payload["y_score"],
            "filename_stem": contour_payload["filename_stem"],
            "categories": contour_payload["categories"],
        }

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
            sample_summary_groups.append(metadata.get("sample_summaries", []))

    merged_data = {key: np.concatenate(value_chunks) for key, value_chunks in arrays_by_key.items() if value_chunks}
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

    for chunk_path in chunk_paths:
        with np.load(chunk_path, allow_pickle=False) as payload:
            metadata = json.loads(np.asarray(payload["metadata_json"]).item())
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
                x_edges = np.asarray(payload[contour_payload_array_key(plot_key, "x_edges")], dtype=np.float64)
                y_edges = np.asarray(payload[contour_payload_array_key(plot_key, "y_edges")], dtype=np.float64)
                weight_category = np.asarray(
                    payload[contour_payload_array_key(plot_key, "weight_category")],
                    dtype=np.float64,
                )
                if plot_key not in contour_payloads:
                    contour_payloads[plot_key] = {
                        "x_score": plot_def["x_score"],
                        "y_score": plot_def["y_score"],
                        "filename_stem": plot_def["filename_stem"],
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
    return histogram_payload, sample_summaries, (weighting_info or {})
