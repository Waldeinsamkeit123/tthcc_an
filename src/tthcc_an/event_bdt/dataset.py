from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

from tthcc_an.config_loader import load_json_maybe_with_comments
from tthcc_an.event_bdt.config import (
    EventBdtConfig,
    EventBdtSampleConfig,
    load_event_bdt_samples_config,
)


TRAIN_LABEL_IGNORE = -1
TRAIN_LABEL_BACKGROUND = 0
TRAIN_LABEL_SIGNAL = 1


def _load_normalization_metadata(
    gen_sumw_file: str | None,
    xsec_file: str | None,
) -> tuple[dict[str, float], dict[str, float]]:
    if gen_sumw_file is None or xsec_file is None:
        return {}, {}
    gen_sumw_path = Path(gen_sumw_file)
    xsec_path = Path(xsec_file)
    if not gen_sumw_path.exists():
        raise FileNotFoundError(f"gen_sumw file does not exist: {gen_sumw_path}")
    if not xsec_path.exists():
        raise FileNotFoundError(f"xsec file does not exist: {xsec_path}")

    gen_sumw_payload = load_json_maybe_with_comments(gen_sumw_path)
    xsec_payload = load_json_maybe_with_comments(xsec_path)

    gen_sumw_map: dict[str, float] = {}
    for dataset, value in gen_sumw_payload.items():
        if isinstance(value, dict):
            gen_sumw_map[dataset] = float(value["gen_sumw"])
        else:
            gen_sumw_map[dataset] = float(value)
    xsec_map = {dataset: float(value) for dataset, value in xsec_payload.items()}
    return gen_sumw_map, xsec_map


def _sample_normalization(
    sample: EventBdtSampleConfig,
    lumi_fb: float | None,
    gen_sumw_map: dict[str, float],
    xsec_map: dict[str, float],
) -> tuple[float, float | None, float | None]:
    gen_sumw = gen_sumw_map.get(sample.dataset)
    xsec_fb = xsec_map.get(sample.dataset)
    if lumi_fb is None or gen_sumw is None or xsec_fb is None:
        return 1.0, gen_sumw, xsec_fb
    if gen_sumw == 0:
        raise ValueError(f"gen_sumw is zero for dataset '{sample.dataset}'.")
    return float(lumi_fb * xsec_fb / gen_sumw), gen_sumw, xsec_fb


def _train_label_for_process(
    process: str,
    signal_processes: list[str],
    background_processes: list[str],
) -> int:
    if process in signal_processes:
        return TRAIN_LABEL_SIGNAL
    if process in background_processes:
        return TRAIN_LABEL_BACKGROUND
    return TRAIN_LABEL_IGNORE


def _flatten_singleton_branch(values: ak.Array) -> np.ndarray:
    flattened = ak.fill_none(ak.firsts(values), np.nan)
    return np.asarray(ak.to_numpy(flattened), dtype=np.float64)


def _extract_branch(
    name: str,
    values: ak.Array,
    flatten_first_branches: set[str],
) -> np.ndarray:
    if name in flatten_first_branches:
        return _flatten_singleton_branch(values)
    return np.asarray(ak.to_numpy(values))


def _selection_mask(expression: str, columns: dict[str, np.ndarray]) -> np.ndarray:
    if not expression:
        first_column = next(iter(columns.values()))
        return np.ones(len(first_column), dtype=bool)
    namespace: dict[str, Any] = {name: value for name, value in columns.items()}
    namespace["np"] = np
    mask = eval(expression, {"__builtins__": {}}, namespace)
    result = np.asarray(mask, dtype=bool)
    if result.ndim != 1:
        raise ValueError("The configured base selection must evaluate to a one-dimensional boolean mask.")
    return result


def _compute_fold_ids(
    run: np.ndarray,
    lumi: np.ndarray,
    event: np.ndarray,
    k_folds: int,
) -> np.ndarray:
    if k_folds <= 0:
        raise ValueError("k_folds must be positive.")
    run_u = np.asarray(run, dtype=np.uint64)
    lumi_u = np.asarray(lumi, dtype=np.uint64)
    event_u = np.asarray(event, dtype=np.uint64)
    hashed = (
        run_u * np.uint64(73856093)
        + lumi_u * np.uint64(19349663)
        + event_u * np.uint64(83492791)
    )
    return np.asarray(hashed % np.uint64(k_folds), dtype=np.int16)


def _save_npz(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    serializable_arrays = dict(arrays)
    serializable_arrays["metadata_json"] = np.array(
        json.dumps(metadata, indent=2, sort_keys=True),
        dtype=np.str_,
    )
    np.savez_compressed(path, **serializable_arrays)


def load_npz_payload(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(np.asarray(payload["metadata_json"]).item())
        arrays = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key != "metadata_json"
        }
    return arrays, metadata


def prepare_event_bdt_inputs(config: EventBdtConfig, force: bool = False) -> Path:
    outpath = config.prepared_inputs_path
    if outpath.exists() and not force:
        print(f"Reusing prepared event-BDT cache: {outpath}", flush=True)
        return outpath

    samples_config = load_event_bdt_samples_config(
        config.samples_config_path,
        max_files_per_sample=config.max_files_per_sample,
    )
    print(
        f"Preparing event-BDT inputs from {len(samples_config.samples)} samples "
        f"into {outpath}",
        flush=True,
    )
    gen_sumw_map, xsec_map = _load_normalization_metadata(
        samples_config.gen_sumw_file,
        samples_config.xsec_file,
    )

    requested_branches = config.requested_branches()
    flatten_first_branches = set(config.flatten_first_branches)

    buffers: dict[str, list[np.ndarray]] = {name: [] for name in requested_branches}
    extra_buffers: dict[str, list[np.ndarray]] = {
        "process": [],
        "label_text": [],
        "role": [],
        "dataset_name": [],
        "sample_name": [],
        "train_label": [],
        "weight_signed": [],
        "train_weight": [],
        "sample_norm": [],
        "fold_id": [],
    }
    sample_summaries: list[dict[str, Any]] = []

    for sample_index, sample in enumerate(samples_config.samples, start=1):
        print(
            f"[{sample_index}/{len(samples_config.samples)}] Reading sample "
            f"{sample.name} ({len(sample.files)} files)",
            flush=True,
        )
        sample_norm, gen_sumw, xsec_fb = _sample_normalization(
            sample,
            samples_config.lumi_fb,
            gen_sumw_map,
            xsec_map,
        )
        label_value = _train_label_for_process(
            sample.process,
            config.signal_processes,
            config.background_processes,
        )
        total_events = 0
        total_selected = 0
        total_weight = 0.0
        total_signed_weight = 0.0
        skipped_files_missing_tree = 0

        for file_path in sample.files:
            with uproot.open(file_path) as root_file:
                if config.tree_name not in root_file:
                    skipped_files_missing_tree += 1
                    continue
                tree = root_file[config.tree_name]
                tree_keys = set(tree.keys())
                missing = [
                    branch
                    for branch in requested_branches
                    if branch != config.weight_branch and branch not in tree_keys
                ]
                if missing:
                    raise KeyError(
                        f"Missing required branches in {file_path}: {', '.join(sorted(missing))}"
                    )

                branches_to_read = [
                    branch
                    for branch in requested_branches
                    if branch == config.weight_branch or branch in tree_keys
                ]

                for arrays in tree.iterate(
                    branches_to_read,
                    library="ak",
                    step_size=config.uproot_step_size,
                ):
                    chunk_columns: dict[str, np.ndarray] = {}
                    n_events = len(arrays[branches_to_read[0]]) if branches_to_read else 0
                    total_events += n_events

                    for branch in requested_branches:
                        if branch in arrays.fields:
                            chunk_columns[branch] = _extract_branch(
                                branch,
                                arrays[branch],
                                flatten_first_branches,
                            )
                        elif branch == config.weight_branch:
                            chunk_columns[branch] = np.ones(n_events, dtype=np.float64)
                        else:
                            raise KeyError(f"Branch '{branch}' is missing in file: {file_path}")

                    mask = _selection_mask(config.base_selection, chunk_columns)
                    if not np.any(mask):
                        continue

                    selected = int(np.sum(mask))
                    total_selected += selected

                    raw_weight = np.asarray(chunk_columns[config.weight_branch][mask], dtype=np.float64)
                    weight_signed = sample_norm * raw_weight
                    train_weight = sample_norm * np.abs(raw_weight)
                    total_weight += float(np.sum(train_weight))
                    total_signed_weight += float(np.sum(weight_signed))

                    run = np.asarray(chunk_columns["run"][mask], dtype=np.int64)
                    lumi = np.asarray(chunk_columns["luminosityBlock"][mask], dtype=np.int64)
                    event = np.asarray(chunk_columns["event"][mask], dtype=np.int64)
                    fold_id = _compute_fold_ids(run, lumi, event, config.k_folds)

                    for branch in requested_branches:
                        buffers[branch].append(np.asarray(chunk_columns[branch][mask]))

                    extra_buffers["process"].append(np.full(selected, sample.process, dtype=np.str_))
                    extra_buffers["label_text"].append(np.full(selected, sample.label, dtype=np.str_))
                    extra_buffers["role"].append(np.full(selected, sample.role, dtype=np.str_))
                    extra_buffers["dataset_name"].append(
                        np.full(selected, sample.dataset, dtype=np.str_)
                    )
                    extra_buffers["sample_name"].append(np.full(selected, sample.name, dtype=np.str_))
                    extra_buffers["train_label"].append(
                        np.full(selected, label_value, dtype=np.int8)
                    )
                    extra_buffers["weight_signed"].append(weight_signed.astype(np.float64))
                    extra_buffers["train_weight"].append(train_weight.astype(np.float64))
                    extra_buffers["sample_norm"].append(
                        np.full(selected, sample_norm, dtype=np.float64)
                    )
                    extra_buffers["fold_id"].append(fold_id)

        sample_summaries.append(
            {
                "sample": sample.name,
                "dataset": sample.dataset,
                "process": sample.process,
                "label": sample.label,
                "role": sample.role,
                "n_files": len(sample.files),
                "skipped_files_missing_tree": skipped_files_missing_tree,
                "n_events": total_events,
                "n_selected_events": total_selected,
                "sample_norm": sample_norm,
                "gen_sumw": gen_sumw,
                "xsec_fb": xsec_fb,
                "selected_train_weight_sum": total_weight,
                "selected_signed_weight_sum": total_signed_weight,
            }
        )
        print(
            f"  Selected {total_selected} / {total_events} events, "
            f"weighted sum = {total_weight:.6g}, skipped files = {skipped_files_missing_tree}",
            flush=True,
        )

    concatenated: dict[str, np.ndarray] = {}
    for name, chunks in buffers.items():
        concatenated[name] = np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
    for name, chunks in extra_buffers.items():
        concatenated[name] = np.concatenate(chunks) if chunks else np.array([])

    metadata = {
        "config_path": str(config.config_path),
        "samples_config_path": str(config.samples_config_path),
        "tree_name": config.tree_name,
        "base_selection": config.base_selection,
        "selection_branches": list(config.selection_branches),
        "features": list(config.features),
        "spectators": list(config.spectators),
        "weight_branch": config.weight_branch,
        "k_folds": config.k_folds,
        "signal_processes": list(config.signal_processes),
        "background_processes": list(config.background_processes),
        "eval_processes_extra": list(config.eval_processes_extra),
        "sample_summaries": sample_summaries,
    }

    outpath.parent.mkdir(parents=True, exist_ok=True)
    _save_npz(outpath, concatenated, metadata)
    print(
        f"Prepared event-BDT cache written to {outpath} "
        f"with {len(concatenated.get('event', []))} selected events.",
        flush=True,
    )
    return outpath
