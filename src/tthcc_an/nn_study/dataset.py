from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

from tthcc_an.config_loader import load_json_maybe_with_comments
from tthcc_an.nn_study.config import NnSample, NnStudyConfig
from tthcc_an.nn_study.definitions import assign_truth_categories, evaluate_mask


@dataclass(frozen=True)
class NnDataset:
    scores: dict[str, np.ndarray]
    truth_index: np.ndarray
    sample_index: np.ndarray
    raw_weight: np.ndarray
    analysis_weight: np.ndarray
    signed_weight: np.ndarray
    analysis_columns: dict[str, np.ndarray]
    sample_summaries: list[dict[str, Any]]
    totals: dict[str, int]

    def score_matrix(self, names: list[str]) -> np.ndarray:
        return np.column_stack([self.scores[name] for name in names])


def _load_normalization_metadata(config: NnStudyConfig) -> tuple[dict[str, float], dict[str, float]]:
    gen_sumw_path = Path(config.gen_sumw_file)
    xsec_path = Path(config.xsec_file)
    if not gen_sumw_path.exists():
        raise FileNotFoundError(f"gen_sumw file does not exist: {gen_sumw_path}")
    if not xsec_path.exists():
        raise FileNotFoundError(f"Cross-section file does not exist: {xsec_path}")
    gen_sumw_payload = load_json_maybe_with_comments(gen_sumw_path)
    xsec_payload = load_json_maybe_with_comments(xsec_path)
    gen_sumw_map = {
        str(dataset): float(value["gen_sumw"] if isinstance(value, dict) else value)
        for dataset, value in gen_sumw_payload.items()
    }
    xsec_map = {str(dataset): float(value) for dataset, value in xsec_payload.items()}
    return gen_sumw_map, xsec_map


def _sample_normalization(
    sample: NnSample,
    config: NnStudyConfig,
    gen_sumw_map: dict[str, float],
    xsec_map: dict[str, float],
) -> tuple[float, float, float]:
    if sample.dataset not in gen_sumw_map:
        raise KeyError(f"Missing gen_sumw for dataset '{sample.dataset}'.")
    if sample.dataset not in xsec_map:
        raise KeyError(f"Missing cross section for dataset '{sample.dataset}'.")
    gen_sumw = gen_sumw_map[sample.dataset]
    xsec_fb = xsec_map[sample.dataset]
    if gen_sumw == 0:
        raise ValueError(f"gen_sumw is zero for dataset '{sample.dataset}'.")
    return config.lumi_fb * xsec_fb / gen_sumw, gen_sumw, xsec_fb


def _to_numpy(values: ak.Array, *, flatten_first: bool) -> np.ndarray:
    if flatten_first:
        values = ak.fill_none(ak.firsts(values), np.nan)
    result = np.asarray(ak.to_numpy(values))
    if result.ndim != 1:
        raise ValueError("NN-study branches must be scalar or configured in flatten_first_branches.")
    return result


def _empty_dataset(
    config: NnStudyConfig,
    score_names: list[str],
    analysis_branches: list[str],
) -> NnDataset:
    return NnDataset(
        scores={name: np.array([], dtype=np.float64) for name in score_names},
        truth_index=np.array([], dtype=np.int16),
        sample_index=np.array([], dtype=np.int16),
        raw_weight=np.array([], dtype=np.float64),
        analysis_weight=np.array([], dtype=np.float64),
        signed_weight=np.array([], dtype=np.float64),
        analysis_columns={
            name: np.array([], dtype=np.float64) for name in analysis_branches
        },
        sample_summaries=[],
        totals={
            "files_discovered": 0,
            "files_attempted": 0,
            "files_processed": 0,
            "files_missing_tree": 0,
            "events_read": 0,
            "events_selected": 0,
            "events_classified": 0,
            "events_unclassified": 0,
        },
    )


def load_nn_dataset(
    config: NnStudyConfig,
    *,
    score_names: list[str] | None = None,
    analysis_branches: list[str] | None = None,
) -> NnDataset:
    loaded_score_names = list(
        config.all_score_names if score_names is None else score_names
    )
    loaded_analysis_branches = list(
        config.analysis_branches
        if analysis_branches is None
        else analysis_branches
    )
    gen_sumw_map, xsec_map = _load_normalization_metadata(config)
    requested = config.requested_branches_for(
        score_names=loaded_score_names,
        analysis_branches=loaded_analysis_branches,
    )
    flatten_first = set(config.flatten_first_branches)
    score_by_name = {score.name: score for score in config.all_scores}
    score_by_branch = {
        score_by_name[name].branch: name for name in loaded_score_names
    }

    score_buffers: dict[str, list[np.ndarray]] = {
        name: [] for name in loaded_score_names
    }
    truth_buffers: list[np.ndarray] = []
    sample_index_buffers: list[np.ndarray] = []
    raw_weight_buffers: list[np.ndarray] = []
    analysis_weight_buffers: list[np.ndarray] = []
    signed_weight_buffers: list[np.ndarray] = []
    analysis_buffers: dict[str, list[np.ndarray]] = {
        name: [] for name in loaded_analysis_branches
    }
    sample_summaries: list[dict[str, Any]] = []
    totals = _empty_dataset(
        config, loaded_score_names, loaded_analysis_branches
    ).totals.copy()

    for sample_number, sample in enumerate(config.samples, start=1):
        sample_norm, gen_sumw, xsec_fb = _sample_normalization(
            sample, config, gen_sumw_map, xsec_map
        )
        summary: dict[str, Any] = {
            "name": sample.name,
            "dataset": sample.dataset,
            "label": sample.label,
            "sample_selection": sample.selection,
            "files_discovered": len(sample.files),
            "files_attempted": 0,
            "files_processed": 0,
            "files_missing_tree": 0,
            "events_read": 0,
            "events_selected": 0,
            "events_classified": 0,
            "events_unclassified": 0,
            "sample_norm": float(sample_norm),
            "gen_sumw": float(gen_sumw),
            "xsec_fb": float(xsec_fb),
            "analysis_weight_sum": 0.0,
            "signed_weight_sum": 0.0,
        }
        print(
            f"[{sample_number}/{len(config.samples)}] {sample.name}: "
            f"{len(sample.files)} files discovered",
            flush=True,
        )
        valid_files = 0
        for path in sample.files:
            if config.max_files_per_sample is not None and valid_files >= config.max_files_per_sample:
                break
            summary["files_attempted"] += 1
            totals["files_attempted"] += 1
            with uproot.open(path) as root_file:
                if config.tree_name not in root_file:
                    summary["files_missing_tree"] += 1
                    totals["files_missing_tree"] += 1
                    continue
                tree = root_file[config.tree_name]
                valid_files += 1
                summary["files_processed"] += 1
                totals["files_processed"] += 1
                missing = sorted(set(requested) - set(tree.keys()))
                if missing:
                    raise KeyError(f"Missing branches in {path}: {', '.join(missing)}")

                for arrays in tree.iterate(requested, library="ak", step_size=config.uproot_step_size):
                    size = len(arrays[requested[0]])
                    summary["events_read"] += size
                    totals["events_read"] += size
                    columns = {
                        branch: _to_numpy(arrays[branch], flatten_first=branch in flatten_first)
                        for branch in requested
                    }
                    mask = evaluate_mask(sample.selection, columns, size)
                    mask &= evaluate_mask(config.selection, columns, size)
                    if not np.any(mask):
                        continue
                    selected_columns = {name: values[mask] for name, values in columns.items()}
                    selected = int(np.sum(mask))
                    truth_index = assign_truth_categories(
                        config.truth_categories, selected_columns, selected
                    )
                    classified = truth_index >= 0
                    n_classified = int(np.sum(classified))
                    n_unclassified = selected - n_classified

                    raw_weight = np.asarray(
                        selected_columns[config.weight_branch], dtype=np.float64
                    )
                    analysis_weight = sample_norm * np.abs(raw_weight)
                    signed_weight = sample_norm * raw_weight
                    for branch, score_name in score_by_branch.items():
                        score_buffers[score_name].append(
                            np.asarray(selected_columns[branch], dtype=np.float64)
                        )
                    truth_buffers.append(truth_index)
                    sample_index_buffers.append(
                        np.full(selected, sample_number - 1, dtype=np.int16)
                    )
                    raw_weight_buffers.append(raw_weight)
                    analysis_weight_buffers.append(analysis_weight)
                    signed_weight_buffers.append(signed_weight)
                    for branch in loaded_analysis_branches:
                        analysis_buffers[branch].append(np.asarray(selected_columns[branch]))

                    summary["events_selected"] += selected
                    summary["events_classified"] += n_classified
                    summary["events_unclassified"] += n_unclassified
                    summary["analysis_weight_sum"] += float(np.sum(analysis_weight[classified]))
                    summary["signed_weight_sum"] += float(np.sum(signed_weight[classified]))
                    totals["events_selected"] += selected
                    totals["events_classified"] += n_classified
                    totals["events_unclassified"] += n_unclassified

        totals["files_discovered"] += len(sample.files)
        sample_summaries.append(summary)

    if not truth_buffers:
        raise ValueError("No events passed the configured sample and event selections.")
    return NnDataset(
        scores={name: np.concatenate(parts) for name, parts in score_buffers.items()},
        truth_index=np.concatenate(truth_buffers),
        sample_index=np.concatenate(sample_index_buffers),
        raw_weight=np.concatenate(raw_weight_buffers),
        analysis_weight=np.concatenate(analysis_weight_buffers),
        signed_weight=np.concatenate(signed_weight_buffers),
        analysis_columns={
            name: np.concatenate(parts) if parts else np.array([], dtype=np.float64)
            for name, parts in analysis_buffers.items()
        },
        sample_summaries=sample_summaries,
        totals=totals,
    )
