from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tthcc_an.nn_study.config import NnStudyConfig
from tthcc_an.nn_study.dataset import NnDataset


CACHE_SCHEMA_VERSION = 1
DEFAULT_CACHE_NAME = "prepared_events.npz"


def resolve_cache_path(config: NnStudyConfig, explicit: str | None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    return config.outdir / "cache" / DEFAULT_CACHE_NAME


def cache_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".metadata.json")


def cache_score_names(config: NnStudyConfig) -> list[str]:
    return list(config.all_score_names)


def cache_analysis_branches(config: NnStudyConfig) -> list[str]:
    branches = list(config.analysis_branches)
    if config.mass_sculpting.enabled:
        branches.extend(
            variable.branch for variable in config.mass_sculpting.variables
        )
    if config.weighting_diagnostics.enabled:
        branches.extend(config.weighting_diagnostics.reweight_variables)
    if (
        config.significance_mass_window.enabled
        and config.significance_mass_window.branch is not None
    ):
        branches.append(config.significance_mass_window.branch)
    return list(dict.fromkeys(branches))


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_hash(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _sample_definitions(config: NnStudyConfig) -> list[dict[str, str]]:
    return [
        {
            "name": sample.name,
            "dataset": sample.dataset,
            "label": sample.label,
            "selection": sample.selection,
        }
        for sample in config.samples
    ]


def _truth_definitions(config: NnStudyConfig) -> list[dict[str, str]]:
    return [
        {
            "name": truth.name,
            "expression": truth.expression,
        }
        for truth in config.truth_categories
    ]


def cache_definition(config: NnStudyConfig) -> dict[str, Any]:
    samples = _sample_definitions(config)
    truths = _truth_definitions(config)
    return {
        "channel": config.channel,
        "year": config.year,
        "energy_tev": config.energy_tev,
        "input_location": config.input_location,
        "sample_file_pattern": config.sample_file_pattern,
        "tree_name": config.tree_name,
        "weight_branch": config.weight_branch,
        "selection": config.selection,
        "selection_branches": config.selection_branches,
        "truth_branches": config.truth_branches,
        "flatten_first_branches": config.flatten_first_branches,
        "truth_definitions": truths,
        "truth_definition_hash": _payload_hash(truths),
        "sample_definitions": samples,
        "sample_definition_hash": _payload_hash(samples),
        "normalization": {
            "lumi_fb": config.lumi_fb,
            "gen_sumw_file": config.gen_sumw_file,
            "gen_sumw_sha256": _sha256_file(config.gen_sumw_file),
            "xsec_file": config.xsec_file,
            "xsec_sha256": _sha256_file(config.xsec_file),
        },
        "max_files_per_sample": config.max_files_per_sample,
    }


def input_file_fingerprint(config: NnStudyConfig) -> dict[str, Any]:
    digest = hashlib.sha256()
    per_sample: dict[str, int] = {}
    total = 0
    for sample in config.samples:
        files = sorted(sample.files)
        per_sample[sample.name] = len(files)
        total += len(files)
        for path in files:
            digest.update(sample.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.encode("utf-8"))
            digest.update(b"\n")
    return {
        "algorithm": "sha256(sample name + sorted discovered file paths)",
        "sha256": digest.hexdigest(),
        "file_count": total,
        "per_sample_file_count": per_sample,
        "content_or_mtime_hashed": False,
    }


def _stored_score_metadata(
    config: NnStudyConfig, dataset: NnDataset
) -> dict[str, dict[str, str]]:
    score_by_name = {score.name: score for score in config.all_scores}
    return {
        name: {
            "branch": score_by_name[name].branch,
            "array_key": f"score_{index:03d}",
            "dtype": str(dataset.scores[name].dtype),
        }
        for index, name in enumerate(cache_score_names(config))
    }


def _stored_analysis_metadata(
    config: NnStudyConfig, dataset: NnDataset
) -> dict[str, dict[str, str]]:
    return {
        branch: {
            "array_key": f"analysis_{index:03d}",
            "dtype": str(dataset.analysis_columns[branch].dtype),
        }
        for index, branch in enumerate(cache_analysis_branches(config))
    }


def write_nn_cache(
    *,
    config: NnStudyConfig,
    dataset: NnDataset,
    cache_path: Path,
) -> dict[str, Any]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_metadata_path(cache_path)
    score_metadata = _stored_score_metadata(config, dataset)
    analysis_metadata = _stored_analysis_metadata(config, dataset)
    arrays: dict[str, np.ndarray] = {
        "truth_index": dataset.truth_index,
        "sample_index": dataset.sample_index,
        "raw_weight": dataset.raw_weight,
        "analysis_weight": dataset.analysis_weight,
        "signed_weight": dataset.signed_weight,
    }
    arrays.update(
        {
            entry["array_key"]: dataset.scores[name]
            for name, entry in score_metadata.items()
        }
    )
    arrays.update(
        {
            entry["array_key"]: dataset.analysis_columns[branch]
            for branch, entry in analysis_metadata.items()
        }
    )

    temporary_cache = cache_path.with_suffix(".tmp.npz")
    np.savez(temporary_cache, **arrays)
    temporary_cache.replace(cache_path)

    definition = cache_definition(config)
    requested_root_branches = config.requested_branches_for(
        score_names=cache_score_names(config),
        analysis_branches=cache_analysis_branches(config),
    )
    metadata: dict[str, Any] = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_format": "numpy np.savez (uncompressed NPZ)",
        "cache_path": str(cache_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "definition": definition,
        "definition_hash": _payload_hash(definition),
        "input_files": input_file_fingerprint(config),
        "stored_arrays": {
            "core": {
                name: {"array_key": name, "dtype": str(array.dtype)}
                for name, array in arrays.items()
                if name
                in {
                    "truth_index",
                    "sample_index",
                    "raw_weight",
                    "analysis_weight",
                    "signed_weight",
                }
            },
            "scores": score_metadata,
            "analysis_columns": analysis_metadata,
        },
        "stored_score_names": list(score_metadata),
        "stored_score_branches": [
            entry["branch"] for entry in score_metadata.values()
        ],
        "stored_analysis_branches": list(analysis_metadata),
        "requested_root_branches": requested_root_branches,
        "number_of_selected_events": int(dataset.truth_index.size),
        "number_of_input_files_discovered": int(
            dataset.totals["files_discovered"]
        ),
        "dataset_totals": dataset.totals,
        "sample_summaries": dataset.sample_summaries,
        "cache_size_bytes": cache_path.stat().st_size,
    }
    temporary_metadata = metadata_path.with_suffix(".tmp.json")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)
    return metadata


def read_cache_metadata(cache_path: Path) -> dict[str, Any]:
    metadata_path = cache_metadata_path(cache_path)
    if not cache_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"NN-study cache is incomplete or missing: {cache_path}. "
            "Rebuild with --prepare-cache --force."
        )
    with metadata_path.open(encoding="utf-8") as source:
        metadata = json.load(source)
    if metadata.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            "NN-study cache schema is incompatible; rebuild with "
            "--prepare-cache --force."
        )
    return metadata


def validate_nn_cache(
    *,
    config: NnStudyConfig,
    cache_path: Path,
    check_input_files: bool,
) -> dict[str, Any]:
    metadata = read_cache_metadata(cache_path)
    current_definition = cache_definition(config)
    cached_definition = metadata.get("definition", {})
    if metadata.get("definition_hash") != _payload_hash(current_definition):
        changed = sorted(
            key
            for key in set(current_definition) | set(cached_definition)
            if current_definition.get(key) != cached_definition.get(key)
        )
        raise ValueError(
            "NN-study cache is incompatible with the current dataset definition; "
            f"changed fields: {', '.join(changed)}. Rebuild with "
            "--prepare-cache --force."
        )

    stored_scores = dict(metadata["stored_arrays"]["scores"])
    score_by_name = {score.name: score for score in config.all_scores}
    missing_scores = [
        f"{name} ({score_by_name[name].branch})"
        for name in cache_score_names(config)
        if name not in stored_scores
        or stored_scores[name].get("branch") != score_by_name[name].branch
    ]
    required_analysis = cache_analysis_branches(config)
    stored_analysis = set(metadata["stored_arrays"]["analysis_columns"])
    missing_analysis = [
        branch for branch in required_analysis if branch not in stored_analysis
    ]
    if missing_scores or missing_analysis:
        missing = [*missing_scores, *missing_analysis]
        raise ValueError(
            "Cache is missing required branch(es): "
            + ", ".join(missing)
            + "; rebuild with --prepare-cache --force."
        )

    if check_input_files:
        current_files = input_file_fingerprint(config)
        if current_files != metadata.get("input_files"):
            raise ValueError(
                "NN-study discovered input-file set differs from the cache; "
                "rebuild with --prepare-cache --force."
            )
    return metadata


def load_nn_cache(
    *,
    config: NnStudyConfig,
    cache_path: Path,
    score_names: list[str],
    analysis_branches: list[str],
) -> tuple[NnDataset, dict[str, Any]]:
    metadata = validate_nn_cache(
        config=config,
        cache_path=cache_path,
        check_input_files=False,
    )
    score_metadata = metadata["stored_arrays"]["scores"]
    analysis_metadata = metadata["stored_arrays"]["analysis_columns"]
    with np.load(cache_path, allow_pickle=False) as arrays:
        required_keys = {
            "truth_index",
            "sample_index",
            "raw_weight",
            "analysis_weight",
            "signed_weight",
            *(score_metadata[name]["array_key"] for name in score_names),
            *(
                analysis_metadata[branch]["array_key"]
                for branch in analysis_branches
            ),
        }
        missing_keys = sorted(required_keys - set(arrays.files))
        if missing_keys:
            raise ValueError(
                "NN-study cache payload is missing array key(s): "
                + ", ".join(missing_keys)
                + "; rebuild with --prepare-cache --force."
            )
        dataset = NnDataset(
            scores={
                name: np.asarray(arrays[score_metadata[name]["array_key"]])
                for name in score_names
            },
            truth_index=np.asarray(arrays["truth_index"]),
            sample_index=np.asarray(arrays["sample_index"]),
            raw_weight=np.asarray(arrays["raw_weight"]),
            analysis_weight=np.asarray(arrays["analysis_weight"]),
            signed_weight=np.asarray(arrays["signed_weight"]),
            analysis_columns={
                branch: np.asarray(
                    arrays[analysis_metadata[branch]["array_key"]]
                )
                for branch in analysis_branches
            },
            sample_summaries=list(metadata["sample_summaries"]),
            totals={
                str(name): int(value)
                for name, value in dict(metadata["dataset_totals"]).items()
            },
        )
    return dataset, metadata
