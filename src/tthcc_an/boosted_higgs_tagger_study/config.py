from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from tthcc_an.config_loader import (
    DEFAULT_CANDIDATE_STRATEGY,
    DEFAULT_ETA_MAX,
    DEFAULT_MSD_WINDOW_HIGH,
    DEFAULT_MSD_WINDOW_LOW,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLOT_OPTIONS,
    DEFAULT_PT_MIN,
    DEFAULT_SCORES,
    DEFAULT_SCORE_HIST_BINS,
    DEFAULT_SIG_EFFS,
    DEFAULT_TARGETS,
    DEFAULT_TREE_NAME,
    DEFAULT_UPROOT_STEP_SIZE,
    DEFAULT_WEIGHT_BRANCH,
    DEFAULT_XBB_VS_XCC_REGION_PRESET,
    SampleConfig,
    StudyConfig,
    expand_file_patterns,
    load_json_maybe_with_comments,
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_boosted_config_payload(
    path: str | Path,
    *,
    _seen: set[Path] | None = None,
) -> dict[str, Any]:
    config_path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if config_path in seen:
        raise ValueError(f"Cyclic boosted-study config inheritance at {config_path}")
    seen.add(config_path)

    payload = load_json_maybe_with_comments(config_path)
    parent = payload.get("extends")
    if parent is None:
        return payload
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    parent_payload = load_boosted_config_payload(parent_path, _seen=seen)
    return _deep_merge(parent_payload, payload)


def resolve_output_dir(
    config_path: str | Path,
    explicit_outdir: str | None,
    repo_root: str | Path,
) -> str:
    if explicit_outdir is not None:
        return explicit_outdir
    payload = load_boosted_config_payload(config_path)
    configured = Path(payload.get("study", {}).get("outdir", DEFAULT_OUTPUT_DIR))
    if configured.is_absolute():
        return str(configured)
    return str((Path(repo_root) / configured).resolve())


def load_study_config(
    path: str | Path,
    max_files_per_sample: int | None,
    default_xsec_file: str,
) -> StudyConfig:
    payload = load_boosted_config_payload(path)
    samples_raw = payload.get("samples", [])
    if not samples_raw:
        raise ValueError(f"No samples found in boosted-study configuration: {path}")

    samples: list[SampleConfig] = []
    for entry in samples_raw:
        files = expand_file_patterns(list(entry["files"]))
        if max_files_per_sample is not None:
            files = files[:max_files_per_sample]
        name = str(entry["name"])
        samples.append(
            SampleConfig(
                name=name,
                dataset=str(entry.get("dataset", name)),
                process=str(entry.get("process", name)),
                label=str(entry.get("label", name)),
                files=files,
            )
        )

    normalization = payload.get("normalization", {})
    return StudyConfig(
        samples=samples,
        gen_sumw_file=normalization.get("gen_sumw_file"),
        xsec_file=normalization.get("xsec_file", default_xsec_file),
        lumi_fb=normalization.get("lumi_fb"),
        study_defaults=dict(payload.get("study", {})),
        plot_defaults=dict(payload.get("plot", {})),
    )
