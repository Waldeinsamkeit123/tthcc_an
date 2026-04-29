from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "outputs/boosted_higgs_tagger_study"
DEFAULT_TREE_NAME = "Events"
DEFAULT_TARGETS = ["hcc", "hbb", "higgs"]
DEFAULT_SCORES = ["auto"]
DEFAULT_SIG_EFFS = [
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
]
DEFAULT_PT_MIN = 200.0
DEFAULT_ETA_MAX = 2.4
DEFAULT_CANDIDATE_STRATEGY = "mass_window_all_jets"
DEFAULT_MSD_WINDOW_LOW = 100.0
DEFAULT_MSD_WINDOW_HIGH = 150.0
DEFAULT_WEIGHT_BRANCH = "weight"
DEFAULT_UPROOT_STEP_SIZE = "200 MB"
DEFAULT_SCORE_HIST_BINS = 2000
DEFAULT_PLOT_OPTIONS = {
    "title_size": 16.0,
    "label_size": 12.5,
    "tick_size": 10.0,
    "legend_size": 10.0,
    "cms_size": 16.0,
    "dpi": 220,
}


@dataclass
class SampleConfig:
    name: str
    dataset: str
    process: str
    label: str
    files: list[str]


@dataclass
class StudyConfig:
    samples: list[SampleConfig]
    gen_sumw_file: str | None
    xsec_file: str | None
    lumi_fb: float | None
    study_defaults: dict[str, Any]
    plot_defaults: dict[str, Any]


def strip_hash_comments(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        in_string = False
        escaped = False
        out_chars: list[str] = []
        for char in line:
            if escaped:
                out_chars.append(char)
                escaped = False
                continue
            if char == "\\":
                out_chars.append(char)
                escaped = True
                continue
            if char == '"':
                out_chars.append(char)
                in_string = not in_string
                continue
            if char == "#" and not in_string:
                break
            out_chars.append(char)
        cleaned_lines.append("".join(out_chars))
    return "\n".join(cleaned_lines)


def load_json_maybe_with_comments(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    resolved_path = Path(path)
    return json.loads(strip_hash_comments(resolved_path.read_text(encoding="utf-8")))


def expand_file_patterns(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            files.extend(matches)
            continue
        if any(token in pattern for token in ["*", "?", "["]):
            raise FileNotFoundError(f"No files matched pattern: {pattern}")
        if not Path(pattern).exists():
            raise FileNotFoundError(f"Input file does not exist: {pattern}")
        files.append(pattern)
    return files


def resolve_output_dir(
    config_path: str | Path,
    explicit_outdir: str | None,
    repo_root: str | Path,
) -> str:
    if explicit_outdir is not None:
        return explicit_outdir
    payload = load_json_maybe_with_comments(config_path)
    configured_outdir = payload.get("study", {}).get("outdir", DEFAULT_OUTPUT_DIR)
    configured_path = Path(configured_outdir)
    if configured_path.is_absolute():
        return str(configured_path)
    return str((Path(repo_root) / configured_path).resolve())


def load_study_config(
    path: str | Path,
    max_files_per_sample: int | None,
    default_xsec_file: str,
) -> StudyConfig:
    payload = load_json_maybe_with_comments(path)
    samples_raw = payload.get("samples", [])
    if not samples_raw:
        raise ValueError(f"No samples found in configuration: {path}")

    samples: list[SampleConfig] = []
    for entry in samples_raw:
        files = expand_file_patterns(list(entry["files"]))
        if max_files_per_sample is not None:
            files = files[:max_files_per_sample]
        name = entry["name"]
        samples.append(
            SampleConfig(
                name=name,
                dataset=entry.get("dataset", name),
                process=entry.get("process", name),
                label=entry.get("label", name),
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
