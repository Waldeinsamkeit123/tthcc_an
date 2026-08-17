from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tthcc_an.config_loader import expand_file_patterns, load_json_maybe_with_comments


DEFAULT_TREE_NAME = "Events"
DEFAULT_WEIGHT_BRANCH = "weight"
DEFAULT_UPROOT_STEP_SIZE = "200 MB"
DEFAULT_OUTDIR = "outputs/trigger_efficiency_signal_2024_hlt_v1"


@dataclass(frozen=True)
class TriggerSample:
    name: str
    dataset: str
    process: str
    label: str
    files: list[str]


@dataclass(frozen=True)
class TriggerVariable:
    name: str
    label: str
    bins: list[float]
    branch: str
    index: int | None
    plot_groups: list[str] | None


@dataclass(frozen=True)
class TriggerEfficiencyConfig:
    config_path: Path
    repo_root: Path
    outdir: Path
    tree_name: str
    samples: list[TriggerSample]
    triggers: list[str]
    trigger_groups: dict[str, list[str]]
    or_groups: dict[str, list[str]]
    plot_groups: dict[str, list[str]]
    variables: list[TriggerVariable]
    weight_branch: str
    gen_sumw_file: str | None
    xsec_file: str | None
    lumi_fb: float | None
    uproot_step_size: str
    max_files_per_sample: int | None

    @property
    def plot_dir(self) -> Path:
        return self.outdir / "plots"

    @property
    def table_dir(self) -> Path:
        return self.outdir / "tables"

    @property
    def summary_path(self) -> Path:
        return self.outdir / "summary.json"


def _resolve_relative_to(path: str | Path, anchor: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (anchor / candidate).resolve()


def _resolve_outdir(repo_root: Path, configured_outdir: str | None, explicit_outdir: str | None) -> Path:
    outdir = explicit_outdir or configured_outdir or DEFAULT_OUTDIR
    candidate = Path(outdir)
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def load_trigger_efficiency_config(
    path: str | Path,
    repo_root: str | Path,
    *,
    outdir: str | None = None,
    max_files_per_sample: int | None = None,
) -> TriggerEfficiencyConfig:
    config_path = Path(path).resolve()
    repo_root_path = Path(repo_root).resolve()
    payload = load_json_maybe_with_comments(config_path)
    study = dict(payload.get("study", {}))

    samples: list[TriggerSample] = []
    for entry in list(payload.get("samples", [])):
        files = expand_file_patterns(list(entry["files"]))
        effective_max_files = max_files_per_sample
        if effective_max_files is None:
            effective_max_files = study.get("max_files_per_sample")
        if effective_max_files is not None:
            files = files[: int(effective_max_files)]
        if not files:
            raise ValueError(f"No input files were found for sample '{entry.get('name', 'unknown')}'.")
        name = str(entry["name"])
        samples.append(
            TriggerSample(
                name=name,
                dataset=str(entry.get("dataset", name)),
                process=str(entry.get("process", name)),
                label=str(entry.get("label", entry.get("process", name))),
                files=files,
            )
        )
    if not samples:
        raise ValueError(f"No samples found in trigger-efficiency config: {config_path}")

    triggers = [str(trigger) for trigger in list(study.get("triggers", []))]
    if not triggers:
        raise ValueError("Config field 'study.triggers' must be a non-empty list.")

    variables: list[TriggerVariable] = []
    for variable in list(study.get("variables", [])):
        variable_name = str(variable["name"])
        index_raw = variable.get("index")
        index = int(index_raw) if index_raw is not None else None
        if index is not None and index < 0:
            raise ValueError(f"Variable {variable_name} index must be non-negative.")
        variables.append(
            TriggerVariable(
                name=variable_name,
                label=str(variable.get("label", variable_name)),
                bins=[float(edge) for edge in list(variable["bins"])],
                branch=str(variable.get("branch", variable_name)),
                index=index,
                plot_groups=[str(group) for group in list(variable["plot_groups"])]
                if variable.get("plot_groups") is not None
                else None,
            )
        )
    if not variables:
        raise ValueError("Config field 'study.variables' must be a non-empty list.")
    for variable in variables:
        if len(variable.bins) < 2:
            raise ValueError(f"Variable '{variable.name}' must define at least two bin edges.")

    normalization = dict(payload.get("normalization", {}))
    xsec_file = normalization.get("xsec_file")
    gen_sumw_file = normalization.get("gen_sumw_file")
    return TriggerEfficiencyConfig(
        config_path=config_path,
        repo_root=repo_root_path,
        outdir=_resolve_outdir(repo_root_path, study.get("outdir"), outdir),
        tree_name=str(study.get("tree_name", DEFAULT_TREE_NAME)),
        samples=samples,
        triggers=triggers,
        trigger_groups={
            str(key): [str(trigger) for trigger in list(value)]
            for key, value in dict(study.get("trigger_groups", {})).items()
        },
        or_groups={
            str(key): [str(trigger) for trigger in list(value)]
            for key, value in dict(study.get("or_groups", {})).items()
        },
        plot_groups={
            str(key): [str(trigger) for trigger in list(value)]
            for key, value in dict(study.get("plot_groups", {})).items()
        },
        variables=variables,
        weight_branch=str(study.get("weight_branch", DEFAULT_WEIGHT_BRANCH)),
        gen_sumw_file=str(_resolve_relative_to(gen_sumw_file, config_path.parent)) if gen_sumw_file else None,
        xsec_file=str(_resolve_relative_to(xsec_file, config_path.parent)) if xsec_file else None,
        lumi_fb=float(normalization["lumi_fb"]) if normalization.get("lumi_fb") is not None else None,
        uproot_step_size=str(study.get("uproot_step_size", DEFAULT_UPROOT_STEP_SIZE)),
        max_files_per_sample=study.get("max_files_per_sample"),
    )
