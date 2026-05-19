from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tthcc_an.config_loader import expand_file_patterns, load_json_maybe_with_comments


DEFAULT_OUTPUT_DIR = "outputs/event_bdt"
DEFAULT_TREE_NAME = "Events"
DEFAULT_WEIGHT_BRANCH = "weight"
DEFAULT_UPROOT_STEP_SIZE = "200 MB"
DEFAULT_K_FOLDS = 5
DEFAULT_RANDOM_SEED = 42
DEFAULT_PREPARED_INPUTS_NAME = "prepared_inputs.npz"
DEFAULT_PREDICTIONS_NAME = "predictions.npz"
DEFAULT_SUMMARY_NAME = "training_summary.json"
DEFAULT_FEATURE_IMPORTANCE_NAME = "feature_importance.json"
DEFAULT_MODEL_DIR_NAME = "models"
DEFAULT_PLOT_DIR_NAME = "plots"
DEFAULT_MODEL_NAME = "xgb_event_bdt"
DEFAULT_NUM_BOOST_ROUND = 500
DEFAULT_EARLY_STOPPING_ROUNDS = 30


@dataclass
class EventBdtSampleConfig:
    name: str
    dataset: str
    process: str
    label: str
    role: str
    files: list[str]


@dataclass
class EventBdtSamplesConfig:
    samples: list[EventBdtSampleConfig]
    gen_sumw_file: str | None
    xsec_file: str | None
    lumi_fb: float | None


@dataclass
class EventBdtConfig:
    config_path: Path
    repo_root: Path
    samples_config_path: Path
    outdir: Path
    tree_name: str
    base_selection: str
    selection_branches: list[str]
    features: list[str]
    spectators: list[str]
    weight_branch: str
    k_folds: int
    random_seed: int
    max_files_per_sample: int | None
    uproot_step_size: str
    flatten_first_branches: list[str]
    signal_processes: list[str]
    background_processes: list[str]
    eval_processes_extra: list[str]
    model_name: str
    prepared_inputs_name: str
    predictions_name: str
    summary_name: str
    feature_importance_name: str
    model_dir_name: str
    plot_dir_name: str
    xgboost_params: dict[str, Any]
    num_boost_round: int
    early_stopping_rounds: int
    reweighting: dict[str, Any]

    def requested_branches(self) -> list[str]:
        requested = set(self.features)
        requested.update(self.spectators)
        requested.update(self.selection_branches)
        if self.weight_branch:
            requested.add(self.weight_branch)
        return sorted(requested)

    @property
    def prepared_inputs_path(self) -> Path:
        return self.outdir / self.prepared_inputs_name

    @property
    def predictions_path(self) -> Path:
        return self.outdir / self.predictions_name

    @property
    def summary_path(self) -> Path:
        return self.outdir / self.summary_name

    @property
    def feature_importance_path(self) -> Path:
        return self.outdir / self.feature_importance_name

    @property
    def model_dir_path(self) -> Path:
        return self.outdir / self.model_dir_name

    @property
    def plot_dir_path(self) -> Path:
        return self.outdir / self.plot_dir_name


def _resolve_relative_to(path: str | Path, anchor: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (anchor / candidate).resolve()


def _resolve_output_dir(repo_root: Path, configured_outdir: str | None) -> Path:
    outdir = configured_outdir or DEFAULT_OUTPUT_DIR
    candidate = Path(outdir)
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def load_event_bdt_samples_config(
    path: str | Path,
    max_files_per_sample: int | None = None,
) -> EventBdtSamplesConfig:
    payload = load_json_maybe_with_comments(path)
    samples_raw = payload.get("samples", [])
    if not samples_raw:
        raise ValueError(f"No samples found in sample configuration: {path}")

    samples: list[EventBdtSampleConfig] = []
    for entry in samples_raw:
        files = expand_file_patterns(list(entry["files"]))
        if max_files_per_sample is not None:
            files = files[:max_files_per_sample]
        if not files:
            raise ValueError(f"No input files were found for sample '{entry.get('name', 'unknown')}'.")
        name = str(entry["name"])
        samples.append(
            EventBdtSampleConfig(
                name=name,
                dataset=str(entry.get("dataset", name)),
                process=str(entry.get("process", name)),
                label=str(entry.get("label", entry.get("process", name))),
                role=str(entry.get("role", "background")),
                files=files,
            )
        )

    normalization = payload.get("normalization", {})
    lumi_fb = normalization.get("lumi_fb")
    return EventBdtSamplesConfig(
        samples=samples,
        gen_sumw_file=normalization.get("gen_sumw_file"),
        xsec_file=normalization.get("xsec_file"),
        lumi_fb=float(lumi_fb) if lumi_fb is not None else None,
    )


def load_event_bdt_config(path: str | Path, repo_root: str | Path) -> EventBdtConfig:
    config_path = Path(path).resolve()
    repo_root_path = Path(repo_root).resolve()
    payload = load_json_maybe_with_comments(config_path)
    study = dict(payload.get("study", {}))

    samples_config_value = payload.get("samples_config") or payload.get("dataset_config")
    if samples_config_value is None:
        raise ValueError("The event-BDT config must define 'samples_config'.")

    samples_config_path = _resolve_relative_to(samples_config_value, config_path.parent)
    outdir = _resolve_output_dir(repo_root_path, study.get("outdir"))

    features = list(study.get("features", []))
    if not features:
        raise ValueError("Config field 'study.features' must be a non-empty list.")
    spectators = list(study.get("spectators", []))
    selection_branches = list(study.get("selection_branches", []))
    if not selection_branches:
        raise ValueError("Config field 'study.selection_branches' must be a non-empty list.")

    flatten_first_branches = list(study.get("flatten_first_branches", []))
    if not flatten_first_branches:
        flatten_first_branches = [
            branch
            for branch in set(features) | set(spectators) | set(selection_branches)
            if branch.startswith("TargetFatJet_")
        ]

    xgboost_payload = {
        "tree_method": "hist",
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "eta": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    xgboost_payload.update(dict(payload.get("xgboost", {})))
    num_boost_round = int(xgboost_payload.pop("num_boost_round", DEFAULT_NUM_BOOST_ROUND))
    early_stopping_rounds = int(
        xgboost_payload.pop("early_stopping_rounds", DEFAULT_EARLY_STOPPING_ROUNDS)
    )

    signal_processes = list(study.get("signal_processes", []))
    background_processes = list(study.get("background_processes", []))
    if not signal_processes or not background_processes:
        raise ValueError(
            "Config fields 'study.signal_processes' and 'study.background_processes' must both be non-empty lists."
        )

    return EventBdtConfig(
        config_path=config_path,
        repo_root=repo_root_path,
        samples_config_path=samples_config_path,
        outdir=outdir,
        tree_name=str(study.get("tree_name", DEFAULT_TREE_NAME)),
        base_selection=str(study.get("base_selection", "")).strip(),
        selection_branches=selection_branches,
        features=features,
        spectators=spectators,
        weight_branch=str(study.get("weight_branch", DEFAULT_WEIGHT_BRANCH)),
        k_folds=int(study.get("k_folds", DEFAULT_K_FOLDS)),
        random_seed=int(study.get("random_seed", DEFAULT_RANDOM_SEED)),
        max_files_per_sample=study.get("max_files_per_sample"),
        uproot_step_size=str(study.get("uproot_step_size", DEFAULT_UPROOT_STEP_SIZE)),
        flatten_first_branches=flatten_first_branches,
        signal_processes=signal_processes,
        background_processes=background_processes,
        eval_processes_extra=list(study.get("eval_processes_extra", [])),
        model_name=str(study.get("model_name", DEFAULT_MODEL_NAME)),
        prepared_inputs_name=str(study.get("prepared_inputs", DEFAULT_PREPARED_INPUTS_NAME)),
        predictions_name=str(study.get("predictions", DEFAULT_PREDICTIONS_NAME)),
        summary_name=str(study.get("summary", DEFAULT_SUMMARY_NAME)),
        feature_importance_name=str(
            study.get("feature_importance", DEFAULT_FEATURE_IMPORTANCE_NAME)
        ),
        model_dir_name=str(study.get("model_dir", DEFAULT_MODEL_DIR_NAME)),
        plot_dir_name=str(study.get("plot_dir", DEFAULT_PLOT_DIR_NAME)),
        xgboost_params=xgboost_payload,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        reweighting=dict(payload.get("reweighting", {})),
    )
