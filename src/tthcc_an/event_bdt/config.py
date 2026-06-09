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
DEFAULT_XGBOOST_COMMON_PARAMS = {
    "tree_method": "hist",
    "eta": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
DEFAULT_BINARY_XGBOOST_PARAMS = {
    **DEFAULT_XGBOOST_COMMON_PARAMS,
    "objective": "binary:logistic",
    "eval_metric": ["auc", "logloss"],
}
DEFAULT_MULTICLASS_XGBOOST_PARAMS = {
    **DEFAULT_XGBOOST_COMMON_PARAMS,
    "objective": "multi:softprob",
    "eval_metric": ["mlogloss", "merror"],
}
BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
TRAINING_GROUP_SIGNAL = "signal"
TRAINING_GROUP_BACKGROUND = "background"
TRAINING_GROUP_CHOICES = {
    TRAINING_GROUP_SIGNAL,
    TRAINING_GROUP_BACKGROUND,
}


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
class EventBdtTrainingClass:
    name: str
    label: str
    group: str
    processes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "group": self.group,
            "processes": list(self.processes),
        }


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
    analysis_branches: list[str]
    spectators: list[str]
    weight_branch: str
    k_folds: int
    random_seed: int
    max_files_per_sample: int | None
    uproot_step_size: str
    flatten_first_branches: list[str]
    training_mode: str
    training_classes: list[EventBdtTrainingClass]
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
        requested.update(self.analysis_branches)
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

    @property
    def class_names(self) -> list[str]:
        return [item.name for item in self.training_classes]

    @property
    def class_labels(self) -> list[str]:
        return [item.label for item in self.training_classes]

    @property
    def class_groups(self) -> list[str]:
        return [item.group for item in self.training_classes]

    @property
    def num_training_classes(self) -> int:
        return len(self.training_classes)

    @property
    def signal_processes(self) -> list[str]:
        processes: list[str] = []
        for item in self.training_classes:
            if item.group == TRAINING_GROUP_SIGNAL:
                processes.extend(item.processes)
        return processes

    @property
    def background_processes(self) -> list[str]:
        processes: list[str] = []
        for item in self.training_classes:
            if item.group == TRAINING_GROUP_BACKGROUND:
                processes.extend(item.processes)
        return processes

    @property
    def signal_class_indices(self) -> list[int]:
        return [
            index
            for index, item in enumerate(self.training_classes)
            if item.group == TRAINING_GROUP_SIGNAL
        ]

    @property
    def background_class_indices(self) -> list[int]:
        return [
            index
            for index, item in enumerate(self.training_classes)
            if item.group == TRAINING_GROUP_BACKGROUND
        ]

    @property
    def process_to_class_index(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for class_index, item in enumerate(self.training_classes):
            for process in item.processes:
                mapping[process] = class_index
        return mapping

    @property
    def training_class_payload(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.training_classes]



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



def _normalize_training_classes(
    study: dict[str, Any],
    eval_processes_extra: list[str],
) -> tuple[str, list[EventBdtTrainingClass]]:
    training_classes_payload = study.get("training_classes")
    if training_classes_payload is None:
        signal_processes = list(study.get("signal_processes", []))
        background_processes = list(study.get("background_processes", []))
        if not signal_processes or not background_processes:
            raise ValueError(
                "Config must define either 'study.training_classes' or both "
                "'study.signal_processes' and 'study.background_processes'."
            )
        return (
            BINARY_MODE,
            [
                EventBdtTrainingClass(
                    name="signal",
                    label="Signal",
                    group=TRAINING_GROUP_SIGNAL,
                    processes=signal_processes,
                ),
                EventBdtTrainingClass(
                    name="background",
                    label="Background",
                    group=TRAINING_GROUP_BACKGROUND,
                    processes=background_processes,
                ),
            ],
        )

    if not isinstance(training_classes_payload, list) or not training_classes_payload:
        raise ValueError("Config field 'study.training_classes' must be a non-empty list.")

    training_classes: list[EventBdtTrainingClass] = []
    seen_names: set[str] = set()
    seen_processes: set[str] = set()
    for index, raw_class in enumerate(training_classes_payload):
        name = str(raw_class.get("name", "")).strip()
        label = str(raw_class.get("label", name)).strip()
        group = str(raw_class.get("group", "")).strip()
        processes = [str(item).strip() for item in raw_class.get("processes", [])]
        if not name:
            raise ValueError(f"Training class at index {index} is missing a non-empty 'name'.")
        if name in seen_names:
            raise ValueError(f"Duplicate training class name: {name}")
        if not label:
            raise ValueError(f"Training class '{name}' is missing a non-empty 'label'.")
        if group not in TRAINING_GROUP_CHOICES:
            choices = ", ".join(sorted(TRAINING_GROUP_CHOICES))
            raise ValueError(
                f"Training class '{name}' has invalid group '{group}'. Allowed: {choices}."
            )
        if not processes:
            raise ValueError(f"Training class '{name}' must define a non-empty process list.")

        overlap = seen_processes.intersection(processes)
        if overlap:
            repeated = ", ".join(sorted(overlap))
            raise ValueError(
                f"Training class '{name}' reuses process(es) already assigned elsewhere: {repeated}"
            )
        eval_overlap = set(processes).intersection(eval_processes_extra)
        if eval_overlap:
            repeated = ", ".join(sorted(eval_overlap))
            raise ValueError(
                f"Training class '{name}' overlaps with eval-only processes: {repeated}"
            )

        seen_names.add(name)
        seen_processes.update(processes)
        training_classes.append(
            EventBdtTrainingClass(
                name=name,
                label=label,
                group=group,
                processes=processes,
            )
        )

    groups = {item.group for item in training_classes}
    if TRAINING_GROUP_SIGNAL not in groups:
        raise ValueError("Config field 'study.training_classes' must include at least one signal class.")
    if TRAINING_GROUP_BACKGROUND not in groups:
        raise ValueError(
            "Config field 'study.training_classes' must include at least one background class."
        )
    if len(training_classes) <= 1:
        raise ValueError("At least two training classes are required.")

    training_mode = BINARY_MODE if len(training_classes) == 2 else MULTICLASS_MODE
    return training_mode, training_classes



def _build_default_xgboost_params(
    training_mode: str,
    num_training_classes: int,
) -> dict[str, Any]:
    if training_mode == MULTICLASS_MODE:
        params = dict(DEFAULT_MULTICLASS_XGBOOST_PARAMS)
        params["num_class"] = num_training_classes
        return params
    return dict(DEFAULT_BINARY_XGBOOST_PARAMS)



def _normalize_xgboost_params(
    raw_xgboost_payload: dict[str, Any],
    training_mode: str,
    num_training_classes: int,
) -> dict[str, Any]:
    xgboost_payload = _build_default_xgboost_params(
        training_mode=training_mode,
        num_training_classes=num_training_classes,
    )
    configured_num_class = raw_xgboost_payload.get("num_class")
    if training_mode == MULTICLASS_MODE:
        if configured_num_class is not None and int(configured_num_class) != num_training_classes:
            raise ValueError(
                "Configured xgboost.num_class does not match the number of training classes: "
                f"{configured_num_class} vs {num_training_classes}."
            )
    else:
        raw_xgboost_payload.pop("num_class", None)
    xgboost_payload.update(raw_xgboost_payload)
    if training_mode == MULTICLASS_MODE:
        xgboost_payload["num_class"] = num_training_classes
    return xgboost_payload



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
    analysis_branches = list(study.get("analysis_branches", []))
    spectators = list(study.get("spectators", []))
    selection_branches = list(study.get("selection_branches", []))
    if not selection_branches:
        raise ValueError("Config field 'study.selection_branches' must be a non-empty list.")

    flatten_first_branches = list(study.get("flatten_first_branches", []))
    if not flatten_first_branches:
        flatten_first_branches = [
            branch
            for branch in set(features) | set(analysis_branches) | set(spectators) | set(selection_branches)
            if branch.startswith("TargetFatJet_")
        ]

    eval_processes_extra = [str(item) for item in study.get("eval_processes_extra", [])]
    training_mode, training_classes = _normalize_training_classes(
        study,
        eval_processes_extra=eval_processes_extra,
    )

    raw_xgboost_payload = dict(payload.get("xgboost", {}))
    num_boost_round = int(raw_xgboost_payload.pop("num_boost_round", DEFAULT_NUM_BOOST_ROUND))
    early_stopping_rounds = int(
        raw_xgboost_payload.pop("early_stopping_rounds", DEFAULT_EARLY_STOPPING_ROUNDS)
    )
    xgboost_payload = _normalize_xgboost_params(
        raw_xgboost_payload=raw_xgboost_payload,
        training_mode=training_mode,
        num_training_classes=len(training_classes),
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
        analysis_branches=analysis_branches,
        spectators=spectators,
        weight_branch=str(study.get("weight_branch", DEFAULT_WEIGHT_BRANCH)),
        k_folds=int(study.get("k_folds", DEFAULT_K_FOLDS)),
        random_seed=int(study.get("random_seed", DEFAULT_RANDOM_SEED)),
        max_files_per_sample=study.get("max_files_per_sample"),
        uproot_step_size=str(study.get("uproot_step_size", DEFAULT_UPROOT_STEP_SIZE)),
        flatten_first_branches=flatten_first_branches,
        training_mode=training_mode,
        training_classes=training_classes,
        eval_processes_extra=eval_processes_extra,
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
