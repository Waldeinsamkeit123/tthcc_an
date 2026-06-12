from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

from tthcc_an.event_bdt.config import (
    EventBdtConfig,
    load_event_bdt_samples_config,
    resolve_input_branch_name,
)
from tthcc_an.event_bdt.dataset import _compute_fold_ids, _extract_branch, _selection_mask
from tthcc_an.event_bdt.features import build_feature_matrix


BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
PREDICTION_MODE_MEAN = "mean"
PREDICTION_MODE_FOLD_ROUTED = "fold_routed"
PREDICTION_MODE_BOTH = "both"
PREDICTION_MODE_CHOICES = [
    PREDICTION_MODE_MEAN,
    PREDICTION_MODE_FOLD_ROUTED,
    PREDICTION_MODE_BOTH,
]
PREDICTION_ENTRY_CHUNK_SIZE = 5000


@dataclass
class FoldModel:
    fold: int
    booster: Any
    best_iteration: int


@dataclass
class TrainedEnsemble:
    training_mode: str
    feature_names: list[str]
    class_names: list[str]
    fold_models: list[FoldModel]
    k_folds: int



def _require_prediction_dependencies():
    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xgboost is required for event-BDT prediction. "
            "Please run inside an environment such as LCG108 where it is available."
        ) from exc
    return xgb



def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")



def _resolve_prediction_outdir(config: EventBdtConfig, configured_outdir: str | None) -> Path:
    if configured_outdir is None:
        return config.outdir / "scored_root"
    candidate = Path(configured_outdir)
    if candidate.is_absolute():
        return candidate
    return (config.repo_root / candidate).resolve()



def _load_training_summary(config: EventBdtConfig) -> dict[str, Any]:
    if not config.summary_path.exists():
        raise FileNotFoundError(
            "Training summary was not found. Run event-BDT training first: "
            f"{config.summary_path}"
        )
    with config.summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)



def load_trained_ensemble(config: EventBdtConfig) -> TrainedEnsemble:
    xgb = _require_prediction_dependencies()
    summary = _load_training_summary(config)
    trained_feature_names = list(summary.get("features", []))
    if not trained_feature_names:
        raise ValueError(f"No trained feature list was found in {config.summary_path}.")
    if trained_feature_names != list(config.features):
        raise ValueError(
            "The training config features do not match the trained model summary. "
            "Retrain the model or use the same config that produced the trained outputs."
        )

    training_mode = str(summary.get("training_mode", config.training_mode))
    class_names = list(summary.get("class_names", config.class_names))
    if training_mode == MULTICLASS_MODE and class_names != list(config.class_names):
        raise ValueError(
            "The configured training classes do not match the trained model summary. "
            "Use the matching config or retrain the models."
        )

    fold_summaries = list(summary.get("fold_summaries", []))
    if not fold_summaries:
        raise ValueError(f"No fold summaries were found in {config.summary_path}.")

    fold_models: list[FoldModel] = []
    for fold_summary in fold_summaries:
        fold = int(fold_summary["fold"])
        best_iteration = int(fold_summary["best_iteration"])
        model_path = config.model_dir_path / f"{config.model_name}.fold{fold}.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained fold model was not found: {model_path}")
        booster = xgb.Booster()
        booster.load_model(model_path)
        fold_models.append(
            FoldModel(
                fold=fold,
                booster=booster,
                best_iteration=best_iteration,
            )
        )

    if len(fold_models) != config.k_folds:
        raise ValueError(
            "The number of trained fold models does not match config.k_folds. "
            f"summary has {len(fold_models)} folds, config requests {config.k_folds}."
        )

    return TrainedEnsemble(
        training_mode=training_mode,
        feature_names=trained_feature_names,
        class_names=class_names,
        fold_models=sorted(fold_models, key=lambda item: item.fold),
        k_folds=len(fold_models),
    )



def _build_score_matrix(
    columns: dict[str, np.ndarray],
    feature_names: list[str],
) -> np.ndarray:
    return build_feature_matrix(columns, feature_names)



def _coerce_prediction_output(
    raw_scores: np.ndarray,
    n_rows: int,
    ensemble: TrainedEnsemble,
) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=np.float32)
    if ensemble.training_mode == MULTICLASS_MODE:
        n_classes = len(ensemble.class_names)
        if scores.ndim == 1:
            scores = scores.reshape(n_rows, n_classes)
        if scores.shape != (n_rows, n_classes):
            raise ValueError(
                "Unexpected multiclass prediction shape from XGBoost: "
                f"got {scores.shape}, expected {(n_rows, n_classes)}."
            )
        return scores
    return scores.reshape(n_rows)



def _predict_mean_scores(
    ensemble: TrainedEnsemble,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    xgb = _require_prediction_dependencies()
    if feature_matrix.shape[0] == 0:
        if ensemble.training_mode == MULTICLASS_MODE:
            return np.zeros((0, len(ensemble.class_names)), dtype=np.float32)
        return np.zeros(0, dtype=np.float32)

    matrix = xgb.DMatrix(
        feature_matrix,
        feature_names=ensemble.feature_names,
        missing=np.nan,
    )
    if ensemble.training_mode == MULTICLASS_MODE:
        prediction_sum = np.zeros(
            (feature_matrix.shape[0], len(ensemble.class_names)),
            dtype=np.float64,
        )
    else:
        prediction_sum = np.zeros(feature_matrix.shape[0], dtype=np.float64)

    for fold_model in ensemble.fold_models:
        prediction_sum += _coerce_prediction_output(
            fold_model.booster.predict(
                matrix,
                iteration_range=(0, fold_model.best_iteration + 1),
            ),
            n_rows=feature_matrix.shape[0],
            ensemble=ensemble,
        )

    return np.asarray(prediction_sum / float(len(ensemble.fold_models)), dtype=np.float32)



def _predict_fold_routed_scores(
    ensemble: TrainedEnsemble,
    feature_matrix: np.ndarray,
    fold_id: np.ndarray,
) -> np.ndarray:
    xgb = _require_prediction_dependencies()
    if feature_matrix.shape[0] == 0:
        if ensemble.training_mode == MULTICLASS_MODE:
            return np.zeros((0, len(ensemble.class_names)), dtype=np.float32)
        return np.zeros(0, dtype=np.float32)

    if ensemble.training_mode == MULTICLASS_MODE:
        scores = np.full(
            (feature_matrix.shape[0], len(ensemble.class_names)),
            np.nan,
            dtype=np.float32,
        )
    else:
        scores = np.full(feature_matrix.shape[0], np.nan, dtype=np.float32)

    for fold_model in ensemble.fold_models:
        mask = fold_id == fold_model.fold
        if not np.any(mask):
            continue
        matrix = xgb.DMatrix(
            feature_matrix[mask],
            feature_names=ensemble.feature_names,
            missing=np.nan,
        )
        scores[mask] = _coerce_prediction_output(
            fold_model.booster.predict(
                matrix,
                iteration_range=(0, fold_model.best_iteration + 1),
            ),
            n_rows=int(np.sum(mask)),
            ensemble=ensemble,
        )
    return scores



def _load_tree_chunk(
    tree: uproot.TTree,
    branch_names: list[str],
    entry_start: int,
    entry_stop: int,
) -> dict[str, ak.Array]:
    return dict(
        tree.arrays(
            branch_names,
            entry_start=entry_start,
            entry_stop=entry_stop,
            library="ak",
            how=dict,
        )
    )



def _prediction_branch_names(
    score_branch: str,
    prediction_mode: str,
    training_mode: str,
    class_names: list[str],
) -> list[str]:
    if training_mode == BINARY_MODE:
        if prediction_mode == PREDICTION_MODE_MEAN:
            return [score_branch]
        if prediction_mode == PREDICTION_MODE_FOLD_ROUTED:
            return [score_branch]
        if prediction_mode == PREDICTION_MODE_BOTH:
            return [f"{score_branch}_mean", f"{score_branch}_fold_routed"]
        raise ValueError(f"Unknown prediction mode: {prediction_mode}")

    if prediction_mode == PREDICTION_MODE_MEAN:
        return [f"{score_branch}_{class_name}" for class_name in class_names]
    if prediction_mode == PREDICTION_MODE_FOLD_ROUTED:
        return [f"{score_branch}_{class_name}" for class_name in class_names]
    if prediction_mode == PREDICTION_MODE_BOTH:
        branches = [f"{score_branch}_mean_{class_name}" for class_name in class_names]
        branches.extend(f"{score_branch}_fold_routed_{class_name}" for class_name in class_names)
        return branches
    raise ValueError(f"Unknown prediction mode: {prediction_mode}")



def _score_chunk(
    chunk_arrays: dict[str, ak.Array],
    *,
    config: EventBdtConfig,
    ensemble: TrainedEnsemble,
    prediction_mode: str,
    score_selection_only: bool,
    score_branch: str,
) -> dict[str, np.ndarray]:
    flatten_first_branches = set(config.flatten_first_branches)
    required_columns = set(config.features)
    required_columns.update(config.selection_branches)
    required_columns.update({"run", "luminosityBlock", "event"})
    required_branches = {resolve_input_branch_name(name) for name in required_columns}

    columns: dict[str, np.ndarray] = {}
    for branch in sorted(required_branches):
        if branch not in chunk_arrays:
            raise KeyError(f"Required branch '{branch}' is missing from prediction input.")
        columns[branch] = _extract_branch(
            branch,
            chunk_arrays[branch],
            flatten_first_branches,
        )
    for column_name in sorted(required_columns):
        if column_name in columns:
            continue
        source_branch = resolve_input_branch_name(column_name)
        if source_branch not in chunk_arrays:
            raise KeyError(
                f"Required source branch '{source_branch}' for column '{column_name}' is missing from prediction input."
            )
        columns[column_name] = _extract_branch(
            column_name,
            chunk_arrays[source_branch],
            flatten_first_branches,
        )

    first_field = next(iter(chunk_arrays), None)
    n_events = len(chunk_arrays[first_field]) if first_field is not None else 0
    base_selection_mask = _selection_mask(config.base_selection, columns)
    score_mask = base_selection_mask if score_selection_only else np.ones(n_events, dtype=bool)

    outputs: dict[str, np.ndarray] = {}
    feature_matrix = _build_score_matrix(columns, ensemble.feature_names)

    if prediction_mode in {PREDICTION_MODE_MEAN, PREDICTION_MODE_BOTH}:
        mean_scores = _predict_mean_scores(ensemble, feature_matrix[score_mask]) if np.any(score_mask) else None
        if ensemble.training_mode == MULTICLASS_MODE:
            for class_index, class_name in enumerate(ensemble.class_names):
                branch_name = f"{score_branch}_{class_name}"
                if prediction_mode == PREDICTION_MODE_BOTH:
                    branch_name = f"{score_branch}_mean_{class_name}"
                scores = np.full(n_events, np.nan, dtype=np.float32)
                if mean_scores is not None:
                    scores[score_mask] = mean_scores[:, class_index]
                outputs[branch_name] = scores
        else:
            branch_name = score_branch if prediction_mode == PREDICTION_MODE_MEAN else f"{score_branch}_mean"
            scores = np.full(n_events, np.nan, dtype=np.float32)
            if mean_scores is not None:
                scores[score_mask] = mean_scores
            outputs[branch_name] = scores

    if prediction_mode in {PREDICTION_MODE_FOLD_ROUTED, PREDICTION_MODE_BOTH}:
        routed_scores = None
        if np.any(score_mask):
            fold_id = _compute_fold_ids(
                np.asarray(columns["run"][score_mask], dtype=np.int64),
                np.asarray(columns["luminosityBlock"][score_mask], dtype=np.int64),
                np.asarray(columns["event"][score_mask], dtype=np.int64),
                config.k_folds,
            )
            routed_scores = _predict_fold_routed_scores(
                ensemble,
                feature_matrix[score_mask],
                fold_id,
            )

        if ensemble.training_mode == MULTICLASS_MODE:
            for class_index, class_name in enumerate(ensemble.class_names):
                branch_name = f"{score_branch}_{class_name}"
                if prediction_mode == PREDICTION_MODE_BOTH:
                    branch_name = f"{score_branch}_fold_routed_{class_name}"
                scores = np.full(n_events, np.nan, dtype=np.float32)
                if routed_scores is not None:
                    scores[score_mask] = routed_scores[:, class_index]
                outputs[branch_name] = scores
        else:
            branch_name = score_branch if prediction_mode == PREDICTION_MODE_FOLD_ROUTED else f"{score_branch}_fold_routed"
            scores = np.full(n_events, np.nan, dtype=np.float32)
            if routed_scores is not None:
                scores[score_mask] = routed_scores
            outputs[branch_name] = scores

    return outputs



def predict_event_bdt_to_root(
    config: EventBdtConfig,
    *,
    samples_config_path: str | Path | None = None,
    outdir: str | None = None,
    score_branch: str = "bdt_score",
    prediction_mode: str = PREDICTION_MODE_MEAN,
    score_selection_only: bool = True,
    force: bool = False,
    max_files_per_sample: int | None = None,
) -> dict[str, Any]:
    if prediction_mode not in PREDICTION_MODE_CHOICES:
        raise ValueError(
            f"Unknown prediction mode: {prediction_mode}. "
            f"Available modes: {', '.join(PREDICTION_MODE_CHOICES)}"
        )

    ensemble = load_trained_ensemble(config)
    sample_config_to_use = samples_config_path or config.samples_config_path
    sample_file_limit = max_files_per_sample
    if sample_file_limit is None:
        sample_file_limit = config.max_files_per_sample
    samples_config = load_event_bdt_samples_config(
        sample_config_to_use,
        max_files_per_sample=sample_file_limit,
    )
    output_root = _resolve_prediction_outdir(config, outdir)
    output_root.mkdir(parents=True, exist_ok=True)

    written_files = 0
    skipped_existing = 0
    skipped_missing_tree = 0
    sample_summaries: list[dict[str, Any]] = []

    for sample in samples_config.samples:
        sample_output_dir = output_root / sample.name
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        sample_written = 0
        sample_skipped_existing = 0
        sample_skipped_missing_tree = 0

        for file_path in sample.files:
            input_path = Path(file_path)
            output_path = sample_output_dir / input_path.name
            if output_path.exists() and not force:
                sample_skipped_existing += 1
                skipped_existing += 1
                continue

            with uproot.open(input_path) as input_file:
                if config.tree_name not in input_file:
                    sample_skipped_missing_tree += 1
                    skipped_missing_tree += 1
                    continue

                tree = input_file[config.tree_name]
                input_branch_names = list(tree.keys())
                output_score_branches = _prediction_branch_names(
                    score_branch,
                    prediction_mode,
                    ensemble.training_mode,
                    ensemble.class_names,
                )
                conflicting_branches = [
                    branch for branch in output_score_branches if branch in input_branch_names
                ]
                if conflicting_branches:
                    raise ValueError(
                        "Prediction output branch already exists in the input Events tree: "
                        f"{', '.join(conflicting_branches)}. "
                        "Choose a different --score-branch name."
                    )

                with uproot.recreate(output_path) as output_file:
                    tree_written = False
                    for entry_start in range(0, tree.num_entries, PREDICTION_ENTRY_CHUNK_SIZE):
                        entry_stop = min(entry_start + PREDICTION_ENTRY_CHUNK_SIZE, tree.num_entries)
                        chunk_arrays = _load_tree_chunk(
                            tree,
                            input_branch_names,
                            entry_start,
                            entry_stop,
                        )
                        output_arrays = dict(chunk_arrays)
                        output_arrays.update(
                            _score_chunk(
                                chunk_arrays,
                                config=config,
                                ensemble=ensemble,
                                prediction_mode=prediction_mode,
                                score_selection_only=score_selection_only,
                                score_branch=score_branch,
                            )
                        )
                        if not tree_written:
                            output_file[config.tree_name] = output_arrays
                            tree_written = True
                        else:
                            output_file[config.tree_name].extend(output_arrays)

            sample_written += 1
            written_files += 1

        sample_summaries.append(
            {
                "sample": sample.name,
                "process": sample.process,
                "n_input_files": len(sample.files),
                "written_files": sample_written,
                "skipped_existing": sample_skipped_existing,
                "skipped_missing_tree": sample_skipped_missing_tree,
            }
        )

    summary = {
        "config_path": str(config.config_path),
        "samples_config_path": str(Path(sample_config_to_use).resolve()),
        "prediction_outdir": str(output_root),
        "training_mode": ensemble.training_mode,
        "class_names": list(ensemble.class_names),
        "prediction_mode": prediction_mode,
        "score_branch": score_branch,
        "score_selection_only": score_selection_only,
        "output_branches": _prediction_branch_names(
            score_branch,
            prediction_mode,
            ensemble.training_mode,
            ensemble.class_names,
        ),
        "output_content": "full_events_tree_with_scores",
        "written_files": written_files,
        "skipped_existing": skipped_existing,
        "skipped_missing_tree": skipped_missing_tree,
        "sample_summaries": sample_summaries,
    }
    _save_json(output_root / "prediction_summary.json", summary)
    return summary
