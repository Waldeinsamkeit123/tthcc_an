from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tthcc_an.event_bdt.config import EventBdtConfig
from tthcc_an.event_bdt.dataset import load_npz_payload, prepare_event_bdt_inputs
from tthcc_an.event_bdt.features import build_feature_matrix, clean_feature_values
from tthcc_an.event_bdt.reweighting import build_reweighted_training_weights


BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"



def _require_training_dependencies():
    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xgboost is required for the event-BDT prototype. "
            "Please run inside an environment such as LCG108 where it is available."
        ) from exc

    try:
        from sklearn.metrics import auc, roc_curve
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for ROC evaluation in the event-BDT prototype."
        ) from exc
    return xgb, roc_curve, auc



def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")



def _save_predictions_npz(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    out = dict(arrays)
    out["metadata_json"] = np.array(json.dumps(metadata, indent=2, sort_keys=True), dtype=np.str_)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)



def load_predictions_payload(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    return load_npz_payload(path)



def _validate_xgboost_configuration(config: EventBdtConfig) -> None:
    objective = str(config.xgboost_params.get("objective", "")).strip()
    if config.training_mode == MULTICLASS_MODE:
        if objective != "multi:softprob":
            raise ValueError(
                "Multiclass training requires xgboost.objective = 'multi:softprob'."
            )
        configured_num_class = config.xgboost_params.get("num_class")
        if configured_num_class is None:
            raise ValueError("Multiclass training requires xgboost.num_class to be configured.")
        if int(configured_num_class) != config.num_training_classes:
            raise ValueError(
                "Configured xgboost.num_class does not match the number of training classes: "
                f"{configured_num_class} vs {config.num_training_classes}."
            )
        return

    if objective and objective != "binary:logistic":
        raise ValueError(
            "Binary training requires xgboost.objective = 'binary:logistic'."
        )



def _coerce_multiclass_scores(
    raw_scores: np.ndarray,
    n_rows: int,
    n_classes: int,
) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(n_rows, n_classes)
    if scores.shape != (n_rows, n_classes):
        raise ValueError(
            "Unexpected multiclass score shape from XGBoost: "
            f"got {scores.shape}, expected {(n_rows, n_classes)}."
        )
    return scores



def _branch_name_for_class(class_name: str) -> str:
    return f"bdt_score_{class_name}"



def _model_mean_branch_name_for_class(class_name: str) -> str:
    return f"bdt_score_model_mean_{class_name}"



def _class_event_counts(labels: np.ndarray, class_names: list[str]) -> dict[str, int]:
    return {
        class_name: int(np.sum(labels == class_index))
        for class_index, class_name in enumerate(class_names)
    }



def _weighted_average_or_nan(values: np.ndarray, weights: np.ndarray) -> float:
    weight_sum = float(np.sum(weights))
    if values.size == 0 or weight_sum <= 0.0:
        return float("nan")
    return float(np.average(values, weights=weights))



def _class_weight_sums(
    labels: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    return {
        class_name: float(np.sum(weights[mask & (labels == class_index)]))
        for class_index, class_name in enumerate(class_names)
    }


def _configured_early_stopping_metric(xgboost_params: dict[str, Any]) -> str | None:
    eval_metric = xgboost_params.get("eval_metric")
    if isinstance(eval_metric, (list, tuple)):
        return str(eval_metric[-1]) if eval_metric else None
    if eval_metric is None:
        return None
    return str(eval_metric)


def _build_early_stopping_callback(xgb: Any, config: EventBdtConfig):
    return xgb.callback.EarlyStopping(
        rounds=config.early_stopping_rounds,
        metric_name=_configured_early_stopping_metric(config.xgboost_params),
        data_name="eval_balanced",
        save_best=False,
        min_delta=config.early_stopping_min_delta,
    )


def _binary_labels_from_groups(labels: np.ndarray, class_groups: list[str]) -> np.ndarray:
    signal_indices = np.array(
        [index for index, group in enumerate(class_groups) if group == "signal"],
        dtype=np.int16,
    )
    background_indices = np.array(
        [index for index, group in enumerate(class_groups) if group == "background"],
        dtype=np.int16,
    )
    binary_labels = np.full(labels.shape, -1, dtype=np.int8)
    binary_labels[np.isin(labels, background_indices)] = 0
    binary_labels[np.isin(labels, signal_indices)] = 1
    return binary_labels



def train_event_bdt(
    config: EventBdtConfig,
    *,
    force_prepare: bool = False,
    force_retrain: bool = False,
) -> dict[str, Any]:
    xgb, roc_curve, auc = _require_training_dependencies()
    _validate_xgboost_configuration(config)
    prepared_path = prepare_event_bdt_inputs(config, force=force_prepare)

    if config.predictions_path.exists() and config.summary_path.exists() and not force_retrain:
        print(
            f"Reusing existing event-BDT training outputs from {config.outdir}",
            flush=True,
        )
        with config.summary_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    arrays, prepared_metadata = load_npz_payload(prepared_path)
    feature_names = list(prepared_metadata["features"])
    class_names = list(prepared_metadata.get("class_names", config.class_names))
    class_labels = list(prepared_metadata.get("class_labels", config.class_labels))
    class_groups = list(prepared_metadata.get("class_groups", config.class_groups))
    training_mode = str(prepared_metadata.get("training_mode", config.training_mode))

    if class_names != list(config.class_names):
        raise ValueError(
            "Prepared inputs do not match the configured training classes. "
            "Rebuild the prepared cache with --force-prepare or use the matching config."
        )
    if training_mode != config.training_mode:
        raise ValueError(
            "Prepared inputs training mode does not match the current config. "
            "Rebuild the prepared cache with --force-prepare."
        )

    X = build_feature_matrix(arrays, feature_names)
    raw_labels = np.asarray(arrays["train_label"], dtype=np.int16)
    base_weights = np.asarray(arrays["train_weight"], dtype=np.float64)
    fold_id = np.asarray(arrays["fold_id"], dtype=np.int16)
    processes = np.asarray(arrays["process"])
    roles = np.asarray(arrays["role"])

    if training_mode == MULTICLASS_MODE:
        train_labels = raw_labels
    else:
        train_labels = _binary_labels_from_groups(raw_labels, class_groups)

    trainable_mask = train_labels >= 0
    if not np.any(trainable_mask):
        raise ValueError("No trainable events were found after selection.")

    if training_mode == MULTICLASS_MODE:
        for class_index, class_name in enumerate(class_names):
            if not np.any(trainable_mask & (train_labels == class_index)):
                raise ValueError(
                    f"No events were found for training class '{class_name}' after selection."
                )
        class_counts = _class_event_counts(train_labels[trainable_mask], class_names)
    else:
        if not np.any(trainable_mask & (train_labels == 1)):
            raise ValueError("No signal events were found after selection.")
        if not np.any(trainable_mask & (train_labels == 0)):
            raise ValueError("No background events were found after selection.")
        class_counts = {
            "signal": int(np.sum(trainable_mask & (train_labels == 1))),
            "background": int(np.sum(trainable_mask & (train_labels == 0))),
        }

    print(
        f"Training event BDT with {X.shape[0]} selected events "
        f"({int(np.sum(trainable_mask))} trainable, {int(np.sum(~trainable_mask))} eval-only) "
        f"and {len(feature_names)} features.",
        flush=True,
    )
    print(
        "Training classes: "
        + ", ".join(f"{name}={count}" for name, count in class_counts.items()),
        flush=True,
    )

    config.model_dir_path.mkdir(parents=True, exist_ok=True)
    config.plot_dir_path.mkdir(parents=True, exist_ok=True)

    if training_mode == MULTICLASS_MODE:
        n_classes = len(class_names)
        full_prediction_sum = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        oof_scores = np.full((X.shape[0], n_classes), np.nan, dtype=np.float64)
    else:
        full_prediction_sum = np.zeros(X.shape[0], dtype=np.float64)
        oof_scores = np.full(X.shape[0], np.nan, dtype=np.float64)

    models_used = 0
    fold_summaries: list[dict[str, Any]] = []
    importance_totals: dict[str, float] = {name: 0.0 for name in feature_names}

    full_matrix = xgb.DMatrix(X, feature_names=feature_names, missing=np.nan)
    column_view = {
        name: clean_feature_values(name, np.asarray(arrays[name], dtype=np.float64))
        for name in feature_names
    }

    for fold in range(config.k_folds):
        fold_train_mask = trainable_mask & (fold_id != fold)
        fold_val_mask = trainable_mask & (fold_id == fold)
        if not np.any(fold_val_mask):
            continue
        if not np.any(fold_train_mask):
            raise ValueError(f"Fold {fold} has no training events.")

        print(
            f"Training fold {fold + 1}/{config.k_folds}: "
            f"{int(np.sum(fold_train_mask))} train, {int(np.sum(fold_val_mask))} validation events.",
            flush=True,
        )

        train_weights, reweighting_summary = build_reweighted_training_weights(
            base_weights=base_weights,
            labels=raw_labels,
            train_mask=fold_train_mask,
            columns=column_view,
            reweighting_config=config.reweighting,
            class_groups=class_groups,
        )
        validation_balanced_weights, validation_reweighting_summary = build_reweighted_training_weights(
            base_weights=base_weights,
            labels=raw_labels,
            train_mask=fold_val_mask,
            columns=column_view,
            reweighting_config=config.reweighting,
            class_groups=class_groups,
        )

        dtrain = xgb.DMatrix(
            X[fold_train_mask],
            label=train_labels[fold_train_mask],
            weight=train_weights[fold_train_mask],
            feature_names=feature_names,
            missing=np.nan,
        )
        dval_physics = xgb.DMatrix(
            X[fold_val_mask],
            label=train_labels[fold_val_mask],
            weight=base_weights[fold_val_mask],
            feature_names=feature_names,
            missing=np.nan,
        )
        dval_balanced = xgb.DMatrix(
            X[fold_val_mask],
            label=train_labels[fold_val_mask],
            weight=validation_balanced_weights[fold_val_mask],
            feature_names=feature_names,
            missing=np.nan,
        )

        evals_result: dict[str, dict[str, list[float]]] = {}
        booster = xgb.train(
            params=dict(config.xgboost_params),
            dtrain=dtrain,
            num_boost_round=config.num_boost_round,
            evals=[
                (dtrain, "train"),
                (dval_physics, "eval_physics"),
                (dval_balanced, "eval_balanced"),
            ],
            callbacks=[_build_early_stopping_callback(xgb, config)],
            evals_result=evals_result,
            verbose_eval=False,
        )

        best_iteration = int(getattr(booster, "best_iteration", config.num_boost_round - 1))
        best_score = getattr(booster, "best_score", None)
        model_path = config.model_dir_path / f"{config.model_name}.fold{fold}.json"
        booster.save_model(model_path)

        if training_mode == MULTICLASS_MODE:
            n_val = int(np.sum(fold_val_mask))
            val_scores = _coerce_multiclass_scores(
                booster.predict(dval_physics, iteration_range=(0, best_iteration + 1)),
                n_rows=n_val,
                n_classes=len(class_names),
            )
            oof_scores[fold_val_mask] = val_scores
            full_prediction_sum += _coerce_multiclass_scores(
                booster.predict(full_matrix, iteration_range=(0, best_iteration + 1)),
                n_rows=X.shape[0],
                n_classes=len(class_names),
            )
        else:
            val_scores = booster.predict(dval_physics, iteration_range=(0, best_iteration + 1))
            oof_scores[fold_val_mask] = np.asarray(val_scores, dtype=np.float64)
            full_prediction_sum += booster.predict(
                full_matrix,
                iteration_range=(0, best_iteration + 1),
            )
        models_used += 1

        print(
            f"  Fold {fold + 1} done. Best iteration = {best_iteration}.",
            flush=True,
        )

        for name, value in booster.get_score(importance_type="gain").items():
            importance_totals[name] = importance_totals.get(name, 0.0) + float(value)

        fold_summaries.append(
            {
                "fold": fold,
                "n_train": int(np.sum(fold_train_mask)),
                "n_val": int(np.sum(fold_val_mask)),
                "best_iteration": best_iteration,
                "best_score": float(best_score) if best_score is not None else None,
                "early_stopping_dataset": "eval_balanced",
                "early_stopping_metric": _configured_early_stopping_metric(config.xgboost_params),
                "early_stopping_min_delta": config.early_stopping_min_delta,
                "base_weight_sum_by_class": _class_weight_sums(
                    train_labels, base_weights, fold_train_mask, class_names
                ),
                "reweighted_weight_sum_by_class": _class_weight_sums(
                    train_labels, train_weights, fold_train_mask, class_names
                ),
                "validation_base_weight_sum_by_class": _class_weight_sums(
                    train_labels, base_weights, fold_val_mask, class_names
                ),
                "validation_reweighted_weight_sum_by_class": _class_weight_sums(
                    train_labels, validation_balanced_weights, fold_val_mask, class_names
                ),
                "reweighting": reweighting_summary,
                "validation_reweighting": validation_reweighting_summary,
                "evals_result": evals_result,
            }
        )

    if models_used == 0:
        raise ValueError("No folds were trained. Check the selected events and k-fold configuration.")

    mean_model_score = full_prediction_sum / float(models_used)
    final_score = mean_model_score.copy()
    final_score[trainable_mask] = oof_scores[trainable_mask]

    process_summaries: dict[str, dict[str, Any]] = {}
    unique_processes = sorted(set(processes.tolist()))
    for process in unique_processes:
        mask = processes == process
        process_summary: dict[str, Any] = {
            "n_events": int(np.sum(mask)),
            "role": str(roles[mask][0]) if np.any(mask) else "",
            "train_weight_sum": float(np.sum(base_weights[mask])),
        }
        if training_mode == MULTICLASS_MODE:
            process_summary["score_mean_by_class"] = {
                class_name: _weighted_average_or_nan(final_score[mask, class_index], base_weights[mask])
                for class_index, class_name in enumerate(class_names)
            }
        else:
            process_summary["score_mean"] = _weighted_average_or_nan(final_score[mask], base_weights[mask])
        process_summaries[process] = process_summary

    feature_importance = {
        name: float(importance_totals.get(name, 0.0) / max(models_used, 1))
        for name in feature_names
    }

    stored_train_label = train_labels if training_mode == BINARY_MODE else raw_labels
    predictions_arrays = {
        "event": np.asarray(arrays["event"]),
        "run": np.asarray(arrays["run"]),
        "luminosityBlock": np.asarray(arrays["luminosityBlock"]),
        "process": processes,
        "label_text": np.asarray(arrays["label_text"]),
        "role": roles,
        "dataset_name": np.asarray(arrays["dataset_name"]),
        "sample_name": np.asarray(arrays["sample_name"]),
        "train_label": stored_train_label.astype(np.int16),
        "train_weight": base_weights,
        "weight_signed": np.asarray(arrays["weight_signed"], dtype=np.float64),
        "fold_id": fold_id,
    }

    summary_payload: dict[str, Any] = {
        "config_path": str(config.config_path),
        "prepared_inputs_path": str(prepared_path),
        "predictions_path": str(config.predictions_path),
        "model_dir": str(config.model_dir_path),
        "training_mode": training_mode,
        "class_names": class_names,
        "class_labels": class_labels,
        "class_groups": class_groups,
        "n_total_events": int(X.shape[0]),
        "n_trainable_events": int(np.sum(trainable_mask)),
        "n_eval_only_events": int(np.sum(~trainable_mask)),
        "n_events_by_class": class_counts,
        "features": feature_names,
        "xgboost_params": dict(config.xgboost_params),
        "num_boost_round": config.num_boost_round,
        "early_stopping_rounds": config.early_stopping_rounds,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "fold_summaries": fold_summaries,
        "process_summaries": process_summaries,
        "feature_importance": feature_importance,
    }
    predictions_metadata = {
        "config_path": str(config.config_path),
        "prepared_inputs_path": str(prepared_path),
        "features": feature_names,
        "spectators": list(prepared_metadata["spectators"]),
        "training_mode": training_mode,
        "class_names": class_names,
        "class_labels": class_labels,
        "class_groups": class_groups,
        "training_classes": list(prepared_metadata.get("training_classes", config.training_class_payload)),
        "signal_processes": list(config.signal_processes),
        "background_processes": list(config.background_processes),
        "eval_processes_extra": list(config.eval_processes_extra),
        "sample_summaries": prepared_metadata["sample_summaries"],
        "fold_summaries": fold_summaries,
    }

    if training_mode == MULTICLASS_MODE:
        weighted_auc_ovr: dict[str, float] = {}
        for class_index, class_name in enumerate(class_names):
            one_vs_rest = (train_labels[trainable_mask] == class_index).astype(np.int8)
            class_scores = final_score[trainable_mask, class_index]
            fpr, tpr, _ = roc_curve(
                one_vs_rest,
                class_scores,
                sample_weight=base_weights[trainable_mask],
            )
            weighted_auc_ovr[class_name] = float(auc(fpr, tpr))
            predictions_arrays[_branch_name_for_class(class_name)] = final_score[:, class_index].astype(np.float64)
            predictions_arrays[_model_mean_branch_name_for_class(class_name)] = mean_model_score[:, class_index].astype(np.float64)

        weighted_macro_auc = float(np.mean(list(weighted_auc_ovr.values())))
        predictions_metadata["weighted_auc_ovr"] = weighted_auc_ovr
        predictions_metadata["weighted_macro_auc"] = weighted_macro_auc
        summary_payload["weighted_auc_ovr"] = weighted_auc_ovr
        summary_payload["weighted_macro_auc"] = weighted_macro_auc
        metric_message = f"Weighted macro AUC = {weighted_macro_auc:.4f}"
    else:
        fpr, tpr, _ = roc_curve(
            train_labels[trainable_mask],
            final_score[trainable_mask],
            sample_weight=base_weights[trainable_mask],
        )
        weighted_auc = float(auc(fpr, tpr))
        predictions_arrays["bdt_score"] = final_score.astype(np.float64)
        predictions_arrays["bdt_score_model_mean"] = mean_model_score.astype(np.float64)
        predictions_metadata["weighted_auc"] = weighted_auc
        summary_payload["weighted_auc"] = weighted_auc
        summary_payload["n_signal_events"] = int(np.sum(trainable_mask & (train_labels == 1)))
        summary_payload["n_background_events"] = int(np.sum(trainable_mask & (train_labels == 0)))
        metric_message = f"Weighted AUC = {weighted_auc:.4f}"

    _save_predictions_npz(config.predictions_path, predictions_arrays, predictions_metadata)
    _save_json(config.summary_path, summary_payload)
    _save_json(config.feature_importance_path, feature_importance)
    print(
        f"Training finished. {metric_message}. Outputs written to {config.outdir}",
        flush=True,
    )
    return summary_payload
