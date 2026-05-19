from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tthcc_an.event_bdt.config import EventBdtConfig
from tthcc_an.event_bdt.dataset import load_npz_payload, prepare_event_bdt_inputs
from tthcc_an.event_bdt.reweighting import build_reweighted_training_weights


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


def train_event_bdt(
    config: EventBdtConfig,
    *,
    force_prepare: bool = False,
    force_retrain: bool = False,
) -> dict[str, Any]:
    xgb, roc_curve, auc = _require_training_dependencies()
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
    X = np.column_stack([np.asarray(arrays[name], dtype=np.float64) for name in feature_names])
    labels = np.asarray(arrays["train_label"], dtype=np.int8)
    base_weights = np.asarray(arrays["train_weight"], dtype=np.float64)
    fold_id = np.asarray(arrays["fold_id"], dtype=np.int16)
    processes = np.asarray(arrays["process"])
    roles = np.asarray(arrays["role"])

    trainable_mask = labels >= 0
    if not np.any(trainable_mask & (labels == 1)):
        raise ValueError("No signal events were found after selection.")
    if not np.any(trainable_mask & (labels == 0)):
        raise ValueError("No background events were found after selection.")

    print(
        f"Training event BDT with {X.shape[0]} selected events "
        f"({int(np.sum(trainable_mask & (labels == 1)))} signal, "
        f"{int(np.sum(trainable_mask & (labels == 0)))} background, "
        f"{int(np.sum(labels < 0))} eval-only) and {len(feature_names)} features.",
        flush=True,
    )

    config.model_dir_path.mkdir(parents=True, exist_ok=True)
    config.plot_dir_path.mkdir(parents=True, exist_ok=True)

    full_prediction_sum = np.zeros(X.shape[0], dtype=np.float64)
    oof_scores = np.full(X.shape[0], np.nan, dtype=np.float64)
    models_used = 0
    fold_summaries: list[dict[str, Any]] = []
    importance_totals: dict[str, float] = {name: 0.0 for name in feature_names}

    full_matrix = xgb.DMatrix(X, feature_names=feature_names, missing=np.nan)
    column_view = {name: np.asarray(arrays[name], dtype=np.float64) for name in feature_names}

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
            labels=labels,
            train_mask=fold_train_mask,
            columns=column_view,
            reweighting_config=config.reweighting,
        )

        dtrain = xgb.DMatrix(
            X[fold_train_mask],
            label=labels[fold_train_mask],
            weight=train_weights[fold_train_mask],
            feature_names=feature_names,
            missing=np.nan,
        )
        dval = xgb.DMatrix(
            X[fold_val_mask],
            label=labels[fold_val_mask],
            weight=base_weights[fold_val_mask],
            feature_names=feature_names,
            missing=np.nan,
        )

        evals_result: dict[str, dict[str, list[float]]] = {}
        booster = xgb.train(
            params=dict(config.xgboost_params),
            dtrain=dtrain,
            num_boost_round=config.num_boost_round,
            evals=[(dtrain, "train"), (dval, "eval")],
            early_stopping_rounds=config.early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=False,
        )

        best_iteration = int(getattr(booster, "best_iteration", config.num_boost_round - 1))
        model_path = config.model_dir_path / f"{config.model_name}.fold{fold}.json"
        booster.save_model(model_path)

        val_scores = booster.predict(dval, iteration_range=(0, best_iteration + 1))
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
                "reweighting": reweighting_summary,
                "evals_result": evals_result,
            }
        )

    if models_used == 0:
        raise ValueError("No folds were trained. Check the selected events and k-fold configuration.")

    mean_model_score = full_prediction_sum / float(models_used)
    final_score = mean_model_score.copy()
    final_score[trainable_mask] = oof_scores[trainable_mask]

    fpr, tpr, _ = roc_curve(
        labels[trainable_mask],
        final_score[trainable_mask],
        sample_weight=base_weights[trainable_mask],
    )
    weighted_auc = float(auc(fpr, tpr))

    predictions_arrays = {
        "event": np.asarray(arrays["event"]),
        "run": np.asarray(arrays["run"]),
        "luminosityBlock": np.asarray(arrays["luminosityBlock"]),
        "process": processes,
        "label_text": np.asarray(arrays["label_text"]),
        "role": roles,
        "dataset_name": np.asarray(arrays["dataset_name"]),
        "sample_name": np.asarray(arrays["sample_name"]),
        "train_label": labels,
        "train_weight": base_weights,
        "weight_signed": np.asarray(arrays["weight_signed"], dtype=np.float64),
        "fold_id": fold_id,
        "bdt_score": final_score.astype(np.float64),
        "bdt_score_model_mean": mean_model_score.astype(np.float64),
    }
    predictions_metadata = {
        "config_path": str(config.config_path),
        "prepared_inputs_path": str(prepared_path),
        "features": feature_names,
        "spectators": list(prepared_metadata["spectators"]),
        "signal_processes": list(config.signal_processes),
        "background_processes": list(config.background_processes),
        "eval_processes_extra": list(config.eval_processes_extra),
        "sample_summaries": prepared_metadata["sample_summaries"],
        "weighted_auc": weighted_auc,
        "fold_summaries": fold_summaries,
    }
    _save_predictions_npz(config.predictions_path, predictions_arrays, predictions_metadata)

    process_summaries: dict[str, dict[str, Any]] = {}
    unique_processes = sorted(set(processes.tolist()))
    for process in unique_processes:
        mask = processes == process
        process_summaries[process] = {
            "n_events": int(np.sum(mask)),
            "role": str(roles[mask][0]) if np.any(mask) else "",
            "train_weight_sum": float(np.sum(base_weights[mask])),
            "score_mean": float(np.average(final_score[mask], weights=base_weights[mask]))
            if np.any(mask) and np.sum(base_weights[mask]) > 0
            else float("nan"),
        }

    feature_importance = {
        name: float(importance_totals.get(name, 0.0) / max(models_used, 1))
        for name in feature_names
    }
    summary_payload = {
        "config_path": str(config.config_path),
        "prepared_inputs_path": str(prepared_path),
        "predictions_path": str(config.predictions_path),
        "model_dir": str(config.model_dir_path),
        "weighted_auc": weighted_auc,
        "n_total_events": int(X.shape[0]),
        "n_trainable_events": int(np.sum(trainable_mask)),
        "n_signal_events": int(np.sum(trainable_mask & (labels == 1))),
        "n_background_events": int(np.sum(trainable_mask & (labels == 0))),
        "n_eval_only_events": int(np.sum(labels < 0)),
        "features": feature_names,
        "xgboost_params": dict(config.xgboost_params),
        "num_boost_round": config.num_boost_round,
        "early_stopping_rounds": config.early_stopping_rounds,
        "fold_summaries": fold_summaries,
        "process_summaries": process_summaries,
        "feature_importance": feature_importance,
    }
    _save_json(config.summary_path, summary_payload)
    _save_json(config.feature_importance_path, feature_importance)
    print(
        f"Training finished. Weighted AUC = {weighted_auc:.4f}. "
        f"Outputs written to {config.outdir}",
        flush=True,
    )
    return summary_payload
