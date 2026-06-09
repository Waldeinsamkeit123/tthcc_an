from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tthcc_an.event_bdt.config import EventBdtConfig, load_event_bdt_config
from tthcc_an.event_bdt.dataset import load_npz_payload, prepare_event_bdt_inputs
from tthcc_an.event_bdt.plotting import (
    plot_feature_mass_correlation_heatmap,
    plot_class_score_shapes,
    plot_ovr_roc_curves,
    plot_pairwise_roc_curve,
    plot_process_score_shapes,
    plot_process_score_weighted_events,
    plot_roc_curve,
    plot_score_vs_mass_grid,
    plot_training_class_score_shapes,
    plot_training_class_score_weighted_events,
)
from tthcc_an.event_bdt.prediction import (
    PREDICTION_MODE_CHOICES,
    predict_event_bdt_to_root,
)
from tthcc_an.event_bdt.training import load_predictions_payload, train_event_bdt
from tthcc_an.metrics import weighted_quantile


REPO_ROOT = Path(__file__).resolve().parents[3]
MASS_DIAGNOSTIC_VARIABLES = [
    "TargetFatJet_mass",
    "TargetFatJet_msoftdrop",
    "CleanedJet_mass",
]
MASS_DIAGNOSTIC_SCORE_NAMES = ["tth", "ttbar", "qcd"]
MASS_RANGE_QUANTILES = (0.005, 0.995)
QCD_SCORE_DROP_TARGETS = [0.70, 0.90, 0.95, 0.99, 0.995, 0.999]

PROCESS_DISPLAY_LABELS = {
    "ttHcc": "ttH(H->cc)",
    "ttHbb": "ttH(H->bb)",
    "ttH_nonbb": "ttH(non-bb)",
    "ttbar": "ttbar",
    "ttbb": "tt+bb",
    "ttll": "tt+ll",
    "ttv": "ttV",
    "single_top": "single top",
    "wjets": "W+jets",
    "zjets": "Z+jets",
    "qcd": "QCD",
    "signal": "Signal (ttHbb/cc)",
    "background": "Background",
}


def _load_config(config_path: str) -> EventBdtConfig:
    return load_event_bdt_config(config_path, REPO_ROOT)



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prototype event-level BDT workflow for ttHcc studies. "
            "The intended first use case is the 2024 0L selection."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Read ROOT files and build a prepared input cache.")
    prepare.add_argument("--config", required=True, help="Path to the event-BDT training config.")
    prepare.add_argument("--force", action="store_true", help="Rebuild the prepared cache even if it exists.")

    train = subparsers.add_parser("train", help="Train the event BDT with k-fold cross validation.")
    train.add_argument("--config", required=True, help="Path to the event-BDT training config.")
    train.add_argument(
        "--force-prepare",
        action="store_true",
        help="Rebuild the prepared cache before training.",
    )
    train.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain even if model outputs already exist.",
    )

    evaluate = subparsers.add_parser("evaluate", help="Make ROC and score-shape plots from saved predictions.")
    evaluate.add_argument("--config", required=True, help="Path to the event-BDT training config.")

    predict = subparsers.add_parser("predict", help="Apply trained event-BDT models and write scored ROOT files.")
    predict.add_argument("--config", required=True, help="Path to the event-BDT training config.")
    predict.add_argument(
        "--samples-config",
        help="Optional sample config to score. Defaults to the training config's samples_config.",
    )
    predict.add_argument(
        "--outdir",
        help="Optional output directory for scored ROOT files. Defaults to <training outdir>/scored_root.",
    )
    predict.add_argument(
        "--score-branch",
        default="bdt_score",
        help="Name of the output score branch to write.",
    )
    predict.add_argument(
        "--prediction-mode",
        choices=PREDICTION_MODE_CHOICES,
        default="mean",
        help="Prediction mode: mean model score, fold-routed score, or both.",
    )
    predict.add_argument(
        "--score-all-events",
        action="store_true",
        help="Score every event instead of only those passing base_selection.",
    )
    predict.add_argument(
        "--max-files-per-sample",
        type=int,
        help="Optional cap on the number of input ROOT files per sample.",
    )
    predict.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing scored ROOT files in the output directory.",
    )

    return parser



def _process_order(processes: np.ndarray) -> list[str]:
    unique_processes = list(dict.fromkeys(processes.tolist()))
    preferred_order = [
        "ttHcc",
        "ttHbb",
        "ttH_nonbb",
        "ttbar",
        "ttbb",
        "ttll",
        "ttv",
        "single_top",
        "wjets",
        "zjets",
        "qcd",
    ]
    process_order = [process for process in preferred_order if process in unique_processes]
    for process in unique_processes:
        if process not in process_order:
            process_order.append(process)
    return process_order



def _filled_string_array(value: str, size: int) -> np.ndarray:
    width = max(1, len(value))
    return np.full(size, value, dtype=f"<U{width}")



def _process_label_map(processes: np.ndarray, label_texts: np.ndarray) -> dict[str, str]:
    process_labels: dict[str, str] = {}
    for process in list(dict.fromkeys(processes.tolist())):
        mask = processes == process
        if not np.any(mask):
            continue
        fallback = str(label_texts[mask][0])
        process_labels[process] = PROCESS_DISPLAY_LABELS.get(process, fallback)
    return process_labels



def _safe_fraction(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)



def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value



def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )



def _format_fraction(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{100.0 * value:7.3f}%"



def _format_weight(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:12.4f}"



def _qcd_score_threshold_scan_paths(config: EventBdtConfig) -> tuple[Path, Path]:
    base = config.outdir / "qcd_score_threshold_scan"
    return base.with_suffix(".txt"), base.with_suffix(".json")



def _build_qcd_score_threshold_scan(
    *,
    class_names: list[str],
    class_labels: dict[str, str],
    score_by_class: dict[str, np.ndarray],
    labels: np.ndarray,
    weights: np.ndarray,
    processes: np.ndarray,
    process_order: list[str],
    process_labels: dict[str, str],
    trainable_mask: np.ndarray,
) -> dict[str, object] | None:
    if "qcd" not in class_names:
        return None

    qcd_class_index = class_names.index("qcd")
    qcd_scores = np.asarray(score_by_class["qcd"], dtype=np.float64)
    qcd_train_mask = trainable_mask & (labels == qcd_class_index)
    qcd_total_weight = float(np.sum(weights[qcd_train_mask]))
    if qcd_total_weight <= 0.0:
        return None

    qcd_train_scores = qcd_scores[qcd_train_mask]
    qcd_train_weights = weights[qcd_train_mask]

    payload_targets: list[dict[str, object]] = []
    for target_drop_fraction in QCD_SCORE_DROP_TARGETS:
        score_cut = float(
            weighted_quantile(
                qcd_train_scores,
                qcd_train_weights,
                max(0.0, min(1.0, 1.0 - target_drop_fraction)),
            )
        )
        keep_mask = qcd_scores <= score_cut
        drop_mask_gt = qcd_scores > score_cut
        drop_mask_ge = qcd_scores >= score_cut
        kept_trainable_mask = trainable_mask & keep_mask

        qcd_weight_drop_gt = float(np.sum(weights[qcd_train_mask & drop_mask_gt]))
        qcd_weight_drop_ge = float(np.sum(weights[qcd_train_mask & drop_mask_ge]))
        qcd_weight_keep = float(np.sum(weights[qcd_train_mask & keep_mask]))
        total_weight_keep_all = float(np.sum(weights[keep_mask]))
        total_weight_keep_trainable = float(np.sum(weights[kept_trainable_mask]))

        target_payload: dict[str, object] = {
            "target_qcd_drop_fraction": float(target_drop_fraction),
            "score_cut": score_cut,
            "qcd_drop_fraction_gt_cut": _safe_fraction(qcd_weight_drop_gt, qcd_total_weight),
            "qcd_drop_fraction_ge_cut": _safe_fraction(qcd_weight_drop_ge, qcd_total_weight),
            "qcd_keep_fraction_le_cut": _safe_fraction(qcd_weight_keep, qcd_total_weight),
            "qcd_weight_total": qcd_total_weight,
            "qcd_weight_keep_le_cut": qcd_weight_keep,
            "total_weight_keep_all_processes": total_weight_keep_all,
            "total_weight_keep_trainable_classes": total_weight_keep_trainable,
            "processes": {},
            "training_classes": {},
        }

        process_payload: dict[str, object] = {}
        for process in process_order:
            process_mask = processes == process
            if not np.any(process_mask):
                continue
            total_weight = float(np.sum(weights[process_mask]))
            kept_weight = float(np.sum(weights[process_mask & keep_mask]))
            process_payload[process] = {
                "label": process_labels.get(process, process),
                "n_total": int(np.sum(process_mask)),
                "n_keep_le_cut": int(np.sum(process_mask & keep_mask)),
                "weight_total": total_weight,
                "weight_keep_le_cut": kept_weight,
                "keep_fraction_le_cut": _safe_fraction(kept_weight, total_weight),
                "remaining_share_of_all_kept": _safe_fraction(kept_weight, total_weight_keep_all),
            }
        target_payload["processes"] = process_payload

        class_payload: dict[str, object] = {}
        for class_index, class_name in enumerate(class_names):
            class_mask = trainable_mask & (labels == class_index)
            if not np.any(class_mask):
                continue
            total_weight = float(np.sum(weights[class_mask]))
            kept_weight = float(np.sum(weights[class_mask & keep_mask]))
            class_payload[class_name] = {
                "label": class_labels.get(class_name, class_name),
                "n_total": int(np.sum(class_mask)),
                "n_keep_le_cut": int(np.sum(class_mask & keep_mask)),
                "weight_total": total_weight,
                "weight_keep_le_cut": kept_weight,
                "keep_fraction_le_cut": _safe_fraction(kept_weight, total_weight),
                "remaining_share_of_trainable_kept": _safe_fraction(kept_weight, total_weight_keep_trainable),
            }
        target_payload["training_classes"] = class_payload
        payload_targets.append(target_payload)

    return {
        "score_branch": "bdt_score_qcd",
        "qcd_class_name": "qcd",
        "qcd_class_label": class_labels.get("qcd", "qcd"),
        "targets": payload_targets,
        "notes": [
            "Thresholds are derived from the weighted QCD score quantile in the trainable QCD class.",
            "Recommended keep region is score <= cut, equivalent to dropping events with score > cut.",
            "Because score values are discrete, the target QCD drop fraction and the actual strict/inclusive fractions can differ slightly at the threshold.",
            "Process-level rows are evaluated on all events in predictions.npz, including eval-only processes.",
            "Training-class rows are evaluated only on trainable events with labels >= 0.",
        ],
    }



def _format_qcd_score_threshold_scan_text(payload: dict[str, object]) -> str:
    lines = [
        "QCD score threshold scan",
        "",
        f"Score branch: {payload['score_branch']}",
        "Keep region: score <= cut",
        "Drop region: score > cut",
        "Ties at the threshold can make the actual QCD drop fraction differ slightly from the target.",
        "",
        "Summary:",
        f"{'Target drop':>12}  {'Score cut':>10}  {'QCD drop > cut':>16}  {'QCD drop >= cut':>17}  {'QCD keep <= cut':>17}",
    ]

    targets = list(payload.get("targets", []))
    for row in targets:
        lines.append(
            f"{_format_fraction(float(row['target_qcd_drop_fraction'])):>12}  "
            f"{float(row['score_cut']):10.6f}  "
            f"{_format_fraction(float(row['qcd_drop_fraction_gt_cut'])):>16}  "
            f"{_format_fraction(float(row['qcd_drop_fraction_ge_cut'])):>17}  "
            f"{_format_fraction(float(row['qcd_keep_fraction_le_cut'])):>17}"
        )

    for row in targets:
        lines.extend(
            [
                "",
                (
                    f"Target QCD drop = {_format_fraction(float(row['target_qcd_drop_fraction']))}, "
                    f"score cut = {float(row['score_cut']):.6f}"
                ),
                "Process-level keep fractions and remaining composition:",
                f"{'Process':<20}  {'Keep <= cut':>12}  {'Share of kept':>14}  {'Weight kept':>12}  {'Weight total':>12}",
            ]
        )
        for process, process_row in dict(row.get("processes", {})).items():
            label = str(process_row.get("label", process))
            lines.append(
                f"{label:<20}  "
                f"{_format_fraction(float(process_row['keep_fraction_le_cut'])):>12}  "
                f"{_format_fraction(float(process_row['remaining_share_of_all_kept'])):>14}  "
                f"{_format_weight(float(process_row['weight_keep_le_cut'])):>12}  "
                f"{_format_weight(float(process_row['weight_total'])):>12}"
            )

        lines.extend(
            [
                "Training-class keep fractions and remaining composition:",
                f"{'Class':<20}  {'Keep <= cut':>12}  {'Share of kept':>14}  {'Weight kept':>12}  {'Weight total':>12}",
            ]
        )
        for class_name, class_row in dict(row.get("training_classes", {})).items():
            label = str(class_row.get("label", class_name))
            lines.append(
                f"{label:<20}  "
                f"{_format_fraction(float(class_row['keep_fraction_le_cut'])):>12}  "
                f"{_format_fraction(float(class_row['remaining_share_of_trainable_kept'])):>14}  "
                f"{_format_weight(float(class_row['weight_keep_le_cut'])):>12}  "
                f"{_format_weight(float(class_row['weight_total'])):>12}"
            )

    return "\n".join(lines) + "\n"



def _mass_correlation_summary_path(config: EventBdtConfig) -> Path:
    return config.outdir / "mass_correlation_summary.json"



def _validate_prepared_predictions_alignment(
    prepared_arrays: dict[str, np.ndarray],
    prediction_arrays: dict[str, np.ndarray],
) -> None:
    keys_to_check = [
        "event",
        "run",
        "luminosityBlock",
        "process",
        "train_label",
        "fold_id",
    ]
    prepared_size = len(np.asarray(prepared_arrays["event"]))
    prediction_size = len(np.asarray(prediction_arrays["event"]))
    if prepared_size != prediction_size:
        raise ValueError(
            "Prepared inputs and predictions have different lengths. "
            "Rebuild the cache and retrain with --force-prepare --force-retrain."
        )
    for key in keys_to_check:
        if key not in prepared_arrays or key not in prediction_arrays:
            raise KeyError(f"Alignment key '{key}' is missing from the prepared or prediction payload.")
        if not np.array_equal(np.asarray(prepared_arrays[key]), np.asarray(prediction_arrays[key])):
            raise ValueError(
                f"Prepared inputs and predictions are misaligned at '{key}'. "
                "Rebuild the cache and retrain with --force-prepare --force-retrain."
            )



def _weighted_pearson_correlation(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> float:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0.0)
    if int(np.sum(valid)) < 2:
        return float("nan")

    x_valid = np.asarray(x[valid], dtype=np.float64)
    y_valid = np.asarray(y[valid], dtype=np.float64)
    w_valid = np.asarray(weights[valid], dtype=np.float64)
    weight_sum = float(np.sum(w_valid))
    if weight_sum <= 0.0:
        return float("nan")

    x_mean = float(np.average(x_valid, weights=w_valid))
    y_mean = float(np.average(y_valid, weights=w_valid))
    x_centered = x_valid - x_mean
    y_centered = y_valid - y_mean
    x_variance = float(np.sum(w_valid * np.square(x_centered)) / weight_sum)
    y_variance = float(np.sum(w_valid * np.square(y_centered)) / weight_sum)
    if x_variance <= 0.0 or y_variance <= 0.0:
        return float("nan")
    covariance = float(np.sum(w_valid * x_centered * y_centered) / weight_sum)
    return float(covariance / np.sqrt(x_variance * y_variance))



def _weighted_correlation_matrix(
    columns: list[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    n_columns = len(columns)
    matrix = np.full((n_columns, n_columns), np.nan, dtype=np.float64)
    for row in range(n_columns):
        for column in range(row, n_columns):
            corr = _weighted_pearson_correlation(columns[row], columns[column], weights)
            matrix[row, column] = corr
            matrix[column, row] = corr
    return matrix



def _weighted_axis_range(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return (0.0, 1.0)

    valid_values = np.asarray(values[valid], dtype=np.float64)
    valid_weights = np.asarray(weights[valid], dtype=np.float64)
    lower = float(weighted_quantile(valid_values, valid_weights, MASS_RANGE_QUANTILES[0]))
    upper = float(weighted_quantile(valid_values, valid_weights, MASS_RANGE_QUANTILES[1]))
    if not np.isfinite(lower) or not np.isfinite(upper):
        lower = float(np.nanmin(valid_values))
        upper = float(np.nanmax(valid_values))
    if lower == upper:
        padding = max(1.0, abs(lower) * 0.02)
        lower -= padding
        upper += padding
    return (lower, upper)



def _build_mass_correlation_diagnostics(
    *,
    config: EventBdtConfig,
    prediction_arrays: dict[str, np.ndarray],
    prepared_arrays: dict[str, np.ndarray],
    class_names: list[str],
    class_labels: dict[str, str],
    score_by_class: dict[str, np.ndarray],
    processes: np.ndarray,
    weights: np.ndarray,
    process_order: list[str],
    process_labels: dict[str, str],
) -> dict[str, object] | None:
    if not config.analysis_branches:
        return None
    if any(score_name not in score_by_class for score_name in MASS_DIAGNOSTIC_SCORE_NAMES):
        return None

    _validate_prepared_predictions_alignment(prepared_arrays, prediction_arrays)

    required_columns = list(dict.fromkeys(list(config.features) + MASS_DIAGNOSTIC_VARIABLES))
    missing_columns = [name for name in required_columns if name not in prepared_arrays]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            "Prepared inputs are missing analysis branches required for the mass diagnostics: "
            f"{missing_text}. Rebuild the cache and retrain with --force-prepare --force-retrain."
        )

    mass_ranges = {
        mass_name: _weighted_axis_range(np.asarray(prepared_arrays[mass_name], dtype=np.float64), weights)
        for mass_name in MASS_DIAGNOSTIC_VARIABLES
    }

    payload: dict[str, object] = {
        "feature_variables": list(config.features),
        "mass_variables": list(MASS_DIAGNOSTIC_VARIABLES),
        "correlation_variables": required_columns,
        "score_names": list(MASS_DIAGNOSTIC_SCORE_NAMES),
        "score_labels": {
            score_name: class_labels.get(score_name, score_name)
            for score_name in MASS_DIAGNOSTIC_SCORE_NAMES
        },
        "processes": [process for process in process_order if np.any(processes == process)],
        "mass_axis_ranges": {
            name: {"min": float(bounds[0]), "max": float(bounds[1])}
            for name, bounds in mass_ranges.items()
        },
        "process_summaries": {},
    }

    column_by_name = {
        name: np.asarray(prepared_arrays[name], dtype=np.float64)
        for name in required_columns
    }
    process_summaries: dict[str, object] = {}
    for process in payload["processes"]:
        process_mask = processes == process
        process_weight = np.asarray(weights[process_mask], dtype=np.float64)
        correlation_columns = [column_by_name[name][process_mask] for name in required_columns]
        correlation_matrix = _weighted_correlation_matrix(correlation_columns, process_weight)

        score_mass_correlations: dict[str, object] = {}
        for score_name in MASS_DIAGNOSTIC_SCORE_NAMES:
            score_row: dict[str, object] = {
                "label": class_labels.get(score_name, score_name),
                "mass_correlations": {},
            }
            process_score = np.asarray(score_by_class[score_name][process_mask], dtype=np.float64)
            for mass_name in MASS_DIAGNOSTIC_VARIABLES:
                process_mass = np.asarray(column_by_name[mass_name][process_mask], dtype=np.float64)
                score_row["mass_correlations"][mass_name] = float(
                    _weighted_pearson_correlation(process_score, process_mass, process_weight)
                )
            score_mass_correlations[score_name] = score_row

        process_summaries[process] = {
            "label": process_labels.get(process, process),
            "n_events": int(np.sum(process_mask)),
            "weight_sum": float(np.sum(process_weight)),
            "correlation_matrix": correlation_matrix.tolist(),
            "score_mass_correlations": score_mass_correlations,
        }

    payload["process_summaries"] = process_summaries
    return payload



def _write_mass_correlation_outputs(
    *,
    config: EventBdtConfig,
    payload: dict[str, object],
    prepared_arrays: dict[str, np.ndarray],
    score_by_class: dict[str, np.ndarray],
    weights: np.ndarray,
    processes: np.ndarray,
    process_labels: dict[str, str],
    class_labels: dict[str, str],
) -> Path:
    summary_path = _mass_correlation_summary_path(config)
    _write_json(summary_path, payload)

    correlation_variables = list(payload["correlation_variables"])
    mass_ranges = {
        name: (
            float(payload["mass_axis_ranges"][name]["min"]),
            float(payload["mass_axis_ranges"][name]["max"]),
        )
        for name in MASS_DIAGNOSTIC_VARIABLES
    }
    for process in payload["processes"]:
        process_mask = processes == process
        process_label = process_labels.get(process, process)
        process_summary = payload["process_summaries"][process]
        plot_feature_mass_correlation_heatmap(
            config.plot_dir_path / f"feature_mass_correlation__{process}.png",
            matrix=np.asarray(process_summary["correlation_matrix"], dtype=np.float64),
            variable_names=correlation_variables,
            process_label=process_label,
        )
        plot_score_vs_mass_grid(
            config.plot_dir_path / f"score_vs_mass__{process}.png",
            process_label=process_label,
            score_by_name={
                score_name: np.asarray(score_by_class[score_name][process_mask], dtype=np.float64)
                for score_name in MASS_DIAGNOSTIC_SCORE_NAMES
            },
            score_labels={
                score_name: class_labels.get(score_name, score_name)
                for score_name in MASS_DIAGNOSTIC_SCORE_NAMES
            },
            mass_by_name={
                mass_name: np.asarray(prepared_arrays[mass_name][process_mask], dtype=np.float64)
                for mass_name in MASS_DIAGNOSTIC_VARIABLES
            },
            mass_names=list(MASS_DIAGNOSTIC_VARIABLES),
            mass_ranges=mass_ranges,
            weights=np.asarray(weights[process_mask], dtype=np.float64),
        )
    return summary_path



def _evaluate_outputs(config: EventBdtConfig) -> dict[str, object]:
    arrays, metadata = load_predictions_payload(config.predictions_path)
    prepared_arrays: dict[str, np.ndarray] | None = None
    training_mode = str(metadata.get("training_mode", config.training_mode))
    labels = arrays["train_label"].astype(int)
    weights = arrays["train_weight"].astype(float)
    processes = arrays["process"].astype(str)
    label_texts = arrays["label_text"].astype(str)

    trainable_mask = labels >= 0
    process_labels = _process_label_map(processes, label_texts)
    process_order = _process_order(processes)

    if training_mode == "multiclass":
        class_names = list(metadata["class_names"])
        class_labels = {
            name: label
            for name, label in zip(class_names, metadata.get("class_labels", class_names), strict=False)
        }
        score_by_class = {
            class_name: arrays[f"bdt_score_{class_name}"].astype(float)
            for class_name in class_names
        }
        weighted_auc_ovr = {
            name: float(value)
            for name, value in metadata["weighted_auc_ovr"].items()
        }
        weighted_macro_auc = float(metadata["weighted_macro_auc"])

        plot_ovr_roc_curves(
            config.plot_dir_path / "roc_ovr.png",
            score_by_class={
                class_name: score_by_class[class_name][trainable_mask]
                for class_name in class_names
            },
            labels=labels[trainable_mask],
            weights=weights[trainable_mask],
            class_names=class_names,
            class_labels=class_labels,
            auc_by_class=weighted_auc_ovr,
            macro_auc=weighted_macro_auc,
        )

        from itertools import combinations

        from sklearn.metrics import roc_auc_score

        class_index_by_name = {name: index for index, name in enumerate(class_names)}
        for positive_name, negative_name in combinations(class_names, 2):
            positive_index = class_index_by_name[positive_name]
            negative_index = class_index_by_name[negative_name]
            pair_mask = trainable_mask & (
                (labels == positive_index) | (labels == negative_index)
            )
            if not np.any(pair_mask):
                continue
            pair_positive = score_by_class[positive_name][pair_mask]
            pair_negative = score_by_class[negative_name][pair_mask]
            pair_denominator = pair_positive + pair_negative
            pair_scores = np.divide(
                pair_positive,
                pair_denominator,
                out=np.full(pair_positive.shape, 0.5, dtype=np.float64),
                where=pair_denominator > 0.0,
            )
            pair_labels = (labels[pair_mask] == positive_index).astype(np.int8)
            pair_weights = weights[pair_mask]
            if not np.any(pair_labels == 1) or not np.any(pair_labels == 0):
                continue
            pair_auc = float(
                roc_auc_score(pair_labels, pair_scores, sample_weight=pair_weights)
            )
            plot_pairwise_roc_curve(
                config.plot_dir_path / f"roc_pairwise__{positive_name}__vs__{negative_name}.png",
                scores=pair_scores,
                labels=pair_labels,
                weights=pair_weights,
                positive_label=class_labels.get(positive_name, positive_name),
                negative_label=class_labels.get(negative_name, negative_name),
                auc_value=pair_auc,
            )

        for class_name in class_names:
            score_label = class_labels.get(class_name, class_name)
            plot_training_class_score_shapes(
                config.plot_dir_path / f"score_by_training_class__{class_name}.png",
                score_name=class_name,
                score_label=score_label,
                scores=score_by_class[class_name][trainable_mask],
                labels=labels[trainable_mask],
                weights=weights[trainable_mask],
                class_names=class_names,
                class_labels=class_labels,
            )
            plot_training_class_score_weighted_events(
                config.plot_dir_path / f"score_by_training_class_weighted_events__{class_name}.png",
                score_name=class_name,
                score_label=score_label,
                scores=score_by_class[class_name][trainable_mask],
                labels=labels[trainable_mask],
                weights=weights[trainable_mask],
                class_names=class_names,
                class_labels=class_labels,
            )
            plot_training_class_score_weighted_events(
                config.plot_dir_path / f"score_by_training_class_weighted_events__{class_name}_logy.png",
                score_name=class_name,
                score_label=score_label,
                scores=score_by_class[class_name][trainable_mask],
                labels=labels[trainable_mask],
                weights=weights[trainable_mask],
                class_names=class_names,
                class_labels=class_labels,
                log_y=True,
            )
            plot_process_score_shapes(
                config.plot_dir_path / f"score_by_process__{class_name}.png",
                scores=score_by_class[class_name],
                weights=weights,
                processes=processes,
                process_order=process_order,
                process_labels=process_labels,
                x_label=f"{score_label} score",
            )
            plot_process_score_weighted_events(
                config.plot_dir_path / f"score_by_process_weighted_events__{class_name}_logy.png",
                scores=score_by_class[class_name],
                weights=weights,
                processes=processes,
                process_order=process_order,
                process_labels=process_labels,
                x_label=f"{score_label} score",
                log_y=True,
            )
            plot_process_score_weighted_events(
                config.plot_dir_path / f"score_by_process_weighted_events__{class_name}.png",
                scores=score_by_class[class_name],
                weights=weights,
                processes=processes,
                process_order=process_order,
                process_labels=process_labels,
                x_label=f"{score_label} score",
            )

        qcd_scan_payload = _build_qcd_score_threshold_scan(
            class_names=class_names,
            class_labels=class_labels,
            score_by_class=score_by_class,
            labels=labels,
            weights=weights,
            processes=processes,
            process_order=process_order,
            process_labels=process_labels,
            trainable_mask=trainable_mask,
        )
        summary: dict[str, float | str] = {"weighted_macro_auc": weighted_macro_auc}
        if qcd_scan_payload is not None:
            txt_path, json_path = _qcd_score_threshold_scan_paths(config)
            txt_path.write_text(_format_qcd_score_threshold_scan_text(qcd_scan_payload), encoding="utf-8")
            _write_json(json_path, qcd_scan_payload)
            summary["qcd_score_threshold_scan_txt"] = str(txt_path)
            summary["qcd_score_threshold_scan_json"] = str(json_path)

        if config.analysis_branches:
            if not config.prepared_inputs_path.exists():
                raise FileNotFoundError(
                    "Prepared event-BDT inputs were not found for the mass diagnostics. "
                    "Rebuild the cache and retrain with --force-prepare --force-retrain: "
                    f"{config.prepared_inputs_path}"
                )
            prepared_arrays, _ = load_npz_payload(config.prepared_inputs_path)
            mass_correlation_payload = _build_mass_correlation_diagnostics(
                config=config,
                prediction_arrays=arrays,
                prepared_arrays=prepared_arrays,
                class_names=class_names,
                class_labels=class_labels,
                score_by_class=score_by_class,
                processes=processes,
                weights=weights,
                process_order=process_order,
                process_labels=process_labels,
            )
            if mass_correlation_payload is not None:
                summary_path = _write_mass_correlation_outputs(
                    config=config,
                    payload=mass_correlation_payload,
                    prepared_arrays=prepared_arrays,
                    score_by_class=score_by_class,
                    weights=weights,
                    processes=processes,
                    process_labels=process_labels,
                    class_labels=class_labels,
                )
                summary["mass_correlation_summary_json"] = str(summary_path)
        return summary

    scores = arrays["bdt_score"].astype(float)
    roc_out = config.plot_dir_path / "roc_oof.png"
    plot_roc_curve(
        roc_out,
        scores=scores[trainable_mask],
        labels=labels[trainable_mask],
        weights=weights[trainable_mask],
        auc_value=float(metadata["weighted_auc"]),
    )
    plot_class_score_shapes(
        config.plot_dir_path / "score_signal_vs_background.png",
        scores=scores[trainable_mask],
        labels=labels[trainable_mask],
        weights=weights[trainable_mask],
    )

    plot_process_score_shapes(
        config.plot_dir_path / "score_by_process.png",
        scores=scores,
        weights=weights,
        processes=processes,
        process_order=process_order,
        process_labels=process_labels,
    )

    tth_family_order = [
        process
        for process in ["ttHcc", "ttHbb", "ttH_nonbb"]
        if process in process_order
    ]
    background_mask = labels == 0
    if np.any(background_mask):
        pieces_scores = [scores[processes == process] for process in tth_family_order]
        pieces_weights = [weights[processes == process] for process in tth_family_order]
        pieces_processes = [
            _filled_string_array(process, int(np.sum(processes == process)))
            for process in tth_family_order
        ]
        pieces_scores.append(scores[background_mask])
        pieces_weights.append(weights[background_mask])
        pieces_processes.append(
            _filled_string_array("background", int(np.sum(background_mask)))
        )
        family_labels = dict(process_labels)
        family_labels["background"] = "Background (combined)"
        plot_process_score_shapes(
            config.plot_dir_path / "score_tth_family.png",
            scores=np.concatenate(pieces_scores),
            weights=np.concatenate(pieces_weights),
            processes=np.concatenate(pieces_processes),
            process_order=tth_family_order + ["background"],
            process_labels=family_labels,
        )

    return {"weighted_auc": float(metadata["weighted_auc"])}



def _metric_line(summary: dict[str, object]) -> str:
    if "weighted_macro_auc" in summary:
        return f"Weighted macro AUC: {summary['weighted_macro_auc']:.4f}"
    return f"Weighted AUC: {summary['weighted_auc']:.4f}"



def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = _load_config(args.config)

    if args.command == "prepare":
        outpath = prepare_event_bdt_inputs(config, force=bool(args.force))
        print("Prepared event-BDT inputs.")
        print(f"Output: {outpath}")
        return

    if args.command == "train":
        summary = train_event_bdt(
            config,
            force_prepare=bool(args.force_prepare),
            force_retrain=bool(args.force_retrain),
        )
        print("Event-BDT training finished.")
        print(f"Output directory: {config.outdir}")
        print(_metric_line(summary))
        return

    if args.command == "evaluate":
        summary = _evaluate_outputs(config)
        print("Event-BDT evaluation plots finished.")
        print(f"Plot directory: {config.plot_dir_path}")
        print(_metric_line(summary))
        if "qcd_score_threshold_scan_txt" in summary:
            print(f"QCD score scan: {summary['qcd_score_threshold_scan_txt']}")
        if "mass_correlation_summary_json" in summary:
            print(f"Mass correlation summary: {summary['mass_correlation_summary_json']}")
        return

    if args.command == "predict":
        summary = predict_event_bdt_to_root(
            config,
            samples_config_path=args.samples_config,
            outdir=args.outdir,
            score_branch=str(args.score_branch),
            prediction_mode=str(args.prediction_mode),
            score_selection_only=not bool(args.score_all_events),
            force=bool(args.force),
            max_files_per_sample=args.max_files_per_sample,
        )
        print("Event-BDT prediction finished.")
        print(f"Output directory: {summary['prediction_outdir']}")
        print(f"Written files: {summary['written_files']}")
        return

    raise ValueError(f"Unknown command: {args.command}")
