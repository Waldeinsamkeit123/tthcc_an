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
    plot_training_metric_curves,
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
CLASS_SCORE_DROP_TARGETS = [0.70, *[value / 100.0 for value in range(90, 100)], 0.995, 0.999]
QCD_SCORE_DROP_TARGETS = CLASS_SCORE_DROP_TARGETS

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



def _class_score_threshold_scan_paths(config: EventBdtConfig) -> tuple[Path, Path]:
    base = config.outdir / "class_score_threshold_scan"
    return base.with_suffix(".txt"), base.with_suffix(".json")



def _qcd_score_threshold_scan_paths(config: EventBdtConfig) -> tuple[Path, Path]:
    base = config.outdir / "qcd_score_threshold_scan"
    return base.with_suffix(".txt"), base.with_suffix(".json")



def _event_bdt_score_branch(class_name: str) -> str:
    return f"bdt_score_{class_name}"



def _safe_significance(signal_weight: float, background_weight: float) -> dict[str, float]:
    signal_weight = float(signal_weight)
    background_weight = float(background_weight)
    total_weight = signal_weight + background_weight
    return {
        "signal_weight_keep": signal_weight,
        "background_weight_keep": background_weight,
        "s_over_b": _safe_fraction(signal_weight, background_weight),
        "s_over_sqrt_s_plus_b": signal_weight / np.sqrt(total_weight) if total_weight > 0.0 else float("nan"),
        "s_over_sqrt_b": signal_weight / np.sqrt(background_weight) if background_weight > 0.0 else float("nan"),
    }


def _score_keep_direction(score_name: str) -> str:
    return "low" if score_name == "qcd" else "high"


def _score_keep_region_text(score_name: str) -> str:
    return "score <= cut" if _score_keep_direction(score_name) == "low" else "score >= cut"


def _score_drop_region_text(score_name: str) -> str:
    return "score > cut" if _score_keep_direction(score_name) == "low" else "score < cut"


def _target_score_cut(
    score_name: str,
    signal_scores: np.ndarray,
    signal_weights: np.ndarray,
    target_drop_fraction: float,
) -> float:
    quantile = 1.0 - target_drop_fraction if _score_keep_direction(score_name) == "low" else target_drop_fraction
    return float(weighted_quantile(signal_scores, signal_weights, max(0.0, min(1.0, quantile))))


def _score_region_masks(score_name: str, scores: np.ndarray, score_cut: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _score_keep_direction(score_name) == "low":
        keep_mask = scores <= score_cut
        drop_mask_strict = scores > score_cut
        drop_mask_inclusive = scores >= score_cut
    else:
        keep_mask = scores >= score_cut
        drop_mask_strict = scores < score_cut
        drop_mask_inclusive = scores <= score_cut
    return keep_mask, drop_mask_strict, drop_mask_inclusive


def _format_metric(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:12.6f}"



def _build_class_score_threshold_scan(
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
    score_payloads: dict[str, object] = {}
    score_order: list[str] = []
    class_index_by_name = {class_name: class_index for class_index, class_name in enumerate(class_names)}

    for signal_class_index, score_name in enumerate(class_names):
        if score_name not in score_by_class:
            continue

        scores = np.asarray(score_by_class[score_name], dtype=np.float64)
        signal_class_mask = trainable_mask & (labels == signal_class_index)
        background_class_mask = trainable_mask & (labels >= 0) & (labels != signal_class_index)
        signal_total_weight = float(np.sum(weights[signal_class_mask]))
        background_total_weight = float(np.sum(weights[background_class_mask]))
        if signal_total_weight <= 0.0:
            continue

        signal_scores = scores[signal_class_mask]
        signal_weights = weights[signal_class_mask]
        keep_direction = _score_keep_direction(score_name)
        keep_region = _score_keep_region_text(score_name)
        drop_region = _score_drop_region_text(score_name)
        targets: list[dict[str, object]] = []
        for target_drop_fraction in CLASS_SCORE_DROP_TARGETS:
            score_cut = _target_score_cut(score_name, signal_scores, signal_weights, target_drop_fraction)
            keep_mask, drop_mask_strict, drop_mask_inclusive = _score_region_masks(score_name, scores, score_cut)
            kept_trainable_mask = trainable_mask & keep_mask

            signal_weight_drop_strict = float(np.sum(weights[signal_class_mask & drop_mask_strict]))
            signal_weight_drop_inclusive = float(np.sum(weights[signal_class_mask & drop_mask_inclusive]))
            signal_weight_keep = float(np.sum(weights[signal_class_mask & keep_mask]))
            background_weight_keep = float(np.sum(weights[background_class_mask & keep_mask]))
            total_weight_keep_all = float(np.sum(weights[keep_mask]))
            total_weight_keep_trainable = float(np.sum(weights[kept_trainable_mask]))

            target_payload: dict[str, object] = {
                "target_signal_drop_fraction": float(target_drop_fraction),
                "score_cut": score_cut,
                "keep_direction": keep_direction,
                "keep_region": keep_region,
                "drop_region": drop_region,
                "signal_drop_fraction_strict": _safe_fraction(signal_weight_drop_strict, signal_total_weight),
                "signal_drop_fraction_inclusive": _safe_fraction(signal_weight_drop_inclusive, signal_total_weight),
                "signal_keep_fraction_in_region": _safe_fraction(signal_weight_keep, signal_total_weight),
                "signal_weight_total": signal_total_weight,
                "signal_weight_keep_in_region": signal_weight_keep,
                "background_weight_total": background_total_weight,
                "background_weight_keep_in_region": background_weight_keep,
                "significance": _safe_significance(signal_weight_keep, background_weight_keep),
                "total_weight_keep_all_processes": total_weight_keep_all,
                "total_weight_keep_trainable_classes": total_weight_keep_trainable,
                "processes": {},
                "training_classes": {},
            }
            if score_name == "qcd":
                target_payload.update(
                    {
                        "target_qcd_drop_fraction": float(target_drop_fraction),
                        "qcd_drop_fraction_gt_cut": target_payload["signal_drop_fraction_strict"],
                        "qcd_drop_fraction_ge_cut": target_payload["signal_drop_fraction_inclusive"],
                        "qcd_keep_fraction_le_cut": target_payload["signal_keep_fraction_in_region"],
                        "qcd_weight_total": signal_total_weight,
                        "qcd_weight_keep_le_cut": signal_weight_keep,
                    }
                )
                auxiliary_significances: dict[str, object] = {}
                tth_index = class_index_by_name.get("tth")
                if tth_index is not None:
                    auxiliary_signal_mask = trainable_mask & (labels == tth_index)
                    auxiliary_background_mask = np.zeros(labels.shape, dtype=bool)
                    background_names = []
                    background_labels = {}
                    for background_name in ["ttbar", "qcd"]:
                        background_index = class_index_by_name.get(background_name)
                        if background_index is None:
                            continue
                        auxiliary_background_mask |= trainable_mask & (labels == background_index)
                        background_names.append(background_name)
                        background_labels[background_name] = class_labels.get(background_name, background_name)
                    auxiliary_signal_weight_keep = float(np.sum(weights[auxiliary_signal_mask & keep_mask]))
                    auxiliary_background_weight_keep = float(np.sum(weights[auxiliary_background_mask & keep_mask]))
                    auxiliary_significances["tth_vs_ttbar_plus_qcd"] = {
                        "signal_class_name": "tth",
                        "signal_class_label": class_labels.get("tth", "tth"),
                        "background_class_names": background_names,
                        "background_class_labels": background_labels,
                        "signal_weight_total": float(np.sum(weights[auxiliary_signal_mask])),
                        "background_weight_total": float(np.sum(weights[auxiliary_background_mask])),
                        **_safe_significance(auxiliary_signal_weight_keep, auxiliary_background_weight_keep),
                    }
                if auxiliary_significances:
                    target_payload["auxiliary_significances"] = auxiliary_significances

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
                    "n_keep_in_region": int(np.sum(process_mask & keep_mask)),
                    "weight_total": total_weight,
                    "weight_keep_in_region": kept_weight,
                    "keep_fraction_in_region": _safe_fraction(kept_weight, total_weight),
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
                    "n_keep_in_region": int(np.sum(class_mask & keep_mask)),
                    "weight_total": total_weight,
                    "weight_keep_in_region": kept_weight,
                    "keep_fraction_in_region": _safe_fraction(kept_weight, total_weight),
                    "remaining_share_of_trainable_kept": _safe_fraction(kept_weight, total_weight_keep_trainable),
                    "is_signal_for_score": bool(class_name == score_name),
                }
            target_payload["training_classes"] = class_payload
            targets.append(target_payload)

        background_class_names = [name for name in class_names if name != score_name]
        score_payloads[score_name] = {
            "score_name": score_name,
            "score_branch": _event_bdt_score_branch(score_name),
            "score_label": class_labels.get(score_name, score_name),
            "signal_class_name": score_name,
            "signal_class_label": class_labels.get(score_name, score_name),
            "background_class_names": background_class_names,
            "background_class_labels": {
                name: class_labels.get(name, name)
                for name in background_class_names
            },
            "keep_direction": keep_direction,
            "keep_region": keep_region,
            "drop_region": drop_region,
            "targets": targets,
        }
        score_order.append(score_name)

    if not score_payloads:
        return None

    return {
        "scan_targets": list(CLASS_SCORE_DROP_TARGETS),
        "score_order": score_order,
        "score_directions": {
            score_name: {
                "keep_direction": _score_keep_direction(score_name),
                "keep_region": _score_keep_region_text(score_name),
                "drop_region": _score_drop_region_text(score_name),
            }
            for score_name in score_order
        },
        "scores": score_payloads,
        "notes": [
            "Thresholds are derived from the weighted score quantile in the corresponding trainable class.",
            "For tth and ttbar, keep region is score >= cut. For qcd, keep region is score <= cut.",
            "For each main score summary, S is the corresponding training class and B is every other trainable class.",
            "The qcd section also includes an auxiliary ttH significance on the same QCD cut, with S=tth and B=ttbar+qcd.",
            "Because score values are discrete, the target class drop fraction and the actual strict/inclusive fractions can differ slightly at the threshold.",
            "Process-level rows are evaluated on all events in predictions.npz, including eval-only processes.",
            "Training-class rows are evaluated only on trainable events with labels >= 0.",
        ],
    }


def _qcd_score_threshold_scan_from_class_scan(payload: dict[str, object] | None) -> dict[str, object] | None:
    if payload is None:
        return None
    scores = payload.get("scores", {})
    if not isinstance(scores, dict) or "qcd" not in scores:
        return None
    qcd_payload = dict(scores["qcd"])
    qcd_payload.update(
        {
            "qcd_class_name": qcd_payload.get("signal_class_name", "qcd"),
            "qcd_class_label": qcd_payload.get("signal_class_label", "QCD"),
            "keep_direction": qcd_payload.get("keep_direction", _score_keep_direction("qcd")),
            "keep_region": qcd_payload.get("keep_region", _score_keep_region_text("qcd")),
            "drop_region": qcd_payload.get("drop_region", _score_drop_region_text("qcd")),
            "notes": payload.get("notes", []),
        }
    )
    return qcd_payload


def _format_significance_columns(row: dict[str, object]) -> str:
    significance = dict(row.get("significance", {}))
    return (
        f"{_format_weight(float(significance.get('signal_weight_keep', float('nan')))):>12}  "
        f"{_format_weight(float(significance.get('background_weight_keep', float('nan')))):>12}  "
        f"{_format_metric(float(significance.get('s_over_b', float('nan')))):>12}  "
        f"{_format_metric(float(significance.get('s_over_sqrt_s_plus_b', float('nan')))):>14}  "
        f"{_format_metric(float(significance.get('s_over_sqrt_b', float('nan')))):>12}"
    )



def _format_score_scan_section(score_payload: dict[str, object], *, title: str) -> list[str]:
    signal_label = str(score_payload.get("signal_class_label", score_payload.get("score_name", "signal")))
    signal_name = str(score_payload.get("signal_class_name", score_payload.get("score_name", "signal")))
    background_labels = dict(score_payload.get("background_class_labels", {}))
    background_text = ", ".join(str(label) for label in background_labels.values()) or "none"
    keep_region = str(score_payload.get("keep_region", "score <= cut"))
    drop_region = str(score_payload.get("drop_region", "score > cut"))
    lines = [
        title,
        f"Score branch: {score_payload['score_branch']}",
        f"Signal class for S: {signal_label} ({signal_name})",
        f"Background classes for B: {background_text}",
        f"Keep region: {keep_region}",
        f"Drop region: {drop_region}",
        "Ties at the threshold can make the actual class drop fraction differ slightly from the target.",
        "",
        "Summary:",
        (
            f"{'Target drop':>12}  {'Score cut':>10}  {'Class drop':>12}  "
            f"{'Class keep':>12}  {'S kept':>12}  {'B kept':>12}  "
            f"{'S/B':>12}  {'S/sqrt(S+B)':>14}  {'S/sqrt(B)':>12}"
        ),
    ]
    targets = list(score_payload.get("targets", []))
    for row in targets:
        lines.append(
            f"{_format_fraction(float(row['target_signal_drop_fraction'])):>12}  "
            f"{float(row['score_cut']):10.6f}  "
            f"{_format_fraction(float(row['signal_drop_fraction_strict'])):>12}  "
            f"{_format_fraction(float(row['signal_keep_fraction_in_region'])):>12}  "
            f"{_format_significance_columns(row)}"
        )

    has_auxiliary_qcd_tth = any(
        "tth_vs_ttbar_plus_qcd" in dict(row.get("auxiliary_significances", {}))
        for row in targets
    )
    if has_auxiliary_qcd_tth:
        lines.extend(
            [
                "",
                "Auxiliary ttH significance on the same cut:",
                (
                    f"{'Target drop':>12}  {'Score cut':>10}  {'S kept':>12}  {'B kept':>12}  "
                    f"{'S/B':>12}  {'S/sqrt(S+B)':>14}  {'S/sqrt(B)':>12}"
                ),
            ]
        )
        for row in targets:
            auxiliary = dict(row.get("auxiliary_significances", {})).get("tth_vs_ttbar_plus_qcd")
            if auxiliary is None:
                continue
            auxiliary = dict(auxiliary)
            lines.append(
                f"{_format_fraction(float(row['target_signal_drop_fraction'])):>12}  "
                f"{float(row['score_cut']):10.6f}  "
                f"{_format_weight(float(auxiliary.get('signal_weight_keep', float('nan')))):>12}  "
                f"{_format_weight(float(auxiliary.get('background_weight_keep', float('nan')))):>12}  "
                f"{_format_metric(float(auxiliary.get('s_over_b', float('nan')))):>12}  "
                f"{_format_metric(float(auxiliary.get('s_over_sqrt_s_plus_b', float('nan')))):>14}  "
                f"{_format_metric(float(auxiliary.get('s_over_sqrt_b', float('nan')))):>12}"
            )

    for row in targets:
        lines.extend(
            [
                "",
                (
                    f"Target class drop = {_format_fraction(float(row['target_signal_drop_fraction']))}, "
                    f"score cut = {float(row['score_cut']):.6f}"
                ),
                "Process-level keep fractions and remaining composition:",
                f"{'Process':<20}  {'Keep in region':>14}  {'Share of kept':>14}  {'Weight kept':>12}  {'Weight total':>12}",
            ]
        )
        for process, process_row in dict(row.get("processes", {})).items():
            label = str(process_row.get("label", process))
            lines.append(
                f"{label:<20}  "
                f"{_format_fraction(float(process_row['keep_fraction_in_region'])):>14}  "
                f"{_format_fraction(float(process_row['remaining_share_of_all_kept'])):>14}  "
                f"{_format_weight(float(process_row['weight_keep_in_region'])):>12}  "
                f"{_format_weight(float(process_row['weight_total'])):>12}"
            )

        lines.extend(
            [
                "Training-class keep fractions and remaining composition:",
                f"{'Class':<20}  {'Role':>8}  {'Keep in region':>14}  {'Share of kept':>14}  {'Weight kept':>12}  {'Weight total':>12}",
            ]
        )
        for class_name, class_row in dict(row.get("training_classes", {})).items():
            label = str(class_row.get("label", class_name))
            role = "S" if bool(class_row.get("is_signal_for_score", False)) else "B"
            lines.append(
                f"{label:<20}  "
                f"{role:>8}  "
                f"{_format_fraction(float(class_row['keep_fraction_in_region'])):>14}  "
                f"{_format_fraction(float(class_row['remaining_share_of_trainable_kept'])):>14}  "
                f"{_format_weight(float(class_row['weight_keep_in_region'])):>12}  "
                f"{_format_weight(float(class_row['weight_total'])):>12}"
            )
    return lines


def _format_class_score_threshold_scan_text(payload: dict[str, object]) -> str:
    lines = [
        "Class score threshold scan",
        "",
        "Keep/drop direction depends on the score.",
        "For tth and ttbar, keep region is score >= cut. For qcd, keep region is score <= cut.",
        "For each main score summary, S is the corresponding training class and B is every other trainable class.",
        "The qcd section also includes an auxiliary ttH significance with S=tth and B=ttbar+qcd.",
        "",
    ]
    scores = dict(payload.get("scores", {}))
    for index, score_name in enumerate(list(payload.get("score_order", []))):
        score_payload = dict(scores.get(score_name, {}))
        if not score_payload:
            continue
        if index:
            lines.extend(["", "=" * 96, ""])
        title = f"Score scan: {score_payload.get('score_label', score_name)} ({score_name})"
        lines.extend(_format_score_scan_section(score_payload, title=title))
    return "\n".join(lines) + "\n"


def _format_qcd_score_threshold_scan_text(payload: dict[str, object]) -> str:
    lines = [
        "QCD score threshold scan",
        "",
    ]
    lines.extend(_format_score_scan_section(payload, title="Score scan: QCD (qcd)"))
    return "\n".join(lines) + "\n"



def _training_metric_names(fold_summaries: list[dict[str, object]]) -> list[str]:
    metric_names: list[str] = []
    for fold_summary in fold_summaries:
        evals_result = fold_summary.get("evals_result", {})
        if not isinstance(evals_result, dict):
            continue
        for dataset_metrics in evals_result.values():
            if not isinstance(dataset_metrics, dict):
                continue
            for metric_name in dataset_metrics:
                if metric_name not in metric_names:
                    metric_names.append(str(metric_name))
    preferred = ["mlogloss", "logloss", "merror", "auc"]
    return [name for name in preferred if name in metric_names] + [
        name for name in metric_names if name not in preferred
    ]



def _safe_plot_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in name)



def _load_training_summary(config: EventBdtConfig) -> dict[str, object] | None:
    if not config.summary_path.exists():
        return None
    with config.summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)



def _write_training_curve_outputs(
    config: EventBdtConfig,
    training_summary: dict[str, object] | None,
) -> list[str]:
    if training_summary is None:
        return []
    fold_summaries = list(training_summary.get("fold_summaries", []))
    if not fold_summaries:
        return []
    outputs: list[str] = []
    for metric_name in _training_metric_names(fold_summaries):
        outpath = config.plot_dir_path / f"training_curve__{_safe_plot_name(metric_name)}.png"
        plot_training_metric_curves(outpath, fold_summaries=fold_summaries, metric_name=metric_name)
        outputs.append(str(outpath))
    return outputs



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

        class_scan_payload = _build_class_score_threshold_scan(
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
        summary: dict[str, object] = {"weighted_macro_auc": weighted_macro_auc}
        if class_scan_payload is not None:
            txt_path, json_path = _class_score_threshold_scan_paths(config)
            txt_path.write_text(_format_class_score_threshold_scan_text(class_scan_payload), encoding="utf-8")
            _write_json(json_path, class_scan_payload)
            summary["class_score_threshold_scan_txt"] = str(txt_path)
            summary["class_score_threshold_scan_json"] = str(json_path)

            qcd_scan_payload = _qcd_score_threshold_scan_from_class_scan(class_scan_payload)
            if qcd_scan_payload is not None:
                qcd_txt_path, qcd_json_path = _qcd_score_threshold_scan_paths(config)
                qcd_txt_path.write_text(_format_qcd_score_threshold_scan_text(qcd_scan_payload), encoding="utf-8")
                _write_json(qcd_json_path, qcd_scan_payload)
                summary["qcd_score_threshold_scan_txt"] = str(qcd_txt_path)
                summary["qcd_score_threshold_scan_json"] = str(qcd_json_path)

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
        training_curve_paths = _write_training_curve_outputs(config, _load_training_summary(config))
        if training_curve_paths:
            summary["training_curve_plots"] = training_curve_paths
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

    summary: dict[str, object] = {"weighted_auc": float(metadata["weighted_auc"])}
    training_curve_paths = _write_training_curve_outputs(config, _load_training_summary(config))
    if training_curve_paths:
        summary["training_curve_plots"] = training_curve_paths
    return summary



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
        training_curve_paths = _write_training_curve_outputs(config, summary)
        print("Event-BDT training finished.")
        print(f"Output directory: {config.outdir}")
        print(_metric_line(summary))
        if training_curve_paths:
            print(f"Training curves: {', '.join(training_curve_paths)}")
        return

    if args.command == "evaluate":
        summary = _evaluate_outputs(config)
        print("Event-BDT evaluation plots finished.")
        print(f"Plot directory: {config.plot_dir_path}")
        print(_metric_line(summary))
        if "class_score_threshold_scan_txt" in summary:
            print(f"Class score scan: {summary['class_score_threshold_scan_txt']}")
        if "qcd_score_threshold_scan_txt" in summary:
            print(f"QCD score scan: {summary['qcd_score_threshold_scan_txt']}")
        if "training_curve_plots" in summary:
            print(f"Training curves: {', '.join(summary['training_curve_plots'])}")
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
