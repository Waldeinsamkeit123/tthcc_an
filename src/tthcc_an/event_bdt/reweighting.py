from __future__ import annotations

from typing import Any

import numpy as np


SIGNAL_GROUP = "signal"
BACKGROUND_GROUP = "background"


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numerator, dtype=np.float64)
    valid = denominator > 0
    out[valid] = numerator[valid] / denominator[valid]
    return out


def _label_group_masks(
    labels: np.ndarray,
    train_mask: np.ndarray,
    class_groups: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    signal_indices = np.array(
        [index for index, group in enumerate(class_groups) if group == SIGNAL_GROUP],
        dtype=np.int16,
    )
    background_indices = np.array(
        [index for index, group in enumerate(class_groups) if group == BACKGROUND_GROUP],
        dtype=np.int16,
    )
    signal_mask = train_mask & np.isin(labels, signal_indices)
    background_mask = train_mask & np.isin(labels, background_indices)
    return signal_mask, background_mask


def _weight_sums_by_label(
    weights: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    class_indices: list[int],
) -> dict[str, float]:
    return {
        str(class_index): float(np.sum(weights[train_mask & (labels == class_index)]))
        for class_index in class_indices
    }


def _normalize_class_weights(
    weights: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    class_indices: list[int],
    target_sum: float,
) -> None:
    for class_index in class_indices:
        class_mask = train_mask & (labels == class_index)
        class_sum = float(np.sum(weights[class_mask]))
        if class_sum > 0.0:
            weights[class_mask] *= target_sum / class_sum


def _build_class_balanced_training_weights(
    *,
    weights: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    columns: dict[str, np.ndarray],
    reweighting_config: dict[str, Any],
    class_groups: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    class_indices = [int(index) for index in np.unique(labels[train_mask]) if index >= 0]
    if len(class_indices) < 2:
        return weights, {"enabled": False, "reason": "fewer than two trainable classes"}

    signal_indices = [
        index for index, group in enumerate(class_groups)
        if group == SIGNAL_GROUP and index in class_indices
    ]
    reference_index = signal_indices[0] if signal_indices else class_indices[0]
    reference_mask = train_mask & (labels == reference_index)
    target_sum = float(np.sum(weights[reference_mask]))
    if target_sum <= 0.0:
        return weights, {"enabled": False, "reason": "reference class has zero weight"}

    initial_weight_sums = _weight_sums_by_label(weights, labels, train_mask, class_indices)
    _normalize_class_weights(weights, labels, train_mask, class_indices, target_sum)

    variable_summaries: list[dict[str, Any]] = []
    for variable in list(reweighting_config.get("variables", [])):
        name = str(variable["name"])
        bins = np.asarray(variable["bins"], dtype=np.float64)
        values = np.asarray(columns[name], dtype=np.float64)
        finite_reference = reference_mask & np.isfinite(values)
        if not np.any(finite_reference):
            variable_summaries.append(
                {
                    "name": name,
                    "status": "skipped",
                    "reason": "no finite reference-class entries",
                }
            )
            continue

        clipped_reference = np.clip(values[finite_reference], bins[0], bins[-1])
        hist_reference, _ = np.histogram(
            clipped_reference,
            bins=bins,
            weights=weights[finite_reference],
        )
        bin_indices = np.clip(
            np.digitize(np.clip(values, bins[0], bins[-1]), bins) - 1,
            a_min=0,
            a_max=len(bins) - 2,
        )

        applied_classes: list[int] = []
        skipped_classes: list[int] = []
        for class_index in class_indices:
            if class_index == reference_index:
                continue
            class_mask = train_mask & (labels == class_index)
            finite_class = class_mask & np.isfinite(values)
            if not np.any(finite_class):
                skipped_classes.append(class_index)
                continue
            clipped_class = np.clip(values[finite_class], bins[0], bins[-1])
            hist_class, _ = np.histogram(
                clipped_class,
                bins=bins,
                weights=weights[finite_class],
            )
            ratio = _safe_ratio(hist_reference, hist_class)
            weights[finite_class] *= ratio[bin_indices[finite_class]]
            applied_classes.append(class_index)

        _normalize_class_weights(weights, labels, train_mask, class_indices, target_sum)
        variable_summaries.append(
            {
                "name": name,
                "status": "applied",
                "bins": bins.tolist(),
                "reference_class_index": reference_index,
                "applied_class_indices": applied_classes,
                "skipped_class_indices": skipped_classes,
            }
        )

    return weights, {
        "enabled": True,
        "mode": "class_balance",
        "reference_class_index": reference_index,
        "target_class_weight_sum": target_sum,
        "class_weight_sums_before": initial_weight_sums,
        "class_weight_sums_after": _weight_sums_by_label(weights, labels, train_mask, class_indices),
        "variables": variable_summaries,
    }


def _build_group_balanced_training_weights(
    *,
    weights: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    columns: dict[str, np.ndarray],
    reweighting_config: dict[str, Any],
    class_groups: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    signal_mask, background_mask = _label_group_masks(labels, train_mask, class_groups)
    if not np.any(signal_mask) or not np.any(background_mask):
        return weights, {"enabled": False, "reason": "missing signal-like or background-like events"}

    target_signal_sum = float(np.sum(weights[signal_mask]))
    background_sum = float(np.sum(weights[background_mask]))
    if background_sum > 0:
        weights[background_mask] *= target_signal_sum / background_sum

    variable_summaries: list[dict[str, Any]] = []
    for variable in list(reweighting_config.get("variables", [])):
        name = str(variable["name"])
        bins = np.asarray(variable["bins"], dtype=np.float64)
        values = np.asarray(columns[name], dtype=np.float64)
        finite_signal = signal_mask & np.isfinite(values)
        finite_background = background_mask & np.isfinite(values)
        if not np.any(finite_signal) or not np.any(finite_background):
            variable_summaries.append(
                {
                    "name": name,
                    "status": "skipped",
                    "reason": "no finite signal-like or background-like entries",
                }
            )
            continue

        clipped_signal = np.clip(values[finite_signal], bins[0], bins[-1])
        clipped_background = np.clip(values[finite_background], bins[0], bins[-1])
        hist_signal, _ = np.histogram(
            clipped_signal,
            bins=bins,
            weights=weights[finite_signal],
        )
        hist_background, _ = np.histogram(
            clipped_background,
            bins=bins,
            weights=weights[finite_background],
        )
        ratio = _safe_ratio(hist_signal, hist_background)
        bin_indices = np.clip(
            np.digitize(np.clip(values, bins[0], bins[-1]), bins) - 1,
            a_min=0,
            a_max=len(bins) - 2,
        )
        scale = np.ones_like(weights)
        scale[finite_background] = ratio[bin_indices[finite_background]]
        weights *= scale

        background_sum = float(np.sum(weights[background_mask]))
        if background_sum > 0:
            weights[background_mask] *= target_signal_sum / background_sum

        variable_summaries.append(
            {
                "name": name,
                "status": "applied",
                "bins": bins.tolist(),
            }
        )

    return weights, {"enabled": True, "mode": "group_balance", "variables": variable_summaries}


def build_reweighted_training_weights(
    base_weights: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    columns: dict[str, np.ndarray],
    reweighting_config: dict[str, Any],
    class_groups: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    weights = np.asarray(base_weights, dtype=np.float64).copy()
    if not bool(reweighting_config.get("enabled", False)):
        return weights, {"enabled": False, "variables": []}

    train_labels = np.unique(labels[train_mask & (labels >= 0)])
    if train_labels.size > 2:
        return _build_class_balanced_training_weights(
            weights=weights,
            labels=labels,
            train_mask=train_mask,
            columns=columns,
            reweighting_config=reweighting_config,
            class_groups=class_groups,
        )

    return _build_group_balanced_training_weights(
        weights=weights,
        labels=labels,
        train_mask=train_mask,
        columns=columns,
        reweighting_config=reweighting_config,
        class_groups=class_groups,
    )
