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

    return weights, {"enabled": True, "variables": variable_summaries}
