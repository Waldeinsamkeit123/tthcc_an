from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairwiseRoc:
    signal: str
    background: str
    signal_efficiency: np.ndarray
    background_efficiency: np.ndarray
    thresholds: np.ndarray
    auc: float
    n_signal: int
    n_background: int
    zero_denominator_events: int
    sklearn_auc: float | None


def pairwise_discriminator(signal_score: np.ndarray, background_score: np.ndarray) -> tuple[np.ndarray, int]:
    signal = np.asarray(signal_score, dtype=np.float64)
    background = np.asarray(background_score, dtype=np.float64)
    denominator = signal + background
    valid_denominator = np.isfinite(denominator) & (denominator > 0)
    discriminator = np.full(denominator.shape, 0.5, dtype=np.float64)
    np.divide(signal, denominator, out=discriminator, where=valid_denominator)
    invalid_count = int(np.sum(~valid_denominator))
    return discriminator, invalid_count


def weighted_roc(
    scores: np.ndarray,
    is_signal: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(is_signal, dtype=bool)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    scores = scores[valid]
    labels = labels[valid]
    weights = weights[valid]
    total_signal = float(np.sum(weights[labels]))
    total_background = float(np.sum(weights[~labels]))
    if scores.size == 0 or total_signal <= 0 or total_background <= 0:
        raise ValueError("ROC computation requires positive-weight signal and background events.")

    order = np.argsort(scores, kind="mergesort")[::-1]
    scores = scores[order]
    labels = labels[order]
    weights = weights[order]
    cumulative_signal = np.cumsum(np.where(labels, weights, 0.0))
    cumulative_background = np.cumsum(np.where(labels, 0.0, weights))

    # Keep the last event at each tied threshold so tied scores move together.
    threshold_ends = np.r_[scores[:-1] != scores[1:], True]
    signal_efficiency = cumulative_signal[threshold_ends] / total_signal
    background_efficiency = cumulative_background[threshold_ends] / total_background
    thresholds = scores[threshold_ends]
    signal_efficiency = np.r_[0.0, signal_efficiency]
    background_efficiency = np.r_[0.0, background_efficiency]
    thresholds = np.r_[np.inf, thresholds]
    auc = float(np.clip(np.trapezoid(signal_efficiency, background_efficiency), 0.0, 1.0))
    return signal_efficiency, background_efficiency, thresholds, auc


def compute_confusion_matrix(
    score_matrix: np.ndarray,
    truth_index: np.ndarray,
    weights: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, int]:
    scores = np.asarray(score_matrix, dtype=np.float64)
    truth = np.asarray(truth_index, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = (
        (truth >= 0)
        & (truth < n_classes)
        & np.isfinite(weights)
        & (weights > 0)
        & np.all(np.isfinite(scores), axis=1)
    )
    matrix = np.zeros((n_classes, n_classes), dtype=np.float64)
    if np.any(valid):
        prediction = np.argmax(scores[valid], axis=1)
        np.add.at(matrix, (truth[valid], prediction), weights[valid])
    return matrix, int(np.sum(valid))


def normalize_confusion(matrix: np.ndarray, axis: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if axis == "truth":
        denominator = np.sum(matrix, axis=1, keepdims=True)
    elif axis == "prediction":
        denominator = np.sum(matrix, axis=0, keepdims=True)
    else:
        raise ValueError("Confusion normalization axis must be 'truth' or 'prediction'.")
    return np.divide(matrix, denominator, out=np.zeros_like(matrix), where=denominator > 0)


def compute_pairwise_rocs(
    score_names: list[str],
    scores: dict[str, np.ndarray],
    truth_index: np.ndarray,
    weights: np.ndarray,
    *,
    validate_with_sklearn: bool,
) -> dict[str, dict[str, PairwiseRoc]]:
    sklearn_roc_auc_score = None
    if validate_with_sklearn:
        try:
            from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
        except ImportError:
            pass

    results: dict[str, dict[str, PairwiseRoc]] = {name: {} for name in score_names}
    truth = np.asarray(truth_index, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    for signal_index, signal_name in enumerate(score_names):
        for background_index, background_name in enumerate(score_names):
            if signal_index == background_index:
                continue
            pair_mask = (truth == signal_index) | (truth == background_index)
            pair_weights = weights[pair_mask]
            labels = truth[pair_mask] == signal_index
            if not np.any(labels) or not np.any(~labels):
                continue
            discriminator, zero_count = pairwise_discriminator(
                scores[signal_name][pair_mask],
                scores[background_name][pair_mask],
            )
            signal_eff, background_eff, thresholds, auc = weighted_roc(
                discriminator, labels, pair_weights
            )
            sklearn_auc = None
            if sklearn_roc_auc_score is not None:
                valid = (
                    np.isfinite(discriminator)
                    & np.isfinite(pair_weights)
                    & (pair_weights > 0)
                )
                sklearn_auc = float(
                    sklearn_roc_auc_score(
                        labels[valid].astype(np.int8),
                        discriminator[valid],
                        sample_weight=pair_weights[valid],
                    )
                )
            results[signal_name][background_name] = PairwiseRoc(
                signal=signal_name,
                background=background_name,
                signal_efficiency=signal_eff,
                background_efficiency=background_eff,
                thresholds=thresholds,
                auc=auc,
                n_signal=int(np.sum(labels)),
                n_background=int(np.sum(~labels)),
                zero_denominator_events=zero_count,
                sklearn_auc=sklearn_auc,
            )
    return results
