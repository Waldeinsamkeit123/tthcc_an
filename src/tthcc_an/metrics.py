from __future__ import annotations

from typing import Any

import numpy as np

from tthcc_an.definitions import TARGET_DEFINITIONS, TRUTH_LABEL_ORDER, TRUTH_LABEL_TO_CODE


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(numerator.shape, np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    out[valid] = numerator[valid] / denominator[valid]
    zero_mask = np.isfinite(numerator) & np.isfinite(denominator) & (denominator <= 0)
    out[zero_mask] = 0.0
    return out


def weighted_sum(weights: np.ndarray) -> float:
    return float(np.sum(weights)) if weights.size else 0.0


def weighted_efficiency(scores: np.ndarray, weights: np.ndarray, cut: float) -> float:
    total = weighted_sum(weights)
    if total <= 0:
        return float("nan")
    return float(weighted_sum(weights[scores >= cut]) / total)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if values.size == 0:
        raise ValueError("Cannot compute a weighted quantile on an empty array.")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    total = cumulative[-1]
    if total <= 0:
        raise ValueError("Weighted quantile requires positive total weight.")
    threshold = quantile * total
    idx = int(np.searchsorted(cumulative, threshold, side="left"))
    idx = min(max(idx, 0), values.size - 1)
    return float(values[idx])


def truth_mask(truth_codes: np.ndarray, label: str) -> np.ndarray:
    return truth_codes == TRUTH_LABEL_TO_CODE[label]


def codes_for_labels(labels: list[str]) -> np.ndarray:
    return np.asarray([TRUTH_LABEL_TO_CODE[label] for label in labels], dtype=np.int32)


def build_target_masks(truth_codes: np.ndarray, target: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    definition = TARGET_DEFINITIONS[target]
    signal_codes = np.asarray([TRUTH_LABEL_TO_CODE[label] for label in definition["signal_labels"]], dtype=truth_codes.dtype)
    background_codes = np.asarray(
        [TRUTH_LABEL_TO_CODE[label] for label in definition["background_labels"]],
        dtype=truth_codes.dtype,
    )
    signal_mask = np.isin(truth_codes, signal_codes)
    background_mask = np.isin(truth_codes, background_codes)
    return signal_mask, background_mask, list(definition["background_labels"])


def compute_working_points(
    scores: np.ndarray,
    truth_codes: np.ndarray,
    weights: np.ndarray,
    signed_weights: np.ndarray,
    target: str,
    sig_effs: list[float],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    signal_mask, background_mask, background_labels = build_target_masks(truth_codes, target)

    signal_scores = scores[signal_mask & valid]
    signal_weights = weights[signal_mask & valid]
    background_scores = scores[background_mask & valid]
    if signal_scores.size == 0 or background_scores.size == 0:
        raise ValueError(f"No usable signal/background jets found for target '{target}'.")

    counts_by_label: dict[str, dict[str, float]] = {}
    for label in TRUTH_LABEL_ORDER:
        label_mask = truth_mask(truth_codes, label) & valid
        counts_by_label[label] = {
            "n_jets": int(np.sum(label_mask)),
            "weight_sum": weighted_sum(weights[label_mask]),
            "signed_weight_sum": weighted_sum(signed_weights[label_mask]),
        }

    rows: list[dict[str, Any]] = []
    for eff in sig_effs:
        cut = weighted_quantile(signal_scores, signal_weights, 1.0 - eff)

        sig_mask = signal_mask & valid
        bkg_mask = background_mask & valid
        sig_pass = sig_mask & (scores >= cut)
        bkg_pass = bkg_mask & (scores >= cut)

        yield_sig_total = weighted_sum(weights[sig_mask])
        yield_bkg_total = weighted_sum(weights[bkg_mask])
        yield_sig_pass = weighted_sum(weights[sig_pass])
        yield_bkg_pass = weighted_sum(weights[bkg_pass])

        row: dict[str, Any] = {
            "target_sig_eff": eff,
            "score_cut": cut,
            "actual_sig_eff": float(yield_sig_pass / yield_sig_total) if yield_sig_total > 0 else float("nan"),
            "bkg_eff": float(yield_bkg_pass / yield_bkg_total) if yield_bkg_total > 0 else float("nan"),
            "n_sig_total": int(np.sum(sig_mask)),
            "n_bkg_total": int(np.sum(bkg_mask)),
            "n_sig_pass": int(np.sum(sig_pass)),
            "n_bkg_pass": int(np.sum(bkg_pass)),
            "yield_sig_total": yield_sig_total,
            "yield_bkg_total": yield_bkg_total,
            "yield_sig_pass": yield_sig_pass,
            "yield_bkg_pass": yield_bkg_pass,
            "yield_sig_pass_signed": weighted_sum(signed_weights[sig_pass]),
            "yield_bkg_pass_signed": weighted_sum(signed_weights[bkg_pass]),
            "purity": float(yield_sig_pass / (yield_sig_pass + yield_bkg_pass))
            if (yield_sig_pass + yield_bkg_pass) > 0
            else float("nan"),
            "s_over_b": float(yield_sig_pass / yield_bkg_pass) if yield_bkg_pass > 0 else float("inf"),
            "s_over_sqrt_b": float(yield_sig_pass / np.sqrt(yield_bkg_pass))
            if yield_bkg_pass > 0
            else float("inf"),
            "s_over_sqrt_s_plus_b": float(yield_sig_pass / np.sqrt(yield_sig_pass + yield_bkg_pass))
            if (yield_sig_pass + yield_bkg_pass) > 0
            else float("nan"),
        }

        for label in TRUTH_LABEL_ORDER:
            label_mask = truth_mask(truth_codes, label) & valid
            label_scores = scores[label_mask]
            label_weights = weights[label_mask]
            row[f"eff__{label}"] = weighted_efficiency(label_scores, label_weights, cut)
            row[f"yield_pass__{label}"] = weighted_sum(weights[label_mask & (scores >= cut)])
            row[f"yield_pass_signed__{label}"] = weighted_sum(signed_weights[label_mask & (scores >= cut)])

        for label in background_labels:
            row[f"n_pass__{label}"] = int(np.sum(truth_mask(truth_codes, label) & valid & (scores >= cut)))

        rows.append(row)
    return rows, counts_by_label


def compute_roc(
    scores: np.ndarray,
    truth_codes: np.ndarray,
    weights: np.ndarray,
    target: str,
) -> dict[str, np.ndarray | float]:
    signal_mask, background_mask, _ = build_target_masks(truth_codes, target)
    valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0) & (signal_mask | background_mask)

    roc_scores = scores[valid]
    roc_labels = signal_mask[valid].astype(int)
    roc_weights = weights[valid]
    if roc_scores.size == 0:
        raise ValueError(f"No finite scores available for ROC computation ({target}).")

    order = np.argsort(roc_scores)[::-1]
    roc_scores = roc_scores[order]
    roc_labels = roc_labels[order]
    roc_weights = roc_weights[order]

    total_signal = weighted_sum(roc_weights[roc_labels == 1])
    total_background = weighted_sum(roc_weights[roc_labels == 0])
    if total_signal <= 0 or total_background <= 0:
        raise ValueError(f"ROC requires both signal and background for target '{target}'.")

    tp = np.cumsum(np.where(roc_labels == 1, roc_weights, 0.0))
    fp = np.cumsum(np.where(roc_labels == 0, roc_weights, 0.0))
    distinct = np.r_[True, roc_scores[1:] != roc_scores[:-1]]
    sig_eff = tp[distinct] / total_signal
    bkg_eff = fp[distinct] / total_background
    thresholds = roc_scores[distinct]

    sig_eff = np.r_[0.0, sig_eff, 1.0]
    bkg_eff = np.r_[0.0, bkg_eff, 1.0]
    thresholds = np.r_[1.0 + 1e-6, thresholds, 0.0]
    auc = float(np.trapz(sig_eff, bkg_eff))
    return {"sig_eff": sig_eff, "bkg_eff": bkg_eff, "thresholds": thresholds, "auc": auc}


def pass_from_hist(hist: np.ndarray) -> np.ndarray:
    return np.flip(np.cumsum(np.flip(hist, axis=-1), axis=-1), axis=-1)


def working_point_bin_index(signal_pass_hist: np.ndarray, total_signal: float, target_eff: float) -> int:
    if total_signal <= 0:
        raise ValueError("Signal total weight must be positive for a working point calculation.")
    signal_eff = signal_pass_hist / total_signal
    eligible = np.flatnonzero(signal_eff >= target_eff)
    if eligible.size == 0:
        return 0
    return int(eligible[-1])


def compute_working_points_from_hist(
    score_payload: dict[str, np.ndarray],
    hist_edges: np.ndarray,
    target: str,
    sig_effs: list[float],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    count_truth = np.asarray(score_payload["count_truth"], dtype=np.int64)
    weight_truth = np.asarray(score_payload["weight_truth"], dtype=np.float64)
    signed_truth = np.asarray(score_payload["signed_truth"], dtype=np.float64)

    signal_codes, background_codes, background_labels = (
        codes_for_labels(TARGET_DEFINITIONS[target]["signal_labels"]),
        codes_for_labels(TARGET_DEFINITIONS[target]["background_labels"]),
        TARGET_DEFINITIONS[target]["background_labels"],
    )
    count_total_truth = np.sum(count_truth, axis=1)
    weight_total_truth = np.sum(weight_truth, axis=1)
    signed_total_truth = np.sum(signed_truth, axis=1)

    signal_weight_hist = np.sum(weight_truth[signal_codes], axis=0)
    background_weight_hist = np.sum(weight_truth[background_codes], axis=0)
    if np.sum(signal_weight_hist) <= 0 or np.sum(background_weight_hist) <= 0:
        raise ValueError(f"No usable signal/background jets found for target '{target}'.")

    count_pass_truth = pass_from_hist(count_truth)
    weight_pass_truth = pass_from_hist(weight_truth)
    signed_pass_truth = pass_from_hist(signed_truth)
    signal_weight_pass = np.sum(weight_pass_truth[signal_codes], axis=0)
    background_weight_pass = np.sum(weight_pass_truth[background_codes], axis=0)
    signal_count_pass = np.sum(count_pass_truth[signal_codes], axis=0)
    background_count_pass = np.sum(count_pass_truth[background_codes], axis=0)

    counts_by_label: dict[str, dict[str, float]] = {}
    for label in TRUTH_LABEL_ORDER:
        code = TRUTH_LABEL_TO_CODE[label]
        counts_by_label[label] = {
            "n_jets": int(count_total_truth[code]),
            "weight_sum": float(weight_total_truth[code]),
            "signed_weight_sum": float(signed_total_truth[code]),
        }

    total_signal = float(np.sum(signal_weight_hist))
    total_background = float(np.sum(background_weight_hist))
    rows: list[dict[str, Any]] = []
    for eff in sig_effs:
        bin_index = working_point_bin_index(signal_weight_pass, total_signal, eff)
        cut = float(hist_edges[bin_index])
        yield_sig_pass = float(signal_weight_pass[bin_index])
        yield_bkg_pass = float(background_weight_pass[bin_index])
        row: dict[str, Any] = {
            "target_sig_eff": eff,
            "score_cut": cut,
            "actual_sig_eff": float(yield_sig_pass / total_signal) if total_signal > 0 else float("nan"),
            "bkg_eff": float(yield_bkg_pass / total_background) if total_background > 0 else float("nan"),
            "n_sig_total": int(np.sum(count_total_truth[signal_codes])),
            "n_bkg_total": int(np.sum(count_total_truth[background_codes])),
            "n_sig_pass": int(signal_count_pass[bin_index]),
            "n_bkg_pass": int(background_count_pass[bin_index]),
            "yield_sig_total": total_signal,
            "yield_bkg_total": total_background,
            "yield_sig_pass": yield_sig_pass,
            "yield_bkg_pass": yield_bkg_pass,
            "yield_sig_pass_signed": float(np.sum(signed_pass_truth[signal_codes, bin_index])),
            "yield_bkg_pass_signed": float(np.sum(signed_pass_truth[background_codes, bin_index])),
            "purity": float(yield_sig_pass / (yield_sig_pass + yield_bkg_pass))
            if (yield_sig_pass + yield_bkg_pass) > 0
            else float("nan"),
            "s_over_b": float(yield_sig_pass / yield_bkg_pass) if yield_bkg_pass > 0 else float("inf"),
            "s_over_sqrt_b": float(yield_sig_pass / np.sqrt(yield_bkg_pass))
            if yield_bkg_pass > 0
            else float("inf"),
            "s_over_sqrt_s_plus_b": float(yield_sig_pass / np.sqrt(yield_sig_pass + yield_bkg_pass))
            if (yield_sig_pass + yield_bkg_pass) > 0
            else float("nan"),
        }
        for label in TRUTH_LABEL_ORDER:
            code = TRUTH_LABEL_TO_CODE[label]
            total_weight = float(weight_total_truth[code])
            pass_weight = float(weight_pass_truth[code, bin_index])
            row[f"eff__{label}"] = float(pass_weight / total_weight) if total_weight > 0 else float("nan")
            row[f"yield_pass__{label}"] = pass_weight
            row[f"yield_pass_signed__{label}"] = float(signed_pass_truth[code, bin_index])
        for label in background_labels:
            code = TRUTH_LABEL_TO_CODE[label]
            row[f"n_pass__{label}"] = int(count_pass_truth[code, bin_index])
        rows.append(row)
    return rows, counts_by_label


def compute_roc_from_hist(
    score_payload: dict[str, np.ndarray],
    hist_edges: np.ndarray,
    target: str,
) -> dict[str, np.ndarray | float]:
    weight_truth = np.asarray(score_payload["weight_truth"], dtype=np.float64)
    signal_codes = codes_for_labels(TARGET_DEFINITIONS[target]["signal_labels"])
    background_codes = codes_for_labels(TARGET_DEFINITIONS[target]["background_labels"])
    signal_hist = np.sum(weight_truth[signal_codes], axis=0)
    background_hist = np.sum(weight_truth[background_codes], axis=0)
    total_signal = float(np.sum(signal_hist))
    total_background = float(np.sum(background_hist))
    if total_signal <= 0 or total_background <= 0:
        raise ValueError(f"ROC requires both signal and background for target '{target}'.")

    sig_eff = np.r_[0.0, np.cumsum(signal_hist[::-1]) / total_signal, 1.0]
    bkg_eff = np.r_[0.0, np.cumsum(background_hist[::-1]) / total_background, 1.0]
    thresholds = np.r_[1.0 + 1e-6, hist_edges[-2::-1], 0.0]
    auc = float(np.trapz(sig_eff, bkg_eff))
    return {"sig_eff": sig_eff, "bkg_eff": bkg_eff, "thresholds": thresholds, "auc": auc}
