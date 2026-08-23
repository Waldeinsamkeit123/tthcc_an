from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tthcc_an.nn_study.config import NnStudyConfig
from tthcc_an.nn_study.dataset import NnDataset
from tthcc_an.nn_study.metrics import (
    compute_confusion_matrix,
    compute_pairwise_rocs,
    normalize_confusion,
)
from tthcc_an.nn_study.plotting import (
    build_auc_matrix,
    plot_auc_matrix,
    plot_confusion_matrix,
)
from tthcc_an.nn_study.reporting import write_json, write_text


WEAVER_REFERENCE = {
    "repository": "https://github.com/hqucms/weaver-core",
    "commit_near_model_deployment": "a57c362bb42891e7d7dc997fd802b4bd15d978c8",
    "base_weight_expression": "weaver/utils/data/config.py:127-142",
    "weight_builder": "weaver/utils/data/preprocess.py:33-72,215-310",
    "sampling": "weaver/utils/dataset.py:101-114,127-181",
    "loss_call": "weaver/utils/nn/tools.py:121-181",
    "default_loss": "weaver/train.py:775-783",
}


def _fraction(value: float, total: float) -> float | None:
    return float(value / total) if total > 0.0 else None


def _quantile_summary(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "quantiles": {},
        }
    levels = [0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0]
    quantiles = np.quantile(finite, levels)
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "quantiles": {
            f"{level:g}": float(value)
            for level, value in zip(levels, quantiles)
        },
    }


def _score_sum_summary(config: NnStudyConfig, dataset: NnDataset) -> dict[str, Any]:
    names = config.score_names
    matrix = dataset.score_matrix(names).astype(np.float64, copy=False)
    finite_rows = np.all(np.isfinite(matrix), axis=1)
    sums = np.sum(matrix[finite_rows], axis=1)
    tolerance = 1e-5
    summary = _quantile_summary(sums)
    summary.update(
        {
            "classes": names,
            "branches": {
                score.name: score.branch for score in config.scores
            },
            "scope": "all persisted multiclass score branches used by argmax",
            "events": int(matrix.shape[0]),
            "finite_events": int(np.sum(finite_rows)),
            "nonfinite_events": int(np.sum(~finite_rows)),
            "negative_score_values": int(np.sum(np.isfinite(matrix) & (matrix < 0.0))),
            "zero_score_values": int(np.sum(np.isfinite(matrix) & (matrix == 0.0))),
            "sum_to_one_tolerance": tolerance,
            "events_outside_sum_to_one_tolerance": int(
                np.sum(np.abs(sums - 1.0) > tolerance)
            ),
            "all_finite_events_sum_to_one": bool(
                sums.size > 0 and np.all(np.abs(sums - 1.0) <= tolerance)
            ),
            "model_output_classes": config.weighting_diagnostics.model_output_classes,
            "persisted_score_classes": config.weighting_diagnostics.persisted_score_classes,
            "persisted_scores_cover_all_model_outputs": (
                set(config.weighting_diagnostics.model_output_classes)
                == set(config.weighting_diagnostics.persisted_score_classes)
            ),
            "pepper_output_semantics": (
                "ONNX output has a final Softmax. Pepper maps returned columns to score_* "
                "branches directly; the configured NNeval ntuple is diagnosed using only "
                "the persisted branches available here."
            ),
        }
    )
    return summary


def _reweight_variable_summary(
    config: NnStudyConfig, dataset: NnDataset
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, edges in config.weighting_diagnostics.reweight_variables.items():
        values = np.asarray(dataset.analysis_columns[name], dtype=np.float64)
        finite = np.isfinite(values)
        inside = finite & (values >= edges[0]) & (values <= edges[-1])
        details = _quantile_summary(values)
        details.update(
            {
                "bin_edges": edges,
                "number_of_bins": len(edges) - 1,
                "finite_events": int(np.sum(finite)),
                "events_inside_configured_range": int(np.sum(inside)),
                "events_outside_configured_range": int(np.sum(finite & ~inside)),
            }
        )
        output[name] = details
    return output


def _class_balanced_analysis_weights(
    truth_index: np.ndarray,
    analysis_weight: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    weights = np.zeros_like(analysis_weight, dtype=np.float64)
    for index in range(n_classes):
        mask = (
            (truth_index == index)
            & np.isfinite(analysis_weight)
            & (analysis_weight > 0.0)
        )
        total = float(np.sum(analysis_weight[mask]))
        if total > 0.0:
            weights[mask] = analysis_weight[mask] / total
    return weights


def _class_weight_comparison(
    config: NnStudyConfig, dataset: NnDataset
) -> dict[str, Any]:
    valid_truth = (dataset.truth_index >= 0) & (
        dataset.truth_index < len(config.truth_categories)
    )
    abs_raw_weight = np.abs(np.asarray(dataset.raw_weight, dtype=np.float64))
    analysis_weight = np.asarray(dataset.analysis_weight, dtype=np.float64)
    totals = {
        "event_count": float(np.sum(valid_truth)),
        "abs_raw_weight": float(np.sum(abs_raw_weight[valid_truth])),
        "analysis_weight": float(np.sum(analysis_weight[valid_truth])),
    }
    classes: dict[str, Any] = {}
    for index, truth in enumerate(config.truth_categories):
        mask = dataset.truth_index == index
        count = int(np.sum(mask))
        abs_sum = float(np.sum(abs_raw_weight[mask]))
        analysis_sum = float(np.sum(analysis_weight[mask]))
        classes[truth.name] = {
            "event_count": count,
            "event_count_fraction": _fraction(float(count), totals["event_count"]),
            "abs_raw_weight_sum": abs_sum,
            "abs_raw_weight_fraction": _fraction(abs_sum, totals["abs_raw_weight"]),
            "analysis_weight_sum": analysis_sum,
            "analysis_weight_fraction": _fraction(
                analysis_sum, totals["analysis_weight"]
            ),
            "configured_class_weight_under_study": config.weighting_diagnostics.class_weights[
                truth.name
            ],
            "training_effective_weight_sum": None,
            "training_effective_weight_fraction": None,
        }
    return {
        "normalization_scope": "classified events in the loaded NNeval ntuples",
        "totals": totals,
        "classes": classes,
        "training_effective_weight_available": False,
        "training_effective_weight_note": (
            "Not replaced by an approximation; exact Weaver training sampling "
            "weights and realized sample multiplicities are unavailable."
        ),
    }


def _mode_class_summary(
    config: NnStudyConfig,
    dataset: NnDataset,
    weights: np.ndarray,
    normalized_matrix: np.ndarray,
) -> dict[str, Any]:
    finite_scores = np.all(
        np.isfinite(dataset.score_matrix(config.score_names)), axis=1
    )
    output: dict[str, Any] = {}
    for index, truth in enumerate(config.truth_categories):
        mask = (dataset.truth_index == index) & finite_scores
        row = normalized_matrix[index]
        off_diagonal = row.copy()
        off_diagonal[index] = -np.inf
        dominant_index = int(np.argmax(off_diagonal))
        has_weight = float(np.sum(weights[mask])) > 0.0
        output[truth.name] = {
            "event_count": int(np.sum(mask)),
            "total_effective_weight": float(np.sum(weights[mask])),
            "argmax_correct_fraction": float(row[index]) if has_weight else None,
            "dominant_misclassification": (
                config.truth_categories[dominant_index].name if has_weight else None
            ),
            "dominant_misclassification_fraction": (
                float(row[dominant_index]) if has_weight else None
            ),
        }
    return output


def _auc_validation(pairwise: dict[str, dict[str, Any]]) -> dict[str, Any]:
    differences = [
        abs(result.auc - result.sklearn_auc)
        for backgrounds in pairwise.values()
        for result in backgrounds.values()
        if result.sklearn_auc is not None
    ]
    return {
        "compared_pairs": len(differences),
        "max_abs_difference": float(max(differences)) if differences else None,
    }


def _key_auc_summary(
    config: NnStudyConfig,
    pairwise_by_mode: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for signal, background in config.weighting_diagnostics.key_auc_pairs:
        pair_name = f"{signal}-vs-{background}"
        output[pair_name] = {}
        for mode, pairwise in pairwise_by_mode.items():
            result = pairwise.get(signal, {}).get(background)
            output[pair_name][mode] = float(result.auc) if result is not None else None
        output[pair_name]["training_weighted"] = None
    return output


def _training_weight_provenance(config: NnStudyConfig) -> dict[str, Any]:
    return {
        "available": False,
        "training_weighted_outputs_written": False,
        "implementation_scope": (
            "The standard Weaver implementation at the cited commit was inspected. "
            "The model-specific NanoTTH training checkout, command, network config, "
            "and auto-generated YAML were not found locally."
        ),
        "declared_configuration_under_study": {
            "use_precomputed_weights": False,
            "reweight_basewgt": "np.abs(weight)",
            "reweight_method": "flat",
            "reweight_threshold": 1,
            "reweight_variables": config.weighting_diagnostics.reweight_variables,
            "reweight_classes": config.score_names,
            "class_weights": config.weighting_diagnostics.class_weights,
        },
        "weaver_reference": WEAVER_REFERENCE,
        "general_algorithm": [
            "Build a base-weighted 2D histogram H[c,b] for each class and reweight-variable bin.",
            "For flat reweighting use f[c,b] = clip(P_threshold(H[c,b]>0) / H[c,b], 0, 1).",
            "Compute E[c] = sum_b H[c,b] * f[c,b] / class_weight[c].",
            "Set M = 0.9 * min_c E[c] and histogram sampling factor h[c,b] = f[c,b] * M / E[c].",
            "When reweight_basewgt is enabled, divide h by the global (100-threshold) percentile of h[c,b(i)] * abs(weight_i).",
            "The DataLoader accepts repeated event indices stochastically; the event weight is not passed to the default CrossEntropyLoss.",
        ],
        "one_bin_reduction": (
            "For one populated bin per variable, f[c]=1 and the deterministic sampling "
            "score is p_i = (class_weight[c] * abs(weight_i) / H[c]) / "
            "P99_training(class_weight[class(i)] * abs(weight_i) / H[class(i)]), "
            "where H[c] is the training-set sum of abs(weight) for class c."
        ),
        "effective_training_contribution": (
            "A fetched event contributes an unweighted CrossEntropyLoss term once per "
            "accepted copy. With default weight_scale=1, each copy is accepted with "
            "probability min(p_i, 1); the per-fetch repeat count is stochastic and capped "
            "at max_resample=10. CrossEntropyLoss then averages accepted terms within each batch."
        ),
        "missing_for_exact_reconstruction": [
            "the model-specific NanoTTH network config and any custom loss/train function",
            "the exact Weaver revision and training command recorded for these ONNX files",
            "the model-specific Weaver/NanoTTH auto YAML containing reweight_hists",
            "the exact training event population and per-class base-weight sums H[c]",
            "the global training sampling-weight percentile used for scaling",
            "the DataLoader fetch partition, sampler options, random seeds, and realized repeated indices",
            "a model artifact that unambiguously links these ONNX files to the supplied training YAML",
        ],
        "why_ntuple_is_insufficient": (
            "The NNeval ROOT files are a physically selected evaluation population, not "
            "the training population used to build H[c]. They contain neither the saved "
            "reweight histograms nor the realized stochastic sampling multiplicities."
        ),
    }


def _format_text(summary: dict[str, Any]) -> str:
    lines = [
        "=== NN Weighting Diagnostics ===",
        f"Channel: {summary['channel']}",
        "Exact training-effective weights: unavailable",
        "Reason: model-specific auto reweight histograms and realized training sampling are missing.",
        "",
        "Score sum:",
        f"  classes: {', '.join(summary['score_sum']['classes'])}",
        f"  finite events: {summary['score_sum']['finite_events']}",
        f"  min / mean / median / max: {summary['score_sum']['min']} / "
        f"{summary['score_sum']['mean']} / {summary['score_sum']['median']} / "
        f"{summary['score_sum']['max']}",
        f"  outside |sum-1| <= 1e-5: "
        f"{summary['score_sum']['events_outside_sum_to_one_tolerance']}",
        "",
        "Per-class argmax diagnostics:",
    ]
    for name, modes in summary["classes"].items():
        lines.append(f"  {name}:")
        for mode, values in modes.items():
            lines.append(
                f"    {mode}: diagonal={values['argmax_correct_fraction']}  "
                f"dominant={values['dominant_misclassification']} "
                f"({values['dominant_misclassification_fraction']})"
            )
    lines.extend(["", "Key pairwise AUC:"])
    for pair, modes in summary["key_auc"].items():
        lines.append(
            f"  {pair}: analysis={modes['analysis_weighted']}  "
            f"unweighted={modes['unweighted']}  training=None"
        )
    return "\n".join(lines) + "\n"


def run_weighting_diagnostics(
    *,
    config: NnStudyConfig,
    dataset: NnDataset,
    plot_dir: Path,
    summary_dir: Path,
    plot_suffix: str,
) -> tuple[list[str], dict[str, Any]]:
    names = config.score_names
    score_matrix = dataset.score_matrix(names)
    n_classes = len(names)
    modes = {
        "analysis_weighted": np.asarray(dataset.analysis_weight, dtype=np.float64),
        "unweighted": np.ones(dataset.truth_index.shape, dtype=np.float64),
        "class_balanced": _class_balanced_analysis_weights(
            dataset.truth_index, dataset.analysis_weight, n_classes
        ),
    }

    plots: list[str] = []
    normalized_confusions: dict[str, np.ndarray] = {}
    confusion_events: dict[str, int] = {}
    titles = {
        "analysis_weighted": "Argmax performance: analysis weighted",
        "unweighted": "Argmax performance: unweighted events",
        "class_balanced": "Argmax performance: class-balanced analysis weight",
    }
    for mode, weights in modes.items():
        matrix, event_count = compute_confusion_matrix(
            score_matrix, dataset.truth_index, weights, n_classes
        )
        normalized = normalize_confusion(matrix, "truth")
        normalized_confusions[mode] = normalized
        confusion_events[mode] = event_count
        path = plot_dir / f"confusion_truth__{mode}{plot_suffix}"
        plot_confusion_matrix(
            outpath=path,
            matrix=normalized,
            truths=config.truth_categories,
            scores=config.scores,
            normalization="truth",
            config=config,
            title=titles[mode] + "\nP(predicted class | truth class)",
        )
        plots.append(str(path))

        if mode == "analysis_weighted":
            pred_path = plot_dir / f"confusion_pred__analysis_weighted{plot_suffix}"
            plot_confusion_matrix(
                outpath=pred_path,
                matrix=normalize_confusion(matrix, "prediction"),
                truths=config.truth_categories,
                scores=config.scores,
                normalization="prediction",
                config=config,
                title=(
                    "Predicted-category composition: analysis weighted\n"
                    "P(truth class | predicted class)"
                ),
            )
            plots.append(str(pred_path))

    pairwise_by_mode: dict[str, dict[str, dict[str, Any]]] = {}
    auc_validation: dict[str, Any] = {}
    for mode in ("analysis_weighted", "unweighted"):
        pairwise = compute_pairwise_rocs(
            names,
            dataset.scores,
            dataset.truth_index,
            modes[mode],
            validate_with_sklearn=config.validate_auc_with_sklearn,
        )
        pairwise_by_mode[mode] = pairwise
        auc_validation[mode] = _auc_validation(pairwise)
        path = plot_dir / f"pairwise_auc_matrix__{mode}{plot_suffix}"
        plot_auc_matrix(
            outpath=path,
            auc_matrix=build_auc_matrix(names, pairwise),
            scores=config.scores,
            config=config,
            title=f"Pairwise AUC: {mode.replace('_', ' ')}",
        )
        plots.append(str(path))

    per_mode_classes = {
        mode: _mode_class_summary(
            config, dataset, weights, normalized_confusions[mode]
        )
        for mode, weights in modes.items()
    }
    classes = {
        name: {
            mode: per_mode_classes[mode][name]
            for mode in modes
        }
        for name in names
    }
    summary = {
        "channel": config.channel,
        "argmax_classes": names,
        "argmax_definition": "argmax over all persisted multiclass score branches listed here",
        "weighting_modes": {
            "analysis_weighted": "sample_norm * abs(weight)",
            "unweighted": "one per event",
            "class_balanced": (
                "analysis_weight / sum(analysis_weight) within each truth class; "
                "each non-empty truth class has total weight one"
            ),
            "training_weighted": "not produced because exact weights are unavailable",
        },
        "class_balanced_truth_matrix_note": (
            "Truth-row normalization cancels any class-wide scale, so this matrix is "
            "expected to match the analysis-weighted truth matrix up to floating precision."
        ),
        "confusion_events": confusion_events,
        "confusion_truth_matrices": {
            mode: matrix.tolist()
            for mode, matrix in normalized_confusions.items()
        },
        "classes": classes,
        "pairwise_auc": {
            mode: {
                signal: {
                    background: float(result.auc)
                    for background, result in backgrounds.items()
                }
                for signal, backgrounds in pairwise.items()
            }
            for mode, pairwise in pairwise_by_mode.items()
        },
        "key_auc": _key_auc_summary(config, pairwise_by_mode),
        "auc_validation": auc_validation,
        "score_sum": _score_sum_summary(config, dataset),
        "reweight_variables": _reweight_variable_summary(config, dataset),
        "training_effective_weight": _training_weight_provenance(config),
        "plots": plots,
    }
    comparison = _class_weight_comparison(config, dataset)
    write_json(summary_dir / "weighting_diagnostics.json", summary)
    write_text(summary_dir / "weighting_diagnostics.txt", _format_text(summary))
    write_json(summary_dir / "class_weight_comparison.json", comparison)
    comparison_lines = [
        "=== Class Weight Comparison ===",
        "Fractions are normalized over classified events in the loaded NNeval ntuples.",
        "Exact training-effective contributions are unavailable and are not approximated.",
        "",
    ]
    for name, values in comparison["classes"].items():
        comparison_lines.append(
            f"{name}: count={values['event_count']} "
            f"({values['event_count_fraction']}), abs(weight)={values['abs_raw_weight_sum']} "
            f"({values['abs_raw_weight_fraction']}), analysis={values['analysis_weight_sum']} "
            f"({values['analysis_weight_fraction']}), training=None"
        )
    write_text(
        summary_dir / "class_weight_comparison.txt",
        "\n".join(comparison_lines) + "\n",
    )
    return plots, summary
