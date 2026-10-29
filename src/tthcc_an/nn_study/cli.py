from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from tthcc_an.nn_study.analysis_regions import (
    _ar_histogram_edges_and_coverage,
    build_ar_mask,
    run_ar_score_distributions,
    run_conditional_qcd_veto,
)
from tthcc_an.nn_study.cache import (
    cache_analysis_branches,
    cache_metadata_path,
    cache_score_names,
    load_nn_cache,
    resolve_cache_path,
    validate_nn_cache,
    write_nn_cache,
)
from tthcc_an.nn_study.config import NnTruthCategory, load_nn_study_config
from tthcc_an.nn_study.dataset import NnDataset, load_nn_dataset
from tthcc_an.nn_study.d_tthz_study import run_d_tthz_study
from tthcc_an.nn_study.mass_sculpting import run_mass_sculpting
from tthcc_an.nn_study.metrics import compute_confusion_matrix, compute_pairwise_rocs, normalize_confusion
from tthcc_an.nn_study.plotting import (
    build_auc_matrix,
    plot_auc_matrix,
    plot_confusion_matrix,
    plot_pairwise_roc,
    plot_score_distribution,
)
from tthcc_an.nn_study.qcd_score_scan import run_qcd_score_scan
from tthcc_an.nn_study.qcd_data_mc import run_qcd_data_mc
from tthcc_an.nn_study.regressed_mass_fine_binning import (
    prepare_data_cache,
    run_regressed_mass_fine_binning,
)
from tthcc_an.nn_study.reporting import format_summary_text, write_json, write_text
from tthcc_an.nn_study.score_significance import run_score_significance
from tthcc_an.nn_study.score_significance_2d import run_score_significance_2d
from tthcc_an.nn_study.tthz_qcd_discriminant import run_tthz_qcd_discriminant
from tthcc_an.nn_study.weighting_diagnostics import run_weighting_diagnostics


REPO_ROOT = Path(__file__).resolve().parents[3]
PLOT_CHOICES = (
    "fine-truth-stacks",
    "scores",
    "confusion",
    "roc",
    "auc",
    "mass-sculpting",
    "score-significance",
    "score-significance-2d",
    "score-qcd-fine-truth-split",
    "ar-score-distributions",
    "qcd-veto-conditional",
    "qcd-score-scan",
    "qcd-data-mc",
    "regressed-mass-fine-binning",
    "regressed-mass-mcstat",
    "tthz-qcd-discriminant",
    "weighting-diagnostics",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Study event-level multiclass NN outputs in Pepper ROOT ntuples."
    )
    parser.add_argument("--config", required=True, help="Path to an NN-study JSON config.")
    parser.add_argument("--outdir", default=None, help="Optional output-directory override.")
    parser.add_argument(
        "--selection",
        default=None,
        help="Optional event-selection expression override; an empty config value keeps all ntuple events.",
    )
    parser.add_argument(
        "--max-files-per-sample",
        type=int,
        default=None,
        help="Limit processed Events files per sample for smoke tests.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=PLOT_CHOICES,
        default=None,
        help="Optional plot groups to produce; summaries are always written.",
    )
    parser.add_argument(
        "--mass-sculpting-score",
        nargs="+",
        default=None,
        help=(
            "Optional score name or branch filter for --only mass-sculpting; "
            "for example: score_qcd."
        ),
    )
    parser.add_argument("--data-blinding", choices=("sidebands", "unblinded"), default=None,
                        help="qcd-data-mc only; defaults to sidebands.")
    parser.add_argument("--allow-unblinded-data", action="store_true",
                        help="Explicitly confirm collaboration authorization for private unblinded qcd-data-mc outputs.")
    parser.add_argument("--qcd-data-mc-selection", choices=("baseline", "tthz", "both"), default=None)
    parser.add_argument("--refresh-qcd-data-cache", action="store_true")
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--prepare-cache",
        action="store_true",
        help="Read ROOT once and write the reusable event-level NN-study cache.",
    )
    cache_mode.add_argument(
        "--prepare-data-cache",
        action="store_true",
        help="Read only real-data score/mass branches into a blinded sideband cache.",
    )
    cache_mode.add_argument(
        "--from-cache",
        action="store_true",
        help="Load the prepared event-level cache without discovering or opening ROOT files.",
    )
    parser.add_argument(
        "--cache-path",
        default=None,
        help=(
            "Optional prepared NPZ path; defaults to "
            "<study.outdir>/cache/prepared_events.npz."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing cache; valid with --prepare-cache or --prepare-data-cache.",
    )
    return parser


def _truth_yields(
    truths: list[Any], truth_index: np.ndarray, dataset: NnDataset
) -> dict[str, dict[str, Any]]:
    yields: dict[str, dict[str, Any]] = {}
    for index, truth in enumerate(truths):
        mask = truth_index == index
        yields[truth.name] = {
            "label": truth.label,
            "events": int(np.sum(mask)),
            "analysis_weight_sum": float(np.sum(dataset.analysis_weight[mask])),
            "signed_weight_sum": float(np.sum(dataset.signed_weight[mask])),
        }
    return yields


def _score_sum_diagnostics(config: Any, dataset: NnDataset) -> dict[str, Any] | None:
    if not all(name in dataset.scores for name in config.score_names):
        return None
    matrix = dataset.score_matrix(config.score_names)
    finite_rows = np.all(np.isfinite(matrix), axis=1)
    sums = np.sum(matrix[finite_rows], axis=1)
    if sums.size == 0:
        return {
            "classes": config.score_names,
            "finite_rows": 0,
            "nonfinite_rows": int(matrix.shape[0]),
            "sum_min": None,
            "sum_max": None,
            "max_abs_deviation_from_one": None,
        }
    return {
        "classes": config.score_names,
        "finite_rows": int(np.sum(finite_rows)),
        "nonfinite_rows": int(np.sum(~finite_rows)),
        "sum_min": float(np.min(sums)),
        "sum_max": float(np.max(sums)),
        "max_abs_deviation_from_one": float(np.max(np.abs(sums - 1.0))),
    }


def _remap_truth_index(
    truth_index: np.ndarray, source_names: list[str], selected_names: list[str]
) -> np.ndarray:
    remapped = np.full(truth_index.shape, -1, dtype=np.int16)
    source_index = {name: index for index, name in enumerate(source_names)}
    for target_index, name in enumerate(selected_names):
        remapped[truth_index == source_index[name]] = target_index
    return remapped


def _qcd_split_fine_truth(
    config: Any, dataset: NnDataset
) -> tuple[list[NnTruthCategory], np.ndarray]:
    required_masks = {"qcdraw", "wcq", "wqq", "zbb", "zcc", "zqq"}
    missing_masks = sorted(required_masks - set(dataset.diagnostic_truth_masks))
    if missing_masks:
        raise ValueError(
            "QCD fine-truth split requires diagnostic masks: "
            + ", ".join(missing_masks)
        )
    if "qcd" not in config.truth_names:
        raise ValueError("QCD fine-truth split requires the merged 'qcd' truth.")

    output_truths = [
        truth for truth in config.truth_categories if truth.name != "qcd"
    ]
    output_index = np.full(dataset.truth_index.shape, -1, dtype=np.int16)
    old_index = {name: index for index, name in enumerate(config.truth_names)}
    for new_index, truth in enumerate(output_truths):
        output_index[dataset.truth_index == old_index[truth.name]] = new_index

    split_definitions = [
        (
            NnTruthCategory("qcdraw", "QCD", "qcdraw", "black"),
            dataset.diagnostic_truth_masks["qcdraw"],
        ),
        (
            NnTruthCategory(
                "w_hadronic", "W hadronic", "wcq | wqq", "#7F3C8D"
            ),
            dataset.diagnostic_truth_masks["wcq"]
            | dataset.diagnostic_truth_masks["wqq"],
        ),
        (
            NnTruthCategory(
                "z_hadronic",
                "Z hadronic",
                "zbb | zcc | zqq",
                "#11A579",
            ),
            dataset.diagnostic_truth_masks["zbb"]
            | dataset.diagnostic_truth_masks["zcc"]
            | dataset.diagnostic_truth_masks["zqq"],
        ),
    ]
    merged_qcd = dataset.truth_index == old_index["qcd"]
    split_count = np.zeros(dataset.truth_index.shape, dtype=np.int8)
    split_union = np.zeros(dataset.truth_index.shape, dtype=bool)
    for _, mask in split_definitions:
        split_count += mask.astype(np.int8)
        split_union |= mask
    if np.any(split_count > 1) or not np.array_equal(split_union, merged_qcd):
        raise ValueError(
            "QCD/W/Z-hadronic diagnostic masks do not exclusively reproduce "
            "the merged qcd truth category."
        )

    for truth, mask in split_definitions:
        output_index[mask] = len(output_truths)
        output_truths.append(truth)
    return output_truths, output_index


def _matrix_payload(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _score_distribution_payload(config: Any, dataset: NnDataset) -> dict[str, Any]:
    n_bins = int(config.plot_options.get("score_bins", 20))
    score_range = [float(value) for value in config.plot_options.get("score_range", [0.0, 1.0])]
    bins = np.linspace(score_range[0], score_range[1], n_bins + 1)
    payload: dict[str, Any] = {"bin_edges": bins.tolist(), "scores": {}}
    for score in config.scores:
        values = np.asarray(dataset.scores[score.name], dtype=np.float64)
        categories: dict[str, Any] = {}
        for index, truth in enumerate(config.prediction_truth_categories):
            mask = (
                (dataset.prediction_truth_index == index)
                & np.isfinite(values)
                & np.isfinite(dataset.analysis_weight)
                & (dataset.analysis_weight > 0.0)
            )
            hist = np.histogram(
                values[mask], bins=bins, weights=dataset.analysis_weight[mask]
            )[0].astype(np.float64)
            total = float(np.sum(hist))
            shape = hist / total if total > 0.0 else np.zeros_like(hist)
            categories[truth.name] = {
                "weighted_yield_bins": hist.tolist(),
                "normalized_shape_bins": shape.tolist(),
                "normalized_shape_integral": (
                    float(np.sum(shape)) if total > 0.0 else None
                ),
            }
        payload["scores"][score.name] = categories
    return payload


def _coarse_union_validation(config: Any, dataset: NnDataset) -> dict[str, Any]:
    if not config.uses_distinct_prediction_truth:
        return {"applicable": False}
    components = {
        "ttHZ": ["ttHcc", "ttHbb", "ttZqq", "ttZcc", "ttZbb"],
        "ttjets": ["ttLF", "ttcj", "ttcc", "tt2c", "ttbj", "ttbb", "tt2b"],
    }
    fine_index = {name: index for index, name in enumerate(config.truth_names)}
    coarse_index = {
        name: index for index, name in enumerate(config.prediction_truth_names)
    }
    results: dict[str, bool] = {}
    for coarse_name, fine_names in components.items():
        if coarse_name not in coarse_index or not set(fine_names) <= set(fine_index):
            continue
        expected = np.isin(
            dataset.truth_index, [fine_index[name] for name in fine_names]
        )
        observed = dataset.prediction_truth_index == coarse_index[coarse_name]
        results[coarse_name] = bool(np.array_equal(expected, observed))
    if "qcd" in coarse_index and dataset.diagnostic_truth_masks:
        diagnostic_union = np.zeros(dataset.truth_index.shape, dtype=bool)
        diagnostic_count = np.zeros(dataset.truth_index.shape, dtype=np.int16)
        for mask in dataset.diagnostic_truth_masks.values():
            diagnostic_union |= mask
            diagnostic_count += mask.astype(np.int16)
        observed = dataset.prediction_truth_index == coarse_index["qcd"]
        results["qcd"] = bool(np.array_equal(diagnostic_union, observed))
        results["qcd_subcategories_exclusive"] = bool(
            np.all(diagnostic_count <= 1)
        )
    return {
        "applicable": True,
        "coarse_equals_configured_fine_union": results,
        "all_valid": bool(results) and all(results.values()),
    }


def _auc_validation(pairwise: dict[str, dict[str, Any]], requested: bool) -> dict[str, Any]:
    differences = [
        abs(result.auc - result.sklearn_auc)
        for backgrounds in pairwise.values()
        for result in backgrounds.values()
        if result.sklearn_auc is not None
    ]
    all_results = [result for backgrounds in pairwise.values() for result in backgrounds.values()]
    return {
        "requested": requested,
        "sklearn_available": any(result.sklearn_auc is not None for result in all_results),
        "compared_pairs": len(differences),
        "max_abs_difference": float(max(differences)) if differences else None,
    }


def _load_config(
    args: argparse.Namespace, *, discover_sample_files: bool
) -> Any:
    return load_nn_study_config(
        args.config,
        REPO_ROOT,
        outdir=args.outdir,
        selection=args.selection,
        max_files_per_sample=args.max_files_per_sample,
        discover_sample_files=discover_sample_files,
    )


def _prepare_cache(args: argparse.Namespace) -> dict[str, Any]:
    if args.only is not None:
        raise ValueError("--prepare-cache cannot be combined with --only.")
    if args.mass_sculpting_score is not None:
        raise ValueError(
            "--prepare-cache cannot be combined with --mass-sculpting-score."
        )
    config_without_files = _load_config(args, discover_sample_files=False)
    cache_path = resolve_cache_path(config_without_files, args.cache_path)
    if cache_path.exists() and not args.force:
        config_with_files = _load_config(args, discover_sample_files=True)
        metadata = validate_nn_cache(
            config=config_with_files,
            cache_path=cache_path,
            check_input_files=True,
        )
        return {
            "operation": "prepare-cache",
            "cache_reused": True,
            "cache_path": str(cache_path),
            "cache_metadata_path": str(cache_metadata_path(cache_path)),
            "number_of_events_selected": metadata["number_of_selected_events"],
            "number_of_files_discovered": metadata[
                "number_of_input_files_discovered"
            ],
            "cache_size_bytes": metadata["cache_size_bytes"],
        }

    config = _load_config(args, discover_sample_files=True)
    dataset = load_nn_dataset(
        config,
        score_names=cache_score_names(config),
        analysis_branches=cache_analysis_branches(config),
    )
    metadata = write_nn_cache(
        config=config,
        dataset=dataset,
        cache_path=cache_path,
    )
    return {
        "operation": "prepare-cache",
        "cache_reused": False,
        "cache_path": str(cache_path),
        "cache_metadata_path": str(cache_metadata_path(cache_path)),
        "number_of_events_selected": metadata["number_of_selected_events"],
        "number_of_files_discovered": metadata[
            "number_of_input_files_discovered"
        ],
        "cache_size_bytes": metadata["cache_size_bytes"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.only and "qcd-data-mc" in args.only:
        if args.only != ["qcd-data-mc"] or not args.from_cache:
            raise ValueError("qcd-data-mc requires --from-cache and must run as a separate --only mode.")
        if args.selection is not None or args.mass_sculpting_score is not None or args.force:
            raise ValueError("qcd-data-mc does not accept selection overrides or --force; use --refresh-qcd-data-cache.")
        config = _load_config(args, discover_sample_files=False)
        return run_qcd_data_mc(
            config=config, mc_cache_path=resolve_cache_path(config, args.cache_path),
            blinding=args.data_blinding or "sidebands",
            allow_unblinded=args.allow_unblinded_data,
            selection=args.qcd_data_mc_selection or "both",
            refresh_data_cache=args.refresh_qcd_data_cache,
        )
    if (args.data_blinding is not None or args.allow_unblinded_data
            or args.qcd_data_mc_selection is not None or args.refresh_qcd_data_cache):
        raise ValueError("Data-blinding options require --only qcd-data-mc.")
    if args.force and not (args.prepare_cache or args.prepare_data_cache):
        raise ValueError("--force requires --prepare-cache or --prepare-data-cache.")
    if args.prepare_cache:
        return _prepare_cache(args)
    if args.prepare_data_cache:
        if args.only is not None or args.mass_sculpting_score is not None or args.selection is not None:
            raise ValueError("--prepare-data-cache cannot use --only, --selection, or --mass-sculpting-score.")
        return prepare_data_cache(_load_config(args, discover_sample_files=False), force=args.force)
    if args.only and set(args.only) & {"regressed-mass-fine-binning", "regressed-mass-mcstat"}:
        if {"regressed-mass-fine-binning", "regressed-mass-mcstat"} <= set(args.only):
            raise ValueError("Choose either the data/MC or MC-only fine-binning mode.")
        if not args.from_cache:
            raise ValueError("Fine-binning studies require --from-cache; prepare real data separately for data/MC plots.")
        if args.selection is not None:
            raise ValueError("regressed-mass-fine-binning does not accept a selection override.")
    config = _load_config(args, discover_sample_files=not args.from_cache)
    if (args.only and "tthz-qcd-discriminant" in args.only
            and not config.tthz_qcd_discriminant_enabled):
        raise ValueError("tthz-qcd-discriminant is disabled in this config.")
    config.outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = config.outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = config.outdir / "summaries"
    plot_groups = set(args.only or ("scores", "confusion", "roc", "auc"))
    if args.only is None and config.mass_sculpting.enabled:
        plot_groups.add("mass-sculpting")
    if args.only is None and config.score_significance.enabled:
        plot_groups.add("score-significance")
    if args.only is None and config.score_significance_2d.enabled:
        plot_groups.add("score-significance-2d")
    if args.only is None and config.score_significance_2d_hierarchical.enabled:
        plot_groups.add("score-significance-2d")
    qcd_split_masks = {"qcdraw", "wcq", "wqq", "zbb", "zcc", "zqq"}
    if (
        args.only is None
        and "qcd" in config.all_score_names
        and qcd_split_masks.issubset(
            {truth.name for truth in config.diagnostic_truth_categories}
        )
    ):
        plot_groups.add("score-qcd-fine-truth-split")
    if args.only is None and config.qcd_score_scan.enabled:
        plot_groups.add("qcd-score-scan")
    if args.only is None and config.analysis_regions.enabled:
        plot_groups.add("ar-score-distributions")
    if (
        args.only is None
        and config.analysis_regions.conditional_qcd_veto.enabled
    ):
        plot_groups.add("qcd-veto-conditional")
    if args.only is None and config.weighting_diagnostics.enabled:
        plot_groups.add("weighting-diagnostics")
    if "mass-sculpting" in plot_groups and not config.mass_sculpting.enabled:
        raise ValueError("mass_sculpting is not enabled in this config.")
    mass_sculpting_score_names: set[str] | None = None
    if args.mass_sculpting_score is not None:
        if "mass-sculpting" not in plot_groups:
            raise ValueError(
                "--mass-sculpting-score requires the mass-sculpting plot group."
            )
        score_by_reference = {
            reference: scan.score_name
            for scan in (
                *config.mass_sculpting.scans,
                *config.mass_sculpting.summary_scans,
            )
            for reference in (scan.score_name, scan.score_branch)
        }
        unknown = sorted(set(args.mass_sculpting_score) - set(score_by_reference))
        if unknown:
            raise ValueError(
                "Unknown configured mass-sculpting scores: " + ", ".join(unknown)
            )
        mass_sculpting_score_names = {
            score_by_reference[reference]
            for reference in args.mass_sculpting_score
        }
    if (
        "score-significance" in plot_groups
        and not config.score_significance.enabled
    ):
        raise ValueError("score_significance is not enabled in this config.")
    if (
        "score-significance-2d" in plot_groups
        and not config.score_significance_2d.enabled
    ):
        raise ValueError("score_significance_2d is not enabled in this config.")
    if "qcd-score-scan" in plot_groups and not config.qcd_score_scan.enabled:
        raise ValueError("qcd_score_scan is not enabled in this config.")
    if (plot_groups & {"regressed-mass-fine-binning", "regressed-mass-mcstat"}
            and not config.regressed_mass_fine_binning.get("enabled")):
        raise ValueError("regressed_mass_fine_binning is not enabled in this config.")
    if (
        "ar-score-distributions" in plot_groups
        and not config.analysis_regions.enabled
    ):
        raise ValueError("analysis_regions is not enabled in this config.")
    if (
        "qcd-veto-conditional" in plot_groups
        and not config.analysis_regions.conditional_qcd_veto.enabled
    ):
        raise ValueError("conditional_qcd_veto is not enabled in this config.")
    if (
        "weighting-diagnostics" in plot_groups
        and not config.weighting_diagnostics.enabled
    ):
        raise ValueError("weighting_diagnostics is not enabled in this config.")

    loaded_score_names: list[str] = []
    if config.uses_distinct_prediction_truth:
        # Coarse score-sum validation is part of every downstream workflow.
        loaded_score_names.extend(config.score_names)
    if "scores" in plot_groups:
        loaded_score_names.extend(config.all_score_names)
    if plot_groups & {"confusion", "roc", "auc"}:
        loaded_score_names.extend(config.score_names)
    if "weighting-diagnostics" in plot_groups:
        loaded_score_names.extend(config.score_names)
    loaded_analysis_branches: list[str] = []
    if "mass-sculpting" in plot_groups:
        loaded_score_names.extend(
            scan.score_name
            for scan in config.mass_sculpting.scans
            if mass_sculpting_score_names is None
            or scan.score_name in mass_sculpting_score_names
        )
        loaded_score_names.extend(
            scan.score_name
            for scan in config.mass_sculpting.summary_scans
            if mass_sculpting_score_names is None
            or scan.score_name in mass_sculpting_score_names
        )
        loaded_analysis_branches.extend(
            variable.branch for variable in config.mass_sculpting.variables
        )
    if "score-significance" in plot_groups:
        loaded_score_names.extend(
            scan.score_name for scan in config.score_significance.scans
        )
    if "score-significance-2d" in plot_groups:
        for settings in (
            config.score_significance_2d,
            config.score_significance_2d_hierarchical,
        ):
            if not settings.enabled:
                continue
            if settings.x_axis is not None:
                loaded_score_names.append(settings.x_axis.score_name)
            if settings.y_axis is not None:
                loaded_score_names.append(settings.y_axis.score_name)
            if settings.third_cut.score_name is not None:
                loaded_score_names.append(settings.third_cut.score_name)
    if "score-qcd-fine-truth-split" in plot_groups:
        if "qcd" not in config.all_score_names:
            raise ValueError(
                "score-qcd-fine-truth-split requires a configured qcd score."
            )
        loaded_score_names.append("qcd")
    if "qcd-score-scan" in plot_groups:
        loaded_score_names.append(config.qcd_score_scan.score_name)
    if plot_groups & {"regressed-mass-fine-binning", "regressed-mass-mcstat"}:
        loaded_score_names.extend(("ttHZ", "qcd"))
        loaded_analysis_branches.append("TargetFatJet_regressed_mass_generic")
    if "tthz-qcd-discriminant" in plot_groups:
        loaded_score_names.extend(("ttHZ", "qcd"))
        loaded_analysis_branches.append("TargetFatJet_regressed_mass_generic")
    if "fine-truth-stacks" in plot_groups:
        loaded_score_names.extend(config.score_names)
        loaded_score_names.extend(config.analysis_discriminants)
    if "ar-score-distributions" in plot_groups:
        loaded_score_names.extend(config.score_names)
        loaded_score_names.extend(config.analysis_discriminants)
        loaded_score_names.extend(
            cut.score_name for cut in config.analysis_regions.ar_cuts
        )
    if "qcd-veto-conditional" in plot_groups:
        conditional = config.analysis_regions.conditional_qcd_veto
        if conditional.tthz_score_name is not None:
            loaded_score_names.append(conditional.tthz_score_name)
        if conditional.qcd_score_name is not None:
            loaded_score_names.append(conditional.qcd_score_name)
    if (
        config.significance_mass_window.enabled
        and plot_groups & {"score-significance", "qcd-score-scan"}
        and config.significance_mass_window.branch is not None
    ):
        loaded_analysis_branches.append(
            config.significance_mass_window.branch
        )
    if "weighting-diagnostics" in plot_groups:
        loaded_analysis_branches.extend(
            config.weighting_diagnostics.reweight_variables
        )
    if args.only is None:
        loaded_analysis_branches.extend(config.analysis_branches)
    loaded_score_names = list(dict.fromkeys(loaded_score_names))
    loaded_analysis_branches = list(dict.fromkeys(loaded_analysis_branches))
    cache_path: Path | None = None
    cache_metadata: dict[str, Any] | None = None
    if args.from_cache:
        cache_path = resolve_cache_path(config, args.cache_path)
        dataset, cache_metadata = load_nn_cache(
            config=config,
            cache_path=cache_path,
            score_names=loaded_score_names,
            analysis_branches=loaded_analysis_branches,
        )
    else:
        dataset = load_nn_dataset(
            config,
            score_names=loaded_score_names,
            analysis_branches=loaded_analysis_branches,
        )
    plot_format = str(config.plot_options.get("format", "png")).lower().lstrip(".")
    if plot_format not in {"png", "pdf"}:
        raise ValueError("plot.format must be either 'png' or 'pdf'.")
    plot_suffix = f".{plot_format}"
    needs_pairwise = bool(plot_groups & {"roc", "auc"})
    pairwise = (
        compute_pairwise_rocs(
            config.score_names,
            dataset.scores,
            dataset.prediction_truth_index,
            dataset.analysis_weight,
            validate_with_sklearn=config.validate_auc_with_sklearn,
        )
        if needs_pairwise
        else {}
    )
    pairwise_unweighted = (
        compute_pairwise_rocs(
            config.score_names,
            dataset.scores,
            dataset.prediction_truth_index,
            np.ones(dataset.prediction_truth_index.shape, dtype=np.float64),
            validate_with_sklearn=config.validate_auc_with_sklearn,
        )
        if needs_pairwise and config.uses_distinct_prediction_truth
        else {}
    )

    plots: list[str] = []
    if "score-qcd-fine-truth-split" in plot_groups:
        split_truths, split_truth_index = _qcd_split_fine_truth(config, dataset)
        score = next(score for score in config.all_scores if score.name == "qcd")
        path = plot_dir / "score_fine_truth" / (
            f"score_qcd__fine_truth__qcd_w_z_split{plot_suffix}"
        )
        plot_score_distribution(
            outpath=path,
            score=score,
            values=dataset.scores["qcd"],
            truth_index=split_truth_index,
            weights=dataset.analysis_weight,
            truths=split_truths,
            config=config,
            normalize=True,
            figsize=(7.8, 5.4),
            legend_columns=3,
            legend_fontsize=7.5,
        )
        plots.append(str(path))

    ar_score_composition_summary: dict[str, Any] | None = None
    if "ar-score-distributions" in plot_groups:
        ar_plots, ar_score_composition_summary = run_ar_score_distributions(
            config=config,
            dataset=dataset,
            plot_dir=plot_dir,
            summary_dir=summary_dir,
            plot_suffix=plot_suffix,
        )
        plots.extend(ar_plots)

    if "fine-truth-stacks" in plot_groups:
        if not config.uses_distinct_prediction_truth or not config.analysis_regions.enabled:
            raise ValueError("fine-truth-stacks requires fine truth and AR configuration.")
        score_by_name = {score.name: score for score in config.all_scores}
        plot_scores = [*config.scores]
        plot_scores.extend(score_by_name[name] for name in config.analysis_discriminants
                           if name not in config.score_names)
        score_range = config.plot_options.get("score_range", [0.0, 1.0])
        inclusive_bins = np.linspace(float(score_range[0]), float(score_range[1]),
                                     int(config.plot_options.get("score_bins", 20)) + 1)
        for region, mask in (
            ("", np.ones(dataset.truth_index.shape, dtype=bool)),
            ("__AR", build_ar_mask(config, dataset)),
        ):
            for score in plot_scores:
                if region:
                    plot_range = config.analysis_regions.ar_score_plot_ranges[score.name]
                    bins, _ = _ar_histogram_edges_and_coverage(
                        plot_range=plot_range,
                        values=dataset.scores[score.name][mask],
                        truth_index=dataset.truth_index[mask],
                        weights=dataset.analysis_weight[mask],
                        number_of_truths=len(config.truth_categories),
                    )
                    display_range = (plot_range.minimum, plot_range.maximum)
                    output_dir = plot_dir / "score_fine_truth_ar"
                else:
                    bins = inclusive_bins
                    display_range = None
                    output_dir = plot_dir / "score_fine_truth"
                path = output_dir / f"{score.branch}__fine_truth{region}__yield__stack{plot_suffix}"
                plot_score_distribution(
                    outpath=path,
                    score=score,
                    values=dataset.scores[score.name][mask],
                    truth_index=dataset.truth_index[mask],
                    weights=dataset.analysis_weight[mask],
                    truths=config.truth_categories,
                    config=config,
                    normalize=False,
                    stack=True,
                    reference_style=True,
                    selection_label=(config.plot_options.get("study_label", "0L V4")
                                     + (", AR" if region else ", inclusive MC")),
                    figsize=(6.67, 6.67),
                    bin_edges=bins,
                    display_range=display_range,
                )
                plots.append(str(path))

    qcd_veto_conditional_summary: dict[str, Any] | None = None
    if "qcd-veto-conditional" in plot_groups:
        conditional_plots, qcd_veto_conditional_summary = (
            run_conditional_qcd_veto(
                config=config,
                dataset=dataset,
                summary_dir=summary_dir,
            )
        )
        plots.extend(conditional_plots)

    if "scores" in plot_groups:
        for score in config.all_scores:
            shape_path = plot_dir / f"score_{score.name}{plot_suffix}"
            yield_path = plot_dir / f"score_{score.name}__yield{plot_suffix}"
            common = {
                "score": score,
                "values": dataset.scores[score.name],
                "truth_index": dataset.prediction_truth_index,
                "weights": dataset.analysis_weight,
                "truths": config.prediction_truth_categories,
                "config": config,
            }
            plot_score_distribution(outpath=shape_path, normalize=True, **common)
            plot_score_distribution(outpath=yield_path, normalize=False, **common)
            plots.extend([str(shape_path), str(yield_path)])

        if config.uses_distinct_prediction_truth:
            fine_truth_dir = plot_dir / "score_fine_truth"
            for score in config.scores:
                for normalize, suffix in [(True, ""), (False, "__yield")]:
                    path = fine_truth_dir / (
                        f"score_{score.name}__fine_truth{suffix}{plot_suffix}"
                    )
                    plot_score_distribution(
                        outpath=path,
                        score=score,
                        values=dataset.scores[score.name],
                        truth_index=dataset.truth_index,
                        weights=dataset.analysis_weight,
                        truths=config.truth_categories,
                        config=config,
                        normalize=normalize,
                        figsize=(7.2, 5.2),
                        legend_columns=3,
                        legend_fontsize=8.0,
                    )
                    plots.append(str(path))

        if config.uses_distinct_prediction_truth and "ttHZ" in dataset.scores:
            fine_components = ["ttHcc", "ttHbb", "ttZcc", "ttZbb", "ttZqq"]
            truth_by_name = {truth.name: truth for truth in config.truth_categories}
            component_truths = [truth_by_name[name] for name in fine_components]
            component_index = _remap_truth_index(
                dataset.truth_index, config.truth_names, fine_components
            )
            composition_dir = plot_dir / "score_composition"
            for normalize, suffix in [(True, ""), (False, "__yield")]:
                path = composition_dir / (
                    f"score_ttHZ__fine_ttHZ_components{suffix}{plot_suffix}"
                )
                plot_score_distribution(
                    outpath=path,
                    score=next(score for score in config.scores if score.name == "ttHZ"),
                    values=dataset.scores["ttHZ"],
                    truth_index=component_index,
                    weights=dataset.analysis_weight,
                    truths=component_truths,
                    config=config,
                    normalize=normalize,
                )
                plots.append(str(path))

    confusion_events = 0
    confusion_summary: dict[str, Any] = {}
    if "confusion" in plot_groups:
        weightings = [("analysis_weighted", dataset.analysis_weight)]
        if config.uses_distinct_prediction_truth:
            weightings.append(
                (
                    "unweighted",
                    np.ones(dataset.prediction_truth_index.shape, dtype=np.float64),
                )
            )
        for weighting_name, confusion_weights in weightings:
            confusion_matrix, valid_events = compute_confusion_matrix(
                dataset.score_matrix(config.score_names),
                dataset.prediction_truth_index,
                confusion_weights,
                len(config.scores),
            )
            if weighting_name == "analysis_weighted":
                confusion_events = valid_events
            confusion_summary[weighting_name] = {
                "events": valid_events,
                "raw_matrix": _matrix_payload(confusion_matrix),
                "truth_normalized": _matrix_payload(
                    normalize_confusion(confusion_matrix, "truth")
                ),
                "prediction_normalized": _matrix_payload(
                    normalize_confusion(confusion_matrix, "prediction")
                ),
            }
            for normalization, short_name in [
                ("truth", "truth"),
                ("prediction", "pred"),
            ]:
                filename = (
                    f"coarse_confusion_matrix_{short_name}__{weighting_name}{plot_suffix}"
                    if config.uses_distinct_prediction_truth
                    else f"confusion_matrix_{short_name}{plot_suffix}"
                )
                path = plot_dir / filename
                plot_confusion_matrix(
                    outpath=path,
                    matrix=normalize_confusion(confusion_matrix, normalization),
                    truths=config.prediction_truth_categories,
                    scores=config.scores,
                    normalization=normalization,
                    config=config,
                    title=(
                        f"Coarse confusion matrix ({weighting_name.replace('_', ' ')})"
                        if config.uses_distinct_prediction_truth
                        else None
                    ),
                )
                plots.append(str(path))

    if "roc" in plot_groups:
        truths_by_name = {
            truth.name: truth for truth in config.prediction_truth_categories
        }
        scores_by_name = {score.name: score for score in config.scores}
        roc_sets = [("analysis_weighted", pairwise)]
        if config.uses_distinct_prediction_truth:
            roc_sets.append(("unweighted", pairwise_unweighted))
        for weighting_name, roc_results in roc_sets:
            for signal_name, backgrounds in roc_results.items():
                if not backgrounds:
                    continue
                filename = (
                    f"coarse_roc_{signal_name}__{weighting_name}{plot_suffix}"
                    if config.uses_distinct_prediction_truth
                    else f"roc_{signal_name}{plot_suffix}"
                )
                path = plot_dir / filename
                plot_pairwise_roc(
                    outpath=path,
                    signal=scores_by_name[signal_name],
                    background_results=backgrounds,
                    truths_by_name=truths_by_name,
                    config=config,
                )
                plots.append(str(path))

    if "auc" in plot_groups:
        auc_sets = [("analysis_weighted", pairwise)]
        if config.uses_distinct_prediction_truth:
            auc_sets.append(("unweighted", pairwise_unweighted))
        for weighting_name, auc_results in auc_sets:
            auc_matrix = build_auc_matrix(config.score_names, auc_results)
            filename = (
                f"coarse_pairwise_auc_matrix__{weighting_name}{plot_suffix}"
                if config.uses_distinct_prediction_truth
                else f"pairwise_auc_matrix{plot_suffix}"
            )
            path = plot_dir / filename
            plot_auc_matrix(
                outpath=path,
                auc_matrix=auc_matrix,
                scores=config.scores,
                config=config,
                title=(
                    f"Coarse pairwise AUC ({weighting_name.replace('_', ' ')})"
                    if config.uses_distinct_prediction_truth
                    else None
                ),
            )
            plots.append(str(path))

    mass_sculpting_summary: dict[str, Any] | None = None
    if "mass-sculpting" in plot_groups:
        mass_plots, mass_sculpting_summary = run_mass_sculpting(
            config=config,
            dataset=dataset,
            plot_dir=plot_dir,
            summary_dir=summary_dir,
            plot_suffix=plot_suffix,
            selected_score_names=mass_sculpting_score_names,
        )
        plots.extend(mass_plots)

    score_significance_summary: dict[str, Any] | None = None
    if "score-significance" in plot_groups:
        significance_plots, score_significance_summary = run_score_significance(
            config=config,
            dataset=dataset,
            plot_dir=plot_dir,
            summary_dir=summary_dir,
            plot_suffix=plot_suffix,
        )
        plots.extend(significance_plots)

    score_significance_2d_summary: dict[str, Any] | None = None
    score_significance_2d_hierarchical_summary: dict[str, Any] | None = None
    if "score-significance-2d" in plot_groups:
        if config.score_significance_2d.enabled:
            significance_2d_plots, score_significance_2d_summary = (
                run_score_significance_2d(
                    config=config,
                    dataset=dataset,
                    plot_dir=plot_dir,
                    summary_dir=summary_dir,
                    plot_suffix=plot_suffix,
                )
            )
            plots.extend(significance_2d_plots)
        if config.score_significance_2d_hierarchical.enabled:
            hierarchical_plots, score_significance_2d_hierarchical_summary = (
                run_score_significance_2d(
                    config=config,
                    dataset=dataset,
                    plot_dir=plot_dir,
                    summary_dir=summary_dir,
                    plot_suffix=plot_suffix,
                    settings=config.score_significance_2d_hierarchical,
                    summary_stem="score_significance_2d_hierarchical",
                )
            )
            plots.extend(hierarchical_plots)

    qcd_score_scan_summary: dict[str, Any] | None = None
    if "qcd-score-scan" in plot_groups:
        qcd_plots, qcd_score_scan_summary = run_qcd_score_scan(
            config=config,
            dataset=dataset,
            plot_dir=plot_dir,
            summary_dir=summary_dir,
            plot_suffix=plot_suffix,
        )
        plots.extend(qcd_plots)

    tthz_qcd_discriminant_summary: dict[str, Any] | None = None
    if "tthz-qcd-discriminant" in plot_groups:
        discriminant_plots, tthz_qcd_discriminant_summary = (
            run_tthz_qcd_discriminant(
                config=config,
                dataset=dataset,
                plot_dir=plot_dir,
                summary_dir=summary_dir,
                plot_suffix=plot_suffix,
            )
        )
        plots.extend(discriminant_plots)

    regressed_mass_mcstat_summary: dict[str, Any] | None = None
    if plot_groups & {"regressed-mass-fine-binning", "regressed-mass-mcstat"}:
        fine_mass_plots, regressed_mass_mcstat_summary = (
            run_regressed_mass_fine_binning(
                include_data="regressed-mass-fine-binning" in plot_groups,
                config=config,
                dataset=dataset,
                plot_dir=plot_dir,
                summary_dir=summary_dir,
                plot_suffix=plot_suffix,
            )
        )
        plots.extend(fine_mass_plots)

    weighting_diagnostics_summary: dict[str, Any] | None = None
    if "weighting-diagnostics" in plot_groups:
        diagnostics_plots, weighting_diagnostics_summary = run_weighting_diagnostics(
            config=config,
            dataset=dataset,
            plot_dir=plot_dir,
            summary_dir=summary_dir,
            plot_suffix=plot_suffix,
        )
        plots.extend(diagnostics_plots)

    d_tthz_summary: dict[str, Any] | None = None
    if args.only is None and "D_ttHZ" in dataset.scores:
        d_tthz_plots, d_tthz_summary = run_d_tthz_study(
            config=config,
            dataset=dataset,
            plot_dir=plot_dir,
            summary_dir=summary_dir,
            plot_suffix=plot_suffix,
            score_significance_summary=score_significance_summary,
            score_significance_2d_summary=score_significance_2d_summary,
            hierarchical_summary=score_significance_2d_hierarchical_summary,
            mass_sculpting_summary=mass_sculpting_summary,
        )
        plots.extend(d_tthz_plots)

    truth_yields = _truth_yields(
        config.truth_categories, dataset.truth_index, dataset
    )
    prediction_truth_yields = _truth_yields(
        config.prediction_truth_categories,
        dataset.prediction_truth_index,
        dataset,
    )
    pairwise_auc = {
        signal: {background: float(result.auc) for background, result in backgrounds.items()}
        for signal, backgrounds in pairwise.items()
    }
    pairwise_details = {
        signal: {
            background: {
                "auc": float(result.auc),
                "sklearn_auc": result.sklearn_auc,
                "n_signal": result.n_signal,
                "n_background": result.n_background,
                "zero_denominator_events": result.zero_denominator_events,
            }
            for background, result in backgrounds.items()
        }
        for signal, backgrounds in pairwise.items()
    }
    pairwise_auc_unweighted = {
        signal: {
            background: float(result.auc)
            for background, result in backgrounds.items()
        }
        for signal, backgrounds in pairwise_unweighted.items()
    }
    pairwise_details_unweighted = {
        signal: {
            background: {
                "auc": float(result.auc),
                "sklearn_auc": result.sklearn_auc,
                "n_signal": result.n_signal,
                "n_background": result.n_background,
                "zero_denominator_events": result.zero_denominator_events,
            }
            for background, result in backgrounds.items()
        }
        for signal, backgrounds in pairwise_unweighted.items()
    }
    diagnostic_truth_yields = {
        name: {
            "events": int(np.sum(mask)),
            "analysis_weight_sum": float(np.sum(dataset.analysis_weight[mask])),
            "signed_weight_sum": float(np.sum(dataset.signed_weight[mask])),
        }
        for name, mask in dataset.diagnostic_truth_masks.items()
    }
    truth_coverage = {
        "selected": dataset.totals["events_selected"],
        "fine": {
            "classified": dataset.totals.get("fine_events_classified", dataset.totals["events_classified"]),
            "unclassified": dataset.totals.get("fine_events_unclassified", dataset.totals["events_unclassified"]),
            "multiply_assigned": dataset.totals.get("fine_events_multiply_assigned", 0),
        },
        "coarse": {
            "classified": dataset.totals.get("coarse_events_classified", dataset.totals["events_classified"]),
            "unclassified": dataset.totals.get("coarse_events_unclassified", dataset.totals["events_unclassified"]),
            "multiply_assigned": dataset.totals.get("coarse_events_multiply_assigned", 0),
        },
        "per_sample_diagnostics": [
            {
                "name": sample["name"],
                "fine_unclassified_diagnostics": sample.get(
                    "fine_unclassified_diagnostics", []
                ),
                "coarse_unclassified_diagnostics": sample.get(
                    "coarse_unclassified_diagnostics", []
                ),
            }
            for sample in dataset.sample_summaries
            if sample.get("fine_events_unclassified", sample["events_unclassified"])
            or sample.get("coarse_events_unclassified", sample["events_unclassified"])
        ],
    }
    summary: dict[str, Any] = {
        "channel": config.channel,
        "config": str(config.config_path),
        "input_source": "cache" if args.from_cache else "root",
        "cache": (
            {
                "path": str(cache_path),
                "metadata_path": str(cache_metadata_path(cache_path)),
                "schema_version": cache_metadata["cache_schema_version"],
                "definition_hash": cache_metadata["definition_hash"],
                "created_utc": cache_metadata["created_utc"],
            }
            if cache_path is not None and cache_metadata is not None
            else None
        ),
        "input_location": config.input_location,
        "sample_file_pattern": config.sample_file_pattern,
        "output_directory": str(config.outdir),
        "tree_name": config.tree_name,
        "selection": config.selection,
        "selection_semantics": (
            "No additional selection; all events already written after the upstream Pepper selection."
            if not config.selection.strip()
            else "Configured selection applied after loading the produced ntuple."
        ),
        "labels": config.truth_names,
        "fine_truth_categories": config.truth_names,
        "prediction_truth_categories": config.prediction_truth_names,
        "score_classes": config.score_names,
        "score_branches": {score.name: score.branch for score in config.scores},
        "auxiliary_scores": [score.name for score in config.auxiliary_scores],
        "auxiliary_score_branches": {
            score.name: score.branch for score in config.auxiliary_scores
        },
        "analysis_discriminants": {
            name: next(score.branch for score in config.all_scores if score.name == name)
            for name in config.analysis_discriminants
        },
        "loaded_score_branches": {
            score.name: score.branch
            for score in config.all_scores
            if score.name in dataset.scores
        },
        "plot_groups": sorted(plot_groups),
        "plot_format": plot_format,
        "truth_definitions": {truth.name: truth.expression for truth in config.truth_categories},
        "prediction_truth_definitions": {
            truth.name: truth.expression
            for truth in config.prediction_truth_categories
        },
        "diagnostic_truth_definitions": {
            truth.name: truth.expression
            for truth in config.diagnostic_truth_categories
        },
        "analysis_branches_retained": list(dataset.analysis_columns),
        "truth_yields": truth_yields,
        "prediction_truth_yields": prediction_truth_yields,
        "diagnostic_truth_yields": diagnostic_truth_yields,
        "truth_coverage": truth_coverage,
        "score_sum_diagnostics": _score_sum_diagnostics(config, dataset),
        "native_score_histograms": (
            _score_distribution_payload(config, dataset)
            if all(name in dataset.scores for name in config.score_names)
            else None
        ),
        "coarse_union_validation": _coarse_union_validation(config, dataset),
        "available_truth_categories": [
            name for name, entry in truth_yields.items() if entry["events"] > 0
        ],
        "pairwise_auc": pairwise_auc,
        "pairwise_details": pairwise_details,
        "pairwise_auc_unweighted": pairwise_auc_unweighted,
        "pairwise_details_unweighted": pairwise_details_unweighted,
        "auc_convention": (
            "Standard ROC AUC: integral of signal efficiency (TPR) over background "
            "efficiency (FPR). ROC plots display x=signal efficiency and y=background efficiency."
        ),
        "auc_validation": _auc_validation(
            pairwise,
            config.validate_auc_with_sklearn and needs_pairwise,
        ),
        "auc_validation_unweighted": _auc_validation(
            pairwise_unweighted,
            config.validate_auc_with_sklearn
            and needs_pairwise
            and config.uses_distinct_prediction_truth,
        ),
        "confusion_events": confusion_events,
        "confusion_matrices": confusion_summary,
        "mass_sculpting": mass_sculpting_summary,
        "score_significance": score_significance_summary,
        "score_significance_2d": score_significance_2d_summary,
        "score_significance_2d_hierarchical": (
            score_significance_2d_hierarchical_summary
        ),
        "D_ttHZ_study": d_tthz_summary,
        "qcd_score_scan": qcd_score_scan_summary,
        "ttHZ_qcd_discriminant": tthz_qcd_discriminant_summary,
        "regressed_mass_mcstat": regressed_mass_mcstat_summary,
        "ar_score_composition": ar_score_composition_summary,
        "qcd_veto_conditional": qcd_veto_conditional_summary,
        "weighting_diagnostics": weighting_diagnostics_summary,
        "normalization": {
            "lumi_fb": config.lumi_fb,
            "xsec_file": config.xsec_file,
            "gen_sumw_file": config.gen_sumw_file,
            "sample_norm": "lumi_fb * xsec_fb / gen_sumw",
        },
        "weight_convention": {
            "raw_branch": config.weight_branch,
            "analysis_weight": "sample_norm * abs(raw event weight)",
            "signed_bookkeeping_weight": "sample_norm * raw event weight",
            "score_shape": "analysis-weighted, normalized independently to unit area per truth class",
            "score_yield": "analysis-weighted expected event yield",
            "confusion": "analysis-weighted",
            "roc_auc": "analysis-weighted",
            "mass_sculpting": (
                "analysis-weighted; each curve independently normalized to unit density "
                "over the plotted mass range"
            ),
            "score_significance": (
                "analysis-weighted S/sqrt(S+B); one truth category is signal and "
                "every other selected MC event is background; both inclusive and "
                "configured mass-window scans are retained"
            ),
            "score_significance_2d": (
                "analysis-weighted S/sqrt(S+B) for two simultaneous score cuts; "
                "one fine truth category is signal and every other selected MC "
                "event is background"
            ),
            "qcd_score_scan": "analysis-weighted expected yields and weighted efficiencies",
            "weighting_diagnostics": (
                "parallel analysis-weighted and unweighted diagnostics; exact training "
                "weights are emitted only when reconstructable"
            ),
        },
        "max_files_per_sample": config.max_files_per_sample,
        "number_of_files_discovered": dataset.totals["files_discovered"],
        "number_of_files_attempted": dataset.totals["files_attempted"],
        "number_of_files_processed": dataset.totals["files_processed"],
        "number_of_files_missing_tree": dataset.totals["files_missing_tree"],
        "number_of_events_read": dataset.totals["events_read"],
        "number_of_events_selected": dataset.totals["events_selected"],
        "number_of_events_classified": dataset.totals["events_classified"],
        "number_of_events_unclassified": dataset.totals["events_unclassified"],
        "number_of_fine_events_multiply_assigned": dataset.totals.get(
            "fine_events_multiply_assigned", 0
        ),
        "number_of_coarse_events_classified": dataset.totals.get(
            "coarse_events_classified", dataset.totals["events_classified"]
        ),
        "number_of_coarse_events_unclassified": dataset.totals.get(
            "coarse_events_unclassified", dataset.totals["events_unclassified"]
        ),
        "number_of_coarse_events_multiply_assigned": dataset.totals.get(
            "coarse_events_multiply_assigned", 0
        ),
        "samples": dataset.sample_summaries,
        "plots": plots,
    }
    if config.uses_distinct_prediction_truth:
        write_json(
            summary_dir / "truth_coverage.json",
            {
                "fine_truth_definitions": summary["truth_definitions"],
                "coarse_truth_definitions": summary[
                    "prediction_truth_definitions"
                ],
                "diagnostic_truth_definitions": summary[
                    "diagnostic_truth_definitions"
                ],
                "coverage": truth_coverage,
                "coarse_union_validation": summary[
                    "coarse_union_validation"
                ],
                "fine_truth_yields": truth_yields,
                "coarse_truth_yields": prediction_truth_yields,
                "diagnostic_truth_yields": diagnostic_truth_yields,
            },
        )
        if confusion_summary:
            write_json(
                summary_dir / "coarse_confusion.json",
                {
                    "classes": config.prediction_truth_names,
                    "matrices": confusion_summary,
                    "prediction": "argmax of native "
                    + config.plot_options.get("study_label", "V4") + " score branches",
                },
            )
        if pairwise or pairwise_unweighted:
            write_json(
                summary_dir / "coarse_roc_auc.json",
                {
                    "classes": config.prediction_truth_names,
                    "analysis_weighted": pairwise_details,
                    "unweighted": pairwise_details_unweighted,
                    "discriminator": "D(A,B) = score_A / (score_A + score_B)",
                },
            )
    write_json(config.outdir / "nn_study_summary.json", summary)
    write_text(config.outdir / "nn_study_summary.txt", format_summary_text(summary))
    return summary


def main() -> None:
    args = _build_parser().parse_args()
    summary = run(args)
    if summary.get("operation") == "qcd-data-mc":
        print("QCD Data/MC diagnostic finished (MC cache + dedicated data histogram cache).")
        for path in summary["plots"]:
            print(f"Plot: {path}")
        for path in summary["summaries"]:
            print(f"Summary: {path}")
        return
    if summary.get("operation") == "prepare-data-cache":
        print("Real-data sideband cache ready.")
        print(f"Cache: {summary['cache_path']}")
        print(f"Data samples: {', '.join(summary['data_samples'])}")
        print(f"Reused existing cache: {summary['cache_reused']}")
        return
    if summary.get("operation") == "prepare-cache":
        print("NN-study cache ready.")
        print(f"Cache: {summary['cache_path']}")
        print(f"Metadata: {summary['cache_metadata_path']}")
        print(f"Reused existing cache: {summary['cache_reused']}")
        print(f"Selected events: {summary['number_of_events_selected']}")
        print(f"Cache size: {summary['cache_size_bytes'] / (1024 ** 2):.2f} MiB")
        return
    print("NN study finished.")
    print(f"Output directory: {summary['output_directory']}")
    print(f"Input source: {summary['input_source']}")
    print(f"Processed Events files: {summary['number_of_files_processed']}")
    print(f"Selected events: {summary['number_of_events_selected']}")
    print(f"Plots: {len(summary['plots'])}")
