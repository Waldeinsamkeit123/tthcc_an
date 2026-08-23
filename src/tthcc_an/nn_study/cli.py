from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from tthcc_an.nn_study.config import load_nn_study_config
from tthcc_an.nn_study.dataset import NnDataset, load_nn_dataset
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
from tthcc_an.nn_study.reporting import format_summary_text, write_json, write_text
from tthcc_an.nn_study.weighting_diagnostics import run_weighting_diagnostics


REPO_ROOT = Path(__file__).resolve().parents[3]
PLOT_CHOICES = (
    "scores",
    "confusion",
    "roc",
    "auc",
    "mass-sculpting",
    "qcd-score-scan",
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
    return parser


def _truth_yields(config: Any, dataset: NnDataset) -> dict[str, dict[str, Any]]:
    yields: dict[str, dict[str, Any]] = {}
    for index, truth in enumerate(config.truth_categories):
        mask = dataset.truth_index == index
        yields[truth.name] = {
            "label": truth.label,
            "events": int(np.sum(mask)),
            "analysis_weight_sum": float(np.sum(dataset.analysis_weight[mask])),
            "signed_weight_sum": float(np.sum(dataset.signed_weight[mask])),
        }
    return yields


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_nn_study_config(
        args.config,
        REPO_ROOT,
        outdir=args.outdir,
        selection=args.selection,
        max_files_per_sample=args.max_files_per_sample,
    )
    config.outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = config.outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = config.outdir / "summaries"
    plot_groups = set(args.only or ("scores", "confusion", "roc", "auc"))
    if args.only is None and config.mass_sculpting.enabled:
        plot_groups.add("mass-sculpting")
    if args.only is None and config.qcd_score_scan.enabled:
        plot_groups.add("qcd-score-scan")
    if args.only is None and config.weighting_diagnostics.enabled:
        plot_groups.add("weighting-diagnostics")
    if "mass-sculpting" in plot_groups and not config.mass_sculpting.enabled:
        raise ValueError("mass_sculpting is not enabled in this config.")
    if "qcd-score-scan" in plot_groups and not config.qcd_score_scan.enabled:
        raise ValueError("qcd_score_scan is not enabled in this config.")
    if (
        "weighting-diagnostics" in plot_groups
        and not config.weighting_diagnostics.enabled
    ):
        raise ValueError("weighting_diagnostics is not enabled in this config.")

    loaded_score_names: list[str] = []
    if "scores" in plot_groups:
        loaded_score_names.extend(config.all_score_names)
    if plot_groups & {"confusion", "roc", "auc"}:
        loaded_score_names.extend(config.score_names)
    if "weighting-diagnostics" in plot_groups:
        loaded_score_names.extend(config.score_names)
    loaded_analysis_branches: list[str] = []
    if "mass-sculpting" in plot_groups:
        loaded_score_names.extend(
            scan.score_name for scan in config.mass_sculpting.scans
        )
        loaded_analysis_branches.extend(
            variable.branch for variable in config.mass_sculpting.variables
        )
    if "qcd-score-scan" in plot_groups:
        loaded_score_names.append(config.qcd_score_scan.score_name)
    if "weighting-diagnostics" in plot_groups:
        loaded_analysis_branches.extend(
            config.weighting_diagnostics.reweight_variables
        )
    if args.only is None:
        loaded_analysis_branches = list(config.analysis_branches)
    loaded_score_names = list(dict.fromkeys(loaded_score_names))
    loaded_analysis_branches = list(dict.fromkeys(loaded_analysis_branches))
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
            dataset.truth_index,
            dataset.analysis_weight,
            validate_with_sklearn=config.validate_auc_with_sklearn,
        )
        if needs_pairwise
        else {}
    )

    plots: list[str] = []
    if "scores" in plot_groups:
        for score in config.all_scores:
            shape_path = plot_dir / f"score_{score.name}{plot_suffix}"
            yield_path = plot_dir / f"score_{score.name}__yield{plot_suffix}"
            common = {
                "score": score,
                "values": dataset.scores[score.name],
                "truth_index": dataset.truth_index,
                "weights": dataset.analysis_weight,
                "truths": config.truth_categories,
                "config": config,
            }
            plot_score_distribution(outpath=shape_path, normalize=True, **common)
            plot_score_distribution(outpath=yield_path, normalize=False, **common)
            plots.extend([str(shape_path), str(yield_path)])

    confusion_events = 0
    if "confusion" in plot_groups:
        confusion_matrix, confusion_events = compute_confusion_matrix(
            dataset.score_matrix(config.score_names),
            dataset.truth_index,
            dataset.analysis_weight,
            len(config.scores),
        )
        for normalization, filename in [
            ("truth", f"confusion_matrix_truth{plot_suffix}"),
            ("prediction", f"confusion_matrix_pred{plot_suffix}"),
        ]:
            path = plot_dir / filename
            plot_confusion_matrix(
                outpath=path,
                matrix=normalize_confusion(confusion_matrix, normalization),
                truths=config.truth_categories,
                scores=config.scores,
                normalization=normalization,
                config=config,
            )
            plots.append(str(path))

    if "roc" in plot_groups:
        truths_by_name = {truth.name: truth for truth in config.truth_categories}
        scores_by_name = {score.name: score for score in config.scores}
        for signal_name, backgrounds in pairwise.items():
            if not backgrounds:
                continue
            path = plot_dir / f"roc_{signal_name}{plot_suffix}"
            plot_pairwise_roc(
                outpath=path,
                signal=scores_by_name[signal_name],
                background_results=backgrounds,
                truths_by_name=truths_by_name,
                config=config,
            )
            plots.append(str(path))

    if "auc" in plot_groups:
        auc_matrix = build_auc_matrix(config.score_names, pairwise)
        path = plot_dir / f"pairwise_auc_matrix{plot_suffix}"
        plot_auc_matrix(outpath=path, auc_matrix=auc_matrix, scores=config.scores, config=config)
        plots.append(str(path))

    mass_sculpting_summary: dict[str, Any] | None = None
    if "mass-sculpting" in plot_groups:
        mass_plots, mass_sculpting_summary = run_mass_sculpting(
            config=config,
            dataset=dataset,
            plot_dir=plot_dir,
            summary_dir=summary_dir,
            plot_suffix=plot_suffix,
        )
        plots.extend(mass_plots)

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

    truth_yields = _truth_yields(config, dataset)
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
    summary: dict[str, Any] = {
        "channel": config.channel,
        "config": str(config.config_path),
        "input_location": config.input_location,
        "output_directory": str(config.outdir),
        "tree_name": config.tree_name,
        "selection": config.selection,
        "selection_semantics": (
            "No additional selection; all events already written after the upstream Pepper selection."
            if not config.selection.strip()
            else "Configured selection applied after loading the produced ntuple."
        ),
        "labels": config.truth_names,
        "score_classes": config.score_names,
        "score_branches": {score.name: score.branch for score in config.scores},
        "auxiliary_scores": [score.name for score in config.auxiliary_scores],
        "auxiliary_score_branches": {
            score.name: score.branch for score in config.auxiliary_scores
        },
        "loaded_score_branches": {
            score.name: score.branch
            for score in config.all_scores
            if score.name in dataset.scores
        },
        "plot_groups": sorted(plot_groups),
        "plot_format": plot_format,
        "truth_definitions": {truth.name: truth.expression for truth in config.truth_categories},
        "analysis_branches_retained": list(dataset.analysis_columns),
        "truth_yields": truth_yields,
        "available_truth_categories": [
            name for name, entry in truth_yields.items() if entry["events"] > 0
        ],
        "pairwise_auc": pairwise_auc,
        "pairwise_details": pairwise_details,
        "auc_convention": (
            "Standard ROC AUC: integral of signal efficiency (TPR) over background "
            "efficiency (FPR). ROC plots display x=signal efficiency and y=background efficiency."
        ),
        "auc_validation": _auc_validation(
            pairwise,
            config.validate_auc_with_sklearn and needs_pairwise,
        ),
        "confusion_events": confusion_events,
        "mass_sculpting": mass_sculpting_summary,
        "qcd_score_scan": qcd_score_scan_summary,
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
        "samples": dataset.sample_summaries,
        "plots": plots,
    }
    write_json(config.outdir / "nn_study_summary.json", summary)
    write_text(config.outdir / "nn_study_summary.txt", format_summary_text(summary))
    return summary


def main() -> None:
    args = _build_parser().parse_args()
    summary = run(args)
    print("NN study finished.")
    print(f"Output directory: {summary['output_directory']}")
    print(f"Processed Events files: {summary['number_of_files_processed']}")
    print(f"Selected events: {summary['number_of_events_selected']}")
    print(f"Plots: {len(summary['plots'])}")
