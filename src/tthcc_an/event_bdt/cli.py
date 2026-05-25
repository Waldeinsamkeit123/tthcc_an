from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tthcc_an.event_bdt.config import EventBdtConfig, load_event_bdt_config
from tthcc_an.event_bdt.dataset import prepare_event_bdt_inputs
from tthcc_an.event_bdt.plotting import (
    plot_class_score_shapes,
    plot_process_score_shapes,
    plot_roc_curve,
)
from tthcc_an.event_bdt.prediction import (
    PREDICTION_MODE_CHOICES,
    predict_event_bdt_to_root,
)
from tthcc_an.event_bdt.training import load_predictions_payload, train_event_bdt


REPO_ROOT = Path(__file__).resolve().parents[3]


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


def _evaluate_outputs(config: EventBdtConfig) -> dict[str, float]:
    arrays, metadata = load_predictions_payload(config.predictions_path)
    labels = arrays["train_label"].astype(int)
    weights = arrays["train_weight"].astype(float)
    scores = arrays["bdt_score"].astype(float)
    processes = arrays["process"].astype(str)
    label_texts = arrays["label_text"].astype(str)

    trainable_mask = labels >= 0
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

    unique_processes = list(dict.fromkeys(processes.tolist()))
    process_labels: dict[str, str] = {}
    for process in unique_processes:
        mask = processes == process
        if np.any(mask):
            process_labels[process] = str(label_texts[mask][0])

    preferred_order = [
        "ttHcc",
        "ttHbb",
        "ttH_nonbb",
        "ttbar",
        "ttbb",
        "single_top",
        "ttv",
        "ttll",
        "wjets",
        "zjets",
        "qcd",
    ]
    process_order = [process for process in preferred_order if process in unique_processes]
    for process in unique_processes:
        if process not in process_order:
            process_order.append(process)

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
        if process in unique_processes
    ]
    background_mask = labels == 0
    if np.any(background_mask):
        pieces_scores = [scores[processes == process] for process in tth_family_order]
        pieces_weights = [weights[processes == process] for process in tth_family_order]
        pieces_processes = [
            np.full(int(np.sum(processes == process)), process, dtype=np.str_)
            for process in tth_family_order
        ]
        pieces_scores.append(scores[background_mask])
        pieces_weights.append(weights[background_mask])
        pieces_processes.append(
            np.full(int(np.sum(background_mask)), "background", dtype=np.str_)
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
        print(f"Weighted AUC: {summary['weighted_auc']:.4f}")
        return

    if args.command == "evaluate":
        summary = _evaluate_outputs(config)
        print("Event-BDT evaluation plots finished.")
        print(f"Plot directory: {config.plot_dir_path}")
        print(f"Weighted AUC: {summary['weighted_auc']:.4f}")
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
