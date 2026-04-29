from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

from tthcc_an.config_loader import (
    DEFAULT_CANDIDATE_STRATEGY,
    DEFAULT_ETA_MAX,
    DEFAULT_MSD_WINDOW_HIGH,
    DEFAULT_MSD_WINDOW_LOW,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLOT_OPTIONS,
    DEFAULT_PT_MIN,
    DEFAULT_SCORES,
    DEFAULT_SCORE_HIST_BINS,
    DEFAULT_SIG_EFFS,
    DEFAULT_TARGETS,
    DEFAULT_TREE_NAME,
    DEFAULT_UPROOT_STEP_SIZE,
    DEFAULT_WEIGHT_BRANCH,
    SampleConfig,
    StudyConfig,
    load_json_maybe_with_comments,
    load_study_config,
    resolve_output_dir,
)
from tthcc_an.definitions import (
    COUNT_FIELDS,
    FATJET_FIELDS,
    FLOAT_FIELDS,
    GLOBALPART3_CONTOUR_PLOT,
    SCORE_INPUT_FIELDS,
    SCORE_LABELS,
    TARGET_DEFINITIONS,
    TRUTH_LABEL_TO_CODE,
    build_process_entries_from_samples as _build_process_entries_from_samples,
    build_process_entries_from_summaries as _build_process_entries_from_summaries,
)
from tthcc_an.metrics import (
    compute_roc,
    compute_roc_from_hist,
    compute_working_points,
    compute_working_points_from_hist,
    safe_divide as _safe_divide,
)
from tthcc_an.payload_io import (
    HISTOGRAM_PAYLOAD_MODE,
    build_histogram_payload_from_raw_data,
    detect_payload_mode as _detect_payload_mode,
    export_chunk_payload,
    export_histogram_payload,
    load_merged_chunk_payloads,
    load_merged_histogram_payloads,
)
from tthcc_an.plotting import (
    build_plot_style,
    compute_globalpart3_region_efficiencies_from_hist_payload,
    compute_globalpart3_region_efficiencies_from_raw,
    plot_background_process_score_distribution,
    plot_background_process_score_distribution_from_hist,
    plot_background_process_working_points,
    plot_background_process_working_points_from_hist,
    plot_globalpart3_contours,
    plot_globalpart3_contours_from_hist,
    plot_roc_curve,
    plot_score_distribution,
    plot_score_distribution_from_hist,
    plot_significance_scan,
)
from tthcc_an.reporting import format_contour_region_efficiency_text, format_summary_text, write_csv, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
FLOAT_STORAGE_DTYPE = np.float32
COUNT_STORAGE_DTYPE = np.int16
WEIGHT_STORAGE_DTYPE = np.float32
TRUTH_CODE_DTYPE = np.int8
PROCESS_CODE_DTYPE = np.int16
CANDIDATE_STRATEGIES = (
    "all_jets",
    "leading_pt",
    "mass_window_all_jets",
    "mass_window_leading_pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Study boosted Higgs AK8 taggers from Pepper-produced ROOT files. "
            "Supports H->cc and H->bb working points, weighted yields, ROC curves, "
            "mass-window selections, and CMS-style plots."
        )
    )
    parser.add_argument(
        "--config",
        "--sample-config",
        dest="config",
        required=True,
        help="Path to the JSON study configuration file.",
    )
    parser.add_argument("--tree-name", default=None, help="Tree name. Defaults to study.tree_name from config.")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Defaults to study.outdir from config.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=sorted(TARGET_DEFINITIONS),
        default=None,
        help="Targets to study. Defaults to study.targets from config.",
    )
    parser.add_argument(
        "--scores",
        nargs="+",
        default=None,
        help=f"Known scores: {', '.join(sorted(SCORE_LABELS))}. Defaults to study.scores from config.",
    )
    parser.add_argument(
        "--sig-effs",
        nargs="+",
        type=float,
        default=None,
        help="Target signal efficiencies. Defaults to study.sig_effs from config.",
    )
    parser.add_argument("--pt-min", type=float, default=None, help="Minimum AK8 pt. Defaults to study.pt_min from config.")
    parser.add_argument("--eta-max", type=float, default=None, help="Maximum |eta|. Defaults to study.eta_max from config.")
    parser.add_argument(
        "--candidate-strategy",
        choices=list(CANDIDATE_STRATEGIES),
        default=None,
        help="Candidate strategy. Defaults to study.candidate_strategy from config.",
    )
    parser.add_argument("--msd-window-low", type=float, default=None)
    parser.add_argument("--msd-window-high", type=float, default=None)
    parser.add_argument("--max-files-per-sample", type=int, default=None)
    parser.add_argument("--gen-sumw-file", default=None)
    parser.add_argument(
        "--xsec-file",
        default=None,
        help="Optional override for the cross section JSON file.",
    )
    parser.add_argument("--lumi-fb", type=float, default=None)
    parser.add_argument("--weight-branch", default=None, help="Event-weight branch. Defaults to study.weight_branch from config.")
    parser.add_argument("--disable-event-weights", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--export-chunk",
        default=None,
        help="Write a slim per-chunk NPZ payload and exit without producing final study outputs.",
    )
    parser.add_argument(
        "--chunk-payload-mode",
        choices=["histogram", "raw"],
        default="histogram",
        help="Payload format written by --export-chunk. Histogram mode keeps merge memory low.",
    )
    parser.add_argument(
        "--merge-chunks",
        nargs="+",
        default=None,
        help="Merge one or more exported chunk NPZ files or glob patterns instead of reading ROOT inputs directly.",
    )
    parser.add_argument(
        "--uproot-step-size",
        default=None,
        help="Chunk size used when iterating over ROOT trees with uproot. Defaults to study.uproot_step_size from config.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only redraw plots from the chosen data source without recomputing tables/summaries.",
    )
    parser.add_argument(
        "--plot-input",
        default=None,
        help="Path to a cached plot_input.npz file produced by a previous run.",
    )
    parser.add_argument("--plot-title-size", type=float, default=None)
    parser.add_argument("--plot-label-size", type=float, default=None)
    parser.add_argument("--plot-tick-size", type=float, default=None)
    parser.add_argument("--plot-legend-size", type=float, default=None)
    parser.add_argument("--plot-cms-size", type=float, default=None)
    parser.add_argument("--plot-dpi", type=int, default=None)
    parser.add_argument(
        "--score-hist-bins",
        type=int,
        default=None,
        help="Number of score bins used by histogram chunk payloads and histogram-based merges. Defaults to study.score_hist_bins from config.",
    )
    return parser.parse_args()


def _configured_value(cli_value: Any, config_value: Any, fallback: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return fallback


def _configured_targets(args: argparse.Namespace, study_config: StudyConfig) -> list[str]:
    configured = _configured_value(args.targets, study_config.study_defaults.get("targets"), list(DEFAULT_TARGETS))
    if not isinstance(configured, list) or not configured:
        raise ValueError("Config field 'study.targets' must be a non-empty list.")
    unknown = [target for target in configured if target not in TARGET_DEFINITIONS]
    if unknown:
        raise ValueError(f"Unknown targets in configuration: {', '.join(sorted(unknown))}")
    return list(configured)


def _configured_scores(args: argparse.Namespace, study_config: StudyConfig) -> list[str]:
    configured = _configured_value(args.scores, study_config.study_defaults.get("scores"), list(DEFAULT_SCORES))
    if not isinstance(configured, list) or not configured:
        raise ValueError("Config field 'study.scores' must be a non-empty list.")
    unknown = [score for score in configured if score != "auto" and score not in SCORE_LABELS]
    if unknown:
        raise ValueError(f"Unknown scores in configuration: {', '.join(sorted(unknown))}")
    return list(configured)


def _configured_sig_effs(args: argparse.Namespace, study_config: StudyConfig) -> list[float]:
    configured = _configured_value(args.sig_effs, study_config.study_defaults.get("sig_effs"), list(DEFAULT_SIG_EFFS))
    if not isinstance(configured, list) or not configured:
        raise ValueError("Config field 'study.sig_effs' must be a non-empty list.")
    return [float(value) for value in configured]


def apply_config_defaults_to_args(args: argparse.Namespace, study_config: StudyConfig) -> argparse.Namespace:
    effective_args = argparse.Namespace(**vars(args))

    effective_args.tree_name = str(
        _configured_value(args.tree_name, study_config.study_defaults.get("tree_name"), DEFAULT_TREE_NAME)
    )
    effective_args.outdir = str(
        _configured_value(args.outdir, study_config.study_defaults.get("outdir"), DEFAULT_OUTPUT_DIR)
    )
    effective_args.targets = _configured_targets(args, study_config)
    effective_args.scores = _configured_scores(args, study_config)
    effective_args.sig_effs = _configured_sig_effs(args, study_config)
    effective_args.pt_min = float(
        _configured_value(args.pt_min, study_config.study_defaults.get("pt_min"), DEFAULT_PT_MIN)
    )
    effective_args.eta_max = float(
        _configured_value(args.eta_max, study_config.study_defaults.get("eta_max"), DEFAULT_ETA_MAX)
    )
    effective_args.candidate_strategy = str(
        _configured_value(
            args.candidate_strategy,
            study_config.study_defaults.get("candidate_strategy"),
            DEFAULT_CANDIDATE_STRATEGY,
        )
    )
    if effective_args.candidate_strategy not in CANDIDATE_STRATEGIES:
        raise ValueError(
            "Config field 'study.candidate_strategy' must be one of: "
            + ", ".join(CANDIDATE_STRATEGIES)
        )
    effective_args.msd_window_low = float(
        _configured_value(
            args.msd_window_low,
            study_config.study_defaults.get("msd_window_low"),
            DEFAULT_MSD_WINDOW_LOW,
        )
    )
    effective_args.msd_window_high = float(
        _configured_value(
            args.msd_window_high,
            study_config.study_defaults.get("msd_window_high"),
            DEFAULT_MSD_WINDOW_HIGH,
        )
    )
    effective_args.weight_branch = str(
        _configured_value(
            args.weight_branch,
            study_config.study_defaults.get("weight_branch"),
            DEFAULT_WEIGHT_BRANCH,
        )
    )
    effective_args.disable_event_weights = bool(
        args.disable_event_weights or bool(study_config.study_defaults.get("disable_event_weights", False))
    )
    effective_args.uproot_step_size = str(
        _configured_value(
            args.uproot_step_size,
            study_config.study_defaults.get("uproot_step_size"),
            DEFAULT_UPROOT_STEP_SIZE,
        )
    )
    effective_args.score_hist_bins = int(
        _configured_value(
            args.score_hist_bins,
            study_config.study_defaults.get("score_hist_bins"),
            DEFAULT_SCORE_HIST_BINS,
        )
    )
    effective_args.plot_title_size = float(
        _configured_value(args.plot_title_size, study_config.plot_defaults.get("title_size"), DEFAULT_PLOT_OPTIONS["title_size"])
    )
    effective_args.plot_label_size = float(
        _configured_value(args.plot_label_size, study_config.plot_defaults.get("label_size"), DEFAULT_PLOT_OPTIONS["label_size"])
    )
    effective_args.plot_tick_size = float(
        _configured_value(args.plot_tick_size, study_config.plot_defaults.get("tick_size"), DEFAULT_PLOT_OPTIONS["tick_size"])
    )
    effective_args.plot_legend_size = float(
        _configured_value(
            args.plot_legend_size,
            study_config.plot_defaults.get("legend_size"),
            DEFAULT_PLOT_OPTIONS["legend_size"],
        )
    )
    effective_args.plot_cms_size = float(
        _configured_value(args.plot_cms_size, study_config.plot_defaults.get("cms_size"), DEFAULT_PLOT_OPTIONS["cms_size"])
    )
    effective_args.plot_dpi = int(
        _configured_value(args.plot_dpi, study_config.plot_defaults.get("dpi"), DEFAULT_PLOT_OPTIONS["dpi"])
    )
    return effective_args


def load_normalization_metadata(
    gen_sumw_file: str | None,
    xsec_file: str | None,
) -> tuple[dict[str, float], dict[str, float]]:
    gen_sumw_payload = load_json_maybe_with_comments(gen_sumw_file)
    xsec_payload = load_json_maybe_with_comments(xsec_file)

    gen_sumw_map: dict[str, float] = {}
    for dataset, value in gen_sumw_payload.items():
        if isinstance(value, dict):
            gen_sumw_map[dataset] = float(value["gen_sumw"])
        else:
            gen_sumw_map[dataset] = float(value)

    xsec_map = {dataset: float(value) for dataset, value in xsec_payload.items()}
    return gen_sumw_map, xsec_map


def _flatten_field(jets: ak.Array, field: str, n_selected: int, fill_value: float | int, dtype: np.dtype[Any] | str) -> np.ndarray:
    if field not in jets.fields:
        return np.full(n_selected, fill_value, dtype=dtype)
    return np.asarray(ak.to_numpy(ak.flatten(jets[field])), dtype=dtype)


def _count_jets(jets: ak.Array) -> int:
    if "pt" not in jets.fields:
        return 0
    return int(ak.sum(ak.num(jets.pt)))


def _select_candidates(jets: ak.Array, args: argparse.Namespace) -> ak.Array:
    mask = (jets.pt >= args.pt_min) & (np.abs(jets.eta) <= args.eta_max)
    jets = jets[mask]

    if args.candidate_strategy in {"mass_window_all_jets", "mass_window_leading_pt"}:
        msd_mask = (jets.msoftdrop >= args.msd_window_low) & (jets.msoftdrop <= args.msd_window_high)
        jets = jets[msd_mask]

    if args.candidate_strategy in {"all_jets", "mass_window_all_jets"}:
        return jets

    ordering = ak.argsort(jets.pt, axis=1, ascending=False)
    jets = jets[ordering]
    return ak.singletons(ak.firsts(jets))


def _broadcast_event_weights(jets: ak.Array, event_weights: ak.Array) -> np.ndarray:
    _, jet_event_weights = ak.broadcast_arrays(jets.pt, event_weights)
    return np.asarray(ak.to_numpy(ak.flatten(jet_event_weights)), dtype=float)


def _resolve_requested_scores(args: argparse.Namespace) -> list[str]:
    if args.scores == ["auto"]:
        requested: set[str] = set()
        for target in args.targets:
            requested.update(TARGET_DEFINITIONS[target]["default_scores"])
        return sorted(requested)
    return list(dict.fromkeys(args.scores))


def _required_fatjet_fields(args: argparse.Namespace) -> list[str]:
    required = {"pt", "eta"} | set(COUNT_FIELDS)
    if args.candidate_strategy in {"mass_window_all_jets", "mass_window_leading_pt"}:
        required.add("msoftdrop")
    for score_name in _resolve_requested_scores(args):
        required.update(SCORE_INPUT_FIELDS.get(score_name, set()))
    return [field for field in FATJET_FIELDS if field in required]


def get_sample_normalization(
    sample: SampleConfig,
    gen_sumw_map: dict[str, float],
    xsec_map: dict[str, float],
    lumi_fb: float | None,
) -> tuple[float, float | None, float | None]:
    gen_sumw = gen_sumw_map.get(sample.dataset)
    xsec_fb = xsec_map.get(sample.dataset)
    if lumi_fb is None or gen_sumw is None or xsec_fb is None:
        return 1.0, gen_sumw, xsec_fb
    if gen_sumw == 0:
        raise ValueError(f"gen_sumw is zero for dataset '{sample.dataset}'")
    return float(lumi_fb * xsec_fb / gen_sumw), gen_sumw, xsec_fb


def load_pepper_fatjets(
    samples: list[SampleConfig],
    args: argparse.Namespace,
    gen_sumw_map: dict[str, float],
    xsec_map: dict[str, float],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    requested_fatjet_fields = _required_fatjet_fields(args)
    chunks: dict[str, list[np.ndarray]] = {field: [] for field in requested_fatjet_fields}
    for key in ["weight", "weight_signed", "process_code"]:
        chunks[key] = []
    process_entries = _build_process_entries_from_samples(samples)
    process_to_code = {entry["process"]: int(entry["code"]) for entry in process_entries}

    sample_summaries: list[dict[str, Any]] = []

    for sample in samples:
        total_events = 0
        total_selected_jets = 0
        total_weight = 0.0
        total_signed_weight = 0.0
        skipped_files_missing_tree = 0
        sample_norm, gen_sumw, xsec_fb = get_sample_normalization(sample, gen_sumw_map, xsec_map, args.lumi_fb)
        weight_branch_found = False

        for file_path in sample.files:
            with uproot.open(file_path) as root_file:
                if args.tree_name not in root_file:
                    skipped_files_missing_tree += 1
                    continue
                tree = root_file[args.tree_name]
                total_events += int(tree.num_entries)

                requested = [f"FatJet_{field}" for field in requested_fatjet_fields if f"FatJet_{field}" in tree.keys()]
                if "FatJet_pt" not in requested or "FatJet_eta" not in requested:
                    raise KeyError(f"Missing required FatJet branches in {file_path}")
                if not args.disable_event_weights and args.weight_branch in tree.keys():
                    requested.append(args.weight_branch)
                    weight_branch_found = True

                for arrays in tree.iterate(requested, library="ak", step_size=args.uproot_step_size):
                    jets = ak.zip(
                        {
                            field: arrays[f"FatJet_{field}"]
                            for field in requested_fatjet_fields
                            if f"FatJet_{field}" in arrays.fields
                        }
                    )
                    jets = _select_candidates(jets, args)
                    n_selected = _count_jets(jets)
                    if n_selected == 0:
                        continue

                    total_selected_jets += n_selected

                    for field in requested_fatjet_fields:
                        if field in FLOAT_FIELDS:
                            chunks[field].append(_flatten_field(jets, field, n_selected, np.nan, FLOAT_STORAGE_DTYPE))
                        else:
                            chunks[field].append(_flatten_field(jets, field, n_selected, 0, COUNT_STORAGE_DTYPE))

                    if not args.disable_event_weights and args.weight_branch in arrays.fields:
                        raw_event_weight = _broadcast_event_weights(jets, arrays[args.weight_branch])
                    else:
                        raw_event_weight = np.ones(n_selected, dtype=float)

                    signed_weight = sample_norm * raw_event_weight
                    analysis_weight = sample_norm * np.abs(raw_event_weight)
                    total_weight += float(np.sum(analysis_weight))
                    total_signed_weight += float(np.sum(signed_weight))

                    chunks["weight"].append(np.asarray(analysis_weight, dtype=WEIGHT_STORAGE_DTYPE))
                    chunks["weight_signed"].append(np.asarray(signed_weight, dtype=WEIGHT_STORAGE_DTYPE))
                    chunks["process_code"].append(
                        np.full(n_selected, process_to_code[sample.process], dtype=PROCESS_CODE_DTYPE)
                    )

        sample_summaries.append(
            {
                "name": sample.name,
                "dataset": sample.dataset,
                "process": sample.process,
                "label": sample.label,
                "n_files": len(sample.files),
                "n_skipped_files_missing_tree": skipped_files_missing_tree,
                "n_events": total_events,
                "n_selected_jets": total_selected_jets,
                "gen_sumw": gen_sumw,
                "xsec_fb": xsec_fb,
                "sample_norm": sample_norm,
                "weight_branch": args.weight_branch if weight_branch_found else None,
                "analysis_weight_sum": total_weight,
                "signed_weight_sum": total_signed_weight,
            }
        )

    merged: dict[str, np.ndarray] = {}
    for key in list(chunks):
        value_chunks = chunks[key]
        if value_chunks:
            merged[key] = np.concatenate(value_chunks)
        else:
            if key in {"weight", "weight_signed"}:
                merged[key] = np.array([], dtype=WEIGHT_STORAGE_DTYPE)
            elif key == "process_code":
                merged[key] = np.array([], dtype=PROCESS_CODE_DTYPE)
            elif key in COUNT_FIELDS:
                merged[key] = np.array([], dtype=COUNT_STORAGE_DTYPE)
            else:
                merged[key] = np.array([], dtype=FLOAT_STORAGE_DTYPE)
        del chunks[key]

    return merged, sample_summaries


def add_truth_and_scores(data: dict[str, np.ndarray]) -> None:
    n_other_from_top = (
        data["n_t1b"]
        + data["n_t2b"]
        + data["n_t1w_c"]
        + data["n_t2w_c"]
        + data["n_t1w_uds"]
        + data["n_t2w_uds"]
        + data["n_t1w_lep"]
        + data["n_t2w_lep"]
    )
    is_top = (
        ((data["n_t1b"] >= 1) & (data["n_t1w"] >= 1))
        | ((data["n_t2b"] >= 1) & (data["n_t2w"] >= 1))
        | ((data["n_topb"] >= 1) & (data["n_topw"] >= 1))
    )
    truth_codes = np.full(data["pt"].shape, TRUTH_LABEL_TO_CODE["other"], dtype=TRUTH_CODE_DTYPE)
    truth_codes[is_top] = TRUTH_LABEL_TO_CODE["top"]
    truth_codes[data["n_hbb"] == 1] = TRUTH_LABEL_TO_CODE["hbb_partial"]
    truth_codes[(data["n_hbb"] >= 2) & (n_other_from_top == 0)] = TRUTH_LABEL_TO_CODE["hbb_pure"]
    truth_codes[(data["n_hbb"] >= 2) & (n_other_from_top > 0)] = TRUTH_LABEL_TO_CODE["hbb_contaminated"]
    truth_codes[data["n_hcc"] == 1] = TRUTH_LABEL_TO_CODE["hcc_partial"]
    truth_codes[(data["n_hcc"] >= 2) & (n_other_from_top == 0)] = TRUTH_LABEL_TO_CODE["hcc_pure"]
    truth_codes[(data["n_hcc"] >= 2) & (n_other_from_top > 0)] = TRUTH_LABEL_TO_CODE["hcc_contaminated"]

    data["n_other_from_top"] = n_other_from_top
    data["truth_code"] = truth_codes
    if {"globalParT3_Xcc", "globalParT3_QCD"} <= data.keys():
        data["gpart_h2cc"] = _safe_divide(data["globalParT3_Xcc"], data["globalParT3_Xcc"] + data["globalParT3_QCD"]).astype(
            FLOAT_STORAGE_DTYPE,
            copy=False,
        )
    if {"globalParT3_Xbb", "globalParT3_QCD"} <= data.keys():
        data["gpart_h2bb"] = _safe_divide(data["globalParT3_Xbb"], data["globalParT3_Xbb"] + data["globalParT3_QCD"]).astype(
            FLOAT_STORAGE_DTYPE,
            copy=False,
        )
    if {"globalParT3_Xbb", "globalParT3_Xcc"} <= data.keys():
        data["gpart_hbb_vs_hcc"] = _safe_divide(
            data["globalParT3_Xbb"],
            data["globalParT3_Xbb"] + data["globalParT3_Xcc"],
        ).astype(FLOAT_STORAGE_DTYPE, copy=False)
    if {"globalParT3_Xbb", "globalParT3_Xcc", "globalParT3_QCD"} <= data.keys():
        data["gpart_higgs_vs_qcd"] = _safe_divide(
            data["globalParT3_Xbb"] + data["globalParT3_Xcc"],
            data["globalParT3_Xbb"] + data["globalParT3_Xcc"] + data["globalParT3_QCD"],
        ).astype(FLOAT_STORAGE_DTYPE, copy=False)
    if "particleNetWithMass_HccvsQCD" in data:
        data["pnet_hcc"] = data["particleNetWithMass_HccvsQCD"].astype(FLOAT_STORAGE_DTYPE, copy=False)
    if "particleNet_XccVsQCD" in data:
        data["pnet_xcc"] = data["particleNet_XccVsQCD"].astype(FLOAT_STORAGE_DTYPE, copy=False)
    if "particleNetLegacy_Xcc" in data:
        data["pnetlegacy_xcc"] = data["particleNetLegacy_Xcc"].astype(FLOAT_STORAGE_DTYPE, copy=False)


def get_available_scores(data: dict[str, np.ndarray]) -> list[str]:
    available = []
    for score_name in SCORE_LABELS:
        if score_name in data and data[score_name].size > 0 and np.any(np.isfinite(data[score_name])):
            available.append(score_name)
    return available


def resolve_scores(requested_scores: list[str], target: str, available_scores: list[str]) -> list[str]:
    if requested_scores == ["auto"]:
        resolved = [score for score in TARGET_DEFINITIONS[target]["default_scores"] if score in available_scores]
    else:
        unknown = [score for score in requested_scores if score not in SCORE_LABELS]
        if unknown:
            raise ValueError(f"Unknown scores requested: {', '.join(sorted(unknown))}")
        resolved = [score for score in requested_scores if score in available_scores]
    if not resolved:
        raise ValueError(f"No usable scores found for target '{target}'. Available scores: {', '.join(available_scores)}")
    return resolved


def make_output_dirs(base: Path) -> dict[str, Path]:
    directories = {
        "base": base,
        "tables": base / "tables",
        "summaries": base / "summaries",
        "plots": base / "plots",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def resolve_weighting_info(args: argparse.Namespace) -> tuple[StudyConfig, argparse.Namespace, dict[str, Any]]:
    study_config = load_study_config(
        args.config,
        args.max_files_per_sample,
        default_xsec_file=str(REPO_ROOT / "crosssections_run3.json"),
    )
    effective_args = apply_config_defaults_to_args(args, study_config)
    effective_gen_sumw_file = args.gen_sumw_file or study_config.gen_sumw_file
    effective_xsec_file = args.xsec_file or study_config.xsec_file or str(REPO_ROOT / "crosssections_run3.json")
    effective_lumi_fb = args.lumi_fb if args.lumi_fb is not None else study_config.lumi_fb

    effective_args.lumi_fb = effective_lumi_fb
    weighting_info = {
        "lumi_fb": effective_lumi_fb,
        "gen_sumw_file": effective_gen_sumw_file,
        "xsec_file": effective_xsec_file,
        "weight_branch": None if args.disable_event_weights else args.weight_branch,
        "analysis_weight_definition": "sample_norm * abs(event_weight_raw)",
        "signed_weight_definition": "sample_norm * event_weight_raw",
    }
    return study_config, effective_args, weighting_info


def _select_scores_to_keep(args: argparse.Namespace, available_scores: list[str]) -> list[str]:
    selected = [score for score in _resolve_requested_scores(args) if score in available_scores]
    if not selected:
        raise ValueError(f"No requested scores are available. Available scores: {', '.join(available_scores)}")
    for score_name in [GLOBALPART3_CONTOUR_PLOT["x_score"], GLOBALPART3_CONTOUR_PLOT["y_score"]]:
        if score_name in available_scores and score_name not in selected:
            selected.append(score_name)
    return selected


def _build_slim_study_data(data: dict[str, np.ndarray], score_names: list[str]) -> dict[str, np.ndarray]:
    slim_data: dict[str, np.ndarray] = {
        "truth_code": np.asarray(data["truth_code"], dtype=TRUTH_CODE_DTYPE),
        "weight": np.asarray(data["weight"], dtype=WEIGHT_STORAGE_DTYPE),
        "weight_signed": np.asarray(data["weight_signed"], dtype=WEIGHT_STORAGE_DTYPE),
    }
    if "process_code" in data:
        slim_data["process_code"] = np.asarray(data["process_code"], dtype=PROCESS_CODE_DTYPE)
    for score_name in score_names:
        slim_data[score_name] = np.asarray(data[score_name], dtype=FLOAT_STORAGE_DTYPE)
    return slim_data


def _build_empty_slim_study_data(
    score_names: list[str],
    include_process_code: bool = True,
) -> dict[str, np.ndarray]:
    slim_data: dict[str, np.ndarray] = {
        "truth_code": np.array([], dtype=TRUTH_CODE_DTYPE),
        "weight": np.array([], dtype=WEIGHT_STORAGE_DTYPE),
        "weight_signed": np.array([], dtype=WEIGHT_STORAGE_DTYPE),
    }
    if include_process_code:
        slim_data["process_code"] = np.array([], dtype=PROCESS_CODE_DTYPE)
    for score_name in score_names:
        slim_data[score_name] = np.array([], dtype=FLOAT_STORAGE_DTYPE)
    return slim_data


def prepare_study_data_from_root(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], argparse.Namespace, dict[str, Any]]:
    study_config, effective_args, weighting_info = resolve_weighting_info(args)

    gen_sumw_map, xsec_map = load_normalization_metadata(
        gen_sumw_file=weighting_info["gen_sumw_file"],
        xsec_file=weighting_info["xsec_file"],
    )
    data, sample_summaries = load_pepper_fatjets(
        samples=study_config.samples,
        args=effective_args,
        gen_sumw_map=gen_sumw_map,
        xsec_map=xsec_map,
    )
    add_truth_and_scores(data)
    available_scores = get_available_scores(data)
    if data["truth_code"].size == 0:
        slim_data = _build_empty_slim_study_data(
            score_names=_resolve_requested_scores(effective_args),
            include_process_code="process_code" in data,
        )
    else:
        slim_data = _build_slim_study_data(data, _select_scores_to_keep(effective_args, available_scores))
    return slim_data, sample_summaries, effective_args, weighting_info


def apply_weighting_info_to_args(
    effective_args: argparse.Namespace,
    weighting_info: dict[str, Any],
) -> argparse.Namespace:
    if effective_args.lumi_fb is None:
        effective_args.lumi_fb = weighting_info.get("lumi_fb")
    if weighting_info.get("weight_branch") is not None:
        effective_args.weight_branch = weighting_info["weight_branch"]
    return effective_args


def render_plots(
    data: dict[str, np.ndarray],
    effective_args: argparse.Namespace,
    study_payload: dict[str, Any],
    outdirs: dict[str, Path],
) -> None:
    plot_style = build_plot_style(effective_args)
    available_scores = get_available_scores(data)
    process_entries = _build_process_entries_from_summaries(study_payload.get("sample_summaries", []))
    has_process_payload = (
        "process_code" in data
        and data["process_code"].shape == data["truth_code"].shape
        and len(process_entries) > 0
    )
    contour_x_score = GLOBALPART3_CONTOUR_PLOT["x_score"]
    contour_y_score = GLOBALPART3_CONTOUR_PLOT["y_score"]
    if contour_x_score in data and contour_y_score in data:
        plot_globalpart3_contours(
            x_scores=data[contour_x_score],
            y_scores=data[contour_y_score],
            truth_codes=data["truth_code"],
            weights=data["weight"],
            outpath=outdirs["plots"] / f"{GLOBALPART3_CONTOUR_PLOT['filename_stem']}.png",
            plot_style=plot_style,
        )
        study_payload["globalpart3_contours"] = {
            "x_score": contour_x_score,
            "y_score": contour_y_score,
            "filename_stem": GLOBALPART3_CONTOUR_PLOT["filename_stem"],
        }

    for target in effective_args.targets:
        resolved_scores = resolve_scores(effective_args.scores, target, available_scores)
        target_payload = study_payload["targets"].setdefault(target, {})
        for score_name in resolved_scores:
            score_payload = target_payload.setdefault(score_name, {})
            plot_score_distribution(
                scores=data[score_name],
                truth_codes=data["truth_code"],
                weights=data["weight"],
                target=target,
                score_name=score_name,
                outpath=outdirs["plots"] / f"{target}__{score_name}__score.png",
                plot_style=plot_style,
            )
            roc_payload = compute_roc(
                scores=data[score_name],
                truth_codes=data["truth_code"],
                weights=data["weight"],
                target=target,
            )
            plot_roc_curve(
                roc_payload=roc_payload,
                target=target,
                score_name=score_name,
                outpath=outdirs["plots"] / f"{target}__{score_name}__roc.png",
                plot_style=plot_style,
            )
            score_payload["roc"] = {
                "auc": float(roc_payload["auc"]),
                "n_points": int(len(np.asarray(roc_payload["sig_eff"]))),
            }
            rows = score_payload.get("working_points")
            if rows is None:
                rows, _ = compute_working_points(
                    scores=data[score_name],
                    truth_codes=data["truth_code"],
                    weights=data["weight"],
                    signed_weights=data["weight_signed"],
                    target=target,
                    sig_effs=effective_args.sig_effs,
                )
                score_payload["working_points"] = rows
            plot_significance_scan(
                rows=rows,
                target=target,
                score_name=score_name,
                outpath=outdirs["plots"] / f"{target}__{score_name}__significance_scan.png",
                plot_style=plot_style,
            )
            if has_process_payload:
                plot_background_process_score_distribution(
                    scores=data[score_name],
                    truth_codes=data["truth_code"],
                    process_codes=data["process_code"],
                    weights=data["weight"],
                    target=target,
                    score_name=score_name,
                    process_entries=process_entries,
                    outpath=outdirs["plots"] / f"{target}__{score_name}__background_process_score.png",
                    plot_style=plot_style,
                )
                plot_background_process_working_points(
                    scores=data[score_name],
                    truth_codes=data["truth_code"],
                    process_codes=data["process_code"],
                    weights=data["weight"],
                    target=target,
                    score_name=score_name,
                    sig_effs=effective_args.sig_effs,
                    process_entries=process_entries,
                    outpath=outdirs["plots"] / f"{target}__{score_name}__background_process_wp.png",
                    plot_style=plot_style,
                )
                target_payload[score_name]["process_groups"] = [entry["process"] for entry in process_entries]


def render_histogram_plots(
    histogram_payload: dict[str, Any],
    effective_args: argparse.Namespace,
    study_payload: dict[str, Any],
    outdirs: dict[str, Path],
) -> None:
    plot_style = build_plot_style(effective_args)
    available_scores = list(histogram_payload["available_scores"])
    hist_edges = np.asarray(histogram_payload["hist_edges"], dtype=np.float64)
    process_entries = list(histogram_payload.get("process_entries", []))
    contour_payload = histogram_payload.get("contour_payloads", {}).get(GLOBALPART3_CONTOUR_PLOT["key"])
    if contour_payload is not None:
        plot_globalpart3_contours_from_hist(
            contour_payload=contour_payload,
            outpath=outdirs["plots"] / f"{GLOBALPART3_CONTOUR_PLOT['filename_stem']}.png",
            plot_style=plot_style,
        )
        study_payload["globalpart3_contours"] = {
            "x_score": contour_payload.get("x_score", GLOBALPART3_CONTOUR_PLOT["x_score"]),
            "y_score": contour_payload.get("y_score", GLOBALPART3_CONTOUR_PLOT["y_score"]),
            "filename_stem": contour_payload.get("filename_stem", GLOBALPART3_CONTOUR_PLOT["filename_stem"]),
        }

    for target in effective_args.targets:
        resolved_scores = resolve_scores(effective_args.scores, target, available_scores)
        target_payload = study_payload["targets"].setdefault(target, {})
        for score_name in resolved_scores:
            score_hist = histogram_payload["score_histograms"][score_name]
            score_payload = target_payload.setdefault(score_name, {})
            plot_score_distribution_from_hist(
                score_payload=score_hist,
                hist_edges=hist_edges,
                target=target,
                score_name=score_name,
                outpath=outdirs["plots"] / f"{target}__{score_name}__score.png",
                plot_style=plot_style,
            )
            roc_payload = compute_roc_from_hist(
                score_payload=score_hist,
                hist_edges=hist_edges,
                target=target,
            )
            plot_roc_curve(
                roc_payload=roc_payload,
                target=target,
                score_name=score_name,
                outpath=outdirs["plots"] / f"{target}__{score_name}__roc.png",
                plot_style=plot_style,
            )
            score_payload["roc"] = {
                "auc": float(roc_payload["auc"]),
                "n_points": int(len(np.asarray(roc_payload["sig_eff"]))),
            }
            rows = score_payload.get("working_points")
            if rows is None:
                rows, _ = compute_working_points_from_hist(
                    score_payload=score_hist,
                    hist_edges=hist_edges,
                    target=target,
                    sig_effs=effective_args.sig_effs,
                )
                score_payload["working_points"] = rows
            plot_significance_scan(
                rows=rows,
                target=target,
                score_name=score_name,
                outpath=outdirs["plots"] / f"{target}__{score_name}__significance_scan.png",
                plot_style=plot_style,
            )
            if process_entries:
                plot_background_process_score_distribution_from_hist(
                    score_payload=score_hist,
                    hist_edges=hist_edges,
                    target=target,
                    score_name=score_name,
                    process_entries=process_entries,
                    outpath=outdirs["plots"] / f"{target}__{score_name}__background_process_score.png",
                    plot_style=plot_style,
                )
                plot_background_process_working_points_from_hist(
                    score_payload=score_hist,
                    hist_edges=hist_edges,
                    target=target,
                    score_name=score_name,
                    sig_effs=effective_args.sig_effs,
                    process_entries=process_entries,
                    outpath=outdirs["plots"] / f"{target}__{score_name}__background_process_wp.png",
                    plot_style=plot_style,
                )
                target_payload[score_name]["process_groups"] = [entry["process"] for entry in process_entries]


def _globalpart3_region_output_paths(outdirs: dict[str, Path]) -> tuple[Path, Path]:
    stem = f"{GLOBALPART3_CONTOUR_PLOT['filename_stem']}__region_efficiencies"
    return outdirs["summaries"] / f"{stem}.txt", outdirs["summaries"] / f"{stem}.json"


def _globalpart3_region_metadata(region_efficiencies: dict[str, dict[str, float]]) -> dict[str, Any]:
    return {
        "x_score": GLOBALPART3_CONTOUR_PLOT["x_score"],
        "y_score": GLOBALPART3_CONTOUR_PLOT["y_score"],
        "weight_definition": "analysis_weight = sample_norm * abs(event_weight_raw)",
        "normalization": "weighted yield in region / total weighted yield of that category",
        "region_definitions": {
            "hcc": {
                "selection": "(x > 0.45) and (0.0 < y <= 0.7)",
                "x_min_exclusive": 0.45,
                "y_min_exclusive": 0.0,
                "y_max_inclusive": 0.7,
            },
            "hbb": {
                "selection": "(x > 0.75) and (0.7 < y <= 1.0)",
                "x_min_exclusive": 0.75,
                "y_min_exclusive": 0.7,
                "y_max_inclusive": 1.0,
            },
            "qcd_others": {
                "selection": "not(Hcc region or Hbb region)",
            },
        },
        "region_efficiencies": region_efficiencies,
    }


def _write_globalpart3_region_outputs(
    region_efficiencies: dict[str, dict[str, float]],
    study_payload: dict[str, Any],
    outdirs: dict[str, Path],
) -> None:
    if not region_efficiencies:
        return
    contour_payload = study_payload.setdefault(
        "globalpart3_contours",
        {
            "x_score": GLOBALPART3_CONTOUR_PLOT["x_score"],
            "y_score": GLOBALPART3_CONTOUR_PLOT["y_score"],
            "filename_stem": GLOBALPART3_CONTOUR_PLOT["filename_stem"],
        },
    )
    contour_payload["region_efficiencies"] = region_efficiencies
    region_summary = _globalpart3_region_metadata(region_efficiencies)
    txt_path, json_path = _globalpart3_region_output_paths(outdirs)
    txt_path.write_text(format_contour_region_efficiency_text(region_efficiencies), encoding="utf-8")
    write_json(json_path, region_summary)


def maybe_write_globalpart3_region_outputs_from_raw(
    data: dict[str, np.ndarray],
    study_payload: dict[str, Any],
    outdirs: dict[str, Path],
) -> None:
    contour_x_score = GLOBALPART3_CONTOUR_PLOT["x_score"]
    contour_y_score = GLOBALPART3_CONTOUR_PLOT["y_score"]
    if contour_x_score not in data or contour_y_score not in data:
        return
    if "truth_code" not in data or "weight" not in data:
        return
    region_efficiencies = compute_globalpart3_region_efficiencies_from_raw(
        x_scores=np.asarray(data[contour_x_score], dtype=np.float64),
        y_scores=np.asarray(data[contour_y_score], dtype=np.float64),
        truth_codes=np.asarray(data["truth_code"], dtype=np.int32),
        weights=np.asarray(data["weight"], dtype=np.float64),
    )
    _write_globalpart3_region_outputs(region_efficiencies, study_payload, outdirs)


def maybe_write_globalpart3_region_outputs_from_hist(
    histogram_payload: dict[str, Any],
    study_payload: dict[str, Any],
    outdirs: dict[str, Path],
) -> None:
    contour_payload = histogram_payload.get("contour_payloads", {}).get(GLOBALPART3_CONTOUR_PLOT["key"])
    if contour_payload is None:
        return
    region_efficiencies = compute_globalpart3_region_efficiencies_from_hist_payload(contour_payload)
    _write_globalpart3_region_outputs(region_efficiencies, study_payload, outdirs)


def finalize_study(
    data: dict[str, np.ndarray],
    sample_summaries: list[dict[str, Any]],
    effective_args: argparse.Namespace,
    weighting_info: dict[str, Any],
) -> dict[str, Any]:
    available_scores = get_available_scores(data)

    outdirs = make_output_dirs(Path(effective_args.outdir))
    study_payload: dict[str, Any] = {
        "available_scores": available_scores,
        "sample_summaries": sample_summaries,
        "process_entries": _build_process_entries_from_summaries(sample_summaries),
        "weighting": weighting_info,
        "targets": {},
    }
    export_chunk_payload(outdirs["base"] / "plot_input.npz", data, sample_summaries, weighting_info)
    maybe_write_globalpart3_region_outputs_from_raw(data, study_payload, outdirs)

    for target in effective_args.targets:
        resolved_scores = resolve_scores(effective_args.scores, target, available_scores)
        study_payload["targets"][target] = {}
        for score_name in resolved_scores:
            rows, counts_by_label = compute_working_points(
                scores=data[score_name],
                truth_codes=data["truth_code"],
                weights=data["weight"],
                signed_weights=data["weight_signed"],
                target=target,
                sig_effs=effective_args.sig_effs,
            )

            stem = f"{target}__{score_name}"
            write_csv(outdirs["tables"] / f"{stem}.csv", rows)
            summary_text = format_summary_text(target, score_name, rows, counts_by_label, sample_summaries, effective_args)
            (outdirs["summaries"] / f"{stem}.txt").write_text(summary_text, encoding="utf-8")

            score_payload: dict[str, Any] = {
                "counts_by_label": counts_by_label,
                "working_points": rows,
            }
            study_payload["targets"][target][score_name] = score_payload

    if not effective_args.skip_plots:
        render_plots(data, effective_args, study_payload, outdirs)

    write_json(outdirs["base"] / "study_summary.json", study_payload)
    return study_payload


def finalize_plots_only(
    data: dict[str, np.ndarray],
    sample_summaries: list[dict[str, Any]],
    effective_args: argparse.Namespace,
    weighting_info: dict[str, Any],
) -> dict[str, Any]:
    outdirs = make_output_dirs(Path(effective_args.outdir))
    study_payload: dict[str, Any] = {
        "available_scores": get_available_scores(data),
        "sample_summaries": sample_summaries,
        "process_entries": _build_process_entries_from_summaries(sample_summaries),
        "weighting": weighting_info,
        "targets": {},
        "mode": "plot_only",
    }
    maybe_write_globalpart3_region_outputs_from_raw(data, study_payload, outdirs)
    render_plots(data, effective_args, study_payload, outdirs)
    write_json(outdirs["base"] / "plot_only_summary.json", study_payload)
    return study_payload


def finalize_histogram_study(
    histogram_payload: dict[str, Any],
    sample_summaries: list[dict[str, Any]],
    effective_args: argparse.Namespace,
    weighting_info: dict[str, Any],
) -> dict[str, Any]:
    available_scores = list(histogram_payload["available_scores"])
    outdirs = make_output_dirs(Path(effective_args.outdir))
    study_payload: dict[str, Any] = {
        "payload_mode": HISTOGRAM_PAYLOAD_MODE,
        "available_scores": available_scores,
        "sample_summaries": sample_summaries,
        "process_entries": histogram_payload.get("process_entries", []),
        "weighting": weighting_info,
        "targets": {},
    }
    export_histogram_payload(outdirs["base"] / "plot_input.npz", histogram_payload, sample_summaries, weighting_info)
    maybe_write_globalpart3_region_outputs_from_hist(histogram_payload, study_payload, outdirs)

    hist_edges = np.asarray(histogram_payload["hist_edges"], dtype=np.float64)
    for target in effective_args.targets:
        resolved_scores = resolve_scores(effective_args.scores, target, available_scores)
        study_payload["targets"][target] = {}
        for score_name in resolved_scores:
            rows, counts_by_label = compute_working_points_from_hist(
                score_payload=histogram_payload["score_histograms"][score_name],
                hist_edges=hist_edges,
                target=target,
                sig_effs=effective_args.sig_effs,
            )
            stem = f"{target}__{score_name}"
            write_csv(outdirs["tables"] / f"{stem}.csv", rows)
            summary_text = format_summary_text(target, score_name, rows, counts_by_label, sample_summaries, effective_args)
            (outdirs["summaries"] / f"{stem}.txt").write_text(summary_text, encoding="utf-8")
            study_payload["targets"][target][score_name] = {
                "counts_by_label": counts_by_label,
                "working_points": rows,
            }

    if not effective_args.skip_plots:
        render_histogram_plots(histogram_payload, effective_args, study_payload, outdirs)

    write_json(outdirs["base"] / "study_summary.json", study_payload)
    return study_payload


def finalize_histogram_plots_only(
    histogram_payload: dict[str, Any],
    sample_summaries: list[dict[str, Any]],
    effective_args: argparse.Namespace,
    weighting_info: dict[str, Any],
) -> dict[str, Any]:
    outdirs = make_output_dirs(Path(effective_args.outdir))
    study_payload: dict[str, Any] = {
        "payload_mode": HISTOGRAM_PAYLOAD_MODE,
        "available_scores": list(histogram_payload["available_scores"]),
        "sample_summaries": sample_summaries,
        "process_entries": histogram_payload.get("process_entries", []),
        "weighting": weighting_info,
        "targets": {},
        "mode": "plot_only",
    }
    maybe_write_globalpart3_region_outputs_from_hist(histogram_payload, study_payload, outdirs)
    render_histogram_plots(histogram_payload, effective_args, study_payload, outdirs)
    write_json(outdirs["base"] / "plot_only_summary.json", study_payload)
    return study_payload


def run_export_chunk(args: argparse.Namespace) -> dict[str, Any]:
    data, sample_summaries, effective_args, weighting_info = prepare_study_data_from_root(args)
    chunk_path = Path(args.export_chunk).resolve()
    available_scores = get_available_scores(data)
    if data["truth_code"].size == 0:
        selected_scores = _resolve_requested_scores(effective_args)
    else:
        selected_scores = _select_scores_to_keep(effective_args, available_scores)
    if args.chunk_payload_mode == "histogram":
        histogram_payload = build_histogram_payload_from_raw_data(
            data=data,
            sample_summaries=sample_summaries,
            score_names=selected_scores,
            n_bins=effective_args.score_hist_bins,
        )
        exported_scores = export_histogram_payload(
            outpath=chunk_path,
            histogram_payload=histogram_payload,
            sample_summaries=sample_summaries,
            weighting_info=weighting_info,
        )
    else:
        exported_scores = export_chunk_payload(
            outpath=chunk_path,
            data=data,
            sample_summaries=sample_summaries,
            weighting_info=weighting_info,
        )
    return {
        "mode": "export_chunk",
        "chunk_path": str(chunk_path),
        "n_jets": int(data["truth_code"].size),
        "available_scores": exported_scores,
        "targets": effective_args.targets,
    }


def run_merge_chunks(args: argparse.Namespace) -> dict[str, Any]:
    _, effective_args, _ = resolve_weighting_info(args)
    payload_mode = _detect_payload_mode(args.merge_chunks or [])
    if payload_mode == HISTOGRAM_PAYLOAD_MODE:
        histogram_payload, sample_summaries, weighting_info = load_merged_histogram_payloads(args.merge_chunks or [])
        effective_args = apply_weighting_info_to_args(effective_args, weighting_info)
        return finalize_histogram_study(histogram_payload, sample_summaries, effective_args, weighting_info)
    data, sample_summaries, weighting_info = load_merged_chunk_payloads(args.merge_chunks or [])
    effective_args = apply_weighting_info_to_args(effective_args, weighting_info)
    return finalize_study(data, sample_summaries, effective_args, weighting_info)


def run_plot_only(args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_plots:
        raise ValueError("--plot-only cannot be combined with --skip-plots.")

    _, effective_args, _ = resolve_weighting_info(args)
    if args.plot_input is not None:
        payload_mode = _detect_payload_mode([args.plot_input])
        if payload_mode == HISTOGRAM_PAYLOAD_MODE:
            histogram_payload, sample_summaries, weighting_info = load_merged_histogram_payloads([args.plot_input])
            effective_args = apply_weighting_info_to_args(effective_args, weighting_info)
            return finalize_histogram_plots_only(histogram_payload, sample_summaries, effective_args, weighting_info)
        data, sample_summaries, weighting_info = load_merged_chunk_payloads([args.plot_input])
        effective_args = apply_weighting_info_to_args(effective_args, weighting_info)
        return finalize_plots_only(data, sample_summaries, effective_args, weighting_info)

    if args.merge_chunks:
        payload_mode = _detect_payload_mode(args.merge_chunks)
        if payload_mode == HISTOGRAM_PAYLOAD_MODE:
            histogram_payload, sample_summaries, weighting_info = load_merged_histogram_payloads(args.merge_chunks)
            effective_args = apply_weighting_info_to_args(effective_args, weighting_info)
            return finalize_histogram_plots_only(histogram_payload, sample_summaries, effective_args, weighting_info)
        data, sample_summaries, weighting_info = load_merged_chunk_payloads(args.merge_chunks)
        effective_args = apply_weighting_info_to_args(effective_args, weighting_info)
        return finalize_plots_only(data, sample_summaries, effective_args, weighting_info)

    data, sample_summaries, effective_args, weighting_info = prepare_study_data_from_root(args)
    return finalize_plots_only(data, sample_summaries, effective_args, weighting_info)


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    data, sample_summaries, effective_args, weighting_info = prepare_study_data_from_root(args)
    return finalize_study(data, sample_summaries, effective_args, weighting_info)


def main() -> None:
    args = parse_args()
    args.outdir = resolve_output_dir(args.config, args.outdir, REPO_ROOT)
    if args.export_chunk and args.merge_chunks:
        raise ValueError("--export-chunk and --merge-chunks cannot be used at the same time.")
    if args.export_chunk and args.plot_input:
        raise ValueError("--export-chunk and --plot-input cannot be used at the same time.")
    if args.merge_chunks and args.plot_input:
        raise ValueError("--merge-chunks and --plot-input cannot be used at the same time.")

    if args.export_chunk:
        payload = run_export_chunk(args)
        print("Boosted Higgs tagger chunk export finished.")
        print(f"Chunk output: {payload['chunk_path']}")
        print(f"Selected jets: {payload['n_jets']}")
        print("Available scores:", ", ".join(payload["available_scores"]))
        return

    if args.plot_only:
        payload = run_plot_only(args)
        print("Boosted Higgs tagger plot-only refresh finished.")
    elif args.merge_chunks:
        payload = run_merge_chunks(args)
    else:
        payload = run_study(args)

    print("Boosted Higgs tagger study finished.")
    print(f"Output directory: {Path(payload.get('outdir', args.outdir)).resolve()}")
    print("Available scores:", ", ".join(payload["available_scores"]))
    for target, score_map in payload["targets"].items():
        print(f"Target {target}: {', '.join(score_map.keys())}")


if __name__ == "__main__":
    main()
