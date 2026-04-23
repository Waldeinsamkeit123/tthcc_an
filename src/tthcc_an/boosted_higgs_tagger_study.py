from __future__ import annotations

import argparse
import csv
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import awkward as ak
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
import uproot


REPO_ROOT = Path(__file__).resolve().parents[2]
FLOAT_STORAGE_DTYPE = np.float32
COUNT_STORAGE_DTYPE = np.int16
WEIGHT_STORAGE_DTYPE = np.float32
TRUTH_CODE_DTYPE = np.int8
PROCESS_CODE_DTYPE = np.int16
RAW_PAYLOAD_MODE = "raw_v1"
HISTOGRAM_PAYLOAD_MODE = "histogram_v1"
DEFAULT_SCORE_HIST_BINS = 2000
DISPLAY_SCORE_BINS = 50
DISPLAY_PROCESS_SCORE_BINS = 50

FATJET_FIELDS = [
    "pt",
    "eta",
    "phi",
    "mass",
    "msoftdrop",
    "n_hbb",
    "n_hcc",
    "n_t1b",
    "n_t2b",
    "n_t1w",
    "n_t2w",
    "n_t1w_c",
    "n_t1w_uds",
    "n_t1w_lep",
    "n_t2w_c",
    "n_t2w_uds",
    "n_t2w_lep",
    "n_topb",
    "n_topw",
    "n_topw_c",
    "n_topw_uds",
    "n_topw_lep",
    "n_nontopw",
    "n_nontopw_c",
    "n_nontopw_uds",
    "n_nontopw_lep",
    "globalParT3_QCD",
    "globalParT3_TopbWq",
    "globalParT3_TopbWqq",
    "globalParT3_Xbb",
    "globalParT3_Xcc",
    "particleNetWithMass_HccvsQCD",
    "particleNetLegacy_Xcc",
    "particleNet_XccVsQCD",
]

COUNT_FIELDS = [
    "n_hbb",
    "n_hcc",
    "n_t1b",
    "n_t2b",
    "n_t1w",
    "n_t2w",
    "n_t1w_c",
    "n_t1w_uds",
    "n_t1w_lep",
    "n_t2w_c",
    "n_t2w_uds",
    "n_t2w_lep",
    "n_topb",
    "n_topw",
    "n_topw_c",
    "n_topw_uds",
    "n_topw_lep",
    "n_nontopw",
    "n_nontopw_c",
    "n_nontopw_uds",
    "n_nontopw_lep",
]

FLOAT_FIELDS = [field for field in FATJET_FIELDS if field not in COUNT_FIELDS]

TRUTH_LABEL_ORDER = [
    "hcc_pure",
    "hcc_contaminated",
    "hcc_partial",
    "hbb_pure",
    "hbb_contaminated",
    "hbb_partial",
    "top",
    "other",
]

TRUTH_LABEL_TO_CODE = {label: index for index, label in enumerate(TRUTH_LABEL_ORDER)}

TRUTH_LABEL_TITLES = {
    "hcc_pure": r"$H\to cc$ pure",
    "hcc_contaminated": r"$H\to cc$ contaminated",
    "hcc_partial": r"$H\to cc$ partial",
    "hbb_pure": r"$H\to bb$ pure",
    "hbb_contaminated": r"$H\to bb$ contaminated",
    "hbb_partial": r"$H\to bb$ partial",
    "top": "top fatjet",
    "other": "other",
}

TRUTH_LABEL_COLORS = {
    "hcc_pure": "#d62728",
    "hcc_contaminated": "#ff9896",
    "hcc_partial": "#c44e52",
    "hbb_pure": "#1f77b4",
    "hbb_contaminated": "#9ecae1",
    "hbb_partial": "#4e79a7",
    "top": "#2ca02c",
    "other": "#7f7f7f",
}

PROCESS_DISPLAY_ORDER = [
    "ttHcc",
    "ttHbb",
    "ttH",
    "ttbar",
    "ttbb",
    "ttv",
    "ttll",
    "single_top",
    "wjets",
    "zjets",
    "qcd",
]

PROCESS_COLORS = {
    "ttHcc": "#d62728",
    "ttHbb": "#1f77b4",
    "ttH": "#9467bd",
    "ttbar": "#ff7f0e",
    "ttbb": "#8c564b",
    "ttv": "#e377c2",
    "ttll": "#bcbd22",
    "single_top": "#17becf",
    "wjets": "#2ca02c",
    "zjets": "#98df8a",
    "qcd": "#7f7f7f",
}

TARGET_DEFINITIONS = {
    "hcc": {
        "title": r"$H\to cc$",
        "signal_labels": ["hcc_pure", "hcc_contaminated"],
        "background_labels": [
            "hcc_partial",
            "hbb_pure",
            "hbb_contaminated",
            "hbb_partial",
            "top",
            "other",
        ],
        "default_scores": ["gpart_h2cc"],
    },
    "hbb": {
        "title": r"$H\to bb$",
        "signal_labels": ["hbb_pure", "hbb_contaminated"],
        "background_labels": [
            "hbb_partial",
            "hcc_pure",
            "hcc_contaminated",
            "hcc_partial",
            "top",
            "other",
        ],
        "default_scores": ["gpart_h2bb"],
    },
    "top": {
        "title": "top",
        "signal_labels": ["top"],
        "background_labels": [
            "hcc_pure",
            "hcc_contaminated",
            "hcc_partial",
            "hbb_pure",
            "hbb_contaminated",
            "hbb_partial",
            "other",
        ],
        "default_scores": ["gpart_top_wq", "gpart_top_wqq"],
    },
}

SCORE_LABELS = {
    "gpart_h2cc": r"gParT3 $H\to cc$ score",
    "gpart_h2bb": r"gParT3 $H\to bb$ score",
    "gpart_top_wq": r"gParT3 top($bWq$) vs QCD",
    "gpart_top_wqq": r"gParT3 top($bWqq$) vs QCD",
    "pnet_hcc": r"ParticleNetWithMass $Hcc$ vs QCD",
    "pnet_xcc": r"ParticleNet $Xcc$ vs QCD",
    "pnetlegacy_xcc": r"ParticleNetLegacy $Xcc$",
}

SCORE_INPUT_FIELDS = {
    "gpart_h2cc": {"globalParT3_Xcc", "globalParT3_QCD"},
    "gpart_h2bb": {"globalParT3_Xbb", "globalParT3_QCD"},
    "gpart_top_wq": {"globalParT3_TopbWq", "globalParT3_QCD"},
    "gpart_top_wqq": {"globalParT3_TopbWqq", "globalParT3_QCD"},
    "pnet_hcc": {"particleNetWithMass_HccvsQCD"},
    "pnet_xcc": {"particleNet_XccVsQCD"},
    "pnetlegacy_xcc": {"particleNetLegacy_Xcc"},
}


@dataclass
class SampleConfig:
    name: str
    dataset: str
    process: str
    label: str
    files: list[str]


@dataclass
class StudyConfig:
    samples: list[SampleConfig]
    gen_sumw_file: str | None
    xsec_file: str | None
    lumi_fb: float | None


@dataclass
class PlotStyle:
    title_size: float
    label_size: float
    tick_size: float
    legend_size: float
    cms_size: float
    dpi: int


def _process_sort_key(process: str) -> tuple[int, str]:
    if process in PROCESS_DISPLAY_ORDER:
        return (PROCESS_DISPLAY_ORDER.index(process), process)
    return (len(PROCESS_DISPLAY_ORDER), process)


def _process_color(process: str, fallback_index: int) -> str:
    if process in PROCESS_COLORS:
        return PROCESS_COLORS[process]
    cmap = plt.get_cmap("tab20")
    return str(matplotlib.colors.to_hex(cmap(fallback_index % cmap.N)))


def _build_process_entries_from_pairs(process_to_label: dict[str, str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for code, (process, label) in enumerate(sorted(process_to_label.items(), key=lambda item: _process_sort_key(item[0]))):
        entries.append(
            {
                "code": code,
                "process": process,
                "label": label,
                "color": _process_color(process, code),
            }
        )
    return entries


def _build_process_entries_from_samples(samples: list["SampleConfig"]) -> list[dict[str, Any]]:
    process_to_label: dict[str, str] = {}
    for sample in samples:
        process_to_label.setdefault(sample.process, sample.label)
    return _build_process_entries_from_pairs(process_to_label)


def _build_process_entries_from_summaries(sample_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    process_to_label: dict[str, str] = {}
    for summary in sample_summaries:
        process_to_label.setdefault(summary["process"], summary["label"])
    return _build_process_entries_from_pairs(process_to_label)


def _score_hist_edges(n_bins: int) -> np.ndarray:
    if n_bins <= 0:
        raise ValueError("--score-hist-bins must be positive.")
    return np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)


def _score_to_hist_bins(scores: np.ndarray, n_bins: int) -> np.ndarray:
    clipped = np.clip(scores, 0.0, 1.0)
    return np.minimum((clipped * n_bins).astype(np.int32), n_bins - 1)


def _payload_array_key(score_name: str, kind: str) -> str:
    return f"{score_name}__{kind}"


def _payload_mode_from_metadata(metadata: dict[str, Any]) -> str:
    return str(metadata.get("payload_mode", RAW_PAYLOAD_MODE))


def _load_payload_metadata(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as payload:
        return json.loads(np.asarray(payload["metadata_json"]).item())


def _detect_payload_mode(chunk_patterns: list[str]) -> str:
    chunk_paths = [Path(path) for path in _expand_file_patterns(chunk_patterns)]
    if not chunk_paths:
        raise ValueError("No chunk NPZ files were found.")
    return _payload_mode_from_metadata(_load_payload_metadata(chunk_paths[0]))


def _score_histograms_from_arrays(
    scores: np.ndarray,
    truth_codes: np.ndarray,
    weights: np.ndarray,
    signed_weights: np.ndarray,
    process_codes: np.ndarray | None,
    n_bins: int,
    n_processes: int,
) -> dict[str, np.ndarray]:
    valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    count_truth = np.zeros((len(TRUTH_LABEL_ORDER), n_bins), dtype=np.int64)
    weight_truth = np.zeros((len(TRUTH_LABEL_ORDER), n_bins), dtype=np.float64)
    signed_truth = np.zeros((len(TRUTH_LABEL_ORDER), n_bins), dtype=np.float64)
    weight_process_truth = np.zeros((n_processes, len(TRUTH_LABEL_ORDER), n_bins), dtype=np.float64)
    if not np.any(valid):
        return {
            "count_truth": count_truth,
            "weight_truth": weight_truth,
            "signed_truth": signed_truth,
            "weight_process_truth": weight_process_truth,
        }

    truth_valid = truth_codes[valid].astype(np.int32, copy=False)
    bins_valid = _score_to_hist_bins(scores[valid], n_bins)
    weights_valid = weights[valid].astype(np.float64, copy=False)
    signed_valid = signed_weights[valid].astype(np.float64, copy=False)
    np.add.at(count_truth, (truth_valid, bins_valid), 1)
    np.add.at(weight_truth, (truth_valid, bins_valid), weights_valid)
    np.add.at(signed_truth, (truth_valid, bins_valid), signed_valid)

    if process_codes is not None and n_processes > 0:
        process_valid = process_codes[valid].astype(np.int32, copy=False)
        np.add.at(weight_process_truth, (process_valid, truth_valid, bins_valid), weights_valid)

    return {
        "count_truth": count_truth,
        "weight_truth": weight_truth,
        "signed_truth": signed_truth,
        "weight_process_truth": weight_process_truth,
    }


def build_histogram_payload_from_raw_data(
    data: dict[str, np.ndarray],
    sample_summaries: list[dict[str, Any]],
    score_names: list[str],
    n_bins: int,
) -> dict[str, Any]:
    process_entries = _build_process_entries_from_summaries(sample_summaries)
    n_processes = len(process_entries)
    process_codes = data.get("process_code")
    score_histograms: dict[str, dict[str, np.ndarray]] = {}
    for score_name in score_names:
        score_histograms[score_name] = _score_histograms_from_arrays(
            scores=np.asarray(data[score_name], dtype=np.float64),
            truth_codes=np.asarray(data["truth_code"], dtype=TRUTH_CODE_DTYPE),
            weights=np.asarray(data["weight"], dtype=np.float64),
            signed_weights=np.asarray(data["weight_signed"], dtype=np.float64),
            process_codes=None if process_codes is None else np.asarray(process_codes, dtype=PROCESS_CODE_DTYPE),
            n_bins=n_bins,
            n_processes=n_processes,
        )
    return {
        "payload_mode": HISTOGRAM_PAYLOAD_MODE,
        "hist_edges": _score_hist_edges(n_bins),
        "available_scores": list(score_names),
        "process_entries": process_entries,
        "score_histograms": score_histograms,
    }


def export_histogram_payload(
    outpath: Path,
    histogram_payload: dict[str, Any],
    sample_summaries: list[dict[str, Any]],
    weighting_info: dict[str, Any],
) -> list[str]:
    arrays: dict[str, np.ndarray] = {
        "hist_edges": np.asarray(histogram_payload["hist_edges"], dtype=np.float64),
    }
    for score_name, score_payload in histogram_payload["score_histograms"].items():
        for kind, array in score_payload.items():
            arrays[_payload_array_key(score_name, kind)] = array

    arrays["metadata_json"] = np.array(
        json.dumps(
            {
                "payload_mode": HISTOGRAM_PAYLOAD_MODE,
                "available_scores": histogram_payload["available_scores"],
                "process_entries": histogram_payload["process_entries"],
                "sample_summaries": sample_summaries,
                "weighting": weighting_info,
                "score_hist_bins": int(len(np.asarray(histogram_payload["hist_edges"])) - 1),
            },
            sort_keys=True,
        )
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outpath, **arrays)
    return list(histogram_payload["available_scores"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Study boosted Higgs AK8 taggers from Pepper-produced ROOT files. "
            "Supports H->cc, H->bb, and top working points, weighted yields, ROC curves, "
            "and CMS-style plots."
        )
    )
    parser.add_argument(
        "--config",
        "--sample-config",
        dest="config",
        required=True,
        help="Path to the JSON study configuration file.",
    )
    parser.add_argument("--tree-name", default="Events")
    parser.add_argument(
        "--outdir",
        default="outputs/boosted_higgs_tagger_study",
        help="Directory where outputs are written.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=sorted(TARGET_DEFINITIONS),
        default=["hcc", "hbb"],
    )
    parser.add_argument(
        "--scores",
        nargs="+",
        default=["auto"],
        help=f"Known scores: {', '.join(sorted(SCORE_LABELS))}",
    )
    parser.add_argument(
        "--sig-effs",
        nargs="+",
        type=float,
        default=[0.1,  0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
    )
    parser.add_argument("--pt-min", type=float, default=200.0)
    parser.add_argument("--eta-max", type=float, default=2.4)
    parser.add_argument(
        "--candidate-strategy",
        choices=["all_jets", "leading_pt", "mass_window_leading_pt"],
        default="all_jets",
    )
    parser.add_argument("--msd-window-low", type=float, default=100.0)
    parser.add_argument("--msd-window-high", type=float, default=150.0)
    parser.add_argument("--max-files-per-sample", type=int, default=None)
    parser.add_argument("--gen-sumw-file", default=None)
    parser.add_argument(
        "--xsec-file",
        default=None,
        help="Optional override for the cross section JSON file.",
    )
    parser.add_argument("--lumi-fb", type=float, default=None)
    parser.add_argument("--weight-branch", default="weight")
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
        default="200 MB",
        help="Chunk size used when iterating over ROOT trees with uproot.",
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
    parser.add_argument("--plot-title-size", type=float, default=16)
    parser.add_argument("--plot-label-size", type=float, default=12.5)
    parser.add_argument("--plot-tick-size", type=float, default=10.0)
    parser.add_argument("--plot-legend-size", type=float, default=10.0)
    parser.add_argument("--plot-cms-size", type=float, default=16)
    parser.add_argument("--plot-dpi", type=int, default=220)
    parser.add_argument(
        "--score-hist-bins",
        type=int,
        default=DEFAULT_SCORE_HIST_BINS,
        help="Number of score bins used by histogram chunk payloads and histogram-based merges.",
    )
    return parser.parse_args()


def _strip_hash_comments(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        in_string = False
        escaped = False
        out_chars: list[str] = []
        for char in line:
            if escaped:
                out_chars.append(char)
                escaped = False
                continue
            if char == "\\":
                out_chars.append(char)
                escaped = True
                continue
            if char == '"':
                out_chars.append(char)
                in_string = not in_string
                continue
            if char == "#" and not in_string:
                break
            out_chars.append(char)
        cleaned_lines.append("".join(out_chars))
    return "\n".join(cleaned_lines)


def _load_json_maybe_with_comments(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(_strip_hash_comments(Path(path).read_text(encoding="utf-8")))


def _expand_file_patterns(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            files.extend(matches)
            continue
        if any(token in pattern for token in ["*", "?", "["]):
            raise FileNotFoundError(f"No files matched pattern: {pattern}")
        if not Path(pattern).exists():
            raise FileNotFoundError(f"Input file does not exist: {pattern}")
        files.append(pattern)
    return files


def load_study_config(path: str, max_files_per_sample: int | None) -> StudyConfig:
    payload = _load_json_maybe_with_comments(path)
    samples_raw = payload.get("samples", [])
    if not samples_raw:
        raise ValueError(f"No samples found in configuration: {path}")

    samples: list[SampleConfig] = []
    for entry in samples_raw:
        files = _expand_file_patterns(entry["files"])
        if max_files_per_sample is not None:
            files = files[:max_files_per_sample]
        name = entry["name"]
        samples.append(
            SampleConfig(
                name=name,
                dataset=entry.get("dataset", name),
                process=entry.get("process", name),
                label=entry.get("label", name),
                files=files,
            )
        )

    normalization = payload.get("normalization", {})
    default_xsec = str(REPO_ROOT / "crosssections_run3.json")
    return StudyConfig(
        samples=samples,
        gen_sumw_file=normalization.get("gen_sumw_file"),
        xsec_file=normalization.get("xsec_file", default_xsec),
        lumi_fb=normalization.get("lumi_fb"),
    )


def load_normalization_metadata(
    gen_sumw_file: str | None,
    xsec_file: str | None,
) -> tuple[dict[str, float], dict[str, float]]:
    gen_sumw_payload = _load_json_maybe_with_comments(gen_sumw_file)
    xsec_payload = _load_json_maybe_with_comments(xsec_file)

    gen_sumw_map: dict[str, float] = {}
    for dataset, value in gen_sumw_payload.items():
        if isinstance(value, dict):
            gen_sumw_map[dataset] = float(value["gen_sumw"])
        else:
            gen_sumw_map[dataset] = float(value)

    xsec_map = {dataset: float(value) for dataset, value in xsec_payload.items()}
    return gen_sumw_map, xsec_map


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(numerator.shape, np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    out[valid] = numerator[valid] / denominator[valid]
    zero_mask = np.isfinite(numerator) & np.isfinite(denominator) & (denominator <= 0)
    out[zero_mask] = 0.0
    return out


def _weighted_sum(weights: np.ndarray) -> float:
    return float(np.sum(weights)) if weights.size else 0.0


def _weighted_efficiency(scores: np.ndarray, weights: np.ndarray, cut: float) -> float:
    total = _weighted_sum(weights)
    if total <= 0:
        return float("nan")
    return float(_weighted_sum(weights[scores >= cut]) / total)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
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

    if args.candidate_strategy == "all_jets":
        return jets

    if args.candidate_strategy == "mass_window_leading_pt":
        msd_mask = (jets.msoftdrop >= args.msd_window_low) & (jets.msoftdrop <= args.msd_window_high)
        jets = jets[msd_mask]

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
    if args.candidate_strategy == "mass_window_leading_pt":
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

    if merged["pt"].size == 0:
        raise ValueError("No AK8 jets passed the requested selection.")
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
    if {"globalParT3_TopbWq", "globalParT3_QCD"} <= data.keys():
        data["gpart_top_wq"] = _safe_divide(
            data["globalParT3_TopbWq"],
            data["globalParT3_TopbWq"] + data["globalParT3_QCD"],
        ).astype(FLOAT_STORAGE_DTYPE, copy=False)
    if {"globalParT3_TopbWqq", "globalParT3_QCD"} <= data.keys():
        data["gpart_top_wqq"] = _safe_divide(
            data["globalParT3_TopbWqq"],
            data["globalParT3_TopbWqq"] + data["globalParT3_QCD"],
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


def _truth_mask(truth_codes: np.ndarray, label: str) -> np.ndarray:
    return truth_codes == TRUTH_LABEL_TO_CODE[label]


def build_target_masks(truth_codes: np.ndarray, target: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    definition = TARGET_DEFINITIONS[target]
    signal_codes = np.asarray([TRUTH_LABEL_TO_CODE[label] for label in definition["signal_labels"]], dtype=TRUTH_CODE_DTYPE)
    background_codes = np.asarray(
        [TRUTH_LABEL_TO_CODE[label] for label in definition["background_labels"]],
        dtype=TRUTH_CODE_DTYPE,
    )
    signal_mask = np.isin(truth_codes, signal_codes)
    background_mask = np.isin(truth_codes, background_codes)
    return signal_mask, background_mask, definition["background_labels"]


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
        label_mask = _truth_mask(truth_codes, label) & valid
        counts_by_label[label] = {
            "n_jets": int(np.sum(label_mask)),
            "weight_sum": _weighted_sum(weights[label_mask]),
            "signed_weight_sum": _weighted_sum(signed_weights[label_mask]),
        }

    rows: list[dict[str, Any]] = []
    for eff in sig_effs:
        cut = _weighted_quantile(signal_scores, signal_weights, 1.0 - eff)

        sig_mask = signal_mask & valid
        bkg_mask = background_mask & valid
        sig_pass = sig_mask & (scores >= cut)
        bkg_pass = bkg_mask & (scores >= cut)

        yield_sig_total = _weighted_sum(weights[sig_mask])
        yield_bkg_total = _weighted_sum(weights[bkg_mask])
        yield_sig_pass = _weighted_sum(weights[sig_pass])
        yield_bkg_pass = _weighted_sum(weights[bkg_pass])

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
            "yield_sig_pass_signed": _weighted_sum(signed_weights[sig_pass]),
            "yield_bkg_pass_signed": _weighted_sum(signed_weights[bkg_pass]),
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
            label_mask = _truth_mask(truth_codes, label) & valid
            label_scores = scores[label_mask]
            label_weights = weights[label_mask]
            row[f"eff__{label}"] = _weighted_efficiency(label_scores, label_weights, cut)
            row[f"yield_pass__{label}"] = _weighted_sum(weights[label_mask & (scores >= cut)])
            row[f"yield_pass_signed__{label}"] = _weighted_sum(signed_weights[label_mask & (scores >= cut)])

        for label in background_labels:
            row[f"n_pass__{label}"] = int(np.sum(_truth_mask(truth_codes, label) & valid & (scores >= cut)))

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

    total_signal = _weighted_sum(roc_weights[roc_labels == 1])
    total_background = _weighted_sum(roc_weights[roc_labels == 0])
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_plot_style(args: argparse.Namespace) -> PlotStyle:
    return PlotStyle(
        title_size=args.plot_title_size,
        label_size=args.plot_label_size,
        tick_size=args.plot_tick_size,
        legend_size=args.plot_legend_size,
        cms_size=args.plot_cms_size,
        dpi=args.plot_dpi,
    )


def format_summary_text(
    target: str,
    score_name: str,
    rows: list[dict[str, Any]],
    counts_by_label: dict[str, dict[str, float]],
    sample_summaries: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    lines: list[str] = []
    lines.append("=== Boosted Higgs Tagger Study ===")
    lines.append(f"Target: {TARGET_DEFINITIONS[target]['title']}")
    lines.append(f"Score: {SCORE_LABELS[score_name]}")
    lines.append(f"Candidate strategy: {args.candidate_strategy}")
    lines.append(f"AK8 selection: pt >= {args.pt_min:.1f} GeV, |eta| <= {args.eta_max:.1f}")
    lines.append(
        f"Weighting: lumi={args.lumi_fb if args.lumi_fb is not None else 'none'} /fb, "
        f"xsec/gen_sumw normalization, weight branch={args.weight_branch}, analysis uses abs(event weight)"
    )
    lines.append("")
    lines.append("Loaded samples:")
    for summary in sample_summaries:
        lines.append(
            f"  - {summary['dataset']}: files={summary['n_files']}, events={summary['n_events']}, "
            f"selected_jets={summary['n_selected_jets']}, xsec_fb={summary['xsec_fb']}, "
            f"gen_sumw={summary['gen_sumw']}, sample_norm={summary['sample_norm']:.8g}"
        )
    lines.append("")
    lines.append("Truth-category content after finite-score selection:")
    for label in TRUTH_LABEL_ORDER:
        entry = counts_by_label[label]
        lines.append(
            f"  - {label}: N={entry['n_jets']}, weight_sum={entry['weight_sum']:.6f}, "
            f"signed_weight_sum={entry['signed_weight_sum']:.6f}"
        )
    lines.append("")
    lines.append(
        f"{'TargetEff':>10}  {'Cut':>10}  {'SigEff':>8}  {'BkgEff':>8}  "
        f"{'Y_sig':>12}  {'Y_bkg':>12}  {'S/B':>10}  {'S/sqrt(S+B)':>12}  "
        f"{'S/sqrt(B)':>10}  {'Purity':>8}"
    )
    lines.append("-" * 136)
    for row in rows:
        purity = row["purity"]
        purity_str = f"{purity*100:7.2f}%" if np.isfinite(purity) else "   nan%"
        s_over_b = row["s_over_b"]
        s_over_b_str = "inf" if np.isinf(s_over_b) else f"{s_over_b:.6f}"
        s_over_sqrt_s_plus_b = row["s_over_sqrt_s_plus_b"]
        s_over_sqrt_s_plus_b_str = (
            f"{s_over_sqrt_s_plus_b:.6f}" if np.isfinite(s_over_sqrt_s_plus_b) else "nan"
        )
        s_over_sqrt_b = row["s_over_sqrt_b"]
        s_over_sqrt_b_str = "inf" if np.isinf(s_over_sqrt_b) else f"{s_over_sqrt_b:.6f}"
        lines.append(
            f"{row['target_sig_eff']*100:8.1f}%  {row['score_cut']:10.6f}  "
            f"{row['actual_sig_eff']*100:7.2f}%  {row['bkg_eff']*100:7.2f}%  "
            f"{row['yield_sig_pass']:12.6f}  {row['yield_bkg_pass']:12.6f}  "
            f"{s_over_b_str:>10}  {s_over_sqrt_s_plus_b_str:>12}  "
            f"{s_over_sqrt_b_str:>10}  {purity_str:>8}"
        )
    return "\n".join(lines) + "\n"


def _cms_label(ax: plt.Axes, plot_style: PlotStyle) -> None:
    hep.cms.label(
        data=False,
        ax=ax,
        fontsize=plot_style.cms_size,
        rlabel="2024 (13.6 TeV)",
    )


def _inside_legend(
    ax: plt.Axes,
    plot_style: PlotStyle,
    *,
    loc: str = "upper right",
    ncol: int = 1,
) -> None:
    legend = ax.legend(
        fontsize=plot_style.legend_size,
        loc=loc,
        bbox_to_anchor=(0.975, 0.985),
        ncol=ncol,
        frameon=False,
        borderpad=0.25,
        borderaxespad=0.0,
        labelspacing=0.3,
        handlelength=1.6,
        columnspacing=0.8,
    )
    if legend is not None:
        legend.set_zorder(5)


def _plain_yaxis(ax: plt.Axes) -> None:
    formatter = mticker.ScalarFormatter(useMathText=False)
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.offsetText.set_visible(False)


def _scientific_tick_label(value: float, _position: float) -> str:
    if not np.isfinite(value):
        return ""
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    mantissa_text = f"{mantissa:.1f}".rstrip("0").rstrip(".")
    if mantissa_text == "1":
        return rf"$10^{{{exponent}}}$"
    return rf"${mantissa_text}\times10^{{{exponent}}}$"


def _scientific_yaxis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_scientific_tick_label))
    ax.yaxis.offsetText.set_visible(False)


def _rebin_histogram_for_display(
    hist: np.ndarray,
    edges: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    display_edges = np.linspace(float(edges[0]), float(edges[-1]), n_bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rebinned, _ = np.histogram(centers, bins=display_edges, weights=hist)
    return rebinned.astype(np.float64), display_edges


def _histogram_density(hist: np.ndarray, edges: np.ndarray) -> np.ndarray:
    total = float(np.sum(hist))
    if total <= 0:
        return np.zeros_like(hist, dtype=np.float64)
    widths = np.diff(edges)
    density = hist.astype(np.float64) / total
    positive = widths > 0
    density[positive] = density[positive] / widths[positive]
    return density


def _save_plot(
    fig: plt.Figure,
    outpath: Path,
    plot_style: PlotStyle,
    *,
    left: float = 0.125,
    right: float = 0.97,
    bottom: float = 0.14,
    top: float = 0.875,
) -> None:
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    fig.savefig(outpath, dpi=plot_style.dpi)
    plt.close(fig)


def plot_significance_scan(
    rows: list[dict[str, Any]],
    target: str,
    score_name: str,
    outpath: Path,
    plot_style: PlotStyle,
) -> None:
    x = np.asarray([row["target_sig_eff"] for row in rows], dtype=float)
    y = np.asarray([row["s_over_sqrt_s_plus_b"] for row in rows], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return

    x = x[valid]
    y = y[valid]
    best_index = int(np.argmax(y))
    best_x = float(x[best_index])
    best_y = float(y[best_index])

    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax.plot(
        x,
        y,
        color="#d62728",
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        label=SCORE_LABELS[score_name],
    )
    ax.scatter([best_x], [best_y], color="#1f77b4", s=42, zorder=4, label=f"Best: {best_x*100:.1f}%")
    ax.set_xlabel("Target signal efficiency", fontsize=plot_style.label_size)
    ax.set_ylabel(r"$S/\sqrt{S+B}$", fontsize=plot_style.label_size)
    ax.set_title(
        f"{TARGET_DEFINITIONS[target]['title']} significance scan",
        fontsize=plot_style.title_size,
    )
    ax.set_xlim(max(0.0, float(np.min(x)) - 0.02), min(1.0, float(np.max(x)) + 0.02))
    if np.any(y > 0):
        ax.set_ylim(0.0, float(np.max(y)) * 1.18)
    ax.tick_params(axis="both", labelsize=plot_style.tick_size)
    ax.grid(True, alpha=0.3)
    ax.annotate(
        f"max={best_y:.3f}\nTargetEff={best_x*100:.1f}%",
        xy=(best_x, best_y),
        xytext=(10, 8),
        textcoords="offset points",
        fontsize=max(plot_style.tick_size - 0.5, 7.5),
        color="#1f77b4",
    )
    _inside_legend(ax, plot_style, loc="upper right", ncol=1)
    _cms_label(ax, plot_style)
    _save_plot(fig, outpath, plot_style, left=0.125, right=0.97, bottom=0.14, top=0.875)


def plot_score_distribution(
    scores: np.ndarray,
    truth_codes: np.ndarray,
    weights: np.ndarray,
    target: str,
    score_name: str,
    outpath: Path,
    plot_style: PlotStyle,
) -> None:
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ordered_labels = TARGET_DEFINITIONS[target]["signal_labels"] + TARGET_DEFINITIONS[target]["background_labels"]
    positive_contents: list[np.ndarray] = []
    for label in ordered_labels:
        mask = _truth_mask(truth_codes, label) & np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
        values = scores[mask]
        label_weights = weights[mask]
        if values.size == 0 or np.sum(label_weights) <= 0:
            continue
        density_hist, _, _ = ax.hist(
            values,
            bins=DISPLAY_SCORE_BINS,
            range=(0.0, 1.0),
            density=True,
            weights=label_weights,
            histtype="step",
            linewidth=1.8,
            label=f"{TRUTH_LABEL_TITLES[label]} (Y={np.sum(label_weights):.2f})",
            color=TRUTH_LABEL_COLORS[label],
        )
        positive_contents.append(np.asarray(density_hist)[np.asarray(density_hist) > 0])
    ax.set_xlabel(SCORE_LABELS[score_name], fontsize=plot_style.label_size)
    ax.set_ylabel("Arbitrary units", fontsize=plot_style.label_size)
    ax.set_title(
        f"{TARGET_DEFINITIONS[target]['title']} score distribution",
        fontsize=plot_style.title_size,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_yscale("log")
    if positive_contents:
        positive = np.concatenate(positive_contents)
        if positive.size > 0:
            ax.set_ylim(max(float(np.min(positive)) * 0.7, 1e-6), float(np.max(positive)) * 1.6)
    ax.tick_params(axis="both", labelsize=plot_style.tick_size)
    ax.grid(True, which="both", alpha=0.3)
    _inside_legend(ax, plot_style, loc="upper right", ncol=2)
    _cms_label(ax, plot_style)
    _save_plot(fig, outpath, plot_style, left=0.125, right=0.97, bottom=0.14, top=0.875)


def plot_background_process_score_distribution(
    scores: np.ndarray,
    truth_codes: np.ndarray,
    process_codes: np.ndarray,
    weights: np.ndarray,
    target: str,
    score_name: str,
    process_entries: list[dict[str, Any]],
    outpath: Path,
    plot_style: PlotStyle,
) -> None:
    signal_mask, background_mask, _ = build_target_masks(truth_codes, target)
    valid = background_mask & np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return

    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    plotted = 0
    positive_contents: list[np.ndarray] = []
    for entry in process_entries:
        mask = valid & (process_codes == int(entry["code"]))
        values = scores[mask]
        label_weights = weights[mask]
        if values.size == 0 or np.sum(label_weights) <= 0:
            continue
        plotted += 1
        density_hist, _, _ = ax.hist(
            values,
            bins=DISPLAY_PROCESS_SCORE_BINS,
            range=(0.0, 1.0),
            density=True,
            weights=label_weights,
            histtype="step",
            linewidth=1.8,
            label=f"{entry['label']} (Y={np.sum(label_weights):.2f})",
            color=entry["color"],
        )
        positive_contents.append(np.asarray(density_hist)[np.asarray(density_hist) > 0])

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_xlabel(SCORE_LABELS[score_name], fontsize=plot_style.label_size)
    ax.set_ylabel("Arbitrary units", fontsize=plot_style.label_size)
    ax.set_title(
        f"{TARGET_DEFINITIONS[target]['title']} background by process",
        fontsize=plot_style.title_size,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_yscale("log")
    if positive_contents:
        positive = np.concatenate(positive_contents)
        if positive.size > 0:
            ax.set_ylim(max(float(np.min(positive)) * 0.7, 1e-6), float(np.max(positive)) * 1.6)
    ax.tick_params(axis="both", labelsize=plot_style.tick_size)
    ax.grid(True, which="both", alpha=0.3)
    _inside_legend(ax, plot_style, loc="upper right", ncol=2)
    _cms_label(ax, plot_style)
    _save_plot(fig, outpath, plot_style, left=0.125, right=0.97, bottom=0.14, top=0.875)


def plot_background_process_working_points(
    scores: np.ndarray,
    truth_codes: np.ndarray,
    process_codes: np.ndarray,
    weights: np.ndarray,
    target: str,
    score_name: str,
    sig_effs: list[float],
    process_entries: list[dict[str, Any]],
    outpath: Path,
    plot_style: PlotStyle,
) -> None:
    signal_mask, background_mask, _ = build_target_masks(truth_codes, target)
    valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    signal_valid = signal_mask & valid
    background_valid = background_mask & valid
    if not np.any(signal_valid) or not np.any(background_valid):
        return

    signal_scores = scores[signal_valid]
    signal_weights = weights[signal_valid]
    if np.sum(signal_weights) <= 0:
        return

    x_positions = np.arange(len(sig_effs), dtype=float)
    cuts = [
        _weighted_quantile(signal_scores, signal_weights, 1.0 - sig_eff)
        for sig_eff in sig_effs
    ]
    process_yields: list[tuple[dict[str, Any], np.ndarray]] = []
    for entry in process_entries:
        process_mask = background_valid & (process_codes == int(entry["code"]))
        yields = []
        for cut in cuts:
            yields.append(_weighted_sum(weights[process_mask & (scores >= cut)]))
        yields_array = np.asarray(yields, dtype=float)
        if np.sum(yields_array) <= 0:
            continue
        process_yields.append((entry, yields_array))

    if not process_yields:
        return

    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    bottom = np.zeros(len(sig_effs), dtype=float)
    for entry, yields_array in process_yields:
        ax.bar(
            x_positions,
            yields_array,
            bottom=bottom,
            width=0.75,
            color=entry["color"],
            edgecolor="black",
            linewidth=0.4,
            label=entry["label"],
        )
        bottom += yields_array

    ax.set_xticks(x_positions, [f"{sig_eff * 100:.0f}%" for sig_eff in sig_effs])
    ax.set_xlabel("Target signal efficiency working point", fontsize=plot_style.label_size)
    ax.set_ylabel("Weighted background yield", fontsize=plot_style.label_size)
    if np.any(bottom > 0):
        ax.set_ylim(0.0, float(np.max(bottom)) * 1.25)
    ax.set_title(
        f"{TARGET_DEFINITIONS[target]['title']}: background vs WP",
        fontsize=plot_style.title_size,
        pad=10,
    )
    _scientific_yaxis(ax)
    ax.tick_params(axis="both", labelsize=plot_style.tick_size)
    ax.grid(True, axis="y", alpha=0.3)
    _inside_legend(ax, plot_style, loc="upper right", ncol=2)
    _cms_label(ax, plot_style)
    _save_plot(fig, outpath, plot_style, left=0.17, right=0.97, bottom=0.14, top=0.85)


def plot_roc_curve(
    roc_payload: dict[str, np.ndarray | float],
    target: str,
    score_name: str,
    outpath: Path,
    plot_style: PlotStyle,
) -> None:
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(8, 6))
    sig_eff = np.asarray(roc_payload["sig_eff"])
    bkg_eff = np.asarray(roc_payload["bkg_eff"])
    auc = float(roc_payload["auc"])
    ax.plot(
        sig_eff,
        np.clip(bkg_eff, 1e-6, None),
        linewidth=2.0,
        color="#d62728",
        label=f"{SCORE_LABELS[score_name]} (AUC={auc:.4f})",
    )
    ax.set_xlabel(
        f"{TARGET_DEFINITIONS[target]['title']} efficiency",
        fontsize=plot_style.label_size,
    )
    ax.set_ylabel("Background efficiency", fontsize=plot_style.label_size)
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1e-4, 1.0)
    ax.set_title(f"{TARGET_DEFINITIONS[target]['title']} ROC", fontsize=plot_style.title_size)
    ax.tick_params(axis="both", labelsize=plot_style.tick_size)
    ax.grid(True, which="both", alpha=0.3)
    _inside_legend(ax, plot_style, loc="upper right", ncol=1)
    _cms_label(ax, plot_style)
    _save_plot(fig, outpath, plot_style, left=0.125, right=0.97, bottom=0.14, top=0.875)


def _codes_for_labels(labels: list[str]) -> np.ndarray:
    return np.asarray([TRUTH_LABEL_TO_CODE[label] for label in labels], dtype=np.int32)


def _pass_from_hist(hist: np.ndarray) -> np.ndarray:
    return np.flip(np.cumsum(np.flip(hist, axis=-1), axis=-1), axis=-1)


def _working_point_bin_index(signal_pass_hist: np.ndarray, total_signal: float, target_eff: float) -> int:
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
        _codes_for_labels(TARGET_DEFINITIONS[target]["signal_labels"]),
        _codes_for_labels(TARGET_DEFINITIONS[target]["background_labels"]),
        TARGET_DEFINITIONS[target]["background_labels"],
    )
    count_total_truth = np.sum(count_truth, axis=1)
    weight_total_truth = np.sum(weight_truth, axis=1)
    signed_total_truth = np.sum(signed_truth, axis=1)

    signal_weight_hist = np.sum(weight_truth[signal_codes], axis=0)
    background_weight_hist = np.sum(weight_truth[background_codes], axis=0)
    if np.sum(signal_weight_hist) <= 0 or np.sum(background_weight_hist) <= 0:
        raise ValueError(f"No usable signal/background jets found for target '{target}'.")

    count_pass_truth = _pass_from_hist(count_truth)
    weight_pass_truth = _pass_from_hist(weight_truth)
    signed_pass_truth = _pass_from_hist(signed_truth)
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
        bin_index = _working_point_bin_index(signal_weight_pass, total_signal, eff)
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
    signal_codes = _codes_for_labels(TARGET_DEFINITIONS[target]["signal_labels"])
    background_codes = _codes_for_labels(TARGET_DEFINITIONS[target]["background_labels"])
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


def plot_score_distribution_from_hist(
    score_payload: dict[str, np.ndarray],
    hist_edges: np.ndarray,
    target: str,
    score_name: str,
    outpath: Path,
    plot_style: PlotStyle,
) -> None:
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ordered_labels = TARGET_DEFINITIONS[target]["signal_labels"] + TARGET_DEFINITIONS[target]["background_labels"]
    weight_truth = np.asarray(score_payload["weight_truth"], dtype=np.float64)
    positive_contents: list[np.ndarray] = []
    for label in ordered_labels:
        hist = weight_truth[TRUTH_LABEL_TO_CODE[label]]
        total = float(np.sum(hist))
        if total <= 0:
            continue
        rebinned_hist, display_edges = _rebin_histogram_for_display(hist, hist_edges, DISPLAY_SCORE_BINS)
        density_hist = _histogram_density(rebinned_hist, display_edges)
        if np.sum(density_hist) <= 0:
            continue
        ax.stairs(
            density_hist,
            display_edges,
            linewidth=1.8,
            label=f"{TRUTH_LABEL_TITLES[label]} (Y={total:.2f})",
            color=TRUTH_LABEL_COLORS[label],
        )
        positive_contents.append(density_hist[density_hist > 0])
    ax.set_xlabel(SCORE_LABELS[score_name], fontsize=plot_style.label_size)
    ax.set_ylabel("A.U.", fontsize=plot_style.label_size)
    ax.set_title(f"{TARGET_DEFINITIONS[target]['title']} score distribution", fontsize=plot_style.title_size)
    ax.set_xlim(0.0, 1.0)
    ax.set_yscale("log")
    if positive_contents:
        positive = np.concatenate(positive_contents)
        if positive.size > 0:
            ax.set_ylim(max(float(np.min(positive)) * 0.7, 1e-6), float(np.max(positive)) * 1.6)
    ax.tick_params(axis="both", labelsize=plot_style.tick_size)
    ax.grid(True, which="both", alpha=0.3)
    _inside_legend(ax, plot_style, loc="upper right", ncol=2)
    _cms_label(ax, plot_style)
    _save_plot(fig, outpath, plot_style, left=0.125, right=0.97, bottom=0.14, top=0.875)


def plot_background_process_score_distribution_from_hist(
    score_payload: dict[str, np.ndarray],
    hist_edges: np.ndarray,
    target: str,
    score_name: str,
    process_entries: list[dict[str, Any]],
    outpath: Path,
    plot_style: PlotStyle,
) -> None:
    background_codes = _codes_for_labels(TARGET_DEFINITIONS[target]["background_labels"])
    process_hist = np.asarray(score_payload["weight_process_truth"], dtype=np.float64)
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    plotted = 0
    positive_contents: list[np.ndarray] = []
    for entry in process_entries:
        hist = np.sum(process_hist[int(entry["code"]), background_codes], axis=0)
        total = float(np.sum(hist))
        if total <= 0:
            continue
        rebinned_hist, display_edges = _rebin_histogram_for_display(
            hist,
            hist_edges,
            DISPLAY_PROCESS_SCORE_BINS,
        )
        density_hist = _histogram_density(rebinned_hist, display_edges)
        if np.sum(density_hist) <= 0:
            continue
        plotted += 1
        ax.stairs(
            density_hist,
            display_edges,
            linewidth=1.8,
            label=f"{entry['label']} (Y={total:.2f})",
            color=entry["color"],
        )
        positive_contents.append(density_hist[density_hist > 0])
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel(SCORE_LABELS[score_name], fontsize=plot_style.label_size)
    ax.set_ylabel("A.U.", fontsize=plot_style.label_size)
    ax.set_title(f"{TARGET_DEFINITIONS[target]['title']} background by process", fontsize=plot_style.title_size)
    ax.set_xlim(0.0, 1.0)
    ax.set_yscale("log")
    if positive_contents:
        positive = np.concatenate(positive_contents)
        if positive.size > 0:
            ax.set_ylim(max(float(np.min(positive)) * 0.7, 1e-6), float(np.max(positive)) * 1.6)
    ax.tick_params(axis="both", labelsize=plot_style.tick_size)
    ax.grid(True, which="both", alpha=0.3)
    _inside_legend(ax, plot_style, loc="upper right", ncol=2)
    _cms_label(ax, plot_style)
    _save_plot(fig, outpath, plot_style, left=0.125, right=0.97, bottom=0.14, top=0.875)


def plot_background_process_working_points_from_hist(
    score_payload: dict[str, np.ndarray],
    hist_edges: np.ndarray,
    target: str,
    score_name: str,
    sig_effs: list[float],
    process_entries: list[dict[str, Any]],
    outpath: Path,
    plot_style: PlotStyle,
) -> None:
    weight_truth = np.asarray(score_payload["weight_truth"], dtype=np.float64)
    process_hist = np.asarray(score_payload["weight_process_truth"], dtype=np.float64)
    signal_codes = _codes_for_labels(TARGET_DEFINITIONS[target]["signal_labels"])
    background_codes = _codes_for_labels(TARGET_DEFINITIONS[target]["background_labels"])
    signal_pass = _pass_from_hist(np.sum(weight_truth[signal_codes], axis=0))
    total_signal = float(np.sum(weight_truth[signal_codes]))
    if total_signal <= 0:
        return

    x_positions = np.arange(len(sig_effs), dtype=float)
    process_background_hist = np.sum(process_hist[:, background_codes], axis=1)
    process_background_pass = _pass_from_hist(process_background_hist)
    process_yields: list[tuple[dict[str, Any], np.ndarray]] = []
    for entry in process_entries:
        process_index = int(entry["code"])
        yields = []
        for eff in sig_effs:
            bin_index = _working_point_bin_index(signal_pass, total_signal, eff)
            yields.append(float(process_background_pass[process_index, bin_index]))
        yields_array = np.asarray(yields, dtype=float)
        if np.sum(yields_array) <= 0:
            continue
        process_yields.append((entry, yields_array))
    if not process_yields:
        return

    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    bottom = np.zeros(len(sig_effs), dtype=float)
    for entry, yields_array in process_yields:
        ax.bar(
            x_positions,
            yields_array,
            bottom=bottom,
            width=0.75,
            color=entry["color"],
            edgecolor="black",
            linewidth=0.4,
            label=entry["label"],
        )
        bottom += yields_array
    ax.set_xticks(x_positions, [f"{sig_eff * 100:.0f}%" for sig_eff in sig_effs])
    ax.set_xlabel("Target signal efficiency working point", fontsize=plot_style.label_size)
    ax.set_ylabel("Weighted background yield", fontsize=plot_style.label_size)
    ax.set_title(
        f"{TARGET_DEFINITIONS[target]['title']}: background vs WP",
        fontsize=plot_style.title_size,
        pad=10,
    )
    _scientific_yaxis(ax)
    ax.tick_params(axis="both", labelsize=plot_style.tick_size)
    ax.grid(True, axis="y", alpha=0.3)
    _inside_legend(ax, plot_style, loc="upper right", ncol=2)
    _cms_label(ax, plot_style)
    _save_plot(fig, outpath, plot_style, left=0.17, right=0.97, bottom=0.14, top=0.85)


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
    study_config = load_study_config(args.config, args.max_files_per_sample)
    effective_gen_sumw_file = args.gen_sumw_file or study_config.gen_sumw_file
    effective_xsec_file = args.xsec_file or study_config.xsec_file or str(REPO_ROOT / "crosssections_run3.json")
    effective_lumi_fb = args.lumi_fb if args.lumi_fb is not None else study_config.lumi_fb

    effective_args = argparse.Namespace(**vars(args))
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
    slim_data = _build_slim_study_data(data, _select_scores_to_keep(args, available_scores))
    return slim_data, sample_summaries, effective_args, weighting_info


def export_chunk_payload(
    outpath: Path,
    data: dict[str, np.ndarray],
    sample_summaries: list[dict[str, Any]],
    weighting_info: dict[str, Any],
) -> list[str]:
    available_scores = get_available_scores(data)
    payload = {key: data[key] for key in ["truth_code", "weight", "weight_signed", *available_scores] if key in data}
    if "process_code" in data:
        payload["process_code"] = data["process_code"]
    payload["metadata_json"] = np.array(
        json.dumps(
            {
                "payload_mode": RAW_PAYLOAD_MODE,
                "available_scores": available_scores,
                "sample_summaries": sample_summaries,
                "process_entries": _build_process_entries_from_summaries(sample_summaries),
                "weighting": weighting_info,
            },
            sort_keys=True,
        )
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outpath, **payload)
    return available_scores


def _merge_sample_summaries(summary_groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    fields_to_sum = [
        "n_files",
        "n_skipped_files_missing_tree",
        "n_events",
        "n_selected_jets",
        "analysis_weight_sum",
        "signed_weight_sum",
    ]

    for summaries in summary_groups:
        for summary in summaries:
            key = (summary["name"], summary["dataset"], summary["process"], summary["label"])
            if key not in merged:
                merged[key] = dict(summary)
                for field in fields_to_sum:
                    merged[key].setdefault(field, 0)
                continue
            for field in fields_to_sum:
                merged[key][field] += summary.get(field, 0)
            if merged[key].get("weight_branch") is None:
                merged[key]["weight_branch"] = summary.get("weight_branch")

    return sorted(merged.values(), key=lambda entry: (entry["process"], entry["dataset"], entry["name"]))


def load_merged_chunk_payloads(chunk_patterns: list[str]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    chunk_paths = [Path(path) for path in _expand_file_patterns(chunk_patterns)]
    if not chunk_paths:
        raise ValueError("No chunk NPZ files were found for merge.")

    arrays_by_key: dict[str, list[np.ndarray]] = {"truth_code": [], "weight": [], "weight_signed": []}
    sample_summary_groups: list[list[dict[str, Any]]] = []
    expected_scores: list[str] | None = None
    weighting_info: dict[str, Any] | None = None
    process_chunk_payloads: list[tuple[np.ndarray, list[dict[str, Any]]]] = []
    process_payload_complete = True
    saw_process_payload = False

    for chunk_path in chunk_paths:
        with np.load(chunk_path, allow_pickle=False) as payload:
            metadata = json.loads(np.asarray(payload["metadata_json"]).item())
            chunk_scores = list(metadata.get("available_scores", []))
            if expected_scores is None:
                expected_scores = chunk_scores
                for score_name in expected_scores:
                    arrays_by_key[score_name] = []
            elif chunk_scores != expected_scores:
                raise ValueError(
                    f"Inconsistent available scores across chunk payloads. "
                    f"Expected {expected_scores}, got {chunk_scores} from {chunk_path}"
                )

            if weighting_info is None:
                weighting_info = metadata.get("weighting", {})

            for key in ["truth_code", "weight", "weight_signed", *(expected_scores or [])]:
                arrays_by_key[key].append(np.asarray(payload[key]))

            chunk_process_entries = metadata.get("process_entries", [])
            if "process_code" in payload.files and chunk_process_entries:
                saw_process_payload = True
                process_chunk_payloads.append(
                    (
                        np.asarray(payload["process_code"], dtype=PROCESS_CODE_DTYPE),
                        list(chunk_process_entries),
                    )
                )
            else:
                process_payload_complete = False
            sample_summary_groups.append(metadata.get("sample_summaries", []))

    merged_data = {key: np.concatenate(value_chunks) for key, value_chunks in arrays_by_key.items() if value_chunks}
    if saw_process_payload and process_payload_complete and len(process_chunk_payloads) == len(chunk_paths):
        process_to_label: dict[str, str] = {}
        for _, chunk_entries in process_chunk_payloads:
            for entry in chunk_entries:
                process_to_label.setdefault(entry["process"], entry.get("label", entry["process"]))

        global_entries = _build_process_entries_from_pairs(process_to_label)
        global_code_by_process = {entry["process"]: int(entry["code"]) for entry in global_entries}
        remapped_chunks: list[np.ndarray] = []
        for codes, chunk_entries in process_chunk_payloads:
            local_process_by_code = {int(entry["code"]): entry["process"] for entry in chunk_entries}
            remapped = np.full(codes.shape, -1, dtype=PROCESS_CODE_DTYPE)
            for local_code, process in local_process_by_code.items():
                remapped[codes == local_code] = global_code_by_process[process]
            if np.any(remapped < 0):
                raise ValueError("Failed to remap process codes while merging chunk payloads.")
            remapped_chunks.append(remapped)
        merged_data["process_code"] = np.concatenate(remapped_chunks)

    sample_summaries = _merge_sample_summaries(sample_summary_groups)
    return merged_data, sample_summaries, (weighting_info or {})


def _expand_process_axis(array: np.ndarray, new_size: int) -> np.ndarray:
    if array.shape[0] >= new_size:
        return array
    expanded = np.zeros((new_size, *array.shape[1:]), dtype=array.dtype)
    expanded[: array.shape[0]] = array
    return expanded


def load_merged_histogram_payloads(
    chunk_patterns: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    chunk_paths = [Path(path) for path in _expand_file_patterns(chunk_patterns)]
    if not chunk_paths:
        raise ValueError("No chunk NPZ files were found for merge.")

    score_histograms: dict[str, dict[str, np.ndarray]] = {}
    expected_scores: list[str] | None = None
    hist_edges: np.ndarray | None = None
    weighting_info: dict[str, Any] | None = None
    sample_summary_groups: list[list[dict[str, Any]]] = []
    global_process_entries: list[dict[str, Any]] = []
    process_to_global_code: dict[str, int] = {}

    for chunk_path in chunk_paths:
        with np.load(chunk_path, allow_pickle=False) as payload:
            metadata = json.loads(np.asarray(payload["metadata_json"]).item())
            if _payload_mode_from_metadata(metadata) != HISTOGRAM_PAYLOAD_MODE:
                raise ValueError(f"Chunk payload {chunk_path} is not a histogram payload.")

            chunk_scores = list(metadata.get("available_scores", []))
            if expected_scores is None:
                expected_scores = chunk_scores
            elif chunk_scores != expected_scores:
                raise ValueError(
                    f"Inconsistent available scores across chunk payloads. "
                    f"Expected {expected_scores}, got {chunk_scores} from {chunk_path}"
                )

            chunk_edges = np.asarray(payload["hist_edges"], dtype=np.float64)
            if hist_edges is None:
                hist_edges = chunk_edges
            elif not np.array_equal(hist_edges, chunk_edges):
                raise ValueError(f"Inconsistent histogram edges found in {chunk_path}.")

            if weighting_info is None:
                weighting_info = metadata.get("weighting", {})

            chunk_process_entries = list(metadata.get("process_entries", []))
            for entry in chunk_process_entries:
                process = entry["process"]
                if process not in process_to_global_code:
                    new_code = len(global_process_entries)
                    process_to_global_code[process] = new_code
                    global_process_entries.append(
                        {
                            "code": new_code,
                            "process": process,
                            "label": entry.get("label", process),
                            "color": entry.get("color", _process_color(process, new_code)),
                        }
                    )
                    for score_payload in score_histograms.values():
                        score_payload["weight_process_truth"] = _expand_process_axis(
                            score_payload["weight_process_truth"],
                            len(global_process_entries),
                        )

            local_process_map = {int(entry["code"]): entry["process"] for entry in chunk_process_entries}
            for score_name in expected_scores or []:
                count_truth = np.asarray(payload[_payload_array_key(score_name, "count_truth")], dtype=np.int64)
                weight_truth = np.asarray(payload[_payload_array_key(score_name, "weight_truth")], dtype=np.float64)
                signed_truth = np.asarray(payload[_payload_array_key(score_name, "signed_truth")], dtype=np.float64)
                chunk_process_hist = np.asarray(
                    payload[_payload_array_key(score_name, "weight_process_truth")],
                    dtype=np.float64,
                )
                if score_name not in score_histograms:
                    score_histograms[score_name] = {
                        "count_truth": np.zeros_like(count_truth),
                        "weight_truth": np.zeros_like(weight_truth),
                        "signed_truth": np.zeros_like(signed_truth),
                        "weight_process_truth": np.zeros(
                            (len(global_process_entries), *chunk_process_hist.shape[1:]),
                            dtype=np.float64,
                        ),
                    }

                score_histograms[score_name]["count_truth"] += count_truth
                score_histograms[score_name]["weight_truth"] += weight_truth
                score_histograms[score_name]["signed_truth"] += signed_truth
                if score_histograms[score_name]["weight_process_truth"].shape[0] < len(global_process_entries):
                    score_histograms[score_name]["weight_process_truth"] = _expand_process_axis(
                        score_histograms[score_name]["weight_process_truth"],
                        len(global_process_entries),
                    )
                for local_code, process in local_process_map.items():
                    global_code = process_to_global_code[process]
                    score_histograms[score_name]["weight_process_truth"][global_code] += chunk_process_hist[local_code]

            sample_summary_groups.append(metadata.get("sample_summaries", []))

    if hist_edges is None:
        raise ValueError("Histogram payload merge found no histogram edges.")

    ordered_process_entries = _build_process_entries_from_pairs(
        {entry["process"]: entry["label"] for entry in global_process_entries}
    )
    if ordered_process_entries:
        reorder = [process_to_global_code[entry["process"]] for entry in ordered_process_entries]
        for score_payload in score_histograms.values():
            score_payload["weight_process_truth"] = score_payload["weight_process_truth"][reorder]

    histogram_payload = {
        "payload_mode": HISTOGRAM_PAYLOAD_MODE,
        "hist_edges": hist_edges,
        "available_scores": list(expected_scores or []),
        "process_entries": ordered_process_entries,
        "score_histograms": score_histograms,
    }
    sample_summaries = _merge_sample_summaries(sample_summary_groups)
    return histogram_payload, sample_summaries, (weighting_info or {})


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
    render_histogram_plots(histogram_payload, effective_args, study_payload, outdirs)
    write_json(outdirs["base"] / "plot_only_summary.json", study_payload)
    return study_payload


def run_export_chunk(args: argparse.Namespace) -> dict[str, Any]:
    data, sample_summaries, effective_args, weighting_info = prepare_study_data_from_root(args)
    chunk_path = Path(args.export_chunk).resolve()
    available_scores = get_available_scores(data)
    selected_scores = _select_scores_to_keep(args, available_scores)
    if args.chunk_payload_mode == "histogram":
        histogram_payload = build_histogram_payload_from_raw_data(
            data=data,
            sample_summaries=sample_summaries,
            score_names=selected_scores,
            n_bins=args.score_hist_bins,
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
    print(f"Output directory: {Path(args.outdir).resolve()}")
    print("Available scores:", ", ".join(payload["available_scores"]))
    for target, score_map in payload["targets"].items():
        print(f"Target {target}: {', '.join(score_map.keys())}")


if __name__ == "__main__":
    main()
