from __future__ import annotations

from typing import Any, TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from tthcc_an.config_loader import SampleConfig


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
        "signal_labels": ["hcc_pure"],
        "background_labels": [
            "hcc_contaminated",
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
        "signal_labels": ["hbb_pure"],
        "background_labels": [
            "hbb_contaminated",
            "hbb_partial",
            "hcc_pure",
            "hcc_contaminated",
            "hcc_partial",
            "top",
            "other",
        ],
        "default_scores": ["gpart_h2bb", "gpart_hbb_vs_hcc"],
    },
    "higgs": {
        "title": r"$H\to bb / cc$",
        "signal_labels": ["hbb_pure", "hcc_pure"],
        "background_labels": [
            "hbb_contaminated",
            "hbb_partial",
            "hcc_contaminated",
            "hcc_partial",
            "top",
            "other",
        ],
        "default_scores": ["gpart_higgs_vs_qcd"],
    },
}

SCORE_LABELS = {
    "gpart_h2cc": r"gParT3 $H\to cc$ score",
    "gpart_h2bb": r"gParT3 $H\to bb$ score",
    "gpart_hbb_vs_hcc": r"gParT3 $H\to bb$ vs $H\to cc$",
    "gpart_higgs_vs_qcd": r"gParT3 Higgs vs QCD",
    "pnet_hcc": r"ParticleNetWithMass $Hcc$ vs QCD",
    "pnet_xcc": r"ParticleNet $Xcc$ vs QCD",
    "pnetlegacy_xcc": r"ParticleNetLegacy $Xcc$",
}

SCORE_INPUT_FIELDS = {
    "gpart_h2cc": {"globalParT3_Xcc", "globalParT3_QCD"},
    "gpart_h2bb": {"globalParT3_Xbb", "globalParT3_QCD"},
    "gpart_hbb_vs_hcc": {"globalParT3_Xbb", "globalParT3_Xcc"},
    "gpart_higgs_vs_qcd": {"globalParT3_Xbb", "globalParT3_Xcc", "globalParT3_QCD"},
    "pnet_hcc": {"particleNetWithMass_HccvsQCD"},
    "pnet_xcc": {"particleNet_XccVsQCD"},
    "pnetlegacy_xcc": {"particleNetLegacy_Xcc"},
}

GLOBALPART3_CONTOUR_PLOT = {
    "key": "gpart_higgs_vs_qcd__vs__gpart_hbb_vs_hcc",
    "x_score": "gpart_higgs_vs_qcd",
    "y_score": "gpart_hbb_vs_hcc",
    "filename_stem": "gpart_higgs_vs_qcd__vs__gpart_hbb_vs_hcc__contours",
}

GLOBALPART3_CONTOUR_HIST_BINS = 150
GLOBALPART3_CONTOUR_SMOOTH_SIGMA = 1.0
GLOBALPART3_CONTOUR_CLIP_EPS = 1.0e-6

GLOBALPART3_CONTOUR_ENCLOSED_FRACTIONS = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]

GLOBALPART3_CONTOUR_CATEGORIES = [
    {
        "key": "hbb_pure",
        "legend_label": "hbb_pure jets & efficiencies (%)",
        "color": "#ff7f0e",
        "truth_codes": [TRUTH_LABEL_TO_CODE["hbb_pure"]],
    },
    {
        "key": "hcc_pure",
        "legend_label": "hcc_pure jets & efficiencies (%)",
        "color": "#d62728",
        "truth_codes": [TRUTH_LABEL_TO_CODE["hcc_pure"]],
    },
    {
        "key": "others",
        "legend_label": "Others jets & efficiencies (%)",
        "color": "#1f77b4",
        "truth_codes": [
            code
            for label, code in TRUTH_LABEL_TO_CODE.items()
            if label not in {"hbb_pure", "hcc_pure"}
        ],
    },
]


def _process_sort_key(process: str) -> tuple[int, str]:
    if process in PROCESS_DISPLAY_ORDER:
        return (PROCESS_DISPLAY_ORDER.index(process), process)
    return (len(PROCESS_DISPLAY_ORDER), process)


def _process_color(process: str, fallback_index: int) -> str:
    if process in PROCESS_COLORS:
        return PROCESS_COLORS[process]
    cmap = plt.get_cmap("tab20")
    return str(matplotlib.colors.to_hex(cmap(fallback_index % cmap.N)))


def process_color(process: str, fallback_index: int) -> str:
    return _process_color(process, fallback_index)


def build_process_entries_from_pairs(process_to_label: dict[str, str]) -> list[dict[str, Any]]:
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


def build_process_entries_from_samples(samples: list["SampleConfig"]) -> list[dict[str, Any]]:
    process_to_label: dict[str, str] = {}
    for sample in samples:
        process_to_label.setdefault(sample.process, sample.label)
    return build_process_entries_from_pairs(process_to_label)


def build_process_entries_from_summaries(sample_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    process_to_label: dict[str, str] = {}
    for summary in sample_summaries:
        process_to_label.setdefault(summary["process"], summary["label"])
    return build_process_entries_from_pairs(process_to_label)
