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
        "default_scores": ["gpart_h2bb", "gpart_xbb_vs_xcc", "gpart_hbb_vs_hcc"],
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
    "gpart_xbb_vs_xcc": r"gParT3 $Xbb$ vs $Xcc$",
    "gpart_hbb_vs_hcc": r"gParT3 $H\to bb$ vs $H\to cc$",
    "gpart_higgs_vs_qcd": r"gParT3 Higgs vs QCD",
    "pnet_hcc": r"ParticleNetWithMass $Hcc$ vs QCD",
    "pnet_xcc": r"ParticleNet $Xcc$ vs QCD",
    "pnetlegacy_xcc": r"ParticleNetLegacy $Xcc$",
}

SCORE_INPUT_FIELDS = {
    "gpart_h2cc": {"globalParT3_Xcc", "globalParT3_QCD"},
    "gpart_h2bb": {"globalParT3_Xbb", "globalParT3_QCD"},
    "gpart_xbb_vs_xcc": {"globalParT3_Xbb", "globalParT3_Xcc"},
    "gpart_hbb_vs_hcc": {"globalParT3_Xbb", "globalParT3_Xcc", "globalParT3_QCD"},
    "gpart_higgs_vs_qcd": {"globalParT3_Xbb", "globalParT3_Xcc", "globalParT3_QCD"},
    "pnet_hcc": {"particleNetWithMass_HccvsQCD"},
    "pnet_xcc": {"particleNet_XccVsQCD"},
    "pnetlegacy_xcc": {"particleNetLegacy_Xcc"},
}

_DEFAULT_CONTOUR_REGION_DEFINITIONS = {
    "qcd_others": {
        "label": "QCD&Others region",
        "kind": "complement",
        "annotation": {"x": 0.15, "y": 0.80},
    },
    "hcc": {
        "label": "Hcc region",
        "kind": "rectangle",
        "x_min_exclusive": 0.45,
        "y_min_exclusive": 0.0,
        "y_max_inclusive": 0.7,
        "annotation": {"x": 0.55, "y": 0.65},
    },
    "hbb": {
        "label": "Hbb region",
        "kind": "rectangle",
        "x_min_exclusive": 0.75,
        "y_min_exclusive": 0.7,
        "y_max_inclusive": 1.0,
        "annotation": {"x": 0.78, "y": 0.88},
    },
}

_DEFAULT_CONTOUR_BOUNDARY_SEGMENTS = [
    {"x": [0.45, 0.45], "y": [0.0, 0.7]},
    {"x": [0.45, 1.0], "y": [0.7, 0.7]},
    {"x": [0.75, 0.75], "y": [0.7, 1.0]},
]

_HBB_VS_HCC_CONTOUR_REGION_DEFINITIONS = {
    "qcd_others": {
        "label": "QCD&Others region",
        "kind": "complement",
        "annotation": {"x": 0.15, "y": 0.80},
    },
    "hcc": {
        "label": "Hcc region",
        "kind": "rectangle",
        "x_min_exclusive": 0.45,
        "y_min_exclusive": 0.0,
        "y_max_inclusive": 0.55,
        "annotation": {"x": 0.53, "y": 0.50},
    },
    "hbb": {
        "label": "Hbb region",
        "kind": "rectangle",
        "x_min_exclusive": 0.75,
        "y_min_exclusive": 0.55,
        "y_max_inclusive": 1.0,
        "annotation": {"x": 0.77, "y": 0.82},
    },
}

_HBB_VS_HCC_CONTOUR_BOUNDARY_SEGMENTS = [
    {"x": [0.45, 0.45], "y": [0.0, 0.55]},
    {"x": [0.45, 1.0], "y": [0.55, 0.55]},
    {"x": [0.75, 0.75], "y": [0.55, 1.0]},
]

DEFAULT_XBB_VS_XCC_REGION_PRESET = "loose"

XBB_VS_XCC_REGION_PRESETS = {
    "loose": {
        "label": "loose",
        "description": "QCD mistag eff = 1%",
        "hcc_x_cut": 0.7333,
        "hbb_x_cut": 0.9133,
        "fixed_x_cut": 0.9133,
    },
    "tight": {
        "label": "tight",
        "description": "QCD mistag eff = 0.1% / 0.4%",
        "hcc_x_cut": 0.9467,
        "hbb_x_cut": 0.9467,
        "fixed_x_cut": 0.9467,
    },
}


def _build_xbb_vs_xcc_contour_region_definitions(preset: str) -> dict[str, dict[str, object]]:
    preset_key = str(preset).strip().lower()
    if preset_key not in XBB_VS_XCC_REGION_PRESETS:
        raise ValueError(
            "Unknown xbb-vs-xcc region preset: "
            f"{preset}. Available presets: {', '.join(sorted(XBB_VS_XCC_REGION_PRESETS))}"
        )
    preset_payload = XBB_VS_XCC_REGION_PRESETS[preset_key]
    hcc_x_cut = float(preset_payload["hcc_x_cut"])
    hbb_x_cut = float(preset_payload["hbb_x_cut"])
    annotation_x = 0.965
    return {
        "qcd_others": {
            "label": "QCD&Others region",
            "kind": "complement",
            "annotation": {"x": 0.16, "y": 0.80},
        },
        "hcc": {
            "label": "Hcc region",
            "kind": "rectangle",
            "x_min_exclusive": hcc_x_cut,
            "y_min_exclusive": 0.0,
            "y_max_inclusive": 0.85,
            "annotation": {"x": annotation_x, "y": 0.68, "ha": "right"},
        },
        "hbb": {
            "label": "Hbb region",
            "kind": "rectangle",
            "x_min_exclusive": hbb_x_cut,
            "y_min_exclusive": 0.85,
            "y_max_inclusive": 1.0,
            "annotation": {"x": annotation_x, "y": 0.95, "ha": "right"},
        },
    }


def _build_xbb_vs_xcc_contour_boundary_segments(preset: str) -> list[dict[str, list[float]]]:
    preset_key = str(preset).strip().lower()
    if preset_key not in XBB_VS_XCC_REGION_PRESETS:
        raise ValueError(
            "Unknown xbb-vs-xcc region preset: "
            f"{preset}. Available presets: {', '.join(sorted(XBB_VS_XCC_REGION_PRESETS))}"
        )
    preset_payload = XBB_VS_XCC_REGION_PRESETS[preset_key]
    hcc_x_cut = float(preset_payload["hcc_x_cut"])
    hbb_x_cut = float(preset_payload["hbb_x_cut"])
    segments = [
        {"x": [hcc_x_cut, hcc_x_cut], "y": [0.0, 0.85]},
        {"x": [hcc_x_cut, 1.0], "y": [0.85, 0.85]},
    ]
    segments.append({"x": [hbb_x_cut, hbb_x_cut], "y": [0.85, 1.0]})
    return segments

def build_globalpart3_contour_plots(
    *,
    xbb_vs_xcc_region_preset: str = DEFAULT_XBB_VS_XCC_REGION_PRESET,
) -> list[dict[str, object]]:
    preset_key = str(xbb_vs_xcc_region_preset).strip().lower()
    if preset_key not in XBB_VS_XCC_REGION_PRESETS:
        raise ValueError(
            "Unknown xbb-vs-xcc region preset: "
            f"{xbb_vs_xcc_region_preset}. Available presets: {', '.join(sorted(XBB_VS_XCC_REGION_PRESETS))}"
        )
    preset_payload = XBB_VS_XCC_REGION_PRESETS[preset_key]
    return [
        {
            "key": "gpart_higgs_vs_qcd__vs__gpart_hbb_vs_hcc",
            "x_score": "gpart_higgs_vs_qcd",
            "y_score": "gpart_hbb_vs_hcc",
            "filename_stem": "gpart_higgs_vs_qcd__vs__gpart_hbb_vs_hcc__contours",
            "region_definitions": _HBB_VS_HCC_CONTOUR_REGION_DEFINITIONS,
            "boundary_segments": _HBB_VS_HCC_CONTOUR_BOUNDARY_SEGMENTS,
            "region_preset": "default",
        },
        {
            "key": "gpart_higgs_vs_qcd__vs__gpart_xbb_vs_xcc",
            "x_score": "gpart_higgs_vs_qcd",
            "y_score": "gpart_xbb_vs_xcc",
            "filename_stem": "gpart_higgs_vs_qcd__vs__gpart_xbb_vs_xcc__contours",
            "region_definitions": _build_xbb_vs_xcc_contour_region_definitions(preset_key),
            "boundary_segments": _build_xbb_vs_xcc_contour_boundary_segments(preset_key),
            "fixed_x_cut": float(preset_payload["fixed_x_cut"]),
            "region_preset": preset_key,
            "region_preset_label": str(preset_payload["label"]),
            "region_preset_description": str(preset_payload["description"]),
        },
    ]


GLOBALPART3_CONTOUR_PLOTS = build_globalpart3_contour_plots()

GLOBALPART3_CONTOUR_PLOT_BY_KEY = {
    plot_def["key"]: plot_def for plot_def in GLOBALPART3_CONTOUR_PLOTS
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

GLOBALPART3_FIXED_OTHER_EFF_TARGETS = [
    0.01,
    0.005,
    0.001,
]

GLOBALPART3_FIXED_X_CUT = 0.9467

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
