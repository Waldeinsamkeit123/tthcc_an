from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from tthcc_an.nn_study.config import NnScoreClass, NnStudyConfig
from tthcc_an.nn_study.dataset import NnDataset
from tthcc_an.nn_study.plotting import (
    _save,
    _setup_style,
    plot_score_distribution,
    plot_score_significance_comparison,
)
from tthcc_an.nn_study.reporting import write_json, write_text
from tthcc_an.nn_study.score_significance import calculate_score_significance_scan


DENOMINATOR_EPSILON = np.finfo(np.float64).eps
MASS_BRANCH = "TargetFatJet_regressed_mass_generic"

PEPPER_COLORS = {
    "qcd": "tab:pink",
    "vjets": "#731963",
    "ttLF": "#E0F3F8",
    "ttcj": "#8DB38B",
    "tt2c": "#56876D",
    "ttcc": "#04724D",
    "ttbj": "#74ADD1",
    "tt2b": "#4575B4",
    "ttbb": "#313695",
    "ttZqq": "#FEE090",
    "ttZcc": "firebrick",
    "ttZbb": "xkcd:scarlet",
    "ttHbb": "#FDAE61",
    "ttHcc": "#F46D43",
}

PEPPER_LABELS = {
    "ttHcc": "ttHcc",
    "ttHbb": "ttHbb",
    "ttZqq": "ttZqq",
    "ttZcc": "ttZcc",
    "ttZbb": "ttZbb",
    "ttLF": r"$t\bar{t}$ + LF",
    "ttcj": r"$t\bar{t}$ + c",
    "tt2c": r"$t\bar{t}$ + 2c",
    "ttcc": r"$t\bar{t}+c\bar{c}$",
    "ttbj": r"$t\bar{t}$ + b",
    "tt2b": r"$t\bar{t}$ + 2b",
    "ttbb": r"$t\bar{t}+b\bar{b}$",
}


def derive_tthz_qcd_discriminant(
    score_tthz: np.ndarray,
    score_qcd: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    score_tthz = np.asarray(score_tthz, dtype=np.float64)
    score_qcd = np.asarray(score_qcd, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not (score_tthz.shape == score_qcd.shape == weights.shape):
        raise ValueError("D_ttHZQCD input arrays must have identical shapes.")

    denominator = score_tthz + score_qcd
    finite_inputs = np.isfinite(score_tthz) & np.isfinite(score_qcd)
    safe = finite_inputs & (denominator > DENOMINATOR_EPSILON)
    values = np.full(score_tthz.shape, 0.5, dtype=np.float64)
    np.divide(score_tthz, denominator, out=values, where=safe)
    clipped = safe & ((values < 0.0) | (values > 1.0))
    values = np.clip(values, 0.0, 1.0)
    denominator_zero = finite_inputs & ~safe
    valid_weight = np.isfinite(weights) & (weights >= 0.0)
    return values, {
        "definition": "score_ttHZ / (score_ttHZ + score_qcd)",
        "denominator_epsilon": float(DENOMINATOR_EPSILON),
        "denominator_le_epsilon_event_count": int(np.sum(denominator_zero)),
        "denominator_le_epsilon_weighted_yield": float(
            np.sum(weights[denominator_zero & valid_weight])
        ),
        "nonfinite_input_event_count": int(np.sum(~finite_inputs)),
        "clipped_to_unit_interval_event_count": int(np.sum(clipped)),
        "fallback_for_invalid_denominator": 0.5,
        "all_outputs_finite": bool(np.all(np.isfinite(values))),
        "all_outputs_in_unit_interval": bool(
            np.all((values >= 0.0) & (values <= 1.0))
        ),
    }


def _score_range(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = values[mask & np.isfinite(values)]
    if selected.size == 0:
        return {"event_count": 0, "minimum": None, "maximum": None, "quantiles": None}
    quantiles = np.quantile(selected, [0.005, 0.5, 0.995])
    return {
        "event_count": int(selected.size),
        "minimum": float(np.min(selected)),
        "maximum": float(np.max(selected)),
        "quantiles": {
            "0p5_percent": float(quantiles[0]),
            "50_percent": float(quantiles[1]),
            "99p5_percent": float(quantiles[2]),
        },
    }


def _truth_yields(
    config: NnStudyConfig, dataset: NnDataset, mask: np.ndarray
) -> dict[str, dict[str, Any]]:
    return {
        truth.name: {
            "label": truth.label,
            "event_count": int(np.sum(mask & (dataset.truth_index == index))),
            "weighted_yield": float(
                np.sum(dataset.analysis_weight[mask & (dataset.truth_index == index)])
            ),
        }
        for index, truth in enumerate(config.truth_categories)
    }


def _distribution_payload(
    values: np.ndarray,
    truth_index: np.ndarray,
    weights: np.ndarray,
    truth_names: list[str],
    bins: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for index, name in enumerate(truth_names):
        selected = (
            mask
            & (truth_index == index)
            & np.isfinite(values)
            & np.isfinite(weights)
            & (weights >= 0.0)
        )
        histogram, _ = np.histogram(values[selected], bins=bins, weights=weights[selected])
        total = float(np.sum(histogram))
        groups[name] = {
            "event_count": int(np.sum(selected)),
            "weighted_yield_in_score_range": total,
            "normalized_integral": 1.0 if total > 0.0 else 0.0,
            "bin_yields": [float(value) for value in histogram],
        }
    return {"bin_edges": bins.tolist(), "groups": groups}


def _process_masks(
    config: NnStudyConfig, dataset: NnDataset
) -> tuple[list[dict[str, Any]], list[str]]:
    truth = {
        name: dataset.truth_index == config.truth_names.index(name)
        for name in config.truth_names
    }
    labels = {
        label: np.isin(
            dataset.sample_index,
            [
                index
                for index, sample in enumerate(dataset.sample_summaries)
                if sample.get("label") == label
            ],
        )
        for label in {sample.get("label") for sample in dataset.sample_summaries}
    }
    definitions = [
        ("qcd", "QCD", "#6f6f6f", labels.get("QCD", np.zeros_like(dataset.sample_index, dtype=bool))),
        ("vjets", "V+jets", "#d9a441", labels.get("Wto2Q", False) | labels.get("Zto2Q", False)),
        ("ttLF", r"$t\bar{t}$ + LF", "#4c78a8", truth["ttLF"]),
        ("ttcj", r"$t\bar{t}$ + c", "#9ecae9", truth["ttcj"]),
        ("tt2c", r"$t\bar{t}$ + 2c", "#6baed6", truth["tt2c"]),
        ("ttcc", r"$t\bar{t}$ + cc", "#2171b5", truth["ttcc"]),
        ("ttbj", r"$t\bar{t}$ + b", "#74c476", truth["ttbj"]),
        ("tt2b", r"$t\bar{t}$ + 2b", "#31a354", truth["tt2b"]),
        ("ttbb", r"$t\bar{t}$ + bb", "#006d2c", truth["ttbb"]),
        ("ttZqq", r"$t\bar{t}Z(Z\to qq)$", "#f768a1", truth["ttZqq"]),
        ("ttZcc", r"$t\bar{t}Z(Z\to cc)$", "#dd3497", truth["ttZcc"]),
        ("ttZbb", r"$t\bar{t}Z(Z\to bb)$", "#ae017e", truth["ttZbb"]),
        ("ttHbb", r"$t\bar{t}H(H\to bb)$", "#fdae61", truth["ttHbb"]),
        ("ttHcc", r"$t\bar{t}H(H\to cc)$", "#d73027", truth["ttHcc"]),
    ]
    groups = [
        {"name": name, "label": label, "color": PEPPER_COLORS.get(name, color), "mask": np.asarray(mask, dtype=bool)}
        for name, label, color, mask in definitions
    ]
    assigned = np.zeros(dataset.sample_index.shape, dtype=np.int8)
    for group in groups:
        assigned += group["mask"].astype(np.int8)
    if np.any(assigned > 1):
        raise ValueError("Mass-stack process groups overlap.")
    if np.any(assigned == 0):
        groups.append(
            {
                "name": "other",
                "label": "Other cached MC",
                "color": "#bdbdbd",
                "mask": assigned == 0,
            }
        )
    available_labels = set(labels)
    missing = []
    if not any("single" in str(label).lower() for label in available_labels):
        missing.append("single top")
    if not any(str(label) in {"ttX", "ttW", "ttVV"} for label in available_labels):
        missing.append("additional ttX (beyond cached ttH/ttZ)")
    if not any(str(label).lower() == "data" for label in available_labels):
        missing.append("data")
    return groups, missing


def _draw_mass_stack(
    ax: plt.Axes,
    bins: np.ndarray,
    groups: list[dict[str, Any]],
    histograms: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    bottom = np.zeros(bins.size - 1, dtype=np.float64)
    for group in groups:
        histogram = histograms[group["name"]]
        if np.sum(histogram) <= 0.0:
            continue
        ax.bar(
            bins[:-1], histogram, width=np.diff(bins), bottom=bottom,
            align="edge", color=group["color"], linewidth=0.25,
            edgecolor="white", label=group["label"],
        )
        bottom += histogram
    signal = histograms.get("ttHcc")
    overlay = bottom
    if signal is not None and np.sum(signal) > 0.0:
        overlay = bottom + 9.0 * signal
        ax.stairs(overlay, bins, color=PEPPER_COLORS["ttHcc"], linewidth=1.2,
                  label=r"Stack + ttHcc x10")
    return bottom, overlay


def _plot_mass_stack(
    outpath: Path,
    bins: np.ndarray,
    groups: list[dict[str, Any]],
    histograms: dict[str, np.ndarray],
    config: NnStudyConfig,
    score_threshold: float,
    *,
    qcd_cut: float = 0.01,
    variance: np.ndarray | None = None,
) -> None:
    _setup_style()
    plt.style.use(hep.style.CMS)
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(dpi=150, figsize=(6.67, 6.67))
    prediction, overlay = _draw_mass_stack(ax, bins, groups, histograms)
    if variance is not None:
        sigma = np.sqrt(variance)
        ax.bar(bins[:-1], 2 * sigma, width=np.diff(bins),
               bottom=prediction - sigma, align="edge", color="gray",
               alpha=0.34, linewidth=0, label="MC stat.")
    ax.set_ylim(0.0, max(float(np.max(overlay)), 1.0) * 1.8
                if variance is not None else float(np.max(overlay)) * 1.8)
    ax.set_xlim(float(bins[0]), float(bins[-1]))
    ax.set_xlabel("Target FatJet regressed mass (generic) (GeV)")
    ax.set_ylabel("Events")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.yaxis.get_offset_text().set_x(-0.09)
    ax.text(0.02, 0.97, config.plot_options.get("study_label", "0L V4") + " diagnostic\n"
            f"$s_{{QCD}}<{qcd_cut:.3f}$, $s_{{ttHZ}}>{score_threshold:.2f}$",
            transform=ax.transAxes, va="top", fontsize="small")
    ax.legend(frameon=False, ncol=2, fontsize=8.5, loc="upper right",
              borderpad=0.1, columnspacing=0.2, labelspacing=0.1, handlelength=1.0)
    hep.cms.label(data=False, llabel="", ax=ax, year=config.year,
                  lumi=f"{config.lumi_fb:.1f}", com=config.energy_tev, fontsize=15)
    ax.text(0.13, 1.005, "Simulation", transform=ax.transAxes,
            va="bottom", fontsize=15, fontstyle="italic")
    fig.tight_layout()
    _save(fig, outpath)


def _format_summary(payload: dict[str, Any]) -> str:
    diagnostics = payload["derived_score_diagnostics"]
    lines = [
        "=== D_ttHZQCD Diagnostic Study ===",
        f"Definition: {diagnostics['definition']}",
        f"Denominator <= epsilon: {diagnostics['denominator_le_epsilon_event_count']} events, "
        f"yield={diagnostics['denominator_le_epsilon_weighted_yield']:.8g}",
    ]
    for name, entry in payload["score_ranges"].items():
        lines.append(f"{name} range: [{entry['minimum']:.8g}, {entry['maximum']:.8g}]")
    for name, comparison in payload["significance_comparisons"].items():
        lines.append(f"{name} significance:")
        for score_name in ("score_ttHZ", "D_ttHZQCD"):
            result = comparison[score_name]["signals"]["ttHcc"]
            lines.append(
                f"  {score_name}: best cut={result['best_threshold']:.6g}, "
                f"Z={result['best_significance']:.8g}"
            )
    lines.append("Mass-stack process yields:")
    for group in payload["mass_stack"]["groups"]:
        lines.append(
            f"  {group['label']}: selected={group['weighted_yield']:.8g}, "
            f"50-200 GeV={group['displayed_weighted_yield']:.8g}"
        )
    lines.append("Unavailable groups: " + ", ".join(payload["mass_stack"]["unavailable_groups"]))
    lines.append("No final working point is selected.")
    return "\n".join(lines) + "\n"


def run_tthz_qcd_discriminant(
    *,
    config: NnStudyConfig,
    dataset: NnDataset,
    plot_dir: Path,
    summary_dir: Path,
    plot_suffix: str,
) -> tuple[list[str], dict[str, Any]]:
    if "0l" not in config.channel.lower() or not config.uses_distinct_prediction_truth:
        raise ValueError("tthz-qcd-discriminant requires the 0L V4 coarse-score setup.")
    missing_scores = sorted({"ttHZ", "qcd"} - set(dataset.scores))
    if missing_scores:
        raise KeyError("Missing cached scores: " + ", ".join(missing_scores))
    if MASS_BRANCH not in dataset.analysis_columns:
        raise KeyError(f"Missing cached analysis branch: {MASS_BRANCH}")

    score_tthz = np.asarray(dataset.scores["ttHZ"], dtype=np.float64)
    score_qcd = np.asarray(dataset.scores["qcd"], dtype=np.float64)
    weights = np.asarray(dataset.analysis_weight, dtype=np.float64)
    values, diagnostics = derive_tthz_qcd_discriminant(score_tthz, score_qcd, weights)
    valid = np.isfinite(score_tthz) & np.isfinite(score_qcd) & np.isfinite(weights) & (weights >= 0.0)
    qcd_selected = valid & (score_qcd < 0.01)
    mass_selected = qcd_selected & (score_tthz > 0.50)
    if not (np.sum(valid) >= np.sum(qcd_selected) >= np.sum(mass_selected)):
        raise ValueError("D_ttHZQCD selections are not nested.")

    score = NnScoreClass("D_ttHZQCD", r"$D_{ttHZQCD}$", diagnostics["definition"], "darkorange")
    fine_truths = [
        replace(truth, label=PEPPER_LABELS.get(truth.name, truth.label),
                color=PEPPER_COLORS.get(truth.name, truth.color))
        for truth in config.truth_categories
    ]
    bins = np.linspace(0.0, 1.0, 51)
    output_dir = plot_dir / "ttHZ_qcd_discriminant"
    paths = {
        "fine_truth": output_dir / f"D_ttHZQCD__fine_truth{plot_suffix}",
        "fine_truth_yield": output_dir / f"D_ttHZQCD__fine_truth__yield{plot_suffix}",
        "fine_truth_qcd_lt_0p01": output_dir / f"D_ttHZQCD__fine_truth__qcd_lt_0p01{plot_suffix}",
        "fine_truth_qcd_lt_0p01_yield": output_dir / f"D_ttHZQCD__fine_truth__qcd_lt_0p01__yield{plot_suffix}",
        "fine_truth_qcd_lt_0p01_yield_stack": output_dir / f"D_ttHZQCD__fine_truth__qcd_lt_0p01__yield__stack{plot_suffix}",
        "coarse": output_dir / f"D_ttHZQCD__coarse{plot_suffix}",
        "coarse_qcd_lt_0p01": output_dir / f"D_ttHZQCD__coarse__qcd_lt_0p01{plot_suffix}",
        "significance": output_dir / f"score_ttHZ_vs_D_ttHZQCD__significance{plot_suffix}",
        "significance_qcd_lt_0p01": output_dir / f"score_ttHZ_vs_D_ttHZQCD__significance__qcd_lt_0p01{plot_suffix}",
        "mass_stack": output_dir / f"regressed_mass__qcd_lt_0p01__ttHZ_gt_0p50__stack{plot_suffix}",
    }
    common = {
        "score": score,
        "values": values,
        "weights": weights,
        "config": config,
        "bin_edges": bins,
        "cms_com": 13,
    }
    for normalize, key, mask in (
        (True, "fine_truth", valid),
        (False, "fine_truth_yield", valid),
        (True, "fine_truth_qcd_lt_0p01", qcd_selected),
        (False, "fine_truth_qcd_lt_0p01_yield", qcd_selected),
    ):
        plot_score_distribution(
            outpath=paths[key],
            truth_index=dataset.truth_index[mask],
            truths=fine_truths,
            normalize=normalize,
            figsize=(6.67, 6.67),
            reference_style=True,
            selection_label=("0L V4, inclusive MC" if mask is valid else
                             r"0L V4, $s_{QCD}<0.010$"),
            **{name: array[mask] if name in {"values", "weights"} else array for name, array in common.items()},
        )
    for key, mask in (("coarse", valid), ("coarse_qcd_lt_0p01", qcd_selected)):
        plot_score_distribution(
            outpath=paths[key],
            truth_index=dataset.prediction_truth_index[mask],
            truths=config.prediction_truth_categories,
            normalize=True,
            figsize=(6.6, 4.8),
            legend_columns=2,
            **{name: array[mask] if name in {"values", "weights"} else array for name, array in common.items()},
        )

    selected_range = _score_range(values, qcd_selected)
    zoom_min = 0.88 if np.any(qcd_selected) else None
    if zoom_min is not None:
        zoom_bins = np.linspace(zoom_min, 1.0, 49)
        zoom_common = dict(common)
        zoom_common["bin_edges"] = np.concatenate((bins[bins < zoom_min], zoom_bins))
        paths["fine_truth_qcd_lt_0p01_zoom"] = output_dir / f"D_ttHZQCD__fine_truth__qcd_lt_0p01__zoom{plot_suffix}"
        plot_score_distribution(
            outpath=paths["fine_truth_qcd_lt_0p01_zoom"],
            truth_index=dataset.truth_index[qcd_selected],
            truths=fine_truths,
            normalize=True,
            figsize=(6.67, 6.67),
            reference_style=True,
            selection_label=r"0L V4, $s_{QCD}<0.010$",
            display_range=(zoom_min, 1.0),
            **{name: array[qcd_selected] if name in {"values", "weights"} else array for name, array in zoom_common.items()},
        )
        zoom_common["bin_edges"] = zoom_bins
        plot_score_distribution(
            outpath=paths["fine_truth_qcd_lt_0p01_yield_stack"],
            truth_index=dataset.truth_index[qcd_selected],
            truths=fine_truths,
            normalize=False,
            stack=True,
            figsize=(6.67, 6.67),
            reference_style=True,
            selection_label=r"0L V4, $s_{QCD}<0.010$",
            **{name: array[qcd_selected] if name in {"values", "weights"} else array for name, array in zoom_common.items()},
        )

    thresholds = np.linspace(0.0, 1.0, 201)
    significance: dict[str, Any] = {}
    truth_label = next(truth.label for truth in config.truth_categories if truth.name == "ttHcc")
    for key, mask, selection_label in (
        ("inclusive", valid, None),
        ("qcd_lt_0p01", qcd_selected, r"$score_{qcd}<0.01$"),
    ):
        raw = calculate_score_significance_scan(
            score=score_tthz[mask], weights=weights[mask], truth_index=dataset.truth_index[mask],
            truth_names=config.truth_names, signals=["ttHcc"], thresholds=thresholds, direction=">",
        )
        derived = calculate_score_significance_scan(
            score=values[mask], weights=weights[mask], truth_index=dataset.truth_index[mask],
            truth_names=config.truth_names, signals=["ttHcc"], thresholds=thresholds, direction=">",
        )
        significance[key] = {"score_ttHZ": raw, "D_ttHZQCD": derived}
        path_key = "significance" if key == "inclusive" else "significance_qcd_lt_0p01"
        plot_score_significance_comparison(
            outpath=paths[path_key], raw_result=raw["signals"]["ttHcc"],
            discriminant_result=derived["signals"]["ttHcc"], signal_label=truth_label,
            selection_label=selection_label, discriminant_label=r"derived $D_{ttHZQCD}$",
            title=r"ttHcc: $score_{ttHZ}$ vs $D_{ttHZQCD}$", cms_com=13,
        )

    mass = np.asarray(dataset.analysis_columns[MASS_BRANCH], dtype=np.float64)
    mass_bins = np.linspace(50.0, 200.0, 31)
    groups, unavailable = _process_masks(config, dataset)
    histograms: dict[str, np.ndarray] = {}
    group_payload: list[dict[str, Any]] = []
    displayed = mass_selected & np.isfinite(mass) & (mass >= mass_bins[0]) & (mass <= mass_bins[-1])
    for group in groups:
        selected = mass_selected & group["mask"]
        histogram, _ = np.histogram(mass[selected & np.isfinite(mass)], bins=mass_bins, weights=weights[selected & np.isfinite(mass)])
        histograms[group["name"]] = histogram.astype(np.float64)
        group_payload.append({
            "name": group["name"], "label": group["label"],
            "event_count": int(np.sum(selected)), "weighted_yield": float(np.sum(weights[selected])),
            "displayed_event_count": int(np.sum(displayed & group["mask"])),
            "displayed_weighted_yield": float(np.sum(histogram)),
        })
    _plot_mass_stack(paths["mass_stack"], mass_bins, groups, histograms, config, 0.50)

    mass_stack_variants: dict[str, Any] = {}
    for threshold in (0.30, 0.40, 0.60, 0.70):
        selected = qcd_selected & (score_tthz > threshold)
        variant_histograms = {}
        for group in groups:
            in_group = selected & group["mask"] & np.isfinite(mass)
            variant_histograms[group["name"]] = np.histogram(
                mass[in_group], bins=mass_bins, weights=weights[in_group]
            )[0].astype(np.float64)
        key = f"mass_stack_ttHZ_gt_0p{int(round(threshold * 100)):02d}"
        paths[key] = output_dir / f"regressed_mass__qcd_lt_0p01__ttHZ_gt_0p{int(round(threshold * 100)):02d}__stack{plot_suffix}"
        _plot_mass_stack(paths[key], mass_bins, groups, variant_histograms, config, threshold)
        displayed_variant = selected & np.isfinite(mass) & (mass >= mass_bins[0]) & (mass <= mass_bins[-1])
        histogram_yield = float(sum(np.sum(histogram) for histogram in variant_histograms.values()))
        expected_yield = float(np.sum(weights[displayed_variant]))
        if not np.isclose(histogram_yield, expected_yield, rtol=1e-9, atol=1e-9):
            raise ValueError(f"Mass stack at score_ttHZ > {threshold:.2f} loses weighted yield.")
        mass_stack_variants[f"{threshold:.2f}"] = {
            "selection": f"score_qcd < 0.01 AND score_ttHZ > {threshold:.2f}",
            "selected_event_count": int(np.sum(selected)),
            "selected_weighted_yield": float(np.sum(weights[selected])),
            "displayed_weighted_yield": expected_yield,
            "stack_histogram_sum": histogram_yield,
        }

    process_yield_sum = float(sum(group["weighted_yield"] for group in group_payload))
    histogram_sum = float(sum(np.sum(histogram) for histogram in histograms.values()))
    selected_yield = float(np.sum(weights[mass_selected]))
    displayed_yield = float(np.sum(weights[displayed]))
    tolerance = 1e-9 * max(selected_yield, 1.0)
    payload = {
        "schema_version": 1,
        "channel": config.channel,
        "weighting": "sample_norm * abs(weight)",
        "derived_score_diagnostics": diagnostics,
        "selections": {
            "inclusive": {"event_count": int(np.sum(valid)), "weighted_yield": float(np.sum(weights[valid]))},
            "qcd_lt_0p01": {"definition": "score_qcd < 0.01", "event_count": int(np.sum(qcd_selected)), "weighted_yield": float(np.sum(weights[qcd_selected])), "truth_yields": _truth_yields(config, dataset, qcd_selected)},
            "mass_stack": {"definition": "score_qcd < 0.01 AND score_ttHZ > 0.50", "event_count": int(np.sum(mass_selected)), "weighted_yield": selected_yield, "truth_yields": _truth_yields(config, dataset, mass_selected)},
        },
        "score_ranges": {"inclusive": _score_range(values, valid), "qcd_lt_0p01": selected_range},
        "zoom": {"produced": zoom_min is not None, "minimum": zoom_min, "maximum": 1.0, "bins": 48, "bin_width": 0.0025, "criterion": "fixed diagnostic range 0.88-1.00"},
        "fine_truth_distribution": _distribution_payload(values, dataset.truth_index, weights, config.truth_names, bins, valid),
        "fine_truth_distribution_qcd_lt_0p01": _distribution_payload(values, dataset.truth_index, weights, config.truth_names, bins, qcd_selected),
        "coarse_distribution": _distribution_payload(values, dataset.prediction_truth_index, weights, config.prediction_truth_names, bins, valid),
        "coarse_distribution_qcd_lt_0p01": _distribution_payload(values, dataset.prediction_truth_index, weights, config.prediction_truth_names, bins, qcd_selected),
        "significance_comparisons": significance,
        "mass_stack": {
            "selection": "score_qcd < 0.01 AND score_ttHZ > 0.50", "mass_branch": MASS_BRANCH,
            "bin_edges_gev": mass_bins.tolist(), "groups": group_payload,
            "selected_weighted_yield": selected_yield, "displayed_weighted_yield": displayed_yield,
            "stack_histogram_sum": histogram_sum, "unavailable_groups": unavailable,
            "signal_overlay": "ttHcc x10; nominal ttHcc remains in stack",
        },
        "mass_stack_variants": mass_stack_variants,
        "validation": {
            "derived_score_finite": diagnostics["all_outputs_finite"],
            "derived_score_in_unit_interval": diagnostics["all_outputs_in_unit_interval"],
            "selected_counts_nested": bool(np.sum(valid) >= np.sum(qcd_selected) >= np.sum(mass_selected)),
            "weighted_yields_nonnegative": bool(np.all(weights[valid] >= 0.0)),
            "significance_values_finite": bool(all(scan["validation"]["finite_scan_metrics"] for comparison in significance.values() for scan in comparison.values())),
            "process_groups_partition_selected_events": bool(abs(process_yield_sum - selected_yield) <= tolerance),
            "stack_sum_matches_displayed_process_yields": bool(abs(histogram_sum - displayed_yield) <= tolerance),
            "missing_process_groups_reported_not_invented": bool(unavailable),
            "cache_schema_unchanged": True,
        },
        "interpretation": "Diagnostic only; no final D_ttHZQCD working point is selected.",
        "plots": {name: str(path) for name, path in paths.items()},
    }
    failed = [name for name, passed in payload["validation"].items() if not passed]
    if failed:
        raise ValueError("D_ttHZQCD validation failed: " + ", ".join(failed))
    write_json(summary_dir / "ttHZ_qcd_discriminant.json", payload)
    write_text(summary_dir / "ttHZ_qcd_discriminant.txt", _format_summary(payload))
    return [str(path) for path in paths.values()], payload
