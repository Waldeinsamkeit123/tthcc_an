from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import mplhep as hep
import numpy as np
import uproot

from tthcc_an.config_loader import load_json_maybe_with_comments
from tthcc_an.nn_study.config import NnStudyConfig
from tthcc_an.nn_study.dataset import NnDataset, _to_numpy
from tthcc_an.nn_study.plotting import _save, _setup_style
from tthcc_an.nn_study.reporting import write_json, write_text
from tthcc_an.nn_study.tthz_qcd_discriminant import MASS_BRANCH, _draw_mass_stack, _plot_mass_stack, _process_masks


def _settings(
    config: NnStudyConfig, *, validate_data: bool = True,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    options = config.regressed_mass_fine_binning
    if not options.get("enabled") or "0l" not in config.channel.lower():
        raise ValueError("regressed_mass_fine_binning requires an enabled 0L config.")
    cuts = [float(value) for value in options["tthz_cuts"]]
    detailed = [float(value) for value in options["mcstat_detailed_cuts"]]
    if not cuts or cuts != sorted(set(cuts)) or not set(detailed) <= set(cuts):
        raise ValueError("Invalid ttHZ cuts or detailed MC-stat cuts.")
    minimum, maximum = map(float, options["mass_range"])
    blind_low, blind_high = map(float, options["blind_mass_range"])
    width = float(options["mass_bin_width"])
    if not (minimum < blind_low < blind_high < maximum and width > 0):
        raise ValueError("Invalid mass or blinding range.")
    bins = np.arange(minimum, maximum + width / 2, width)
    if not np.isclose(bins[-1], maximum) or not all(
        np.any(np.isclose(bins, boundary)) for boundary in (blind_low, blind_high)
    ):
        raise ValueError("Mass bins must end at the maximum and contain blind boundaries.")
    blinded = (bins[:-1] >= blind_low) & (bins[1:] <= blind_high)
    if validate_data:
        samples = [str(name) for name in options["data_samples"]]
        trigger_config = load_json_maybe_with_comments(options["data_trigger_config"])
        if samples != trigger_config["dataset_trigger_order"] or any(
            not trigger_config["dataset_trigger_map"].get(sample) for sample in samples
        ):
            raise ValueError("Data sample order must match Pepper's trigger priority.")
    return options, bins, blinded


def _data_cache_path(config: NnStudyConfig) -> Path:
    return config.outdir / "cache" / "prepared_data_mass_sidebands.npz"


def _data_cache_definition(config: NnStudyConfig, options: dict[str, Any], bins: np.ndarray) -> dict[str, Any]:
    trigger_config = load_json_maybe_with_comments(options["data_trigger_config"])
    return {
        "source": "real 2024 Pepper-selected data",
        "input_location": config.input_location,
        "data_samples": options["data_samples"],
        "max_files_per_sample": config.max_files_per_sample,
        "dataset_trigger_order": trigger_config["dataset_trigger_order"],
        "dataset_trigger_map": {
            name: trigger_config["dataset_trigger_map"][name]
            for name in options["data_samples"]
        },
        "qcd_cut": float(options["qcd_cut"]),
        "tthz_cuts": [float(value) for value in options["tthz_cuts"]],
        "mass_edges": bins.tolist(),
        "blind_mass_range": [float(value) for value in options["blind_mass_range"]],
    }


def _unblinded_data_histograms(
    score_qcd: np.ndarray, score_tthz: np.ndarray, mass: np.ndarray,
    *, qcd_cut: float, cuts: list[float], bins: np.ndarray,
    blind_range: tuple[float, float],
) -> np.ndarray:
    low, high = blind_range
    valid = (np.isfinite(score_qcd) & np.isfinite(score_tthz) & np.isfinite(mass)
             & (score_qcd < qcd_cut) & ((mass < low) | (mass > high)))
    return np.asarray([
        np.histogram(mass[valid & (score_tthz > cut)], bins=bins)[0]
        for cut in cuts
    ], dtype=np.float64)


def prepare_data_cache(config: NnStudyConfig, *, force: bool = False) -> dict[str, Any]:
    options, bins, blinded = _settings(config)
    path = _data_cache_path(config)
    if path.exists() and not force:
        load_data_cache(config)
        return {"operation": "prepare-data-cache", "cache_reused": True,
                "cache_path": str(path), "data_samples": options["data_samples"]}
    counts = np.zeros((len(options["tthz_cuts"]), len(bins) - 1), dtype=np.float64)
    processed_files = 0
    for sample in options["data_samples"]:
        files = sorted((Path(config.input_location) / sample).glob("*.root"))
        if not files:
            raise FileNotFoundError(f"No real-data ROOT files for {sample}.")
        if config.max_files_per_sample is not None:
            files = files[:config.max_files_per_sample]
        for filename in files:
            with uproot.open(filename) as root_file:
                if config.tree_name not in root_file:
                    continue
                if ("dsname" not in root_file or str(root_file["dsname"]) != sample
                        or "Cutnames" not in root_file
                        or "Trigger" not in str(root_file["Cutnames"])
                        or "HasTargetFatJet" not in str(root_file["Cutnames"])):
                    raise ValueError(f"Data provenance/trigger selection not verified: {filename}")
                tree = root_file[config.tree_name]
                branches = ["score_qcd", "score_ttHZ", MASS_BRANCH, config.weight_branch]
                missing = set(branches) - set(tree.keys())
                if missing:
                    raise KeyError(f"Missing data branches in {filename}: {sorted(missing)}")
                processed_files += 1
                for arrays in tree.iterate(branches, library="ak", step_size=config.uproot_step_size):
                    data_weights = np.asarray(ak.to_numpy(arrays[config.weight_branch]))
                    if not np.all(data_weights == 1):
                        raise ValueError(f"Real data must have unit event weights: {filename}")
                    counts += _unblinded_data_histograms(
                        np.asarray(ak.to_numpy(arrays["score_qcd"]), dtype=np.float64),
                        np.asarray(ak.to_numpy(arrays["score_ttHZ"]), dtype=np.float64),
                        _to_numpy(arrays[MASS_BRANCH], flatten_first=True),
                        qcd_cut=float(options["qcd_cut"]),
                        cuts=[float(value) for value in options["tthz_cuts"]],
                        bins=bins,
                        blind_range=tuple(map(float, options["blind_mass_range"])),
                    )
    if processed_files == 0:
        raise ValueError("No real-data Events trees were processed.")
    counts[:, blinded] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary, counts=counts,
        definition=json.dumps(_data_cache_definition(config, options, bins), sort_keys=True),
    )
    temporary.replace(path)
    return {"operation": "prepare-data-cache", "cache_reused": False,
            "cache_path": str(path), "data_samples": options["data_samples"],
            "files_processed": processed_files}


def load_data_cache(config: NnStudyConfig) -> np.ndarray:
    options, bins, blinded = _settings(config)
    path = _data_cache_path(config)
    if not path.exists():
        raise FileNotFoundError(f"Missing real-data cache: {path}. Run --prepare-data-cache first.")
    with np.load(path, allow_pickle=False) as payload:
        definition = json.loads(str(payload["definition"].item()))
        counts = np.asarray(payload["counts"], dtype=np.float64)
    if definition != _data_cache_definition(config, options, bins):
        raise ValueError("Real-data cache settings changed; rerun --prepare-data-cache --force.")
    if (counts.shape != (len(options["tthz_cuts"]), len(bins) - 1)
            or not np.all(np.isnan(counts[:, blinded]))
            or not np.all(np.isfinite(counts[:, ~blinded]))
            or np.any(counts[:, ~blinded] < 0)):
        raise ValueError("Invalid or unblinded real-data cache content.")
    return counts


def _mc_histograms(
    config: NnStudyConfig, dataset: NnDataset, bins: np.ndarray,
    options: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], dict[float, dict[str, dict[str, np.ndarray]]]]:
    groups, unavailable = _process_masks(config, dataset)
    mass = np.asarray(dataset.analysis_columns[MASS_BRANCH], dtype=np.float64)
    qcd = np.asarray(dataset.scores["qcd"], dtype=np.float64)
    tthz = np.asarray(dataset.scores["ttHZ"], dtype=np.float64)
    weights = np.asarray(dataset.analysis_weight, dtype=np.float64)
    baseline = (np.isfinite(mass) & np.isfinite(qcd) & np.isfinite(tthz)
                & np.isfinite(weights) & (weights >= 0)
                & (mass >= bins[0]) & (mass <= bins[-1])
                & (qcd < float(options["qcd_cut"])))
    by_cut: dict[float, dict[str, dict[str, np.ndarray]]] = {}
    for cut in map(float, options["tthz_cuts"]):
        selected = baseline & (tthz > cut)
        processes: dict[str, dict[str, np.ndarray]] = {}
        for group in groups:
            in_group = selected & group["mask"]
            values = mass[in_group]
            group_weights = weights[in_group]
            processes[group["name"]] = {
                "sumw": np.histogram(values, bins=bins, weights=group_weights)[0],
                "sumw2": np.histogram(values, bins=bins, weights=group_weights**2)[0],
                "count": np.histogram(values, bins=bins)[0],
            }
        for field, direct_weights in (("sumw", weights[selected]),
                                      ("sumw2", weights[selected]**2)):
            stacked = np.sum([entry[field] for entry in processes.values()], axis=0)
            direct = np.histogram(mass[selected], bins=bins, weights=direct_weights)[0]
            np.testing.assert_allclose(stacked, direct, rtol=1e-9, atol=1e-7,
                                       err_msg=f"{field} stack mismatch at score_ttHZ > {cut:g}")
        by_cut[cut] = processes
    return groups, [name for name in unavailable if name != "data"], by_cut


def _region_metrics(entry: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    yields = entry["sumw"][mask]
    variances = entry["sumw2"][mask]
    populated = (yields > 0) & (variances > 0)
    relative = np.sqrt(variances[populated]) / yields[populated]
    total_yield = float(np.sum(yields))
    total_sumw2 = float(np.sum(variances))
    return {
        "raw_event_count": int(np.sum(entry["count"][mask])),
        "weighted_yield": total_yield,
        "sumw2": total_sumw2,
        "integrated_mc_stat": float(np.sqrt(total_sumw2)),
        "integrated_neff": total_yield**2 / total_sumw2 if total_sumw2 > 0 else None,
        "median_bin_relative_mc_stat": float(np.median(relative)) if relative.size else None,
        "maximum_bin_relative_mc_stat": float(np.max(relative)) if relative.size else None,
        "nonempty_bins": int(relative.size),
        "bins_rel_stat_gt_20_percent": int(np.sum(relative > 0.20)),
        "bins_rel_stat_gt_50_percent": int(np.sum(relative > 0.50)),
        "bins_rel_stat_gt_100_percent": int(np.sum(relative > 1.00)),
    }


def _ratio_arrays(
    data: np.ndarray, prediction: np.ndarray, variance: np.ndarray,
    blinded: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positive = np.isfinite(prediction) & (prediction > 0) & np.isfinite(variance) & (variance >= 0)
    relative_mc_stat = np.full(prediction.shape, np.nan, dtype=np.float64)
    relative_mc_stat[positive] = np.sqrt(variance[positive]) / prediction[positive]
    valid_data = positive & ~blinded & np.isfinite(data)
    ratio = np.full(prediction.shape, np.nan, dtype=np.float64)
    data_error = np.full(prediction.shape, np.nan, dtype=np.float64)
    ratio[valid_data] = data[valid_data] / prediction[valid_data]
    data_error[valid_data] = np.sqrt(data[valid_data]) / prediction[valid_data]
    return ratio, data_error, relative_mc_stat, valid_data


def _plot_data_mc(
    path: Path, bins: np.ndarray, groups: list[dict[str, Any]],
    processes: dict[str, dict[str, np.ndarray]], data: np.ndarray,
    blinded: np.ndarray, blind_range: tuple[float, float],
    config: NnStudyConfig, qcd_cut: float, tthz_cut: float,
    ratio_limits: tuple[float, float],
) -> None:
    _setup_style()
    plt.style.use(hep.style.CMS)
    plt.rcParams.update({"font.size": 13})
    fig, (ax, ratio_ax) = plt.subplots(
        2, 1, dpi=150, figsize=(7.1, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
    )
    histograms = {name: entry["sumw"] for name, entry in processes.items()}
    prediction, overlay = _draw_mass_stack(ax, bins, groups, histograms)
    variance = np.sum([entry["sumw2"] for entry in processes.values()], axis=0)
    sigma = np.sqrt(variance)
    positive = prediction > 0
    ax.bar(bins[:-1][positive], 2 * sigma[positive], width=np.diff(bins)[positive],
           bottom=np.maximum(prediction[positive] - sigma[positive], 0),
           align="edge", color="gray", alpha=0.34, linewidth=0, label="MC stat.")
    centers = (bins[:-1] + bins[1:]) / 2
    visible_data = ~blinded & np.isfinite(data)
    ax.errorbar(centers[visible_data], data[visible_data],
                yerr=np.sqrt(data[visible_data]), fmt="o", color="black",
                markersize=3.5, capsize=1.5, linestyle="none", label="Data", zorder=7)
    ratio, data_error, relative_mc_stat, valid_data = _ratio_arrays(
        data, prediction, variance, blinded
    )
    ratio_ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ratio_ax.bar(bins[:-1][positive], 2 * relative_mc_stat[positive],
                 width=np.diff(bins)[positive], bottom=1 - relative_mc_stat[positive],
                 align="edge", color="gray", alpha=0.34, linewidth=0)
    ratio_ax.errorbar(centers[valid_data], ratio[valid_data],
                      yerr=data_error[valid_data], fmt="o", color="black",
                      markersize=3.5, capsize=1.5, linestyle="none", zorder=7)
    for axis in (ax, ratio_ax):
        axis.axvspan(*blind_range, color="gray", alpha=0.12, zorder=5)
        axis.set_xlim(float(bins[0]), float(bins[-1]))
    ax.text(sum(blind_range) / 2, 0.84, "Blinded",
            transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
    ax.set_ylim(0, 1.8 * max(float(np.max(overlay)), float(np.nanmax(data)), 1.0))
    ax.set_ylabel("Events")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.yaxis.get_offset_text().set_x(-0.09)
    ax.text(0.02, 0.97, config.plot_options.get("study_label", "0L V4") + " diagnostic\n"
            f"$s_{{QCD}}<{qcd_cut:.3f}$, $s_{{ttHZ}}>{tthz_cut:.2f}$",
            transform=ax.transAxes, va="top", fontsize="small")
    ax.legend(frameon=False, ncol=2, fontsize=7.5, loc="upper right",
              borderpad=0.1, columnspacing=0.2, labelspacing=0.1, handlelength=1.0)
    hep.cms.label(data=True, llabel="Internal", ax=ax, year=config.year,
                  lumi=f"{config.lumi_fb:.1f}", com=config.energy_tev, fontsize=15)
    stable = valid_data & (prediction >= max(1.0, 0.001 * float(np.max(prediction))))
    lower, upper = ratio_limits
    if np.any(stable):
        lower = min(lower, float(np.min(ratio[stable] - data_error[stable])) - 0.1)
        upper = max(upper, float(np.max(ratio[stable] + data_error[stable])) + 0.1)
    ratio_ax.set_ylim(max(0, lower), upper)
    ratio_ax.set_ylabel("Data / Pred.")
    ratio_ax.set_xlabel("Target FatJet regressed mass (generic) (GeV)")
    fig.subplots_adjust(left=0.14, right=0.97, top=0.89, bottom=0.10, hspace=0.05)
    _save(fig, path)


def _plot_mcstat_heatmap(
    path: Path, bins: np.ndarray, groups: list[dict[str, Any]],
    processes: dict[str, dict[str, np.ndarray]],
    config: NnStudyConfig, qcd_cut: float, threshold: float,
    blind_range: tuple[float, float],
) -> None:
    relative = np.full((len(groups), len(bins) - 1), np.nan, dtype=np.float64)
    for row, group in enumerate(groups):
        entry = processes[group["name"]]
        positive = entry["sumw"] > 0
        relative[row, positive] = (
            np.sqrt(entry["sumw2"][positive]) / entry["sumw"][positive]
        )
    colors = ListedColormap(["#deebf7", "#9ecae1", "#fdd49e", "#fc8d59", "#b30000"])
    colors.set_bad("#e6e6e6")
    colors.set_over("#67000d")
    norm = BoundaryNorm([0, 0.1, 0.2, 0.5, 1.0, 1.5], colors.N)
    _setup_style()
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(dpi=150, figsize=(10.0, max(5.2, 0.38 * len(groups) + 1.5)))
    image = ax.imshow(np.ma.masked_invalid(relative), aspect="auto", interpolation="nearest",
                      cmap=colors, norm=norm)
    ax.set_yticks(range(len(groups)), [group["label"] for group in groups])
    tick_bins = np.arange(0, len(bins), 5)
    ax.set_xticks(tick_bins - 0.5, [f"{bins[index]:g}" for index in tick_bins])
    for boundary in blind_range:
        ax.axvline(np.searchsorted(bins, boundary) - 0.5,
                   color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Target FatJet regressed mass (generic) (GeV)")
    ax.set_ylabel("MC process")
    fig.suptitle(
        f"MC stat. per process: $s_{{QCD}}<{qcd_cut:g}$, "
        f"$s_{{ttHZ}}>{threshold:.2f}$",
        y=0.97,
        fontsize=18,
    )
    bar = fig.colorbar(image, ax=ax, extend="max", pad=0.02,
                       ticks=[0.05, 0.15, 0.35, 0.75, 1.25])
    bar.ax.set_yticklabels(["<10%", "10-20%", "20-50%", "50-100%", ">100%"])
    bar.set_label(r"$\sqrt{\sum w^2}/\sum w$")
    hep.cms.label(data=False, llabel="", ax=ax, year=config.year,
                  lumi=f"{config.lumi_fb:.1f}", com=config.energy_tev, fontsize=15)
    fig.subplots_adjust(left=0.24, right=0.88, bottom=0.14, top=0.80)
    _save(fig, path)


def _format_mcstat_summary(payload: dict[str, Any], study_label: str = "0L V4") -> str:
    def number(value: Any) -> str:
        return "-" if value is None else f"{value:.5g}"

    lines = [
        f"=== {study_label} regressed-mass MC-stat diagnostic ===",
        "Real data: " + " + ".join(payload["dataset_trigger_order"])
        + "; Pepper trigger priority " + " > ".join(payload["dataset_trigger_order"]),
        "Weight: sample_norm * abs(weight); MC stat. = sqrt(sum w^2)",
        "Observed data in " + "-".join(f"{mass:g}" for mass in payload["blind_mass_range_gev"])
        + " GeV: blinded and not reported",
        "Unavailable MC groups: " + ", ".join(payload["unavailable_mc_groups"]),
    ]
    if payload["data_source"] is None:
        lines[1] = "Real data: not loaded (MC-only study)."
        lines[3] = "Observed data yields: not reported (MC-only study)."
    for cut, result in payload["thresholds"].items():
        lines.append(f"\nscore_qcd < {payload['qcd_cut']:g}, score_ttHZ > {cut}")
        for sideband, side in result["data_sidebands"].items():
            lines.append(f"  {sideband}: N_data={side['N_data']}, "
                         f"N_MC={side['N_MC']:.6g}, Data/MC={number(side['Data_over_MC'])}")
        for region in payload["regions"]:
            lines.append(f"  {region}: process | raw N | yield | sumw2 | sigma | N_eff | "
                         "median rel | max rel | nonempty | >20% | >50% | >100%")
            for name, process in result["processes"].items():
                stats = process["regions"][region]
                lines.append(
                    f"    {name} | {stats['raw_event_count']} | {number(stats['weighted_yield'])} | "
                    f"{number(stats['sumw2'])} | {number(stats['integrated_mc_stat'])} | "
                    f"{number(stats['integrated_neff'])} | "
                    f"{number(stats['median_bin_relative_mc_stat'])} | "
                    f"{number(stats['maximum_bin_relative_mc_stat'])} | "
                    f"{stats['nonempty_bins']} | {stats['bins_rel_stat_gt_20_percent']} | "
                    f"{stats['bins_rel_stat_gt_50_percent']} | "
                    f"{stats['bins_rel_stat_gt_100_percent']}"
                )
    for cut, regions in payload["background_mcstat_rankings"].items():
        for region, entries in regions.items():
            lines.append(f"\nWorst background MC stat., score_ttHZ > {cut}, {region}:")
            for rank, entry in enumerate(entries, start=1):
                lines.append(
                    f"  {rank}. {entry['name']}: yield={entry['weighted_yield']:.6g}, "
                    f"N_eff={number(entry['integrated_neff'])}, "
                    f"median={number(entry['median_bin_relative_mc_stat'])}, "
                    f"max={number(entry['maximum_bin_relative_mc_stat'])}, "
                    f"bins >50%={entry['bins_rel_stat_gt_50_percent']}"
                )
    return "\n".join(lines) + "\n"


def run_regressed_mass_fine_binning(
    *, config: NnStudyConfig, dataset: NnDataset,
    plot_dir: Path, summary_dir: Path, plot_suffix: str,
    include_data: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    if plot_suffix != ".png":
        raise ValueError("regressed-mass-fine-binning requires PNG output.")
    options, bins, blinded = _settings(config, validate_data=include_data)
    data_counts = load_data_cache(config) if include_data else None
    groups, unavailable, by_cut = _mc_histograms(config, dataset, bins, options)
    blind_low, blind_high = map(float, options["blind_mass_range"])
    regions = {
        "full": np.ones(len(bins) - 1, dtype=bool),
        "lower_sideband": bins[1:] <= blind_low,
        "blinded_window": blinded,
        "upper_sideband": bins[:-1] >= blind_high,
    }
    signal_names = {"ttHcc", "ttHbb", "ttZqq", "ttZcc", "ttZbb"}
    qcd_cut = float(options["qcd_cut"])
    qcd_tag = f"{qcd_cut:g}".replace(".", "p")
    output_dir = plot_dir / "ttHZ_qcd_discriminant"
    plots: list[str] = []
    thresholds: dict[str, Any] = {}
    rankings: dict[str, Any] = {}
    for cut_index, cut in enumerate(map(float, options["tthz_cuts"])):
        processes = by_cut[cut]
        tag = f"0p{int(round(cut * 100)):02d}"
        suffix = "data_blinded" if include_data else "stack"
        path = output_dir / f"regressed_mass__qcd_lt_{qcd_tag}__ttHZ_gt_{tag}__{suffix}.png"
        if data_counts is not None:
            _plot_data_mc(
                path, bins, groups, processes, data_counts[cut_index], blinded,
                (blind_low, blind_high), config, qcd_cut, cut,
                tuple(map(float, options["ratio_ylim"])),
            )
        else:
            _plot_mass_stack(
                path, bins, groups,
                {name: entry["sumw"] for name, entry in processes.items()}, config, cut,
                qcd_cut=qcd_cut,
                variance=np.sum([entry["sumw2"] for entry in processes.values()], axis=0),
            )
        plots.append(str(path))
        total = {
            field: np.sum([entry[field] for entry in processes.values()], axis=0)
            for field in ("sumw", "sumw2", "count")
        }
        process_metrics = {
            group["name"]: {
                "per_bin_neff": [
                    float(yield_**2 / variance) if variance > 0 else None
                    for yield_, variance in zip(processes[group["name"]]["sumw"],
                                                processes[group["name"]]["sumw2"])
                ],
                "label": group["label"],
                "is_signal": group["name"] in signal_names,
                "regions": {name: _region_metrics(processes[group["name"]], mask)
                            for name, mask in regions.items()},
            }
            for group in groups
        }
        sidebands = {}
        for name in (("lower_sideband", "upper_sideband") if include_data else ()):
            mask = regions[name]
            observed = int(np.sum(data_counts[cut_index, mask]))
            expected = float(np.sum(total["sumw"][mask]))
            sidebands[name] = {
                "N_data": observed, "N_MC": expected,
                "Data_over_MC": observed / expected if expected > 0 else None,
            }
        thresholds[f"{cut:.2f}"] = {
            "selection": f"score_qcd < {float(options['qcd_cut']):g} AND score_ttHZ > {cut:.2f}",
            "total_mc": {name: _region_metrics(total, mask) for name, mask in regions.items()},
            "processes": process_metrics,
            "data_sidebands": sidebands,
        }
        if cut in map(float, options["mcstat_detailed_cuts"]):
            heatmap = output_dir / f"mcstat_relative__qcd_lt_{qcd_tag}__ttHZ_gt_{tag}.png"
            _plot_mcstat_heatmap(heatmap, bins, groups, processes, config, qcd_cut, cut,
                                 (blind_low, blind_high))
            plots.append(str(heatmap))
            rankings[f"{cut:.2f}"] = {}
            for region in ("full", "blinded_window"):
                backgrounds = [
                    {"name": name, **entry["regions"][region]}
                    for name, entry in process_metrics.items()
                    if not entry["is_signal"]
                    and entry["regions"][region]["median_bin_relative_mc_stat"] is not None
                ]
                rankings[f"{cut:.2f}"][region] = sorted(
                    backgrounds,
                    key=lambda entry: entry["median_bin_relative_mc_stat"], reverse=True,
                )
    payload = {
        "schema_version": 1,
        "weighting": "sample_norm * abs(weight)",
        "data_source": "real 2024 Pepper-selected ParkingHH and JetMET; no pseudodata" if include_data else None,
        "dataset_trigger_order": options["data_samples"],
        "data_cache": str(_data_cache_path(config)) if include_data else None,
        "qcd_cut": qcd_cut,
        "mass_bin_edges_gev": bins.tolist(),
        "blind_mass_range_gev": [blind_low, blind_high],
        "observed_data_in_blinded_window_reported": False,
        "observed_data_blind_rule": f"{blind_low:g} <= mass <= {blind_high:g} GeV",
        "data_error": "sqrt(N_data), matching Pepper make_plots.py" if include_data else None,
        "mc_stat": "sqrt(sum of per-event analysis_weight squared)",
        "ratio": "unblinded N_data / total stacked MC (nominal signals included)" if include_data else None,
        "ratio_mc_stat_band": "1 +/- sqrt(total MC sumw2) / total MC sumw" if include_data else None,
        "regions": {
            "full": [float(bins[0]), float(bins[-1])],
            "lower_sideband": [float(bins[0]), blind_low],
            "blinded_window": [blind_low, blind_high],
            "upper_sideband": [blind_high, float(bins[-1])],
        },
        "unavailable_mc_groups": unavailable,
        "thresholds": thresholds,
        "background_mcstat_rankings": rankings,
        "plots": plots,
    }
    summary_name = "regressed_mass_mcstat" if include_data else "regressed_mass_mcstat__mc_only"
    write_json(summary_dir / f"{summary_name}.json", payload)
    write_text(summary_dir / f"{summary_name}.txt", _format_mcstat_summary(
        payload, config.plot_options.get("study_label", "0L V4")
    ))
    return plots, payload
