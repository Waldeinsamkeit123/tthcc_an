from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

from tthcc_an.config_loader import load_json_maybe_with_comments
from tthcc_an.trigger_efficiency.config import TriggerEfficiencyConfig, TriggerSample, TriggerVariable
from tthcc_an.trigger_efficiency.plotting import plot_trigger_group


def _load_normalization_metadata(
    gen_sumw_file: str | None,
    xsec_file: str | None,
) -> tuple[dict[str, float], dict[str, float]]:
    if gen_sumw_file is None or xsec_file is None:
        return {}, {}
    gen_sumw_path = Path(gen_sumw_file)
    xsec_path = Path(xsec_file)
    if not gen_sumw_path.exists():
        raise FileNotFoundError(f"gen_sumw file does not exist: {gen_sumw_path}")
    if not xsec_path.exists():
        raise FileNotFoundError(f"xsec file does not exist: {xsec_path}")

    gen_sumw_payload = load_json_maybe_with_comments(gen_sumw_path)
    xsec_payload = load_json_maybe_with_comments(xsec_path)
    gen_sumw_map: dict[str, float] = {}
    for dataset, value in gen_sumw_payload.items():
        if isinstance(value, dict):
            gen_sumw_map[str(dataset)] = float(value["gen_sumw"])
        else:
            gen_sumw_map[str(dataset)] = float(value)
    xsec_map = {str(dataset): float(value) for dataset, value in xsec_payload.items()}
    return gen_sumw_map, xsec_map


def _sample_normalization(
    sample: TriggerSample,
    lumi_fb: float | None,
    gen_sumw_map: dict[str, float],
    xsec_map: dict[str, float],
) -> tuple[float, float | None, float | None]:
    gen_sumw = gen_sumw_map.get(sample.dataset)
    xsec_fb = xsec_map.get(sample.dataset)
    if lumi_fb is None or gen_sumw is None or xsec_fb is None:
        return 1.0, gen_sumw, xsec_fb
    if gen_sumw == 0:
        raise ValueError(f"gen_sumw is zero for dataset '{sample.dataset}'.")
    return float(lumi_fb * xsec_fb / gen_sumw), gen_sumw, xsec_fb


def _extract_array(values: ak.Array) -> np.ndarray:
    if isinstance(values, ak.Array) and values.ndim > 1:
        values = ak.fill_none(ak.firsts(values), np.nan)
    return np.asarray(ak.to_numpy(values))


def _extract_variable_array(values: ak.Array, variable: TriggerVariable) -> np.ndarray:
    if isinstance(values, ak.Array) and values.ndim > 1:
        if variable.index is None:
            values = ak.fill_none(ak.firsts(values), np.nan)
        else:
            values = ak.pad_none(values, variable.index + 1, axis=1, clip=False)[:, variable.index]
            values = ak.fill_none(values, np.nan)
    elif variable.index is not None:
        raise ValueError(
            f"Variable {variable.name} requests index {variable.index} "
            f"from scalar branch {variable.branch}."
        )
    return np.asarray(ak.to_numpy(values))


def _histogram(values: np.ndarray, weights: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    hist, _ = np.histogram(values[finite], bins=bins, weights=weights[finite])
    hist_w2, _ = np.histogram(values[finite], bins=bins, weights=weights[finite] * weights[finite])
    return hist.astype(np.float64), hist_w2.astype(np.float64)


def _all_trigger_branches(config: TriggerEfficiencyConfig) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for triggers in [config.triggers, *config.trigger_groups.values(), *config.or_groups.values()]:
        for trigger in triggers:
            if trigger not in seen:
                seen.add(trigger)
                ordered.append(trigger)
    return ordered


def _empty_accumulators(config: TriggerEfficiencyConfig) -> dict[str, dict[str, Any]]:
    accumulators: dict[str, dict[str, Any]] = {}
    for variable in config.variables:
        n_bins = len(variable.bins) - 1
        accumulators[variable.name] = {
            "denominator": {sample.process: np.zeros(n_bins, dtype=np.float64) for sample in config.samples},
            "denominator_w2": {sample.process: np.zeros(n_bins, dtype=np.float64) for sample in config.samples},
            "numerator": {
                trigger: {sample.process: np.zeros(n_bins, dtype=np.float64) for sample in config.samples}
                for trigger in _all_trigger_branches(config)
            },
        }
        for or_name in config.or_groups:
            accumulators[variable.name]["numerator"][or_name] = {
                sample.process: np.zeros(n_bins, dtype=np.float64) for sample in config.samples
            }
    return accumulators


def _validate_group_triggers(config: TriggerEfficiencyConfig) -> None:
    empty_groups = [
        group_name
        for group_name, triggers in {**config.trigger_groups, **config.or_groups, **config.plot_groups}.items()
        if not triggers
    ]
    if empty_groups:
        raise ValueError(f"Trigger groups must not be empty: {empty_groups}")


def _read_and_accumulate(
    config: TriggerEfficiencyConfig,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    _validate_group_triggers(config)
    gen_sumw_map, xsec_map = _load_normalization_metadata(config.gen_sumw_file, config.xsec_file)
    accumulators = _empty_accumulators(config)
    sample_summaries: list[dict[str, Any]] = []

    variable_branches = [variable.branch for variable in config.variables]
    trigger_branches = _all_trigger_branches(config)
    required_branches = sorted(set(variable_branches + trigger_branches + [config.weight_branch]))

    for sample in config.samples:
        sample_norm, gen_sumw, xsec_fb = _sample_normalization(
            sample,
            config.lumi_fb,
            gen_sumw_map,
            xsec_map,
        )
        n_events = 0
        n_files_read = 0
        skipped_files_missing_tree = 0
        for file_path in sample.files:
            with uproot.open(file_path) as root_file:
                if config.tree_name not in root_file:
                    skipped_files_missing_tree += 1
                    continue
                tree = root_file[config.tree_name]
                tree_keys = set(tree.keys())
                missing = [branch for branch in required_branches if branch != config.weight_branch and branch not in tree_keys]
                if missing:
                    raise KeyError(f"Missing required branches in {file_path}: {', '.join(missing)}")
                branches_to_read = [branch for branch in required_branches if branch in tree_keys]
                n_files_read += 1

                for arrays in tree.iterate(branches_to_read, library="ak", step_size=config.uproot_step_size):
                    if not branches_to_read:
                        continue
                    chunk_size = len(arrays[branches_to_read[0]])
                    n_events += int(chunk_size)
                    columns: dict[str, ak.Array] = {}
                    for branch in branches_to_read:
                        columns[branch] = arrays[branch]
                    if config.weight_branch in columns:
                        raw_weight = np.asarray(_extract_array(columns[config.weight_branch]), dtype=np.float64)
                    else:
                        raw_weight = np.ones(chunk_size, dtype=np.float64)
                    analysis_weight = sample_norm * np.abs(raw_weight)

                    trigger_values = {
                        trigger: np.asarray(_extract_array(columns[trigger]), dtype=bool)
                        for trigger in trigger_branches
                    }
                    or_values = {
                        or_name: np.logical_or.reduce([trigger_values[trigger] for trigger in triggers])
                        for or_name, triggers in config.or_groups.items()
                    }
                    all_trigger_values = {**trigger_values, **or_values}

                    for variable in config.variables:
                        values = np.asarray(_extract_variable_array(columns[variable.branch], variable), dtype=np.float64)
                        bins = np.asarray(variable.bins, dtype=np.float64)
                        denominator, denominator_w2 = _histogram(values, analysis_weight, bins)
                        accumulators[variable.name]["denominator"][sample.process] += denominator
                        accumulators[variable.name]["denominator_w2"][sample.process] += denominator_w2
                        valid_base = np.isfinite(values) & np.isfinite(analysis_weight) & (analysis_weight > 0)
                        for trigger_name, trigger_mask in all_trigger_values.items():
                            pass_mask = valid_base & trigger_mask
                            numerator, _ = np.histogram(values[pass_mask], bins=bins, weights=analysis_weight[pass_mask])
                            accumulators[variable.name]["numerator"][trigger_name][sample.process] += numerator.astype(np.float64)

        sample_summaries.append(
            {
                "sample": sample.name,
                "dataset": sample.dataset,
                "process": sample.process,
                "label": sample.label,
                "n_files": len(sample.files),
                "n_files_read": n_files_read,
                "skipped_files_missing_tree": skipped_files_missing_tree,
                "n_events": n_events,
                "sample_norm": sample_norm,
                "gen_sumw": gen_sumw,
                "xsec_fb": xsec_fb,
            }
        )
    return accumulators, sample_summaries


def _efficiency(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def _efficiency_error(eff: float, den: float, den_w2: float) -> float:
    if den <= 0 or den_w2 <= 0 or not np.isfinite(eff):
        return float("nan")
    neff = den * den / den_w2
    if neff <= 0:
        return float("nan")
    return float(np.sqrt(max(eff * (1.0 - eff) / neff, 0.0)))


def _write_tables(
    config: TriggerEfficiencyConfig,
    accumulators: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    config.table_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    process_labels = {sample.process: sample.label for sample in config.samples}
    for variable in config.variables:
        bins = np.asarray(variable.bins, dtype=np.float64)
        variable_acc = accumulators[variable.name]
        for trigger in [*config.triggers, *config.or_groups.keys()]:
            for process in process_labels:
                denominator = variable_acc["denominator"][process]
                denominator_w2 = variable_acc["denominator_w2"][process]
                numerator = variable_acc["numerator"][trigger][process]
                for index in range(len(bins) - 1):
                    den = float(denominator[index])
                    num = float(numerator[index])
                    eff = _efficiency(num, den)
                    rows.append(
                        {
                            "variable": variable.name,
                            "variable_label": variable.label,
                            "trigger": trigger,
                            "process": process,
                            "process_label": process_labels[process],
                            "bin_low": float(bins[index]),
                            "bin_high": float(bins[index + 1]),
                            "denominator_weight": den,
                            "numerator_weight": num,
                            "efficiency": eff,
                            "efficiency_unc": _efficiency_error(eff, den, float(denominator_w2[index])),
                        }
                    )
    table_path = config.table_dir / "trigger_efficiencies.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _default_plot_groups(config: TriggerEfficiencyConfig) -> dict[str, list[str]]:
    groups = dict(config.trigger_groups)
    if config.or_groups:
        groups["or_summary"] = list(config.or_groups.keys())
    return groups


def _plot_groups_for_variable(config: TriggerEfficiencyConfig, variable: TriggerVariable) -> dict[str, list[str]]:
    if variable.plot_groups is None:
        return dict(config.plot_groups) if config.plot_groups else _default_plot_groups(config)

    available_groups = _default_plot_groups(config)
    available_groups.update(config.plot_groups)
    missing = [group for group in variable.plot_groups if group not in available_groups]
    if missing:
        raise KeyError(f"Variable {variable.name} references unknown plot groups: {missing}")
    return {group: available_groups[group] for group in variable.plot_groups}


def _make_plots(config: TriggerEfficiencyConfig, accumulators: dict[str, dict[str, Any]]) -> list[str]:
    config.plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[str] = []
    processes = [sample.process for sample in config.samples]
    process_labels = {sample.process: sample.label for sample in config.samples}
    for variable in config.variables:
        variable_acc = accumulators[variable.name]
        bins = np.asarray(variable.bins, dtype=np.float64)
        groups = _plot_groups_for_variable(config, variable)
        for group_name, triggers in groups.items():
            outpath = config.plot_dir / f"eff_vs_{variable.name}__{group_name}.png"
            plot_trigger_group(
                outpath=outpath,
                group_label=f"{group_name}: {variable.label}",
                triggers=triggers,
                variable_label=variable.label,
                bin_edges=bins,
                processes=processes,
                process_labels=process_labels,
                denominator=variable_acc["denominator"],
                denominator_w2=variable_acc["denominator_w2"],
                numerator=variable_acc["numerator"],
            )
            plot_paths.append(str(outpath))
    return plot_paths


def run_trigger_efficiency(config: TriggerEfficiencyConfig) -> dict[str, Any]:
    config.outdir.mkdir(parents=True, exist_ok=True)
    accumulators, sample_summaries = _read_and_accumulate(config)
    rows = _write_tables(config, accumulators)
    plot_paths = _make_plots(config, accumulators)

    summary = {
        "config_path": str(config.config_path),
        "outdir": str(config.outdir),
        "tree_name": config.tree_name,
        "weight_branch": config.weight_branch,
        "normalization": {
            "gen_sumw_file": config.gen_sumw_file,
            "xsec_file": config.xsec_file,
            "lumi_fb": config.lumi_fb,
            "weight": "sample_norm * abs(weight)",
        },
        "samples": sample_summaries,
        "triggers": list(config.triggers),
        "trigger_groups": dict(config.trigger_groups),
        "or_groups": dict(config.or_groups),
        "plot_groups": dict(config.plot_groups),
        "variables": [
            {
                key: value
                for key, value in {
                    "name": variable.name,
                    "label": variable.label,
                    "bins": list(variable.bins),
                    "branch": variable.branch if variable.branch != variable.name else None,
                    "index": variable.index,
                    "plot_groups": variable.plot_groups,
                }.items()
                if value is not None
            }
            for variable in config.variables
        ],
        "n_table_rows": len(rows),
        "plots": plot_paths,
        "tables": [str(config.table_dir / "trigger_efficiencies.csv")],
    }
    config.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
