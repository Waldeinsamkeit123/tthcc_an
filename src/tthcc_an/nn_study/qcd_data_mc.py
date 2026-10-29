from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from tthcc_an.config_loader import load_json_maybe_with_comments, strip_json_line_comments
from tthcc_an.nn_study.cache import load_nn_cache
from tthcc_an.nn_study.dataset import _to_numpy
from tthcc_an.nn_study.plotting import _save, _setup_style
from tthcc_an.nn_study.reporting import write_json, write_text

MASS = "TargetFatJet_regressed_mass_generic"
DATA_BRANCHES = ["score_qcd", "score_ttHZ", MASS, "weight", "run", "luminosityBlock", "event"]
FORBIDDEN_CUTS = {"QCDVeto", "ttHZScoreCut", "BDTQCDVeto", "BDTttHScoreCut"}


def _settings(config, blinding: str, allow_unblinded: bool):
    options = config.qcd_data_mc
    if not options.get("enabled") or "0l" not in config.channel.lower():
        raise ValueError("qcd_data_mc requires an enabled 0L configuration.")
    if blinding not in {"sidebands", "unblinded"}:
        raise ValueError("Unknown data blinding mode.")
    if blinding == "unblinded" and not allow_unblinded:
        raise ValueError("Unblinding requires --allow-unblinded-data and collaboration authorization.")
    if config.selection.strip() or config.max_files_per_sample is not None:
        raise ValueError("qcd-data-mc requires the complete baseline MC cache, without selection overrides.")
    for sample in config.samples:
        if sample.selection.strip() not in {"", "tt_hf_flavor == 5", "tt_hf_flavor != 5"}:
            raise ValueError(f"Only existing TT stitching is allowed, not additional cuts: {sample.name}")
    low, high = map(float, options["log10_range"])
    count = int(options["bins"])
    if not (np.isfinite(low) and np.isfinite(high) and low < high <= 0 and count > 0):
        raise ValueError("Invalid log10 score binning.")
    threshold = float(options["tthz_threshold"])
    blind_low, blind_high = map(float, options["blind_mass_range"])
    if not 0 <= threshold <= 1 or not np.isfinite(blind_low + blind_high) or blind_low >= blind_high:
        raise ValueError("Invalid ttHZ threshold or blind mass interval.")
    if any(not 0 < float(value) <= 1 for value in options["reference_thresholds"]):
        raise ValueError("Reference thresholds must lie in (0, 1].")
    return options, np.linspace(low, high, count + 1)


def _restrict_private_directory(directory, options):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    if directory.stat().st_mode & 0o077 or directory.stat().st_uid != os.getuid():
        raise PermissionError("Private output directory must be owned and accessible only by this user.")
    resolved = directory.resolve()
    if resolved.parts[1] != "eos":
        return
    endpoint = options["private_eos_mgm"]
    eos_path = str(resolved)
    if resolved.parts[2].startswith("home-"):
        eos_path = "/eos/user/" + resolved.parts[2][5:] + "/" + "/".join(resolved.parts[3:])
    command = ["eos", endpoint, "attr"]
    current = subprocess.run(command + ["-V", "get", "sys.acl", eos_path],
                             check=True, capture_output=True, text=True).stdout.strip()
    owner = str(os.getuid())
    denied = set()
    for rule in filter(None, current.split(",")):
        kind, identity, rights = rule.split(":", 2)
        if kind != "u" or not identity.isdigit():
            raise PermissionError("Private EOS output needs explicit review of non-user ACL rules.")
        if identity != owner:
            denied.add(identity)
    # EOS adds space-level grants after directory ACLs; explicit denials override them.
    acl = f"u:{owner}:rwx" + "".join(f",u:{uid}:!r!x!w" for uid in sorted(denied))
    subprocess.run(command + ["set", "sys.acl=" + acl, eos_path],
                   check=True, capture_output=True, text=True)
    user_acl = subprocess.run(command + ["-V", "get", "user.acl", eos_path],
                              check=True, capture_output=True, text=True).stdout.strip()
    if user_acl:
        raise PermissionError("Private EOS output needs review of an additional user.acl.")
    effective = subprocess.run(command + ["-V", "get", "sys.acl", eos_path],
                               check=True, capture_output=True, text=True).stdout.strip()
    rights_by_user = {}
    for rule in filter(None, effective.split(",")):
        kind, identity, rights = rule.split(":", 2)
        if kind != "u":
            raise PermissionError("Unexpected non-user EOS ACL on private output.")
        rights_by_user[identity] = rights_by_user.get(identity, "") + rights
    if any("!r" not in rights or "!x" not in rights for uid, rights in rights_by_user.items() if uid != owner):
        raise PermissionError("Inherited EOS read access remains on private output.")


def _output_paths(config, blinding: str):
    root = config.outdir / "plots" / "qcd_data_mc"
    if blinding == "unblinded":
        root = root / "private_unblinded"
        resolved = root.resolve()
        if {part.lower() for part in resolved.parts} & {"www", "public", "public_html", "wwwroot", "web"}:
            raise ValueError("Unblinded results cannot be written under public/web directories.")
        # EOS checks the direct parent, so protect all three output directories.
        for directory in (root, root / "summaries", root / "cache"):
            _restrict_private_directory(directory, config.qcd_data_mc)
        return root, root / "summaries", root / "cache" / "data_histograms.json"
    return root, config.outdir / "summaries" / "qcd_data_mc", config.outdir / "cache" / "qcd_data_mc__sidebands.json"


def _selection_masks(tthz, mass, options, blinding):
    baseline = np.ones(mass.shape, dtype=bool)
    if blinding == "sidebands":
        low, high = options["blind_mass_range"]
        baseline &= np.isfinite(mass) & ((mass < low) | (mass > high))
    threshold = float(options["tthz_threshold"])
    tag = f"ttHZ_ge_{threshold:.2f}".replace(".", "p")
    return {"baseline": baseline, tag: baseline & np.isfinite(tthz) & (tthz >= threshold)}


def _histogram(score, weights, edges):
    if not np.all(np.isfinite(weights)):
        raise ValueError("Non-finite event weights cannot be histogrammed.")
    finite = np.isfinite(score)
    positive = finite & (score > 0)
    logscore = np.full(score.shape, np.nan)
    logscore[positive] = np.log10(score[positive])
    masks = {
        "zero": finite & (score == 0), "negative": finite & (score < 0),
        "nonfinite": ~finite, "underflow": positive & (logscore < edges[0]),
        "overflow": positive & (logscore > edges[-1]),
    }
    visible = positive & (logscore >= edges[0]) & (logscore <= edges[-1])
    masks["in_range"] = visible
    return {
        "sumw": np.histogram(logscore[visible], bins=edges, weights=weights[visible])[0].tolist(),
        "sumw2": np.histogram(logscore[visible], bins=edges, weights=weights[visible] ** 2)[0].tolist(),
        "score_quality": {
            name: {"events": int(np.sum(mask)), "sumw": float(np.sum(weights[mask]))}
            for name, mask in masks.items()
        },
    }


def _add_histograms(target, source):
    for key in ("sumw", "sumw2"):
        target[key] = (np.asarray(target[key]) + source[key]).tolist()
    for name, values in source["score_quality"].items():
        for key, value in values.items():
            target["score_quality"][name][key] += value


def _load_pepper_json(path):
    # These Pepper inputs are JSON with line/block comments; preserve quoted paths.
    text = re.sub(
        r'"(?:\\.|[^"\\])*"|/\*.*?\*/',
        lambda match: match[0] if match[0].startswith('"') else " ",
        Path(path).read_text(), flags=re.S,
    )
    return json.loads(strip_json_line_comments(text))


def _trigger_definition(config, options):
    trigger = _load_pepper_json(options["data_trigger_config"])
    streams = options["data_samples"]
    if streams != trigger["dataset_trigger_order"] or not streams:
        raise ValueError("Data streams must match the official Pepper trigger priority.")
    previous = []
    selection = {}
    for stream in streams:
        positive = trigger["dataset_trigger_map"][stream]
        if not positive:
            raise ValueError(f"No triggers configured for {stream}.")
        selection[stream] = {"require_any": positive, "veto_any": list(previous)}
        previous.extend(positive)
    return {"priority": streams, "trigger_selection": selection,
            "source": options["data_trigger_config"],
            "source_sha256": hashlib.sha256(Path(options["data_trigger_config"]).read_bytes()).hexdigest()}


def _read_data_file(job):
    path, stream, tree_name, options, blinding = job
    with uproot.open(path) as source:
        if tree_name not in source:
            return None
        if "dsname" not in source or str(source["dsname"]) != stream or "Cutnames" not in source:
            raise ValueError(f"Invalid data provenance in {path}")
        cuts = set(ast.literal_eval(str(source["Cutnames"])))
        if not {"Trigger", "HasTargetFatJet", "NNEvaluation"} <= cuts or cuts & FORBIDDEN_CUTS:
            raise ValueError(f"Expected baseline process-stage V5 data in {path}")
        tree = source[tree_name]
        if any(name.startswith("cutflags") for name in tree.keys()):
            raise ValueError(f"Unresolved stored baseline flags in {path}")
        missing = set(DATA_BRANCHES) - set(tree.keys())
        if missing:
            raise ValueError(f"Missing data fields in {path}: {sorted(missing)}")
        arrays = tree.arrays(DATA_BRANCHES, library="ak")
        if not np.all(ak.to_numpy(arrays["weight"]) == 1):
            raise ValueError(f"Observed data have non-unit weights: {path}")
        mass = _to_numpy(arrays[MASS], flatten_first=True)
        tthz = np.asarray(ak.to_numpy(arrays["score_ttHZ"]), dtype=float)
        keep = _selection_masks(tthz, mass, options, blinding)["baseline"]
        # Blinded observations are discarded before any reportable aggregation or ID audit.
        return {
            name: (mass if name == MASS else np.asarray(ak.to_numpy(arrays[name])))[keep]
            for name in DATA_BRANCHES if name != "weight"
        }


def _data_histograms(config, options, edges, blinding, path, refresh):
    trigger = _trigger_definition(config, options)
    jobs = []
    digest = hashlib.sha256()
    for stream in options["data_samples"]:
        directory = Path(config.input_location) / stream
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing V5 data stream directory: {directory}")
        files = sorted(directory.glob("*.root"))
        if not files:
            raise FileNotFoundError(f"No data ROOT files in {directory}")
        for filename in files:
            info = filename.stat()
            digest.update(f"{filename}\0{info.st_size}\0{info.st_mtime_ns}\n".encode())
            jobs.append((filename, stream, config.tree_name, options, blinding))
    definition = {
        "schema_version": 1, "input_location": config.input_location,
        "tree": config.tree_name, "branches": DATA_BRANCHES, "trigger": trigger,
        "files_sha256": digest.hexdigest(), "files": len(jobs),
        "log10_bin_edges": edges.tolist(), "blinding": blinding,
        "blind_mass_range": options["blind_mass_range"],
        "tthz_threshold": options["tthz_threshold"],
    }
    if path.exists() and not refresh:
        cached = json.loads(path.read_text())
        if cached["definition"] != definition:
            raise ValueError("QCD data histogram cache inputs/settings changed; use --refresh-qcd-data-cache.")
        return cached
    empty = np.array([], dtype=float)
    labels = _selection_masks(empty, empty, options, blinding)
    histograms = {name: _histogram(empty, empty, edges) for name in labels}
    seen = {}
    duplicates = {stream: 0 for stream in options["data_samples"]}
    valid_files = {stream: 0 for stream in options["data_samples"]}
    print(f"Preparing {blinding} data histogram cache: {len(jobs)} files.", flush=True)
    with ThreadPoolExecutor(max_workers=int(options.get("data_workers", 8))) as pool:
        # Bound pending arrays while the exact event-ID audit consumes each batch.
        batches = (jobs[start:start + 64] for start in range(0, len(jobs), 64))
        results = (arrays for batch in batches for arrays in pool.map(_read_data_file, batch))
        for index, (job, arrays) in enumerate(zip(jobs, results), start=1):
            stream = job[1]
            if arrays is not None:
                valid_files[stream] += 1
                keys = zip(arrays["run"], arrays["luminosityBlock"], arrays["event"])
                keep = np.ones(len(arrays["score_qcd"]), dtype=bool)
                for row, key in enumerate(keys):
                    if key in seen:
                        if seen[key] != stream:
                            raise ValueError("Cross-stream duplicate events found: verify Pepper trigger orthogonality.")
                        keep[row] = False
                        duplicates[stream] += 1
                    else:
                        seen[key] = stream
                score = np.asarray(arrays["score_qcd"][keep], dtype=float)
                masks = _selection_masks(arrays["score_ttHZ"][keep], arrays[MASS][keep], options, blinding)
                for name, mask in masks.items():
                    _add_histograms(histograms[name], _histogram(score[mask], np.ones(np.sum(mask)), edges))
            if index % 1000 == 0:
                print(f"  {blinding}: {index}/{len(jobs)} data files checked", flush=True)
    if not all(valid_files.values()):
        raise ValueError("At least one data stream has no valid baseline Events trees.")
    result = {
        "definition": definition, "histograms": histograms,
        "overlap_checks": {
            "scope": blinding, "key": ["run", "luminosityBlock", "event"],
            "cross_stream_duplicates": 0, "within_stream_duplicates_removed": duplicates,
            "valid_files": valid_files,
            "provenance": "Each file verifies dsname, Trigger/HasTargetFatJet/NNEvaluation and absence of downstream score cuts; exact event-ID overlap checked in the reported population.",
        },
    }
    temporary = path.with_suffix(".tmp.json")
    write_json(temporary, result)
    temporary.replace(path)
    return result


def _mc_weights(config, dataset, options):
    normalization = _load_pepper_json(options["normalization_config"])
    lumi = float(normalization["luminosity"])
    if normalization.get("mc_lumifactors"):
        raise ValueError("This diagnostic expects process weights without luminosity normalization.")
    xsecs = _load_pepper_json(options["xsec_file"])
    sums = load_json_maybe_with_comments(config.gen_sumw_file)
    weights = np.empty_like(dataset.raw_weight, dtype=float)
    qcd = np.zeros(weights.shape, dtype=bool)
    vjets = np.zeros(weights.shape, dtype=bool)
    table = []
    definitions = {sample.name: sample for sample in config.samples}
    if set(np.unique(dataset.sample_index)) - set(range(len(dataset.sample_summaries))):
        raise ValueError("Invalid MC sample indices in the prepared cache.")
    for index, summary in enumerate(dataset.sample_summaries):
        sample = definitions[summary["name"]]
        gen = sums[sample.dataset]
        gen = float(gen["gen_sumw"] if isinstance(gen, dict) else gen)
        xsec = xsecs[sample.dataset]
        if isinstance(xsec, str):
            if not re.fullmatch(r"[0-9eE+*/().\s-]+", xsec):
                raise ValueError(f"Non-arithmetic cross-section expression for {sample.dataset}")
            xsec = eval(xsec, {"__builtins__": {}})
        xsec = float(xsec)
        if not np.isfinite(gen) or gen == 0 or not np.isfinite(xsec) or not np.isfinite(lumi) or lumi <= 0:
            raise ValueError(f"Invalid physical normalization for {sample.name}")
        mask = dataset.sample_index == index
        weights[mask] = dataset.raw_weight[mask] * lumi * xsec / gen
        is_qcd = sample.name.startswith(("QCD-", "QCD_"))
        is_vjets = sample.name.startswith(("Wto2Q", "Zto2Q", "WJets", "ZJets", "DY"))
        qcd[mask] = is_qcd
        vjets[mask] = is_vjets
        table.append({"name": sample.name, "dataset": sample.dataset, "stitching": sample.selection,
                      "group": "QCD_multijet" if is_qcd else ("V+jets" if is_vjets else "other_nonQCD"),
                      "xsec_fb": xsec, "gen_sumw": gen, "sample_norm": lumi * xsec / gen})
    if not np.all(np.isfinite(weights)):
        raise ValueError("Invalid signed physical MC event weights.")
    return weights, qcd, vjets, table, lumi


def _ratios(data, prediction, variance):
    valid = np.isfinite(prediction) & (prediction > 0)
    ratio = np.full(data.shape, np.nan)
    error = np.full(data.shape, np.nan)
    relative = np.full(data.shape, np.nan)
    ratio[valid] = data[valid] / prediction[valid]
    error[valid] = np.sqrt(data[valid]) / prediction[valid]
    relative[valid] = np.sqrt(variance[valid]) / prediction[valid]
    return ratio, error, relative, valid


def _nullable(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def _plot(path, summary, config):
    _setup_style()
    plt.style.use(hep.style.CMS)
    plt.rcParams.update({"font.size": 12})
    fig, (ax, ratio_ax) = plt.subplots(2, 1, sharex=True, figsize=(9, 7.5), dpi=150,
                                     gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.06})
    edges = np.asarray(summary["log10_bin_edges"])
    centers = (edges[:-1] + edges[1:]) / 2
    data = np.asarray(summary["data_counts"])
    ax.errorbar(centers, data, yerr=np.sqrt(data), fmt="o", color="black", markersize=3, label="Data", zorder=6)
    ratio_positive = []
    upper_max = max(float(np.max(data + np.sqrt(data))), 1.0)
    for key, color, label in (("nonQCD_MC", "royalblue", "non-QCD MC"),
                               ("allMC", "orangered", "all MC")):
        prediction = np.asarray(summary[key]["sumw"])
        variance = np.asarray(summary[key]["sumw2"])
        ax.stairs(np.where(prediction > 0, prediction, np.nan), edges, color=color, label=label, linewidth=1.6)
        ratio, error, relative, valid = _ratios(data, prediction, variance)
        visible = valid & (ratio > 0)
        ratio_ax.errorbar(centers[visible], ratio[visible], yerr=error[visible],
                          fmt="o", color=color, markersize=2.6, linewidth=0.9)
        ratio_positive.extend(ratio[visible].tolist())
        upper_max = max(upper_max, float(np.max(prediction + np.sqrt(variance))))
        if key == "allMC":
            low = np.where(valid, np.maximum(prediction - np.sqrt(variance), 0.1), np.nan)
            high = np.where(valid, prediction + np.sqrt(variance), np.nan)
            ax.fill_between(edges, np.r_[low, low[-1]], np.r_[high, high[-1]], step="post",
                            color="gray", alpha=0.25, label="MC stat.")
            low = np.where(valid, np.maximum(1 - relative, 1e-6), np.nan)
            high = np.where(valid, 1 + relative, np.nan)
            ratio_ax.fill_between(edges, np.r_[low, low[-1]], np.r_[high, high[-1]], step="post",
                                  color="gray", alpha=0.25)
    for cut in summary["reference_thresholds"]:
        x = np.log10(cut)
        for axis in (ax, ratio_ax):
            axis.axvline(x, color="0.5", linestyle=":", linewidth=0.8)
        ax.text(x, 0.52, f"{cut:g}", rotation=90, transform=ax.get_xaxis_transform(),
                ha="right", va="bottom", fontsize=8, color="0.4")
    ax.set_yscale("log")
    ax.set_ylim(0.1, upper_max * 50)
    ax.set_ylabel("Events / log$_{10}$ score bin")
    ax.legend(loc="upper right", ncol=2, fontsize=10, frameon=False)
    ax.text(0.02, 0.97, summary["selection_label"] + "\n" + summary["mass_selection_label"]
            + "\nIncomplete MC prediction; no QCD cut", transform=ax.transAxes,
            va="top", fontsize=10)
    hep.cms.label(data=True, llabel="Internal", ax=ax, year=config.year,
                  lumi=f"{summary['luminosity_fb']:g}", com=config.energy_tev, fontsize=15)
    ratio_ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ratio_ax.set_yscale("log")
    ratio_ax.set_ylim(min([0.3, *ratio_positive]) / 2, max([3.0, *ratio_positive]) * 2)
    ratio_ax.set_ylabel("Data / MC")
    ratio_ax.set_xlabel(r"$\log_{10}(\mathrm{score}_{qcd})$")
    ratio_ax.set_xlim(edges[0], edges[-1])
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.11, top=0.90)
    _save(fig, path)


def run_qcd_data_mc(*, config, mc_cache_path, blinding="sidebands", allow_unblinded=False,
                    selection="both", refresh_data_cache=False):
    options, edges = _settings(config, blinding, allow_unblinded)
    if selection not in {"baseline", "tthz", "both"}:
        raise ValueError("Unknown QCD Data/MC selection.")
    dataset, metadata = load_nn_cache(config=config, cache_path=mc_cache_path,
                                      score_names=["qcd", "ttHZ"], analysis_branches=[MASS])
    weights, qcd, vjets, samples, lumi = _mc_weights(config, dataset, options)
    plot_dir, summary_dir, data_path = _output_paths(config, blinding)
    data = _data_histograms(config, options, edges, blinding, data_path, refresh_data_cache)
    available = {p.name for p in Path(config.input_location).iterdir() if p.is_dir()}
    extra = sorted(available - {s["name"] for s in samples} - set(options["data_samples"]))
    masks = _selection_masks(dataset.scores["ttHZ"], dataset.analysis_columns[MASS], options, blinding)
    plots, summaries = [], []
    for name, mask in masks.items():
        if (selection == "baseline" and name != "baseline") or (selection == "tthz" and name == "baseline"):
            continue
        hists = {key: _histogram(dataset.scores["qcd"][mask & group], weights[mask & group], edges)
                 for key, group in {"nonQCD_MC": ~qcd, "QCD_multijet": qcd, "Vjets": vjets}.items()}
        combined = json.loads(json.dumps(hists["nonQCD_MC"]))
        _add_histograms(combined, hists["QCD_multijet"])
        direct = _histogram(dataset.scores["qcd"][mask], weights[mask], edges)
        for field in ("sumw", "sumw2"):
            np.testing.assert_allclose(combined[field], direct[field], rtol=1e-9, atol=1e-6)
        hists["allMC"] = combined
        observed = np.asarray(data["histograms"][name]["sumw"])
        ratios = {}
        for key in ("nonQCD_MC", "allMC"):
            ratio, error, band, valid = _ratios(observed, np.asarray(hists[key]["sumw"]), np.asarray(hists[key]["sumw2"]))
            ratios[key] = {"values": _nullable(ratio), "data_stat_error": _nullable(error),
                           "mc_relative_stat": _nullable(band), "valid_bins": valid.tolist(),
                           "invalid_bin_count": int(np.sum(~valid)),
                           "nonpositive_ratio_bins_omitted_on_log_axis": int(np.sum(valid & (ratio <= 0)))}
        blind_low, blind_high = options["blind_mass_range"]
        mass_label = (f"Mass sidebands only ({blind_low:g}-{blind_high:g} GeV excluded)" if blinding == "sidebands"
                      else "PRIVATE: mass-inclusive, unblinded")
        payload = {
            "schema_version": 1, "blinding": blinding, "study": name,
            "selection_label": "Baseline" if name == "baseline" else f"score_ttHZ >= {options['tthz_threshold']:g}",
            "mass_selection_label": mass_label,
            "mass_selection": f"mass < {blind_low:g} OR mass > {blind_high:g}" if blinding == "sidebands" else "none",
            "qcd_score_cut": None, "flavour_category_selection": None,
            "baseline": "Pepper V5 process-stage selection, no postproc score or category cuts",
            "log10_bin_edges": edges.tolist(), "linear_score_bin_edges": (10.0 ** edges).tolist(),
            "reference_thresholds": options["reference_thresholds"], "references_are_cuts": False,
            "data_counts": observed.astype(int).tolist(), "data_error": "sqrt(N), Pepper convention",
            "data_score_quality": data["histograms"][name]["score_quality"],
            **hists, "ratios": ratios,
            "grouping": "QCD_multijet = originating QCD-* / QCD_* samples only; nonQCD = all other configured MC, including W/Z+jets; allMC = nonQCD + QCD_multijet",
            "included_samples": samples, "available_mc_not_in_cache": extra,
            "prediction_complete": False,
            "missing_processes": ["single top", "additional ttV", "ttH non-bb outside configured ttHcc"],
            "closure_claim": "Preliminary diagnostic of the configured cache; incomplete MC, no closure fit",
            "luminosity_fb": lumi, "mc_weight": "lumi_fb * xsec_fb / gen_sumw * signed stored process weight",
            "weight_provenance": options["weight_provenance"],
            "correction_limitations": options["correction_limitations"],
            "normalization_config": options["normalization_config"], "xsec_file": options["xsec_file"],
            "gen_sumw_file": config.gen_sumw_file,
            "mc_cache": str(mc_cache_path), "mc_cache_definition_hash": metadata["definition_hash"],
            "data_cache": str(data_path), "data_streams": options["data_samples"],
            "trigger_stream_definition": data["definition"]["trigger"], "overlap_checks": data["overlap_checks"],
        }
        suffix = "sidebands_blinded" if blinding == "sidebands" else "unblinded"
        stem = f"qcd_score_data_mc__{name}__{suffix}"
        image_path, json_path = plot_dir / f"{stem}.png", summary_dir / f"{stem}.json"
        json.dumps(payload, allow_nan=False)
        _plot(image_path, payload, replace(config, lumi_fb=lumi))
        write_json(json_path, payload)
        text_path = json_path.with_suffix(".txt")
        write_text(text_path, "\n".join([
            payload["selection_label"], mass_label, "No QCD cut; no flavour/postproc category cuts.",
            payload["mc_weight"], f"Luminosity: {lumi:g} fb^-1", payload["weight_provenance"],
            payload["closure_claim"], "Available MC absent from cache: " + ", ".join(extra),
            "bin_low bin_high Data nonQCD QCD allMC Data/nonQCD Data/allMC",
            *[f"{edges[i]:.3f} {edges[i+1]:.3f} {int(observed[i])} "
              f"{hists['nonQCD_MC']['sumw'][i]:.8g} {hists['QCD_multijet']['sumw'][i]:.8g} "
              f"{hists['allMC']['sumw'][i]:.8g} {ratios['nonQCD_MC']['values'][i]} {ratios['allMC']['values'][i]}"
              for i in range(len(observed))],
        ]) + "\n")
        plots.append(str(image_path))
        summaries.extend([str(json_path), str(text_path)])
    return {"operation": "qcd-data-mc", "plots": plots, "summaries": summaries}
