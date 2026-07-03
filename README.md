# tthcc_an

Analysis repository for Run 3 boosted `ttHcc` / `ttHbb` studies, including
AK8 boosted Higgs tagger studies and a prototype event-level BDT workflow.

## Overview

This repository is used for additional studies of the boosted `ttHcc` analysis.
The original analysis framework is based on:

- `pepper-tth`: https://gitlab.cern.ch/tthcc-run-3/pepper-tth.git

The current focus of this repository is:

- AK8 boosted Higgs tagger studies using Pepper-produced ROOT ntuples
- a prototype 2024 `0L` event-level BDT workflow for `ttH`-enriched selection studies

## Repository Layout

- `config/`: study configuration files
  - `config/event_bdt/`: event-BDT sample and training configs
- `scripts/`: runnable entrypoints and submission helpers
- `src/`: core analysis code
  - `src/tthcc_an/event_bdt/`: event-BDT prototype modules
- `outputs/`: local study outputs
- `condor/`: prepared HTCondor workflows and chunk outputs

## Main Entry Points

This repository currently has two main user-facing workflows:

- boosted Higgs tagger studies:

```bash
python scripts/run_boosted_higgs_tagger_study.py
```

- event-level BDT prototype:

```bash
python scripts/run_event_bdt.py
```

- signal trigger-efficiency studies:

```bash
python scripts/run_trigger_efficiency.py
```

## Boosted Higgs Tagger Study

The boosted-study script is:

```bash
python scripts/run_boosted_higgs_tagger_study.py
```

It reads Pepper-produced ROOT files with an `Events` tree containing `FatJet_*`
branches such as:

- truth matching counters: `FatJet_n_hcc`, `FatJet_n_hbb`, `FatJet_n_t1b`, ...
- gParT3 branches: `FatJet_globalParT3_Xcc`, `FatJet_globalParT3_Xbb`, `FatJet_globalParT3_QCD`, ...
- ParticleNet branches: `FatJet_particleNetWithMass_HccvsQCD`, `FatJet_particleNet_XccVsQCD`, ...
- event weight branch: `weight`

Main features:

- supports `H->cc`, `H->bb`, and combined `H->bb / cc` (`higgs`) targets
- reads multiple `ttHcc`, `ttHbb`, and `ttbar` samples from JSON config
- only reads the `FatJet` branches needed for the requested scores
- supports `all_jets`, `leading_pt`, `mass_window_all_jets`, and `mass_window_leading_pt`
- studies working points on mass-window-selected jets by default
- computes weighted working points at chosen signal efficiencies
- writes tables, text summaries, JSON summaries, ROC curves, and score plots
- writes 2D cumulative contour studies in the `gParT3 Higgs vs QCD` vs `gParT3 Hbb vs Hcc` and `gParT3 Higgs vs QCD` vs `gParT3 Xbb vs Xcc` planes
- stores a reusable `plot_input.npz` cache for later plot-only redraws
- uses histogram chunk payloads by default for low-memory HTCondor merges
- uses `mplhep` with CMS style for figures

Currently supported scores:

- `gpart_h2cc = Xcc / (Xcc + QCD)`
- `gpart_h2bb = Xbb / (Xbb + QCD)`
- `gpart_xbb_vs_xcc = Xbb / (Xbb + Xcc)`
- `gpart_hbb_vs_hcc = Xbb / (Xbb + Xcc)`
- `gpart_higgs_vs_qcd = (Xbb + Xcc) / (Xbb + Xcc + QCD)`
- `pnet_hcc = particleNetWithMass_HccvsQCD`
- `pnet_xcc = particleNet_XccVsQCD`
- `pnetlegacy_xcc = particleNetLegacy_Xcc`

Default `auto` score selection is target-dependent:

- `hcc -> gpart_h2cc`
- `hbb -> gpart_h2bb, gpart_xbb_vs_xcc, gpart_hbb_vs_hcc`
- `higgs -> gpart_higgs_vs_qcd`

## Weighting

The weighted study uses:

- `gen_sumw`
- `xsec`
- `lumi_fb`
- Pepper event branch `weight`

The sample normalization is:

```text
sample_norm = lumi_fb * xsec / gen_sumw
```

The script keeps two jet-level weights:

- analysis weight: `sample_norm * abs(weight)`
- signed bookkeeping weight: `sample_norm * weight`

The analysis weight is used for:

- weighted signal/background efficiencies
- working point cuts
- ROC curves
- score-distribution plots

The signed weight is kept for bookkeeping in the output tables.

## Truth Categories

Each selected AK8 jet is assigned one truth label:

- `hcc_pure`
- `hcc_contaminated`
- `hcc_partial`
- `hbb_pure`
- `hbb_contaminated`
- `hbb_partial`
- `top`
- `other`

For `target = hcc`:

- signal = `hcc_pure`
- background = `hcc_contaminated + hcc_partial + hbb_* + top + other`

For `target = hbb`:

- signal = `hbb_pure`
- background = `hbb_contaminated + hbb_partial + hcc_* + top + other`

For `target = higgs`:

- signal = `hbb_pure + hcc_pure`
- background = `hbb_contaminated + hbb_partial + hcc_contaminated + hcc_partial + top + other`

## 2D Contour Plots

The study writes two dedicated 2D contour plots:

- `plots/gpart_higgs_vs_qcd__vs__gpart_hbb_vs_hcc__contours.png`
- `plots/gpart_higgs_vs_qcd__vs__gpart_xbb_vs_xcc__contours.png`

These plots are not tied to one study target. They show the weighted fat-jet
distributions in two score planes:

- `gpart_higgs_vs_qcd` vs `gpart_hbb_vs_hcc`
- `gpart_higgs_vs_qcd` vs `gpart_xbb_vs_xcc`

The three plotted truth populations are:

- `hbb_pure`
- `hcc_pure`
- `Others = everything except hbb_pure and hcc_pure`

The contours are cumulative enclosed-fraction contours:

- each category is histogrammed separately in the fixed range `[0, 1] x [0, 1]`
- the 2D histogram is Gaussian-smoothed before thresholding
- the smoothed histogram is normalized to unit integral for that category
- contour thresholds correspond to enclosed fractions `10%` through `90%`
- the plot uses filled `contourf` bands plus overlaid contour lines

For each contour study, the code also writes summary text/JSON products:

- `summaries/...__region_efficiencies.txt`
- `summaries/...__fixed_other_efficiency_scan.txt`
- `summaries/...__fixed_x_ycut_scan.txt`

## Config Files

Available configs in this repository:

- [config/samples.example.json](/eos/user/h/hanw/ttHcc/tthcc_an/config/samples.example.json)
- [config/samples_2024_add_nonttbarmatch_allmc.json](/eos/user/h/hanw/ttHcc/tthcc_an/config/samples_2024_add_nonttbarmatch_allmc.json)
- [config/event_bdt/samples_2024_0l_jecs_v1.json](/eos/user/h/hanw/ttHcc/tthcc_an/config/event_bdt/samples_2024_0l_jecs_v1.json)
- [config/event_bdt/train_ttHcc_0l_3class_baseline_jecs.json](/eos/user/h/hanw/ttHcc/tthcc_an/config/event_bdt/train_ttHcc_0l_3class_baseline_jecs.json)
- [config/event_bdt/train_ttHcc_0l_4class_baseline_jecs.json](/eos/user/h/hanw/ttHcc/tthcc_an/config/event_bdt/train_ttHcc_0l_4class_baseline_jecs.json)

The 2024 config already contains:

- the sample list
- all MC datasets listed in the corresponding `gen_sumw` JSON
- process-group labels for `ttbar`, `tt+bb`, `ttV`, `tt+ll`, `single top`, `W+jets`, `Z+jets`, `QCD`, and `ttH*`
- the 2024 luminosity
- the `gen_sumw` JSON path
- the cross section JSON path
- default study options such as `outdir`, `targets`, `scores`, `sig_effs`, and selection cuts
- default plot styling options

The study output path is controlled by `study.outdir` in the JSON config.

- if `study.outdir` is relative, it is resolved from the repository root
- if `study.outdir` is absolute, for example `/eos/user/...`, it is used directly
- `--outdir` is only needed when you want a temporary override from the command line

Command-line options now follow this priority:

- explicit CLI argument
- value from the JSON config
- built-in fallback

For normal analysis updates, you only need to edit the JSON files in `config/`.
The Python module [src/tthcc_an/config_loader.py](/eos/user/h/hanw/ttHcc/tthcc_an/src/tthcc_an/config_loader.py:133) is only the internal loader/validator for those JSON files.

At the moment the shipped 2024 config defaults to:

- `targets = ["hcc", "hbb", "higgs"]`
- `scores = ["auto"]`
- `candidate_strategy = "mass_window_all_jets"`
- `xbb_vs_xcc_region_preset = "loose"`

For the `gpart_higgs_vs_qcd` vs `gpart_xbb_vs_xcc` contour plot, two region
presets are available:

- `loose`: `Hcc x > 0.7333`, `Hbb x > 0.9133`, corresponding to the `QCD` mistag reference at `1%`
- `tight`: `Hcc x > 0.9467`, `Hbb x > 0.9467`, corresponding to the `QCD` mistag reference at `0.1% / 0.5%`

The shipped 2024 config currently points to:

- event ntuples under `/eos/user/h/hanw/ttHcc/pepper_data/2024/add_nonttbarmatch_event`
- normalization metadata in `/eos/user/h/hanw/ttHcc/pepper_data/2024/add_nonttbarmatch/gen_sumws.json`

If you update the ROOT ntuples or `gen_sumws.json`, rerun the full study or a
chunked Condor workflow. `--plot-only` only redraws from an existing
`plot_input.npz` cache and does not pick up changed inputs.

## Common 2024 Commands

Quick smoke test after updating ntuples or `gen_sumws.json`:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh
cd /eos/user/h/hanw/ttHcc/tthcc_an

python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --max-files-per-sample 1 \
  --xbb-vs-xcc-region-preset tight \
  --outdir outputs/boosted_higgs_tagger_study_2024_smoke_tight
```

Full local rerun for the `QCD mistag eff = 0.1% / 0.5%` `xbb-vs-xcc` study:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh
cd /eos/user/h/hanw/ttHcc/tthcc_an

python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --xbb-vs-xcc-region-preset tight \
  --outdir outputs/boosted_higgs_tagger_study_2024_tight
```

Current plot-only refresh for the existing tight study cache at
`outputs/boosted_higgs_tagger_study_2024_tight_20260525`:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh
cd /eos/user/h/hanw/ttHcc/tthcc_an

python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --plot-only \
  --plot-input outputs/boosted_higgs_tagger_study_2024_tight_20260525/plot_input.npz \
  --outdir outputs/boosted_higgs_tagger_study_2024_tight_20260525 \
  --xbb-vs-xcc-region-preset tight
```

Recommended chunked Condor rerun for the same study:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh
cd /eos/user/h/hanw/ttHcc/tthcc_an

python scripts/submit_boosted_higgs_tagger_condor.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --workflow-mode chunked \
  --files-per-chunk 20 \
  --condor-dir condor/boosted_higgs_tagger_2024_tight \
  --outdir outputs/boosted_higgs_tagger_study_2024_tight \
  --submit \
  -- --xbb-vs-xcc-region-preset tight
```

## 0L Event-BDT Prototype

The repository also contains an actively used event-level BDT workflow for the
2024 `0L` study on the JEC-updated event ntuples:

- training sample config: [config/event_bdt/samples_2024_0l_jecs_v1.json](/eos/user/h/hanw/ttHcc/tthcc_an/config/event_bdt/samples_2024_0l_jecs_v1.json)
- current 3-class baseline: [config/event_bdt/train_ttHcc_0l_3class_baseline_jecs.json](/eos/user/h/hanw/ttHcc/tthcc_an/config/event_bdt/train_ttHcc_0l_3class_baseline_jecs.json)
- current 4-class baseline: [config/event_bdt/train_ttHcc_0l_4class_baseline_jecs.json](/eos/user/h/hanw/ttHcc/tthcc_an/config/event_bdt/train_ttHcc_0l_4class_baseline_jecs.json)
- CLI: [scripts/run_event_bdt.py](/eos/user/h/hanw/ttHcc/tthcc_an/scripts/run_event_bdt.py:1)

Current scope of the workflow:

- channel: `0L`
- classification is defined by `study.training_classes`
- each class entry provides:
  - `name`: machine name used in output branches and plot filenames
  - `label`: human-readable label used in plots
  - `group`: `signal` or `background`, used for binary fallback and reweighting
  - `processes`: the sample-process names assigned to that class
- eval-only samples remain outside training and can still appear in process-level plots through `study.eval_processes_extra`
- data source: Pepper-produced `Events` trees under `/eos/user/h/hanw/ttHcc/pepper_data/2024/JECs_v1_events`
- normalization metadata: `/eos/user/h/hanw/ttHcc/pepper_data/2024/JECs_v1/gen_sumws.json`

The current `samples_2024_0l_jecs_v1.json` keeps these processes active:

- `qcd`
- `ttbar`
- `ttHbb`
- `ttHcc`
- `ttH_nonbb` as `eval_only`

Wider backgrounds such as `W+jets`, `Z+jets`, `ttV`, `single top`, `ttbb`, and
`ttll` are still kept in the JSON as commented entries. The training-class
definitions already allow `ttbb` / `ttll`, so if you uncomment them in the
sample config they automatically flow into the `ttbar` class.

Two multiclass setups are currently maintained:

- `3-class`: `tth = ttHbb + ttHcc`, `ttbar = ttbar + ttbb + ttll`, `qcd = qcd`
  with output `outputs/event_bdt_ttHcc_0l_3class_baseline_jecs_v3`
- `4-class`: `tthbb = ttHbb`, `tthcc = ttHcc`, `ttbar = ttbar + ttbb + ttll`,
  `qcd = qcd` with output `outputs/event_bdt_ttHcc_0l_4class_baseline_jecs_v1`

The baseline configs use the same simple `0L` preselection:

- `ntargetfatjet >= 1`
- `TargetFatJet_pt >= 300`
- `|TargetFatJet_eta| <= 2.4`

Both active configs currently use:

- `k_folds = 5`
- reweighting in `TargetFatJet_pt`, `HT`, and `CleanedHT`
- analysis branches `TargetFatJet_msoftdrop`,
  `TargetFatJet_regressed_mass_generic`, and
  `TargetFatJet_regressed_mass_x2p`

The current feature set mixes event-level, jet-level, and target-fatjet
observables such as:

- `MET_pt`, `HT`, `CleanedHT`
- `Cleaned_HTb`, `Cleaned_HTc`, `Cleaned_HTcb`
- `CleanedJet_nbtag`, `CleanedJet_nctag`, `ncleanedjet`
- leading cleaned jets `CleanedJet_pt__1..4` and `CleanedJet_tag__1..4`
- `TargetFatJet_pt`, `TargetFatJet_bbtagged`, `TargetFatJet_cctagged`
- `minDR_b`, `DR_Tarb1`, `DR_Tarb2`, `minDR_TarClean`,
  `minDEta_TarClean`, `minDPhi_TarClean`

XGBoost configuration is now driven from `study.training_classes`:

- `len(training_classes) == 2` selects binary mode
- `len(training_classes) >= 3` selects multiclass mode
- if `xgboost.objective` is omitted, it defaults to `binary:logistic` or `multi:softprob`
- if `xgboost.eval_metric` is omitted, it defaults to `['auc', 'logloss']` or `['mlogloss', 'merror']`
- if `xgboost.num_class` is omitted in multiclass mode, it is set automatically from the number of classes
- if a non-eval sample is active in the training sample config, it must be covered by one of the configured classes

### Event-BDT Commands

Recommended current `3-class` workflow:

1. set up the runtime environment and enter the repository:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh
cd /eos/user/h/hanw/ttHcc/tthcc_an
```

2. build or refresh the prepared cache when branches or selections changed:

```bash
python scripts/run_event_bdt.py \
  prepare \
  --config config/event_bdt/train_ttHcc_0l_3class_baseline_jecs.json \
  --force
```

3. run the current `3-class` training:

```bash
python scripts/run_event_bdt.py \
  train \
  --config config/event_bdt/train_ttHcc_0l_3class_baseline_jecs.json
```

4. redraw the `3-class` diagnostics from the saved payloads:

```bash
python scripts/run_event_bdt.py \
  evaluate \
  --config config/event_bdt/train_ttHcc_0l_3class_baseline_jecs.json
```

5. run the analogous `4-class` training or refresh its evaluation products:

```bash
python scripts/run_event_bdt.py \
  train \
  --config config/event_bdt/train_ttHcc_0l_4class_baseline_jecs.json

python scripts/run_event_bdt.py \
  evaluate \
  --config config/event_bdt/train_ttHcc_0l_4class_baseline_jecs.json
```

`evaluate` does not retrain. It redraws from the saved `predictions.npz`,
`prepared_inputs.npz`, and `training_summary.json`, so it can also regenerate
the training-curve plots for older runs after code-only plotting updates.

### Event-BDT Outputs

For a standard multiclass event-BDT training run, the output directory contains:

- `prepared_inputs.npz`
- `predictions.npz`
- `training_summary.json`
- `feature_importance.json`
- `mass_correlation_summary.json` when `study.analysis_branches` is configured and the corresponding branches are present in `prepared_inputs.npz`
- `models/*.json`
- `plots/training_curve__mlogloss.png`
- `plots/training_curve__merror.png`
- `plots/roc_ovr.png`
- `plots/score_by_training_class__<class>.png` for each configured training class
- `plots/score_by_training_class_weighted_events__<class>.png` for each configured training class
- `plots/score_by_training_class_weighted_events__<class>_logy.png` for each configured training class
- `plots/score_by_training_class_weighted_events__tth_scores_qcd_drop_<97|98|99>_logy.png` for multiclass configs that contain `tthbb`, `tthcc`, `ttbar`, and `qcd`
- `plots/score_by_process__<class>.png` for each configured training class
- `plots/score_by_process_weighted_events__<class>.png` for each configured training class
- `plots/score_by_process_weighted_events__<class>_logy.png` for each configured training class
- `plots/feature_mass_correlation__<process>.png` for each process in the prepared/prediction payload when `study.analysis_branches` is configured
- `plots/score_vs_mass__<process>.png` for each process in the prepared/prediction payload when `study.analysis_branches` is configured
- `class_score_threshold_scan.txt` and `class_score_threshold_scan.json`
- `qcd_score_threshold_scan.txt` and `qcd_score_threshold_scan.json` when the multiclass config contains a `qcd` training class
- `plots/tth_score_study/*`, `tth_score_study_summary.txt`, and `tth_score_study_summary.json` for the current multiclass JECs studies

The threshold-scan products follow the current analysis semantics:

- `signal-like` class scans keep `score >= cut` for `bdt_score_tth`,
  `bdt_score_ttbar`, `bdt_score_tthbb`, and `bdt_score_tthcc`
- the QCD scan keeps `bdt_score_qcd <= cut`
- `qcd_score_threshold_scan` also reports an auxiliary `ttH` significance on the
  same cut:
  - `3-class`: `S = ttH(bb+cc)`, `B = ttbar + QCD`
  - `4-class`: `S = ttHbb + ttHcc`, `B = ttbar + QCD`

The `tth_score_study` output currently includes:

- pairwise `ttH`-score ROC curves such as `ttHbb vs QCD`, `ttHbb vs ttbar`,
  `ttHcc vs QCD`, and `ttHcc vs ttbar`
- QCD-cut scans in the `100 <= TargetFatJet_msoftdrop <= 150 GeV` window with
  `ttHbb/sqrt(QCD)` and `ttHcc/sqrt(QCD)`
- normalized mass-sculpting overlays for `TargetFatJet_msoftdrop`,
  `TargetFatJet_regressed_mass_generic`, and
  `TargetFatJet_regressed_mass_x2p`
- dedicated QCD-only fine-binning comparisons, including a single-panel
  overlay of the three mass definitions in the `100-150 GeV` window
- signal-only peak and width summaries for `ttHbb` and `ttHcc`

When `predict` is used, scored ROOT files keep the full original `Events` tree
and append one or more BDT score branches:

- multiclass branches follow `bdt_score_<class-name>`
- the current `3-class` config writes `bdt_score_tth`, `bdt_score_ttbar`, and
  `bdt_score_qcd`
- the current `4-class` config writes `bdt_score_tthbb`, `bdt_score_tthcc`,
  `bdt_score_ttbar`, and `bdt_score_qcd`

By default, every event is preserved in the output tree, but only events that
pass the configured `base_selection` receive a finite score. Other events keep
`NaN` in the added score branch. Use `--score-all-events` if you want to score
all events.

The event-BDT prototype is intended for local iteration and design studies. It
has a local `predict` mode for rewriting ROOT files, but it still does not have
a dedicated Condor submission helper.

## Typical Commands

### Full weighted 2024 run

```bash
LCG108
cd /eos/user/h/hanw/ttHcc/tthcc_an

python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json
```

### Restrict scores or targets temporarily

```bash
LCG108
cd /eos/user/h/hanw/ttHcc/tthcc_an

python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --targets hcc \
  --scores gpart_h2cc \
  --sig-effs 0.3 0.5 0.7 \
  --outdir outputs/boosted_higgs_tagger_study_2024_hcc
```

### Use a different candidate strategy temporarily

```bash
python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --candidate-strategy mass_window_all_jets \
  --msd-window-low 100 \
  --msd-window-high 150 \
  --outdir outputs/boosted_higgs_tagger_study_2024_msd
```

### Switch the `xbb` vs `xcc` contour region preset

```bash
python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --merge-chunks 'condor/<tag>/chunk_outputs/chunk_*.npz' \
  --plot-only \
  --xbb-vs-xcc-region-preset tight \
  --outdir outputs/boosted_higgs_tagger_xbbxcc_tight
```

## Outputs

For a standard run, the output directory contains:

- `tables/*.csv`
- `summaries/*.txt`
- `plots/*.png`
- `study_summary.json`
- `plot_input.npz`

In addition to the truth-category plots, the script now also writes process-group plots such as:

- `plots/*__background_process_score.png`
- `plots/*__background_process_wp.png`
- `plots/*__significance_scan.png`
- `plots/gpart_higgs_vs_qcd__vs__gpart_hbb_vs_hcc__contours.png`

The text summaries include:

- target signal efficiency
- score cut
- actual signal efficiency
- background efficiency
- weighted signal yield
- weighted background yield
- `S/B`
- `S/sqrt(S+B)`
- `S/sqrt(B)`
- purity

The CSV tables also contain:

- per-category efficiencies
- per-category weighted pass yields
- signed weighted pass yields
- unweighted pass counts for background categories

## Plot-Only Workflow

After a full run, the script stores:

- `<study.outdir>/plot_input.npz`

This is a slim cache containing:

- direct-run mode: jet-level arrays such as `truth_code`, `process_code`, `weight`, and requested scores
- chunked HTCondor mode: compact score histograms, truth/process yields, and sample metadata

You can redraw plots later without rereading ROOT files or recomputing the main
study outputs.

### Redraw plots only from `plot_input.npz`

```bash
LCG108
cd /eos/user/h/hanw/ttHcc/tthcc_an

python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --plot-input /path/to/<study.outdir>/plot_input.npz \
  --plot-only
```

### Redraw plots only with smaller fonts

```bash
LCG108
cd /eos/user/h/hanw/ttHcc/tthcc_an

python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --plot-input /path/to/<study.outdir>/plot_input.npz \
  --plot-only \
  --plot-title-size 11 \
  --plot-label-size 9.5 \
  --plot-tick-size 8.5 \
  --plot-legend-size 6.5 \
  --plot-cms-size 9.5
```

## Refreshing Results From Existing Chunk Outputs

If a full study was already run on HTCondor, you do not need to reread ROOT
files. You can reuse the saved chunk `.npz` files.

### Rebuild tables and summaries from chunk outputs

```bash
LCG108
cd /eos/user/h/hanw/ttHcc/tthcc_an

python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --merge-chunks 'condor/<tag>/chunk_outputs/chunk_*.npz' \
  --skip-plots
```

This is useful after changing:

- summary formatting
- table content
- significance columns
- target definitions
- score definitions

It is also the recommended refresh path after changing study logic such as:

- which truth categories belong to `signal` or `background`
- default target lists or default scores
- histogram-based plot content

It will also regenerate `plot_input.npz`.

### Redraw plots directly from chunk outputs

```bash
LCG108
cd /eos/user/h/hanw/ttHcc/tthcc_an

python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --merge-chunks 'condor/<tag>/chunk_outputs/chunk_*.npz' \
  --plot-only \
  --outdir outputs/boosted_higgs_tagger_study_2024_replot
```

This is useful if an older output directory does not yet have `plot_input.npz`.

If you changed plotting code only, `--plot-only` is usually enough. If you
changed target definitions, summary logic, or histogram payload content, prefer
`--merge-chunks` without `--plot-only` so tables and summaries are rebuilt too.

## HTCondor Workflow

For large studies on lxplus, the recommended mode is the chunked HTCondor DAG:

- chunk jobs read subsets of ROOT files and export compact histogram `.npz` payloads by default
- one merge job runs afterwards and writes the final outputs

Recommended submission:

If `study.outdir` is already set in the JSON config, the default command only
needs `--config` plus the batch-resource settings:

```bash
python scripts/submit_boosted_higgs_tagger_condor.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --files-per-chunk 60 \
  --request-memory "8 GB" \
  --merge-request-memory "24 GB" \
  --job-flavour workday \
  --merge-job-flavour tomorrow \
  --submit
```

For the current all-MC configuration, `--files-per-chunk 60` is a good starting
point if you want fewer Condor jobs than the more conservative smaller-chunk setup.

The workflow directory under `condor/<tag>/` contains:

- `chunk_configs/`: per-chunk JSON manifests
- `chunk_outputs/`: slim `.npz` payloads
- `chunks.sub`: chunk-job submit file
- `merge.sub`: merge-job submit file
- `workflow.dag`: DAG file
- `job_metadata.json`: summary of the prepared workflow, final output path, and submit settings

Because this repository and its outputs live on EOS, the submission helper uses
CERN `EosSubmit`.

If you want to prepare the workflow without immediate submission, omit
`--submit`.

After submission, the helper prints:

- `Prepared workflow directory`
- `Metadata file`
- `Final study output`

Keep the workflow directory, because it is the main place to inspect Condor
status and logs for that run.

You can also forward extra study arguments after `--`, for example:

```bash
python scripts/submit_boosted_higgs_tagger_condor.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --files-per-chunk 60 \
  --request-memory "8 GB" \
  --merge-request-memory "24 GB" \
  -- --targets hcc hbb higgs
```

### Check Condor Status

The fastest first check is:

```bash
condor_q -nobatch $USER
```

If you want an auto-refreshing view:

```bash
watch -n 30 'condor_q -nobatch $USER'
```

When jobs are submitted through `EosSubmit`, they may land on a remote schedd.
In that case, plain `condor_q` can show `0 jobs` even though the workflow is
running. Then use:

```bash
condor_q -global -nobatch $USER
```

Once you know the schedd hostname, for example `<schedd>`, query that schedd
directly:

```bash
condor_q -name <schedd> -nobatch $USER
```

To keep watching that schedd:

```bash
watch -n 30 'condor_q -name <schedd> -nobatch $USER'
```

If you want to follow one specific workflow, first note the workflow directory
printed at submission time, for example `condor/<tag>/`. Then inspect:

```bash
tail -f condor/<tag>/workflow.dag.dagman.out
```

and, when the merge stage starts:

```bash
tail -f condor/<tag>/logs/merge/job.*.log
```

Useful workflow files are:

- `condor/<tag>/job_metadata.json`
- `condor/<tag>/workflow.dag.dagman.out`
- `condor/<tag>/logs/chunks/`
- `condor/<tag>/logs/merge/`

If you need to remove a workflow, first identify the cluster ids with
`condor_q`, then run:

```bash
condor_rm -name <schedd> <cluster_id_1> <cluster_id_2>
```

## Manual Chunk Export / Merge

For debugging, manual chunk export and merge are also supported.

These examples keep `outdir` on the command line because they are usually
temporary debug runs. For regular studies, prefer setting `study.outdir` in the
config file.

### Export one chunk

```bash
LCG108
python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --scores gpart_h2cc gpart_h2bb gpart_higgs_vs_qcd \
  --chunk-payload-mode histogram \
  --export-chunk outputs/debug/chunk_0000.npz \
  --outdir outputs/debug/final
```

### Merge chunk payloads

```bash
LCG108
python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --scores gpart_h2cc gpart_h2bb gpart_higgs_vs_qcd \
  --merge-chunks 'outputs/debug/*.npz' \
  --outdir outputs/debug/final
```

## Environment

Clone the repository and set up your working environment:

```bash
git clone git@github.com:Waldeinsamkeit123/tthcc_an.git
cd tthcc_an
```

If you already use an alias like:

```bash
alias LCG108='source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh'
```

then just run `LCG108` before launching the script.

For the boosted-study workflow, the minimal fallback Python dependencies are
listed in `requirements.txt`.

For the event-BDT prototype, the recommended environment is still `LCG108`,
because it provides the extra ML dependencies that the prototype expects:

- `xgboost`
- `scikit-learn`

## Signal Trigger-Efficiency Study

The trigger-efficiency workflow studies AK8/PFHT HLT paths on signal-only
Pepper ntuples where the baseline, mass-window, and fatjet-tagging selections
have already been applied upstream.

Run it with:

```bash
LCG108
cd /eos/user/h/hanw/ttHcc/tthcc_an

python scripts/run_trigger_efficiency.py \
  --config config/trigger_efficiency/signal_2024_hlt_v1.json
```

The default config uses only:

- `ttHbb`: `/eos/user/h/hanw/ttHcc/pepper_data/2024/HLT_v1_events/TTH-Hto2B_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8`
- `ttHcc`: `/eos/user/h/hanw/ttHcc/pepper_data/2024/HLT_v1_events/TTH-Hto2C_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8`

The denominator is the weighted signal yield in each x-axis bin after the
upstream Pepper baseline. The numerator is the same yield with the given HLT
path passing. The analysis weight is:

```text
sample_norm = lumi_fb * xsec / gen_sumw
analysis_weight = sample_norm * abs(weight)
```

The workflow produces efficiency curves versus:

- `TargetFatJet_pt`
- `genhiggs_pt`

Each plot overlays exactly two curves: `ttHbb` and `ttHcc`. The configured HLT
paths are grouped into:

- `ak8_inclusive`
- `ak8_softdrop`
- `ak8_pnetbb`
- `pfht_multijet_pnet`
- `or_summary`

Outputs are written under `outputs/trigger_efficiency_signal_2024_hlt_v1/`:

- `tables/trigger_efficiencies.csv`
- `summary.json`
- `plots/eff_vs_TargetFatJet_pt__*.png`
- `plots/eff_vs_genhiggs_pt__*.png`

For a quick validation run:

```bash
python scripts/run_trigger_efficiency.py \
  --config config/trigger_efficiency/signal_2024_hlt_v1.json \
  --max-files-per-sample 1 \
  --outdir outputs/trigger_efficiency_signal_2024_hlt_v1_smoke
```
