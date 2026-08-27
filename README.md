# tthcc_an

Analysis repository for Run 3 boosted `ttHcc` / `ttHbb` studies, including
AK8 boosted Higgs tagger studies, an event-level NN output study, and a
prototype event-level BDT workflow.

## Overview

This repository is used for additional studies of the boosted `ttHcc` analysis.
The original analysis framework is based on:

- `pepper-tth`: https://gitlab.cern.ch/tthcc-run-3/pepper-tth.git

The current focus of this repository is:

- AK8 boosted Higgs tagger studies using Pepper-produced ROOT ntuples
- a 2024 boosted `1L` event-level multiclass NN output study
- a prototype 2024 `0L` event-level BDT workflow for `ttH`-enriched selection studies

## Repository Layout

- `config/`: study configuration files
  - `config/event_bdt/`: event-BDT sample and training configs
  - `config/nn_study/`: event-level NN-study configs
- `scripts/`: runnable entrypoints and submission helpers
- `src/`: core analysis code
  - `src/tthcc_an/event_bdt/`: event-BDT prototype modules
  - `src/tthcc_an/nn_study/`: event-level NN score-study modules
- `outputs/`: local study outputs
- `condor/`: prepared HTCondor workflows and chunk outputs

## Main Entry Points

The main user-facing workflows are:

- boosted Higgs tagger studies:

```bash
python scripts/run_boosted_higgs_tagger_study.py
```

- event-level BDT prototype:

```bash
python scripts/run_event_bdt.py
```

- event-level NN output study:

```bash
python scripts/run_nn_study.py \
  --config config/nn_study/nn_study_2024_1l_v1.json
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

## 2024 Event-Level NN Study

The independent NN-study subsystem is:

- entry point: `scripts/run_nn_study.py`
- implementation: `src/tthcc_an/nn_study/`
- v1 configs: `config/nn_study/nn_study_2024_{0l,1l}_v1.json`
- real-mass-trained v2 configs:
  `config/nn_study/nn_study_2024_{0l,1l}_v2_predMass.json`
- zero-mass-trained v3 configs:
  `config/nn_study/nn_study_2024_{0l,1l}_v3_zeroMass.json`
- ROOT inputs: `/eos/user/h/hanw/ttHcc/pepper_data/2024/NNeval_{0L,1L}_v1_pred_RealMass_events`
- Pepper normalization outputs: `/eos/user/h/hanw/ttHcc/pepper_data/2024/NNeval_{0L,1L}_v1_pred_RealMass`
- v2 ROOT inputs: `/eos/user/h/hanw/ttHcc/pepper_data/2024/NNeval_{0L,1L}_v2_event`
- v2 normalization outputs: `/eos/user/h/hanw/ttHcc/pepper_data/2024/NNeval_{0L,1L}_v2`
- v3 ROOT inputs: `/eos/user/h/hanw/ttHcc/pepper_data/2024/NNeval_{0L,1L}_v3_event`
- v3 normalization outputs: `/eos/user/h/hanw/ttHcc/pepper_data/2024/NNeval_{0L,1L}_v3`

The v2 model training replaces the randomized target-fatjet mass used by v1
with the real target-fatjet mass. The v2 configs keep the same NN-study truth,
sample-stitching, weighting, mass-sculpting, and score-scan definitions so the
two model versions remain directly comparable.

The v3 model sets the target-fatjet mass input to zero during training. Its
configs preserve the same analysis definitions for a direct v1/v2/v3 model
comparison. The 0L v3 config already points to the expected
`NNeval_0L_v3/gen_sumws.json`; that production is allowed to be incomplete, so
the config cannot run until the normalization file and required sample outputs
exist.

The actual 1L ntuples contain 12 softmax outputs named:

```text
score_ttHcc  score_ttHbb  score_ttZcc  score_ttZbb  score_ttZqq  score_ttLF
score_ttcj   score_ttcc   score_tt2c   score_ttbj   score_ttbb   score_tt2b
```

Both configs also declare the composite outputs `score_ttH`, `score_ttZ`,
`score_ttX`, `score_Xbb`, `score_Xcc`, and `score_ttjets` under
`auxiliary_scores`. They receive shape and expected-yield plots, but do not
extend the mutually exclusive truth classes used by confusion matrices and
pairwise ROC/AUC calculations. The 0L config additionally includes
`score_qcd` as a full classification output with truth `n_gentop == 0`.

The deployed 0L ONNX metadata lists 18 outputs, additionally including
`zbb`, `zcc`, `zqq`, `wcq`, and `wqq`. Those five branches are not present in
the current NNeval ROOT files. The 13 persisted branches, including
`score_qcd`, sum to one within float precision, so NN-study argmax diagnostics
use exactly those 13 available branches. The weighting summary records this
ONNX/ntuple difference rather than inventing unavailable class scores.

`score_ttZqq` is still present. The 12 scores sum to one within float
precision in the inspected files. Truth labeling uses the current Pepper
branches `higgs_decay`, `z_decay`, `n_gentop`, `tt_hf_flavor`, and
`tt_hf_count`. The category convention is:

- `ttHcc/ttHbb`: `higgs_decay == 4/5`
- `ttZqq/ttZcc/ttZbb`: two generated tops and `z_decay == 1..3/4/5`
- `ttLF`: two generated tops, no H/Z decay label, and `tt_hf_flavor == 0`
- `ttcj/tt2c/ttcc`: charm flavor with `tt_hf_count <= 1/== 2/> 2`
- `ttbj/tt2b/ttbb`: bottom flavor with `tt_hf_count <= 1/== 2/> 2`

The files are written after the Pepper `HasTargetFatJet` cut, following
`ReqMassWindow`. They have exactly one target fatjet in the inspected sample,
while `nJet` can be as low as 3. Therefore the shipping config applies no
additional event cut: an empty `study.selection` means all events already
stored in the ntuple. Optional selections can be supplied in config or with
`--selection`.

The nominal sample composition follows the current 1L Pepper stitching:

- Three decay-channel datasets each provide the ttHcc, ttHbb, and ttZ samples;
  stable sample labels combine each set for population-level plots.
- inclusive `TTto*` contributes only LF/charm classes.
- `TTBBto*` and `TTto*-BBDPS` contribute only bottom classes.
- data, `TTH-HtoNon2B`, single-top, W+jets, and unrelated ttV samples are not
  part of this truth-class performance study.

This avoids reusing the inclusive `TTto*` bottom component on top of the
dedicated bottom samples. ROOT files without an `Events` tree are skipped and
reported; `--max-files-per-sample` counts valid `Events` files.

Run the study with:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh

python scripts/run_nn_study.py \
  --config config/nn_study/nn_study_2024_1l_v1.json
```

Recommended smoke test:

```bash
python scripts/run_nn_study.py \
  --config config/nn_study/nn_study_2024_1l_v1.json \
  --max-files-per-sample 1 \
  --outdir /tmp/nn_study_2024_1l_v1_smoke
```

### Prepared NN-study cache

For repeated studies, prepare one event-level cache containing every configured
score plus the union of analysis branches required by the enabled studies:

```bash
python scripts/run_nn_study.py \
  --config config/nn_study/nn_study_2024_0l_v2_predMass.json \
  --prepare-cache
```

The default payload is `<study.outdir>/cache/prepared_events.npz`, with
`prepared_events.metadata.json` beside it. The payload is an uncompressed NPZ:
this uses more space than compressed NPZ but avoids repeated decompression CPU
cost for interactive plotting. The metadata records the schema, stored arrays,
sample summaries, normalization, structural definition hash, discovered input
file list hash, event counts, and cache size.

All existing modes can then run without discovering or opening ROOT files:

```bash
python scripts/run_nn_study.py \
  --config config/nn_study/nn_study_2024_0l_v2_predMass.json \
  --from-cache \
  --only qcd-score-scan

python scripts/run_nn_study.py \
  --config config/nn_study/nn_study_2024_0l_v2_predMass.json \
  --from-cache \
  --only mass-sculpting

python scripts/run_nn_study.py \
  --config config/nn_study/nn_study_2024_0l_v2_predMass.json \
  --from-cache
```

Use `--cache-path PATH` to share a cache with an overridden output directory.
`--max-files-per-sample` is part of the cache definition, so it must match
between prepare and reuse. Re-running `--prepare-cache` reuses a compatible
payload after checking the discovered input file list; add `--force` to rebuild.

Changes to the input location/pattern, tree, event or sample selection, truth or
sample definitions, weight branch, luminosity, gen-sumw/xsec files, score branch
mapping, or required analysis branches invalidate the cache. Plot styling,
binning, thresholds (including `qcd_score_scan.reference_threshold`), scan-point
choices, plot groups, and output directory do not. `--from-cache` deliberately
does not stat or glob ROOT inputs; if a ROOT file is replaced in place under the
same filename, rebuild explicitly with `--prepare-cache --force`.

An lxplus 0L v2 smoke test with one valid file per sample selected 7,553 events
and produced a 1.85 MiB cache. Measured wall times were 23.35 s to prepare,
20.10 s for a direct-ROOT QCD scan versus 4.73 s from cache, and 22.05 s for a
direct-ROOT mass study versus 5.41 s from cache. These are smoke-test timings,
not full-production benchmarks.

The workflow writes per-class and auxiliary shape/expected-yield score PNGs, row/column
normalized confusion matrices, one pairwise ROC PNG per available signal truth
class, `pairwise_auc_matrix.png`, `nn_study_summary.json`, and
`nn_study_summary.txt`. Shape plots normalize each truth class to unit area.
Yield, confusion, and ROC/AUC products use:

```text
sample_norm = lumi_fb * xsec / gen_sumw
analysis_weight = sample_norm * abs(weight)
signed_bookkeeping_weight = sample_norm * weight
```

### Training-weight diagnostics

Both configs enable an independent diagnostic mode:

```bash
python scripts/run_nn_study.py \
  --config config/nn_study/nn_study_2024_1l_v1.json \
  --only weighting-diagnostics

python scripts/run_nn_study.py \
  --config config/nn_study/nn_study_2024_0l_v1.json \
  --only weighting-diagnostics
```

This mode reads only the persisted multiclass scores, truth and stitching
branches, raw event weight, and the configured training reweight variables
`MET_pt` and `MET_phi`. It writes:

- `confusion_truth__analysis_weighted.png` for
  `P(predicted class | truth class)` using the physical analysis weight
- `confusion_truth__unweighted.png` using one entry per event
- `confusion_truth__class_balanced.png`, where each truth class is rescaled to
  unit total analysis weight
- `confusion_pred__analysis_weighted.png` for the physical predicted-category
  composition `P(truth class | predicted class)`
- `pairwise_auc_matrix__{analysis_weighted,unweighted}.png`
- `summaries/weighting_diagnostics.{json,txt}` and
  `summaries/class_weight_comparison.{json,txt}`

The class-balanced truth matrix is expected to equal the analysis-weighted
truth matrix: row normalization cancels a class-wide scale. It is retained as
an explicit definition check and must not be called training-weighted.

The relevant standard implementation was traced through Weaver commit
`a57c362bb42891e7d7dc997fd802b4bd15d978c8`, immediately before model
deployment. The model-specific NanoTTH training checkout, command, network
config, and auto-generated YAML were not found locally. In the standard Weaver
path, class/bin sampling factors change DataLoader event multiplicities by
random acceptance and repetition. The sampling weight is not passed to the
default cross-entropy loss; accepted copies enter an unweighted loss that is
averaged over the batch. A model-specific custom loss cannot be excluded
without the missing network config.

For the configured single wide `MET_pt`/`MET_phi` bin, the deterministic
sampling score reduces to

```text
p_i = [class_weight[c] * abs(weight_i) / H[c]]
      / P99_training(class_weight[class(i)] * abs(weight_i) / H[class(i)])
```

where `H[c]` is the training-population sum of `abs(weight)` in class `c`.
With the default sampler, copies are accepted with probability
`min(p_i, 1)` and the per-fetch repeat count is stochastic and capped at 10.
Thus this is class-level flattening plus within-class `abs(weight)` sampling;
the expected class totals are proportional to the configured class weights
before clipping and fetch effects.

Exact training-effective contributions cannot be reconstructed from the
current NNeval ROOT files. The model-specific Weaver `.auto.yaml` reweight
histograms, exact training population and percentile, DataLoader fetch
partition, random seeds, and realized repeated indices are not available in
the model directory or ntuples. Therefore no file named `training_weighted`
is produced, and no class-balanced approximation is silently substituted.
The existing expected-yield, QCD scan, mass-sculpting, composition, and
physics ROC/AUC products continue to use `sample_norm * abs(weight)`.

### Score-cut significance scans

Both channels enable config-driven significance scans for the auxiliary
scores `score_ttX` and `score_ttLF`. The metric is strictly

```text
Z = S / sqrt(S + B)
```

For each curve, `S` is exactly one NN-study truth category (`ttHcc` or
`ttHbb`) and `B` is every other selected MC event. Thus ttHbb is background
for the ttHcc curve, ttHcc is background for the ttHbb curve, and all ttZ,
ttbar, QCD, and W/Z+jets contributions are also background. The weight for
both S and B is `sample_norm * abs(weight)`. No additional mass window is
applied beyond the current upstream/configured NN-study event population.

The configured scans are independent of the sparse mass-sculpting cuts:

- `score_ttX > cut`: 201 linear points over exactly `0 <= cut <= 1`
- `score_ttLF < cut`: 161 linear points over exactly `0 <= cut <= 0.8`
- 0L `score_qcd < cut`: the existing 121-point log scan over `1e-6` to `1`,
  augmented with the seven exact candidate thresholds

The no-score-cut baseline uses the full selected MC population before any
score requirement. In particular, `score_ttLF < 0.8` is not treated as the
uncut baseline. Scan maxima are diagnostics only and are never written back as
final working points.

Run only the ttX/ttLF scans with:

```bash
python scripts/run_nn_study.py \
  --config config/nn_study/nn_study_2024_1l_v1.json \
  --only score-significance
```

Outputs are
`plots/score_significance/score_tt{X,LF}__significance_s_over_sqrt_s_plus_b.png`
and `summaries/score_significance.{json,txt}`. The summaries store every scan
point's S, B, Z, relative significance, baseline, and numerical scan maximum.

For each signal/background pair, the discriminator is
`score_signal / (score_signal + score_background)`; a zero denominator maps
to 0.5. The reported AUC is the standard area of signal efficiency (TPR) versus
background efficiency (FPR), even though the displayed ROC axes are
`x = signal efficiency` and `y = background efficiency`.

The loader also retains `TargetFatJet_mass`, `TargetFatJet_msoftdrop`,
`TargetFatJet_regressed_mass_generic`, `TargetFatJet_regressed_mass_x2p`,
and `TargetFatJet_random_mass` during a full run.

The 1L config enables a config-driven mass-sculpting study of
`TargetFatJet_regressed_mass_generic`. It compares the uncut shape with:

- `score_ttX > 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9`
- `score_ttLF < 0.01, 0.02, 0.05, 0.1`

The configs produce each scan for five populations: all selected MC, the
label-combined ttHcc sample, the label-combined ttHbb sample, the label-combined
ttZ sample, and all remaining configured background samples after excluding
those three labels. The three physics populations are kept separate. Curves use
`sample_norm * abs(weight)` and are independently
normalized to unit density over the plotted range. The range uses the weighted
`0.5%` and `99.5%` mass quantiles plus 4% padding, with 45 equal-width bins.
Each plot includes an uncut reference and a lower panel showing the retained
fraction per mass bin. Machine-readable and text diagnostics are written to
`summaries/mass_sculpting_summary.{json,txt}`.

Run only this study, without loading unrelated score/analysis branches or
computing score plots, confusion matrices, ROC curves, or AUCs:

```bash
python scripts/run_nn_study.py \
  --config config/nn_study/nn_study_2024_1l_v1.json \
  --only mass-sculpting
```

The 0L config runs the same `score_ttX` and `score_ttLF` scans and additionally
runs `score_qcd < 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1`.

The `mass_sculpting.populations` entries support independent `samples`,
`sample_labels`, and `truth_categories` filters, plus corresponding exclusion
fields for sample-level complements.
Non-default populations are appended to output
filenames, for example
`mass_sculpting__TargetFatJet_regressed_mass_generic__score_ttX__ttHcc_sample.png`.
Together with `variables` and `scans`, this allows later category-specific or
0L extensions without adding a new script. Direct mode reads the selected ROOT
events; after `--prepare-cache`, use `--from-cache --only mass-sculpting` to
avoid that repeated ROOT scan.

Both configs set `study.sample_file_pattern` to
`{input_location}/{name}/*.root`; this global template replaces per-sample
`files` entries. For a future prediction production with the same directory
layout, update only `study.input_location`, `normalization.gen_sumw_file`, and
the desired `study.outdir`. Sample rows only define dataset names, labels, and
optional stitching selections.

The 0L config also enables a dedicated QCD-score working-point scan using the
actual `score_qcd` branch and the cut direction `score_qcd < cut`. It applies
no extra `nJet` requirement: the population is the same configured MC sample
set after the existing per-sample stitching and empty NN-study selection. The
named candidate thresholds are `0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05,
0.1`, supplemented by a fine log-spaced scan from `1e-6` to `1`.

The main groups are `ttHcc`, QCD truth (`n_gentop == 0`), and `tt+X = ttLF +
ttcj + ttcc + tt2c + ttbj + ttbb + tt2b`. Consequently, the QCD truth group
also contains selected `Wto2Q`/`Zto2Q` sample events with `n_gentop == 0`.
Expected yields and efficiencies use `sample_norm * abs(weight)`. The current
scan is MC-only; data files are not mixed into normalized expected yields.

Run only the QCD-score scan, without regular score, confusion, ROC/AUC, or
mass-sculpting products:

```bash
python scripts/run_nn_study.py \
  --config config/nn_study/nn_study_2024_0l_v1.json \
  --only qcd-score-scan
```

It writes QCD-scan PNG diagnostics under `plots/qcd_score_scan/` and the complete
fine scan plus candidate table to `summaries/qcd_score_scan.{json,txt}`. The
diagnostics include weighted score/yield curves, ttHcc and QCD efficiencies,
QCD rejection, QCD/tt+X, a ROC-like working-point curve, and relative
`S/sqrt(B)` for `B=QCD` and `B=QCD+tt+X`. These significance values are
diagnostics only and do not select a working point automatically.

The same QCD workflow additionally writes
`plots/qcd_score_scan/qcd_cut_scan__significance_s_over_sqrt_s_plus_b.png`.
It contains separate ttHcc and ttHbb `S/sqrt(S+B)` curves using the
truth-category/everything-else definitions above, marks the seven configured
candidate cuts, and annotates each numerical scan maximum. The existing
QCD-scan JSON/text summary records the full curves, candidate-point S/B/Z, true
uncut baselines, and best-point metadata.

The additional `qcd_score_distribution.png` compares independently normalized
weighted shapes for `ttHcc`, `tt+light`, `tt+>=1c`, `tt+>=1b`, and QCD using
logarithmic bins in the linear `score_qcd` value. Its vertical line and shaded
high-score rejected region are controlled only by
`qcd_score_scan.reference_threshold`; this is a visualization reference, not
a chosen final cut. `qcd_score_distribution__yield.png` provides the same
grouping without unit normalization.

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
