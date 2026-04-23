# tthcc_an

Analysis repository for the Run 3 boosted `ttHcc` / `ttHbb` study.

## Overview

This repository is used for additional studies of the boosted `ttHcc` analysis.
The original analysis framework is based on:

- `pepper-tth`: https://gitlab.cern.ch/tthcc-run-3/pepper-tth.git

The current focus of this repository is AK8 boosted Higgs tagger studies using
Pepper-produced ROOT ntuples.

## Repository Layout

- `config/`: study configuration files
- `scripts/`: runnable entrypoints and submission helpers
- `src/`: core analysis code
- `outputs/`: local study outputs
- `condor/`: prepared HTCondor workflows and chunk outputs

## Main Study Script

The main script is:

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

- supports `H->cc` and `H->bb` targets
- reads multiple `ttHcc`, `ttHbb`, and `ttbar` samples from JSON config
- only reads the `FatJet` branches needed for the requested scores
- supports `all_jets`, `leading_pt`, and `mass_window_leading_pt`
- computes weighted working points at chosen signal efficiencies
- writes tables, text summaries, JSON summaries, ROC curves, and score plots
- stores a reusable `plot_input.npz` cache for later plot-only redraws
- uses histogram chunk payloads by default for low-memory HTCondor merges
- uses `mplhep` with CMS style for figures

Currently supported scores:

- `gpart_h2cc = Xcc / (Xcc + QCD)`
- `gpart_h2bb = Xbb / (Xbb + QCD)`
- `pnet_hcc = particleNetWithMass_HccvsQCD`
- `pnet_xcc = particleNet_XccVsQCD`
- `pnetlegacy_xcc = particleNetLegacy_Xcc`

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

- signal = `hcc_pure + hcc_contaminated`
- background = `hcc_partial + hbb_* + top + other`

For `target = hbb`:

- signal = `hbb_pure + hbb_contaminated`
- background = `hbb_partial + hcc_* + top + other`

## Config Files

Available configs in this repository:

- [config/samples.example.json](/eos/user/h/hanw/ttHcc/tthcc_an/config/samples.example.json)
- [config/samples_2024_add_nonttbarmatch_allmc.json](/eos/user/h/hanw/ttHcc/tthcc_an/config/samples_2024_add_nonttbarmatch_allmc.json)

The 2024 config already contains:

- the sample list
- all MC datasets listed in the corresponding `gen_sumw` JSON
- process-group labels for `ttbar`, `tt+bb`, `ttV`, `tt+ll`, `single top`, `W+jets`, `Z+jets`, `QCD`, and `ttH*`
- the 2024 luminosity
- the `gen_sumw` JSON path
- the cross section JSON path

## Typical Commands

### Full weighted 2024 run

```bash
LCG108
cd /eos/user/h/hanw/ttHcc/tthcc_an

python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --outdir outputs/boosted_higgs_tagger_study_2024
```

### Restrict scores or targets

```bash
LCG108
cd /eos/user/h/hanw/ttHcc/tthcc_an

python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --targets hcc \
  --scores gpart_h2cc pnet_hcc \
  --sig-effs 0.3 0.5 0.7 \
  --outdir outputs/boosted_higgs_tagger_study_2024_hcc
```

### Use a different candidate strategy

```bash
python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --candidate-strategy mass_window_leading_pt \
  --msd-window-low 100 \
  --msd-window-high 150 \
  --outdir outputs/boosted_higgs_tagger_study_2024_msd
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

- `plot_input.npz`

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
  --plot-input outputs/boosted_higgs_tagger_study_2024/plot_input.npz \
  --plot-only \
  --outdir outputs/boosted_higgs_tagger_study_2024
```

### Redraw plots only with smaller fonts

```bash
LCG108
cd /eos/user/h/hanw/ttHcc/tthcc_an

python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --plot-input outputs/boosted_higgs_tagger_study_2024/plot_input.npz \
  --plot-only \
  --outdir outputs/boosted_higgs_tagger_study_2024 \
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
  --outdir outputs/boosted_higgs_tagger_study_2024 \
  --skip-plots
```

This is useful after changing:

- summary formatting
- table content
- significance columns

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

## HTCondor Workflow

For large studies on lxplus, the recommended mode is the chunked HTCondor DAG:

- chunk jobs read subsets of ROOT files and export compact histogram `.npz` payloads by default
- one merge job runs afterwards and writes the final outputs

Recommended submission:

```bash
python scripts/submit_boosted_higgs_tagger_condor.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --outdir outputs/boosted_higgs_tagger_study_2024 \
  --files-per-chunk 60 \
  --request-memory "8 GB" \
  --merge-request-memory "16 GB" \
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

Because this repository and its outputs live on EOS, the submission helper uses
CERN `EosSubmit`.

If you want to prepare the workflow without immediate submission, omit
`--submit`.

You can also forward extra study arguments after `--`, for example:

```bash
python scripts/submit_boosted_higgs_tagger_condor.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --outdir outputs/boosted_higgs_tagger_study_2024 \
  --files-per-chunk 60 \
  --request-memory "8 GB" \
  --merge-request-memory "24 GB" \
  -- --scores gpart_h2cc gpart_h2bb
```

## Manual Chunk Export / Merge

For debugging, manual chunk export and merge are also supported.

### Export one chunk

```bash
LCG108
python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --scores gpart_h2cc gpart_h2bb \
  --chunk-payload-mode histogram \
  --export-chunk outputs/debug/chunk_0000.npz \
  --outdir outputs/debug/final
```

### Merge chunk payloads

```bash
LCG108
python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --scores gpart_h2cc gpart_h2bb \
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

The minimal Python dependencies are listed in `requirements.txt` as a fallback
reference.
