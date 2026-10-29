from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import awkward as ak
import numpy as np
import uproot

from tthcc_an.nn_study.cache import (
    cache_analysis_branches, cache_score_names, load_nn_cache, resolve_cache_path,
    validate_nn_cache, write_nn_cache,
)
from tthcc_an.nn_study.cli import _build_parser, run
from tthcc_an.nn_study.config import NnSample, load_nn_study_config
from tthcc_an.nn_study.dataset import load_nn_dataset
from tthcc_an.nn_study.regressed_mass_fine_binning import (
    _data_cache_path, run_regressed_mass_fine_binning,
)

REPO = Path(__file__).resolve().parents[1]
MASS = "TargetFatJet_regressed_mass_generic"
MODULE = "tthcc_an.nn_study.regressed_mass_fine_binning"


def config_path(channel: str, version: int = 5) -> Path:
    return REPO / f"config/nn_study/nn_study_2024_{channel}_v{version}.json"


def load_config(channel: str, version: int = 5):
    return load_nn_study_config(config_path(channel, version), REPO,
                                discover_sample_files=False)


class V5StudyTest(unittest.TestCase):
    def test_configs_preserve_native_studies_and_separate_versions(self) -> None:
        for channel in ("0l", "1l"):
            with self.subTest(channel=channel), patch(
                "tthcc_an.nn_study.config.expand_file_patterns",
                side_effect=AssertionError("Config-only load must not discover ROOT"),
            ):
                config = load_config(channel)
                v4 = load_config(channel, 4)
                raw = json.loads(config_path(channel).read_text())
                self.assertEqual(config.samples, v4.samples)
                self.assertEqual(config.truth_categories, v4.truth_categories)
                self.assertEqual(config.prediction_truth_categories, v4.prediction_truth_categories)
                self.assertEqual(config.diagnostic_truth_categories, v4.diagnostic_truth_categories)
                self.assertEqual(config.score_names, ["ttHZ", "ttjets"] + (["qcd"] if channel == "0l" else []))
                self.assertEqual(config.all_scores, v4.scores)
                self.assertEqual(config.analysis_discriminants, [])
                self.assertFalse(config.score_significance_2d_hierarchical.enabled)
                self.assertFalse(config.tthz_qcd_discriminant_enabled)
                self.assertTrue(v4.tthz_qcd_discriminant_enabled)
                self.assertEqual(config.sample_file_pattern, "{input_location}/{name}/*.root")
                base = f"/eos/user/h/hanw/ttHcc/pepper_data/2024/NNeval/{channel.upper()}/{channel.upper()}_v5"
                self.assertEqual(config.input_location, base + "_events")
                self.assertEqual(config.gen_sumw_file, base + "/gen_sumws.json")
                self.assertEqual(config.outdir, REPO / f"outputs/nn_study_2024_{channel}_v5")
                self.assertEqual(resolve_cache_path(config, None), config.outdir / "cache/prepared_events.npz")
                self.assertNotEqual(resolve_cache_path(config, None), resolve_cache_path(v4, None))
                self.assertNotEqual(_data_cache_path(config), _data_cache_path(v4))
                self.assertTrue(raw["study"]["require_complete_samples"])
                self.assertFalse(any("D_ttHZ" in branch for branch in config.requested_branches))
                self.assertEqual(cache_score_names(config), config.score_names)
                self.assertEqual(config.score_significance.scans, [s for s in v4.score_significance.scans if s.score_name != "D_ttHZ"])
                self.assertEqual(config.mass_sculpting.scans, [s for s in v4.mass_sculpting.scans if s.score_name != "D_ttHZ"])
                self.assertEqual(config.mass_sculpting.populations, v4.mass_sculpting.populations)
                self.assertEqual(config.mass_sculpting.summary_scans, v4.mass_sculpting.summary_scans)
                self.assertEqual(config.score_significance_2d, v4.score_significance_2d)
                self.assertEqual(config.qcd_score_scan, v4.qcd_score_scan)
                self.assertTrue(config.analysis_regions.enabled)
                ar = {cut.score_name: cut.threshold for cut in config.analysis_regions.ar_cuts}
                self.assertEqual(ar["ttHZ"], 0.70)
                if channel == "0l":
                    self.assertEqual(ar["qcd"], 0.10)
                    self.assertEqual(config.regressed_mass_fine_binning["qcd_cut"], 0.01)
                    self.assertEqual(config.regressed_mass_fine_binning, v4.regressed_mass_fine_binning)
                    self.assertEqual(config.analysis_regions.conditional_qcd_veto, v4.analysis_regions.conditional_qcd_veto)

    def test_incomplete_inputs_fail_before_root_loading(self) -> None:
        for channel in ("0l", "1l"):
            with patch("pathlib.Path.is_file", return_value=False), patch(
                "tthcc_an.nn_study.config.expand_file_patterns"
            ) as discovery:
                with self.assertRaisesRegex(FileNotFoundError, "Production incomplete: gen_sumw"):
                    load_nn_study_config(config_path(channel), REPO)
                discovery.assert_not_called()
            with patch("pathlib.Path.is_file", return_value=True), patch(
                "tthcc_an.nn_study.config.expand_file_patterns", return_value=[]
            ):
                with self.assertRaisesRegex(FileNotFoundError, "no ROOT files for configured samples"):
                    load_nn_study_config(config_path(channel), REPO)

    def test_explicit_derived_study_is_disabled_before_cache_loading(self) -> None:
        for channel in ("0l", "1l"):
            args = _build_parser().parse_args([
                "--config", str(config_path(channel)), "--from-cache",
                "--only", "tthz-qcd-discriminant",
            ])
            with patch("tthcc_an.nn_study.cli.load_nn_cache") as reader:
                with self.assertRaisesRegex(ValueError, "disabled in this config"):
                    run(args)
                reader.assert_not_called()

    def _synthetic(self, directory: Path, channel: str, with_derived: bool = False):
        arrays = {
            "score_ttHZ": np.array([0.8, 0.6, 0.2, 0.1]),
            "weight": np.array([1.0, -2.0, 3.0, 4.0]),
            "higgs_decay": np.array([4, 5, 0, 0]),
            "z_decay": np.array([0, 0, 0, 0 if channel == "0l" else 4]),
            "w_decay": np.zeros(4, dtype=int),
            "n_gentop": np.array([2, 2, 2, 0 if channel == "0l" else 2]),
            "tt_hf_flavor": np.zeros(4, dtype=int),
            "tt_hf_count": np.zeros(4, dtype=int),
            MASS: ak.Array([[90.0], [120.0], [155.0], [110.0]]),
        }
        arrays["score_ttjets"] = 1 - arrays["score_ttHZ"]
        if channel == "0l":
            arrays["score_qcd"] = np.full(4, 0.005)
            arrays["score_ttjets"] -= arrays["score_qcd"]
        if with_derived:
            arrays["D_ttHZ"] = np.full(4, 0.123)
        root = directory / "events.root"
        with uproot.recreate(root) as output:
            output["Events"] = arrays
        for filename in ("gen_sumws.json", "xsecs.json"):
            (directory / filename).write_text(json.dumps({"synthetic": 1.0}))
        config = replace(
            load_config(channel), outdir=directory,
            analysis_branches=[MASS],
            samples=[NnSample("synthetic", "synthetic", "synthetic", [str(root)], "")],
            gen_sumw_file=str(directory / "gen_sumws.json"),
            xsec_file=str(directory / "xsecs.json"), lumi_fb=1.0,
        )
        return config, load_nn_dataset(config)

    def test_native_only_root_cache_and_input_separation(self) -> None:
        for channel in ("0l", "1l"):
            for with_derived in (False, True):
                with self.subTest(channel=channel, with_derived=with_derived), TemporaryDirectory() as temporary:
                    config, dataset = self._synthetic(Path(temporary), channel, with_derived)
                    self.assertEqual(set(dataset.scores), set(config.score_names))
                    np.testing.assert_allclose(sum(dataset.scores.values()), 1.0)
                    np.testing.assert_allclose(dataset.analysis_weight, [1, 2, 3, 4])
                    cache = resolve_cache_path(config, None)
                    metadata = write_nn_cache(config=config, dataset=dataset, cache_path=cache)
                    self.assertEqual(set(metadata["stored_arrays"]["scores"]), set(config.score_names))
                    with patch("uproot.open", side_effect=AssertionError("Cache must not open ROOT")):
                        loaded, _ = load_nn_cache(config=config, cache_path=cache,
                                                  score_names=cache_score_names(config),
                                                  analysis_branches=cache_analysis_branches(config))
                    np.testing.assert_array_equal(loaded.analysis_weight, dataset.analysis_weight)
                    with self.assertRaisesRegex(ValueError, "input_location"):
                        validate_nn_cache(config=replace(config, input_location=load_config(channel, 4).input_location),
                                          cache_path=cache, check_input_files=False)

    def test_mcstat_without_data_or_trigger_files(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, dataset = self._synthetic(root, "0l")
            config = replace(config, regressed_mass_fine_binning={
                **config.regressed_mass_fine_binning, "data_trigger_config": "/missing/trigger.json",
            })
            with patch(MODULE + ".load_data_cache", side_effect=AssertionError("No data in MC-only mode")), patch(
                MODULE + ".load_json_maybe_with_comments", side_effect=AssertionError("No trigger config in MC-only mode")
            ), patch(MODULE + "._plot_mass_stack") as stacks, patch(
                MODULE + "._plot_mcstat_heatmap"
            ) as heatmaps, patch(MODULE + "._plot_data_mc") as data_plot:
                plots, summary = run_regressed_mass_fine_binning(
                    config=config, dataset=dataset, plot_dir=root / "plots",
                    summary_dir=root / "summaries", plot_suffix=".png", include_data=False,
                )
            self.assertEqual(len(plots), 7)
            self.assertEqual(stacks.call_count, 5)
            self.assertEqual(heatmaps.call_count, 2)
            data_plot.assert_not_called()
            self.assertIsNone(summary["data_source"])
            self.assertIsNone(summary["data_cache"])
            self.assertNotIn("N_data", json.dumps(summary))
            self.assertAlmostEqual(summary["thresholds"]["0.50"]["total_mc"]["full"]["weighted_yield"], 3.0)
            self.assertAlmostEqual(summary["thresholds"]["0.50"]["total_mc"]["full"]["sumw2"], 5.0)
            self.assertTrue((root / "summaries/regressed_mass_mcstat__mc_only.json").is_file())
            self.assertFalse((root / "summaries/regressed_mass_mcstat.json").exists())
            self.assertIn("0L V5", (root / "summaries/regressed_mass_mcstat__mc_only.txt").read_text())
            with self.assertRaises(FileNotFoundError):
                run_regressed_mass_fine_binning(
                    config=config, dataset=dataset, plot_dir=root / "plots",
                    summary_dir=root / "summaries", plot_suffix=".png",
                )


if __name__ == "__main__":
    unittest.main()
