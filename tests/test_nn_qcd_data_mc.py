from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import awkward as ak
import numpy as np
import uproot

from tthcc_an.nn_study.cli import _build_parser, run
from tthcc_an.nn_study.config import NnSample, load_nn_study_config
from tthcc_an.nn_study.qcd_data_mc import (
    MASS, _data_histograms, _histogram, _mc_weights, _output_paths, _ratios,
    _read_data_file, _restrict_private_directory, _selection_masks, _settings, run_qcd_data_mc,
)

REPO = Path(__file__).resolve().parents[1]
MODULE = "tthcc_an.nn_study.qcd_data_mc"
CONFIG = REPO / "config/nn_study/nn_study_2024_0l_v5.json"


class QcdDataMcTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = load_nn_study_config(CONFIG, REPO, discover_sample_files=False)
        trigger = self.root / "trigger.json"
        trigger.write_text(json.dumps({
            "dataset_trigger_order": ["ParkingHH", "JetMET"],
            "dataset_trigger_map": {"ParkingHH": ["HLT_Parking"], "JetMET": ["HLT_HT"]},
        }))
        normalization = self.root / "normalization.json"
        normalization.write_text('/* "luminosity": 999, disabled */\n' + json.dumps({
            "luminosity": 109.95, "mc_lumifactors": False, "quoted": "keep /* literal */ text",
        }))
        names = ["QCD-4Jets", "Wto2Q", "Zto2Q", "TTto2L"]
        xsecs, sums = self.root / "xsecs.json", self.root / "sumw.json"
        xsecs.write_text("/* cross sections in fb */\n" + json.dumps({name: "2*(3-1)/2" for name in names}))
        sums.write_text(json.dumps({name: {"gen_sumw": 20.0} for name in names}))
        self.options = {**self.config.qcd_data_mc, "data_workers": 2,
                        "data_trigger_config": str(trigger), "normalization_config": str(normalization),
                        "xsec_file": str(xsecs)}
        self.config = replace(self.config, outdir=self.root / "output", input_location=str(self.root / "events"),
                              gen_sumw_file=str(sums), qcd_data_mc=self.options,
                              samples=[NnSample(name, name, name, [], "tt_hf_flavor != 5" if name == "TTto2L" else "")
                                       for name in names])
        self.edges = np.linspace(-6, 0, 61)
        self.mc = SimpleNamespace(
            raw_weight=np.array([2., -1., 3., 4., 5.]), sample_index=np.array([0, 0, 1, 2, 3]),
            sample_summaries=[{"name": name} for name in names],
            scores={"qcd": np.array([0.9, 0.01, 0.8, 0.05, 1e-5]), "ttHZ": np.array([.7, .9, .9, .2, .69])},
            analysis_columns={MASS: np.array([90., 120., 150., 155., 160.])},
            truth_index=np.zeros(5, dtype=int),  # Even W/Z assigned qcd truth stay non-QCD.
        )
        self.write_data("ParkingHH", [1, 2, 3, 4, 5, 1], [90, 100, 125, 150, 160, 90],
                        [1e-5, .002, .9, .8, .999, 1e-5], [.7, .9, .9, .2, .69, .7])
        self.write_data("JetMET", [10, 11], [80, 155], [0, np.nan], [.1, .8])

    def write_data(self, stream, ids, masses, scores, tthz, *, downstream=False):
        path = Path(self.config.input_location) / stream / "test.root"
        path.parent.mkdir(parents=True, exist_ok=True)
        with uproot.recreate(path) as output:
            output["dsname"] = stream
            output["Cutnames"] = str(["Trigger", "HasTargetFatJet", "NNEvaluation"] + (["QCDVeto"] if downstream else []))
            output["Events"] = {
                "event": np.asarray(ids, dtype=np.uint64), "run": np.ones(len(ids), dtype=np.uint32),
                "luminosityBlock": np.ones(len(ids), dtype=np.uint32), "weight": np.ones(len(ids)),
                "score_qcd": np.asarray(scores, dtype=float), "score_ttHZ": np.asarray(tthz, dtype=float),
                MASS: ak.Array([[float(mass)] for mass in masses]),
            }
        return path

    def test_no_qcd_cut_and_symmetric_mass_sidebands(self):
        mass = np.array([90, 100, 125, 150, 160, np.nan])
        tthz = np.array([.7, .9, .9, .2, .69, .8])
        side = _selection_masks(tthz, mass, self.options, "sidebands")
        np.testing.assert_array_equal(side["baseline"], [True, False, False, False, True, False])
        np.testing.assert_array_equal(side["ttHZ_ge_0p70"], [True, False, False, False, False, False])
        self.assertTrue(_selection_masks(tthz, mass, self.options, "unblinded")["baseline"].all())
        path = Path(self.config.input_location) / "ParkingHH/test.root"
        arrays = _read_data_file((path, "ParkingHH", "Events", self.options, "sidebands"))
        self.assertEqual(set(arrays[MASS]), {90, 160})
        self.assertIn(.999, arrays["score_qcd"])

    def test_score_flows_and_signed_sumw2(self):
        scores = np.array([0., -1., np.nan, np.inf, 1e-7, 1e-6, .9, 1., 1.1])
        weights = np.array([1., 1., 1., 1., 1., 2., -3., 4., 1.])
        hist = _histogram(scores, weights, self.edges)
        self.assertEqual(sum(hist["sumw"]), 3)
        self.assertEqual(sum(hist["sumw2"]), 29)
        self.assertEqual({key: value["events"] for key, value in hist["score_quality"].items()},
                         {"zero": 1, "negative": 1, "nonfinite": 2, "underflow": 1, "overflow": 1, "in_range": 3})

    def test_physical_weights_and_sample_origin_grouping(self):
        weights, qcd, vjets, samples, lumi = _mc_weights(self.config, self.mc, self.options)
        self.assertEqual(lumi, 109.95)
        np.testing.assert_allclose(weights, self.mc.raw_weight * 10.995)
        np.testing.assert_array_equal(qcd, [True, True, False, False, False])
        np.testing.assert_array_equal(vjets, [False, False, True, True, False])
        self.assertEqual(samples[-1]["stitching"], "tt_hf_flavor != 5")
        self.assertLess(weights[1], 0)

    def test_invalid_ratios_and_mc_statistics(self):
        ratio, error, relative, valid = _ratios(np.array([4., 1., 1., 0.]),
                                               np.array([2., 0., -1., 2.]), np.array([9., 0., 4., 1.]))
        np.testing.assert_array_equal(valid, [True, False, False, True])
        self.assertEqual(ratio[0], 2)
        self.assertEqual(error[0], 1)
        self.assertEqual(relative[0], 1.5)
        self.assertTrue(np.isnan(ratio[1:3]).all())
        self.assertEqual(ratio[3], 0)

    def test_blinding_optin_and_selection_guards(self):
        with self.assertRaisesRegex(ValueError, "requires --allow-unblinded"):
            _settings(self.config, "unblinded", False)
        _settings(self.config, "unblinded", True)
        with self.assertRaisesRegex(ValueError, "without selection"):
            _settings(replace(self.config, selection="score_qcd < .1"), "sidebands", False)
        with self.assertRaisesRegex(ValueError, "complete baseline"):
            _settings(replace(self.config, max_files_per_sample=1), "sidebands", False)
        with self.assertRaisesRegex(ValueError, "public/web"):
            _output_paths(replace(self.config, outdir=self.root / "public"), "unblinded")
        path = self.write_data("JetMET", [20], [90], [.9], [.8], downstream=True)
        with self.assertRaisesRegex(ValueError, "baseline process-stage"):
            _read_data_file((path, "JetMET", "Events", self.options, "sidebands"))

    def test_cache_and_overlap_audit(self):
        cache = self.root / "cache.json"
        result = _data_histograms(self.config, self.options, self.edges, "sidebands", cache, False)
        self.assertEqual(sum(result["histograms"]["baseline"]["sumw"]), 2)
        self.assertEqual(result["overlap_checks"]["within_stream_duplicates_removed"]["ParkingHH"], 1)
        self.assertEqual(result["definition"]["trigger"]["trigger_selection"]["JetMET"]["veto_any"], ["HLT_Parking"])
        with patch(MODULE + ".uproot.open", side_effect=AssertionError("Histogram cache must not open ROOT")):
            self.assertEqual(result, _data_histograms(self.config, self.options, self.edges, "sidebands", cache, False))
        with self.assertRaisesRegex(ValueError, "cache inputs/settings changed"):
            _data_histograms(self.config, self.options, self.edges, "unblinded", cache, False)
        self.write_data("JetMET", [1], [90], [.9], [.8])
        with self.assertRaisesRegex(ValueError, "Cross-stream duplicate"):
            _data_histograms(self.config, self.options, self.edges, "sidebands", cache, True)

    def test_four_outputs_and_private_summary_separation(self):
        payloads = {}
        with patch(MODULE + ".load_nn_cache", return_value=(self.mc, {"definition_hash": "synthetic"})):
            for mode in ("sidebands", "unblinded"):
                result = run_qcd_data_mc(config=self.config, mc_cache_path=self.root / "mc.npz",
                                        blinding=mode, allow_unblinded=mode == "unblinded")
                self.assertEqual(len(result["plots"]), 2)
                for image in result["plots"]:
                    self.assertGreater(Path(image).stat().st_size, 1000)
                for filename in result["summaries"]:
                    if not filename.endswith(".json"):
                        continue
                    summary = json.loads(Path(filename).read_text())
                    payloads[mode, summary["study"]] = summary
                    np.testing.assert_allclose(summary["allMC"]["sumw"],
                                               np.array(summary["nonQCD_MC"]["sumw"]) + summary["QCD_multijet"]["sumw"])
                    np.testing.assert_allclose(summary["allMC"]["sumw2"],
                                               np.array(summary["nonQCD_MC"]["sumw2"]) + summary["QCD_multijet"]["sumw2"])
                    self.assertIsNone(summary["qcd_score_cut"])
                    self.assertFalse(summary["references_are_cuts"])
                    self.assertEqual("private_unblinded" in filename, mode == "unblinded")
        side = payloads["sidebands", "baseline"]
        full = payloads["unblinded", "baseline"]
        self.assertEqual(sum(side["data_counts"]), 2)
        self.assertEqual(sum(full["data_counts"]), 5)
        self.assertAlmostEqual(sum(side["QCD_multijet"]["sumw"]), 2 * 10.995)
        self.assertNotEqual(side["data_cache"], full["data_cache"])
        private = self.config.outdir / "plots/qcd_data_mc/private_unblinded"
        self.assertEqual(private.stat().st_mode & 0o777, 0o700)

    def test_eos_private_acl_denies_inherited_users(self):
        import os
        owner = os.getuid()
        directory = self.root / "private"
        inherited = "u:42:rx"
        effective = f"u:{owner}:rwx,u:42:!r!x!w,u:42:rx"
        with patch("pathlib.Path.resolve", return_value=Path("/eos/home-h/hanw/private")), patch(
            MODULE + ".subprocess.run",
            side_effect=[SimpleNamespace(stdout=inherited), SimpleNamespace(),
                         SimpleNamespace(stdout=""), SimpleNamespace(stdout=effective)],
        ) as command:
            _restrict_private_directory(directory, self.options)
        self.assertIn(f"sys.acl=u:{owner}:rwx,u:42:!r!x!w", command.call_args_list[1].args[0])
        self.assertEqual(command.call_args_list[0].args[0][-1], "/eos/user/h/hanw/private")
        with patch("pathlib.Path.resolve", return_value=Path("/eos/home-h/hanw/private")), patch(
            MODULE + ".subprocess.run", return_value=SimpleNamespace(stdout="g:123:rx"),
        ):
            with self.assertRaisesRegex(PermissionError, "non-user ACL"):
                _restrict_private_directory(directory, self.options)

    def test_cli_stays_isolated(self):
        for tail in (["--only", "qcd-data-mc"],
                     ["--from-cache", "--only", "qcd-data-mc", "qcd-score-scan"],
                     ["--from-cache", "--only", "qcd-score-scan", "--allow-unblinded-data"]):
            args = _build_parser().parse_args(["--config", str(CONFIG), *tail])
            with patch("tthcc_an.nn_study.cli.run_qcd_data_mc") as diagnostic:
                with self.assertRaises(ValueError):
                    run(args)
                diagnostic.assert_not_called()


if __name__ == "__main__":
    unittest.main()
