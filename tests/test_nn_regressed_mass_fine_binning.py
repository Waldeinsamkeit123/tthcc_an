from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from tthcc_an.nn_study.regressed_mass_fine_binning import (
    _data_cache_definition,
    _data_cache_path,
    _mc_histograms,
    _plot_data_mc,
    _plot_mcstat_heatmap,
    _ratio_arrays,
    _region_metrics,
    _unblinded_data_histograms,
    _settings,
    load_data_cache,
)


class RegressedMassFineBinningTest(unittest.TestCase):
    def test_blinding_ratio_and_mc_stat(self) -> None:
        bins = np.arange(50.0, 205.0, 5.0)
        mass = np.asarray([99.0, 100.0, 125.0, 150.0, 150.001, 200.0])
        counts = _unblinded_data_histograms(
            np.full(6, 0.005), np.full(6, 0.8), mass,
            qcd_cut=0.01, cuts=[0.5, 0.7], bins=bins,
            blind_range=(100.0, 150.0),
        )
        blinded = (bins[:-1] >= 100) & (bins[1:] <= 150)
        np.testing.assert_array_equal(np.sum(counts, axis=1), [3, 3])
        self.assertTrue(np.all(counts[:, blinded] == 0))
        data = counts[0].copy()
        data[blinded] = np.nan
        prediction = np.full(30, 2.0)
        variance = np.full(30, 1.0)
        prediction[0] = 0.0
        ratio, errors, band, visible = _ratio_arrays(data, prediction, variance, blinded)
        self.assertFalse(np.any(visible[blinded]))
        self.assertTrue(np.all(np.isnan(ratio[blinded])))
        self.assertTrue(np.all(np.isnan(errors[blinded])))
        np.testing.assert_allclose(band[blinded], 0.5)
        self.assertTrue(np.isnan(band[0]))
        metrics = _region_metrics({
            "count": np.asarray([2, 1, 0]),
            "sumw": np.asarray([3.0, 4.0, 0.0]),
            "sumw2": np.asarray([5.0, 16.0, 0.0]),
        }, np.asarray([True, True, True]))
        self.assertAlmostEqual(metrics["integrated_neff"], 49 / 21)
        self.assertAlmostEqual(metrics["integrated_mc_stat"], np.sqrt(21))
        self.assertEqual(metrics["nonempty_bins"], 2)
        self.assertEqual(metrics["bins_rel_stat_gt_50_percent"], 2)

    def test_process_stack_variance_and_plot_rendering(self) -> None:
        names = ["ttHcc", "ttHbb", "ttZqq", "ttZcc", "ttZbb", "ttLF",
                 "ttcj", "tt2c", "ttcc", "ttbj", "tt2b", "ttbb", "qcd"]
        config = SimpleNamespace(truth_names=names, year="2024", lumi_fb=109.08,
                                 energy_tev=13.6, plot_options={})
        dataset = SimpleNamespace(
            truth_index=np.asarray([12, 12, 0, 0]),
            sample_index=np.asarray([0, 0, 1, 1]),
            sample_summaries=[{"label": "QCD"}, {"label": "ttHcc"}],
            scores={"qcd": np.asarray([0.005, 0.005, 0.005, 0.005]),
                    "ttHZ": np.asarray([0.8, 0.6, 0.8, 0.6])},
            analysis_columns={"TargetFatJet_regressed_mass_generic":
                              np.asarray([90.0, 120.0, 90.0, 120.0])},
            analysis_weight=np.asarray([2.0, 3.0, 4.0, 5.0]),
        )
        bins = np.arange(50.0, 205.0, 5.0)
        groups, _, by_cut = _mc_histograms(
            config, dataset, bins, {"qcd_cut": 0.01, "tthz_cuts": [0.5, 0.7]}
        )
        selected = by_cut[0.5]
        self.assertAlmostEqual(sum(np.sum(value["sumw"]) for value in selected.values()), 14)
        self.assertAlmostEqual(sum(np.sum(value["sumw2"]) for value in selected.values()), 54)
        blinded = (bins[:-1] >= 100) & (bins[1:] <= 150)
        data = np.zeros(len(bins) - 1)
        data[blinded] = np.nan
        with TemporaryDirectory() as directory:
            stack_path = Path(directory) / "data_blinded.png"
            heatmap_path = Path(directory) / "mcstat.png"
            _plot_data_mc(stack_path, bins, groups, selected, data, blinded,
                          (100, 150), config, 0.01, 0.5, (0.5, 1.5))
            _plot_mcstat_heatmap(heatmap_path, bins, groups, selected, config,
                                 0.01, 0.5, (100, 150))
            self.assertTrue(stack_path.is_file())
            self.assertTrue(heatmap_path.is_file())

    def test_data_cache_rejects_blind_counts_and_partial_cache_as_full(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trigger_path = root / "trigger.json"
            trigger_path.write_text(json.dumps({
                "dataset_trigger_order": ["ParkingHH", "JetMET"],
                "dataset_trigger_map": {"ParkingHH": ["HLT_A"], "JetMET": ["HLT_B"]},
            }))
            options = {
                "enabled": True, "qcd_cut": 0.01,
                "tthz_cuts": [0.30, 0.40, 0.50, 0.60, 0.70],
                "mcstat_detailed_cuts": [0.50, 0.70],
                "mass_range": [50, 200], "mass_bin_width": 5,
                "blind_mass_range": [100, 150],
                "data_samples": ["ParkingHH", "JetMET"],
                "data_trigger_config": str(trigger_path),
            }
            config = SimpleNamespace(
                regressed_mass_fine_binning=options, channel="0L V4",
                outdir=root, input_location="/unused", max_files_per_sample=1,
            )
            _, bins, blinded = _settings(config)
            cache_path = _data_cache_path(config)
            cache_path.parent.mkdir(parents=True)
            definition = json.dumps(_data_cache_definition(config, options, bins))
            counts = np.zeros((5, 30))
            np.savez_compressed(cache_path, counts=counts, definition=definition)
            with self.assertRaisesRegex(ValueError, "unblinded"):
                load_data_cache(config)
            counts[:, blinded] = np.nan
            np.savez_compressed(cache_path, counts=counts, definition=definition)
            self.assertTrue(np.all(np.isnan(load_data_cache(config)[:, blinded])))
            config.max_files_per_sample = None
            with self.assertRaisesRegex(ValueError, "settings changed"):
                load_data_cache(config)


if __name__ == "__main__":
    unittest.main()
