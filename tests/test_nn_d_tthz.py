from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import uproot

from tthcc_an.nn_study.cache import (
    cache_score_names,
    load_nn_cache,
    write_nn_cache,
)
from tthcc_an.nn_study.config import NnSample, load_nn_study_config
from tthcc_an.nn_study.d_tthz_study import validate_persisted_d_tthz
from tthcc_an.nn_study.dataset import load_nn_dataset
from tthcc_an.nn_study.score_significance import calculate_score_significance_scan
from tthcc_an.nn_study.score_significance_2d import _category_rectangle_yields


REPO_ROOT = Path(__file__).resolve().parents[1]
V4_CONFIG = REPO_ROOT / "config/nn_study/nn_study_2024_0l_v4.json"


class PersistedDtthzTest(unittest.TestCase):
    def _synthetic_dataset(self, directory: Path):
        root_path = directory / "events.root"
        d_values = np.asarray([0.8, 0.2, 0.6], dtype=np.float32)
        arrays = {
            "D_ttHZ": d_values,
            "score_ttHZ": np.asarray([0.4, 0.1, 0.3], dtype=np.float32),
            "score_ttjets": np.asarray([0.1, 0.4, 0.2], dtype=np.float32),
            "score_qcd": np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
            "weight": np.asarray([1.0, -2.0, 3.0]),
            "higgs_decay": np.asarray([4, 0, 0], dtype=np.int32),
            "z_decay": np.asarray([0, 0, 0], dtype=np.int32),
            "w_decay": np.asarray([0, 0, 0], dtype=np.int32),
            "n_gentop": np.asarray([2, 2, 0], dtype=np.int32),
            "tt_hf_flavor": np.asarray([0, 0, 0], dtype=np.int32),
            "tt_hf_count": np.asarray([0, 0, 0], dtype=np.int32),
        }
        with uproot.recreate(root_path) as output:
            output["Events"] = arrays
        gen_sumw = directory / "gen_sumws.json"
        xsecs = directory / "xsecs.json"
        gen_sumw.write_text(json.dumps({"synthetic": 1.0}), encoding="utf-8")
        xsecs.write_text(json.dumps({"synthetic": 1.0}), encoding="utf-8")
        config = load_nn_study_config(
            V4_CONFIG,
            repo_root=REPO_ROOT,
            discover_sample_files=False,
        )
        config = replace(
            config,
            samples=[NnSample("synthetic", "synthetic", "synthetic", [str(root_path)], "")],
            gen_sumw_file=str(gen_sumw),
            xsec_file=str(xsecs),
            lumi_fb=1.0,
            analysis_branches=[],
            mass_sculpting=replace(config.mass_sculpting, enabled=False),
            significance_mass_window=replace(
                config.significance_mass_window, enabled=False, branch=None
            ),
        )
        return config, load_nn_dataset(config, analysis_branches=[])

    def test_v4_loads_persisted_branch_and_older_configs_do_not_require_it(self) -> None:
        v4 = load_nn_study_config(V4_CONFIG, repo_root=REPO_ROOT, discover_sample_files=False)
        self.assertIn("D_ttHZ", cache_score_names(v4))
        self.assertEqual(v4.analysis_discriminants, ["D_ttHZ"])
        self.assertEqual(
            next(score.branch for score in v4.auxiliary_scores if score.name == "D_ttHZ"),
            "D_ttHZ",
        )
        for name in (
            "nn_study_2024_0l_v1.json",
            "nn_study_2024_0l_v2_predMass.json",
            "nn_study_2024_0l_v3_zeroMass.json",
            "nn_study_2024_1l_v4.json",
        ):
            config = load_nn_study_config(
                REPO_ROOT / "config/nn_study" / name,
                repo_root=REPO_ROOT,
                discover_sample_files=False,
            )
            self.assertNotIn("D_ttHZ", cache_score_names(config))
            self.assertEqual(config.analysis_discriminants, [])
            self.assertFalse(config.score_significance_2d_hierarchical.enabled)

    def test_root_branch_load_validation_and_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config, dataset = self._synthetic_dataset(directory)
            np.testing.assert_allclose(dataset.scores["D_ttHZ"], [0.8, 0.2, 0.6])
            validation = validate_persisted_d_tthz(dataset)
            self.assertEqual(validation["nonfinite_event_count"], 0)
            self.assertEqual(validation["out_of_range_event_count"], 0)
            self.assertLess(validation["max_absolute_difference"], 1e-7)

            cache_path = directory / "prepared_events.npz"
            metadata = write_nn_cache(
                config=config, dataset=dataset, cache_path=cache_path
            )
            self.assertEqual(
                metadata["stored_arrays"]["scores"]["D_ttHZ"]["branch"],
                "D_ttHZ",
            )
            loaded, _ = load_nn_cache(
                config=config,
                cache_path=cache_path,
                score_names=cache_score_names(config),
                analysis_branches=[],
            )
            np.testing.assert_array_equal(
                loaded.scores["D_ttHZ"], dataset.scores["D_ttHZ"]
            )

    def test_greater_than_significance_and_hierarchical_rectangle(self) -> None:
        d_score = np.asarray([0.2, 0.5, 0.7, 0.9])
        qcd_score = np.asarray([0.01, 0.10, 0.15, 0.30])
        weights = np.asarray([1.0, 2.0, 3.0, 4.0])
        truth = np.asarray([0, 1, 0, 1])
        result = calculate_score_significance_scan(
            score=d_score,
            weights=weights,
            truth_index=truth,
            truth_names=["ttHcc", "background"],
            signals=["ttHcc"],
            thresholds=np.asarray([0.0, 0.5, 1.0]),
            direction=">",
        )
        middle = result["signals"]["ttHcc"]["points"][1]
        self.assertEqual(middle["signal_weighted_yield"], 3.0)
        self.assertEqual(middle["background_weighted_yield"], 4.0)

        config = load_nn_study_config(V4_CONFIG, repo_root=REPO_ROOT, discover_sample_files=False)
        actual, _ = _category_rectangle_yields(
            score_x=d_score,
            score_y=qcd_score,
            weights=weights,
            truth_index=truth,
            number_of_truths=2,
            x_axis=config.score_significance_2d_hierarchical.x_axis,
            y_axis=config.score_significance_2d_hierarchical.y_axis,
        )
        x_index = config.score_significance_2d_hierarchical.x_axis.thresholds.index(0.6)
        y_index = config.score_significance_2d_hierarchical.y_axis.thresholds.index(0.2)
        self.assertEqual(float(np.sum(actual[:, x_index, y_index])), 3.0)


if __name__ == "__main__":
    unittest.main()
