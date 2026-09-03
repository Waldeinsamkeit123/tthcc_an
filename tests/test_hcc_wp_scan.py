from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import awkward as ak
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tthcc_an.boosted_higgs_tagger_study.dataset import select_candidates
from tthcc_an.boosted_higgs_tagger_study.cli import (
    _build_empty_slim_study_data,
    _score_names_for_empty_study_data,
)
from tthcc_an.boosted_higgs_tagger_study.definitions import (
    TRUTH_LABEL_ORDER,
    TRUTH_LABEL_TO_CODE,
)
from tthcc_an.boosted_higgs_tagger_study.inclusive_higgs_wp import (
    NON_HIGGS_TRUTH_LABELS,
    build_inclusive_higgs_wp_aggregate,
    derive_inclusive_higgs_wp_results,
    normalize_inclusive_higgs_wp_config,
)
from tthcc_an.boosted_higgs_tagger_study.inclusive_higgs_reporting import (
    inclusive_higgs_purity_scan,
    inclusive_higgs_significance_scan,
)
from tthcc_an.boosted_higgs_tagger_study.scans import (
    EventLevelUnavailableError,
    build_hcc_wp_scan_aggregate,
    derive_event_level_results,
    derive_jet_level_results,
    normalize_hcc_wp_scan_config,
)


def scan_config(*, event_level: bool = False) -> dict[str, object]:
    return normalize_hcc_wp_scan_config(
        {
            "enabled": True,
            "x_min": 0.0,
            "x_max": 1.0,
            "x_points": 3,
            "y_min": 0.0,
            "y_max": 1.0,
            "y_points": 3,
            "background_efficiency_constraints": [0.2, 0.6],
            "event_level_enabled": event_level,
            "event_signal_processes": ["ttHcc"],
        }
    )


def process_entries() -> list[dict[str, object]]:
    return [
        {"code": 0, "process": "ttHcc", "label": "ttHcc", "color": "red"},
        {"code": 1, "process": "qcd", "label": "QCD", "color": "gray"},
        {"code": 2, "process": "ttHbb", "label": "ttHbb", "color": "blue"},
    ]


def make_data(
    truth_labels: list[str],
    x: list[float],
    y: list[float],
    weights: list[float],
    processes: list[int] | None = None,
    event_indices: list[int] | None = None,
) -> dict[str, np.ndarray]:
    n_entries = len(truth_labels)
    data = {
        "truth_code": np.asarray(
            [TRUTH_LABEL_TO_CODE[label] for label in truth_labels],
            dtype=np.int8,
        ),
        "gpart_higgs_vs_qcd": np.asarray(x, dtype=np.float64),
        "gpart_xbb_vs_xcc": np.asarray(y, dtype=np.float64),
        "weight": np.asarray(weights, dtype=np.float64),
        "weight_signed": np.asarray(weights, dtype=np.float64),
        "process_code": np.asarray(
            processes if processes is not None else [0] * n_entries,
            dtype=np.int16,
        ),
    }
    if event_indices is not None:
        data["event_index"] = np.asarray(event_indices, dtype=np.int64)
    return data


class HccJetScanTest(unittest.TestCase):
    def test_x_lower_and_y_upper_cut_directions(self) -> None:
        aggregate = build_hcc_wp_scan_aggregate(
            make_data(["hcc_pure"], [0.8], [0.2], [2.0]),
            process_entries(),
            scan_config(),
        )
        signal = aggregate["truth_yields"][TRUTH_LABEL_TO_CODE["hcc_pure"]]
        self.assertEqual(signal[1, 1], 2.0)
        self.assertEqual(signal[2, 1], 0.0)
        self.assertEqual(signal[1, 0], 0.0)

    def test_hcc_truth_signal_and_background_definition(self) -> None:
        aggregate = build_hcc_wp_scan_aggregate(
            make_data(
                TRUTH_LABEL_ORDER,
                [0.9] * len(TRUTH_LABEL_ORDER),
                [0.1] * len(TRUTH_LABEL_ORDER),
                [float(index + 1) for index in range(len(TRUTH_LABEL_ORDER))],
            ),
            process_entries(),
            scan_config(),
        )
        result = derive_jet_level_results(aggregate)
        self.assertEqual(result["baseline"]["S"], 1.0)
        self.assertEqual(result["baseline"]["B"], sum(range(2, 9)))

    def test_weighted_metrics_are_finite_and_correct(self) -> None:
        aggregate = build_hcc_wp_scan_aggregate(
            make_data(
                ["hcc_pure", "hbb_pure"],
                [0.8, 0.8],
                [0.2, 0.2],
                [3.0, 4.0],
                [0, 2],
            ),
            process_entries(),
            scan_config(),
        )
        result = derive_jet_level_results(aggregate)
        self.assertAlmostEqual(result["signal_over_background"][1, 1], 0.75)
        self.assertAlmostEqual(result["significance"][1, 1], 3.0 / np.sqrt(7.0))
        self.assertTrue(np.all(np.isfinite(result["signal_over_background"])))
        self.assertTrue(np.all(np.isfinite(result["significance"])))

    def test_global_and_constrained_optima(self) -> None:
        aggregate = build_hcc_wp_scan_aggregate(
            make_data(
                ["hcc_pure", "hcc_pure", "hbb_pure", "other"],
                [0.9, 0.6, 0.7, 0.4],
                [0.2, 0.4, 0.2, 0.2],
                [4.0, 2.0, 20.0, 20.0],
                [0, 0, 2, 1],
            ),
            process_entries(),
            scan_config(),
        )
        result = derive_jet_level_results(aggregate)
        recommendations = {
            row["label"]: row for row in result["recommendations"]
        }
        global_row = recommendations["global_max"]
        self.assertEqual((global_row["x_cut"], global_row["y_cut"]), (0.5, 0.5))
        constrained = recommendations["background_eff_le_20pct"]
        self.assertLessEqual(constrained["total_background_efficiency"], 0.2)
        self.assertEqual(
            constrained["S_over_sqrt_S_plus_B"],
            max(
                row["S_over_sqrt_S_plus_B"]
                for row in recommendations.values()
                if row["total_background_efficiency"] <= 0.2
            ),
        )


class HccEventScanTest(unittest.TestCase):
    def test_multiple_passing_jets_count_event_once(self) -> None:
        aggregate = build_hcc_wp_scan_aggregate(
            make_data(
                ["hcc_pure", "hcc_pure", "other"],
                [0.9, 0.8, 0.9],
                [0.1, 0.2, 0.1],
                [5.0, 5.0, 7.0],
                [0, 0, 1],
                [10, 10, 11],
            ),
            process_entries(),
            scan_config(event_level=True),
        )
        result = derive_event_level_results(aggregate)
        self.assertEqual(result["baseline"]["S"], 5.0)
        self.assertEqual(result["baseline"]["B"], 7.0)
        self.assertEqual(result["signal"][1, 1], 5.0)

    def test_missing_legacy_event_identity_is_explicit(self) -> None:
        with self.assertRaisesRegex(
            EventLevelUnavailableError,
            "event-level information unavailable in legacy cache",
        ):
            build_hcc_wp_scan_aggregate(
                make_data(["hcc_pure"], [0.9], [0.1], [1.0]),
                process_entries(),
                scan_config(event_level=True),
            )


class CandidateSelectionTest(unittest.TestCase):
    def test_pt_eta_and_open_msoftdrop_selection(self) -> None:
        jets = ak.Array(
            [
                {
                    "pt": [299.0, 300.0, 301.0, 301.0, 301.0],
                    "eta": [0.0, 2.4, 0.0, 0.0, 2.41],
                    "msoftdrop": [100.0, 50.0, 50.1, 199.9, 100.0],
                }
            ]
        )
        args = argparse.Namespace(
            pt_min=300.0,
            eta_max=2.4,
            candidate_strategy="mass_window_all_jets",
            msd_window_low=50.0,
            msd_window_high=200.0,
            msd_window_inclusive=False,
        )
        selected = select_candidates(jets, args)
        self.assertEqual(ak.to_list(selected.pt), [[301.0, 301.0]])
        self.assertEqual(ak.to_list(selected.msoftdrop), [[50.1, 199.9]])


class EmptyChunkTest(unittest.TestCase):
    def test_hcc_scan_scores_are_retained_for_empty_chunks(self) -> None:
        args = argparse.Namespace(
            targets=["hcc"],
            scores=["auto"],
            xbb_vs_xcc_region_preset="tight",
            hcc_wp_scan={"enabled": True},
            inclusive_higgs_wp={"enabled": True},
            y_split_study={"enabled": True},
        )
        score_names = _score_names_for_empty_study_data(args)
        data = _build_empty_slim_study_data(score_names)
        aggregate = build_hcc_wp_scan_aggregate(
            data,
            process_entries(),
            scan_config(event_level=True),
        )

        self.assertIn("gpart_higgs_vs_qcd", score_names)
        self.assertIn("gpart_xbb_vs_xcc", score_names)
        self.assertEqual(aggregate["truth_yields"].sum(), 0.0)
        self.assertEqual(aggregate["event_process_yields"].sum(), 0.0)



class InclusiveHiggsWPTest(unittest.TestCase):
    def config(self, *, scan_points=11, targets=None):
        return normalize_inclusive_higgs_wp_config(
            {
                "enabled": True,
                "scan_points": scan_points,
                "background_definition": "non_higgs_truth",
                "target_efficiencies": targets or [0.25, 0.5],
            }
        )

    def test_non_higgs_definition_and_selector_ignore_process_qcd(self) -> None:
        self.assertEqual(
            set(NON_HIGGS_TRUTH_LABELS),
            set(TRUTH_LABEL_ORDER) - {"hcc_pure", "hbb_pure"},
        )
        data = make_data(
            ["hcc_pure", "hbb_pure", "other", "other"],
            [0.9, 0.9, 0.8, 0.2],
            [0.1, 0.9, 0.2, 0.8],
            [100.0, 1.0, 1.0, 1.0],
            [1, 2, 0, 0],
            [1, 2, 3, 4],
        )
        result = derive_inclusive_higgs_wp_results(
            build_inclusive_higgs_wp_aggregate(
                data,
                process_entries(),
                self.config(scan_points=11, targets=[0.5]),
            )
        )
        recommendation = result["recommendations"][0]
        self.assertAlmostEqual(recommendation["x_cut"], 0.3)
        self.assertEqual(
            recommendation["achieved_non_higgs_jet_efficiency"], 0.5
        )
        self.assertEqual(recommendation["qcd_process_jet_efficiency"], 1.0)

    def test_x_boundary_non_higgs_weighting_and_no_y_requirement(self) -> None:
        data = make_data(
            ["hcc_pure", "hbb_pure", "other", "other"],
            [0.5, 0.9, 0.5, 0.4],
            [0.0, 1.0, 0.0, 1.0],
            [2.0, 3.0, 1.0, 3.0],
            [0, 2, 1, 1],
            [1, 2, 3, 4],
        )
        del data["gpart_xbb_vs_xcc"]
        aggregate = build_inclusive_higgs_wp_aggregate(
            data, process_entries(), self.config()
        )
        result = derive_inclusive_higgs_wp_results(aggregate)
        cut_index = int(np.flatnonzero(np.isclose(result["x_cuts"], 0.5))[0])
        self.assertEqual(
            aggregate["truth_yields"][TRUTH_LABEL_TO_CODE["hcc_pure"], cut_index],
            2.0,
        )
        self.assertAlmostEqual(result["non_higgs_jet_efficiency"][cut_index], 0.25)
        self.assertAlmostEqual(
            inclusive_higgs_purity_scan(result)[cut_index], 5.0 / 6.0
        )
        self.assertAlmostEqual(
            inclusive_higgs_significance_scan(result)[cut_index], 5.0 / np.sqrt(6.0)
        )

    def test_loosest_weighted_threshold(self) -> None:
        data = make_data(
            ["hcc_pure", "hbb_pure", "other", "other"],
            [0.9, 0.9, 0.8, 0.4],
            [0.1, 0.9, 0.2, 0.8],
            [1.0, 1.0, 1.0, 1.0],
            [0, 2, 1, 1],
            [1, 2, 3, 4],
        )
        result = derive_inclusive_higgs_wp_results(
            build_inclusive_higgs_wp_aggregate(
                data, process_entries(), self.config()
            )
        )
        recommendations = {
            row["target_non_higgs_jet_efficiency"]: row
            for row in result["recommendations"]
        }
        self.assertAlmostEqual(recommendations[0.5]["x_cut"], 0.5)
        self.assertAlmostEqual(recommendations[0.25]["x_cut"], 0.9)
        for row in recommendations.values():
            index = int(np.flatnonzero(np.isclose(result["x_cuts"], row["x_cut"]))[0])
            self.assertLessEqual(
                row["achieved_non_higgs_jet_efficiency"],
                row["target_non_higgs_jet_efficiency"],
            )
            self.assertGreater(
                result["non_higgs_jet_efficiency"][index - 1],
                row["target_non_higgs_jet_efficiency"],
            )

    def test_five_target_mapping_and_monotonicity(self) -> None:
        targets = [0.001, 0.005, 0.01, 0.02, 0.05]
        qcd_weights = [0.001, 0.004, 0.005, 0.01, 0.03, 0.95]
        data = make_data(
            ["hcc_pure", "hbb_pure"] + ["other"] * len(qcd_weights),
            [0.95, 0.9, 0.99, 0.95, 0.9, 0.8, 0.7, 0.1],
            [0.0] * (2 + len(qcd_weights)),
            [1.0, 1.0] + qcd_weights,
            [0, 2] + [1] * len(qcd_weights),
            list(range(2 + len(qcd_weights))),
        )
        result = derive_inclusive_higgs_wp_results(
            build_inclusive_higgs_wp_aggregate(
                data,
                process_entries(),
                self.config(scan_points=1001, targets=targets),
            )
        )
        self.assertEqual(
            [row["target_non_higgs_jet_efficiency"] for row in result["recommendations"]],
            targets,
        )
        for row in result["recommendations"]:
            index = int(round(row["x_cut"] * 1000))
            self.assertLessEqual(
                row["achieved_non_higgs_jet_efficiency"],
                row["target_non_higgs_jet_efficiency"] + 1.0e-12,
            )
            if index > 0:
                self.assertGreater(
                    result["non_higgs_jet_efficiency"][index - 1],
                    row["target_non_higgs_jet_efficiency"] - 1.0e-12,
                )
        self.assertTrue(np.all(np.diff(result["non_higgs_jet_efficiency"]) <= 1.0e-12))
        self.assertTrue(
            np.all(np.diff(result["event_efficiencies"]["ttHcc"]) <= 1.0e-12)
        )
        self.assertTrue(
            np.all(np.diff(result["event_efficiencies"]["ttHbb"]) <= 1.0e-12)
        )

    def test_event_deduplication_and_candidate_baselines(self) -> None:
        data = make_data(
            ["hcc_pure", "hcc_pure", "hbb_pure", "other", "other", "other"],
            [0.9, 0.8, 0.7, 0.9, 0.8, 0.1],
            [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
            [5.0, 5.0, 3.0, 7.0, 7.0, 11.0],
            [0, 0, 2, 1, 1, 1],
            [10, 10, 20, 30, 30, 31],
        )
        result = derive_inclusive_higgs_wp_results(
            build_inclusive_higgs_wp_aggregate(
                data, process_entries(), self.config(targets=[0.5])
            )
        )
        cut_index = int(np.flatnonzero(np.isclose(result["x_cuts"], 0.5))[0])
        self.assertEqual(result["event_baselines"]["ttHcc"], 5.0)
        self.assertEqual(result["event_baselines"]["ttHbb"], 3.0)
        self.assertEqual(result["event_baselines"]["qcd"], 18.0)
        self.assertEqual(result["event_yields"]["ttHcc"][cut_index], 5.0)
        self.assertEqual(result["event_yields"]["ttHbb"][cut_index], 3.0)
        self.assertEqual(result["event_yields"]["qcd"][cut_index], 7.0)
        self.assertAlmostEqual(
            result["event_efficiencies"]["qcd"][cut_index], 7.0 / 18.0
        )


if __name__ == "__main__":
    unittest.main()
