from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tthcc_an.boosted_higgs_tagger_study.definitions import TRUTH_LABEL_TO_CODE
from tthcc_an.boosted_higgs_tagger_study.cache import build_histogram_payload_from_raw_data
from tthcc_an.boosted_higgs_tagger_study.inclusive_higgs_wp import normalize_inclusive_higgs_wp_config
from tthcc_an.boosted_higgs_tagger_study.y_split_cache import (
    export_y_split_aggregate,
    load_y_split_aggregate,
)
from tthcc_an.boosted_higgs_tagger_study.y_split_reporting import (
    derive_y_split_rows,
    hcc_vs_hbb_purity_scan,
    hcc_vs_hbb_significance_scan,
)
from tthcc_an.boosted_higgs_tagger_study.strategy_reporting import (
    derive_strategy_comparison,
)
from tthcc_an.boosted_higgs_tagger_study.scans import (
    build_hcc_wp_scan_aggregate,
    normalize_hcc_wp_scan_config,
)
from tthcc_an.boosted_higgs_tagger_study.y_split_study import (
    EVENT_STATES,
    build_y_split_aggregate,
    merge_y_split_aggregates,
    normalize_y_split_config,
    resolve_inclusive_x_working_points,
)
from tests.test_hcc_wp_scan import make_data, process_entries


def config() -> dict[str, object]:
    payload = normalize_y_split_config(
        {
            "enabled": True,
            "inclusive_higgs_non_higgs_targets": [0.01, 0.02],
            "scan": {"min": 0.0, "max": 1.0, "points": 3},
            "reference_points": [0.5],
        }
    )
    payload["resolved_x_working_points"] = [
        {"non_higgs_target_efficiency": 0.01, "x_cut": 0.5},
        {"non_higgs_target_efficiency": 0.02, "x_cut": 0.4},
    ]
    return payload


class YSplitJetTest(unittest.TestCase):
    def test_boundary_partition_weighting_and_conditional_rows(self) -> None:
        data = make_data(
            ["hcc_pure", "hbb_pure", "other"],
            [0.9, 0.9, 0.9],
            [0.5, 0.5001, 0.2],
            [2.0, 3.0, 7.0],
            [0, 2, 1],
            [10, 20, 30],
        )
        aggregate = build_y_split_aggregate(data, process_entries(), config())
        y_index = 1
        hcc = TRUTH_LABEL_TO_CODE["hcc_pure"]
        hbb = TRUTH_LABEL_TO_CODE["hbb_pure"]
        self.assertEqual(aggregate["jet_truth_region_yields"][0, hcc, y_index, 0], 2.0)
        self.assertEqual(aggregate["jet_truth_region_yields"][0, hcc, y_index, 1], 0.0)
        self.assertEqual(aggregate["jet_truth_region_yields"][0, hbb, y_index, 0], 0.0)
        self.assertEqual(aggregate["jet_truth_region_yields"][0, hbb, y_index, 1], 3.0)

        purity = hcc_vs_hbb_purity_scan(aggregate)
        self.assertEqual(purity[y_index], 1.0)
        self.assertAlmostEqual(purity[-1], 2.0 / 5.0)
        significance = hcc_vs_hbb_significance_scan(aggregate)
        self.assertAlmostEqual(significance[y_index], 2.0 / np.sqrt(2.0))
        self.assertAlmostEqual(significance[-1], 2.0 / np.sqrt(5.0))

        row = derive_y_split_rows(aggregate)[y_index]
        self.assertAlmostEqual(
            row["jet_conditional_hcc_to_Hcc"]
            + row["jet_conditional_hcc_to_Hbb"],
            1.0,
        )
        self.assertAlmostEqual(
            row["jet_conditional_hbb_to_Hcc"]
            + row["jet_conditional_hbb_to_Hbb"],
            1.0,
        )
        self.assertAlmostEqual(
            row["jet_conditional_qcd_to_Hcc"]
            + row["jet_conditional_qcd_to_Hbb"],
            1.0,
        )

    def test_config_has_only_common_x_working_points(self) -> None:
        payload = config()
        self.assertNotIn("x_Hcc", payload)
        self.assertNotIn("x_Hbb", payload)
        self.assertEqual(
            [row["non_higgs_target_efficiency"] for row in payload["resolved_x_working_points"]],
            [0.01, 0.02],
        )


class YSplitEventTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data = make_data(
            [
                "hcc_pure", "hcc_pure",
                "hbb_pure", "hbb_pure",
                "other", "other", "other",
            ],
            [0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.1],
            [0.2, 0.3, 0.7, 0.8, 0.2, 0.8, 0.2],
            [5.0, 5.0, 3.0, 3.0, 7.0, 7.0, 11.0],
            [0, 0, 2, 2, 1, 1, 1],
            [10, 10, 20, 20, 30, 30, 31],
        )
        self.aggregate = build_y_split_aggregate(
            self.data, process_entries(), config()
        )

    def test_multijet_dedup_overlap_raw_counts_and_weights(self) -> None:
        state = {name: index for index, name in enumerate(EVENT_STATES)}
        values = self.aggregate["state_weighted_yields"][0, :, 1]
        counts = self.aggregate["state_raw_counts"][0, :, 1]
        self.assertEqual(values[0, state["hcc_only"]], 5.0)
        self.assertEqual(counts[0, state["hcc_only"]], 1)
        self.assertEqual(values[2, state["hbb_only"]], 3.0)
        self.assertEqual(counts[2, state["hbb_only"]], 1)
        self.assertEqual(values[1, state["both"]], 7.0)
        self.assertEqual(counts[1, state["both"]], 1)
        self.assertEqual(values[1, state["neither"]], 11.0)
        self.assertEqual(counts[1, state["neither"]], 1)
        self.assertEqual(self.aggregate["baseline_raw_counts"][1], 2)
        self.assertEqual(self.aggregate["inclusive_raw_counts"][0, 1], 1)

    def test_event_hcc_and_hbb_are_not_forced_exclusive(self) -> None:
        row = derive_y_split_rows(self.aggregate)[1]
        self.assertEqual(row["qcd_Hcc_pass_weighted_yield"], 7.0)
        self.assertEqual(row["qcd_Hbb_pass_weighted_yield"], 7.0)
        self.assertEqual(row["qcd_both_weighted_yield"], 7.0)
        self.assertEqual(row["qcd_both_raw_event_count"], 1)

    def test_cache_round_trip_and_process_remapped_merge(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        metadata = export_y_split_aggregate(arrays, self.aggregate)
        loaded = load_y_split_aggregate(arrays, {"y_split_study": metadata})
        merged = merge_y_split_aggregates([loaded, loaded])
        self.assertTrue(
            np.array_equal(
                loaded["state_raw_counts"], self.aggregate["state_raw_counts"]
            )
        )
        original = {
            entry["process"]: int(entry["code"])
            for entry in self.aggregate["process_entries"]
        }
        result = {
            entry["process"]: int(entry["code"])
            for entry in merged["process_entries"]
        }
        for process in original:
            self.assertEqual(
                merged["baseline_raw_counts"][result[process]],
                2 * self.aggregate["baseline_raw_counts"][original[process]],
            )




class UnifiedBoostedWorkflowTest(unittest.TestCase):
    def test_one_candidate_payload_builds_inclusive_and_y_split_aggregates(self) -> None:
        data = make_data(
            ["hcc_pure", "hbb_pure", "other"],
            [0.9, 0.8, 0.2],
            [0.2, 0.8, 0.4],
            [2.0, 3.0, 7.0],
            [0, 1, 2],
            [10, 20, 30],
        )
        summaries = [
            {"process": "ttHcc", "label": "ttHcc"},
            {"process": "ttHbb", "label": "ttHbb"},
            {"process": "qcd", "label": "QCD"},
        ]
        inclusive_config = normalize_inclusive_higgs_wp_config(
            {
                "enabled": True,
                "scan_points": 3,
                "target_efficiencies": [0.01, 0.02],
            }
        )
        payload = build_histogram_payload_from_raw_data(
            data=data,
            sample_summaries=summaries,
            score_names=["gpart_higgs_vs_qcd", "gpart_xbb_vs_xcc"],
            n_bins=10,
            contour_plot_defs=[],
            inclusive_higgs_wp_config=inclusive_config,
            y_split_config=config(),
        )
        self.assertIsNotNone(payload["inclusive_higgs_wp"])
        self.assertIsNotNone(payload["y_split_study"])
        self.assertTrue(payload["y_split_study"]["availability"]["event_raw_counts"])
        self.assertTrue(
            payload["y_split_study"]["availability"]["event_weighted_complete"]
        )

    def test_diagnostic_candidates_are_not_final_selection(self) -> None:
        candidates = config()["diagnostic_candidates"]
        self.assertEqual(
            [(item["y_split"], item["role"]) for item in candidates],
            [
                (0.80, "nominal_candidate_for_further_study"),
                (0.85, "historical_reference_candidate"),
            ],
        )

class StrategyComparisonTest(unittest.TestCase):
    def test_exact_candidates_and_historical_availability(self) -> None:
        strategy_config = normalize_y_split_config(
            {
                "enabled": True,
                "inclusive_higgs_non_higgs_targets": [0.01, 0.02],
                "scan": {"min": 0.0, "max": 1.0, "points": 6},
                "reference_points": [0.8, 0.85],
            }
        )
        strategy_config["resolved_x_working_points"] = [
            {"non_higgs_target_efficiency": 0.01, "achieved_non_higgs_jet_efficiency": 1.0, "x_cut": 0.5},
            {"non_higgs_target_efficiency": 0.02, "achieved_non_higgs_jet_efficiency": 1.0, "x_cut": 0.4},
        ]
        data = make_data(
            ["hcc_pure", "hbb_pure", "other"],
            [0.9, 0.9, 0.9],
            [0.2, 0.9, 0.2],
            [2.0, 3.0, 7.0],
            [0, 2, 1],
            [10, 20, 30],
        )
        y_split = build_y_split_aggregate(
            data, process_entries(), strategy_config
        )
        hcc_config = normalize_hcc_wp_scan_config(
            {
                "enabled": True,
                "x_min": 0.0,
                "x_max": 1.0,
                "x_points": 21,
                "y_min": 0.0,
                "y_max": 1.0,
                "y_points": 21,
                "event_level_enabled": True,
                "event_signal_processes": ["ttHcc"],
            }
        )
        hcc = build_hcc_wp_scan_aggregate(
            data, process_entries(), hcc_config
        )

        nominal, tighter, historical = derive_strategy_comparison(y_split, hcc)
        self.assertEqual(nominal["inclusive_non_higgs_target"], 0.02)
        self.assertEqual(nominal["x_cut"], 0.4)
        self.assertEqual(nominal["y_split"], 0.8)
        self.assertAlmostEqual(
            nominal["non_higgs_to_Hcc"] + nominal["non_higgs_to_Hbb"],
            nominal["achieved_non_higgs_jet_efficiency"],
        )
        self.assertAlmostEqual(
            nominal["achieved_non_higgs_jet_efficiency"]
            + nominal["non_higgs_rejected"],
            1.0,
        )
        self.assertTrue(nominal["event_level_available"])
        self.assertEqual(tighter["inclusive_non_higgs_target"], 0.01)
        self.assertEqual(tighter["x_cut"], 0.5)
        self.assertEqual(historical["x_convention"], ">")
        self.assertAlmostEqual(historical["x_cut_evaluated"], 0.95)
        self.assertFalse(historical["event_level_available"])
        self.assertIsNone(historical["ttHcc_Hcc_pass_weighted_yield"])


class InclusiveWPResolutionTest(unittest.TestCase):
    def test_x_working_points_are_loaded_by_target_not_hard_coded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inclusive.json"
            path.write_text(
                json.dumps(
                    {
                        "recommendations": [
                            {"target_non_higgs_jet_efficiency": 0.02, "achieved_non_higgs_jet_efficiency": 0.019, "x_cut": 0.8123},
                            {"target_non_higgs_jet_efficiency": 0.01, "achieved_non_higgs_jet_efficiency": 0.009, "x_cut": 0.9234},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            payload = normalize_y_split_config(
                {
                    "enabled": True,
                    "inclusive_higgs_non_higgs_targets": [0.01, 0.02],
                    "inclusive_higgs_result": str(path),
                }
            )
            resolved = resolve_inclusive_x_working_points(
                payload, repo_root=REPO_ROOT
            )
            self.assertEqual(
                [row["x_cut"] for row in resolved],
                [0.9234, 0.8123],
            )


if __name__ == "__main__":
    unittest.main()
