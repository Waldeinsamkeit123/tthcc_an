from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tthcc_an.nn_study.analysis_regions import (
    _ar_histogram_edges_and_coverage,
    build_ar_mask,
    qcd_veto_threshold,
    run_conditional_qcd_veto,
)
from tthcc_an.nn_study.config import load_nn_study_config
from tthcc_an.nn_study.dataset import NnDataset
from tthcc_an.nn_study.plotting import _qcd_veto_axis_ticks
from tthcc_an.nn_study.qcd_low_score import (
    LOW_SCORE_THRESHOLDS,
    build_detailed_low_score_scan,
    build_logx_distribution,
)
from tthcc_an.nn_study.qcd_score_scan import (
    _normalized_variable_bin_density,
    _qcd_region_visualization_bins,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _dataset(
    scores: dict[str, np.ndarray],
    truth_index: np.ndarray,
    analysis_columns: dict[str, np.ndarray] | None = None,
) -> NnDataset:
    size = truth_index.size
    return NnDataset(
        scores=scores,
        truth_index=truth_index,
        prediction_truth_index=np.full(size, -1, dtype=np.int16),
        truth_match_count=np.ones(size, dtype=np.int16),
        prediction_truth_match_count=np.ones(size, dtype=np.int16),
        diagnostic_truth_masks={},
        sample_index=np.zeros(size, dtype=np.int16),
        raw_weight=np.ones(size),
        analysis_weight=np.ones(size),
        signed_weight=np.ones(size),
        analysis_columns=analysis_columns or {},
        sample_summaries=[],
        totals={"events_selected": size},
    )


class AnalysisRegionsTest(unittest.TestCase):
    def test_v4_ar_masks_and_shared_qcd_veto_threshold(self) -> None:
        config_0l = load_nn_study_config(
            REPO_ROOT / "config/nn_study/nn_study_2024_0l_v4.json",
            repo_root=REPO_ROOT,
            discover_sample_files=False,
        )
        dataset_0l = _dataset(
            {
                "ttHZ": np.asarray([0.70, 0.70, 0.69, 0.90]),
                "qcd": np.asarray([0.074, 0.075, 0.01, 0.20]),
            },
            np.zeros(4, dtype=np.int16),
        )
        np.testing.assert_array_equal(
            build_ar_mask(config_0l, dataset_0l),
            [True, False, False, False],
        )
        self.assertEqual(qcd_veto_threshold(config_0l), 0.075)

        config_1l = load_nn_study_config(
            REPO_ROOT / "config/nn_study/nn_study_2024_1l_v4.json",
            repo_root=REPO_ROOT,
            discover_sample_files=False,
        )
        dataset_1l = _dataset(
            {"ttHZ": np.asarray([0.70, 0.699, 0.90])},
            np.zeros(3, dtype=np.int16),
        )
        np.testing.assert_array_equal(
            build_ar_mask(config_1l, dataset_1l), [True, False, True]
        )
        self.assertIsNone(qcd_veto_threshold(config_1l))
        self.assertEqual(
            (
                config_1l.analysis_regions.ar_score_plot_ranges["ttHZ"].minimum,
                config_1l.analysis_regions.ar_score_plot_ranges["ttHZ"].maximum,
                config_1l.analysis_regions.ar_score_plot_ranges["ttHZ"].bins,
            ),
            (0.70, 1.00, 30),
        )
        self.assertEqual(
            config_0l.analysis_regions.ar_score_plot_ranges["qcd"].bins, 20
        )
        self.assertEqual(
            config_0l.analysis_regions.ar_score_plot_ranges["D_ttHZ"].minimum,
            0.70,
        )

    def test_ar_histogram_edges_preserve_tiny_float_excursions(self) -> None:
        config = load_nn_study_config(
            REPO_ROOT / "config/nn_study/nn_study_2024_1l_v4.json",
            repo_root=REPO_ROOT,
            discover_sample_files=False,
        )
        plot_range = config.analysis_regions.ar_score_plot_ranges["ttjets"]
        values = np.asarray([-5e-7, 0.1, 0.3000004])
        edges, coverage = _ar_histogram_edges_and_coverage(
            plot_range=plot_range,
            values=values,
            truth_index=np.zeros(3, dtype=np.int16),
            weights=np.asarray([1.0, 2.0, 3.0]),
            number_of_truths=1,
        )
        self.assertLess(edges[0], 0.0)
        self.assertGreater(edges[-1], 0.30)
        self.assertEqual(coverage["below_nominal_event_count"], 1)
        self.assertEqual(coverage["above_nominal_event_count"], 1)
        self.assertEqual(coverage["histogram_weighted_yield"], 6.0)
        self.assertTrue(coverage["weighted_yield_preserved"])

        with self.assertRaisesRegex(ValueError, "extends .* above"):
            _ar_histogram_edges_and_coverage(
                plot_range=plot_range,
                values=np.asarray([0.31]),
                truth_index=np.zeros(1, dtype=np.int16),
                weights=np.ones(1),
                number_of_truths=1,
            )

    def test_qcd_veto_boundary_tick_is_only_on_left_axis(self) -> None:
        left_ticks, right_ticks = _qcd_veto_axis_ticks(0.075, 1.0)
        np.testing.assert_allclose(left_ticks, [0.025, 0.05, 0.075])
        np.testing.assert_allclose(right_ticks, [0.2, 0.4, 0.6, 0.8, 1.0])
        self.assertEqual(np.count_nonzero(left_ticks == 0.075), 1)
        self.assertEqual(np.count_nonzero(right_ticks == 0.075), 0)

    def test_detailed_low_score_scan_stages_and_monotonicity(self) -> None:
        config = load_nn_study_config(
            REPO_ROOT / "config/nn_study/nn_study_2024_0l_v4.json",
            repo_root=REPO_ROOT,
            discover_sample_files=False,
        )
        names = ["ttHcc", "ttHbb", "ttZqq", "ttLF", "ttcj", "ttbj", "qcd", "qcd"]
        truth = np.asarray([config.truth_names.index(name) for name in names])
        score_qcd = np.asarray([0.02, 0.04, 0.06, 0.08, 0.03, 0.05, 0.09, 0.11])
        dataset = _dataset(
            {
                "ttHZ": np.full(8, 0.8),
                "ttjets": np.full(8, 0.1),
                "qcd": score_qcd,
            },
            truth,
            {
                "TargetFatJet_regressed_mass_generic": np.asarray(
                    [125.0, 125.0, 125.0, 125.0, 125.0, 125.0, 160.0, 125.0]
                )
            },
        )
        scan = build_detailed_low_score_scan(
            config=config,
            dataset=dataset,
            score=score_qcd,
            weights=dataset.analysis_weight,
            eligible=np.ones(8, dtype=bool),
        )
        self.assertEqual(tuple(scan["thresholds"]), LOW_SCORE_THRESHOLDS)
        self.assertEqual(scan["stages"]["ar"]["before_qcd_cut_event_count"], 8)
        self.assertEqual(
            scan["stages"]["ar_mass_window"]["before_qcd_cut_event_count"], 7
        )
        row = next(
            item
            for item in scan["stages"]["ar"]["rows"]
            if item["threshold"] == 0.1
        )
        self.assertEqual(row["groups"]["qcd"]["event_count"], 1)
        self.assertFalse(scan["validation"]["flavour_tagging_applied"])
        self.assertTrue(
            all(
                value
                for key, value in scan["validation"].items()
                if key != "flavour_tagging_applied"
            )
        )

    def test_logx_distribution_tracks_underflow_and_normalizes(self) -> None:
        config = load_nn_study_config(
            REPO_ROOT / "config/nn_study/nn_study_2024_0l_v4.json",
            repo_root=REPO_ROOT,
            discover_sample_files=False,
        )
        qcd_index = config.truth_names.index("qcd")
        score = np.asarray([0.0, 5e-6, 5e-5, 0.05])
        dataset = _dataset(
            {
                "ttHZ": np.full(4, 0.8),
                "ttjets": np.full(4, 0.1),
                "qcd": score,
            },
            np.full(4, qcd_index),
        )
        bins, histograms, metadata = build_logx_distribution(
            config=config,
            dataset=dataset,
            score=score,
            weights=dataset.analysis_weight,
            eligible=np.ones(4, dtype=bool),
        )
        self.assertEqual(len(bins), 51)
        self.assertEqual(metadata["score_qcd_nonpositive_event_count"], 1)
        self.assertEqual(metadata["positive_underflow_event_count"], 1)
        self.assertAlmostEqual(float(np.sum(histograms["qcd"])), 1.0)

    def test_conditional_qcd_rows_and_strict_upper_cut(self) -> None:
        config = load_nn_study_config(
            REPO_ROOT / "config/nn_study/nn_study_2024_0l_v4.json",
            repo_root=REPO_ROOT,
            discover_sample_files=False,
        )
        truth = np.asarray([0, 1, 2, 5, 12], dtype=np.int16)
        dataset = _dataset(
            {
                "ttHZ": np.asarray([0.8, 0.8, 0.8, 0.8, 0.8]),
                "qcd": np.asarray([0.05, 0.075, 0.10, 0.15, 0.20]),
            },
            truth,
        )
        with tempfile.TemporaryDirectory() as temporary:
            _, payload = run_conditional_qcd_veto(
                config=config,
                dataset=dataset,
                summary_dir=Path(temporary),
            )
        self.assertEqual(len(payload["rows"]), 8)
        row = next(
            entry
            for entry in payload["rows"]
            if entry["tthz_threshold"] == 0.7
            and entry["qcd_threshold"] == 0.075
        )
        self.assertEqual(row["weighted_yields"]["ttHcc"], 1.0)
        self.assertEqual(row["weighted_yields"]["ttHbb"], 0.0)

    def test_qcd_veto_variable_bins_and_global_density(self) -> None:
        bins = _qcd_region_visualization_bins(0.075)
        widths = np.diff(bins)
        self.assertEqual(np.count_nonzero(bins == 0.075), 1)
        np.testing.assert_allclose(widths[:20], 0.005)
        np.testing.assert_allclose(widths[20:], 0.05)
        histogram = np.arange(1, widths.size + 1, dtype=np.float64)
        density = _normalized_variable_bin_density(histogram, widths)
        self.assertAlmostEqual(float(np.sum(density * widths)), 1.0)


if __name__ == "__main__":
    unittest.main()
