from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from tthcc_an.nn_study.plotting import _histograms_by_truth, plot_score_distribution
from tthcc_an.nn_study.tthz_qcd_discriminant import (
    derive_tthz_qcd_discriminant,
)


class TthzQcdDiscriminantTest(unittest.TestCase):
    def test_numerical_protection_and_unit_interval(self) -> None:
        values, diagnostics = derive_tthz_qcd_discriminant(
            np.asarray([0.8, 0.0, np.nan, 2.0, -1.0]),
            np.asarray([0.2, 0.0, 0.1, -1.0, 2.0]),
            np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        )
        np.testing.assert_allclose(values, [0.8, 0.5, 0.5, 1.0, 0.0])
        self.assertEqual(diagnostics["denominator_le_epsilon_event_count"], 1)
        self.assertEqual(diagnostics["denominator_le_epsilon_weighted_yield"], 2.0)
        self.assertEqual(diagnostics["nonfinite_input_event_count"], 1)
        self.assertEqual(diagnostics["clipped_to_unit_interval_event_count"], 2)
        self.assertTrue(diagnostics["all_outputs_finite"])
        self.assertTrue(diagnostics["all_outputs_in_unit_interval"])

    def test_weighted_score_stack_preserves_bin_yields(self) -> None:
        values = np.asarray([0.25, 0.25, 0.75])
        truth_index = np.asarray([0, 1, 0])
        weights = np.asarray([2.0, 3.0, 4.0])
        bins = np.asarray([0.0, 0.5, 1.0])
        truths = [SimpleNamespace(label="A", color="red"),
                  SimpleNamespace(label="B", color="blue")]
        histograms = _histograms_by_truth(values, truth_index, weights, truths, bins, normalize=False)
        np.testing.assert_allclose(np.sum(histograms, axis=0), [5.0, 4.0])
        config = SimpleNamespace(plot_options={"score_dpi": 60}, year=2024,
                                 lumi_fb=1.0, energy_tev=13)
        with TemporaryDirectory() as directory:
            outpath = Path(directory) / "stack.png"
            plot_score_distribution(
                outpath=outpath, score=SimpleNamespace(label="score"),
                values=values, truth_index=truth_index, weights=weights,
                truths=truths, config=config, normalize=False, stack=True,
                reference_style=True, bin_edges=bins,
            )
            self.assertTrue(outpath.is_file())
            with self.assertRaisesRegex(ValueError, "weighted yields"):
                plot_score_distribution(
                    outpath=outpath, score=SimpleNamespace(label="score"),
                    values=values, truth_index=truth_index, weights=weights,
                    truths=truths, config=config, normalize=True, stack=True,
                    bin_edges=bins,
                )


if __name__ == "__main__":
    unittest.main()
