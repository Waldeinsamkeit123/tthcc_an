from __future__ import annotations

import unittest

import numpy as np

from tthcc_an.nn_study.config import NnScoreSignificance2DAxis
from tthcc_an.nn_study.score_significance_2d import (
    _category_rectangle_yields,
)


class ScoreSignificance2DTest(unittest.TestCase):
    def test_rectangle_accumulation_matches_strict_brute_force_cuts(self) -> None:
        rng = np.random.default_rng(42)
        size = 2000
        score_x = rng.uniform(-0.2, 1.2, size)
        score_y = rng.uniform(-0.2, 1.2, size)
        weights = rng.uniform(0.0, 3.0, size)
        truth_index = rng.integers(-1, 3, size)
        x_thresholds = [0.2, 0.5, 0.8]
        y_thresholds = [0.1, 0.4, 0.9]

        for x_direction in ("<", ">"):
            for y_direction in ("<", ">"):
                x_axis = NnScoreSignificance2DAxis(
                    "x", "score_x", x_direction, x_thresholds
                )
                y_axis = NnScoreSignificance2DAxis(
                    "y", "score_y", y_direction, y_thresholds
                )
                actual, _ = _category_rectangle_yields(
                    score_x=score_x,
                    score_y=score_y,
                    weights=weights,
                    truth_index=truth_index,
                    number_of_truths=3,
                    x_axis=x_axis,
                    y_axis=y_axis,
                )
                category = np.where(
                    (truth_index >= 0) & (truth_index < 3),
                    truth_index + 1,
                    0,
                )
                for x_index, x_threshold in enumerate(x_thresholds):
                    x_keep = (
                        score_x < x_threshold
                        if x_direction == "<"
                        else score_x > x_threshold
                    )
                    for y_index, y_threshold in enumerate(y_thresholds):
                        y_keep = (
                            score_y < y_threshold
                            if y_direction == "<"
                            else score_y > y_threshold
                        )
                        keep = x_keep & y_keep
                        expected = np.bincount(
                            category[keep],
                            weights=weights[keep],
                            minlength=4,
                        )
                        np.testing.assert_allclose(
                            actual[:, x_index, y_index],
                            expected,
                            rtol=1e-12,
                            atol=1e-12,
                        )


if __name__ == "__main__":
    unittest.main()
