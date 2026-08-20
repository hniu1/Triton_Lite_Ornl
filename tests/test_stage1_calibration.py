import unittest

import numpy as np

from stage1_calibrate_wet_threshold import scores_from_counts, select_best, threshold_grid


class Stage1CalibrationTest(unittest.TestCase):
    def test_threshold_grid_includes_endpoints(self):
        self.assertTrue(np.allclose(threshold_grid(0.1, 0.2, 0.05), [0.1, 0.15, 0.2]))

    def test_selection_honors_minimum_precision(self):
        rows = scores_from_counts(
            np.array([0.2, 0.5]), np.array([90, 70]), np.array([90, 10]), np.array([10, 30])
        )
        selected = select_best(rows, "csi", min_precision=0.8)
        self.assertEqual(selected["threshold"], 0.5)

    def test_invalid_grid_is_rejected(self):
        with self.assertRaises(ValueError):
            threshold_grid(0.8, 0.2, 0.1)

    def test_csi_counts(self):
        row = scores_from_counts([0.5], np.array([6]), np.array([2]), np.array([2]))[0]
        self.assertAlmostEqual(row["precision"], 0.75)
        self.assertAlmostEqual(row["recall"], 0.75)
        self.assertAlmostEqual(row["csi"], 0.6)


if __name__ == "__main__":
    unittest.main()
