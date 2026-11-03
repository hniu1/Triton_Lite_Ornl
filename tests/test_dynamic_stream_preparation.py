import unittest

import numpy as np
import pandas as pd

from stage2.prepare_dynamic_stream import accumulate_pixels, make_frames, new_accumulators
from stage2.summarize_dynamic_stream import candidate_coverage, quantile_summary


class DynamicStreamPreparationTest(unittest.TestCase):
    def test_direction_reversal_and_transitions_are_distinguished(self):
        # One persistently wet cell reverses velocity; one dry cell becomes wet.
        h = np.array([[1., 0.], [1., 0.2]], dtype=np.float32)
        qx = np.array([[1., 0.], [-1., 0.2]], dtype=np.float32)
        qy = np.zeros_like(h)
        state, changes = new_accumulators(1, 2, (1,))
        accumulate_pixels(state, changes, 0, h, qx, qy)
        c = changes[1]
        self.assertEqual(c['newly_wet_cells'][0, 0], 1)
        self.assertEqual(c['newly_dry_cells'][0, 0], 0)
        self.assertEqual(c['wet_union_cells'][0, 0], 2)
        self.assertEqual(c['common_wet_velocity_cells'][0, 0], 1)
        self.assertAlmostEqual(c['depth_sq_sum'][0, 0], 0.04)
        self.assertAlmostEqual(c['velocity_vector_sq_sum'][0, 0], 4)

    def test_missing_depth_and_shallow_velocity_are_excluded(self):
        h = np.array([[np.nan, 0.07, 1.], [0.3, 0.08, 1.]], dtype=np.float32)
        qx = np.ones_like(h)
        qy = np.zeros_like(h)
        state, changes = new_accumulators(1, 2, (1,))
        accumulate_pixels(state, changes, 0, h, qx, qy)
        self.assertEqual(state['valid_cells'][0, 0], 2)
        self.assertEqual(state['velocity_cells'][0, 0], 1)
        self.assertEqual(changes[1]['pair_cells'][0, 0], 2)
        self.assertEqual(changes[1]['common_wet_velocity_cells'][0, 0], 1)
        self.assertEqual(changes[1]['newly_wet_cells'][0, 0], 0)
        self.assertTrue(np.isfinite(changes[1]['depth_sq_sum']).all())

    def test_spatial_chunking_preserves_statistics(self):
        rng = np.random.default_rng(11)
        h = rng.uniform(0, 1, (16, 12)).astype(np.float32)
        qx = rng.normal(0, 0.1, h.shape).astype(np.float32)
        qy = rng.normal(0, 0.1, h.shape).astype(np.float32)
        whole, whole_changes = new_accumulators(1, 16)
        split, split_changes = new_accumulators(1, 16)
        accumulate_pixels(whole, whole_changes, 0, h, qx, qy)
        for cut in (slice(0, 5), slice(5, 12)):
            accumulate_pixels(split, split_changes, 0, h[:, cut], qx[:, cut], qy[:, cut])
        for key in whole:
            np.testing.assert_allclose(whole[key], split[key], rtol=1e-6)
        for lag in whole_changes:
            for key in whole_changes[lag]:
                np.testing.assert_allclose(whole_changes[lag][key], split_changes[lag][key], rtol=1e-6)

    def test_time_alignment_and_pixel_weighted_event_aggregation(self):
        state, changes = new_accumulators(2, 2, (1,))
        for tile, n_pixels, change in ((0, 1, 2.), (1, 3, 0.)):
            h = np.stack([np.ones(n_pixels), np.ones(n_pixels) + change]).astype(np.float32)
            accumulate_pixels(state, changes, tile, h, np.zeros_like(h), np.zeros_like(h))
        layout = dict(stream_tiles=[dict(block_index=10), dict(block_index=20)], step_hours=0.5)
        event = dict(time_start_hours=0.5, event_id='synthetic', split='train')
        states, local, event_states, event_changes = make_frames(layout, event, state, changes)
        self.assertEqual(states.time_hours.tolist(), [0.5, 1., 0.5, 1.])
        self.assertEqual(local.origin_time_index.tolist(), [0, 0])
        self.assertEqual(local.target_time_index.tolist(), [1, 1])
        self.assertEqual(local.interval_hours.tolist(), [0.5, 0.5])
        self.assertAlmostEqual(event_changes.depth_change_rms_m.iloc[0], 1.)
        self.assertEqual(event_states.valid_cells.tolist(), [4, 4])

    def test_coverage_thresholds_are_derived_from_training_only(self):
        frame = pd.DataFrame(dict(event_id=['train']*4 + ['val']*4,
            split=['train']*4 + ['val']*4, interval_hours=[0.5]*8,
            origin_time_index=[0]*8,
            depth_change_rms_m=[1., 2., 3., 4., 100., 200., 300., 400.],
            velocity_vector_change_rms_mps=[1., 2., 3., 4., 100., 200., 300., 400.]))
        coverage = candidate_coverage(frame)
        median = coverage[coverage.training_positive_percentile == 50]
        self.assertTrue((median.depth_threshold_m == 2.5).all())
        self.assertTrue((median.velocity_vector_threshold_mps == 2.5).all())
        self.assertEqual(median.loc[median.split == 'train', 'joint_change_pairs'].iloc[0], 2)
        self.assertEqual(median.loc[median.split == 'val', 'joint_change_pairs'].iloc[0], 4)

    def test_summary_keeps_unavailable_and_zero_change_counts(self):
        frame = pd.DataFrame(dict(split=['train']*3, interval_hours=[0.5]*3,
            depth_change_rms_m=[0., 2., np.nan], velocity_vector_change_rms_mps=[0., np.nan, np.nan]))
        rows = quantile_summary(frame).set_index('metric')
        self.assertEqual(rows.loc['depth_change_rms_m', 'unavailable_pairs'], 1)
        self.assertEqual(rows.loc['depth_change_rms_m', 'positive_pairs'], 1)
        self.assertEqual(rows.loc['depth_change_rms_m', 'all_eligible_p50'], 1.)
        self.assertTrue(np.isnan(rows.loc['velocity_vector_change_rms_mps', 'positive_p50']))


if __name__ == '__main__':
    unittest.main()
