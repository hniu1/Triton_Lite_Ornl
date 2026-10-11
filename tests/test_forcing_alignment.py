import unittest
import numpy as np
from stage2.audit_forcing_alignment import config_number, reconstruct_forcing


class ForcingAlignmentTest(unittest.TestCase):
    def test_interpolation_preserves_sensor_order_and_relative_time(self):
        raw = np.empty((3, 301))
        raw[:, 0] = [0, 3, 6]
        raw[:, 1:] = np.arange(300)[None, :] + raw[:, :1]*2
        times = np.array([.5, 1, 3, 6])
        actual = reconstruct_forcing(raw, times)
        np.testing.assert_array_equal(actual, np.arange(300)[None, :] + times[:, None]*2)
        self.assertEqual(actual.dtype, np.float32)

    def test_invalid_times_and_out_of_support_fail(self):
        raw = np.zeros((3, 301))
        raw[:, 0] = [0, 3, 6]
        with self.assertRaisesRegex(ValueError, "beyond"):
            reconstruct_forcing(raw, np.array([.5, 7]))
        raw[:, 0] = [0, 3, 3]
        with self.assertRaisesRegex(ValueError, "strictly"):
            reconstruct_forcing(raw, np.array([.5, 3]))

    def test_configuration_time_is_read_not_inferred(self):
        self.assertEqual(config_number("# sim_start_time=0\nsim_start_time=432000.0 # checkpoint\n", "sim_start_time"), 432000)
        with self.assertRaisesRegex(ValueError, "Missing"):
            config_number("checkpoint_id=240", "sim_start_time")
