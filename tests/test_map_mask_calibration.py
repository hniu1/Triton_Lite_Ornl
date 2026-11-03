import unittest
import numpy as np
from stage2.calibrate_map_mask import empty_hist, add_hist, table, select_candidate


class MapMaskCalibrationTest(unittest.TestCase):
    def test_histogram_matches_native_threshold_counts_at_boundaries(self):
        truth=np.array([[0,.05,.10],[.01,.25,np.nan]],dtype=np.float32)
        depth=np.array([[.05,.075,.10],[.15,.30,.50]],dtype=np.float32)
        probability=np.array([[0,.10,.50],[.95,.90,.99]],dtype=np.float32)
        valid=np.array([[True,True,True],[True,True,False]])
        hist=empty_hist();add_hist(hist,truth,depth,probability,valid)
        for row in table(hist):
            prediction=(depth>=np.float32(row['depth_threshold_m'])) & (probability>=np.float32(row['probability_threshold'])) & valid
            wet=(truth>=np.float32(.05)) & valid
            self.assertEqual(row['tp'],int((prediction&wet).sum()))
            self.assertEqual(row['fp'],int((prediction&~wet).sum()))
            self.assertEqual(row['fn'],int((~prediction&wet).sum()))
            self.assertEqual(row['valid_cells'],5)

    def test_choice_requires_recall_and_balances_events(self):
        baseline=dict(family='depth_only',recall=.99,min_event_recall=.96,macro_event_f1=.80,depth_threshold_m=.05,probability_threshold=0)
        aggressive=dict(baseline,recall=.89,min_event_recall=.70,macro_event_f1=.95,depth_threshold_m=.3)
        balanced=dict(baseline,recall=.96,min_event_recall=.92,macro_event_f1=.90,depth_threshold_m=.1)
        self.assertEqual(select_candidate([baseline,aggressive,balanced]),balanced)
        self.assertIsNone(select_candidate([aggressive]))
        unsafe_frame=dict(balanced,min_frame_recall=.8)
        self.assertIsNone(select_candidate([unsafe_frame]))
        self.assertIsNone(select_candidate([balanced],family='depth_and_wet_head'))

    def test_true_shallow_flooding_is_not_relabelled_dry(self):
        truth=np.array([.06],dtype=np.float32);depth=np.array([.06],dtype=np.float32);prob=np.ones(1,dtype=np.float32)
        hist=empty_hist();add_hist(hist,truth,depth,prob,np.ones(1,dtype=bool))
        high=next(r for r in table(hist) if r['depth_threshold_m']==.1 and r['probability_threshold']==0)
        self.assertEqual(high['fn'],1)
        self.assertEqual(high['tn'],0)


if __name__=='__main__':unittest.main()
