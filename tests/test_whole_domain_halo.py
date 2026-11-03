import unittest
from types import SimpleNamespace
import numpy as np
from stage2.whole_domain_halo import verify_orientation, stitch, FrameVariable
from stage2.plot_whole_domain_halo import max_pool, masked_prediction, signed_error_pool


class WholeDomainTest(unittest.TestCase):
    def test_north_up_axes_match_transform(self):
        metadata=dict(nc_crs='EPSG:26916',transform=[10,0,100,0,-10,200])
        report=verify_orientation(np.array([105,115,125]),np.array([195,185]),metadata)
        self.assertEqual(report['extent_m'],[100,130,180,200])
        self.assertEqual(report['origin'],'upper')
        with self.assertRaises(ValueError):
            verify_orientation(np.array([105,115,125]),np.array([185,195]),metadata)
        with self.assertRaises(ValueError):
            verify_orientation(np.array([115,125,135]),np.array([195,185]),metadata)

    def test_tile_placement_preserves_cardinal_corners_and_unpads(self):
        labels=np.array([[10,10,20],[10,10,20],[30,30,30]])
        grid=np.full((3,3),np.nan)
        window=SimpleNamespace(row_start=0,row_stop=2,col_start=0,col_stop=2,height=2,width=2)
        patch=np.zeros((4,4));patch[1:3,1:3]=[[1,2],[3,4]]
        stitch(grid,patch,labels,window,10)
        np.testing.assert_equal(grid[:2,:2],[[1,2],[3,4]])
        self.assertTrue(np.isnan(grid[2,0]))
        self.assertTrue(np.isnan(grid[0,2]))

    def test_cached_frames_retain_signed_values_and_reject_future_reads(self):
        raw=np.arange(3*4*5,dtype=np.float32).reshape(3,4,5)-20
        cached=FrameVariable(raw);cached.set_times([0,1])
        np.testing.assert_equal(cached[0,slice(1,3),slice(2,4)],raw[0,1:3,2:4])
        with self.assertRaises(ValueError):cached[2,slice(None),slice(None)]
        cached.set_times([1,2]);self.assertNotIn(0,cached.frames)

    def test_max_pool_retains_small_channels_and_edge_cells(self):
        grid=np.zeros((5,3));grid[1,1]=3;grid[4,2]=4;grid[0,2]=np.nan
        np.testing.assert_equal(max_pool(grid,2),[[3,0],[0,0],[0,4]])


class MaskedDisplayTest(unittest.TestCase):
    def test_probability_mask_precedes_display_pooling(self):
        depth=np.array([[5,.10],[.07,.01]],dtype=np.float32)
        probability=np.array([[.1,.9],[.9,.9]],dtype=np.float32)
        displayed=max_pool(masked_prediction(depth,probability,.075,.5),2)
        self.assertAlmostEqual(float(displayed[0,0]),.10)
        self.assertEqual(float(depth[0,0]),5)

    def test_raw_error_pool_keeps_largest_magnitude_and_sign(self):
        error=np.array([[.2,-.5,.8],[-.1,.1,np.nan]])
        np.testing.assert_equal(signed_error_pool(error,2),[[-.5,.8]])


if __name__=='__main__':unittest.main()
