import json
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from data_preprocessing.m7_build_routed_tile_forcing import upstream_hops,routed_weights
from stage2.data import RandomLeadTimeBatchSampler
from stage2.direct import depth_extent_metrics_only,horizon_selection_score
from stage2.losses import delta_aware_loss_terms


class FixedLocalTest(unittest.TestCase):
    def test_routing_excludes_downstream_and_unrelated_points(self):
        routing={'A':'B','B':'C','C':'outside','D':'outside'}
        ancestors=upstream_hops(routing)
        point_xy=np.array([[1,0],[2,0],[.1,0],[.2,0]])
        weights,audit=routed_weights(np.array([[0,0],[0,0]]),point_xy,['B',''],np.array(['A','B','C','D']),ancestors,k=8)
        self.assertAlmostEqual(float(weights[0,0]),.3,places=6)
        self.assertAlmostEqual(float(weights[0,1]),.7,places=6)
        self.assertEqual(float(weights[0,2]),0)
        self.assertEqual(float(weights[0,3]),0)
        self.assertEqual(audit[1]['mode'],'outside_basin_nearest_fallback')
        np.testing.assert_allclose(weights.sum(1),1)
        with self.assertRaises(ValueError):upstream_hops({'A':'B','B':'A'})

    def test_fixed_lead_sampler_never_samples_other_horizons(self):
        class Sampler:
            def __len__(self):return 3
            def __iter__(self):return iter([[(0,11,0)],[(0,100,1)],[(1,200,2)]])
        result=list(RandomLeadTimeBatchSampler(Sampler(),[10],45))
        self.assertEqual([b[0][3] for b in result],[10,10,10])
        self.assertEqual(result[0][0][1]-result[0][0][3],1)

    def test_balanced_multi_lead_counts_alignment_and_reproducibility(self):
        class Sampler:
            def __len__(self):return 750
            def __iter__(self):
                return iter([[(0,30,b) for b in range(8)] for _ in range(750)])
        from collections import Counter
        leads=[4,8,10,12,16,24]
        sampler=RandomLeadTimeBatchSampler(Sampler(),leads,45,balanced=True)
        first=list(sampler)
        self.assertEqual(Counter(b[0][3] for b in first),{h:125 for h in leads})
        self.assertTrue(all(len({k[3] for k in batch})==1 for batch in first))
        self.assertEqual(first,list(sampler))
        sampler.set_epoch(1)
        second=list(sampler)
        self.assertNotEqual(first,second)
        self.assertEqual(Counter(b[0][3] for b in second),{h:125 for h in leads})
        self.assertTrue(all(k[:3]==(0,30,i) for batch in second for i,k in enumerate(batch)))

    def test_depth_loss_never_uses_discharge_targets_or_predictions(self):
        config=json.loads(Path('processed_data/stage2_corrected_training/20261009_fold0/source_config/run_config.json').read_text())
        config['depth_extent_only']=True
        args=Namespace(**config)
        depth=torch.full((1,4,4),.2,requires_grad=True)
        wet=torch.zeros_like(depth,requires_grad=True)
        component=torch.full_like(depth,float('nan'),requires_grad=True)
        component_delta=torch.full((1,2,4,4),float('nan'),requires_grad=True)
        reference=torch.full_like(depth,.3)
        origin=torch.full_like(depth,.1)
        bad=torch.full_like(depth,float('nan'))
        batch=dict(depth=reference,previous_depth=origin,true_previous_depth=origin,mask=torch.ones_like(depth),
            component_x=bad,component_y=bad,previous_component_x=bad,previous_component_y=bad)
        output=(depth,wet,component,component,depth-origin,component_delta)
        result=delta_aware_loss_terms(output,batch,args,torch.device('cpu'))
        self.assertTrue(torch.isfinite(result['loss']))
        result['loss'].backward()
        self.assertIsNone(component.grad)
        self.assertIsNone(component_delta.grad)
        self.assertIsNotNone(depth.grad)
        self.assertIsNotNone(wet.grad)
        self.assertEqual(float(result['derived_velocity']),0)

    def test_checkpoint_score_has_no_velocity_or_component_term(self):
        metrics={'10':{'direct':dict(depth_wet_rmse=.2,wet_f1=.8,physical_score=1000,derived_velocity_rmse=1e9)}}
        self.assertAlmostEqual(horizon_selection_score(metrics,1,depth_extent_only=True),.3)
        filtered=depth_extent_metrics_only(metrics)
        self.assertNotIn('derived_velocity_rmse',filtered['10']['direct'])
        self.assertNotIn('physical_score',filtered['10']['direct'])
        self.assertAlmostEqual(filtered['10']['direct']['depth_extent_score'],.3)


if __name__=='__main__':unittest.main()
