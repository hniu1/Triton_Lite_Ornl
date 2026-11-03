import json
import unittest
from argparse import Namespace
from pathlib import Path
import torch
from stage2.losses import delta_aware_loss_terms
from stage2.depth_diagnostics import DepthDiagnostics


class DeepRisingTest(unittest.TestCase):
    def loss(self, prediction, target, origin, weight, mask=None):
        config=json.loads(Path('processed_data/stage2_corrected_training/20261009_fold0/source_config/run_config.json').read_text())
        config.update(depth_extent_only=True,deep_rising_loss_weight=weight,
                      deep_rising_depth_threshold=1.,deep_rising_change_threshold=.1)
        q=torch.zeros_like(prediction)
        batch=dict(depth=target,previous_depth=origin,true_previous_depth=origin,
                   mask=torch.ones_like(target) if mask is None else mask,
                   component_x=q,component_y=q,previous_component_x=q,previous_component_y=q)
        output=(prediction,q,q,q,prediction-origin,torch.stack([q,q],dim=1))
        return delta_aware_loss_terms(output,batch,Namespace(**config),torch.device('cpu'))

    def test_targeted_loss_excludes_shallow_stable_draining_and_invalid_cells(self):
        target=torch.tensor([[[2.,.5,2.,2.,3.]]])
        origin=torch.tensor([[[1.,.1,2.,3.,1.]]])
        pred=torch.tensor([[[1.,0.,1.,1.,0.]]],requires_grad=True)
        mask=torch.tensor([[[1.,1.,1.,1.,0.]]])
        old=self.loss(pred,target,origin,0.,mask)
        new=self.loss(pred,target,origin,1.,mask)
        self.assertAlmostEqual(float(new['deep_rising']),.5)
        self.assertAlmostEqual(float(new['loss']-old['loss']),.5,places=6)
        grad=torch.autograd.grad(new['deep_rising'],pred)[0]
        torch.testing.assert_close(grad,torch.tensor([[[-1.,0.,0.,0.,0.]]]))

    def test_extra_loss_penalizes_overprediction_and_empty_subset_is_zero(self):
        target=torch.tensor([[[2.]]]); origin=torch.tensor([[[1.]]])
        pred=torch.tensor([[[3.]]],requires_grad=True)
        term=self.loss(pred,target,origin,1.)['deep_rising']
        self.assertGreater(float(torch.autograd.grad(term,pred)[0]),0)
        empty=self.loss(pred,target,target,1.)['deep_rising']
        self.assertEqual(float(empty),0.)
        self.assertEqual(float(torch.autograd.grad(empty,pred)[0]),0.)

    def test_diagnostics_preserve_bias_sign_mask_and_empty_bins(self):
        acc=DepthDiagnostics()
        acc.update(torch.tensor([[[1.,4.,99.]]]),torch.tensor([[[2.,2.,2.]]]),
                   torch.tensor([[[1.,1.,1.]]]),torch.tensor([[[1.,1.,0.]]]))
        result=acc.finalize()['deep_rising_ge1m_change_ge0.1m']
        self.assertEqual(result['cells'],2)
        self.assertEqual(result['bias_m'],.5)
        self.assertEqual(result['mae_m'],1.5)
        self.assertEqual(result['underprediction_gt0_3m_fraction'],.5)
        self.assertIsNone(acc.finalize()['5_to_infm']['bias_m'])


if __name__=='__main__':unittest.main()
