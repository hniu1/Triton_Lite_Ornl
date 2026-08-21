import unittest
from argparse import Namespace

import torch

from data_preprocessing.m4_build_stage1_sampling_index import classify
from stage1_train import depth_bin_weights, soft_dice_loss, speed_aware_component_losses


class Stage1LossTest(unittest.TestCase):
    def test_depth_bin_weights(self):
        args = Namespace(
            depth_weight_shallow=1.0,
            depth_weight_moderate=2.0,
            depth_weight_deep=3.0,
            depth_weight_extreme=4.0,
            depth_moderate_threshold=0.25,
            depth_deep_threshold=1.0,
            depth_extreme_threshold=2.0,
        )
        target = torch.tensor([0.10, 0.25, 1.0, 2.0])
        self.assertTrue(
            torch.equal(depth_bin_weights(target, args), torch.tensor([1.0, 2.0, 3.0, 4.0]))
        )

    def test_soft_dice_is_near_zero_for_perfect_prediction(self):
        logits = torch.tensor([[[-20.0, 20.0], [20.0, -20.0]]])
        target = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        mask = torch.ones_like(target)
        self.assertLess(float(soft_dice_loss(logits, target, mask, 1.0)), 1e-6)

    def test_strict_dynamic_categories(self):
        self.assertEqual(classify(0.0, 0.0, 0.10, 1.0, 0.10), "dry")
        self.assertEqual(classify(0.02, 3.0, 0.10, 1.0, 0.10), "boundary")
        self.assertEqual(classify(0.20, 0.9, 0.10, 1.0, 0.10), "wet")
        self.assertEqual(classify(0.20, 1.2, 0.10, 1.0, 0.10), "deep")

    def test_speed_aware_loss_is_zero_for_exact_vectors(self):
        args = Namespace(
            velocity_weight_reference_speed=0.25, velocity_weight_cap=3.0,
            velocity_weight_scale=2.0, component_huber_delta=0.25,
            direction_min_speed=0.05, speed_loss_weight=0.5,
            direction_loss_weight=0.1,
        )
        x = torch.tensor([[[0.1, 0.0]]])
        y = torch.tensor([[[0.0, 0.2]]])
        wet = torch.ones_like(x)
        losses = speed_aware_component_losses(x, y, x, y, wet, args)
        for loss in losses:
            self.assertLess(float(loss), 1e-6)

    def test_speed_aware_loss_has_finite_gradients_at_zero_speed(self):
        args = Namespace(
            velocity_weight_reference_speed=0.25, velocity_weight_cap=3.0,
            velocity_weight_scale=2.0, component_huber_delta=0.25,
            direction_min_speed=0.05, speed_loss_weight=0.5,
            direction_loss_weight=0.1,
        )
        x = torch.zeros((1, 2, 2), requires_grad=True)
        y = torch.zeros((1, 2, 2), requires_grad=True)
        target_x = torch.zeros_like(x)
        target_y = torch.zeros_like(y)
        wet = torch.ones_like(x)
        total, *_ = speed_aware_component_losses(x, y, target_x, target_y, wet, args)
        total.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(y.grad).all())


if __name__ == "__main__":
    unittest.main()
