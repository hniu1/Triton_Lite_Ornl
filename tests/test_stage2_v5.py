import unittest
from types import SimpleNamespace

import torch

from stage2.data import (
    DirectHorizonDataset,
    RandomLeadTimeBatchSampler,
    lead_time_features,
)
from stage2.direct import select_forecast_batches
from stage2.losses import delta_aware_loss_terms
from stage2.model import Stage2DirectLeadModel, load_v4b_initialization


class _BaseDataset:
    def __len__(self):
        return 20

    def __getitem__(self, key):
        _, time_index, block = (int(value) for value in key)
        value = torch.full((8, 8), float(time_index))
        return {
            "event": torch.zeros(12, 5),
            "time_index": torch.tensor(time_index),
            "time_features": torch.full((4,), float(time_index)),
            "block_features": torch.full((3,), float(block)),
            "static": torch.zeros(2, 8, 8),
            "mask": torch.ones(8, 8),
            "depth": value,
            "component_x": value + 1.0,
            "component_y": value + 2.0,
        }


class _BatchSampler:
    def __init__(self):
        self.epoch = None

    def __len__(self):
        return 2

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        yield [(0, 10, 0), (0, 10, 1)]
        yield [(0, 11, 2), (0, 11, 3)]


def _model_config():
    return {
        "event_features": 5,
        "block_features": 3,
        "static_channels": 2,
        "temporal_channels": 4,
        "temporal_layers": 2,
        "event_embedding_dim": 6,
        "conditioning_dim": 8,
        "base_channels": 4,
        "dropout": 0.0,
        "history_states": 2,
        "history_fusion": "adapter",
        "use_activity_gate": True,
        "activity_gate_initial_bias": -1.5,
    }


class Stage2V5Test(unittest.TestCase):
    def test_forecast_selection_reserves_the_full_horizon(self):
        batches = select_forecast_batches(
            n_times=[10],
            n_blocks=3,
            batch_size=2,
            max_horizon=4,
            time_stride=1,
            batch_limit=0,
            minimum_start_time=1,
        )
        self.assertEqual({start for _, start, _ in batches}, {1, 2, 3, 4, 5})

    def test_lead_features_are_bounded_and_horizon_specific(self):
        short = lead_time_features(1, 48)
        long = lead_time_features(48, 48)
        self.assertEqual(short.shape, (4,))
        self.assertAlmostEqual(float(long[0]), 1.0)
        self.assertFalse(torch.equal(short, long))

    def test_direct_dataset_reads_origin_target_and_older_only(self):
        dataset = DirectHorizonDataset(
            _BaseDataset(), max_horizon_steps=8, history_states=2
        )
        sample = dataset[(0, 10, 2, 4)]
        self.assertEqual(int(sample["time_index"]), 10)
        self.assertEqual(int(sample["origin_time_index"]), 6)
        self.assertEqual(int(sample["older_time_index"]), 5)
        self.assertTrue(
            torch.equal(sample["previous_depth"], torch.full((8, 8), 6.0))
        )
        self.assertTrue(torch.equal(sample["depth"], torch.full((8, 8), 10.0)))
        self.assertEqual(float(sample["lead_hours"]), 2.0)

    def test_random_lead_sampler_uses_one_lead_per_spatial_batch(self):
        base = _BatchSampler()
        sampler = RandomLeadTimeBatchSampler(base, horizons=[1, 3, 8], seed=12)
        sampler.set_epoch(4)
        batches = list(sampler)
        self.assertEqual(base.epoch, 4)
        for batch in batches:
            self.assertEqual(len({key[3] for key in batch}), 1)
            self.assertIn(batch[0][3], {1, 3, 8})

    def test_direct_model_returns_expected_fields_and_shapes(self):
        model = Stage2DirectLeadModel(**_model_config())
        batch = 2
        result = model(
            event=torch.zeros(batch, 12, 5),
            target_time_index=torch.tensor([8, 8]),
            target_time_features=torch.zeros(batch, 4),
            origin_time_features=torch.zeros(batch, 4),
            lead_features=torch.zeros(batch, 4),
            block_features=torch.zeros(batch, 3),
            static=torch.zeros(batch, 2, 8, 8),
            mask=torch.ones(batch, 8, 8),
            origin_depth=torch.ones(batch, 8, 8),
            origin_component_x=torch.zeros(batch, 8, 8),
            origin_component_y=torch.zeros(batch, 8, 8),
            older_depth=torch.ones(batch, 8, 8),
            older_component_x=torch.zeros(batch, 8, 8),
            older_component_y=torch.zeros(batch, 8, 8),
        )
        self.assertEqual(len(result), 9)
        self.assertEqual(result[0].shape, (batch, 8, 8))
        self.assertEqual(result[5].shape, (batch, 2, 8, 8))

    def test_v4b_initialization_preserves_condition_prefix(self):
        config = _model_config()
        direct = Stage2DirectLeadModel(**config)
        source_state = {
            name: value.clone()
            for name, value in direct.state_dict().items()
            if not name.startswith(("origin_time_encoder.", "lead_encoder."))
        }
        condition = source_state["conditioning.0.weight"]
        source_state["conditioning.0.weight"] = condition[:, :-64].clone()
        checkpoint = {"model_state_dict": source_state}
        loaded, adapted, skipped = load_v4b_initialization(direct, checkpoint)
        source = source_state["conditioning.0.weight"]
        target = direct.state_dict()["conditioning.0.weight"]
        self.assertIn("conditioning.0.weight", adapted)
        self.assertTrue(torch.equal(target[:, : source.shape[1]], source))
        self.assertTrue(torch.equal(target[:, source.shape[1] :], torch.zeros_like(target[:, source.shape[1] :])))
        self.assertIn("wet_head.weight", loaded)
        self.assertEqual(skipped, [])

    def test_direct_loss_is_finite_and_differentiable(self):
        shape = (2, 2, 2)
        depth = torch.full(shape, 0.2, requires_grad=True)
        wet_logits = torch.zeros(shape, requires_grad=True)
        component_x = torch.full(shape, 0.02, requires_grad=True)
        component_y = torch.full(shape, -0.01, requires_grad=True)
        depth_delta = torch.full(shape, 0.1, requires_grad=True)
        component_delta = torch.zeros(2, 2, 2, 2, requires_grad=True)
        activity_logits = torch.zeros(shape, requires_grad=True)
        batch = {
            "mask": torch.ones(shape),
            "depth": torch.full(shape, 0.25),
            "component_x": torch.full(shape, 0.025),
            "component_y": torch.full(shape, -0.0125),
            "previous_depth": torch.full(shape, 0.1),
            "previous_component_x": torch.full(shape, 0.01),
            "previous_component_y": torch.full(shape, -0.005),
            "true_previous_depth": torch.full(shape, 0.1),
        }
        args = SimpleNamespace(
            wet_threshold=0.05,
            depth_weight_shallow=1.0,
            depth_weight_moderate=2.0,
            depth_weight_deep=3.0,
            depth_weight_extreme=4.0,
            depth_moderate_threshold=0.25,
            depth_deep_threshold=1.0,
            depth_extreme_threshold=2.0,
            depth_log_loss_weight=1.0,
            depth_physical_loss_weight=1.0,
            transition_depth_loss_weight=0.5,
            dry_depth_loss_weight=0.1,
            wet_loss_weight=0.2,
            wet_dice_loss_weight=0.15,
            wet_pos_weight=1.25,
            component_loss_weight=0.5,
            dry_component_loss_weight=0.05,
            stable_depth_threshold=0.01,
            stable_extent_threshold=0.01,
            rapid_depth_threshold=0.1,
            rapid_extent_threshold=0.05,
            rapid_depth_delta_loss_weight=1.0,
            stable_depth_delta_loss_weight=0.75,
            component_delta_loss_weight=0.5,
            derived_velocity_loss_weight=1.0,
            derived_velocity_loss_type="mse",
            derived_velocity_huber_delta=0.25,
            storage_change_loss_weight=0.5,
            activity_gate_loss_weight=0.02,
        )
        values = delta_aware_loss_terms(
            (
                depth,
                wet_logits,
                component_x,
                component_y,
                depth_delta,
                component_delta,
                activity_logits,
            ),
            batch,
            args,
            torch.device("cpu"),
        )
        self.assertTrue(torch.isfinite(values["loss"]))
        values["loss"].backward()
        self.assertIsNotNone(depth.grad)
        self.assertIsNotNone(component_delta.grad)


if __name__ == "__main__":
    unittest.main()
