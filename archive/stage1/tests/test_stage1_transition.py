import unittest
import json
from types import SimpleNamespace

import torch
import numpy as np
import pandas as pd

from stage1_data import SpatiotemporalBatchSampler
from stage1_transition_data import (
    Stage1TransitionDataset,
    Stage1TransitionSequenceDataset,
    TransitionBatchSampler,
)
from stage1_transition_model import (
    Stage1StateTransitionModel,
    load_timestamp_backbone,
    load_transition_checkpoint_compatible,
)
from stage1_transition_rollout import reconstruct_components, select_rollout_batches
from stage1_transition_compare import evaluate_acceptance
from stage1_transition_multistep_train import scheduled_state
from stage1_transition_regime_eval import classify_transition_regimes
from data_preprocessing.m5_build_transition_sampling_index import (
    classify_transition_groups,
)
from data_preprocessing.m6_build_paired_transition_index import (
    classify_regime as classify_local_transition_regime,
    patch_statistics,
)
from stage1_transition_sampling import (
    LocalTransitionAwareBatchSampler,
    TransitionAwareBatchSampler,
)
from stage1_transition_multistep_v2_train import (
    build_optimizer,
    delta_aware_loss_terms,
    operational_selection_score,
    validate_transition_regime_fractions,
    weighted_masked_huber,
    weighted_masked_mse,
)
from stage1_transition_operational_gate import evaluate_operational_gate
from stage1_transition_whole_area_compare import evaluate_whole_area_gate


class _Dataset:
    def __len__(self):
        return 10

    def __getitem__(self, key):
        _, time_index, _ = key
        value = torch.full((2, 2), float(time_index))
        return {
            "depth": value,
            "component_x": value + 1,
            "component_y": value + 2,
            "time_index": torch.tensor(time_index),
            "time_features": torch.full((4,), float(time_index)),
        }


class _Sampler:
    def __len__(self):
        return 3

    def __iter__(self):
        yield [(0, 0, 0)]
        yield [(0, 1, 0)]
        yield [(0, 2, 0)]


class TransitionTest(unittest.TestCase):
    def test_dataset_pairs_previous_and_target_states(self):
        sample = Stage1TransitionDataset(_Dataset(), lag=1)[(0, 3, 0)]
        self.assertTrue(torch.equal(sample["depth"], torch.full((2, 2), 3.0)))
        self.assertTrue(torch.equal(sample["previous_depth"], torch.full((2, 2), 2.0)))
        self.assertEqual(int(sample["previous_time_index"]), 2)

    def test_sequence_dataset_ends_at_target(self):
        sample = Stage1TransitionSequenceDataset(_Dataset(), steps=3)[(0, 5, 0)]
        self.assertEqual(sample["sequence_depth"].shape, (4, 2, 2))
        self.assertEqual(sample["sequence_depth"][:, 0, 0].tolist(), [2, 3, 4, 5])
        self.assertEqual(sample["sequence_time_index"].tolist(), [3, 4, 5])

    def test_sequence_dataset_can_include_two_history_states(self):
        sample = Stage1TransitionSequenceDataset(
            _Dataset(), steps=3, history_states=2
        )[(0, 5, 0)]
        self.assertEqual(sample["sequence_depth"].shape, (5, 2, 2))
        self.assertEqual(
            sample["sequence_depth"][:, 0, 0].tolist(), [1, 2, 3, 4, 5]
        )
        self.assertEqual(sample["sequence_time_index"].tolist(), [3, 4, 5])
        self.assertEqual(int(sample["sequence_history_states"]), 2)

    def test_two_state_sequence_rejects_one_state_boundary(self):
        dataset = Stage1TransitionSequenceDataset(
            _Dataset(), steps=3, history_states=2
        )
        with self.assertRaisesRegex(IndexError, "Target time 3"):
            dataset[(0, 3, 0)]

    def test_scheduled_state_probability_extremes(self):
        truth = torch.zeros(2, 2, 2)
        predicted = torch.ones(2, 2, 2)
        self.assertTrue(torch.equal(scheduled_state(predicted, truth, 0.0), truth))
        self.assertTrue(torch.equal(scheduled_state(predicted, truth, 1.0), predicted))
        random_values = torch.tensor([[[0.1]], [[0.9]]])
        mixed = scheduled_state(predicted, truth, 0.5, random_values)
        self.assertTrue(torch.equal(mixed[0], predicted[0]))
        self.assertTrue(torch.equal(mixed[1], truth[1]))

    def test_transition_regimes_distinguish_stable_fill_drain_and_rapid(self):
        previous = torch.tensor(
            [
                [[0.20, 0.20], [0.00, 0.00]],
                [[0.10, 0.10], [0.00, 0.00]],
                [[0.40, 0.40], [0.20, 0.20]],
                [[0.00, 0.00], [0.00, 0.00]],
            ]
        )
        target = torch.tensor(
            [
                [[0.205, 0.205], [0.00, 0.00]],
                [[0.30, 0.30], [0.20, 0.20]],
                [[0.10, 0.10], [0.00, 0.00]],
                [[0.20, 0.20], [0.20, 0.20]],
            ]
        )
        regimes, diagnostics = classify_transition_regimes(
            previous, target, torch.ones_like(previous), wet_threshold=0.05
        )
        self.assertEqual(regimes["stable"].tolist(), [True, False, False, False])
        self.assertEqual(regimes["filling"].tolist(), [False, True, False, True])
        self.assertEqual(regimes["draining"].tolist(), [False, False, True, False])
        self.assertEqual(regimes["rapid"].tolist(), [False, True, True, True])
        self.assertGreater(float(diagnostics["signed_depth_change"][1]), 0.0)
        self.assertLess(float(diagnostics["signed_depth_change"][2]), 0.0)

    def test_sampling_regimes_keep_rapid_fill_and_drain_distinct(self):
        regimes, direction, rapid = classify_transition_groups(
            wet_delta=np.array([0.001, 0.02, -0.02, 0.10, -0.10]),
            depth_delta=np.array([0.01, 0.06, -0.06, 0.30, -0.30]),
        )
        self.assertEqual(
            regimes.tolist(),
            ["stable", "filling", "draining", "rapid_filling", "rapid_draining"],
        )
        self.assertEqual(rapid.tolist(), [False, False, False, True, True])
        self.assertGreater(direction[3], 0)
        self.assertLess(direction[4], 0)

    def test_exact_paired_transition_statistics_and_regimes(self):
        patch = np.array([[0.0, 0.10], [0.20, 9.0]], dtype=np.float32)
        mask = np.array([[True, True], [True, False]])
        wet_fraction, mean_wet, p90_wet, max_depth, mean_cell = patch_statistics(
            patch, mask, wet_threshold=0.05
        )
        self.assertAlmostEqual(wet_fraction, 2.0 / 3.0)
        self.assertAlmostEqual(mean_wet, 0.15, places=6)
        self.assertAlmostEqual(p90_wet, 0.19, places=6)
        self.assertAlmostEqual(max_depth, 0.20, places=6)
        self.assertAlmostEqual(mean_cell, 0.10, places=6)

        labels, direction, activity = classify_local_transition_regime(
            extent_delta=np.array([0.001, 0.02, -0.02, 0.10, -0.10]),
            storage_delta=np.array([0.001, 0.02, -0.02, 0.10, -0.10]),
            stable_extent_threshold=0.01,
            stable_storage_threshold=0.01,
            rapid_extent_threshold=0.05,
            rapid_storage_threshold=0.05,
        )
        self.assertEqual(
            labels.tolist(),
            ["stable", "filling", "draining", "rapid_filling", "rapid_draining"],
        )
        self.assertGreater(direction[3], 0.0)
        self.assertLess(direction[4], 0.0)
        self.assertGreaterEqual(activity[3], 2.0)

    def test_local_transition_sampler_preserves_verified_anchor_and_quotas(self):
        categories = ["dry", "dry", "boundary", "boundary", "wet", "wet", "deep", "deep"]
        candidates = pd.DataFrame(
            {
                "event_id": ["E1"] * 8,
                "time_index": [6] * 8,
                "anchor_block": np.arange(8),
                "phase": ["rising"] * 8,
                "category": categories,
                "wet_fraction": [0.0, 0.0, 0.05, 0.08, 0.40, 0.35, 0.80, 0.70],
                "local_transition_regime": [
                    "stable", "stable", "stable", "stable",
                    "rapid_filling", "stable", "stable", "stable",
                ],
                "local_transition_activity": [0.0, 0.0, 0.1, 0.1, 4.0, 0.2, 0.1, 0.1],
            }
        )
        sampler = LocalTransitionAwareBatchSampler(
            candidates=candidates,
            event_ids=["E1"],
            n_times=[10],
            n_blocks=8,
            batch_size=4,
            batches_per_epoch=3,
            seed=9,
            category_fractions={name: 0.25 for name in ("dry", "boundary", "wet", "deep")},
            phase_fractions={"rising": 1.0},
            transition_regime_fractions={"rapid_filling": 1.0},
            target_wet_cell_fraction=0.1,
            strict_category_quotas=True,
        )
        category_by_block = dict(enumerate(categories))
        for batch in sampler:
            blocks = [block for _, _, block in batch]
            self.assertIn(4, blocks)
            self.assertEqual(
                sorted(category_by_block[block] for block in blocks),
                sorted(["dry", "boundary", "wet", "deep"]),
            )

        key = (0, 6)
        for regime in np.unique(sampler.local_regimes[key]):
            expected = np.asarray(
                [
                    index
                    for index in np.flatnonzero(sampler.local_regimes[key] == regime)
                    if sampler._maximum_wet_fraction_with_anchor(key, int(index))
                    + 1e-8
                    >= sampler.target_wet_cell_fraction
                ],
                dtype=np.int64,
            )
            self.assertTrue(
                np.array_equal(
                    sampler._eligible_anchor_indices(key, str(regime)), expected
                )
            )

    def test_masked_mse_matches_rmse_squared_objective(self):
        prediction = torch.tensor([[1.0, 2.0], [100.0, 100.0]])
        target = torch.zeros_like(prediction)
        mask = torch.ones_like(prediction)
        loss = weighted_masked_mse(
            prediction, target, mask, torch.tensor([True, False])
        )
        self.assertAlmostEqual(float(loss), 2.5)

    def test_v2_trainer_validates_transition_regime_fractions(self):
        fractions = validate_transition_regime_fractions(
            {"stable": 4.0, "rapid_filling": 6.0}
        )
        self.assertAlmostEqual(fractions["stable"], 0.4)
        self.assertAlmostEqual(fractions["rapid_filling"], 0.6)

    def test_velocity_persistence_reconstructs_unit_discharge(self):
        previous_depth = torch.tensor([[[0.20, 0.00]]])
        previous_x = torch.tensor([[[0.10, 5.00]]])
        previous_y = torch.tensor([[[-0.20, 5.00]]])
        predicted_depth = torch.tensor([[[0.40, 0.30]]])
        learned_x = torch.full_like(previous_x, 9.0)
        learned_y = torch.full_like(previous_y, 9.0)
        result_x, result_y = reconstruct_components(
            "velocity_persistence",
            previous_depth,
            previous_x,
            previous_y,
            predicted_depth,
            learned_x,
            learned_y,
            torch.ones_like(previous_depth),
            wet_threshold=0.05,
        )
        self.assertTrue(torch.allclose(result_x, torch.tensor([[[0.20, 0.00]]])))
        self.assertTrue(torch.allclose(result_y, torch.tensor([[[-0.40, 0.00]]])))

    def test_transition_sampler_can_target_one_change_regime(self):
        rows = []
        categories = ["dry", "boundary", "wet", "deep"]
        for time_index, regime in [(10, "stable"), (20, "rapid_draining")]:
            for block, category in enumerate(categories):
                rows.append(
                    {
                        "event_id": "D001",
                        "time_index": time_index,
                        "anchor_block": block,
                        "category": category,
                        "phase": "recession",
                        "wet_fraction": 0.8 if category != "dry" else 0.0,
                        "transition_regime": regime,
                    }
                )
        sampler = TransitionAwareBatchSampler(
            candidates=pd.DataFrame(rows),
            event_ids=["D001"],
            n_times=[30],
            n_blocks=4,
            batch_size=4,
            batches_per_epoch=5,
            seed=7,
            category_fractions={name: 0.25 for name in categories},
            phase_fractions={"recession": 1.0},
            transition_regime_fractions={"rapid_draining": 1.0},
            target_wet_cell_fraction=0.5,
            strict_category_quotas=True,
        )
        batches = list(sampler)
        self.assertEqual(len(batches), 5)
        self.assertTrue(all(batch[0][1] == 20 for batch in batches))

    def test_weighted_masked_huber_uses_only_selected_samples(self):
        prediction = torch.tensor([[[1.0]], [[3.0]]])
        target = torch.zeros_like(prediction)
        mask = torch.ones_like(prediction)
        first = weighted_masked_huber(
            prediction, target, mask, torch.tensor([True, False]), delta=1.0
        )
        second = weighted_masked_huber(
            prediction, target, mask, torch.tensor([False, True]), delta=1.0
        )
        empty = weighted_masked_huber(
            prediction, target, mask, torch.tensor([False, False]), delta=1.0
        )
        self.assertAlmostEqual(float(first), 0.5)
        self.assertAlmostEqual(float(second), 2.5)
        self.assertEqual(float(empty), 0.0)

    def test_delta_aware_loss_is_finite_and_backpropagates(self):
        batch_size, rows, cols = 2, 3, 3
        previous = torch.zeros(batch_size, rows, cols)
        target_depth = previous.clone()
        target_depth[1] = 0.3
        depth_delta = torch.full_like(previous, 0.02, requires_grad=True)
        component_delta = torch.full(
            (batch_size, 2, rows, cols), 0.01, requires_grad=True
        )
        depth = (previous + depth_delta).clamp_min(0)
        wet_logits = torch.zeros_like(previous, requires_grad=True)
        component_x = component_delta[:, 0]
        component_y = component_delta[:, 1]
        activity_logits = torch.zeros_like(previous, requires_grad=True)
        batch = {
            "mask": torch.ones_like(previous),
            "depth": target_depth,
            "component_x": torch.zeros_like(previous),
            "component_y": torch.zeros_like(previous),
            "previous_depth": previous,
            "previous_component_x": torch.zeros_like(previous),
            "previous_component_y": torch.zeros_like(previous),
            "true_previous_depth": previous,
            "true_previous_component_x": torch.zeros_like(previous),
            "true_previous_component_y": torch.zeros_like(previous),
        }
        args = SimpleNamespace(
            wet_threshold=0.05,
            depth_log_loss_weight=1.0,
            depth_physical_loss_weight=1.0,
            transition_depth_loss_weight=0.5,
            dry_depth_loss_weight=0.1,
            wet_loss_weight=0.2,
            wet_dice_loss_weight=0.15,
            wet_pos_weight=1.25,
            component_loss_weight=0.5,
            dry_component_loss_weight=0.05,
            depth_weight_shallow=1.0,
            depth_weight_moderate=2.0,
            depth_weight_deep=3.0,
            depth_weight_extreme=4.0,
            depth_moderate_threshold=0.25,
            depth_deep_threshold=1.0,
            depth_extreme_threshold=2.0,
            stable_depth_threshold=0.01,
            stable_extent_threshold=0.01,
            rapid_depth_threshold=0.10,
            rapid_extent_threshold=0.05,
            rapid_depth_delta_loss_weight=1.0,
            stable_depth_delta_loss_weight=0.5,
            component_delta_loss_weight=0.25,
            derived_velocity_loss_weight=0.10,
            storage_change_loss_weight=0.25,
            activity_gate_loss_weight=0.15,
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
                depth_delta,
                component_delta,
            ),
            batch,
            args,
            torch.device("cpu"),
        )
        self.assertTrue(all(torch.isfinite(value) for value in values.values()))
        values["loss"].backward()
        self.assertIsNotNone(depth_delta.grad)
        self.assertIsNotNone(component_delta.grad)
        self.assertIsNotNone(activity_logits.grad)
        self.assertIn("derived_velocity", values)
        self.assertIn("storage_change", values)
        self.assertGreater(float(values["activity_gate"]), 0.0)

    def test_operational_gate_rejects_stable_drift(self):
        method = {
            "depth_wet_rmse": 0.4,
            "component_rmse": 0.02,
            "derived_velocity_rmse": 0.02,
            "wet_f1": 0.9,
            "physical_score": 0.5,
        }
        persistence = dict(method, depth_wet_rmse=0.5, physical_score=0.6)
        rollout = {
            "metrics": {
                str(horizon): {
                    "autoregressive": dict(method),
                    "persistence": dict(persistence),
                }
                for horizon in (6, 12, 24)
            }
        }
        regimes = {
            "metrics": {
                regime: {
                    "model": dict(method),
                    "persistence": dict(persistence),
                }
                for regime in ("stable", "filling", "draining", "rapid")
            }
        }
        accepted = evaluate_operational_gate(
            rollout, rollout, regimes, regimes
        )
        self.assertTrue(accepted["accepted"])
        drifting = json.loads(json.dumps(regimes))
        drifting["metrics"]["stable"]["model"]["depth_wet_rmse"] = 0.8
        rejected = evaluate_operational_gate(
            rollout, rollout, drifting, regimes
        )
        self.assertFalse(rejected["accepted"])
        self.assertFalse(
            next(
                item for item in rejected["checks"]
                if item["name"] == "stable_depth_near_persistence"
            )["passed"]
        )

    def test_whole_area_gate_rejects_dynamic_persistence_failure(self):
        rows = []
        for time_index in (60, 140, 240, 360, 440):
            rows.append(
                {
                    "time_index": time_index,
                    "gated_depth_wet_rmse": 0.10,
                    "persistence_depth_wet_rmse": 0.20,
                    "gated_component_rmse": 0.01,
                    "persistence_component_rmse": 0.02,
                    "gated_wet_f1": 0.95,
                    "persistence_wet_f1": 0.94,
                    "gated_derived_velocity_rmse": 0.01,
                    "persistence_derived_velocity_rmse": 0.01,
                }
            )
        candidate = pd.DataFrame(rows)
        reference = candidate.copy()
        accepted = evaluate_whole_area_gate(candidate, reference)
        self.assertTrue(accepted["accepted"])
        failed = candidate.copy()
        failed.loc[failed.time_index == 140, "gated_depth_wet_rmse"] = 0.30
        rejected = evaluate_whole_area_gate(failed, reference)
        self.assertFalse(rejected["accepted"])
        self.assertFalse(
            next(
                item for item in rejected["checks"]
                if item["name"] == "t140_dynamic_depth_beats_persistence"
            )["passed"]
        )

    def test_sampler_excludes_targets_without_history(self):
        batches = list(TransitionBatchSampler(_Sampler(), lag=1))
        self.assertEqual(batches, [[(0, 1, 0)], [(0, 2, 0)]])

    def test_base_sampler_counts_batches_after_minimum_time_filter(self):
        sampler = SpatiotemporalBatchSampler(
            n_events=1, n_times=[5], n_blocks=2, batch_size=2,
            batches_per_epoch=1, time_stride=2, seed=1, shuffle=False,
            minimum_time_index=1,
        )
        batches = list(sampler)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0][1], 2)

    def test_rollout_selection_reserves_full_horizon(self):
        batches = select_rollout_batches(
            n_times=[10], n_blocks=5, batch_size=2, max_horizon=4,
            time_stride=2, batch_limit=0,
        )
        self.assertTrue(batches)
        self.assertTrue(all(start + 4 < 10 for _, start, _ in batches))
        self.assertEqual({start for _, start, _ in batches}, {0, 2, 4})
        self.assertTrue(all(len(blocks) <= 2 for _, _, blocks in batches))
        history_batches = select_rollout_batches(
            n_times=[10], n_blocks=2, batch_size=2, max_horizon=4,
            time_stride=2, batch_limit=0, minimum_start_time=1,
        )
        self.assertEqual({start for _, start, _ in history_batches}, {1, 3, 5})

    def test_acceptance_requires_persistence_and_reference_improvement(self):
        reference = {"test": {
            "depth_wet_rmse": 0.8, "component_rmse": 0.05,
            "wet_f1": 0.90, "physical_score": 1.0,
        }}
        candidate = {
            "test": {
                "depth_wet_rmse": 0.6, "component_rmse": 0.039,
                "wet_f1": 0.931, "physical_score": 0.75,
            },
            "persistence_test": {
                "depth_wet_rmse": 0.7, "component_rmse": 0.04,
                "wet_f1": 0.935, "physical_score": 0.85,
            },
        }
        self.assertTrue(evaluate_acceptance(candidate, reference)["accepted"])
        candidate["test"]["wet_f1"] = 0.90
        self.assertFalse(evaluate_acceptance(candidate, reference)["accepted"])

    def test_zero_initialized_residual_heads_start_at_persistence(self):
        model = Stage1StateTransitionModel(
            event_features=3, block_features=2, static_channels=2,
            temporal_channels=4, temporal_layers=2, event_embedding_dim=4,
            conditioning_dim=4, base_channels=4, dropout=0.0,
        ).eval()
        batch = 2
        previous_depth = torch.rand(batch, 8, 8)
        previous_x = torch.rand(batch, 8, 8)
        previous_y = torch.rand(batch, 8, 8)
        output = model(
            torch.rand(batch, 4, 3), torch.tensor([2, 2]), torch.rand(batch, 4),
            torch.rand(batch, 2), torch.rand(batch, 2, 8, 8), torch.ones(batch, 8, 8),
            previous_depth, previous_x, previous_y, shared_event_time=True,
        )
        self.assertTrue(torch.allclose(output[0], previous_depth))
        self.assertTrue(torch.allclose(output[2], previous_x))
        self.assertTrue(torch.allclose(output[3], previous_y))

    def test_two_state_activity_gate_suppresses_residual(self):
        model = Stage1StateTransitionModel(
            event_features=3, block_features=2, static_channels=2,
            temporal_channels=4, temporal_layers=2, event_embedding_dim=4,
            conditioning_dim=4, base_channels=4, dropout=0.0,
            history_states=2, use_activity_gate=True,
        ).eval()
        with torch.no_grad():
            model.depth_delta_head.bias.fill_(1.0)
            model.activity_head.bias.fill_(-20.0)
        batch = 2
        previous = torch.rand(batch, 8, 8)
        output = model(
            torch.rand(batch, 4, 3), torch.tensor([2, 2]), torch.rand(batch, 4),
            torch.rand(batch, 2), torch.rand(batch, 2, 8, 8),
            torch.ones(batch, 8, 8), previous, torch.zeros_like(previous),
            torch.zeros_like(previous), shared_event_time=True,
            older_depth=previous * 0.9,
            older_component_x=torch.zeros_like(previous),
            older_component_y=torch.zeros_like(previous),
        )
        self.assertEqual(len(output), 9)
        self.assertTrue(torch.allclose(output[0], previous, atol=1e-7))
        self.assertLess(float(torch.sigmoid(output[6]).max()), 1e-7)

    def test_checkpoint_loader_expands_state_encoder_with_zero_delta_channels(self):
        kwargs = dict(
            event_features=3, block_features=2, static_channels=2,
            temporal_channels=4, temporal_layers=2, event_embedding_dim=4,
            conditioning_dim=4, base_channels=4, dropout=0.0,
        )
        parent = Stage1StateTransitionModel(**kwargs)
        child = Stage1StateTransitionModel(
            **kwargs, history_states=2, use_activity_gate=True
        )
        checkpoint = {"model_state_dict": parent.state_dict()}
        _, adapted, _ = load_transition_checkpoint_compatible(child, checkpoint)
        self.assertTrue(adapted)
        parent_weight = next(
            value for name, value in parent.state_dict().items()
            if name.startswith("state_encoder") and value.ndim == 4
        )
        child_weight = next(
            value for name, value in child.state_dict().items()
            if name.startswith("state_encoder") and value.ndim == 4
        )
        self.assertTrue(torch.equal(child_weight[:, :3], parent_weight))
        self.assertEqual(int(torch.count_nonzero(child_weight[:, 3:])), 0)

    def test_zero_history_adapter_preserves_parent_prediction(self):
        kwargs = dict(
            event_features=3, block_features=2, static_channels=2,
            temporal_channels=4, temporal_layers=2, event_embedding_dim=4,
            conditioning_dim=4, base_channels=4, dropout=0.0,
        )
        parent = Stage1StateTransitionModel(**kwargs).eval()
        child = Stage1StateTransitionModel(
            **kwargs, history_states=2, history_fusion="adapter"
        ).eval()
        load_transition_checkpoint_compatible(
            child, {"model_state_dict": parent.state_dict()}
        )
        batch = 2
        inputs = dict(
            event=torch.rand(batch, 4, 3),
            time_index=torch.tensor([2, 2]),
            time_features=torch.rand(batch, 4),
            block_features=torch.rand(batch, 2),
            static=torch.rand(batch, 2, 8, 8),
            mask=torch.ones(batch, 8, 8),
            previous_depth=torch.rand(batch, 8, 8),
            previous_component_x=torch.rand(batch, 8, 8),
            previous_component_y=torch.rand(batch, 8, 8),
        )
        parent_output = parent(**inputs)
        child_output = child(
            **inputs,
            older_depth=torch.zeros(batch, 8, 8),
            older_component_x=torch.zeros(batch, 8, 8),
            older_component_y=torch.zeros(batch, 8, 8),
        )
        for expected, actual in zip(parent_output, child_output):
            self.assertTrue(torch.equal(expected, actual))

    def test_adaptation_parameters_receive_separate_learning_rate(self):
        model = Stage1StateTransitionModel(
            event_features=3, block_features=2, static_channels=2,
            temporal_channels=4, temporal_layers=2, event_embedding_dim=4,
            conditioning_dim=4, base_channels=4, dropout=0.0,
            history_states=2, history_fusion="adapter", use_activity_gate=True,
        )
        optimizer = build_optimizer(model, 2e-6, 5e-4, 1e-5)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [2e-6, 5e-4])
        adaptation_ids = {
            id(parameter)
            for group in optimizer.param_groups[1:]
            for parameter in group["params"]
        }
        expected_ids = {
            id(parameter) for name, parameter in model.named_parameters()
            if name.startswith(("history_adapter.", "activity_head."))
        }
        self.assertEqual(adaptation_ids, expected_ids)

    def test_operational_selection_score_includes_velocity(self):
        metrics = {
            "physical_score": 0.5,
            "derived_velocity_rmse": 0.2,
        }
        self.assertAlmostEqual(operational_selection_score(metrics, 2.0), 0.9)

    def test_timestamp_backbone_loading_skips_incompatible_heads(self):
        model = Stage1StateTransitionModel(
            event_features=3, block_features=2, static_channels=2,
            temporal_channels=4, temporal_layers=2, event_embedding_dim=4,
            conditioning_dim=4, base_channels=4, dropout=0.0,
        )
        checkpoint = {"model_state_dict": {
            "event_encoder.blocks.0.conv.weight": torch.ones_like(
                model.event_encoder.blocks[0].conv.weight
            ),
            "depth_head.weight": torch.ones(1, 4, 1, 1),
        }}
        loaded, skipped = load_timestamp_backbone(model, checkpoint)
        self.assertIn("event_encoder.blocks.0.conv.weight", loaded)
        self.assertIn("depth_head.weight", skipped)


if __name__ == "__main__":
    unittest.main()
