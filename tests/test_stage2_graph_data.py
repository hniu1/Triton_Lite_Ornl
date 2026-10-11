import unittest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from stage2.graph_data import GraphNeighborhoodDataset, TileTopology, select_center_forecasts
from stage2.direct import evaluate_direct_horizons
from stage2.model import Stage2DirectLeadModel
from stage2.direct import predict_direct
from stage2.graph_learning_check import cohort_statistics, map_learning_gate


class _DirectDataset:
    block_rows = pd.DataFrame({"block_index": [10, 30, 20, 40]})
    max_horizon_steps = 48
    history_states = 1
    step_hours = 0.5
    events = pd.DataFrame({"event_id": ["synthetic"], "n_times": [64]})

    def __len__(self):
        return 100

    def __getitem__(self, key):
        event, target, block, lead = key
        return {"depth": torch.full((8, 8), float(target)),
                "previous_depth": torch.full((8, 8), float(target-lead)),
                "older_depth": torch.full((8, 8), float(target-lead-1)),
                "component_x": torch.zeros(8, 8), "component_y": torch.zeros(8, 8),
                "previous_component_x": torch.zeros(8, 8), "previous_component_y": torch.zeros(8, 8),
                "mask": torch.ones(8, 8), "static": torch.zeros(1, 8, 8),
                "time_features": torch.zeros(4), "origin_time_features": torch.zeros(4),
                "lead_features": torch.zeros(4),
                "event": torch.full((64, 2), float(event)),
                "time_index": torch.tensor(target),
                "lead_steps": torch.tensor(lead),
                "block_features": torch.tensor([float(block)])}


class GraphDataTest(unittest.TestCase):
    def test_raster_edges_exclude_diagonals_and_nodata(self):
        grid = np.array([[10, 10, -1, 40], [30, 20, -1, 40]])
        topology = TileTopology.from_raster(grid, [10, 30, 20, 40], chunk_rows=1)
        self.assertEqual(topology.neighbors, ((1, 2), (0, 2), (0, 1), ()))
        self.assertEqual(topology.metadata()["isolated_nodes"], 1)
        identical = TileTopology.from_raster(grid, [10, 30, 20, 40], chunk_rows=3)
        self.assertEqual(topology.digest, identical.digest)
        diagonal = TileTopology.from_raster(np.array([[10, -1], [-1, 30]]), [10, 30])
        self.assertEqual(diagonal.edge_index.shape, (2, 0))

    def test_invalid_raster_lookup_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            TileTopology.from_raster(np.array([[10, 30]]), [10])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            TileTopology([10, 10], [])

    def test_hop_expansion_and_truncation(self):
        topology = TileTopology([10, 30, 20, 40], [(10, 30), (30, 20), (20, 40)])
        nodes, edges, truncated = topology.neighborhood({0}, 2)
        self.assertEqual(nodes, [0, 1, 2])
        self.assertEqual(edges.shape, (2, 4))
        self.assertEqual(truncated.tolist(), [False, False, True])

    def test_loader_preserves_time_alignment_and_scores_only_centers(self):
        topology = TileTopology([10, 30, 20, 40], [(10, 30), (30, 20), (20, 40)])
        dataset = GraphNeighborhoodDataset(_DirectDataset(), topology, 2)
        sampler = [[(0, 50, 0, 6)], [(0, 55, 3, 12)]]
        batches = list(DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=0))
        batch = batches[0]
        self.assertEqual(batch["graph_block_positions"].tolist(), [0, 1, 2])
        self.assertEqual(batch["graph_center_mask"].tolist(), [True, False, False])
        self.assertTrue(torch.all(batch["previous_depth"] == 44))
        self.assertTrue(torch.all(batch["older_depth"] == 43))
        output = (batch["depth"], batch["depth"] + 1)
        central_output, central_batch = select_center_forecasts(output, batch)
        self.assertEqual(central_output[0].shape, (1, 8, 8))
        self.assertEqual(central_batch["block_features"].tolist(), [[0.0]])
        self.assertFalse(any(key.startswith("graph_") for key in central_batch))
        # Changing unused context labels cannot change the central target.
        batch["depth"][1:] = 999
        _, central_batch = select_center_forecasts(output, batch)
        self.assertTrue(torch.all(central_batch["depth"] == 50))
        with self.assertRaisesRegex(ValueError, "share event"):
            dataset.make_batch([(0, 50, 0, 6), (1, 50, 1, 6)])
        with self.assertRaisesRegex(ValueError, "share event"):
            dataset.make_batch([(0, 50, 0, 6), (0, 50, 1, 12)])

    def test_graph_evaluation_preserves_central_persistence_cohort(self):
        topology = TileTopology([10, 30, 20, 40], [(10, 30), (30, 20), (20, 40)])
        plain = _DirectDataset()
        graph = GraphNeighborhoodDataset(plain, topology, 2)
        config = dict(event_features=2, block_features=1, static_channels=1,
                      temporal_channels=4, temporal_layers=2, event_embedding_dim=6,
                      conditioning_dim=8, base_channels=4, dropout=0,
                      history_states=1, history_fusion="concat")
        baseline = Stage2DirectLeadModel(**config)
        candidate = Stage2DirectLeadModel(**config, graph_layers=2)
        candidate.load_state_dict(baseline.state_dict(), strict=False)
        options = dict(device=torch.device("cpu"), horizons=[1, 2], batch_size=1,
                       batch_limit=1, start_time_stride=12, wet_threshold=0.05,
                       component_semantics="unit_discharge")
        reference, count = evaluate_direct_horizons(baseline, plain, **options)
        actual, graph_count = evaluate_direct_horizons(candidate, graph, **options)
        self.assertEqual(count, graph_count)
        for lead in ("1", "2"):
            self.assertEqual(reference[lead]["persistence"], actual[lead]["persistence"])
            self.assertEqual(reference[lead]["direct"], actual[lead]["direct"])

    def test_capacity_control_has_no_inference_neighbor_or_target_dependence(self):
        torch.manual_seed(45)
        topology = TileTopology([10, 30, 20, 40], [(10, 30), (30, 20), (20, 40)])
        dataset = GraphNeighborhoodDataset(_DirectDataset(), topology, 2)
        batch = dataset.make_batch([(0, 50, 0, 6)])
        config = dict(event_features=2, block_features=1, static_channels=1,
                      temporal_channels=4, temporal_layers=2, event_embedding_dim=6,
                      conditioning_dim=8, base_channels=4, dropout=0,
                      history_states=1, history_fusion="concat", graph_layers=2)
        local = Stage2DirectLeadModel(**config, graph_mode="self").eval()
        graph = Stage2DirectLeadModel(**config).eval()
        graph.load_state_dict(local.state_dict())
        self.assertEqual(sum(p.numel() for p in local.parameters()),
                         sum(p.numel() for p in graph.parameters()))
        with torch.no_grad():
            local.graph_projection.weight.copy_(torch.eye(8))
            graph.graph_projection.weight.copy_(torch.eye(8))
            before_local = predict_direct(local, batch)[1][0].clone()
            before_graph = predict_direct(graph, batch)[1][0].clone()
            batch["previous_depth"][1:] += 10
            after_local = predict_direct(local, batch)[1][0]
            after_graph = predict_direct(graph, batch)[1][0]
            torch.testing.assert_close(before_local, after_local, rtol=0, atol=0)
            self.assertFalse(torch.equal(before_graph, after_graph))
            batch["depth"] += 999
            torch.testing.assert_close(predict_direct(graph, batch)[1][0], after_graph, rtol=0, atol=0)

    def test_learning_check_counts_only_active_central_labels(self):
        topology = TileTopology([10, 30, 20, 40], [(10, 30), (30, 20), (20, 40)])
        dataset = GraphNeighborhoodDataset(_DirectDataset(), topology, 2)
        batch = dataset.make_batch([(0, 50, 0, 6)])
        statistics = cohort_statistics(batch, 0.05)
        self.assertEqual(statistics["target_wet_cells"], 64)
        self.assertEqual(statistics["active_depth_change_mean"], 6)
        batch["depth"][0] = 0
        statistics = cohort_statistics(batch, 0.05)
        self.assertEqual(statistics["target_wet_cells"], 0)
        self.assertEqual(statistics["newly_dry_cells"], 64)

    def test_loss_reduction_does_not_override_worse_map_metrics(self):
        initial = {"loss": 10, "metrics": {"depth_wet_rmse": 1, "component_rmse": 0.1,
                                          "derived_velocity_rmse": 0.1, "wet_f1": 0.5}}
        worse = {"loss": 1, "metrics": {"depth_wet_rmse": 2, "component_rmse": 0.1,
                                       "derived_velocity_rmse": 0.1, "wet_f1": 0.4}}
        self.assertFalse(map_learning_gate(initial, worse))
        better = {"loss": 1, "metrics": {"depth_wet_rmse": 0.5, "component_rmse": 0.05,
                                        "derived_velocity_rmse": 0.05, "wet_f1": 0.8}}
        self.assertTrue(map_learning_gate(initial, better))


if __name__ == "__main__":
    unittest.main()
