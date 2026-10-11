import unittest

import torch

from stage2.graph import TileGraphCommunication
from stage2.model import Stage2DirectLeadModel


class TileGraphTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.graph = TileGraphCommunication(4, edge_features=2)
        self.nodes = torch.randn(3, 4)
        self.edges = torch.tensor([[0, 1], [1, 0]])
        self.groups = torch.zeros(3, dtype=torch.long)
        self.attributes = torch.randn(2, 2)

    def test_neighbor_influence_is_local_and_differentiable(self):
        nodes = self.nodes.clone().requires_grad_()
        result = self.graph(nodes, self.edges, self.groups, self.attributes)
        altered = nodes.detach().clone()
        altered[0] += 3
        changed = self.graph(altered, self.edges, self.groups, self.attributes)
        self.assertFalse(torch.allclose(result[1], changed[1]))
        self.assertTrue(torch.equal(result[2], nodes[2]))
        self.assertTrue(torch.equal(result[2], changed[2]))
        result[1].sum().backward()
        self.assertGreater(float(nodes.grad[0].abs().sum()), 0)

    def test_node_relabeling_preserves_predictions(self):
        original = self.graph(self.nodes, self.edges, self.groups, self.attributes)
        order = torch.tensor([2, 0, 1])
        inverse = torch.argsort(order)
        relabeled = self.graph(self.nodes[order], inverse[self.edges],
                               self.groups[order], self.attributes)
        torch.testing.assert_close(relabeled, original[order])

    def test_invalid_groups_indices_and_attributes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cross"):
            self.graph(self.nodes, self.edges, torch.tensor([0, 1, 0]), self.attributes)
        with self.assertRaisesRegex(ValueError, "missing tile"):
            self.graph(self.nodes, torch.tensor([[0], [3]]), self.groups, torch.zeros(1, 2))
        with self.assertRaisesRegex(ValueError, "edge attributes"):
            self.graph(self.nodes, self.edges, self.groups)

    def test_empty_graph_is_identity(self):
        result = self.graph(self.nodes, torch.empty(2, 0, dtype=torch.long),
                            self.groups, torch.empty(0, 2))
        self.assertTrue(torch.equal(result, self.nodes))

    def test_direct_model_warm_start_and_graph_gradient(self):
        config = dict(event_features=5, block_features=3, static_channels=2,
                      temporal_channels=4, temporal_layers=2, event_embedding_dim=6,
                      conditioning_dim=8, base_channels=4, dropout=0.0,
                      history_states=1, history_fusion="concat")
        baseline = Stage2DirectLeadModel(**config).eval()
        graph_model = Stage2DirectLeadModel(**config, graph_layers=2).eval()
        incompatible = graph_model.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(all(key.startswith("graph_") for key in incompatible.missing_keys))
        inputs = dict(event=torch.zeros(2, 12, 5), target_time_index=torch.tensor([8, 8]),
                      target_time_features=torch.zeros(2, 4), origin_time_features=torch.zeros(2, 4),
                      lead_features=torch.zeros(2, 4), block_features=torch.zeros(2, 3),
                      static=torch.randn(2, 2, 8, 8), mask=torch.ones(2, 8, 8),
                      origin_depth=torch.ones(2, 8, 8), origin_component_x=torch.zeros(2, 8, 8),
                      origin_component_y=torch.zeros(2, 8, 8))
        graph_inputs = dict(graph_edge_index=self.edges, graph_group_ids=torch.zeros(2, dtype=torch.long))
        expected = baseline(**inputs)
        actual = graph_model(**inputs, **graph_inputs)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        # Wet head is nonzero at initialization, allowing the graph projection
        # to learn even while the residual depth/discharge heads start at zero.
        actual[1].sum().backward()
        self.assertGreater(float(graph_model.graph_projection.weight.grad.abs().sum()), 0)
        with torch.no_grad():
            graph_model.graph_projection.weight.fill_(0.1)
        graph_model.zero_grad()
        graph_model(**inputs, **graph_inputs)[1].sum().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in
                               graph_model.graph_communication.parameters() if p.grad is not None), 0)
        with self.assertRaisesRegex(ValueError, "Graph model requires"):
            graph_model(**inputs)
        with self.assertRaisesRegex(ValueError, "graph-disabled"):
            baseline(**inputs, **graph_inputs)
        inputs["target_time_index"] = torch.tensor([8, 9])
        with self.assertRaisesRegex(ValueError, "share origin"):
            graph_model(**inputs, **graph_inputs)


if __name__ == "__main__":
    unittest.main()
