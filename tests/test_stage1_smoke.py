import unittest

import torch

from stage1_model import Stage1TimestampModel


class Stage1ModelSmokeTest(unittest.TestCase):
    def test_shapes_and_causality(self):
        torch.manual_seed(7)
        model = Stage1TimestampModel(
            event_features=5,
            block_features=3,
            static_channels=2,
            temporal_channels=8,
            temporal_layers=4,
            event_embedding_dim=8,
            conditioning_dim=8,
            base_channels=4,
            dropout=0.0,
        )
        model.train()
        event = torch.randn(2, 12, 5)
        time_index = torch.tensor([4, 7])
        time_features = torch.randn(2, 4)
        block_features = torch.randn(2, 3)
        static = torch.randn(2, 2, 80, 80)
        mask = torch.ones(2, 80, 80)
        outputs = model(event, time_index, time_features, block_features, static, mask)
        self.assertEqual([tuple(value.shape) for value in outputs], [(2, 80, 80)] * 4)

        changed = event.clone()
        changed[0, 5:] += 100.0
        with torch.no_grad():
            first = model(event, time_index, time_features, block_features, static, mask)[0]
            second = model(changed, time_index, time_features, block_features, static, mask)[0]
        self.assertTrue(torch.allclose(first[0], second[0], atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
