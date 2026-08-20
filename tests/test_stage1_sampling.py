import unittest

import pandas as pd

from stage1_data import BalancedLabelBatchSampler, LabelAwareBatchSampler


class LabelAwareSamplerTest(unittest.TestCase):
    def test_batches_share_event_time_and_contain_anchor(self):
        candidates = pd.DataFrame(
            [
                {"event_id": "D001", "time_index": 5, "anchor_block": 4, "category": "dry", "phase": "quiet"},
                {"event_id": "D001", "time_index": 20, "anchor_block": 20, "category": "wet", "phase": "peak"},
                {"event_id": "D002", "time_index": 25, "anchor_block": 30, "category": "deep", "phase": "recession"},
                {"event_id": "D002", "time_index": 10, "anchor_block": 12, "category": "boundary", "phase": "rising"},
            ]
        )
        sampler = LabelAwareBatchSampler(
            candidates=candidates,
            event_ids=["D001", "D002"],
            n_times=[40, 40],
            n_blocks=50,
            batch_size=8,
            batches_per_epoch=25,
            seed=42,
            category_fractions={"dry": 0.15, "boundary": 0.25, "wet": 0.40, "deep": 0.20},
            phase_fractions={"quiet": 0.15, "rising": 0.30, "peak": 0.30, "recession": 0.25},
        )
        batches = list(sampler)
        self.assertEqual(len(batches), 25)
        for batch in batches:
            self.assertEqual(len(batch), 8)
            self.assertEqual(len({(key[0], key[1]) for key in batch}), 1)
            self.assertEqual([key[2] for key in batch], list(range(batch[0][2], batch[0][2] + 8)))

    def test_balanced_batches_are_unique_and_reach_wet_target(self):
        rows = []
        specifications = [
            ("dry", 0.0, 8),
            ("boundary", 0.05, 8),
            ("wet", 0.25, 8),
            ("deep", 0.50, 8),
        ]
        block = 0
        wet_by_block = {}
        for category, wet_fraction, count in specifications:
            for _ in range(count):
                rows.append(
                    {
                        "event_id": "D001",
                        "time_index": 20,
                        "anchor_block": block,
                        "category": category,
                        "phase": "peak",
                        "wet_fraction": wet_fraction,
                    }
                )
                wet_by_block[block] = wet_fraction
                block += 1
        sampler = BalancedLabelBatchSampler(
            candidates=pd.DataFrame(rows),
            event_ids=["D001"],
            n_times=[40],
            n_blocks=32,
            batch_size=16,
            batches_per_epoch=10,
            seed=42,
            category_fractions={"dry": 0.125, "boundary": 0.25, "wet": 0.375, "deep": 0.25},
            phase_fractions={"peak": 1.0},
            target_wet_cell_fraction=0.15,
        )
        for batch in sampler:
            blocks = [key[2] for key in batch]
            self.assertEqual(len(blocks), 16)
            self.assertEqual(len(set(blocks)), 16)
            self.assertEqual({(key[0], key[1]) for key in batch}, {(0, 20)})
            self.assertGreaterEqual(sum(wet_by_block[value] for value in blocks) / 16, 0.15)


if __name__ == "__main__":
    unittest.main()
