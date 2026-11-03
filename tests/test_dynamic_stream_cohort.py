import unittest
import pandas as pd
import torch
from stage2.dynamic_stream_cohort import select
from stage2.direct import evaluate_direct_horizons
from stage2.model import Stage2DirectLeadModel
from test_stage2_graph_data import _DirectDataset


class ConditionalTest(unittest.TestCase):
    def test_joint_gates_support_and_target_selection(self):
        rule = dict(interval_hours=3, minimum_support_fraction=.01, minimum_target_index=49,
                    max_times_per_event=10, max_tiles_per_time=8,
                    thresholds=dict(event_depth_change_rms_m=.1, event_velocity_vector_change_rms_mps=.01,
                                    tile_depth_change_rms_m=.1, tile_velocity_vector_change_rms_mps=.01))
        frame = pd.DataFrame(dict(interval_hours=[3]*5, target_time_index=[49,49,50,50,48],
             block_index=[10,20,10,20,10], depth_change_rms_m=[.2]*5,
             velocity_vector_change_rms_mps=[.02,.001,.02,.02,.02], wet_union_cells=[100]*5,
             common_wet_velocity_cells=[100,100,100,1,100]))
        events = pd.DataFrame(dict(interval_hours=[3,3,3], origin_time_index=[43,44,42],
             depth_change_rms_m=[.2,.01,.2], velocity_vector_change_rms_mps=[.02]*3))
        eligible, batches = select(frame, events, {10:6400,20:6400}, rule)
        self.assertEqual(len(eligible), 1)
        self.assertEqual(batches, [dict(target_time_index=49, block_indices=[10])])
        with self.assertRaises(ValueError):
            select(frame.iloc[:0], events, {10:6400,20:6400}, rule)

    def test_same_target_different_origins_and_persistence(self):
        dataset = _DirectDataset()
        model = Stage2DirectLeadModel(event_features=2, block_features=1, static_channels=1,
            temporal_channels=4, temporal_layers=2, event_embedding_dim=6,
            conditioning_dim=8, base_channels=4, dropout=0, history_states=1, history_fusion="concat")
        result, count = evaluate_direct_horizons(model, dataset, torch.device('cpu'), [1,48],
            batch_size=2, batch_limit=1, start_time_stride=12, wet_threshold=.05,
            component_semantics='unit_discharge', target_batches=[(0,49,[0,1])],depth_diagnostics=True)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(result['1']['persistence']['depth_wet_rmse'], 1)
        self.assertAlmostEqual(result['48']['persistence']['depth_wet_rmse'], 48)
        self.assertEqual(result['1']['observed_origin_to_target_change']['target_wet_cells'], 128)
        diag=result['48']['depth_diagnostics']['persistence']['deep_rising_ge1m_change_ge0.1m']
        self.assertEqual(diag['cells'],128)
        self.assertEqual(diag['bias_m'],-48.)
        self.assertEqual(diag['rmse_m'],48.)


class EarlyTargetTest(unittest.TestCase):
    def test_early_target_is_scored_only_at_supported_leads(self):
        class TwoStateDataset(_DirectDataset):
            history_states = 2
            def __getitem__(self, key):
                item = super().__getitem__(key)
                item["older_component_x"] = torch.zeros(8,8)
                item["older_component_y"] = torch.zeros(8,8)
                return item
        dataset = TwoStateDataset()
        config = dict(event_features=2, block_features=1, static_channels=1,
            temporal_channels=4, temporal_layers=2, event_embedding_dim=6,
            conditioning_dim=8, base_channels=4, dropout=0, history_states=2, history_fusion="concat")
        model = Stage2DirectLeadModel(**config)
        options = dict(device=torch.device('cpu'), horizons=[1,5,6,48], batch_size=1,
            batch_limit=1, start_time_stride=12, wet_threshold=.05,
            component_semantics='unit_discharge', target_batches=[(0,6,[0]),(0,49,[1])],
            allow_partial_target_history=True, return_per_event=True,depth_diagnostics=True)
        results, count, per_event = evaluate_direct_horizons(model, dataset, **options)
        self.assertEqual(count,2)
        self.assertEqual(per_event['synthetic']['48']['depth_diagnostics']['persistence']['5_to_infm']['cells'],64)
        for lead, pairs in [(1,2),(5,2),(6,1),(48,1)]:
            self.assertEqual(results[str(lead)]['cohort_coverage']['tile_target_pairs'],pairs)
            self.assertAlmostEqual(results[str(lead)]['persistence']['depth_wet_rmse'],lead)
        # Graph context must preserve exactly the same scored centers at each lead.
        from stage2.graph_data import GraphNeighborhoodDataset, TileTopology
        graph = GraphNeighborhoodDataset(dataset, TileTopology([10,30,20,40],[(10,30),(30,20)]),2)
        graph_model = Stage2DirectLeadModel(**config,graph_layers=2)
        graph_model.load_state_dict(model.state_dict(),strict=False)
        graph_results, _, _ = evaluate_direct_horizons(graph_model,graph,**options)
        for lead in results:
            self.assertEqual(results[lead]['persistence'],graph_results[lead]['persistence'])
            self.assertEqual(results[lead]['cohort_coverage'],graph_results[lead]['cohort_coverage'])
        options['allow_partial_target_history'] = False
        with self.assertRaisesRegex(ValueError,'lacks history'):
            evaluate_direct_horizons(model,dataset,**options)


if __name__ == '__main__':
    unittest.main()
