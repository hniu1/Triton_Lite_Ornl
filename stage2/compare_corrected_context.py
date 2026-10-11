"""Write a baseline/halo/halo+graph comparison after all evaluations finish."""

import argparse
import csv
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--halo', type=Path, required=True)
    parser.add_argument('--halo-graph', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    runs = dict(baseline=args.baseline, halo=args.halo, halo_graph=args.halo_graph)
    evaluations, configs = {}, {}
    for arm, directory in runs.items():
        evaluations[arm] = json.loads((directory / 'test_dense_24h_direct_metrics.json').read_text())
        configs[arm] = json.loads((directory / 'run_config.json').read_text())
        progress = json.loads((directory / 'training_progress.json').read_text())
        assert progress['completed_epochs'] == progress['requested_epochs'] == 10
        assert configs[arm]['initial_checkpoint'] is None
    reference = evaluations['baseline']
    for arm, result in evaluations.items():
        for key in ('split', 'split_events', 'forecast_mode', 'forcing_availability',
                    'horizons_steps', 'step_hours', 'cohort_batches', 'batch_size',
                    'start_time_stride', 'wet_probability_threshold',
                    'wet_classification_mode', 'component_policy'):
            assert result[key] == reference[key], (arm, key)
        for key in ('manifest_dir', 'sampling_index_dir', 'split_events', 'seed',
                    'data_split_seed', 'epochs', 'batch_size', 'train_batches_per_epoch',
                    'selection_horizons_steps', 'selection_batches',
                    'backbone_learning_rate', 'new_conditioning_learning_rate'):
            assert configs[arm][key] == configs['baseline'][key], (arm, key)
        for horizon, metrics in reference['metrics'].items():
            for key, value in metrics['persistence'].items():
                assert math.isclose(value, result['metrics'][horizon]['persistence'][key],
                                    rel_tol=1e-9, abs_tol=1e-12), (arm, horizon, key)
    rows = []
    for horizon, metrics in reference['metrics'].items():
        for arm in ('persistence', 'baseline', 'halo', 'halo_graph'):
            m = metrics['persistence'] if arm == 'persistence' else evaluations[arm]['metrics'][horizon]['direct']
            rows.append(dict(lead_hours=metrics['lead_hours'], model=arm,
                             **{key: m[key] for key in ('depth_wet_rmse', 'wet_f1',
                                'component_rmse', 'derived_velocity_rmse', 'physical_score')}))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Corrected-data context comparison', '',
             'Fold 0, seed 45; validation-selected checkpoints; sampled held-out evaluation.',
             'Graph adds parameters and neighborhood computation, so this comparison does not isolate capacity from communication.', '',
             '| Lead (h) | Model | Wet-depth RMSE (m) | Wet F1 | HU/HV RMSE (m²/s) | Derived velocity RMSE (m/s) |',
             '|---|---|---|---|---|---|']
    for row in rows:
        if row['lead_hours'] in (0.5, 3, 6, 12, 24):
            lines.append(f"| {row['lead_hours']:g} | {row['model']} | {row['depth_wet_rmse']:.6f} | {row['wet_f1']:.6f} | {row['component_rmse']:.6f} | {row['derived_velocity_rmse']:.6f} |")
    (args.output_dir / 'COMPARISON.md').write_text('\n'.join(lines) + '\n')
    print(f'Wrote comparison to {args.output_dir}')


if __name__ == '__main__':
    main()
