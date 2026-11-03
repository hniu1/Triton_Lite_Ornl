"""Report a matched B/B1 validation comparison without inspecting test events."""
import argparse
import json
from pathlib import Path


def run(root):
    values={name:json.loads((root/'evaluation'/f'{name}_validation.json').read_text()) for name in ('B','B1')}
    a,b=values['B'],values['B1']
    for key in ('split','split_events','horizons_steps','cohort_batches','batch_size',
                'start_time_stride','wet_classification_mode','wet_probability_threshold'):
        if a[key]!=b[key]:raise ValueError(f'Unmatched validation configuration: {key}')
    assert a['split']=='val'
    av,bv=a['metrics']['10'],b['metrics']['10']
    assert av['persistence']==bv['persistence']
    pairs=sum(v['10']['tile_target_pairs'] for v in a['per_event_metrics'].values())
    rows=['# B1 deep-rising validation comparison','',
          f"Fixed direct 5 h lead; {a['cohort_batches']} matched batches, {pairs} tile/target pairs across {len(a['split_events'])} validation events. Raw depth-defined wet extent at 0.05 m. No test events evaluated.",
          '', '| Model | Wet MAE (m) | Wet RMSE (m) | Raw extent F1 |',
          '|---|---:|---:|---:|']
    for name,v in [('Persistence',av['persistence']),('B',av['direct']),('B1',bv['direct'])]:
        rows.append(f"| {name} | {v['depth_wet_mae']:.4f} | {v['depth_wet_rmse']:.4f} | {v['wet_f1']:.4f} |")
    rows+=['','| True-depth group | Cells | B bias (m) | B1 bias (m) | B MAE (m) | B1 MAE (m) | B RMSE (m) | B1 RMSE (m) |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return 'n/a' if x is None else f'{x:.4f}'
    for group,x in av['depth_diagnostics']['direct'].items():
        y=bv['depth_diagnostics']['direct'][group]
        assert x['cells']==y['cells']
        cells=' | '.join(fmt(z) for z in [x['bias_m'],y['bias_m'],x['mae_m'],y['mae_m'],x['rmse_m'],y['rmse_m']])
        rows.append(f"| {group} | {x['cells']} | {cells} |")
    rows+=['','Interpret bias together with MAE/RMSE: a mean closer to zero can hide opposing errors. Inspect per-event diagnostics and extent precision/recall before deciding whether B1 improved.',
           'The extra loss is symmetric physical-depth Huber on reference depth >=1 m and reference 5 h rise >=0.1 m; weight 1. This is one prespecified experiment, not an optimized setting.',
           'Checkpoint selection remains the original 80-batch validation depth/extent score. This larger comparison is also validation, not independent test confirmation.','']
    (root/'evaluation'/'REPORT.md').write_text('\n'.join(rows))
    print('\n'.join(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
