"""Compare fixed-lead B with persistence and the earlier Halo reference."""
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT=Path('results/stage2_fixed5h_local/20261022_fold0')


def main():
    # Head-based extent can look good while raw predicted depth stays too wide.
    for arm in ['B','previous_halo']:
        directory=ROOT/'seed_45' if arm=='B' else Path('results/stage2_corrected_context/20261009_fold0/halo/seed_45')
        for cohort in ['representative','conditional']:
            path=ROOT/f'evaluation/{arm}_{cohort}_depth.json'
            if not path.exists():
                command=[sys.executable,'-u','-m','stage2.evaluate_direct','--run-dir',str(directory),
                    '--split','test','--horizons','10','--batch-size','8','--evaluation-batches','80',
                    '--start-time-stride','12','--per-event','--wet-classification-mode','depth',
                    '--output-path',str(path),'--device','cuda','--disable-cudnn']
                if cohort=='conditional':command+=['--cohort',str(ROOT/'conditional_cohort_revalidated.json')]
                subprocess.run(command,check=True)
    rows=[]
    for extent_source in ['head','depth']:
        suffix='' if extent_source=='head' else '_depth'
        for cohort in ['representative','conditional']:
            current=json.loads((ROOT/f'evaluation/B_{cohort}{suffix}.json').read_text())
            reference=json.loads((ROOT/f'evaluation/previous_halo_{cohort}{suffix}.json').read_text())
            for key in ['split_events','horizons_steps','batch_size','cohort_batches','wet_classification_mode']:
                if current[key]!=reference[key]:raise ValueError(f'Comparison setting mismatch: {key}')
            if cohort=='conditional' and current['cohort_sha256']!=reference['cohort_sha256']:
                raise ValueError('Conditional cohort mismatch')
            c=current['metrics']['10'];r=reference['metrics']['10']
            for key in ['depth_wet_rmse','wet_f1','depth_all_rmse']:
                if key in c['persistence'] and not math.isclose(c['persistence'][key],r['persistence'][key],rel_tol=1e-9,abs_tol=1e-12):
                    raise ValueError(f'Persistence mismatch: {key}')
            for model,values in [('persistence',c['persistence']),('previous_halo',r['direct']),('B_local_depth_extent',c['direct'])]:
                rows.append(dict(cohort=cohort,extent_source=extent_source,model=model,lead_hours=5,
                    **{key:values[key] for key in ['depth_wet_rmse','wet_f1','wet_precision','wet_recall']},
                    depth_improvement_vs_persistence_percent=100*(1-values['depth_wet_rmse']/c['persistence']['depth_wet_rmse'])))
    with (ROOT/'evaluation/comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# B: fixed 5-hour localized Halo, depth and extent only','',
        'Raw predictions; original 0.05 m wet definition; no threshold fitted to test data.',
        'The previous Halo reference used multiple training leads and discharge/velocity supervision. This comparison does not isolate the causal benefit of localization.',
        'Extent is reported both from the probability head at 0.5 and directly from raw predicted depth >=0.05 m. The depth-defined score exposes shallow false wet predictions.',
        '', '| Cohort | Extent source | Model | Wet-depth RMSE (m) | Wet F1 | Depth improvement vs persistence |',
        '|---|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['cohort']} | {row['extent_source']} | {row['model']} | {row['depth_wet_rmse']:.4f} | {row['wet_f1']:.4f} | {row['depth_improvement_vs_persistence_percent']:.1f}% |")
    (ROOT/'evaluation/REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
