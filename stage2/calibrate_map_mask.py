"""Choose a visualization mask on validation floods; audit a frozen test sequence.

Reference flood definition stays depth >=0.05 m for all candidate masks.
No thresholds are selected from D005 test data. Native counts are retained.
"""
import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pandas as pd
import torch
from torch.utils.data import default_collate

from stage2.calibrate_extent import score_counts
from stage2.direct import move_direct_batch, predict_direct
from stage2.model import Stage2DirectLeadModel
from stage2.prepare_dynamic_stream import digest_file, footprint, write_new_json
from stage2.target_integrity import require_corrected_targets
from stage2.whole_domain_halo import DEFAULT_RUN, FrameVariable, build_dataset

ROOT=Path('results/stage2_whole_domain_halo/20261022_mask_calibration')
PREPARATION=Path('results/stage2_stream_flood/20261011_fold0/preparation')
DEPTHS=np.array([.05,.075,.10,.125,.15,.175,.20,.25,.30,.40,.50],dtype=np.float32)
PROBABILITIES=np.array([0,.05,.10,.20,.30,.40,.50,.60,.70,.80,.90,.95],dtype=np.float32)


def empty_hist():
    return np.zeros((2,len(DEPTHS)+1,len(PROBABILITIES)+1),dtype=np.int64)


def add_hist(hist,truth,depth,probability,valid):
    if not (truth.shape==depth.shape==probability.shape==valid.shape):
        raise ValueError('Candidate arrays do not align')
    valid=valid & np.isfinite(truth) & np.isfinite(depth) & np.isfinite(probability)
    wet=truth >= np.float32(.05)
    di=np.searchsorted(DEPTHS,depth,side='right')
    pi=np.searchsorted(PROBABILITIES,probability,side='right')
    codes=di*(len(PROBABILITIES)+1)+pi
    size=(len(DEPTHS)+1)*(len(PROBABILITIES)+1)
    for category in (0,1):
        counts=np.bincount(codes[valid & (wet==bool(category))].ravel(),minlength=size)
        hist[category]+=counts.reshape(hist.shape[1:])


def table(hist):
    rows=[]
    total=hist.sum();positive=hist[1].sum();negative=hist[0].sum()
    tails=hist[:,::-1,::-1].cumsum(1).cumsum(2)[:,::-1,::-1]
    for di,d in enumerate(DEPTHS):
        for pi,p in enumerate(PROBABILITIES):
            tp=int(tails[1,di+1,pi+1]);fp=int(tails[0,di+1,pi+1])
            fn=int(positive)-tp;tn=int(negative)-fp
            rows.append(dict(depth_threshold_m=round(float(d),3),probability_threshold=round(float(p),3),
                family='depth_only' if p==0 else 'depth_and_wet_head',
                tp=tp,fp=fp,fn=fn,tn=tn,valid_cells=int(total),**score_counts(tp,fp,fn,tn)))
    return rows


def select_candidate(rows,family=None):
    eligible=[r for r in rows if r['recall']>=.95 and r['min_event_recall']>=.90
              and r.get('min_frame_recall',1)>=.90
              and (family is None or r['family']==family)]
    if not eligible:
        return None
    # Selection never rewards dry-background accuracy. Equal weight per event.
    return max(eligible,key=lambda r:(r['macro_event_f1'],r['recall'],-r['depth_threshold_m'],-r['probability_threshold']))


def write_csv(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plan(args):
    config=json.loads((args.run_dir/'run_config.json').read_text())
    rule_path=Path('results/stage2_stream_flood/20261011_fold0/RULE_available_history.json')
    rule=json.loads(rule_path.read_text());thresholds=rule['thresholds'];frames=[]
    for event_id in config['split_events']['val']:
        path=PREPARATION/'events'/event_id/'event_changes.parquet'
        f=pd.read_parquet(path)
        f=f[(f.interval_hours==3)&(f.depth_change_rms_m>=thresholds['event_depth_change_rms_m'])&
            (f.velocity_vector_change_rms_mps>=thresholds['event_velocity_vector_change_rms_mps'])].copy()
        f['target_index']=f.origin_time_index+6
        # Match the map target-time grid: even event-relative hours and 5 h lead.
        f=f[(f.target_index>=11)&(((.5+.5*f.target_index)%2)==0)].sort_values('target_index')
        if len(f)<3:
            raise ValueError(f'Insufficient validation flood targets: {event_id}')
        for rank in np.linspace(0,len(f)-1,3,dtype=int):
            target=int(f.iloc[rank].target_index)
            frames.append(dict(event_id=event_id,target_index=target,origin_index=target-10,
                               target_hours=.5+.5*target,origin_hours=.5+.5*(target-10),lead_hours=5,
                               changes_sha256=digest_file(path)))
    args.root.mkdir(parents=True,exist_ok=True)
    value=dict(checkpoint=str((args.run_dir/'best_model.pt').resolve()),checkpoint_sha256=digest_file(args.run_dir/'best_model.pt'),
        run_config_sha256=digest_file(args.run_dir/'run_config.json'),split='val',events=config['split_events']['val'],
        selection_rule_path=str(rule_path.resolve()),selection_rule_sha256=digest_file(rule_path),
        selection='Three evenly ranked flood-active target times per validation event; all model tiles and valid cells.',
        lead_hours=5,truth_wet_depth_m=.05,depth_thresholds_m=DEPTHS.tolist(),probability_thresholds=PROBABILITIES.tolist(),
        objective='Maximize mean per-event wet-extent F1 subject to pooled recall >=0.95 and every event recall >=0.90; no silent constraint relaxation.',
        frames=frames)
    write_new_json(args.root/'PLAN.json',value)
    print(pd.DataFrame(frames)[['event_id','target_hours','origin_hours']].to_string(index=False),flush=True)


def calibrate(args):
    started=time.monotonic();planned=json.loads((args.root/'PLAN.json').read_text())
    config=json.loads((args.run_dir/'run_config.json').read_text());require_corrected_targets(config['manifest_dir'])
    if digest_file(args.run_dir/'best_model.pt')!=planned['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    if digest_file(args.run_dir/'run_config.json')!=planned['run_config_sha256']:
        raise ValueError('Training configuration changed')
    if digest_file(planned['selection_rule_path'])!=planned['selection_rule_sha256']:
        raise ValueError('Flood-selection rule changed')
    dataset=build_dataset(config,args.batch_size,split='val')
    if dataset.events.event_id.tolist()!=planned['events']:
        raise ValueError('Validation event identities differ')
    device=torch.device(args.device);torch.backends.cudnn.enabled=False
    checkpoint=torch.load(args.run_dir/'best_model.pt',map_location=device,weights_only=False)
    model=Stage2DirectLeadModel(**checkpoint['model_config']).to(device);model.load_state_dict(checkpoint['model_state_dict']);model.eval()
    accumulated={event_id:empty_hist() for event_id in planned['events']};per_frame=[];sources={}
    original_handle=dataset.base_dataset._handle
    for event_id in planned['events']:
        event_position=dataset.events.index[dataset.events.event_id==event_id].item()
        event=dataset.events.iloc[event_position];path=Path(event.resolved_netcdf_path);identity=footprint(path);sources[event_id]=identity
        changes=PREPARATION/'events'/event_id/'event_changes.parquet'
        frames=[f for f in planned['frames'] if f['event_id']==event_id]
        if any(digest_file(changes)!=f['changes_sha256'] for f in frames):
            raise ValueError('Validation characterization changed')
        with netCDF4.Dataset(path) as source:
            names=dataset.base_dataset.variable_names
            cached={names[k]:FrameVariable(source.variables[names[k]]) for k in ('depth','component_x','component_y')}
            proxy=SimpleNamespace(variables=cached)
            dataset.base_dataset._handle=lambda requested: proxy
            for frame in frames:
                origin,target=frame['origin_index'],frame['target_index']
                print(f"Loading {event_id}: target {frame['target_hours']:g} h",flush=True)
                for variable in cached.values():variable.set_times([origin-1,origin,target])
                hist=empty_hist()
                with torch.inference_mode():
                    for start in range(0,len(dataset.block_rows),args.batch_size):
                        positions=range(start,min(start+args.batch_size,len(dataset.block_rows)))
                        batch=default_collate([dataset[(event_position,target,b,10)] for b in positions])
                        truth=batch['depth'].numpy();valid=batch['mask'].numpy()>.5
                        batch=move_direct_batch(batch,device);output=predict_direct(model,batch)
                        depth=output[0].cpu().numpy();prob=torch.sigmoid(output[1]).cpu().numpy()
                        add_hist(hist,truth,depth,prob,valid)
                        if start%(args.batch_size*100)==0:
                            print(f"{event_id} {frame['target_hours']:g} h: {min(start+args.batch_size,len(dataset.block_rows))}/{len(dataset.block_rows)} tiles; {time.monotonic()-started:.0f}s",flush=True)
                accumulated[event_id]+=hist;per_frame.append(dict(**frame,histogram=hist.tolist()))
        if footprint(path)!=identity:raise ValueError('Validation source changed')
    dataset.base_dataset._handle=original_handle
    total=sum(accumulated.values(),empty_hist());rows=table(total)
    event_tables={event:table(hist) for event,hist in accumulated.items()}
    for i,row in enumerate(rows):
        row['macro_event_f1']=float(np.mean([values[i]['f1'] for values in event_tables.values()]))
        row['min_event_recall']=min(values[i]['recall'] for values in event_tables.values())
        baseline=rows[0]
        row['false_wet_removed_fraction']=1-row['fp']/max(baseline['fp'],1)
        row['eligible']=row['recall']>=.95 and row['min_event_recall']>=.90
    selected=select_candidate(rows);best_depth=select_candidate(rows,'depth_only');best_gated=select_candidate(rows,'depth_and_wet_head')
    report=dict(status='calibration_complete' if selected else 'no_candidate_meets_recall_constraints',plan_sha256=digest_file(args.root/'PLAN.json'),
        selection_split='val',reference_wet_depth_m=.05,selected=selected,best_depth_only=best_depth,best_gated=best_gated,
        baseline=rows[0],candidate_rows=rows,per_event_candidates=event_tables,
        sources=sources,seconds=time.monotonic()-started,
        interpretation='Threshold changes the displayed/postprocessed flood extent, not the trained depth model or original RMSE. D005 test results cannot select a candidate.')
    write_new_json(args.root/'CALIBRATION.json',report)
    write_csv(args.root/'validation_candidates.csv',rows)
    np.savez_compressed(args.root/'validation_histograms.npz',**accumulated)
    write_new_json(args.root/'validation_frames.json',dict(frames=per_frame))
    print(json.dumps({key:report[key] for key in ['status','selected','best_depth_only','baseline']},indent=2),flush=True)


def audit(args):
    calibration=json.loads((args.root/'CALIBRATION.json').read_text())
    # Protect early flood frames as well as event-level averages. This uses
    # validation histograms only and runs before any D005 maps are inspected.
    validation_frames=json.loads((args.root/'validation_frames.json').read_text())['frames']
    frame_tables=[table(np.asarray(frame['histogram'],dtype=np.int64)) for frame in validation_frames]
    for i,row in enumerate(calibration['candidate_rows']):
        row['min_frame_recall']=min(values[i]['recall'] for values in frame_tables)
        row['eligible']=row['recall']>=.95 and row['min_event_recall']>=.90 and row['min_frame_recall']>=.90
    calibration['selected']=select_candidate(calibration['candidate_rows'])
    calibration['best_depth_only']=select_candidate(calibration['candidate_rows'],'depth_only')
    calibration['best_gated']=select_candidate(calibration['candidate_rows'],'depth_and_wet_head')
    calibration['parent_calibration_sha256']=digest_file(args.root/'CALIBRATION.json')
    calibration['additional_safety_constraint']='Recall >=0.90 in every individual validation frame; prevents an event average from hiding early-flood omissions. Applied before D005 audit; all selection remains validation-only.'
    write_new_json(args.root/'CALIBRATION_FRAME_SAFE.json',calibration)
    write_csv(args.root/'validation_candidates_frame_safe.csv',calibration['candidate_rows'])
    if not calibration['selected']:
        raise ValueError('No acceptable validation threshold; cannot declare plot settings ready')
    manifest=json.loads((args.maps_dir/'MANIFEST.json').read_text())
    if manifest['status']!='complete':raise ValueError('Test maps incomplete')
    aggregate=empty_hist();rows=[];error_edges=np.r_[np.arange(0,1.001,.01),2,4,8,np.inf];error_hist=np.zeros(len(error_edges)-1,dtype=np.int64)
    for i,frame in enumerate(manifest['frames']):
        hist=empty_hist();over_range=0;error_count=0
        with netCDF4.Dataset(args.maps_dir/'depth_maps.nc') as maps:
            for start in range(0,len(maps.dimensions['y']),512):
                slices=(i,slice(start,start+512),slice(None))
                truth=np.ma.filled(maps['true_depth'][slices],np.nan)
                depth=np.ma.filled(maps['halo_depth'][slices],np.nan)
                prob=np.ma.filled(maps['halo_wet_probability'][slices],np.nan)
                valid=np.isfinite(truth)&np.isfinite(depth)&np.isfinite(prob)
                add_hist(hist,truth,depth,prob,valid)
                union=valid&((truth>=.05)|(depth>=.05));error=np.abs((depth-truth)[union])
                error_hist+=np.histogram(error,bins=error_edges)[0]
                over_range+=int((error>.3).sum());error_count+=len(error)
        aggregate+=hist;candidate_rows=table(hist)
        for label,candidate in [('baseline',calibration['baseline']),('selected',calibration['selected']),('depth_only',calibration['best_depth_only'])]:
            if not candidate:continue
            found=next(r for r in candidate_rows if r['depth_threshold_m']==candidate['depth_threshold_m'] and r['probability_threshold']==candidate['probability_threshold'])
            rows.append(dict(event_id=manifest['event_id'],target_hours=frame['target_hours'],variant=label,
                **found,error_over_0p3m_fraction=over_range/max(error_count,1)))
        print(f"Audited D005 {frame['target_hours']:g} h",flush=True)
    all_rows=table(aggregate);test_summary={}
    for label,candidate in [('baseline',calibration['baseline']),('selected',calibration['selected']),('depth_only',calibration['best_depth_only'])]:
        if candidate:
            test_summary[label]=next(r for r in all_rows if r['depth_threshold_m']==candidate['depth_threshold_m'] and r['probability_threshold']==candidate['probability_threshold'])
    validation_frames=json.loads((args.root/'validation_frames.json').read_text())['frames']
    validation_frame_scores=[]
    candidate=calibration['selected']
    for frame in validation_frames:
        scores=next(r for r in table(np.asarray(frame['histogram'],dtype=np.int64))
            if r['depth_threshold_m']==candidate['depth_threshold_m'] and r['probability_threshold']==candidate['probability_threshold'])
        validation_frame_scores.append(dict(event_id=frame['event_id'],target_hours=frame['target_hours'],**scores))
    write_csv(args.root/'validation_selected_per_frame.csv',validation_frame_scores)
    selected_test_rows=[r for r in rows if r['variant']=='selected']
    q95=float(error_edges[min(np.searchsorted(np.cumsum(error_hist),.95*error_hist.sum())+1,len(error_edges)-1)])
    report=dict(status='ready_for_new_plots',calibration_sha256=digest_file(args.root/'CALIBRATION_FRAME_SAFE.json'),
        recommended_mask=calibration['selected'],recommended_depth_only=calibration['best_depth_only'],test_audit=test_summary,
        validation_worst_frame=min(validation_frame_scores,key=lambda r:r['recall']),
        D005_worst_frame=min(selected_test_rows,key=lambda r:r['recall']),
        error_colormap='RdBu',error_definition='Halo minus reference; positive blue, negative red',
        error_limits_m=[-.3,.3],error_over_range_note='Saturate and mark colorbar extensions; do not clip stored data',
        raw_test_error_approximate_95th_percentile_m=q95,
        error_panel_note='Keep raw prediction error visible, including any cells hidden by the depth/probability mask; use the same native error definition across variants.',
        aggregation_note='Use native-resolution zooms to distinguish maximum-pooling display width from actual prediction width.',
        full_test_candidates_are_not_used_for_selection=True)
    write_new_json(args.root/'READY.json',report);write_csv(args.root/'D005_mask_audit.csv',rows)
    write_new_json(args.root/'D005_error_histogram.json',dict(edges=[float(v) if np.isfinite(v) else None for v in error_edges],counts=error_hist.tolist()))
    lines=['# Halo map threshold assessment','', 'Validation-selected mask; D005 is an audit only. The reference wet definition remains depth >=0.05 m.', '',
           'Selection maximizes equal-event mean F1, requiring pooled recall >=95% and recall >=90% in every validation event AND individual frame.', '',
           '| Variant | Depth cutoff (m) | Wet probability cutoff | Validation precision | Validation recall | Validation mean event F1 | D005 precision | D005 recall |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,c in [('baseline',calibration['baseline']),('selected',calibration['selected']),('depth_only',calibration['best_depth_only'])]:
        if c:
            test=test_summary[label]
            lines.append(f"| {label} | {c['depth_threshold_m']:g} | {c['probability_threshold']:g} | {c['precision']:.4f} | {c['recall']:.4f} | {c['macro_event_f1']:.4f} | {test['precision']:.4f} | {test['recall']:.4f} |")
    lines.extend(['',f"Worst single-frame recall: validation {report['validation_worst_frame']['recall']:.3f}; D005 {report['D005_worst_frame']['recall']:.3f}. Validation recall safeguards do not guarantee recall at every test instant.",'','Error colors: blue = overprediction; red = underprediction; fixed ±0.3 m with overflow indicated.',
                  'Thresholding is visualization/postprocessing, not new model training. Preserve original plots and numerical depth errors.',
                  'See validation_candidates.csv, D005_mask_audit.csv and READY.json for complete counts and early/late failures.'])
    (args.root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    # Concrete previews keep the original 28-figure sequence unchanged.
    from stage2.plot_whole_domain_halo import run as render
    preview_times=[6.0,76.0]
    render(SimpleNamespace(input_dir=args.maps_dir,display_factor=4,tile_grid=False,
        settings=args.root/'READY.json',output_dir=args.root/'preview',target_hours=preview_times))
    with (args.root/'REPORT.md').open('a') as handle:
        handle.write('\n## Preview figures\n\n')
        for target in preview_times:
            stem=f"target_{target:06.1f}h_lead_05.0h.png"
            handle.write(f'![Target {target:g} h](preview/{stem})\n\n')
    print('Threshold assessment and early/late previews are ready.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','calibrate','audit'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    p.add_argument('--maps-dir',type=Path,default=Path('results/stage2_whole_domain_halo/20261021_D005_fixed5h_flood'))
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=p.parse_args();dict(plan=plan,calibrate=calibrate,audit=audit)[args.action](args)
