"""Revalidate a fixed cohort after copied source files acquired new timestamps.

Never bypass the original guards or modify the historical layout/cohort.
Recompute all held-out selection statistics before publishing a new binding.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage2.prepare_dynamic_stream import digest_file,footprint,write_new_json

ROOT=Path('results/stage2_fixed5h_local/20261022_fold0')
OLD=Path('results/stage2_stream_flood/20261011_fold0/test_preparation')


def initialize():
    old_path=OLD/'LAYOUT.json';old=json.loads(old_path.read_text());fresh=copy.deepcopy(old)
    for key,field in [('manifest','manifest_sha256'),('lookup','lookup_sha256'),('feature_names','feature_names_sha256')]:
        if digest_file(old['source_files'][key]['path'])!=old[field]:raise ValueError(f'Historical {key} content changed')
    features=np.load(old['source_files']['static_features']['path'],mmap_mode='r')
    masks=np.load(old['source_files']['static_masks']['path'],mmap_mode='r')
    names=json.loads(Path(old['source_files']['feature_names']['path']).read_text())['feature_names']
    stream=(features[:,names.index('stream_mask')]>0.5)&(masks>0.5)
    if hashlib.sha256(stream.tobytes()).hexdigest()!=old['stream_mask_sha256']:
        raise ValueError('Historical stream mask changed')
    legacy=Path('processed_data/blockwise_global/milestone_03_labels_10m/block_index_10m.npy')
    if digest_file(old['grid_path'])!=digest_file(legacy):raise ValueError('Raster grid differs from preserved copy')
    previous=np.load('results/stage2_corrected_context/20261009_fold0/halo/seed_45/normalization.npz')
    current=np.load(ROOT/'seed_45/normalization.npz')
    if any(not np.array_equal(previous[key],current[key]) for key in previous.files):
        raise ValueError('Normalization changed')
    fresh['source_files']={key:footprint(value['path']) for key,value in old['source_files'].items()}
    fresh['source_sha256']={key:digest_file(value['path']) for key,value in old['source_files'].items()}
    for event in fresh['events']:
        before=event['source'];after=footprint(before['path'])
        if before['path']!=after['path'] or before['size']!=after['size']:
            raise ValueError('Source event path or size changed')
        event['source']=after;fresh['source_files']['hydraulics_'+event['event_id']]=after
    fresh['parent_layout_sha256']=digest_file(old_path)
    fresh['rebinding_note']='Source mtimes changed. Historical available hashes, stream mask and raster reference match; normalization matches. Full hydraulic selector equality must pass the dependent comparison. Original full static-file SHA-256 values were not recorded; current full hashes are pinned, not claimed identical to unrecorded historical hashes.'
    destination=ROOT/'source_reaudit';destination.mkdir()
    write_new_json(destination/'LAYOUT.json',fresh)
    (destination/'events.txt').write_text('\n'.join(e['event_id'] for e in fresh['events'])+'\n')
    print('Prepared strict source reaudit for eight held-out events',flush=True)


def verify():
    folder=ROOT/'source_reaudit';layout=json.loads((folder/'LAYOUT.json').read_text());reports=[]
    for event in layout['events']:
        event_id=event['event_id'];path=folder/'events'/event_id
        status=json.loads((path/'PREPARATION.json').read_text())
        if status['status']!='complete' or status['layout_sha256']!=digest_file(folder/'LAYOUT.json'):
            raise ValueError('Incomplete or mismatched rescan')
        files={}
        for name in ['tile_state.parquet','tile_changes.parquet','event_state.parquet','event_changes.parquet']:
            original=pd.read_parquet(OLD/'events'/event_id/name);new=pd.read_parquet(path/name)
            pd.testing.assert_frame_equal(original,new,check_exact=True)
            files[name]=dict(rows=len(new),exact_historical_statistics_match=True,new_sha256=digest_file(path/name))
        # Pin a full content hash now; old preparation only recorded a footprint.
        source=event['source'];before=footprint(source['path'])
        if before!=source:raise ValueError('Hydraulic source changed during reaudit')
        content_hash=digest_file(source['path'])
        if footprint(source['path'])!=before:raise ValueError('Hydraulic source changed while hashing')
        reports.append(dict(event_id=event_id,statistics=files,source=source,source_sha256=content_hash))
        print(event_id,'all four full-series selection tables exactly match',flush=True)
    report=dict(status='passed',parent_layout_sha256=layout['parent_layout_sha256'],
        current_layout_sha256=digest_file(folder/'LAYOUT.json'),events=reports,
        selection_changes='none; thresholds, times and tile indices remain fixed',
        limitations=layout['rebinding_note'])
    write_new_json(folder/'SOURCE_REAUDIT.json',report)
    original_path=ROOT/'conditional_cohort.json';cohort=json.loads(original_path.read_text())
    cohort['parent_binding_sha256']=digest_file(original_path)
    cohort['layout_path']=str((folder/'LAYOUT.json').resolve());cohort['layout_sha256']=digest_file(folder/'LAYOUT.json')
    cohort['source_reaudit_path']=str((folder/'SOURCE_REAUDIT.json').resolve())
    cohort['source_reaudit_sha256']=digest_file(folder/'SOURCE_REAUDIT.json')
    write_new_json(ROOT/'conditional_cohort_revalidated.json',cohort)
    print('Published new source binding; original artifacts preserved',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['initialize','verify'])
    args=p.parse_args();(initialize if args.action=='initialize' else verify)()
