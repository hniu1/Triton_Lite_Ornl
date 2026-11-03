"""Build tile forcing weights using same/upstream HUC12 membership and distance.

Routing is at subwatershed scale, not cell-level flow connectivity. Tiles whose
centers are outside the mapped basin have an explicit nearest-point fallback.
"""
import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from data_preprocessing.m0_generate_netcdf_from_zip import CONASAUGA_DEFAULT_ZIP_DIR
from stage2.prepare_dynamic_stream import digest_file, write_new_json

ROOT=Path('processed_data/stage2_fixed5h_local/20261022_fold0/forcing')


def upstream_hops(downstream):
    """Return ancestors and hop distances, rejecting loops in the mapped network."""
    result={key:{key:0} for key in downstream}
    for source in downstream:
        seen={source};target=downstream[source];hops=1
        while target in downstream:
            if target in seen:raise ValueError('Cycle in HUC12 routing')
            seen.add(target);result[target][source]=hops
            target=downstream[target];hops+=1
    return result


def routed_weights(tile_xy,point_xy,tile_huc,point_huc,ancestors,k=8,same_share=.7):
    weights=np.zeros((len(tile_xy),len(point_xy)),dtype=np.float32);audit=[]
    for index,(xy,huc) in enumerate(zip(tile_xy,tile_huc)):
        distance=np.sqrt(((point_xy-xy)**2).sum(1))
        same=np.flatnonzero(point_huc==huc) if huc else np.array([],dtype=int)
        up=np.array([i for i,h in enumerate(point_huc) if huc and h!=huc and h in ancestors[huc]],dtype=int)
        def nearest(candidates):
            return candidates[np.argsort(distance[candidates],kind='stable')[:min(k,len(candidates))]]
        same=nearest(same);up=nearest(up)
        if len(same):
            group=1-same_share if len(up) else 0
            local=1/np.maximum(distance[same],100.);weights[index,same]=(1-group)*local/local.sum()
            if len(up):
                local=1/(np.maximum(distance[up],100.)*np.array([1+ancestors[huc][point_huc[i]] for i in up]))
                weights[index,up]=group*local/local.sum()
            mode='same_and_upstream' if len(up) else 'same_huc_only'
        else:
            selected=nearest(np.arange(len(point_xy)));local=1/np.maximum(distance[selected],100.)
            weights[index,selected]=local/local.sum();mode='outside_basin_nearest_fallback' if not huc else 'empty_huc_nearest_fallback'
        audit.append(dict(block_index=index,huc12=huc or '',mode=mode,
            n_same_huc_points=len(same),n_upstream_points=len(up),n_weighted_points=int((weights[index]>0).sum())))
    if not np.isfinite(weights).all() or (weights<0).any() or not np.allclose(weights.sum(1),1,atol=1e-6):
        raise ValueError('Invalid forcing weights')
    return weights,audit


def assign_huc(points,hucs):
    joined=gpd.sjoin(points,hucs[['HUC12','geometry']],how='left',predicate='within')
    if joined.index.duplicated().any():raise ValueError('Ambiguous HUC membership')
    return joined['HUC12'].reindex(points.index).fillna('').to_numpy(dtype=str)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=ROOT)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    paths=dict(blocks=Path('processed_data/timestamp_stage1/m2_blocks/blocks.parquet'),
        lookup=Path('processed_data/timestamp_stage1/m3_labels_10m/block_index_lookup.parquet'),
        locations=Path('shapefiles/flow_locations/flow_locations.shp'),hucs=Path('shapefiles/shapefiles/ConasaugaBasinHUC12.shp'))
    hucs=gpd.read_file(paths['hucs']).to_crs('EPSG:26916')
    for key in ('HUC12','ToHUC'):hucs[key]=hucs[key].astype(str).str.zfill(12)
    if hucs.HUC12.duplicated().any():raise ValueError('Duplicate HUC12 IDs')
    points=gpd.read_file(paths['locations']).to_crs(hucs.crs).sort_values('Loc').reset_index(drop=True)
    if not np.array_equal(points.Loc.to_numpy(),np.arange(1,301)):raise ValueError('Expected ordered Loc1..300')
    xy=points[['X','Y']].to_numpy(float)
    if not np.allclose(np.c_[points.geometry.x,points.geometry.y],xy,atol=1e-6,rtol=0):
        raise ValueError('Point coordinates do not match attributes')
    source_checks=[]
    for event_id in pd.read_csv('processed_data/timestamp_stage1/m1_events/events.csv').event_id:
        archive_path=Path(CONASAUGA_DEFAULT_ZIP_DIR)/f'{event_id}.zip'
        with zipfile.ZipFile(archive_path) as archive:
            cfg=archive.read(f'{event_id}/input/cfg/D001.cfg').decode()
            match=re.search(r'^\s*src_loc_file\s*=\s*"([^"]+)"',cfg,re.MULTILINE)
            if not match:raise ValueError('Missing source location file in simulator config')
            member=f'{event_id}/{match.group(1)}';raw=archive.read(member)
            actual=np.loadtxt(io.StringIO(raw.decode()),delimiter=',',comments='%')
            if actual.shape!=(300,2) or not np.allclose(actual,xy,atol=1e-6,rtol=0):
                raise ValueError(f'Forcing point order/coordinates differ for {event_id}')
            source_checks.append(dict(event_id=event_id,member=member,sha256=hashlib.sha256(raw).hexdigest(),matches=True))
    lookup=pd.read_parquet(paths['lookup']).sort_values('block_index')
    if not np.array_equal(lookup.block_index,np.arange(len(lookup))):raise ValueError('Noncontiguous block indices')
    blocks=lookup.merge(pd.read_parquet(paths['blocks']),on=['watershed_id','block_id'],validate='one_to_one').sort_values('block_index').reset_index(drop=True)
    tiles=gpd.GeoDataFrame(blocks,geometry=gpd.points_from_xy(blocks.centroid_x,blocks.centroid_y),crs=hucs.crs)
    tile_huc=assign_huc(tiles,hucs);point_huc=assign_huc(points,hucs)
    if (point_huc=='').any():raise ValueError('Forcing point outside mapped HUC12s')
    routing=dict(zip(hucs.HUC12,hucs.ToHUC));ancestors=upstream_hops(routing)
    weights,audit=routed_weights(blocks[['centroid_x','centroid_y']].to_numpy(),xy,tile_huc,point_huc,ancestors)
    np.save(args.output_dir/'tile_forcing_weights.npy',weights)
    pd.DataFrame(audit).assign(block_id=blocks.block_id.to_numpy()).to_csv(args.output_dir/'tile_mapping.csv',index=False)
    pd.DataFrame(dict(Loc=points.Loc.astype(int),huc12=point_huc,X=xy[:,0],Y=xy[:,1])).to_csv(args.output_dir/'point_mapping.csv',index=False)
    provenance={}
    for name,path in paths.items():
        files=list(path.parent.glob(path.stem+'.*')) if path.suffix=='.shp' else [path]
        provenance[name]={str(f.resolve()):digest_file(f) for f in files}
    metadata=dict(status='passed',n_tiles=len(blocks),n_forcing_locations=300,crs=str(hucs.crs),
        ordering='Rows follow block_index; columns follow ascending Loc1..Loc300',
        mapping_counts=pd.DataFrame(audit)['mode'].value_counts().to_dict(),
        weighting='Up to 8 nearest same-HUC12 and 8 nearest ancestor-HUC12 points; 70/30 mixture where both exist; inverse-distance with 100 m floor, upstream weights additionally divided by 1+HUC hops; renormalized per group.',
        interpretation='Weighted standardized forcing anomaly, not summed physical inflow. Same-HUC membership does not prove cell-level upstream connectivity. Outside-basin centers have explicit nearest-8 fallback. No travel-time alignment is asserted.',
        source_location_checks=source_checks,source_hashes=provenance,weights_sha256=digest_file(args.output_dir/'tile_forcing_weights.npy'))
    write_new_json(args.output_dir/'metadata.json',metadata)
    print(json.dumps({k:metadata[k] for k in ['status','n_tiles','n_forcing_locations','mapping_counts']},indent=2),flush=True)


if __name__=='__main__':main()
