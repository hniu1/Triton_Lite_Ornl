"""Reconstruct north-up whole-domain Halo forecasts and reference depth maps.

Fixed-origin lead sequence or a planned sequence of moving observed origins.
No forecast chaining.
Full-frame input caching changes I/O only: original tile/halo adapters and
training normalization are retained. Reference targets never enter predict_direct.
"""
import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import torch
from torch.utils.data import default_collate

from stage2.data import DirectHorizonDataset
from stage2.direct import move_direct_batch, predict_direct
from stage2.model import Stage2DirectLeadModel
from stage2.prepare_dynamic_stream import digest_file, footprint
from stage2.target_integrity import require_corrected_targets
from triton_core.data import prepare_hydraulic_data
from triton_core.runtime import resolve_device

DEFAULT_RUN = Path('results/stage2_corrected_context/20261009_fold0/halo/seed_45')
DEFAULT_OUTPUT = Path('results/stage2_whole_domain_halo/20261021_D005_origin124h')
BOUNDARY = Path('shapefiles/shapefiles/ConasaugaBasinHUC12.shp')


def verify_orientation(x, y, metadata):
    """Bind native row/column order to the projected raster affine transform."""
    a, b, c, d, e, f = metadata['transform']
    if not (a > 0 and e < 0 and b == 0 and d == 0):
        raise ValueError('Expected north-up unrotated raster')
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) < 0)):
        raise ValueError('Coordinates are not west-to-east, north-to-south')
    if not np.allclose(x, c + a * (np.arange(len(x)) + .5), atol=1e-5, rtol=0):
        raise ValueError('NetCDF x axis does not match tile raster transform')
    if not np.allclose(y, f + e * (np.arange(len(y)) + .5), atol=1e-5, rtol=0):
        raise ValueError('NetCDF y axis does not match tile raster transform')
    if metadata['nc_crs'] != 'EPSG:26916':
        raise ValueError('Unexpected coordinate reference system')
    return dict(crs=metadata['nc_crs'], origin='upper', north_at_top=True,
        transform=metadata['transform'], coordinate_axes_verified=True,
        extent_m=[c, c+a*len(x), f+e*len(y), f], shape=[len(y),len(x)])


def stitch(grid, patch, labels, window, block_index):
    """Undo center padding and place a tile using original raster windows."""
    r0 = (patch.shape[0] - window.height) // 2
    c0 = (patch.shape[1] - window.width) // 2
    native = patch[r0:r0+window.height, c0:c0+window.width]
    destination = grid[window.row_start:window.row_stop, window.col_start:window.col_stop]
    mask = labels[window.row_start:window.row_stop, window.col_start:window.col_stop] == block_index
    destination[mask] = native[mask]


class FrameVariable:
    def __init__(self, source):
        self.source = source
        self.frames = {}

    def set_times(self, indices):
        self.frames = {i: frame for i, frame in self.frames.items() if i in indices}
        for index in indices:
            if index not in self.frames:
                raw = self.source[index, :, :]
                self.frames[index] = np.asarray(np.ma.filled(raw, 0), dtype=np.float32)

    def __getitem__(self, key):
        index, rows, cols = key
        if int(index) not in self.frames:
            raise ValueError('Unplanned hydraulic input time requested')
        return self.frames[int(index)][rows, cols]


def build_dataset(config, batch_size, split="test"):
    if config.get('forcing_weights_sha256') and digest_file(config['forcing_weights_path']) != config['forcing_weights_sha256']:
        raise ValueError('Forcing weights changed since training')
    bundle = prepare_hydraulic_data(
        manifest_dir=Path(config['manifest_dir']), events_csv=Path(config['events_csv']),
        blocks_parquet=Path(config['blocks_parquet']), labels_10m_dir=Path(config['labels_10m_dir']),
        static_rasters_dir=Path(config['static_rasters_dir']), base_dir=Path('.'),
        test_events=config['test_events'], val_fraction=.2, seed=int(config['data_split_seed']),
        batch_size=batch_size, train_batches_per_epoch=1, eval_batches=1,
        train_time_stride=1, eval_time_stride=1, wet_threshold=float(config['wet_threshold']),
        netcdf_chunk_cache_mb=int(config['netcdf_chunk_cache_mb']),
        max_open_netcdf_handles=int(config['max_open_netcdf_handles']))
    if bundle.split_events != config['split_events']:
        raise ValueError('Reconstructed event split differs from training')
    dataset = DirectHorizonDataset(getattr(bundle, f"{split}_dataset"),
        max_horizon_steps=int(config['max_horizon_steps']), step_hours=float(config['step_hours']),
        history_states=int(config['model_config']['history_states']),
        hydraulic_halo_cells=int(config['hydraulic_halo_cells']),
        forcing_weights_path=Path(config["forcing_weights_path"]) if config.get("forcing_weights_path") else None)
    return dataset


def run(args):
    started = time.monotonic()
    config = json.loads((args.run_dir / 'run_config.json').read_text())
    require_corrected_targets(config['manifest_dir'])
    if config.get('graph_hops', 0):
        raise ValueError('Whole-domain tile visualization does not support graph checkpoints')
    device = resolve_device(args.device)
    torch.backends.cudnn.enabled = False
    dataset = build_dataset(config, args.batch_size)
    event_rows = dataset.events[dataset.events.event_id == args.event_id]
    if len(event_rows) != 1:
        raise ValueError('Choose exactly one held-out event')
    event_position = int(event_rows.index[0])
    event = event_rows.iloc[0]
    plan = json.loads(args.plan.read_text()) if args.plan else None
    if plan and plan['event_id'] != args.event_id:
        raise ValueError('Plan event differs from requested event')
    origin = int(args.origin_index)
    leads = [int(round(h / dataset.step_hours)) for h in args.lead_hours]
    if any(abs(l * dataset.step_hours - h) > 1e-8 for l,h in zip(leads,args.lead_hours)):
        raise ValueError('Lead is not on the model time grid')
    if not plan and (origin < dataset.history_states - 1 or origin + max(leads) >= int(event.n_times)):
        raise ValueError('Origin/target outside available event history')
    if not plan and (min(leads) < 1 or max(leads) > dataset.max_horizon_steps):
        raise ValueError('Lead exceeds trained horizon')
    if plan:
        if digest_file(Path(plan['lead_selection_metrics_path'])) != plan['lead_selection_metrics_sha256']:
            raise ValueError('Lead-selection evidence changed')
        if digest_file(Path(plan['selection_rule_path'])) != plan['selection_rule_sha256']:
            raise ValueError('Flood-selection rule changed')
        frame_keys = [(int(f['origin_index']),int(f['target_index']),int(f['lead_steps'])) for f in plan['frames']]
    else:
        frame_keys = [(origin,origin+lead,lead) for lead in leads]
    for origin_index,target_index,lead_steps in frame_keys:
        if not (origin_index >= dataset.history_states - 1 and target_index < int(event.n_times)
                and target_index-origin_index == lead_steps and 1 <= lead_steps <= dataset.max_horizon_steps):
            raise ValueError('Invalid planned origin/target/lead')
    if not frame_keys or len(set(frame_keys)) != len(frame_keys):
        raise ValueError('Empty or duplicate frame plan')
    source_path = Path(event.resolved_netcdf_path)
    source_identity = footprint(source_path)
    checkpoint_path = args.run_dir / 'best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Stage2DirectLeadModel(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    labels = dataset.block_index_grid
    valid = labels >= 0
    metadata = json.loads((Path(config['labels_10m_dir']) / 'labels_10m_metadata.json').read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    with netCDF4.Dataset(source_path) as source:
        x = np.asarray(source.variables['x'][:]); y = np.asarray(source.variables['y'][:])
        orientation = verify_orientation(x,y,metadata)
        names = dataset.base_dataset.variable_names
        cached = {names[key]: FrameVariable(source.variables[names[key]])
                  for key in ("depth", "component_x", "component_y")}
        proxy = SimpleNamespace(variables=cached)
        original_handle = dataset.base_dataset._handle
        def cached_handle(path):
            if Path(path).resolve() != source_path.resolve():
                raise ValueError('Unexpected event requested')
            return proxy
        dataset.base_dataset._handle = cached_handle
        origin_hours = float(event.time_start) + origin * dataset.step_hours
        manifest = dict(status='running', model='halo_local' if config.get('forcing_weights_path') else 'halo', event_id=args.event_id,
            trained_horizons_steps=config.get('trained_horizons_steps'),
            trained_outputs=config.get('trained_outputs',['depth','wet_extent','unit_discharge']),
            forcing_weights_sha256=config.get('forcing_weights_sha256'),
            origin_index=None if plan else origin, origin_hours=None if plan else origin_hours,
            observed_history_indices=None if plan else [origin-1,origin],
            observed_history_hours=None if plan else [origin_hours-dataset.step_hours,origin_hours],
            lead_hours=[f[2]*dataset.step_hours for f in frame_keys], n_tiles=len(dataset.block_rows),
            coverage='All valid cells of every model tile, including non-stream tiles',
            forecast_mode='moving observed origins at fixed lead, not autoregressive' if plan else 'fixed-origin independent direct predictions, not autoregressive',
            plan=plan, plan_sha256=digest_file(args.plan) if plan else None,
            forcing_availability=config['forcing_availability'],
            event_time_note='Hours relative to event start; NetCDF CF epoch is not a verified calendar date',
            checkpoint=str(checkpoint_path.resolve()),checkpoint_epoch=checkpoint.get('epoch'),
            checkpoint_sha256=digest_file(checkpoint_path), run_config_sha256=digest_file(args.run_dir/'run_config.json'),
            source=source_identity, orientation=orientation,
            wet_depth_m=float(config['wet_threshold']), boundary_path=str(BOUNDARY.resolve()),
            io_note='Full hydraulic frames cached in memory; original tile, halo and normalization adapters retained.')
        (args.output_dir/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
        with netCDF4.Dataset(args.output_dir/'depth_maps.nc','w') as out:
            out.createDimension('frame',len(frame_keys));out.createDimension('y',len(y));out.createDimension('x',len(x))
            for name,values,dimension,units in [('x',x,'x','m'),('y',y,'y','m'),('lead_hours',[f[2]*dataset.step_hours for f in frame_keys],'frame','h'),
                ('origin_hours',[float(event.time_start)+f[0]*dataset.step_hours for f in frame_keys],'frame','h'),
                ('target_hours',[float(event.time_start)+f[1]*dataset.step_hours for f in frame_keys],'frame','h')]:
                v=out.createVariable(name,'f8',(dimension,));v[:]=values;v.units=units
            mapping=out.createVariable('transverse_mercator','i4')
            mapping.setncatts(source.variables['transverse_mercator'].__dict__)
            variables={}
            for name in ['true_depth','halo_depth','halo_wet_probability']:
                v=out.createVariable(name,'f4',('frame','y','x'),zlib=True,complevel=2,
                    chunksizes=(1,256,256),fill_value=-9999.)
                v.units='1' if 'probability' in name else 'm';v.grid_mapping='transverse_mercator';variables[name]=v
            out.Conventions='CF-1.8';out.event_id=args.event_id;out.forecast_mode=manifest['forecast_mode']
            out.crs=orientation['crs'];out.forcing_availability=config['forcing_availability']
            out.description='Native model depth without wet-head gating; error is Halo minus truth; north-up native row order.'
            for frame_index,(origin,target,lead) in enumerate(frame_keys):
                origin_hours=float(event.time_start)+origin*dataset.step_hours
                print(f'Loading hydraulic frames: origin={origin}, target={target}',flush=True)
                for variable in cached.values():
                    variable.set_times([origin-1,origin,target])
                prediction=np.full(labels.shape,np.nan,np.float32)
                reference=np.full(labels.shape,np.nan,np.float32)
                probability=np.full(labels.shape,np.nan,np.float32)
                with torch.inference_mode():
                    for start in range(0,len(dataset.block_rows),args.batch_size):
                        positions=range(start,min(start+args.batch_size,len(dataset.block_rows)))
                        batch=default_collate([dataset[(event_position,target,p,lead)] for p in positions])
                        # Check reference reconstruction before moving to GPU.
                        truth=batch['depth'].numpy()
                        batch=move_direct_batch(batch,device)
                        output=predict_direct(model,batch)
                        depth=output[0].detach().cpu().numpy()
                        wet=torch.sigmoid(output[1]).detach().cpu().numpy()
                        for i,position in enumerate(positions):
                            block=int(dataset.block_rows.iloc[position].block_index)
                            window=dataset.block_windows[block]
                            for grid,patch in [(prediction,depth[i]),(reference,truth[i]),(probability,wet[i])]:
                                stitch(grid,patch,labels,window,block)
                        if start % (args.batch_size*50)==0:
                            print(f'lead={lead*dataset.step_hours:g}h: {min(start+args.batch_size,len(dataset.block_rows))}/{len(dataset.block_rows)} tiles; elapsed={time.monotonic()-started:.0f}s',flush=True)
                if not (np.isfinite(prediction[valid]).all() and np.isfinite(reference[valid]).all()):
                    raise ValueError('Mosaic has missing valid cells')
                # Independent check against the unmodified NetCDF frame.
                native_truth=cached[names['depth']].frames[target]
                if not np.array_equal(reference[valid],native_truth[valid]):
                    raise ValueError('Mosaic reference differs from native source; placement/orientation failure')
                for name,grid in [('true_depth',reference),('halo_depth',prediction),('halo_wet_probability',probability)]:
                    variables[name][frame_index]=np.ma.masked_invalid(grid)
                out.sync()
                truth_wet=valid & (reference>=float(config['wet_threshold']))
                delta=prediction-reference
                record=dict(lead_hours=lead*dataset.step_hours,origin_index=origin,origin_hours=origin_hours,target_index=target,
                    target_hours=float(event.time_start)+target*dataset.step_hours,
                    valid_cells=int(valid.sum()),true_wet_cells=int(truth_wet.sum()),
                    depth_rmse_all_valid_m=float(np.sqrt(np.mean(np.square(delta[valid]),dtype=np.float64))),
                    depth_rmse_true_wet_m=float(np.sqrt(np.mean(np.square(delta[truth_wet]),dtype=np.float64))),
                    true_max_depth_m=float(reference[valid].max()),halo_max_depth_m=float(prediction[valid].max()))
                records.append(record)
                print(json.dumps(record),flush=True)
        dataset.base_dataset._handle=original_handle
    if footprint(source_path)!=source_identity:
        raise ValueError('Source data changed during prediction')
    with (args.output_dir/'metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    manifest.update(status='predictions_complete',seconds=time.monotonic()-started,frames=records)
    (args.output_dir/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Predictions complete: {args.output_dir}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    p.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--event-id',default='D005')
    p.add_argument('--origin-index',type=int,default=247)
    p.add_argument('--lead-hours',type=float,nargs='+',default=list(range(2,25,2)))
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument("--plan",type=Path)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cuda')
    args=p.parse_args();run(args)


if __name__=='__main__':
    main()
