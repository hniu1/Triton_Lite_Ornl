"""Characterize stream-tile hydraulic changes before choosing evaluation thresholds.

Process NetCDF spatial chunks once across all times; retain scalar tile/time
summaries rather than storing another copy of hydraulic maps. No model weights
or predictions are used. Default processing covers training and validation only.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from stage2.target_integrity import require_corrected_targets


TASK_ID = 'dynamic_stream_flood_eval'
DEFAULT_CONFIG = Path('results/stage2_corrected_baseline/20261009_fold0/seed_45/run_config.json')
DEFAULT_ROOT = Path('results/stage2_stream_flood/20261011_fold0/preparation')
INTERVALS = (1, 6, 12)
STATE_FIELDS = ('valid_cells', 'wet_cells', 'depth_sum', 'depth_sq_sum',
                'velocity_cells', 'speed_sum')
CHANGE_FIELDS = ('pair_cells', 'wet_union_cells', 'common_wet_velocity_cells',
                 'depth_abs_sum', 'depth_sq_sum', 'velocity_vector_sq_sum',
                 'newly_wet_cells', 'newly_dry_cells')


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def footprint(path):
    p = Path(path).resolve()
    stat = p.stat()
    return dict(path=str(p), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def write_new_json(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def initialize(root, config_path):
    config = json.loads(config_path.read_text())
    manifest_dir = Path(config['manifest_dir'])
    require_corrected_targets(manifest_dir)
    metadata = json.loads((manifest_dir / 'dynamic_metadata.json').read_text())
    manifest_path = manifest_dir / 'dynamic_manifest.parquet'
    manifest = pd.read_parquet(manifest_path)
    static_dir, label_dir = Path(config['static_rasters_dir']), Path(config['labels_10m_dir'])
    names_path = static_dir / 'block_static_feature_names.json'
    names = json.loads(names_path.read_text())['feature_names']
    features = np.load(static_dir / 'block_static_features.npy', mmap_mode='r')
    masks = np.load(static_dir / 'block_static_masks.npy', mmap_mode='r')
    lookup_path = label_dir / 'block_index_lookup.parquet'
    lookup = pd.read_parquet(lookup_path)
    static_lookup = pd.read_parquet(static_dir / 'block_static_lookup.parquet')
    if not lookup.equals(static_lookup[lookup.columns]):
        raise ValueError('Static and hydraulic tile lookup differ')
    valid = masks > 0.5
    stream = (features[:, names.index('stream_mask')] > 0.5) & valid
    stream_counts = stream.sum(axis=(1, 2))
    counts = valid.sum(axis=(1, 2))
    labels = lookup['block_index'].astype(int).to_numpy()
    selected = lookup.loc[stream_counts[labels] > 0].copy()
    selected['valid_cells'] = counts[selected.block_index.to_numpy()]
    selected['stream_cells'] = stream_counts[selected.block_index.to_numpy()]
    if selected.empty:
        raise ValueError('No stream tiles')
    grid_path = label_dir / 'block_index_10m.npy'
    grid = np.load(grid_path, mmap_mode='r')
    grid_counts = np.bincount(np.asarray(grid[grid >= 0], dtype=np.int64), minlength=len(counts))
    if not np.array_equal(grid_counts[labels], counts[labels]):
        raise ValueError('Raster tile cells disagree with static valid masks')
    training_validation = sorted(config['split_events']['train'] + config['split_events']['val'])
    if set(training_validation) & set(config['split_events']['test']):
        raise ValueError('Train/validation and test overlap')
    events = []
    for event_id in training_validation:
        rows = manifest.loc[manifest.event_id == event_id]
        if len(rows) != 1:
            raise ValueError(f'Missing/duplicate event {event_id}')
        row = rows.iloc[0]
        if float(row.time_step) != 0.5:
            raise ValueError('Expected verified half-hour axis')
        events.append(dict(event_id=event_id,
                           split='train' if event_id in config['split_events']['train'] else 'val',
                           n_times=int(row.n_times), time_start_hours=float(row.time_start),
                           source=footprint(row.path_to_netcdf)))
    root.mkdir(parents=True, exist_ok=True)
    layout = dict(task_id=TASK_ID, status='layout_ready', config=str(config_path.resolve()),
                  manifest_sha256=digest_file(manifest_path), lookup_sha256=digest_file(lookup_path),
                  feature_names_sha256=digest_file(names_path),
                  stream_mask_sha256=hashlib.sha256(stream.tobytes()).hexdigest(),
                  source_files={key: footprint(path) for key, path in {
                      'manifest': manifest_path, 'lookup': lookup_path, 'feature_names': names_path,
                      'static_features': static_dir / 'block_static_features.npy',
                      'static_masks': static_dir / 'block_static_masks.npy', 'block_grid': grid_path}.items()},
                  grid_path=str(grid_path.resolve()), variable_names=metadata['variable_names'],
                  stream_tiles=selected[['block_index', 'block_id', 'valid_cells', 'stream_cells']].to_dict('records'),
                  total_tiles=len(lookup), n_stream_tiles=len(selected), events=events,
                  held_out_events_not_processed=config['split_events']['test'],
                  step_hours=0.5, intervals_steps=list(INTERVALS),
                  wet_depth_m=0.05, velocity_min_depth_m=0.1,
                  note='0.1 m is a diagnostic velocity denominator floor/eligibility rule, not a change-selection threshold; no cohort selected.')
    write_new_json(root / 'LAYOUT.json', layout)
    selected.to_csv(root / 'stream_tiles.csv', index=False)
    (root / 'events.txt').write_text('\n'.join(training_validation) + '\n')
    print(f'Prepared {len(selected)} stream tiles and {len(events)} train/validation events', flush=True)


def new_accumulators(n_tiles, n_times, intervals=INTERVALS):
    state = {key: np.zeros((n_tiles, n_times), dtype=np.float64) for key in STATE_FIELDS}
    changes = {lag: {key: np.zeros((n_tiles, n_times - lag), dtype=np.float64)
                     for key in CHANGE_FIELDS} for lag in intervals}
    return state, changes


def accumulate_pixels(state, changes, tile, depth, qx, qy, wet_depth=0.05, velocity_depth=0.1):
    """Accumulate one spatial fragment of a tile, shaped [time, valid pixels]."""
    available = np.isfinite(depth) & (depth >= 0)
    wet = available & (depth >= wet_depth)
    velocity_ok = (available & (depth >= velocity_depth) & np.isfinite(qx) & np.isfinite(qy))
    denominator = np.maximum(np.where(available, depth, velocity_depth), velocity_depth)
    ux = np.divide(qx, denominator, out=np.zeros_like(depth), where=velocity_ok)
    uy = np.divide(qy, denominator, out=np.zeros_like(depth), where=velocity_ok)
    values = {'valid_cells': available.sum(1), 'wet_cells': wet.sum(1),
              'depth_sum': np.where(available, depth, 0).sum(1, dtype=np.float64),
              'depth_sq_sum': np.square(np.where(available, depth, 0)).sum(1, dtype=np.float64),
              'velocity_cells': velocity_ok.sum(1),
              'speed_sum': np.sqrt(ux * ux + uy * uy).sum(1, dtype=np.float64)}
    for key, value in values.items():
        state[key][tile] += value
    for lag, sums in changes.items():
        pair = available[:-lag] & available[lag:]
        union = pair & (wet[:-lag] | wet[lag:])
        common_velocity = velocity_ok[:-lag] & velocity_ok[lag:]
        delta = np.where(union, depth[lag:] - depth[:-lag], 0)
        du = np.where(common_velocity, ux[lag:] - ux[:-lag], 0)
        dv = np.where(common_velocity, uy[lag:] - uy[:-lag], 0)
        values = {'pair_cells': pair.sum(1), 'wet_union_cells': union.sum(1),
                  'common_wet_velocity_cells': common_velocity.sum(1),
                  'depth_abs_sum': np.abs(delta).sum(1, dtype=np.float64),
                  'depth_sq_sum': np.square(delta).sum(1, dtype=np.float64),
                  'velocity_vector_sq_sum': (du * du + dv * dv).sum(1, dtype=np.float64),
                  'newly_wet_cells': (pair & ~wet[:-lag] & wet[lag:]).sum(1),
                  'newly_dry_cells': (pair & wet[:-lag] & ~wet[lag:]).sum(1)}
        for key, value in values.items():
            sums[key][tile] += value


def ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float),
                     where=denominator > 0)


def make_frames(layout, event, state, changes):
    n_tiles, n_times = state['valid_cells'].shape
    blocks = np.array([t['block_index'] for t in layout['stream_tiles']])
    raw_state = pd.DataFrame({key: value.ravel() for key, value in state.items()})
    raw_state['block_index'] = np.repeat(blocks, n_times)
    raw_state['time_index'] = np.tile(np.arange(n_times), n_tiles)
    raw_state['time_hours'] = event['time_start_hours'] + raw_state.time_index * layout['step_hours']
    raw_state['depth_mean_m'] = ratio(raw_state.depth_sum.to_numpy(), raw_state.valid_cells.to_numpy())
    raw_state['wet_fraction'] = ratio(raw_state.wet_cells.to_numpy(), raw_state.valid_cells.to_numpy())
    raw_state['mean_speed_mps'] = ratio(raw_state.speed_sum.to_numpy(), raw_state.velocity_cells.to_numpy())
    frames = []
    for lag, sums in changes.items():
        nt = n_times - lag
        frame = pd.DataFrame({key: value.ravel() for key, value in sums.items()})
        frame['block_index'] = np.repeat(blocks, nt)
        frame['origin_time_index'] = np.tile(np.arange(nt), n_tiles)
        frame['target_time_index'] = frame.origin_time_index + lag
        frame['origin_hours'] = event['time_start_hours'] + frame.origin_time_index * layout['step_hours']
        frame['interval_hours'] = lag * layout['step_hours']
        frame['depth_change_mae_m'] = ratio(frame.depth_abs_sum.to_numpy(), frame.wet_union_cells.to_numpy())
        frame['depth_change_rms_m'] = np.sqrt(ratio(frame.depth_sq_sum.to_numpy(), frame.wet_union_cells.to_numpy()))
        frame['velocity_vector_change_rms_mps'] = np.sqrt(ratio(
            frame.velocity_vector_sq_sum.to_numpy(), frame.common_wet_velocity_cells.to_numpy()))
        frames.append(frame)
    local = pd.concat(frames, ignore_index=True)
    for frame in (raw_state, local):
        frame['event_id'], frame['split'] = event['event_id'], event['split']
    # Aggregate numerators/counts, not averages of differently sized tile averages.
    event_change = local.groupby(['event_id', 'split', 'interval_hours', 'origin_time_index'],
                                 as_index=False)[list(CHANGE_FIELDS)].sum()
    event_change['depth_change_rms_m'] = np.sqrt(ratio(
        event_change.depth_sq_sum.to_numpy(), event_change.wet_union_cells.to_numpy()))
    event_change['velocity_vector_change_rms_mps'] = np.sqrt(ratio(
        event_change.velocity_vector_sq_sum.to_numpy(), event_change.common_wet_velocity_cells.to_numpy()))
    event_state = raw_state.groupby(['event_id', 'split', 'time_index', 'time_hours'],
                                   as_index=False)[list(STATE_FIELDS)].sum()
    event_state['wet_fraction'] = ratio(event_state.wet_cells.to_numpy(), event_state.valid_cells.to_numpy())
    return raw_state, local, event_state, event_change


def verify_sources(layout):
    for value in layout['source_files'].values():
        if footprint(value['path']) != value:
            raise ValueError(f'Preparation input changed: {value["path"]}')


def process_event(root, event_id, max_chunks=None):
    started = time.monotonic()
    layout = json.loads((root / 'LAYOUT.json').read_text())
    verify_sources(layout)
    event = next(e for e in layout['events'] if e['event_id'] == event_id)
    if footprint(event['source']['path']) != event['source']:
        raise ValueError('Corrected NetCDF changed after layout preparation')
    destination = root / ('smoke' if max_chunks else 'events') / event_id
    destination.mkdir(parents=True, exist_ok=False)
    grid = np.load(layout['grid_path'], mmap_mode='r')
    blocks = np.array([t['block_index'] for t in layout['stream_tiles']])
    index_map = np.full(int(grid.max()) + 1, -1, dtype=int)
    index_map[blocks] = np.arange(len(blocks))
    state, changes = new_accumulators(len(blocks), event['n_times'], layout['intervals_steps'])
    processed, covered_cells = 0, np.zeros(len(blocks), dtype=np.int64)
    with netCDF4.Dataset(event['source']['path']) as ds:
        variables = [ds.variables[layout['variable_names'][name]] for name in ('depth', 'component_x', 'component_y')]
        if any(v.shape != (event['n_times'], *grid.shape) for v in variables):
            raise ValueError('Hydraulic array/grid shape mismatch')
        for variable in variables:
            variable.set_var_chunk_cache(16 * 1024**2, 1009, 0.75)
        chunking = variables[0].chunking()
        row_step, col_step = chunking[1:] if chunking != 'contiguous' else (256, 256)
        for r0 in range(0, grid.shape[0], row_step):
            for c0 in range(0, grid.shape[1], col_step):
                patch = np.asarray(grid[r0:r0 + row_step, c0:c0 + col_step])
                positions = np.full(patch.shape, -1, dtype=int)
                valid = patch >= 0
                positions[valid] = index_map[patch[valid]]
                selected = positions >= 0
                if not selected.any():
                    continue
                # Each source spatial chunk is read once per variable across all times.
                arrays = []
                for variable in variables:
                    values = variable[:, r0:r0 + row_step, c0:c0 + col_step]
                    if np.ma.isMaskedArray(values):
                        values = values.filled(np.nan)
                    arrays.append(np.asarray(values[:, selected], dtype=np.float32))
                    del values
                tile_positions = positions[selected]
                for tile in np.unique(tile_positions):
                    mask = tile_positions == tile
                    accumulate_pixels(state, changes, int(tile), *(a[:, mask] for a in arrays),
                                      layout['wet_depth_m'], layout['velocity_min_depth_m'])
                    covered_cells[tile] += int(mask.sum())
                processed += 1
                if processed % 10 == 0:
                    print(f'{event_id}: {processed} spatial chunks; {time.monotonic()-started:.1f}s', flush=True)
                if max_chunks and processed >= max_chunks:
                    break
            if max_chunks and processed >= max_chunks:
                break
    expected_cells = np.array([t['valid_cells'] for t in layout['stream_tiles']])
    if not max_chunks and not np.array_equal(covered_cells, expected_cells):
        raise ValueError('Scan did not cover every valid stream-tile cell exactly once')
    frames = make_frames(layout, event, state, changes)
    for name, frame in zip(('tile_state', 'tile_changes', 'event_state', 'event_changes'), frames):
        frame.to_parquet(destination / f'{name}.parquet', index=False)
    report = dict(status='smoke_passed' if max_chunks else 'complete', task_id=TASK_ID,
                  event_id=event_id, split=event['split'], source=event['source'],
                  layout_sha256=digest_file(root / 'LAYOUT.json'),
                  covered_cells=covered_cells.tolist(), expected_cells=expected_cells.tolist(),
                  spatial_chunks=processed, seconds=time.monotonic()-started,
                  n_tile_change_rows=len(frames[1]), thresholds_selected=False)
    write_new_json(destination / 'PREPARATION.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'event_id', 'spatial_chunks', 'seconds')}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest='mode', required=True)
    initialize_parser = sub.add_parser('layout')
    initialize_parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    event_parser = sub.add_parser('event')
    event_parser.add_argument('--event-id', required=True)
    event_parser.add_argument('--max-spatial-chunks', type=int)
    args = parser.parse_args()
    if args.mode == 'layout':
        initialize(args.root, args.config)
    else:
        if args.max_spatial_chunks is not None and args.max_spatial_chunks < 1:
            parser.error('Smoke chunk count must be positive')
        process_event(args.root, args.event_id, args.max_spatial_chunks)


if __name__ == '__main__':
    main()
