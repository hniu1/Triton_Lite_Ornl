"""Freeze training-only rules, select held-out targets, and validate shared cohorts."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage2.prepare_dynamic_stream import digest_file, footprint, write_new_json, verify_sources

ROOT = Path('results/stage2_stream_flood/20261011_fold0')


def freeze(root):
    prep = root / 'preparation'
    layout = json.loads((prep / 'LAYOUT.json').read_text())
    verify_sources(layout)
    event_frames, tile_frames = [], []
    for event in layout['events']:
        if event['split'] != 'train':
            continue
        folder = prep / 'events' / event['event_id']
        for filename, frames in [('event_changes.parquet', event_frames), ('tile_changes.parquet', tile_frames)]:
            frame = pd.read_parquet(folder / filename)
            frames.append(frame[frame.interval_hours == 3])
    thresholds = {}
    for scope, frames in [('event', event_frames), ('tile', tile_frames)]:
        frame = pd.concat(frames)
        for field in ('depth_change_rms_m', 'velocity_vector_change_rms_mps'):
            values = frame[field].to_numpy()
            values = values[np.isfinite(values) & (values > 0)]
            thresholds[f'{scope}_{field}'] = float(np.quantile(values, .75))
    rule = dict(task_id=layout['task_id'], training_layout_sha256=digest_file(prep / 'LAYOUT.json'),
                threshold_source='positive training-only 75th percentiles', interval_hours=3,
                thresholds=thresholds, minimum_support_fraction=.01, minimum_target_index=49,
                horizons_steps=list(range(1, 49)), max_times_per_event=10, max_tiles_per_time=8,
                sampling='Evenly spaced ranks of sorted eligible target times; evenly spaced ranks of sorted eligible block indices at each selected time.',
                selection='Both event depth/velocity gates AND both tile depth/velocity gates; stream tiles only.',
                retrospective=True)
    write_new_json(root / 'RULE.json', rule)
    config = json.loads(Path(layout['config']).read_text())
    manifest = pd.read_parquet(layout['source_files']['manifest']['path'])
    test_layout = copy.deepcopy(layout)
    test_layout['events'] = []
    for event_id in config['split_events']['test']:
        rows = manifest[manifest.event_id == event_id]
        if len(rows) != 1:
            raise ValueError(f'Missing test event {event_id}')
        row = rows.iloc[0]
        test_layout['events'].append(dict(event_id=event_id, split='test', n_times=int(row.n_times),
                                         time_start_hours=float(row.time_start), source=footprint(row.path_to_netcdf)))
    test_layout['held_out_events_not_processed'] = []
    test_layout['frozen_rule_sha256'] = digest_file(root / 'RULE.json')
    destination = root / 'test_preparation'
    destination.mkdir()
    write_new_json(destination / 'LAYOUT.json', test_layout)
    (destination / 'events.txt').write_text('\n'.join(e['event_id'] for e in test_layout['events']) + '\n')
    print(json.dumps(rule, indent=2))


def select(frame, event_frame, valid_cells, rule):
    t = rule['thresholds']
    events = event_frame[(event_frame.interval_hours == rule['interval_hours']) &
                        (event_frame.depth_change_rms_m >= t['event_depth_change_rms_m']) &
                        (event_frame.velocity_vector_change_rms_mps >= t['event_velocity_vector_change_rms_mps'])]
    # Event summaries store the interval origin; add six half-hour steps.
    times = set((events.origin_time_index + 6).astype(int))
    support = np.ceil(frame.block_index.map(valid_cells) * rule['minimum_support_fraction'])
    eligible = frame[(frame.interval_hours == rule['interval_hours']) &
                     frame.target_time_index.isin(times) &
                     (frame.target_time_index >= rule['minimum_target_index']) &
                     (frame.depth_change_rms_m >= t['tile_depth_change_rms_m']) &
                     (frame.velocity_vector_change_rms_mps >= t['tile_velocity_vector_change_rms_mps']) &
                     (frame.wet_union_cells >= support) & (frame.common_wet_velocity_cells >= support)].copy()
    batches = []
    available = sorted(eligible.target_time_index.unique())
    if not available:
        raise ValueError('No eligible targets; thresholds must not be silently relaxed')
    for rank in np.linspace(0, len(available) - 1, min(len(available), rule['max_times_per_event']), dtype=int):
        target = int(available[rank])
        blocks = sorted(eligible.loc[eligible.target_time_index == target, 'block_index'].unique())
        ranks = np.linspace(0, len(blocks) - 1, min(len(blocks), rule['max_tiles_per_time']), dtype=int)
        batches.append(dict(target_time_index=target, block_indices=[int(blocks[r]) for r in ranks]))
    return eligible, batches


def merge(root, rule_path=None):
    rule_path = rule_path or root / "RULE.json"
    rule = json.loads(rule_path.read_text())
    prep = root / 'test_preparation'
    layout = json.loads((prep / 'LAYOUT.json').read_text())
    if digest_file(root / 'RULE.json') != layout['frozen_rule_sha256']:
        raise ValueError('Frozen rule changed')
    if rule_path != root / "RULE.json":
        if rule.get("parent_rule_sha256") != layout["frozen_rule_sha256"]:
            raise ValueError("Revised rule does not reference original frozen rule")
        original = json.loads((root / "RULE.json").read_text())
        for key in ("thresholds", "interval_hours", "minimum_support_fraction"):
            if rule[key] != original[key]:
                raise ValueError("Change-selection thresholds must remain frozen")
    verify_sources(layout)
    batches, coverage, frames = [], [], []
    counts = {t['block_index']: t['valid_cells'] for t in layout['stream_tiles']}
    for event in layout['events']:
        folder = prep / 'events' / event['event_id']
        status = json.loads((folder / 'PREPARATION.json').read_text())
        if status['status'] != 'complete':
            raise ValueError('Incomplete event preparation')
        if status['layout_sha256'] != digest_file(prep / 'LAYOUT.json'):
            raise ValueError('Preparation identity mismatch')
        eligible, selected = select(pd.read_parquet(folder / 'tile_changes.parquet'),
                                    pd.read_parquet(folder / 'event_changes.parquet'), counts, rule)
        for batch in selected:
            batch['event_id'] = event['event_id']
        batches.extend(selected)
        frames.append(eligible)
        coverage.append(dict(event_id=event['event_id'], eligible_target_times=int(eligible.target_time_index.nunique()),
                             eligible_tile_target_pairs=len(eligible), eligible_tiles=int(eligible.block_index.nunique()),
                             selected_target_times=len(selected), selected_tile_target_pairs=sum(len(b['block_indices']) for b in selected)))
    destination = root / 'cohort'
    destination.mkdir()
    pd.concat(frames).to_parquet(destination / 'eligible_targets.parquet', index=False)
    write_new_json(destination / 'cohort.json', dict(task_id=rule['task_id'], rule=rule,
        rule_sha256=digest_file(rule_path), rule_path=str(rule_path.resolve()), layout_path=str((prep / 'LAYOUT.json').resolve()),
        layout_sha256=digest_file(prep / 'LAYOUT.json'), split='test', coverage=coverage, batches=batches))
    pd.DataFrame(coverage).to_csv(destination / 'coverage.csv', index=False)
    lead_coverage = []
    for lead in rule["horizons_steps"]:
        available = [b for b in batches if b["target_time_index"] - lead >= rule.get("history_states", 2) - 1]
        lead_coverage.append(dict(lead_hours=lead * layout["step_hours"], batches=len(available),
            tile_target_pairs=sum(len(b["block_indices"]) for b in available),
            events=len({b["event_id"] for b in available}), event_ids=" ".join(sorted({b["event_id"] for b in available}))))
    pd.DataFrame(lead_coverage).to_csv(destination / "coverage_by_lead.csv", index=False)
    print(pd.DataFrame(coverage).to_string(index=False), flush=True)


def load_batches(path, dataset, horizons, batch_size):
    cohort = json.loads(path.read_text())
    layout_path = Path(cohort['layout_path'])
    if digest_file(layout_path) != cohort['layout_sha256']:
        raise ValueError('Cohort source identity changed')
    layout = json.loads(layout_path.read_text())
    if "rule_path" in cohort and digest_file(cohort["rule_path"]) != cohort["rule_sha256"]:
        raise ValueError("Evaluation rule changed")
    verify_sources(layout)
    if list(horizons) != cohort['rule']['horizons_steps']:
        raise ValueError('Conditional evaluation requires all frozen leads')
    event_positions = {str(e): i for i, e in enumerate(dataset.events.event_id)}
    block_positions = {int(b): i for i, b in enumerate(dataset.block_rows.block_index)}
    selected, seen = [], set()
    for b in cohort['batches']:
        event, target = b['event_id'], int(b['target_time_index'])
        positions = [block_positions[int(v)] for v in b['block_indices']]
        if not positions or len(positions) > batch_size or len(set(positions)) != len(positions):
            raise ValueError('Invalid conditional batch')
        n_times = int(dataset.events.iloc[event_positions[event]].n_times)
        required_lead = min(horizons) if cohort["rule"].get("lead_eligibility") == "available_origin_history" else max(horizons)
        if target - required_lead < dataset.history_states - 1 or target >= n_times:
            raise ValueError('Target lacks forecast history')
        for block in positions:
            key = (event, target, block)
            if key in seen:
                raise ValueError('Duplicate conditional target')
            seen.add(key)
        selected.append((event_positions[event], target, positions))
    if set(event_positions) != {b['event_id'] for b in cohort['batches']}:
        raise ValueError('Cohort does not cover exactly the evaluation events')
    return selected, cohort


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'merge'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--rule-path', type=Path)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.root)
    else:
        merge(args.root, args.rule_path)
