"""Summarize observed hydraulic change distributions without selecting a test cohort."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stage2.prepare_dynamic_stream import DEFAULT_ROOT, digest_file, verify_sources


METRICS = ('depth_change_rms_m', 'velocity_vector_change_rms_mps')


def quantile_summary(frame):
    rows = []
    for (split, interval), group in frame.groupby(['split', 'interval_hours']):
        for metric in METRICS:
            values = group[metric].to_numpy()
            finite = values[np.isfinite(values)]
            positive = finite[finite > 0]
            row = dict(split=split, interval_hours=interval, metric=metric,
                       total_pairs=len(values), eligible_pairs=len(finite),
                       positive_pairs=len(positive), unavailable_pairs=len(values)-len(finite))
            for population, array in (('all_eligible', finite), ('positive', positive)):
                for percentile in (10, 25, 50, 75, 90, 95, 99):
                    row[f'{population}_p{percentile}'] = float(np.percentile(array, percentile)) if len(array) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def candidate_coverage(frame):
    """Illustrate joint depth/velocity cutoffs derived from training only."""
    rows = []
    for interval in sorted(frame.interval_hours.unique()):
        train = frame[(frame.split == 'train') & (frame.interval_hours == interval)]
        for percentile in (50, 75, 90, 95):
            thresholds = {}
            for metric in METRICS:
                values = train[metric].to_numpy()
                values = values[np.isfinite(values) & (values > 0)]
                thresholds[metric] = float(np.percentile(values, percentile)) if len(values) else np.nan
            for (event_id, split), group in frame[frame.interval_hours == interval].groupby(['event_id', 'split']):
                joint = np.ones(len(group), dtype=bool)
                for metric in METRICS:
                    joint &= np.isfinite(group[metric]) & (group[metric] > 0) & (group[metric] >= thresholds[metric])
                # Illustrate event-time breadth; no time gate is chosen here.
                per_time = pd.DataFrame(dict(origin=group.origin_time_index.to_numpy(), selected=joint))
                fractions = per_time.groupby('origin').selected.mean()
                rows.append(dict(event_id=event_id, split=split, interval_hours=interval,
                                 training_positive_percentile=percentile,
                                 depth_threshold_m=thresholds[METRICS[0]],
                                 velocity_vector_threshold_mps=thresholds[METRICS[1]],
                                 total_tile_time_pairs=len(group), joint_change_pairs=int(joint.sum()),
                                 joint_fraction=float(joint.mean()),
                                 times_with_joint_change=int((fractions > 0).sum()),
                                 max_fraction_tiles_jointly_changing=float(fractions.max())))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root
    layout = json.loads((root / 'LAYOUT.json').read_text())
    train_count = sum(event['split'] == 'train' for event in layout['events'])
    val_count = sum(event['split'] == 'val' for event in layout['events'])
    verify_sources(layout)
    layout_digest = digest_file(root / 'LAYOUT.json')
    local_frames, state_frames, event_frames, reports = [], [], [], []
    fields = ['event_id', 'split', 'interval_hours', 'origin_time_index', 'block_index',
              'wet_union_cells', 'common_wet_velocity_cells', 'newly_wet_cells', 'newly_dry_cells', *METRICS]
    for event in layout['events']:
        directory = root / 'events' / event['event_id']
        report = json.loads((directory / 'PREPARATION.json').read_text())
        if (report['status'] != 'complete' or report['layout_sha256'] != layout_digest
                or report['event_id'] != event['event_id'] or report['split'] != event['split']
                or report['source'] != event['source']
                or report['covered_cells'] != report['expected_cells']):
            raise ValueError(f'Incomplete/mismatched preparation for {event["event_id"]}')
        local = pd.read_parquet(directory / 'tile_changes.parquet', columns=fields)
        expected = layout['n_stream_tiles'] * sum(event['n_times'] - lag for lag in layout['intervals_steps'])
        if len(local) != expected or set(local.event_id) != {event['event_id']}:
            raise ValueError('Incomplete tile/time series')
        if local.duplicated(['event_id', 'block_index', 'interval_hours', 'origin_time_index']).any():
            raise ValueError('Duplicate tile/time change records')
        local_frames.append(local)
        state_frames.append(pd.read_parquet(directory / 'event_state.parquet'))
        event_frames.append(pd.read_parquet(directory / 'event_changes.parquet'))
        reports.append({k: report[k] for k in ('event_id', 'split', 'seconds', 'spatial_chunks')})
    local = pd.concat(local_frames, ignore_index=True)
    if set(local.event_id) & set(layout['held_out_events_not_processed']):
        raise ValueError('Held-out test data entered threshold characterization')
    quantiles = quantile_summary(local)
    coverage = candidate_coverage(local)
    event_states = pd.concat(state_frames, ignore_index=True)
    event_changes = pd.concat(event_frames, ignore_index=True)
    destination = root / 'summary'
    destination.mkdir(exist_ok=False)
    quantiles.to_csv(destination / 'change_quantiles.csv', index=False)
    coverage.to_csv(destination / 'candidate_joint_coverage.csv', index=False)
    event_states.to_csv(destination / 'event_state_timeseries.csv', index=False)
    event_changes.to_csv(destination / 'event_change_timeseries.csv', index=False)
    pd.DataFrame(reports).to_csv(destination / 'processing_times.csv', index=False)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for row, interval in enumerate(sorted(local.interval_hours.unique())):
        subset = event_changes[event_changes.interval_hours == interval]
        for column, metric in enumerate(METRICS):
            axis = axes[row, column]
            for split, color, label in (('train', '#0072B2', 'Training'), ('val', '#D55E00', 'Validation')):
                group = subset[subset.split == split]
                for _, event in group.groupby('event_id'):
                    axis.plot(event.origin_time_index * layout['step_hours'] + 0.5,
                              event[metric], color=color, alpha=0.15, linewidth=0.7)
                median = group.groupby('origin_time_index')[metric].median()
                axis.plot(median.index * layout['step_hours'] + 0.5, median,
                          color=color, linewidth=2, label=f'{label} event median')
            axis.set_title(f'Change over {interval:g} h', loc='left')
            axis.set_ylabel('Wet-union depth change RMS (m)' if column == 0 else 'Common-wet velocity-vector change RMS (m/s)')
            axis.set_ylim(bottom=0)
            axis.grid(alpha=0.2)
            axis.spines[['top', 'right']].set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel('Event-relative origin time (hours)')
    figure.suptitle('Observed hydraulic changes on stream-containing tiles', fontsize=16, fontweight='bold')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False)
    figure.text(0.5, 0.025, f'{layout["n_stream_tiles"]} stream tiles · {train_count} training / {val_count} validation events · thin lines: individual events\n'
                'Observed state changes, not forecast errors. Velocity uses cells with depth ≥0.1 m at both endpoints.',
                ha='center', fontsize=9)
    figure.tight_layout(rect=(0, 0.065, 1, 0.91))
    for suffix in ('png', 'pdf'):
        figure.savefig(destination / f'observed_change_timeseries.{suffix}', dpi=180)
    plt.close(figure)

    lines = ['# Dynamic Flood Evaluation on Stream Tiles — preparation summary', '',
             'Observed hydraulic changes characterized using training and validation events only.',
             'No final thresholds, evaluation cohort, model scores or test-event processing are produced.', '',
             '![Observed changes](observed_change_timeseries.png)', '',
             '[PDF](observed_change_timeseries.pdf) · [Change quantiles](change_quantiles.csv) · '
             '[Illustrative joint-threshold coverage](candidate_joint_coverage.csv)', '',
             '## Definitions', '',
             '- Depth change RMS: root mean squared depth difference on cells wet at either endpoint (≥0.05 m).',
             '- Velocity-vector change RMS: root mean squared Euclidean velocity-vector difference; '
             'depth ≥0.1 m and finite HU/HV required at both endpoints. Newly wet/dry cells are counted separately.',
             '- Event curves aggregate pixel squared differences and eligible counts over the 258 stream tiles; '
             'they do not summarize the entire 6,900-tile domain.',
             '- Candidate coverage uses 50th/75th/90th/95th percentiles of positive training changes '
             'for depth and velocity jointly. These are illustrative choices, not adopted thresholds.',
             '- Per-tile state/change Parquet files retain every half-hour state and all three intervals. '
             'Origins near the end or at index zero remain diagnostic data; later forecast eligibility '
             'must separately require history and target availability.', '',
             '## Next decision', '',
             'Choose a change interval, event-time gate, minimum affected-cell support, and depth/velocity '
             'thresholds. Then freeze the selection rule before processing held-out test events.', '']
    (destination / 'SUMMARY.md').write_text('\n'.join(lines))
    with (destination / 'PREPARATION_SUMMARY.json').open('x') as f:
        json.dump(dict(status='characterization_complete', task_id=layout['task_id'],
                       layout_sha256=layout_digest, train_events=train_count, validation_events=val_count,
                       test_events_processed=0, stream_tiles=layout['n_stream_tiles'],
                       tile_change_pairs=len(local), thresholds_selected=False,
                       per_event_processing=reports), f, indent=2)
        f.write('\n')
    print(f'Wrote observed-change summaries to {destination}', flush=True)


if __name__ == '__main__':
    main()
