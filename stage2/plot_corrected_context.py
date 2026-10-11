"""Plot corrected baseline, halo, and halo+graph comparisons from the saved CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


MODELS = {
    'persistence': ('Persistence', '#777777', '--'),
    'baseline': ('Direct baseline', '#0072B2', '-'),
    'halo': ('Halo', '#D55E00', '-'),
    'halo_graph': ('Halo + graph', '#009E73', '-'),
}
PANELS = [
    ('depth_wet_rmse', 'Wet-depth RMSE', 'RMSE (m)'),
    ('wet_f1', 'Wet F1', 'F1'),
    ('component_rmse', 'HU/HV RMSE', 'RMSE (m²/s)'),
    ('derived_velocity_rmse', 'Derived velocity RMSE', 'RMSE (m/s)'),
]
CAPTION = ('Fold 0 · seed 45 · eight held-out events · sampled evaluation · forcing known through target time\n'
           '48 half-hour leads; validation-selected checkpoints. One run per model; no uncertainty bands.')


def read_comparison(path):
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if {row['model'] for row in rows} != set(MODELS):
        raise ValueError('Expected persistence, baseline, halo, and halo_graph')
    data = {}
    for model in MODELS:
        selected = sorted((row for row in rows if row['model'] == model),
                          key=lambda row: float(row['lead_hours']))
        leads = np.array([float(row['lead_hours']) for row in selected])
        if not np.array_equal(leads, np.arange(1, 49) / 2):
            raise ValueError(f'{model}: expected one row for each of 48 half-hour leads')
        data[model] = {key: np.array([float(row[key]) for row in selected])
                       for key in ('lead_hours', 'physical_score', *(p[0] for p in PANELS))}
        if not all(np.isfinite(values).all() for values in data[model].values()):
            raise ValueError(f'{model}: nonfinite values')
    return data


def style_axis(axis):
    axis.set_xlim(0.5, 24)
    axis.set_xticks([0.5, 3, 6, 12, 18, 24])
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.spines[['top', 'right']].set_visible(False)
    axis.tick_params(length=3)


def finish(figure, axes, title, destination, stem, legend_models):
    figure.suptitle(title, fontsize=18, fontweight='bold', y=0.975)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles[:legend_models], labels[:legend_models], loc='upper center',
                  bbox_to_anchor=(0.5, 0.925), ncol=legend_models, frameon=False)
    figure.text(0.5, 0.025, CAPTION, ha='center', fontsize=9, color='#555555')
    figure.tight_layout(rect=(0, 0.095, 1, 0.855), h_pad=2.1, w_pad=2.2)
    for suffix in ('png', 'pdf'):
        figure.savefig(destination / f'{stem}.{suffix}', dpi=220, facecolor='white')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=Path(
        'results/stage2_corrected_context/20261009_fold0/comparison/comparison.csv'))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    destination = args.output_dir or args.csv.parent / 'plots'
    destination.mkdir(parents=True, exist_ok=True)
    data = read_comparison(args.csv)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.labelsize': 11, 'axes.titlesize': 12,
                         'pdf.fonttype': 42})
    leads = data['baseline']['lead_hours']

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    for index, (axis, (metric, title, label)) in enumerate(zip(axes.flat, PANELS)):
        for model, (name, color, linestyle) in MODELS.items():
            axis.plot(leads, data[model][metric], label=name, color=color,
                      linestyle=linestyle, linewidth=2)
        direction = 'higher is better' if metric == 'wet_f1' else 'lower is better'
        axis.set_title(f'{chr(65 + index)}   {title} · {direction}', loc='left')
        axis.set_ylabel(label)
        if metric == 'wet_f1':
            axis.set_ylim(0.90, 1.002)
        else:
            axis.set_ylim(bottom=0)
        style_axis(axis)
    for axis in axes[-1]:
        axis.set_xlabel('Forecast lead (hours)')
    finish(figure, axes, 'Corrected-data forecasts: performance across lead time',
           destination, 'absolute_metrics', 4)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    gain_rows = []
    for index, (axis, (metric, title, _)) in enumerate(zip(axes.flat, PANELS)):
        for model in ('halo', 'halo_graph'):
            baseline, candidate = data['baseline'][metric], data[model][metric]
            if metric == 'wet_f1':
                gain = 100 * (candidate - baseline)
                unit = 'F1 change (percentage points)'
            else:
                if np.any(baseline <= 0):
                    raise ValueError(f'{metric}: baseline RMSE must be positive')
                gain = 100 * (baseline - candidate) / baseline
                unit = 'RMSE reduction (%)'
            name, color, linestyle = MODELS[model]
            axis.plot(leads, gain, label=name, color=color,
                      linestyle=linestyle, linewidth=2)
            gain_rows.extend(dict(lead_hours=float(lead), model=model, metric=metric,
                                  improvement=float(value), unit=unit)
                             for lead, value in zip(leads, gain))
        axis.axhline(0, color='#555555', linewidth=1, linestyle='--')
        axis.set_title(f'{chr(65 + index)}   {title}', loc='left')
        axis.set_ylabel(unit)
        style_axis(axis)
    for axis in axes[-1]:
        axis.set_xlabel('Forecast lead (hours)')
    finish(figure, axes, 'Gains over the direct baseline (positive = improvement)',
           destination, 'gains_vs_baseline', 2)
    with (destination / 'gains_vs_baseline.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gain_rows[0]))
        writer.writeheader()
        writer.writerows(gain_rows)

    figure, axes = plt.subplots(1, 1, figsize=(11, 6), squeeze=False)
    axis = axes[0, 0]
    for model, (name, color, linestyle) in MODELS.items():
        axis.plot(leads, data[model]['physical_score'], label=name, color=color,
                  linestyle=linestyle, linewidth=2)
    axis.set_ylabel('Physical score (lower is better)')
    axis.set_xlabel('Forecast lead (hours)')
    axis.set_ylim(bottom=0)
    style_axis(axis)
    finish(figure, axes, 'Supplementary: combined physical score',
           destination, 'physical_score', 4)
    (destination / 'PLOTS.md').write_text(
        '# Corrected-data comparison plots\n\n'
        'Source: [comparison.csv](../comparison.csv). Fold 0, seed 45; sampled held-out '
        'evaluation of eight events. Known forcing through target time. All 48 leads '
        'are plotted without smoothing; aggregate CSV data provide no uncertainty bands.\n\n'
        '## Absolute metrics\n\n![Absolute metrics](absolute_metrics.png)\n\n'
        '[PDF](absolute_metrics.pdf). Lower RMSE and higher wet F1 are better. '
        'Velocity RMSE pools derived x/y components on true wet cells; it is not speed-magnitude RMSE.\n\n'
        '## Gains over baseline\n\n![Gains over baseline](gains_vs_baseline.png)\n\n'
        '[PDF](gains_vs_baseline.pdf) · [Plotted values](gains_vs_baseline.csv). '
        'Positive means improvement: RMSE reduction is 100 × (baseline − candidate) / baseline; '
        'F1 change is 100 × (candidate − baseline), in percentage points.\n\n'
        '## Supplementary physical score\n\n![Physical score](physical_score.png)\n\n'
        '[PDF](physical_score.pdf). This combined score can hide tradeoffs between outputs.\n\n'
        'Graph adds parameters and neighborhood computation, so gains cannot be attributed '
        'solely to communication. These single-fold, single-seed results do not establish '
        'an overall winner across outputs and leads.\n\n'
        'Reproduce from the repository root:\n\n'
        '```bash\npython -m stage2.plot_corrected_context\n```\n')
    print(f'Saved three figures as PNG/PDF, gain values, and PLOTS.md in {destination}')


if __name__ == '__main__':
    main()
