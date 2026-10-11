# Corrected-data baseline, fold 0

The 9 October 2026 baseline retrains the existing direct U-Net from random
weights using corrected, validated hydraulic targets. Legacy metrics and
checkpoints do not establish forecasting skill and provide no learned weights
to this run. The baseline and subsequent halo/graph comparison are now complete;
the corrected held-out results and measured Riker runtimes are recorded below.

## Data and fixed experiment

### Slide-ready experimental setup

**Question:** Does hydraulic neighborhood context improve direct flood forecasts
relative to the corrected-data U-Net and full-state persistence?

| Item | Completed corrected-data experiment |
|---|---|
| Domain and data | Conasauga regional domain; 40 TRITON events with corrected chronological ordering and signed HU/HV targets |
| Split | Whole-event fold 0: 25 training / 7 validation / 8 test events; one model seed (45) |
| Forecast inputs | Observed depth, HU/HV at the origin and preceding half-hour; static terrain/block features; forcing known through the requested target time |
| Forecast outputs | Depth, HU, HV and wet probability for the central 80 × 80 tile, approximately 10 m cells; velocity derived from HU/HV and depth |
| Forecast task | Direct origin-to-target prediction at 48 leads: 0.5–24 h; no intermediate hydraulic states supplied |
| Training | Random initialization; 10 epochs × 750 balanced batches; eight scored central tiles per batch; randomly sampled training lead |
| Optimizer and objective | AdamW, learning rate 1e-4; identical depth, wet-extent, discharge, derived-velocity and auxiliary loss settings across arms |
| Checkpoint selection | Validation only, 80 batches at 0.5, 3, 6, 12 and 24 h; mean physical score plus derived-velocity RMSE |
| Held-out evaluation | Same deterministic central-tile cohort at all 48 leads; 80 batches of up to eight centers, with per-event metrics; wet depth threshold 0.05 m, wet-head probability threshold 0.5 |
| Compute | Riker, one NVIDIA L40S GPU per training/evaluation job |

| Comparison arm | Spatial context |
|---|---|
| Persistence | Freeze origin depth, HU and HV at all leads; no training |
| Direct U-Net baseline | Independent tile; no hydraulic halo or graph |
| Halo | Eight-cell hydraulic border (96 × 96 input context), central 80 × 80 output |
| Halo + graph | Same halo plus two message-passing layers over two-hop, four-neighbor raster-tile adjacency; losses and metrics score centers only |

**Suggested slide footnote:** Single-fold, single-seed, sampled held-out
model-development comparison. Future forcing is assumed known. Graph adds
parameters and computation; this is not a matched-capacity communication ablation.

The baseline/halo/halo+graph selected checkpoints are epochs **10/6/10**.
Saved `run_config.json` files in each run directory are the detailed experiment
record, including full event lists, architecture and loss weights. The
older leadership briefing retains legacy results and should not supply the
accuracy numbers for these corrected-data slides.

- All 40 events passed 108 sampled raw comparisons each and the independent
  forcing audit. Calendar epoch interpretation remains unverified; forecast
  leads use the verified half-hour event-relative axis.
- Corrected manifest: `processed_data/stage2_corrected_training/20261009_fold0/manifest/`.
  It resolves directly to the validated NetCDF files, through an isolated set
  of symlinks. Legacy manifest and outputs are preserved.
- The original fold-0 split is preserved: 25 training, 7 validation, 8 test
  events, data seed 4300. Test events are D005, D014, D018, D023, D024, D028,
  D033, and D037. Full lists are in `split_events.json`.
- Sampling uses 50,000 requested candidates per **training event only**, with
  duplicates removed, balanced batches of 8 and the prior category/phase
  quotas. Corrected manifest identity is hashed into the sampling metadata;
  legacy or mismatched sampling is rejected.
- Forcing means and standard deviations are recomputed from the 25 training
  events. Shared block/terrain statistics remain appropriate to the same
  geographic domain. No hydraulic-target normalization is fit or reused;
  depth/discharge transforms are fixed by the existing model/loss formulation.
- Precheck verifies all training events have feasible sampler batches,
  finite normalization, and finite real hydraulic batches at leads 0.5, 3,
  6, 12, and 24 hours. It records normalization and provenance.
- Fresh training reads model architecture from configuration only. It does
  not open a legacy checkpoint. Model seed 45; no halo or graph; ten epochs,
  750 batches per epoch, batch size 8, learning rates 1e-4, existing hydraulic
  loss weights, validation checkpoint selection at five leads.
- All **37 regression tests passed**. Required scheduler requests passed
  test-only validation. The working `triton` Python environment and GNU MKL
  threading setting are used instead of the damaged `triton_andes` environment.

## Submitted dependency chain

**Completion verified on 9 October 2026:** all 25 sampling tasks and jobs
187926–187929 completed with Slurm exit code `0:0`. Training completed all ten
epochs and selected epoch 10. Held-out evaluation completed all 48 leads for
the eight test events. The submission-time table below is retained as history.

| Job | Work | Start condition |
|---|---|---|
| 187917 | 25 training-event sampling shards | Submitted; first four running |
| 187926 | Merge and train-only normalization/data checks | All sampling shards succeed |
| 187927 | GPU forward/backward and one-epoch, 20-batch training smoke | Merge and precheck succeed |
| 187928 | Fresh ten-epoch baseline training | Smoke succeeds |
| 187929 | Held-out evaluation at all 48 half-hour leads through 24 h | Training succeeds |

The sampling array permits eight concurrent tasks, but the scheduler currently
limits this account to four running jobs. Each downstream dependency uses
`afterok` and cancels on an invalid dependency, preventing a failed preparation
or smoke check from launching training/evaluation. GPU queue timing remains
uncertain; submission is not evidence that training has begun.

Both smoke and training skip automatic test evaluation. The final evaluation
uses the validation-selected checkpoint, the learned discharge policy, 80
deterministic sampled origin/tile batches, batch size 8, and per-event reporting,
comparing to persistence. This is sampled held-out evidence, not exhaustive
whole-domain evaluation. Default wet-head threshold 0.5 is fixed in advance.

### Where to find velocity results

`seed_45/test_dense_24h_direct_metrics.json` contains aggregate results under
`metrics`, plus a separate `per_event_metrics` section. Lead key `"48"` is
24 hours. The aggregate `direct.derived_velocity_rmse` is
**0.02628369885 m/s**, versus **0.02290585859 m/s** under
`persistence.derived_velocity_rmse`. The corresponding
`direct_vs_persistence.derived_velocity_rmse_change_percent` is **+14.7466%**.
The metric pools x/y component errors over true wet cells, deriving velocity
from predicted HU/HV divided by predicted depth and target HU/HV divided by
target depth, with depths clamped to the wet threshold. It is not speed-magnitude
RMSE. The evaluation console prints depth and F1 only; read the JSON for velocity.

## Completed comparison: halo and halo + graph

The proposed next diagnostic is documented in
[dynamic-flood evaluation on stream tiles](STREAM_FLOOD_EVALUATION.md). The
user requested substantial event/time and local depth/velocity changes;
selection criteria are under discussion and no new jobs are submitted.

On 9 October 2026 the user authorized training both context variants and
comparing them with the corrected baseline. Each starts from random weights,
using the same corrected manifest, sampling, fold-0 split, seed 45, loss,
ten epochs × 750 batches × eight scored central tiles, optimizer settings,
five-lead validation selection, and 80-batch/48-lead held-out evaluation.

- **Halo:** existing eight-cell hydraulic halo encoder, no graph.
- **Halo + graph:** the same halo, two adjacency-message layers and two-hop
  raster-tile neighborhoods. Context tiles supply inputs; only central tiles
  contribute to losses and reported metrics. This is a local tile graph,
  not the deferred hierarchical HUC12 graph. The existing halo implementation
  encodes the larger input before cropping its features to the central tile;
  this is not a new full-context-through-all-encoder-stages architecture.

This user-authorized corrected-data experiment replaces the old legacy-data
screening gate for these two runs. No legacy diagnostic or learned weights are
used. A separate smoke per arm runs one epoch with 20 real training batches at
the full eight-center batch size, eight validation batches, and no test batches.
It requires finite results/weights, a validation-selected checkpoint, and a
nonzero graph projection for the graph arm. It checks execution and learning
updates, not forecast quality. Full training depends on smoke success.

| Arm | Smoke | Training | Held-out evaluation |
|---|---|---|---|
| Halo | 188117 | 188118 | 188119 |
| Halo + graph | 188120 | 188121 | 188122 |

**Completion verified on 11 October 2026:** all seven jobs 188117–188123
are `COMPLETED` with exit code `0:0`, including both smoke checks, both training
runs, both held-out evaluations, and the final comparison. Both models completed
all ten epochs. Validation selected epoch 6 for halo and epoch 10 for halo + graph.
Downstream jobs used `afterok` and cancellation on invalid dependencies.

### Held-out results and runtime

The comparison report checks matching evaluation settings and persistence
metrics at all 48 leads. Results below are aggregate sampled test metrics at
**24 hours**; lower RMSE and higher F1 are better.

| Model | Wet-depth RMSE (m) | Wet F1 | HU/HV RMSE (m²/s) | Derived velocity RMSE (m/s) |
|---|---|---|---|---|
| Persistence | 0.394970 | 0.904722 | 0.049936 | 0.022906 |
| Direct baseline | 0.311821 | 0.909993 | 0.049404 | 0.026284 |
| Halo | 0.284225 | 0.920416 | 0.049360 | 0.028320 |
| Halo + graph | 0.328463 | 0.914553 | 0.049684 | 0.027243 |

Halo has the lowest 24-hour depth error and highest wet F1 among these models,
but its derived velocity error is higher than the baseline's. Halo + graph
does not improve 24-hour depth over either halo or baseline. Its depth error
at 6 hours is lower (0.078013 m, versus halo 0.087975 m and baseline 0.098715 m),
so the comparison varies by lead. Both context models worsen short-lead depth:
at 0.5 hours, baseline/halo/halo+graph RMSE is 0.061850/0.149238/0.123550 m.
All three learned models have higher 24-hour velocity error than persistence.
These results do not establish an overall winner across all outputs and leads.

| Model | Training runtime | Held-out evaluation runtime |
|---|---|---|
| Direct baseline | 1:41:23 | 0:26:50 |
| Halo | 1:56:52 | 0:34:58 |
| Halo + graph | 14:20:34 | 3:18:34 |

Graph training took approximately 7.4 times the halo-only runtime.

See the [comparison report](../../results/stage2_corrected_context/20261009_fold0/comparison/COMPARISON.md)
for selected leads and [comparison CSV](../../results/stage2_corrected_context/20261009_fold0/comparison/comparison.csv)
for all 48 leads. Each arm's `test_dense_24h_direct_metrics.json` also contains
per-event results. No additional training is established by this comparison.

The [plot guide](../../results/stage2_corrected_context/20261009_fold0/comparison/plots/PLOTS.md)
includes four-panel absolute metrics, gains over the direct baseline, and a
supplementary physical-score curve. All figures are available as PNG and PDF.
Reproduce them with `python -m stage2.plot_corrected_context` from the repository
root. Positive gains mean improvement; F1 changes use percentage points and
RMSE changes use relative percent reduction. No uncertainty bands are inferred
from the single-fold, single-seed aggregate CSV.

Workflows: `25_corrected_context_train.sh`, parameterized
`24_corrected_baseline_eval.sh`, and `26_corrected_context_compare.sh`.
Outputs: `results/stage2_corrected_context/20261009_fold0/{halo,halo_graph}/seed_45/`.
Smoke outputs are separate `smoke_job_<id>/` directories. The comparison writes
`comparison/COMPARISON.md` and `comparison/comparison.csv`, including derived
velocity RMSE. It checks matching settings and persistence metrics to verify
cohort consistency. Per-event results remain in each evaluation JSON.

This is a single-seed, single-fold comparison of model packages. Graph adds
parameters and neighborhood computation; gains do not independently establish
the value of communication versus added capacity. Old architecture results
remain invalid for corrected-data scientific claims.

Fresh outputs: `results/stage2_corrected_baseline/20261009_fold0/seed_45/`.
Per-epoch checkpoints, normalization, `training_progress.json`, and timings
are preserved. The smoke output is separate and has no held-out test results.

Logs:

```bash
tail -f slurm_output/stage2-corrected-sampling-187917_1.err
tail -f slurm_output/stage2-corrected-merge-187926.out
tail -f slurm_output/stage2-corrected-baseline-187927.out
tail -f slurm_output/stage2-corrected-baseline-187928.out
tail -f slurm_output/stage2-corrected-eval-187929.out
```

Inspect the matching `.err` files too; Python logging uses stderr. For Monday's
slides, report verified data repair, the frozen experiment, and newly obtained
evaluation evidence. Claim forecasting improvement only if corrected held-out
results support it; do not carry forward legacy performance claims.
