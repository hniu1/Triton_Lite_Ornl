# Project Guide

This is the starting point for working in this repository. The active research
path is the timestamp-conditioned **Stage-1 model**. The blockwise peak model is
retained as a legacy baseline, not as the current development target.

## Where things live

| Location | Purpose |
|---|---|
| `workflows/stage1/` | Ordered Slurm jobs for the active Stage-1 pipeline |
| `workflows/blockwise/` | Legacy event-peak baseline jobs |
| `workflows/uncertainty/` | Optional ensemble uncertainty job |
| `workflows/stage1_transition/` | State-aware Transition V1 smoke and training jobs |
| `data_preprocessing/` | Data creation and sampling-index tools |
| `plot/` | Plotting and comparison utilities |
| `tests/` | Unit and smoke tests |
| `docs/stage1/` | Stage-1 architecture and experiment documentation |
| `docs/project/` | Historical decisions, challenges, and validation plans |
| `processed_data/` | Generated model inputs; do not edit manually |
| `results/` | Checkpoints, metrics, predictions, and figures |
| `slurm_output/` | Scheduler stdout and stderr logs |

The Python modules remain at repository root because they import each other as
top-level modules and existing checkpoints/jobs rely on that execution layout.
They are grouped below by role so it is clear which entry point to use.

## Current operational workflow

The currently trained run is `results/stage1_timestamp_max`. Its D030 gated
whole-area evaluation is under
`results/stage1_timestamp_max/D030_whole_area_interval20_gated/`.

The normal post-training sequence is:

```bash
sbatch workflows/stage1/06_stage1_evaluate_and_plot_max.sh
sbatch workflows/stage1/06b_stage1_calibrate_wet_threshold.sh
sbatch workflows/stage1/09_stage1_D030_whole_area_interval20.sh
```

For a dependency-safe calibration and inference submission:

```bash
calibration_job=$(sbatch --parsable workflows/stage1/06b_stage1_calibrate_wet_threshold.sh)
sbatch --dependency=afterok:${calibration_job} workflows/stage1/09_stage1_D030_whole_area_interval20.sh
```

## Stage-1 workflow order

### Data/index preparation

1. `workflows/stage1/01_stage1_build_manifest.sh`
2. `workflows/stage1/04_stage1_build_dense_sampling_index.sh`
3. `workflows/stage1/04b_stage1_merge_dense_sampling_index.sh`
4. `workflows/stage1/04c_stage1_dense_sampling_diagnostics.sh`

The older `01b`, `02`, `02b`, and `02c` jobs reproduce the original sparse and
stratified sampling experiments. They are retained for comparison but should
not be the default for the next model.

### Training

- `workflows/stage1/05_stage1_train_max.sh`: current maximum-performance run.
- `workflows/stage1/02_stage1_train.sh`: earlier stratified baseline.

For the planned operational-model update, create a new versioned training job
instead of modifying a completed experiment in place.

The first controlled V4 update is documented in
`docs/stage1/STAGE1_V4_FINE_TUNING.md`:

- `workflows/stage1/10_stage1_v4_sampling_diagnostics.sh` verifies strict quotas.
- `workflows/stage1/11_stage1_v4_finetune.sh` runs the five-epoch warm-start ablation.

### Evaluation and inference

- `workflows/stage1/06_stage1_evaluate_and_plot_max.sh`: sampled evaluation and representative plots.
- `workflows/stage1/06b_stage1_calibrate_wet_threshold.sh`: validation-only wet-threshold calibration.
- `workflows/stage1/07_stage1_fair_compare_and_predictions.sh`: matched comparison with the prior run.
- `workflows/stage1/08_stage1_depth_bin_analysis.sh`: depth/speed metrics by depth bin.
- `workflows/stage1/09_stage1_D030_whole_area_interval20.sh`: gated whole-domain D030 maps.

## Python entry points

### Active Stage-1

| File | Role |
|---|---|
| `stage1_train.py` | Train and select a checkpoint |
| `stage1_evaluate.py` | Evaluate a checkpoint independently |
| `stage1_predict.py` | Predict selected timestamps and blocks |
| `stage1_whole_area_inference.py` | Reconstruct whole-domain maps |
| `stage1_calibrate_wet_threshold.py` | Calibrate the wet-head threshold on validation data |
| `stage1_depth_bin_evaluate.py` | Depth, speed, and direction diagnostics |
| `stage1_sampling_diagnostics.py` | Verify realized sampler distributions |
| `stage1_data.py` | Dataset, normalization, and samplers |
| `stage1_model.py` | Timestamp-conditioned neural network |

### Legacy blockwise baseline

`train_blockwise_matrix.py`, `tune_blockwise_matrix.py`, and
`predict_blockwise_matrix.py` are the executable entry points. The
`blockwise_*.py` modules support both that baseline and some shared Stage-1
data utilities, so they must not be deleted merely because the baseline is
inactive.

### Optional uncertainty analysis

`compute_uncertainty_ensemble.py` is launched by
`workflows/uncertainty/compute_uncertainty_ensemble.sh`. It currently targets
the legacy blockwise model and is not yet integrated into Stage-1.

## Documentation map

- `README.md`: model overview and data contracts.
- `docs/stage1/STAGE1_TIMESTAMP_SURROGATE.md`: detailed active architecture.
- `docs/stage1/STAGE1_MAX_PERFORMANCE.md`: current completed training experiment.
- `docs/stage1/STAGE1_V4_FINE_TUNING.md`: implemented controlled V4 ablation.
- `docs/stage1/OPERATIONAL_SURROGATE_DIRECTION.md`: data sufficiency,
  state-transition architecture decision, branch strategy, and fast-win plan.
- `docs/stage1/STAGE1_TRANSITION_V1.md`: implemented one-step transition prototype.
- `docs/stage1/PROGRESS_SINCE_GATED_BASELINE.md`: report- and slide-ready change
  log and performance progression from the gated baseline through Transition V1.
- `docs/stage1/TRITON_COMPONENT_SEMANTICS_AUDIT.md`: source-code evidence that
  the stored U/V targets are unit discharge (HU/HV), not velocity.
- `results/stage1_timestamp_max/D030_whole_area_interval20_gated/RESULTS_SUMMARY.md`:
  gated results and prioritized improvement plan.
- `docs/project/`: historical status and publication/validation notes.

## Rules for new experiments

1. Give each experiment a new result directory; never overwrite a completed run.
2. Add one numbered Slurm job under `workflows/stage1/`.
3. Record the exact configuration in the result directory.
4. Calibrate thresholds on validation events only.
5. Keep D030 and other declared test events out of training and calibration.
6. Run whole-domain and phase-specific evaluation before declaring improvement.
