# Stage-1 Transition V1

Transition V1 is the first fast-win implementation from
`OPERATIONAL_SURROGATE_DIRECTION.md`. It tests whether hydraulic history removes
the depth and drainage failures of the independent-timestamp model.

## Contract

For target timestep `t`, each sample contains:

```text
TRITON depth/component state at t-1
+ forcing history through t
+ target time features
+ block attributes
+ static terrain and mask
→ residual state change and state at t
```

The half-hour lag is configurable but V1 uses `lag=1`.

## Implementation

- `stage1_transition_data.py`: pairs existing netCDF states without duplicating data.
- `stage1_transition_model.py`: state encoder plus residual hydraulic heads.
- `stage1_transition_train.py`: one-step training, persistence baseline, and selection.
- `workflows/stage1_transition/01_transition_v1_smoke.sh`: ten-batch GPU safety run.
- `workflows/stage1_transition/02_transition_v1_train.sh`: five-epoch experiment.
- `workflows/stage1_transition/03_transition_v1_continue.sh`: lower-rate convergence run
  from the selected V1 checkpoint.
- `stage1_transition_rollout.py` and workflow `04`: paired teacher-forced,
  autoregressive, and persistence evaluation at 1, 6, 12, and 24 steps.
- `stage1_transition_whole_area.py` and workflow `05`: raw, wet-gated, and
  persistence reconstruction at active and known D030 drainage-failure times.
- `stage1_transition_multistep_train.py` and workflows `06`-`08`: scheduled
  exposure to predicted states over six steps, selected using fully
  autoregressive three-hour validation and followed by the same horizon test.
- `stage1_transition_regime_eval.py` and workflows `09`-`10`: identical
  stable, filling, draining, and rapid-change diagnostics for the continued
  one-step reference and multi-step candidate.
- `stage1_transition_operational_gate.py`: joint rollout/regime promotion gate
  against exact persistence and the accepted Continued V1 checkpoint.

The residual heads are initialized to zero, so the untrained model begins as a
persistence forecast. Shape-compatible forcing, terrain, conditioning, and
decoder tensors are initialized from the Stage-1 Max checkpoint. The new state
encoder and residual heads are learned on transition pairs.

## Evaluation gate

`metrics.json` records both `test` and `persistence_test`. V1 must beat
persistence before rollout or architecture expansion is justified.

Primary comparisons are:

- wet-depth RMSE;
- velocity/component RMSE;
- wet F1 and CSI;
- composite physical score.

The first job sequence is:

```bash
smoke_job=$(sbatch --parsable workflows/stage1_transition/01_transition_v1_smoke.sh)
sbatch --dependency=afterok:${smoke_job} workflows/stage1_transition/02_transition_v1_train.sh
```

## Remaining limits

The original V1 training stage was teacher-forced and one-step only. The
current extension now provides scheduled six-step training, autoregressive
rollouts, regime-stratified diagnostics, and whole-domain reconstruction. It
still lacks tile halos, cross-block flux conservation, uncertainty calibration,
and training-event diversity beyond the existing single-watershed archive.

The transition-aware V2 design and its deliberately separate workflows are
documented in `STAGE1_TRANSITION_V2.md`.
