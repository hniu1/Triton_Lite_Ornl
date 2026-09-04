# Progress Since the Gated Stage-1 Baseline

## Executive summary

The gated Stage-1 timestamp model improved flood-extent classification, but it
did not solve depth, flow, or drainage-state errors. Subsequent experiments
showed that stricter sampling and stronger losses alone were insufficient. We
therefore changed the model formulation from independent timestamp prediction
to hydraulic state transition:

```text
Previous hydraulic state + forcing history + terrain
→ change in hydraulic state → next hydraulic state
```

The first completed Transition V1 experiment is the strongest result so far.
On held-out event D030, it reduced one-step wet-depth RMSE by 14.3% relative to
an exact persistence baseline and by approximately 21% relative to the sampled
Stage-1 Max evaluation. It also reduced the composite physical score by 12.4%
relative to persistence. However, wet F1 was slightly worse than persistence,
and autoregressive and whole-domain transition performance have not yet been
measured.

## Starting point: gated whole-domain Stage-1 Max

The reference version is:

```text
results/stage1_timestamp_max/D030_whole_area_interval20_gated
```

It calibrated the wet-head threshold on validation events and selected a
threshold of 0.85. The mask was applied to depth and both flow components before
whole-domain evaluation.

Across 24 D030 timestamps, gating changed flood-extent behavior substantially:

| Whole-domain metric | Ungated | Gated | Effect |
|---|---:|---:|---|
| Wet precision | 0.431 | **0.834** | Large reduction in false flooding |
| Wet recall | **0.982** | 0.815 | More conservative inundation mask |
| Wet F1 | 0.555 | **0.789** | +0.234 absolute |
| All-cell depth MAE | 0.0298 m | **0.0187 m** | False dry-area depth removed |
| Wet-depth RMSE | **0.856 m** | 0.875 m | No depth improvement |
| Component RMSE | 0.0802 | 0.0802 | No flow improvement |

The model still retained flooding when D030 had drained, particularly near
timesteps 140, 360, and 440. This indicated missing hydraulic memory rather
than only a thresholding problem.

## Changes implemented after the gated version

### 1. Reproducible diagnostics and project organization

- Preserved Stage-1 Max, V4, gated maps, checkpoints, and metrics as separate
  versioned results.
- Organized active Stage-1, legacy blockwise, uncertainty, and transition jobs
  into separate workflow directories.
- Added project and model-direction documentation so training, evaluation, and
  result provenance can be followed without relying on job history.
- Added sampler, depth-bin, speed, direction, calibration, and whole-domain
  diagnostics.

This work did not directly change model accuracy, but it made the subsequent
experiments controlled and reproducible.

### 2. Controlled V4 sampler and loss experiment

V4 tested whether the remaining failures could be fixed without changing the
timestamp-model architecture.

Implemented changes:

- enforced strict batch quotas: 12.5% dry, 25% boundary, 31.25% wet, and
  31.25% deep samples;
- balanced quiet, rising, peak, and recession phases;
- coupled positive depth with wet probability during training;
- increased physical-depth and dry-depth supervision;
- reduced excessive wet-class and Dice weights;
- added speed-weighted vector, speed-magnitude, and flow-direction losses;
- added finite-loss and finite-gradient safety checks;
- warm-started from the Stage-1 Max checkpoint for a controlled five-epoch
  comparison.

### 3. Fundamental model and data review

The review identified the central structural limitation: the timestamp model
predicted every state independently from forcing and terrain. It did not know
the previous water depth or flow state, even though storage, propagation, and
drainage depend on hydraulic history.

The existing data were reorganized conceptually as transition pairs:

- 40 TRITON events;
- 480 half-hour states per event;
- 19,160 adjacent state transitions;
- 31 training events, 8 validation events, and D030 held out for testing.

This is sufficient for a transition prototype, but not sufficient to claim
operational generalization across watersheds, boundary conditions, initial
wetness, or roughness uncertainty.

### 4. Separate hydraulic state-transition model

A new model generation was implemented without changing the checkpoint
contract of the previous timestamp model.

Key changes:

- each target at time `t` is paired with true depth and x/y flow components at
  `t-1`;
- timestamps without an available previous state are excluded before sampler
  batch limits are applied;
- a new raster state encoder processes
  `log1p(previous_depth)`, previous x-component, and previous y-component;
- the forcing encoder, terrain encoder, conditioning network, and compatible
  U-Net weights are warm-started from Stage-1 Max;
- separate residual heads predict changes in depth and flow components;
- the predicted next state equals the previous state plus the learned change;
- residual heads are zero-initialized, so the initial model is exactly a
  persistence forecast rather than an arbitrary state;
- training includes absolute-state, transition-delta, wet/dry, dry-depth, wet
  component, and dry-component objectives;
- evaluation reports the learned model and an exact persistence baseline on
  identical samples;
- 18 unit/regression tests and a real-data forward/backward integration test
  passed before GPU submission.

### 5. Transition V1 training and convergence continuation

- Smoke job `3444268` completed successfully.
- Five-epoch Transition V1 job `3444269` completed successfully.
- All 164 model tensors from the selected epoch-5 transition checkpoint were
  verified as loadable for continuation.
- A five-epoch continuation at learning rate `1e-5` was submitted as job
  `3445828`; it is still running at the time of this summary.
- The continuation writes to `results/stage1_transition_v1_continued`, leaving
  the completed V1 result unchanged.

## Performance progression

The archived TRITON source audit confirmed that component values are HU/HV unit
discharge in `m²/s`, despite legacy netCDF velocity names and attributes. The
reported numerical errors do not change; only their physical interpretation
and units are corrected.

### Controlled sampled D030 comparison

| Model | Wet-depth RMSE | Component RMSE | Wet F1 | Wet CSI | Physical score |
|---|---:|---:|---:|---:|---:|
| Stage-1 Max timestamp | 0.8603 m | 0.0950 | 0.8302 | 0.7096 | 1.1353 |
| V4 timestamp fine-tune | 0.9034 m | 0.0927 | 0.8274 | 0.7056 | 1.1752 |
| Transition V1 | **0.6767 m** | **0.0391** | **0.9310** | **0.8710** | **0.7893** |

Relative to Stage-1 Max, Transition V1 produced:

- 21.3% lower wet-depth RMSE;
- 18.2% lower all-cell depth RMSE;
- 58.9% lower component RMSE;
- +0.101 absolute wet F1;
- +0.161 absolute wet CSI;
- 30.5% lower composite physical score.

The timestamp and transition samplers use the same split and broad evaluation
configuration, but the transition evaluation excludes invalid `t=0` targets.
The resulting deterministic sample changes slightly, so these percentages are
strong directional evidence rather than a perfectly paired comparison.

### Exact Transition V1 versus persistence comparison

This is the primary controlled result because both predictions are evaluated
on exactly the same D030 samples.

| Metric | Persistence | Transition V1 | Improvement |
|---|---:|---:|---:|
| Wet-depth RMSE | 0.7896 m | **0.6767 m** | **14.3% lower** |
| All-cell depth RMSE | 0.1416 m | **0.1295 m** | **8.6% lower** |
| Component RMSE | 0.03941 | **0.03907** | **0.9% lower** |
| Wet F1 | **0.9350** | 0.9310 | 0.004 lower |
| Wet CSI | **0.8779** | 0.8710 | 0.0069 lower |
| Physical score | 0.9009 | **0.7893** | **12.4% lower** |

Interpretation:

- the learned transition adds meaningful depth skill beyond copying the
  previous half-hour state;
- the very small flow improvement shows that persistence still explains most
  short-interval component skill;
- the learned wet head does not yet beat the previous-state inundation mask;
- hydraulic state input is promising, but one-step teacher-forced skill is not
  evidence of stable long-duration simulation.

### Why V4 was still an important result

V4 did not improve the overall model:

- wet-depth RMSE worsened by 5.0% relative to Stage-1 Max;
- component RMSE improved by only 2.4%;
- wet F1 decreased by 0.003;
- physical score worsened by 3.5%.

This negative controlled experiment was useful because it showed that sampler
and loss tuning alone could not close the performance gap. It motivated the
state-transition architecture rather than another large timestamp-model run.

## What can and cannot be claimed now

Supported by completed results:

- previous hydraulic state is an important missing input;
- residual state-transition prediction improves one-step D030 depth;
- Transition V1 is substantially better than the prior timestamp formulation
  on sampled D030 metrics;
- the project should continue along the state-aware model path.

Not yet supported:

- improvement over the gated model on directly comparable whole-domain maps;
- stable 3-, 6-, or 12-hour autoregressive forecasts;
- correction of the specific drainage failures at timesteps 140, 360, and 440;
- mass or cross-block flux consistency;
- operational reliability outside D030 or outside the current watershed;
- calibrated uncertainty and automatic fallback to TRITON.

## Suggested slide wording

### Slide: Why the model formulation changed

> Wet-head gating fixed much of the flood-extent overprediction but left depth,
> flow, and drainage errors largely unchanged. A controlled sampler-and-loss
> fine-tune also failed, indicating that the main limitation was missing
> hydraulic state rather than insufficient scalar loss tuning.

### Slide: Main technical innovation

> We reformulated the surrogate as a residual hydraulic state-transition model.
> Instead of reconstructing each timestamp independently, the model advances
> the previous depth and flow state using forcing history, terrain, and learned
> state increments.

### Slide: Main completed result

> On held-out event D030, Transition V1 reduced one-step wet-depth RMSE by 14.3%
> relative to exact persistence and by approximately 21% relative to the prior
> sampled timestamp model. The composite physical score improved by 12.4%
> relative to persistence.

### Slide: Remaining gap

> The result is a successful one-step proof of concept, not yet an operational
> flood surrogate. Wet extent does not yet beat persistence, and multi-step
> rollout, whole-domain drainage, cross-block flux, uncertainty, and broader
> event generalization remain to be demonstrated.

## Recommended next discussion point

After the continuation run finishes, freeze the best converged checkpoint and
evaluate autoregressive horizons of 1, 6, 12, and 24 steps. The go/no-go test is
whether the model beats persistence through at least 12 hours and removes the
D030 drainage failures without unstable error growth. Only then should we add
tile halos, conservation constraints, and new TRITON training scenarios.

## Continuation update

The lower-rate continuation selected epoch 4 and reduced held-out D030
wet-depth RMSE from 0.6767 m to 0.6150 m. It reduced physical score from 0.7893
to 0.7266 and slightly improved wet F1 from 0.9310 to 0.9324. Relative to exact
persistence, the continued checkpoint lowers wet-depth RMSE by 22.1% and
physical score by 19.4%. It passes all automated acceptance gates and is the
current preferred one-step checkpoint pending rollout and whole-domain tests.
