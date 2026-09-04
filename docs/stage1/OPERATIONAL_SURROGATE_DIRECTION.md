# Operational TRITON Surrogate: Data Decision and Fast-Win Plan

## Decision summary

The current dataset is **sufficient for a focused state-transition prototype**,
but it is **not sufficient to claim an operationally reliable surrogate**.

We should not continue tuning the current independent-timestamp model as the
main path. We should retain its useful components and checkpoints, then develop
a new state-aware model on a separate branch:

```text
feature/state-transition-surrogate
```

The fastest informative experiment is to use existing TRITON states as
historical hydraulic input and predict the next state or state increment. This
requires no new TRITON simulations for the first prototype.

## Why a structural change is justified

TRITON is a two-dimensional hydrodynamic solver that advances water depth and
two flow components through time using the shallow-water equations. Its state
contains hydraulic memory: storage, propagation, and drainage depend on the
previous water state, not only on the current forcing and terrain.

The present Stage-1 model receives forcing history, requested time, block
attributes, and static terrain, then predicts each block-local timestamp
independently. It does not receive the previous depth/flow state and does not
exchange flux with neighboring blocks. This creates a fundamental information
gap that loss-weight tuning cannot remove.

The current results support this diagnosis:

| Evidence | Result | Implication |
|---|---:|---|
| Ungated whole-domain D030 wet F1 | 0.555 | Raw depth produces excessive inundation |
| Gated whole-domain D030 wet F1 | 0.789 | Classification/gating helps extent substantially |
| Gated whole-domain wet-depth RMSE | 0.875 m | Depth magnitude remains inaccurate |
| Gated unit-discharge component RMSE | 0.0802 m²/s | Gating does not improve wet-cell flow |
| V4 wet-depth RMSE | 0.903 m | Sampling and loss changes alone made depth worse |
| V4 unit-discharge component RMSE | 0.0927 m²/s | Only a small RMSE improvement versus 0.0950 m²/s sampled baseline |
| D030 transition failures | t=0, 140, 360, 440 | Model retains flooding when the true system drains |

V4 corrected the sampler exactly and introduced stronger losses, but its
physical score worsened from 1.135 to 1.175. This is strong evidence that the
remaining limitation is not primarily a category quota or scalar loss-weight
problem.

## What data are currently available

The dynamic manifest contains:

- 40 TRITON events from one watershed;
- 480 half-hour states per event;
- 240 hours per event;
- 19,160 adjacent transitions across all events;
- 31 training events;
- 8 validation events;
- D030 as the held-out test event.

The legacy netCDF attributes label depth in meters and both components in
`m s-1`. A subsequent audit of the archived executable source proved that the
components are the conserved HU/HV unit-discharge fields in `m²/s`; see
`TRITON_COMPONENT_SEMANTICS_AUDIT.md`. The numerical arrays are unchanged.

### What this dataset is sufficient for

It can support:

- proof that previous hydraulic state improves one-step prediction;
- prediction of `state(t+1)` or `state(t+1) - state(t)`;
- short autoregressive rollouts;
- comparison of direct-state and residual-state prediction;
- rising, peak, recession, and quiet-phase evaluation;
- D030 held-out transition testing;
- initial testing of overlapping spatial context.

The 19,160 transitions are useful supervised examples, even though adjacent
states within one event are correlated.

### What this dataset is not sufficient for

It cannot establish operational generalization because there are only 40
independent hydrographs and only one watershed/static domain. It does not
demonstrate reliability across:

- unseen watersheds or topographies;
- rainfall and inflow patterns outside the 40 events;
- different initial wetness or hydraulic states;
- uncertain Manning roughness;
- alternative boundary conditions;
- different runoff/infiltration assumptions;
- events beyond the simulated magnitude range;
- observational and TRITON model discrepancy.

Millions of spatial patches do not replace independent event diversity. The
effective sample size for event-level generalization remains approximately 40,
not the number of patches.

## Should we use historical data?

Yes, but “historical data” has two distinct meanings.

### Historical hydraulic state: use immediately

For every transition, use recent TRITON state as input:

```text
h(t), component_x(t), component_y(t)
+ forcing over t to t+1
+ static terrain and block information
→ h(t+1), component_x(t+1), component_y(t+1)
```

Prefer residual prediction:

```text
predicted_state(t+1) = state(t) + predicted_change(t→t+1)
```

This gives the network the information required to distinguish rising water
from recession and persistent storage.

For later experiments, include several recent states or state changes, such as
`t-1`, `t-2`, and `t-4`, when one state is insufficient.

### Historical observations: add for calibration and correction

Observed gauge stage/discharge, remotely sensed inundation, and high-water
marks can reduce TRITON-to-reality bias. They should not be mixed casually with
simulation targets. Each observation type needs explicit spatial/temporal
alignment and an uncertainty model.

The first state-transition prototype should use internally consistent TRITON
trajectories. Observational fine-tuning should follow once the simulator
surrogate is stable.

## Should we use the same model?

Reuse components, but create a new model class and data contract.

### Reuse

- event split and normalization infrastructure;
- causal forcing encoder;
- static terrain encoder;
- block attributes;
- U-Net spatial decoder;
- wet-head calibration workflow;
- whole-domain reconstruction and evaluation;
- sampler and phase diagnostics.

### Replace or extend

- add previous `h/component_x/component_y` as dynamic raster channels;
- predict state increments instead of an independent absolute map;
- add a state encoder before the spatial bottleneck;
- train one-step and multi-step rollouts;
- add neighboring context or halo cells after the first prototype;
- add conservation and boundary-flux losses after verifying the state semantics.

The proposed class should be separate, for example:

```text
Stage1StateTransitionModel
```

Do not silently change `Stage1TimestampModel`, because completed checkpoints
and evaluations depend on its current architecture.

## Why use a separate branch

This work changes sample keys, data reads, model inputs, output interpretation,
training objectives, inference, and evaluation. It is a new model generation,
not a minor V4 setting change.

A separate branch protects:

- reproducibility of Stage-1 Max and V4;
- the calibrated D030 gated results;
- current checkpoints and scripts;
- controlled comparison between architectures;
- collaborator understanding of which pipeline is authoritative.

Recommended branch:

```bash
git switch -c feature/state-transition-surrogate
```

Create versioned modules and workflows rather than overwriting the existing
ones. Suggested names:

```text
stage1_transition_data.py
stage1_transition_model.py
stage1_transition_train.py
stage1_transition_evaluate.py
workflows/stage1_transition/
results/stage1_transition_v1/
```

## Fast-win experiment using current data

The objective is not yet to build the final operational system. It is to answer
one decisive question:

> Does providing the previous hydraulic state remove the large drainage and
> depth failures that persisted after sampling, gating, and loss changes?

### Experiment A: baselines

Evaluate on exactly the same event split:

1. Persistence: `predicted_state(t+1) = state(t)`.
2. Current Stage-1 timestamp model.
3. Current gated Stage-1 result.

Persistence is essential because half-hour states may change slowly. A learned
transition model is only useful if it beats persistence, especially during
rapid transitions.

### Experiment B: teacher-forced one-step transition

Inputs:

- true state at `t`;
- forcing at `t` and `t+1`, plus recent forcing summaries;
- current six static rasters and block mask;
- current block attributes and time features.

Targets:

- change in depth;
- change in both flow components;
- wet/dry state at `t+1`.

Start with the current 80×80 block and no architecture-wide graph. This keeps
the prototype small and isolates the value of hydraulic history.

### Experiment C: short rollout

Evaluate autoregressive rollouts of:

- 1 step: 0.5 hours;
- 6 steps: 3 hours;
- 12 steps: 6 hours;
- 24 steps: 12 hours.

Train initially with true previous state, then introduce scheduled sampling so
the model also sees its own predicted state. Report error growth versus rollout
horizon.

### Transition-aware sampling

Sample by the magnitude and sign of true state change, not only absolute depth:

- stable dry;
- filling/rising;
- stable wet/peak;
- draining/recession;
- rapid transition;
- wet/dry boundary movement.

The t=140, 360, and 440 D030 neighborhoods should be explicit evaluation cases,
not training cases.

## Fast-win success criteria

Proceed to the larger connected model only if transition V1 demonstrates:

- better performance than persistence at 3, 6, and 12 hours;
- at least 20% lower D030 wet-depth RMSE than the current gated model;
- materially reduced false persistence at t=140, 360, and 440;
- improved recession precision without collapsing recall;
- bounded rollout error through at least 12 hours;
- no numerical instability at wet/dry boundaries;
- inference meaningfully faster than the equivalent TRITON interval.

Failure to beat persistence would mean the chosen state resolution, forcing
inputs, or transition interval is inadequate and should be corrected before
adding model complexity.

## What follows after the fast win

If state input helps, the next development sequence is:

1. Add overlapping tile halos to provide neighboring hydraulic state.
2. Add cross-block flux consistency and discrete mass-conservation diagnostics.
3. Compare direct learned transition against coarse-TRITON plus learned
   correction/downscaling.
4. Generate a designed ensemble of additional TRITON events spanning forcing,
   initial states, roughness, and boundary uncertainty.
5. Add calibrated uncertainty and automatic TRITON fallback.
6. Validate on multiple held-out events and, ultimately, unseen watersheds.

The preferred operational design remains hybrid:

```text
fast state-aware surrogate
→ immediate screening, mapping, and uncertainty
→ TRITON for high-risk, uncertain, or out-of-distribution cases
```

## Immediate recommendation

Use the current data to build Transition V1 now. Do not launch another large
independent-timestamp training campaign. Preserve the current model as the
baseline and gated screening product, while developing the state-transition
surrogate independently.

The current data can tell us whether hydraulic history is the missing signal.
Additional diverse TRITON simulations will still be required before the model
can be considered operationally reliable.

## References

- [ORNL TRITON documentation](https://triton.ornl.gov/)
- [TRITON multi-GPU hydrodynamic model paper](https://www.sciencedirect.com/science/article/pii/S1364815221000773)
- [Physics-informed hydrodynamic surrogate study](https://www.sciencedirect.com/science/article/pii/S0048969723074430)
- [SWE-GNN flood-surrogate abstract](https://agu.confex.com/agu/fm23/meetingapp.cgi/Paper/1417355)
