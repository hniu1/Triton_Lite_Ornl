# Stage-1 Transition V2: Regime-Aware Rollout Fine-Tuning

## Purpose

Transition V2 is the next controlled iteration after the six-step
scheduled-sampling experiment. It addresses two failures exposed by the D030
whole-area key-time diagnostic:

- insufficient state change during rapid filling and drainage;
- unnecessary residual change when persistence is already accurate.

V2 retains the same residual state-transition model so the experiment isolates
the effects of transition sampling, multi-step exposure, and delta objectives.
It does not claim to solve missing cross-tile flux or operational data
coverage.

## Transition-aware index

`data_preprocessing/m5_build_transition_sampling_index.py` derives change
labels from the existing M4 dense index without rescanning raw netCDF files.
It matches blocks sampled at both `t-1` and `t` and computes median changes in
wet fraction and maximum depth. Event/time groups without a matched block use
group-aggregate changes as a documented fallback.

The output is:

```text
processed_data/timestamp_stage1/m5_transition_sampling_index/
├── sampling_candidates.parquet
├── sampling_metadata.json
├── transition_groups.parquet
└── transition_sampling_summary.csv
```

Each event/time group is assigned one mutually exclusive regime:

- stable;
- filling;
- draining;
- rapid filling;
- rapid draining.

The five D030 key times provide a sanity check: t=60 and t=440 are stable,
t=140 and t=360 are rapid drainage, and t=240 is rapid filling. D030 remains
held out and does not contribute to training.

## Training changes

`stage1_transition_sampling.py` first chooses a transition regime, then retains
the existing phase and dry/boundary/wet/deep quotas within a feasible
event/time group. The default 500-batch target mix is:

| Regime | Fraction |
|---|---:|
| Stable | 0.25 |
| Filling | 0.10 |
| Draining | 0.10 |
| Rapid filling | 0.275 |
| Rapid draining | 0.275 |

`stage1_transition_multistep_v2_train.py` adds five objectives to the existing
loss:

1. stronger exact depth-delta recovery on rapid transitions;
2. near-persistence depth-delta accuracy on stable transitions;
3. explicit HU/HV unit-discharge delta accuracy on all hydraulically active
   transitions;
4. guarded derived-velocity consistency (`HU/h`, `HV/h`) on true wet cells so
   depth and conserved components cannot improve independently while producing
   unstable velocities;
5. patch-integrated storage-change accuracy, with extra weight on rapid
   transitions, so filling and drainage volume errors cannot be hidden by
   pixelwise averaging.

Training still uses six-step scheduled state exposure. Optimization/sampling
randomness is separated from the parent data-split seed, so a new optimization
seed cannot silently reshuffle training and validation events.

## Evaluation and promotion

V2 must be evaluated with the same D030 samples as its references:

- autoregressive horizons of 1, 6, 12, and 24 half-hour steps;
- stable, filling, draining, and rapid one-step strata;
- exact persistence;
- accepted Continued V1;
- multi-step V1.

`stage1_transition_operational_gate.py` rejects promotion unless useful
horizons beat persistence, rapid regimes improve, stable-state drift remains
small, derived velocity stays bounded, and performance does not regress beyond
tolerance relative to the reference checkpoint.

## Staged workflows

The V2 workflows are intentionally separate from completed V1 jobs:

```text
11_transition_multistep_v2_smoke.sh
12_transition_multistep_v2_train.sh
13_transition_multistep_v2_rollout.sh
14_transition_multistep_v2_regimes.sh
14b_transition_multistep_v2_whole_area.sh
15b_transition_multistep_v2_accept.sh
```

Workflow 11 must pass before workflow 12 is released. Workflows 13, 14, and
14b must depend on successful training. The parent path in workflows 11 and 12
must be verified against completed V1 rollout, regime, and whole-area evidence
before submission. Workflow 15b depends on all three evaluations and applies
the rollout/regime and whole-area gates separately against Continued V1 and
Multi-step V1, while every gate also enforces persistence limits.

For multi-step V1, workflow `15_transition_multistep_accept.sh` runs the joint
rollout/regime gate and the separate five-key-time whole-domain gate once all
dependent outputs exist. Both acceptance JSON files must pass before treating
that checkpoint as operationally promoted.

## Forcing-alignment and additional-history audit

The V2 follow-up audit verified that D030 has 480 TRITON outputs from 0.5 to
240.0 hours at a 0.5-hour interval and an event tensor with exactly 480 rows.
The event preprocessing uses the legacy-compatible 0.5-hour resampling grid
with its zero-hour row dropped. The causal encoder gathers the target
`time_index`, so a transition from `t-1` to `t` sees forcing through `t` but no
future forcing. There is no evidence of a one-step forcing/output offset in the
current data contract.

Adding a second hydraulic history state remains potentially useful for a
stable-update safeguard, but it is not sufficient by itself. Across 18,642
non-D030 transitions, using the previous transition as a direct stable/dynamic
predictor gives stable F1 of about 0.788. However, the key D030 rapid-drainage
transitions at `t=140` and `t=360` follow several stable transitions, while
`t=240` follows alternating rapid drainage and filling. Therefore a future
history-aware model must combine state trend with current forcing; it must not
simply extrapolate the sign of the previous state delta.

## M5 regime-label reliability finding

M5 is suitable for the controlled V2 experiment but is not yet a definitive
local-transition index. Its event/time label matches anchors that happened to
be sampled at both `t-1` and `t` in M4. Of 19,160 non-initial groups, 35.8% have
at most one matched anchor, 49.3% have at most two, and 15.4% use the changing
candidate-pool aggregate because no anchor matches. The median matched count is
three. Therefore the group-level label can be noisy, particularly when a rapid
label is determined by one spatial patch.

This does **not** invalidate the V2 physical losses: after a batch is loaded,
V2 classifies every patch from its actual consecutive TRITON states before
applying stable and rapid penalties. The uncertainty affects which event/time
groups are oversampled. If V2 does not provide a large gain, the next sampling
iteration should compute paired `t-1`/`t` statistics directly for each M4
candidate (or for a fixed spatial panel) and balance verified local
transitions, rather than increasing the current M5 regime weights.

That paired-index correction is implemented as
`data_preprocessing/m6_build_paired_transition_index.py`. It reads the exact
previous patch for each existing M4 candidate and records local extent change,
mean-cell-depth storage change, robust depth change, activity, and a local
transition regime. `m6_merge_paired_transition_index.py` validates and merges
the 40 event shards. Workflows 16 and 17 build and merge the index without
changing the running V2 experiment; a later trainer must explicitly opt into
M6 and guarantee locally verified transition examples in each batch.
