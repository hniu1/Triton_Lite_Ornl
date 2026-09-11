# Stage-1 Transition V3: Exact Local Transitions and 12-step Hydraulics

## Decision basis

Multi-step V2 is not the V3 parent. Although V2 improved rapid-transition F1
from 0.3197 to 0.3651 and slightly improved one-step dynamic depth, it was worse
than Multi-step V1 in depth at every rollout horizon. At horizons 12 and 24,
V2 depth RMSE was 9.6% and 14.0% worse than Multi-step V1 and also worse than
persistence. Its six-step held-out velocity RMSE was 9.4% worse than V1.

V3 therefore warm-starts from Multi-step V1. It uses V2 as diagnostic evidence
about sampling, stable drift, wet extent, and loss scaling—not as a promoted
checkpoint.

## Problems addressed

1. M5 assigns one regime to a heterogeneous event/time group. The median group
   has three consecutive matched anchors, and 35.8% have at most one.
2. Six-step training is being evaluated out to 24 steps, where error grows
   sharply.
3. Stable depth remains more than 18 times persistence even after V2.
4. V2's weighted velocity term contributed about 0.008% of epoch-3 training
   loss and used Huber loss, while promotion is based on tail-sensitive RMSE.

## Controlled V3 changes

- M6 reads the exact `t-1` patch for every M4 candidate and records local
  extent, storage, robust-depth, activity, and transition-regime statistics.
- `LocalTransitionAwareBatchSampler` guarantees at least one verified local
  transition of the requested regime in every batch while preserving
  dry/boundary/wet/deep quotas and the wet-cell target.
- Stable sampling increases from 25% to 40%; filling, draining, rapid filling,
  and rapid draining use 10%, 10%, 20%, and 20%.
- Training rollout length increases from 6 to 12 steps. Batch count is halved
  to keep total state advances and wall time comparable.
- Stable-delta weight increases from 0.5 to 1.5.
- Component-delta and storage-change weights remain active at 0.5 so depth
  stabilization cannot improve by silently degrading transported quantities.
- Derived velocity uses masked MSE with weight 1.0 so training aligns with the
  operational RMSE gate.
- Learning rate decreases from `3e-6` to `2e-6`; scheduled prediction exposure
  ends at 0.75 rather than V2's 0.80.

V3 intentionally keeps the same residual U-Net and event split. This isolates
verified local sampling, rollout horizon, and physically scaled objectives
before adding a second history state or spatial halos.

## Parallel inference safeguard

V2 is also being evaluated with velocity-persistence reconstruction:

```text
predicted HU(t+1) = previous u(t) * predicted h(t+1)
predicted HV(t+1) = previous v(t) * predicted h(t+1)
```

This leaves depth and wet extent unchanged and tests a practical division of
labor in which the surrogate advances storage rapidly while TRITON resumes
momentum/flooding dynamics from a conservative prior-velocity initialization.
It is a diagnostic mode, not a silent replacement for learned HU/HV.

## Staging and promotion

The original workflow-16 array (`3446034`) was serialized by the account's
one-active-job association limit. D002 and D003 completed, then the remaining
array tasks were held rather than spending roughly 40 sequential allocations.
Workflow 16b performs the same exact computation inside one allocation with up
to eight event workers, skips validated existing shards, and refuses success
unless all 40 shard files are complete.

- Parallel M6 build: workflow 16b, job `3446148`.
- M6 merge: workflow 17, job `3446149`, dependent on `3446148`.
- V3 12-step smoke: workflow 18, job `3446150`, dependent on `3446149`.
- The initial smoke (`3446150`) exposed and led to a fix for a fraction-helper
  class-reference error. Corrected smoke `3446157` was finite but its two
  deterministic evaluation batches were both dry, so it was not accepted as
  sufficient validation. Strengthened smoke `3446158` used 16 evaluation
  batches, covered 22,518 D030 wet cells, exercised a nonzero velocity-MSE
  loss, and passed checkpoint reload.
- Full V3 training: workflow 19, job `3446159`.
- Rollout, regime, and whole-area evaluation: workflows 20-22, jobs `3446160`,
  `3446161`, and `3446162`, dependent on successful full training.
- Dual-reference gates: workflow 23, job `3446163`, dependent on all three
  evaluations.

V3 must be evaluated at rollout horizons 1, 6, 12, and 24, in all transition
regimes, and over the five D030 whole-area key times. Promotion requires the
existing persistence gates and no material regression relative to both
Continued V1's stable behavior and Multi-step V1's dynamic behavior.
