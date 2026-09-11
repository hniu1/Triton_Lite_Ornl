# Stage-1 Transition V4b: Fast History and Gate Adaptation

## Why V4a is not sufficient

V4a's selected checkpoint improved held-out 12-step depth RMSE over persistence
by 24%, but its standard rollout lost to persistence at 1, 6, 12, and 24
steps. At 24 steps, velocity RMSE was 0.1781 versus 0.0126 for persistence and
wet/dry F1 was 0.5218 versus 0.8868. Teacher-forced performance remained much
better than autoregressive performance, identifying error accumulation as the
dominant failure.

The selected checkpoint also showed that the intended V4a mechanisms had
barely trained. The new history-channel norm was only 0.0018 times the
pretrained state-channel norm. The activity-gate bias remained 1.9995 and its
weight norm was only 0.0021, leaving an almost spatially constant gate near
sigmoid(2) = 0.88.

## Controlled V4b change

- Preserve the pretrained three-channel hydraulic state encoder exactly.
- Add a zero-initialized 1x1 adapter that injects the latest-minus-older state
  into the current state before the pretrained encoder.
- Train only the new history adapter and activity head at 1e-3 while retaining
  2e-6 for the pretrained hydraulic backbone.
- Initialize the activity gate at -1.5, or sigmoid(-1.5) = 0.18, so the model
  begins conservatively near persistence rather than passing almost every raw
  residual.
- Reduce the auxiliary gate-loss weight from 0.05 to 0.02 so it guides the
  adapter without dominating the hydraulic objectives.

This is an optimization and inductive-bias correction, not a capacity increase.
The first promotion criterion is improvement over both V4a and persistence at
the standard rollout horizons, with no regression hidden by aggregate loss.
