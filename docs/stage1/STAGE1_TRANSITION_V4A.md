# Stage-1 Transition V4a: Two-state History and Persistence Gate

## Evidence from V3

V3 reduced stable-regime depth RMSE from 0.0945 (Multi-step V1) to 0.0522,
but filling depth worsened from 0.3731 to 0.4561 and rapid depth worsened from
1.3126 to 1.5169. Rollout velocity remained far outside persistence bounds.
Epoch 1 was selected; higher predicted-state exposure in epochs 2 and 3
degraded validation. These results show that more stable weighting and longer
training alone cannot distinguish stable, filling, and draining behavior from
one input state.

## Controlled V4a changes

- Input the latest state and its change from one additional historical state.
- Predict a spatial activity gate and apply residual updates as
  `next = previous + gate * learned_delta`.
- Supervise the gate from exact consecutive states: stable patches target zero;
  changing wet/depth cells in dynamic patches target one.
- Warm-start all compatible dynamics from Multi-step V1. The three new delta
  input channels start at zero and the new gate is the only random head.
- Reduce stable sampling to 30% and stable-delta weight to 0.75 because the
  explicit persistence gate now supplies the stable inductive bias.
- Limit scheduled predicted-state exposure to 0.15-0.40, based on V3's
  degradation at 0.55 and 0.75.
- Select checkpoints with `physical_score + derived_velocity_rmse` and retain
  every epoch checkpoint.

V4a retains the residual U-Net, exact M6 index, event split, 12-step rollout,
and remaining physical losses. Direct bounded velocity output is deferred to
V4b so its necessity can be decided from V4a evidence.

## Promotion sequence

Workflow 24 is a four-batch smoke with 16 validation/test batches. Full workflow
25 is submitted only if the smoke confirms two-state checkpoint initialization,
nonzero wet and velocity metrics, finite gate loss/gradients, checkpoint reload,
and correct saved model configuration.

The first successful smoke used gate-loss weight 0.15 and improved D030 smoke
depth, velocity, F1, and physical score relative to the V3 smoke. Its gate term
contributed about two thirds of total test loss, so the controlled full setting
reduces that weight to 0.05 and requires a matching second smoke before submit.

## Full-run evidence

Training job 3446677 completed successfully after correcting the exact-index
minimum target time for two-state 12-step sequences. Epoch 1 was selected;
epochs 2 and 3 reduced derived velocity somewhat but worsened depth and F1.

On held-out sequence batches, epoch 1 achieved depth RMSE 0.4577 versus 0.6049
for persistence, but velocity RMSE was 0.1085 versus 0.0109 and F1 was 0.6816
versus 0.9105. The standard rollout then lost to persistence in depth and F1
at every horizon:

| Horizon | V4a depth | Persistence depth | V4a F1 | Persistence F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 step / 0.5 h | 0.3916 | 0.3713 | 0.8921 | 0.9188 |
| 6 steps / 3 h | 0.4456 | 0.4222 | 0.8243 | 0.9337 |
| 12 steps / 6 h | 0.6048 | 0.5535 | 0.7224 | 0.8749 |
| 24 steps / 12 h | 0.6403 | 0.4968 | 0.5218 | 0.8868 |

At 24 steps, V4a velocity RMSE was 0.1781 versus 0.0126 for persistence.
Teacher-forced 24-step metrics remained much stronger than autoregressive
metrics, showing that accumulated self-state error is the principal failure.

Checkpoint inspection showed that the selected history-channel norm was only
0.0018 times the pretrained state-channel norm. The gate bias remained 1.9995
and the gate weight norm was only 0.0021, leaving the gate nearly constant at
sigmoid(2) = 0.88. V4a therefore did not meaningfully learn its intended new
mechanisms and is rejected for promotion. V4b addresses this optimization
failure with separately parameterized, faster-trained history and gate
adapters.
