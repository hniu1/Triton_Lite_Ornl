# Stage-1 V4 Controlled Fine-Tuning

## Purpose

V4 is a short, isolated experiment intended to test the highest-value changes
before committing to another full training campaign. It warm-starts from
`results/stage1_timestamp_max/best_model.pt` and writes only to
`results/stage1_timestamp_v4_finetune/`.

No completed Stage-1 Max checkpoints, metrics, or gated D030 results are
overwritten.

## Implemented changes

### Strict batch composition

The prior balanced sampler could replace dry, boundary, or wet quota selections
with deep candidates to reach its 15% wet-cell target. This produced 66% deep
patches despite a configured 25% deep quota.

Strict mode now rejects event/timestamp groups unless they can satisfy both the
exact category quotas and wet-cell target. If a selected batch needs more wet
cells, replacements may occur only within the same category.

For batch size 16, V4 uses:

| Category | Count | Fraction |
|---|---:|---:|
| Dry | 2 | 12.5% |
| Boundary | 4 | 25.0% |
| Wet | 5 | 31.25% |
| Deep | 5 | 31.25% |

A direct feasibility check against the current dense index found 5,267 eligible
event/timestamp groups. In a deterministic 100-batch check, every batch met the
exact quotas; minimum mean wet fraction was 0.1501 and mean was 0.1781.

### Depth/wet coupling

With `--couple-depth-with-wet-probability`, training and inference use:

```text
effective_depth = softplus_depth * sigmoid(wet_logits)
```

This lets depth errors update the wet head and discourages contradictory
high-depth/low-wet predictions. The final operational binary gate remains a
separate calibrated inference step.

### Speed-aware velocity objective

The optional `speed_aware` component loss combines:

```text
weighted vector Huber
+ 0.5 * speed-magnitude Huber
+ 0.1 * direction loss
```

Wet-cell weights increase with true component magnitude, capped to prevent a
few extreme cells from dominating. Direction loss is active only where true
unit-discharge magnitude is at least 0.05 m²/s. The legacy CLI retains
`speed_aware` naming for checkpoint compatibility.

### Revised fine-tuning weights

- learning rate: `3e-5`;
- epochs: 5;
- physical-depth loss weight: 1.0;
- dry-depth loss weight: 0.10;
- wet positive weight: 1.25;
- wet Dice weight: 0.15.

## Execution gate

First submit:

```bash
sbatch workflows/stage1/10_stage1_v4_sampling_diagnostics.sh
```

Review
`results/stage1_timestamp_v4_finetune/sampling_diagnostics/strict_sampler.json`.
Only if category, phase, and wet-cell distributions are acceptable, submit:

```bash
sbatch workflows/stage1/10b_stage1_v4_training_smoke.sh
```

The smoke run checks finite forward losses, backward gradients, checkpointing,
and held-out evaluation using only ten training batches. After it succeeds,
submit:

```bash
sbatch workflows/stage1/11_stage1_v4_finetune.sh
```

## Scope limitation

This controlled run rebalances existing quiet/rising/peak/recession labels. It
does not yet add forcing derivatives, time-since-peak, explicit rapid-drainage
labels, neighboring-block context, or whole-domain checkpoint selection. Those
larger changes should be implemented only after measuring this ablation.
