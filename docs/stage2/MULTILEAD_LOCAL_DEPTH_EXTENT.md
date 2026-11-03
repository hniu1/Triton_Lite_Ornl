# Local forcing, depth/extent training at 2–12 h

User approved one training job on 2026-10-23, extending B to direct leads
**2, 4, 5, 6, 8 and 12 h** (half-hour steps 4, 8, 10, 12, 16, 24).

Use B's original loss with **deep-rising extra loss weight zero**. Preserve
the architecture, localized forcing weights, 8-cell Halo, two observed
hydraulic states, seed 45, data split seed 4300, 25/7/8 event split,
optimizer and learning rates. Train from fresh initialization, without
loading B or B1 weights. Known forcing through target remains an input
assumption. No recursive predictions or intermediate reference-state input.

## Budget and balancing

Ten epochs ×750 batches ×8 central tiles, matching B's total training budget.
Each shuffled six-batch cycle contains all six leads exactly once. Each
epoch therefore has **125 batches /1,000 tile examples per lead**: 10,000
examples per lead over training, versus 60,000 examples at 5 h for B.
This is a shared-model experiment at a fixed total budget, not equal
per-lead exposure. Reduced 5 h accuracy could reflect this allocation;
do not attribute it solely to architecture or multi-lead interference.

The existing balanced spatial/category/phase sampler supplies target tiles
and times; only horizon assignment changes. Require enough origin history
at the longest 12 h lead (minimum target index 25). Thus the training target
pool excludes some early times available to fixed-5-hour B. Record this
coverage difference when comparing models. B1's targeted loss is not used.

Balanced horizon assignment is opt-in; old random sampling is unchanged by
default. One lead is shared across each spatial batch. Seed plus epoch
controls the shuffled schedule reproducibly.

## Validation and execution

At each epoch, report each lead on 80 deterministic validation-origin
batches. Origins are shared across leads and eligible for the longest lead;
targets differ. Select a single checkpoint by the equal mean across the six
leads of wet-depth RMSE +0.5×(1−wet-head F1), preserving B's component-free
score definition. This may not choose the best checkpoint for every lead.
Subsequent matched B comparisons should explicitly freeze the same origins
or targets; default fixed-5 h evaluation uses a different eligibility grid.

One L40S job (8 h limit) runs a 24-batch GPU smoke first. Every lead must
contribute 32 tile examples, outputs/weights must be finite, local encoder
gradients nonzero and discharge heads frozen. Full training starts only
after the smoke passes, from fresh weights rather than smoke weights.
No held-out test evaluation, separate job, retry or map generation is included.

Root: `results/stage2_multilead_local/20261023_fold0/`.
Wrapper: `workflows/stage2/46_multilead_local_round.sh`.
Helper: `workflows/stage2/45_multilead_local_train.sh`.
Run outputs: `seed_45/{training_progress.json,metrics.json,best_model.pt}`.
Plan/provenance: `PLAN.json`; scheduler record: `JOBS.json`.

After reviewing training/validation, evaluate per-lead MAE/RMSE, raw-depth
extent precision/recall/F1, deep-water signed bias and persistence on matched
cases. Do not transfer the old model's visualization threshold to this model.

Submitted **job 194250** on 2026-10-23 after **63 regression tests passed**,
including exact per-lead balance, time/tile alignment and epoch
reproducibility. Python compilation and shell syntax checks passed.
Completion is not yet verified.

## Completed training

Job 194250 completed exit 0:0 in 02:01:30. All ten epochs finished; best
shared checkpoint is epoch 8. On the per-lead sampled validation cohort,
wet-depth RMSE is 0.0409, 0.0425, 0.0459, 0.0521, 0.0775 and 0.1455 m
at 2, 4, 5, 6, 8 and 12 h respectively. Persistence is better at 2 h;
the model is better at the other five leads. Raw depth-defined extent and
depth-bin bias require further evaluation; logged F1 uses the wet head.
No new evaluation jobs have been submitted for this review.

Summary, training curves and per-lead CSV/plots:
`results/stage2_multilead_local/20261023_fold0/RESULTS_SUMMARY.md`.

## Requested whole-domain maps

Submitted **job 194328** on 2026-10-23 using
`workflows/stage2/47_multilead_D005_whole_domain_maps.sh`.
Uses best epoch 8, fixed 5 h direct lead and the exact 28 D005 target times
from the previous B visualization (6–18 h and 66–106 h at 2 h spacing).
All domain tiles; north up; reference/prediction wet threshold 0.05 m;
blue-positive/red-negative raw error at ±0.3 m. This preserves a matched
map comparison and does not include every slowly changing flooded time.
Figures identify the multi-lead model. Output:
`results/stage2_whole_domain_halo/20261023_D005_multilead_fixed5h_flood/`.
One inference-and-plotting job only, no retraining. Six relevant tests and
shell syntax check passed. Completion is not yet verified.
