# B1: deep-rising depth accuracy at a fixed 5 h lead

User requested one further experimental round on 2026-10-22. One allocation
will run baseline validation, a smoke check, fresh training and matched
validation. No additional job, retry, test evaluation or map generation is
included. Baseline B and its artifacts are preserved.

## Hypothesis and one controlled change

B improved extent strongly, but D005 native maps show negative depth bias
in some deep regions. That test-event observation motivates a hypothesis;
it does not establish consistent negative bias on validation events.
Do not tune this experiment's settings on held-out test events.

Keep B's model, local forcing weights, two observed hydraulic states, 8-cell
halo, train/validation/test event split, sampling, seed 45, fresh initialization,
optimizer, fixed 5 h lead and ten epochs ×750 batches ×8 tiles.
Keep depth/extent outputs and all existing loss weights. No recursion.

The only training change is an extra symmetric Huber physical-depth term:

`B1 loss = B loss + 1.0 * mean_selected Huber(predicted_depth, target_depth; delta=1 m)`

Selected valid cells have **target depth >=1 m** and **target minus observed
origin depth >=0.1 m over 5 h**. Normalize by selected valid-cell weight;
an empty subset contributes exactly zero. This emphasizes absolute depth
accuracy during deep-water rises, penalizing both error signs. It does not
add a positive offset or change the inference architecture. The original
weighted physical-depth loss already exists; this is an additional targeted
term normalized separately. Thresholds/weight are prespecified, not optimized.
Reference targets are used for loss masks only, never inference conditioning.

## Validation and decision

Before training, evaluate the unchanged B checkpoint on **256 deterministic
validation batches**, up to eight tiles each, with 5 h forecasts. After
training, evaluate B1 on identical cases. Checkpoint selection remains B's
original 80-batch wet-depth RMSE +0.5×(1−wet-head F1) rule.

Both final validation comparisons use raw depth-defined extent >=0.05 m,
without threshold calibration. Report MAE/RMSE, signed bias and frequencies
of under/overprediction exceeding 0.3 m by true target-depth bins:
0.05–0.25, 0.25–1, 1–2, 2–5 and >=5 m, plus the selected deep-rising subset.
Report pooled and per-event diagnostics, cell counts and persistence.
The targeted subset uses the same reference-defined mask for both models.
If a bin has no cells, its error measures are null, not zero.

Assess reduced deep-rising MAE/RMSE together with bias; cancellation of errors
is insufficient. Also inspect other depth bins, raw extent precision/recall/F1
and event-level consistency. If the baseline does not show systematic negative
bias, interpret B1 as an accuracy experiment rather than assuming a bias fix.
If deep-rising cells are scarce, this sampled comparison is inconclusive and
a deliberately stratified validation cohort is needed before further tuning.
Do not declare success from lower training loss: B1 adds an objective, so
its total loss magnitude is not directly comparable with B's.

The comparison is validation, not independent held-out confirmation. Further
test evaluation needs a reviewed validation result and user authorization.
Reducing underprediction must not be achieved by unacceptable overprediction
or loss of the improved inundation extent. No gain is guaranteed.

## Artifacts and execution

- Job wrapper: `workflows/stage2/44_B1_deep_rising_round.sh` (one L40S, 8 h limit).
- Training helper: `workflows/stage2/43_B1_deep_rising_train.sh`.
- Root: `results/stage2_fixed5h_deep_rising/20261022_fold0/`.
- Smoke: `smoke_job_JOBID/`; finite outputs/gradients, fixed lead, nonzero
  targeted loss, frozen discharge head and no test evaluation are required.
- Fresh training: `seed_45/`; smoke weights are discarded.
- Comparison: `evaluation/{B,B1}_validation.json` and `evaluation/REPORT.md`.
- Loss settings and deep-water cell-weight thresholds are pinned in run config.

New loss weight defaults to zero, preserving existing training objectives.
Depth diagnostics are opt-in and do not change forecasts or scoring masks.

Submitted **job 193961** on 2026-10-22. All **62 regression tests passed**;
Python compilation and both shell syntax checks passed. Completion is not
yet verified. Frozen `PLAN.json` records source/checkpoint/forcing hashes.

## Completed outcome

Job 193961 completed exit 0:0 in 01:11:47. Ten epochs; best epoch 10.
The matched 256-batch /2,028-pair seven-event validation comparison finds
a 3.6% deep-rising RMSE reduction, but overall wet RMSE rises 5.7%, MAE
rises 11.8%, and raw extent F1 falls 0.9789 to 0.9614. Keep B as the
working model. No held-out evaluation or new jobs were submitted.
Detailed report and training/comparison figures:
`results/stage2_fixed5h_deep_rising/20261022_fold0/RESULTS_SUMMARY.md`.
