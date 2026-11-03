# B: fixed 5-hour localized Halo, depth and wet/dry extent

Status: **TRAINING AND EVALUATION COMPLETE.** Training 193869 completed all
10 epochs in 1h24m28s; best checkpoint is epoch 10. Source reaudit array 193914
completed all eight events with exit `0:0`, finding exact historical equality
of every state/change selection table. Revalidated evaluation 193915 completed
successfully in 7m02s. The initial evaluation 193870 failure is resolved.

[Completed results, MAE/RMSE comparisons and charts](../../results/stage2_fixed5h_local/20261022_fold0/evaluation/RESULTS_SUMMARY.md).
[Training summary and loss curves](../../results/stage2_fixed5h_local/20261022_fold0/seed_45/TRAINING_SUMMARY.md).

On the matched selected flood cases (544 tile/target pairs, eight events), B
achieves wet-depth MAE **0.1500 m**, RMSE **0.2180 m**, and depth-defined extent
F1 **0.8860** at a 0.05 m cutoff. Previous Halo gives 0.1620 m, 0.2380 m and
0.5110; persistence gives 0.2960 m, 0.3489 m and 0.9056. B's separate wet-head
F1 is 0.9339. These two extent definitions must not be conflated.

B reduces wet-depth RMSE by 8.4% against previous Halo and 37.5% against
persistence. False wet cells fall about 89.8% against previous Halo, while
raw-depth wet recall is 95.0%. Depth RMSE improves against previous Halo in
seven of eight events, but D005 worsens (0.1639 to 0.2004 m). This is meaningful
progress but not a universal win or evidence of a standalone hydraulic surrogate.
The comparison changes forcing localization, training lead and output losses
together; it does not identify localization's causal contribution. No A/C model
was trained, and no additional Slurm jobs were submitted to summarize results.

## Task and training

Predict depth and wet/dry extent directly at **origin +5 h**, using observed
depth and signed HU/HV at origin and origin−0.5 h, six terrain raster channels,
seven tile descriptors and known forcing through the target. Use an 8-cell
hydraulic halo. This is one direct forecast, not a recursive sequence.

The training sampler is explicitly restricted to **lead step 10** on the
0.5 h time axis. Setting max lead to 10 alone would still train leads 1–10;
the new `--train-horizons 10` restriction prevents that mistake. Minimum target
index is 11, allowing the two earliest observed origin states.

Run from fresh initialization, seed 45, fold-0 data split seed 4300:
25 training, seven validation and eight held-out events. No legacy or prior
Halo weights are loaded. Train for **10 epochs ×750 batches ×8 center tiles**,
using AdamW at 1e-4, weight decay 1e-5, the existing strict category/phase-balanced
sampler with a 15% wet-cell target, and the original corrected-data normalization
fit on training events only. Holdouts: D005, D014, D018, D023, D024, D028,
D033, D037. Test data do not select checkpoints or fit weights.

Losses retain wet/dry BCE and Dice, wet-depth physical/log losses, dry-depth
penalty, transition/change, storage-change and activity objectives. Discharge,
dry-discharge, discharge-change, derived-velocity, velocity-direction and
boundary-discharge losses are disabled and their computations skipped. Observed
HU/HV remain inputs. Compatibility discharge-correction heads stay frozen at
zero; their outputs are not trained predictions and are excluded from reports.

Checkpoint selection uses validation **wet-depth RMSE +0.5×(1−wet F1)** at
5 h, with the original head probability threshold 0.5. No discharge or velocity
metric contributes. Sampling, raw evaluation wet depth and extent thresholds
are not changed using test results. The old 10 cm display threshold is not
assumed calibrated for this new model.

## Local forcing mapping

Global context remains: all 300 forcing curves pass through the existing
causal encoder. Add a separate 64-value local forcing embedding from one
weighted combination of the 300 standardized forcing curves for each tile.
This is a weighted anomaly signal, **not a physical sum of inflows**.

The static mapping was audited against **all 40 original simulator archives**:
Loc1–Loc300 in the point shapefile exactly match the simulator's source-location
coordinates and order to 1e-6 m. Source configs explicitly reference those
locations and declare 300 inflow sources. All 300 points lie inside one of the
34 HUC12 polygons; geometry coordinates agree with their X/Y attributes.

- Use up to eight nearest same-HUC12 points, with inverse-distance weights
  and a 100 m distance floor.
- Where upstream HUC12s exist, also use up to eight nearest points in those
  ancestors. Obtain ancestors from the original `ToHUC` network, rejecting
  loops. Their weights additionally decrease with `1 + HUC hop count`.
- Allocate 70% weight to the same HUC12 and 30% upstream when both exist;
  use 100% same-HUC weight for mapped headwater HUCs.
- For tile centers outside the mapped basin, explicitly fall back to the
  eight nearest forcing locations. Retain global context everywhere.

Tile-center mapping counts:

| Mapping | Tiles |
|---|---:|
| Same HUC12 plus upstream | 1,620 |
| Same HUC12 only | 1,638 |
| Outside-basin nearest-point fallback | 3,642 |

The model raster covers more than the mapped basin. These fallback tiles are
not claimed to have verified upstream assignments. Membership is based on tile
center; boundary tiles can span more than one HUC12. Same-HUC membership and
HUC-level ancestor routing do **not** establish exact within-HUC cell-to-source
connectivity or travel times. The 70/30 weighting and nearest-point cap are
fixed initial design choices, not hydrologically calibrated contributions.

Artifacts: `processed_data/stage2_fixed5h_local/20261022_fold0/forcing/`
contains weights, per-tile and per-point CSVs, source-file hashes, 40 simulator
location checks and a metadata report. The weight SHA-256 is pinned in each
run config and checked during evaluation. The legacy nearest-eight weights
were not overwritten.

## Evaluation and interpretation

After training, evaluate raw predictions against persistence and replay the
**previous existing Halo checkpoint** on exactly matched cases. No reference
model is retrained. Report both:

1. Representative sampled domain/time evaluation, 80 batches of eight centers.
2. Dynamic-stream conditional evaluation, **68 batches /544 tile-target pairs**,
   obtained by restricting the existing frozen cohort to valid 5 h origins.
   All eight held-out events remain represented.

Report depth RMSE and wet precision/recall/F1, including both wet-head extent
at probability 0.5 and **depth-defined extent at predicted depth >=0.05 m**.
The latter exposes the shallow false wet predictions that motivated this work;
a good wet-head score alone cannot establish narrow, accurate depth maps.
Also retain per-event metrics and verify matching persistence across models.

The previous Halo reference was multi-lead and jointly supervised for
hydraulics; B changes lead training, localization and supervision together.
This is a practical next-model comparison, not a causal localization ablation.
The test fold has already been inspected, so results remain exploratory.
Future improvements need another fold or independent event confirmation.

Root: `results/stage2_fixed5h_local/20261022_fold0/`

- `smoke_job_193868/`: one epoch, 20 train batches, eight validation batches;
  no held-out evaluation. Smoke requires finite outputs, exactly 160 lead-10
  samples, nonzero local-encoder training gradients and frozen discharge heads.
- `seed_45/`: full training, best checkpoint, per-epoch checkpoints and progress.
- `conditional_cohort.json`, `conditional_rule.json`: matched fixed-lead cases
  with parent-cohort hashes and unchanged source identities.
- `evaluation/`: new B and previous Halo representative/conditional metrics,
  head- and depth-defined extent, CSV comparison and `REPORT.md`.
- `JOBS.json`: submitted job IDs.

Smoke, training and evaluation requests passed Slurm test-only validation.
**59 regression tests passed**, including direction-aware exclusion of
unrelated/downstream HUCs, the fixed-lead sampler, complete independence of the
depth loss from discharge targets/predictions, and velocity-free selection.
Full training starts only after a finite GPU smoke and nonzero local-gradient
check. Smoke uses fresh weights; the full run also starts fresh.
