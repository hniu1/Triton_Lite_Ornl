# Minimum Publishable Validation Plan

This document defines the minimum set of analyses required to claim publishable scientific validation for the current blockwise flood depth model.

## 1) Required Validation Questions

1. Does the model generalize to unseen events (including extremes)?
2. Are wet probabilities calibrated for decision use?
3. Are predictive uncertainties reliable and sharp?
4. Is performance robust under realistic perturbations?
5. Do latent representations encode physically meaningful structure?

## 2) Minimal Experiment Set (Do Not Skip)

1. Event-level generalization split
2. Multi-seed uncertainty ensemble
3. Wet probability calibration
4. Robustness stress matrix
5. Latent representation analysis
6. Statistical confidence and significance

## 3) Data Splits (Minimum)

1. Train/validation/test at event level only (no pixel leakage).
2. In-domain test split with unseen normal events.
3. OOD test split with unseen high-intensity events.
4. Spatial holdout split by withheld block region.

Acceptance condition:
1. All headline metrics must be reported separately for in-domain, OOD, and spatial holdout splits.

## 4) Uncertainty and Calibration (Minimum)

1. Train 5 model seeds with identical hyperparameters.
2. Use ensemble mean as prediction.
3. Construct 90% prediction intervals for depth from ensemble quantiles.
4. Evaluate interval coverage and mean interval width.
5. Compute wet reliability diagram, Brier score, and ECE.

Required report table columns:
1. Split
2. RMSE
3. F1
4. CSI
5. Brier
6. ECE
7. PI90 coverage
8. PI90 width

Acceptance condition:
1. PI90 coverage is close to nominal on in-domain (target 0.90 +/- 0.04).
2. ECE <= 0.05 on in-domain test.

## 5) Robustness Analysis (Minimum)

Run all tests on in-domain and OOD test sets.

1. Rainfall amplitude scaling: -10%, +10%, +20%.
2. Temporal jitter: shift rainfall pulse by +/- 1 timestep.
3. Static channel dropout: one channel removed at inference, one at a time.
4. Static raster noise: low Gaussian perturbation to each static channel.

Required outputs:
1. Relative drop in RMSE, F1, CSI for each perturbation.
2. Worst-case and median degradation per split.

Acceptance condition:
1. No catastrophic collapse under moderate perturbations.
2. Median F1 drop under +/- 10% scaling <= 5 percentage points.

## 6) Latent Representation Analysis (Minimum)

1. Extract event embeddings and fused latent vectors for test samples.
2. Visualize with UMAP (fixed random seed).
3. Run linear probe from latent vectors to predict at least two known attributes:
4. Peak-depth class.
5. Event rainfall-volume class.
6. Quantify representation stability across seeds using CKA or nearest-neighbor overlap.

Acceptance condition:
1. Probe accuracy exceeds random baseline by a meaningful margin.
2. Representation stability is reported and not omitted.

## 7) Physics Consistency Checks (Minimum)

1. Monotonic stress test:
2. Increase rainfall forcing in controlled increments.
3. Verify inundated area and mean depth generally increase.
4. Hydro-geomorphic consistency:
5. Confirm deeper predictions concentrate in lower relative-elevation and higher flow-accumulation zones.

Acceptance condition:
1. Monotonic pass rate and geomorphic consistency statistics are explicitly reported.

## 8) Statistical Rigor (Minimum)

1. Bootstrap 95% confidence intervals for all headline metrics.
2. Paired significance test against the current baseline model.
3. Report both effect size and p-value.

Acceptance condition:
1. Claims of improvement must include confidence intervals and significance.

## 9) Required Figures for Publication (Minimum)

1. Main performance table with 95% confidence intervals (in-domain, OOD, spatial holdout).
2. Wet reliability diagram with ECE and Brier.
3. Depth interval coverage plot (nominal vs empirical).
4. Robustness heatmap of metric degradation.
5. Depth-bin error chart.
6. UMAP latent map with regime coloring.
7. Physics consistency summary figure.

## 10) Minimal Go/No-Go Criteria

A model version is publishable only if all criteria below are met:

1. Meets or exceeds current benchmark for depth RMSE and wet F1 on in-domain test.
2. Maintains acceptable OOD performance without failure modes.
3. Passes uncertainty calibration gates (ECE and PI90 coverage targets).
4. Passes robustness gates for moderate perturbations.
5. Includes latent and physics consistency evidence with quantitative results.
6. Includes confidence intervals and significance tests for all core claims.

## 11) Recommended Execution Order

1. Freeze splits and evaluation scripts.
2. Train 5-seed ensemble.
3. Run baseline and uncertainty/calibration metrics.
4. Run robustness battery.
5. Run latent and physics checks.
6. Produce publication tables and figures.
7. Perform statistical testing and finalize conclusions.

## 12) Out of Scope for Minimum Plan

1. Full conformalized uncertainty study.
2. Large ablation grid over architecture variants.
3. Comprehensive explainability suite beyond latent probes.
4. Multi-basin transfer-learning campaign.

These can be added later as extensions, but they are not required for a minimum publishable validation package.
