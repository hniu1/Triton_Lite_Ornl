# Flood Model Status and Challenges Summary

## Bottom line
The current weakness appears to be a combination of model-objective design issues and threshold/selection/calibration issues, not just a single bad hyperparameter.

The blockwise 10m concept is still likely valid, but the current optimization and evaluation setup is not reliably optimizing spatial flood extent.

## Current model (V2)
- Architecture: dual-head blockwise matrix model (`BlockwiseFloodMatrixModel`).
- Inputs: event time-series features + static block features.
- Outputs:
  - Depth head: continuous flood depth map.
  - Wet head: wet/dry logits/probabilities.
- Loss:
  - Depth-weighted masked Huber.
  - Auxiliary wet/dry BCE.
  - Combined objective with tunable auxiliary weight and depth weighting.

## What we observed
- Visually, predicted flooded area is too sparse under default wet thresholding.
- At wet threshold 0.5, wet prediction has very high precision but poor recall, so extent is severely under-predicted.
- Threshold calibration improves extent visualization (best F1 for D040 occurred at a very low threshold), but depth quality remains weak in many areas.
- There is metric inconsistency between:
  - Inference summary metrics (`results_blockwise_matrix_predictions_test_v2/metrics.json`).
  - Cell-level reconstructed diagnostics (`results_blockwise_matrix_predictions_test_v2/plots/depth_bin_metrics_cells.csv`).

This inconsistency means model quality can look very different depending on which metric pipeline is used.

## Diagnosis: design vs parameter selection
This is likely both:
- Not only parameter tuning.
- Not only architecture.

Most likely stacked issues:
1. Objective mismatch: current loss can improve while extent remains poor.
2. Threshold mismatch: default wet threshold (0.5) is not calibrated for this event/data distribution.
3. Checkpoint selection mismatch: selecting by aggregate validation loss does not guarantee good extent quality.
4. Evaluation contract mismatch: competing metric pipelines disagree, so optimization target is ambiguous.

## Core challenge
We need one model that is simultaneously good at:
1. Flood extent detection (where water is present).
2. Flood depth estimation (how deep it is).

Right now, training and selection do not enforce this trade-off strongly or consistently.

## Recommended next actions (priority order)
1. Fix evaluation contract first.
   - Define one authoritative metric pipeline for model selection and reporting.
   - Ensure training-time and reconstructed cell-level evaluation are consistent.

2. Add extent metrics to checkpoint selection.
   - Include wet F1, CSI, and recall (at calibrated threshold), not RMSE alone.

3. Keep threshold calibration explicit.
   - Calibrate wet threshold per event/watershed for fair extent reporting.

4. Retrain V3 with stronger extent emphasis.
   - Increase auxiliary wet loss weight.
   - Rebalance depth weighting so deep-channel focus does not collapse extent.

5. Re-run D040 with fixed evaluation.
   - Compare baseline vs V2/V3 using one trusted script and both depth + extent metrics.

## Practical interpretation for collaborators
The current results should be treated as "promising but not deployment-ready".
Primary blocker is spatial extent reliability, plus metric definition inconsistency. Until evaluation is unified and checkpoint selection includes extent quality, model iterations may continue to optimize the wrong objective.
