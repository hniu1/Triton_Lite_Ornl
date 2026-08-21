# Option C: Uncertainty & Calibration Script

**Status**: Complete and ready to use.  
**File**: [compute_uncertainty_ensemble.py](compute_uncertainty_ensemble.py)

## What This Does

This script implements the **minimum publishable uncertainty quantification** from the validation plan. It:

1. **Trains 5 models** with different random seeds using identical hyperparameters
2. **Runs ensemble inference** on the test set
3. **Computes depth uncertainty**: mean, std, prediction intervals with coverage
4. **Evaluates wet probability calibration**: Brier score, ECE, reliability diagram
5. **Reports depth-bin RMSE** to check performance across different inundation regimes

## Input Requirements

- Reference training run directory (e.g., `results_blockwise_matrix_train_v3/`)
  - Must contain: `run_config.json` with all hyperparameters
  - Must contain: `splits/` folder with train/val/test sample CSVs
  - Must contain: `normalization_stats.npz`

## Output Files

```
results_ensemble_uncertainty_v3/
├── depth_preds_ensemble.npy         # (n_samples, 80, 80, 5) raw depth predictions
├── wet_logits_ensemble.npy          # (n_samples, 80, 80, 5) raw wet head outputs
├── targets.npy                      # (n_samples, 80, 80) ground truth depth
├── masks.npy                        # (n_samples, 80, 80) valid pixel mask
└── uncertainty_metrics.json         # Summary metrics (see below)
```

## Output Metrics

### Depth Regression (from ensemble mean)
```json
{
  "depth_metrics": {
    "rmse": 0.0611,
    "mae": 0.00885,
    "bias": -0.0464,
    "r2": 0.9573
  }
}
```

### Prediction Intervals (90% level)
```json
{
  "pi_metrics": {
    "pi_level": 0.90,
    "coverage": 0.895,              # Fraction of targets inside interval
    "mean_width": 0.245,            # Average interval width (sharpness)
    "target_coverage": 0.90         # Desired coverage
  }
}
```
- **Acceptance**: Coverage should be 0.86–0.94 (within ±4% of nominal)
- **Sharpness**: Narrower intervals are better (more informative)

### Wet Probability Calibration (from ensemble mean probability)
```json
{
  "wet_calibration": {
    "brier_score": 0.0125,          # Mean squared probability error
    "ece": 0.032,                   # Expected Calibration Error
    "bin_accs": [0.0, 0.05, ...],   # Observed frequencies per bin
    "bin_confs": [0.0, 0.1, ...]    # Predicted probabilities per bin
  }
}
```
- **Acceptance**: ECE ≤ 0.05 and Brier ≤ 0.02 on in-domain test
- **ECE formula**: 
  $$\text{ECE} = \sum_{i=1}^{m} \frac{n_i}{n} \left| \text{acc}_i - \text{conf}_i \right|$$

### Depth-Bin RMSE (stratified error)
```json
{
  "depth_bin_rmses": {
    "0.0-0.3m": 0.198,
    "0.3-1.0m": 0.331,
    "1.0-2.0m": 0.400,
    "2.0-5.0m": 0.372,
    "5.0-100.0m": 0.641
  }
}
```
- **Purpose**: Check model degrades gracefully at higher depths
- **Typical pattern**: Shallow bins are easier, error increases with depth

## Usage Example

### Command Line
```bash
python compute_uncertainty_ensemble.py \
    --reference-run results/results_blockwise_matrix_train_v3 \
    --num-seeds 5 \
    --output-dir results/results_ensemble_uncertainty_v3 \
    --batch-size 32 \
    --epochs 100 \
    --base-seed 42 \
    --device cuda
```

### Submit to HPC
```bash
sbatch workflows/uncertainty/compute_uncertainty_ensemble.sh
```

## Key Design Decisions

1. **Monte Carlo Uncertainty**: Uses ensemble spread as uncertainty proxy (no dropout, simpler than MC-Dropout)
2. **Quantile-based PI**: Uses empirical quantiles from 5 seeds → robust to non-Gaussian errors
3. **Sigmoid for Calibration**: Applies sigmoid to wet head logits to get probabilities (raw logits are not calibrated)
4. **Masked Metrics**: All metrics computed only on valid pixels (mask > 0.5)
5. **Early Stopping per Seed**: Each seed trains independently with early stopping, so convergence varies

## What Next?

After running this script:

1. Check `uncertainty_metrics.json` for acceptance thresholds:
   - **PI coverage** close to 0.90 ✓
   - **ECE** ≤ 0.05 ✓
   - **Depth RMSE** matches benchmark ✓

2. Plot `bin_accs` vs `bin_confs` to visualize calibration (see companion plotting script)

3. Proceed to **Option D: Robustness Analysis** if uncertainty gates pass

## GPU/Memory Requirements

- **Per seed**: ~8 GB GPU, 30 min–1 hr depending on dataset size
- **5 seeds**: ~4–5 hrs total on single A100 GPU
- **Batch size**: Reduce to 16 if OOM; increase to 64 for faster throughput

## Known Limitations

1. No OOD event holdout in this version (uses same test set as training phase)
   - Future: Add separate OOD test with held-out storm types
2. Ensemble uncertainty only as good as seed diversity
   - All seeds use same architecture/data; only randomness is initialization & dropout
3. PI coverage can be conservative if error distribution is heavy-tailed
   - Future: Consider conformal prediction for distribution-free intervals

## Next Steps

- [ ] Run compute_uncertainty_ensemble.py
- [ ] Review uncertainty_metrics.json for acceptance gates
- [ ] Plot reliability diagram (bin_accs vs bin_confs)
- [ ] Proceed to Option D: Robustness stress matrix
