# Stage 2 V5 Forecasting Plan

## Objective

Predict 10 m hydraulic maps at every half-hour lead from 0.5 through 24 hours.
The first V5 experiment is an in-domain Conasauga model-development result, not
a claim of transfer to an unseen regional domain.

## Frozen comparison contract

- Forecast origin: observed TRITON state at time `t`, plus the state at `t-1`.
- Target: depth, HU, and HV at `t + lead`.
- Leads: 1 through 48 half-hour steps.
- Forcing: known through the target valid time; the causal encoder cannot use
  forcing after the target.
- Intermediate hydraulic states: unavailable to the direct model.
- Baselines: full-state persistence, V4b recursive learned components, V4b
  velocity persistence, and V4b component persistence.
- Spatial cohort: identical max-48-step-eligible origins within each split.
- Metrics: all-cell and wet-cell depth, wet precision/recall/F1/CSI, HU/HV,
  derived velocity, and the established physical score.

Checkpoint selection uses the mean validation score over fixed leads 1, 6, 12,
24, and 48. It does not use D030 test metrics.

## Experiment sequence

| ID | Experiment | Structural question |
|---|---|---|
| V4b-D1 | Velocity persistence | Is learned velocity causing rollout drift? |
| V4b-D2 | Component persistence | Is learned HU/HV causing rollout drift? |
| V5-A | Direct lead-conditioned tile model | Does removing recursion fix long-lead accumulation? |
| V5-B | Input halo, central-tile output | How much error is missing immediate neighbor state? |
| V5-C | Block-specific forcing | Is global compression hiding spatial forcing effects? |
| V5-D | Hierarchical tile/HUC12 graph | Does watershed communication improve propagation? |
| V5-E | Coupled variables and flux constraints | Does structural physics fix velocity and mass errors? |
| V5-F | Targeted attention/operator ablation | Does a newer backbone add value after inputs are fixed? |

Only one structural change is promoted at a time. Each candidate must use the
same split, origins, leads, persistence definitions, and metrics.

## Event-held-out ablation protocol

The 40 events are assigned to five complete-event folds, balanced by forcing
severity. The first screening run holds out all eight events in fold 0 and
compares four nested variants: direct baseline, 8-cell hydraulic halo, halo
plus velocity-direction/boundary-flux supervision, and the same model with a
tile-local forcing encoder. V4b initialization is disabled for these runs to
prevent weights previously trained on fold events from leaking into the test.

The localized signal uses inverse-distance weights over the eight nearest
ordered `Loc` points from the 300-point forcing shapefile. The global forcing
encoder remains present, making this an added local signal rather than a loss
of regional context. Full five-fold training is reserved for the winning
screening formulation.

## V5-A implementation

The model computes

```text
S_hat(t + lead) = S(t) + Delta_theta(
    S(t), S(t-1), terrain, forcing[0:t+lead], t, lead
)
```

It reuses the V4b temporal and tile encoders, history adapter, decoder, and
heads. New origin-time and lead-time encoders are added. Compatible V4b weights
initialize V5; new conditioning columns begin at zero, so initialization is
well defined rather than silently randomizing the old conditioning path.

The model is trained with a uniformly sampled lead per spatial batch. Validation
and testing are deterministic and use a common 48-step-eligible cohort.

## Data interpretation

The pipeline label `watershed_id=conasauga` denotes one regional static domain.
The repository also contains 34 HUC12 subwatersheds. The later graph experiment
will use tiles as fine nodes and HUC12 units as coarse routing nodes.

The existing 40 events are sufficient for V5 development and controlled
in-domain comparisons. A final reliability claim requires newly generated,
untouched TRITON events and eventually additional regional domains.

## Commands

```bash
# Two no-training V4b diagnostics as a two-task array
sbatch workflows/stage2/00_v4b_component_policy_diagnostics.sh

# Validate the V5 data/model/training path
sbatch workflows/stage2/01_v5_direct_smoke.sh

# After the smoke succeeds
train_job=$(sbatch --parsable workflows/stage2/02_v5_direct_train.sh)
sbatch --dependency=afterok:${train_job} \
  workflows/stage2/03_v5_direct_dense_24h_eval.sh

# New event-held-out screening ladder (smoke -> 4 training tasks -> 4 evals)
smoke_job=$(sbatch --parsable workflows/stage2/04_v5b_enhanced_smoke.sh)
ablation_job=$(sbatch --parsable --dependency=afterok:${smoke_job} \
  workflows/stage2/05_v5_ablation_fold0.sh)
sbatch --dependency=aftercorr:${ablation_job} \
  workflows/stage2/06_v5_ablation_fold0_eval.sh
```
