# Project Guide: Stage 2 V5

**6 October 2026:** confirmed legacy target ordering/sign defects. Training and
forecast evaluation are on hold until corrected assets are rebuilt/validated.
See [target-data audit](docs/stage2/TARGET_DATA_AUDIT.md). Old scores and graph
diagnostics cannot establish scientific model promotion.

## Start here

| Location | Purpose |
|---|---|
| `docs/stage2/LEADERSHIP_BRIEFING.md` | Presentation outline, completed results, and gated roadmap |
| `docs/stage2/RIKER_EXECUTION.md` | Current Riker resources and retained Python environment |
| `docs/stage2/V5D_GRAPH_COMMUNICATION.md` | Graph hypothesis, architecture, implementation steps and gates |
| `docs/stage2/STAGE2_V5_PLAN.md` | Frozen forecast/evaluation contract and experiment order |
| `stage2/data.py` | Direct origin/target pairing and lead sampling |
| `stage2/model.py` | Lead-conditioned V5-A tile model |
| `stage2/graph.py` | Optional tile communication; not conservative flux exchange |
| `stage2/graph_data.py` | Raster adjacency and connected-neighborhood batches |
| `stage2/train_direct.py` | Training and validation-horizon checkpoint selection |
| `stage2/evaluate_direct.py` | Dense half-hour evaluation through 24 hours |
| `triton_core/` | Shared active data loader, encoder primitives, metrics, and runtime |
| `workflows/stage2/` | Active Slurm workflows |
| `archive/stage1/` | Frozen Stage-1 implementation, docs, and workflows |

## Active execution order

Compute platform: **Riker**. Existing results below were generated on Andes.
Use `14_riker_environment_smoke.sh` to validate the existing environment on an
allocated L40S GPU before new experiments. See `RIKER_EXECUTION.md` for
resource requests; historical runtime/concurrency estimates do not apply.

Current graph smoke chain: environment job `186107` followed by graph job
`186108` (`15_v5d_graph_riker_smoke.sh`). These are pipeline checks, not full
training or a graph-versus-U-Net performance comparison.
Diagnostic `186109` completed but reduced loss while map metrics worsened.
The revised diagnostic is job `186110` (`16_v5d_graph_learning_check.sh`):
balanced flood-expansion/recession learning check and GPU-cost measurements. The guarded
three-arm screening recipe `17_v5d_controlled_screening.sh` is prepared but
not submitted. See `V5D_GRAPH_COMMUNICATION.md` for gates and interpretation.
Job `186110` finished with results but deliberately failed the quality gate:
depth/extent learned, discharge/velocity worsened. Larger runs remain on hold
pending loss/dynamics diagnostics; see its `SCIENTIFIC_REVIEW.md`.

1. `00_v4b_component_policy_diagnostics.sh`
2. `01_v5_direct_smoke.sh`
3. `02_v5_direct_train.sh`
4. `03_v5_direct_dense_24h_eval.sh`

The full training job must depend on successful smoke completion. Dense
evaluation must depend on successful training completion.

## Model promotion order

1. V5-A: direct lead-conditioned tile model
2. V5-B: larger input halo with central `80 x 80` output
3. V5-C: block-specific forcing aggregation or cross-attention
4. V5-D: hierarchical tile/HUC12 graph
5. V5-E: coupled hydraulic variables and mass/boundary-flux constraints
6. V5-F: targeted Transformer or neural-operator comparison

Do not combine two structural changes in one promotion experiment. Preserve the
same event split, 48-step-eligible origins, persistence definitions, and metric
implementation.

## Stage-1 archive policy

Stage-1 code is frozen under `archive/stage1/code/`; its tests and workflows are
archived alongside it. Active reusable components live under `triton_core/`.
There are no Stage-1 compatibility modules at the repository root. New model
work belongs in `stage2/`.

Stage-1 results remain in `results/stage1_*` so comparison jobs do not duplicate
large checkpoints. Shared preprocessing stays under `data_preprocessing/`.

## Testing

```bash
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
python -m unittest discover -s tests
bash -n workflows/stage2/*.sh
```
