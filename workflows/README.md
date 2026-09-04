# Scheduler Workflows

- `stage1/` contains the active timestamp-conditioned pipeline in execution
  order. See `PROJECT_GUIDE.md` before submitting jobs.
- `blockwise/` contains the legacy event-peak baseline.
- `uncertainty/` contains optional ensemble analysis for the legacy baseline.
- `stage1_transition/` contains the state-aware Transition V1 prototype,
  multi-step diagnostics, the rejected V2 regime-aware experiment, and the V3
  exact-local/12-step workflow, V4a two-state gate experiment, and V4b
  fast-adapter follow-up. See `docs/stage1/STAGE1_TRANSITION_V1.md`,
  `docs/stage1/STAGE1_TRANSITION_V2.md`, and
  `docs/stage1/STAGE1_TRANSITION_V3.md`,
  `docs/stage1/STAGE1_TRANSITION_V4A.md`, and
  `docs/stage1/STAGE1_TRANSITION_V4B.md`. V3 uses workflows 16b-23; workflow 19
  must remain gated on a successful workflow-18 smoke, and workflows 20-23 are
  evaluation/promotion checks rather than training steps. V4a uses workflows
  24-29; workflow 30 is the controlled V4b smoke and must pass before a full
  V4b chain is promoted.

All jobs change to the repository root before invoking Python, so they should
be submitted from the repository root using their full path, for example:

```bash
sbatch workflows/stage1/06b_stage1_calibrate_wet_threshold.sh
```
