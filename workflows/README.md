# Scheduler Workflows

- `stage2/` contains the V5 direct-horizon experiments and the controlled V4b
  component-policy diagnostics. See `docs/stage2/STAGE2_V5_PLAN.md` for the
  frozen evaluation contract and experiment order.
- `blockwise/` contains the legacy event-peak baseline.
- `uncertainty/` contains optional ensemble analysis for the legacy baseline.
- Archived Stage-1 timestamp and transition workflows are under
  `archive/stage1/workflows/`. They are retained for reproducibility and are no
  longer active development targets.

All jobs change to the repository root before invoking Python, so they should
be submitted from the repository root using their full path, for example:

```bash
sbatch workflows/stage2/01_v5_direct_smoke.sh
```
