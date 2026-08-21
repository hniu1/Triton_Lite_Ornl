# Scheduler Workflows

- `stage1/` contains the active timestamp-conditioned pipeline in execution
  order. See `PROJECT_GUIDE.md` before submitting jobs.
- `blockwise/` contains the legacy event-peak baseline.
- `uncertainty/` contains optional ensemble analysis for the legacy baseline.

All jobs change to the repository root before invoking Python, so they should
be submitted from the repository root using their full path, for example:

```bash
sbatch workflows/stage1/06b_stage1_calibrate_wet_threshold.sh
```

