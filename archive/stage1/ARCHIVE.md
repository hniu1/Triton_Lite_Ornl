# Stage 1 Archive

Stage 1 was frozen when active development moved to the
`Stage2_GNN_Prediction` branch. This directory preserves:

- `code/`: complete timestamp and state-transition implementations;
- `docs/`: model descriptions, experiment reports, and component audit;
- `workflows/stage1/`: timestamp-surrogate Slurm jobs;
- `workflows/stage1_transition/`: V1 through V4b transition jobs;
- `tests/`: the original Stage-1 regression suite;
- `PROJECT_GUIDE.md` and `README.md`: the original Stage-1 navigation and
  project description.

Checkpoints and metrics remain in their original `results/stage1_*` paths to
avoid copying large artifacts and to preserve Stage-2 baseline references.
Shared input preparation remains under the repository-level
`data_preprocessing/` directory.

Archived Python entry points can be run directly from the repository root, for
example:

```bash
python archive/stage1/code/stage1_transition_rollout.py --help
```

The archived script directory is placed on Python's import path automatically,
so its historical absolute imports continue to resolve without root-level
compatibility files. Active V5 code uses the neutral `triton_core/` package.

Run the archived regression suite independently with:

```bash
PYTHONPATH=archive/stage1/code:. python -m unittest discover -s archive/stage1/tests
```
