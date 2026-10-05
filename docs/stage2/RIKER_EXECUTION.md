# Stage 2 execution on OLCF Riker

Verified from `riker-login1.olcf.ornl.gov` on 5 October 2026.

The existing Python environment is retained:

```bash
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/cli138/proj-shared/7hn/envs/triton_andes
```

Despite its historical name, this environment imports PyTorch 2.5.1 with a
CUDA 11.8 build, NumPy, pandas, netCDF4, and Matplotlib on Riker's login host.
GPU execution must be checked on an allocated compute node.

Riker's Slurm `gpu` partition has eight GPU nodes, each advertising two NVIDIA
L40S GPUs. `batch` is the default CPU partition; `test` is also available.
The scheduler reports 23,444 MiB maximum memory per requested CPU on the GPU
partition. Submission-policy checks additionally require at least 16 CPUs
and 375,104 MiB memory per GPU. The prepared request uses
`--cpus-per-gpu=16` and `--mem-per-gpu=375104M`, rather than historical
per-task/node flags. The original eight-CPU/96-GiB request was rejected.

The prepared preflight uses account `cli138`, one L40S GPU, 16 CPUs, and
375,104 MiB (about 366 GiB):

```bash
sbatch workflows/stage2/14_riker_environment_smoke.sh
```

It prints GPU/driver details, executes GPU forward/backward, and runs active
regression tests. Successful execution establishes environment compatibility;
it does not establish training throughput. It has not yet been submitted.
`sbatch --test-only` accepted the corrected request on 5 October 2026; this
validates submission eligibility, not actual GPU runtime compatibility.
All 13 active regression tests passed on the login host. Its CUDA stub-driver
warning is another reason to verify CUDA on an allocated GPU node.

Before new training, benchmark the unchanged V5-B checkpoint on a small frozen
cohort and record elapsed time, GPU memory, CPU/memory requests, and numerical
agreement with saved results. Measure training/evaluation time on Riker before
setting budgets. Andes timings and the previous one-job concurrency observation
are historical, not verified Riker limits.

Historical workflows remain reproduction records. New Riker workflows should
use these resources and the same model/data paths. Use new output directories
when benchmarking so old results remain available.
