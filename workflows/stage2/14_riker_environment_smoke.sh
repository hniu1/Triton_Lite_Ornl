#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_riker_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=375104M
#SBATCH -t 00:30:00
#SBATCH -o ./slurm_output/stage2-riker-smoke-%j.out
#SBATCH -e ./slurm_output/stage2-riker-smoke-%j.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/cli138/proj-shared/7hn/envs/triton_andes
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-matplotlib-${SLURM_JOB_ID}"
hostname
nvidia-smi
python -u -c 'import torch; from stage2.model import Stage2DirectLeadModel; print("torch:", torch.__version__, "CUDA build:", torch.version.cuda); assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0)); x=torch.randn(64,64,device="cuda",requires_grad=True); x.square().mean().backward(); torch.cuda.synchronize(); print("GPU forward/backward passed")'
python -m unittest discover -s tests
