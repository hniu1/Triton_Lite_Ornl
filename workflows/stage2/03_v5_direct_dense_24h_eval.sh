#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_v5_eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o ./slurm_output/stage2-v5-direct-eval-%j.out
#SBATCH -e ./slurm_output/stage2-v5-direct-eval-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

python -u -m stage2.evaluate_direct \
  --run-dir results/stage2_v5_direct \
  --checkpoint best_model.pt \
  --output-path results/stage2_v5_direct/test_dense_24h_direct_metrics.json \
  --split test --horizons {1..48} \
  --evaluation-batches 100 --batch-size 16 --start-time-stride 12 \
  --device cuda --disable-cudnn

