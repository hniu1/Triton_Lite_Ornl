#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_depth_bins
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o ./slurm_output/stage1-depth-bins-%j.out
#SBATCH -e ./slurm_output/stage1-depth-bins-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes

CURRENT=results/stage1_timestamp_max
PREVIOUS=results/stage1_timestamp_stratified
PLOTS=${CURRENT}/plots

python -u stage1_depth_bin_evaluate.py --run-dir "${PREVIOUS}" --output-path "${CURRENT}/previous_depth_bin_metrics.json" --device cuda
python -u stage1_depth_bin_evaluate.py --run-dir "${CURRENT}" --output-path "${CURRENT}/depth_bin_metrics.json" --device cuda
python -u plot/plot_stage1_depth_bins.py --previous "${CURRENT}/previous_depth_bin_metrics.json" --current "${CURRENT}/depth_bin_metrics.json" --output-dir "${PLOTS}"
