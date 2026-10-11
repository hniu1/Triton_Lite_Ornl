#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_graph_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=375104M
#SBATCH -t 01:00:00
#SBATCH -o ./slurm_output/stage2-graph-smoke-%j.out
#SBATCH -e ./slurm_output/stage2-graph-smoke-%j.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/cli138/proj-shared/7hn/envs/triton_andes
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-matplotlib-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
graph_smoke_dir="results/stage2_v5d_graph_smoke/job_${SLURM_JOB_ID}"
hostname
nvidia-smi
python -m unittest discover -s tests
python -u -m stage2.train_direct \
  --initial-run-dir results/stage1_transition_multistep_v4b \
  --output-dir "${graph_smoke_dir}" \
  --test-events D005 D014 D018 D023 D024 D028 D033 D037 \
  --data-split-seed 4300 --seed 45 --no-v4b-initialization \
  --hydraulic-halo-cells 8 --graph-layers 2 --graph-hops 2 \
  --epochs 1 --batch-size 1 --train-batches-per-epoch 4 \
  --selection-batches 1 --selection-horizons 1 6 24 48 \
  --backbone-learning-rate 0.0001 --new-conditioning-learning-rate 0.0001 \
  --netcdf-chunk-cache-mb 32 --max-open-netcdf-handles 4 \
  --num-workers 0 --device cuda --disable-cudnn
python -u -m stage2.evaluate_direct \
  --run-dir "${graph_smoke_dir}" --split val \
  --horizons 1 6 24 48 --evaluation-batches 1 --batch-size 1 \
  --device cuda --disable-cudnn
python -u -m stage2.verify_graph_smoke --run-dir "${graph_smoke_dir}"
