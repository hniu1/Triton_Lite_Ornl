#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_graph_screen
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=375104M
#SBATCH --array=0-2
#SBATCH -t 08:00:00
#SBATCH -o ./slurm_output/stage2-graph-screen-%A-%a.out
#SBATCH -e ./slurm_output/stage2-graph-screen-%A-%a.err

# Prepared screening recipe, NOT submitted automatically. Review benchmark
# memory/runtime first. Pass the successful learning-check report as argument 1.
set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/cli138/proj-shared/7hn/envs/triton_andes
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-matplotlib-${SLURM_JOB_ID}"
learning_report="${1:?Pass the reviewed, successful LEARNING_CHECK_RESULTS.json}"
python -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"] == "passed" and r.get("promotion_ready",False), "Learning/map gates not passed"; assert all(a["learning_gate_passed"] and a["map_learning_gate_passed"] for a in r["arms"].values())' "${learning_report}"
case "${SLURM_ARRAY_TASK_ID}" in
  0) arm=unet; layers=0; mode=neighbors ;;
  1) arm=local_control; layers=2; mode=self ;;
  2) arm=neighbor_graph; layers=2; mode=neighbors ;;
  *) exit 2 ;;
esac
screen_dir="results/stage2_v5d_screening/job_${SLURM_ARRAY_JOB_ID}/${arm}"
python -u -m stage2.train_direct \
  --initial-run-dir results/stage1_transition_multistep_v4b \
  --output-dir "${screen_dir}" \
  --test-events D005 D014 D018 D023 D024 D028 D033 D037 \
  --data-split-seed 4300 --seed 45 --no-v4b-initialization \
  --hydraulic-halo-cells 8 --graph-layers "${layers}" --graph-mode "${mode}" --graph-hops 2 \
  --epochs 3 --batch-size 4 --train-batches-per-epoch 250 \
  --selection-batches 40 --selection-horizons 1 6 12 24 48 \
  --backbone-learning-rate 0.0001 --new-conditioning-learning-rate 0.0001 \
  --netcdf-chunk-cache-mb 32 --max-open-netcdf-handles 4 \
  --num-workers 0 --device cuda --disable-cudnn
python -u -m stage2.evaluate_direct \
  --run-dir "${screen_dir}" --split val \
  --evaluation-batches 80 --batch-size 4 --per-event --device cuda --disable-cudnn
