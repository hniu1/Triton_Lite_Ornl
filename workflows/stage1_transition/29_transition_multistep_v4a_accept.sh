#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_v4a_gate
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o ./slurm_output/stage1-transition-v4a-gate-%j.out
#SBATCH -e ./slurm_output/stage1-transition-v4a-gate-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

candidate=results/stage1_transition_multistep_v4a
v1=results/stage1_transition_multistep_v1
v3=results/stage1_transition_multistep_v3

for reference in v1 v3; do
  if [[ "$reference" == v1 ]]; then reference_dir=$v1; else reference_dir=$v3; fi
  python stage1_transition_operational_gate.py \
    --candidate-rollout "$candidate/rollout_metrics.json" \
    --reference-rollout "$reference_dir/rollout_metrics.json" \
    --candidate-regimes "$candidate/regime_metrics.json" \
    --reference-regimes "$reference_dir/regime_metrics.json" \
    --output-path "$candidate/operational_acceptance_vs_${reference}.json"
  python stage1_transition_whole_area_compare.py \
    --candidate-csv "$candidate/D030_whole_area_key_times/whole_area_metrics.csv" \
    --reference-csv "$reference_dir/D030_whole_area_key_times/whole_area_metrics.csv" \
    --output-path "$candidate/D030_whole_area_key_times/whole_area_acceptance_vs_${reference}.json"
done
