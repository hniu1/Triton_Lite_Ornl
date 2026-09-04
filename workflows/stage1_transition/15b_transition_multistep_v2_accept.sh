#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_multi_v2_gate
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o ./slurm_output/stage1-transition-multistep-v2-gate-%j.out
#SBATCH -e ./slurm_output/stage1-transition-multistep-v2-gate-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

candidate_dir=results/stage1_transition_multistep_v2
continued_dir=results/stage1_transition_v1_continued
multistep_v1_dir=results/stage1_transition_multistep_v1

# Continued V1 is the stable-behavior reference; Multi-step V1 is the dynamic
# transition reference. Run both gates so V2 cannot hide a regression behind a
# single aggregate comparison.
python stage1_transition_operational_gate.py \
  --candidate-rollout "$candidate_dir/rollout_metrics.json" \
  --reference-rollout "$continued_dir/rollout_metrics.json" \
  --candidate-regimes "$candidate_dir/regime_metrics.json" \
  --reference-regimes "$continued_dir/regime_metrics.json" \
  --output-path "$candidate_dir/operational_acceptance_vs_continued_v1.json"

python stage1_transition_operational_gate.py \
  --candidate-rollout "$candidate_dir/rollout_metrics.json" \
  --reference-rollout "$multistep_v1_dir/rollout_metrics.json" \
  --candidate-regimes "$candidate_dir/regime_metrics.json" \
  --reference-regimes "$multistep_v1_dir/regime_metrics.json" \
  --output-path "$candidate_dir/operational_acceptance_vs_multistep_v1.json"

python stage1_transition_whole_area_compare.py \
  --candidate-csv "$candidate_dir/D030_whole_area_key_times/whole_area_metrics.csv" \
  --reference-csv "$continued_dir/D030_whole_area_key_times/whole_area_metrics.csv" \
  --output-path "$candidate_dir/D030_whole_area_key_times/whole_area_acceptance_vs_continued_v1.json"

python stage1_transition_whole_area_compare.py \
  --candidate-csv "$candidate_dir/D030_whole_area_key_times/whole_area_metrics.csv" \
  --reference-csv "$multistep_v1_dir/D030_whole_area_key_times/whole_area_metrics.csv" \
  --output-path "$candidate_dir/D030_whole_area_key_times/whole_area_acceptance_vs_multistep_v1.json"
