#!/bin/bash
set -euo pipefail

cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes

python stage1_transition_operational_gate.py \
  --candidate-rollout results/stage1_transition_multistep_v1/rollout_metrics.json \
  --reference-rollout results/stage1_transition_v1_continued/rollout_metrics.json \
  --candidate-regimes results/stage1_transition_multistep_v1/regime_metrics.json \
  --reference-regimes results/stage1_transition_v1_continued/regime_metrics.json \
  --output-path results/stage1_transition_multistep_v1/operational_acceptance.json

python stage1_transition_whole_area_compare.py \
  --candidate-csv results/stage1_transition_multistep_v1/D030_whole_area_key_times/whole_area_metrics.csv \
  --reference-csv results/stage1_transition_v1_continued/D030_whole_area_key_times/whole_area_metrics.csv \
  --output-path results/stage1_transition_multistep_v1/D030_whole_area_key_times/whole_area_acceptance.json
