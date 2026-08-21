#!/bin/bash
# compute_uncertainty_ensemble.sh
# Submission script for multi-seed ensemble uncertainty analysis on ORNL Andes

#SBATCH -A cli138
#SBATCH -J ens_unc_bw
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -o ./slurm_output/ensemble-unc-%j.out
#SBATCH -e ./slurm_output/ensemble-unc-%j.err

set -euo pipefail

module load cuda/11.0.2
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes

cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl

mkdir -p slurm_output

PYTHON_BIN=/lustre/orion/proj-shared/cli138/7hn/envs/triton_andes/bin/python

REFERENCE_RUN=${REFERENCE_RUN:-results/results_blockwise_matrix_train_v3}
OUTPUT_DIR=${OUTPUT_DIR:-results/results_ensemble_uncertainty_v3}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints}
NUM_SEEDS=${NUM_SEEDS:-5}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}
BASE_SEED=${BASE_SEED:-42}
PI_LEVELS=${PI_LEVELS:-0.80 0.90 0.95}

# Train 5-seed ensemble and compute uncertainty
${PYTHON_BIN} -u compute_uncertainty_ensemble.py \
    --reference-run "${REFERENCE_RUN}" \
    --num-seeds "${NUM_SEEDS}" \
    --output-dir "${OUTPUT_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --base-seed "${BASE_SEED}" \
    --pi-levels ${PI_LEVELS}

echo "Done!"
