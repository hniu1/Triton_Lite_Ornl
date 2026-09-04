#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s1_pair_parallel
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH -t 06:00:00
#SBATCH -o ./slurm_output/stage1-paired-index-parallel-%j.out
#SBATCH -e ./slurm_output/stage1-paired-index-parallel-%j.err

set -euo pipefail
cd /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/triton_andes
mkdir -p slurm_output

source_root=${SOURCE_ROOT:-processed_data/timestamp_stage1/m4_sampling_index_dense}
output_root=${OUTPUT_ROOT:-processed_data/timestamp_stage1/m6_paired_transition_index}
parallel_events=${PARALLEL_EVENTS:-8}

if (( parallel_events < 1 || parallel_events > 8 )); then
  echo "PARALLEL_EVENTS must be between 1 and 8" >&2
  exit 2
fi

for batch_start in 1 9 17 25 33; do
  pids=()
  labels=()
  batch_stop=$((batch_start + 7))
  for event_number in $(seq "${batch_start}" "${batch_stop}"); do
    event_id=$(printf 'D%03d' "${event_number}")
    shard_dir="${output_root}/shards/${event_id}"
    if [[ -s "${shard_dir}/sampling_candidates.parquet" && -s "${shard_dir}/sampling_metadata.json" ]]; then
      echo "Skipping completed ${event_id}" >&2
      continue
    fi
    while (( ${#pids[@]} >= parallel_events )); do
      batch_status=0
      for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
          batch_status=1
        fi
      done
      if (( batch_status != 0 )); then
        echo "At least one paired-index worker failed" >&2
        exit 1
      fi
      pids=()
      labels=()
    done
    srun --exclusive -N 1 -n 1 -c 2 \
      python -u data_preprocessing/m6_build_paired_transition_index.py \
        --manifest-dir processed_data/timestamp_stage1/m3_dynamic_manifest \
        --labels-10m-dir processed_data/timestamp_stage1/m3_labels_10m \
        --static-rasters-dir processed_data/timestamp_stage1/m2_5_static_rasters \
        --input-candidates "${source_root}/shards/${event_id}/sampling_candidates.parquet" \
        --output-dir "${shard_dir}" --event-ids "${event_id}" \
        --wet-threshold 0.05 \
        --stable-storage-threshold 0.01 --stable-extent-threshold 0.01 \
        --rapid-storage-threshold 0.05 --rapid-extent-threshold 0.05 \
        --netcdf-chunk-cache-mb 512 --overwrite &
    pids+=("$!")
    labels+=("${event_id}")
  done
  batch_status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      batch_status=1
    fi
  done
  if (( batch_status != 0 )); then
    echo "At least one paired-index worker failed in batch ${batch_start}-${batch_stop}" >&2
    exit 1
  fi
done

completed=$(find "${output_root}/shards" -mindepth 2 -maxdepth 2 \
  -name sampling_candidates.parquet -size +0c | wc -l)
if [[ "${completed}" -ne 40 ]]; then
  echo "Expected 40 completed paired-index shards, found ${completed}" >&2
  exit 1
fi
echo "Completed all ${completed} exact paired-transition shards" >&2
