#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_corrected_baseline
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=375104M
#SBATCH -t 24:00:00
#SBATCH -o ./slurm_output/stage2-corrected-baseline-%j.out
#SBATCH -e ./slurm_output/stage2-corrected-baseline-%j.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /ccs/home/haoranniu/miniconda3/envs/triton
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-mpl-${SLURM_JOB_ID}"
data_root=processed_data/stage2_corrected_training/20261009_fold0
mode="${1:-train}"
python -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "passed"' \
  "${data_root}/TRAINING_PRECHECK.json"
python -u -c 'import torch; assert torch.cuda.is_available(); print("GPU:",torch.cuda.get_device_name(0)); x=torch.randn(64,64,device="cuda",requires_grad=True); x.square().mean().backward(); torch.cuda.synchronize(); print("GPU forward/backward passed")'
case "${mode}" in
  smoke)
    run_dir="results/stage2_corrected_baseline/20261009_fold0/smoke_job_${SLURM_JOB_ID}"
    epochs=1; batches=20; selection_batches=8 ;;
  train)
    run_dir=results/stage2_corrected_baseline/20261009_fold0/seed_45
    epochs=10; batches=750; selection_batches=80 ;;
  *) echo "Unknown mode: ${mode}" >&2; exit 2 ;;
esac
test ! -e "${run_dir}"
python -u -m stage2.train_direct \
  --initial-run-dir "${data_root}/source_config" --output-dir "${run_dir}" \
  --test-events D005 D014 D018 D023 D024 D028 D033 D037 \
  --data-split-seed 4300 --seed 45 --no-v4b-initialization \
  --hydraulic-halo-cells 0 --graph-layers 0 --max-horizon-steps 48 --step-hours 0.5 \
  --epochs "${epochs}" --batch-size 8 --train-batches-per-epoch "${batches}" \
  --selection-batches "${selection_batches}" --selection-horizons 1 6 12 24 48 \
  --selection-start-time-stride 12 \
  --backbone-learning-rate 0.0001 --new-conditioning-learning-rate 0.0001 \
  --selection-derived-velocity-weight 1.0 --netcdf-chunk-cache-mb 32 \
  --max-open-netcdf-handles 4 --num-workers 2 --device cuda --disable-cudnn \
  --save-every-epoch --skip-test-evaluation
if [[ "${mode}" == smoke ]]; then
  python - "${run_dir}" <<'PY'
import json, math, sys
from pathlib import Path
p=Path(sys.argv[1]); c=json.loads((p/'run_config.json').read_text()); m=json.loads((p/'metrics.json').read_text())
assert c['initial_checkpoint'] is None and c['initialization']['loaded_tensors']==[]
assert m['test_batches']==0 and math.isfinite(m['best_selection_score'])
assert all(math.isfinite(v) for v in m['history'][0]['train'].values() if isinstance(v,(int,float)))
(p/'SMOKE_PASSED.json').write_text(json.dumps(dict(status='passed',test_batches=0,legacy_weights_used=False),indent=2)+'\n')
print('Corrected training smoke passed; no held-out test batches evaluated.')
PY
fi
