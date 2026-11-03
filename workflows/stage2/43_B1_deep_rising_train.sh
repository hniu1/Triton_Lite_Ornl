#!/bin/bash
#SBATCH -A cli138
#SBATCH -J s2_fixed5h_deep_B1
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=375104M
#SBATCH -t 08:00:00
#SBATCH -o ./slurm_output/stage2-fixed5h-local-%j.out
#SBATCH -e ./slurm_output/stage2-fixed5h-local-%j.err

set -euo pipefail
cd /lustre/orion/cli138/proj-shared/7hn/triton/Triton_Lite_Ornl
source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /ccs/home/haoranniu/miniconda3/envs/triton
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
export MPLCONFIGDIR="${TMPDIR:-/tmp}/triton-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/triton-cache-${SLURM_JOB_ID}"
mode=${1:-smoke}
root=results/stage2_fixed5h_deep_rising/20261022_fold0
data_root=processed_data/stage2_corrected_training/20261009_fold0
case "$mode" in
  smoke) run_dir=${root}/smoke_job_${SLURM_JOB_ID}; epochs=1; batches=20; selection=8 ;;
  train)
    smoke=${2:?Specify successful smoke directory}
    python -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="passed" and r["experiment"]=="B1_fixed5h_deep_rising"' "${smoke}/SMOKE_PASSED.json"
    run_dir=${root}/seed_45; epochs=10; batches=750; selection=80 ;;
  *) exit 2 ;;
esac
test ! -e "$run_dir"
mkdir -p "$root"
python -c 'import json; assert json.load(open("processed_data/stage2_fixed5h_local/20261022_fold0/forcing/metadata.json"))["status"]=="passed"'
python -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="passed"' "${data_root}/TRAINING_PRECHECK.json"
python -u -m stage2.train_direct \
  --initial-run-dir "${data_root}/source_config" --output-dir "$run_dir" \
  --test-events D005 D014 D018 D023 D024 D028 D033 D037 \
  --data-split-seed 4300 --seed 45 --no-v4b-initialization \
  --hydraulic-halo-cells 8 --graph-layers 0 --graph-hops 0 \
  --forcing-weights-path processed_data/stage2_fixed5h_local/20261022_fold0/forcing/tile_forcing_weights.npy \
  --local-forcing-embedding-dim 64 --max-horizon-steps 10 --train-horizons 10 \
  --depth-extent-only --step-hours 0.5 \
  --deep-rising-loss-weight 1 --deep-rising-depth-threshold 1 --deep-rising-change-threshold .1 \
  --epochs "$epochs" --batch-size 8 --train-batches-per-epoch "$batches" \
  --selection-batches "$selection" --selection-horizons 10 --selection-start-time-stride 12 \
  --backbone-learning-rate .0001 --new-conditioning-learning-rate .0001 \
  --selection-derived-velocity-weight 0 --netcdf-chunk-cache-mb 32 \
  --max-open-netcdf-handles 4 --num-workers 2 --device cuda --disable-cudnn \
  --save-every-epoch --skip-test-evaluation
if [[ "$mode" == smoke ]]; then
python - "$run_dir" <<'PY'
import json,math,sys,torch
from pathlib import Path
p=Path(sys.argv[1]);c=json.loads((p/'run_config.json').read_text());m=json.loads((p/'metrics.json').read_text())
assert c['trained_horizons_steps']==c['selection_horizons_steps']==[10]
assert c['loss_config']['deep_rising_loss_weight']==1
assert c['depth_extent_only'] and c['trained_outputs']==['depth','wet_extent']
assert c['initial_checkpoint'] is None and m['test_batches']==0
assert c['model_config']['local_forcing_embedding_dim']==64
for key in ['component_loss_weight','dry_component_loss_weight','component_delta_loss_weight','derived_velocity_loss_weight']:
    assert c['loss_config'][key]==0
record=m['history'][0]
assert record['train']['loss_deep_rising']>0
assert record['train']['horizon_sample_counts']=={'10':160}
assert record['train']['local_forcing_gradient_norm_max']>0
assert all(math.isfinite(v) for v in record['train'].values() if isinstance(v,(int,float)))
assert 'derived_velocity_rmse' not in record['validation_horizons']['10']['direct']
s=torch.load(p/'best_model.pt',map_location='cpu',weights_only=False)['model_state_dict']
assert all(torch.isfinite(v).all() for v in s.values())
assert s['component_delta_head.weight'].count_nonzero()==0 and s['component_delta_head.bias'].count_nonzero()==0
report=dict(status='passed',experiment='B1_fixed5h_deep_rising',test_batches=0,
            fixed_lead_hours=5,local_gradient_max=record['train']['local_forcing_gradient_norm_max'],epoch_seconds=record['epoch_seconds'])
(p/'SMOKE_PASSED.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
PY
fi
