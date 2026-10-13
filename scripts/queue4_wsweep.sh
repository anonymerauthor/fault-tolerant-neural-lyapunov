#!/usr/bin/env bash
# Sweep of the area-preserving weight w for seed 1 (width 8) and width 4.
# w = 0 (plain fault-aware) and w = 1 were run before; this adds w = 0.1 and 0.3.
set -u
cd "$(dirname "$0")/.."
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
SP="--masses 0.15 --setpoints 0.0 0.5 1.0"
rho_of() { python -c "import json,sys,math; print(math.floor(json.load(open(sys.argv[1]+'/summary.json'))[-1]['rho']*1e4)/1e4)" "$1"; }
log() { echo "[$(date '+%F %T')] $*" | tee -a results/logs/queue_progress.txt; }
log "queue4 (w sweep) starting"

run_case() {  # name init_run init_rho w extra_args
  local NAME=$1 INIT=$2 R0=$3 W=$4; shift 4
  local TAG=${NAME}_w${W}
  local F=runs/${TAG}_fa G=runs/${TAG}_kappa
  python src/train_multimode_yang.py --out $F $SP --kappa 0.005 --fault_aware --keep_masks "$@" \
    --init_ckpt $INIT/ckpt_scale1.0.pth --limit_scales 1.0 --max_iter 30 --samples_per_iter 8192 --buffer_size 65536 \
    --roa_preserve_rho $R0 --roa_weight $W > results/logs/queue4_${TAG}_fa.txt 2>&1
  local R=$(rho_of $F); log "$TAG trained, rho=$R"
  python src/finetune_grid.py --run $F --rho $R --kappa 0.01 --margin 0 --out $G --rounds 20 > results/logs/queue4_${TAG}_grid.txt 2>&1
  cp $F/summary.json $G/ 2>/dev/null
  log "$TAG grid: $(grep -E '^\{"round"' results/logs/queue4_${TAG}_grid.txt | tail -1)"
  python src/fault_screen.py --run $G --rho $R --kappa 0.005 --no_crown > results/logs/queue4_${TAG}_screen.txt 2>&1
  log "$TAG screened: $(grep -E '^mode' results/logs/queue4_${TAG}_screen.txt | tr '\n' ' ')"
  log "$TAG area: $(CUDA_VISIBLE_DEVICES= python src/roa_area.py $G:$R 2>/dev/null | grep '^{')"
}

for W in 0.1 0.3; do
  run_case s1 runs/s1_nominal 0.1311 $W --delta_scores --seed 1
  run_case h4 runs/cap_h4_shared 0.1276 $W --hidden 4
done
log "QUEUE4 DONE"
