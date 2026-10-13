#!/usr/bin/env bash
# Fault-aware training WITH an area-preserving regularizer (candidate states near the boundary of the
# initial certified set must stay inside the new one). Re-runs the two cases where plain fault-aware
# training shrank the certified area: hidden width 4 (48.0 -> 0.4) and seed 1 (26.1 -> 15.3).
set -u
cd "$(dirname "$0")/.."
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
SP="--masses 0.15 --setpoints 0.0 0.5 1.0"
rho_of() { python -c "import json,sys,math; print(math.floor(json.load(open(sys.argv[1]+'/summary.json'))[-1]['rho']*1e4)/1e4)" "$1"; }
log() { echo "[$(date '+%F %T')] $*" | tee -a results/logs/queue_progress.txt; }
until grep -q "S2 FIX DONE" results/logs/queue_progress.txt 2>/dev/null; do sleep 60; done
log "queue3 (area-preserving fault-aware training) starting"

run_case() {  # name init_run init_rho extra_args
  local NAME=$1 INIT=$2 R0=$3; shift 3
  local F=runs/${NAME}_fa_roa G=runs/${NAME}_roa_kappa
  python src/train_multimode_yang.py --out $F $SP --kappa 0.005 --fault_aware --keep_masks "$@" \
    --init_ckpt $INIT/ckpt_scale1.0.pth --limit_scales 1.0 --max_iter 30 --samples_per_iter 8192 --buffer_size 65536 \
    --roa_preserve_rho $R0 --roa_weight 1.0 > results/logs/queue3_${NAME}_fa.txt 2>&1
  local R=$(rho_of $F); log "$NAME fa+roa trained, rho=$R"
  python src/finetune_grid.py --run $F --rho $R --kappa 0.01 --margin 0 --out $G --rounds 40 > results/logs/queue3_${NAME}_grid.txt 2>&1
  cp $F/summary.json $G/ 2>/dev/null
  log "$NAME grid: $(grep -E '^\{"round"' results/logs/queue3_${NAME}_grid.txt | tail -1)"
  python src/fault_screen.py --run $G --rho $R --kappa 0.005 --no_crown > results/logs/queue3_${NAME}_screen.txt 2>&1
  log "$NAME screened: $(grep -E '^mode' results/logs/queue3_${NAME}_screen.txt | tr '\n' ' ')"
  log "$NAME area: $(CUDA_VISIBLE_DEVICES= python src/roa_area.py $G:$R 2>/dev/null | grep '^{')"
}

run_case h4 runs/cap_h4_shared 0.1276 --hidden 4
run_case s1 runs/s1_nominal 0.1311 --delta_scores --seed 1
log "QUEUE3 DONE"
