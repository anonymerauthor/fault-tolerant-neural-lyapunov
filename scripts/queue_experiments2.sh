#!/usr/bin/env bash
# Experiment queue run after the F2c formal verification:
#   B1  random fault injection baseline (k = 8 faults per step)
#   S   seeds 1 and 2 of the full pipeline (nominal -> fault-aware -> grid + kappa slack)
#   C   capacity: hidden width 4 (nominal screening, fault-aware, grid + kappa slack)
# Every model is screened with PGD (fault_screen.py); results go to results/logs/queue_*.txt
set -u
cd "$(dirname "$0")/.."
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
SP="--masses 0.15 --setpoints 0.0 0.5 1.0"
# rho is truncated DOWN to 4 decimals: rounding up would let {V <= rho} reach the box boundary
rho_of() { python -c "import json,sys,math; print(math.floor(json.load(open(sys.argv[1]+'/summary.json'))[-1]['rho']*1e4)/1e4)" "$1"; }
log() { echo "[$(date '+%F %T')] $*" | tee -a results/logs/queue_progress.txt; }

log "queue v2 (rho truncated down) starting"

# ---- B1: random fault injection (already trained in v1; re-screen with truncated rho) ----
R=$(rho_of runs/b1_inject); log "B1 rho=$R"
python src/fault_screen.py --run runs/b1_inject --rho $R --kappa 0.005 > results/logs/queue_b1_screen.txt 2>&1
log "B1 screened: $(grep -E '^mode' results/logs/queue_b1_screen.txt | tr '\n' ' ')"

# ---- S: seeds ----
for SEED in 1 2; do
  N=runs/s${SEED}_nominal; F=runs/s${SEED}_faultaware; G=runs/s${SEED}_kappa
  if [ ! -f $N/summary.json ]; then  # seed 1 nominal was already trained in v1
    python src/train_multimode_yang.py --out $N $SP --kappa 0.005 --delta_scores --seed $SEED \
      --init_ckpt runs/pend2_yang/ckpt_scale1.0.pth --limit_scales 0.3 0.6 1.0 --max_iter 20 > results/logs/queue_s${SEED}_nom.txt 2>&1
  fi
  R=$(rho_of $N); log "seed $SEED nominal trained, rho=$R"
  python src/fault_screen.py --run $N --rho $R --kappa 0.005 --no_crown > results/logs/queue_s${SEED}_nom_screen.txt 2>&1
  log "seed $SEED nominal screened: $(grep -E '^mode' results/logs/queue_s${SEED}_nom_screen.txt | tr '\n' ' ')"
  python src/train_multimode_yang.py --out $F $SP --kappa 0.005 --fault_aware --delta_scores --keep_masks --seed $SEED \
    --init_ckpt $N/ckpt_scale1.0.pth --limit_scales 1.0 --max_iter 30 --samples_per_iter 8192 --buffer_size 65536 \
    > results/logs/queue_s${SEED}_fa.txt 2>&1
  R=$(rho_of $F); log "seed $SEED fault-aware trained, rho=$R"
  python src/finetune_grid.py --run $F --rho $R --kappa 0.01 --margin 0 --out $G --rounds 40 > results/logs/queue_s${SEED}_grid.txt 2>&1
  log "seed $SEED grid: $(grep -E '^\{"round"' results/logs/queue_s${SEED}_grid.txt | tail -1)"
  python src/fault_screen.py --run $G --rho $R --kappa 0.005 --no_crown > results/logs/queue_s${SEED}_screen.txt 2>&1
  log "seed $SEED fault-aware screened: $(grep -E '^mode' results/logs/queue_s${SEED}_screen.txt | tr '\n' ' ')"
done

# ---- C: capacity, hidden width 4 ----
R=$(rho_of runs/cap_h4_shared)
python src/fault_screen.py --run runs/cap_h4_shared --rho $R --kappa 0.005 --no_crown > results/logs/queue_h4_nom_screen.txt 2>&1
log "h4 nominal screened (rho=$R): $(grep -E '^mode' results/logs/queue_h4_nom_screen.txt | tr '\n' ' ')"
python src/train_multimode_yang.py --out runs/h4_faultaware $SP --hidden 4 --kappa 0.005 --fault_aware --keep_masks \
  --init_ckpt runs/cap_h4_shared/ckpt_scale1.0.pth --limit_scales 1.0 --max_iter 30 --samples_per_iter 8192 \
  --buffer_size 65536 > results/logs/queue_h4_fa.txt 2>&1
R=$(rho_of runs/h4_faultaware); log "h4 fault-aware trained, rho=$R"
python src/finetune_grid.py --run runs/h4_faultaware --rho $R --kappa 0.01 --margin 0 --out runs/h4_kappa --rounds 40 \
  > results/logs/queue_h4_grid.txt 2>&1
log "h4 grid: $(grep -E '^\{"round"' results/logs/queue_h4_grid.txt | tail -1)"
python src/fault_screen.py --run runs/h4_kappa --rho $R --kappa 0.005 --no_crown > results/logs/queue_h4_screen.txt 2>&1
log "h4 fault-aware screened: $(grep -E '^mode' results/logs/queue_h4_screen.txt | tr '\n' ' ')"
log "QUEUE DONE"
