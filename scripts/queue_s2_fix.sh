#!/usr/bin/env bash
# Seed 2: the trained level rho=0.1847 lies at min V on the box boundary, so {V<=rho} touches the boundary
# and next states leave the box there. Use a slightly smaller level (0.16) for fine-tuning and screening.
set -u
cd "$(dirname "$0")/.."
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
log() { echo "[$(date '+%F %T')] $*" | tee -a results/logs/queue_progress.txt; }
until grep -q "QUEUE DONE" results/logs/queue_progress.txt 2>/dev/null; do sleep 60; done
python src/finetune_grid.py --run runs/s2_faultaware --rho 0.16 --kappa 0.01 --margin 0 --out runs/s2_kappa --rounds 40 > results/logs/queue_s2_grid.txt 2>&1
log "seed 2 grid (rho=0.16): $(grep -E '^\{"round"' results/logs/queue_s2_grid.txt | tail -1)"
python src/fault_screen.py --run runs/s2_kappa --rho 0.16 --kappa 0.005 --no_crown > results/logs/queue_s2_screen.txt 2>&1
log "seed 2 fault-aware screened (rho=0.16): $(grep -E '^mode' results/logs/queue_s2_screen.txt | tr '\n' ' ')"
log "S2 FIX DONE"
