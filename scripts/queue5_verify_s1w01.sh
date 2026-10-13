#!/usr/bin/env bash
# Formal fault-bank verification of seed 1, w = 0.1 (PGD: 0/531, area 47.6), after the w sweep finishes.
set -u
cd "$(dirname "$0")/.."
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
log() { echo "[$(date '+%F %T')] $*" | tee -a results/logs/queue_progress.txt; }
until grep -q "QUEUE4 DONE" results/logs/queue_progress.txt 2>/dev/null; do sleep 60; done
log "queue5: formal verification of s1_w0.1_kappa starting"
python src/verify_faults.py --run runs/s1_w0.1_kappa --rho 0.1807 --kappa 0.005 --chunk 15 --timeout 1800 > results/logs/verify_faults_s1w01.txt 2>&1
log "queue5 result: $(grep 'ALL VERIFIED' results/logs/verify_faults_s1w01.txt) ; non-safe chunks: $(grep '^{"mode"' results/logs/verify_faults_s1w01.txt | grep -vc '"safe": 4, "falsified": 0, "timeout": 0')"
log "QUEUE5 DONE"
