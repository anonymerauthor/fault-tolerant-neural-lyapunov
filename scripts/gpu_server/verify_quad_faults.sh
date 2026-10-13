#!/usr/bin/env bash
# Per-fault formal verification of the quadrotor fault-aware model (as run for the paper, Oct 2026).
# Settings that worked on an RTX A6000 (48 GB) with a 50 GB host-memory limit:
#   one controller per problem (--chunk 1), batch 150000, timeout 2400 s, two processes in parallel.
# Variants 59, 61-64 (W2[2,3], W2[2,5..8]) have PGD counterexamples and are skipped; variant 0 (nominal)
# is verified separately with --nominal_only. Total ~29 GPU-hours (~15 h wall clock).
# Usage (on the GPU server, after scripts/gpu_server/setup.sh):  bash scripts/gpu_server/verify_quad_faults.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
mkdir -p results/logs results/quad_verify
for P in 0 1; do
  L=$(python -c "
skip={0,59,61,62,63,64}; v=[i for i in range(75) if i not in skip]; print(' '.join(str(x) for x in v[$P::2]))")
  nohup python src/verify_faults_quad.py --ckpt runs/quad_fa2/ckpt.pth --rho 0.5 --kappa 0.0 --chunk 1 \
    --chunks $L --timeout 2400 --batch 150000 --out results/quad_verify/qfa2_single_p$P \
    > results/logs/qfa2_single_p$P.txt 2>&1 < /dev/null &
done
wait
grep -h "^{" results/logs/qfa2_single_p0.txt results/logs/qfa2_single_p1.txt | python -c "
import sys, json
r = [json.loads(l) for l in sys.stdin]
bad = [x['variants'][0] for x in r if x['safe'] != x['regions']]
print('done', len(r), 'verified', len(r) - len(bad), 'not verified:', bad)"
