#!/usr/bin/env bash
# Step 1 on the GPU server: time the NOMINAL formal verification of Yang et al.'s pretrained quadrotor
# (12 regions). This is the baseline cost; the fault bank costs a multiple of it.
#   bash scripts/gpu_server/verify_quad_nominal.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
R=$ROOT/external/Lyapunov_Stable_NN_Controllers
export CONFIG_PATH=$R/verification
export PYTHONPATH="$R:$ROOT/external/alpha-beta-CROWN/complete_verifier"
mkdir -p results/logs
cd external/alpha-beta-CROWN/complete_verifier
/usr/bin/time -v python abcrown.py --config $CONFIG_PATH/quadrotor2d_state_feedback_lyapunov_in_levelset.yaml \
  > $ROOT/results/logs/verify_quad_nominal_gpu.txt 2>&1
grep -E "Result:|Final verified|total verified|mean time|Elapsed|Maximum resident" $ROOT/results/logs/verify_quad_nominal_gpu.txt
