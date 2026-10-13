#!/usr/bin/env bash
# One-time environment setup on a Linux GPU server with a preinstalled PyTorch.
#   bash scripts/gpu_server/setup.sh
# Pins the same code versions that were used locally:
#   Lyapunov_Stable_NN_Controllers  f6ac480 (Yang et al., ICML 2024)
#   alpha-beta-CROWN                4ff0615 (Jan 2025 release; auto_LiRPA submodule 0.6.0)
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)

echo "== GPU =="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

mkdir -p external && cd external
if [ ! -d Lyapunov_Stable_NN_Controllers ]; then
  git clone https://github.com/Verified-Intelligence/Lyapunov_Stable_NN_Controllers.git
  git -C Lyapunov_Stable_NN_Controllers checkout -q f6ac480
fi
if [ ! -d alpha-beta-CROWN ]; then
  git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
  git -C alpha-beta-CROWN checkout -q 4ff0615
  git -C alpha-beta-CROWN submodule update --init -q
fi
cd "$ROOT"

# The system Python is "externally managed" (PEP 668): use a venv that reuses the preinstalled torch.
if [ ! -d .venv ]; then python -m venv --system-site-packages .venv; fi
source .venv/bin/activate

# Python packages. numpy < 2 keeps the 2025 verifier and our code on the versions used locally.
pip install -q "numpy<2" "scipy<1.15" "pandas<2.3" "control==0.10.2" "hydra-core==1.3.2" \
  "onnx==1.17.0" "onnxruntime==1.20.1" onnxoptimizer skl2onnx onnxsim sortedcontainers gurobipy \
  matplotlib pyyaml tqdm appdirs graphviz packaging psutil termcolor sympy
pip install -q "git+https://github.com/Verified-Intelligence/onnx2pytorch@8afb2243fddd1a4732883f1e2959f7e83f8201c5"
pip install -q --no-deps -e external/alpha-beta-CROWN/auto_LiRPA

echo "== import check =="
python -c "import numpy, torch, auto_LiRPA, onnx2pytorch, onnxoptimizer, skl2onnx, gurobipy, control, hydra; print('imports ok, numpy', numpy.__version__)"

echo "== equivalence tests =="
python tests/test_quad_bank.py | tail -1
echo "setup done"
