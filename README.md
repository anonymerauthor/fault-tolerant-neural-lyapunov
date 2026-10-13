# Fault-Tolerant Neural Lyapunov Certificates

Code, trained models, and result summaries for the paper
**"Certified Closed-Loop Stability of Sign-Masked Neural Controllers Under Sign-Bit Faults"** (under review).

Formal stability certificates for neural network controllers are usually proved for one fixed set of
weights. This repository extends such a certificate to **every single sign-bit fault** of the controller
weights and verifies it with α,β-CROWN:

- **Sign-masked controllers.** The controller is a shared weight vector multiplied by a sign mask; a sign-bit
  fault is a flipped mask entry. Normalizing the network output at the equilibrium keeps the equilibrium fixed
  under every fault, so the nominal Lyapunov function and sublevel set remain valid targets.
- **Fault-aware training.** The worst Lyapunov-condition violation over all operating modes and all single
  sign faults is minimized in one batched pass, inside the training loop of Yang et al. (ICML 2024).
- **Grid falsification on the unclipped margin**, which finds small violating sets near the equilibrium that
  gradient-based (PGD) falsification misses.
- **Fault-bank verification.** A chunk of faulty controllers is stacked into one block-diagonal network and
  verified with a single α,β-CROWN problem.

## Main results

| System | Controllers | Result |
|---|---|---|
| Inverted pendulum, 3 set-points, 2-8-8-8-1 | 3 × 177 single faults | nominal: 15–21 % of faults break the certificate; after fault-aware training all 531 faulty controllers formally verified (two seeds) |
| Planar quadrotor (6 states), 6-8-2 | 74 single faults | nominal fine-tuned: 24/74 broken; fault-aware: 5/74 broken, the other 69 formally verified |

Additional analyses: all 46 728 double faults (grid falsification), certified dwell times for set-point
switches under faults, CROWN bounds on the excluded box around the origin, and local exponential stability of
all faulty linearizations.

## Repository layout

```
src/                  training, falsification, verification, analysis, figures
  masked_mlp.py, masked_controller.py   sign-masked network and multi-mode controller
  setpoint.py                           set-point (error-coordinate) dynamics for the pendulum
  train_multimode_yang.py               nominal and fault-aware training (pendulum)
  fault_aware.py                        batched worst-case loss over modes x faults
  fault_screen.py                       PGD screening of all single faults, CROWN jump bounds
  finetune_grid.py                      grid falsifier + fine-tuning with fixed V and rho
  verify_faults.py, verify_faults_quad.py  fault-bank construction and alpha-beta-CROWN runs
  quad.py, train_quad.py                planar quadrotor (Yang et al. model) screening and training
  double_faults.py                      grid falsification of all double faults
  dwell_times.py                        certified set-point switching times
  hole_check.py, local_stability.py     bounds on the excluded box H, local (linearized) stability
  make_figures.py, make_quad_figure.py, make_tradeoff.py   figures
tests/                equivalence tests of the fault bank and the bounds
scripts/              experiment queues used for the pendulum runs
scripts/gpu_server/   setup and quadrotor verification on a Linux GPU server
runs/                 trained models (checkpoints, arguments, training logs, PGD screening tables)
results/              summaries: double faults, dwell times, local stability, quadrotor screening/verification
```

Run names in `runs/`:

| Run | Description |
|---|---|
| `pend2_yang` | warm-start controller (two masses) used to initialize the set-point models |
| `e2b_masked`, `s1_nominal`, `s2_nominal` | nominal three-set-point controllers, seeds 0, 1, 2 |
| `f2_faultaware`, `f2b_grid`, `f2c_kappa` | seed 0: fault-aware training, grid fine-tuning, decay-rate slack (verified) |
| `b1_inject` | random fault-injection baseline (8 faults per step) |
| `s1_*`, `s2_*` | seeds 1 and 2; `s1_w0.1_*`, `s1_w0.3_*`, `s1_fa_roa`/`s1_roa_kappa` region-preserving weight 0.1, 0.3, 1 (`s1_w0.1_kappa` verified) |
| `cap_h4_shared`, `h4_*` | narrow network (4 hidden units per layer) |
| `e1*`, `e2a_shared`, `cap_h2_*`, `grow_h2_*` | shared controller vs. sign masks (capacity comparison) |
| `quad_nominal`, `quad_fa`, `quad_fa2` | quadrotor: nominal fine-tuning, fault-aware stage 1 and 2 (`quad_fa2` verified) |

## Setup

Tested with Python 3.11/3.12, PyTorch 2.6–2.8 with CUDA, NumPy 1.26. The two external code bases are used
unmodified at fixed commits:

| Repository | Commit |
|---|---|
| [Verified-Intelligence/Lyapunov_Stable_NN_Controllers](https://github.com/Verified-Intelligence/Lyapunov_Stable_NN_Controllers) | `f6ac480` |
| [Verified-Intelligence/alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) | `4ff0615` (auto_LiRPA 0.6.0) |

On Linux:

```bash
bash scripts/gpu_server/setup.sh     # clones both repositories into external/, creates .venv, runs a test
```

The script creates a virtual environment that reuses the system PyTorch (`--system-site-packages`) and
installs the pinned packages of `requirements.txt` plus `auto_LiRPA` in editable mode. The newest
alpha-beta-CROWN requires NumPy 2 and a newer PyTorch and is not compatible with these commits.

Run the verifier with the external repositories on the path:

```bash
export PYTHONPATH="$(pwd)/external/Lyapunov_Stable_NN_Controllers:$(pwd)/external/alpha-beta-CROWN/complete_verifier"
```

Do not add `external/alpha-beta-CROWN` itself to `PYTHONPATH` (its `auto_LiRPA/` folder shadows the package).

## Reproducing the experiments

Pendulum (one 8 GB GPU is sufficient). Exact arguments of every run are stored in `runs/<run>/args.json`.

```bash
# nominal three-set-point controller (seed 0)
python src/train_multimode_yang.py --out runs/e2b_masked --masses 0.15 --setpoints 0.0 0.5 1.0 \
    --kappa 0.005 --delta_scores --init_ckpt runs/pend2_yang/ckpt_scale1.0.pth --limit_scales 0.3 0.6 1.0 --max_iter 20
# PGD screening of all 531 single faults (and CROWN jump bounds)
python src/fault_screen.py --run runs/e2b_masked --rho 0.136 --kappa 0.005
# fault-aware training, grid fine-tuning with decay-rate slack, fault-bank verification
python src/train_multimode_yang.py --out runs/f2_faultaware --masses 0.15 --setpoints 0.0 0.5 1.0 \
    --kappa 0.005 --fault_aware --keep_masks --init_ckpt runs/e2b_masked/ckpt_scale1.0.pth
python src/finetune_grid.py --run runs/f2_faultaware --rho 0.165 --kappa 0.005 --out runs/f2b_grid
python src/finetune_grid.py --run runs/f2b_grid --rho 0.165 --kappa 0.01 --margin 0 --out runs/f2c_kappa
python src/verify_faults.py --run runs/f2c_kappa --rho 0.165 --kappa 0.005 --chunk 15
# analyses
python src/double_faults.py --run runs/f2c_kappa --rho 0.165 --out results/double/pairs_f2c_kappa.json
python src/dwell_times.py --run runs/f2c_kappa --rho 0.165 --kappa 0.005
python src/hole_check.py
python src/local_stability.py
```

The seed, capacity, and region-preserving experiments are in `scripts/queue_*.sh`.

Quadrotor (the per-fault verification took about 29 GPU-hours on a 48 GB GPU):

```bash
python src/train_quad.py --out runs/quad_fa --rho 0.5
python src/train_quad.py --out runs/quad_fa2 --init runs/quad_fa/ckpt.pth --rho 0.5 \
    --max_iter 20 --samples_per_iter 32768 --buffer_size 131072 --pgd_steps 300
python src/quad.py --ckpt runs/quad_fa2/ckpt.pth --rho 0.5 --kappa 0 --out results/quadscreen_quad_fa2_rho0.5.csv
bash scripts/gpu_server/verify_quad_faults.sh
```

## Citation

The citation will be added after the review process.

## License

MIT (see `LICENSE`). The external repositories keep their own licenses.
