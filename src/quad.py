"""2-D quadrotor (6 states, 2 thrusts) of Yang et al. (2024) as a single-mode sign-masked controller.

load_quad(ckpt) returns the plant, a MaskedController (one mode, masks +1) holding the controller
weights, the quadratic Lyapunov function, the verification box, and the verified level rho.
`python src/quad.py --screen` runs the single-sign-fault PGD screening (cf. fault_screen.py).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
import math
import sys
import time
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LYA = ROOT / "external" / "Lyapunov_Stable_NN_Controllers"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(LYA))
sys.modules.setdefault("wandb", types.ModuleType("wandb"))

import neural_lyapunov_training.dynamical_system as dynamical_system  # noqa: E402
import neural_lyapunov_training.lyapunov as lyapunov  # noqa: E402
import neural_lyapunov_training.quadrotor2d as quadrotor2d  # noqa: E402
import neural_lyapunov_training.train_utils as train_utils  # noqa: E402
from masked_controller import MaskedController  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOX = [0.75, 0.75, math.pi / 2, 4.0, 4.0, 3.0]
U_MAX = 5.9596
YANG_CKPT = LYA / "models" / "quadrotor2d_state_feedback.pth"
YANG_RHO = 1.3392


def load_quad(ckpt=YANG_CKPT):
    cont = quadrotor2d.Quadrotor2DDynamics()
    dyn = dynamical_system.SecondOrderDiscreteTimeSystem(cont, dt=0.01)
    u_eq = [cont.mass * cont.gravity / 2] * 2
    ctrl = MaskedController(6, 2, 8, 2, 1, u_lo=[0.0, 0.0], u_up=[U_MAX, U_MAX], x_eq=[0.0] * 6, u_eq=u_eq,
                            score_scale=10.0)
    V = lyapunov.NeuralNetworkQuadraticLyapunov(goal_state=torch.zeros(6), x_dim=6, R_rows=6, eps=0.01)
    sd = torch.load(ckpt, map_location="cpu")
    if "ctrl" in sd:  # checkpoint written by our own training scripts
        ctrl.load_state_dict(sd["ctrl"])
        V.load_state_dict(sd["V"])
    else:  # Yang et al. layout
        sd = sd.get("state_dict", sd)
        with torch.no_grad():
            for l in range(ctrl.phi.n_layers):
                ctrl.phi.weights[l].copy_(sd[f"controller.net.{2 * l}.weight"])
                ctrl.phi.biases[l].copy_(sd[f"controller.net.{2 * l}.bias"])
            for s in list(ctrl.phi.w_scores) + list(ctrl.phi.b_scores):
                s.fill_(0.05 / ctrl.phi.score_scale)  # all masks +1
        V.load_state_dict({k[len("lyapunov."):]: v for k, v in sd.items() if k.startswith("lyapunov.")})
    lim = torch.tensor(BOX)
    return dyn, ctrl, V, lim


def screen(ckpt, rho, kappa, out_csv, n=32768, steps=300, seed=0, shard=0, nshards=1):
    """Variants are split round-robin over `nshards` processes (run them in parallel, then merge CSVs)."""
    import fault_screen as FS
    torch.manual_seed(seed + shard)
    dyn, ctrl, V, lim = load_quad(ckpt)
    dyn.to(DEV); ctrl.to(DEV).eval(); V.to(DEV).eval()
    lo, up = (-lim).to(DEV), lim.to(DEV)
    x_eq, u_eq = (t.to(DEV) for t in ctrl.eq(0))
    nominal = ctrl.hard_masks(0)
    rho_t = torch.tensor(rho, device=DEV)
    rows, t0 = [], time.time()
    variants = [None] + FS.all_bits(ctrl)
    for vi, bit in enumerate(variants):
        if vi % nshards != shard:
            continue
        mask = nominal if bit is None else FS.flip(nominal, bit)
        c = FS.FixedMaskController(ctrl, mask, x_eq, u_eq, False).to(DEV)
        lf = lyapunov.LyapunovDerivativeLoss(dyn, c, V, box_lo=lo, box_up=up, rho_multiplier=1.0, kappa=kappa,
                                             hard_max=True)
        lf.get_rho = lambda: rho_t
        x0 = lo + torch.rand(n, 6, device=DEV) * (up - lo)
        adv = train_utils.pgd_attack(x0, lf, eps=up, steps=steps, lower_boundary=lo, upper_boundary=up,
                                     direction="minimize").detach()
        with torch.no_grad():
            v = torch.relu(-lf(adv)).squeeze(-1)
        name = "nominal" if bit is None else f"{bit[0]}{bit[1]}[{','.join(map(str, bit[2]))}]"
        rows.append(dict(bit=name, pgd_max_viol=v.max().item(), pgd_n_viol=int((v > 0).sum())))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    faults = [r for r in rows if r["bit"] != "nominal"]
    nom = [r["pgd_max_viol"] for r in rows if r["bit"] == "nominal"]
    print(json.dumps(dict(shard=shard, nominal_viol=nom[0] if nom else None,
                          broken=sum(1 for r in faults if r["pgd_max_viol"] > 0), faults=len(faults),
                          seconds=round(time.time() - t0))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(YANG_CKPT))
    ap.add_argument("--rho", type=float, default=YANG_RHO)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--out", default="results/quad_faults_nominal.csv")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    out = args.out if args.nshards == 1 else args.out.replace(".csv", f"_shard{args.shard}.csv")
    screen(args.ckpt, args.rho, args.kappa, ROOT / out, shard=args.shard, nshards=args.nshards)
