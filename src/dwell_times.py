"""Guaranteed set-point switching times (Section "Faults at arbitrary times", funnels).

For set-points i -> k, eps_ik is the largest level such that
    {x : V(x - x*_i) <= eps}  is contained in  F_k = {x : x - x*_k in B, V(x - x*_k) <= rho}.
It is computed soundly on a grid: eps_ik = min over grid nodes x NOT in F_k (with a Lipschitz margin)
of V(x - x*_i) - L_V h / sqrt(2), where h is the grid spacing and L_V an upper bound of the
Lipschitz constant of V (spectral-norm product of the network part + sqrt(2)||eps I + R^T R||_2).
Then T_ik = ceil(log(eps_ik / rho) / log(1 - kappa)) steps; by the single-fault certificate this
holds for every single sign fault of mode i. We also report the observed worst number of steps
over trajectories that start on the boundary of F_i under all single faults.

Usage: python src/dwell_times.py --run runs/f2c_kappa --rho 0.165 --kappa 0.005
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import math
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "Lyapunov_Stable_NN_Controllers"))
sys.modules.setdefault("wandb", types.ModuleType("wandb"))
import fault_screen as FS  # noqa: E402
from fault_aware import FaultAwareLoss  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lipschitz_V(V):
    L = 1.0
    for mod in V.net if hasattr(V, "net") else []:
        if isinstance(mod, torch.nn.Linear):
            L *= torch.linalg.matrix_norm(mod.weight.detach(), 2).item()
    A = V.eps * torch.eye(V.x_dim) + V.R.detach().T @ V.R.detach()
    return L + math.sqrt(V.x_dim) * torch.linalg.matrix_norm(A, 2).item()


def own_controls(loss, ctrl, m, zz):
    """u of variant f (single-fault stack of mode m) at its own states zz[f]: (F, P, 1)."""
    sw, sb = loss.stacks[m]
    x_eq, u_eq = ctrl.eq(m)
    x_eq, u_eq = x_eq.to(zz.device), u_eq.to(zz.device)

    def phi(h):
        for l in range(ctrl.phi.n_layers):
            h = torch.einsum("foi,fbi->fbo", sw[l] * ctrl.phi.weights[l], h) + (sb[l] * ctrl.phi.biases[l]).unsqueeze(1)
            if l < ctrl.phi.n_layers - 1:
                h = ctrl.activation()(h)
        return h

    F = zz.shape[0]
    raw = phi(zz + x_eq) - phi(x_eq.view(1, 1, -1).expand(F, 1, -1)) + u_eq
    f1 = torch.relu(raw - ctrl.u_lo) + ctrl.u_lo
    return -(torch.relu(ctrl.u_up - f1) - ctrl.u_up)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--kappa", type=float, default=0.005)
    ap.add_argument("--h", type=float, default=0.004)
    ap.add_argument("--sim_steps", type=int, default=3000)
    ap.add_argument("--out", default="results/dwell_times.json")
    args = ap.parse_args()
    a, ck, systems, ctrl, V, lim, ec = FS.load_run(ROOT / args.run)
    V = V.to(DEV); ctrl = ctrl.to(DEV)
    for s_ in systems:
        s_.to(DEV)
    LV = lipschitz_V(V.cpu()); V = V.to(DEV)
    L = len(systems)
    xeq = [ctrl.eq(m)[0].to(DEV) for m in range(L)]
    lim = lim.to(DEV)
    # grid over the union of the boxes x*_m + B
    lo = torch.stack([x - lim for x in xeq]).min(0).values
    hi = torch.stack([x + lim for x in xeq]).max(0).values
    t = [torch.arange(float(lo[d]), float(hi[d]) + args.h, args.h, device=DEV) for d in range(2)]
    marg = LV * args.h / math.sqrt(2)
    res = {"run": args.run, "rho": args.rho, "kappa": args.kappa, "L_V": LV, "h": args.h, "margin": marg, "pairs": []}
    eps = [[None] * L for _ in range(L)]
    for i in range(L):
        for k in range(L):
            if i == k:
                continue
            best = float("inf")
            for x0 in t[0].split(256):
                X = torch.stack(torch.meshgrid(x0, t[1], indexing="ij"), -1).reshape(-1, 2)
                zk = X - xeq[k]
                # nodes whose whole cell may leave F_k: V(zk) + marg > rho or the cell touches the box boundary
                out = (V(zk).squeeze(-1) + marg > args.rho) | ((zk.abs() + args.h) > lim).any(1)
                if out.any():
                    best = min(best, (V(X[out] - xeq[i]).squeeze(-1) - marg).min().item())
            eps[i][k] = best
            T = math.ceil(math.log(best / args.rho) / math.log(1 - args.kappa)) if 0 < best < args.rho else (0 if best >= args.rho else None)
            res["pairs"].append({"i": i, "k": k, "eps": best, "T_steps": T,
                                 "T_sec": None if T is None else T * a["dt"]})
    # observed: worst steps to reach {V(x - x*_i) <= eps_ik} from the boundary of F_i, over all single faults
    loss = FaultAwareLoss(systems, ctrl.to(DEV), V, -lim, lim, 1.0, args.kappa, ec, include_faults=True)
    th = torch.linspace(0, 2 * math.pi, 721, device=DEV)[:-1]
    for p in res["pairs"]:
        if p["T_steps"] is None:
            continue
        i = p["i"]
        # boundary points of F_i (in error coordinates): bisection along rays
        d = torch.stack([th.cos(), th.sin()], 1)
        r_lo, r_hi = torch.zeros(len(th), device=DEV), torch.full((len(th),), float(lim.min()), device=DEV)
        for _ in range(40):
            mid = (r_lo + r_hi) / 2
            inside = V(d * mid.unsqueeze(1)).squeeze(-1) <= args.rho
            r_lo, r_hi = torch.where(inside, mid, r_lo), torch.where(inside, r_hi, mid)
        z = d * r_lo.unsqueeze(1)                                    # (P, 2)
        dyn = systems[i]
        steps_needed = torch.full((loss.stacks[i][0][0].shape[0], z.shape[0]), -1, device=DEV)
        zz = z.unsqueeze(0).expand(steps_needed.shape[0], -1, -1).clone()   # (F, P, 2)
        for s in range(args.sim_steps):
            Vz = V(zz.reshape(-1, 2)).view(zz.shape[0], -1)
            hit = (Vz <= p["eps"]) & (steps_needed < 0)
            steps_needed[hit] = s
            if (steps_needed >= 0).all():
                break
            F, P = zz.shape[0], zz.shape[1]
            uu = own_controls(loss, ctrl, i, zz)                     # (F, P, 1)
            zz = dyn(zz.reshape(-1, 2), uu.reshape(-1, 1)).view(F, P, 2)
        p["observed_worst_steps"] = int(steps_needed.max().item())
        p["observed_all_reached"] = bool((steps_needed >= 0).all().item())
    Path(ROOT / args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
