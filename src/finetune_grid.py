"""Fault-aware fine-tuning with a dense-grid falsifier (complements PGD).

PGD can miss tiny violation regions near the equilibrium where V is small (observed in F2:
mode 1, fault w0[4,1]). Loop:
  1. evaluate FaultAwareLoss (all modes x all single-bit faults) on a dense grid of the box,
  2. stop if no grid point inside {V <= rho} violates the condition (with a small safety margin),
  3. otherwise train on [violating points + their jittered neighbours + PGD/random buffer].
rho is held fixed via a frozen x_boundary (same certificate as the subsequent verification).

Usage: python src/finetune_grid.py --run runs/f2_faultaware --rho 0.165 --kappa 0.005 --out runs/f2b_grid
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import shutil
import sys
import time
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


@torch.no_grad()
def grid_violations(loss: FaultAwareLoss, lo, up, n: int, margin: float, rho: float, chunk: int = 4096,
                    fine: float = 1.0):
    """Multi-resolution grid: whole box (n x n) + fine grid on [-fine, fine]^2 (n x n).
    Returns points inside {V <= rho} with raw margin < margin, the worst raw margin, and #strict violations."""
    grids = []
    for a, b in ((lo, up), (torch.clamp(-fine * torch.ones_like(lo), lo, up), torch.clamp(fine * torch.ones_like(up), lo, up))):
        t0 = torch.linspace(float(a[0]), float(b[0]), n, device=DEV)
        t1 = torch.linspace(float(a[1]), float(b[1]), n, device=DEV)
        grids.append(torch.stack(torch.meshgrid(t0, t1, indexing="ij"), -1).reshape(-1, 2))
    Z = torch.cat(grids)
    bad, worst, n_strict = [], float("inf"), 0
    for c in Z.split(chunk):
        mg, Vz = loss.raw_margin(c)
        ins = (Vz <= rho) & (c.norm(dim=1) > 1e-3)          # exclude the verifier's tiny hole at 0
        if ins.any():
            worst = min(worst, mg[ins].min().item())
            v = ins & (mg < margin)
            n_strict += int((ins & (mg < 0)).sum())
            if v.any():
                bad.append(c[v])
    return (torch.cat(bad) if bad else torch.empty(0, 2, device=DEV)), worst, n_strict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--kappa", type=float, default=0.005)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=800)
    ap.add_argument("--margin", type=float, default=1e-4, help="required slack at grid points")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--train_V", action="store_true", help="also train V (default: V frozen, rho fixed)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    src = (ROOT / args.run).resolve()
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    targs = json.loads((src / "args.json").read_text())
    a, ck, systems, ctrl, V, lim, ec = FS.load_run(src)
    ctrl.to(DEV).train(); V.to(DEV).train()
    for s in systems:
        s.to(DEV)
    for p in list(ctrl.phi.w_scores) + list(ctrl.phi.b_scores):
        p.requires_grad_(False)  # masks frozen (fault positions are relative to them)
    lo, up = (-lim).to(DEV), lim.to(DEV)
    xb = ck["x_boundary"].to(DEV)
    loss = FaultAwareLoss(systems, ctrl, V, lo, up, rho_multiplier=1.0, kappa=args.kappa, error_coords=ec)
    loss.x_boundary = xb
    # rho is a FIXED constant (the verification spec uses V(z) <= rho directly)
    rho_t = torch.tensor(args.rho, device=DEV)
    loss.get_rho = lambda: rho_t
    if not args.train_V:
        for p in V.parameters():
            p.requires_grad_(False)
    params = [p for p in list(ctrl.parameters()) + list(V.parameters()) if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    log = []
    t_start = time.time()
    for r in range(args.rounds):
        bad, worst, n_strict = grid_violations(loss, lo, up, args.grid, args.margin, args.rho)
        with torch.no_grad():
            rho_now = loss.get_rho().item()
        rec = dict(round=r, n_below_margin=int(bad.shape[0]), n_violations=n_strict, worst=worst, rho=rho_now,
                   t=round(time.time() - t_start))
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if n_strict == 0:
            break
        cell = (up - lo) / (args.grid - 1)
        for _ in range(args.steps):
            jit = bad[torch.randint(0, bad.shape[0], (min(2048, 8 * bad.shape[0]),), device=DEV)]
            jit = (jit + (torch.rand_like(jit) - 0.5) * 2 * cell).clamp(lo, up)
            rnd = lo + torch.rand(2048, 2, device=DEV) * (up - lo)
            x = torch.cat([bad[:4096], jit, rnd])
            mg, Vz = loss.raw_margin(x)
            l = (torch.relu(args.margin - mg) * (Vz <= args.rho).float()).mean()
            opt.zero_grad(); l.backward(); opt.step()
    # save in the layout of train_multimode_yang.py so that fault_screen / verify_faults work unchanged
    shutil.copy(src / "args.json", out / "args.json")
    torch.save({"ctrl": ctrl.state_dict(), "V": V.state_dict(), "scale": targs["limit_scales"][-1],
                "x_boundary": xb.cpu(), "rho": args.rho},
               out / f"ckpt_scale{targs['limit_scales'][-1]:.1f}.pth")
    (out / "finetune_log.json").write_text(json.dumps(log, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
