"""Close the verification gap at the excluded box H around the origin (Remark 1).

The fault bank is verified on Omega_rho minus H, where H = 0.001 * B per axis. Here we bound, with plain CROWN
(no branching; H is tiny), for every controller j of every mode:
    c_H  = max over z in H of max( V(z), V(f(z, pi_j(z))) )
and check that f(H) stays inside B (lower bounds of all box margins > 0).
If c_H < rho, then for every trajectory V_{t+1} <= max((1 - kappa) V_t, c_H): outside H by the verified
decrease, inside H by the bound. Hence Omega_rho and every sublevel set {V <= c} with c >= c_H are forward
invariant, and V_t <= max((1 - kappa)^t V_0, c_H) for every fault sequence.

Usage: python src/hole_check.py
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fault_screen as FS  # noqa: E402,F401  (sets up paths)
from verify_faults import FaultBankModel  # noqa: E402
from verify_faults_quad import QuadFaultBank  # noqa: E402
import quad  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOLE = 0.001


class NextV(nn.Module):
    """Outputs [V(z) (1), V(z+_j) (F), box margins (...)] from a fault-bank model."""

    def __init__(self, bank, kind):
        super().__init__()
        self.bank, self.kind, self.F = bank, kind, bank.F

    def forward(self, z):
        o = self.bank(z)
        F = self.F
        Vz = o[:, -1:]
        dV = o[:, :F]
        kappa = self.bank.kappa
        Vn = (1 - kappa) * Vz - dV
        margins = o[:, F:-1]
        return torch.cat([Vz, Vn, margins], dim=1)


def bounds(model, lim):
    lo, up = (-HOLE * lim).view(1, -1).to(DEV), (HOLE * lim).view(1, -1).to(DEV)
    bm = BoundedModule(model.to(DEV).eval(), torch.zeros_like(lo), device=DEV, verbose=0)
    x = BoundedTensor((lo + up) / 2, PerturbationLpNorm(norm=float("inf"), x_L=lo, x_U=up))
    lb, ub = bm.compute_bounds(x=(x,), method="CROWN")
    return lb.squeeze(0).cpu(), ub.squeeze(0).cpu()


def pendulum(run, rho, kappa):
    a, ck, systems, ctrl, V, lim, ec = FS.load_run(ROOT / run)
    out = []
    for mode in range(len(systems)):
        bank = FaultBankModel(str(ROOT / run), mode, kappa, -lim, lim, faults=True)
        lb, ub = bounds(NextV(bank, "pend"), lim)
        F = bank.F
        out.append(dict(mode=mode, controllers=F, V_H=float(ub[0]), Vnext_H=float(ub[1:1 + F].max()),
                        min_box_margin=float(lb[1 + F:].min())))
    c = max(max(r["V_H"], r["Vnext_H"]) for r in out)
    return dict(model=run, rho=rho, c_H=c, c_over_rho=c / rho, box_ok=all(r["min_box_margin"] > 0 for r in out),
                modes=out)


def quadrotor(ckpt, rho, skip):
    _, _, _, lim = quad.load_quad(ckpt)
    res = []
    bank_all = QuadFaultBank(ckpt, 0.0, True)
    keep = [i for i in range(bank_all.F) if i not in skip]
    for i in keep:
        bank = QuadFaultBank(ckpt, 0.0, True, i, i + 1)
        lb, ub = bounds(NextV(bank, "quad"), lim)
        res.append((float(ub[0]), float(ub[1]), float(lb[2:].min())))
    c = max(max(a, b) for a, b, _ in res)
    return dict(model=ckpt, rho=rho, controllers=len(res), c_H=c, c_over_rho=c / rho,
                box_ok=all(m > 0 for _, _, m in res))


if __name__ == "__main__":
    r = [pendulum("runs/f2c_kappa", 0.165, 0.005), pendulum("runs/s1_w0.1_kappa", 0.1807, 0.005)]
    for x in r:
        print(json.dumps({k: v for k, v in x.items() if k != "modes"}), flush=True)
    q = quadrotor(str(ROOT / "runs/quad_fa2/ckpt.pth"), 0.5, {59, 61, 62, 63, 64})
    print(json.dumps(q), flush=True)
    (ROOT / "results/hole_check.json").write_text(json.dumps(r + [q], indent=1))
