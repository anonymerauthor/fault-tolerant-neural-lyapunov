"""Local exponential stability at the equilibrium for every controller (nominal and single-fault).

The verifier excludes a small box H around the equilibrium. Inside H we use Lyapunov's indirect
method: if no ReLU pre-activation is zero at the equilibrium and the input is not saturated there,
the closed loop z -> f(z, pi(z)) is smooth near z = 0, and a Schur-stable Jacobian implies local
exponential stability. For each controller we report the smallest |pre-activation| at the
equilibrium, the saturation margin of u*, and the spectral radius of the closed-loop Jacobian.

Usage: python src/local_stability.py
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fault_screen as FS  # noqa: E402
import quad  # noqa: E402

torch.set_default_dtype(torch.float64)


def check(c: FS.FixedMaskController, dyn, n):
    z0 = torch.zeros(1, n)
    # pre-activations of all hidden layers at z = 0
    h, pre = z0, []
    for mod in c.net:
        h = mod(h)
        if isinstance(mod, torch.nn.Linear):
            pre.append(h.abs().min().item())
    pre = pre[:-1]  # last Linear is the output layer
    u0 = c(z0)
    sat = min((u0 - c.u_lo).min().item(), (c.u_up - u0).min().item())
    J = torch.autograd.functional.jacobian(lambda z: dyn(z.view(1, n), c(z.view(1, n))).view(n), torch.zeros(n))
    return min(pre), sat, torch.linalg.eigvals(J).abs().max().item()


def run_pendulum(run):
    a, ck, systems, ctrl, V, lim, ec = FS.load_run(ROOT / run)
    ctrl = ctrl.double()
    res = []
    for m, dyn in enumerate(systems):
        dyn = dyn.double()
        x_eq, u_eq = ctrl.eq(m)
        nom = ctrl.hard_masks(m)
        for bit in [None] + FS.all_bits(ctrl):
            mask = nom if bit is None else FS.flip(nom, bit)
            c = FS.FixedMaskController(ctrl, mask, x_eq.double(), u_eq.double(), ec).double()
            res.append(check(c, dyn, 2))
    return res


def run_quad(ckpt, skip):
    dyn, ctrl, V, lim = quad.load_quad(ckpt)
    ctrl, dyn = ctrl.double(), dyn.double()
    x_eq, u_eq = ctrl.eq(0)
    nom = ctrl.hard_masks(0)
    res = []
    for i, bit in enumerate([None] + FS.all_bits(ctrl)):
        if i in skip:
            continue
        mask = nom if bit is None else FS.flip(nom, bit)
        c = FS.FixedMaskController(ctrl, mask, x_eq.double(), u_eq.double(), False).double()
        res.append(check(c, dyn, 6))
    return res


def summary(name, res):
    pre, sat, rad = zip(*res)
    d = dict(model=name, controllers=len(res), min_preact=min(pre), min_sat_margin=min(sat),
             max_spectral_radius=max(rad), all_schur=all(r < 1 for r in rad))
    print(json.dumps(d))
    return d


if __name__ == "__main__":
    out = [summary("pendulum seed0 fault-aware", run_pendulum("runs/f2c_kappa")),
           summary("pendulum seed1 w0.1", run_pendulum("runs/s1_w0.1_kappa")),
           summary("quadrotor fa2 (certified 70)", run_quad(str(ROOT / "runs/quad_fa2/ckpt.pth"), {59, 61, 62, 63, 64}))]
    (ROOT / "results/local_stability.json").write_text(json.dumps(out, indent=1))


def jacobians_pendulum(run):
    a, ck, systems, ctrl, V, lim, ec = FS.load_run(ROOT / run)
    ctrl = ctrl.double()
    out = []
    for m, dyn in enumerate(systems):
        dyn = dyn.double()
        x_eq, u_eq = ctrl.eq(m)
        nom = ctrl.hard_masks(m)
        Js = []
        for bit in [None] + FS.all_bits(ctrl):
            mask = nom if bit is None else FS.flip(nom, bit)
            c = FS.FixedMaskController(ctrl, mask, x_eq.double(), u_eq.double(), ec).double()
            Js.append(torch.autograd.functional.jacobian(lambda z: dyn(z.view(1, 2), c(z.view(1, 2))).view(2), torch.zeros(2)))
        out.append(torch.stack(Js))
    return out


def jacobians_quad(ckpt, skip):
    dyn, ctrl, V, lim = quad.load_quad(ckpt)
    ctrl, dyn = ctrl.double(), dyn.double()
    x_eq, u_eq = ctrl.eq(0)
    nom = ctrl.hard_masks(0)
    Js = []
    for i, bit in enumerate([None] + FS.all_bits(ctrl)):
        if i in skip:
            continue
        mask = nom if bit is None else FS.flip(nom, bit)
        c = FS.FixedMaskController(ctrl, mask, x_eq.double(), u_eq.double(), False).double()
        Js.append(torch.autograd.functional.jacobian(lambda z: dyn(z.view(1, 6), c(z.view(1, 6))).view(6), torch.zeros(6)))
    return [torch.stack(Js)]


def common_quadratic(A, steps=4000, lr=0.01, seed=0):
    """Search P = L L^T minimising max_j ||L^T A_j L^{-T}||_2 (the contraction factor of the common
    quadratic Lyapunov function x^T P^{-1}... in the norm |x|_P = |L^{-1} x|).  A value < 1 certifies
    that every switching sequence of the A_j is exponentially stable (joint spectral radius < 1)."""
    n = A.shape[-1]
    # start from the nominal discrete Lyapunov solution
    A0 = A[0]
    P = torch.eye(n, dtype=A.dtype)
    for _ in range(2000):
        P = A0 @ P @ A0.T + torch.eye(n, dtype=A.dtype)
    L = torch.linalg.cholesky(P).clone().requires_grad_(True)
    opt = torch.optim.Adam([L], lr=lr)
    best = float("inf")
    for _ in range(steps):
        Lt = torch.tril(L)
        M = torch.linalg.solve_triangular(Lt, A @ Lt, upper=False)  # L^{-1} A_j L
        val = torch.linalg.matrix_norm(M, 2).max()
        best = min(best, val.item())
        opt.zero_grad(); val.backward(); opt.step()
    return best


def joint():
    res = {}
    for name, Jl in [("pendulum seed0 fault-aware", jacobians_pendulum("runs/f2c_kappa")),
                     ("pendulum seed1 w0.1", jacobians_pendulum("runs/s1_w0.1_kappa")),
                     ("quadrotor fa2 (certified 70)", jacobians_quad(str(ROOT / "runs/quad_fa2/ckpt.pth"), {59, 61, 62, 63, 64}))]:
        res[name] = [common_quadratic(J) for J in Jl]
        print(name, res[name], flush=True)
    return res
