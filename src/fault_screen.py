"""Single-bit (sign) fault screening of a trained multi-mode masked controller.

For every mode m and every mask bit k (each weight and bias), the faulty controller uses mask
m with bit k flipped, i.e. parameter k changes sign. Because the controller is normalized as
u = phi(x) - phi(x*) + u*, the equilibrium is preserved under ANY mask fault, so the same
Lyapunov certificate (V, rho, kappa) can be checked for the faulty controller.

Per (mode, bit) we record
  s_k       analytic sensitivity (Corollary 1, mask-invariant)
  jump      CROWN-certified sup_z |u_faulty(z) - u(z)| over the box (2-D grid splitting)
  pgd_viol  max violation of the Lyapunov condition found by PGD (0 => candidate for formal proof)

Usage: python src/fault_screen.py --run runs/e2b_masked --rho 0.136 --kappa 0.005
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

ROOT = Path(__file__).resolve().parents[1]
LYA = ROOT / "external" / "Lyapunov_Stable_NN_Controllers"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(LYA))
sys.modules.setdefault("wandb", types.ModuleType("wandb"))

import neural_lyapunov_training.lyapunov as lyapunov  # noqa: E402
import neural_lyapunov_training.train_utils as train_utils  # noqa: E402
from masked_controller import MaskedController  # noqa: E402
from masked_mlp import bit_sensitivities  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FixedMaskController(nn.Module):
    """u(z) = clamp(phi(z + x*; mask) - phi(x*; mask) + u*), plain layers (auto_LiRPA friendly)."""

    def __init__(self, ctrl: MaskedController, mask, x_eq, u_eq, error_coords: bool):
        super().__init__()
        mw, mb = mask
        layers = []
        for l in range(ctrl.phi.n_layers):
            lin = nn.Linear(ctrl.phi.dims[l], ctrl.phi.dims[l + 1])
            A = (mw[l] * ctrl.phi.weights[l]).detach()
            b = (mb[l] * ctrl.phi.biases[l]).detach()
            if error_coords and l == 0:
                b = b + A @ x_eq
            lin.weight.data, lin.bias.data = A.clone(), b.clone()
            layers.append(lin)
            if l < ctrl.phi.n_layers - 1:
                layers.append(ctrl.activation())
        self.net = nn.Sequential(*layers)
        z_eq = torch.zeros_like(x_eq) if error_coords else x_eq
        with torch.no_grad():
            self.register_buffer("offset", (self.net(z_eq.unsqueeze(0)) - u_eq).squeeze(0))
        self.register_buffer("u_lo", ctrl.u_lo.clone())
        self.register_buffer("u_up", ctrl.u_up.clone())

    def forward(self, z):
        raw = self.net(z) - self.offset
        f1 = torch.relu(raw - self.u_lo) + self.u_lo
        return -(torch.relu(self.u_up - f1) - self.u_up)


class _Diff(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = a, b

    def forward(self, z):
        return self.b(z) - self.a(z)


@torch.no_grad()
def crown_jump(c0, c1, lo, up, splits=32) -> float:
    d = lo.numel()
    g = [torch.linspace(float(lo[i]), float(up[i]), splits + 1) for i in range(d)]
    L = torch.stack(torch.meshgrid(*[t[:-1] for t in g], indexing="ij"), -1).reshape(-1, d).to(DEV)
    U = torch.stack(torch.meshgrid(*[t[1:] for t in g], indexing="ij"), -1).reshape(-1, d).to(DEV)
    model = BoundedModule(_Diff(c0, c1).to(DEV).eval(), torch.zeros(1, d, device=DEV), device=DEV, verbose=0)
    lb, ub = model.compute_bounds(x=(BoundedTensor((L + U) / 2, PerturbationLpNorm(norm=float("inf"), x_L=L, x_U=U)),),
                                  method="CROWN")
    return torch.maximum(ub.abs(), lb.abs()).max().item()


def load_run(run: Path):
    a = json.loads((run / "args.json").read_text())
    ck = torch.load(run / f"ckpt_scale{a['limit_scales'][-1]:.1f}.pth", map_location="cpu")
    if a.get("setpoints"):
        from setpoint import ErrorCoordDynamics, setpoint_equilibrium
        systems = [ErrorCoordDynamics(t, dt=a["dt"], mass=a["masses"][0]) for t in a["setpoints"]]
        eqs = [setpoint_equilibrium(t, a["masses"][0]) for t in a["setpoints"]]
        x_eq, u_eq, error_coords = [e[0] for e in eqs], [e[1] for e in eqs], True
    else:
        from train_pendulum_multimode import make_dynamics
        systems = [make_dynamics(m, a["dt"])[1] for m in a["masses"]]
        x_eq, u_eq, error_coords = [0.0, 0.0], [0.0], False
    ctrl = MaskedController(2, 1, a["hidden"], a["nlayer"], len(systems), u_lo=[-a["u_max"]], u_up=[a["u_max"]],
                            x_eq=x_eq, u_eq=u_eq, score_scale=a.get("score_scale", 1.0),
                            delta_scores=a.get("delta_scores", False))
    ctrl.load_state_dict(ck["ctrl"])
    V = lyapunov.NeuralNetworkLyapunov(goal_state=torch.zeros(2), hidden_widths=[16, 16, 8], x_dim=2, R_rows=3,
                                       absolute_output=True, eps=a.get("V_eps", 0.01), activation=nn.LeakyReLU,
                                       V_psd_form="L1")
    V.load_state_dict(ck["V"])
    lim = torch.tensor([a["limit"][i] * a["limit_scales"][-1] for i in range(2)])
    return a, ck, systems, ctrl, V, lim, error_coords


def all_bits(ctrl):
    out = []
    for l in range(ctrl.phi.n_layers):
        for idx in torch.cartesian_prod(*[torch.arange(n) for n in ctrl.phi.weights[l].shape]).tolist():
            out.append(("w", l, tuple(idx)))
        for i in range(ctrl.phi.biases[l].shape[0]):
            out.append(("b", l, (i,)))
    return out


def flip(mask, bit):
    kind, l, idx = bit
    mw = [t.clone() for t in mask[0]]
    mb = [t.clone() for t in mask[1]]
    (mw if kind == "w" else mb)[l][idx] *= -1
    return mw, mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--kappa", type=float, default=0.005)
    ap.add_argument("--pgd_samples", type=int, default=16384)
    ap.add_argument("--pgd_steps", type=int, default=150)
    ap.add_argument("--no_crown", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    run = (ROOT / args.run).resolve()
    a, ck, systems, ctrl, V, lim, error_coords = load_run(run)
    ctrl.to(DEV).eval(); V.to(DEV).eval()
    for s in systems:
        s.to(DEV)
    lo, up = (-lim).to(DEV), lim.to(DEV)
    xb = ck["x_boundary"].to(DEV)
    with torch.no_grad():
        rho_mult = args.rho / V(xb).min().item()  # LyapunovDerivativeLoss uses rho = mult * min V(x_boundary)
    r0 = float(lim.norm())
    sens = bit_sensitivities(ctrl.phi, r0)
    bits = all_bits(ctrl)

    out = run / f"faults_rho{args.rho:g}_k{args.kappa:g}.csv"
    rows, t0 = [], time.time()
    for m, dyn in enumerate(systems):
        x_eq, u_eq = (t.to(DEV) for t in ctrl.eq(m))
        nominal_mask = ctrl.hard_masks(m)
        c_nom = FixedMaskController(ctrl, nominal_mask, x_eq, u_eq, error_coords).to(DEV)
        for bit in [None] + bits:
            mask = nominal_mask if bit is None else flip(nominal_mask, bit)
            c = c_nom if bit is None else FixedMaskController(ctrl, mask, x_eq, u_eq, error_coords).to(DEV)
            lf = lyapunov.LyapunovDerivativeLoss(dyn, c, V, box_lo=lo, box_up=up, rho_multiplier=rho_mult,
                                                 kappa=args.kappa, hard_max=True)
            lf.x_boundary = xb
            x0 = lo + torch.rand(args.pgd_samples, 2, device=DEV) * (up - lo)
            adv = train_utils.pgd_attack(x0, lf, eps=up, steps=args.pgd_steps, lower_boundary=lo,
                                         upper_boundary=up, direction="minimize").detach()
            with torch.no_grad():
                v = torch.relu(-lf(adv)).squeeze(-1)
            if bit is None:
                s_k, jump, name = 0.0, 0.0, "nominal"
            else:
                kind, l, idx = bit
                s_k = float(sens[l][0 if kind == "w" else 1][idx])
                jump = float("nan") if args.no_crown else crown_jump(c_nom, c, lo, up)
                name = f"{kind}{l}[{','.join(map(str, idx))}]"
            rows.append(dict(mode=m, bit=name, s_k=s_k, crown_jump=jump, pgd_max_viol=v.max().item(),
                             pgd_n_viol=int((v > 0).sum())))
        n_ok = sum(1 for r in rows if r["mode"] == m and r["bit"] != "nominal" and r["pgd_max_viol"] == 0)
        print(f"mode {m}: nominal viol={rows[-len(bits) - 1]['pgd_max_viol']:.2e}; "
              f"{n_ok}/{len(bits)} single-bit faults with no PGD counterexample  ({time.time() - t0:.0f}s)", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    main()
