"""Fault-bank verification for the 2-D quadrotor (6 states, 2 thrusts, quadratic V).

Same idea as verify_faults.py: the nominal controller and a chunk of its single sign-bit faults
are stacked as one block network, and one alpha-beta-CROWN problem checks, for every variant f,
    dV_f = (1-kappa) V(x) - V(x+_f) >= 0   and   x+_f inside the box,   whenever V(x) <= rho.
The plant step follows SecondOrderDiscreteTimeSystem (explicit Euler velocity, midpoint position,
dt = 0.01). All tensors stay 2-D (batch x features).

Usage: python src/verify_faults_quad.py --ckpt <ckpt> --rho 1.3392 --kappa 0 --chunk 15
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
LYA = ROOT / "external" / "Lyapunov_Stable_NN_Controllers"
ABC = ROOT / "external" / "alpha-beta-CROWN" / "complete_verifier"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(LYA))


def _lin(W, b=None):
    lin = nn.Linear(W.shape[1], W.shape[0], bias=b is not None)
    lin.weight.data = W.clone()
    if b is not None:
        lin.bias.data = b.clone()
    return lin


class QuadFaultBank(nn.Module):
    def __init__(self, ckpt: str, kappa: float, faults: bool = True, start: int = 0, end: int = None):
        super().__init__()
        import quad
        from verify_faults import _variant_masks
        dyn, ctrl, V, lim = quad.load_quad(ckpt)
        cont = dyn.continuous_time_system
        x_eq, u_eq = ctrl.eq(0)
        sw, sb = _variant_masks(ctrl, 0)
        if not faults:
            sw, sb = [s[:1] for s in sw], [s[:1] for s in sb]
        else:
            sw, sb = [s[start:end] for s in sw], [s[start:end] for s in sb]
        F = sw[0].shape[0]
        self.F = F
        L = ctrl.phi.n_layers
        layers = []
        for l in range(L):
            A = sw[l] * ctrl.phi.weights[l].detach()           # (F, out, in)
            b = sb[l] * ctrl.phi.biases[l].detach()            # (F, out)
            W = A.reshape(F * A.shape[1], A.shape[2]) if l == 0 else torch.block_diag(*A)
            layers.append(_lin(W, b.reshape(-1)))
            if l < L - 1:
                layers.append(ctrl.activation())
        self.net = nn.Sequential(*layers)                      # x (B,6) -> raw (B, 2F), [u1_f, u2_f] per f
        with torch.no_grad():
            self.register_buffer("offset", self.net(x_eq.unsqueeze(0)).squeeze(0) - u_eq.repeat(F))
        self.u_lo, self.u_up = float(ctrl.u_lo[0]), float(ctrl.u_up[0])
        # thrust sum / difference per variant: (B,2F) -> (B,F)
        S = torch.zeros(F, 2 * F); D = torch.zeros(F, 2 * F)
        idx = torch.arange(F)
        S[idx, 2 * idx] = 1; S[idx, 2 * idx + 1] = 1
        D[idx, 2 * idx] = 1; D[idx, 2 * idx + 1] = -1
        self.usum, self.udiff = _lin(S), _lin(D)
        self.rep = _lin(torch.ones(F, 1))                       # (B,1) -> (B,F)
        self.m, self.g, self.len, self.I = cont.mass, cont.gravity, cont.length, cont.inertia
        self.dt, self.kappa = dyn.dt, kappa
        # quadratic V(x) = x^T M x with M = eps I + R^T R
        M = V.eps * torch.eye(6) + V.R.detach().T @ V.R.detach()
        self.Mx = _lin(M.T)                                     # x -> x M
        self.Mblk = _lin(torch.block_diag(*[M.T] * F))          # flat (B,6F) -> per-variant x M
        self.sum6 = _lin(torch.ones(1, 6))
        self.sumblk = _lin(torch.block_diag(*[torch.ones(1, 6)] * F))
        # placement of per-variant state components into the flat (B, 6F) vector
        P = []
        for k in range(6):
            Pk = torch.zeros(6 * F, F)
            Pk[6 * idx + k, idx] = 1
            P.append(_lin(Pk))
        self.place = nn.ModuleList(P)
        # tile x (B,6) -> (B,6F) for the fused decrease (x+_f - x)^T M (x+_f + x)
        T = torch.zeros(6 * F, 6)
        for k in range(6):
            T[6 * idx + k, k] = 1
        self.tile = _lin(T)
        self.lo = [-float(v) for v in lim]
        self.up = [float(v) for v in lim]

    def forward(self, x):
        raw = self.net(x) - self.offset
        f1 = torch.relu(raw - self.u_lo) + self.u_lo
        u = -(torch.relu(self.u_up - f1) - self.u_up)           # (B, 2F)
        s, d = self.usum(u), self.udiff(u)                      # (B, F)
        th = x[:, 2:3]
        sin_F, cos_F = self.rep(torch.sin(th)), self.rep(torch.cos(th))
        a1 = (-1.0 / self.m) * sin_F * s
        a2 = (1.0 / self.m) * cos_F * s - self.g
        a3 = (self.len / self.I) * d
        qd = [self.rep(x[:, 3 + k:4 + k]) for k in range(3)]   # current velocities (B,F)
        q = [self.rep(x[:, k:k + 1]) for k in range(3)]
        qd_next = [qd[0] + self.dt * a1, qd[1] + self.dt * a2, qd[2] + self.dt * a3]
        q_next = [q[k] + (qd_next[k] + qd[k]) * (self.dt / 2) for k in range(3)]  # midpoint position
        comps = q_next + qd_next                                # 6 tensors (B,F)
        xn = sum(self.place[k](comps[k]) for k in range(6))     # (B, 6F)
        Vx = self.sum6(self.Mx(x) * x)                          # (B,1)
        # fused decrease (as fuse_dV in Yang et al.): V(x) - V(x+) = -(x+ - x)^T M (x+ + x); x+ - x = O(dt)
        xt = self.tile(x)
        dV = -self.sumblk((xn - xt) * self.Mblk(xn + xt)) - self.kappa * Vx
        margins = [comps[k] - self.lo[k] for k in range(6)] + [self.up[k] - comps[k] for k in range(6)]
        return torch.cat([dV] + margins + [Vx], dim=1)          # (B, 13F + 1)


def create_quad_fault_bank(ckpt, kappa, faults=True, start=0, end=None):
    return QuadFaultBank(ckpt, kappa, faults, start, end)


def write_specs(vdir: Path, lim, hole: float, n_out: int, rho: float, tol: float = 1e-6):
    from neural_lyapunov_training.models import box_data
    _, _, dmax, dmin, _ = box_data(lower_limit=[-v for v in lim], upper_limit=list(lim), ndim=6, scale=1.0,
                                   hole_size=hole)
    (vdir / "specs").mkdir(parents=True, exist_ok=True)
    names = []
    for r in range(dmin.shape[0]):
        p = vdir / "specs" / f"spec_{r}.vnnlib"
        with open(p, "w") as f:
            for i in range(6):
                f.write(f"(declare-const X_{i} Real)\n")
            for j in range(n_out):
                f.write(f"(declare-const Y_{j} Real)\n")
            for i in range(6):
                f.write(f"(assert (<= X_{i} {float(dmax[r, i])}))\n(assert (>= X_{i} {float(dmin[r, i])}))\n")
            f.write("(assert (or\n")
            for j in range(n_out - 1):
                f.write(f"  (and (<= Y_{j} -{tol}))\n")
            f.write("))\n")
            f.write(f"(assert (<= Y_{n_out - 1} {rho}))\n")
        names.append(f"specs/{p.name}")
    (vdir / "specs" / "spec.csv").write_text("\n".join(names) + "\n")
    return len(names)


CONFIG = """general:
  conv_mode: matrix
  enable_incomplete_verification: false
  root_path: {root}
  csv_name: specs/spec.csv
model:
  name: >-
    Customized("{factory}", "create_quad_fault_bank", ckpt="{ckpt}", kappa={kappa}, faults={faults},
    start={start}, end={end})
  input_shape: [-1, 6]
attack:
  pgd_order: skip
solver:
  batch_size: {batch}
  min_batch_size_ratio: 0.
  bound_prop_method: crown
bab:
  override_timeout: {timeout}
  decision_thresh: -1.e-6
  branching:
    method: {branching}
    input_split:
      enable: True
      ibp_enhancement: True
      compare_with_old_bounds: True
      adv_check: -1
      sb_coeff_thresh: 0.01
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=15)
    ap.add_argument("--chunks", type=int, nargs="*", default=None)
    ap.add_argument("--hole", type=float, default=0.001)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--batch", type=int, default=50000)
    ap.add_argument("--branching", default="sb", choices=["sb", "naive"])
    ap.add_argument("--nominal_only", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import quad
    ckpt = str(Path(args.ckpt).resolve())
    vdir = (ROOT / args.out).resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(LYA), str(ABC), str(ROOT / "src")])
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    n_var = 1 if args.nominal_only else QuadFaultBank(ckpt, args.kappa).F
    summary = {"ckpt": ckpt, "rho": args.rho, "kappa": args.kappa, "chunks": []}
    for ci, st in enumerate(range(0, n_var, args.chunk)):
        if args.chunks is not None and ci not in args.chunks:
            continue
        en = min(st + args.chunk, n_var)
        cdir = vdir / f"chunk{ci:02d}_{st}-{en}"
        cdir.mkdir(parents=True, exist_ok=True)
        n_out = 13 * (en - st) + 1
        n_reg = write_specs(cdir, quad.BOX, args.hole, n_out, args.rho)
        cfg = cdir / "config.yaml"
        cfg.write_text(CONFIG.format(root=cdir.as_posix(), factory=(ROOT / "src" / "verify_faults_quad.py").as_posix(),
                                     ckpt=Path(ckpt).as_posix(), kappa=args.kappa, faults=not args.nominal_only,
                                     start=st, end=en, batch=args.batch, timeout=args.timeout,
                                     branching=args.branching))
        logf = cdir / "abcrown.log"
        with open(logf, "w") as f:
            subprocess.run([sys.executable, "abcrown.py", "--config", str(cfg)], cwd=ABC, env=env,
                           stdout=f, stderr=subprocess.STDOUT)
        txt = logf.read_text(errors="ignore")
        safe = re.search(r"total verified \(safe/unsat\): (\d+)", txt)
        uns = re.search(r"total falsified \(unsafe/sat\): (\d+) , timeout: (\d+)", txt)
        mt = re.search(r"mean time for ALL instances \(total \d+\):([\d.]+), max time: ([\d.]+)", txt)
        rec = dict(chunk=ci, variants=[st, en], regions=n_reg, safe=int(safe.group(1)) if safe else None,
                   falsified=int(uns.group(1)) if uns else None, timeout=int(uns.group(2)) if uns else None,
                   mean_time=float(mt.group(1)) if mt else None, max_time=float(mt.group(2)) if mt else None)
        summary["chunks"].append(rec)
        print(json.dumps(rec), flush=True)
        (vdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("ALL VERIFIED:", all(r["safe"] == r["regions"] for r in summary["chunks"]))


if __name__ == "__main__":
    main()
