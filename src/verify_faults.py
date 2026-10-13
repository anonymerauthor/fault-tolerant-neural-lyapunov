"""Formal verification of ALL single sign-bit faults of one mode in ONE alpha-beta-CROWN problem.

The "fault bank" model stacks the nominal controller and its n_params single-bit-faulty
variants as one block-diagonal network (F = 1 + n_params parallel controllers sharing the
input z). For the set-point pendulum in error coordinates z = x - x* (explicit Euler),
it outputs, for every variant f,
    dV_f   = (1-kappa) V(z) - V(z+_f)                         (Lyapunov decrease margin)
    box_f  = margins of z+_f inside the verification box     (forward invariance)
plus V(z). The VNNLIB property asks for a counterexample with V(z) <= rho and ANY margin < 0;
"safe" therefore certifies the Lyapunov condition for the nominal controller and every
single-bit fault at once (Lemma 5 keeps the equilibrium at z = 0 for every variant).

Usage:
  python src/verify_faults.py --run runs/f2_faultaware --rho 0.165 --kappa 0.005 [--modes 0 1 2]
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


def _variant_masks(ctrl, mode):
    """Nominal mask + all single-bit flips, in FaultAwareLoss order (all weights, then all biases)."""
    from fault_aware import _fault_stack
    mw, mb = ctrl.hard_masks(mode)
    st = _fault_stack(mw + mb)
    return st[:len(mw)], st[len(mw):]


class VPlain(nn.Module):
    """Linear/ReLU-only copy of NeuralNetworkLyapunov (absolute_output, L1 psd, goal = 0):
        V(x) = |phi(x) - phi(0)| + || (eps I + R^T R) x ||_1
    phi(0) is folded into the last bias; no broadcasting ops (auto_LiRPA-friendly on (B, F, 2))."""

    def __init__(self, V):
        super().__init__()
        assert V.absolute_output and V.V_psd_form == "L1" and V.nominal is None
        assert torch.all(V.goal_state == 0)
        layers = [type(l)(*([] if not isinstance(l, nn.Linear) else [l.in_features, l.out_features]))
                  if isinstance(l, nn.Linear) else l for l in V.net]
        for new, old in zip(layers, V.net):
            if isinstance(old, nn.Linear):
                new.load_state_dict(old.state_dict())
        with torch.no_grad():
            phi0 = V.net(torch.zeros(1, V.x_dim)).squeeze(0)
            layers[-1].bias -= phi0
        self.phi = nn.Sequential(*layers)
        n = V.x_dim
        M = V.eps * torch.eye(n) + V.R.detach().T @ V.R.detach()
        self.psd = nn.Linear(n, n, bias=False)
        self.psd.weight.data = M.T.clone()          # x @ M
        self.sum = nn.Linear(n, 1, bias=False)
        self.sum.weight.data = torch.ones(1, n)

    def forward(self, x):
        p = self.phi(x)
        r = self.psd(x)
        return torch.relu(p) + torch.relu(-p) + self.sum(torch.relu(r) + torch.relu(-r))


class VBlock(nn.Module):
    """F parallel copies of VPlain on a flat input (B, 2F) = [x_1, ..., x_F]; output (B, F).
    Built only from block-diagonal Linear layers and ReLUs (fully 2-D for auto_LiRPA)."""

    def __init__(self, vp: VPlain, F: int):
        super().__init__()
        layers = []
        for l in vp.phi:
            if isinstance(l, nn.Linear):
                lin = nn.Linear(l.in_features * F, l.out_features * F)
                lin.weight.data = torch.block_diag(*[l.weight.data] * F)
                lin.bias.data = l.bias.data.repeat(F)
                layers.append(lin)
            else:
                layers.append(l)
        self.phi = nn.Sequential(*layers)
        n = vp.psd.in_features
        self.psd = nn.Linear(n * F, n * F, bias=False)
        self.psd.weight.data = torch.block_diag(*[vp.psd.weight.data] * F)
        self.sum = nn.Linear(n * F, F, bias=False)
        self.sum.weight.data = torch.block_diag(*[vp.sum.weight.data] * F)

    def forward(self, x):
        p = self.phi(x)
        r = self.psd(x)
        return torch.relu(p) + torch.relu(-p) + self.sum(torch.relu(r) + torch.relu(-r))


class FaultBankModel(nn.Module):
    def __init__(self, run: str, mode: int, kappa: float, box_lo, box_up, faults: bool = True,
                 start: int = 0, end: int = None):
        super().__init__()
        import fault_screen as FS
        from setpoint import DAMPING, G, LENGTH
        a, ck, systems, ctrl, V, lim, ec = FS.load_run(Path(run))
        assert ec, "fault bank currently implemented for set-point (error-coordinate) pendulum runs"
        x_eq, u_eq = ctrl.eq(mode)
        sw, sb = _variant_masks(ctrl, mode)
        if not faults:
            sw, sb = [s[:1] for s in sw], [s[:1] for s in sb]
        else:  # variant subset [start, end) (index 0 = nominal)
            sw, sb = [s[start:end] for s in sw], [s[start:end] for s in sb]
        F = sw[0].shape[0]
        self.F = F
        L = ctrl.phi.n_layers
        layers = []
        for l in range(L):
            A = sw[l] * ctrl.phi.weights[l].detach()      # (F, out, in)
            b = sb[l] * ctrl.phi.biases[l].detach()       # (F, out)
            if l == 0:
                b = b + torch.einsum("foi,i->fo", A, x_eq)  # fold x* (error coordinates)
                W = A.reshape(F * A.shape[1], A.shape[2])   # shared input z
            else:
                W = torch.block_diag(*A)                     # (F*out, F*in)
            lin = nn.Linear(W.shape[1], W.shape[0])
            lin.weight.data, lin.bias.data = W.clone(), b.reshape(-1).clone()
            layers.append(lin)
            if l < L - 1:
                layers.append(ctrl.activation())
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.register_buffer("offset", self.net(torch.zeros(1, 2)).squeeze(0) - u_eq)  # (F,)
        self.u_lo, self.u_up = float(ctrl.u_lo), float(ctrl.u_up)
        self.V = VPlain(V)
        self.Vb = VBlock(self.V, F)
        P_th = torch.zeros(F, 2 * F)
        P_thd = torch.zeros(F, 2 * F)
        P_th[torch.arange(F), 2 * torch.arange(F)] = 1.0
        P_thd[torch.arange(F), 2 * torch.arange(F) + 1] = 1.0
        self.place_th = nn.Linear(F, 2 * F, bias=False)
        self.place_th.weight.data = P_th.T.clone()
        self.place_thd = nn.Linear(F, 2 * F, bias=False)
        self.place_thd.weight.data = P_thd.T.clone()
        m = a["masses"][0]
        self.dt, self.kappa = a["dt"], kappa
        self.c_damp, self.c_grav, self.c_u = -DAMPING / (m * LENGTH ** 2), G / LENGTH, 1.0 / (m * LENGTH ** 2)
        self.theta_star = float(x_eq[0])
        self.lo = [float(v) for v in box_lo]
        self.up = [float(v) for v in box_up]
        self.register_buffer("ones_F", torch.ones(1, F))

    def forward(self, z):
        B = z.shape[0]
        raw = self.net(z) - self.offset                                    # (B, F)
        f1 = torch.relu(raw - self.u_lo) + self.u_lo
        u = -(torch.relu(self.u_up - f1) - self.u_up)
        th, thd = z[:, 0:1], z[:, 1:2]
        acc = self.c_damp * thd + self.c_grav * torch.sin(th + self.theta_star)   # (B, 1)
        thd_next = thd @ self.ones_F + self.dt * (acc @ self.ones_F + self.c_u * u)  # (B, F)
        th_next = th + self.dt * thd                                       # (B, 1) explicit Euler
        th_next_F = th_next @ self.ones_F                                  # (B, F)
        # flat (B, 2F) next-state vector built with constant linear maps; V of all variants via VBlock
        zn = self.place_th(th_next_F) + self.place_thd(thd_next)          # (B, 2F)
        Vz = self.V(z)                                                     # (B, 1)
        Vn = self.Vb(zn)                                                   # (B, F)
        dV = (1 - self.kappa) * Vz - Vn                                    # (B, F)
        m_th = torch.cat([th_next - self.lo[0], self.up[0] - th_next], dim=1)        # (B, 2)
        m_thd = torch.cat([thd_next - self.lo[1], self.up[1] - thd_next], dim=1)     # (B, 2F)
        return torch.cat([dV, m_th, m_thd, Vz], dim=1)                     # (B, 3F + 3)


def create_fault_bank(run, mode, kappa, box_lo, box_up, faults=True, start=0, end=None):
    return FaultBankModel(run, mode, kappa, box_lo, box_up, faults, start, end)


def write_specs(vdir: Path, lim, hole: float, n_out: int, rho: float, tol: float = 1e-6):
    from neural_lyapunov_training.models import box_data
    _, _, dmax, dmin, _ = box_data(lower_limit=[-lim[0], -lim[1]], upper_limit=[lim[0], lim[1]], ndim=2,
                                   scale=1.0, hole_size=hole)
    (vdir / "specs").mkdir(parents=True, exist_ok=True)
    names = []
    for r in range(dmin.shape[0]):
        p = vdir / "specs" / f"spec_{r}.vnnlib"
        with open(p, "w") as f:
            f.write("; fault-bank Lyapunov property\n")
            for i in range(2):
                f.write(f"(declare-const X_{i} Real)\n")
            for j in range(n_out):
                f.write(f"(declare-const Y_{j} Real)\n")
            for i in range(2):
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
    Customized("{factory}", "create_fault_bank", run="{run}", mode={mode}, kappa={kappa},
    box_lo=[{lo0}, {lo1}], box_up=[{up0}, {up1}], faults={faults}, start={start}, end={end})
  input_shape: [-1, 2]
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
      sb_coeff_thresh: 0.001
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--kappa", type=float, default=0.005)
    ap.add_argument("--modes", type=int, nargs="*", default=None)
    ap.add_argument("--hole", type=float, default=0.001)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--branching", default="naive", choices=["sb", "naive"])
    ap.add_argument("--chunk", type=int, default=30, help="variants per verification problem")
    ap.add_argument("--chunks", type=int, nargs="*", default=None, help="subset of chunk indices to run")
    ap.add_argument("--nominal_only", action="store_true", help="sanity check: verify only the nominal variant")
    args = ap.parse_args()

    run = (ROOT / args.run).resolve()
    targs = json.loads((run / "args.json").read_text())
    lim = [targs["limit"][i] * targs["limit_scales"][-1] for i in range(2)]
    tag = "nominal" if args.nominal_only else "faults"
    vdir = run / f"verify_{tag}_rho{args.rho:g}_k{args.kappa:g}"
    modes = args.modes if args.modes is not None else list(range(len(targs["setpoints"])))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(LYA), str(ABC), str(ROOT / "src")])
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    summary = {"rho": args.rho, "kappa": args.kappa, "limit": lim, "faults": not args.nominal_only, "modes": []}
    for m in modes:
        n_var = 1 if args.nominal_only else FaultBankModel(str(run), m, args.kappa, [-lim[0], -lim[1]], lim).F
        starts = list(range(0, n_var, args.chunk))
        for ci, st in enumerate(starts):
            if args.chunks is not None and ci not in args.chunks:
                continue
            en = min(st + args.chunk, n_var)
            mdir = vdir / f"mode{m}" / f"chunk{ci:02d}_{st}-{en}"
            mdir.mkdir(parents=True, exist_ok=True)
            F = en - st
            n_out = 3 * F + 3
            n_regions = write_specs(mdir, lim, args.hole, n_out, args.rho)
            cfg = mdir / "config.yaml"
            cfg.write_text(CONFIG.format(root=mdir.as_posix(), factory=(ROOT / "src" / "verify_faults.py").as_posix(),
                                         run=run.as_posix(), mode=m, kappa=args.kappa, lo0=-lim[0], lo1=-lim[1],
                                         up0=lim[0], up1=lim[1], faults=not args.nominal_only, start=st, end=en,
                                         batch=args.batch, timeout=args.timeout, branching=args.branching))
            logf = mdir / "abcrown.log"
            with open(logf, "w") as f:
                subprocess.run([sys.executable, "abcrown.py", "--config", str(cfg)], cwd=ABC, env=env,
                               stdout=f, stderr=subprocess.STDOUT)
            txt = logf.read_text(errors="ignore")
            safe = re.search(r"total verified \(safe/unsat\): (\d+)", txt)
            uns = re.search(r"total falsified \(unsafe/sat\): (\d+) , timeout: (\d+)", txt)
            mt = re.search(r"mean time for ALL instances \(total \d+\):([\d.]+), max time: ([\d.]+)", txt)
            rec = dict(mode=m, chunk=ci, variants=[st, en], regions=n_regions,
                       safe=int(safe.group(1)) if safe else None,
                       falsified=int(uns.group(1)) if uns else None, timeout=int(uns.group(2)) if uns else None,
                       mean_time=float(mt.group(1)) if mt else None, max_time=float(mt.group(2)) if mt else None)
            summary["modes"].append(rec)
            print(json.dumps(rec), flush=True)
            (vdir / "summary_partial.json").write_text(json.dumps(summary, indent=2))
    summary["all_verified"] = all(r["safe"] == r["regions"] for r in summary["modes"])
    (vdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("ALL VERIFIED:", summary["all_verified"])


if __name__ == "__main__":
    main()
