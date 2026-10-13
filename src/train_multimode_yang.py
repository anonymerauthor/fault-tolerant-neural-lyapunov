"""Multi-mode training that reuses the training loop of Yang et al. (2024) unchanged.

The only new piece is `MultiModeLoss`: it evaluates the Yang et al. Lyapunov-derivative loss
for every (plant mode, mask) pair with a COMMON Lyapunov function and returns the worst
(minimum) output. Output >= 0 for all x in the box  <=>  every mode satisfies the condition,
so PGD, buffers, rho and the certificate format of the original pipeline apply as is.

Initialization: shared weights <- Yang et al.'s pendulum initial controller (m = 0.15),
all mode masks <- +1 (identical controllers), V <- Yang et al.'s initial Lyapunov function.

Usage:
  python src/train_multimode_yang.py --smoke
  python src/train_multimode_yang.py --out runs/pend2_yang
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
LYA = ROOT / "external" / "Lyapunov_Stable_NN_Controllers"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(LYA))
sys.modules.setdefault("wandb", types.ModuleType("wandb"))

import neural_lyapunov_training.lyapunov as lyapunov  # noqa: E402
import neural_lyapunov_training.train_utils as train_utils  # noqa: E402
from masked_controller import MaskedController, hamming  # noqa: E402
from train_pendulum_multimode import make_dynamics  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log = logging.getLogger("train_yang")


class MultiModeLoss(nn.Module):
    def __init__(self, systems, ctrl: MaskedController, V, box_lo, box_up, rho_multiplier, kappa, hard_max=True,
                 error_coords: bool = False):
        super().__init__()
        self.controller = ctrl
        self.lyapunov = V
        self.kappa = kappa
        self.box_lo, self.box_up = box_lo, box_up
        self.per_mode = nn.ModuleList([
            lyapunov.LyapunovDerivativeLoss(d, ctrl.mode_module(m, error_coords), V, box_lo=box_lo, box_up=box_up,
                                            rho_multiplier=rho_multiplier, kappa=kappa, hard_max=hard_max)
            for m, d in enumerate(systems)])
        self._x_boundary = None

    @property
    def x_boundary(self):
        return self._x_boundary

    @x_boundary.setter
    def x_boundary(self, xb):
        self._x_boundary = xb
        for lf in self.per_mode:
            lf.x_boundary = xb

    def get_rho(self):
        return self.per_mode[0].get_rho()

    def forward(self, x, save_new_x=False):
        outs = torch.cat([lf(x) for lf in self.per_mode], dim=-1)
        return outs.min(dim=-1, keepdim=True).values


def init_from_yang(ctrl: MaskedController, V, score0: float):
    sd = torch.load(LYA / "examples/data/pendulum/state_feedback/lyaloss_init.pth", map_location="cpu")
    sd = sd.get("state_dict", sd)
    with torch.no_grad():
        for l in range(ctrl.phi.n_layers):
            ctrl.phi.weights[l].copy_(sd[f"controller.net.{2 * l}.weight"])
            ctrl.phi.biases[l].copy_(sd[f"controller.net.{2 * l}.bias"])
        for s in list(ctrl.phi.w_scores) + list(ctrl.phi.b_scores):
            s.fill_(score0 / ctrl.phi.score_scale)  # all masks +1  -> every mode starts as the Yang et al. initial controller
    V.load_state_dict({k[len("lyapunov."):]: v for k, v in sd.items() if k.startswith("lyapunov.")})


def init_from_ckpt(ctrl: MaskedController, V, path, score0: float):
    """Warm start: shared weights and V from a previous run; masks reset to all +1 (deltas = 0)."""
    ck = torch.load(path, map_location="cpu")
    with torch.no_grad():
        for l in range(ctrl.phi.n_layers):
            ctrl.phi.weights[l].copy_(ck["ctrl"][f"phi.weights.{l}"])
            ctrl.phi.biases[l].copy_(ck["ctrl"][f"phi.biases.{l}"])
            # the previous run had identical (all +1) masks; keep that as the starting point
        for s in list(ctrl.phi.w_scores) + list(ctrl.phi.b_scores):
            s.zero_()
            s[0].fill_(score0 / ctrl.phi.score_scale)
            if not ctrl.phi.delta_scores:
                s[1:].fill_(score0 / ctrl.phi.score_scale)
    V.load_state_dict(ck["V"])


def init_by_lqr_regression(ctrl: MaskedController, V, systems, setpoints, mass, limit, error_coords, iters=800):
    """Architecture-agnostic init: fit each mode's controller to the LQR law of its linearization
    (at its set-point), then take V from Yang et al.'s initial Lyapunov function."""
    import numpy as np
    import scipy.linalg
    from setpoint import G, LENGTH, DAMPING
    ks = []
    for t in (setpoints or [0.0] * len(systems)):
        m_ = mass if setpoints else None
        mm = m_ if m_ is not None else 0.15
        A = np.array([[0.0, 1.0], [G / LENGTH * np.cos(t), -DAMPING / (mm * LENGTH ** 2)]])
        B = np.array([[0.0], [1.0 / (mm * LENGTH ** 2)]])
        S = scipy.linalg.solve_continuous_are(A, B, np.eye(2), np.eye(1) * 100)
        ks.append(torch.tensor(-(B.T @ S) / 100, dtype=torch.float32, device=DEV))
    z = (torch.rand(50000, 2, device=DEV) - 0.5) * 2 * limit
    params = [p for p in ctrl.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=0.01)
    for _ in range(iters):
        loss = 0
        for m in range(len(systems)):
            u_eq = ctrl.eq(m)[1]
            target = torch.clamp(z @ ks[m].T + u_eq, ctrl.u_lo, ctrl.u_up)
            loss = loss + nn.functional.mse_loss(ctrl.mode_module(m, error_coords)(z), target)
        opt.zero_grad(); loss.backward(); opt.step()
    log.info(f"[init] LQR regression mse={loss.item():.5f}")
    sd = torch.load(LYA / "examples/data/pendulum/state_feedback/lyaloss_init.pth", map_location="cpu")
    sd = sd.get("state_dict", sd)
    V.load_state_dict({k[len("lyapunov."):]: v for k, v in sd.items() if k.startswith("lyapunov.")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/pend2_yang")
    ap.add_argument("--masses", type=float, nargs="+", default=[0.15, 0.25])
    ap.add_argument("--limit", type=float, nargs=2, default=[12.0, 12.0])
    ap.add_argument("--limit_scales", type=float, nargs="+",
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--kappa", type=float, default=0.01)
    ap.add_argument("--rho_multiplier", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--u_max", type=float, default=6.0)
    ap.add_argument("--max_iter", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--samples_per_iter", type=int, default=16384)
    ap.add_argument("--buffer_size", type=int, default=131072)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--pgd_steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l1_reg", type=float, default=1e-3)
    ap.add_argument("--score_scale", type=float, default=10.0)
    ap.add_argument("--score0", type=float, default=0.05, help="initial (scaled) mask score")
    ap.add_argument("--setpoints", type=float, nargs="+", default=None,
                    help="set-point modes theta* (rad) at mass masses[0]; error coordinates")
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--nlayer", type=int, default=4)
    ap.add_argument("--init", choices=["auto", "lqr"], default="auto")
    ap.add_argument("--roa_preserve_rho", type=float, default=0.0,
                    help="if > 0: keep the initial certified set {V0 <= this} inside the new one (candidate-ROA regularizer)")
    ap.add_argument("--roa_weight", type=float, default=1.0)
    ap.add_argument("--fault_sample_k", type=int, default=0,
                    help="with --fault_aware: random-fault-injection baseline using k random faults per step")
    ap.add_argument("--keep_masks", action="store_true", help="with --init_ckpt: also load the trained masks")
    ap.add_argument("--fault_aware", action="store_true",
                    help="worst case over all modes and all single sign-bit faults (masks frozen)")
    ap.add_argument("--delta_scores", action="store_true", help="Hamming-proximity parametrization")
    ap.add_argument("--freeze_masks", action="store_true", help="shared-controller baseline (all masks +1)")
    ap.add_argument("--init_ckpt", default=None, help="warm start from a checkpoint of this script")
    ap.add_argument("--kappa_check", type=float, default=None, help="kappa for the end-of-stage PGD check")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.out += "_smoke"
        args.limit_scales, args.max_iter, args.epochs, args.samples_per_iter, args.pgd_steps = [0.1], 2, 2, 2048, 20
    args.V_eps = 0.01  # Yang et al. Lyapunov architecture

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.FileHandler(out / "train.log", mode="w"), logging.StreamHandler()])
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    train_utils.set_seed(args.seed)

    if args.setpoints:
        from setpoint import ErrorCoordDynamics, setpoint_equilibrium
        systems = [ErrorCoordDynamics(t, dt=args.dt, mass=args.masses[0]).to(DEV) for t in args.setpoints]
        eqs = [setpoint_equilibrium(t, args.masses[0]) for t in args.setpoints]
        x_eq, u_eq = [e[0] for e in eqs], [e[1] for e in eqs]
    else:
        systems = [make_dynamics(m, args.dt)[1].to(DEV) for m in args.masses]
        x_eq, u_eq = [0.0, 0.0], [0.0]
    n_modes = len(systems)
    error_coords = bool(args.setpoints)
    ctrl = MaskedController(2, 1, args.hidden, args.nlayer, n_modes, u_lo=[-args.u_max],
                            u_up=[args.u_max], x_eq=x_eq, u_eq=u_eq, score_scale=args.score_scale,
                            delta_scores=args.delta_scores)
    V = lyapunov.NeuralNetworkLyapunov(goal_state=torch.zeros(2), hidden_widths=[16, 16, 8], x_dim=2, R_rows=3,
                                       absolute_output=True, eps=args.V_eps, activation=nn.LeakyReLU,
                                       V_psd_form="L1")
    yang_arch = (args.hidden, args.nlayer) == (8, 4)
    if args.freeze_masks or args.fault_aware:
        for s in list(ctrl.phi.w_scores) + list(ctrl.phi.b_scores):
            s.requires_grad_(False)
    if args.init_ckpt and args.keep_masks:  # resume exactly (weights, masks, V)
        ck0 = torch.load(ROOT / args.init_ckpt, map_location="cpu")
        ctrl.load_state_dict(ck0["ctrl"])
        V.load_state_dict(ck0["V"])
    elif args.init_ckpt:  # warm start (any architecture): phase 2 of two-phase training
        init_from_ckpt(ctrl, V, ROOT / args.init_ckpt, args.score0)
    elif args.init == "auto" and yang_arch:
        init_from_yang(ctrl, V, args.score0)
    else:
        with torch.no_grad():  # all masks +1 at start (deltas 0)
            for s in list(ctrl.phi.w_scores) + list(ctrl.phi.b_scores):
                s.zero_()
                s[0].fill_(args.score0 / ctrl.phi.score_scale)
                if not ctrl.phi.delta_scores:
                    s[1:].fill_(args.score0 / ctrl.phi.score_scale)
        ctrl.to(DEV); V.to(DEV)
        init_by_lqr_regression(ctrl, V, systems, args.setpoints, args.masses[0],
                               torch.tensor(args.limit, device=DEV) * args.limit_scales[0], error_coords)
    kappa_check = args.kappa if args.kappa_check is None else args.kappa_check
    ctrl.to(DEV)
    V.to(DEV)

    limit_full = torch.tensor(args.limit, device=DEV)
    cand = None
    if args.roa_preserve_rho > 0:
        # states near the boundary of the INITIAL certified set; the new V must keep them inside {V <= rho}
        g = torch.linspace(-1.0, 1.0, 600, device=DEV)
        Zg = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2) * limit_full * args.limit_scales[-1]
        with torch.no_grad():
            V0 = V(Zg).squeeze(-1)
        ring = Zg[(V0 <= args.roa_preserve_rho) & (V0 >= 0.9 * args.roa_preserve_rho)]
        cand = ring[torch.randperm(ring.shape[0], device=DEV)[:1024]]
        log.info(f"[roa] {ring.shape[0]} states near the initial level set; using {cand.shape[0]} candidates")
    t0 = time.time()
    summary = []
    for scale in args.limit_scales:
        lo, up = -limit_full * scale, limit_full * scale
        if args.fault_aware:
            from fault_aware import FaultAwareLoss
            loss = FaultAwareLoss(systems, ctrl, V, lo, up, args.rho_multiplier, args.kappa, error_coords,
                                  sample_k=args.fault_sample_k)
        else:
            loss = MultiModeLoss(systems, ctrl, V, lo, up, args.rho_multiplier, args.kappa, error_coords=error_coords)
        train_utils.train_lyapunov_with_buffer(
            derivative_lyaloss=loss, positivity_lyaloss=None, observer_loss=None,
            lower_limit=lo, upper_limit=up, grid_size=torch.tensor([50, 50], device=DEV),
            learning_rate=args.lr, lr_controller=args.lr, weight_decay=0.0, max_iter=args.max_iter,
            enable_wandb=False, derivative_ibp_ratio=0, derivative_sample_ratio=1,
            positivity_ibp_ratio=0, positivity_sample_ratio=0, save_best_model=None,
            pgd_steps=args.pgd_steps, buffer_size=args.buffer_size, batch_size=args.batch_size,
            epochs=args.epochs, samples_per_iter=args.samples_per_iter, l1_reg=args.l1_reg,
            num_samples_per_boundary=1024, V_decrease_within_roa=True,
            Vmin_x_boundary_weight=0, Vmax_x_boundary_weight=0,
            candidate_roa_states=cand, candidate_roa_states_weight=(args.roa_weight if cand is not None else 0),
            always_candidate_roa_regulizer=cand is not None, logger=log)
        # PGD check per mode (same as the end of Yang et al.'s script)
        viols = []
        if args.fault_aware:
            from fault_aware import FaultAwareLoss
            chk = FaultAwareLoss(systems, ctrl, V, lo, up, args.rho_multiplier, kappa_check, error_coords)
            chk.x_boundary = loss.x_boundary
            x0 = lo + torch.rand(4096, 2, device=DEV) * (up - lo)
            adv = train_utils.pgd_attack(x0, chk, eps=up, steps=300, lower_boundary=lo, upper_boundary=up,
                                         direction="minimize").detach()
            pf = chk.per_fault_violation(adv)
            viols = [float(v[0]) for v in pf]  # nominal controllers
            n_broken = [int((v[1:] > 0).sum()) for v in pf]
            log.info("FAULTS broken single-bit faults per mode (PGD): " + json.dumps(n_broken))
        for lf in ([] if args.fault_aware else loss.per_mode):
            x0 = lo + torch.rand(32768, 2, device=DEV) * (up - lo)
            check = lyapunov.LyapunovDerivativeLoss(lf.dynamics, lf.controller, V, box_lo=lo, box_up=up,
                                                    rho_multiplier=args.rho_multiplier, kappa=kappa_check, hard_max=True)
            check.x_boundary = loss.x_boundary
            adv = train_utils.pgd_attack(x0, check, eps=up, steps=300, lower_boundary=lo, upper_boundary=up,
                                         direction="minimize").detach()
            with torch.no_grad():
                viols.append(torch.relu(-check(adv)).max().item())
        rec = dict(scale=scale, rho=loss.get_rho().item(), pgd_max_viol=viols, kappa_check=kappa_check,
                   hamming=[hamming(ctrl, 0, j) for j in range(1, n_modes)], t=round(time.time() - t0))
        summary.append(rec)
        log.info("STAGE " + json.dumps(rec))
        torch.save({"ctrl": ctrl.state_dict(), "V": V.state_dict(), "scale": scale,
                    "x_boundary": loss.x_boundary.cpu(), "rho": rec["rho"]}, out / f"ckpt_scale{scale:.1f}.pth")
        (out / "summary.json").write_text(json.dumps(summary, indent=1))

    V_sd = {f"lyapunov.{k}": v.cpu() for k, v in V.state_dict().items()}
    for m in range(n_modes):
        sd = {f"controller.{k}": v for k, v in ctrl.export_mode_state_dict(m, error_coords).items()}
        sd.update(V_sd)
        torch.save(sd, out / f"verify_mode{m}.pth")
    log.info(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
