"""Joint training of a sign-masked multi-mode controller and a COMMON neural
Lyapunov function for the inverted pendulum with several mass modes.

Loss and certificate format follow Yang et al. (ICML 2024): for every mode m,
    V(x) <= rho  =>  V(f_m(x, u_m(x))) - (1-kappa) V(x) <= 0  and  f_m(x,u_m(x)) in box,
with rho = rho_multiplier * min_{x on box boundary} V(x) (shared by all modes).

Smoke test:  python src/train_pendulum_multimode.py --smoke
Full run:    python src/train_pendulum_multimode.py --out runs/pendulum_2mode
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Anaconda MKL + torch both ship libiomp5md.dll on Windows

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "Lyapunov_Stable_NN_Controllers"))
if "wandb" not in sys.modules:  # train_utils imports wandb only for optional logging
    import types
    sys.modules["wandb"] = types.ModuleType("wandb")

import neural_lyapunov_training.dynamical_system as dynamical_system  # noqa: E402
import neural_lyapunov_training.lyapunov as lyapunov  # noqa: E402
import neural_lyapunov_training.pendulum as pendulum  # noqa: E402
import neural_lyapunov_training.train_utils as train_utils  # noqa: E402
from masked_controller import MaskedController, hamming  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log = logging.getLogger("train")


def make_dynamics(mass: float, dt: float):
    cont = pendulum.PendulumDynamics(m=mass, l=0.5, beta=0.1)
    disc = dynamical_system.SecondOrderDiscreteTimeSystem(
        cont, dt=dt,
        position_integration=dynamical_system.IntegrationMethod.ExplicitEuler,
        velocity_integration=dynamical_system.IntegrationMethod.ExplicitEuler)
    return cont, disc


def lqr(cont):
    """Continuous LQR on the upright linearization (same as Yang et al.)."""
    m, l, b, g = cont.m, cont.l, cont.beta, cont.g
    A = np.array([[0.0, 1.0], [g / l, -b / (m * l * l)]])
    B = np.array([[0.0], [1.0 / (m * l * l)]])
    Q, R = np.eye(2), np.eye(1) * 100
    S = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = -np.linalg.solve(R, B.T @ S)
    return torch.tensor(K, dtype=torch.float32), torch.tensor(S, dtype=torch.float32)


def init_by_lqr(ctrl, V, lqrs, limit, iters_c, iters_v):
    """Regress mode m onto its LQR law, and V onto the quadratic of mode 0."""
    x = (torch.rand(50000, 2, device=DEV) - 0.5) * 2 * limit
    opt = torch.optim.Adam(ctrl.parameters(), lr=0.01)
    for it in range(iters_c):
        loss = sum(nn.functional.mse_loss(ctrl(x, m), torch.clamp(x @ K.to(DEV).T, ctrl.u_lo, ctrl.u_up))
                   for m, (K, _) in enumerate(lqrs))
        opt.zero_grad(); loss.backward(); opt.step()
    log.info(f"[init] controller LQR-fit mse={loss.item():.4f}")
    S0 = lqrs[0][1].to(DEV)
    target = (x * (x @ S0)).sum(1, keepdim=True)
    opt = torch.optim.Adam(V.parameters(), lr=0.01)
    for it in range(iters_v):
        loss = nn.functional.mse_loss(V(x), target)
        opt.zero_grad(); loss.backward(); opt.step()
    log.info(f"[init] V quadratic-fit mse={loss.item():.4f}")


def worst_violation(lossfn, lo, up, n, steps):
    x0 = lo + torch.rand(n, 2, device=DEV) * (up - lo)
    adv = train_utils.pgd_attack(x0, lossfn, eps=(up - lo) / 2, steps=steps,
                                 lower_boundary=lo, upper_boundary=up, direction="minimize").detach()
    with torch.no_grad():
        viol = torch.relu(-lossfn(adv)).squeeze(1)
    return adv, viol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/pendulum_2mode")
    ap.add_argument("--masses", type=float, nargs="+", default=[0.15, 0.25])
    ap.add_argument("--limit", type=float, nargs=2, default=[12.0, 12.0])
    ap.add_argument("--limit_scales", type=float, nargs="+",
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--kappa", type=float, default=0.01)
    ap.add_argument("--rho_multiplier", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--u_max", type=float, default=6.0)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--nlayer", type=int, default=4)
    ap.add_argument("--max_iter", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--samples_per_iter", type=int, default=16384)
    ap.add_argument("--buffer_size", type=int, default=131072)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--pgd_steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--V_eps", type=float, default=0.1)
    ap.add_argument("--roa_weight", type=float, default=1.0)
    ap.add_argument("--roa_frac", type=float, default=0.5,
                    help="candidate ROA states at this fraction of the box (axes + diagonals)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.out = args.out + "_smoke"
        args.limit_scales = [0.1, 0.2]
        args.max_iter, args.epochs, args.samples_per_iter, args.pgd_steps = 2, 2, 2048, 20

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "train.log", mode="w"), logging.StreamHandler()], force=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    train_utils.set_seed(args.seed)

    systems = [make_dynamics(m, args.dt) for m in args.masses]
    lqrs = [lqr(c) for c, _ in systems]
    n_modes = len(args.masses)
    ctrl = MaskedController(2, 1, args.hidden, args.nlayer, n_modes,
                            u_lo=[-args.u_max], u_up=[args.u_max], x_eq=[0.0, 0.0], u_eq=[0.0]).to(DEV)
    V = lyapunov.NeuralNetworkLyapunov(
        goal_state=torch.zeros(2), hidden_widths=[16, 16, 8], x_dim=2, R_rows=3,
        absolute_output=True, eps=args.V_eps, activation=nn.LeakyReLU, V_psd_form="L1").to(DEV)
    for _, d in systems:
        d.to(DEV)

    limit_full = torch.tensor(args.limit, device=DEV)
    init_by_lqr(ctrl, V, lqrs, limit_full * args.limit_scales[0], 300 if not args.smoke else 50,
                600 if not args.smoke else 50)
    log.info(f"[init] Hamming(mode0, mode1) = {hamming(ctrl, 0, 1) if n_modes > 1 else 0}")

    params = list(ctrl.parameters()) + list(V.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    history = []
    t_start = time.time()
    for scale in args.limit_scales:
        lo, up = -limit_full * scale, limit_full * scale
        losses = [lyapunov.LyapunovDerivativeLoss(d, ctrl.mode_module(m), V, box_lo=lo, box_up=up,
                                                  rho_multiplier=args.rho_multiplier, kappa=args.kappa,
                                                  hard_max=True)
                  for m, (_, d) in enumerate(systems)]
        buffers = [torch.empty(0, 2, device=DEV) for _ in range(n_modes)]
        dirs = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]],
                            dtype=torch.float32, device=DEV)
        x_cand = args.roa_frac * dirs * up  # candidate states that should lie inside {V <= rho}
        for it in range(args.max_iter):
            xb = train_utils.calc_V_extreme_on_boundary_pgd(V, lo, up, num_samples_per_boundary=1024,
                                                            eps=up, steps=50, direction="minimize")
            for lf in losses:
                lf.x_boundary = xb
            for m, lf in enumerate(losses):
                adv, _ = worst_violation(lf, lo, up, args.samples_per_iter, args.pgd_steps)
                buffers[m] = torch.cat([buffers[m], adv])[-args.buffer_size:]
            n = min(b.shape[0] for b in buffers)
            for ep in range(args.epochs):
                perm = torch.randperm(n, device=DEV)
                for k in range(0, n, args.batch_size):
                    idx = perm[k:k + args.batch_size]
                    loss = sum(torch.relu(-lf(buffers[m][idx])).mean() for m, lf in enumerate(losses))
                    if args.roa_weight > 0:  # keep the certified sublevel set from shrinking
                        loss = loss + args.roa_weight * torch.relu(V(x_cand) / losses[0].get_rho() - 1).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            # evaluation with fresh PGD
            viols = []
            for lf in losses:
                _, v = worst_violation(lf, lo, up, 8192, args.pgd_steps)
                viols.append(v.max().item())
            rho = losses[0].get_rho().item()
            ham = hamming(ctrl, 0, 1) if n_modes > 1 else 0
            rec = dict(scale=scale, it=it, loss=float(loss.item()), rho=rho, max_viol=viols, hamming01=ham,
                       t=round(time.time() - t_start, 1))
            history.append(rec)
            log.info(json.dumps(rec))
            if max(viols) == 0.0 and it >= 1:
                break  # PGD finds no counterexample for any mode -> next (larger) region
        torch.save({"ctrl": ctrl.state_dict(), "V": V.state_dict(), "scale": scale,
                    "x_boundary": xb.cpu(), "rho": rho}, out / f"ckpt_scale{scale:.1f}.pth")

    (out / "history.json").write_text(json.dumps(history, indent=1))
    # export per-mode checkpoints in the Yang et al. verification layout
    V_sd = {f"lyapunov.{k}": v.cpu() for k, v in V.state_dict().items()}
    for m in range(n_modes):
        sd = {f"controller.{k}": v for k, v in ctrl.export_mode_state_dict(m).items()}
        sd.update(V_sd)
        torch.save(sd, out / f"verify_mode{m}.pth")
    log.info(f"done in {time.time() - t_start:.0f}s; final rho={rho:.4f}, viol={viols}")


if __name__ == "__main__":
    main()
