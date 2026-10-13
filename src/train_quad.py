"""Train the 2-D quadrotor controller (Yang et al.'s architecture) with a FIXED certified level rho,
either nominally (--nominal) or fault-aware (worst case over all 74 single sign faults).

The loop is train_utils.train_lyapunov_with_buffer of Yang et al. (2024), unchanged. The level is
fixed (get_rho -> rho) so that training targets exactly the set {V <= rho} that is verified later;
kappa_train > kappa_verify provides the decrease slack that keeps branch-and-bound tractable.

Usage:
  python src/train_quad.py --out runs/quad_nominal --nominal --rho 0.5
  python src/train_quad.py --out runs/quad_fa --rho 0.5
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import quad  # noqa: E402  (sets up paths and the wandb stub)
from fault_aware import FaultAwareLoss  # noqa: E402
import neural_lyapunov_training.train_utils as train_utils  # noqa: E402

DEV = quad.DEV
log = logging.getLogger("train_quad")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=str(quad.YANG_CKPT))
    ap.add_argument("--nominal", action="store_true", help="train the nominal controller only (no faults)")
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--kappa", type=float, default=0.01)
    ap.add_argument("--max_iter", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--samples_per_iter", type=int, default=8192)
    ap.add_argument("--buffer_size", type=int, default=65536)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--pgd_steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_controller", type=float, default=1e-4)
    ap.add_argument("--l1_reg", type=float, default=1e-3)
    ap.add_argument("--train_V", action="store_true", help="also train V (default: V frozen)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.FileHandler(out / "train.log", mode="w"), logging.StreamHandler()])
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    train_utils.set_seed(args.seed)

    dyn, ctrl, V, lim = quad.load_quad(args.init)
    for s in list(ctrl.phi.w_scores) + list(ctrl.phi.b_scores):
        s.requires_grad_(False)  # masks frozen; faults are defined relative to them
    if not args.train_V:
        for p in V.parameters():
            p.requires_grad_(False)
    dyn.to(DEV); ctrl.to(DEV); V.to(DEV)
    lo, up = (-lim).to(DEV), lim.to(DEV)
    loss = FaultAwareLoss([dyn], ctrl, V, lo, up, rho_multiplier=1.0, kappa=args.kappa, error_coords=False,
                          include_faults=not args.nominal)
    # PGD needs gradients on all samples_per_iter states: raise the no-grad chunking threshold
    loss.max_grad_batch = max(loss.max_grad_batch, args.samples_per_iter)
    rho_t = torch.tensor(args.rho, device=DEV)
    loss.get_rho = lambda: rho_t
    loss.x_boundary = torch.zeros(1, 6, device=DEV)  # unused (rho fixed) but read by the loop
    t0 = time.time()
    train_utils.train_lyapunov_with_buffer(
        derivative_lyaloss=loss, positivity_lyaloss=None, observer_loss=None,
        lower_limit=lo, upper_limit=up, grid_size=torch.tensor([10] * 6, device=DEV),
        learning_rate=args.lr, lr_controller=args.lr_controller, weight_decay=0.0, max_iter=args.max_iter,
        enable_wandb=False, derivative_ibp_ratio=0, derivative_sample_ratio=1,
        positivity_ibp_ratio=0, positivity_sample_ratio=0, save_best_model=None,
        pgd_steps=args.pgd_steps, buffer_size=args.buffer_size, batch_size=args.batch_size,
        epochs=args.epochs, samples_per_iter=args.samples_per_iter, l1_reg=args.l1_reg,
        num_samples_per_boundary=500, V_decrease_within_roa=True,
        Vmin_x_boundary_weight=0, Vmax_x_boundary_weight=0,
        candidate_roa_states=None, candidate_roa_states_weight=0, logger=log)
    # PGD check of all variants with kappa_check = 0 (the verified condition)
    chk = FaultAwareLoss([dyn], ctrl, V, lo, up, 1.0, 0.0, False, include_faults=True)
    chk.get_rho = lambda: rho_t
    chk.x_boundary = loss.x_boundary
    x0 = lo + torch.rand(4096, 6, device=DEV) * (up - lo)
    adv = train_utils.pgd_attack(x0, chk, eps=up, steps=300, lower_boundary=lo, upper_boundary=up,
                                 direction="minimize").detach()
    pf = chk.per_fault_violation(adv)[0]
    rec = dict(rho=args.rho, nominal_viol=float(pf[0]), broken=int((pf[1:] > 0).sum()), faults=int(pf.numel() - 1),
               t=round(time.time() - t0))
    log.info("DONE " + json.dumps(rec))
    torch.save({"ctrl": ctrl.state_dict(), "V": V.state_dict(), "rho": args.rho}, out / "ckpt.pth")
    (out / "summary.json").write_text(json.dumps([rec], indent=1))


if __name__ == "__main__":
    main()
