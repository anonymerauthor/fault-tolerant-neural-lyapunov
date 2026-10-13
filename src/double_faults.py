"""Grid falsification of all DOUBLE sign faults (pairs of flipped mask bits) of a pendulum controller.

For every mode and every pair (i, j), i < j, of mask bits, the faulty controller is evaluated with
the un-hinged certificate margin (fault_aware.FaultAwareLoss.raw_margin, but per variant) on a
multi-resolution grid of the box restricted to {V <= rho} (the verifier's small hole around 0 is
excluded). A pair is "broken" if some grid point has a negative margin. This is falsification only:
a broken pair has a concrete counterexample; an unbroken pair is not certified.

--singles evaluates the single faults instead (sanity check: a model whose single faults were all
formally verified must give zero broken singles here).

Usage: python src/double_faults.py --run runs/f2c_kappa --rho 0.165 --out results/double_f2c.json
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
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
from masked_mlp import bit_sensitivities  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def grid_points(lo, up, n, fine):
    out = []
    for a, b in ((lo, up), (torch.clamp(-fine * torch.ones_like(lo), lo, up),
                            torch.clamp(fine * torch.ones_like(up), lo, up))):
        t = [torch.linspace(float(a[i]), float(b[i]), n, device=DEV) for i in range(2)]
        out.append(torch.stack(torch.meshgrid(*t, indexing="ij"), -1).reshape(-1, 2))
    return torch.cat(out)


def flip_stacks(mask, pairs):
    """mask: per-layer tensors (weights then biases); pairs: (F, k) flat bit indices.
    Returns per-layer stacks (F, *shape) with the listed bits negated."""
    sizes = [m.numel() for m in mask]
    flat = torch.cat([m.reshape(-1) for m in mask]).unsqueeze(0).repeat(pairs.shape[0], 1)
    rows = torch.arange(pairs.shape[0], device=pairs.device).unsqueeze(1).expand_as(pairs)
    flat[rows, pairs] *= -1
    out, off = [], 0
    for m, s in zip(mask, sizes):
        out.append(flat[:, off:off + s].reshape(-1, *m.shape))
        off += s
    return out


@torch.no_grad()
def per_variant_margin(loss, m, dyn, z, Vz):
    """(F,) min over z of min((1-k)V(z) - V(z+), box margin of z+)."""
    u = loss._controller_all(z, m)
    F, B = u.shape[0], z.shape[0]
    zn = dyn(z.unsqueeze(0).expand(F, -1, -1).reshape(F * B, -1), u.reshape(F * B, -1))
    Vn = loss.lyapunov(zn).view(F, B)
    dV = (1 - loss.kappa) * Vz.view(1, B) - Vn
    zn = zn.view(F, B, -1)
    box = torch.minimum((zn - loss.box_lo).min(-1).values, (loss.box_up - zn).min(-1).values)
    return torch.minimum(dV, box).min(dim=1).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--kappa", type=float, default=0.005)
    ap.add_argument("--grid", type=int, default=300)
    ap.add_argument("--fine", type=float, default=1.0)
    ap.add_argument("--fchunk", type=int, default=512)
    ap.add_argument("--zchunk", type=int, default=4096)
    ap.add_argument("--singles", action="store_true")
    ap.add_argument("--limit_pairs", type=int, default=0, help="smoke test: only the first k pairs per mode")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a, ck, systems, ctrl, V, lim, ec = FS.load_run(ROOT / args.run)
    ctrl.to(DEV); V.to(DEV)
    for s in systems:
        s.to(DEV)
    lo, up = (-lim).to(DEV), lim.to(DEV)
    loss = FaultAwareLoss(systems, ctrl, V, lo, up, 1.0, args.kappa, ec, include_faults=False)
    Z = grid_points(lo, up, args.grid, args.fine)
    Vz_all = V(Z).squeeze(-1)
    keep = (Vz_all <= args.rho) & (Z.norm(dim=1) > 1e-3)
    Z, Vz_all = Z[keep], Vz_all[keep]
    sens = None
    try:
        sens = [float(x) for x in bit_sensitivities(ctrl.phi, float(lim.norm()))]
    except Exception:
        pass

    t0 = time.time()
    res = {"run": args.run, "rho": args.rho, "kappa": args.kappa, "grid": args.grid, "n_points": int(Z.shape[0]),
           "singles": args.singles, "modes": []}
    for m, dyn in enumerate(systems):
        mw, mb = ctrl.hard_masks(m)
        mask = [t.to(DEV) for t in mw + mb]
        N = sum(t.numel() for t in mask)
        if args.singles:
            pairs = torch.arange(N, device=DEV).unsqueeze(1)
        else:
            pairs = torch.combinations(torch.arange(N, device=DEV), 2)
        if args.limit_pairs:
            pairs = pairs[: args.limit_pairs]
        worst = torch.full((pairs.shape[0],), float("inf"), device=DEV)
        for f0 in range(0, pairs.shape[0], args.fchunk):
            p = pairs[f0:f0 + args.fchunk]
            st = flip_stacks(mask, p)
            loss.stacks = [(st[:len(mw)], st[len(mw):])] * len(systems)
            w = worst[f0:f0 + args.fchunk]
            for zc, vc in zip(Z.split(args.zchunk), Vz_all.split(args.zchunk)):
                w.copy_(torch.minimum(w, per_variant_margin(loss, m, dyn, zc, vc)))
        broken = (worst < 0).nonzero().squeeze(-1)
        rec = {"mode": m, "N": N, "variants": int(pairs.shape[0]), "broken": int(broken.numel()),
               "broken_pairs": pairs[broken].tolist(), "worst": float(worst.min())}
        res["modes"].append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "broken_pairs"}), f"t={time.time() - t0:.0f}s", flush=True)
    res["sensitivity"] = sens
    res["time"] = time.time() - t0
    Path(ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_text(json.dumps(res))
    tot = sum(r["broken"] for r in res["modes"]); var = sum(r["variants"] for r in res["modes"])
    print(f"TOTAL broken {tot}/{var} ({100 * tot / var:.2f}%) points={res['n_points']} time={res['time']:.0f}s")


if __name__ == "__main__":
    main()
