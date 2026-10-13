"""Mask-hypercube transition paths between two modes of a MaskedController.

A path flips the bits in F = {k : m_i[k] != m_j[k]} one at a time (or in groups).
For every step we report
  * the analytic per-bit bound s_k (Corollary 1, mask-invariant),
  * the CROWN-certified jump sup_x |u_{p_{t+1}}(x) - u_{p_t}(x)| over the box.

Observation used by the orderings below: with the analytic bound, step t costs s_{k_t}
independent of the current mask, so the bottleneck max_k s_k is the same for every
single-flip order; the order only matters for the (context-dependent) certified jumps.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from masked_mlp import bit_sensitivities  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"

Mask = tuple[list[torch.Tensor], list[torch.Tensor]]


def flip_set(mi: Mask, mj: Mask) -> list[tuple[str, int, tuple]]:
    """List of differing positions as (kind, layer, index)."""
    out = []
    for kind, (Li, Lj) in (("w", (mi[0], mj[0])), ("b", (mi[1], mj[1]))):
        for l, (a, b) in enumerate(zip(Li, Lj)):
            for idx in (a != b).nonzero().tolist():
                out.append((kind, l, tuple(idx)))
    return out


def apply_flip(m: Mask, flip) -> Mask:
    kind, l, idx = flip
    w = [t.clone() for t in m[0]]
    b = [t.clone() for t in m[1]]
    tgt = w if kind == "w" else b
    tgt[l][idx] = -tgt[l][idx]
    return w, b


def sensitivity_of(ctrl, flip, r0) -> float:
    sens = bit_sensitivities(ctrl.phi, r0)
    kind, l, idx = flip
    return float(sens[l][0 if kind == "w" else 1][idx])


class _PlainCtrl(nn.Module):
    """u(x) = clamp(phi(x) - phi(0)) for a fixed mask, written with plain layers (for auto_LiRPA)."""

    def __init__(self, ctrl, mask: Mask):
        super().__init__()
        layers = []
        for l in range(ctrl.phi.n_layers):
            lin = nn.Linear(ctrl.phi.dims[l], ctrl.phi.dims[l + 1])
            lin.weight.data = (mask[0][l] * ctrl.phi.weights[l]).detach().clone()
            lin.bias.data = (mask[1][l] * ctrl.phi.biases[l]).detach().clone()
            layers.append(lin)
            if l < ctrl.phi.n_layers - 1:
                layers.append(ctrl.activation())
        self.net = nn.Sequential(*layers)
        with torch.no_grad():  # phi(x*) - u* is a constant for a fixed mask
            self.register_buffer("offset", (self.net(ctrl.x_eq.unsqueeze(0)) - ctrl.u_eq).squeeze(0))
        self.register_buffer("u_lo", ctrl.u_lo.detach().clone())
        self.register_buffer("u_up", ctrl.u_up.detach().clone())

    def forward(self, x):
        raw = self.net(x) - self.offset
        f1 = torch.relu(raw - self.u_lo) + self.u_lo
        return -(torch.relu(self.u_up - f1) - self.u_up)


class _Diff(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = a, b

    def forward(self, x):
        return self.b(x) - self.a(x)


@torch.no_grad()
def certified_jump(ctrl, m_from: Mask, m_to: Mask, lo, up, splits=32) -> float:
    """CROWN upper bound on max_x |u_to(x) - u_from(x)| over the box, via uniform input splitting (2-D)."""
    diff = _Diff(_PlainCtrl(ctrl, m_from), _PlainCtrl(ctrl, m_to)).to(DEV).eval()
    d = len(lo)
    g = [torch.linspace(float(lo[i]), float(up[i]), splits + 1) for i in range(d)]
    L = torch.stack(torch.meshgrid(*[t[:-1] for t in g], indexing="ij"), -1).reshape(-1, d).to(DEV)
    U = torch.stack(torch.meshgrid(*[t[1:] for t in g], indexing="ij"), -1).reshape(-1, d).to(DEV)
    model = BoundedModule(diff, torch.zeros(1, d, device=DEV), device=DEV, verbose=0)
    ptb = PerturbationLpNorm(norm=float("inf"), x_L=L, x_U=U)
    lb, ub = model.compute_bounds(x=(BoundedTensor((L + U) / 2, ptb),), method="CROWN")
    return torch.maximum(ub.abs(), lb.abs()).max().item()


def order_flips(ctrl, flips, strategy: str, r0: float, seed: int = 42, mi: Mask = None, lo=None, up=None):
    if strategy == "random":
        rnd = random.Random(seed)
        f = list(flips)
        rnd.shuffle(f)
        return f
    if strategy == "sens_asc":
        return sorted(flips, key=lambda k: sensitivity_of(ctrl, k, r0))
    if strategy == "sens_desc":
        return sorted(flips, key=lambda k: -sensitivity_of(ctrl, k, r0))
    if strategy == "greedy_crown":
        # at each step pick the remaining flip with the smallest certified jump from the current mask
        cur, rest, out = mi, list(flips), []
        while rest:
            costs = [certified_jump(ctrl, cur, apply_flip(cur, k), lo, up, splits=16) for k in rest]
            k = rest.pop(min(range(len(rest)), key=costs.__getitem__))
            out.append(k)
            cur = apply_flip(cur, k)
        return out
    raise ValueError(strategy)


def build_path(ctrl, mi: Mask, mj: Mask, strategy: str, lo, up, r0: float, splits=32):
    flips = flip_set(mi, mj)
    order = order_flips(ctrl, flips, strategy, r0, mi=mi, lo=lo, up=up)
    path, steps = [mi], []
    cur = mi
    for k in order:
        nxt = apply_flip(cur, k)
        steps.append(dict(flip=k, s_k=sensitivity_of(ctrl, k, r0),
                          crown_jump=certified_jump(ctrl, cur, nxt, lo, up, splits)))
        path.append(nxt)
        cur = nxt
    assert all(torch.equal(a, b) for a, b in zip(cur[0] + cur[1], mj[0] + mj[1]))
    return path, steps
