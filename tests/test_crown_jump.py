"""Certified jump bound via CROWN on the difference network pi^(j) - pi^(i).

Compares, on the box ||x||_inf <= a:
  empirical max |jump| (sampling + gradient ascent)  <=  CROWN (with uniform input splitting)
  <= Theorem 1 (r0 = a * sqrt(d)).
Run: python tests/test_crown_jump.py
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from masked_mlp import MaskedMLP, jump_bound  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class DiffNet(nn.Module):
    """Two masked copies of the same shared weights, written as plain Linear layers."""

    def __init__(self, net: MaskedMLP, mi, mj, act_cls):
        super().__init__()
        self.a = self._build(net, *mi, act_cls)
        self.b = self._build(net, *mj, act_cls)

    @staticmethod
    def _build(net, mw, mb, act_cls):
        layers = []
        for l in range(net.n_layers):
            lin = nn.Linear(net.dims[l], net.dims[l + 1])
            lin.weight.data = (mw[l] * net.weights[l]).detach().clone()
            lin.bias.data = (mb[l] * net.biases[l]).detach().clone()
            layers.append(lin)
            if l < net.n_layers - 1:
                layers.append(act_cls())
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.b(x) - self.a(x)


def crown_max_abs(diff: nn.Module, a: float, d: int, splits: int) -> float:
    """Upper bound on max |diff(x)| over [-a,a]^d via CROWN on a uniform grid of sub-boxes."""
    g = torch.linspace(-a, a, splits + 1)
    lo_1d, hi_1d = g[:-1], g[1:]
    grids_lo = torch.stack(torch.meshgrid(*[lo_1d] * d, indexing="ij"), -1).reshape(-1, d)
    grids_hi = torch.stack(torch.meshgrid(*[hi_1d] * d, indexing="ij"), -1).reshape(-1, d)
    center = ((grids_lo + grids_hi) / 2).to(DEV)
    model = BoundedModule(diff.to(DEV), torch.zeros(1, d, device=DEV), device=DEV, verbose=0)
    ptb = PerturbationLpNorm(norm=float("inf"), x_L=grids_lo.to(DEV), x_U=grids_hi.to(DEV))
    lb, ub = model.compute_bounds(x=(BoundedTensor(center, ptb),), method="CROWN")
    return torch.maximum(ub.abs(), lb.abs()).max().item()


def empirical_box(diff, a, d, n=50000, steps=200):
    x = (torch.rand(n, d) * 2 - 1) * a
    with torch.no_grad():
        best = diff(x).abs().max().item()
        x = x[diff(x).abs().squeeze(-1).topk(64).indices].clone()
    x.requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.02 * a)
    for _ in range(steps):
        loss = -diff(x).abs().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(-a, a)
            best = max(best, diff(x).abs().max().item())
    return best


def run(trials=12, seed=42, a=2.0):
    torch.manual_seed(seed)
    print(f"{'L':>2} {'w':>3} {'act':>5} {'#flip':>6} {'emp':>8} {'CROWN8':>8} {'CROWN32':>8} {'Thm1':>8} {'e/C32':>6} {'e/Thm1':>7} {'t(s)':>5}")
    ok = True
    for t in range(trials):
        depth = [1, 2, 3][t % 3]
        width = [8, 16, 32][(t // 3) % 3]
        act = [nn.Tanh, nn.ReLU][t % 2]
        net = MaskedMLP([2] + [width] * depth + [1], n_modes=1, activation=act).requires_grad_(False)
        mi = tuple([m.detach() for m in ms] for ms in net.masks(0))
        frac = [0.05, 0.2][t % 2]
        mj = ([torch.where(torch.rand_like(m) < frac, -m, m) for m in mi[0]],
              [torch.where(torch.rand_like(m) < frac, -m, m) for m in mi[1]])
        mj[0][0][0, 0] = -mi[0][0][0, 0]
        n_flip = sum(int((p != q).sum()) for p, q in zip(mi[0] + mi[1], mj[0] + mj[1]))
        diff = DiffNet(net, mi, mj, act).requires_grad_(False)
        emp = empirical_box(diff.cpu(), a, 2)
        t0 = time.time()
        c8 = crown_max_abs(diff, a, 2, 8)
        c32 = crown_max_abs(diff, a, 2, 32)
        dt = time.time() - t0
        thm = jump_bound(net, mi, mj, r0=a * 2 ** 0.5)["theorem1"]
        ok &= emp <= c32 + 1e-5 and c32 <= c8 + 1e-5
        print(f"{depth:>2} {width:>3} {act.__name__:>5} {n_flip:>6} {emp:>8.4f} {c8:>8.4f} {c32:>8.4f} {thm:>8.3f} {emp/c32:>6.3f} {emp/thm:>7.3f} {dt:>5.1f}")
        diff.cpu()
    print("SOUND (emp <= CROWN32 <= CROWN8):", ok)
    return ok


if __name__ == "__main__":
    assert run()
