"""Numerical check of Theorem 1 / Corollary 1.

Soundness: bound >= empirical max jump (sampling + gradient ascent) for every trial.
Tightness: report empirical / bound ratios.
Run: python tests/test_jump_bound.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from masked_mlp import MaskedMLP, jump_bound  # noqa: E402


def empirical_max_jump(net, mi, mj, r0, d_in, n_samples=20000, n_ascent=200):
    x = torch.randn(n_samples, d_in)
    x = r0 * x / x.norm(dim=1, keepdim=True) * torch.rand(n_samples, 1) ** (1 / d_in)
    with torch.no_grad():
        j = (net.forward_with_masks(x, *mj) - net.forward_with_masks(x, *mi)).norm(dim=1)
    best = j.max().item()
    # projected gradient ascent from the 64 best samples
    x = x[j.topk(64).indices].clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.05 * r0)
    for _ in range(n_ascent):
        jump = (net.forward_with_masks(x, *mj) - net.forward_with_masks(x, *mi)).norm(dim=1)
        opt.zero_grad()
        (-jump.sum()).backward()
        opt.step()
        with torch.no_grad():
            n = x.norm(dim=1, keepdim=True).clamp_min(r0)
            x.mul_(r0 / n)
            jump = (net.forward_with_masks(x, *mj) - net.forward_with_masks(x, *mi)).norm(dim=1)
            best = max(best, jump.max().item())
    return best


def run(trials=30, seed=42):
    torch.manual_seed(seed)
    rows = []
    for t in range(trials):
        depth = [1, 2, 3][t % 3]
        width = [8, 16, 32][(t // 3) % 3]
        act = [nn.Tanh, nn.ReLU][t % 2]
        dims = [2] + [width] * depth + [1]
        net = MaskedMLP(dims, n_modes=2, activation=act)
        net.requires_grad_(False)
        mi = tuple([m.detach() for m in ms] for ms in net.masks(0))
        # mode j: flip a random fraction of mode-i bits (Hamming distance control)
        frac = [0.01, 0.05, 0.2, 0.5][t % 4]
        mj = ([torch.where(torch.rand_like(m) < frac, -m, m) for m in mi[0]],
              [torch.where(torch.rand_like(m) < frac, -m, m) for m in mi[1]])
        mj[0][0][0, 0] = -mi[0][0][0, 0]  # guarantee at least one flipped bit
        r0 = 3.0
        emp = empirical_max_jump(net, mi, mj, r0, d_in=2)
        b = jump_bound(net, mi, mj, r0)
        n_flip = sum(int((a != c).sum()) for a, c in zip(mi[0] + mi[1], mj[0] + mj[1]))
        rows.append((depth, width, act.__name__, frac, n_flip, emp, b["theorem1"], b["corollary1"]))
    print(f"{'L':>2} {'w':>3} {'act':>5} {'frac':>5} {'#flip':>6} {'empirical':>10} {'Thm1':>10} {'Cor1':>10} {'emp/Thm1':>9} {'emp/Cor1':>9}")
    ok = True
    for d, w, a, f, n, e, t1, c1 in rows:
        ok &= (e <= t1 + 1e-5) and (t1 <= c1 + 1e-5)
        print(f"{d:>2} {w:>3} {a:>5} {f:>5.2f} {n:>6} {e:>10.4f} {t1:>10.4f} {c1:>10.4f} {e/t1:>9.3f} {e/c1:>9.3f}")
    print("SOUND (empirical <= Thm1 <= Cor1 in all trials):", ok)
    return ok


if __name__ == "__main__":
    assert run()
