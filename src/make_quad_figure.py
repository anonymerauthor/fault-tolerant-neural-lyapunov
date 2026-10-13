"""Quadrotor figure (CPU only): V(x)/rho along closed-loop trajectories that start near the boundary of
{V <= rho}, (a) nominal fine-tuned controller under its most damaging single sign fault, (b) the
fault-aware controller under the same fault, (c) the fault-aware controller under its worst remaining fault.

Usage: python src/make_quad_figure.py
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import quad  # noqa: E402
import fault_screen as FS  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "font.family": "serif", "pdf.fonttype": 42})
RHO = 0.5


def screen_table(name):
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / f"results/quadscreen_{name}_rho0.5_shard*.csv"))])
    return d[d.bit != "nominal"].drop_duplicates("bit").set_index("bit")


def paper_name(name):
    """code name w0[4,5] (0-based layer and indices) -> paper name W_1[5,6] (1-based)."""
    k, l, idx = name[0], int(name[1]), [int(i) + 1 for i in name[3:-1].split(",")]
    return f"${k.upper() if k == 'w' else k}_{l + 1}[{','.join(map(str, idx))}]$"


def bit_of(ctrl, name):
    return next(b for b in FS.all_bits(ctrl) if f"{b[0]}{b[1]}[{','.join(map(str, b[2]))}]" == name)


@torch.no_grad()
def rollout(ckpt, bit_name, x0, steps):
    dyn, ctrl, V, lim = quad.load_quad(ckpt)
    mask = FS.flip(ctrl.hard_masks(0), bit_of(ctrl, bit_name))
    c = FS.FixedMaskController(ctrl, mask, ctrl.eq(0)[0], ctrl.eq(0)[1], False)
    x, Vs = x0.clone(), []
    for _ in range(steps):
        Vs.append(V(x).squeeze(-1))
        x = dyn(x, c(x))
    return torch.stack(Vs, 1).numpy(), V


def main(n_init=16, steps=300, seed=0):
    nom, fa2 = screen_table("quad_nominal"), screen_table("quad_fa2")
    fixed = nom[(nom.pgd_max_viol > 0) & (fa2.reindex(nom.index).pgd_max_viol == 0)]
    worst_nom = fixed.pgd_max_viol.idxmax()
    worst_fa2 = fa2.pgd_max_viol.idxmax()
    ck_nom, ck_fa2 = str(ROOT / "runs/quad_nominal/ckpt.pth"), str(ROOT / "runs/quad_fa2/ckpt.pth")
    _, _, V, lim = quad.load_quad(ck_fa2)
    g = torch.Generator().manual_seed(seed)
    X = (torch.rand(400000, 6, generator=g) * 2 - 1) * lim
    with torch.no_grad():
        v = V(X).squeeze(-1)
    X0 = X[(v <= RHO) & (v >= 0.8 * RHO)][:n_init]  # V is shared by both controllers (frozen)
    panels = [(ck_nom, worst_nom, "(a) nominal, " + paper_name(worst_nom)),
              (ck_fa2, worst_nom, "(b) fault-aware, " + paper_name(worst_nom)),
              (ck_fa2, worst_fa2, "(c) fault-aware, " + paper_name(worst_fa2))]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.9), sharey=True)
    info = {"worst_nominal_fixed": worst_nom, "worst_fa2": worst_fa2, "n_init": int(X0.shape[0]), "max_V_over_rho": []}
    for ax, (ck, b, title) in zip(axs, panels):
        Vs, _ = rollout(ck, b, X0, steps)
        t = np.arange(steps) * 0.01
        for row in Vs:
            ax.semilogy(t, np.maximum(row / RHO, 1e-6), lw=0.6, color="#1b6ca8")
        ax.axhline(1.0, color="k", ls=":", lw=0.7)
        ax.set_ylim(1e-3, 30)
        ax.set_xlabel("time (s)")
        ax.set_title(title, loc="left", fontsize=7)
        info["max_V_over_rho"].append(float(Vs.max() / RHO))
    axs[0].set_ylabel(r"$V(x)/\rho$")
    fig.tight_layout()
    out = ROOT / "paper/figs/fig_quad.pdf"
    fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"), dpi=170)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
