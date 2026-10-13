"""Figures for the paper (CPU only).

fig_regions.pdf     certified funnels {x : V(x - x*_m) <= rho} for the three set-points and a
                    switched trajectory 0 -> 0.5 -> 1.0 rad (fault-aware controller)
fig_faultmap.pdf    (a) fraction of single sign-bit faults that break the certificate, per layer,
                    nominal vs fault-aware training; (b) CROWN jump bound vs PGD violation (nominal)
fig_faulttraj.pdf   V along closed-loop trajectories under the most damaging fault of the nominal
                    model, nominal vs fault-aware controller
Usage: python src/make_figures.py --nominal runs/e2b_masked:0.136 --robust runs/f2c_kappa:0.165
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
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
import fault_screen as FS  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "font.family": "serif", "pdf.fonttype": 42})
COL = ["#1b6ca8", "#d1495b", "#2a9d8f"]


def load(spec):
    run, rho = spec.split(":")
    a, ck, systems, ctrl, V, lim, ec = FS.load_run(ROOT / run)
    ctrl.eval(); V.eval()
    return dict(a=a, ck=ck, systems=systems, ctrl=ctrl, V=V, lim=lim, rho=float(rho), run=ROOT / run)


@torch.no_grad()
def simulate(M, schedule, x0, steps, fault=None):
    """schedule: list of (mode, n_steps); x in absolute coordinates. fault = (mode, bit) or None."""
    ctrl, V = M["ctrl"], M["V"]
    x = torch.tensor(x0, dtype=torch.float32).view(1, 2)
    xs, Vs, ms = [x.clone()], [], []
    for mode, n in schedule:
        x_eq, u_eq = ctrl.eq(mode)
        dyn = M["systems"][mode]
        mask = ctrl.hard_masks(mode)
        if fault is not None and fault[0] == mode:
            mask = FS.flip(mask, fault[1])
        c = FS.FixedMaskController(ctrl, mask, x_eq, u_eq, True)
        for _ in range(n):
            z = x - x_eq
            Vs.append(V(z).item()); ms.append(mode)
            x = dyn(z, c(z)) + x_eq
            xs.append(x.clone())
    return torch.cat(xs).numpy(), np.array(Vs), np.array(ms)


def fig_regions(R, out):
    ctrl, V, rho = R["ctrl"], R["V"], R["rho"]
    th = np.linspace(-4.0, 5.0, 500); thd = np.linspace(-12, 12, 500)
    TH, THD = np.meshgrid(th, thd)
    X = torch.tensor(np.stack([TH.ravel(), THD.ravel()], 1), dtype=torch.float32)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for m, t in enumerate(R["a"]["setpoints"]):
        x_eq = ctrl.eq(m)[0]
        with torch.no_grad():
            Vm = V(X - x_eq).numpy().reshape(TH.shape)
        ax.contour(TH, THD, Vm, levels=[rho], colors=[COL[m]], linewidths=1.0)
        ax.plot([t], [0], "o", color=COL[m], ms=3)
        ax.plot([], [], color=COL[m], lw=1, label=rf"$\theta^*={t:g}$ rad")
    xs, _, ms = simulate(R, [(0, 60), (1, 60), (2, 80)], [-0.4, 1.5], 0)
    ax.plot(xs[:, 0], xs[:, 1], "k-", lw=0.7, label="switched trajectory")
    ax.plot(xs[0, 0], xs[0, 1], "kx", ms=4)
    ax.set_xlabel(r"$\theta$ (rad)"); ax.set_ylabel(r"$\dot\theta$ (rad/s)")
    ax.set_xlim(-6.6, 5.0); ax.legend(loc="lower left", frameon=False, ncol=1, handlelength=1.5)
    fig.tight_layout(); fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"), dpi=170); plt.close(fig)


def layer_of(bit):
    return bit[:2]


def fig_faultmap(Nom, Rob, out):
    dn = pd.read_csv(sorted(Nom["run"].glob("faults_*.csv"))[0]); dn = dn[dn.bit != "nominal"]
    dr = pd.read_csv(sorted(Rob["run"].glob("faults_*.csv"))[0]); dr = dr[dr.bit != "nominal"]
    order = ["w0", "b0", "w1", "b1", "w2", "b2", "w3", "b3"]
    fn = dn.assign(L=dn.bit.map(layer_of)).groupby("L").apply(lambda g: (g.pgd_max_viol > 0).mean())
    fr = dr.assign(L=dr.bit.map(layer_of)).groupby("L").apply(lambda g: (g.pgd_max_viol > 0).mean())
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.1))
    idx = np.arange(len(order))
    axs[0].bar(idx - 0.2, [fn.get(o, 0) * 100 for o in order], 0.4, color=COL[1], label="nominal training")
    axs[0].bar(idx + 0.2, [fr.get(o, 0) * 100 for o in order], 0.4, color=COL[0], label="fault-aware training")
    axs[0].set_xticks(idx); axs[0].set_xticklabels([rf"${o[0].upper() if o[0]=='w' else 'b'}_{int(o[1])+1}$" for o in order])
    axs[0].set_ylabel("faults breaking the\ncertificate (%)"); axs[0].legend(frameon=False)
    axs[0].set_title("(a)", loc="left")
    if max(fr.get(o, 0) for o in order) == 0:
        axs[0].text(0.52, 0.55, "fault-aware: 0 in every layer", transform=axs[0].transAxes, ha="center",
                    color=COL[0], fontsize=7)
    broken = dn.pgd_max_viol > 0
    axs[1].scatter(dn.crown_jump[~broken], dn.s_k[~broken], s=5, c=COL[0], label="certificate kept")
    axs[1].scatter(dn.crown_jump[broken], dn.s_k[broken], s=7, c=COL[1], marker="^", label="certificate broken")
    axs[1].set_xscale("log"); axs[1].set_yscale("log")
    axs[1].set_xlabel("CROWN bound on control jump"); axs[1].set_ylabel(r"analytic sensitivity $s_j$")
    axs[1].legend(frameon=False); axs[1].set_title("(b)", loc="left")
    fig.tight_layout(); fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"), dpi=170); plt.close(fig)


def fig_faulttraj(Nom, Rob, out, n_init=12, steps=300, seed=0):
    dn = pd.read_csv(sorted(Nom["run"].glob("faults_*.csv"))[0]); dn = dn[dn.bit != "nominal"]
    worst = dn.sort_values("pgd_max_viol", ascending=False).iloc[0]
    mode = int(worst["mode"])
    bit_name = worst["bit"]
    bit = next(b for b in FS.all_bits(Nom["ctrl"]) if f"{b[0]}{b[1]}[{','.join(map(str, b[2]))}]" == bit_name)
    rng = np.random.default_rng(seed)
    x_eq = Nom["ctrl"].eq(mode)[0].numpy()
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.0), sharey=True)
    for ax, M, title in ((axs[0], Nom, "(a) nominal training"), (axs[1], Rob, "(b) fault-aware training")):
        # initial states sampled inside the certified set of this mode
        th = np.linspace(-12, 12, 300)
        Z = torch.tensor(np.stack(np.meshgrid(th, th), -1).reshape(-1, 2), dtype=torch.float32)
        with torch.no_grad():
            inside = Z[(M["V"](Z).squeeze(-1) <= M["rho"])].numpy()
        pick = inside[rng.choice(len(inside), n_init, replace=False)]
        for z0 in pick:
            _, Vs, _ = simulate(M, [(mode, steps)], z0 + x_eq, 0, fault=(mode, bit))
            ax.semilogy(np.arange(len(Vs)) * M["a"]["dt"], np.maximum(Vs, 1e-8), lw=0.7, color=COL[0])
        ax.axhline(M["rho"], color="k", ls=":", lw=0.7)
        ax.set_ylim(1e-6, 2.0)
        ax.set_xlabel("time (s)"); ax.set_title(title, loc="left")
    axs[0].set_ylabel(r"$V(x-x^*)$")
    fig.tight_layout(); fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"), dpi=170); plt.close(fig)
    return dict(mode=mode, bit=bit_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominal", required=True)
    ap.add_argument("--robust", required=True)
    ap.add_argument("--out", default="paper/figs")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    Nom, Rob = load(args.nominal), load(args.robust)
    fig_regions(Rob, out / "fig_regions.pdf")
    fig_faultmap(Nom, Rob, out / "fig_faultmap.pdf")
    info = fig_faulttraj(Nom, Rob, out / "fig_faulttraj.pdf")
    print("figures written to", out, "| trajectory fault:", info)


if __name__ == "__main__":
    main()
