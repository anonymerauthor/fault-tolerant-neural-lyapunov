"""Trade-off between fault coverage and certified area (results/tradeoff.json -> paper/figs/fig_tradeoff.pdf).

JSON format: {"<series>": {"n_faults": int, "points": [{"label": str, "broken": int, "area": float}, ...]}}
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "font.family": "serif", "pdf.fonttype": 42})
COL = {"seed 1, width 8": "#1b6ca8", "width 4": "#d1495b", "seed 0, width 8": "#2a9d8f"}
MARK = {"nominal": "s", "w=0": "o", "w=0.1": "o", "w=0.3": "o", "w=1": "o"}


def main():
    data = json.loads((ROOT / "results" / "tradeoff.json").read_text())
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for name, ser in data.items():
        n = ser["n_faults"]
        pts = ser["points"]
        trained = [p for p in pts if p["label"] != "nominal"]
        trained.sort(key=lambda p: float(p["label"].split("=")[1]))
        xs = [100 * p["broken"] / n for p in trained]
        ys = [p["area"] for p in trained]
        c = COL.get(name, "k")
        ax.plot(xs, ys, "-o", color=c, ms=3.5, lw=0.9, label=f"{name}: fault-aware")
        for i, (p, x, y) in enumerate(zip(trained, xs, ys)):
            # place the label above the point unless a nearby point of the series lies higher
            near = [yy for j, (xx, yy) in enumerate(zip(xs, ys)) if j != i and abs(xx - x) < 1.5 and abs(yy - y) < 6]
            dy = -8 if any(yy > y for yy in near) else 3
            ax.annotate(p["label"], (x, y), textcoords="offset points", xytext=(4, dy), fontsize=6, color=c)
        for p in pts:
            if p["label"] == "nominal":
                ax.plot(100 * p["broken"] / n, p["area"], "s", color=c, mfc="none", ms=5,
                        label=f"{name}: nominal")
    ax.set_xlabel("single sign faults breaking the certificate (%)")
    ax.set_ylabel(r"area of $\Omega_\rho$")
    ax.set_ylim(-3, 80)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    out = ROOT / "paper" / "figs" / "fig_tradeoff.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=170)
    print("written", out)


if __name__ == "__main__":
    main()
