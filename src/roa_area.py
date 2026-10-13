"""Area of the certified region {z in box : V(z) <= rho}, as a fraction of the box (grid estimate).

rho values from different runs are not comparable (different V); the area is.
Usage: python src/roa_area.py runs/e2a_shared:0.166 runs/e2b_masked:0.136 [--grid 1000]
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "Lyapunov_Stable_NN_Controllers"))
import neural_lyapunov_training.lyapunov as lyapunov  # noqa: E402


def area(run: Path, rho: float, grid: int) -> dict:
    a = json.loads((run / "args.json").read_text())
    ck = torch.load(run / f"ckpt_scale{a['limit_scales'][-1]:.1f}.pth", map_location="cpu")
    V = lyapunov.NeuralNetworkLyapunov(goal_state=torch.zeros(2), hidden_widths=[16, 16, 8], x_dim=2, R_rows=3,
                                       absolute_output=True, eps=a.get("V_eps", 0.01), activation=nn.LeakyReLU,
                                       V_psd_form="L1")
    V.load_state_dict(ck["V"])
    lim = [a["limit"][i] * a["limit_scales"][-1] for i in range(2)]
    t0, t1 = torch.linspace(-lim[0], lim[0], grid), torch.linspace(-lim[1], lim[1], grid)
    Z = torch.stack(torch.meshgrid(t0, t1, indexing="ij"), -1).reshape(-1, 2)
    with torch.no_grad():
        frac = (V(Z).squeeze(-1) <= rho).float().mean().item()
    return dict(run=run.name, rho=rho, box=lim, area_fraction=frac, area=frac * 4 * lim[0] * lim[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="+", help="run_dir:rho")
    ap.add_argument("--grid", type=int, default=1000)
    args = ap.parse_args()
    for it in args.items:
        run, rho = it.rsplit(":", 1)
        print(json.dumps(area(ROOT / run, float(rho), args.grid)))


if __name__ == "__main__":
    main()
