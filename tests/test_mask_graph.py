"""Smoke test of mask_graph on a trained checkpoint.

Run: python tests/test_mask_graph.py runs/pendulum_2mode_smoke/ckpt_scale0.2.pth
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from masked_controller import MaskedController  # noqa: E402
from mask_graph import build_path, certified_jump  # noqa: E402


def main(ckpt_path, strategies=("random", "sens_asc", "sens_desc")):
    run = Path(ckpt_path).parent
    a = json.loads((run / "args.json").read_text())
    ck = torch.load(ckpt_path, map_location="cpu")
    ctrl = MaskedController(2, 1, a["hidden"], a["nlayer"], len(a["masses"]),
                            u_lo=[-a["u_max"]], u_up=[a["u_max"]], x_eq=[0.0, 0.0], u_eq=[0.0])
    ctrl.load_state_dict(ck["ctrl"])
    ctrl.eval()
    lim = [a["limit"][0] * ck["scale"], a["limit"][1] * ck["scale"]]
    lo, up = [-lim[0], -lim[1]], lim
    r0 = (lim[0] ** 2 + lim[1] ** 2) ** 0.5
    mi, mj = ctrl.hard_masks(0), ctrl.hard_masks(1)
    direct = certified_jump(ctrl, mi, mj, lo, up)
    print(f"box={lim}, direct switch certified jump = {direct:.4f}")
    for s in strategies:
        _, steps = build_path(ctrl, mi, mj, s, lo, up, r0)
        cj = [st["crown_jump"] for st in steps]
        sk = [st["s_k"] for st in steps]
        print(f"{s:>12}: steps={len(steps)}  max step jump (CROWN)={max(cj):.4f}  "
              f"sum={sum(cj):.4f}  max s_k={max(sk):.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
