"""FaultAwareLoss (batched over all single-bit faults) must match per-fault evaluation.

Run: python tests/test_fault_aware.py runs/e2b_masked
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fault_screen as FS  # noqa: E402
from fault_aware import FaultAwareLoss  # noqa: E402


def main(run):
    a, ck, systems, ctrl, V, lim, ec = FS.load_run(ROOT / run)
    lo, up = -lim, lim
    loss = FaultAwareLoss(systems, ctrl, V, lo, up, rho_multiplier=1.0, kappa=0.005, error_coords=ec)
    loss.x_boundary = ck["x_boundary"]
    z = lo + torch.rand(2048, 2) * (up - lo)
    allb = FS.all_bits(ctrl)  # FaultAwareLoss order: all weight bits (layer-major), then all bias bits
    bits = [b for b in allb if b[0] == "w"] + [b for b in allb if b[0] == "b"]
    worst = 0.0
    for m, dyn in enumerate(systems):
        u_all = loss._controller_all(z, m)
        x_eq, u_eq = ctrl.eq(m)
        nominal = ctrl.hard_masks(m)
        for fi in [0] + random.Random(0).sample(range(1, len(bits) + 1), 25):
            mask = nominal if fi == 0 else FS.flip(nominal, bits[fi - 1])
            c = FS.FixedMaskController(ctrl, mask, x_eq, u_eq, ec)
            worst = max(worst, (c(z) - u_all[fi]).abs().max().item())
    print(f"max |batched - per-fault| control difference: {worst:.2e}")
    print("PASS" if worst < 1e-4 else "FAIL")
    return worst < 1e-4


if __name__ == "__main__":
    assert main(sys.argv[1])
