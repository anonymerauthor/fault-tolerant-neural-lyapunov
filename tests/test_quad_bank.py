"""QuadFaultBank (verification graph) must match the training-side plant/controller/V for every variant.

Run: python tests/test_quad_bank.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fault_screen as FS  # noqa: E402
import quad  # noqa: E402
from verify_faults import _variant_masks  # noqa: E402
from verify_faults_quad import QuadFaultBank  # noqa: E402


def main():
    torch.manual_seed(0)
    dyn, ctrl, V, lim = quad.load_quad()
    x = (torch.rand(2048, 6) * 2 - 1) * lim
    worst = 0.0
    for st, en in ((0, 15), (60, 75)):
        bank = QuadFaultBank(str(quad.YANG_CKPT), kappa=0.0, start=st, end=en)
        F = bank.F
        with torch.no_grad():
            out = bank(x)
            sw, sb = _variant_masks(ctrl, 0)
            x_eq, u_eq = ctrl.eq(0)
            for f in range(F):
                mask = ([s[st + f] for s in sw], [s[st + f] for s in sb])
                c = FS.FixedMaskController(ctrl, mask, x_eq, u_eq, False)
                xn = dyn(x, c(x))
                dV = (V(x) - V(xn)).squeeze(-1)
                e1 = (out[:, f] - dV).abs().max().item()
                low = torch.stack([out[:, F + k * F + f] for k in range(6)], 1)   # lower margins x+ - lo
                e2 = (low - (xn + lim)).abs().max().item()
                worst = max(worst, e1, e2)
            e3 = (out[:, -1] - V(x).squeeze(-1)).abs().max().item()
            worst = max(worst, e3)
        print(f"variants {st}-{en}: worst so far {worst:.2e}")
    print("PASS" if worst < 1e-4 else "FAIL", f"(worst {worst:.2e})")
    return worst < 1e-4


if __name__ == "__main__":
    assert main()
