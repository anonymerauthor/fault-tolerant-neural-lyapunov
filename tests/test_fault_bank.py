"""FaultBankModel (verification graph) must agree with FaultAwareLoss (training graph).

For random z: dV_f = (1-k)V(z) - V(z+_f) and z+_f from the bank must match the batched
training computation for every variant f; V(z) must match.
Run: python tests/test_fault_bank.py runs/f2_faultaware
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fault_screen as FS  # noqa: E402
from fault_aware import FaultAwareLoss  # noqa: E402
from verify_faults import FaultBankModel  # noqa: E402


def main(run):
    a, ck, systems, ctrl, V, lim, ec = FS.load_run(ROOT / run)
    lo, up = -lim, lim
    fa = FaultAwareLoss(systems, ctrl, V, lo, up, 1.0, 0.005, ec)
    z = lo + torch.rand(1024, 2) * (up - lo)
    worst = 0.0
    for m, dyn in enumerate(systems):
        bank = FaultBankModel(str(ROOT / run), m, 0.005, lo.tolist(), up.tolist())
        F = bank.F
        with torch.no_grad():
            out = bank(z)
            u = fa._controller_all(z, m)
            zn = dyn(z.unsqueeze(0).expand(F, -1, -1).reshape(-1, 2), u.reshape(-1, 1)).view(F, -1, 2)
            Vz = V(z)
            dV = ((1 - 0.005) * Vz.unsqueeze(0) - V(zn.reshape(-1, 2)).view(F, -1, 1)).squeeze(-1).T
        e1 = (out[:, :F] - dV).abs().max().item()
        e2 = (out[:, F + 2:2 * F + 2] - (zn[:, :, 1].T - lo[1])).abs().max().item()
        e3 = (out[:, -1:] - Vz).abs().max().item()
        print(f"mode {m}: F={F}  |dV diff|={e1:.2e}  |thd margin diff|={e2:.2e}  |V diff|={e3:.2e}")
        worst = max(worst, e1, e2, e3)
    print("PASS" if worst < 1e-4 else "FAIL", f"(worst {worst:.2e})")
    return worst < 1e-4


if __name__ == "__main__":
    assert main(sys.argv[1])
