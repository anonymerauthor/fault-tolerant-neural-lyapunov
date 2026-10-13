"""The exported per-mode verification model must equal the training-time controller and dynamics.

Checks, for random masks and setpoints:
  1. controllers.NeuralNetworkController loaded from export_mode_state_dict(m, error_coords=True)
     with x_eq = 0, u_eq = u*_m  ==  MaskedController mode view in error coordinates.
  2. verify_factory.create_pendulum_setpoint(...) builds the same closed loop (z+).
  3. z = 0 is an equilibrium of every mode (u(0) = u*, z+(0) = 0).
Run: python tests/test_setpoint_export.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "Lyapunov_Stable_NN_Controllers"))
import neural_lyapunov_training.controllers as controllers  # noqa: E402
from masked_controller import MaskedController  # noqa: E402
from setpoint import ErrorCoordDynamics, setpoint_equilibrium  # noqa: E402
import verify_factory  # noqa: E402


def main():
    torch.manual_seed(0)
    setpoints = [0.0, 0.5, 1.0]
    eqs = [setpoint_equilibrium(t) for t in setpoints]
    ctrl = MaskedController(2, 1, 8, 4, 3, u_lo=[-6.0], u_up=[6.0], x_eq=[e[0] for e in eqs],
                            u_eq=[e[1] for e in eqs], score_scale=10.0, delta_scores=True)
    with torch.no_grad():  # random but distinct masks
        for s in list(ctrl.phi.w_scores) + list(ctrl.phi.b_scores):
            s.copy_(0.1 * torch.randn_like(s))
    z = (torch.rand(5000, 2) - 0.5) * 24
    worst = 0.0
    for m, t in enumerate(setpoints):
        x_eq, u_eq = eqs[m]
        ref = ctrl.mode_module(m, error_coords=True)(z)
        nn_ctrl = controllers.NeuralNetworkController(
            nlayer=4, in_dim=2, out_dim=1, hidden_dim=8, clip_output="clamp",
            u_lo=torch.tensor([-6.0]), u_up=torch.tensor([6.0]),
            x_equilibrium=torch.zeros(2), u_equilibrium=torch.tensor(u_eq))
        nn_ctrl.load_state_dict(ctrl.export_mode_state_dict(m, error_coords=True))
        out = nn_ctrl(z)
        d1 = (out - ref).abs().max().item()
        dyn = ErrorCoordDynamics(t)
        loss = verify_factory.create_pendulum_setpoint(
            t, controller_parameters={"nlayer": 4, "hidden_dim": 8, "clip_output": "clamp",
                                      "u_lo": torch.tensor([-6.0]), "u_up": torch.tensor([6.0])},
            lyapunov_parameters={"hidden_widths": [16, 16, 8], "R_rows": 3, "absolute_output": True,
                                 "eps": 0.01, "activation": torch.nn.LeakyReLU, "V_psd_form": "L1"},
            loss_parameters={"kappa": 0.005}, loss_func="lyapunov.LyapunovDerivativeSimpleLossWithVBox")
        loss.controller.load_state_dict(nn_ctrl.state_dict())
        d2 = (loss.dynamics(z, loss.controller(z)) - dyn(z, ref)).abs().max().item()
        z0 = torch.zeros(1, 2)
        u0 = ctrl.mode_module(m, error_coords=True)(z0)
        d3 = max((u0 - torch.tensor(u_eq)).abs().max().item(), dyn(z0, u0).abs().max().item())
        print(f"mode {m} (theta*={t}): |export-ref|={d1:.2e}  |verify loop - train loop|={d2:.2e}  "
              f"equilibrium residual={d3:.2e}")
        worst = max(worst, d1, d2, d3)
    print("PASS" if worst < 1e-4 else "FAIL", f"(worst {worst:.2e})")
    return worst < 1e-4


if __name__ == "__main__":
    assert main()
