"""Mode-switched controller: shared weights + per-mode sign masks, with the
equilibrium normalization and relu-based clamp used by Yang et al. (2024):

    u_m(x) = clamp( phi_m(x) - phi_m(x*) + u*, u_lo, u_up ),   phi_m = MaskedMLP with mask m.

`export_mode` returns a plain `controllers.NeuralNetworkController` whose weights
are M_m * W, so that the existing alpha-beta-CROWN pipeline can verify each mode.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from masked_mlp import MaskedMLP


class MaskedController(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, nlayer, n_modes, u_lo, u_up,
                 x_eq, u_eq, activation=nn.ReLU, score_init="shared", score_scale: float = 1.0,
                 delta_scores: bool = False):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (nlayer - 1) + [out_dim]
        self.phi = MaskedMLP(dims, n_modes, activation=activation, score_scale=score_scale,
                             delta_scores=delta_scores)
        if score_init == "shared":
            # all modes start from the same all-(+1) mask -> small Hamming distances
            with torch.no_grad():
                for s in list(self.phi.w_scores) + list(self.phi.b_scores):
                    s.copy_(0.5 + 0.05 * torch.randn_like(s))
        self.nlayer, self.hidden_dim, self.activation = nlayer, hidden_dim, activation
        self.register_buffer("u_lo", torch.as_tensor(u_lo, dtype=torch.float32))
        self.register_buffer("u_up", torch.as_tensor(u_up, dtype=torch.float32))
        self.register_buffer("x_eq", torch.as_tensor(x_eq, dtype=torch.float32))
        self.register_buffer("u_eq", torch.as_tensor(u_eq, dtype=torch.float32))

    @property
    def n_modes(self):
        return self.phi.n_modes

    def eq(self, mode: int):
        """Equilibrium (x*, u*) of a mode; x_eq/u_eq may be shared (1-D) or per mode (2-D)."""
        x_eq = self.x_eq[mode] if self.x_eq.dim() == 2 else self.x_eq
        u_eq = self.u_eq[mode] if self.u_eq.dim() == 2 else self.u_eq
        return x_eq, u_eq

    def forward(self, x, mode: int):
        mw, mb = self.phi.masks(mode)
        x_eq, u_eq = self.eq(mode)
        raw = (self.phi.forward_with_masks(x, mw, mb)
               - self.phi.forward_with_masks(x_eq.unsqueeze(0), mw, mb) + u_eq)
        f1 = torch.relu(raw - self.u_lo) + self.u_lo
        return -(torch.relu(self.u_up - f1) - self.u_up)

    def mode_module(self, mode: int, error_coords: bool = False) -> nn.Module:
        return _ModeView(self, mode, error_coords)

    @torch.no_grad()
    def hard_masks(self, mode: int):
        mw, mb = self.phi.masks(mode)
        return [m.detach().clone() for m in mw], [m.detach().clone() for m in mb]

    @torch.no_grad()
    def export_mode_state_dict(self, mode: int, error_coords: bool = False) -> dict:
        """State dict in the layout of controllers.NeuralNetworkController (net.0, net.2, ...).

        error_coords=True: the exported network takes z = x - x*_m; the offset is folded into the
        first-layer bias, b1' = b1 + A1 x*_m, so that net(z) = phi_m(z + x*_m)."""
        mw, mb = self.hard_masks(mode)
        sd = {}
        for l in range(self.phi.n_layers):
            A = mw[l] * self.phi.weights[l]
            b = mb[l] * self.phi.biases[l]
            if error_coords and l == 0:
                b = b + A @ self.eq(mode)[0]
            sd[f"net.{2 * l}.weight"] = A.cpu().clone()
            sd[f"net.{2 * l}.bias"] = b.cpu().clone()
        return sd


class _ModeView(nn.Module):
    """Adapter so a single mode looks like a standard controller `u = c(x)`."""

    def __init__(self, parent: MaskedController, mode: int, error_coords: bool = False):
        super().__init__()
        self.parent = [parent]  # avoid registering parent parameters twice
        self.mode = mode
        self.error_coords = error_coords

    def parameters(self, recurse: bool = True):
        return self.parent[0].parameters(recurse)

    def forward(self, x):
        if self.error_coords:  # input is z = x - x*_mode
            x = x + self.parent[0].eq(self.mode)[0]
        return self.parent[0](x, self.mode)


@torch.no_grad()
def hamming(ctrl: MaskedController, i: int, j: int) -> int:
    (wi, bi), (wj, bj) = ctrl.hard_masks(i), ctrl.hard_masks(j)
    return sum(int((a != b).sum()) for a, b in zip(wi + bi, wj + bj))
