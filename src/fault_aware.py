"""Fault-aware Lyapunov loss: the worst case over all modes AND all single sign-bit faults.

For mode m with hard mask M_m, the faulty masks are M_m with one bit flipped (every weight and
bias). All F = 1 + n_params variants (nominal + faults) are evaluated in ONE batched pass:
the masked weights are stacked into tensors of shape (F, out, in) and applied with einsum.

The returned value has the same meaning as Yang et al.'s LyapunovDerivativeLoss output
(>= 0 <=> the condition holds), minimised over modes and faults, so it can be plugged into
train_utils.train_lyapunov_with_buffer unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from masked_controller import MaskedController


def _fault_stack(mask: list[torch.Tensor]) -> list[torch.Tensor]:
    """For a list of per-layer mask tensors, return per-layer stacks with all single-bit flips.

    Output: list over layers of tensors (1 + n_bits, *shape); index 0 is the nominal mask.
    Flips of layer l only change slice l; other layers repeat the nominal mask.
    """
    sizes = [m.numel() for m in mask]
    n = 1 + sum(sizes)
    stacks, offset = [], 1
    for l, m in enumerate(mask):
        s = m.unsqueeze(0).repeat(n, *[1] * m.dim()).clone()
        flat = s.view(n, -1)
        idx = torch.arange(sizes[l], device=m.device)
        flat[offset + idx, idx] *= -1
        offset += sizes[l]
        stacks.append(s)
    return stacks


class FaultAwareLoss(nn.Module):
    def __init__(self, systems, ctrl: MaskedController, V, box_lo, box_up, rho_multiplier, kappa,
                 error_coords: bool, include_faults: bool = True, sample_k: int = 0):
        super().__init__()
        self.controller, self.lyapunov, self.systems = ctrl, V, systems
        self.kappa, self.rho_multiplier = kappa, rho_multiplier
        self.box_lo, self.box_up = box_lo, box_up
        self.error_coords = error_coords
        # sample_k > 0: baseline "random fault injection" (each forward uses the nominal mask and
        # k randomly drawn single-bit faults instead of all of them)
        self.sample_k = sample_k
        self.x_boundary = None
        self.stacks = []  # per mode: (weight mask stacks, bias mask stacks)
        with torch.no_grad():
            for m in range(len(systems)):
                mw, mb = ctrl.hard_masks(m)
                if include_faults:
                    # one flip index space over [weights..., biases...]: build both stacks consistently
                    allm = mw + mb
                    st = _fault_stack(allm)
                    self.stacks.append((st[:len(mw)], st[len(mw):]))
                else:
                    self.stacks.append(([w.unsqueeze(0) for w in mw], [b.unsqueeze(0) for b in mb]))

    def get_rho(self):
        return self.rho_multiplier * self.lyapunov(self.x_boundary).min()

    def _controller_all(self, z, m):
        """u for all masks of mode m: (F, B, 1)."""
        ctrl = self.controller
        sw, sb = self.stacks[m]
        if self.sample_k and self.training_sample:
            n = sw[0].shape[0]
            idx = torch.cat([torch.zeros(1, dtype=torch.long),
                             1 + torch.randperm(n - 1)[: self.sample_k]]).to(sw[0].device)
            sw, sb = [t[idx] for t in sw], [t[idx] for t in sb]
        x_eq, u_eq = ctrl.eq(m)
        x = z + x_eq if self.error_coords else z
        F = sw[0].shape[0]

        def phi(inp):  # inp (F, B, d)
            h = inp
            for l in range(ctrl.phi.n_layers):
                A = sw[l] * ctrl.phi.weights[l]          # (F, out, in)
                b = sb[l] * ctrl.phi.biases[l]           # (F, out)
                h = torch.einsum("foi,fbi->fbo", A, h) + b.unsqueeze(1)
                if l < ctrl.phi.n_layers - 1:
                    h = ctrl.activation()(h)
            return h

        xs = x.unsqueeze(0).expand(F, -1, -1)
        raw = phi(xs) - phi(x_eq.view(1, 1, -1).expand(F, 1, -1)) + u_eq
        f1 = torch.relu(raw - ctrl.u_lo) + ctrl.u_lo
        return -(torch.relu(ctrl.u_up - f1) - ctrl.u_up)

    # Inputs larger than this are only evaluated for logging by the training loop (full-buffer
    # statistics, never back-propagated): evaluate them in chunks without a graph to bound memory.
    max_grad_batch = 8192
    training_sample = True  # set False to evaluate all faults (diagnostics) even with sample_k

    def forward(self, z, save_new_x=False):
        if z.shape[0] > self.max_grad_batch:
            with torch.no_grad():
                return torch.cat([self._forward(c) for c in z.split(self.max_grad_batch)])
        return self._forward(z)

    def _forward(self, z):
        B = z.shape[0]
        rho = self.get_rho()
        Vz = self.lyapunov(z)                                    # (B, 1)
        loss1 = rho - Vz
        worst = None
        for m, dyn in enumerate(self.systems):
            u = self._controller_all(z, m)                       # (F, B, 1)
            F = u.shape[0]
            zz = z.unsqueeze(0).expand(F, -1, -1).reshape(F * B, -1)
            zn = dyn(zz, u.reshape(F * B, -1))
            Vn = self.lyapunov(zn).view(F, B, 1)
            loss2 = torch.relu(Vn - (1 - self.kappa) * Vz.unsqueeze(0))
            zn = zn.view(F, B, -1)
            loss3 = (torch.relu(self.box_lo - zn) + torch.relu(zn - self.box_up)).sum(-1, keepdim=True)
            out = -torch.minimum(loss1.unsqueeze(0).expand(F, -1, -1), loss2 + loss3)  # (F, B, 1)
            mmin = out.min(dim=0).values
            worst = mmin if worst is None else torch.minimum(worst, mmin)
        return worst

    def raw_margin(self, z):
        """Un-hinged certificate margin: min over modes, variants of
        min( (1-k)V(z) - V(z+), box margins of z+ ).  Returns (margin (B,), V(z) (B,)).
        The condition holds at z iff V(z) > rho or margin >= 0."""
        Vz = self.lyapunov(z)                                   # (B, 1)
        worst = None
        for m, dyn in enumerate(self.systems):
            u = self._controller_all(z, m)
            F, B = u.shape[0], z.shape[0]
            zn = dyn(z.unsqueeze(0).expand(F, -1, -1).reshape(F * B, -1), u.reshape(F * B, -1))
            Vn = self.lyapunov(zn).view(F, B, 1)
            dV = (1 - self.kappa) * Vz.unsqueeze(0) - Vn
            zn = zn.view(F, B, -1)
            box = torch.minimum((zn - self.box_lo).min(-1, keepdim=True).values,
                                (self.box_up - zn).min(-1, keepdim=True).values)
            mm = torch.minimum(dV, box).min(dim=0).values       # (B, 1)
            worst = mm if worst is None else torch.minimum(worst, mm)
        return worst.squeeze(-1), Vz.squeeze(-1)

    def per_fault_violation(self, z):
        """Diagnostics: (mode, F) max violation over the given z."""
        res = []
        prev, self.training_sample = self.training_sample, False
        with torch.no_grad():
            rho = self.get_rho(); Vz = self.lyapunov(z)
            for m, dyn in enumerate(self.systems):
                u = self._controller_all(z, m); F, B = u.shape[0], z.shape[0]
                zn = dyn(z.unsqueeze(0).expand(F, -1, -1).reshape(F * B, -1), u.reshape(F * B, -1))
                Vn = self.lyapunov(zn).view(F, B, 1)
                l2 = torch.relu(Vn - (1 - self.kappa) * Vz.unsqueeze(0))
                zn = zn.view(F, B, -1)
                l3 = (torch.relu(self.box_lo - zn) + torch.relu(zn - self.box_up)).sum(-1, keepdim=True)
                v = torch.relu(torch.minimum((rho - Vz).unsqueeze(0).expand(F, -1, -1), l2 + l3))
                res.append(v.amax(dim=(1, 2)))
        self.training_sample = prev
        return res
