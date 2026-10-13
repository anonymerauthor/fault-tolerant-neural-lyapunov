"""Shared-weight MLP whose operating modes are selected by per-mode sign masks.

pi^(m)(x) = clip(A_L h_{L-1} + c_L),  h_l = act(A_l h_{l-1} + c_l),
A_l = M_l * W_l,  c_l = mu_l * b_l,  M_l, mu_l in {-1, +1}.

Masks are parameterized by real scores; sign(score) in the forward pass and a
straight-through estimator (identity gradient, clipped to |score| <= 1) backward.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        ctx.save_for_backward(scores)
        return torch.where(scores >= 0, 1.0, -1.0).to(scores.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (scores,) = ctx.saved_tensors
        return grad_out * (scores.abs() <= 1).to(grad_out.dtype)


class MaskedMLP(nn.Module):
    def __init__(self, dims: list[int], n_modes: int, activation=nn.Tanh, u_lo=None, u_up=None,
                 score_scale: float = 1.0, delta_scores: bool = False):
        super().__init__()
        self.dims = dims
        self.n_modes = n_modes
        # mask = sign(score_scale * raw_score); score_scale > 1 raises the effective learning rate of
        # the masks under Adam without changing the optimizer of the shared weights
        self.score_scale = score_scale
        # delta_scores: score of mode m>0 is s_0 + delta_m, so an L1 penalty on the parameters
        # (applied by the training loop) acts as a Hamming-proximity regularizer between modes
        self.delta_scores = delta_scores
        self.act = activation()
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.w_scores = nn.ParameterList()
        self.b_scores = nn.ParameterList()
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layer = nn.Linear(d_in, d_out)
            self.weights.append(nn.Parameter(layer.weight.detach().clone()))
            self.biases.append(nn.Parameter(layer.bias.detach().clone()))
            self.w_scores.append(nn.Parameter(0.5 * torch.randn(n_modes, d_out, d_in)))
            self.b_scores.append(nn.Parameter(0.5 * torch.randn(n_modes, d_out)))
        self.register_buffer("u_lo", None if u_lo is None else torch.as_tensor(u_lo, dtype=torch.float32))
        self.register_buffer("u_up", None if u_up is None else torch.as_tensor(u_up, dtype=torch.float32))

    @property
    def n_layers(self) -> int:
        return len(self.weights)

    def masks(self, mode: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Hard sign masks of one mode (STE-differentiable)."""
        def eff(s):
            return s[0] + s[mode] if (self.delta_scores and mode > 0) else s[mode]
        mw = [SignSTE.apply(self.score_scale * eff(s)) for s in self.w_scores]
        mb = [SignSTE.apply(self.score_scale * eff(s)) for s in self.b_scores]
        return mw, mb

    def forward_with_masks(self, x, mw, mb):
        h = x
        for l in range(self.n_layers):
            h = h @ (mw[l] * self.weights[l]).T + mb[l] * self.biases[l]
            if l < self.n_layers - 1:
                h = self.act(h)
        if self.u_lo is not None:
            h = torch.maximum(torch.minimum(h, self.u_up), self.u_lo)
        return h

    def forward(self, x, mode: int):
        mw, mb = self.masks(mode)
        return self.forward_with_masks(x, mw, mb)


# ---------------------------------------------------------------------------
# Theorem 1 / Corollary 1 bounds (theory/theorem1_jump_bound.md)
# ---------------------------------------------------------------------------

def _abs_spec(w: torch.Tensor) -> float:
    return torch.linalg.matrix_norm(w.abs(), ord=2).item()


@torch.no_grad()
def mask_invariant_constants(net: MaskedMLP, r0: float):
    """rho_l = || |W_l| ||_2 and activation-norm bounds H_l (Lemma 0, Lemma 1)."""
    rho = [_abs_spec(w) for w in net.weights]
    H = [r0]
    for l in range(net.n_layers):
        H.append(rho[l] * H[-1] + net.biases[l].norm().item())
    return rho, H


@torch.no_grad()
def jump_bound(net: MaskedMLP, masks_i, masks_j, r0: float) -> dict:
    """Return Theorem 1 (spectral) and Corollary 1 (additive l1) bounds on
    sup_{||x||<=r0} ||pi^(j)(x) - pi^(i)(x)||_2."""
    (mw_i, mb_i), (mw_j, mb_j) = masks_i, masks_j
    rho, H = mask_invariant_constants(net, r0)
    L = net.n_layers
    thm1, cor1 = 0.0, 0.0
    for l in range(L):
        tail = 1.0
        for k in range(l + 1, L):
            tail *= rho[k]
        fw = (mw_i[l] != mw_j[l]).float()
        fb = (mb_i[l] != mb_j[l]).float()
        dw = net.weights[l].abs() * fw
        db = net.biases[l].abs() * fb
        thm1 += tail * (2 * _abs_spec(dw) * H[l] + 2 * db.norm().item())
        cor1 += tail * (2 * dw.sum().item() * H[l] + 2 * db.sum().item())
    return {"theorem1": thm1, "corollary1": cor1}


@torch.no_grad()
def bit_sensitivities(net: MaskedMLP, r0: float) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per-parameter sensitivities s_k = 2 Gamma_l |theta_k| (Corollary 1)."""
    rho, H = mask_invariant_constants(net, r0)
    L = net.n_layers
    out = []
    for l in range(L):
        tail = 1.0
        for k in range(l + 1, L):
            tail *= rho[k]
        out.append((2 * tail * H[l] * net.weights[l].abs(), 2 * tail * net.biases[l].abs()))
    return out
