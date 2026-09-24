"""
LDGA diagnostics (design doc C, sec. 5). Pure functions shared by the model
(fixed-point exit), the training logs and analyze_ldga.py.

  frequency_response(theta, lam)   g(lambda) = sum_m theta_m lambda^m  (nominal response)
  attention_spectrum(A)            eigenvalues of the (non-symmetric) attention graph, fp64
  spectral_gap(eig)                1 - |lambda_2|
  dirichlet_energy_grid(z, grid)   mean ||x_i/|x_i| - x_j/|x_j|||^2 over 4-neighbour patch pairs
  effective_rank(z)                exp(H(sigma / sum sigma)) of the centred patch-token matrix
  rel_state_change(z, z_prev)      ||z_t - z_{t-1}||_F / ||z_{t-1}||_F per sample

Dirichlet energy and effective rank match docs A/B, so all three papers use
comparable oversmoothing measures.
"""
from __future__ import annotations

import torch


def frequency_response(theta: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """g(lambda) = sum_m theta_m lambda^m (eq. C1 read as a polynomial filter).

    theta: (..., M+1); lam: (L,) real. Returns (..., L). The attention graph is
    non-symmetric, so this is a *nominal* response on the real line; show it next
    to the actual eigenvalue distribution (doc sec. 13)."""
    theta = theta.double()
    lam = lam.to(theta.device).double()
    powers = lam[None, :] ** torch.arange(theta.shape[-1], device=theta.device,
                                          dtype=torch.float64)[:, None]      # (M+1, L)
    return theta @ powers


def attention_spectrum(A: torch.Tensor) -> torch.Tensor:
    """Eigenvalues of row-stochastic attention matrices, computed in fp64.

    A: (..., T, T). Returns complex (..., T) sorted by decreasing modulus, so
    [..., 0] is the Perron eigenvalue (= 1) and [..., 1] is lambda_2."""
    eig = torch.linalg.eigvals(A.detach().cpu().double())
    order = eig.abs().argsort(dim=-1, descending=True)
    return eig.gather(-1, order)


def spectral_gap(eig: torch.Tensor) -> torch.Tensor:
    """1 - |lambda_2| of spectra sorted by `attention_spectrum`. (...,) float64."""
    return 1.0 - eig[..., 1].abs()


def dirichlet_energy_grid(z: torch.Tensor, grid: int) -> torch.Tensor:
    """E = mean over 4-neighbour patch pairs of ||x^_i - x^_j||^2 with x^ = x/||x||.

    z: (N, P, D) patch tokens (prefix tokens removed), P = grid^2. Returns (N,)."""
    x = torch.nn.functional.normalize(z.float(), dim=-1)
    x = x.reshape(x.shape[0], grid, grid, -1)
    dh = (x[:, :, 1:] - x[:, :, :-1]).pow(2).sum(-1).flatten(1)
    dv = (x[:, 1:, :] - x[:, :-1, :]).pow(2).sum(-1).flatten(1)
    return torch.cat([dh, dv], 1).mean(1)


def effective_rank(z: torch.Tensor) -> torch.Tensor:
    """exp(H(sigma / sum sigma)) of the token-centred (P, D) matrix, per sample -> (N,)."""
    x = z.float()
    x = x - x.mean(1, keepdim=True)
    s = torch.linalg.svdvals(x)
    p = s / s.sum(-1, keepdim=True).clamp_min(1e-12)
    return torch.exp(-(p * (p + 1e-12).log()).sum(-1))


def rel_state_change(z: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
    """||z_t - z_{t-1}||_F / ||z_{t-1}||_F per sample, in fp32 (sec. 6). (N, ...) -> (N,)."""
    z, zp = z.float().flatten(1), z_prev.float().flatten(1)
    return (z - zp).norm(dim=1) / zp.norm(dim=1).clamp_min(1e-12)
