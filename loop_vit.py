"""
LoopViT + LDGA (Loop Diffusion Graph Attention) for image classification.

Base architecture: "LoopViT: Scaling Visual ARC with Looped Transformers"
(arXiv:2602.02156), ported to whole-image classification (see the original
`loopvit_paper/loop_vit.py`). Everything below the LDGA additions is the same
model; with `diffusion="none"` and `loop_relax=False` the outputs are
bit-identical to the original (tests/test_ldga.py::test_t1_baseline_equivalence).

LoopViT recap
-------------
        z_0   = emb(x)
        z_t+1 = M_theta(z_t + e_t)          eq. (2) in the paper
        y     = head(z_T)
B distinct Hybrid blocks (2D-RoPE MHSA + Heterogeneous ConvGLU) are applied T
times; e_t is a learned step embedding with identity extrapolation past T.

LDGA (design doc C, "Looping as Diffusion")
-------------------------------------------
Softmax attention A is row-stochastic, so A.V is a one-hop low-pass filter on
the attention graph, and LoopViT applies it B*T times with tied weights. LDGA
replaces A.V by a learnable polynomial graph filter:

    Attn_LDGA(V) = sum_{m=0}^{M} theta_m A^m V                          (eq. C1)

  none   theta = (0, 1, 0, ...)                       vanilla, 1-hop low-pass
  ppr    theta_m = alpha (1-alpha)^m, alpha = sigmoid(a)   personalised PageRank
  heat   theta_m = e^-tau tau^m / m!, tau = softplus(s)    heat kernel exp(-tau(I-A))
  gpr    free theta_m (GPR-GNN style), can be high-pass (e.g. (1,-1) = (I-A)V)

ppr/heat are renormalised to sum to 1 after truncation (diff_renorm). gpr is
initialised to (0, 1, 0, ...) so at init LDGA-gpr is exactly vanilla LoopViT.
`diff_schedule="per_step"` gives one coefficient set per loop step (identity
extrapolated past T_train); `diff_heads=k` diffuses only the first k heads.

diff_impl="sdpa" computes A^m V by feeding the previous hop back in as the
values of the same fused attention (never materialises A); "dense" builds A
once in fp32 (exact, for analysis).

Optional Euler-style damping of the loop (eq. C2, `loop_relax`):
    z_{t+1} = z_t + eta_t (M_theta(z_t + e_t) - z_t),   eta_t = softplus(.) = 1 at init

Dynamic exit (sec. 6): besides the paper's entropy criterion, a sample can halt
at a fixed point, ||z_t - z_{t-1}|| / ||z_{t-1}|| < exit_fp_eps. `dynamic_forward`
compacts the active batch so exited samples really stop costing compute
(`block_apps`).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldga_stats import rel_state_change

DIFFUSION_MODES = ("none", "ppr", "heat", "gpr")
DIFF_SCHEDULES = ("shared", "per_step")
DIFF_IMPLS = ("sdpa", "dense")
GPR_INITS = ("vanilla", "ppr")
EXIT_MODES = ("entropy", "fixedpoint", "both", "either")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class LoopViTConfig:
    image_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    num_classes: int = 10

    dim: int = 384
    core_depth: int = 4          # B: distinct hybrid blocks in the shared core
    loop_steps: int = 3          # T: iterations of the core during training
    num_heads: int = 6
    mlp_ratio: float = 4.0       # FFN hidden = dim * mlp_ratio
    qkv_bias: bool = True
    dropout: float = 0.0
    attn_dropout: float = 0.0
    drop_path: float = 0.1       # max stochastic-depth rate over the B*T applications

    ffn: str = "hybrid"          # "hybrid" (ConvGLU + DW-Conv) | "vanilla" (plain MLP)
    rope: bool = True            # 2D rotary embeddings in the attention
    step_embedding: bool = True  # e_t in eq. (2)
    num_cls_tokens: int = 1      # the paper's "task tokens"; they bypass the DW-Conv
    pool: str = "cls"            # "cls" | "mean"

    # dynamic exit (inference only)
    exit_tau: float = 0.05       # halt a sample once its entropy (nats) drops below this
    min_loop_steps: int = 1
    max_loop_steps: int = 0      # 0 = use loop_steps

    # --- LDGA: loop diffusion graph attention ---
    diffusion: str = "none"           # "none" | "ppr" | "heat" | "gpr"
    diff_hops: int = 3                # M
    diff_heads: int = -1              # -1 = all heads, else first k heads
    diff_schedule: str = "shared"     # "shared" | "per_step"
    diff_renorm: bool = True          # renormalise truncated ppr/heat coefficients to sum to 1
    diff_impl: str = "sdpa"           # "sdpa" (repeated fused attention) | "dense" (materialise A once)
    ppr_alpha_init: float = 0.2
    heat_tau_init: float = 1.0
    gpr_init: str = "vanilla"         # "vanilla" | "ppr"

    # --- optional ODE-style damping of the loop ---
    loop_relax: bool = False

    # --- dynamic exit criteria ---
    exit_mode: str = "entropy"        # "entropy" | "fixedpoint" | "both" | "either"
    exit_fp_eps: float = 0.01         # relative state change ||z_t - z_{t-1}|| / ||z_{t-1}||

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def drop_path(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Stochastic depth. The rate is passed per call because a shared block is
    applied at several depths and each application gets its own rate."""
    if p == 0.0 or not training:
        return x
    keep = 1.0 - p
    mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
    return x * mask / keep


def softplus_inv(y: float) -> float:
    """x with softplus(x) = y (y > 0)."""
    return y + math.log(-math.expm1(-y))


class RMSNorm(nn.Module):
    """Pre-norm of choice in the paper (eq. 6/7): cheaper than LayerNorm and
    numerically better behaved when the same weights run many times."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return x32.to(dtype) * self.weight


class PatchEmbed(nn.Module):
    def __init__(self, image_size, patch_size, in_chans, dim):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                      # (N, C, H, W)
        x = self.proj(x)                       # (N, D, g, g)
        return x.flatten(2).transpose(1, 2)    # (N, P, D)


class Rotary2D(nn.Module):
    """Axial 2D RoPE: half of each head's channels encode the patch row, the
    other half the column. The first `n_prefix` tokens (CLS / task tokens) are
    left un-rotated, matching `no_rope=num_task_tokens` in the reference code."""

    def __init__(self, head_dim: int, grid: int, n_prefix: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"2D RoPE needs head_dim divisible by 4, got {head_dim}")
        q = head_dim // 4                                   # frequencies per axis
        freqs = 1.0 / (theta ** (torch.arange(q).float() / q))
        ang = torch.outer(torch.arange(grid).float(), freqs)         # (grid, q)
        row = ang[:, None, :].expand(grid, grid, q)
        col = ang[None, :, :].expand(grid, grid, q)
        a = torch.cat([row, col], dim=-1).reshape(grid * grid, 2 * q)  # (P, head_dim/2)
        self.register_buffer("cos", a.cos()[None, None], persistent=False)
        self.register_buffer("sin", a.sin()[None, None], persistent=False)
        self.n_prefix = n_prefix

    def forward(self, t):                      # (N, h, T, d)
        p = self.n_prefix
        prefix, x = t[:, :, :p, :], t[:, :, p:, :]
        x1, x2 = x.chunk(2, dim=-1)
        cos, sin = self.cos.to(x.dtype), self.sin.to(x.dtype)
        x = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return torch.cat([prefix, x], dim=2) if p else x


class DiffusionCoeffs(nn.Module):
    """Filter coefficients theta of eq. C1 for loop step t (doc sec. 2.1 / 2.3 / 4.1).

    ppr : parameter `a` (sigmoid(a) = alpha)  -> theta_m = alpha (1 - alpha)^m
    heat: parameter `s` (softplus(s) = tau)   -> theta_m = e^-tau tau^m / m!
    gpr : parameter `theta` directly (init "vanilla" = (0, 1, 0, ...) or "ppr")
    Shared schedule: one set per head, shape (1, h[, M+1]); per_step: (T_train, h[, M+1]),
    indexed with min(t, T_train - 1) (identity extrapolation, like `_step_vector`)."""

    def __init__(self, mode: str, heads: int, hops: int, schedule: str, steps: int,
                 renorm: bool, alpha0: float, tau0: float, gpr_init: str):
        super().__init__()
        if mode not in DIFFUSION_MODES[1:]:
            raise ValueError(f"DiffusionCoeffs mode must be one of {DIFFUSION_MODES[1:]}")
        S = steps if schedule == "per_step" else 1
        self.mode, self.hops, self.renorm = mode, hops, renorm
        self.register_buffer("m", torch.arange(hops + 1, dtype=torch.float32), persistent=False)
        if mode == "ppr":
            self.a = nn.Parameter(torch.full((S, heads), math.log(alpha0 / (1 - alpha0))))
        elif mode == "heat":
            self.s = nn.Parameter(torch.full((S, heads), softplus_inv(tau0)))
        else:
            if gpr_init == "vanilla":
                th = torch.zeros(hops + 1)
                th[1] = 1.0
            else:   # GPR-GNN recipe: renormalised PPR
                th = alpha0 * (1 - alpha0) ** torch.arange(hops + 1, dtype=torch.float32)
                th = th / th.sum()
            self.theta = nn.Parameter(th.expand(S, heads, hops + 1).clone())

    @property
    def num_steps(self) -> int:
        p = self.theta if self.mode == "gpr" else (self.a if self.mode == "ppr" else self.s)
        return p.shape[0]

    def _theta(self, rows: slice | int) -> torch.Tensor:
        """theta for the given schedule rows, fp32, last dim M+1."""
        with torch.autocast(device_type=self.m.device.type, enabled=False):
            m = self.m
            if self.mode == "gpr":
                return self.theta[rows].float()
            if self.mode == "ppr":
                al = torch.sigmoid(self.a[rows].float())[..., None]
                th = al * (1 - al) ** m
            else:
                tau = F.softplus(self.s[rows].float())[..., None]
                th = torch.exp(-tau + m * tau.log() - torch.lgamma(m + 1))
            if self.renorm:
                th = th / th.sum(-1, keepdim=True)
            return th

    def forward(self, t: int) -> torch.Tensor:
        """theta for loop step t: (h, M+1) fp32."""
        return self._theta(min(t, self.num_steps - 1))

    def table(self) -> torch.Tensor:
        """theta for every schedule row: (S, h, M+1) fp32."""
        return self._theta(slice(None))

    def alpha(self) -> torch.Tensor | None:
        """Effective alpha (S, h) for ppr, else None."""
        return torch.sigmoid(self.a.float()) if self.mode == "ppr" else None

    def tau(self) -> torch.Tensor | None:
        """Effective tau (S, h) for heat, else None."""
        return F.softplus(self.s.float()) if self.mode == "heat" else None


class Attention(nn.Module):
    """MHSA with optional 2D rotary embeddings (eq. 3-5) and an optional
    polynomial graph filter over the attention graph (LDGA, eq. C1)."""

    def __init__(self, dim, num_heads, grid, n_prefix, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0, rope=True, diffusion="none", hops=3,
                 diff_heads=-1, schedule="shared", steps=1, renorm=True, impl="sdpa",
                 alpha0=0.2, tau0=1.0, gpr_init="vanilla"):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        if diffusion not in DIFFUSION_MODES:
            raise ValueError(f"diffusion must be one of {DIFFUSION_MODES}, got {diffusion!r}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rotary = Rotary2D(self.head_dim, grid, n_prefix) if rope else None

        self.diffusion, self.impl, self.hops = diffusion, impl, hops
        self.diff_heads = num_heads if diff_heads == -1 else diff_heads
        # coefficients only exist when diffusing, so a "none" model has exactly
        # the original state-dict keys.
        self.coeffs = (DiffusionCoeffs(diffusion, self.diff_heads, hops, schedule, steps, renorm,
                                       alpha0, tau0, gpr_init)
                       if diffusion != "none" else None)
        # analysis hook: when True, forward stores the dense fp32 attention graph
        # (N, h, T, T) in `last_A` and the per-head filter (h, M+1) in `last_theta`.
        self.record = False
        self.last_A: torch.Tensor | None = None
        self.last_theta: torch.Tensor | None = None

    def _qkv(self, x):
        N, T, _ = x.shape
        q, k, v = (self.qkv(x)
                   .reshape(N, T, 3, self.num_heads, self.head_dim)
                   .permute(2, 0, 3, 1, 4))            # 3 x (N, h, T, d)
        if self.rotary is not None:
            q, k = self.rotary(q), self.rotary(k)
        return q, k, v

    def _dense_A(self, q, k) -> torch.Tensor:
        """A = softmax(QK^T / sqrt(d)) in fp32 with autocast disabled: (N, h', T, T)."""
        with torch.autocast(device_type=q.device.type, enabled=False):
            return ((q.float() @ k.float().transpose(-2, -1)) * self.scale).softmax(-1)

    def head_theta(self, t: int = 0) -> torch.Tensor:
        """The filter every head applies at step t, as (h, M+1) fp32. Vanilla heads
        (all heads for diffusion='none', else heads >= diff_heads) are (0, 1, 0, ...)."""
        M = self.hops if self.coeffs is not None else 1
        th = torch.zeros(self.num_heads, M + 1, device=self.qkv.weight.device)
        th[:, 1] = 1.0
        if self.coeffs is not None:
            th[:self.diff_heads] = self.coeffs(t).detach()
        return th

    def _mix(self, q, k, v, t: int) -> torch.Tensor:
        """Attention output before the output projection, (N, h, T, d): eq. C1 on
        heads [0, diff_heads), plain SDPA on the rest."""
        drop = self.attn_drop if self.training else 0.0
        if self.coeffs is None:
            return F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        th = self.coeffs(t)                                   # (hd, M+1) fp32
        hd = self.diff_heads
        qd, kd, vd = q[:, :hd], k[:, :hd], v[:, :hd]
        if self.impl == "sdpa":
            # A^m V = SDPA(q, k, A^{m-1} V): the previous hop is fed back in as values
            u = vd
            out = th[:, 0].view(1, hd, 1, 1).to(u.dtype) * u
            for m in range(1, self.hops + 1):
                u = F.scaled_dot_product_attention(qd, kd, u)
                out = out + th[:, m].view(1, hd, 1, 1).to(u.dtype) * u
        else:   # dense: materialise A once in fp32
            with torch.autocast(device_type=q.device.type, enabled=False):
                A = self._dense_A(qd, kd)
                u = vd.float()
                out = th[:, 0].view(1, hd, 1, 1) * u
                for m in range(1, self.hops + 1):
                    u = A @ u
                    out = out + th[:, m].view(1, hd, 1, 1) * u
            out = out.to(v.dtype)
        if hd < self.num_heads:
            rest = F.scaled_dot_product_attention(q[:, hd:], k[:, hd:], v[:, hd:], dropout_p=drop)
            out = torch.cat([out, rest.to(out.dtype)], dim=1)
        return out

    def forward(self, x, t: int = 0, return_A: bool = False):
        N, T, D = x.shape
        q, k, v = self._qkv(x)
        out = self._mix(q, k, v, t)
        A = None
        if return_A or self.record:
            with torch.no_grad():
                A = self._dense_A(q.detach(), k.detach())       # analysis only
            if self.record:
                self.last_A, self.last_theta = A, self.head_theta(t)
        out = out.transpose(1, 2).reshape(N, T, D)
        out = self.proj_drop(self.proj(out))
        return (out, A) if return_A else out


class HeteroConvGLU(nn.Module):
    """The paper's Heterogeneous ConvGLU FFN (eq. 8-10, Fig. 4D).

    The gate branch is split by token type: CLS / task tokens bypass the
    depth-wise convolution so the abstract rule token is never smeared across
    the grid, while image tokens are folded back into (g, g) and convolved."""

    def __init__(self, dim, hidden, grid, n_prefix, drop=0.0):
        super().__init__()
        self.grid, self.n_prefix = grid, n_prefix
        self.fc1 = nn.Linear(dim, hidden * 2)          # -> [X_gate, X_val]
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):                              # (N, T, D)
        gate, val = self.fc1(x).chunk(2, dim=-1)
        p, g = self.n_prefix, self.grid
        task, img = gate[:, :p], gate[:, p:]
        N, P, H = img.shape
        img = self.dw(img.transpose(1, 2).reshape(N, H, g, g)).flatten(2).transpose(1, 2)
        gate = torch.cat([task, img], dim=1) if p else img
        return self.drop(self.fc2(self.act(gate) * val))


class MLP(nn.Module):
    """Vanilla FFN, for the Hybrid-vs-Vanilla ablation (`ffn: vanilla`)."""

    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class HybridBlock(nn.Module):
    """One layer of the recurrent core: pre-norm MHSA + ConvGLU (Fig. 4C)."""

    def __init__(self, cfg: LoopViTConfig, grid: int):
        super().__init__()
        hidden = int(cfg.dim * cfg.mlp_ratio)
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = Attention(cfg.dim, cfg.num_heads, grid, cfg.num_cls_tokens,
                              cfg.qkv_bias, cfg.attn_dropout, cfg.dropout, cfg.rope,
                              diffusion=cfg.diffusion, hops=cfg.diff_hops,
                              diff_heads=cfg.diff_heads, schedule=cfg.diff_schedule,
                              steps=cfg.loop_steps, renorm=cfg.diff_renorm, impl=cfg.diff_impl,
                              alpha0=cfg.ppr_alpha_init, tau0=cfg.heat_tau_init,
                              gpr_init=cfg.gpr_init)
        self.norm2 = RMSNorm(cfg.dim)
        if cfg.ffn == "hybrid":
            self.ffn = HeteroConvGLU(cfg.dim, hidden, grid, cfg.num_cls_tokens, cfg.dropout)
        elif cfg.ffn == "vanilla":
            self.ffn = MLP(cfg.dim, hidden, cfg.dropout)
        else:
            raise ValueError(f"ffn must be 'hybrid' or 'vanilla', got {cfg.ffn!r}")

    def forward(self, x, dp: float = 0.0, t: int = 0):
        x = x + drop_path(self.attn(self.norm1(x), t=t), dp, self.training)
        x = x + drop_path(self.ffn(self.norm2(x)), dp, self.training)
        return x


# --------------------------------------------------------------------------- #
# LoopViT
# --------------------------------------------------------------------------- #
class LoopViT(nn.Module):
    def __init__(self, cfg: LoopViTConfig):
        super().__init__()
        if cfg.pool not in ("cls", "mean"):
            raise ValueError("pool must be 'cls' or 'mean'")
        if cfg.pool == "cls" and cfg.num_cls_tokens < 1:
            raise ValueError("pool='cls' needs num_cls_tokens >= 1")
        if cfg.min_loop_steps < 1:
            raise ValueError("min_loop_steps must be >= 1")
        self._validate_ldga(cfg)
        self.cfg = cfg
        D, n_cls = cfg.dim, cfg.num_cls_tokens

        self.patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.in_chans, D)
        n_tok = self.patch_embed.num_patches + n_cls
        self.cls_token = nn.Parameter(torch.zeros(1, n_cls, D)) if n_cls else None
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tok, D))
        self.pos_drop = nn.Dropout(cfg.dropout)

        # M_theta: B distinct blocks, reused on every iteration.
        self.core = nn.ModuleList([HybridBlock(cfg, self.patch_embed.grid)
                                   for _ in range(cfg.core_depth)])

        # e_t: one learned vector per iteration; steps past T reuse the last one.
        self.step_embed = (nn.Parameter(torch.zeros(cfg.loop_steps, 1, 1, D))
                           if cfg.step_embedding else None)

        # eta_t of eq. C2; softplus(log(e - 1)) = 1, so at init the loop is unchanged.
        self.eta_raw = (nn.Parameter(torch.full((cfg.loop_steps,), math.log(math.e - 1)))
                        if cfg.loop_relax else None)

        self.norm = RMSNorm(D)
        self.head = nn.Linear(D, cfg.num_classes)
        self._init_weights()

    @staticmethod
    def _validate_ldga(cfg: LoopViTConfig):
        """Config checks of design doc C, sec. 3."""
        if cfg.diffusion not in DIFFUSION_MODES:
            raise ValueError(f"diffusion must be one of {DIFFUSION_MODES}")
        if cfg.diff_schedule not in DIFF_SCHEDULES:
            raise ValueError(f"diff_schedule must be one of {DIFF_SCHEDULES}")
        if cfg.diff_impl not in DIFF_IMPLS:
            raise ValueError(f"diff_impl must be one of {DIFF_IMPLS}")
        if cfg.gpr_init not in GPR_INITS:
            raise ValueError(f"gpr_init must be one of {GPR_INITS}")
        if cfg.exit_mode not in EXIT_MODES:
            raise ValueError(f"exit_mode must be one of {EXIT_MODES}")
        if cfg.diff_hops < 1:
            raise ValueError("diff_hops must be >= 1")
        if not (cfg.diff_heads == -1 or 1 <= cfg.diff_heads <= cfg.num_heads):
            raise ValueError(f"diff_heads must be -1 or in [1, num_heads={cfg.num_heads}]")
        if not 0.0 < cfg.ppr_alpha_init < 1.0:
            raise ValueError("ppr_alpha_init must be in (0, 1)")
        if cfg.heat_tau_init <= 0:
            raise ValueError("heat_tau_init must be > 0")
        if cfg.attn_dropout > 0 and cfg.diffusion != "none":
            raise ValueError("attn_dropout > 0 is not supported with diffusion != 'none' "
                             "(per-hop dropout is not implemented)")

    # ---- init --------------------------------------------------------------
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.step_embed is not None:
            nn.init.trunc_normal_(self.step_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                w = m.weight.data
                nn.init.xavier_uniform_(w.view(w.shape[0], -1))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ---- pieces ------------------------------------------------------------
    def _drop_path_rates(self, total: int):
        """Linearly increasing rate over the UNROLLED depth (1 ... B*T)."""
        if total <= 1:
            return [0.0] * total
        return [self.cfg.drop_path * i / (total - 1) for i in range(total)]

    def _step_vector(self, t: int):
        """e_t, with identity extrapolation past the training budget."""
        if self.step_embed is None:
            return None
        return self.step_embed[min(t, self.step_embed.shape[0] - 1)]

    def _eta(self, t: int) -> torch.Tensor:
        """eta_t = softplus(eta_raw[min(t, T_train - 1)]) (eq. C2)."""
        return F.softplus(self.eta_raw[min(t, self.eta_raw.shape[0] - 1)])

    def embed(self, x):
        """z_0 = emb(x)."""
        h = self.patch_embed(x)
        if self.cls_token is not None:
            h = torch.cat([self.cls_token.expand(h.shape[0], -1, -1), h], dim=1)
        return self.pos_drop(h + self.pos_embed)

    def core_step(self, h, t: int, rates=None, block_hook=None):
        """One application of the shared core: z_{t+1} = M_theta(z_t + e_t).

        `t` reaches every attention (per-step coefficients). `block_hook(b, h)` is
        called after each block, for per-application analysis."""
        e = self._step_vector(t)
        if e is not None:
            h = h + e
        B = len(self.core)
        for b, blk in enumerate(self.core):
            h = blk(h, rates[t * B + b] if rates else 0.0, t=t)
            if block_hook is not None:
                block_hook(b, h)
        return h

    def advance(self, h, t: int, rates=None, block_hook=None):
        """One loop step including the optional relaxation of eq. C2."""
        h_new = self.core_step(h, t, rates, block_hook)
        if self.eta_raw is None:
            return h_new
        return h + self._eta(t) * (h_new - h)

    def loop(self, h, num_steps: int, return_all: bool = False):
        rates = self._drop_path_rates(len(self.core) * num_steps)
        states = []
        for t in range(num_steps):
            h = self.advance(h, t, rates)
            if return_all:
                states.append(h)
        return states if return_all else h

    def classify(self, h):
        h = self.norm(h)
        pooled = (h[:, :self.cfg.num_cls_tokens].mean(dim=1) if self.cfg.pool == "cls"
                  else h[:, self.cfg.num_cls_tokens:].mean(dim=1))
        return self.head(pooled)

    # ---- forward -----------------------------------------------------------
    def forward(self, x, num_steps: int | None = None):
        return self.classify(self.loop(self.embed(x), num_steps or self.cfg.loop_steps))

    def logits_per_step(self, x, num_steps: int | None = None):
        """Logits read out after every iteration: list of (N, num_classes).
        Shows the 'crystallization' of the prediction across the loop."""
        states = self.loop(self.embed(x), num_steps or self.cfg.loop_steps, return_all=True)
        return [self.classify(s) for s in states]

    def logits_and_states(self, x, num_steps: int | None = None):
        """(list of per-step logits, [z_0, z_1, ..., z_S]) for the per-step diagnostics."""
        z0 = self.embed(x)
        hs = self.loop(z0, num_steps or self.cfg.loop_steps, return_all=True)
        return [self.classify(s) for s in hs], [z0] + hs

    @staticmethod
    def entropy(logits):
        """H_t = -sum_c p log p, in nats (eq. 11, over classes instead of pixels)."""
        logp = F.log_softmax(logits.float(), dim=-1)
        return -(logp.exp() * logp).sum(-1)

    @staticmethod
    def exit_decision(ent, fp, t: int, tau, fp_eps: float, mode: str):
        """Exit rule of sec. 6 (before the min_steps check). ent/fp: (n,)."""
        ent_ok = ent < tau
        fp_ok = (fp < fp_eps) if (t >= 1 and fp is not None) else torch.zeros_like(ent_ok)
        if mode == "entropy":
            return ent_ok
        if mode == "fixedpoint":
            return fp_ok
        if mode == "both":
            return ent_ok & fp_ok
        return ent_ok | fp_ok

    @torch.no_grad()
    def dynamic_forward(self, x, tau=None, min_steps: int | None = None,
                        max_steps: int | None = None, exit_mode: str | None = None,
                        fp_eps: float | None = None):
        """Dynamic Exit (sec. 3.3 of the paper, extended by sec. 6 of design C).

        Per sample: iterate until the exit rule fires, then freeze that sample's
        state (z_k = z_t) and drop it from the active batch, so further
        iterations cost it nothing. `tau` may be a float or an (N,) tensor.

        Returns a dict with the final `logits`, per-sample `exit_steps`, the
        (N, steps_run) `entropy` and `fp_trace` traces (fp_trace is the relative
        state change, NaN at t=0 and for samples that had already exited),
        `steps_run` (the deepest sample) and `block_apps` (block applications
        actually executed, summed over samples -- the real compute metric)."""
        cfg = self.cfg
        tau = cfg.exit_tau if tau is None else tau
        min_steps = cfg.min_loop_steps if min_steps is None else min_steps
        max_steps = max_steps or cfg.max_loop_steps or cfg.loop_steps
        mode = exit_mode or cfg.exit_mode
        fp_eps = cfg.exit_fp_eps if fp_eps is None else fp_eps
        if mode not in EXIT_MODES:
            raise ValueError(f"exit_mode must be one of {EXIT_MODES}")
        tau_t = tau if torch.is_tensor(tau) else None

        h = self.embed(x).clone()
        N, B = x.shape[0], len(self.core)
        dev = x.device
        done = torch.zeros(N, dtype=torch.bool, device=dev)
        exit_steps = torch.full((N,), max_steps, dtype=torch.long, device=dev)
        logits = ent = None
        ent_trace, fp_trace, block_apps, t = [], [], 0, 0
        for t in range(max_steps):
            active = (~done).nonzero(as_tuple=True)[0]
            h_prev_a = h[active]
            h_new_a = self.advance(h_prev_a, t)
            block_apps += active.numel() * B
            h[active] = h_new_a.to(h.dtype)

            lg_a = self.classify(h_new_a)
            if logits is None:
                logits = torch.zeros(N, lg_a.shape[-1], dtype=lg_a.dtype, device=dev)
                ent = torch.zeros(N, device=dev)
            logits[active] = lg_a.to(logits.dtype)
            ent_a = self.entropy(lg_a)
            ent[active] = ent_a

            fp = torch.full((N,), float("nan"), device=dev)
            fp_a = None
            if t >= 1:
                fp_a = rel_state_change(h_new_a, h_prev_a)
                fp[active] = fp_a
            ent_trace.append(ent.clone())
            fp_trace.append(fp)

            if (t + 1) >= min_steps:
                tau_a = tau_t[active] if tau_t is not None else tau
                newly_a = self.exit_decision(ent_a, fp_a, t, tau_a, fp_eps, mode)
                newly = active[newly_a]
                exit_steps[newly] = t + 1
                done[newly] = True
            if bool(done.all()):
                break
        return {"logits": logits, "exit_steps": exit_steps,
                "entropy": torch.stack(ent_trace, dim=1),
                "fp_trace": torch.stack(fp_trace, dim=1),
                "steps_run": t + 1, "block_apps": block_apps}

    # ---- LDGA bookkeeping --------------------------------------------------
    def diffusion_values(self) -> dict | None:
        """Learned filters, or None for diffusion='none'. Keys: `theta` (B, S, hd, M+1),
        `theta_sum` (B, S, hd), and `alpha` / `tau` (B, S, hd) for ppr / heat."""
        if self.cfg.diffusion == "none":
            return None
        with torch.no_grad():
            cs = [blk.attn.coeffs for blk in self.core]
            th = torch.stack([c.table().cpu() for c in cs])
            out = {"theta": th, "theta_sum": th.sum(-1)}
            if self.cfg.diffusion == "ppr":
                out["alpha"] = torch.stack([c.alpha().cpu() for c in cs])
            if self.cfg.diffusion == "heat":
                out["tau"] = torch.stack([c.tau().cpu() for c in cs])
        return out

    def eta_values(self) -> torch.Tensor | None:
        """Learned eta_t (T_train,) of eq. C2, or None without loop_relax."""
        if self.eta_raw is None:
            return None
        return F.softplus(self.eta_raw.detach().float()).cpu()

    def set_record_attention(self, on: bool = True):
        for blk in self.core:
            blk.attn.record = on
            blk.attn.last_A = blk.attn.last_theta = None

    def param_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        per_block = sum(p.numel() for p in self.core[0].parameters())
        core = per_block * len(self.core)
        T = self.cfg.loop_steps
        ldga = sum(p.numel() for blk in self.core if blk.attn.coeffs is not None
                   for p in blk.attn.coeffs.parameters())
        return {
            "total_params": total,
            "params_per_block": per_block,
            "core_params (shared)": core,
            "other_params (embed/norm/head)": total - core,
            "ldga_params": ldga,
            "relax_params": self.eta_raw.numel() if self.eta_raw is not None else 0,
            "block_applications": len(self.core) * T,
            "untied_equivalent_params": total + core * (T - 1),
        }

    def flops(self, num_steps: int | None = None) -> dict:
        return flop_report(self.cfg, num_steps)


def build_model(**kwargs) -> LoopViT:
    return LoopViT(LoopViTConfig(**kwargs))


# --------------------------------------------------------------------------- #
# Compute accounting (design doc C, sec. 7.3)
# --------------------------------------------------------------------------- #
def flop_report(cfg: LoopViTConfig, num_steps: int | None = None) -> dict:
    """Analytic forward FLOPs per image (2 x MACs; norms, softmax and activations
    ignored). The `sdpa` impl recomputes QK^T for every hop; `dense` computes it
    once and repeats only the A @ u products."""
    S = num_steps or cfg.loop_steps
    g = cfg.image_size // cfg.patch_size
    P = g * g
    Tt, D, H = P + cfg.num_cls_tokens, cfg.dim, int(cfg.dim * cfg.mlp_ratio)
    qkv_proj = 2 * Tt * D * 3 * D + 2 * Tt * D * D
    qk = av = 2 * Tt * Tt * D                       # all heads together
    vanilla_attn = qk + av
    if cfg.diffusion == "none":
        attn = vanilla_attn
    else:
        f = (cfg.num_heads if cfg.diff_heads == -1 else cfg.diff_heads) / cfg.num_heads
        M = cfg.diff_hops
        diff = M * (qk + av) if cfg.diff_impl == "sdpa" else qk + M * av
        attn = (1 - f) * vanilla_attn + f * diff
    if cfg.ffn == "hybrid":
        ffn = 2 * Tt * D * 2 * H + 2 * P * H * 9 + 2 * Tt * H * D
    else:
        ffn = 2 * Tt * D * H * 2
    block = qkv_proj + attn + ffn
    block_vanilla = qkv_proj + vanilla_attn + ffn
    embed = 2 * P * cfg.in_chans * cfg.patch_size ** 2 * D
    head = 2 * D * cfg.num_classes
    total = embed + head + cfg.core_depth * S * block
    return {"steps": S, "gflops_per_block": block / 1e9,
            "gflops_per_block_vanilla": block_vanilla / 1e9,
            "block_overhead": block / block_vanilla - 1.0,
            "gflops_embed_head": (embed + head) / 1e9,
            "gflops_total": total / 1e9}


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _fmt(n: int) -> str:
    return f"{n:,} ({n / 1e6:.2f}M)"


def variant_name(cfg: LoopViTConfig) -> str:
    if cfg.diffusion == "none":
        name = "LoopViT (vanilla)"
    else:
        name = f"LDGA-{cfg.diffusion}" + (" (per_step)" if cfg.diff_schedule == "per_step" else "")
    return name + (" + loop_relax" if cfg.loop_relax else "")


def print_model_summary(model: LoopViT, batch_size: int = 1, device="cpu"):
    cfg = model.cfg
    line = "=" * 78
    print(line)
    print(f" {variant_name(cfg)} -- LoopViT (arXiv:2602.02156) for classification")
    print(line)
    print(f" image {cfg.image_size}x{cfg.image_size}, patch {cfg.patch_size} -> "
          f"{model.patch_embed.num_patches} patches + {cfg.num_cls_tokens} CLS token(s)")
    print(f" dim {cfg.dim}, heads {cfg.num_heads}, mlp_ratio {cfg.mlp_ratio}, "
          f"classes {cfg.num_classes}, pool '{cfg.pool}'")
    print(f" recurrent core: B={cfg.core_depth} hybrid blocks x T={cfg.loop_steps} "
          f"iterations = {cfg.core_depth * cfg.loop_steps} block applications")
    unrolled = " | ".join(
        f"t={t + 1}: apps {t * cfg.core_depth + 1}-{(t + 1) * cfg.core_depth}"
        for t in range(cfg.loop_steps))
    print(f" unrolled: {unrolled}  (every iteration reuses blocks 1-{cfg.core_depth})")
    print(f" ffn={cfg.ffn}, rope={cfg.rope}, step_embedding={cfg.step_embedding}, "
          f"drop_path={cfg.drop_path}")
    if cfg.diffusion != "none":
        extra = {"ppr": f"alpha_init={cfg.ppr_alpha_init}", "heat": f"tau_init={cfg.heat_tau_init}",
                 "gpr": f"gpr_init={cfg.gpr_init}"}[cfg.diffusion]
        print(f" LDGA: diffusion={cfg.diffusion}, hops M={cfg.diff_hops}, heads={cfg.diff_heads}, "
              f"schedule={cfg.diff_schedule}, renorm={cfg.diff_renorm}, impl={cfg.diff_impl}, {extra}")
    print(f" loop_relax={cfg.loop_relax}")
    print(f" dynamic exit: mode={cfg.exit_mode}, tau={cfg.exit_tau} nats, "
          f"fp_eps={cfg.exit_fp_eps}, min_steps={cfg.min_loop_steps}, "
          f"max_steps={cfg.max_loop_steps or cfg.loop_steps}")
    print(line)
    for k, v in model.param_report().items():
        print(f" {k:<32} {_fmt(v) if 'params' in k and not k.startswith(('ldga', 'relax')) else v}")
    fl = model.flops()
    print(f" {'GFLOPs / image (analytic)':<32} {fl['gflops_total']:.3f} "
          f"(block {fl['gflops_per_block']:.3f}, {fl['block_overhead'] * 100:+.1f}% vs vanilla block)")
    print(line)

    try:
        from torchinfo import summary
        summary(model,
                input_size=(batch_size, cfg.in_chans, cfg.image_size, cfg.image_size),
                depth=2, device=device,
                col_names=("input_size", "output_size", "num_params"),
                row_settings=("var_names",))
        print(" Note: '(recursive)' rows are the same block run again on a later\n"
              " iteration; its parameters are counted once, its compute every time.")
    except ImportError:
        print(" (pip install torchinfo for the layer-by-layer table)")
        print(model)
    print(line)
