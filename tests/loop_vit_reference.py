"""
LoopViT for image classification.

This is the architecture from "LoopViT: Scaling Visual ARC with Looped
Transformers" (arXiv:2602.02156), ported from per-pixel ARC grid prediction to
whole-image classification. Everything that makes the paper's model what it is
is kept; only the input head (RGB patches instead of a discrete colour grid) and
the output head (one class label instead of 30x30 colours) change.

The three ideas of the paper
----------------------------
1. Weight-tied recurrent core ("scaling Time, not Space").
   A stack of B *distinct* hybrid blocks -- the core trunk M_theta -- is applied
   T times. Compute is B*T block applications, parameters are only B blocks:

        z_0   = emb(x)
        z_t+1 = M_theta(z_t + e_t)          eq. (2) in the paper
        y     = head(z_T)

   e_t is a learned per-step embedding that tells the shared blocks which
   iteration they are on. For inference steps beyond the training budget the
   paper reuses the last one (identity extrapolation), which is what
   `_step_vector` does, so you can run a model trained with T=3 at T=6.

2. Hybrid encoder block (local conv + global attention).
      Z'   = Z + MHSA(RMSNorm(Z))                      eq. (6)
      Zout = Z' + ConvGLU(RMSNorm(Z'))                 eq. (7)
   MHSA uses 2D rotary embeddings; the CLS/task tokens are left un-rotated.
   The FFN is the *Heterogeneous* ConvGLU: the gate branch is split into task
   tokens and image tokens, only the image tokens are reshaped to the 2D grid
   and passed through a 3x3 depth-wise convolution (the "cellular automaton
   update"), then the two are re-assembled:
      [X_gate, X_val] = Linear1(Z)                     eq. (8)
      G_img_hat       = Flatten(DWConv(Reshape(G_img))) eq. (9)
      ConvGLU(Z)      = Linear2(sigma(X_gate_hat) * X_val)  eq. (10)
   Set `ffn: vanilla` to get a plain MLP instead -- that is the paper's
   "Hybrid vs Vanilla" ablation (Fig. 7).

3. Dynamic Exit via entropy crystallization (inference only, no parameters).
   After every iteration the predictive entropy H_t = -sum p log p is measured;
   once H_t < tau the sample's state is frozen (z_k = z_t for k > t) and it
   stops consuming compute. Easy images exit after a couple of iterations, hard
   ones use the full budget. See `dynamic_forward`.

Training follows the paper's fixed-depth protocol: unroll exactly
`loop_steps` iterations, supervise the final output only. `deep_supervision` in
train.py adds a loss on the intermediate read-outs, which is *not* in the paper
and is off by default.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Attention(nn.Module):
    """MHSA with optional 2D rotary embeddings (eq. 3-5)."""

    def __init__(self, dim, num_heads, grid, n_prefix, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0, rope=True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rotary = Rotary2D(self.head_dim, grid, n_prefix) if rope else None

    def forward(self, x):
        N, T, D = x.shape
        q, k, v = (self.qkv(x)
                   .reshape(N, T, 3, self.num_heads, self.head_dim)
                   .permute(2, 0, 3, 1, 4))            # 3 x (N, h, T, d)
        if self.rotary is not None:
            q, k = self.rotary(q), self.rotary(k)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        x = x.transpose(1, 2).reshape(N, T, D)
        return self.proj_drop(self.proj(x))


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
                              cfg.qkv_bias, cfg.attn_dropout, cfg.dropout, cfg.rope)
        self.norm2 = RMSNorm(cfg.dim)
        if cfg.ffn == "hybrid":
            self.ffn = HeteroConvGLU(cfg.dim, hidden, grid, cfg.num_cls_tokens, cfg.dropout)
        elif cfg.ffn == "vanilla":
            self.ffn = MLP(cfg.dim, hidden, cfg.dropout)
        else:
            raise ValueError(f"ffn must be 'hybrid' or 'vanilla', got {cfg.ffn!r}")

    def forward(self, x, dp: float = 0.0):
        x = x + drop_path(self.attn(self.norm1(x)), dp, self.training)
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

        self.norm = RMSNorm(D)
        self.head = nn.Linear(D, cfg.num_classes)
        self._init_weights()

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

    def embed(self, x):
        """z_0 = emb(x)."""
        h = self.patch_embed(x)
        if self.cls_token is not None:
            h = torch.cat([self.cls_token.expand(h.shape[0], -1, -1), h], dim=1)
        return self.pos_drop(h + self.pos_embed)

    def core_step(self, h, t: int, rates=None):
        """One application of the shared core: z_{t+1} = M_theta(z_t + e_t)."""
        e = self._step_vector(t)
        if e is not None:
            h = h + e
        B = len(self.core)
        for b, blk in enumerate(self.core):
            h = blk(h, rates[t * B + b] if rates else 0.0)
        return h

    def loop(self, h, num_steps: int, return_all: bool = False):
        rates = self._drop_path_rates(len(self.core) * num_steps)
        states = []
        for t in range(num_steps):
            h = self.core_step(h, t, rates)
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

    @staticmethod
    def entropy(logits):
        """H_t = -sum_c p log p, in nats (eq. 11, over classes instead of pixels)."""
        logp = F.log_softmax(logits.float(), dim=-1)
        return -(logp.exp() * logp).sum(-1)

    @torch.no_grad()
    def dynamic_forward(self, x, tau: float | None = None, min_steps: int | None = None,
                        max_steps: int | None = None):
        """Dynamic Exit (sec. 3.3). Per sample: iterate until the predictive
        entropy falls below `tau`, then freeze that sample's state so further
        iterations cost it nothing.

        Returns a dict with the final `logits`, the per-sample `exit_steps`,
        the (N, steps_run) `entropy` trace and `steps_run` (the deepest sample).
        """
        cfg = self.cfg
        tau = cfg.exit_tau if tau is None else tau
        min_steps = cfg.min_loop_steps if min_steps is None else min_steps
        max_steps = max_steps or cfg.max_loop_steps or cfg.loop_steps

        h = self.embed(x)
        N = x.shape[0]
        done = torch.zeros(N, dtype=torch.bool, device=x.device)
        exit_steps = torch.full((N,), max_steps, dtype=torch.long, device=x.device)
        trace, logits, t = [], None, 0
        for t in range(max_steps):
            updated = self.core_step(h, t)
            h = torch.where(done.view(-1, 1, 1), h, updated)   # frozen: z_k = z_t
            logits = self.classify(h)
            ent = self.entropy(logits)
            trace.append(ent)
            newly = (~done) & (ent < tau) if (t + 1) >= min_steps else torch.zeros_like(done)
            exit_steps = torch.where(newly, torch.full_like(exit_steps, t + 1), exit_steps)
            done = done | newly
            if bool(done.all()):
                break
        return {"logits": logits, "exit_steps": exit_steps,
                "entropy": torch.stack(trace, dim=1), "steps_run": t + 1}

    # ---- bookkeeping -------------------------------------------------------
    def param_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        per_block = sum(p.numel() for p in self.core[0].parameters())
        core = per_block * len(self.core)
        T = self.cfg.loop_steps
        return {
            "total_params": total,
            "params_per_block": per_block,
            "core_params (shared)": core,
            "other_params (embed/norm/head)": total - core,
            "block_applications": len(self.core) * T,
            "untied_equivalent_params": total + core * (T - 1),
        }


def build_model(**kwargs) -> LoopViT:
    return LoopViT(LoopViTConfig(**kwargs))


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _fmt(n: int) -> str:
    return f"{n:,} ({n / 1e6:.2f}M)"


def print_model_summary(model: LoopViT, batch_size: int = 1, device="cpu"):
    cfg = model.cfg
    line = "=" * 78
    print(line)
    print(" LoopViT (paper architecture, arXiv:2602.02156) for classification")
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
    print(f" dynamic exit: tau={cfg.exit_tau} nats, min_steps={cfg.min_loop_steps}, "
          f"max_steps={cfg.max_loop_steps or cfg.loop_steps}")
    print(line)
    for k, v in model.param_report().items():
        print(f" {k:<32} {_fmt(v) if 'params' in k else v}")
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
