# Design Doc C — Loop Diffusion Graph Attention (LDGA) for LoopViT

> **Paper working title:** *Looping as Diffusion: Learnable Spectral Filters on Attention Graphs in Looped Vision Transformers*
> **Method name:** LDGA (Loop Diffusion Graph Attention)
> **Base code:** `loopvit.py` (LoopViT port of arXiv:2602.02156 for image classification) and `train.py`
> **Audience:** a coding agent (CLI) implementing this, plus the author writing the paper.
> **Independent of Doc A (LCGA) and Doc B (LRGA).** Build it in its own branch or repo.

---

## 0. Instructions for the coding agent (read first)

1. **Do not change baseline behaviour.** With `diffusion="none"` and `loop_relax=False` (defaults), outputs must be **exactly** those of the current `loopvit.py` for the same weights (test T1).
2. Put all new behaviour behind `LoopViTConfig` fields. Don't fork the model class.
3. In the `dense` implementation, compute attention matrices in **fp32** with autocast disabled. Cast back only for the final matmul with `V`.
4. `forward`, `logits_per_step`, `dynamic_forward`, `param_report` and `print_model_summary` must keep working.
5. Work in the phases in Section 9, in order, and run the tests after each phase.

---

## 1. Motivation

A softmax attention matrix `A` is **row-stochastic**: every row sums to 1, so `A·1 = 1`. Applying `A` to values is therefore a **one-hop, low-pass graph filter**. It averages each token with its neighbours and damps high-frequency differences between tokens. Stacking many of these filters is a known route to **oversmoothing / rank collapse**, where tokens become near-identical ("attention loses rank", and ViT attention analysed as a low-pass filter).

LoopViT is **especially exposed**: it applies *the same* filter family `B·T` times (12 by default, more when you extrapolate `T`). One fixed hop per application, repeated with tied weights, is the worst case for smoothing.

LDGA replaces the one-hop `A·V` with a **learnable polynomial graph filter** over the attention graph:

```
Attn_LDGA(V) = Σ_{m=0}^{M} θ_m · A^m V                                             (C1)
```

and reads the loop as **discretised diffusion time**. Each block application diffuses information over the content graph `A`, and the coefficients `θ` control *how*: low-pass (PPR, heat kernel) or any learned shape, including **high-pass / anti-smoothing** components (negative θ).

Why this suits LoopViT specifically:

- The loop already *is* an iterative dynamical system with tied weights, so a diffusion/ODE reading is natural, not bolted on.
- Only `(M+1)` scalars per head per block are added (e.g. 4 × 6 × 4 = 96 parameters).
- The learned `θ` are directly interpretable as a **frequency response**. That gives the paper its central figure: *what filter does a looped ViT learn, and does it change over loop steps?*
- A new **fixed-point exit** follows from the diffusion reading: stop when the state reaches a steady state (Section 6).

---

## 2. Formulation

Notation: `N` batch, `h` heads, `T_tok = P + n_prefix`, `d` head dim, `M` hops (`diff_hops`), `t` loop step.

For each head, `A = softmax(QKᵀ/√d)` (RoPE applied as now) is the **attention graph** of that block application. LDGA keeps `Q, K, V`, the output projection and the rest of the block unchanged, and only replaces `A·V` with eq. C1.

### 2.1 Filter families (`diffusion`)

| Mode | Coefficients θ_m | Learnable | Character |
|---|---|---|---|
| `none` | θ = (0, 1, 0, …) | — | vanilla (1-hop low-pass) |
| `ppr` | `α(1−α)^m`, α = σ(a) | `a` per head | personalised PageRank, low-pass |
| `heat` | `e^{−τ} τ^m / m!`, τ = softplus(s) | `s` per head | heat kernel `exp(−τ(I−A))`, low-pass |
| `gpr` | free `θ_m ∈ ℝ` | all `θ_m` per head | general polynomial filter (GPR-GNN style), can be high-pass |

**Truncation renormalisation** (`diff_renorm=True`, default for `ppr`/`heat`): replace θ_m with θ_m / Σθ, so the truncated filter still preserves constants (`Σθ = 1` ⇒ the output is an affine combination). Don't renormalise `gpr`, whose freedom is the point; log `Σθ` instead.

**`gpr` initialisation** (`gpr_init`):
- `"vanilla"` (default): θ = (0, 1, 0, …, 0). **The model is exactly vanilla LoopViT at init.** This is the safe start and makes T2 possible.
- `"ppr"`: θ_m = α(1−α)^m with α = `ppr_alpha_init`, renormalised (the GPR-GNN recipe).

### 2.2 Hop 0 and the residual

The θ_0·V term is a "stay at yourself" term inside attention. It is distinct from the block's residual (`x + Attn(...)`), because V is projected. Keep it: it is what lets `gpr` express high-pass filters, e.g. θ = (1, −1) ⇒ `(I − A)V`, a graph Laplacian.

### 2.3 Loop-time schedule (`diff_schedule`)

- `"shared"` (default): one coefficient set per (block, head), reused at every loop step.
- `"per_step"`: a table indexed by `min(t, T_train − 1)` (identity extrapolation, like `_step_vector`). Shapes: `gpr` → `(T_train, h, M+1)`, `ppr`/`heat` → `(T_train, h)`. This lets the model learn, say, *low-pass early and sharpening later*, which becomes a figure.

### 2.4 Head subset (`diff_heads`)

`diff_heads = -1` (default) applies LDGA to all heads. `k > 0` applies it to the first `k` heads only; the others stay 1-hop. This is useful if full diffusion hurts.

### 2.5 Optional: loop relaxation (`loop_relax`, the ODE step size)

Reading the loop as an explicit Euler step suggests a learnable damping per step:

```
z_{t+1} = z_t + η_t · ( M_θ(z_t + e_t) − z_t ),     η_t = softplus-param, init so η_t = 1      (C2)
```

With η = 1 this is exactly the current loop. `η_t` uses a table of size `T_train` with identity extrapolation. It's an **ablation**, not the core contribution. Report whether η < 1 is learned (damped / implicit-like dynamics) and whether it helps extrapolation.

---

## 3. Config additions (`LoopViTConfig`)

```python
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
```

Validation: `diff_hops >= 1`. `diff_heads in {-1} ∪ [1, num_heads]`. `gpr_init="vanilla"` requires `diff_hops >= 1`. Raise if `attn_dropout > 0` with `diffusion != "none"` **unless** you implement per-hop dropout (see 4.1). Keep it simple: raise.

---

## 4. Code changes

### 4.1 Filter coefficients module

```python
class DiffusionCoeffs(nn.Module):
    """Produces theta of shape (h, M+1) for step t (eq. C1, Section 2.1/2.3)."""
    def __init__(self, mode, heads, hops, schedule, steps, renorm, alpha0, tau0, gpr_init): ...
    def forward(self, t: int) -> torch.Tensor: ...   # fp32 (h, M+1)
```
- `ppr`: parameter `a` with `σ(a) = α0` → θ_m = α(1−α)^m.
- `heat`: parameter `s` with `softplus(s) = τ0` → θ_m = e^{−τ} τ^m / m!. Compute `m!` via `torch.lgamma(m+1)` in log space.
- `gpr`: parameter `theta` directly.
- `per_step` adds a leading `steps` dimension; index `min(t, steps−1)`.

### 4.2 Attention

```python
def forward(self, x, t: int = 0, return_A: bool = False):
    q, k, v = ...                                    # (N,h,T,d), RoPE as now
    if self.diffusion == "none":
        return <existing SDPA path>
    th = self.coeffs(t)                              # (h, M+1) fp32
    hd = self.diff_heads                             # heads [0:hd] diffuse, rest vanilla
    ...
```

**`diff_impl="sdpa"`** (default, flash-friendly, never materialises `A`). `A^m V` is computed by feeding the previous hop's output back in as the *values* of the same attention:

```python
u = v_d                                               # hop 0: (N,hd,T,d)
out = th[:, 0].view(1, hd, 1, 1) * u
for m in range(1, M + 1):
    u = F.scaled_dot_product_attention(q_d, k_d, u)   # A @ u
    out = out + th[:, m].view(1, hd, 1, 1).to(u.dtype) * u
```
Each hop recomputes `QKᵀ` inside SDPA. That's cheap at 197 tokens (Section 7.3).

**`diff_impl="dense"`** (analysis, exact spectra):
```python
with torch.autocast(device_type=x.device.type, enabled=False):
    A = (q_d.float() @ k_d.float().transpose(-2, -1) * scale).softmax(-1)   # (N,hd,T,T)
    u = v_d.float(); out = th0 * u
    for m in 1..M: u = A @ u; out = out + th_m * u
out = out.to(v.dtype)
```
Return `A` when `return_A=True` (for analysis only).

Concatenate the diffused heads with the vanilla heads (the remaining `h − hd`, via the existing SDPA call) along the head dimension, then apply the output projection as now.

### 4.3 HybridBlock / LoopViT wiring

- `HybridBlock.forward(x, dp, t=0)` passes `t` down to `Attention` (needed for `per_step`).
- `core_step(h, t, rates)` passes `t` to each block.
- **Loop relaxation** (if `loop_relax`), inside `loop()` and `dynamic_forward`:
  ```python
  h_new = self.core_step(h, t, rates)
  eta = self._eta(t)                 # softplus param table, init softplus^{-1}(1)
  h = h + eta * (h_new - h)
  ```
  Parameters: `self.eta_raw = nn.Parameter(torch.full((loop_steps,), math.log(math.e - 1)))`, so softplus gives 1.0.

### 4.4 `param_report`

Add `"ldga_params"` (the coefficient parameters across blocks) and `"relax_params"`.

---

## 5. Diagnostics (`ldga_stats.py`)

Pure functions shared by training logs and the analysis script:

- `frequency_response(theta, lam) -> g(λ) = Σ θ_m λ^m`, evaluated on `λ ∈ [−1, 1]` (nominal response), per block/head/step.
- `attention_spectrum(A)`: eigenvalues of `A` (dense impl) for a handful of images, `torch.linalg.eigvals` on 197×197 in fp64. `A` is non-symmetric, so eigenvalues are complex. Plot `|λ|` and report the spectral gap `1 − |λ_2|`. Frame the response plot as *nominal* (it evaluates `g` on the real line) and overlay the actual `|λ|` distribution.
- `dirichlet_energy_grid(z)`: mean `‖x̂_i − x̂_j‖²` over 4-neighbour patch pairs, `x̂ = x/‖x‖`. This matches Docs A/B so all three papers use comparable measures.
- `effective_rank(z)`: `exp(H(σ/Σσ))` of the centred patch-token matrix.
- `rel_state_change(z_t, z_{t−1}) -> (N,)`: `‖z_t − z_{t−1}‖_F / ‖z_{t−1}‖_F`, per sample, used for the exit.

---

## 6. Dynamic exit with a fixed-point criterion

The diffusion view suggests a steady state: stop iterating once the state no longer moves.

```
ent_ok = H_t < tau
fp_ok  = (t >= 1) & (rel_state_change(z_t, z_{t−1}) < exit_fp_eps)
exit_mode: entropy | fixedpoint | both (AND) | either (OR);   only once t+1 >= min_steps
```

The fixed-point criterion works for **vanilla** LoopViT as well, and that is a useful baseline row in the exit Pareto figure.

Also **fix the known compute waste** in `dynamic_forward`. It currently runs `core_step` on samples that have already exited. Run only the active subset:

```python
active = (~done).nonzero(as_tuple=True)[0]
h_prev_a = h[active]
h_new_a = self.core_step(h_prev_a, t)
if self.cfg.loop_relax: h_new_a = h_prev_a + eta_t * (h_new_a - h_prev_a)
h = h.clone(); h[active] = h_new_a
```
Return `"fp_trace"` `(N, steps_run)` (NaN at t=0) and `"block_apps"` (total block applications actually run, summed over samples).

---

## 7. Training-script changes (`train.py`)

### 7.1 Flags
`--diffusion --diff-hops --diff-heads --diff-schedule --diff-renorm --diff-impl --ppr-alpha-init --heat-tau-init --gpr-init --loop-relax --exit-mode --exit-fp-eps`.

### 7.2 Logging (once per epoch, on validation)
- accuracy per step (T = 1..T_train) and when extrapolating (T up to 2·T_train);
- θ per block/head (/step), `Σθ`, and for ppr/heat the effective α/τ;
- Dirichlet energy and effective rank after each loop step;
- relative state change per step;
- η_t if `loop_relax`;
- throughput, peak memory, and **measured FLOPs** (fvcore/ptflops, or the analytic count in 7.3).

### 7.3 Compute accounting (put this in the paper)
Rough per-block FLOPs at 224/16, D=384, hidden=1536, 197 tokens:
- QKV + out-proj: `197·384·384·4·2 ≈ 0.23 GFLOP`
- ConvGLU fc1 (384→3072) + fc2 (1536→384): `≈ 0.47 + 0.23 = 0.70 GFLOP`
- attention `QKᵀ` + `AV`: `≈ 2·197²·384·2 ≈ 0.06 GFLOP`

Each extra hop adds about 0.06 GFLOP with `sdpa` (QKᵀ recomputed) or about 0.03 with `dense`, i.e. **≈ 3–6% per hop**. So `M=3` costs about +12–18% block FLOPs. The paper must include a **compute-matched baseline**: vanilla LoopViT at a slightly larger `T`, or a slightly wider `dim`, with equal FLOPs.

---

## 8. Analysis script (`analyze_ldga.py`)

1. **Learned frequency responses** `g(λ)` per block (and per step, if `per_step`). Central figure: is the learned filter low-pass, band-pass or high-pass, and does it change across loop steps?
2. **θ heatmaps** (blocks × hops, per head).
3. **Oversmoothing curves:** Dirichlet energy and effective rank against unrolled depth (1..B·T, extended to 2·T_train), for vanilla vs ppr vs heat vs gpr.
4. **Attention spectra:** `|λ|` histograms of `A` per step (dense impl), and the spectral gap over steps.
5. **Extrapolation curve:** accuracy against inference `T` from 1 to 2·T_train, per variant.
6. **Exit Pareto:** accuracy vs mean `block_apps` for entropy / fixedpoint / both / either, sweeping `tau` and `exit_fp_eps`.
7. **CLS attention after diffusion:** effective CLS→patch weights `Σθ_m (A^m)[0,:]` (dense impl) reshaped to `g×g`, per step, for 8 fixed images.
8. If `loop_relax`: learned η_t.

---

## 9. Implementation phases (for the CLI)

| Phase | Work | Done when |
|---|---|---|
| 0 | Thread `t` through `core_step` → `HybridBlock` → `Attention`; `diffusion="none"` path untouched | T1 passes (bit-identical to original) |
| 1 | `DiffusionCoeffs` (ppr / heat / gpr, shared schedule) + `sdpa` impl, all heads | T2–T5 pass |
| 2 | `dense` impl, `per_step` schedule, `diff_heads` subset | T6–T8 pass |
| 3 | `loop_relax`, fixed-point exit, batch compaction in `dynamic_forward` | T9–T11 pass |
| 4 | `train.py` flags, logging, FLOP counting | short training run end to end for each mode |
| 5 | `ldga_stats.py` + `analyze_ldga.py` | all eight outputs listed in Section 8 produced |
| 6 (optional) | Single-file `kaggle_ldga.py` export | runs in a fresh Kaggle notebook |

---

## 10. Tests (`tests/test_ldga.py`, pytest, CPU, tiny config)

Tiny config: `image_size=32, patch_size=8, dim=64, num_heads=4, core_depth=2, loop_steps=3, num_classes=5, diff_hops=3`.

- **T1 baseline equivalence:** `diffusion="none"` matches the original `loopvit.py` exactly after copying the state dict (`strict=True`).
- **T2 gpr-vanilla equivalence:** `diffusion="gpr", gpr_init="vanilla"`. Load vanilla weights (`strict=False`; the only missing keys are the coefficients). Outputs equal vanilla to `atol=1e-5`.
- **T3 coefficient formulas:** ppr and heat θ match the closed forms, and with `diff_renorm` they sum to 1 (`atol=1e-6`).
- **T4 hop correctness:** the `sdpa` impl equals an explicit `Σθ_m A^m V` computed densely in fp64 (`atol=1e-4`).
- **T5 gradients:** after one backward, every coefficient parameter has a non-zero gradient.
- **T6 dense vs sdpa:** both implementations agree (`atol=1e-5`) for all three modes.
- **T7 per_step + extrapolation:** `model(x, num_steps=2*loop_steps)` runs, and the coefficients used past `T_train` equal those of the last trained step.
- **T8 head subset:** with `diff_heads=1`, heads 1..h−1 equal the vanilla computation.
- **T9 relax identity:** `loop_relax=True` at init (η=1) gives the same outputs as `loop_relax=False`.
- **T10 exit equivalence:** `dynamic_forward` with `tau=-1`, `exit_fp_eps=-1` equals `forward(x, max_steps)` in eval mode.
- **T11 compaction + fp16:** the compacted run equals a per-sample reference and `block_apps` is correct. No NaN/Inf under autocast (CUDA fp16 if available, else CPU bf16).
- **T12 high-pass sanity:** with gpr θ = (1, −1, 0, 0) and a constant `V` across tokens, the attention output is ≈ 0 (the Laplacian kills constants).

---

## 11. Experiments for the paper

### 11.1 Configs
- **Small:** `32/4` (64 patches), `dim=192, heads=6, B=4, T=3`, `M ∈ {2,3,4}`. CIFAR-10/100.
- **Base:** `224/16`, `dim=384, heads=6, B=4, T=3`, `M=3`. Tiny-ImageNet plus PlantVillage, PlantDoc, FGVC Plant Pathology.
- From scratch, identical schedule, 3 seeds, mean ± std, paired tests.

### 11.2 Main comparison
| Model | Purpose |
|---|---|
| LoopViT vanilla, T=3 | baseline |
| LoopViT vanilla, compute-matched (larger T or dim) | rules out "just more FLOPs" |
| LDGA-ppr | fixed-shape low-pass diffusion |
| LDGA-heat | fixed-shape low-pass diffusion |
| **LDGA-gpr (per_step)** | proposed: learned filter with a loop-time schedule |
| LDGA-gpr (shared) | isolates the value of the step schedule |
| Untied ViT, B·T layers | reference |

### 11.3 Ablations
`M` · `diff_heads` · `gpr_init` (vanilla vs ppr) · `diff_renorm` on/off for ppr/heat · `loop_relax` on/off · `ffn=vanilla` (does diffusion matter more without the ConvGLU's local prior?) · train `T ∈ {2,3,4}` with test-time extrapolation.

### 11.4 Hypotheses (state them before running)
- **H1:** pure low-pass diffusion (ppr/heat) *increases* oversmoothing and does not beat vanilla. LDGA-gpr beats vanilla at matched compute.
- **H2:** learned gpr filters develop **negative / high-pass components**, stronger at later loop steps (`per_step`). The model learns to counteract the smoothing the loop induces.
- **H3:** LDGA-gpr keeps Dirichlet energy and effective rank higher across unrolled depth, and degrades less when extrapolating to T > T_train.
- **H4:** the fixed-point exit (alone or with entropy) matches or beats entropy-only on the accuracy vs `block_apps` curve.

Note: H1's first half predicts a *negative* result for ppr/heat. That's intended, because it motivates the learned filter. Report it as it comes out.

---

## 12. Related work to cite and differentiate (verify each reference before citing)

- **MAGNA (multi-hop attention graph diffusion):** the closest prior method, multi-hop diffusion of attention in GNNs on fixed input graphs. Differentiation: LDGA runs on the *self-attention graph of a looped, weight-tied ViT*, with learnable (incl. high-pass) filters, a loop-time schedule, a fixed-point exit, and extrapolation over T.
- **GPR-GNN, APPNP, GDC (graph diffusion convolution):** polynomial / PPR / heat filters on graphs.
- **GRAND and continuous-depth models (Neural ODE, Deep Equilibrium Models):** diffusion/ODE readings of depth. The fixed-point exit and `loop_relax` connect here.
- **Oversmoothing / rank collapse in transformers:** "attention loses rank" analyses, and ViT attention as a low-pass filter with anti-oversmoothing fixes (Fourier-domain analysis of deep ViTs). Position LDGA as a looped-ViT-specific fix with interpretable filters.
- **LoopViT (arXiv:2602.02156)**, **Universal Transformers / ACT / PonderNet**.
- The author's **IMHA** (iterative attention): situate LDGA as the spectral counterpart.

## 13. Claims to avoid (reviewer-proofing)

- `A` is non-symmetric, so a "frequency response" on `λ ∈ [−1,1]` is **nominal**. Always show it next to the actual eigenvalue distribution, and don't call it an exact spectral decomposition.
- Don't call it "continuous-time" or an "ODE solver" unless `loop_relax` is on, and even then call it an *Euler-style reading*, not a solver.
- Don't claim efficiency. LDGA *adds* ~3–6% FLOPs per hop. Any efficiency claim belongs to the exit mechanism and must come from the Pareto plot.
- Say "attention graph" or "diffusion over the attention graph", not "topology".

## 14. Relationship to Docs A and B

- **A (LCGA)** carries the *graph* across loop steps: temporal memory over edges.
- **B (LRGA)** changes *which* edges exist at each step: structure selection.
- **C (LDGA)** changes *how information spreads* over a given graph: the filter.

The three are orthogonal. Keep each paper to its single claim, and consider a combined model only after all three have standalone results.
