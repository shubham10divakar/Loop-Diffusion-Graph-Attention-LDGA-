"""
Design doc C, sec. 10: LDGA tests (pytest, CPU, tiny config).

    python -m pytest tests -q
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import loop_vit_reference as ref
from ldga_stats import (attention_spectrum, dirichlet_energy_grid, effective_rank,
                        frequency_response, rel_state_change, spectral_gap)
from loop_vit import LoopViT, LoopViTConfig, flop_report

TINY = dict(image_size=32, patch_size=8, dim=64, num_heads=4, core_depth=2,
            loop_steps=3, num_classes=5, diff_hops=3)
REF_TINY = {k: v for k, v in TINY.items() if k != "diff_hops"}
MODES = ["ppr", "heat", "gpr"]


def random_head(model):
    """The head is zero-initialised, so every logit would be 0 and output
    comparisons would pass trivially. Give it deterministic random weights."""
    g = torch.Generator().manual_seed(123)
    with torch.no_grad():
        model.head.weight.copy_(torch.randn(model.head.weight.shape, generator=g) * 0.5)
        model.head.bias.copy_(torch.randn(model.head.bias.shape, generator=g) * 0.1)
    return model


def make(**kw) -> LoopViT:
    torch.manual_seed(0)
    return random_head(LoopViT(LoopViTConfig(**{**TINY, **kw})).eval())


def randomize_coeffs(model: LoopViT, seed: int = 1, scale: float = 0.7):
    """Move every coefficient parameter away from its init."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for blk in model.core:
            for p in blk.attn.coeffs.parameters():
                p.add_(torch.randn(p.shape, generator=g).to(p.device) * scale)


def x_batch(n=6, seed=0):
    return torch.randn(n, 3, 32, 32, generator=torch.Generator().manual_seed(seed))


def tokens(n=4, T=17, seed=3):
    return torch.randn(n, T, TINY["dim"], generator=torch.Generator().manual_seed(seed))


def vanilla_ref():
    torch.manual_seed(0)
    return random_head(ref.LoopViT(ref.LoopViTConfig(**REF_TINY)).eval())


# --------------------------------------------------------------------------- #
def test_t1_baseline_equivalence():
    old = vanilla_ref()
    new = LoopViT(LoopViTConfig(**TINY, diffusion="none")).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    x = x_batch()
    with torch.no_grad():
        assert torch.equal(old(x), new(x))
        for a, b in zip(old.logits_per_step(x, 5), new.logits_per_step(x, 5)):
            assert torch.equal(a, b)
        do, dn = old.dynamic_forward(x, tau=0.5), new.dynamic_forward(x, tau=0.5)
        assert torch.equal(do["exit_steps"], dn["exit_steps"])
        assert torch.allclose(do["logits"], dn["logits"], atol=1e-6)


@pytest.mark.parametrize("impl", ["sdpa", "dense"])
def test_t2_gpr_vanilla_equivalence(impl):
    old = vanilla_ref()
    new = LoopViT(LoopViTConfig(**TINY, diffusion="gpr", gpr_init="vanilla", diff_impl=impl)).eval()
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected and missing and all(".attn.coeffs." in k for k in missing)
    x = x_batch()
    with torch.no_grad():
        assert torch.allclose(old(x), new(x), atol=1e-5)
        assert torch.allclose(old(x, 6), new(x, 6), atol=1e-5)


@pytest.mark.parametrize("renorm", [True, False])
def test_t3_coefficient_formulas(renorm):
    M = TINY["diff_hops"]
    m = torch.arange(M + 1, dtype=torch.float64)
    for mode in ("ppr", "heat"):
        model = make(diffusion=mode, diff_renorm=renorm, ppr_alpha_init=0.3, heat_tau_init=1.7)
        c = model.core[0].attn.coeffs
        th = c(0).double()
        if mode == "ppr":
            al = c.alpha()[0].double()[:, None]
            assert torch.allclose(al, torch.full_like(al, 0.3), atol=1e-6)
            closed = al * (1 - al) ** m
        else:
            tau = c.tau()[0].double()[:, None]
            assert torch.allclose(tau, torch.full_like(tau, 1.7), atol=1e-6)
            closed = torch.exp(-tau) * tau ** m / torch.tensor(
                [math.factorial(int(i)) for i in m], dtype=torch.float64)
        if renorm:
            closed = closed / closed.sum(-1, keepdim=True)
            assert torch.allclose(th.sum(-1), torch.ones(th.shape[0], dtype=torch.float64), atol=1e-6)
        assert torch.allclose(th, closed, atol=1e-6)
    # gpr is never renormalised; the "ppr" init is the renormalised GPR-GNN recipe
    g = make(diffusion="gpr", gpr_init="ppr", ppr_alpha_init=0.2).core[0].attn.coeffs(0)
    closed = 0.2 * 0.8 ** m
    assert torch.allclose(g.double(), (closed / closed.sum()).expand_as(g), atol=1e-6)
    v = make(diffusion="gpr").core[0].attn.coeffs(0)
    assert torch.equal(v, torch.tensor([0.0, 1.0, 0.0, 0.0]).expand_as(v))


@pytest.mark.parametrize("mode", MODES)
def test_t4_sdpa_matches_dense_fp64(mode):
    m = make(diffusion=mode, diff_impl="sdpa")
    randomize_coeffs(m)
    attn = m.core[0].attn
    x = tokens()
    with torch.no_grad():
        q, k, v = attn._qkv(x)
        got = attn._mix(q, k, v, t=0)
        th = attn.coeffs(0).double()
        A = ((q.double() @ k.double().transpose(-2, -1)) * attn.scale).softmax(-1)
        u, want = v.double(), th[:, 0].view(1, -1, 1, 1) * v.double()
        for hop in range(1, TINY["diff_hops"] + 1):
            u = A @ u
            want = want + th[:, hop].view(1, -1, 1, 1) * u
        assert torch.allclose(got.double(), want, atol=1e-4)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("schedule", ["shared", "per_step"])
def test_t5_gradients(mode, schedule):
    m = make(diffusion=mode, diff_schedule=schedule).train()
    F.cross_entropy(m(x_batch()), torch.arange(6) % 5).backward()
    for blk in m.core:
        for name, p in blk.attn.coeffs.named_parameters():
            assert p.grad is not None, name
            # every (step, head) row that was used must receive gradient
            assert (p.grad.flatten(1) if p.ndim > 1 else p.grad).abs().sum(-1).gt(0).all(), name


@pytest.mark.parametrize("mode", MODES)
def test_t6_dense_vs_sdpa(mode):
    a = make(diffusion=mode, diff_impl="sdpa")
    randomize_coeffs(a)
    b = make(diffusion=mode, diff_impl="dense")
    b.load_state_dict(a.state_dict(), strict=True)
    x = x_batch()
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-5)
        assert torch.allclose(a(x, 5), b(x, 5), atol=1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_t7_per_step_extrapolation(mode):
    m = make(diffusion=mode, diff_schedule="per_step")
    randomize_coeffs(m)
    c = m.core[0].attn.coeffs
    assert c.num_steps == TINY["loop_steps"]
    assert not torch.allclose(c(0), c(2))
    for t in range(TINY["loop_steps"], 2 * TINY["loop_steps"]):
        assert torch.equal(c(t), c(TINY["loop_steps"] - 1))
    with torch.no_grad():
        out = m(x_batch(), num_steps=2 * TINY["loop_steps"])
    assert out.shape == (6, 5) and torch.isfinite(out).all()
    # the step index really reaches the attention: shifting t changes the output
    x = tokens()
    with torch.no_grad():
        assert not torch.allclose(m.core[0].attn(x, t=0), m.core[0].attn(x, t=2))
        assert torch.equal(m.core[0].attn(x, t=5), m.core[0].attn(x, t=2))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("impl", ["sdpa", "dense"])
def test_t8_head_subset(mode, impl):
    m = make(diffusion=mode, diff_heads=1, diff_impl=impl)
    randomize_coeffs(m)
    attn = m.core[0].attn
    assert attn.coeffs(0).shape == (1, TINY["diff_hops"] + 1)
    x = tokens()
    with torch.no_grad():
        q, k, v = attn._qkv(x)
        got = attn._mix(q, k, v, t=0)
        vanilla = F.scaled_dot_product_attention(q, k, v)
        assert torch.allclose(got[:, 1:], vanilla[:, 1:], atol=1e-6)
        assert not torch.allclose(got[:, :1], vanilla[:, :1], atol=1e-3)
        th = attn.head_theta(0)
        assert torch.equal(th[1:], torch.tensor([0.0, 1, 0, 0]).expand(3, 4))


@pytest.mark.parametrize("diffusion", ["none", "gpr"])
def test_t9_relax_identity(diffusion):
    a = make(diffusion=diffusion)
    if diffusion != "none":
        randomize_coeffs(a)
    b = make(diffusion=diffusion, loop_relax=True)
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=False)
    assert missing == ["eta_raw"] and not unexpected
    assert torch.allclose(b.eta_values(), torch.ones(TINY["loop_steps"]), atol=1e-6)
    x = x_batch()
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-5)
        assert torch.allclose(a(x, 6), b(x, 6), atol=1e-5)
    # eta != 1 actually damps the loop
    with torch.no_grad():
        b.eta_raw.fill_(-1.0)
        assert not torch.allclose(a(x), b(x), atol=1e-3)


@pytest.mark.parametrize("diffusion", ["none"] + MODES)
@pytest.mark.parametrize("exit_mode", ["entropy", "fixedpoint", "both", "either"])
@pytest.mark.parametrize("relax", [False, True])
def test_t10_never_exit_equals_forward(diffusion, exit_mode, relax):
    m = make(diffusion=diffusion, exit_mode=exit_mode, loop_relax=relax)
    if diffusion != "none":
        randomize_coeffs(m)
    if relax:
        with torch.no_grad():
            m.eta_raw.copy_(torch.tensor([0.3, -0.5, 1.2]))
    x = x_batch()
    with torch.no_grad():
        d = m.dynamic_forward(x, tau=-1.0, fp_eps=-1.0, max_steps=5)
        ref_logits = m(x, num_steps=5)
    assert torch.allclose(d["logits"], ref_logits, atol=1e-6)
    assert d["steps_run"] == 5 and (d["exit_steps"] == 5).all()
    assert d["block_apps"] == 6 * 5 * TINY["core_depth"]
    assert torch.isnan(d["fp_trace"][:, 0]).all()
    assert torch.isfinite(d["fp_trace"][:, 1:]).all()


@pytest.mark.parametrize("diffusion", ["none"] + MODES)
def test_t11_compaction(diffusion):
    m = make(diffusion=diffusion, loop_relax=diffusion == "gpr")
    if diffusion != "none":
        randomize_coeffs(m)
    x = x_batch(8)
    # half the samples exit at step 1, the rest run the full budget
    tau = torch.tensor([1e9, -1, 1e9, -1, 1e9, -1, 1e9, -1.0])
    with torch.no_grad():
        d = m.dynamic_forward(x, tau=tau, max_steps=4)
        expect = torch.tensor([1, 4, 1, 4, 1, 4, 1, 4])
        assert torch.equal(d["exit_steps"], expect)
        for i in range(8):
            one = m(x[i:i + 1], num_steps=int(expect[i]))
            assert torch.allclose(d["logits"][i:i + 1], one, atol=1e-5), i
    assert d["block_apps"] == int(expect.sum()) * TINY["core_depth"]
    # frozen samples keep NaN state changes after exiting
    assert torch.isnan(d["fp_trace"][0, 1:]).all()


def test_t11_fixedpoint_exit_uses_state_change():
    m = make(diffusion="gpr", exit_mode="fixedpoint")
    randomize_coeffs(m)
    x = x_batch()
    with torch.no_grad():
        _, hs = m.logits_and_states(x, 4)
        rc = torch.stack([rel_state_change(hs[t + 1], hs[t]) for t in range(1, 4)], 1)
        eps = float(rc[:, 0].median())
        d = m.dynamic_forward(x, fp_eps=eps, max_steps=4)
    # never exits at step 1 (undefined at t=0); exits at step 2 iff change < eps
    assert (d["exit_steps"] >= 2).all()
    assert torch.equal(d["exit_steps"] == 2, rc[:, 0] < eps)
    assert torch.allclose(d["fp_trace"][:, 1], rc[:, 0], atol=1e-5)


@pytest.mark.parametrize("diffusion", ["none"] + MODES)
@pytest.mark.parametrize("impl", ["sdpa", "dense"])
def test_t11_low_precision(diffusion, impl):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.bfloat16
    m = make(diffusion=diffusion, diff_impl=impl, loop_relax=True).to(dev)
    if diffusion != "none":
        randomize_coeffs(m, scale=2.0)
    x = x_batch().to(dev) * 50          # large activations stress the logits
    with torch.no_grad(), torch.autocast(device_type=dev, dtype=dtype):
        logits = m(x, 6)
        d = m.dynamic_forward(x, max_steps=6, exit_mode="either")
    assert torch.isfinite(logits).all() and torch.isfinite(d["logits"]).all()
    assert torch.isfinite(d["fp_trace"][:, 1:].nan_to_num(0.0)).all()


@pytest.mark.parametrize("impl", ["sdpa", "dense"])
def test_t12_high_pass_kills_constants(impl):
    m = make(diffusion="gpr", diff_impl=impl)
    attn = m.core[0].attn
    with torch.no_grad():
        attn.coeffs.theta.copy_(torch.tensor([1.0, -1.0, 0.0, 0.0]).expand_as(attn.coeffs.theta))
        x = tokens(T=17)[:, :1].expand(-1, 17, -1).contiguous()   # same token everywhere
        q, k, v = attn._qkv(x)
        assert torch.allclose(v, v[:, :, :1].expand_as(v))        # V constant across tokens
        out = attn._mix(q, k, v, t=0)                             # (I - A) V
        assert out.abs().max() < 1e-5
        # vanilla (0, 1) keeps the constant
        attn.coeffs.theta.copy_(torch.tensor([0.0, 1.0, 0.0, 0.0]).expand_as(attn.coeffs.theta))
        assert torch.allclose(attn._mix(q, k, v, t=0), v, atol=1e-5)


# --------------------------------------------------------------------------- #
# extras
# --------------------------------------------------------------------------- #
def test_config_validation():
    with pytest.raises(ValueError):
        make(diffusion="gpr", attn_dropout=0.1)
    make(diffusion="none", attn_dropout=0.1)               # vanilla keeps attention dropout
    for bad in (dict(diff_hops=0), dict(diff_heads=0), dict(diff_heads=5),
                dict(diffusion="bogus"), dict(diff_schedule="x"), dict(diff_impl="x"),
                dict(gpr_init="x"), dict(exit_mode="graph"), dict(ppr_alpha_init=1.0),
                dict(heat_tau_init=0.0)):
        with pytest.raises(ValueError):
            make(**{"diffusion": "gpr", **bad})


def test_param_report_and_values():
    M1, h, B, T = TINY["diff_hops"] + 1, TINY["num_heads"], TINY["core_depth"], TINY["loop_steps"]
    assert make().param_report()["ldga_params"] == 0
    assert make().diffusion_values() is None and make().eta_values() is None
    rep = make(diffusion="gpr", diff_schedule="per_step", loop_relax=True).param_report()
    assert rep["ldga_params"] == B * T * h * M1 and rep["relax_params"] == T
    assert make(diffusion="ppr").param_report()["ldga_params"] == B * h
    assert make(diffusion="heat", diff_heads=2).param_report()["ldga_params"] == B * 2
    vals = make(diffusion="ppr", diff_schedule="per_step").diffusion_values()
    assert vals["theta"].shape == (B, T, h, M1) and vals["alpha"].shape == (B, T, h)
    assert torch.allclose(vals["theta_sum"], torch.ones(B, T, h), atol=1e-6)
    assert "tau" in make(diffusion="heat").diffusion_values()


def test_flop_report():
    base = dict(image_size=224, patch_size=16, dim=384, num_heads=6, core_depth=4, loop_steps=3)
    v = flop_report(LoopViTConfig(**base))
    assert v["block_overhead"] == 0.0
    assert 0.9 < v["gflops_per_block"] < 1.1                    # ~0.23 + 0.70 + 0.06 (doc 7.3)
    s = flop_report(LoopViTConfig(**base, diffusion="gpr", diff_hops=3))
    d = flop_report(LoopViTConfig(**base, diffusion="gpr", diff_hops=3, diff_impl="dense"))
    assert 0.09 < s["block_overhead"] < 0.2 and 0.0 < d["block_overhead"] < s["block_overhead"]


def test_record_attention_and_stats():
    m = make(diffusion="gpr", diff_heads=2)
    randomize_coeffs(m)
    m.set_record_attention(True)
    with torch.no_grad():
        m(x_batch())
    attn = m.core[0].attn
    A = attn.last_A
    assert A.shape == (6, 4, 17, 17)
    assert torch.allclose(A.sum(-1), torch.ones(6, 4, 17), atol=1e-5)
    assert attn.last_theta.shape == (4, 4)
    eig = attention_spectrum(A[:1, :1])
    assert torch.allclose(eig[..., 0].abs(), torch.ones(1, 1, dtype=torch.float64), atol=1e-8)
    gap = spectral_gap(eig)
    assert (gap >= -1e-9).all() and (gap <= 1 + 1e-9).all()
    m.set_record_attention(False)
    assert attn.last_A is None
    # frequency response: vanilla = identity line, Laplacian (1, -1) = 1 - lambda
    lam = torch.linspace(-1, 1, 5)
    assert torch.allclose(frequency_response(torch.tensor([0.0, 1, 0, 0]), lam), lam.double())
    assert torch.allclose(frequency_response(torch.tensor([[1.0, -1]]), lam)[0], 1 - lam.double())
    # oversmoothing measures: identical tokens -> zero energy, rank ~1 (degenerate)
    z = torch.randn(2, 16, 8)
    assert (dirichlet_energy_grid(z, 4) > 0).all()
    assert torch.allclose(dirichlet_energy_grid(z[:, :1].expand(-1, 16, -1), 4), torch.zeros(2), atol=1e-6)
    assert (effective_rank(z) > 1).all()


def test_block_hook_counts_applications():
    m = make(diffusion="heat")
    seen = []
    with torch.no_grad():
        h = m.embed(x_batch())
        for t in range(4):
            h = m.advance(h, t, block_hook=lambda b, hh: seen.append((t, b)))
    assert seen == [(t, b) for t in range(4) for b in range(TINY["core_depth"])]
