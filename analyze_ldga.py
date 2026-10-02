"""
LDGA analysis (design doc C, sec. 8). Takes one or more checkpoints (e.g. the
vanilla LoopViT and the LDGA runs on the same dataset) and a data split, and
writes figures + JSON to --out-dir:

  1. freq_response_<label>.png  learned nominal response g(lambda) = sum theta_m lambda^m per
                                block (and step), over the actual Re(lambda) distribution of A
  2. theta_<label>.png          theta heatmaps (blocks x hops, per head [and step])
  3. oversmoothing.png          Dirichlet energy and effective rank vs unrolled depth
                                1..B*S (S = 2 T_train by default), all checkpoints
  4. spectra_<label>.png        |lambda| histograms of A per step + spectral gap 1-|lambda_2|
  5. accuracy_per_step.png      extrapolation curve: accuracy vs inference T = 1..2 T_train
  6. exit_pareto.png            accuracy vs mean block applications for entropy / fixedpoint /
                                both / either exits, sweeping tau and exit_fp_eps
  7. cls_maps_<label>.png       effective CLS->patch weights sum_m theta_m (A^m)[0, :] after
                                diffusion (last block, head mean) per step on 8 images
  8. eta.png                    learned eta_t (runs trained with --loop-relax)
  summary.json                  every number behind the figures

    python analyze_ldga.py --ckpt runs/loopvit_plantodc/best.pt runs/ldga_plantodc/best.pt
    python analyze_ldga.py --ckpt runs/ldga_plantodc/best.pt --max-images 500 --out-dir analysis/plantodc

The data split is rebuilt from the arguments stored in the (first) checkpoint,
so the same classes / val split are used; --dataset / --train-dir / --val-dir /
--data-root override them.

The frequency response is *nominal*: A is non-symmetric, so g is evaluated on
the real line and shown next to the real parts of A's actual eigenvalues (doc
sec. 13). Spectra use the dense fp32 attention graph (computed separately, so it
works for sdpa-trained checkpoints) and fp64 eigenvalues.

The exit Pareto is computed offline from one full-depth pass: freezing a sample
at step t (z_k = z_t) means its final logits are exactly its step-t logits, so
every (mode, tau, eps) point is a re-read of the same trajectories.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

from ldga_stats import (attention_spectrum, dirichlet_energy_grid, effective_rank,
                        frequency_response, rel_state_change, spectral_gap)
from loop_vit import LoopViT, LoopViTConfig, variant_name

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------------------- #
# plotting helpers (also used by train.py)
# --------------------------------------------------------------------------- #
def plot_theta_heatmap(theta: torch.Tensor, path: str, title: str = "learned theta"):
    """theta: (B, S, hd, M+1). One panel (blocks x hops) per (step, head)."""
    plt = _plt()
    g = theta.float().numpy()
    B, S, H, M1 = g.shape
    vmax = max(float(abs(g).max()), 1e-6)
    fig, axes = plt.subplots(S, H, figsize=(0.8 + 0.5 * M1 * H, 0.7 + 0.45 * B * S),
                             squeeze=False)
    for s in range(S):
        for h in range(H):
            ax = axes[s, h]
            im = ax.imshow(g[:, s, h, :], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(M1), [str(m) for m in range(M1)], fontsize=7)
            ax.set_yticks(range(B), [f"b{b}" for b in range(B)], fontsize=7)
            if s == 0:
                ax.set_title(f"head {h}", fontsize=8)
            if h == 0 and S > 1:
                ax.set_ylabel(f"step {s + 1}", fontsize=8)
            if s == S - 1:
                ax.set_xlabel("hop m", fontsize=7)
            if B * M1 <= 32:
                for b in range(B):
                    for m in range(M1):
                        ax.text(m, b, f"{g[b, s, h, m]:+.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig.suptitle(title)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_frequency_response(theta: torch.Tensor, path: str, title: str = "g(lambda)",
                            eig_re: dict | None = None):
    """Nominal response g(lambda) on lambda in [-1, 1]: one panel per (step, block),
    one line per head, vanilla g = lambda dashed. `eig_re[b]` (1-D tensor of real
    parts of A's eigenvalues for block b) is drawn as a background histogram."""
    plt = _plt()
    B, S, H, _ = theta.shape
    lam = torch.linspace(-1, 1, 201)
    g = frequency_response(theta, lam).float()                  # (B, S, H, L)
    fig, axes = plt.subplots(S, B, figsize=(3.2 * B, 2.6 * S), squeeze=False, sharex=True)
    for s in range(S):
        for b in range(B):
            ax = axes[s, b]
            if eig_re is not None and b in eig_re and len(eig_re[b]):
                ax2 = ax.twinx()
                ax2.hist(eig_re[b].numpy(), bins=60, range=(-1, 1), color="0.8", alpha=0.7)
                ax2.set_yticks([])
                ax.set_zorder(ax2.get_zorder() + 1)
                ax.patch.set_visible(False)
            for h in range(H):
                ax.plot(lam, g[b, s, h], lw=1.2, label=f"h{h}")
            ax.plot(lam, lam, "k--", lw=0.8, label="vanilla")
            ax.axhline(0, color="k", lw=0.4)
            ax.set_title(f"block {b}" + (f", step {s + 1}" if S > 1 else ""), fontsize=9)
            if s == S - 1:
                ax.set_xlabel("lambda (nominal, real)")
    axes[0, 0].set_ylabel("g(lambda)")
    axes[0, -1].legend(fontsize=6)
    fig.suptitle(title + ("  [grey: Re(eig A)]" if eig_re else ""), y=1.02)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def pareto_front(points):
    """points: list of (cost, acc). Returns the non-dominated subset sorted by cost."""
    front, best = [], -1.0
    for c, a in sorted(points, key=lambda p: (p[0], -p[1])):
        if a > best:
            front.append((c, a))
            best = a
    return front


# --------------------------------------------------------------------------- #
# data collection
# --------------------------------------------------------------------------- #
def load_model(path: str, device, impl: str | None = None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["model_cfg"])
    if impl:
        cfg["diff_impl"] = impl
    model = LoopViT(LoopViTConfig(**cfg))
    model.load_state_dict(ckpt["model"])
    return model.eval().to(device), ckpt


@torch.no_grad()
def collect(model: LoopViT, loader, device, steps: int, max_images: int):
    """One full-depth pass: per-step logits and relative state change, and
    Dirichlet energy / effective rank after every block application."""
    cfg = model.cfg
    npre, grid, B = cfg.num_cls_tokens, model.patch_embed.grid, cfg.core_depth
    out = {k: [] for k in ("logits", "labels", "fp", "dirichlet", "erank")}
    seen = 0
    for x, y in loader:
        if seen >= max_images:
            break
        x, y = x[: max_images - seen].to(device), y[: max_images - seen]
        h = model.embed(x)
        lg, fp, de, er = [], [], [], []

        def hook(b, hb):
            if b < B - 1:     # the last block's state is recorded after the (optional) relaxation
                de.append(dirichlet_energy_grid(hb[:, npre:], grid).cpu())
                er.append(effective_rank(hb[:, npre:]).cpu())

        for t in range(steps):
            h_prev = h
            h = model.advance(h, t, block_hook=hook)
            de.append(dirichlet_energy_grid(h[:, npre:], grid).cpu())
            er.append(effective_rank(h[:, npre:]).cpu())
            lg.append(model.classify(h).float().cpu())
            fp.append(torch.full((x.shape[0],), float("nan")) if t == 0
                      else rel_state_change(h, h_prev).cpu())
        out["logits"].append(torch.stack(lg, 1))            # (n, S, C)
        out["labels"].append(y)
        out["fp"].append(torch.stack(fp, 1))                # (n, S)
        out["dirichlet"].append(torch.stack(de, 1))         # (n, B*S)
        out["erank"].append(torch.stack(er, 1))
        seen += x.shape[0]
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def collect_graphs(model: LoopViT, x: torch.Tensor, steps: int, spectrum_images: int):
    """Dense attention graphs on a few fixed images. Returns
    cls_eff (k, S, P): effective CLS->patch weights sum_m theta_m (A^m)[0, :], last block,
    head mean; eig (S, B, n, h, T) complex spectra of A for the first `spectrum_images`."""
    cfg = model.cfg
    npre, B = cfg.num_cls_tokens, cfg.core_depth
    model.set_record_attention(True)
    h, cls_eff, eigs = model.embed(x), [], []
    for t in range(steps):
        per_block = []

        def hook(b, hb):
            attn = model.core[b].attn
            per_block.append((attn.last_A.float(), attn.last_theta.float()))

        h = model.advance(h, t, block_hook=hook)
        A, th = per_block[-1]                                  # last block
        if npre >= 1:
            r = torch.zeros(A.shape[0], A.shape[1], A.shape[-1], device=A.device)
            r[..., 0] = 1.0                                    # e_0: the CLS query
            eff = th[None, :, 0, None] * r
            for m in range(1, th.shape[-1]):
                r = (r.unsqueeze(-2) @ A).squeeze(-2)          # row 0 of A^m
                eff = eff + th[None, :, m, None] * r
            cls_eff.append(eff.mean(1)[:, npre:].cpu())       # (k, P)
        if spectrum_images:
            eigs.append(torch.stack([attention_spectrum(a[:spectrum_images]) for a, _ in per_block]))
    model.set_record_attention(False)
    return (torch.stack(cls_eff, 1) if cls_eff else None,
            torch.stack(eigs) if eigs else None)


def simulate_exit(logits, labels, ent, fp, B, mode, tau, eps, min_steps=1):
    """Offline dynamic exit over precomputed trajectories. Returns (acc, mean block apps)."""
    N, S, _ = logits.shape
    ent_ok = ent < tau
    fp_ok = torch.nan_to_num(fp, nan=float("inf")) < eps
    ok = {"entropy": ent_ok, "fixedpoint": fp_ok, "both": ent_ok & fp_ok,
          "either": ent_ok | fp_ok}[mode].clone()
    ok[:, : max(min_steps - 1, 0)] = False
    ok[:, S - 1] = True
    first = ok.int().argmax(1)                               # 0-based exit step
    pred = logits[torch.arange(N), first].argmax(-1)
    return (pred == labels).float().mean().item(), ((first + 1) * B).float().mean().item()


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--labels", nargs="*", default=None, help="legend names (default: variant/run dir)")
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--dataset", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--train-dir", default=None)
    p.add_argument("--val-dir", default=None)
    p.add_argument("--max-images", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--steps", type=int, default=0, help="loop steps to analyse (0 = 2 x loop_steps)")
    p.add_argument("--num-maps", type=int, default=8, help="fixed images for the CLS maps")
    p.add_argument("--spectrum-images", type=int, default=4,
                   help="images (of the --num-maps ones) whose attention spectra are computed")
    p.add_argument("--impl", choices=["sdpa", "dense"], default=None,
                   help="override diff_impl of the checkpoints (same function, same weights)")
    p.add_argument("--min-steps", type=int, default=1)
    p.add_argument("--taus", type=float, nargs="*", default=None)
    p.add_argument("--fp-eps", type=float, nargs="*", default=None)
    p.add_argument("--out-dir", default=None, help="default: <first ckpt dir>/analysis")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    from data import build_dataloaders, collect_samples
    plt = _plt()
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.ckpt[0])), "analysis")
    os.makedirs(out_dir, exist_ok=True)

    models = [load_model(c, device, args.impl) for c in args.ckpt]
    labels = args.labels or []
    for i, (m, ck) in enumerate(models[len(labels):], start=len(labels)):
        labels.append(f"{variant_name(m.cfg)}:{os.path.basename(os.path.dirname(os.path.abspath(args.ckpt[i])))}")
    classes = models[0][1]["classes"]
    for (m, ck), c in zip(models, args.ckpt):
        if ck["classes"] != classes:
            raise SystemExit(f"{c} was trained on different classes than {args.ckpt[0]}")

    # ---- data (rebuilt from the first checkpoint's training args) ----------
    targs = models[0][1].get("args") or {}
    pool, val_pool, _ = collect_samples(
        args.dataset or targs.get("dataset"), args.data_root or targs.get("data_root", "datasets"),
        targs.get("dataset_registry", "datasets.yaml"), args.train_dir or targs.get("train_dir"),
        args.val_dir or targs.get("val_dir"), targs.get("merge_splits", True))
    img_size = models[0][0].cfg.image_size
    train_loader, val_loader, names = build_dataloaders(
        pool, val_pool, img_size, args.batch_size, args.num_workers,
        None, "first", classes, targs.get("max_per_class"), targs.get("val_split", 0.1),
        "none", targs.get("seed", 42), pin_memory=False,
        fast_decode=targs.get("fast_decode", True))
    if args.split == "val":
        if val_loader is None:
            raise SystemExit("no validation split available; use --split train")
        loader = val_loader
    else:   # deterministic, un-augmented view of the train split
        from data import SampleListDataset, build_transforms
        loader = torch.utils.data.DataLoader(SampleListDataset(
            train_loader.dataset.samples, names, build_transforms(img_size)[1],
            train_loader.dataset.draft),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # fixed images for CLS maps / spectra: the first --num-maps of the split
    map_x = []
    for x, _ in loader:
        map_x.append(x)
        if sum(len(v) for v in map_x) >= args.num_maps:
            break
    map_x = torch.cat(map_x)[: args.num_maps]
    map_imgs = [(im * STD + MEAN).clamp(0, 1) for im in map_x]

    summary = {"split": args.split, "classes": len(classes), "runs": {}}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    results = []
    for (model, ck), label in zip(models, labels):
        cfg = model.cfg
        S = args.steps or 2 * cfg.loop_steps
        print(f"[analyze] {label}: {S} steps on up to {args.max_images} {args.split} images")
        r = collect(model, loader, device, S, args.max_images)
        r["cls_eff"], r["eig"] = collect_graphs(model, map_x.to(device), S,
                                                min(args.spectrum_images, len(map_x)))
        r.update(label=label, cfg=cfg, S=S, model=model)
        results.append(r)
        summary["runs"][label] = {"diffusion": cfg.diffusion, "diff_schedule": cfg.diff_schedule,
                                  "diff_hops": cfg.diff_hops, "loop_relax": cfg.loop_relax,
                                  "loop_steps": cfg.loop_steps, "images": len(r["labels"]),
                                  "gflops_per_image": model.flops()["gflops_total"]}

    # ---- 1 + 2. frequency responses and theta heatmaps ------------------------
    for r in results:
        vals = r["model"].diffusion_values()
        if vals is None:
            continue
        eig_re = None
        if r["eig"] is not None:        # (S, B, n, h, T); real parts per block, all steps
            eig_re = {b: r["eig"][:, b].real.flatten().float() for b in range(r["eig"].shape[1])}
        plot_frequency_response(vals["theta"], os.path.join(out_dir, f"freq_response_{_safe(r['label'])}.png"),
                                title=f"{r['label']}: nominal g(lambda)", eig_re=eig_re)
        plot_theta_heatmap(vals["theta"], os.path.join(out_dir, f"theta_{_safe(r['label'])}.png"),
                           title=f"{r['label']}: learned theta")
        lam = torch.linspace(-1, 1, 41)
        summary["runs"][r["label"]].update(
            {k: v.tolist() for k, v in vals.items()},
            freq_response_lambda=lam.tolist(),
            freq_response=frequency_response(vals["theta"], lam).tolist(),
            theta_neg_frac=float((vals["theta"] < 0).float().mean()))

    # ---- 3. oversmoothing vs unrolled depth ----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, r in enumerate(results):
        col, B = colors[i % len(colors)], r["cfg"].core_depth
        depth = range(1, B * r["S"] + 1)
        de, er = r["dirichlet"].mean(0), r["erank"].mean(0)
        axes[0].plot(depth, de, "o-", ms=3, color=col, label=r["label"])
        axes[1].plot(depth, er, "o-", ms=3, color=col, label=r["label"])
        for a in axes:
            a.axvline(B * r["cfg"].loop_steps, color=col, ls=":", lw=0.8)
        summary["runs"][r["label"]].update(dirichlet_energy_per_app=de.tolist(),
                                           effective_rank_per_app=er.tolist())
    axes[0].set(xlabel="unrolled depth (block applications)", ylabel="Dirichlet energy (normalised tokens)",
                title="Dirichlet energy (dotted = B*T_train)")
    axes[1].set(xlabel="unrolled depth (block applications)", ylabel="effective rank",
                title="Effective rank of patch tokens")
    axes[0].legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, "oversmoothing.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- 4. attention spectra ------------------------------------------------
    for r in results:
        if r["eig"] is None:
            continue
        eig = r["eig"]                                           # (S, B, n, h, T)
        S, B = eig.shape[:2]
        gap = spectral_gap(eig)                                  # (S, B, n, h)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        cmap = plt.get_cmap("viridis")
        for t in range(S):
            axes[0].hist(eig[t].abs().flatten().numpy(), bins=50, range=(0, 1), histtype="step",
                         color=cmap(t / max(S - 1, 1)), label=f"t={t + 1}", density=True)
        axes[0].set(xlabel="|lambda|", ylabel="density", yscale="log",
                    title="|eigenvalues| of A (all blocks / heads)")
        axes[0].legend(fontsize=7)
        g = gap.mean(dim=(2, 3))                                 # (S, B)
        for b in range(B):
            axes[1].plot(range(1, S + 1), g[:, b], "o-", label=f"block {b}")
        axes[1].axvline(r["cfg"].loop_steps, color="k", ls=":", lw=0.8)
        axes[1].set(xlabel="loop step", ylabel="1 - |lambda_2|", title="spectral gap")
        axes[1].legend(fontsize=7)
        fig.suptitle(f"{r['label']}: attention spectra ({eig.shape[2]} images)")
        fig.savefig(os.path.join(out_dir, f"spectra_{_safe(r['label'])}.png"), dpi=130,
                    bbox_inches="tight")
        plt.close(fig)
        summary["runs"][r["label"]].update(
            spectral_gap_step_block=g.tolist(),
            eig_abs_mean_step=eig.abs().mean(dim=(1, 2, 3, 4)).tolist())

    # ---- 5. extrapolation curve ----------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, r in enumerate(results):
        col, T = colors[i % len(colors)], r["cfg"].loop_steps
        acc = (r["logits"].argmax(-1) == r["labels"][:, None]).float().mean(0).tolist()
        ent = LoopViT.entropy(r["logits"])
        fp = r["fp"]
        ax.plot(range(1, r["S"] + 1), acc, "o-", color=col, label=r["label"])
        ax.axvline(T, color=col, ls=":", lw=0.8)
        summary["runs"][r["label"]].update(
            acc_per_step=acc, pred_entropy_per_step=ent.mean(0).tolist(),
            state_change_mean=[float("nan")] + fp[:, 1:].mean(0).tolist())
    ax.set(xlabel="inference loop steps T", ylabel="accuracy",
           title="Accuracy vs T (dotted = T_train)")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, "accuracy_per_step.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- 6. exit Pareto ------------------------------------------------------
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    mode_col = {"entropy": "C0", "fixedpoint": "C1", "both": "C2", "either": "C3"}
    for i, r in enumerate(results):
        ax = axes[0, i]
        B, C = r["cfg"].core_depth, r["logits"].shape[-1]
        ent = LoopViT.entropy(r["logits"])
        taus = args.taus or [float(v) for v in torch.logspace(-3, math.log10(math.log(C)), 12)]
        epss = args.fp_eps or [float(v) for v in torch.logspace(-4, 0, 12)]
        grid = {"entropy": [(t, float("inf")) for t in taus],
                "fixedpoint": [(-1.0, e) for e in epss],
                "both": [(t, e) for t in taus for e in epss],
                "either": [(t, e) for t in taus for e in epss]}
        pareto = {}
        for mode, pts in grid.items():
            res = [(tau, eps) + simulate_exit(r["logits"], r["labels"], ent, r["fp"], B, mode, tau,
                                              eps, args.min_steps) for tau, eps in pts]
            front = pareto_front([(c, a) for _, _, a, c in res])
            pareto[mode] = {"points": [{"tau": t, "eps": e, "acc": a, "block_apps": c}
                                       for t, e, a, c in res], "front": front}
            ax.scatter([c for *_, c in res], [a for _, _, a, _ in res], s=8, alpha=0.35,
                       color=mode_col[mode])
            ax.plot(*zip(*front), "-", color=mode_col[mode], label=f"exit: {mode}")
        fixed = [((t + 1) * B, (r["logits"][:, t].argmax(-1) == r["labels"]).float().mean().item())
                 for t in range(r["S"])]
        ax.plot(*zip(*fixed), "k--o", ms=3, label="fixed T")
        ax.set(xlabel="mean block applications / image", ylabel="accuracy", title=r["label"])
        ax.legend(fontsize=8)
        summary["runs"][r["label"]]["exit_pareto"] = pareto
        summary["runs"][r["label"]]["fixed_steps"] = fixed
    fig.savefig(os.path.join(out_dir, "exit_pareto.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- 7. CLS attention after diffusion --------------------------------------
    for r in results:
        if r["cls_eff"] is None:
            continue
        grid = r["model"].patch_embed.grid
        maps = r["cls_eff"]                                      # (k, S, P)
        k, S = maps.shape[0], maps.shape[1]
        signed = bool((maps < 0).any())
        vmax = float(maps.abs().max()) or 1.0
        fig, axes = plt.subplots(S + 1, k, figsize=(1.6 * k, 1.6 * (S + 1)), squeeze=False)
        for j in range(k):
            img = map_imgs[j].permute(1, 2, 0).numpy()
            axes[0, j].imshow(img)
            for t in range(S):
                m = maps[j, t].view(1, 1, grid, grid)
                m = torch.nn.functional.interpolate(m, size=img.shape[:2], mode="bilinear")[0, 0]
                axes[t + 1, j].imshow(img)
                if signed:
                    axes[t + 1, j].imshow(m.numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.6)
                else:
                    axes[t + 1, j].imshow(m.numpy(), cmap="inferno", alpha=0.6)
        for a in axes.ravel():
            a.set_xticks([])
            a.set_yticks([])
        axes[0, 0].set_ylabel("image", fontsize=8)
        for t in range(S):
            axes[t + 1, 0].set_ylabel(f"t={t + 1}", fontsize=8)
        fig.suptitle(f"{r['label']}: effective CLS weights sum_m theta_m (A^m)[0,:], last block, head mean"
                     + (" (red +, blue -)" if signed else ""), fontsize=9)
        fig.savefig(os.path.join(out_dir, f"cls_maps_{_safe(r['label'])}.png"), dpi=110,
                    bbox_inches="tight")
        plt.close(fig)

    # ---- 8. eta_t ------------------------------------------------------------
    relaxed = [r for r in results if r["model"].eta_values() is not None]
    if relaxed:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for i, r in enumerate(relaxed):
            eta = r["model"].eta_values()
            ax.plot(range(1, len(eta) + 1), eta, "o-", label=r["label"])
            summary["runs"][r["label"]]["eta"] = eta.tolist()
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set(xlabel="loop step t", ylabel="eta_t", title="Learned loop relaxation (1 = plain loop)")
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(out_dir, "eta.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[analyze] wrote figures and summary.json to {out_dir}")
    for label, run in summary["runs"].items():
        print(f"  {label}: acc/step {[round(a, 4) for a in run['acc_per_step']]}")
        if "theta_neg_frac" in run:
            print(f"  {' ' * len(label)}  negative theta fraction {run['theta_neg_frac']:.3f}")


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


if __name__ == "__main__":
    main()
