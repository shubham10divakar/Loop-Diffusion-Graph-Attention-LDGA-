"""
Evaluate saved checkpoints with the metrics a paper needs, and plot training curves.

    # best checkpoint of a run (a run folder means its best.pt)
    python evaluate.py --ckpt runs/ldga_plant-pathology-2021

    # any checkpoint files, e.g. two epochs of one run
    python evaluate.py --ckpt runs/ldga_plant-pathology-2021/checkpoints/epoch_020.pt runs/ldga_plant-pathology-2021/checkpoints/epoch_040.pt

    # every saved epoch of a run (+ metrics-vs-epoch plot), or best + last
    python evaluate.py --ckpt runs/ldga_plant-pathology-2021 --which all
    python evaluate.py --ckpt runs/ldga_plant-pathology-2021 --which best last

    # several runs side by side -> one comparison table (CSV / Markdown / LaTeX)
    python evaluate.py --ckpt runs/loopvit_plant-pathology-2021 runs/ldga_plant-pathology-2021 \
                       --labels LoopViT LDGA --summary-dir evaluation/plant-pathology-2021

    # inference-time options: more loop steps (extrapolation), dynamic exit, no bootstrap
    python evaluate.py --ckpt runs/ldga_plant-pathology-2021 --loop-steps 6 --dynamic-exit --bootstrap 0

    # only redraw the training curves of a run from its log.csv
    python evaluate.py --curves runs/ldga_plant-pathology-2021

The data split is rebuilt from the settings stored in the checkpoint, so a checkpoint
is always scored on exactly the validation images it was selected on.

Metrics (metrics.json, report.txt):
    accuracy, balanced accuracy, top-3 / top-5 accuracy, precision / recall (sensitivity) /
    F1 (macro, weighted, micro), specificity (macro), MCC, Cohen's kappa,
    AUROC and AUPRC (one-vs-rest; macro, weighted, micro), log loss, Brier score,
    expected calibration error (15 bins), with bootstrap 95 % confidence intervals
    (--bootstrap N) for accuracy, macro F1, MCC and macro AUROC. Model cost: parameters,
    analytic GFLOPs/image and measured inference throughput (forward pass only).
Per class (per_class.csv): support, precision, recall, specificity, F1, AUROC, AP.
Figures: confusion_matrix.png (counts + row-normalised), roc_curves.png, pr_curves.png,
reliability.png, per_class_metrics.png. Also predictions.csv (path, true, predicted,
confidence, probabilities) and confusion_matrix.csv.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import time

import numpy as np
import torch

from loop_vit import LoopViT, LoopViTConfig

KEY_METRICS = ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
               "f1_weighted", "mcc", "cohen_kappa", "auroc_macro", "auprc_macro", "ece")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def load_checkpoint(path: str, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = LoopViT(LoopViTConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model"])
    return model.eval().to(device), ckpt


@torch.no_grad()
def predict(model: LoopViT, loader, device, loop_steps: int | None = None, amp: bool = True,
            dynamic_exit: bool = False):
    """Softmax probabilities for every image of `loader` (in loader order)."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    probs, labels, dyn_probs, dyn_steps = [], [], [], []
    dyn_apps, model_sec = 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            logits = model(x, loop_steps)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        model_sec += time.perf_counter() - t0
        probs.append(torch.softmax(logits.float(), -1).cpu())
        labels.append(y)
        if dynamic_exit:
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                d = model.dynamic_forward(x)
            dyn_probs.append(torch.softmax(d["logits"].float(), -1).cpu())
            dyn_steps.append(d["exit_steps"].float().cpu())
            dyn_apps += d["block_apps"]
    out = {"prob": torch.cat(probs).double().numpy(), "y": torch.cat(labels).numpy(),
           "model_sec": model_sec}
    if dynamic_exit:
        out.update(dyn_prob=torch.cat(dyn_probs).double().numpy(),
                   dyn_steps=torch.cat(dyn_steps).numpy(), dyn_block_apps=dyn_apps / len(out["y"]))
    return out


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def expected_calibration_error(prob, y, bins: int = 15):
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = (pred[m] == y[m]).mean(), conf[m].mean()
            ece += m.mean() * abs(acc - c)
            rows.append((float(lo), float(hi), int(m.sum()), float(acc), float(c)))
    return float(ece), rows


def _per_class_auc(y, prob):
    from sklearn.metrics import average_precision_score, roc_auc_score
    C = prob.shape[1]
    auc, ap = np.full(C, np.nan), np.full(C, np.nan)
    for c in range(C):
        pos = y == c
        if 0 < pos.sum() < len(y):          # undefined when a class is absent (or alone)
            auc[c] = roc_auc_score(pos, prob[:, c])
            ap[c] = average_precision_score(pos, prob[:, c])
    return auc, ap


def _bootstrap(y, prob, n: int, seed: int = 0):
    from sklearn.metrics import f1_score, matthews_corrcoef
    rng = np.random.default_rng(seed)
    C, N = prob.shape[1], len(y)
    stats = {"accuracy": [], "f1_macro": [], "mcc": [], "auroc_macro": []}
    for _ in range(n):
        i = rng.integers(0, N, N)
        yb, pb = y[i], prob[i]
        pred = pb.argmax(1)
        stats["accuracy"].append((pred == yb).mean())
        stats["f1_macro"].append(f1_score(yb, pred, labels=np.arange(C), average="macro",
                                          zero_division=0))
        stats["mcc"].append(matthews_corrcoef(yb, pred))
        stats["auroc_macro"].append(np.nanmean(_per_class_auc(yb, pb)[0]))
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in stats.items()}


def compute_metrics(y, prob, names, bootstrap: int = 0, seed: int = 0):
    """-> (metrics dict, per-class rows, confusion matrix)."""
    from sklearn import metrics as skm
    C = len(names)
    labels = np.arange(C)
    pred = prob.argmax(1)
    onehot = np.eye(C)[y]
    m = {"n_images": int(len(y)), "n_classes": C,
         "accuracy": skm.accuracy_score(y, pred),
         "balanced_accuracy": skm.recall_score(y, pred, labels=labels, average="macro",
                                               zero_division=0)}
    for k in (3, 5):
        if C > k:
            m[f"top{k}_accuracy"] = skm.top_k_accuracy_score(y, prob, k=k, labels=labels)
    for avg in ("macro", "weighted", "micro"):
        p, r, f, _ = skm.precision_recall_fscore_support(y, pred, labels=labels, average=avg,
                                                         zero_division=0)
        m[f"precision_{avg}"], m[f"recall_{avg}"], m[f"f1_{avg}"] = p, r, f
    m["mcc"] = skm.matthews_corrcoef(y, pred)
    m["cohen_kappa"] = skm.cohen_kappa_score(y, pred, labels=labels)

    cm = skm.confusion_matrix(y, pred, labels=labels)
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    with np.errstate(divide="ignore", invalid="ignore"):
        spec = tn / (tn + fp)
    m["specificity_macro"] = float(np.nanmean(spec))

    auc, ap = _per_class_auc(y, prob)
    support = onehot.sum(0)
    ok = ~np.isnan(auc)
    m["auroc_macro"] = float(np.nanmean(auc))
    m["auroc_weighted"] = float((auc[ok] * support[ok]).sum() / support[ok].sum())
    m["auroc_micro"] = skm.roc_auc_score(onehot.ravel(), prob.ravel())
    m["auprc_macro"] = float(np.nanmean(ap))
    m["auprc_weighted"] = float((ap[ok] * support[ok]).sum() / support[ok].sum())
    m["auprc_micro"] = skm.average_precision_score(onehot.ravel(), prob.ravel())
    pc = np.clip(prob, 1e-12, 1)
    m["log_loss"] = skm.log_loss(y, pc / pc.sum(1, keepdims=True), labels=labels)
    m["brier"] = float(((prob - onehot) ** 2).sum(1).mean())
    m["ece"], _ = expected_calibration_error(prob, y)
    m = {k: float(v) if not isinstance(v, int) else v for k, v in m.items()}
    if bootstrap:
        m["ci95"] = _bootstrap(y, prob, bootstrap, seed)
        m["ci95"]["n_bootstrap"] = bootstrap

    p, r, f, s = skm.precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    rows = [{"class": names[c], "support": int(s[c]), "precision": float(p[c]),
             "recall": float(r[c]), "specificity": float(spec[c]), "f1": float(f[c]),
             "auroc": float(auc[c]), "ap": float(ap[c])} for c in range(C)]
    return m, rows, cm


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def plot_confusion(cm, names, path, title=""):
    plt = _plt()
    C = len(names)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    size = max(5.0, 0.38 * C + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(2 * size + 1, size))
    for ax, mat, sub, fmt in ((axes[0], cm, "counts", "d"), (axes[1], norm, "row-normalised", ".2f")):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=None if fmt == "d" else 1)
        ax.set_title(f"{title} confusion matrix ({sub})".strip(), fontsize=10)
        ax.set_xticks(range(C), names, rotation=90, fontsize=7 if C > 15 else 8)
        ax.set_yticks(range(C), names, fontsize=7 if C > 15 else 8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        if C <= 20:
            thr = mat.max() / 2
            for i in range(C):
                for j in range(C):
                    ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=7,
                            color="white" if mat[i, j] > thr else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(y, prob, names, out_dir, title=""):
    from sklearn.metrics import auc as sk_auc, precision_recall_curve, roc_curve
    plt = _plt()
    C = len(names)
    onehot = np.eye(C)[y]
    show_legend = C <= 12
    grid = np.linspace(0, 1, 501)
    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(7, 6))
        interp = []
        for c in range(C):
            if not (0 < onehot[:, c].sum() < len(y)):
                continue
            if kind == "roc":
                fx, fy, _ = roc_curve(onehot[:, c], prob[:, c])
                interp.append(np.interp(grid, fx, fy))
            else:
                fy, fx, _ = precision_recall_curve(onehot[:, c], prob[:, c])
                interp.append(np.interp(grid, fx[::-1], fy[::-1]))
            ax.plot(fx, fy, lw=1, alpha=0.8 if show_legend else 0.35,
                    label=f"{names[c]} ({sk_auc(fx, fy):.3f})" if show_legend else None)
        if kind == "roc":
            fx, fy, _ = roc_curve(onehot.ravel(), prob.ravel())
        else:
            fy, fx, _ = precision_recall_curve(onehot.ravel(), prob.ravel())
        ax.plot(fx, fy, "k-", lw=2.2, label=f"micro-average ({sk_auc(fx, fy):.3f})")
        if interp:
            mean = np.mean(interp, 0)
            ax.plot(grid, mean, "k--", lw=2.2, label=f"macro-average ({sk_auc(grid, mean):.3f})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], ":", color="grey", lw=1)
            ax.set_xlabel("false positive rate")
            ax.set_ylabel("true positive rate")
        else:
            ax.set_xlabel("recall")
            ax.set_ylabel("precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{title} {'ROC' if kind == 'roc' else 'precision-recall'} curves "
                     f"(one-vs-rest, area in brackets)".strip(), fontsize=10)
        ax.legend(fontsize=7, loc="lower right" if kind == "roc" else "lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{kind}_curves.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_reliability(prob, y, path, title=""):
    plt = _plt()
    ece, rows = expected_calibration_error(prob, y)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.5, 6.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    mids = [(lo + hi) / 2 for lo, hi, *_ in rows]
    ax.bar(mids, [r[3] for r in rows], width=1 / 15, edgecolor="k", alpha=0.8, label="accuracy")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{title} reliability diagram (ECE {ece:.4f})".strip(), fontsize=10)
    ax.legend(fontsize=8)
    ax2.bar(mids, [r[2] for r in rows], width=1 / 15, edgecolor="k", color="grey")
    ax2.set_xlabel("confidence")
    ax2.set_ylabel("images")
    ax2.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class(rows, path, title=""):
    plt = _plt()
    C = len(rows)
    fig, ax = plt.subplots(figsize=(7, max(3.0, 0.32 * C + 1.2)))
    yy = np.arange(C)
    for k, (key, off) in enumerate((("precision", -0.27), ("recall", 0.0), ("f1", 0.27))):
        ax.barh(yy + off, [r[key] for r in rows], height=0.26, label=key)
    ax.set_yticks(yy, [f"{r['class']} (n={r['support']})" for r in rows], fontsize=7 if C > 15 else 8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title(f"{title} per-class metrics".strip(), fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _floats(s):
    try:
        return [float(v) for v in str(s).split()]
    except ValueError:
        return []


def plot_training_curves(run_dir: str, path: str | None = None):
    """Loss / accuracy / LR / per-step accuracy / filter sign / epoch time from log.csv."""
    log = os.path.join(run_dir, "log.csv")
    if not os.path.exists(log):
        return None
    with open(log, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("epoch")]
    if not rows:
        return None
    plt = _plt()

    def col(k):
        out = []
        for r in rows:
            try:
                out.append(float(r[k]))
            except (KeyError, TypeError, ValueError):
                out.append(math.nan)
        return np.array(out)

    ep = col("epoch")
    val_acc = col("val_acc")
    best = int(np.nanargmax(val_acc)) if not np.all(np.isnan(val_acc)) else None
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    a = axes.ravel()
    a[0].plot(ep, col("train_loss"), label="train (label-smoothed)")
    a[0].plot(ep, col("val_loss"), label="val")
    a[0].set_title("loss")
    a[1].plot(ep, col("train_acc"), label="train")
    a[1].plot(ep, val_acc, label="val")
    if not np.all(np.isnan(col("dyn_acc"))):
        a[1].plot(ep, col("dyn_acc"), "--", label="val, dynamic exit")
    a[1].set_title("accuracy")
    a[2].plot(ep, col("lr"))
    a[2].set_title("learning rate")
    steps = [_floats(r.get("val_acc_per_step", "")) for r in rows]
    S = max((len(s) for s in steps), default=0)
    for t in range(S):
        a[3].plot(ep, [s[t] if t < len(s) else math.nan for s in steps], lw=1,
                  label=f"T={t + 1}")
    a[3].set_title("val accuracy per loop step (incl. extrapolation)")
    if not np.all(np.isnan(col("theta_neg_frac"))):
        a[4].plot(ep, col("theta_neg_frac"), label="fraction of negative theta")
        a[4].plot(ep, col("theta_sum_mean"), label="mean sum theta")
    a[4].set_title("LDGA filter coefficients")
    a[5].plot(ep, col("sec"), label="epoch time (s)")
    a[5].set_title("epoch time (s)")
    for ax in a:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if best is not None:
            ax.axvline(ep[best], color="grey", ls=":", lw=1)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    title = os.path.basename(os.path.normpath(run_dir))
    if best is not None:
        title += f"  (best val acc {val_acc[best]:.4f} at epoch {int(ep[best])}, dotted line)"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path = path or os.path.join(run_dir, "training_curves.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# one checkpoint -> folder of results
# --------------------------------------------------------------------------- #
def _fmt(v):
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def run_evaluation(ckpt_path: str, loader, device, out_dir: str, model=None, ckpt=None,
                   loop_steps: int | None = None, bootstrap: int = 1000, dynamic_exit: bool = False,
                   amp: bool = True, label: str | None = None, split: str = "val"):
    """Score one checkpoint on `loader` and write metrics, tables and figures to out_dir."""
    if model is None:
        model, ckpt = load_checkpoint(ckpt_path, device)
    names = list(ckpt["classes"])
    os.makedirs(out_dir, exist_ok=True)
    T = loop_steps or model.cfg.loop_steps
    label = label or os.path.splitext(os.path.basename(ckpt_path))[0]
    print(f"[eval] {ckpt_path} (epoch {ckpt.get('epoch')}) on {len(loader.dataset)} {split} images, "
          f"T = {T}")

    res = predict(model, loader, device, T, amp, dynamic_exit)
    y, prob = res["y"], res["prob"]
    m, rows, cm = compute_metrics(y, prob, names, bootstrap)
    fl = model.flops(T)
    m.update(checkpoint=os.path.abspath(ckpt_path), label=label, split=split,
             epoch=ckpt.get("epoch"), loop_steps=T,
             trained_loop_steps=model.cfg.loop_steps,
             params_m=sum(p.numel() for p in model.parameters()) / 1e6,
             gflops_per_image=fl["gflops_total"],
             images_per_s=len(y) / res["model_sec"] if res["model_sec"] > 0 else math.nan,
             ms_per_image=1000 * res["model_sec"] / max(len(y), 1),
             batch_size=loader.batch_size, device=str(device))
    if dynamic_exit:
        dp = res["dyn_prob"]
        dm, _, _ = compute_metrics(y, dp, names, 0)
        m["dynamic_exit"] = {"exit_mode": model.cfg.exit_mode, "exit_tau": model.cfg.exit_tau,
                             "exit_fp_eps": model.cfg.exit_fp_eps,
                             "accuracy": dm["accuracy"], "f1_macro": dm["f1_macro"],
                             "mcc": dm["mcc"], "auroc_macro": dm["auroc_macro"],
                             "mean_steps": float(res["dyn_steps"].mean()),
                             "block_apps_per_image": res["dyn_block_apps"],
                             "gflops_per_image": fl["gflops_embed_head"]
                             + res["dyn_block_apps"] * fl["gflops_per_block"]}

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    with open(os.path.join(out_dir, "per_class.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "confusion_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted"] + names)
        for n, r in zip(names, cm):
            w.writerow([n] + list(map(int, r)))
    samples = getattr(loader.dataset, "samples", None)
    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "predicted", "correct", "confidence"] + [f"p_{n}" for n in names])
        for i in range(len(y)):
            p = int(prob[i].argmax())
            w.writerow([samples[i][0] if samples else i, names[y[i]], names[p], int(p == y[i]),
                        f"{prob[i, p]:.5f}"] + [f"{v:.5f}" for v in prob[i]])

    # human-readable report (also printed)
    ci = m.get("ci95", {})
    lines = [f"checkpoint : {ckpt_path}", f"epoch      : {m['epoch']}",
             f"split      : {split} ({m['n_images']} images, {m['n_classes']} classes), "
             f"loop steps T = {T} (trained with {model.cfg.loop_steps})",
             f"model      : {m['params_m']:.2f} M params, {m['gflops_per_image']:.2f} GFLOPs/image, "
             f"{m['images_per_s']:.0f} img/s ({m['ms_per_image']:.2f} ms/image, batch {m['batch_size']}, "
             f"{device})", ""]
    for k in ("accuracy", "balanced_accuracy", "top3_accuracy", "top5_accuracy", "precision_macro",
              "recall_macro", "specificity_macro", "f1_macro", "precision_weighted",
              "recall_weighted", "f1_weighted", "f1_micro", "mcc", "cohen_kappa", "auroc_macro",
              "auroc_weighted", "auroc_micro", "auprc_macro", "auprc_weighted", "auprc_micro",
              "log_loss", "brier", "ece"):
        if k in m:
            extra = (f"   95% CI [{ci[k][0]:.4f}, {ci[k][1]:.4f}]" if k in ci else "")
            lines.append(f"{k:<20}{m[k]:.4f}{extra}")
    if dynamic_exit:
        d = m["dynamic_exit"]
        lines += ["", f"dynamic exit ({d['exit_mode']}): acc {d['accuracy']:.4f}, F1 {d['f1_macro']:.4f}, "
                      f"MCC {d['mcc']:.4f} @ {d['mean_steps']:.2f} steps, "
                      f"{d['gflops_per_image']:.2f} GFLOPs/image"]
    w = max(len(r["class"]) for r in rows) + 2
    lines += ["", f"{'class':<{w}}{'n':>6}{'prec':>8}{'recall':>8}{'spec':>8}{'f1':>8}{'auroc':>8}{'ap':>8}"]
    for r in rows:
        lines.append(f"{r['class']:<{w}}{r['support']:>6}" + "".join(
            f"{_fmt(r[k]):>8}" for k in ("precision", "recall", "specificity", "f1", "auroc", "ap")))
    report = "\n".join(lines)
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)

    try:
        plot_confusion(cm, names, os.path.join(out_dir, "confusion_matrix.png"), label)
        plot_roc_pr(y, prob, names, out_dir, label)
        plot_reliability(prob, y, os.path.join(out_dir, "reliability.png"), label)
        plot_per_class(rows, os.path.join(out_dir, "per_class_metrics.png"), label)
    except Exception as ex:      # figures must never lose the numbers
        print(f"[warn] evaluation figures not written: {ex}")
    print(f"[eval] wrote {out_dir}")
    return m


# --------------------------------------------------------------------------- #
# comparison tables
# --------------------------------------------------------------------------- #
def write_summary(results: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cols = ["label", "epoch", "loop_steps"] + list(KEY_METRICS) + ["params_m", "gflops_per_image",
                                                                   "images_per_s"]
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["checkpoint"])
        for m in results:
            w.writerow([m.get(c) for c in cols] + [m["checkpoint"]])

    def cell(m, k):
        v = m.get(k)
        if isinstance(v, float):
            s = f"{100 * v:.2f}" if k in KEY_METRICS and k != "ece" else f"{v:.4f}" if k == "ece" else f"{v:.2f}"
            if k in m.get("ci95", {}):
                lo, hi = m["ci95"][k]
                s += f" ({100 * lo:.1f}-{100 * hi:.1f})"
            return s
        return str(v)

    show = ["label", "epoch", "accuracy", "balanced_accuracy", "f1_macro", "mcc", "cohen_kappa",
            "auroc_macro", "auprc_macro", "ece", "params_m", "gflops_per_image"]
    head = ["Model", "Epoch", "Acc", "Bal. Acc", "Macro F1", "MCC", "Kappa", "AUROC", "AUPRC",
            "ECE", "Params (M)", "GFLOPs"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(cell(m, k) for k in show) + " |" for m in results]
    note = "Metrics in %, except ECE, params and GFLOPs; 95 % bootstrap CI in brackets."
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + f"\n\n{note}\n")
    tex = ["\\begin{tabular}{l" + "c" * (len(head) - 1) + "}", "\\toprule",
           " & ".join(head) + " \\\\", "\\midrule"]
    tex += [" & ".join(cell(m, k).replace("_", "\\_").replace("%", "\\%") for k in show) + " \\\\"
            for m in results]
    tex += ["\\bottomrule", "\\end{tabular}", f"% {note}"]
    with open(os.path.join(out_dir, "summary.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print("\n".join(md))
    print(f"[eval] comparison table: {out_dir}/summary.csv | summary.md | summary.tex")


def plot_metrics_vs_epoch(results: list, path: str):
    plt = _plt()
    rs = sorted((m for m in results if m.get("epoch") is not None), key=lambda m: m["epoch"])
    if len(rs) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ep = [m["epoch"] for m in rs]
    for k in ("accuracy", "f1_macro", "mcc", "auroc_macro", "balanced_accuracy"):
        ax.plot(ep, [m[k] for m in rs], "o-", ms=3, label=k)
    ax.set_xlabel("epoch")
    ax.set_title("validation metrics of the saved epoch checkpoints")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def _expand(paths, which, epochs):
    """Run folders -> checkpoint files according to --which / --epochs."""
    out = []
    for p in paths:
        if os.path.isfile(p):
            out.append(p)
            continue
        if not os.path.isdir(p):
            raise SystemExit(f"not a checkpoint or run folder: {p}")
        for w in which:
            if w in ("best", "last"):
                f = os.path.join(p, f"{w}.pt")
                if os.path.exists(f):
                    out.append(f)
                else:
                    print(f"[warn] {f} does not exist")
            elif w == "all":
                out += sorted(glob.glob(os.path.join(p, "checkpoints", "epoch_*.pt")))
        for e in epochs or []:
            f = os.path.join(p, "checkpoints", f"epoch_{e:03d}.pt")
            if os.path.exists(f):
                out.append(f)
            else:
                print(f"[warn] {f} does not exist")
    return list(dict.fromkeys(out))


def _default_out(ckpt_path, loop_steps, split):
    run = os.path.dirname(os.path.abspath(ckpt_path))
    if os.path.basename(run) == "checkpoints":
        run = os.path.dirname(run)
    name = os.path.splitext(os.path.basename(ckpt_path))[0]
    if loop_steps:
        name += f"_T{loop_steps}"
    if split != "val":
        name += f"_{split}"
    return os.path.join(run, "eval", name)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="*", default=[], help="checkpoint files and/or run folders")
    p.add_argument("--which", nargs="*", default=["best"], choices=["best", "last", "all"],
                   help="for run folders: best.pt, last.pt and/or every checkpoints/epoch_*.pt")
    p.add_argument("--epochs", nargs="*", type=int, default=None,
                   help="for run folders: also these epoch checkpoints, e.g. --epochs 10 20 30")
    p.add_argument("--labels", nargs="*", default=None, help="names for the comparison table")
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--loop-steps", type=int, default=None,
                   help="inference loop steps T (default: the trained T)")
    p.add_argument("--dynamic-exit", action="store_true", help="also score the dynamic exit")
    p.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for 95%% CIs (0 = off)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default=None, help="results folder (one checkpoint only)")
    p.add_argument("--summary-dir", default=None,
                   help="where the comparison table goes (default: next to the results)")
    p.add_argument("--curves", nargs="*", default=None, metavar="RUN_DIR",
                   help="only (re)draw training_curves.png for these run folders")
    # data overrides, in case the dataset moved since training
    p.add_argument("--dataset", default=None)
    p.add_argument("--dataset-registry", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--train-dir", default=None)
    p.add_argument("--val-dir", default=None)
    args = p.parse_args()

    if args.curves is not None:
        for r in args.curves:
            print(f"[curves] {plot_training_curves(r) or f'no log.csv in {r}'}")
        if not args.ckpt:
            return
    ckpts = _expand(args.ckpt, args.which, args.epochs)
    if not ckpts:
        raise SystemExit("no checkpoints to evaluate (give --ckpt files or run folders)")
    if args.out_dir and len(ckpts) > 1:
        raise SystemExit("--out-dir works with one checkpoint; use --summary-dir for several")
    labels = list(args.labels or [])

    from data import SampleListDataset, build_dataloaders, build_transforms, collect_samples
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    loaders, results = {}, []
    for i, path in enumerate(ckpts):
        model, ckpt = load_checkpoint(path, device)
        a = ckpt.get("args") or {}
        key = (args.dataset or a.get("dataset"), args.data_root or a.get("data_root", "datasets"),
               args.dataset_registry or a.get("dataset_registry", "datasets.yaml"),
               args.train_dir or a.get("train_dir"), args.val_dir or a.get("val_dir"),
               a.get("merge_splits", True), a.get("max_per_class"), a.get("val_split", 0.1),
               a.get("seed", 42), model.cfg.image_size, tuple(ckpt["classes"]),
               a.get("fast_decode", True))
        if key not in loaders:
            pool, val_pool, _ = collect_samples(key[0], key[1], key[2], key[3], key[4], key[5])
            tr, va, names = build_dataloaders(
                pool, val_pool, key[9], args.batch_size, args.num_workers, None, "first",
                list(key[10]), key[6], key[7], "none", key[8],
                pin_memory=device.type == "cuda", fast_decode=key[11])
            if args.split == "train":     # un-augmented view of the training images
                va = torch.utils.data.DataLoader(
                    SampleListDataset(tr.dataset.samples, names, build_transforms(key[9])[1],
                                      tr.dataset.draft),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            if va is None:
                raise SystemExit("this dataset has no validation split; use --split train")
            loaders[key] = va
        if i < len(labels):
            label = labels[i]
        else:
            run = os.path.dirname(os.path.abspath(path))
            run = os.path.dirname(run) if os.path.basename(run) == "checkpoints" else run
            label = f"{os.path.basename(run)}/{os.path.splitext(os.path.basename(path))[0]}"
        out = args.out_dir or _default_out(path, args.loop_steps, args.split)
        results.append(run_evaluation(path, loaders[key], device, out, model, ckpt, args.loop_steps,
                                      args.bootstrap, args.dynamic_exit, args.amp, label, args.split))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(results) > 1:
        runs = {os.path.dirname(_default_out(c, None, "val")) for c in ckpts}
        sdir = args.summary_dir or (runs.pop() if len(runs) == 1 else "evaluation")
        write_summary(results, sdir)
        if len({os.path.dirname(_default_out(c, None, "val")) for c in ckpts}) == 1:
            plot_metrics_vs_epoch(results, os.path.join(sdir, "metrics_vs_epoch.png"))
    elif args.summary_dir:
        write_summary(results, args.summary_dir)


if __name__ == "__main__":
    main()
