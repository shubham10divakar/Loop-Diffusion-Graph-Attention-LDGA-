"""
Describe datasets: images per class, the train / val split used for training, image
properties and data-quality checks. Everything a paper's dataset section needs.

    python describe_dataset.py --dataset plant-pathology-2021
    python describe_dataset.py --dataset plant-pathology-2020 plant-pathology-2021
    python describe_dataset.py --all                          # every dataset in datasets.yaml

    # slower checks: decode every image (corrupt files) and hash every file (duplicates,
    # duplicates across classes = label noise, across train / val = leakage)
    python describe_dataset.py --dataset plant-pathology-2021 --verify --duplicates

The split is computed exactly as train.py does, from config.yaml (val_split, seed,
merge_splits, max_per_class, dataset_registry); CLI flags override it.

Per dataset, in dataset_stats/<name>/:
    report.txt              everything below, readable (also printed)
    summary.json            all numbers
    class_distribution.csv  per class: total, train, val, share of the dataset
    class_distribution.png  stacked train / val bars per class
    image_sizes.png         width / height / aspect-ratio / file-size histograms
    sample_grid.png         a few images of every class
    problems.csv            unreadable / corrupt / duplicate files (when there are any)
With several datasets: dataset_stats/datasets_summary.csv | .md | .tex (one row per dataset).

train.py writes class_distribution.csv / .png of the actual split to every run folder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from data import collect_samples, load_registry, long_path, split_samples


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------------------- #
# class distribution (cheap: no image is opened). Also used by train.py.
# --------------------------------------------------------------------------- #
def class_distribution(train_samples, val_samples, names):
    tr, va = Counter(y for _, y in train_samples), Counter(y for _, y in val_samples)
    total = len(train_samples) + len(val_samples)
    return [{"class": n, "total": tr[i] + va[i], "train": tr[i], "val": va[i],
             "share_%": round(100 * (tr[i] + va[i]) / max(total, 1), 2)}
            for i, n in enumerate(names)]


def distribution_stats(rows):
    t = np.array([r["total"] for r in rows])
    return {"classes": len(rows), "images": int(t.sum()),
            "train": int(sum(r["train"] for r in rows)), "val": int(sum(r["val"] for r in rows)),
            "min_class": rows[int(t.argmin())]["class"], "min_class_images": int(t.min()),
            "max_class": rows[int(t.argmax())]["class"], "max_class_images": int(t.max()),
            "median_class_images": float(np.median(t)), "mean_class_images": float(t.mean()),
            "imbalance_ratio": float(t.max() / max(t.min(), 1)),
            # 1 = perfectly balanced, -> 0 = everything in one class
            "normalised_entropy": float(-(t / t.sum() * np.log(t / t.sum() + 1e-12)).sum()
                                        / math.log(len(t))) if len(t) > 1 else 1.0}


def distribution_table(rows) -> str:
    w = max(len("class"), max(len(r["class"]) for r in rows)) + 2
    lines = [f"{'class':<{w}}{'total':>8}{'train':>8}{'val':>7}{'share':>8}"]
    lines += [f"{r['class']:<{w}}{r['total']:>8}{r['train']:>8}{r['val']:>7}{r['share_%']:>7.2f}%"
              for r in rows]
    s = distribution_stats(rows)
    lines.append(f"{'TOTAL':<{w}}{s['images']:>8}{s['train']:>8}{s['val']:>7}{100:>7.2f}%")
    return "\n".join(lines)


def write_class_distribution(rows, out_dir, title=""):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "class_distribution.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    try:
        plt = _plt()
        C = len(rows)
        order = sorted(range(C), key=lambda i: -rows[i]["total"])
        fig, ax = plt.subplots(figsize=(8, max(3.0, 0.3 * C + 1.4)))
        yy = np.arange(C)
        tr = [rows[i]["train"] for i in order]
        va = [rows[i]["val"] for i in order]
        ax.barh(yy, tr, label="train")
        ax.barh(yy, va, left=tr, label="val")
        for y, i in zip(yy, order):
            ax.text(rows[i]["total"], y, f" {rows[i]['total']}", va="center", fontsize=7)
        ax.set_yticks(yy, [rows[i]["class"] for i in order], fontsize=7 if C > 15 else 8)
        ax.invert_yaxis()
        ax.set_xlabel("images")
        s = distribution_stats(rows)
        ax.set_title(f"{title} class distribution: {s['images']} images, {C} classes, "
                     f"imbalance {s['imbalance_ratio']:.1f}x".strip(), fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.margins(x=0.12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as ex:
        print(f"[warn] class distribution plot not written: {ex}")


# --------------------------------------------------------------------------- #
# image properties and quality checks
# --------------------------------------------------------------------------- #
def _header(path):
    try:
        with Image.open(long_path(path)) as im:
            return path, im.size[0], im.size[1], im.mode, im.format, os.path.getsize(long_path(path)), None
    except Exception as ex:
        return path, 0, 0, None, None, 0, f"{type(ex).__name__}: {ex}"


def _verify(path):
    try:
        with Image.open(long_path(path)) as im:
            im.load()
        return None
    except Exception as ex:
        return f"{type(ex).__name__}: {ex}"


def _md5(path):
    h = hashlib.md5()
    with open(long_path(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pmap(fn, items, workers):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items, chunksize=64))


def _stat(a):
    a = np.asarray(a, dtype=float)
    return {"min": float(a.min()), "median": float(np.median(a)), "mean": float(a.mean()),
            "max": float(a.max())} if len(a) else {}


def pixel_stats(paths, n, size=224, seed=0):
    """Per-channel RGB mean / std (0-1 scale) over n random images resized to size x size."""
    pick = random.Random(seed).sample(paths, min(n, len(paths)))
    s, s2, cnt = np.zeros(3), np.zeros(3), 0
    for p in pick:
        try:
            with Image.open(long_path(p)) as im:
                im.draft("RGB", (size, size))
                a = np.asarray(im.convert("RGB").resize((size, size)), dtype=np.float64) / 255.0
        except Exception:
            continue
        a = a.reshape(-1, 3)
        s += a.sum(0)
        s2 += (a ** 2).sum(0)
        cnt += len(a)
    if not cnt:
        return None
    mean = s / cnt
    return {"images": len(pick), "mean": mean.round(4).tolist(),
            "std": np.sqrt(np.maximum(s2 / cnt - mean ** 2, 0)).round(4).tolist()}


def plot_image_sizes(w, h, fsize, path, title=""):
    plt = _plt()
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    ax[0].hist(w, bins=40)
    ax[0].set_title("width (px)")
    ax[1].hist(h, bins=40)
    ax[1].set_title("height (px)")
    ax[2].hist(np.asarray(w) / np.maximum(h, 1), bins=40)
    ax[2].set_title("aspect ratio (w / h)")
    ax[3].hist(np.asarray(fsize) / 1024, bins=40)
    ax[3].set_title("file size (KB)")
    for a in ax:
        a.set_ylabel("images")
        a.grid(alpha=0.3)
    fig.suptitle(f"{title} image properties".strip(), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_sample_grid(by_class, path, per_class=4, seed=0, title=""):
    plt = _plt()
    names = sorted(by_class)
    rng = random.Random(seed)
    C = len(names)
    fig, axes = plt.subplots(C, per_class, figsize=(1.6 * per_class + 2.2, 1.6 * C), squeeze=False)
    for r, n in enumerate(names):
        pick = rng.sample(by_class[n], min(per_class, len(by_class[n])))
        for c in range(per_class):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c < len(pick):
                try:
                    with Image.open(long_path(pick[c])) as im:
                        im.draft("RGB", (160, 160))
                        im = im.convert("RGB")
                        im.thumbnail((160, 160))
                        ax.imshow(im)
                except Exception:
                    pass
            if c == 0:
                ax.set_ylabel(n, rotation=0, ha="right", va="center", fontsize=7)
    fig.suptitle(f"{title} samples per class".strip(), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def describe(name, cfg, out_root, workers=16, pixel_n=500, verify=False, duplicates=False,
             per_class=4, scan=True):
    print(f"\n========== {name} ==========")
    pool, val_pool, source = collect_samples(name, cfg["data_root"], cfg["dataset_registry"],
                                             None, None, cfg["merge_splits"])
    spec = load_registry(cfg["dataset_registry"])["datasets"].get(name, {})
    train_s, val_s, chosen, _ = split_samples(pool, val_pool, None, "first", None,
                                              cfg["max_per_class"], cfg["val_split"], cfg["seed"])
    rows = class_distribution(train_s, val_s, chosen)
    out = os.path.join(out_root, name)
    write_class_distribution(rows, out, name)
    split_of = {p: "train" for p, _ in train_s}
    split_of.update({p: "val" for p, _ in val_s})
    label_of = {p: chosen[y] for p, y in train_s + val_s}
    paths = list(label_of)

    info = {"dataset": name, "source": source, "format": spec.get("format", "folder"),
            "split": ("given val folder" if val_pool else
                      f"{cfg['val_split']:.0%} stratified hold-out, seed {cfg['seed']}"
                      + (", train + test pooled" if cfg["merge_splits"] else "")),
            **distribution_stats(rows)}
    problems = []
    if scan:
        print(f"[describe] reading headers of {len(paths)} images ...")
        hdr = _pmap(_header, paths, workers)
        ok = [r for r in hdr if r[6] is None]
        problems += [{"path": r[0], "class": label_of[r[0]], "split": split_of[r[0]],
                      "problem": "unreadable", "detail": r[6]} for r in hdr if r[6]]
        w, h = [r[1] for r in ok], [r[2] for r in ok]
        fs = [r[5] for r in ok]
        info.update(
            width=_stat(w), height=_stat(h),
            aspect_ratio=_stat(np.asarray(w) / np.maximum(h, 1)) if ok else {},
            most_common_sizes={f"{a}x{b}": c for (a, b), c in Counter(zip(w, h)).most_common(5)},
            distinct_sizes=len(set(zip(w, h))),
            color_modes=dict(Counter(r[3] for r in ok)), file_formats=dict(Counter(r[4] for r in ok)),
            file_size_kb=_stat(np.asarray(fs) / 1024), disk_size_gb=round(sum(fs) / 1024 ** 3, 3),
            unreadable=len(hdr) - len(ok))
        if ok:
            plot_image_sizes(w, h, fs, os.path.join(out, "image_sizes.png"), name)
    if pixel_n:
        print(f"[describe] pixel mean / std over {min(pixel_n, len(paths))} images ...")
        info["pixel_stats_rgb"] = pixel_stats(paths, pixel_n)
    if verify:
        print(f"[describe] decoding all {len(paths)} images ...")
        errs = _pmap(_verify, paths, workers)
        bad = [(p, e) for p, e in zip(paths, errs) if e]
        problems += [{"path": p, "class": label_of[p], "split": split_of[p], "problem": "corrupt",
                      "detail": e} for p, e in bad]
        info["corrupt"] = len(bad)
    if duplicates:
        print(f"[describe] hashing all {len(paths)} files ...")
        groups = defaultdict(list)
        for p, hsh in zip(paths, _pmap(_md5, paths, workers)):
            groups[hsh].append(p)
        dup = [g for g in groups.values() if len(g) > 1]
        cross_class = [g for g in dup if len({label_of[p] for p in g}) > 1]
        leak = [g for g in dup if len({split_of[p] for p in g}) > 1]
        info.update(duplicate_groups=len(dup), duplicate_files=sum(len(g) - 1 for g in dup),
                    duplicate_groups_across_classes=len(cross_class),
                    duplicate_groups_across_train_val=len(leak))
        for k, g in enumerate(dup):
            for p in g:
                problems.append({"path": p, "class": label_of[p], "split": split_of[p],
                                 "problem": "duplicate" + (" (different classes)" if g in cross_class else "")
                                 + (" (train/val leak)" if g in leak else ""),
                                 "detail": f"group {k}"})
    info["per_class"] = rows

    if per_class:
        by_class = defaultdict(list)
        for p in paths:
            by_class[label_of[p]].append(p)
        try:
            plot_sample_grid(by_class, os.path.join(out, "sample_grid.png"), per_class, title=name)
        except Exception as ex:
            print(f"[warn] sample grid not written: {ex}")
    if problems:
        with open(os.path.join(out, "problems.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(problems[0]))
            wr.writeheader()
            wr.writerows(problems)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(info, f, indent=2)

    def st(k, unit=""):
        d = info.get(k) or {}
        return (f"min {d['min']:.0f}{unit}, median {d['median']:.0f}{unit}, max {d['max']:.0f}{unit}"
                if d else "n/a")
    lines = [f"dataset        : {name}", f"source         : {source}",
             f"split          : {info['split']}",
             f"classes        : {info['classes']}",
             f"images         : {info['images']} (train {info['train']}, val {info['val']})",
             f"per class      : min {info['min_class_images']} ({info['min_class']}), median "
             f"{info['median_class_images']:.0f}, max {info['max_class_images']} ({info['max_class']})",
             f"imbalance      : {info['imbalance_ratio']:.1f}x (largest / smallest class), normalised "
             f"entropy {info['normalised_entropy']:.3f} (1 = balanced)"]
    if scan:
        ar = info.get("aspect_ratio") or {}
        lines += [f"width          : {st('width', ' px')}", f"height         : {st('height', ' px')}",
                  f"aspect ratio   : " + (f"min {ar['min']:.2f}, median {ar['median']:.2f}, max {ar['max']:.2f}"
                                          if ar else "n/a"),
                  f"common sizes   : {info['most_common_sizes']} ({info['distinct_sizes']} distinct)",
                  f"colour modes   : {info['color_modes']}", f"file formats   : {info['file_formats']}",
                  f"file size      : {st('file_size_kb', ' KB')}; total {info['disk_size_gb']} GB",
                  f"unreadable     : {info['unreadable']}"]
    if info.get("pixel_stats_rgb"):
        ps = info["pixel_stats_rgb"]
        lines.append(f"pixel mean/std : mean {ps['mean']}, std {ps['std']} (RGB, 0-1, {ps['images']} images; "
                     f"ImageNet: [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225])")
    if verify:
        lines.append(f"corrupt        : {info['corrupt']}")
    if duplicates:
        lines.append(f"duplicates     : {info['duplicate_files']} extra copies in {info['duplicate_groups']} groups; "
                     f"{info['duplicate_groups_across_classes']} groups span classes, "
                     f"{info['duplicate_groups_across_train_val']} span train / val")
    if problems:
        lines.append(f"problems       : {len(problems)} rows in problems.csv")
    report = "\n".join(lines) + "\n\n" + distribution_table(rows)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"[describe] wrote {out}")
    return info


def write_overview(infos, out_root):
    cols = [("dataset", "Dataset"), ("classes", "Classes"), ("images", "Images"), ("train", "Train"),
            ("val", "Val"), ("min_class_images", "Min/class"), ("max_class_images", "Max/class"),
            ("imbalance_ratio", "Imbalance"), ("median_size", "Median size"), ("disk_size_gb", "Size (GB)")]
    rows = []
    for i in infos:
        r = {k: i.get(k) for k, _ in cols}
        r["imbalance_ratio"] = f"{i['imbalance_ratio']:.1f}x"
        if i.get("width"):
            r["median_size"] = f"{i['width']['median']:.0f}x{i['height']['median']:.0f}"
        rows.append(r)
    with open(os.path.join(out_root, "datasets_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k, _ in cols])
        w.writeheader()
        w.writerows(rows)
    head = [h for _, h in cols]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(str(r[k] if r[k] is not None else "-") for k, _ in cols) + " |" for r in rows]
    with open(os.path.join(out_root, "datasets_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    tex = ["\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", "\\toprule", " & ".join(head) + " \\\\",
           "\\midrule"]
    tex += [" & ".join(str(r[k] if r[k] is not None else "-").replace("_", "\\_") for k, _ in cols) + " \\\\"
            for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(out_root, "datasets_summary.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print("\n" + "\n".join(md))
    print(f"[describe] overview: {out_root}/datasets_summary.csv | .md | .tex")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", nargs="*", default=[], help="dataset names (registry or folder)")
    p.add_argument("--all", action="store_true", help="every dataset in the registry")
    p.add_argument("--config", default="config.yaml", help="split settings are read from here")
    p.add_argument("--dataset-registry", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--val-split", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--merge-splits", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--out-dir", default="dataset_stats")
    p.add_argument("--pixel-stats", type=int, default=500,
                   help="images sampled for the RGB mean / std (0 = off)")
    p.add_argument("--samples-per-class", type=int, default=4, help="images per class in sample_grid.png (0 = off)")
    p.add_argument("--no-scan", action="store_true", help="skip reading image headers (sizes, modes, disk size)")
    p.add_argument("--verify", action="store_true", help="decode every image to find corrupt files (slow)")
    p.add_argument("--duplicates", action="store_true", help="hash every file to find duplicates (slow)")
    p.add_argument("--workers", type=int, default=16, help="threads for scanning / hashing")
    args = p.parse_args()

    cfg = {"dataset_registry": "datasets.yaml", "data_root": "datasets", "val_split": 0.1, "seed": 42,
           "merge_splits": True, "max_per_class": None}
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config) as f:
            y = yaml.safe_load(f) or {}
        cfg.update({k: y[k] for k in cfg if k in y})
    for k in cfg:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    names = list(args.dataset)
    if args.all:
        names += [n for n in load_registry(cfg["dataset_registry"])["datasets"] if n not in names]
    if not names:
        raise SystemExit("give --dataset <name ...> or --all")
    os.makedirs(args.out_dir, exist_ok=True)
    infos, seen = [], set()
    for n in names:
        try:
            info = describe(n, cfg, args.out_dir, args.workers, args.pixel_stats, args.verify,
                            args.duplicates, args.samples_per_class, not args.no_scan)
        except SystemExit as ex:
            print(f"[describe] skipped {n}: {ex}")
            continue
        key = (info["source"], info["images"])       # registry aliases (plantdoc / plantodc)
        if key not in seen:
            seen.add(key)
            infos.append(info)
    if len(infos) > 1:
        write_overview(infos, args.out_dir)


if __name__ == "__main__":
    main()
