"""
Image data pipeline.

The entry point is a dataset *name*. It is looked up first in the dataset registry
(`datasets.yaml`, config key `dataset_registry`), which says where each dataset lives
and how its labels are stored:

    format: folder   <path>/[train|test|val]/<class>/*.jpg   or   <path>/<class>/*.jpg
    format: csv      <path>/<csv> lists the images in <path>/<image_dir> with their labels
                     (one-hot columns, one label column, or space-separated multi-labels,
                     each label combination being one class)

    python train.py --config config.yaml --dataset plant-pathology-2021

A name that is not in the registry is searched as a folder under the registry root,
`data_root`, `datasets/` and `dataset/`. A folder that only wraps the class folders
in a single sub-folder (e.g. `plantvillage/color/<class>/`) is descended into.
`--train-dir` / `--val-dir` skip the name lookup.

Every dataset is turned into one pool of labelled images per class. With
`merge_splits` (default) the train/ and test/ (val/, valid/) folders are pooled, and
`val_split` of each class is held out for validation (seeded, so the split is the
same in every run and in analyze_ldga.py). With `merge_splits: false`, an existing
test/val folder is used as the validation set instead.

The training loader uses `ResumableSampler`: the shuffle order of epoch e is a
pure function of (seed, e), so a run resumed mid-epoch skips exactly the
batches it had already seen.

Configurable:
  * num_classes      - use only N of the classes (None = all)
  * class_selection  - "first" (alphabetical) or "random" (seeded) when N < total
  * classes          - explicit list of class names (overrides the two above)
  * max_per_class    - cap images per class (None = all), applied before the split
  * val_split        - fraction of each class held out for validation
  * fast_decode      - decode JPEGs at a reduced scale (>= 2x image_size); large speed-up
                       for the 2k-4k px plant-pathology / cassava photos
"""
from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VAL_DIR_NAMES = ("test", "val", "valid", "validation")
FALLBACK_ROOTS = ("datasets", "dataset")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".ppm", ".pgm")


def long_path(path: str) -> str:
    """On Windows, return an extended-length (\\\\?\\) path so files past the 260-char
    MAX_PATH limit still open (e.g. PlantVillage inside a deeply nested repo)."""
    if os.name != "nt" or path.startswith("\\\\?\\"):
        return path
    p = os.path.abspath(path)
    if len(p) < 240:
        return path
    return "\\\\?\\UNC\\" + p[2:] if p.startswith("\\\\") else "\\\\?\\" + p


def _has_class_dirs(path: str) -> bool:
    return os.path.isdir(path) and any(
        os.path.isdir(os.path.join(path, d)) for d in os.listdir(path))


def _subdirs(path: str):
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def _has_images(path: str) -> bool:
    return any(f.lower().endswith(IMG_EXTS) for f in os.listdir(path))


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def load_registry(path: str | None) -> dict:
    """{'root': str | None, 'datasets': {name: spec}}; empty when there is no file."""
    if not path or not os.path.isfile(path):
        if path:
            print(f"[data] dataset registry not found: {path} (folder lookup only)")
        return {"root": None, "datasets": {}}
    import yaml
    with open(path) as f:
        reg = yaml.safe_load(f) or {}
    root = reg.get("root")
    if root and not os.path.isabs(root):    # relative roots are relative to the registry file
        root = os.path.join(os.path.dirname(os.path.abspath(path)), root)
    return {"root": root, "datasets": reg.get("datasets") or {}}


# --------------------------------------------------------------------------- #
# sample collection: everything becomes {class_name: [paths]}
# --------------------------------------------------------------------------- #
def _scan_class_dirs(path: str, pool: dict):
    for c in _subdirs(path):
        cdir = os.path.join(path, c)
        pool[c].extend(os.path.join(cdir, f) for f in sorted(os.listdir(cdir))
                       if f.lower().endswith(IMG_EXTS))


def _unwrap(base: str) -> str:
    # unwrap single wrapper folders, e.g. plantvillage/color/<class>/*.jpg
    while True:
        subs = _subdirs(base)
        if (len(subs) == 1 and subs[0] not in ("train",) + VAL_DIR_NAMES
                and not _has_images(os.path.join(base, subs[0]))
                and _has_class_dirs(os.path.join(base, subs[0]))):
            base = os.path.join(base, subs[0])
            print(f"[data] descending into single sub-folder: {base}")
        else:
            return base


def _collect_folder(base: str, merge_splits: bool):
    """-> (pool, val_pool or None, description)."""
    base = _unwrap(base)
    train = os.path.join(base, "train")
    if not _has_class_dirs(train):          # flat: <base>/<class>/*.jpg
        pool = defaultdict(list)
        _scan_class_dirs(base, pool)
        return pool, None, f"{base} (flat)"
    splits = [os.path.join(base, n) for n in VAL_DIR_NAMES if _has_class_dirs(os.path.join(base, n))]
    pool = defaultdict(list)
    _scan_class_dirs(train, pool)
    if merge_splits:
        for s in splits:
            _scan_class_dirs(s, pool)
        names = ["train"] + [os.path.basename(s) for s in splits]
        return pool, None, f"{base} ({' + '.join(names)} pooled)"
    if not splits:
        return pool, None, f"{train}"
    val_pool = defaultdict(list)
    _scan_class_dirs(splits[0], val_pool)
    return pool, val_pool, f"{train} | val from {splits[0]}"


def _collect_csv(base: str, spec: dict):
    """CSV-labelled images -> (pool, None, description)."""
    csv_path = os.path.join(base, spec.get("csv", "train.csv"))
    img_dir = os.path.join(base, spec.get("image_dir", "images"))
    img_col = spec.get("image_col", "image")
    ext = spec.get("image_ext", "")
    label_cols = spec.get("label_cols")
    label_col = spec.get("label_col")
    if not (label_cols or label_col):
        raise SystemExit(f"registry entry for {base} needs label_cols or label_col")
    label_map = None
    if spec.get("label_map"):
        with open(os.path.join(base, spec["label_map"])) as f:
            label_map = {str(k): v for k, v in json.load(f).items()}

    pool, missing, bad = defaultdict(list), 0, 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if label_cols:
                hot = [c for c in label_cols if str(row[c]).strip() in ("1", "1.0")]
                if len(hot) != 1:
                    bad += 1
                    continue
                name = hot[0]
            else:
                value = str(row[label_col]).strip()
                if label_map is not None:
                    name = label_map.get(value, value)
                else:   # multi-label "scab frog_eye_leaf_spot" -> class "scab+frog_eye_leaf_spot"
                    name = "+".join(value.split())
            fname = row[img_col].strip()
            if ext and not fname.lower().endswith(IMG_EXTS):
                fname += ext
            path = os.path.join(img_dir, fname)
            if not os.path.isfile(long_path(path)):
                missing += 1
                continue
            pool[name].append(path)
    for name in pool:
        pool[name].sort()
    if missing:
        print(f"[data] WARNING: {missing} images listed in {csv_path} were not found and are skipped")
    if bad:
        print(f"[data] WARNING: {bad} rows of {csv_path} without exactly one label are skipped")
    return pool, None, f"{csv_path} -> {img_dir}"


def _find_base(roots, dataset: str) -> str:
    if os.path.isdir(dataset):
        return dataset
    for root in roots:
        cand = os.path.join(root, dataset)
        if os.path.isdir(cand):
            return cand
    lines = [f"  {r}/: {[d for d in _subdirs(r) if not d.startswith('_')]}"
             for r in roots if os.path.isdir(r)]
    raise SystemExit(f"dataset not found: {dataset} (not in the registry; looked under {roots})"
                     + ("\navailable:\n" + "\n".join(lines) if lines else ""))


def collect_samples(dataset: str | None, data_root: str = "datasets",
                    registry: str | None = "datasets.yaml", train_dir: str | None = None,
                    val_dir: str | None = None, merge_splits: bool = True):
    """Resolve a dataset to (pool, val_pool, description).

    pool / val_pool map class name -> list of image paths. val_pool is None unless
    merge_splits is off and the dataset ships a labelled test/val split (or --val-dir)."""
    if train_dir:
        pool = defaultdict(list)
        _scan_class_dirs(train_dir, pool)
        if not val_dir:
            return pool, None, train_dir
        vpool = defaultdict(list)
        _scan_class_dirs(val_dir, vpool)
        if merge_splits:
            for c, paths in vpool.items():
                pool[c].extend(paths)
            return pool, None, f"{train_dir} + {val_dir} pooled"
        return pool, vpool, f"{train_dir} | val from {val_dir}"
    if not dataset:
        raise SystemExit("give either --dataset <name> or --train-dir")

    reg = load_registry(registry)
    spec = reg["datasets"].get(dataset)
    if spec is not None:
        base = spec.get("path", dataset)
        if not os.path.isabs(base) and reg["root"]:
            base = os.path.join(reg["root"], base)
        if not os.path.isdir(base):
            raise SystemExit(f"registry entry '{dataset}' points to a missing folder: {base}")
        fmt = spec.get("format", "folder")
        if fmt == "csv":
            return _collect_csv(base, spec)
        if fmt == "folder":
            return _collect_folder(base, merge_splits)
        raise SystemExit(f"registry entry '{dataset}': unknown format {fmt!r} (folder | csv)")

    roots = [r for r in [reg["root"], data_root] if r]
    roots += [r for r in FALLBACK_ROOTS if r not in roots]
    return _collect_folder(_find_base(roots, dataset), merge_splits)


# --------------------------------------------------------------------------- #
# datasets / loaders
# --------------------------------------------------------------------------- #
class ResumableSampler(Sampler):
    """Random permutation that depends only on (seed, epoch), with an optional
    start offset so an interrupted epoch can be resumed exactly."""

    def __init__(self, n: int, seed: int = 42):
        self.n, self.seed, self.epoch, self.start = n, seed, 0, 0

    def set_epoch(self, epoch: int, start: int = 0):
        self.epoch, self.start = epoch, start

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        return iter(torch.randperm(self.n, generator=g)[self.start:].tolist())

    def __len__(self):
        return self.n - self.start


def load_image(path: str, draft: int | None = None) -> Image.Image:
    """PIL RGB image. With `draft`, JPEGs are decoded at the largest 1/2^k scale
    that keeps both sides >= draft (much faster for multi-megapixel photos)."""
    with open(long_path(path), "rb") as f:
        img = Image.open(f)
        if draft:
            img.draft("RGB", (draft, draft))
        return img.convert("RGB")


class SampleListDataset(Dataset):
    def __init__(self, samples, classes, transform=None, draft: int | None = None):
        self.samples = samples          # list of (path, label)
        self.classes = classes
        self.transform = transform
        self.draft = draft

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = load_image(path, self.draft)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_transforms(image_size: int, augment: str = "basic"):
    resize = int(round(image_size / 0.875))
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train = [transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
             transforms.RandomHorizontalFlip()]
    if augment == "trivial":
        train.append(transforms.TrivialAugmentWide())
    elif augment == "none":
        train = [transforms.Resize((image_size, image_size))]
    train += [transforms.ToTensor(), norm]
    evalt = [transforms.Resize(resize), transforms.CenterCrop(image_size),
             transforms.ToTensor(), norm]
    return transforms.Compose(train), transforms.Compose(evalt)


def _select_classes(all_classes, num_classes, class_selection, classes, seed):
    if classes:
        missing = [c for c in classes if c not in all_classes]
        if missing:
            raise ValueError(f"Classes not found in the dataset: {missing}")
        return list(classes)
    if num_classes is None or num_classes >= len(all_classes):
        if num_classes is not None and num_classes > len(all_classes):
            print(f"[data] asked for {num_classes} classes, dataset has "
                  f"{len(all_classes)} - using all.")
        return list(all_classes)
    if class_selection == "random":
        return sorted(random.Random(seed).sample(list(all_classes), num_classes))
    return list(all_classes)[:num_classes]


def build_dataloaders(pool, val_pool=None, image_size=224, batch_size=64,
                      num_workers=4, num_classes=None, class_selection="first",
                      classes=None, max_per_class=None, val_split=0.1,
                      augment="basic", seed=42, pin_memory=True, fast_decode=True):
    """pool / val_pool: {class name: [paths]} from `collect_samples`."""
    rng = random.Random(seed)
    train_tf, eval_tf = build_transforms(image_size, augment)
    draft = 2 * image_size if fast_decode else None

    all_classes = sorted(c for c, paths in pool.items() if paths)
    chosen = _select_classes(all_classes, num_classes, class_selection, classes, seed)
    name_to_label = {c: i for i, c in enumerate(chosen)}

    by_class = {}
    for c in chosen:
        paths = sorted(pool.get(c, []))
        rng.shuffle(paths)
        by_class[c] = paths[:max_per_class] if max_per_class else paths
    empty = [c for c in chosen if not by_class[c]]
    if empty:
        raise ValueError(f"No images found for classes: {empty}")

    train_samples, val_samples = [], []
    if val_pool:
        for c in chosen:
            train_samples += [(p, name_to_label[c]) for p in by_class[c]]
            val_samples += [(p, name_to_label[c]) for p in sorted(val_pool.get(c, []))]
    else:
        # stratified hold-out: at least one val image per class when possible
        for c in chosen:
            paths = by_class[c]
            n_val = int(round(len(paths) * val_split)) if val_split > 0 else 0
            if val_split > 0 and len(paths) > 1:
                n_val = max(1, n_val)
            val_samples += [(p, name_to_label[c]) for p in paths[:n_val]]
            train_samples += [(p, name_to_label[c]) for p in paths[n_val:]]

    train_ds = SampleListDataset(train_samples, chosen, train_tf, draft)
    val_ds = SampleListDataset(val_samples, chosen, eval_tf, draft) if val_samples else None

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=ResumableSampler(len(train_ds), seed),
                              num_workers=num_workers, pin_memory=pin_memory,
                              drop_last=len(train_ds) > batch_size,
                              persistent_workers=num_workers > 0)
    val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory,
                             persistent_workers=num_workers > 0)
                  if val_ds else None)

    tr_counts, va_counts = defaultdict(int), defaultdict(int)
    for _, y in train_samples:
        tr_counts[y] += 1
    for _, y in val_samples:
        va_counts[y] += 1
    print(f"[data] {len(all_classes)} classes in the dataset; using {len(chosen)}")
    print(f"[data] train images: {len(train_ds)} | val images: {len(val_samples)}"
          f" ({'given val split' if val_pool else f'{val_split:.0%} stratified split, seed {seed}'})")
    preview = ", ".join(f"{c}={tr_counts[i]}/{va_counts[i]}" for i, c in enumerate(chosen[:12]))
    print(f"[data] per-class train/val: {preview}{' ...' if len(chosen) > 12 else ''}")
    return train_loader, val_loader, chosen
