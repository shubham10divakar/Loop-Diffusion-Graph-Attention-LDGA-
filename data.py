"""
Image-folder data pipeline.

The primary entry point is a dataset *name* under a data root:

    datasets/<dataset_name>/train/<class>/*.jpg
    datasets/<dataset_name>/test/<class>/*.jpg     # or val/ or valid/ (optional)

    python train.py --config config.yaml --dataset plantvillage

A flat folder also works -- if `datasets/<dataset_name>/` holds the class
folders directly, `val_split` of each class is held out for validation:

    datasets/<dataset_name>/<class>/*.jpg

If the name is not found under `data_root`, the sibling roots `datasets/` and
`dataset/` are tried too, and a folder that only wraps the class folders in a
single sub-folder (e.g. `dataset/plantvillage/color/<class>/`) is descended into.

`--train-dir` / `--val-dir` override the name-based lookup entirely.

The training loader uses `ResumableSampler`: the shuffle order of epoch e is a
pure function of (seed, e), so a run resumed mid-epoch skips exactly the
batches it had already seen.

Configurable:
  * num_classes      - use only N of the class folders (None = all)
  * class_selection  - "first" (alphabetical) or "random" (seeded) when N < total
  * classes          - explicit list of folder names (overrides the two above)
  * max_per_class    - cap images per class (None = all)
  * val_split        - fraction of train held out per class when there is no val dir
"""
from __future__ import annotations

import os
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

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


def _find_base(data_root: str, dataset: str) -> str:
    if os.path.isdir(dataset):
        return dataset
    roots = [data_root] + [r for r in FALLBACK_ROOTS if r != data_root]
    for root in roots:
        cand = os.path.join(root, dataset)
        if os.path.isdir(cand):
            return cand
    lines = []
    for root in roots:
        if os.path.isdir(root):
            lines.append(f"  {root}/: {[d for d in _subdirs(root) if not d.startswith('_')]}")
    raise SystemExit(f"dataset folder not found: {dataset} (looked under {roots})"
                     + ("\navailable:\n" + "\n".join(lines) if lines else ""))


def resolve_dataset_dirs(data_root: str, dataset: str | None,
                         train_dir: str | None, val_dir: str | None):
    """Turn (data_root, dataset name) into concrete train/val directories.
    Explicit train_dir/val_dir always win."""
    if train_dir:
        return train_dir, val_dir
    if not dataset:
        raise SystemExit(
            "give either --dataset <name> (looked up under --data-root) or --train-dir")

    base = _find_base(data_root, dataset)
    # unwrap single wrapper folders, e.g. plantvillage/color/<class>/*.jpg
    while True:
        subs = _subdirs(base)
        if (len(subs) == 1 and subs[0] not in ("train",) + VAL_DIR_NAMES
                and not _has_images(os.path.join(base, subs[0]))
                and _has_class_dirs(os.path.join(base, subs[0]))):
            base = os.path.join(base, subs[0])
            print(f"[data] descending into single sub-folder: {base}")
        else:
            break

    train = os.path.join(base, "train")
    if not _has_class_dirs(train):
        # flat layout: datasets/<name>/<class>/*.jpg
        return base, val_dir
    if val_dir is None:
        for name in VAL_DIR_NAMES:
            cand = os.path.join(base, name)
            if _has_class_dirs(cand):
                val_dir = cand
                break
    return train, val_dir


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


class SampleListDataset(Dataset):
    def __init__(self, samples, classes, transform=None):
        self.samples = samples          # list of (path, label)
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = default_loader(long_path(path))      # PIL RGB
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
            raise ValueError(f"Classes not found in train folder: {missing}")
        return list(classes)
    if num_classes is None or num_classes >= len(all_classes):
        if num_classes is not None and num_classes > len(all_classes):
            print(f"[data] asked for {num_classes} classes, folder has "
                  f"{len(all_classes)} - using all.")
        return list(all_classes)
    if class_selection == "random":
        return sorted(random.Random(seed).sample(list(all_classes), num_classes))
    return list(all_classes)[:num_classes]


def _group(samples, idx_to_name, keep, max_per_class, rng):
    by_class = defaultdict(list)
    for path, idx in samples:
        name = idx_to_name[idx]
        if name in keep:
            by_class[name].append(path)
    for name in by_class:
        rng.shuffle(by_class[name])
        if max_per_class:
            by_class[name] = by_class[name][:max_per_class]
    return by_class


def build_dataloaders(train_dir, val_dir=None, image_size=224, batch_size=64,
                      num_workers=4, num_classes=None, class_selection="first",
                      classes=None, max_per_class=None, val_split=0.1,
                      augment="basic", seed=42, pin_memory=True):
    rng = random.Random(seed)
    train_tf, eval_tf = build_transforms(image_size, augment)

    base = datasets.ImageFolder(train_dir)
    chosen = _select_classes(base.classes, num_classes, class_selection, classes, seed)
    name_to_label = {c: i for i, c in enumerate(chosen)}
    idx_to_name = {i: c for c, i in base.class_to_idx.items()}

    train_by_class = _group(base.samples, idx_to_name, set(chosen), max_per_class, rng)
    empty = [c for c in chosen if not train_by_class.get(c)]
    if empty:
        raise ValueError(f"No images found for classes: {empty}")

    train_samples, val_samples = [], []
    if val_dir:
        vbase = datasets.ImageFolder(val_dir)
        vidx_to_name = {i: c for c, i in vbase.class_to_idx.items()}
        val_by_class = _group(vbase.samples, vidx_to_name, set(chosen), None, rng)
        for c in chosen:
            train_samples += [(p, name_to_label[c]) for p in train_by_class[c]]
            val_samples += [(p, name_to_label[c]) for p in val_by_class.get(c, [])]
    else:
        # stratified hold-out: at least one val image per class when possible
        for c in chosen:
            paths = train_by_class[c]
            n_val = int(round(len(paths) * val_split)) if val_split > 0 else 0
            if val_split > 0 and len(paths) > 1:
                n_val = max(1, n_val)
            val_samples += [(p, name_to_label[c]) for p in paths[:n_val]]
            train_samples += [(p, name_to_label[c]) for p in paths[n_val:]]

    train_ds = SampleListDataset(train_samples, chosen, train_tf)
    val_ds = SampleListDataset(val_samples, chosen, eval_tf) if val_samples else None

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=ResumableSampler(len(train_ds), seed),
                              num_workers=num_workers, pin_memory=pin_memory,
                              drop_last=len(train_ds) > batch_size,
                              persistent_workers=num_workers > 0)
    val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory,
                             persistent_workers=num_workers > 0)
                  if val_ds else None)

    counts = defaultdict(int)
    for _, y in train_samples:
        counts[y] += 1
    print(f"[data] {len(base.classes)} class folders in {train_dir}; using {len(chosen)}")
    print(f"[data] train images: {len(train_ds)} | val images: {len(val_samples)}"
          f" ({'from ' + val_dir if val_dir else f'{val_split:.0%} split of train'})")
    preview = ", ".join(f"{c}={counts[i]}" for i, c in enumerate(chosen[:10]))
    print(f"[data] per-class train counts: {preview}{' ...' if len(chosen) > 10 else ''}")
    return train_loader, val_loader, chosen
