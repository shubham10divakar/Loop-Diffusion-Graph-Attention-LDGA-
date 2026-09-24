"""
Download and prepare public fine-grained classification datasets into a
plain ImageFolder layout that train.py / data.py already understand:

    datasets/<name>/train/<class>/*.jpg
    datasets/<name>/test/<class>/*.jpg

Usage
-----
    python downloads.py --dataset all
    python downloads.py --dataset cub200 flowers102
    python downloads.py --dataset cars --kaggle-dataset jessicali9530/stanford-cars-dataset

    # then, e.g. (train.py resolves datasets/cub200/{train,test} from the name)
    python train.py --config config.yaml --dataset cub200 --num-classes 200

Datasets
--------
  aircraft     FGVC-Aircraft, 100 "variant" classes         torchvision, fully automatic
  cub200       Caltech-UCSD Birds-200-2011, 200 classes     Caltech tarball, fully automatic
  flowers102   Oxford Flowers-102, 102 classes              torchvision, fully automatic
  food101      Food-101, 101 classes                        torchvision, fully automatic
  cars         Stanford Cars, 196 classes                   the official host is dead; needs Kaggle
                                                              (see --kaggle-dataset below)

Raw downloads/archives are cached under `datasets/_raw/<name>/` and the
per-class images are hardlinked (not copied) into `datasets/<name>/...`
whenever the two live on the same drive, so exporting costs no extra disk.
Pass --copy to force real copies (e.g. --root and the raw cache are on
different drives). Re-running is idempotent: a dataset whose output folder
already has files is skipped.

Stanford Cars
-------------
The original ai.stanford.edu hosting for Stanford Cars has been down for
years, so torchvision's own downloader for it refuses to run. This script
falls back to the Kaggle CLI:

    pip install kaggle
    # create an API token at https://www.kaggle.com/settings -> "Create New Token"
    # and place the downloaded kaggle.json at ~/.kaggle/kaggle.json
    python downloads.py --dataset cars

It accepts either a devkit+.mat mirror (the original layout) or a
pre-split "train/<class>/*.jpg, test/<class>/*.jpg" mirror -- whichever
the chosen --kaggle-dataset slug provides.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

INVALID_CHARS = '<>:"/\\|?*'


def _sanitize(name: str) -> str:
    for c in INVALID_CHARS:
        name = name.replace(c, "_")
    return name.strip() or "unnamed"


def _link_or_copy(src: Path, dst: Path, link: bool):
    if dst.exists():
        return
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _export(image_files, labels, class_names, out_dir: Path, link: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, label in zip(image_files, labels):
        path = Path(path)
        dst_dir = out_dir / _sanitize(str(class_names[label]))
        dst_dir.mkdir(exist_ok=True)
        _link_or_copy(path, dst_dir / path.name, link)
    n = sum(1 for p in out_dir.rglob("*") if p.is_file())
    print(f"[downloads] {out_dir}: {n} images across {len(set(labels))} classes")


def _copy_class_tree(src_root: Path, out_dir: Path, link: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for cls_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        dst_dir = out_dir / _sanitize(cls_dir.name)
        dst_dir.mkdir(exist_ok=True)
        for f in cls_dir.iterdir():
            if f.is_file():
                _link_or_copy(f, dst_dir / f.name, link)
                n += 1
    print(f"[downloads] {out_dir}: {n} images from {src_root}")


def _already_done(out_dir: Path) -> bool:
    return out_dir.exists() and any(p.is_file() for p in out_dir.rglob("*"))


# --------------------------------------------------------------------------- #
# FGVC-Aircraft
# --------------------------------------------------------------------------- #
def prepare_aircraft(raw_root: Path, out_root: Path, link: bool, args):
    from torchvision.datasets import FGVCAircraft

    for split, out_split in [("trainval", "train"), ("test", "test")]:
        ds = FGVCAircraft(str(raw_root), split=split, annotation_level="variant", download=True)
        _export(ds._image_files, ds._labels, ds.classes, out_root / out_split, link)


# --------------------------------------------------------------------------- #
# Caltech-UCSD Birds-200-2011
# --------------------------------------------------------------------------- #
CUB_URL = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
CUB_MANUAL_HELP = (
    "Automatic CUB-200-2011 download failed (the Caltech mirror may have moved). "
    "Download CUB_200_2011.tgz yourself from the official page "
    "http://www.vision.caltech.edu/datasets/cub_200_2011/ and extract it so that "
    "{base}/images exists, then re-run this script."
)


def prepare_cub200(raw_root: Path, out_root: Path, link: bool, args):
    from torchvision.datasets.utils import download_and_extract_archive

    base = raw_root / "CUB_200_2011"
    if not (base / "images").exists():
        raw_root.mkdir(parents=True, exist_ok=True)
        try:
            download_and_extract_archive(CUB_URL, download_root=str(raw_root), filename="CUB_200_2011.tgz")
        except Exception as e:
            raise RuntimeError(CUB_MANUAL_HELP.format(base=base)) from e
    if not (base / "images").exists():
        raise RuntimeError(CUB_MANUAL_HELP.format(base=base))

    images = {}
    with open(base / "images.txt") as f:
        for line in f:
            iid, rel = line.strip().split(" ", 1)
            images[iid] = rel
    is_train = {}
    with open(base / "train_test_split.txt") as f:
        for line in f:
            iid, flag = line.strip().split(" ", 1)
            is_train[iid] = flag == "1"

    for iid, rel in images.items():
        cls, name = rel.split("/", 1)
        out_split = "train" if is_train[iid] else "test"
        dst_dir = out_root / out_split / _sanitize(cls)
        dst_dir.mkdir(parents=True, exist_ok=True)
        _link_or_copy(base / "images" / rel, dst_dir / name, link)
    for split in ("train", "test"):
        d = out_root / split
        n = sum(1 for p in d.rglob("*") if p.is_file())
        print(f"[downloads] {d}: {n} images")


# --------------------------------------------------------------------------- #
# Oxford Flowers-102
# --------------------------------------------------------------------------- #
def prepare_flowers102(raw_root: Path, out_root: Path, link: bool, args):
    from torchvision.datasets import Flowers102

    class_names = [f"{i:03d}" for i in range(1, 103)]
    train_ds = Flowers102(str(raw_root), split="train", download=True)
    val_ds = Flowers102(str(raw_root), split="val", download=True)
    test_ds = Flowers102(str(raw_root), split="test", download=True)

    # official split is train=10/class, val=10/class, test=the rest;
    # train on train+val, evaluate on test.
    _export(list(train_ds._image_files) + list(val_ds._image_files),
             list(train_ds._labels) + list(val_ds._labels),
             class_names, out_root / "train", link)
    _export(test_ds._image_files, test_ds._labels, class_names, out_root / "test", link)


# --------------------------------------------------------------------------- #
# Food-101
# --------------------------------------------------------------------------- #
def prepare_food101(raw_root: Path, out_root: Path, link: bool, args):
    from torchvision.datasets import Food101

    for split, out_split in [("train", "train"), ("test", "test")]:
        ds = Food101(str(raw_root), split=split, download=True)
        _export(ds._image_files, ds._labels, ds.classes, out_root / out_split, link)


# --------------------------------------------------------------------------- #
# Stanford Cars (official host is dead -> Kaggle mirror)
# --------------------------------------------------------------------------- #
CARS_KAGGLE_HELP = (
    "Stanford Cars: the original ai.stanford.edu hosting for this dataset has been "
    "taken down, so it can't be fetched from the official source. Use a Kaggle mirror:\n"
    "  1. pip install kaggle\n"
    "  2. create an API token at https://www.kaggle.com/settings (\"Create New Token\")\n"
    "     and put the downloaded file at ~/.kaggle/kaggle.json\n"
    "  3. re-run: python downloads.py --dataset cars "
    "--kaggle-dataset <owner>/<dataset-slug>\n"
    "Any mirror works, either the original devkit+.mat layout or a pre-split "
    "train/<class>/*.jpg + test/<class>/*.jpg layout."
)


def _find_train_test_dirs(root: Path):
    if not root.exists():
        return None
    for train_dir in root.rglob("*"):
        if train_dir.is_dir() and train_dir.name.lower() == "train":
            classes = [c for c in train_dir.iterdir() if c.is_dir()]
            if not classes:
                continue
            test_dir = train_dir.parent / "test"
            if test_dir.is_dir():
                return train_dir, test_dir
    return None


def prepare_stanford_cars(raw_root: Path, out_root: Path, link: bool, args):
    import scipy.io as sio

    has_devkit = next(raw_root.rglob("cars_meta.mat"), None) is not None if raw_root.exists() else False
    class_dirs = _find_train_test_dirs(raw_root)

    if not has_devkit and class_dirs is None:
        if shutil.which("kaggle") is None:
            raise RuntimeError(CARS_KAGGLE_HELP)
        raw_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", args.kaggle_dataset, "-p", str(raw_root), "--unzip"],
            check=True,
        )
        has_devkit = next(raw_root.rglob("cars_meta.mat"), None) is not None
        class_dirs = _find_train_test_dirs(raw_root)

    if class_dirs is not None:
        train_dir, test_dir = class_dirs
        _copy_class_tree(train_dir, out_root / "train", link)
        _copy_class_tree(test_dir, out_root / "test", link)
        return

    devkit_meta = next(raw_root.rglob("cars_meta.mat"), None)
    if devkit_meta is None:
        raise RuntimeError(
            f"Couldn't find a usable Stanford Cars layout under {raw_root}.\n" + CARS_KAGGLE_HELP
        )
    classes = sio.loadmat(str(devkit_meta), squeeze_me=True)["class_names"].tolist()

    for ann_name, img_dirname, out_split in [
        ("cars_train_annos.mat", "cars_train", "train"),
        ("cars_test_annos_withlabels.mat", "cars_test", "test"),
    ]:
        ann_path = next(raw_root.rglob(ann_name), None)
        img_dir = next((p for p in raw_root.rglob(img_dirname) if p.is_dir()), None)
        if ann_path is None or img_dir is None:
            print(f"[downloads] cars: skipping {out_split} (missing {ann_name} or {img_dirname}/)")
            continue
        anns = sio.loadmat(str(ann_path), squeeze_me=True)["annotations"]
        image_files = [img_dir / str(a["fname"]) for a in anns]
        labels = [int(a["class"]) - 1 for a in anns]
        _export(image_files, labels, classes, out_root / out_split, link)


DATASETS = {
    "aircraft": prepare_aircraft,
    "cub200": prepare_cub200,
    "flowers102": prepare_flowers102,
    "food101": prepare_food101,
    "cars": prepare_stanford_cars,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", nargs="+", choices=list(DATASETS) + ["all"], default=["all"])
    p.add_argument("--root", type=str, default="datasets", help="output root: <root>/<name>/{train,test}")
    p.add_argument("--raw-root", type=str, default=None, help="raw download cache (default: <root>/_raw)")
    p.add_argument("--copy", action="store_true", help="copy files instead of hardlinking")
    p.add_argument("--clean-raw", action="store_true", help="delete a dataset's raw cache after exporting it")
    p.add_argument("--kaggle-dataset", type=str, default="jessicali9530/stanford-cars-dataset",
                   help="Kaggle dataset slug to use for --dataset cars")
    args = p.parse_args()

    names = list(DATASETS) if "all" in args.dataset else args.dataset
    root = Path(args.root)
    raw_root = Path(args.raw_root) if args.raw_root else root / "_raw"
    link = not args.copy

    results = {}
    for name in names:
        print(f"\n=== {name} ===")
        out_dir = root / name
        if _already_done(out_dir):
            print(f"[downloads] {out_dir} already has files, skipping (delete it to re-export)")
            results[name] = "already done"
            continue
        try:
            DATASETS[name](raw_root / name, out_dir, link, args)
        except Exception as e:
            print(f"[downloads] {name} FAILED: {e}")
            results[name] = f"failed: {e}"
            continue
        results[name] = "ok"
        if args.clean_raw:
            shutil.rmtree(raw_root / name, ignore_errors=True)

    print("\n=== summary ===")
    for name, status in results.items():
        print(f"  {name:12s} {status}")


if __name__ == "__main__":
    main()
