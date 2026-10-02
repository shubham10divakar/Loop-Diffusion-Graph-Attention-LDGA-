# LDGA — Loop Diffusion Graph Attention for Looped Vision Transformers

> *Looping as Diffusion: Learnable Spectral Filters on Attention Graphs in Looped Vision Transformers*

This repo implements **LDGA (Loop Diffusion Graph Attention)** on top of **LoopViT**
(Shu et al., [arXiv:2602.02156](https://arxiv.org/abs/2602.02156)), ported to image
classification. The full design is in
[`design_C_loop_diffusion_graph_attention.md`](design_C_loop_diffusion_graph_attention.md).

A softmax attention matrix `A` is row-stochastic, so `A·V` is a one-hop **low-pass** filter
on the attention graph. LoopViT applies that filter `B·T` times with tied weights, which is
the worst case for oversmoothing. LDGA replaces `A·V` with a learnable **polynomial graph
filter** and reads the loop as discretised diffusion time:

```
Attn_LDGA(V) = Σ_{m=0}^{M} θ_m · A^m V                                   (eq. C1)

ppr   θ_m = α(1−α)^m,        α = σ(a)          personalised PageRank (low-pass)
heat  θ_m = e^{−τ} τ^m / m!,  τ = softplus(s)   heat kernel exp(−τ(I−A)) (low-pass)
gpr   free θ_m                                  learned; can be high-pass, e.g. (1,−1) = (I−A)V
```

* `gpr` starts at θ = (0, 1, 0, …), so at initialisation LDGA-gpr **is** vanilla LoopViT.
* It adds only `(M+1)` scalars per head per block (96 for the default config with a shared
  schedule, 288 with `per_step`).
* The learned θ are directly readable as a **frequency response** `g(λ) = Σ θ_m λ^m`.
* **Fixed-point exit:** a sample halts once its state stops moving,
  `‖z_t − z_{t−1}‖ / ‖z_{t−1}‖ < ε`. This can be combined with the paper's entropy exit.
* Optional **loop relaxation** (eq. C2): `z_{t+1} = z_t + η_t (M_θ(z_t + e_t) − z_t)`, η = 1 at init.

### LDGA-gpr, the main method

"GPR" stands for **Generalized PageRank** (from GPR-GNN): the coefficients of the graph filter
are learned instead of fixed. LDGA applies this to the **self-attention graph of a looped,
weight-tied ViT** instead of a fixed input graph.

* **ppr and heat** fix the shape of θ and learn one scalar (α or τ). The filter is always
  **low-pass**: it can only smooth.
* **gpr** leaves every θ_m free, and θ_m can be negative. The filter can then be **high-pass** or
  band-pass. For example, θ = (1, −1) gives `(I − A)V`, which sharpens differences between
  tokens instead of averaging them. This lets the model counteract the smoothing the loop causes.
* **Init:** θ = (0, 1, 0, 0), so `out = A·V`, exactly vanilla LoopViT. Any change in behaviour
  is learned. `--gpr-init ppr` starts from PPR coefficients instead.
* **`per_step` schedule** (`--variant ldga`): every loop iteration has its own θ, so the filter
  can change over "diffusion time", e.g. become more high-pass at later steps (H2).
  `--variant ldga-gpr-shared` uses one θ per (block, head) for every iteration, to measure what
  the schedule adds.
* **Cost:** (M+1) scalars per head per block (96 shared / 288 per-step with the defaults), and
  about +6 % FLOPs per hop. It is compared against vanilla LoopViT at `dim 408` so compute is matched.

---

## 1. Files

| file | what |
|---|---|
| `loop_vit.py` | the model: `LoopViT` + `LoopViTConfig` with the LDGA filters, loop relaxation, fixed-point exit, batch-compacting `dynamic_forward` and analytic FLOP count |
| `ldga_stats.py` | diagnostics shared by model, training and analysis: frequency response, attention spectrum, Dirichlet energy, effective rank, relative state change |
| `train.py` | training: YAML + CLI, variants, early stopping, checkpoint every epoch, resume (also mid-epoch), per-epoch LDGA logging |
| `describe_dataset.py` | dataset description: images per class (train / val split), imbalance, image sizes, colour modes, disk size, pixel mean / std, sample grid, optional corrupt-file and duplicate / leakage checks, overview table (CSV / Markdown / LaTeX) |
| `evaluate.py` | paper metrics for any saved checkpoint (accuracy, balanced acc, precision / recall / specificity / F1, MCC, kappa, AUROC, AUPRC, log loss, Brier, ECE, bootstrap 95 % CIs), confusion matrix, ROC / PR / reliability plots, comparison tables (CSV / Markdown / LaTeX), training curves |
| `analyze_ldga.py` | analysis figures + JSON (frequency responses, θ heatmaps, oversmoothing, spectra, extrapolation, exit Pareto, CLS maps, η) |
| `predict.py` | run a checkpoint on images, with loop-count override, dynamic exit and per-step state change |
| `data.py` | dataset registry lookup (folder + CSV labels), pooled stratified train/val split, resumable sampler |
| `datasets.yaml` | dataset registry: paths into design A's `dataset/` and each dataset's label format |
| `downloads.py` | fetch aircraft / cub200 / flowers102 / food101 / cars into the `datasets/` layout |
| `config.yaml` | every setting in one place |
| `tests/test_ldga.py` | tests T1–T12 from the design doc (+ extras) |
| `tests/loop_vit_reference.py` | the original, unmodified LoopViT, used to prove baseline equivalence |

## 2. Variants

Select with `--variant` (a preset for `--diffusion` and `--diff-schedule`) or set the flags directly.

| `--variant` | `diffusion` | `diff_schedule` | what it is |
|---|---|---|---|
| `ldga` / `ldga-gpr` | `gpr` | `per_step` | **LDGA, main method.** Learned filter with a loop-time schedule. Default in `config.yaml` |
| `ldga-gpr-shared` | `gpr` | `shared` | learned filter, one set per (block, head): isolates the value of the step schedule |
| `ldga-ppr` | `ppr` | `shared` | fixed-shape low-pass diffusion (PPR) |
| `ldga-heat` | `heat` | `shared` | fixed-shape low-pass diffusion (heat kernel) |
| `loopvit` | `none` | – | vanilla LoopViT baseline, bit-identical to the original code (test T1) |

**Compute-matched baseline.** LDGA adds FLOPs (≈ +6 % per hop with `sdpa`). Vanilla LoopViT at
`--dim 408` costs 13.54 GFLOPs/image vs 13.48 for LDGA-gpr with M = 3 (`sdpa`, 224/16), against
12.05 for vanilla at `dim 384`. `train.py --summary-only` prints the analytic count for any config.

Ablation switches (design doc §11.3):

| flag | values | ablation |
|---|---|---|
| `--diff-hops` | `2`, `3`, `4` | filter order M |
| `--diff-heads` | `-1` / `k` | diffuse all heads or only the first k |
| `--gpr-init` | `vanilla` / `ppr` | gpr initialisation |
| `--diff-renorm` | `true` / `false` | renormalise truncated ppr / heat coefficients |
| `--loop-relax` | `false` / `true` | learnable Euler step η_t |
| `--ffn vanilla` | | does diffusion matter more without the ConvGLU's local prior? |
| `--loop-steps` | `2`, `3`, `4` | training T (extrapolation is logged every epoch) |
| `--exit-mode` | `entropy` / `fixedpoint` / `both` / `either` | dynamic-exit criterion |
| `--diff-impl` | `sdpa` / `dense` | same function; `dense` materialises A once in fp32 |

## 3. Install

```bash
pip install -r requirements.txt
```

Tested with Python 3.14, torch 2.14 + CUDA 12.6 on an RTX 3060 (Windows 11). On Ampere
cards, AMP uses bf16.

## 4. Datasets

Datasets are read in place, mostly from design A's `dataset/` folder; nothing is copied.
Where each one lives and how it is labelled is set in **`datasets.yaml`** (config key
`dataset_registry`):

| `--dataset` | source | classes | images |
|---|---|---|---|
| `plant-pathology-2020` | FGVC7 `train.csv`, one-hot columns | 4 | 1 821 |
| `plant-pathology-2021` | FGVC8 `train.csv`, space-separated labels; each combination is one class | 12 | 18 632 |
| `cassava` | `train.csv` + `label_num_to_disease_map.json` | 5 | 21 397 |
| `plantdoc` (= `plantodc`) | `train/` + `test/` folders, pooled | 27 | 2 920 |
| `plantvillage` | `color/<class>/` folders | 38 | 54 305 |
| `rice-leaf-bd` | RiceLeafDiseaseBD `Original images/<class>/` (§4.2) | 6 | 9 769 |
| `paddy` | Paddy Doctor (Kaggle) `train_images/<class>/` (§4.2) | 10 | 10 407 |

**One protocol for every dataset:** all labelled images go into one pool, and `--val-split`
(default 10 %) of each class is held out for validation. The split is stratified and seeded,
so it is identical across variants and in `analyze_ldga.py`. With `--merge-splits true` (the
default), a dataset's `train/` and `test/` (or `val/`, `valid/`) folders are pooled first. Use
`--merge-splits false` to keep its own test folder as the validation set. The Kaggle test
sets (plant-pathology, cassava) have no labels, so only their train CSV is used.

### 4.1 Plant-pathology datasets

| dataset | classes (images) |
|---|---|
| `plant-pathology-2020` (FGVC7) | healthy (516), multiple_diseases (91), rust (622), scab (592) |
| `plant-pathology-2021` (FGVC8) | scab (4826), healthy (4624), frog_eye_leaf_spot (3181), rust (1860), complex (1602), powdery_mildew (1184), scab+frog_eye_leaf_spot (686), scab+frog_eye_leaf_spot+complex (200), frog_eye_leaf_spot+complex (165), rust+frog_eye_leaf_spot (120), rust+complex (97), powdery_mildew+complex (87) |

* **Labels:** FGVC7 stores them as one-hot columns, and every image has exactly one.
* **Multi-label images:** about 1,350 FGVC8 images have more than one disease
  (`scab frog_eye_leaf_spot`). Each label **combination** is treated as its own class
  (`scab+frog_eye_leaf_spot`), which is the usual single-label setup and gives 12 classes.
  True multi-label training (sigmoid + BCE) is not implemented.
* **Test sets:** both Kaggle test sets are unlabelled (FGVC8 ships only 3 test images), so only
  `train.csv` is used, split 90 / 10.
* **Class imbalance:** both datasets are imbalanced (e.g. `multiple_diseases` has 91 images,
  9 of them in validation), so look at per-class results, not only overall accuracy.
* **Image size:** the images are large (FGVC7 2048×1365, FGVC8 up to 4000×2672), so keep
  `fast_decode: true`. If the GPU is still waiting on data, raise `--num-workers`.
* **Duplicates with conflicting labels** (found with `describe_dataset.py --duplicates`):
  * FGVC7: 1 image appears twice, labelled `multiple_diseases` and `scab`.
  * FGVC8: 27 images appear twice with different labels (e.g. `complex` vs `rust`), and 3 of
    those pairs fall on both sides of the train / val split.
  * These come from the original Kaggle data and are kept as they are. Mention them in the
    paper. `dataset_stats/<name>/problems.csv` lists every file involved.

### 4.2 Rice datasets

#### RiceLeafDiseaseBD (`rice-leaf-bd`)

Location: `D:/D/my docs/my docs/ideas/attention based works/Datasets/RiceLeafDiseaseBD/RiceLeafDiseaseBD/`.
Full notes are in `DATASET_INFO.md` in that folder.

| class | images | train | val |
|---|---|---|---|
| Blast | 1326 | 1193 | 133 |
| Brown spot | 2178 | 1960 | 218 |
| Healthy | 1575 | 1417 | 158 |
| Leaf smut | 724 | 652 | 72 |
| Rice Tungro | 2244 | 2020 | 224 |
| Sheath blight | 1722 | 1550 | 172 |
| **total** | **9769** | **8792** | **977** |

* **Training uses `Original images/`, not `Annotated images ( visual with labels)/`.**
  Classification needs one label per image, and the class folder is that label. The
  annotated folder is for object detection:
  * `labels/*.txt` are YOLO bounding boxes for lesions.
  * `visuals/*.jpg` are the same photos with red boxes drawn on them; a model would learn
    the boxes, not the disease.
  * It has no `Healthy` class.
* **Image properties:** every image is 1024 × 1024 RGB JPEG, 3.1 GB in total. The originals
  were mostly 1600 × 1200 phone photos (`Dataset metadata.xlsx`).
* **Imbalance:** mild, 3.1× (Leaf smut 724 vs Rice Tungro 2,244). There is no official
  train / test split, so the usual 90 / 10 stratified split applies.
* **Quality:** 0 corrupt files.
* **Duplicates:** 39 duplicate pairs. **8 pairs carry different labels**, mostly
  Rice Tungro ↔ Sheath blight, and 5 pairs straddle train / val. They are kept as is; the
  list is in `DATASET_INFO.md` and `dataset_stats/rice-leaf-bd/problems.csv`.

```bash
python train.py --config config.yaml --dataset rice-leaf-bd
python train.py --config config.yaml --dataset rice-leaf-bd --smote true
```

#### Paddy Doctor (`paddy`)

Location: `D:/D/my docs/my docs/ideas/attention based works/Datasets/paddy-disease-classification/`.
Full notes are in `DATASET_INFO.md` in that folder.

| class | images | train | val |
|---|---|---|---|
| bacterial_leaf_blight | 479 | 431 | 48 |
| bacterial_leaf_streak | 380 | 342 | 38 |
| bacterial_panicle_blight | 337 | 303 | 34 |
| blast | 1738 | 1564 | 174 |
| brown_spot | 965 | 869 | 96 |
| dead_heart | 1442 | 1298 | 144 |
| downy_mildew | 620 | 558 | 62 |
| hispa | 1594 | 1435 | 159 |
| normal | 1764 | 1588 | 176 |
| tungro | 1088 | 979 | 109 |
| **total** | **10407** | **9367** | **1040** |

* **Training uses `train_images/<class>/`.** `test_images/` (3,469) is the Kaggle test set
  and has no labels.
* **`train.csv`** matches the folder labels exactly (0 mismatches). Its `variety` (rice
  cultivar, 10 values, 67 % ADT45) and `age` (45–82 days) columns are not used.
* **Images:** 480 × 640 portrait RGB JPEG (4 are landscape), 0.76 GB in total.
* **Imbalance:** 5.2× (normal 1,764 vs bacterial_panicle_blight 337).
* **Quality:** 0 corrupt files.
* **Duplicates:** 74 duplicates in 72 groups, all within the same class, so none has
  conflicting labels. 10 groups straddle train / val.

```bash
python train.py --config config.yaml --dataset paddy
python train.py --config config.yaml --dataset paddy --smote true
```

### 4.3 Describing a dataset (images per class and more)

```bash
python describe_dataset.py --dataset plant-pathology-2021
python describe_dataset.py --dataset plant-pathology-2020 plant-pathology-2021 --verify --duplicates
python describe_dataset.py --all                  # every dataset in datasets.yaml + overview table
```

The split is computed exactly as in training: `val_split`, `seed`, `merge_splits` and
`max_per_class` are read from `config.yaml`, and the same flags override them. Results go to
`dataset_stats/<name>/`:

| file | content |
|---|---|
| `report.txt` | source, split, classes, images (train / val), smallest / median / largest class, imbalance ratio and normalised entropy, width / height / aspect ratio, most common sizes, colour modes, file formats, file size and total disk size, unreadable files, RGB pixel mean / std (`--pixel-stats N` images; ImageNet values shown for comparison), and the per-class table |
| `class_distribution.csv` / `.png` | per class: total, train, val, share of the dataset (stacked bars) |
| `image_sizes.png` | width, height, aspect-ratio and file-size histograms |
| `sample_grid.png` | `--samples-per-class` (default 4) images of every class |
| `summary.json` | every number |
| `problems.csv` | unreadable files, corrupt files (`--verify`, decodes every image) and duplicates (`--duplicates`, hashes every file): duplicates labelled with different classes and duplicates that leak across train / val are flagged |
| `../datasets_summary.csv` / `.md` / `.tex` | one row per dataset (with several datasets or `--all`) |

`train.py` also writes `class_distribution.csv` / `.png` of the actual split to every run
folder and prints the per-class table at start-up.

Overview, with the default 90 / 10 split and seed 42:

| Dataset | Classes | Images | Train | Val | Min/class | Max/class | Imbalance | Median size | Size (GB) |
|---|---|---|---|---|---|---|---|---|---|
| plant-pathology-2020 | 4 | 1821 | 1639 | 182 | 91 | 622 | 6.8x | 2048x1365 | 0.374 |
| plant-pathology-2021 | 12 | 18632 | 16769 | 1863 | 87 | 4826 | 55.5x | 4000x2672 | 14.989 |
| cassava | 5 | 21397 | 19256 | 2141 | 1087 | 13158 | 12.1x | 800x600 | 2.384 |
| plantdoc | 27 | 2920 | 2627 | 293 | 42 | 238 | 5.7x | 640x560 | 0.886 |
| plantvillage | 38 | 54305 | 48875 | 5430 | 152 | 5507 | 36.2x | 256x256 | 0.792 |
| rice-leaf-bd | 6 | 9769 | 8792 | 977 | 724 | 2244 | 3.1x | 1024x1024 | 3.086 |
| paddy | 10 | 10407 | 9367 | 1040 | 337 | 1764 | 5.2x | 480x640 | 0.763 |

### 4.4 SMOTE for imbalanced classes (optional, `--smote true`)

SMOTE (Chawla et al., 2002) adapted to images, the same method as design A. For every
train class smaller than the target, synthetic images are added. Each one blends a real
image with one of its k nearest neighbours from the same class:
`x = (1 − λ)·a + λ·b`, with λ ~ U(0, 1) and neighbours found on 16×16 RGB thumbnails.
The normal train augmentation is then applied to the blend.

```bash
python train.py --config config.yaml --dataset plant-pathology-2021 --smote true --save-every-steps 500
python train.py --config config.yaml --dataset plant-pathology-2021 --smote true --smote-target median
python train.py --config config.yaml --dataset plant-pathology-2020 --smote true --smote-k 5 --smote-target 600
```

| flag | default | meaning |
|---|---|---|
| `--smote` | `false` | turn SMOTE on |
| `--smote-k` | `5` | nearest neighbours to blend with |
| `--smote-target` | `max` | grow every train class to: the largest class (`max`), the median class (`median`) or an image count |

* **Only the train split changes.** Validation stays real images only, so SMOTE and
  non-SMOTE runs are scored on the same images.
* **Deterministic:** the synthetic samples depend only on `seed`, so a resumed run sees
  the same ones.
* **Separate run folder:** SMOTE runs default to `runs/<variant>_<dataset>_smote` (or
  `_smote-median`, `_smote-600`), so they never mix with normal runs. Resume refuses a
  checkpoint whose SMOTE settings differ.
* **Visible split:** the per-class table at start-up and `class_distribution.csv` / `.png`
  show the real and synthetic images separately (`+smote`, `train used` columns, hatched bars).
* **Epoch size:** with `max`, every class grows to the largest one.
  * plant-pathology-2020: +601 images (1,639 → 2,240 train).
  * plant-pathology-2021: +35,347 images (16,769 → 52,116), so each epoch takes about 3×
    longer. `--smote-target median` adds only 5,177.
  * The neighbour search takes about 20 s on plant-pathology-2021.

### 4.5 Lookup rules

* To add a dataset, add an entry to `datasets.yaml` (`format: folder` or `format: csv`; the
  file header documents the keys).
* A name that is not in the registry is searched as a folder under the registry root,
  `--data-root`, `datasets/` and `dataset/`.
* A folder that only wraps the class folders in one sub-folder, such as
  `plantvillage/color/`, is descended into automatically.
* `--train-dir` / `--val-dir` accept any path on disk and skip the name lookup.
* `--fast-decode true` (default) decodes JPEGs at the smallest 1/2^k scale that is still
  ≥ 2 × image size. This matters for the 2–4k px plant-pathology and cassava photos.
* On Windows, image paths longer than the 260-character `MAX_PATH` limit are opened through
  the `\\?\` long-path prefix.

Fetch the public benchmarks again if needed:

```bash
python downloads.py --dataset cub200 flowers102 aircraft food101
```

## 5. Commands

All commands below run from the repo root. `run_commands.txt` has them in one place.

### 5.0 Quick start: plant pathology

```bash
python describe_dataset.py --dataset plant-pathology-2020 plant-pathology-2021   # look at the data first
python train.py --config config.yaml --dataset plant-pathology-2020
python train.py --config config.yaml --dataset plant-pathology-2021 --save-every-steps 500
```

With `--config config.yaml`, all of the following is **on by default**; no extra flags are needed:

| feature | default |
|---|---|
| early stopping | `early_stopping: true`, `patience: 15`, `monitor: val_acc`, `min_delta: 0.0` |
| checkpoint every epoch | `save_every: 1`, `keep_checkpoints: 0` (keep all) |
| best / latest checkpoint | `best.pt` (whenever val acc improves), `last.pt` (every epoch and on Ctrl+C) |
| resume | `resume: auto`: re-run the same command to continue from `last.pt`, also mid-epoch |
| console logs and log files | per-epoch summary; `log.csv`, `metrics.jsonl`, `run_history.log` |
| graphs | `training_curves.png`, `theta_heatmap.png` and `freq_response.png` every epoch; `class_distribution.png` at the start |
| final evaluation | `final_eval: true`: `best.pt` gets the full metric report (§5.10) in `eval/best/` |

Not on by default: `save_every_steps: 0`, so `last.pt` is only written at the end of each
epoch. Add `--save-every-steps 500` for large datasets (plant-pathology-2021, cassava,
PlantVillage), so a crash loses at most 500 steps.

Tips:
* **Disk space:** each checkpoint holds the model and optimizer state, roughly 120 MB
  (estimate). Keeping every epoch of 100 costs about 12 GB per run;
  `--keep-checkpoints 10` keeps only the newest 10 (`best.pt` / `last.pt` are always kept).
* **GPU memory:** if CUDA runs out of memory, use `--batch-size 32`.
* **Windows data loading:** if training hangs at the first batch, use `--num-workers 0`.

### 5.1 Tests (CPU, a few seconds; T11 also uses CUDA fp16 when available)

```bash
python -m pytest tests -q
```

### 5.2 Model summary (no data needed)

```bash
python train.py --config config.yaml --summary-only --num-classes 38
python train.py --config config.yaml --summary-only --num-classes 38 --variant loopvit --dim 408
```

### 5.3 Train the new variant (LDGA-gpr, per_step) on each dataset

```bash
python train.py --config config.yaml --dataset plant-pathology-2020
python train.py --config config.yaml --dataset plant-pathology-2021
python train.py --config config.yaml --dataset cassava
python train.py --config config.yaml --dataset plantodc
python train.py --config config.yaml --dataset plantvillage
python train.py --config config.yaml --dataset cub200
python train.py --config config.yaml --dataset flowers102
python train.py --config config.yaml --dataset aircraft
python train.py --config config.yaml --dataset food101
```

Each run writes to `runs/<variant>_<dataset>/`, for example `runs/ldga_plantodc/`.

### 5.4 Main comparison (same dataset, every variant)

```bash
python train.py --config config.yaml --dataset plantodc --variant loopvit
python train.py --config config.yaml --dataset plantodc --variant loopvit --dim 408 --output-dir runs/loopvit-cm_plantodc
python train.py --config config.yaml --dataset plantodc --variant ldga-ppr
python train.py --config config.yaml --dataset plantodc --variant ldga-heat
python train.py --config config.yaml --dataset plantodc --variant ldga
python train.py --config config.yaml --dataset plantodc --variant ldga-gpr-shared
```

### 5.5 Ablations (give each its own `--output-dir`)

```bash
python train.py --config config.yaml --dataset plantodc --diff-hops 2              --output-dir runs/abl_M2
python train.py --config config.yaml --dataset plantodc --diff-hops 4              --output-dir runs/abl_M4
python train.py --config config.yaml --dataset plantodc --diff-heads 3             --output-dir runs/abl_heads3
python train.py --config config.yaml --dataset plantodc --gpr-init ppr             --output-dir runs/abl_gpr_init_ppr
python train.py --config config.yaml --dataset plantodc --variant ldga-ppr --diff-renorm false --output-dir runs/abl_ppr_norenorm
python train.py --config config.yaml --dataset plantodc --loop-relax true          --output-dir runs/abl_relax
python train.py --config config.yaml --dataset plantodc --ffn vanilla              --output-dir runs/abl_ffn_vanilla
python train.py --config config.yaml --dataset plantodc --loop-steps 2             --output-dir runs/abl_T2
python train.py --config config.yaml --dataset plantodc --seed 1 --output-dir runs/ldga_plantodc_s1   # seeds
```

### 5.6 Small / fast config (CIFAR-like, design doc §11.1)

```bash
python train.py --config config.yaml --dataset plantodc --image-size 32 --patch-size 4 --dim 192 --num-heads 6 --core-depth 4 --loop-steps 3
```

### 5.7 Subsets, epochs, early stopping

```bash
python train.py --config config.yaml --dataset plantvillage --num-classes 10 --max-per-class 300
python train.py --config config.yaml --dataset plantvillage --classes Apple___Apple_scab Apple___Black_rot Apple___healthy
python train.py --config config.yaml --dataset cub200 --epochs 200 --early-stopping true --patience 20 --min-delta 0.001 --monitor val_acc
python train.py --config config.yaml --dataset cub200 --early-stopping false
```

### 5.8 Checkpoints and resume

Every run saves:

| file | when |
|---|---|
| `last.pt` | every epoch; every `--save-every-steps N` optimizer steps; and on Ctrl+C |
| `best.pt` | whenever the early-stopping monitor improves |
| `checkpoints/epoch_XXX.pt` | every `--save-every` epochs (default 1 = **every epoch**). `--keep-checkpoints N` keeps only the newest N |

Each checkpoint stores the model, optimizer, AMP scaler, RNG states, LR-schedule step,
early-stopping counter and the position inside the epoch. Writes are atomic.

```bash
# resume where you stopped: `resume: auto` is the default, so re-run the SAME command
python train.py --config config.yaml --dataset plantodc

# resume from a specific checkpoint
python train.py --config config.yaml --dataset plantodc --resume runs/ldga_plantodc/checkpoints/epoch_020.pt
python train.py --config config.yaml --dataset plantodc --resume runs/ldga_plantodc/best.pt

# also save inside long epochs (e.g. food101), so a crash loses at most 500 steps
python train.py --config config.yaml --dataset food101 --save-every-steps 500

# train longer than originally planned: raise --epochs and re-run
python train.py --config config.yaml --dataset plantodc --epochs 150

# ignore an existing last.pt and start fresh (overwrites the run folder's logs)
python train.py --config config.yaml --dataset plantodc --resume none
```

A mid-epoch resume skips exactly the batches already seen. On resume, rows in `log.csv` /
`metrics.jsonl` from after the checkpoint are dropped. A run that already early-stopped
refuses to continue unless you raise `--patience` or pass `--early-stopping false`. Resume
also checks that the classes and model architecture match the checkpoint. Inference-only
knobs (`exit_*`, `min/max_loop_steps`, and `diff_impl`, since `sdpa` and `dense` compute the
same function) may change freely.

### 5.9 Predict

```bash
python predict.py --ckpt runs/ldga_plantodc/best.pt --images some_folder/
python predict.py --ckpt runs/ldga_plantodc/best.pt --images a.jpg --per-step           # prediction, entropy, state change per step
python predict.py --ckpt runs/ldga_plantodc/best.pt --images a.jpg --loop-steps 6       # extrapolate past T
python predict.py --ckpt runs/ldga_plantodc/best.pt --images some_folder/ --dynamic-exit --exit-mode entropy --exit-tau 0.05
python predict.py --ckpt runs/ldga_plantodc/best.pt --images some_folder/ --dynamic-exit --exit-mode both --exit-tau 0.1 --exit-fp-eps 0.005 --loop-steps 6
```

### 5.10 Evaluation: paper metrics from any checkpoint

When training ends (finished or early-stopped), `best.pt` is evaluated automatically and
the results go to `runs/<name>/eval/best/`. This is controlled by `final_eval: true` and
`final_eval_ckpt: best | last | both`. `training_curves.png` is redrawn every epoch. Any
checkpoint can be evaluated again later:

```bash
python evaluate.py --ckpt runs/ldga_plant-pathology-2021                    # a run folder = its best.pt
python evaluate.py --ckpt runs/ldga_plant-pathology-2021 --which best last
python evaluate.py --ckpt runs/ldga_plant-pathology-2021 --which all        # every saved epoch + metrics_vs_epoch.png
python evaluate.py --ckpt runs/ldga_plant-pathology-2021 --epochs 10 20 30  # chosen epochs
python evaluate.py --ckpt runs/ldga_plant-pathology-2021/checkpoints/epoch_025.pt
python evaluate.py --ckpt runs/ldga_plant-pathology-2021 --loop-steps 6 --dynamic-exit   # extrapolation / exit
python evaluate.py --curves runs/ldga_plant-pathology-2021                 # only redraw training_curves.png

# comparison table across runs (CSV, Markdown and LaTeX)
python evaluate.py --ckpt runs/loopvit_plant-pathology-2021 runs/loopvit-cm_plant-pathology-2021                           runs/ldga-ppr_plant-pathology-2021 runs/ldga-heat_plant-pathology-2021                           runs/ldga-gpr-shared_plant-pathology-2021 runs/ldga_plant-pathology-2021                    --labels LoopViT "LoopViT (dim 408)" LDGA-ppr LDGA-heat LDGA-gpr-shared LDGA-gpr                    --summary-dir evaluation/plant-pathology-2021
```

Each checkpoint is scored on the same validation split it was selected on. The split is
rebuilt from the settings saved in the checkpoint, and `--dataset` / `--dataset-registry`
override them if the data has moved. Results go to `runs/<name>/eval/<checkpoint>[_T<steps>]/`:

| file | content |
|---|---|
| `report.txt` | every metric (printed too), with 95 % CIs and a per-class table |
| `metrics.json` | accuracy, balanced accuracy, top-3/5, precision / recall / F1 (macro, weighted, micro), specificity, MCC, Cohen's kappa, AUROC and AUPRC (one-vs-rest; macro, weighted, micro), log loss, Brier, ECE, bootstrap 95 % CIs (accuracy, macro F1, MCC, macro AUROC; `--bootstrap N`, default 1000), parameters, GFLOPs/image, measured img/s and ms/image; dynamic-exit accuracy / F1 / MCC / steps / GFLOPs with `--dynamic-exit` |
| `per_class.csv` | support, precision, recall, specificity, F1, AUROC, AP per class |
| `confusion_matrix.csv` / `.png` | counts and row-normalised |
| `roc_curves.png`, `pr_curves.png` | one-vs-rest curves per class plus micro / macro averages |
| `reliability.png` | calibration diagram with ECE |
| `per_class_metrics.png` | precision / recall / F1 bars per class |
| `predictions.csv` | image path, true label, prediction, confidence and all class probabilities |

With several checkpoints, `summary.csv`, `summary.md` and `summary.tex` (a booktabs table)
are written to `--summary-dir`, or to the run's `eval/` folder when every checkpoint comes
from one run.

### 5.11 Analysis (design doc §8)

```bash
# compare vanilla vs the LDGA filters on the same validation split
python analyze_ldga.py --ckpt runs/loopvit_plantodc/best.pt runs/ldga-ppr_plantodc/best.pt runs/ldga-heat_plantodc/best.pt runs/ldga_plantodc/best.pt \
                       --labels vanilla ppr heat gpr --out-dir analysis/plantodc

# a single run, more images, custom exit sweep
python analyze_ldga.py --ckpt runs/ldga_cub200/best.pt --max-images 3000 --taus 0.01 0.05 0.1 0.3 --fp-eps 1e-3 1e-2 5e-2
```

Outputs:

| file | content |
|---|---|
| `freq_response_<label>.png` | learned nominal `g(λ)` per block (and step), over the actual Re(λ) distribution of A |
| `theta_<label>.png` | θ heatmaps (blocks × hops, per head [and step]) |
| `oversmoothing.png` | Dirichlet energy and effective rank vs unrolled depth 1..B·2T |
| `spectra_<label>.png` | \|λ\| histograms of A per step and the spectral gap `1 − |λ_2|` over steps |
| `accuracy_per_step.png` | accuracy vs inference T = 1..2·T_train (extrapolation) |
| `exit_pareto.png` | accuracy vs mean block applications for entropy / fixedpoint / both / either exits |
| `cls_maps_<label>.png` | effective CLS→patch weights `Σ θ_m (A^m)[0,:]` per step on 8 fixed images |
| `eta.png` | learned η_t (runs with `--loop-relax`) |
| `summary.json` | every number behind the figures |

### 5.12 Plant pathology (every variant, same pooled 90 / 10 split)

```bash
python train.py --config config.yaml --dataset plant-pathology-2020 --variant loopvit
python train.py --config config.yaml --dataset plant-pathology-2020 --variant loopvit --dim 408 --output-dir runs/loopvit-cm_plant-pathology-2020
python train.py --config config.yaml --dataset plant-pathology-2020 --variant ldga
python train.py --config config.yaml --dataset plant-pathology-2020 --variant ldga-gpr-shared
python train.py --config config.yaml --dataset plant-pathology-2020 --variant ldga-ppr
python train.py --config config.yaml --dataset plant-pathology-2020 --variant ldga-heat

python train.py --config config.yaml --dataset plant-pathology-2021 --variant loopvit
python train.py --config config.yaml --dataset plant-pathology-2021 --variant loopvit --dim 408 --output-dir runs/loopvit-cm_plant-pathology-2021
python train.py --config config.yaml --dataset plant-pathology-2021 --variant ldga
python train.py --config config.yaml --dataset plant-pathology-2021 --variant ldga-gpr-shared
python train.py --config config.yaml --dataset plant-pathology-2021 --variant ldga-ppr
python train.py --config config.yaml --dataset plant-pathology-2021 --variant ldga-heat

# paper metrics table (MCC, AUROC, F1, ... with CIs)
python evaluate.py --ckpt runs/loopvit_plant-pathology-2021 runs/loopvit-cm_plant-pathology-2021 runs/ldga-ppr_plant-pathology-2021 runs/ldga-heat_plant-pathology-2021 runs/ldga-gpr-shared_plant-pathology-2021 runs/ldga_plant-pathology-2021                    --labels LoopViT LoopViT-cm LDGA-ppr LDGA-heat LDGA-gpr-shared LDGA-gpr --summary-dir evaluation/plant-pathology-2021

# LDGA analysis figures
python analyze_ldga.py --ckpt runs/loopvit_plant-pathology-2021/best.pt runs/ldga-ppr_plant-pathology-2021/best.pt runs/ldga-heat_plant-pathology-2021/best.pt runs/ldga_plant-pathology-2021/best.pt                        --labels vanilla ppr heat gpr --out-dir analysis/plant-pathology-2021
```

Add `--save-every-steps 500` to the FGVC8 runs so that a crash loses at most 500 steps.
`analyze_ldga.py` rebuilds the same validation split from the checkpoint's saved settings.

### 5.13 Replication: PlantVillage runs

Same settings as the design A replication (100 epochs, batch 16, checkpoint every epoch,
early stop on val loss with patience 10, auto-resume). Only `--variant` and `--output-dir`
change:

```bash
python train.py --config config.yaml --dataset plantvillage --variant loopvit         --epochs 100 --batch-size 16 --save-every 1 --keep-checkpoints 0 --early-stopping true --monitor val_loss --patience 10 --resume auto --output-dir runs/loopvit_plantvillage
python train.py --config config.yaml --dataset plantvillage --variant loopvit --dim 408 --epochs 100 --batch-size 16 --save-every 1 --keep-checkpoints 0 --early-stopping true --monitor val_loss --patience 10 --resume auto --output-dir runs/loopvit-cm_plantvillage
python train.py --config config.yaml --dataset plantvillage --variant ldga            --epochs 100 --batch-size 16 --save-every 1 --keep-checkpoints 0 --early-stopping true --monitor val_loss --patience 10 --resume auto --output-dir runs/ldga_plantvillage
python train.py --config config.yaml --dataset plantvillage --variant ldga-gpr-shared --epochs 100 --batch-size 16 --save-every 1 --keep-checkpoints 0 --early-stopping true --monitor val_loss --patience 10 --resume auto --output-dir runs/ldga-gpr-shared_plantvillage
python train.py --config config.yaml --dataset plantvillage --variant ldga-ppr        --epochs 100 --batch-size 16 --save-every 1 --keep-checkpoints 0 --early-stopping true --monitor val_loss --patience 10 --resume auto --output-dir runs/ldga-ppr_plantvillage
python train.py --config config.yaml --dataset plantvillage --variant ldga-heat       --epochs 100 --batch-size 16 --save-every 1 --keep-checkpoints 0 --early-stopping true --monitor val_loss --patience 10 --resume auto --output-dir runs/ldga-heat_plantvillage
```

## 6. What is logged every epoch

```
epoch  12/100 | lr 4.31e-04 | train loss 1.8123 acc 0.4712 | 212 img/s, 5.84 GB
    val loss 1.6002 acc 0.5340 | acc per step [0.402 0.498 0.534] extrap [0.538 0.537 0.531]
    entropy/step [1.912 1.405 1.101 1.050 1.041 1.040] | state change/step [nan 0.0412 0.0187 0.0121 0.0109 0.0105]
    dirichlet/step [0.7514 0.7411 0.7290 0.7123 0.6920 0.6690] | eff. rank/step [24.3 24.3 24.4 24.4 24.4 24.4]
    exit(entropy) acc 0.5310 @ 2.41 steps, 9.6 block apps/img, 10.87 GFLOPs/img (fixed depth 13.48)
    theta mean/hop [+0.051 +0.912 -0.084 +0.013] | sum theta 0.892 | neg frac 0.31
```

* **acc per step / extrap:** validation accuracy after each step 1..T, then T+1..2T.
* **state change/step:** mean `‖z_t − z_{t−1}‖ / ‖z_{t−1}‖`, the fixed-point exit signal.
* **dirichlet / eff. rank:** oversmoothing of the patch tokens after each step (first
  `--eval-diag-images` validation images).
* **exit(...):** dynamic-exit accuracy, average steps, block applications actually executed
  and the resulting analytic GFLOPs. The exit compacts the batch.
* **theta:** mean θ per hop, mean `Σθ` and the fraction of negative coefficients (the H2
  signal: does the model learn high-pass components?). `alpha` / `tau` for ppr / heat,
  `eta/step` with `--loop-relax`.

Files in `runs/<name>/`: `class_distribution.csv` / `.png` (images per class in this run's
train / val split), `training_curves.png` (loss, accuracy, LR, accuracy per loop step,
θ sign, epoch time; redrawn every epoch, with the best epoch marked), `eval/` (final
evaluation, §5.10), `log.csv`, `metrics.jsonl` (full per-epoch record including raw θ,
α/τ and η), `theta_heatmap.png` and `freq_response.png` (latest epoch), `config.json`
(settings, parameter report and FLOPs), `last.pt`, `best.pt`, `checkpoints/`, and
`run_history.log` (append-only record of every invocation).

## 7. Configuration reference

| group | keys (defaults) |
|---|---|
| data | `dataset`, `dataset_registry: datasets.yaml`, `data_root: datasets`, `merge_splits: true`, `train_dir`, `val_dir`, `num_classes`, `class_selection: first`, `classes`, `max_per_class`, `val_split: 0.1`, `augment: basic`, `num_workers: 4`, `fast_decode: true`, `smote: false`, `smote_k: 5`, `smote_target: max` |
| model | `image_size: 224`, `patch_size: 16`, `dim: 384`, `core_depth: 4` (B), `loop_steps: 3` (T), `num_heads: 6`, `mlp_ratio: 4.0`, `dropout`, `attn_dropout`, `drop_path: 0.1`, `ffn: hybrid`, `rope: true`, `step_embedding: true`, `num_cls_tokens: 1`, `pool: cls` |
| LDGA | `variant`, `diffusion: gpr`, `diff_hops: 3`, `diff_heads: -1`, `diff_schedule: per_step`, `diff_renorm: true`, `diff_impl: sdpa`, `ppr_alpha_init: 0.2`, `heat_tau_init: 1.0`, `gpr_init: vanilla`, `loop_relax: false` |
| evaluation | `final_eval: true`, `final_eval_ckpt: best`, `eval_bootstrap: 1000` |
| exit | `exit_mode: entropy`, `exit_tau: 0.05`, `exit_fp_eps: 0.01`, `min_loop_steps: 1`, `max_loop_steps: 0` (= T), `eval_dynamic_exit: true`, `eval_extrapolate: true`, `eval_diag_images: 512` |
| training | `epochs: 100`, `batch_size: 64`, `lr: 5e-4`, `min_lr: 1e-5`, `weight_decay: 0.05`, `warmup_epochs: 5`, `label_smoothing: 0.1`, `grad_clip: 1.0`, `deep_supervision: 0.0`, `amp: true`, `seed: 42`, `device: auto`, `output_dir: null` |
| checkpoints | `resume: auto`, `save_every: 1`, `keep_checkpoints: 0`, `save_every_steps: 0` |
| early stopping | `early_stopping: true`, `patience: 15`, `min_delta: 0.0`, `monitor: val_acc` |

## 8. Implementation notes

* **Baseline behaviour is unchanged.** With `diffusion=none` and `loop_relax=false` the model
  has exactly the original state-dict keys and produces bit-identical outputs (test T1).
* **`sdpa` never materialises A.** `A^m V` is computed by feeding the previous hop back in as
  the values of the same fused attention, so each hop recomputes `QKᵀ` (flash-friendly).
  `dense` builds A once in fp32 with autocast disabled; both agree to 1e-5 (test T6).
* **Coefficients are fp32.** θ is computed with autocast disabled; ppr / heat use
  `lgamma` for `m!` in log space.
* **Per-step schedules are identity-extrapolated.** θ (and η) are indexed with
  `min(t, T_train−1)`, like the step embeddings, so a model trained at T=3 still runs at T=6
  (test T7).
* **No weight decay on θ / α / τ / η.** Decay would pull gpr towards θ = 0 rather than its
  vanilla init.
* **`attn_dropout` must be 0 with diffusion.** Per-hop dropout is not implemented, so the
  config raises instead of silently doing something else.
* **`dynamic_forward` compacts the batch.** It runs the core only on active samples. It
  returns `exit_steps`, `entropy`, `fp_trace` and `block_apps`, and matches per-sample
  reference runs (tests T10, T11).
* **The frequency response is nominal.** A is non-symmetric; `g(λ)` is evaluated on the real
  line and always plotted over the actual eigenvalue distribution (doc §13).

## 9. Hypotheses and status

Hypotheses (design doc §11.4), stated before running:

* **H1:** pure low-pass diffusion (ppr / heat) *increases* oversmoothing and does not beat
  vanilla. LDGA-gpr beats vanilla at matched compute.
* **H2:** learned gpr filters develop **negative / high-pass** components, stronger at later loop
  steps (`per_step`).
* **H3:** LDGA-gpr keeps Dirichlet energy and effective rank higher across unrolled depth, and
  degrades less when run for more loop steps than it was trained with (T > T_train).
* **H4:** the fixed-point exit (alone or with entropy) matches or beats the entropy-only exit on
  the accuracy vs `block_apps` curve.

H1's first half predicts a negative result for ppr / heat. That is intended, because it
motivates the learned filter; report it as it comes out.

Claims to avoid (doc §13): the frequency response is **nominal** (A is non-symmetric). Don't
call the model "continuous-time" or an "ODE solver"; with `--loop-relax` it is an
*Euler-style reading*. Don't claim efficiency for LDGA itself, because it adds FLOPs; any
efficiency claim belongs to the exit and must come from the Pareto plot. Say "attention graph",
not "topology".

**Status** (2026-10-02):
* **Done:**
  * the model and all variants; all 80 tests pass
  * the dataset registry (`datasets.yaml`) with the pooled 90 / 10 split
  * the evaluation pipeline (`evaluate.py`, final evaluation at the end of training)
  * the dataset description (`describe_dataset.py`)
  * optional SMOTE for minority classes (`--smote true`, §4.4)
  * smoke runs of every variant, resume, predict, analyze and evaluate, including on
    `plant-pathology-2020` / `-2021`
* **Data findings:**
  * plant-pathology-2021 is heavily imbalanced (55.5x), so report macro F1, MCC and balanced
    accuracy alongside accuracy
  * both plant-pathology sets and RiceLeafDiseaseBD contain duplicate images with
    conflicting labels (§4.1, §4.2); they are kept as is
* **Next:** full training runs, starting with plant pathology (§5.0, §5.12). H1–H4 are untested.
* **Open decision:** whether to drop the conflicting duplicates before splitting. Decide
  before the real runs, because it changes the split.
* **Not done:** a Kaggle single-file export (optional phase 6 of the design doc).

## 10. Credits

The base architecture is LoopViT:

```bibtex
@article{shu2026loopvit,
  title={LoopViT: Scaling Visual ARC with Looped Transformers},
  author={Shu, Wen-Jie and Qiu, Xuerui and Zhu, Rui-Jie and Chen, Harold Haodong and Liu, Yexin and Yang, Harry},
  journal={arXiv preprint arXiv:2602.02156},
  year={2026}
}
```

LDGA, the fixed-point exit and this classification code are by Shubham Divakar.
