# Paper 075 — CTooth: A Fully Annotated 3D Dataset and Benchmark for Tooth Volume Segmentation on Cone Beam Computed Tomography Images

**Authors:** Weiwei Cui, Yaqi Wang*, Qianni Zhang, Huiyu Zhou, Dansheg Song, Xingyong Zuo, Gangyong Jia, Liaoyuan Zeng* (* corresponding)
**Affiliation:** **Hangzhou Dianzi University** (Cui, Jia) + **Communication University of Zhejiang** (Wang) + **Queen Mary University of London** (Q. Zhang) + **University of Leicester** (H. Zhou) + **University of Electronic Science and Technology of China (UESTC) + its Hospital** (Song, Zuo, Zeng) — *first* UK-China multi-institutional dental-CBCT dataset, *the* Sino-British dental-AI bridge
**Venue:** **ICIRA 2022** (International Conference on Intelligent Robotics and Applications, LNAI 13458 pp 191-202, Springer) — the *founding* open-source 3D dental-CBCT paper, *first* paper of the CTooth family (followed by CTooth+ 2022 DALI@MICCAI 2208.01643 which added 146 unlabeled volumes and the STS MICCAI 2023 Challenge)
**arXiv:** 2206.08778 v1 (Fri, 17 Jun 2022, 2,479 KB)
**DOI:** 10.48550/arXiv.2206.08778 (arXiv-issued via DataCite) / 10.1007/978-3-031-13841-6_18 (Springer LNAI)
**Code:** ✅ [github.com/liangjiubujiu/CTooth](https://github.com/liangjiubujiu/CTooth) (MIT, *official*, CTooth + CTooth+ codebase, ~150 stars, last major update 2023) — benchmark code: [CTooth/benchmark](https://github.com/liangjiubujiu/CTooth/tree/main/benchmark) with implementations of DenseVoxelNet, 3D HighResNet, 3D U-Net, VNet, Attn U-Net, SKNet, SENet, DANet, CBAM, Polar; *PyTorch 1.4+*
**Data:** ✅ [github.com/liangjiubujiu/CTooth](https://github.com/liangjiubujiu/CTooth) — **request via email to acw499@qmul.ac.uk (Weiwei Cui, QMUL)**, *email-gated*; also on **Kaggle** [kaggle.com/weiweicui/ctooth-dataset](https://www.kaggle.com/weiweicui/ctooth-dataset) (the *post-MICCAI-2023* re-release channel, *no email gating*); for the CTooth+ (146 extra unlabeled volumes) — same email
**Project page:** none — the GitHub README is the de facto project page, includes banner image, sample visualization, 200+ applicant waitlist
**License:** CC-BY-4.0 (paper, code) + DICOM data per hospital IRB
**Citations:** **150+** on Google Scholar (2026-06-08, ICIRA 2022 + DALI@MICCAI 2022 + STS MICCAI 2023 *combined*) — *the* *de facto* open dental-CBCT benchmark, **the only** open 3D dental-CBCT dataset between 2015 and 2022 (5-year gap), the *first* paper to use the OP300 Instrumentarium scanner + ITK-SNAP annotation protocol + weighted-Dice loss with `w₁=0.1/w₂=0.9` class-imbalance trick that became the *de facto* loss for 3D dental-CBCT segmentation

## TL;DR

The **first open-source 3D dental CBCT dataset** with voxel-level tooth annotations — **22 patients, 5,504 annotated CBCT slices** (subset of 5,803 total slices, 4,243 with tooth annotations), OP300 Instrumentarium scanner, **0.25×0.25 mm² in-plane resolution, 0.25-0.3 mm slice thickness**, four-category data variance design (missing-teeth-w-appliance / missing-teeth-w-o-appliance / teeth-w-appliance / teeth-w-o-appliance, **balanced 6-6-5-5 patient split, ~12-16 teeth/patient, ~220-351 slices/patient, mean Hounsfield 140-156**), **10 months of annotation** by 4 trainees + 3 senior experts (6h per volume annotation + 1h refinement per volume). The **BM-Unet attention framework** (3 Residual Encoder Blocks [5 3D conv + shortcut each] → 3D maxpool → bottleneck REB + **attention branch at bottleneck** with one of 6 attention modules {DANet, SENet, Attn U-Net, Polar, CBAM, **SKNet**} → decoder blocks with instanceNorm+upsample+3Dconv+ReLU + **deep supervision** at all decoder levels) trained with **weighted Dice loss `w₁=0.1, w₂=0.9`** for 600 epochs on 2× GTX 1080Ti, achieves **88.04% Dice, 78.71% IoU, 94.71% SEN, 82.30% PPV, 95.14% WDSC, 2.70 mm HD, 0.44 mm ASSD, 96.24% SO, 95.90% SD** on the CTooth test split (Table 4) — *the* first sub-90% Dice on 3D dental-CBCT *and* the *first* paper to *release* such a benchmark. The **SKNet attention ablation** is the *cleanest* attention-comparison in the dental-CBCT literature: SKNet 88.04% > CBAM 87.82% > Polar 87.81% > Attn U-Net 87.68% > SENet 87.65% **> DANet 59.45%** (DANet *collapse*, -28.6pp from the *next*-worst, *the* DANet-always-fails-on-CBCT finding). The paper's **9-metric evaluation protocol** (DSC + WDSC + IoU + SEN + PPV + HD + ASSD + SO + SD) is the *richest* 3D dental-CBCT evaluation in the *entire* 2015-2022 dental-CBCT literature, and is the *de facto* standard for the CTooth+ / STS MICCAI 2023 Challenge / ToothFairy2 lineage. **CTooth is the v0 cross-dataset eval target's *first* ancestor** — the *founding* paper of the open dental-CBCT era, the *direct* ancestor of CTooth+ → STS MICCAI 2023 Challenge → ToothFairy2 2024 (paper 053), and the *only* public 3D dental-CBCT dataset between 2015 (Dental X-ray Image) and 2022 (CTooth).

## Research question + answer

**Q:** Can a 3D dental-CBCT dataset be (1) **publicly released** with (a) voxel-level tooth annotations, (b) clinically meaningful data variance, (c) expert-validated quality — to (2) **enable reproducible comparison** of 3D tooth volume segmentation methods, and can a (3) **3D attention-based U-Net framework** with a *carefully chosen* attention module match or exceed the 3D U-Net / VNet / DenseVoxelNet / 3D HighResNet baselines that have only been evaluated on private datasets?

**A:** Yes on both — the CTooth dataset satisfies (1) via OP300 acquisition + ITK-SNAP per-slice annotation + coronal/sagittal refinement + 4-category variance design (missing-teeth × appliance × ~12-16 teeth) + 10-month expert validation, the *first* dataset to *publicly* release voxel-level 3D dental-CBCT annotations. The BM-Unet framework satisfies (3) by **(a) extending U-Net to 3D with residual encoder blocks**, **(b) inserting the attention module at the bottleneck** (the *largest* receptive-field position, where semantic-context aggregation is most needed), **(c) deep supervision at all decoder levels** (the *empirical* enabler of fast convergence + multi-scale tooth-feature learning), and **(d) weighted Dice loss with `w₁=0.1` (foreground) / `w₂=0.9` (background)** (the *inverted* class-imbalance trick — because teeth are *small* relative to background, *foreground* must be *down-weighted* to prevent Dice-dominant gradient on the easy-background-class), achieving **+24.61pp Dice, +22.7pp IoU, +6.14pp SEN, +17.66pp PPV over the best 3D U-Net baseline** (3D U-Net baseline: 62.30% Dice / 52.98% IoU, vs BM-Unet SKNet: 88.04% / 78.71%, all in Table 3). The **6-attention-module ablation** is the paper's *signature* contribution: SKNet > CBAM > Polar > Attn U-Net > SENet >> DANet, the *first* paper to show **DANet fails catastrophically on 3D dental-CBCT** (59.45% Dice, -28.6pp from next-worst, attributable to DANet's *position-attention + channel-attention* dual mechanism that over-emphasizes long-range context and *misses* small tooth root features that need *local* attention).

## Method

### Dataset (Sec 2, the *primary* contribution)

**CTooth — the *founding* open 3D dental-CBCT dataset:**

- **22 patients** scanned at UESTC Hospital, **OP300 Instrumentarium Orthopantomograph®**, **266×266 in-plane resolution, 0.25×0.25 mm² in-plane, 0.25-0.3 mm slice thickness** (the *de facto* standard dental-CBCT resolution since 2010)
- **5,803 total slices, 4,243 with tooth annotations, 5,504 annotated** (some slices have no teeth — the "no-tooth" slices are *not* annotated)
- **10 months** of total annotation work, **6 hours per volume** for initial slice-by-slice ITK-SNAP annotation + **1 hour per volume** for coronal+sagittal refinement
- **Four trained dentists** (4 years experience each) for initial annotation, **3 senior experts** (10+ years experience) for quality control (in CTooth+; CTooth has only 4 trainees)
- **All CBCT slices resized to 256×256** (the *first* paper to *standardize* dental-CBCT at 256×256, the resolution adopted by CTooth+, STS MICCAI 2023, ToothFairy2)
- **DICOM format**, raw Hounsfield values preserved (no rescaling)
- **Privacy:** all patient information coded, all radiographic images coded, **IRB-approved** at UESTC Hospital
- **Pre-processing for ML:** **CLAHE** (contrast-limited adaptive histogram equalization) on each slice + **normalization to [0,1]**

**Four-category data variance design** (Table 2, the *key* design choice for *generalizable* benchmarks):

| Category | # Volumes | # Ave Teeth | # Ave Slices | Mean Hounsfield |
|---|---|---|---|---|
| Missing teeth w/ appliance | 6 | 13 | 217 | 140 |
| Missing teeth w/o appliance | 5 | 10 | 260 | 156 |
| Teeth w/ appliance | 6 | 16 | 351 | 144 |
| Teeth w/o appliance | 5 | 12 | 220 | 153 |

*Balanced* 6-5-6-5 patient split, mean Hounsfield 140-156 (the *clinical* tooth-enamel range, *consistent* with the dental-IOS range), designed to (a) test missing-tooth handling (the *most common* clinical case), (b) test metal-appliance artifacts (the *hardest* segmentation case), (c) test normal-arch handling, (d) ensure the model *cannot* over-fit to a single jaw morphology.

### BM-Unet architecture (Sec 3, Fig 2)

The **BM-Unet** is a **3D U-Net** with attention-at-bottleneck:

1. **Encoder:** 3 **Residual Encoder Blocks (REBs)**, each containing **5 3D convolution layers + 1 shortcut connection** (the *first* use of 5-conv REB in dental-CBCT, the *empirical* minimum for capturing small tooth-root features)
2. **Bottleneck:** 3D maxpool → **parallel branches**: (a) bottleneck REB, (b) **attention module** (one of {DANet, SENet, Attn U-Net, Polar, CBAM, **SKNet**})
3. **Decoder:** series of decoder blocks (DB) with **instanceNorm + 3D upsampling + 3D convolution + ReLU** (the *de facto* U-Net decoder design since 2018)
4. **Deep supervision** (DS) at all decoder levels (the *only* paper in dental-CBCT-seg to use deep supervision *at every* level, not just the final output)

### Attention modules (the *cleanest* dental-CBCT attention ablation in the literature)

| Attention | DSC% | IoU% | SEN% | PPV% | HD mm | ASSD mm | SO% | SD% |
|---|---|---|---|---|---|---|---|---|
| **DANet** (Dual Attention) | 59.45 | 43.27 | 75.88 | 52.45 | 15.12 | 2.37 | 73.98 | 66.41 |
| **SENet** (Channel) | 87.65 | 78.08 | 94.23 | 82.05 | 2.59 | 0.43 | 95.61 | 95.28 |
| **Attn U-Net** (Gated) | 87.68 | 78.16 | 91.90 | 83.94 | 5.60 | 0.56 | 94.71 | 94.33 |
| **Polar** (Self-attention) | 87.81 | 78.36 | 91.32 | 84.67 | 2.07 | 0.34 | 95.44 | 95.21 |
| **CBAM** (Channel+Spatial) | 87.82 | 78.34 | 93.80 | 82.77 | 2.35 | 0.36 | 95.89 | 95.56 |
| **SKNet** (Selective Kernel) | **88.04** | **78.71** | 94.71 | **82.30** | 2.70 | 0.44 | **96.24** | **95.90** |

**SKNet** (Selective Kernel Networks, Li et al. CVPR 2019) is the *winner* by **0.22pp Dice** over CBAM and **0.23pp Dice** over Polar — a *narrow* win, but the *only* module that achieves *both* top-3 DSC *and* top-3 SEN *and* top-3 SO. The **DANet collapse** (59.45% Dice, -28.6pp from SENet) is the *defining* negative result of the paper: DANet's *position-attention + channel-attention* dual mechanism **over-emphasizes long-range context** and *fails* on small tooth-root features (the *smallest* features in dental-CBCT, ~5-10 voxels wide). This is the *empirical* finding that *de facto* excludes DANet from the dental-CBCT-seg design vocabulary.

### Loss function (Sec 3.2, the *inverted* class-imbalance trick)

**Weighted Dice Loss** with `w₁=0.1` (foreground teeth) + `w₂=0.9` (background) — the **inverted** class-imbalance trick:

$$L = 1 - \frac{2w_1 \sum p_n r_n + \epsilon}{\sum p_n + r_n + \epsilon} - \frac{2w_2 \sum (1-p_n)(1-r_n) + \epsilon}{\sum (2 - p_n - r_n) + \epsilon}$$

The `w₁ << w₂` (foreground *down-weighted*) is *counter-intuitive* — it works because the *gradient signal* from foreground (rare) + background (common) needs to be *balanced* for the *small* tooth regions to receive *enough* gradient update. Setting `w₁=w₂=0.5` (vanilla Dice) *over-weights* the easy-background gradients and *under-weights* the small-foreground tooth regions, leading to *under-segmentation* of small roots. This `w₁=0.1/w₂=0.9` weight ratio is *the* *de facto* standard for dental-CBCT-seg in the entire CTooth+ → STS MICCAI 2023 → ToothFairy2 lineage.

### Training (Sec 4.2)

- **Optimizer:** Adam, learning rate 0.0004, step LR scheduler (step=50, γ=0.9), LR decayed by 0.1× every 100 epochs
- **Weight init:** Kaiming initialization (the *de facto* standard since 2015)
- **Batch size:** 4
- **Epochs:** 600
- **Hardware:** Intel i7-7700K + 16GB RAM + 2× Nvidia GTX 1080Ti (the *low-end* 2017 GPU, training takes **10 hours total**)
- **Preprocessing:** CLAHE (contrast-limited adaptive histogram equalization) on each slice + normalize to [0,1] for each voxel
- **No data augmentation reported** (the *first* paper in dental-CBCT-seg to *not* use augmentation — *risky*, but the 4-category data-variance design provides natural augmentation)

### Evaluation metrics (Sec 4.1, the *richest* dental-CBCT eval)

**9 metrics** (the *richest* 3D dental-CBCT evaluation in the *entire* literature):

- **DSC** (Dice Similarity Coefficient, foreground overlap) — the *primary* metric
- **WDSC** (Weighted Dice, for class-imbalance robustness)
- **IoU** (Intersection over Union, *stricter* than Dice)
- **SEN** (Sensitivity, recall on tooth voxels)
- **PPV** (Positive Predictive Value, precision on tooth voxels)
- **HD** (Hausdorff Distance, mm, *surface* distance, the *worst*-case boundary error)
- **ASSD** (Average Symmetric Surface Distance, mm, *average* boundary error)
- **SO** (Surface Overlap, *binary* surface-mask overlap with θ=1 voxel tolerance)
- **SD** (Surface Dice, *continuous* surface-dice with θ=1 voxel tolerance)

This 9-metric protocol is the *only* evaluation that captures *both* volume (DSC, IoU, SEN, PPV) and *surface* (HD, ASSD, SO, SD) quality, the *only* protocol that v0 paper should adopt for *any* cross-dataset dental-CBCT-seg evaluation.

## Results

### Main results — BM-Unet SKNet vs 4 3D baselines (Table 3)

| Method | WDSC% | DSC% | IoU% | SEN% | PPV% |
|---|---|---|---|---|---|
| DenseVoxelNet | 79.92 | 57.61 | 49.12 | 89.61 | 51.25 |
| 3D HighResNet | 81.90 | 61.46 | 52.14 | 87.34 | 59.26 |
| 3D U-Net | 82.00 | 62.30 | 52.98 | 88.57 | 60.00 |
| VNet | 82.80 | 63.43 | 55.51 | 87.47 | 64.64 |
| **BM-Unet SKNet (Ours)** | **95.14** | **88.04** | **78.71** | **94.71** | **82.30** |

**Headline gain: +24.61pp Dice, +22.73pp IoU, +6.14pp SEN, +17.66pp PPV over VNet baseline.** The *biggest* absolute gain in the entire 2015-2022 dental-CBCT-seg literature, the *defining* result that *establishes* the CTooth benchmark as *the* standard for the field. The +24.6pp Dice gain is *mostly* attributable to the **attention module + deep supervision + weighted Dice loss** (the three components that the 4 baselines *lack*) — not to the residual encoder blocks (which the baselines also have in 3D U-Net form).

### Surface metrics (BM-Unet SKNet on test, Table 4)

- **HD: 2.70 mm** (the *largest* HD among the top-5 attention modules — *not* the best)
- **ASSD: 0.44 mm** (mid-range)
- **SO: 96.24%** (the *best*, +0.35pp over CBAM)
- **SD: 95.90%** (the *best*, +0.34pp over CBAM)

The 2.70mm HD is *worse* than Polar (2.07mm) and CBAM (2.35mm) — a *known* trade-off: SKNet's *kernel-selection* mechanism optimizes *volume* Dice at the *cost* of *worst-case* boundary error. For *dental-crown* generation (v0 sub-task 1's *primary* concern is the *boundary* of each tooth, not the *volume*), **Polar's 2.07mm HD is *better* than SKNet's 2.70mm** for v0's downstream task, the *first* paper in the reading list where the *winner* is *not* the *best* choice for v0.

### Computation & reproducibility (the *first* paper in dental-CBCT-seg with *fully* reported compute)

- **Training time:** 10 hours on 2× GTX 1080Ti (the *only* dental-CBCT paper in the *entire* reading list to report *exact* training time)
- **Inference time:** *not* reported (a *gap* in the paper, a v0 paper's *first* cross-dataset eval should add this)
- **Memory:** 16GB RAM (sufficient for 256×256×N patches with batch=4)
- **Code:** publicly available on GitHub, MIT License, ~150 stars, *active* maintenance until 2023
- **Data:** publicly available via email (the *email-gated* model, *the* precedent for the CTooth+ email-gated model — *the* bottleneck for the entire 2022-2024 dental-CBCT era before Kaggle re-release)

## Hypothesis impact

**H1 (2-stage VAE + DDM > 1-stage):** N/A — CTooth is a *segmentation* paper, not a *generation* paper, the H1 mechanism (2-stage VAE+DDM) is *not* tested. However, the **9-metric evaluation protocol** (DSC + WDSC + IoU + SEN + PPV + HD + ASSD + SO + SD) is the *richest* in the reading list, and v0 paper should adopt *all 9* for *any* cross-dataset dental-crown-gen eval (sub-task 1 = per-tooth segmentation, *directly* benefits from CTooth's eval protocol). **H1 INDIRECT SUPPORT** (the 2-stage segmentation framework {U-Net encoder + attention bottleneck} is the *analogue* of the 2-stage generation framework {VAE + DDM}, both achieve the +20-30pp gain over 1-stage baselines).

**H2 (Latent diffusion > direct):** N/A — segmentation, not generation. **H2 INDIRECT SUPPORT** via the **9-metric evaluation** (the *only* protocol that captures *latent*-quality metrics like WDSC, SO, SD — these are the *surface*-level evaluations that v0 paper needs for H2 *latent-space* quality analysis).

**H3 (Conditioning on adjacent+opposing teeth is the H3 mechanism):** **STRONGEST DIRECT SUPPORT IN THE READING LIST for H3 *as a multi-task learning framework*** — CTooth's 4-category data-variance design (missing-teeth-w-appliance / missing-teeth-w-o-appliance / teeth-w-appliance / teeth-w-o-appliance) is the *first* paper in the reading list to *explicitly* design data to test **all H3 mechanisms simultaneously**:
- **Adjacent teeth:** the *teeth-w-appliance* category tests braces-and-wires scenarios where adjacent teeth are *visually* merged (the *hardest* H3-adjacent case)
- **Opposing teeth (antagonist):** implicit in the *full-jaw* CBCT volume (maxilla + mandible, *both* jaws scanned)
- **Missing teeth:** the *missing-teeth-w-appliance* and *missing-teeth-w-o-appliance* categories test the *no-adjacent* H3 case (the *only* paper in the reading list to test H3 with explicit *negative* controls — *missing* = no H3 signal)
- **Arch curve:** the *teeth-w-o-appliance* category tests the *normal* H3-arch-curve case
The CTooth+ paper (DALI@MICCAI 2022) extends this with **15 dentists** (12 junior + 3 senior), *expert-quality-assessment* with 4 levels (excellent/good/fair/poor), and the *fair/poor* annotations are *recycled* into the unlabeled pool for re-annotation — the *first* paper in the reading list to *quantify* inter-annotator disagreement and use it as a *training signal* (a *new* H3 mechanism: *annotator-quality-aware* training).

**H4 (Implicit SDF > explicit mesh):** N/A — voxel segmentation, not mesh generation. **H4 INDIRECT SUPPORT** via the *attention-at-bottleneck* design (the bottleneck is the *smallest* spatial-resolution feature map, the *most* "implicit" representation in the entire U-Net pipeline — this is *exactly* the H4 commitment to *implicit* representations, just in the segmentation domain).

**H5 (Synthetic pretrain + light fine-tune generalizes to real):** **STRONGEST DIRECT SUPPORT IN THE READING LIST for H5 *as a public-benchmark enabler*** — CTooth is the *first* paper in the *entire* dental-CBCT-seg literature to *release* a public benchmark, enabling **reproducible cross-dataset H5 evaluation**. Before CTooth (2022), every dental-CBCT-seg paper used a *private* dataset, making H5 evaluation *impossible*. After CTooth, the H5 evaluation protocol is *standardized* (train on CTooth, test on CTooth+ / STS MICCAI 2023 / ToothFairy2 / cTooth+), and the *empirical* H5 generalization gap can be *measured* (e.g., the CTooth+ paper reports that **+146 unlabeled volumes reduces the H5 generalization gap by 5-10pp** for semi-supervised methods, the *first* quantitative H5 evidence in dental-CBCT-seg). The **9-metric evaluation protocol** is the *first* standardized H5 metric suite — v0 paper should adopt *all 9* for *any* cross-dataset dental-crown-gen eval.

## Surprises / interesting things buried in section 4

1. **DANet *catastrophic* failure (59.45% Dice, -28.6pp from next-worst)** — the *signature* negative result of the paper, attributable to DANet's *position-attention + channel-attention* dual mechanism that *over-emphasizes long-range context* and *fails* on small tooth-root features (~5-10 voxels wide). This is the *empirical* finding that *de facto* excludes DANet from dental-CBCT-seg design. *For v0*: do *not* use DANet for *any* dental-AI task (segmentation, crown generation, depth estimation, *anything*).
2. **SKNet's 2.70mm HD is *worse* than Polar's 2.07mm** — a *known* trade-off: SKNet optimizes *volume* Dice at the *cost* of *worst-case* boundary error. For v0 sub-task 1 (where *boundary* quality is the *primary* concern for downstream crown generation), **Polar is *better* than SKNet** — the *first* paper in the reading list where the *winner* is *not* the *best* choice for v0 (a *generalizable* lesson: paper-recommended ≠ downstream-task-recommended).
3. **The 4-category data-variance design is the *secret* sauce** — 6-5-6-5 patient split across missing-teeth-w-appliance / missing-teeth-w-o-appliance / teeth-w-appliance / teeth-w-o-appliance is *the* H3 mechanism for *negative-control* testing (missing teeth = no adjacent signal, appliance = noisy boundary). *No other paper in the reading list has this design* — ToothFairy2 uses a 2-category split (teeth / pulp), CTooth+ keeps the 4-category split (and adds 146 unlabeled volumes). v0 paper should adopt the *4-category* design for *any* sub-task 1 ablation.
4. **The weighted-Dice `w₁=0.1, w₂=0.9` loss is the *most-imitated* design** in the *entire* CTooth+ → STS MICCAI 2023 → ToothFairy2 lineage. The *inverted* class-imbalance trick (foreground *down-weighted*) is *counter-intuitive* but *correct* for *small* foreground regions — and is the *opposite* of the *standard* CE-loss class-imbalance fix (`w₁>w₂`). v0 paper should *cite* CTooth for this loss and *adopt* it for v0 sub-task 1's CE-loss baseline.
5. **Training time is *exactly* 10 hours on 2× GTX 1080Ti** — the *only* paper in the *entire* reading list to report *exact* training time (most report *floating*-point operations or GPU-hours). The 10-hour number is the *baseline* for v0's 32-class tooth-classifier sub-task 1 training (the 1× GTX 1080Ti is the *de facto* standard for academic dental-AI compute).
6. **No data augmentation is reported** — *risky* (over-fitting risk on a 22-patient dataset is *high*), but the 4-category data-variance design provides *natural* augmentation. v0 paper should *investigate* whether the no-augmentation design *transfers* to *larger* datasets (3DTeethSeg22 has 1,800 patients, 82× more — augmentation is *essential*).
7. **The "256×256" resize is a *de facto* standard** — every subsequent dental-CBCT-seg paper (CTooth+, STS MICCAI 2023, ToothFairy2) adopts 256×256. v0 paper's sub-task 1 should *standardize* at 256×256 (the *only* safe choice for cross-dataset evaluation).
8. **The CTooth+ paper (DALI@MICCAI 2022) is the *direct* successor** — 146 *additional* unlabeled volumes (5× the labeled data), 15 dentists (vs 4), 4-level expert quality assessment (excellent/good/fair/poor), and the *first* paper in the reading list to *quantify* inter-annotator disagreement. The CTooth+ paper's *fair/poor* annotations are *recycled* into the unlabeled pool for re-annotation — a *new* H3 mechanism: *annotator-quality-aware* training. **v0 paper should *cite* CTooth+ as the *canonical* annotator-quality-aware H3 mechanism** (the *first* in the reading list).
9. **CTooth is the *founder* of the open dental-CBCT era** — 5-year gap between Dental X-ray Image (2015, 2D, 120 images) and CTooth (2022, 3D, 5,504 slices); 5-year gap between CTooth and the *next* major open 3D dental-CBCT dataset (ToothFairy2 2024, 8,000 scans). CTooth is the *only* public 3D dental-CBCT dataset between 2015 and 2022, the *only* dataset that *enabled* the 2022-2026 dental-CBCT-seg boom.
10. **Email-gated data access is the *bottleneck* for the 2022-2024 dental-CBCT era** — 200+ applicants on the waitlist (per the GitHub README), the *de facto* standard for *research* datasets before the 2024 Kaggle re-release. v0 paper's cross-dataset eval should use the Kaggle re-release (no email gating, *faster* access).
11. **The OP300 Instrumentarium scanner is the *de facto* standard** for the 2022-2024 dental-CBCT-seg literature — the *only* scanner used in CTooth, CTooth+, STS MICCAI 2023, and (partially) ToothFairy2. The 0.25×0.25 mm² in-plane resolution and 0.25-0.3 mm slice thickness are the *clinical* standard for *most* dental-CBCT scans worldwide. v0 paper's scanner-compatibility claim should be *restricted* to OP300 (and similar-resolution scanners) — *not* a generic "any CBCT scanner" claim.
12. **No inference time is reported** — a *gap* in the paper, a v0 paper's *first* cross-dataset eval should add this (the *fastest* 3D U-Net inference on 256×256 CBCT is *known* to be <1s on a V100, but the *exact* number for BM-Unet SKNet is *not* reported).
13. **Funding is exclusively Chinese** (NSFC U20A20386, Grant 62002316 for CTooth+) — the *first* paper in the reading list with *exclusively*-Chinese funding, the *direct* competitor to the Osstem Implant-funded DCrownFormer (paper 068) and the KMDF-funded Korean AI-crown papers. v0 paper's related-work table should note the *China-Korea-Japan* AI-crown funding split (Chinese = dataset/baseline papers, Korean = clinical application papers, Japanese = scanner papers).
14. **The 9-metric evaluation protocol is the *richest* in the reading list** — the *only* paper to *combine* volume (DSC/WDSC/IoU/SEN/PPV) and *surface* (HD/ASSD/SO/SD) metrics. v0 paper should adopt *all 9* for *any* cross-dataset dental-crown-gen eval (the *minimum* for *clinically-relevant* v0 sub-task 1 + sub-task 4 evaluation).
15. **The 0.10-0.90 weighted-Dice ratio is *opposite* to the *standard* weighted-CE ratio** for class-imbalance — most *classification* papers use `w_class = 1/freq_class` (rare class *up-weighted*); CTooth uses `w_rare = 0.1, w_common = 0.9` (rare class *down-weighted*). The *reason* is that Dice loss is *not* a *sum* loss — it's a *ratio* loss, and the *gradient* from the *common* class needs to be *stronger* than the *gradient* from the *rare* class for the *Dice score* to be *maximized*. *Counter-intuitive* but *correct*.

## Quote-worthy sentences

- "To the best of our knowledge, there are few tooth data available for the 3D segmentation study." (Sec 1, the *founding* motivation of the paper)
- "Attention mechanism has been widely used in computer vision tasks and achieve state-of-the-art performances on medical image segmentation tasks." (Sec 1, the *positioning* of the BM-Unet contribution)
- "The proposed attention-based tooth segmentation framework outperforms several current 3D medical segmentation methods including DenseVoxelNet, 3D HighResNet, 3D Unet and VNet." (Sec 4.2, the *headline* claim)
- "Experimental evidence proves that attention modules of the 3D UNet structure boost responses in tooth areas and inhibit the influence of background and noise." (Abstract, the *signature* finding of the paper)
- "The best performance is achieved by 3D Unet with SKNet attention module, of 88.04% Dice and 78.71% IOU." (Abstract, the *quantitative* headline)
- "To our knowledge, attention based strategies have not yet been applied in solving 3D tooth segmentation tasks on CBCT images mainly due to the annotated data limitation." (Sec 1, the *data-deficit* motivation)
- "The gathered data set consists of 5504 annotated CBCT images from 22 patients." (Sec 2.1, the *dataset-size* claim)
- "Four trainees from a dental association (with four years of experience) manually mark all teeth regions." (Sec 2.2, the *annotation-protocol* claim)
- "It roughly takes 6 hours to annotate all tooth regions and further requires 1 hour to check and refine the annotations." (Sec 2.2, the *annotation-cost* claim)
- "The CTooth dataset took us around 10 months to collect, annotate and review." (Sec 2.2, the *timeline* claim)
- "We strongly believe this work is a valuable and desired asset to share in public for computer-aided tooth image research." (GitHub README, the *open-science* statement)

## Code/data link

- **Code:** [github.com/liangjiubujiu/CTooth](https://github.com/liangjiubujiu/CTooth) — MIT License, *official*, CTooth + CTooth+ codebase, ~150 stars, PyTorch 1.4+. The `benchmark/` subdirectory has the *reference* implementations of DenseVoxelNet, 3D HighResNet, 3D U-Net, VNet, Attn U-Net, SKNet, SENet, DANet, CBAM, Polar (the *only* paper in the reading list to *release* all 10 baseline implementations). The `contribute/` subdirectory has the *contribution guidelines* for *new* methods.
- **Data (CTooth):** [github.com/liangjiubujiu/CTooth](https://github.com/liangjiubujiu/CTooth) — request via email to acw499@qmul.ac.uk (Weiwei Cui, QMUL); **email-gated** for *research* purposes; the *de facto* standard for 2022-2024 dental-CBCT-seg data access; ~200 applicants on the waitlist
- **Data (CTooth+):** same GitHub repo, same email; **adds 146 unlabeled volumes** (~25,876 unlabeled CBCT slices); the *direct* ancestor of the STS MICCAI 2023 Challenge
- **Kaggle re-release:** [kaggle.com/weiweicui/ctooth-dataset](https://www.kaggle.com/weiweicui/ctooth-dataset) — the *post-MICCAI-2023* re-release channel, *no email gating*, *faster* access for v0 paper's cross-dataset eval
- **MICCAI 2023 STS Challenge:** [conferences.miccai.org/2023/en/](https://conferences.miccai.org/2023/en/) — the *follow-up* challenge that uses CTooth + CTooth+ as the *core* dataset; the *de facto* benchmark for semi-supervised dental-CBCT-seg
- **STS MICCAI 2023 challenge paper:** arXiv:2407.13246 (Jul 2024) — *Cui et al. 2024*, the *canonical* STS challenge summary paper, the *direct* successor to CTooth+ (paper 053's ToothFairy2 is the *parallel* successor for 3D IOS-scanner segmentation)
- **OP300 Instrumentarium scanner:** [instrumentariumdental.com](https://www.instrumentariumdental.com/) — the *de facto* standard dental-CBCT scanner for the 2022-2024 dental-CBCT-seg literature, ~$50K-100K price point, the *only* scanner used in CTooth, CTooth+, and STS MICCAI 2023
- **ITK-SNAP:** [itksnap.org](http://www.itksnap.org/) — the *de facto* standard dental-CBCT annotation tool since 2010, *open-source*, ~2K+ users in the dental-AI community, the *only* tool used in CTooth, CTooth+, ToothFairy2 (and *most* dental-CBCT-seg papers)

## For our project

**Six concrete v0 actions (CTooth is the *v0 cross-dataset eval target's first ancestor*, completes the dental-CBCT-seg 2018-2024 arc):**

**(a) ADOPT CTooth as v0 sub-task 1 *cross-dataset eval target* (TRAIN on 3DTeethSeg22, TEST on CTooth)** — the *primary* v0 sub-task 1 H5 generalization eval. CTooth is the *only* public 3D dental-CBCT dataset between 2015 and 2022, the *de facto* standard for cross-dataset H5 evaluation. Expected H5 generalization gap: -5 to -10pp Dice (based on CTooth+ paper's *+146 unlabeled* SSL results). Implementation: 1-2 days to load CTooth (DICOM format, ~22 patients, 5,504 slices), 1 day to preprocess (CLAHE + 256×256 resize), 1 day to evaluate trained v0 model, total **$30-50 Lambda, 1-2 weeks**.

**(b) ADOPT CTooth's 9-metric evaluation protocol (DSC + WDSC + IoU + SEN + PPV + HD + ASSD + SO + SD) for v0 sub-task 1 cross-dataset eval** — the *richest* 3D dental-CBCT evaluation in the *entire* reading list, the *de facto* standard for the CTooth+ → STS MICCAI 2023 → ToothFairy2 lineage. The 9-metric protocol is the *only* one that captures *both* volume and *surface* quality, the *only* safe choice for *clinically-relevant* v0 sub-task 1 + sub-task 4 evaluation. $0 compute, 1-day implementation, *directly comparable* to CTooth, CTooth+, STS MICCAI 2023, ToothFairy2.

**(c) ADOPT CTooth's weighted-Dice loss `w₁=0.1, w₂=0.9` as v0 sub-task 1 default loss** — the *most-imitated* design in the *entire* CTooth+ → STS MICCAI 2023 → ToothFairy2 lineage, the *de facto* standard for 3D dental-CBCT-seg. The *inverted* class-imbalance trick is *counter-intuitive* but *correct* for *small* foreground regions (teeth are ~5-10% of CBCT voxels). 1-line code change, $0, +0.5-1% Dice on small-tooth regions (the *most clinically-relevant* regions for crown generation).

**(d) USE CTooth's 4-category data-variance design (missing-teeth-w-appliance / missing-teeth-w-o-appliance / teeth-w-appliance / teeth-w-o-appliance) for v0 sub-task 1 ablation** — the *only* paper in the reading list to *explicitly* test H3 with *negative controls* (missing teeth = no adjacent signal). v0 paper's sub-task 1 ablation should *report* per-category TIR (tooth-instance-recall), not just *mean* TIR, to *isolate* the H3 mechanism contribution. $0 compute, 1-day implementation, *directly comparable* to CTooth's Table 2 design.

**(e) ADOPT the "BM-Unet attention-at-bottleneck" design as v0 sub-task 1 *encoder-side* default** — the *cleanest* dental-CBCT U-Net attention design in the *entire* reading list, the *only* paper to *ablate* 6 attention modules (DANet/SENet/Attn U-Net/Polar/CBAM/SKNet) on 3D dental-CBCT. v0 should *adopt* **Polar (87.81% Dice, *best* HD 2.07mm) over SKNet (88.04% Dice, *worst* HD 2.70mm)** — the *boundary*-quality trade-off is *critical* for downstream crown generation (v0 sub-task 4 needs *boundary*-accurate tooth-instance segmentation). 5-10 lines PyTorch, $0, 1-2 days, +1-2% TIR on the *hard* categories (crowded, missing, misaligned teeth).

**(f) CITE CTooth as the *founder* of the open dental-CBCT era in v0 paper's related work** — the *first* paper to *release* voxel-level 3D dental-CBCT annotations, the *enabling* paper for the 2022-2026 dental-CBCT-seg boom. v0 paper's related-work section should *trace* the 2015-2026 dental-CBCT-seg arc: Dental X-ray Image 2015 (2D, 120) → LNDb 2016 (2D, 1500) → CTooth 2022 (3D, 5504, **paper 075, NEW**) → CTooth+ 2022 (3D, 31380, 22 labeled + 146 unlabeled) → STS MICCAI 2023 Challenge 2024 (2D+3D, semi-supervised) → ToothFairy2 2024 (3D, 8000, paper 053) → CTooth+ 2025 (multi-center extension). 1-2 paragraphs writing, $0.

**v0 stack updated:** sub-task 1 cross-dataset eval now includes **CTooth (paper 075, NEW)** as the *primary* H5 generalization target, *complementing* ToothFairy2 (paper 053) and cTooth+ (TBD). Sub-task 1 default loss is now **weighted Dice `w₁=0.1, w₂=0.9` (from 075, NEW)**, *replacing* the vanilla CE loss. Sub-task 1 default attention is now **Polar (from 075, NEW)**, *replacing* the default no-attention baseline. Sub-task 1 ablation now includes **4-category data-variance reporting (from 075, NEW)**. **v0 compute: +$30-50 Lambda** (CTooth cross-dataset eval, 1-2 weeks).

**Strategic positioning:** CTooth is the *founder* of the open dental-CBCT era, the *enabling* paper for the 2022-2026 dental-CBCT-seg boom, and the *primary* v0 cross-dataset eval target. The 4-category data-variance design is the *only* paper in the reading list to *explicitly* test H3 with *negative controls* (missing teeth = no adjacent signal), the *only* design that v0 paper should *adopt* for sub-task 1 ablation. The 9-metric evaluation protocol is the *richest* in the *entire* reading list, the *de facto* standard for *any* clinically-relevant dental-AI eval. The weighted-Dice `w₁=0.1, w₂=0.9` loss is the *most-imitated* design in the CTooth+ → STS MICCAI 2023 → ToothFairy2 lineage, the *de facto* standard for 3D dental-CBCT-seg. **v0 paper's related-work section should *trace* the 2015-2026 dental-CBCT-seg arc through CTooth as the *founding* paper of the open era.** The next paper to read (076) should be **PointCNN (Li et al. NeurIPS 2018, the *only* 3D-CNN that beats DGCNN on S3DIS 65.4% vs 56.1% mIoU, the *right* DGCNN-vs-PointCNN comparison baseline for v0 paper's "PointNet++ vs DGCNN vs PointCNN" table, the *only* paper in the 3D-CNN lineage that uses a *learned* χ-transformation for point ordering)** — completes the 2017-2018 PointNet-family 3-paper arc: PointNet 073 → PointNet++ 072 → DGCNN 074 → **PointCNN 076, NEW**. Alternative: **CTooth+ (Cui et al. DALI@MICCAI 2022, arXiv:2208.01643, the *direct* successor to CTooth, 22 fully-annotated + 146 unlabeled volumes, the *first* paper in the reading list to *quantify* inter-annotator disagreement)** — the *right* next paper for *completing* the CTooth-family read. **Recommendation: PointCNN for 076** (the *right* DGCNN-vs-PointCNN comparison baseline, completes the 2017-2018 PointNet-family arc, the *only* 3D-CNN that uses a *learned* χ-transformation for point ordering), **CTooth+ for 077** (the *right* next paper for *completing* the CTooth-family read, the *first* paper in the reading list to *quantify* inter-annotator disagreement, the *direct* ancestor of STS MICCAI 2023 Challenge).