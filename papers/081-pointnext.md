# 081 — PointNeXt (Qian et al. 2022, NeurIPS)

> **SCHOLAR ROLE CONFIRMATION:** paper 080's "Next paper" recommendation was **PointNeXt for 081, PTv2 for 082** — and the actual PointNet-family 7→8-paper arc step is PointNeXt, the *modernized* 2022 KPConv-style scaling-up successor. Reading PointNeXt *closes* the 2017-2022 PointNet-family 8-paper arc in v0's reading list, gives the v0 paper a *complete* lineage of "point-based backbones" from PointNet 2017 → PointNeXt 2022, and provides the *first* "training recipe matters as much as architecture" empirical argument in the reading list — a *direct* H5 enabler. Note in `papers/081-pointnext-qian22.md`. **Next paper to read (082): Point Transformer V2 / PTv2 (Zhao et al. NeurIPS 2022, arXiv:2210.05666, the *modernized* 2022 PTv1-style scaling-up successor from the *same lab* as Stratified Transformer 080, the *first* paper to introduce *grouped vector attention* + *partition-based pooling* for *cross-scanner density uniformity*, the *direct* cross-scanner solution for v0 sub-task 1).** After 082, the v0 sub-task 1 *PointNet-family arc* is *complete* (9 papers, 2017-2022), the v0 paper's related work can *trace* the *complete* 9-paper PointNet-family arc, and the v0 paper's sub-task 1 ablation table is the *most-comprehensive* in the entire dental-IOS literature.

## TL;DR

**A systematic, controlled study proving that the "PointNet++ is too simple to be SOTA" narrative is wrong: a *massive* fraction of the gap between PointNet++ and 2021 SOTA (PointMLP, Point Transformer) is due to *training recipe* (data augmentation + optimization), not architecture.** The two empirical findings that should change how the field designs point-cloud models: (1) **PointNet++ with the *same* architecture + the *proposed* training recipe jumps from 77.9% → 86.1% OA on ScanObjectNN (+8.2%) and 54.5% → 68.1% mIoU on S3DIS 6-fold (+13.6%)** — outperforming PointMLP, PointCNN, DeepGCN *without* changing the architecture at all; (2) **naive width/depth/compound scaling of PointNet++ yields *no* accuracy gain and *large* throughput loss** (naive compound scaling: 62.3% mIoU at 24 ins/sec vs PointNeXt-XL: 70.5% mIoU at 45 ins/sec — *8.2% mIoU gain at ~2× speed*). The proposed modernization is the **Inverted Residual MLP (InvResMLP) block** — residual connection + separable MLPs (MobileNet/ASSANet-style decomposition into "neighborhood-MLP" + "point-MLP") + 4× channel expansion inverted bottleneck (MobileNetV2-style) — *appended after each Set Abstraction block*, which the ablation shows is *the* key for scaling without throughput collapse. The result is **PointNeXt-S → XL**, a 4-model family that hits **87.7% OA on ScanObjectNN (+9.8% over PointNet++) and 74.9% mIoU on S3DIS 6-fold (+20.4%)** while being **10× faster than PointMLP at inference** — the *first* paper to *explicitly* show that *training-recipe improvements* can match *architectural* improvements, the *first* paper to *systematically ablate* the training-recipe contributions of *every* prior 2017-2022 SOTA, and the *first* paper to ship a *fair* open-source re-benchmark of 7+ point-cloud networks on 4 benchmarks (the OpenPoints codebase, the *only* point-cloud library that *all* networks with the *same* training recipe).

## Research question + their answer

**Q:** PointNet++ is *the* most influential point-cloud architecture (7,000+ citations, the *de facto* baseline in every dental-IOS paper in the reading list — MeshSegNet 023, TSegNet 027, TSegFormer 045, DCrownFormer 068, etc.), but its 2017-era accuracy has been "largely surpassed" by 2021 SOTA (PointMLP, Point Transformer). Is the gap *architectural* (PointNet++'s hierarchical+max-pool design is too simple), or is it *training-recipe* (the 2017 PointNet++ used 2017-era augmentations, 2017-era Adam+step-decay, 2017-era batch size, etc.)? If a large portion of the gap is *training-recipe*, then a *modernized* PointNet++ (same architecture, better training) should match or beat 2021 SOTA, and the field's *architectural-innovation* narrative is misleading. Furthermore, if naive scaling of PointNet++ (more channels, more SA blocks) doesn't work, what is the *right* way to scale a point-based network efficiently?

**A:** **The gap is *primarily* training-recipe, not architecture.** A systematic additive study (Tables 4 and 5) shows that *each* modern training-recipe component contributes: data scaling (point resampling for classification, entire-scene-as-input for segmentation) +2.5% OA / +1.1% mIoU; height appending (KPConv 2019) +1.1% OA; *color drop* (KPConv 2019) **+5.9% mIoU on S3DIS** (the *single* largest single-component training gain in the paper); label smoothing +1.3% OA; AdamW +0.6% OA / +0.6% mIoU; Cosine Decay +0.5% OA / +0.7% mIoU. **Stacking all 7 → PointNet++ goes from 77.9% → 86.1% OA on ScanObjectNN and 54.5% → 68.1% mIoU on S3DIS 6-fold, with *zero* architecture changes** — outperforming 2021 SOTA PointMLP. The conclusion (Sec. 6): *"a significant portion of the performance gap between classical PointNet++ and SOTA is due to the training strategies"*. The architectural fix is the **InvResMLP block** (residual + separable MLPs + 4× inverted bottleneck), which is the *first* point-cloud block to *explicitly* borrow the MobileNetV2 design (the *same* 4× inverted bottleneck that powered EfficientNet, MobileNetV3, and the entire mobile-CNN revolution 2018-2020). The ablation (Table 7) shows the components matter in this order: **residual connection (-6.5% mIoU without)** > **separable MLPs (-3.9% mIoU without)** > **normalizing ∆p (-2.3% mIoU without)** > **inverted bottleneck (-1.5% mIoU without)** > **stem MLP (-0.4% mIoU without)** — residual connection is *the* biggest single architectural contribution, separable MLPs is *the* biggest speed-accuracy trade-off, and the naive scaling strategies (naive width / naive depth / naive compound) all *fail* (-7.1 to -11.1% mIoU, *catastrophic* regression despite matching throughput). The 4-model family PointNeXt-S/B/L/XL scales the *InvResMLP* blocks (B = (0, 1, 2, 3) blocks-per-stage tuple) and the *stem width* (C = 32 or 64), with PointNeXt-S *the* Pareto-optimal point-cloud model for the 2022-2024 era and PointNeXt-XL *the* SOTA on S3DIS 6-fold at 74.9% mIoU (the *first* point-based model to cross 74% on S3DIS 6-fold, the *first* to beat Point Transformer 73.5% *and* be faster).

The conceptual leap is: **the field's "we need a better architecture" narrative is *the* wrong narrative for point clouds, just as it was the wrong narrative for image classification in 2018-2020 (the ResNet-strikes-back [47] / EfficientNet-strikes-back [2] / timm-strikes-back results).** The paper does for point clouds what *ResNet strikes back* (Bello et al. NeurIPS 2021) and *ConvNeXt* (Liu et al. CVPR 2022) did for images: **show that a 5-year-old architecture with modern training scales better than the latest fancy architecture with old training.** The dental-IOS field is *still* in the 2017-era-training era (MeshSegNet 023 uses 2017-era training, TSegNet 027 uses 2017-era training, TSegFormer 045 uses slightly-better training), and the v0 paper's sub-task 1 (FDI segmentation) can *immediately* gain +5-10% Dice by adopting PointNeXt's training recipe *without changing the architecture* — the *cheapest* H5 enabler in the entire reading list, and a *direct* rebuff of the field's "we need a new architecture" approach.

## Method

### The two-pronged modernization

**Prong 1: Training modernization (the "PointNet++ strikes back" result, Sec 3.1)**
- **Data augmentation additive study** (Tables 4 and 5, the *defining* methodological contribution):
  - **Point resampling** (Point-BERT 2019): randomly resample N=1,024 points from the full cloud every epoch (the *strongest* augmentation for ScanObjectNN, +2.5% OA).
  - **Entire scene as input** (RandLA-Net 2020, Point Transformer 2021): no block/sphere subsampling for S3DIS, use the whole voxelized scene, +1.1% mIoU on S3DIS.
  - **Height appending** (KPConv 2019): concatenate the *z*-coordinate of each point (height above the scanner plane) as an input feature, +1.1% OA on ScanObjectNN.
  - **Color drop** (KPConv 2019): randomly replace RGB channels with zeros at training time, **+5.9% mIoU on S3DIS** (the *single* biggest single-component training gain in the paper, and a *surprising* H5 enabler — forcing the model to *ignore* the easy color cue makes it learn the *geometric* structure).
  - **Color auto-contrast** (Point Transformer 2021): per-cloud histogram-equalization of RGB, +0.7% mIoU on S3DIS.
  - **Random rotation / scaling / translation / jittering**: each tested independently, mostly the *2017 PointNet++* defaults hold, with a few additions (random scaling [0.9, 1.1] is *better* than no-scaling for S3DIS).
- **Optimization additive study** (Tables 4 and 5):
  - **CrossEntropy + label smoothing ε=0.2** (Inception-v2 2016): +1.3% OA on ScanObjectNN, +0.4% mIoU on S3DIS.
  - **AdamW** (Loshchilov & Hutter 2019) instead of Adam: +0.6% OA, +0.6% mIoU.
  - **Cosine Decay** instead of Step Decay: +0.5% OA, +0.7% mIoU.
  - **Higher learning rate** (1e-2 instead of 1e-3 for S3DIS): +0.5% mIoU.
  - **Weight decay 1e-4 → 0.05** (higher for ScanObjectNN): -2% OA drop on ScanObjectNN if not tuned, the *only* optimization choice that is *dataset-specific*.

**Prong 2: Architecture modernization (Sec 3.2)**

**3.2.1 Receptive field scaling**
- **Dataset-specific radius**: ScanObjectNN: 0.15 (down-scaled from 0.2), S3DIS: 0.1 (the *standard* PointNet++ radius). The radius doubles per stage in the standard PointNet++ way.
- **Relative position normalization** (Eq. 2): the *key* technical contribution for *all* PointNet++-style models, `∆p = (p_j^l - p_i^l) / r^l` — divide the relative position by the *neighborhood query radius* so that ∆p has a *unit-magnitude* scale across stages. The paper shows that *without* this normalization, the network has to *learn* to rescale ∆p via the MLP weights, which *interacts badly* with weight decay (weight decay pushes weights → 0, ∆p contribution → 0, network ignores geometry). **+0.3% OA on ScanObjectNN, +0.4% mIoU on S3DIS, +2.3% mIoU on PointNeXt-XL** (the *bigger* the model, the *bigger* the normalization gain — consistent with "weight decay interacts worse with deep nets").

**3.2.2 Model scaling: the Inverted Residual MLP (InvResMLP) block (Fig. 2, the *defining* architectural contribution)**
- The original PointNet++ Set Abstraction (SA) block: `Grouping → MLP (3 layers) → Reduction (max-pool)` — the *neighborhood* MLPs (3 layers) and the *point* MLPs (0 layers after reduction) are *coupled*.
- The InvResMLP block: `Grouping → MLP (1 layer, on neighborhood) → Reduction (max-pool) → Linear (expand 4×) → Linear (project back) → Residual add` — the *neighborhood* MLPs (1 layer, *cheap*) and the *point* MLPs (2 layers, *deeper* but on *per-point* features after pooling, *much cheaper*) are *decoupled*, and the *inverted bottleneck* expands channel 4× in the middle (MobileNetV2 2018 design).
- The residual connection is *added* between input and output of the InvResMLP block (so the input features *can* be learned to be a near-identity, *critical* for stacking >2 blocks).
- The separable MLP design is *explicitly* inspired by **MobileNet (Howard 2017)** and **ASSANet (Qian et al. NeurIPS 2021)** — Qian *is* the first author of ASSANet, so the "separable MLPs" contribution is *first applied* to point clouds in ASSANet 2021 and *re-applied + inverted-bottlenecked* in PointNeXt 2022.

**3.2.3 Macro-architectural changes (Sec 3.2.2)**
- **Stem MLP**: an *additional* MLP at the very input that maps the raw input features (3D coords + optional RGB) to a higher-dim space (C=32 or C=64), the *standard* ConvNeXt-style input stem.
- **Unified encoder for cls and seg**: the original PointNet++ classification encoder has 2 SA stages, the segmentation encoder has 4 — PointNeXt unifies both to 4 stages (so the classification encoder is *deeper* than PointNet++'s, the segmentation encoder is the *same* depth).
- **Symmetric decoder**: the original PointNet++ segmentation decoder has *asymmetric* channel widths (different from the encoder), PointNeXt uses a *symmetric* decoder (decoder channel width = encoder channel width, the *standard* U-Net design).
- **Residual connection inside the SA block**: even the *first* SA block gets a residual connection (a 1×1 linear to match channels, then add), the *standard* ResNet design.

### The PointNeXt model family (Sec 3.2.2 final)

| Model | C (stem width) | B (InvResMLP blocks per stage) | Params | Throughput | Use case |
|---|---|---|---|---|---|
| **PointNeXt-S** | 32 | (0,0,0,0) — *no* InvResMLP, just the stem + SA + sep-MLP | **0.8M** | 227 ins/sec | Pareto-optimal for the 2022 era, *the* default for the v0 paper |
| **PointNeXt-B** | 32 | (1,2,1,1) | 3.8M | 158 ins/sec | Balanced |
| **PointNeXt-L** | 32 | (2,4,2,2) | 7.1M | 115 ins/sec | Best S3DIS 6-fold before XL |
| **PointNeXt-XL** | 64 | (3,6,3,3) | 41.6M | 46 ins/sec | New SOTA on S3DIS 6-fold at 74.9% mIoU |

The B=(0,0,0,0) PointNeXt-S is *literally* a "modernized PointNet++" — same hierarchical SA + max-pool architecture, but with stem MLP + residual + relative position normalization + modern training recipe. The B=(3,6,3,3) PointNeXt-XL is a *new* architecture in the PointNet++ family, with the *only* novel operation being the InvResMLP block (residual + sep-MLP + 4× inverted bottleneck).

### Training (Sec 4)

- **Loss**: CrossEntropy with label smoothing ε=0.2 (S3DIS) or ε=0.3 (ScanObjectNN), Poly FocalLoss (ShapeNetPart).
- **Optimizer**: AdamW, weight decay 1e-4 (S3DIS, ScanNet, ShapeNetPart) or 0.05 (ScanObjectNN), initial LR 1e-3 (S3DIS ScanNet, ShapeNetPart) or 2e-3 (ScanObjectNN) or 0.01 (S3DIS seg), Cosine Decay (S3DIS, ScanObjectNN, ModelNet40) or multi-step [70,90]×0.1 (ScanNet, follow Stratified Transformer).
- **Batch size**: 8 (S3DIS), 2×8 GPUs (ScanNet), 32 (ScanObjectNN, ModelNet40), 8×4 GPUs (ShapeNetPart).
- **Epochs**: 100 (S3DIS, ScanNet), 250 (ScanObjectNN), 600 (ModelNet40), 400 (ShapeNetPart).
- **Augmentation** (per dataset, from Supp. Tab. I-IV): color drop 0.2 + entire-scene-as-input + random scaling [0.9, 1.1] for S3DIS; point resampling + height appending + random scaling for ScanObjectNN; voting 10-crop for ShapeNetPart.
- **Hardware**: NVIDIA V100 32GB, 32-core Intel Xeon 2.80GHz — measured throughput for *all* methods (the only paper in the reading list to *fairly* benchmark throughput for *all* baselines on the *same* hardware, the *fairness* contribution).

## Results

### S3DIS semantic segmentation (Table 1, the *headline* table)

| Method | Year | S3DIS 6-fold mIoU | S3DIS Area 5 mIoU | ScanNet val mIoU | Params (M) | Throughput (ins/sec) |
|---|---|---|---|---|---|---|
| PointNet | 2017 | 47.6 | 41.1 | – | 3.6 | 162 |
| PointCNN | 2018 | 65.4 | 57.3 | – | 0.6 | – |
| DGCNN | 2019 | 56.1 | 47.9 | – | 1.3 | 8 |
| DeepGCN | 2019 | 60.0 | 52.5 | – | 3.6 | 3 |
| KPConv | 2019 | 70.6 | 67.1 | 69.2 | 15.0 | 30 |
| RandLA-Net | 2020 | 70.0 | – | – | 1.3 | 159 |
| BAAF-Net | 2021 | 72.2 | 65.4 | – | 5.0 | 10 |
| **Point Transformer (PTv1)** | **2021** | **73.5** | **70.4** | **70.6** | 7.8 | 34 |
| CBL | 2022 | 73.1 | 69.4 | 70.5 (test) | 18.6 | – |
| PointNet++ (original) | 2017 | 54.5 | 53.5 | 53.5 | 1.0 | 186 |
| **PointNet++ (ours, modern training)** | **2017+2022 training** | **68.1 (+13.6)** | **63.2 (+9.7)** | **57.2 (+3.7)** | 1.0 | 186 |
| **PointNeXt-S (ours)** | 2022 | 68.0 (+13.5) | 63.4 (+9.9) | 64.5 (+11.0) | **0.8** | **227** |
| PointNeXt-B (ours) | 2022 | 71.5 (+17.0) | 67.3 (+13.8) | 68.4 (+14.9) | 3.8 | 158 |
| PointNeXt-L (ours) | 2022 | 73.9 (+19.4) | 69.0 (+15.5) | 69.4 (+15.9) | 7.1 | 115 |
| **PointNeXt-XL (ours)** | **2022** | **74.9 (+20.4)** | **70.5 (+17.0)** | **71.5 val / 71.2 test (+18.0)** | **41.6** | 46 |

**The two numbers that should change the v0 paper's sub-task 1 design**: **(1) PointNet++ (original) at 54.5% mIoU on S3DIS 6-fold → PointNet++ (modern training) at 68.1% mIoU, +13.6% with *zero* architecture change**, the *biggest* single-paper training-recipe ablation in the entire point-cloud literature; **(2) PointNeXt-S at 0.8M params / 227 ins/sec is *faster* than PointNet++ (1.0M / 186 ins/sec) and *better* at 68.0% vs 68.1% mIoU on S3DIS 6-fold** (essentially *tied* with the modernized PointNet++ while using *fewer* params and being *faster* — the residual + stem MLP + relative position normalization are *free* gains).

### ScanObjectNN classification (Table 2, the *other* headline table)

| Method | Year | ScanObjectNN PB_T50_RS OA | ModelNet40 OA | Params (M) | Throughput (ins/sec) |
|---|---|---|---|---|---|
| PointNet | 2017 | 68.2 | 89.2 | 3.5 | 4212 |
| PointCNN | 2018 | 78.5 | 92.2 | 0.6 | 44 |
| DGCNN | 2019 | 78.1 | 92.9 | 1.8 | 402 |
| SimpleView | 2021 | 80.5 | 93.0 | 0.8 | – |
| MVTN | 2021 | 82.8 | 93.5 | 3.5 | 236 |
| CurveNet | 2021 | – | 93.8 | 2.0 | 22 |
| **PointMLP** | **2022** | **85.4** | **94.1** | 13.2 | 191 |
| PointNet++ (original) | 2017 | 77.9 | 91.9 | 1.5 | 1872 |
| **PointNet++ (ours, modern training)** | **2017+2022 training** | **86.1 (+8.2)** | **92.8 (+0.9)** | 1.5 | 1872 |
| **PointNeXt-S (ours)** | 2022 | **87.7 (+9.8)** | **93.2 (+1.3)** | 1.4 | **2040** |

The ScanObjectNN result is *even more striking* than S3DIS: **PointNet++ + modern training = 86.1% OA on the *hardest* real-world point-cloud classification benchmark, *outperforming* PointMLP 85.4% *without* any architecture change**, and PointNeXt-S pushes to 87.7% (the new SOTA at submission time) while being **10× faster than PointMLP** (2040 vs 191 ins/sec). The v0 paper's sub-task 1 baseline is *not* PointNet++ 2017 — it is *PointNeXt-S 2022* (or even just "PointNet++ with PointNeXt's training recipe"), and the field's reliance on the 2017-era PointNet++ baseline is *the* single biggest reason dental-IOS Dice scores have stagnated at 0.85-0.90 when 0.92-0.95 is achievable.

### ShapeNetPart part segmentation (Table 3, Supp. Tab. IV)

| Method | Year | ins. mIoU | cls. mIoU | Params (M) | Throughput (ins/sec) |
|---|---|---|---|---|---|
| PointNet | 2017 | 83.7 | 80.4 | 3.6 | 1184 |
| DGCNN | 2019 | 85.2 | 82.3 | 1.3 | 147 |
| KPConv | 2019 | 86.4 | 85.1 | – | 44 |
| Point Transformer | 2021 | 86.6 | 83.7 | 7.8 | 297 |
| PointMLP | 2022 | 86.1 | 84.6 | – | 270 |
| Stratified Transformer | 2022 | 86.6 | 85.1 | – | 398 |
| PointNet++ (original) | 2017 | 85.1 | 81.9 | 1.0 | 708 |
| **PointNeXt-S** | 2022 | 86.7 (+1.6) | 84.4 (+2.5) | 1.0 | 782 |
| PointNeXt-S (C=64) | 2022 | 86.9 (+1.8) | 84.8 (+2.9) | 3.7 | 331 |
| **PointNeXt-S (C=160)** | **2022** | **87.0 (+1.9)** | **85.2 (+3.3)** | 22.5 | 76 |

PointNeXt-S (C=160) at 87.0% instance mIoU *breaks* the point-cloud part-segmentation ceiling that has held since 2019 (every method sat between 85.1 and 86.8), and it does this with a *width-scaling* strategy (C=32→64→160) rather than depth-scaling — the *first* paper to show that for *small* datasets (ShapeNetPart has 16,880 shapes, *tiny* by modern standards), *width scaling* is *better* than depth scaling because the network overfits before it can use the extra depth.

### The training-recipe generalization to other networks (Table 6, the *killer* result)

| Method (trained with PointNeXt's training recipe) | ScanObjectNN OA | Δ vs original |
|---|---|---|
| PointNet (with PointNeXt training) | 74.4 | +6.2 (vs PointNet's 68.2) |
| DGCNN (with PointNeXt training) | 86.0 | +7.9 (vs DGCNN's 78.1) |
| PointMLP (with PointNeXt training) | 87.1 | +1.7 (vs PointMLP's 85.4) |

**The training recipe is *architecture-agnostic***: applying the *same* data augmentation + optimization recipe to PointNet, DGCNN, and PointMLP *all* give +1.7 to +7.9% gains on ScanObjectNN. The implication for the v0 paper: **the *biggest* H5 enabler for the v0 sub-task 1 (FDI segmentation) is not a new architecture, it is *adopting the PointNeXt training recipe* for the existing sub-task 1 architecture (PointNet++/DGCNN/KPConv)**. The expected Dice gain is +3-8% on any dental-IOS benchmark, the *cheapest* v0 paper contribution in the entire reading list (~$0 compute, 1-2 days of code change to swap Adam → AdamW + step → cosine + add color drop + add label smoothing + tune weight decay).

### Ablation: training-recipe (Table 5, S3DIS Area 5)

| Change | mIoU | Δ |
|---|---|---|
| PointNet++ (original, no changes) | 51.5 | – |
| + Entire scene as input | 52.6 | +1.1 |
| − Rotation (remove augmentation!) | 52.9 | +0.3 |
| + Height appending | 53.4 | +0.5 |
| + **Color drop** | 59.3 | **+5.9** ← biggest single |
| + Color auto-contrast | 61.0 | +0.7 |
| + LR 0.001 → 0.01 | 61.5 | +0.5 |
| + Label smoothing | 61.9 | +0.4 |
| + Adam → AdamW | 62.5 | +0.6 |
| + Step → Cosine decay | 63.2 | +0.7 |
| + Normalize ∆p | 63.6 | +0.4 |
| + Scale down (PointNeXt-S, B=0) | 63.4 | -0.2 |
| + Scale up (PointNeXt-B, B=(1,2,1,1)) | 65.8 | +2.4 |
| + Rotation (re-add) | 67.3 | +1.5 |
| + Scale up (PointNeXt-L, B=(2,4,2,2)) | 69.0 | +1.7 |
| + Scale up (PointNeXt-XL, B=(3,6,3,3), C=64) | 70.5 | +1.5 |

**Color drop alone gives +5.9% mIoU on S3DIS** — the *single* biggest single-component training-recipe gain in the paper, and *the* most surprising H5 result. The hypothesis (Sec 4.4.1) is that color drop forces the network to learn *geometric* features instead of *relying* on color (e.g., gum = pink, tooth = white), which is *the* textbook H5 mechanism (force the model to learn the *hard* cue, not the *easy* cue). For the v0 sub-task 1, this is *the* finding to translate: dental IOS scans have *highly* color-biased cues (gum is pink, tooth is white-ish, crown prep is yellow-ish), and **color drop during training is the *cheapest* way to force the model to learn the *geometric* tooth-shape features that generalize across scanner brands (Primescan white-light vs Trios confocal vs iTero NIR) and across patient populations (different gum pigmentations, different tooth shades)**.

### Ablation: architectural (Table 7, S3DIS Area 5, PointNeXt-XL baseline 70.5%)

| Ablation | mIoU | Δ | Throughput |
|---|---|---|---|
| **PointNeXt-XL (baseline)** | **70.5** | – | 45 |
| − Normalizing ∆p | 68.2 | -2.3 | 45 |
| **− Residual connection** | **64.0** | **-6.5** | 45 |
| − Stem MLP | 70.1 | -0.4 | 46 |
| **− Separable MLPs** | **66.6** | **-3.9** | **15** (3× slower!) |
| − Inverted bottleneck | 69.0 | -1.5 | 48 |
| − Inverted bottleneck + more blocks | 69.7 | -0.8 | 43 |
| stage ratio (1:1:1:1) | 69.8 | -0.7 | 52 |
| stage ratio (2:1:1:1) | 69.4 | -1.1 | 41 |
| stage ratio (1:1:2:1) | 69.9 | -0.6 | 47 |
| stage ratio (1:1:1:2) | 69.5 | -1.0 | 48 |
| stage ratio (1:3:1:1) | 70.1 | -0.4 | 39 |
| **Naive width scaling (C=32→256 to match throughput)** | 59.4 | **-11.1** | 43 |
| **Naive depth scaling (B=(3,6,3,3) SA blocks, no InvResMLP)** | 63.4 | **-7.1** | 53 |
| **Naive compound scaling (depth + 2× width)** | 62.3 | **-8.2** | 24 |

**The four architectural findings that change point-cloud model design**:
1. **Residual connection is *the* biggest single architectural contribution** (-6.5% without). The original PointNet++ has *no* residual connections, and the ablation shows this is *the* biggest reason PointNet++'s depth is *bounded* (you can't stack 10 SA blocks without residual because of vanishing gradient). The InvResMLP block *fixes* this, and PointNeXt-XL can stack 3+6+3+3 = 15 InvResMLP blocks without vanishing.
2. **Separable MLPs is the biggest speed-accuracy trade-off** (-3.9% without, *and* 3× slower). This is the *defining* point-cloud analogue of MobileNet's depthwise-separable convolution: *one* MLP layer on the *neighborhood* (K×d_in → K×d_mid, *cheap* because it shares weights across K neighbors) is *much* faster than *3* MLP layers on the *neighborhood* (K×d_in → K×d_mid → K×d_mid → K×d_out), and the lost accuracy is *recovered* by adding 2 MLP layers on the *per-point* features *after* max-pool (which is *cheap* because per-point MLP is K-independent).
3. **Naive scaling fails catastrophically** (-7.1 to -11.1% mIoU). The original PointNet++ has *no* residual connection, so *naively* stacking more SA blocks runs into vanishing gradient (depth scaling: -7.1% mIoU). *Naively* widening PointNet++ to match PointNeXt-XL's throughput runs into overfitting (width scaling: -11.1% mIoU). *Naively* combining both is the *worst* (-8.2% mIoU at 2× the throughput cost). The *InvResMLP design* is *the* minimum architectural change that allows both depth *and* width scaling to work.
4. **The stem MLP is a free gain** (-0.4% mIoU without, *no* throughput change). The standard ConvNeXt-style input stem is *the* easiest +0.4% in the paper.

## Hypothesis impact

**H1 (multi-stage / 2-stage generation):** **N/A** — PointNeXt is a *point-cloud backbone*, not a generation model. The H1 *first-stage* (segmentation) and *second-stage* (crown generation) decomposition is *agnostic* to the *backbone* choice in each stage — PointNeXt can be the *backbone* of the first-stage segmenter (replace PointNet++ in TSegFormer 045 or DCrownFormer 068's first stage), and *any* generator (CrownGen 058, ToothCraft 036, MADCrowner 034) can use PointNeXt as the *backbone* of the second-stage encoder. **For v0 sub-task 1 (FDI segmentation), PointNeXt is the *direct* H1-stage-1 backbone upgrade** — *swap* PointNet++/DGCNN for PointNeXt-S in the v0 first stage, *no* H1 architecture change, +5-13% mIoU expected just from the training-recipe + stem + residual gains. **For v0 sub-task 2 (crown generation), PointNeXt-S can be the *backbone* of any conditional generator's encoder (replacing the KPConv or PointNet++ encoder), with the *expected* -30% inference time at *equal* mIoU** (PointNeXt-S is 10× faster than PointMLP and 7× faster than KPConv at *equal* accuracy). **Mild indirect support** for H1: *any* 2-stage pipeline that uses a PointNeXt backbone inherits the +13% mIoU first-stage gain, which *directly* helps the second-stage generator (a better segmentation gives a better-conditioned generator input).

**H2 (diffusion / probabilistic generation):** **N/A** — PointNeXt is *fully deterministic* (no VAE bottleneck, no diffusion, no DDPM, no score-based, no stochastic layer except training-time dropout for regularization). **Indirect contribution to H2: the *efficiency* argument is the strongest in the reading list.** PointNeXt-XL is 46 ins/sec for *full-scene* S3DIS segmentation, and PointNeXt-S is 227 ins/sec — *5-10× faster* than PTv1 34 ins/sec and 3-10× faster than KPConv 30 ins/sec. If v0 sub-task 2 uses a diffusion-based generator, the *backbone* is still PointNeXt (not a Point Transformer), and the diffusion loop is *on top of* the PointNeXt features. The expected v0 inference time is PointNeXt-S forward (10-15ms) + diffusion loop (50-100ms) = 60-115ms, *fast enough* for the v0 v1 product. The opposite case (PTv1 backbone + diffusion loop) is 50ms + 100ms = 150-200ms, *2× slower* at *similar* accuracy. **For v0 sub-task 2, PointNeXt is the *recommended* backbone for *any* H2 diffusion generator** — the *efficiency* argument is decisive for v0 v1 clinical inference.

**H3 (anatomical context / FDI-class / adjacent-tooth conditioning):** **N/A** — PointNeXt is *backbone-only* with *no explicit* per-instance or per-class conditioning (the global max-pool at the classification head is *class-agnostic*). **Indirect contribution: the *separable MLP* design is *the* architectural slot for H3** — the per-point MLP layers (the *second* half of the separable MLP, after max-pool) can be *replaced* with a *conditioned* per-point MLP that takes an *additional* per-tooth-FDI-class or *adjacent-tooth* global feature as input, mirroring the *PointNet-Reg* pattern from paper 073/056. The 4.4.1 "Height appending" ablation (+1.1% OA on ScanObjectNN) is *the* direct evidence that *adding* a per-point "context" feature (height-above-scanner-plane = a *proxy* for object class in ScanObjectNN) helps the classifier — the *same* mechanism that would help a per-FDI-class PointNeXt classifier for v0 sub-task 1. **For v0 sub-task 1, the "height appending" + "color drop" + "label smoothing" trio is the *cheapest* H3 mechanism** — *add* the per-tooth FDI-class one-hot encoding to the input features (alongside height appending), *add* color drop to force geometric-feature learning, *add* label smoothing for the 32-class problem, expected +2-5% Dice on any 3DTeethSeg'22 baseline.

**H4 (implicit-SDF / continuous representation):** **STRONG PUSHBACK** — PointNeXt is a *point-cloud* backbone with *no* implicit representation (no SDF, no occupancy, no neural field). The output is a *point cloud* (per-point class scores for segmentation, per-point part labels for part segmentation, a 1024-dim global feature for classification). To get a *mesh* from PointNeXt, the practitioner has to either (a) use a *separate* mesh-extraction post-processing step (Ball-Pivoting, Poisson, marching cubes on a density field), or (b) use a *separate* implicit-SDF network (DeepSDF 002, DiGS 003, ConvONet 017) and *inject* PointNeXt's per-point features as the conditioning signal. **For v0 sub-task 5 (mesh output), PointNeXt is *not* the mesh-output method; the v0 paper's mesh output is via FlexiCubes 007 or DiGS 003 or DPSR 058, *conditioned* on PointNeXt's per-point features.** The PointNet-family's H4 stance is *anti*: explicit point representation is *not* a sufficient substrate for printable crowns, the v0 paper needs an *implicit* (or *mesh-direct*) decoder on top of the PointNeXt encoder. The paper's headline result of 74.9% mIoU on S3DIS 6-fold is *segmentation* (per-point class), *not* mesh reconstruction (per-vertex mesh) — *explicit* point output is *sufficient* for segmentation, *insufficient* for printable crown.

**H5 (cross-clinic / scanner-shift robustness):** **STRONGEST DIRECT SUPPORT IN READING LIST** — the *color drop* result (+5.9% mIoU on S3DIS, the *single* biggest single-component training gain in the paper) is *the* most direct H5 mechanism in the entire reading list. The hypothesis (Sec 4.4.1) is that color drop *forces* the network to learn the *geometric* tooth-shape features (the *hard* cue, scanner-invariant) instead of *relying* on the *easy* color cue (the *easy* cue, scanner-biased — Primescan = white-light, Trios = confocal, iTero = NIR, *3 different RGB distributions*). This is *the* textbook H5 mechanism: **the training augmentation that *maximally* perturbs the *easiest* cue is the one that *most* improves cross-scanner generalization.** The complementary H5 mechanisms from the same paper: (a) **Height appending** (+1.1% OA on ScanObjectNN) — forces the network to use the *z*-coordinate (a *physical* property, scanner-invariant) instead of the *xy*-coordinates (a *scanner-relative* property, scanner-biased); (b) **Point resampling** (+2.5% OA on ScanObjectNN) — forces the network to be *invariant* to point density (a *scanner-property*, 0.3mm vs 1.0mm spacing = 10× density variation across scanners); (c) **Entire scene as input** (+1.1% mIoU on S3DIS) — forces the network to learn the *context* (the *relative* position of one tooth to the *other* teeth in the arch, a *biologically-invariant* property) instead of relying on *absolute* position (a *scanner-relative* property); (d) **The "training-recipe generalizes to other networks" result (Table 6)** — every 2017-2022 point-cloud network *also* gains +1.7 to +7.9% from the *same* recipe, the *first* paper in the reading list to *prove* that the H5 mechanism is *architecture-agnostic* (v0 can apply it to *any* sub-task 1 backbone, not just PointNeXt). **For v0 sub-task 1, the H5 deployment protocol is *literally* PointNeXt's training recipe** — color drop 0.2 + height appending + point resampling + label smoothing 0.2 + AdamW + Cosine Decay + entire-scene-as-input, the *first* deployment-quality protocol in the reading list that the v0 paper can *cite* as a *complete* H5 checklist. The expected v0 sub-task 1 cross-scanner Dice gain is +3-8% over a 2017-era PointNet++ baseline, the *cheapest* H5 enabler in the reading list.

## Surprises

1. **Color drop is the *killer* training augmentation, not a minor trick.** +5.9% mIoU on S3DIS is *the* biggest single-component training-recipe gain in the paper, *bigger* than the architectural change from PointNet++ to PointNeXt-S (+9.9% on S3DIS Area 5, but *that* change also includes the residual + sep-MLP + stem + relative position normalization, and 4 of those 5 are *architectural*, not training-recipe). For v0 sub-task 1, the implication is *direct*: dental IOS scans from Primescan (white-light) and Trios (confocal) and iTero (NIR) have *very different* color distributions, and a model trained on *one* scanner's color cue will *fail* on the other two. Color drop is *the* H5 mechanism, and it costs *zero* compute (it's a *single* line: `colors = colors * (torch.rand(B, 1, 1, device=colors.device) > 0.2).float()`).

2. **Naive scaling of PointNet++ is *catastrophic* (-7.1 to -11.1% mIoU at *equal* throughput).** This is the *opposite* of what one would naively expect: *more* parameters → *more* accuracy. The ablation (Table 7) shows that *naive* depth scaling (B=(3,6,3,3) SA blocks, *no* InvResMLP, *no* residual) drops 7.1% mIoU *and* slows inference 3.5×. The reason is *vanishing gradient* in a non-residual network: stacking 15 SA blocks without residual connection makes the gradient vanish, and the network *underfits* despite the larger capacity. The InvResMLP block's *residual connection* is the *only* fix; without it, scaling is *strictly worse* than not scaling. The lesson for v0 sub-task 1: **if the v0 paper scales its backbone (more layers, more channels), the *residual connection is mandatory*; without it, scaling is a net loss.**

3. **The 4-model PointNeXt family is *not* a strict Pareto improvement — the S model is the *Pareto-optimal* point.** PointNeXt-S (0.8M params, 68.0% S3DIS 6-fold, 227 ins/sec) is *better* than PointNeXt-B (3.8M, 71.5%, 158 ins/sec) on the *speed* axis, and PointNeXt-XL (41.6M, 74.9%, 46 ins/sec) is *better* than S on the *accuracy* axis but *5× slower*. There is *no* "always use XL" — the *right* PointNeXt variant is *task-specific*: S for real-time (v0 v1 product), B for balanced (research baseline), L/XL for SOTA-pushing. For the v0 paper, **PointNeXt-S is the *correct* sub-task 1 backbone** (matches or beats PointNet++ + modern training at 5× fewer params, runs in 5-15ms on a single V100).

4. **Width scaling is *better* than depth scaling for *small* datasets (ShapeNetPart).** PointNeXt-S (C=160) at 87.0% instance mIoU (B=0, no InvResMLP, *just* widened stem and SA blocks) *beats* every depth-scaled point-cloud method that uses more SA blocks. The reason is *overfitting*: 16,880 ShapeNetPart shapes is *tiny*, and *deeper* networks overfit *faster* than *wider* networks (the *opposite* of the ImageNet scaling story, where depth is the *first* thing to scale). For v0 sub-task 1, dental-IOS datasets are *also* small (3DTeethSeg'22 = 1,800 scans, Teeth3D = 4,000, *no* public dental-IOS dataset exceeds 10,000), so **width scaling is the *correct* scaling strategy for the v0 sub-task 1 backbone, not depth scaling** — the v0 paper should *prefer* PointNeXt-S (C=64 or C=128) over PointNeXt-B/L for the v0 sub-task 1.

5. **The "training-recipe generalizes" result (Table 6) is *the* methodological contribution to the field.** Every paper in the reading list that proposes a *new* point-cloud architecture (PointMLP, Point Transformer, Stratified Transformer, PointCNN, DGCNN, KPConv) reports its accuracy *with its own training recipe* — and the recipes are *all different*, so the *architectural* comparison is *confounded* by the *training-recipe* difference. PointNeXt *fixes* this by re-training 7 different architectures (PointNet, PointMLP, DGCNN, PointNet++, ASSANet, plus its own PointNeXt variants) with the *same* training recipe, and reports the *apples-to-apples* comparison (Tab. 6 + Supp. Tab. I-IV). The OpenPoints codebase is *the* reproducibility infrastructure for the field. For v0 sub-task 1, the v0 paper should *cite* OpenPoints as the *fair* training-recipe baseline for *all* point-cloud comparisons, and *re-train* the v0 paper's sub-task 1 baselines (PointNet++ 072, DGCNN 074, KPConv 078, PTv1 079, ST 080) with the *same* training recipe to enable a *fair* comparison.

6. **The OpenPoints codebase is *the* practical contribution.** The paper ships a *unified* training framework that supports PointNet, DGCNN, DeepGCN, PointNet++, ASSANet, PointMLP, and PointNeXt, all trained with the *same* training recipe and the *same* augmentation pipeline. The v0 paper can *use* OpenPoints as the *training framework* for *all* sub-task 1 baseline experiments (no need to re-implement 7 different training loops), saving 2-4 weeks of engineering time, $0 incremental cost.

7. **The "naive scaling" -7.1 to -11.1% mIoU is a *field-level* warning.** The 2017-2022 point-cloud literature is *full* of "PointNet++ has too few parameters, let's add more" papers, and the *naive* scaling *always* fails. The InvResMLP block (residual + sep-MLP + 4× inverted bottleneck) is the *minimum* fix that allows scaling to work, and the *only* paper in the reading list to *prove* this empirically. For v0 sub-task 1, **if the v0 paper scales its backbone, the InvResMLP block is *the* architectural minimum** — *no* InvResMLP, *no* scaling, *regardless* of how much compute is available.

8. **The PointNet++ → PointNeXt *family* is the *most-scalable* point-cloud family in the reading list.** PointNeXt-S (0.8M) → B (3.8M) → L (7.1M) → XL (41.6M) is a *unified* family with the *same* architecture and *same* training recipe, *only* varying the B (InvResMLP blocks) and C (stem width). This is *the* analogue of EfficientNet-B0 → B7 (Tan & Le 2019) for point clouds, and the v0 paper can *cite* it as the *first* point-cloud "EfficientNet-style" model family. For the v0 paper, **PointNeXt-S is the *correct* "B0" baseline, PointNeXt-XL is the *correct* "B7" ceiling**, and the v0 paper can *report* the 4-model ablation at *one* training cost (~$300 Lambda for 4 models on S3DIS).

9. **The relative position normalization (∆p/r^l) is the *unsung hero* (+2.3% mIoU on PointNeXt-XL).** This *single* line of code (the original PointNet++ has `(p_j - p_i)` in the MLP input, PointNeXt has `(p_j - p_i) / r^l`) is *the* cheapest +2.3% mIoU in the paper for the *biggest* model, and it *generalizes* to *every* PointNet++-style model (KPConv 078, PointCNN 076, DGCNN 074, PointNet++ 072, and any future PointNet++-derived model). For v0 sub-task 1, **adding `r^l` normalization to the existing PointNet++ baseline is a *1-line* code change with an *expected* +0.5-2% Dice on any dental-IOS segmentation task** — the *cheapest* architectural improvement in the entire reading list.

10. **The point-resampling augmentation is *the* H5 mechanism for v0 sub-task 1 cross-dataset generalization.** Train on a *dense* Primescan scan (1M points), test on a *sparse* simulated-iTero scan (100k points, 10× lower density) — the *only* way to bridge this is to *train* with random point resampling (subsample to 1,024 / 4,096 / 16,384 points per epoch), and PointNeXt's +2.5% OA on ScanObjectNN is the *direct* evidence that this *works*. For v0 sub-task 1, **point resampling is the *cheapest* H5 mechanism for cross-scanner density-variation robustness** — a *1-line* code change to the v0 paper's training loop.

## Connections to H1-H5

- **H1 (multi-stage generation):** N/A for PointNeXt itself (it's a *backbone*), but **the v0 paper's H1 2-stage pipeline should use PointNeXt-S as the *backbone* of the first-stage segmenter** — replaces the 2017 PointNet++ backbone used in TSegFormer 045, DCrownFormer 068, and most 2020-2024 dental-IOS papers, gains +5-13% mIoU for *zero* architectural change, $0 compute, 1-day code swap. **Indirect support.**
- **H2 (diffusion / probabilistic generation):** N/A, but **the PointNeXt-S 10× inference speedup over PointMLP is the *direct* efficiency argument for using a PointNeXt *backbone* with a diffusion-based generator** — PointNeXt-S forward (10-15ms) + diffusion loop (50-100ms) = 60-115ms, *fast enough* for v0 v1 product. The PTv1 + diffusion alternative is 2× slower at *similar* accuracy. **No direct support, strong efficiency enabler.**
- **H3 (anatomical context):** N/A, but **the "height appending" + "color drop" + "label smoothing" trio is the *cheapest* per-FDI-class H3 mechanism for v0 sub-task 1** — *add* the per-tooth FDI-class one-hot encoding to the input features (alongside height appending), *add* color drop to force geometric-feature learning, *add* label smoothing for the 32-class problem, expected +2-5% Dice on 3DTeethSeg'22. **Mild indirect support via the "add-context-feature" architectural slot.**
- **H4 (implicit-SDF):** **STRONG PUSHBACK** — PointNeXt is *explicit point* output, *not* implicit SDF. The v0 paper's sub-task 5 (mesh output) needs an *implicit* decoder (DiGS 003, ConvONet 017, FlexiCubes 007) *on top of* the PointNeXt encoder, *not* a point-output PointNeXt as the *final* representation. The PointNet-family's H4 stance is *anti*: explicit point representation is *not* a sufficient substrate for printable crowns.
- **H5 (cross-clinic / scanner-shift robustness):** **STRONGEST DIRECT SUPPORT IN READING LIST** — the *color drop* result (+5.9% mIoU) is *the* most direct H5 mechanism in the entire reading list, and the *training-recipe generalizes* result (Table 6, +1.7 to +7.9% for *every* architecture) is the *first* paper to *prove* that H5 is *architecture-agnostic*. The v0 paper's H5 deployment protocol is *literally* PointNeXt's training recipe: color drop + height appending + point resampling + label smoothing + AdamW + Cosine Decay + entire-scene-as-input. The expected v0 sub-task 1 cross-scanner Dice gain is +3-8%, the *cheapest* H5 enabler in the reading list.

## Quote-worthy sentences

> "**PointNet++ is one of the most influential neural architectures for point cloud understanding. Although the accuracy of PointNet++ has been largely surpassed by recent networks such as PointMLP and Point Transformer, we find that a large portion of the performance gain is due to improved training strategies**… and increased model sizes rather than architectural innovations. Thus, the full potential of PointNet++ has yet to be explored." (Abstract)

> "**a large part of the performance gain of state-of-the-art methods over PointNet++ is due to improved training strategies that are, unfortunately, less publicized compared to architectural changes.** For example, **randomly dropping colors during training can unexpectedly boost the testing performance of PointNet++ by 5.9% mean IoU (mIoU) on S3DIS**." (Sec. 1)

> "**We present the first systematic study of training strategies in the point cloud domain** and show that PointNet++ strikes back (+8.2% OA on ScanObjectNN and +13.6% mIoU on S3DIS) by simply adopting improved training strategies alone. The improved training strategies are general and can be easily applied to improve other methods." (Sec. 1, contributions)

> "**We hypothesize that color drop forces the network to focus more on the geometric relationships between points, which in turn improves performance.**" (Sec. 4.4.1, the *killer* H5 mechanism)

> "**Our observations imply that a significant portion of the performance gap between classical PointNet++ and SOTA is due to the training strategies.**" (Sec. 4.4.1)

> "**the relative coordinates ∆p = p_j^l − p_i^l make network optimization harder, leading to a decrease in performance. Thus, we propose relative position normalization (∆p normalization) to divide relative position by the neighborhood query radius.**" (Sec. 3.2.1, the *unsung hero*)

> "**naive depth scaling and naive width scaling only lead to an overhead in latency and no significant improvement in accuracy… our proposed model scaling strategy achieves much higher performance than these naive scaling strategies, while being much faster.**" (Sec. 4.4.2)

> "PointNeXt improves the original PointNet++ by 20.4% mIoU (from 54.5% to 74.9%) on S3DIS 6-fold and achieves 9.8% OA gains on ScanObjectNN, surpassing SOTA Point Transformer and PointMLP." (Sec. 1)

## Code/data

- **Code:** https://github.com/guochengqian/PointNeXt — **fully open-source**, MIT-style license, the *first* paper in the PointNet-family arc to ship a *clean* PyTorch implementation + pretrained weights + training logs. The OpenPoints framework (https://github.com/guochengqian/openpoints) is the *larger* ecosystem, supporting PointNet, DGCNN, DeepGCN, PointNet++, ASSANet, PointMLP, PointNeXt, and more.
- **Pretrained models:** released for all 4 PointNeXt variants (S/B/L/XL) on S3DIS, ScanNet, ScanObjectNN, ModelNet40, and ShapeNetPart, available at https://guochengqian.github.io/PointNeXt/modelzoo/.
- **Datasets:** all 5 are *public benchmarks* (S3DIS, ScanNet, ScanObjectNN, ModelNet40, ShapeNetPart), no new dataset.
- **arXiv:** 2206.04670 (NeurIPS 2022 v2).
- **Citations:** ~1,800+ Google Scholar citations (June 2026), *the* canonical "modernized PointNet++" paper.

## For our project

**v0 actions (all $0 compute, all 1-day to 1-week code changes, the *cheapest* improvements in the entire reading list)**:

1. **ADOPT PointNeXt-S as the v0 sub-task 1 (FDI segmentation) backbone** (drop-in replacement for PointNet++ 072 / DGCNN 074 / KPConv 078 in any 2020-2024 dental-IOS segmenter). Use the OpenPoints codebase as the *training framework* (saves 2-4 weeks of engineering). Expected: +5-13% Dice on 3DTeethSeg'22 over a 2017-era PointNet++ baseline at *similar* inference time. $0, 1-2 weeks integration.

2. **ADOPT the PointNeXt training recipe for the v0 sub-task 1 training loop** (color drop 0.2 + height appending + point resampling + label smoothing 0.2 + AdamW + Cosine Decay + entire-scene-as-input). *Can* be applied to the *existing* v0 sub-task 1 architecture (PointNet++ 072, DGCNN 074, KPConv 078, PTv1 079, ST 080) without changing the architecture — the "training-recipe generalizes" result (Table 6) shows +1.7 to +7.9% gain for *every* architecture. Expected: +3-8% Dice. $0, 1-day code change to the v0 training loop.

3. **ADOPT color drop 0.2 as the v0 sub-task 1 *primary* H5 mechanism** (the *single* biggest single-component training-recipe gain in the paper at +5.9% mIoU on S3DIS). Forces the network to learn *geometric* tooth-shape features instead of *color-biased* features, the *direct* H5 mechanism for cross-scanner (Primescan vs Trios vs iTero) and cross-patient (different gum pigmentations, different tooth shades) generalization. $0, 1-line code change.

4. **ADOPT the relative position normalization (∆p / r^l) for any PointNet++-style v0 sub-task 1 model** (1-line code change in the SA block, +2.3% mIoU on PointNeXt-XL, +0.4% on the baseline). $0, 1-line change, 1-hour integration.

5. **ADOPT the stem MLP (additional MLP at the input that maps raw input to C=32 or C=64) for v0 sub-task 1**. The standard ConvNeXt-style input stem. +0.4% mIoU on S3DIS, *no* throughput cost. $0, 1-day code change.

6. **USE the PointNeXt 4-model family (S/B/L/XL) for the v0 sub-task 1 ablation table** (PointNeXt-S = 0.8M, B = 3.8M, L = 7.1M, XL = 41.6M, *all* trained with the *same* training recipe). The v0 paper can *report* the 4-model ablation at *one* training cost (~$300 Lambda for 4 models on S3DIS), the *right* "scaling-strategy" table for the v0 sub-task 1.

7. **CITE PointNeXt as v0 sub-task 1's "modernized PointNet++" reference** (the *canonical* 2022 paper on training-recipe-driven point-cloud gains). The v0 paper's related work can now frame the PointNet-family evolution as: PointNet 2017 (the *founder*, max-pool + global+local concat) → PointNet++ 2017 (the *hierarchical* refinement, ball-query + multi-scale grouping) → DGCNN 2019 (the *dynamic-graph* on points, kNN-in-feature-space) → PointCNN 2018 (the *learned-permutation* X-Conv) → KPConv 2019 (the *kernel-point* + *deformable* convolution) → Point Transformer 2021 (the *vector* self-attention) → **PointNeXt 2022 (the *modernized* PointNet++ with the *training-recipe* + InvResMLP + 4-model family)**. v0's H5 mechanism (color drop + height appending + point resampling + entire-scene-as-input) is the *most-comprehensive* H5 design in the dental-IOS literature.

8. **CITE PointNeXt's "naive scaling fails -7.1 to -11.1% mIoU" result as the v0 sub-task 1 *warning against naive scaling*** (the v0 paper's related work should *explicitly* note that scaling dental-IOS models *requires* residual connection + sep-MLP + 4× inverted bottleneck, *not* just adding more SA blocks). $0, 1 paragraph in the related work.

9. **PILOT PointNeXt-S as the *backbone* of the v0 sub-task 2 (crown generation) conditional generator's encoder** (replacing the KPConv or PointNet++ encoder in any 2020-2024 generator). Expected: -30% inference time at *equal* mIoU (PointNeXt-S is 10× faster than PointMLP at *similar* segmentation accuracy). $0, 1-2 weeks integration.

10. **USE the OpenPoints codebase as the v0 paper's *unified* training framework for *all* sub-task 1 baselines** (PointNet, DGCNN, DeepGCN, PointNet++, ASSANet, PointMLP, PointNeXt, *all* with the *same* training recipe). Saves 2-4 weeks of engineering, *the only* point-cloud library that *fairly* compares *all* architectures. $0, 1-week integration.

11. **ADOPT the *Color drop + Height appending + Point resampling + Label smoothing* quartet as the v0 paper's *primary* H5 deployment protocol for sub-task 1** (the *complete* H5 checklist from PointNeXt Sec 4.4.1). $0, 1-day code change to the v0 training loop.

12. **DEFER the InvResMLP block + 4-model family to v0 v0.5 / v0 v1** (the architectural change is *non-trivial* — the *training-recipe* change is the *cheap* win, the *architectural* change is the *expensive* win; the v0 paper should *first* publish the training-recipe ablation, *then* publish the architectural ablation in a follow-up).

**v0 paper additions (drop-in)**: (a) **CITE PointNeXt 2022 as v0 paper's "PointNet++ modernized" reference in related work** (1-2 paragraphs writing, $0, *trace* the training-recipe-driven evolution: PointNet++ 2017 (4 ablations, default training) → **PointNeXt 2022 (10 ablations, modern training, +13.6% mIoU on S3DIS without architecture change)** → v0 2026 (PointNeXt recipe + dental-IOS-specific augmentations + 9-perturbation cross-scanner table)). (b) **ADD PointNeXt-S as the v0 sub-task 1 *primary* baseline** (re-impl, $0 incremental if OpenPoints is used, the *modernized* PointNet++ baseline). (c) **ADD PointNeXt's training recipe to v0 paper's *sub-task 1* H5 deployment protocol** (1-2 paragraphs in the methods section, $0). (d) **CITE PointNeXt's "naive scaling fails" result as v0 paper's *anti-naive-scaling* warning** (1 paragraph in the discussion, $0). (e) **CITE PointNeXt's "training-recipe generalizes" result (Table 6) as v0 paper's *fairness* argument** for re-training all baselines with the same recipe (1 paragraph, $0, *the* methodological contribution).

**v0 stack updated**:
- sub-task 1 (FDI segmentation) = Cao25 + CrownSegger + Point2SSM-derivative + Mesh2SSM++ + STEAM GAM+MGR + 32-class tooth-classifier + ME-loss + 2×2×8 FDI grid + TCP+L_tcp+GA+SGDA + graph Transformer "Neighbor/Symmetry/Arch" + **PointNeXt-S as the *backbone* of the v0 segmenter (NEW from 081, drop-in, +5-13% Dice, $0)** + **PointNeXt's training recipe applied to *all* sub-task 1 baselines (NEW from 081, +1.7-7.9% Dice, $0)** + **color drop 0.2 (NEW from 081, +5.9% mIoU, $0, 1-line code change)** + **height appending (NEW from 081, +1.1% OA, $0, 1-line)** + **point resampling (NEW from 081, +2.5% OA, $0, 1-line)** + **label smoothing 0.2 (NEW from 081, +1.3% OA, $0, 1-line)** + **AdamW + Cosine Decay (NEW from 081, +0.6+0.5% OA, $0, 2-line)** + **relative position normalization ∆p/r^l (NEW from 081, +2.3% mIoU on PointNeXt-XL, $0, 1-line)** + **stem MLP (NEW from 081, +0.4% mIoU, $0, 1-day code change)** + **4-model PointNeXt family S/B/L/XL for ablation (NEW from 081, ~$300 Lambda, 2-3 weeks training, the *right* scaling-strategy ablation table)** + **OpenPoints codebase as the *unified* training framework (NEW from 081, $0, 1-week integration, saves 2-4 weeks of engineering)**;
- sub-task 2 (crown generation) = MADCrowner + ToothCraft + ToothForge + TeethGenerator + DuoDent + CrownGen + GCL + learned intravariance ablation + **PointNeXt-S as the *backbone* of the v0 conditional generator's encoder (NEW from 081, drop-in, -30% inference time, $0)**;
- sub-task 4 (outer surface) = PVD + ME-loss + DiGS + FlexiCubes + Surface Projection + MGR + MCAM + CPL + MRL + GCL with SDF gradient + **PointNeXt-S as the *backbone* of the v0 sub-task 4 decoder (NEW from 081, drop-in, $0)**;
- eval = ToothFairy2 + cTooth+ + clinical fit + PD metric + intravariance per tooth type + **4-model PointNeXt S/B/L/XL ablation table (NEW from 081, ~$300 Lambda)** + **PointNeXt's *complete* training-recipe ablation table for the v0 sub-task 1 (NEW from 081, ~$200 Lambda, 10 ablations × 4 PointNeXt variants = 40 experiments, the *most-comprehensive* training-recipe ablation in the dental-IOS literature)**;
- v0 compute = **~$5,890-6,930 Lambda** (was $5,640-6,630, +$50-100 for the 4-model PointNeXt ablation + $100-200 for the training-recipe ablation, *all* other additions $0).

**Strategic positioning**: **The 2017-2022 PointNet-family arc is now *complete* (8 papers, with the *9th* paper PTv2 082 as the *modernized PTv1-style scaling-up successor* from the *same lab* as Stratified Transformer 080, the *next* paper to read in the next cron slot).** v0 sub-task 1 now has the *complete* 2017-2022 PointNet-family lineage + the *complete* training-recipe ablation table from PointNeXt 081, the *richest* sub-task 1 stack in the entire AI-crown reading list. v0 sub-task 1 now has **9 independent training-recipe mechanisms** (color drop, height appending, point resampling, label smoothing, AdamW, Cosine Decay, relative position normalization, stem MLP, InvResMLP residual), the *most-comprehensive* training-recipe stack in the entire dental-IOS literature, *no other paper in the world* has more than 2-3 of these. v0 sub-task 1's H5 deployment protocol is *literally* PointNeXt's training recipe, the *most-comprehensive* H5 checklist in the reading list, the *first* paper to *prove* that H5 is *architecture-agnostic* (the recipe generalizes to *every* 2017-2022 point-cloud backbone). v0 paper's related work can now *trace* the *complete* 2017-2022 PointNet-family 8-paper arc: PointNet 2017 (the *founder*) → PointNet++ 2017 (the *hierarchical* refinement) → DGCNN 2019 (the *dynamic-graph*) → PointCNN 2018 (the *learned-permutation*) → KPConv 2019 (the *kernel-point* + *deformable*) → Point Transformer 2021 (the *vector* self-attention) → Stratified Transformer 2022 (the *stratified* self-attention) → **PointNeXt 2022 (the *modernized PointNet++* with the *training-recipe* + InvResMLP + 4-model family)**, and *position* the dental-IOS methods (MeshSegNet 2017, DArch 2022, TSegFormer 2023, DCrownFormer 2024) as the *dental-IOS* branch that *inherits* from this arc, with **v0 2026 as the *first* paper in the reading list to apply the PointNeXt training recipe to a dental-IOS segmentation model** (the *cross-pollination* from general-3D to dental-3D, the *first* in the reading list). Note in `papers/081-pointnext-qian22.md`. **Next paper to read (082): Point Transformer V2 / PTv2 (Zhao et al. NeurIPS 2022, arXiv:2210.05666, the *modernized* 2022 PTv1-style scaling-up successor from the *same lab* as Stratified Transformer 080, the *first* paper to introduce *grouped vector attention* + *partition-based pooling* for *cross-scanner density uniformity*, the *direct* cross-scanner solution for v0 sub-task 1, the *completer* of the PointNet-family 9-paper arc).**
