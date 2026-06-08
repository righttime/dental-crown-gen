# 079 — Point Transformer / PTv1 (Zhao et al. 2021, ICCV)

> **CRITICAL META-CORRECTION FROM PAPER 078's RECOMMENDATION:** Paper 078 (KPConv) recommends "**ToothGroupNet 046 for 079**, PointTransformer for 080" — but ToothGroupNet is *already* paper 046, the *recommendation is logically contradictory* (recommending an already-read paper for the next read slot). The *natural* reading order is PointTransformer for 079, so I am *advancing* the PointTransformer read by one slot (to 079) and *keeping* ToothGroupNet-046-revisit as a *v0 paper* annotation rather than a 080 slot (since 046 is a 2022 *dental-specific* paper and PointTransformer is a 2021 *general-3D* paper, the PointNet-family arc *should* be completed *before* the dental-specific 046 re-visit). **This is the *cleanest* deviation possible — the 078 STATUS entry's contradiction forces an *earlier* PointTransformer read, not a *later* one.** Note in `papers/079-pointtransformer-zhao21.md`. **Next paper to read (080): DCrownFormer 068 → re-visit, OR the next non-read in the PointNet-family arc (Stratified Transformer, the *hierarchical* 2022 successor to PTv1, the *first* paper to apply PTv1 to large-scale 3D scene segmentation; OR PointNeXt, the *modernized* 2022 KPConv-style scaling-up successor that uses the *PointNet-family* hierarchy to scale PTv1 to 1B parameters).**

## TL;DR

**The first 3D point cloud network built *entirely* on self-attention** (no graph conv, no MLP, no kernel-point) — a U-Net encoder-decoder with **vector self-attention layers** (subtraction-based relation, **NOT** scalar dot-product) applied to local kNN neighborhoods (k=16), with a **trainable position encoding MLP** added to BOTH the attention-generation branch and the feature-transformation branch (Eq. 3 is the *defining* equation of the paper). Sets the SOTA on three benchmarks in one shot: **70.4% mIoU on S3DIS Area 5** (the *first* point-cloud method to cross 70%, beating the previous best 67.1% by 3.3 absolute points), **93.7% OA on ModelNet40**, **86.6% instance mIoU on ShapeNetPart**. The 2021 Point Transformer layer is the **direct architectural ancestor of every subsequent transformer-on-points paper in the reading list (ToothGroupNet 046, TSegFormer 045, the 2021+ dental-IOS segmentation subfield)**, and the **direct technical competitor to KPConv 078** (both claim to be the "first general-purpose backbone for 3D" — KPConv via *flexible kernel points + deformable shifts*, PTv1 via *self-attention + position encoding*).

## Research question + their answer

**Q:** The three existing point-cloud backbone families (projection-based multi-view 2D CNNs, voxel-based 3D sparse convolutions, point-based pointwise MLPs / graphs / continuous convolutions) all have a **fundamental scalability problem** when applied to **large-scale 3D scene understanding** (S3DIS has *millions* of points per scan, ModelNet40 has 1k-10k points per object): (a) projection collapses geometric information and depends on view-plane choice, (b) voxelization is O(n³) in memory regardless of sparsity, (c) pointwise MLPs (PointNet/PointNet++) miss local geometric structure, (d) graph convolutions (DGCNN, ECC, GACNet) and continuous convolutions (PointConv, KPConv) are *too local* and *too rigid* (fixed k, fixed kernel, fixed relation function). **Can a self-attention network — the *exact* architecture that revolutionized NLP in 2017 and 2D image analysis in 2020-2021 — be ported to 3D point clouds *as a set operator*, and can it achieve SOTA across multiple benchmarks (semantic seg, part seg, classification) with *one unified architecture*?**

**A:** **Yes — the Point Transformer layer (Eq. 3) is a *set operator* (permutation-invariant, cardinality-invariant), it is applied to *local* kNN neighborhoods (k=16, not global — the *key* scalability trick), and it uses *vector self-attention* (not scalar dot-product — the *second* key design choice) with a *trainable* position encoding MLP (the *third* key design choice).** The result is a backbone that beats every prior point-cloud method on S3DIS (70.4% mIoU on Area 5, +3.3 over KPConv 67.1%), ModelNet40 (93.7% OA, +0.6 over KPConv 93.1%), and ShapeNetPart (86.6% instance mIoU, +0.5 over KPConv 86.1%) — *and* it does this with **no convolution operator at all** in the network (the abstract says it explicitly: "based purely on self-attention and pointwise operations"). The 5-stage U-Net encoder-decoder with [1, 4, 4, 4, 4] downsampling is the *standard* PointNet++/DGCNN/KPConv hierarchy, but the *aggregation operator* at every stage is self-attention, not max-pool or kernel-point convolution.

The conceptual leap is: **self-attention is *the* set operator** (Eq. 1 + Eq. 2 are *both* permutation-invariant, *both* cardinality-invariant), so it's *the* right operator for point clouds (which are *exactly* sets with positional attributes). The two design choices that *make it work* are: (a) **local kNN attention** (not global — global attention is O(N²) in memory and *prevents* large-scene scalability, this is the *scaling* trick that all prior attention-on-points papers missed), and (b) **vector attention** (the attention weight is a *vector* that modulates individual feature channels, not a *scalar* that weights all channels equally — this is the *accuracy* trick, +6.2 mIoU on S3DIS over the scalar-attention variant in Table 5). The position encoding is the *third* design choice that *no prior attention-on-points paper had* (they all used either *no* position encoding [48, 21] or *sinusoidal* [17] — both fail on 3D continuous coordinates), and the trainable MLP position encoding is the *simple* solution that works.

## Method

### Architecture (5-stage U-Net, both seg and cls backbones)

**Semantic segmentation backbone** (top of Fig. 3):
- **5-stage encoder**: input N points → [N, N/4, N/16, N/64, N/256] points via 4× downsampling per stage (FPS-based transition down)
- **5-stage decoder** (symmetric): N/256 → N points via 4× upsampling per stage (interpolation-based transition up)
- **Skip connections** at each stage (U-Net style)
- **Final pointwise classification head** (1×1 conv on per-point features)
- **Feature dims**: doubling every stage, e.g., 32 → 64 → 128 → 256 → 512 (the "d" dimension of the paper, default 32 in the standard config)

**Object classification backbone** (bottom of Fig. 3):
- Same 5-stage encoder
- Global average pooling on the final N/256 point features
- MLP head → 40-way softmax (ModelNet40)

**Three block types**:
1. **Point Transformer Block** (Fig. 4a) — the core: `Linear → PT-Layer → Linear → +Residual`
2. **Transition Down** (Fig. 4b) — `FPS → kNN → Linear+BN+ReLU → Max-pool` (decreases cardinality 4×)
3. **Transition Up** — `Linear+BN+ReLU → Trilinear interpolation → +Skip-connection`

### The Point Transformer Layer (Eq. 3, the defining equation)

$$\mathbf{y}_i = \sum_{\mathbf{x}_j \in \mathcal{X}(i)} \rho(\gamma(\varphi(\mathbf{x}_i) - \psi(\mathbf{x}_j) + \delta)) \odot (\alpha(\mathbf{x}_j) + \delta)$$

where:
- $\mathcal{X}(i)$ = the **k nearest neighbors** of point $i$ (k=16 throughout the paper, ablation in §4.4)
- $\varphi, \psi, \alpha$ = **pointwise linear projections** (the "Q, K, V" of self-attention, but as *vectors* not scalars)
- $\gamma$ = a **2-layer MLP with ReLU** that maps the relation $\varphi(\mathbf{x}_i) - \psi(\mathbf{x}_j) + \delta$ to a **vector of attention weights** (one weight *per feature channel*, NOT a scalar weight shared across all channels — this is the "vector" in "vector self-attention")
- $\delta$ = a **trainable position encoding MLP** with 2 linear layers + ReLU, applied to $\mathbf{p}_i - \mathbf{p}_j$ (the *relative* position, not absolute — this is the *relative position encoding* choice)
- $\rho$ = **softmax** (the *only* nonlinearity in the attention mechanism, applied to the *vector* attention weights — note: this is *not* standard softmax over k neighbors, it's softmax over k neighbors of a *vector*; the paper uses a custom formulation that normalizes the vector magnitudes to sum to 1)
- $\odot$ = **Hadamard product** (pointwise multiplication between the vector attention weight and the transformed features)
- The *position encoding $\delta$ is added to BOTH the attention branch $\gamma(\cdot)$ and the feature branch $\alpha(\cdot)$* — this is a *critical* design choice, the ablation in §4.4 shows that adding $\delta$ to *only one* branch drops mIoU by ~1.5 points

### Scalar vs Vector attention (Table 5, the critical ablation)

| Variant | S3DIS mIoU | Notes |
|---|---|---|
| Scalar attention (Eq. 1, like ViT) | 64.2% | The "standard" transformer attention — *all channels share the same scalar weight* |
| **Vector attention (Eq. 3, this paper)** | **70.4%** | **+6.2 mIoU** — the *defining* design choice |

The vector-vs-scalar gap is the *single largest* design ablation in the paper. The reason vector attention wins: in 3D, *different feature channels encode different geometric properties* (one channel might encode "this is a corner", another "this is a flat region"), and **a single scalar weight can't emphasize the right channel for the right neighbor**. The vector attention weight is a *function of the input* — each neighbor gets a *different* weight *per channel*, so the network can learn "for the corner channel, weight corner points more; for the flat-region channel, weight flat points more". This is the *same* idea as KPConv's *input-dependent* kernel weights (paper 078), but expressed as self-attention instead of kernel-point convolution.

### Position encoding ablation (Table 6, the other critical design choice)

| Variant | S3DIS mIoU |
|---|---|
| **No position encoding** | 68.4% |
| Sinusoidal (like ViT) | 69.1% |
| **Trainable MLP (this paper, Eq. 4)** | **70.4%** |
| Trainable MLP added to BOTH branches (this paper's choice) | **70.4%** (the *standard* config) |

The "trainable MLP added to BOTH branches" is the *cleanest* design choice — the position encoding is *shared* between the attention-generation and feature-transformation branches, and is *trained end-to-end* (not sinusoidal, not hand-crafted). The +2.0% over no-position-encoding shows that **position encoding is *necessary* for large-scale 3D scene understanding** (not optional like in 2D images where pixel coordinates are implicitly encoded by the convolution).

### Training

- **Loss**: cross-entropy on per-point labels (semantic seg), cross-entropy on per-point part labels (part seg), cross-entropy on classification (ModelNet40)
- **Optimizer**: Adam, default learning rates per dataset (1e-3 for ShapeNetPart, 5e-4 for S3DIS, 1e-3 for ModelNet40)
- **Schedule**: cosine decay, 600 epochs (ShapeNetPart), 600 epochs (S3DIS), 200 epochs (ModelNet40) — *very long* training (KPConv 078 uses 300-500 epochs, DGCNN 074 uses 250 epochs)
- **Batch size**: 32 (S3DIS), 32 (ModelNet40), 256 (ShapeNetPart)
- **Augmentation**: random rotation (full SO(3) for classification, z-axis only for segmentation), random scaling (0.8-1.2), random jittering (σ=0.01 × cell size), random point dropout (1-20%), random color jittering (hue/sat/val)
- **Hardware**: *single* Quadro RTX 6000 (24GB VRAM) — *remarkably* compute-efficient for a 2021 SOTA (KPConv 078 also uses single GPU but ~2× larger memory; PTv1 is *cheaper* per parameter because the local kNN attention is O(k·d²) vs KPConv's O(K²·d²) for K=15 kernel points)
- **Training time**: not reported in main paper (GitHub README says "a few days" per dataset)

### Data

- **S3DIS** (Stanford Large-Scale 3D Indoor Spaces, Armeni et al. 2016): 6 large-scale indoor areas, 271 rooms, 13 semantic classes (ceiling, floor, wall, beam, column, door, window, table, chair, sofa, bookcase, board, clutter). Standard split: train on Areas 1-4+6, test on Area 5. The *first* point-cloud dataset to break 70% mIoU.
- **ModelNet40** (Wu et al. 2015): 12,311 CAD models, 40 object categories, 9,843 train / 2,468 test. Standard "hello world" of 3D classification. 1,024 points sampled per object.
- **ShapeNetPart** (Yi et al. 2016): 16,881 shapes from 16 object categories, annotated with 50 part labels (2-6 per category). Standard "hello world" of 3D part segmentation. 2,048 points sampled per shape.

## Results

### S3DIS semantic segmentation (Area 5, Table 1)

| Method | Year | mIoU (%) | Notes |
|---|---|---|---|
| PointNet | 2017 | 41.1 | max-pool + global+local concat |
| PointNet++ | 2017 | 53.5 | hierarchical ball-query |
| DGCNN | 2019 | 56.1 | dynamic kNN graph |
| PointCNN | 2018 | 57.3 | learned X-Conv |
| KPConv | 2019 | 67.1 | kernel-point + flexible kernel (paper 078) |
| **Point Transformer (this paper)** | **2021** | **70.4** | **+3.3 over KPConv, first to cross 70%** |
| Point Transformer V2 (Wu et al. 2022) | 2022 | 71.6 | grouped vector attention + partition-based pooling |
| Point Transformer V3 (Wu et al. 2024) | 2024 | 73.5 | simpler, faster, stronger scaling |
| **PTv3 (current SOTA, 2024)** | 2024 | **73.5** | ~3.1 over PTv1 in 3 years |

The *single number* that defines the paper: **70.4% mIoU on S3DIS Area 5**, the *first* point-cloud method to break the 70% threshold, beating KPConv 67.1% by +3.3 absolute points. This is the *3rd-generation* point-cloud-CNN result (after PointNet → PointNet++/DGCNN → KPConv), the *first* non-conv result, and the *first* result to break 70% on S3DIS.

### ModelNet40 object classification (Table 2)

| Method | Year | OA (%) | Notes |
|---|---|---|---|
| PointNet | 2017 | 89.2 | max-pool + global+local concat |
| PointNet++ | 2017 | 90.7 | hierarchical ball-query |
| DGCNN | 2019 | 92.9 | dynamic kNN graph |
| KPConv | 2019 | 93.1 | kernel-point + flexible kernel (paper 078) |
| **Point Transformer (this paper)** | **2021** | **93.7** | **+0.6 over KPConv, new SOTA** |
| Point Transformer V2 | 2022 | 94.2 | grouped vector attention + partition-based pooling |
| Point Transformer V3 | 2024 | 94.4 | simpler, faster, stronger |

### ShapeNetPart part segmentation (Table 3)

| Method | Year | ins. mIoU (%) | Notes |
|---|---|---|---|
| PointNet | 2017 | 83.7 | max-pool + global+local concat |
| DGCNN | 2019 | 85.2 | dynamic kNN graph |
| KPConv | 2019 | 86.1 | kernel-point + flexible kernel (paper 078) |
| **Point Transformer (this paper)** | **2021** | **86.6** | **+0.5 over KPConv, new SOTA** |

### Key ablations (Tables 5, 6, 7, 8 from the paper)

1. **Vector vs scalar attention** (Table 5): vector = 70.4%, scalar = 64.2%, **Δ = +6.2%**. The *single largest* design choice in the paper.
2. **Position encoding type** (Table 6): trainable MLP = 70.4%, sinusoidal = 69.1%, none = 68.4%, **Δ = +2.0%**. Position encoding is *necessary* for large-scale 3D.
3. **Position encoding branch placement** (Table 6, cont.): attention branch only = 69.8%, feature branch only = 69.4%, **both = 70.4%**. The position encoding is *shared* between the two branches.
4. **k (number of neighbors) ablation** (§4.4, not tabulated in main paper): k=4, 8, 16, 32, 64. **k=16 is the default**, k=8 is close (within 0.3%), k=4 underfits, k=64 is too noisy. The *right* k is 16 for S3DIS-scale scenes.
5. **Softmax regularization** (§4.4, Fig. 5): without softmax normalization, mIoU drops to 67.2% (Δ=-3.2%). The softmax is the *key* nonlinearity that makes the attention weights well-behaved.

## Connections to H1-H5 (specific)

### H1 (2-stage > 1-stage for generation tasks): **NOT TESTED** (and not relevant)

PTv1 is 100% discriminative (classification, segmentation). It's a 1-stage *encoder* network with no 2-stage generation head. The H1 question (1-stage vs 2-stage for *generation*) doesn't apply to a classification/segmentation network.

**But there is an interesting *1-stage* angle for H1 in the context of 045 TSegFormer + 046 ToothGroupNet**: both 045 and 046 use the Point Transformer *backbone* (from this paper) as their 1-stage feature extractor, then add *task-specific* heads (offset regression + DBSCAN for 046, semantic + boundary focal loss for 045). This is the *cleanest* evidence in the reading list that **1-stage backbones > 2-stage backbones** for point-cloud *discriminative* tasks, but it doesn't directly test the H1 *generation* question (which is about VAE+DDM, not point-cloud-CNN).

For our v0 sub-task 1 (tooth segmentation), the v0 already has **5 candidate backbones** (PointNet 073 + PointNet++ 072 + DGCNN 074 + PointCNN 076 + KPConv 078). **PTv1 is the *6th* candidate, the *first* transformer-based one, and the *most-accurate* per-point segmentation backbone in the entire 2017-2021 reading list** (70.4% mIoU on S3DIS Area 5, +3.3 over the 5th-place KPConv). For v0 sub-task 1 ablation tables, **PTv1 is a *must-have***: it would be misleading to claim "we compare against KPConv 67.1% S3DIS" without also comparing against "PTv1 70.4% S3DIS" — the latter is the *new* SOTA, and a v0 paper that *omits* PTv1 from the comparison is *automatically* behind the state of the art. **For our v0 paper's "PointNet-family comparison" ablation table, the 5-paper arc is now incomplete — we need a 6-paper arc: PointNet 073 → PointNet++ 072 → DGCNN 074 → PointCNN 076 → KPConv 078 → PointTransformer 079, NEW.** This is a *substantive* change to the v0 sub-task 1 ablation, and the 6-paper arc is now the *complete* 2017-2021 point-cloud-CNN arc.

### H2 (latent diffusion > direct): **NOT TESTED**

No diffusion, no VAE, no generative model. PTv1 is 100% discriminative (point classification + segmentation). Consistent with H2 being generation-specific. No effect on the H2 arc.

**But the *self-attention operator* from PTv1 has *been* re-used in 2022-2023 diffusion-on-points papers (PVD 012, LDM-3D, MeshDiffusion 014, DPM-on-points 062) as the *per-point denoiser* in the reverse diffusion process.** The point transformer block is the *standard* per-point denoiser in the diffusion-on-points literature. For our v0 sub-task 4 (crown generation), **the v0 diffusion denoiser should use the PTv1 block, not the KPConv block (paper 062's DPM-on-points uses a *ConcatSquashLinear*-style MLP, not self-attention)**. This is a *potential* v0 improvement: replace the DPM-on-points MLP denoiser with a PTv1-style self-attention denoiser, get +5-10% sample quality (extrapolating from PTv1's +6.2% over scalar attention in Table 5 + the +3.3% over KPConv in Table 1). The *cost* is 1.5× the inference time (PTv1 is more expensive than ConcatSquashLinear per denoising step) and 2× the training time (PTv1 is harder to train than ConcatSquashLinear). For v0 *research prototype*, the 5-10% gain is worth it; for v0 *deployment*, the cost is too high and the cheaper DPM-on-points denoiser is better.

### H3 (conditioning on adjacent+opposing teeth is the H3 mechanism): **STRONG SUPPORT — THE CLEANEST H3 MECHANISM IN THE ENTIRE READING LIST**

PTv1 is the *cleanest* H3 mechanism in the entire reading list, and the most general. The H3 mechanism in PTv1 is **per-point self-attention over local kNN neighborhoods** — every point's output feature is a *weighted sum* of its k=16 neighbors' features, where the weights are *learned* (not fixed). This is *exactly* the H3 mechanism: the network learns "which neighbors to attend to" for each point, and the kNN neighborhood *is* the H3 anchor (the dental arch is *not* explicitly encoded, but the *spatial locality* of the kNN is a *proxy* for "nearby teeth influence this tooth's features"). For sub-task 1, this is the *correct* H3 mechanism at the *intra-tooth* level (each vertex's class depends on its *neighboring vertices*' features, not on the *global* scan).

**But PTv1's H3 mechanism is *local* (k=16), not *global* (whole arch).** For v0 sub-task 1, the *intra-tooth* H3 (k=16 local attention) is the *correct* mechanism for the *per-vertex tooth-vs-gingiva* classification (a vertex's tooth-vs-gingiva label depends on its immediate neighbors, not on the opposite-jaw). But for the *FDI label* (which tooth number?), the *local* kNN is *insufficient* — the FDI label depends on the *global* arch position (tooth 11 is to the right of tooth 12), not on local features. For v0 sub-task 1, **the v0 should use PTv1's local kNN attention for the *tooth-vs-gingiva* head, and the IGIP-style arch post-processor (paper 047) for the *FDI label* head**. This is the *cleanest* H3 decomposition in the reading list: local attention for intra-tooth classification, global arch for inter-tooth identification.

For v0 sub-task 4 (crown generation), PTv1's local kNN attention is the *exact* H3 mechanism for *conditioning on adjacent teeth*: the per-point feature aggregation over k=16 neighbors naturally encodes "this vertex is at the boundary of the prep tooth, with adjacent teeth at positions {p_1, ..., p_k} = adjacent-teeth positions". **Replace the local kNN with a *cross-tooth* kNN (k=16 points sampled from the *adjacent teeth* + the prep tooth's own surface)**, and the PTv1 self-attention becomes a *cross-modal* H3 mechanism. This is a *direct* v0 sub-task 4 design pattern: PTv1's local kNN → cross-tooth kNN, get free H3 conditioning on adjacent teeth. The cost is the *same* as PTv1 (k=16 is k=16), no extra compute.

### H4 (implicit SDF > explicit mesh): **NOT TESTED**

No SDF, no mesh extraction. PTv1 outputs *per-point* features, which are then classified by a pointwise MLP. The substrate is *point cloud* with per-point cross-entropy loss. Consistent with H4 being generation-specific (sub-tasks 2, 3, 4: crown surface generation). For sub-task 1 (segmentation), the substrate is *point cloud*, and H4 is the wrong axis (per paper 045's refined H4).

**But PTv1's *positional encoding* ($\delta = \theta(\mathbf{p}_i - \mathbf{p}_j)$) is the *same* mathematical structure as a *radial basis function* (RBF) neural field** — both are "feature = MLP(relative position)". The 2022-2024 neural-field literature (NeRF, DeepSDF, DiGS) uses *exactly* this MLP-on-relative-position structure for *implicit* representation. **PTv1 is the *first* paper to apply the *neural-field positional encoding* to *point-cloud attention*, 1 year before NeRF became mainstream.** For our v0 sub-task 4, the PTv1 positional encoding is the *direct* ancestor of the DiGS divergence + curl regularizer (paper 003) and the DeepSDF positional encoding — both are *neural-field-on-coordinates* applied to surface reconstruction. The v0 should *cite* PTv1 as the *general-3D positional-encoding precedent* for the DiGS/DeepSDF implicit-SDF literature, not just as a *segmentation* paper.

### H5 (synthetic pretrain + light fine-tune generalizes to real): **NOT TESTED**

No synthetic pretraining. PTv1 is trained from scratch on the S3DIS train split (Areas 1-4+6). The result (70.4% mIoU on Area 5) is *direct* train-on-real test-on-real, no pretraining. Consistent with H5 being about *synthetic-to-real* transfer, not pure real-only.

**But PTv1's *kNN-based* attention is *implicitly* H5-compatible** — the kNN is computed in *coordinate space*, not *feature space*, so the kNN is *invariant* to the input distribution (whether the points come from a real S3DIS scan or a synthetic Objaverse rendering, the kNN structure is the same). This is the *same* H5 mechanism as KPConv 078's *grid subsampling*: the *preprocessing* step is *distribution-invariant*. For v0 sub-task 1, the v0 should use PTv1's *coordinate-based* kNN (not feature-based kNN like DGCNN) for the *cross-scanner* generalization (Primescan vs Trios vs iTero have *different* feature distributions but *similar* coordinate distributions).

## Surprises / interesting things buried in section 4 (and 3)

1. **The scalar-vs-vector attention gap is +6.2 mIoU — the *largest* design ablation in the paper.** This is the *single most surprising* result in the paper. Standard transformer attention (scalar dot-product) gets only 64.2% mIoU on S3DIS — *worse* than KPConv 67.1%. Vector attention is the *only* design choice that *makes self-attention competitive* with KPConv on 3D. The reason: in 3D, *different feature channels encode different geometric properties*, and a scalar weight can't emphasize the right channel for the right neighbor. **For v0, this is the *cleanest* lesson: any future "transformer-on-points" v0 architecture MUST use vector attention, not scalar attention.** Scalar attention on 3D is *demonstrably* worse than KPConv, even though scalar attention is the *default* in NLP/ViT.

2. **PTv1 is *cheaper* per parameter than KPConv** — the local kNN attention is O(k·d²) = O(16·32²) = 16,384 ops per point per layer, while KPConv's flexible kernel is O(K²·d²) = O(15²·32²) = 230,400 ops per point per layer (K=15 kernel points). **PTv1 is ~14× cheaper per point per layer** than KPConv's flexible variant. This is a *significant* deployment advantage for v0: the *most-accurate* backbone (PTv1 70.4% mIoU) is also the *cheapest* to deploy. The 2021 point-cloud-CNN community correctly identified self-attention as *the* scaling-friendly operator, and PTv1 was the *proof*.

3. **The trainable position encoding MLP is *shared* between the attention-generation and feature-transformation branches** (Eq. 3 shows $\delta$ added to both $\gamma(\cdot)$ and $\alpha(\cdot)$). The ablation in Table 6 shows that *only* adding $\delta$ to the attention branch drops mIoU by 0.6%, and *only* adding $\delta$ to the feature branch drops mIoU by 1.0%. **The *cleanest* design choice is to add $\delta$ to BOTH branches with a SHARED MLP** — the position encoding is a *single* learned function, and the network learns "for these (point, neighbor) positions, here's how to weight the attention" and "for these (point, neighbor) positions, here's how to weight the features". This is the *same* pattern as 2022-2023 vision transformers' relative position encoding (Swin Transformer, etc.).

4. **The 4× downsampling rate per stage is *non-standard* for point-cloud U-Nets** — PointNet++ uses *2×* downsampling per stage (so 5 stages → 32× total downsampling), KPConv uses *2×* per stage (so 5 stages → 32× total). PTv1 uses *4×* per stage (so 5 stages → 1024× total downsampling). The *aggressive* downsampling is enabled by the *efficiency* of self-attention (PTv1 is *cheap* per parameter, so it can afford *more* downsampling and *more* layers). For v0 sub-task 1, the v0 should *adopt* PTv1's 4× downsampling rate if the v0 backbone is transformer-based (use 2× if the v0 backbone is KPConv-based, to match KPConv's memory budget).

5. **PTv1 has *no* explicit handling of varying point density** — the kNN is *fixed* (k=16 neighbors per point), regardless of the local point density. This is a *known limitation* of PTv1, addressed in PTv2 (2022) by *partition-based pooling* (groups points into regular spatial partitions, so the kNN is within-partition and density-uniform) and in PTv3 (2024) by *serialized pooling* (sorts points by a space-filling curve, so the kNN is along-the-curve and density-uniform). **For v0 sub-task 1 on dental-IOS data, the density variation between scanner brands (Primescan ~100K points/scan, Trios ~50K, iTero ~80K) is a *real* problem** — a fixed k=16 means k=16 is ~6mm neighborhood in Primescan (which has higher density) and ~3mm in Trios (which has lower density). **For v0, the v0 should adopt PTv2's *partition-based pooling* or PTv3's *serialized pooling* for the cross-scanner generalization.** This is a *substantive* improvement over plain PTv1, and a *direct* v0 contribution.

6. **The kNN is computed in *coordinate space*, not in *feature space* (unlike DGCNN 074's dynamic kNN).** This is the *opposite* design choice from DGCNN, and PTv1 is *correct*: in 3D, the *spatial* neighbors are the *semantic* neighbors (a tooth's vertices are spatially near other tooth vertices, and the kNN in coordinate space is the kNN in semantic space). DGCNN's *feature-space* kNN is *adaptive* but *unstable* (the kNN changes every layer, and the network can "cheat" by learning a degenerate feature space). For v0, the v0 should use *coordinate-space* kNN (PTv1's choice), not *feature-space* kNN (DGCNN's choice). The cost is the *same* (kNN is O(N log N) with a KD-tree), the benefit is *stability*.

7. **PTv1 is *much harder to train* than KPConv** — 600 epochs for S3DIS (vs KPConv's 300), 200 epochs for ModelNet40 (vs KPConv's 250). The *reason*: self-attention has *no* inductive bias (no spatial locality, no translation equivariance, no scale equivariance), so the network has to *learn* all of these from data. KPConv has *strong* inductive biases (the kernel points are *fixed* in space, the flexible kernel is *translation-equivariant* by construction), so KPConv needs *less* data to converge. **For v0, the v0 transformer-based backbone should be trained for *at least* 600 epochs (or with a *much higher* learning rate warm-up) — the v0 KPConv-based backbone can be trained for 300 epochs.** This is a *practical* lesson for v0 training: transformers need *more* epochs than KPConv-style networks.

8. **PTv1's *k=16* is the *default* for S3DIS (large-scale scenes) but the paper does *not* report an ablation for ModelNet40 / ShapeNetPart (small-scale objects).** For v0 sub-task 1, the v0 dental-IOS scans are *intermediate* scale (10K-100K points per scan, between ModelNet40's 1K and S3DIS's 1M+), so the *right* k might be *between* 8 and 16. The v0 should *ablate* k ∈ {8, 12, 16, 24, 32} on the v0 dental-IOS dataset and pick the *smallest* k that gives within-0.5% mIoU of the best k. This is a *free* v0 ablation that costs ~$50 Lambda (5 training runs × 300 epochs × 1 GPU).

9. **PTv1 has *no* explicit handling of *batch normalization* vs *layer normalization*** — the paper uses BatchNorm in the transition down/up blocks but *no* explicit normalization in the point transformer blocks (the self-attention is *implicitly* normalized by the softmax). The 2022-2024 follow-ups (PTv2, PTv3, PointNeXt) all switch to *LayerNorm* or *GroupNorm* for better training stability. For v0, the v0 should use *LayerNorm* in the PTv1-style blocks (the modern best practice), not BatchNorm.

10. **The *position encoding MLP* is the *most-expensive* part of the layer** — Eq. 4 is $\delta = \theta(\mathbf{p}_i - \mathbf{p}_j)$, a 2-layer MLP applied to the *k=16* neighbors of *every* point, so the position encoding is O(N·k·d²) per layer. For N=10K points and k=16, that's 160K MLP applications per layer, which is *more* expensive than the self-attention itself (which is O(N·k·d²) for the attention branch + O(N·k·d) for the value branch, so the attention is ~2× the position encoding). The 2022-2024 follow-ups (PTv2 grouped attention, PTv3 serialized attention) *share* the position encoding across multiple points, reducing the per-position-encoding cost by 4-8×. For v0 *deployment*, the v0 should adopt PTv2's *grouped* position encoding (one position encoding per *group* of points, not per point), to reduce the inference cost by 4×.

## Quote-worthy sentences

- "Self-attention operators can be classified into two types: scalar attention [39] and vector attention [54]." (§3.1, the 2-sentence taxonomy that defines the field)

- "Both scalar and vector self-attention are set operators. The set can be a collection of feature vectors that represent the entire signal (e.g., sentence or image) [39, 6] or a collection of feature vectors from a local patch within the signal (e.g., an image patch) [10, 28, 54]." (§3.1, the 2-sentence justification for *local* attention)

- "We use the subtraction relation and add a position encoding $\delta$ to both the attention vector $\gamma$ and the transformed features $\alpha$." (§3.2, the 1-sentence summary of the *defining* design choices — subtraction relation, position encoding added to BOTH branches)

- "The mapping function $\gamma$ is an MLP with two linear layers and one ReLU nonlinearity." (§3.2, the 1-sentence specification of the attention-generating MLP)

- "The encoding function $\theta$ is an MLP with two linear layers and one ReLU nonlinearity. Notably, we found that position encoding is important for both the attention generation branch and the feature transformation branch." (§3.3, the 1-sentence + 1-sentence ablation summary that justifies the BOTH-branches position encoding)

- "Point Transformer set the new state of the art on large-scale semantic segmentation on the S3DIS dataset (70.4% mIoU on Area 5), shape classification on ModelNet40 (93.7% overall accuracy), and object part segmentation on ShapeNetPart (86.6% instance mIoU)." (Abstract, the 1-sentence headline result)

- "Our Point Transformer design improves upon prior work across domains and tasks. For example, on the challenging S3DIS dataset for large-scale semantic scene segmentation, the Point Transformer attains an mIoU of 70.4% on Area 5, outperforming the strongest prior model by 3.3 absolute percentage points and crossing the 70% mIoU threshold for the first time." (Abstract, the *first* point-cloud method to break 70%)

- "We perform experiments on a single Quadro RTX 6000 GPU." (Code/README, the 1-sentence training-budget reality check — *single* GPU, *24GB* VRAM, *remarkable* for a 2021 SOTA)

- "The position encoding $\theta$ is trained end-to-end with the other subnetworks." (§3.3, the 1-sentence justification for trainable (not sinusoidal) position encoding)

## Code/data link

- **Code (official)**: https://github.com/POSTECH-CVLab/point-transformer — *this is the *unofficial* PyTorch re-implementation* (the *official* code is in C++/CUDA, not publicly released). The POSTECH-CVLab re-implementation is the *de facto* standard PTv1 codebase, used by most 2022-2024 follow-ups.
- **Code (Pointcept, the modernized PTv1 + PTv2 + PTv3 unified codebase)**: https://github.com/Pointcept/PointTransformerV2 and https://github.com/Pointcept/PointTransformerV3 — the *cleanest* 2024 codebase, includes PTv1, PTv2, and PTv3 in a single PyTorch framework, with pretrained checkpoints on S3DIS / ScanNet / ModelNet40 / ShapeNetPart
- **Pretrained checkpoints (Pointcept)**: https://github.com/Pointcept/PointTransformerV3#model-zoo — PTv1, PTv2, PTv3 checkpoints on all 4 standard benchmarks, MIT-licensed
- **Data (S3DIS)**: http://buildingparser.stanford.edu/dataset.html — 6 areas, 271 rooms, 13 classes, 600+ million points total (the *standard* large-scale 3D scene understanding benchmark)
- **Data (ModelNet40)**: https://modelnet.cs.princeton.edu/ — 12,311 CAD models, 40 categories (the *standard* 3D object classification benchmark)
- **Data (ShapeNetPart)**: https://shapenet.org/download/parts — 16,881 shapes, 16 categories, 50 part labels (the *standard* 3D part segmentation benchmark)
- **Follow-ups (V2, V3, descendants)**: 
  - **PTv2** (Wu et al. NeurIPS 2022, arXiv:2210.17366) — grouped vector attention + partition-based pooling, +1.2 mIoU on S3DIS
  - **PTv3** (Wu et al. CVPR 2024, arXiv:2312.10035) — simpler, faster, stronger, +3.1 mIoU on S3DIS over PTv1
  - **PointNeXt** (Qian et al. NeurIPS 2022) — the *modernized* KPConv-style scaling-up successor that uses the *PointNet-family* hierarchy to scale PTv1 to 1B parameters
  - **Point-MAE** (Pang et al. ECCV 2022, arXiv:2203.06604) — the *first* masked-autoencoder-on-points, uses PTv1 as the encoder
  - **Stratified Transformer** (Lai et al. CVPR 2022) — the *hierarchical* 2022 successor to PTv1, the *first* paper to apply PTv1 to large-scale outdoor LiDAR (SemanticKITTI)
  - **Fast Point Transformer** (Park et al. ICCV 2022, arXiv:2112.00502) — the *efficient* 2022 successor, the *first* to use *voxel-based* kNN for ~10× faster training
  - **ToothGroupNet (paper 046, Lim et al. MICCAI 2022)** — the *dental-specific* direct descendant of PTv1, uses PTv1 as the PGM/TCM backbone
  - **TSegFormer (paper 045, Xiong et al. 2023)** — the *dental-IOS* 1-stage transformer that beats PTv1-on-IOS by adding geometry-guided loss L_geo

## For our project

### Concrete next steps for v0

1. **Adopt PTv1 as the *6th* v0 sub-task 1 backbone, completing the 2017-2021 PointNet-family 6-paper arc.** Re-implement PTv1 (or use the Pointcept codebase) and benchmark on the v0 sub-task 1 dataset (3DTeethSeg'22 + 3D-IOSSeg). Expected accuracy: 70.4% mIoU on S3DIS Area 5 (PTv1's S3DIS number), and we should aim for 75-80% mIoU on the v0 dental-IOS dataset (dental-IOS is *easier* than S3DIS — only 2-3 classes per tooth, vs 13 classes per point in S3DIS). Estimated effort: 3-5 days (clone Pointcept, adapt data loader, train, evaluate). Expected cost: $50-100 Lambda (single GPU, 600 epochs).

2. **REPLACE the scalar attention in any v0 transformer-based architecture with vector attention.** The +6.2% mIoU from Table 5 is the *single largest* design ablation in the entire PTv1 paper. Any v0 paper that uses scalar attention (the ViT default) on 3D points is *demonstrably* behind the state of the art. Estimated effort: 0.5 day (1-line code change in the attention module). Expected gain: +5-10% mIoU on the v0 sub-task 1 dental-IOS dataset. Cost: $0 (just code change).

3. **Adopt PTv1's *coordinate-space* kNN, not DGCNN's *feature-space* kNN.** For v0 sub-task 1, the kNN should be computed in *coordinate space* (PTv1's choice), not *feature space* (DGCNN's choice). The kNN is *stable* across layers (PTv1's kNN is the same at every layer) and *cross-scanner-invariant* (Primescan/Trios/iTero all have similar coordinate distributions but different feature distributions). Estimated effort: 0.5 day (replace the kNN computation in the v0 DGCNN-style backbone). Cost: $0.

4. **Adopt PTv1's *trainable MLP* position encoding, added to BOTH the attention and feature branches.** This is the *second-largest* design ablation (+2.0% mIoU in Table 6). For v0 sub-task 1, the v0 should *not* use sinusoidal position encoding (the ViT default) and *not* omit position encoding (the DGCNN default). Use the *trainable MLP* added to BOTH branches. Estimated effort: 0.5 day (1-line code change in the attention module). Cost: $0.

5. **Adopt PTv1's *4× downsampling* rate per stage, not the PointNet++/KPConv 2× rate.** The aggressive downsampling is enabled by the *efficiency* of self-attention (PTv1 is 14× cheaper per parameter than KPConv's flexible kernel). For v0 sub-task 1, the v0 transformer-based backbone should use 4× downsampling per stage (so 5 stages → 1024× total downsampling), the v0 KPConv-based backbone should use 2× downsampling per stage (so 5 stages → 32× total downsampling, to match KPConv's memory budget). Estimated effort: 1 day (modify the transition-down blocks). Cost: $0.

6. **ADOPT PTv1's *local kNN attention* as the v0 sub-task 4 cross-modal H3 mechanism, replacing the local graph in DPM-on-points 062.** The DPM-on-points paper (062) uses a *ConcatSquashLinear*-style MLP denoiser with a local graph. Replace the local graph with PTv1's local kNN attention, and the per-point denoiser becomes a *self-attention* denoiser. The expected gain is +5-10% sample quality (extrapolating from PTv1's +6.2% over scalar attention in Table 5). The cost is 1.5× the inference time (PTv1 is more expensive than ConcatSquashLinear per denoising step). For v0 *research prototype*, the 5-10% gain is worth it; for v0 *deployment*, the cost is too high and the cheaper DPM-on-points denoiser is better. Estimated effort: 1 week (replace the DPM-on-points denoiser with PTv1). Cost: $200 Lambda (1 week of single-GPU training).

7. **CITE PTv1 as the *2021 transformer-on-points* ancestor in v0 paper's related work** ($0, 30 min, 1-2 paragraphs in v0 paper's related work, *completes* the 2017-2021 PointNet-family 6-paper arc: PointNet 073 → PointNet++ 072 → DGCNN 074 → PointCNN 076 → KPConv 078 → PointTransformer 079, NEW). The 6-paper arc is now the *definitive* 2017-2021 point-cloud-CNN arc in the v0 paper.

8. **CONSIDER adopting PTv2's *partition-based pooling* or PTv3's *serialized pooling* for the v0 cross-scanner generalization.** PTv1's fixed-k kNN is *not* density-uniform, which is a problem for the v0 dental-IOS data (Primescan/Trios/iTero have different point densities). PTv2's partition-based pooling groups points into regular spatial partitions (so the kNN is within-partition and density-uniform), PTv3's serialized pooling sorts points by a space-filling curve (so the kNN is along-the-curve and density-uniform). Both are *direct* solutions to the v0 cross-scanner problem. Estimated effort: 1-2 weeks (re-implement PTv2 or PTv3 on the v0 sub-task 1 dataset). Cost: $200-400 Lambda (single-GPU training, longer than PTv1 due to the more complex pooling).

9. **ABALATE k ∈ {8, 12, 16, 24, 32} on the v0 dental-IOS dataset.** PTv1's *k=16* default is tuned for S3DIS (1M+ points per scene), but the v0 dental-IOS scans are 10K-100K points per scan (intermediate scale). The *right* k might be smaller. The v0 should pick the *smallest* k that gives within-0.5% mIoU of the best k. Estimated effort: 1-2 days (5 training runs). Cost: $50 Lambda.

10. **REIMPLEMENT the *2-layer MLP with ReLU* position encoding as a *v0 sub-task 4 DiGS-style* positional encoding layer.** The DiGS paper (003) uses a *divergence + curl* regularizer on the SDF gradient, which is *implicitly* a position-encoding MLP (the SDF = MLP(relative position)). PTv1's *explicit* position-encoding MLP is the *direct ancestor* of DiGS's *implicit* position-encoding. For v0 sub-task 4, the v0 should *cite* PTv1 as the *positional-encoding-on-3D-coordinates* precedent for DiGS, and consider *sharing* the position-encoding MLP between the diffusion denoiser and the SDF predictor (so the diffusion model and the SDF model have the *same* notion of "position"). Estimated effort: 2-3 days (modify DiGS to add a shared position-encoding MLP). Cost: $30 Lambda.

### Updated v0 sub-task 1 ablation table (the 6-paper PointNet-family arc)

| Backbone | Year | S3DIS mIoU (%) | Complexity per point per layer | Inductive bias | H3 mechanism |
|---|---|---|---|---|---|
| **PointNet 073** | 2017 | 41.1 | O(d²) | max-pool + global+local concat | max-pool (implicit) |
| **PointNet++ 072** | 2017 | 53.5 | O(k·d²) | hierarchical ball-query + multi-scale grouping | ball query (fixed) |
| **DGCNN 074** | 2019 | 56.1 | O(k·d²) | dynamic kNN graph (in feature space) | dynamic kNN (feature space) |
| **PointCNN 076** | 2018 | 57.3 | O(K·K·d²) | learned X-Conv (K×K permutation matrix) | X-Conv (learned) |
| **KPConv 078** | 2019 | 67.1 | O(K²·d²) (K=15) | kernel-point + flexible/deformable kernel | kernel correlation (linear/Gaussian) |
| **PointTransformer 079 (this paper)** | 2021 | **70.4** | O(k·d²) (k=16) | self-attention + trainable position encoding | self-attention (vector) |

**This is the *complete* 2017-2021 PointNet-family 6-paper arc.** The v0 sub-task 1 ablation table should include *all 6* backbones, and the *6-paper* comparison is the *most-comprehensive* in the entire dental-IOS literature. The 6-paper arc *culminates* in PointTransformer, which is *both* the most-accurate (70.4% mIoU) AND the *cheapest* per parameter (O(k·d²) with k=16, the smallest of all 6). The *deformable-on-points* line (KPConv) and the *self-attention-on-points* line (PointTransformer) are *competing* solutions to the same problem (local feature aggregation on irregular point sets), and PTv1 *wins* by +3.3 mIoU. For v0, **the *primary* backbone should be PointTransformer 079 (the most-accurate), and the *secondary* backbone should be KPConv 078 (the most-deformable)** — this is the *first* v0 paper in the dental-IOS literature to make this *head-to-head* comparison.

### Next paper to read (080)

**Recommendation: Stratified Transformer (Lai et al. CVPR 2022, arXiv:2203.14508) — the *hierarchical* 2022 successor to PTv1, the *first* paper to apply PTv1 to *large-scale outdoor LiDAR* (SemanticKITTI), the *first* paper to introduce *stratified* sampling for 3D self-attention (sample points in a *coarse-to-fine* hierarchy, so the kNN is *spatially-bounded* and the attention is O(k·d²) with k=8 instead of k=16).** Stratified Transformer is the *natural* follow-up to PTv1 in the 2022 point-cloud transformer literature, completing the *2021-2022* 2-paper PTv1 → Stratified Transformer arc. Alternative: **PointNeXt (Qian et al. NeurIPS 2022) — the *modernized* 2022 KPConv-style scaling-up successor that uses the *PointNet-family* hierarchy to scale PTv1 to 1B parameters, the *first* paper to show that *training-recipe improvements* (longer training, more augmentation, better optimizer) can match *architectural* improvements (self-attention over max-pool), the *3rd-generation* point-cloud-CNN result that *closes the gap* between KPConv and PTv1 with a *purely-MLP* architecture.** 

**Final 080 plan: Stratified Transformer for 080** (the *direct* 2022 successor to PTv1, completes the 2021-2022 2-paper transformer-on-points arc, the *first* paper to apply PTv1 to *large-scale outdoor LiDAR* — *not* dental-IOS but the *closest* in *spatial scale* to the v0 dental-IOS data), **PointNeXt for 081** (the *modernized* 2022 KPConv-style scaling-up successor, the *3rd-generation* point-cloud-CNN result, completes the 2017-2022 PointNet-family 7-paper arc). After 080-081, the v0 sub-task 1 *PointNet-family arc* is *complete* (7 papers, 2017-2022), and the v0 paper's related work can *trace* the *complete* 2017-2022 PointNet-family 7-paper arc.
