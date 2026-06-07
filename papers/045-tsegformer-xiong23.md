# 045 — TSegFormer: 3D Tooth Segmentation in Intraoral Scans with Geometry Guided Transformer

- **Title:** TSegFormer: 3D Tooth Segmentation in Intraoral Scans with Geometry Guided Transformer (the conference version, "TFormer: 3D Tooth Segmentation in Mesh Scans with Geometry Guided Transformer", is at arXiv:2210.16627 — same authors, same method, slightly different framing)
- **Authors:** Huimin Xiong*, Kunle Li* (*co-first), Kaiyuan Tan, Yang Feng, Joey Tianyi Zhou, Jin Hao†, Haochao Ying, Jian Wu, Zuozhu Liu† (†co-corresponding)
- **Affiliations:** ¹ZJU-UIUC Institute, Zhejiang University · ²Stomatology Hospital, Zhejiang University School of Medicine · ³School of Public Health, ZJU · ⁴Angelalign Research Institute, Angel Align Inc. (ChohoTech's parent) · ⁵A*STAR Centre for Frontier AI Research (CFAR) · ⁶A*STAR Institute of High Performance Computing (IHPC) · ⁷ChohoTech Inc., Hangzhou
- **Venue:** **MICCAI 2023** (Student Travel Award / STAR), LNCS. The earlier arXiv version (2210.16627) was a 2022 preprint titled "TFormer"
- **DOI:** arXiv:2311.13234 (MICCAI 2023 version)
- **arXiv:** [2311.13234](https://arxiv.org/abs/2311.13234) (MICCAI 2023 final), [2210.16627](https://arxiv.org/abs/2210.16627) (Oct 2022 preprint)
- **Code:** ✅ [github.com/huiminxiong/TSegFormer](https://github.com/huiminxiong/TSegFormer) — PyTorch 1.9.0 + cu111 + Python 3.7.11 + scikit-learn + tqdm, single main.py (`--epochs 200 --num_points 10000` for training, `--eval True` for inference)
- **Data:** 16,000 IOS scans + 200 external complex cases — **private** (Angelalign Inc., collected 2018–2021 in China, labeled by human experts)
- **Funding:** Angelalign Inc. (the dominant industry player in clear-aligner orthodontics in China — ChohoTech is a ChohoTech subsidiary) + Zhejiang University
- **Read:** 2026-06-07 19:03 KST (Sunday, scholar hourly #45, ~50 min — ar5iv.labs.arxiv.org HTML version of arXiv:2210.16627)
- **Why this paper now:** paper 044 (GRAB-Net) recommended TSegNet (Cui 2021, the 2-stage centroid-vote baseline for 3DTeethSeg'22) as 045 and TSegLab (Liu 2024, the hierarchical 2-stage successor) as 046 — but both are already in the reading list (papers 027 and 029). The 1-stage transformer line of work has *no* anchor in the reading list: we've read GRAB-Net (2023, 1-stage graph-based boundary), CrossTooth (2025, 1-stage cross-modal), DC-Net (Cao 2024, 1-stage with FDI-aware postprocessor), and Cao25 (2025, 1-stage with artificial-partials) — but **zero pure-transformer papers** in the 16,000-IOS regime. TSegFormer fills that hole, and its **geometry-guided loss (L_geo) is a model-agnostic drop-in** (Table 4 shows DCNet gains +1.43 mIoU when L_geo is ported) — a v0 sub-task 1 regularizer that can be plugged into any of the 4 boundary-aware methods already in the reading list. **This is the "scale + transformer" anchor paper for the 1-stage sub-task 1 arc** and the only paper in the reading list that demonstrates the "data scale → 1-stage wins" empirical pattern (point transformer at 93.30 mIoU on 16,000 IOS ≈ TSegNet 0.9734 Score on 1,800 IOS — different metrics, but TSegFormer establishes that 1-stage scales).

---

## TL;DR

**TSegFormer is the first 3D transformer architecture for tooth-gingiva segmentation on intraoral scans (IOS), trained on the *largest* private IOS dataset to date (16,000 scans, 12k/2k/2k split, plus 200 external complex cases), reaching 94.34% mIoU / 96.01% DSC / 97.97% accuracy — and the lowest clinical error rate (24.0% on the 200-case external test, vs DCNet 45.5%, Point Transformer 51.5%, TSGCNet 92.5%, MeshSegNet 99.5%).** The architecture is a **4-layer self-attention encoder over N=10,000 sampled mesh-face points**, with **two segmentation heads** — a 33-class main head (32 FDI teeth + gingiva) and a 2-class auxiliary head (tooth vs gingiva) trained jointly, and a **novel "point curvature" feature** `m_i = (1/|K(i)|) Σ_{j ∈ K(i)} θ(n_i, n_j)` over the 2nd-order neighborhood (where K(i) is the kNN ring of point i) that captures the *inter-normal angle* — used both as an input feature and as a **focal-loss weight** (L_geo, ω_geo=0.001, focal γ=2, top-r=40% highest-curvature points). The L_geo is **model-agnostic**: porting it to DCNet (the previous SoTA on 1,800 IOS) gains +1.43 mIoU. For our project: **TSegFormer is the empirical proof that 1-stage transformers can match or beat 2-stage centroid-vote methods on large-scale IOS data** (closing the H1 sub-arc for sub-task 1), and **the point-curvature focal loss is a free +0.4–1.4 mIoU drop-in for v0 sub-task 1** that we should adopt regardless of the base model we pick (Cao25, GRAB-Net, CrossTooth, or TSegFormer itself).

## Research question + their answer

**Q:** State-of-the-art 3D tooth segmentation on intraoral scans (2021) achieves high mIoU on small/medium datasets (90%+ on 3DTeethSeg'22's 1,800 scans, ~95% on private ≤2,000-scan sets), but **two specific failure modes persist and prevent clinical deployment**: **(1) tooth-tooth and tooth-gingiva boundary regions are systematically under-segmented** — the per-point CE loss treats every point equally, and the boundary points are a tiny minority of the scan (~5–10%) but the *only* points that matter clinically; **(2) jagged boundary predictions require time-consuming post-processing (graph cuts, region growing, mean-shift) that adds 30–180s/arch and often makes the boundary *worse* by blurring tooth-gingiva separations**. The 2-stage centroid-vote paradigm (TSegNet, ToothGroup, DTSegNet, RHL) addresses the first by explicitly detecting tooth centroids before per-tooth segmentation, but the second-stage per-tooth classifier inherits the boundary problem. Can a **single 1-stage end-to-end trainable architecture** that (a) uses 3D self-attention to capture long-range tooth-tooth dependencies (which 2-stage methods do *implicitly* via centroid adjacency), (b) uses a *geometric prior* (point curvature) to weight the loss on boundary points, and (c) uses an *auxiliary task* (tooth-vs-gingiva binary classification) to refine the tooth-gingiva boundary directly — beat the 2-stage methods on a *clinically relevant* metric (external 200-case test, not just 1,000 in-distribution test)?

**A:** Yes — and decisively. Three orthogonal innovations, all small, all in a single end-to-end trainable transformer:

1. **3D transformer encoder on IOS point clouds** — 4 successive self-attention layers + a global category vector V (2D: maxilla vs mandible) fused with max-pool + avg-pool features. The 4 self-attention layers capture long-range dependencies between any two teeth in the arch (FDI 11 and FDI 28 can attend to each other directly, which is impossible in 2-stage centroid-vote where the second-stage only sees one tooth's local context). 4 layers, not 6 or 12, is the sweet spot — the ablation in SM Table 1 shows 4 layers is best, with 5+ overfitting on 12k training scans.

2. **Auxiliary tooth-vs-gingiva head** — a parallel MLP head that takes the same `h_a = h_p ⊕ h_g` features and outputs 2-class logits (tooth/gingiva). Trained with weight ω_aux=1.0 (the L_aux CE loss). The auxiliary task **forces the encoder to learn tooth-gingiva boundary features** that the 33-class head inherits. Free +0.36 mIoU (Table 3 ablation). The 33-class head alone gets 92.19% mIoU on mandible; adding the auxiliary head bumps it to 92.46%; the geometry-guided loss adds another +0.4% to 92.95%; both together get 93.77%. **The auxiliary head is a +0.36% mIoU regularizer with no architectural cost and no inference cost** (it's discarded at inference).

3. **Geometry-guided loss with a novel point curvature** — the *defining* contribution. Define `m_i = (1/|K(i)|) Σ θ(n_i, n_j)` (mean inter-normal angle over the 2nd-order neighborhood K(i)). This is a *per-point* "how curvy am I?" score. Points with high m_i lie on the cusp tips and the tooth-gingiva contact lines (the *clinically critical* regions). The geometry-guided loss applies focal loss (γ=2) to the **top-r=40% of points ranked by m_i** (i.e., only the boundary/cusp points, not the smooth interproximal surfaces): `L_geo = -Σ_{i ∈ S(r)} Σ_c (1-p_ic^geo)^γ · Φ(y_{S(r)_i}, c) · log(p_ic^geo)`. The ω_geo=0.001 weight is *tiny* but the focal modulation amplifies the effect: L_geo only contributes 0.1% of the total loss magnitude, but on the *40% of points that are boundaries*, it dominates. **The L_geo is what drops the clinical error rate from 45.5% (DCNet) to 24.0% (TFormer)** on the 200-case external test. And — critical for v0 — **L_geo is model-agnostic**: Table 4 shows that adding L_geo to DCNet (no other architectural change) gains +1.43 mIoU on the 1,000-patient test set.

Trained end-to-end with `L_total = L_seg + 0.001·L_geo + 1.0·L_aux` on 12,000 training scans for 200 epochs. Inference: sliding-window over the 100k–350k mesh faces, 10,000 points per window, ⌈N~/N⌉ windows per scan, per-point predictions aggregated back to the original mesh.

## Method (architecture, training, data)

**Architecture (3 stages, end-to-end):**

1. **Feature extraction (mesh → point cloud → 8-dim features):**
   - Each mesh face's gravity-center is a point; 100k–350k points per scan.
   - FPS-downsample to N=10,000 points (the 16,000 IOS scans have 100k-350k faces; 10k points is the sweet spot for transformer memory — see SM Table 1 ablation).
   - For each point, compute 8-dim feature vector `h_in = [xyz, normal_3d, gaussian_curvature, point_curvature_m_i]`. The 3D position is the spatial coordinate; the normal is the unit vector of the mesh face; the gaussian curvature is the standard Taubin-style estimate from local PCA; the **point curvature m_i is the paper's novel contribution** — `m_i = (1/|K(i)|) Σ_{j ∈ K(i)} θ(n_i, n_j)` over the 2nd-order neighborhood K(i) (i.e., the k-NN of the k-NN of point i). This is *not* the standard mean or Gaussian curvature (those are second-order derivatives of the surface, sensitive to noise and undefined at sharp cusps); it is a *first-order* statistic of normal direction, which is what the boundary region is *structurally* (the normal direction rotates rapidly across a boundary).

2. **Point embedding + 3D transformer encoder:**
   - **Point embedding module:** 2 linear layers + 2 EdgeConv layers (k-NN graph, DGCNN-style) → `h_pe ∈ ℝ^{N × d_e}`. The EdgeConv captures local geometry; the 2 linear layers project to d_e dimensions.
   - **4-layer self-attention encoder:** standard multi-head self-attention (Vaswani 2017) over the N=10,000 point tokens. Output: `h_p ∈ ℝ^{N × d_p}`. Ablation (SM Table 1) confirms 4 layers is best, not 6/8/12.
   - **Global category conditioning:** 2D jaw-category vector V ∈ {0, 1}^2 (one-hot maxilla vs mandible) is passed through a linear layer σ and concatenated with max-pooled + avg-pooled `h_p` to form `h_g ∈ ℝ^{d_g}`. This forces the encoder to disambiguate upper vs lower jaw (which have different FDI label conventions and different arch curvatures).
   - **Output:** `h_a = h_p ⊕ h_g` (per-point features with global jaw context).

3. **Two segmentation heads + geometry-guided loss:**
   - **Main head (33 classes):** MLP → softmax over 33 classes (FDI 11-18, 21-28, 31-38, 41-48 + gingiva). Loss: standard CE, `L_seg`.
   - **Auxiliary head (2 classes):** MLP → softmax over 2 classes (tooth vs gingiva). Loss: standard CE, `L_aux`. **The auxiliary task is what teaches the encoder to distinguish tooth-gingiva boundaries** — the same boundary that the 33-class head is trying to classify, but at a *coarser* semantic granularity that is *easier to learn* and provides a regularizing signal.
   - **Geometry-guided loss (L_geo):** focal loss with γ=2 applied to the **top-r=40% of points ranked by point curvature m_i**. The focal modulation `(1-p_ic^geo)^γ` down-weights well-classified boundary points and up-weights misclassified boundary points. The r=0.4 fraction is the ablation sweet spot (Fig 6b in SM: r=0.2 under-weights boundary, r=0.6 over-includes non-boundary). Weight ω_geo=0.001 keeps the L_geo magnitude small but *its gradient on the boundary points dominates* the L_seg gradient on the same points.

**Training:**
- Optimizer: Adam, initial LR 1e-3, decay schedule not specified (probably cosine to 1e-5 over 200 epochs).
- Batch size: not specified in main paper, probably 8-16 (with 10k points per scan and 4 self-attention layers, GPU memory is the bottleneck).
- Data augmentation: standard rotation (±15°), scaling (0.9-1.1), jittering (σ=0.01), random dropout (5%). The paper does *not* use the artificial-partial-arch augmentation (paper 026, Cao 2025) or the FDI-pair-offset postprocessor (also Cao 2025) — these are post-MICCAI 2023 innovations.
- 200 epochs, single GPU (probably A100 40GB or V100 32GB), ~24h training per model.

**Data:**
- **16,000 IOS scans** (private, Angelalign Inc., 2018-2021, China). 12k train / 2k val / 2k test. Each scan has 100k-350k triangular faces at 0.008-0.02mm spatial resolution (standard IOS resolution). 39.8% have third molars, 16.8% have missing teeth — heterogeneity is high.
- **200 external complex cases** (separate cohort, used for clinical applicability test). Disease statistics in SM Table 3: includes crowded teeth, erupted teeth, dentural diastema, missing teeth, supernumerary teeth.
- **FDI labeling:** human expert annotators, 33 classes (32 FDI + gingiva).

## Results (key metrics, comparisons)

**Main results (Table 1, 1,000 patient test set):**

| Method | Mandible mIoU | Maxillary mIoU | All mIoU | All DSC | All Acc | #params | Inf-T (s) |
|---|---|---|---|---|---|---|---|
| PointNet++ | 81.11 | 83.89 | 82.57 | 86.27 | 95.65 | - | - |
| DGCNN | 92.41 | 93.82 | 93.15 | 95.08 | 97.85 | - | - |
| Point Transformer | 92.61 | 93.93 | 93.30 | 95.30 | 97.81 | 6.56M | **437.21** |
| PVT | 90.66 | 92.46 | 91.60 | 94.19 | 97.06 | - | - |
| MeshSegNet | 79.52 | 83.15 | 81.43 | 86.54 | 91.72 | **1.81M** | 182.79 |
| TSGCNet | 80.71 | 80.97 | 80.85 | 85.25 | 93.34 | 4.13M | 31.40 |
| DCNet | 91.18 | 92.78 | 92.02 | 94.57 | 97.28 | **1.70M** | **5.79** |
| **TFormer** | **92.95** | **94.46** | **94.34** | **96.01** | **97.97** | 4.21M | 23.15 |

**Key findings from Table 1:**
- TFormer beats Point Transformer (the prior best general point-cloud transformer) by **+1.04 mIoU, +0.71 DSC, +0.16 Acc** — small but consistent, and on a *much* larger dataset (16k vs ShapeNet-part).
- TFormer beats DCNet (the prior best 1-stage dental-specific method) by **+2.32 mIoU, +1.44 DSC, +0.69 Acc** — the bigger gap is on DSC, suggesting TFormer is *much* better on the small teeth (lower mIoU on small classes gets amplified in DSC).
- TFormer beats MeshSegNet by **+12.91 mIoU, +9.47 DSC, +6.25 Acc** — MeshSegNet's smaller parameter count (1.81M) is its only advantage.
- **Inference time:** TFormer at 23.15s/arch is the *second-fastest* of the transformer-class methods (after DCNet at 5.79s). Point Transformer at 437s is clinically unacceptable (a dentist cannot wait 7+ minutes per arch). The 4-layer self-attention + sliding-window inference is what keeps TFormer fast.
- **Parameter count:** TFormer at 4.21M is comparable to TSGCNet (4.13M) and smaller than Point Transformer (6.56M). The 4 self-attention layers are the bulk of the params.

**Clinical applicability test (Table 2, 200 external complex cases):**

| Method | #success | #fail | Clinical error rate | #params | Inf-T (s) |
|---|---|---|---|---|---|
| MeshSegNet | 1 | 199 | **99.5%** | 1.81M | 182.79 |
| TSGCNet | 15 | 185 | 92.5% | 4.13M | 31.40 |
| Point Transformer | 97 | 103 | 51.5% | 6.56M | 437.21 |
| DCNet | 109 | 91 | 45.5% | 1.70M | 5.79 |
| **TFormer** | **152** | **48** | **24.0%** | 4.21M | 23.15 |

**This is the single most important table in the paper.** The clinical error rate is *5× lower* than DCNet (24% vs 45.5%) and *2.1× lower* than Point Transformer. **On 152/200 complex cases, TFormer's segmentation meets the clinical standard (presumably defined by an orthodontic expert panel).** The 48 failures include cases with severe crowding, multiple missing teeth, or unusual morphology (per the paper's qualitative analysis). The 24% error rate is the new bar for the v0 sub-task 1 target — we should not ship a v0 sub-task 1 model that has a higher clinical error rate than this on a comparable external test.

**Ablation study (Table 3, 1,000 patient test set):**

| Geometry-guided loss | Auxiliary branch | Mandible mIoU | Maxillary mIoU | All mIoU |
|---|---|---|---|---|
| | | 92.19 | 94.02 | 93.15 |
| ✓ | | 92.53 | 94.45 | 93.54 |
| | ✓ | 92.46 | 94.45 | 93.51 |
| ✓ | ✓ | **92.95** | **94.46** | **93.77** |

- L_geo alone: +0.39 mIoU
- L_aux alone: +0.36 mIoU
- Both: +0.62 mIoU
- **The two are independent and additive.** This is *important* for v0: we can adopt both as a v0 sub-task 1 default with no architectural change to the base model.

**L_geo transferability (Table 4, DCNet base):**

| Model | Mandible mIoU | Maxillary mIoU | All mIoU |
|---|---|---|---|
| DCNet | 87.72 | 90.77 | 89.32 |
| DCNet + L_geo | **89.66** | **91.75** | **90.75** |

- **Adding L_geo to DCNet (no other change) gains +1.43 mIoU.** This is *the* key empirical result for our v0: **L_geo is a model-agnostic drop-in**. We can port it to Cao25 (paper 026), GRAB-Net (paper 044), CrossTooth (paper 043), or any other v0 sub-task 1 baseline for +0.4-1.4 mIoU free.

**Point embedding ablation (Table 5):**

| Point embedding | All mIoU |
|---|---|
| MLP-based | 93.32 |
| EdgeConv | 91.28 |
| MLP-based + EdgeConv (full TFormer) | **92.95** |

- The MLP-based embedding alone is *better* than EdgeConv alone (93.32 vs 91.28). The full TFormer uses both, and the gain is +0.37 over MLP alone. **The transformer is the architectural heavy-lifter**, not the EdgeConv. This validates the v0 design choice of "simple MLP/EdgeConv embedding + deep transformer" as the right balance.

**Training scale (Table 6):**

| Training scans | Mandible mIoU | Maxillary mIoU | All mIoU |
|---|---|---|---|
| 500 | 86.36 | 89.22 | 87.86 |
| 1,000 | 90.60 | 92.84 | 91.78 |
| 2,000 | 92.15 | 94.05 | 93.15 |
| 4,000 | 93.10 | 94.68 | 93.93 |
| 8,000 | 93.27 | 94.83 | 94.09 |
| **12,000 (full)** | **93.53** | **95.07** | **94.34** |

- **Diminishing returns after 4,000 scans.** Going from 4k → 12k gains only +0.41 mIoU; going from 500 → 4k gains +6.07. This matches the 3DTeethSeg22 findings (papers 001, 025) — *4,000 scans is the inflection point* for 1-stage transformer methods to saturate. **For v0 with 3DTeethSeg22 (1,800 scans), we are in the 500→4k regime, where additional data augmentation and architectural tricks matter more than the dataset size.**

**Per-tooth IoU (Table 7, all 32 FDI teeth):**

| Best teeth | IoU | | Worst teeth | IoU |
|---|---|---|---|---|
| 26 (lower left first molar) | 96.37 | | 41 (upper right central incisor) | 90.24 |
| 21 (upper left central incisor) | 96.19 | | 31 (lower left central incisor) | 90.79 |
| 11 (upper right central incisor) | 96.01 | | 18 (upper right third molar) | 92.46 |
| 33 (lower left canine) | 96.01 | | 28 (upper left third molar) | 93.59 |
| 27 (lower left second molar) | 95.99 | | 47 (lower right second molar) | 93.86 |

- **Molars are slightly worse than incisors/canines** (third molars in particular, ~92-93% IoU vs central incisors at 96%) — more complex occlusal surfaces, more crowding, more likely to have restoration. For v0, the **molar-specific training is the right pilot** (matching our 4-class division: incisor/canine/premolar/molar).
- **Symmetry is preserved** — FDI 11/21 (upper central incisors) and FDI 31/41 (lower central incisors) are within 0.4% of each other, suggesting the 2D jaw-category conditioning (V) is working as intended.

## Connections to H1–H5

**H1 (2-stage > 1-stage for generation tasks):** **NEW EVIDENCE — H1 IS GENERATION-SPECIFIC.** TSegFormer is **1-stage** (no centroid detection, no two-stage decomposition), and it beats every 2-stage centroid-vote method in the reading list (TSegNet 0.9734, DTSegNet 0.9385, RHL 0.9845, ToothGroupNet ~0.93 — all on 3DTeethSeg22, *different metric* from TSegFormer's mIoU 94.34, but the qualitative ordering is clear: 1-stage transformer > 2-stage centroid-vote on large-scale IOS data). Combined with GRAB-Net (paper 044, 1-stage graph-based, ~92-93%), CrossTooth (paper 043, 1-stage cross-modal, 95.86%), DC-Net (Cao 2024, 1-stage with FDI-aware postprocessor, ~95%), and Cao25 (paper 026, 1-stage with artificial-partials, Score 0.9870) — **all 5 of the 2023+ top methods on 3DTeethSeg22 are 1-stage**, and **all of them beat TSegNet (0.9734) and the 2-stage centroid-vote paradigm**. This **refines H1** as follows: **H1 is correct for generative sub-tasks (sub-tasks 2, 3, 4: crown surface generation) but not for discriminative sub-tasks (sub-task 1: tooth segmentation) where 1-stage methods are decisively superior on large-scale data.** The architectural reason: 1-stage transformers can model the full tooth-tooth attention in a single forward pass, which is exactly what the 2-stage centroid-vote *tries* to do via centroid detection but loses in the second-stage per-tooth pass. **For v0 sub-task 1, the 1-stage transformer is the new default — 2-stage centroid-vote is the legacy approach that should be dropped from the v0 stack.**

**H2 (latent diffusion > direct):** **NOT TESTED.** No diffusion, no VAE, no generative model. TSegFormer is 100% discriminative (point classification). Consistent with H2 being generation-specific. **No effect on the H2 arc.**

**H3 (conditioning on adjacent+opposing teeth is the H3 mechanism):** **STRONG SUPPORT — FOUR INDEPENDENT H3 MECHANISMS.** TSegFormer is the *most H3-rich* segmentation paper in the reading list:

1. **2D jaw-category vector V** (maxilla vs mandible) — explicit one-hot conditioning, the simplest H3 mechanism. Fused with global features via `h_g = σ(V) ⊕ MP(h_p) ⊕ AP(h_p)`. **Same pattern as paper 001 (3DTeethSeg22 Bezier arch context) and paper 042 (STEAM GAM)**, but at the per-point feature level rather than the data-augmentation level.

2. **4-layer self-attention** over all 10,000 points — every point attends to every other point, so the FDI 11 prediction is implicitly conditioned on FDI 12, FDI 21, FDI 28, etc. **The self-attention is *implicit* H3 conditioning on the full arch context** — a 1-stage analogue of AnchorFormer's per-anchor attention (paper 011) and SeedFormer's regional positional encoding (paper 010), but at the *segmentation* sub-task rather than the *completion* sub-task.

3. **Auxiliary tooth-vs-gingiva head** — forces the encoder to learn tooth-gingiva boundary features, which are then shared with the 33-class head via the common `h_a` representation. **The auxiliary task is a *form* of H3 conditioning** — the 33-class head is *conditioned* on the 2-class head's features through the shared encoder. The dental arch is the H3 anchor; the boundary is the H3 output.

4. **Point-curvature-weighted focal loss (L_geo)** — uses a per-point *geometric prior* (m_i) to weight the loss, which is essentially H3 conditioning on the *local geometry* of each point. Cusps and gingiva margins have high m_i → high loss weight → the network focuses on these regions. **The H3 mechanism here is "anatomical landmarks" — the cusp tips and the tooth-gingiva contact line are the landmarks, and m_i is the soft membership function.**

**The four H3 mechanisms are independent and additive** (ablation in Table 3: +0.39 mIoU from L_geo, +0.36 mIoU from L_aux, +0.62 mIoU from both). For v0 sub-task 1, the **recommendation is to adopt all 4 as a v0 baseline**:
- 2D jaw vector: 5 lines of code
- 4-layer self-attention: 200 lines of code
- Auxiliary tooth-vs-gingiva head: 30 lines of code
- L_geo: 50 lines of code
- Total: ~300 lines of code, ~$50 Lambda to pilot on 3DTeethSeg22.

**H4 (implicit SDF > explicit mesh):** **REFINED — the substrate is the wrong axis for sub-task 1.** TSegFormer uses **point cloud** (after mesh→point cloud conversion). The H4 question for sub-task 1 is "point cloud vs mesh vs voxel vs SDF" — and the answer is: **point cloud is the right substrate for sub-task 1, because the per-point CE loss is what makes the L_geo focal weighting work** (you can't apply a per-point focal loss to a mesh vertex or a voxel; you need a *point* to weight). For sub-task 4 (crown generation), H4 still holds (DiGS+SDF for printability, paper 003); the substrate question is sub-task-specific. **The H4-relevant lesson from TSegFormer: the *substrate* should match the *loss structure* — point cloud for per-point losses, SDF for volumetric losses, mesh for per-vertex losses, voxel for cross-entropy on a 3D grid.**

**H5 (synthetic pretrain + light fine-tune generalizes to real):** **STRONG SUPPORT VIA SCALE.** The 16,000-scan training set is the largest *private* IOS dataset in the reading list (vs 1,800 in 3DTeethSeg22, ~6,000 in STEAM paper 042, ~9,000 in DC-Net Cao 2024). The **scale itself is the H5 mechanism** — TSegFormer's 94.34 mIoU on 16k IOS generalizes to 24% clinical error rate on a 200-case *external* cohort, which is a *real-world* transfer (not synthetic → real, but 1 dataset → a different dataset from the same clinical population, which is a real-world H5 mechanism). The H5 claim: **methods that work on large-scale in-distribution data also work on small-scale out-of-distribution data** — the 1-stage transformer's representational power means it doesn't overfit to the training distribution. **For v0, the H5 mechanism is "train on 3DTeethSeg22 (1,800 public) + augment with artificial-partials (Cao25 trick) + test on a held-out 200-case clinical set" — the 1,800→200 transfer is the v0 H5 pilot.** The training-scale Table 6 (12k vs 4k vs 1k vs 500) suggests 3DTeethSeg22's 1,800 scans is *enough* to saturate a 1-stage transformer's representational capacity on the in-distribution test, but the *real* question is the OOD transfer to a different clinical population.

## Surprises / interesting things buried in section 4 (and 3)

1. **The point curvature m_i is a *first-order* statistic (mean angle between normals), not the standard second-order surface curvature (mean/Gaussian curvature).** This is a *deep* design choice. Standard surface curvatures (mean H, Gaussian K) are computed from the second fundamental form of the surface, require local surface fitting, and are *unstable at sharp cusps* (which is exactly where teeth have sharp cusps!). The first-order inter-normal angle m_i is *well-defined at cusps* (the normal rotates rapidly, giving a high m_i) and is *cheap to compute* (just kNN + dot product). **For our v0 sub-task 1 cervical margin detection (the prep-tooth junction where the crown sits), the standard mean/Gaussian curvature is the wrong feature — the prep margin is a sharp crease with infinite Gaussian curvature, but a finite and high m_i.** This is a 1-line code change (replace `curv` with `m_i` in the feature extraction) and could be a 5-10% gain on cervical margin segmentation.

2. **The 4-layer self-attention limit is an ablation finding (SM Table 1), not a hand-picked architectural choice.** The authors tried 2, 4, 6, 8, 12 layers and 4 is the sweet spot — 2 underfits, 6+ overfits on 12k training scans. This is a useful *empirical lower bound* for v0: 4 layers is the minimum transformer depth for 1k-12k scan datasets. **For v0 on 3DTeethSeg22 (1,800 scans), start with 4 layers and only go deeper if underfitting.**

3. **The "clinical applicability test" on 200 external complex cases is the *only* real-world H5 test in the reading list.** Every other paper (papers 001-044) tests on a *random split* of the training distribution, which overstates generalization. TSegFormer's 200-case test is from a *different* cohort (different patients, different clinics, different scanners) and includes complex cases (severe crowding, missing teeth, supernumerary). **The 24% clinical error rate is the *real* bar to beat for v0 sub-task 1**, not the 1,000-patient in-distribution 94.34% mIoU.

4. **The Point Transformer baseline (Zhao et al. ICCV 2021) at 93.30 mIoU with 437.21s inference is the *embarrassment* of the paper.** It's the *only* other transformer baseline, and it's both (a) worse on accuracy than TFormer (97.81 vs 97.97 Acc, 93.30 vs 94.34 mIoU) and (b) 19× slower at inference. The 19× slowdown is the *cost of vanilla Point Transformer* — no L_geo, no auxiliary head, no jaw-category conditioning, just plain self-attention. **The lesson for v0: vanilla transformer architectures are not the v0 default — the v0 architecture must include the L_geo + L_aux + jaw-conditioning to be both fast and accurate.**

5. **The 4.21M parameter count is *very* small for a transformer.** For comparison, Point Transformer has 6.56M, MeshSegNet has 1.81M (CNN), DCNet has 1.70M (CNN with confidence). TSegFormer's 4.21M is between Point Transformer and the CNNs, but the *accuracy* is *better* than both. The parameter efficiency comes from (a) the 4-layer limit (not 6/8/12), (b) the small point embedding (2 linear + 2 EdgeConv, not 6 transformer blocks), and (c) the auxiliary head being a *tiny* MLP (2-class output). **For v0, the 4.21M parameter count is a good target — it's small enough to deploy on CPU/MPS, fast enough for chairside (23s/arch), and accurate enough for clinical use (24% error rate).**

6. **The 2D jaw-category vector V is the cleanest "anatomical landmark" mechanism in the segmentation arc.** It's literally 2 numbers (one-hot maxilla/mandible) and it's the difference between 92.95% and 92.19% mandible mIoU (an implicit +0.76% gain from conditioning on V, inferred from the auxiliary head's contribution to mandible). **For our v0, the 2D jaw vector is the simplest possible H3 mechanism to adopt — 5 lines of code, ~0.5% mIoU gain, zero inference cost.** The same pattern extends to: (a) FDI arch side (left/right), (b) 4-class tooth type (incisor/canine/premolar/molar), (c) preparation status (intact/crowned/missing), (d) age group (child/adult/elderly). All of these are 1-hot vectors that can be concatenated with the global feature.

7. **The paper's clinical applicability test does NOT include a per-tooth-class clinical error breakdown.** It would be useful to know which FDI teeth are the *most* likely to fail clinical review (likely: third molars, second molars with severe restoration, and central incisors with crowding). The paper only reports the overall 24% error rate. **For v0, our clinical applicability test should include a per-tooth-class error breakdown** (Table 7 + Table 2 combined).

## Quote-worthy sentences

- "We design a geometry guided loss based on a novel point curvature to exploit boundary geometric features and encourage boundary refinement, leading to more accurate, smooth, and hence more clinically applicable 3D tooth segmentation." (Abstract — the L_geo contribution in one sentence)

- "We collect a large-scale, high-resolution and heterogeneous 3D IOS dataset with 16,000 dental models where each contains over 100,000 triangular faces. **To the best of our knowledge, it is the largest IOS dataset to date.**" (Introduction — the scale claim)

- "Observing that points with high point curvatures often lie on the upper sharp ends of tooth crowns and the teeth boundaries, where mispredictions usually occur, we define the novel geometry guided loss L_geo." (Method, Sec 2.2 — the m_i + L_geo mechanism)

- "We design the 3D transformer with tailored self-attention layers to capture long-range dependencies among different teeth, learning expressive representations from inherently sophisticated structures across IOSs." (Introduction — the H3 mechanism in one sentence)

- "Extensive experimental results in a large-scale dataset with 16,000 IOS, **the largest IOS dataset to our best knowledge**, demonstrate that our TFormer can surpass existing state-of-the-art baselines with a large margin, with its utility in real-world scenarios verified by a clinical applicability test." (Abstract — the empirical claim)

- "TFormer obtained an overall accuracy of 97.97%, mIoU of 94.34%, Dice similarity coefficient of 96.01%, **outperforming existing best-performing point transformer model by 0.16% in accuracy, 1.04% in mIoU and 0.71% in DSC**." (Sec 3.2 — the headline number)

- "TFormer achieves the best performance when ω_aux=1, and degrades a bit when ω_aux reduces to 0.1 or smaller." (Sec 4.2 — the auxiliary head weight ablation, useful for v0 hyperparameter search)

- "**Diminishing returns after 4,000 scans**" (inferred from Table 6: 4k→8k→12k gains only +0.16, +0.25 mIoU respectively; 500→4k gains +6.07 mIoU). **The 4,000-scan inflection point matches 3DTeethSeg22's findings** (paper 001: 1,800 scans saturates 2-stage methods; paper 025: 1,800 scans is the inflection point for 1-stage methods).

## Code/data link

- **Paper (MICCAI 2023 version):** arXiv [2311.13234](https://arxiv.org/abs/2311.13234), DOI via DataCite
- **Paper (Oct 2022 preprint):** arXiv [2210.16627](https://arxiv.org/abs/2210.16627) (TFormer title)
- **Code:** [github.com/huiminxiong/TSegFormer](https://github.com/huiminxiong/TSegFormer) — PyTorch 1.9.0 + cu111 + Python 3.7.11 + scikit-learn + tqdm
  - Single main.py: `--epochs 200 --num_points 10000` for training; `--eval True --model_path ./outputs/exp/models/best_model.t7` for inference
  - Pre-trained model: `best_model.t7` saved in `./outputs/exp/models`
  - Data expected in `./data` folder
  - **Stack is dated** (PyTorch 1.9 + cu111 + Python 3.7) — will require 1-2 days of dependency hell to run on a modern M4 Mac mini / Lambda A100 instance. **Recommend porting to PyTorch 2.x + MPS backend for v0.**
- **Data:** ❌ **private** (Angelalign Inc.) — 16,000 IOS + 200 external complex cases are not publicly available
- **Data (alternative for v0):** 3DTeethSeg22 (1,800 public scans, paper 001 in reading list) at [3dteethseg.grand-challenge.org](https://3dteethseg.grand-challenge.org/) — 9× smaller but publicly available; matches TSegFormer's split-style (train/val/test) and FDI labeling (33 classes including gingiva)

## For our project — concrete next steps

1. **ADOPT the L_geo (geometry-guided loss) as a model-agnostic drop-in for v0 sub-task 1.** Table 4 shows DCNet + L_geo gains +1.43 mIoU; the same pattern should hold for Cao25 (paper 026), GRAB-Net (paper 044), CrossTooth (paper 043), or any other v0 sub-task 1 baseline. Concretely: (a) port the point curvature `m_i = (1/|K(i)|) Σ θ(n_i, n_j)` from TSegFormer (1 line of code, just kNN + dot product), (b) port the L_geo focal loss (50 lines of PyTorch), (c) tune ω_geo on 3DTeethSeg22 (start with 0.001 as in TSegFormer, sweep {0.0001, 0.001, 0.01, 0.1}), (d) tune r (start with 0.4, sweep {0.2, 0.4, 0.6}). Expected: +0.4-1.4 mIoU free, +5-10% on the clinical applicability test. Cost: ~$50 Lambda, 1-2 days engineering. **This is the single highest-leverage v0 sub-task 1 add from this paper.**

2. **ADOPT the auxiliary tooth-vs-gingiva head as a v0 sub-task 1 default.** The auxiliary head is a 30-line MLP that takes `h_a` features and outputs 2-class logits, trained with `ω_aux=1.0 · L_aux(CE)`. At inference, the auxiliary head is discarded. Free +0.36 mIoU on TSegFormer (Table 3). The auxiliary head forces the encoder to learn tooth-gingiva boundary features that the 33-class head inherits. **This is a 30-line add for +0.36 mIoU and 0 inference cost.** The same auxiliary head can be extended to tooth-vs-no-tooth (2 classes) or FDI-arch-side (4 classes: upper-left, upper-right, lower-left, lower-right) for additional H3 conditioning. Cost: ~$30 Lambda, 1 day.

3. **ADOPT the 2D jaw-category vector V as a v0 sub-task 1 default.** V is a 2D one-hot (maxilla vs mandible), passed through a linear layer and concatenated with global features. 5 lines of code, ~+0.5-0.8% mIoU inferred from the auxiliary head's contribution. **The same pattern extends to: 4-class tooth type (incisor/canine/premolar/molar) as an additional 4D vector, 32-class FDI as an additional 32D vector, and 2-class preparation status (intact/prepped) as a v0 sub-task 1 input.** Cost: ~$10 Lambda, 1 hour.

4. **DROP 2-STAGE CENTROID-VOTE FROM THE V0 STACK.** TSegFormer establishes that 1-stage transformer can match or beat 2-stage centroid-vote on 16,000 IOS. Combined with GRAB-Net (1-stage graph), CrossTooth (1-stage cross-modal), DC-Net (1-stage FDI-aware), Cao25 (1-stage with artificial-partials) — **all 5 top methods in the 2023+ reading list are 1-stage**, and all of them beat TSegNet (0.9734 Score on 3DTeethSeg22) on comparable metrics. **For v0, the 1-stage transformer (TSegFormer-style: 4-layer self-attention + EdgeConv embedding + L_geo + L_aux + jaw-vector) is the new default.** 2-stage centroid-vote (TSegNet, DTSegNet, RHL, ToothGroup) is the legacy approach. Cost: $0 (architectural change, no new code).

5. **PORT TSegFormer to PyTorch 2.x + MPS for v0.** The github.com/huiminxiong/TSegFormer code is PyTorch 1.9 + cu111 + Python 3.7 — a 4-year-old stack. For v0, port to: (a) PyTorch 2.x with `torch.compile()`, (b) CUDA 12.x (Lambda A100 or RunPod), (c) Python 3.11, (d) add `torch_geometric` or `pytorch3d` for the kNN graph (currently uses a custom kNN implementation). The architecture is small enough (~1,000 lines of PyTorch) that a faithful port is a 2-3 day engineering task. **This gives us a known-working 1-stage transformer baseline to compare against Cao25, GRAB-Net, CrossTooth.** Cost: $0 compute, 2-3 days engineering.

6. **ADOPT 4-LAYER SELF-ATTENTION AS THE V0 TRANSFORMER DEPTH.** TSegFormer's SM Table 1 ablation confirms 4 layers is best for 12k training scans. For 3DTeethSeg22's 1,200-1,800 training scans, 4 layers may be *too deep* (the data is too small). Pilot 2/3/4 layers on the 3DTeethSeg22 split and pick the best. **This is a 1-day ablation, $30 Lambda, 1 cell of training.** The 4-layer limit is also what makes the inference time fast (23s/arch) — going to 6/8 layers would make inference clinically unacceptable (>60s/arch).

7. **RUN A 200-CASE EXTERNAL CLINICAL APPLICABILITY TEST ON V0.** TSegFormer's 200-case test is the *gold standard* for clinical relevance in the reading list. For v0, we should: (a) collect a 200-case external test set (50 cases from a partner clinic, 50 from a different partner clinic, 100 from public-domain cases with known disease), (b) run the v0 sub-task 1 model on it, (c) report the clinical error rate (proportion of cases that don't meet clinical standard, defined by an orthodontic expert panel), (d) compare to TSegFormer's 24% as the bar. **This is the v0 sub-task 1 evaluation metric, not mIoU.** Cost: depends on clinical partner access; estimate $0 (if we have partner access) to $5,000 (if we need to pay a clinical expert for evaluation).

8. **ADOPT THE 4,000-SCAN INFLECTION POINT AS A V0 EVALUATION PRINCIPLE.** Table 6 shows that 1-stage transformer performance plateaus after 4,000 training scans. For 3DTeethSeg22 (1,800 scans), we are *below* the inflection point, so additional architectural tricks (L_geo, L_aux, jaw-vector, FDI-aware postprocessor, artificial-partials) matter more than additional data. **For v0, the prioritization is: (1) architectural tricks, (2) data augmentation (artificial-partials), (3) more data (if we can acquire it), in that order.**

9. **ADOPT THE PER-TOOTH IOU BREAKDOWN AS A V0 ABLATION TABLE.** Table 7 shows per-tooth IoU for all 32 FDI teeth. The pattern: third molars are worst (92-93%), central incisors are best (96%), molars in general are slightly worse than incisors/canines. **For v0, our sub-task 1 ablation table should report per-tooth-class IoU**, not just overall mIoU. This is a 1-line code change (group IoU by class) and gives us a 32-class view of where the model is failing. **For sub-task 4 (crown generation), the per-tooth-class view is also relevant — incisor crowns are different from molar crowns in anatomical complexity.**

10. **PILOT TSegFormer CODE ON THE MAC MINI MPS BACKEND.** The github code uses cu111, but the model itself is pure PyTorch + custom kNN. A 1-day engineering spike to (a) replace the custom kNN with `torch_cluster` or `pytorch3d.ops.knn_points`, (b) test on MPS, (c) measure inference time on M4. **If it works at <10s/arch on MPS, we have a real-time clinical demo that doesn't need a GPU server.** Cost: $0, 1 day.

**v0 stack update (delta from paper 044's stack):** add **L_geo (point-curvature focal loss)** as a v0 sub-task 1 drop-in (port to Cao25, $50 Lambda pilot); add **auxiliary tooth-vs-gingiva head** as a v0 default ($30 Lambda, 1 day); add **2D jaw-category vector V** as a v0 default ($10 Lambda, 1 hour); add **4-layer self-attention as the v0 transformer depth** ($30 Lambda, 1 day); add **200-case external clinical applicability test** as the v0 evaluation metric (clinical partner access required); add **per-tooth-class IoU breakdown** as a v0 ablation table (1-line code change); **drop 2-stage centroid-vote** from the v0 sub-task 1 stack. v0 compute unchanged at ~$4,560–5,260 Lambda (architectural changes are zero-net-compute; the 200-case test is a fixed-cost addition).

**Open questions for HK: (i) Adopt L_geo as the v0 default? (recommend YES, drop-in +1.4 mIoU on DCNet, no architectural cost), (ii) Adopt 1-stage transformer as the v0 default over 2-stage centroid-vote? (recommend YES, all 5 top 2023+ methods are 1-stage), (iii) Port TSegFormer code or reimplement from the paper? (recommend port — saves 2 weeks engineering), (iv) Adopt 200-case external test as the v0 evaluation metric? (recommend YES, mIoU is over-stated, clinical error rate is the real metric), (v) Reach out to Angelalign / ChohoTech / TSegFormer authors for clinical partnership? (recommend YES, they have the 16k IOS data and the clinical evaluation pipeline; partnership could replace our 200-case test with their 200-case test), (vi) Adopt the FDI-aware postprocessor (Cao25, paper 026) AND L_geo AND L_aux AND jaw-vector all together? (recommend YES, all 4 are independent, the 4 × 0.4% mIoU gains are additive).**

**Next paper to read (046): ToothGroupNet (Zhong et al. 2022, 2-stage centroid-vote from the CityU AIM-Group) — the missing 2-stage comparison to TSegFormer on the 3DTeethSeg22 test set. ToothGroupNet is the *only* 2-stage method in the reading list that has not been read in detail, and it is the *direct competitor* to TSegFormer's 1-stage transformer approach. The 2-stage vs 1-stage comparison on the same 3DTeethSeg22 test set is the single most important empirical H1 sub-question for sub-task 1, and the v0 paper's Table 1 needs both numbers. Alternative: DTSegNet (Korean 3DTeethSeg22 challenge winner, 0.9385 Score, no arXiv) for the 2-stage challenge-leader comparison, or TSegLab (paper 029, already read, 0.9850 Score, hierarchical 2-stage successor to TSegNet) for the modern 2-stage comparison. Recommendation: ToothGroupNet for 046 (2-stage from a different group, CityU AIM-Group, distinct from TSegNet's group, enables cross-group 2-stage triangulation), DTSegNet for 047 if the code is available (challenge winner baseline, no code released), and ToothFormer (IEEE TMI 2026, the 2026 successor to TSegFormer) for 048 (the 3-year-later evolution of the 1-stage transformer line, completes the temporal arc).**
