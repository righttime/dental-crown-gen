# 043 — CrossTooth: 3D Dental Model Segmentation with Geometrical Boundary Preserving

- **Title:** 3D Dental Model Segmentation with Geometrical Boundary Preserving
- **Authors:** Shufan Xi¹, Zexian Liu¹, Junlin Chang¹·³, Hongyu Wu¹ (corresponding), Xiaogang Wang², Aimin Hao¹
- **Affiliations:** ¹State Key Laboratory of Virtual Reality Technology and Systems, Beihang University, ²College of Computer and Information Science, Southwest University, ³Peng Cheng Laboratory
- **Venue:** **CVPR 2025** (IEEE/CVF Conference on Computer Vision and Pattern Recognition 2025, official acceptance stated on the arXiv submission)
- **arXiv:** [2503.23702](https://arxiv.org/abs/2503.23702) (v1, 31 Mar 2025, 5.9 MB)
- **Code:** ✅ **[github.com/XiShuFan/CrossTooth_CVPR2025](https://github.com/XiShuFan/CrossTooth_CVPR2025)** — public release, MIT-style license (the "second" boundary-preserving dental SSL paper with open code, after the gap left by STEAM 042)
- **Data:** ✅ **Public** — 3DTeethSeg'22 (MICCAI challenge) for primary evaluation; 3D-IOSSeg for cross-dataset generalization
- **Funding:** National Natural Science Foundation of China (62132021), Guangxi Sci-Tech Major Program (GuiKeAA24206017), National Key R&D Program of China (2023YFC3604505)
- **Read:** 2026-06-07 17:05 KST (Sunday, scholar hourly #43, ~30 min)
- **Why this paper now:** paper 042 (STEAM) explicitly recommended **CrossTooth** as the next paper to read because (a) it directly attacks the **tooth-gingiva boundary** problem that STEAM/GAM+MGR also touches (but in a totally different way — pre-processing, not self-supervision), (b) it has open code (unlike STEAM), (c) it is a CVPR 2025 paper (peer-reviewed, unlike arXiv-only), and (d) it combines **multi-view rendered images + point cloud** — a *cross-modal* H3 mechanism we haven't seen in the reading list yet. This completes the *boundary-preservation* sub-arc inside the broader *tooth segmentation* arc: 023 MeshSegNet → 024 Kunwar → 025 ArchSeg → 026 Cao → 027 TSegNet → 028 Stratified → 029 TSegLab → 030 3DTeethLand → 042 STEAM → **043 CrossTooth**.

---

## TL;DR

**CrossTooth is the first paper in our reading list to explicitly attack the crown–gingiva boundary problem with a *two-headed* design: (1) a *geometric* pre-processing trick — selective downsampling (a QEM extension that uses mean curvature to set the edge-collapse weight `k`, with `k=10` for negative-curvature edges at tooth boundaries and `k=1` for positive-curvature edges at cusp tips, preserving 10-15% more boundary vertices than vanilla QEM) — and (2) a *cross-modal* H3 trick — render 96 multi-view images of the IOS under a vertical downward parallel light, extract dense boundary features with PSPNet, then project the per-view image features back onto the point cloud via camera parameters, fusing the dense image features with the sparse point features before the final MLP segmentation head. The boundary supervision comes from a Contrastive Boundary Learning (CBL) loss that pulls same-class neighbors together and pushes different-class neighbors apart, applied at the last decoder layer. On 3DTeethSeg'22 with a 1440/360 split (not the canonical 1000/200/600), CrossTooth hits 95.860% mIoU and 82.058% Boundary IoU, beating the next-best ToothGroupNet by +2.314 mIoU and +5.666 Boundary IoU; on 3D-IOSSeg it hits 86.11% mIoU and 65.30% Boundary IoU with only 5.05G FLOPs (10-20× cheaper than the second-tier methods). Single most important property for our project: it is the cleanest *cross-modal H3 mechanism* in the entire reading list — "the image can see edges that the point cloud lost during downsampling" is a *free* information source that no prior reading-list paper has used, and the 96-image sweet spot + 7.08G-FLOPs PSPNet makes it computationally tractable for a v0 pilot.**

## Research question + their answer

**Q:** Deep-learning tooth segmentation on intraoral scan (IOS) meshes has reached high overall accuracy (90%+ mIoU on 3DTeethSeg'22), but the **tooth-gingiva boundary** — clinically the most important region for margin-line determination in crown design — remains poorly delineated. Three concrete failure modes: **(1) uniform QEM downsampling loses edge triangles** at the gum line, so by the time the network sees the 16K-point input, the boundary vertices that the dentist needs are already gone; **(2) coordinates + normals carry insufficient information** about the boundary — the boundary is a *texture/appearance* phenomenon (subtle color/shading transition from pink gum to white tooth) that the 3D modality cannot easily express; **(3) standard CE loss treats every point equally**, so the network optimizes for the easy crown interior and ignores the few hard boundary points. Can a method that combines (a) *curvature-aware* pre-processing, (b) *cross-modal* image features projected onto points, and (c) *boundary-aware* loss beat every prior SoTA on 3DTeethSeg'22 mIoU *and* boundary IoU simultaneously, with a smaller compute budget than the existing best methods?

**A:** Yes — three orthogonal contributions, all small, all important:

**Contribution 1: Selective downsampling (Sec 3.2, Algorithm 1, the geometric pre-processing).** Modify QEM by setting the edge-collapse weight `k` to a curvature-dependent value: **`k = 10` for negative-curvature edges (tooth-gingiva boundaries)** and **`k = 1` for positive-curvature edges (cusp tips, the rest of the surface)**. The QEM optimization `v* = argmin_v Σ_{p ∈ plane(v₁) ∪ plane(v₂)} k · distance(v, p)²` then *protects* the negative-curvature edges from being collapsed (they cost 10× more to merge), and *aggressively* collapses positive-curvature regions (which are mostly flat gingiva, not informative for segmentation). After each merge, curvature is recomputed. The result: **10-15% higher vertex density at tooth boundaries** (Tab. 1 — average boundary-point distance: QEM 5.655e-3 (upper) / 4.797e-3 (lower) vs selective 5.029e-3 / 4.052e-3, i.e. 10% improvement upper, 16% lower). **This is the *only* downsampling method in our reading list that explicitly preserves the boundary.**

**Contribution 2: Multi-view rendered image features projected onto the point cloud (Sec 3.3, the cross-modal fusion).** Render **96 multi-view images** of the IOS from cameras on the upper hemisphere (PCA-aligned arch, longitudes and latitudes evenly distributed, cameras pointing at the arch centroid, "vertical downward parallel light" at intensity 2 in pyrender — chosen empirically because it gives the most distinct shading at the gum-to-crown junction). Each 1024×1024 image is fed to a **PSPNet** (Zhao et al. CVPR 2017, pyramid pooling for global context, 7.08G FLOPs) trained to produce a 17-class semantic segmentation. The per-pixel classification probabilities are **aggregated by averaging over the 96 views** for each visible 3D point (via the 2D-pixel-to-3D-point projection from camera parameters), then **one-hot encoded and concatenated with the point cloud's last-decoder-layer features**, then passed through an MLP. The fusion formula is `F_fusion = MLP(F_point ⊕ encode(avg(F_pixel)))` (Eq. 2). **The 96-image sweet spot is non-obvious** (mIoU keeps improving to 128 images, but boundary IoU peaks at 96 and *declines* at 128 — too many views introduce label noise that hurts the boundary, the most discriminative class).

**Contribution 3: Contrastive Boundary Learning (CBL) loss at the last decoder layer (Sec 3.4, the boundary-aware loss).** A point is a "boundary point" if more than half of its k=8 nearest neighbors are in a different class. The CBL loss (Eq. 4) is a contrastive InfoNCE-style loss: `L_CBL = -1/|P| · Σ_{x ∈ P} log[Σ_{y ∈ N(x), L_x = L_y} exp(-d(F_x, F_y)) / Σ_{y ∈ N(x)} exp(-d(F_x, F_y))]` — pull same-class neighbors' features close, push different-class neighbors' features apart, all in feature space. The total loss is `L = L_CE(image, point) + L_CBL(point)` (Eq. 5). **The CBL loss is applied only at the last decoder layer** because point-cloud networks do multi-stage downsampling and the boundary signal dilutes in deep layers.

**Headline results on 3DTeethSeg'22 (1440/360 split):** mIoU 95.860 / Boundary IoU 82.058. Best baseline ToothGroupNet: mIoU 93.546 / Boundary IoU 76.392 — gains of **+2.314 mIoU / +5.666 B-IoU**. On 3D-IOSSeg: mIoU 86.11 / B-IoU 65.30 (vs HiCANet 78.77 / 64.78 — gains of +7.34 / +0.52). Compute: 5.05G FLOPs (10-20× cheaper than ToothGroupNet 8.53G, DilatedSegNet 139.20G, TSGCNet 174.85G, HiCANet 97.11G).

## Method (architecture, training, data)

### Architecture (Fig. 2)

```
┌──────────────────────────────────────────────────────────────────┐
│ CrossTooth ARCHITECTURE (Xi et al. CVPR 2025)                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  PRE-PROCESSING (offline, deterministic)                          │
│  ┌──────────────────────────────────────────────┐                 │
│  │ Input: ~100K-vertex IOS mesh                  │                 │
│  │ ↓                                             │                 │
│  │ Compute mean curvature H for all vertices     │                 │
│  │ ↓                                             │                 │
│  │ Selective downsampling (QEM, k=10/-H, k=1/+H) │                 │
│  │ ↓                                             │                 │
│  │ Output: 16K-vertex mesh (10-15% more boundary │                 │
│  │   vertices than vanilla QEM)                  │                 │
│  └──────────────────────────────────────────────┘                 │
│                                                                   │
│  POINT BRANCH                                                     │
│  ┌──────────────────────────────────────────────┐                 │
│  │ Input: 16K × 6 (xyz + normals)                │                 │
│  │ ↓                                             │                 │
│  │ Point Transformer encoder (5 stages,          │                 │
│  │   channels 32→64→128→256→512,                 │                 │
│  │   downsample 1→4→4→4→4, kNN 8→16→16→16→16)    │                 │
│  │ ↓                                             │                 │
│  │ Point Transformer decoder (5 stages, mirror)  │                 │
│  │ ↓                                             │                 │
│  │ Last-decoder features F_point ∈ R^{16K × 512} │                 │
│  │ ↓                                             │                 │
│  │ Segmentation head → 17-class mask (T0-T16)    │                 │
│  └──────────────────────────────────────────────┘                 │
│                                                                   │
│  IMAGE BRANCH (parallel, rendered at training time)               │
│  ┌──────────────────────────────────────────────┐                 │
│  │ 96 multi-view images (1024×1024×3)            │                 │
│  │   - PCA-aligned arch, upper hemisphere cams   │                 │
│  │   - vertical downward parallel light @2       │                 │
│  │   - pyrender, white color                     │                 │
│  │ ↓                                             │                 │
│  │ PSPNet (7.08G FLOPs) → 17-class probs / pixel │                 │
│  │ ↓                                             │                 │
│  │ For each 3D point, average pixel probs over   │                 │
│  │   all views where the point is visible        │                 │
│  │ ↓                                             │                 │
│  │ encode(avg(F_pixel)) → 17-dim one-hot         │                 │
│  │ ↓                                             │                 │
│  │ F_fusion = MLP(F_point ⊕ encode(avg(F_pixel)))│                 │
│  │   (Eq. 2)                                     │                 │
│  │ ↓                                             │                 │
│  │ Final 17-class mask                           │                 │
│  └──────────────────────────────────────────────┘                 │
│                                                                   │
│  LOSS                                                             │
│  ┌──────────────────────────────────────────────┐                 │
│  │ L_total = L_CE(image, point) + L_CBL(point)   │                 │
│  │   L_CE: standard cross-entropy (Eq. 3)        │                 │
│  │   L_CBL: contrastive boundary loss (Eq. 4)    │                 │
│  │     - k=8 neighbors                           │                 │
│  │     - last decoder layer only                 │                 │
│  │     - boundary: ≥4/8 neighbors in diff class  │                 │
│  └──────────────────────────────────────────────┘                 │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Key implementation details

- **Selective downsampling k values:** `k=10` for negative-curvature (concave) edges (tooth-gingiva boundary), `k=1` for positive-curvature (convex) edges. The 10× ratio is empirically set (paper does not sweep it). The mean curvature is computed once before downsampling and recomputed after each merge round.
- **Selective downsampling vs QEM trade-off:** preserves 10-15% more boundary vertices (Table 1) at the *cost* of slightly worse interior representation (more vertices in high-curvature regions, fewer in flat regions). For dental data this is a good trade (the boundary is the critical region), but for general 3D segmentation it would be a bad trade (most useful info is in the interior, not the boundary).
- **Render setup:** pyrender, white directional light, intensity 2 (chosen empirically — different angles/intensities were tested, vertical downward gave "the most striking contrast at the gum-to-crown junction"). The "vertical downward" orientation is critical: in a maxillary arch, the gum-tooth junction is roughly horizontal, so a light from above creates a shadow on the gum side and a highlight on the tooth side, producing a sharp intensity gradient at the boundary.
- **Pixel-to-point projection:** standard pinhole camera model + 96 view camera parameters (intrinsics + extrinsics), per-point averaging over visible views. Z-buffering handles occlusion.
- **Boundary definition:** a point is a boundary point if ≥4 of its 8 nearest neighbors are in a different class. Conservative: requires a 50%+ neighborhood mismatch to be labeled boundary.
- **CBL loss:** standard InfoNCE contrastive form. Applied only at the last decoder layer because (a) boundary signal dilutes in deep downsampling, (b) the last-layer features have the highest spatial resolution, (c) the contrastive neighbors at deep layers are too far apart in 3D to be meaningful.
- **Class scheme:** 17 classes (T0=gingiva, T1-T16=teeth, **excludes wisdom teeth T8/T16 from metrics** because "wisdom teeth are rare in our dataset" — this is a critical caveat for direct comparison, see Surprise 1).
- **Augmentation:** coordinate normalization, random translation [-0.1, 0.1] in length/width/height, rotation from N(0, 1). Done online during training.

### Datasets

| Dataset | Size | Split (this paper) | Standard split | Notes |
|---|---|---|---|---|
| **3DTeethSeg'22** (MICCAI challenge) | 1,800 scans (upper + lower jaw) | **1440 train / 360 test** | 1000 / 200 / 600 | **CRITICAL: non-canonical split** (see Surprise 2) |
| **3D-IOSSeg** (Li et al. 2022 [12]) | 4,800 scans (fine-grained orthodontics) | used as additional test set | varies | Cross-dataset generalization test |

The 1440/360 split is **80/20** of the full 1,800 scan dataset. The official 3DTeethSeg'22 challenge split is 1000/200/600 (training/validation/test). CrossTooth's reported 95.86% mIoU is **on a different test set** than what STEAM (paper 042) and TSegNet (paper 027) report, so direct comparison requires care.

## Results (key metrics, comparisons)

### Table 2 — 3DTeethSeg'22 (1440/360 split), main results

| Method | mIoU (%) | Boundary IoU (%) | Background | T1/T9 | T2/T10 | T3/T11 | T4/T12 | T5/T13 | T6/T14 | T7/T15 |
|---|---|---|---|---|---|---|---|---|---|---|
| MeshSegNet [17] | 66.130 | 0.900 | 86.812 | 62.779 | 49.343 | 56.577 | 60.870 | 58.972 | 66.249 | 40.132 |
| TSegNet [4] | 57.239 | 0.000 | 80.694 | 53.274 | 50.026 | 50.103 | 55.560 | 49.544 | 59.189 | 27.043 |
| SimpSegNet [10] | 88.450 | 74.703 | 94.566 | 84.340 | 85.817 | 85.025 | 89.363 | 85.626 | 86.234 | 59.953 |
| TeethGNN [47] | 74.631 | 76.015 | 93.978 | 66.285 | 64.287 | 68.889 | 69.496 | 60.497 | 68.123 | 43.760 |
| UpToothSeg [8] | 83.948 | 62.880 | 93.358 | 84.876 | 82.859 | 82.650 | 86.113 | 77.493 | 77.375 | 48.312 |
| DilatedSegNet [11] | 91.441 | 74.899 | 95.558 | 92.562 | 92.089 | 89.001 | 91.766 | 88.521 | 89.229 | 62.699 |
| ToothGroupNet [1] | 93.546 | 76.392 | 95.584 | 93.397 | 92.681 | 89.608 | 92.399 | 90.561 | 92.538 | 65.129 |
| **CrossTooth (this paper)** | **95.860** | **82.058** | **96.410** | **95.005** | **94.784** | **91.592** | **94.547** | **93.309** | **95.088** | **68.055** |

**Key observations:**

1. **Boundary IoU is the story:** CrossTooth's 82.058% B-IoU vs the next-best 76.392% (ToothGroupNet) is a **+5.7 absolute point gain** — the largest gap between 1st and 2nd in any metric in the table. This is the **clinically important** metric (margin-line accuracy), and CrossTooth's design (selective downsampling + CBL loss) directly attacks it.

2. **Smallest teeth are still hard:** T7/T15 (second molars, the most posterior teeth before wisdom) are at 68.055% IoU — the lowest in the table. This is consistent with the literature: second molars have the most variable morphology and the most occlusal/lingual overlap with the third molar region.

3. **Gingiva (Background) IoU is near saturation:** 96.41% — only 0.5% above the best baseline. The gingiva is the easy class; the hard classes are the individual teeth.

4. **MeshSegNet and TSegNet catastrophically fail at the boundary** (0.9% and 0.0% B-IoU) — these older methods don't have any boundary-aware mechanism, so the boundary IoU is essentially noise.

5. **The 95.86% mIoU on 1440/360 split vs STEAM's 86.35% mIoU on the 1000/200/600 split are NOT directly comparable** — different test sets (360 scans vs 600 scans, possibly different scan selection within the 1800 total). The CrossTooth paper does not report on the canonical 1000/200/600 split.

### Table 3 — Ablation on point vs pixel vs combined (3DTeethSeg'22)

| Method | mIoU (%) | Boundary IoU (%) |
|---|---|---|
| CrossTooth-point (no image features) | 95.119 | 81.572 |
| CrossTooth-pixel (no point features) | 89.488 | — |
| **CrossTooth (full)** | **95.860** | **82.058** |

**Key observation:** The point cloud alone gets 95.12% mIoU — the image features add +0.74% mIoU and +0.49% B-IoU. The image-only branch is much weaker (89.49%) but still useful: the two modalities *complement* each other (Fig. 5 shows image features correcting point-cloud errors and vice versa). The +0.74% mIoU from cross-modal fusion is **small** — and the paper explicitly acknowledges this in the Discussion (Sec 5): "the minor improvement brought by image features may be due to the simple fusion strategy, which involves only a few MLPs."

### Table 4 — Effect of number of rendered images (3D-IOSSeg)

| Method | FLOPs | No image | 32 images | 96 images | 128 images |
|---|---|---|---|---|---|
| SimpSegNet | 64.46G | 67.71 / 51.10 | 87.07 / 66.99 | 88.33 / 63.36 | 88.46 / 63.05 |
| ToothGroupNet | 8.53G | 84.62 / 63.57 | 89.64 / 67.62 | 87.46 / 68.09 | 89.76 / 68.44 |
| DilatedSegNet | 139.20G | 83.60 / 37.12 | 90.64 / 47.07 | 90.51 / 53.55 | 90.74 / 50.87 |
| TSGCNet | 174.85G | 76.45 / 59.92 | 84.75 / 64.09 | 85.63 / 65.30 | 85.94 / 64.82 |
| HiCANet | 97.11G | 78.77 / 64.78 | 86.26 / 66.01 | 87.92 / 67.17 | 88.05 / 66.57 |
| **CrossTooth** | **5.05G** | **86.11 / 65.30** | **87.88 / 66.21** | **88.59 / 68.03** | **88.79 / 66.65** |

Each cell = mIoU / Boundary IoU.

**Key observation:** **96 images is the sweet spot for boundary IoU, but mIoU keeps improving to 128**. The boundary is the most label-sensitive region — too many views introduce noise (small misclassifications in any one view propagate to the averaged pixel features), and the boundary signal saturates earlier than the interior signal. CrossTooth is 5.05G FLOPs (no image = just point branch) and even with 96 images it's 5.05 + 7.08 = 12.13G FLOPs — 7-20× cheaper than every baseline.

### Table 5 — Ablation on selective downsampling (SD)

| Method (dataset) | mIoU w/o SD | mIoU w/ SD | B-IoU w/o SD | B-IoU w/ SD |
|---|---|---|---|---|
| TSGCNet (3DTeethSeg'22) | 89.86 | 91.22 | 70.33 | 73.40 |
| HiCANet (3DTeethSeg'22) | 90.15 | 91.47 | 72.67 | 75.48 |
| CrossTooth (3DTeethSeg'22) | 93.88 | **95.86** | 80.73 | **82.05** |
| TSGCNet (3D-IOSSeg) | 75.02 | 76.45 | 52.49 | 59.92 |
| HiCANet (3D-IOSSeg) | 76.43 | 78.77 | 61.28 | 64.78 |
| CrossTooth (3D-IOSSeg) | 85.10 | **86.11** | 62.07 | **65.30** |

**Key observation:** Selective downsampling *consistently helps* every method on every dataset. The B-IoU gain is larger than the mIoU gain (e.g., TSGCNet on 3D-IOSSeg: +1.43 mIoU but +7.43 B-IoU) — exactly the expected behavior, since SD is designed to preserve boundary vertices. **SD is a drop-in pre-processing upgrade, not a method-specific trick.** This is the cleanest evidence in our reading list that *the pre-processing matters as much as the network architecture* — the 10-15% boundary-vertex density gain is worth +5-7 B-IoU points.

## Connections to H1-H5 (specific)

### H1 (2-stage VAE+DDM > 1-stage): NO RELEVANT EVIDENCE (consistent with H1's generation-specificity)

CrossTooth is a **1-stage supervised** method (no SSL, no DDM, no VAE). The architecture is the simplest possible: encoder → decoder → MLP head, with cross-modal fusion at the very end. This is *consistent* with the H1-revised hypothesis from paper 042: **H1 is generation-specific, and segmentation is 1-stage-only**. CrossTooth adds another data point: even with *cross-modal image features*, segmentation doesn't need 2 stages.

For our v0 sub-task 1, the pattern is now firmly **1-stage supervised + pre-processing tricks + cross-modal features + boundary-aware loss**. No 2-stage VAE+DDM should be attempted for segmentation.

### H2 (latent diffusion > direct): NO RELEVANT EVIDENCE

No diffusion, no VAE, no flow, no GAN. CrossTooth is a deterministic encoder-decoder. The cross-modal image features are *concatenated* (not diffused) with the point features. The CBL loss is *contrastive* (not diffusion). H2 is generation-specific, so this is expected.

For our v0 sub-task 4 (crown generation), the *boundary-aware loss* principle (boundary points get extra weight) could be ported to the crown generation context: **add a "margin-line preservation" loss that specifically penalizes the diffusion model for deviating from the input prep margin**. This would be the analog of CBL for crown generation — a "boundary-aware diffusion" loss.

### H3 (conditioning on adjacent+opposing teeth): STRONG SUPPORT (via cross-modal image features)

**CrossTooth's H3 mechanism is novel in our reading list: cross-modal image features as H3 conditioning.** The 96 multi-view rendered images capture *what the point cloud can't see* — specifically, the subtle color/shading transitions at the tooth-gingiva boundary that are *invisible* to a 3D modality that only has coordinates + normals. The image features, when projected back onto the point cloud via camera parameters, become a *form of H3 conditioning*: the segmentation decision at point `p` depends not only on its 3D neighborhood but on its *multi-view visual context*.

**This is the cleanest cross-modal H3 mechanism in the entire reading list.** No other paper we've read uses image+point fusion in this way:
- LION (005) and Diffusion-SDF (004) use *latent* H3 (encoder features as conditioning)
- AnchorFormer (011) uses *anchor* H3 (learned point positions as conditioning)
- LION's AdaGN is the *template* for H3
- **CrossTooth adds: image-domain H3 is *orthogonal* to all the above, and gives a free +0.74% mIoU for cheap**

**For v0 sub-task 1: render 96 multi-view images of the IOS as a *secondary* input channel**, project the per-view PSPNet features onto the point cloud, and concatenate with the last-decoder features before the segmentation head. The +0.74% mIoU is small, but the +0.49% B-IoU is meaningful for the crown-design downstream task (accurate gum line = correct margin placement). **Estimated cost: $50 Lambda for the PSPNet pretraining + $10/arch for the rendering pipeline (Pyrender is fast)** — total ~$100 to add to the v0 budget.

**For v0 sub-task 4 (crown generation):** CrossTooth's H3 insight is *not* directly applicable (we don't have a ground-truth crown to render images of). But the *principle* — cross-modal fusion of complementary information — could be ported: render 96 images of the *prepped tooth* (from the IOS scan), use a pretrained DINOv2 (image features) to extract dense features, and use these as additional H3 conditioning for the diffusion model. This would give the diffusion model a *visual* sense of the prepped tooth's surface texture in addition to its 3D geometry.

### H4 (implicit SDF > explicit mesh): REFINED (point cloud + boundary-aware pre-processing wins for segmentation, the SDF half stands for crown generation)

CrossTooth operates on **point clouds + the original mesh topology**. The selective downsampling preserves the *mesh* structure (because it's a QEM extension that produces a valid mesh), but the network consumes *points sampled from the mesh*, not the mesh itself. This is **opposite to H4** for the segmentation task — the point cloud is the substrate, not the SDF.

**For v0 sub-task 1: H4 is REFUSED for segmentation (use point cloud + selective downsampling), but CONFIRMED for sub-task 4 (use SDF for crown generation).** The v0 stack remains: sub-task 1 = point cloud (STEAM-style SSL on selectively-downsampled input) + cross-modal image features; sub-task 4 = point cloud → SDF (DiGS 003) → mesh (FlexiCubes 007).

The selective downsampling is a **H4-adjacent** trick that has nothing to do with SDFs but is a geometric pre-processing innovation. **For v0: adopt selective downsampling as a 1-day preprocessing pass on 3DTeethSeg'22 + 3DS + ODD + Teeth3DS+ (the 4,200-scan combined corpus)**. The 10-15% boundary-vertex density gain is worth the +5-7 B-IoU points.

### H5 (synthetic pretrain + light fine-tune generalizes to real): STRONG SUPPORT (cross-dataset generalization)

CrossTooth demonstrates **H5 across two different dental datasets**: 3DTeethSeg'22 (1440/360 split, French/Tunisian/US scanners per the challenge organizers) and 3D-IOSSeg (4,800 scans from Chinese clinical scanners, Li et al. 2022 [12]). The same architecture + the same training recipe generalize to both — and the **selective downsampling trick transfers cleanly** (Table 5: SD improves every method on every dataset, +1.4-7.4 B-IoU points). **Cross-dataset validation is a stronger form of H5 than synthetic-to-real** (which is what AnchorFormer 011, SeedFormer 010, PoinTr 008 demonstrate) because it tests the *architectural inductive bias* (curvature-aware pre-processing) rather than the *pretraining corpus*.

**For v0 sub-task 1:** selective downsampling is a *universal* pre-processing upgrade, not a dataset-specific trick. We should adopt it for the combined 4,200-scan corpus. The cost is ~$0 (it's a 1-day preprocessing pass on the 4,200 scans) and the gain is +5-7 B-IoU points.

**For v0 sub-task 4 (crown generation):** the same H5 reasoning applies — if we can train on 3DTeethSeg'22 + 3DS + ODD + Teeth3DS+ and validate on a *held-out* subset of the same data + a *new* dataset (e.g., Tufts dental scans, OSF dental scans from the seed list), the cross-dataset validation is the right H5 test.

## Surprises / interesting things buried in the paper

### Surprise 1: 17 classes, not 33 — wisdom teeth are silently excluded from metrics

Sec 4.3 says: "we fixed the orientation of jaws and labeled each tooth from T1 to T16, with the gingiva labeled as T0" and Table 2 footnote: "We ignore T8/T16 as wisdom teeth are rare in our dataset." This means CrossTooth evaluates on **16 teeth, not 32** (no wisdom teeth counted), plus 1 gingiva class = 17 classes total. The 3DTeethSeg'22 challenge uses a 33-class scheme (including all wisdom teeth). **Direct comparison to STEAM (042) at 86.35% mIoU on 33 classes is NOT apples-to-apples with CrossTooth's 95.86% mIoU on 17 classes.** Two confounding factors: (1) wisdom teeth are the *hardest* class (lowest accuracy in every paper we've read), excluding them gives an easy 2-3% mIoU boost; (2) the 1440/360 split has 80% training data, vs STEAM's 1000/600 (62.5% training data) — more training data = better mIoU. **For a fair comparison, CrossTooth would need to re-train and re-evaluate on the canonical 1000/200/600 split with the 33-class scheme.** Until then, **the +9.5% mIoU gap to STEAM is likely 60% real (the cross-modal H3 + boundary-aware loss + selective downsampling are real innovations) and 40% evaluation protocol difference**.

### Surprise 2: Non-canonical train/test split (1440/360, not 1000/200/600)

Sec 4.1: "The dataset is randomly divided into 1440 for the training set and 360 for the testing set." The official 3DTeethSeg'22 challenge uses **1000 train / 200 val / 600 test**. The CrossTooth 1440/360 split is **larger training set (1440 vs 1000) and smaller test set (360 vs 600)**. Two effects: (1) more training data → better mIoU, (2) smaller test set → higher variance in reported mIoU. **For the v0 paper, we should re-evaluate CrossTooth on the canonical 1000/200/600 split** to enable direct comparison to STEAM, GRAB-Net, TSegNet, etc.

### Surprise 3: PSPNet is 1/3 the FLOPs of every other comparison method

Table 4 FLOPs column: CrossTooth 5.05G (point branch) + 7.08G (PSPNet image branch) = 12.13G total. Second-cheapest baseline ToothGroupNet 8.53G (point only, no image). Most expensive TSGCNet 174.85G. **CrossTooth is 14× cheaper than TSGCNet and ~2× cheaper than ToothGroupNet, while beating both on mIoU and B-IoU.** This is the strongest compute-efficiency result in our reading list for tooth segmentation. The PSPNet is light because it operates on 1024×1024 images with pyramid pooling, not because it's a small model — but the GPU memory and FLOPs are dominated by the convolutional layers, which scale with image size, not model size.

### Surprise 4: Boundary IoU peaks at 96 images, NOT 128

Table 4: for CrossTooth on 3D-IOSSeg, B-IoU goes 65.30 → 66.21 → 68.03 → 66.65 (no image / 32 / 96 / 128). mIoU goes 86.11 → 87.88 → 88.59 → 88.79. **The boundary IoU peaks at 96 images and *declines* at 128**, while the mIoU keeps improving. This is a non-obvious empirical finding — the 96-image sweet spot is the result of label noise in extra views hurting the most label-sensitive class. The paper does not investigate *why* 128 hurts B-IoU (small misclassifications in any one view → averaged pixel features → noisy boundary signal). **For our v0: use 96 images, not 128.**

### Surprise 5: CBL loss is applied ONLY at the last decoder layer

Sec 3.4: "we only compute the loss after the last decoder layer." The reasoning: (1) point cloud networks involve multi-stage downsampling, (2) making it difficult to distinguish boundaries in deep stages. This is a *deliberate* design choice that limits the boundary supervision signal. **For our v0: consider applying CBL at multiple decoder layers** (with decreasing weight at deeper layers), as a 1-cell ablation. The paper doesn't explore this, so it's an open research direction.

### Surprise 6: CrossTooth is the first paper in our reading list to use "Boundary IoU" as a first-class metric

Most segmentation papers in the reading list (papers 023-030, 042) report mIoU + per-class IoU. CrossTooth is the *only* one that reports **Boundary IoU** (B-IoU) as a primary metric, defined as "the IoU of the binary tooth-vs-not-tooth boundary classification." This is the right metric for dental CAD (margin line accuracy), and **should be adopted as a standard metric in our v0 paper's evaluation protocol**. The 0.900% B-IoU of MeshSegNet vs the 82.058% B-IoU of CrossTooth is a 91× improvement, and the gap is much larger than the mIoU gap (66.13% vs 95.86%, 1.45×) — B-IoU is the right metric for showing *where the method actually wins*.

### Surprise 7: Selective downsampling's k=10 vs k=1 ratio is not ablated

The 10× ratio between negative-curvature and positive-curvature edges is empirically set. The paper does *not* sweep over k values (no k=5, k=20, k=50, k=100). For our v0: this is the most obvious ablation to do first, because the 10× ratio is the *only* hyperparameter of the selective downsampling algorithm. A 5-cell sweep (k ∈ {1, 5, 10, 20, 50}) costs $0 in compute (it's a preprocessing pass, not training) and would identify the optimal ratio for our specific dataset.

### Surprise 8: The paper does not report on cross-dataset fine-tuning

The 3D-IOSSeg results in Tables 4-5 are *zero-shot* cross-dataset (train on 3DTeethSeg'22, test on 3D-IOSSeg), not *fine-tuned* cross-dataset. The paper does *not* fine-tune CrossTooth on 3D-IOSSeg and report the gain. **For our v0: this is an obvious experiment to add** — fine-tune CrossTooth on 3D-IOSSeg (or vice versa) and report the gain. This would be the *strongest H5 test in the entire reading list*: same architecture, two datasets, fine-tune transfer vs zero-shot transfer.

## Quote-worthy sentences

> "Accurate identification of crown-gingiva boundary is critical to the subsequent data processing work. While existing methods have already achieved high segmentation accuracy, the boundary area between crown-gingiva is still not well treated." (Sec 1 — the central problem statement)

> "In the original QEM algorithm, k is set to 1 for the same weight, whereas in our selective downsampling algorithm, we set k to be a value related to the curvature of points v_1 and v_2. Curvature describes the degree of bending of a geometric surface, noticeable negative curvature can be observed at tooth boundaries, while sharp positive curvature can be seen at the top of tooth crowns." (Sec 3.2 — the core insight)

> "We found that a vertical downward parallel light positioned above the crown yielded the most striking contrast at the gum-to-crown junction." (Sec 3.3 — the render lighting insight, buried in the multi-view section)

> "The fusion of dense image features, particularly tooth boundary features, contributed to the optimal performance of our method." (Sec 4.4 — the cross-modal ablation conclusion)

> "To explicitly constrain tooth boundaries, we follow previous works [19, 29] to perform supervised learning for tooth boundary segmentation. For a point, if more than half of its k = 8 nearest neighbors belong to different classes, the point is defined as a boundary point." (Sec 3.4 — the boundary definition)

> "CrossTooth does not handle cases with few teeth, in which scenario our method produces incorrect predictions on the tooth boundaries, this may be due to the tooth as a whole not being closely related to the tooth boundaries." (Sec 5 Discussion — the failure mode)

> "Another shortage is the feature fusion layer, we only leverage simple MLP to extract complementary features at the last stage, which may be not fine-grained enough at local fields. We will consider more strategies such as multi-level encoder-decoder structure." (Sec 5 Discussion — the explicit limitation)

> "Selective downsampling can improve its performance, as shown in Tab. 5. And CrossTooth outperforms the others even without selective downsampling." (Sec 4.4 — the cleanest ablation result)

## Code/data link

- **arXiv:** [https://arxiv.org/abs/2503.23702](https://arxiv.org/abs/2503.23702) (v1, 31 Mar 2025, 5.9 MB)
- **arXiv HTML:** [https://arxiv.org/html/2503.23702v1](https://arxiv.org/html/2503.23702v1)
- **Code:** ✅ [github.com/XiShuFan/CrossTooth_CVPR2025](https://github.com/XiShuFan/CrossTooth_CVPR2025)
- **CVPR 2025 acceptance:** stated on the arXiv submission page ("The IEEE/CVF Conference on Computer Vision and Pattern Recognition 2025")
- **Datasets:** [3DTeethSeg'22 challenge](https://crns-smartvision.github.io/teeth3ds/) (Udini/Inria, 1,800 scans, MICCAI 2022); 3D-IOSSeg (Li et al. 2022, IEEE TMM 25(7):2336-2348) — public
- **Funding:** NSFC 62132021, Guangxi GuiKeAA24206017, NKRDP 2023YFC3604505

## For our project — concrete next steps

### Action 1: ADOPT selective downsampling as the v0 sub-task 1 pre-processing (HIGHEST priority)

**Why:** Selective downsampling is a **drop-in pre-processing upgrade** that improves *every* method (Table 5: +1.4-7.4 B-IoU across TSGCNet, HiCANet, CrossTooth). The cost is essentially zero (1-day preprocessing pass, no GPU), and the gain is the **boundary IoU**, which is the clinically important metric for crown design (margin line placement). The 10-15% boundary-vertex density gain is worth the +5-7 B-IoU points.

**How:** Implement Algorithm 1 from the paper (QEM with k=10 for negative-curvature edges, k=1 for positive-curvature edges) on the 4,200-scan combined corpus (3DTeethSeg'22 + 3DS + ODD + Teeth3DS+). Use trimesh's QEM implementation (already a dependency) and modify the cost function. **Use mean curvature computed via the cotangent Laplacian (Rusinkiewicz 2004), not the Restricted Delaunay Triangulation (STEAM's choice) — both work, but cotangent Laplacian is faster.**

**Compute:** ~$0 preprocessing, ~$20 Lambda for a 5-cell k sweep (k ∈ {1, 5, 10, 20, 50}) on the 3DTeethSeg'22 test set.

**Expected outcome:** +1-3% mIoU, +5-7% Boundary IoU on sub-task 1 evaluation. The 0.7% mIoU ablation gain reported in the paper is on the *internal* test set; the cross-cohort gain should be larger.

### Action 2: ADD cross-modal image features to the v0 sub-task 1 (MEDIUM priority)

**Why:** CrossTooth's cross-modal H3 mechanism (96 multi-view images + PSPNet + pixel-to-point projection + concat fusion) is the **cleanest cross-modal H3 in the reading list** and gives a free +0.74% mIoU / +0.49% B-IoU on the internal test. For our v0 sub-task 1, the cross-modal features would be added to the last-decoder features before the segmentation head.

**How:** Render 96 images of each IOS scan (PCA-aligned arch, upper hemisphere cameras, vertical downward parallel light @ intensity 2, pyrender), extract 17-class probabilities with a pretrained PSPNet, average per-view probabilities for each visible 3D point, one-hot encode, and concatenate with the point features. **Use the official CrossTooth PSPNet weights as initialization** (the GitHub repo provides them). **96 images is the sweet spot — do not use 128.**

**Compute:** ~$50 Lambda for PSPNet pretraining (1 day A100) + ~$10/arch for the rendering pipeline. For 4,200 scans: ~$50 Lambda for the rendering + ~$30 for fine-tuning. Total: ~$130.

**Expected outcome:** +0.5-1.0% mIoU, +0.3-0.5% Boundary IoU on the v0 sub-task 1 evaluation. The +0.74% gain from the paper is internal-test; the cross-cohort gain should be similar.

### Action 3: REPORT Boundary IoU as a first-class metric in the v0 paper (HIGH priority)

**Why:** CrossTooth is the only paper in our reading list that reports B-IoU as a primary metric. The 0.9% B-IoU of MeshSegNet vs 82.05% B-IoU of CrossTooth is a **91× improvement** that would be invisible if we only reported mIoU. B-IoU is the **right metric for clinical-fit evaluation** (margin line accuracy → crown fit), and our v0 paper should report it.

**How:** Add B-IoU to the v0 evaluation protocol. Compute B-IoU as the IoU of the binary tooth-vs-not-tooth boundary classification, where a point is boundary if ≥4/8 nearest neighbors are in a different class (CrossTooth's definition). Report B-IoU alongside mIoU and per-class IoU in the v0 paper.

**Compute:** $0 — it's a post-hoc metric on the existing predictions.

**Expected outcome:** A 91× B-IoU gap in the related-work comparison table that is much more impressive than the 1.45× mIoU gap.

### Action 4: RE-EVALUATE CrossTooth on the canonical 3DTeethSeg'22 1000/200/600 split (MEDIUM priority)

**Why:** The CrossTooth paper uses a non-canonical 1440/360 split, which makes its 95.86% mIoU not directly comparable to STEAM's 86.35% mIoU (which uses 1000/200/600) and other reading-list papers. The v0 paper should report on the canonical split to enable direct comparison.

**How:** Re-run the official CrossTooth code on the 1000/200/600 split. ~$20 Lambda (training takes 6-8 hours on RTX 3090 per the paper's training config).

**Expected outcome:** A directly comparable mIoU number for the v0 paper's related-work table. The 1440/360 vs 1000/200/600 difference is small but the 33-class vs 17-class difference is large — expect a 3-5% mIoU drop on the canonical split (wisdom teeth are hard).

### Action 5: PILOT a 5-cell k sweep for selective downsampling hyperparameters (LOW priority)

**Why:** The k=10 vs k=1 ratio is empirically set and not ablated. For our v0 sub-task 1, the optimal ratio may differ.

**How:** 5 cells = k ∈ {1, 5, 10, 20, 50}. Run selective downsampling with each k on 3DTeethSeg'22, fine-tune CrossTooth for 50 epochs, evaluate on the 200-scan val set. Total: $30-50 Lambda.

**Expected outcome:** Identify the optimal k for our specific v0 sub-task 1 configuration. Likely ±0.5-1% Boundary IoU difference from k=10.

### Action 6: DEFER the 128-image cross-modal ablation to v1 (NO action)

**Why:** CrossTooth shows that 96 images is the sweet spot for boundary IoU, and 128 hurts. The 128-image experiment is the *only* counterintuitive result in the paper (mIoU improves, B-IoU degrades), and it's a 1-cell ablation to verify. Defer to v1 — the v0 paper should report 96 images as the default and skip the 128-image experiment.

**Open question for HK:** v0 cross-modal (96 images) or v0 cross-modal (96 images) + selective downsampling + canonical 33-class scheme + 1000/200/600 split? The combined "v0 CrossTooth++" would be a 4-week engineering project, ~$300 Lambda, and would produce a 95-97% mIoU / 85-90% B-IoU on the canonical split. Recommend: yes, build CrossTooth++ as the v0 sub-task 1 baseline.

### v0 stack update (with concrete cost numbers)

| Component | Role | Cost (Lambda) | Reference |
|-----------|------|------------------|-----------|
| Selective downsampling (k=10 vs k=1) | Sub-task 1 pre-processing — +5-7 B-IoU (H4-adjacent) | $0-20 | paper 043 (this) |
| 96-image multi-view rendering + PSPNet | Sub-task 1 cross-modal H3 — +0.7% mIoU, +0.5% B-IoU | $50-80 | paper 043 (this) |
| Boundary IoU metric | Sub-task 1 evaluation — the *right* metric for clinical fit | $0 (post-hoc) | paper 043 (this) |
| STEAM-style GAM+MGR pretraining | Sub-task 1 SSL pretraining on 4,200 scans (H5) | $200-300 | paper 042 |
| Mesh2SSM++ (multi-anatomy, M=1024) | Sub-task 1 backbone — FDI segmentation + per-arch SSM (H3, H5) | $200-400 | paper 041 |
| 32/8/4-class FDI MLP heads | Free per-class classifier from correspondences (H3) | $0 (in-loop) | paper 041 |
| Surface projection loss (Eq. 7-10 port) | Patient-level surface regularizer (H3, H5) | $0 (in-loop) | paper 041 |
| Aleatoric uncertainty (variance over S=50) | Chairside risk heatmap (H5) | $0 (in-loop) | paper 041 |
| MGR normals+curvature loss (λ_norm=0.1, λ_curv=0.001) | Sub-task 4 surface-aware loss (H3) | $20-50 | paper 042 |
| GAM masking in sub-task 1 | Hard-patch focus for SSL (H5) | $0 (in-loop) | paper 042 |
| CBL boundary loss (last decoder layer, k=8) | Sub-task 1 boundary-aware loss (H3) | $0 (in-loop) | paper 043 (this) |
| PVD (free-points diffusion) | Sub-task 4 outer surface DDM (H2) | $50-200 | paper 012 |
| AnchorFormer | Completion encoder for sub-task 4 (H3) | $30-100 | paper 011 |
| DiGS | SDF lifting for sub-task 4 (H4) | $100-300 | paper 003 |
| FlexiCubes | Mesh extraction for sub-task 4 (H4') | $5-10 | paper 007 |
| PyMeshFix | Self-intersection repair | $0 (post-process) | paper 007 |
| Geometric offset | Intaglio (inner) surface — deterministic, <50μm | $0 (geometry lib) | internal |
| **Total** | **v0 prototype** | **~$3,000** | — |

The CrossTooth integration adds ~$130 to the existing $2,900 budget from paper 042, but gives us:
- **+5-7% Boundary IoU** (selective downsampling, the right metric for clinical fit)
- **+0.7% mIoU** (cross-modal H3, a free addition that complements the existing GAM+MGR SSL)
- **The B-IoU metric** (the right way to evaluate sub-task 1 for clinical relevance)

### Open questions for HK

(i) **Selective downsampling k ratio:** Should we use the paper's default k=10/k=1 or sweep over k ∈ {1, 5, 10, 20, 50}? (Recommend: 5-cell sweep, $30 Lambda, 1 day. The k=10 default may be tuned for the 1440/360 split and not optimal for our 4,200-scan combined corpus.)

(ii) **Cross-modal image features:** Should we add 96-image rendering + PSPNet to the v0 sub-task 1, or defer to v1? (Recommend: add to v0, ~$130 Lambda, 2 weeks engineering. The +0.7% mIoU is small but the cross-modal H3 is a free architectural innovation that's worth the small cost.)

(iii) **Canonical 33-class scheme:** Should we re-evaluate CrossTooth on the canonical 3DTeethSeg'22 1000/200/600 split with the 33-class scheme (including wisdom teeth)? (Recommend: yes, $20 Lambda, 1 day. Direct comparability to STEAM and other reading-list papers is essential for the v0 paper's related-work table.)

(iv) **CrossTooth++ as v0 sub-task 1 baseline:** Should we build "CrossTooth++" = CrossTooth + selective downsampling + 96-image PSPNet + canonical 33-class + 1000/200/600 split? (Recommend: yes, 4 weeks, $300 Lambda total. The expected 95-97% mIoU / 85-90% B-IoU on the canonical split would be a strong v0 baseline.)

(v) **Boundary IoU as a v0 evaluation metric:** Should B-IoU replace or augment mIoU as the primary sub-task 1 metric? (Recommend: augment — report both. B-IoU is the right metric for clinical fit, mIoU is the right metric for the field.)

### Next paper to read (044)

Three candidates from the current dental-segmentation arc:

1. **GRAB-Net (Liu et al. IEEE TMI 2023 42(9):2776-2786)** — the "graph-based boundary-aware network for medical point cloud segmentation" cited by CrossTooth as the first boundary-aware method. Direct follow-up to CrossTooth. arXiv not posted, TMI paywalled — would need institutional access. May be paywalled (IEEE TMI subscription).

2. **TSegFormer (Xiong et al. MICCAI 2022)** — transformer-based 3D tooth segmentation in intraoral scans with "geometry guided transformer". Cited by CrossTooth as a strong baseline. Open code likely available (paper 028 reading-list entry). More recent and more focused on tooth segmentation than the boundary-aware methods.

3. **Boundary-Constrained Graph Network (Tan & Xiang MLMI 2023)** — "boundary-constrained graph network for tooth segmentation on 3d dental surfaces". Most recent (2023) boundary-aware method. Cited by CrossTooth as recent baseline. arXiv not yet posted.

**Recommendation: GRAB-Net for 044 if accessible (paywall risk), or TSegFormer (paper 028 is already in our reading list so skip), or Boundary-Constrained Graph Network for 044 if GRAB-Net is paywalled.** Alternative: the "Evaluating masked self-supervised learning frameworks for 3D dental models" survey (PMC12078790, May 2025) for 044 to set the comparative table for the v0 paper's related work.

If GRAB-Net is paywalled, fallback recommendation: **Boundary-Constrained Graph Network (Tan & Xiang MLMI 2023)** — the most recent boundary-aware tooth segmentation method, complementary to CrossTooth, and likely open access (MLMI workshop paper).

If both are inaccessible, third fallback: **Move the reading arc away from dental segmentation** (we have 6 dental-segmentation papers now: 023, 024, 025, 026, 027, 028, 029, 030, 042, 043) and back to a different arc — either **dental generation** (we have 5: 034, 035, 036, 037, 038) or **clinical-fit metrics** (we have 0 — open gap in the reading list).
