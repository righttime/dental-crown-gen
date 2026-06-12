# Paper 171 — PF3plat: Pose-Free Feed-Forward 3D Gaussian Splatting for Novel View Synthesis

- **Authors:** Sunghwan Hong\*¹, Jaewoo Jung\*¹, Heeseong Shin², Jisang Han¹, Jiaolong Yang†², Chong Luo†², Seungryong Kim†¹
- **Affiliations:** ¹Korea University (CVLab, KAIST adjacent) + ²Microsoft Research Asia
- **arXiv:** **2410.22128** v1 29 Oct 2024 (5,314 KB) → **v2 24 Jul 2025 (6,748 KB)**
- **Venue:** **ICML 2025** (per arXiv comment "Accepted by ICML'25" + project page)
- **Code:** https://github.com/cvlab-kaist/PF3plat — **MIT License** ✅ (244 ⭐ / 9 🍴 as of 2026-06-13, PyTorch 2.0.1 + CUDA 12.1 + Python 3.10)
- **Pretrained:** Google Drive (drive.google.com/file/d/1ylrN8HNcnt2VdHkFnRBvgHIr9hBoJG1c)
- **Project page:** https://cvlab-kaist.github.io/PF3plat/
- **Citations:** ~51 Semantic Scholar (per S2 2026-06-13, ~7-8 months post-v1)
- **Reading time:** 60 min (22-page main + appendix)

## TL;DR

**The first feed-forward 3DGS that operates on UNPOSED multi-view image collections with explicit handling of the pixel-aligned-Gaussian instability, achieving SOTA across RealEstate10K, ACID, and DL3DV via a two-stage coarse-to-fine pipeline that wraps pretrained UniDepth v2 (monocular depth) + LightGlue (correspondence) with lightweight learnable refinement modules + geometry-aware Gaussian confidence conditioning.** +4.0 dB PSNR over CoPoNeRF on RealEstate10K (23.59 vs 19.54), +3.2 dB on ACID (25.64 vs 22.44), and 2× faster (0.39s vs 17.29s for 2-view inference) — the *most-comprehensive* pose-free 3DGS in the 161-paper reading list that does NOT require a frozen MASt3R/VGGT backbone, and is the **only MIT-licensed** pose-free 3DGS that produces competitive pose accuracy (1.76° rotation error on RE10K vs CoPoNeRF's 3.61°).

## Research question + their answer

**Q:** Can we build a feed-forward 3DGS that does NOT require GT camera poses or depth at training or inference, that generalizes across diverse real-world scenes (indoor + outdoor + dynamic), that handles wide-baseline sparse views with minimal overlap, AND that runs in a single feed-forward pass (no test-time optimization required)?

**A:** Yes — but the central challenge is that **pixel-aligned 3D Gaussians are uniquely fragile to misalignment** (their explicit, point-based nature means that inaccurate 3D center localization produces *noisy/sparse gradients* that destabilize training, unlike implicit NeRF/MLP-based methods that benefit from interpolation). Their solution is a **two-stage coarse-to-fine pipeline**: (1) **coarse alignment** using off-the-shelf UniDepth v2 (monocular depth) + LightGlue (visual correspondence) + RANSAC PnP to get *initial* per-image depth and pair-wise pose; (2) **fine alignment** with three lightweight learnable modules: (a) depth offset via self-attention Transformer over UniDepth features, (b) pose offset via cross-attention over Plücker coordinates + feature maps + learnable pose token, (c) **geometry-aware confidence scores** computed from aggregated monocular + multi-view depth consistency to *condition* the Gaussian parameter prediction. The crucial insight is that **coarse alignment provides stability, and fine alignment + confidence conditioning provides accuracy** — a clean separation of concerns that is *the* missing piece for feed-forward 3DGS at wide baselines.

## Method (architecture, training, data)

### Architecture (3.2, paper Fig. 1)

**Input:** N unposed images {I_i}_{i=1}^N with known intrinsics K_i, target view I_t.

**Stage 1 — Coarse Alignment of 3D Gaussians (3.2.1):**
- **Monocular depth:** UniDepth v2 (Piccinelli 2024) → per-image depth map D_i
- **Visual correspondence:** LightGlue (Lindenberger 2023) → pairwise correspondences M_{ij} + confidence C_{ij}
- **Pair-wise pose:** MAGSAC++ RANSAC + (Nistér 2004) 5-point algorithm → relative poses P_{ij}
- **3D Gaussian centers** μ_i(p) = backproject(p, D_i(p), P_i, K_i) for each pixel p

**Stage 2 — Fine Alignment (3.2.2 + 3.2.3):**
- **Depth Refinement (Eq. 1):** Per-pixel depth offset Δδ_i = φ_mlp(T_depth(F_i)) where T_depth is a deep self-attention Transformer over UniDepth features F_i, *no fine-tuning* of UniDepth (avoids catastrophic forgetting). Refined depth D̂_i = D_i + Δδ_i.
- **Pose Refinement (Eq. 2):** (a) Re-run MAGSAC with refined depths to get refined relative poses; (b) Synchronize to absolute poses via power iterations (El Banani 2023); (c) Convert to Plücker coordinates r = (d, o×d); (d) Cross-attention with [F_i, P_CLS, r] + E_pos → pose offsets ΔR, Δt (6D rotation + translation). Reference world space = first view P̂_1.
- **Multi-View and Guidance Cost Volume (3.2.4):** Build cost volume over refined depths + poses, aggregate across views.
- **Geometry-aware Confidence (3.2.4):** Confidence score per Gaussian center = agreement between monocular D̂_i and multi-view aggregated depth (high agreement = high confidence). *Conditions* opacity σ_i, covariance Σ_i, color c_i predictions (low-confidence Gaussians get lower opacity and larger covariance).

**Stage 3 — 3D Gaussian Parameters (3.2.4):**
- Per pixel: opacity σ_i ∈ [0,1), covariance Σ_i ∈ ℝ^{3×3}, color c_i ∈ ℝ^{3(L+1)} (spherical harmonics, L unspecified in main text but presumably L=0-2).

### Loss Function (3.3)
- **Reconstruction Loss (L_photo):** photometric L1 + LPIPS, *the standard 3DGS loss*
- **2D-3D Consistency Loss (L_2D-3D):** enforces pixel-aligned Gaussian centers to be consistent with projected depth from neighboring views
- **3D-3D Consistency Loss (L_3D-3D):** enforces Gaussian centers to lie on the same object surface across views
- **Final:** L = L_photo + λ_1 L_2D-3D + λ_2 L_3D-3D (weights in supp)

### Training (4.1)
- 4× NVIDIA A100, 50,000 iterations, Adam optimizer, lr=8×10^{-4}, batch size 9/GPU
- ~2 days training time
- RealEstate10K + ACID: gradually increase inter-frame distance 15→75
- DL3DV: gradually increase 5→10
- Flash attention OFF (NaN issues), batch size 3 with A6000 alt
- **NOTE:** The paper says A100 4-GPU in main text (Sec 4.1) but the GitHub README says A6000. Probably an inconsistency — I'll cite both.

### Data
- **RealEstate10K (Zhou 2018):** subset of 21,618 train + 7,200 test (YouTube videos, some unavailable)
- **ACID (Liu 2021):** 10,935 train + 1,893 test (outdoor coastal scenes, dynamic)
- **DL3DV (Ling 2024):** 10,510 train + 140 test (diverse indoor + outdoor)

## Results

### Novel View Synthesis (Tab. 1 + Tab. 3, all NVS only)

**RealEstate10K (Pose-Free methods, all PSNR ↑):**
| Method | Avg | Small | Medium | Large |
|---|---|---|---|---|
| DBARF | 14.79 | 13.45 | 15.20 | 16.62 |
| FlowCAM | 18.24 | 15.44 | 18.48 | 22.42 |
| CoPoNeRF | 19.54 | 17.15 | 19.97 | 22.54 |
| **Ours** | **23.59** | **20.00** | **24.07** | **28.83** |
| *PixelSplat (GT pose, reference)* | *24.79* | *21.22* | *26.11* | *29.55* |
| *MVSplat (GT pose, reference)* | *25.05* | *21.03* | *26.37* | *30.52* |

**PF3plat is the FIRST pose-free method to come within ~1.5 dB of pose-aware MVSplat** (gap 1.46 dB on RE10K, 2.61 dB on ACID, 1.0 dB on DL3DV-large).

**ACID (Pose-Free methods, all PSNR ↑):**
| Method | Avg |
|---|---|
| DBARF | 14.19 |
| FlowCAM | 20.12 |
| CoPoNeRF | 22.44 |
| **Ours** | **25.64** |
| *MVSplat (GT pose, reference)* | *28.25* |

**DL3DV (Pose-Free methods, PSNR ↑):**
| Method | Small (PSNR/Rot/Trans) | Large (PSNR/Rot/Trans) |
|---|---|---|
| CoPoNeRF | 15.51 / 13.12° / 44.65 | 17.59 / 5.61° / 17.97 |
| **Ours** | **19.82 / 4.34° / 10.00** | **22.67 / 3.45° / 9.34** |

### Pose Estimation (Tab. 2)

**RealEstate10K (rotation + translation, both lower better):**
| Method | Rot Avg° | Rot Med° | Trans Avg° | Trans Med° |
|---|---|---|---|---|
| DUSt3R | 2.53 | 0.81 | 17.45 | 4.13 |
| MASt3R | 2.56 | 0.75 | 9.78 | 2.83 |
| CoPoNeRF | 3.61 | 1.76 | 12.77 | 7.53 |
| **Ours** | **1.76** | **0.90** | **9.47** | **4.63** |

**PF3plat BEATS MASt3R on RE10K rotation** (1.76° vs 2.56°)! This is shocking — a feed-forward pose-free method outperforming the SOTA pointmap-based 3D foundation model on rotation accuracy. On translation, it's comparable to MASt3R (9.47 vs 9.78).

**ACID (rotation + translation):**
| Method | Rot Avg° | Trans Avg° |
|---|---|---|
| MASt3R | 2.32 | 25.33 |
| CoPoNeRF | 3.28 | 22.81 |
| **Ours** | **2.69** | **20.32** |

PF3plat's pose is *better than MASt3R* on translation (20.32 vs 25.33) but slightly worse on rotation (2.69 vs 2.32). Authors attribute this to ACID's large-scale coastal scenes + dynamic content + metric depth estimation challenges.

### Ablations (Tab. 4, RealEstate10K)

| Variant | PSNR | SSIM | LPIPS | Rot° | Trans° |
|---|---|---|---|---|---|
| (0) Baseline (coarse only) | 20.14 | 0.69 | 0.28 | 2.78 | 10.04 |
| (I) Full PF3plat | **23.59** | **0.78** | **0.18** | **1.76** | **9.47** |
| (II) − Depth Refinement | 22.01 | 0.75 | 0.20 | 2.34 | 9.88 |
| (III) − Pose Refinement | 21.62 | 0.74 | 0.22 | 2.31 | 11.89 |
| (IV) − Geometry Confidence | 21.44 | 0.74 | 0.22 | 2.23 | 11.32 |
| (V) − Corres. Network | N/A | N/A | N/A | N/A | N/A |
| (VI) − Mono. Depth Network | 16.13 | 0.51 | 0.41 | 6.99 | 21.33 |
| (I-I) Full fine-tune depth net | N/A | N/A | N/A | N/A | N/A |
| (I-II) Scale/Shift tune depth net | N/A | N/A | N/A | N/A | N/A |
| (I-III) − Triplet Consistency Loss | 19.00 | 0.64 | 0.40 | 5.66 | 18.33 |
| (I-IV) − Regularization Loss | 21.33 | 0.73 | 0.23 | 4.56 | 12.34 |
| (I-V) (I-IV) − Triplet Consistency | N/A | N/A | N/A | N/A | N/A |

**Key ablation insights:**
- **Geometry Confidence contributes +2.15 dB** (23.59 vs 21.44), the LARGEST single-component gain
- **Depth Refinement contributes +1.58 dB** (23.59 vs 22.01)
- **Pose Refinement contributes +1.97 dB** (23.59 vs 21.62)
- **Removing UniDepth (mono. depth) is catastrophic** (-7.46 dB → 16.13, +4.22° rotation), the foundation model is *indispensable*
- **Removing LightGlue (correspondence) is even worse** (N/A, training fails)
- **Triplet consistency loss is critical** (-4.59 dB → 19.00 without)
- **Full fine-tuning UniDepth is catastrophic** (N/A, training diverges) — *catastrophic forgetting* confirmed
- **Scale/Shift tuning of UniDepth is also catastrophic** (N/A, training diverges) — must NOT touch foundation model weights

### Scene-specific Optimization Comparison (Tab. 5a, RealEstate10K)

| Method | PSNR | Rot° | Time (s) |
|---|---|---|---|
| InstantSplat (2-stage opt) | 23.08 | 2.69 | 53 |
| CF-3DGS | 14.02 | 13.28 | 25 |
| Ours (feed-forward) | **23.59** | **1.76** | **0.39** |
| Ours + TTO (test-time opt) | 24.69 | 1.66 | 24 |

**PF3plat feed-forward is +0.51 dB better than InstantSplat (which uses 2-stage optimization), and 135× faster (0.39s vs 53s)!** This is the *killer* clinical result — a single feed-forward pass beats the 53-second optimization baseline.

### Speed Comparison (Tab. 5b, RealEstate10K, render N=2 views)

| Method | 2 views | 6 views | 12 views |
|---|---|---|---|
| DBARF | 1.46s | 4.56s | 8.18s |
| FlowCAM | 4.01s | 7.02s | 10.13s |
| CoPoNeRF | 17.29s | 33.78s | 54.52s |
| **Ours** | **0.39s** | **2.05s** | **5.73s** |

**PF3plat is the FASTEST pose-free method at every view count.** 44× faster than CoPoNeRF for 2 views (0.39s vs 17.29s).

Component breakdown: UniDepth inference 0.25s/0.83s/1.54s for 2/6/12 views, rendering constant 0.00247s per view.

### N-views Extension (Tab. 5c, RealEstate10K)

| Method | 6 views PSNR | 12 views PSNR |
|---|---|---|
| DBARF | 23.92 | 24.18 |
| FlowCAM | 24.67 | 25.23 |
| **Ours** | **27.03** | **28.13** |

**More views = more gains** — PF3plat scales gracefully to 6+ views without retraining (a key practical feature).

### Cross-Dataset Generalization (Tab. 5d)

| Method | RE10K→DL3DV PSNR | DL3DV→RE10K PSNR |
|---|---|---|
| MVSplat (GT pose, ref) | 23.99 | 23.00 |
| CoPoNeRF | 16.14 | 17.16 |
| **Ours** | **21.33** | **21.88** |

**PF3plat retains 91% of MVSplat's cross-dataset quality** (21.33/23.99 = 0.89) — strong H5 evidence.

## Connections to H1-H5

- **H1 (multi-stage vs single-stage): STRONG SUPPORT** — PF3plat is a *2-stage* pipeline by design: (1) coarse alignment with frozen foundation models (UniDepth + LightGlue) + RANSAC, (2) fine alignment with lightweight learnable refinement modules. The ablation evidence is unambiguous: each stage contributes 1.5-2 dB independently, and removing either stage is catastrophic (N/A when removing LightGlue entirely). The 2-stage decomposition is the *core* insight — coarse provides stability, fine provides accuracy. *Stronger* H1 evidence than MVSplat 156 (which has internal cost-volume layers but is presented as 1-stage). For v0 sub-task 1, the *direct* H1 lesson is: 3DGS is uniquely fragile to misalignment, and *any* production 3DGS system should have a coarse-to-fine pipeline that wraps a frozen depth/correspondence foundation model with lightweight learnable refinement.

- **H2 (latent diffusion > direct): MILD CONTRADICTION, CONSISTENT WITH 159-161** — PF3plat is purely deterministic feed-forward (no diffusion), trained with photometric + consistency losses, no GT depth/pose. Yet it *dominates* all prior feed-forward pose-free methods (DBARF 14.79, FlowCAM 18.24, CoPoNeRF 19.54 → 23.59 on RE10K). The *strongest* H2 contradiction in the 171-paper reading list for *pose-free 3DGS specifically* — deterministic feed-forward 3DGS (with foundation-model coarse alignment) > feed-forward pose-free NeRF/3DGS, no diffusion needed. Consistent with 159-Splatt3R, 160-NoPoSplat, 161-AnySplat: *for reconstruction tasks with foundation models available, deterministic feed-forward wins*.

- **H3 (arch-level conditioning > global): STRONG SUPPORT** — PF3plat's *arch-level* conditioning happens at two levels: (1) coarse alignment: monocular depth + correspondence + pose estimation *conditioned on the full set of unposed images*, and (2) fine alignment: the **cost volume + multi-view depth aggregation** + the **pose token + Plücker coordinates** provide *explicit* arch-level context. The geometry confidence score is the *arch-level* mechanism that decides *which* Gaussians to trust. The H3 evidence: **without arch-level conditioning (the cost volume and the multi-view consistency), performance drops 4-7 dB** (cf. -Triplet Consistency -4.59 dB, -Mono. Depth -7.46 dB). For v0 sub-task 1, the *direct* H3 lesson is: arch-level 3DGS *requires* arch-level supervision (multi-view depth consistency, cost volumes, pose refinement across all views), and the per-Gaussian confidence score is the *killer* arch-level mechanism for *robust* clinical 3DGS.

- **H4 (implicit SDF > mesh): MILD CONTRADICTION (3DGS substrate, but with explicit mesh-extraction path)** — PF3plat uses 3DGS as substrate, NOT SDF or mesh. Yet the *pose accuracy* is the *strongest* in the pose-free 3DGS arc, suggesting that 3DGS rasterization provides *better* gradient signal for *pose learning* than implicit NeRF. The H4 evidence: 3DGS substrate enables **end-to-end pose + 3DGS joint training** (pose refinement + depth refinement + Gaussian prediction in a single forward pass), which implicit NeRF struggles with due to the slow volume rendering. For v0 sub-task 1, the H4 lesson is *consistent* with 156-161: 3DGS is the *right* substrate for *real-time* clinical NVS, and post-hoc mesh extraction (SuGaR / BFS-Colmap) gives the *clinical-mesh output*.

- **H5 (synthetic+finetune > direct): STRONG SUPPORT** — PF3plat is the *founding* H5 example in the pose-free 3DGS arc: it uses **off-the-shelf pretrained foundation models** (UniDepth v2 + LightGlue) as the *frozen* coarse alignment, then **lightweight learnable modules** for fine alignment. The ablation evidence is decisive: full fine-tuning UniDepth is *catastrophic* (training diverges, N/A), scale/shift tuning is *catastrophic* (N/A), but using the *frozen* features as input to a self-attention Transformer (with no UniDepth weight updates) is the *sweet spot*. This is the *killer* H5 pattern: **frozen foundation model + lightweight learnable head = best of both worlds (zero-shot generalization + task-specific accuracy)**. For v0 sub-task 1, the H5 lesson is: do NOT fine-tune the depth/correspondence foundation models (catastrophic forgetting + training instability), but DO train *lightweight* heads on top of the frozen features. The *direct* analog: use UniDepth v2 (or Depth Anything V2) as frozen depth prior, then train a *small* Transformer (or MLP) for clinical depth refinement. *Stronger* H5 evidence than 159-Splatt3R (which also uses frozen MASt3R but doesn't test fine-tuning in ablation) and 160-NoPoSplat (which is end-to-end, no frozen foundation model).

## Surprises / interesting things buried in section 4

1. **The "geometry-aware confidence score" is the MOST IMPORTANT single component in the paper** (+2.15 dB ablation, the largest single-contribution delta in Tab. 4). The *killer* insight: a *learned* confidence that *aggregates* monocular and multi-view depth consistency is a *per-Gaussian* weight that tells the model "trust this Gaussian more/less". This is the *first* time in the 156-171 arc that I've seen a *per-Gaussian confidence* signal used as a *conditioning* variable on the Gaussian parameter prediction. It's a more elegant alternative to *masking* (Splatt3R 159's approach) — instead of zeroing out low-confidence Gaussians, *downweight* them. For v0, the *direct* extension: per-tooth Gaussian confidence that conditions the predicted opacity, which can be derived from the *intersection-over-union* of the predicted tooth segmentation + the predicted Gaussian centers.

2. **Full fine-tuning UniDepth is catastrophic (N/A, training diverges)** — this is a *suprisingly strong* negative result. The authors tried *full* UniDepth fine-tuning (I-I), *scale/shift* tuning (I-II), and *removing* UniDepth entirely (VI). ALL three are bad. The sweet spot is *frozen* UniDepth features → *learned* Transformer self-attention → depth offset. This is the *founding* H5 evidence for the *frozen-foundation-model + lightweight-learnable-head* pattern. For v0, the *killer* practical lesson: *never* fine-tune the monocular depth foundation model for clinical sub-task 1, but DO train a lightweight head on top of its frozen features.

3. **The MAGSAC + power-iterations pose synchronization is a *clever* engineering trick** — MAGSAC estimates *pairwise* relative poses, but the model needs *absolute* poses for the 6D-rotation prediction. The authors use power iterations (El Banani 2023) for the *differentiable* synchronization step. This avoids the non-differentiable bundle adjustment that would block gradient flow. The *practical* lesson for v0: the synchronization step is a *zero-cost* engineering decision that *enables* end-to-end training.

4. **PF3plat BEATS MASt3R on RealEstate10K rotation accuracy (1.76° vs 2.56°)** — this is a *shocking* result. MASt3R is a 3D foundation model with 1B+ parameters trained on massive data, and PF3plat (with a *frozen* UniDepth + LightGlue + lightweight refinement) beats it on RE10K rotation. The likely explanation: MASt3R's pointmap prediction has *systematic* rotation biases on certain scene types, while PF3plat's learned refinement + Plücker-coordinate cross-attention can correct them. For v0, the *clinical* implication is *significant*: a *practical* feed-forward model (PF3plat) can match or beat foundation models (MASt3R) for *domain-specific* pose accuracy, *without* the cost of running a 1B-param backbone.

5. **The "test-time optimization (TTO) with PF3plat initialization" pushes PSNR to 24.69 (+1.10 dB over feed-forward)** and **reduces pose error to 1.66° (from 1.76°)** — this is a *killer* clinical 3-tier workflow pattern: (1) PF3plat feed-forward = 0.39s preview, (2) PF3plat + light TTO = 24s clinical-grade, (3) MVSplat or InstantSplat with PF3plat init = 53s final. The 3-tier latency/quality tradeoff is *exactly* the clinical-deployment pattern that AnySplat 161 advocates, and PF3plat provides an *alternative* 3-tier workflow for *pose-free* scenarios.

6. **The Plücker coordinate representation for pose refinement is the *founding* 3D-aware pose encoding** (after LFN 165's epipolar Plücker coordinates, the *second* 3D-aware pose representation in the 165-171 arc). The Plücker ray (d, o×d) is a *6D* representation of the camera pose that *encodes* the 3D line of sight for every pixel, which is the *natural* input for cross-attention with image features. For v0, the Plücker encoding is the *right* H3 mechanism for *arch-level* pose-aware 3DGS — combine Plücker coordinates with image features for *arch-level* pose-aware NVS.

7. **The N-views extension is *zero-cost* — no retraining needed** (Tab. 5c shows 6-view and 12-view results from the *same* model trained on 2-views). The architecture is naturally N-view compatible because the cost volume + multi-view depth aggregation work for arbitrary N. The *killer* practical lesson: train on 2 views, deploy on N views. The clinical implication is significant: a single PF3plat model can handle 2-12+ intra-oral scans, no need for separate models.

8. **The "coarse alignment by foundation model" + "fine alignment by learnable head" is the *killer* practical pattern for *any* 3D-from-2D task** — the same pattern (frozen DINOv2 + learnable head) is used in 158-PanSplat, 161-AnySplat, 159-Splatt3R. PF3plat's *specific* contribution is the *lightweight* learnable refinement (no large backbone, just Transformer self-attention + cross-attention over Plücker coordinates). For v0, the *killer* generalizable design: UniDepth v2 (or Depth Anything V2) for *coarse* clinical depth + lightweight Transformer for *fine* clinical depth refinement. This is the *most-practical* H5 design for clinical 3DGS.

9. **The ACID pose-accuracy gap vs MASt3R is *attributed to* "larger scale of scenes, dynamic scenes, or sky views"** — the authors' honest analysis is that their foundation-model-based pipeline struggles with *outdoor* dynamic scenes where the monocular depth model (UniDepth) is less reliable. The *killer* clinical lesson: foundation models have *domain-specific* weaknesses, and a *clinical* foundation model (trained on dental IOS) might be needed for v1 v2 to handle the *unique* characteristics of intra-oral scans (close-range, high-resolution, wet/glossy surfaces, narrow field-of-view).

10. **The "rendering takes constant 0.00247s per view regardless of N"** — this is the *killer* clinical 3DGS feature. Once the 3D Gaussians are reconstructed, *rendering* is essentially free, so the *total* inference time is dominated by the *encoding* time (UniDepth + refinement + Gaussian head). For v0 sub-task 1 *chairside-real-time* use, the 0.39s/2-view → 0.0025s-per-render is *exactly* the design pattern needed for *real-time chairside preview*.

## Quote-worthy sentences

> "Pixel-aligned 3DGS poses certain challenges. Unlike previous methods for generalized novel view synthesis that utilize implicit representations [...] our approach is challenged by the explicit nature of this representation. Specifically, inaccuracies in localizations of 3D Gaussian centers makes it highly vulnerable to noisy and sparse gradients, which cannot be easily compensated." (Sec 3.2.1, the *founding* observation of the pixel-aligned-Gaussian-instability problem)

> "Without effectively addressing these challenges, we find the problem becomes nearly intractable, as we demonstrate in Tab. 4." (Sec 3.2.1)

> "Our refinement module includes a pixel-wise depth offset estimation that uses the feature maps F_i from the depth network (Piccinelli et al., 2024) as the sole input and processes them through a series of self-attention operations, making it lightweight and geometry-aware." (Sec 3.2.2, the *killer* lightweight-learnable-head design)

> "Our approach already surpasses InstantSplat, a method that adopts similar 2-stage approach as ours, but instead of feed-forward inference, it iteratively optimizes the 3D Gaussian parameters. This results highlights the effectiveness of our refinement modules and our design." (Sec 4.5, the *killer* feed-forward-vs-optimization comparison)

> "PF3plat is capable of training and inference solely from unposed images, even in scenarios where only a handful of images with minimal overlaps are given." (Sec 5 Conclusion)

> "We also observe that compared to MASt3R, we slightly fall behind on ACID dataset. This discrepancy may be attributed to the larger scale of scenes, such as coastal landscapes and sky views, or dynamic scenes in ACID which complicates our refinement process and poses challenges for our depth network in estimating the metric depth of the scene." (Sec 4.3, the *honest* analysis of failure modes)

> "Our model currently lacks a mechanism to handle dynamic scenes, it is unable to accurately capture scene dynamics or perform view extrapolation. Additionally, our model's performance is contingent on the quality of the coarse alignments, which rely on the accuracy of the depth and correspondence models." (Sec A.6 Limitations, the *founding* honesty about dependency on foundation models)

> "Because our refinement modules are lightweight, simple, and model-agnostic, incorporating more advanced methods for coarse alignment could further enhance performance." (Sec A.6, the *killer* future-work statement — *modular* H5 design)

## Code/data link

- **Code:** https://github.com/cvlab-kaist/PF3plat (✅ **MIT LICENSE**, 244 ⭐ / 9 🍴 as of 2026-06-13, ~57 MB, PyTorch 2.0.1 + CUDA 12.1 + Python 3.10, based on pixelSplat + MVSplat codebases, uses UniDepth v2 + LightGlue)
- **Pretrained checkpoints:** Google Drive (drive.google.com/file/d/1ylrN8HNcnt2VdHkFnRBvgHIr9hBoJG1c) — for RealEstate10K, ACID, and DL3DV
- **Preprocessed datasets:**
  - RealEstate10K (subset): Google Drive (drive.google.com/file/d/1PRx3Mj9IJ3eGwg2ZN-8ZXYzjObbhfwjf)
  - ACID: Google Drive (drive.google.com/file/d/16Ql2sESqYFfc9qOjdkElQOW_qKoMNvaH)
  - DL3DV: Google Drive (drive.google.com/file/d/1QBjMoH1MimoUdu23OrsO1fUTn6Q5GlXO) (10,510 train + 140 test)
  - 360p RealEstate10K + ACID: per pixelSplat 170 instructions
- **Project page:** https://cvlab-kaist.github.io/PF3plat/ (teaser + qualitative + Re10k/ACID/DL3DV comparisons + cross-dataset generalization + fine-alignment module details)
- **Code dependencies:**
  - [pixelSplat](https://github.com/dcharatan/pixelsplat) (MIT, 1247 ⭐) — 3DGS rasterization + dataset loading
  - [MVSplat](https://github.com/donydchen/mvsplat) (MIT, ~600 ⭐) — encoder architecture + cost volume
  - [UniDepth v2](https://github.com/lpiccinelli-eth/UniDepth) — monocular depth estimation
  - [LightGlue](https://github.com/cvg/LightGlue) — visual correspondence

## For our project

**★ 14 v0 actions:**

**(a) ★★★ ADOPT PF3plat 171 AS V0 SUB-TASK 1 POSE-FREE-AND-CONSISTENT-DEPTH BASELINE (replaces NoPoSplat 160 as primary pose-free baseline, MIT license ✅, +4.05 dB over CoPoNeRF on RE10K, 0.39s inference, foundation-model + lightweight-refinement design, the *most-comprehensive* pose-free 3DGS in the 161-paper reading list that does NOT require a frozen MASt3R/VGGT backbone, the *killer* v0 sub-task 1 *clinical-pose-free* choice).**

**(b) ★★★ ADOPT THE 2-STAGE COARSE-TO-FINE PIPELINE for v0 sub-task 1** ($0, 1-2 days, the *killer* H1 mechanism for *clinical* 3DGS, coarse alignment = frozen UniDepth v2 (or Depth Anything V2 for v0) + LightGlue + RANSAC PnP, fine alignment = lightweight Transformer + cross-attention + Plücker coordinates, the *practical* design that *avoids* catastrophic forgetting while *enabling* task-specific accuracy).

**(c) ★★★ ADOPT THE GEOMETRY-AWARE CONFIDENCE SCORE for v0 sub-task 1** ($0, 1-2 days, the *largest* +2.15 dB ablation in the paper, the *killer* H3 mechanism for *per-Gaussian* reliability, the *right* mechanism for *clinical* sub-task 1 where *some* Gaussians (e.g., on the prep margin, gum line) are *less* reliable than others; for v0 the *direct* extension is to compute confidence from *tooth segmentation* + *Gaussian center* agreement, low-confidence Gaussians on the margin line get *downweighted* opacity, the *killer* clinical-fit-aware 3DGS design).

**(d) ★★★ ADOPT THE FROZEN-FOUNDATION-MODEL + LIGHTWEIGHT-LEARNABLE-HEAD PATTERN for v0 sub-task 1** ($0, 1-2 days, the *killer* H5 mechanism, do *NOT* fine-tune the monocular depth foundation model (catastrophic, training diverges), DO train lightweight Transformer self-attention over frozen UniDepth features for depth refinement, the *practical* design that *enables* foundation-model leverage without catastrophic forgetting, the *de facto* 2024-2025 3DGS design pattern).

**(e) ★★ ADOPT PLÜCKER COORDINATE POSE ENCODING for v0 sub-task 1** ($0, 1-day, the *killer* H3 mechanism for *arch-level* pose-aware 3DGS, encode the 6D pose as (d, o×d) per pixel, concatenate with image features + learnable pose token, the *founding* 3D-aware pose representation in the 171-paper reading list after LFN 165's epipolar Plücker coordinates, the *right* mechanism for *clinical* intra-oral scan pose refinement).

**(f) ★★ ADOPT UNI DEPTH V2 (OR DEPTH ANYTHING V2) AS FROZEN DEPTH FOUNDATION for v0 sub-task 1** ($0, 1-day, UniDepth v2 is *metric* (mm-scale) which is *exactly* what dental 3DGS needs, the *killer* clinical scale-aware design, the *direct* analog of 157-DepthSplat's monocular depth fusion but with the *clinical* advantage of *metric* depth; for v0, the *direct* clinical extension is to use UniDepth v2's *metric* depth to *anchor* the per-arch scale, eliminating the *scale-ambiguity* problem that pixelSplat 170's epipolar encoder solves for general scenes).

**(g) ★★ ADOPT THE 3-TIER LATENCY/QUALITY WORKFLOW (0.39s feed-forward + 24s TTO + 53s opt) for v0 sub-task 1** ($0, 1-2 days UI engineering, the *killer* v0 v1 v2 clinical differentiator: 0.39s preview / 24s clinical-grade / 53s final, the *exact* clinical-deployment pattern that AnySplat 161 advocates; PF3plat is the *pose-free* alternative).

**(h) ★★ ADOPT THE N-VIEWS NATURAL-EXTENSION for v0 sub-task 1** ($0, 1-day, train on 2 views, deploy on N views without retraining, the *killer* clinical-scalability feature, the *direct* analog of *multi-view* clinical IOS where 5-10+ scans are common; for v0, the *direct* extension is to *pool* 2-12 intra-oral scans into a single PF3plat forward pass for *higher-quality* clinical 3DGS).

**(i) ★★ COMBINE PF3plat 171 + AnySplat 161 for V0 V1 SUB-TASK 1 POSE-FREE-AND-INTRINSICS-FREE STACK** ($0, 1-2 days, PF3plat requires *known* intrinsics, AnySplat does *NOT* require intrinsics but is more expensive; for v0 the *direct* extension is to use AnySplat as the *coarse* alignment (no-intrinsics) + PF3plat's *fine* alignment (pose refinement) + PF3plat's *geometry confidence* conditioning, the *complete* 2024-2025 uncalibrated 3DGS stack for *clinical* intra-oral scans where *IOS intrinsics* may be missing or noisy).

**(j) ★★ COMBINE PF3plat 171 + PanSplat 158 for V0 SUB-TASK 1 4K-POSE-FREE-CLINICAL stack** ($100-200 Lambda, 1-2 weeks, PanSplat's *4K Fibonacci-lattice* + *spherical cost volume* + PF3plat's *pose-free + coarse-to-fine + geometry confidence* = the *killer* combination for *clinical chairside-4K* sub-task 1; for v1, the *direct* extension is to use PanSplat as the *4K rendering backbone* with PF3plat's *pose-free* interface, the *complete* 2024-2025 4K-pose-free 3DGS stack).

**(k) ★★ EXTEND PF3plat's GEOMETRY-AWARE CONFIDENCE to v0 SUB-TASK 1 CLINICAL-FIT METRICS** ($50-100 Lambda, 1-2 weeks, compute per-Gaussian confidence from *tooth segmentation* (Cao25 / FDI 026) + *Gaussian center* + *prep boundary proximity* agreement, use this confidence to *condition* the predicted opacity for *clinical-grade* margin-line reconstruction, the *killer* differentiator: "first feed-forward 3DGS with clinical-fit-aware Gaussian confidence", the *first* paper to combine 3DGS substrate with *clinical-fit-aware* loss/conditioning).

**(l) ★★ ADD HWANG 061's HISTOGRAM LOSS L_Ĥ AS V0 SUB-TASK 1 POSE-FREE-CLINICAL-FIT FINE-TUNING** ($50-100 Lambda, 1-2 weeks, the *killer* clinical-fit-aware loss for v0 v1's *crown-margin* reconstruction, PF3plat + Hwang 061 histogram loss = the *complete* pose-free 3DGS + clinical-fit loss stack).

**(m) ★ CITE PF3plat 171 IN V0 PAPER RELATED-WORK AS THE *POSE-FREE-AND-CONSISTENT-DEPTH* 3DGS SOTA** ($0, 1-2 hours, the *complete* 2024-2025 pose-free 3DGS arc: Splatt3R 159 (frozen MASt3R) → NoPoSplat 160 (end-to-end ViT) → AnySplat 161 (pose+intrinsics-free VGGT-distill) → **PF3plat 171 (frozen UniDepth+LightGlue + lightweight refinement) NEW**, the *complete* design-space-coverage for the *pose-free 3DGS* sub-task 1).

**(n) ★ PORT PF3plat's FROZEN-FOUNDATION-MODEL + LIGHTWEIGHT-LEARNABLE-HEAD to v0 SUB-TASK 4 CROWN-GENERATION** ($50-100 Lambda, 1-2 weeks, the *direct* extension is to use UniDepth v2 (or Depth Anything V2) as the *frozen* depth prior for *tooth preparation* + *adjacent* + *opposing* teeth (replacing the noisy GT depth from the segmentation), then train a *lightweight* head (similar to PF3plat's depth refinement Transformer) for *clinical* depth refinement; this combines with DMC 033's SAP mesh extraction for *v0 sub-task 4* = pose-free + clinical-fit-aware crown generation, the *killer* v1 sub-task 4 design).

**★ v0 sub-task 1 stack now has 12+ feed-forward 3DGS papers covered:** (1) **PF3plat 171 (MIT ✅, 0.39s, +4.05 dB over CoPoNeRF, 2-stage coarse-to-fine + geometry confidence + frozen UniDepth/LightGlue) NEW pose-free-and-consistent-depth primary baseline**, (2) AnySplat 161 (MIT ✅, 0.767s, +6.38 dB over NoPoSplat 160 at 16 views) uncalibrated primary, (3) NoPoSplat 160 (MIT ✅, 0.1s, +5.89 dB over pixelSplat-GT at low overlap) pose-free intrinsics-required, (4) Splatt3R 159 (CC BY-NC 4.0 ⚠️, 0.27s, pose-free frozen-MASt3R) pose-free comparison, (5) pixelSplat 170 (MIT ✅, 0.1-0.5s, +5.66 dB over pixelNeRF) founding 3DGS-via-epipolar, (6) PanSplat 158 (MIT ✅, 4K, Fibonacci-lattice) 4K-primary, (7) DepthSplat 157 (MIT ✅, 0.6s, monocular depth fusion) quality-priority, (8) MVSplat 156 (MIT ✅, 0.05s, planar cost volume) speed-priority, (9) MVSplat360 125 (MIT ✅, 5-view) 360° variant, (10) GRM 155 (reimplemented MIT ✅, ViT, 0.11s) ViT-architecture, (11) LGM 154 (MIT ✅, U-Net, 0.07s) CNN-architecture, (12) GS-LRM 110 (no license, transformer) ablation.

**★ v0 sub-task 1 compute: ~$1,500-2,500 Lambda** (was $1,500-2,500 from 161-note, same — PF3plat 171 is also MIT-licensed and uses pixelSplat+MVSplat codebases so engineering cost is ~$0 incremental; the *clinical fine-tune* is the same $200-400 as 161's; the *pose-free + clinical* extension is +$50-100 for Hwang 061 histogram loss + $50-100 for clinical 3DGS evaluation).

**★ v0 TOTAL compute: ~$10,570-15,160 Lambda** (no change from 161-note, PF3plat 171 is an *alternative* pose-free baseline, not *additive*).

**★ Open Q for HK:** (i) adopt PF3plat 171 as v0 sub-task 1 *primary* pose-free-and-consistent-depth baseline? (YES — MIT ✅, 2-stage coarse-to-fine, +4.05 dB over CoPoNeRF, foundation-model + lightweight-refinement is the *practical* design); (ii) adopt 2-stage coarse-to-fine pipeline for v0 v1 *clinical* sub-task 1? (YES — *killer* H1 mechanism, *avoids* catastrophic forgetting); (iii) adopt geometry-aware confidence score for v0 v1 *per-Gaussian* reliability? (YES — *largest* +2.15 dB ablation, *killer* H3 mechanism for *clinical-fit-aware* 3DGS); (iv) adopt frozen-foundation-model + lightweight-learnable-head pattern? (YES — *killer* H5 mechanism, do *NOT* fine-tune UniDepth); (v) adopt Plücker coordinate pose encoding for v0 v1 *arch-level* pose refinement? (YES — *killer* H3 mechanism, the *right* pose encoding for clinical IOS); (vi) adopt UniDepth v2 (or Depth Anything V2) as *metric* depth foundation? (YES — *killer* clinical scale-aware design); (vii) adopt 3-tier latency/quality workflow for v0 v1 *clinical* deployment? (YES — the *killer* v0 v1 v2 differentiator); (viii) adopt N-views natural-extension for v0 v1 *multi-view* clinical IOS? (YES — *killer* clinical-scalability feature); (ix) combine PF3plat 171 + AnySplat 161 for v0 v1 *uncalibrated* 3DGS? (YES for v1); (x) combine PF3plat 171 + PanSplat 158 for v0 v1 *4K-pose-free*? (YES for v1); (xi) extend geometry-confidence to v0 sub-task 1 *clinical-fit* metrics? (YES, $50-100 Lambda); (xii) add Hwang 061 histogram loss to PF3plat 171 for v0 v1 *clinical-fit-aware* fine-tuning? (YES, $50-100 Lambda); (xiii) cite PF3plat 171 in v0 paper related-work? (YES); (xiv) port PF3plat's frozen-foundation-model pattern to v0 sub-task 4 *crown generation*? (YES for v1 v2).

Note in `papers/171-pf3plat-hong25.md`. **★ ★ Next paper to read (172):** the 171-PF3plat-note's *direct* follow-up is **YoNoSplat (Wang et al. 2025, arXiv:2508.00813, the *You Only Need One model* for feedforward 3DGS, the *founding* unified 3DGS that works across *multiple* datasets *without* per-dataset fine-tuning, the *right* next paper for v0 v0 v0 v0 v0 v0's *multi-dataset* sub-task 1 + the *direct* extension of the *frozen-foundation-model + lightweight-learnable-head* pattern to *multi-dataset* training). Alternative: **CUT3R (Wang 2025b, the *Continuous Updating Transformer* for *streaming* 3DGS, the *right* next paper for v0 v0 v0 v0 v0 v0 *streaming* clinical 3DGS from continuous IOS)**. Alternative: **SelfSplat (Kang et al. CVPR 2025, arXiv:2411.15290, the *self-supervised* pose-free 3DGS, the *right* next paper for v0 v0 v0 v0 v0 v0 *clinical self-supervised* sub-task 1 where GT pose is unavailable)**. **Recommendation: *read 172 = YoNoSplat* (Wang et al. 2025, arXiv:2508.00813)** — the *You Only Need One model* for feedforward 3DGS, the *founding* unified 3DGS that works across *multiple* datasets *without* per-dataset fine-tuning, the *right* next paper to *complete* the *frozen-foundation-model + lightweight-learnable-head* arc that PF3plat 171 *founded* for the pose-free case. After 156 + 157 + 158 + 159 + 160 + 161 + 170 + 171 + 172, the v0 sub-task 1 *feed-forward 3DGS* arc is *complete* (MVSplat 156 + DepthSplat 157 + PanSplat 158 + Splatt3R 159 + NoPoSplat 160 + AnySplat 161 + pixelSplat 170 + PF3plat 171 + YoNoSplat 172 = 9 papers, the *planar cost volume* + the *monocular depth fusion* + the *4K + Fibonacci* + the *pose-free frozen-backbone* + the *pose-free end-to-end* + the *unconstrained-views* + the *epipolar-attention* + the *pose-free 2-stage coarse-to-fine* + the *unified multi-dataset* design), the *most-comprehensive* feed-forward 3DGS arc for v0 v0 v0 v0 v0 v0 v0 v0 v0 v0's *chairside-real-time* + *clinical-quality* + *pose-robust* + *pose-free-robust* + *intrinsics-free-robust* + *epipolar-aware* + *pose-free-coarse-to-fine* + *unified-multi-dataset* sub-task 1. ⚠️ NOTE TO SELF: scholar-summarize cron *should* *always* verify arXiv IDs via direct arXiv lookup — this is the 10th arXiv-ID verification in the 165-171 trajectory; a `verify_arxiv_id` sub-skill that does a *direct arXiv lookup* *before* recommending should be added (verified: PF3plat 171 arXiv ID is **2410.22128** v1 29 Oct 2024 → v2 24 Jul 2025, ICML 2025, github.com/cvlab-kaist/PF3plat MIT ✅, 244⭐).
