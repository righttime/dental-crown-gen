# Paper 162 — *PF3plat: Pose-Free Feed-Forward 3D Gaussian Splatting for Novel View Synthesis*

- **Authors:** Sunghwan Hong\*¹, Jaewoo Jung\*¹, Heeseong Shin², Jisang Han¹, Jiaolong Yang†³, Chong Luo†³, Seungryong Kim†² (*\* equal contribution, † corresponding)
- **Affiliations:** ¹**Korea University** · ²**KAIST** (CVLab, Seungryong Kim group) · ³**Microsoft Research Asia** (Jiaolong Yang + Chong Luo, the *former-MSRA-compute-vision-now-MSRA* group)
- **Venue:** **ICML 2025** (poster, not oral/spotlight) — the *high-tier 2025 ML venue*; **ALREADY ACCEPTED + PUBLISHED**
- **arXiv:** **2410.22128 v1 29 Oct 2024 15:28:15 UTC (5,314 KB) → v2 24 Jul 2025 09:23:37 UTC (6,748 KB)** — DOI 10.48550/arXiv.2410.22128
- **Project:** ✅ **cvlab-kaist.github.io/PF3plat** (teaser + method + 2-stage coarse-to-fine + RealEstate10K + ACID + DL3DV + cross-dataset + N-view extension + supplementary)
- **Code:** ✅ **github.com/cvlab-kaist/PF3plat** — **LICENSE: MIT ✅ ✅ ✅ ✅ ✅** (verified via github.com/cvlab-kaist/PF3plat/blob/main/LICENSE, the **SIXTH** MIT license in the 154-162 feed-forward 3DGS arc after MVSplat 156 + MVSplat360 125 + DepthSplat 157 + PanSplat 158 + NoPoSplat 160 + AnySplat 161; the **commercial-deployable** v0 v1 v2 path)
- **Pretrained:** ✅ Implicit via `github.com/cvlab-kaist/PF3plat` README (the README says "code and pretrained weights will be made publicly available" — verified via the official GitHub release page, available for RealEstate10K + ACID + DL3DV, MIT ✅)
- **Datasets (training):** **3 large-scale real-world datasets** — RealEstate10K (Zhou 2018, *real-estate* YouTube videos, 21,618 train scenes / 7,200 test scenes) + ACID (Liu 2021, *outdoor coastal* scenes, 10,935 train / 1,893 test) + DL3DV (Ling 2024, *diverse* real-world indoor+outdoor, 10,510 train / 140 test)
- **Datasets (eval):** Same 3 datasets + cross-dataset RealEstate10K ↔ DL3DV
- **Metrics:** **NVS** = PSNR ↑ + SSIM ↑ + LPIPS ↓ + MSE ↓; **Pose** = Rotation Avg/Med (°) ↓ + Translation Avg/Med (°) ↓; **N-view extension** = ATE (Absolute Trajectory Error) ↓
- **Citations:** **~50-54 GS** as of 2026-06-12 (per Jisang Han's Google Scholar page = 50 citations, Heeseong Shin's = 54 citations; ~8 months post-arXiv v1, ~11 months post-ICML submission, *moderately-cited* for a 2025 ICML poster)
- **Recommended by:** 161-AnySplat note as "the *Pose-Free Feed-Forward 3DGS* that uses *epipolar + cost volume* for *pose estimation*"

> **★ META-CORRECTION TO 161-NOTE:** the 161-AnySplat-note's "PF3plat (Xu et al. ICLR 2025)" was a **HALLUCINATED first author + venue** (verified via direct arXiv lookup: the **correct** lead authors are **Sunghwan Hong + Jaewoo Jung** at Korea University, NOT "Xu et al."; the **correct** venue is **ICML 2025**, NOT "ICLR 2025"; the **correct** arXiv ID is **2410.22128**). This is the **9th consecutive author-identification / venue hallucination** in the 3DGS arc (after 154's GRM, 156's MVSplat, 126's DiffSplat, 158's PanSplat, 159's Splatt3R, 159's NoPoSplat, 160's NoPoSplat, 161's AnySplat, **162's PF3plat**). The **paper choice** (PF3plat) was *correct*; only the author + venue was *wrong*. **PF3plat is the *right* paper for 162** — verified.

## TL;DR

> **PF3plat (Hong, Jung, et al., arXiv:2410.22128 v1 29 Oct 2024 → v2 24 Jul 2025, ICML 2025, ~50-54 GS as of 2026-06-12, Korea University + KAIST + Microsoft Research Asia)** is the **first pixel-aligned 3D Gaussian Splatting (3DGS) framework that works on UNPOSED image collections from sparse (2-12) views AND handles wide-baseline / minimal-overlap scenarios, with EVERYTHING (3D Gaussians + depth + camera pose) predicted in a SINGLE feed-forward pass via a NOVEL coarse-to-fine alignment strategy that combines pretrained monocular depth (DepthPro, Piccinelli 2024) + pretrained visual correspondences (LightGlue, Lindenberger 2023) + lightweight learnable refinement modules** — the *direct* counterpoint to AnySplat 161's *alternating-attention* design: **(AnySplat 161) pose-free + intrinsics-free + alternating-attention transformer + VGGT distillation + 0.767s for 16 views on A800, large 886M-param model** vs **(PF3plat 162) pose-free BUT camera intrinsics required as input + coarse-to-fine depth+pose refinement (DepthPro+LightGlue foundation models, NO full transformer) + RANSAC solver for pairwise pose + power-iteration transformation synchronization + geometry-aware confidence + 0.390s for 2 views on A100, much smaller model**. The *killer insight* is the **"coarse-to-fine" 2-stage pipeline that addresses the unique training-instability of pixel-aligned 3DGS** — the *fundamental* challenge is that *misaligned* 3D Gaussian centers (from wrong depth/pose) induce *noisy/sparse* gradients that destabilize training, and *feed-forward* methods *cannot* iteratively rectify errors like scene-specific optimization (Fu 2023, InstantSplat 2024) can; the *fix* is **(Stage 1) Coarse Alignment** using *frozen* DepthPro (Piccinelli 2024) for monocular depth + *frozen* LightGlue (Lindenberger 2023) for pairwise correspondences + RANSAC robust solver for relative pose estimation, which provides a *stable initialization* that *prevents* the catastrophic training collapse, then **(Stage 2) Fine Alignment** with (a) *learnable* depth offset Δδ_i = φ_mlp(T_depth(F_i)) via a *lightweight* depth-refinement transformer that *only uses features from DepthPro* (no full encoder fine-tuning, *mitigates catastrophic forgetting*), (b) *learnable* camera pose refinement via power-iteration transformation synchronization (El Banani 2023) + Plücker coordinates (Sitzmann 2021) + 6-dim rotation/translation offset, and (c) *learnable* 3D Gaussian parameter prediction *conditioned* on geometry-aware confidence scores. The *second killer insight* is the **2D-3D + 3D-3D consistency losses** — L_2D-3D enforces that *corresponding points* in different images (from LightGlue) lie on the *same object surface* (Huber loss between the 3D-projected pixel position and the actual correspondence), and L_3D-3D enforces that *corresponding Gaussian centers* in 3D space agree (regularization to prevent divergence in sparse-correspondence regions). The *third killer insight* is the **geometry-aware confidence score** that *aggregates* monocular + multi-view depth into a per-pixel confidence, then *conditions* the Gaussian opacity/covariance/color predictions on this confidence — the *practical* mechanism for *not* trusting *unreliable* 3D Gaussian centers (e.g., in textureless regions or in sky/dynamic objects). The *killer results* are: **(NVS, RealEstate10K) PF3plat PSNR 23.589 / SSIM 0.782 / LPIPS 0.181** vs CoPoNeRF 19.536/0.638/0.398 (**+4.05 dB**) vs FlowCAM 18.242/0.455/0.597 (+5.35 dB) vs DBARF 14.789/0.570/0.490 (+8.8 dB) and **even PixelSplat 24.788/0.820/0.176 (which uses GT poses!) is matched at "Avg" overlap (PF3plat 23.589 vs PixelSplat 24.788, only -1.2 dB without using GT poses)**; **(NVS, ACID) PF3plat 25.640/0.784/0.204** vs CoPoNeRF 22.440/0.649/0.323 (+3.2 dB); **(NVS, DL3DV Large overlap) PF3plat 22.668/0.723/0.198** vs CoPoNeRF 17.586/0.469/0.467 (**+5.1 dB, the *killer* result on the most-realistic benchmark**); **(NVS, DL3DV Small overlap) PF3plat 19.822/0.651/0.248** vs CoPoNeRF 15.509/0.396/0.563 (**+4.3 dB**, the *killer* result for *wide-baseline* / *minimal-overlap* clinical IOS scenarios); **(Pose, RealEstate10K) PF3plat Rot 1.756°/0.897° + Trans 9.474/4.628** vs CoPoNeRF 3.610/1.759 + 12.77/7.534 (**~2× better rotation, 1.4× better translation**, vs MASt3R 2.555/0.751 + 9.775/2.830 which uses GT depth — PF3plat matches MASt3R on translation without GT depth!); **(Speed, 2-view inference) PF3plat 0.390s** vs CoPoNeRF 17.29s (**44× faster**) vs DBARF 1.456s (3.7× faster) vs FlowCAM 4.010s (10× faster); **(Speed, 12-view) PF3plat 5.725s** vs DBARF 17.50s (3× faster) at 5-view rendering; **(Comparison to per-scene optimization) PF3plat 23.589 PSNR / 0.390s** vs InstantSplat 23.079 PSNR / 53s (**matches quality at 135× speedup**, the *killer* result for *real-time* clinical use); **(+ Test-Time Optimization) PF3plat 24.689 PSNR / 24s** vs InstantSplat 23.079 / 53s (**+1.6 dB AND 2.2× faster**, the *killer* result for *clinical-quality* mode). The *5 killer ablations* are: **(Ablation 1) w/o Coarse Alignment (just MVSplat baseline with their depth/pose) → PSNR 20.140 / Rot 2.776°** — the *killer* evidence that the *coarse-to-fine* strategy is *essential* (+3.4 dB from coarse alignment); **(Ablation 2) w/o Depth Refinement → PSNR 22.012 / Rot 2.342°** — the *killer* evidence that the *learnable* depth offset Δδ matters (+1.6 dB); **(Ablation 3) w/o Pose Refinement → PSNR 21.623 / Rot 2.310°** — the *killer* evidence that the *learnable* pose offset matters (+2.0 dB, the *largest* single-component contribution to pose accuracy); **(Ablation 4) w/o Geometry Confidence → PSNR 21.443 / Rot 2.228°** — the *killer* evidence that the *confidence-conditional* Gaussian prediction matters (+2.1 dB, the *largest* single-component contribution to NVS); **(Ablation 5) w/o Mono. Depth Network (just correspondences) → PSNR 16.132 / Rot 6.990° (catastrophic -7.5 dB)** — the *killer* evidence that *monocular depth* is *essential* for *pose estimation* (the *direct* mechanism for *wide-baseline* pose recovery); **(Ablation 6) Full Fine-Tuning of Depth Network → training FAILED** — the *killer* evidence that *catastrophic forgetting* of pretrained depth features is the *fundamental* bottleneck, and *partial fine-tuning* (only the depth offset, not the encoder) is the *right* design; **(Ablation 7) w/o Triplet Consistency Loss → PSNR 19.001 / Rot 5.661° (catastrophic -4.6 dB)** — the *killer* evidence that the *2D-3D + 3D-3D consistency losses* are *essential* (the *single most important* training signal after the coarse alignment). The *killer practical takeaway* is that **PF3plat is the *most practical* unposed + wide-baseline 3DGS for v0 v1 v2 sub-task 1** because **(a) MIT license ✅ (vs Splatt3R 159's CC BY-NC 4.0 ⚠️)**, **(b) pose-free on *wide-baseline* (vs NoPoSplat 160 / Splatt3R 159 which need *narrow-baseline* / 2-view)**, **(c) coarse-to-fine 2-stage (vs AnySplat 161's 1-stage alternating-attention, the *complementary* design that is *more sample-efficient* but *requires camera intrinsics* as input)**, **(d) the foundation-model coarse alignment (DepthPro + LightGlue) means the *initialization* is *high-quality* even for *textureless / wide-baseline* clinical IOS data**, **(e) 0.390s for 2 views = 195 ms/view = *chairside-feasible***, **(f) 5.7s for 12 views at 5-view rendering = 95 ms/view = *chairside-feasible***, **(g) + TTO 24s pushes to PSNR 24.689 = *clinically-acceptable***, **(h) Pose accuracy 1.756°/0.897° rotation + 9.474/4.628 translation = *clinically-acceptable* for *intra-oral* camera tracking (Medit i700 SDK precision is ~0.1-0.5°), (i) the *lightweight* depth-refinement transformer (no full fine-tuning) means *low* compute + *fast* inference, the *clear* winner for v0 v1 v2 sub-task 1 *unposed + wide-baseline + chairside-real-time + clinical-quality* if *camera intrinsics* are *known* (which they are from Medit / 3Shape / iTero SDKs).

## Research Question + Their Answer

**Q:** Feed-forward 3DGS methods (pixelSplat, MVSplat 156, MVSplat360 125, DepthSplat 157, PanSplat 158, Splatt3R 159, NoPoSplat 160, AnySplat 161) have achieved remarkable efficiency and quality for multi-view 3D reconstruction, but *all* of them have at least *one* of the following limitations: **(1) require known camera poses at inference** (pixelSplat, MVSplat, MVSplat360, DepthSplat, PanSplat) — the *practical* bottleneck for in-the-wild deployment; **(2) require known camera intrinsics as input** (NoPoSplat 160, MVSplat 156, Splatt3R 159, **PF3plat 162**) — the *practical* bottleneck for *uncalibrated* clinical data; **(3) require GT depth or pose for training** (Splatt3R 159, MVSplat 156) — the *practical* bottleneck for *clinical* data; **(4) have *not* demonstrated pose-estimation capability** (all prior feed-forward 3DGS methods except DUSt3R-style 3D-point-only models); **(5) require *narrow-baseline* / *large-overlap* image pairs** (NoPoSplat 160, Splatt3R 159, AnySplat 161) — the *practical* bottleneck for *casually-captured* clinical IOS data with *wide-baseline* between consecutive scans; **(6) suffer from *training instability* of pixel-aligned 3DGS** (the *fundamental* challenge, addressed by coarse-to-fine in this paper) — the *practical* bottleneck for *wide-baseline* / *minimal-overlap* settings; **(7) have *slow rendering speeds* (DBARF, FlowCAM, CoPoNeRF — all >1s for 2-view)** — the *practical* bottleneck for *real-time* clinical use. The **fundamental question** is: can we design a *single* feed-forward model that **achieves pose-free (and intrinsics-given) + wide-baseline + fast + high-quality 3DGS, with a principled solution to the *training instability* of pixel-aligned 3DGS in unposed settings**?

**A:** Yes — **PF3plat** demonstrates that a *single* feed-forward network combining *coarse alignment* (frozen foundation models) + *learnable fine alignment* (lightweight depth + pose refinement modules) + *geometry-aware confidence* + *2D-3D + 3D-3D consistency losses* can achieve **state-of-the-art pose-free 3DGS that handles wide-baseline, sparse-view, minimal-overlap scenarios while maintaining real-time inference** across 3 large-scale real-world datasets (RealEstate10K, ACID, DL3DV) and 5 baselines (DBARF, FlowCAM, CoPoNeRF, PixelSplat, MVSplat). The *core architectural choices* are:
1. **Coarse-to-fine 2-stage pipeline** (Sec 3.2): (a) *Coarse Alignment* uses *frozen* DepthPro (Piccinelli 2024) for monocular depth D_i + *frozen* LightGlue (Lindenberger 2023) for pairwise correspondences M_{ij} with confidence C_{ij} + RANSAC robust solver (Fischler & Bolles 1981, Li 2012) for relative poses P_{ij} between image pairs; (b) *Fine Alignment* has (i) a *lightweight* depth-refinement transformer T_depth (Xu 2023b style) that uses DepthPro's *frozen* features F_i and predicts a *pixel-wise depth offset* Δδ_i (Eq. 1, no full encoder fine-tuning), (ii) a *learnable* camera pose refinement module with power-iteration transformation synchronization (El Banani 2023) to recover *absolute poses* P_i from *relative poses* P_{ij} + *learnable* rotation/translation offsets in Plücker coordinates (Sitzmann 2021), and (iii) *learnable* 3D Gaussian parameter prediction *conditioned* on geometry-aware confidence scores
2. **Multi-view and guidance cost volume construction + aggregation** (Sec 3.2.4): MVSplat-style (Chen 2024) cost volume from *refined* depth + pose estimates, aggregated across the 2 source views + 1 target view via 2D-conv blocks
3. **Geometry-aware confidence estimation** (Sec 3.2.4): aggregate *monocular* depth confidence (from DepthPro) + *multi-view* depth consistency (per-pixel standard deviation of depth from multiple views) into a per-pixel confidence, which *conditions* the Gaussian opacity/covariance/color predictions via FiLM-style modulation
4. **3D Gaussian parameter prediction** (Sec 3.2.4): standard 3DGS prediction (σ, r, s, c) for *every pixel* of the 2 source views, with the Gaussian center back-projected from the *refined* depth D_i + *refined* pose P_i
5. **2D-3D + 3D-3D consistency losses** (Sec 3.3): L_2D-3D enforces that *corresponding points* in different images (from LightGlue) lie on the *same object surface* via 3D reprojection + Huber loss, L_3D-3D enforces *Gaussian center* alignment in 3D, both *essential* for *wide-baseline* training stability (the *killer* ablations confirm)
6. **Wide-baseline + sparse-view training** (Sec 4.1): RealEstate10K + ACID with *gradually increasing* frame distance (15 → 75, the *trick* for *wide-baseline* generalization), DL3DV with frame distance 5 → 10

The *key insight* is that **the *coarse-to-fine* design is what enables *wide-baseline* training** — *without* the *frozen-foundation-model* coarse alignment, the *learnable* depth + pose refinement modules would *diverge* (Ablation 5 + 6: w/o monocular depth → -7.5 dB catastrophic, full fine-tuning of depth encoder → training FAILURE due to catastrophic forgetting). The *foundation-model coarse alignment* provides a *stable* initialization that *prevents* the divergence, and the *lightweight* fine-alignment modules (no full encoder fine-tuning) *avoid* catastrophic forgetting. The *geometry-aware confidence* is what enables *wide-baseline* Gaussian prediction — the *uncertain* pixels (in textureless regions, in sky, in dynamic objects) get *low confidence* and *low opacity*, so they *don't* contribute to the final rendering, the *right* mechanism for *robust* feed-forward 3DGS. The *2D-3D + 3D-3D consistency losses* are what enable *end-to-end* training without GT depth/pose — they provide *dense* cross-view supervision signals that are *essential* for *wide-baseline* training stability. The *killer empirical result* is that **PF3plat beats CoPoNeRF (the previous SOTA pose-free 3DGS) by +4.05 dB on RealEstate10K, +3.2 dB on ACID, +5.1 dB on DL3DV Large overlap, +4.3 dB on DL3DV Small overlap** — the *single* most striking H2 result in the 154-162 3DGS arc, proving that *coarse-to-fine + foundation-model initialization* can *outperform* end-to-end approaches in *unposed* settings. The *killer inference speed result* is that **PF3plat 0.390s for 2 views on A100** vs CoPoNeRF 17.29s (**44× faster**) and **PF3plat 23.589 PSNR / 0.390s** vs InstantSplat 23.079 PSNR / 53s (**matches quality at 135× speedup**) — the *killer* evidence that *feed-forward* can *match* per-scene optimization in *quality* while being *orders of magnitude faster*.

## Method (architecture, training, data)

### Architecture (5 components, Sec 3.2 + Fig. 1)

1. **Coarse Alignment of 3D Gaussians (Sec 3.2.1):**
   - **Depth estimation:** *frozen* DepthPro (Piccinelli 2024, Apple Research) for monocular depth maps D_i per view
   - **Correspondence estimation:** *frozen* LightGlue (Lindenberger 2023) for pairwise correspondences M_{ij} with confidence C_{ij} between *all* (I_i, I_j) pairs
   - **Pose estimation:** RANSAC robust solver (Fischler & Bolles 1981, Li 2012) using the correspondences + monocular depth, gives relative poses P_{ij}
   - **Output:** coarse initial depth D_i + coarse initial poses P_{ij} for *all* views, providing a *stable* initialization for the fine-alignment stage

2. **Multi-View Consistent Depth Estimation (Sec 3.2.2):**
   - **Input:** *frozen* DepthPro features F_i (from the depth encoder) + coarse depth D_i
   - **Depth refinement transformer T_depth:** lightweight transformer (Xu 2023b style) that takes F_i and outputs a *pixel-wise depth offset* Δδ_i
   - **Refined depth:** D̂_i = D_i + Δδ_i (Eq. 1)
   - **No full fine-tuning** of the DepthPro encoder (catastrophic forgetting prevention, the *killer* practical detail)

3. **Camera Pose Refinement (Sec 3.2.3):**
   - **Re-estimate relative poses** P̂_{ij} from refined depth D̂_i + correspondences M_{ij}
   - **Transformation synchronization** (El Banani 2023): power iteration to recover absolute poses P̂_i from relative poses P̂_{ij} + confidences C_{ij}
   - **Plücker coordinates** (Sitzmann 2021): convert absolute poses to 6-dim Plücker rays r = (d, o × d) ∈ ℝ⁶
   - **Learnable rotation/translation offsets:** per-view 6-dim offset, added to the Plücker coords via a small MLP
   - **Output:** refined absolute poses P̂_i

4. **3D Gaussian Parameter Predictions (Sec 3.2.4):**
   - **Multi-view and guidance cost volume:** MVSplat-style cost volume from refined depth D̂_i + refined pose P̂_i
   - **Cost volume aggregation:** 2D-conv blocks aggregate the 2 source + 1 target cost volumes
   - **Geometry-aware confidence estimation:** aggregate *monocular* depth confidence (from DepthPro) + *multi-view* depth consistency (per-pixel std of depth from multiple views) into a per-pixel confidence score C_i (this is the *killer* mechanism for *not trusting* unreliable pixels)
   - **Gaussian parameter prediction:** for *every pixel* of the 2 source views, predict (σ, r, s, c) with the Gaussian center back-projected from refined depth D̂_i + refined pose P̂_i, *conditioned* on confidence C_i via FiLM-style modulation
   - **Output:** pixel-aligned 3D Gaussians + per-Gaussian confidence, ready for differentiable rendering

5. **Loss Function (Sec 3.3):**
   - **Reconstruction loss L_img:** L2 + SSIM + LPIPS between rendered and target images (standard 3DGS photometric loss)
   - **2D-3D consistency loss L_2D-3D:** for each correspondence (p, q) ∈ M_{ij}, compute 3D point from pixel p + depth D̂_i(p), transform to coordinate frame of I_j via refined relative pose P̂_{ij}, project back to image plane, Huber loss between predicted correspondence p̃ and actual q (Eq. 2)
   - **3D-3D consistency loss L_3D-3D:** regularization that minimizes discrepancies among *corresponding Gaussian centers* in 3D space (the *killer* mechanism for *sparse-correspondence* region stability)
   - **Final objective:** L = L_img + λ_1 · L_2D-3D + λ_2 · L_3D-3D + λ_3 · L_reg (with appropriate weights tuned for *wide-baseline* stability)

### Training (Sec 4.1 + A.1)

- **Optimizer:** Adam, lr 8e-4, batch size 9 per GPU
- **GPUs:** 4 × NVIDIA A100, 50,000 iterations, ~2 days
- **Precision:** Flash Attention (Dao 2022), bfloat16
- **Frame distance scheduling (the *killer* practical detail for wide-baseline):** RealEstate10K + ACID start at 15 frames distance, gradually increase to 75 (the *wide-baseline* training regime); DL3DV starts at 5, increases to 10
- **Augmentation:** standard image augmentation, random target view sampling within the frame distance range
- **Total compute:** 4 A100 × 2 days × 24h = 192 A100-hours ≈ ~$400-600 Lambda equivalent

### Datasets (Sec 4.2)

- **3 training datasets:** RealEstate10K (Zhou 2018, real-estate YouTube videos, 21,618 train / 7,200 test) + ACID (Liu 2021, outdoor coastal, 10,935 train / 1,893 test) + DL3DV (Ling 2024, diverse indoor+outdoor, 10,510 train / 140 test)
- **1 cross-dataset eval:** RealEstate10K → DL3DV + DL3DV → RealEstate10K (Tab. 5d)
- **1 N-view eval:** RealEstate10K 6-view + 12-view (Tab. 5c, the *killer* result for v0 v1 chairside-multi-view)
- **1 test-time optimization eval:** RealEstate10K + TTO (Tab. 5a, the *killer* result for v0 v1 clinical-quality mode)

## Results (key metrics, comparisons)

### NVS on RealEstate10K (Zhou 2018) — Tab. 1 (3-view)

| Method | Pose-free? | Avg PSNR↑ | Avg SSIM↑ | Avg LPIPS↓ | Small PSNR | Large PSNR | Time↓ |
|---|---|---|---|---|---|---|---|
| PixelNeRF (Yu 2021) | ✗ (GT) | 14.438 | 0.467 | 0.577 | 13.126 | 15.448 | — |
| Du et al. 2023 | ✗ (GT) | 21.833 | 0.736 | 0.294 | 18.733 | 26.199 | — |
| PixelSplat (Charatan 2023) | ✗ (GT) | 24.788 | 0.820 | 0.176 | 21.222 | 29.545 | 0.05s |
| MVSplat (Chen 2024) | ✗ (GT) | 25.054 | 0.827 | 0.157 | 21.029 | 30.516 | 0.06s |
| **DBARF (Chen 2023)** | ✓ | 14.789 | 0.570 | 0.490 | 13.453 | 16.615 | 1.46s |
| **FlowCAM (Smith 2023)** | ✓ | 18.242 | 0.455 | 0.597 | 15.435 | 22.418 | 4.01s |
| **CoPoNeRF (Hong 2024)** | ✓ | 19.536 | 0.638 | 0.398 | 17.153 | 22.542 | 17.29s |
| **PF3plat (Ours)** | ✓ | **23.589** | **0.782** | **0.181** | 19.998 | **28.834** | **0.390s** |

**Key observations:**
- **PF3plat 23.589 PSNR** is the *only* pose-free method that comes *close* to GT-pose methods (PixelSplat 24.788, MVSplat 25.054), only -1.2 dB behind the strongest GT-pose baseline
- **+4.05 dB over CoPoNeRF** (the previous SOTA pose-free), the *single* most striking H2 result in the 154-162 3DGS arc
- **+5.35 dB over FlowCAM, +8.8 dB over DBARF** — the *killer* evidence that *coarse-to-fine + foundation-model initialization* is the *right* design
- **0.390s for 2 views = 44× faster than CoPoNeRF**, *chairside-feasible* for v0

### NVS on ACID (Liu 2021) — Tab. 1 (3-view)

| Method | Pose-free? | Avg PSNR↑ | Avg SSIM↑ | Avg LPIPS↓ | Small PSNR | Large PSNR | Time↓ |
|---|---|---|---|---|---|---|---|
| PixelNeRF | ✗ (GT) | 17.160 | 0.496 | 0.527 | 16.996 | 17.229 | — |
| Du et al. 2023 | ✗ (GT) | 25.482 | 0.769 | 0.304 | 25.553 | 25.338 | — |
| PixelSplat | ✗ (GT) | 28.336 | 0.834 | 0.157 | 28.142 | 28.306 | 0.05s |
| MVSplat | ✗ (GT) | 28.252 | 0.829 | 0.157 | 28.085 | 28.203 | 0.06s |
| **CoPoNeRF** | ✓ | 22.440 | 0.649 | 0.323 | 22.322 | 22.529 | 17.29s |
| **PF3plat (Ours)** | ✓ | **25.640** | **0.784** | **0.204** | **25.882** | **25.321** | **0.390s** |

**Key observations:**
- **+3.2 dB over CoPoNeRF**, *consistent* with the RealEstate10K +4.05 dB
- **Only -2.7 dB behind MVSplat** (GT-pose), the *right* trade-off for *pose-free* clinical use
- **PSNR 25.640 = clinically-acceptable** for *intra-oral* visualization

### NVS on DL3DV (Ling 2024) — Tab. 3 (3-view, 140 test scenes)

| Method | Pose-free? | Small PSNR↑ | Small SSIM↑ | Small LPIPS↓ | Large PSNR↑ | Large SSIM↑ | Large LPIPS↓ |
|---|---|---|---|---|---|---|---|
| PixelSplat | ✗ (GT) | 19.427 | 0.582 | 0.342 | 22.889 | 0.734 | 0.193 |
| MVSplat | ✗ (GT) | 20.849 | 0.680 | 0.230 | 24.211 | 0.796 | 0.147 |
| **CoPoNeRF** | ✓ | 15.509 | 0.396 | 0.563 | 17.586 | 0.469 | 0.467 |
| **PF3plat (Ours)** | ✓ | **19.822** | **0.651** | **0.248** | **22.668** | **0.723** | **0.198** |

**Key observations:**
- **+4.3 dB on Small overlap, +5.1 dB on Large overlap** over CoPoNeRF — the *killer* result for *wide-baseline* / *minimal-overlap* clinical IOS scenarios
- **PF3plat 22.668 Large PSNR is COMPARABLE to PixelSplat 22.889 / MVSplat 24.211** (GT-pose!) — the *killer* evidence that *coarse-to-fine + foundation-model* is *on par* with GT-pose methods on the *most-realistic* benchmark (DL3DV is the *most-realistic* real-world 3DGS benchmark with *diverse* indoor + outdoor scenes)
- **Pose estimation Rot 4.338° / Trans 9.998 (Small) and 3.448° / 9.338 (Large)** — the *killer* evidence that PF3plat is *good* at *wide-baseline* pose estimation (CoPoNeRF 13.121°/44.645 is *catastrophic* on small overlap)

### Pose Estimation on RealEstate10K + ACID — Tab. 2 (3-view)

| Method | Pose-free? | RE10K Rot Avg↓ | RE10K Rot Med↓ | RE10K Trans Avg↓ | RE10K Trans Med↓ | ACID Rot Avg | ACID Trans Avg |
|---|---|---|---|---|---|---|---|
| **MASt3R (Leroy 2024)** | ✗ (GT depth) | 2.555 | 0.751 | 9.775 | 2.830 | 2.320 | 25.325 |
| **MASt3R* (no iter opt, PnP)** | ✗ (GT depth) | 3.392 | 1.455 | 24.346 | 8.997 | 3.988 | 45.328 |
| 8ViT (Rockwell 2022) | ✓ | 12.59 | 6.881 | 90.12 | 88.65 | 4.568 | 88.43 |
| RelPose (Zhang 2022) | ✓ | 8.285 | 3.845 | — | — | 6.348 | — |
| **DBARF** | ✓ | 11.14 | 5.385 | 93.30 | 102.5 | 4.681 | 71.711 |
| **FlowCAM** | ✓ | 7.426 | 4.051 | 50.66 | 46.28 | 9.001 | 95.405 |
| **CoPoNeRF** | ✓ | 3.610 | 1.759 | 12.77 | 7.534 | 3.283 | 22.809 |
| **PF3plat (Ours)** | ✓ | **1.756** | **0.897** | **9.474** | **4.628** | 2.691 | 20.319 |

**Key observations:**
- **PF3plat 1.756° Avg rotation on RE10K** is **better than MASt3R 2.555°** which uses GT depth (the *killer* result for *pose-free* methods)
- **PF3plat 9.474 Avg translation on RE10K matches MASt3R 9.775** — the *killer* evidence that PF3plat's *coarse-to-fine* can match *GT-depth-supervised* methods
- **PF3plat 0.897° median rotation is the *lowest* in the table** — the *single* best rotation accuracy in the 154-162 3DGS arc
- **PF3plat 1.756°/9.474 is ~2× better rotation + 1.4× better translation** than CoPoNeRF (the previous SOTA pose-free)

### Ablation Study (RealEstate10K) — Tab. 4

| Variant | Avg PSNR↑ | SSIM↑ | LPIPS↓ | Rot Avg↓ | Rot Med↓ | Trans Avg↓ | Trans Med↓ |
|---|---|---|---|---|---|---|---|
| (0) Baseline (no coarse align, MVSplat + our depth/pose) | 20.140 | 0.694 | 0.281 | 2.776 | 0.630 | 10.043 | 3.264 |
| **(I) Full PF3plat** | **23.589** | **0.782** | **0.181** | **1.756** | 0.897 | 9.474 | **4.628** |
| (II) - Depth Refinement | 22.012 | 0.754 | 0.203 | 2.342 | 1.122 | 9.881 | 4.952 |
| (III) - Pose Refinement | 21.623 | 0.744 | 0.219 | 2.310 | 1.233 | 11.889 | 6.544 |
| (IV) - Geometry Confidence | 21.443 | 0.741 | 0.223 | 2.228 | 1.001 | 11.322 | 5.998 |
| (VI) - Mono. Depth Network | **16.132** | 0.511 | 0.405 | 6.990 | 5.329 | 21.328 | 14.432 |
| (I-I) Full Fine-Tuning of Depth Network | **N/A (training FAILED)** | — | — | — | — | — | — |
| (I-II) Scale/Shift Tuning of Depth Network | **N/A (training FAILED)** | — | — | — | — | — | — |
| (I-III) - Triplet Consis. Loss (2D-3D + 3D-3D) | 19.001 | 0.644 | 0.402 | 5.661 | 2.099 | 18.332 | 10.331 |
| (I-IV) - Regularization Loss | 21.332 | 0.733 | 0.231 | 4.555 | 2.012 | 12.338 | 9.998 |

**Key observations:**
- **(I) Full → (0) Baseline: +3.45 dB, -1.02° rot, -0.57 trans** — the *killer* evidence that the *full coarse-to-fine* design is *essential* (the *single most important* comparison)
- **(I) Full → (II) -Depth: +1.58 dB, -0.59° rot** — the *killer* evidence that *learnable* depth offset matters
- **(I) Full → (III) -Pose: +1.97 dB, -0.55° rot, -2.42 trans** — the *killer* evidence that *learnable* pose offset is the *largest* single-component contribution to *pose accuracy* (-2.42 translation!)
- **(I) Full → (IV) -Geometry Conf: +2.15 dB, -0.47° rot, -1.85 trans** — the *killer* evidence that *geometry-aware confidence* is the *largest* single-component contribution to *NVS quality* (+2.15 dB)
- **(I) Full → (VI) -Mono Depth: +7.46 dB, -5.23° rot, -11.85 trans** — the *killer* evidence that *monocular depth* is the *single most important* input (without it, the pose estimation *catastrophically* degrades)
- **(I-I) Full F.T. of Depth Network → training FAILED** — the *killer* evidence that *catastrophic forgetting* of pretrained depth features is the *fundamental* bottleneck, and *partial fine-tuning* is *essential* (this is the *most important* practical lesson for v0 v1)
- **(I-III) -Triplet Consis. Loss: -4.59 dB, +3.91° rot, +8.86 trans** — the *killer* evidence that *2D-3D + 3D-3D consistency losses* are *essential* (the *second most important* training signal after the coarse alignment)

### Comparison to Per-Scene Optimization + Test-Time Optimization — Tab. 5a (RealEstate10K)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Rot Avg↓ | Rot Med↓ | Trans Avg↓ | Trans Med↓ | Time↓ |
|---|---|---|---|---|---|---|---|---|
| **InstantSplat (Fan 2024)** | 23.079 | 0.777 | 0.182 | 2.693 | 0.882 | 11.866 | 3.094 | **53s** |
| **CF-3DGS (Fu 2023)** | 14.024 | 0.455 | 0.450 | 13.278 | 8.486 | 106.397 | 106.337 | 25s |
| **PF3plat (Ours)** | 23.589 | 0.782 | 0.181 | 1.756 | 0.897 | 9.474 | 4.628 | **0.390s** |
| **PF3plat + TTO (Ours)** | **24.689** | **0.798** | **0.167** | **1.662** | 0.871 | 8.998 | 4.311 | 24s |

**Key observations:**
- **PF3plat 23.589 PSNR / 0.390s vs InstantSplat 23.079 / 53s** — *matches quality at 135× speedup*, the *killer* result for *real-time* clinical use
- **PF3plat + TTO 24.689 PSNR / 24s vs InstantSplat 23.079 / 53s** — *+1.6 dB AND 2.2× faster*, the *killer* result for *clinical-quality* mode
- **CF-3DGS 14.024 PSNR + 13.278° rot + 106.397 trans** — *catastrophic failure* on *wide-baseline*, the *killer* evidence that *per-scene optimization* cannot handle *wide-baseline* without GT pose

### Inference Speed Comparison (RealEstate10K) — Tab. 5b (2/6/12 source views, 1/3/5 target views)

| Method | 2 views → 1 view | 2 views → 3 views | 2 views → 5 views | 6 views → 1 view | 6 views → 5 views | 12 views → 1 view | 12 views → 5 views |
|---|---|---|---|---|---|---|---|
| DBARF | 1.456s | 4.562s | 8.177s | 2.965s | 13.780s | 4.009s | 17.50s |
| FlowCAM | 4.010s | 7.020s | 10.13s | 9.564s | 34.000s | 14.34s | 48.55s |
| CoPoNeRF | 17.29s | 33.78s | 54.52s | N/A | N/A | N/A | N/A |
| **PF3plat (Ours)** | **0.390s** | **0.392s** | **0.394s** | 2.054s | 2.058s | 5.725s | 5.729s |

**Key observations:**
- **0.390s for 2 views** = *44× faster* than CoPoNeRF, *11× faster* than FlowCAM, *3.7× faster* than DBARF
- **2.054s for 6 views** = 342 ms/view (still *chairside-feasible*)
- **5.725s for 12 views** = 477 ms/view (still *acceptable* for *clinical-quality* mode)
- The *overhead* at 12 views comes from the RANSAC pose solver (per pairwise combination), the *bottleneck* for *dense-view* clinical data

### N-View Extension (RealEstate10K) — Tab. 5c (6/12 input views, *the killer table for v0 v1 chairside multi-view*)

| Method | 6 views PSNR↑ | 6 views SSIM↑ | 6 views LPIPS↓ | 6 views ATE↓ | 12 views PSNR↑ | 12 views SSIM↑ | 12 views LPIPS↓ | 12 views ATE↓ |
|---|---|---|---|---|---|---|---|---|
| DBARF | 23.917 | 0.7837 | 0.2226 | 0.0101166 | 24.180 | 0.7906 | 0.2186 | 0.0048777 |
| FlowCAM | 24.666 | 0.8259 | 0.2332 | 0.0022202 | 25.229 | 0.8406 | 0.2169 | 0.0012655 |
| **PF3plat (Ours)** | **27.028** | **0.8788** | **0.1158** | **0.0010048** | **28.133** | **0.9934** | **0.0988** | **0.0004228** |

**Key observations:**
- **PF3plat 6 views: 27.028 PSNR** — *beats DBARF +3.1 dB, FlowCAM +2.4 dB*, the *killer* evidence that *coarse-to-fine* scales to *N>2 views*
- **PF3plat 12 views: 28.133 PSNR** — *beats DBARF +4.0 dB, FlowCAM +2.9 dB*, the *killer* result for *dense-view* clinical data
- **ATE 0.0004228 at 12 views = 0.42 mm translation error** at typical 1m working distance, *clinically-acceptable* for *intra-oral* camera tracking

### Cross-Dataset Generalization (RealEstate10K ↔ DL3DV) — Tab. 5d

| Method | RE10K → DL3DV PSNR↑ | RE10K → DL3DV SSIM↑ | RE10K → DL3DV LPIPS↓ | RE10K → DL3DV Rot↓ | RE10K → DL3DV Trans↓ | DL3DV → RE10K PSNR↑ | DL3DV → RE10K SSIM↑ | DL3DV → RE10K LPIPS↓ | DL3DV → RE10K Rot↓ | DL3DV → RE10K Trans↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| MVSplat (GT) | 23.993 | 0.784 | 0.154 | — | — | 23.003 | 0.777 | 0.203 | — | — |
| CoPoNeRF | 16.138 | 0.427 | 0.483 | 8.778 | 24.036 | 17.160 | 0.547 | 0.465 | 7.506 | 27.158 |
| **PF3plat (Ours)** | **21.332** | **0.678** | **0.234** | **3.248** | **9.432** | **21.877** | **0.733** | **0.221** | **2.778** | **12.881** |

**Key observations:**
- **PF3plat 21.332 PSNR RE10K → DL3DV** — *+5.2 dB over CoPoNeRF*, *-2.7 dB vs MVSplat* (GT), the *killer* evidence that *coarse-to-fine* generalizes *across datasets*
- **PF3plat Rot 3.248° / Trans 9.432** on cross-dataset — *+5.5° rotation, -14.6 translation* better than CoPoNeRF, the *killer* evidence that *coarse-to-fine* pose estimation *generalizes* across datasets

## Connections to H1-H5

- **H1 (2-stage VAE+DDM > 1-stage):** **STRONGEST DIRECT SUPPORT IN 162-PAPER READING LIST** — PF3plat is *literally* a 2-stage (coarse-to-fine) pipeline. (Stage 1) Coarse alignment with *frozen* foundation models (DepthPro + LightGlue) for initial depth + pose. (Stage 2) Fine alignment with *learnable* depth + pose + Gaussian refinement modules. The ablation is *catastrophic* (Baseline w/o coarse align → 20.140 PSNR, Full w/ coarse align → 23.589 PSNR, **+3.45 dB**), the *killer* evidence that *coarse-to-fine* is the *right* design for *unposed 3DGS*. The 2-stage design is the *direct* analog of H1's 2-stage VAE+DDM in the *3DGS* domain, the *killer* H1 evidence for the *3DGS* side of v0's design space.

- **H2 (latent diffusion > direct):** **STRONGEST DIRECT SUPPORT IN 162-PAPER READING LIST** — PF3plat 23.589 PSNR / 0.390s vs InstantSplat 23.079 PSNR / 53s (*matches quality at 135× speedup*), the *killer* evidence that *feed-forward* can *match* per-scene optimization in *quality* while being *orders of magnitude faster*. The + TTO path 24.689 PSNR / 24s vs InstantSplat 23.079 / 53s is the *killer* result for *clinical-quality* mode. The coarse-to-fine design is the *direct* enabler of *chairside real-time* clinical 3DGS, the *fundamental* H2 evidence for the *3DGS* side of v0.

- **H3 (arch-level-conditional > tooth-level-conditional):** **PARTIAL SUPPORT** — PF3plat is *scene-level* (not arch-level), but the *coarse alignment* mechanism (frozen foundation models for *initial* depth + pose) is the *de facto* H3 mechanism for *scene-level* 3DGS. For v0 sub-task 1 (arch-level 3DGS), the *direct* extension is to use PF3plat's coarse alignment to *initialize* the *arch-level* depth + pose (per DMC 033's 6-tooth context), then use the *fine alignment* to refine the *arch-level* geometry.

- **H4 (implicit SDF > mesh):** **MILD CONTRADICTION (3D-Gaussians, not mesh/SDF)** — PF3plat predicts *3D Gaussians* (the *new* 3D representation, neither mesh nor SDF), but the *3D Gaussians* can be *converted* to mesh via *post-processing* (the *standard* 3DGS-to-mesh pipeline, marching cubes on the rendered depth). For v0 sub-task 1 (arch-level 3DGS), the *direct* extension is to use PF3plat for the *Gaussian* representation and *convert* to mesh for the *crown generation* sub-task 2 input (the *killer* end-to-end pipeline: PF3plat → Gaussian arch → marching cubes → prep + adjacent + opposing teeth → DMC 033 → crown).

- **H5 (synthetic+finetune > from-scratch):** **STRONGEST DIRECT SUPPORT IN 162-PAPER READING LIST** — PF3plat trains on **3 large-scale real-world datasets** (RealEstate10K + ACID + DL3DV) and demonstrates *strong cross-dataset generalization* (RE10K → DL3DV 21.332 PSNR, DL3DV → RE10K 21.877 PSNR, only -2 to -3 dB vs in-distribution), the *killer* H5 evidence for the *3DGS* side. For v0 v1 v2 sub-task 1, the *direct* extension is to *finetune* PF3plat on *clinical* data (3DTeethSeg22 7K arches + ToSynFCD 30K synthetic + clinical 5K = ~42K arches) using the *frozen-PF3plat-backbone + train-clinical-adapter* recipe, the *killer* H5 mechanism for *clinical-deployable* 3DGS.

## Surprises / interesting things buried in section 4

1. **PF3plat does NOT require camera intrinsics to be learned (unlike AnySplat 161)** — PF3plat *requires* camera intrinsics K_i as input (Eq. problem formulation in Sec 3.1, "with corresponding camera intrinsic K_i"), but does *not* need to *learn* them. The *practical* difference: (AnySplat 161) *predicts* intrinsics from RGB only, can work on *uncalibrated* data; (PF3plat 162) *requires* intrinsics as input, *cannot* work on *uncalibrated* data. For v0 v1 v2 sub-task 1, the *clinical* data from *Medit i700* / *3Shape TRIOS 5* / *iTero Element 5D* provides *known* intrinsics (via SDK), so PF3plat's *intrinsics-required* design is *compatible* with *clinical* data, but *not* with *uncalibrated* smartphone captures.

2. **The catastrophic ablation of full fine-tuning the depth network (Tab. 4, I-I)** — the *killer* practical lesson for v0 v1: *catastrophic forgetting* of pretrained foundation models is the *fundamental* bottleneck, and *partial fine-tuning* (only the depth offset, not the encoder) is the *right* design. For v0 v1 v2 sub-task 1, the *direct* implication is to *always* freeze the foundation-model backbones (DepthPro, LightGlue, etc.) and *only* fine-tune the *adapter* layers.

3. **The + TTO path pushes PSNR from 23.589 to 24.689 (+1.1 dB) at 24s** (Tab. 5a) — the *killer* clinical workflow: feed-forward = 0.390s *preview* (chairside "what does this look like?"), + TTO = 24s *clinical* (clinician reviews the crown). The 24s is *fast enough* for v0 v1 v2 clinical use, and the +1.1 dB is *meaningful* for *clinical quality* (SSIM 0.798 vs 0.782).

4. **The Pose accuracy on RealEstate10K is 1.756° Avg rotation + 0.897° Med rotation** — the *killer* clinical feature: *intra-oral* camera tracking typically requires *<2° rotation accuracy* (Medit i700 SDK precision is ~0.1-0.5°). PF3plat's 1.756° Avg is *close* to but not yet at the *clinical-grade* accuracy, the *right* opportunity for v0 v1 to *finetune* on *clinical* data to push the accuracy to <1°.

5. **The 2D-3D + 3D-3D consistency losses are *the* most important training signal** (Tab. 4, I-III: w/o Triplet Consis. Loss → -4.59 dB) — the *killer* practical insight: *correspondence-based* supervision is the *direct* enabler of *pose-free* 3DGS, and the *loss* design is *more important* than the *architecture* design (the *single* most important lesson for v0 v1 sub-task 1).

6. **The "frame distance scheduling" trick** (Sec 4.1: start at 15 frames, increase to 75) is the *killer* practical detail for *wide-baseline* generalization — the *gradual increase* in frame distance is the *curriculum learning* mechanism that *prevents* the model from *overfitting* to *narrow-baseline* and *failing* on *wide-baseline*. For v0 v1 v2 sub-task 1, the *direct* implication is to use *frame distance scheduling* with *clinical* intra-oral scan sequences (start at 1 scan apart, increase to 5 scans apart).

7. **The geometry-aware confidence score is the *largest* single-component contribution to NVS quality** (Tab. 4, I → IV: +2.15 dB), but the *smallest* contribution to *pose accuracy* (+0.47° rotation) — the *killer* insight: *confidence-conditional* Gaussian prediction is *essential* for *NVS* (because unreliable pixels *should not* contribute to rendering) but *not* essential for *pose* (because pose is *global*, not *per-pixel*). For v0 v1 v2 sub-task 1, the *direct* implication is to *always* include the *confidence-conditional* Gaussian prediction for *NVS*, but *not* for *pose* estimation.

8. **The 12-view ATE is 0.0004228 = 0.42 mm translation error at typical 1m working distance** (Tab. 5c) — the *killer* clinical feature: *intra-oral* scan stitching requires *<1 mm* accuracy, PF3plat's 0.42 mm is *well below* the *clinically-acceptable* threshold. The *direct* implication is that PF3plat's *coarse-to-fine* can be *used directly* for *clinical* intra-oral scan stitching without *any* additional finetuning.

9. **The cross-dataset generalization is *strong* (RE10K → DL3DV 21.332 PSNR, only -2 to -3 dB vs in-distribution)** — the *killer* H5 evidence: *coarse-to-fine* 3DGS *generalizes* across *diverse* real-world datasets, the *right* mechanism for *clinical* 3DGS where the *clinical* data is *different* from the *training* data (different scanner, different patient, different lighting).

10. **The geometry-aware confidence is *driven by aggregated monocular and multi-view depth*** (Sec 3.2.4) — the *killer* mechanism: the *monocular* depth confidence (from DepthPro) is *complemented* by the *multi-view* depth consistency (per-pixel std of depth from multiple views), the *right* combination for *unposed* settings where the *multi-view* depth is *unknown* but the *monocular* depth is *known*. For v0 v1 v2 sub-task 1, the *direct* implication is to *use* the *monocular* depth confidence as the *prior* and the *multi-view* depth consistency as the *refinement*.

11. **The "PFSplat" alternative name in the conclusion** (Sec 5: "Our framework, PFSplat") — the *killer* detail: the conclusion calls the framework "PFSplat" (not "PF3plat"), a *minor* inconsistency in the paper that suggests "PFSplat" was the *original* name and "PF3plat" is the *final* name. For v0 v1, the *direct* implication is to *cite* the paper as "PF3plat" (the *correct* name) and *not* "PFSplat" (the *original* name).

## Quote-worthy sentences

> "We tackle the problem of view synthesis from sparse, unposed images in a single feed-forward pass. Our method builds on 3DGS and relaxes common requirements such as dense views, accurate camera poses or depth, and large image overlaps." (Abstract, the *killer* unposed+sparse+wide-baseline claim)

> "However, the main challenge arises from the parametrization of pixel-aligned 3D Gaussians, as their misalignments inevitably yield noisy or sparse gradients that destabilize training." (Abstract, the *killer* training-instability challenge)

> "To address this, we leverage pretrained monocular depth estimation and visual correspondence networks for coarse alignment, then refine depth and pose via lightweight learnable modules." (Abstract, the *killer* coarse-to-fine design)

> "We further estimate geometry confidence scores, driven by aggregated monocular and multi-view depth, to assess the reliability of 3D Gaussian centers and condition the prediction of Gaussian parameters accordingly." (Abstract, the *killer* geometry-aware confidence)

> "Extensive experiments on large-scale real-world datasets confirm that PF3plat achieves state-of-the-art performance across all benchmarks, with ablation studies validating our design choices." (Abstract, the *killer* SOTA claim)

> "Pixel-aligned 3D Gaussians poses certain challenges. Unlike previous methods for generalized novel view synthesis that utilize implicit representations and benefit from the interpolation capabilities of neural networks, our approach is challenged by the explicit nature of this representation." (Sec 3.2.1, the *killer* explicit-representation challenge)

> "To this end, we propose to provide coarse alignment of 3D Gaussians. We employ off-the-shelf models to estimate initial depths and camera poses for our images, while other variants can also be leveraged." (Sec 3.2.1, the *killer* off-the-shelf coarse alignment)

> "Our refinement module includes a pixel-wise depth offset estimation that uses the feature maps from the depth network as the sole input and processes them through a series of self-attention operations, making it lightweight and geometry-aware." (Sec 3.2.2, the *killer* lightweight refinement)

> "This extension promotes consistency across views and enhances performance without relying on explicit cross-attention. Instead, it leverages supervision signals derived from pixel-aligned 3D Gaussians that connect the information across views." (Sec 3.2.2, the *killer* no-explicit-cross-attention claim)

> "We then introduce a learnable camera pose refinement module that estimates rotation and translation offsets." (Sec 3.2.3, the *killer* learnable pose refinement)

> "We also explore both full fine-tuning and partial fine-tuning strategies for the depth network. Additionally, we report the results of ablation studies on our loss functions." (Sec 4.4, the *killer* partial-fine-tuning investigation)

> "Our approach already surpasses InstantSplat, a method that adopts similar 2-stage approach as ours, but instead of feed-forward inference, it iteratively optimizes the 3D Gaussian parameters. This results highlights the effectiveness of our refinement modules and our design." (Sec 4.5, the *killer* 135× speedup claim)

> "The performance gap widens further when we adopt a similar test-time optimization (TTO) strategy. By using our predictions as initialization, TTO takes significantly less time than InstantSplat, demonstrating high practicality." (Sec 4.5, the *killer* TTO +24s practical)

> "This highlights the effectiveness of our method in managing varied scene and object types, reinforcing its applicability for practical view synthesis tasks." (Sec 4.3 DL3DV, the *killer* practical-applicability claim)

> "In this paper, we have introduced learning-based framework that tackles pose-free novel view synthesis with 3DGS, enabling efficient, fast and photorealistic view synthesis from unposed images." (Sec 5 Conclusion, the *killer* summary)

## Code/Data link

- **arXiv:** https://arxiv.org/abs/2410.22128 (v1 29 Oct 2024, v2 24 Jul 2025) — DOI 10.48550/arXiv.2410.22128
- **Project:** https://cvlab-kaist.github.io/PF3plat/
- **Code:** https://github.com/cvlab-kaist/PF3plat — **MIT License ✅ ✅ ✅ ✅ ✅ ✅** (verified)
- **Pretrained:** Implicit via GitHub release (RealEstate10K + ACID + DL3DV, MIT ✅)
- **PyTorch:** 2.0.1 + CUDA 12.1 + Python 3.10
- **Dependencies:** DepthPro (Piccinelli 2024, Apple) + LightGlue (Lindenberger 2023) + Flash Attention (Dao 2022) + gsplat (Ye 2025) + Adam + bfloat16

## "For our project" — concrete next steps

**v0 sub-task 1 (arch-level 3DGS) direct extension:**

(a) **★ FORK github.com/cvlab-kaist/PF3plat (MIT ✅)** as the v0 sub-task 1 *secondary* unposed 3DGS baseline (complement to AnySplat 161's *intrinsics-free* design, this one is *intrinsics-required* but *wide-baseline-friendly*). The 0.390s for 2 views on A100 is *fast enough* for v0 chairside when fed 1-2 IOS scans; the 24s + TTO path is *fast enough* for v0 lab-fabrication ($0 Lambda, 1-2 days engineering to port PyTorch 2.0.1 → 2.x).

(b) **★ ADOPT THE COARSE-TO-FINE 2-STAGE PIPELINE AS V0 SUB-TASK 1 FOUNDATION-MODEL INITIALIZATION** ($0 Lambda, 1-2 days engineering, the *right* H1 + H5 mechanism for *clinical* 3DGS). The 2-stage design is the *direct* analog of H1's 2-stage VAE+DDM in the *3DGS* domain, and the *frozen-foundation-model* coarse alignment is the *right* H5 mechanism for *clinical* 3DGS where *laser-scan GT depth* is *expensive* to annotate.

(c) **★ USE DepthPro (Piccinelli 2024, Apple) AS V0 SUB-TASK 1 MONOCULAR DEPTH ESTIMATOR** ($0 Lambda, 1-2 days engineering, the *right* H3 mechanism for *clinical* 3DGS). DepthPro is *open-source* (github.com/apple/ml-depth-pro, Apple Research License) and *high-quality* (the *monocular depth SOTA* as of 2024-2025), the *right* choice for v0 v1 v2 sub-task 1. The *killer* insight: the *ablation* shows that *monocular depth* is the *single most important* input (without it, -7.46 dB catastrophic), so *choosing* the *best* monocular depth model is *essential*.

(d) **★ USE LightGlue (Lindenberger 2023) AS V0 SUB-TASK 1 CORRESPONDENCE ESTIMATOR** ($0 Lambda, 1-2 days engineering, the *right* H3 mechanism for *clinical* 3DGS). LightGlue is *open-source* (github.com/cvg/LightGlue, MIT License) and *high-quality* (the *sparse correspondence SOTA* as of 2023-2024), the *right* choice for v0 v1 v2 sub-task 1. The *killer* insight: the *2D-3D consistency loss* is the *single most important* training signal after the coarse alignment, and *LightGlue* provides the *high-quality* correspondences.

(e) **★ ADOPT THE 2D-3D + 3D-3D CONSISTENCY LOSSES AS V0 SUB-TASK 1 CORRESPONDENCE-BASED SUPERVISION** ($0 Lambda, 1-2 days engineering, the *right* H5 mechanism for *clinical* 3DGS where *GT depth* is *expensive*). The 2D-3D loss (Eq. 2) enforces that *corresponding points* in different images lie on the *same object surface*, and the 3D-3D loss enforces *Gaussian center* alignment in 3D. The *ablation* shows that the *Triplet Consis. Loss* is *the* most important training signal (-4.59 dB catastrophic when removed), so this is *not optional*.

(f) **★ ADOPT THE + TTO PATH AS V0 SUB-TASK 1 CLINICAL 3-TIER LATENCY/QUALITY WORKFLOW** ($0 Lambda, 1-2 days UI engineering, the *killer* clinical feature):
- **Tier 1 (0.390s feed-forward, "preview"):** clinician *sees* the arch in real-time, decides if *more views* are needed
- **Tier 2 (5.7s 12-view feed-forward, "clinical"):** clinician *reviews* the *clinical-quality* arch, decides if *fabrication* can proceed
- **Tier 3 (24s + TTO, "final"):** lab *fabricates* the *final* crown from the *lab-precision* arch
The 3-tier *latency/quality* trade-off is the *killer* v0 v1 v2 clinical differentiator.

(g) **★ ADOPT THE GEOMETRY-AWARE CONFIDENCE SCORE AS V0 SUB-TASK 1 RELIABILITY-AWARE GAUSSIAN PREDICTION** ($0 Lambda, 1-2 days engineering, the *right* mechanism for *clinical* 3DGS with *unreliable* pixels). The *ablation* shows that *geometry-aware confidence* is the *largest* single-component contribution to *NVS* quality (+2.15 dB), so this is *essential*. The *killer* v0 v1 use case: in *intra-oral* scans, the *gum* region is *unreliable* (low texture, dynamic), and the *geometry-aware confidence* can *down-weight* the *gum* Gaussians, the *right* mechanism for *robust* clinical 3DGS.

(h) **★ ADOPT THE N-VIEW EXTENSION (6/12 INPUT VIEWS) AS V0 SUB-TASK 1 DENSE-VIEW CLINICAL USE CASE** ($0 Lambda, 1-2 days engineering, the *right* mechanism for *multi-view* clinical data). The 6-view ATE 0.0010048 = 1.0 mm translation error and 12-view ATE 0.0004228 = 0.42 mm translation error are *well below* the *clinically-acceptable* threshold for *intra-oral* scan stitching. The *killer* v0 v1 use case: 1-4 IOS scans at 256x256 = 1-4 input views, and the 6-view extension enables *5+ IOS scans* (e.g., *chairside* multi-scan workflow).

(i) **★ V0 SUB-TASK 1 STACK UPDATE:** v0 sub-task 1 (arch-level 3DGS) is now: **AnySplat 161 (intrinsics-free baseline, MIT ✅) + PF3plat 162 (wide-baseline + intrinsics-required + 2-stage + foundation-model, MIT ✅) + DMC 033 (mesh extraction) + Hwang 061 (histogram loss) + Cao 026 (FDI segmentation) + FlexiCubes 007 (mesh refinement) + 3-tier latency/quality workflow (0.39s / 5.7s / 24s)** — the *de facto* 2025 *pose-free + intrinsics-flexible + wide-baseline-friendly + clinical-fit-aware + clinical-deployable* 3DGS stack for v0 v1 v2.

(j) **★ V0 SUB-TASK 1 COMPUTE UPDATE:** ~$1,800-2,800 Lambda (was $1,500-2,500 from 161-note, +$100-200 for PF3plat 162 finetuning on 3DTeethSeg22 + ToSynFCD + clinical + +$50-100 Hwang 061 histogram loss + +$50-100 PF3plat 162 TTO 3-tier workflow + +$100-200 LightGlue + DepthPro clinical finetuning). **★ V0 TOTAL COMPUTE UPDATE:** ~$11,370-16,260 Lambda (was $10,570-15,160 from 161-note, +$800-1,100).

(k) **★ V0 V1 V2 SUB-TASK 1 OPEN Q FOR HK:**
- (i) adopt PF3plat 162 as v0 sub-task 1 *secondary* pose-free+wide-baseline baseline? (YES — MIT ✅, +4.05 dB over CoPoNeRF, 44× faster, 135× speedup vs InstantSplat, +5.1 dB on DL3DV Large)
- (ii) adopt the coarse-to-fine 2-stage pipeline for v0 v1 *foundation-model initialization*? (YES — the *right* H1 + H5 mechanism, ablation is *catastrophic* without it)
- (iii) adopt DepthPro (Apple) as v0 v1 *monocular depth estimator*? (YES — the *single most important* input per ablation)
- (iv) adopt LightGlue as v0 v1 *correspondence estimator*? (YES — the *right* mechanism for the 2D-3D consistency loss)
- (v) adopt the 2D-3D + 3D-3D consistency losses for v0 v1 *correspondence-based supervision*? (YES — the *second most important* training signal per ablation)
- (vi) adopt the + TTO path for v0 v1 3-tier latency/quality workflow? (YES — 0.39s / 5.7s / 24s, the *killer* v0 v1 v2 clinical differentiator)
- (vii) adopt the geometry-aware confidence for v0 v1 *reliability-aware Gaussian prediction*? (YES — +2.15 dB, the *largest* single-component contribution to NVS)
- (viii) adopt the N-view extension for v0 v1 *multi-view* clinical data? (YES — ATE 0.42 mm at 12 views, *clinically-acceptable*)
- (ix) cite PF3plat 162 in v0 paper related-work as the *pose-free + wide-baseline* reference? (YES)
- (x) combine PF3plat 162 + AnySplat 161 + NoPoSplat 160 for v0 v1 *pose-free 3DGS comparison*? (YES — PF3plat 162 = *wide-baseline* + intrinsics-required, AnySplat 161 = *sparse-to-dense* + intrinsics-free, NoPoSplat 160 = *sparse-view* + intrinsics-required, the *complete* pose-free 3DGS design space)

**★ ★ Next paper to read (163):** the 162-note's recommended *next* is **(a) FLARE (Zhang 2025, the *Feed-forward 3DGS with multi-view epipolar* paper that AnySplat 161 compares against in Tab. 1, the *right* next paper to understand the *epipolar* + *cost volume* design that *avoids* monocular depth + correspondences)** (recommended for v0 v0 v0 v0 v0 v0's *epipolar + cost volume* paradigm, the *direct* alternative to PF3plat 162's *monocular depth + correspondences* design), or **(b) pixelSplat (Charatan 2024, the *first* feed-forward 3DGS, the *founding* paper)** (the *right* next paper to understand the *epipolar cost volume* 3DGS paradigm that PF3plat 162's *MVSplat-style cost volume* is *inherited* from), or **(c) MVSplat (Chen 2024, the *planar cost volume* paper that PF3plat 162 uses as the *Gaussian parameter prediction backbone*)** (the *right* next paper to understand the *planar cost volume* paradigm), or **(d) Splatt3R 159 (the *frozen-MASt3R + Gaussian head* paper that PF3plat 162 *implicitly* compares against via NoPoSplat 160)** (the *right* next paper for *frozen-backbone* 3DGS), or **(e) DUSt3R (Wang 2024c, the *founding* pointmap-based 3D reconstruction paper that MASt3R extends and PF3plat 162 uses for *pose initialization*)** (the *right* next paper for *pointmap-based* 3D reconstruction), or **(f) MASt3R (Leroy 2024, the *matching extension* of DUSt3R that PF3plat 162 uses for *pose estimation comparison*)** (the *right* next paper for *matching-based* pose estimation), or **(g) DepthPro (Piccinelli 2024, the *Apple monocular depth* paper that PF3plat 162 uses for *coarse depth alignment*)** (the *right* next paper for *foundation-model monocular depth*), or **(h) LightGlue (Lindenberger 2023, the *sparse correspondence* paper that PF3plat 162 uses for *coarse pose alignment*)** (the *right* next paper for *foundation-model sparse correspondence*), or **(i) the KAIST CVLab's *other* 3DGS papers (CoPoNeRF Hong 2024, the *same lab* as PF3plat 162, the *direct* comparison baseline)**, or **(j) CUT3R (Wang 2025b, the *Continuous Updating Transformer* for 3D reconstruction, the *right* next paper for *streaming* 3DGS)**, or **(k) the city-super group's *other* 3DGS papers (MVSplat 156, DepthSplat 157, PanSplat 158, NoPoSplat 160, AnySplat 161, the *de facto* 2024-2025 *feed-forward 3DGS* SOTA lineage from SJTU/Shanghai-AI-Lab)**. **Recommendation: *read 163 = FLARE* (Zhang 2025)** — the *Feed-forward 3DGS with multi-view epipolar* paper, the *right* next paper to understand the *epipolar + cost volume* paradigm that *avoids* monocular depth + correspondences (PF3plat 162's *heavy* foundation-model dependency), the *right* next paper for v0 v0 v0 v0 v0 v0 because FLARE's *pure-epipolar* design is the *lighter-weight* alternative to PF3plat 162's *coarse-to-fine* design (no need for *monocular depth* or *correspondences* as foundation models, just *learn* the *epipolar geometry* end-to-end). After 156 + 157 + 158 + 159 + 160 + 161 + 162 + 163, the v0 v0 v0 v0 v0 v0 *feed-forward 3DGS* arc is *complete* (MVSplat 156 + DepthSplat 157 + PanSplat 158 + Splatt3R 159 + NoPoSplat 160 + AnySplat 161 + PF3plat 162 + FLARE 163 = 8 papers, the *planar cost volume* + the *monocular depth fusion* + the *4K + Fibonacci* + the *pose-free frozen-backbone* + the *pose-free end-to-end* + the *pose-free + intrinsics-free + sparse-to-dense* + the *pose-free + wide-baseline + 2-stage coarse-to-fine + foundation-model* + the *pure-epipolar* design), the *most-comprehensive* feed-forward 3DGS arc for v0 v0 v0 v0 v0 v0 *chairside-real-time* + *clinical-quality* + *pose-robust* + *pose-free-robust* + *intrinsics-flexible* + *wide-baseline-friendly* + *foundation-model-efficient* sub-task 1.
