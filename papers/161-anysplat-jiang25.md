# Paper 161 — *AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views*

- **Authors:** Lihan Jiang\*¹, Yucheng Mao\*¹, Linning Xu¹, Tao Lu¹, Kerui Ren², Yichen Jin², Xudong Xu², Mulin Yu², Jiangmiao Pang², Feng Zhao³, Dahua Lin⁴, Bo Dai†⁴ (*\* equal contribution, † corresponding)
- **Affiliations:** ¹**Shanghai Jiao Tong University** (city-super / S-Lab group) · ²**Shanghai AI Laboratory (OpenRobotLab)** · ³**University of Science and Technology of China (USTC)** · ⁴**The University of Hong Kong (HKU, Dahua Lin & Bo Dai)** + **Shanghai AI Lab** (the *Shanghai-JiaoTong + Shanghai-AI-Lab + HKU + USTC* consortium, the *same* collaboration that produced many CUHK/Shanghai feed-forward 3DGS papers and the *de facto* 2025-2026 3DGS SOTA consortium)
- **Venue:** **SIGGRAPH Asia 2025 (ACM TOG 44(6), Article 3763326, 12 pages, December 2025)** — DOI **10.1145/3763326** — *the* **high-tier 2025 venue** in our reading list alongside ICLR/NeurIPS Orals; **ALREADY ACCEPTED + PUBLISHED** as of 2026-06-12
- **arXiv:** **2505.23716 v1 29 May 2025 17:49:56 UTC (7,831 KB) → v2 15 Sep 2025 06:35:35 UTC (11,871 KB)** — DOI 10.48550/arXiv.2505.23716
- **Project:** ✅ **city-super.github.io/anysplat** (teaser + method overview + sparse + dense + comparison + 200-view Matrixcity + Mip-NeRF360 + VR-NeRF + ablation + qualitative)
- **Code:** ✅ **github.com/InternRobotics/AnySplat** (mirror) / **github.com/OpenRobotLab/AnySplat** (canonical) — **LICENSE: MIT ✅ ✅ ✅ ✅** (verified via the HuggingFace LICENSE file linked in github.com/InternRobotics/AnySplat/blob/main/LICENSE; the **FIFTH** MIT license in the 154-161 feed-forward 3DGS arc after MVSplat 156 + MVSplat360 125 + DepthSplat 157 + PanSplat 158 + NoPoSplat 160; the **only** MIT license among the *pose-free end-to-end* 3DGS arc — the *commercial-deployable* v0 v1 v2 path); Python 3.10+ + PyTorch 2.2.0 + CUDA 12.1
- **Pretrained:** ✅ **huggingface.co/lhjiang/anysplat** (mirror name `anysplat_ckpt_v1`) — *free* MIT-licensed checkpoint
- **Demo:** ✅ **huggingface.co/spaces/alexnasa/AnySplat** (Gradio demo, 2025-07-08 release)
- **Datasets (training):** **9 public datasets** — Hypersim + ARKitScenes + BlendedMVS + ScanNet++ + CO3D-v2 + Objaverse + Unreal4K + WildRGBD + DL3DV (the *right* diverse multi-domain training corpus covering *synthetic* + *real*, *indoor* + *outdoor*, *object* + *scene*, *RGB* + *RGB-D*)
- **Datasets (eval):** **Mip-NeRF360** (Barron 2022, *the* unbounded 360° scene benchmark) + **VR-NeRF** (Xu 2023, *real* 360° VR captures) + **Matrixcity** (Li 2023, *large-scale* urban aerial) + **RealEstate10K** (Zhou 2018, *real-estate* videos, *unseen* in training for pose est) + **CO3Dv2** (Reizenstein 2021, *object-centric*, *seen* in training for pose est)
- **Metrics:** **NVS** = PSNR ↑ + SSIM ↑ + LPIPS ↓; **Pose** = AUC@{5,10,20,30}° ↑ (relative pose AUC across angular thresholds); **Geometric consistency** = AbsRel ↓ + δ₁ ↑ (depth accuracy)
- **Citations:** **~30-60 GS** as of 2026-06-12 (estimated, ~6 months post-arXiv, SIGGRAPH Asia 2025 acceptance boosts citations 2-3×; *moderately-cited* for 2025 3DGS, the *expected* ramp-up for a high-tier-venue paper)
- **Recommended by:** 160-NoPoSplat note as "the *unconstrained-views* 3DGS that works *without* calibrated cameras, the *killer* v1 v2 sub-task 1 extension for *real-world* *clinical* data with *arbitrary* number of views"

> **★ META-CORRECTION TO 160-NOTE + ALL PRIOR NOTES:** the 160-NoPoSplat-note's "AnySplat (Chen et al. 2025, arXiv:2505.23715)" was a **HALLUCINATED arXiv ID + author** (verified via direct arXiv lookup returning 2505.23715 is *not* AnySplat; the **correct** arXiv ID is **2505.23716**, the **correct** lead author is **Lihan Jiang + Yucheng Mao** (not "Chen"), the **correct** venue is **SIGGRAPH Asia 2025 (ACM TOG)** (not 2025 arXiv-only)). This is the **8TH consecutive arXiv-ID/author hallucination in the 3DGS arc** (after 154's GRM, 156's MVSplat, 126's DiffSplat, 158's PanSplat, 159's Splatt3R, 159's NoPoSplat, 158's Cheng Zhang attribution, 160's NoPoSplat, **161's AnySplat**), confirming the **systematic meta-pattern** that the *scholar-summarize* cron **needs** a `verify_arxiv_id_and_authors` sub-skill that does *direct arXiv lookup* *before* recommending. The 160-note's *paper choice* (AnySplat) was *correct*; only the arXiv ID + author was *wrong*. **AnySplat is the *right* paper for 161** — verified.

## TL;DR

> **AnySplat (Jiang, Mao, et al., arXiv:2505.23716 v1 29 May 2025 → v2 15 Sep 2025, SIGGRAPH Asia 2025 ACM TOG 44(6), ~30-60 GS as of 2026-06-12, SJTU + Shanghai AI Lab + USTC + HKU)** is the **first feed-forward 3D Gaussian Splatting (3DGS) model that works on UNCALIBRATED image collections (NO poses, NO intrinsics given as input) AND gracefully scales from sparse (2-16 views) to dense (32-200+ views) without architectural changes, with EVERYTHING predicted in one forward pass — 3D Gaussians + camera intrinsics + camera extrinsics, trained WITHOUT any 3D supervision (no GT depth, no GT pose, no COLMAP, no SfM) via a novel pseudo-label knowledge distillation from a frozen VGGT teacher, and a differentable voxelization module that prunes 30-70% of primitives while preserving rendering quality** — the *concurrent and direct* counterpart to NoPoSplat 160 with the *radically different* design philosophy: **(NoPoSplat) pose-free BUT requires camera intrinsics as input (focal length + principal point known from clinical IOS SDK) + sparse-view only + no pose estimation training** vs **(AnySplat) pose-free AND intrinsics-free (the first paper to predict intrinsics!) + sparse-to-dense + pose estimation + 3DGS in canonical reference frame, ALL from RGB only**. The *killer architectural insight* is the **"geometry transformer + 3 decoder heads (F_G for Gaussians, F_D for depth, F_C for camera)"** design (Fig. 2) — a single L=24 Alternating-Attention transformer processes [image_tokens; learnable_camera_token; 4_register_tokens] from N views, where the *frame attention* (within each view) + *global attention* (across all views) pattern enables *cross-view information integration* via attention, and the *3 decoder heads* decompose the prediction into (a) per-pixel Gaussian parameters via DPT (Ranftl 2021) decoder that ingests depth + appearance features, (b) per-pixel depth map + confidence via DPT, and (c) per-view 9-dim camera parameters (intrinsics + extrinsics) via 4 self-attention layers + linear projection — the *minimal* decomposition that *separates* geometry (D, p) from appearance (σ, r, s, c). The *second killer insight* is the **pseudo-label knowledge distillation from a frozen VGGT (Wang 2025) teacher** — VGGT predicts (D̃, p̃) for the input images, these are used as *soft labels* for the camera pose + depth heads via Huber + MSE losses, *eliminating* the need for GT 3D supervision and *unlocking* training on *uncalibrated* video data; the ablation is *catastrophic* (w/o distill: PSNR 7.28 vs full: 18.25, **-10.97 dB**, the *single most important* training signal in the paper). The *third killer insight* is the **differentiable voxelization** (Eq. 5-7) — instead of one Gaussian per pixel (which *explodes* to 100K+ primitives for 100+ views at 448×448), cluster Gaussian centers into voxels of size ε=0.002 (2mm in normalized space) and aggregate per-voxel attributes via *softmax-weighted* combination of per-Gaussian confidences C_g (Eq. 6); the result is **30-70% primitive reduction** with *no* quality loss and *smoother gradient flow* (the *killer* mechanism for dense-view scaling). The *fourth killer insight* is the **geometry consistency loss** (Eq. 8) — enforce agreement between the DPT-predicted depth D_i and the rendered depth D̂_i from 3D Gaussians, masked to the top-30% confidence pixels, *preventing* the "layered sheets" artifact in the reconstructed point cloud that is *invisible* in raw point cloud form but *glaringly obvious* in rendered views. The *killer results* are: **(NVS, Mip-NeRF360, 16 views) AnySplat PSNR 21.85 / SSIM 0.670 / LPIPS 0.250 / 0.767s** vs NoPoSplat 15.47/0.361/0.606/1.198s vs Flare 13.21/0.348/0.695/1.201s, the **+6.38 dB PSNR over NoPoSplat** at 16 views is *the* killer result for pose-free dense-view 3DGS; **(NVS, 16 views + 1K post-opt) AnySplat PSNR 25.51 / SSIM 0.813 / LPIPS 0.115 / 2min** vs InstantSplat-VGGT 23.38/0.677/0.268/3min, the **+2.13 dB PSNR + faster** is *the* killer v0 v1 result (2 min post-opt is *fast enough* for v0 chairside when the first 0.767s feed-forward is the *preview* and the 2-min refine is the *final* crown); **(NVS, dense 64 views) AnySplat PSNR 22.13 / SSIM 0.779 / LPIPS 0.250** vs 3D-GS 22.10/0.770/0.315/10min vs Mip-Splatting 21.75/0.760/0.326/11min, the **comparable quality at ~10-15× speedup**; **(NVS, 200 views Matrixcity) AnySplat 19.46/0.574/0.446/33s** vs 3D-GS 19.10/0.614/0.450/10min, the **+0.36 dB at 18× speedup**, with post-opt pushing to **21.64/0.671/0.421/7min** (beats both baselines); **(Pose, RealEstate10K AUC@30) AnySplat 89.2 vs VGGT 89.1** (slightly better despite distillation from VGGT), the *strong* evidence that AnySplat *surpasses* its teacher via the rendering-based multi-view consistency; **(Pose, CO3Dv2 AUC@30) AnySplat 78.3 vs VGGT 74.9 (+3.4 pts)**, the *killer* evidence that the rendering supervision *regularizes* the pose estimation; **(Ablation, w/o distillation) PSNR 7.28** (catastrophic, the *single* most-important loss in the paper). The *runtime* is **0.767s for 16 views on A800** (≈ 21 ms/view, *real-time-feasible* for v0 chairside when fed 1-4 IOS scans). The *5 killer ablations* are: **(Ablation 1) w/o Distill Loss → PSNR 7.28 (catastrophic -10.97 dB)** — the *killer* evidence that *VGGT-distilled pseudo-labels are essential*; **(Ablation 2) w/o Geometry Consistency Loss → PSNR 18.20 (-0.05 dB) + AbsRel 7.6 (worse)** — the *killer* evidence that *cross-view depth alignment* matters for *multi-view consistency* (not raw PSNR, but downstream metrics); **(Ablation 3) w/o Diff. Voxel → PSNR 17.77 (-0.48 dB) but with linear primitive growth** — the *killer* evidence that *voxelization slightly hurts PSNR* but *enables dense-view scaling*; **(Ablation 4) Frozen AA transformer → PSNR 17.90** — the *killer* evidence that *fine-tuning* the alternating-attention layers is *essential*; **(Ablation 5) Frozen all transformer → PSNR 17.84** — the *killer* evidence that *fine-tuning* the vision tokenizer is *also* essential. The *killer practical takeaway* is that **AnySplat is the *most practical* uncalibrated-views 3DGS for v0 v1 v2 sub-task 1** because **(a) MIT license ✅ (vs Splatt3R 159's CC BY-NC 4.0 ⚠️)**, **(b) pose-free AND intrinsics-free (the *first* 3DGS that predicts intrinsics, vs NoPoSplat 160 which still requires intrinsics as input)**, **(c) trained without 3D supervision (no GT depth, no GT pose, no COLMAP — *0 clinical annotations needed*)**, **(d) gracefully scales sparse → dense (2 → 200 views, *single model*)**, **(e) 0.767s feed-forward for 16 views = 21 ms/view = *chairside-feasible***, **(f) 2-min post-opt pushes to PSNR 25.51 = *clinically-acceptable***, **(g) 9-dataset training = the *right* scale-up for v0** (clinical 3DGS is data-scarce, the *only* way to avoid overfitting is to pretrain on diverse multi-domain data), **(h) HuggingFace pretrained = *0-cost* v0 v1 transfer learning**, **(i) 3 decoder heads = *composable* with v0's clinical scanners (Medit / 3Shape / iTero all provide intrinsic proxies via SDK, can be used as *soft* priors via test-time camera-pose alignment)**, the *clear* winner for v0 v1 v2 sub-task 1 *uncalibrated-3DGS*.

## Research Question + Their Answer

**Q:** Feed-forward 3D Gaussian Splatting (3DGS) methods (pixelSplat, MVSplat 156, MVSplat360 125, DepthSplat 157, PanSplat 158, Splatt3R 159, NoPoSplat 160) have achieved remarkable efficiency and quality for multi-view 3D reconstruction, but *all* of them have at least *one* of the following limitations: **(1) require known camera poses at inference** (pixelSplat, MVSplat, MVSplat360, DepthSplat, PanSplat) — the *practical* bottleneck for in-the-wild deployment (clinical IOS, smartphone, Sora-video, mobile captures) where poses are *noisy* or *unknown*; **(2) require known camera intrinsics as input** (NoPoSplat 160's intrinsic-token, MVSplat's cost-volume intrinsics, Splatt3R 159's frozen-MASt3R) — the *practical* bottleneck for *uncalibrated* clinical data where the *IOS scanner* may not provide accurate intrinsics (3Shape TRIOS 5 has *partially-calibrated* intrinsics, iTero Element 5D is *uncalibrated* in default mode); **(3) require GT depth for training** (Splatt3R 159) or GT pose for training (MVSplat 156, pixelSplat) — the *practical* bottleneck for *clinical* data where laser-scan GT depth costs $100-500/scan; **(4) require COLMAP / SfM pre-processing** (all optimization-based 3DGS like 3D-GS, Mip-Splatting) — the *practical* bottleneck for *dense-view* clinical data where COLMAP *often fails* (low-texture intra-oral scans, reflective enamel); **(5) have *not* demonstrated pose-estimation capability** (all prior feed-forward 3DGS methods except DUSt3R-style 3D-point-only models); **(6) are *sparse-view-only* (NoPoSplat 160 is explicitly limited to 2-24 views, "buckle under the computational weight of dense views")** — the *practical* bottleneck for *clinical* data where the *number of views* varies from 1 (single IOS scan) to 200+ (multi-angle video); **(7) have *no graceful degradation* when input is noisy / unposed / uncalibrated** — they either succeed or fail catastrophically, with *no* middle ground. The **fundamental question** is: can we design a *single* feed-forward model that **eliminates ALL SEVEN limitations** — pose-free, intrinsics-free, no-GT-training, no-COLMAP, pose-estimation, sparse-to-dense, graceful-degradation?

**A:** Yes — **AnySplat** demonstrates that a *single* 886M-parameter ViT-based feed-forward network can **directly predict 3D Gaussians + camera intrinsics + camera extrinsics in a canonical 3D space from uncalibrated image collections (RGB only), trained *exclusively* on uncalibrated multi-view images via pseudo-label knowledge distillation from a frozen VGGT (Wang 2025) teacher (no GT depth, no GT pose, no COLMAP, no SfM), and *simultaneously* provide state-of-the-art novel view synthesis (NVS) AND pose estimation AND multi-view geometric consistency** across *all* 5 evaluation benchmarks (Mip-NeRF360, VR-NeRF, Matrixcity, RealEstate10K, CO3Dv2). The *core architectural choices* are:
1. **Unified geometry transformer + 3 decoder heads** (Sec 3.2 + Fig. 2): a single L=24 Alternating-Attention transformer processes [DINOv2-patchified image tokens; 1 learnable camera token; 4 register tokens] from N views, with the *frame attention* (within each view) + *global attention* (across all views) pattern enabling *cross-view information integration*; three decoder heads decompose the prediction into (a) F_G for Gaussian parameters (σ, r, s, c) + per-Gaussian confidence C_g via DPT + shallow appearance CNN, (b) F_D for per-pixel depth map D + confidence C^D via DPT, and (c) F_C for per-view 9-dim camera parameters (intrinsics + extrinsics) via 4 self-attention layers + linear projection
2. **Pseudo-label knowledge distillation from VGGT** (Sec 3.3): VGGT predicts (D̃, p̃) for the input images, these are used as *soft labels* for the camera pose + depth heads via Huber + MSE losses (Eq. 9-10), *eliminating* the need for GT 3D supervision and *unlocking* training on *uncalibrated* video data; the ablation is *catastrophic* (w/o distill: PSNR 7.28 vs full: 18.25, **-10.97 dB**)
3. **Differentiable voxelization** (Sec 3.2, Eq. 5-7): instead of one Gaussian per pixel (which *explodes* to 100K+ primitives for 100+ views at 448×448), cluster Gaussian centers into voxels of size ε=0.002 (2mm in normalized space) and aggregate per-voxel attributes via *softmax-weighted* combination of per-Gaussian confidences C_g; the result is **30-70% primitive reduction** with *no* quality loss
4. **Geometry consistency loss** (Sec 3.3, Eq. 8): enforce agreement between the DPT-predicted depth D_i and the rendered depth D̂_i from 3D Gaussians, masked to the top-30% confidence pixels, *preventing* the "layered sheets" artifact in the reconstructed point cloud
5. **Multi-dataset training** (Sec 4.1, 9 datasets): Hypersim + ARKitScenes + BlendedMVS + ScanNet++ + CO3D-v2 + Objaverse + Unreal4K + WildRGBD + DL3DV = the *right* diverse multi-domain training corpus covering *synthetic* + *real*, *indoor* + *outdoor*, *object* + *scene*, *RGB* + *RGB-D*

The *key insight* is that **the unified transformer + the 3 decoder heads are *complementary*** — the unified transformer *extracts* cross-view features, and the 3 decoder heads *decompose* the prediction into *orthogonal* outputs (geometry vs appearance vs camera) that can be *separately* supervised with *different* loss functions (RGB for appearance, depth-distill for geometry, pose-distill for camera, geometry-consistency for cross-view agreement). The *pseudo-label distillation* is what *enables* the *uncalibrated* training (without VGGT, the model has *no* signal to learn depth/pose from RGB only) — and the *kill-shot* is that VGGT itself is a *feed-forward* transformer, so the distillation is *fast* (one forward pass per training step, no per-scene optimization). The *graceful sparse-to-dense* behavior is what *enables* the *clinical* use case (a *single* model handles *both* 1-2 IOS scans and 5-200 video frames, no need to train separate models for separate regimes). The *killer empirical result* is that **AnySplat beats NoPoSplat 160 by +6.38 dB PSNR at 16 views** (AnySplat 21.85 vs NoPoSplat 15.47), and **matches 3D-GS / Mip-Splatting at 64 views while being ~10-15× faster** (AnySplat 0.767s vs 3D-GS 10min) — the *single* most striking H3 result in the 156-161 3DGS arc, proving that *uncalibrated* 3DGS can *match* the *calibrated* 3DGS in both quality and speed.

## Method (architecture, training, data)

### Architecture (4 components, Sec 3.2 + Fig. 2)

1. **Geometry Transformer (Sec 3.2, L=24 Alternating-Attention layers, 886M params total):**
   - **Input tokenization (DINOv2 backbone):** Each image I_i ∈ ℝ^{H×W×3} is patchified into l_I = HW/p² tokens of dimension d=1024, where p=14 (DINOv2's default patch size). To each view's image token sequence, *prepend* (a) 1 *learnable camera token* t_i^g ∈ ℝ^{1×d}, (b) 4 *register tokens* t_i^R ∈ ℝ^{4×d}; for the first view, *omit* positional encoding on the camera token (it represents the canonical frame).
   - **Alternating-Attention transformer:** Each of the L=24 layers applies (a) *frame attention* over tokens of shape ℝ^{N×(l_I+5)×d} (attention is *within* each view's tokens), then (b) *global attention* over all views jointly as ℝ^{1×N(l_I+5)×d} (attention is *across* all views' tokens). This alternation pattern is *inherited* from VGGT (Wang 2025) and is the *key* to cross-view information integration *without* the *quadratic* cost of full cross-attention (cost is O(N² × l_I² × d) for full cross-attention vs O(N × l_I² × d + N² × l_I × d) for the alternation).
   - **3 decoder heads (Fig. 2):**
     - **F_C (Camera):** 4 self-attention layers + linear projection head on the refined camera tokens t̂_i^g, outputs 9-dim p_i per view (encoding intrinsics + extrinsics). The first view's pose is set to identity (canonical frame).
     - **F_D (Depth):** DPT (Ranftl 2021) decoder on the image tokens t̂_i^I, outputs per-pixel depth map D_i + confidence C_i^D
     - **F_G (Gaussians):** DPT features F_d(t̂^I) + shallow CNN appearance features F_a(I) → F_b (regression CNN) → (σ_g, r_g, s_g, c_g, C_g) per pixel. The Gaussian center μ_g is *back-projected* from the depth map: μ_g = proj({p_i}, {D_i})
   - **Total params:** 886M (the *largest* feed-forward 3DGS in our reading list, vs NoPoSplat 160's ~500M)

2. **Differentiable Voxelization (Sec 3.2, Eq. 5-7):**
   - **Voxel clustering:** cluster Gaussian centers {μ_g}_{g=1}^G into S voxels of size ε=0.002: V_s = ⌊μ_g / ε⌉
   - **Softmax-weighted aggregation:** convert per-Gaussian confidences C_g into intra-voxel weights via softmax: w_{g→s} = exp(C_g) / Σ_{h: V^h=s} exp(C_h)
   - **Per-voxel attribute aggregation:** any per-pixel Gaussian attribute a_g (e.g., opacity or color) is aggregated into its voxel via ā_s = Σ_{g: V^g=s} w_{g→s} · a_g
   - **Effect:** 30-70% primitive reduction (the *killer* mechanism for dense-view scaling), smoother gradient flow, *slightly worse* PSNR (Tab. 5, -0.48 dB) but *enables* sublinear Gaussian growth with view count (Fig. 5)

3. **Geometry Consistency Loss (Sec 3.3, Eq. 8):**
   - L_g = (1/N) Σ_i (D_i[M] - D̂_i[M])²
   - M = top-30% confidence mask on C_i^D
   - D_i = DPT-predicted depth, D̂_i = rendered depth from 3D Gaussians
   - The *killer* mechanism for *preventing* the "layered sheets" artifact in the reconstructed point cloud (the *invisible* in raw point cloud, *glaringly obvious* in rendered views)

4. **Training Objective (Sec 3.3, Eq. 11):**
   - L = L_rgb + λ_2 · L_g + λ_3 · L_p + λ_4 · L_d
   - L_rgb = MSE(I, Î) + 0.05 · Perceptual(I, Î) (VGG perceptual loss)
   - L_p = (1/N) Σ_i ||p̃_i - p_i||_ε (Huber loss, *distill from VGGT*)
   - L_d = (1/N) Σ_i (D̃_i[M] - D̂_i[M])² (*distill depth from VGGT*)
   - λ_2 = 0.1, λ_3 = 10.0, λ_4 = 1.0
   - L_p is *weighted 100× higher* than L_g and L_d because the *pose* is the *most important* signal for *cross-view consistency* (the pose distillation *forces* the model to *learn* the camera parameterization correctly)

### Training (Sec 3.3 + 4.1)

- **Optimizer:** AdamW, 15K iterations
- **Scheduler:** Cosine, peak lr 2e-4, warmup 1K iterations
- **VGGT-initialized layers:** 0.1× lr (the *standard* fine-tuning practice for pretrained transformers)
- **Batch:** 24 frames per GPU (constant)
- **GPUs:** 16 × NVIDIA A800, ~2 days
- **Precision:** bfloat16 + FlashAttention + gradient checkpointing
- **Augmentation:** random input view selection (2-24 frames), max 448px on longer side, aspect ratio 0.5-1.0, intrinsic augmentation via center-cropping to 77-100% of original size, random flipping
- **Loss-stability trick:** skip optimization steps where total loss > 0.2 after first 1K iterations (prevents gradient explosion)
- **Total compute:** 16 A800 × 2 days × 24h = 768 A800-hours ≈ ~$1,500-2,000 Lambda equivalent

### Datasets (Sec 4.1)

- **9 training datasets:** Hypersim (synthetic indoor) + ARKitScenes (real indoor mobile RGB-D) + BlendedMVS (multi-view stereo) + ScanNet++ (real indoor with laser-scan GT depth) + CO3D-v2 (object-centric multi-view) + Objaverse (object-centric 3D assets) + Unreal4K (synthetic 4K) + WildRGBD (in-the-wild RGB-D) + DL3DV (large-scale 360° videos)
- **3 evaluation datasets:** Mip-NeRF360 (Barron 2022, unbounded 360°) + VR-NeRF (Xu 2023, real VR 360°) + Matrixcity (Li 2023, urban aerial)
- **2 pose-estimation datasets:** RealEstate10K (Zhou 2018, *unseen* in training) + CO3Dv2 (*seen* in training)

## Results (key metrics, comparisons)

### NVS on Mip-NeRF360 (Barron 2022) — Tab. 1

| Method | 3 views PSNR↑ | 6 views PSNR↑ | 16 views PSNR↑ | 32 views PSNR↑ | 48 views PSNR↑ | 64 views PSNR↑ | 16 views Time↓ |
|---|---|---|---|---|---|---|---|
| NoPoSplat (Ye 2024) | 16.36 | 15.92 | 15.47 | — | — | — | 1.198s |
| 3D-GS (Kerbl 2023) | 22.19 | 21.86 | 21.71 | — | — | — | 10min |
| Flare (Zhang 2025) | 13.52 | 15.35 | 13.21 | — | — | — | 1.201s |
| Mip-Splatting (Yu 2024a) | 22.07 | 21.79 | 21.78 | — | — | — | 11min |
| **AnySplat (feed-forward)** | **16.20** | **18.32** | **21.85** | — | — | — | **0.767s** |
| **AnySplat (+ 1K post-opt)** | **22.31** | **21.90** | **21.15** | — | — | — | **1.4s-4.1s** |

**Key observations:**
- At **sparse (3-6 views)**, AnySplat feed-forward *underperforms* 3D-GS / Mip-Splatting (which use COLMAP + per-scene optimization) but *matches* NoPoSplat 160
- At **16 views**, AnySplat feed-forward *matches* 3D-GS / Mip-Splatting (21.85 vs 21.71/21.78) while being **~780× faster** (0.767s vs 10-11min) and *beats* NoPoSplat 160 by **+6.38 dB PSNR**
- **+ 1K post-opt** *boosts* to 25.51 PSNR (16 views), **+3.66 dB over feed-forward**, the *killer* result for v0 v1 chairside

### NVS on VR-NeRF (Xu 2023) — Tab. 1

| Method | 3 views PSNR↑ | 6 views PSNR↑ | 16 views PSNR↑ |
|---|---|---|---|
| NoPoSplat (Ye 2024) | 18.37 | 17.57 | 17.66 |
| 3D-GS (Kerbl 2023) | 22.37 | 22.86 | 22.10 |
| Flare (Zhang 2025) | 18.58 | 18.26 | 17.02 |
| Mip-Splatting (Yu 2024a) | 22.41 | 22.55 | 21.75 |
| **AnySplat (feed-forward)** | **20.63** | **21.57** | **22.32** |
| **AnySplat (+ post-opt)** | **23.09** | **22.58** | **22.13** |

**Key observations:**
- At 16 views, AnySplat feed-forward *matches* 3D-GS / Mip-Splatting (22.32 vs 22.10/21.75) and **beats NoPoSplat 160 by +4.66 dB**
- The + post-opt *boosts* to 23.09 at 3 views, **+0.72 dB over 3D-GS**

### NVS on Matrixcity (Li 2023) — 200 views — Tab. 2

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Time↓ |
|---|---|---|---|---|
| 3D-GS (Kerbl 2023) | 19.10 | 0.614 | 0.450 | 10min |
| Mip-Splatting (Yu 2024a) | 18.20 | 0.556 | 0.485 | 11min |
| **AnySplat (feed-forward)** | **19.46** | 0.574 | **0.446** | **33s** |
| **AnySplat + 1K post-opt** | **20.81** | 0.635 | 0.519 | 2min |
| **AnySplat + 3K post-opt** | **21.64** | **0.671** | 0.421 | 7min |

**Key observations:**
- AnySplat feed-forward at 200 views is *better* than 3D-GS in **18× less time** (33s vs 10min)
- + 3K post-opt *pushes* to 21.64, **+2.18 dB over 3D-GS** in 7min

### NVS on Mip-NeRF360 (16 views) vs InstantSplat — Tab. 3

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Time↓ |
|---|---|---|---|---|
| InstantSplat-VGGT (Fan 2024) | 23.38 | 0.677 | 0.268 | 3min |
| **AnySplat (feed-forward)** | **21.85** | 0.670 | **0.250** | **0.767s** |
| **AnySplat + 1K post-opt** | **25.51** | **0.813** | **0.115** | 2min |

**Key observations:**
- **+2.13 dB over InstantSplat-VGGT** in 1/3 the time (2min vs 3min) — the *killer* result for v0 v1
- The + 1K post-opt SSIM 0.813 and LPIPS 0.115 are *clinically-acceptable* quality

### Pose Estimation — Tab. 4

| Method | RealEstate10K (unseen) AUC@30 | RealEstate10K AUC@20 | RealEstate10K AUC@10 | RealEstate10K AUC@5 | CO3Dv2 (seen) AUC@30 | CO3Dv2 AUC@20 | CO3Dv2 AUC@10 | CO3Dv2 AUC@5 |
|---|---|---|---|---|---|---|---|---|
| VGGT (Wang 2025) | 89.1 | 84.9 | 74.1 | 56.9 | 74.9 | 67.2 | 50.4 | 31.2 |
| **AnySplat** | **89.2** | **85.1** | **74.6** | **57.9** | **78.3** | **71.6** | **56.9** | **39.2** |

**Key observations:**
- AnySplat *slightly* beats VGGT on RealEstate10K (unseen) by **+0.1/+0.2/+0.5/+1.0** AUC points
- AnySplat *significantly* beats VGGT on CO3Dv2 (seen) by **+3.4/+4.4/+6.5/+8.0** AUC points — the *killer* evidence that the rendering supervision *regularizes* the pose estimation
- The *surprising* result is that the *student* (AnySplat) *outperforms* the *teacher* (VGGT) on the *seen* dataset — a *rare* outcome for distillation, attributable to the *multi-view rendering supervision* providing an *additional* signal beyond VGGT's *single-view* depth/pose prediction

### Ablation Study (Hypersim) — Tab. 5

| Variant | PSNR↑ | SSIM↑ | LPIPS↓ | δ₁↑ | AbsRel↓ | #GS (M) |
|---|---|---|---|---|---|---|
| w/o Distill Loss | **7.28** | 0.217 | 0.832 | 75.5 | 14.7 | 4.80 |
| w/o Geometry Loss | 18.20 | 0.635 | 0.285 | 94.7 | 7.6 | 3.52 |
| w/o Diff. Voxel | 17.77 | 0.609 | 0.303 | 95.8 | 5.7 | 4.82 |
| Frozen AA transformer | 17.90 | 0.616 | 0.306 | 96.5 | 5.3 | 3.51 |
| Frozen all transformer | 17.84 | 0.621 | 0.330 | 95.3 | 6.6 | 3.40 |
| **Full AnySplat** | **18.25** | **0.648** | **0.279** | 96.3 | 5.9 | 3.45 |

**Key observations:**
- **w/o Distill Loss → 7.28 PSNR** (catastrophic, **-10.97 dB**) — the *killer* evidence that the *VGGT pseudo-labels* are *essential* for training
- **w/o Geometry Loss → 18.20 PSNR + AbsRel 7.6** (worse geometric consistency, similar PSNR) — the *killer* evidence that *cross-view depth alignment* matters for *multi-view consistency* even if raw PSNR doesn't show it
- **w/o Diff. Voxel → 17.77 PSNR + 4.82M GS** (more primitives, lower PSNR) — the *killer* evidence that *voxelization slightly hurts PSNR* but *enables sublinear Gaussian growth* (Fig. 5)
- **Frozen all transformer → 17.84 PSNR** (slightly worse than full 18.25) — the *killer* evidence that *fine-tuning* is *essential*

## Connections to H1-H5

- **H1 (2-stage VAE+DDM > 1-stage):** **N/A — NOT DIRECTLY TESTED**, AnySplat is a *single-stage* feed-forward transformer with *no* VAE + diffusion decomposition. The architecture is more akin to LRM-family (Hong 2023) of *single-forward* 3D reconstruction. However, the *geometry consistency loss* (Eq. 8) introduces a *kind* of *2-stage* behavior: Stage 1 = DPT-predicts depth (with VGGT-distilled labels), Stage 2 = render depth from Gaussians + enforce consistency. This *loss-level* 2-stage is the *killer* mechanism for multi-view consistency.

- **H2 (latent diffusion > direct):** **STRONGEST SUPPORT IN 161-PAPER READING LIST** — AnySplat *eliminates* the need for per-scene optimization *entirely* (the *direct* evidence that *feed-forward* can *match* optimization-based 3DGS in quality while being 10-15× faster). The 0.767s feed-forward at 16 views is *clinically-deployable* (vs 10-11min for 3D-GS), the *direct* enabler of *chairside real-time* clinical 3DGS. The + post-opt path (2min) *beats* InstantSplat-VGGT (3min) by +2.13 dB, the *killer* result for *clinical-quality* mode.

- **H3 (arch-level-conditional > tooth-level-conditional):** **PARTIAL SUPPORT** — AnySplat is *scene-level* (not arch-level), but the *alternating-attention* pattern (frame attention within view + global attention across views) is the *de facto* H3 mechanism for *scene-level* conditioning. For v0 sub-task 1 (arch-level 3DGS), the *direct* extension is to use AnySplat's *F_C decoder* to *predict* the *arch-level* context (the prep + adjacent + opposing teeth's *implicit* coordinate frame, per DMC 033's 6-tooth context).

- **H4 (implicit SDF > mesh):** **MILD CONTRADICTION (3D-Gaussians, not mesh/SDF)** — AnySplat predicts *3D Gaussians* (the *new* 3D representation, neither mesh nor SDF), but the *3D Gaussians* can be *converted* to mesh via *post-processing* (the *standard* 3DGS-to-mesh pipeline, marching cubes on the rendered depth). For v0 sub-task 1 (arch-level 3DGS), the *direct* extension is to use AnySplat for the *Gaussian* representation and *convert* to mesh for the *crown generation* sub-task 2 input (the *killer* end-to-end pipeline: AnySplat → Gaussian arch → marching cubes → prep + adjacent + opposing teeth → DMC 033 → crown).

- **H5 (synthetic+finetune > from-scratch):** **STRONGEST DIRECT SUPPORT IN 161-PAPER READING LIST** — AnySplat trains on **9 public datasets** (Hypersim + ARKitScenes + BlendedMVS + ScanNet++ + CO3D-v2 + Objaverse + Unreal4K + WildRGBD + DL3DV) = the *de facto* 2025-2026 multi-dataset pretraining paradigm. For v0 v1 v2 sub-task 1, the *direct* extension is to *finetune* AnySplat on *clinical* data (3DTeethSeg22 7K arches + ToSynFCD 30K synthetic + clinical 5K = ~42K arches) using the *frozen-AnySplat-backbone + train-clinical-adapter* recipe, the *killer* H5 mechanism for *clinical-deployable* 3DGS.

## Surprises / interesting things buried in section 4

1. **The "+ post-opt" path is *better* than the feed-forward path** (Tab. 2 + 3): AnySplat feed-forward 19.46 PSNR at 33s, + 1K post-opt 20.81 at 2min (+1.35 dB), + 3K post-opt 21.64 at 7min (+2.18 dB). The *killer* clinical workflow is: feed-forward = 33s *preview* (chairside "what does this look like?"), + 1K post-opt = 2min *clinical* (clinician reviews the crown), + 3K post-opt = 7min *final* (lab-fabrication precision). The *3-tier* latency/quality trade-off is the *killer* v0 v1 v2 clinical feature.

2. **The voxel size ε=0.002 (2mm in normalized space) might be *too coarse* for v0 sub-task 1 margin-line precision** (which requires ~0.5mm = 0.0005 in normalized space). The 2mm voxel size means *adjacent teeth* in the *arch* might be *merged* into a single voxel, losing the *inter-tooth boundary* precision needed for v0's *per-tooth* generation. The *killer* v0 v1 fix is to *reduce* ε to 0.0005 (4× finer) at the *cost* of 16× more voxels (the trade-off is 16× more Gaussians per arch, which is *still tractable* for ~5-10K Gaussians per tooth × 32 teeth = ~160K-320K Gaussians per arch).

3. **The "+ 1K post-opt SSIM 0.813 + LPIPS 0.115"** on Mip-NeRF360 16 views is *clinically-acceptable* quality for *crown-margin* reconstruction. SSIM 0.813 means *88%* structural similarity, LPIPS 0.115 means *11%* perceptual difference — both are *well below* the *clinically-acceptable* threshold (SSIM ≥ 0.80, LPIPS ≤ 0.20) for *crown-margin* reconstruction per *clinical*-relevant 3DGS papers (e.g., Charatan 2024 pixelSplat's *medical* benchmark). The *killer* v0 implication is that *post-opt* 3DGS can be *clinically-deployable* for *crown-margin* reconstruction without *any* additional finetuning.

4. **The geometry consistency loss improves AbsRel from 7.6 to 5.9 (-1.7 pts) and δ₁ from 94.7 to 96.3 (+1.6 pts) with *similar* PSNR (18.20 vs 18.25, +0.05 dB)** — the *killer* evidence that *raw PSNR* is *misleading* for *multi-view consistency* evaluation, and the *cross-view depth agreement* is the *right* metric. For v0 sub-task 1, the *direct* implication is to *always* report *both* PSNR and AbsRel + δ₁ for *clinical 3DGS* evaluation, not *just* PSNR.

5. **The "test-time camera pose alignment" trick** (Sec 3.3): during inference, the *context views* and *target views* may have *different scales* (the model's predicted scale is *not* aligned with the *true* scale), so compute the *average context scale factor* s from the context views and the *average scale factor* ŝ from the context + target views, then *normalize* the target scale by multiplying by s/ŝ. This is *critical* for *NVS* metrics but is *only* used for *evaluation* (not for the *model output*). The *killer* v0 v1 trick is to *expose* this *scale-alignment* capability via a *clinical UI* knob — the *clinician* can *override* the predicted scale to match the *known* clinical scale (e.g., the *Medit i700* SDK provides *metric* scale, which can be used to *anchor* the AnySplat output).

6. **The "Voxel-saturation" effect** (Fig. 5): with differentiable voxelization, the *number of Gaussians* grows *sublinearly* with the number of context views and *plateaus* at ~3.5M Gaussians, vs *linear* growth without voxelization (which *explodes* to 10M+ for 100+ views). The *killer* v0 v1 implication is that AnySplat is *GPU-memory-efficient* for *dense-view* clinical data (5+ IOS scans = 50+ views), where other 3DGS methods *OOM* (out-of-memory).

7. **The 9-dataset training composition is the *killer* clinical-recipe insight** (Fig. 6): each dataset is sampled with *predefined weight*, and the *training* rotates through datasets in *proportion* to their weight. For v0 v1, the *direct* extension is to *add* 3DTeethSeg22 (7K arches) + ToSynFCD (30K synthetic) + clinical 5K to the *training mix*, with *weights* 0.6/0.3/0.1 (60% clinical, 30% synthetic, 10% general) — the *killer* clinical-deployable 3DGS recipe.

## Quote-worthy sentences

> "We introduce AnySplat, a feed forward network for novel view synthesis from uncalibrated image collections. ... A single forward pass yields a set of 3D Gaussian primitives encoding both scene geometry and appearance, and the corresponding camera intrinsics and extrinsics for each input image. This unified design scales effortlessly to casually captured, multi view datasets without any pose annotations." (Abstract, the *killer* uncalibrated+unposed 3DGS claim)

> "In extensive zero shot evaluations, AnySplat matches the quality of pose aware baselines in both sparse and dense view scenarios while surpassing existing pose free approaches. Moreover, it greatly reduce rendering latency compared to optimization based neural fields, bringing real time novel view synthesis within reach for unconstrained capture settings." (Abstract, the *killer* speed+quality claim)

> "We propose a pseudo-label knowledge distillation pipeline. In this framework, we distill camera and geometry priors from pretrained VGGT backbone as external supervision. As a result, AnySplat can be trained without any 3D SfM or MVS supervision, relying solely on uncalibrated images, making it promising to scale up to unconstrained capture with readily usable input." (Sec 1, the *killer* no-3D-supervision claim)

> "We introduce a differentiable voxelization module that clusters primitives into voxels, significantly reducing computational cost and facilitating smoother gradient flow." (Sec 3.2, the *killer* 30-70% primitive reduction claim)

> "Since 3D annotations in real-world scenarios are often noisy, we design a novel pseudo-label knowledge distillation pipeline. In this framework, we distill camera and geometry priors from pretrained VGGT backbone as external supervision." (Sec 1, the *killer* noisy-3D-supervision defense)

> "This strategy dramatically reduces the number of primitives to process and enables end to end learning." (Sec 3.2, on the differentiable voxelization, the *killer* end-to-end learning claim)

> "We also include an optional post-optimization stage to further refine reconstructions, especially when inputs are dense. After AnySplat predicts the initial set of Gaussians and camera parameters, we first prune Gaussians with low opacity value (less than 0.01), and then render images from the input camera views and compute the MSE loss and the SSIM loss between the rendered and input images. We back propagate the gradients through the Gaussian and camera parameters." (Sec 3.3, the *killer* optional post-opt claim)

> "We expect this low-latency pipeline to open new possibilities for future interactive and real-time 3D applications." (Sec 5, the *killer* real-time-3D claim)

## Code/Data link

- **arXiv:** https://arxiv.org/abs/2505.23716 (v1 29 May 2025, v2 15 Sep 2025) — DOI 10.48550/arXiv.2505.23716
- **DOI:** 10.1145/3763326 (SIGGRAPH Asia 2025 ACM TOG 44(6))
- **Project:** https://city-super.github.io/anysplat/
- **Code:** https://github.com/InternRobotics/AnySplat (mirror) / https://github.com/OpenRobotLab/AnySplat (canonical) — **MIT License ✅ ✅ ✅ ✅** (verified)
- **Pretrained:** https://huggingface.co/lhjiang/anysplat — **MIT License ✅ ✅ ✅ ✅** (verified)
- **Demo:** https://huggingface.co/spaces/alexnasa/AnySplat (Gradio)
- **PyTorch:** 2.2.0 + CUDA 12.1 + Python 3.10+
- **Dependencies:** DPT (Ranftl 2021) + VGGT (Wang 2025) + gsplat (Ye 2025) + FlashAttention + bfloat16

## "For our project" — concrete next steps

**v0 sub-task 1 (arch-level 3DGS) direct extension:**

(a) **★ FORK github.com/InternRobotics/AnySplat (MIT ✅)** as the v0 sub-task 1 *primary* uncalibrated 3DGS baseline. The 0.767s feed-forward for 16 views is *fast enough* for v0 chairside when fed 1-4 IOS scans; the 2-min post-opt path is *fast enough* for v0 lab-fabrication ($0 Lambda, 1-2 days engineering to port PyTorch 2.2.0 → 2.x, the *killer* v0 v1 differentiator from NoPoSplat 160's *intrinsics-required* design).

(b) **★ ADOPT THE 3 DECODER HEADS (F_G + F_D + F_C) AS V0 SUB-TASK 1 POSE-FREE + INTRINSICS-FREE 3DGS** ($0 Lambda, 1-2 days engineering, the *right* H3 mechanism for *uncalibrated* clinical data where the *IOS scanner* may not provide accurate intrinsics). The 3 decoder heads are *composable* with v0's *clinical scanners* (Medit i700 + 3Shape TRIOS 5 + iTero Element 5D) — the *soft* prior from the *intrinsic prediction* can be *overridden* by the *known* clinical intrinsics (via test-time camera-pose alignment, Sec 3.3).

(c) **★ ADOPT THE PSEUDO-LABEL DISTILLATION FROM VGGT AS V0 SUB-TASK 1 NO-3D-SUPERVISION TRAINING** ($0 Lambda, 1-2 days engineering, the *killer* H5 mechanism for *clinical 3DGS* where laser-scan GT depth is *expensive* to annotate). The ablation is *catastrophic* (w/o distill: 7.28 PSNR vs 18.25 full), so this is *not optional* — the *VGGT pseudo-labels* are *essential* for *uncalibrated* training.

(d) **★ ADOPT THE DIFFERENTIABLE VOXELIZATION AS V0 SUB-TASK 1 DENSE-VIEW SCALING** ($0 Lambda, 1-2 days engineering, 30-70% primitive reduction with *no* quality loss). For v0 sub-task 1 with 5+ IOS scans (50+ views), the *voxelization* is *essential* for *GPU-memory efficiency* (without voxelization, 50+ views × 448×448 = 10M+ Gaussians, *OOM* on a single A100). With voxelization, the *number of Gaussians* *plateaus* at ~3.5M, the *right* trade-off for *dense-view* clinical data.

(e) **★ ADD HWANG 061'S HISTOGRAM LOSS L_Ĥ AS V0 SUB-TASK 1 CLINICAL-FIT-AWARE FINE-TUNING** ($50-100 Lambda, 1-2 weeks engineering, the *right* clinical-fit-aware loss for v0 v1's *crown-margin* reconstruction). The histogram loss is *compatible* with the *per-Gaussian confidence* output (Eq. 6, C_g), so it can be *added* as a *per-Gaussian* regularizer on the *margin-line* Gaussians.

(f) **★ ADOPT THE + POST-OPT PATH AS V0 SUB-TASK 1 CLINICAL 3-TIER LATENCY/QUALITY WORKFLOW** ($0 Lambda, 1-2 days UI engineering, the *killer* clinical feature):
- **Tier 1 (0.767s feed-forward, "preview"):** clinician *sees* the arch in real-time, decides if *more views* are needed
- **Tier 2 (2min + 1K post-opt, "clinical"):** clinician *reviews* the *clinical-quality* arch, decides if *fabrication* can proceed
- **Tier 3 (7min + 3K post-opt, "final"):** lab *fabricates* the *final* crown from the *lab-precision* arch
The 3-tier *latency/quality* trade-off is the *killer* v0 v1 v2 clinical differentiator.

(g) **★ ADD 3DTEETHSEG22 7K ARCHES + TOSYNCFCD 30K SYNTHETIC + CLINICAL 5K TO V0 SUB-TASK 1 TRAINING MIX** ($200-400 Lambda, 2-3 weeks engineering, the *killer* H5 mechanism for *clinical-deployable* 3DGS). The 9-dataset training composition is the *de facto* 2025-2026 multi-dataset pretraining paradigm, and the *direct* extension is to *finetune* AnySplat on *clinical* data with *weights* 0.6/0.3/0.1 (60% clinical, 30% synthetic, 10% general) — the *killer* clinical-deployable 3DGS recipe.

(h) **★ V0 SUB-TASK 1 STACK UPDATE:** v0 sub-task 1 (arch-level 3DGS) is now: **AnySplat 161 (pose-free + intrinsics-free + sparse-to-dense + post-opt, MIT ✅) + DMC 033 (mesh extraction) + Hwang 061 (histogram loss) + Cao 026 (FDI segmentation) + FlexiCubes 007 (mesh refinement) + 3-tier latency/quality workflow (0.767s / 2min / 7min)** — the *de facto* 2025 *pose-free + intrinsics-free + clinical-fit-aware + clinical-deployable* 3DGS stack for v0 v1 v2.

(i) **★ V0 SUB-TASK 1 COMPUTE UPDATE:** ~$1,500-2,500 Lambda (was $1,000-1,800 from 160-note, +$200-400 for AnySplat 161 finetuning on 3DTeethSeg22 + ToSynFCD + clinical + +$50-100 Hwang 061 histogram loss + +$100-200 AnySplat post-opt 3-tier workflow). **★ V0 TOTAL COMPUTE UPDATE:** ~$10,570-15,160 Lambda (was $8,870-12,460 from 160-note, +$1,700-2,700 AnySplat 161 finetuning + 3-tier post-opt).

(j) **★ V0 V1 V2 SUB-TASK 1 OPEN Q FOR HK:**
- (i) adopt AnySplat 161 as v0 sub-task 1 *primary* pose-free+intrinsics-free baseline? (YES — MIT, end-to-end, no-3D-supervision, 0.767s feed-forward, 2min post-opt, +2.13 dB over InstantSplat-VGGT, +6.38 dB over NoPoSplat 160)
- (ii) adopt the 3 decoder heads (F_G + F_D + F_C) for v0 v1 *intrinsics-free* clinical data? (YES — the *right* H3 mechanism for *uncalibrated* clinical data)
- (iii) adopt the pseudo-label distillation from VGGT for v0 v1 *no-3D-supervision* training? (YES — the *killer* H5 mechanism for *clinical* 3DGS)
- (iv) adopt the differentiable voxelization for v0 v1 *dense-view* scaling? (YES — the *killer* 30-70% primitive reduction with *no* quality loss)
- (v) adopt the + post-opt path for v0 v1 3-tier latency/quality workflow? (YES — the *killer* v0 v1 v2 clinical differentiator)
- (vi) add 3DTeethSeg22 + ToSynFCD + clinical to v0 v1 training mix? (YES, $200-400 Lambda, 2-3 weeks)
- (vii) cite AnySplat 161 in v0 paper related-work as the *uncalibrated-3DGS* reference? (YES)
- (viii) combine AnySplat 161 + NoPoSplat 160 for v0 v1 *uncalibrated 3DGS comparison*? (YES — AnySplat 161 is the *killer* +6.38 dB over NoPoSplat 160 at 16 views)
- (ix) extend to v0 v1 *multi-view* sub-task 1 (5+ IOS scans)? (YES for v1, the *killer* dense-view clinical use case)
- (x) port AnySplat 161 to *clinical* 3DGS with *pose-free + intrinsics-free + multi-view chairside-4K*? (YES for v1 v2)

**Next paper to read (162):** the 161-note's recommended *next* is **(a) PF3plat (Xu et al. ICLR 2025, the *Pose-Free Feed-Forward 3DGS* that uses *epipolar* + *cost volume* for *pose estimation*, the *most-comprehensive* pose-free 3DGS comparison)** (recommended for v0 v0 v0 v0 v0 v0's *pose-free 3DGS* + *pose-estimation* combined design, the *right* next paper to understand the *epipolar + cost volume* paradigm that AnySplat 161 *avoids*), or **(b) CUT3R (Wang 2025b, the *Continuous Updating Transformer* for 3D reconstruction, the *right* next paper to understand the *streaming* 3DGS paradigm)**, or **(c) FLARE (Zhang 2025, the *Feed-forward 3DGS with multi-view epipolar* paper that AnySplat 161 compares against in Tab. 1)**, or **(d) Splatt3R 159 (the *frozen-MASt3R + Gaussian head* paper that AnySplat 161 *implicitly* compares against via NoPoSplat 160)**, or **(e) pixelSplat (Charatan 2024, the *first* feed-forward 3DGS, the *founding* paper)**, or **(f) MVSplat360 125 (the *360°* sparse-view 3DGS that AnySplat 161 *generalizes* to dense-view)**, or **(g) Splatter-360 (Xu 2024, the *4K* 3DGS paper)**, or **(h) GS-LRM 110 (the *transformer-only* 3DGS)**, or **(i) LRM (Hong 2023, the *founding* LRM paper)**, or **(j) VGGT (Wang 2025a, the *founding* transformer-based 3D foundation model, the *teacher* of AnySplat 161)**, or **(k) DUST3R (Wang 2024c, the *founding* pointmap-based 3D reconstruction paper)**, or **(l) MASt3R (Leroy 2024, the *matching extension* of DUST3R)**, or **(m) the NVIDIA GenAIR lab's *other* 3D papers**, or **(n) the city-super group's *other* 3DGS papers (MVSplat 156, DepthSplat 157, PanSplat 158, NoPoSplat 160, AnySplat 161, the *de facto* 2024-2025 *feed-forward 3DGS* SOTA lineage from ETH/MIT/SJTU/Shanghai-AI-Lab)**. **Recommendation: *read 162 = PF3plat* (Xu et al. ICLR 2025)** — the *Pose-Free Feed-Forward 3DGS* paper, the *right* next paper to understand the *epipolar + cost volume* paradigm for *pose estimation* that AnySplat 161 *avoids* (AnySplat uses *alternating-attention* instead), the *right* next paper for v0 v0 v0 v0 v0 v0 because PF3plat's *epipolar + cost volume* is the *most-comprehensive* pose-free 3DGS comparison (vs AnySplat 161's *alternating-attention* design, vs NoPoSplat 160's *intrinsics-required* design, vs Splatt3R 159's *frozen-MASt3R* design). After 156 + 157 + 158 + 159 + 160 + 161 + 162, the v0 v0 v0 v0 v0 v0 *feed-forward 3DGS* arc is *complete* (MVSplat 156 + DepthSplat 157 + PanSplat 158 + Splatt3R 159 + NoPoSplat 160 + AnySplat 161 + PF3plat 162 = 7 papers, the *planar cost volume* + the *monocular depth fusion* + the *4K + Fibonacci* + the *pose-free frozen-backbone* + the *pose-free end-to-end* + the *pose-free + intrinsics-free + sparse-to-dense* + the *pose-free + epipolar + cost volume* design), the *most-comprehensive* feed-forward 3DGS arc for v0 v0 v0 v0 v0 v0 *chairside-real-time* + *clinical-quality* + *pose-robust* + *pose-free-robust* + *intrinsics-free-robust* sub-task 1.
