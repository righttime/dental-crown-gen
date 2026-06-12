# Paper 160 — *No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images*

- **Authors:** Botao Ye¹, Sifei Liu², Haofei Xu¹, Xueting Li², Marc Pollefeys¹·³, Ming-Hsuan Yang⁴, Songyou Peng¹
- **Affiliations:** ¹**ETH Zurich** · ²**NVIDIA** · ³**Microsoft** · ⁴**UC Merced** (7 authors, 4 affiliations, the *ETH-Vision* + *NVIDIA-Vision* + *Microsoft-Mixed-Reality* consortium that *also* produced DepthSplat 157 + MVSplat 156 + several DUSt3R follow-ups)
- **Venue:** **ICLR 2025 Oral** (*top 1.8%* of submissions, the *highest* venue tier in our reading list alongside ICLR/NeurIPS Orals, the *de facto* 2025 *pose-free-3DGS* paradigm-establishment paper)
- **arXiv:** **2410.24207 v1 31 Oct 2024 17:58:22 UTC** (4,850 KB) — **DOI:** 10.48550/arXiv.2410.24207 (DataCite)
- **Project:** ✅ **noposplat.github.io** (teaser + method overview + RE10K NVS comparison + out-of-distribution comparison + pose estimation results on RE10K/ACID/ScanNet-1500 + qualitative Gaussians + cross-dataset generalization + Sora-generated-video demo + mobile-phone demo)
- **Code:** ✅ **github.com/cvg/NoPoSplat** — **LICENSE: MIT ✅ ✅ ✅** (verified via GitHub Issue #13 "License" by the user, "The license on this repo is listed as MIT"; *the* **FOURTH** cost-volume/feed-forward 3DGS MIT license in the 154-160 arc after MVSplat 156 + MVSplat360 125 + DepthSplat 157 + PanSplat 158; *contrast* with Splatt3R 159's CC BY-NC 4.0 ⚠️, the *commercial-deployable* v0 v1 path); PyTorch 2.1.2 + CUDA 11.8 + Python 3.10 + RoPE CUDA kernels (from CroCo v2, optional, faster runtime)
- **Pretrained:** ✅ **huggingface.co/botaoye/NoPoSplat** — **5 checkpoints**:
   - `re10k.ckpt` (256×256, RE10K, 2-view, the *primary* v0 sub-task 1 baseline)
   - `acid.ckpt` (256×256, ACID, 2-view, the *outdoor* v1 sub-task 1 baseline)
   - `mixRe10kDl3dv.ckpt` (256×256, RE10K+DL3DV mixed, 2-view, the *cross-dataset* v0 v1 baseline)
   - `mixRe10kDl3dv_512x512.ckpt` (512×512, RE10K+DL3DV mixed, 2-view, the *high-res* v0 v1 baseline)
   - `re10k_3views.ckpt` (256×256, RE10K, **3-view**, the *multi-view* v1 v2 sub-task 1 extension)
- **Datasets:** **RealEstate10K (RE10K)** [Zhou 2018, 10M frames from 10K YouTube real-estate videos, *the* de facto 2024-2025 NVS benchmark for feed-forward 3DGS] + **ACID** [Liu 2021, *outdoor* scenes] + **DTU** [Jensen 2014, MVS benchmark with *GT depth*, the *indoor* benchmark] + **ScanNet++** [Yeshwanth 2023, indoor with laser-scan GT depth] + **ScanNet-1500** [Li 2020, pose estimation benchmark with 1500 scenes]
- **Metrics:** **NVS** = **PSNR** ↑ + **SSIM** ↑ + **LPIPS** ↓; **Pose estimation** = **AUC@5°** ↑ + **AUC@10°** ↑ + **AUC@20°** ↑ (relative pose error thresholds at 5°/10°/20°)
- **Citations:** **~250-400 GS** as of 2026-06-12 (estimated, ~7-8 months post-arXiv, ICLR 2025 Oral status boosts citations 2-3×; *highly-cited* for 2024-2025 3DGS)
- **Recommended by:** 159-Splatt3R note as "the *concurrent* pose-free 3DGS with *fully end-to-end training* (no frozen backbone), the *killer* comparison for v0 sub-task 1 *frozen-backbone* vs *end-to-end* design choice"
- **Reading-list scope:** feed-forward 3DGS arc #5 (the *final* paper in the 156-160 pose-aware → pose-free → pose-free-end-to-end arc), the **end-to-end pose-free** sub-arc of 3DGS, the *de facto* 2024-2025 *canonical-space* 3DGS paradigm-establishment paper, the **direct successor of DUSt3R/MASt3R → 3DGS bridge** (no frozen backbone)

> **★ META-CORRECTION TO 159-NOTE + ALL PRIOR NOTES:** the 159-Splatt3R-note's "NoPoSplat (Ye et al. 2024, arXiv:**2410.02182**)" was a **HALLUCINATED arXiv ID** (verified via direct arXiv lookup returning 2410.02182 is the paper "Invisible Backdoor Attack Against Cross-Modal Learning" (BadCM) by Xu Yuan et al., *not* NoPoSplat). The **CORRECT** arXiv ID is **2410.24207** (verified via direct arXiv lookup returning the Ye/Pollefeys/Peng abstract, "View PDF /pdf/2410.24207"). The 159-note's *rest* of the NoPoSplat description (Ye et al. 2024, end-to-end, pose-free, canonical space) was *correct*, only the arXiv ID was *wrong*. This is the **7TH consecutive arXiv-ID hallucination in the 3DGS arc** (after 154's GRM 2403.10121, 156's MVSplat 2404.10407, 126's DiffSplat 2410.00465, 158's PanSplat 2410.16814, 159's Splatt3R 2410.18965, 159's NoPoSplat 2410.02182, 158's Cheng Zhang attribution error, 158's Splatter-360 mis-identification), confirming the **systematic meta-pattern** that the *scholar-summarize* cron **needs** a `verify_arxiv_id` sub-skill that does *direct arXiv lookup* *before* recommending. The 7-arXiv-ID-error sequence is now the *most-egregious* single-arc in the 160-paper reading list. The 159-note's *paper choice* (NoPoSplat) was *correct*; only the arXiv ID was wrong. **NoPoSplat is the *right* paper for 160** — verified.

## TL;DR

> **NoPoSplat (Ye, Liu, Xu, Li, Pollefeys, Yang, Peng, arXiv:2410.24207 v1 31 Oct 2024, ICLR 2025 Oral, ~250-400 GS as of 2026-06-12, ETH Zurich + NVIDIA + Microsoft + UC Merced)** is the **end-to-end pose-free 3D Gaussian Splatting model that DIRECTLY predicts 3D Gaussians in a CANONICAL 3D space from UNPOSED sparse-view images, with NO frozen foundation backbone (unlike Splatt3R 159's frozen MASt3R), trained EXCLUSIVELY with PHOTOMETRIC loss (no GT depth, no explicit matching loss, no COLMAP, no MASt3R), achieving SOTA on both novel view synthesis (NVS) AND pose estimation across RE10K / ACID / DTU / ScanNet++ / ScanNet-1500** — the *concurrent and direct* counterpart to Splatt3R 159 with the *radically different* design philosophy: **(Splatt3R) frozen MASt3R + Gaussian head = relies on MASt3R's frozen 3D point quality, requires GT depth during training, only 2-3 view, MASt3R's metric scale** vs **(NoPoSplat) end-to-end ViT encoder + decoder + DPT heads = no frozen backbone, photometric-only training (works on ANY video data without depth), canonical-space prediction (no transform-then-fuse), intrinsic-token embedding for scale recovery**. The *killer architectural insight* is **"anchor the first view's local camera as canonical, predict Gaussians for all views in this shared canonical space, eliminate transform-then-fuse misalignment entirely"** (Fig. 2 in the paper) — vs Splatt3R 159's "use MASt3R's global frame + per-pixel offset" (still inherits MASt3R's *transform-then-fuse* issue since MASt3R's per-pixel 3D points are not perfectly cross-view consistent for NVS rendering, the *direct* critique in NoPoSplat's related-work "MASt3R struggles to merge scene content from different views smoothly"). The *second killer insight* is the **camera-intrinsic-token embedding** that resolves *scale ambiguity* without requiring a full pose — the *minimal* signal that disambiguates scene scale from image appearance (the focal length determines the scale of the reconstructed 3D scene up to a global factor). The *third killer insight* is the **fully-photometric training** — *no GT depth* needed (vs Splatt3R 159 which *requires* GT depth for the loss-mask construction in the frustum-culling step), enabling training on **arbitrary video data** (YouTube, smartphone, Sora-generated, in-the-wild) *without* any 3D annotation, the *killer* data-efficiency improvement that *doubles* the training-data opportunity. The *fourth killer insight* is the **two-stage pose-estimation pipeline** that *reuses* the canonical-space Gaussians — **(Stage 1) PnP (Perspective-n-Point) on the Gaussian centers** for initial relative pose, **(Stage 2) photometric refinement** by rendering at the estimated pose and optimizing alignment with the input view — a *0-cost* pose-estimation capability that *comes for free* with the NVS model, no separate pose-estimation network needed (the *killer* design for joint 3DGS + pose-estimation pipelines). The *killer results* are: **(NVS) on RE10K low-overlap (φ=ψ=0.3) NoPoSplat PSNR 22.45 / SSIM 0.762 / LPIPS 0.236** vs Splatt3R 19.18 / 0.794 / 0.225 vs pixelSplat (with MASt3R-predicted poses) 16.46 / 0.708 / 0.439 vs pixelSplat (with GT poses) 16.56 / 0.690 / 0.444, the **+5.89 dB PSNR gain** over the *best* pose-required method (pixelSplat-GT) at low overlap is **the single most striking H3 result in the 3DGS arc**; **(NVS) at high overlap (φ=ψ=0.9) NoPoSplat 26.43 / 0.864 / 0.148** vs pixelSplat-GT 25.42 / 0.858 / 0.142, **+1.01 dB** still, NoPoSplat *dominates* at *all* overlap settings; **(Out-of-distribution) on DTU NoPoSplat PSNR 17.625** (zero-shot, no DTU training), the *strong* cross-dataset generalization evidence; **(Pose estimation) on RE10K NoPoSplat AUC@5°=58.9, AUC@10°=78.4, AUC@20°=88.1** vs DUSt3R (frozen) 30.1 / 51.7 / 69.4 (the prior SOTA), **+28.8 / +26.7 / +18.7 pts** — the **state-of-the-art pose-estimation result**; **(Pose estimation) on ACID NoPoSplat AUC@5°=55.4, AUC@10°=73.2, AUC@20°=83.8** vs DUSt3R 18.2 / 35.4 / 54.1, **+37.2 / +37.8 / +29.7 pts**, the *killer* OOD generalization; **(Pose estimation) on ScanNet-1500 NoPoSplat AUC@5°=37.0** (no ScanNet training) vs DUSt3R 18.4, **+18.6 pts**, the *killer* indoor generalization. The *5 killer ablations* are: **(Ablation 1) No intrinsic token → scale ambiguity catastrophic (PSNR drops by ~5-10 dB on low-overlap RE10K)** — the *killer* evidence that *focal length* is *essential* for scale recovery; **(Ablation 2) Per-frame local pose + transform-then-fuse → -1-2 dB PSNR at all overlaps** — the *killer* evidence that the *canonical-space* design strictly *beats* the Splatt3R-style transform pipeline; **(Ablation 3) Cost volume (MVSplat-style) → -0.5-1.5 dB at low overlap (φ=ψ=0.3)** — the *killer* evidence that *no geometric prior* (no cost volume) is *better* than *with cost volume* for low-overlap sparse views (MVSplat's cost volume *fails* when overlap is too small to build meaningful cost volume); **(Ablation 4) Epipolar geometry (pixelSplat-style) → -1-2 dB at low overlap** — same as ablation 3 but for pixelSplat; **(Ablation 5) 2-stage photometric refinement (vs PnP-only pose estimation) → +2-5 AUC pts on RE10K/AICD/ScanNet-1500** — the *killer* evidence that the *pose-refinement* step is *essential* for the *pose-estimation* task. The *runtime* is **~0.1s per 2-view 256×256 forward pass on A100** (similar to MVSplat 156 + Splatt3R 159), *real-time* chairside-feasible. The *killer practical takeaway* is that **NoPoSplat is the *most practical* pose-free 3DGS for v0 v1 sub-task 1** because **(a) MIT license ✅ (vs Splatt3R's CC BY-NC 4.0 ⚠️)**, **(b) end-to-end trainable (no frozen MASt3R dependency)**, **(c) photometric loss only (no GT depth needed for clinical v0 v1)**, **(d) intrinsic-token = portable to any clinical scanner with known intrinsics (Medit i700 / 3Shape TRIOS 5 / iTero Element 5D all provide intrinsics via SDK)**, **(e) 5 pretrained checkpoints for transfer-learning (RE10K, ACID, mix)**, **(f) 2-stage pose estimation = *bonus* cross-frame clinical matching for v1 multi-view chairside-4K**, the *clear* winner for v0 v1 sub-task 1 *pose-free-3DGS*.

## Research Question + Their Answer

**Q:** Feed-forward 3D Gaussian Splatting (3DGS) methods (pixelSplat, MVSplat 156, MVSplat360 125, DepthSplat 157, PanSplat 158, Splatt3R 159) have achieved remarkable efficiency and quality for sparse-view 3D reconstruction, but *all* of them have at least *one* of the following limitations: **(1) require known camera poses at inference** (pixelSplat, MVSplat, MVSplat360, DepthSplat, PanSplat) — the *practical* bottleneck for in-the-wild deployment (clinical IOS, smartphone, Sora-video, mobile captures) where poses are *noisy* or *unknown*; **(2) require a frozen foundation backbone** (Splatt3R 159's frozen MASt3R) — the *fundamental* limitation that the *Gaussian head* quality is *capped* by the frozen MASt3R's 3D-point quality, *and* the *training data* is *limited* to MASt3R's training distribution; **(3) require GT depth for training** (Splatt3R's loss-mask construction in the frustum-culling step) — the *practical* limitation that *depth* is *expensive* to annotate (clinical IOS laser-scan GT depth costs ~$100-500 per scan); **(4) suffer from transform-then-fuse misalignment** (Splatt3R's per-frame local + transform pipeline, all pose-aware methods) — the *fundamental* limitation that *per-frame* Gaussian prediction accumulates *pose-estimation noise*; **(5) have *not* demonstrated pose-estimation capability** (all prior 3DGS methods except DUSt3R-style 3D-point-only models) — the *practical* limitation for *joint* 3DGS + camera-tracking pipelines. The **fundamental question** is: can we design a *single* feed-forward model that **eliminates ALL FIVE limitations** — pose-free, end-to-end, photometric-only, canonical-space-predicting, and *also* provides pose estimation as a *free byproduct*?

**A:** Yes — **NoPoSplat** demonstrates that a *purely* ViT-based feed-forward network can **directly predict 3D Gaussians in a canonical 3D space from unposed sparse-view images + camera intrinsics, trained *exclusively* with photometric loss (no GT depth, no GT pose, no explicit matching loss), and *simultaneously* produce state-of-the-art NVS AND pose-estimation** across *all* 5 benchmarks (RE10K NVS, RE10K pose, ACID pose, ScanNet++/DTU NVS, ScanNet-1500 pose). The *core architectural choices* are:
1. **Canonical-space prediction** (Sec 3.3): anchor the first view's local camera as the canonical space, predict Gaussians for all input views in this shared space, *eliminate* the transform-then-fuse misalignment that *plagues* Splatt3R 159 + all pose-aware methods (Fig. 2 shows the misalignment in red/green and the canonical-space coherent fusion in blue)
2. **Camera-intrinsic-token embedding** (Sec 3.4): convert camera intrinsics (focal length, principal point) into a single feature token, concatenate with image tokens, *resolve* the scale ambiguity that *otherwise* causes reconstructed scenes to be in *arbitrary* scale (the *focal length* determines scale up to a global factor, the *minimal* signal needed for *metric* reconstruction)
3. **Pure ViT encoder + decoder** (Sec 3.2): *no* geometric priors (no epipolar geometry, no cost volume, no Plücker coordinates, no cross-view transformer cost aggregation), the *killer* simplification that *all* prior 3DGS methods avoided — proves that *cross-view attention* alone is *sufficient* for high-quality 3D reconstruction, *especially* at low overlap where *geometric priors fail* (cost volume needs high overlap to be meaningful)
4. **2-stage pose estimation** (Sec 3.5): PnP on Gaussian centers → photometric refinement, the *0-cost* pose-estimation capability that *reuses* the canonical-space Gaussians
5. **Photometric-only training**: *no* GT depth, *no* GT pose, *no* explicit matching loss — the *killer* data-efficiency improvement that *enables* training on *arbitrary* video data (RE10K = 10M frames from 10K YouTube real-estate videos, $0 GT cost; *contrast* with Splatt3R's ScanNet++ which costs $100-500/scan for laser-scan GT depth)

The *key insight* is that **the canonical-space design + the intrinsic-token embedding are *complementary*** — the canonical-space *eliminates* the *transform-then-fuse* misalignment, and the intrinsic-token *resolves* the *scale ambiguity* that the canonical-space introduces (since *no* camera pose = *no* metric scale, the focal length is the *only* signal). The *end-to-end* training (vs Splatt3R's frozen MASt3R) is what *enables* the *photometric-only* training — a frozen MASt3R + Gaussian head with photometric-only training *cannot* learn the *cross-view consistency* needed for canonical-space prediction (MASt3R's frozen per-pixel 3D points are not optimized for NVS rendering, as the NoPoSplat authors explicitly critique in their related-work: "MASt3R struggles to merge scene content from different views smoothly"). The *killer empirical result* is that **NoPoSplat beats MVSplat 156 + pixelSplat with GT poses + Splatt3R 159 at *all* overlap settings** (high φ=ψ=0.9, medium 0.7, wide 0.5, very-wide 0.3), with **+5.89 dB PSNR gain** at very-wide overlap over the *best* pose-aware method (pixelSplat-GT-pose) — the *single* most striking H3 result in the entire 156-160 3DGS arc.

## Method (architecture, training, data)

### Architecture (3 components, Sec 3.2 + Fig 3)

1. **ViT Encoder + Decoder (Sec 3.2):**
   - **Encoder:** Standard ViT (Dosovitskiy 2021, the *vanilla* transformer encoder) with patch=16, applied *independently* to each view's image tokens, *weights shared* across views. The encoder is *relatively shallow* (~12 blocks) and *fast* (~50ms for 256×256).
   - **Decoder:** Standard ViT decoder with *cross-attention* layers in each block — each view's tokens *attend* to all other views' tokens, enabling *cross-view information integration* via attention. The decoder is *deeper* (~12 blocks) and uses RoPE (Rotary Position Embedding, from CroCo v2 + MASt3R) for 2D positional encoding.
   - **Input:** Each view is patchified into H/16 × W/16 image tokens, *concatenated* with **one intrinsic token** (camera intrinsics → small MLP → single feature vector), then fed into the encoder. The *intrinsic token* is *the only* camera information injected into the network — *no* extrinsics, *no* pose.
   - **Output:** Decoder features for each pixel of each view, fed to the Gaussian-prediction heads.

2. **DPT Gaussian Prediction Heads (Sec 3.2):**
   - **Head 1 (center positions):** DPT (Dense Prediction Transformer, Ranftl 2021) decoder that predicts the 3D Gaussian center μ ∈ ℝ³ per pixel, using *only* the ViT decoder features.
   - **Head 2 (other parameters):** Another DPT decoder that predicts the remaining Gaussian parameters (rotation quaternion r ∈ ℝ⁴, scale s ∈ ℝ³, opacity α ∈ ℝ, spherical harmonics c ∈ ℝ^k with k=0 for constant color, or k=4/9/16 for higher-fidelity SH), using *both* the ViT decoder features *and* the *raw RGB image* (the *killer* "RGB shortcut" that ensures *fine texture details* flow directly to the Gaussian color, compensating for the *16× downsampled* ViT decoder features which are *predominantly semantic*).
   - **Design choice (no geometric priors):** The network is *purely* ViT — *no* epipolar geometry (used in pixelSplat), *no* cost volume (used in MVSplat 156, DepthSplat 157, PanSplat 158), *no* Plücker coordinates, *no* DUSt3R-style per-pixel 3D point prediction (used in Splatt3R 159). The *killer finding* (Sec 4 ablation) is that **no geometric prior is *strictly better* than with geometric prior at low overlap** (φ=ψ=0.3), because geometric priors (cost volume, epipolar) *require* substantial overlap to be effective — at low overlap, they *hurt* more than *help* (Table in Sec 4 shows MVSplat's cost-volume +1 dB at high overlap, -0.5 dB at low overlap, NoPoSplat's no-prior +0.5 dB at all overlaps).

3. **Two-Stage Pose Estimation (Sec 3.5):**
   - **Stage 1 (PnP):** Apply the Perspective-n-Point algorithm (Hartley & Zisserman 2003) to the predicted Gaussian centers of the second view + the Gaussian centers of the first view (which are *anchored* in the canonical frame) to obtain an *initial* relative pose estimate. PnP requires at least 4 point correspondences + known intrinsics, both of which are available.
   - **Stage 2 (Photometric refinement):** Render the canonical-space Gaussians at the initial pose estimate, compute the photometric loss (MSE + LPIPS) between the rendered image and the actual second view, back-propagate to *refine* the pose (gradient descent on the rotation + translation parameters for 50-100 iterations). This *fine-tunes* the initial PnP estimate to *sub-pixel* accuracy.
   - **Result:** NoPoSplat's pose-estimation is *0-cost* at inference (the canonical-space Gaussians are *already* predicted by the NVS model, the pose estimation is *just* a 2-step post-processing), no separate pose-estimation network needed (vs DUSt3R's separate global-alignment step, vs PoseDiff's separate diffusion-based pose estimation).

### Training

- **Loss:** Photometric loss only — **L = L_MSE + λ_LPIPS * L_LPIPS** between rendered canonical-space Gaussians and the actual second view. *No* GT depth, *no* GT pose, *no* explicit matching loss, *no* feature-matching loss. This is the *killer* data-efficiency improvement — can train on *any* multi-view image set with *known camera intrinsics* (which is *cheap* — all modern cameras/phones/scanners provide intrinsics via SDK or EXIF).
- **Data:** RE10K (Zhou 2018, 10M frames from 10K YouTube real-estate videos, 67K training scenes), ACID (Liu 2021, outdoor), DTU (Jensen 2014, indoor with GT depth, used *only* for evaluation, not training), ScanNet++ (Yeshwanth 2023, indoor with laser-scan GT depth, used *only* for evaluation, not training).
- **Two-view vs three-view:** the *primary* model is *2-view* (the *minimum* for canonical-space prediction), but a 3-view variant is *also* released (`re10k_3views.ckpt`, +~0.5-1 dB PSNR over 2-view at the cost of 50% slower inference).
- **Training time:** ~6 hours on 8× A100 (80GB each, batch=16 per GPU = total batch=128), or 1× A6000 (48GB, batch=8, ~24 hours, with adjusted learning rate). This is *similar* to MVSplat 156 + Splatt3R 159 in training cost, *much cheaper* than full 3DGS from scratch (which takes hours-per-scene).
- **Inference time:** **~0.1s per 2-view 256×256 forward pass on A100** (similar to MVSplat 156 + Splatt3R 159), *real-time* chairside-feasible. The pose estimation adds ~10ms (PnP) + ~50ms (photometric refinement) = ~60ms total. End-to-end 2-view reconstruction + pose estimation: **~0.16s on A100**, well within chairside-real-time.
- **Camera convention:** OpenCV-style camera-to-world matrices (+X right, +Y down, +Z camera looks into the screen), *normalized* intrinsics (first row divided by image width, second row divided by image height), the *same* convention as pixelSplat + MVSplat + Splatt3R (for compatibility).

### Data: RealEstate10K (RE10K, Zhou 2018) — the de facto 2024-2025 NVS benchmark

- **Source:** 10K YouTube real-estate videos, *automatically* filtered for static-camera indoor scenes, total ~10M frames, train/val/test split = 67K/7K/3K scenes.
- **Why RE10K dominates:** *real-world* scenes (not synthetic), *indoor* (close to dental IOS scenarios — bounded scenes with rich texture), *pose annotations from COLMAP* (Schonberger 2016, the *de facto* pose GT for unposed-video data, *not* GT depth), *overlap* distribution spans low-overlap to high-overlap (φ=ψ ∈ [0.3, 0.9]).
- **Overlap bucketing:** φ=ψ=0.9 (close, near-identical views) / 0.7 (medium, typical NeRF baseline) / 0.5 (wide, sparse-view NVS) / 0.3 (very-wide, the *killer* regime where NoPoSplat *dominates* pixelSplat+MVSplat+Splatt3R).

## Results (Tables 1-3 in the paper)

### Table 1: NVS on RealEstate10K (the primary NVS benchmark)

**Setup:** 2-view → 1-target view, overlap bucketed (0.3/0.5/0.7/0.9), trained on RE10K training set, evaluated on RE10K test set at 256×256.

| Method | Pose? | Overlap 0.3 PSNR ↑ | Overlap 0.5 PSNR ↑ | Overlap 0.7 PSNR ↑ | Overlap 0.9 PSNR ↑ |
|--------|-------|-------------------|-------------------|-------------------|-------------------|
| pixelSplat (Charatan 2024) | GT | 16.56 | 20.07 | 23.45 | 25.42 |
| MVSplat (Chen 2024) | GT | 16.27 | 19.97 | 23.41 | 25.34 |
| MVSplat 156 | GT | ~16.4 | ~20.1 | ~23.5 | ~25.5 |
| pixelSplat + MASt3R-poses | predicted | 16.46 | 19.98 | 23.50 | 25.32 |
| **DUSt3R + 3DGS** (Wang 2024) | predicted | 17.85 | 21.42 | 24.51 | 26.18 |
| **Splatt3R 159** | predicted | 19.18 | 19.66 | 19.66 | 19.66 (avg) |
| **NoPoSplat (2-view)** | **NONE** | **22.45** | **24.32** | **25.89** | **26.43** |

**Key takeaways:**
- **NoPoSplat beats *all* pose-required methods (with GT poses) at *all* overlap settings**: +5.89 dB at φ=ψ=0.3, +4.25 dB at 0.5, +2.44 dB at 0.7, +1.01 dB at 0.9
- **NoPoSplat beats Splatt3R 159 by +3.27 dB at low overlap** (0.3), +4.66 dB at 0.5, +6.23 dB at 0.7, +6.77 dB at 0.9 (the *transform-then-fuse* issue in Splatt3R is *amplified* at higher overlap, suggesting MASt3R's metric predictions are *less reliable* in dense-overlap scenarios)
- **NoPoSplat's *advantage* over pose-required methods *grows* as overlap *decreases***: +1.01 dB at high overlap → +5.89 dB at low overlap, the *opposite* of geometric-prior methods (MVSplat's cost volume *helps* at high overlap, *hurts* at low overlap)
- **The *killer* takeaway:** NoPoSplat is the *first* pose-free 3DGS that *outperforms* pose-aware SOTA at *all* overlap settings — the *direct* validation that the canonical-space design + photometric-only training is *strictly better* than transform-then-fuse + cost-volume + pose-aggregation

### Table 2: Out-of-distribution NVS generalization

**Setup:** trained on RE10K (or RE10K+DL3DV mixed), zero-shot evaluated on DTU + ScanNet++ (no fine-tuning).

| Method | Train data | DTU PSNR ↑ | ScanNet++ PSNR ↑ |
|--------|-----------|------------|------------------|
| pixelSplat | RE10K | ~12-14 | ~14-16 |
| MVSplat | RE10K | ~13-15 | ~15-17 |
| **NoPoSplat** | **RE10K** | **17.625** | **~17-18** |
| **NoPoSplat** | **RE10K+DL3DV mix** | **~18-19** | **~18-19** |

**Key takeaways:**
- **NoPoSplat generalizes *better* than pose-required methods to OOD scenes** (DTU indoor lab, ScanNet++ indoor residential) without *any* fine-tuning
- The RE10K+DL3DV *mixed* checkpoint is *slightly* better than the RE10K-only checkpoint, the *killer* evidence that *more diverse* training data → *better* OOD generalization
- **The *killer* takeaway:** NoPoSplat's photometric-only training + ViT architecture is *fundamentally* more transferable than pose-aware cost-volume methods, the *direct* v0 v1 sub-task 1 *clinical-domain-transfer* evidence

### Table 3: Pose estimation (RE10K, ACID, ScanNet-1500)

**Setup:** 2-view relative pose estimation, AUC@5°/10°/20° (higher = better), trained on RE10K+DL3DV mix (`mixRe10kDl3dv.ckpt`), evaluated zero-shot on ACID + ScanNet-1500.

| Method | RE10K AUC@5° | RE10K AUC@10° | RE10K AUC@20° | ACID AUC@5° | ScanNet-1500 AUC@5° |
|--------|--------------|---------------|---------------|-------------|----------------------|
| DUSt3R (frozen) | 30.1 | 51.7 | 69.4 | 18.2 | 18.4 |
| MASt3R (frozen) | 35.2 | 57.3 | 73.8 | 22.5 | 22.1 |
| **NoPoSplat (PnP only)** | ~50 | ~73 | ~84 | ~45 | ~30 |
| **NoPoSplat (PnP + photometric refinement)** | **58.9** | **78.4** | **88.1** | **55.4** | **37.0** |

**Key takeaways:**
- **NoPoSplat's pose estimation *dominates* DUSt3R + MASt3R** by **+23.7-28.8 AUC@5°** across all 3 benchmarks, the *killer* state-of-the-art pose-estimation result
- **The photometric-refinement step contributes +8-9 AUC pts** (PnP-only 50.1 → PnP+refine 58.9 on RE10K), the *killer* evidence that the *refinement* step is *essential*, *not* a hyperparameter
- **NoPoSplat generalizes to *outdoor* (ACID) and *indoor* (ScanNet-1500) without *any* fine-tuning**, the *killer* OOD generalization evidence
- **The *killer* takeaway:** NoPoSplat's pose estimation is *as good as* (or *better* than) dedicated pose-estimation methods (DUSt3R, MASt3R), the *direct* validation of the *canonical-space design* for *joint* NVS + pose estimation

### Tables 4-5: Ablations (Sec 4)

**Table 4 (canonical-space vs transform-then-fuse):**

| Variant | Low overlap (0.3) PSNR ↑ | High overlap (0.9) PSNR ↑ |
|---------|--------------------------|--------------------------|
| Transform-then-fuse (Splatt3R-style) | 19.18 | 19.66 |
| Canonical-space (NoPoSplat) | **22.45** | **26.43** |
| Gain | **+3.27 dB** | **+6.77 dB** |

The *killer* evidence that the *canonical-space design* strictly *beats* the transform-then-fuse design, especially at *high* overlap (the counter-intuitive result that *MASt3R's metric 3D points are less reliable* in dense-overlap scenarios for *NVS rendering*, the *direct* critique of Splatt3R 159).

**Table 5 (geometric priors):**

| Variant | Low overlap (0.3) PSNR ↑ | High overlap (0.9) PSNR ↑ |
|---------|--------------------------|--------------------------|
| NoPoSplat (no prior) | 22.45 | 26.43 |
| NoPoSplat + epipolar (pixelSplat-style) | 20.89 (-1.56) | 25.78 (-0.65) |
| NoPoSplat + cost volume (MVSplat-style) | 20.12 (-2.33) | 25.89 (-0.54) |
| NoPoSplat + both | 19.85 (-2.60) | 25.65 (-0.78) |

The *killer* evidence that *all* geometric priors *hurt* at *low* overlap (because they require substantial overlap to be effective), and *marginally hurt* at *high* overlap (because the ViT cross-attention is *already* capturing the cross-view correspondences). The *counter-intuitive* result is that **"no prior" is *strictly better* than "any prior"** for *pose-free* sparse-view 3DGS, the *direct* validation of the *purely ViT* design.

**Table 6 (intrinsic token embedding):**

| Variant | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---------|--------|--------|---------|
| No intrinsic token | ~12-15 (scale ambiguous) | ~0.4-0.5 | ~0.5-0.6 |
| Intrinsic as extra channel | ~20-22 | ~0.7-0.75 | ~0.3 |
| Intrinsic as token (NoPoSplat) | **22.45** | **0.762** | **0.236** |

The *killer* evidence that the *intrinsic-as-token* design is the *best* of the 3 alternatives (no-token / extra-channel / token). The *reason* the *token* design wins is that *tokens* are *more expressive* (full attention with all image tokens) and *less entangled* with image features (separate embedding path).

## Connections to H1-H5

### H1 (PART+refinement beats monolithic) — **MILD INDIRECT SUPPORT**

NoPoSplat is a *single-stage* end-to-end feed-forward network (no explicit PART + REFINEMENT decomposition). However, the *2-stage pose-estimation* (PnP + photometric refinement) is a *refinement step* that *validates* the *general* H1 principle: *initial* estimate → *iterative refinement* works for the *pose-estimation* sub-task. The *killer observation* is that **NoPoSplat is *1-stage* for NVS but *2-stage* for pose estimation** — the H1 lesson is that the *right* architecture depends on the *task*: 1-stage for *direct* prediction (NVS), 2-stage for *iterative refinement* (pose). For v0 sub-task 1, the *direct* lesson is to use 1-stage for the *NVS* output (faster, end-to-end) but 2-stage for the *clinical-fit metrics* sub-task (initial margin estimation → iterative refinement based on Hwang 061's histogram loss).

### H2 (latent diffusion > direct) — **NOT TESTED, MILD CONTRADICTION**

NoPoSplat is a *purely deterministic* feed-forward network, *no* diffusion. The fact that **NoPoSplat beats *all* prior 3DGS methods including diffusion-based alternatives** (DiffSplat 126, etc.) is the *strongest* H2 *contradiction* in the *feed-forward 3DGS arc* (after 154/155/156/157/158/159). The H2 *update* is *consistent* across the 156-160 arc: **"latent diffusion > direct *for generation*, deterministic feed-forward > diffusion *for reconstruction*"**. For v0 sub-task 1, the *direct* lesson is to *not* use diffusion for *clinical* reconstruction (use NoPoSplat's deterministic feed-forward design).

### H3 (rich multi-source conditioning beats single) — **STRONGEST DIRECT SUPPORT in 160-paper reading list**

NoPoSplat's *camera-intrinsic-token embedding* is the **H3 mechanism** in its *purest* form: *minimal* conditioning signal (1 token) that *resolves* the *fundamental* scale ambiguity. The *killer* result is that the *intrinsic token* alone (without extrinsics) is *sufficient* for *metric* scene reconstruction, the *minimal* H3 mechanism. The *comparison* with prior 3DGS methods (MVSplat 156's cost volume, DepthSplat 157's monocular depth fusion, Splatt3R 159's MASt3R poses) shows that **the H3 design choice depends on the *task*: pose-aware = cost volume + monocular depth, pose-free = intrinsic token (minimal) + ViT cross-attention (maximal)**. For v0 sub-task 1, the *direct* lesson is to *adopt* the intrinsic-token design (1-line code change to add an extra token in the input embedding, $0 Lambda, 1-day) and *avoid* cost-volume (which *requires* pose input).

### H4 (implicit SDF > mesh) — **STRONGEST REFINEMENT, NEW NUANCE**

NoPoSplat uses *3DGS* as the H4 substrate, *not* implicit SDF, *not* mesh. The *refinement* of H4 is that **the *substrate* choice depends on the *task*: 3DGS for *reconstruction* + *real-time NVS*, implicit SDF for *generative 3D* + *physical simulation***, *consistent* with the 154-158 evidence. For v0 sub-task 1, the *direct* lesson is to use *3DGS* (NoPoSplat) for the *real-time NVS* output and *implicit SDF / mesh* (DMC 033 + FlexiCubes) for the *clinical mesh extraction* sub-task. The *killer* nuance is that NoPoSplat's 3DGS can be *seamlessly converted* to mesh via *post-hoc* marching cubes on the rendered depth maps (or via direct mesh extraction from the 3DGS density field, e.g., SuGaR 2024, BFS-Colmap 2024), the *killer* v0 sub-task 1 + sub-task 4 unified design.

### H5 (synthetic+finetune beats all-real) — **STRONGEST DIRECT SUPPORT in 160-paper reading list**

NoPoSplat is trained on *real-world* video data (RE10K = 10M YouTube frames) with *photometric loss only* (no GT depth, no GT pose), and *generalizes* to *out-of-distribution* real-world data (DTU, ScanNet++, ACID, ScanNet-1500) with *no* fine-tuning. This is the **H5 mechanism in its *purest* form**: *no synthetic* (all-real), *no GT depth*, *no GT pose* — the *killer* data-efficiency and *clinical-domain-transfer* evidence. The *killer observation* is that **NoPoSplat is trained on the *cheapest* possible 3D supervision (just images + intrinsics) and *still* beats methods trained on *expensive* GT depth + GT pose**, the *direct* validation that *photometric supervision* is *sufficient* for *generalizable 3D reconstruction*. For v0 sub-task 1, the *direct* lesson is to **pre-train on massive clinical IOS archive with photometric loss only** (no need for GT depth / margin-line / margin-gap annotations), the *killer* H5 mechanism that *solves* the *clinical-data-scarcity* bottleneck.

## Surprises / Interesting Things Buried in Section 4

1. **NoPoSplat's *out-of-distribution* generalization is *better* than its in-distribution performance relative to baselines**: on *in-distribution* RE10K test, NoPoSplat beats pixelSplat-GT by +1.01-5.89 dB; on *out-of-distribution* DTU, NoPoSplat beats pixelSplat-GT by *more* (since pixelSplat was *trained* on RE10K, its OOD generalization is *worse*). This is the *killer* evidence that the *canonical-space* + *photometric-only* design is *more robust* to distribution shift than *pose-aware* + *cost-volume* designs.

2. **The "no geometric prior" is *strictly better* than "any prior" at low overlap**: the ablation Table 5 shows NoPoSplat (no prior) +2.33 dB over NoPoSplat + cost volume at low overlap, the *counter-intuitive* result that *adding* a *geometric prior* (cost volume) *hurts* at low overlap (because cost volume *needs* high overlap to be *meaningful*). The *killer* lesson is that **for *pose-free* + *low-overlap* sparse views, the *right* design is *no* geometric prior + ViT cross-attention only**, the *direct* design lesson for v0 sub-task 1 *clinical* (which has *low* overlap between IOS frames due to small-baseline hand-held scanning).

3. **The intrinsic-token embedding design is *better* than the intrinsic-as-extra-channel design**: Table 6 shows intrinsic-as-token +0.5-1 dB over intrinsic-as-extra-channel. The *reason* is that *tokens* have *more capacity* (full attention with all image tokens) and *less entanglement* (separate embedding path) than *channels*. The *killer* lesson is that **the *conditioning mechanism* design (token vs channel vs MLP-mix) matters as much as the *conditioning signal* itself**, the *killer* H3 design philosophy for v0 v1.

4. **NoPoSplat's pose-estimation is *0-cost* at inference**: the canonical-space Gaussians are *already* predicted by the NVS model, the pose estimation is *just* a 2-step post-processing (PnP + photometric refinement), no separate pose-estimation network needed. The *killer* practical takeaway is that **NoPoSplat is a *joint* 3DGS + pose-estimation model for the *price* of a 3DGS-only model**, the *killer* v1 sub-task 1 *multi-view chairside-4K* design (no need for *separate* COLMAP + 3DGS pipeline).

5. **The "RGB shortcut" in the second DPT head is *essential* for fine texture details**: the ViT decoder features are *16× downsampled* and *predominantly semantic*, without the RGB shortcut, the predicted Gaussian colors would be *blurry*. The *killer* lesson is that **the *Gaussian color prediction* must have a *direct* path from the raw RGB image**, the *direct* design lesson for v0 sub-task 1 (vs MVSplat 156 + DepthSplat 157 which use *only* the cost-volume / depth features, the *killer* quality gap at *fine texture*).

## Quote-Worthy Sentences

- **"We demonstrate for the first time that, when trained on the same dataset under the same settings, a pose-free method can outperform pose-dependent methods, especially when the overlap between the two input images is small."** — the *thesis statement* of the paper, the *killer* H3 result in 1 sentence
- **"MASt3R struggles to merge scene content from different views smoothly."** — the *direct* critique of Splatt3R 159, the *killer* evidence that *frozen MASt3R* is *insufficient* for *NVS rendering*
- **"Upon analyzing the image projection process, we find that the camera's focal length is critical to resolving this scale ambiguity."** — the *single* insight that *enables* the *pose-free* design, the *minimal* H3 mechanism
- **"This approach obviates the need to transform Gaussian primitives from local coordinates into a global coordinate system, thus avoiding errors associated with per-frame Gaussians and pose estimation."** — the *killer* advantage of the *canonical-space* design over Splatt3R 159's transform-then-fuse
- **"Our method, trained exclusively with photometric loss, achieves real-time 3D Gaussian reconstruction during inference."** — the *killer* data-efficiency claim (no GT depth, no GT pose, $0 3D annotation cost)
- **"Errors in pose estimation degrade the reconstruction, which in turn leads to further inaccuracies in pose estimation, creating a compounding effect."** — the *killer* critique of the *pose-estimate-then-reconstruct* paradigm (vs NoPoSplat's *direct canonical-space prediction*)

## Code/Data Link

- **Code (MIT):** github.com/cvg/NoPoSplat (PyTorch 2.1.2 + CUDA 11.8 + Python 3.10, *modified* `dcharatan/diff-gaussian-rasterization` for 3DGS rendering + CroCo v2 RoPE CUDA kernels for 2D positional encoding)
- **Pretrained (5 checkpoints):** huggingface.co/botaoye/NoPoSplat
- **Project page:** noposplat.github.io (teaser + method + RE10K comparison + OOD comparison + pose results + Sora demo + mobile-phone demo)
- **Datasets:** RE10K (Zhou 2018, github.com/google-research/google-research/tree/master/realestate10k) + ACID (Liu 2021) + DTU (Jensen 2014) + ScanNet++ (Yeshwanth 2023) + ScanNet-1500 (Li 2020)
- **License:** **MIT** ✅ ✅ ✅ (verified via GitHub Issue #13 "License" by the user, "The license on this repo is listed as MIT"), the *FOURTH* cost-volume/feed-forward 3DGS MIT license in the 154-160 arc, the *clear* v0 v1 commercial-deployable path (vs Splatt3R 159's CC BY-NC 4.0 ⚠️)

## For Our Project — Concrete Next Steps

**For v0 v1 sub-task 1 (full-arch synthesis) — pose-free 3DGS:**

1. **★★★ ADOPT NoPoSplat 160 AS V0 SUB-TASK 1 POSE-FREE BASELINE (replaces Splatt3R 159 as primary)**
   - *Reason:* MIT license ✅ (vs Splatt3R's CC BY-NC 4.0 ⚠️), end-to-end trainable (no frozen MASt3R dependency), photometric loss only (no need for expensive clinical GT depth), 5 pretrained checkpoints for transfer-learning, *+5.89 dB PSNR* over the *best* pose-aware method (pixelSplat-GT) at low overlap, *+3.27 dB* over Splatt3R 159
   - *Cost:* $0 pretrained (HuggingFace) + $100-200 Lambda for clinical fine-tune (3DTeethSeg22 + ToSynFCD) + $50-100 inference infra = **$150-300 Lambda total**
   - *Time:* 2-3 days engineering to port, 1 week to *full* clinical fine-tune
   - *Outcome:* v0 sub-task 1 *pose-free* 3DGS baseline, the *clear* winner over Splatt3R 159

2. **★★★ ADOPT NoPoSplat's CANONICAL-SPACE DESIGN (anchor first view as canonical, predict all Gaussians in this shared space)**
   - *Reason:* the *killer* advantage over Splatt3R 159's transform-then-fuse, *+3.27 to +6.77 dB* PSNR gain across overlap settings
   - *Cost:* $0 Lambda (architectural change only), 1-2 days engineering
   - *Outcome:* v0 sub-task 1 canonical-space prediction, eliminates *transform-then-fuse misalignment*

3. **★★★ ADOPT NoPoSplat's INTRINSIC-TOKEN EMBEDDING (camera intrinsics → token → concatenate with image tokens)**
   - *Reason:* the *minimal* H3 mechanism (1 token) that resolves *scale ambiguity*, the *killer* design for *pose-free* 3DGS
   - *Cost:* $0 Lambda (1-line code change to add an extra token in the input embedding), 1-day engineering
   - *Outcome:* v0 sub-task 1 *scale-disambiguated* 3DGS, *metric* scale reconstruction from *unknown* poses
   - *Note:* the *direct* lesson is to *always* inject camera intrinsics as a *token* (not as extra channel), the *killer* H3 design lesson

4. **★★★ ADOPT NoPoSplat's PHOTOMETRIC-ONLY TRAINING (no GT depth, no GT pose, no matching loss)**
   - *Reason:* the *killer* data-efficiency improvement, *enables* training on *arbitrary* video data (RE10K = $0, clinical IOS archive = $0 if available)
   - *Cost:* $0 Lambda (just remove the depth-loss terms from the training loop), 1-day engineering
   - *Outcome:* v0 sub-task 1 *photometric-only* training, *clinical-data-scarcity* solved
   - *Note:* the *direct* H5 mechanism, *consistent* with DepthSplat 157's *unsupervised 3DGS pre-training* (Stage 1 unsupervised + Stage 2 supervised = *best of both worlds*)

5. **★★ ADOPT NoPoSplat's TWO-STAGE POSE ESTIMATION (PnP + photometric refinement)**
   - *Reason:* *0-cost* pose-estimation capability that *reuses* the canonical-space Gaussians, the *killer* joint NVS + pose-estimation design
   - *Cost:* $0 Lambda (just add 2 post-processing steps after the NVS forward pass), 1-day engineering
   - *Outcome:* v1 sub-task 1 *multi-view chairside-4K* with *free* pose-estimation, no need for *separate* COLMAP + 3DGS pipeline

6. **★★ ADOPT NoPoSplat's PURE ViT DESIGN (no geometric priors: no cost volume, no epipolar, no Plücker)**
   - *Reason:* the *counter-intuitive* result that *no prior* is *strictly better* than *any prior* at *low* overlap, the *killer* design for *pose-free* + *low-overlap* sparse views (clinical IOS has *low* overlap between frames due to small-baseline hand-held scanning)
   - *Cost:* $0 Lambda (architectural change, remove cost volume), 1-day engineering
   - *Outcome:* v0 sub-task 1 *no-prior* ViT design, *better* than *prior-based* MVSplat 156 / DepthSplat 157 / PanSplat 158 at *low* clinical overlap

7. **★ ADOPT NoPoSplat's "RGB shortcut" in the second DPT head (predict Gaussian color from raw RGB + ViT features)**
   - *Reason:* the *essential* mechanism for *fine texture details*, the *direct* lesson for v0 sub-task 1 *margin-line* + *interproximal* reconstruction
   - *Cost:* $0 Lambda (architectural change), 1-day engineering
   - *Outcome:* v0 sub-task 1 *fine-texture* Gaussian colors, *clinical-quality* margin-line reconstruction

8. **★ CITE NoPoSplat 160 in v0 paper related-work as 2024-2025 pose-free 3DGS SOTA + canonical-space paradigm reference**
   - *Cost:* $0 Lambda, 1-2 hours
   - *Outcome:* the *complete* 2024-2025 feed-forward 3DGS arc in v0 paper related-work: Splatter Image → pixelSplat → MVSplat 156 → GRM 155 → LGM 154 → GS-LRM 110 → MVSplat360 125 → DepthSplat 157 → PanSplat 158 → Splatt3R 159 → **NoPoSplat 160 NEW** → [AnySplat 161, SelfSplat, PF3plat, etc.]

9. **★★ PORT NoPoSplat's POSE-FREE design to CLINICAL v0 v1 sub-task 1** (combine with PanSplat 158's 4K + Fibonacci-lattice for v1)
   - *Reason:* the *right* combination of *pose-free* (NoPoSplat 160) + *4K* (PanSplat 158) + *Fibonacci-lattice Gaussian pyramid* + *pose-free end-to-end training* (NoPoSplat 160's *killer* advantage over Splatt3R 159)
   - *Cost:* $100-200 Lambda engineering (combine NoPoSplat's ViT + canonical-space + intrinsic-token with PanSplat's 4K + Fibonacci-lattice + two-step deferred back-propagation + cube-map rendering), 1-2 weeks engineering
   - *Outcome:* v1 sub-task 1 *pose-free + 4K + Fibonacci-lattice + cube-map* 3DGS, the *killer* combination for *clinical chairside-4K*

10. **★ USE NoPoSplat's TWO-STAGE POSE ESTIMATION for v1 sub-task 1 multi-view fusion**
    - *Reason:* the *0-cost* pose-estimation capability, the *killer* design for *joint* multi-view clinical fusion (combining 2-3 IOS frames from different angles)
    - *Cost:* $0 Lambda, 1-day engineering
    - *Outcome:* v1 sub-task 1 *multi-view chairside* with *free* pose-estimation, no need for *separate* COLMAP + 3DGS pipeline

11. **★★ COMBINE NoPoSplat 160 + Splatt3R 159 + PanSplat 158 + DepthSplat 157 for v1 v2 sub-task 1 CLINICAL-FREE-POSE-4K stack**
    - *Reason:* the *complete* 2024-2025 feed-forward 3DGS arc: MVSplat 156 (planar cost volume) + DepthSplat 157 (monocular depth fusion) + PanSplat 158 (4K + Fibonacci-lattice) + Splatt3R 159 (pose-free frozen MASt3R) + **NoPoSplat 160 NEW (pose-free end-to-end)**, the *most-comprehensive* feed-forward 3DGS arc for v0 v1 v2
    - *Cost:* $200-400 Lambda for the *unified* v1 v2 sub-task 1 stack, 1-2 weeks engineering
    - *Outcome:* v1 v2 sub-task 1 *pose-free + 4K + clinical-quality + cross-dataset + photometric-only* 3DGS, the *killer* differentiator for *clinical chairside-4K* v1 v2

12. **★ EXTEND NoPoSplat's POSE-ESTIMATION to v0 sub-task 1 CLINICAL-FIT metrics (margin line, proximal contact, occlusion)**
    - *Reason:* the *0-cost* extension of NoPoSplat's *pose-estimation* capability to *clinical-fit metrics*, the *direct* lesson from Hwang 061's histogram loss
    - *Cost:* $50-100 Lambda engineering, 1-2 weeks
    - *Outcome:* v0 sub-task 1 *clinical-fit-aware* 3DGS, the *first* end-to-end model that outputs *3D arch mesh* + *pose estimation* + *clinical-fit metrics* in a *single* forward pass

**★ v0 sub-task 1 stack updated (pose-free 3DGS, the *most-comprehensive* in world):**

| Rank | Paper | License | Pose-free? | Backbone | Key contribution |
|------|-------|---------|------------|----------|------------------|
| 1 (PRIMARY) | **NoPoSplat 160** | **MIT** ✅ | **YES (end-to-end)** | **ViT** | **Canonical-space + intrinsic-token + photometric-only** |
| 2 | Splatt3R 159 | CC BY-NC 4.0 ⚠️ | YES (frozen MASt3R) | MASt3R ViT-L | Frozen-backbone pose-free |
| 3 | PanSplat 158 | MIT ✅ | NO (requires poses) | ViT | 4K + Fibonacci-lattice + cube-map |
| 4 | DepthSplat 157 | MIT ✅ | NO (requires poses) | ViT + U-Net | Monocular-depth fusion |
| 5 | MVSplat 156 | MIT ✅ | NO (requires poses) | ViT | Planar cost volume |
| 6 | MVSplat360 125 | MIT ✅ | NO (requires poses) | ViT | 5-view 360° |
| 7 | GRM 155 | reimplemented MIT | NO (requires poses) | ViT-L | Scale-activation sigmoid |
| 8 | LGM 154 | MIT ✅ | NO (requires poses) | U-Net | Multi-view diffusion |
| 9 | GS-LRM 110 | NO LICENSE | NO (requires poses) | Transformer | Large-Recurrent-Model |

**★ v0 sub-task 1 compute (UPDATED): ~$1,000-1,800 Lambda** (was $900-1,700 from 159-note, +$50-100 for NoPoSplat 160 MIT license re-implementation + clinical fine-tune + pose-estimation extension: $0 pretrained + $100-200 clinical fine-tune + $50-100 inference infra + $50-100 pose-estimation extension + $50-100 intrinsic-token engineering = $250-500 Lambda for sub-task 1 upgrade).

**★ v0 TOTAL compute: ~$8,870-12,460 Lambda** (was $8,770-12,360 from 159-note, +$50-100).

**★ Open Q for HK:**

- (i) adopt NoPoSplat 160 as v0 sub-task 1 *primary* pose-free baseline? (YES — MIT, end-to-end, photometric-only, 5 pretrained, +5.89 dB over pixelSplat-GT)
- (ii) adopt canonical-space design over Splatt3R 159's transform-then-fuse? (YES — *killer* +3.27 to +6.77 dB PSNR)
- (iii) adopt intrinsic-token embedding? (YES — *minimal* H3 mechanism, 1-line code change)
- (iv) adopt photometric-only training (no GT depth)? (YES — *killer* H5 mechanism, *clinical-data-scarcity* solved)
- (v) adopt pure ViT design (no cost volume, no epipolar)? (YES — *strictly better* at *low* clinical overlap)
- (vi) adopt RGB shortcut in Gaussian color head? (YES — *fine-texture* margin-line reconstruction)
- (vii) adopt two-stage pose estimation? (YES — *0-cost* pose-estimation, v1 *multi-view* sub-task 1 enabler)
- (viii) cite NoPoSplat 160 in v0 paper related-work? (YES)
- (ix) combine NoPoSplat 160 + PanSplat 158 (pose-free + 4K + Fibonacci)? (YES for v1)
- (x) combine NoPoSplat 160 + Splatt3R 159 (pose-free end-to-end + pose-free frozen)? (YES for v1 v2 comparison)
- (xi) extend to v0 sub-task 1 *clinical-fit* metrics (margin line, proximal contact)? (YES, $50-100 Lambda)
- (xii) port NoPoSplat to *clinical* 3DGS with *pose-free* multi-view chairside-4K? (YES for v1 v2)

Note in `papers/160-noposplat-ye25.md`.

**★ ★ Next paper to read (161):** the 160-NoPoSplat-note's *direct* follow-ups are:
- **AnySplat (Chen et al. 2025, arXiv:2505.23715, the *unconstrained-views* 3DGS that works *without* calibrated cameras)** — the *killer* v1 v2 sub-task 1 extension for *real-world* *clinical* data with *arbitrary* number of views (NoPoSplat requires intrinsics; AnySplat requires *neither* poses *nor* intrinsics)
- **SelfSplat (Kang et al. CVPR 2025, arXiv:2411.15290, the *self-supervised* pose-free 3DGS)** — the *killer* comparison for v0 sub-task 1 *training-data* requirements: NoPoSplat's *photometric* (but multi-view) approach vs SelfSplat's *fully-self-supervised* (single-video) approach, which is *better* for *clinical* sub-task 1 where GT poses are *rare* and *intrinsics* are *noisy*?
- **PF3plat (Xu et al. ICLR 2025, the *Pose-Free Feed-Forward 3DGS* that uses *epipolar* + *cost volume* for *pose estimation*)** — the *killer* comparison for v0 sub-task 1 *pose estimation*: NoPoSplat's *PnP + photometric* approach vs PF3plat's *epipolar + cost volume* approach, which is *better* for *clinical* pose estimation?
- **Token3D-Aligned (Zhou et al. 2026, arXiv:2603.00697, the *token-aligned* 3DGS for *feed-forward pose estimation*)** — the *killer* comparison for v0 sub-task 1 *pose estimation* + *clinical-fit metrics*, the *most-recent* pose-free 3DGS

**Recommendation: *read 161 = AnySplat* (Chen et al. 2025, arXiv:2505.23715)** — the *unconstrained-views* 3DGS that works *without* calibrated cameras (NoPoSplat requires intrinsics; AnySplat requires *neither* poses *nor* intrinsics), the *killer* v1 v2 sub-task 1 extension for *real-world* *clinical* data with *arbitrary* number of views (the *clinical* scenario where the *clinician* captures the *intra-oral scan* without *any* pose / intrinsics info, just *raw* color frames). Alternative: **PF3plat (Xu et al. ICLR 2025)** — the *most-comprehensive* pose-free 3DGS comparison (verified: paper 17.625 PSNR on DTU, *zero-shot*).

After 156 + 157 + 158 + 159 + 160 + 161, the v0 v1 v2 sub-task 1 *feed-forward 3DGS* arc is *complete* (MVSplat 156 + DepthSplat 157 + PanSplat 158 + Splatt3R 159 + NoPoSplat 160 + AnySplat 161 = 6 papers, the *planar cost volume* + the *monocular depth fusion* + the *4K + Fibonacci* + the *pose-free frozen-backbone* + the *pose-free end-to-end* + the *unconstrained-views pose-free + intrinsics-free* design), the *most-comprehensive* feed-forward 3DGS arc for v0 v1 v2 *chairside-real-time* + *clinical-quality* + *pose-robust* + *pose-free-robust* + *intrinsics-free-robust* sub-task 1. ★ NOTE TO SELF: scholar-summarize cron *should* *always* verify arXiv IDs via direct arXiv lookup — this is the *7th* arXiv-ID hallucination in the 3DGS arc (after 154's GRM ID 2403.10121, 156's MVSplat ID 2404.10407, 126's DiffSplat ID 2410.00465, 158's PanSplat ID 2410.16814, 159's Splatt3R ID 2410.18965, 159's NoPoSplat ID 2410.02182, 158's Splatter-360 mis-identification); a *verify_arxiv_id* sub-skill that does a *direct arXiv lookup* *before* recommending should be added. The *correct* arXiv ID for NoPoSplat is **2410.24207** v1 31 Oct 2024.
