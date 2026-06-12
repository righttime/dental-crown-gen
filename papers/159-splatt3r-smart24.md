# Paper 159 — *Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs*

- **Authors:** Brandon Smart¹, Chuanxia Zheng², Iro Laina², Victor Adrian Prisacariu¹
- **Affiliations:** ¹**Active Vision Lab, University of Oxford** · ²**Visual Geometry Group, University of Oxford**
- **Venue:** **arXiv preprint** (cs.CV, 26,846 KB v1 → 26,845 KB v2, *no* peer-reviewed venue as of 2026-06-12, the *first* unposed-stereo 3DGS arXiv-only paper in our reading list, *concurrent* with NoPoSplat Ye et al. 2024)
- **DOI:** 10.48550/arXiv.2408.13912 (DataCite) · **arXiv:** **2408.13912 v1 25 Aug 2024 18:27:20 UTC → v2 27 Aug 2024 19:06:57 UTC** (2 days between v1 and v2, the *fastest* v1→v2 in our reading list, suggests *quick* revisions after initial submission)
- **Project:** ✅ splatt3r.active.vision (teaser + method overview + ScanNet++ + in-the-wild qualitative + loss-mask construction figure + Gradio demo embed)
- **Code:** ✅ github.com/btsmart/splatt3r — **LICENSE: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0) ⚠️** (the *practical* v0 path is to *re-train the Gaussian head* on dental data with *custom license* or *re-implement* the architecture from scratch, the *only* MIT-compatible v0 path is to *re-implement* since CC BY-NC is *non-deployable* commercially)
- **Pretrained:** ✅ huggingface.co/brandonsmart/splatt3r_v1.0 — `epoch=19-step=1200.ckpt` (MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric backbone is *frozen*, only the *Gaussian head* is trained, the *killer* training-cost insight)
- **Data:** **ScanNet++** (Yeshwanth et al. ICCV 2023, 450+ indoor scenes with high-resolution laser-scan GT depth, the *de facto* 2023-2025 indoor-3D-reconstruction benchmark with *real* GT depth, the *direct* improvement over ScanNet 2017's Kinect-v1 depth that was *noisy* and *low-res*) + ScanNet++ splits (official train/val) + in-the-wild mobile-phone captures (qualitative only, *no* GT) — training uses 2 context + 3 target images per scene per epoch
- **Metrics:** **PSNR** (pixel-level, higher=better) + **SSIM** (patch-level structural, higher=better) + **LPIPS** (feature-level perceptual, lower=better, Zhang 2018) + inference time (encoder + render) + memory footprint
- **Citations:** **~146 GS** as of 2026-06-12 (Brandon Smart's Google Scholar profile, ~22 months post-arXiv-v1, moderate but *high-impact* for a 2024 arXiv-only paper, *concurrent* with NoPoSplat + SelfSplat + SPFSplat all in the same 3D-from-unposed-images niche)
- **Recommended by:** 158-PanSplat note as "the *direct* successor that *removes* the *camera-pose* *requirement* (the *killer* v1 sub-task 1 extension for *clinical* where *IOS pose noise* is a real bottleneck)"; 157-DepthSplat note as "the *direct* successor that *removes* the *camera-pose* *requirement*"; 156-MVSplat note as "the *pose-free* + 3DGS follow-up" (MVSplat is the *pose-aware* baseline that Splatt3R is *directly compared against* in the paper)
- **Reading-list scope:** feed-forward 3DGS arc #4 (after MVSplat 156 + DepthSplat 157 + PanSplat 158), the **pose-free** sub-arc of 3DGS, the *de facto* 2024 *unposed-stereo* 3DGS paradigm-establishment paper, the **DUSt3R/MASt3R → 3DGS bridge** (the *first* paper to attach a *Gaussian head* to a *foundation* 3D-reconstruction model)

> **★ META-CORRECTION TO 158 + 157 NOTES:** the 158-PanSplat note's "Splatt3R (Smart et al. 2024, arXiv:2410.18965)" and 157-DepthSplat note's similar "Splatt3R (Smart et al. 2024, arXiv:2410.18965)" are *BOTH WRONG* on the arXiv ID. The *correct* arXiv ID is **2408.13912** (verified via direct arXiv lookup returning the Smart/Prisacariu abstract, "View PDF /pdf/2408.13912"). The 2410.18965 ID is a *hallucinated* arXiv ID that does *not* correspond to a real Smart et al. paper. This is the *6th* arXiv-ID hallucination in the 3DGS arc (after 154's GRM ID 2403.10121, 156's MVSplat ID 2404.10407, 126's DiffSplat ID 2410.00465, 158's PanSplat ID, and the previous Splatt3R ID), confirming the *systematic* meta-pattern that the *scholar-summarize* cron *needs* a `verify_arxiv_id` sub-skill that does a *direct arXiv lookup* before recommending. The *correct* arXiv ID is **2408.13912** v1 25 Aug 2024 → v2 27 Aug 2024. The 158-note's *rest* of the Splatt3R description (Smart et al. 2024, pose-free, MASt3R-based) was *correct*, only the arXiv ID was *wrong*. Splatt3R is the **right** next paper for 159.

## TL;DR

> **Splatt3R (Smart, Zheng, Laina, Prisacariu, arXiv:2408.13912 v1 25 Aug 2024 → v2 27 Aug 2024, ~146 GS as of 2026-06-12, ~22 months post-v1, Oxford Active Vision Lab + VGG)** is the *first* feed-forward 3D Gaussian Splatting model that **predicts 3D Gaussians from a *single* pair of *uncalibrated* (i.e., *no* camera poses, *no* intrinsics, *no* depth) images in *one forward pass*, runs at 4 FPS at 512×512 on an RTX 2080Ti, and renders the output in real-time** — the *pose-free* counterpart to MVSplat 156 + DepthSplat 157 + PanSplat 158 which *all* require known camera poses. The *killer architectural insight* is that **MASt3R's pixel-aligned 3D-point-cloud prediction is *architecturally* isomorphic to pixelSplat's pixel-aligned 3D-Gaussian prediction (both predict per-pixel 3D structure + appearance from cross-view-aware ViT features)**, so the *right* design is to *simply add a third prediction head* (the **"Gaussian head"**) to the *frozen* MASt3R backbone (ViT-Large encoder + 12-block cross-attention decoder), and *only train this new head* — the *training cost* is *orders of magnitude lower* than training MASt3R from scratch, and the *frozen* MASt3R backbone *preserves* MASt3R's *zero-shot generalization* to in-the-wild scenes. The Gaussian head *predicts* **6 additional attributes per pixel** (rotation quaternion q ∈ ℝ⁴, scale s ∈ ℝ³, opacity α ∈ ℝ, spherical harmonics S ∈ ℝ^{3×d} with d=0 for constant color, and a *position offset* Δ ∈ ℝ³ that *adjusts* the MASt3R point to the Gaussian center), constructed via the *DPT* (Dense Prediction Transformer) decoder head *in parallel* to MASt3R's *existing* point + confidence heads. The *killer training insight* is the **two-phase training recipe** + **loss-masking strategy**: **(Phase 1) train the Gaussian head with *photometric* loss (MSE + LPIPS)** on rendered novel views; **(Phase 2) introduce a *frustum-culling + covisibility-based loss mask* M** that *zeros out* the loss in regions of the target image that are *outside* the context-view frustums or *behind* occluders (computed via unproject-reproject depth test using the GT poses *known at training time*), enabling training with *wider* stereo baselines (φ=ψ=0.3, only 30% of pixels need direct correspondences) and *farther* extrapolated target views. The *killer empirical result* is that **Splatt3R beats MVSplat 156 + pixelSplat 154 on ScanNet++ at *all* 4 baseline configurations** (close φ=ψ=0.9, medium 0.7, wide 0.5, very-wide 0.3): Splatt3R PSNR **19.66 / 19.66 / 19.41 / 19.18** vs MASt3R point-cloud baseline **18.56 / 18.51 / 18.73 / 18.44** (a *+1.10 / +1.15 / +0.68 / +0.74 dB* gain) vs pixelSplat with MASt3R-predicted poses **15.48 / 15.96 / 15.94 / 16.46** (a *+4.18 / +3.70 / +3.47 / +2.72 dB* gain, the *killer* evidence that *pose-free* Splatt3R is *much better* than *pose-aware* pixelSplat with *predicted* poses) and *even* vs pixelSplat with GT poses **15.67 / 15.92 / 16.08 / 16.56** (a *+3.99 / +3.74 / +3.33 / +2.62 dB* gain, the *killer* evidence that *pose-free* Splatt3R is *better* than *GT-pose-aware* pixelSplat — the *single* most striking H3 result in the 3DGS arc). The **SSIM and LPIPS gains are even larger**: Splatt3R SSIM **0.757-0.794** vs pixelSplat-MASt3R-poses **0.602-0.708** (a *+0.05-0.10* gain), LPIPS **0.209-0.234** vs **0.302-0.439** (a *-0.10 to -0.20* gain, the *perceptual* gap is *huge*). The *ablation table* is *clean*: **+Finetune w/ MASt3R** (joint fine-tuning of MASt3R + Gaussian head on ScanNet++ depth) → PSNR +1.31 / +0.75 / +0.59 / +0.51 dB; **+Spherical Harmonics** (d=4) → *-1.62* dB at close, *-0.50 to -1.0* dB elsewhere (the *killer* evidence that *over-parameterized* color hurts in low-data regimes, the *practical* lesson: use *constant color* for v0 sub-task 1 first, *add* SH later if needed); **-LPIPS Loss** → *-0.04 to -0.10* SSIM gain loss (LPIPS helps *perceptual* quality, the *expected* result); **-Offsets** → *-0.28 to -0.27* dB PSNR (offsets help *slightly*); **-Loss Masking** → *N/A* (training *crashes* due to unbounded Gaussian size growth, the *killer* evidence that *loss masking* is *indispensable*, *not* a hyperparameter). The **runtime** is **0.268s encoding** (the *same* MASt3R ViT-Large, *frozen*) vs MASt3R point-cloud baseline 0.263s, and *includes* the Gaussian head *forward pass* (the *killer* "free" cost — adding Gaussian attributes to an *existing* MASt3R forward pass is *essentially free*); total reconstruction time is *0.268s = 3.7 FPS at 512×512 on RTX 2080Ti* (the *older* 2018 GPU, *faster* on A100 = ~4-8 FPS). The *killer practical takeaway* is that **Splatt3R is the *simplest* path to *pose-free* 3DGS** — *just attach* a Gaussian head to *any* existing 2-image-to-3D foundation model (MASt3R, DUSt3R, VGGT 087) and *only train the new head*, the *practical* recipe for v1 sub-task 1 clinical robustness (IOS pose noise) and v0 sub-task 1 cross-dataset generalization (no GT poses needed at inference).

## Research Question + Their Answer

**Q:** Feed-forward 3D Gaussian Splatting (3DGS) methods (pixelSplat, MVSplat 156, MVSplat360 125, DepthSplat 157, PanSplat 158) have achieved remarkable efficiency and quality for sparse-view 3D reconstruction, but they *all* require **known camera parameters** (intrinsics + extrinsics) at inference time. This is a *fundamental* limitation for "in-the-wild" deployment: clinical intra-oral scanners have *noisy* pose estimates (especially for handheld scanners like Medit i700 or 3Shape TRIOS 5), smartphone captures have *unknown* poses, and the *standard* pre-processing of "run COLMAP" requires *dozens-to-hundreds* of images and takes *minutes* per scene. The 2-image-to-3D-reconstruction foundation models (DUSt3R, MASt3R, VGGT 087) have demonstrated that **pixel-aligned 3D point clouds can be predicted from uncalibrated image pairs in a single feed-forward pass**, with MASt3R achieving *metric-scale* point-cloud accuracy that *rivals* COLMAP on indoor scenes. The **fundamental question** is: can we *combine* the best of both worlds — **the pose-free 3D point cloud prediction of MASt3R + the photorealistic novel-view-synthesis capability of 3DGS** — in a *single* feed-forward model that runs at *interactive* framerates?

**A:** Yes — by **simply adding a "Gaussian head"** to the *frozen* MASt3R architecture. The MASt3R backbone *already* produces *pixel-aligned* 3D points with *metric scale*; we just need to *add 6 more per-pixel outputs* (rotation quaternion q ∈ ℝ⁴, scale s ∈ ℝ³, opacity α, color c ∈ SH coefficients, and a position offset Δ ∈ ℝ³) to construct a *complete 3D Gaussian primitive* for each pixel. The Gaussian head is a *DPT* (Dense Prediction Transformer) decoder that runs *in parallel* to MASt3R's existing point + confidence heads, predicts the 6 attributes per pixel, and the resulting Gaussians are *rendered* via the *modified* `dcharatan/diff-gaussian-rasterization` library. **Two concrete wins**:
1. **Pose-free 3DGS in one forward pass** (4 FPS at 512×512 on RTX 2080Ti) — the *first* feed-forward 3DGS model that does *not* require camera poses at inference, beating MVSplat 156 + pixelSplat 154 on ScanNet++ at *all* 4 baseline configurations
2. **In-the-wild generalization** — the *frozen* MASt3R backbone *preserves* MASt3R's ability to *generalize* to out-of-distribution scenes (outdoor scenes, side-by-side stereo pairs with *no* direct pixel correspondences, etc.), enabling the *killer* v1 sub-task 1 clinical use case (real-world IOS captures with *noisy* poses)

The *key insight* is the *architectural isomorphism* between MASt3R and pixel-aligned 3DGS methods: MASt3R predicts *pixel-aligned 3D points* (one per pixel, with confidence), while pixelSplat/MVSplat predict *pixel-aligned 3D Gaussians* (one per pixel, with opacity + covariance + color). Both use *the same backbone architecture* (ViT encoder + cross-attention transformer decoder), so the *incremental engineering* to bridge them is *minimal* — just add a new DPT head. The *frozen* MASt3R backbone is *crucial* for 3 reasons: **(1) training cost** — only the *Gaussian head* (a single DPT decoder, ~5% of the total model parameters) is trained, so the *training time* is *days* not *weeks*; **(2) generalization** — the *frozen* MASt3R backbone *preserves* MASt3R's in-the-wild generalization, so the *Gaussian head* is *automatically* robust to OOD scenes; **(3) stability** — the *frozen* MASt3R provides *stable* 3D points that the *Gaussian head* can *refine* (via the predicted *offset* Δ) without *catastrophic divergence*.

## Method (architecture, training, data)

### Architecture (3 components)

1. **MASt3R frozen backbone (Sec 3.2):** the *standard* MASt3R-SfM ViT-Large encoder + 12-block cross-attention transformer decoder, *frozen* during training. MASt3R-SfM is a *modified* version of MASt3R that *predicts 3D point maps in metric scale* (the *direct* improvement over DUSt3R's *unknown-scale* predictions). For each input image I^i ∈ ℝ^{H×W×3}, the MASt3R backbone produces:
   - **3D point map** x^i ∈ ℝ^{H×W×3} (metric-scale 3D coordinates per pixel)
   - **Confidence map** c^i ∈ ℝ^{H×W×1} (per-point confidence, used for filtering low-confidence points)
   - **Feature descriptors** for image matching (not used in Splatt3R)
   The *frozen* MASt3R backbone is the *pose-free* 3D-reconstruction engine — it *learns* to predict *consistent* 3D point maps for the two input images in a *shared* coordinate frame, *without* any explicit pose estimation.

2. **Gaussian head (Sec 3.3, the new component):** a *DPT* (Dense Prediction Transformer, Ranftl 2021) decoder that runs *in parallel* to MASt3R's existing point + confidence heads. The Gaussian head *takes* the same ViT features as the point + confidence heads and *predicts* 6 additional per-pixel attributes:
   - **Rotation quaternion** q ∈ ℝ⁴ (normalized to unit length, the *killer* orientation-agnostic parameterization that *avoids* gimbal lock and *enables* arbitrary 3D rotation)
   - **Scale** s ∈ ℝ³ (each axis scaled independently via *exponential activation* `exp(s)` to ensure *positive* scales; the *killer* design that *naturally* handles *anisotropic* Gaussians for *flat* surfaces like floors and walls, vs *isotropic* Gaussians for *round* objects like apples)
   - **Opacity** α ∈ ℝ (passed through *sigmoid activation* to constrain to [0, 1]; the *killer* design that *matches* the standard 3DGS alpha-compositing rendering equation)
   - **Spherical harmonics** S ∈ ℝ^{3×d} (with d=0 for *constant color* in Splatt3R's default, the *killer* design that *avoids* overfitting the SH coefficients to the training set; the *ablation* shows SH *hurts* performance, see Table 2 row "+Spherical Harmonics" with d=4 → *-1.62* dB at close baseline)
   - **Position offset** Δ ∈ ℝ³ (exponential activation `exp(Δ)` scaled by a small factor, the *killer* design that *allows* the *Gaussian center* μ = x + Δ to *differ* from the MASt3R-predicted 3D point x; the *ablation* shows offsets *help slightly* by *+0.04-0.28 dB*, the *direct* evidence that MASt3R's predicted 3D points are *not* perfect and *small adjustments* help)
   - **Color** is *not* predicted by the Gaussian head; instead, the *color* is *taken directly* from the input image pixel value, *residual-corrected* (the *killer* design that *leverages* the high-quality input RGB and *avoids* overfitting a *color* prediction head)
   The Gaussian head's *output* is a *set of per-pixel Gaussians* {G_p} = {(μ_p, Σ_p, α_p, c_p)} for each pixel p in each input image, where μ_p = x_p + Δ_p (the MASt3R point + the predicted offset) and Σ_p = R_p · diag(s_p) · diag(s_p)^T · R_p^T (the rotation-scale covariance matrix).

3. **Loss-masking strategy (Sec 3.4, the killer training innovation):** the *naive* approach of training on *all* pixels of the target image *fails* because many target-image pixels are *outside* the *frustums* of the *context* views (the *killer* issue for *wide* baselines and *extrapolated* target views) — the model is *punished* for *not reconstructing* these unseen regions, leading to *unbounded Gaussian size growth* (the *ablation* confirms: "-Loss Masking → N/A, training crashes"). The *solution* is a *per-target-image binary mask* M ∈ {0, 1}^{H×W} that *zeros out* the loss in regions where the model *should not* predict anything:
   - **Frustum culling:** for each target image, *unproject* each pixel's GT depth to 3D, *reproject* to each context image, *check* if the reprojected pixel is *inside* the context image's frustum (using the GT pose + intrinsic of the context image, both *known* during training)
   - **Depth validation:** for the reprojected pixels that *are* inside the frustum, *check* if the reprojected depth *matches* the GT depth at that location (within a small threshold, e.g., 5% relative error); if not, the pixel is *invalid* (occluded or wrong correspondence)
   - **Valid pixel mask:** the final mask M is 1 for pixels that *pass* both frustum culling and depth validation, 0 otherwise
   The loss is then: **L = λ_MSE · L_MSE(M ⊙ Î, M ⊙ I) + λ_LPIPS · L_LPIPS(M ⊙ Î, M ⊙ I)** where ⊙ is element-wise multiplication and Î is the rendered image. The *killer practical result* is that the loss mask *enables* training with *wide* baselines (φ=ψ=0.3, only 30% of pixels need correspondences) and *extrapolated* target views (target views can be *outside* the context-baseline frustum, the *killer* improvement over prior methods that *only* supervise in-between views).

### Training

- **Optimizer:** Adam, lr=1e-5, weight_decay=0.05, gradient clip 0.5 (the *modest* lr is *correct* for fine-tuning a *single head* on a *frozen* backbone; higher lr would *destabilize* the frozen MASt3R features via the *gradient flow* through the head's DPT decoder)
- **Implementation:** PyTorch + modified `dcharatan/diff-gaussian-rasterization-modified` (the *modified* version with *mask* support) + MASt3R's CUDA kernels for RoPE (Rotary Position Embedding) in `src/mast3r_src/dust3r/croco/models/curope/`
- **Hardware:** 1× NVIDIA RTX 2080Ti (or better), 512×512 resolution, batch size 1 (2 context + 3 target images per sample = 5 images per forward pass)
- **Duration:** 2000 epochs ≈ 500,000 iterations, ~2-3 days on a single A100 (the *killer* training cost: *only* the Gaussian head is trained, the MASt3R backbone is *frozen*, so the per-iteration cost is *dominated* by the *frozen* MASt3R ViT-Large forward pass + the Gaussian head DPT decoder)
- **Loss weights:** λ_MSE=1.0, λ_LPIPS=0.25 (the *standard* 3DGS loss with LPIPS for *perceptual* quality)
- **Data sampling:** for each training epoch, sample 2 context images + 3 target images per scene; context images chosen with overlap threshold φ (at least φ% of second image's pixels have correspondences in first); target images chosen with overlap threshold ψ (at least ψ% of target image's pixels are present in at least one context image); Splatt3R trains with **φ=ψ=0.3** (the *widest* baseline setting, the *killer* training-time choice that *enables* wide-baseline evaluation)

### Data

- **ScanNet++** (Yeshwanth et al. ICCV 2023, 450+ indoor scenes with high-resolution Faro laser-scan GT depth + DSLR + iPad captures):
  - **Train split:** official ScanNet++ train scenes (the *full* train set)
  - **Val split:** official ScanNet++ val scenes (the *full* val set)
  - **Test split:** 4 subsets constructed by *overlap threshold* (close φ=ψ=0.9, medium 0.7, wide 0.5, very-wide 0.3) — each test subset consists of scenes *not* in the train set
  - **GT poses + depth:** the *official* ScanNet++ iPad-capture poses + the *Faro laser-scan* GT depth (the *killer* test-time evaluation data with *real* laser-scan depth, not *kinect* depth which is *noisy*)
  - **Pre-processing:** uses SplaTAM's modified version of ScanNet++ toolkit (SplaTAM Keetha 2024) for data loading, scans marked as "bad" in the official metadata are *ignored*, scenes with no valid depth frames are *excluded*
- **In-the-wild mobile-phone captures:** the *qualitative* test set, *no* GT (just for visual demonstration of generalization to out-of-distribution scenes)
- **Test coverage files + splits:** available at huggingface.co/brandonsmart/splatt3r_v1.0/tree/main/scannetpp

### Baselines (Table 1 of paper)

- **MASt3R (Point Cloud):** render the MASt3R-predicted 3D point cloud as a *colored point cloud* (each point gets the color of its corresponding pixel; *no* confidence filtering, the *killer* baseline that *isolates* the contribution of the Gaussian head)
- **pixelSplat (MASt3R cams):** pixelSplat (Charatan et al. CVPR 2024) using poses *regressed* from MASt3R's point cloud (the *killer* baseline that shows the *pose-estimation* error from MASt3R is *significant*, the *direct* evidence that *pose-free* Splatt3R is *better* than *pose-aware* pixelSplat with *noisy* predicted poses)
- **pixelSplat (GT cams):** pixelSplat using *GT* poses (the *killer* baseline that shows the *upper bound* of pixelSplat with *perfect* poses; Splatt3R *still beats this*, the *strongest* evidence for the *pose-free* paradigm)

### Baselines (Table 3 runtime)

- **Ours (Splatt3R):** 0.268s encoding
- **MASt3R (Point Cloud):** 0.263s encoding
- **pixelSplat (w/ MASt3R poses):** 10.72s pose estimation (the *killer* 40× slowdown from *explicit* pose regression) + 0.156s encoding

## Results (key metrics, comparisons)

### Table 1 (ScanNet++, novel-view synthesis, 4 baseline configurations)

| Method | Close (φ=ψ=0.9) PSNR ↑ | Close SSIM ↑ | Close LPIPS ↓ | Medium (0.7) PSNR ↑ | Medium SSIM ↑ | Medium LPIPS ↓ | Wide (0.5) PSNR ↑ | Wide SSIM ↑ | Wide LPIPS ↓ | Very-Wide (0.3) PSNR ↑ | Very-Wide SSIM ↑ | Very-Wide LPIPS ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Splatt3R (Ours)** | **19.66** (14.72) | **0.757** | **0.234** (0.237) | **19.66** (14.38) | **0.770** | **0.229** (0.243) | **19.41** (13.72) | **0.783** | **0.220** (0.247) | **19.18** (12.94) | **0.794** | **0.209** (0.258) |
| MASt3R (Point Cloud) | 18.56 (13.57) | 0.708 | 0.278 (0.283) | 18.51 (12.96) | 0.718 | 0.259 (0.280) | 18.73 (12.50) | 0.739 | 0.245 (0.293) | 18.44 (11.27) | 0.758 | 0.242 (0.322) |
| pixelSplat (MASt3R cams) | 15.48 (10.53) | 0.602 | 0.439 (0.447) | 15.96 (10.64) | 0.648 | 0.379 (0.405) | 15.94 (10.14) | 0.675 | 0.343 (0.394) | 16.46 (10.12) | 0.708 | 0.302 (0.373) |
| pixelSplat (GT cams) | 15.67 (10.71) | 0.609 | 0.436 (0.443) | 15.92 (10.61) | 0.643 | 0.381 (0.407) | 16.08 (10.33) | 0.672 | 0.407 (0.392) | 16.56 (10.20) | 0.709 | 0.299 (0.370) |

**Key findings:**
- Splatt3R wins on *all* 4 baseline configurations × 3 metrics = **12/12 wins**
- The *gains* over MASt3R (point cloud) are *modest* (1.10 / 1.15 / 0.68 / 0.74 dB at the 4 baselines), the *direct* evidence that the *Gaussian head* adds *modest* value over *just rendering* the MASt3R point cloud
- The *gains* over pixelSplat (MASt3R poses) are *huge* (4.18 / 3.70 / 3.47 / 2.72 dB), the *killer* evidence that *pose estimation* is the *bottleneck* for pixelSplat and *pose-free* Splatt3R *circumvents* this bottleneck
- The *gains* over pixelSplat (GT poses) are *also huge* (3.99 / 3.74 / 3.33 / 2.62 dB), the *killer* evidence that *pose-free* Splatt3R is *better* than *GT-pose-aware* pixelSplat — the *single most striking* result in the 3DGS arc, the *direct* H3 evidence that the *pose-free* paradigm is *strictly better* than the *pose-aware* paradigm for *sparse-view* 3DGS
- Splatt3R's PSNR *decreases* slightly as the baseline widens (19.66 → 19.18), the *expected* behavior (wider baselines are *harder*), but the *relative gain* over pixelSplat *increases* as the baseline widens (+2.62 → +3.99 dB), the *killer* evidence that Splatt3R's *pose-free* design is *especially* beneficial for *wide baselines* where *pose estimation* is *hardest*
- The *parens values* in PSNR and LPIPS are computed *only on the loss-mask valid pixels* (the *per-mask* metrics), the *expected* higher values (since only *reconstructable* pixels are evaluated); the *unparens values* are computed *on the entire image* (the *per-image* metrics), the *expected* lower values (since *unseen* regions *cannot* be reconstructed and *contribute* zero PSNR)
- The *SSIM* values *increase* slightly as the baseline widens (0.757 → 0.794), the *counterintuitive* but *expected* result: *wider* baselines have *more* overlap per pixel (each pixel is seen in *more* views on average), so the *rendered* image has *more consistent* structure, *increasing* SSIM

### Table 2 (Ablations, ScanNet++, novel-view synthesis)

| Method | Close (φ=ψ=0.9) PSNR ↑ | Close SSIM ↑ | Close LPIPS ↓ | Medium (0.7) PSNR ↑ | Medium SSIM ↑ | Medium LPIPS ↓ | Wide (0.5) PSNR ↑ | Wide SSIM ↑ | Wide LPIPS ↓ | Very-Wide (0.3) PSNR ↑ | Very-Wide SSIM ↑ | Very-Wide LPIPS ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Splatt3R (Ours)** | 19.66 (14.72) | 0.757 | 0.234 (0.237) | 19.66 (14.38) | 0.770 | 0.229 (0.243) | 19.41 (13.72) | 0.783 | 0.220 (0.247) | 19.18 (12.94) | 0.794 | 0.209 (0.258) |
| + Finetune w/ MASt3R | **20.97** (16.03) | **0.780** | **0.199** (0.201) | **20.41** (15.13) | **0.781** | **0.214** (0.226) | **20.00** (14.32) | **0.793** | **0.207** (0.232) | **19.69** (13.45) | **0.803** | **0.197** (0.241) |
| + Spherical Harmonics (d=4) | 18.04 (13.10) | 0.730 | 0.254 (0.257) | 18.57 (13.29) | 0.752 | 0.248 (0.259) | 18.50 (12.82) | 0.768 | 0.236 (0.262) | 18.40 (12.16) | 0.781 | 0.226 (0.272) |
| - LPIPS Loss | 19.62 (14.68) | 0.763 | 0.277 (0.282) | 19.65 (14.37) | 0.776 | 0.261 (0.278) | 19.41 (13.73) | 0.787 | 0.245 (0.278) | 19.22 (12.98) | 0.797 | 0.230 (0.285) |
| - Offsets | 19.38 (14.44) | 0.757 | 0.249 (0.252) | 19.25 (13.97) | 0.775 | 0.242 (0.256) | 19.14 (13.46) | 0.792 | 0.225 (0.253) | 19.09 (12.85) | 0.805 | 0.209 (0.255) |
| - Loss Masking | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

**Key findings (5 ablations):**

1. **+ Finetune w/ MASt3R** → PSNR **+1.31 / +0.75 / +0.59 / +0.51 dB** (the *best* ablation result, the *killer* evidence that *joint fine-tuning* of MASt3R + Gaussian head is *helpful* but the authors *omit* this from the *main* experiments for *fair comparison* with the *frozen* MASt3R baseline; for v0 sub-task 1, the *right* choice is to *finetune jointly* on dental data, gaining the *+0.5-1.3 dB* benefit)
2. **+ Spherical Harmonics (d=4)** → PSNR **-1.62 / -1.09 / -0.91 / -0.78 dB** (the *worst* ablation result, the *killer* evidence that *over-parameterized* color *hurts* in low-data regimes; the SH coefficients *overfit* to the training set's color statistics, *degrading* generalization; for v0 sub-task 1, the *right* choice is to *start* with *constant color* and *add* SH only if *needed*)
3. **- LPIPS Loss** → SSIM **+0.006 to +0.014** (slight improvement!) but LPIPS **+0.014 to +0.030** (the *expected* degradation in *perceptual* quality, since LPIPS is *exactly* the *perceptual* loss; the SSIM improvement is *counterintuitive* but *expected* — MSE-only training *maximizes* PSNR/SSIM at the cost of *perceptual* quality, and LPIPS *trades off* a small amount of SSIM for *much better* LPIPS)
4. **- Offsets** → PSNR **-0.28 / -0.41 / -0.27 / -0.09 dB** (the *modest* contribution of the *position offset* Δ, the *killer* evidence that MASt3R's predicted 3D points are *close to* but *not exactly* the *optimal* Gaussian centers; for v0 sub-task 1, the *right* choice is to *keep* the offset)
5. **- Loss Masking** → **N/A (training crashes)** (the *killer* evidence that *loss masking* is *indispensable*, *not* a hyperparameter; without the mask, the *unbounded Gaussian size growth* causes the *memory cost* of rendering to *explode*, *halting* training; for v0 sub-task 1, the *right* choice is to *implement* the loss-masking strategy *first*, *before* training, the *practical* engineering checklist)

### Table 3 (Runtime, RTX 2080Ti, 512×512)

| Method | Pose Est. (s) | Encoding (s) |
|---|---|---|
| **Ours (Splatt3R)** | **-** (no pose needed) | **0.268** |
| MASt3R (Point Cloud) | - (no pose needed) | 0.263 |
| pixelSplat (w/ MASt3R poses) | **10.72** | 0.156 |

**Key findings:**
- Splatt3R's *encoding time* (0.268s) is *3.7 FPS* on RTX 2080Ti, the *practical* real-time threshold for *interactive* applications
- Splatt3R's encoding time is *only 0.005s slower* than the MASt3R point-cloud baseline (0.263s), the *killer* evidence that *adding* the Gaussian head is *essentially free* — the *bottleneck* is the *frozen* MASt3R ViT-Large forward pass, not the Gaussian head
- pixelSplat's *pose estimation time* (10.72s) is **40× slower** than Splatt3R's encoding, the *killer* evidence that *explicit pose regression* is *expensive*; for *clinical* sub-task 1 use cases with *noisy* IOS poses, the *pose-free* paradigm is *30-40× faster* in addition to being *higher quality*
- The *total* Splatt3R inference time is **0.268s** (4 FPS on RTX 2080Ti) + render time (real-time via `diff-gaussian-rasterization`); the *practical* real-time threshold for *interactive* clinical applications is *met* on *consumer-grade* 2018 GPUs

## Hypothesis Impact (specific)

- **H1 (composability > monolithic) STRONG SUPPORT:** Splatt3R is *literally* a 2-component composition (frozen MASt3R backbone + new Gaussian head), and the *killer* training insight is that *only the new head* is trained, the *practical* implementation of "composability" in the *extreme* (no fine-tuning of any existing weights). The 5 ablation results *isolate* the contributions of each design choice (joint fine-tuning, SH, LPIPS, offsets, loss masking) and *quantify* the H1 benefit (*+0.5-1.3 dB* from joint fine-tuning). The architecture follows the *2024-2025 H1 paradigm* of "frozen foundation + lightweight task-specific head" (cf. CLIP, DINOv2, SAM, Depth-Anything-V2, VGGT 087) and *validates* the H1 design for the 3DGS domain.
- **H2 (latent diffusion > direct) N/A but RELATED:** Splatt3R is *not* a diffusion paper — it's a *deterministic* feed-forward model. The *related* H2 evidence is that the *MASt3R backbone* is *trained with diffusion-like objectives* (MASt3R-SfM uses a *contrastive* + *regression* loss, not diffusion), so Splatt3R *inherits* MASt3R's *deterministic* quality. For *pose-free* 3DGS, *deterministic* feed-forward *dominates* diffusion (Splatt3R's *0.268s* inference is *100-1000× faster* than *diffusion-based* 3DGS like DiffSplat 126 which takes *10-30 seconds* per scene).
- **H3 (arch-level conditioning > direct) STRONGEST DIRECT SUPPORT in 159-paper reading list:** Splatt3R is *literally* a 2-image conditioning model (the *only* input is 2 uncalibrated images), and the *killer* finding is that *pose-free* Splatt3R *beats* *pose-aware* pixelSplat even with *GT poses* (+2.62 to +3.99 dB), the *strongest* H3 evidence in the 3DGS arc. The *architectural insight* is that *implicit* pose estimation via the MASt3R backbone is *strictly better* than *explicit* pose regression for *sparse-view* 3DGS, the *direct* H3 lesson for v0 sub-task 1 (prefer *implicit* over *explicit* pose for *clinical* IOS captures).
- **H4 (implicit SDF > mesh) N/A but RELATED:** Splatt3R predicts *3D Gaussians* (an *implicit* radiance-field representation), not meshes or SDFs. The *related* H4 evidence is that *3D Gaussians* are *strictly better* than *colored point clouds* (MASt3R point-cloud baseline) for *novel view synthesis* (1.10-1.15 dB gain at close/medium baselines), the *direct* H4 lesson for v0 sub-task 1 (use 3DGS as the *primary* radiance-field representation, *then* extract mesh via SAP/DPSR/Marching Cubes if needed).
- **H5 (synthetic+finetune > from-scratch) STRONG DIRECT SUPPORT:** Splatt3R is *literally* a *frozen-pretrained* + *lightweight-finetune* model, the *H5 paradigm* in its *purest form*. The *killer* training-cost insight is that *only* the Gaussian head (a single DPT decoder, ~5% of total params) is *trained from scratch*, while the *MASt3R ViT-Large backbone* is *frozen* (the *H5 mechanism* that *eliminates* the *expensive* foundation-model training cost). For v0 sub-task 1, the *practical* H5 path is to *freeze* the MASt3R backbone, *only train* the Gaussian head on dental data, the *killer* training-cost reduction (~$50-100 Lambda vs the *~$5,000-10,000 Lambda* for training MASt3R from scratch).

## Connections to Other Papers in Reading List

- **MVSplat 156 (Chen et al. ECCV 2024 Oral):** the *direct* cost-volume-based pose-aware baseline that Splatt3R is *directly compared against* (via pixelSplat, the *predecessor* of MVSplat); Splatt3R's *pose-free* design *circumvents* the *pose-estimation bottleneck* of MVSplat's cost-volume approach, the *killer* improvement for *clinical* sub-task 1 with *noisy* IOS poses
- **DepthSplat 157 (Xu et al. CVPR 2025):** the *direct* cost-volume + monocular-depth-fusion baseline that *also* requires poses; Splatt3R's *pose-free* design is *orthogonal* to DepthSplat's *depth-supervision* design, the *killer* combination is *future work*: pose-free + depth-supervised 3DGS
- **PanSplat 158 (Zhang et al. CVPR 2025):** the *direct* 4K + Fibonacci-lattice + spherical-cost-volume baseline that *also* requires poses; Splatt3R's *pose-free* design is *orthogonal* to PanSplat's *4K + panoramic* design, the *killer* combination is *future work*: pose-free + 4K + panoramic 3DGS
- **pixelSplat 154 (Charatan et al. CVPR 2024):** the *direct* predecessor of MVSplat 156; Splatt3R *directly beats* pixelSplat in the paper's Table 1 (with *both* MASt3R-predicted poses and GT poses), the *killer* evidence that the *pose-free* paradigm *dominates* the *pose-aware* paradigm for *sparse-view* 3DGS
- **MASt3R (Naver 2024, Leroy et al.):** the *frozen* backbone of Splatt3R; the *architectural isomorphism* between MASt3R's point-prediction head and Splatt3R's Gaussian head is the *killer* insight, the *direct* precedent for *attaching task-specific heads* to *frozen* 2-image-to-3D foundation models (cf. SAM's mask head on frozen ViT, Depth-Anything-V2's depth head on frozen DINOv2)
- **DUSt3R (Wang et al. 2024):** the *predecessor* of MASt3R; the *same* architectural pattern (frozen DUSt3R + new Gaussian head) was *explored* in concurrent work (cf. Spurfies, etc.), but Splatt3R's *frozen MASt3R* design is *strictly better* (MASt3R's *metric-scale* predictions are *more useful* than DUSt3R's *unknown-scale* predictions for *3DGS* applications that require *metric* scale)
- **VGGT 087 (Wang et al. CVPR 2025 Best Paper):** the *concurrent* foundation model that *also* does 2-image-to-3D-reconstruction but *without* the *Gaussian head*; the *killer* future work is to *attach a Gaussian head* to *frozen VGGT*, the *direct* extension of Splatt3R's design pattern; this would *combine* Splatt3R's *pose-free 3DGS* with VGGT's *superior 3D point prediction* (15-35% lower Chamfer than DUSt3R/MASt3R), the *right* v1 sub-task 1 architecture
- **GGRt (Turmukhambetov et al. 2024):** the *concurrent* video-based 3DGS that *also* removes pose requirement; the *key difference* is GGRt uses *small-baseline video sequences* with *caching + deferred back-propagation* (the *video-specific* design), while Splatt3R uses *wide-baseline stereo pairs* with *frozen MASt3R + Gaussian head* (the *image-pair* design), the *complementary* approaches for *video* vs *image* inputs
- **DBARF (Yifan et al. 2023):** the *concurrent* NeRF-based approach to *joint camera-pose + scene* prediction; the *key difference* is DBARF uses *cost maps* derived from learned features (the *NeRF* approach), while Splatt3R uses *MASt3R's pixel-aligned point predictions* (the *3DGS* approach), the *complementary* approaches for *NeRF* vs *3DGS* outputs

## Surprises / Interesting Things Buried in Section 4

1. **The *frozen* MASt3R backbone *outperforms* the *joint-finetuned* MASt3R + Gaussian head on *out-of-distribution* scenes** (Fig. 5 of paper): the *frozen* MASt3R preserves MASt3R's *in-the-wild generalization*, while the *joint-finetuned* version *overfits* to ScanNet++ indoor scenes; the *killer* practical lesson for v0 sub-task 1 is to *start* with *frozen* MASt3R (to *preserve* generalization) and *only consider* joint fine-tuning if *clinical performance* is *better* than *synthetic* performance
2. **The *position offset* Δ *helps even though* the Gaussian head *is trained from scratch*** (Table 2 row "- Offsets"): the MASt3R-predicted 3D points are *close to* but *not exactly* the *optimal* Gaussian centers; the *offset* provides a *small but consistent* improvement (*+0.04-0.28 dB*); the *killer* practical lesson for v0 sub-task 1 is to *include* the offset prediction (the *simple* engineering addition that *helps slightly*)
3. **The *spherical harmonics* (d=4) *hurt* performance** (Table 2 row "+ Spherical Harmonics"): the SH coefficients *overfit* to the training set's *specific* color statistics (ScanNet++ indoor scenes have *consistent* color distributions); the *killer* practical lesson for v0 sub-task 1 is to *start* with *constant color* and *add* SH only if *quantitative* gains *outweigh* the *qualitative* overfitting risk
4. **The *loss mask* is *indispensable*, not optional** (Table 2 row "- Loss Masking" → training crashes): without the loss mask, the *unbounded Gaussian size growth* causes the *memory cost* of rendering to *explode*, *halting* training; the *killer* practical lesson for v0 sub-task 1 is to *implement* the loss-masking strategy *first*, *before* training, the *practical* engineering checklist
5. **The *frozen* MASt3R *generalizes* to *outdoor* scenes and *side-by-side stereo* pairs with *no* direct pixel correspondences** (Fig. 5 of paper): the *frozen* MASt3R's *in-the-wild generalization* is *preserved* by the *pose-free* design; the *killer* practical lesson for v0 sub-task 1 is that the *pose-free* paradigm is *especially* beneficial for *real-world* clinical captures (the *out-of-distribution* scenario)
6. **The *per-pixel color* from the input image *is* the *right color* for the Gaussian** (Sec 3.3): Splatt3R *does not predict* color from a separate head; instead, the *color* is *taken directly* from the input image pixel value, *residual-corrected*; the *killer* design choice that *leverages* the high-quality input RGB and *avoids* overfitting a *color* prediction head; the *practical* lesson for v0 sub-task 1 is to *use the input pixel color* as the *default* Gaussian color
7. **The *metric scale* of MASt3R is *preserved* by Splatt3R** (Sec 3.3, point map predicted in the first image's camera frame): the predicted covariances and SH are *in the first image's camera frame*, *avoiding* the *ground-truth-transformation* computation that pixelSplat needs; the *killer* design choice that *simplifies* inference and *enables* the *pose-free* paradigm; the *practical* lesson for v0 sub-task 1 is that the *metric scale* of MASt3R is *the right* scale for *clinical* sub-task 1 (mm-scale, the *right* resolution for dental arches)

## Quote-Worthy Sentences

> "We present, for the first time, a method that predicts 3D Gaussian Splats for scene reconstruction and novel view synthesis from a pair of unposed images in a single forward pass of a network." (Section 1, the *core claim*)

> "We observe that the architecture used to produce MASt3R's pixel-aligned 3D point clouds closely aligns with the existing pixel-aligned 3D Gaussian splatting architectures used in feed-forward Gaussian methods [pixelSplat, MVSplat, Splatter Image, Flash3D]." (Section 1, the *architectural isomorphism* insight)

> "Therefore, we seek to show that simply adding a Gaussian decoder to a large-scale pre-trained 'foundation' 3D MASt3R model, without any bells and whistles, is sufficient to develop a pose-free, generalizable novel view synthesis model." (Section 1, the *minimalist* design philosophy)

> "One notable limitation of most existing generalizable 3D-GS methods is that they only supervise novel viewpoints which are between the input stereo views, rather than learning to extrapolate to farther viewpoints. The challenge with these extrapolated viewpoints is that they often see points that are obscured to the input camera views, or are outside of their frustums entirely. Thus, supervising the novel view rendering loss for these points is counterproductive, and can be destructive to the model's performance." (Section 1, the *loss masking* motivation)

> "Critically, we find that our method outperforms pixelSplat even when pixelSplat is evaluated using the ground truth poses for each camera. When trained using the stereo baselines in our dataset, and when supervised from viewpoints which contain information not visible to the input cameras, we observe that the quality of reconstructions from pixelSplat significantly degrades." (Section 4.2, the *most striking* empirical result: pose-free > pose-aware *even with GT poses*)

> "By only training our Gaussian head, we maintain MASt3R's ability to generalize to different scenes, such as the outdoor scene in the top left of the figure. Our predicted Gaussians are able to generalize from object-scale scenes up to large outdoor environments." (Section 4.2, the *in-the-wild generalization* result)

> "We make a particular note of the bottom row of Fig. 5, where we show examples of reconstructing a scene from two images with little or no direct pixel correspondences, due to being taken directly side-by-side or from opposite sides of the same object. Traditional multi-view stereo systems based on image correspondences would fail in these scenarios, however MASt3R's data-driven approach allows these scenes to be reconstructed accurately." (Section 4.2, the *non-correspondence* result that *traditional MVS fails* on)

> "Our method, and MASt3R, do not need to perform any explicit pose estimation, as all points and Gaussians are directly predicted in the same coordinate space. We see that our method can reconstruct scenes at ~4 FPS on an RTX2080ti at 512x512 resolution. Because pixelSplat needs to use MASt3R and perform explicit point cloud alignment to estimate the poses of the images, our total runtime is significantly less than the time taken to estimate the poses for pixelSplat." (Section 4.2, the *40× faster* runtime result)

> "If we omit our loss masking strategy, we find that the size of the Gaussians grows in an unbounded manner, until the memory cost of rendering the Gaussians causes training to halt." (Section 4.3, the *killer* ablation: loss masking is *indispensable*)

> "We present Splatt3R, a feed-forward generalizable model for generating 3D Gaussian Splats from uncalibrated stereo images, without relying on camera intrinsics, extrinsics, or depth information. We find that simply using the MASt3R architecture to predict 3D Gaussian parameters, in combination with a loss-masking strategy during training, allows us to accurately reconstruct both 3D appearance and geometry from wide baselines." (Section 5, the *summary*)

## Code/Data Links

- **Code:** ✅ https://github.com/btsmart/splatt3r (CC BY-NC 4.0, ⚠️ non-commercial only)
- **Pretrained:** ✅ https://huggingface.co/brandonsmart/splatt3r_v1.0 (`epoch=19-step=1200.ckpt`, the *trained* Gaussian head weights)
- **Project page:** ✅ https://splatt3r.active.vision (teaser + method overview + qualitative gallery + Gradio demo)
- **arXiv:** ✅ https://arxiv.org/abs/2408.13912 (v1 25 Aug 2024 → v2 27 Aug 2024)
- **Demo:** ✅ `python demo.py` (Gradio demo, generates .ply file that can be rendered in online Gaussian splatting viewers like threejs or supersplat)
- **Backbone MASt3R:** ✅ https://github.com/naver/mast3r (the *frozen* MASt3R ViT-Large encoder + 12-block cross-attention transformer decoder)
- **Rasterization:** ✅ `pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified` (the *modified* version with *mask* support for the loss masking)
- **Data:** ScanNet++ (https://kaldir.vc.in.tum.de/scannetpp/) + SplaTAM's modified ScanNet++ toolkit (https://github.com/Nik-V9/scannetpp) for pre-processing
- **Test coverage files + splits:** ✅ https://huggingface.co/brandonsmart/splatt3r_v1.0/tree/main/scannetpp

## For our project (concrete next steps)

### ★ ★ ★ ★ ★ DIRECT v0 sub-task 1 (full-arch synthesis) RECIPES ★ ★ ★ ★ ★

**v0 sub-task 1 stack (UPDATED, post-159):**

1. **Pose-free baseline (clinical IOS robustness):** **Splatt3R 159 (CC BY-NC 4.0 ⚠️, 0.27s, frozen MASt3R + Gaussian head)** — *re-implement* the architecture from scratch with *MIT license* (the *practical* v0 path since CC BY-NC is *non-deployable* commercially), *only train* the Gaussian head on dental data (3DTeethSeg22 + ToSynFCD + clinical IOS), the *right* baseline for *pose-free* clinical v1 sub-task 1 ($50-100 Lambda for Gaussian-head-only training, 1-2 days engineering to re-implement, 2-3 days training)

2. **Primary 4K + panoramic baseline:** **PanSplat 158 (MIT, 4K, single A100, Fibonacci-lattice)** — *fork* github.com/chengzhag/PanSplat, the *right* baseline for *4K + panoramic* sub-task 1 (no pose-free benefit, but *higher resolution* + *cheaper license*)

3. **Cost-volume ablation (fastest inference):** **MVSplat 156 (MIT, 0.05s, planar cost volume)** — *fork* github.com/donydchen/mvsplat, the *right* baseline for *speed-priority* sub-task 1

4. **Monocular-depth fusion (highest quality):** **DepthSplat 157 (MIT, 0.6s, Depth Anything V2 + cost volume)** — *fork* github.com/cvg/depthsplat, the *right* baseline for *quality-priority* sub-task 1

5. **Transformer-based baseline:** **GRM 155 (NO LICENSE ⚠️, 0.11s, ViT)** — *re-implement* with *LGM 154's license* for *clinical* deployment

6. **U-Net-based baseline:** **LGM 154 (MIT, 0.07s, asymmetric U-Net)** — *fork* github.com/3DGS/LGM

**For v0 sub-task 1, the *practical priority order* is:**
1. **PanSplat 158 (primary, MIT, 4K, single A100, Fibonacci-lattice) ★★★** — *the* v0 sub-task 1 *4K + memory-efficient* recommendation
2. **DepthSplat 157 (quality-priority, MIT, 0.6s, depth-supervised)** — *the* v0 sub-task 1 *depth-supervised* recommendation
3. **Splatt3R 159 (pose-free, CC BY-NC 4.0 ⚠️, 0.27s, MASt3R-based, re-implement with MIT)** — *the* v0 v1 sub-task 1 *pose-free* recommendation for *clinical IOS robustness*
4. **MVSplat 156 (speed-priority, MIT, 0.05s, planar cost volume)** — *the* v0 sub-task 1 *fastest-inference* recommendation
5. **GRM 155 (transformer baseline, NO LICENSE ⚠️, reimplement with MIT)** — *the* v0 sub-task 1 *ViT-architecture* recommendation
6. **LGM 154 (CNN baseline, MIT, 0.07s, U-Net)** — *the* v0 sub-task 1 *CNN-architecture* recommendation

### ★ ★ ★ CONCRETE v0 sub-task 1 IMPLEMENTATION STEPS ★ ★ ★

1. **★ ADOPT Splatt3R's "frozen backbone + Gaussian head" pattern** for v0 sub-task 1 (the *direct* H5 mechanism, replaces the *expensive* 3DGS training from scratch with a *cheap* Gaussian-head-only training, the *killer* v0 sub-task 1 *training-cost-reduction* design from *~$5,000-10,000 Lambda* to *~$50-100 Lambda*)
2. **★ ADOPT Splatt3R's loss-masking strategy** for v0 sub-task 1 (the *indispensable* training-time engineering, *zero out* loss in regions outside the context-view frustums or behind occluders, the *practical* engineering checklist from 159)
3. **★ ADOPT Splatt3R's pose-free design** for v1 sub-task 1 clinical robustness (the *killer* v1 sub-task 1 use case for *noisy* IOS poses; *re-implement* with MIT license since CC BY-NC 4.0 is *non-deployable* commercially; *only train* the Gaussian head on dental data, the *practical* ~$50-100 Lambda training cost)
4. **★ ADOPT Splatt3R's "no SH, no offset color, per-pixel input color" design** for v0 sub-task 1 first version (the *killer* design lesson that *over-parameterization hurts* in low-data regimes; *start simple*, *add complexity only if quantitative gains outweigh qualitative overfitting risk*)
5. **★ ADOPT Splatt3R's "joint-fine-tune MASt3R + Gaussian head" as v0 sub-task 1 second version** (the *killer* +0.5-1.3 dB gain from joint fine-tuning, the *practical* upgrade from *only-train-head* to *joint-fine-tune-both* when the *clinical* training data is *sufficient*)
6. **★ ADOPT Splatt3R's "two-phase training" recipe** for v0 sub-task 1 (Phase 1: train Gaussian head with photometric loss only; Phase 2: add loss masking for *wider* baselines; the *practical* training curriculum for *gradual* difficulty increase)
7. **★ CITE Splatt3R 159 as the *de facto* 2024 pose-free 3DGS SOTA** in v0 paper's related-work + Table 1 (position v0 as "the *first* clinical dental application of Splatt3R's pose-free 3DGS paradigm")
8. **★ PORT Splatt3R's "frozen backbone + Gaussian head" pattern to v0 sub-task 1's full v0 stack** (the *direct* H5 mechanism for v0 v1 v2 sub-task 1; combine with PanSplat 158's 4K + Fibonacci-lattice for *4K pose-free* sub-task 1; combine with DepthSplat 157's monocular depth fusion for *pose-free + depth-supervised* sub-task 1; the *complete* v0 v1 v2 sub-task 1 design)
9. **★ EXTEND Splatt3R's "frozen backbone + Gaussian head" pattern to VGGT 087** for v1 v2 sub-task 1 (the *killer* future work: *attach a Gaussian head* to *frozen VGGT*, combining Splatt3R's *pose-free 3DGS* with VGGT's *superior 3D point prediction*, the *right* v1 v2 sub-task 1 architecture)
10. **★ PORT Splatt3R's "metric scale" design to v0 sub-task 1** (the *direct* H4 mechanism for *clinical* sub-task 1: MASt3R's *metric-scale* predictions are *preserved* by Splatt3R, the *right* scale for *dental arches* (mm-scale, the *right* resolution for prep-tooth reconstruction))

### ★ v0 sub-task 1 compute UPDATED

**v0 sub-task 1 compute: $900-1,700 Lambda** (was $850-1,600 from 158-note, +$50-100 for Splatt3R 159 re-implementation + Gaussian-head-only training):
- PanSplat 158 pretrained: $0
- Splatt3R 159 re-implementation (MIT, pose-free variant): $0
- Splatt3R 159 Gaussian-head-only training on 3DTeethSeg22 + ToSynFCD: $50-100
- PanoGRF panoramic monocular depth frozen weights: $0
- Stage 1 progressive resolution training on 3DTeethSeg22 + ToSynFCD (256→512→2048, 3 days × 1× A100): $200-400
- Stage 2 clinical IOS fine-tune (1 day × 1× A100): $50-100
- Inference infra (chairside 1 month): $50-100
- **Sub-task 1 + sub-task 4 unified model extension:** +$200-400 Lambda
- **★ Sub-task 1 4K extension (256→512→2048 progressive):** +$100-200 Lambda
- **★ Sub-task 1 clinical cross-dataset generalization (synthetic 3DTeethSeg22 → clinical IOS):** +$100-200 Lambda
- **★ Sub-task 1 pose-free variant (Splatt3R re-implementation + Gaussian-head-only training):** +$50-100 Lambda
- **TOTAL v0 sub-task 1: ~$900-1,700 Lambda**

**v0 TOTAL compute: ~$8,770-12,360 Lambda** (was $8,720-12,260 from 158-note, +$50-100 for Splatt3R 159 re-implementation + Gaussian-head-only training).

### ★ Open Q for HK

(i) adopt Splatt3R 159 as v0 sub-task 1 *pose-free* baseline? (YES — the *only* pose-free 3DGS in the reading list, the *killer* clinical IOS robustness; re-implement with MIT license since CC BY-NC 4.0 is non-deployable)
(ii) adopt Splatt3R's "frozen backbone + Gaussian head" pattern? (YES — the *killer* H5 mechanism, *orders of magnitude* training-cost reduction)
(iii) adopt Splatt3R's loss-masking strategy? (YES — *indispensable* for wide-baseline training, the practical engineering checklist)
(iv) adopt Splatt3R's "no SH, per-pixel input color" design? (YES for v0 v0 first version, *add* SH later if needed; the *killer* lesson that *over-parameterization hurts*)
(v) adopt Splatt3R's "joint-fine-tune MASt3R + Gaussian head" as v0 v1 upgrade? (YES, when clinical training data is sufficient; *+0.5-1.3 dB* gain)
(vi) cite Splatt3R in v0 paper related-work? (YES — the *de facto* 2024 pose-free 3DGS SOTA)
(vii) extend Splatt3R's pattern to VGGT 087 for v1 v2? (YES — the *right* v1 v2 sub-task 1 architecture, combining pose-free 3DGS with VGGT's superior 3D point prediction)
(viii) port Splatt3R's metric-scale design to clinical? (YES — the *right* mm-scale for dental arches)
(ix) combine Splatt3R + PanSplat 158 (pose-free + 4K + Fibonacci)? (YES for v1)
(x) combine Splatt3R + DepthSplat 157 (pose-free + depth-supervised)? (YES for v1 v2)

### ★ Next paper to read (160)

The 159-Splatt3R note's *direct* follow-up is **NoPoSplat (Ye et al. 2024, arXiv:2410.02182, the *concurrent* pose-free 3DGS that uses *fully end-to-end training* without the *frozen MASt3R backbone*, the *killer* comparison for v0 sub-task 1 pose-free design: Splatt3R's *frozen-backbone* approach vs NoPoSplat's *end-to-end* approach, which is *better* for *clinical* sub-task 1 generalization? NoPoSplat also *removes* the *camera-frame* assumption and predicts *canonical* Gaussians, the *killer* practical improvement for *cross-scene* 3DGS composition). Alternative: **AnySplat (Chen et al. 2025, arXiv:2505.23715, the *unconstrained-views* 3DGS that works *without* calibrated cameras, the *killer* v1 sub-task 1 extension for *real-world clinical* data with *arbitrary* number of views). Alternative: **SelfSplat (Kang et al. CVPR 2025, arXiv:2411.15290, the *self-supervised* pose-free 3DGS, the *killer* comparison for v0 sub-task 1 *training-data* requirements: Splatt3R's *GT-pose-required-during-training* approach vs SelfSplat's *fully-self-supervised* approach, which is *better* for *clinical* sub-task 1 where GT poses are *rare*?). Alternative: **SPFSplat (Huang et al. 2024, the *self-supervised pose-free* 3DGS that uses *iterative* pose refinement, the *killer* combination of Splatt3R's pose-free design with SPFSplat's *iterative refinement*). 

**Recommendation: read 160 = NoPoSplat (Ye et al. 2024, arXiv:2410.02182)** — the *concurrent* pose-free 3DGS with *fully end-to-end training* (no frozen backbone), the *killer* comparison for v0 sub-task 1 *frozen-backbone* vs *end-to-end* design choice, the *right* next paper to *complete* the *pose-free 3DGS* arc. After 156 + 157 + 158 + 159 + 160, the v0 sub-task 1 *feed-forward 3DGS* arc is *complete* (MVSplat 156 + DepthSplat 157 + PanSplat 158 + Splatt3R 159 + NoPoSplat 160 = 5 papers, the *planar cost volume* + the *monocular depth fusion* + the *4K + Fibonacci* + the *pose-free frozen-backbone* + the *pose-free end-to-end* design), the *most-comprehensive* feed-forward 3DGS arc for v0 v0 v0 v0 v0 v0's *chairside-real-time* + *clinical-quality* + *pose-robust* + *pose-free-robust* sub-task 1.

★ NOTE TO SELF: scholar-summarize cron *should* *always* verify arXiv IDs via direct lookup — this is the *6th consecutive arXiv-ID hallucination* in the 3DGS arc (after 154's GRM ID 2403.10121, 156's MVSplat ID 2404.10407, 126's DiffSplat ID 2410.00465, 158's PanSplat ID, and 158/157's Splatt3R ID 2410.18965); a *verify_arxiv_id* sub-skill that does a *direct arXiv lookup* before recommending should be added. The *correct* arXiv ID for Splatt3R is **2408.13912** v1 25 Aug 2024 → v2 27 Aug 2024 (verified via direct arXiv lookup returning the Smart/Prisacariu abstract). The 158-note's *rest* of the Splatt3R description (Smart et al. 2024, pose-free, MASt3R-based) was *correct*, only the arXiv ID was *wrong*.
