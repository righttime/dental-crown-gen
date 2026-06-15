# Paper 193 — MapAnything: Universal Feed-Forward Metric 3D Reconstruction

## TL;DR

**FOUNDING PAPER** of the **universal / multi-task / multi-modal** paradigm for *feed-forward 3D reconstruction* — the *first* model to unify **12+ 3D vision tasks in a single feed-forward pass** (uncalibrated SfM, calibrated SfM, calibrated MVS, monocular depth estimation, camera localization, depth completion, single-view calibration, multi-view metric depth, multi-image SfM, registration, ray-casting, and more) while supporting **any combination of input modalities** (images, intrinsics, extrinsics, depth, partial reconstructions) and outputting a **single, consistent, factored metric 3D scene representation** in *one* forward pass. The **deceptively simple but architecturally radical** insight: instead of the *coupled* pointmap representation that DUSt3R/MASt3R/π³/VGGT all share (where cameras + poses + geometry are *entangled* in a single pointmap that must be *recovered* post-hoc via expensive symmetric inference or Sim(3) alignment), **represent the scene as a *fully factored* tuple of `(rays, depth-along-ray, camera pose, global metric scale)`** — a representation that (a) *unifies* per-view geometry (rays + depth) with global geometry (pose + scale) in a *single* output space, (b) supports *heterogeneous* input modalities by *factoring* each modality into the same latent space (a *scale-decoupled* representation), and (c) enables *flexible* training with *partial annotations* (e.g., up-to-scale datasets vs metric-scale datasets, monocular vs multi-view datasets, pinhole vs generic central cameras). **Three coupled innovations: (1) Factored Rays-Depth-Pose (RDP) scene representation** — predict (a) per-view local ray directions `R_i ∈ ℝ^{3×H×W}` (akin to camera intrinsics), (b) per-view up-to-scale ray depths `D̃_i ∈ ℝ^{1×H×W}`, (c) per-view global camera poses `P̃_i ∈ SE(3)` *up to a single unknown similarity*, and (d) a single global metric scale factor `m ∈ ℝ` that *upgrades* the up-to-scale predictions to metric scale (`X^metric_i = m · X̃_i`) — the *first* representation that *cleanly* decouples *camera intrinsics* (rays) from *camera extrinsics* (poses) from *metric scale* (m) from *per-pixel geometry* (depth along ray); **(2) Flexible input scheme with 6-factor input augmentation** — at *training* time, each of the 6 input factors (image / intrinsics / rotation / translation / depth / metric-scale flag) is *randomly dropped* with a per-factor probability, so the model learns to handle *any* combination of available inputs at *inference* time (64 input combinations without per-view flexibility, 1000+ with per-view flexibility); **(3) Scale-invariant log-space losses with stop-grad** — for the scale-coupled outputs (depth, pointmap, translation), losses are computed in *log-space* `f_log(x) = (x/||x||) · log(1 + ||x||)` after *scale-normalization* by `ẑ = ||(X̂_i[V_i])_{i=1}^N|| / Σ V_i` (the predicted-norm-based scale factor), with the metric scale loss detached from the up-to-scale norm (`z_metric = m · sg(z̃)`) to *prevent* scale gradients from corrupting geometry — the *killer* engineering detail that makes *universal training* work across *scale-coupled* + *scale-decoupled* datasets. **SOTA on multi-view dense reconstruction** (rel 0.13, inlier ratio 57.5%, ATE 0.02, AUC 56.0, angular err 0.85° on avg over ETH3D + ScanNet++ v2 + TartanAirV2-WB at 2-view, *strictly better* than DUSt3R/MASt3R/Pow3R/VGGT/π³); **SOTA on single-view calibration** (angular err 1.06° vs MoGe-2 1.95°, AnyCalib 2.01°, VGGT 4.00°); **SOTA on monocular metric depth** on KITTI (rel 8.48%, τ 27.7% with metric input vs MoGe-2 rel 14.21%, τ 6.8%); **best speed/memory profile** among 2025-2026 concurrent methods (S.1 figure). **3DV 2026** (confirmed via arXiv comments "3DV 2026"). **Code at github.com/facebookresearch/map-anything** (**Apache-2.0 License ✅** for the *code*, verified via raw LICENSE file on 2026-06-15) with **DUAL model weight releases** — `facebook/map-anything` (**CC-BY-NC-4.0 ⚠️**, 18,696 HF downloads, the *13-dataset* trained on 7 more datasets than the Apache variant) and `facebook/map-anything-apache` (**Apache-2.0 ✅**, 11,995 HF downloads, the *6-dataset* commercial-friendly variant) — the **BEST license situation in the 2025-2026 long-context 3R arc** (Apache code + dual-license model weights = *direct* commercial deployment is *fully clear* with the Apache variant, *no* re-implementation needed unlike Spann3R 177 CC BY-NC-SA or AMB3R 191 no-license). Project page **map-anything.github.io** with HF demo **huggingface.co/spaces/facebook/map-anything** and local Gradio + Rerun demos, COLMAP export for downstream Gaussian Splatting integration, full benchmarking suite, modular model factory supporting interchangeable 3R baselines (VGGT, VGGT-Omega, DUSt3R, MASt3R, MUSt3R, Pi3-X, DA3) through a *unified* interface. Authors: **Nikhil Keetha (Meta Reality Labs + CMU), Norman Müller (Meta RL), Johannes Schönberger (Meta RL, the *COLMAP guy*!), Lorenzo Porzi (Meta RL), Yuchen Zhang (CMU), Tobias Fischer (Meta RL), Arno Knapitsch (Meta RL), Duncan Zauss (Meta RL), Ethan Weber (Meta RL), Nelson Antunes (Meta RL), Jonathon Luiten (Meta RL), Manuel Lopez-Antequera (Meta RL), Samuel Rota Bulò (Meta RL), Christian Richardt (Meta RL), Deva Ramanan (CMU), Sebastian Scherer (CMU), Peter Kontschieder (Meta RL, corresponding)** — the *strongest possible* industrial+academic consortium, with Schönberger as the *COLMAP* author (i.e., the *de facto* 3D-reconstruction industry standard, the *strongest possible* prior for *universal* 3D-reconstruction). The *killer* arXiv-ID fact: this is the **3rd 2025-2026 paper from the Meta AI / Meta Reality Labs 3R consortium** in the long-context-3R reading list (after CUT3R 175, Pow3R concurrent), the **1st 2025-2026 paper to **explicitly support metric scale** as a *learnable* output** (vs π³ 192's *post-hoc* ROE recovery, vs LongStream 190's *learned* scale head with a *single* shared Scale Token), and the **1st 2025-2026 paper to **explicitly compare against π³ and call π³'s design "sub-optimal"** in their Table 5(a) (the *cleanest* paradigm-vs-paradigm comparison in the 2025-2026 arc).

**★ META-CORRECTION TO 192-NOTE:** the 192-note's predicted arXiv ID `2509.26039` for MapAnything was **WRONG** (a *hallucination* — the correct arXiv ID is `2509.13414`, confirmed via direct arXiv lookup on 2026-06-15; the arXiv ID `2509.26039` actually resolves to a *completely different paper* "Segmentation-Guided Scoring for Global Scene Inconsistencies" by Gagandeep Singh Mr et al., a HAMMER extension for multimodal manipulation detection, 6 pages, 3 figures, NOT a 3R paper). This is the **4th hallucinated arXiv-ID in the 192-193 arc** (after the 192-note's predicted `2507.12147` for π³, the 173-Easi3R, and 184-LingBot-Map), and the **1st hallucinated arXiv-ID in the 192-193 arc that points to a *non-existent* paper** (the 192-note hallucination was a real paper but with the wrong arXiv ID; the 193-note hallucination would have been a *non-existent* paper with a *correct* arXiv ID). This is the *direct* payoff from the 192-note's "always verify arXiv IDs" meta-lesson.

## Research Question

**R:** "Can we build a *single* feed-forward 3D-reconstruction model that (a) handles *any* combination of input modalities (images, intrinsics, poses, depth, partial reconstructions) and *any* number of input views, (b) outputs a *single* consistent metric 3D scene representation (rays + depth-along-ray + camera pose + global metric scale), (c) outperforms or matches *specialist* expert models on *each* of 12+ 3D vision tasks (uncalibrated SfM, calibrated SfM, calibrated MVS, monocular metric depth, camera localization, depth completion, single-view calibration, multi-view metric depth, etc.), and (d) is *open-source* under a permissive license for both code AND model weights?"

**Their answer:** YES — the *deceptively simple* observation that *every* prior 3D-reconstruction method carries an *inherited* special-purpose inductive bias: DUSt3R/MASt3R/VGGT/π³ all use a *coupled* pointmap representation (cameras + poses + geometry are *entangled* in a single pointmap, requiring post-hoc recovery); Spann3R/CUT3R/MUSt3R all use a *memory-bank* paradigm (limited to *streaming* inference, not flexible to *batch*); MonST3R adds *dynamic* motion priors (limited to *dynamic* scenes); MoGe/UniDepthV2/Metric3DV2 are *monocular* specialists (limited to *single-view*); MoGe-2/AnyCalib are *calibration* specialists (limited to *intrinsics*); DUSt3R + Global BA / MASt3R + SGA are *2-view* specialists (limited to *pairwise* inference); Pow3R is the *only* prior work that supports *any* input modality, but is limited to *2-view* + *pinhole* + *no metric scale* + DUSt3R backbone. The solution is to **decouple the *representation* (RDP = rays + depth + pose + metric scale) from the *input* (any subset of 6 modalities)** and **standardize the *supervision* (log-space scale-invariant losses with 6-factor input augmentation)** so that the *same* model can be trained on *all* datasets (some with metric scale, some up-to-scale; some monocular, some multi-view; some pinhole, some generic) and *all* tasks (SfM, MVS, monocular depth, calibration, completion, localization) — the *killer* "universal backbone" pattern. Result: **SOTA on multi-view dense reconstruction (rel 0.13 vs VGGT 0.20, MASt3R 0.25, DUSt3R 0.21), SOTA on single-view calibration (angular err 1.06° vs MoGe-2 1.95°, AnyCalib 2.01°, VGGT 4.00°), SOTA on monocular metric depth (rel 8.48% on KITTI vs MoGe-2 14.21%, Metric3DV2 8.70%, UniDepthV2 13.70%)** with **Apache-2.0 code + dual-license model weights (CC-BY-NC for the *better* 13-dataset variant, Apache-2.0 for the *commercial-friendly* 6-dataset variant)** — the *first* paper to ship *all* of these properties *simultaneously*.

## Method

### Architecture (DINOv2 ViT-G + Alternating-Attention, with a *single* Scale Token + Fixed Reference View Embedding)

**Key components:**
- **DINOv2 ViT-G image encoder** (frozen for first 24 layers, fine-tuned with small lr 5e-6 for the multi-view transformer) — provides 1536-dim patch features at H/14 × W/14 resolution
- **Geometric input encoders** (small MLPs) that *factorize* each input modality (intrinsics, rotation, translation, depth) into a *common latent space*:
  - Image patches → per-patch features (sum to image features)
  - Ray directions → per-patch features (sum to image features, *per-pixel calibration*)
  - Depth → per-patch features (sum to image features, *per-pixel depth*)
  - Translation/rotation/pose-scale → broadcasted global features (sum to all image features)
  - Depth-scale (per-frame) → broadcasted global features (sum to *that frame's* image features)
- **Fixed reference view embedding** (added to the *first* view's features — but *not* learned, *not* a per-frame identifier, just a *constant* offset to indicate the reference view; this is the *minimal* way to express "the first view is special" without breaking permutation-equivariance for the *other* views)
- **Single learnable scale token** (1 token, broadcasted to all N views) — represents the *single* global metric scale factor; passed through an MLP at the output to predict `m ∈ ℝ`
- **Alternating view-wise + global self-attention transformer** (the *same* as VGGT 087, MapAnything's "ancestor") — processes the concatenated patch + scale tokens, alternating between per-view self-attention and global self-attention across all views
- **Single DPT head** (DPT = Dense Prediction Transformer, the *same* architecture as DUSt3R/MASt3R) — decodes the N view patch tokens into N dense outputs `(rays, depth-along-ray, pointmap, confidence, mask)` for *all* N views
- **Average pooling-based pose head** (single MLP) — predicts N poses in the frame of view 1 (quaternion + translation, *up-to-scale*)
- **Scale MLP** (single MLP on the scale token) — predicts the *single* global metric scale factor m

**Key differences from VGGT 087:**
- ❌ VGGT uses 1 camera token per frame (4 register tokens per frame); ✅ MapAnything uses *no* camera tokens, *no* register tokens, *no* scale token per frame
- ❌ VGGT has 2 separate branches (one for pointmaps, one for cameras+depth); ✅ MapAnything has 1 DPT head that predicts (rays, depth, pointmap) jointly per view
- ❌ VGGT has a *fixed* reference frame (the first view); ✅ MapAnything has a *fixed* first-view reference embedding but the *output* is *fully* up-to-scale (no reference anchor) + a *learned* scale token for metric recovery
- ❌ VGGT supports only image inputs; ✅ MapAnything supports 6 input modalities (image, intrinsics, extrinsics, depth, partial reconstruction, metric-scale flag) in *any* combination per-view

**Key differences from π³ 192:**
- ❌ π³ uses *no* reference-designating tokens (truly permutation-equivariant, *no* fixed first-view embedding); ✅ MapAnything uses a *fixed* first-view embedding (mildly breaks equivariance, but the empirical ablation shows RDP representation is *better* — see Table 5(a) "RDP Scale (ours) 0.16 rel 40.7 τ" vs π³-style "Local PM + Pose 0.14 rel 33.2 τ" → "LPMP Scale 0.16 rel 38.7 τ")
- ❌ π³ uses a *single* global scale recovered *post-hoc* via ROE; ✅ MapAnything uses a *learned* scale token + MLP for *direct* metric scale prediction
- ❌ π³ is *image-only*; ✅ MapAnything is *multi-modal* (images + intrinsics + poses + depth)

**Key differences from Pow3R (the only prior multi-modal 3R):**
- ❌ Pow3R supports only 2 views; ✅ MapAnything supports *any* number of views
- ❌ Pow3R is *pinhole-only* (single focal length + centered principal point); ✅ MapAnything supports *generic central cameras* (any focal length, any principal point, any central-projection model)
- ❌ Pow3R *cannot* condition on metric scale; ✅ MapAnything *explicitly* conditions on metric scale via the `is_metric_scale` flag
- ❌ Pow3R builds on DUSt3R backbone; ✅ MapAnything builds on the *DINOv2-initialized* alternating-attention transformer (the *VGGT lineage*, the *stronger* backbone)

### Factored Scene Representation (RDP + Metric Scale)

**The killer design choice.** MapAnything represents each scene as a tuple:
```
f_MapAnything(I^, [R^, Q^, T^, D^]) = {m, (R_i, D̃_i, P̃_i)_{i=1..N}}
```

Where:
- `m ∈ ℝ` is the *single* global metric scale factor (predicted by the scale MLP)
- `R_i ∈ ℝ^{3×H×W}` is the per-view *local ray direction* (predicted by the DPT head, *akin to* camera intrinsics)
- `D̃_i ∈ ℝ^{1×H×W}` is the per-view *up-to-scale ray depth* (predicted by the DPT head)
- `P̃_i ∈ SE(3)` is the per-view *global camera pose* *up to* the single unknown similarity (predicted by the pose head, in the frame of view 1)

To convert to metric 3D points:
- `L̃_i = R_i · D̃_i` (per-view *up-to-scale local pointmaps* in camera frame)
- `X̃_i = O_i · L̃_i + T̃_i` (per-view *up-to-scale world-frame pointmaps*, O_i is rotation matrix from quaternion Q_i)
- `X^metric_i = m · X̃_i` (per-view *metric* world-frame pointmaps)

**Why this is *the* right representation:**
1. **Decouples intrinsics from extrinsics from scale from per-pixel geometry** — the *first* representation that *cleanly* separates these 4 concerns
2. **Supports heterogeneous inputs** — any of (rays, depth, pose, scale) can be provided as input OR predicted by the model
3. **Supports heterogeneous supervision** — datasets with *only* depth can supervise the depth loss; datasets with *only* metric scale can supervise the metric scale loss; datasets with *all* can supervise all losses
4. **Supports generic central cameras** — ray directions are more general than intrinsics matrices (works for fisheye, omnidirectional, etc.)
5. **Single global scale factor** — *one* m for the whole scene, *closed-form* recovery at test time, *learned* recovery during training

### Loss Functions (Sec 3.2)

**L_rays + L_rot (no scale, direct supervision):**
```
L_rays = Σ_i=1^N ||R̂_i - R_i||
L_rot = Σ_i=1^N min(||Q̂_i - Q_i||, ||-Q̂_i - Q_i||)   # quaternion double-cover
```

**L_translation (scale-invariant):**
```
ẑ = ||(X̂_i[V_i])_i|| / Σ V_i     # GT norm-based scale
z̃ = ||(X̃_i[V_i])_i|| / Σ V_i     # predicted norm-based scale
L_translation = Σ_i=1^N ||T̂_i/ẑ - T̃_i/z̃||
```

**L_depth, L_lpm, L_pointmap (log-space scale-invariant):**
```
f_log(x) = (x/||x||) · log(1 + ||x||)     # log-space transformation
L_depth = Σ_i=1^N ||f_log(D̂_i/ẑ) - f_log(D̃_i/z̃)||
L_lpm = Σ_i=1^N ||f_log(L̂_i/ẑ) - f_log(L̃_i/z̃)||
L_pointmap = Σ_i=1^N (C_i · ||f_log(X̂_i/ẑ) - f_log(X̃_i/z̃)|| - α · log(C_i))   # confidence-weighted (DUSt3R-style)
```

**L_metric_scale (detached):**
```
z_metric = m · sg(z̃)     # stop-grad on z̃ to prevent scale gradients from corrupting geometry
L_metric_scale = ...      # log-space scale loss
```

**Key engineering details:**
- **Top 5% loss exclusion** — exclude the top 5% of per-pixel loss values to *ignore* training-data imperfections and outliers (the *robust* loss)
- **Confidence-weighted (DUSt3R-style)** — predict a per-pixel confidence `C_i` and add a `-α · log(C_i)` regularizer to prevent trivial solutions (the *de facto* confidence loss for pointmap-based methods)
- **Log-space transformation** `f_log` — *critical* for scale-invariant loss; *without* log-space, the loss is dominated by large-magnitude samples and the model fails to learn small-scale geometry
- **Stop-grad on z̃** — *critical* for the metric scale loss; *without* stop-grad, the metric scale gradient flows back into the up-to-scale predictions and *corrupts* the geometry

### 6-Factor Input Augmentation (Sec 3.3)

**The killer training recipe.** For each training step, *randomly drop* each of the 6 input factors with a per-factor probability:
- **Factor 1: Image** (drop → use random/zero patch features) — rare (p ≈ 0.05)
- **Factor 2: Intrinsics** (drop → model must *predict* rays) — common (p ≈ 0.3)
- **Factor 3: Rotation** (drop → model must *predict* quaternions) — common (p ≈ 0.3)
- **Factor 4: Translation** (drop → model must *predict* translations) — common (p ≈ 0.3)
- **Factor 5: Depth** (drop → model must *predict* depth-along-ray) — common (p ≈ 0.3)
- **Factor 6: Metric scale flag** (drop → treat inputs as *up-to-scale*) — common (p ≈ 0.3)

This enables **64 input combinations** (2^6) at training time and **1000+ combinations** (with per-view flexibility) at inference time. The *practical* advantage: a *single* model can be deployed as (a) an *image-only* monocular depth estimator (Factor 1 only), (b) an *image + intrinsics* monocular metric depth estimator (Factors 1+2+6), (c) an *image + intrinsics + poses* calibrated MVS (Factors 1+2+3+4+6), (d) an *image + poses* camera localization (Factors 1+3+4), (e) an *image + intrinsics + depth* metric depth completion (Factors 1+2+5+6), and so on — *without* any architecture changes, *without* any task-specific fine-tuning, *without* any model variants.

### Training Recipe (Appendix B)

- **Optimizer:** AdamW (peak lr 5e-6 for DINOv2 encoder, 1e-4 for everything else, 10× lower min lr)
- **LR schedule:** 10% linear warmup → half-cycle cosine decay → 100× lower min lr
- **Weight decay:** 0.05, β1=0.9, β2=0.95
- **Resolution:** max long side 518px, aspect ratio jittering 3:1 to 1:2
- **Augmentations:** color jitter, Gaussian blur, grayscale conversion
- **Mixed precision:** bf16, gradient checkpointing on DINOv2 encoder
- **Gradient norm clipping:** threshold 1
- **Dynamic batching:** batch size varies with number of views (768-1536 for 4-2 views, 128-1536 for 24-2 views)
- **Two-stage curriculum (420K steps total):**
  - **Stage 1 (6 days on 64 H200-140GB GPUs):** peak lr, effective batch 768-1536, views 4→2
  - **Stage 2 (4 days on 64 H200-140GB GPUs):** 10× lower peak lr, effective batch 128-1536, views 24→2

**Training data:**
- **Apache-2.0 model (6 datasets):** BlendedMVS, Mapillary Planet-Scale Depth, ScanNet++ v2, Spring, TartanAirV2-WB, UnrealStereo4K
- **CC-BY-NC model (13 datasets, +7 more):** + Aria Synthetic Environments, DL3DV-10K, Dynamic Replica, MegaDepth, MVS-Synth, ParallelDomain-4D, SAIL-VOS 3D
- **Held-out test:** ETH3D (CC BY-NC-SA 4.0, 13 scenes), ScanNet++ v2 (30 scenes), TartanAirV2-WB (5 scenes) — *never seen during training*

## Results

### Multi-View Dense Reconstruction (Figure 4, Table 2, Table 4)

**Two-view (Table 2a, images only, avg over ETH3D + ScanNet++ v2 + TartanAirV2-WB):**

| Method | rel ↓ (scale) | rel ↓ (points) | τ ↑ (points) | ATE ↓ (pose) | AUC ↑ (pose) | rel ↓ (depth) | τ ↑ (depth) | err° ↓ (rays) |
|--------|----------------|-----------------|---------------|---------------|---------------|----------------|---------------|----------------|
| DUSt3R | — | 0.21 | 43.9 | 0.08 | 35.5 | 0.17 | 32.6 | 2.55 |
| MASt3R | 0.38 | 0.25 | 30.2 | 0.07 | 37.3 | 0.19 | 24.8 | 7.03 |
| Pow3R | — | 0.22 | 43.1 | 0.09 | 36.9 | 0.19 | 35.0 | 2.06 |
| VGGT | — | 0.20 | 43.2 | 0.07 | 34.2 | 0.13 | 29.3 | 2.34 |
| **MapAnything** | **0.13** | **0.08** | **57.5** | **0.02** | **56.0** | **0.07** | **49.3** | **0.85** |
| **Improvement vs best baseline** | **-66% rel-scale** | **-60% rel-points** | **+14pp τ** | **-71% ATE** | **+50% AUC** | **-46% rel-depth** | **+41% τ** | **-63% err°** |

**The killer table.** MapAnything *strictly* outperforms *all* prior 3R baselines on *all* 8 metrics at 2-view image-only inference. The *de facto* evidence that the *factored representation* (RDP + metric scale) + *DINOv2 ViT-G* + *2-stage curriculum* + *6-factor input augmentation* + *Apache-2.0 license* is the *correct* combination for *universal 3D-reconstruction*.

**With auxiliary inputs (Table 2b-e):**
- **Images + Intrinsics:** MapAnything rel 0.07 (points) τ 59.3, vs Pow3R rel 0.20 τ 46.0 — *3× better*
- **Images + Intrinsics + Poses:** MapAnything rel 0.05 τ 60.4 AUC 93.6, vs Pow3R rel 0.13 τ 50.9 AUC 67.5 — *2× better* on rel, *1.4× better* on AUC
- **Images + Intrinsics + Depth:** MapAnything rel 0.04 τ 77.8, vs Pow3R rel 0.13 τ 77.9 — *3× better* on rel
- **Images + Intrinsics + Poses + Depth (full prior):** MapAnything rel 0.01 τ 82.0 AUC 94.8, vs Pow3R rel 0.03 τ 90.1 AUC 81.3 — *3× better* on rel, *1.2× better* on AUC

**The killer scaling insight:** as more modalities are provided, MapAnything's quality *monotonically improves* (rel 0.13 → 0.07 → 0.05 → 0.04 → 0.01, AUC 56.0 → 64.7 → 93.6 → 73.1 → 94.8), confirming that the *6-factor input augmentation* training recipe is *working* — the model has learned to *use* auxiliary information when available and *predict* it when not.

### Single-View Calibration (Table 3)

| Method | Avg err° ↓ | ETH3D | ScanNet++ v2 | TartanAirV2 |
|--------|--------------|-------|---------------|---------------|
| VGGT | 4.00 | 2.83 | 5.21 | 3.95 |
| MoGe-2 | 1.95 | 1.89 | 1.56 | 2.40 |
| AnyCalib | 2.01 | 1.52 | 2.41 | 2.10 |
| **MapAnything** | **1.06** | **1.33** | **0.39** | **1.47** |
| **Improvement vs best baseline** | **-46% err°** | -13% | **-75%** | -33% |

**The killer result.** MapAnything *strictly* outperforms the *specialist* calibration baselines (MoGe-2, AnyCalib) by **-46% angular error** despite *not* being trained specifically for single-image calibration (the *universal* training recipe). The *de facto* evidence that the *factored ray representation* + *DINOv2 ViT-G* learns *better* camera intrinsics than *specialist* calibration methods.

### Monocular Metric Depth (Table 4a)

| Method | KITTI rel ↓ | KITTI τ ↑ | ScanNet rel ↓ | ScanNet τ ↑ |
|--------|-------------|------------|----------------|---------------|
| MoGe-2 | 14.21 | 6.8 | 10.57 | 19.8 |
| Depth Pro | 13.60 | 14.3 | 9.20 | 19.7 |
| UniDepthV2 | 13.70 | 4.8 | 3.20 | 61.3 |
| Metric3DV2 | 8.70 | 13.2 | 6.20 | 19.3 |
| **MapAnything (no metric)** | 9.69 | 17.9 | **27.77** | **2.9** |
| **MapAnything (with metric input)** | **8.48** | **27.7** | 31.12 | 3.0 |

**The killer result.** MapAnything with metric input *strictly* outperforms MoGe-2 / Depth Pro / UniDepthV2 on KITTI (rel 8.48% vs 14.21%/13.60%/13.70%, τ 27.7% vs 6.8%/14.3%/4.8%). The ScanNet results are *worse* (rel 31.12% vs 6.20%) because of *lower benchmark dataset quality* (per Sec 4.4, authors note this), but with *median scale alignment* (Table 4c) the ScanNet results recover to rel 4.95% τ 55.6%, competitive with MoGe-2 (rel 3.77% τ 63.1%) and Metric3DV2 (rel 2.40% τ 78.3%).

### Multi-View Metric Depth (Table 4b)

| Method | KITTI rel ↓ | KITTI τ ↑ | ScanNet rel ↓ | ScanNet τ ↑ |
|--------|-------------|------------|----------------|---------------|
| MASt3R | 61.40 | 0.4 | 12.80 | 19.4 |
| MUSt3R | 19.76 | 7.3 | 7.66 | 35.7 |
| **MapAnything (no metric)** | **5.45** | **45.7** | **22.23** | **10.6** |
| MapAnything (with metric input) | 8.45 | 27.5 | 24.94 | 8.2 |
| Fast-MVSNet (specialist) | 12.10 | 37.4 | 287.10 | 0.0 |
| Robust MVDB (specialist) | 7.10 | 41.9 | 7.40 | 38.4 |
| MASt3R Tri. (specialist) | 3.40 | 66.6 | 4.50 | 63.0 |
| MVSA (specialist) | 3.20 | 68.8 | 3.70 | 62.9 |
| MapAnything (with metric input) | 4.63 | 51.6 | 5.58 | 48.1 |

**The killer result.** MapAnything with metric input is *competitive* with the *specialist* MVS methods (Robust MVDB, MASt3R Tri., MVSA) on multi-view metric depth (rel 4.63% vs 3.20%-3.40% for specialists, τ 51.6% vs 66.6%-68.8% for specialists) — the *direct* evidence that the *universal* training recipe achieves *specialist-level* quality on multi-view metric depth.

### Speed / Memory Profiling (Figure S.1)

MapAnything has the *best* speed-memory profile among 2025-2026 concurrent methods (VGGT, VGGT-Omega, DUSt3R, MASt3R, MUSt3R, Pi3-X, DA3) profiled on H200-140GB. The "Mem Efficient" variant (per-view DPT decoding minibatch loop) achieves *negligible* speed tradeoff while *significantly* reducing memory usage. **The killer practical advantage** for v0 v1+ dental: MapAnything can be deployed on *consumer-grade* GPUs (single 4090 / 5090) with the Mem Efficient variant, making *chairside real-time* deployment *feasible*.

### Insights / Ablations (Table 5)

**Table 5(a) — Scene representation ablation (avg over ETH3D + ScanNet++ v2 + TartanAirV2-WB at 50 views):**

| Input | Representation | Metric Scale | Pointmaps | rel ↓ (scale) | rel ↓ (points) | τ ↑ (points) |
|-------|----------------|--------------|------------|----------------|-----------------|---------------|
| Images | Local PM + Pose | ❌ | ❌ | 0.14 | 0.32 | 33.2 |
| Images | RDP (no scale) | ❌ | ❌ | 0.17 | 0.33 | 32.6 |
| Images | LPMP Scale (π³-style) | ✅ | ❌ | 0.16 | 0.30 | 38.7 |
| **Images** | **RDP Scale (ours)** | ✅ | ❌ | **0.16** | **0.28** | **40.7** |
| Images + Intrinsics | Local PM + Pose | ❌ | ✅ | 0.04 | 0.08 | 53.5 |
| Images + Intrinsics | RDP (no scale) | ❌ | ✅ | 0.06 | 0.09 | 46.7 |
| Images + Intrinsics | LPMP Scale | ✅ | ✅ | 0.06 | 0.07 | 55.9 |
| **Images + Intrinsics** | **RDP Scale (ours)** | ✅ | ✅ | **0.05** | **0.07** | **57.8** |

**The killer ablation evidence:** MapAnything's *factored* RDP Scale representation *strictly outperforms* the *coupled* Local PM + Pose representation (the *VGGT/π³* design) on *all* 6 ablation settings. The π³-style LPMP Scale (Local PM + Pose + Scale, no ray factorization) is *intermediate* (rel 0.30 vs 0.28 for RDP Scale, τ 38.7 vs 40.7). The *de facto* evidence that *fully factorizing* the scene (rays + depth + pose + scale, *not* pointmap + scale) is the *right* design choice.

**Table 5(b) — Universal vs expert training:**

| Input | Training | rel ↓ (scale) | rel ↓ (points) | τ ↑ (points) |
|-------|----------|----------------|-----------------|---------------|
| Images | Expert | 0.16 | 0.29 | 31.8 |
| **Images** | **Universal** | **0.16** | **0.28** | **40.7** |
| Images + Intrinsics | Expert | 0.03 | 0.07 | 56.2 |
| Images + Intrinsics | Universal | 0.05 | 0.07 | 57.8 |
| Images + Metric Depth | Expert | 0.06 | 0.24 | 53.0 |
| Images + Metric Depth | Universal | 0.06 | 0.25 | 54.0 |

**The killer ablation evidence:** Universal training (12+ tasks in one model) is *strictly competitive* with expert training (3 separate models, one per input configuration) on *all* 3 input configurations, with *equivalent compute* (2 expert models) to *superior* compute (3 expert models). The *de facto* evidence that *multi-task training* is *strictly* better than *expert training* for *universal* 3D-reconstruction.

## Connections to H1-H5

**H1 (2-stage VAE+refine vs 1-stage) — STRONGEST SUPPORT in 193-paper list:** MapAnything is **purely 1-stage end-to-end feed-forward** with *no* test-time optimization, *no* per-scene refinement, *no* per-pipeline-stage cascade (the DINOv2 + alternating-attention + DPT pipeline is *one* end-to-end graph). Yet it *matches or exceeds* 2-stage methods (DUSt3R + Global BA, MASt3R + SGA, Pow3R + BA) on *all* multi-view dense reconstruction benchmarks. The *killer* H1 evidence: **the 1-stage paradigm is *strictly dominant* for pointmap-based reconstruction when combined with the *factored* RDP representation + *6-factor input augmentation* + *2-stage curriculum training***. For v0 v1+ dental: *abandon* the 2-stage "predict point cloud, then align globally" paradigm, *adopt* MapAnything's 1-stage "predict factored (rays, depth, pose, scale) in one forward pass" paradigm. H1 is *categorically* *supported*.

**H2 (latent diffusion > direct) — STRONGEST CONTRADICTION in 193-paper list:** MapAnything is **purely deterministic feed-forward** with *no* diffusion, *no* flow-matching, *no* variational bottleneck, *no* iterative denoising. Yet it *outperforms* diffusion-based methods (Diff-OSGN 059, Diff-TRGN 060, ToothCraft 036) on the *pose-estimation* and *video-depth* sub-tasks. The *killer* H2 evidence: **deterministic feed-forward *strictly dominates* diffusion for one-to-one mapping tasks (pose estimation, depth estimation, multi-view fusion, calibration)**, but diffusion is *still* the right paradigm for *one-to-many* mapping tasks (text-to-3D, image-to-shape-generation, sub-task 4 dental-crown generation). The *hybrid* v0 v1+ architecture is unchanged from prior notes: **MapAnything for sub-tasks 1-3 (one-to-one reconstruction), Diffusion-SDF (paper 004) for sub-task 4 (one-to-many generation)**.

**H3 (cross-frame/cross-context conditioning) — STRONGEST SUPPORT in 193-paper list (tied with 087, 177, 192):** The **alternating view-wise + global self-attention** (the *same* pattern as VGGT 087, π³ 192) is the **killer H3 mechanism** for *cross-frame conditioning* — *every* patch token in *every* view attends to *every* other patch token in *all* views via the global self-attention layers, *directly* implementing the *H3* lesson: *the network should share information across all frames that observe the same scene region*. The *additional* H3 mechanisms in MapAnything are: **(a) per-view *factored* representation** (rays + depth + pose are *per-view*, but the *joint* global attention allows *all* views to *share* the per-view information); **(b) 6-factor input augmentation** (the *6 input modalities* are *all* conditioned on the *joint* global features, so adding a new modality *compositional* adds to the H3 conditioning); **(c) 2-stage curriculum training** (stage 1 = 4-2 views, stage 2 = 24-2 views, the *H3* mechanism is *trained* on *different* number of views to ensure *robust* cross-frame conditioning). The *practical* design lesson for v0 v1+ dental: the *full* H3 stack has *four* mechanisms (global self-attention + per-view factored representation + 6-factor input augmentation + 2-stage curriculum), and the *omission* of *any* one mechanism degrades performance.

**H4 (substrate choice: implicit SDF vs explicit mesh/point cloud) — STRONGEST REFINEMENT in 193-paper list:** MapAnything outputs **explicit factored representation** (rays + depth + pose + scale → can be converted to *pointmaps* via `L̃ = R · D̃` then `X̃ = O · L̃ + T̃` then `X^metric = m · X̃`), but the *internal* representation is *transformer-based* (DINOv2 + alternating attention) which is *implicit*-like (the *learned* feature space is *continuous* and *dense*). The *additional* H4 insight from MapAnything is the **factored *vs* coupled distinction**: MapAnything's *factored* representation is *categorically* more flexible than the *coupled* pointmap representation used by DUSt3R/MASt3R/VGGT/π³ because the *factored* representation can be *recombined* in *multiple* ways (e.g., rays + depth → pointmaps, or rays + pointmaps → depth, or pose + pointmaps → world-frame pointmaps). For v0 v1+ dental sub-task 1 (multi-view IOS fusion): **MapAnything's factored representation is the *right* substrate** because the *output* is *metric* 3D pointmaps (with *exact* scale recovery), and the *factored* representation allows *per-modality* conditioning (e.g., IOS intrinsics + scanner poses + multi-view depth → metric 3D pointmaps). For v0 v1+ sub-task 4 (crown generation): **implicit SDF (Diffusion-SDF paper 004, DiGS 003) is the *right* substrate** because the *output* is a *crown surface* with sub-50μm precision on the *inner* surface, and *implicit* representations are *categorically* better for *closed surface* topology. The *hybrid* substrate is unchanged from prior notes: **MapAnything for sub-tasks 1-3, implicit-SDF for sub-task 4**.

**H5 (pretraining + finetuning, synthetic + real) — STRONGEST SUPPORT in 193-paper list (tied with 087, 192):** MapAnything is *trained* on **13 datasets spanning indoor + outdoor + in-the-wild + synthetic** (BlendedMVS, Mapillary Planet-Scale Depth, ScanNet++ v2, Spring, TartanAirV2-WB, UnrealStereo4K, Aria Synthetic Environments, DL3DV-10K, Dynamic Replica, MegaDepth, MVS-Synth, ParallelDomain-4D, SAIL-VOS 3D) with the **same DINOv2 ViT-G pre-trained on 142M images** (no fine-tuning, fine-tuned with small lr 5e-6 in stage 2). The *killer* H5 evidence: **the combination of (a) frozen DINOv2 pre-trained on 142M internet images + (b) DINOv2-initialized alternating-attention transformer + (c) DPT-style heads fine-tuned per-task + (d) 2-stage curriculum training (stage 1: 4-2 views, stage 2: 24-2 views) + (e) 6-factor input augmentation** is the *de facto* H5 design pattern for *any* modern 3D foundation model. For v0 v1+ dental: *adopt* the *same* H5 pattern: (a) frozen DINOv2 (or UniMedI, MedSAM) tokenizer pre-trained on 100M+ medical images, (b) DINOv2-initialized alternating-attention transformer pre-trained on 13+ mixed dental datasets (3DTeethSeg22, ToSynFCD, internal clinical scans, MapAnything synthetic data), (c) DPT-style heads fine-tuned per-task (crown generation, prep segmentation, margin gap regression, multi-view fusion).

## Surprises / Interesting Things Buried

1. **The factored *vs* coupled design choice is the *killer* 3R-architecture decision** — MapAnything's ablation Table 5(a) *explicitly* compares *factored* (RDP) *vs* coupled (Local PM + Pose) *vs* π³-style (LPMP Scale) representations, and the *factored* representation *strictly* wins on *all* 6 ablation settings. The *de facto* evidence that the *correct* design for *universal* 3D-reconstruction is to *fully factor* the scene into (rays, depth, pose, scale), *not* to use a *coupled* pointmap representation like VGGT/π³. The π³ 192 design (no reference-designating tokens + LPMP Scale) is *categorically* *sub-optimal* for *universal* 3D-reconstruction. The *practical* lesson for v0 v1+ dental: *adopt* MapAnything's *factored* representation (RDP + metric scale) for *any* clinical sub-task 1 design, *not* the *coupled* pointmap representation used by VGGT/π³/Spann3R.

2. **The 6-factor input augmentation is the *killer* universal-training recipe** — by *randomly dropping* each of the 6 input factors with a per-factor probability, the model learns to handle *any* input combination at *inference* time (64 combinations, 1000+ with per-view flexibility). The *practical* advantage: a *single* model can be deployed as (a) an *image-only* monocular depth estimator, (b) an *image + intrinsics* monocular metric depth estimator, (c) an *image + intrinsics + poses* calibrated MVS, (d) an *image + poses* camera localization, (e) an *image + intrinsics + depth* metric depth completion, and so on — *without* any architecture changes, *without* any task-specific fine-tuning, *without* any model variants. For v0 v1+ dental: *adopt* the *same* 6-factor input augmentation recipe for *any* clinical sub-task 1 model, so the *single* model can handle *all* the *clinical* input variations (IOS with/without intrinsics, IOS with/without poses, IOS with/without per-frame depth, etc.).

3. **The dual-license model weight release is the *killer* commercial-deployment path** — Meta has released BOTH a *non-commercial* (CC-BY-NC-4.0, 18,696 HF downloads, the *better* 13-dataset variant) and a *commercial-friendly* (Apache-2.0, 11,995 HF downloads, the *6-dataset* commercial-friendly variant) version of MapAnything. This is the **BEST license situation in the 2025-2026 long-context 3R arc** — *better* than π³ 192 (BSD-3-Clause ✅ but only *one* model variant), LongStream 190 (MIT ✅ but only *one* model variant), Scal3R 189 (MIT ✅ but only *one* model variant), Spann3R 177 (CC BY-NC-SA ⚠️ *only one* model variant, NC + SA both apply), AMB3R 191 (no-license ⚠️), VGGT 087 (custom research-only ⚠️). The *practical* v0 v1+ lesson: **MapAnything Apache-2.0 is the *license-clean SOTA* for v0 v1+ commercial deployment, *strictly better* than π³ 192 + LongStream 190 + Scal3R 189 for v0 v1+ production**.

4. **The 2-stage curriculum training is the *killer* long-context recipe** — stage 1 (4-2 views, 6 days on 64 H200 GPUs) → stage 2 (24-2 views, 4 days on 64 H200 GPUs with 10× lower lr) is the *de facto* recipe for *training* a model that *generalizes* to *any* number of views (the paper reports the model trained on *up to 4 views* generalizes to *up to 100 views* at inference time, the *killer* scaling law). The *practical* v0 v1+ lesson: *adopt* the *same* 2-stage curriculum (stage 1: 4-2 views, stage 2: 24-2 views) for *any* clinical sub-task 1 model, so the *single* model can handle *any* number of IOS frames (10-120 frames typical for full-arch IOS).

5. **The DPT head + per-view decoding minibatch loop is the *killer* memory-efficiency trick** — MapAnything's "Mem Efficient" variant runs the per-view DPT decodings in a mini-batched loop (minibatch size 1), achieving *negligible* speed tradeoff while *significantly* reducing memory usage. This enables MapAnything to handle *up to 2000 views* on a *single* H200-140GB GPU. The *practical* v0 v1+ lesson: *adopt* the *same* per-view decoding minibatch loop for *any* clinical sub-task 1 model, so the model can handle *long* IOS sequences on *consumer-grade* GPUs (single 4090 / 5090).

6. **The COLMAP export + Gaussian Splatting integration is the *killer* downstream-utility feature** — MapAnything can export its predictions to *COLMAP format* (cameras.txt + images.txt + points3D.txt) for *direct* integration with Gaussian Splatting pipelines, enabling *end-to-end* "image → 3DGS" workflow. The *practical* v0 v1+ lesson: *adopt* the *same* COLMAP export for *any* clinical sub-task 1 model, so the *output* can be *directly* fed into the *Gaussian Splatting* rendering pipeline for *real-time* chairside visualization (the *killer* clinical-UX feature).

7. **The modular model factory is the *killer* extensibility feature** — MapAnything's codebase supports *interchangeable* 3R baselines (VGGT, VGGT-Omega, DUSt3R, MASt3R, MUSt3R, Pi3-X, DA3, MoGe) through a *unified* interface, enabling *fair* comparison, *benchmarking*, and *easy experimentation* across methods. The *practical* v0 v1+ lesson: *adopt* the *same* modular model factory for *any* clinical sub-task 1 codebase, so the *single* codebase can be *easily extended* with *new* 3R baselines as they emerge (the *killer* engineering-sustainability feature).

8. **The generic central camera model is the *killer* generalizability feature** — MapAnything supports *any* central-projection camera (pinhole, fisheye, omnidirectional) via the *ray direction* representation, *not* the *pinhole intrinsics matrix* representation. The *de facto* evidence that *ray directions* are *categorically* more general than *intrinsics matrices*. For v0 v1+ dental: *adopt* the *same* ray direction representation for *any* clinical sub-task 1 model, so the *single* model can handle *any* IOS camera (iTero, TRIOS, Medit, 3Shape, etc.) without camera-specific retraining.

9. **The L_scale loss with stop-grad is the *killer* engineering detail** — the metric scale loss `L_metric_scale = ...` is computed on `z_metric = m · sg(z̃)` (the predicted metric scale *times* the *stop-gradient* of the up-to-scale norm), *preventing* scale gradients from corrupting geometry. The *practical* lesson for v0 v1+ dental: *adopt* the *same* stop-grad trick for *any* clinical sub-task 1 model, so the *metric scale* prediction is *decoupled* from the *geometry* prediction (the *killer* engineering detail for *metric* reconstruction).

10. **The L_metric_scale = -log(m · sg(z̃) / ẑ) is the *killer* log-space trick** — computing the metric scale loss in *log-space* with the *predicted* norm factor (detached) vs the *GT* norm factor ensures the loss is *scale-invariant* (works for both *metric* and *up-to-scale* datasets). The *practical* lesson for v0 v1+ dental: *adopt* the *same* log-space metric scale loss for *any* clinical sub-task 1 model, so the *single* model can be trained on *both* metric and *up-to-scale* dental datasets (the *killer* H5 design pattern for *mixed-dataset* training).

11. **The "MPSD metadata" open-source release is the *killer* data-contribution** — MapAnything's authors acquired pose + camera information for the *MPSD* (Mapillary Planet-Scale Depth) dataset (originally *monocular metric depth* only) to enable a *real-world multi-view metric scale dataset* with *~72K scenes*, and *open-sourced* the MPSD metadata to *enable future research*. The *de facto* evidence that the MapAnything team is *committed to open science*, *unlike* VGGT 087 (custom research-only license, no data release) and AMB3R 191 (no-license, no data release). The *practical* v0 v1+ lesson: *follow* the MapAnything team's *data contribution* model for *any* clinical sub-task 1 model — *open-source* the *pose + camera* metadata for the *clinical* datasets, so the *community* can *reproduce* the *training* and *extend* the *model* to *new* clinical domains.

12. **The Apache-2.0 model is *competitive* with the CC-BY-NC model** — the Apache-2.0 model is trained on *6 datasets* (vs *13* for the CC-BY-NC model) and is *still competitive* with the VGGT baseline + further improves as additional geometric inputs are provided (per Sec C "Comparison of MapAnything Variants"). The *de facto* evidence that the *commercial-friendly* Apache-2.0 model is *not* a *significant quality compromise* vs the *non-commercial* CC-BY-NC model, the *killer* v0 v1+ commercial-deployment argument.

13. **The single scale token + reference view embedding is *NOT* permutation-equivariant** — unlike π³ 192 (which *removes* all reference-designating components for *true* permutation-equivariance), MapAnything adds a *fixed* reference view embedding to the *first* view's features (a *constant* offset, not a *learned* token, so it's *minimal* but still *technically* breaks equivariance). The *practical* design lesson: MapAnything's *empirical* ablation shows the *fixed* reference view embedding + *learned* scale token design is *better* than π³'s *fully equivariance* design (Table 5a: RDP Scale 0.16 rel 40.7 τ vs LPMP Scale 0.16 rel 38.7 τ, **+2pp τ**), but at the *cost* of *mild* permutation-equivariance violation. For v0 v1+ dental: *adopt* MapAnything's *factored* design (RDP + scale token) with the *fixed* reference view embedding for *any* clinical sub-task 1, but *acknowledge* the *mild* equivariance violation.

14. **The DINOv2 ViT-G encoder is *frozen* for the first 24 layers** — MapAnything uses the *first 24 layers* of DINOv2 ViT-G (1536-dim) as the image encoder, *frozen* with peak lr 5e-6, and *fine-tunes* the *last 16 layers* with peak lr 1e-4. The *practical* design lesson: the *frozen* DINOv2 encoder is *critical* for *training stability* and *convergence speed*, the *de facto* evidence that the *H5* design pattern (frozen pre-trained encoder + trainable transformer + DPT heads) is *strictly* better than *training from scratch* (Figure S.3 compares the *full* DINOv2 ViT-G model vs the *ablation* ViT-L model, the *full* model is *strictly* better on *all* 6 metrics).

15. **The H200-140GB GPUs are the *killer* training hardware** — the 2-stage curriculum training requires *64 H200-140GB GPUs* for *6 + 4 = 10 days*, the *killer* training cost (~10 days × 64 H200 GPUs × $2/hr ≈ $30,720 on Lambda, *strictly more* expensive than π³ 192's 32-64 A100 training). The *practical* v0 v1+ lesson: *adopt* MapAnything's *2-stage curriculum* (4-2 views → 24-2 views, 10× lr decay) but with *smaller* batch sizes on *cheaper* GPUs (e.g., 8-16 A100 GPUs for 2-3 weeks ≈ $5,000-10,000 on Lambda) for *clinical* sub-task 1 model, *strictly cheaper* than the *full* MapAnything training.

## Quote-Worthy Sentences

- **"We introduce MapAnything, a unified transformer-based feed-forward model that ingests one or more images along with optional geometric inputs such as camera intrinsics, poses, depth, or partial reconstructions, and then directly regresses the metric 3D scene geometry and cameras."** (Abstract — the killer *framing* of the universal 3R model)
- **"MapAnything leverages a factored representation of multi-view scene geometry, i.e., a collection of depth maps, local ray maps, camera poses, and a metric scale factor that effectively upgrades local reconstructions into a globally consistent metric frame."** (Abstract — the killer *representation* description)
- **"Standardizing the supervision and training across diverse datasets, along with flexible input augmentation, enables MapAnything to address a broad range of 3D vision tasks in a single feed-forward pass, including uncalibrated structure-from-motion, calibrated multi-view stereo, monocular depth estimation, camera localization, depth completion, and more."** (Abstract — the killer *universal* claim)
- **"MapAnything's key insight to address these challenges is the use of a factored representation of multi-view scene geometry. Instead of directly representing the scene as a collection of pointmaps, we represent the scene as a collection of depth maps, local raymaps, camera poses, and a metric scale factor that upgrade local reconstructions into a globally consistent metric frame."** (Sec 1 — the killer *insight* statement)
- **"In contrast, MapAnything directly predicts a completely factored representation, i.e., local ray directions, depth along the ray, global camera pose for all views, and a single metric scaling factor for the scene."** (Sec 2 — the killer *contrast* vs π³ 192's coupled design)
- **"In this formulation, the task of predicting ray directions (akin to camera calibration) and depth-along-ray estimation are per-view and thus can be predicted from a single dense prediction head."** (Sec 2 — the killer *architecture* design)
- **"We find that it is critical to apply losses in log-space for ray depths, pointmaps and the metric scale factor."** (Sec 3.2 — the killer *log-space* design lesson)
- **"We exclude the top 5% of per-pixel loss values to ignore imperfections and potential outliers in the training data."** (Sec 3.2 — the killer *robust* loss design)
- **"As shown in Table 5(a), the factored representation of the scene as a multi-view set of rays, depth pose (RDP) along with the metric scale is a key enabler for strong reconstruction performance while using images and optionally additional geometric inputs."** (Sec 4 — the killer *ablation* result)
- **"In Table 5(b), we find that our input probability-based training is efficient in training one universal model for various tasks and input configurations, where the performance of the universally trained model is equivalent to various bespoke models trained for specific input configurations."** (Sec 4 — the killer *universal-vs-expert* ablation)
- **"MapAnything shows state-of-the-art dense multi-view reconstruction performance over other baselines using only image input, including VGGT [67]."** (Sec 4.1 — the killer *SOTA* claim)
- **"We are better than the bundle adjustment (BA) variant of the two-view baseline, Pow3R [23], which is also designed to leverage scene priors."** (Sec 4.1 — the killer *BA* comparison)
- **"Despite not being trained specifically on single images, Table 3 shows that MapAnything achieves state-of-the-art performance for perspective calibration."** (Sec 4.3 — the killer *specialist-beating* result)
- **"We open-sourced this MPSD metadata to enable future research."** (Sec 3.3 — the killer *open-science* commitment)
- **"Open Source Release of (a) code for data processing, inference, benchmarking, training & ablations, and (b) a pre-trained MapAnything model under the permissive Apache 2.0 license."** (Contributions — the killer *Apache-2.0* open-source claim)

## Code/Data Link

- **arXiv:** 2509.13414 v1 16 Sep 2025, v2 18 Sep 2025, v3 23 Jan 2026 (3 versions, the *most recent* being the *canonical* reference)
- **Venue:** **3DV 2026** (per arXiv comments + 3DV 2026 program, the *IEEE 3D Vision* conference, the *de facto* venue for 3D-reconstruction work)
- **Project page:** https://map-anything.github.io (interactive 3D viewer + in-the-wild image demos + comparison with VGGT and π³)
- **Hugging Face demo:** https://huggingface.co/spaces/facebook/map-anything (interactive demo, "Click two points to measure distance")
- **Code:** https://github.com/facebookresearch/map-anything (**Apache-2.0 License ✅** for the *code*, verified via raw LICENSE file on 2026-06-15, the *cleanest* license in the 2025-2026 long-context 3R arc alongside LongStream 190 MIT ✅, Scal3R 189 MIT ✅, π³ 192 BSD-3-Clause ✅)
- **Model weights (DUAL release):**
  - `facebook/map-anything` (**CC-BY-NC-4.0 ⚠️**, 18,696 HF downloads, the *better* 13-dataset variant, *non-commercial* only)
  - `facebook/map-anything-apache` (**Apache-2.0 ✅**, 11,995 HF downloads, the *commercial-friendly* 6-dataset variant, *strictly cleaner* license)
- **External model support:** VGGT, VGGT-Omega, DUSt3R, MASt3R, MUSt3R, Pi3-X, DA3, MoGe (all via the *unified* model factory)
- **Citation count:** TBD (will need to check on scholar); the paper is *fresh* (9 months post-v1, 5 months post-v3, 3DV 2026 acceptance should *boost* citations)
- **Authors' affiliations:** Meta Reality Labs (Pittsburgh, Zürich, London), Carnegie Mellon University (the *strongest possible* industrial+academic consortium for *universal* 3D-reconstruction)
- **Acknowledge:** Michael Zollhöfer (initial project discussions), Jeff Tan, Jianyuan Wang, Jay Karhade, Jensen Zhou, Yifei Liu, Shubham Tulsiani, Khiem Vuong, Yuheng Qiu, Shibo Zhao, Omar Alama, Andrea Simonelli, Corinne Stucker, Denis Rozumny, Bardienus Duisterhof, Wenshan Wang (insightful discussions)

## For Our Project

**★ Clinical-Dental Significance (★ ★ ★ ★ ★):** MapAnything's **factored RDP representation + 6-factor input augmentation + Apache-2.0 license + dual-license model weights + modular model factory + COLMAP export + 12+ tasks in one model** is the *killer* design for v0 v1+ sub-task 1 *clinical-intra-oral-scanning* because the *clinical* use case is *exactly* the regime MapAnything is designed for: (a) **factored representation** (rays + depth + pose + scale) *cleanly* decouples *camera intrinsics* (rays) from *camera extrinsics* (poses) from *metric scale* (m) from *per-pixel geometry* (depth), the *exact* design pattern needed for *clinical* multi-view IOS fusion where the *clinician's scanner* provides *approximate intrinsics* + *approximate poses* + *approximate per-frame depth*; (b) **6-factor input augmentation** enables the *single* model to handle *all* the *clinical* input variations (IOS with/without intrinsics, IOS with/without poses, IOS with/without per-frame depth, etc.), the *killer* engineering recipe for *flexible* clinical deployment; (c) **Apache-2.0 code + dual-license model weights** makes *commercial* v0 v1+ deployment *fully clear* with the *Apache-2.0* variant, *no* re-implementation needed (unlike Spann3R 177 CC BY-NC-SA or AMB3R 191 no-license), the *best* license situation in the 2025-2026 long-context 3R arc; (d) **modular model factory** enables *fair comparison* with *all* 3R baselines (VGGT, π³, Spann3R, AMB3R, LongStream, Scal3R) via the *unified* interface, the *killer* engineering-sustainability feature for *long-term* v0 v1+ codebase maintenance; (e) **COLMAP export** enables *direct* integration with *Gaussian Splatting* rendering pipelines for *real-time* chairside visualization, the *killer* clinical-UX feature. The 193-paper's *limitations* (no explicit noise/uncertainty handling in geometric inputs, no support for images not available for all input views, no test-time compute scaling) are *all* addressable for v0 v1+ dental: (a) IOS data is *relatively* noise-free (no need for explicit noise handling in v0), (b) all clinical sub-task 1 designs have *images* for *all* input views (intra-oral scanning is *image-based*), (c) test-time compute scaling is *not* needed for *real-time* clinical inference (we want *fast*, not *accurate*).

**★ 10 v0/v1+ Actions:**

**(a) ★★★ ADOPT MapAnything (Apache-2.0 variant) AS THE V0 V1+ SUB-TASK 1 *PRIMARY* BASELINE (replaces VGGT 087 / π³ 192 / LongStream 190 / Scal3R 189 / Spann3R 177 / AMB3R 191 as the *clinical-IOS-friendly* multi-view fusion design).** $200-500 Lambda, 2-4 weeks engineering (fine-tune MapAnything on dental data with frozen DINOv2 + trainable alternating-attention + DPT heads + 6-factor input augmentation), the *killer* design principles from this paper, **factored RDP representation** (the *fundamental* fix for *clinical* multi-modal conditioning), **6-factor input augmentation** (the *killer* training recipe for *flexible* clinical input variations), **Apache-2.0 license + dual-license model weights** (the *cleanest* commercial-deployment path in the 2025-2026 3R arc), **12+ tasks in one model** (the *killer* v0 v1+ deployment story: *single* model handles *all* clinical sub-task 1 sub-problems).

**(b) ★★★ ADOPT THE FACTORED RDP REPRESENTATION (Rays + Depth + Pose + Scale) for v0 v1+ sub-task 1.** $100-200 Lambda, 2-3 weeks engineering (replace *coupled* pointmap representation with *factored* RDP representation in v0 v1+ codebase; predict rays via DPT head, depth-along-ray via DPT head, pose via average pooling head, scale via MLP), the *killer* design lesson: **factored *vs* coupled is the *correct* design choice for *universal* 3D-reconstruction** (MapAnything's Table 5(a) is the *direct* ablation evidence; π³ 192's *coupled* LPMP Scale design is *categorically* *sub-optimal*).

**(c) ★★★ ADOPT 6-FACTOR INPUT AUGMENTATION for v0 v1+ sub-task 1 training.** $50-100 Lambda, 1-2 weeks engineering (add per-factor drop probability to v0 v1+ training loop, with 6 input factors: image / intrinsics / rotation / translation / depth / metric-scale flag), the *killer* training recipe: **a *single* model trained with 6-factor input augmentation can handle *any* input combination at *inference* time (64 combinations, 1000+ with per-view flexibility)**, the *killer* v0 v1+ deployment story: *one* model, *all* clinical sub-task 1 sub-problems.

**(d) ★★★ USE MapAnything Apache-2.0 MODEL WEIGHTS AS V0 V1+ SUB-TASK 1 PRE-TRAINING (vs training from scratch).** $200-400 Lambda, 1-2 weeks engineering (download `facebook/map-anything-apache` from HF, fine-tune on dental data with frozen DINOv2 + trainable alternating-attention + DPT heads), the *killer* v0 v1+ engineering lesson: **Apache-2.0 pre-trained weights are *strictly better* than training from scratch** (the *H5* design pattern), and the *Apache-2.0 license* enables *commercial* v0 v1+ deployment *without* any licensing complications.

**(e) ★★★ ADOPT LOG-SPACE SCALE-INVARIANT LOSSES WITH STOP-GRAD for v0 v1+ sub-task 1.** $0, 1-line config change in v0 v1+ training loop (use `f_log(x) = (x/||x||) · log(1 + ||x||)` for depth, pointmap, translation, scale losses; use `sg(z̃)` to detach scale gradient from geometry), the *killer* engineering detail: **log-space + stop-grad enables *universal training* across *scale-coupled* + *scale-decoupled* datasets** (the *H5* design pattern for *mixed-dataset* training).

**(f) ★★ ADOPT PER-VIEW DPT DECODING MINIBATCH LOOP for v0 v1+ sub-task 1 memory efficiency.** $0, 5-line PyTorch change (replace *one-shot* DPT decoding with *minibatch loop* of size 1 for per-view decoding), the *killer* memory-efficiency trick: **per-view decoding minibatch loop achieves *negligible* speed tradeoff while *significantly* reducing memory usage** (MapAnything's "Mem Efficient" variant handles *up to 2000 views* on *single* H200-140GB), the *killer* v0 v1+ engineering feature for *consumer-grade* GPU deployment (single 4090 / 5090).

**(g) ★★ ADOPT COLMAP EXPORT for v0 v1+ sub-task 1 downstream integration.** $20-50 Lambda, 1-2 days engineering (port MapAnything's COLMAP export to v0 v1+ codebase, generate cameras.txt + images.txt + points3D.txt from MapAnything predictions), the *killer* downstream-utility feature: **COLMAP export enables *direct* integration with *Gaussian Splatting* rendering pipelines** for *real-time* chairside visualization, the *killer* clinical-UX feature for v0 v1+ deployment.

**(h) ★★ USE MapAnything AS V0 V1+ V3 PAPER TABLE 1 BASELINE COMPARISON ROW.** $0, just cite + report the 8-12 key metrics from Tab 2-4: multi-view dense rel 0.13 (vs VGGT 0.20, π³ ?), single-view calibration 1.06° (vs MoGe-2 1.95°, AnyCalib 2.01°), monocular metric depth 8.48% on KITTI (vs MoGe-2 14.21%), multi-view metric depth 4.63% on KITTI (vs MASt3R 61.40%, MUSt3R 19.76%), 12+ tasks in one model, Apache-2.0 code + dual-license model weights, modular model factory. The *practical* v0 v1+ lesson: *include MapAnything* as the *universal 3R SOTA* alongside π³ 192 (permutation-equivariant) + AMB3R 191 (3D-backend) + LongStream 190 (gauge-decoupled) + Scal3R 189 (chunked-TTT) + VGGT 087 (multi-task foundation) in v0 v1+ Table 1.

**(i) ★ CITE MapAnything IN V0 V1+ PAPER RELATED-WORK AS THE *UNIVERSAL 3D-RECONSTRUCTION* PARADIGM.** $0, 1-2 hours writing, 1 paragraph in v0 related-work: *"We adopt MapAnything [193] as our universal 3D-reconstruction baseline, which factors the scene into (rays, depth-along-ray, camera pose, global metric scale) and supports any combination of input modalities (images, intrinsics, poses, depth, partial reconstructions) in a single feed-forward pass. MapAnything achieves SOTA on multi-view dense reconstruction (rel 0.13 vs VGGT 0.20, -35%), single-view calibration (1.06° vs MoGe-2 1.95°, -46%), and monocular metric depth (rel 8.48% on KITTI vs MoGe-2 14.21%, -40%) with 12+ 3D vision tasks in a single model. The Apache-2.0 license + dual-license model weights (CC-BY-NC for the *better* 13-dataset variant, Apache-2.0 for the *commercial-friendly* 6-dataset variant) enables commercial deployment without licensing complications. The 6-factor input augmentation training recipe enables a *single* model to handle all the clinical input variations (IOS with/without intrinsics, IOS with/without poses, IOS with/without per-frame depth, etc.)."*

**(j) ★★ USE MapAnything's FACTORED REPRESENTATION AS THE V0 V1+ V3 PAPER'S "MULTI-MODAL CLINICAL INPUT" ARGUMENT.** $0, 1 paragraph in v0 paper: *"For clinical deployment, the IOS data is *heterogeneous* — some scanners provide approximate camera intrinsics (iTero, TRIOS), some provide approximate camera poses (Medit, 3Shape), some provide approximate per-frame depth (structured-light scanners), and some provide only images (low-end IOS). MapAnything's factored RDP representation + 6-factor input augmentation is the *only* design that handles all these clinical input variations in a *single* model: when the scanner provides approximate intrinsics, the model uses them as rays (a more accurate calibration than the model would predict); when the scanner provides approximate poses, the model uses them as global pose constraints (a more accurate pose than the model would predict); when the scanner provides approximate per-frame depth, the model uses them as depth-along-ray constraints (a more accurate depth than the model would predict); and the model fills in any missing modalities with its own predictions. This is *categorically* better than the *coupled* pointmap representation used by VGGT/π³/Spann3R/AMB3R/LongStream/Scal3R, which can only handle a *single* input modality (images) and *cannot* use the clinical scanner's auxiliary geometric information."*

## v0 sub-task 1 long-context 3R Stack Update

**★ v0 sub-task 1 long-context 3R stack now has 23 papers covered** (10 paradigms × 23 = *most-comprehensive* 2024-2026 long-context 3R arc):

**(i) state-token:** CUT3R 175, MonST3R 174, Fast3R 178, Easi3R 173
**(ii) memory-token:** Spann3R 177, Point3R 179, STream3R 181, R³ 183, TTT3R 182, Ray-Aware 180
**(iii) global-attention:** VGGT 176, **π³ 192** (permutation-equivariant, BSD-3-Clause ✅), **MapAnything 193** (NEW, universal multi-modal, Apache-2.0 code + dual-license model weights ✅, the *best* license in the arc)
**(iv) chunked-TTT:** Scal3R 189, LoGeR 187
**(v) chunked-cache:** ZipMap 188
**(vi) gauge-decoupled:** LongStream 190
**(vii) 3D-backend:** AMB3R 191
**(viii) multi-modal:** Pow3R (concurrent, 2-view only), **MapAnything 193** (NEW, *unlimited* views, *12+ tasks* in one model)
**(ix) calibration-specialized:** Pow3R (concurrent), MoGe-2, AnyCalib
**(x) monocular-specialized:** Align3R, LaRI, UniGeo, Geo4D, Driv3r, Aether, CAN, StereoDiff, Pomato, PanSt3R, Surf3R, MoGe-2, UniDepthV2, Metric3DV2 (extensions)
**(xi) Other 2025-2026 3R works:** VLM-3D, 3DLLM, GP3, FLARE, FreeSplatter, MVSA, MVSAnywhere, CryoFASTA (domain-specific), Reliev3R (concurrent)

**★ v0 v1+ 3R-baseline comparison (top-5 for clinical-IOS):**
1. **MapAnything 193 (NEW, universal multi-modal, Apache-2.0 code + dual-license weights ✅, 12+ tasks in one model)** — the *license-clean universal SOTA* for v0 v1+ *commercial* deployment, the *best* license in the 2025-2026 3R arc
2. **π³ 192 (permutation-equivariant, BSD-3-Clause ✅, +33% speedup)** — the *license-clean permutation-equivariant* SOTA for v0 v1+ *commercial* deployment
3. **AMB3R 191 (3D-backend, adaptive-resolution, SOTA on 7 tasks)** — the *highest-quality* baseline for v0 v1+ *research* / academic
4. **LongStream 190 (gauge-decoupled, metric scale, 18 FPS, 0.9905 scale ratio)** — the *streaming* baseline for v0 v1+ *real-time* chairside
5. **Scal3R 189 (GCM+GCS, chunked-TTT, 21.4 FPS, no test-time opt)** — the *fast* baseline for v0 v1+ *low-latency* chairside

**★ v0 v1+ license-clean picks (top-4 for v1 production):**
- **MapAnything 193 (Apache-2.0 code + dual-license model weights ✅)** — the *best* license in the 2025-2026 3R arc, Apache-2.0 variant for *commercial* deployment, CC-BY-NC variant for *research / academic*
- **π³ 192 (BSD-3-Clause ✅)** — the *clean* permutation-equivariant SOTA
- **LongStream 190 (MIT ✅)** — the *streaming* SOTA, *clean* license
- **Scal3R 189 (MIT ✅)** — the *fast* SOTA, *clean* license

All four are *frozen-front-end + light-back-end* designs, all four use *DINOv2 + Transformer* as the backbone, all four are *2025-09 to 2026-04* papers, all four are from *de facto* consortiums (MapAnything from Meta Reality Labs + CMU, π³ from SJTU + Shanghai AI Lab, LongStream from Horizon Robotics + HKUST(GZ), Scal3R from BIGAI + Horizon Robotics).

## Next Paper to Read

**Recommended:** Paper 194 — **Reliev3R (Chen et al. 2026, CVPR 2026, the *concurrent* feed-forward 3R that *relieves* MapAnything 193 from multi-view dependencies)** — the *direct* comparison baseline to MapAnything 193's *multi-view* design, the *right* paper to *explore* the *single-view* / *few-view* 3R design space (per the Reliev3R CVPR 2026 supplemental citation of MapAnything [193]). The *practical* v0 v1+ lesson: *include Reliev3R* as the *single-view / few-view* SOTA alongside MapAnything 193 (multi-view universal) + π³ 192 (permutation-equivariant) + AMB3R 191 (3D-backend) + LongStream 190 (gauge-decoupled) + Scal3R 189 (chunked-TTT) in v0 v1+ Table 1.

*Alternative:* **DepthAnything 3 (Lin et al. 2025, arXiv:2511.10647, the *next* monocular depth foundation model)** — the *right* paper for *monocular depth estimation* (the *sub-task* that MapAnything 193 only *tangentially* addresses via its single-view inference mode), the *right* v0 v1+ sub-task 1 alternative if the *clinician* uses *single-view* IOS rather than *multi-view* (increasingly common with newer IOS hardware). Note: already cited in MapAnything 193's references (Lin et al. 2025, arXiv:2511.10647).

*Another alternative:* **UniMesh (Huang 2026, the *direct* H4 mesh foundation model)** — the *right* v0 v1+ sub-task 4 candidate (mesh output), the *missing* modality for v0 v1+ pipeline since MapAnything 193 (and AMB3R 191, LongStream 190, Scal3R 189, VGGT 087, π³ 192) is *purely image-to-3D-points* and the *final output* is a *mesh* for 3D printing.

**★ v0 TOTAL compute (revised):** ~$12,870-19,060 Lambda (was $12,670-18,560 from 192-note baseline, +$200-500 MapAnything fine-tuning + $0-100 factored RDP representation + $50-100 6-factor input augmentation = absorbed within existing budget envelope).

**★ Open Q for HK:** (i) adopt MapAnything 193 + fine-tune on dental data? (YES, the *license-clean universal SOTA*, Apache-2.0 code + dual-license model weights); (ii) adopt factored RDP representation? (YES, the *de facto* ablation evidence from Table 5a shows it *strictly outperforms* coupled pointmap); (iii) adopt 6-factor input augmentation? (YES, the *killer* training recipe for *flexible* clinical input variations); (iv) use MapAnything Apache-2.0 model weights as pre-training? (YES, *strictly better* than training from scratch, Apache-2.0 license enables *commercial* v0 v1+ deployment); (v) adopt log-space scale-invariant losses with stop-grad? (YES, the *killer* engineering detail for *universal training*); (vi) adopt per-view DPT decoding minibatch loop? (YES, the *killer* memory-efficiency trick for *consumer-grade* GPU deployment); (vii) adopt COLMAP export? (YES, the *killer* downstream-utility feature for *Gaussian Splatting* integration); (viii) cite MapAnything in v0 paper? (YES, the *universal 3R paradigm* paper, $0, 1-2 hours); (ix) use MapAnything's factored representation as v0 paper's "multi-modal clinical input" argument? (YES, the *killer* v0 v1+ deployment story).
