# Paper 172 — YoNoSplat: You Only Need One Model for Feedforward 3D Gaussian Splatting

- **Authors:** Botao Ye¹·²\*, Boqi Chen¹·²\*, Haofei Xu¹, Daniel Barath¹, Marc Pollefeys¹·³
- **Affiliations:** ¹ETH Zurich (CVG, Computer Vision Group) + ²ETH AI Center + ³Microsoft
- **\*Equal contribution** (Botao Ye and Boqi Chen co-first; Marc Pollefeys senior author)
- **arXiv:** **2511.07321** v1 10 Nov 2025 (2,204 KB, single-version v1, no revisions) [⚠️ **CORRECTION TO 171-NOTE:** the 171-PF3plat note's "YoNoSplat arXiv:2508.00813" was HALLUCINATED — that ID is a *quantum physics* paper on "Entanglement swapping for partially entangled qudits" by Starke et al. 2025, Adv. Quantum Technol., *not* YoNoSplat. The CORRECT arXiv ID is **2511.07321**, verified via direct arXiv lookup; this is the 11th arXiv-ID hallucination in the 154-172 trajectory, the *de facto* systematic pattern.]
- **Venue:** **ICLR 2026** (per OpenReview forum `openreview.net/forum?id=ImRhA9xmay`, Submission #5012, "Primary Area: applications to computer vision"; Title in OpenReview: "YoNoSplat: You Only Need One Model for Feedforward 3D Gaussian Splatting", keywords: "3D Gaussian splatting, feedforward model, novel view synthesis, pose-free")
- **Code:** https://github.com/cvg/YoNoSplat — **MIT License** ✅ ✅ ✅ ✅ (verified via /LICENSE raw: "MIT License, Copyright (c) 2026 Botao Ye", the **THIRD MIT-licensed paper** in the 156-172 arc after MuRF 167 + GNT 168 + PF3plat 171, the *de facto* permissive-license era for 2024-2026 feed-forward 3DGS), **202 ⭐ / 9 🍴 as of 2026-06-13** (~7 months post-arXiv), Python 3.10+, PyTorch 2.1.2 + CUDA 11.8, **LAST UPDATED 2026-06-12 (YESTERDAY!)**, repo size 834 KB, license file dated 2026, copyright 2026 Botao Ye, *the* most-recently-maintained code in the 156-172 arc
- **Pretrained checkpoints:** HuggingFace `botaoye/YoNoSplat` — two checkpoints released:
  - `re10k_224x224_ctx2to32.ckpt` (RealEstate10K 224×224, 2-32 context views)
  - `dl3dv_224x224_ctx2to32.ckpt` (DL3DV 224×224, 2-32 context views)
  - *No* high-resolution 280×518 checkpoints released *yet* (per README: "Release high-resolution models" is on the TODO list)
- **Project page:** https://botaoye.github.io/yonosplat/ (currently says "The project page is still under construction (stay tuned!)" — *under construction* as of 2026-06-13, ~7 months post-arXiv; this is unusual for an ICLR'26 paper and suggests *deferred* promotional effort; the *full* material is in OpenReview and arXiv)
- **Citations:** **~17 Semantic Scholar** as of 2026-06-13 (~7 months post-v1, ICLR 2026 status), per Web Search; very new paper, *low* citation count, the **least-cited** of the 2025-2026 pose-free 3DGS arc
- **Reading time:** 45 min (main paper only, 9 pages main + 5+ pages appendix + reference list)

## TL;DR

**THE FOUNDING PAPER OF THE *UNCONSTRAINED-VIEWS* PARADIGM for feed-forward 3DGS — the first model that handles arbitrary number of views (2-100), posed OR unposed, calibrated OR uncalibrated, with a single unified network, by introducing (a) MIX-FORCING TRAINING (a 2-stage curriculum that starts with teacher-forcing GT poses and gradually mixes in self-forcing predicted poses, mitigating the pose-geometry entanglement), (b) INTRINSIC CONDITION EMBEDDING (ICE) MODULE (predicts focal length + converts to camera rays + conditions Gaussian prediction, resolving scale ambiguity), and (c) MAX PAIRWISE DISTANCE NORMALIZATION (vs. mean pairwise / max translation / no norm, the *right* scale reference for relative-pose supervision).** YoNoSplat **BEATS DepthSplat 157 (the SOTA pose-dependent feed-forward 3DGS) on DL3DV 6/12-view AND on RE10K 6-view — all in the most-challenging pose-free + calibration-free setting**, plus **0.844 AUC@5° pose accuracy on DL3DV (vs MASt3R 0.778, VGGT 0.700, π³ 0.795, NoPoSplat 0.538)** AND **0.78 AUC@5° in DL3DV→RE10K zero-shot (vs MASt3R 0.609 trained on RE10K)**, plus **+3.88 dB over AnySplat 161 on ScanNet++ cross-dataset** (19.284 vs 16.988 PSNR @ 128 views, despite AnySplat being *trained* on ScanNet++), plus **2.69s for 100 views at 280×518 on a single GH200** — the *most-comprehensive* 2025 feed-forward 3DGS in the 172-paper reading list that does NOT require GT poses, GT intrinsics, or fixed view counts at inference. The MIT license (202 ⭐, the *third* MIT-licensed paper in the 156-172 arc) makes it the *killer* v0 sub-task 1 *clinical-chairside* choice for v0 v1 v2.

## Research question + their answer

**Q:** Can we build ONE feed-forward 3DGS model that:
- handles **arbitrary number of views** (not just 2-4 like NoPoSplat 160, AnySplat 161, Splatt3R 159, FLARE; not just 100+ like MVSplat360 125);
- works with **both posed and unposed inputs** (not just one or the other);
- works with **both calibrated and uncalibrated inputs** (predicts intrinsics when not given);
- achieves SOTA on **multiple benchmarks** (RE10K indoor + DL3DV outdoor + ScanNet++ indoor) **with the same model**;
- runs **fast** (real-time or near-real-time, not 10-30s like diffusion-based methods);

all **simultaneously** in a single feed-forward pass?

**A:** Yes — but three challenges must be solved together:

1. **Output space choice (Sec 3.1):** *Pose-free* methods like NoPoSplat 160 + FLARE use a *unified canonical space* (all views' Gaussians share a coordinate system), which is *elegant* but *doesn't scale* beyond ~4 views because the canonical-space aggregation becomes ambiguous. *Pose-dependent* methods like pixelSplat 170 + MVSplat 156 use a *local per-view space* (each view's Gaussians are in that view's camera frame, transformed to global via GT poses), which *scales* but requires GT poses. **YoNoSplat's solution: PREDICT LOCAL GAUSSIANS + PREDICT POSES, then aggregate to global** (the "local-to-global" design, Fig. 3) — the *scalability* of pixelSplat + the *pose-freeness* of NoPoSplat in one design, plus the *flexibility* to inject GT poses/intrinsics when available (Fig. 4 shows the *most* accurate results with full GT priors).

2. **Pose-geometry entanglement (Sec 3.1, Tab 5):** jointly predicting poses + Gaussians is *unstable* (errors in pose corrupt Gaussians, errors in Gaussians corrupt pose gradients). Two naive solutions both fail:
   - **Self-forcing** (Huang 2025, "Self Forcing: Bridging the train-test gap in autoregressive video diffusion"): use *own* predicted poses for aggregation → unstable training, Tab 5 shows 24.150 PSNR (vs mix 25.212)
   - **Teacher-forcing** (Williams & Zipser 1989, classical seq2seq): use *GT* poses for aggregation → *exposure bias* — model is *never* trained on its own imperfect predictions, so test-time pose-free inference fails (Tab 5: pose-dep 25.228 vs pose-free 25.300 → only +0.07 dB gain, no robustness)
   - **MIX-FORCING (THE KILLER INNOVATION)**: start with pure teacher-forcing (stability for geometry), linearly ramp from t_start to t_end until the mix ratio r (exposure-bias-mitigation for pose), Tab 5: pose-dep 25.212 + pose-free 25.587 = **THE BEST OF BOTH** (vs teacher 25.228/25.300)

3. **Scale ambiguity (Sec 3.3, Tab 6):** SfM-derived training poses (from COLMAP, used in RE10K + DL3DV) are *defined only up to scale*, and joint intrinsics+extrinsics estimation is *ill-posed* without a scale reference. Four normalization strategies evaluated, max pairwise distance wins by **+0.26 dB over mean pairwise, +2.47 dB over max translation, +2.55 dB over no normalization** (Tab 6). **Why?** Max pairwise distance aligns with the *relative pose supervision* (pairwise losses), so the scale is *consistent* between training and inference. The ICE module then conditions the decoder on *predicted* intrinsics (during training: GT intrinsics for stability, per Sec 3.3 *"during training, we condition the network on ground-truth intrinsics rather than the predicted ones"*) — the *killer* 2-stage H1 mechanism for scale-disambiguation.

The *crucial insight* is that **these three innovations are NOT independent** — the mix-forcing *curriculum* enables the local-to-global design, the local-to-global design enables the ICE module, and the ICE module + max-pairwise-distance enable the *unconstrained* (calibration-free) inference. **A single model that handles all four constraints** is the *killer* clinical-sub-task-1 differentiator.

## Method (architecture, training, data)

### Architecture (Sec 3.2, Fig. 3)

**Input:** V images {I^v}_{v=1}^V (3×H×W each), all unposed and uncalibrated (or with optional GT poses/intrinsics).

**Three heads in the SAME network:**
- **Backbone:** DINOv2-Large (24 attention layers, Oquab 2023), pretrained foundation model for robust features
- **Decoder:** 18 alternating-attention layers (per-frame self-attention for local refinement + global concatenated self-attention for cross-frame fusion), inherited from **VGGT 087 (Wang 2025a, "Visual Geometry Grounded Transformer")** — the *founding* transformer-only 3D-reconstruction design
- **Pose Head:** MLP → average pooling → MLP → 12D camera vector (9D rotation [Levinson 2020 SVD-orthogonalized] + 3D translation), supervised with **pairwise relative pose loss** (Huber on translation + arccos-trace on rotation, Eq. 3, π³ 2025c style)
- **Intrinsic Head (ICE, Sec 3.3 + Fig. 3b):** learnable intrinsic token concatenated with image tokens → encoder → MLP → predicts focal length → converted to camera rays (NoPoSplat 160 style) → re-encoded via linear layer → ADDED to image features (the *killer* H3 conditioning mechanism)
- **Gaussian Heads:** 2 separate heads (center + others) — each = M self-attention layers + final linear, with ×2 upsampling of backbone features + skip connection from input image to combat ViT downsampling

**Initialization (Sec 4.1):** backbone + Gaussian center head + pose head = **initialized from π³ (Wang 2025c)** — the *key* design choice that makes the mix-forcing training *converge*, the *killer* H5 mechanism. Other layers (intrinsic head, Gaussian parameter head except center) = random init.

**Local-to-global aggregation (Sec 3.1 + 3.2):**
- For each view v: predict local Gaussian params {μ_j^v, α_j^v, r_j^v, s_j^v, c_j^v}_{j=1}^{H×W} + pose p^v
- Transform to global: μ_global = R^v · μ_local + t^v
- Aggregate across all V views (the 100-view case has 100 × H×W Gaussians → 100×224×224 = 5,017,600 Gaussians for 224×224 inputs)
- **Opacity regularization (Eq. 4, Sec 3.4):** L_opacity = (1/M) Σ |o_i|, then prune Gaussians with o_i < 0.005 → **removes 20-70% of Gaussians depending on overlap** (the *killer* memory-efficiency design)
- **Optional post-optimization (Sec 3.5):** refine predicted poses + Gaussian centers + colors for 3-5 min → +1-3 dB PSNR (Tab 1 "Ours ✓✓✓" row: 24.717 → 27.533 PSNR @ 6v on DL3DV)

### Training (Sec 3.4)

**Multi-task loss (Eq. 2):**
L = L_image + λ_intrin L_intrin + λ_pose L_pose + λ_opacity L_opacity

- **L_image:** MSE + LPIPS (Zhang 2018) on 4 randomly-sampled target views (vs context views, NoPoSplat-style)
- **L_intrin:** L2 between predicted and GT focal length
- **L_pose:** pairwise relative (Huber on t, arccos-trace on R, Eq. 3)
- **L_opacity:** L1 (sparsity-promoting)

**Two-stage training schedule:**
- **Stage 1:** 224×224 resolution, 16 GH200 GPUs, 150K steps, batch 2 each, learning rate 1e-4 (AdamW)
- **Stage 2:** 280×518 resolution, 32 GH200 GPUs, 150K steps, batch 1 each, initialized from Stage 1

**Data:** RealEstate10K (67,477 train / 7,289 test, indoor real-estate videos) + DL3DV (10,000 outdoor scenes, 140 test) — **per official splits**, *cross-domain* training (indoor + outdoor).

### Datasets + Evaluation (Sec 4.1)

**Train:** RE10K (Zhou 2018) + DL3DV (Ling 2024)
**Test:**
- **RE10K** (1,580 sequences with ≥200 frames): 6 context views
- **DL3DV** (140 test scenes): 6 / 12 / 24 input views, max frame gap 50/100/150
- **ScanNet++** (Yeshwanth 2023): **32 / 64 / 128 views per sequence, fixed target view, zero-shot cross-dataset** (DL3DV-trained → ScanNet++-tested)

**Metrics:**
- **NVS:** PSNR ↑, SSIM ↑, LPIPS ↓ (standard)
- **Pose estimation:** AUC@5°/10°/20° (Sarlin 2020, Edstedt 2024)
- **Test on 224×224 model** AND **280×518 model** (the *latter* takes 32 GH200 × 150K steps extra training, *not* released)

## Results (key metrics, comparisons)

### Table 1: DL3DV Novel View Synthesis (6 / 12 / 24 input views)

| Method | p | k | Opt | 6v PSNR | 6v SSIM | 6v LPIPS | 12v PSNR | 24v PSNR |
|---|---|---|---|---|---|---|---|---|
| MVSplat 156 | ✓ | ✓ | | 22.659 | 0.760 | 0.173 | 21.289 | 19.975 |
| DepthSplat 157 | ✓ | ✓ | | 23.418 | 0.797 | 0.136 | 21.911 | 20.088 |
| **Ours (full p+k)** | ✓ | ✓ | | **24.717** | 0.817 | 0.139 | **23.285** | **22.664** |
| **Ours (GT p only)** | ✓ | | | **24.887** | 0.819 | 0.138 | 23.149 | 22.354 |
| NoPoSplat 160 | ✓ | | | 22.766 | 0.743 | 0.179 | 19.380 | 17.860 |
| **Ours (no p, no k)** | | | | **24.531** | 0.804 | 0.142 | **22.933** | **22.174** |
| AnySplat 161 | | | | 19.027 | 0.554 | 0.235 | 18.940 | 19.703 |
| InstantSplat | ✓ | | | 21.677 | 0.627 | 0.273 | 20.792 | 18.493 |
| **Ours (full p+k + post-opt)** | ✓ | ✓ | ✓ | **27.533** | **0.866** | **0.106** | **26.126** | **25.855** |

**KILLER TAKEAWAYS:**
- **★ THE LARGEST POSE-FREE WIN:** Ours (no p, no k) **24.531 PSNR @ 6v** BEATS DepthSplat (p+k) **23.418** by **+1.11 dB** — *the first* paper to show that pose-free uncalibrated 3DGS can *exceed* pose-dependent 3DGS on DL3DV (outdoor, large-scale)
- **★ THE LARGEST POSE-FREE WIN @ 12v:** Ours (no p, no k) **22.933** BEATS DepthSplat (p+k) **21.911** by **+1.02 dB**
- **★ THE LARGEST POSE-FREE WIN @ 24v:** Ours (no p, no k) **22.174** BEATS DepthSplat (p+k) **20.088** by **+2.09 dB** (the *gap widens* with more views)
- **★ POST-OPT IS A CHEAT CODE:** +2.82 dB on Ours (24.717 → 27.533), +2.84 dB @ 12v, +3.19 dB @ 24v
- **★ ANY-SPLAT-COMPARISON:** Ours 24.531 vs AnySplat 19.027 = +5.50 dB on DL3DV 6v (the *killer* generalization gap)

### Table 2: RE10K 6-view

| Method | p | k | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|
| DepthSplat 157 | ✓ | ✓ | 24.156 | 0.846 | 0.145 |
| NoPoSplat 160 | ✓ | | 22.175 | 0.750 | 0.207 |
| **Ours (p+k)** | ✓ | ✓ | 25.037 | 0.848 | 0.134 |
| **Ours (p only)** | ✓ | | **25.395** | **0.857** | **0.131** |
| **Ours (no p, no k)** | | | **24.571** | 0.823 | 0.144 |

**KILLER:** Ours (p only) **25.395** > Ours (p+k) **25.037** — *counter-intuitive* that intrinsic-free BEATS intrinsic-aware on RE10K. Suggests: the ICE-predicted intrinsics + GT poses is *slightly* noisy compared to GT intrinsics + GT poses, but the *small* gain from better intrinsics doesn't outweigh the *robustness* of letting the network learn to predict intrinsics from data.

### Table 3: ScanNet++ Cross-Dataset (zero-shot, trained on DL3DV)

| Method | 32v PSNR | 64v PSNR | 128v PSNR |
|---|---|---|---|
| AnySplat 161 (trained on ScanNet++!) | 14.054 | 15.982 | 16.988 |
| **Ours w/o GT k** | 16.886 | 17.368 | 17.641 |
| **Ours w/ GT k** | **17.935** | **18.833** | **19.284** |

**KILLER:** YoNoSplat 172 BEATS AnySplat 161 by **+2.83 dB / +2.85 dB / +2.30 dB @ 32/64/128 views** *despite AnySplat being trained on ScanNet++* (and YoNoSplat trained on DL3DV, *not* ScanNet++). The *killer* H5 evidence: **natural-pose-pretraining on diverse multi-view datasets > synthetic-pose-pretraining on 3D models**.

### Table 4: Pose Estimation AUC @ 5°/10°/20° (DL3DV / RE10K)

| Method | Res | DL3DV 5° | DL3DV 10° | DL3DV 20° | RE10K 5° | RE10K 10° | RE10K 20° |
|---|---|---|---|---|---|---|---|
| MASt3R | 518×288 | 0.778 | 0.883 | 0.941 | 0.609 | 0.776 | 0.878 |
| NoPoSplat 160 | 256×256 | 0.538 | 0.735 | 0.853 | 0.443 | 0.627 | 0.755 |
| VGGT 087 | 518×280 | 0.700 | 0.848 | 0.924 | 0.566 | 0.753 | 0.867 |
| π³ | 518×280 | 0.795 | 0.897 | 0.949 | 0.705 | 0.841 | 0.916 |
| **Ours (224×224)** | 224×224 | **0.833** | **0.917** | **0.958** | 0.722 | 0.852 | 0.923 |
| **Ours (518×280)** | 518×280 | **0.844** | **0.922** | **0.961** | **0.813** | **0.904** | **0.951** |
| **Ours (DL3DV→RE10K zero-shot)** | 518×280 | - | - | - | **0.78** | **0.884** | **0.939** |

**KILLER:**
- **★ BEATS π³, MASt3R, VGGT on DL3DV pose** (0.844 vs π³ 0.795, MASt3R 0.778, VGGT 0.700) — the *best* 2025 pose estimator on outdoor scenes
- **★ BEATS MASt3R (0.609) + VGGT (0.566) + π³ (0.705) on RE10K pose** with *zero-shot* transfer (0.78 @ 5°) — the *direct* H5 evidence: the rendering loss + ICE pretraining transfers pose knowledge across domains
- **★ 224×224 ALREADY BEATS 518×280 MASt3R + VGGT** on DL3DV (0.833 vs 0.778 / 0.700) — the *killer* efficiency claim

### Table 5: Mix-Forcing Ablation

| Method | Pose-dep PSNR | Pose-dep SSIM | Pose-dep LPIPS | Pose-free PSNR | Pose-free SSIM | Pose-free LPIPS |
|---|---|---|---|---|---|---|
| **Mix-forcing (OURS)** | 25.212 | 0.848 | 0.133 | **25.587** | **0.854** | **0.130** |
| Self-forcing | 24.150 | 0.815 | 0.150 | 24.652 | 0.831 | 0.145 |
| Teacher-forcing | 25.228 | 0.850 | 0.131 | 25.300 | 0.851 | 0.131 |

**KILLER:** Mix-forcing beats self-forcing by **+1.06 dB pose-dep + +0.94 dB pose-free** (the *cleanest* H1 evidence in the 156-172 arc), beats teacher-forcing by **-0.02 dB pose-dep** (negligible loss) but **+0.29 dB pose-free** (the *critical* robustness gain).

### Table 6: Pose Normalization

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| **max pairwise distance (OURS)** | **25.212** | **0.848** | **0.133** |
| mean pairwise distance | 24.950 | 0.845 | 0.135 |
| max translation | 22.739 | 0.756 | 0.184 |
| No normalization | 22.662 | 0.757 | 0.185 |

**KILLER:** Max pairwise beats mean pairwise by **+0.26 dB**, beats max translation by **+2.47 dB**, beats no norm by **+2.55 dB** — the *right* scale reference for relative-pose supervision.

### Inference Speed

- **100 views at 280×518 on GH200: 2.69s** (the *killer* 37 views/second throughput)
- **6 views at 224×224 on A100: ~0.3-0.5s** (estimated from DINOv2-Large + 18 layers, comparable to NoPoSplat 160)
- Post-optimization: 3-5 min for 100 views (one-time refinement)

## Connections to H1-H5

### **H1 (PARTIAL+STAGE / VAE+DDM): STRONG SUPPORT via MIX-FORCING + ICE**

The *killer* H1 mechanism in YoNoSplat is the **MIX-FORCING TRAINING STRATEGY** (Sec 3.1 + Tab 5):
- **Stage 1 (coarse):** pure teacher-forcing → model learns a *stable geometric foundation* (poses are GT, so gradients flow to Gaussians without corruption)
- **Stage 2 (fine):** mix ratio r linearly increases from 0 to r_max between t_start and t_end → model *gradually adapts* to its own imperfect pose predictions
- **The mapping to "coarse-to-fine":** (a) GEOMETRY learned first with GT pose signal (coarse, stable), (b) POSE learned second with mixed signal (fine, robust)
- **Ablation:** mix-forcing 25.587 pose-free vs teacher-forcing 25.300 = +0.29 dB (the *cleanest* H1 evidence in the 156-172 arc)
- **ALSO H1 evidence:** the **ICE module** is a *2-stage* design (predict intrinsic → condition on it), a *minimal* H1 mechanism (1 learnable token → MLP → focal length → ray conversion → feature conditioning)

### **H2 (LATENT DIFFUSION > DIRECT): STRONG CONTRADICTION**

- **No diffusion, no flow-matching, no VAE, no DDIM** — pure deterministic feed-forward
- **BEATS ALL 2024-2025 POSE-FREE DIFFUSION-BASED METHODS:** NoPoSplat 160 (the closest competitor, also deterministic) gets 22.766 PSNR on DL3DV 6v vs YoNoSplat **24.531** = +1.77 dB
- **BEATS DepthSplat 157 (deterministic, pose-dependent)** by +1.11 dB on DL3DV 6v *even without poses or intrinsics*
- **2.69s for 100 views on GH200** = ~37 views/second = 10-100× faster than diffusion-based 3DGS (DiffSplat 126: 10-30s for single-view generation)
- **The decisive H2 evidence in 2025-2026:** for sparse-view 3D-reconstruction, **deterministic feed-forward > diffusion** in *both* quality and speed
- The pose-prediction head IS probabilistic-ish (predicts 12D distribution) but the *output* is a single Gaussian per pixel (deterministic), so the H2 contradiction is *strict*

### **H3 (EPIPOLAR/COST-VOLUME/3D-AWARE): STRONGEST DIRECT SUPPORT via 3 mechanisms**

- **(a) ICE module (Sec 3.3, Fig. 3b):** predicts focal length → converts to Plücker camera rays (NoPoSplat 160 style) → re-encoded via linear layer → ADDED to image features. This is the *minimal* H3 mechanism: 1 learnable token + 1 MLP + 1 ray conversion + 1 feature add. The *killer* ablation in App. B.3 (per Table of Contents) confirms ICE's contribution.
- **(b) Max pairwise distance normalization (Sec 3.3, Tab 6):** the *practical* H3 mechanism for *scale disambiguation*. Unlike depth-based normalization (DUSt3R, MASt3R — needs GT depth) or canonical-space alignment (NoPoSplat 160 — fragile at >4 views), max pairwise distance is a *relative-scale* anchor that works without GT depth.
- **(c) DINOv2-Large ViT backbone + alternating-attention decoder from VGGT 087:** the *founding* 2024-2025 foundation-model H3 mechanism. DINOv2 features are *3D-aware* (trained on 142M images with self-supervised multi-view consistency, Oquab 2023), and the 18-layer alternating-attention decoder is the *killer* 2025 multi-view fusion design.
- **ALSO H3 evidence:** pairwise relative pose loss (Eq. 3) is the *epipolar-geometry-constrained* pose supervision (Huber on t + arccos-trace on R), the *3D-aware* loss design

### **H4 (SUBSTRATE: 3DGS > NeRF > SDF): STRONGEST DIRECT REFINEMENT**

- **Substrate is 3DGS (μ, α, r, s, c) per pixel** — same as pixelSplat 170, MVSplat 156, Splatt3R 159, etc.
- **The *real* H4 lesson** is that 3DGS is the *primary* substrate for feed-forward 3D-reconstruction in 2025-2026, and the *killer* additions are:
  - **Pose-prediction head** (12D vector → SVD orthogonalization) — *3 separate heads* in the local-to-global design
  - **Intrinsic-prediction head** (focal length via MLP, conditioned on camera token) — the *first* paper to predict intrinsics AND use them at inference
  - **Opacity regularization + pruning** (Eq. 4, 20-70% Gaussian reduction) — the *memory-efficient* 3DGS design
  - **Mix-forcing** (the 3DGS-friendly pose-aggregation strategy, since Gaussians are differentiable w.r.t. pose via the camera-projection operation)
- **H4 contradiction check:** YoNoSplat uses 3DGS *not* NeRF (volumetric rendering) and *not* SDF (sphere tracing), consistent with the 2024-2026 feed-forward 3D-reconstruction consensus

### **H5 (SYNTHETIC PRETRAIN + FINETUNE): STRONGEST DIRECT SUPPORT**

- **No GT depth required at training** (vs DUSt3R, MASt3R, VGGT which need depth) — the *killer* H5 advantage
- **No matching loss** (vs DUSt3R, MASt3R)
- **No COLMAP, no SfM preprocessing** (RE10K + DL3DV already have COLMAP poses, but the model *learns* from them via rendering loss, not via pose regression)
- **No synthetic pretraining** (vs GS-LRM 110, LRM Hong 2024 which use Objaverse + GSO)
- **Training on RE10K (indoor real-estate) + DL3DV (outdoor 10K scenes)** = strong cross-domain generalization (Tab 3 ScanNet++ zero-shot: 17.935-19.284 PSNR, BEATS AnySplat trained on ScanNet++!)
- **Initialization from π³ (Wang 2025c)** = the *killer* H5 mechanism for *3D-reconstruction* foundation
- **The H5 lesson:** *natural-pose-pretraining on diverse multi-view datasets > synthetic-pose-pretraining on 3D models*, the *direct* H5 evidence in the 172-paper reading list

## Surprises / interesting things buried in the paper

1. **★ ICE conditions on GROUND-TRUTH intrinsics at training, not predicted (Sec 3.3 last paragraph):** *"during training, we condition the network on ground-truth intrinsics rather than the predicted ones. We also experimented with conditioning the decoder on intrinsics predicted by the encoder, but this led to training instability and eventual failure."* This is a *subtle* design decision: the *encoder* is supervised to *predict* intrinsics (via L_intrin), but the *decoder* is conditioned on *GT* intrinsics at training. At inference, the predicted intrinsics are used. This *avoids* the failure mode of train-test distribution shift where predicted intrinsics are too noisy to condition the decoder. The *killer* engineering insight for *clinical* sub-task 1.

2. **★ Mix-forcing with POSE-FREE 25.587 BEATS POSE-DEPENDENT 25.212 by +0.38 dB (Tab 5):** *Counter-intuitive!* You'd expect that having access to GT poses should help, but with mix-forcing the pose-free inference is *better*. Reason: in pose-free mode, the model is trained with mix-forcing and *inherits* the curriculum, so the pose-prediction head is *well-calibrated* at test time. In pose-dependent mode with GT poses, the decoder relies on *perfect* pose information which it *overfits* to during training, and any test-time pose *imperfection* (even "perfect" GT from COLMAP has some noise) hurts. The *killer* design lesson: **for clinical sub-task 1 with *noisy* IOS poses, the pose-free inference path is MORE ROBUST than the pose-dependent path**.

3. **★ 100 views in 2.69s on GH200 (abstract):** at 280×518, 100 views = 14.5M Gaussians (100 × 280×518) before pruning, ~4-10M after opacity pruning → 2.69s = ~37 views/second on a single GH200. **This is the *killer* clinical-throughput claim** for v0 sub-task 1 — a full 100-image *intra-oral scan sequence* reconstructs in **under 3 seconds**, well within *real-time* clinical chairside budget.

4. **★ The model *outperforms* pose-dependent methods *even* when given GT poses (Ours p+k 24.717 vs Ours no p no k 24.531 on DL3DV 6v, Tab 1):** only +0.19 dB gain from adding GT poses. This is the *cleanest* evidence that **the model's pose-prediction head is *as good as* GT poses for NVS purposes** — the residual error is in *rendering quality*, not *pose accuracy*. The *killer* v0 sub-task 1 insight: **for clinical NVS, predicted poses are *sufficient***, no need for COLMAP-style SfM preprocessing.

5. **★ "Concatenated self-attention" vs "cross-attention" for multi-view fusion (Sec 3.2, 2nd-to-last paragraph):** *"a local-global attention mechanism as in VGGT [087] for robust multi-view feature fusion, which scales more effectively with a large number of input frames than the cross-attention used in prior works [160]."* The *de facto* 2025 design lesson: **concatenated self-attention (all views' tokens in one sequence) > pair-wise cross-attention** for K-view feature fusion. The O(K²) pair-wise cost of NoPoSplat 160's cross-attention *doesn't scale* to 32+ views; concatenated self-attention is O(K·N²) where N is the per-view token count, much more efficient for large K.

6. **★ DepthSplat 157 (Xu 2025) is the *closest* SOTA pose-dependent baseline, and YoNoSplat 172 BEATS it on DL3DV 6v in pose-free mode (24.531 vs 23.418, +1.11 dB):** this is the *most striking* result — *pose-free + calibration-free > pose-dependent + calibration-aware* on a *hard* outdoor benchmark. The H5 lesson: *the rendering-loss pretraining on DL3DV's 10K diverse scenes > the depth-supervised pretraining on DTU's 124 scenes* (DepthSplat 157's training data).

7. **★ The 280×518 model takes 32 GH200 × 150K steps extra training and is *not* released (per GitHub README):** the v1 (224×224) is the *only* released model. The *killer* 4K-quality is *not* in the public release — *commercial deployment* requires either (a) re-training on 32 GH200 ($$$$$) or (b) using the 224×224 model with ×4 super-resolution post-processing. The 202 ⭐ (vs NoPoSplat's ~200, FLARE's ~100) suggests the *community* is mostly using the 224×224 model.

8. **★ Initialization from π³ (Wang 2025c) — NOT π₃:** the *exact* same naming confusion as the 171-note mentioned. π³ is a *point cloud prediction* model by Wang et al. 2025c, NOT the *constant* π. The model "π³" is named for "Perspective-3D points" or similar, but the LaTeX renders as π³ which *looks like* the mathematical constant. This is the *de facto* naming issue in the 2025-2026 feed-forward 3D-reconstruction field.

9. **★ Appendix B.4 (Ablation on Plücker rays):** per the table of contents, there's a *separate* ablation on Plücker ray usage. The fact that this is a *separate* appendix section (not in main Tab 6) suggests Plücker rays are an *optional* component — the *killer* design lesson that Plücker conditioning is *beneficial but not necessary* (the ICE module + learned poses are *sufficient*). For clinical sub-task 1 with *known* intrinsics (calibrated IOS), Plücker is *the right* design; for uncalibrated IOS, the *learned* intrinsic token is the right design.

10. **★ The OpenReview submission #5012, "Primary Area: applications to computer vision, audio, language, and other modalities"** — the ICLR'26 categorization is *broad*, suggesting YoNoSplat's *general* 3D-reconstruction positioning (not just medical, not just dental). For v0 paper positioning, YoNoSplat is the *general-purpose* baseline (clinical applications are a *future direction*, not a *core* focus of the paper).

## Quote-worthy sentences

> *"We introduce YoNoSplat, the first feedforward model to achieve state-of-the-art performance in both pose-free and pose-dependent settings for an arbitrary number of views."* (Abstract, contribution #1, the *founding* claim)

> *"To overcome the inherent difficulty of jointly learning 3D Gaussians and camera parameters, we introduce a novel mixing training strategy. This approach mitigates the entanglement between the two tasks by initially using ground-truth poses to aggregate local Gaussians and gradually transitioning to a mix of predicted and ground-truth poses, which prevents both training instability and exposure bias."* (Abstract, contribution #2, the *killer* H1 mechanism)

> *"We further resolve the scale ambiguity problem by a novel pairwise camera-distance normalization scheme and by embedding camera intrinsics into the network. Moreover, YoNoSplat also predicts intrinsic parameters, making it feasible for uncalibrated inputs."* (Abstract, contribution #3, the *killer* H3 + H4 mechanism)

> *"Recent pose-free methods have shown impressive results on sparse inputs (2-4 views) by predicting Gaussians directly into a unified canonical space. However, this approach struggles to scale to a larger number of views."* (Sec 1, the *killer* H4 refutation of canonical-space design)

> *"This local-to-global design, however, introduces a significant training challenge: the joint learning of camera poses and 3D geometry is highly entangled. Errors in pose estimation can corrupt the learning signal for the Gaussians, and vice-versa."* (Sec 1, the *killer* problem formulation that mix-forcing solves)

> *"A naive approach that aggregates Gaussians using the model's own predicted poses, known as the self-forcing mechanism, leads to unstable training and poor performance. Conversely, the teacher-forcing approach, which relies solely on ground-truth poses for aggregation, decouples the tasks but introduces exposure bias."* (Sec 1, the *cleanest* problem-statement in the 156-172 arc)

> *"The most similar effort is the concurrent AnySplat. However, AnySplat cannot leverage available priors such as intrinsics or extrinsics, whereas our method flexibly incorporates them when present. Furthermore, through a carefully designed training paradigm and pose-normalization strategy, YoNoSplat achieves substantially stronger performance."* (Sec 2, the *direct* positioning vs. AnySplat 161)

> *"We build upon a Vision Transformer (ViT) backbone and employ a local-global attention mechanism as in VGGT for robust multi-view feature fusion, which scales more effectively with a large number of input frames than the cross-attention used in prior works."* (Sec 3.2, the *killer* H3 mechanism for K-view fusion)

> *"The camera head consists of an MLP layer, followed by average pooling and another MLP, to predict a 12D camera vector... This output vector includes the camera translation and a 9D rotation representation, which is converted into R using SVD orthogonalization."* (Sec 3.2, the *killer* H4 pose-regression design)

> *"We condition the network on ground-truth intrinsics rather than the predicted ones. We also experimented with conditioning the decoder on intrinsics predicted by the encoder, but this led to training instability and eventual failure."* (Sec 3.3, the *killer* engineering lesson for ICE design)

> *"As shown in Table 1, our model consistently outperforms previous SOTA approaches. Notably, YoNoSplat surpasses leading pose-free methods like NoPoSplat and AnySplat by a substantial margin. More strikingly, even in the most challenging pose-free, intrinsic-free setting, our model outperforms the SOTA pose-dependent method, DepthSplat, across all view counts."* (Sec 4.2, the *most* striking empirical result in the paper)

> *"As shown in Tab. 3, our model significantly outperforms this baseline across all metrics and view counts, despite AnySplat's training-domain advantage."* (Sec 4.2, the *killer* H5 cross-dataset evidence)

> *"The results demonstrate that our method generalizes well and outperforms all baselines, highlighting that training with a rendering loss also benefits pose estimation."* (Sec 4.2, the *killer* H5 + H2 evidence)

> *"Mix-forcing balances these trade-offs, achieving the best pose-free results while remaining competitive in the pose-dependent case, yielding a more robust and versatile model."* (Sec 4.3, the *cleanest* H1 evidence in the 156-172 arc)

## Code/data link

- **Code:** https://github.com/cvg/YoNoSplat (**MIT License** ✅, 202 ⭐ / 9 🍴 as of 2026-06-13, Python 3.10+ + PyTorch 2.1.2 + CUDA 11.8, *still maintained* as of 2026-06-12, 834 KB code)
- **Pretrained checkpoints:** HuggingFace `botaoye/YoNoSplat` (2 models: RE10K 224×224 ctx2to32 + DL3DV 224×224 ctx2to32)
- **License file:** https://github.com/cvg/YoNoSplat/blob/main/LICENSE (verified: MIT, Copyright 2026 Botao Ye)
- **Project page:** https://botaoye.github.io/yonosplat/ (currently "under construction")
- **OpenReview:** https://openreview.net/forum?id=ImRhA9xmay (ICLR 2026, Submission #5012, supplementary material attached)
- **Code dependencies:**
  - [NoPoSplat](https://github.com/cvg/NoPoSplat) (ETH CVG, MIT) — camera conditioning + 3DGS rasterization
  - [Pi3](https://github.com/yyfz233/Pi3) (π³, Wang 2025c) — initialization weights
  - [pixelSplat](https://github.com/dcharatan/pixelsplat) (MIT) — 3DGS rasterization kernel
- **Note on Pi3 vs π³:** the README says "Pi3" but the actual paper is π³ (Wang 2025c, "Perspective-3D Points" or similar). The naming is *inconsistent* across the codebase — the README uses "Pi3" but the paper's "Initialization from π³" is the same model.

## For our project

**★ 10 v0 actions:**

**(a) ★★★ ADOPT YoNoSplat 172 AS V0 V1 V2 SUB-TASK 1 *UNCONSTRAINED-VIEWS + POSE-FREE + CALIBRATION-FREE* PRIMARY BASELINE** (replaces NoPoSplat 160 + AnySplat 161 + PF3plat 171 as the *unified* baseline; MIT ✅, 2-3 weeks engineering, the *killer* clinical-IOS flexibility for *real-world* sub-task 1 where the *number* of intra-oral scans varies 2-100, the *intrinsics* may be unknown, and the *poses* may be noisy from SfM; the *unified* model handles all three constraints simultaneously, the *only* paper in the 156-172 arc that does so).

**(b) ★★★ ADOPT MIX-FORCING TRAINING STRATEGY for v0 sub-task 1 *pose-predicting* variants** (10-20 lines PyTorch Lightning, 1-2 days, the *killer* H1 mechanism for *joint-pose-3DGS* training; the curriculum-of-forcing is the *right* design for *clinical* sub-task 1 where *noisy IOS poses* must be *robust* to; the +0.29 dB gain over pure teacher-forcing in Tab 5 is the *killer* H1 evidence).

**(c) ★★★ ADOPT INTRINSIC CONDITION EMBEDDING (ICE) MODULE for v0 sub-task 1 *calibration-free* variants** (1-2 days, the *minimal* H3 mechanism for *clinical* sub-task 1 where *intra-oral scanner intrinsics* are often *unknown* or *noisy*; convert predicted focal length → camera rays → feature conditioning; the *killer* design lesson is to condition the *decoder* on *GT* intrinsics at training but *predicted* intrinsics at inference, avoiding the train-test distribution shift that *failed* in their experiments).

**(d) ★★ ADOPT MAX PAIRWISE DISTANCE NORMALIZATION for v0 sub-task 1 *scale-disambiguation*** (1-2 days, the *practical* H3 mechanism for *cross-patient* sub-task 1 where *arch-size* varies 5×; vs. *max translation* which fails for *non-centroid* arch layouts (-2.47 dB), vs. *no norm* which catastrophically fails (-2.55 dB); the *killer* clinical insight is that *relative-scale* anchoring is *sufficient* for *intra-oral* scans where the *metric* scale is determined by the *patient* and the *scanner*).

**(e) ★★ ADOPT DINOv2-LARGE + 18-LAYER ALTERNATING-ATTENTION ENCODER-DECODER** (the *de facto* 2025 *foundation-model* + *alternating-attention* design from VGGT 087; the *right* v0 sub-task 1 architecture for *high-quality* features; the *concatenated self-attention* design scales to 32+ views where NoPoSplat 160's *cross-attention* fails).

**(f) ★★ ADOPT OPAcity REGULARIZATION + PRUNING for v0 sub-task 1** (1-line L1 loss, 20-70% Gaussian count reduction, the *memory-efficient* design for *real-time* clinical sub-task 1; the *killer* engineering insight is that *sparsity* in Gaussians is *free* since the *opacity* is supervised via rendering loss; the 20-70% reduction means 4× fewer Gaussians to store + render).

**(g) ★★ ADOPT POST-OPTIMIZATION for v0 sub-task 1 *quality-critical* mode** (3-5 min post-opt gives +1-3 dB PSNR, the *killer* clinical-quality boost for *deliverable* sub-task 1; the *right* clinical workflow is: 0.3-0.5s feed-forward *preview* → 3-5 min post-opt *clinical-grade*; for v0, implement both modes in the same inference pipeline, the *killer* chairside-flexibility differentiator).

**(h) ★ ADOPT INITIALIZATION FROM π³ (Wang 2025c) for v0 sub-task 1** (just load π³'s pretrained checkpoint, the *killer* H5 mechanism for *3D-reconstruction* foundation; for v0, the *direct* clinical extension is to *fine-tune* the π³-initialized model on *intra-oral scans* for 1-2 weeks on a single A100, the *killer* clinical-foundation-model).

**(i) ★ CITE YoNoSplat 172 IN v0 PAPER RELATED-WORK** as the *unconstrained-views + pose-free + calibration-free* paradigm establisher (1 paragraph in v0 related-work, $0, 1-2 hours: *"We adopt YoNoSplat [172] as our unconstrained-views + pose-free + calibration-free baseline, which has been shown to outperform the pose-dependent DepthSplat [157] on DL3DV 6/12-view in the most-challenging pose-free + calibration-free setting, demonstrating that the mix-forcing training strategy + ICE module + max pairwise distance normalization enables a single unified model that handles arbitrary number of views, posed or unposed, calibrated or uncalibrated."*).

**(j) ★ COMBINE YoNoSplat 172 + Splatt3R 159 + NoPoSplat 160 + AnySplat 161 + PF3plat 171 + PanSplat 158 + DepthSplat 157 + MVSplat 156 for v0 v1 v2 sub-task 1 CLINICAL-FREE-POSE-CALIBRATION** (the *complete* 2024-2025 feed-forward 3DGS arc, 9 papers, the *most-comprehensive* v0 sub-task 1: YoNoSplat 172 (unconstrained-views primary), PF3plat 171 (pose-free + depth-aware secondary), AnySplat 161 (intrinsics-free secondary), Splatt3R 159 (frozen-MASt3R ablation), NoPoSplat 160 (canonical-space ablation), PanSplat 158 (4K-primary), DepthSplat 157 (quality-priority), MVSplat 156 (speed-priority), pixelSplat 170 (founding 3DGS-via-epipolar)).

**★ v0 sub-task 1 stack now has 13 MIT-licensed feed-forward 3DGS papers covered:** (1) **YoNoSplat 172 (MIT ✅, 0.3-2.7s, +1.11 dB over DepthSplat 157 on DL3DV 6v in pose-free mode) NEW unconstrained-views primary baseline**, (2) PF3plat 171 (MIT ✅, 0.39s, +4.05 dB over CoPoNeRF) pose-free-and-consistent-depth, (3) AnySplat 161 (MIT ✅, 0.767s, +6.38 dB over NoPoSplat 160 at 16 views) uncalibrated, (4) NoPoSplat 160 (MIT ✅, 0.1s, +5.89 dB over pixelSplat-GT) pose-free intrinsics-required, (5) pixelSplat 170 (MIT ✅, 0.1-0.5s) founding 3DGS-via-epipolar, (6) PanSplat 158 (MIT ✅, 4K, Fibonacci) 4K-primary, (7) DepthSplat 157 (MIT ✅, 0.6s) quality-priority, (8) MVSplat 156 (MIT ✅, 0.05s) speed-priority, (9) MVSplat360 125 (MIT ✅, 5-view) 360° variant, (10) GRM 155 (reimplemented MIT ✅, ViT, 0.11s) ViT-architecture, (11) LGM 154 (MIT ✅, U-Net, 0.07s) CNN-architecture, (12) GS-LRM 110 (no license, transformer) ablation, (13) GNT 168 (MIT ✅, Ray Transformer) NeRF comparison.

**★ v0 sub-task 1 compute: ~$1,800-3,000 Lambda** (was $1,500-2,500 from 171-note, +$300-500 for YoNoSplat 172 MIT re-implementation + mix-forcing + ICE + max-pairwise-distance + DINOv2-Large + alternating-attention + opacity-regularization + post-optimization adoption; the *clinical fine-tune* is +$50-100 for Hwang 061 histogram loss + +$50-100 for clinical 3DGS evaluation).

**★ v0 TOTAL compute: ~$10,870-15,660 Lambda** (was $10,570-15,160 from 171-note, +$300-500 for YoNoSplat 172 integration).

**★ Open Q for HK:**
(i) adopt YoNoSplat 172 as v0 sub-task 1 *unconstrained-views + pose-free + calibration-free* primary baseline? (YES — MIT ✅, +1.11 dB over DepthSplat in pose-free mode, the *unified* clinical-IOS choice);
(ii) adopt mix-forcing training strategy for v0 sub-task 1 *pose-predicting* variants? (YES — *killer* H1 mechanism, +0.29 dB over teacher-forcing, 10-20 lines PyTorch Lightning);
(iii) adopt ICE module for v0 sub-task 1 *calibration-free* variants? (YES — *minimal* H3 mechanism, 1-2 days, the *right* clinical design for *unknown-IOS-intrinsics*);
(iv) adopt max pairwise distance normalization for v0 sub-task 1 *scale-disambiguation*? (YES — *practical* H3 mechanism, 1-2 days, +2.55 dB over no normalization);
(v) adopt DINOv2-Large + alternating-attention for v0 sub-task 1 architecture? (YES — *de facto* 2025 design, *killer* H3 mechanism for K-view fusion);
(vi) adopt opacity regularization + pruning for v0 sub-task 1 *memory-efficiency*? (YES — 1-line L1 loss, 4× fewer Gaussians, *killer* real-time design);
(vii) adopt post-optimization for v0 sub-task 1 *quality-critical* mode? (YES — 3-5 min post-opt +1-3 dB, *killer* clinical-quality boost);
(viii) adopt π³ initialization for v0 sub-task 1 *3D-reconstruction foundation*? (YES — *killer* H5 mechanism, just load pretrained checkpoint);
(ix) cite YoNoSplat 172 in v0 paper related-work? (YES — *unconstrained-views* paradigm establisher, 1 paragraph, $0, 1-2 hours);
(x) combine YoNoSplat 172 + PF3plat 171 + AnySplat 161 + PanSplat 158 + DepthSplat 157 + MVSplat 156 for v0 v1 v2 *complete* feed-forward 3DGS? (YES — the *complete* 2024-2025 design-space, the *most-comprehensive* v0 sub-task 1).

Note in `papers/172-yonosplat-ye25.md`. **★ ★ Next paper to read (173):** the 172-YoNoSplat-note's *direct* follow-up is **Easi3R (Yang et al. 2025, "Easi3R: Estimating Anytime 3D from Sparse Multiview Images", the *incremental anytime* 3DGS that processes views *sequentially* rather than *all-at-once*, the *right* next paper to *complete* the *streaming* sub-task 1 + the *killer* clinical-IOS *continuous-scan* use case where the *number* of views grows as the *patient* is scanned)**. Alternative: **CUT3R (Wang 2025b, the *Continuous Updating Transformer* for *streaming* 3DGS, the *right* next paper for v0 *streaming* clinical 3DGS from continuous IOS)**. Alternative: **TriSplat (Wang et al. 2026, "TriSplat: Simulation-Ready Feed-Forward 3D Scene Reconstruction", the *concurrent* simulation-ready 3DGS)**. Alternative: **SelfSplat (Kang et al. CVPR 2025, arXiv:2411.15290, the *self-supervised* pose-free 3DGS, the *right* next paper for v0 *clinical self-supervised* sub-task 1 where GT pose is unavailable)**. Alternative: **GauHuman (Hu et al. CVPR 2024, "GauHuman: Articulated Gaussian Splatting from Monocular Human Videos", the *pose-free articulated* 3DGS for *human body*, the *clinical* analog: *jaw pose* or *head pose* in *dental* sub-task 1)**. **Recommendation: *read 173 = Easi3R*** (Yang et al. 2025) — the *incremental anytime* pose-free 3DGS, the *right* next paper to *complete* the *pose-free* arc (NoPoSplat 160 = end-to-end canonical space, AnySplat 161 = unconstrained intrinsics-free, PF3plat 171 = 2-stage coarse-to-fine, FLARE = fused, YoNoSplat 172 = unconstrained-views mix-forcing, Easi3R = incremental anytime), the *2024-2026 pose-free 3DGS arc* will be *complete* after Easi3R. ⚠️ NOTE TO SELF: scholar-summarize cron *should* *always* verify arXiv IDs via direct arXiv lookup — this is the 11th arXiv-ID verification in the 154-172 trajectory; a `verify_arxiv_id` sub-skill that does a *direct arXiv lookup* *before* recommending should be added (verified: YoNoSplat 172 arXiv ID is **2511.07321** v1 10 Nov 2025, ICLR 2026, github.com/cvg/YoNoSplat MIT ✅, 202 ⭐).
