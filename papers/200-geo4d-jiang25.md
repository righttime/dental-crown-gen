# Paper 200 — Geo4D (Zeren Jiang et al., 2025)

⚠️ **arXiv ID correction from paper 199's recommendation:** the 199-note recommended arXiv:2509.19213 (hallucinated). The actual arXiv ID is **arXiv:2504.07961** (v1 10 Apr 2025, v2 19 Aug 2025). This is the **13th arXiv-ID hallucination** prevented in the 156-200 arc by direct arXiv lookup. The 199-note's "Geo4D (Jiang 2025, arXiv:2509.19213)" was wrong on the arXiv ID — the *authors and topic* were correct, the *arXiv ID* was invented.

## TL;DR

**Geo4D repurposes a *pre-trained video generator* (DynamiCrafter) as a *monocular 4D dynamic-scene reconstructor* by predicting THREE redundant-yet-complementary geometric modalities simultaneously (DUSt3R-style viewpoint-invariant point maps + disparity maps for dynamic range + Plücker ray maps defined for all pixels incl. sky), aligning them via a *group-wise post-processing optimization* that extends DUSt3R's pairwise alignment to overlapping video clips, with sliding-window inference (V=16 frames, stride s=4, DDIM 5 steps) — trained on *5 synthetic datasets only* (Spring + BEDLAM + PointOdyssey + TartanAir + VirtualKITTI) and generalizing zero-shot to in-the-wild real videos** — achieving **SOTA on Sintel/Bonn/KITTI video depth (AbsRel 0.205 vs MonST3R 0.335 = -39%) + best camera-rotation estimation (RPE-R 0.547 Sintel vs MonST3R 0.732 = -25%; TUM-Dyn 0.635 vs 1.217 = -48%)** but **worse translation estimation (ATE 0.185 vs 0.108 Sintel)** at **1.27× MonST3R's speed**, ICCV 2025 **Highlight**, code on GitHub ⚠️ **NO LICENSE** (GitHub API `license: null`), 434 ⭐, last push 2025-06-06 (1 year before our read, *not* actively maintained). The **founding paper of the "3-modality-prediction + multi-modal-alignment + synthetic-only-training" paradigm** for video-3D, and the *direct competitor* to Aether 199 (paper 199) in the 4D-vision-via-video-diffusion space — Geo4D wins on depth + camera-rotation; Aether wins on unified prediction + planning.

## Research Question

**Question:** Recent dynamic 4D reconstruction methods (MonST3R, Easi3R) extend DUSt3R to dynamic scenes, but (a) they require *significant amounts of training data with 3D annotations for dynamic supervision*, (b) this data is *difficult to collect for dynamic scenes* (especially in-the-wild), and (c) DUSt3R-based methods use only *point maps* — a representation with *limited dynamic range* (not defined for points at infinity like the sky) and *suboptimal for fast motion / camera motion*. **Can we repurpose a *pre-trained video generator* (a "foundation" video diffusion model with strong dynamic-priors for camera motion + object motion + perspective) as a monocular 4D reconstructor — predicting multiple redundant-yet-complementary geometric modalities and fusing them via test-time optimization — to achieve SOTA 4D reconstruction with *synthetic-only* training data?**

## Method

### Architecture: DynamiCrafter fine-tuned for 3-modality prediction

- **Backbone:** **DynamiCrafter** (a "foundation" latent video diffusion model, similar to SVD / AnimateDiff), a 2-stage VAE+U-Net architecture
- **Output:** **3 modalities per pixel per frame**:
  1. **Point map** `X^i ∈ ℝ^{H×W×3}` (DUSt3R-style viewpoint-invariant, expressed in the *first-frame* reference frame, encodes camera motion + intrinsics + scene depth)
  2. **Disparity map** `D^i ∈ ℝ^{H×W×1}` (better dynamic range than point maps — disparity=0 for points at infinity like the sky)
  3. **Ray map** `r^i ∈ ℝ^{H×W×6}` (**Plücker coordinates**: `r=(d, m)` where `d = R^T K^{-1}(u,v,1)^T` is the ray direction and `m = -R^T t × d` is the moment; defined for *all* image pixels regardless of scene geometry)
- **No camera parameters as input** — implicitly estimated by the model

### 3.1 Video Diffusion Backbone (DynamiCrafter)

- Latent diffusion: `z_0 = E(x)` (VAE encode), `z_t = √α̅_t · z_0 + √(1-α̅_t) · ε` (forward diffusion), denoising U-Net predicts noise
- DDIM sampling with **5 steps** (not 50, not 100 — they find 4D-reconstruction is "more deterministic" than video generation, Tab. 6)

### 3.2 Multi-modal Geometric 4D Diffusion (the **killer** section)

- **VAE fine-tuning for point maps (Eq. 3, the killer insight):** the *pre-trained image encoder-decoder* is repurposed for disparity and ray maps without modification, but the **point map decoder is fine-tuned** with an **uncertainty-weighted L1 loss**:
  ```
  L = -Σ_{u,v} ln(1/(√2 σ_{u,v})) · exp(-√2 · |D(E(X))_{u,v} - X_{u,v}| / σ_{u,v})
  ```
  where `σ ∈ ℝ^{H×W}` is the **uncertainty map** predicted by an additional branch of the VAE decoder. **The encoder is unchanged** to modify the latent space as little as possible; instead, point maps are *normalized to [-1, 1]* to match the pre-trained image encoder's input range.
- **Hybrid video conditioning (Sec 3.2):**
  - *Local stream:* VAE encoder features are **channel-concatenated** to the noised latents (the standard LVDM conditioning)
  - *Global stream:* each frame is passed through **CLIP** + a **lightweight learnable query transformer** (1-layer cross-attention) → vectors injected into the U-Net via **cross-attention in each block** (similar to IP-Adapter)

### 3.3 Multi-modal Alignment (the **second** killer section)

**Problem:** the 3 modalities are *non-independent*, and *processing all frames of a long video simultaneously* is computationally prohibitive. So:
1. **Temporal sliding window** splits the video into overlapping clips of `V=16` frames with stride `s=4` (default)
2. For each clip, predict (X^{i,g}, D^{i,g}, r^{i,g}) via the diffusion model
3. **Group-wise alignment optimization** fuses all clips + all modalities into a globally coherent 4D reconstruction

**Alignment losses (4 terms, Eq. 4-10):**
- **L_p (point map alignment, Eq. 4):** `Σ_{g,i,u,v} ||X^i - λ_p^g · P_p^g · X^{i,g}_{u,v}||_1 / σ^{i,g}_{u,v}` — extends DUSt3R's pairwise to **group-wise** with **uncertainty weighting**, recovers (K, R, o, D_p, λ_p^g, P_p^g)
- **L_d (disparity alignment, Eq. 5):** `Σ_{g,i} ||D_p^i - λ_d^g · D_d^{i,g} - β_d^g||_1` — aligns pointmap-derived disparity with model-predicted disparity via *scale* λ_d^g + *shift* β_d^g
- **L_c (camera trajectory alignment, Eq. 6-8):** extracts (R_c, o_c) from ray maps via Plücker-coordinate optimization (Ray Diffusion style), then aligns with (R_p, o_p) from point maps
- **L_s (camera trajectory smoothness, Eq. 9):** `Σ_i (||R_p^i^T R_p^{i+1} - I||_F + ||o_p^{i+1} - o_p^i||_2)` — MonST3R's smoothness prior
- **Final loss (Eq. 10):** `L_all = α_1 L_p + α_2 L_d + α_3 L_c + α_4 L_s` with weights (1, 2, 0.005, 0.015) — disparity is the *strongest* signal (α_2 = 2x), camera-trajectory alignment is *weakest* (α_3 = 0.005x)

**Optimization algorithm (Supp. Mat Alg. 1):**
1. Predict all 3 modalities for each clip
2. Initialize K via minimization of pointmap projection error in first frame
3. Initialize R, o via RANSAC PnP
4. Initialize R_c, o_c from ray maps via Eqs. 6-7
5. Alternate between: solving L_d, then solving L_c, then jointly minimizing L_p + L_s, then jointly minimizing L_all
6. Iterate until convergence

## Results

### Video Depth Estimation (Tab. 1, **SOTA on all 3 benchmarks**)

| Method | Sintel AbsRel ↓ | Sintel δ_1.25 ↑ | Bonn AbsRel ↓ | Bonn δ_1.25 ↑ | KITTI AbsRel ↓ | KITTI δ_1.25 ↑ |
|---|---|---|---|---|---|---|
| **Marigold** (single-frame) | 0.532 | 51.5 | 0.091 | 93.1 | 0.149 | 79.6 |
| **Depth-Anything-V2** (single-frame) | 0.367 | 55.4 | 0.106 | 92.1 | 0.140 | 80.4 |
| **NVDS** (video) | 0.408 | 48.3 | 0.167 | 76.6 | 0.253 | 58.8 |
| **ChronoDepth** (video) | 0.687 | 48.6 | 0.100 | 91.1 | 0.167 | 75.9 |
| **DepthCrafter** (video, concurrent) | 0.270 | 69.7 | 0.071 | 97.2 | 0.104 | 89.6 |
| **MonST3R** (joint depth+pose) | 0.335 | 58.5 | 0.063 | 96.4 | 0.104 | 89.5 |
| **Geo4D (Ours)** | **0.205** | **73.5** | **0.059** | **97.2** | **0.086** | **93.7** |

**Improvement vs prior SOTA:**
- vs **MonST3R** (joint depth+pose SOTA): Sintel AbsRel **-39%** (0.335→0.205), δ_1.25 **+15pts** (58.5→73.5); Bonn AbsRel -6% (0.063→0.059); KITTI AbsRel -17% (0.104→0.086), δ_1.25 +4.7pts (89.5→93.7)
- vs **DepthCrafter** (concurrent video-depth): Sintel AbsRel -24% (0.270→0.205); despite Geo4D solving a *more general* problem (4D not just depth)

### Camera Pose Estimation (Tab. 2, **best rotation, comparable translation**)

| Method | Sintel ATE ↓ | Sintel RPE-T ↓ | Sintel RPE-R ↓ | TUM-Dyn ATE ↓ | TUM-Dyn RPE-T ↓ | TUM-Dyn RPE-R ↓ |
|---|---|---|---|---|---|---|
| Robust-CVD | 0.360 | 0.154 | 3.443 | 0.153 | 0.026 | 3.528 |
| CasualSAM | 0.141 | 0.035 | 0.615 | 0.071 | 0.010 | 1.712 |
| **MonST3R** | **0.108** | 0.042 | 0.732 | **0.063** | **0.009** | 1.217 |
| **Geo4D (Ours)** | 0.185 | 0.063 | **0.547** | 0.073 | 0.020 | **0.635** |

**Key finding:** Geo4D achieves **best camera-rotation** (RPE-R 0.547 Sintel = -25% vs MonST3R 0.732; TUM-Dyn 0.635 = -48% vs MonST3R 1.217) but **worse translation** (ATE 0.185 Sintel = +71% vs MonST3R 0.108; TUM-Dyn 0.073 = +16%). **The first method to use a generative model for camera-pose estimation** ("To the best of our knowledge, Geo4D is the first method that uses a generative model to estimate camera parameters in a dynamic scene" — Tab. 2 caption).

### Ablation: Multi-modal Representation (Tab. 3, **both training supervision AND inference alignment matter**)

| Train PointMap | Train Disp | Train Ray | Infer PointMap | Infer Disp | Infer Ray | AbsRel ↓ | δ_1.25 ↑ | ATE ↓ | RPE-T ↓ | RPE-R ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| ✓ | — | — | ✓ | — | — | 0.232 | 71.3 | 0.335 | 0.076 | 0.731 |
| ✓ | ✓ | ✓ | ✓ | — | — | 0.223 | 72.5 | 0.237 | 0.070 | 0.566 |
| ✓ | ✓ | ✓ | — | ✓ | — | 0.211 | 73.4 | — | — | — |
| ✓ | ✓ | ✓ | — | — | ✓ | — | — | 0.268 | 0.192 | 1.476 |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **0.205** | **73.5** | **0.185** | **0.063** | **0.547** |

**Two findings:**
- **Multi-modal training supervision matters** (rows 1 vs 2): adding disparity + ray map *training* improves AbsRel from 0.232→0.223 (-4%) and RPE-R from 0.731→0.566 (-23%) even when inference uses only pointmap — the modalities serve as **additional supervisory signals**
- **Multi-modal inference alignment matters MORE** (rows 2-4 vs 5): the full multi-modal-inference wins AbsRel 0.205 vs 0.223 single-best (-8%), and the *ray-only inference* RPE-R is *catastrophic* at 1.476 (2.7× worse than full)

### Ablation: Sliding Window Stride (Tab. 4, **s=4 best tradeoff**)

| Stride s | FPS ↑ | AbsRel ↓ | δ_1.25 ↑ | ATE ↓ | RPE-T ↓ | RPE-R ↓ |
|---|---|---|---|---|---|---|
| 15 | 0.92 | 0.213 | 72.4 | 0.210 | 0.092 | 0.574 |
| 8 | 1.24 | 0.212 | 72.8 | 0.222 | 0.074 | 0.524 |
| **4** | **1.89** | **0.205** | **73.5** | **0.185** | **0.063** | **0.547** |
| 2 | 3.26 | 0.204 | 72.9 | 0.181 | 0.058 | 0.518 |

**Trade-off:** shorter stride → marginal +0.001 AbsRel improvement but 1.7× slower (s=2 vs s=4). They choose **s=4 for main results**.

### Ablation: DDIM Sampling Steps (Supp. Tab. 6, **5 steps optimal**)

| DDIM steps | AbsRel ↓ | δ_1.25 ↑ | ATE ↓ | RPE-T ↓ | RPE-R ↓ |
|---|---|---|---|---|---|
| 1 | 0.221 | 70.7 | 0.234 | 0.072 | 0.753 |
| **5** | **0.205** | **73.5** | **0.185** | **0.063** | **0.547** |
| 10 | 0.207 | 73.2 | 0.212 | 0.071 | 0.508 |
| 25 | 0.220 | 72.2 | 0.211 | 0.074 | 0.564 |

**Critical insight (4D-specific):** "Compared to the video generation task, where a larger number of denoising steps usually produces a more detailed generated video, **4D reconstruction is a more deterministic task, which requires fewer steps**." This *contradicts* the standard diffusion wisdom (more steps = better) and is *consistent* with R³ 183's "learned confidence > probabilistic" finding.

### Ablation: Fine-tuned Point Map VAE (Supp. Tab. 7, **+3.5% AbsRel**)

| PointMap VAE fine-tuned? | AbsRel ↓ | δ_1.25 ↑ | ATE ↓ | RPE-T ↓ | RPE-R ↓ |
|---|---|---|---|---|---|
| No (repurpose as-is) | 0.212 | 72.1 | 0.192 | 0.061 | 0.577 |
| **Yes (uncertainty-weighted L1)** | **0.205** | **73.5** | **0.185** | **0.063** | **0.547** |

Fine-tuning the pointmap decoder with the uncertainty-weighted L1 loss (Eq. 3) gives -3.3% AbsRel and -5.2% RPE-R.

### Efficiency (Sec 4.5)

- **Geo4D: 1.89 FPS at s=4** (s=2: 3.26 FPS, s=8: 1.24 FPS, s=15: 0.92 FPS)
- **MonST3R: 0.41 FPS (2.41s/frame)** — Geo4D is **1.27× faster** than MonST3R (the direct competitor)
- **DDIM 5 steps** (not 50) is the efficiency trick — 10× faster than standard 50-step diffusion
- 4× H100 training, ~1 week total

## Training Details (Supp. Sec 6)

- **5 synthetic datasets, ZERO real data (the killer H5 lesson):**
  - **Spring** (Mehl 2023): 6K frames, 37 sequences, 16.7% ratio — high-resolution outdoor
  - **BEDLAM** (Black 2023): 380K frames, 10K sequences, 33.3% ratio — bodies, indoor/outdoor (largest share)
  - **PointOdyssey** (Zheng 2023): 200K frames, 131 sequences, 16.7% ratio — indoors/outdoors, long-term point tracking (max-pool inpainting for missing depth)
  - **TartanAir** (Wulff 2022): 1000K frames, 163 sequences, 16.7% ratio — indoors/outdoors
  - **VirtualKITTI** (Cabon 2020): 43K frames, 320 sequences, 16.7% ratio — driving
- **16 frames per sample** with random stride from {1, 2, 3} for frame-rate adaptation
- **Progressive training curriculum:**
  1. Train point maps only at fixed 512×320
  2. Add multi-resolution training (512×384, 512×320, 576×256, 640×192)
  3. Progressively add ray + depth modalities
- **AdamW, lr=1e-5, batch=32, 4×H100, ~1 week** (the **cheapest** 4D training in the reading list)
- **Initialized from DynamiCrafter** weights (the *killer* transfer-learning move)
- Alignment optimization: `(α_1, α_2, α_3, α_4) = (1, 2, 0.005, 0.015)` to roughly equalize loss scales (disparity is the strongest signal)

## Connections to Hypotheses (H1-H5)

### H1: 2-stage (VAE encoder + diffusion decoder) > 1-stage feed-forward for dental crown generation
**STRONG SUPPORT (3-stage design).** Geo4D is a 3-stage design: (1) frozen VAE from DynamiCrafter, (2) trainable U-Net denoising, (3) **post-processing group-wise alignment optimization**. The H1 lesson for v0: **post-processing alignment is a *third stage* that matters** — the 4.5% AbsRel improvement from full multi-modal-inference (Tab. 3 row 5 vs row 2) is a 1.4× improvement for free. For v0 sub-task 1, add a third stage of *clinical-fit-aware* alignment optimization on top of v0's 2-stage backbone.

### H2: Latent diffusion > direct deterministic prediction for 3D shape generation
**STRONG SUPPORT (with caveat).** The entire paper is built on latent diffusion from a video generator, and the synthetic-only training is the *killer* demonstration that H2's "strong prior" claim is *correct* for natural-video priors (motion, perspective, object shape). **Caveat:** DDIM 5 steps is *optimal* (Tab. 6) — 4D reconstruction is "more deterministic than video generation" — so H2 is *less* needed for the 4D output than for the video input. For v0 sub-task 1: **pre-train on synthetic IOS data with latent video diffusion, but use *few* denoising steps at inference** (the 5-step trick saves 10× inference cost).

### H3: Strong conditioning of input → output = better
**STRONGEST DIRECT SUPPORT (the founding paper of "multi-modality alignment as H3 mechanism").** The *entire* contribution (ii) is "suggest generating multiple partially redundant geometric modalities and fusing them at test time via lightweight optimization." The 3 modalities (point + depth + ray) are *complementary* (different dynamic range, different scene-geometry coverage, different sensitivity), and the **group-wise alignment optimization** is the *killer* H3 design — Tab. 3 shows multi-modal inference improves AbsRel from 0.223 (pointmap-only inference) to 0.205 (full) = -8% for free, and RPE-R from 0.566 to 0.547 = -3% (the second-best uses only disparity training supervision). For v0 sub-task 1: **predict 3 complementary modalities (point + depth + ray) and align them in post-processing** — the H3 paradigm is *the* contribution of Geo4D.

### H4: Implicit (SDF) > mesh/pointmap for 3D shape representation
**STRONG CONTRADICTION (for full-arch 4D) / NEUTRAL (for closed crown).** Geo4D explicitly adopts DUSt3R's *pointmap* representation (with Plücker ray maps) — no SDF, no mesh, no implicit. The 4D scene is represented as `(X, D, r)` per pixel per frame. H4 is *refuted* for the v0 sub-task 1 (full-arch 4D) case, but *supported* for the v0 sub-task 2 (closed crown) case via DMC 033's SAP + FlexiCubes 007 indicator grid. For v0: keep H4 only for sub-task 2.

### H5: Synthetic pre-training + small real-data fine-tuning > large real-data from scratch
**STRONGEST DIRECT SUPPORT (founding paper of "synthetic-only + pretrained video model = strong 4D generalization").** Trained on 5 *synthetic* datasets only (Spring + BEDLAM + PointOdyssey + TartanAir + VirtualKITTI, ZERO real data) and generalizes to in-the-wild real videos (Sintel synthetic + Bonn real indoor + KITTI real driving + TUM-Dynamics real). The pretrained video model's "strong dynamic prior" (motion, perspective) is the transfer mechanism — *not* real-data fine-tuning. **The killer H5 lesson for v0 sub-task 1:** **pre-train on synthetic IOS data + 3DTeethSeg22 (the *dental equivalent of synthetic Spring/BEDLAM*) + ToSynFCD, with the *transfer* mechanism being a pre-trained video model on natural video + a pre-trained dental foundation model on dental data, not real clinical IOS data collection.** This is the H5 paradigm-shifter for v0.

## Surprises / Interesting Things Buried in Section 4 + Supp

1. **5-step DDIM is optimal for 4D reconstruction (Supp. Tab. 6)** — a 1-step baseline achieves 0.221 AbsRel (only -7% worse than 5-step) and 10/25 steps are *worse* than 5 steps. **The 4D task is more deterministic than video generation**, contradicting standard diffusion wisdom. *For v0 sub-task 1, this is the killer efficiency insight — use few denoising steps, not many.*
2. **The point map decoder needs fine-tuning with uncertainty-weighted L1 (Eq. 3, Supp. Tab. 7)** — repurposing the pre-trained VAE as-is *fails* for point maps (only -0.7% AbsRel) because the pre-trained image distribution is fundamentally different from the 3D-coordinate distribution. The uncertainty branch + weighted L1 is the *killer* design pattern for adapting VAEs to non-natural-image modalities. *For v0 sub-task 1, this is the killer pattern for adapting Wan2.1/SVD VAE to depth/disparity/ray maps.*
3. **The ray map is the worst single-modality inference (RPE-R 1.476 in Tab. 3 row 4) but the *best* single-modality for *training* supervision** — ray maps carry camera-rotation info, so they help the model learn 3D but don't directly align 3D structure. **The lesson for v0: use multiple modalities as *training* supervision (regardless of which is used for *inference*) — the modalities are *complementary training signals* even when they're not the best inference output.**
4. **Synthetic-only training generalizes to real** (Sec 4.1 + Sec 4.2): KITTI is *real driving*, Bonn is *real indoor* — and Geo4D beats *real-data-trained* MonST3R. This is the *strongest* evidence in our reading list that **pretrained video models + synthetic data = real-data generalization** for 4D.
5. **Group-wise alignment vs pairwise (MonST3R):** Geo4D's group-wise alignment is the *key* qualitative improvement — Fig. 3 shows "Geo4D successfully tracks the racing car in 4D, whereas MonST3R struggles due to the rapid motion between pairs of images" — the *group-wise* design is robust to fast motion between pairs but not between groups.
6. **Geo4D wins camera *rotation* but loses camera *translation*** (Tab. 2) — translation needs *more accurate depth* which is harder for video-diffusion (translation is integrated over time, errors compound). For v0 sub-task 1, this is the *killer* design lesson: **predict 3D scene + camera rotation first, refine camera translation separately** (or use depth-priors for translation).
7. **Geo4D is 1.27× faster than MonST3R** (1.89 FPS vs 0.41 FPS at the same setting) — the DDIM 5-step trick is the *efficiency* secret.
8. **No license (GitHub API `license: null`)** — the code is *released* (gdown checkpoint links work) but *not commercial-deployable*. The same restriction as WinT3R 185. The Apache-2.0/MIT-licensed Aether 199 remains the *commercial-deployable* alternative.
9. **Code last pushed 2025-06-06 (1 year before our read)** — *not* actively maintained. The paper is *completed* but the codebase is in a "frozen" state. For v0, expect to *fork and modernize* (PyTorch 2.x, Pytorch3D install, etc.).
10. **The OptimiCrafter / 0.5-second Stride Ablation** (Tab. 4): even s=15 (low-overlap clips) gives 0.213 AbsRel — the model is *robust* to low-overlap sliding windows, suggesting the *alignment optimization* (not the overlap) is the key.

## Quote-Worthy Sentences

- **Intro:** *"We show that a pre-trained off-the-shelf video generator can be turned into an effective monocular feed-forward 4D reconstructor."* — the founding claim of the "video diffusion for 3D" paradigm
- **Sec 3.2:** *"These modalities are redundant in principle, but complementary in practice."* — the *H3 lesson* in one sentence
- **Sec 3.2 (uncertainty-weighted L1):** *"The encoder is unchanged to modify the latent space as little as possible; instead, we normalize the point maps to the range [-1, 1] to make them more compatible with the pre-trained image encoder."* — the *killer* insight for repurposing pretrained VAEs to non-natural-image modalities
- **Sec 3.3 (multi-modal alignment):** *"Furthermore, processing all frames of a long monocular video simultaneously with a video diffusion model is computationally prohibitive. Therefore, during inference, we use a temporal sliding window that segments the video into multiple overlapping clips, with partial overlap to facilitate joining them."* — the *killer* design pattern for long-video 3D
- **Sec 4.2:** *"despite solving a more general problem"* (4D vs depth-only) — the *founding modesty* that makes Geo4D's SOTA more impressive
- **Tab. 2 caption:** *"To the best of our knowledge, Geo4D is the first method that uses a generative model to estimate camera parameters in a dynamic scene."* — the *founding claim* of generative camera-pose estimation
- **Sec 4.5:** *"Note that MonST3R requires 2.41 seconds to process one frame under the same setting, so our method is 1.27 times faster than MonST3R."* — the *efficiency* narrative
- **Supp. Tab. 6:** *"4D reconstruction is a more deterministic task, which requires fewer steps. Similar phenomena are also observed in [DepthCrafter], which uses a video generator for video depth estimation."* — the *killer* 4D-specific efficiency insight
- **Sec 9 Limitations:** *"our approach can struggle in cases involving significant changes in focal length or extreme camera motion throughout a sequence. This limitation likely stems from the lack of focal length variation in our training data."* — the *killer* design lesson: include focal-length variation in training data for v0
- **Sec 9 Limitations:** *"due to the inherent temporal attention mechanism in our network architecture, our approach currently supports only monocular video input. Extending the method to handle multi-view images or videos is a promising direction for future work."* — the *killer* future direction: multi-view 3D-from-video (directly applicable to v0 sub-task 1's multi-buccal-view IOS)

## Code / Data Links

- **Code:** https://github.com/jzr99/Geo4D — 434 ⭐ / 16 🍴 / 210.6 MB / ⚠️ **NO LICENSE** (GitHub API `license: null`) / last push 2025-06-06 / main branch
  - Submodules: `dust3r/` (DUSt3R fork), `lvdm/` (Latent Video Diffusion Model from DynamiCrafter), `main/`, `configs/inference_geo4d.yaml`, `scripts/infer_geo4d.sh`, `scripts/eval_geo4d.sh`
  - Checkpoints: gdown links (not GitHub Releases, but Google Drive) — fine-tuned VAE + whole model
  - Visualization: `viser/visualizer.py`
- **Project page:** https://geo4d.github.io
- **Paper PDF:** https://arxiv.org/pdf/2504.07961 (v2)
- **OpenAccess (ICCV 2025):** https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_Geo4D_Leveraging_Video_Generators_for_Geometric_4D_Scene_Reconstruction_ICCV_2025_paper.pdf
- **YouTube video:** https://youtu.be/HHQG26mZicE
- **Dependencies:** python 3.8.5, `pip install -r requirements.txt` + Pytorch3D (the *C++/CUDA*-heavy one, not the wheel)

## For Our Project

### ★ v0 Sub-Task 1 (Full-Arch 3D Reconstruction) Impact

Geo4D is **partially applicable** to v0 sub-task 1 because:
- ❌ Geo4D is a *video* model (16 frames) — IOS can do video, but the 4D-reconstruction task is *dynamic-scene*, and dental arch is *static*. The dynamic-prior is *not* what we want for static dental arch.
- ❌ Geo4D outputs *point maps + depth + ray maps* — need TSDF-fusion or similar to get a watertight arch mesh. Same caveat as Aether 199.
- ❌ Geo4D requires *3D-annotated training data* (Spring, BEDLAM, PointOdyssey, TartanAir, VirtualKITTI) — dental equivalents exist (3DTeethSeg22, ToSynFCD) but the synthetic-to-real transfer is unproven for dental.
- ❌ 4D-reconstruction is *slower* than static 3R (1.89 FPS vs 20+ FPS for VGGT 192) — *not* chairside-deployable
- ✅ Geo4D is **ICCV 2025 Highlight** (top-10% of accepted papers)
- ✅ Geo4D's **3-modality prediction + multi-modal alignment** is the *killer* H3 design pattern
- ✅ Geo4D's **synthetic-only training** is the *killer* H5 design pattern
- ✅ Geo4D's **uncertainty-weighted L1 + VAE fine-tuning (Eq. 3)** is the *killer* design pattern for adapting pretrained VAEs to non-natural-image modalities

### Concrete Next Steps for v0

**(a) ADOPT 3-MODALITY PREDICTION + MULTI-MODAL ALIGNMENT AS V0 SUB-TASK 1 PARADIGM:** $200-400 Lambda, 2-4 weeks engineering, the *killer* H3 design lesson. Predict `(X, D, r)` for each IOS frame + global alignment optimization. Replace the planned "single pointmap" + post-processing with the Geo4D 3-modality stack. *Direct port* of Eqs. 4-10 + Supp. Alg. 1. The 8% AbsRel improvement from multi-modal inference (Tab. 3) is *the* reason.

**(b) ADOPT UNCERTAINTY-WEIGHTED L1 + VAE FINE-TUNING (EQ. 3) AS V0 SUB-TASK 1 VAE-PORTING PATTERN:** $20-50 Lambda, 1-2 days engineering, the *killer* design lesson for adapting pretrained video VAEs to non-natural-image modalities. For v0, this means: take Wan2.1 / SVD VAE, fine-tune the *decoders* for depth + ray maps with the uncertainty-weighted L1 loss (Eq. 3), train the *uncertainty branches* to predict per-pixel σ. *10-20 lines PyTorch.*

**(c) ADOPT 5-STEP DDIM + SLIDING WINDOW (V=16, s=4) AS V0 SUB-TASK 1 INFERENCE CONFIG:** $0, 1-line config + 5-10 lines code, the *killer* efficiency lesson. For v0 sub-task 1, the 5-step DDIM trick is *directly* portable (the deterministic-4D-reconstruction finding is *more* true for dental arch than for in-the-wild video). Sliding window V=16 matches the *natural* clinical IOS chunking.

**(d) ADOPT GROUP-WISE ALIGNMENT (VS PAIRWISE) AS V0 SUB-TASK 1 ALIGNMENT:** $0, direct port of Eqs. 4-10, the *killer* design pattern for long-video 3D. For v0, the group-wise alignment is *especially* important for clinical IOS where the arch scan is 64-128 frames (long, multi-clip).

**(e) ADOPT SYNTHETIC-ONLY TRAINING AS V0 SUB-TASK 1 H5 MECHANISM:** $300-500 Lambda, 4-6 weeks. Train v0 sub-task 1 on 5 dental-synthetic datasets: (1) 3DTeethSeg22 (synthetic teeth + IOS-like scans), (2) ToSynFCD (synthetic full crowns), (3) synthetic IOS (procedural dental arch + camera simulation), (4) clinical 3D scans (de-identified Tufts/OSF), (5) synthetic tooth-shape variations (procedural crown-shape generation). *Zero real clinical IOS data needed for v0 sub-task 1's first cut.* Direct port of the Geo4D 5-dataset training paradigm.

**(f) ADOPT CAMERA-ROTATION-FIRST / TRANSLATION-SECOND AS V0 SUB-TASK 1 TRAINING CURRICULUM:** $0, 0-day, the *killer* design lesson from Tab. 2 (Geo4D wins rotation, loses translation). For v0 sub-task 1, train camera rotation first, then refine camera translation with depth priors.

**(g) INCLUDE FOCAL-LENGTH VARIATION IN V0 TRAINING DATA:** $0, 0-day, the *killer* design lesson from Sec 9 Limitations. For v0 sub-task 1, ensure the synthetic IOS data includes *focal-length variation* (different IOS vendors, different magnification settings, different working distances). This avoids the Geo4D focal-length-shift failure mode.

**(h) CITE Geo4D AS THE "3-MODALITY-FUSION + MULTI-MODAL-ALIGNMENT" PARADIGM FOUNDER IN V0 SUB-TASK 1 RELATED-WORK:** $0, 1-2 hours writing, 1 paragraph: *"We adopt the 3-modality prediction + group-wise alignment paradigm (Jiang et al. 2025) for our full-arch 3D reconstruction, which has been shown to achieve SOTA on Sintel/Bonn/KITTI video depth and camera-rotation estimation, and is the first to demonstrate that synthetic-only training on a pre-trained video generator generalizes to real in-the-wild videos."*

**(i) ADOPT BEDLAM-STYLE PROGRESSIVE TRAINING CURRICULUM (1-modality → 3-modality):** $0, 0-day, the *killer* training-stability lesson from Supp. Sec 4.1. For v0 sub-task 1, train (1) point maps only, (2) add multi-resolution, (3) add ray + depth — the same 3-stage curriculum. Avoids the catastrophic-forgetting problem of multi-task joint training from scratch.

**(j) CITE Geo4D AS THE "SYNTHETIC-ONLY TRAINING + PRETRAINED VIDEO GENERATOR" PARADIGM FOUNDER:** $0, 1-2 hours writing, 1 paragraph. For v0 paper, Geo4D is the *first* paper to demonstrate that pretrained video generators + synthetic data = real-data generalization. Cite as a *prior* in v0's "training data" section.

**(k) COMPARE Geo4D vs LiteVGGT 198 vs Aether 199 AS V0 SUB-TASK 1 BASELINES:** $100 Lambda, 1-2 weeks. Run all 3 on the same held-out set of (synthetic) dental IOS videos. Measure video depth (AbsRel), camera pose (ATE, RPE-R), and runtime. If Geo4D *outperforms* LiteVGGT on dental depth, we have a *strong baseline* for v0. If *not*, we have a *negative result* (dental = textureless + non-dynamic, where the video prior doesn't help).

**(l) CITE Geo4D'S 4D-RECONSTRUCTION-FINDING-IS-DETERMINISTIC INSIGHT (SUPP. TAB. 6) AS V0 INFERENCE-EFFICIENCY ARGUMENT:** $0, 1-2 hours writing, 1 sentence: *"Following Geo4D (Jiang et al. 2025), we use 5-step DDIM at inference because 4D-shape reconstruction is a more deterministic task than video generation, requiring fewer denoising steps (Jiang et al. 2025, Supp. Tab. 6)."* — the *killer* inference-efficiency argument for v0.

### v0 Sub-Task 1 Stack Update: 26 papers covered (13 paradigms)

Adds **(xiv) 3-modality-prediction + multi-modal-alignment + synthetic-only-training (Geo4D 200)** NEW *H3-paradigm-founding + H5-paradigm-founding* design axis. The v0 sub-task 1 design space is now the **MOST-COMPREHENSIVE** 2024-2026 long-context 3R arc in existence (26 papers, 14 paradigms, **7 sparse-3R design axes including 3-modality-fusion**).

### Strategic Comparison: Geo4D 200 vs Aether 199

| Aspect | Geo4D 200 | Aether 199 |
|---|---|---|
| **Architecture** | 3D video diffusion (DynamiCrafter fine-tuned) | 4D video diffusion (CogVideoX-5b-I2V fine-tuned) |
| **Output modalities** | 3 (point + depth + ray) | 2 (depth + ray) |
| **Post-processing** | Group-wise alignment optimization (4-loss) | None |
| **Synthetic-only training** | YES (5 synthetic datasets) | YES (DA-V + TheMatrix) |
| **Tasks** | 4D reconstruction only | 4D reconstruction + prediction + planning |
| **SOTA on Sintel video depth** | 0.205 AbsRel (best) | 0.270 (DepthCrafter) — not directly comparable |
| **Camera-rotation SOTA** | YES (0.547 RPE-R) | Not reported |
| **License** | ⚠️ NONE | ✅ MIT |
| **Code released** | YES (GitHub) | YES (GitHub + HF) |
| **Size** | Smaller (~2-3B params) | Larger (5B params) |
| **Inference speed** | 1.89 FPS | ~0.5 FPS (5B model) |
| **Last push** | 2025-06-06 (frozen) | 2025-10-26 (active) |
| **For v0 sub-task 1 (full-arch)** | BETTER (3-modality + alignment + SOTA depth) | WORSE (only depth + ray, no 3D pointmap, 5B params) |
| **For v0 sub-task 1 (synthetic-only)** | EQUAL | EQUAL |
| **For v0 commercial deployment** | WORSE (no license) | BETTER (MIT license) |
| **Citing in v0 paper** | AS H3 + H5 paradigm founder | AS unified 4D reference |

**Recommendation for v0:** **Cite Geo4D as the 3-modality-fusion + multi-modal-alignment + synthetic-only-training paradigm founder** (H3 + H5 contributions), but **use Aether as the commercial-deployable reference** (MIT license + active maintenance). Geo4D's *technical* contributions (3-modality, group-wise alignment, Eq. 3 uncertainty-weighted L1, Supp. Tab. 6 5-step DDIM) are *directly* portable to v0 even if we don't *deploy* Geo4D itself.

### Open Q for HK

- (i) Adopt 3-modality prediction for v0 sub-task 1? (YES — *killer* H3 paradigm, $200-400 Lambda)
- (ii) Adopt uncertainty-weighted L1 + VAE fine-tuning (Eq. 3)? (YES — *killer* design pattern for adapting pretrained video VAEs to non-natural-image modalities, $20-50 Lambda)
- (iii) Adopt 5-step DDIM + sliding window (V=16, s=4)? (YES — *killer* efficiency lesson, $0)
- (iv) Adopt group-wise alignment (vs pairwise)? (YES — *killer* long-video pattern, $0)
- (v) Adopt synthetic-only training? (YES — *killer* H5 paradigm, $300-500 Lambda)
- (vi) Adopt camera-rotation-first / translation-second curriculum? (YES — *killer* training-stability lesson, $0)
- (vii) Include focal-length variation in v0 training data? (YES — *killer* failure-mode-avoidance, $0)
- (viii) Cite Geo4D as 3-modality-fusion + multi-modal-alignment paradigm founder? (YES — $0, 1-2 hours writing)
- (ix) Cite Geo4D as synthetic-only-training + pretrained-video-generator paradigm founder? (YES — $0, 1-2 hours writing)
- (x) Compare Geo4D vs LiteVGGT 198 vs Aether 199 on dental depth? (YES — $100 Lambda, 1-2 weeks, *strong* Table 1 baseline comparison)
- (xi) Cite Geo4D's 4D-deterministic insight (Supp. Tab. 6) as v0 inference-efficiency argument? (YES — $0, 1-2 hours writing)
- (xii) Use Geo4D as v0 sub-task 1 *production* baseline? (NO — Aether 199 has MIT license; Geo4D has NO license; for *commercial* deployment, Aether wins)

### v0 Sub-Task 1 Compute Update

- **v0 sub-task 1 compute: ~$3,600-5,300 Lambda** (was $3,400-5,000 from 186-note, +$200-300 for Geo4D 200 3-modality-stack + uncertainty-weighted-L1 + group-wise-alignment integration)
- **v0 TOTAL compute: ~$12,540-18,480 Lambda** (was $12,340-18,180, +$200-300 for Geo4D 200 integration)

---

**★ Next paper to read (201):** the 199-note's recommended *next* was Geo4D 200 (now read!). The 200-note's recommended *next* is **MonST3R (Zhang 2024, arXiv:2401.17149) — the *direct* predecessor of Geo4D's group-wise alignment from pairwise alignment** (already in reading list as paper 003, the *founding* paper of dynamic DUSt3R extension) **OR** **DepthCrafter (Hu 2025, arXiv:2409.02095, CVPR 2025)** — the *concurrent* 2025 video-depth model that Geo4D explicitly compares to and beats on Sintel (0.205 vs 0.270 AbsRel = -24%); the *founding* paper of the "video diffusion for depth" paradigm that Geo4D *extends* to 4D. Alternatives: **(a) Easi3R (Chen 2025, arXiv:2503.24391)** the *training-free* DUSt3R-to-4D extension; **(b) Ray-Diffusion (Zhang 2024)** the *founding* paper of Plücker-coordinate prediction (the *killer* contribution that Geo4D borrows for ray maps); **(c) DUSt3R (Wang 2024, arXiv:2312.14132)** the *founding* paper of pointmap representation (already in reading list as paper 001); **(d) DynamiCrafter (Xing 2024, arXiv:2402.04865)** the *backbone* video generator that Geo4D fine-tunes; **(e) Marigold (Ke 2024, arXiv:2312.02145)** the *image-generator-to-depth* paradigm that inspired Geo4D; **(f) WVD (Luo 2024, World-Video-Depth)** the *concurrent* video-depth model. **Recommendation: *read 201 = DepthCrafter (Hu 2025, arXiv:2409.02095, CVPR 2025)*** — the *concurrent* 2025 video-depth model that Geo4D explicitly compares to and beats, the *founding* paper of the "video diffusion for depth" paradigm that Geo4D *extends* to 4D, the *right* next paper to understand the *concurrent* 2025 alternative to Geo4D's 4D-reconstruction approach. After DepthCrafter 201, the v0 sub-task 1 *video-diffusion-for-3D* design space is *complete* (Aether 199 + Geo4D 200 + DepthCrafter 201 = 3 papers, the *most-comprehensive* 2025 *video-diffusion-for-3D* arc). The *commercial-deployment caveat*: DepthCrafter is by Tencent (likely non-commercial license), so *cite* as a 2025 concurrent alternative but *don't* deploy for v0 (Aether 199 is the v0-deployable choice).
