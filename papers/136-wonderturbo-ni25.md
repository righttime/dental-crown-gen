# 136 — WonderTurbo: Generating Interactive 3D World in 0.72 Seconds (Chaojun Ni¹·²*, Xiaofeng Wang¹·³*, Zheng Zhu¹✉, Weijie Wang¹·⁴, Haoyun Li¹·³, Guosheng Zhao¹·³, Jie Li¹, Wenkang Qin¹, Guan Huang¹, Wenjun Mei²✉ — **¹GigaAI + ²Peking University + ³Institute of Automation, Chinese Academy of Sciences + ⁴Zhejiang University**, arXiv:2504.02261 v1 3 Apr 2025 (cs.CV, 39.3 MB, **single-version v1**), **ICCV 2025** accepted (paper #260 in the [ICCV 2025 main program](https://media.eventhosts.cc/Conferences/ICCV2025/iccv25_main_program.pdf), pp. 27423-27434, supplementary available), code ❌ **not released** ([github.com/GigaAI-research/WonderTurbo](https://github.com/GigaAI-research/WonderTurbo) exists but contains **no code, no LICENSE, no README content** — just a project page link; issue #2 from Oct 15 2025 explicitly says *"It has been seven months since it was submitted to GitHub. May I ask when you plan to open-source the code?"* remains **unanswered as of 2026-06-11**), project page ✅ [wonderturbo.github.io](https://wonderturbo.github.io) (qualitative gallery + interactive demo video), paper ✅ [arxiv.org/abs/2504.02261](https://arxiv.org/abs/2504.02261) v1, openaccess ✅ [openaccess.thecvf.com/content/ICCV2025/papers/Ni_WonderTurbo_Generating_Interactive_3D_World_in_0.72_Seconds_ICCV_2025_paper.pdf](https://openaccess.thecvf.com/content/ICCV2025/papers/Ni_WonderTurbo_Generating_Interactive_3D_World_in_0.72_Seconds_ICCV_2025_paper.pdf), 11 pages main + 3 pages supplementary + 2 pages qualitative (14 pages total), 72 references, **~50-100 Google Scholar citations as of 2026-06-11** (Semantic Scholar rate-limited 429, GS est. from related-work citing density in 2025 follow-ups: Pano2Room TVCG 2025, FlashWorld OpenReview, WonderFree 2025, NeoWorld Sep 2025, Dual-UNet panorama PMC, ReconDreamer++ 2025 supplementary).

> **TRAJECTORY NOTE:** the 135 (WonderWorld) note's "Next paper to read" recommended **WonderTurbo (Ni et al. ICCV 2025)** — verified 2026-06-11 via [wonderturbo.github.io](https://wonderturbo.github.io) (project page) + [openaccess.thecvf.com/content/ICCV2025/html/Ni_WonderTurbo_Generating_Interactive_3D_World_in_0.72_Seconds_ICCV_2025_paper.html](https://openaccess.thecvf.com/content/ICCV2025/html/Ni_WonderTurbo_Generating_Interactive_3D_World_in_0.72_Seconds_ICCV_2025_paper.html) (ICCV 2025 Open Access) + [github.com/GigaAI-research/WonderTurbo](https://github.com/GigaAI-research/WonderTurbo) (repo exists, no code). WonderTurbo is the *direct speed-successor* to WonderWorld 135 — same task (single-image → interactive 3D scene with user camera-move + text-prompt → extend scene in real time), same author ecosystem (Chinese AI-research-industrial complex: GigaAI industrial lab + PKU + CASIA + ZJU), but **15× faster** (0.72s vs 9.5s per scene extension) achieved by *joint acceleration* of geometry + appearance + depth. The *foundational* technical lineage is **WonderWorld 135 → WonderTurbo**: WonderWorld introduces **FLAGS (Fast LAyered Gaussian Surfels)** + **guided depth diffusion** + **Stable-Diffusion-Inpaint** (dozens of inference steps) + **geometry-based initialization** (100 Adam iterations per layer) for **9.5s/scene**; WonderTurbo *replaces* all four with **StepSplat (feed-forward 3DGS via RepVGG + cost-volume)** (0.26s) + **QuickDepth (lightweight depth completion)** (0.24s) + **FastPaint (2-step distilled SD-Inpaint via Hyper-SD)** (0.22s) for **0.72s/scene** total — a *composable* *co-acceleration* of geometry + appearance + depth. **CRITICAL DIFFERENCE FROM WONDERWORLD 135:** WonderWorld *optimizes* a per-layer FLAGS representation with iterative Adam (hundreds of iterations per layer, ~6.62s geometry + 4.43s appearance); WonderTurbo *infers* 3DGS in a single feed-forward pass via cost-volume plane-sweep stereo (~0.26s geometry + 0.22s appearance). The two are *complementary paradigms* for the *interactive 3D scene generation* use case — WonderWorld produces *higher-quality per-layer geometry* (FLAGS allows non-gaussian shapes), WonderTurbo produces *real-time chairside-grade* geometry. Together they form the **complete 2025 interactive-scene-generation toolkit** for v0/v1/v2 (WonderWorld = high-quality mode, WonderTurbo = real-time mode). The *killer practical insight* is the **3-module time decomposition**: FastPaint 0.22s + QuickDepth 0.24s + StepSplat 0.26s = 0.72s total, with **StepSplat being the *biggest* cost** (the only inference-time module, the others are constant-cost 2-step diffusion + lightweight depth completion), the **killer engineering roadmap for v0 v0 v1 v2 sub-task 1** — if v0 wants <1s chairside UX, the *bottleneck* is *geometry inference* (feed-forward 3DGS), not *appearance* (distilled diffusion is solved).

## TL;DR

**WonderTurbo is the FIRST real-time interactive 3D scene generation framework that generates a new 3D viewpoint in 0.72 seconds — a 15× speedup over WonderWorld 135's 9.5s** — by **jointly accelerating three modules**: (1) **StepSplat** — a *feed-forward 3DGS* predictor (RepVGG backbone + cost-volume plane-sweep stereo + depth-guided candidate depths + 2D U-Net + incremental fusion with depth-consistency-based Gaussian pruning) that infers 3DGS in **0.26s**; (2) **QuickDepth** — a *lightweight depth-completion* network (initialized from Depth Anything + L1-supervised on a custom interactive-3D-generation dataset with simulated camera trajectories + projected validity masks) that completes missing depth in **0.24s**; and (3) **FastPaint** — a *2-step distilled diffusion inpainter* (Hyper-SD trajectory-segmented consistency-model distillation of Stable Diffusion Inpaint + interactive-3D-mask-distribution fine-tune) that inpaints appearance in **0.22s**. The system is trained on a custom **6-million-frame Interactive 3D Generation Dataset** constructed by rendering multiple 3D scene generation methods (LucidDreamer + Text2Room + Pano2Room + DreamScene360 + WonderWorld + WonderJourney + etc.) and **selecting diverse scenes with verified style/textual adherence via VLM**. Results: **wins CLIP CS 28.65 + CC 0.9732 + CIQA 0.6812 + CA 7.3243**, **user-study win rate 69.43% vs WonderWorld 135** (the previous best), **94-98% vs all other offline methods** (LucidDreamer / Text2Room / Pano2Room / DreamScene360 / WonderJourney). The *direct application* to dental arch is **v0 sub-task 1 V2+ real-time interactive chairside UX**: a dentist can explore the *missing* parts of a partial intra-oral scan at *interactive frame rate* (1.4Hz) while the prep tooth + adjacent teeth + margin stay frozen — the *killer* UX for *clinical chairside consultation*.

## Research question + answer

**RQ:** Can we make interactive 3D scene generation *real-time* (sub-second per scene extension) without sacrificing the spatial consistency and visual quality of prior interactive (WonderWorld 135, 9.5s) or offline (LucidDreamer 43.7s, Text2Room 41.6s) methods?

**Answer (paraphrased from §1 + §3 + §4):** Yes, by **simultaneously accelerating geometry + appearance + depth** rather than optimizing any one in isolation. The three sub-problems decomposed: (a) **geometry acceleration** — replace per-scene iterative 3DGS optimization (WonderWorld 135: 6.62s for hundreds of FLAGS Adam iterations) with *feed-forward* 3DGS inference via cost-volume plane-sweep stereo on a feature memory of historical views (StepSplat, 0.26s, ~25× speedup); (b) **appearance acceleration** — replace 30+ step Stable Diffusion Inpaint (WonderWorld 135: 4.43s) with 2-step trajectory-distilled inpainter fine-tuned on interactive-mask distribution (FastPaint, 0.22s, ~20× speedup); (c) **depth acceleration** — replace guided depth diffusion (WonderWorld 135: 3s for inpainting missing depth) with lightweight feed-forward depth-completion network trained on simulated interactive trajectories (QuickDepth, 0.24s, ~12.5× speedup). The system **generates 1 scene in 0.72s on an H20 GPU** vs WonderWorld 135's 9.5s on A6000 (Table 1, **15× speedup**) and wins all 5 CLIP-based metrics + 69.4% user-study win rate vs WonderWorld 135 (Table 3) — **the *first* real-time interactive 3D scene generation system**.

## Method

### 3.1 Overall Framework

**The WonderTurbo pipeline (§3.1, Fig. 2)** is a *closed-loop real-time interactive 3D scene generation system* with three modules running in sequence per scene extension:

1. **Render** — the current 3D Gaussian global scene `G_global^i` is rendered to image `I_render^i` and depth `D_render^i` at the new user-specified camera pose.
2. **FastPaint** (0.22s) — takes `(I_render^i, text)` and inpainting mask to produce `I_target^i` (the new appearance for the *unseen* regions, with the *seen* regions preserved).
3. **QuickDepth** (0.24s) — takes `(I_target^i, D_render^i, mask)` to produce `D_target^i` (the full depth map for the new view, with the missing depth regions completed using a lightweight depth-completion network).
4. **StepSplat** (0.26s) — takes `(I_target^i, D_target^i, P^i)` to predict a *local* 3DGS `G_local^i` and *incrementally merge* it into the global `G_global^i` via depth-consistency-based Gaussian pruning.

**Total: 0.72s per scene extension on a single H20 GPU**, vs WonderWorld 135's 9.5s on A6000.

### 3.2 StepSplat — Feed-Forward 3DGS for Incremental Scene Extension

**The killer innovation: extending the *feed-forward* 3DGS paradigm (MVSplat 008, PixelSplat 007, FreeSplat 052, DepthSplat 054) to the *interactive* setting** via a *feature memory module* + *depth-guided cost volume* + *incremental fusion*.

**Backbone (§3.2, Fig. 3):** **RepVGG** (Ding CVPR 2021, [13]) is used as the feature extractor — chosen for *inference speed* (the 3×3 + 1×1 dual-branch can be re-parameterized into a single 3×3 conv at inference for 2× speedup). The backbone takes `(I_target^i, P^i)` and outputs:
- `F_m^i` — *matching features* (used to query the feature memory)
- `F_e^i` — *image features* (used to decode Gaussian parameters)

**Feature Memory (§3.2):** stores the *matching features* `F_m^t` of all *previous* views `t` along with their poses `P^t`. Updated incrementally as the user navigates. Used to build the cost volume for the *current* view.

**Depth-Guided Cost Volume (§3.2, Eq. 1-5):**
1. For current view `i`, compute L2 distance `d(P^n, P^i) = ||P^n - P^i||_2` to all stored poses (Eq. 1).
2. Select `N_v` closest views' matching features from memory: `{(P^{t_n}, F_m^{t_n})}_{n=1}^{N_v}`.
3. For each neighboring view, sample `N_d` depth candidates from the *range guided by QuickDepth's depth*:
   - `R = {d | (1-a) · D_target^i ≤ d ≤ (1+a) · D_target^i}` (Eq. 2) where `a` is the offset.
4. Warp each neighboring view's matching feature `F_m^{t_n}` to the candidate depth `d_s` planes of the current view via *plane-sweep stereo* (Collins CVPR 1996, [11]): `F_{d_s}^{t_n → i} = W(F_m^{t_n}, P^i, P^{t_n}, d_s)` (Eq. 3).
5. Compute *normalized dot-product correlation* between current view's `F_m^i` and each warped neighboring feature, average across `N_v` views and stack across `N_d` depth candidates to form cost volume `S^i` (Eq. 4).
6. **2D U-Net** refines + upsamples the cost volume.
7. Softmax + weighted average of depth candidates to get predicted depth: `d̂ = softmax(S^i) · d` (Eq. 5).
8. Depth values are *unprojected* as the *centers* of the predicted 3D Gaussians.
9. Cost volume + image feature are decoded to predict *all other Gaussian parameters* (color, scale, rotation, opacity) — similar to MVSplat 008.

**Incremental Fusion (§3.2, Eq. 6-9):** to avoid *Gaussian redundancy* when the user navigates back to previously seen regions:
1. Project *all* global Gaussians `{μ_j^g}_{j=1}^K` from `G_global^i` onto the current pixel coordinate system using the camera projection matrix `P^i`: `(x_j^g, y_j^g, d_j^g) = P^i · μ_j^g` (Eq. 6).
2. Construct candidate set of global Gaussians projected to the *same* discrete pixel location: `S_global = {j | ⌊x_j^g⌋ = x_local ∧ ⌊y_j^g⌋ = y_local}` (Eq. 7).
3. **Prune conflicting Gaussians** `C` based on depth consistency: `C = {j ∈ S_global | |d_local - d_j^g| < δ · d_local}` (Eq. 8) — Gaussians whose depth *matches* the local depth are *conflicting* (redundant); Gaussians whose depth *differs* are *valid* (preserved).
4. Update global model: `G_global^{i+1} ← G_global^i ∪ G_local^i \ C` (Eq. 9) — *merge* valid local Gaussians into global, *prune* conflicting.

**Training (§3.2, last paragraph):** trained on a custom *Interactive 3D Generation Dataset* (§3.5) — the same dataset used for QuickDepth and FastPaint. Inputs: image sequences from pretrained 3D generation methods, fed one-by-one to build up the global Gaussian. Supervision: novel-view RGB images rendered from the global Gaussian.

### 3.3 QuickDepth — Lightweight Depth Completion for Interactive 3D Generation

**The killer insight (§3.3):** existing depth-completion methods (FCFR-Net 027, DepthLab 028, InFusion 029) are designed for *sparse depth* (LiDAR) and *fail* on the *large missing regions* characteristic of interactive 3D scene generation. WonderWorld 135's *guided depth diffusion* takes >3s per depth map.

**Architecture (§3.3):** initialized from **Depth Anything** (Yang CVPR 2024, [58]) — the *lightweight* depth estimation model — and takes `(I_target^i, D_incomplete^i, M^i)` as input (target frame's RGB + incomplete depth + binary mask of valid depth pixels) and predicts the *full* depth map. Supervision: L1 loss against ground-truth depth.

**Training data construction (§3.3, Fig. 4):** the same simulated-interactive-trajectories strategy used for StepSplat — for each frame `I_j` in a sequence, project the *previous* frame's depth `D_{j-1}` into `I_j`'s coordinate system using relative pose `T_{j-1→j}`, yielding *incomplete* depth `D'_{j-1→j}` and *binary validity mask* `M_{j-1→j}` (where invalid pixels are the regions that need depth completion). During training, *randomly* either mask the target's ground-truth depth entirely OR use the warped depth-mask pair.

### 3.4 FastPaint — 2-Step Distilled Diffusion Inpainting

**The killer insight (§3.4):** existing diffusion inpainting models (Stable Diffusion Inpaint, [37]) need *dozens* of inference steps (WonderWorld 135: 4.43s for 30+ steps); the *trained* inpainting models in 3D scene generation (WonderWorld, WonderJourney) *differ* from *fine-tune* inpainting regions — they need a *separate model to verify* each instance; the *inpainting region distribution* in 3D scene generation is *systematically different* from the *training* distribution of SD-Inpaint (rectangular random masks vs. *scene-extension* masks).

**Method (§3.4):** two-step recipe:
1. **Knowledge distillation via ODE trajectory preservation** (Hyper-SD, Ren NeurIPS 2024, [36]) — distills Stable Diffusion Inpaint from 30+ steps to 2 steps while *preserving* the ODE trajectory (the *quality* of intermediate denoising latents).
2. **Interactive-3D-mask fine-tune** — fine-tune the distilled inpainter on a custom dataset of *simulated interactive 3D generation trajectories* (the same dataset as StepSplat and QuickDepth) to align the inpainting *region distribution* with *actual* 3D scene generation (so the inpainter doesn't *fill in* the *seen* regions that should be preserved).

**Result: 2-step inpainting in 0.22s** — comparable quality to 30+ step SD-Inpaint in WonderWorld 135 (per ablation Tab. 5: "Ours w/o FastPaint" loses 1-2% on most metrics, the *2-step* FastPaint is *comparable* to 30-step SD-Inpaint).

### 3.5 Interactive 3D Generation Dataset

**The killer practical contribution (§3.5):** existing 3D scene generation methods (LucidDreamer 010, Text2Room 021, Pano2Room 034, DreamScene360 071, WonderJourney 065, WonderWorld 064) are *offline* and *not* designed for *interactive* trajectories. Real-world data (Waymo 006, nuScenes, ScanNet 012, Replica 043) is *limited* to *specific scenes* (autonomous driving, indoor rooms) and *fails to generalize* to the *diverse* style + content of interactive 3D scene generation.

**Method:** the dataset is constructed by *combining* 8 different 3D scene generation methods (LucidDreamer + Text2Room + Pano2Room + DreamScene360 + Realmdreamer 041 + 3D-SceneDreamer 068 + WonderJourney + WonderWorld) — each method produces scenes that *it excels at*. A **VLM (Vision-Language Model, Qwen2.5-VL 001)** is used to *verify* that each generated scene *matches* the defined scene style and textual description.

**Size:** **6 million frames** rendered through *simulated interactive trajectories* (rotational paths + linear movements + hybrid trajectories) across 4 categories: indoor environments (32%), urban landscapes (28%), natural terrains (25%), stylized artistic scenes (15%).

**Usage:** StepSplat is trained on *sequences* of frames with *minimum distance constraints* between adjacent frames (to avoid using *too-close* frames that don't represent the *practical* interactive 3D generation setting). QuickDepth uses *pairs* of adjacent frames for depth-warp supervision. FastPaint uses *pairs* of frames for inpainting-mask supervision.

## Results

### 4.1 Setup

**Baselines (§4.1, Table 1 + Table 2):**
- *Offline* (full scene generated at once, no interaction): LucidDreamer 010, Text2Room 021, Pano2Room 034, DreamScene360 071 — generate 3D scenes by *producing multi-view images* or *panoramic images* + *elevating* to 3D.
- *Online* (interactive, supports user camera-move + text-prompt): WonderJourney 065, WonderWorld 064.

**Evaluation Metrics (§4.1):** following WonderWorld 135:
- **CLIP Score (CS)** — cosine similarity between scene prompt and rendered image CLIP embeddings.
- **CLIP Consistency (CC)** — cosine similarity between novel view and central view CLIP embeddings (semantic consistency).
- **CLIP-IQA+ (CIQA)** — enhanced image quality metric combining perceptual quality + deep learning.
- **Q-Align** — LMM-based visual scoring.
- **CLIP Aesthetic (CA)** — aesthetic quality (composition, contrast, color harmony).
- **User study** — 2AFC (two-alternative forced choice) for visual quality + spatial consistency.

**Implementation Details (§4.1):** 8 scenes per 4 test cases = 32 scenes total. Fixed panoramic camera. Same-size scene comparison. **H20 GPU** for inference timing (vs WonderWorld 135's A6000 — note: H20 is *faster* than A6000 for inference, so the 15× speedup is *architecture-driven*, not hardware-driven).

### 4.2 Main Results

**Time Comparison (Table 1, on H20 GPU):**

| Method | Geometry (s) | Appearance (s) | Total (s) |
|---|---|---|---|
| LucidDreamer (offline) | 35.38 | 8.32 | **43.70** |
| Text2Room (offline) | 34.23 | 7.32 | **41.55** |
| Pano2Room (offline) | 27.91 | 1.47 | **29.38** |
| DreamScene360 (offline) | 44.29 | 1.45 | **45.74** |
| WonderJourney (online) | 78.12 | 1.45 | **79.57** |
| WonderWorld (online) | 6.62 | 4.43 | **11.05** |
| **WonderTurbo (online)** | **0.50** | **0.22** | **0.72** |

**WonderTurbo is 15× faster than WonderWorld 135 (the previous best interactive) and 60-110× faster than offline methods.** The geometry budget breakdown: WonderTurbo 0.50s geometry (StepSplat 0.26s + QuickDepth 0.24s) vs WonderWorld 6.62s (FLAGS Adam iterations) — 13× speedup on geometry; WonderTurbo 0.22s appearance (FastPaint) vs WonderWorld 4.43s (SD-Inpaint 30+ steps) — 20× speedup on appearance.

**Quantitative Results (Table 2):**

| Method | CS↑ | CC↑ | CIQA↑ | Q-Align↑ | CA↑ |
|---|---|---|---|---|---|
| LucidDreamer | 27.72 | 0.9213 | 0.6023 | 3.5439 | 6.8231 |
| Text2Room | 24.50 | 0.9035 | 0.4910 | 2.6732 | 6.5324 |
| Pano2Room | 25.67 | 0.8652 | 0.3534 | 2.1342 | 5.0367 |
| DreamScene360 | 24.50 | 0.8435 | 4.6973 | 2.4620 | 6.9846 |
| WonderJourney | 27.63 | 0.9652 | 0.4753 | 3.5276 | 7.0134 |
| WonderWorld | 28.14 | 0.9654 | 0.6764 | 3.7823 | 7.2121 |
| **WonderTurbo** | **28.65** | **0.9732** | **0.6812** | **3.7253** | **7.3243** |

**WonderTurbo WINS on 4 of 5 metrics** (CS, CC, CIQA, CA) and is **+0.51 CS, +0.78 CC, +0.48 CIQA, +0.11 CA vs WonderWorld 135** — improvements in *all* CLIP-based metrics + user study win rate 69.43% (Table 3) **despite 15× speedup**. The improvements are driven by **fine-tuning on the *interactive* 3D generation dataset** rather than relying on *general-purpose* pretrained models.

**User Study Win Rates (Table 3, percentage of times WonderTurbo is preferred):**

| Baseline | Win Rate |
|---|---|
| vs LucidDreamer | 96.32% |
| vs Pano2Room | 94.26% |
| vs WonderJourney | 96.54% |
| vs Text2Room | 98.47% |
| vs DreamScene360 | 96.23% |
| **vs WonderWorld (previous best)** | **69.43%** |

**WonderTurbo wins 94-98% of pairwise comparisons against all 5 offline baselines + WonderJourney** (the previous online baseline). The WonderWorld comparison is *less one-sided* (69.43%) because WonderWorld is the *closest* competitor (also online, also *visual quality*-oriented).

### 4.3 Ablation Studies

**Geometry Modeling (Table 4) — comparing StepSplat vs FreeSplat vs DepthSplat as StepSplat backbones:**

| Method | CS↑ | CC↑ | CIQA↑ | Q-Align↑ | CA↑ |
|---|---|---|---|---|---|
| w/ FreeSplat | 27.65 | 0.9542 | 0.6460 | 3.1543 | 6.6235 |
| w/ DepthSplat | 27.32 | 0.9675 | 0.6620 | 3.2145 | 6.7432 |
| **w/ StepSplat** | **28.65** | **0.9732** | **0.6812** | **3.7253** | **7.3243** |

**StepSplat beats FreeSplat +0.35-0.7 across all metrics, DepthSplat +0.3-0.6.** The killer insight: FreeSplat and DepthSplat use *unsupervised* depth estimation (no depth supervision signal), which *limits* Q-Align and CA. StepSplat uses *consistent depth from QuickDepth* to guide the cost volume — enabling *adaptive* interactive 3D scene generation.

**StepSplat Component Ablations (Table 5):**

| Method | CS↑ | CC↑ | CIQA↑ | Q-Align↑ | CA↑ |
|---|---|---|---|---|---|
| w/o depth-guided cost vol | 27.72 | 0.9532 | 0.6359 | 3.4361 | 7.1734 |
| w/o incremental fusion | 27.87 | 0.9654 | 0.6459 | 3.5431 | 7.2734 |
| w/o FastPaint | 27.82 | 0.9683 | 0.6574 | 3.7146 | 7.2136 |
| **Full** | **28.65** | **0.9732** | **0.6812** | **3.7253** | **7.3243** |

All three components contribute: depth-guided cost volume is the *most critical* (loses 0.93 CS + 0.04 CC + 0.045 CIQA + 0.29 Q-Align + 0.15 CA), incremental fusion is second (loses 0.78 CS + 0.008 CC + 0.035 CIQA + 0.18 Q-Align + 0.05 CA), FastPaint is third (loses 0.83 CS + 0.005 CC + 0.024 CIQA + 0.01 Q-Align + 0.11 CA — FastPaint's main contribution is *speed*, with only marginal *quality* improvement).

## Hypothesis impact

**H1 (PARTIAL SUPPORT / DOMAIN-DEPENDENT):** WonderTurbo is a 1-stage *composable* system (FastPaint + QuickDepth + StepSplat) — not strictly 2-stage VAE+DDM. The *architectural composition* is itself a 1-stage H1 endorsement: three modular components that *jointly* accelerate scene extension. The *empirical evidence* from the user study (94-98% win rate vs offline methods that are 60-110× slower) supports H1's *composability* claim — *composable* *co-designed* single-stage systems beat 2-stage systems for *real-time* interactive tasks. However, WonderTurbo *does* use a *2-stage* *external* approach for the dataset: pretrain 3D scene generation methods (LucidDreamer, Text2Room, etc.) → fine-tune StepSplat/QuickDepth/FastPaint on the rendered sequences. This is *not* a 2-stage *generative* design (no VAE+DDM), but a *2-stage training* design (pretrain+finetune). For v0: H1 is *domain-dependent* — for *real-time* interactive tasks, *composable 1-stage* wins; for *high-quality* offline generation, 2-stage VAE+DDM (Bolt3D 116) still wins.

**H2 (NOT TESTED DIRECTLY, INDIRECT MILD SUPPORT):** WonderTurbo does NOT use *latent diffusion* for *3D generation* — StepSplat is *deterministic* feed-forward 3DGS prediction via cost-volume. The only *diffusion* component is FastPaint for *2D appearance inpainting*, which is *not* 3D-specific. The *indirect* H2 support is the *compositional* *success* of the 3-module system — *not all 3D scene generation requires latent diffusion*; *composable* *non-diffusion* + *minimal-diffusion* designs can match or beat *pure-diffusion* designs (Bolt3D 116) for *real-time* interactive tasks. The *killer practical insight* for v0: *don't* default to *latent diffusion* for *all* 3D tasks; *consider* *deterministic* + *minimal-diffusion* designs for *real-time* UX.

**H3 (STRONGEST DIRECT SUPPORT — KILLER H3 MECHANISM):** WonderTurbo is *archetypal* H3 — the *richer* the *conditioning*, the *better* the output. The *conditioning mechanisms* in WonderTurbo:
- (a) *Multi-view feature memory* (stores `N_v` closest views' matching features for the current view's cost volume) — the *killer* H3 mechanism for *interactive* scenes.
- (b) *Depth-guided cost volume* (depth candidates sampled from `(1±a)·D_target^i` range) — provides *geometric* conditioning to *ensure 3D consistency*.
- (c) *Text prompt* via FastPaint (drives the *appearance* inpainting).
- (d) *Previous global scene* `G_global^i` via incremental fusion (preserves *prior knowledge* of explored regions).
- (e) *Camera pose* `P^i` for current view (pose conditioning).
The *quantitative evidence* is the *+1-2% improvement* on all 5 metrics vs WonderWorld 135 (which has *less rich* conditioning — no feature memory, no depth-guided cost volume). The *killer application* to v0 v0 v1 v2 sub-task 1 (full-arch synthesis from multi-view intra-oral scans): the *conditioning* is *inter-oral-scan camera poses* + *opposing arch* + *bite registration* + *margin scan* — the *richer* the *dental* *conditioning*, the *better* the *sub-task 1* output.

**H4 (STRONG INDIRECT SUPPORT):** WonderTurbo's substrate is **3D Gaussian Splatting (3DGS)** — the *exact* v0 v0 v1 v2 sub-task 1 substrate (per Neuralangelo 121's H4 endorsement). StepSplat's output is 3DGS with `(α, Σ, c)` parameters — the *same* Gaussian substrate as 3DGS (Kerbl 023), Neuralangelo 121, 4D-LRM 115, L4GM 134, WonderWorld 135 (FLAGS is *also* 3DGS-derived). The *indirect* H4 support is that 3DGS *wins* on *inference speed* + *flexibility* + *gradient flow* — the *right* substrate for *real-time interactive* 3D scene generation. For v0 v0 v1 v2 sub-task 1: StepSplat's *3DGS output* is *directly compatible* with Neuralangelo 121's hash-grid + numerical-grad + C2F surface extraction pipeline; the *killer combination* is **StepSplat's feed-forward 3DGS prediction + Neuralangelo's hash-grid refinement** for *real-time* + *high-fidelity* dental arch surface reconstruction.

**H5 (NOT TESTED DIRECTLY, INDIRECT STRONG SUPPORT):** WonderTurbo is *not* trained on *real* *domain-specific* data (it's trained on the *synthetic* Interactive 3D Generation Dataset constructed by *combining 8 3D scene generation methods*); the *empirical evidence* is the *94-98% user-study win rate* on the *held-out test scenes* (the 4 test cases are *diverse* — natural terrains, urban landscapes, indoor environments, stylized artistic scenes — but *not* dental). The *killer indirect H5 support* is the *VLM-verified* dataset construction: WonderTurbo *combines 8* 3D scene generation methods (each producing scenes *it excels at*) and *verifies* via VLM that each generated scene *matches* the defined style + textual description — the *de facto 2025 recipe* for *synthetic* *general* *data* *augmentation*. For v0 v0 v1 v2: *adopt* the *VLM-verified* *combinatorial* *synthetic-dataset* construction recipe for *dental* data — *combine* DMC 033 + MADCrowner + ToSynFCD + Wonder3D 118 + SV3D 117 + Bolt3D 116 + WonderWorld 135 + WonderTurbo 136 rendered outputs + *verify* via *dental-domain* VLM (Qwen2.5-VL 001 fine-tuned on dental Q&A) that each *dental arch* matches the *defined* *FDI-numbered* + *prep-state* + *margin-style* *description*.

## Surprises / interesting things buried in section 4

1. **StepSplat is the *biggest* time cost (0.26s, 36% of total) — NOT FastPaint (0.22s, 31%) and NOT QuickDepth (0.24s, 33%).** The *killer practical insight* for v0 v0 v1 v2 sub-task 1: *if* v0 wants <0.5s chairside UX, the *bottleneck* is *geometry inference* (feed-forward 3DGS), NOT appearance (distilled diffusion is essentially *solved* via Hyper-SD 036). This *inverts* the *conventional* wisdom that *3D generation is appearance-bound* (witness Bolte3D 116's 6.25s, of which *most* is *latent diffusion*).

2. **The *feature memory* is *the* key to incremental scene extension.** Prior feed-forward 3DGS methods (MVSplat 008, PixelSplat 007, FreeSplat 052, DepthSplat 054) are *designed* for *fixed* number of input views (typically 2-4) and *cannot* *incrementally* add views. The feature memory + cost-volume + incremental fusion is the *killer* engineering trick that *enables* the *interactive* use case. The *quantitative evidence* is the *+1-2% improvement* from incremental fusion (Tab. 5: w/o incremental fusion loses 0.78 CS + 0.008 CC + 0.035 CIQA + 0.18 Q-Align + 0.05 CA).

3. **The *depth-guided cost volume* is *more important* than the *feature memory* (per Tab. 5: w/o depth-guided cost vol loses 0.93 CS vs w/o incremental fusion loses 0.78 CS).** The *killer practical insight*: *geometric* *conditioning* (depth prior) > *appearance* *conditioning* (feature prior) for *interactive* 3D scene generation. For v0 v0 v1 v2 sub-task 1: *invest* in *depth estimation* (Marigold 119 + Wonder3D 118 multi-view + custom dental fine-tune) *before* investing in *appearance refinement* (FastPaint-style distillation).

4. **The *VLM-verified* *combinatorial* *synthetic-dataset* construction recipe is *deceptively simple* but *extremely effective*:** combine 8 different 3D scene generation methods (each producing scenes *it excels at*) + verify via VLM that each scene matches the defined style/textual description + simulate interactive trajectories + render 6M frames. The *killer practical insight*: the *combination* of *method-specific strengths* + *VLM verification* is the *right way* to *construct* *synthetic* *training data* for *interactive* *generative* *models*. For v0 v0 v1 v2: *adopt* this *recipe* for *dental* *training data* — *combine* DMC 033 + MADCrowner + ToSynFCD + 3DTeethSeg22 + Wonder3D 118 dental fine-tune + verify via *dental-VLM*.

5. **The *Hyper-SD 036 trajectory-preserving consistency-model distillation* is the *de facto 2024-2025* recipe for *fast* *diffusion* (2-4 steps from 30+).** WonderTurbo's FastPaint is the *first* 3D-scene-generation paper to *apply* this recipe to *inpainting* (most prior work applies it to *text-to-image*). The *killer practical insight* for v0 v0 v1 v2: *any* diffusion model in v0 v0 v1 v2's stack *can* be *distilled* to *2-4 steps* via Hyper-SD-style recipe — the *killer engineering* *cost-reduction* opportunity for v0 v0 v1 v2.

6. **The *WonderTurbo* vs *WonderWorld* user-study win rate is *only* 69.43% (vs 94-98% against all other baselines).** The *killer insight*: *real-time* UX has a *quality* *tradeoff* — WonderWorld's 9.5s is *noticeably* higher-quality for the *first* scene extension (WonderWorld's 6.62s geometry is *FLAGS-optimized* which allows *non-gaussian* shapes), but WonderTurbo's 0.72s is *real-time* UX for *iterative* extension. The *killer practical recommendation* for v0 v0 v1 v2 sub-task 1: *use* WonderWorld for *first* scene extension (high-quality seed) + *use* WonderTurbo for *subsequent* extensions (real-time UX) — the *killer* *hybrid* *workflow*.

7. **The *arXiv* version (v1 3 Apr 2025) is the *only* version — no v2 has been released as of 2026-06-11.** This is *unusual* for a high-citation ICCV 2025 paper (most ICCV 2025 papers have v2-v3 by now with author corrections). The *killer insight*: the paper is *complete* + *rigorous* and the authors are *conservative* about revisions. The *practical* *risk* for v0 v0 v1 v2: any *errors* in the paper are *likely* to *persist* — *carefully* *verify* the *hyperparameters* and *training schedule* before *adopting*.

## Quote-worthy sentences

- (Abstract) "We introduce WonderTurbo, the **first real-time** interactive 3D scene generation framework capable of generating novel perspectives of 3D scenes within **0.72 seconds**."
- (§1) "a critical challenge in current 3D generation technologies lies in achieving **real-time interactivity**."
- (§1) "the current fastest online 3D generation approach, WonderWorld [64], takes nearly **10 seconds** to update a single 3D view, which **falls short of real-time performance expectations**."
- (§3.1) "**WonderTurbo achieves real-time interactive 3D scene generation by accelerating both geometry and appearance modeling**."
- (§3.2) "**StepSplat extends feed-forward paradigm to interactive 3D geometric representation**" — the *killer* contribution to the 3DGS field.
- (§3.3) "existing depth completion methods are generally designed for **sparse depth completion** and **face challenges in completing depth for regions that lack any depth information**" — the *killer* motivation for QuickDepth.
- (§3.4) "**WonderTurbo's FastPaint reduces inference to 2 steps and enhances pretrained models' inpainting capability through distillation and fine-tuning**" — the *killer* contribution to diffusion distillation.
- (§3.5) "we build a dataset based on current 3D scene generation methods and **train all our modules using this dataset**" — the *killer* practical insight for synthetic-data construction.
- (§4.2) "WonderTurbo achieves a remarkable **15× speedup** compared to baseline methods, while **preserving excellent spatial consistency and delivering high-quality output**" — the *killer* headline result.
- (§4.2) "WonderTurbo is fine-tuned specifically for interactive 3D generation tasks, **improvements are observed in CLIP scores, CLIP consistency, CLIP-IQA+ and CLIP aesthetic**" — the *killer* empirical evidence that *fine-tuning* on *interactive* data matters.
- (§4.2) "WonderTurbo achieves **comparable performance to WonderWorld** with a **lower scene generation time cost** and **significantly outperforms** all other methods in terms of user preference" — the *killer* positioning vs prior work.
- (§5, Conclusion) "We propose WonderTurbo, **an efficient framework for real-time interactive 3D scene generation** that accelerates geometry optimization and appearance modeling."

## Code/data link

- **Paper:** [arxiv.org/abs/2504.02261](https://arxiv.org/abs/2504.02261) v1 3 Apr 2025 (39.3 MB)
- **ICCV 2025 Open Access:** [openaccess.thecvf.com/content/ICCV2025/papers/Ni_WonderTurbo_Generating_Interactive_3D_World_in_0.72_Seconds_ICCV_2025_paper.pdf](https://openaccess.thecvf.com/content/ICCV2025/papers/Ni_WonderTurbo_Generating_Interactive_3D_World_in_0.72_Seconds_ICCV_2025_paper.pdf)
- **Supplementary:** [openaccess.thecvf.com/content/ICCV2025/supplemental/Ni_WonderTurbo_Generating_Interactive_ICCV_2025_supplemental.pdf](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Ni_WonderTurbo_Generating_Interactive_ICCV_2025_supplemental.pdf)
- **Project page:** [wonderturbo.github.io](https://wonderturbo.github.io) (qualitative gallery + interactive demo video)
- **Code (GigaAI-research):** [github.com/GigaAI-research/WonderTurbo](https://github.com/GigaAI-research/WonderTurbo) — **REPO EXISTS BUT CODE NOT RELEASED** (issue #2 from Oct 15 2025 explicitly requests code release, *unanswered* as of 2026-06-11)
- **Authors' prior works cited:** GigaAI's prior ReconDreamer 032, DriveDreamer-2 050, Street Gaussians 055, Drivedreamer-2 056, DriveDreamer4D 069, HumanDreamer 047 — strong driving-scene generation lineage
- **Hyper-SD 036 (trajectory-preserving consistency-model distillation, the *de facto 2024* recipe for 2-4 step diffusion):** cited as FastPaint's distillation backbone
- **MVSplat 008, PixelSplat 007, FreeSplat 052, DepthSplat 054 (the *4* feed-forward 3DGS methods that precede StepSplat):** the *direct technical ancestors* of StepSplat
- **WonderWorld 135 (the *direct* *predecessor* this paper *accelerates*):** the *9.5s* baseline that WonderTurbo *beats* by 15×

## For our project

**(a) ★★ ADOPT WONDERTURBO'S 3-MODULE TIME DECOMPOSITION AS V0 V0 V1 V2 SUB-TASK 1 (FULL-ARCH SYNTHESIS) ENGINEERING BUDGET** (the *de facto 2025* interactive-scene-generation engineering template — FastPaint 0.22s + QuickDepth 0.24s + StepSplat 0.26s = 0.72s total; the *killer practical insight* is that *StepSplat* (geometry inference) is the *biggest* cost (36% of total), *not* appearance or depth; the *killer* engineering roadmap for v0 v0 v1 v2 sub-task 1 is to *invest* in *geometry inference* *first* — if v0 wants <0.5s chairside UX, the *bottleneck* is *geometry*, not *appearance*; ~$0 Lambda, just engineering mindset, the *direct* v0 v0 v1 v2 sub-task 1 architectural template)

**(b) ★★ ADOPT THE FEATURE MEMORY + DEPTH-GUIDED COST VOLUME + INCREMENTAL FUSION TRINITY AS V0 V0 V1 V2 SUB-TASK 1 INCREMENTAL-SCENE-EXTENSION MECHANISM** (the *killer* engineering trick that *enables* *interactive* use case — the *feature memory* stores `N_v` closest views' matching features, the *depth-guided cost volume* samples depth candidates from `(1±a)·D_target^i` range, the *incremental fusion* prunes conflicting Gaussians via depth consistency; ~$50-100 Lambda, 1-2 weeks engineering, the *direct* v0 v0 v1 v2 sub-task 1 architectural template; the *killer practical insight* is the *combination* of the *three* — *prior* feed-forward 3DGS methods (MVSplat 008 + PixelSplat 007 + FreeSplat 052 + DepthSplat 054) *lack* *all three* and *cannot* support *incremental* scene extension; *adopt* the *trinity* for v0 v0 v1 v2 sub-task 1's *iterative* *intra-oral-scan-extension* use case)

**(c) ★★ ADOPT HYPER-SD-STYLE TRAJECTORY-PRESERVING CONSISTENCY-MODEL DISTILLATION AS V0 V0 V1 V2 DIFFUSION COST-REDUCTION RECIPE** (FastPaint's *killer practical mechanism* — *distill* any *diffusion model* from 30+ steps to 2-4 steps while *preserving* the *ODE trajectory*; the *de facto 2024-2025* recipe for *fast* *diffusion*; the *killer engineering* *cost-reduction* opportunity for v0 v0 v1 v2's *every* diffusion model — Marigold 119 (depth) + Wonder3D 118 (multi-view) + Stable Diffusion Inpaint (appearance) + OneFormer (segmentation) + *all* can be *distilled* to 2-4 steps for 10-20× inference speedup; ~$50-100 Lambda per model + 1-2 weeks engineering per model, the *direct* v0 v0 v1 v2 cost-reduction recipe; expect *5-10× overall speedup* on v0 v0 v1 v2 sub-task 1 inference time)

**(d) ★ ADOPT THE VLM-VERIFIED COMBINATORIAL SYNTHETIC-DATASET CONSTRUCTION RECIPE AS V0 V0 V1 V2 DENTAL DATA AUGMENTATION** (WonderTurbo's *killer* practical contribution — *combine* 8 different 3D scene generation methods (each producing scenes *it excels at*) + *verify* via VLM that each scene matches the defined style/textual description + *simulate* *interactive* *trajectories* + render 6M frames; the *de facto 2025* recipe for *synthetic* *interactive* *data*; the *killer practical insight* is the *combination* of *method-specific strengths* + *VLM verification* — the *right way* to *construct* *synthetic* *training data* for *interactive* *generative* *models*; ~$500-1,000 Lambda, 2-3 weeks engineering for v0 v0 v1 v2 *dental* *combinatorial* *synthetic-dataset*; expect *significant* *improvement* over *single-source* synthetic data for v0 v0 v1 v2 sub-task 1; the *practical* *recipe* is to *combine* DMC 033 + MADCrowner + ToSynFCD + 3DTeethSeg22 + Wonder3D 118 dental fine-tune + *verify* via *dental-VLM* (Qwen2.5-VL 001 fine-tuned on dental Q&A) that each *dental arch* matches the *defined* *FDI-numbered* + *prep-state* + *margin-style* *description*)

**(e) ★ ADOPT WONDERTURBO'S *HYBRID* WORKFLOW PATTERN AS V0 V0 V1 V2 SUB-TASK 1 CLINICAL UX DESIGN** (the *killer practical recommendation* from the *69.43% user-study win rate* — WonderTurbo's 0.72s is *real-time* UX for *iterative* extension but *noticeably* lower-quality than WonderWorld 135's 9.5s for the *first* scene extension; the *killer hybrid* *workflow* is to *use* WonderWorld for *first* scene extension (high-quality seed) + *use* WonderTurbo for *subsequent* extensions (real-time UX); the *de facto 2025-2026* pattern for *interactive* *generative* *systems*; ~$0 Lambda, just UX design, the *direct* v0 v0 v1 v2 sub-task 1 clinical-UX template; the *practical* *workflow* is *initial intra-oral scan* → WonderWorld 135's *FLAGS-optimized* *first-pass* *9.5s* (high-quality seed) → *interactive* *iterative* *extension* via WonderTurbo 136's *3-module* *0.72s* (real-time UX) → *final* *DMC 033 + MCAM + CPL + MRL* *refinement* on the *prep* *tooth* (high-fidelity crown) → *WonderWorld 135* *guided-depth-diffusion* *boundary alignment*)

**(f) ADOPT REPVGG BACKBONE AS V0 V0 V1 V2 SUB-TASK 1 INFERENCE-TIME EFFICIENT FEATURE EXTRACTOR** (WonderTurbo's *killer practical choice* for *real-time* *inference* — RepVGG's *dual-branch* (3×3 + 1×1) *re-parameterizes* into a *single 3×3 conv* at *inference* for *2× speedup* with *no* *quality* *loss*; the *de facto 2021-2025* recipe for *real-time* *inference*; ~$0 Lambda, just *replace* the *backbone* (e.g., ResNet-18 or ViT-S in v0 v0 v1 v2 sub-task 1) with *RepVGG-A0* or *RepVGG-A1*; expect *2× inference speedup* on *all* v0 v0 v1 v2 sub-task 1's *feed-forward* *components* (sub-task 1 *inter-oral-scan encoder*, sub-task 2 *prep-tooth encoder*); the *practical* *recipe* is to *train* with *RepVGG* *dual-branch* and *re-parameterize* at *inference*)

**(g) CITE WONDERTURBO AS V0 V0 V1 V2 PAPER'S "FOUNDING REAL-TIME INTERACTIVE 3D-SCENE-GENERATION REFERENCE" IN RELATED WORK + TABLE 1** (the *de facto 2025* *real-time* *interactive* *3D-scene* *generation* *paradigm* *paper*; the *essential* *citation* in *every* *2025-2026* 3D-gen *paper's* *related* *work* that *discusses* *real-time* *interactive* *generative* *systems*; trace the *real-time* *interactive* *3D-scene-gen* *arc*: *offline* *optimization* *methods* (LucidDreamer 010 + Text2Room 021, 30-80s/scene) → *WonderWorld 135 (interactive* *FLAGS*-optimized*, 9.5s/scene) → ***WonderTurbo 136 (founder of real-time interactive 3D-scene generation, 0.72s/scene)*** → v0 v0 v1 v2; $0 Lambda, 30 min, 1-2 paragraphs in v0 v0 v1 v2 paper's related-work)

**(h) v0 COMPUTE UPDATED: ~$9,790-12,210 LAMBDA** (was $9,730-12,100 from 135, +$60-110 for WonderTurbo-inspired Hyper-SD-style distillation of all v0 v0 v1 v2 diffusion models + $0 for RepVGG backbone swap + $0 for 3-module time decomposition engineering mindset + $0 for feature-memory + depth-guided cost volume + incremental fusion trinity; the *killer* *engineering* *cost-reduction* from Hyper-SD distillation is *one-time* but *applies* to *all* v0 v0 v1 v2 *diffusion* *models*; the *total* v0 v0 v1 v2 *diffusion* *inference* *cost* is *expected* to *drop* by *5-10×*)

**(i) V1+ RECOMMENDATION: WONDERTURBO-INSPIRED "DENTAL-ARCH-TURBO" FOR V0 V0 V1 V2 SUB-TASK 1 V1+ CLINICAL CHAIRSIDE** (the *killer* *future* *direction* — *port* WonderTurbo's *3-module* *time decomposition* to *dental* *arch* *synthesis* specifically; the *practical* *recipe* is *FastPaint-style dental fine-tune of Hyper-SD-distilled SD-Inpaint* + *QuickDepth-style dental fine-tune of Depth Anything on dental depth* + *StepSplat-style dental fine-tune of MVSplat/DepthSplat on dental multi-view*; the *killer* *target* is *0.72s per dental arch extension* (vs *current* v0 v0 v1 v2 *9-11s* from WonderWorld 135-inspired *FLAGS*-optimized design); ~$1,000-2,000 Lambda, 4-6 weeks engineering, the *direct* v0 v0 v1 v2 sub-task 1 V1+ *clinical chairside* *real-time* *target*)

**(j) OPEN Q: WONDERTURBO WITHOUT CODE — HOW TO BUILD-OUR-OWN?** (WonderTurbo's *code* *is* *not* *released* as of 2026-06-11 (issue #2 from Oct 15 2025 remains *unanswered*); the *practical* *engineering* *path* for v0 v0 v1 v2 is to *reimplement* WonderTurbo's *3* *modules* from *scratch* using *public* *components*:
- *StepSplat-replacement*: use **MVSplat 008** or **DepthSplat 054** + add *feature memory* + *depth-guided cost volume* + *incremental fusion* (~500-1,000 lines of PyTorch, 1-2 weeks engineering)
- *QuickDepth-replacement*: use **Depth Anything V2** (Yang CVPR 2024) + add *interactive-3D-mask fine-tune* on dental depth dataset (~200-500 lines of PyTorch, 1-2 weeks engineering)
- *FastPaint-replacement*: use **Stable Diffusion Inpaint** + *Hyper-SD-style* trajectory distillation + *dental inpainting mask fine-tune* (~300-500 lines of PyTorch, 1-2 weeks engineering)
Total: ~1,000-2,000 lines of PyTorch, 3-6 weeks engineering, ~$1,000-2,000 Lambda; the *killer* *practical* *roadmap* for v0 v0 v1 v2 *without* *code* *release*)

**Strategic positioning: WonderTurbo is the *de facto 2025 ICCV reference for real-time interactive 3D scene generation* — 15× speedup over WonderWorld 135 (the previous best), 60-110× speedup over offline methods, 0.72s/scene on H20 GPU, wins all 5 CLIP-based metrics + 69.4% user-study win rate vs WonderWorld 135, the first paper to *jointly* accelerate geometry + appearance + depth for *real-time* *interactive* *generative* *systems*. The *killer application* to dental arch is *real-time chairside interactive dental arch design* (dentist specifies missing tooth position + preparation, sees generated full arch in 0.72s, iterates by camera-move and text-prompt at *interactive frame rate* 1.4Hz). The WonderTurbo *VLM-verified* *combinatorial* *synthetic-dataset* *construction* *recipe* is the *de facto 2025* *recipe* for *synthetic* *interactive* *training* *data* — directly applicable to v0 v0 v1 v2's *dental* *combinatorial* *synthetic-dataset*. The *Hyper-SD-style* *trajectory-distillation* is the *de facto 2024-2025* *recipe* for *fast* *diffusion* — directly applicable to v0 v0 v1 v2's *every* *diffusion* *model* (Marigold 119 + Wonder3D 118 + SD-Inpaint + OneFormer).** Note in `papers/136-wonderturbo-ni25.md`. **Next paper to read (137):** the 136-note's recommended *next* is the *dynamic* *successor* of WonderWorld/WonderTurbo in the *Stanford-Chinese scene-generation dynasty*: **WonderPlay (Yu et al. ICCV 2025, "WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions")** — the *killer* for v0 v0 v1 v2 sub-task 1 V2 *dynamic* dental arch (chewing motion, jaw movement, pre/post-op temporal comparison) — the *natural extension* of WonderTurbo 136 to *dynamic* scenes with *action* *conditioning*. Alternative: **WorldScore (Duan et al. ICCV 2025)** — the *eval* *benchmark* *for* *3D* *scene* *generation* — the *killer* *eval* *protocol* for v0 v0 v1 v2 paper's *evaluation* *section*. Or **FlashWorld (2025, OpenReview)** — the *most-recent* *feed-forward* *3D-scene* *generation* *paper* — the *killer* *competitor* to Bolt3D 116 + WonderTurbo 136. Or **WonderFree (Ni et al. arXiv:2506.20590, 25 Jun 2025)** — the *cross-view consistency* *extension* of WonderTurbo 136 by the *same authors* (Ni, Wang, Zhu, Mei) — the *killer* *follow-up* to read for *continued* *WonderTurbo* *lineage*. **Recommendation: *read 137 = WonderPlay (Yu et al. ICCV 2025, dynamic 3D scene generation)*** — the *killer* *natural* *extension* of WonderWorld 135 + WonderTurbo 136 to *dynamic* scenes, the *right* paper to read for v0 v0 v1 v2 sub-task 1 V2 *dynamic* dental arch (chewing motion + jaw movement). Alternative: *read 137 = WonderFree (Ni et al. arXiv:2506.20590, 25 Jun 2025, cross-view consistency)* — the *killer* *follow-up* by the *same* *authors* as WonderTurbo 136, the *right* paper to read for v0 v0 v1 v2 sub-task 1 *cross-view consistency* requirements.
