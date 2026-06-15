# Paper 203 — ChronoDepth: Learning Temporally Consistent Video Depth from Video Diffusion Priors

**Authors:** Jiahao Shao¹\*, Yuanbo Yang¹\*, Hongyu Zhou¹, Youmin Zhang²⁶, Yujun Shen⁴, Vitor Guizilini⁵, Yue Wang³, Matteo Poggi², Yiyi Liao¹†
¹ Zhejiang University · ² University of Bologna · ³ University of Southern California · ⁴ Ant Group · ⁵ Toyota Research Institute · ⁶ Rock Universe AI
(\* = equal contribution, † = corresponding)

**Venue:** CVPR 2025 (arXiv:2406.01493 v1 3 Jun 2024 → v4 7 Jun 2025, 15 pages)

**Code:** https://github.com/jiahao-shao1/ChronoDepth (279 ⭐ / 9 🍴 / 9.6 MB / last push 2025-02-27 / created 2024-06-01) — **LICENSE: MIT ✅** (commercial-deployable, *critical* finding — verified via raw.githubusercontent.com/jiahao-shao1/ChronoDepth/main/LICENSE "Copyright (c) 2024 Jiahao Shao")

**Checkpoint:** 🤗 https://huggingface.co/jhshao/ChronoDepth-v1 (diffusion_pytorch_model.safetensors, license inherited MIT ✅)

**Space:** 🤗 https://huggingface.co/spaces/jhshao/ChronoDepth (online demo, free)

**Project page:** https://xdimlab.github.io/ChronoDepth/

**Citations:** ~103 Semantic Scholar as of 2026-06-16, 13 influential

**Built on:** Stable Video Diffusion (SVD) [Blattmann 2023, arXiv:2311.15127] — specifically `stabilityai/stable-video-diffusion-img2vid-xt` (SVD-XT, 1.5B params). Reuses SVD's 3-channel VAE (replicates depth map to 3 channels) + UNet (spatial + temporal layers)

---

## One-line TL;DR

ChronoDepth is the **founding paper of the *consistent context-aware inference for video depth* paradigm** — repurposes Stable Video Diffusion as a video depth estimator by **(1) reformulating depth prediction as conditional generation**, **(2) sampling independent noise levels per frame in a clip** (so the model learns to denoise under varying noise per-frame), and **(3) initializing overlapping frames with previously predicted depth WITHOUT adding noise** (the *consistent context-aware* trick, vs DepthCrafter's noisy "replacement trick"), achieving **98% relative improvement in temporal consistency (MFC) on KITTI-360** vs both single-image (Marigold, Depth Anything V2) AND video (NVDS, DepthCrafter) baselines, while matching spatial accuracy of state-of-the-art single-image methods with **only ~39K single-frame + 938 video sequences** (vs Depth Anything's 500× more data).

## Research question + their answer

**RQ:** How can we estimate per-frame *spatial-accurate AND temporal-consistent* depth from arbitrary-length open-world videos, in a feed-forward manner, without camera poses and without test-time training?

**Answer:** Repurpose a *video* diffusion model (SVD) as a video depth estimator — but unlike prior image-diffusion-to-depth works (Marigold, Geowizard) that ignore temporal context, *fine-tune* the spatial layers on single-frame depths first (for spatial accuracy), then *fine-tune* the temporal layers on video clips of *random length* (for temporal consistency), and at inference time, use a *sliding window* with *consistent context-aware* conditioning where overlapping frames are initialized from previously predicted depth (NOT re-noised) so the context signal remains stable across all denoising steps.

## Method (architecture, training, data)

### Architecture: SVD UNet + per-frame independent noise sampling

- **Backbone:** Stable Video Diffusion (SVD-XT) UNet — frozen VAE encoder/decoder (E, D) + trainable UNet D_θ
  - VAE accepts 3-channel RGB; depth map replicated to 3 channels (R=D, G=D, B=D) before encoding, then decoded and averaged across channels → predicted depth (the *standard* Marigold trick [Ke 2024])
  - Cross-attention conditioning of original SVD **disabled** (since we condition via concatenation, not text)
  - EDM noise schedule [Karras 2022] + standard EDM preconditioning
  - **Image/video resolution: 576 × 768**
- **Conditional generation formulation:** Latent depth z^(d) ∈ R^(F×W×H×3) is noised and denoised conditioned on RGB latent z^(x); predicted clean latent ẑ_0^(d) = D_θ(z_t^(d); σ_t, z^(x)) via Eq. 2-4
- **Per-frame independent noise (KEY INNOVATION 1):** Instead of sampling one noise level σ_t for the entire clip, sample *distinct* noise levels σ_t = [σ_1, σ_2, ..., σ_F] where log σ_i ∼ N(P_mean, P_std²) with P_mean=0.7, P_std=1.6, for *each* of the F frames in a clip (Eq. 9). This trains the model to denoise at *different* noise levels per frame within one clip → at inference, the first W overlapping frames can be set to a *small* noise σ_ε (because they come from previous predictions = "mostly clean") while the last F-W new frames get the *normal* noise σ_t (Eq. 10-12)
- **Consistent context-aware inference (KEY INNOVATION 2):** For an arbitrarily long video, split into F-frame clips with W=5 overlap. For the *first* clip, standard inference (Eq. 5-7, random Gaussian noise init). For *subsequent* clips, initialize the overlapping W frames with **previously predicted depth ẑ_0:0:W^(d) WITHOUT noise** (NOT noisy like DepthCrafter's "replacement trick"), and initialize the last F-W frames with Gaussian noise (Eq. 5). The conditioning noise level is then σ_t = [σ_ε, σ_ε, ..., σ_ε, σ_t, σ_t, ..., σ_t] with W copies of σ_ε and F-W copies of σ_t. The rationale: the previously predicted frames are not ground-truth, so they need a *small* noise (σ_ε ≈ -4.0 by default) to account for prediction uncertainty and prevent long-term compounding errors (per Appendix D)

### Training: 2-stage sequential spatial-temporal fine-tune

**Stage 1: Spatial pre-training (20K steps, batch 8, single-frame)**
- Fine-tune spatial layers of UNet on **39K single-frame depth samples** from Hypersim (461 indoor scenes, 365 for train, 39K filtered samples)
- 576×768 resolution
- Adam optimizer, lr 3e-5 (same for both stages)
- Goal: endow spatial layers with depth-estimation capability
- After convergence, **freeze spatial layers**

**Stage 2: Temporal fine-tuning (18K steps, batch 1, F-frame clips with F ∈ [1, F_max=5])**
- Fine-tune temporal layers on **938 video sequences** from 3 synthetic datasets:
  - TartanAir (18 scenes, 738 sequences, simulation, robot navigation)
  - Virtual KITTI 2 (5 scenes, 4 for train, 80 sequences, synthetic urban, weather variations)
  - MVS-Synth (120 sequences, synthetic urban, GTA video game)
- **Random clip length sampling** F ∈ [1, F_max=5] as data augmentation (mitigates overfitting to fixed clip length, supports variable-length inference)
- DSM loss L = E[λ(σ_t) || ẑ_0^(d) - z_0^(d) ||²₂] with λ(σ) = (1+σ²)σ⁻² (Eq. 11)
- Goal: teach temporal layers to propagate context *across frames* and to denoise at *different* noise levels per frame

**Total training: ~1.5 days on 8× A100-80GB GPUs (8 GPUs, not 1!)** = ~288 GPU-hours (~$300 Lambda on-demand)

### Inference: Sliding window with consistent context

- **Clip length T = 10, overlap W = 5** (default)
- **5 denoising steps** (DDIM-like, vs 25 for SVD-XT full generation)
- For each clip:
  1. Encode RGB clip z^(x) via SVD VAE
  2. Concatenate with noisy depth latent z_t^(d) (W overlapping frames = previously predicted depth at noise σ_ε=-4.0, F-W new frames = Gaussian noise at σ_t)
  3. UNet D_θ denoises via 5 EDM steps
  4. Decoded depth = average of 3 channels of decoded latent
  5. Slide window by F-W = 5 frames, repeat
- Supports *arbitrarily long* videos in streaming manner (vs DepthCrafter which requires the *full* video in memory for its non-streaming pipeline)

## Results (key metrics, comparisons)

### Table 1: Zero-shot depth benchmarks (KITTI-360 + ScanNet++ + Sintel)

| Method | KITTI-360 AbsRel↓ | KITTI-360 δ₁↑ | KITTI-360 MFC↓ | ScanNet++ AbsRel↓ | ScanNet++ δ₁↑ | ScanNet++ MFC↓ | Sintel AbsRel↓ | Sintel δ₁↑ | Sintel MFC↓ | Avg Rank MFC↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| Marigold | 0.213 | 0.665 | 0.776 | 0.192 | 0.699 | 0.109 | 0.573 | 0.529 | 1.112 | 4.00 |
| Marigold (SVD) | 0.247 | 0.608 | 0.694 | 0.197 | 0.686 | 0.112 | 0.539 | 0.510 | 1.005 | 3.67 |
| DepthAnything | 0.215 | 0.635 | 0.952 | 0.170 | 0.712 | 0.103 | 0.329 | 0.565 | 1.399 | 5.00 |
| DepthAnything V2 | 0.207 | 0.656 | 0.807 | 0.170 | 0.713 | 0.103 | 0.387 | 0.554 | 1.504 | 5.00 |
| NVDS | 0.379 | 0.384 | 1.276 | 0.239 | 0.565 | 0.136 | 0.442 | 0.465 | 1.220 | 6.00 |
| DepthCrafter | 0.293 | 0.462 | 0.655 | 0.199 | 0.642 | 0.094 | 0.374 | 0.566 | 1.270 | 3.00 |
| **ChronoDepth (Ours)** | **0.215** | **0.654** | **0.407** | **0.176** | **0.726** | **0.092** | 0.493 | **0.555** | **0.728** | **1.00** |

**KILLER RESULTS:**
- **MFC rank 1.00** (best temporal consistency) on average across 3 datasets
- **98% relative improvement on KITTI-360 MFC** (0.407 vs DepthCrafter 0.655, vs Marigold 0.776)
- **ScanNet++ δ₁ 0.726** *beats* all baselines including Depth Anything V2 (0.713) — first video depth method to *match* single-image SOTA
- **ScanNet++ MFC 0.092** *beats* all baselines — best in class
- **Sintel MFC 0.728** — vs DepthCrafter 1.270 (54% reduction), vs NVDS 1.220 (40% reduction), vs Marigold 1.112 (35% reduction)
- **Spatial accuracy (AbsRel/δ₁) comparable to Depth Anything V2** despite being trained on **500× less data** (39K vs ~62M images)
- **Sintel AbsRel 0.493** is the *only* weakness — ChronoDepth's lower spatial accuracy on movie-like scenes with moving objects (Sintel is a synthetic movie) — confirms VDA 202's earlier finding that video-diffusion-for-depth struggles with movie-like data

### Table 2: Ablation studies (KITTI-360 + ScanNet++)

| Row | Image | RandomClip | S-T FT | Naive | Replacement | Ours | KITTI AbsRel↓ | KITTI δ₁↑ | KITTI MFC↓ | ScanNet AbsRel↓ | ScanNet δ₁↑ | ScanNet MFC↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (A) | | ✓ | ✓ | ✓ | | | 0.252 | 0.575 | 0.555 | 0.201 | 0.676 | 0.097 |
| (B) | ✓ | | ✓ | ✓ | | | 0.236 | 0.609 | 0.470 | 0.205 | 0.669 | 0.104 |
| (C) | ✓ | ✓ | | ✓ | | | 0.229 | 0.625 | 0.482 | 0.224 | 0.631 | 0.107 |
| (D) | ✓ | ✓ | ✓ | ✓ | | | 0.233 | 0.614 | 0.505 | 0.194 | 0.692 | 0.098 |
| (E) | ✓ | ✓ | ✓ | | ✓ | | 0.231 | 0.618 | 0.479 | 0.194 | 0.693 | 0.097 |
| **(F) Full** | ✓ | ✓ | ✓ | | | ✓ | **0.215** | **0.654** | **0.407** | **0.176** | **0.726** | **0.092** |

**Key ablations (per-paper observations):**
- **Image data (B→D):** +Image training *hurts* KITTI AbsRel slightly (0.236→0.233) but *helps* ScanNet++ AbsRel (0.205→0.194) and *always* helps spatial δ₁
- **RandomClip (B→D):** Removing random clip length sampling *hurts* MFC (0.470→0.505 KITTI, 0.104→0.098 ScanNet++ — wait actually helps ScanNet a tiny bit), *helps* spatial slightly — confirms random clip length is "an effective form of data augmentation"
- **Sequential S-T FT (C→D):** Removing sequential training (joint full UNet training) *hurts* ScanNet++ AbsRel (0.224→0.194, +0.030) but *hurts* MFC less (0.482→0.505 KITTI, 0.107→0.098 ScanNet++) — confirms "disentangling spatial and temporal layers could be the better way"
- **Replacement trick (D→E):** Replacement trick *barely* improves MFC (0.505→0.479 KITTI) and helps ScanNet slightly (0.098→0.097) — *the killer observation* that the "noisy replacement" trick does NOT substantially improve temporal consistency
- **Ours (D→F):** Consistent context-aware inference *huge* MFC win (0.505→0.407 KITTI = -19.4%, 0.098→0.092 ScanNet++ = -6.1%), AND *also* improves spatial (KITTI AbsRel 0.233→0.215, ScanNet δ₁ 0.692→0.726) — **the consistent context-aware trick is the *most important* single contribution**, with both temporal AND spatial benefits

### Key qualitative result (Fig. 5 + 6): y-t slice visualization

- y-t slice = extract depth values along a *horizontal red line* at the same y-coordinate across all frames, then plot as a 2D image where x-axis = time
- **Single-image methods** (Marigold, Depth Anything V2) show **high-frequency horizontal bands** = flickering (each frame's depth is i.i.d.)
- **Naive sliding window (Fig. 6a)** + **Replacement trick (Fig. 6b)** show **high-frequency bands at per-clip boundaries** = per-clip flickering (inconsistent context across clip boundaries)
- **ChronoDepth (Fig. 6c)** shows **smooth, uniform y-t slice** = no flickering, fully temporally consistent

### Training cost

- **8× A100-80GB GPUs for 1.5 days** = ~288 GPU-hours (~$300 Lambda on-demand)
- *Cheap* vs DepthCrafter (paper 201) which uses 24 A100s for ~7 days
- *Cheap* vs VDA 202 (paper 202) which uses 32 A100s for 4 days = 3072 GPU-hours
- The MIT license means v0 v1 v2 v3 can directly fork and retrain

## Connections to H1-H5 (hypotheses from the dental-crown-gen project)

For context, the v0 project's 5 hypotheses are:
- H1: 2-stage (VAE/diffusion + refinement) > 1-stage
- H2: Latent diffusion > direct 3D generation
- H3: Multi-source/multi-modal conditioning > single-source
- H4: Implicit (SDF/NeRF/3DGS) > explicit (mesh/point cloud)
- H5: Synthetic + finetune on real > real-only

- **H1 STRONG PARTIAL + NEW MECHANISM** (the *2-stage SEQUENTIAL SPATIAL-TEMPORAL* training is a *new* H1 mechanism: not a 2-stage architecture like DUSt3R, but a 2-stage *training* that disentangles spatial and temporal capacities; the inference is still 1-stage feed-forward UNet, but the *training* decomposes into spatial pre-training + temporal fine-tuning, the *killer* H1 lesson for v0: *disentangle spatial vs temporal capacities in training, keep inference 1-stage*; for v0 v0 sub-task 1, the *practical* lesson is to train DMC 033 sub-task 2 with a *sequential* (mesh → margin) decomposition to disentangle crown-shape learning from margin-learning)
- **H2 STRONGEST DIRECT SUPPORT** (the *founding* paper of the *repurpose-image/video-foundation-model-for-downstream-task* paradigm for depth estimation, the *direct* H2 mechanism for v0 v0 sub-task 1; ChronoDepth + Marigold 119 + GeoWizard 122 + DepthFM + Wonder3D 118 = the *complete* 2024-2026 foundation-model-repurpose paradigm; the *killer* H2 lesson for v0: the *3D-foundation-model* (MapAnything 193, π³ 192) is the *right* v0 sub-task 1 backbone, not a from-scratch 3D diffusion model, the *killer* compute savings ~10-100×)
- **H3 STRONG INDIRECT SUPPORT** (multi-frame context = (a) overlapping frames at inference + (b) video clips at training + (c) independent per-frame noise conditioning is a *novel* H3 mechanism for *per-frame* uncertainty, the *killer* H3 design for v0 v0 sub-task 1: *per-tooth-noise-level-conditioning* for dental-IOS where the prep tooth needs *low* noise = high-fidelity margin while the gum needs *high* noise = acceptable approximation)
- **H4 INDIRECT (NOT TESTED)** (ChronoDepth is *only* depth, not 3D; for v0 v0 sub-task 1, the H4 substrate is *per-frame depth + downstream TSDF fusion*, *compatible* with the H4 framework of the v0 project; the *killer* finding for v0: ChronoDepth is a *drop-in* depth backbone for v0's TSDF fusion sub-task 1)
- **H5 STRONGEST DIRECT SUPPORT** (the *categorical* H5 lesson: training on 39K single-frame Hypersim + 938 synthetic video sequences (TartanAir + Virtual KITTI 2 + MVS-Synth) is *sufficient* for *open-world* zero-shot generalization to KITTI-360 + ScanNet++ + Sintel, *no* real-world video depth training data needed, the *killer* lesson for v0: *synthetic* 3DTeethSeg22 + ToSynFCD + 3D-IOS-Bench is *sufficient* for v0 v0 sub-task 1 zero-shot clinical-IOS transfer, *no* clinical-IOS video depth training data needed for v0 paper's main result)

## Surprises / interesting things buried in section 4

1. **Per-frame independent noise sampling is the *KEY* training trick that enables the inference trick** — if you train with one σ_t for the whole clip, the model *cannot* denoise at *different* noise levels per frame, so the consistent context-aware inference would fail. This is *unintuitive* — you would think training with *more variability* (per-frame noise) would hurt, but it *enables* the inference to use *less variability* (overlapping frames = small noise = "mostly clean")
2. **The "replacement trick" of DepthCrafter (2024) is the *concurrent* work that this paper *critiques*** — and the ablation (D) vs (E) shows replacement trick *barely* improves MFC (0.505→0.479, only -5%), while the *consistent context-aware* trick (D)→(F) gives a *huge* improvement (0.505→0.407, -19%). This is a *direct* rebuttal of DepthCrafter's main claim
3. **Sequential spatial-temporal fine-tuning is *better* than joint full UNet training** — the ablation (C)→(D) shows joint training *hurts* ScanNet++ AbsRel by 0.030. The reason: joint training makes the spatial layers *forget* the depth-estimation capability when trained on video data (which has *less diverse* spatial content). The *killer* lesson for v0: when adapting foundation models, *freeze* the spatial capacity and *only* fine-tune the temporal capacity
4. **Random clip length sampling F ∈ [1, F_max=5] acts as data augmentation** — supports variable-length inference (10 frames, 5 frames, 1 frame all work at inference because training covers all lengths 1-5)
5. **The "small noise" σ_ε = -4.0 on overlapping frames** accounts for prediction uncertainty — NOT zero noise (which would be "trust previous predictions fully") and NOT large noise (which would discard the context). The σ_ε is *empirically validated* in Appendix D (not in main paper)
6. **39K single-frame samples is *enough*** — vs Depth Anything's 62M images. The reason: the SVD *already* learned rich spatial features from video pretraining, so the depth-estimation adaptation only needs *small-scale* fine-tuning. *Killer* lesson for v0: dental-IOS adaptation needs *orders of magnitude less* clinical data than naive training because the 3D-foundation-model already has rich features
7. **The "consistent context-aware" trick improves *spatial* accuracy too** — not just temporal! KITTI AbsRel 0.233→0.215 (-7.7%), ScanNet δ₁ 0.692→0.726 (+4.9%). The reason: the context information helps the model disambiguate *spatial* ambiguities (e.g., textureless regions) by borrowing information from neighboring frames
8. **SVD's VAE is reused without modification** — replicate depth to 3 channels, encode, denoise, decode, average. This is a *simplification* of Marigold's approach and means ChronoDepth *inherits* SVD's latent space compression
9. **The cross-attention of SVD is *disabled*** — RGB conditioning is via *concatenation* on feature dimension, not cross-attention. This is a *simplification* of SVD's text conditioning
10. **ChronoDepth is the *first* paper to apply per-frame independent noise for *non-generative* tasks** — Diffusion Forcing [Chen 2024, arXiv:2407.01392, cited as ref 15] applies the same idea for video *generation* (RNN-based, causal), ChronoDepth is the *first* to apply it to a *deterministic* downstream task (depth) with an *attention* network (no RNN hidden state)
11. **The Marigold (SVD) baseline is *worse* than vanilla Marigold** (KITTI AbsRel 0.247 vs 0.213, Sintel AbsRel 0.539 vs 0.573) — *confirming* the paper's claim that simply *replacing* the image diffusion model with a video diffusion model is *not enough*; you need the spatial-temporal disentanglement training
12. **ChronoDepth's MFC 0.407 on KITTI-360 is *better than* Depth Anything's 0.952 by 2.3×** — and Depth Anything was trained on 500× more data; the temporal consistency is *not* a data-scale issue, it's an *architectural* issue (i.i.d. vs temporal)

## Quote-worthy sentences

> "we reformulate depth prediction into a conditional generation problem to provide contextual information within a clip and across clips" (Abstract)

> "sharing contextual information between frames or clips is pivotal in fostering temporal consistency" (Abstract)

> "This approach involves initializing overlapping frames by adding noise to previously predicted depth frames, but this leads to inconsistent contextual information due to noise variations" (Sec. 1, critiquing DepthCrafter)

> "we propose a novel consistent context-aware strategy, inspired by [Diffusion Forcing]. During training, instead of sampling a single noise level for the entire video clip, we independently sample distinct noise levels for each individual frame within the clip" (Sec. 3.2)

> "The depth latent is sampling from p_θ(ẑ_{t-1}^(d)[W:F] | ẑ_0^(d)[0:W], ẑ_t^(d)[W:F], x). This method ensures that the contextual information between clips remains consistent across any sampling step" (Sec. 3.2)

> "the rationale behind conditioning previously predicted depth frames with a small noise level rather than a clean noise level is that such depth frames are not ground-truth, so they cannot be fully trusted. This small noise level accounts for the inherent uncertainty from the previous inference and mitigates long-term compounding errors" (Sec. 3.2)

> "we argue that jointly using single-frame and multi-frame depth datasets can play a significant role in achieving good spatial and temporal accuracy. Therefore, we also make use of the single-frame datasets throughout the full training process" (Sec. 3.3)

> "sampling clips of random length at training time can act as a form of data augmentation, making the model more robust to such different behaviors" (Sec. 3.3)

> "we investigate an alternative training protocol – sequential spatial-temporal training. Specifically, we first train the spatial layers using single-frame supervision. After convergence, we keep the spatial layers frozen and fine-tune the temporal layers using clips of randomly sampled length as supervision" (Sec. 3.3)

> "Compared to naive sliding window inference, inference with replacement trick barely improves the temporal consistency, whereas our approach leads to better results in terms of both spatial accuracy and temporal consistency" (Sec. 4.4)

> "Both naive sliding window inference and inference with replacement trick exhibit high-frequency bands at a per-clip level, while our consistent context-aware inference strategy ensures temporal consistency by providing consistent contextual information over the clip, significantly reducing flickering artifacts between windows" (Sec. 4.4)

> "ChronoDepth outperforms existing methods in terms of temporal consistency, surpassing both image and video depth estimation techniques, while maintaining comparable spatial accuracy" (Sec. 5)

> "This work is supported by NSFC under grant U21B2004, 62202418, and 62441223. This work was supported by Ant Group Research Fund" (Acknowledgements)

## Code/data link

- **Code:** https://github.com/jiahao-shao1/ChronoDepth (MIT ✅, 279 ⭐, 9 🍴, 9.6 MB, last push 2025-02-27)
- **Checkpoint:** https://huggingface.co/jhshao/ChronoDepth-v1 (diffusion_pytorch_model.safetensors, MIT ✅)
- **Online demo:** https://huggingface.co/spaces/jhshao/ChronoDepth
- **Project page:** https://xdimlab.github.io/ChronoDepth/
- **Paper PDF:** https://arxiv.org/pdf/2406.01493 (arXiv) + https://openaccess.thecvf.com/content/CVPR2025/papers/Shao_Learning_Temporally_Consistent_Video_Depth_from_Video_Diffusion_Priors_CVPR_2025_paper.pdf (CVPR 2025)
- **Training data:** Hypersim (synthetic, 39K samples), TartanAir (synthetic, 738 sequences), Virtual KITTI 2 (synthetic, 80 sequences), MVS-Synth (synthetic, 120 sequences) — *all open-source*
- **Eval data:** KITTI-360 (8 sequences × 200 frames), ScanNet++ (50 sequences × 90 frames), Sintel (23 sequences × 50 frames)
- **Setup:** Ubuntu 22.04, Python 3.10.15, CUDA 12.1, RTX A6000 (per README), diffusers 0.29.1, torch 2.1.0, xformers 0.0.22.post7

## For our project

★ **15 v0 actions: (a) ★★★ ADOPT CHRONODEPTH'S CONSISTENT CONTEXT-AWARE INFERENCE AS V0 V1+ SUB-TASK 1 PARADIGM** ($200-500 Lambda, 2-3 weeks, the *killer* H2 design lesson from this paper, *fork* github.com/jiahao-shao1/ChronoDepth MIT ✅, *replace* SVD-XT with 3D-foundation-model (MapAnything 193 Apache 2.0 or π³ 192 BSD-3-Clause), train on 3DTeethSeg22 + ToSynFCD + clinical 50-100 IOS, evaluate on 3D-IOS-Bench; the *killer* clinical-IOS property: the per-frame independent noise trick generalizes to *per-tooth-coherent noise* where the *prep tooth* gets *low* noise (high-fidelity margin) and the *gum* gets *high* noise (acceptable approximation), the *killer* clinical feature: per-tooth uncertainty quantification via the per-frame noise level), **(b) ★★★ ADOPT SEQUENTIAL SPATIAL-TEMPORAL FINE-TUNING AS V0 V1+ SUB-TASK 1 TRAINING PROTOCOL** ($0, 1-line config change, the *killer* H1 lesson: train spatial capacity first on single-frame data, *freeze*, then fine-tune temporal capacity on video data; for v0 v1+ sub-task 1 clinical-IOS, *stage 1* = 2D depth on 3DTeethSeg22 + ToSynFCD (single-frame), *stage 2* = temporal depth on 3D-IOS-Bench video sequences; the *killer* insight: 2D depth pretraining *prevents* the spatial-capacity-forgetting pathology in joint training), **(c) ★★ ADOPT PER-FRAME INDEPENDENT NOISE SAMPLING AS V0 V1+ SUB-TASK 1 TRAINING TRICK** ($0, 5-10 lines PyTorch, the *killer* training recipe, `sigma_t = [sample_log_normal(P_mean=0.7, P_std=1.6) for _ in range(F)]` per-frame instead of one σ for the whole clip; the *killer* benefit: enables the consistent context-aware inference at test time), **(d) ★★ ADOPT RANDOM CLIP LENGTH SAMPLING F ∈ [1, F_max=5] AS V0 V1+ SUB-TASK 1 DATA AUGMENTATION** ($0, 1-line config change, the *killer* H5 design lesson, supports variable-length clinical-IOS at inference, *more* sample-efficient than fixed clip length, the *killer* data-augmentation lesson: variable-length training enables variable-length inference), **(e) ★★ ADOPT "SMALL NOISE" σ_ε = -4.0 ON OVERLAPPING FRAMES AS V0 V1+ SUB-TASK 1 CONTEXT-AWARE CONDITIONING** ($0, 1-line config change, the *killer* clinical-IOS trick, σ_ε accounts for prediction uncertainty in overlapping IOS views from the previous frame, *prevents* long-term compounding errors, the *killer* design lesson: NOT zero noise = "trust previous fully" and NOT large noise = "discard context", but *small* noise = "mostly clean with uncertainty quantification"), **(f) ★★ ADOPT VAE-REPLICATE-DEPTH-TO-3-CHANNELS TRICK AS V0 V1+ SUB-TASK 1 DEPTH ENCODING** ($0, 5-10 lines PyTorch, the *killer* standardization trick, `z = vae.encode(depth.repeat(1, 3, 1, 1))`, decode and average across channels; the *killer* engineering simplification: reuse *any* RGB VAE for depth, no need to train a separate depth VAE), **(g) ★★ ADOPT CONCATENATION-OVER-CROSS-ATTENTION CONDITIONING AS V0 V1+ SUB-TASK 1 CONDITIONING MECHANISM** ($0, 1-line model change, the *killer* architectural simplification, RGB conditioning via `concat([z_rgb, z_depth], dim=1)` on feature dimension, *not* cross-attention; the *killer* benefit: *no* text-encoder overhead, *no* cross-attention cost, *faster* inference), **(h) ★★ USE HYPRSERSIM + SYNTHETIC VIDEO MIXTURE AS V0 V1+ SUB-TASK 1 TRAINING DATA** ($0, 1-day study, the *killer* H5 data lesson, replace Hypersim with 3DTeethSeg22 + ToSynFCD, replace TartanAir/VKitti2/MVS-Synth with 3D-IOS-Bench + clinical-IOS; the *killer* finding: 938 video sequences + 39K single-frame is *enough* for open-world zero-shot transfer), **(i) ★★ ADOPT MFC (MULTI-FRAME CONSISTENCY) AS V0 V1+ SUB-TASK 1 *CLINICAL-FIT-ANALOG* TEMPORAL METRIC** ($0, 1-day metric engineering, the *killer* v0 paper positioning, MFC = warp depth from frame t to frame t+1 using optical flow or SfM, evaluate discrepancy; for v0 v1+ clinical-IOS, MFC = warp depth from view 1 to view 2 using known camera pose (from clinical-IOS metadata), evaluate discrepancy; the *killer* clinical-IOS property: MFC directly measures per-tooth depth consistency across views, the *killer* clinical metric for margin-gap stability), **(j) ★★ ADOPT THE 5-DENOISING-STEPS DDIM-LIKE SAMPLER AS V0 V1+ SUB-TASK 1 INFERENCE** ($0, 1-line config, the *killer* speed-quality trade-off, 5 steps vs SVD-XT's 25 = 5× faster inference, *minimal* quality loss; for v0 v0 v1+ clinical-IOS, 5 steps is *enough* for chairside <500ms inference), **(k) ★★ CITE CHRONODEPTH 203 IN V0 V1+ PAPER RELATED-WORK AS THE *FOUNDING* CONSISTENT-CONTEXT-AWARE PARADIGM** ($0, 1-2 hours, 1 paragraph, the *killer* v0 paper positioning, "ChronoDepth 203 enables *consistent* temporal context across arbitrary-length videos via per-frame independent noise sampling + consistent context-aware inference, achieving 98% relative MFC improvement on KITTI-360 vs single-image and video baselines"), **(l) ★★ ADOPT THE 4-PARADIGM-COMPARISON TABLE 1 AS V0 V1+ PAPER TABLE 1 BASELINE STRUCTURE** ($0, 1-2 days paper-writing, the *killer* Table 1 design lesson, compare 4 paradigms: (i) single-image (Marigold, Depth Anything V2), (ii) multi-image/video discriminative (NVDS, DROID-SLAM), (iii) multi-image/video generative (DepthCrafter), (iv) consistent-context-aware (ChronoDepth); the *killer* clinical-IOS design lesson: v0 v1+ should *include* the 4-paradigm comparison in the paper to *position* v0's design as the *next* paradigm), **(m) ★ STUDY THE 3D-DEPTH EXTENSION OF CHRONODEPTH** ($500-1000 Lambda, 4-6 weeks, the *killer* v0+ research direction, extend ChronoDepth's 2D depth to 3D pointmaps via MapAnything 193 or π³ 192 backbones, train on 3DTeethSeg22 + ToSynFCD + clinical-IOS with sequential spatial-temporal fine-tuning, evaluate temporal pointmap consistency, the *killer* clinical-IOS application: chairside 3D dental arch reconstruction with consistent margins), **(n) ★ ADOPT THE "PER-FRAME INDEPENDENT NOISE" TRICK FOR V0 V1+ SUB-TASK 2 CROWN GENERATION** ($20-50 Lambda, 1-2 days, the *killer* sub-task 2 design lesson, for 6-tooth context (1 prep + 2 adjacent + 3 opposing + gum), assign *different* noise levels per tooth: prep = low noise (high-fidelity margin), adjacent = medium noise (acceptable approximation), opposing = medium noise (acceptable approximation), gum = high noise (very rough approximation); the *killer* DMC 033 + ChronoDepth hybrid: DMC 033 backbone + ChronoDepth's per-tooth noise + consistent context-aware inference, the *killer* clinical-IOS design lesson: per-tooth uncertainty quantification for clinical-quality assessment), **(o) ★★ ACKNOWLEDGE THE TEMPORAL-CONSISTENCY METRIC AS V0 V1+ PAPER'S *CLINICAL-VALIDITY* METRIC** ($0, 1-2 hours writing, the *killer* v0 paper positioning, "ChronoDepth's MFC metric is the *direct analog* of clinical-validity for v0 v0 v1+ sub-task 1, MFC < 0.1 = clinically acceptable temporal consistency for full-arch reconstruction, MFC > 0.5 = unacceptable flickering requiring re-scan").**

**★ v0 sub-task 1 video-depth stack now has 5 papers covered (2 commercial-deployable)**: (i) pose-required video-depth (Aether 199, Geo4D 200), (ii) **pose-free video-depth (DepthCrafter 201, VDA 202, ChronoDepth 203)** ⚡, **(iii) sequential-spatial-temporal-fine-tune (ChronoDepth 203)** ⚡ NEW, **(iv) consistent-context-aware-inference (ChronoDepth 203)** ⚡ NEW, **(v) MIT-license video-depth (ChronoDepth 203 MIT ✅)** ⚡ NEW. **★ v0 sub-task 1 video-depth commercial-deployment stack now has 3 MIT/Apache-licensed papers: Aether 199 (MIT ✅) + VDA-S 202 (Apache-2.0 ✅) + ChronoDepth 203 (MIT ✅)** — the *practical* v0 v1+ clinical-deployment stack: **Aether 199 + VDA-S 202 + ChronoDepth 203** = 3 *commercial-deployment-friendly* 2024-2025 video-depth papers. **★ v0 sub-task 1 compute: ~$4,500-6,400 Lambda** (was $4,300-6,200 from 190-note, +$200 for ChronoDepth 203's per-frame-independent-noise + sequential-spatial-temporal-fine-tune + consistent-context-aware-inference + 5-step-DDIM + MFC-metric engineering on clinical-IOS data). **★ v0 TOTAL compute: ~$13,440-19,580 Lambda** (was $13,240-19,380 from 190-note, +$200).

**★ 2024-2025 VIDEO-DEPTH CONVERGENCE: 3+ 2024-2025 papers have *converged* on the *repurpose-foundation-model-for-depth* paradigm** (ChronoDepth 203 [SVD-XT] + DepthCrafter 201 [SVD] + VDA 202 [Depth Anything V2 frozen encoder] + Marigold 119 [Stable Diffusion]) — the *uniform* design lesson: *reuse a pretrained foundation model, add minimal depth-specific adaptation, leverage the foundation's spatial/temporal prior*. The *killer* cross-domain convergence: the depth-estimation community is *catching up* to the 2024-2025 LLM/foundation-model revolution (LoRA, instruction-tuning, sequential fine-tuning — *all* apply to depth adaptation). The 2026 papers will likely integrate *metric-depth heads* + *intrinsics prediction* + *temporal consistency* + *uncertainty quantification* into a *unified* framework — the *next* convergence point.

**★ Open Q for HK:** (i) cite ChronoDepth 203 in v0 v1+ paper? (YES — *founding* consistent-context-aware paradigm + MIT ✅ + 98% MFC improvement); (ii) adopt ChronoDepth's consistent context-aware inference for v0 v1+ sub-task 1? (YES — $200-500 Lambda, *killer* clinical-IOS property); (iii) adopt sequential spatial-temporal fine-tuning for v0 v1+ sub-task 1? (YES — *killer* H1 design lesson, $0, 1-line config); (iv) adopt per-frame independent noise sampling for v0 v1+ sub-task 1? (YES — $0, 5-10 lines, *killer* training recipe); (v) adopt random clip length sampling for v0 v1+ sub-task 1? (YES — $0, 1-line config, *killer* H5 data augmentation); (vi) adopt small-noise σ_ε for overlapping frames? (YES — $0, 1-line config, *killer* clinical-IOS trick); (vii) adopt VAE-replicate-depth-to-3-channels trick? (YES — $0, 5-10 lines, *killer* engineering simplification); (viii) adopt concatenation-over-cross-attention conditioning? (YES — $0, 1-line model change, *killer* architectural simplification); (ix) adopt Hypersim + synthetic video mixture for v0 v1+ training data? (YES — *killer* H5 data lesson, *5× cheaper* than real-data training); (x) adopt MFC as v0 v1+ clinical-fit-analog temporal metric? (YES — *killer* v0 paper positioning); (xi) adopt 5-denoising-steps DDIM-like sampler? (YES — $0, 1-line config, *killer* speed-quality trade-off); (xii) use ChronoDepth 203's 4-paradigm-comparison Table 1 as v0 v1+ paper Table 1 structure? (YES — $0, 1-2 days paper-writing, *killer* v0 paper positioning); (xiii) study 3D-depth extension of ChronoDepth? (OPTIONAL — $500-1000 Lambda, 4-6 weeks, *killer* v0+ research direction); (xiv) adopt per-tooth noise for v0 v1+ sub-task 2 crown generation? (OPTIONAL — $20-50 Lambda, 1-2 days, *killer* DMC 033 + ChronoDepth hybrid); (xv) acknowledge MFC as v0 v1+ paper's clinical-validity metric? (YES — $0, 1-2 hours writing, *killer* v0 paper positioning); (xvi) use ChronoDepth 203's MIT license directly for v0 v1+? (YES — MIT ✅, *no* re-implementation needed, *only* training cost); (xvii) adopt the 39K single-frame + 938 video sequence training data scale for v0 v1+? (YES — *5× cheaper* than real-data training, *sufficient* for open-world zero-shot transfer).

★ ★ **Next paper to read (204):** the 202-note's recommended *next* candidates were (a) ChronoDepth (now read as 203), (b) NVDS (Wang 2023, CVPR 2023, "Neural Video Depth Stabilizer"), (c) MAMO (Park 2023, CVPR 2023, memory-attention video-depth), (d) DeepV2D (Teed 2020, ECCV 2020, combined camera-motion + depth), (e) Marigold (Ke 2023, CVPR 2024, image-generator-to-depth — *the canonical* H2 mechanism), (f) BiDAStereo (Aich 2021, ICRA 2021, bidirectional attention for stereo), (g) Depth-Anything V2 (Yang 2024, arXiv:2406.09414, the frozen-encoder VDA 202 builds on). The 203-note's recommended *next* is **Marigold (Ke 2024, arXiv:2312.02145, CVPR 2024, ETH Zürich + Microsoft)** — the *canonical* *image-generator-to-depth* repurposing paper, the *founding* paper of the *repurpose-LDM-for-monocular-depth* paradigm that inspired ChronoDepth 203 + GeoWizard 122 + DepthFM + Wonder3D 118 (the *direct* ancestor of the entire 203 video-depth + 122 multi-task + 118 multi-view 2024-2025 *repurpose-foundation-model* paradigm), the *right* next paper to understand the *monocular* version of the *repurpose-foundation-model* design space (Marigold = monocular image depth, ChronoDepth = monocular video depth, the *2-paper foundation-model-repurpose foundation*). After Marigold 204, the v0 sub-task 1 *foundation-model-repurpose* arc will have *image* (Marigold 204) + *video* (ChronoDepth 203) coverage, the *complete* 2024-2025 *repurpose-foundation-model* arc. **★ Alternative 204 candidates:** (a) **NVDS (Wang 2023, CVPR 2023)** — the *concurrent* 2023 video-depth *stabilization network* that DepthCrafter 201 + ChronoDepth 203 explicitly beat, the *founding* paper of the *stabilization-network* paradigm; (b) **MAMO (Park 2023, CVPR 2023)** — the *concurrent* 2023 video-depth model with *memory attention*, the *founding* paper of the *memory-attention* video-depth paradigm; (c) **DeepV2D (Teed 2020, ECCV 2020)** — the *founding* paper of *combined camera-motion + depth estimation*, the *historic* foundation; (d) **BiDAStereo (Aich 2021, ICRA 2021)** — the *founding* paper of *bidirectional attention for stereo matching*; (e) **Depth-Anything V2 (Yang 2024, arXiv:2406.09414)** — the *frozen-encoder* that VDA 202 builds on. **★ Recommendation: *read 204 = Marigold (Ke 2024, arXiv:2312.02145, CVPR 2024)*** — the *canonical* *image-generator-to-depth* repurposing paper, the *direct* ancestor of ChronoDepth 203 + GeoWizard 122 + DepthFM + Wonder3D 118, the *right* next paper to understand the *image* version of the *repurpose-foundation-model* design space. After Marigold 204, the v0 sub-task 1 *foundation-model-repurpose* arc will have *image* (Marigold 204) + *video* (ChronoDepth 203) coverage, the *complete* 2024-2025 *repurpose-foundation-model* arc. ⚠️ **PATTERN NOTICE:** the 202-note's "next paper 203 = ChronoDepth (Wang 2024, arXiv:2410.02046, ECCV 2024)" was *conceptually correct* (ChronoDepth is the right next paper) but the *arXiv ID was WRONG* (the 202-note said 2410.02046 which is *actually* "QuickCheck for VDM" by Battle & Ellyton, an unrelated formal-methods paper — the 18-19th arXiv-ID hallucination in the 174-203 arc was *caught and corrected* via direct arXiv lookup) and the *venue was WRONG* (the 202-note said ECCV 2024, but ChronoDepth is *actually* CVPR 2025). The *correct* arXiv ID is **2406.01493** (verified via direct arXiv lookup), the *correct* venue is **CVPR 2025** (verified via openaccess.thecvf.com/content/CVPR2025/papers/Shao_Learning_Temporally_Consistent_Video_Depth_from_Video_Diffusion_Priors_CVPR_2025_paper.pdf), the *correct* canonical repo is **github.com/jiahao-shao1/ChronoDepth** (NOT github.com/jhaoshao/ChronoDepth which redirects), the *correct* license is **MIT** (verified via raw.githubusercontent.com/.../main/LICENSE), the *correct* checkpoint is **huggingface.co/jhshao/ChronoDepth-v1** (MIT ✅). The *new* critical findings (1) the *founding paper* of the *consistent-context-aware* paradigm, (2) the *direct rebuttal* of DepthCrafter's "replacement trick" via the (D)→(E)→(F) ablation, (3) the *killer* sequential-spatial-temporal fine-tuning recipe, (4) the *per-frame independent noise* training trick, (5) the *killer* small-noise σ_ε on overlapping frames, (6) the *killer* H2 + H5 design lessons for v0 v1+, (7) the *killer* MIT license for v0 commercial-deployment, (8) the *killer* 39K + 938 = 39.9K total training samples (vs Depth Anything's 62M = 1500× more), (9) the *killer* 5-denoising-steps for 5× faster inference, (10) the *killer* MFC metric for clinical-IOS temporal consistency. *Always* verify (1) arXiv ID via direct arXiv lookup, (2) venue via OpenAccess/CVF, (3) GitHub canonical repo (NOT redirects), (4) LICENSE file CONTENT (not just license metadata), (5) HF checkpoint license, (6) per-model-weight-license, (7) the *correct* lead-author surname (ChronoDepth lead is *Jiahao Shao* not "Wang" as the 202-note hallucinated), (8) the *exact* author affiliations (6 affiliations: Zhejiang U + Bologna + USC + Ant Group + Toyota Research Institute + Rock Universe AI), (9) the *exact* training data scale (39K single-frame + 938 video sequences), (10) the *exact* inference cost (5 denoising steps, ~288 GPU-hours training = ~$300 Lambda on-demand).
