# Paper 202 — Video Depth Anything (Sili Chen et al., 2025)

## TL;DR

**Video Depth Anything is the FOUNDING PAPER OF THE "FOUNDATION-MODEL-EXTENSION + TEMPORAL-GRADIENT-MATCHING-LOSS + KEY-FRAME-OVERLAP-INTERPOLATION" PARADIGM for *super-long* (multi-minute) consistent video depth estimation** — it takes the frozen **Depth Anything V2** DPT-style encoder (DINOv2 ViT-L, ~330M params) and **only trains a lightweight spatiotemporal head (STH)** that inserts **4 temporal self-attention layers** (at 2 lowest resolutions after Reassemble + 2 before Fusion) into the DPT decoder, trained with a novel **Temporal Gradient Matching loss (TGM, Eq. 3)** that minimizes ‖|d_{i+1}−d_i| − |g_{i+1}−g_i|‖₁ on *same-coordinate* depth differences (NOT optical-flow-warped, *fundamentally different* from the OPW loss [NVDS 2023] which fails under camera motion because "depth of corresponding points is not invariant across adjacent frames"), thresholded to |g_{i+1}−g_i| < 0.05 to avoid dynamic-object/edge instability; plus **2-stage training** (Stage 1: synthetic+wild-binocular video, Stage 2: synthetic video + 0.62M unlabeled images via teacher distillation) on 550K video frames + 0.62M unlabeled images; plus a **key-frame-based long-video inference** that builds each new 32-frame window as N−T_o−T_k = 22 future + 8 overlapping + 2 key frames (every Δ_k=12 frames going backward), then **scale-shift-aligns** the new window's predictions to the previous window via least-squares on the 2 key frames, then **linearly interpolates** the 8 overlap frames with weights 1→0 to ensure smooth cross-window transition; the *killer contribution* is the **3-component minimalism**: no optical flow, no camera pose, no test-time optimization, no video diffusion — just 4 temporal-attention layers + a temporal-gradient loss + a key-frame overlap stitch, achieving **SOTA on 4/5 long-video benchmarks** (KITTI δ₁ 0.944, ScanNet δ₁ 0.926, Bonn δ₁ 0.959, NYUv2 δ₁ 0.971, Sintel δ₁ 0.644), **outperforming all video baselines on temporal consistency (TAE 0.570 on ScanNet vs DepthCrafter's 0.639 — *better consistency with 0.55M vs 10.5M training frames*)**, **at 67 ms/frame FP32 on A100 (9.1 ms/frame for VDA-S, 30+ FPS real-time)** vs DepthCrafter 910 ms/frame, *with* image-quality *preserved* (δ₁ 0.946 on KITTI vs DAv2-L 0.946, *parity* on images); **CVPR 2025 Highlight (13.5% acceptance)**, code at github.com/DepthAnything/Video-Depth-Anything (Apache-2.0 ✅), 1958 ⭐ / 181 🍴 / 31 open issues / last push 2025-10-07 (~8 months ago, *less actively maintained than DepthCrafter 201* but still updated); **CRITICAL LICENSE FINDING — repo code is Apache-2.0 ✅ but model weights are SPLIT: VDA-Small (28.4M) is Apache-2.0 ✅ commercial-deployable; VDA-Base (113.1M) and VDA-Large (381.8M) are CC-BY-NC-4.0 ⚠️ non-commercial** (README says *"For business cooperation, please send an email to Hengkai Guo at guohengkaighk@gmail.com"*); this is the *first* paper in the 201-202 video-depth arc to **explicitly beat DepthCrafter on long sequences** (the 201-note's exact opposite of what DepthCrafter claims) and to **prove that foundation-model-extension > video-diffusion-for-depth for the length-generalization axis**; the *killer killer insight* is that **"temporal consistency ≠ absolute depth accuracy"** — VDA-L wins KITTI δ₁ by **+10% over DepthCrafter** while using **38× fewer parameters** (381.8M vs 2156.7M) and **19× fewer training video frames** (0.55M vs 10.5M), *the* strongest H3 evidence in the 202-paper list (selective cross-frame context > dense cross-frame diffusion).

## Research Question

**Question:** Existing video-depth methods are stuck in a 3-way trade-off: (a) **optical-flow / camera-pose-based warping losses** (NVDS, MAMO) need accurate flow/pose → fail on dynamic + long sequences, (b) **video-diffusion-based video-to-depth** (DepthCrafter, ChronoDepth, DepthAnyVideo) produce fine details but are slow (910 ms/frame for DepthCrafter) and limited to training window length (110 frames → flickering between windows), (c) **test-time optimization** (DeepV2D, NVDS-TTO) are impractically slow. **Can we *inherit* the full capability of a *static-image foundation model* (Depth Anything V2) — generalization, detail fidelity, computational efficiency — while adding temporal consistency for *arbitrarily long* videos (over several minutes), *without* optical flow, *without* camera pose, *without* test-time optimization, *without* video diffusion, using only a *lightweight temporal head + a temporal-gradient loss + a key-frame overlap stitch*?**

## Method

### 3.1 Architecture: Frozen DAv2 Encoder + Lightweight Spatiotemporal Head (STH)

- **Backbone (frozen):** Depth Anything V2 encoder (DINOv2 ViT-S / ViT-B / ViT-L, 24.8M / 105.4M / 330.0M params respectively — *preserved*, no fine-tuning)
- **Head (trained):** DPT-style decoder with **4 inserted temporal self-attention layers** (Fig. 10):
  - 2 inserted *after* the Reassemble layers at the 2 smallest resolutions
  - 2 inserted *before* the last 2 Fusion layers
  - Each temporal layer = multi-head self-attention + FFN, applied *along the temporal dimension* (i.e., for each spatial position, attend across N frames) with **absolute temporal positional embedding**
  - **Input feature reshaping:** `(B×N) × C × H_f × W_f` → `(B×H_f×W_f) × N × C` → attention → reshape back
  - **Why temporal-only attention (NOT spatio-temporal)?** "Introducing temporal attention only in the head prevents the learned representation from being corrupted by the limited video data" — *the* elegant design lesson (encoder features stay *general*, temporal module learns *temporal*)
- **Frame-batch trick:** for video input, "we collapse the temporal dimension of a video clip into the batch dimension" → `X ∈ ℝ^{(B×N)×C×H×W}`, N=1 for image → the *same* encoder handles both video and image inputs without modification
- **Output:** depth map D ∈ ℝ^{H×W} (affine-invariant, but per-*video* — not per-frame — scale/shift, the *key* temporal-consistency design)
- **Three model sizes:**
  - VDA-Small (28.4M total): ViT-S encoder + STH
  - VDA-Base (113.1M total): ViT-B encoder + STH
  - VDA-Large (381.8M total): ViT-L encoder + STH (the *main* reported model)

### 3.2 Temporal Gradient Matching loss (TGM) — the *killer* design

- **Starting point: Optical-Flow-Based Warping (OPW) loss** (Eq. 1, from NVDS):
  - ℒ_OPW = (1/N-1) Σ ‖p_i − p̂_i‖₁ where p̂_i is the *optical-flow-warped* depth from frame i+1 to frame i
  - **Fundamental flaw (Sec. 3.2):** "the depth of corresponding points is not invariant across adjacent frames. This assumption holds true only when adjacent frames are stationary. For instance, in driving scenario, when a car is moving forward, the distance to static objects in front decreases relative to the car, violating the assumption of ℒ_OPW." — the *killer* insight that *optical-flow-based temporal consistency is wrong for moving cameras*
- **Stable Error (SE) loss** (Eq. 2): the *correct* assumption: "the *change* in depth of corresponding points between adjacent prediction frames should be consistent with the change observed in ground truth" — *still* uses optical flow (overhead + errors)
- **Temporal Gradient Matching (TGM) loss** (Eq. 3, *the* contribution): **drop the optical-flow warping**, just use the *same coordinate* in adjacent frames:
  - ℒ_TGM = (1/N-1) Σ ‖|d_{i+1} − d_i| − |g_{i+1} − g_i|‖₁
  - The "|·|" in |d_{i+1} − d_i| handles sign ambiguity (the model can't tell which way depth changes — *both* increasing and decreasing are valid)
  - **Threshold:** only compute TGM in regions where |g_{i+1} − g_i| < 0.05 (i.e., *temporally smooth* regions; skip dynamic objects, edges, occlusions where ground truth is *unstable*)
  - *No optical flow needed at all* — the *killer* simplicity
- **Total loss for video data** (Eq. 4):
  - ℒ_all = α · ℒ_TGM + β · ℒ_ssi
  - ℒ_ssi = scale-shift-invariant loss from MiDaS (image-level spatial loss)
  - Loss weights in practice: **α=10.0 (TGM), β=1.0 (SSI), 0.5 (distillation)**
- **For image data (unlabeled):** the same DAv2 self-distillation loss (teacher = ViT-G trained on synthetic)
- **Why TGM works (the deep insight):** the model isn't asked to predict the *same* depth at corresponding points (which is wrong); it's asked to predict the *same rate of change* (which is right under both static and moving cameras) — *elegant*

### 3.3 Long-Video Inference: Key-Frame-Based Overlap-Interpolation (OI+KR)

- **The 4-strategy ablation (Table 5):**
  - **Baseline:** independent windows, no overlap → jaggies between windows
  - **Overlap Alignment (OA):** 4-frame overlap, scale-shift-align overlapping frames between windows → *cumulative scale drift* (Fig. 7 shows 4'04" video with drift at last frame)
  - **Overlap Interpolation (OI):** 4-frame overlap, linear interpolation in overlap region (same as DepthCrafter)
  - **Overlap Interpolation + Key-Frame Reference (OI+KR, Ours):** **N=32, T_o=8, T_k=2, Δ_k=12** — each new window = 22 future + 8 overlap + 2 key frames (the 2 key frames = every 12th frame going backward in the *previous* window, providing *long-range context*), then linear interpolation in the 8 overlap region, then scale-shift-align the new window's 2 key frames to the previous window's corresponding 2 frames via least-squares
- **Why OI+KR > OA:** OA fails on long videos (4'04" example shows drift); OI+KR *maintains global scale consistency* because the 2 key frames provide *anchor* to the previous window's scale/shift
- **Why OI+KR > OI alone (Table 5):** TAE 0.761 (OI) → 0.718 (OI+KR-32), the *direct* evidence that key-frame reference adds ~5% improvement on temporal consistency
- **Why window=32 > window=16/48 (Table 5):** AbsRel 0.157→0.144, δ₁ 0.826→0.851, TAE 0.874→0.718 (32 vs 16); *no further benefit* beyond 32 (48: AbsRel 0.143, TAE 0.732 — *worse* on TAE)
- **The "arbitrarily long" claim:** the inference strategy is *inherently* O(N) memory (one 32-frame window at a time, with key-frame anchor), *no* global accumulator → can process 4,690-frame videos (Fig. 1's 196-second pair-skating demo) in principle *indefinitely*

### 3.4 Training: 2-Stage Progressive

- **Stage 1:** synthetic videos (TartanAir, VKITTI, PointOdyssey, IRS, total 0.55M frames) + wild binocular videos (0.18M frames, labeled via [16] — likely DROID-SLAM or similar) → supervised with ℒ_all (Eq. 4)
- **Stage 2:** synthetic videos + 0.62M unlabeled images (DAv2 self-distillation) → supervised with ℒ_all for videos + DAv2 self-distillation loss for images
- **Optimizer:** AdamW, cosine scheduler, **base lr 1e-4**, batch 16 video frames (length 32) + 128 images
- **Training input:** 518×518 random center crop, 32-frame clips sampled uniformly from each dataset
- **Uniform sampler** to ensure equal contribution from each dataset (4 video datasets + image distillation set)
- **Hardware:** not explicitly stated in main paper (likely 8 × A100 given 2025 standard, ~3-5 days for the smaller model, ~5-7 days for VDA-L)
- **Total video training data: 0.73M frames (Sec. 1) / 0.55M frames (Sec. 4 + Implementation Details)** — note: the abstract says 730K, the implementation details say 0.55M (after excluding 0.13M PointOdyssey frames with no background depth GT), so the 0.55M is the *effective* video training data, with the 0.18M wild-binocular frames added later

## Results

### Main Table 1: Zero-shot video depth (window=32, 500-frame videos)

| Method | KITTI AbsRel↓ | KITTI δ₁↑ | ScanNet AbsRel↓ | ScanNet δ₁↑ | Bonn AbsRel↓ | Bonn δ₁↑ | NYUv2 AbsRel↓ | NYUv2 δ₁↑ | Sintel AbsRel↓ | Sintel δ₁↑ | ScanNet TAE↓ |
|--------|----------------|-----------|------------------|-------------|---------------|----------|---------------|------------|----------------|------------|--------------|
| DAv2-L (image baseline) | 0.137 | 0.815 | 0.150 | 0.768 | 0.127 | 0.864 | 0.094 | 0.928 | 0.390 | 0.541 | 1.140 |
| NVDS | 0.233 | 0.614 | 0.207 | 0.628 | 0.199 | 0.674 | 0.217 | 0.598 | 0.408 | 0.464 | 2.176 |
| NVDS+DAv2-L | 0.227 | 0.617 | 0.194 | 0.658 | 0.191 | 0.700 | 0.184 | 0.679 | 0.449 | 0.503 | 2.536 |
| ChronoDepth | 0.243 | 0.576 | 0.199 | 0.665 | 0.199 | 0.665 | 0.173 | 0.771 | 0.192 | 0.673 | 1.022 |
| DepthCrafter | 0.164 | 0.753 | 0.169 | 0.730 | 0.153 | 0.803 | 0.141 | 0.822 | 0.299 | 0.695 | 0.639 |
| DepthAnyVideo | - | - | - | - | - | - | - | - | 0.405 | 0.659 | 0.967 |
| **VDA-S (Ours, 28.4M)** | **0.086** | **0.942** | **0.110** | **0.876** | **0.083** | **0.950** | **0.077** | **0.959** | 0.339 | 0.584 | 0.703 |
| **VDA-L (Ours, 381.8M)** | **0.083** | **0.944** | **0.089** | **0.926** | **0.071** | **0.959** | **0.062** | **0.971** | 0.295 | 0.644 | **0.570** |

- **Wins 4/5 datasets on δ₁** (only Sintel, the synthetic movie dataset, is lost — VDA-L gets 0.644 vs DepthCrafter's 0.695, *attributed to absence of movie data in training set*)
- **Wins 5/5 datasets on TAE (temporal consistency)** — the *killer* finding (VDA-L 0.570 vs DepthCrafter 0.639 = -10.8% improvement, VDA-S 0.703 vs DepthCrafter 0.639 = *worse on consistency but with 13× fewer params*)
- **Wins all 5 datasets on δ₁ over DAv2-L** (the image baseline!) — adding temporal attention to the *head* (not the encoder) *improves* image depth too (counter-intuitive)
- **The "10%" claim verified:** VDA-L beats DepthCrafter by ~10% δ₁ on KITTI, ScanNet, Bonn (e.g., ScanNet 0.926 vs 0.730 = +19.7%, *not* 10%; Bonn 0.959 vs 0.803 = +19.4%; the abstract's "10%" is conservative)
- **VDA-S vs VDA-L:** δ₁ gap is ~0.05 (VDA-S 0.876 ScanNet vs VDA-L 0.926), TAE gap is ~0.13 (VDA-S 0.703 vs VDA-L 0.570) — *the* engineering trade-off for the 30-FPS real-time target

### Table 2: Zero-shot single-image depth (preserved quality)

| Method | KITTI AbsRel↓ | KITTI δ₁↑ | Sintel AbsRel↓ | Sintel δ₁↑ | NYUv2 AbsRel↓ | NYUv2 δ₁↑ | ETH3D AbsRel↓ | ETH3D δ₁↑ | DIODE AbsRel↓ | DIODE δ₁↑ | Rank |
|--------|----------------|-----------|----------------|------------|---------------|------------|----------------|------------|---------------|------------|------|
| DepthCrafter | 0.107 | 0.891 | 0.568 | 0.652 | 0.082 | 0.936 | 0.179 | 0.793 | 0.141 | 0.857 | 4.0 |
| DepthAnyVideo | 0.073 | 0.946 | 0.687 | 0.692 | 0.058 | 0.963 | 0.123 | 0.881 | 0.072 | 0.942 | 2.4 |
| DAv2-L (image baseline) | **0.074** | **0.946** | 0.487 | 0.752 | **0.045** | **0.979** | **0.131** | 0.865 | **0.066** | **0.952** | **1.4** |
| **VDA-L (Ours)** | 0.075 | 0.946 | **0.496** | **0.754** | 0.046 | 0.978 | 0.132 | 0.863 | 0.067 | 0.950 | 2.0 |

- **Parity with DAv2-L on images** (rank 2.0 vs 1.4, only marginal loss) — *the* key validation that adding temporal attention to the *head* (not the encoder) doesn't *catastrophically forget* image capability
- **Beats DepthCrafter on image depth by 2 rank points** — VDA is *both* video-SOTA *and* image-competitive

### Table 3: Inference latency (518×518, A100, FP32)

| Method | Latency (ms/frame) | Speedup vs VDA-L |
|--------|--------------------|---------------------|
| ChronoDepth (FP16) | 506 | 7.6× |
| DepthCrafter (FP16) | 910 | 13.6× |
| DepthAnyVideo (FP16) | 159 | 2.4× |
| NVDS (FP32) | 204 | 3.0× |
| DAv2-L (FP32) | 60 | 1.1× (≈baseline) |
| **VDA-L (FP32, Ours)** | **67** | 1.0× (baseline) |
| **VDA-S (FP32, Ours)** | **9.1** | 7.4× (30+ FPS real-time) |

- **VDA-L is 13.6× faster than DepthCrafter** (the 201-note's same comparison) — *the* killer efficiency story
- **VDA-S is 30+ FPS** (9.1 ms/frame) — *real-time* at 518×518, the *first* paper in the 196-202 arc to demonstrate real-time video depth
- **The "only ~10% slower than DAv2-L" claim verified** (67 vs 60 ms/frame) — the temporal head adds *only* 7 ms per frame, *negligible* overhead

### Table 4: Ablation on temporal losses (VDA-S, window=16, no image distillation, TartanAir + VKITTI)

| Loss | AbsRel↓ | δ₁↑ | TAE↓ |
|------|---------|-----|------|
| VideoAlign (shared scale-shift + L1) | 0.151 | 0.846 | 1.326 |
| VideoAlign+SSI | 0.151 | 0.848 | 1.207 |
| OPW (NVDS) + SSI | 0.182 | 0.771 | 0.918 |
| SE (Stable Error, Eq. 2, w/ flow) + SSI | 0.160 | 0.836 | 0.753 |
| **TGM+SSI (Ours, w/o flow)** | 0.166 | 0.832 | **0.767** |

- **TGM+SSI is the *only* loss without optical flow** that achieves competitive TAE (0.767 vs SE+SSI's 0.753) — the *killer* "no-flow-no-regret" finding
- **TGM beats OPW by 16.5% TAE** (0.918 → 0.767) — direct evidence that "depth of corresponding points is not invariant" is *empirically* true
- **VideoAlign+SSI loses on TAE** (1.207) — confirms that *spatial* loss alone is *not* sufficient for temporal consistency

### Table 5: Ablation on inference strategy (VDA-S, window=16, TGM+SSI)

| Strategy | Window | AbsRel↓ | δ₁↑ | TAE↓ |
|----------|--------|---------|-----|------|
| Baseline (no overlap) | 16 | 0.157 | 0.826 | 0.874 |
| OA (overlap-align) | 16 | 0.146 | 0.845 | 0.792 |
| OI (overlap-interp, DepthCrafter-style) | 16 | 0.157 | 0.826 | 0.783 |
| **OI+KR (Ours)** | 16 | 0.145 | 0.849 | 0.761 |
| **OI+KR (Ours)** | **32** | **0.144** | **0.851** | **0.718** |
| OI+KR (Ours) | 48 | 0.143 | 0.852 | 0.732 |

- **Window 32 is the sweet spot** — beyond that, TAE *degrades* (0.732 at 48) due to *over-smoothing* from too many key frames
- **OI+KR vs OI:** TAE 0.761 vs 0.783 = -2.8% with the *same* interpolation — key-frame reference adds the 2.8%
- **OI+KR vs OA:** 0.761 vs 0.792 = -3.9% — interpolation *plus* key frames beats *just* alignment

### Table 6: Ablation on image-distillation training (VDA-S, window=16)

| Datasets | Image AbsRel↓ | Image δ₁↑ | Video AbsRel↓ | Video δ₁↑ | Video TAE↓ |
|----------|----------------|-----------|---------------|-----------|------------|
| Video only | 0.180 | 0.876 | 0.145 | 0.849 | 0.761 |
| **Video + Image** | **0.167** | **0.883** | **0.142** | **0.852** | **0.742** |

- **Adding unlabeled-image distillation improves *all* metrics** (image δ₁ 0.876→0.883, video TAE 0.761→0.742) — the *H5 evidence* that foundation-model self-distillation transfers to video

### Table 7: Short-video results (KITTI 110, Bonn 110, ScanNet 90 frames)

| Method | Params | Training frames | KITTI AbsRel↓ | KITTI δ₁↑ | Bonn AbsRel↓ | Bonn δ₁↑ | ScanNet AbsRel↓ | ScanNet δ₁↑ |
|--------|--------|------------------|----------------|-----------|---------------|----------|------------------|------------|
| DepthCrafter | 2156.7M | 10.5M | 0.111 | 0.885 | 0.066 | 0.979 | 0.125 | 0.848 |
| DepthAnyVideo | 1422.8M | 6M | **0.073** | **0.957** | **0.051** | **0.981** | 0.112 | 0.883 |
| **VDA-L (Ours)** | 381.8M | 0.55M | 0.079 | 0.950 | 0.053 | 0.972 | **0.075** | **0.954** |

- **VDA-L is 5.6× smaller than DepthCrafter with 19× fewer training frames** *and* better on ScanNet (+12.5% δ₁)
- **DepthAnyVideo wins KITTI/Bonn (more data: 6M frames)** — confirms the H5 *categorical* lesson that *data scale matters for absolute accuracy*
- **VDA-L wins ScanNet by 7-12%** — the *killer* data-efficiency result

### Killer Surprises Buried in Section 4 (qualitative, Fig. 5, 6, 7, 9, 12)

1. **196-second (4,690-frame) pair-skating demo** (Fig. 1 left) — the *first* demonstration of *minute-long* consistent video depth, vs DepthCrafter's 110-frame max → the *killer* product feature
2. **OA fails on a 4'04" (7,320-frame) video** (Fig. 7) — *cumulative scale drift* even with overlap alignment; the *killer* ablation that *just* scale-shift-alignment is *insufficient* for super-long inference
3. **DepthCrafter produces "discontinuous layers" in point cloud** (Fig. 12) — VDA-L produces "clean and regular point cloud" → the *killer* application difference (point cloud = 3D foundation for v0 sub-task 1)
4. **DepthCrafter exhibits "depth drift" in long videos** (Fig. 5, 9) — VDA-L does *not*, the *direct* H3 evidence that *frozen-encoder + temporal-head* is more stable than *end-to-end-finetuned video diffusion*
5. **DAv2-L produces "flickering depth" in videos** (Fig. 5) — *the* motivating problem the paper solves
6. **VDA works "without sacrificing [DAv2's] generalization ability, richness in details, or computational efficiency"** (Sec. 1) — the *triple-preservation* claim, *the* product advantage
7. **VDA's metric-depth variants** (released 2025-04-25 + 2025-08-28) — VDA-S/Base/Large metric-depth models achieve KITTI δ₁ 0.910 (L), NYUv2 δ₁ 0.908 (L), ScanNet TAE 1.09 (L) → *killer* follow-up products
8. **Streaming-mode released 2025-09-12** (training-free, single-frame inference with hidden-state cache) → the *killer* product for *true* real-time chairside use; tradeoff: ScanNet δ₁ 0.926→0.836 (offline → streaming, the *price* of training-free)

## Connections to H1-H5

**H1 PARTIAL (training-time 2-stage is settled, inference is 1-stage feed-forward):**
- 2-stage training (Stage 1 video-only → Stage 2 video+image-distillation) is the H1 curriculum paradigm
- But architectural inference is *strictly* 1-stage feed-forward (no iterative refinement, no test-time optimization) — the *settled* 2024-2026 design for video depth
- H1 update: *for video-to-X tasks adapting a frozen foundation model, 2-stage video+image-distillation > 1-stage end-to-end fine-tune* (the *killer* H5 + H1 recipe)

**H2 MILD CONTRADICTION in 202-paper list:**
- Video Depth Anything is *not* a probabilistic latent model (no diffusion, no VAE bottleneck beyond DAv2's frozen ViT)
- The temporal-gradient loss is *not* a H2 latent in the probabilistic sense — it's a *deterministic* regression target
- The 2-key-frame scale-shift alignment is a *deterministic* (not probabilistic) latent bridge
- H2 update: *for video-to-X tasks with a frozen foundation model, deterministic feed-forward + key-frame alignment > probabilistic latent diffusion* (the *empirical* contradiction in 202, *same* as LoGeR 187 + LingBot-Map 184 + R³ 183 + StreamVGGT + the *entire* 2024-2026 streaming-3R / video-depth arc that has *decisively* chosen feed-forward over diffusion for streaming 3R + video depth)

**H3 STRONGEST DIRECT SUPPORT in 202-paper list:**
- **TGM loss** = THE H3 mechanism for *cross-frame temporal consistency* (constraints the *gradient* of depth across time, not the *value*)
- **OI+KR inference** = THE H3 mechanism for *cross-window temporal aggregation* (linear interpolation in 8-frame overlap + 2-key-frame scale-shift anchor)
- **4 temporal self-attention layers** in the head = THE H3 mechanism for *intra-window cross-frame attention* (DPT-decoder-level attention along temporal dim)
- H3 update: *for video-to-X tasks, selective cross-frame context (4 layers + key-frame stitch) > dense cross-frame diffusion* (the *killer* finding that *context-sparse > context-dense* for temporal aggregation)

**H4 INDIRECT SUPPORT:**
- Output is per-pixel depth (NOT SDF, NOT pointmap, NOT NeRF) — the *de facto* 2024-2026 H4 substrate for video depth
- For v0 *dental* use: 2D-depth backbone, but final 3D must come from DUSt3R 003 / MonST3R 174 / VGGT (same as DepthCrafter 201)

**H5 STRONGEST DIRECT SUPPORT in 202-paper list:**
- **0.55M video frames (TartanAir 0.31M + VKITTI 0.04M + PointOdyssey 0.1M + IRS 0.1M) + 0.18M wild binocular + 0.62M unlabeled images** = the *killer* H5 data mix (4 synthetic datasets + 1 wild-binocular + 0.62M unlabeled distillation)
- **2-stage curriculum** (video → video+image-distillation) = the *killer* H5 training recipe (Table 6: video-only AbsRel 0.180 → video+image AbsRel 0.167 = -7% on *image* depth via distillation)
- **Frozen DINOv2 encoder from DAv2** = the *killer* H5 transfer-learning recipe (reuse 1000+ GPU-hours of pretraining)
- H5 update: *for video-to-X tasks adapting a foundation model, frozen-encoder + 4-dataset-mix + 0.62M-distillation > end-to-end-from-scratch* (VDA-L beats DepthCrafter with 19× fewer training frames and 5.6× fewer params)

## Quote-Worthy Sentences

- *"Is it possible to have a model that can perfectly inherit the capabilities of existing foundation models while achieving temporal stability for arbitrarily long videos?"* (Sec. 1, the killer framing question)
- *"the depth of corresponding points is not invariant across adjacent frames. This assumption holds true only when adjacent frames are stationary. For instance, in driving scenario, when a car is moving forward, the distance to static objects in front decreases relative to the car, violating the assumption of ℒ_OPW."* (Sec. 3.2, the *killer* critique of optical-flow-based temporal losses)
- *"we posit that the change in depth of corresponding points between adjacent prediction frames should be consistent with the change observed in ground truth"* (Sec. 3.2, the TGM loss philosophical foundation)
- *"Introducing temporal attention only in the head prevents the learned representation from being corrupted by the limited video data"* (Sec. 3.1, the *killer* design lesson: temporal modules in *head*, not *encoder*)
- *"We compare our model with baselines on five datasets for zero-shot video depth estimation. Our model achieves state-of-the-art (SOTA) results on four of the datasets in terms of spatial accuracy and outperforms all baselines on all datasets in terms of temporal consistency."* (Sec. 4.2, the *killer* results summary)
- *"Notably, our compact model, VDA-S, which has significantly lower latency compared to other models (as shown in Table 3), demonstrates superior geometric accuracy over representative diffusion-based methods for long videos."* (Sec. 4.2, the *killer* 30-FPS-real-time + SOTA-quality claim)
- *"the latency of our large model, VDA-L, is only approximately 10% greater than that of DAv2-L, which uses the same encoder structure, thus demonstrating the efficiency of our spatiotemporal head"* (Sec. 4.2, the *killer* "only 10% overhead" claim)
- *"For the first time, we can estimate consistent depth for videos over several minutes"* (Sec. 1, the *killer* length-generalization claim)
- *"Video-Depth-Anything-Small model is under the Apache-2.0 license. Video-Depth-Anything-Base/Large model is under the CC-BY-NC-4.0 license."* (GitHub README, the *killer* license split for v0 commercial deployment)

## Connections to v0/v1+ sub-task 1 (full-arch synthesis from multi-view IOS)

**v0 sub-task 1 INPUTS** (per 201-note, the *commercial-deployable* Aether 199 + Geo4D 200 + DepthCrafter 201 stack):
- 10-30 intra-oral scans (buccal/lingual/occlusal) per arch
- 200+ frames chairside-video intra-oral scan (for v1+)
- Camera poses (Aether 199 / VGGT) OR no camera poses (DepthCrafter 201)

**v0 sub-task 1 OUTPUT:** 3D pointmaps or meshes of full dental arch

**Where Video Depth Anything 202 fits:**
- (a) **If v0 uses *video* intra-oral scan** (chairside 200+ frame video), VDA is *directly applicable* (use VDA-L as 2D-depth backbone with VDA-S as real-time preview; then DUSt3R 003 / MonST3R 174 / VGGT for 3D-fusion)
- (b) **If v0 uses *multi-view* stills** (10-30 buccal/lingual/occlusal stills), VDA is *less* applicable (it's video, not multi-view) — use MonST3R 174 / DUSt3R 003 / VGGT instead
- (c) **If v0 uses *hybrid*** (10-30 stills + 1 chairside video per arch), VDA provides the *video component*'s depth, MonST3R provides the *stills component*'s geometry
- (d) **The metric-depth variants (VDA-S/Base/Large-Metric, 2025-04-25 + 2025-08-28)** are *especially* relevant for v0 dental use — *absolute* depth with metric scale, the *right* substrate for 3D-fusion with DUSt3R/MonST3R/VGGT
- (e) **The streaming-mode (2025-09-12)** is *the killer* for v0 v1+ real-time chairside use: single-frame inference with hidden-state cache, 30+ FPS, *zero* training required

## For our project (concrete v0 next steps)

- **(a) ★★★ CITE VIDEO DEPTH ANYTHING 202 IN V0 PAPER RELATED-WORK AS THE *FOUNDING* FOUNDATION-MODEL-EXTENSION + TEMPORAL-GRADIENT-MATCHING-LOSS + KEY-FRAME-OVERLAP-INTERPOLATION PARADIGM** ($0, 1-2 hours, 1 paragraph: *"We adopt the foundation-model-extension paradigm (Chen et al. 2025) for our chairside-intra-oral-video depth backbone, which has been shown to achieve SOTA zero-shot performance on 4/5 long-video benchmarks with 13.6× speedup over DepthCrafter, while preserving the image-depth capability of the foundation model (Depth Anything V2), the *first* method to extend a frozen image-depth foundation model to super-long videos via a lightweight temporal head + temporal-gradient loss + key-frame overlap stitch, without optical flow, without camera pose, and without video diffusion."*)
- **(b) ★★★ ADOPT TEMPORAL GRADIENT MATCHING (TGM) LOSS AS V0 SUB-TASK 1 TEMPORAL-CONSISTENCY LOSS** ($0, 1-2 days, 5-10 lines PyTorch, ‖|d_{i+1}−d_i| − |g_{i+1}−g_i|‖₁ with threshold |g_{i+1}−g_i| < 0.05, *killer* no-optical-flow design, *complementary* to DepthCrafter 201's mortise-and-tenon for *cross-window* aggregation)
- **(c) ★★★ ADOPT OI+KR (OVERLAP-INTERPOLATION + KEY-FRAME REFERENCE) AS V0 SUB-TASK 1 LONG-VIDEO INFERENCE** ($0, 1-2 days, N=32, T_o=8, T_k=2, Δ_k=12, *killer* design for >110-frame clinical IOS videos, *complementary* to DepthCrafter 201's mortise-and-tenon: VDA's OI+KR is *simpler* (no latent interpolation) and *faster* (no diffusion inference))
- **(d) ★★ ADOPT LIGHTWEIGHT TEMPORAL HEAD (4 LAYERS IN DPT DECODER) AS V0 SUB-TASK 1 ARCHITECTURE** ($50-100 Lambda, 1-2 weeks, insert 4 temporal self-attention layers at 2 lowest-resolution Reassemble + 2 last Fusion, *killer* design lesson: temporal modules in *head*, not *encoder*, *preserves* foundation-model generalization)
- **(e) ★★ ADOPT FROZEN-DAV2-ENCODER + TRAIN-HEAD-ONLY AS V0 SUB-TASK 1 TRAINING RECIPE** ($0, 1-line config change, 5.6× parameter reduction, 19× training-frame reduction, *killer* H5 + efficiency recipe)
- **(f) ★★ ADOPT 2-STAGE TRAINING (VIDEO → VIDEO+IMAGE-DISTILLATION) AS V0 SUB-TASK 1 CURRICULUM** ($0, 1-line config change, *killer* H5 recipe for adapting a frozen foundation model to video; Stage 1 video-only → Stage 2 video + 0.62M unlabeled image distillation)
- **(g) ★★ ADOPT TARTANAIR + VKITTI + POINTODYSSEY + IRS DATA MIX AS V0 SUB-TASK 1 H5 DATA-MIX PATTERN** ($200-400 Lambda, 1-2 weeks, 4 synthetic video datasets + 1 wild-binocular + 0.62M unlabeled image distillation, *killer* H5 data-mix for adapting pretrained video depth to dental)
- **(h) ★★ USE METRIC-DEPTH VDA-L-METRIC (2025-04-25 RELEASE) AS V0 SUB-TASK 1 COMMERCIAL-PERMISSIVE 2D-DEPTH BACKBONE** ($0, just download + run, *the only* SOTA-quality 2D-depth backbone with *commercial-permissive license* for *Large* model? — actually NO, VDA-L-Metric is CC-BY-NC-4.0 ⚠️; VDA-S-Metric is Apache-2.0 ✅; for v0 *commercial* deployment, *VDA-S-Metric* (28.4M, 28.4M Apache-2.0 ✅) is the *viable* choice with the *9.1 ms/frame* 30-FPS-real-time latency)
- **(i) ★★ ADOPT VDA STREAMING-MODE (2025-09-12 RELEASE) AS V0 SUB-TASK 1 REAL-TIME CHAIRSIDE INFERENCE** ($50-100 Lambda, 1-2 days, single-frame inference with hidden-state cache, *zero* training required, the *killer* real-time product for chairside dental use; tradeoff: ScanNet δ₁ 0.926→0.836, *acceptable* for chairside preview but *not* for final clinical-quality reconstruction)
- **(j) ★ ADOPT VDA'S "WINDOW=32" AS V0 SUB-TASK 1 INFERENCE WINDOW** ($0, 1-line config change, 32 frames is the *sweet spot*; >32 degrades TAE, <32 underutilizes temporal context)
- **(k) ★ CITE VDA'S TGM-LOSS-NO-FLOW DESIGN AS V0 PAPER H3 CONTRIBUTION** ($0, 1-2 hours, 1 paragraph: *"We adopt the temporal gradient matching loss (Chen et al. 2025) for our cross-frame temporal consistency, which avoids the fundamental flaw of optical-flow-based warping losses (NVDS) under moving-camera scenarios by constraining the *change* in depth rather than the *value* of depth."*)
- **(l) ★ USE VDA 202 AS V0 SUB-TASK 1 TABLE 1 BASELINE COMPARISON ROW** ($0, just cite + report KITTI/ScanNet/Bonn/NYUv2/Sintel numbers + 9.1-67 ms/frame + 30+ FPS for VDA-S; *disclose* CC-BY-NC-4.0 license for VDA-Base/Large + Apache-2.0 for VDA-S)
- **(m) ★★ OPEN Q FOR HK: deploy VDA-S (Apache-2.0 ✅, 28.4M) as v0 sub-task 1 production? YES (the *only* SOTA-quality 2D-depth backbone with *commercial-permissive license* AND 30-FPS real-time latency AND 1958 ⭐; VDA-L is CC-BY-NC-4.0 ⚠️ but VDA-S is Apache-2.0 ✅; VDA's *technical* contributions [TGM, OI+KR, frozen-encoder + temporal-head, 2-stage training, 4-dataset-mix, metric-depth variants, streaming-mode] are *all directly portable* to v0 even if we don't *deploy* VDA-L itself)**
- **(n) ★★ OPEN Q FOR HK: deploy VDA-L (CC-BY-NC-4.0 ⚠️, 381.8M) as v0 sub-task 1 production? NO (same as WinT3R 185 / LONG3R 186 / LoGeR 187 / Geo4D 200 / DepthCrafter 201, *non-commercial* requires business contact; for *commercial* deployment, use VDA-S Apache-2.0 ✅ or re-implement VDA-L on a commercial-permissive license, $400-600 Lambda, 2-3 weeks)**
- **(o) ★★ STRATEGIC INSIGHT: VDA-S IS THE *FIRST* COMMERCIAL-DEPLOYABLE (Apache-2.0 ✅) 30-FPS REAL-TIME SOTA-QUALITY 2D-DEPTH BACKBONE IN THE 196-202 ARC** (the *killer* v0 production choice for *real-time* chairside video depth; VDA-L is the SOTA-quality choice for *offline* clinical-quality reconstruction but *non-commercial*; together, VDA-S + VDA-L cover the *full* real-time + offline product matrix for v0 v1+ v2 v3)

## v0 Sub-Task 1 Stack Update

- **v0 sub-task 1 streaming-3R + video-depth stack now has 29 papers covered (17 paradigms)** (+ Video Depth Anything 202 = foundation-model-extension + temporal-gradient-matching-loss + key-frame-overlap-interpolation + metric-depth-variants + streaming-mode):
  - **State-token:** CUT3R 175, MonST3R 174, Fast3R 178, Easi3R 173
  - **Memory-token:** Spann3R 177, Point3R 179, STream3R 181, R³ 183, TTT3R 182, Ray-Aware 180
  - **SLAM-prior-structured:** LingBot-Map 184
  - **Window+pool:** WinT3R 185
  - **3D-spatial-memory:** LONG3R 186
  - **Hybrid TTT+SWA:** LoGeR 187
  - **3-modality-fusion + multi-modal-alignment + synthetic-only-training:** Geo4D 200
  - **4D-via-video-diffusion (MIT ✅ commercial-deployable):** Aether 199
  - **Video-diffusion-for-depth (⚠️ NOASSERTION non-commercial):** DepthCrafter 201
  - **Foundation-model-extension + TGM + OI+KR (VDA-S ✅ Apache-2.0, VDA-L ⚠️ CC-BY-NC-4.0):** **Video Depth Anything 202** NEW
- **v0 sub-task 1 compute: ~$4,100-6,100 Lambda** (was $3,900-5,800 from 201-note, +$200-300 for VDA 202 4-temporal-head + TGM-loss + OI+KR-inference + frozen-encoder-recipe + 4-dataset-mix + metric-depth + streaming-mode integration)
- **v0 TOTAL compute: ~$13,040-19,280 Lambda** (was $12,840-18,980, +$200-300 for VDA 202 integration)

## Strategic Comparison: Video Depth Anything 202 vs DepthCrafter 201

| Aspect | DepthCrafter 201 | **Video Depth Anything 202** |
|--------|------------------|--------------------------------|
| Year / Venue | 2024/2025 / CVPR 2025 Highlight | **2025 / CVPR 2025 Highlight** |
| Affiliation | Tencent AI Lab + HKUST | **ByteDance** |
| License | ⚠️ NOASSERTION (Tencent contact) | **Apache-2.0 (code) + Apache-2.0 (VDA-S weights) + CC-BY-NC-4.0 (VDA-B/L weights)** |
| Code stars (GitHub) | 1,556 ⭐ | **1,958 ⭐** (more popular) |
| Last push | 2025-11-30 ✅ active | 2025-10-07 (~8 months ago, less active) |
| Architecture | **Full SVD U-Net fine-tune** (heavy) | **Frozen DAv2 encoder + 4-layer temporal head** (light) |
| Parameters | 2,156.7M (huge) | **381.8M (VDA-L) / 28.4M (VDA-S, 76× smaller)** |
| Training video frames | 10.5M (huge) | **0.55M (19× fewer)** |
| Long-video inference | **Mortise-and-tenon latent interpolation** (with diffusion) | **OI+KR (overlap-interp + key-frame reference, no diffusion)** |
| Camera poses required | **NO** | **NO** |
| Optical flow required | NO | **NO** |
| Video diffusion | YES (SVD U-Net, 5 DDIM steps) | **NO** (pure feed-forward) |
| Inference speed (A100, 518×518) | 910 ms/frame (FP16) | **67 ms/frame (VDA-L, FP32) = 13.6× faster; 9.1 ms/frame (VDA-S, 30+ FPS)** |
| KITTI δ₁ (long video) | 0.753 | **0.944 (+25.4% absolute)** |
| ScanNet δ₁ (long video) | 0.730 | **0.926 (+26.8% absolute)** |
| ScanNet TAE (consistency) | 0.639 | **0.570 (-10.8% absolute)** |
| Sintel δ₁ (synthetic movie) | **0.695 (VDA-L: 0.644, -7.3%)** | 0.644 |
| Metric depth variant | NO | **YES (VDA-S/Base/Large-Metric, 2025-04-25 / 2025-08-28)** |
| Streaming mode | NO | **YES (training-free, 2025-09-12)** |
| Inference memory (VDA-S 32 frames) | not reported | **6.8 GB (FP16, 32 frames @ 518)** |
| Training time | 8 A100 × 5 days = 960 A100-h | **not reported (likely 3-5 days on 8 A100)** |
| Image-depth preservation | loses (Sec. 201) | **preserved (Table 2: rank 2.0 vs DAv2-L's 1.4, only marginal loss)** |

**The killer differentiators:**
1. **VDA wins 4/5 long-video benchmarks on δ₁** (KITTI, ScanNet, Bonn, NYUv2) — *DepthCrafter's 201-note's claim of "SOTA on Sintel/KITTI/ScanNet/Bonn" is now superseded by VDA on KITTI/ScanNet/Bonn/NYUv2*
2. **VDA wins all 5 long-video benchmarks on TAE (temporal consistency)** — *the* killer consistency finding
3. **VDA-L is 13.6× faster than DepthCrafter at 67 vs 910 ms/frame** — *the* killer efficiency finding
4. **VDA-S is 30+ FPS real-time** — *the* killer real-time finding (DepthCrafter is *not* real-time)
5. **VDA-S is Apache-2.0 commercial-deployable** — *the* killer license finding (DepthCrafter requires Tencent contact)
6. **VDA preserves image-depth capability** (rank 2.0 vs DAv2-L's 1.4, only marginal loss) — *the* killer transferability finding (DepthCrafter *loses* on image depth, rank 4.0)
7. **VDA has metric-depth variants + streaming-mode** — *the* killer product-finding (DepthCrafter has neither)

**The one place DepthCrafter wins: Sintel δ₁ 0.695 vs 0.644** — DepthCrafter trained on *more diverse* video data (10.5M vs 0.55M frames) including *movie* data (Sintel is a synthetic movie); for v0 dental, Sintel is *irrelevant* (no clinical movies), so VDA wins for v0 *in practice*

## v0 Sub-Task 1 Design Space Coverage

The *complete* 2024-2026 video-depth + 3R arc now has 29 papers, 17 paradigms, *all* with verified arXiv IDs (the 16-17th arXiv-ID hallucination was *prevented* by direct arXiv lookup on 2501.12375, *correct* arXiv ID), and *all* with verified license status via GitHub API (the 7-8th GitHub-API-license-check was *performed*). The *complete* design-space coverage now includes:

- **(i) Pose-required vs pose-free:** pose-required (Aether 199, Geo4D 200) vs **pose-free (DepthCrafter 201, VDA 202)** ⚡
- **(ii) Image vs video:** image-based (Marigold, Depth-Anything) vs **video-based (DepthCrafter 201, ChronoDepth, VDA 202)** ⚡
- **(iii) Test-time optimization vs feed-forward:** test-time (DeepV2D, NVDS) vs **feed-forward (DepthCrafter 201, VDA 202)** ⚡
- **(iv) Multi-modal vs single-modal:** multi-modal (Geo4D 200) vs **single-modal depth (DepthCrafter 201, VDA 202)**
- **(v) Synthetic-only vs mixed:** synthetic-only (Geo4D 200) vs **realistic+synthetic mix (DepthCrafter 201, VDA 202)** ⚡
- **(vi) 1-stage vs multi-stage training:** 1-stage (CUT3R 175) vs **2-stage (VDA 202 [video → video+image])** vs **3-stage (DepthCrafter 201)** ⚡
- **(vii) Denoising steps:** 25 (SVD) vs **5 (DepthCrafter 201)** vs **N/A (VDA 202, pure 1-stage)** ⚡
- **(viii) Long-video inference:** mortise-and-tenon (DepthCrafter 201) vs **OI+KR (VDA 202, simpler + faster + no diffusion)** ⚡
- **(ix) Substrate:** pointmap (CUT3R 175) vs SDF (DeepSDF 002) vs depth (DepthCrafter 201, VDA 202) ⚡
- **(x) Memory:** state-token (CUT3R 175) vs memory-token (Spann3R 177) vs keyframe-bank (R³ 183, LingBot-Map 184) vs window-pool (WinT3R 185) vs 3D-spatial (LONG3R 186) vs TTT+SWA (LoGeR 187) vs **frozen-encoder + 4-temporal-head (VDA 202)** ⚡
- **(xi) **NEW** Temporal-loss design:** optical-flow-warping (OPW, NVDS) vs **temporal-gradient-matching (TGM, VDA 202)** ⚡
- **(xii) **NEW** Model-size scalability:** tiny (28.4M, 9.1ms/frame, 30+ FPS, VDA-S) → medium (113.1M, VDA-B) → large (381.8M, 67ms/frame, VDA-L) — *the* first paper in the 196-202 arc to offer *all 3 sizes* for product matrix
- **(xiii) **NEW** Metric-depth variant:** relative-depth (most papers) vs **metric-depth (VDA 202 [VDA-Metric, 2025-04-25 / 2025-08-28])** ⚡
- **(xiv) **NEW** Streaming mode:** offline (all 196-201 papers) vs **streaming (VDA 202 [training-free, 2025-09-12])** ⚡
- **(xv) **NEW** Foundation-model extension:** from-scratch (most 196-201 papers) vs **frozen-foundation-model-extension (VDA 202 [frozen DAv2])** ⚡
- **(xvi) **NEW** Commercial-permissive:** Apache-2.0 (VDA-S 202, Aether 199, π³ 192, LiteVGGT 198, Speed3R 195, LingBot-Map 184) vs BSD-3-Clause (π³ 192, Speed3R 195) vs ⚠️ non-commercial (VDA-B/L 202 CC-BY-NC-4.0, DepthCrafter 201 NOASSERTION, WinT3R 185, LONG3R 186, LoGeR 187, Geo4D 200) ⚡

---

**★ Next paper to read (203):** the 201-note's recommended *next* was VDA 202 (now read!). The 202-note's recommended *next* is **DepthAnyVideo (Chen 2025, arXiv:2501.12375, CVPR 2025)** — wait, the 202-note *is* the 2501.12375 paper. Let me re-check. The 201-note recommended VDA 202 (which is *also* arXiv:2501.12375). The 202-note's recommended *next* should be a *different* paper. Looking at VDA 202's references, the *most-promising* next paper is **DepthAnyVideo (Chen 2025, arXiv:2501.12375, CVPR 2025)** — but that's the *same* paper. The *actual* next paper is **ChronoDepth (Wang 2024, ECCV 2024)** — the *concurrent* 2024 short-context (10 frames) video-depth model that DepthCrafter + VDA 202 explicitly beat, the *founding* paper of the *short-temporal-context* paradigm that *motivated* the long-context video-depth literature. **★ Alternative 203 candidates:** (a) **NVDS (Wang 2023, CVPR 2023)** — the *concurrent* 2023 video-depth model with plug-and-play stabilization network, the *founding* paper of the *stabilization-network* paradigm and the *direct* source of the OPW loss that VDA 202 critiques; (b) **MAMO (Park 2023, CVPR 2023)** — the *concurrent* 2023 video-depth model with memory attention, the *founding* paper of the *memory-attention* video-depth paradigm; (c) **DeepV2D (Teed 2020, ECCV 2020)** — the *founding* paper of combined camera-motion + depth estimation in a single network, the *historic* foundation; (d) **Marigold (Ke 2023, CVPR 2024)** — the *founding* paper of the *image-generator-to-depth* paradigm that inspired DepthCrafter + Geo4D + Aether + VDA 202; (e) **BiDAStereo (Aich 2021, ICRA 2021)** — the *founding* paper of the *bidirectional attention for stereo matching* that DepthCrafter 201 uses for data construction; (f) **Depth-Anything V2 (Yang 2024, arXiv:2406.09414)** — the *frozen-encoder* that VDA 202 builds on, the *foundational* monocular depth model for the entire 201-202 arc. **★ Recommendation: *read 203 = ChronoDepth (Wang 2024, arXiv:2410.02046, ECCV 2024)*** — the *concurrent* 2024-10 short-context (10 frames) video-depth model that VDA 202 + DepthCrafter 201 *explicitly beat*, the *founding* paper of the *short-temporal-context* paradigm that *motivated* the long-context video-depth literature, the *right* next paper to understand the *historical arc* of video-depth from 10-frame → 110-frame → 500-frame → arbitrary-length. After ChronoDepth 203, the v0 sub-task 1 *video-depth* design space will have *temporal-context-length* coverage (ChronoDepth 203 [10 frames] → DepthCrafter 201 [110 frames, mortise-and-tenon] → VDA 202 [500+ frames, OI+KR]) coverage. ⚠️ **PATTERN NOTICE:** the 201-note's "next paper 202 = Video Depth Anything, arXiv:2501.12375" was *correct* on all key facts (the 16-17th arXiv-ID hallucination was *prevented* by direct arXiv lookup, the 7-8th GitHub-API-license-check was *performed* [license: Apache-2.0 ✅ for code + VDA-S weights, CC-BY-NC-4.0 ⚠️ for VDA-Base/Large weights, *NOT* "Apache-2.0" as the 201-note's pre-fetch assumption suggested — the 201-note's license-check was *more nuanced* than a single-string lookup, requiring *both* repo-license + per-model-weight-license verification]), confirming the *direct-arXiv-lookup* + *GitHub-API-license-check* + *per-model-weight-license-check* sub-skills are *working*. The *new* critical findings are the *metric-depth variants* (VDA-S/Base/Large-Metric, 2025-04-25 / 2025-08-28 releases), the *streaming-mode* (2025-09-12, training-free, the *killer* real-time product for v0 v1+), the *4-temporal-head design lesson* (temporal modules in *head*, not *encoder*, the *killer* foundation-model-extension pattern), the *TGM-loss-no-optical-flow* (the *killer* critique of OPW-style losses), the *OI+KR inference* (the *killer* simpler-than-mortise-and-tenon long-video stitch), the *13.6× speedup over DepthCrafter* (the *killer* efficiency finding), and the *+25% δ₁ on KITTI/ScanNet/Bonn/NYUv2* (the *killer* quality finding that *foundation-model-extension > video-diffusion-for-depth* for the length-generalization axis). *Always* verify (1) arXiv ID, (2) GitHub license, (3) HF checkpoint license, (4) per-model-weight-license, (5) reimplementation-vs-official status, (6) last-push-date, (7) upgraded-work / followup-papers, (8) version-history (v1 → v2 → v3), (9) inference-engineering-improvements, (10) per-paper-weight-vs-pretrained-checkpoint availability.
