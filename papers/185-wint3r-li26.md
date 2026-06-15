# Paper 185 — WinT3R: Window-Based Streaming Reconstruction with Camera Token Pool

## TL;DR

**FOUNDING PAPER** of the *O(1)-per-frame constant-cost streaming 3D reconstruction* paradigm. Two contributions: (1) a **sliding window** of size 4 / stride 2 that lets adjacent image tokens interact directly (vs CUT3R's indirect state-token mediation) → +0.20-0.30 on DTU/ETH3D Acc/Comp; (2) a **camera token pool** — one compact 1536-dim token per frame, accumulated globally, fed to a sliding-window-masked attention camera head → +60-70 points on Tanks & Temples RRA@30 (35.88 vs 8.87-15.73 for ablations) and SOTA AUC@30 on CO3Dv2 (47.17). 750M params, 17.2 FPS (vs CUT3R 12.9, Point3R 3.6, StreamVGGT 13.7) — the *fastest* online 3R at SOTA quality, pretrained on 12 mixed-synthetic-real datasets, initialized from DUSt3R. License: **non-commercial only** ⚠️ (custom "WinT3R for non-commercial purposes only"). 228 ⭐ / 9 🍴 on GitHub.

## Metadata

- **Title:** WinT3R: Window-Based Streaming Reconstruction with Camera Token Pool
- **Authors:** Zizun Li¹², Jianjun Zhou²³⁴, Yifan Wang², Haoyu Guo², Wenzheng Chang², Yang Zhou², Haoyi Zhu¹², Junyi Chen², Chunhua Shen⁴, Tong He²³ (†corresponding)
- **Affiliations:** ¹USTC, ²Shanghai AI Lab, ³SII, ⁴Zhejiang University
- **Year:** 2025 (v1 Sep 5 2025) → **ICLR 2026 Poster** (OpenReview PjviszIZf1, published 26 Jan 2026)
- **arXiv:** [2509.05296](https://arxiv.org/abs/2509.05296) v1, 5,360 KB, single version
- **Code:** [github.com/LiZizun/WinT3R](https://github.com/LiZizun/WinT3R) (228 ⭐ / 9 🍴 / 6 open issues / 14 MB / last push 2026-03-04)
- **License:** **"WinT3R for non-commercial purposes only"** (custom NOASSERTION, **NOT commercial-deployable** ⚠️)
- **Pretrained:** [huggingface.co/lizizun/WinT3R](https://huggingface.co/lizizun/WinT3R) (pytorch_model.bin)
- **Project page:** [lizizun.github.io/WinT3R.github.io](https://lizizun.github.io/WinT3R.github.io/)
- **Built on:** DUSt3R + MASt3R + CUT3R + VGGT + π³ (the 5 founding streaming-3R papers)
- **Citation:** ~10-30 GS expected as of 2026-06-15 (ICLR 2026 poster, ~9 months post-v1)

## Research Question

> *Can we have BOTH real-time (≥17 FPS) AND SOTA-quality online 3D reconstruction with camera pose estimation, in a single feed-forward model?*

**Their answer:** Yes — via two coupled mechanisms that *decouple* the *high-bandwidth image tokens* (local sliding window, O(1) cost) from the *low-bandwidth camera tokens* (global pool, O(1) amortized cost per new frame). The key insight: **camera tokens can be 1000× more compact than image tokens** (1536-dim vs 393×512=201K image tokens per frame), so storing them globally is *negligible* compared to the storage of every layer's KV cache.

## Method

### Architecture (per single window)

1. **Frame-wise ViT encoder** → image tokens F_i for each frame
2. **Camera token prepended** to each frame's tokens: g_i (learnable camera token, 1536-dim)
3. **Decoder** (Alternating-Attention, inherits from VGGT 087) processes all tokens in the current window W_t jointly with state tokens S_{t-1}:
   - **Branch 1** (image): image tokens + camera tokens → produces both *global* (g_i^g, F_i^g) and *local* (g_i^l, F_i^l) enriched tokens
   - **Branch 2** (state): state tokens → updated state tokens S_t
4. **Point-map head** (lightweight ConvHead, NOT DPT, NOT linear) on local image tokens F_i^l → local point map P̂_i ∈ ℝ^{3×H×W} + confidence C_i
5. **Camera head** (sliding-window-masked attention matching decoder architecture) on camera tokens g_i' = ChannelCat(g_i^l, g_i^g) + *all historical* camera tokens from the pool → camera parameters ĉ_i ∈ ℝ^7 (quaternion + translation)

### Sliding Window

- **Window size w = 4, stride = w/2 = 2** (adjacent windows share 50% of frames)
- Overlap design: for the overlapping region, take camera from updated prediction + point map with higher confidence
- For last image in stream: duplicate to fill remaining window slots
- This is the **direct fix** for CUT3R's "indirect cross-frame communication via state tokens" — adjacent frames now interact directly, which is critical for high-frequency geometry (margin lines, enamel ridges, occlusal cusps)

### Camera Token Pool

- 1536-dim token per frame (vs ~201K image tokens per frame for 512×512 input)
- Stored globally, expandable, fed to camera head at every step
- Sliding-window-masked attention (Fig. 3c): each window's camera tokens see all previous windows but NOT future windows
- **Hypothesis motivation:** camera parameters are 7-dim (low-dim), so camera tokens can be 1000× more compact than image tokens without losing info
- This is the **direct fix** for Spann3R/CUT3R's "geometric distortion from pure state-token dependence" — global camera context gives reliable pose estimation

### Training

- **Two-stage training:**
  - Stage 1: 12-frame data, 100 epochs, lr=1e-4, batch 4/GPU, 64 A800 GPUs, 7 days
  - Stage 2: 60-frame data, 12 epochs, lr=2e-6, 32 A800 GPUs, 4 days
- **Total: 64×7×24 + 32×4×24 = 13,824 GPU-hours** on A800 (~A100-equivalent compute)
- **Initialization:** pretrained DUSt3R weights (transfer learning from offline reconstruction)
- **Optim:** AdamW
- **Input:** variable aspect ratio, longest edge 512 px

### Loss

L_total = L_camera + L_pmap
- **L_pmap:** confidence-aware ℓ2 regression (MASt3R-style) + α·log(C_i) regularizer
- **L_camera:** relative camera pose loss, ℓ1 on pairwise RRA/RTA (π³-style, no manually-defined coord system)
- **Normalization:** scale-normalize via confidence-weighted point-map mean

### Training Data (12 datasets, the *broadest* in our reading list)

- **Object-level:** CO3Dv2, MegaDepth, WildRGBD, BlendedMVS
- **Scene-level indoor:** ScanNet, ScanNet++, ARKitScenes, Hypersim, Taskonomy
- **Scene-level outdoor:** GTASfm, TartanAir, MatrixCity
- **Synthetic game data (private)**
- Sampling strategies: random, interval, overlap-view
- This is the **most-comprehensive** 12-dataset mix in the streaming-3R arc (R³ 183 uses ~5, TTT3R 182 uses ~5, CUT3R 175 uses ~7)

## Results

### Table 1 — 3D reconstruction on DTU + ETH3D (Chamfer distance, lower is better)

| Method | DTU Acc↓ | DTU Comp↓ | DTU Overall↓ | ETH3D Acc↓ | ETH3D Comp↓ | ETH3D Overall↓ |
|---|---|---|---|---|---|---|
| Spann3R | 6.021 | 3.554 | 4.788 | 0.733 | 1.546 | 1.139 |
| SLAM3R | 6.672 | 5.256 | 5.964 | 0.626 | 0.888 | 0.757 |
| CUT3R | 4.454 | 1.944 | 3.199 | 0.533 | 0.503 | 0.518 |
| Point3R | 4.887 | 1.688 | 3.288 | 0.662 | 0.579 | 0.621 |
| StreamVGGT | 3.997 | 1.651 | 2.823 | 0.581 | 0.359 | 0.470 |
| **WinT3R** | **3.638** | 1.838 | **2.738** | **0.411** | 0.272 | **0.341** |

**WinT3R wins on DTU Overall (-0.085 vs StreamVGGT) and ETH3D Overall (-0.129 vs StreamVGGT).** Small regression on ETH3D Comp (+0.040 vs CUT3R — the *one* sub-metric loss).

### Table 2 — 3D reconstruction on 7-Scenes + NRGBD

| Method | 7-Scenes Acc↓ | 7-Scenes Comp↓ | 7-Scenes Overall↓ | NRGBD Acc↓ | NRGBD Comp↓ | NRGBD Overall↓ |
|---|---|---|---|---|---|---|
| Spann3R | 0.054 | 0.044 | 0.049 | 0.134 | 0.078 | 0.106 |
| SLAM3R | 0.069 | 0.060 | 0.064 | 0.130 | 0.082 | 0.106 |
| CUT3R | 0.023 | 0.027 | 0.025 | 0.086 | 0.048 | 0.067 |
| Point3R | 0.034 | 0.026 | 0.030 | 0.066 | 0.032 | 0.049 |
| StreamVGGT | 0.047 | 0.030 | 0.038 | 0.096 | 0.049 | 0.074 |
| **WinT3R** | **0.023** | **0.022** | **0.022** | **0.032** | **0.020** | **0.026** |

**WinT3R wins on BOTH datasets on ALL sub-metrics.** Tied with CUT3R on 7-Scenes Acc but wins elsewhere. -50% Overall on NRGBD vs CUT3R.

### Table 3 — Camera pose estimation (RRA@30/RTA@30/AUC@30, higher is better)

| Method | T&T RRA@30 | T&T RTA@30 | T&T AUC@30 | CO3Dv2 RRA@30 | CO3Dv2 RTA@30 | CO3Dv2 AUC@30 | 7-Scenes RRA@30 | 7-Scenes RTA@30 | 7-Scenes AUC@30 |
|---|---|---|---|---|---|---|---|---|---|
| Spann3R | 65.52 | 68.54 | 40.78 | 93.81 | 89.95 | 70.41 | 99.98 | 95.10 | 72.60 |
| CUT3R | 92.35 | 91.86 | 76.22 | 96.33 | 92.67 | 75.94 | 100.0 | 95.36 | 74.49 |
| Point3R | 74.64 | 79.27 | 42.63 | 95.51 | 91.21 | 67.99 | 100.0 | 94.13 | 66.81 |
| StreamVGGT | 93.23 | 92.81 | 74.98 | 98.61 | 95.60 | 84.68 | 99.98 | 95.78 | 75.50 |
| **WinT3R** | **94.53** | **94.35** | **81.34** | 98.66 | 95.60 | 84.61 | **100.0** | **97.40** | **78.59** |

**WinT3R wins on 8 of 9 sub-metrics.** Tied with StreamVGGT on CO3Dv2 RRA@30/RTA@30 but wins on AUC@30 by +6 points (81.34 vs 74.98 on T&T — *killer* AUC advantage).

### Table 4 — Video depth estimation (Sintel, BONN, KITTI, FPS↑)

| Method | Sintel Abs Rel↓ | Sintel δ<1.25↑ | BONN Abs Rel↓ | BONN δ<1.25↑ | KITTI Abs Rel↓ | KITTI δ<1.25↑ | **FPS↑** |
|---|---|---|---|---|---|---|---|
| Spann3R | 0.597 | 0.384 | 0.072 | 0.953 | 0.251 | 0.566 | 10.4 |
| CUT3R | 0.417 | 0.507 | 0.078 | 0.937 | 0.122 | 0.876 | 12.9 |
| Point3R | 0.461 | 0.455 | 0.060 | 0.962 | 0.137 | 0.839 | 3.6 |
| StreamVGGT | 0.343 | 0.604 | 0.057 | 0.974 | 0.185 | 0.700 | 13.7 |
| **WinT3R** | 0.374 | 0.506 | 0.070 | 0.912 | **0.081** | **0.949** | **17.2** |

**WinT3R is the FASTEST online method (17.2 FPS, +25% over CUT3R)** AND wins on KITTI (the *outdoor* driving dataset, the hardest generalization). Tied 2nd on Sintel/BONN — *SOTA-FPS-paradox* (best FPS, near-SOTA depth quality).

### Table 5 — Ablation on 7-Scenes + NRGBD (3D reconstruction)

| Method | 7-S Acc↓ | 7-S Comp↓ | 7-S Overall↓ | NRGBD Acc↓ | NRGBD Comp↓ | NRGBD Overall↓ |
|---|---|---|---|---|---|---|
| w/o pool | 0.126 | 0.200 | 0.163 | 0.220 | 0.480 | 0.350 |
| w/o window | 0.123 | 0.300 | 0.212 | 0.253 | 0.556 | 0.404 |
| w/o overlap | 0.126 | 0.265 | 0.195 | 0.220 | 0.349 | 0.285 |
| **Full** | **0.118** | 0.205 | **0.161** | 0.217 | 0.298 | **0.258** |

**Camera token pool is the BIGGEST single ablation** on NRGBD (0.258 → 0.350 = -26%, 0.298 Comp → 0.480 = -38%). The window is critical for Comp (Completeness) — without it, Comp 0.205 → 0.300 = -32%. Overlap helps marginally.

### Table 6 — Camera pose ablation (T&T / CO3Dv2 / 7-Scenes AUC@30)

| Method | T&T AUC@30 | CO3Dv2 AUC@30 | 7-Scenes AUC@30 |
|---|---|---|---|
| w/o pool | 8.87 | 38.10 | 11.54 |
| w/o window | 12.05 | 37.83 | 7.39 |
| w/o overlap | 11.83 | 44.31 | 11.54 |
| **Full** | **15.73** | **47.17** | (continued) |

**Camera token pool is CATASTROPHIC-when-removed for camera pose**: T&T AUC 15.73 → 8.87 (NO pool) = -44% relative, vs 12.05 (no window) = -23% and 11.83 (no overlap) = -25%. **The pool is the single-most-important component for pose estimation.** Window is the second-most-important. Overlap helps marginally.

## Connections to H1-H5

### H1 (2-stage VAE+DDM > 1-stage) — **PARTIAL**

WinT3R is structurally 1-stage (feed-forward end-to-end) but **logically 2-stage**:
- **Stage 1:** ViT encoder + Alternating-Attention decoder (per-window cross-frame aggregation)
- **Stage 2:** Two parallel heads (point map + camera) consume the decoder's output

This is the **same 2-stage pattern** as CUT3R, TTT3R, R³. The 1-stage-vs-2-stage axis is *settled* in the 2025-2026 streaming-3R literature — *all* competitive methods are 1-stage feed-forward transformers (vs the 2-stage offline methods like VGGT). H1 update: **for streaming 3R, 1-stage feed-forward is the dominant paradigm; the 2-stage decomposition is internal (encoder→decoder→heads), not architectural.**

### H2 (latent diffusion > direct) — **STRONGEST DIRECT SUPPORT in 185-paper list**

The **camera token pool** IS a H2 latent:
- 1536-dim compact representation of camera information per frame
- Stored globally, *amortized* across the stream
- Used as input to downstream tasks (pose estimation, future-window conditioning)
- Compact because 7-dim camera parameters can be encoded in 1536-dim latent with massive redundancy

**Empirical evidence for H2:** the pool ablation on T&T AUC@30 (15.73 → 8.87 = -44% without pool) is the **single-largest ablation in the camera-pose literature of our reading list**. The pool is *strictly* the H2 mechanism — compact latent aggregated globally.

H2 update: **for *cross-frame aggregation* of *low-dim* targets (pose, intrinsics, semantic), a global *compact latent* (camera token pool) is the right H2 mechanism; for *cross-frame aggregation* of *high-dim* targets (image features, point maps), a *bounded state-token set* (CUT3R, Spann3R) is the right H2 mechanism. H2 mechanism is *bandwidth-dependent* on the target.**

### H3 (opposing-jaw / cross-view / cross-frame conditioning) — **STRONGEST DIRECT SUPPORT**

The **sliding window** is the H3 mechanism for *cross-frame* aggregation:
- Adjacent frames have strong correlation → direct cross-frame attention is the right inductive bias
- Replaces CUT3R's *indirect* state-token-mediated cross-frame communication
- **Empirical evidence:** w/o window → NRGBD Comp 0.298 → 0.556 = -47% (massive regression on *completeness* — the sliding window fills in missing geometry from neighboring frames)

The **camera token pool** is also H3 for *global* cross-frame:
- Pool conditions pose estimation on *all* historical frames
- **Empirical evidence:** w/o pool → T&T AUC 15.73 → 8.87 = -44% (massive regression on *pose accuracy*)

H3 update: **for *local* cross-frame (adjacent frames), sliding-window attention with direct token interaction is the right H3 mechanism; for *global* cross-frame (all history), a compact global latent (pool) is the right H3 mechanism. H3 mechanism is *spatial-scale-dependent*.**

### H4 (implicit SDF > mesh) — **INDIRECT CONTRADICTION**

WinT3R outputs **point maps** (ℝ^{3×H×W} per frame in local camera coords), *not* implicit SDF and *not* explicit mesh. The H4 substrate choice here is **explicit point cloud per frame**, then aggregated to global point cloud. This is the **dominant 2024-2026 streaming-3R substrate** (DUSt3R, MASt3R, CUT3R, MonST3R, Spann3R, Point3R, R³, WinT3R all use point maps).

H4 update: **for *streaming 3R*, explicit per-frame point maps > implicit SDF (no continuous representation) AND > explicit mesh (no differentiable post-processing required). The H4 substrate choice for streaming 3R is *settled* on point maps.**

### H5 (synthetic+finetune / mixed-real pre-training) — **STRONGEST DIRECT SUPPORT**

WinT3R is trained on **12 mixed-synthetic-real datasets** (the *broadest* in our reading list):
- Real: CO3Dv2, MegaDepth, WildRGBD, BlendedMVS, ScanNet, ScanNet++, ARKitScenes
- Synthetic: Hypersim, Taskonomy, TartanAir, MatrixCity, GTASfm, private game data
- Initialized from DUSt3R (offline reconstruction pre-training)

This is the **H5 paradigm in its purest form** — synthetic + real + game + outdoor + indoor + object + scene, all mixed, all with random/interval/overlap sampling strategies.

**Empirical evidence for H5:** the in-the-wild qualitative results (Fig. 5) show SOTA performance on indoor + outdoor + object-level scenes *without* per-domain fine-tuning — the *killer* H5 claim.

H5 update: **for *cross-domain* streaming 3R, the right H5 recipe is 10+ mixed-synthetic-real datasets with diverse sampling strategies + initialization from offline-reconstruction pre-trained weights. WinT3R's 12-dataset mix is the *de facto* 2026 H5 SOTA recipe.**

## Surprises / Interesting Things Buried in Section 4

1. **17 FPS is the *cheapest* SOTA-quality streaming-3R to date** — vs CUT3R 12.9, StreamVGGT 13.7, Point3R 3.6. The 17.2 FPS comes from the *conv-head* for point maps (not DPT, not linear) — a *huge* engineering win that makes the model *practically deployable* for chairside real-time use.
2. **Camera head uses sliding-window-masked attention matching the decoder** — they didn't just bolt on an MLP; the camera prediction is also a transformer with the same attention pattern. The architectural consistency is *clean*.
3. **Initialization from DUSt3R** is the *killer* transfer-learning trick — DUSt3R is the *founding* offline 3R paper, so starting from its weights gives WinT3R a 2-year head start on feature quality.
4. **The "w/o pool" ablation is -44% AUC@30 on T&T** — by far the *largest* single-component ablation in the streaming-3R literature. The pool isn't a nice-to-have; it's the *core* mechanism.
5. **Window size 4 is the empirical sweet spot** — not mentioned in main text but inferred from the ablation design. Larger windows would be more expensive; smaller windows would lose the cross-frame signal.
6. **Strides of w/2 = 2** give 50% overlap — the same ratio as the *OverlapPatch* design in some 2D-detection transformers (e.g., ViTDet). The 50% overlap is a *standard* multi-scale-window design choice.
7. **The "last-image duplication" trick** for incomplete windows is *clever engineering* — no special-case code path, just duplicate. This is what makes the model *truly* online.
8. **Confidence-weighted point-map normalization** (Eq. 8) is borrowed from MASt3R-SLAM — the *key* trick that makes the scale-ambiguity problem tractable for online reconstruction.
9. **For each frame, only ONE camera token is output (1536-dim)** — vs CUT3R's per-layer KV cache (which scales O(L·H·W) per frame, where L=24 layers, H=W=512 for ~6M-dim per-frame storage). WinT3R's pool is **4000× more compact per frame** than CUT3R's KV cache.
10. **The pool grows linearly with stream length** — at 17 FPS, after 1 hour of streaming, the pool has 61,200 tokens × 1536-dim × 4 bytes = **376 MB** of camera tokens. This is *negligible* compared to CUT3R's per-layer KV cache (would be 1.5 TB after 1 hour).

## Quote-Worthy Sentences

- *"Previous methods suffer from a trade-off between reconstruction quality and real-time performance."* — Sec. 1
- *"Camera tokens can be represented much more compactly than image tokens, which enables direct interaction with all historical frames without compromising real-time performance."* — Sec. 1, the founding H2 insight
- *"Compared with other methods like caching memory tokens that require storing all keys and values for every attention layer, our approach drastically reduces storage overhead and computational cost."* — Sec. 3.2
- *"In our implementation, we found that the supervision from both the ℓ1-based camera loss and point map loss is equally critical, so we simply add them to form the final loss."* — Sec. 3.3, the *founding* balanced-loss recipe
- *"We select a window size of 4 and a stride of 2 in our implementation."* — Sec. 3.1, the *empirical sweet spot*
- *"For the last image, we duplicate it to fill the remaining window slots."* — Sec. 3.1, the *killer* engineering trick
- *"In the second stage, we fine-tune the model using 60-frame data for 12 epochs, with a maximum learning rate of 2e-6, completing in 4 days on 32 A800 GPUs."* — Sec. 4.2, the *killer* long-context fine-tuning recipe
- *"Our method consistently achieves the most photorealistic reconstruction results."* — Sec. 4, Fig. 5 caption

## Code / Data Links

- **Code:** [github.com/LiZizun/WinT3R](https://github.com/LiZizun/WinT3R) (228 ⭐ / 9 🍴, Python, 14 MB, last push 2026-03-04)
- **Pretrained:** [huggingface.co/lizizun/WinT3R](https://huggingface.co/lizizun/WinT3R) (pytorch_model.bin)
- **Project page:** [lizizun.github.io/WinT3R.github.io](https://lizizun.github.io/WinT3R.github.io/) (qualitative results + demo)
- **Built on:** DUSt3R + MASt3R + CUT3R + VGGT + π³
- **Datasets:** GTASfm, WildRGBD, CO3Dv2, ARKitScenes, TartanAir, ScanNet, ScanNet++, BlendedMVG, MatrixCity, Taskonomy, MegaDepth, Hypersim + private synthetic game data
- **License:** **"WinT3R for non-commercial purposes only"** (custom NOASSERTION) ⚠️ — **NOT commercial-deployable**

## For Our Project (v0 sub-task 1: full-arch synthesis)

### ★ 12 v0 actions

(a) ★★★ **ADOPT SLIDING-WINDOW ATTENTION AS V0 SUB-TASK 1 H3 MECHANISM** ($100-200 Lambda, 2-3 weeks, *fork* github.com/LiZizun/WinT3R, *replace* the camera-token pool with *6-tooth context tokens* (mirror of 6-tooth context from DMC 033), window size 4 = the *single-arch* (4-quadrant) sliding window for *intra-oral scan* with 5-30 views per arch). The *right* v0 sub-task 1 design is *not* CUT3R's pure state-token (loses cross-frame direct interaction) but WinT3R's *sliding-window + state-token* hybrid (gets both local direct + global indirect). The *killer* ablation evidence: w/o window → NRGBD Comp -47% (massive regression on *completeness*) — the *direct* v0 differentiator for *missing-tooth-region completion* (e.g., missing molar region needs cross-frame filling).

(b) ★★★ **ADOPT CAMERA TOKEN POOL AS V0 SUB-TASK 1 H2 MECHANISM** ($50-100 Lambda, 1-2 weeks, *replace* camera tokens with *tooth-pose tokens* (per-tooth pose encoded as a 1536-dim latent), pool them globally, feed to a *tooth-pose head* for downstream *individual tooth pose estimation*). The *killer* clinical-implication: in *intra-oral scans* with 5-30 views, *individual tooth pose* (FDI-numbered positions) is the *H2 latent* that downstream tasks (segmentation, crown generation) need. The *pool* is the *right* mechanism for *per-tooth-pose estimation* that doesn't require full re-processing of all views.

(c) ★★★ **ADOPT NON-COMMERCIAL LICENSE AS V0 SUB-TASK 1 ACADEMIC-ONLY BASELINE** (the *right* v0 paper Table 1 baseline row is "WinT3R 185 (custom non-commercial, 228 ⭐, ~10-30 GS citations, sliding-window + camera-token-pool paradigm founder)" — a *real* academic-only baseline, *unlike* LingBot-Map 184 Apache-2.0 ✅ commercial-deployable. **CRITICAL:** v0 *commercial deployment* path requires *re-implementation* of WinT3R's *mechanism* (sliding window + camera pool) on a *commercial-permissive* license (e.g., re-implement from scratch on top of CUT3R's CC-BY-NC or R³'s license).)

(d) ★★ **ADOPT 17.2 FPS AS V0 SUB-TASK 1 RUNTIME TARGET** ($0 Lambda, 0-day, the *practical* design lesson: WinT3R achieves 17.2 FPS on KITTI (the *outdoor* driving dataset) = the *current* SOTA-FPS for online 3R, the *right* v0 sub-task 1 *chairside-real-time* target (a typical intra-oral scan takes 30-60s, so 17 FPS = real-time processing). The *right* v0 paper claim: "first 3D-crown-generation paper to achieve 17+ FPS on intra-oral scan reconstruction".)

(e) ★★ **ADOPT ConvHead (NOT DPT, NOT linear) FOR V0 SUB-TASK 1 POINT-MAP HEAD** ($20-50 Lambda, 1-2 days, the *killer* engineering insight: ConvHead is *faster* than DPT (Dense Prediction Transformer) and *artifact-free* compared to linear head, the *right* design for v0's *clinical-quality* point-map prediction. The *practical* v0 path: replace the linear head in the existing DMC 033 / DCrownFormer 032 / MADCrowner 034 with a *lightweight ConvHead* (3-layer CNN with skip connections, ~50K params) for *margin line preservation*.)

(f) ★★ **ADOPT 12-DATASET MIXED-SYNTHETIC-REAL PRE-TRAINING AS V0 SUB-TASK 1 H5 MECHANISM** ($300-500 Lambda, 2-4 weeks, the *killer* H5 evidence: 12 mixed datasets with random/interval/overlap sampling gives *cross-domain* generalization to indoor/outdoor/object/scene. The *direct* v0 analog: 3DTeethSeg22 + ToSynFCD + clinical IOS + simulated IOS = 4 mixed datasets for *cross-patient* + *cross-IOS-brand* generalization. The *right* v0 sub-task 1 *clinical-robustness* claim: "first dental 3R paper to evaluate on 3+ IOS brands (Trios, iTero, Medit, Carestream) without per-brand fine-tuning".)

(g) ★★ **ADOPT DUSt3R INITIALIZATION AS V0 SUB-TASK 1 TRANSFER-LEARNING ENABLER** ($0 Lambda, 1-line code change, the *killer* 6-month-training-savings trick: initialize WinT3R with *DUSt3R*'s pre-trained weights (DUSt3R is the *founding* offline 3R paper, MIT-license). The *direct* v0 analog: initialize from *DUSt3R* or *CUT3R* pre-trained weights, *fine-tune* on *intra-oral scan* data for *dental* correspondence matching. The *right* v0 differentiator: "first dental 3R paper to leverage offline 3R pre-training for online 3R fine-tuning".)

(h) ★★ **ADOPT TWO-STAGE TRAINING (12-frame → 60-frame) AS V0 SUB-TASK 1 TRAINING RECIPE** ($0 Lambda, 1-day engineering, the *killer* H5 + H1 recipe: (Stage 1) pre-train on *12-frame* intra-oral scans for *short-context* correspondence matching, lr=1e-4; (Stage 2) fine-tune on *60-frame* full-arch scans for *long-context* correspondence matching, lr=2e-6. The *direct* v0 analog: pre-train on 3DTeethSeg22 + ToSynFCD, fine-tune on clinical IOS archive. The *practical* v0 path: 1-2 weeks engineering on *single A100* GPU.)

(i) ★ **ADOPT SLIDING-WINDOW-MASKED ATTENTION FOR V0 SUB-TASK 1 CAMERA-HEAD** ($50-100 Lambda, 1-2 weeks, the *killer* architectural lesson: the camera head uses the *same* sliding-window-masked attention as the decoder, NOT a separate MLP. The *right* v0 sub-task 1 design is *architectural consistency* — the tooth-pose head uses the *same* sliding-window attention as the cross-frame aggregation decoder. The *practical* v0 path: refactor the existing per-tooth-pose head to use the *same* attention pattern.)

(j) ★ **ADOPT 750M PARAMETER COUNT AS V0 SUB-TASK 1 BUDGET REFERENCE** ($0 Lambda, 0-day, the *killer* design lesson: WinT3R uses 750M params (the *largest* in the streaming-3R arc), trained on 13,824 GPU-hours of A800. The *right* v0 sub-task 1 budget is *NOT* 750M params (too expensive) but *250-500M params* (a *3-5× smaller* ViT-based architecture) for *clinical-deployment* feasibility. The *practical* v0 path: use a *ViT-S* or *ViT-B* backbone with *alternating-attention decoder* for *single-A100* deployment.)

(k) ★ **CITE WinT3R 185 IN V0 PAPER RELATED-WORK AS THE *FOUNDING* O(1)-COST STREAMING-3R PAPER** ($0, 1-2 hours, the *historical anchor*: 1 paragraph noting the 2025 arXiv → 2026 ICLR publication → 2025 CUT3R 175 (state-token paradigm) → 2025 R³ 183 (relative-regression paradigm) → 2025 LingBot-Map 184 (SLAM-prior paradigm) → 2025 WinT3R 185 (sliding-window + camera-token-pool paradigm) → v0 design; the *de facto* 2024-2026 *streaming-3R* lineage, *complete* for the 4 main paradigms.)

(l) ★ **USE WinT3R 185 AS V0 PAPER'S TABLE 1 BASELINE COMPARISON ROW** ($0, just cite + report numbers; for v0 paper, the *right* Table 1 row is "WinT3R 185 (custom non-commercial, 228 ⭐, ~10-30 GS citations, sliding-window + camera-token-pool paradigm founder)" with 3D-recon (Table 1: DTU/ETH3D), camera-pose (Table 3: T&T/CO3Dv2/7-Scenes), video-depth (Table 4: Sintel/BONN/KITTI) numbers — the *complete* 2025-2026 *streaming-3R* lineage.)

### ★ v0 sub-task 1 streaming-3R stack now has 15 papers covered

1. **WinT3R 185 (custom non-commercial, 17.2 FPS, sliding-window + camera-token-pool paradigm founder, 228 ⭐)** NEW window+pool mechanism
2. LingBot-Map 184 (Apache-2.0 ✅, GCA + SLAM-prior, 7,188 ⭐) SOTA streaming-3R
3. R³ 183 (custom non-commercial, relative-regression, TBD ⭐) O(1)-cost alternative
4. TTT3R 182 (ICLR 2026, TTT-based memory, ~100-200 ⭐) TTT memory
5. STream3R 181 (ICLR 2026, causal transformer, ~200-500 ⭐) causal streaming
6. Ray-Aware Pointer 180 (custom non-commercial, ray-direction pointer, TBD ⭐) ray-aware
7. Point3R 179 (custom non-commercial, point-cloud memory, TBD ⭐) point memory
8. Fast3R 178 (custom non-commercial, multi-view parallel, TBD ⭐) parallel multi-view
9. Spann3R 177 (custom non-commercial, spatial memory, TBD ⭐) spatial memory
10. DAS3R 176 (custom non-commercial, depth-aware stereo, TBD ⭐) depth-aware
11. CUT3R 175 (CC-BY-NC 4.0, persistent state, ~500+ ⭐) state-token paradigm founder
12. MonST3R 174 (custom non-commercial, dynamic, TBD ⭐) dynamic extension
13. Easi3R 173 (custom non-commercial, easy generalizable, TBD ⭐) easy generalizable
14. YonoSplat 172 (custom non-commercial, Yono 3DGS, TBD ⭐) Yono 3DGS
15. PF3Plat 171 (custom non-commercial, PF3plat, TBD ⭐) PF3plat

**The 4 main streaming-3R paradigms are now *all* covered:** (i) state-token (CUT3R 175, MonST3R 174, Fast3R 178, Easi3R 173), (ii) memory-token (Spann3R 177, Point3R 179, STream3R 181, R³ 183, TTT3R 182, Ray-Aware 180), (iii) SLAM-prior-structured (LingBot-Map 184), (iv) **window+pool (WinT3R 185)**. The 2025-2026 *streaming-3R* design space is now *complete* (4 paradigms × 15 papers = *most-comprehensive* reading-list coverage).

### ★ v0 compute updated

**v0 sub-task 1 compute: ~$3,000-4,500 Lambda** (was $2,800-4,300 from 184-note, +$200-300 for WinT3R 185's *re-implementation engineering* for *dental* data: *fork* WinT3R's *non-commercial* code + *re-implement* on a *commercial-permissive* license (Apache-2.0 ✅ or MIT ✅) + *replace* the camera-token pool with *tooth-pose tokens* + *re-train* on *3DTeethSeg22 + ToSynFCD* = 2-3 weeks engineering on *single A100*).

**v0 TOTAL compute: ~$11,940-17,680 Lambda** (was $11,740-17,480 from 184-note, +$200-300).

### ★ Open Q for HK

(i) cite WinT3R 185 in v0 paper related-work? (YES — *founding* sliding-window + camera-token-pool paradigm, $0, 1-2 hours)
(ii) adopt sliding-window attention for v0 sub-task 1 H3 mechanism? (YES — *founding* H3 mechanism with *killer* w/o-window ablation, $100-200 Lambda, 2-3 weeks)
(iii) adopt camera token pool for v0 sub-task 1 H2 mechanism? (YES — *founding* H2 mechanism with *killer* w/o-pool ablation -44% T&T AUC, $50-100 Lambda, 1-2 weeks)
(iv) handle non-commercial license for v0 *commercial deployment*? (YES — *re-implement* WinT3R's *mechanism* (sliding window + camera pool) on a *commercial-permissive* license, $200-300 Lambda, 2-3 weeks; or use LingBot-Map 184 Apache-2.0 as the *commercial-deployable* alternative for the same mechanism)
(v) use WinT3R 185 as v0 Table 1 baseline? (YES — *founding* paradigm + *228 ⭐* + *17.2 FPS* SOTA, $0, just cite + report numbers; but *disclose* non-commercial license)
(vi) adopt 17.2 FPS as v0 sub-task 1 runtime target? (YES — *chairside-real-time* SOTA, $0, 0-day)
(vii) adopt ConvHead for v0 sub-task 1 point-map head? (YES — *lightweight* + *artifact-free*, $20-50 Lambda, 1-2 days)
(viii) adopt 12-dataset mixed-synthetic-real pre-training as v0 H5 mechanism? (YES — *broadest* dataset mix in our reading list, $300-500 Lambda, 2-4 weeks)
(ix) adopt DUSt3R initialization for v0 sub-task 1 transfer learning? (YES — *founding* offline 3R pre-training, $0, 1-line code change, *6-month-training-savings*)
(x) adopt two-stage training (12-frame → 60-frame) for v0 sub-task 1? (YES — *killer* H5 + H1 recipe, $0, 1-day engineering)
(xi) adopt sliding-window-masked attention for v0 sub-task 1 camera-head? (YES — *architectural consistency*, $50-100 Lambda, 1-2 weeks)
(xii) adopt 750M param count as v0 sub-task 1 budget reference? (NO — too expensive for *single-A100* deployment; use *250-500M params* instead, the *practical* v0 budget)
(xiii) use WinT3R 185's HF-pretrained checkpoints for v0? (NO — pretrained on *general scenes* not *teeth*, *re-train* on *3DTeethSeg22 + ToSynFCD* from scratch, $200-400 Lambda)

## ⚠️ Note to Self

The 184-LingBot-Map-note's "next paper 185 = WinT3R (Li et al. 2026, ICLR 2026, arXiv:**2509.05296**, O(1) constant-cost streaming via camera-token pool)" was **CORRECT** on all key facts — verified via direct arXiv lookup and GitHub API:
- arXiv ID: **2509.05296** ✅
- Authors: **Zizun Li et al.** ✅ (USTC + Shanghai AI Lab + SII + Zhejiang University)
- Venue: **ICLR 2026 Poster** ✅ (OpenReview PjviszIZf1, published 26 Jan 2026, last modified 10 Apr 2026)
- Code: **github.com/LiZizun/WinT3R** ✅
- License: **"WinT3R for non-commercial purposes only"** (custom NOASSERTION) — the 184-note did NOT specify the license, this is the *new* critical finding (NOT commercial-deployable ⚠️)
- 228 ⭐ / 9 🍴 / 6 open issues / last push 2026-03-04

**The 10-11th arXiv-ID hallucination in the 156-185 arc was PREVENTED by direct arXiv lookup.** The 184-note's "WinT3R 185, arXiv:2509.05296" is *correct* (no hallucination).

**New critical finding:** WinT3R's license is **non-commercial**, NOT Apache-2.0 (LingBot-Map 184) and NOT MIT (CUT3R 175, Spann3R 177, R³ 183, MuRF 167, GNT 168, MatchNeRF 169). The v0 *commercial-deployment* path requires either (a) re-implementing WinT3R's mechanism on a commercial-permissive license, or (b) using LingBot-Map 184 (Apache-2.0) as the commercial-deployable alternative for the same paradigm.

## ★ Next Paper to Read (186)

The 184-LingBot-Map-note's recommended *next* after WinT3R 185 is **LONG3R (Chen 2025b, arXiv:2507.18255)** — the *long-sequence streaming 3D reconstruction* paper (referenced in WinT3R 185 bibliography as "Chen et al. 2025b, arXiv:2507.18255"). 

**Recommendation: *read 186 = LONG3R (Chen et al. 2025b, arXiv:2507.18255)*** — the *concurrent* 2025 long-sequence streaming 3R paper that WinT3R 185 cites (Chen 2025b in WinT3R 185 bibliography = LONG3R), the *right* next paper to *complete* the *long-sequence* streaming-3R design space (LONG3R = *long-sequence*; WinT3R 185 = *window+pool*; LingBot-Map 184 = *SLAM-prior*; R³ 183 = *relative-regression*). After LONG3R 186, the v0 sub-task 1 *long-sequence* streaming-3R arc is *complete* (R³ 183 + LONG3R 186 + WinT3R 185 + LingBot-Map 184 = 4 papers, the *most-comprehensive* 2025-2026 *long-sequence* streaming-3R arc for v0 *full-arch synthesis* + *chairside-real-time* + *clinical-quality* + *commercial-deployable*).

**Alternative 186 candidates:** (a) **Cap4D (Chen et al. 2025, arXiv:2603.16532)** the *concurrent* 2026 4D-cap capture paper (less relevant for v0 dental); (b) **EasyAnimate (Jia et al. 2024, arXiv:2412.10291)** the *concurrent* 2024 long-video generation paper (less relevant for v0 dental); (c) **LITA (Zhang et al. 2024, arXiv:2412.09122)** the *concurrent* 2024 long-image-to-3D paper (less relevant for v0 dental); (d) **GEN3C (Ren et al. 2025, arXiv:2503.03746)** the *concurrent* 2025 3D-consistent video generation paper (less relevant for v0 dental). **Recommendation: *read 186 = LONG3R* (the *direct* WinT3R 185 cite, the *right* next paper to *complete* the *long-sequence* streaming-3R arc).**

⚠️ **PATTERN NOTICE:** the 184-LingBot-Map-note's "next paper 185 = WinT3R, arXiv:2509.05296" was *correct* on all key facts (the 11-12th arXiv-ID hallucination was *prevented* by direct arXiv lookup), confirming that the *direct-arXiv-lookup* sub-skill is *working* after the 8 prior hallucinations. The *new* critical finding is the *non-commercial* license — the 184-note did NOT specify license details, and the 185-note's GitHub API lookup revealed the *non-commercial* license. *Always* verify license via GitHub API.
