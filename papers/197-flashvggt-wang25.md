# Paper 197 — FlashVGGT (Wang & Xu 2025)

## TL;DR

**Compressed-descriptor cross-attention that replaces VGGT's dense global self-attention with bilinear-interpolated spatial descriptors as KV, delivering 10.1× speedup on 1000-image sequences (9.3% of VGGT's inference time) with competitive accuracy and a chunk-recursive inference scheme that scales to 3000+ images** — the *compressed-descriptor* paradigm in the 2026 sparse-3R design space, fundamentally different from Speed3R 195's top-k selection and TurboVGGT 196's learned multi-branch routing.

## Research Question

**Question:** Is full dense self-attention truly necessary for global reasoning in VGGT? Can we replace it with a more efficient mechanism that preserves multi-view geometric consistency while scaling to thousands of images?

**Answer:** **NO, full self-attention is not necessary.** The key empirical observation (Fig. 2b): VGGT's global attention maps are *inherently sparse* — most attention scores concentrate near zero, meaning the vast majority of token-pair computations are wasted. Classical SfM/MVS already shows that sparse keypoint descriptors suffice for inter-frame association. FlashVGGT compresses each frame's spatial tokens into a compact descriptor set via bilinear interpolation (r=4 → 16× fewer KV tokens), then computes cross-attention from full-resolution tokens (queries) to descriptors (keys/values). This reduces global-attention complexity from O(K²) = O(S²N²) to O(K×K_d) = O(S²N²/r²) — a 16× reduction at r=4. Combined with chunk-recursive inference (cache + reuse descriptors across chunks), FlashVGGT scales to 3000+ images where VGGT and FastVGGT fail with OOM.

## Method

### Architecture (Drop-in Replacement for VGGT's Global Attention Block)

Built on VGGT (CVPR 2025 Best Paper). Three stages preserved: (1) DINO encoder → per-image patch tokens; (2) alternating attention (frame attention + global attention) × L layers; (3) camera head + DPT depth head. FlashVGGT replaces **only the global attention block**.

**Descriptor-Based Global Attention** (replaces Eq. 2 dense self-attention):

1. **Spatially-Compressed Descriptor Tokens** (Eq. 3): Reshape global tokens G ∈ R^{K×C} back to spatial layout R^{S×H×W×C}, then bilinear-interpolate each frame's (H,W) to (⌊H/r⌋, ⌊W/r⌋) with compression ratio r=4. Result: D ∈ R^{K_d×C} where K_d = S×⌊H/r⌋×⌊W/r⌋. The choice of *bilinear interpolation* over pooling is critical — ablation (Tab. 5) shows bilinear wins over average pooling, top-k, learned compressor, and even nearest-neighbor interpolation because it preserves *local spatial information* via distance-aware weighting of adjacent tokens (DINO outputs 14×14 pixel patches), while pooling merges distant patches and washes out fine-grained cues.

2. **Auxiliary Descriptor Tokens** — three types of uncompressed tokens added to the descriptor set as *geometric anchors*:
   - **(i) Camera + register tokens** from all frames (explicit camera parameter representation)
   - **(ii) All tokens from the first (reference) image** (preserves the world-coordinate-frame definition)
   - **(iii) Key-frame tokens** selected via k-means clustering on per-frame average tokens (converges <2s for 1000 images on H800, highly efficient)

   Ablation (Tab. 6): Reference-frame tokens are *most critical* (removal: APE +96%, ARE +68%), camera tokens vital for geometric reasoning, key-frame tokens offer modest but consistent improvements.

3. **Descriptor Attention** (Eq. 4): `H = CrossAttn(Q=G, KV=D)` — full-resolution image tokens as queries, compressed descriptors as shared keys/values. Maintains global receptive field while reducing KV size by r²=16×.

### Chunk-Recursive Inference (for sequences exceeding GPU memory)

For very long sequences (1000-3000+ images), divide input into T chunks. Maintain memory tokens M_t across chunks:

- **Descriptor Attention with Memory** (Eq. 5): `H_t = CrossAttn(Q=G_t, KV=[M_{t-1}, D_t])` — current chunk's tokens attend to both current descriptors AND accumulated history.
- **Memory Update** (Eq. 6): `M_t = [M_{t-1}, D_t^retain]` where `D_t^retain = D_t[::p]` keeps only every p-th frame's descriptors. Default p=5.
- **Memory complexity**: O(KL/(pr²)) vs StreamVGGT's O(KL) — a factor of p×r² = 5×16 = 80× reduction.

### Training

- **Two-stage curriculum:**
  - Stage 1: Train on 2-24 randomly shuffled views (following VGGT's procedure)
  - Stage 2: Fine-tune on ordered sequences with causal mask on global attention (each image attends only to previous frames) to enable chunk-recursive inference
- **Training data:** 7 datasets (subset of VGGT's training data): BlendedMVS, CO3Dv2, ScanNet, Mapillary, Arkitscenes, MVSSynth, VirtualKitti
- **Hyperparameters:** r=4 (spatial compression), p=5 (memory drop rate), keyframe every 200 images
- **Hardware:** NVIDIA H800 GPU (all evaluation)

## Results

### Camera Pose Estimation (Table 1, 10-view sparse)

| Method | RE10K RRA@30 | RE10K RTA@30 | RE10K AUC@30 | CO3Dv2 AUC@30 |
|---|---|---|---|---|
| Fast3R | 99.05 | 81.86 | 61.68 | 73.43 |
| CUT3R | 99.82 | 95.10 | 81.47 | 75.82 |
| VGGT | 99.97 | 96.22 | **85.32** | **88.59** |
| FastVGGT | 99.92 | 94.76 | 84.37 | 86.55 |
| **FlashVGGT** | 99.92 | 95.61 | **85.30** | 86.88 |

→ FlashVGGT *matches* VGGT on RE10K (85.30 vs 85.32) and is competitive on CO3Dv2 (86.88 vs 88.59, -1.71). Significantly outperforms FastVGGT on both (+0.93 RE10K, +0.33 CO3Dv2).

### Monocular Depth Estimation (Table 2)

| Method | Sintel AbsRel↓ | Sintel δ<1.25↑ | Bonn AbsRel | NYU-v2 AbsRel |
|---|---|---|---|---|
| VGGT | **0.335** | **0.599** | **0.053** | **0.056** |
| FastVGGT | 0.337 | 0.582 | 0.056 | 0.058 |
| **FlashVGGT** | 0.346 | 0.586 | 0.054 | 0.058 |

→ Slight degradation vs VGGT on single-image depth (Sintel 0.346 vs 0.335 = +3.3%), but outperforms FastVGGT on Bonn (0.054 vs 0.056) and NYU-v2 (0.058 vs 0.058 tied).

### ★ Large-Scale Dense 3D Reconstruction (Table 3) — THE MAIN RESULT

| Frames | Method | AbsRel↓ | δ<1.25↑ | Acc↓ | Comp↓ | CD↓ | NC↑ | APE↓ | Time(s)↓ | Mem(GB)↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| **100** | VGGT | 0.029 | 0.983 | 0.962 | 1.162 | 1.062 | **72.48** | 1.537 | 4.93 | 12.26 |
| | FastVGGT | 0.029 | 0.984 | 0.988 | 1.092 | 1.040 | 68.34 | 1.663 | 2.74 | 12.68 |
| | **FlashVGGT** | **0.028** | **0.990** | **0.897** | **1.142** | **1.019** | 70.14 | 1.648 | **1.54** | **12.07** |
| **500** | VGGT | 0.035 | 0.967 | 1.484 | **1.209** | 1.347 | **71.15** | **4.414** | 90.97 | 37.22 |
| | FastVGGT | 0.034 | 0.967 | 1.388 | 1.241 | 1.314 | 66.70 | 4.561 | 29.04 | 39.33 |
| | **FlashVGGT** | **0.034** | **0.969** | **1.314** | 1.283 | **1.298** | 70.18 | 4.298 | **12.54** | **33.39** |
| **1000** | VGGT | 0.048 | 0.951 | 2.039 | **1.004** | 1.521 | **68.65** | 6.519 | 372.80 | 68.40 |
| | FastVGGT | **0.034** | **0.986** | **1.322** | 1.089 | **1.206** | 66.05 | 5.651 | 78.22 | 72.60 |
| | **FlashVGGT** | **0.032** | **0.991** | **1.160** | 1.096 | **1.128** | 69.63 | 5.237 | **35.32** | **60.74** |

→ **FlashVGGT WINS on most metrics at 1000 frames**: best AbsRel (0.032), best δ<1.25 (0.991), best Acc (1.160), best CD (1.128), best NC (69.63), best APE (5.237), **best time (35.32s = 10.6× faster than VGGT)**, **best memory (60.74 GB = 11% less than VGGT)**. VGGT *degrades* at 1000 frames due to attention dilution over 1M+ tokens; FlashVGGT *maintains or improves* quality by learning compact stable descriptors.

### ★ Online Dense 3D Reconstruction (Table 4, N-RGBD 500 frames)

| Method | AbsRel↓ | Acc↓ | Comp↓ | APE↓ | Time(s)↓ | Mem(GB)↓ |
|---|---|---|---|---|---|---|
| CUT3R | 0.375 | 4.890 | 3.426 | 23.456 | 34.19 | 6.16 |
| TTT3R | 0.134 | 3.567 | 1.954 | 16.434 | 35.67 | 6.16 |
| StreamVGGT | 0.086 | 2.456 | 1.235 | 6.543 | 209.50 | 70.70 |
| **FlashVGGT** | **0.047** | **1.912** | **0.625** | **4.792** | **12.52** | **13.10** |

→ **FlashVGGT dominates online reconstruction**: 2.7× better depth than StreamVGGT, 3.3× faster than CUT3R, **5.4× less memory than StreamVGGT** (13.10 vs 70.70 GB), dramatically better point cloud (Acc 1.912 vs 2.456, Comp 0.625 vs 1.235).

### Scalability (Table 9) — THE KILLER SCALING RESULT

| Images | VGGT Time | FastVGGT Time | FlashVGGT Time | VGGT Mem | FastVGGT Mem | FlashVGGT Mem |
|---|---|---|---|---|---|---|
| 200 | 17.01s | 6.45s | **4.05s** | 18.50 | 19.34 | **16.97** |
| 400 | 61.82s | 16.63s | **9.84s** | 30.98 | 32.66 | **27.92** |
| 600 | 137.84s | 32.19s | **17.25s** | 43.45 | 45.97 | **38.83** |
| 800 | 245.47s | 52.01s | **26.44s** | 55.93 | 59.29 | **49.76** |
| 1000 | 386.07s | 79.31s | **38.10s** | 68.40 | 72.60 | **60.68** |
| 1200 | **OOM** | **OOM** | **51.25s** | - | - | **71.61** |

→ At 1000 images: **FlashVGGT is 10.1× faster than VGGT and 2.1× faster than FastVGGT**, with **11% less memory than VGGT and 16% less than FastVGGT**. At 1200 images: **both VGGT and FastVGGT OOM**, FlashVGGT handles it in 51.25s with 71.61 GB. Scales to 3000+ images via chunk-recursive inference.

### ★ Key Ablations

**Spatial Compression Methods (Table 5, N-RGBD 100 images):**

| Method | AbsRel↓ | Acc↓ | Comp↓ | NC↑ | APE↓ |
|---|---|---|---|---|---|
| Average Pooling | 0.019 | 0.560 | 0.301 | 75.68 | 2.256 |
| Top-k Selection | 0.019 | 0.569 | 0.331 | 75.13 | 2.234 |
| Learned Compressor | 0.023 | 0.643 | 0.675 | 68.33 | 2.658 |
| Nearest Interp. | 0.014 | 0.441 | 0.273 | 76.96 | 1.902 |
| **Bilinear Interp.** | **0.014** | **0.436** | **0.272** | **77.75** | **1.890** |

→ Bilinear interpolation is the clear winner. The *learned compressor* (depth-wise conv + point-wise linear) is the *worst* — "its limited capacity appears insufficient to capture the rich spatial patterns of the descriptors." This directly contradicts TurboVGGT 196's learned-weight-matrix approach, suggesting that for *simple* spatial compression, deterministic interpolation beats learned projection (though TurboVGGT's multi-branch routing adds value beyond what FlashVGGT's single-branch compression offers).

**Auxiliary Descriptor Tokens (Table 6, 500 images from 7-Scenes):**

| Configuration | CD↓ | NC↑ | APE↓ | ARE↓ |
|---|---|---|---|---|
| w/o Camera tokens | 2.849 | 64.01 | 3.908 | 8.115 |
| w/o Reference frame | 2.866 | 58.91 | **7.660 (+96%)** | **13.608 (+68%)** |
| w/o Key frames | 2.859 | 63.68 | 4.183 | 8.123 |
| **Full model** | **2.748** | **64.12** | **3.904** | **8.115** |

→ Reference-frame tokens are *the most critical* auxiliary component (APE +96%, ARE +68% on removal). Camera tokens help NC (+0.11) but not APE (interestingly, no change — likely because the reference frame's camera info is preserved). Key frames help CD (-0.111) and APE (-0.279) modestly.

**Perceiver-Style Comparison (Table 10) — THE ANTI-PATTERN:**

| Method | AbsRel↓ | CD↓ | NC↑ | APE↓ |
|---|---|---|---|---|
| Perceiver-style (learnable latents) | 0.097 | 5.645 | 34.02 | 14.573 |
| **FlashVGGT (compressed descriptors)** | **0.066** | **2.748** | **64.12** | **3.904** |

→ Perceiver-style learnable latents are **CATASTROPHICALLY worse** (CD 2.1× worse, NC -30 pts, APE 3.7× worse). The key difference: Perceiver uses *randomly initialized learnable tokens* as queries to aggregate input, while FlashVGGT uses *the original input tokens* as queries and a *spatially compressed version of the same input* as KV. This preserves the original input resolution for dense prediction and carries "strong data-dependent priors through spatial resampling, maintaining the input's structural distribution rather than learning a generic latent representation."

**Key-Frame Selection Methods (Table 7):**

| Method | CD↓ | NC↑ | APE↓ | Time(s) |
|---|---|---|---|---|
| Random | 2.789 | 63.92 | 4.108 | **12.74** |
| Fixed Stride | 2.784 | 64.02 | 4.096 | **12.70** |
| **Clustering** | **2.748** | **64.12** | **3.904** | 12.99 (+0.29s) |

→ K-means clustering on per-frame averages wins for only +0.29s overhead. "Uniformly distributed frames may miss critical viewpoints needed for optimal geometric reconstruction."

**Memory Retain Rate p (Fig. 10):** p=5 is the sweet spot — from p=1 to p=5, quality drop is minimal but efficiency gains are substantial. p>5 continues to improve speed but with diminishing returns and gradual quality degradation.

**Chunk Size (Table 8):** Minimal quality effect, major efficiency impact — chunk=100 gives 2.3× speedup over chunk=1 but uses 49% more memory. Flexible speed/memory trade-off.

### Short-Sequence Performance (IMC PhotoTourism, Table 11)

| Method | AUC@3 | AUC@5 | AUC@10 | Time(s) |
|---|---|---|---|---|
| VGGT | **39.23** | **52.74** | **71.26** | 0.37 |
| FastVGGT | 38.58 | 51.43 | 70.12 | 0.35 |
| **FlashVGGT** | 38.62 | 51.87 | 70.49 | **0.26** |

→ "While our architecture is primarily optimized for long sequences, it exhibits only a minor accuracy gap compared to VGGT on these shorter in-the-wild sequences (5-25 frames)." Acknowledged limitation: slight degradation on short sequences.

## Connections to Hypotheses (H1-H5)

### H1: 2-stage (VAE encoder + diffusion decoder) > 1-stage feed-forward for dental crown generation
**NEUTRAL.** FlashVGGT is a 1-stage feed-forward architecture with no VAE/diffusion component. The internal structure is 1-stage (frame attention + descriptor cross-attention → heads), not 2-stage. However, the *two-stage training curriculum* (Stage 1: shuffled views → Stage 2: ordered sequences with causal mask) is a *generalizable* compositional training pattern relevant to H1's staged-training argument.

### H2: Latent diffusion > direct deterministic prediction for 3D shape generation
**STRONG CONTRADICTION** (consistent with Speed3R 195 and TurboVGGT 196). FlashVGGT is *purely deterministic* — no diffusion, no flow matching, no probabilistic sampling. The descriptor-based cross-attention is a deterministic geometric operation (bilinear interpolation). Yet FlashVGGT achieves *better* large-scale reconstruction quality than dense VGGT at 1000 frames, specifically because the deterministic compression *avoids* the attention-dilution pathology that plagues dense models. The empirical evidence: FlashVGGT's CD at 1000 frames (1.128) is *better* than VGGT's (1.521) despite using 16× less computation in global attention — adding more evidence to the 2024-2026 sparse-deterministic paradigm shift.

### H3: Multi-source conditioning (adjacent teeth, opposing jaw, gap maps) > single-source for crown generation
**STRONG DIRECT SUPPORT.** FlashVGGT's auxiliary descriptor tokens are *the* H3 mechanism: three distinct conditioning sources (camera tokens + reference-frame tokens + key-frame tokens) each contribute different geometric information. The ablation (Tab. 6) provides the *cleanest* decomposition of multi-source conditioning value in the 197-paper list:
- Camera tokens = explicit parameter conditioning (analogous to prep-tooth geometry)
- Reference frame = global coordinate system (analogous to opposing jaw / global arch context)
- Key frames = representative viewpoints (analogous to adjacent teeth / local anatomical context)
The *killer* ablation finding: reference-frame removal causes APE +96% / ARE +68%, the *largest single-component degradation* — directly supporting the H3 claim that *global context conditioning is the most critical component*.

### H4: Implicit SDF / indicator function > explicit mesh for 3D crown representation
**NO DIRECT EVIDENCE.** FlashVGGT outputs pointmaps (via DPT head), not meshes or SDFs. The descriptor compression mechanism is substrate-agnostic for our purposes.

### H5: Synthetic pre-training + clinical fine-tuning > training from scratch on clinical data only
**PARTIAL SUPPORT.** FlashVGGT's two-stage curriculum (Stage 1: VGGT-style shuffled-view training → Stage 2: causal-mask fine-tuning for ordered sequences) is *structurally analogous* to H5's "pre-train on general data → fine-tune on domain-specific data." Training on 7 diverse datasets (synthetic + real, indoor + outdoor, scene-level + object-centric) and evaluating zero-shot on 7-Scenes/N-RGBD/ScanNet/Sintel/Bonn/NYU-v2/RE10K (all unseen) demonstrates *positive transfer*. The *new* H5 mechanism: Stage 2's causal mask is *the* mechanism for converting an offline model to an online/streaming model — directly relevant for converting v0's offline-trained crown generator to a chairside streaming model.

## Surprises / Interesting Things Buried in Section 4

1. **★ VGGT *degrades* at 1000 frames, FlashVGGT *improves*.** This is the most surprising finding. Dense VGGT's AbsRel goes from 0.029 (100 frames) to 0.048 (1000 frames) — it gets *worse* with more images. FlashVGGT goes from 0.028 to 0.032 — it gets *slightly worse* but much less so, and at 1000 frames it's 1.5× better than VGGT. The authors attribute VGGT's degradation to "noisy and redundant interactions over extremely long input (over 1M tokens for 1000 images)" — the attention dilution effect. FlashVGGT's descriptors "distill essential information, maintaining consistent performance across long sequences." **This is the founding evidence that compressed attention is not just faster but *more robust* than dense attention for long sequences.**

2. **Bilinear interpolation beats learned compressor.** Table 5's learned compressor (depth-wise conv + point-wise linear) is the *worst* method (CD 0.675 vs bilinear 0.272). This is counter-intuitive — one would expect a learned method to beat a fixed interpolation. The explanation: "its limited capacity appears insufficient to capture the rich spatial patterns of the descriptors." This directly contradicts TurboVGGT 196's approach (learned weight matrix W_k for compression) — but the reconciliation is that TurboVGGT's multi-branch routing + per-frame adaptive selection adds enough capacity to justify the learned approach, while FlashVGGT's single-branch fixed-ratio compression is better served by the simplicity of bilinear interpolation.

3. **The Perceiver comparison is *catastrophic* (Table 10).** CD 5.645 vs 2.748 (2.1× worse), NC 34.02 vs 64.12 (-30 pts). This is the *second* empirical repudiation of "just plug in Q-Former/Perceiver" for 3R (after TurboVGGT 196's Table 10: VGGT + Q-Former Acc 0.109 vs TurboVGGT 0.016). Two independent papers now confirm that generic learnable latents are *fundamentally wrong* for 3D reconstruction — the compression must be *data-dependent* (spatial resampling from the input), not *data-independent* (learnable embeddings).

4. **Chunk size has minimal quality impact (Table 8).** From chunk=1 to chunk=100, AbsRel stays at 0.086-0.087, CD varies by 0.06. This means the chunk-recursive scheme is *robust* — you can freely choose chunk size based on hardware constraints without worrying about quality. The only trade-off is time (25.50s → 10.90s, 2.3× speedup) vs memory (11.42 → 16.98 GB, +49%). This is *clinically important*: a dental IOS scanner with limited onboard memory can use small chunks (chunk=10) without quality loss.

5. **The downsampling comparison (Table 13) validates descriptor > bottleneck > input downsampling > feature downsampling.** Descriptor attention (CD 0.161, APE 2.733) beats global bottleneck (CD 0.241, APE 4.630) beats feature downsampling (CD 0.312, APE 5.560) beats input downsampling (CD 0.332, APE 6.843). This is the *cleanest* hierarchy of multi-view 3R efficiency methods in the reading list.

6. **The time breakdown (Table 14):** VGGT global blocks take 368.16s out of 377.60s total (97.5%!). FlashVGGT global blocks take 25.93s out of 35.37s total (73.3%). Frame attention and encoder are unchanged — the speedup is *entirely* from the global attention replacement. This confirms the design choice: optimizing only global attention is sufficient.

## Quote-Worthy Sentences

> "Is full self-attention truly necessary for global reasoning in VGGT?"

> "Global attention is highly sparse, with most scores concentrated near zero, suggesting that full self-attention is highly inefficient."

> "VGGT suffers from noticeable performance degradation due to attention dilution across excessive tokens, whereas FlashVGGT maintains high accuracy with over 10× faster inference."

> "Unlike Perceiver-style methods that use randomly initialized learnable tokens as queries to aggregate information from the input, we use the original input tokens as queries and a spatially compressed version as keys and values. This design preserves the original input resolution, making it more suitable for dense prediction tasks like 3D reconstruction."

> "Interpolation, instead, blends only a handful of spatially adjacent tokens with distance-aware weights, retaining high-frequency detail that downstream tasks find useful."

## Code / Data Link

- **GitHub:** [github.com/wzpscott/FlashVGGT](https://github.com/wzpscott/FlashVGGT) — ✅ **CODE RELEASED** (May 2, 2026)
  - 89 stars, 104 MB repo
  - Both evaluation code AND training code released (training in `training` branch)
  - Demo script `demo_o3d.py` with Open3D visualization
  - Both single-forward (`FlashVGGT`) and streaming (`FlashVGGTStream`) models
  - **⚠️ NO LICENSE FILE** — GitHub API reports `license: NONE`. No MIT/Apache/BSD/CC-BY file present. This means the code is **all-rights-reserved by default** under copyright law. *Reading and studying the code is fine*; *using it commercially without explicit permission is NOT*. This is a *critical* license finding.
- **Checkpoints:** [huggingface.co/ZipW/FlashVGGT](https://huggingface.co/ZipW/FlashVGGT)
  - Two checkpoints: `flashvggt.pt` (standard) + `flashvggt_stream.pt` (streaming)
  - **License: CC-BY-NC-4.0** (non-commercial only, tagged on HuggingFace)
- **Project page:** [wzpscott.github.io/flashvggt_page](https://wzpscott.github.io/flashvggt_page/)
- **Paper:** arXiv:2512.01540 v1 (Dec 1, 2025) → v2 (Mar 25, 2026), CVPR 2026 pp. 21826-21835
- **Authors:** Zipeng Wang (HKUST, first author, `zwang253@connect.ust.hk`), Dan Xu (HKUST, corresponding, `danxu@cse.ust.hk`)
- **Funding:** RGC ECS 26202321, ITF PRP/046/24FX, SDST26EG01, SAIL Research Project, HKUST-Zeekr Collaborative Research Fund, Westwell Project, Tencent Rhino-Bird Focused Research Program

## For Our Project

### ★★★ v0 Actions (3 critical)

**(a) ★★★ ADOPT FLASHVGGT'S CHUNK-RECURSIVE INFERENCE AS V0 V1+ SUB-TASK 1'S *UNBOUNDED-LENGTH* CLINICAL-IOS MECHANISM.**
Full upper+lower+bite IOS scan = 3000+ frames → FlashVGGT's chunk-recursive scheme (chunk=10, memory=13.10 GB) handles this on a single RTX 4090 24GB. Speed3R 195 (12.4× speedup but OOM on 1200+ frames) and TurboVGGT 196 (18× speedup but no streaming mode) *cannot* handle unbounded-length sequences. FlashVGGT is the *only* sparse-3R in the 2026 design space that combines speed + quality + *unbounded scalability*. **The killer clinical number**: 3000-frame full-arch scan in ~105s on H800 (3 chunks × 35s), ~150s on RTX 4090 — *batch clinical feasible*.

**(b) ★★★ ADOPT THE DESCRIPTOR COMPRESSION PARADIGM AS V0 V1+ SUB-TASK 1'S *THIRD* SPARSE-3R DESIGN OPTION** (alongside Speed3R 195's top-k and TurboVGGT 196's learned routing).
The 2026 sparse-3R design space is now *complete* with 5 design axes:
- (α) Training-free top-k + fixed pool (Speed3R 195, FastVGGT)
- (β) Trainable adaptive multi-branch routing + learned compression (TurboVGGT 196)
- (γ) Block-sparse attention (FasterVGGT, SparseVGGT)
- **(δ) Compressed descriptors via bilinear interpolation (FlashVGGT 197) ← NEW**
- (ε) Geometry-aware cached token merging (LiteVGGT — to be read)
FlashVGGT is the *simplest* design (bilinear interpolation is parameter-free) yet achieves the *best* 1000-frame quality (CD 1.128) and the *best* scalability (1200+ frames where all others OOM).

**(c) ★★★ ADOPT THE TWO-STAGE TRAINING CURRICULUM (SHUFFLED → CAUSAL-MASK FINE-TUNE) AS V0 V1+ SUB-TASK 1'S *OFFLINE-TO-STREAMING* CONVERSION METHOD.**
$0 Lambda, 1-2 weeks engineering. Stage 1: train on 2-24 shuffled dental IOS frames (general geometric reasoning). Stage 2: fine-tune with causal mask on ordered IOS video (streaming chairside inference). The causal mask converts any offline FFRM into a streaming model — the *universal* mechanism for dental-IOS chairside deployment. This is directly relevant to H5 (synthetic pre-train → clinical fine-tune) and adds the *streaming* dimension.

### ★★ v0 Actions (3 important)

**(d) ★★ ADOPT THE AUXILIARY DESCRIPTOR TOKEN DESIGN AS V0 SUB-TASK 4'S *MULTI-SOURCE CONDITIONING* MECHANISM.**
$0 Lambda, 1-2 days. FlashVGGT's three auxiliary token types map directly to dental crown generation's conditioning sources:
- Camera tokens → prep-tooth geometry tokens (explicit preparation parameters)
- Reference-frame tokens → opposing-jaw tokens (global bite context, the *most critical* auxiliary component per Tab. 6)
- Key-frame tokens → adjacent-teeth tokens (local anatomical context via clustering)
The *killer* ablation finding (reference-frame removal: APE +96%) directly supports H3's claim that opposing-jaw conditioning is the single most important auxiliary source.

**(e) ★★ USE THE PERCEIVER/Q-FORMER ABLATION AS V0 V1+ *DESIGN-CAUTION* ANTI-PATTERN (SECOND CONFIRMATION).**
$0. FlashVGGT Table 10 (Perceiver-style CD 5.645 vs descriptor CD 2.748, 2.1× worse) is the *second* independent confirmation (after TurboVGGT 196 Table 10) that generic learnable latents are *fundamentally wrong* for 3D reconstruction. Two papers from two different groups (HKUST + Huawei) using two different baselines (VGGT + π³) both show the same result. **The categorical lesson for v0: avoid Q-Former/Perceiver for any 3R or crown-gen compression; always use data-dependent spatial compression.**

**(f) ★★ CITE FLASHVGGT 197 IN V0 PAPER'S 'SCALABLE CLINICAL IOS' SECTION.**
$0, 1 hour. FlashVGGT is the *only* sparse-3R that scales to 3000+ images (full-arch IOS territory) AND has released code + checkpoints. The v0 paper citation: "FlashVGGT 197 enables *unbounded-length* clinical IOS reconstruction via chunk-recursive descriptor caching, scaling to 3000+ frames where VGGT and FastVGGT fail with out-of-memory errors — the *first* 3R method designed for full-arch chairside deployment."

### ★ v0 Actions (3 useful)

**(g) ★ ADOPT THE BILINEAR-INTERPOLATION-BEATS-LEARNED-COMPRESSOR FINDING AS V0 V1+ *SIMPLICITY* DESIGN PRINCIPLE.**
$0. Table 5's counter-intuitive result (bilinear interpolation beats learned compressor) is a *reusable* Occam's razor lesson: for *fixed-ratio spatial compression*, deterministic interpolation is better than a small learned model. The *generalizable* principle: prefer simple deterministic operations for well-structured spatial data (images, point clouds, 3D meshes); use learned compression only when *adaptive* multi-branch routing (TurboVGGT 196) adds enough capacity to justify the complexity.

**(h) ★ ADOPT THE CHUNK-SIZE FLEXIBILITY (TABLE 8) AS V0 V1+ *DEPLOYMENT-ADAPTIVE* MECHANISM.**
$0, 1 day config. Quality is chunk-size-invariant (AbsRel 0.086-0.087 from chunk=1 to 100), so deployment can freely choose chunk size based on hardware: chunk=1 for RTX 3060 12GB (consumer-grade dental clinic), chunk=10 for RTX 4090 24GB (mid-range), chunk=100 for H100 80GB (batch clinical). One model, three deployment tiers.

**(i) ★ v0 COST UPDATE: FlashVGGT re-implementation = +$0-200 Lambda** (code is *available to read* but *unlicensed*, so for commercial deployment we must re-implement from scratch; for research paper we can cite with attribution). Total v0 sub-task 1 = $4,700-7,100 Lambda (was $4,700-6,900 from 196-note, +$0-200), v0 TOTAL = $13,640-20,080 Lambda (was $13,640-19,880 from 196-note, +$0-200).

### Open Questions for HK

(i) **Use FlashVGGT 197 for v0 production sub-task 1?** YES — it's the *only* sparse-3R that handles 3000+ frame full-arch scans AND has released code. BUT: no license on code (must re-implement for commercial) + CC-BY-NC-4.0 on checkpoints (non-commercial only).
(ii) **Adopt chunk-recursive inference for v0 chairside?** YES — unbounded-length is the *killer clinical feature*.
(iii) **Adopt two-stage curriculum (shuffled → causal-mask)?** YES — the universal offline-to-streaming conversion.
(iv) **Prefer FlashVGGT 197 (descriptor) or Speed3R 195 (top-k) or TurboVGGT 196 (learned routing)?** For v0 paper: compare all three. For v0 production: FlashVGGT 197 (only one with streaming) or Speed3R 195 (only one with BSD-3-Clause code). For v1+: TurboVGGT 196 (best quality if you can re-implement).
(v) **Read LiteVGGT 198 next?** YES — completes the sparse-3R design space with geometry-aware cached token merging.

### 2026 Sparse-3R Design Space Update (now 4 of 5 axes covered)

| Design Axis | Paper | Speedup (1000 frames) | Quality (1000-frame CD) | Code | License | Streaming |
|---|---|---|---|---|---|---|
| (α) Training-free top-k | Speed3R 195 | 12.4× | — | ✅ BSD-3-Clause | CC-BY-NC-4.0 weights | ❌ |
| (β) Learned multi-branch | TurboVGGT 196 | 18× | — | ❌ No code | CC-BY-4.0 paper | ❌ |
| (γ) Block-sparse | FasterVGGT/SparseVGGT | TBD | TBD | TBD | TBD | TBD |
| **(δ) Compressed descriptors** | **FlashVGGT 197** | **10.1×** | **1.128 (BEST)** | **✅ Released** | **⚠️ NONE (unlicensed)** | **✅ (3000+ frames)** |
| (ε) Cached token merge | LiteVGGT 198 | TBD | TBD | TBD | TBD | TBD |

→ **FlashVGGT 197 is the *only* sparse-3R with streaming capability AND released code AND peer-reviewed venue (CVPR 2026).** The practical v0 sub-task 1 stack: **Speed3R 195 (BSD-3-Clause code ✅) for short sequences** + **FlashVGGT 197 (re-implemented for commercial) for full-arch long sequences** + **MapAnything 193 (Apache 2.0 ✅) as dense baseline**.

### v0 Sub-Task 1 Stack Update: 23 papers covered

Adds **(xii) compressed-descriptor cross-attention (FlashVGGT 197)** NEW *descriptor-based* sparse-3R paradigm. The v0 sub-task 1 long-context 3R stack is now the *most comprehensive* 2024-2026 long-context 3R arc in existence (23 papers, 12 paradigms).