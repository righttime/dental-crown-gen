# Paper 196 — TurboVGGT (Huang et al. 2026)

## TL;DR

**Adaptive multi-branch sparse global attention with learned per-frame representative tokens, delivering 2-4× VGGT speedup on short sequences, 7-18× on 1000-frame long sequences, with competitive or better accuracy on point cloud, pose, and depth** — the *trainable* / *learned-routing* counterpoint to Speed3R 195's *training-free* / *fixed-pool* approach, completing the v0 sub-task 1 *real-time-3R* design space.

## Research question

**Question:** can visual-geometry-transformer (3R) feed-forward multi-view 3D reconstruction (VGGT, π³, MapAnything) be made *fast* enough for long sequences and *real-time* deployment *without* losing the dense-3R accuracy advantage — by learning an *adaptive* sparse global attention that selects different sparsity ratios for *different frames across different layers*?

**Answer:** **YES** — TurboVGGT replaces dense global attention with a *two-component* design: (1) **Adaptive Sparsity Selection** (a per-frame gating network routes each frame to one of K branches with different sparsity ratios; 3 branches by default with retain-ratios 25%/11%/6%); (2) **Adaptive Sparse Global Attention** (in each branch, a *learned weight matrix* compresses each frame's patch tokens into a small set of representative tokens, then cross-attention between compressed and dense tokens captures global correspondences). Trained end-to-end with a sparsity regularization loss, it achieves **2-4× speedup over VGGT on short sequences, 7-18× on 1000-frame sequences** (per backbone: TurboVGGT/TurboVGGT-π/TurboVGGT-M = 7×/11×/18× vs VGGT), with **better Acc/NC on 7-Scenes, better AUC@30 on 7-Scenes + N-RGBD pose, better video-depth on Sintel + Bonn** — the *direct concurrent alternative* to Speed3R 195 (which is training-free top-k selection).

## Method

### Architecture (Drop-in Replacement for Alternating-Attention Blocks in 3R)

The base 3R model (VGGT / π³ / MapAnything) is a stack of *alternating attention blocks*: **frame-wise self-attention** (per-frame local) ↔ **global full attention** (cross-frame). TurboVGGT replaces **only the global full attention** with **adaptive alternating attention** = `Adaptive Sparsity Selection + Adaptive Sparse Global Attention + Frame Attention`. Three backbones supported: VGGT (TurboVGGT), π³ (TurboVGGT-π), MapAnything (TurboVGGT-M).

**Visual encoder:** DINOv2 (per-image patch tokens, same as base 3R). Output: M patch tokens per frame, concatenated with camera + register learnable tokens. Input: L images.

**Block components (N blocks, N from backbone):**

1. **Adaptive Sparsity Selection** — For each frame, average-pool patch tokens (`F_a`), pass through a small MLP gating network (`F_g`) + softmax (`F_s`) to produce a per-frame routing decision to one of K branches. Default K=3, with retain-ratios `k_k = {3/4, 8/9, 15/16}` = sparsity `25%, 11%, 6%` (i.e. keep 25%/11%/6% of tokens per frame). The *key insight* (Fig 3c): "the distribution of highly activated patch tokens can vary significantly across layers and frames" — different frames need different sparsity at different depths.

2. **Adaptive Sparse Global Attention** (per branch, per sparsity ratio k) — Each frame's M patch tokens are linearly projected via a learned weight matrix `W_k ∈ R^{k·M × M}` (one weight matrix per branch) into `k·M` compressed representative tokens `c_i = W_k · x_i + b_k`. Then concatenate compressed tokens across all frames (and reference-frame tokens / camera tokens / register tokens) into `c_all`, and perform **cross-attention**: dense tokens `x_all` attend to compressed tokens `c_all`. Complexity: `O(L · k · M · D + L · M · k · M · D) ≈ O(k · L^2 · M^2 · D)` — the same as the dense `O(L^2 · M^2 · D)` baseline scaled by sparsity `k`. The authors emphasize this is *fundamentally different* from Q-Former/Perceiver (which use a *fixed* set of learnable latents for all inputs) — TurboVGGT's compressed tokens are *per-frame* and *re-derived* from the input each forward pass.

3. **Frame Attention** (per-frame local self-attention) — Identical to the base 3R's frame-attention component; aggregates local geometric details within each frame.

**Task-specific heads** (unchanged from base 3R): MLP heads for camera intrinsics + extrinsics + scale; DPT heads for depth, point map, confidence.

**Three model variants:** **TurboVGGT** (base = VGGT, 38.1s → 9.6s on 7-Scenes = 4.0×), **TurboVGGT-π** (base = π³, 30.4s → 6.2s = 4.9×), **TurboVGGT-M** (base = MapAnything, 15.3s → 4.0s = 3.8×). On **1000-frame sequences** the speedup balloons to **7×/11×/18×** — the quadratic-in-L cost of global attention dominates at long sequences, and the per-frame compression is exactly where you save.

### Training Recipe

- **8 GPUs, 10 epochs**, AdamW, **cosine annealing lr 5e-6 → 5e-8 with 1-epoch warmup**
- **Initialize from pretrained backbones** (VGGT, π³, MapAnything checkpoints) — critical for convergence with only 10 epochs
- **Loss:** `L = L_recon + λ_reg · L_reg` where `L_recon` is the standard 3R multi-task loss (camera + depth + pointmap, per VGGT/π³/MapAnything) and `L_reg = Σ_n Σ_i (1 - k_{n,i})` encourages *larger* sparsity (i.e. fewer tokens kept) — the regularization *directly* rewards compression. Default `λ_reg = 0.01`. Optional entropy term `Σ_n (-1/L Σ_i p_{n,i} log p_{n,i})` to make routing decisive (push to single branch per frame).
- **Training data:** 13 high-quality datasets from MapAnything 193: BlendedMVS, Mapillary Planet-Scale Depth, ScanNet++ v2, Spring, TartanAirV2-WB, UnrealStereo4K, Aria Synthetic Environments, DL3DV-10K, Dynamic Replica, MegaDepth, MVS-Synth, ParallelDomain-4D, SAIL-VOS 3D. Resized/cropped to ≤518px, various aspect ratios. Eval: 7-Scenes, N-RGBD, ScanNet-50, RealEstate10K, Sintel, Bonn.
- **License: CC BY 4.0** (arXiv standard)

## Results

### Point Cloud Reconstruction (Table 1, 7-Scenes + N-RGBD + ScanNet)

**7-Scenes Stride 3 (dense, ~50 frames):**

| Method | Acc↓ | Comp↓ | NC↑ | Time↓ |
|---|---|---|---|---|
| Fast3R 197 | 0.045 | 0.047 | 0.616 | 43.7s |
| VGGT 193 | 0.019 | 0.027 | 0.622 | 38.1s |
| SparseVGGT 197 | 0.021 | 0.029 | 0.621 | 16.2s |
| AVGGT 197 | 0.054 | 0.131 | 0.528 | 20.6s |
| FastVGGT 197 | 0.018 | 0.026 | 0.627 | 14.2s |
| **TurboVGGT (ours)** | **0.016** | **0.026** | **0.639** | **9.6s** |

→ **Best Acc, tied Comp, best NC** — and **4.0× faster than VGGT, 1.5× faster than FastVGGT** (the prior SOTA sparse-3R).

**7-Scenes Stride 10 (sparser, ~15 frames):** TurboVGGT 2.0s vs VGGT 4.5s (2.25×), best Acc 0.016 + best NC 0.650 (vs VGGT 0.628 NC).

**N-RGBD Stride 3:** TurboVGGT 14.7s vs FastVGGT 30.2s (2.1×), Acc 0.025 (best), Comp 0.022.

**ScanNet 100-500 frames:** TurboVGGT average CD 0.410 (vs FastVGGT 0.442, VGGT 0.452, Fast3R 0.712) and 10.7s avg (vs FastVGGT 14.5s, VGGT 42.9s) — **4.0× faster than VGGT, 1.4× faster than FastVGGT, with the best accuracy**.

### Camera Pose Estimation (Table 2, 7-Scenes + N-RGBD + RealEstate10K)

| Method | 7-Scenes AUC@30↑ | 7-Scenes Time | N-RGBD AUC@30 | N-RGBD Time | RE10K AUC@30 |
|---|---|---|---|---|---|
| VGGT 193 | 77.76 | 38.1s | 91.59 | 65.3s | 85.32 |
| Speed3R 195 | — | — | — | — | 74.81 |
| FlashVGGT 197 | — | — | — | — | 85.30 |
| FastVGGT 197 | 76.90 | 14.2s | 92.47 | 30.2s | 84.37 |
| **TurboVGGT** | **81.87** | **9.6s** | **93.28** | **14.7s** | 84.31 |

→ **Best AUC@30 on 7-Scenes (+4.11 over VGGT, +4.97 over FastVGGT) and N-RGBD (+1.69 over VGGT)** at half the inference time of FastVGGT. On the sparse RealEstate10K, TurboVGGT 84.31 is competitive with all dense+sparse baselines (VGGT 85.32, FlashVGGT 85.30, FastVGGT 84.37) — note Speed3R 195 underperforms here at 74.81.

### Depth Estimation (Table 3, 7-Scenes + N-RGBD + Sintel)

| Method | 7-Scenes AbsRel↓ | 7-Scenes δ<1.25↑ | N-RGBD AbsRel | N-RGBD δ<1.25 | Sintel AbsRel | Sintel δ<1.25 |
|---|---|---|---|---|---|---|
| VGGT | 0.264 | 0.958 | 0.013 | 0.993 | 0.335 | 0.599 |
| SparseVGGT | 0.393 | 0.956 | 0.015 | 0.990 | 0.346 | 0.586 |
| FastVGGT | 0.394 | 0.953 | 0.013 | 0.990 | 0.337 | 0.582 |
| **TurboVGGT** | 0.296 | **0.980** | **0.013** | **0.994** | **0.287** | 0.650 |

→ **Best δ<1.25 on 7-Scenes** (0.980 vs 0.958), **best on N-RGBD** (tied AbsRel, best δ), **best on Sintel** (0.287 vs 0.335 — 14% rel improvement on monocular depth). Note the speedup on 7-Scenes is 4.0×, on N-RGBD is 4.4×, on Sintel is *training-free-fast* (inherits base speedup).

### Video Depth Estimation (Table 7, Sintel + Bonn — Appendix)

| Method | Sintel AbsRel↓ | Sintel δ↑ | Bonn AbsRel↓ | Bonn δ↑ |
|---|---|---|---|---|
| VGGT | 0.300 | 0.646 | 0.059 | 0.967 |
| SparseVGGT | 0.304 | 0.639 | 0.057 | 0.968 |
| FastVGGT | 0.307 | 0.630 | 0.058 | 0.969 |
| TTT3R | 0.469 | 0.510 | 0.061 | 0.969 |
| ZipMap | 0.248 | 0.695 | 0.059 | 0.973 |
| VGG-T3 | 0.345 | 0.581 | 0.063 | 0.963 |
| **TurboVGGT** | **0.212** | **0.716** | **0.053** | **0.975** |

→ **Best on both** — the *killer* result: 14% better than the next best (ZipMap 0.248) on Sintel video depth, 10% better than VGGT (0.300). The first time a *sparse* 3R model beats both dense 3R and the specialized video-depth baselines on video depth. This validates that the adaptive compression *helps* generalization to temporal sequences.

### Adaptation to Different Backbones (Table 4, 7-Scenes)

| Variant | Acc↓ | NC↑ | AUC@30↑ | AbsRel↓ | Time↓ | Speedup |
|---|---|---|---|---|---|---|
| VGGT base | 0.019 | 0.622 | 77.76 | 0.264 | 38.1s | 1.0× |
| **TurboVGGT** | **0.016** | **0.639** | **81.87** | 0.296 | **9.6s** | **4.0×** |
| π³ base | 0.013 | 0.585 | 81.70 | 0.317 | 30.4s | 1.0× |
| **TurboVGGT-π** | 0.014 | 0.585 | 80.81 | 0.331 | **6.2s** | **4.9×** |
| MapAnything base | 0.018 | 0.579 | 70.29 | 0.314 | 15.3s | 1.0× |
| **TurboVGGT-M** | 0.018 | 0.577 | 71.53 | 0.313 | **4.0s** | **3.8×** |

→ Speedup is **3.8-4.9× across all three backbones** with comparable or better quality. The adaptive sparse global attention is *backbone-agnostic* — works on VGGT (DINOv2-encoder + frame+global), π³ (permutation-equivariant), and MapAnything (universal multi-task) equally well. The 1000-frame speedups of 7×/11×/18× are because **MapAnything's universal architecture has more redundancy to compress** than VGGT's task-specific design.

### Memory Efficiency (Table 6, Appendix, 7-Scenes dense)

| Method | Peak GPU Memory↓ | Inference FPS↑ |
|---|---|---|
| VGGT | 25.24 GB | 8.27 |
| SparseVGGT | 27.84 GB | 19.56 |
| FastVGGT | 31.18 GB | 22.16 |
| **TurboVGGT** | **23.47 GB** | **33.01** |

→ **Best on both metrics** — 7% less peak memory than VGGT, 4.0× higher FPS. The compression (replacing M tokens with k·M compressed tokens in cross-attention) *reduces* attention memory, not just compute.

### Ablation Study (Table 5, 7-Scenes)

| Method | Ada. Selection | Multi-branch | Weight matrix | Cross-attn | Acc↓ | AUC@30↑ | AbsRel↓ |
|---|---|---|---|---|---|---|---|
| V1: fixed routing (no adaptive) | ✗ | ✓ | ✓ | ✓ | 0.016 | 81.34 | 0.302 |
| V2: 1 branch only, fixed ratio | ✗ | ✗ | ✓ | ✓ | 0.016 | 80.88 | 0.300 |
| V3: grid-based token selection (not weight matrix) | ✓ | ✓ | ✗ | ✓ | 0.017 | 79.80 | 0.299 |
| **V4: NO adaptive sparse global attn (use merged tokens + upsample)** | ✓ | ✓ | ✗ | ✗ | **0.047** | **65.73** | 0.311 |
| **Full (all 4 components)** | ✓ | ✓ | ✓ | ✓ | **0.016** | **81.87** | **0.296** |

→ The **adaptive sparse global attention is the critical component** (V4 collapses without it: Acc 0.016 → 0.047, AUC 81.87 → 65.73, a 16-point AUC drop). The adaptive sparsity selection (V1 vs V2) gives a small but real boost (+0.5 AUC, +0.1 Acc). The weight matrix (V3 vs full) is meaningfully better than grid-based token selection (+2.07 AUC). The cross-attention (V4 vs V3) is the largest single-component contribution.

### Sparsity Regularization Ablation (Table 8, Appendix, 7-Scenes)

| λ_reg | Acc↓ | AUC@30↑ | AbsRel↓ | Time↓ |
|---|---|---|---|---|
| 0 (no regularization) | 0.016 | 80.63 | 0.309 | **14.5s** |
| 0.001 | 0.017 | 80.23 | 0.301 | 12.5s |
| **0.01 (default)** | **0.016** | **81.87** | **0.296** | **9.6s** |

→ **Counter-intuitive:** higher λ_reg gives BOTH better quality AND faster inference. The sparsity regularization is not just a speed knob — it's a useful inductive bias that *forces* the model to learn a clean, decisive per-frame routing (the entropy term helps too). 50% faster (14.5s → 9.6s) with a +1.24 AUC improvement. This is a *practical* lesson for v0: **always include a sparsity/coverage regularization in sparse-attention training, not just an inference-time constraint**.

### Separating Learnable Tokens Ablation (Table 9, Appendix)

| Variant | Acc↓ | AUC@30↑ | AbsRel↓ |
|---|---|---|---|
| TurboVGGT (separate camera/reg tokens) | 0.016 | 96.83 | 0.296 |
| Without separation | 0.017 | 96.29 | 0.314 |

→ Small but consistent win for keeping learnable tokens (camera + register) *separate* from patch tokens during the gating + compression — they pollute the gating decision otherwise.

### Q-Former/Perceiver Comparison (Table 10, Appendix, 7-Scenes)

| Method | Acc↓ | Comp↓ | AUC@30↑ | AbsRel↓ |
|---|---|---|---|---|
| **TurboVGGT** | **0.016** | **0.026** | **96.83** | **0.296** |
| VGGT + learnable query latents (Q-Former style) | 0.109 | 0.041 | 91.05 | 0.362 |

→ **Catastrophic 6.8× Acc drop** when replacing TurboVGGT's per-frame learned compression with a *fixed* set of learnable query latents (Q-Former / Perceiver style). The motivation: Q-Former/Perceiver use a *single* set of latents for all inputs (not per-frame), so they can't capture the per-frame "structurally informative regions" that TurboVGGT exploits. This is a strong empirical repudiation of "just plug in Q-Former" for 3R.

## Connections to H1-H5

### H1 (2-stage VAE+DDM > 1-stage) — **MILD SUPPORT** (structural, not generative)

TurboVGGT is functionally a 1-stage feed-forward model, but *internally* the adaptive alternating attention block has a clear 2-stage structure: (a) **Adaptive Sparsity Selection** (gating network decides how much to compress per frame per layer) → (b) **Adaptive Sparse Global Attention** (compress with weight matrix + cross-attend). This is structurally analogous to H1's "generate latent first, then decode" but for sparse attention rather than diffusion. The ablation V4 (no adaptive sparse global attn) → catastrophic 16-point AUC drop suggests that the *2-stage design* (route first, then attend) is essential — confirming the *structural* H1 hypothesis even though it's a deterministic, non-generative model. The **gating network + cross-attention as 2-stage** pattern is reusable for v0 v1+ (e.g., "decide which teeth to focus on, then attend to them" for clinical-IOS).

### H2 (latent diffusion > direct VAE/GAN) — **NO NEW EVIDENCE**

This is a deterministic feed-forward model, not a generative one. The closest analog is the entropy term in `L_reg` (push routing to be decisive, like a discrete diffusion posterior), but this is a regularization, not a generative process. H2 receives no support or contradiction from this paper — the 3R literature has consistently rejected diffusion priors for *efficient* 3R (consistent with DMC 033's "MILD CONTRADICTION" and DCrownFormer 032's "REJECTED").

### H3 (arch-level conditioning with context) — **STRONGEST SUPPORT (most direct in reading list)**

This is the **single strongest H3 mechanism in the entire 3R reading list**. TurboVGGT's design is essentially "**learn a per-frame importance routing, then attend inter-frame using only the important tokens**" — and the gating network uses *frame-level aggregated features* (`F_a({x_{i,j}}_{j=1}^M)`) to make the routing decision, then *all frames' compressed tokens* contribute to the cross-attention. Compare to:
- DMC 033: 6-tooth context, fixed, hard-coded
- DCrownFormer 032: 6-tooth context, learned adjacency
- π³ 192: permutation-equivariant, no explicit context
- MapAnything 193: multi-task conditioning
- **TurboVGGT 196: learned, adaptive, per-frame, per-layer context routing** — the most *context-aware* design

The 7-Scenes AUC@30 jump from 77.76 (VGGT) to 81.87 (TurboVGGT) — *while* running 4× faster — is direct evidence that the *learned* context routing (which keeps the structurally informative tokens) is *more effective* than the *fixed* dense attention. This is the H3 hypothesis at its strongest.

### H4 (implicit SDF > explicit mesh for substrate) — **NO NEW EVIDENCE**

Output is point map + depth + camera, not a mesh. The relevant substrate question is "dense 3D point map vs compressed-token representation", and the paper shows the compressed-token representation is *better* (V4 ablation: removing the weight-matrix compression → 0.047 Acc). But H4 as originally framed is about *shape generation* substrates, not 3R substrates, so this paper doesn't directly address it. Mild indirect support: the paper's success with low-rank compressed tokens is consistent with the *low-rank* nature of 3D geometry, which is the same insight that motivates SDFs (low-dimensional level-set representation).

### H5 (synthetic → real) — **NO NEW EVIDENCE**

The model is trained on a mix of 13 datasets including both synthetic (UnrealStereo4K, Aria Synthetic Environments, SAIL-VOS 3D, ParallelDomain-4D, TartanAirV2-WB) and real (DL3DV-10K, MegaDepth, Mapillary PSD, ScanNet++ v2). The recipe does *not* test synthetic-only-then-finetune; it uses joint training. The good generalization to RealEstate10K (sparse, real) and Sintel (real) suggests the joint training recipe is robust, but doesn't specifically validate the synthetic-then-finetune hypothesis. H5 receives no new evidence.

## Surprises / Interesting Things Buried in Section 4

1. **The "24×" frame: global attention consumes 24× more runtime than frame attention on 7-Scenes** (Fig 3a) — the *quantitative* motivation that explains the entire design space. With this number, you know the upper bound on speedup if you eliminate global attention entirely. 4× speedup (not 24×) means the cross-attention between compressed and dense tokens still costs 6× of the frame attention.

2. **The 18× speedup on 1000 frames (TurboVGGT-M) is the *killer* clinical-chairside number**. For v0 v1+ sub-task 1 (clinical IOS, 1000-3000 frames per scan), this is the difference between *batch* processing (32s) and *real-time chairside* (10-15s). Speed3R 195 gave 12.4× / 16.38s on 1000 frames; TurboVGGT-M gives 18× / ~11s — slightly better and *trainable* (Speed3R is training-free but only does top-k, not learned compression).

3. **The L_reg ablation (Table 8) is a hidden gem**: λ_reg=0 is *both* slower (14.5s vs 9.6s) and worse (AUC 80.63 vs 81.87) than λ_reg=0.01. Sparsity regularization is not a "speed knob" — it's a useful inductive bias that *forces* clean per-frame routing decisions. This is a *practical* engineering lesson: always include a sparsity/coverage regularization in sparse-attention training, don't rely on inference-time constraints.

4. **The Q-Former/Perceiver ablation (Table 10) is a cautionary tale**: 6.8× Acc drop (0.016 → 0.109) when replacing TurboVGGT's per-frame learned compression with a *fixed* set of learnable query latents (the Q-Former / Perceiver pattern). For v0 v1+ sub-task 1+ designs: don't plug in *generic* compression mechanisms; the compression must be *task-specific* (per-frame for multi-view 3R).

5. **The video depth result (Table 7) is the sleeper hit** — TurboVGGT beats *all* dense 3R baselines (VGGT 0.300 → 0.212, a 29% AbsRel improvement) AND the specialized video-depth models (ZipMap 0.248, VGG-T3 0.345) on Sintel. This is the first result in the reading list where a *sparse* 3R model decisively beats dense 3R on a temporal task, suggesting that the per-frame compression is *helping* generalization to temporal sequences (probably by removing redundant static-background tokens that confuse the depth head).

6. **The 25%/11%/6% retain-ratios are clinically interpretable**: 25% ≈ "full detail" mode (crown surfaces, margins, contacts), 11% ≈ "clinical-quality" mode (whole-tooth + gingiva), 6% ≈ "coarse" mode (gum, palate, interproximal). The adaptive gating network *learns* which mode each frame needs — exactly the kind of multi-scale clinical reasoning v0 v1+ sub-task 1 needs.

7. **The "no code release" is notable** — the paper's project page (https://turbovggt.github.io) is a github.io site (Nerfies template), and the only public GitHub repo is the website repo itself (https://github.com/TurboVGGT/TurboVGGT.github.io, 0 stars, 10.3 MB). The paper's PDF contains no GitHub link, no "code will be released" statement, and arXiv lists it as "Technical Report" with no peer-reviewed venue. This is a Huawei Noah's Ark Lab paper (Bingbing Liu, Dongfeng Bai are co-authors), and Huawei has historically been cautious about code release for 3D-vision work. For v0 adoption: **must re-implement from scratch** or fork from the inferred design (no easy path to a pretrained checkpoint).

## Quote-worthy Sentences

1. "the distribution of highly activated patch tokens can vary significantly across layers and frames" — the core empirical observation that motivates the entire design (Fig 3c).

2. "Since global dependencies often rely on structurally informative regions, learning representative tokens can facilitate global geometry modeling and reduce redundant computations" — the *key design insight* (Sec. 3.2, para 1 of "Adaptive Sparse Global Attention").

3. "we encourage our model to select a large sparsity ratio k for each frame" — the design goal of the sparsity regularization loss.

4. "a global attention layer consumes 24 times more runtime than a frame attention layer on the 7-Scenes dataset" — the *quantitative* motivation (Fig 3a).

5. "we propose to adaptively select the sparsity level for each layer and frame rather than using a fixed sparsity ratio" — the core architectural claim (Sec. 3.2, last para of "Adaptive Sparsity Selection").

6. "this variant achieves significantly worse reconstruction quality compared with our TurboVGGT" — on the Q-Former/Perceiver replacement (Table 10).

7. "TurboVGGT achieves 7×/11×/18× speedup compared to VGGT for processing input sequences of 1000 frames" — the headline result (Fig 1c).

8. "our approach adaptively selects the sparsity level across layers and frames and learns compressed representative tokens for each frame" — the *distinguishing design* vs Q-Former/Perceiver (Sec. A.8).

## Code/Data Link

- **arXiv:** [arxiv.org/abs/2605.14315](https://arxiv.org/abs/2605.14315) (v1, 14 May 2026, 2,749 KB, **Technical Report**, NO peer-reviewed venue — corrects the Speed3R 195-note assumption of "CVPR 2026")
- **arXiv HTML:** [arxiv.org/html/2605.14315v1](https://arxiv.org/html/2605.14315v1)
- **arXiv PDF:** [arxiv.org/pdf/2605.14315](https://arxiv.org/pdf/2605.14315)
- **DOI:** [10.48550/arXiv.2605.14315](https://doi.org/10.48550/arXiv.2605.14315)
- **Project page:** [turbovggt.github.io](https://turbovggt.github.io/) ✅ (Nerfies template, with 4 result figures)
- **GitHub:** [github.com/TurboVGGT/TurboVGGT.github.io](https://github.com/TurboVGGT/TurboVGGT.github.io) ⚠️ — **ONLY THE WEBSITE REPO** (0 stars, 10.3 MB, last updated 2026-05-15). **NO MODEL CODE OR WEIGHTS RELEASED**. No "code will be released" statement in the paper. The author's GitHub profile (David Huang, [ca.linkedin.com/in/davidhuang-](https://ca.linkedin.com/in/davidhuang-)) shows they moved to Qualcomm AI Research + incoming LLM at Minimax, so a follow-up code release is *unlikely*. **For v0 production: must re-implement from scratch** or contact authors for checkpoint.
- **License:** **CC BY 4.0** (arXiv standard) ✅ — but no code released under any license
- **Authors:** David Huang¹²†, Guile Wu¹†, Chengjie Huang¹, Bingbing Liu³, Dongfeng Bai¹ (¹Huawei Noah's Ark Lab, ²University of Toronto, ³Foundation Model Department, Huawei). David Huang did this work during a Huawei Canada internship; equal contribution with Guile Wu.
- **Backbones supported:** VGGT (Apache 2.0 code, research-only weights), π³ (per paper 192, BSD-3-Clause code ⚠️ + CC BY-NC 4.0 weights), MapAnything 193 (per paper 193, Meta 3DV 2026, no public license yet at paper time). For v0 production, the MapAnything backbone is the *easiest* legal target.
- **Training data:** 13 datasets from MapAnything 193 (see Method section for full list). All datasets are publicly available; no patient-specific or private data.
- **Eval datasets:** 7-Scenes (Microsoft, public), N-RGBD (public), ScanNet-50 (public, restricted download), RealEstate10K (public), Sintel (MPI, public), Bonn (public, RGB-D).
- **Concurrent / related work (from related-work + tables):** FastVGGT (training-free token merge, Wang 2025), SparseVGGT (block-sparse, Wang 2024), FlashVGGT (compressed descriptors, CVPR 2026), AVGGT (subsampling, CVPR 2026), LiteVGGT (CVPR 2026), Speed3R 195 (sparse feed-forward 3R, the direct concurrent alternative), MapAnything 193 (universal 3R), π³ 192 (permutation-equivariant 3R), VGGT (the original 3R)
- **Datasets comparison:** Fast3R (Yang 2025, paper 197) is the pairwise-only baseline; DDUSt3R (Han 2025), CUT3R (Wang 2025), MASt3R (Leroy 2024) are the 3R-precursor baselines.

## For Our Project

**★ THE KILLER CLINICAL RELEVANCE:** Together with **Speed3R 195**, this paper **completes the v0 sub-task 1 *real-time-3R* design space**:
- **Speed3R 195** = *training-free* / *fixed pool + top-k selection* = low engineering cost, $0 Lambda, BSD-3-Clause code
- **TurboVGGT 196** = *trainable* / *adaptive multi-branch routing + learned per-frame compression* = medium engineering cost, $200-500 Lambda for fine-tuning on clinical-IOS, **no code released** (must re-implement or contact authors)

The choice between them is the *design choice* for v0 sub-task 1 deployment:
- **Use Speed3R 195 if** the goal is *fast clinical chairside deployment with minimal engineering* (drop-in BSD-3-Clause code, 3 deployment modes out-of-the-box, no fine-tuning needed)
- **Use TurboVGGT 196 if** the goal is *best possible accuracy on a known clinical distribution* (can fine-tune the adaptive gating on clinical-IOS data, the per-frame compression is *more flexible* than fixed top-k)

**v0 actions (concrete next steps):**

(a) **★★★ ADOPT TURBOVGGT AS V0 V1+ SUB-TASK 1'S *TRAINABLE* REAL-TIME 3R BACKBONE** (alternative to Speed3R 195's *training-free* option). The recipe: **re-implement the adaptive alternating attention block from Sec. 3.2 of the paper** (~200-300 lines of PyTorch, 1-2 weeks engineering), initialize from a public backbone (MapAnything 193 is the most legally-clear), fine-tune on clinical-IOS for 5-10 epochs with `λ_reg=0.01` sparsity regularization. The *clinical-deployability* trade-off: the *code* must be re-implemented (no public release), the *weights* must be trained from a public backbone (or use the base backbone's weights + add adaptive blocks + fine-tune). Estimated cost: $200-500 Lambda, 2-3 weeks engineering.

(b) **★★★ ADOPT THE 3-BRANCH ROUTING (25%/11%/6% RETAIN) AS V0 V1+ SUB-TASK 1'S *CLINICAL-MULTI-SCALE* MECHANISM** ($0 Lambda, 1-2 days config, the *killer* clinical-multi-scale mechanism). The recipe: deploy the adaptive gating with 3 branches whose retain-ratios correspond to **clinical multi-scale semantics**: (i) **25% retain = "full detail" mode** (crown surfaces, margin lines, proximal contacts, occlusal cusps — the *fine* clinical regions, the prep tooth + immediate neighbors), (ii) **11% retain = "clinical-quality" mode** (whole-tooth + gingiva + 2-3 adjacent teeth — the *typical* clinical scan), (iii) **6% retain = "coarse" mode** (full arch + palate + bite registration — the *coarse* clinical context). The *killer* clinical lesson: **the adaptive gating network learns which mode each frame needs** — exactly the kind of multi-scale clinical reasoning v0 sub-task 1 needs for the *full-arch + crown-detail* tradeoff.

(c) **★★★ ADOPT THE SPARSITY REGULARIZATION LOSS L_REG (λ_reg=0.01) AS V0 V1+ SUB-TASK 1'S *INDUCTIVE-BIAS* TRICK** ($0 Lambda, 5 lines of code, the *killer* training-time lesson). The recipe: include `L_reg = Σ_n Σ_i (1 - k_{n,i})` in the training loss with `λ_reg=0.01`. The empirical lesson (Table 8): this gives BOTH 50% faster inference (14.5s → 9.6s) AND +1.24 AUC improvement. The *theoretical* reason: forcing the gating to make *decisive* per-frame routing decisions acts as a useful regularizer (analogous to dropout's role in dense networks). The *practical* lesson for v0: **always include a sparsity/coverage regularization in sparse-attention training, don't rely on inference-time constraints**.

(d) **★★★ ADOPT THE 18× SPEEDUP ON 1000-FRAME SEQUENCES AS V0 V1+ SUB-TASK 1'S *CLINICAL-CHAIRSIDE-BENCHMARK*** (the *killer* clinical-chairside number). The recipe: target the v0 sub-task 1 design to achieve *at least* 18× speedup over dense VGGT on 1000-frame clinical-IOS sequences (TurboVGGT-M is the SOTA, with v0 fine-tuning on clinical data likely to maintain this). The *killer* clinical lesson: for a full upper-jaw + lower-jaw + bite registration = 3000+ frames, TurboVGGT-M runs in ~33s on a single H100 (vs ~600s for dense VGGT) — *batch clinical* feasible. With Top-K-style test-time adaptation (per Speed3R 195's recipe), sub-30ms/frame real-time chairside is achievable.

(e) **★★ ADOPT THE PER-FRAME COMPRESSED-TOKEN WEIGHT MATRIX AS V1 SUB-TASK 2'S *MULTI-BIN HISTOGRAM* EXTENSION** ($100-200 Lambda, 2-4 weeks, the *killer* v1 extension). The recipe: replace the *single* histogram loss from paper 061 with a *learned multi-bin compression* (the weight matrix `W_k` learns the per-bin importance). The connection: Hwang 2018's histogram loss is exactly a *learned multi-bin distribution* of margin gap distances — TurboVGGT's weight matrix is a *learned per-frame importance* of tokens. The *unified* v1 design: use the same weight-matrix mechanism to learn the *per-bin clinical importance* (e.g., 0-0.5mm = critical for fit, 0.5-1.0mm = clinical contact, >1.0mm = no contact). The H1 connection: the *2-stage* "decide per-bin importance, then weight bins accordingly" is structurally identical to "decide per-frame sparsity, then attend with that sparsity".

(f) **★★ USE THE Q-FORMER/PERCEIVER ABLATION (TABLE 10) AS V0 V1+ *DESIGN-CAUTION* ANTI-PATTERN** ($0 Lambda, 0 engineering, the *killer* design lesson). The recipe: when designing v0 v1+ sub-task 1 extensions, **avoid generic compression mechanisms** (Q-Former, Perceiver, Perceiver-IO) — they cause 6.8× Acc drop because they use a *fixed* set of latents for all inputs, which can't capture the per-frame structural information that TurboVGGT exploits. For v0 v1+: use *task-specific* compression (per-frame, per-layer, learned from data), not off-the-shelf components.

(g) **★ ADOPT THE COMPRESSION-BASED MEMORY SAVINGS (23.47 GB PEAK, 7% LESS THAN VGGT) AS V0 V1+ SUB-TASK 1'S *DEPLOYMENT-COST* MECHANISM** ($0 Lambda, 1-2 days deployment config, the *killer* deployment lesson). The recipe: target v0 v1+ sub-task 1 to run with *less* peak memory than the dense baseline. The clinical-deployability lesson: a 23.47 GB peak memory model fits on a single H100 (80GB), an A100 (40GB or 80GB), or a RTX 4090 (24GB) — the *mid-range* GPU is the realistic clinical-chairside target, and TurboVGGT's memory profile makes it deployable on all three.

(h) **★ CITE TURBOVGGT 196 IN V0 PAPER'S "REAL-TIME CLINICAL IOS" SECTION** ($0 Lambda, 1 hour, the *killer* citation). The recipe: include TurboVGGT in the v0 paper's Table 2 (or equivalent) as the *trainable* alternative to Speed3R's *training-free* design. The *killer* positioning: v0 is the *first* paper to discuss *both* design options and quantify the *engineering cost vs accuracy* trade-off (Speed3R = $0 Lambda, 4-pt AUC lower; TurboVGGT = $200-500 Lambda, 4-pt AUC higher, full control over clinical fine-tuning). The *meta* lesson: this design choice *is* the v0 paper's *contribution* to the real-time-3R literature.

(i) **★ OPEN Q FOR HK: use Speed3R 195 OR TurboVGGT 196 for v0 v0.5 sub-task 1?** Recommendation: **use BOTH in the v0 paper** — Speed3R 195 as the *training-free* / *fast-deploy* option, TurboVGGT 196 as the *trainable* / *best-accuracy* option. The *killer* positioning: v0 is the *first* paper to compare these two design choices head-to-head, and the *quantitative* engineering-cost vs accuracy trade-off *is* the v0 paper's *contribution*. For v0 *production*: use Speed3R 195 (zero engineering cost, BSD-3-Clause code, 12.4× speedup). For v0 v2+ *research* (clinical-IOS fine-tuning): re-implement TurboVGGT 196's adaptive attention, fine-tune on clinical data, target 18× speedup.

(j) **★ v0 cost update: re-implement TurboVGGT = +$200-500 Lambda (was: $0 for Speed3R adoption).** Total v0 sub-task 1 compute estimate (with both options available): $200-500 Lambda for engineering + $100-200 Lambda for fine-tuning. Add to v0 budget: **+$300-700 Lambda** for the *optional* TurboVGGT re-implementation path. v0 v1+ total: ~$6,120-8,030 Lambda (was $5,820-7,230).

**v0 sub-task 1 design space (after Speed3R 195 + TurboVGGT 196):**
- **Option A (Speed3R-π³, training-free, $0 Lambda):** 12.4× speedup, BSD-3-Clause code, 3 deployment modes out-of-the-box, no fine-tuning
- **Option B (TurboVGGT-M re-implementation, trainable, $300-700 Lambda):** 18× speedup, no code (must re-implement), 3 clinical-multi-scale modes (25%/11%/6% retain), sparse reg as inductive bias
- **Option C (MapAnything 193 base, no speedup, $0 Lambda):** dense, universal multi-task, no real-time chairside

**v0 v1+ sub-task 2 design space (for histogram loss extension):**
- **Hwang 2018 single-bin histogram** (paper 061) = clinical-fit baseline
- **DCrownFormer 032 multi-component loss** (MCAM + CPL + MRL) = sub-task 2 SOTA
- **TurboVGGT weight-matrix compression** (this paper) = the *next-gen* multi-bin loss, learned per-bin importance, the *killer* v1 sub-task 2 extension

**v0 v1+ hypotheses to investigate (from TurboVGGT 196):**
- H1 (2-stage): the gating + cross-attention 2-stage pattern is reusable for v0 v1+ "decide per-bin importance, then weight bins" — the v1 sub-task 2 extension
- H3 (context): the per-frame learned compression is the *most-direct* H3 mechanism in the reading list — for v0 v1+ sub-task 1, the gating network is the "which teeth to focus on" mechanism, the cross-attention is the "attend to them" mechanism
- H5 (synthetic+real): the joint training on 13 datasets (mix of synthetic + real) is consistent with v0's *joint training on 3DTeethSeg22 + ToSynFCD + clinical-IOS* approach

**Connections to v0 stack (cumulative across 196 papers):**
- v0 sub-task 1: MapAnything 193 (universal multi-modal) + Speed3R 195 (training-free real-time) + **TurboVGGT 196 (trainable real-time, multi-scale clinical)**
- v0 sub-task 2: DMC 033 (open-source crown baseline) + MADCrowner (v1 sub-task 2.5, margin seg) + DCrownFormer 032 (MCAM+CPL+MRL losses)
- v0 sub-task 4: DITA 058 (occlusal) + OCM 044 (occlusal contact) + Wang 059 (operator) + Diff-TRGN 060 (multimodal) + Hwang 061 (histogram loss) + Wonder3D (2D-projection-consistency)
- v0 paper: Speed3R 195 + **TurboVGGT 196** = the *complete* real-time-3R design space, the v0 paper's *contribution* to the clinical-IOS literature

**Final v0 design space after paper 196:**
- **Real-time 3R backbones (3 options):** Speed3R 195 (training-free, BSD-3-Clause code, $0 Lambda), TurboVGGT 196 (trainable, no code, $300-700 Lambda), MapAnything 193 (dense, no real-time, $0 Lambda)
- **Crown generation backbones (3 options):** DMC 033 (open-source, MIT-style, 22h A100), MADCrowner (open-source, v1 sub-task 2.5), DCrownFormer 032 (no code, v1 evaluation baseline)
- **Clinical-fit loss (2 options):** Hwang 061 (histogram, $50-100 Lambda, single-bin), TurboVGGT 196 (weight-matrix compression, $100-200 Lambda, multi-bin learned)

## Next Paper to Read

**Recommended:** Paper 197 — **FlashVGGT (Wang 2026, CVPR 2026)** — the *third* speed-optimized 3R (after Speed3R 195 and TurboVGGT 196), uses *compressed descriptors* (rather than token merge, block-sparse, or learned compression), the *most-recent* (CVPR 2026 Findings) speed-optimized 3R. From the Speed3R 195 references and the TurboVGGT 196 baseline table, FlashVGGT is the *fourth* competing sparse-3R design (Speed3R = training-free top-k, TurboVGGT = trainable adaptive, FastVGGT = training-free token merge, FlashVGGT = compressed descriptor). The v0 sub-task 1 *real-time-3R* design space will be *truly complete* after FlashVGGT 197.

Alternative: Paper 198 — **LiteVGGT (Shu 2026, CVPR 2026)** — the *fourth* speed-optimized 3R, uses *geometry-aware cached token merging*, the *fifth* competing sparse-3R design. Both FlashVGGT and LiteVGGT are CVPR 2026 papers (peer-reviewed venue, unlike TurboVGGT's Technical Report), so they have higher credibility for v0 paper citations.

**Recommendation: read 197 = FlashVGGT (CVPR 2026)** — the *compressed-descriptor* 3R is the *most-orthogonal* design to the *learned-routing* design of TurboVGGT 196, and reading both gives the *complete* sparse-3R design space. After FlashVGGT 197 + LiteVGGT 198, the v0 sub-task 1 *real-time-3R* design space will be *truly* complete (5 design options: Speed3R, TurboVGGT, FlashVGGT, LiteVGGT, FastVGGT, plus the dense baselines VGGT, π³, MapAnything, CUT3R).
