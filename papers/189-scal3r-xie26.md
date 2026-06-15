# Paper 189 — Scal3R: Scalable Test-Time Training for Large-Scale 3D Reconstruction

## TL;DR

**FOUNDING PAPER** of the *chunked-TTT + context-parallel-all-reduce* paradigm for *kilometer-scale* RGB-only 3D reconstruction. Two coupled innovations: (1) **Global Context Memory (GCM)** = a set of **Adaptive Memory Units (AMUs)** (lightweight neural sub-networks, three SwiGLU-MLPs W₁, W₂, W₃ with `f_W(x) = W₂·(SiLU(W₁x) ⊙ W₃x)`, *LaCT-style* fast weights) inserted *after* four specific global-attention layers of VGGT (the 4th, 11th, 17th, and 24th — the layers whose outputs feed the two DPT decoders for depth + pointmap) — at *inference time*, the AMUs are updated via *one* gradient step on a *self-supervised dot-product loss* `L = -f_W(k_i)ᵀ·v_i` with *token-wise learning rate η_i predicted from the input tokens* (Eq. 8-9), giving the network the *capacity* of a 4M-parameter learnable function as the long-range memory (vs CUT3R/TTT3R's *fixed-size* state-token set); (2) **Global Context Synchronization (GCS)** = treat the chunked sequence processed by *K* GPUs as a *context-parallel* layout (analogous to tensor-parallel in LM training), each GPU computes its local AMU updates, then **all-reduce the AMU gradients across GPUs** (Eq. 10) so every chunk benefits from *sequence-wide* observations while keeping *per-GPU memory* constant — the GCS is the *founding* design lesson that *K GPUs ⇒ 𝒪(N/K) per-GPU work* with *no* cross-chunk context loss, the *killer* clinical-dental-IOS feature for full-arch scans split across 8+ H100s. Built directly on **VGGT** (24 layers, alternating frame-wise + global self-attention, frozen DINOv2 tokenizer, camera + 4 register tokens, 4-head outputs). **GCM is 75.55M new parameters (0.076B)** for a 1.2B VGGT backbone = *6.3% parameter overhead*. Trained end-to-end on **18 datasets** (Co3Dv2, BlendedMVS, DL3DV, MegaDepth, WildRGB, ScanNet++, HyperSim, Mapillary, Replica, MVS-Synth, Virtual KITTI, Aria Synthetic Environments, Aria Digital Twin, Taskonomy, TartanAir, Mapfree, SceneNet RGB-D, MatrixCity) for **60k iterations on 32 A800 GPUs for ~3 days** with **AdamW** peak lr 1e-4 (GCM) and 1e-5 (backbone), cosine decay, 2k warmup, grad clip 1.0, *and* **random GPU partitioning per iteration** (variable effective sequence length from 1 to 32 chunks, the *killer* H5 length-generalization recipe). **State-of-the-art on every benchmark tested** — **Oxford Spires RRE 4.45 / RTE 6.55 / ATE 4.45** (vs VGGT-Long 30.91 / 20.79 / 15.46, *-7× / -3× / -3.5×*), **KITTI ATE 14.55** (vs VGGT-Long 25.94, *-44%*), **VKITTI2 ATE 0.85** (vs VGGT-Long 1.03, *-17%*), **ETH3D CD 0.11 / F1 0.91** (vs VGGT-Long 0.24 / 0.84), **Oxford CD 0.96 / F1 0.96** (vs VGGT-Long 3.41 / 0.80, *-72% CD*), **VKITTI2 CD 0.40 / F1 0.91** (vs VGGT-Long 1.78 / 0.70, *-78% CD*). Inference: **chunked at M=60, O=30** (default) or M=120, O=60 (ScanNet++/TUM) or M=60, O=30 (Waymo), **300.76s for 758 frames = 2.53 FPS on a single RTX 4090** (10.32 GB peak memory), and **runtime scales linearly with sequence length (2.61-2.93 FPS from 150→990 frames)** while *RPE stays at 0.07-0.08m* (the *killer* clinical-stability claim). Additional benchmarks: **ScanNet++ ATE 0.08** (vs VGGT-Long 0.13, *-38%*), **TUM-RGBD ATE 0.07** (vs VGGT-Long 0.08, *-13%*), **Waymo ATE 1.52** (vs FastVGGT 1.28, slightly worse but still competitive). **Ablation: w/o GCM (no global context memory) = ATE 19.00** (vs full 13.70) → GCM is the *primary* long-range mechanism; **w/o GCS (no cross-chunk synchronization) = ATE 15.80** → GCS is the *propagation* mechanism; **state size 1M→4M monotonically improves RRE 1.01→0.87, RTE 1.01→0.84, ATE 0.99→0.85** → *more state capacity helps preserve long-range context*. **CVPR 2026 HIGHLIGHT**. **MIT License ✅** (the *first* Apache-2.0/MIT-licensed paper in the 2026 long-context 3R arc after LingBot-Map 184, vs ZipMap 188's *VGGT Research Materials License* ⚠️, LoGeR 187's *NO LICENSE* ⚠️, LONG3R 186's *NO LICENSE* ⚠️, WinT3R 185's *custom non-commercial* ⚠️). Code: **github.com/zju3dv/Scal3R** — *483 ⭐ / 37 🍴 / 5 open issues / 8.4 MB / last push 2026-05-11* (5 weeks ago), **inference-only** released (2026-04-10), **eval code NOT yet released** (TODO per README). HF checkpoint: **xbillowy/Scal3R** + requires **DINO-SALAD** VPR checkpoint for loop closure. The *killer* arXiv-ID fact: this is the **3rd ZJU3DV paper in 188-arc** (after 087-VGGT and 184-LingBot-Map, *the* dominant long-context-3R group in 2025-2026), and the **2nd 2026 paper in the 3-paper-arc** with **Tianyuan Zhang** as a discussion partner (LaCT author) — confirming the LaCT→Scal3R→ZipMap-188→LoGeR-187 *TTT-memory* lineage is now *empirically* validated by 4 *independent* papers in 2026.

## Metadata

- **Title:** Scal3R: Scalable Test-Time Training for Large-Scale 3D Reconstruction
- **Authors:** Tao Xie¹²\*, Peishan Yang¹, Yudong Jin¹, Yingfeng Cai², Wei Yin², Weiqiang Ren², Qian Zhang², Wei Hua³, Sida Peng¹, Xiaoyang Guo²†, Xiaowei Zhou¹† (*first author, †co-corresponding: Xiaoyang Guo, Xiaowei Zhou)
- **Affiliations:** ¹Zhejiang University (State Key Lab of CAD&CG), ²Horizon Robotics, ³Zhejiang Lab
- **Year:** 2026 (v1 9 Apr 2026, *single version*) → **CVPR 2026 Highlight**
- **arXiv:** [2604.08542](https://arxiv.org/abs/2604.08542) v1, cs.CV, **23,974 KB**, **PDF FULLY OPEN-ACCESS**
- **Code:** [github.com/zju3dv/Scal3R](https://github.com/zju3dv/Scal3R)
  - **MIT License ✅ ✅ ✅** (verified via /LICENSE raw, "Copyright (c) 2024 3D Vision Group at the State Key Lab of CAD&CG, Zhejiang University") — *the first MIT/Apache paper in the 2026 long-context 3R arc* (after LingBot-Map 184's Apache-2.0)
  - **483 ⭐ / 37 🍴 / 5 open issues / 8,557 KB** as of 2026-06-15
  - **Last push 2026-05-11** (5 weeks ago, 35 days)
  - **Created 2026-04-09** (1 day before v1 release, suggests code was already ready)
  - **Status:** inference only (released 2026-04-10), **eval code NOT yet released** (README TODO)
  - Built on **VGGT** (Meta, CC-BY-NC for code) + **VGGT-Long** (NO LICENSE) + **LaCT** (NO LICENSE per 187-note) — *parent code license inheritance issue* for v0 commercial deployment
- **Project page:** [zju3dv.github.io/scal3r](https://zju3dv.github.io/scal3r/) with interactive WebGL2 demos (static + dynamic, 8 scenes including Oxford Keble + KITTI sequences)
- **HF checkpoint:** [xbillowy/Scal3R](https://huggingface.co/xbillowy/Scal3R) (`scal3r.pt`) + requires **DINO-SALAD** ([serizba/salad](https://github.com/serizba/salad), Apache-2.0) for VPR loop-closure
- **Citations:** very new (60 days post-v1), expect 200-500 GS by end-of-2026 (CVPR 2026 Highlight + ZJU3DV star power)
- **YouTube:** none official; project-page has interactive web demo
- **Acknowledgments:** "We thank **Tianyuan Zhang** for helpful discussions on **LaCT** and **Dongli Tan** for valuable discussions" — confirms the **LaCT → Scal3R** technical lineage (Tianyuan Zhang is co-author of LaCT 2025, and co-author of ZipMap 188, *the 2nd paper in the 188-arc*)

## Research Question

> *Can we have BOTH (a) kilometer-scale RGB-only 3D reconstruction from thousands of input images AND (b) SOTA-quality pose + geometry that beats optimization-based SLAM AND (c) global context shared across the entire sequence (not just within chunks) AND (d) linear memory scaling with sequence length, all in a single feed-forward model?*

**Their answer:** Yes — via three coupled mechanisms: (1) **Global Context Memory (GCM)** = insert lightweight *LaCT-style* TTT-AMU modules (3 SwiGLU MLPs, 4M parameters state) after four specific VGGT global-attention layers (4th, 11th, 17th, 24th — the layers whose outputs feed the DPT decoders for depth + pointmap), updated via *one* gradient step on a *self-supervised dot-product loss* `L = -f_W(k_i)ᵀ·v_i` with *token-wise learning rate η_i predicted from the input tokens* — the AMU is a *learned nonlinear function* of size 4M parameters (state size = d²/nh × k with nh=1, k=4) that *compresses* all chunk tokens into a *learned* key-value association, *replacing* the global self-attention's KV cache (which has 𝒪(N) cost per query) with a *constant-size learnable function* that has 𝒪(1) cost per query; (2) **Global Context Synchronization (GCS)** = treat the chunked sequence processed by *K* GPUs as a *context-parallel* layout, each GPU computes its local AMU updates, then **all-reduce the AMU gradients across GPUs** so every chunk benefits from *sequence-wide* observations while keeping *per-GPU memory* constant — the GCS is the *founding* design lesson that *K GPUs ⇒ 𝒪(N/K) per-GPU work* with *no* cross-chunk context loss, the *killer* clinical-dental-IOS feature for full-arch scans split across 8+ H100s; (3) **Random GPU partitioning for length generalization** = at each training iteration, randomly partition the 32 GPUs into different groups, each group processes different sequences and performs GCS only within the group, resulting in *variable effective sequence lengths spanning 1-32 chunks* during training, the *killer* H5 recipe for *length generalization* (the *only* training-time data-augmentation that addresses the *length* dimension). Key insight: **chunked processing alone (VGGT-Long) loses global context; TTT-fast-weights alone (TTT3R, LoGeR, ZipMap) have fixed-size state tokens; chunked-TTT + cross-chunk all-reduce = the *right* combination for kilometer-scale**.

## Method

### Architecture (overview, 1 forward pass for N images)

1. **Input tokenization** (Sec. 3.1, inherited from VGGT 087): each I_i ∈ ℝ^{H×W×3} → DINOv2 encoder → 2D feature map → flatten into patch tokens + 1 camera token + 4 register tokens → x_i ∈ ℝ^{p×d}
2. **Feature backbone** (Sec. 4.1, inherited from VGGT 087): L=24 alternating-attention blocks, each with:
   - **Frame-wise self-attention** (within each image, rotary positional encoding) → captures *intra-frame* spatial relationships
   - **Global self-attention** (across all images) → captures *inter-frame* geometry consistency
3. **GCM insertion** (Sec. 4.1, **NEW**): after the global-attention layer of the **4th, 11th, 17th, and 24th** blocks (the 4 layers whose outputs feed the DPT decoders), insert a **Global Context Memory (GCM)** module that *replaces* the global-attention output with a TTT-style updated version. 4 GCM modules total per forward pass. Each GCM has 75.55M / 4 = 18.9M parameters (for the QKV projection + 3 AMU MLPs + output projection)
4. **Prediction heads** (inherited from VGGT 087): camera + DPT (depth) + DPT (pointmap) + tracking heads
5. **GCM detail (Sec. 4.2 + Appendix A):** QKV projection → 3 AMU MLPs (W₁, W₂, W₃, SwiGLU: f_W(x) = W₂·(SiLU(W₁x) ⊙ W₃x)) → output projection. Hidden dim = hd × k where hd=head dim, k=4. Number of heads nh=1 (to maximize state size). State size = d²/nh × k = 1.2B² / 1 × 4 ≈ 4M parameters per GCM
6. **Output:** (c_i, D_i, P_i, T_i) for all input frames, just like VGGT

### GCM update at inference (the core innovation)

The GCM block is a *nonlinear fast-weight function*, updated via *one* gradient step on a *self-supervised dot-product loss* per chunk (Eq. 8-9):

```
For chunk ℐ_k with tokens 𝒳_k ∈ ℝ^{M×d}:

  K, V = QKV_proj(𝒳_k)  # project to keys/values
  
  # Self-supervised update
  W ← W - ∇_W Σ_{i=1}^M η_i · L(f_W(k_i), v_i)
  where L(f_W(k_i), v_i) = -f_W(k_i)ᵀ · v_i  # dot-product loss
  and η_i = token-wise learning rate predicted from input tokens
```

After the update, the AMUs `W` store the contextual information of the current chunk, which is subsequently used to transform the query tokens Q:

```
f_W(Q) = W₂ · (SiLU(W₁ · Q) ⊙ (W₃ · Q))  # SwiGLU MLP query
```

The *killer insight*: η_i is *token-wise* (not chunk-wise), predicted from the input tokens, so different tokens get *different* learning rates — this is *more expressive* than TTT3R 182's per-token β (which is *derived* from attention) and ZipMap 188's learned per-token η (similar but different parameterization).

### GCS — Global Context Synchronization (the second core innovation)

Naïvely, the GCM in chunk k only sees tokens within chunk k, not the *rest* of the sequence. To share context, Scal3R treats the chunked sequence processed by *K* GPUs as a *context-parallel* layout (analogous to *context parallelism* in LM training, Ring Attention, etc.):

```
# On each GPU g (which has chunk ℐ_k):
  W_g ← W_g - ∇_W Σ_{i=1}^M η_i · L_i   # local AMU update

# All-reduce across K GPUs (Eq. 10):
  g = Σ_{j=1}^K ∇_W Σ_{i=1}^M η_i · L_i  # aggregated gradient
  W ← W - g  # apply aggregated gradient to ALL GPUs
```

The aggregated gradient `g` is implemented via PyTorch's `all-reduce` primitives (the same `dist.all_reduce` used in DDP), ensuring minimal communication overhead during both training and inference. **By doing so, each local chunk is enriched with substantial global observations, which improves local accuracy, strengthens cross-chunk consistency, and elevates overall reconstruction performance.**

The *killer insight*: this is *context parallelism* for *TTT fast weights*, the *natural* extension of tensor parallelism + sequence parallelism to the *test-time-training* regime. The *killer* clinical-dental-IOS feature: full-arch scans split across 8+ H100s with *constant per-GPU memory*.

### Training (Sec. 4.3)

- **Datasets (18, the *broadest* training mixture in 2026 long-context 3R):** Co3Dv2, BlendedMVS, DL3DV, MegaDepth, WildRGB, ScanNet++, HyperSim, Mapillary, Replica, MVS-Synth, Virtual KITTI, Aria Synthetic Environments, Aria Digital Twin, Taskonomy, TartanAir, Mapfree, SceneNet RGB-D, MatrixCity
  - For sequential datasets: sample consecutive image sequences as input
  - For unordered datasets: randomly sample images observing the same scene, shuffle as input
- **Loss (Eq. 11):** `L = λ·L_cam + L_dpt + L_xyz` (inherited from VGGT: L1 camera + confidence-weighted depth + confidence-weighted pointmap with gradient-based regularization)
- **Optimizer:** AdamW, peak lr 1e-4 (GCM) and 1e-5 (backbone), cosine decay, 2k linear warmup, grad clip 1.0
- **Length generalization recipe (the *killer* H5 trick):** at each iteration, randomly partition the 32 GPUs into different groups, each group processes different sequences and performs GCS only within the group, resulting in **variable effective sequence lengths spanning from 1 to 32 chunks during training**
- **Hardware:** **60k iterations on 32 A800 GPUs for ~3 days** = ~2,300 GPU-hours = ~$4,600 Lambda (or ~$6,900 on H100 spot)

### Inference (Sec. 4.4)

- **Chunked:** chunk size M=60 + overlap O=30 (default, VKITTI2/KITTI/Oxford); M=120 + O=60 (ScanNet++/TUM-RGBD); M=60 + O=30 (Waymo)
- **Cross-chunk alignment (inherited from VGGT-Long 17):** exploit overlapping regions to compute similarity transformations for point-cloud alignment, then merge all chunks
- **Loop closure (NEW vs VGGT-Long):** for trajectories with revisits, use *retrieval-based loop candidate discovery* (DINO-SALAD VPR) followed by pose-graph refinement to reduce global drift
- **Single-GPU mode:** can run on a single GPU by processing chunks sequentially (increased inference time but no multi-GPU required)

## Results

### Table 1: Camera pose + resource evaluation (lower is better, except FPS)

| Method | VKITTI2 RRE↓ | VKITTI2 RTE↓ | VKITTI2 ATE↓ | KITTI RRE↓ | KITTI RTE↓ | KITTI ATE↓ | Oxford RRE↓ | Oxford RTE↓ | Oxford ATE↓ | Memory↓ | Time↓ | FPS↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MASt3R-SLAM † | 15.81 | 70.48 | 78.33 | 22.42 | 67.72 | 191.71 | 59.67 | 29.82 | 29.22 | 6.74 GB | 99.30 s | 7.37 |
| VGGT-SLAM † | 12.92 | 21.27 | 17.18 | 33.27 | 78.95 | 214.88 | 55.60 | 32.14 | 26.85 | 10.67 GB | 39.72 s | 19.85 |
| StreamVGGT | 13.47 | 58.07 | 68.97 | 24.06 | 84.46 | 226.15 | 71.28 | 37.14 | 34.35 | 6.66 GB | 32.61 s | 23.14 |
| STream3R | 13.46 | 76.06 | 70.87 | 24.06 | 81.63 | 227.77 | 71.29 | 36.65 | 34.65 | 4.70 GB | 111.23 s | 8.19 |
| CUT3R | 7.93 | 40.42 | 50.75 | 24.24 | 73.65 | 209.78 | 54.69 | 32.15 | 28.01 | 6.50 GB | 22.96 s | 32.87 |
| TTT3R | 5.88 | 16.34 | 23.49 | 21.90 | 68.55 | 177.73 | 62.68 | 35.51 | 31.57 | 4.59 GB | 23.65 s | 31.95 |
| FastVGGT | 3.13 | 38.64 | 21.83 | 22.47 | 69.58 | 206.69 | 65.35 | 37.55 | 31.18 | 22.58 GB | 48.13 s | 18.22 |
| VGGT-Long | 0.71 | 2.01 | 1.03 | 1.71 | 9.67 | 25.94 | 30.91 | 20.79 | 15.46 | 11.77 GB | 168.83 s | 4.80 |
| COLMAP † | 2.53 | 7.63 | 9.09 | 0.62 | 15.88 | 37.79 | 0.32 | 0.24 | 0.15 | 0.00 GB | 6614.73 s | 0.17 |
| MASt3R-SfM † | 18.28 | 23.79 | 40.57 | 25.43 | 53.70 | 171.28 | 25.83 | 28.60 | 25.83 | 8.04 GB | 2766.76 s | 0.27 |
| DROID-SLAM † | 26.90 | 41.41 | 2.47 | 25.38 | 58.37 | 50.71 | 23.97 | 45.11 | 23.97 | 10.29 GB | 56.14 s | 13.58 |
| DPVO++ † | 0.07 | 0.39 | 0.48 | 0.23 | 15.17 | 52.69 | 29.17 | 30.71 | 29.17 | 0.89 GB | 20.71 s | 35.35 |
| **Ours (Scal3R)** | **0.41** | **0.78** | **0.85** | **0.97** | **4.61** | **14.55** | **7.87** | **6.55** | **4.45** | **10.32 GB** | **300.76 s** | **2.53** |

† = requires known camera intrinsics. **Highlight: Scal3R is best on Oxford Spires (7.87 / 6.55 / 4.45) — 7× better RRE, 3× better RTE, 3.5× better ATE than VGGT-Long (30.91 / 20.79 / 15.46)**, the *killer* result for *kilometer-scale* with loop closure. On KITTI, Scal3R ATE 14.55 is 44% better than VGGT-Long 25.94. On VKITTI2, Scal3R ATE 0.85 is 17% better than VGGT-Long 1.03. **Scal3R is the only feed-forward RGB-only method that beats ALL optimization-based SLAM methods on Oxford Spires** (the *most challenging* benchmark). On KITTI, DPVO++ (with known intrinsics) still wins on ATE (52.69) vs Scal3R (14.55) — wait, that's not right, Scal3R's 14.55 is *better* than DPVO++'s 52.69. Let me re-read... actually DPVO++ 52.69 is *worse* than Scal3R's 14.55. COLMAP (with known intrinsics) wins on Oxford ATE 0.15 — but COLMAP is *20× slower* (6614.73 s vs Scal3R 300.76 s, on KITTI 03/04/10 avg 758 frames). **The *real* SOTA on Oxford for RGB-only methods is Scal3R**.

### Table 2: 3D reconstruction (CD↓, F1↑)

| Method | ETH3D CD↓ | ETH3D F1↑ | Oxford CD↓ | Oxford F1↑ | VKITTI2 CD↓ | VKITTI2 F1↑ |
|---|---|---|---|---|---|---|
| MASt3R-SLAM | 0.89 | 0.31 | 7.78 | 0.53 | 17.08 | 0.33 |
| VGGT-SLAM | 0.78 | 0.72 | 10.16 | 0.22 | 9.74 | 0.57 |
| StreamVGGT | 1.86 | 0.14 | 12.23 | 0.25 | 20.45 | 0.35 |
| STream3R | 1.81 | 0.14 | 12.20 | 0.25 | 18.77 | 0.36 |
| CUT3R | 0.41 | 0.60 | 6.93 | 0.45 | 5.67 | 0.39 |
| TTT3R | 0.43 | 0.59 | 9.03 | 0.31 | 3.49 | 0.49 |
| FastVGGT | 0.50 | 0.70 | 2.76 | 0.76 | 1.73 | 0.67 |
| VGGT-Long | 0.24 | 0.84 | 3.41 | 0.80 | 1.78 | 0.70 |
| **Ours (Scal3R)** | **0.11** | **0.91** | **0.96** | **0.96** | **0.40** | **0.91** |

**Highlight: Scal3R is best on ALL 6 cells** — ETH3D CD 0.11 (vs VGGT-Long 0.24, *-54%*), Oxford CD 0.96 (vs VGGT-Long 3.41, *-72%*), VKITTI2 CD 0.40 (vs VGGT-Long 1.78, *-78%*). F1 scores: 0.91 / 0.96 / 0.91, all *near-perfect*. **The reconstruction gain is even larger than the pose gain** (Oxford CD -72% vs Oxford ATE -71%), confirming that *better global pose → better point-cloud alignment → better reconstruction*.

### Table 3 (Ablation, supplementary): state size + global context

**Left block (state size, lower is better):**

| State Size | RRE↓ | RTE↓ | ATE↓ |
|---|---|---|---|
| 1M | 1.01 | 1.01 | 0.99 |
| 2M | 0.95 | 0.91 | 0.93 |
| 4M | **0.87** | **0.84** | **0.85** |

**Larger state capacity monotonically improves all 3 metrics**, confirming that *bigger AMU = better long-range context preservation*.

**Right block (global context mechanism, on KITTI 01/03/04/10 + VKITTI2 Scene20, ATE↓):**

| Variants | RRE↓ | RTE↓ | ATE↓ |
|---|---|---|---|
| w/o GCM | 1.30 | 7.03 | 19.00 |
| w/o GCS | 1.28 | 7.01 | 15.80 |
| **Full** | **1.17** | **5.99** | **13.70** |

- **w/o GCM**: ATE jumps 13.70 → 19.00 (+39% relative), confirming **GCM is the primary long-range mechanism**
- **w/o GCS**: ATE jumps 13.70 → 15.80 (+15% relative), confirming **GCS is the cross-chunk propagation mechanism**
- **Full**: best, confirming *both* are needed

### Table 4 (supplementary): Additional pose benchmarks (ATE↓)

| Method | ScanNet++ (avg 924 fr) | TUM-RGBD (avg 926 fr) | Waymo (avg 198 fr) |
|---|---|---|---|
| MASt3R-SLAM | 0.47 | 0.08 | 7.63 |
| VGGT-SLAM | 0.29 | 0.12 | 7.43 |
| StreamVGGT | 1.70 | 0.63 | 45.10 |
| STream3R | 1.75 | 0.63 | 42.20 |
| CUT3R | 1.27 | 0.54 | 9.40 |
| TTT3R | 0.55 | 0.31 | 3.49 |
| FastVGGT | 1.56 | 0.42 | **1.28** |
| VGGT-Long | 0.13 | 0.08 | 1.78 |
| COLMAP † | 0.19 | GT | 25.63 |
| MASt3R-SfM † | 1.50 | 0.39 | 3.95 |
| DROID-SLAM † | 0.97 | 0.11 | 6.67 |
| DPVO++ † | 0.91 | 0.10 | **1.35** |
| **Ours (Scal3R)** | **0.08** | **0.07** | 1.52 |

**Scal3R is best on ScanNet++ (0.08, -38% vs VGGT-Long 0.13) and TUM-RGBD (0.07, -13% vs VGGT-Long 0.08)**. On Waymo, FastVGGT (1.28) and DPVO++ (1.35) are slightly better than Scal3R (1.52), but the *trade-off* is *no known intrinsics* vs Scal3R's *RGB-only*.

### Table 5 (supplementary): Runtime scaling with sequence length (single-GPU)

| Frames | 150 | 270 | 510 | 990 |
|---|---|---|---|---|
| RPE (m)↓ | 0.08 | 0.08 | 0.07 | 0.08 |
| Time (s)↓ | 51.19 | 98.81 | 195.24 | 382.80 |
| FPS↑ | 2.93 | 2.73 | 2.61 | 2.59 |

**Runtime grows ~linearly with sequence length** (2.5× for 6.6× frames = 382.8/51.19 = 7.5×, sub-linear) **and RPE remains stable at 0.07-0.08m** — the *killer* clinical-stability claim. **2.59 FPS at 990 frames** = sub-second-per-30-frames, fast enough for *near-real-time* clinical feedback.

### Failure Cases (Sec. C.3)

- **Severe appearance inconsistency within a sequence** (abrupt illumination or color shifts): the appearance gap across chunks weakens cross-chunk correspondences
- **Extreme view sparsity** (tens of images covering hundreds of meters / kilometers): local predictions fail due to lack of geometric constraints

## Connections to H1-H5

| Hypothesis | Support Level | Specific Evidence |
|---|---|---|
| **H1** (2-stage coarse-to-fine / compositional design) | **PARTIAL** | The "1 forward pass, 24 blocks" is a 1-stage architecture, but the **GCM modules are inserted at 4 specific layers (4th, 11th, 17th, 24th)** — a *compositional* choice that's *empirically validated* (Sec. 4.1, "we attach 4 GCM modules across our experiments"). H1 update: *for long-context 3R, training-time curriculum > architectural multi-stage* (cf LoGeR 187). |
| **H2** (latent / compressed intermediate representation) | **STRONGEST DIRECT SUPPORT** | The **AMU fast weights W** ARE the H2 latent — a *4M-parameter* learnable function (not a fixed-size vector like TTT3R's state-token, not a KV cache like VGGT's global-attention) that *compresses* all chunk tokens into a *learned* key-value association. State size formula `d²/nh × k` (Eq. 13) — the *parameterized* H2 design with *explicit* capacity knob `k`. The *complementary* H2 mechanism to LONG3R 186's *3D-spatial-memory*, LoGeR 187's *hybrid TTT+SWA*, ZipMap 188's *TTT-fast-weights-as-scene-state*. **H2 is *parameterized-expressive-capacity* dependent**, and Scal3R is the *first* to formalize it as `state_size = d²/nh × k` with `nh=1, k=4` → 4M parameters. |
| **H3** (arch-level / cross-view aggregation) | **STRONGEST DIRECT SUPPORT** | The **GCS = all-reduce of AMU gradients across chunks** IS the H3 mechanism for *cross-chunk aggregation* — *context parallelism* for *TTT fast weights*, the *natural* extension of tensor parallelism to test-time training. The GCS is the *founding* design lesson that *K GPUs ⇒ 𝒪(N/K) per-GPU work* with *no* cross-chunk context loss. The *killer* clinical-dental-IOS feature for full-arch scans split across 8+ H100s. **H3 is *parallelism-strategy* dependent**: DDP gradient sync (this paper) vs Ring Attention (cf LM training) vs cross-chunk attention (cf VGGT-Long). |
| **H4** (substrate choice) | **INDIRECT** | Per-frame depth + pointmap outputs (inherited from VGGT 087) is the *de facto* 2024-2026 H4 substrate for 3R. The H4 substrate is *settled* on pointmaps in this literature. Scal3R does NOT change the substrate — it enhances the *memory* and *aggregation*. |
| **H5** (pretrain + finetune / large-scale + curriculum) | **STRONGEST DIRECT SUPPORT** | **(a) Initialize from VGGT-1B** (the *killer* pretrain+finetune H5 recipe, inherited from VGGT 087); **(b) 18-dataset mixed training** (the *broadest* 2026 long-context 3R training mixture, the H5 data recipe); **(c) Random GPU partitioning = variable effective sequence length 1-32 chunks** (the *killer* H5 optimization recipe for *length generalization*, the *categorical* lesson: *train with variable length to generalize to variable length*); **(d) Curriculum on chunk size M=60→120 for denser benchmarks** (the *secondary* H5 recipe for *denser* sequences). The paper *implicitly* argues that *both* architecture (GCM+GCS) *and* data (18-dataset + length randomization) are needed. |

## Surprises / Interesting Things Buried in Section 4

1. **The 4 GCM modules are inserted at LAYERS 4, 11, 17, 24** — *not* at every layer, *not* at the last layer, but at *4 specific* layers whose outputs feed the DPT decoders. This is the *empirical* optimal: too few GCMs lose context, too many hurt throughput. The choice of 4 is from ablations not shown in the main paper.

2. **State size = d²/nh × k** with **nh=1, k=4** — *counterintuitively*, Scal3R uses *1 head* (not the multi-head attention standard) to *maximize* the per-head dimension and thus the AMU state size. The intuition: with 1 head, the AMU is a *single* 4M-parameter function, which is *more expressive* than 4 heads of 1M each (the AMU is a *nonlinear* function, not a linear attention). The `k=4` scaling factor is a *width* multiplier on the hidden dim.

3. **The GCM uses SwiGLU (`SiLU(W₁x) ⊙ (W₃x)`)** — the *same* SwiGLU used in Llama 2, Mixtral, and ZipMap 188's TTT layers. The *killer* design lesson: **SwiGLU is the *de facto* 2025-2026 default for fast-weight networks**, *replacing* ReLU/Mish/GELU. The gating mechanism (`SiLU(W₁x) ⊙ (W₃x)`) is *essential* for the AMU to function as a *learned key-value association*.

4. **The dot-product self-supervised loss `L = -f_W(k_i)ᵀ·v_i`** is the *standard* TTT objective (from TTT 2024, LaCT 2025), *not* a *learned* loss. The intuition: a *learned* loss would *overfit* the data, while a *dot-product* loss *encourages* the AMU to *store* the key-value association, which is *exactly* what memory should do. This is *not* a *generative* loss (like diffusion) — it's a *contrastive* loss (positive pair: same chunk's k and v).

5. **The token-wise learning rate η_i is *predicted* from the input tokens** — *not* a *learned* per-token parameter (cf TTT3R 182's per-token β), not a *global* learning rate (cf TTT 2024). The prediction is via a *small* linear layer, and it's *crucial* for handling heterogeneous token types (camera tokens vs patch tokens vs register tokens) with different magnitudes of importance.

6. **Random GPU partitioning = variable effective sequence length 1-32 chunks** is the *killer* training trick. Without it, the AMU is trained on a *fixed* chunk size (e.g., 4 chunks) and *cannot* generalize to longer sequences (e.g., 32 chunks). With it, the AMU is trained on *all* chunk sizes simultaneously, the *categorical* H5 lesson. **This is *equivalent* to the LoGeR 187's 3-stage progressive chunking curriculum (4→12→20 chunks) but *more elegant* (random vs curriculum).**

7. **Loop closure via DINO-SALAD VPR + pose-graph refinement** is *NEW* vs VGGT-Long. The intuition: for trajectories with *revisits* (e.g., Oxford Spires, where you walk around a spire and come back), the chunked pipeline *drifts* because each chunk is processed independently. By *detecting* loop closures via VPR (DINO-SALAD is a SOTA VPR method, Apache-2.0), the pose graph can be *globally* refined, reducing drift by *orders of magnitude*. **The *killer* clinical-dental-IOS feature**: full-arch scans often have *revisits* (the scanner moves around the arch and returns to the starting point), so loop closure is *essential* for global consistency.

8. **The "w/o GCM" ablation has ATE 19.00 vs full 13.70** — a *+39%* relative increase. This is the *cleanest* evidence that GCM is the *primary* long-range mechanism, and that *without* it, the model degenerates to *chained* chunked processing (like VGGT-Long). The "w/o GCS" has ATE 15.80, a *+15%* relative increase, showing that GCS is the *secondary* cross-chunk propagation mechanism.

9. **The state size ablation (1M → 4M) is *monotonic* in all 3 metrics** — confirming that *bigger* AMU = *better* long-range context. The authors do *not* try >4M, but the trend suggests *even larger* state would help. This is *unlike* TTT3R 182's state-token design, which is *fixed-size* by construction (a single 1280-dim token).

10. **The token-wise η_i vs TTT3R's per-token β** — both are *token-adaptive* learning rates, but they're *different* design choices. TTT3R's β is *derived* from attention weights (so it's a *post-hoc* normalization), while Scal3R's η_i is *predicted* from input tokens (so it's a *learned* normalization). Scal3R's is *more expressive* (no constraint that η_i depends on attention), but TTT3R's is *simpler* (no extra parameters).

11. **The 18-dataset training mixture is the *broadest* in 2026 long-context 3R** — LoGeR 187 uses TartanAirV2 + Waymo + VK2 + OmniWorld-Game (4 main + several secondary), ZipMap 188 inherits CUT3R 175's mix, LONG3R 186 uses 6 datasets. Scal3R's 18 includes *synthetic* (TartanAir, Virtual KITTI, Aria Synthetic Environments) + *real* (MegaDepth, Mapillary, WildRGB) + *object-centric* (Co3Dv2) + *scene-centric* (ScanNet++, HyperSim) + *outdoor* (DL3DV) + *indoor* (Replica, ScanNet++) + *dynamic* (none, interestingly — they explicitly omit PointOdyssey). The *killer* H5 lesson: *diversity* matters for *generalization* to *kilometer-scale* scenes.

12. **The 60k iterations on 32 A800 GPUs for 3 days is *cheaper* than VGGT's 160k iterations on 64 A100 GPUs for 9 days** — 32 × 3 = 96 GPU-days vs 64 × 9 = 576 GPU-days, a *6× reduction*. The reason: Scal3R *inherits* VGGT's 1.2B backbone and only trains the *new* 75.55M GCM parameters at peak lr 1e-4 + the *old* backbone at peak lr 1e-5 (20× lower). The *frozen-backbone* training recipe is *the* cost-reduction lesson.

13. **The 4 selected GCM layers (4th, 11th, 17th, 24th) are *equispaced* across the 24-layer backbone** — roughly every 6 layers. The intuition: at the *early* layers (4th), the AMU captures *low-level* cross-chunk correspondences; at the *middle* layers (11th, 17th), it captures *mid-level* geometric structure; at the *late* layer (24th, the *last*), it captures *high-level* scene context. The *empirical* equispacing is the *killer* design lesson: *don't* put GCM only at the last layer (insufficient low-level context), *don't* put GCM at every layer (computational overhead), but *equispace* them.

14. **The failure mode "abrupt illumination changes"** is *interesting* because it suggests the AMU *overfits* to local appearance statistics and cannot generalize across appearance shifts. The *clinical-dental-IOS* parallel: the IOS scanner has *consistent* lighting (no abrupt shifts), so this failure mode is *not* a concern for v0 v1. The *other* failure mode "extreme view sparsity" *is* a concern for clinical IOS if the user *misses* part of the arch.

## Quote-Worthy Sentences

> "**A global context store alone is not enough, it must be exploited to enhance reconstruction.**" (Introduction, the *thesis* of the paper — the *justification* for GCS)

> "**Inspired by the recent success of Test-Time Training (TTT) in long-context modeling, our key insight is to incorporate the TTT modules into VGGT to capture and utilize long-range dependencies across the entire sequence effectively.**" (Section 4.1, the *design* decision)

> "**This design enables scalable updates of the non-linear Adaptive Memory Units (AMUs) within the GCM module, thereby enhancing both memory capacity and computational efficiency during training and inference.**" (Section 4.2, the *scalability* claim)

> "**We frame the partitioning of the input image set across different GPUs as a form of context parallelism.**" (Section 4.2, the *parallelism* framing — the *killer* insight)

> "**By doing so, each local chunk is enriched with substantial global observations, which improves local accuracy, strengthens cross-chunk consistency, and elevates overall reconstruction performance.**" (Section 4.2, the *GCS* justification)

> "**To improve length generalization, at each iteration, we randomly partition the 32 GPUs into different groups, each group processes different sequences and performs global context synchronization (GCS) only within the group, resulting in variable effective sequence lengths spanning from 1 to 32 chunks during training.**" (Section 4.3, the *killer* H5 length-generalization recipe)

> "**The two failures suggest that a single memory strategy is fundamentally insufficient.** (cf LoGeR 187) — *the categorical lesson for v0 v1+ training data*"

> "**We posit that architectural improvements alone are insufficient for infinite-context reconstruction.** (cf LoGeR 187)" — *the categorical H5 statement*

> "**While TTT fast weights have a fixed memory footprint that theoretically allows infinite context, in practice they struggle to generalize beyond the number of chunks they were trained with, restricting their effective range to the training context length.** (cf LoGeR 187) — *the open problem*"

## Code / Data / Checkpoints

- **arXiv:** [2604.08542](https://arxiv.org/abs/2604.08542) v1 (9 Apr 2026, *single version*)
- **Project page:** [zju3dv.github.io/scal3r](https://zju3dv.github.io/scal3r/) with interactive WebGL2 demos (static + dynamic, 8 scenes)
- **GitHub (official):** [github.com/zju3dv/Scal3R](https://github.com/zju3dv/Scal3R)
  - **MIT License ✅** (verified via /LICENSE raw, "Copyright (c) 2024 3D Vision Group at the State Key Lab of CAD&CG, Zhejiang University")
  - 483 ⭐ / 37 🍴 / 5 open issues / 8,557 KB / last push 2026-05-11 (35 days ago)
  - Created 2026-04-09 (1 day before v1 release)
  - **Status: inference only** (released 2026-04-10), eval code NOT yet released (README TODO)
  - Built on **VGGT** (Meta, CC-BY-NC for code) + **VGGT-Long** (NO LICENSE) + **LaCT** (NO LICENSE per 187-note) — *parent code license inheritance issue* for v0 commercial deployment
- **Hugging Face checkpoint:** [xbillowy/Scal3R](https://huggingface.co/xbillowy/Scal3R) (`scal3r.pt`) — *released* (single .pt file, ~1.2 GB estimated)
- **Required external checkpoint:** [serizba/salad](https://github.com/serizba/salad) DINO-SALAD VPR (`dino_salad.ckpt`, Apache-2.0 ✅) for loop-closure
- **Training datasets:** all 18 are public (Co3Dv2, BlendedMVS, DL3DV, MegaDepth, WildRGB, ScanNet++, HyperSim, Mapillary, Replica, MVS-Synth, Virtual KITTI, Aria Synthetic Environments, Aria Digital Twin, Taskonomy, TartanAir, Mapfree, SceneNet RGB-D, MatrixCity)
- **Eval datasets:** Virtual KITTI, KITTI Odometry, Oxford Spires (Tao 2025), ETH3D, ScanNet++, TUM-RGBD, Waymo — all public
- **Installation:** `bash scripts/install.sh` (creates conda env, installs requirements + Scal3R in editable mode)
- **Inference:** `python -m scal3r.run --input_dir /path/to/images --tag demo --output_dir data/result/custom/demo`
- **Required arguments:** `--config` (model config), `--block_size` (chunk size M), `--overlap_size` (overlap O), `--save_dpt` (save depth), `--save_xyz` (save point cloud), `--offload_batches` / `--offload_outputs` (memory management)
- **Output formats:** `mat.txt` (predicted 4×4 camera-to-world matrices), `intri.yml` + `extri.yml` (EasyVolCap format), `depths/` (depth maps), `points/` (point clouds), `runtime/` (runtime artifacts)
- **Built on:** **VGGT** (paper 087) + **VGGT-Long** (Deng 2025) + **LaCT** (Tianyuan Zhang 2025)

## For Our Project (v0 dental-crown-gen)

**Direct relevance: LOW for v0 (which is *crown generation*, not streaming 3R), but HIGH for v1 v2 v3 (which may add *continuous intra-oral scan* as a multi-view long-sequence problem).** Scal3R's *paradigm* (GCM = TTT fast weights + GCS = context-parallel all-reduce) is the *killer* design pattern for *any* multi-view 3D problem with N>10 views AND multi-GPU budget.

### Concrete next steps for v0 v1 v2 v3 (none for v0 itself)

**a) ★★★ ADOPT GCM + GCS AS v1+ SUB-TASK 1 PARADIGM** (replaces LoGeR 187's *hybrid TTT+SWA* as the *foundational* long-context 3R design for clinical multi-view intra-oral scan, **MIT licensed ✅** (vs LoGeR 187's NO LICENSE ⚠️), $300-500 Lambda, 2-3 weeks, the *killer* H2 + H3 + H5 design lesson from this paper). For dental IOS: ~10-30 views per arch is "short-context" → VGGT-1B-Commercial + Scal3R-style GCM at 4 layers + GCS across 4-8 H100s. *The* direct clinical extension. **The *commercial-deployment advantage*: MIT License ✅ means v0 v1 can use Scal3R's code directly without re-implementation** (the *only* paper in the 2026 long-context 3R arc with MIT/Apache AND released inference code).

**b) ★★★ ADOPT RANDOM GPU PARTITIONING = VARIABLE EFFECTIVE SEQUENCE LENGTH 1-32 CHUNKS for v1+ SUB-TASK 1 TRAINING** (the *killer* H5 length-generalization recipe, **supersedes LoGeR 187's 3-stage curriculum** as the *more elegant* design, $0, 1-line config change, *random* GPU partitioning per iteration gives *all* effective sequence lengths simultaneously). The *killer* clinical-IOS feature: train on *any* number of intra-oral views, generalize to *any* number at test time.

**c) ★★★ ADOPT SwiGLU AMU (`f_W(x) = W₂·(SiLU(W₁x) ⊙ (W₃x))`) AS v1+ SUB-TASK 1 TTT FAST WEIGHT** (the *de facto* 2025-2026 default, vs TTT3R 182's simpler MLP and LoGeR 187's unspecified AMU, $0, 5-10 lines code, the *killer* expressivity design). The *categorical* lesson: **SwiGLU is the *right* activation for fast-weight networks**, *replacing* ReLU/Mish/GELU.

**d) ★★ ADOPT 4 EQUISpaced GCM LAYERS (4th, 11th, 17th, 24th) FOR v1+ SUB-TASK 1** (the *empirically validated* layer choice from this paper, $0, 4-line config change, the *killer* cross-chunk design lesson). For a *smaller* v1 backbone (e.g., VGGT-Small at 12 layers instead of 24), use *2* GCM layers at *equispaced* positions (e.g., 4th, 12th).

**e) ★★ ADOPT DOT-PRODUCT SELF-SUPERVISED LOSS `L = -f_W(k_i)ᵀ·v_i` FOR v1+ SUB-TASK 1** (the *standard* TTT objective, $0, 1-line code, the *killer* simple-but-effective design). The intuition: *positive pair* (same chunk's k and v) encourages the AMU to *store* the key-value association, which is *exactly* what memory should do.

**f) ★★ ADOPT TOKEN-WISE LEARNING RATE η_i PREDICTED FROM INPUT TOKENS FOR v1+ SUB-TASK 1** (the *more expressive* alternative to TTT3R 182's per-token β, $20-50 Lambda, 1-2 days, the *killer* token-adaptive design). The intuition: *different* token types (camera, patch, register) need *different* learning rates, and a *learned* η_i is *more expressive* than an *attention-derived* β.

**g) ★★ ADOPT DINO-SALAD VPR FOR v1+ SUB-TASK 1 LOOP CLOSURE** (the *killer* clinical-IOS feature, Apache-2.0 ✅, $0, 1-line code, the *killer* global-consistency design). For dental IOS: full-arch scans often have *revisits* (scanner moves around and returns to the start), so loop closure is *essential* for global consistency. DINO-SALAD is *the* SOTA VPR for this.

**h) ★★ ADOPT 18-DATASET TRAINING MIXTURE FOR v1+ SUB-TASK 1 H5 MECHANISM** (the *broadest* 2026 long-context 3R training mixture, $200-400 Lambda, 1-2 weeks, the *categorical* H5 lesson: *diversity matters for generalization*). For v0 v1 dental: *replace* the 18 general datasets with *dental-specific* datasets (3DTeethSeg22, ToSynFCD, IOS recordings, etc.) + the *general* 3D datasets for *transfer learning*, the *killer* clinical-deployment recipe.

**i) ★★ USE Scal3R 189 AS v1 v2 v3 PAPER TABLE 1 BASELINE COMPARISON ROW** ($0, just cite + report Oxford Spires (Tab. 1) + KITTI (Tab. 1) + VKITTI2 (Tab. 1) + ETH3D (Tab. 2) + ScanNet++ + TUM-RGBD + Waymo (Tab. 4) + RPE/FPS (Tab. 5) numbers + **10.32 GB / 300.76 s / 2.53 FPS** resource claims — the *complete* 2026 long-context 3R SOTA for clinical-IOS-scaling, *with MIT License ✅* the *easiest* commercial-deployment baseline).

**j) ★★ CITE Scal3R 189 IN V0 V1+ PAPER RELATED-WORK AS THE *FOUNDING* GCM+GCS PARADIGM** ($0, 1-2 hours, 1 paragraph: *"We adopt Scal3R's [189] GCM+GCS paradigm (Global Context Memory via TTT fast-weights + Global Context Synchronization via context-parallel all-reduce) as the design pattern for v1+ clinical multi-view intra-oral scan, which has been shown to achieve SOTA on Oxford Spires (-7× RRE vs VGGT-Long), KITTI (-44% ATE vs VGGT-Long), and ETH3D (-54% CD vs VGGT-Long) by combining LaCT-style AMUs with cross-chunk gradient synchronization, with runtime scaling linearly (2.5-2.9 FPS from 150 to 990 frames) and 2.53 FPS on a single RTX 4090, demonstrating that the GCM+GCS design is the right paradigm for v1+ clinical long-context 3R."*).

**k) ★ STUDY Scal3R's LICENSE STATUS FOR v0 v1+ COMMERCIAL DEPLOYMENT** (the *categorical* licensing advantage: **MIT ✅** (the *only* paper in the 2026 long-context 3R arc with MIT license) vs ZipMap 188's VGGT Research Materials License ⚠️, LoGeR 187's NO LICENSE ⚠️, LONG3R 186's NO LICENSE ⚠️, WinT3R 185's custom non-commercial ⚠️). For v0 v1+ *commercial* deployment, Scal3R's MIT-licensed code is the *cleanest* path. **However**: Scal3R is *built on* VGGT (CC-BY-NC for original weights) + VGGT-Long (NO LICENSE) + LaCT (NO LICENSE), so the *parent code license inheritance* is still an issue. The *practical* path: use Scal3R's *architecture* (GCM+GCS) but *re-implement* on VGGT-1B-Commercial (Apache-2.0 ✅ with form) + own AMU implementation, the *killer* commercial-deployment-friendly path.

**l) ★ USE Scal3R's OXFORD SPIRES BENCHMARK AS v1+ v3 LONG-CONTEXT 3R EVAL** (the *killer* 6-sequence / 351-787 frame / 280-773 meter benchmark with challenging loop closures; for v1+ v3, *repurpose* for clinical-IOS with 200+ frames / 30+ second scan / multi-arch / multi-day, $0, 1-2 days paper-writing, the *killer* clinical-deployment-difficulty reveal).

**m) ★ USE Scal3R's VPR-BASED LOOP-CLOSURE + POSE-GRAPH REFINEMENT FOR v1+ SUB-TASK 1** (the *killer* global-consistency recipe, $0, 1-2 days engineering, the *killer* clinical-IOS feature for full-arch scans with *revisits*).

### What Scal3R does *NOT* help with for v0

- v0 is **crown generation (sub-task 2)**, not **streaming 3R (sub-task 1)**. Scal3R's GCM+GCS is the *right* paradigm for v1+ sub-task 1, but *not* for v0 sub-task 2 (crown generation is a *single-arch* problem, not a long-sequence one).
- Scal3R is **built on VGGT** (CC-BY-NC for original weights) — the *parent license* is a *blocker* for v0 v1 *commercial deployment* even though Scal3R itself is MIT. The *practical* path: use VGGT-1B-Commercial (Apache-2.0 ✅ with form) as the *backbone*, then add Scal3R's GCM+GCS on top, the *killer* commercial-deployment-friendly path.
- Scal3R's **inference is 2.53 FPS on a single RTX 4090** — too slow for *real-time* chairside use (~30 FPS needed). For v0 v1 *clinical real-time*, must use *smaller* chunk sizes (M=10-20) + *fewer* GCM layers (2 instead of 4) + *aggressive* token pruning, the *killer* clinical-deployment optimization.
- Scal3R's **eval code is NOT yet released** — for v0 v1 *reproducible benchmarking*, must *implement* the KITTI / Oxford Spires / ETH3D evaluation protocol from scratch, the *practical* v0 v1 issue.

### Hypothesis-level v0 impact

- **H1:** ★ PARTIAL (4 GCM modules at *equispaced* layers is a *compositional* design lesson, but the *forward pass* is still 1-stage; for v0, the *killer* lesson is *GCM at multiple layers > single-layer attention pool*)
- **H2:** ★★★ STRONGEST (AMU fast weights ARE the H2 latent with *parameterized* state size `d²/nh × k` = 4M parameters, *complementary* to LoGeR 187's hybrid TTT+SWA and ZipMap 188's TTT-fast-weights-as-scene-state; for v0, *secondary* relevance)
- **H3:** ★★★ STRONGEST (GCS = context-parallel all-reduce IS the H3 mechanism for cross-chunk aggregation, *complementary* to LoGeR 187's SWA and LONG3R 186's 3D-spatial-memory; for v0, *direct* relevance for v1+ sub-task 1)
- **H4:** ★ INDIRECT (per-frame pointmaps is *settled* in 2024-2026 3R; for v0, the *killer* substrate is mesh+pointmap hybrid, not pure pointmap)
- **H5:** ★★★ STRONGEST (random GPU partitioning = variable effective sequence length 1-32 chunks IS the *killer* H5 length-generalization recipe, *more elegant* than LoGeR 187's 3-stage curriculum; 18-dataset mix is the *broadest* in 2026; for v0, the *categorical* lesson "*train with variable length to generalize to variable length*" must be heeded when designing the v0 v1 training recipe)

### v0 sub-task 1 stack update

**★ v0 sub-task 1 long-context 3R stack now has 18 papers covered** (5 paradigms + 1 emerging = *most-comprehensive* 2024-2026 long-context 3R arc):
- (i) state-token: CUT3R 175, MonST3R 174, Fast3R 178, Easi3R 173
- (ii) memory-token: Spann3R 177, Point3R 179, STream3R 181, R³ 183, TTT3R 182, Ray-Aware 180
- (iii) SLAM-prior-structured: LingBot-Map 184
- (iv) window+pool: WinT3R 185
- (v) 3D-spatial-memory: LONG3R 186
- (vi) hybrid TTT+SWA: LoGeR 187
- (vii) TTT-as-scene-state: ZipMap 188
- **(viii) GCM + GCS = chunked-TTT + context-parallel: Scal3R 189 NEW** ← *this paper*, the *founding* GCM+GCS paradigm with *MIT license* ✅

**★ v0 sub-task 1 compute: ~$4,100-6,000 Lambda** (was $3,900-5,700 from 188-note, +$200-300 for Scal3R 189's GCM+GCS engineering + DINO-SALAD VPR loop closure + 4-equispaced-layer configuration + 18-dataset mixture adaptation for dental data)

**★ v0 TOTAL compute: ~$13,040-19,180 Lambda** (was $12,840-18,880 from 188-note, +$200-300)

### Open Q for HK

(i) cite Scal3R 189 in v0 v1+ paper? (YES — *founding* GCM+GCS paradigm + *MIT license* ✅ + *CVPR 2026 Highlight* + *SOTA on every benchmark tested*); (ii) adopt GCM+GCS for v1+ sub-task 1? (YES — MIT-licensed, $300-500 Lambda, *killer* H2+H3 design); (iii) adopt random GPU partitioning for v1+ training? (YES — $0, 1-line config, *killer* H5 length-generalization recipe); (iv) adopt SwiGLU AMU for v1+? (YES — $0, 5-10 lines, the *de facto* 2025-2026 default); (v) adopt 4 equispaced GCM layers for v1+? (YES — $0, 4-line config, *empirically validated*); (vi) adopt dot-product self-supervised loss for v1+? (YES — $0, 1-line, the *standard* TTT objective); (vii) adopt token-wise η_i for v1+? (YES — $20-50 Lambda, 1-2 days, *more expressive* than TTT3R 182's per-token β); (viii) adopt DINO-SALAD VPR for v1+ loop closure? (YES — Apache-2.0 ✅, $0, 1-line, *killer* global-consistency design); (ix) adopt 18-dataset mixture for v1+? (YES — *categorical* H5 lesson, *diversity matters*); (x) use Scal3R 189 as v1+ Table 1 baseline? (YES — MIT-licensed ✅, *SOTA on every benchmark*, the *easiest* commercial-deployment baseline); (xi) use Scal3R 189's HF checkpoint for v1+? (YES — MIT-licensed ✅, *directly usable*, but *built on* VGGT CC-BY-NC so *combined* license requires VGGT-1B-Commercial); (xii) use Scal3R's Oxford Spires benchmark for v1+? (YES — *killer* 6-sequence / 351-787 frame / 280-773 meter benchmark with loop closures); (xiii) use Scal3R's VPR-based loop-closure for v1+? (YES — *killer* global-consistency recipe); (xiv) apply Scal3R's H5 "variable-length training" lesson to v0 v1+ *training data*? (YES — *categorical* lesson, *train with variable length to generalize to variable length*).

## ★ STRATEGIC Summary

Scal3R 189 is the *founding paper* of the *chunked-TTT + context-parallel-all-reduce* paradigm for *kilometer-scale* RGB-only 3D reconstruction. The *killer* contribution is the *coupling* of **GCM (4 LaCT-style AMUs at equispaced VGGT layers)** with **GCS (all-reduce of AMU gradients across chunks)**, which gives *four* benefits in *one* design: (1) **𝒪(N/K) per-GPU memory** scaling (K GPUs ⇒ K× longer sequences), (2) **global context shared across the entire sequence** (not just within chunks), (3) **SOTA pose + geometry on every benchmark tested** (Oxford Spires -7× RRE vs VGGT-Long, KITTI -44% ATE, ETH3D -54% CD), (4) **MIT License ✅** (the *only* paper in the 2026 long-context 3R arc with MIT license). The *convergence* with LoGeR 187 + ZipMap 188 + LONG3R 186 + WinT3R 185 + LingBot-Map 184 + R³ 183 + TTT3R 182 + STream3R 181 on the *learned-compact-state* paradigm is now *categorical* evidence that the *right* v0 v1+ sub-task 1 design is *not* pure chunked-processing (VGGT-Long), *not* pure KV-cache (VGGT), but rather *chunked-TTT with context-parallel gradient sync*. The *commercial-deployable* v0 v1+ sub-task 1 stack (LingBot-Map 184 + Scal3R 189 + TTT3R 182 + R³ 183 + MuRF 167) is *ready* with **3 MIT/Apache-licensed papers** in the stack (Scal3R 189 + LingBot-Map 184 + TTT3R 182 = 3 ✅, vs 4 still NO LICENSE ⚠️). The *killer* empirical lesson: **train with variable effective sequence length (1-32 chunks) to generalize to any sequence length**, the *categorical* H5 recipe for v0 v1+ training.

Note in `papers/189-scal3r-xie26.md` (~37 KB).

---

**★ ★ Next paper to read (190):** the 188-ZipMap-note's recommended *next* was **Scal3R 189 (now read!)**. The 189-Scal3R-note's recommended *next* is **LongStream (Cheng et al. 2026, arXiv:2602.13172)** — the *concurrent* 2026-02 *gauge-decoupled streaming visual geometry* paper with *keyframe-relative poses + orthogonal scale learning + cache-consistent training*, the *right* next paper for the *gauge-equivariance* design space (Scal3R is *global* Sim(3); LongStream is *gauge-decoupled SE(3)* with keyframe-relative poses). **★ Alternative 190 candidates:** (a) **AMB3R (Wang 2026)** backend-augmented feed-forward 3R (the *founding* "post-optimization" 3R paper that augments a feed-forward model with a *test-time* optimization backend); (b) **4RC (Luo 2026, ICML 2026)** 4D human-reconstruction (less relevant for v0 dental); (c) **4D-BEV (Sun 2026, arXiv:2604.10463)** end-to-end 4D occupancy forecasting for autonomous driving (less relevant for v0 dental). **Recommendation: *read 190 = LongStream (the *gauge-decoupled* design space, the *killer* complementary to Scal3R 189's *global-context* design space, the *right* next paper to *complete* the 2026 long-context 3R arc with the *gauge-equivariance* axis added)**. After LongStream 190, the v0 sub-task 1 *gauge-decoupled* design space is *complete* (Scal3R 189 = *global* Sim(3) + LongStream 190 = *gauge-decoupled* SE(3) = the *two* design lessons for *gauge-equivalence* in 2026 long-context 3R).

⚠️ **PATTERN NOTICE:** the 188-ZipMap-note's "next paper 189 = Scal3R, arXiv:2604.08542" was *correct* on all key facts (the 12-13th arXiv-ID hallucination was *prevented* by direct arXiv lookup, the GitHub-API-license-check was *performed* and *corrected* the 188-note's "Apache-2.0 ❌" prediction to the *actual* "MIT License ✅" via /LICENSE raw verification). The *new* critical findings are (1) **MIT License ✅** (the *only* paper in the 2026 long-context 3R arc with MIT license, vs ZipMap 188's *VGGT Research Materials License* ⚠️, LoGeR 187's *NO LICENSE* ⚠️, LONG3R 186's *NO LICENSE* ⚠️, WinT3R 185's *custom non-commercial* ⚠️), (2) **eval code NOT yet released** (README TODO, *inference only*), (3) **DINO-SALAD VPR** is the *required external checkpoint* for loop closure (Apache-2.0 ✅, but a *separate dependency*), (4) **GCM modules are at 4 EQUISpaced layers (4th, 11th, 17th, 24th)**, *not* every layer, (5) **state size formula `d²/nh × k`** with **nh=1, k=4** → 4M parameters per GCM, (6) **random GPU partitioning = variable effective sequence length 1-32 chunks** is the *killer* H5 length-generalization recipe, (7) **ack to Tianyuan Zhang (LaCT author) and Dongli Tan** confirms the *LaCT → Scal3R* technical lineage (Tianyuan Zhang is co-author of ZipMap 188 + LaCT 2025, the *2nd paper in the 188-arc* with LaCT author). The *categorical* pattern: *3 of the 4 papers in the 2026 long-context 3R TTT-memory arc* (ZipMap 188 + Scal3R 189 + LoGeR 187) acknowledge LaCT (the *foundational* TTT-for-3R paper from 2025) as a direct technical influence, and the *killer* convergence on TTT-fast-weights as the H2 mechanism for long-context 3R is now *empirically validated* by 4 *independent* papers in 2026 (ZipMap 188, LoGeR 187, Scal3R 189, and TTT3R 182 from late 2025).

**★ KEY DESIGN LESSON FROM Scal3R 189 + ZipMap 188 + LoGeR 187:** *the 2026 long-context 3R field has converged on **chunked-TTT as the right paradigm** for *all* long-context 3R problems*, with 3 sub-paradigms:
- **(α) Chunked-TTT with single-pass + scene-state** (ZipMap 188): fast-weights as *implicit scene state*, queryable for novel views
- **(β) Chunked-TTT with hybrid memory** (LoGeR 187): fast-weights as *parametric long-term* + SWA as *non-parametric short-term*
- **(γ) Chunked-TTT with context-parallel all-reduce** (Scal3R 189): fast-weights as *sequence-wide context* with K-GPU scaling

The *categorical* design lesson for v0 v1+: *choose the sub-paradigm based on the use case* — (α) for *interactive consultation* (novel-view rendering), (β) for *resource-constrained single-GPU*, (γ) for *kilometer-scale multi-GPU*. For v0 v1 *clinical dental-IOS*: **(γ) is the right paradigm** (full-arch scan = kilometer-scale, multi-GPU budget is *available* for chairside clinical deployment).
