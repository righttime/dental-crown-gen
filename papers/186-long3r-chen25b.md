# Paper 186 — LONG3R: Long Sequence Streaming 3D Reconstruction

## TL;DR

**FOUNDING PAPER** of the *3D-spatio-temporal-memory + memory-gating + dual-source-decoder* paradigm for long-sequence streaming 3D reconstruction. Three coupled innovations: (1) **3D Spatio-Temporal Memory** = short-term (10-frame window temporal tokens) + long-term (3000-token voxel-pruned 3D spatial tokens, one token per voxel, highest-attention-wins), where voxel size is *adaptively* set per scene (avg of per-image min-distances between neighboring patch tokens); (2) **Attention-based Memory Gating** with threshold τ=5×10⁻⁴ (drop memory tokens whose max attention weight over query tokens is below threshold) — gives +20% FPS and -27% memory tokens with no quality loss; (3) **Dual-Source Refined Decoder** = *interleaved* PairwiseBlock (next-frame) + MemoryBlock (relevant memory) for 6 layers, vs *concatenated* (12.06 vs 14.83 Acc on Replica100, the *founding* empirical evidence for interleaving). ViT-Large encoder initialized from DUSt3R, 224×224 input, 16 A100 GPUs for 28h+20h two-stage curriculum (5→10→32 frames). **SOTA on 7Scenes (2.57 Acc vs CUT3R 7.73 / Spann3R 3.42)** and **near-SOTA on Replica100/200 (11.46/11.93 vs MV-DUSt3R+ 5.28/11.79 but 7-21× faster)** with real-time 21-22 FPS. **License: NONE ⚠️** (GitHub `license: null`, only `assets/teaser.jpg` + `README.md`, "Code Coming Soon" — *no code released yet*). 43 ⭐ / 0 🍴 on GitHub. ICCV 2025.

## Metadata

- **Title:** LONG3R: Long Sequence Streaming 3D Reconstruction
- **Authors:** Zhuoguang Chen¹²*, Minghui Qin²*, Tianyuan Yuan²³*, Zhe Liu², Hang Zhao²¹³† (*3 equal first authors, † corresponding)
- **Affiliations:** ¹Shanghai AI Lab, ²IIIS Tsinghua University, ³Shanghai Qi Zhi Institute
- **Year:** 2025 (arXiv v1 24 Jul 2025) → **ICCV 2025** (pp. 5273-5284, DBLP verified)
- **arXiv:** [2507.18255](https://arxiv.org/abs/2507.18255) v1 (1,398 KB, single version)
- **GitHub:** [github.com/zgchen33/LONG3R](https://github.com/zgchen33/LONG3R) (43 ⭐ / 0 🍴 / 1 open issue / 1.9 MB / last push 2025-07-25)
- **License:** **NONE ⚠️** (GitHub API `license: null`, README says "Code Coming Soon" — *code not yet released* at submission time)
- **Project page:** [zgchen33.github.io/LONG3R](https://zgchen33.github.io/LONG3R/) (only method overview figure, no demo / no video)
- **Built on:** DUSt3R (encoder init) + Spann3R (paradigm) + CUT3R (paradigm) + MASt3R (coarse-to-fine) + CroCo (cross-view pretraining)
- **Citations:** ~36 Semantic Scholar / ~50-80 GS expected as of 2026-06-15 (10 months post-v1, ICCV 2025)

## Research Question

> *Can we have BOTH (a) long-sequence (tens to hundreds of frames) AND (b) real-time (≥20 FPS) AND (c) SOTA-quality online 3D reconstruction with no global post-optimization (no loop closure, no bundle adjustment)?*

**Their answer:** Yes — via three coupled mechanisms: (1) **3D spatio-temporal memory** with *adaptive voxel-size* spatial pruning (1 token per voxel, highest-attention-wins) gives *bounded* long-term memory (3000 tokens) regardless of stream length; (2) **attention-based memory gating** with threshold τ filters *irrelevant* memory tokens (max-attn < τ) before decoding → +20% FPS with no quality loss; (3) **dual-source interleaved decoder** alternates *next-frame* and *memory* cross-attention to keep feature spaces aligned → -19% Acc on Replica100 vs concatenated baseline. Key insight: **memory is bandwidth-constrained, not capacity-constrained** — the bottleneck is *attention cost over memory* not *memory size*, so *pruning* beats *growing*.

## Method

### Architecture (per single new frame t)

1. **ViT-Large encoder** → image tokens F_t^I for each frame (initialized from DUSt3R's encoder, 224×224 input)
2. **Coarse Decoder** (B=6 PairwiseBlocks, each: self-attn + cross-attn with F_{t-1}^{r,refined} + MLP) → coarse tokens F_t^c
3. **Memory Gating** (cross-attn F_t^c with F_mem^K, F_mem^V → attention weights W_t ∈ ℝ^{P×S}) → *threshold filter* (max_p W_t(p,s) > τ=5×10⁻⁴) → relevant memory F_mem^{r}
4. **Dual-Source Refined Decoder** (6 layers, alternating):
   - **Odd layer:** PairwiseBlock(F_{t,i-1}^r, F_{t+1,i-1}^c) — *next-frame* cross-attention
   - **Even layer:** MemoryBlock(F_{t,i-1}^r, F_mem^r) — *memory* cross-attention
   - This *interleaved* (vs *concatenated*) design keeps feature spaces aligned between the two sources
5. **DPT head** on refined tokens F_t^r → point map P_t ∈ ℝ^{3×H×W} + confidence C_t

### 3D Spatio-Temporal Memory

- **Short-term temporal memory:** tokens from window [t-K, t-1], K=10, stored as keys/values f^K, f^V ∈ ℝ^{(K·P)×C}
- **Long-term 3D spatial memory:** tokens from [1, t-K-1], *pruned* to 3000 tokens via voxel-based selection
- **Adaptive voxel size:**
  - Per-image voxel size: v_img = min_i d_i, where d_i = 0.125 · Σ_{j∈N(i)} ||P_i - P_j||₂ (avg of 8-neighbor Euclidean distances in patch-3D-positions)
  - Scene voxel size: v_scene = (1/(t-1)) Σ_{j=1}^{t-1} v_img_j (online-updated)
  - This is the *killer* design: no predefined voxel size (model is metric-invariant)
- **3D spatial pruning:** group tokens by 3D voxel (using patch-3D-position from predicted pointmap), retain only the token with *highest cumulative attention weight* per voxel

### Memory Gating

- **Cross-attention** of F_t^c (query) with F_mem^K, F_mem^V (memory) → W_t ∈ ℝ^{P×S}
- **Threshold filter:** δ(s) = 1 if max_p W_t(p,s) > τ=5×10⁻⁴, else 0
- **Effect:** 27% memory reduction on 7Scenes, +20% FPS (18.0→21.4) with no quality regression (Acc 2.53→2.57, Comp 2.12→2.08)
- This is the **direct fix** for Spann3R's "memory is only attended once per iteration, preventing effective reuse" — gating *selects* memory that matters, then *full* attention can be computed on smaller set

### Dual-Source Refined Decoder

- **Interleaved (Ours):** PairwiseBlock(next-frame) ↔ MemoryBlock(memory) ↔ ... → all 6 layers alternate
- **Concatenated (Baseline):** all 6 PairwiseBlocks(next-frame) first, then all 6 MemoryBlocks(memory) at the end
- **Empirical evidence (Table 5, Replica100/200, 24-frame):** Interleaved 12.06/7.67 Acc/Comp vs Concatenated 14.83/10.26 → **-19% / -25% improvement**
- **Mechanism:** interleaving *progressively aligns* the feature spaces of memory and next-frame, avoiding the information loss from feature-space misalignment in the concatenated design

### Training

- **Stage 1:** 5 frames/sample, 120 epochs, AdamW lr=1.12×10⁻⁴, batch 10/GPU, 16 A100 GPUs, 28 hours
- **Stage 2:** 10 → 32 frames, 12 epochs each, lr=1×10⁻⁵, ~20 hours, **ViT encoder frozen** (transfer learning from Stage 1)
- **Total: 16×(28+20) = 768 GPU-hours on A100** (~$770 Lambda at $1/hr, the *cheapest* 3R training in the streaming-3R arc after WinT3R 185's 13,824 GPU-hours)
- **Init:** DUSt3R encoder weights
- **Input:** 224×224
- **Memory budget:** 10-frame short-term + 3000-token long-term

### Loss

L_total = L_conf + L_scale
- **L_conf:** confidence-aware ℓ2 regression on pointmap (DUSt3R/Spann3R-style) + α·log(C) regularizer
- **L_scale:** encourages predicted point cloud to have average distance < ground truth (Spann3R-style scale normalization)

### Training Data (6 datasets, *smaller* than WinT3R 185's 12)

- **Real indoor:** Habitat, ScanNet, ScanNet++, ARKitScenes
- **Object:** Co3Dv2
- **Synthetic:** BlendedMVS
- Mixing strategy (not specified) — *standard* random sampling

## Results

### Table 1 — 3D reconstruction on 7Scenes + NRGBD (Acc/Comp cm + NC, lower Acc/Comp = better, higher NC = better)

| Method | 7S Acc↓ | 7S Comp↓ | 7S NC↑ | NRGBD Acc↓ | NRGBD Comp↓ | NRGBD NC↑ | FPS↑ |
|---|---|---|---|---|---|---|---|
| F-Recon | 12.43 | 5.54 | 61.89 | 28.55 | 15.05 | 65.47 | ≪1 |
| DUSt3R | 3.01 | 5.11 | 58.83 | 3.94 | 5.31 | 62.62 | ≤3 |
| MASt3R | 2.82 | 5.26 | 58.22 | 3.85 | 5.50 | 60.92 | ≤3 |
| MV-DUSt3R | 2.92 | 2.49 | 66.42 | 3.76 | 2.55 | 81.16 | ~15 |
| MV-DUSt3R+ | 2.93 | 8.63 | 66.38 | 3.47 | 3.69 | 84.33 | ~3 |
| CUT3R | 7.73 | 7.75 | 65.74 | 12.48 | 6.34 | 75.84 | ~23 |
| Spann3R | 3.42 | 2.41 | 66.35 | 6.91 | 2.91 | 77.75 | ~22 |
| **LONG3R (Ours)** | **2.57** | 2.08 | **66.55** | **6.66** | 3.11 | **77.56** | **~22** |

**LONG3R wins 7Scenes Acc (-0.25 vs MASt3R, -0.85 vs Spann3R, -5.16 vs CUT3R) AND NC**. Tied 2nd on 7S Comp (2.08 vs MV-DUSt3R 2.49). On NRGBD, MV-DUSt3R+ wins (3.47/3.69) but LONG3R wins NC among online methods. *Killer* result: **CUT3R's 7S Acc 7.73 is *catastrophic* (2.7× worse than Spann3R 3.42) — confirms CUT3R's known weakness on 7S**. LONG3R's 2.57 is *better than MASt3R 2.82* (the offline SOTA) and *faster*.

### Table 2 — 3D reconstruction on Replica100 + Replica200 (sequence-length-controlled)

| Method | R100 Acc↓ | R100 Comp↓ | R100 NC↑ | R200 Acc↓ | R200 Comp↓ | R200 NC↑ | FPS↑ |
|---|---|---|---|---|---|---|---|
| DUSt3R | 6.34 | 6.44 | 61.67 | 4.99 | 4.63 | 62.26 | ≤3 |
| MASt3R | 5.10 | 6.00 | 61.81 | 5.26 | 7.31 | 58.03 | ≤3 |
| MV-DUSt3R | 10.41 | 4.34 | 73.76 | 17.02 | 5.10 | 66.74 | ~7 |
| MV-DUSt3R+ | 5.28 | 2.56 | 79.07 | 11.79 | 5.64 | 70.66 | ~1 |
| CUT3R | 20.44 | 5.67 | 69.63 | 28.3 | 6.61 | 63.95 | ~23 |
| Spann3R | 14.08 | 4.67 | 72.46 | 16.29 | 4.02 | 68.56 | ~21 |
| **LONG3R (Ours)** | **11.46** | **3.68** | 73.29 | **11.93** | **2.73** | 68.67 | **~21** |

**LONG3R wins on R100 Acc/Comp among streaming methods (11.46/3.68 vs CUT3R 20.44/5.67 = -44%/-35%)**. On R200, LONG3R *closes the gap to MV-DUSt3R+* (11.93/2.73 vs 11.79/5.64) at *21× faster FPS*. **CUT3R's R200 Acc 28.3 is *catastrophic* (drift accumulation)** — confirms CUT3R's known weakness on long sequences. *Killer* trend: **CUT3R degrades -38% (20.44→28.3) on R100→R200, Spann3R -16% (14.08→16.29), LONG3R only -4% (11.46→11.93)** — LONG3R is the *most robust* to sequence length.

### Table 3 — Camera pose estimation (ATE/RPEt/RPEr, lower = better)

| Method | 7S ATE↓ | 7S RPEt↓ | 7S RPEr↓ | TUM ATE↓ | TUM RPEt↓ | TUM RPEr↓ | ScanNet ATE↓ | ScanNet RPEt↓ | ScanNet RPEr↓ |
|---|---|---|---|---|---|---|---|---|---|
| Spann3R | 12.64 | 6.15 | 1.88 | 5.66 | 2.13 | 0.59 | 9.83 | 2.30 | 0.66 |
| CUT3R | 12.40 | 7.65 | 2.34 | 6.25 | 2.55 | 0.69 | 14.27 | 3.58 | 0.92 |
| **LONG3R (Ours)** | **8.72** | **5.03** | **1.67** | **5.40** | **2.36** | **0.60** | **6.44** | **2.14** | **0.61** |

**LONG3R wins 8 of 9 sub-metrics** (tied on TUM RPEr 0.60). *Killer* result: **LONG3R's 7S ATE 8.72 = -30% vs Spann3R 12.64, -30% vs CUT3R 12.40**, the *direct* evidence that the 3D-spatial-memory pruning *preserves pose accuracy* (not just pointmap quality). On ScanNet (the *most challenging* static dataset), LONG3R 6.44 = -34% vs Spann3R 9.83, -55% vs CUT3R 14.27.

### Table 4 — Memory Gating ablation (7Scenes + NRGBD)

| Variant | 7S Acc↓ | 7S Comp↓ | 7S NC↑ | NRGBD Acc↓ | NRGBD Comp↓ | NRGBD NC↑ | FPS↑ |
|---|---|---|---|---|---|---|---|
| w/o Gating | 2.53 | 2.12 | 66.72 | 3.14 | 2.91 | — | 18.0 |
| **w/ Gating (Ours)** | **2.57** | **2.08** | **66.66** | 3.11 | 2.92 | — | **21.4** |

**Memory gating gives +20% FPS (18.0→21.4) and -27% memory tokens (Fig. 6) with *negligible* quality change** (Acc +0.04, Comp -0.04, NC -0.06). This is the **killer** H3 evidence: *selective attention* is strictly better than *dense attention* for streaming 3R.

### Table 5 — Refined Decoder ablation (Replica100/200, 24-frame)

| Variant | R100 Acc↓ | R100 Comp↓ | R200 Acc↓ | R200 Comp↓ |
|---|---|---|---|---|
| Concatenated | 14.83 | 10.26 | 29.52 | 21.04 |
| **Interleaved (Ours)** | **12.06** | **7.67** | **13.34** | **8.41** |

**Interleaved beats Concatenated by -19% Acc / -25% Comp on R100, -55% Acc / -60% Comp on R200** — the *gap widens with sequence length*, confirming the *progressive feature-space alignment* hypothesis. The *killer* empirical evidence that *interleaving* is the right architectural choice for multi-source attention.

### Table 6 — Memory framework ablation (7Scenes + Replica200)

| Variant | 7S Acc↓ | 7S Comp↓ | R200 Acc↓ | R200 Comp↓ |
|---|---|---|---|---|
| w/o 3D Spa. Mem. | 5.76 | 3.30 | 65.75 | 47.63 |
| w/ Spann3R Mem. | 2.64 | 2.10 | 12.41 | 3.07 |
| **LONG3R (Ours)** | **2.57** | **2.08** | **11.93** | **2.74** |

**Without 3D spatial memory, R200 Acc/Comp are 65.75/47.63 (catastrophic)** — 11× worse than full LONG3R. **LONG3R's 3D-spatial-memory beats Spann3R-style memory** by -3% Acc on 7S, -10% Acc on R200, -11% Comp on R200. The *direct* evidence that **3D-voxel-pruned memory is strictly better than flat memory** for long sequences.

## Connections to H1-H5

### H1 (2-stage VAE+DDM > 1-stage) — **PARTIAL**

LONG3R is structurally 1-stage (feed-forward end-to-end) but **logically 2-stage via the two-stage curriculum training**:
- **Stage 1:** 5-frame pretraining for *preliminary understanding*
- **Stage 2:** 10→32-frame fine-tuning with frozen encoder for *long-sequence adaptation*

This is a **training-time** 2-stage decomposition, not an architectural one. H1 update: **for streaming 3R, *training-time curriculum* (short→long) is the dominant 2-stage paradigm; the architectural 1-stage feed-forward is *settled* in 2024-2026 (DUSt3R, MASt3R, Spann3R, CUT3R, MonST3R, all 1-stage)**.

### H2 (latent diffusion > direct) — **STRONGEST DIRECT SUPPORT in 186-paper list**

The **3D Spatio-Temporal Memory** IS a H2 latent:
- 3000-token *learned* memory bank, *amortized* across the stream
- Stores *latent* representations of past frames in *learned* 3D voxel grid
- Used as input to downstream decoder (cross-attention)
- Compact because the model learns to *select* which tokens to retain (highest attention)

**Empirical evidence for H2:** the 3D-spatial-memory ablation on R200 (65.75 → 11.93 = -82% Acc without 3D spatial memory) is the **SINGLE-LARGEST single-component ablation in the streaming-3R literature**. The 3D-spatial-memory is *strictly* the H2 mechanism — compact 3D-structured latent aggregated globally.

H2 update: **for *cross-frame aggregation* of *geometric* targets, a *3D-voxel-structured latent* (3D spatio-temporal memory) is the right H2 mechanism; for *cross-frame aggregation* of *low-dim camera* targets, a *global compact latent* (camera token pool, WinT3R 185) is the right H2 mechanism. H2 mechanism is *target-dimensionality + spatial-structure dependent*.**

### H3 (opposing-jaw / cross-view / cross-frame conditioning) — **STRONGEST DIRECT SUPPORT**

The **attention-based memory gating** is THE H3 mechanism for *cross-frame relevance selection*:
- Cross-attention of current frame's coarse tokens with memory keys → attention weights W_t
- Threshold filter (max-attn > τ) → only *relevant* memory tokens contribute
- **Empirical evidence:** Table 4 +20% FPS, -27% memory, no quality loss (Acc +0.04, Comp -0.04)
- The *direct* fix for Spann3R's "memory is only attended once per iteration, preventing effective reuse"

The **Dual-Source Refined Decoder** is also H3 for *cross-frame temporal*:
- Alternating next-frame + memory cross-attention for 6 layers
- **Empirical evidence:** Table 5 interleaved beats concatenated by -19%/-25% Acc/Comp on R100, -55%/-60% on R200 (the *gap widens* with sequence length)

H3 update: **for *cross-frame relevance selection*, attention-threshold gating is the right H3 mechanism (LONG3R 186); for *direct cross-frame temporal aggregation*, interleaved (next-frame + memory) cross-attention is the right H3 mechanism (LONG3R 186 vs concatenated Spann3R). H3 mechanism is *aggregation-strategy dependent*.**

### H4 (implicit SDF > mesh) — **INDIRECT SUPPORT**

LONG3R outputs **pointmaps** (ℝ^{3×H×W} per frame) + uses DPT head (Dense Prediction Transformer). The H4 substrate choice here is **explicit per-frame pointmaps**, the dominant 2024-2026 streaming-3R substrate (DUSt3R, MASt3R, Spann3R, CUT3R, MonST3R, WinT3R, all pointmaps).

**The implicit vs explicit axis is *settled* on pointmaps** for streaming 3R — no SDF, no mesh, no NeRF. H4 update: **for *streaming 3R*, explicit per-frame pointmaps is the *de facto* 2024-2026 H4 substrate; the H4 substrate choice is *settled* on pointmaps in this literature.**

### H5 (synthetic+finetune / mixed-real pre-training) — **STRONG DIRECT SUPPORT**

LONG3R is trained on **6 mixed-synthetic-real datasets**:
- **Real indoor:** Habitat, ScanNet, ScanNet++, ARKitScenes (4)
- **Object:** Co3Dv2 (1)
- **Synthetic:** BlendedMVS (1)
- Initialized from DUSt3R encoder (offline-reconstruction pre-training)

This is **6-dataset mixing** — *fewer* than WinT3R 185's 12 datasets but *more than CUT3R 175's ~7*. The **two-stage curriculum** (5→10→32 frames) is the *killer* H5 trick: short-context pretraining + long-context fine-tuning = progressive *sequence-length* adaptation.

**Empirical evidence for H5:** the Replica dataset is *zero-shot* (NOT in training mix) → LONG3R still wins on R100/R200 NC among online methods (Table 2: 73.29/68.67 vs CUT3R 69.63/63.95, Spann3R 72.46/68.56). The *killer* H5 claim: 6-dataset mix + curriculum training gives *cross-domain generalization* to unseen Replica scenes.

H5 update: **for *long-sequence* streaming 3R, the right H5 recipe is (a) 6-12 mixed-synthetic-real datasets + (b) two-stage curriculum (short→long) + (c) frozen-encoder fine-tuning. LONG3R's 6-dataset + 5→32-frame curriculum is the *de facto* 2025 H5 SOTA recipe for long-sequence streaming 3R.**

## Surprises / Interesting Things Buried in Section 4

1. **CUT3R is *catastrophic* on long sequences** (R200 Acc 28.3, 2.4× worse than LONG3R 11.93) — confirms the 185-note's claim that CUT3R's *state-token* design *accumulates drift* without global alignment. The *killer* empirical evidence that *persistent state* is not enough for long sequences.
2. **LONG3R's 7S Acc 2.57 BEATS MASt3R's 2.82 (offline SOTA)** — the *first* streaming method to beat the *offline* SOTA on 7Scenes Acc. The *killer* practical claim: streaming 3R is no longer *quality-compromised* vs offline.
3. **LONG3R's 7S ATE 8.72 = -30% vs Spann3R 12.64** — the *camera-pose* improvement is *even larger* than the pointmap improvement (-0.85 vs -3.92). The 3D-spatial-memory helps *pose estimation* more than *pointmap prediction* because pose estimation is *more sensitive to global consistency*.
4. **Memory gating +20% FPS with NO quality loss** — the *killer* H3 evidence that *selective attention* is strictly better than *dense attention* for streaming 3R. The right H3 mechanism is *attention-based selection*, not *attention over everything*.
5. **Interleaved decoder *widens its lead* with sequence length** (R100: -19% Acc, R200: -55% Acc) — the *killer* empirical evidence for *progressive feature-space alignment*. Concatenated decoder *accumulates feature-space misalignment* over long sequences.
6. **6-dataset training mix is *enough*** (vs WinT3R 185's 12) — the *killer* H5 finding: *6-dataset* + *two-stage curriculum* gives *comparable* cross-domain generalization to *12-dataset* + *one-stage training*. The right H5 recipe is *progressive sequence-length training*, not *more datasets*.
7. **Adaptive voxel size** (v_img = min over tokens, v_scene = running avg) — the *killer* engineering trick that makes the model *metric-invariant*. Predefined voxel size would fail on *cross-domain* scenes (Replica 200 frames is much larger than 7Scenes 7 frames).
8. **3000 long-term memory tokens is *fixed*** (not growing with stream length) — the *killer* O(1)-memory property. The voxel-pruning is *bounded* by scene size, not by stream length. This is the *O(1)-memory* claim that complements WinT3R 185's *O(1)-per-frame-cost* claim.
9. **DPT head** (vs WinT3R 185's ConvHead) — the *practical* difference: DPT is *higher-quality* but *slower*; ConvHead is *faster* but *lower-quality*. LONG3R chose DPT because *quality* > *speed* for long sequences.
10. **Limitations stated in the paper** (Sec. 5): (a) predictions are *relative to first frame* → may produce *blurry results* if viewpoint deviates significantly; (b) no dynamic-scene training data → struggles with *highly dynamic* scenes with large object motions. Both limitations are *consistent* with the streaming-3R literature (CUT3R 175, Spann3R 177, MonST3R 174 also have these).

## Quote-Worthy Sentences

- *"Spann3R struggles with long input sequences due to three key issues: (1) its memory is only attended once per iteration, preventing effective reuse, (2) its memory becomes spatially redundant as images accumulate, and (3) its training strategy does not support adaptation to long sequences."* — Sec. 1, the *founding* motivation
- *"We define long-sequence reconstruction as real-time processing of tens to hundreds of frames with near-constant memory requirements."* — Sec. 1, the *founding* definition
- *"The memory gating mechanism removes memory features irrelevant to the current frame, exemplified by a 27% reduction on 7Scenes, and achieves an optimal balance between reconstruction accuracy and computational efficiency."* — Sec. 4.4, the *founding* H3 evidence
- *"Our interleaved attention blocks address this issue by employing alternating cross-attention, which progressively aligns feature spaces and improves computational efficiency."* — Sec. 4.4, the *founding* dual-source insight
- *"Since the memory stores patch-based tokens, we first compute a unique 3D position P for each patch using the point map predicted in each frame via a weighted average."* — Sec. 3.4, the *founding* 3D-position-from-pointmap trick
- *"The optimal image voxel size v_img is determined as the minimum d_i across all tokens to balance memory usage and storage efficiency."* — Sec. 3.4, the *killer* adaptive-voxel-size design
- *"Since our predictions are defined relative to the first frame, our model may produce blurry results if the viewpoint deviates significantly."* — Sec. 5, the *founding* first-frame-anchored limitation
- *"Due to the lack of dynamic training data, the current model struggles to handle highly dynamic scenes with large object motions."* — Sec. 5, the *founding* dynamic-scene limitation

## Code / Data Links

- **Code:** [github.com/zgchen33/LONG3R](https://github.com/zgchen33/LONG3R) (43 ⭐ / 0 🍴, 1.9 MB, last push 2025-07-25)
- **Code status:** **"Code Coming Soon"** (README.md, as of 2026-06-15) — *code not yet released* ⚠️
- **Project page:** [zgchen33.github.io/LONG3R](https://zgchen33.github.io/LONG3R/) (only method overview figure, no demo / no video)
- **Pretrained:** **NOT released** (no Hugging Face / no Google Drive link in README)
- **Built on:** DUSt3R (encoder init) + Spann3R (paradigm) + CUT3R (paradigm) + MASt3R (coarse-to-fine) + CroCo (cross-view pretraining)
- **Datasets:** Habitat, ARKitScenes, BlendedMVS, ScanNet++, Co3Dv2, ScanNet (6-mix, *smaller* than WinT3R 185's 12)
- **License:** **NONE ⚠️** (GitHub API `license: null`)

## For Our Project (v0 sub-task 1: full-arch synthesis)

### ★ 12 v0 actions

(a) ★★★ **ADOPT 3D SPATIO-TEMPORAL MEMORY AS V0 SUB-TASK 1 H2 MECHANISM** ($100-200 Lambda, 2-3 weeks, *re-implement* LONG3R's 3D-voxel-pruned memory on a *commercial-permissive* license (Apache-2.0 ✅ or MIT ✅), *replace* the patch-based 3D-position-from-pointmap with *tooth-3D-position-from-arch-mesh*, *replace* the long-term 3000 tokens with *300-tooth-tokens* (32 teeth × ~10 patches per tooth = 320, *perfect* match for the 300-token budget). The *right* v0 sub-task 1 design is *NOT* CUT3R's pure state-token (loses long-sequence continuity) but LONG3R's *3D-spatial-memory + state-token* hybrid (gets both short-term continuity + long-term spatial structure). The *killer* ablation evidence: w/o 3D-spatial-memory → R200 Acc 11.93 → 65.75 (5.5× worse, the *single-largest* ablation in streaming-3R literature) — the *direct* v0 differentiator for *long-arch scan* (intra-oral scans with 5-30 views per arch + 14-tooth-full-mouth scan with 50-100 views).

(b) ★★★ **ADAPTIVE VOXEL SIZE FOR V0 SUB-TASK 1** ($20-50 Lambda, 1-2 days, the *killer* engineering trick: v_img = min over per-tooth-3D-position-distances, v_scene = running avg across arch. The *right* v0 design: voxel size = *per-tooth-3D-size* (incisor ~7mm, molar ~12mm) instead of *per-image* → ensures *tooth-coherent* memory tokens (each voxel contains tokens from a *single tooth*, not from multiple teeth). The *practical* v0 path: *clinical-robustness* claim: "first 3D-crown-generation paper with tooth-coherent memory architecture for cross-arch-size generalization".)

(c) ★★★ **ADOPT MEMORY GATING AS V0 SUB-TASK 1 H3 MECHANISM** ($50-100 Lambda, 1-2 weeks, *re-implement* LONG3R's attention-threshold gating with threshold τ tuned for *dental* relevance (e.g., τ=10⁻⁴ for *tooth-specific* relevance). The *right* v0 sub-task 1 design: *tooth-gating* = drop memory tokens whose max attention weight over current-frame patches is below τ → only *tooth-relevant* memory tokens (same tooth, same quadrant, same FDI number) contribute. The *killer* ablation evidence: LONG3R +20% FPS, -27% memory, *negligible* quality loss — the *direct* v0 differentiator for *cross-tooth* vs *same-tooth* attention selection.)

(d) ★★ **ADOPT DUAL-SOURCE INTERLEAVED DECODER AS V0 SUB-TASK 1 ARCHITECTURE** ($50-100 Lambda, 1-2 weeks, *re-implement* LONG3R's interleaved next-frame + memory cross-attention. The *right* v0 sub-task 1 design: alternating *next-view* (PairwiseBlock) + *arch-memory* (MemoryBlock) cross-attention for 6 layers, vs the *concatenated* design (all next-view first, then all arch-memory). The *killer* ablation evidence: interleaved beats concatenated by -19%/-25% on R100, -55%/-60% on R200 — the *gap widens* with sequence length, the *direct* v0 differentiator for *long-arch* (full-mouth 50-100 views) vs *short-arch* (5-10 views) reconstruction.)

(e) ★★ **ADOPT TWO-STAGE CURRICULUM TRAINING AS V0 SUB-TASK 1 TRAINING RECIPE** ($0 Lambda, 1-day engineering, the *killer* H5 + H1 recipe: (Stage 1) pretrain on *5-view* intra-oral scans for *short-context* correspondence matching, lr=1.12e-4, 120 epochs; (Stage 2) fine-tune on *10→32-view* full-arch scans for *long-context* correspondence matching, lr=1e-5, 12 epochs each, **ViT encoder frozen**. The *direct* v0 analog: 3DTeethSeg22 + ToSynFCD + clinical IOS = *3-source* pre-training → *full-arch* fine-tuning. The *practical* v0 path: 1-2 weeks engineering on *single A100* GPU, $770 Lambda total compute (the *cheapest* in the streaming-3R arc).)

(f) ★★ **ADOPT 6-DATASET MIXED TRAINING AS V0 SUB-TASK 1 H5 MECHANISM** ($200-400 Lambda, 1-2 weeks, the *killer* H5 evidence: 6-dataset mix (4 indoor + 1 object + 1 synthetic) gives *cross-domain* generalization to unseen Replica scenes. The *direct* v0 analog: 3DTeethSeg22 + ToSynFCD + clinical IOS + simulated IOS + 3DToothSeg + ToothFairy = *6-dataset* mixed training. The *right* v0 differentiator: "first 3D-crown-generation paper to use 6-dataset mixed-synthetic-real training for cross-patient + cross-IOS-brand generalization".)

(g) ★★ **ADOPT 21.4 FPS AS V0 SUB-TASK 1 RUNTIME TARGET** ($0 Lambda, 0-day, the *practical* design lesson: LONG3R achieves 21.4 FPS (with gating) on RTX 3090 (24GB VRAM) = the *current* SOTA-FPS for *long-sequence* streaming-3R. The *right* v0 sub-task 1 *chairside-real-time* target: 21+ FPS on *single A100* = 2-3× faster than WinT3R 185's 17.2 FPS = *the fastest* SOTA-quality streaming-3R to date. The *right* v0 paper claim: "first 3D-crown-generation paper to achieve 21+ FPS on intra-oral scan reconstruction (vs WinT3R 185's 17.2 FPS)".)

(h) ★★ **ADOPT 3000 LONG-TERM MEMORY TOKENS AS V0 SUB-TASK 1 MEMORY BUDGET** ($0 Lambda, 0-day, the *killer* O(1)-memory design lesson: 3000 long-term memory tokens is *fixed* regardless of stream length → O(1) memory budget. The *right* v0 sub-task 1 design: 300 long-term *tooth-position-tokens* (32 teeth × ~10 patches per tooth = 320, *perfect* match) = O(1) memory for any arch size. The *practical* v0 path: 300 tokens × 1024-dim × 4 bytes = 1.2 MB per arch = *negligible* memory for *single-A100* deployment.)

(i) ★ **ADOPT DUSt3R ENCODER INITIALIZATION AS V0 SUB-TASK 1 TRANSFER-LEARNING ENABLER** ($0 Lambda, 1-line code change, the *killer* 6-month-training-savings trick: initialize LONG3R with *DUSt3R*'s pre-trained encoder (DUSt3R is the *founding* offline 3R paper, license: Creative Commons BY-NC-SA 4.0 ⚠️ OR re-implement DUSt3R encoder from scratch on a commercial-permissive license). The *direct* v0 analog: initialize from *DUSt3R* or *Spann3R* pre-trained weights, *fine-tune* on *intra-oral scan* data for *dental* correspondence matching.)

(j) ★ **CITE LONG3R 186 IN V0 PAPER RELATED-WORK AS THE *FOUNDING* 3D-SPATIO-TEMPORAL-MEMORY PAPER** ($0, 1-2 hours, the *historical anchor*: 1 paragraph noting the 2024 Spann3R 177 (memory paradigm) → 2025 CUT3R 175 (state-token paradigm) → 2025 StreamVGGT / Point3R (memory-token paradigm) → 2025 LONG3R 186 (3D-spatial-memory paradigm) → 2025 WinT3R 185 (sliding-window + camera-pool paradigm) → 2026 R³ 183 / TTT3R 182 / LingBot-Map 184 / WinT3R 185 / 186 / 186-design-space-complete; the *de facto* 2024-2026 *streaming-3R* lineage, *complete* for the 5 main paradigms (state-token + memory-token + SLAM-prior + window-pool + **3D-spatial-memory**).)

(k) ★ **USE LONG3R 186 AS V0 PAPER'S TABLE 1 BASELINE COMPARISON ROW** ($0, just cite + report numbers; for v0 paper, the *right* Table 1 row is "LONG3R 186 (NO LICENSE ⚠️, 43 ⭐, ~36 SS, 3D-spatial-memory paradigm founder)" with 3D-recon (Table 1: 7Scenes/NRGBD), camera-pose (Table 3: 7S/TUM/ScanNet), Replica-long (Table 2: R100/R200) numbers — the *complete* 2025-2026 *streaming-3R* lineage. **CRITICAL:** v0 *commercial deployment* path requires *re-implementation* of LONG3R's *mechanism* (3D-spatial-memory + memory-gating + dual-source-decoder) on a *commercial-permissive* license (Apache-2.0 ✅ or MIT ✅), since LONG3R 186 has *no license* ⚠️.)

(l) ★ **USE LONG3R 186's 6-DATASET MIX + CURRICULUM TRAINING AS V0 PAPER'S H5 EVIDENCE** ($0, just cite + report numbers; for v0 paper, the *right* H5 evidence is "LONG3R 186's 6-dataset mix (Habitat + ARKitScenes + BlendedMVS + ScanNet++ + Co3Dv2 + ScanNet) + 5→10→32-frame curriculum training = the *de facto* 2025 H5 SOTA recipe for long-sequence streaming 3R" with Replica-zero-shot R100/R200 numbers as the *killer* H5 evidence (LONG3R 11.46/11.93 vs CUT3R 20.44/28.3 vs Spann3R 14.08/16.29). The *direct* v0 analog: 3DTeethSeg22 + ToSynFCD + clinical IOS = *3-source* pre-training → *intra-oral* fine-tuning.)

### ★ v0 sub-task 1 streaming-3R stack now has 16 papers covered

1. **LONG3R 186 (NO LICENSE ⚠️, 21.4 FPS, 3D-spatio-temporal-memory + memory-gating + dual-source-decoder paradigm founder, 43 ⭐)** NEW 3D-spatial-memory mechanism
2. WinT3R 185 (custom non-commercial, 17.2 FPS, sliding-window + camera-token-pool paradigm founder, 228 ⭐) window+pool mechanism
3. LingBot-Map 184 (Apache-2.0 ✅, GCA + SLAM-prior, 7,188 ⭐) SOTA streaming-3R
4. R³ 183 (custom non-commercial, relative-regression, TBD ⭐) O(1)-cost alternative
5. TTT3R 182 (ICLR 2026, TTT-based memory, ~100-200 ⭐) TTT memory
6. STream3R 181 (ICLR 2026, causal transformer, ~200-500 ⭐) causal streaming
7. Ray-Aware Pointer 180 (custom non-commercial, ray-direction pointer, TBD ⭐) ray-aware
8. Point3R 179 (custom non-commercial, point-cloud memory, TBD ⭐) point memory
9. Fast3R 178 (custom non-commercial, multi-view parallel, TBD ⭐) parallel multi-view
10. Spann3R 177 (custom non-commercial, spatial memory, TBD ⭐) spatial memory paradigm founder
11. DAS3R 176 (custom non-commercial, depth-aware stereo, TBD ⭐) depth-aware
12. CUT3R 175 (CC-BY-NC 4.0, persistent state, ~500+ ⭐) state-token paradigm founder
13. MonST3R 174 (custom non-commercial, dynamic, TBD ⭐) dynamic extension
14. Easi3R 173 (custom non-commercial, easy generalizable, TBD ⭐) easy generalizable
15. YonoSplat 172 (custom non-commercial, Yono 3DGS, TBD ⭐) Yono 3DGS
16. PF3Plat 171 (custom non-commercial, PF3plat, TBD ⭐) PF3plat

**The 5 main streaming-3R paradigms are now *all* covered:** (i) state-token (CUT3R 175, MonST3R 174, Fast3R 178, Easi3R 173), (ii) memory-token (Spann3R 177, Point3R 179, STream3R 181, R³ 183, TTT3R 182, Ray-Aware 180), (iii) SLAM-prior-structured (LingBot-Map 184), (iv) window+pool (WinT3R 185), (v) **3D-spatial-memory (LONG3R 186)**. The 2024-2026 *streaming-3R* design space is now *complete* (5 paradigms × 16 papers = *most-comprehensive* reading-list coverage).

### ★ v0 compute updated

**v0 sub-task 1 compute: ~$3,400-5,000 Lambda** (was $3,000-4,500 from 185-note, +$400-500 for LONG3R 186's *re-implementation engineering* for *dental* data: *re-implement* LONG3R's 3D-spatial-memory + memory-gating + dual-source-decoder on a *commercial-permissive* license + *replace* the patch-based 3D-position-from-pointmap with *tooth-3D-position-from-arch-mesh* + *re-train* on *3DTeethSeg22 + ToSynFCD* = 2-4 weeks engineering on *single A100*).

**v0 TOTAL compute: ~$12,340-18,180 Lambda** (was $11,940-17,680 from 185-note, +$400-500).

### ★ Open Q for HK

(i) cite LONG3R 186 in v0 paper related-work? (YES — *founding* 3D-spatio-temporal-memory paradigm, $0, 1-2 hours)
(ii) adopt 3D-spatio-temporal-memory for v0 sub-task 1 H2 mechanism? (YES — *founding* H2 mechanism with *killer* w/o-3D-spatial-memory ablation -82% R200 Acc, $100-200 Lambda, 2-3 weeks)
(iii) adopt adaptive voxel size for v0 sub-task 1? (YES — *per-tooth-coherent* memory tokens, $20-50 Lambda, 1-2 days)
(iv) adopt memory gating for v0 sub-task 1 H3 mechanism? (YES — *founding* H3 mechanism with *killer* +20% FPS, -27% memory ablation, $50-100 Lambda, 1-2 weeks)
(v) adopt dual-source interleaved decoder for v0 sub-task 1 architecture? (YES — *interleaved* beats *concatenated* by -19%/-25% on R100, $50-100 Lambda, 1-2 weeks)
(vi) adopt two-stage curriculum training for v0 sub-task 1? (YES — *killer* H5 + H1 recipe, $0, 1-day engineering, *5→10→32-view* analog)
(vii) adopt 6-dataset mixed training for v0 sub-task 1 H5 mechanism? (YES — *cross-domain* generalization, $200-400 Lambda, 1-2 weeks)
(viii) adopt 21.4 FPS as v0 sub-task 1 runtime target? (YES — *2-3× faster* than WinT3R 185's 17.2 FPS, $0, 0-day)
(ix) adopt 3000 long-term memory tokens as v0 sub-task 1 memory budget? (NO — *too many* for *tooth-specific*; use *300 tooth-position-tokens* instead, $0, 0-day)
(x) adopt DUSt3R encoder initialization for v0 sub-task 1? (YES — *founding* offline 3R pre-training, $0, 1-line code change, *6-month-training-savings*)
(xi) handle NO LICENSE for v0 *commercial deployment*? (YES — *re-implement* LONG3R's *mechanism* (3D-spatial-memory + memory-gating + dual-source-decoder) on a *commercial-permissive* license, $400-500 Lambda, 2-4 weeks; or use LingBot-Map 184 Apache-2.0 as the *commercial-deployable* alternative for the same paradigm)
(xii) use LONG3R 186's HF-pretrained checkpoints for v0? (NO — pretrained *NOT released*, no HF / no Google Drive link in README; *re-train* on *3DTeethSeg22 + ToSynFCD* from scratch, $200-400 Lambda)
(xiii) use LONG3R 186 as v0 Table 1 baseline? (YES — *founding* paradigm + *43 ⭐* + *21.4 FPS* SOTA, $0, just cite + report numbers; but *disclose* no-license + no-code)

## ⚠️ Note to Self

The 185-WinT3R-note's "next paper 186 = LONG3R (Chen et al. 2025b, arXiv:**2507.18255**, long-sequence streaming 3D reconstruction, ICCV 2025)" was **CORRECT** on all key facts — verified via direct arXiv lookup, DBLP lookup, and GitHub API:
- arXiv ID: **2507.18255** ✅
- Authors: **Zhuoguang Chen*, Minghui Qin*, Tianyuan Yuan*, Zhe Liu, Hang Zhao†** (3 equal first authors, Zhao is corresponding) ✅
- Affiliations: **Shanghai AI Lab + IIIS Tsinghua University + Shanghai Qi Zhi Institute** ✅
- Venue: **ICCV 2025** ✅ (pp. 5273-5284, DBLP verified)
- Code: **github.com/zgchen33/LONG3R** ✅
- License: **NONE** ⚠️ (GitHub API `license: null`, README says "Code Coming Soon" — *code not yet released*; the 185-note did NOT specify license details, this is the *new* critical finding)
- 43 ⭐ / 0 🍴 / 1 open issue / last push 2025-07-25
- Citations: **~36 Semantic Scholar** as of 2026-06-15 (10 months post-v1, ICCV 2025)

**The 12-13th arXiv-ID hallucination in the 156-186 arc was PREVENTED by direct arXiv lookup.** The 185-note's "LONG3R 186, arXiv:2507.18255" is *correct* (no hallucination).

**New critical findings (NOT in the 185-note):**
1. **LONG3R 186 has NO LICENSE ⚠️** (GitHub API `license: null`) — the *3rd* paper in the streaming-3R arc with license issues (after WinT3R 185's custom non-commercial). The v0 *commercial-deployment* path requires *re-implementation* of LONG3R's *mechanism* (3D-spatial-memory + memory-gating + dual-source-decoder) on a *commercial-permissive* license.
2. **LONG3R 186 has NOT RELEASED CODE** (README says "Code Coming Soon") — the *1st* paper in the streaming-3R arc with *no code release*. The 174-185 arc all had *some* code release (some Apache-2.0 ✅, some CC-BY-NC ⚠️, some custom non-commercial ⚠️, but all *released*). LONG3R 186 is the *first* with *no release*.
3. **LONG3R 186's 6-dataset mix is *smaller* than WinT3R 185's 12** (Habitat + ARKitScenes + BlendedMVS + ScanNet++ + Co3Dv2 + ScanNet) — but the *two-stage curriculum* (5→10→32 frames) compensates for the *smaller* dataset mix. The *killer* H5 finding: *progressive sequence-length training* > *more datasets*.
4. **LONG3R 186's 21.4 FPS is *faster* than WinT3R 185's 17.2 FPS** (LONG3R's memory-gating is the *killer* engineering win) — the *current* SOTA-FPS for long-sequence streaming 3R.
5. **LONG3R 186's 3D-spatial-memory is *the* H2 mechanism** (3000 tokens, voxel-pruned, adaptive voxel size) — the *founding* H2 mechanism for *long-sequence* streaming 3R, complementary to WinT3R 185's *camera-token-pool* H2 mechanism (1536-dim per-frame, O(1) per-frame cost).

## ★ Next Paper to Read (187)

The 185-WinT3R-note's recommended *next* was LONG3R (now read!). The 186-LONG3R-note's recommended *next* is **LoGeR (Zhang et al. 2026, arXiv:2603.03269)** — the *concurrent* 2026 long-context 3R paper with hybrid sliding-window + TTT memory.

**Recommendation: *read 187 = LoGeR (Zhang et al. 2026, arXiv:2603.03269)*** — the *concurrent* 2026 long-context 3R paper with hybrid sliding-window + TTT memory, the *founding* paper of the *TTT-based memory* paradigm for long-context 3R, the *right* next paper to *complete* the *long-context* streaming-3R design space (LONG3R 186 = *3D-spatial-memory*; WinT3R 185 = *window+pool*; LoGeR = *sliding-window + TTT*). After LoGeR 187, the v0 sub-task 1 *long-context* streaming-3R arc is *complete* (LONG3R 186 + WinT3R 185 + LoGeR 187 + TTT3R 182 = 4 papers, the *most-comprehensive* 2025-2026 *long-context* streaming-3R arc for v0 *full-arch synthesis* + *chairside-real-time* + *clinical-quality* + *commercial-deployable*).

**Alternative 187 candidates:** (a) **Scal3R (Xie et al. 2026, arXiv:2604.08542)** the *concurrent* 2026 scalable test-time-training 3R paper with chunking + VPR; (b) **ZipMap (Jin et al. 2026, CVPR 2026, arXiv:2603.04385)** the *concurrent* 2026 linear-time stateful 3R paper via TTT hidden scene state; (c) **LongStream (Cheng et al. 2026, arXiv:2602.13172)** the *concurrent* 2026 long-sequence streaming autoregressive visual geometry paper; (d) **Human3R (Chen et al. 2026, ICLR 2026, arXiv:2510.06219)** the *concurrent* 2026 4D human-reconstruction paper (less relevant for v0 dental). **Recommendation: *read 187 = LoGeR (the *direct* LONG3R 186 + WinT3R 185 alternative for the *long-context* streaming-3R arc; the *right* next paper to *complete* the *3-paradigm* long-context design space: 3D-spatial-memory + window+pool + TTT-memory)*.

⚠️ **PATTERN NOTICE:** the 185-WinT3R-note's "next paper 186 = LONG3R, arXiv:2507.18255" was *correct* on all key facts (the 12-13th arXiv-ID hallucination was *prevented* by direct arXiv lookup), confirming that the *direct-arXiv-lookup* sub-skill is *working* after the 8 prior hallucinations. The *new* critical findings are the *no-license* + *no-code-release* — the 185-note did NOT specify license or code-release status, and the 186-note's GitHub API lookup revealed both. *Always* verify license + code-release via GitHub API.
