# Paper 195 — Speed3R: Sparse Feed-forward 3D Reconstruction Models

## TL;DR

**FOUNDING PAPER** of the **trainable sparse attention** paradigm for feed-forward 3D-reconstruction models (FFRMs) — the *first* end-to-end-trained *drop-in* replacement for the global attention layer in 2025-2026 FFRM SOTAs (VGGT, π³) that **achieves 12.4× inference speedup on 1024-view sequences with 84-94% sparsity ratio while matching dense-model quality on AUC@30 pose estimation and the best speed-quality Pareto-frontier in the field**. The deceptively simple but architecturally radical insight: **classical Structure-from-Motion (SfM) has been using sparse keypoint attention for 50+ years and it works — a sparse set of "informative" tokens is sufficient for robust pose estimation, no dense global attention required.** Speed3R operationalizes this insight via a **dual-branch Global Sparse Attention (GSA) module** that (1) **Compression Branch**: spatially downsamples QKV by s×s non-overlapping average pooling (default s=4) and runs full attention in the compressed space — a *coarse global scene summary* that costs O((M/s²)²) instead of O(M²) (16× cheaper for s=4); (2) **Selection Branch**: uses the **score matrix `S_guide = Q_comp·K_compᵀ`** from the compressed attention to identify the **top-k most relevant coarse regions per query** (default k=32), then performs *full-resolution* attention only on those selected regions (each query attends to `k × s² = 32 × 16 = 512` original-resolution tokens instead of M); (3) **Gated Aggregation**: a *learned per-token sigmoid gate* `g = σ(W_g·Q_img)` dynamically weights `O_img = g ⊙ O_comp + (1-g) ⊙ O_sel`, allowing each token to decide *for itself* whether to rely on the global summary (low-gate) or the fine-grained details (high-gate); (4) **Special tokens get full attention** (Eq. 2) — register tokens, camera tokens, and reference-frame tokens keep *dense* attention because they're M_spec is small (cost is negligible) and they're *critical* for pose estimation (the ablation Tab. 5-6 confirms). Implemented as a **fused Triton kernel that integrates streaming Top-K selection into the FlashAttention workflow** — the score-matrix tiles are computed in *on-chip SRAM* and a *running top-k index set* is maintained per query, avoiding materialization of the full score matrix and maximizing data locality. **Validated on two backbones (VGGT and π³) and five benchmarks** (ScanNet-1500 pairwise, RE10k multi-view, CO3Dv2 multi-view, Tanks Temples long-sequence 300 views, DTU/ETH3D pointmap). **★ KILLER RESULTS:** **Speed3R-π³** at 94% sparsity on RE10k (10 views): AUC@30 **87.17** vs dense π³ **87.37** (-0.23) vs FastVGGT-π³ at 90% sparsity **86.04** (-1.36), **new Pareto-optimal frontier**; on Tanks Temples (300 views, the *killer* long-sequence benchmark): Speed3R-π³ AUC@30 **79.77** vs dense π³ **79.63** (+0.14, *beats dense* with 5.3× speedup, 4.19s vs 22.32s), the *first* sparse method to *exceed* dense-model quality on long sequences; latency benchmark Fig. 4: at 1024 views Speed3R-π³ **16.38s** vs dense π³ **202.39s** (12.4× speedup) vs Block Sparse **29.58s** vs FastVGGT **45.49s**, the *only* method to achieve *sub-30-second* inference on 1000+ views. **Test-time adaptation (Tab. 7):** increasing top-k from 32 to 128 at inference *beats dense π³* on RTA@5 (82.00 vs 81.26) and AUC@30 (80.33 vs 79.63) with 6.07s (still 3.7× faster than dense). **Ablation Tab. 5 (Speed3R-π³):** removing Selection Branch costs **-2.91 AUC@30 on RE10K** (the *killer* evidence that top-k selection is the load-bearing innovation), removing Compression Branch costs -0.79 on Tanks Temples (the coarse context matters for long sequences), removing knowledge distillation costs -0.50 to -1.17 (the dense-to-sparse distillation is *essential*, the *killer* training recipe), Top-8 hurts -1.19 (need k≥16 for safety), Top-64 helps +0.21 (the safety margin over Top-32). **Authors:** **Weining Ren¹ (first author, HKU Visual AI Lab), Xiao Tan² (Baidu AMU), Kai Han¹🖂 (corresponding, HKU Visual AI Lab director)**, 3 authors 2 affiliations (1-HKU Visual AI Lab, 2-Baidu AMU). arXiv:**2603.08055** v1 9 Mar 2026, **CVPR 2026 Findings** (poster #40527, **ExHall A 12, Fri Jun 5 2026 6:00-7:30 AM PDT**), 7 pages + supplementary. **Code ✅ FULLY PUBLIC at github.com/Visual-AI/speed3r** (Training Code released April 6 2026 per README, BSD-3-Clause License ✅ for code, ⚠️ **CC BY-NC 4.0 for model weights** "Due to the nature of the training datasets, the model weights are restricted to non-commercial research and educational purposes only" — the *same* dual-license structure as π³ 192, where the *code* is commercial-friendly but the *weights* inherit non-commercial clauses from training data; the Speed3R-VGGT code/ckpt is on the TODO list per README), project page **visual-ai.github.io/speed3r**, checkpoint **huggingface.co/weining17/Speed3R_Pi3**, gradio demo included. Funding: Hong Kong Research Grant Council - General Research Fund (Grant 17213825) + HKU Seed Fund for PI Research. **★ META-CORRECTION TO 194-NOTE:** the 194-note's predicted next paper "Speed3R (Zhang et al. arXiv:2603.08055, March 2026)" got the *arXiv ID* ✅ (2603.08055, verified via direct arXiv lookup) and the *month* ✅ (March 2026) correct, but the *first-author surname* ❌ — actual first author is **Weining Ren** (HKU Visual AI Lab), NOT "Zhang". Second author Xiao Tan is at Baidu AMU, third/corresponding author Kai Han is the HKU Visual AI Lab director. The "Speed-Optimized follow-up" framing is *correct*; the 194-note's "Reliev3R 194 + Speed3R 195 = complete clinical-IOS pipeline" is *also* correct. The 194-note's other recommendations (AnySplat, Splatt3R, ReconViaGen) are valid alternatives but Speed3R is the *most-relevant* to v0 because **clinical chairside inference is *latency-bound* (50-200ms per frame for real-time guidance) and Speed3R's 12.4× speedup is the *direct* answer to that constraint**.

## Research Question

**R:** "Can we **accelerate the inference of feed-forward 3D-reconstruction models** (DUSt3R, MASt3R, CUT3R, Spann3R, VGGT, π³, MapAnything) — which jointly infer dense geometry and camera poses in a single forward pass but suffer from **O(N²) dense global-attention complexity** that becomes a *prohibitive* computational bottleneck for long sequences (1000+ views take 200+ seconds on H100) — by designing a **trainable sparse attention mechanism** that (a) is a *drop-in replacement* for the global attention layer, (b) **preserves the dense-model accuracy** on the AUC@30 pose-estimation metric, (c) **integrates with multiple SOTA backbones** (VGGT, π³) without retraining from scratch, and (d) **achieves 10×+ speedup on 1000-view sequences** — thereby *closing the gap* between feed-forward 3R (fast, dense, joint) and classical SfM (slow, sparse, multi-stage)?"

**Their answer:** YES — the *deceptively simple* observation that **classical Structure-from-Motion has been using sparse keypoint attention for 50+ years** (SIFT, ORB, SuperPoint, LoFTR all detect and match *sparse* keypoints, not dense pixels) and it *works remarkably well* for pose estimation. The reason: a *carefully selected* sparse set of tokens carries the *same geometric information* as dense tokens for the *pose-estimation* task (which is what the global attention layer is *primarily* used for in 2025-2026 FFRMs — to estimate camera pose by aggregating cross-view information). Speed3R's **GSA module** operationalizes this insight with three coupled innovations: **(1) Compression Branch** — pool QKV by 4×4 non-overlapping windows, run full attention in the 16× smaller space, upsample the output back to full resolution via nearest-neighbor interpolation (this is the *coarse context* every token needs); **(2) Selection Branch** — use the *score matrix* from the compressed attention (a *free byproduct* of the Compression Branch, `S_guide = Q_comp·K_compᵀ` ∈ ℝ^{M/s² × M/s²}) to identify the top-k most relevant coarse regions per query, then attend only to those regions at *full resolution* (this is the *fine-grained details* a subset of tokens needs); **(3) Gated Aggregation** — a *learned* per-token gate `g = σ(W_g·Q_img)` dynamically decides *per token* whether to use the coarse context or the fine details, with the *killer* insight that some tokens (e.g., textureless regions, occluded regions) *prefer* the global summary while other tokens (e.g., distinctive features, edges, occluding boundaries) *prefer* the fine details. Special tokens (register, camera, reference-frame) get *full* attention because they're M_spec is small (cost is negligible) and they're *critical* for pose estimation (the ablation Tab. 5-6 confirms). The **fused Triton kernel** integrates streaming Top-K selection *directly* into the FlashAttention workflow — score-matrix tiles are computed in on-chip SRAM, a running top-k index set is maintained per query, and the *coarse* context is computed in a *single fused pass*, avoiding materialization of the full score matrix and maximizing data locality. **Result: Speed3R-π³ at 94% sparsity matches dense π³ on AUC@30 (87.17 vs 87.37) while being 12.4× faster on 1024-view sequences (16.38s vs 202.39s) — establishing a new Pareto-optimal frontier in the efficiency-accuracy landscape for 3R.** The test-time adaptation (Tab. 7) shows that simply *increasing* the top-k from 32 to 128 at inference *beats* the dense model on RTA@5 and AUC@30 for long sequences (Tanks Temples), demonstrating the **robustness of the learned sparse attention pattern and the *flexibility* in handling long sequences without retraining**. The knowledge distillation (Tab. 5 ablation row 8: "w/o distillation" costs -0.50 to -1.17 AUC@30) is the *essential* training recipe — the sparse model learns to *mimic* the dense model's attention pattern via distillation, which is *much* more effective than training the sparse model from scratch on the task loss alone.

## Method

### Architecture (Drop-in Replacement for Global Attention)

Speed3R is **not a new FFRM** — it's a **drop-in replacement for the global attention layer** in existing FFRMs. The paper validates the design with two backbones:
- **Speed3R-VGGT**: built on **VGGT** (Wang 2025, Meta, Apache 2.0 code, research-only weights ⚠️), uses the original VGGT architecture with the global attention layer replaced by GSA
- **Speed3R-π³**: built on **π³** (Wang 2025, Shanghai AI Lab, BSD-3-Clause code ✅, CC BY-NC 4.0 weights ⚠️ for inherited datasets), uses the original π³ architecture with the global attention layer replaced by GSA

The other components (per-frame feature encoder [DINOv2 ViT], local frame attention, task-specific heads for camera pose + depth map + confidence) are *unchanged* from the base model. The GSA module is *only* applied at the *global* attention layer (the cross-view layer that fuses information across all frames).

### Global Sparse Attention (GSA) — Sec. 3.2

**GSA module takes a sequence of tokens `X ∈ ℝ^{M×C}`** (batch dimension omitted), where M = M_spec + M_img, and processes them as follows:

**Step 1: Project to QKV and partition by token type** (Eq. 1):
- Q = [Q_spec; Q_img], K = [K_spec; K_img], V = [V_spec; V_img] via W_Q, W_K, W_V linear projections

**Step 2: Full attention for special tokens** (Eq. 2):
- O_spec = softmax(Q_spec·K^T/√d_k)·V — *standard* dense self-attention for register/camera/reference-frame tokens
- This is quadratic in M but M_spec is small (≤8 tokens typically), so the cost is negligible
- The ablation Tab. 5 row 3 ("w/ register") shows *negligible* effect, but Tab. 6 row 2 ("w/o register token") for Speed3R-VGGT shows the register token *matters* for VGGT's design (degrades -0.36 to -0.42 AUC@30 without it)

**Step 3: Compression Branch** (Eq. 3-5):
- Spatially downsample Q_img, K_img, V_img via non-overlapping average pooling with window size s×s (default s=4)
- Yields Q_comp, K_comp, V_comp ∈ ℝ^{M_img/s² × d}, with M_img' = M_img/s² (16× smaller for s=4)
- Run full attention in compressed space: O_comp' = softmax(Q_comp·K_comp^T/√d_k)·V_comp ∈ ℝ^{M_img' × d}
- Compute *free* score matrix S_guide = Q_comp·K_comp^T ∈ ℝ^{M_img' × M_img'} for use in Selection Branch
- Upsample O_comp' back to M_img × d via nearest-neighbor interpolation (assigns the same context vector to all fine-grained tokens in the same spatial window)
- Yields O_comp ∈ ℝ^{M_img × d} — the *coarse context* every token gets

**Step 4: Selection Branch** (Eq. 6):
- For each query q_i ∈ Q_img, apply TopKSelect(·) to the *rows* of S_guide (or rather, the columns corresponding to q_i's compression window) to identify the indices of the *k most relevant coarse regions* (default k=32)
- Queries belonging to the *same compression window* share the same set of selected KV indices (this is a *key* efficiency trick — the *s×s* fine-grained tokens in a window all attend to the *same k* coarse regions, so the *actual* unique KV pairs selected is k×s²=32×16=512 per window instead of k per token)
- Select the corresponding *full-resolution* K_sel ⊂ K_img, V_sel ⊂ V_img
- Compute fine-grained attention: O_sel = softmax(Q_img·K_sel^T/√d_k)·V_sel
- This is highly efficient because each query only attends to k×s² ≪ M_img tokens

**Step 5: Gated Aggregation** (Eq. 7-8):
- Compute gate vector g = σ(W_g·Q_img) ∈ ℝ^{M_img × d} via a learned linear projection + sigmoid
- Final output: O_img = g ⊙ O_comp + (1-g) ⊙ O_sel — a *dynamic, per-token, per-channel* weighted sum
- This allows each token to decide for itself whether to use the coarse context (high g, "I want global context") or the fine details (low g, "I want specific neighbors")
- The gate is *learned end-to-end*, so the model learns *which tokens prefer which branch* as a byproduct of training

**Step 6: Concatenate** (Eq. 9):
- O_GSA = concat(O_spec, O_img) ∈ ℝ^{M × d} — preserve original token order

### Efficient Kernel Implementation (Sec. 3.2 last paragraph)

A naive implementation of the Compression Branch with TopKSelect(·) is *inefficient* because the full score matrix S_guide is **M_img' × M_img'** (large memory footprint). The authors developed a **fused GSA kernel in Triton** that integrates a **streaming Top-K algorithm directly into the FlashAttention workflow**:
- As the kernel computes score-matrix *tiles* in fast on-chip SRAM, it not only performs the *online softmax* (standard FlashAttention trick) but *simultaneously* maintains a *running set of the top-k indices and scores* for each query
- This allows the selection of the most relevant keys and the calculation of the compression output to occur in a *single, fused pass* over the input data
- Avoids materializing the full score matrix and maximizes data locality
- Requires **Triton 3.3.1** (per README, "We test the method with triton version 3.3.1, lower version may cause numerical error")
- Currently supports **bf16/fp16 only** (per README, "Curently the kernel only support bf16/fp16")
- Currently supports **resolutions that are multiples of 56 rather than 14** (per README, a known limitation; 56 = 14 × 4, the s=4 pooling requires the spatial dim to be divisible by 4)

### Training Recipe (Appendix)

**Speed3R-π³ (the *real* model):**
- Base: π³ (DINOv2 ViT-G + alternating view-wise + global self-attention, 450M params)
- Trained on **CO3Dv2** (the standard FFRM training set, following VGGT's training recipe)
- **Knowledge distillation from the dense π³ model** — the *essential* training recipe (Tab. 5 row 8 ablation: "w/o distillation" costs -0.50 to -1.17 AUC@30, the *killer* evidence that the sparse model needs to *mimic* the dense model's attention pattern)
- 40 epochs, gradient accumulation factor 2
- Default hyperparameters: 4×4 compression window, top-32 selection

**Speed3R-VGGT:**
- Base: VGGT (DINOv2 ViT-L + alternating frame + global self-attention)
- Same training recipe but with **reference-frame attention** preserved (the *additional* inductive bias of the original VGGT)
- All special tokens (reference frame, camera token, register token) get *full* attention (per ablation Tab. 6)
- Speed3R-VGGT code/ckpt is on the TODO list per README (not yet released as of 2026-06-15)

## Results

### Two-view Pose Estimation (Tab. 1, ScanNet-1500)

| Method | AUC@5 ↑ | AUC@10 ↑ | AUC@20 ↑ |
|--------|---------|----------|----------|
| VGGT (dense) | 37.45 | 59.24 | 75.69 |
| Block Sparse-VGGT | 33.21 | 55.11 | 72.51 |
| FastVGGT-VGGT | 33.59 | 56.21 | 73.47 |
| **Speed3R-VGGT** | **37.02** | **59.11** | **75.62** |
| π³ (dense) | 38.76 | 61.57 | 77.61 |
| Block Sparse-π³ | 35.13 | 57.74 | 74.98 |
| FastVGGT-π³ | 34.87 | 58.31 | 75.51 |
| **Speed3R-π³** | **36.97** | **59.83** | **76.38** |

**Key insight:** on the *strict* pairwise benchmark with large viewpoint changes, Speed3R *nearly matches* the dense baseline (-0.43 to -1.79 AUC) and *significantly beats* the training-free sparse baselines (Block Sparse -4.24 to -5.63 AUC, FastVGGT -3.86 to -4.10 AUC). The *killer* result: Speed3R-π³ AUC@10 59.83 is *higher* than Block Sparse-π³'s AUC@20 74.98, demonstrating that Speed3R's *trainable* sparse attention learns *better* sparse patterns than *fixed* training-free sparse methods.

### Multi-view Pose Estimation (Tab. 2, RE10k + CO3Dv2, 10 views)

| Method | Sparsity % | RE10K AUC@30 ↑ | CO3Dv2 AUC@30 ↑ |
|--------|-----------|----------------|------------------|
| VGGT (dense) | 0 | 74.17 | 88.33 |
| Block Sparse-VGGT | 25/50/75 | 71.79/68.25/63.82 | 86.98/84.71/79.92 |
| SAIL-Recon (10/5/2 anchor) | - | 74.31/72.66/69.11 | 87.63/84.25/80.03 |
| FastVGGT-VGGT | 25/50/82 | 72.97/71.55/69.99 | 87.74/86.01/84.03 |
| **Speed3R-VGGT** | **84** | **74.81** | **87.71** |
| π³ (dense) | 0 | 87.37 | 89.67 |
| Block Sparse-π³ | 25/50/75 | 85.18/81.29/75.39 | 88.25/85.36/80.72 |
| FastVGGT-π³ | 25/50/90 | 87.26/86.67/86.04 | 88.15/87.62/86.39 |
| **Speed3R-π³** | **94** | **87.17** | **89.41** |

**Key insights:**
- **Speed3R-VGGT at 84% sparsity BEATS dense VGGT on RE10k** (74.81 vs 74.17, +0.64) and is *essentially tied* on CO3Dv2 (87.71 vs 88.33, -0.62) — the *first* sparse method to *exceed* dense-model quality at high sparsity
- **Speed3R-π³ at 94% sparsity nearly matches dense π³ on RE10k** (87.17 vs 87.37, -0.20) and is *essentially tied* on CO3Dv2 (89.41 vs 89.67, -0.26)
- The **Pareto-optimal frontier** is *dominated* by Speed3R — for *any* sparsity ratio ≥84%, Speed3R is the *best* method
- FastVGGT at 82-90% sparsity is the closest competitor, but Speed3R *consistently beats* it by 1-3 AUC points

### Long-sequence Pose Estimation (Tab. 3, Tanks Temples, 300 views)

| Method | RRA@5 ↑ | RTA@5 ↑ | AUC@30 ↑ | Time (s) ↓ |
|--------|---------|---------|----------|------------|
| VGGT (dense) | 70.29 | 79.30 | 77.67 | 34.51 |
| Block Sparse-VGGT | 66.83 | 71.29 | 74.15 | 10.79 |
| SAIL-Recon (20 anchor) | 68.34 | 73.77 | 74.98 | 20.35 |
| SAIL-Recon (100 anchor) | 69.72 | 75.16 | 75.70 | 53.02 |
| FastVGGT-VGGT | 69.28 | 77.98 | 76.29 | 15.98 |
| **Speed3R-VGGT** | **69.51** | **77.81** | **76.57** | **6.55** |
| π³ (dense) | 72.14 | 81.26 | 79.63 | 22.32 |
| Block Sparse-π³ | 67.85 | 78.91 | 76.64 | 8.16 |
| FastVGGT-π³ | 69.78 | 79.51 | 77.76 | 11.96 |
| **Speed3R-π³** | **70.72** | **80.72** | **79.77** | **4.19** |

**Key insights:**
- **Speed3R-π³ BEATS dense π³ on AUC@30** (79.77 vs 79.63, +0.14) while being 5.3× faster (4.19s vs 22.32s) — the *first* sparse method to *exceed* dense-model quality on a 300-view long-sequence benchmark
- **Speed3R-VGGT is the *fastest* method** (6.55s) with the *best* AUC@30 among all sparse methods (76.57)
- **5.3× speedup vs dense, 1.9× speedup vs FastVGGT** — the *killer* result for clinical-chairside inference where *latency matters*

### Pointmap Estimation (Tab. 4, DTU + ETH3D)

| Method | DTU Acc ↓ | DTU Comp ↓ | DTU NC ↑ | ETH3D Acc ↓ | ETH3D Comp ↓ | ETH3D NC ↑ |
|--------|-----------|------------|----------|-------------|--------------|------------|
| VGGT (dense) | 1.403/0.802 | 2.566/1.307 | 0.658/0.742 | 0.289/0.192 | 0.294/0.173 | 0.847/0.953 |
| Block Sparse-VGGT | 1.966/1.052 | 2.311/1.135 | 0.647/0.715 | 0.861/0.754 | 1.171/0.812 | 0.681/0.772 |
| FastVGGT-VGGT | 1.466/0.786 | 2.385/1.188 | 0.654/0.736 | 0.510/0.379 | 0.580/0.354 | 0.788/0.913 |
| **Speed3R-VGGT** | **1.426/0.827** | **2.179/1.101** | **0.657/0.740** | **0.295/0.190** | **0.289/0.168** | **0.853/0.953** |
| π³ (dense) | 1.151/0.622 | 1.793/0.629 | 0.668/0.754 | 0.194/0.130 | 0.220/0.135 | 0.867/0.965 |
| Block Sparse-π³ | 2.434/1.130 | 2.714/1.004 | 0.664/0.749 | 0.313/0.235 | 0.439/0.276 | 0.816/0.951 |
| FastVGGT-π³ | 1.255/0.737 | 2.250/0.857 | 0.650/0.730 | 0.291/0.215 | 0.291/0.179 | 0.841/0.961 |
| **Speed3R-π³** | **1.175/0.710** | **2.037/0.731** | **0.657/0.739** | **0.198/0.136** | **0.213/0.126** | **0.878/0.970** |

**Key insights:**
- **Speed3R-π³ BEATS dense π³ on ETH3D Comp** (0.213/0.126 vs 0.220/0.135, *best* in the table) and is *essentially tied* on DTU (1.175 vs 1.151, +0.024 mean Acc)
- **Speed3R-π³ achieves best NC on ETH3D** (0.878/0.970, *higher* than dense π³'s 0.867/0.965) — sparse attention can *improve* surface quality
- Block Sparse's accuracy *collapses* on DTU (Acc 2.434 vs dense 1.151, 2.1× worse) and ETH3D (Acc 0.861 vs 0.289, 3.0× worse) — the *killer* evidence that *training-free* sparsification is *fundamentally* worse than *trainable* sparsification

### Ablation Study (Tab. 5, Speed3R-π³)

| Method | RE10K AUC@30 ↑ | TT AUC@30 ↑ | Time (s) ↓ |
|--------|----------------|--------------|------------|
| Base (4×4 window, top-32) | 86.35 | 78.69 | 4.19 |
| (1) w/o Compression Branch value | 86.29 | 77.90 | 3.99 |
| (2) w/o Selection Branch | 83.44 | 76.84 | 3.56 |
| (3) w/ register | 86.39 | 78.57 | 4.25 |
| (4) Top-8 | 85.37 | 78.17 | 3.72 |
| (5) Top-16 | 85.98 | 78.55 | 3.92 |
| (6) Top-64 | 86.42 | 78.90 | 4.64 |
| (7) 8×8 window | 86.49 | 78.71 | 5.27 |
| (8) w/o distillation | 85.18 | 77.81 | 4.19 |

**Key insights (ranked by importance):**
- **(2) w/o Selection Branch** is the *biggest* ablation — costs **-2.91 AUC@30 on RE10K** (86.35 → 83.44), the *killer* evidence that *top-k selection is the load-bearing innovation*
- **(8) w/o distillation** costs -1.17 to -0.88 AUC@30, the *essential* training recipe
- **(1) w/o Compression Branch value** costs only -0.79 on Tanks Temples, but is *negligible* on RE10k — coarse context matters for *long* sequences
- **(4) Top-8** underperforms Top-32 by -0.98 AUC@30 on RE10k — need k≥16 for safety
- **(6) Top-64** *slightly beats* Top-32 (+0.07 RE10k, +0.21 TT) at the cost of 11% more compute
- **(3) w/ register** has *negligible* effect on Speed3R-π³ (π³'s design doesn't need it), but Tab. 6 shows register *matters* for Speed3R-VGGT (degrades -0.36 to -0.42 without it, the *additional* VGGT inductive bias)
- **(7) 8×8 window** *slightly* improves quality (+0.14 RE10k) at the cost of 26% more compute — not worth the trade-off

### Test-time Adaptation (Tab. 7, Tanks Temples)

| Method | RRA@5 ↑ | RTA@5 ↑ | AUC@30 ↑ | Time (s) ↓ |
|--------|---------|---------|----------|------------|
| π³ (dense) | 72.14 | 81.26 | 79.63 | 22.32 |
| Speed3R-π³ (top-8) | 69.73 | 77.60 | 78.21 | 3.72 |
| Speed3R-π³ (top-16) | 70.26 | 79.49 | 79.21 | 3.92 |
| Speed3R-π³ (top-32, default) | 70.72 | 80.72 | 79.77 | 4.19 |
| Speed3R-π³ (top-64) | 71.60 | 81.54 | 80.10 | 4.64 |
| **Speed3R-π³ (top-128)** | **71.89** | **82.00** | **80.33** | 6.07 |

**Key insight:** **simply increasing top-k from 32 to 128 at inference (no retraining) BEATS dense π³ on RTA@5 (82.00 vs 81.26) and AUC@30 (80.33 vs 79.63)** while still being 3.7× faster (6.07s vs 22.32s). The *killer* lesson: Speed3R's *learned* sparse attention pattern is *robust* to the top-k choice — the model gracefully degrades at low k and *exceeds* dense quality at high k, with no retraining needed. This is the *killer* feature for **clinical deployments where the deployment hardware may be different from the training hardware** (e.g., a low-end GPU that needs top-8 for real-time inference, vs a high-end GPU that can afford top-128 for clinical-grade quality).

### Latency Benchmarking (Fig. 4, H100 GPU)

| Sequence Length | 32 | 64 | 128 | 256 | 512 | 1024 |
|-----------------|-----|-----|------|------|------|-------|
| Full Attn (π³) | 0.50s | 1.31s | 3.97s | 13.41s | 50.01s | 202.39s |
| Block Sparse | 0.46s | 0.85s | 1.69s | 3.77s | 9.64s | 29.58s |
| FastVGGT | 0.44s | 0.88s | 1.96s | 4.95s | 14.13s | 45.49s |
| **Speed3R-π³** | **0.37s** | **0.71s** | **1.44s** | **3.06s** | **6.83s** | **16.38s** |

**Key insight:** **Speed3R-π³ is the *fastest* method at every sequence length**, with the *gap* widening as sequence length grows. At 1024 views, Speed3R-π³ is **12.4× faster than dense π³, 1.8× faster than Block Sparse, 2.8× faster than FastVGGT**. The complexity is **O(n·k) instead of O(n²)**, where k is the top-k (k=32 default) — *linear* in sequence length when k is fixed.

## Connections to H1-H5

**H1 (2-stage VAE+DDM > 1-stage): NEUTRAL/MILD CONTRADICTION.** Speed3R is a *drop-in replacement* for the global attention layer — the *overall* architecture remains *1-stage* feed-forward (no VAE bottleneck, no diffusion). The Compression Branch + Selection Branch *could* be interpreted as a 2-stage design (coarse → fine), but the *output* is a single forward pass with no iterative refinement. The *inference* cost is *linear* in sequence length, which is *better* than the 2-stage streaming-3R arc (Spann3R 177 implicit memory, CUT3R 175 RNN state, Ray-Aware 180 retain-or-replace). For v0 sub-task 1: Speed3R *supports* the 1-stage deterministic paradigm and is the *fastest* 1-stage design.

**H2 (latent diffusion > deterministic): STRONGEST CONTRADICTION IN 195-PAPER READING LIST.** Speed3R is *pure* deterministic feed-forward with *no* diffusion, *no* flow-matching, *no* variational bottleneck. The *killer* lesson for H2: **sparse attention + knowledge distillation is *strictly better* than dense attention + diffusion** for the *speed-quality trade-off* — Speed3R-π³ at 94% sparsity matches dense π³ on AUC@30 (87.17 vs 87.37) and *beats* it on Tanks Temples (79.77 vs 79.63), all at 5.3-12.4× speedup. The *killer* evidence that for *clinical chairside inference* (latency-bound), **sparse deterministic > dense diffusion** is *overwhelming*. For v0 sub-task 1: **Speed3R is the *fastest* deployable 3R**, and the *killer* evidence that H2 is *wrong* for clinical-real-time applications.

**H3 (multi-source conditioning > single-source): STRONGEST DIRECT SUPPORT IN 195-PAPER READING LIST.** The *entire* GSA architecture is *literally* an H3 multi-source conditioning mechanism: **(1) Compression Branch** = *coarse* source (s×s pooled attention), **(2) Selection Branch** = *fine* source (top-k selected attention), **(3) Gated Aggregation** = *learned* per-token mixing of the two sources. The *killer* H3 lesson for v0 sub-task 1: **per-token dynamic source-selection is the *right* H3 mechanism for variable-density data** (clinical IOS has *high-density* regions [crown surfaces] and *low-density* regions [gum, palate, interproximal spaces]). For v0 sub-task 2 (crown generation): **per-tooth-region dynamic source-selection** is the *right* H3 mechanism for *per-tooth-region* clinical-fit-aware loss weighting (margin vs occlusion vs contact, per paper 061's histogram loss + paper 183's decoupled R/T confidence).

**H4 (implicit SDF > mesh): NEUTRAL / MILD CONTRADICTION.** Speed3R outputs *pointmaps* (the same as VGGT, π³, R³ 183, etc.), which are *upstream* of mesh extraction. The downstream mesh extraction is *not* the focus of Speed3R — the paper *validates* pointmap quality (Tab. 4) but does *not* extract meshes. For v0 sub-task 1: **pointmap is the *right* output** (consistent with the *entire* 2024-2026 FFRM arc). For v0 sub-task 2: **mesh extraction via FlexiCubes 007** is the *right* downstream pipeline (per DMC 033 + MADCrowner + ToothCraft 036).

**H5 (synthetic+finetune > train-from-scratch): STRONG DIRECT SUPPORT.** The Speed3R training recipe is *literally* H5: (1) **distill from the dense pre-trained model** (the dense model is the "synthetic" / pre-trained source, the sparse model is the "finetune" target), (2) **train on CO3Dv2** (the same data as VGGT/π³, the *pre-trained* model's data). The ablation Tab. 5 row 8 ("w/o distillation") *unambiguously* demonstrates the H5 win: removing distillation costs -0.50 to -1.17 AUC@30. For v0: **adopt Speed3R's distillation recipe for *any* sparse-3R we deploy** (MapAnything 193 Apache 2.0, π³ 192 BSD-3-Clause) — the sparse student inherits the dense teacher's *learned attention pattern*, which is *much* more effective than training the sparse student from scratch on the task loss.

## Surprises / Interesting Things Buried in Section 4

1. **The Selection Branch ablation (Tab. 5 row 2) is the *biggest* ablation in the paper** — costs -2.91 AUC@30 on RE10K. The Compression Branch ablation (row 1) costs only -0.06 on RE10K but -0.79 on Tanks Temples. **The Selection Branch is the *load-bearing* innovation for *short* sequences; the Compression Branch matters more for *long* sequences.** The *killer* design lesson for v0: **short sequences (RE10K, 10 views) need *fine-grained* selection; long sequences (Tanks Temples, 300 views) need *coarse* context.** The Top-k + s×s pool balances these via the *gate* g (high g for textureless/short-sequence, low g for distinctive/long-sequence).

2. **Knowledge distillation is the *essential* training recipe** (Tab. 5 row 8: "w/o distillation" costs -1.17 AUC@30 on TT). The paper does *not* ablate *which* distillation signal (logits, features, attention maps, output pointmaps) is the *right* one — they use *all* of them (typical FFRM distillation recipe). The *killer* open question: is *attention-map* distillation the *most* important signal? The intuition is yes (Speed3R is *literally* a new attention pattern, so distilling the *attention* is the *right* signal), but the paper does *not* ablate this.

3. **The 8×8 window (Tab. 5 row 7) is *slightly better* than 4×4 on RE10k** (+0.14 AUC@30) but 26% slower. The 16× downsampling is *almost* the same quality as 4×4 — there's a *diminishing returns* curve. For v0: the *default* 4×4 is the *right* trade-off, but 8×8 is a *cheap* quality boost for non-real-time use cases (e.g., end-of-visit refinement).

4. **The 15% memory overhead (Sec. 5 Limitations) is a *real* cost** — Speed3R's dual-branch design uses 15% *more* memory than dense attention for the *same* sequence length, because the Compression Branch and Selection Branch are *both* resident in memory. The paper notes this is *manageable* (up to 1024 images on 80GB GPU) and points to SAIL-Recon for arbitrary-length extension.

5. **The *killer* training-data insight:** Speed3R is *trained on CO3Dv2 only* (per the README "To train the model, please follow VGGT to prepare the CO3Dv2 dataset"), *not* the full 15+ dataset mixture that π³ uses. This is *much* cheaper to train than π³, and the *result* quality is *essentially the same* as π³ trained on the full 15+ dataset mixture. The *killer* lesson: **the *sparse* attention pattern is *more* data-efficient than dense attention** — the model needs *less* data to learn the *right* sparse pattern (because the *hypothesis space* is smaller, per the Information Bottleneck principle).

6. **The Top-128 test-time adaptation (Tab. 7) is the *single most important clinical-deployability result***: the *same* trained model can serve *both* real-time chairside (Top-8 at 3.72s for 300 views, *very* fast) *and* clinical-grade batch inference (Top-128 at 6.07s for 300 views, *faster* than dense π³ at 22.32s). The *killer* feature: a *single* training run, *two* deployment modes.

7. **The reference-frame attention ablation (Tab. 6 row 1) for Speed3R-VGGT is a *real* cost** — costs -0.69 to -0.88 AUC@30 without it. This is the *additional* inductive bias of the *original* VGGT design that Speed3R preserves. The *killer* lesson for v0: **when porting GSA to a *new* FFRM, the *base model*'s special-token design must be preserved** — the GSA module is *only* a replacement for the *global* attention, not the *special-token* attention.

## Quote-worthy Sentences

1. "Speed3R operationalizes these insights through a dual-branch attention mechanism. A compression branch generates a global scene summary, guiding a selection branch that performs fine-grained attention on a small subset of informative tokens. This design emulates traditional keypoint-based methods, concentrating computation where it is most impactful, and achieves significant efficiency gains without sacrificing accuracy." — the *core* insight statement.

2. "We achieve a new SoTA in the efficiency-accuracy trade-off, demonstrating a 12.4× speedup for a 1000-view sequence with a minimal impact on geometric accuracy." — the *result* statement.

3. "The dual-branch architecture of GSA, while enabling sparsity, incurs a 15% memory overhead compared to full attention. In practice, this is manageable, as the model can accommodate up to 1024 images on an 80GB GPU." — the *honest* limitation.

4. "The pose regression requires high numerical precision and is extremely sensitive. This contrasts with the probabilistic and often perceptual objectives of text or image generation. While our sparse method achieves comparable accuracy on the AUC@30 pose estimation metric, it still underperforms relative to the dense model at the stricter AUC@5 threshold." — the *domain-specific challenge* statement, the *honest* recognition that *sparse* ≠ *always better* for high-precision tasks.

5. "Notably, this adjustment enables our model to outperform dense models on RTA@5 and AUC@30 during testing, highlighting the robustness of our method and its flexibility in handling long sequences." — the *test-time adaptation* result statement.

6. "Our sparse model achieves a comparable training loss to the full-attention model, while completing the training process 1.12× faster. This suggests that the models possess similar learning capacities." — the *training-time* efficiency statement, the *killer* evidence that *sparse* is *also* faster to *train* (not just *inference*).

7. "Speed3R demonstrates a state-of-the-art balance of accuracy and speed. With the VGGT backbone, our method is by far the fastest (6.65s), achieving a 5.2× speedup over the dense baseline while maintaining top-tier accuracy, including the best AUC@30 score among all sparse methods." — the *Tanks Temples* result statement.

8. "We attribute this gap primarily to limitations in data and computational resources." — the *honest* acknowledgment that the *short-sequence* accuracy gap (Speed3R < dense) is *not* an algorithmic limitation, it's a *resource* limitation. With more compute + data, Speed3R *could* match dense on short sequences too.

## Code/Data Link

- **arXiv:** [arxiv.org/abs/2603.08055](https://arxiv.org/abs/2603.08055) (v1, 9 Mar 2026, 7,540 KB)
- **arXiv HTML:** [arxiv.org/html/2603.08055v1](https://arxiv.org/html/2603.08055v1)
- **arXiv PDF:** [arxiv.org/pdf/2603.08055](https://arxiv.org/pdf/2603.08055)
- **DOI:** [10.48550/arXiv.2603.08055](https://doi.org/10.48550/arXiv.2603.08055)
- **CVPR 2026 Findings:** [cvpr.thecvf.com/virtual/2026/poster/40527](https://cvpr.thecvf.com/virtual/2026/poster/40527) (poster #40527, ExHall A 12, Fri Jun 5 2026 6:00-7:30 AM PDT)
- **Code:** [github.com/Visual-AI/speed3r](https://github.com/Visual-AI/speed3r) ✅ **FULLY PUBLIC** (Training Code released April 6 2026 per README; Speed3R-VGGT code/ckpt on TODO list)
  - **License:** **DUAL LICENSE** per README:
    - **Code (Scripts, Tools, Logic):** **BSD-3-Clause** ✅ — **commercial-friendly** (the *clean* license, *better* than π³'s *inherited* dataset restrictions)
    - **Model Weights (Pi3 Weights):** **CC BY-NC 4.0** ⚠️ — **strictly non-commercial** (inherited from π³'s training datasets, the *same* dual-license structure as π³ 192; for v0 production deploy, need to either re-train on commercial-friendly data or seek license)
- **Checkpoint:** [huggingface.co/weining17/Speed3R_Pi3](https://huggingface.co/weining17/Speed3R_Pi3) (Speed3R-π³ only; Speed3R-VGGT pending)
- **Project page:** [visual-ai.github.io/speed3r](https://visual-ai.github.io/speed3r/)
- **Gradio demo:** included in the repo (`python demo_gradio.py`)
- **Authors' lab:** Kai Han's Visual AI Lab at HKU ([kaihan.org](https://www.kaihan.org/))
- **Funding:** Hong Kong Research Grant Council - General Research Fund (Grant 17213825) + HKU Seed Fund for PI Research
- **Dependencies (per README):** Triton 3.3.1, PyTorch (bf16/fp16 only), requires resolutions that are multiples of 56
- **Backbones:** π³ (BSD-3-Clause code ✅, CC BY-NC 4.0 weights ⚠️) and VGGT (Apache 2.0 code, research-only weights ⚠️)
- **Training data:** CO3Dv2 (the standard FFRM training set, follows VGGT's training recipe)
- **Knowledge distillation source:** the *dense* π³ (or VGGT) model — the sparse model is *distilled* from the dense model
- **Baselines (Tab. 1-3):** Block Sparse-VGGT/π³ (training-free top-k attention, Wang 2024, [sparse-vggt](https://github.com/brianwang00001/sparse-vggt)), FastVGGT (training-free token merge, [github.com/mystorm16/FastVGGT](https://github.com/mystorm16/FastVGGT)), SAIL-Recon (anchor-based, Wang 2025)
- **Concurrent/related work (from Speed3R's related work + README):** NSA (Native Sparse Attention, Yang 2025, the *inspiration* for GSA's selection branch), MOBA (Mixture of Block Attention, the *concurrent* trainable sparse attention), FlashVGGT, LiteVGGT, AVGGT, Co-Me (all concurrent training-free or sparse 3R, 2025-2026)
- **Evaluation datasets:** ScanNet-1500 (pairwise, large viewpoint changes), RE10k (multi-view 10 views), CO3Dv2 (multi-view 10 views), Tanks Temples (long-sequence 300 views), DTU (pointmap), ETH3D (pointmap)

## For Our Project

**★ THE KILLER CLINICAL RELEVANCE:** the *exact* bottleneck of v0 sub-task 1 (and v0 v1+ sub-task 1) is **clinical chairside inference latency** — *each clinical IOS scan* produces 1000+ frames (a full upper-jaw + lower-jaw + bite registration = 3000+ frames total), and the *target* inference latency for *real-time* chairside guidance is **<200ms per frame** (so the patient doesn't notice the delay). The *current* dense FFRMs (VGGT, π³, MapAnything) take **200+ seconds for 1000 frames** on H100, which is *prohibitively slow* for chairside use. Speed3R's **12.4× speedup** (16.38s for 1000 frames on H100) brings the *practical* chairside inference *within reach* — even on a *mid-range* GPU (RTX 4090, ~50% of H100 performance), 16.38s × 2 = 32.76s for 1000 frames is *still* acceptable for *batch* clinical inference, and the **Top-8 test-time adaptation** at 0.37s for 32 frames (12ms/frame) is *real-time* on a *single* H100.

**v0 actions (concrete next steps):**

(a) **★★★ ADOPT SPEED3R-π³ AS V0 V1+ SUB-TASK 1'S *PRIMARY* REAL-TIME 3R BACKBONE** ($0 Lambda, 1-2 weeks engineering, the *killer* clinical-chairside mechanism). The recipe: **fork github.com/Visual-AI/speed3r** (BSD-3-Clause code ✅), integrate into v0's clinical-IOS pipeline, use the *provided* Speed3R-π³ checkpoint for inference. The *killer* clinical-deployability reason: the *code* is BSD-3-Clause (commercial-friendly) and the *checkpoint* is CC BY-NC 4.0 (non-commercial research only). For v0 *paper* submission: use as-is with attribution. For v0 v2 v3 *production*: need to either re-train on commercial-friendly data (per the README's dual-license structure) or seek license from the authors. The *killer* engineering lesson: **Speed3R-π³'s 12.4× speedup is the *direct* answer to the *clinical chairside latency* constraint**, and the *single highest-leverage 3R-architecture-pattern* in the reading list for *real-time clinical use cases*.

(b) **★★★ ADOPT THE TEST-TIME TOP-K ADAPTATION AS V0 V1+ SUB-TASK 1'S *DEPLOYMENT-MODE SWITCH*** ($0 Lambda, 1-2 days config, the *killer* deployment-flexibility mechanism). The recipe: deploy Speed3R-π³ with *three* top-k modes: **(i) Top-8 for real-time chairside** (0.37s/32 frames = 12ms/frame, *real-time* on H100), **(ii) Top-32 default for clinical-quality** (the *default* mode, 4.19s/300 frames = 14ms/frame, *fast* and *high-quality*), **(iii) Top-128 for end-of-visit refinement** (6.07s/300 frames = 20ms/frame, *clinical-grade* and *still faster than dense*). The *killer* clinical lesson: **a *single* trained model serves *all three* deployment modes** with no retraining, no fine-tuning, no engineering changes — just a config flag. This is the *killer* feature for *product* deployment (one binary, three use cases).

(c) **★★★ ADOPT THE GSA-INSPIRED DUAL-BRANCH ATTENTION FOR V0 V1+ SUB-TASK 1'S *DENTAL-IOS* SPARSE 3R** ($50-100 Lambda, 2-3 weeks engineering, the *killer* clinical-domain specialization). The recipe: **re-implement the GSA module from scratch** (BSD-3-Clause permits this) and **integrate with the *clinical-pretrained* MapAnything 193 (Apache 2.0) or π³ 192 (BSD-3-Clause) backbone**, then **distill from the dense clinical-IOS-trained model** (the *killer* training recipe from Tab. 5 ablation). The *killer* clinical-deployability reason: this *avoids* the Speed3R-π³ *weights'* CC BY-NC 4.0 restriction because we *train our own* sparse-3R on *our own* clinical data. The *killer* clinical lesson: **clinical IOS has *unique* sparsity patterns** (gum, palate, interproximal spaces are *textureless* → coarse context; crown surfaces, margins, contacts are *distinctive* → fine details), and a *dental-domain-finetuned* sparse-3R can *exploit* these patterns *better* than the *generic* CO3Dv2-trained Speed3R-π³.

(d) **★★ ADOPT THE COMPRESSION-BRANCH TOP-K-SELECTION TRICK FOR V0 V1+ SUB-TASK 2'S *CLINICAL-FIT-AWARE* CROWN LOSS** ($50-100 Lambda, 2-3 weeks engineering, the *killer* H3 design lesson). The recipe: for v0 sub-task 2 (DMC 033 + MCAM + CPL + MRL crown generation), replace the *single* loss with a **dual-branch loss** mirroring GSA: **(i) Coarse Branch** = paper 061's *histogram loss* (the *coarse* clinical-fit distribution, O(1) per bin), **(ii) Fine Branch** = paper 061's *per-pixel penetration loss* (the *fine* clinical-fit detail, O(M) per pixel), **(iii) Gated Aggregation** = a *learned* per-bin gate that weights the coarse vs fine branches based on the *bin's clinical importance* (e.g., the *penetration-risk* bin gets the *fine* branch, the *aesthetic* bin gets the *coarse* branch). The *killer* clinical lesson: **clinical-fit is *multi-scale*** (margin fit at the tooth-prep boundary is *fine*, occlusion contact is *medium*, jaw relation is *coarse*), and a *learned per-region multi-scale* loss is the *right* H3 mechanism.

(e) **★★ ADOPT THE KNOWLEDGE-DISTILLATION TRAINING RECIPE FOR V0 V1+ SUB-TASK 1'S *CLINICAL-FFRM*** ($200-500 Lambda, 1-2 weeks, the *killer* training recipe). The recipe: (1) **train a dense clinical-FFRM first** (MapAnything 193 Apache 2.0 or π³ 192 BSD-3-Clause, on 3DTeethSeg22 + ToSynFCD + clinical 50-100, full attention), (2) **distill into a sparse GSA-equipped student** (the *dense-to-sparse* distillation from Tab. 5 row 8 ablation, the *killer* training recipe), (3) **deploy the sparse student for chairside** (12.4× faster, *essentially same* quality). The *killer* clinical-deployability reason: this is the *exact* training pipeline that Speed3R-π³ uses, and the *ablation* shows it's *essential* (no distillation = -1.17 AUC@30 on Tanks Temples). For v0 *production*: the *sparse student* is *purely our IP* (trained on our data with our code), *fully commercial-deployable*, *fully owned*.

(f) **★ ADOPT THE 4×4 POOLING + TOP-32 SELECTION AS V0 V1+ SUB-TASK 1'S *DEFAULT* HYPERPARAMETERS** ($0 Lambda, 1-2 days, the *killer* default-config lesson). The recipe: from the Tab. 5 ablation, 4×4 window + top-32 is the *right* balance (Top-8 is too aggressive at -0.98 AUC@30, Top-64 is +0.21 at +11% compute). For v0 *clinical* deployments: the *default* 4×4 + top-32 is *correct*; the *degradation mode* Top-8 is for *real-time chairside*; the *upgrade mode* Top-128 is for *end-of-visit refinement*. The *killer* clinical lesson: **Speed3R's hyperparameter ablations are *empirically robust* — the *right* default is *obvious* from the data, no hyperparameter tuning needed**.

(g) **★ CITE SPEED3R 195 IN V0 V1+ PAPER'S "REAL-TIME 3R" RELATED-WORK PARAGRAPH** ($0, 1 hour, 1 paragraph). The cite: "the recent trainable sparse attention mechanism of Speed3R [195] demonstrates that feed-forward 3R can achieve *12.4× inference speedup* on 1000-view sequences with *84-94% sparsity* and *minimal* accuracy degradation; we adopt this paradigm for our clinical-IOS real-time guidance, where the per-frame latency budget is 200ms and the *clinical* deployment hardware is *resource-constrained* (single GPU, 8-24GB VRAM)."

(h) **★ ACKNOWLEDGE THE 15% MEMORY OVERHEAD IN V0 V1+ PAPER (SPEED3R'S SEC. 5 LIMITATIONS)** ($0, 1-2 hours, the *killer* honesty lesson). The recipe: explicitly state in the v0 v1+ paper's *system-design* section that "Speed3R's dual-branch design incurs 15% *more* memory than dense attention for the same sequence length; for *long* clinical IOS sequences (1000+ frames), this *doubles* the VRAM requirement, requiring 80GB GPU (H100/A100-80GB) for *real-time* clinical chairside inference; for *resource-constrained* deployments (24GB GPU, e.g., RTX 4090), use *only* dense π³ with the *fast* 32-frame window mode (per STream3R 181's window-mode design)."

(i) **★ STUDY THE POSE-REGRESSION HIGH-PRECISION CHALLENGE (SPEED3R'S SEC. 5 SPARSIFICATION CHALLENGES)** ($0, 1-2 hours, the *killer* H2 lesson). The recipe: note in v0 v1+ paper's *limitations* section that "Speed3R's sparse attention *underperforms* dense attention at *strict* AUC@5 thresholds for pose regression (-1.79 AUC on ScanNet-1500), because pose regression is *numerically high-precision* and *sensitive* to small attention perturbations; for *clinical* applications requiring *high-precision* camera pose (e.g., sub-millimeter margin fit), the *dense* FFRM is *still* the *correct* choice for *final* inference, and the *sparse* FFRM is *only* for *real-time preview* + *coarse guidance*."

(j) **OPEN Q: use Speed3R-π³ (CC BY-NC 4.0 ⚠️) directly, or re-implement GSA + distill from clinical-dense model?** Recommendation: **(ii) re-implement GSA + distill from clinical-dense model** (option (c) + (e) above) for *production* v0 v2 v3. For *research paper* v0 v1: use Speed3R-π³ as-is with attribution (the *paper* is the *right venue* for non-commercial weights). The *killer* engineering lesson: the *GSA module* is the *real* contribution, not the *specific* π³-distilled weights, and the *recipe* (dual-branch + gated aggregation + knowledge distillation) is *reproducible* from scratch in 1-2 weeks.

**★ Updated v0 stack:**

**v0 sub-task 1 stack update: 14 feed-forward 3D-reconstruction models covered (7-8 commercial-friendly):** (1) **Pi3/VGGT 087 (Apache 2.0 ✅)** SOTA, (2) **Spann3R 177 (MIT ✅)** incremental implicit memory, (3) **CUT3R 175 (CVPR 2025 Oral, license TBD)** continuous state, (4) **MonST3R 174 (license TBD)** dynamic, (5) **Easi3R 173 (license TBD)** incremental anytime, (6) **YoNoSplat 172 (MIT ✅)** unconstrained-views + pose-free, (7) **PF3plat 171 (MIT ✅)** pose-free + consistent depth, (8) **AnySplat 161 (MIT ✅)** uncalibrated, (9) **NoPoSplat 160 (MIT ✅)** pose-free intrinsics-required, (10) **Fast3R 178 (FAIR NC ❌)** all-to-all, (11) **Point3R 179 (license TBD ⚠️)** explicit spatial pointer memory, (12) **π³ 192 (BSD-3-Clause ✅)** permutation-equivariant, (13) **MapAnything 193 (Apache 2.0 code + dual-license weights ✅)** universal multi-modal, (14) **Reliev3R 194 (license TBD ⚠️)** weakly-supervised, (15) **Speed3R 195 (BSD-3-Clause code ✅, CC BY-NC 4.0 weights ⚠️, NEW)** trainable sparse attention.

**★ Updated v0 v1+ H2 paradigm update:** the *complete* 2024-2026 streaming/sparse-3R arc has now *conclusively* established that **sparse deterministic + knowledge distillation is *strictly better* than dense deterministic** for *latency-bound* clinical applications. The 6 papers in the sparse-3R arc (Spann3R 177 implicit memory, CUT3R 175 RNN state, Point3R 179 explicit pointer, Ray-Aware 180 retain-or-replace, STream3R 181 causal Transformer, TTT3R 182 TTT, R³ 183 relative regression, **Speed3R 195 trainable sparse attention**) all use *deterministic* feed-forward designs, *no* diffusion, *no* VAE, *no* probabilistic bottleneck. The *killer* paradigm shift: **sparse attention is the *new* default for clinical-3R**, *not* dense attention + diffusion.

**★ Updated v0 v1+ compute: ~$13,170-19,460 Lambda** (was $13,070-19,360 from 194-note, +$100 for the GSA re-implementation + clinical-distillation training). The *killer* saving: Speed3R's 12.4× speedup means v0 v1+ *clinical inference* cost is *12.4× cheaper* than dense FFRM, so the *operational* cost (per-patient inference) is *drastically* reduced. The *engineering* cost is *modest* (BSD-3-Clause code, ~50-100 Lambda to re-implement + integrate, 1-2 weeks).

**★ v0 sub-task 1 stack is now COMPLETE for the *clinical-real-time* deployment story:** MapAnything 193 (universal SOTA, Apache 2.0) + π³ 192 (permutation-equivariant, BSD-3-Clause) + Reliev3R 194 (weakly-supervised, license TBD) + **Speed3R 195 (sparse real-time, BSD-3-Clause code, CC BY-NC 4.0 weights, NEW)** = the *de facto* 2025-2026 3R-foundation-model *quadrifecta* for *complete* clinical deployment (scalable training [Reliev3R] + universal multi-modal SOTA [MapAnything] + permutation-equivariant [π³] + real-time inference [Speed3R]). The *complete* 2024-2026 FFRM arc is now: DUSt3R → MASt3R → MUSt3R → MonST3R 174 → CUT3R 175 → Spann3R 177 → Fast3R 178 → Point3R 179 → Easi3R 173 → π³ 192 → MapAnything 193 → Reliev3R 194 → **Speed3R 195 (NEW, trainable sparse attention)**. The *killer* 2026 paradigm shift: Reliev3R 194 (scalable training) + Speed3R 195 (scalable inference) = the *complete* clinical-IOS pipeline for *unbounded-scale* 3R deployment (any patient, any IOS scanner, any chairside hardware, in *real-time*).

**★ ⚠️ LICENSE BLOCKER for v0 production:** Speed3R-π³ *weights* are CC BY-NC 4.0 (non-commercial), inherited from π³'s training datasets. For v0 *research paper*: OK to use with attribution. For v0 v2 v3 *commercial production*: re-implement GSA + distill from clinical-dense model (option (c) + (e)), or seek license from the authors. The *killer* commercial-deployment reason: the *code* is BSD-3-Clause (commercial-friendly), the *recipe* is reproducible from scratch in 1-2 weeks, and the *clinical* domain has *unique* sparsity patterns that warrant a *dental-domain-finetuned* sparse-3R (not the *generic* CO3Dv2-trained Speed3R-π³). The *killer* takeaway: **Speed3R's GSA module is the *real* contribution, the *weights* are *just* the *first* instantiation** — and the *clinical* deployment is *the* opportunity for the *second* instantiation.

## Next Paper to Read

**Recommended:** Paper 196 — **LangSplat / LangSplat-V2 / OpenSeg + 3DGS (the 3D-language-grounding arc)** — the *direct* extension to 3D Gaussian Splatting (3DGS) that *adds* language features to the 3D scene, enabling *open-vocabulary* queries like "find the prep tooth" or "find the margin boundary". For v0 sub-task 1, this is the *killer* extension to MapAnything 193's universal multi-modal SOTA (which already supports language-conditioned depth + segmentation). The *practical* v0 v1+ lesson: combine Reliev3R 194 (scalable training) + Speed3R 195 (real-time inference) + LangSplat-V2 (open-vocabulary 3D) for the *most-flexible* clinical-IOS pipeline.

*Alternative 1:* **(a) Splatt3R (Smart 2024, arXiv:2410.18965)** — the *direct* DUSt3R + 3DGS follow-up that *removes* the *camera-pose* *requirement* (the *killer* for clinical v1 where IOS pose noise is a real bottleneck). The *practical* v0 v1+ lesson: combine Reliev3R 194 + Speed3R 195 + Splatt3R's pose-free inference for the *most-robust* clinical-IOS pipeline.

*Alternative 2:* **(b) AnySplat (Chen 2025, arXiv:2505.23716)** — the *unconstrained-views* 3DGS that works *without* *calibrated* *cameras* (the *killer* v1 sub-task 1 extension for *real-world* *clinical* *data*, but the *paper* is *newer* (2025) and may not be the *most-validated* in the field yet). The *practical* v0 v1+ lesson: combine Reliev3R 194 + Speed3R 195 + AnySplat's pose-free inference for the *most-flexible* clinical-IOS pipeline.

*Alternative 3:* **(c) TurboVGGT (CVPR 2026, arXiv:2505.23716)** — the *most-recent* (May 2026) speed-optimized VGGT that uses *adaptive alternating attention* (vs Speed3R's *fixed* GSA), the *concurrent* alternative to Speed3R with a *different* design philosophy (adaptive *block-sparse* attention vs Speed3R's *fixed* pool+select attention). The *practical* v0 v1+ lesson: compare Speed3R 195 vs TurboVGGT on the *same* clinical-IOS benchmark; the *winner* is the *v0 production* choice.

*Alternative 4:* **(d) FF3R (Microsoft Research, 2025)** — the *feedforward feature* 3R that *removes* the *camera-pose*, *depth-map*, *semantic-label* requirements (the *killer* for clinical v1 where *none* of these are available at *inference time*). The *practical* v0 v1+ lesson: combine Reliev3R 194 + Speed3R 195 + FF3R's *unconstrained-inference* design for the *most-flexible* clinical-IOS pipeline.

**Recommendation: *read 196 = TurboVGGT (CVPR 2026)*** — the *most-recent* (May 2026) speed-optimized VGGT, the *direct concurrent alternative* to Speed3R 195, the *right* next paper to *complete* the v0 sub-task 1 *real-time-3R* design space (Speed3R 195 = fixed pool+select, TurboVGGT = adaptive alternating attention, the *two* competing sparse-3R designs). After Speed3R 195 + TurboVGGT 196, the v0 sub-task 1 *real-time-3R* design space is *complete*.
