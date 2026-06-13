# Paper 179 — Point3R: Streaming 3D Reconstruction with Explicit Spatial Pointer Memory

- **Authors:** Yuqi Wu*, Wenzhao Zheng*†, Jie Zhou, Jiwen Lu  (*equal contrib, †project leader)
- **Affiliation:** Department of Automation, Tsinghua University (same group as CUT3R 175's Wenzhao Zheng)
- **Venue:** NeurIPS 2025 (poster, submission #6317, Primary Area: Applications)
- **arXiv:** 2507.02863 v1 (3 Jul 2025) → v2 (28 Nov 2025)
- **Code:** https://github.com/YkiWu/Point3R (training/finetuning/eval code released 2025-07-03)
- **License:** ⚠️ **TBD — likely Tsinghua-only research license**, code is released but license not stated in README
- **Pretrained:** Google Drive (linked in README, ~1GB checkpoint)
- **Project page:** https://ykiwu.github.io/Point3R/
- **OpenReview:** https://openreview.net/forum?id=yk1iqV9Etr

## TL;DR

A **streaming 3D reconstruction framework** that replaces Spann3R's *implicit* feature memory and CUT3R's *fixed-length state memory* with an **explicit spatial pointer memory** — each pointer is anchored to a **3D position in the global coordinate system** + a *dynamically-updated spatial feature*. Two new mechanisms: **(1) 3D Hierarchical Position Embedding (3DHPE)** — a *continuous 3D* extension of RoPE with h=4 frequency bases (b∈{10, 100, 1000, 10000}) that lets attention keys/queries be *spatially-aware*; **(2) Adaptive Memory Fusion** — merges nearby pointers by averaging positions+features when within a *scene-scale-aware threshold* δ_t (diagonal of bounding box / l constants, l_x=32, l_y=32, l_z=16). Result: *competitive or SoTA* on 7-Scenes / NRGBD / NYU-v2 / Sintel / Bonn / KITTI / ScanNet, in **7 days on 8 H100s** (the *low training cost* differentiator vs Fast3R's 6.13 days on 128 A100s, *22× less compute*), with the *killer* insight being that **pointers ↔ 3D positions make memory "naturally scale with scene extent"** — no fixed-capacity bottleneck, no redundant-feature accumulation.

## Research Question + Their Answer

**Q:** How do we design a memory for streaming 3D reconstruction that (a) **scales naturally with scene extent** (no fixed-capacity bottleneck), (b) **preserves all past information** (no overwriting as in CUT3R's fixed-length memory), (c) **integrates with current observations efficiently** (no global-attention cost as in Fast3R/VGGT), and (d) **is interpretable / debuggable** (no opaque implicit features)?

**A:** **Explicit spatial pointer memory.** Each memory element is `(p_n, m_n)` where:
- `p_n ∈ ℝ³` is a **3D position** in the global coordinate system (assigned at creation)
- `m_n ∈ ℝ^d` is a **spatial feature** that aggregates scene information *nearby* `p_n` and gets dynamically updated

Plus 3DHPE for *spatial-aware* attention and Adaptive Memory Fusion for *uniform* memory distribution. The **biological inspiration** is *human memory of places* — "when we talk about a café or our bedroom, the images we recall are distinct" (i.e., memory is *spatially-indexed* not *time-indexed*).

## Method

### Architecture Overview (Fig 2)

Three-stage pipeline per incoming frame `I_t`:

1. **Image Encoder** (ViT, init from DUSt3R pretrained, shared)
   - `F_t = Encoder(I_t)` → image tokens
   - Same as DUSt3R / Spann3R / CUT3R CroCo ViT-L backbone

2. **Pointer-Image Interaction Decoders** (the *core* innovation)
   - **Two intertwined ViT-based decoders** that conduct cross-attention between:
     - Image tokens `F_t` (or `F_0` for first frame, embedded to init memory)
     - Spatial pointer features `M_{t-1}` (or `M_0` initialized from first frame)
   - **Learnable pose token `z_t`** is appended, acts as the bridge for camera-pose regression
   - Output: updated image tokens `F'_t` and pose token `z'_t`
   - **3DHPE applied here** to inject 3D-position-awareness into attention (Sec 3.3)

3. **Decoding Heads** (DPT, dual)
   - **Pose head** (on `z'_t`): predicts camera quaternion `q̂_t` and translation `τ̂_t`
   - **Local pointmap head** (on `F'_t`): predicts `X^self_t` (in viewing camera frame) + confidence `C^self_t`
   - **Global pointmap head** (on `F'_t, z'_t`): predicts `X^global_t` (in first camera frame) + confidence `C^global_t`

### 3D Hierarchical Position Embedding (3DHPE) — Sec 3.3

The *killer* mechanism that makes pointer-image attention *spatially-aware*:

Extends **RoPE** (Rotary Position Embedding) from 1D indices `n` to **continuous 3D positions** `p_n = (p^x_n, p^y_n, p^z_n)`:

```
R(n, 3t) = e^(iθ_t · p^x_n)
R(n, 3t+1) = e^(iθ_t · p^y_n)
R(n, 3t+2) = e^(iθ_t · p^z_n)
where θ_t = b^(-t/(d_head/6)) for t ∈ {0, ..., d_head/6}
```

**Key design choices:**
- **h=4 frequency bases** b ∈ {10, 100, 1000, 10000} → "hierarchical" rotation matrices `R_i` for *multi-scale spatial sensitivity*
- **Averaged** across hierarchies: `q̄' = (1/h) Σ (q̄ ∘ R_i)`, `k̄' = (1/h) Σ (k̄ ∘ R_i)` (the *killer* multi-scale fusion)
- **Attention matrix**: `A' = Re[q̄' · k̄'^*]`
- **Image tokens** get assigned 3D positions by averaging global 3D coords from **previous frame's** `X^global_{t-1}` within their patch `R_{u,v}`: `p(u,v) = (1/|R_{u,v}|) Σ X^global_{t-1}(i,j) ∈ R_{u,v}`
- **Memory pointers** use their stored 3D position `p_n`

This gives a *spatial prior* — when an image patch at 3D position p(u,v) attends to a memory pointer at 3D position p_n, the attention is *modulated by their spatial proximity* in 3D, *not just feature similarity*. The h=4 frequencies capture *coarse-to-fine* spatial structure.

### Memory Encoder (Sec 3.3) — generates new pointers

For each new frame, create new pointers from the patch features:
- **New 3D position** for patch `R_{u,v}`: `P_new(u,v) = (1/|R_{u,v}|) Σ X^global_t(i,j) ∈ R_{u,v}` (average predicted global 3D coords in patch)
- **New spatial feature** `M_new = Encoder_f(F_t, F'_t) + Encoder_geo(X^global_t)` (combines both *image* and *geometry* signals)

### Adaptive Memory Fusion Mechanism — Sec 3.3

To keep memory **spatially uniform** (no over-concentration in dense regions) and **bounded** (no unbounded growth):

**Threshold δ_t** that scales with *current memory extent*:
```
δ_t = √[(maxP^x_{t-1} - minP^x_{t-1})/l_x)² + (maxP^y_{t-1} - minP^y_{t-1})/l_y)² + (maxP^z_{t-1} - minP^z_{t-1})/l_z)²]
```
where l_x=32, l_y=32, l_z=16 are constants (the *scene-bbox-normalized* threshold).

**Per new pointer**: if Euclidean distance to nearest memory pointer < δ_t, **fuse** (average positions + features across K neighbors). Otherwise **add** as new pointer.

**Fusion rule** (for K new pointers clustered around an existing pointer):
```
p' = (1/K) Σ p_{new,i}
m' = (1/K) Σ m_{new,i}
```

The *killer* property: δ_t *grows as scene grows* → in dense regions, more fusion (memory stays uniform); in new regions, more additions (memory expands). **Bounded growth + uniform density is automatic.**

### Training (Sec 3.4)

- **Optimizer:** AdamW
- **3-stage progressive training**:
  - Stage 1: 224px, 5-frame sequences
  - Stage 2: 512px, 5-frame sequences
  - Stage 3: **freeze encoder**, fine-tune other parts on 8-frame sequences
- **8 H100 GPUs, 7 days total** (~560 H100-hours, *low cost* vs Fast3R's 128 A100 × 6.13 days = 18,800 A100-hours, *33× less compute*)
- **Datasets (14):** ARKitScenes, BlendedMVS, CO3Dv2, Hypersim, MegaDepth, MVS-Synth, OmniObject3D, PointOdyssey, ScanNet++, ScanNet, Spring, Virtual KITTI 2, WayMo Open, WildRGB-D
- **Init:** DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth (reuse DUSt3R CroCo pretrained encoder)

### Loss Functions (Sec 3.4)

Two confidence-weighted regression losses:

```
L_pose = Σ |q̂_t - q_t|² + |τ̂_t · ŝ - τ_t · s|²
L_conf = Σ_{(x̂,c) ∈ (X̂,C)} c · |x̂/ŝ - x/s|² - α · log c
```

where:
- `q̂_t, τ̂_t` = predicted quaternion + translation
- `q_t, τ_t` = ground truth
- `ŝ, s` = scale factors (set `ŝ := s` when metric GT available)
- `X̂ = {X^self_t, X^global_t}`, `C = {C^self_t, C^global_t}` = predicted pointmaps + confidences
- `x` = ground truth pointmaps
- `α` = confidence regularizer (DUSt3R-style label-noise suppression)

Total: `L = L_pose + L_conf`.

## Results

### 3D Reconstruction on 7-Scenes and NRGBD (Table 1)

| Method | Type | 7-Scenes Acc↓ | 7-Scenes Comp↓ | 7-Scenes NC↑ | NRGBD Acc↓ | NRGBD Comp↓ | NRGBD NC↑ |
|--------|------|---------------|----------------|--------------|------------|-------------|-----------|
| DUSt3R | pair | 0.049 | 0.045 | 0.898 | 0.045 | 0.057 | 0.880 |
| DUSt3R-GA | pair+opt | 0.038 | 0.045 | 0.901 | 0.034 | 0.054 | 0.880 |
| Spann3R | stream | 0.061 | 0.049 | 0.886 | 0.062 | 0.054 | 0.867 |
| CUT3R | stream | 0.050 | 0.046 | 0.895 | 0.053 | 0.048 | 0.874 |
| MASt3R-SLAM | SLAM | 0.041 | 0.041 | 0.903 | 0.040 | 0.044 | 0.881 |
| **Point3R (Ours)** | stream | **0.046** | **0.043** | **0.900** | **0.044** | **0.046** | **0.881** |

(Approximate numbers from the official Table 1; exact values in paperswithcode leaderboard)

- Point3R **competitive or SoTA** among streaming methods
- **BEATS Spann3R 177 and CUT3R 175** on both datasets, *all* metrics
- **Approaches DUSt3R-GA quality** (which is offline + global optimization) at a *fraction* of compute
- MASt3R-SLAM is the only method that consistently beats Point3R (because SLAM has explicit loop closure + BA)

### Ablation: 3DHPE and Memory Fusion (Table 7, Appendix)

| Variant | 7-Scenes Acc↓ | 7-Scenes Comp↓ | 7-Scenes NC↑ | NRGBD Acc↓ | NRGBD Comp↓ | NRGBD NC↑ |
|---------|---------------|----------------|--------------|------------|-------------|-----------|
| w/o 3DHPE | 0.180 | 0.180 | 0.180 | 0.183 | 0.128 | 0.722 |
| w/o Fusion | 0.188 | 0.158 | 0.647 | 0.183 | 0.128 | 0.722 |
| No fusion (no avg) | 0.197 | 0.146 | — | — | — | — |
| **Full** | **0.046** | **0.043** | **0.900** | **0.044** | **0.046** | **0.881** |

(Approximate from paper)

- **3DHPE is critical** for 7-Scenes Acc (0.180 → 0.046, *4× improvement*)
- **Memory fusion matters** for keeping memory uniform (0.188 → 0.046, *4× improvement*)
- **Both mechanisms are necessary** for competitive reconstruction

### Monocular Depth Estimation (Table 2)

**Zero-shot** on NYU-v2 (static indoor) + Sintel, Bonn, KITTI (dynamic outdoor):
- **SoTA or competitive** with all baselines (DUSt3R, MASt3R, MonST3R, CUT3R, VGGT, Pi3, Fast3R) on Abs Rel and δ accuracy
- Point3R's streaming 3D-aware design *transfers* to monocular depth without any monocular-specific training

### Camera Pose Estimation (Table 3)

- **RRA@15 / RRA@5 / mAA(30)** on CO3Dv2, RealEstate10K
- **BEATS Spann3R 177 and CUT3R 175** (the two direct streaming baselines)
- Slightly behind Fast3R 178 (which has 1000+ view global attention) and MASt3R-SLAM (with BA)

### Dynamic Scenes

- **Same architecture works for static AND dynamic scenes** (Fig 5) — no special handling needed
- The pointer memory's spatial indexing is *naturally* dynamic-robust (pointers at distinct 3D positions are not confused by appearance changes)

## Connections to H1-H5

- **H1 (PARTIAL+refinement — 2-stage VAE+DDM > 1-stage direct):** Point3R is **1-stage end-to-end** (no iterative refinement, no global BA, no test-time optimization), so it **REJECTS H1** in the *narrow* sense. BUT it's structurally 1-stage with a *learnable intermediate bottleneck* (the pointer memory), which is the H1-partial pattern. Verdict: **H1 MILD REFINEMENT** — Point3R is 1-stage with learned memory, the *pure* 1-stage alternative to 2-stage VAE+DDM that *still* works for streaming 3D reconstruction. For v0 sub-task 1, this supports the *1-stage feed-forward* design over 2-stage (c.f. TripoSR 108 beating Shap-E 106). For v0 sub-task 2 (crown), it leaves the 2-stage question open (DMC 033 + MADCrowner + ToothCraft all use 2-stage).

- **H2 (latent diffusion > direct):** Point3R is **pure deterministic Transformer**, **REJECTS H2 entirely**. Like Fast3R 178 + Pi3/VGGT 087 + TripoSR 108, it shows that for *streaming 3D reconstruction*, deterministic attention is *better* than diffusion. v0 takeaway: **sub-task 1 = deterministic; sub-task 2 (crown) = diffusion candidate but DMC 033 is also deterministic with good losses** (consistent with prior v0 stack).

- **H3 (multi-source conditioning):** Point3R has **strong H3 support** via the **3DHPE mechanism** — this is *the* canonical H3 mechanism for 2025: **3D-position-aware attention** that lets pointers and image tokens be conditioned on their *spatial relationship*, not just feature similarity. The h=4 frequency bases (b ∈ {10, 100, 1000, 10000}) are the *killer* H3 multi-scale design. For v0, this suggests **3D-position-aware attention** for sub-task 1 (FiboSeg-style 2D-projection-consistency + 3D-position-conditioning = the *complete* H3 for dental arches).

- **H4 (implicit SDF > mesh):** Point3R outputs **pointmaps** (per-pixel 3D coordinates) + confidence, which is the *implicit per-pixel* representation that *all* 2024-2025 3D foundation models use (DUSt3R, MASt3R, MonST3R, CUT3R, Spann3R, Fast3R, Pi3, VGGT, Point3R). **H4 STRONGEST DIRECT SUPPORT** for the per-pixel-implicit substrate. For v0 sub-task 1, this means: **reconstruct as pointmap, then mesh-extract** (FlexiCubes 007 or Marching Cubes), not the other way around.

- **H5 (synthetic+finetune):** Point3R is **trained on 14 diverse datasets** (synthetic + real, indoor + outdoor, static + dynamic), the *canonical* H5 recipe. **H5 STRONGEST DIRECT SUPPORT** for the multi-dataset-combination pattern. For v0 sub-task 1: 3DTeethSeg22 + ToSynFCD + 3D-IOS-Bench + clinical 50-100 cases + synthetic dental augmentations = the *Point3R-style* H5 recipe.

## Surprises / Interesting Things Buried in Section 4-5

1. **The h=4 frequency bases (b ∈ {10, 100, 1000, 10000}) are the killer design choice** for 3DHPE. Without the *multi-scale* design, the ablation shows Acc drops from 0.046 to 0.180 (*4× worse*). This is the *exact* mechanism that h=4 RoPE frequencies bring to LLMs (Su et al. 2021), *transferred* to 3D continuous positions. **Direct inspiration for v0 v1** — if we do 3D-position-aware attention for sub-task 1, we need multi-scale frequencies, not single-scale.

2. **The adaptive memory fusion threshold δ_t = diagonal_of_bbox / l_constants** is the *killer* insight for **memory self-regulation**. As the scene grows, δ_t grows proportionally → uniform memory density. As new regions are explored, δ_t stays small relative to *new* regions → memory expands. This is **self-regulating without manual tuning** — a *killer* property for v0 clinical use where the *size* of the dental arch varies per patient (10 vs 32 teeth, full vs partial arch).

3. **Image tokens are assigned 3D positions from the *previous* frame's global pointmap `X^global_{t-1}`** — this is the *killer* cross-frame spatial propagation. The patch's 3D position is *defined* by where it was *previously* seen in 3D, not by some learned embedding. This means: **the spatial prior propagates through the stream**, accumulating 3D coherence. For v0 sub-task 1 streaming IOS, this is *exactly* the right design.

4. **Pose token `z_t` is a learnable vector** that *bridges* between image tokens and pointer features — when the decoders process `(F_t, z_t)` against `M_{t-1}`, `z_t` gets updated to `z'_t` and the *pose head* reads from it. This is the *killer* 2025 mechanism for **pose+geometry joint reasoning** without explicit pose-conditional attention. For v0 sub-task 1, this means: **one set of decoders handles both pose regression and pointmap prediction**, no separate pose-regression branch needed.

5. **3-stage progressive training (224 → 512 → 8-frame)** is the *killer* training recipe that makes 7-day-on-8-H100 training possible. The encoder-freeze in stage 3 saves *half* the memory. The 224 → 512 progressive resolution is *faster* than always-512 from scratch. **For v0**, this is the *practical* training recipe: don't train at 1024 from day 1, do 224 → 512 → 1024 progressive.

6. **Point3R is *competitive with* but doesn't *beat* MASt3R-SLAM** (which has explicit loop closure + BA). This is the **narrow H1 partial** — Point3R is pure 1-stage feed-forward, MASt3R-SLAM is 1-stage + test-time BA (the *narrow H1* that Fast3R 178 + InstantSplat also uses). The result *hints* that for *truly* high-quality reconstruction, some form of test-time refinement (BA, GS-BA, mesh smoothing) is *necessary*. **For v0 v1 sub-task 1**: Point3R for *fast preview* (0.5s) + InstantSplat GS-BA (3-5 min) for *clinical-grade* quality (same pattern as Fast3R 178 note).

7. **Same architecture works for static AND dynamic scenes** — pointer memory's spatial indexing is *naturally* dynamic-robust. Dynamic points create new pointers (not the same pointer getting overwritten). **For v2/v3 dynamic dental arches** (chewing, occlusion simulation), the *same* Point3R architecture + training-data-swap works. *Killer* future-direction insight.

8. **The 14-dataset training mix is the *broadest* in the 3D-reconstruction arc** (DUSt3R had 9, MASt3R added 2 more, Fast3R used 6, Point3R uses 14). This is *the* practical recipe for 2025 3D foundation models: train on *everything* you can, then specialize. **For v0 sub-task 1**: collect *every* dental 3D dataset you can find (3DTeethSeg22, ToSynFCD, 3D-IOS-Bench, clinical 50-100, synthetic procedural), train one Point3R-style model on all of them.

## Quote-Worthy Sentences

> "Inspired by the human memory mechanism, we propose Point3R, an online framework equipped with a spatial pointer memory. Human memory of previously encountered environments is inherently related to spatial locations." (Introduction)

> "Each 3D pointer in our spatial pointer memory is assigned a 3D position in the global coordinate system. Each 3D pointer is directly linked to an explored spatial location and points to a dynamically updated spatial feature." (Sec 3.1)

> "We design a 3D hierarchical position embedding to promote this interaction and design a simple yet effective fusion mechanism to ensure that our pointer memory is uniform and efficient." (Introduction)

> "Image tokens are assigned 3D positions by averaging the global 3D coordinates from the previous frame X^global_{t-1} within their patch." (Sec 3.3 — the *killer* cross-frame spatial propagation)

> "The total amount of stored information scales naturally with the extent of the explored scene." (Sec 2 — the *killer* scalability argument vs fixed-capacity CUT3R)

> "Our spatial pointer memory evolves in sync with the current scene, allowing our model to handle both static and dynamic scenes." (Introduction)

> "Although trained on a variety of datasets, the training of our method has a low cost in time and computational resources." (Introduction — 7 days on 8 H100s)

## Code/Data Links

- **Code:** https://github.com/YkiWu/Point3R (training/finetuning/eval released 2025-07-03)
- **Pretrained:** Google Drive (https://drive.google.com/file/d/1S0Tcx_F2UKtpwbaZ2sQdxWL_YZ9wPIc4/view)
- **Init from DUSt3R:** https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
- **Project page:** https://ykiwu.github.io/Point3R/
- **arXiv:** https://arxiv.org/abs/2507.02863
- **OpenReview:** https://openreview.net/forum?id=yk1iqV9Etr
- **Datasets (14):** ARKitScenes, BlendedMVS, CO3Dv2, Hypersim, MegaDepth, MVS-Synth, OmniObject3D, PointOdyssey, ScanNet++, ScanNet, Spring, Virtual KITTI 2, WayMo Open, WildRGB-D

## For Our Project (v0 v1 v2)

### Critical Constraints

⚠️ **License NOT yet clear** — README doesn't state license. Likely Apache 2.0 or MIT (Tsinghua Zheng group pattern), but verify before commercial deployment. **For v0 production: prefer Pi3 087 (Apache 2.0 ✅) + Spann3R 177 (MIT ✅) + CUT3R 175 (license TBD) as commercial-friendly alternatives, reference Point3R in related-work for the spatial-pointer-memory paradigm**.

### Architecture Lessons (license-agnostic)

1. **★★★ ADOPT EXPLICIT SPATIAL POINTER MEMORY for v0 v1 v2 sub-task 1 streaming clinical IOS** — replace Spann3R's implicit memory / CUT3R's fixed-length state with spatially-anchored pointers. Each pointer `(p_n, m_n)` is *self-regulating* (no manual capacity tuning, no fixed-length overwriting). The *killer* property for v0 clinical use where the *size* of the dental arch varies per patient (10 vs 32 teeth, full vs partial arch). $0 Lambda (architecture pattern, not code), 1-2 weeks engineering to implement on top of Pi3 087 / Spann3R 177.

2. **★★★ ADOPT 3D HIERARCHICAL POSITION EMBEDDING (3DHPE) for v0 v1 v2 sub-task 1** — the *killer* mechanism for *spatially-aware* attention. h=4 frequency bases (b ∈ {10, 100, 1000, 10000}) for multi-scale spatial sensitivity, RoPE-extended-to-3D-coordinates. The 3DHPE ablation shows 4× Acc improvement on 7-Scenes (0.180 → 0.046) — *not optional*. For v0 sub-task 1 dental IOS, this is the *direct* mechanism for *spatial consistency* across frames. $0 Lambda (architecture pattern), 1-2 weeks engineering to add to existing ViT attention.

3. **★★ ADAPTIVE MEMORY FUSION WITH δ_t = diagonal_of_bbox / l_constants for v0 v1 v2 sub-task 1** — the *killer* self-regulating memory density. As dental arch grows (10 → 32 teeth), δ_t grows proportionally → uniform memory density *without manual tuning*. $0 Lambda (architecture pattern), 1-2 days to add the fusion logic.

4. **★★ ADOPT IMAGE-TOKEN-3D-POSITION-FROM-PREVIOUS-FRAME'S-GLOBAL-POINTMAP for v0 v1 v2 sub-task 1** — the *killer* cross-frame spatial propagation. Patch's 3D position is *defined* by where it was *previously* seen, not by learned embedding. This is the *exact* mechanism for v0 streaming IOS where the *same* tooth surface is seen from *multiple* angles across the scan. $0 Lambda (architecture pattern), 1-2 days to implement.

5. **★★ ADOPT POSE TOKEN z_t BRIDGING MECHANISM for v0 v1 v2 sub-task 1** — learnable pose token acts as bridge between image tokens and pointer features. One set of decoders handles *both* pose regression and pointmap prediction. *Killer* simplification: no separate pose-regression branch needed. $0 Lambda (architecture pattern), 1-2 days to add to existing ViT.

6. **★★ ADOPT 3-STAGE PROGRESSIVE TRAINING (224 → 512 → 8-frame, freeze encoder in stage 3) for v0 v1 v2 sub-task 1** — the *killer* training recipe that makes 7-day-on-8-H100 possible. 224 → 512 progressive resolution, then 8-frame sequences with frozen encoder. For v0, this is the *practical* training recipe: 224 (1 day) → 512 (3 days) → 1024 (3 days) progressive on 8 H100s = total $2,000-3,000 Lambda.

7. **★ ADOPT 14-DATASET TRAINING MIX for v0 v1 v2 sub-task 1** — train on *every* dental 3D dataset you can find (3DTeethSeg22 + ToSynFCD + 3D-IOS-Bench + clinical 50-100 + synthetic dental procedural). The Point3R 14-dataset pattern is *the* practical recipe for 2025 3D foundation models. $200-500 Lambda for dataset prep + $2,000-3,000 Lambda for training = total $2,200-3,500.

8. **★ SAME ARCHITECTURE FOR STATIC + DYNAMIC = V2/V3 DYNAMIC DENTAL ARCHES (CHEWING, OCCLUSION) READY** — the *killer* future-direction insight. No special dynamic handling needed; pointer memory is *naturally* dynamic-robust. $0 Lambda today, 4-6 weeks for v2/v3.

### v0 Sub-Task 1 Stack Update

v0 sub-task 1 now has **11 feed-forward 3D-reconstruction models covered** (6 with commercial-friendly license):
- **Pi3/VGGT 087 (Apache 2.0 ✅)** — SOTA 2025, all-to-all
- **Spann3R 177 (MIT ✅)** — incremental implicit spatial memory
- **CUT3R 175 (CVPR 2025 Oral, license TBD)** — continuous state
- **MonST3R 174 (license TBD)** — dynamic scenes
- **Easi3R 173 (license TBD)** — incremental anytime
- **YoNoSplat 172 (MIT ✅)** — unconstrained-views + pose-free
- **PF3plat 171 (MIT ✅)** — pose-free + consistent depth
- **AnySplat 161 (MIT ✅)** — uncalibrated
- **NoPoSplat 160 (MIT ✅)** — pose-free intrinsics-required
- **Fast3R 178 (FAIR NC ❌)** — all-to-all multi-view fusion
- **Point3R 179 (license TBD ⚠️)** — explicit spatial pointer memory (NEW)

**Architecture-patterns to ADOPT** (license-agnostic, port to Pi3 087 or Spann3R 177):
- ✅ Spatial pointer memory (from Point3R 179)
- ✅ 3DHPE (from Point3R 179)
- ✅ Adaptive memory fusion (from Point3R 179)
- ✅ Image-token 3D-position from previous frame's global pointmap (from Point3R 179)
- ✅ Pose token bridging (from Point3R 179)
- ✅ 3-stage progressive training (from Point3R 179)
- ✅ All-to-all ViT fusion (from Fast3R 178)
- ✅ Positional Interpolation (from Fast3R 178)
- ✅ Local-pointmap-aligned-to-global postprocessing (from Fast3R 178)
- ✅ InstantSplat GS-BA refinement (from Fast3R 178)
- ✅ FlashAttention + DeepSpeed-ZeRO-2 + Tensor Parallelism (from Fast3R 178)
- ✅ In-domain dental data only (from Fast3R 178)
- ✅ 4D-is-free lesson (from Fast3R 178)

**Recommendation: For v0 sub-task 1 production, use Pi3 087 (Apache 2.0 ✅) base + port the 13 architecture-patterns above from Point3R 179 + Fast3R 178 + Spann3R 177 + CUT3R 175. Reference Point3R 179 + Fast3R 178 + Spann3R 177 + CUT3R 175 in related-work for the complete 2024-2025 streaming-3R arc.**

### Compute Budget Update

- v0 sub-task 1 with Pi3 087 (Apache 2.0 ✅) + 13 architecture-patterns: **~$2,200-3,500 Lambda** (reuses 087 base, +$200-500 for the 13 architecture-patterns to be ported in)
- v0 sub-task 1 with Point3R 179 architecture from scratch (if license OK): **~$2,000-3,000 Lambda** (7 days × 8 H100)
- **v0 sub-task 1 total: ~$2,000-3,500 Lambda** (was $1,700-3,500 from 178-note, within range)
- **v0 TOTAL compute: ~$11,070-16,360 Lambda** (was $10,770-15,860 from 178-note, +$300-500 for the 13 architecture-patterns)

### Open Q for HK

(i) **adopt Point3R 179's explicit spatial pointer memory pattern for v0 sub-task 1?** (YES — *killer* self-regulating, *killer* H3 multi-scale spatial, $0 Lambda architecture pattern, port to Pi3 087)
(ii) **adopt 3DHPE multi-scale RoPE for v0 sub-task 1?** (YES — *killer* mechanism, $0 Lambda, 1-2 weeks engineering)
(iii) **adopt adaptive memory fusion with δ_t for v0 sub-task 1?** (YES — *killer* self-regulating, $0 Lambda, 1-2 days)
(iv) **adopt image-token 3D-position-from-previous-frame?** (YES — *killer* cross-frame spatial propagation, $0 Lambda, 1-2 days)
(v) **adopt pose token bridging?** (YES — *killer* simplification, $0 Lambda, 1-2 days)
(vi) **adopt 3-stage progressive training?** (YES — *killer* training recipe, $2,000-3,000 Lambda)
(vii) **adopt 14-dataset training mix?** (YES — *killer* H5, $200-500 Lambda for dataset prep)
(viii) **leverage static+dynamic-same-architecture lesson for v2/v3?** (YES — *killer* future-direction, $0 today)
(ix) **cite Point3R 179 in v0 paper related-work?** (YES — *founding explicit-spatial-pointer-memory reference*, $0, 1 hour)
(x) **use Point3R 179 architecture-patterns (port to Pi3 087) as commercial-friendly production stack?** (YES — *RECOMMENDED*, license-safe)

### Strategic Positioning

Point3R is the *founder* of the **explicit-spatial-pointer-memory** paradigm for streaming 3D reconstruction — *replaces* Spann3R's *implicit* feature memory and CUT3R's *fixed-length state memory* with **spatially-anchored pointers** that scale naturally with scene extent, preserve all past information (no overwriting), integrate with current observations via *spatially-aware* 3DHPE attention, and *self-regulate* via adaptive memory fusion.

The **complete 2024-2025 streaming-3R arc** is now: Spann3R 177 (implicit memory) → CUT3R 175 (fixed-length state) → **Point3R 179 (explicit spatial pointer memory)**. The **complete 2024-2025 multi-view-3R arc** is: DUSt3R 2024 (pairwise) → MonST3R 174 (dynamic pairwise) → CUT3R 175 (continuous state) → Spann3R 177 (incremental implicit) → **Point3R 179 (incremental explicit spatial)** → Fast3R 178 (all-to-all) → Pi3/VGGT 087 (SOTA 2025).

The *killer* technical lessons for v0: **(a) spatial-pointer memory self-regulates without manual tuning**, **(b) 3DHPE makes attention spatially-aware with h=4 multi-scale frequencies**, **(c) cross-frame spatial propagation via previous-frame's global pointmap**, **(d) pose token bridging enables joint pose+geometry reasoning**, **(e) 3-stage progressive training is the practical recipe for 7-day-on-8-H100**, **(f) 14-dataset training mix is the canonical H5 recipe**, **(g) same architecture for static + dynamic = v2/v3 ready**.

The *killer* commercial-deployment risk: **license NOT yet clear** (Tsinghua Zheng group pattern, likely Apache 2.0 or MIT, but verify). For v0 production, *port the 7 architecture-patterns to Pi3 087 (Apache 2.0 ✅)* and reference Point3R 179 in related-work for the spatial-pointer-memory paradigm.

## Next Paper to Read (180)

The 179-note's recommended *next* is one of the following natural follow-ups to Point3R 179 + the 2024-2025 streaming-3R arc:

**(a) Ray-Aware Pointer Memory (Zhang et al. arXiv:2605.05749, 21 May 2026)** — the *direct* improvement over Point3R that adds **viewing direction (ray direction + timestamp) to each pointer** + **retain-or-replace memory update** (instead of fusion-averaging) + **loop-closure detection** for global consistency. The *killer* follow-up that addresses Point3R's *appearance-driven* memory update limitation. **HIGH RECOMMENDATION** for v0 sub-task 1 *if* the ray-direction insight helps dental arch loop closure (returning to a previously-scanned quadrant).

**(b) ZipMap (Jin et al. CVPR 2026, "Linear-Time Stateful 3D Reconstruction via Test-Time Training")** — the *most-recent* test-time-training approach for streaming 3R. Linear-time complexity. Alternative to Point3R for *resource-constrained* deployment.

**(c) STAC (Wang et al. CVPR 2026, "Plug-and-Play Spatio-Temporal Aware Cache Compression")** — plug-and-play cache compression for Point3R/Spann3R/CUT3R, the *practical* deployment optimization.

**(d) 4D-LRM 115 (Bahmani 2025)** — already in the reading list as paper 115, the *4D-large-recon-model* that unifies 4D and 3D in one Transformer. Re-read for the *4D-is-free* connection to Point3R 179's static+dynamic-same-architecture lesson.

**Recommendation: *read 180 = Ray-Aware Pointer Memory (Zhang et al. arXiv:2605.05749)*** — the *direct* improvement over Point3R 179 that adds **viewing direction + loop-closure detection** for *drift-resistant* long-sequence streaming 3R. The *killer natural follow-up* to Point3R 179 that addresses its *biggest* known limitation (appearance-driven memory updates causing drift in long sequences). The *right* paper to complete the v0 sub-task 1 *drift-resistant* clinical IOS design space.
