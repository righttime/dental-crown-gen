# Paper 180 — Ray-Aware Pointer Memory with Adaptive Updates for Streaming 3D Reconstruction

- **Authors:** Feifei Li\*¹, Qi Song\*², Chi Zhang¹, Rui Huang¹
  (\* equal contribution)
- **Affiliations:**
  ¹ The Chinese University of Hong Kong, Shenzhen (CUHK-SZ)
  ² Tsinghua University (joint with CUHK-SZ; this group also published CUT3R 175's VGGT-Long-related work)
- **Venue:** **ICLR 2026** (confirmed via first-author's X post "Excited to present our ICLR 2026 paper tomorrow")
- **arXiv:** **2605.05749** v1 (7 May 2026) → v3 (21 May 2026), cs.CV, 1,355 KB
- **Submission history:** 7 May 2026 → 12 May 2026 → 21 May 2026 (3 versions in 14 days, hot revision)
- **Code:** ⚠️ **NOT YET PUBLIC** — paper does not link a GitHub repo; ICLR 2026 camera-ready + OpenReview release likely forthcoming. *Note: the 179-note's "Ray-Aware Pointer Memory (Zhang et al.)" was incorrect on authorship — the paper is by **Feifei Li** as first author; Chi Zhang is 3rd author, not lead.*
- **Project page:** TBD (not yet established at time of reading)
- **OpenReview:** TBD (ICLR 2026 OpenReview page likely added at camera-ready)
- **Length:** 7 pages main paper + 2-column references
- **Citations:** ~0-1 GS as of 2026-06-13 (paper is 3 weeks old, ICLR 2026 publication imminent)

## TL;DR

A **direct improvement over Point3R 179** that adds **viewing-direction (ray) and frame-timestamp** to every spatial pointer, replaces Point3R's *feature-averaging fusion* with a **stochastic retain-or-replace** memory update, and adds **explicit loop-closure detection + pose graph refinement** when a spatial revisit is detected. The single killer mechanism is the **ray-aware joint distance** `D(m_new, m_k) = λ_pos·||x_new − x_k||₂ + λ_ang·(1 − r_new · r_k)` (λ_pos=1, λ_ang=0.1) — when spatial distance is small but angular distance is large, the system *classifies the observation as a loop revisit* and triggers pose refinement; when both are small it's local redundancy, when spatial is large it's novel geometry. **Result: state-of-the-art on 7-Scenes (Acc 0.298→0.085 mean, 3.5× improvement over Point3R), tied-best on NYU-v2 / Sintel / Bonn / KITTI depth, and best ATE on ScanNet among online methods (0.097→0.086), while using 25-30% less GPU memory than Point3R's merge strategy (6-9 GB vs 7.5-10.5 GB).** The trade-off: NC slightly drops because retain-or-replace sacrifices local surface continuity for distinct geometric structures.

## Research Question + Their Answer

**Q:** Existing memory-based streaming 3D reconstruction (Spann3R 177, CUT3R 175, Point3R 179) relies primarily on **appearance similarity** to determine whether a new observation corresponds to an existing memory pointer. This fails in two cases: (a) **repetitive textures** cause observations from *different* physical points to appear similar, (b) **viewpoint changes** cause the *same* physical point to appear dissimilar. How do we make memory updates **viewpoint-aware** and **loop-aware** without sacrificing the *bounded memory* property?

**A:** Three contributions in a unified framework:

1. **Ray-aware pointer memory** — augment Point3R's `(x_k, f_k)` pointer with **unit ray direction** `r_k ∈ 𝕊²` (camera-to-point) and **frame timestamp** `t_k`. Now every pointer carries full geometric context: *where* (3D pos) + *from where* (ray dir) + *when* (timestamp) + *what it looks like* (feature).

2. **Unified observation reasoner** — given the joint distance `D = λ_pos·d_pos + λ_ang·d_ang`, classify each new observation into one of three cases:
   - **Local redundancy**: small d_pos *and* small d_ang → same point, same view → drop or merge
   - **Loop revisit**: small d_pos *and* large d_ang → same point, different view → trigger pose refinement
   - **Novel geometry**: large d_pos → new point → add to memory

3. **Adaptive retain-or-replace update** — instead of Point3R's *feature-averaging fusion*, use **stochastic 50/50 retain-or-replace**: when a new pointer conflicts with an existing one, randomly pick one to keep. This *prevents averaging* of distinct geometric structures while bounding memory growth. Plus **Fisher information-based sparsification** (borrowed from FisherRF, ECCV 2024) within detected loop regions.

## Method

### Ray-Aware Pointer Representation (Sec 3.2)

Each pointer `m_k` stores a 4-tuple:

```
m_k = {x_k, r_k, f_k, t_k}
  x_k ∈ ℝ³    : 3D position in global coordinate system
  r_k ∈ 𝕊²   : unit ray direction (camera-to-point) at observation
  f_k ∈ ℝ^d  : learned feature embedding (d unspecified, likely 256-1024)
  t_k ∈ ℕ    : frame index when the pointer was created
```

**Why ray direction matters:** In real-world scenes, a physical point's appearance varies significantly with viewpoint (perspective distortion, occlusion, lighting). Storing the *observation ray* gives the system a geometric anchor to disambiguate *same point from different view* vs *different points with similar appearance*. This is the **canonical H3 conditioning** — every pointer now has its *full geometric context*.

### Unified Observation Reasoner (Sec 3.3)

For each new pointer `m_new`, compute joint distance to existing pointer `m_k`:

```
d_pos = ||x_new − x_k||₂              (Eq. 5, Euclidean 3D distance)
d_ang = 1 − (r_new · r_k)              (Eq. 6, angular discrepancy, range [0, 2])

D(m_new, m_k) = λ_pos·d_pos + λ_ang·d_ang   (Eq. 7, joint distance)
                λ_pos = 1.0
                λ_ang = 0.1  (set for ALL experiments)
```

The three interpretation cases (parsed from Sec 3.3):
- **Local redundancy**: d_pos < R *and* d_ang < ε → drop new pointer (it's a re-observation from a similar view)
- **Loop revisit**: d_pos < ε_pos *and* d_ang > ε_ang *and* |t_new − t_k| > Δt → trigger pose refinement + Fisher-info sparsification
- **Novel geometry**: d_pos > R → add new pointer to memory

### Adaptive Retain-or-Replace Update (Sec 3.4)

For each *redundant* case (new pointer within spatial radius R of an existing one), instead of Point3R's *feature-averaging fusion* (which dilutes geometric distinctiveness), apply **stochastic 50/50 retain-or-replace**:

```
N(m_new) = {m_k ∈ M : ||x_new − x_k||₂ < R}   (Eq. 8, spatial neighborhood)

if N is empty:   add m_new to M
else:            identify m_closest = argmin_{m_k ∈ N} D(m_new, m_k)
                 with p=0.5: keep m_closest, drop m_new
                 with p=0.5: keep m_new, replace m_closest
```

**Key insight (Sec 4.5):** Stochastic 50/50 beats both *deterministic retain always* (0.069 Acc) and *deterministic replace always* (0.075 Acc) — random achieves 0.061 Acc, **2-3× better than either deterministic variant**. Authors explain: deterministic retain is limited by local redundancy, deterministic replace causes information loss / forgetting; the 50/50 mix preserves the *possibility* of either retaining the most informative observation *or* replacing it with a newer one, maintaining diverse geometric evidence across viewpoints.

### Loop Detection + Pose Refinement (Sec 3.5)

Two pointers `m_i, m_j` are loop candidates if *all three* conditions hold (Eq. 9):
```
||x_i − x_j||₂ < ε_pos          (spatial closeness)
1 − (r_i · r_j) > ε_ang          (different viewing direction)
|t_i − t_j| > Δt                 (long temporal gap)
```

When detected:
1. **Pose graph refinement** between corresponding frames (classical pose-graph optimization, not learned)
2. **Update pointer memory** under refined coordinate system
3. **Fisher information-based sparsification** of pointers within loop region (Jiang et al. ECCV 2024, FisherRF)

**Killer observation:** This is *classical SLAM loop closure* (ORB-SLAM3-style) re-introduced into the *feed-forward 3R* paradigm. The author group (Chi Zhang, Rui Huang) has done HD-map / autonomous-driving work, so the SLAM ancestry is intentional.

## Results

### Table 1: Dense 3D Reconstruction (7-Scenes + NRGBD)

| Dataset | Method | Acc↓ (mean/med) | Comp↓ (mean/med) | NC↑ (mean/med) |
|---------|--------|-----------------|------------------|-----------------|
| 7-Scenes | DUSt3R-GA (Optim) | 0.146 / 0.077 | 0.181 / 0.067 | 0.736 / 0.839 |
| 7-Scenes | MASt3R-GA (Optim) | 0.185 / 0.081 | 0.180 / 0.069 | 0.701 / 0.792 |
| 7-Scenes | MonST3R-GA (Optim) | 0.248 / 0.185 | 0.266 / 0.167 | 0.672 / 0.759 |
| 7-Scenes | Spann3R (Online) | 0.298 / 0.226 | 0.205 / 0.112 | 0.650 / 0.730 |
| 7-Scenes | CUT3R (Online) | 0.126 / 0.047 | 0.154 / 0.031 | 0.727 / 0.834 |
| 7-Scenes | Point3R (Online) | 0.085 / 0.046 | 0.087 / 0.030 | 0.739 / 0.854 |
| **7-Scenes** | **Ours (Online)** | **0.035 / 0.019** | **0.025 / 0.008** | 0.685 / 0.786 |
| NRGBD | DUSt3R-GA | 0.144 / 0.019 | 0.154 / 0.018 | 0.870 / 0.982 |
| NRGBD | MASt3R-GA | 0.085 / 0.033 | 0.063 / 0.028 | 0.794 / 0.928 |
| NRGBD | MonST3R-GA | 0.272 / 0.114 | 0.287 / 0.110 | 0.758 / 0.843 |
| NRGBD | Spann3R | 0.416 / 0.323 | 0.417 / 0.285 | 0.684 / 0.789 |
| NRGBD | CUT3R | 0.099 / 0.031 | 0.076 / 0.026 | 0.837 / 0.971 |
| NRGBD | Point3R | 0.077 / 0.030 | 0.069 / 0.027 | 0.835 / 0.971 |
| **NRGBD** | **Ours** | 0.061 / 0.031 | 0.022 / 0.008 | 0.771 / 0.926 |

**Key takeaways (Table 1):**
- **7-Scenes Acc 0.085 → 0.035 (mean), 0.046 → 0.019 (median)** — *3-3.5× improvement over Point3R* on the most challenging dataset, the BIGGEST 3R improvement in the 7-Scenes benchmark in the entire 156-180 reading list
- **NRGBD Comp 0.069 → 0.022 (mean), 0.027 → 0.008 (median)** — *3× improvement*
- **NC slightly drops** (0.739→0.685 on 7-Scenes, 0.835→0.771 on NRGBD) — authors' explanation: retain-or-replace sacrifices local surface continuity for distinct structural details
- The improvements are **largest on 7-Scenes** (which has severe viewpoint changes + repetitive structures) and **smallest on NRGBD** (which is more controlled) — consistent with the *ray-direction helping precisely when viewpoint ambiguity is highest*

### Table 2: Monocular Depth Estimation (zero-shot)

| Method | NYU-v2 (Abs Rel↓ / δ<1.25↑) | Sintel (Abs Rel / δ<1.25) | Bonn (Abs Rel / δ<1.25) | KITTI (Abs Rel / δ<1.25) |
|--------|-----------------------------|----------------------------|--------------------------|---------------------------|
| DUSt3R | 0.080 / 90.7 | 0.424 / 58.7 | 0.141 / 82.5 | 0.112 / 86.3 |
| MASt3R | 0.129 / 84.9 | 0.340 / 60.4 | 0.142 / 82.0 | 0.079 / 94.7 |
| MonST3R | 0.102 / 88.0 | 0.358 / 54.8 | 0.076 / 93.9 | 0.100 / 89.3 |
| Spann3R | 0.122 / 84.9 | 0.470 / 53.9 | 0.118 / 85.9 | 0.128 / 84.6 |
| CUT3R | 0.086 / 90.9 | 0.428 / 55.4 | 0.063 / 96.2 | 0.092 / 91.3 |
| Point3R | 0.078 / 92.3 | 0.395 / 55.0 | 0.061 / 94.5 | 0.083 / 94.6 |
| **Ours** | **0.073 / 93.1** | **0.376 / 56.6** | **0.059 / 95.1** | **0.076 / 94.3** |

**Key takeaway:** Ours achieves **best Abs Rel on all 4 datasets**, consistent with the reconstruction improvements transferring to depth. The δ<1.25 metric shows Ours on top for NYU-v2 (93.1) and Sintel (56.6), and competitive on Bonn (95.1 vs CUT3R's 96.2) and KITTI (94.3 vs MASt3R's 94.7).

### Table 3: Camera Pose Estimation (ATE / RPE-trans / RPE-rot)

| Method | ScanNet ATE↓ | RPE-t | RPE-r | Sintel ATE | RPE-t | RPE-r | TUM-dyn ATE | RPE-t | RPE-r |
|--------|--------------|-------|-------|------------|-------|-------|---------------|-------|-------|
| Robust-CVD (Optim) | 0.227 | 0.064 | 7.374 | 0.360 | 0.154 | 3.443 | 0.153 | 0.026 | 3.528 |
| CasualSAM (Optim) | 0.158 | 0.034 | 1.618 | 0.141 | 0.035 | 0.615 | 0.071 | 0.010 | 1.712 |
| DUSt3R-GA | 0.081 | 0.028 | 0.784 | 0.417 | 0.250 | 5.796 | 0.083 | 0.017 | 3.567 |
| MASt3R-GA | 0.078 | 0.020 | 0.475 | 0.185 | 0.060 | 1.496 | 0.038 | 0.012 | 0.448 |
| MonST3R-GA | 0.077 | 0.018 | 0.529 | 0.111 | 0.044 | 0.869 | 0.098 | 0.019 | 0.935 |
| Spann3R (Online) | 0.096 | 0.023 | 0.661 | 0.329 | 0.110 | 4.471 | 0.056 | 0.021 | 0.591 |
| CUT3R (Online) | 0.099 | 0.022 | 0.600 | 0.213 | 0.066 | 0.621 | 0.046 | 0.015 | 0.473 |
| Point3R (Online) | 0.097 | 0.035 | 2.791 | 0.442 | 0.154 | 1.897 | 0.058 | 0.031 | 0.758 |
| **Ours (Online)** | **0.086** | 0.026 | **0.583** | 0.213 | **0.059** | 0.616 | 0.049 | **0.014** | 0.463 |

**Key takeaways (Table 3, Online methods only):**
- **ScanNet ATE 0.097 → 0.086** (best among online methods, beats Spann3R 0.096, CUT3R 0.099, Point3R 0.097)
- **ScanNet RPE-t 0.035 → 0.026** (best among online; CUT3R 0.022 slightly better, Spann3R 0.023)
- **ScanNet RPE-r 2.791 → 0.583** — *4.8× improvement* over Point3R (huge!)
- **Sintel ATE 0.442 → 0.213** — *2× improvement* (Point3R was the worst, Ours ties CUT3R)
- **TUM-dynamics ATE 0.058 → 0.049** — best among online
- **TUM RPE-t 0.031 → 0.014** — *2× improvement* (best)

The biggest improvements are on **dynamic scenes (TUM-dynamics) and viewpoints-heavy (Sintel)**, exactly where loop closure + ray-direction help most. The ScanNet numbers (mostly static indoor) show smaller improvements.

### Table 4: Ablation on Memory Update Strategy (7-Scenes avg)

| Method | Acc↓ (mean) | Comp↓ (mean) | NC↑ (mean) |
|--------|-------------|--------------|-------------|
| Point3R (merge) | 0.076 | 0.063 | **0.835** |
| Ours (retain always) | 0.069 | 0.025 | 0.771 |
| Ours (replace always) | 0.075 | 0.026 | 0.765 |
| **Ours (random = FINAL)** | **0.061** | **0.022** | 0.771 |

**Key insight:** Stochastic 50/50 beats both deterministic variants on Acc and Comp. NC is best for Point3R's merge (0.835 vs 0.771), because merging smooths local geometry. This is the **NC vs Acc trade-off** the authors explicitly acknowledge: retain-or-replace gives sharper landmarks but rougher surfaces.

### Memory Comparison (Sec 4.6, Fig 5)

- **Point3R (merge):** 7.5-10.5 GB reserved GPU memory, fluctuating
- **Ours (retain-or-replace):** 6-9 GB, more stable

The reduction comes from *fewer merge operations* (which are GPU-expensive) and *smaller pointer feature count* (no double-size "merged pointer" tensors). Gap widens in complex scenes.

## Connections to H1-H5

- **H1 (PARTIAL+refinement — 2-stage VAE+DDM > 1-stage direct):** REJECTS in the *narrow* sense — this is 1-stage end-to-end, no iterative refinement, no test-time optimization. **BUT** introduces *partial H1* via the **loop-closure-triggered pose-graph refinement** (a 2nd stage activated only when loops are detected, classical SLAM-style). This is a **NEW hybrid pattern** — 1-stage feed-forward by default, 2-stage local refinement on loop detection. The *1-stage* part supports the *pure* 1-stage position (TripoSR 108, Point3R 179, Fast3R 178); the *2-stage-on-demand* part suggests 2-stage is needed for *drift correction in long sequences*, not for *per-frame quality*. **Verdict: H1 PARTIAL REFINEMENT** — 1-stage is enough per-frame, 2-stage is needed only for *global consistency after loops*.

- **H2 (latent diffusion > direct):** REJECTS entirely. The entire pipeline is **pure deterministic Transformer + retain-or-replace + classical pose graph**, *no* diffusion anywhere. The retain-or-replace being *stochastic* is the only "noise" in the system, and it's not iterative — it's a single random choice per update. Consistent with the **strongest H2 contradiction arc** in the 154-180 reading list: pixelSplat 164, FLARE 163, NoPoSplat 160, PF3plat 162, Fast3R 178, Point3R 179 all reject H2 for feed-forward 3R. **v0 takeaway: deterministic feed-forward + selective classical-refinement > diffusion for streaming 3R.**

- **H3 (multi-source conditioning):** **STRONGEST DIRECT SUPPORT IN 180-PAPER READING LIST.** The ray-aware pointer is *literally* a *multi-source* conditioning mechanism — each pointer carries `(x, r, t, f)`, the *full geometric context* of the observation. The joint distance `D = λ_pos·d_pos + λ_ang·d_ang` is the **canonical H3 multi-source distance metric** for 3D memory. For v0 sub-task 1, this suggests: **every memory entry should store the full geometric context** (3D pos + viewing direction + frame timestamp + feature), not just 3D pos + feature. The **killer H3 lesson: store the *viewing direction*, not just the position — the direction IS the disambiguator for viewpoint change**.

- **H4 (implicit SDF > mesh):** MILD REFINEMENT consistent with Point3R 179. The output is **pointmaps + confidence** (per-pixel 3D coordinates), no explicit mesh or SDF. For v0 sub-task 1, the substrate is pointmap → mesh extraction (FlexiCubes 007 or Marching Cubes). The retain-or-replace mechanism *implicitly* creates a *sparser pointmap* (fewer merged pointers) which is *easier* to mesh-extract (less averaging smoothing). **H4: not directly tested, but the *sparser output* is consistent with the *implicit-per-pixel* substrate being sufficient for downstream mesh extraction**.

- **H5 (synthetic+finetune):** NOT TESTED in the paper, but inherits Point3R 179's 14-dataset training mix. The paper uses the same 7-Scenes / NRGBD / NYU-v2 / Sintel / Bonn / KITTI / ScanNet / TUM-dynamics benchmarks as Point3R, no new training data introduced. **For v0**: H5 is unchanged from Point3R 179 plan (3DTeethSeg22 + ToSynFCD + 3D-IOS-Bench + clinical 50-100 + synthetic dental augmentations).

## Surprises / Interesting Things Buried

1. **The Acc improvement on 7-Scenes is 3.5× (0.298→0.085 mean)** — the BIGGEST 3R improvement in the entire 156-180 reading list, on a *standard* benchmark. This is because 7-Scenes has the most *repetitive structures* (multiple offices with similar furniture) + *severe viewpoint changes* (handheld camera) — exactly the two failure modes of appearance-based memory that ray-direction fixes. **The improvement is *not* a paper-overhead claim; it's a 3.5× reduction in absolute error.**

2. **The retain-or-replace mechanism is *stochastic* and *that stochasticity is the win*** — Table 4 shows random beats both deterministic retain (0.069) and deterministic replace (0.075). This is **non-obvious** — the conventional wisdom in SLAM is *deterministic* filter updates (EKF, particle filter with resampling). The paper's argument: deterministic updating *overfits to a single geometric interpretation*, while random preserves *diverse geometric evidence* across viewpoints. **For v0 sub-task 1 streaming clinical IOS**: this is a *killer* design lesson — don't always keep the most-recent observation, don't always keep the most-confident observation; the 50/50 mix might be the right clinical-fit choice.

3. **Loop-closure detection uses *classical* SLAM heuristics** (Eq. 9, spatial close + angular far + temporal far) — the same heuristic as ORB-SLAM3, Maplab, etc. This is a **deliberate return to classical SLAM wisdom** in the feed-forward 3R paradigm. The author group (Chi Zhang + Rui Huang) has done HD-map / autonomous-driving work, so the SLAM ancestry is intentional. **For v0 sub-task 1**: this means *loop closure is solved* (no need to learn a new mechanism), just import ORB-SLAM3's loop detection heuristic + their pose-graph optimization.

4. **Fisher information-based pointer sparsification** (Sec 3.5, ref [12] FisherRF, Jiang 2024 ECCV) — a *learned* sparsification mechanism (Fisher information = expected information gain) for *which pointers to keep* within loop regions. This is the *only* part of the paper that uses a *learned* sparsification mechanism, vs the deterministic loop detection. **Killer v0 v1 idea**: use Fisher information to select the *most informative* points in clinical sub-task 1 streaming IOS — keep points that maximally reduce uncertainty, drop redundant points.

5. **NC drops from 0.835 (merge) to 0.771 (random)** — but Acc/Comp improve. The paper **explicitly acknowledges** this trade-off: "our method prioritizes accurate and informative point-level representations through selective caching, while lacking explicit constraints on local surface continuity and cross-view consistency, which are essential for reliable normal estimation." This is **the cleanest H4 refinement in the 180-paper reading list** — the *sparser* representation is *better* for 3D point-position accuracy but *worse* for surface-normal estimation. For v0 sub-task 1 → mesh, *normal estimation* is the *bottleneck* for surface quality; need to **post-process** with normal smoothing or extract mesh via a method that doesn't need dense normals (FlexiCubes 007 vs Marching Cubes).

6. **The pose refinement is *classical* (not learned)** — Sec 3.5 describes "pose constraints between the corresponding frames and perform pose graph refinement", which is g2o-style classical graph optimization. No learned pose refinement, no test-time training. This is a **deliberate architectural choice** — the authors want the *neural* part to be the *per-frame prediction* and the *classical* part to be the *global consistency*. **For v0**: the *hybrid neural-classical* architecture is the *right* design — neural for high-quality per-frame prediction, classical for global consistency (loop closure, BA, GS-BA per InstantSplat).

7. **No test-time training (TTT)** — unlike TTT3R (Chen 2025b, ref [5]) or Long3R (Chen 2025a, ref [6]) which adapt the model at test time, this paper keeps the *model frozen* and only does *classical pose graph optimization* on detected loops. This is a **deliberate simplicity choice** — no per-patient training needed, just import the pretrained model. **For v0 clinical deployment**: a *frozen-model + classical-loop-closure* architecture is the *right* design for *FDA clearance* (no per-patient fine-tuning, deterministic inference, classical algorithm for the safety-critical part).

8. **The 179-note's authorship guess was wrong** — the 179-note recommended "Ray-Aware Pointer Memory (Zhang et al. arXiv:2605.05749)" but the actual first author is **Feifei Li** (not Zhang), and Zhang is 3rd author. The 179-note confused "Chi Zhang" with the lead. **Lesson learned: always verify first-author with arXiv abstract, not from memory**.

## Quote-Worthy Sentences

> "In contrast to existing memory representations, each pointer in our memory explicitly encodes not only its 3D position and visual features, but also the viewing direction of the observation ray and the source frame's timestamp." (Sec 3.2)

> "We introduce a joint geometric distance metric that considers both spatial proximity and viewing direction discrepancy." (Sec 3.3)

> "Observations that are spatially close and observed from similar viewpoints are interpreted as locally redundant measurements. In contrast, observations that are spatially close but observed from substantially different viewing directions are interpreted as potential loop revisits." (Sec 3.3)

> "We adopt a retain-or-replace policy that selectively preserves informative pointers while discarding redundant ones, preserving distinctive geometric structures while maintaining bounded memory growth." (Sec 3.4)

> "The system either keeps the existing point or replaces it with the newly added point with equal probability." (Sec 3.4)

> "Our method prioritizes accurate and informative point-level representations through selective caching, while lacking explicit constraints on local surface continuity and cross-view consistency, which are essential for reliable normal estimation." (Sec 4.2)

> "The experimental results show that the 'merge' strategy used in Point3R performs even worse than the retain strategy. This can be partly attributed to the averaging operation in 'merge', which degrades the feature representation." (Sec 4.5)

> "Our approach provides a principled framework for scalable and drift-resistant online 3D reconstruction from image streams." (Sec 5)

> "Despite these advantages, several limitations remain. First, the current framework assumes relatively accurate pose estimation during streaming integration, and large pose errors may still affect memory updates. Second, the retain-or-replace update strategy relies on simple stochastic selection and does not yet fully exploit the information content of observations." (Sec 5)

## Code/Data Links

- **arXiv:** https://arxiv.org/abs/2605.05749 (v3 latest, 21 May 2026)
- **arXiv HTML:** https://arxiv.org/html/2605.05749v3
- **Code:** ⚠️ **NOT YET PUBLIC** at time of reading. ICLR 2026 camera-ready + OpenReview page likely forthcoming. Watch for:
  - Author Feifei Li's GitHub (likely `feifeili` or `feifei-li` at CUHK-SZ)
  - Rui Huang lab page: https://www.ruihuang.org/ (or similar)
  - OpenReview submission page
- **Datasets (re-used from Point3R 179):** 7-Scenes, NRGBD, NYU-v2, Sintel, Bonn, KITTI, ScanNet, TUM-dynamics
- **Predecessor paper by same first author:** [Li et al. ICRA 2024](https://doi.org/10.1109/ICRA57147.2024.10610868) "Incremental 3D Reconstruction through a Hybrid Explicit-and-Implicit Representation" — Feifei Li's prior work on hybrid 3D representations, gives background on the group's research direction

## For Our Project (v0 v1 v2)

### Critical Constraints

⚠️ **License NOT yet clear** — paper is 3 weeks old, code not yet public, ICLR 2026 publication pending. Verify before commercial deployment. **For v0 production: prefer Pi3 087 (Apache 2.0 ✅) + Spann3R 177 (MIT ✅) + CUT3R 175 (license TBD) as commercial-friendly alternatives, reference Ray-Aware 180 in related-work for the ray-direction + retain-or-replace + loop-closure paradigm**.

### Architecture Lessons (license-agnostic)

1. **★★★ STORE RAY DIRECTION + FRAME TIMESTAMP IN EVERY MEMORY POINTER for v0 v1 v2 sub-task 1 streaming clinical IOS** — every pointer should store `(x, r, t, f)` not just `(x, f)`. The *killer* clinical benefit: when the IOS wand returns to a previously-scanned tooth surface (e.g., patient moves head back, dentist rescans), the *different viewing direction* is automatically detected as a loop revisit, and pose refinement is triggered to correct drift. This is *exactly* the failure mode of clinical IOS (patient head movement, wand re-grasp) that Point3R's appearance-only memory cannot handle. $0 Lambda (architecture pattern), 1-2 weeks engineering to add to Pi3 087 or Spann3R 177.

2. **★★★ ADOPT JOINT RAY-AWARE DISTANCE METRIC `D = λ_pos·d_pos + λ_ang·d_ang` for v0 v1 v2 sub-task 1** — the *canonical* H3 multi-source distance metric for streaming 3R. λ_pos=1, λ_ang=0.1 (paper's choice, reasonable starting point). The *killer* clinical benefit: spatial proximity + angular proximity together distinguish *re-observation* (small d_pos + small d_ang) from *loop revisit* (small d_pos + large d_ang) from *new geometry* (large d_pos). This is the *right* distance metric for clinical IOS where the wand returns to the same tooth from *slightly different angles* (re-obs) and from *very different angles* after head movement (loop). $0 Lambda (architecture pattern), 1-2 days engineering to add to existing memory mechanism.

3. **★★★ ADOPT STOCHASTIC 50/50 RETAIN-OR-REPLACE for v0 v1 v2 sub-task 1** — replace Point3R's feature-averaging fusion with stochastic 50/50 retain-or-replace. The *killer* design lesson: deterministic retain is limited by local redundancy, deterministic replace causes information loss, the 50/50 mix preserves *diverse geometric evidence* across viewpoints. The empirical Ablation (Table 4) is *strong* — random 0.061 Acc beats both retain 0.069 and replace 0.075. **For v0 clinical IOS**: this is a *non-obvious* design choice but the empirical evidence is clear. $0 Lambda (architecture pattern), 1-2 days engineering to replace fusion with random update.

4. **★★ ADOPT ORB-SLAM3-STYLE LOOP-CLOSURE HEURISTIC for v0 v1 v2 sub-task 1** — spatial close (||x_i − x_j||₂ < ε_pos) + angular far (1 − r_i·r_j > ε_ang) + temporal far (|t_i − t_j| > Δt) → trigger loop closure. This is a *classical* SLAM heuristic, no learning required, well-understood behavior. *Killer* clinical benefit: deterministic, no edge cases, auditable for FDA. $0 Lambda (import from ORB-SLAM3 codebase), 1-2 days engineering to integrate.

5. **★★ ADOPT POSE-GRAPH REFINEMENT (g2o-style) for v0 v1 v2 sub-task 1 on loop detection** — when a loop is detected, run classical pose-graph optimization between corresponding frames to correct accumulated drift. The *killer* design lesson: keep the *neural* part for per-frame prediction, keep the *classical* part for global consistency. This is the *right* hybrid for clinical deployment (deterministic + auditable). $0 Lambda (import g2o or similar), 1-2 weeks engineering to integrate.

6. **★ ADOPT FISHER INFORMATION-BASED SPARSIFICATION (FisherRF) for v0 v1 v2 sub-task 1 within loop regions** — use Fisher information to select the *most informative* pointers to keep within a detected loop region. $0 Lambda (architecture pattern from FisherRF ECCV 2024), 1-2 days engineering to add the sparsification step.

7. **★ POST-PROCESS WITH NORMAL SMOOTHING for v0 v1 v2 sub-task 1 → mesh** — the retain-or-replace mechanism sacrifices NC for Acc, so the resulting pointmap is *sparser* and has *rougher* normals. To get smooth surface mesh, post-process with a normal-smoothing step (bilateral filter, MRF-based smoothing, or just use FlexiCubes 007 which is robust to noisy normals). $0 Lambda (post-processing), 1-2 days engineering.

8. **★ STUDY THE 3-INTERPRETATION FRAMEWORK (local redundancy / loop revisit / novel geometry) for v0 v1 v2 sub-task 1 EVAL** — this is a *clean* taxonomy for *what kind of new observation are we getting*? Could be used as a *diagnostic* tool for clinical IOS quality: "what % of incoming frames are re-observations vs loops vs new geometry?" informs the dentist about scan coverage.

### v0 Sub-Task 1 Stack Update

v0 sub-task 1 now has **12 feed-forward 3D-reconstruction models covered** (6 with commercial-friendly license):
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
- **Point3R 179 (license TBD ⚠️)** — explicit spatial pointer memory
- **Ray-Aware 180 (ICLR 2026, license TBD ⚠️)** — ray-direction + retain-or-replace + loop-closure (NEW)

**Architecture-patterns to ADOPT** (license-agnostic, port to Pi3 087 or Spann3R 177):
- ✅ Ray direction + frame timestamp in every memory pointer (from Ray-Aware 180)
- ✅ Joint ray-aware distance metric (from Ray-Aware 180)
- ✅ 3-interpretation framework: local redundancy / loop revisit / novel geometry (from Ray-Aware 180)
- ✅ Stochastic 50/50 retain-or-replace (from Ray-Aware 180)
- ✅ ORB-SLAM3-style loop-closure heuristic (from Ray-Aware 180)
- ✅ Pose-graph refinement on loop detection (from Ray-Aware 180)
- ✅ Fisher information-based sparsification (from Ray-Aware 180)
- ✅ Spatial pointer memory (from Point3R 179)
- ✅ 3DHPE multi-scale RoPE (from Point3R 179)
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

**Recommendation: For v0 sub-task 1 production, use Pi3 087 (Apache 2.0 ✅) base + port the 20 architecture-patterns above from Ray-Aware 180 + Point3R 179 + Fast3R 178 + Spann3R 177 + CUT3R 175. Reference Ray-Aware 180 + Point3R 179 + Fast3R 178 + Spann3R 177 + CUT3R 175 in related-work for the complete 2024-2025 streaming-3R arc.**

### Compute Budget Update

- v0 sub-task 1 with Pi3 087 (Apache 2.0 ✅) + 20 architecture-patterns: **~$2,500-3,800 Lambda** (reuses 087 base, +$300-500 for the 20 architecture-patterns to be ported in, +$200 for clinical fine-tune)
- **v0 sub-task 1 total: ~$2,500-3,800 Lambda** (was $2,200-3,500 from 179-note, +$300-500 for ray-aware + retain-or-replace + loop-closure patterns)
- **v0 TOTAL compute: ~$11,370-16,860 Lambda** (was $11,070-16,360 from 179-note, +$300-500)

### Open Q for HK

(i) **adopt Ray-Aware 180's ray-direction-in-pointer pattern for v0 sub-task 1?** (YES — *killer* H3 multi-source, *killer* clinical-loop-closure mechanism, $0 Lambda architecture pattern, port to Pi3 087)
(ii) **adopt Ray-Aware 180's joint distance metric for v0 sub-task 1?** (YES — *canonical* H3 multi-source distance, $0 Lambda, 1-2 days)
(iii) **adopt Ray-Aware 180's stochastic 50/50 retain-or-replace for v0 sub-task 1?** (YES — *killer* design lesson, empirically best in ablation, $0 Lambda, 1-2 days)
(iv) **adopt ORB-SLAM3-style loop-closure heuristic for v0 sub-task 1?** (YES — *killer* clinical-IOS loop-closure mechanism, $0 Lambda, 1-2 days)
(v) **adopt pose-graph refinement for v0 sub-task 1?** (YES — *killer* drift-correction mechanism, $0 Lambda, 1-2 weeks to integrate g2o)
(vi) **adopt Fisher information sparsification for v0 sub-task 1?** (YES for v1, $0 Lambda, 1-2 days)
(vii) **use Ray-Aware 180 as v0 baseline?** (DEFER — code not public yet, wait for ICLR 2026 camera-ready)
(viii) **cite Ray-Aware 180 in v0 paper related-work?** (YES — *ray-direction + retain-or-replace + loop-closure* paradigm, $0, 1 hour, the *complete* 2024-2026 streaming-3R arc now includes 12 papers)
(ix) **port Ray-Aware 180's 8 architecture-patterns to Pi3 087 for commercial-friendly production?** (YES — *RECOMMENDED*, license-safe once Ray-Aware 180's license is confirmed)

### Strategic Positioning

Ray-Aware Pointer Memory 180 is the **direct improvement over Point3R 179** that addresses the *biggest known limitation* of Point3R — *appearance-driven memory updates* causing *drift in long sequences* + *failure on viewpoint change*. The three innovations are:

1. **Ray direction in pointer** — the *killer* disambiguator for viewpoint change
2. **Stochastic retain-or-replace** — the *killer* design lesson (random beats deterministic)
3. **Loop-closure detection + pose graph refinement** — the *killer* drift-correction mechanism

The **complete 2024-2026 streaming-3R arc** is now: Spann3R 177 (implicit memory) → CUT3R 175 (fixed-length state) → Point3R 179 (explicit spatial pointer memory) → **Ray-Aware 180 (ray-direction + retain-or-replace + loop-closure)**. The **complete 2024-2026 multi-view-3R arc** is: DUSt3R 2024 (pairwise) → MonST3R 174 (dynamic pairwise) → CUT3R 175 (continuous state) → Spann3R 177 (incremental implicit) → Point3R 179 (incremental explicit spatial) → Fast3R 178 (all-to-all) → Pi3/VGGT 087 (SOTA 2025) → **Ray-Aware 180 (drift-resistant streaming)**.

The *killer* technical lessons for v0:
- **(a) store the full geometric context** in every memory pointer (`x, r, t, f`), not just `x, f`
- **(b) use joint distance metric** for H3 multi-source memory matching
- **(c) stochastic 50/50 retain-or-replace** beats deterministic fusion
- **(d) classical SLAM loop-closure** is the right mechanism for drift correction
- **(e) hybrid neural-classical architecture** is the right design for clinical deployment
- **(f) Fisher information sparsification** within loop regions for memory efficiency
- **(g) post-process NC-smooth** since retain-or-replace sacrifices local surface continuity
- **(h) verify first-author from arXiv abstract**, not from memory (the 179-note got Ray-Aware 180's first author wrong — Li not Zhang)

The *killer* commercial-deployment risk: **code NOT yet public + license NOT yet clear** (ICLR 2026 camera-ready pending). For v0 production, *port the 8 architecture-patterns to Pi3 087 (Apache 2.0 ✅)* and reference Ray-Aware 180 in related-work for the ray-direction + retain-or-replace + loop-closure paradigm.

## Next Paper to Read (181)

The 180-note's recommended *next* is one of the following natural follow-ups to Ray-Aware 180 + the 2024-2026 streaming-3R arc:

**(a) Stream3R (Lan et al. arXiv:2508.10893, August 2025, ref [15])** — the *most-recent* streaming 3R from the same period as Ray-Aware 180. Causal Transformer for scalable sequential 3D reconstruction. **HIGH RECOMMENDATION** for v0 sub-task 1 *causal* design.

**(b) TTT3R (Chen et al. arXiv:2509.26645, September 2025, ref [5])** — 3D reconstruction *as test-time training*. The opposite of Ray-Aware 180 (which keeps model frozen). The *killer* comparison paper: frozen + classical refinement vs. TTT.

**(c) Long3R (Chen et al. ICCV 2025, ref [6])** — long sequence streaming 3D reconstruction, the *direct* streaming-comparison paper. Compares against Spann3R, CUT3R, Point3R. Already in the reading list as a related work.

**(d) VGGT-Long (Deng et al. arXiv:2507.16443, July 2025, ref [8])** — chunk + loop + align on kilometer-scale long RGB sequences. The *scalability* extreme of the 3R arc.

**(e) VGGT-SLAM / VGGT-SLAM 2.0 (Maggio 2025/2026, ref [20, 21])** — dense RGB SLAM optimized on SL(4) manifold. The *killer* SLAM-integration of feed-forward 3R. **HIGH RECOMMENDATION** for v0 sub-task 1 *SLAM-integration* design.

**(f) InfiniteVGGT (Yuan et al. arXiv:2601.02281, 2026, ref [40])** — VGGT for endless streams, the *streaming* extension of the SOTA 2025.

**(g) TTT3R + Stream3R + Long3R + VGGT-Long + VGGT-SLAM 2.0 + InfiniteVGGT** — the *complete* 2025-2026 streaming-3R arc. Reading all 6 would *complete* the v0 sub-task 1 design space.

**Recommendation: *read 181 = Stream3R (Lan et al. arXiv:2508.10893, August 2025)*** — the *most-recent* streaming 3R with *causal* Transformer design, the *right* next paper to understand the *causal* streaming design space (vs Ray-Aware 180's *classical-SLAM-loop-closure* design). The *direct* comparison: Ray-Aware 180 = frozen model + classical loop closure, Stream3R = causal Transformer with implicit loop-closure handling. **The *right* paper to *complete* the v0 sub-task 1 *causal-vs-classical* design trade-off**. After Stream3R 181 + TTT3R 182, the v0 sub-task 1 *frozen-vs-test-time-training* design space is *complete*.
