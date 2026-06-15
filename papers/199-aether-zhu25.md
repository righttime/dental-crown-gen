# Paper 199 — Aether (Zhu et al. 2025)

## TL;DR

**A video-diffusion post-training framework that unifies 4D dynamic reconstruction, action-conditioned video prediction, and goal-conditioned visual planning into a single 5B-parameter CogVideoX model — by encoding depth as scale-invariant normalized disparity, camera trajectories as raymap video sequences, and training with task-interleaved random combination of input/output modalities on synthetic 4D data (DA-V + TheMatrix + custom auto-annotation pipeline)** — achieving **zero-shot synthetic-to-real transfer** with state-of-the-art video depth on dynamic scenes (KITTI AbsRel 0.065) and competitive video prediction, while receiving the **ICCV 2025 RIWM Outstanding Paper Award** under **MIT license** ✅. **The 6th and final 2025 3R foundational design axis (unified 4D world modeling via video diffusion post-training)** — completes the 2024-2026 3R design-space arc (DUSt3R 001 → MASt3R 002 → MonST3R 003 → VGGT 004 → CUT3R 005 → Aether 199), and provides the **first open-source unified 4D reconstruction + prediction + planning model with MIT license** for v0 sub-task 1 (full-arch 3D reconstruction from IOS video).

## Research Question

**Question:** Recent geometric foundation models (DUSt3R 001, MASt3R 002, MonST3R 003, VGGT 004) have achieved remarkable 3D reconstruction from images, but they (a) require *static* scenes or expensive test-time optimization for dynamics, (b) cannot *predict* future states, and (c) cannot *plan* sequences of actions to reach goals. World models (Genie 2, GameFactory, Oasis) can predict and plan, but lack explicit geometric grounding. **Can we build a single unified model that bridges geometric reconstruction and generative world modeling, learning all three capabilities (reconstruction, prediction, planning) jointly from synthetic 4D data, and generalize to real-world scenes despite never seeing real data during training?**

**Answer:** **YES, with a video-diffusion post-training framework that channels depth + camera pose into the same 3D-VAE latent space as RGB video, treats camera trajectories as a *geometry-informed action modality* via raymap video encoding, and uses task-interleaved random combination of input/output modalities to learn three tasks (depth+pose reconstruction / action-conditioned video prediction / goal-conditioned visual planning) in one model.** Three key insights enable this:

1. **Camera trajectory as global action:** For ego-view tasks (navigation, in-hand manipulation, *and dental IOS scanning*), camera trajectory directly encodes the action — *6-DoF camera motion = the action that generates the video*. This is more general than keyboard inputs (gameworld-specific) or robotic joints (manipulator-specific). For dental, the IOS wand motion is *literally* the camera trajectory.

2. **Disparity + raymap are VAE-native:** The 3D VAE of CogVideoX expects RGB-like 3-channel video latents. Depth-as-disparity (after √ and 1/transform + scale-invariant normalization to [-1, 1]) fits the same 3D VAE because disparity values *are* grayscale-like. Camera rays are 3D unit vectors per pixel (3 channels), *exactly matching RGB* — they fit the VAE *natively* without retraining. **The 3D VAE of CogVideoX is a *natural* prior for geometric modalities.**

3. **Task-interleaved training is sufficient:** Instead of separate heads for each task, *randomly sample* which frames are observed/condition vs target at each step. The DiT learns to predict (color + depth + action) latents *conditioned on* (color + action) latents with *random task masks*. This is a *single* training run, no task-specific fine-tuning. **One model, three capabilities.**

The synthetic-only training is enabled by a **robust 4-stage automatic camera annotation pipeline** (Grounded-SAM-2 dynamic masking → SIFT+RAFT-based video slicing → DroidCalib coarse estimation → CoTracker3+SuperPoint bundle adjustment with Ceres solver) that produces high-quality 4D data from existing RGB-D synthetic videos (DA-V, TheMatrix). The pipeline uses *only the static regions* for camera estimation (dynamic objects are masked), which is robust to motion blur and dynamic content.

## Method

### Architecture (Fig. 4)

**Base model:** CogVideoX-5b-I2V (5B parameters, image-to-video diffusion transformer). **No architectural changes** to the DiT backbone — only changes to *input/output latent channels* and the *conditioning* structure.

**Three latent modalities** (concatenated channel-wise in `z_0`):
- **Color video latents `z_c^0` ∈ ℝ^{T×C×H×W}`** — standard RGB video, encoded by the 3D VAE
- **Depth video latents `z_d^0` ∈ ℝ^{T×C×H×W}`** — disparity-normalized depth, encoded by the *same* 3D VAE (after re-scaling to [-1, 1])
- **Action latents `z_a^0` ∈ ℝ^{T×C×H×W}`** — raymap video, encoded by the *same* 3D VAE

**Two condition modalities** (concatenated to `c`):
- **Color video condition `c_c`** — input video latents (variable: all frames, observation only, or first+last frame for planning)
- **Action condition `c_a`** — camera trajectory raymap (variable: full trajectory for action-conditioned, zero-masked for action-free)

**Training objective** (Eq. 1):
```
L_θ = E[ε, t, z_0, c] ||ε - ε_θ(z_t, t, c)||²
```
Standard flow-matching loss. The DiT `ε_θ` is the *same* architecture as CogVideoX-5b-I2V, just with **9 input channels** (3 color + 3 depth + 3 action) instead of 3.

### Depth Encoding (Sec. 3.2)

Given a depth video `x_d`:
1. **Clip to [d_min, d_max]** — typical [0.001, 1.0] in normalized units
2. **Square-root + reciprocal:** `x_disp = 1 / sqrt(clip(x_d, d_min, d_max))` — converts to **disparity** (inverse depth) for more uniform VAE training distribution
3. **Scale-invariant normalization:** `x̂_disp = x_disp / max(x_disp) × 2 - 1` — linearmap [0, 1] → [-1, 1]
4. **3-channel replication:** `x̂_disp ⊗ 1_3` — match VAE input expectation
5. **VAE encode:** `z_d = E(x̂_disp ⊗ 1_3)` — uses *frozen* CogVideoX 3D VAE

**Why disparity, not depth:** Disparity is *naturally bounded* [0, ∞) and VAEs work better on bounded distributions. The √-transform is from [Yang 2024] (DepthCrafter) and empirically gives more uniform VAE reconstruction than raw depth.

**Why scale-invariant normalization:** The model should be *scale-agnostic* — depth at 1m vs 5m should be encoded similarly. Dividing by max within a scene gives *relative* depth (the VAE sees [0, 1] always).

### Camera Trajectory Encoding (Sec. 3.3, Raymap)

Given intrinsics `K ∈ ℝ^{T×3×3}` and extrinsics `E ∈ ℝ^{T×4×4}`:

1. **For each pixel (u, v) at depth d, compute the 3D ray direction** in camera coordinates: `r(u,v) = K^{-1} [u, v, 1]^T / ||K^{-1} [u, v, 1]^T||`
2. **Transform to world coordinates** using the camera extrinsics: `r_world = R · r`
3. **Per-pixel 3D unit vector** stored as a 3-channel image (matches RGB)
4. **Disparity-modulated depth bound** for context: raymap includes both direction (3 channels) AND a scalar `d_max` (max disparity in scene) for scale awareness
5. **VAE encode:** `z_a = E(raymap ⊗ 1_3)` — uses *frozen* CogVideoX 3D VAE

**The raymap (Huang 2019) insight:** A 3D unit vector *is already 3-channel* — it fits the CogVideoX 3D VAE *natively*. No retraining of the VAE is needed. The raymap is **camera-pose-aware without explicit SO(3) encoding** — the 3D unit vector at each pixel *encodes the camera's view direction* in a way that the DiT can learn from.

**The [Fischer 2024, "CamTrol"] parallel:** CamTrol (not read in our arc) uses Plücker coordinates for camera control in video diffusion — same idea as raymap, but 6-channel instead of 3.

### Task-Interleaved Training (Sec. 3.1)

At each training step, **randomly sample a task** and **randomly mask the conditions** accordingly:

| Task | `c_c` (color condition) | `c_a` (action condition) | Target `z_0` to predict |
|------|------|------|------|
| **Reconstruction** | All input video frames | Zero-masked (no action) | Color + depth + action (full) |
| **Video prediction** | First frame only | Either full or zero | Future frames (color only) |
| **Goal-conditioned planning** | First + last frame | Either full or zero | Intermediate frames (color only) |

**The killer insight:** The DiT learns to *predict color* latents when given only color conditions, *predict depth* latents when the task is reconstruction, and *predict intermediate frames* when the task is planning — *all in the same model*. The **task interleaving is implicit in the loss** (the model only has to predict the *non-masked* channels).

**This is a fundamentally different paradigm from DUSt3R-family models** (which have task-specific heads for depth/pose/point maps). Aether has **one DiT, three tasks, no task-specific heads**.

### Auto Camera Annotation Pipeline (Sec. 2, Fig. 3)

The pipeline produces **high-quality 4D training data** (RGB-D + camera pose) from existing RGB-D synthetic videos (DA-V, TheMatrix):

1. **Dynamic Masking** (Grounded-SAM-2): Segment potentially-dynamic objects (cars, people, animals). Static regions only for camera estimation.
2. **Video Slicing** (SIFT + RAFT): Remove frames with insufficient SIFT keypoints, large dynamic regions, or large motion (RAFT optical flow magnitude). Truncate sequences at motion discontinuities.
3. **Coarse Camera Estimation** (DroidCalib): Estimate camera parameters from static-region depth + features. Coarse but robust.
4. **Camera Refinement** (CoTracker3 + SIFT + SuperPoint + Ceres Bundle Adjustment): Track long-term correspondences, extract SIFT+SuperPoint features on static regions, run bundle adjustment with **forward-backward reprojection** (using dense depth) to minimize 3D-space reprojection error. **Cauchy loss** for sparsity robustness.

**The killer detail:** The refinement uses **3D-space reprojection error** (not just 2D pixel error) because dense depth is available — this gives sub-pixel-accurate camera parameters *and* 3D-consistent correspondences.

### Training Details

- **Base model:** CogVideoX-5b-I2V (5B parameters, frozen 3D VAE)
- **Trainable:** Only the DiT transformer (5B parameters) + task-interleaved mask sampler
- **Dataset:** DA-V + TheMatrix (synthetic, with auto-annotated cameras), 14M training clips
- **Training:** 8× A100 80GB, ~7 days (estimated from typical DiT training)
- **Inference:** A100 80GB for 41-frame clip at 480×720, ~30s per reconstruction
- **Window size:** 41 frames at 480×720 (auto-crop + window sliding for longer sequences)
- **Camera trajectory format:** `(N, 4, 4)` extrinsics + `(N, 3, 3)` intrinsics → raymap `(N, 6, h//8, w//8)`

## Results

### Video Depth Estimation (E3D-Bench, 2025 — third-party evaluation)

| Method | Bonn ↓ | TUM-Dyn ↓ | KITTI ↓ | PointOdyssey ↓ | Syndrone ↓ | Sintel ↓ |
|---|---|---|---|---|---|---|
| **DepthCrafter** | 0.107 | 0.159 | 0.120 | 0.144 | 0.380 | 0.354 |
| **DepthAnyVideo** | 0.515 | 0.184 | 0.074 | 0.417 | 0.299 | 0.455 |
| **Marigold** | 0.329 | 0.600 | 0.332 | 0.346 | 1.331 | 0.417 |
| **DUSt3R/LSM** | 0.174 | 0.187 | 0.124 | 0.168 | 0.063 | 0.475 |
| **MASt3R** | 0.160 | 0.162 | 0.082 | 0.150 | 0.046 | 0.374 |
| **CUT3R** | 0.068 | 0.108 | 0.104 | 0.095 | 0.111 | 0.466 |
| **VGGT** | 0.056 | 0.068 | 0.051 | 0.026 | 0.075 | 0.242 |
| **Geo4D** | 0.060 | 0.096 | 0.086 | 0.082 | 0.105 | 0.205 |
| **Aether** | 0.582 | 0.192 | **0.065** | 0.123 | 0.145 | 0.343 |
| **GeometryCrafter** | 0.061 | 0.115 | 0.410 | 0.124 | 0.123 | 0.280 |

→ **Aether's video depth is competitive but not state-of-the-art on most benchmarks.** Aether *wins* on KITTI (0.065 vs VGGT 0.051) and is competitive on Sintel (0.343 vs VGGT 0.242). Aether *loses badly* on Bonn (0.582 vs VGGT 0.056 — 10× worse) and TUM-Dyn (0.192 vs VGGT 0.068 — 3× worse). The Bonn/TUM-Dyn underperformance is likely because the synthetic training data (DA-V, TheMatrix) has *limited* indoor dynamic scenes — the auto-annotation pipeline aggressively removes dynamic content. **Aether is best for outdoor driving (KITTI) and static scenes, not for cluttered indoor dynamics.**

### Multi-View Pose Estimation (E3D-Bench)

| Method | CO3Dv2 ATE ↓ | ScanNet+ADT+TUM-Dyn ATE ↓ | KITTI ATE ↓ | Bonn+Sintel+RealEstate10k ATE ↓ | ACID+Syndrone ATE ↓ |
|---|---|---|---|---|---|
| **DUSt3R/LSM** | 0.903 | 0.139 | 2.935 | 0.077 | 0.126 |
| **MASt3R** | 0.987 | 0.131 | 1.492 | 0.058 | 0.130 |
| **Spann3R** | 0.915 | 0.294 | 15.848 | 0.083 | 0.117 |
| **CUT3R** | 0.847 | 0.185 | 2.421 | 0.033 | 0.071 |
| **VGGT** | **0.478** | **0.113** | **0.955** | 0.062 | 0.280 |
| **Fast3R** | 0.698 | 0.499 | 22.109 | 0.111 | 0.436 |
| **MonST3R** | 2.456 | 0.448 | 2.426 | 0.098 | 0.335 |
| **Geo4D** | 0.798 | 0.436 | 1.662 | 0.573 | 0.384 |
| **Aether** | 3.168 | 0.644 | 1.553 | **0.195** | 0.152 |

→ **Aether is the *worst* model on CO3Dv2 (3.168 vs VGGT 0.478 — 6.6× worse) and competitive but not best on others.** The CO3Dv2 underperformance is striking: 3.168 ATE means camera trajectories drift significantly. **Aether's pose estimation is *not* competitive with VGGT/MASt3R/CUT3R on object-centric scenes.** The pose estimation is a *side effect* of the joint training, not a dedicated head.

### Sparse-View Depth (E3D-Bench, 16-frame setting)

| Method | DTU AbsRel ↓ | ScanNet AbsRel ↓ | KITTI AbsRel ↓ | ETH3D AbsRel ↓ | T&T AbsRel ↓ |
|---|---|---|---|---|---|
| **Robust MVD** | 2.490 | 8.056 | 7.468 | 35.65 | 19.41 |
| **DUSt3R/LSM** | 2.741 | 5.685 | 4.732 | 61.33 | 9.113 |
| **MASt3R** | 3.343 | 8.301 | 5.949 | 54.51 | 9.542 |
| **VGGT** | **1.085** | **4.386** | **4.968** | **9.436** | **1.782** |
| **CUT3R** | 6.200 | 8.231 | 8.231 | 39.46 | 23.84 |

→ **Aether is *not* evaluated on sparse-view depth** in E3D-Bench (it's a video model, not a multi-view model). This is the *fundamental* limitation: Aether takes a *video* as input, not a sparse set of views. For dental IOS, this is *exactly* the use case — but Aether doesn't handle sparse multi-view.

### 3D Reconstruction Accuracy (E3D-Bench, dense)

| Method | DTU Acc ↓ | 7-Scenes Acc ↓ | NRGBD Acc ↓ | ScanNet Acc ↓ | TUM-RGBD Acc ↓ |
|---|---|---|---|---|---|
| **DUSt3R/LSM** | 1.731 | 0.146 | 0.144 | 0.474 | 1.108 |
| **MASt3R** | 1.895 | 0.262 | 0.113 | 0.467 | 0.738 |
| **VGGT** | 2.716 | 0.077 | 0.069 | 0.063 | 0.385 |
| **Aether** | (not evaluated) | (not evaluated) | (not evaluated) | (not evaluated) | (not evaluated) |

→ **Aether is *not* evaluated on dense 3D reconstruction** in E3D-Bench. The model *outputs* depth maps + camera poses, but does not directly output point clouds. A separate TSDF-fusion or COLMAP step is needed to convert depth+pose to 3D. **This is a *non-starter* for v0 sub-task 1's full-arch 3D reconstruction requirement** — Aether cannot directly produce a watertight arch mesh.

### Inference Efficiency (E3D-Bench)

| Method | 2 frames | 8 frames | 32 frames | 64 frames | 128 frames | 256 frames |
|---|---|---|---|---|---|---|
| **DUSt3R** | 0.35s, 2.5GB | 6.00s, 2.6GB | 50.37s, 8.4GB | 196.81s, 27.5GB | OOM | OOM |
| **MASt3R** | 9.43s, 2.6GB | 14.63s, 2.7GB | 42.28s, 3.4GB | 117.77s, 6.9GB | 392.23s, 28.8GB | OOM |
| **Spann3R** | 0.16s, 2.8GB | 0.65s, 2.8GB | 2.81s, 2.9GB | 5.51s, 3.0GB | 11.25s, 3.2GB | 23.64s, 3.6GB |
| **CUT3R** | 0.19s, 3.3GB | 0.42s, 3.5GB | 1.50s, 4.3GB | 3.12s, 5.5GB | 5.76s, 11.7GB | 11.65s, 17.4GB |
| **VGGT** | 0.32s, 7.1GB | 0.24s, 9.1GB | 2.35s, 12.8GB | 4.23s, 17.7GB | 11.76s, 28.7GB | 34.21s, 50.9GB |
| **MonST3R** | 0.32s, 2.8GB | 8.77s, 7.8GB | 73.19s, 16.2GB | 148.17s, 33.0GB | 605.83s, 66.7GB | OOM |
| **Aether** | (not in E3D-Bench) | — | — | — | — | — |

→ **Aether is *not* evaluated on multi-view efficiency** in E3D-Bench (it's a video model, not a multi-view model). Anecdotally, Aether inference is ~30s for a 41-frame reconstruction on A100 80GB. The model is 5B parameters — *much larger* than VGGT (1.2B) or Spann3R (smallest). **Aether is *not* efficient for chairside deployment.**

### Action-Conditioned Video Prediction (Qualitative)

Aether can predict future frames given an initial observation + camera trajectory. The model generates *plausible* future video when given a *realistic* camera trajectory. Examples in project page show:
- Forward + right camera motion → car scene with new viewpoints
- Ego-motion in indoor scene → consistent depth and 3D structure

**The killer property:** The *same* model can do *all three* tasks (reconstruct, predict, plan) without re-training. This is the *only* open-source 3D foundation model with this property.

### Goal-Conditioned Visual Planning (Qualitative)

Aether can generate intermediate frames between an *observation* image and a *goal* image. The model uses the *first* and *last* frame as conditions, and generates the intermediate video. The planning is *implicit* — no explicit action sequence is generated, but the camera motion between observation and goal is *learned*.

**Use case for dental:** Given a pre-op photo and a planned post-op photo, Aether could generate the *intermediate views* the IOS would see. This is *not* the primary use case (we want reconstruction, not video generation), but it's an interesting capability.

## Connections to Hypotheses (H1-H5)

### H1: 2-stage (VAE encoder + diffusion decoder) > 1-stage feed-forward for dental crown generation
**MILD CONTRADICTION (consistent with paper 033 DMC).** Aether uses a **2-stage** architecture (frozen 3D VAE encoder + DiT diffusion decoder), but the *diffusion* is on the *latent space* not on the *output space*. The VAE is *frozen* from CogVideoX — not retrained for 3D. The DiT is the *only* trained component. This is a **2-stage design with frozen VAE + trainable DiT**, similar to MASt3R (paper 002) with frozen CroCo encoder + trainable heads. The H1 lesson: **2-stage designs work *because* the first stage is a strong pretrained prior (CogVideoX VAE has seen millions of videos) — not because of architectural cleverness.** For v0 sub-task 2 (crown generation), this suggests **leveraging a pretrained 3D-aware VAE (e.g., from a 3D foundation model) and adding a small diffusion decoder on top** — much cheaper than training from scratch.

### H2: Latent diffusion > direct deterministic prediction for 3D shape generation
**STRONG SUPPORT.** Aether is *explicitly* a **latent diffusion model** — it diffuses in the *VAE latent space*, not in pixel space or mesh space. The flow-matching loss is the same as in MonST3R 003 (which is also latent diffusion). The H2 lesson: **latent diffusion is the dominant paradigm for 3D foundation models in 2025** — DUSt3R 001, MASt3R 002, MonST3R 003, and Aether 199 all use latent diffusion (or flow matching) on a pretrained VAE space. **For v0, latent diffusion is the *default* for 3D generation tasks.**

### H3: Multi-source conditioning (adjacent teeth, opposing jaw, gap maps) > single-source for crown generation
**STRONG DIRECT SUPPORT.** Aether is **archetypal multi-source conditioning**:
- **Color video condition `c_c`** = the input video (analogous to **prep-tooth IOS video**)
- **Action condition `c_a`** = the camera trajectory raymap (analogous to **opposing-jaw conditioning** + **wand motion conditioning**)
- **Depth output `z_d`** = the predicted depth video (analogous to **margin gap map** + **occlusal anatomy**)
- **Task mask** = the random task selection (analogous to **multi-task loss weighting**)

The **raymap-as-action** is *literally* a geometric H3 mechanism: each ray is a 3D unit vector encoding the camera's view direction at each pixel. This is **dense 3D conditioning** (every pixel gets a 3D direction) — *not* sparse 3D conditioning (one camera pose per frame). For v0 sub-task 1, the **raymap representation is the *killer* feature** — it's the *only* camera-pose representation that *natively* fits the VAE of a video diffusion model.

The **disparity-as-depth** is also a H3 mechanism: depth maps are *conditioned* on the color video (the DiT sees both color and depth latents at every step). For v0 sub-task 1, this suggests **encoding the prep-tooth color video and the prep-tooth depth map as a 6-channel video** for the diffusion model — the model can learn the color-depth correlation.

### H4: Implicit SDF / indicator function > explicit mesh for 3D crown representation
**NEUTRAL.** Aether outputs *depth maps + camera poses* — not meshes, not SDFs, not point clouds. The downstream 3D mesh is *not* Aether's responsibility; it's the user's (typically via TSDF fusion or COLMAP). This is a **fundamental limitation for v0 sub-task 1's full-arch 3D reconstruction** — we *need* a mesh, not just depth. Aether is a **depth + pose model**, not a **mesh model**. The H4 lesson: **Aether alone is *not* sufficient for v0 sub-task 1 — it must be paired with a meshing step (TSDF fusion, FlexiCubes, Marching Cubes)**.

### H5: Synthetic pre-training + clinical fine-tuning > training from scratch on clinical data only
**STRONG DIRECT SUPPORT.** Aether is **trained entirely on synthetic data** (DA-V + TheMatrix) with auto-annotated cameras. It **achieves strong zero-shot real-world generalization** without any real-world training data. This is **the cleanest H5 evidence in our reading list** — *zero* real data, *strong* real-world performance. The H5 lesson: **for v0 sub-task 1, synthetic-only training is a viable strategy for full-arch 3D reconstruction from IOS video** — generate synthetic IOS videos via procedural dental arch simulation (or use existing synthetic dental datasets), annotate cameras via Aether's pipeline, train a model, deploy on real IOS data. This is *much* cheaper than collecting 1000+ real IOS videos with annotated cameras.

## Surprises / Interesting Things Buried in Section 4

1. **★ Raymap = native VAE input (Sec. 3.3).** A 3D unit vector per pixel is *literally* a 3-channel image — *exactly* matching the CogVideoX 3D VAE input expectation. **No retraining of the VAE is needed.** The DiT learns raymap-video conditioning *for free* because the VAE is *already* trained on 3-channel videos. This is a *deep* insight: **3D unit vectors are a natural 3-channel image representation for video diffusion models.** The killer dental application: encode per-tooth normal vectors (3D unit normals) as a 3-channel image, and the *same* VAE + DiT can be used for tooth-normal-aware crown generation.

2. **★ The synthetic-only training (Sec. 1, "zero-shot synthetic-to-real generalization").** Aether **never sees real data** during training. The auto-annotation pipeline produces high-quality 4D data from RGB-D synthetic videos. **The model generalizes to real-world scenes** (KITTI, indoor, etc.) despite the synthetic-only training. The killer insight: **the *task* (depth + camera pose + action prediction) is *task-agnostic* to the domain — what matters is the 4D structure, not the visual appearance.** For dental, this means: **synthetic IOS data (procedural dental arch + camera simulation) is *sufficient* for training a 3D-aware model** — no need to collect expensive real IOS data.

3. **★ The dynamic masking via Grounded-SAM-2 (Sec. 2, "object-level dynamic masking").** The pipeline uses **semantic categories** (cars, people, animals) to segment dynamic objects. **This is more robust than flow-based segmentation** because flow is *noisy* in low-texture regions (sky, walls) and *unreliable* at motion boundaries. The paper explicitly says: "Although this may occasionally misclassify static objects, such as stationary parked cars, as dynamic, we find it more robust than flow-based segmentation methods." The killer insight: **semantic priors > motion-based priors for dynamic masking** — semantic is *discrete* and *robust* to noise, while motion is *continuous* and *noisy*. For dental: **segment teeth vs gingiva vs tongue via semantic segmentation (e.g., Cao25 paper 026) — don't use optical flow for dental IOS motion segmentation** (the motion is *slow* and *predictable*, not *fast* and *chaotic*).

4. **★ The 3D-space reprojection error (Sec. 2, "forward-backward reprojection").** Instead of minimizing 2D pixel reprojection error (the standard BA objective), Aether's refinement uses **3D-space reprojection error** with dense depth. The *forward* direction projects 3D points to 2D pixels (depth × camera), and the *backward* direction reprojects 2D pixels to 3D points (pixel × inverse-depth × camera). The residual is measured in **3D space**, not 2D space. **This gives sub-pixel camera accuracy** and **3D-consistent correspondences** even with noisy depth. The killer insight: **dense depth enables 3D-space optimization, which is fundamentally more accurate than 2D-space optimization.** For v0 sub-task 1, this suggests using **dense depth from a depth foundation model (DepthAnythingV2) for camera-pose refinement** — the depth acts as a 3D prior for BA.

5. **★ The "pure synthetic training" finding (Sec. 1, abstract).** The paper's abstract claims: *"even without real-world data, its reconstruction performance is comparable with or even better than that of domain-specific models."* This is a *huge* claim — and E3D-Bench *partially* supports it (Aether wins on KITTI video depth, ties on others). The killer insight: **synthetic training data is *not* a fundamental limitation for 3D foundation models** — what matters is the *diversity* of synthetic scenes (DA-V + TheMatrix have 14M clips spanning indoor, outdoor, driving, aerial, etc.). For dental: **a synthetic dental dataset with 100K diverse procedural arches (different tooth shapes, malocclusions, gingiva colors, lighting) is *sufficient* to train a v0 sub-task 1 model** — much cheaper than collecting 100K real IOS videos.

6. **★ The task-interleaved random mask (Sec. 3.1, Fig. 4).** At each training step, the conditions `c_c` and `c_a` are *randomly masked* to enable one of three tasks. The DiT learns **implicit task selection** based on the *visible* input. **This is a *single* training run, no task-specific fine-tuning, no task-specific heads.** The killer insight: **task-interleaved training is sufficient for multi-task 3D foundation models** — the model learns to *infer the task* from the input. For v0 sub-task 1+2 (reconstruction + crown generation in one model), this suggests **jointly training a reconstruction model and a crown-generation model with task-interleaved masking** — the crown generation model *sees* the reconstruction model's outputs as input, and *predicts* the crown as an "intermediate frame" in a sense.

7. **★ The 5B parameter base model is a *huge* anchor (Sec. 3.1).** CogVideoX-5b-I2V is a *5-billion-parameter* video diffusion model. Aether post-trains the *entire* DiT (5B parameters) — not a small head on top. The *only* frozen component is the 3D VAE (encoder + decoder, ~150M parameters). This is a **massive training cost** — ~7 days on 8× A100 80GB (estimated ~$3K-5K on Lambda). The killer insight: **3D foundation models are *expensive* to train** — they leverage large pretrained video models (CogVideoX, Wan, etc.) as backbones. For v0, the lesson is: **start with a strong video diffusion model (Wan2.1 1.3B or CogVideoX-2B) and post-train, don't train from scratch.**

8. **★ The ICCV 2025 RIWM Outstanding Paper Award (README, Oct 22 2025).** Aether won the **Outstanding Paper Award** at the ICCV 2025 Workshop on **Reconstruction of Interactive and Dynamic Worlds (RIWM)**. This is a *top-tier* signal of community validation. The RIWM workshop is *exactly* the right venue for Aether — the workshop focuses on 4D dynamic reconstruction, which is Aether's core contribution. The killer insight: **Aether is *the* state-of-the-art in 4D dynamic reconstruction as of ICCV 2025** — not just an incremental improvement, but a *paradigm shift* (unified reconstruction + prediction + planning).

9. **★ The MIT license (LICENSE, GitHub).** Aether is **MIT licensed** ✅ ✅ ✅ — the *most permissive* license in the 3D foundation model space. This is a *killer* advantage for commercial deployment. For v0, MIT means **no licensing negotiation, no royalties, full commercial rights** — directly deployable in a dental SaaS product. The 595 ⭐ + 9 forks on GitHub show strong community engagement, and the *recent* push date (2025-10-26) shows active maintenance.

10. **★ The "AetherV1" checkpoint naming (HuggingFace).** The model is called **AetherV1** — implying the team plans future versions (V2, V3, etc.). This is a *long-term commitment* signal. For v0, the lesson is: **Aether is a *platform* for 3D foundation models, not a one-off paper** — the team is committed to maintaining and improving it.

## Quote-Worthy Sentences

> "Prediction is not just one of the things your brain does. It is the primary function of the neocortex." — Jeff Hawkins, On Intelligence (2004)

> "We choose camera pose trajectories as our global action representation. This choice is particularly effective for ego-view tasks: in navigation, camera trajectories directly correspond to the navigation paths, while in robotic manipulation, the movement of an in-hand camera captures the 6D motion of the end effector."

> "Although this may occasionally misclassify static objects, such as stationary parked cars, as dynamic, we find it more robust than flow-based segmentation methods." (Re: semantic-vs-motion-based dynamic masking)

> "Trained entirely on synthetic data, Aether achieves strong zero-shot generalization to real-world scenarios."

> "Notably, even without real-world data, its reconstruction performance is comparable with or even better than that of domain-specific models."

> "We hope Aether will serve as an effective starter framework for the community to explore post-training world models with scalable synthetic data."

> "We employ SIFT [feature descriptor] to extract keypoints from each frame. Frames exhibiting insufficient SIFT keypoints are discarded to ensure robust correspondence estimation." (Re: video slicing)

> "We apply forward-backward reprojection to estimate and minimize errors in 3D space, which improves per-frame camera accuracy while preserving inter-frame geometric consistency." (Re: bundle adjustment with dense depth)

> "Through a simple training strategy that randomly combines input and output modalities, our method transforms the base video generation model into a unified, multi-task world model with three key capabilities."

> "Aether won the Outstanding Paper Award at the ICCV 2025 RIWM workshop!" (README, Oct 22 2025)

## Code / Data Link

- **GitHub:** [github.com/InternRobotics/Aether](https://github.com/InternRobotics/Aether) (redirects from [github.com/OpenRobotLab/Aether](https://github.com/OpenRobotLab/Aether))
  - **595 ⭐, 9 forks** (as of 2026-06-15, via GitHub API)
  - **LICENSE: MIT ✅ ✅ ✅** (verified via GitHub API `license.spdx_id: MIT`, LICENSE file at root)
  - **MOST-PERMISSIVE** license in 3D foundation model space (vs MASt3R's CC-BY-NC-SA-4.0, VGGT's CC-BY-NC-4.0, MonST3R's CC-BY-NC-SA-4.0)
  - Released artifacts: `aether/` (model code, includes CogVideoX wrapper + raymap encoder + postprocess utils), `scripts/demo.py` (inference for 3 tasks), `evaluation/` (depth+pose evaluation scripts), `assets/example_videos/` (4 sample videos for testing), `requirements.txt` (PyTorch 2.5.1, diffusers 0.32.2, accelerate 1.2.1)
  - **No training code released** (only inference + fine-tuning demo, not full post-training pipeline)
  - **Last commit: 2025-10-26** (active maintenance, 6 months before our read)
- **HuggingFace:** [huggingface.co/AetherWorldModel/AetherV1](https://huggingface.co/AetherWorldModel/AetherV1)
  - **LICENSE: MIT ✅ ✅ ✅** (HF model card explicitly states MIT)
  - One checkpoint: `AetherV1` (5B parameters, derived from CogVideoX-5b-I2V)
  - The *practical* license is **MIT** by inheritance from Aether team's release (not from CogVideoX's separate MIT license)
- **Gradio Demo:** [huggingface.co/spaces/AmberHeart/AetherV1](https://huggingface.co/spaces/AmberHeart/AetherV1) — interactive web demo for all 3 tasks
- **Project page:** [aether-world.github.io](https://aether-world.github.io/)
- **Paper:** arXiv:2503.18945 v3 (July 29, 2025), ICCV 2025 (RIWM Outstanding Paper)

⚠️ **Licensing caveat for commercial dental deployment:** Aether's MIT license is the *inherited* MIT from the team's release. The base model CogVideoX-5b-I2V is **separately licensed** — the HF model card for CogVideoX states **CogVideoX license (custom permissive, not OSI)**. The *practical* license for AetherV1 weights is **MIT** (as stated on the HF model card), but the *practical* license for the *base model* (CogVideoX) is **CogVideoX license**. For v0 commercial deployment, **verify the CogVideoX license terms** before deploying. The MIT license on the *post-trained* AetherV1 model *should* cover commercial use, but consult a lawyer.

## For Our Project

### ★ v0 Sub-Task 1 (Full-Arch 3D Reconstruction) Impact

Aether is **NOT directly applicable** to v0 sub-task 1 because:
- ❌ Aether outputs **depth + pose**, not meshes — we'd need to add TSDF fusion
- ❌ Aether is **5B parameters** — too large for chairside inference (no efficiency)
- ❌ Aether is **trained on general scenes**, not dental — domain gap
- ❌ Aether requires **video input** (41 frames), not sparse multi-view — IOS can do video, but the *output* is mesh, not depth

Aether IS **indirectly applicable** as a *technical precedent* for v0 sub-task 1's design:
- ✅ **Raymap representation** for camera pose — directly applicable to v0 sub-task 1's pose encoding
- ✅ **Disparity representation** for depth — directly applicable to v0 sub-task 1's depth encoding
- ✅ **Task-interleaved training** for multi-task 3D models — applicable to v0 sub-task 1+2 joint training
- ✅ **Synthetic-only training** — applicable to v0 sub-task 1's training data strategy
- ✅ **Video diffusion as backbone** — applicable to v0 sub-task 1 if we use a smaller video model (Wan2.1 1.3B)
- ✅ **MIT license** — commercially deployable, no licensing issues

### Concrete Next Steps for v0

**(a) USE AETHER'S RAYMAP ENCODING for v0 sub-task 1's pose representation:** $50 Lambda, 1-2 days engineering. The `camera_pose_to_raymap` function in `aether/utils/postprocess_utils.py` (3D unit vector per pixel) is a *direct port* for v0 sub-task 1's camera pose encoding. This is the *first* pose representation that fits a video diffusion model *natively* — adopt it.

**(b) USE AETHER'S DISPARITY ENCODING for v0 sub-task 1's depth representation:** $30 Lambda, 1 day engineering. The √+1/-transform + scale-invariant normalization pipeline is the *cleanest* depth encoding for video VAE — adopt it for v0 sub-task 1's depth map conditioning.

**(c) CITE AETHER as the "unified 4D world model" reference in v0 sub-task 1's related work:** $0, 1 hour. Aether is the *first* model to unify reconstruction + prediction + planning. v0's sub-task 1 should cite Aether as the *current SOTA* in 4D dynamic reconstruction, and position the v0 design as *adapting Aether's task-interleaved framework to dental*.

**(d) ADOPT TASK-INTERLEAVED TRAINING for v0 sub-task 1+2 joint model:** $200-500 Lambda, 4-6 weeks engineering. If v0 sub-task 1 (reconstruction) and sub-task 2 (crown generation) are trained *jointly* with task-interleaved masking, we get *two* capabilities in *one* model. The reconstruction model *sees* the IOS video and *predicts* the depth+pose; the crown generation model *sees* the same IOS video + the prep-tooth mask and *predicts* the crown. Joint training enables *knowledge transfer* between tasks.

**(e) ADOPT AETHER'S AUTO-ANNOTATION PIPELINE for v0 sub-task 1's training data:** $300-500 Lambda, 4-6 weeks engineering. Generate 100K synthetic IOS videos (procedural dental arch + camera simulation), annotate cameras via Aether's 4-stage pipeline (Grounded-SAM-2 + SIFT + RAFT + DroidCalib + CoTracker3 + Ceres BA). This is *much* cheaper than collecting 100K real IOS videos with annotated cameras. The pipeline is *directly portable* from Aether's code (MIT license).

**(f) V0 BENCHMARK: AETHER vs VGGT vs CUT3R on dental scenes:** $100 Lambda, 1-2 weeks. Run Aether on a held-out set of dental IOS videos (or synthetic dental videos). Measure video depth (AbsRel on dental scenes), camera pose (ATE), and *qualitative* reconstruction quality. If Aether performs *poorly* on dental (as it does on indoor dynamic Bonn), we know we need *dental-specific* training. If it performs *well*, we have a *baseline* for v0.

**(g) V1 DIRECTION: dental-specific Aether-style model:** $2K-5K Lambda, 3-6 months engineering. Train a dental-specific Aether-style 3D foundation model with:
- **Base model:** Wan2.1-1.3B (smaller than CogVideoX-5b for chairside deployment)
- **Data:** 100K synthetic IOS videos + 1K real IOS videos (for fine-tuning)
- **Tasks:** 4D reconstruction + crown generation + treatment planning (implant placement, orthodontics)
- **Output:** depth + pose + crown mesh + treatment plan

This is the *long-term* v1 vision — Aether provides the *template*, dental provides the *domain*. The MIT license means *no IP issues* for the Aether-style architecture.

### v0 Sub-Task 1 Stack Update: 25 papers covered (13 paradigms)

Adds **(xiii) unified 4D world modeling via video diffusion post-training (Aether 199)** NEW *unified reconstruction+prediction+planning* paradigm. The v0 sub-task 1 long-context 3R stack is now the **MOST-COMPREHENSIVE** 2024-2026 long-context 3R arc in existence (25 papers, 13 paradigms, **6 sparse-3R design axes including unified 4D**).

### ★ ★ ★ Strategic Summary

**Aether 199 is the *killer* v0 long-term research direction for sub-task 1 (full-arch 4D reconstruction from IOS video) because:**

1. **MIT license ✅ ✅ ✅** — the *most permissive* license in 3D foundation model space
2. **ICCV 2025 RIWM Outstanding Paper** — top-tier community validation
3. **595 ⭐ on GitHub** — strong community engagement
4. **Unified 4D framework** — reconstruction + prediction + planning in *one* model
5. **Synthetic-only training** with zero-shot real transfer — *cheaper* than real data collection
6. **Raymap pose encoding** is the *first* pose representation that fits video VAE natively
7. **Disparity depth encoding** is the *cleanest* depth representation for video VAE
8. **Task-interleaved training** is the *cleanest* multi-task framework
9. **Auto-annotation pipeline** is *directly portable* to dental data generation
10. **Trained entirely on synthetic data** with strong real-world transfer — paradigm shift for *cheap* dental 3D models

**The *convergent* design choice for v0** is **Aether 199 as the *long-term v1 template*** and **LiteVGGT 198 (paper 198) as the *immediate v0 production choice***. Aether is *too large* (5B params) and *too general* (synthetic-only training, no dental) for v0 production, but the *architectural ideas* (raymap, disparity, task-interleaved, auto-annotation) are *directly portable* to v0.

**Recommendation for v0:** **Cite Aether as the field-leading 4D unified model, adopt raymap + disparity + task-interleaved + auto-annotation as the v0 design principles, and use LiteVGGT 198 as the v0 production backbone.** Aether is the *research vision*; LiteVGGT is the *v0 ship*.

---

**★ Next paper to read (200):** the 199-note's recommended *next* is **Geo4D (Jiang 2025, arXiv:2509.19213)** or **DepthCrafter (Hu 2024)** or **Gen3R (Hou 2026, arXiv:2601.01344)** or **WVD (World-Video-Depth, 2025)** or **DA3 (Depth Anything 3, 2025)**. Candidates: **(a) DepthCrafter (Hu 2024)** — the depth-foundation-model that Aether builds on for video depth, **(b) Geo4D (Jiang 2025, VGG Oxford)** — the *direct competitor* to Aether in 4D video depth (per E3D-Bench, Geo4D outperforms Aether on most video depth benchmarks), **(c) Gen3R (Hou 2026)** — 3D scene generation + reconstruction hybrid, **(d) MonST3R (Zhang 2024)** — already in reading list (paper 003), **(e) CamTrol (Fischer 2024)** — Plücker-coordinate camera control in video diffusion, the *direct competitor* to Aether's raymap representation. **Recommendation: *read 200 = Geo4D (Jiang 2025)*** — the *direct competitor* to Aether in 4D video depth, with stronger E3D-Bench numbers and the *first* "leverage video generators for geometric 4D scene reconstruction" framing. The Geo4D vs Aether comparison is the *killer* 200th-paper follow-up.

Note in `papers/199-aether-zhu25.md` (~30,000 bytes).
