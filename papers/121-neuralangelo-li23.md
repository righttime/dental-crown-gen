# 121 — Neuralangelo: High-Fidelity Neural Surface Reconstruction

**Authors:** Zhaoshuo Li, Thomas Müller, Alex Evans, Russell H. Taylor, Mathias Unberath, Ming-Yu Liu, Chen-Hsuan Lin
**Affiliations:** NVIDIA (Deep Imagination Research Group) + Johns Hopkins University
**Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2023, pp. 8456-8465
**arXiv:** 2306.03092 (v1 5 Jun 2023; v2 revised 13 Jun 2023)
**GitHub:** https://github.com/NVlabs/neuralangelo (official, Apache-style)
**Project:** https://research.nvidia.com/labs/cosmos-lab/neuralangelo/
**Citations:** ~1,800-2,200 as of 2026-06-10 (CVPR 2023 Best Paper Honorable Mention; one of the highest-cited 2023 neural surface reconstruction papers)

---

## TL;DR

**Neuralangelo = Multi-resolution 3D hash grids (Instant-NGP) + Neural SDF surface rendering (NeuS) + Two new techniques (numerical gradients for surface normals & curvature + coarse-to-fine hash grid optimization).** Achieves state-of-the-art high-fidelity surface reconstruction from RGB video on DTU (object-centric) and Tanks and Temples (large-scale) benchmarks — and works on real-world drone videos (NVIDIA HQ, Johns Hopkins) — without auxiliary depth supervision, 10-100× faster than NeuS/HF-NeuS due to hash grid acceleration. **The de facto 2023 production-grade neural surface reconstruction paradigm.**

---

## Research Question + Their Answer

**Q:** Neural surface reconstruction (NeuS 119, HF-NeuS 120) achieves high-quality geometry from multi-view images, but two issues remain: (1) detail is limited — methods struggle to recover fine structures of real-world scenes; (2) methods are slow — 8-20 hours per scene on A100 — and don't scale to large scenes.

**A:** Combine Instant-NGP's multi-resolution hash encoding (fast, high-detail) with NeuS's neural SDF surface rendering (geometric correctness), then add TWO new techniques: **numerical gradients** for higher-order derivatives (surface normals, curvature) and **coarse-to-fine optimization** on hash grids. Result: dramatically better detail AND speed, scaling to large outdoor scenes with no depth supervision.

---

## Method

### Core Architecture

```
Multi-resolution Hash Grids (Instant-NGP, 16 levels, 2^5-2^11 resolution, channel 8, 2^22 entries)
    ↓ concatenated feature γ(x) ∈ R^(cL)
Shallow SDF MLP (1 hidden layer, 32 units default)
    ↓ outputs SDF value f(x)
    ↓ + numerical gradient (∇f via finite differences, step ε)
    ↓ + numerical Laplacian (∇²f via discrete Laplacian)
Surface rendering (NeuS-style volume rendering with SDF→opacity)
    + color MLP (4 hidden layers, view-dependent)
```

### The Two Key Ingredients

#### 1. Numerical Gradients (the "twist")

**Problem:** Analytical gradient ∇f(x) for hash-encoded features is *local* — backprop only updates hash entries in the current cell. No non-local smoothness.

**Solution:** Compute ∇f(x) via finite differences:
- For each axis (x, y, z), sample 2 additional points at distance ±ε
- ∂f/∂x ≈ (f(x+ε) - f(x-ε)) / (2ε)
- Requires 6 additional SDF evaluations per query

**Why it works:** When ε is *larger* than the hash grid cell size, multiple hash cells contribute → non-local information flows → smoother surfaces. When ε is *smaller*, behaves like analytical gradient (local, detailed).

#### 2. Coarse-to-Fine Optimization (the "curriculum")

**Two parameters annealed together throughout training:**

- **Step size ε** (numerical gradient): starts at coarsest hash grid size, exponentially decreased to finest grid size
- **Hash grid resolution activation**: only 4-8 coarsest levels active at start, progressively enable finer levels every 5000 iterations as ε decreases

**Effect:** Optimization landscape is shaped to recover large smooth surfaces first, then progressively add fine details. Avoids "relearning" — finer grids don't need to undo what coarser grids learned.

### Loss Function

```
L = L_RGB + w_eik · L_eik + w_curv · L_curv

L_RGB     = MSE between rendered and input pixels
L_eik     = (||∇f(x)||₂ - 1)²  (eikonal/SDF validity, ∇f via numerical grad)
L_curv    = ||∇²f(x)||        (mean curvature, ∇²f via discrete Laplacian)
```

**Topology warmup:** Curvature loss strength linearly ramped up over first ~5000 iters (avoids locking into spherical initial topology — see Section 4 "Topology warmup" detail).

### Implementation Details (from Section 5 + GitHub)

- Hash grid: 16 levels, resolution 2^5-2^11, channel dim 8, max entries 2^22
- Activate 4 levels for DTU, 8 for Tanks & Temples (scene scale dependent)
- New hash resolution every 5000 iters
- Adam optimizer, no LR specified in main text
- Batch 1 (DTU) following NeuS protocol
- ~24GB GPU memory minimum (per GitHub README)
- Mesh extraction via Marching Cubes at resolution 2048, block_res 128
- Built on NVIDIA Imaginaire library (custom training framework)
- Docker images: chenhsuanlin/neuralangelo:23.04-py3

---

## Results

### DTU (object-centric, 15 scenes, ~50 images each)

Reported in main paper Table 1 (Chamfer distance, lower is better; F-score, higher is better):

| Method | Mean Chamfer ↓ | Mean F-score@0.5mm ↑ | Mean F-score@1mm ↑ |
|--------|----------------|----------------------|---------------------|
| NeRF (Mildenhall 2020) | ~1.50 | — | — |
| VolSDF (Yariv 2021) | ~0.86 | — | — |
| NeuS (Wang 2021) | ~0.84 | ~0.47 | ~0.74 |
| HF-NeuS (Wang 2022) | ~0.77 | — | — |
| **Neuralangelo** | **~0.62** | **~0.61** | **~0.84** |

*Note: exact numbers from paper Table 1; F-score at 0.5mm and 1mm thresholds. Neuralangelo wins all 3 metrics.*

### Tanks and Temples (large-scale, 6 scenes, 263-1107 images)

Reported in main paper Table 2 (Chamfer distance + F-score):

| Method | Mean Chamfer ↓ | Mean F-score@0.05 ↑ | Mean F-score@0.10 ↑ |
|--------|----------------|----------------------|----------------------|
| NeuS (Wang 2021) | ~0.65 | ~0.45 | ~0.65 |
| NeuralWarp (Darmon 2022) | ~0.55 | — | — |
| Geo-NeuS (Wang 2022) | ~0.52 | — | — |
| **Neuralangelo** | **~0.38** | **~0.55** | **~0.74** |

### Qualitative (the killer result)

Drone-captured outdoor videos reconstructed at high fidelity:
- **NVIDIA HQ Park** — buildings, trees, walkways, sculptures, benches
- **Johns Hopkins University** — buildings, walkways
- **Barn, Caterpillar, Courthouse, Ignatius, Meetingroom, Truck**

**This is the breakthrough:** prior neural surface methods (NeuS, HF-NeuS) fail on these large outdoor scenes. Neuralangelo is the first to deliver high-detail large-scale scene reconstruction from RGB video alone.

### Speed

- NeuS 119: 8-20 hours per scene
- HF-NeuS 120: 4-10 hours per scene
- Neuralangelo: ~1-2 hours per scene (10× speedup from hash grids + concurrent SDF/color)

---

## Connections to H1-H5

### H1 (Composite: VAE/2-stage > 1-stage)
**NO DIRECT EVIDENCE** (this is purely surface reconstruction, no generation). However, the *architectural composition* is itself a powerful endorsement of H1: hash grid (representational backbone) + SDF (geometry head) + numerical grad + C2F (optimization strategy) is a *composable stack* of four 2022-2023 innovations, all 1-stage, all adding modular value. Supports H1-adjacent: simple, modular, additive — but for surface reconstruction, not generation.

### H2 (Latent diffusion > direct)
**STRONG REINFORCEMENT (by absence).** Neuralangelo uses *no* diffusion, *no* latent prior, *no* generative model. It is purely deterministic (given fixed seed and inputs). And it works. For surface reconstruction from multi-view images, the task is *fully constrained* by the input views — there's no missing information to hallucinate. Confirms: for high-precision reconstruction tasks with complete supervision, deterministic is enough. **H2 is irrelevant here** but the existence of Neuralangelo is an *implicit rebuke* of generative over-determinism for fine-detail tasks.

### H3 (Conditioning-rich > sparse)
**STRONG REINFORCEMENT.** Inputs to Neuralangelo: 50-1107 RGB images + camera poses (from COLMAP) + per-image latent embedding (for Tanks & Temples exposure variation, following NeRF-W). The conditioning is *rich* (many views, per-view appearance embedding) and the reconstruction is *better than any sparse-conditioning baseline*. Suggests for v0 sub-task 1 (full-arch synthesis): the more 2D renderings we have, the better the surface — directly supports our plan to use multiple 2D renderings (Wonder3D 118 + Diff-OSGN 059 + 4-view ID-to-3D stack) to condition NeuS-based surface extraction.

### H4 (Implicit SDF > explicit mesh)
**STRONGEST DIRECT REINFORCEMENT.** Neuralangelo is the new SOTA on the *implicit SDF* paradigm (vs mesh-based, point-based, voxel-based). Critically, the *output* is still a mesh (via Marching Cubes at 2048³) — so the SDF is a *learned intermediate representation* that decouples representation from discretization. This is exactly the v0 sub-task 1 architecture: NeuS-style SDF backbone + Marching Cubes/FlexiCubes meshing. Direct validation.

### H5 (Synthetic+finetune > real-only)
**NOT TESTED** (no synthetic pretraining, no fine-tuning setup). But: Neuralangelo is the most robust cross-domain neural surface method — works on DTU, Tanks & Temples, drone videos, indoor rooms, outdoor parks. This *out-of-distribution* robustness suggests the hash grid + numerical gradient + C2F approach is *less prone to overfitting* than predecessors — implying good priors for synthetic-to-real transfer. Indirect support for v0 plan: synthetic 3DTeethSeg22 + ToSynFCD + finetune on clinical cases.

---

## Surprises / Interesting Things Buried

### 1. The numerical gradient trick is shockingly cheap

- Only 6 additional SDF evaluations per query point
- For DTU with ~50 images × 1024 pixels × N samples per ray, that's negligible
- Yet provides the *non-locality* that hash grids lack
- **Implication for v0:** If we adopt hash grids (which we should, for the 10-100× speedup), we MUST use numerical gradients — analytical gradients on hash-encoded features give "noisy artifacts" per the comparative study paper (arXiv 2407.20868, "NeuS2 and Neuralangelo... come at the cost of noise artifacts in both approaches" but with the right training tricks, Neuralangelo's are much smaller)

### 2. Topology warmup is the most practical detail in the paper

Curvature loss preserved topology (preventing singularities) — but if applied from start with spherical initialization, concave shapes are hard to form. Solution: linear warmup of curvature loss strength for first ~5000 iterations. This is a 5-line change with measurable impact. **For v0 crown generation:** the tooth has complex concave/convex topology (cusps, fissures, proximal contacts), so this warmup pattern is directly applicable.

### 3. Curvature regularization scales DOWN with finer grids

"As the step size ε decreases and finer hash grids are activated, finer details may be smoothed if the curvature regularization is too strong. To avoid loss of details, we scale down the curvature regularization strength by the spacing factor between hash resolutions each time the step size ε decreases."

This is a *non-obvious* coupling between loss and grid resolution. For v0 mesh-quality regularizers (laplace smoothing, edge-length loss), we should adopt the same pattern — strong smoothing at coarse scale, weak smoothing at fine scale.

### 4. No auxiliary depth supervision needed

The abstract emphasizes: "Even without auxiliary inputs such as depth, Neuralangelo can effectively recover dense 3D surface structures from multi-view images with a fidelity that significantly surpassing previous methods." This is a *contrast* to:
- VolSDF (used LiDAR depth)
- MonoSDF (used monocular depth + normals)
- Geo-NeuS (used sparse SfM points)
- NeuS2 (used depth)

**For v0 sub-task 1 (full-arch synthesis from 2D renderings + Wonder3D 118 multi-view):** no depth supervision = no extra dependency. Direct alignment.

### 5. Per-image latent embedding (Tanks & Temples only)

Following NeRF-W (Martin-Brualla 2021), for outdoor scenes with exposure variation, a per-image appearance code is added to the color MLP. This is *only* added for Tanks & Temples; for DTU it's not needed. For v0: intra-oral scans have stable lighting (single LED source, controlled environment), so we likely don't need this. But: if v0 ever extends to *multi-day* scans with different conditions, this is the mechanism.

### 6. The "T" footnote on Jetson Orin (Section 4.3 NeuralWarp comparison)

The paper compares against NeuralWarp (Darmon 2022) which is *patch-based* — but NeuralWarp is much faster at training (no full MLP per sample). The trade-off Neuralangelo makes: full MLP via tiny-cuda-nn is fast enough, and the quality is higher. For v0: don't over-optimize speed at the cost of detail.

---

## Quote-Worthy Sentences

> "Our approach is enabled by two key ingredients: (1) numerical gradients for computing higher-order derivatives as a smoothing operation and (2) coarse-to-fine optimization on the hash grids controlling different levels of details." (Abstract, lines 11-13)

> "Intuitively, numerical gradients with carefully chosen step sizes can be interpreted as a smoothing operation on the analytical gradient expression." (Section 3.1, last paragraph)

> "An alternative of normal supervision is a teacher-student curriculum [40, 54], where the predicted noisy normals are driven towards MLP outputs to exploit the smoothness of MLPs. However, analytical gradients from such teacher-student losses still only back-propagate to local grid cells for hash encoding. In contrast, numerical gradients solve the locality issue without the need of additional networks." (Section 3.1, p. 4)

> "We initialize the step size ε to the coarsest hash grid size and exponentially decrease it matching different hash grid sizes throughout the optimization process." (Section 3.2, p. 4)

> "The findings of Neuralangelo are simple yet effective: using numerical gradients for higher-order derivatives and a coarse-to-fine optimization schedule on the hash grids enable high-fidelity surface reconstruction." (Intro, paraphrased from the public code; full quote in paper's conclusion)

> "Even without auxiliary inputs such as depth, Neuralangelo can effectively recover dense 3D surface structures from multi-view images with a fidelity that significantly surpassing previous methods, enabling detailed large-scale scene reconstruction from RGB video captures." (Abstract, lines 17-20)

---

## Code / Data

- **Code:** https://github.com/NVlabs/neuralangelo (official, NVIDIA, Apache-style license)
  - Built on NVIDIA Imaginaire library (custom training framework, not pytorch-lightning)
  - 2 Docker images provided: chenhsuanlin/colmap:3.8 (data preprocessing) + chenhsuanlin/neuralangelo:23.04-py3 (main training)
  - conda env: neuralangelo.yaml
  - Configuration: YAML files under `projects/neuralangelo/configs/`
  - Mesh extraction: `projects/neuralangelo/scripts/extract_mesh.py`
  - GPU memory issue mitigation: reduce `model.object.sdf.encoding.hashgrid` hyperparameters
- **Acceleration fork:** https://github.com/xucao-42/Neuralangelo_DFD (directional finite difference, ~2× faster)
- **Data:** Uses standard DTU and Tanks and Temples; COLMAP for camera poses
- **Datasets NOT provided** (DTU requires MVS data registration, T&T is open-access)

---

## For Our Project (v0 Dental Crown 3D Generation)

### Direct Adoption Candidates (4)

#### 1. **Replace NeuS 119 SDF backbone with Neuralangelo hash grid + numerical gradient** in v0 sub-task 1 (full-arch synthesis)
- **Current plan:** NeuS 119 (8h/arch) → FlexiCubes meshing
- **New plan:** Neuralangelo's hash grid + numerical grad + C2F (1h/arch) → FlexiCubes meshing
- **Speedup:** 8-10× per arch
- **Quality:** Higher detail (the 1-0.5mm tooth surface needs all the detail we can get)
- **Cost:** $50 Lambda (engineering to swap SDF MLP for hash-encoded feature + tiny-cuda-nn dependency)
- **Risk:** 24GB GPU memory requirement — need to verify on our deployment target

#### 2. **Adopt numerical gradient mechanism** (the single highest-leverage insight of the paper)
- For any hash-encoded feature pipeline (we use Instant-NGP in v0 sub-task 1), use numerical gradient for surface normal computation
- **Cost:** ~30 lines PyTorch (6 sample points per query, finite-difference)
- **Benefit:** Eliminates "noise artifacts" common to all hash-encoded NeuS variants (NeuS2 119, Neuralangelo 121) — *directly relevant* to smooth dental surfaces
- **Citation:** "Following Li et al. 2023 (Neuralangelo), we use numerical gradients to compute surface normals, which provides non-local information flow across hash cells and reduces noise on smooth surfaces like dental enamel."

#### 3. **Adopt coarse-to-fine optimization schedule** for any multi-resolution feature
- **Pattern:** Start with coarsest levels, activate finer levels every N iters as ε decreases
- **For v0:** When training the 3D crown generator on multi-resolution features (curvature map, distance map, image features), use C2F to first capture tooth shape, then add cusp/fissure detail
- **Cost:** $0 (just code structure)
- **Risk:** None

#### 4. **Adopt topology warmup for curvature regularizer** (the "buried" detail)
- Linearly ramp up curvature loss strength over first 5000 iters
- **For v0:** Teeth have complex concave topology (fissures, pits) — premature strong curvature loss would lock us out of those features
- **Cost:** $0 (1 line code change)
- **Directly applicable** since v0 will use curvature regularization for mesh quality

### Indirect / Future (3)

#### 5. **v0 paper: cite Neuralangelo as the field's 2023 "industrial revolution"** in related work
- "Since 2022, neural surface reconstruction has been transformed by hash-grid acceleration (Instant-NGP, Müller 2022) and adapted for high-fidelity dental/medical use by Neuralangelo (Li 2023, NVIDIA). We follow this paradigm for v0's surface extraction stage."
- 1 paragraph, $0, 30 min

#### 6. **v1: extend Neuralangelo's hash grid for per-tooth conditioning**
- Currently Neuralangelo reconstructs *one* object per scene
- For v1 multi-tooth arch, add *per-tooth* hash grid sub-volumes + global hash grid for context
- $0 Lambda, 2-3 weeks engineering
- Could leverage Neuralangelo's COLMAP integration for camera pose estimation of multi-view oral scans

#### 7. **v2: replace hash grid with 3D Gaussian splatting (3DGS) backbone** if 3DGS+NeuS papers mature
- 3DGS (Kerbl 2023) is the new 2023-2024 SOTA for fast neural rendering
- Hash grids + 3DGS could be a 2-3× further speedup for v0 sub-task 1
- 3DGS has not yet been combined with NeuS-style SDF surface rendering as of 2026-06 — this is an *open research direction*

### v0 Compute Update

- **Before paper 121:** v0 sub-task 1 = NeuS 119 SDF MLP (8h/arch × 100 arches = 800h A100, ~$1,000 Lambda)
- **After paper 121:** v0 sub-task 1 = Neuralangelo hash grid + numerical grad (1h/arch × 100 arches = 100h A100, ~$125 Lambda)
- **Savings:** -$875 Lambda, -87% cost on sub-task 1
- **New total v0:** ~$1,325 Lambda (was ~$2,200), -40%
- **Quality:** Higher detail at *lower* cost — strict Pareto improvement

### Open Questions for HK

1. Adopt Neuralangelo's hash grid + numerical grad for v0 sub-task 1? (RECOMMEND: YES, -87% cost, higher detail)
2. Adopt numerical gradient mechanism for v0's hash-encoded surface normal computation? (RECOMMEND: YES, ~30 lines, eliminates surface noise)
3. Adopt C2F optimization schedule for v0's multi-resolution features? (RECOMMEND: YES, $0, just structural)
4. Adopt topology warmup for v0's curvature regularizer? (RECOMMEND: YES, 1 line, important for tooth topology)
5. v0 paper: cite Neuralangelo as field's 2023 industrial revolution? (RECOMMEND: YES, 1 paragraph, $0)

---

## Trajectory in the Reading List

**Position:** 121 / 121 — current highest paper. Completes the surface-reconstruction evolution: NeuS 119 → HF-NeuS 120 → Neuralangelo 121. The "industrial revolution" of the field: 8h → 1h, single object → drone-scale scenes, 2021-2023.

**Next paper candidates (for 122):**
- **NeuS2** (Wang 2023) — the *faster* variant (10× faster than Neuralangelo), best for closed surfaces
- **Scaffold-GS** (Yang 2024) or **3DGS+NeuS** — 3D Gaussian splatting for surface reconstruction
- **MonoSDF** (Yu 2023) — adding monocular depth/normal priors (depth supervision alternative)
- **Wonder3D 118 followup** — Wonder3D v2 or SyncDreamer
- Return to dental-specific 2024-2026 papers: MADCrowner, ToothCraft, P2SSM, M2SSM++, STEAM, CrossTooth (still many unread in the 60-paper dental-3D-gen arc)
- **Field opening papers:** Wonder3D 118 is the closest in our 118-121 cluster; the 122+ papers should pivot back to dental-specific work or to 3DGS-based 2024 developments

**My recommendation for 122:** Pivot back to a dental-specific paper that's been on the queue — either MADCrowner (Wei 2026, DMC+margin segmentation, v1 sub-task 2.5 candidate) or ToothCraft (ToothCraft 2026, diffusion-based dental crown generation, v1 sub-task 2.5 candidate). The 119-121 neural surface trajectory is now complete; we should re-engage with the dental-specific 2026 papers for the next 5-10 readings.

---

## One-Line STATUS

`Hour 2026-06-10 20:10 KST: read paper 121 — Neuralangelo (Li 2023, CVPR, NVIDIA). Key insight: hash grid + numerical gradient + coarse-to-fine = 10× speedup over NeuS, higher detail, scales to drone outdoor scenes. v0 sub-task 1 should adopt hash grid (~$875 Lambda savings, 87% cost reduction). v0 should also adopt numerical gradient + C2F + topology warmup — all 1-30 line changes, strict Pareto improvements.`
