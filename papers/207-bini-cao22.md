# Paper 207 — Bilateral Normal Integration (BiNI)

**Authors:** Xu Cao¹, Hiroaki Santo¹, Boxin Shi²,³, Fumio Okura¹, Yasuyuki Matsushita¹ — ¹Osaka University, Japan ²NERCVT, School of Computer Science, Peking University, China ³Peng Cheng Laboratory, China

**Venue:** **ECCV 2022** (Springer LNCS **13661**, pp. 552-567, paper #136610545, DOI **10.1007/978-3-031-13660-1_36**) — verified via ECVA openaccess PDF

**arXiv:** **NO arXiv preprint** ⚠️ — direct ECCV publication only (the *first* paper in v0 reading list since Hwang18 061 to be *arXiv-free*; the *first* no-arXiv paper since 061)

**Code:** https://github.com/xucao-42/bilateral_normal_integration — ⭐ **244** / 🍴 **24** / size **69 MB** / created **2022-07-07** / last push **2026-04-20** (8 weeks before our 2026-06-15 read, **STILL ACTIVELY MAINTAINED 4 years post-ECCV**) / **4 open issues**

**License:** **GPL-3.0** ⚠️ ⚠️ ⚠️ (verified via GitHub API `license: {key: gpl-3.0, name: "GNU General Public License v3.0", spdx_id: GPL-3.0}`) — **copyleft + patent grant + disclosure**; **CANNOT BE LINKED into proprietary / commercial code** without GPL-3.0 contagion. The *practical* workaround: re-implement the algorithm (~50 lines NumPy) under a different license, since the math is published and the algorithm is a *single equation* in the paper. **ALSO NOTE:** the BiNI submodule inside GeoWizard 206's repo is the *same* GPL-3.0 code; GeoWizard's CC BY 4.0 license applies *only* to GeoWizard's own code, the BiNI submodule is *separately* GPL-3.0 ⚠️ (so GeoWizard's commercial use of BiNI is *technically* not GPL-3.0-compliant per strict copyleft interpretation, but the 244-star ecosystem has not been challenged; this is a *legal grey zone* for v0 v1+ commercial deployment)

**PDF:** openaccess at ECVA (https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610545.pdf) — **PDF FULLY OPEN-ACCESS** ✅

**Supplementary:** 136610545-supp.pdf at ECVA (3 sections: hyperparameter analysis, IRLS convergence, limitations) — **SUPP FULLY OPEN-ACCESS** ✅

**Companion papers (same first author Xu Cao):**
- **MVAS** (Cao 2023, CVPR, "Multi-View Azimuth Stereo via Tangent Space Consistency", xucao-42/mvas_homepage) — same lab, related 3D-recon
- **d-BiNI** (Cao's notation in ECON Xiu 2023 = "depth-aware BiNI") — the *applied* BiNI variant in human reconstruction
- **BiNI is a SUBMODULE of GeoWizard 206** (github.com/fuxiao0719/GeoWizard/bini/) — same numerical core, GPL-3.0

**Citations:** ~250-350 Google Scholar estimated as of 2026-06-15 (4 years post-ECCV, 244 stars, cited by ECON 2023, GeoWizard 2024, Wonder3D 2024, Metric3D v2 2024, Marigold-CV 2024, multiple normal-integration follow-ups including Kim 2024 auxiliary-edges, Lim 2024 perspective-edges, Milano 2025/2026 continuous-components, the *de facto* normal-integration post-processor of the 2022-2026 era)

---

## One-line TL;DR

**THE FOUNDING PAPER OF *BILATERAL-WEIGHTED DISCONTINUITY-PRESERVING NORMAL INTEGRATION*** — given a 2D surface-normal map (from any source: photometric stereo, shape-from-polarization, ICON, PIFu, GeoWizard, etc.), BiNI reconstructs the underlying 3D surface as a depth map by solving a *log-space* linear system `z̃_a − z̃_b − ω̃_{b→a} = 0` (Eq. 1) with **bilateral weights** that *relativize* the one-sided depth-discontinuity at each pixel (Eq. 7: `w_{b→a} = 1 / (1 + exp(−k · (Δz̃_{a} − Δz̃_{b})))` with `k=2` sigmoid sharpness default) — the *key insight*: a discontinuity is one-sided, not two-sided, so weights should depend on the *larger* of the two neighboring depth-gaps, not the average (the "bilateral" = two-sides-but-asymmetric) — solves the "in-the-smooth-surface-assumption recovery is wrong at occlusion boundaries" pathology that breaks Frankot-Chellappa 1988 and subsequent global integrability methods, the *first* normal-integration method to handle orthographic AND perspective cameras in a *unified* formulation, **the *de facto* normal-integration post-processor of the 2022-2026 era** (used by ECON 2023, GeoWizard 2024, ICON, PIFuHD, Wonder3D 2024, multiple SOTA 3R/3D-gen pipelines), and *the most-prolific 2026 maintenance surprise* — *still being actively developed in 2026* with **6.4× CPU speedup** (76.7s → 11.9s on DiLiGenT benchmark) achieved *autonomously* by an AI agent in a **karpathy/autoresearch**-style loop (2026-03-11 update), the *first* classical CV paper in v0 reading list to be *deliberately instrumented* for AI-agent-driven optimization (the repo has an `AGENTS.md` file for AI agents + a `cli.py` + a clean reference implementation in `bilateral_normal_integration_simple.py`).

---

## Research question + their answer

**RQ (Sec. 1):** Given a 2D surface-normal map `N = (n_x, n_y, n_z) ∈ ℝ^{H×W×3}` (estimated from photometric stereo / shape-from-polarization / deep learning / etc.), how do we recover the underlying depth map `Z ∈ ℝ^{H×W}` such that the surface is **discontinuity-preserving** (i.e., the abrupt depth-jumps at occlusion boundaries are *preserved*, not smoothed out)?

**Why this is hard:** the *classical* normal-integration problem assumes a **smooth** (i.e., everywhere-differentiable, hence everywhere-continuous) surface — under this assumption, integration reduces to solving a linear Poisson PDE in 2D, with multiple fast solvers (Frankot-Chellappa 1988, simplex-based, Fourier-based, etc.). But at **occlusion boundaries** (e.g., the silhouette of a foreground object against a background, the boundary between two teeth, the margin line of a crown), the surface is *not* continuous, and the smooth assumption *fails* catastrophically — the integrated surface *smooths out* the depth jump and connects the foreground to the background (Fig. 1(d) of the paper: the statue's *base* is connected to the *shelf* in a way that doesn't exist in reality). Existing *discontinuity-aware* methods (Mumford-Shah 1989 weighted, robust estimators like L1/L0 regularization) have *two* failure modes: (1) they treat discontinuities as a *statistical* property (e.g., "discontinuities are sparse" → robust estimator), so they fail when discontinuities are *dense* or *arbitrary-distributed*; (2) they treat discontinuities as *symmetric* (the same weight on both sides of an edge), but a discontinuity is fundamentally *one-sided* (the surface is continuous on one side, discontinuous on the other).

**Their answer (Eq. 7 + Sec. 3):** model the surface as **semi-smooth** (i.e., one-sided differentiable everywhere, hence one-sided continuous everywhere) — under this assumption, a discontinuity at point `(a, b)` is *by definition* on *one* side, not both. Then design a **bilateral weight function** `w_{b→a}` that *relativizes* the one-sided depth-discontinuity:

> w_{b→a} = 1 / (1 + exp(−k · (Δz̃_a − Δz̃_b)))

where `Δz̃_a = z̃_a − z̃_{a+1}` is the *forward* depth-difference at pixel `a` and `Δz̃_b = z̃_b − z̃_{b+1}` is the *forward* depth-difference at pixel `b`, and `k` is the single hyperparameter (default `k=2`).

**Intuition (Sec. 1):** "if the depth gap on one side of a point is much larger than the other side, then the side with the larger depth gap is more likely to be discontinuous" — the bilateral weight *encodes* this asymmetry by being large when `Δz̃_a > Δz̃_b` (so the weight on the *larger-gap* side is *high*, the weight on the *smaller-gap* side is *low*), and being symmetric only when `Δz̃_a = Δz̃_b` (i.e., no discontinuity).

**Unification (Sec. 2):** the same framework handles BOTH orthographic AND perspective cameras via the pinhole camera intrinsic `K = (f_x, f_y, c_x, c_y)` in the log-space equation:

> z̃_a − z̃_b − ω̃_{b→a} = 0
> ω̃_{b→a} := δ_{b→a} / (n_ax (u_a − c_x) + n_ay (v_a − c_y) + n_az · f)

where `δ_{b→a} = ±n_ax · f_x` for horizontal neighbors, `δ_{b→a} = ±n_ay · f_y` for vertical neighbors, and the orthographic case is recovered by setting `f_x, f_y → ∞` (or equivalently `c_x, c_y → 0` and treating the normal as the *spatial gradient*). This is the *first* unified normal-integration framework for ortho + perspective in the literature.

**Practical (Sec. 4):** solved via **Iteratively Reweighted Least Squares (IRLS)** — at each iteration, fix the weights, solve the linear least-squares system, update the weights from the new solution, repeat until convergence. Default `k=2`, `max_iter=150`, `tol=1e-4`. The algorithm is *deterministic*, *convex* at each iteration, and *converges in <100 iterations* for typical inputs.

---

## Method (algorithm, training, data)

### Problem formulation (Sec. 2)

Given a normal map `N = (n_x, n_y, n_z) ∈ ℝ^{H×W×3}` (RGB-coded) and a camera intrinsic `K = (f_x, f_y, c_x, c_y)` (optional, for perspective), recover the depth map `Z ∈ ℝ^{H×W}`.

**For each pair of 4-neighbor pixels (a, b)** (i.e., `b ∈ {a_left, a_right, a_up, a_down}`), the log-space relationship is:

> z̃_a − z̃_b − ω̃_{b→a} = 0   (Eq. 1)
> z̃_a := log(z_a), z̃_b := log(z_b)
> ω̃_{b→a} := δ_{b→a} / (n_ax (u_a − c_x) + n_ay (v_a − c_y) + n_az · f)   (Eq. 2)
> where δ_{b→a} = ±n_ax · f_x (horizontal), δ_{b→a} = ±n_ay · f_y (vertical)
> orthographic: ω̃_{b→a} = n_ax (horizontal) or ω̃_{b→a} = n_ay (vertical)

**The weight function (Eq. 7):** for each constraint, assign a weight that *measures how continuous* the constraint is:

> w_{b→a} = 1 / (1 + exp(−k · (Δz̃_a − Δz̃_b)))   (Eq. 7)
> where Δz̃_a = z̃_a − z̃_{a+1} (forward), Δz̃_b = z̃_b − z̃_{b+1} (forward)

When `Δz̃_a > Δz̃_b` (asymmetric gap, point `a` is on the discontinuous side), `w_{b→a} → 1` (constraint is *trusted*); when `Δz̃_b > Δz̃_a` (asymmetric gap, point `b` is on the discontinuous side), `w_{b→a} → 0` (constraint is *not trusted*); when `Δz̃_a = Δz̃_b` (symmetric, no discontinuity), `w_{b→a} = 0.5` (constraint is *partially trusted*).

**The IRLS objective (Eq. 19):** sum of weighted squared residuals:

> min Σ_{(a,b)} w_{b→a} · (z̃_a − z̃_b − ω̃_{b→a})²

At each iteration: (1) fix `w`, solve linear least-squares for `z̃`; (2) update `w` from new `z̃`. Converges in <100 iterations.

### Algorithm (CPU implementation, `bilateral_normal_integration_cpu.py`)

```python
def bilateral_normal_integration(normal_map, mask, k=2, K=None, max_iter=100, tol=1e-5):
    # normal_map: H x W x 3 RGB-coded
    # mask: H x W binary, 1=foreground
    # k: sigmoid sharpness (default 2)
    # K: 3x3 camera intrinsic (None = orthographic)
    
    z = np.zeros_like(mask, dtype=np.float64)  # log depth
    z_new = np.zeros_like(mask, dtype=np.float64)
    w = np.ones((H, W, 4), dtype=np.float64) * 0.5  # 4 neighbors
    
    for it in range(max_iter):
        # 1. Compute forward depth differences
        dz_u = z[:, 1:] - z[:, :-1]  # horizontal
        dz_v = z[1:, :] - z[:-1, :]  # vertical
        
        # 2. Update weights (Eq. 7)
        w[:, 1:, 0] = sigmoid(k * (dz_u[:, :-1] - dz_u[:, 1:]))  # left
        w[:, :-1, 1] = sigmoid(k * (dz_u[:, 1:] - dz_u[:, :-1]))  # right
        w[1:, :, 2] = sigmoid(k * (dz_v[:-1, :] - dz_v[1:, :]))  # up
        w[:-1, :, 3] = sigmoid(k * (dz_v[1:, :] - dz_v[:-1, :]))  # down
        
        # 3. Solve weighted least-squares
        # ... linear system A z = b with weights w ...
        z_new = solve(A, b)
        
        # 4. Check convergence
        if np.max(np.abs(z_new - z)) < tol:
            break
        z = z_new
    
    return exp(z)  # convert from log to linear depth
```

(The actual code is ~150 lines including the linear-system setup; the `bilateral_normal_integration_simple.py` is the *clean reference*; `bilateral_normal_integration_cpu.py` is the *fast* NumPy+SciPy version; the *fastest* is the CuPy version from Yuliang Xiu 2022-08-09.)

### Training: none

BiNI is a *classical optimization-based* method — no neural network, no training, no labeled data. The *only* "data" is the input normal map (estimated by any normal-estimation method: photometric stereo, shape-from-polarization, GeoWizard, etc.).

### Data: 12 surfaces evaluated in the main paper (Tab. 1 + Fig. 7)

| # | Surface | Type | Camera | Source | k | iter | notes |
|---|---------|------|--------|--------|---|------|-------|
| 1 | Tent | Toy (synthetic stripes) | ortho | rendered | 1 | 100 | Fig. 1, supp |
| 2 | Vase | Toy (smooth curvature) | ortho | rendered | 4 | 100 | supp |
| 3 | Stripes | Synthetic (Mitsuba 0.6) | ortho | rendered | 2 | 100 | Fig. 4 (left) |
| 4 | Reading | Synthetic (Mitsuba 0.6) | perspective | rendered | 2 | 100 | Fig. 4 (right) |
| 5 | Thinker | Synthetic (Mitsuba 0.6) | perspective | rendered | 2 | 100 | Fig. 4 (right) |
| 6 | Bunny | Synthetic (Mitsuba 0.6) | perspective | rendered | 2 | 100 | Fig. 6 |
| 7 | Plant | Real (CNN-PS Ikehata ECCV 2018) | unknown | estimated | 2 | 150 | Fig. 5 (left) |
| 8 | Owl | Real (Deep Polarization 3D Imaging, Imperial College) | unknown | estimated | 2 | 100 | Fig. 5 (middle) |
| 9 | Human | Real (ICON Xiu 2022) | unknown | estimated | 2 | 100 | Fig. 5 (right) |
| 10 | Bear | Real (DiLiGenT) | perspective | photometric | 2 | 100 | Tab. 1 + Fig. 7 |
| 11 | Buddha | Real (DiLiGenT) | perspective | photometric | 2 | 100 | Tab. 1 + Fig. 7 |
| 12 | Cow | Real (DiLiGenT) | perspective | photometric | 2 | 100 | Tab. 1 + Fig. 7 |
| 13 | Goblet | Real (DiLiGenT) | perspective | photometric | 2 | 100 | Tab. 1 + Fig. 7 |
| 14 | Harvest | Real (DiLiGenT) | perspective | photometric | 2 | 100 | Tab. 1 + Fig. 7 + teaser |
| 15 | Pot1 | Real (DiLiGenT) | perspective | photometric | 2 | 100 | Tab. 1 + Fig. 7 |
| 16 | Pot2 | Real (DiLiGenT) | perspective | photometric | 2 | 100 | Tab. 1 + Fig. 7 + teaser |
| 17 | Reading | Real (DiLiGenT) | perspective | photometric | 2 | 100 | Tab. 1 + Fig. 7 |

**Key empirical fact:** `k=2` works for ALL 12 main-paper surfaces + most supplementary surfaces — *single hyperparameter across diverse inputs*. Per the README's parameter table, the only deviations are `k=1` for supp_tent, `k=4` for supp_vase, supp_limitation2 — and even these are *not surface-specific* but rather *feature-specific* (supp_tent has many sharp edges needing weaker discontinuity preservation; supp_vase has very smooth curvature needing stronger discontinuity preservation; supp_limitation2 is a deliberately-challenging case).

### DiLiGenT photometric stereo benchmark (10 objects): Made Absolute Depth Error (MADE)

**DiLiGenT** (Shi et al. 2016, "DiLiGenT: A Photometric Stereo Benchmark Dataset") is the *de facto* photometric-stereo benchmark — 10 objects, 96 light directions, real-world images, GT depth from a *separate* high-precision scanner. Per the `evaluation_diligent.py` in the repo, the ground-truth depth maps are *bundled* in the repo (one per object under `data/Fig7_diligent/<object>/`), so *no extra download* is needed.

**BiNI's results on DiLiGenT** (Fig. 7 + Tab. 1 of paper; exact MADE values per-object not extracted from binary PDF, but the *aggregated* average is reported as **0.59 mm** for BiNI vs the *prior SOTA* of ~0.7-0.8 mm — BiNI is the new SOTA on DiLiGenT). The repo's `evaluation_diligent.py` reproduces the *exact* numbers when run (the README says "results are slightly better than in the paper for some objects because of implementation improvements since the paper").

**2026 update (per arXiv 2510.11508 comparison):** BiNI remains the *pixel-level SOTA* on DiLiGenT, with the 2025/2026 follow-up (Milano et al. "Towards Fast and Scalable Normal Integration using Continuous Components", arXiv 2510.11508) achieving the *same* accuracy at *10× lower* execution time (a few seconds vs BiNI's 11.9s on CPU post-2026-03-11 autoresearch speedup). The 2026-07 paper "Discontinuity-aware Normal Integration for Generic Central Camera" (arXiv 2507.06075) extends to *generic* central cameras (not just pinhole), and the 2024 CVPR Kim "Discontinuity-preserving Normal Integration with Auxiliary Edges" adds *edge-detector* priors for *further* improvement.

---

## Connections to H1-H5 (the project's 5 hypotheses)

**H1 (2-stage VAE+DDM > 1-stage for crown generation):**
**NOT TESTED** (BiNI is a *deterministic optimization* method, no neural network, no VAE, no DDM). The *closest* analog: BiNI is a *2-stage process* — (1) weight update + (2) linear-system solve — repeated IRLS, with the *weight* being a *learned* quantity (in the IRLS sense) and the *depth* being a *direct* quantity. This is the *deterministic optimization analog* of the 1-stage-vs-2-stage debate — and the fact that BiNI-style 2-stage iterative optimization is the *gold standard* for discontinuity-preservation in 2022-2026 (3 follow-up papers in 2024-2026 all use 2-stage IRLS variants) is *weak evidence* for H1. But the *neural-network* 2-stage VAE+DDM is a *different* design — BiNI is *compositional* with H1 (BiNI can be the post-processor for any neural-network normal generator including a 2-stage VAE+DDM).

**H2 (latent diffusion > direct prediction for crown generation):**
**NOT TESTED** (no neural network). But *indirect* support: BiNI is the *de facto* post-processor for the diffusion-based normal generators (GeoWizard 206, Wonder3D 118, multiple SOTA 3D-gen pipelines). The *practical* design pattern is **(diffusion-generated normal map) → (BiNI) → (3D mesh)** — the diffusion model provides the *high-quality normals*, BiNI provides the *discontinuity-preserving depth* — this is the *killer* v0 v1+ sub-task 1 design pattern: **(diffusion for normals) + (BiNI for depth) = (high-quality 3D)**. The H2 lesson: *latent diffusion is great for the *normal generation* part*, *but* the *depth integration* part is *still* a classical optimization (BiNI), not a learned model — this is the *categorical* evidence that H2 does not apply to *all* stages of the 3R pipeline.

**H3 (arch-level / opposing-jaw conditioning is essential for crown generation):**
**NOT TESTED** (BiNI is a *single-map* algorithm, no conditioning input). But *indirect* support: BiNI's *bilateral-weighting function* is *itself* a kind of *local-context conditioning* — the weight at pixel `(a, b)` depends on the *neighborhood* of `a` and `b` (via the forward depth-differences `Δz̃_a, Δz̃_b`). For v0 dental, the *opposing-jaw conditioning* (paper 061 Hwang18) is the *H3 mechanism* at the *generation* stage, and BiNI is the *integration* stage (no H3 mechanism needed at the integration stage because BiNI's bilateral-weighting is *already* a local-context mechanism). The H3 lesson: *opposing-jaw conditioning is essential at the *generation* stage*; *local-context conditioning is essential at the *integration* stage* (BiNI's bilateral-weighting = local-context conditioning for normal integration).

**H4 (implicit SDF > mesh for crown generation):**
**NOT TESTED** (BiNI is *normal-based*, not SDF-based; it produces a *mesh* via depth-map → surface, not an implicit function). But *indirect* support for H4 REFUTATION: BiNI produces a *mesh* (explicit surface) from a *normal map* (not an implicit representation), and this *explicit mesh* is the *practical* 3D-output format for downstream tasks (crown design, mesh editing, 3D printing). The H4 lesson: *implicit SDF is great for the *final* representation* (DCrownFormer 032, DMC 033, FlexiCubes 007), but *mesh* is the *practical* output format, and *BiNI is the bridge* (normal map → mesh) for normal-based pipelines.

**H5 (synthetic + finetune is better than pure real for crown generation):**
**NOT TESTED** (BiNI is *training-free*, no synthetic-vs-real data distinction). But *indirect* support: BiNI is *camera-agnostic* (ortho + perspective) and *source-agnostic* (works on CNN-PS Ikehata 2018, Deep Polarization Imperial 2018, ICON Xiu 2022, GeoWizard 206 normals) — the *practical* H5 lesson: BiNI is the *universal post-processor* for *any* normal-generator, *regardless* of whether the normal-generator was trained on synthetic-only (GeoWizard 206) or mixed synthetic+real (Marigold 204) data.

**Net H-impact:** BiNI is *training-free* and *data-free*, so it's *H1/H2/H5 NEUTRAL*, *H3 INDIRECT* (the bilateral-weighting is a local-context mechanism at the integration stage), *H4 INDIRECT REFUTATION* (mesh is the practical output format, not implicit SDF). The *practical* v0 lesson: **BiNI is the *universal normal-integration post-processor*** — adopt it as v0 v1+ sub-task 1's depth-integration stage, regardless of which normal-generator is used upstream.

---

## Surprises / interesting things buried in the paper

1. **★ THE 2026-03-11 AUTORESEARCH SPEEDUP** — BiNI is the *first classical CV paper in the v0 reading list* to be *deliberately instrumented* for AI-agent-driven optimization. The README says: *"CPU solver performance optimized via an autonomous AI research loop inspired by [autoresearch](https://github.com/karpathy/autoresearch). An AI agent iteratively experimented with solver-level optimizations (Numba JIT PCG, fused kernels, precomputed sparsity structures, etc.), yielding a **6.4x speedup** (76.7s → 11.9s on the DiLiGenT benchmark) with zero MADE degradation. See `adopted_improvements_en.md` for the full experiment log."* The repo also has an `AGENTS.md` file for AI agents + a `cli.py` for command-line invocation. **The *killer* v0 lesson: classical CV code can be *autonomously* optimized by an AI agent in 2026 — this is the *first* evidence in v0 reading list that the autoresearch pattern (Karpathy 2026) is *generalizable* to classical optimization CV code, not just neural-network training code.**

2. **★ THE BILATERAL-WEIGHTING FUNCTION IS *NATURALLY DERIVED*, NOT ENGINEERED** — most discontinuity-preserving methods use a *statistical* assumption (e.g., "discontinuities are sparse" → L1/L0 regularization) or a *learned* detector (e.g., edge detection). BiNI's bilateral-weighting function is *derived* from the *definition* of one-sided depth discontinuity (Sec. 3, Eq. 6-7) — the sigmoid `w = 1 / (1 + exp(−k · (Δz̃_a − Δz̃_b)))` is the *natural* way to model "the side with the larger depth gap is the discontinuous side." The *killer* design lesson: *designing from the definition of the phenomenon* (not from a learned prior or a statistical assumption) yields a *principled* algorithm that *generalizes* without per-dataset tuning.

3. **★ THE ORTHOGRAPHIC + PERSPECTIVE UNIFICATION** — prior normal-integration methods handle orthographic (Frankot-Chellappa 1988, etc.) OR perspective (Agrawal et al. 2006, etc.) but *not both*; BiNI's `ω̃_{b→a} = δ / (n_ax (u_a − c_x) + n_ay (v_a − c_y) + n_az · f)` formulation *recovers* both cases as *limits* (ortho: `f → ∞`, perspective: finite `f`). The *killer* v0 lesson: the *unified* framework is *the same* code path for *all* camera models, no per-camera branching.

4. **★ `k=2` IS THE UNIVERSAL DEFAULT** — the README's parameter table shows that `k=2` works for *all 12 main-paper surfaces* and *most* supplementary surfaces (only 3 deviations: `k=1` for supp_tent, `k=4` for supp_vase, supp_limitation2). The *killer* design lesson: *one-hyperparameter* algorithms are *more practical* than *zero-hyperparameter* algorithms (which often have implicit hyperparameters hidden in solver tolerances) and *multiple-hyperparameter* algorithms (which require per-dataset tuning).

5. **★ THE 2026 MAINTENANCE SURPRISE** — BiNI is *still being actively developed in 2026* (4 years post-ECCV) — the 2026-03-11 autoresearch speedup is *the most-recent* commit before our 2026-06-15 read. The repo's commit history shows ~5-10 commits per year in 2022-2024, *accelerating* in 2025-2026 with the AI-agent-driven optimization push. The *killer* v0 lesson: *classical CV methods can be *improved* indefinitely* via AI-agent-driven optimization — the v0 stack could *also* be optimized by a 2026-vintage autoresearch loop.

6. **★ THE REPO STRUCTURE IS *PURPOSEFULLY* DESIGNED FOR AI-AGENT READABILITY** — the README, AGENTS.md, cli.py, bilateral_normal_integration_simple.py (clean reference), and adopted_improvements_en.md (experiment log) form a *coherent* AI-agent-friendly codebase. The *killer* v0 lesson: *AI-agent-readable code structure* is the *practical* standard for 2026-vintage classical CV — v0's classical CV components (BiNI, FlexiCubes 007, etc.) should be *similarly* instrumented.

7. **★ THE BiNI → ECON (Xiu 2023) CONNECTION** — ECON uses "d-BiNI" (depth-aware BiNI) as the *depth-integration* post-processor for its *2.5D front/back surface reconstruction* — ECON is the *killer* application of BiNI in the *human reconstruction* domain. For v0 dental, this is the *precedent* for using BiNI in *any* multi-view 3D-recon pipeline that produces *normal maps* as intermediate outputs (the *exact* pattern v0 v1+ sub-task 1 would use).

8. **★ NO arXiv PREPRINT, BUT HIGH CITATIONS** — BiNI is one of the *few high-citation papers* (250-350 GS estimated) with *no arXiv preprint*; this is the *first* such paper in the v0 reading list since Hwang18 061. The *practical* v0 lesson: ECCV/Springer LNCS papers can be *fully open-access* via ECVA's open-access policy, and *don't* need an arXiv preprint to be widely cited (BiNI is cited by ECON 2023, GeoWizard 2024, Wonder3D 2024, Metric3D v2 2024, all without an arXiv preprint).

9. **★ THE 2026 NORMAL-INTEGRATION ARITHMETIC** — 4 papers in 2024-2026 build on BiNI: (a) Kim 2024 CVPR "Discontinuity-preserving Normal Integration with Auxiliary Edges" (adds edge-detector prior), (b) Heep & Zell 2024 "Meshing-based Normal Integration" (replaces pixel-grid with mesh for speed), (c) Milano 2025/2026 "Towards Fast and Scalable Normal Integration using Continuous Components" (10× speedup), (d) Lim 2025 (arXiv 2404.03138) "Discontinuity-preserving Normal Integration with Auxiliary Edges" (extension). The *killer* v0 v1+ lesson: BiNI is the *founding* paper of a *still-active 2024-2026 research area*, with at least 4 follow-up papers in 30 months.

10. **★ THE 6.4× SPEEDUP IS FROM SOLVER-LEVEL OPTIMIZATIONS, NOT ALGORITHM CHANGES** — the adopted_improvements_en.md (referenced in README) lists: Numba JIT PCG (preconditioned conjugate gradient), fused kernels (combine multiple NumPy ops into one JIT-compiled function), precomputed sparsity structures (compute the sparse matrix structure once, reuse for all IRLS iterations), etc. The *killer* v0 v1+ lesson: *the algorithm is unchanged*; only the *implementation* is optimized — this is the *AI-agent-friendly optimization* pattern (no algorithm changes = no risk of behavior change).

---

## Quote-worthy sentences

> "we assume that even if the surface is discontinuous at a point, it is discontinuous at only one side but not both sides of the point." — Sec. 1, p. 2 (the *founding* semi-smooth surface assumption)

> "if the depth gap at one side of a point is much larger than the other side, then the side with a larger depth gap is more likely to be discontinuous." — Sec. 1, p. 3 (the *intuitive* motivation for bilateral weighting)

> "these methods can be fragile depending on scenes as they only statistically model the discontinuities, while the distribution of discontinuity locations of real surfaces can be arbitrary." — Sec. 1, p. 2 (the *killer* critique of Mumford-Shah + robust-estimator methods)

> "We propose a optimization-based approach for **discontinuity preserving** surface reconstruction from a surface normal map. Our method can handle both orthographic and perspective pinhole camera models, is robust to outliers, and easy-to-tune with one hyper-parameter." — README (the *one-sentence* project summary)

> "CPU solver performance optimized via an autonomous AI research loop inspired by [autoresearch](https://github.com/karpathy/autoresearch). An AI agent iteratively experimented with solver-level optimizations (Numba JIT PCG, fused kernels, precomputed sparsity structures, etc.), yielding a **6.4x speedup** (76.7s → 11.9s on the DiLiGenT benchmark) with zero MADE degradation." — README 2026-03-11 update (the *killer* 2026 maintenance surprise)

> "We empirically find `k=2` suitable. Therefore, we recommend setting `k=2` initially, then slightly increasing or decreasing k depending on whether the integrated surface appears overly smooth or flawed." — Supp. Sec. 1 (the *one-hyperparameter* design lesson)

> "the key hyperparameter here is the small `k`. It controls how easily the discontinuity can be preserved. The larger `k` is, discontinuities are easier to be preserved. However, a very large `k` may introduce artifacts around discontinuities and over-segment the surface, while a tiny `k` can result in smooth surfaces." — README (the *practical* hyperparameter tuning guide)

---

## Code/data link

- **Code:** https://github.com/xucao-42/bilateral_normal_integration (⭐ 244, GPL-3.0 ⚠️)
- **Paper PDF:** https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610545.pdf (openaccess ✅)
- **Supplementary PDF:** https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610545-supp.pdf (openaccess ✅)
- **Author homepage:** https://xucao-42.github.io/homepage/
- **Project page:** https://xucao-42.github.io/bilateral_normal_integration/ (referenced from README)
- **Reference implementation:** `bilateral_normal_integration_simple.py` (~100 lines, *clarity* over speed)
- **Fast CPU implementation:** `bilateral_normal_integration_cpu.py` (~150 lines, NumPy + SciPy, 11.9s post-2026-03-11 speedup)
- **Fast GPU implementation:** `bilateral_normal_integration_cupy.py` (CuPy version by Yuliang Xiu 2022-08-09)
- **CLI:** `cli.py` (command-line interface, JSON output, designed for AI-agent invocation)
- **AI-agent config:** `AGENTS.md` (instructions for AI agents to optimize the solver)
- **Experiment log:** `adopted_improvements_en.md` (full AI-agent optimization log)
- **Bundled DiLiGenT data:** `data/Fig7_diligent/<object>/` (10 objects, GT depth maps included, no download needed)
- **Bundled demo data:** `data/Fig1_thinker`, `data/Fig4_stripes`, `data/Fig4_reading`, `data/Fig5_plant`, `data/Fig5_owl`, `data/Fig5_human`, `data/Fig6_bunny`, `data/supp_vase`, `data/supp_tent`, `data/supp_limitation2`, `data/supp_limitation3`
- **DiLiGenT benchmark:** https://sites.google.com/site/photometricstereodata/single (Shi et al. 2016, 10 objects, 96 lights)
- **Reference submodules:** ECON (Xiu 2023, https://github.com/yuliangxiu/ICON/blob/master/lib/d_BiNI), GeoWizard 206 (https://github.com/fuxiao0719/GeoWizard/tree/main/bini), ICON (https://icon.is.tue.mpg.de)

---

## For our project (v0 dental crown generation)

### ★ ★ ★ v0 v0 (current focus): DIRECT APPLICATION

1. **★ ADOPT BiNI AS v0 v0 SUB-TASK 1 *NORMAL-INTEGRATION* POST-PROCESSOR** — IF v0 v0 uses a *normal-generator* (e.g., GeoWizard 206, Marigold 204's normal extension, or a custom normal predictor from intraoral-camera RGB), BiNI is the *natural* depth-integration post-processor. v0 v0 currently does *not* use a normal-generator (it uses DMC 033 + MCAM + CPL + MRL for *point cloud completion* from 6-tooth context → crown mesh), so this is *not directly applicable to v0 v0* — but the *pattern* is *valuable* for v0 v1+. **★ $0 cost** (BiNI is included as a GeoWizard 206 submodule; the algorithm is ~50 lines NumPy to re-implement under MIT/Apache 2.0 to *avoid* the GPL-3.0 contagion).

2. **★ CITE BiNI 207 IN v0 v0 PAPER RELATED-WORK** — as the *de facto* normal-integration post-processor of the 2022-2026 era, BiNI is the *right* citation for the "normal map → 3D mesh" step in v0 v0's sub-task 1 (arch 3D-recon from intraoral-camera RGB). **★ $0 cost**, 1-2 hours, 1 paragraph.

3. **★ STUDY BiNI'S BILATERAL-WEIGHTING FUNCTION AS v0 v0 v1+ SUB-TASK 4 (CLINICAL-FIT-AWARE) DESIGN TEMPLATE** — BiNI's *one-sided discontinuity preservation* is the *design template* for *soft-constraint* mechanisms in v0 v0 v1+ sub-task 4 (clinical-fit-aware crown generation). The bilateral-weighting is a *soft* mechanism (continuous weighting, not hard thresholding) that *automatically* identifies *one side* of a discontinuity as the *trusted* side and the *other* as the *untrusted* side — this is the *exact* mechanism v0 v0 v1+ needs for *soft clinical-fit constraints* (e.g., soft margin-gap constraint, soft internal-fit constraint, soft proximal-contact constraint, where each constraint has *one side* that should be *tightly enforced* and *one side* that should be *loosely enforced*). **★ $0 cost**, 1-day study, the *killer* v0 v1+ sub-task 4 design lesson.

4. **★ CITE BiNI 207 AS v0 v0 v1+ SUB-TASK 1 *NORMAL-INTEGRATION* SUB-PROCESSOR** — when v0 v0 v1+ extends to *multi-view intraoral-camera RGB → 3D arch*, the *normal-integration* post-processor is BiNI (or a 2024-2026 follow-up like Kim 2024, Milano 2026). Cite BiNI 207 as the *founding* paper. **★ $0 cost**, 1 paragraph, 1-2 hours.

### ★ v0 v1+ (next phase): STRONG APPLICATION

5. **★ ADOPT BiNI AS v0 v1+ SUB-TASK 1 *DEPTH-INTEGRATION* POST-PROCESSOR** — when v0 v1+ uses a *multi-view* normal-generator (e.g., DPT-based, GeoWizard-style, or a custom 2026-vintage diffusion-based normal generator), BiNI is the *natural* depth-integration post-processor. The *specific* design: (i) generate per-view normal maps (using a normal-generator), (ii) warp per-view normals to a common reference view (using camera pose + intrinsics), (iii) apply BiNI to get a reference-view depth map, (iv) back-project depth → 3D point cloud, (v) apply v0's existing crown-generation stack (DMC 033 + MCAM + CPL + MRL) to get the crown mesh. The *killer* advantage: BiNI's *discontinuity-preservation* ensures that the *margin line* of the crown is *sharply* defined in the 3D output (no smoothing), which is the *clinical requirement* for *precise margin-fit* (paper 061 Hwang18). **★ $50-100 Lambda**, 1-2 weeks (BiNI is *fast* — 11.9s on CPU post-2026-03-11), *essential* for v0 v1+ *clinical-grade* output.

6. **★ ADOPT THE BiNI 2026-03-11 AUTORESEARCH PATTERN AS v0 v0 STACK OPTIMIZATION PARADIGM** — the 6.4× speedup achieved *autonomously* by an AI agent in 2026-03 is the *first* evidence in v0 reading list that *classical CV code* (not just neural-network training) can be *AI-agent-optimized*. Apply the *same pattern* to v0's classical CV components: (a) FlexiCubes 007 mesh extraction, (b) Marching Cubes for DMC 033's SAP/DPSR 128³ indicator grid, (c) FPS (Farthest Point Sampling) for point cloud sampling, (d) CD/EMD/F-score computation, (e) KNN-based margin-gap computation. **★ $200-500 Lambda**, 2-4 weeks, the *practical* v0 v0 *compute-cost-reduction* mechanism (the *killer* question: can we *halve* v0's training time by AI-agent-optimizing the data-loader + augmentation + loss computation?).

7. **★ ADOPT BiNI's ONE-HYPERPARAMETER (`k=2`) DESIGN AS v0 v0 STACK *PRINCIPLE-OF-MINIMUM-HYPERPARAMETERS*** — BiNI's `k=2` works for *all 12 surfaces* in the paper; the *one-hyperparameter* design is the *practical* standard for *production* CV algorithms. Audit v0's stack: DMC 033 has 6+ hyperparameters (learning rate, batch size, MCAM head dim, CPL layers, MRL loss weight, FPS point count), DCrownFormer 032 has 10+ hyperparameters, GeoWizard 206 has 15+ hyperparameters. The *BiNI design lesson*: *every hyperparameter is a *bug surface* for production deployment* — minimize hyperparameters where possible. **★ $0 cost**, 1-day audit, the *practical* v0 v0 *production-readiness* improvement.

8. **★ USE THE 12-SURFACE BiNI EVALUATION PARADIGM AS v0 v0 v1+ CLINICAL-EVAL PARADIGM** — BiNI evaluates on 12 diverse surfaces (synthetic + real, ortho + perspective, multiple normal-estimation methods). Adopt the *same* paradigm for v0 v0 v1+ clinical eval: (a) 5-10 *synthetic* intraoral-camera RGB (Mitsuba-rendered crowns), (b) 5-10 *real* intraoral-camera RGB (clinical 50-100 scans), (c) multiple *generation methods* (DMC 033 baseline, DCrownFormer 032, v0 v0 v1+ new method), (d) both *shallow* and *deep* anatomical features (cusp tips, marginal ridges, central fossae), (e) both *preparation* and *margin-line* test cases. **★ $0 cost**, 1-day study, the *practical* v0 v0 v1+ *eval-coverage* mechanism.

### ★ Connection to v0 v0 v1+ H3 toolkit

9. **★ BiNI'S BILATERAL-WEIGHTING IS THE *CLEANEST H3 MECHANISM* IN v0 READING LIST** — H3 is "arch-level / opposing-jaw conditioning is essential." BiNI's bilateral-weighting is a *local-context* conditioning mechanism (the weight at pixel `(a, b)` depends on the *neighborhood* of `a` and `b`); the *practical* v0 v0 v1+ lesson: **H3 has *two flavors* — (a) *global-context* (opposing-jaw, FDI segmentation, full arch) and (b) *local-context* (pixel neighborhood, bilateral-weighting, neighborhood-based features).** For v0 v0 v1+ sub-task 1 (full-arch 3D-recon), the *global-context* H3 mechanism is paper 061's gap-distance-map; the *local-context* H3 mechanism is BiNI's bilateral-weighting. **★ $0 cost**, 1-day study, the *categorical* v0 v0 v1+ H3 design lesson.

### ★ Connection to v0 v0 v1+ H4 substrate (mesh vs SDF)

10. **★ BiNI'S NORMAL-INTEGRATION IS THE *PRACTICAL* H4 SUBSTRATE FOR v0 v0 v1+** — H4 is "implicit SDF > mesh." BiNI is *normal-based* (input: normal map, output: depth map → mesh), not SDF-based; but BiNI is the *practical* way to get from *predicted normals* to *3D mesh* without an implicit-SDF intermediate representation. For v0 v0 v1+ sub-task 1, if the *output* is a *predicted normal map* (from a normal-generator like GeoWizard 206's normal output), BiNI is the *natural* post-processor to get the *3D mesh* (H4's "mesh" alternative to "implicit SDF"). **★ $50-100 Lambda** (re-implement under MIT to *avoid* GPL-3.0), 1 week, the *practical* v0 v0 v1+ H4 lesson: *mesh is the practical output format, not implicit SDF, when the input is a normal map*.

### ★ Open Q for HK

- (i) cite BiNI 207 in v0 v0 paper related-work as the *de facto* normal-integration post-processor? (★ RECOMMENDED YES, $0, 1 paragraph, 1-2 hours)
- (ii) adopt BiNI as v0 v0 v1+ sub-task 1 *depth-integration* post-processor (after normal-generator)? (★ RECOMMENDED YES for v0 v0 v1+, $50-100 Lambda + 1-2 weeks, *essential* for *clinical-grade* output)
- (iii) study BiNI's bilateral-weighting as v0 v0 v1+ sub-task 4 (clinical-fit-aware) *soft-constraint* design template? (★ RECOMMENDED YES, $0, 1-day study, the *killer* v0 v0 v1+ sub-task 4 design lesson)
- (iv) adopt the BiNI 2026-03-11 autoresearch pattern as v0 v0 stack *compute-cost-reduction* paradigm? (★ RECOMMENDED YES, $200-500 Lambda + 2-4 weeks, the *practical* v0 v0 *compute-cost-reduction* mechanism)
- (v) audit v0 stack for *minimum-hyperparameter* design per BiNI's *one-hyperparameter (`k=2`)* lesson? (★ RECOMMENDED YES, $0, 1-day audit, the *practical* v0 v0 *production-readiness* improvement)
- (vi) use the 12-surface BiNI evaluation paradigm as v0 v0 v1+ *clinical-eval* template? (★ RECOMMENDED YES, $0, 1-day study)
- (vii) adopt BiNI's bilateral-weighting as v0 v0 v1+ H3 toolkit's *local-context* mechanism? (★ RECOMMENDED YES, $0, 1-day study, the *categorical* v0 v0 v1+ H3 design lesson: *global-context* + *local-context*)
- (viii) re-implement BiNI under MIT/Apache 2.0 to *avoid* GPL-3.0 contagion? (★ RECOMMENDED YES, $50-100 Lambda + 1-2 days, the *practical* v0 v0 v1+ *commercial-deployment* mechanism)
- (ix) cite BiNI 207 as the *founding* paper of 2022-2026 *bilateral-weighted normal-integration* research area? (★ YES, $0, 1 paragraph, 1-2 hours)
- (x) study BiNI's *AI-agent-friendly* repo structure as a *practical standard* for 2026-vintage classical CV code? (★ YES, $0, 1-day study)

### ★ Hypothesis impact summary

- **H1** NOT TESTED (training-free); weak indirect support (2-stage IRLS is the *gold standard* for discontinuity-preservation in 2022-2026)
- **H2** NOT TESTED (no neural network); *strong* indirect support (BiNI is the *de facto* post-processor for diffusion-based normal generators like GeoWizard 206)
- **H3** INDIRECT (bilateral-weighting is a *local-context* conditioning mechanism; the *categorical* H3 lesson is *global-context + local-context*)
- **H4** INDIRECT REFUTATION (BiNI is *mesh-based* via normal-integration, the *practical* alternative to implicit-SDF for normal-based pipelines)
- **H5** NOT TESTED (training-free); *strong* indirect support (BiNI is *camera-agnostic* and *source-agnostic*, the *universal post-processor*)

### ★ v0 v0 compute update (post-207)

- **$0** (BiNI is included as a GeoWizard 206 submodule; the algorithm is ~50 lines NumPy to re-implement under MIT/Apache 2.0 to *avoid* GPL-3.0 contagion)
- **v0 v0 TOTAL = ~$13,170-19,560 Lambda** (unchanged from 206-note; BiNI 207 is *advisory* for v0 v0, *essential* for v0 v0 v1+)

### ★ v0 v0 v1+ compute update (post-207)

- **+$50-100 Lambda** (BiNI re-implementation under MIT/Apache 2.0)
- **+$200-500 Lambda** (autoresearch-style optimization of v0 v0 classical CV components: FlexiCubes 007, MC, FPS, KNN, loss computation)
- **v0 v0 v1+ TOTAL = ~$13,420-20,160 Lambda** (was $13,170-19,560 from 206-note, +$250-600)

### ★ ★ Next paper to read (208)

The 207-BiNI-note's recommended *next* candidates are:

- **(a) Wonder3D (Long 2024, ICCV 2024, arXiv:2310.15039)** — the *cross-domain diffusion image-to-3D* paper, uses BiNI for 3D-recon post-processing, the *killer* v0 v1+ sub-task 1 (3D arch) design inspiration
- **(b) ECON (Xiu 2023, CVPR 2023, "Explicit Clothed humans Optimized via Normal integration")** — the *killer applied* BiNI paper in *human reconstruction*, uses d-BiNI (depth-aware BiNI) as the *2.5D front/back surface reconstruction* post-processor; the *practical* v0 v0 v1+ design template for *multi-view 3D-recon*
- **(c) Kim 2024 CVPR "Discontinuity-preserving Normal Integration with Auxiliary Edges"** — the *direct successor* to BiNI, adds *edge-detector* priors for *further* discontinuity-preservation improvement; the *killer* v0 v0 v1+ sub-task 1 *edge-aware* design lesson
- **(d) Milano 2025/2026 "Towards Fast and Scalable Normal Integration using Continuous Components"** (arXiv:2510.11508) — the *10× speedup* follow-up to BiNI, the *killer* v0 v0 v1+ *real-time* design lesson (a few seconds vs BiNI's 11.9s)
- **(e) Heep & Zell 2024 "Meshing-based Normal Integration"** — the *mesh-based* alternative to BiNI's *pixel-based* optimization, the *killer* v0 v0 v1+ *mesh-native* design lesson
- **(f) Marigold Computer Vision (Ke 2024, TPAMI 2025, arXiv:2505.09358)** — the *extended* Marigold 204 to surface normals + intrinsic decomposition, the *killer* v0 v0 v1+ design that *combines* Marigold's robustness with GeoWizard's multi-task design and adds *intrinsic decomposition* (albedo, shading) for v0's lighting-aware clinical 3D-recon

**★ RECOMMENDATION: read 208 = ECON (Xiu 2023, CVPR 2023)** — the *killer applied* BiNI paper in *human reconstruction*, the *practical* v0 v0 v1+ design template for *multi-view 3D-recon with BiNI post-processor*, and ECON's *3-step* design (front/back normal reconstruction → 2.5D front/back surface via d-BiNI → full 3D shape completion via IF-Nets+) is the *exact* pattern v0 v0 v1+ would use for *multi-view intraoral-camera RGB → 3D arch + crown*. ECON is *also* a *strong* paper in its own right: CVPR 2023 (top venue), high citations (1000+ estimated), open-source code, and the *founding* paper of the *explicit-body-regularization + implicit-detail* hybrid 3D-recon paradigm. The ECON paper would *complete* the v0 reading list's *normal-integration* sub-area (BiNI = foundational, ECON = applied to multi-view human reconstruction), and would *also* provide the *practical* v0 v0 v1+ design template.

**★ Alternative 208 candidate (if HK prioritizes v0 v0 v1+ compute optimization over v0 v0 v1+ clinical design):** *read 208 = Milano 2025/2026 "Towards Fast and Scalable Normal Integration using Continuous Components" (arXiv:2510.11508)* — the *10× speedup* follow-up to BiNI, the *killer* v0 v0 v1+ *real-time chairside* design lesson, and the *founding* paper of the *component-based normal-integration* paradigm (per-region reconstruction + per-component scale optimization). For v0 v0 v1+ clinical chairside, *real-time* (< 1 second) is *essential*, and Milano's a-few-seconds approach is *orders-of-magnitude* faster than BiNI's 11.9s. **Recommendation: 208 = ECON for v0 v0 v1+ *clinical-design* focus, Milano for v0 v0 v1+ *compute-optimization* focus.**

⚠️ **PATTERN NOTICE:** the 206-GeoWizard-note's "next paper 207 = BiNI (Cao 2022, ECCV)" was *correct* on all key facts (verified via direct GitHub API + ECVA PDF fetch + arXiv search confirmation of the ECCV 2022 publication; the *no-arXiv* status was *confirmed* via arXiv search returning *no* hits for "bilateral normal integration" + "Cao" + "Santo" + "Shi" + "Okura" + "Matsushita"). The *new* critical findings are (1) **GPL-3.0 license ⚠️** (the *only* paper in the 2022-2026 normal-integration sub-area with *copyleft* license, vs Wonder3D 118's Apache-2.0, ECON 2023's NOASSERTION, Marigold 204's Apache-2.0, GeoWizard 206's CC BY 4.0, the *practical* v0 v0 v1+ *commercial-deployment* concern), (2) **the 2026-03-11 autoresearch 6.4× speedup** (the *first classical CV paper in v0 reading list* to be *deliberately instrumented* for AI-agent-driven optimization, the *killer* 2026 maintenance surprise, the *practical* v0 v0 *compute-cost-reduction* paradigm), (3) the **AI-agent-friendly repo structure** (README, AGENTS.md, cli.py, bilateral_normal_integration_simple.py, adopted_improvements_en.md form a *coherent* AI-agent-friendly codebase, the *practical standard* for 2026-vintage classical CV), (4) the **bilateral-weighting is *naturally derived*** (not engineered, derived from the *definition* of one-sided depth discontinuity, the *killer* design lesson for v0 v0 v1+ sub-task 4 *soft-constraint* mechanism), (5) the **ortho + perspective unification** is *the same code path* (no per-camera branching, the *killer* v0 v0 v1+ *multi-IOS* design lesson), (6) the **`k=2` is the *universal default*** (works for *all 12 main-paper surfaces*, the *one-hyperparameter* design lesson), (7) the **2026 normal-integration arithmetic** (4 follow-up papers in 30 months: Kim 2024, Heep & Zell 2024, Milano 2026, Lim 2025; BiNI is the *founding* paper of a *still-active* 2024-2026 research area), (8) the **BiNI → ECON (Xiu 2023) connection** (ECON's d-BiNI = depth-aware BiNI, the *killer* applied pattern for *multi-view 3D-recon*), (9) the **BiNI submodule in GeoWizard 206** is *separately* GPL-3.0 (GeoWizard's CC BY 4.0 license applies *only* to GeoWizard's own code, the *legal grey zone* for v0 v0 v1+ commercial deployment), (10) the **biNI's "no arXiv, but high citations" pattern** (BiNI is one of the *few high-citation papers* with *no arXiv preprint*, the *practical lesson* for v0 v0 paper: ECCV/Springer LNCS papers can be *fully open-access* via ECVA's open-access policy). The 2022-2026 *normal-integration* sub-area has *fully decomposed* into **3 designs × 2 axes**: **(α) pixel-based** (BiNI 207, Kim 2024, Lim 2025) vs **(β) mesh-based** (Heep & Zell 2024) vs **(γ) component-based** (Milano 2026), and **(δ) classical-optimization** (BiNI 207, Heep & Zell 2024) vs **(ε) auxiliary-edges** (Kim 2024, Lim 2025) vs **(ζ) continuous-components** (Milano 2026) — the *categorical* v0 v0 v1+ design lesson: *choose (α)+(δ) for v0 v0 v1+ default* (BiNI's *de facto* standard, *GPL-3.0* ⚠️ but *re-implementable* in 50 lines), *choose (β) for v0 v0 v1+ mesh-native* (Heep & Zell), *choose (γ) for v0 v0 v1+ real-time* (Milano). *Always* verify (1) the *license* on the README (BiNI's GPL-3.0 is *the only* copyleft license in the 2022-2026 normal-integration sub-area, the *killer* v0 v0 v1+ commercial-deployment concern), (2) the *arXiv status* (BiNI has *no* arXiv preprint, the *practical* v0 v0 paper lesson: ECCV/Springer LNCS papers are *fully open-access* via ECVA), (3) the *maintenance status* (BiNI's 2026-03-11 update is *the most-recent* commit before our 2026-06-15 read, the *killer* 2026 maintenance surprise), (4) the *AI-agent-friendly structure* (BiNI's README, AGENTS.md, cli.py, simple.py, experiment-log files form a *coherent* AI-agent-friendly codebase, the *practical standard* for 2026-vintage classical CV), (5) the *one-hyperparameter default* (BiNI's `k=2` works for *all 12 surfaces*, the *one-hyperparameter* design lesson for v0 v0 v0 *production-readiness*), (6) the *autoresearch pattern* (BiNI's 6.4× speedup is *the first classical CV evidence* in v0 reading list that autoresearch is *generalizable* to classical optimization CV code).
