# Paper 014 — MeshDiffusion: Score-based Generative 3D Mesh Modeling

- **Title:** MeshDiffusion: Score-based Generative 3D Mesh Modeling
- **Authors:** Zhen Liu¹²*, Yao Feng²³, Michael J. Black², Derek Nowrouzezahrai⁴, Liam Paull¹, Weiyang Liu²⁵
- **Affiliations:** ¹Mila, Université de Montréal · ²Max Planck Institute for Intelligent Systems — Tübingen · ³ETH Zürich · ⁴McGill University · ⁵University of Cambridge
- **Year:** 2023 (arXiv v1: 14 Mar 2023; v2: 15 Apr 2023)
- **Venue:** ICLR 2023 (Spotlight — Notable-top-25%)
- **Links:**
  - Paper (arXiv v2, 2023-04-15): https://arxiv.org/abs/2303.08133
  - PDF: https://arxiv.org/pdf/2303.08133
  - OpenReview (ICLR 2023): https://openreview.net/forum?id=0cpMApF9p6
  - Project page (interactive demos): https://meshdiffusion.github.io/
  - Semantic Scholar: https://www.semanticscholar.org/paper/MeshDiffusion%3A-Score-based-Generative-3D-Mesh-Liu-Feng/b7a783e3897baed760fb91cd1289dd0e353377f5
- **Code:** https://github.com/lzzcd001/MeshDiffusion — **MIT license**, 831★ / 42 forks, last commit 2024-05-20. Python ≥3.8, CUDA 11.6, PyTorch ≥1.6, PyTorch3D, ml_collections. Pretrained weights (res 64 + 128) for chair/car/airplane/table/rifle on HuggingFace + Google Drive mirror; processed DMTet datasets on HuggingFace.
- **Cite count:** 232 (Semantic Scholar, mid-2026); 14 influential
- **Funding:** Samsung Electronics, Max Planck Society, NSERC Discovery Grants (RGPIN-5011360, RGPIN-04653)
- **Read:** 2026-06-06 (Saturday, scholar weekly #14, ~50 min)

---

## TL;DR

**MeshDiffusion is the first diffusion model that generates 3D meshes *directly* — no post-hoc mesh extraction step** — by training a 3D-CNN U-Net score-based diffusion model on **DMTet (Deep Marching Tetrahedra)** parameters: 3D vertex deformations + SDF values stored on a regular tetrahedral grid initialized via body-centered cubic (BCC) tiling. The key tricks: (1) **BCC tiling preserves 3D translational symmetry** so a 3D CNN is well-posed on the grid; (2) **SDF values are normalized to ±1** to prevent the well-known DMTet sensitivity where tiny SDF noise creates large topology flips; (3) **two-pass reconstruction**: first fit a DMTet to every ground-truth mesh via differentiable rendering, then train a standard DDPM score model on the fitted tetrahedral grids. SoTA on ShapeNet chair/car/airplane/table/rifle (MMD 13.21/4.97/3.61/11.41/3.12, COV-CD 46.00/34.07/47.34/49.56/52.63 — best on most categories vs IM-GAN / SDF-StyleGAN / GET3D). Trained at resolution 64³ in 90k iterations on 8× A100-80GB for 2-3 days. The conditional generation story (Sec 4.4) is the most directly relevant for us: fix the observed vertices from a single RGBD view, let diffusion fill in the occluded ones via the **replacement method** — the analog of "fix the existing 30 teeth, generate the missing 2".

## Research question

> Voxel-based, point-cloud-based, and SDF-based 3D generative models all produce outputs that are **not** triangle meshes — they require a *post-hoc* mesh extraction step (marching cubes, marching tetrahedra, FlexiCubes, NDC). The extracted mesh is dense, over-tessellated, and **over-smoothed** — sharp features and artist-like triangulation are lost. **Can we generate a *mesh* directly, with sharp features and the topology flexibility to vary per-instance?**

Their answer: **yes — train a diffusion model on the *parameters* of a differentiable isosurface extractor (DMTet) rather than on a separate, post-hoc extraction.** DMTet is a function `(deformation, SDF) → mesh`, so by diffusing in `(deformation, SDF)`-space and then calling DMTet at the end, the output *is* a mesh. The output mesh inherits DMTet's properties: adaptive triangulation, sharp features, watertight, and **topology-varying** (since SDF sign-changes can create or destroy mesh components). This is the **mesh-as-state** alternative to MeshGPT's **mesh-as-sequence** (paper 013): MeshDiffusion is parallel + score-based; MeshGPT is autoregressive + discrete.

The four contributions (per Sec 1):
1. **A direct parametrization of meshes** using DMTet (regular tetrahedral grid, BCC tiling), enabling diffusion on a fixed-size tensor regardless of mesh topology.
2. **A simple 3D-CNN U-Net diffusion model** (not GNN) — argues CNN's translational-symmetry inductive bias wins over GNN here.
3. **SDF normalization** (sign-only, ±1) — a single design change that fixes DMTet's worst failure mode (topology flips from SDF noise).
4. **State-of-the-art results** on ShapeNet vs IM-GAN, SDF-StyleGAN, GET3D — including a 64³ resolution and a 128³ "inner-structure" extension.

The conceptual lineage: this is the **mesh-native diffusion model** that pairs with MeshGPT (paper 013) just as LION (paper 005) pairs with PVD (paper 012). The four papers together give a 2×2 grid of (representation × generative family): point-cloud (PVD, LION) + mesh (MeshGPT, MeshDiffusion) × denoising (PVD, MeshDiffusion) + latent (LION, MeshGPT) [where "latent" means compressing through an autoencoder first].

## Method

### 4.1 The DMTet parameterization (the *what* we're diffusing)

**DMTet** (Deep Marching Tetrahedra; Shen et al., NeurIPS 2021) is a differentiable isosurface extractor. It represents a mesh as:
- A **regular tetrahedral grid** in some cube (here, [-1,1]³) with vertices at the corners of tetrahedra
- At each vertex: a **3D deformation vector** `δᵢ ∈ ℝ³` (offsets the vertex from its initial position) + an **SDF value** `sᵢ ∈ ℝ` (negative inside, positive outside)
- A **triangulation rule** (marching tetrahedra) extracts the zero-isosurface of the SDF after deformation, producing a triangle mesh

The point: **deformation + SDF is a fixed-size tensor** (one 4-dim vector per tetrahedral vertex) regardless of how many triangles the output mesh has. Topology changes happen naturally — adding/removing a sign-change adds/removes a mesh component.

**BCC tiling trick:** instead of a naive tetrahedral grid (which has translation-breaking boundary vertices), use **body-centered cubic** tiling [Labelle & Shewchuk, SIGGRAPH 2007]. In BCC, each cell center has 8 tetrahedra around it, and the result is a uniform tetrahedral grid with full 3D translational symmetry — *except* for boundary cells. The authors **discard the boundary tetrahedra** to restore full translational symmetry, then **augment with a cubic lattice** (one extra vertex per BCC cell) and a **binary mask** indicating "BCC vertex" vs "augmented-cubic-lattice vertex". The 3D CNN can now stride cleanly through both vertex types, and the score-matching loss is masked to BCC vertices only.

### 4.2 The two-pass dataset preparation (the *how we got the data*)

DMTet is *differentiable* — you can optimize `(deformation, SDF)` to fit a target mesh via a rendering loss. The paper uses **two reconstruction passes**:

**Pass 1** — fit DMTet to ground truth mesh (ShapeNet, chair/car/airplane/table/rifle):
- Render multiview RGBD images (random cubemap lighting, diffuse-only material)
- Loss: `L = α_image L_image + α_depth L_depth + α_chamfer L_chamfer + α_SDF L_SDF` (L_silhouette implicit in L_image)
- α_image = α_chamfer = 1.0, α_depth = 100.0, α_SDF = 0.2 (linear decay 0.2 → 0.01)
- 5000 iterations, Adam, lr 5e-4
- Periodically scale SDFs by 0.4 for the first 2000 iters to "encourage topology to come from SDFs, not deformation"
- Cull floaters: set SDF to +1 outside visual hull
- 20-30 minutes per mesh on a Quadro RTX 6000

**Pass 2** — normalize SDFs to ±1 (sign only). This is the **single biggest design choice** in the paper. The authors observe that *tiny* SDF noise (±0.01) can flip a vertex's sign, which can change the topology of the output mesh (e.g., create a hole). The denoising L2 loss treats a 0.01 perturbation the same as a 0.5 perturbation, so it doesn't learn to avoid topology flips. By clamping to ±1, the SDF is now **discrete-valued**, and small noise is bounded.

Interestingly, the paper *does not* use a categorical/discrete diffusion model (D3PM-style). They empirically found that **treating the normalized SDFs as floats and training a Gaussian DDPM works** — the float-to-sign round is done as a final inference step. (Sec 4.3.)

### 4.3 The diffusion model itself

**Score-based SDE formulation** (Song et al. 2021) with discrete-time DDPM parameters (β_t schedule, 1000 steps). Network: a **3D U-Net** built by taking the DDPM 2D U-Net and swapping every 2D op for its 3D counterpart (3×3×3 convs, 3D pooling, 3D upsample).

Architecture details (Table 4):
- **Res 64:** base width 64, 5 pooling stages, attention at the 3rd encoder/decoder stage, total depth ~26 ResBlocks. ~30M params (estimated).
- **Res 128:** base width 128, 6 pooling stages, attention at the 4th stage, ~80M params.
- Input: `(B, 4, 64, 64, 64)` tensor = `[deformation (3) | SDF (1)]` plus a binary mask channel `[vertex_type]` (1 for BCC vertex, 0 for augmented-cubic-lattice).
- Output: predicted score / noise, same shape as input.
- **Loss is masked** to BCC vertices only (the augmented-cubic-lattice vertices are dummy carry-along inputs).

**Training:** 90k iterations, batch size 48, **8× A100-80GB** for 2-3 days per category. **Discrete-time, category-specific** — separate model per ShapeNet category (chair, car, airplane, table, rifle). **No hyperparameter tuning of the SDE** — uses the same β_t schedule as DDPM, which they call out as evidence that the method is "easy to train".

### 4.4 Conditional generation (the *closest thing to our use case*)

The paper's Sec 4.4 + Algorithm 2 is the most relevant for us. The task: given a single **RGBD** view of an object, generate the full 3D mesh (the "occluded" side is missing).

Two stages:
1. **Fit DMTet to the single RGBD view** (using the same reconstruction loss as Pass 1, but on one view). This produces a DMTet where some vertices are well-constrained (the visible side) and some are unconstrained (the occluded side).
2. **Run MeshDiffusion to "fix" the unconstrained vertices** using the **replacement method** (Ho et al. 2022b, a.k.a. classifier-free-guidance-style inference): at every reverse step, sample `x_{t-1}` from the model's prediction, then **replace** the well-constrained vertices' values with the forward-diffused version of the ground-truth `x_0` (i.e., sample `x_{t-1}^{fixed} ~ p(x_{t-1} | x_0^{fixed})`). The model then sees partial-correct input at every step and "fills in" the unconstrained vertices.

**Optionally**, near the end of the denoising process (last T=50 steps of 1000), allow the fixed vertices to update *slightly* — empirically, the single-view fit is imperfect even for the visible side, so this gives a small correction.

This is **architecturally identical** to what we'd want for outer-surface generation from a partial intra-oral scan: fix the existing 30 teeth's DMTet, let diffusion generate the missing 2.

## Results

### Unconditional generation (Table 1, ShapeNet, lower is better for MMD/JSD, higher for COV)

| Category | Method | MMD-CD | COV-CD | 1-NNA-CD | JSD |
|---|---|---|---|---|---|
| Chair | IM-GAN | 13.928 | 49.64 | 58.59 | 6.298 |
| Chair | SDF-StyleGAN | 15.763 | 45.60 | 63.25 | 6.846 |
| Chair | GET3D | 15.972 | 43.36 | 75.26 | 4.732 |
| **Chair** | **MeshDiffusion** | **13.212** | **46.00** | **53.69** | 5.038 |
| Car | GET3D | 6.243 | 15.04 | 75.26 | 69.107 |
| **Car** | **MeshDiffusion** | **4.972** | **34.07** | **81.43** | 12.384 |
| **Airplane** | **MeshDiffusion** | **3.612** | **47.34** | **66.44** | **11.366** |
| **Rifle** | **MeshDiffusion** | **3.124** | **52.63** | **57.68** | **19.353** |
| **Table** | **MeshDiffusion** | **11.405** | **49.56** | **59.35** | **4.310** |

**MeshDiffusion wins on MMD on all 5 categories and COV-CD on 4/5.** GET3D wins on JSD on chair/car (3D FID analog) and 1-NNA-CD on car — the latter is consistent with MeshDiffusion generating more *diverse* shapes that are slightly farther from the validation set.

### Ablation (Table 3, Chair)

| Variant | MMD | COV-CD | 1-NNA-CD |
|---|---|---|---|
| Full MeshDiffusion | **13.212** | 46.00 | **53.69** |
| w/o smoothing post-proc | 13.885 | 43.36 | 60.88 |
| w/o SDF normalization | 14.324 | 44.76 | 63.94 |
| GAN on tetrahedra | 16.116 | 45.13 | 72.97 |

- **SDF normalization** is worth -1.1 MMD and -10 1-NNA — the single most important design choice.
- **Smoothing** (Laplacian, λ=0.25, 5 iters) is worth -0.7 MMD and -7 1-NNA — important for clean surfaces.
- **Diffusion > GAN** by 2.9 MMD and 19 1-NNA — the headline H2-style result ("diffusion > GAN") holds even in the mesh-native setting.

### Inner-structure generation (Table 5, Car, res 128)

MeshDiffusion is **explicitly trained with 3D information**, so it generates inner structure (engine block, seats) that GET3D (which only optimizes 2D rendering) misses. **MMD-CD 6.80 vs GET3D's 9.85 (31% improvement), COV-CD 64.13 vs 42.93 (50% improvement)** on surface point clouds of the *same* generated cars. **This is the strongest H4 result in our reading list** — direct mesh generation with inner structure is competitive with, and beats, SDF-based + FlexiCubes pipelines for inner-geometry-aware generation.

### Qualitative (Figure 1, 5, 12-16)

Generated chairs, cars, airplanes, rifles, tables — all with **sharp features** (car fenders, rifle barrels, table legs), **adaptive triangulation** (more triangles at high-curvature regions), and **clean topology** (no marching-cubes-style aliasing). Compared to GET3D (Figure 10), MeshDiffusion has fewer visual artifacts and more "novel-but-coherent" samples.

### Interpolation (Sec 5.4, Figure 7)

DDIM inference (100 steps) with spherical interpolation in latent noise space produces smooth shape interpolations — useful for the "give the dentist a slider" UX idea from LION (paper 005).

## Connections to our hypotheses

- **H1 (2-stage > 1-stage): STRONG support.** The paper is a 2-stage architecture by construction: (1) DMTet-fit to ground truth, (2) diffusion on the fitted grids. The alternative would be end-to-end: train a diffusion model that *renders* and back-propagates a 2D loss — but the paper explicitly argues against this (Sec 4.2: "the differentiable renderer is useful only during the tetrahedral grid creation process"). The 2-stage approach is what makes the 2-3 day training time tractable; end-to-end alternatives like GET3D need a discriminator + adversarial training and are less stable. **For us: this is direct evidence that the v0 stack should keep generation (LION/Diffusion-SDF/PVD) separate from output rendering (DiGS+FlexiCubes).**

- **H2 (diffusion on point cloud > mesh-based VAE): REFINES, mild contradiction.** The paper is *diffusion on mesh*, and it beats every VAE-based mesh method in Table 1 (IM-GAN, SDF-StyleGAN, GET3D). **This actually *strengthens* the diffusion half of H2 (diffusion > GAN/VAE) but *weakens* the point-cloud half of H2 (point cloud > mesh)**, because it shows direct-mesh diffusion is competitive. **Restate H2 as: "diffusion on latent 3D representations (point cloud, SDF, or mesh) > explicit mesh VAE."** The representation choice is then secondary; the *generative family* (diffusion vs VAE) is primary.

- **H3 (conditioning on adjacent + opposing teeth): STRONG support.** The conditional generation (Sec 4.4) is *exactly* the H3 template: fix observed vertices (analog of "existing 30 teeth"), let diffusion generate missing vertices (analog of "missing 2 teeth"). The replacement method is the cleanest implementation of the H3 conditioning we've seen — it requires *no* encoder, *no* cross-attention, just a partial observation that's held fixed through the reverse process. **For us: a tooth-completion pipeline using MeshDiffusion is the most architecturally-faithful H3 implementation in our reading list.**

- **H4 (implicit SDF > explicit mesh): REFINES — direct contradiction on the "implicit" half, support for the "SDF" half.** The paper uses SDFs *inside* a mesh representation, so it's a *mesh* output, not an *implicit* one. The "SDF" half of H4 (SDF is the right data substrate for distance fields, sharpness, watertightness) is supported — DMTet's SDF-driven isosurface extraction gives clean sharp features. But the "implicit" half of H4 (SDF should be the *output* representation, decoded to mesh on demand) is **contradicted** — MeshDiffusion shows the *mesh* is the right output representation, and DMTet is a differentiable decoder for it. **For us: this suggests the v0 stack may want a "mesh-output" final stage (DMTet or FlexiCubes) rather than holding the SDF around indefinitely.**

- **H5 (synthetic data bootstraps training): STRONG support.** The paper is trained on ShapeNet only — pure synthetic. The 50k iterations of differentiable rendering per ShapeNet mesh is the "synthetic data preparation pipeline" that bootstraps the diffusion model. **For us: this is the cleanest precedent for a "synthetic CAD library → DMTet → diffusion" v0 pipeline.**

- **H6 (mesh output requires post-processing): partial contradiction.** The paper's outputs require Laplacian smoothing (λ=0.25, 5 iters) and remeshing as a *post-processing* step (Sec 5.2). So H6 ("output a watertight, print-ready mesh") is *not* solved by the generative model alone — but the post-processing is mild (a few seconds of CPU time, well-understood algorithms). **For us: budget 10-30s of post-processing time per generated crown.**

## Surprises / interesting things buried in section 4

- **SDF normalization is the single biggest design lever.** Ablation: -1.1 MMD and -10 1-NNA from a one-line change (clamp SDFs to ±1). The intuition: DMTet's sign function is discontinuous at 0, so any L2 loss is going to underweight the boundary. Clamping to ±1 makes the SDF *qualitatively* discrete without needing a discrete diffusion model. The author note that they "empirically found that it suffices to treat the normalized SDFs as float numbers" — so D3PM-style categorical diffusion isn't necessary. This is **counterintuitive** and worth a follow-up paper.

- **CNN beats GNN for tetrahedral grids.** Most prior work on DMTet used GNNs (e.g., DMTet's original paper). The authors argue 3D CNN wins because: (a) translational symmetry of BCC tiling is *exactly* the inductive bias of 3D convs, (b) CNNs have lower memory cost at the resolutions we care about (64³-128³). They don't ablate this — would be a useful follow-up.

- **2-3 days on 8× A100-80GB for one category.** Per category, not per dataset. So the full ShapeNet-55 reproduction would be ~150-200 A100-days. **For us: the v0 budget is single-class, but the "1 day per tooth class" extrapolation means the full dentition (4 classes × 7 tooth positions) is ~28 A100-days = ~$1,500 on Lambda.** Comparable to LION, more expensive than PVD.

- **Resolution ceiling at 64³ is a hard limitation.** The authors state that "fine details cannot be fully captured with the current resolution of 64 during the dataset creation stage" and the 128³ extension is mentioned as a workaround that increases memory 8× and time ~2-3×. **For dental use: cusps and fissures are ~200-500μm features, requiring at least 128³ to resolve, and the post-processing smoothing will erase any sub-cusp detail. This is a real ceiling on what MeshDiffusion can produce for our use case.**

- **Sec 4.2's two-pass reconstruction is a template for our data pipeline.** Step-by-step: (1) fit DMTet to ground-truth mesh via differentiable rendering with image + depth + chamfer + SDF losses, (2) clamp SDFs to ±1, (3) train diffusion. For us, this means: (1) fit a DMTet to every 3DTeethSeg22 tooth, (2) clamp SDFs, (3) train a per-FDI-class diffusion model. The "render the DMTet, compute losses, backprop" loop is **already implemented in the MeshDiffusion GitHub** — we'd just swap the dataset from ShapeNet to 3DTeethSeg22.

- **The cull-floaters trick (Sec A.2).** Setting SDF to +1 for tetrahedral vertices outside the visual hull is a hack, but it's a *crucial* hack — without it, the diffusion model generates floating disconnected blobs in the empty space. **For us: when fitting DMTet to a partial intra-oral scan, we'd set SDF to +1 for tetrahedra outside the partial scan's visual hull, so the diffusion doesn't hallucinate in unseen regions.**

- **Inner-structure generation is the killer feature (Sec G, Table 5).** MeshDiffusion at 128³ generates inner car structures (engine block, seats) that GET3D misses. The metric: surface-point-cloud MMD 6.80 vs 9.85 (31% better), COV 64.13% vs 42.93% (50% better). This is the **strongest H4 contradiction in our reading list** — the paper is a mesh-output method (not implicit), and it produces better *inner* structure than the SDF-based GET3D baseline. **For us: a tooth's interior (pulp chamber, dentin-enamel junction) is not our target, but a tooth's *intaglio surface* (the inside of the crown that sits on the prep) is — and inner-structure competence suggests this might just work.**

## Quote-worthy sentences

- **On the central thesis:** "Compared to other 3D representations like voxels and point clouds, meshes are more desirable in practice, because (1) they enable easy and arbitrary manipulation of shapes for relighting and simulation, and (2) they can fully leverage the power of modern graphics pipelines which are mostly optimized for meshes." (Abstract)
- **On SDF normalization:** "it is more desirable for the underlying shape to be captured mostly by the topology implied by SDF values, not the vertex deformation" (Sec A.2) — the design rationale for why SDF normalization + periodic SDF rescaling pushes the model toward SDF-driven topology changes.
- **On CNN vs GNN:** "we argue that it is better to use convolutional neural networks (CNNs) which generally have better model capacity and contextual information than GNNs due to the embedded spatial priors in the convolution operators." (Sec 4.1) — a strong (and probably correct) claim specific to BCC-tiled tetrahedral grids.
- **On limitations:** "the differentiable renderer is useful only during the tetrahedral grid creation process, while in principle we believe there can be ways to incorporate the differentiable render in the training and inference process of diffusion models." (Sec 6) — the most honest admission of where the paper falls short.
- **On topology sensitivity:** "the numbers of vertices and faces are indefinite for general object categories, and the underlying topology varies wildly and edges have to be generated at the same time." (Sec 1) — the key challenge that DMTet parameterization solves.
- **On why meshes matter for printing (our angle):** "modern graphics pipelines are built and optimized for explicit geometry representations like meshes, making meshes one of the most desirable final 3D shape representations." (Sec 1) — directly applicable: STL files for 3D printing *are* meshes; the entire printing pipeline (slicing, support generation, G-code) expects mesh input.

## Code/data link

- **Code (MIT):** https://github.com/lzzcd001/MeshDiffusion — Python ≥3.8, CUDA 11.6, PyTorch ≥1.6, PyTorch3D, ml_collections. Includes the differentiable DMTet renderer, the two-pass reconstruction training script, and the diffusion training script.
- **Pretrained models (chair, car, airplane, table, rifle at res 64 + chair/car at res 128):** HuggingFace `lzzcd001/MeshDiffusion_models`, plus Google Drive backup.
- **Processed DMTet datasets (res 64 cubic grids):** HuggingFace `lzzcd001/MeshDiffusion_DMTet_Dataset` — useful as a reference for our own DMTet-fits-to-3DTeethSeg22 pipeline.
- **No pre-trained dental model** — would need to train from scratch on 3DTeethSeg22.

## For our project

**Concrete next steps (in order of priority):**

1. **Promote MeshDiffusion to the v0 short-list for sub-task 4 (outer surface generation).** Currently the v0 stack is PVD-AF-DiGS-FC. Add **PVD-AF-DiGS-FC + MeshDiffusion-on-DMTet as an alternative path** that skips the SDF-lifting step and produces a mesh directly. The conditional generation from a partial scan (Sec 4.4) is the most direct H3 implementation in our reading list.

2. **Adopt the two-pass reconstruction as our v0 data preparation pipeline.** For every tooth in 3DTeethSeg22: (a) render 8-16 views with random cubemap lighting, (b) fit a DMTet via the `L_image + L_depth + L_chamfer + L_SDF` loss, (c) clamp SDFs to ±1, (d) save the (deformation, SDF) tensor as the training data. This is **exactly** what the MeshDiffusion GitHub does for ShapeNet, and we can repurpose the code. Estimated: ~1 hour per tooth on a single A100, so 3DTeethSeg22's ~24,000 teeth = ~24,000 A100-hours = ~$50,000 on Lambda. **This is too expensive for a v0 — pilot on a 1,000-tooth subset first ($2,000) and see if the conditional generation results justify the full dataset.**

3. **Adopt the cull-floaters trick (Sec A.2) for partial-scan inference.** At inference time, when the input is a partial arch (most teeth observed, some missing), set SDF to +1 for tetrahedral vertices outside the observed region. This prevents the diffusion model from hallucinating in unseen regions — **exactly the dental clinical fit property we need** (don't generate a crown that floats above the gum).

4. **Adopt the SDF normalization ablation insight for our DiGS+FlexiCubes stack (paper 003 + 007).** DiGS produces a continuous SDF; the FlexiCubes (paper 007) extractor is sensitive to the *gradient* of the SDF, not its absolute value. **If we normalize the DiGS output to ±1 before extraction (analog of MeshDiffusion's SDF clamping), we may get cleaner topologies with fewer holes** — a 1-line code change worth piloting on our v0.

5. **Compute budget update for v0:**
   - PVD-AF-DiGS-FC: ~$2,200 (unchanged)
   - **MeshDiffusion-on-DMTet pilot (1,000-tooth): ~$2,500** (mostly the DMTet fitting, not the diffusion training)
   - Full 3DTeethSeg22 MeshDiffusion (24,000 teeth): **~$50,000** — out of v0 scope, queue for v1
   - v0 decision gate: pilot PVD-AF-DiGS-FC *and* MeshDiffusion-on-DMTet on the same 1,000-tooth subset, pick the one with the better intaglio fit (< 50μm margin gap) and outer-surface diversity.

6. **Open question for HK: diffusion-on-DMTet vs autoregressive-on-mesh (MeshGPT, paper 013)?** MeshGPT is autoregressive + discrete (VQ codebook over faces), MeshDiffusion is score-based + continuous. For dental use:
   - MeshGPT's 30-90s/mesh inference is too slow for a clinical workflow.
   - MeshDiffusion's ~10-30s/mesh inference (1000 DDPM steps) is also slow but DDIM can cut to 100 steps in 1-3s.
   - **Recommendation: MeshDiffusion for the v0 outer-surface prototype (DDIM inference, 100 steps), MeshGPT only as a research comparison** (and the non-commercial Audi license is a production blocker).

7. **Architectural lesson for the H2 final form: latent diffusion on DMTet.** Sec 6 explicitly notes that "it is a promising approach to train diffusion models on the latent space produced by a regularized autoencoder, the same strategy adopted in [LION, Diffusion-SDF]." So the **2-stage autoencoder-then-diffusion** pattern of LION could be applied to DMTet parameters: train a DMTet autoencoder (encoder: mesh → (δ, s) tensor; decoder: DMTet isosurface), then train a latent DDM in the encoded (δ, s)-space. This would give us a **LION-architecture analog in mesh space** — call it **LionMesh** (informal name for the v1 candidate).

8. **Update the v0 architecture name to reflect the H2/H4 restatement:**
   - Old: PVD-AF-DiGS-FC (point-cloud DDM + completion encoder + SDF lifting + FlexiCubes)
   - New: **PVD-MD-DiGS-FC** (point-cloud DDM as primary, MeshDiffusion on DMTet as alternative mesh-direct path, DiGS as SDF-lifting, FlexiCubes as final mesh extractor)
   - The stack now has **two parallel generative paths** (point-cloud DDM and mesh-native DDM) that converge on DiGS+FlexiCubes for the SDF+mesh output.

**Compute note:** Pilot budget (PVD-AF-DiGS-FC + MeshDiffusion-on-DMTet, 1,000 teeth) is ~$4,500 on Lambda, less than the previous $4,000 estimate was — well within v0 scope.

**Next paper to read:** **CIGS** (Zhang et al., CVPR 2024) — the latest 2024 SoTA diffusion-on-implicit-field that closes the H2 × H4 intersection with a different architectural choice (continuous normal field as the diffusion target, not a discrete tetrahedral grid). Or **DPM** (Luo & Hu, CVPR 2021) for the transformer-based point-cloud DDM that tests whether a better backbone helps H2. Or **PolyDiff** (Alliegro et al., ICCV 2023) — the autoregressive-diffusion-on-mesh hybrid that combines the best of MeshGPT and MeshDiffusion.

## Reference

```bibtex
@inproceedings{liu2023meshdiffusion,
  title={MeshDiffusion: Score-based Generative 3D Mesh Modeling},
  author={Liu, Zhen and Feng, Yao and Black, Michael J and Nowrouzezahrai, Derek and Paull, Liam and Liu, Weiyang},
  booktitle={ICLR},
  year={2023},
  note={Spotlight}
}

@article{liu2023meshdiffusion_arxiv,
  title={MeshDiffusion: Score-based Generative 3D Mesh Modeling},
  author={Liu, Zhen and Feng, Yao and Black, Michael J and Nowrouzezahrai, Derek and Paull, Liam and Liu, Weiyang},
  journal={arXiv preprint arXiv:2303.08133},
  year={2023}
}
```

---

*Scholar's note: MeshDiffusion is the missing fourth pillar in our 2×2 grid of (representation × generative family) for 3D shape generation: point-cloud (PVD, LION) × mesh (MeshGPT, MeshDiffusion) × denoising (PVD, MeshDiffusion) × latent (LION, MeshGPT). It directly supports H1 (2-stage), H3 (conditional generation from partial observation), H5 (synthetic-only), and refines H2 (diffusion > VAE, but representation is secondary) and H4 (SDF > no-SDF, but mesh > implicit is the output). The conditional generation from partial scan (Sec 4.4) is the cleanest H3 implementation in our reading list — fix the observed vertices, fill in the missing ones via diffusion replacement. Recommended reading order: read MeshDiffusion *after* MeshGPT (paper 013) to compare autoregressive vs score-based mesh generation. Action item for Red: pilot MeshDiffusion-on-DMTet on a 1,000-tooth subset of 3DTeethSeg22 as a v0 alternative to PVD-AF-DiGS-FC. Action item for HK: decide if we want the "two parallel generative paths" architecture (PVD-MD-DiGS-FC) for v0, or if MeshDiffusion's compute cost ($50K for full 3DTeethSeg22) is a v1-only concern.*
