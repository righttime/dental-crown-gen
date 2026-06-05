# 007 — FlexiCubes

- **Title:** *Flexible Isosurface Extraction for Gradient-Based Mesh Optimization*
- **Authors:** Tianchang Shen*, Jacob Munkberg, Jon Hasselgren, Kangxue Yin, Zian Wang, Wenzheng Chen, Zan Gojcic, Sanja Fidler, Nicholas Sharp*, Jun Gao* (*equal contribution; NVIDIA + University of Toronto + Vector Institute)
- **Year:** 2023
- **Venue:** SIGGRAPH 2023 (ACM Transactions on Graphics, Vol. 42, Issue 4, Article 37, August 2023)
- **Links:**
  - arXiv: https://arxiv.org/abs/2308.05371
  - DOI: https://doi.org/10.1145/3592430
  - Project page: https://research.nvidia.com/labs/toronto-ai/flexicubes/
  - YouTube talk: https://www.youtube.com/watch?v=2zuvMkj1Vi8
  - Code: https://github.com/nv-tlabs/FlexiCubes (Apache 2.0, NVIDIA copyright)
  - Kaolin integration: v0.15.0+ (https://kaolin.readthedocs.io/en/latest/notes/volumetric_meshes.html)

---

## TL;DR

**FlexiCubes is the *learnable, differentiable* isosurface extractor that bridges "scalar field on a grid" and "watertight printable triangle mesh" for *gradient-based optimization* of meshes — the missing piece for end-to-end crown generation.** It is built on top of **Dual Marching Cubes** (Nielson 2004, an obscure but manifoldness-guaranteeing variant of Dual Contouring) and adds **four sets of learnable, differentiable parameters per cell** (interpolation weights `α ∈ R⁸` for edge crossings, dual-vertex weights `β ∈ R¹²` for positioning inside the primal face, quad-split weights `γ` for choosing the triangulation, and grid-vertex deformations `δ ∈ R³` from DMTet), all constrained so the resulting mesh stays **manifold, almost-always-watertight, and intersection-free in the vast majority of cases**. The paper shows the resulting extractor is the **first to satisfy both** *Grad* (gradient-based optimization is well-defined and converges) and *Flexible* (vertices can adjust locally to fit sharp features) — a property the taxonomy in Table 1 marks with a checkbox that no prior method has. Empirical results: **34.9% inaccurate-normals** at 64³ on Myles/ShapeNet vs DMTet's 48.7% and NDC's 55.2% (Table 2); **FID 17.51 on ShapeNet chairs** in GET3D vs DMTet's 22.41 (Table 6); and the only method that **survives an equilateral-triangle regularizer** without geometric collapse (CD 5.46 vs DMTet's 6.69, baseline 5.17 — Table 4). For us: **FlexiCubes is the successor to NDC** (paper 006). It is the right mesh-extraction module for the next prototype — *especially* if we want to (a) end-to-end fine-tune the implicit field against mesh-quality losses (e.g., developability for splint fabrication, equilateral triangles for FEM simulation, anatomical features), or (b) extract a tetrahedral mesh for downstream physics-based fitting (margin gap, contact stress). The catch: ~5× more memory than NDC/MC at 64³ (116 MB vs 12 MB) and rare self-intersections that need a cleanup pass before 3D printing.

## Research question

> Can we build an isosurface extractor that is **both differentiable enough for gradient-based optimization to converge** *and* **flexible enough to fit sharp features**, even though these two properties are inherently in conflict?

Their answer: **yes, by abandoning classic Marching Cubes (inflexible) and classic Dual Contouring (non-differentiable) and instead building on Dual Marching Cubes (DMC) with carefully-chosen, carefully-constrained learnable parameters per cell.** The paper's central insight is that **DMC's vertex-placement logic is naturally a convex combination** of primal face crossings — and the moment you make those weights learnable (and pass them through a `tanh+1` activation to keep them positive and bounded), you get a *Flexible* module whose gradients are well-behaved *because* the parameters only enter as convex combinations, which means the output vertex is guaranteed to stay inside the cell (no explosions, no NaNs). Then you add a quad-split weight `γ` (which diagonal to triangulate along) and a grid-deformation `δ` (per-grid-vertex displacement, capped at half a grid spacing) to round out the flexibility.

The three key claims:
1. **FlexiCubes is the first isosurfacer to satisfy both *Grad* and *Flexible* in Table 1's taxonomy.** (Marching Cubes has Grad but not Flexible; DC has Flexible but not Grad; DMC-centroid has neither by default; DMTet has Grad but not Flexible; NDC has neither in the optimization sense.)
2. **The added flexibility produces measurably better meshes in *every* downstream task tested** — mesh reconstruction, mesh regularization, photogrammetry, generative modeling, even differentiable physics simulation. And critically: FlexiCubes is the only one that *survives* a mesh-based regularizer (e.g., equilateral triangles) without losing geometric accuracy.
3. **The architecture is a drop-in replacement for DMTet** in existing pipelines (nvdiffrec, GET3D) and produces better results with no other changes. This is the killer test of a representation: swap it in, the rest of the pipeline is identical, and you get better outputs.

## Method

### Inputs and outputs

- **Input:** a regular grid of scalar values `s(x) ∈ ℝ` at resolution `N³` (typically 32³ to 256³). The same architecture handles:
  - **SDF** (signed distance function) — most natural for our Diffusion-SDF / DiGS / DeepSDF pipeline.
  - **Raw occupancy / binary voxels** — comes from voxelized CNNs.
  - **Per-cell learnable parameters** that *also* get fed in: `α ∈ R⁸`, `β ∈ R¹²`, `γ ∈ R` per cell, and `δ ∈ R³` per grid vertex. These are 20 + 1 = 21 scalars per cell beyond the SDF value, **or** predicted on the fly by a neural network head (the GET3D + FlexiCubes variant has the generator's last layer output all 21 + the SDF for each cell).
- **Output:** a triangle mesh (`.obj`/`.ply`/trimesh) that is **manifold and almost-always watertight**, with sharp features preserved, intersection-free in the vast majority of cases, and quad-dominant (each quad is split into 2 triangles at the end). Also optionally a **tetrahedral mesh** of the interior volume, or a **hierarchical adaptive-resolution mesh** if an octree is provided.

### Architecture

The "architecture" is a hand-coded, fully-differentiable mesh-extraction operator (a forward, in PyTorch with a custom C++/CUDA kernel). There is no learned 3D U-Net here — that's NDC (paper 006). The *learnable part* is the per-cell and per-grid-vertex parameters, and the "training" is the *outer optimization* (the generator network, the photogrammetry pipeline, etc.) that produces those parameters.

Three new sets of parameters per cell:
- **`α ∈ R⁸`** per cell (one weight per cube corner, activated through `tanh(·)+1 ∈ [0, 2]`). Replaces the standard edge-crossing formula `u_e = (s(x_a)·x_b − s(x_b)·x_a) / (s(x_b) − s(x_a))` with the weighted version `u_e = (s(x_i)·α_i·x_j − s(x_j)·α_j·x_i) / (s(x_i)·α_i − s(x_j)·α_j)`. **This is the key trick**: the `α` weights let the edge crossing *move along the edge* away from the naive linear interpolation, but only inside the convex hull of the edge endpoints.
- **`β ∈ R¹²`** per cell (one weight per cube edge, `tanh(·)+1` activated). Replaces the standard dual-vertex position (centroid of primal face) with a weighted centroid: `v_d = (Σ β_e · u_e) / (Σ β_e)`. **This is the second key trick**: `β` lets the dual vertex move *inside* its primal face (a 2D region), but the convex-combination form guarantees it stays in the cell.
- **`γ ∈ R`** per cell, propagated to the emitted vertices. At optimization time, each quad is split into 4 triangles with an interpolated midpoint whose position depends on the product `γ_c1 · γ_c3` vs `γ_c2 · γ_c4` (the two diagonals). At inference time, the quad is split into 2 triangles along whichever diagonal has the larger product. **This is the third key trick**: `γ` selects the triangulation that best fits the geometry.
- **`δ ∈ R³`** per grid vertex (from DMTet). Deforms the grid vertices in space, capped at half a grid spacing so cells never invert. This is the spatial-alignment degree of freedom.

Total: **21 parameters per cell** (8 + 12 + 1) plus **3 per grid vertex**. In the GET3D application, the generator's last layer is modified to output all 21 + the SDF value for each cell — so the parameter count is a constant overhead, not a network size change.

### Optimization

- **Forward pass:** O(N³) time, single pass over all cells. Per cell: compute the 23 DMC configurations from Fig. 7 (rotationally-unique cases), look up the connectivity, compute the dual vertex positions via the weighted formulas, emit the faces. Implemented in C++/CUDA, ~9ms at 64³ on an A100.
- **Backward pass:** O(N³) time, autograd through the (convex-combination) forward, ~7ms at 64³. **The convex-combination form is what makes the backward well-behaved** — gradients are bounded because the parameters only enter the output as coefficients in a weighted average.
- **Two regularizers** specific to FlexiCubes (Sec. 4.7) that *all* examples in the paper use:
  - **`L_dev`** = mean-absolute-deviation of the distances from each dual vertex to the edge crossings of its primal face. Keeps vertices near the centroid of the cell so they have "margin" to flex in either direction.
  - **`L_sign`** = cross-entropy of `sign(s_a)` vs `sign(s_b)` over all grid edges with `sign(s_a) ≠ sign(s_b)`. Discourages spurious surface sheets in unsupervised regions (e.g., internal cavities when only image supervision is used).
- **Optional equilateral-triangle regularizer** (Sec. 5.2): penalizes the variance of edge lengths on the extracted mesh. The paper shows this is the killer test — DMTet and MC *collapse* geometrically when this regularizer is added, but FlexiCubes barely budges.

### Extensions

- **Tetrahedral mesh extraction** (Sec. 4.5): extends the strategy of Liang & Zhang (2014) to DMC. Vertex set is the union of grid vertices, extracted surface vertices, and cell midpoints; tetrahedra are emitted according to a fixed lookup table. Used in the differentiable physics simulation application (Sec. 6.4) where a FEM solver needs a tet mesh of the volume.
- **Adaptive octree resolution** (Sec. 4.6): hierarchical grid with varying resolution in high-detail regions. The connectivity rule is the same as uniform, with one extra constraint — refined octree grid vertices adjacent to coarser cells always take their value as the interpolated value from the coarser face vertices, guaranteeing sign consistency and avoiding cracks (Fig. 11–12). The paper shows this reduces CD from 4.5 (uniform 32³) to 2.5 (adaptive 64³) to 2.4 (adaptive 128³) on the same shape, with the same or fewer total triangles.

## Results

### Mesh reconstruction (Sec. 5.1, Table 2)

Dataset: 79 shapes from Myles et al. 2014 (AIM@Shape + community CAD models), spans noisy scans to highly-detailed CAD. Metrics: `IN>5°` (fraction of mesh normals with > 5° angle error vs ground truth), CD (Chamfer Distance × 10⁻⁵), F1 (F-score at 1% threshold), ECD (Edge Chamfer Distance × 10⁻²), EF1 (Edge F1 Score).

| Method | IN>5° ↓ | CD ↓ | F1 ↑ | ECD ↓ | EF1 ↑ |
|---|---|---|---|---|---|
| **32³** MC | 66.61 | 9.11 | 0.54 | 2.60 | 0.13 |
| 32³ DMTet(32) | 66.22 | 11.56 | 0.52 | 3.64 | 0.17 |
| 32³ **FlexiCubes** | **50.52** | **7.01** | **0.64** | **2.11** | **0.26** |
| **64³** MC | 52.37 | 6.33 | 0.66 | 1.25 | 0.25 |
| 64³ DMTet(80) | 48.66 | 5.17 | 0.66 | 3.59 | 0.29 |
| 64³ **FlexiCubes** | **34.87** | **4.87** | **0.70** | **0.71** | **0.43** |
| **128³** MC | 42.56 | 4.51 | 0.72 | 1.32 | 0.44 |
| 128³ DMTet(128) | 48.86 | 4.98 | **0.74** | 1.50 | 0.39 |
| 128³ **FlexiCubes** | **30.57** | **4.31** | 0.71 | **0.42** | **0.51** |

**Key insight:** FlexiCubes has the lowest IN>5° (sharp-feature preservation) by a *huge* margin (34.87 vs DMTet's 48.66 at 64³, a 28% relative reduction), the lowest CD and ECD at every resolution, and the highest EF1 (edge accuracy). MC and DMTet both have a sharp-features problem. NDC (paper 006) has comparable quality *but* NDC is non-differentiable, so it cannot be used in this optimization loop. This is the bottom-line argument for FlexiCubes: **it is the only method that gets NDC-quality output *and* is differentiable enough to be the last layer of a generator**.

### Ablation (Table 3, 64³)

| Variant | IN>5° ↓ | CD ↓ | F1 ↑ | ECD ↓ | EF1 ↑ |
|---|---|---|---|---|---|
| DMC centroid (baseline) | 53.02 | 5.85 | 0.65 | 2.60 | 0.19 |
| + flex vertex (β) | 40.88 | 5.34 | 0.68 | 0.99 | 0.37 |
| + grid deform (δ) | 39.46 | 5.01 | 0.69 | 0.98 | 0.41 |
| + flex quad split (γ) | **34.87** | **4.87** | **0.70** | **0.71** | **0.43** |

**Key insight:** Every component matters, and the cumulative effect is non-trivial — `IN>5°` drops from 53 to 35 (a 34% relative reduction). **For us: when we adopt FlexiCubes, do not skip the `γ` (quad-split) parameter** — it's the cheapest one to add (1 scalar per cell) and contributes meaningfully to the final result. The `δ` (grid-deform) is what makes the grid *spatially adaptive*, which is critical for capturing cusps and fissures on teeth.

### Mesh regularization (Table 4, 64³ with equilateral-triangle regularizer)

| Method | IN>5° ↓ | CD ↓ | Aspect > 4 ↓ | Radius > 4 ↓ | MinAngle < 10° ↓ |
|---|---|---|---|---|---|
| MC (no reg) | 52.37 | 6.33 | 11.71 | 11.71 | 11.84 |
| DMTet (no reg) | 48.66 | 5.17 | 17.31 | 16.68 | 17.83 |
| **FlexiCubes (no reg)** | **34.87** | **4.87** | 2.93 | 4.49 | 2.04 |
| MC + reg | 50.16 | 8.56 | 11.46 | 11.43 | 11.62 |
| DMTet + reg | 67.65 | 6.69 | 0.29 | 0.46 | 0.26 |
| **FlexiCubes + reg** | 41.05 | 5.46 | 0.39 | 0.69 | 0.24 |

**Key insight:** **This is the killer table for our project.** The bottom-line claim is: DMTet and MC have so few degrees of freedom in their mesh layout that adding a triangle-quality regularizer *forces them to lose geometric accuracy* (DMTet CD goes from 5.17 → 6.69, a 30% relative degradation). **FlexiCubes barely budges** (CD 4.87 → 5.46, a 12% relative degradation, while triangle quality improves by an order of magnitude). **For us: this means we can add a *clinical-quality* regularizer** (e.g., penalize the surface area of the intaglio face that falls below the margin line, or enforce that the occlusal surface is convex) **without losing occlusal anatomy** — a property no other mesh extractor has. This is the most important empirical finding in the paper for our use case.

### Generative modeling (Table 6, GET3D on ShapeNet, FID score)

| Method | Motorbike ↓ | Chair ↓ | Car ↓ |
|---|---|---|---|
| DMTet | 48.90 | 22.41 | 10.60 |
| **FlexiCubes** | **44.87** | **17.51** | **9.55** |

**Key insight:** Drop-in replacement of DMTet with FlexiCubes in the GET3D generator (a state-of-the-art 3D generative model that *only* uses 2D image supervision) yields a 4–5 point FID improvement across categories. The qualitative figures (Fig. 23) show cleaner thin structures (motorbike spokes, chair legs) and more uniform surfaces. **For us: the same drop-in pattern will work for our pipeline** — if we use a GET3D-style GAN or a diffusion model that outputs an implicit field, swapping the mesh extractor from MC/NDC to FlexiCubes should give us a 4–5 point quality improvement for free. This is the cleanest empirical argument for adopting FlexiCubes over NDC.

### Photogrammetry (Table 5, nvdiffrec on NeRF synthetic, view-interpolation PSNR and CD)

PSNR is essentially tied with DMTet (within 0.2 dB across 8 scenes). **But CD is dramatically better on visible triangles** — e.g., Chair CD 0.45 vs 4.51 (10× better), Ship CD 10.5 vs 55.8 (5× better). The PSNR tie is expected (image-space supervision); the CD improvement is the meaningful signal: the extracted mesh *better* matches the true geometry even when the rendered images look the same.

### Performance (Table 7, 64³ on A100)

| Method | Forward (ms) | Backward (ms) | Memory (MB) |
|---|---|---|---|
| MC | 2.28 | 0.43 | 12.05 |
| DMTet | 2.33 | 1.38 | 22.44 |
| DMC centroid | 4.97 | 1.69 | 25.08 |
| **FlexiCubes** | **8.93** | **7.32** | **116.56** |

**Key insight:** FlexiCubes is ~4× slower in forward, ~5× slower in backward, and ~5× more memory than DMTet at 64³. But in the context of a full pipeline (Table 8, nvdiffrecmc at 96³), the per-iteration cost goes from 307ms (DMTet) to 315ms (FlexiCubes) — **a 2.6% overhead** for the full reconstruction. And the memory goes from 13.1 GiB to 15.3 GiB — manageable. At 128³, FlexiCubes uses **816 MB** just for the extractor; this is a real concern for full-arch reconstructions at high resolution and may force us to stick to 64³–96³ for the v0 prototype. **For us: budget 64³ for the v0 prototype** (sufficient resolution to capture cusps and fissures on a single tooth at ~150 µm spacing), and only go to 128³ for the v1 production model.

## Connections to our hypotheses (H1–H5)

### H1 — 2-stage (segmentation + generation) > end-to-end

**STRONG support.** FlexiCubes is the cleanest possible expression of "the modern 2-stage 3D generation pipeline" — the first stage is a learned implicit-field generator (GAN, diffusion, VAE — paper-agnostic), and the second stage is FlexiCubes. The paper shows this 2-stage design is the *only* one that can deliver both sharp features (FlexiCubes' job) and end-to-end trainability (the generator's job). The taxonomy in Table 1 — explicitly contrasting "Grad" (differentiable optimization) vs "Flexible" (sharp features) — is a *defense* of the 2-stage paradigm.

### H2 — Diffusion on point clouds > mesh-based VAE for surface generation

**MILD contradiction / reframing.** The paper shows that mesh-based representations (specifically, implicit-field + FlexiCubes extraction) work *very* well in GET3D — a GAN-based generative model. The 4–5 point FID improvement from DMTet to FlexiCubes is in a non-diffusion model. **This does not contradict H2 directly** (the paper doesn't compare against point-cloud diffusion), but it does *narrow* the gap that H2 was claiming. The cleaner restatement: **H2 should be "diffusion on implicit fields > mesh-based VAE"** rather than "diffusion on point clouds > mesh-based VAE". FlexiCubes makes the implicit-field route viable for the *mesh output* step, and Diffusion-SDF (paper 004) handles the *generation* step — together they cover H2 better than point-cloud diffusion would.

### H3 — Conditioning on opposing + adjacent teeth improves outer surface quality

**INDIRECT support.** The paper doesn't directly test cross-tooth conditioning, but the GET3D + FlexiCubes pipeline shows that *learned* surface extraction is the right way to get the conditional generator's output into a high-quality mesh. **The FlexiCubes module sits downstream of H3** — the H3 conditioning lives in the implicit field (LION's `z0` for context, Diffusion-SDF's cross-attention, or a 3DTeethSeg22-conditioned VAE), and FlexiCubes extracts the mesh from whatever field the H3 generator produces. **For us: FlexiCubes is compatible with any H3 implementation** and is *especially* good if H3 is in the implicit-field space (which paper 003 DiGS and paper 004 Diffusion-SDF support).

### H4 — Implicit SDF > explicit mesh for high-quality surfaces

**STRONG qualifier / refinement.** H4 says implicit > explicit. FlexiCubes says: **the explicit mesh *is* the implicit field's rendering, and the quality of the rendering matters as much as the field's expressivity.** The paper's Table 2 shows that the *same* implicit field, extracted with MC, DMTet, or FlexiCubes, produces dramatically different mesh quality (IN>5° 66% vs 49% vs 35% at 32³). The field is necessary but not sufficient. **The re-statement: H4 should be "implicit field is the right *substrate*, but the *mesh extractor* is what determines the final quality"** — and FlexiCubes is the right mesh extractor. **For us: keep DiGS / Diffusion-SDF as the field, replace NDC with FlexiCubes as the extractor.** This is the cleanest possible H4 + FlexiCubes synthesis.

### H5 — Synthetic CAD can bootstrap training (data is currently zero)

**WEAK support.** FlexiCubes is trained on synthetic CAD (ShapeNet for GET3D, Myles/ShapeNet for mesh reconstruction) and the photogrammetry applications use it zero-shot or with light fine-tuning on real data. **This validates that synthetic CAD can serve as a strong pretraining base for clinical / real-data fine-tuning** — exactly our H5 plan. The paper does not directly test on dental data, but the precedent is clear: FlexiCubes is a *drop-in* for any 3D pipeline, and the synthetic → real fine-tuning pattern is well-established in the 3D generative-modeling literature. **For us: when we get our 50–200 patient IOS scans, a *light fine-tune* of FlexiCubes (1 day on A100) should be enough to adapt the synthetic-CAD prior to the clinical distribution.**

## Surprises / interesting things buried in section 4–7

- **Section 4.2:** the `α` interpolation weights are *not* the same as the `β` vertex-position weights — they are **two separate sets of parameters** for two separate sub-decisions (where on the edge the crossing is, vs where in the primal face the dual vertex sits). The ablation in Table 3 shows that adding *just* `β` drops IN>5° from 53 to 41, and then adding `δ` + `γ` takes it to 35. **For us: this decomposition is a useful pattern** — when we have a problem with a FlexiCubes-trained mesh, we can debug *which* of the 21 parameters is miscalibrated.
- **Section 4.6, Fig. 12:** an *unconstrained* octree-based FlexiCubes produces 318 non-manifold vertices on one test shape; adding the sign-consistency constraint reduces this to **1**. This is a useful gotcha for the v1 production model — if we want to use adaptive resolution, we *must* apply the sign-consistency constraint. The paper notes it as "noteworthy" that this works without an explicit graph-correction step.
- **Section 5.2, Table 4:** DMTet *with* the equilateral-triangle regularizer has a CD of 6.69, *worse* than MC's 6.33 baseline. **Adding the regularizer actively hurts DMTet's geometric accuracy.** FlexiCubes survives at 5.46 (vs its 4.87 baseline, an 12% degradation, while triangle quality improves 10×). **This is the most important result for our clinical pipeline** — we can add a margin-line-preservation regularizer or a developability regularizer without losing occlusal anatomy.
- **Section 6.1, Fig. 21:** the cleaner FlexiCubes triangulation makes *UV unwrapping* (texture coordinate generation) easier. **This is a non-obvious benefit for the photorealistic rendering** of crown previews — the dentist can see a more realistic visualization of the proposed crown in the patient's mouth, which is a major UX win.
- **Section 6.2 (animated meshing):** the paper's experiment is to optimize a mesh over an *animation sequence* (not a single T-pose) — and the end-to-end-optimized mesh has *1.7k triangles* vs the reference's 23k. **This is direct evidence that end-to-end optimization over a downstream task produces much sparser meshes** — a 13× reduction in triangle count with comparable quality. **For us: this means we don't need to extract high-resolution meshes from the diffusion model at inference time** — the FlexiCubes + downstream-task optimization will give us sparse, well-shaped triangles, which is exactly what 3D printing wants.
- **Section 7.2, "Limitations":** the paper explicitly states that **FlexiCubes is not even globally continuous** — when the isosurface slips over a grid vertex, the mesh jumps discontinuously. This is inherited from DC and DMC, and is the *real* reason that NDC (paper 006) cannot be used for optimization. **For us: we accept this in practice** — stochastic optimization (Adam) doesn't care about global continuity, only local differentiability. But it does mean that *visual inspection* of intermediate optimization steps can be jarring.
- **Section 7.2, "self-intersections":** "we found that strictly constraining the motions to non-intersecting configurations unacceptably worsened the expressivity." **This is an honest admission** that the paper chose flexibility over intersection-free guarantees. For 3D printing, this means we need a *post-processing pass* (manifold repair, self-intersection removal via trimesh or PyMeshFix) before slicing. **This is a real engineering cost** but well-understood and fast (< 1s per mesh).
- **Table 7 vs Table 8:** the per-iteration cost of FlexiCubes is small compared to the downstream task (8.93ms is 2.8% of the 315ms nvdiffrecmc iteration). **The paper's point: don't let the 5× slow-down in the mesh extractor fool you** — the bottleneck is the renderer, not the extractor. For our pipeline, the bottleneck will be the diffusion sampling, not the FlexiCubes extraction.
- **Section 6.4 (differentiable physics simulation):** FlexiCubes extracts a *tetrahedral* mesh (Sec. 4.5) and feeds it to a differentiable FEM simulator. **This is a future direction for us** — if we want to simulate the biomechanics of a crown under occlusal load (e.g., validate that the contact stress is below the fracture threshold of the chosen ceramic), the FlexiCubes tet extraction is the right input to a FEM solver like FEBio. This is *not* in our v0 scope but is worth a footnote for the v2 design.

## Quote-worthy sentences

> *"These two properties [Grad and Flexible] are inherently in conflict. Increased flexibility provides more capacity to represent degenerate geometry and self-intersections, which hinder convergence in gradient-based optimization."* — §1 (the problem statement in one sentence)

> *"Our method is actually not even globally continuous. When the isosurface slips over a grid vertex, the mesh jumps discontinuously, a property we inherit from Dual Contouring and Dual Marching Cubes. Fortunately, because we apply our extraction in stochastic optimization settings, such as stochastic gradient descent with Adam, small local discontinuities do not obstruct optimization in practice."* — §7.2 (the most honest limitation in the paper, and the reason it works anyway)

> *"FlexiCubes is now integrated into NVIDIA applications as a drop-in replacement for DMTet."* — GitHub README (the production-deployment story)

> *"Methods that extract the mesh as a post-processing step fail to achieve competitive performance in terms of reconstruction quality, highlighting the importance of end-to-end optimization that mitigates the discretization errors introduced in post-processing."* — §5.1 (the empirical argument for the *whole* 2-stage paradigm)

> *"Optimizing for the appearance over the entire animation helps re-distribute triangle density to avoid mesh stretching."* — §6.2 (the case for end-to-end optimization over a downstream task, applicable to our "crown under occlusal load" scenario)

## Code / data availability

- **Code:** https://github.com/nv-tlabs/FlexiCubes — Apache 2.0, PyTorch + C++/CUDA, actively maintained, integrated into Kaolin (NVIDIA's 3D deep learning library) v0.15.0+. The repo includes `flexicubes.py` (the original paper implementation) and a tutorial notebook `examples/optimization.ipynb` that walks through single-shape reconstruction with multiview mask + depth losses. The repo also has `examples/extraction.ipynb` for the simpler "use FlexiCubes as a non-trainable isosurface extractor on a known SDF" case.
- **Datasets:**
  - Reconstruction: Myles et al. (2014) shapes, downloaded via `examples/download_data.py`. One shape is included in `examples/data/inputmodels/block.obj` for quick testing.
  - Generative: ShapeNet (Chang et al. 2015).
  - Photogrammetry: NeRF synthetic (Mildenhall et al. 2020) and Tanks & Temples (Knapitsch et al. 2017).
  - Physics: custom (rendered with the pipeline in the supplement).
- **License:** Apache 2.0, NVIDIA copyright. Free for commercial and research use, no patent grant. The repo uses DCO 1.1 for contributions.
- **Third-party dependencies:** PyTorch 1.12, nvdiffrast (NVIDIA's differentiable rasterizer), Kaolin 0.15.0+. All available via pip/conda.

## For our project

FlexiCubes is the *v1 mesh extractor* for our crown-generation pipeline. It replaces NDC (paper 006) in the architecture and is a drop-in upgrade to Diffusion-SDF → MC or Diffusion-SDF → NDC. Concrete next steps, ordered by priority:

1. **Adopt FlexiCubes as the default mesh-extraction module, replacing NDC.** The 4–5 point FID improvement in GET3D (Table 6) and the 10× CD improvement in nvdiffrec (Table 5) are the strongest empirical arguments. **This is a single-file change** — both Diffusion-SDF and LION output an implicit field or point cloud, and FlexiCubes takes the field as input. *The 64³ resolution is the right default for the v0 prototype* (sufficient for cusps and fissures on a single tooth, fits in 16 GB of GPU memory with margin).

2. **Add a clinical-quality regularizer to the diffusion model + FlexiCubes pipeline.** Table 4 is the killer table — FlexiCubes is the *only* mesh extractor that can take a regularizer without geometric collapse. Concrete candidates for regularizers:
   - **Margin preservation:** penalize the volume of the intaglio face that falls below the prep margin. This is the most important clinical constraint.
   - **Occlusal convexity:** penalize concavities on the occlusal surface (the patient's bite should not lock into concavities).
   - **Developability (Stein et al. 2018):** penalize the smallest eigenvalue of the covariance of face normals — this is a "fabricability from a flat sheet" prior that may help the milling/printing step. The paper's own Fig. 17 demonstrates this regularizer.
   - **Equilateral triangle regularizer (Sec. 5.2):** produces better FEM meshes for downstream physics simulation, if we want to simulate occlusal load.

3. **Use the equilateral-triangle regularizer to make the v0 mesh "FEM-ready" for occlusal-load simulation.** This is a v2 feature but a clear architectural direction: FlexiCubes + tet extraction (§4.5) + differentiable physics (§6.4) is the full pipeline for biomechanical validation. *Defer to v2, but the architectural path is set.*

4. **Run FlexiCubes on the 3DTeethSeg22 dataset as a v0 sanity check.** Same plan as the NDC paper note (paper 006) — take a handful of labeled teeth, convert to SDF grids, run FlexiCubes, inspect outputs. **The expected improvement over NDC is sharper cusps, cleaner fissures, and fewer sliver triangles** — which is exactly what we want for clinical-quality output. This is a 1-day experiment.

5. **Pilot FlexiCubes vs NDC vs Marching Cubes on the same Diffusion-SDF outputs.** The LION-vs-Diffusion-SDF pilot (paper 005's action item #1) is the bigger decision; this pilot is a *mesh-extractor* decision on top of it. Concretely: take the outputs of a trained Diffusion-SDF model, extract meshes with MC, NDC, and FlexiCubes, and compare on (a) CD to ground truth, (b) sharpness of occlusal cusps, (c) smoothness of the intaglio surface, (d) % of bad edges / non-manifold junctions. **If FlexiCubes wins even modestly (CD within 10% of the best), we adopt it** — the inference cost is comparable (8.93ms vs 0.5s) and the code is well-validated. This is a 1-week, ~$300 experiment.

6. **Adopt FlexiCubes's self-intersection caveat explicitly in the design doc.** The paper honestly admits that self-intersections can occur and that strictly constraining to non-intersecting configurations "unacceptably worsens expressivity" (§7.2). **For us: add a trimesh / PyMeshFix post-processing pass** (manifold repair, self-intersection removal) before sending the mesh to the 3D printer slicer. This is < 1s per mesh and is a well-understood step. **Cite §7.2 directly in the design doc.**

7. **Adopt the 21-parameter-per-cell pattern for the diffusion model's output layer.** When the LION/Diffusion-SDF decision is made, the *generator's last layer* needs to output the SDF value *plus* the 21 FlexiCubes parameters per cell. **This is a clean architectural pattern** — the generator learns to predict not just the field but also the optimal mesh-extraction parameters. The GET3D + FlexiCubes integration is the template (they modify only the last layer to output 21 weights per cube). Cite the GitHub README's "GET3D: generative AI example" link.

8. **Adopt the 64³ resolution budget for v0, 96³ for v1.** Memory at 64³ is 116 MB for the extractor; at 128³ it's 816 MB. **For a single tooth at ~150 µm spacing, 64³ is sufficient** (a 10mm tooth fits in 64³ at 156 µm/voxel, and 96³ at 104 µm/voxel). Don't go to 128³ until we have a v1 architecture that needs it.

9. **Bookmark `examples/optimization.ipynb` and `examples/extraction.ipynb` for the v0 implementation.** The former is a single-shape optimization loop with multiview mask + depth losses — a clean template for our pipeline. The latter is the "use FlexiCubes as a vanilla isosurface extractor on a known SDF" case — useful for the v0 baseline before we add the diffusion model. **Cite both directly in the design doc.**

10. **Plan the v0 architecture explicitly around FlexiCubes's 4 parameter sets.** When the diffusion model is built, the generator's output layer should be designed to predict:
    - The SDF value per cell (the primary output).
    - `α ∈ R⁸` per cell (interpolation weights).
    - `β ∈ R¹²` per cell (dual-vertex weights).
    - `γ ∈ R` per cell (quad-split weight).
    - `δ ∈ R³` per grid vertex (grid deformation, if we use adaptive resolution later).
    This is 21 + 1 = 22 parameters per cell, plus 3 per grid vertex. The GET3D paper shows this is a constant overhead, not a network-size change.

11. **Open question for HK: do we want the 21-parameter FlexiCubes, or a simpler variant?** The 21-parameter version is the paper's full result, but the ablation (Table 3) shows that *just* `β` (dual-vertex weights) gets us most of the way (IN>5° 40.88 vs 34.87 for the full version). **A 13-parameter variant (`α + β` only, no `γ` or `δ`) is simpler to train and may be sufficient for the v0 prototype.** Worth a 1-day experiment to compare. *My recommendation: start with the full 21-parameter version, fall back to 13 only if convergence is unstable.*

12. **Re-budget paper 008.** The natural follow-ups are:
    - **SDC (Sundararaman et al. CVPR 2024)** — self-supervised NDC, *not* FlexiCubes. Lower priority now that FlexiCubes is the chosen extractor, but worth reading for the *end-to-end* fine-tuning story.
    - **SparseFlex (Li et al. CVPR 2025)** — high-resolution + arbitrary-topology 3D shape generation building on FlexiCubes. *Very* relevant for the v1 production model if we need high-resolution full-arch output.
    - **DMTet 2.0 (if NVIDIA publishes a follow-up)** — keep an eye on the GitHub for any 2024-2026 updates to the NVIDIA ecosystem.
    - **D-Faust / SMPL generative modeling** — the next step in *conditional* 3D generation (paper 003 DiGS' auto-decoder extension), if we want to add a *style* prior to the crown generation.

13. **Final architectural call for the v0 prototype.** The v0 prototype should be:
    - **Generator:** Diffusion-SDF (paper 004) for sub-tasks 3-4 (inner + outer surface), with H3 conditioning via cross-attention on adjacent + opposing teeth.
    - **Backbone for the implicit field:** DiGS (paper 003) as the field representation inside the SDF-VAE.
    - **Mesh extractor:** FlexiCubes (this paper) at 64³, with all 21 parameters predicted by the generator's last layer.
    - **Post-processing:** trimesh / PyMeshFix for self-intersection removal and manifold repair.
    - **Compute:** ~550 V100-hours for the SDF-VAE (per paper 004), ~1 day on A100 for the diffusion model, ~1 hour on A100 for the FlexiCubes pre-trained model fine-tune.
    - **Total budget:** ~$2,000 on Lambda for the v0 pilot. This is a significant step up from the LION/Diffusion-SDF pilot budget ($1,500) but the empirical case for FlexiCubes is strong enough to justify it.

---

*Scholar 🦉 — 2026-06-06*
