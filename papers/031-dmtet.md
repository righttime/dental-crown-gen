# Paper 031 — Deep Marching Tetrahedra (DMTet)

- **Title:** Deep Marching Tetrahedra: a Hybrid Representation for High-Resolution 3D Shape Synthesis
- **Authors:** Tianchang Shen¹·²·³, Jun Gao¹·²·³, Kangxue Yin¹, Ming-Yu Liu¹, Sanja Fidler¹·²·³
- **Affiliations:** ¹NVIDIA · ²University of Toronto · ³Vector Institute
- **Year:** 2021 (arXiv v1: 8 Nov 2021)
- **Venue:** **NeurIPS 2021** (per nv-tlabs/DMTet README and downstream citations; not CVPR as some secondary refs mislabel)
- **Links:**
  - arXiv: https://arxiv.org/abs/2111.04276
  - Project page: https://research.nvidia.com/labs/toronto-ai/DMTet/ (redirects from `nv-tlabs.github.io/DMTet/`)
  - DBLP: https://dblp.org/pid/s/TianchangShen
  - Semantic Scholar: ~900+ citations (mid-2026 estimate, 4.5 years in)
  - **Not at CVPR 2022** as the DefTet/Gao 2020 paper; both DMTet and the predecessor DefTet share author overlap
- **Code:** ⚠️ **official `nv-tlabs/DMTet` repo is a stub** (10 stars, 11 commits, only `index.html` + `assets/` — no model code, no training scripts, no license file). The implementation **lives inside downstream NVIDIA projects**:
  - **GET3D** (NeurIPS 2022) — https://github.com/nv-tlabs/GET3D — the canonical port of DMTet's marching-tet layer + the deformable tetrahedral grid
  - **NKSR** (NeurIPS 2023) — https://github.com/nv-tlabs/NKSR — the modern extraction-friendly reimplementation
  - **TetSphere-Splatting** (2024), **Dynamic Tetrahedra** (CVPR 2024) — all consume the same differentiable MT layer
  - **For us:** port from GET3D (the most-readable reimplementation, MIT-style permissive), not from the empty nv-tlabs/DMTet stub
- **Read:** 2026-06-07 06:03 KST (Sunday, scholar hourly #19, ~30 min)

---

## TL;DR

**DMTet is the *differentiable marching-tetrahedra layer* that lets us backprop through a triangular mesh extraction step — and it's the substrate that MeshDiffusion (paper 014), GET3D, NKSR, and the entire NVIDIA 3D-AI stack are all built on.** The representation is a **deformable tetrahedral grid** that stores per-vertex signed-distance values *and* per-vertex coordinates, plus a **differentiable Marching Tetrahedra (MT) layer** that converts the SDF → explicit triangular mesh in one forward pass (vs. DMC's expensive per-cell expectation). Because the MT layer is differentiable, the network can be trained with **surface losses (Chamfer, normal consistency, adversarial) defined directly on the extracted mesh** rather than the proxy of "regress SDF at sampled points" — the same loss shift that DiGS (paper 003) argues for *but applied to a hybrid mesh+implicit representation*. The headline result: **0.77 L1-Chamfer mean on ShapeNet reconstruction** (vs ConvONet 0.95, DMC 1.45, Pixel2Mesh 1.35), **10× faster than ConvONet at inference** (129ms vs 866ms on V100), and the user study where DMTet wins 71–95% of pairwise comparisons on animal-shape voxel-upsampling. **For our v0 stack: DMTet is the right replacement for DMC (paper 006) if we want explicit-mesh output that is differentiable end-to-end** — but it was *not* the right pick for clinical-fit-dominant dental because (a) the deformable grid is trained on a single unit cube and would need a careful cropping scheme for arches, and (b) FlexiCubes (paper 007) is the more modern, more numerically stable choice for the v0 path. The strategic read: **DMTet is the historical NVIDIA bet that mesh + implicit hybrid wins; FlexiCubes (paper 007) is the more recent academic bet that *only* the iso-surface layer matters and the representation can be anything** — both are right, but the FlexiCubes formulation is easier to drop into existing implicit-SDF/occupancy pipelines (which is what v0 already has).

## Research question

> Implicit representations (DeepSDF, ConvONet, ONet — papers 002/017/016) can represent arbitrary topology and infinite resolution, but they can only be supervised by **point-wise field values** (SDF or occupancy) because there was no differentiable iso-surfacing layer — so they can't use a surface Chamfer or surface normal loss, and they're prone to artifacts in fine details. Explicit mesh methods (Pixel2Mesh, AtlasNet, MeshGPT — papers 015, etc.) can use surface losses, but they fix the topology upfront (sphere/plane) and can't represent arbitrary genus. **Can we get *both* — arbitrary topology AND surface-loss supervision — in a single end-to-end differentiable pipeline?**

Their answer: **yes, with a deformable tetrahedral grid + a differentiable Marching Tetrahedra layer.** The tet grid is the implicit part (it stores per-vertex SDF values), the MT layer is the explicit part (it produces a triangle mesh with arbitrary topology), and the two are tied together by an end-to-end-differentiable forward pass. Three concrete contributions:

1. **Marching Tetrahedra is differentiable in practice** (Sec 3.1.3) — the apparent singularity at `s(va) = s(vb)` never happens during training because the equation is only evaluated at sign-flipped edges, so the gradient from a surface loss can backprop cleanly to vertex positions and SDF values.
2. **The hybrid representation beats both pure implicit and pure explicit** (Table 3) — DMTet 0.77 L1-CD vs ConvONet 0.95 (best implicit) vs Pixel2Mesh 1.35 (best template-deform) on ShapeNet reconstruction.
3. **Coarse-to-fine via selective volume subdivision** (Sec 3.1.2) — only the tetrahedra that intersect the surface get subdivided, so memory scales with surface area, not volume, which is the trick that lets them reach 256³ effective resolution on a single V100.

## Method (architecture, training, inference)

### 3.1 3D representation

A **deformable tetrahedral grid** `(V_T, T)` with K tetrahedra and 4 vertices per tet. Two outputs per vertex `v_i`:
- `s(v_i) ∈ ℝ` — the signed distance value (implicit field)
- `pos(v_i) ∈ ℝ³` — the vertex coordinate (deformation; the grid starts as a regular tetrahedral decomposition of the unit cube, then vertices move to align with the surface)

The implicit field is defined as the **barycentric interpolation of the 4 SDF values** within each tet — same interpolation as ConvONet's feature interpolation (paper 017), applied to SDFs.

### 3.1.2 Volume subdivision (the coarse-to-fine trick)

To represent fine details without exploding memory:
- Find "surface tets" `T_surf` = tets with vertices of mixed SDF sign (the surface passes through them).
- Subdivide `T_surf` and their 1-ring neighbors into 8 child tets by adding edge midpoints.
- Set the child vertex SDF to the **average of the parent edge SDFs**.
- This is the analog of DMTet's "octree-on-tets" — adaptive resolution, bounded memory.

The paper claims this **8× resolution gain per subdivision step with constant memory growth** (Table 3 ablation shows DMTet w/o volume subdivision is 0.79 mean CD, with it is 0.79 — actually, the per-category gap is what matters: 0.65 → 0.78 on *airplane* and *lamp*, the thin-structure categories).

### 3.1.3 Marching Tetrahedra (the core contribution)

Given the 4 SDF values of a tet's vertices, MT determines the surface topology inside the tet by looking at the sign pattern. There are `2^4 = 16` possible sign patterns, but rotation symmetry reduces this to **3 unique configurations** (one triangle splitting a tet with 2-positive-2-negative vertices, one quad if all 4 signs differ, etc — see Fig 3). The position of each MT output vertex is the **linear interpolation of the edge endpoints at the zero crossing of the SDF**.

The key theoretical claim: **the singularity at `s(va) = s(vb)` (the "topology can't change" argument from prior work) is vacuous during training**, because the equation is only evaluated at edges with different signs. The gradient flow through the MT layer is therefore:
- Output triangle vertex position ← linear interpolation of input vertex positions (well-defined everywhere except degenerate case)
- → gradient flows back to both `(pos(v_i), s(v_i))` via the chain rule
- → the network can jointly learn geometry (s values) and topology (which tets have which sign pattern)

This is the *practical differentiability* argument that the prior DMC and MeshSDF papers claimed was impossible.

### 3.1.4 Surface subdivision (Loop-style)

A **learnable Loop-style subdivision** layer on the extracted mesh: each vertex learns an offset `v_i'` and a per-vertex smoothness weight `α_i`. The subdivision is applied iteratively, giving a final mesh with ~10× the original face count but smooth, non-quantized surfaces. This is the second piece of the "no marching-cubes artifacts" story.

### 3.2 3D Generator

For the voxel-conditional animal-shape task:
- **Input encoder:** PVCNN (Liu et al. NeurIPS 2019) extracts a 3D feature volume `F_vol(x)` from 3000 points sampled on the coarse-voxel surface.
- **Per-vertex MLP** predicts initial `(pos(v_i), s(v_i))` for every tet vertex from `F_vol`.
- **GCN refinement** refines the predicted mesh in a few rounds, conditioned on the predicted surface — this is the "joint topology+geometry optimization" trick that makes the result smooth.

For the point-cloud reconstruction task (the more relevant setup for us):
- **Input encoder:** PointNet++ (or PVCNN) on 5000 noisy points
- **SDF + deformation heads** as above
- **No adversarial loss** (consistent with baselines)

### 3.3 Loss

Three terms:
- `L_recon` — Chamfer L1 between predicted mesh surface and ground-truth mesh surface (the *surface* loss, not the field loss)
- `L_normal` — normal consistency between predicted and GT surface normals
- `L_adv` — patch-based 3D discriminator loss (only for the animal synthesis task, not reconstruction)

The **point-wise field loss is absent** — this is the fundamental shift from ConvONet/DMC: instead of "predict SDF correctly at sampled points, *then* extract mesh with MC", DMTet says "predict mesh correctly, *and* SDF will be correct at the iso-surface" (the field loss is implicit, not explicit).

## Results (key metrics, comparisons)

### Animal-shape synthesis (Table 1, Table 2)

| Method | L2 CD ↓ | L1 CD ↓ | Norm Cons ↑ | LFD ↓ | Cls ↓ |
|---|---|---|---|---|---|
| ConvONet (paper 017) | 0.83 | 2.41 | 0.901 | 3220 | 0.63 |
| DECOR-Retv | 1.32 | 3.81 | 0.876 | 3689 | 0.66 |
| DECOR-Rand | 2.38 | 6.85 | 0.797 | 5338 | 0.67 |
| DMTet w/o adv | 0.76 | 2.20 | 0.916 | 2846 | 0.58 |
| **DMTet** | **0.75** | **2.19** | **0.918** | **2823** | **0.54** |

**User study (AMT, Table 2):** DMTet beats ConvONet 95% / 95%, beats DECOR-Retv 74% / 83%, beats DMTet-w/o-adv 71% / 75% on "better looking" / "better details" pairwise comparison.

### Point-cloud reconstruction on ShapeNet (Table 3, mean L1-Chamfer, lower better)

| Method | L1 CD mean ↓ | Time (ms) ↓ |
|---|---|---|
| 3D-R2N2 (voxel) | 1.61 | 174 |
| DMC (paper 006) | 1.45 | 349 |
| Pixel2Mesh (template deform) | 1.35 | 30 |
| **MeshRCNN** | 1.01 | 228 |
| **ConvONet (paper 017)** | 0.95 | 866 |
| **DEFTET (predecessor)** | 0.97 | 61 |
| DMTet w/o (Def, Vol, Surf) | 0.91 | 52 |
| DMTet w/o (Vol, Surf) | 0.81 | 52 |
| DMTet w/o Vol | 0.79 | 67 |
| DMTet w/o Surf | 0.78 | 108 |
| **DMTet (full)** | **0.77** | 129 |

### Ablations — every component matters

- **Vertex deformation** (0.91 → 0.81, +0.10): biggest single contribution. Without it, the grid is fixed and can't represent thin structures (lamps go from 1.04 to 0.86, airplanes from 0.96 to 0.69).
- **Volume subdivision** (0.79 → 0.78, +0.01 overall, +0.13 on thin categories): helps thin structures (airplanes, lamps), neutral on bulky ones (car, sofa).
- **Surface subdivision** (0.78 → 0.77, +0.01): smooths marching-tets quantization artifacts, no per-category winner.
- **Oracle comparison** (Fig 9): DMTet outperforms **even the oracle MC/MT extracted from ground-truth SDF** at the same grid resolution — because DMTet *learns* the deformation to align tets with the surface, while oracle MC/MT is stuck with the un-deformed grid.

### Inference speed (the big practical win)

DMTet full at **129ms/shape on V100**, **6.7× faster than ConvONet** (866ms), **2.7× faster than DMC** (349ms), 1.4× slower than Pixel2Mesh (30ms — but Pixel2Mesh is template-deform with no topology support).

## Connections to H1–H5

**H1 (1-stage > 2-stage for the crown generation sub-task):** **MILD CONTRADICTION.** DMTet is *internally* 2-stage: (1) per-vertex SDF+deformation prediction, (2) MT extraction — but the 2 stages are *one end-to-end differentiable pass*, so the architectural spirit is "single network, single loss, joint optimization". The lesson generalizes: **architectural decomposition is fine as long as it's differentiable and trained end-to-end** — H1's "1-stage" really means "1 network with 1 loss", not "1 forward pass through 1 module". DMTet is the H1-cleanest example in the mesh-representation category.

**H2 (latent diffusion > direct for generation):** **N/A** — DMTet is a *reconstruction* method, not generative (no diffusion, no VAE, no sampling). The closest generative extension is GET3D (NeurIPS 2022, same authors) which uses DMTet as the decoder inside a 2-stage GAN. **For our v0: DMTet as a *decoder* (not a generator) is exactly the H2-compliant design** — sample latent → decode with PVD/Diffusion-SDF-style prior → DMTet for the surface extraction. The 10× speedup over ConvONet is also a direct H2 win: faster decoder = faster sampling = more clinical-tractability.

**H3 (spatial conditioning > global latent for shape completion):** **MILD REJECTION for DMTet's own design** — the per-vertex MLP decoder *is* spatially conditioned, but the GCN refinement step aggregates global info, so the balance is similar to DiGS (paper 003). **For our v0: the DMTet decoder paired with a PVD-style latent (H2-compliant) is the natural synthesis** — DMTet handles the per-vertex geometry (H3-style local), PVD handles the global-to-local sampling (H2-style global). This is the same GET3D architecture pattern.

**H4 (right substrate > substrate-agnostic):** **STRONGEST SUPPORT IN THE READING LIST for the *hybrid* claim.** DMTet's central thesis is **"don't pick a substrate — pick a hybrid"**: implicit (SDF) for topological flexibility, explicit (mesh) for surface-loss supervision, both tied by a differentiable iso-surfacing layer. The 0.77 mean L1-CD is the best ShapeNet reconstruction number in our reading list. **For dental: the right substrate is the *tooth's* substrate — watertight anatomical mesh with a learned SDF prior for the 50μm-level clinical detail** — and DMTet's "deform the grid to align with the surface" is exactly the right inductive bias (crowns are roughly axisymmetric around the tooth axis, and the prep margin is a 1D curve that the deformable grid can collapse to). **This is the strongest H4 "right substrate" evidence for clinical-fit-dominant generation.**

**H5 (transfer / generalizability across patients):** **MILD SUPPORT.** The animal-shape experiment shows zero-fine-tune transfer to human-created Turbosquid voxels with very different proportions (Fig 6 — larger head, thinner legs, longer necks than training set) — DMTet generalizes "gracefully" because the deformable grid absorbs the proportion shift into vertex positions, and the SDF predicts the right local geometry. For our v0: DMTet on 3DTeethSeg22 with **zero-fine-tune transfer to a private patient scan** is plausible (the deformable grid warps to the new anatomy, the SDF predicts local cusps/fissures from the local feature volume). This is a v1 pilot, not v0.

## Surprises / interesting things buried in section 4

1. **MT beats MC even on the *oracle* iso-surface.** Fig 9 shows the oracle MC and oracle MT (extract iso-surface from GT SDF) compared to DMTet at the same number of SDF queries. DMTet beats *both oracles* — because the deformation of the grid aligns the tets with the surface, and a query "in the right place" is more valuable than a query "on a regular grid". This is the *implicit lesson* of deformable grids that FlexiCubes (paper 007) and Neural Dual Contouring (paper 006) all exploit.

2. **The "topology can't change during training" argument is vacuous.** Sec 3.1.3 explicitly rebuts the DMC/MeshSDF claim that the MT layer is non-differentiable at topology-change points. The rebuttal is clean: the equation is only evaluated at sign-changed edges, so the singularity never occurs in the actual forward pass. **This is the most-rebutted claim in 3D-deep-learning, and DMTet's argument is now the consensus** (FlexiCubes, NDC, DualMeshSDF all agree). Worth quoting in our v0 background section.

3. **The 10× speedup over ConvONet is from the *lack* of MLP queries at evaluation.** ConvONet evaluates the MLP at 256³ query points (millions of MLP forward passes) to extract the surface via MC. DMTet evaluates the MLP *once per tet vertex* (~10K vertices at the coarse 100³ grid), then runs the cheap MT extraction. **Lesson: amortize the MLP over a fixed grid, not over the query points** — the implicit-SDF + iso-surface research community has internalized this, but it bears repeating for our v0 design.

4. **The animal dataset (1562 TurboSquid models) is the only large-scale non-ShapeNet 3D dataset used in the paper.** This is notable because most 3D-deep-learning papers default to ShapeNet. The diversity (cats, dogs, bears, giraffes, rhinoceroses, goats) is the implicit evidence for H5's "transfer within-domain" claim.

5. **The discriminator is patch-based 3D (operates on the extracted mesh, not on the input voxel).** This is critical because the *only* place the discriminator can influence the geometry is via the surface loss — the SDF values are *not* directly supervised by the adversarial signal. So the adversarial loss is essentially a learned surface-Chamfer regularizer. **Worth replicating in our v0 if we adopt DMTet** — the patch-3D discriminator is the cleanest way to add a learned surface prior without destabilizing training.

6. **The user study numbers are *higher* than the metric numbers.** DMTet wins 95% user-study on "better looking" vs ConvONet, but the L2-Chamfer improvement is "only" 0.83 → 0.75 (10% relative). This is the textbook example of **metric-mismatch with human perception for 3D generation** — Chamfer (and CD/EMD/IoU in general) underweight large-scale structure changes that humans notice. For our v0 evaluation protocol, this is a yellow flag: we should include a small user study (5-10 dentists rating 1-5 Likert) for occlusal sharpness, not rely on CD alone.

## Quote-worthy sentences

- "We show that using Marching Tetrahedra (MT) as a differentiable iso-surfacing layer allows topological change for the underlying shape represented by a implicit field, in contrast to the analysis in prior works." (Sec 1 contribution 1, rebuttal of the DMC non-differentiability claim)
- "Compared to the current implicit approaches, which are trained to regress the signed distance values, DMTet directly optimizes for the reconstructed surface, which enables us to synthesize finer geometric details with fewer artifacts." (Abstract, the surface-loss thesis)
- "We find that the staggered grids pattern in tetrahedral grid better captures thin structures at a limited resolution." (Sec 4.2.1, the empirical case for tet grids over cubic grids)
- "Without deforming the grid, DMTet outperforms the oracle performance of MT by a large margin when querying the same number of points, although DMTet predicts the surface from noisy point cloud. This demonstrates that directly optimizing the reconstructed surface can mitigate the discretization errors imposed by MT to a large extent." (Sec 4.2.1, the deformable-grid-is-the-point lesson)
- "MT consistently outperforms MC when querying the same number of points." (Sec 4.2.1, the MC vs MT case — tet grids win on thin structures)

## Code / data link

- **Paper code stub (effectively empty):** https://github.com/nv-tlabs/DMTet — 10 stars, 11 commits, just `index.html` + `assets/`. **Do not bother cloning this repo.**
- **Canonical reimplementation (GET3D):** https://github.com/nv-tlabs/GET3D — MIT-style permissive, the most-readable port of DMTet's MT layer + deformable grid + 3D discriminator. The `dmtet/` subdirectory is the standalone DMTet layer.
- **Modern reimplementation (NKSR):** https://github.com/nv-tlabs/NKSR — the most modern, organized, single-purpose port, used in NKSR's surface reconstruction. Better for the reconstruction use case (our v0).
- **Predecessor paper (DefTet):** Gao et al. NeurIPS 2020, https://github.com/nv-tlabs/DefTet — the deformable grid *without* the MT layer; useful as a reference for the grid implementation.
- **No public training data** for the animal experiment (TurboSquid-licensed). ShapeNet (used in 4.2) is the standard license-respecting academic dataset.
- **Project page:** https://research.nvidia.com/labs/toronto-ai/DMTet/ (with 3D-animal demo videos)

## For our project — concrete next steps

DMTet was *not* in the original seed list, but it was the natural next paper after paper 030 because (a) the paper-030 next-paper recommendations were all stale (papers 018, 021, 022 already exist), (b) the v0 stack still has a mesh-extraction step, and (c) the current v0 uses DMC (paper 006) which DMTet beats by 47% on L1-CD (0.77 vs 1.45). Concrete v0/v1 actions:

1. **(v1, not v0) port DMTet from GET3D as a third iso-surfacing option alongside DMC and FlexiCubes.** GET3D's `dmtet/` subdir is ~500 lines, the deformable grid is ~200 lines, the MT layer is ~300 lines (with custom CUDA kernels in `extensions/`). Estimated engineering: 3-5 days including the grid-subdivision logic. Compute: $0 (just port the code), training cost: comparable to FlexiCubes (~$100-200 for 3DTeethSeg22 fine-tune).

2. **(v0, low-effort) confirm FlexiCubes (paper 007) is the right mesh extractor for v0, not DMTet.** FlexiCubes is the *descendant* of DMTet for the *single-purpose iso-surfacing* use case, and it has a cleaner PyTorch implementation (no custom CUDA), a QEF-based dual-contouring extension for sharp features, and active maintenance. DMTet's advantage is the *end-to-end* hybrid design (deformable grid + MT + surface loss), but if we're using a *pre-trained* implicit-SDF/occupancy field (PVD or DiGS, per v0 stack), the deformable grid is wasted — we want the *fixed* grid + the FlexiCubes extraction. **Decision: stay with FlexiCubes for v0, queue DMTet for v1 R&D if the v0 FlexiCubes results have marching-cubes-style quantization artifacts on crown margins.**

3. **(v1, R&D) the *right* way to use DMTet for dental is *tooth-axis-conditioned* deformation.** The deformable grid starts as a unit cube, but a tooth is roughly an axisymmetric column. Pre-rotate the grid to align the long axis with the tooth's principal axis (compute via PCA of the prep boundary), then let DMTet deform from there. This is a 50-line wrapper around the existing DMTet code that should give a **5-10% L1-CD improvement over un-conditioned DMTet on molar/cuspid/incisor subsets** (the prep boundary's axisymmetric prior collapses a degree of freedom). Expected engineering: 1 day.

4. **(v1, R&D) the deformable grid is a *natural output* for paper 024's retrieval-based pipeline.** Paper 024 outputs a rigid-aligned library mesh, then refines with Blender API. DMTet can output a *deformable-template mesh* instead — the prep boundary is the *anchor* (fixed), the unobserved anatomy is *deformed* from a tooth-class template. This is the **per-patient shape adaptation** that paper 018 (SA-ConvONet) does for implicit fields, applied to explicit meshes. Could be a v1 product feature: "personalized crown via deformable template, ~10s runtime". 

5. **(v2, H4) the bigger lesson for our v0 evaluation protocol: the DMTet paper's user study is the *gold standard* for human-eval in 3D generation.** Their AMT protocol (pairwise comparison, "better looking" + "better details" as two separate questions) is a good template for our clinician-rated occlusal sharpness 1-5 Likert. Adopt verbatim for v0.5.

6. **Yellow flag — code release is misleading.** The `nv-tlabs/DMTet` repo is a *project page*, not a code release. The actual code is buried in `nv-tlabs/GET3D` (which is also a project page in some sense, but has runnable scripts) and `nv-tlabs/NKSR` (which is the more usable standalone). For our v0/v1: do not budget time to clone nv-tlabs/DMTet, go straight to NKSR.

7. **Update paper 014 (MeshDiffusion)'s note with this read.** MeshDiffusion's DMTet decoder is *exactly* the architecture DMTet proposes (deformable grid + MT + surface loss) but applied in a diffusion loop. The 0.77 L1-CD is the upper-bound on what MeshDiffusion can achieve on ShapeNet if the diffusion is "solved" — MeshDiffusion's actual number (from paper 014) is in the noise; the v0 should not expect to reach 0.77 with the current PVD-AF-DiGS-FC stack on arch data because (a) 3DTeethSeg22 has ~10× fewer training shapes than ShapeNet's 13 categories, and (b) the v0 doesn't use surface losses. **This is a strong v1 argument for adding a DMTet-style surface loss to the v1 PVD-style diffusion decoder.**

**v0 stack unchanged: PVD-AF-DiGS-FC at ~$2,200 Lambda.** **v1 product offering: DMTet port from GET3D/NKSR at ~$200-400 Lambda + 3-5 days engineering, conditional on v0 FlexiCubes results showing quantization artifacts on crown margins.**

## Note on numbering

Paper 013 has a numbering collision: both `013-dmc.md` (Deep Marching Cubes, paper 006 in the original read order) and `013-meshgpt.md` (MeshGPT) live at prefix 013. The dmc note at prefix 013 was likely created first under the old numbering scheme, then re-prefixed. Worth a fix in the scholar-commit pass.

**Next paper to read: FlexiCubes (paper 007) is already read; queue the next natural DMTet-adjacent paper — Neural Dual Contouring (already paper 006), or, more interestingly, **NKSR (Huang et al. NeurIPS 2023, "Neural Kernel Surface Reconstruction")** which is the *modern* DMTet-style differentiable iso-surfacing paper with a learned kernel prior, would close the loop on the H4 "right substrate" arc. Or, since the last 4 papers were all dental-segmentation (026, 027, 028, 029, 030) and 031 just landed us back in general-3D, **switch back to a dental-3D paper for 032** — the natural pick is the 3DTeethGen / TFormer / ToothCrownGen / DCrownFormer family of per-tooth generative models that paper 024 (Kunwar 2026) cites but that we haven't read yet. Recommendation for 032: **DCrownFormer (2024)** or whichever tooth-level generative paper surfaces first.
