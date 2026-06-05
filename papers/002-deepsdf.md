# 002 — DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation

- **Title:** DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation
- **Authors:** Jeong Joon Park* (UW), Peter Florence* (MIT), Julian Straub (FRL), Richard Newcombe (FRL), Steven Lovegrove (FRL)  *equal contribution
- **Year:** 2019 (CVPR 2019; arXiv:1901.05103 v1, 16 Jan 2019)
- **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2019
- **Links:**
  - Paper: https://arxiv.org/abs/1901.05103
  - Official code: https://github.com/facebookresearch/DeepSDF (MIT)
- **Code/data:** PyTorch implementation released by the authors; experiments on ShapeNet v1/v2 (chairs, planes, sofas, tables, lamps, cars, etc.)

---

## TL;DR

DeepSDF represents a **class** of 3D shapes (not just one) as a continuous signed-distance function implemented by an 8-layer fully-connected net, conditioned on a learned per-shape latent code via an **auto-decoder** (no encoder — codes and decoder weights are jointly optimized with a Gaussian prior on the codes). It hits SoTA on ShapeNet reconstruction and partial-depth completion at a tiny 7.4 MB model size, with watertight surfaces and analytical normals from `∂f/∂x`. The two main limitations it does *not* solve: no SE(3) equivariance (must pre-align to canonical frame) and slow auto-decoding inference (latent optimization at test time).

## Research question

> Can we learn a **continuous**, **generative**, **topology-free** 3D shape representation that (a) represents an entire class of shapes (not one mesh), (b) supports shape completion from partial/noisy point clouds, (c) provides analytic surface normals, and (d) is dramatically smaller than voxel grids or template meshes?

Their answer: yes — by treating a shape's surface as the **zero level-set of a learned SDF**, with the SDF implemented as a feed-forward MLP `f_θ(z, x) → ℝ` that takes a latent code `z` (the shape) and a 3D query point `x` and regresses the signed distance. Surface extraction is then a standard Marching Cubes call on the field.

## Method

### Core idea
Classical SDFs are *analytical* (one per shape) or *voxelized* (a 3D grid of distances). DeepSDF is neither: it's a **neural field** — an MLP that can be evaluated at any `(z, x)`, and whose zero level-set is the surface. The surface is **implicit** (never discretized at training time) but can be **explicitly** extracted for downstream use via Marching Cubes or raycasting.

### The L1 clamped loss
Not a vanilla regression. The loss is:
```
L(f_θ(x), s) = | clamp(f_θ(x), δ) − clamp(s, δ) |
```
The `clamp(·, δ)` saturates both predictions and GT at `±δ` (default `δ = 0.01` for the unit-sphere-normalized shapes). The motivation is twofold:
- **Numerical:** predictions far from the surface don't need to be exact — gradient signal should concentrate on the surface.
- **Application:** "metric" SDF is only required near the surface; far-field values are just for raycast step sizes and physics proxies. Smaller `δ` → sharper surface; larger `δ` → better far-field for ray-tracing/robotics. They report a clean Chamfer-vs-`δ` tradeoff curve in Fig. 15.

### Two forms
- **(a) Single-shape DeepSDF** (Fig. 3a): one MLP per shape. Essentially memorizes one SDF. 8 FC layers, 512-dim hidden, ReLU, weight-norm (not batch-norm — they found batch-norm unstable for SDF regression), 0.2 dropout, `tanh` output, Adam, 8 GPUs × 8h. Achieves "trivially low" loss for one shape but is useless as a method.
- **(b) Coded-shape DeepSDF** (Fig. 3b): the interesting one. Latent code `z_i` concatenated to the 3D query `x` as input, **re-injected at layer 4** as a skip connection. Latent dim: **256 for reconstruction**, **128 for completion** (smaller for completion to encourage tighter class manifolds).

### The auto-decoder formulation (the actual contribution)

Park et al. push back against the standard VAE/AE pipeline. The argument:

> "Since the trained encoder is unused at test time, it is unclear whether using the encoder is the most effective use of computational resources during training."

So they drop the encoder. Each shape `i` has a latent `z_i` that's **initialized** from `N(0, 0.01²)` and **jointly optimized** with the decoder weights `θ`. The objective is the **negative log posterior** under a likelihood `p_θ(s | z, x) = exp(−L)` and a **zero-mean Gaussian prior** `p(z) = N(0, σ²I)`:

```
arg min_{θ, {z_i}}  Σ_i  [ Σ_j  L(f_θ(z_i, x_j), s_j)   +   (1/σ²) ||z_i||² ]
                          ← reconstruction loss →       ← code prior (L2 reg.) →
```

This is mathematically equivalent to a VAE without the encoder amortization, and the L2 term is doing the work of the KL divergence. They show on MNIST that the auto-decoder (AD) is on par with or better than VAE/AE reconstructions at all tested code dims (2D, 5D, 15D) and that the VAE's encoder can be replaced by optimization at test time with minor quality loss. **The big practical win:** at inference, you only need the decoder. You can also handle **arbitrary** input modalities (point clouds, depth maps, partial meshes) by writing the appropriate sampling-with-SDF-value routine — no retraining.

### Data preparation (the part most papers under-specify)

ShapeNet meshes are **not** watertight out of the box. So they:
1. Normalize each mesh to a unit sphere (radius 1/1.03).
2. Render the mesh from 100 virtual cameras on the unit sphere; back-project depth to oriented surface points.
3. Discard meshes with >2% double-sided triangles (means it's not closed).
4. Sample **250,000** surface points (area-weighted) + 25,000 free-space points uniformly in the unit sphere.
5. For each surface point, perturb along ±xyz with Gaussian noise (σ=0.05 and σ=0.016) to generate *near-surface* volume samples. This is the **most important step** — it forces the network to learn the zero-crossing, not just the surface points.
6. Build a KD-tree over the oriented surface points; sign of SDF for a query is `sign(<x − nearest_surface, surface_normal>)`.

### Training details
- Adam, decoder LR `1e-5 × batch_size`, code LR `1e-3`.
- `σ = 0.01` (code prior std).
- 16,384 SDF samples per shape per batch.
- 8 NVIDIA GPUs, ~8 hours, 1000 epochs.
- 8 FC layers, 512 hidden, 0.2 dropout, weight norm, ReLU, `tanh` final.

## Results

### Table 1 — model comparison
| Method | Representation | Topology | Closed surface | Oriented normals | Model size | Inference | Tasks |
|---|---|---|---|---|---|---|---|
| 3D-EPN | 32³ voxel SDF | ✗ | ✓ | ✓ | 0.42 GB | – | C |
| OGN | 256³ octree | ✗ | ✓ | ✓ (8 dir) | 0.54 GB | 0.32 s | K |
| AtlasNet-Sphere | 1 parametric mesh | sphere only | ✗ | ✗ | 0.015 GB | 0.01 s | K, U |
| AtlasNet-25 | 25 parametric meshes | ✓ | ✗ | ✗ | 0.172 GB | 0.32 s | K, U |
| **DeepSDF** | **continuous SDF** | **✓** | **✓** | **✓** | **0.0074 GB** | **9.72 s** | K, U, C |

DeepSDF is **~70× smaller than OGN** while being the only method that handles complex topologies *and* provides oriented normals *and* does completion.

### Table 2 — Known-shape reconstruction on cars (Chamfer × 10³, EMD)
| Method | CD mean | CD median | EMD mean | EMD median |
|---|---|---|---|---|
| OGN | 0.167 | 0.127 | 0.043 | 0.042 |
| AtlasNet-Sph. | 0.210 | 0.185 | 0.046 | 0.045 |
| AtlasNet-25 | 0.157 | 0.140 | 0.060 | 0.060 |
| **DeepSDF** | **0.084** | **0.058** | **0.043** | **0.042** |

DeepSDF nearly halves the mean Chamfer. EMD ties OGN because EMD is dominated by point-distribution shape, not precision.

### Table 3 — Test-shape reconstruction across ShapeNet classes
| CD mean | chair | plane | table | lamp | sofa |
|---|---|---|---|---|---|
| AtlasNet-Sph. | 0.752 | 0.188 | 0.725 | 2.381 | 0.445 |
| AtlasNet-25 | 0.368 | 0.216 | 0.328 | 1.182 | 0.411 |
| **DeepSDF** | **0.204** | **0.143** | **0.553** | **0.832** | **0.132** |

Mesh accuracy (90th-percentile distance) for DeepSDF is **3–10× lower** than AtlasNet — basically orders of magnitude better on fine details.

### Table 4 — Shape completion from single-view depth (vs 3D-EPN)
DeepSDF wins on CD (mean and median) on all three classes tested (chair, plane, sofa). On **planes** it's 4.7× better on median CD (0.37 vs 1.63 × 10⁻³) and 4.4× better on mesh completion (0.722 vs 0.165). The visual completions (Fig. 8) are striking — 3D-EPN gives lumpy voxelized blobs, DeepSDF gives clean continuous surfaces with correct topology.

### Table 5 — Mesh completion & normal cosine similarity
DeepSDF mesh completion 0.88–0.97 across classes. AtlasNet-25 plateaus at 0.53–0.94. Normal cosine similarity: DeepSDF 0.86–0.92 vs AtlasNet ~0.79 — the SDF zero level-set gives oriented normals for free (analytical gradient through the MLP), while AtlasNet's parametric patches have to compute normals per-triangle with arbitrary orientation.

### Noise robustness (Fig. 10–11, Suppl. B)
The killer demo: add Gaussian noise to the *inverse depth* (simulating a Kinect V1). As input noise grows from `α=0` to `α=0.05`, **raw** point-cloud-to-ground-truth Chamfer grows **superlinearly** (raw points get spread out). **DeepSDF completion** Chamfer grows only **linearly** and stays well below the raw curve. The shape prior in the learned code space is doing serious denoising. **Implication for our problem:** IOS scans have ~30–80 µm noise; DeepSDF-style completion is robust to that.

### Failure modes (Fig. 7, two right-most columns)
- **Convergence failure** in latent optimization on out-of-distribution test shapes — when the test shape has no nearby training shape, the auto-decoder diverges.
- **Lack of training data** in fine-detail regions — thin structures (chair legs, antennae) get smoothed out because the L1 loss has no incentive to preserve high-curvature detail at small spatial scale.

## Connections to our hypotheses

- **H1 (2-stage > end-to-end):** **Inconclusive.** DeepSDF is single-stage (decoder-only). But the 2-stage structure in our pipeline (segmentation → generation) maps to *decomposing* the problem: use 3DTeethSeg22's segmentation as the front-end, then DeepSDF-style auto-decoder as the per-tooth generator. So 2-stage at the *pipeline* level, single-stage at the *generation* level. This is the right architecture for us: it lets us swap in a handcrafted "is this tooth missing?" check (no need to retrain a generative model) and only deploy the neural generator where we need fine geometry.

- **H2 (diffusion > mesh VAE for surface):** **No evidence here** — DeepSDF is an auto-decoder, not a diffusion model. But it does demonstrate that **implicit continuous representations can outperform mesh-based VAEs** (AtlasNet) on surface quality by a large margin (Tables 3, 5). So H2's claim should be read as "**continuous representations** (diffusion being one instance) > mesh VAE," not "diffusion specifically > mesh VAE." Diffusion on neural fields (e.g., DiffusionSDF, LION) is a 2022-23 development we should read next. **For our project:** skip mesh-VAE entirely; go implicit.

- **H3 (conditioning on opposing + adjacent teeth improves outer surface):** **Strong conceptual support, but no direct evidence in the paper.** The auto-decoder formulation naturally supports **conditioning**: `f_θ(z, x)` can be extended to `f_θ(z, x, c_context)` where `c_context` is a feature vector from the adjacent/opposing teeth. The MAP inference (Eq. 10) at test time becomes: given a partial observation of the **context** (existing teeth) plus a **code for the missing tooth's class** (FDI number, adjacent FDI numbers), infer `z` for the missing tooth. This is exactly the conditioning pattern H3 calls for. The paper shows it works for completing a shape from its own partial scan; we extend it to "complete a *different* shape (the missing tooth) given a *context* of *other* shapes (the scan)."

- **H4 (implicit SDF > explicit mesh):** **STRONG support.** This is the *entire point* of the paper. Quantitative wins on every metric:
  - Mesh completion: 0.93–0.97 (DeepSDF) vs 0.53–0.94 (AtlasNet-25) vs 0.26–0.93 (AtlasNet-Sphere)
  - Normal cosine similarity: 0.86–0.92 (DeepSDF) vs 0.72–0.86 (AtlasNet)
  - Model size: 7.4 MB (DeepSDF) vs 172 MB (AtlasNet-25) — 23× smaller
  - Topology: DeepSDF handles arbitrary genus; AtlasNet-sphere can't even do handles
  - **For dental crowns specifically:** a crown has a complex topology (a closed surface with a concavity for the prepared tooth — genus-1 in some sense). Explicit mesh methods would need careful remeshing for the intaglio; SDF just learns the field. **Strong endorsement of H4.**

- **H5 (synthetic CAD can bootstrap training):** **Indirect support.** The paper trains entirely on synthetic ShapeNet meshes — there is *zero* real-world data. They use the same data prep pipeline (100 virtual cameras, oriented surface points, signed distance from KD-tree). The same exact pipeline can be applied to **synthetic dental CAD** (open dental CAD repos, or 3Shape/exocad sample files) to bootstrap training before we have any real IOS data. **Important: synthetic only works if the synthetic data is distributionally similar to real crowns in terms of feature diversity, not just topology.** A simple "library of 200 CAD crowns" may not cover the variability we need.

## Surprises / things buried in the paper

1. **The L1 clamp loss is the secret sauce.** Most papers use L2 on the SDF value. Park et al. use **clamped L1** with `δ=0.01` of the unit-sphere radius. The clamp does two things: (a) prevents far-field samples from dominating gradients, (b) explicitly trades off surface precision vs raycast step efficiency. Almost every subsequent neural-field paper inherits this loss. **We should use it for any SDF regression on crowns.**

2. **Skip connection at layer 4 is critical.** They re-inject the (latent + xyz) vector at the middle layer, not just at the input. Without it, the network "saturates at 4 layers" (Fig. 14). This is now a standard INRP (implicit neural representation) trick.

3. **The auto-decoder is not slower to train than a VAE** despite jointly optimizing the codes. The trick is that the code LR (1e-3) is 100× the decoder LR (1e-5), so codes catch up to the decoder quickly. This is the opposite of most VAE setups. **We should adopt this LR ratio.**

4. **No encoder, but code-prior L2 acts as a regularizer.** Drop the L2 term and the codes drift to whatever the decoder can fit; you get a degenerate latent space with bad interpolation. The Gaussian prior is the *only* thing giving the latent space structure. This is identical in role to a VAE's KL term, but trivially implemented. **For our project:** a Gaussian prior on per-tooth-class codes means similar FDI-numbered teeth should cluster in latent space, enabling the arch-curve prior from paper 001 to live in the code space.

5. **Per-shape "shape completion" is just MAP inference** (Eq. 10). This is conceptually beautiful: completion is **the same operation** as training-time code optimization, just initialized differently and run on partial data. There's no separate "completion network" — the decoder's prior over the data manifold is the completion model. **This is exactly the architecture pattern we want for crown completion.**

6. **Inference is slow (9.72 s per shape)** because of the latent optimization. This is the main limitation: not a training-time cost, but a per-test-sample cost. Acceptable for our offline CAD-generation use case (clinic wait time is hours, not seconds) but a deal-breaker for real-time. Future work (DiGS, DiffusionSDF) tries to fix this.

7. **The data-prep section is the most under-appreciated part.** 7 pages of the paper are spent on data preparation, virtual cameras, oriented normals, watertight-mesh filtering, and the ±0.05/0.016 surface-perturbation trick. The lesson: **getting the SDF samples right is harder and more important than the network architecture.** For us, getting the crown-mesh → SDF-sample pipeline right will determine our model quality.

8. **Marching Cubes at 512³ resolution is the eval bottleneck.** For quantitative evaluation they run MC at 512³ to get a mesh, then sample 30k points. For our printed crowns, we'd want 1024³ (or even use the continuous field directly via raycasting for visualization) — the marching-cubes resolution is a *quality* knob we have direct control over.

9. **Analytical normals are free.** `∂f_θ(x)/∂x` is a single backward pass. For 3D printing, you need correct outward-pointing normals; for rendering, you need smooth normals. The implicit representation gives both. This is a major win over mesh-based methods that have to compute normals from face orientation.

10. **Two follow-ups we *must* read next for the field's state-of-the-art:**
    - **DiGS (Ben-Shabat et al., CVPR 2022)** — divergence-guided shape implicit representation. Fixes DeepSDF's main weakness (slow inference, bad thin structures) by adding a divergence regularizer on the SDF gradient field.
    - **DiffusionSDF / LION (2022-23)** — diffusion models on neural implicit fields. Bridges DeepSDF (continuous) with diffusion (H2). This is the likely winner for our use case.

## Code & data

- **Code:** https://github.com/facebookresearch/DeepSDF (MIT, maintained). Includes:
  - `train_deep_sdf.py` — training loop with the auto-decoder + L1 clamp
  - `mesh_query.py` — differentiable SDF query
  - Pre-processing utilities for ShapeNet
  - Pre-trained models for chairs, planes, sofas, etc.
- **Data:** ShapeNet (v1 and v2). Both need separate downloads.
- **Reproducibility:** easy — code runs on a single GPU with reasonable defaults.

## Follow-on work that directly applies to dental crowns

These are *not* in this paper but are essential context for our project — they were found by searching for DeepSDF applications in dentistry:

- **Occudent (Park et al., MICCAI 2023)** — 3D teeth reconstruction from panoramic radiographs using **neural implicit functions** with a "Conditional eXcitation (CX)" module that fuses tooth shape + class embeddings. Trained on real panoramic radiographs, not synthesized. **This is the closest direct precedent to our project.** It uses the same DeepSDF-style representation but with class-conditional code injection — a clean example of H3-style conditioning. IoU ~0.65 on test set. Code not released.
- **ToothInpaintor (Yue et al., 2024)** — 3D tooth inpainting from a partial 3D dental model + 2D panoramic image. Uses diffusion to complete the missing region. Directly relevant to our sub-task 4 (outer surface generation). Worth reading next.
- **DenGaussDiff (Yang et al., 2025, PMC)** — 3D dental crown reconstruction from 5 intraoral images using **3D Gaussian neural fields** + diffusion priors. PSNR +1.33%, SSIM +3.9%, LPIPS -4.3% vs baselines on a 1000-case clinical dataset. This is essentially "DiffusionSDF + dental" and represents SOTA as of 2025. **Critical: it shows the implicit-SDF + diffusion combo works for crowns at clinical scale.** Should be a primary reference for our project.

## For our project

Concrete next steps ordered by priority:

1. **Clone DeepSDF and run it on a toy dental dataset** to validate the pipeline end-to-end. Don't use a fancy model yet — just confirm we can take crown meshes, sample SDF points, train, and extract a marching-cubes mesh. Estimated effort: 2 days.

2. **Adopt the L1-clamp loss, layer-4 skip connection, code-LR-100×-decoder-LR trick, and Gaussian code prior verbatim.** These are the four things that make DeepSDF work. Don't reinvent.

3. **Replace ShapeNet with the 3DTeethSeg22 train set (1,200 scans, paper 001).** Use 3DTeethSeg22's per-tooth masks to extract per-tooth point clouds, then per-tooth SDF samples. The auto-decoder learns a per-tooth SDF conditioned on a code that we expect to encode the tooth's position in the arch + local geometry.

4. **For the "missing tooth" generation, modify the auto-decoder conditioning:**
   - Old: `f_θ(z, x) → SDF(x)` — `z` is the missing tooth's code
   - New: `f_θ(z, x, c_context) → SDF(x)` — `c_context` is a feature from the adjacent + opposing teeth encoded by a PointNet/PointTransformer
   - The MAP inference becomes: given the *context* (existing teeth's SDF samples) and a *class code* (FDI number of the missing tooth), solve for `z` of the missing tooth.
   - This is **the** architectural change for H3.

5. **Skip mesh-VAE approaches entirely.** DeepSDF's results make a strong case that for our use case (closed surfaces, complex topology, smooth normals, watertight output for 3D printing), implicit is the right bet. We have 0% reason to use mesh-VAE.

6. **Plan for slow inference from the start.** DeepSDF's 9.72s per shape is a non-issue for clinical CAD (minutes-to-hours turnaround is fine) but a real issue if we want to do iterative design or interactive previews. DiGS or DiffusionSDF will give us 10–100× speedup at the cost of some quality. We may end up using DiGS as the production model and DeepSDF as the high-quality reference.

7. **Read DiGS next** (Ben-Shabat et al., CVPR 2022). It directly addresses DeepSDF's main weaknesses (slow inference, thin-structure failure) and is in our seed list. Should be paper 003.

8. **Read DiffusionSDF / LION** before committing to an architecture. Diffusion on neural fields is the most likely "best of both worlds" for H2 (diffusion) + H4 (implicit). This would be paper 004 or 005.

9. **Critical open question for HK:** **Do we want generation to be a single forward pass (diffusion) or an optimization at inference (DeepSDF auto-decoder)?** The single-pass direction lets us use diffusion (H2) and is fast at inference but needs lots of training data. The optimization direction (DeepSDF) works with less data and gives a posterior, but is slow at inference. For a clinical CAD product, I'd lean **optimization** (we have time, we don't have data); for a consumer app, **single-pass**. Discuss with HK before committing to an architecture.

10. **The data-prep pipeline is the work.** Building the crown-mesh → oriented-surface-points → SDF-samples pipeline is going to take longer than the model implementation. Allocate 1 week of engineering just for data prep, mirroring the paper's emphasis on this.

---
*Scholar 🦉 — 2026-06-05*
