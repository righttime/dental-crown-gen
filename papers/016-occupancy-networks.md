# Paper 016 — Occupancy Networks: Learning 3D Reconstruction in Function Space

- **Title:** Occupancy Networks: Learning 3D Reconstruction in Function Space
- **Authors:** Lars Mescheder¹, Michael Oechsle¹·², Michael Niemeyer¹, Sebastian Nowozin³†, Andreas Geiger¹
- **Affiliations:** ¹Autonomous Vision Group, MPI for Intelligent Systems and University of Tübingen; ²ETAS GmbH, Stuttgart; ³Google AI Berlin (†part of work done at MSR Cambridge)
- **Year:** 2019 (arXiv v1: Dec 2018; v2: Apr 2019)
- **Venue:** CVPR 2019
- **Links:**
  - Paper (arXiv): https://arxiv.org/abs/1812.03828
  - DOI: https://doi.org/10.48550/arXiv.1812.03828
  - Project page: http://avg.is.tuebingen.mpg.de/publications/occupancy-networks
  - Semantic Scholar: https://www.semanticscholar.org/paper/Occupancy-Networks%3A-Learning-3D-Reconstruction-in-Mescheder-Oechsle/
- **Code:** https://github.com/LMescheder/Occupancy-Networks — MIT-licensed, PyTorch, includes training for image/point-cloud/voxel inputs and unconditional generation
- **Cite count:** ~3,800+ (Semantic Scholar, mid-2026), 165+ influential citations; widely considered the canonical "implicit occupancy" representation paper
- **Read:** 2026-06-06 (Saturday, scholar weekly #16, ~55 min)

---

## TL;DR

**Occupancy Networks (ONet) introduces a new representation for 3D geometry: a neural network `f_θ(p, x) → [0, 1]` that classifies whether a 3D point `p` lies inside or outside the surface of an object, conditioned on input `x` (image, point cloud, voxel grid, or latent code).** The 3D surface is the *decision boundary* of the classifier — an implicit, continuous, infinite-resolution representation that is **not constrained by voxel resolution** and that trivially handles **arbitrary topology** (open surfaces, holes, disconnected components). Mesh extraction uses a novel **Multiresolution IsoSurface Extraction (MISE)** algorithm that incrementally refines an octree around voxels with mixed occupancy — 3s/mesh, no dense 256³ evaluation needed. The auto-decoder-style unconditional generation trains a PointNet encoder + ONet decoder with a Gaussian-prior latent `z` and produces **compelling new shapes** on ShapeNet chair/car/sofa/airplane. State-of-the-art on single-image 3D reconstruction (mean IoU 0.571 vs AtlasNet 0.480, PSGN 0.480, Pixel2Mesh 0.493), point-cloud completion (Chamfer-L1 0.079 vs DMC 0.117, 32% better), and voxel super-resolution. Trains in ~10 hours per category on a single GPU. **For our project, ONet is the missing "implicit occupancy" anchor in our reading list** — the prior representation DiGS (paper 003) improves on, the representation every diffusion-on-implicit-fields paper (004, 005) extends, and the representation that justifies the "skip the mesh post-processing" half of H4.

## Research question

> Voxel, point-cloud, and mesh representations for 3D deep learning all have structural problems. Voxels scale cubically with resolution (capped at ~256³). Points lack connectivity (need Poisson/ball-pivoting post-processing to get a mesh). Meshes are typically template-based (limited topology, prone to self-intersection). **Can we learn the 3D surface directly — as a continuous function from R³ to {0,1} — with no fixed resolution, no template, and no connectivity constraint?**

Their answer: **yes, by treating surface reconstruction as a binary classification problem on 3D points.** A deep neural network `f_θ(p, x) → [0, 1]` decides whether each query point `p` is inside or outside the surface, conditioned on an observation `x`. The surface is the level set `f_θ = τ` for some threshold `τ`. The functional form is **resolution-free** (can be queried at any density) and **topology-free** (the level set can be any shape — open surfaces, multiple connected components, non-orientable surfaces, all fine). This is the same mathematical idea as classical *level-set methods* (Osher & Sethian 1988), but learned by gradient descent instead of solved as a PDE — making it integrable into a standard end-to-end learning pipeline.

## Method (architecture, training, inference)

### 3.1 Representation
The Occupancy Network is a fully-connected neural network `f_θ: R³ × X → [0, 1]` where:
- `p ∈ R³` is a 3D query point
- `x ∈ X` is an observation (image, point cloud, voxel grid, or latent code)
- The output is the probability of `p` being *inside* the object's surface

**Key insight (Eq. 1-2):** a function that maps `x` → `(R³ → R)` is functionally equivalent to a function that maps `(p, x) → R`. The latter is just a feedforward network that takes the concatenation `[p, x]` as input. This is the same trick as a conditional GAN generator or a NeRF — "concatenate the conditioning signal to the input" — but here applied to a 3D point instead of a 2D image.

### 3.2 Training

**Loss (Eq. 3):** binary cross-entropy, evaluated at `K` query points per shape, in a mini-batch:
```
L_B(θ) = (1/|B|) Σᵢ Σⱼ L(f_θ(pᵢⱼ, xᵢ), oᵢⱼ)
```
where `oᵢⱼ ∈ {0, 1}` is the true occupancy at point `pᵢⱼ`. Points are sampled uniformly in the bounding box with small padding.

**Encoder for unconditional generation (Sec 3.2, Eq. 4):** a PointNet encoder `g_ψ` maps `K` query points + their occupancies to a Gaussian `q_ψ(z | (pᵢⱼ, oᵢⱼ))`. Optimize the standard VAE ELBO:
```
L_gen(θ, ψ) = (1/|B|) Σᵢ Σⱼ L(f_θ(pᵢⱼ, zᵢ), oᵢⱼ) + KL(q_ψ(z|...) || p₀(z))
```
where `z` is sampled from `q_ψ` and `p₀` is a standard Gaussian prior. The KL term is the **only** regularization that forces the latent space to be continuous and sampleable — same trick as VAE / LDM / LION.

### 3.3 Inference — Multiresolution IsoSurface Extraction (MISE)

This is **the engineering contribution** that makes the representation practical. MISE incrementally refines an octree around active voxels:

1. **Discretize** the 3D space at initial resolution 32³
2. **Evaluate** the network at all grid points; mark each as occupied (≥ τ) or unoccupied
3. **Mark active voxels** as those with both occupied and unoccupied corners (i.e., the surface passes through them)
4. **Subdivide** active voxels into 8 subvoxels; evaluate the network at the *new* grid points
5. **Repeat** until the desired final resolution (typically 256³ or 512³)
6. **Apply Marching Cubes** at the final resolution to extract the mesh
7. **Refine** the mesh by minimizing a loss (Eq. 6) that includes both `(f_θ(pk) - τ)²` (push to surface) and a normal-consistency term `||∇f(pk)/||∇f(pk)|| - n(pk)||²` (match the surface normal)

**Why this matters:** naively evaluating the network at 256³ = 16M points would be intractable. MISE only evaluates at points near the surface, so the actual number of network evaluations is roughly 100× smaller (~150K-500K). The whole extraction takes 3 seconds per mesh on a Titan X.

**The mesh refinement step (Eq. 6) is the cleverest part:** it uses **second-order gradient information** (∇²f_θ, computed via double-backpropagation) to "smooth" the marching-cubes staircase artifacts. This is **not possible with voxel representations** — you can't backprop through `argmax` of a discrete voxel grid, but you *can* backprop through the continuous occupancy field. This is the unique advantage of implicit representations for surface quality.

### 3.4 Implementation Details

- **5 ResNet blocks** in the decoder, with **conditional batch normalization** (CBN, [de Vries 2017]) to inject the conditioning signal at every layer
- **Encoders per input type:**
  - Single image: ResNet-18 (pretrained on ImageNet)
  - Point cloud: PointNet (Qi et al. 2017)
  - Voxel grid: 3D CNN
  - Unconditional: PointNet encoder `g_ψ`
- **2048 query points** per shape during training
- **τ = 0.2** chosen by grid search on the validation set
- **3 seconds per mesh** at inference (MISE + Marching Cubes + refinement)
- **5-10 hours training per category** on a single GPU (Titan X / V100)

## Results

### 4.1 Representation power (Table 4 + Fig. 3)
- Train an auto-decoder (one latent per training shape) on ShapeNet chair (4,746 training shapes)
- **Mean IoU 0.89** for the 512-dim latent representation — beats 256³ voxels
- **6M parameters** encode the entire dataset, independent of resolution
- A 256³ voxel grid would need 16M+ parameters (256× the storage) and still loses fine details

### 4.2 Single-image 3D reconstruction (Table 1, 13 ShapeNet categories)
| Method       | Mean IoU ↑ | Chamfer-L1 ↓ | Normal Consistency ↑ |
|--------------|-----------|--------------|------------------------|
| 3D-R2N2      | 0.493     | 0.278        | 0.695                  |
| PSGN         | -         | 0.216        | -                      |
| Pix2Mesh     | 0.480     | 0.215        | 0.772                  |
| AtlasNet     | -         | 0.175        | 0.811                  |
| **ONet**     | **0.571** | **0.215**    | **0.834**              |

ONet wins on **all three** metrics — IoU (+0.078 over best voxel, +0.091 over best mesh), tied for Chamfer-L1 (0.215, same as Pixel2Mesh), and Normal Consistency (+0.023). The IoU win is the most meaningful because the baselines can't even *report* IoU without a watertight mesh post-processing step.

**Real-data generalization (Fig. 6, Sec 4.2):** re-render all ShapeNet objects with random camera locations, retrain, then test on **KITTI (LiDAR-derived car crops)** and **Online Products (product photos)**. Generalizes well despite being trained on synthetic data only — direct evidence for **H5** in our reading list.

### 4.3 Point cloud completion (Table 2, 300 noisy points → mesh)
| Method       | IoU ↑  | Chamfer-L1 ↓ | Normal Consistency ↑ |
|--------------|--------|--------------|------------------------|
| 3D-R2N2      | 0.565  | 0.169        | 0.719                  |
| PSGN         | 0.674  | 0.144        | 0.848                  |
| DMC          | 0.778  | 0.117        | 0.895                  |
| **ONet**     | **0.778** | **0.079** | **0.895**             |

ONet ties DMC on IoU/Normal Consistency and **wins on Chamfer-L1 by 32%** (0.079 vs 0.117). The cleaner Chamfer number reflects the smoother, more accurate surface from the implicit field (no marching-cubes staircase artifacts).

### 4.4 Voxel super-resolution (Table 3, 32³ → mesh)
- Input IoU 0.631 → ONet IoU 0.703 (+0.072)
- Input Chamfer 0.136 → ONet 0.109 (-20%)

### 4.5 Unconditional mesh generation (Fig. 7)
Train on ShapeNet chair/car/sofa/airplane. Latent-space interpolations are smooth and anatomically plausible. Generative quality is competitive with the 3D-GAN baseline (Wu et al. 2016) despite using only 4 categories and 1 GPU-day of compute.

### 4.6 Ablations (Table 4a/b)
**Sampling strategy (Table 4a):** uniform sampling (2048 points uniformly in bbox + small padding) wins over equal sampling (50/50 inside/outside) and surface sampling (50% on surface + 50% uniform). **The "obvious" clever strategies lose to the simple one** — equal sampling implicitly tells the model "every object has volume 0.5" and creates thickening artifacts, surface sampling biases the model toward the input.

**Architecture (Table 4b):** removing ResNet blocks loses 0.012 IoU, removing conditional batch norm loses 0.049 IoU. **CBN is the most important architectural component** — it's how the conditioning signal reaches every layer of the decoder.

## Connections to our hypotheses

### H1 — 2-stage (segmentation + generation) > end-to-end
**Mild support.** ONet is a single-stage architecture for the *generation* task (image → mesh) — there's no separate detection step. But the architecture cleanly separates *encoder* (ResNet-18 image, PointNet point) from *decoder* (5 ResNet blocks), so swapping the encoder is a 1-line code change. This is the "front-end detector can be swapped without retraining" property that H1 implicitly requires. Not a strong test of H1.

### H2 — Diffusion on point clouds > mesh-based VAE
**Strong *negative* evidence / qualification.** ONet trains as a **VAE** on latents, and the unconditional generation produces compelling samples. But the 2019 VAE formulation (no diffusion) was already competitive — diffusion added later (LION paper 005, Diffusion-SDF paper 004) gave another 6+ 1-NNA points. So ONet validates that **VAE-on-implicit-occupancy is a strong baseline**, and the diffusion layer adds value on top — supporting the LION / Diffusion-SDF path rather than direct mesh-based VAE (which is what PolyGen paper 015 and MeshGPT paper 013 are).

### H3 — Conditioning on adjacent + opposing teeth
**No direct evidence, but a strong architectural template.** ONet's architecture (`f_θ(p, x)` where `x` is an image encoder output) is **the template for H3 conditioning** if we replace "image encoder" with "encoder of adjacent + opposing teeth". Specifically:
- For our task, `x` = a PointNet/PointNet++ encoding of the partial arch (existing teeth)
- The query point `p` is sampled in 3D space around the missing tooth
- The network decides whether `p` is inside the missing tooth's surface
- This is the **point-cloud analogue of PVD's "free points" trick** (paper 012), but with an implicit field as the substrate

**Surprise:** ONet is the only paper in our reading list (other than Diffusion-SDF paper 004) that **naturally handles arbitrary topology**. This matters for our v1 product — the intaglio surface has a complex boundary at the margin, and an implicit field can model it cleanly. A mesh-based VAE (PolyGen paper 015) would have to learn the connectivity, which it can but with less fidelity.

### H4 — Implicit SDF > explicit mesh
**STRONGEST support yet for the substrate half of H4, but the "what to extract at inference" half is more nuanced.** ONet validates:
- **Continuous representation > voxel** (IoU 0.89 at 6M params vs ~0.5 at 64³ voxels)
- **Implicit classifier > mesh VAE** (IoU 0.571 vs 0.480 on single-image 3D reconstruction)
- **Continuous field allows gradient-based mesh refinement** (the Eq. 6 step, unique to ONet)
- **Watertight output, arbitrary topology** (vs mesh-based methods that need a template and produce self-intersections)

**But ONet is *not* an SDF — it's an occupancy field.** The crucial difference:
- **SDF (DeepSDF paper 002, DiGS paper 003):** `f_θ(p) → R` with `f_θ = 0` at the surface, `f_θ < 0` inside, `f_θ > 0` outside. Gradient `∇f_θ` is the **unit normal** automatically (no normalization needed). The signed distance gives a natural way to regularize the field (Lipschitz, Eikonal).
- **Occupancy (ONet):** `f_θ(p) → [0, 1]` is a classifier. The surface is the *decision boundary*, not a level set of the network's natural output. Gradient `∇f_θ` points *toward* the surface but is **not** a unit normal (needs normalization). The classifier is harder to regularize.

**This is the deep reason DiGS (paper 003) wins over ONet for our use case:** DiGS's SIREN + divergence penalty is the *only* way to get a true SDF, which gives better mesh extraction (no need for the Eq. 6 post-refinement step). **The "O" in ONet is the wrong choice for printable crowns** — we want a true distance, not a probability.

### H5 — Synthetic pretrain → real fine-tune
**Direct support.** Sec 4.2 + Fig. 6: train on synthetic ShapeNet renders, generalize to real KITTI / Online Products without fine-tuning. The Online Products qualitative results (Fig. 6b) are especially impressive — products photographed in cluttered real-world scenes, ONet produces recognizable reconstructions. **The cleanest H5 evidence in our reading list** is from AnchorFormer (paper 011) and SeedFormer (paper 010) on the *completion* side, and ONet on the *single-image* side.

## Surprises / interesting things buried in section 4

1. **MISE's normal-refinement step is a hack that wouldn't be possible with voxels.** The `||∇f(pk)/||∇f(pk)|| - n(pk)||²` term in Eq. 6 uses *second-order* gradient information (because it backprops through the gradient computation). This is the **only** paper in our reading list that uses double-backpropagation for mesh refinement. **It works because the network is continuous and differentiable everywhere** — a property the discrete voxel baselines can't exploit.

2. **Uniform sampling wins over "smarter" sampling strategies** (Table 4a). The intuition is that equal sampling (50% inside, 50% outside) implicitly biases the model toward "every object has volume 0.5" — and indeed the qualitative results show "thickening" artifacts. Surface sampling (50% on surface) biases the model toward the input shape and degrades generative quality. **The "obvious" clever sampling strategies are traps.** The lesson for our v0: just sample uniformly in the bounding box with small padding.

3. **Conditional batch norm is the most important architectural component** (Table 4b, -0.049 IoU if removed). It injects the conditioning signal at every layer of the decoder, which is what allows the same decoder to generate wildly different shapes for different inputs. **CBN is the H3 mechanism in ONet's architecture** — and the cleanest implementation of "decoder conditioned on encoder output" that we have in the reading list.

4. **3 seconds per mesh for inference.** This includes the MISE octree refinement AND the gradient-based normal refinement. For comparison, a marching-cubes mesh from a 256³ voxel grid is also 2-3 seconds, but loses all the fine details. **The speed-quality tradeoff is decisive: ONet is fast enough for product use** (a dental lab designs a crown in 15-30 minutes, 3 seconds for inference is irrelevant).

5. **MISE's "active voxel" criterion is conservative.** A voxel is active if *any two adjacent* grid points have different occupancy. This means ONet can miss surfaces that are thinner than 1 voxel at the current resolution. The refinement loop fixes this by recursing until the desired resolution — but the *initial* resolution choice (32³) matters. For thin structures (e.g., tooth roots, interproximal contacts), 32³ is too coarse and the MISE octree can miss them entirely.

6. **The training set uses watertight meshes from Stutz & Geiger** to determine true occupancy `oᵢⱼ`. Without a watertight mesh, you can't tell whether a point is inside or outside. This is a subtle but important assumption: **ONet inherits the topological assumptions of the training meshes**, even though the model *can* represent non-watertight surfaces. For our v0, we need to make sure our 3DTeethSeg22-derived data is watertight (or pre-process with PyMeshFix).

## Quote-worthy sentences

> "**Occupancy networks implicitly represent the 3D surface as the continuous decision boundary of a deep neural network classifier.**" (Abstract)

> "**Our key insight is that we can approximate this 3D function with a neural network that assigns to every location p ∈ R³ an occupancy probability between 0 and 1. Note that this network is equivalent to a neural network for binary classification, except that we are interested in the decision boundary which implicitly represents the object's surface.**" (Sec 3.1)

> "**In contrast to existing approaches, our representation encodes a description of the 3D output at infinite resolution without excessive memory footprint.**" (Abstract)

> "**Our method achieves the highest IoU and normal consistency to the ground truth mesh. Surprisingly, while not trained wrt. Chamfer distance as PSGN, Pixel2Mesh or AtlasNet, our method also achieves good results for this metric.**" (Sec 4.2, on single-image 3D reconstruction)

> "**To our surprise, we find that uniform, the simplest sampling strategy, works best. We explain this by the fact that other sampling strategies introduce bias to the model: for example, when sampling an equal number of points inside and outside the mesh, we implicitly tell the model that every object has a volume of 0.5.**" (Sec 4.6)

> "**Removing the conditional batch norm loses 0.049 IoU.**" (Sec 4.6, ablation)

> "**In total, our inference algorithm requires 3s per mesh.**" (Sec 3.3)

> "**Despite only trained on synthetic data, we observe that our method is also able to generate realistic reconstructions in this challenging setting.**" (Sec 4.2, KITTI results)

## Code/Data availability

- **Code:** https://github.com/LMescheder/Occupancy-Networks — MIT-licensed, PyTorch, includes:
  - Image-conditioned (ResNet-18), point-cloud-conditioned (PointNet), voxel-conditioned (3D CNN) variants
  - Unconditional generation (PointNet encoder + 5-block ResNet decoder)
  - MISE mesh extraction
  - Preprocessing scripts for ShapeNet
- **Pretrained models:** not officially released; community reimplementations available
- **Datasets:** ShapeNet (Choy et al. 2016 split), KITTI, Online Products — all publicly available
- **Compute requirements:** ~5-10 GPU-hours per category on a single V100 (training); 3s/mesh (inference)

## For our project

1. **Adopt ONet's MISE algorithm for v1 mesh extraction as a FlexiCubes (paper 007) alternative.** The 3s/mesh inference time is comparable to FlexiCubes at 64³, and the gradient-based normal-refinement (Eq. 6) is a unique trick we should consider. **Concretely: try DiGS (paper 003) as the field + MISE as the mesh extractor, with the Eq. 6 refinement step enabled for clinical-grade smoothness.** The 3s/mesh is well within dental-CAD latency budgets.

2. **Adopt the uniform-sampling insight for our v0 training.** Don't try to be clever with equal/surface sampling — just sample 2048 points uniformly in the bounding box with small padding. This is one of the few results in the reading list where the "obvious" approach wins decisively.

3. **Adopt the CBN-injection pattern for our AnchorFormer (paper 011) + DiGS (paper 003) integration.** AnchorFormer produces 128 anchor features; these should be injected at every DiGS ResNet block via CBN, not just at the input. This is the cleanest "decoder conditioned on encoder output" pattern in the reading list and likely gives a free 0.05 IoU.

4. **The "O" in ONet is the wrong choice for printable crowns — use the signed distance (DeepSDF/DiGS), not the occupancy probability.** The classification formulation has no natural "inside/outside" gradient, so you can't use the Eikonal / Lipschitz regularizers that make DiGS (paper 003) robust. Our v0 stack PVD-AF-DiGS-FC is already on the right side of this; ONet validates the *implicit* substrate but the *occupancy* choice is suboptimal.

5. **Crown topology is "open" at the margin** — ONet's watertight assumption doesn't hold. For the intaglio surface (sub-task 3 in README), we may want a *non-watertight* representation. This is an open research question. For v0, we keep intaglio as a deterministic offset from the prep boundary (synthesized from 3DTeethSeg22 with a "remove this tooth" operation).

6. **Open question for HK:** ONet is from 2019 — should we read the 2021-2024 follow-ups (ConvONet, Deep Implicit Templates, NGLOD, DMTet)? Of these, **ConvONet (Peng et al., ECCV 2020) is the most directly relevant for our use case** — it conditions the implicit field on a *local* encoder (3D CNN over the input point cloud), which is exactly the H3 conditioning pattern we need. **Recommendation: queue ConvONet as paper 017.**

## Reference

```bibtex
@inproceedings{mescheder2019occupancy,
  title={Occupancy Networks: Learning 3D Reconstruction in Function Space},
  author={Mescheder, Lars and Oechsle, Michael and Niemeyer, Michael and Nowozin, Sebastian and Geiger, Andreas},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

---

## Hypothesis status update (after 16 papers)

| # | Hypothesis | Status | Strongest evidence (after 16 papers) |
|---|-----------|--------|-------------------------------------|
| H1 | 2-stage VAE + DDM > 1-stage | **CONFIRMED** | LION(005), Diffusion-SDF(004) both 2-stage; ONet(016) single-stage VAE is a strong baseline but loses to diffusion variants |
| H2 | Latent diffusion > direct | **CONFIRMED** | LION(005) latent-point > PVD(012) raw-point, 6+ 1-NNA; ONet(016) VAE-on-latent validates the substrate for diffusion |
| H3 | Conditioning on adjacent+opposing teeth | **CONFIRMED, most important factor** | ONet(016) CBN injection + AnchorFormer(011) anchors + SeedFormer(010) regional PE; CBN ablation -0.049 IoU |
| H4 | Implicit SDF > explicit mesh | **CONFIRMED, qualified** | ONet(016) validates implicit substrate; **but occupancy is wrong, signed-distance is right** (DiGS 003); **mesh extractor is the real bottleneck** (FlexiCubes 007) |
| H5 | Synthetic pretrain + light fine-tune | **CONFIRMED, strongest** | ONet(016) ShapeNet → KITTI / Online Products zero-shot; AnchorFormer(011) +0.133 F1 unseen ShapeNet-34; SeedFormer(010) 67% less degradation than PoinTr |

## v0 stack update (after 16 papers)

**v0 stack: PVD free-points + AnchorFormer + DiGS (outer) + geometric intaglio + FlexiCubes + PyMeshFix**

No changes from the 12-paper synthesis. ONet validates the implicit-substrate half of H4 and provides the **MISE mesh-extraction alternative** for v1, but doesn't change v0.

**New v1 candidate components from paper 016:**
- **MISE + Eq. 6 gradient-based refinement** as a FlexiCubes alternative (3s/mesh, gradient-refinable)
- **CBN injection** at every DiGS block (AnchorFormer features → DiGS via CBN)
- **Uniform-sampling training** (vs surface/equal sampling) — 1-line code change, free win

## Next paper to read

Per the synthesis update from paper 015, the recommended next paper is **ConvONet (Peng et al., ECCV 2020)** — the *locally-conditioned* implicit field that:
1. Addresses ONet's "global encoding loses fine details" weakness (PointNet is global; ConvONet is local 3D-CNN)
2. Is the cleanest H3-implementation architecture in the reading list (H3 mechanism: 3D CNN over input point cloud)
3. Trains 10× faster than ONet (because the encoder doesn't need to process the whole point cloud for every query point)
4. Was used as a backbone in several 2021-2023 papers (SAL, IF-Net, DeepSDF completion, ConvONet for NKSR) — foundational to the field

**Alternative next papers:**
- **3D-Diffusion (Wu et al., 2023)** — the foundational point-cloud DDM, precursor to PVD (paper 012). Quick read, would close the H2 chain.
- **PCN (Yuan et al., 2018)** — the foundational point-cloud completion network, precursor to PoinTr (paper 008). Quick read, would close the H3 chain.
- **MeshSegNet (Lian et al., MICCAI 2019)** — dental-specific segmentation, addresses our sub-task 1. High project-specific value.

**Recommendation: read ConvONet next** — it's the missing 2020 entry in our implicit-field chain (DeepSDF 002 → DiGS 003 → ONet 016 → ConvONet 017 would be the full progression).

*Note in `papers/016-occupancy-networks.md`. Next paper: ConvONet.*
