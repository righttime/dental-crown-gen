# Paper 015 — PolyGen: An Autoregressive Generative Model of 3D Meshes

- **Title:** PolyGen: An Autoregressive Generative Model of 3D Meshes
- **Authors:** Charlie Nash¹, Yaroslav Ganin¹, S. M. Ali Eslami¹, Peter W. Battaglia¹
- **Affiliations:** ¹DeepMind, London
- **Year:** 2020 (arXiv v1: 23 Feb 2020)
- **Venue:** ICML 2020 (PMLR 119:7220–7229)
- **Links:**
  - Paper (arXiv): https://arxiv.org/abs/2002.10880
  - ICML proceedings: https://proceedings.mlr.press/v119/nash20a.html
  - ICML PDF: http://proceedings.mlr.press/v119/nash20a/nash20a.pdf
  - Semantic Scholar: https://www.semanticscholar.org/paper/PolyGen%3A-An-Autoregressive-Generative-Model-of-3D-Nash-Ganin/84fae4c57e9f65459cf404866f83ff0b70fd7b75
  - 3DV Geometry summary: https://3dv-geometry.github.io/Geometry/Shape-Generation/PolyGen.html
- **Code:** https://github.com/google-deepmind/deepmind-research/tree/master/polygen — Apache 2.0, with `modules.py` (model code) + `data_utils.py` + two Colabs (training from scratch on a toy dataset, sampling from a ShapeNet-pretrained checkpoint). ReZero-style residual connections and shifting-only augmentation differ from the paper for stability/simplicity.
- **Pretrained checkpoints:** Google Cloud Storage bucket `deepmind-research-polygen` (class-conditional ShapeNet, longer sequences than the paper)
- **Cite count:** 351 (Semantic Scholar, mid-2026); 14 influential
- **Read:** 2026-06-06 (Saturday, scholar weekly #15, ~50 min)

---

## TL;DR

**PolyGen is the *first* deep generative model to produce a triangle/polygon mesh *directly* as a sequence of tokens — vertices first, then faces — using a stack of two autoregressive Transformers with the second one using *pointer networks* to refer back to those vertices.** The split is the key idea: a **vertex model** is a masked Transformer decoder over a flattened sequence of 8-bit-quantized `(z, y, x)` triples sorted from low to high, outputting a categorical distribution per coordinate; a **face model** is a Transformer encoder + masked Transformer decoder + **mesh pointer network** that emits one vertex index per token, scored by a dot-product against the encoded vertex embeddings. Because vertices are sorted and pointer networks refer to existing vertices only, PolyGen can represent n-gons of arbitrary size, which makes the output mesh **human-like** (compact n-gons for flat surfaces, triangles for curvature) — not the dense over-tessellated marching-cubes output that every other pipeline in our reading list produces. SoTA on ShapeNet log-likelihood (2.46 bits/vertex for the vertex model, 1.79 bits/vertex for the face model — far below the uniform baseline's 24.08 and Draco's 27.68). Conditional generation on class labels, images (256² RGB → 16×16×256 features), and voxels (28³ → 7×7×7×256) all work, with **voxel conditioning dropping the vertex-model NLL from 2.46 → 2.19** (the most informative context). On image/voxel-conditioned chamfer reconstruction vs AtlasNet, PolyGen loses on 1 sample but **wins on 10 samples** — direct evidence that the model is genuinely multi-modal. Trained on 4× V100 in 1M + 0.5M steps. **For our project, PolyGen is the *third* and earliest entry in the "modern mesh-native generation" arc** (after MeshGPT paper 013 and MeshDiffusion paper 014), and the only one that produces *n-gons* rather than triangles — the *only* one that would directly give us a mesh with the right *connectivity* for downstream mesh boolean operations (margin fitting, contact analysis).

## Research question

> Voxel, point-cloud, and implicit-SDF generative models all require **post-hoc** mesh extraction (marching cubes, dual contouring, marching tetrahedra). The extracted mesh is dense, over-smoothed, and **not artist-like** — it's the representation we need for 3D printing but **not** the representation humans create or want. **Can we model the *mesh* directly — vertex coordinates *and* face connectivity — as a sequence, and have a Transformer learn to do it?**

Their answer: **yes, but only by splitting the task.** A single Transformer modelling `(vertices, faces)` jointly would have to learn the joint distribution over a high-dimensional discrete space where the same shape can be triangulated many ways. The PolyGen insight is to factor via the chain rule: `p(M) = p(F|V) · p(V)`. Model vertices *first* (an ordered sequence of coordinates, easy for a Transformer), then model faces *conditioned on those vertices* (a sequence of vertex indices, harder but with a built-in dependency on a fixed set of embeddings the model can refer to). This **factorization is the paper's intellectual core** — and the reason their samples look qualitatively different from every other 2020-2023 generative mesh model.

The four contributions (per Sec 1):
1. **A direct 3D-mesh generative model** (the first) that outputs a polygon mesh as a sequence, trained by maximum likelihood.
2. **A vertex model** that models sorted, 8-bit-quantized vertex sequences as a categorical distribution with a masked Transformer decoder.
3. **A face model** that uses a *mesh pointer network* to emit vertex indices referring back to the generated vertex set, enabling variable-length n-gons and avoiding the over-triangulation problem.
4. **Conditional generation on class, image, and voxel** context — with quantitative conditioning-gain measurements (Table 3) showing that richer context helps vertex prediction but *not* face prediction (because faces are mostly determined by the vertices themselves).

The conceptual lineage: PolyGen is the **mesh-as-sequence** (autoregressive) counterpart to MeshDiffusion's **mesh-as-state** (score-based) and MeshGPT's **mesh-as-sequence-on-triangles** (also autoregressive but with simpler tri-plane tokens). The 2×2 grid is now complete:
- **PolyGen (015)**: vertex-first, autoregressive, n-gons, *low* inference cost (one forward pass, no iterative sampling)
- **MeshGPT (013)**: triangle-first, autoregressive, triangles, *high* inference cost (30-90s for 6k tokens)
- **MeshDiffusion (014)**: DMTet-state, denoising, triangles, *medium* inference cost (1-3s with DDIM 100 steps)
- **GET3D (not read yet)**: tri-plane + 2D CNN, GAN, triangles, fast

## Method

### 2.1 The n-gon mesh representation

**Why n-gons and not triangles?** The paper's strongest *philosophical* claim: "Human created meshes are compact, and reuse geometric primitives to efficiently represent real-world objects." A flat surface should be a single polygon, not 800 triangles. This is the same argument the dental CAD community makes for STL files with reduced triangles. N-gons are generated via Blender's planar decimation modifier with tolerance ∈ [1°, 20°] — faces with normals within tolerance are merged.

**The cost:** n-gons do *not* uniquely define a 3D surface when Nᵢ > 3, *unless* the vertices are coplanar. For non-planar n-gons, the renderer must triangulate (e.g., by projecting to the best-fit plane), which can introduce artifacts. The paper claims most generated n-gons are *near* planar, but acknowledges it's a minor issue. **This is the most critical caveat for our project** — a crown occlusal surface is full of high-curvature cusps and fossae that *must* be triangulated, and any non-planar n-gon is a potential surface artifact. (Note: FlexiCubes, paper 007, gives us differentiable triangulation as a downstream step that could clean these up.)

### 2.2 Vertex model — masked Transformer decoder

**The setup:**
1. Sort vertices lexicographically by `(z, y, x)` ascending — vertical axis first.
2. **8-bit uniform quantization** of each coordinate into 256 bins. This (a) reduces mesh size, (b) merges nearby vertices that fall in the same bin, (c) makes the output space discrete so a categorical distribution works. Trade-off: 14+ bits is typical for lossy mesh compression; the paper notes higher resolutions are future work.
3. Flatten the sorted sequence into a 3N-length sequence of coordinate tokens: `v₁, v₂, …, v₃ₙ = (z₁, y₁, x₁, z₂, y₂, x₂, …)`. A stopping token `s` (placed only after an `x` coordinate) marks end-of-mesh.
4. Three learned embeddings per token: **coordinate embedding** (is this `x`, `y`, or `z`?), **position embedding** (which vertex?), **value embedding** (the 8-bit quantized value).
5. **Masked Transformer decoder** (Vaswani+2017 + Child+2019 / Parisotto+2019 LN-inside-residual variant) with **18 blocks**, 256-dim embeddings, 1024-dim FC layers, 0.2 dropout.
6. Output: per-step **categorical** distribution over the 256 quantized values (plus the stop token). Trained by maximum likelihood (cross-entropy).
7. **Masking at inference (not training!)** restricts predictions to valid ranges (e.g., `z_k ≥ z_{k-1}` because of the sort order). Masking at training *hurts* performance (over-constrains the data), so it's inference-only — a small but interesting design choice. Effect: -0.01 bits/vertex in Table 1 (reassigns invalid probability mass to valid values).

The vertex model alone is a competent **PointGrow-style** point cloud generator (cf. Sun et al. 2020). The face model is what makes the output a *mesh*.

**Vertex-model variants explored in Appendix E** (all 1.5× faster training, slightly worse NLL):
- **Mixture of 40 discretized logistics** (PixelCNN++-style) over a full 3-vertex triplet, with a 22-layer torso — 3.01 bits/vertex (worse, but 7.19 steps/sec vs 2.98 base)
- **MADE decoder** instead of Transformer for per-vertex output — 2.65 bits/vertex, 7.02 steps/sec
- **Transformer decoder with concatenated triplets** — 2.50 bits/vertex, 4.07 steps/sec
- **+ transformer triplet embedding** — 2.48 bits/vertex, 4.60 steps/sec
- **Base (per-coordinate, 18 blocks)** — 2.46 bits/vertex, 2.98 steps/sec

The base model wins on NLL but is the *slowest* — a classic autoregressive quality/speed trade-off. For our project, the **MADE variant** is interesting: 2.36× faster training, only 0.19 bits/vertex worse, and the MADE factorization `p(zₙ|hₙ)p(yₙ|zₙ, hₙ)p(xₙ|zₙ, yₙ, hₙ)` is *exactly* the autoregressive structure we want for ordered tooth-coordinate generation.

### 2.3 Face model — Transformer encoder + masked decoder + mesh pointer network

**The setup:**
1. Given the generated vertex set V (size Nᵥ), order faces by lowest vertex index, then next-lowest, etc. Within each face, cyclically permute so the lowest index is first. Concatenate face tuples `(f₁⁽ⁱ⁾, f₂⁽ⁱ⁾, …, f_{Nᵢ}⁽ⁱ⁾)` into a flat sequence, with a **new-face token `n`** between faces and a **stopping token `s`** at the end.
2. **Transformer encoder E** (bidirectional) embeds the vertex set into contextual embeddings `{eᵥ}`. Also includes embeddings for `n` and `s` (so the encoder output is Nᵥ + 2 embeddings).
3. **Masked Transformer decoder D** processes the flat face sequence, with each face token's value embedding obtained by **indexing into the vertex embeddings** (pointer-network style). Outputs a pointer vector `pₖ` at each step.
4. **Distribution over the next face token:** `softmax_k(pₖᵀ eₖ)` — dot product between the pointer vector and each vertex embedding, then softmax. This is the **mesh pointer network** proper (Figure 5).
5. **12 decoder blocks**, same embedding/FC dimensions as vertex model. Optionally cross-attends to vertex embeddings (ablation: cross-attention *hurts* face-model performance — the encoder already gives a strong context, cross-attention is redundant → overfit).

**Why pointer networks over a fixed-vocabulary softmax?** Because Nᵥ varies per mesh. A fixed-vocabulary softmax would need a "max Nᵥ" assumption and waste capacity; pointer networks scale to any Nᵥ and the output dimension equals the input vertex count. This is the same trick as in the original Vinyals+2015 pointer networks for the TSP.

**Masking at inference:** The same kind of validity-mask trick as the vertex model. Constraints (Appendix F):
- New-face tokens `n` cannot repeat (Eq. 24)
- First vertex of new face ≥ first vertex of previous face (Eq. 25)
- Within-face vertex indices > first index of that face (Eq. 26) — a *face can have at most one vertex with index < first*, namely itself
- Within-face indices are unique (Eq. 27)
- First index of new face ≤ lowest unreferenced vertex (Eq. 28) — **faces are emitted in increasing order of lowest index, and every face introduces at least one new vertex**; this is what guarantees mesh connectivity

These masks are the equivalent of "type checking" for faces — they constrain the autoregressive output to *always* be a valid mesh. Crucially, *no* topological constraint is enforced: self-intersections, non-manifold edges, and degenerate faces are *not* prevented. **This is the biggest "be careful" flag for our use case** — dental crown surfaces are delicate, and self-intersecting intaglio surfaces would print as broken parts.

### 2.4 Masking invalid predictions

Critical implementation detail: **mask at inference, not at training.** Ablation (Table 1):
- "PolyGen" (no mask): 2.46 bits/vertex
- "+ valid predictions": 2.47 bits/vertex (essentially zero difference)
- "PolyGen - valid predictions": invalid predictions are *not* masked at evaluation, gives inflated (worse) NLL

The paper's interpretation: the model already assigns low probability to invalid outputs, so masking at training over-constrains it. At inference, masking is "free insurance" — the model *wants* to output valid sequences, the mask just removes the (rare) invalid option.

### 2.5 Conditional mesh generation

Two conditioning mechanisms, depending on input type:
- **Global features** (class label): learned class embedding → linear projection → add to intermediate Transformer representations `H_MMH` after self-attention, broadcast over the sequence. Per-block injection (Eq. 15-16).
- **High-dimensional inputs** (image, voxel): pre-activation ResNet encoder (Table 4) → context embedding sequence → cross-attention in the Transformer decoder (Eq. 17-18), original Vaswani+2017 machine-translation style.

**Image encoder (Table 4a):** 4 conv stages, 256² → 128² → 64² → 32² → 16², with channel counts 64 → 64 → 128 → 256. Output: 16×16×256 = 65,536 spatial embeddings for cross-attention. ResNet blocks with 3×3 convs, stride-2 downsample, pre-activation.

**Voxel encoder (Table 4b):** 4 conv stages, 28³ → 14³ → 14³ → 7³ → 7³, channels 8 → 64 → 64 → 256. Output: 7×7×7×256 = 343 spatial embeddings. **This is *much* smaller than the image encoder output** (343 vs 65,536), but the input is also 200× smaller (21,952 vs 196,608 values).

**Cross-attention hurts the face model:** the encoder's bidirectional context is so strong that the additional cross-attention is redundant and overfits. The vertex model benefits (richer context = better vertex prediction) but the face model does not (face structure is mostly determined by the vertex positions, which the encoder already captured).

### 3 Experiments

**Data:** ShapeNet Core v2, 92.5/2.5/5 train/val/test split, filtered to ≤800 vertices and ≤2800 face indices. **50 augmentations per mesh** (Appendix A): (1) per-axis scaling sx,sy,sz ∈ [0.75, 1.25], (2) piecewise-linear warping with 5 sub-intervals per axis (log-normal gradients, σ²=0.5), (3) planar decimation with random tolerance ∈ [1°, 20°]. Each augmentation is rendered with random lighting (one 20W area light + 0-10 15W point lights), random camera (distance 1.25-1.5, focal length 35-50mm, filter 1.5-2), random materials (noise shader + color ramp). 256×256 renders for image-conditional training.

**Training:** 1M steps vertex model + 0.5M steps face model, 4× V100, batch size 16, Adam, lr 3e-4 with cosine annealing, 5000-step linear warmup, gradient clip norm 1.0, dropout 0.2.

**Table 1 — Unconditional NLL on ShapeNet (bits/vertex, lower is better):**
| Model | Vertices | Faces | Total | Vert Acc | Face Acc |
|---|---|---|---|---|---|
| Uniform | 24.08 | 39.73 | 63.81 | 0.004 | 0.002 |
| Valid predictions only | 21.41 | 25.79 | 47.20 | 0.009 | 0.038 |
| **Draco (Google, 8-bit)** | 27.68 total | | | | |
| **PolyGen** | **2.46** | **1.79** | **4.26** | **0.851** | **0.900** |
| + valid predictions | 2.47 | 1.82 | 4.29 | 0.851 | 0.900 |
| - discrete embedding (V) | 2.56 | — | — | 0.844 | — |
| - data augmentation | 3.39 | 2.52 | 5.91 | 0.803 | 0.868 |
| + cross-attention (F) | — | 1.87 | — | — | 0.899 |

**Table 3 — Conditional NLL (best per column):**
| Context | Vertices | Faces | Total | Vert Acc | Face Acc |
|---|---|---|---|---|---|
| None | 2.46 | 1.79 | 4.26 | 0.851 | 0.900 |
| Class | 2.43 | 1.81 | 4.24 | 0.853 | 0.899 |
| Image | 2.30 | 1.81 | 4.11 | 0.857 | 0.900 |
| Image + pooling | 2.35 | 1.78 | 4.13 | 0.856 | 0.900 |
| **Voxel** | **2.19** | 1.82 | **4.01** | **0.859** | 0.900 |
| Voxel + pooling | 2.28 | 1.79 | 4.07 | 0.856 | 0.900 |

**Key observations:**
1. **PolyGen beats Draco by 6.5× on bits/vertex** (4.26 vs 27.68) — and Draco is the *production* mesh compressor used by Google in Chrome, Draco, etc. This is the single most impressive number in the paper.
2. **Class conditioning is essentially free** (2.43 vs 2.46, 1.3% improvement) — the model already learns class structure from unlabeled training.
3. **Voxel conditioning wins**, followed by image, then class — the expected ordering by information content.
4. **Face-model conditional performance does NOT improve** with richer context — the vertex encoder already provides the relevant structure; cross-attention is redundant.
5. **Cross-attention in the face model slightly *hurts*** (1.79 → 1.87 bits/vertex) — the unexpected result, attributed to overfitting.

**Table 2 — Vertex model variants:** the base per-coordinate model wins on NLL (2.46), but the MADE-decoder variant is 2.36× faster at training (7.02 vs 2.98 steps/sec) at the cost of only 0.19 bits/vertex (2.65). The mixture-of-logistics variant is 2.4× faster but loses 0.55 bits/vertex (3.01).

**Figure 9 — Chamfer reconstruction (image/voxel conditioned) vs AtlasNet:**
- 1 sample: PolyGen *loses* to AtlasNet (AtlasNet directly optimizes chamfer)
- **10 samples: PolyGen wins** — the model is genuinely multi-modal and the best of 10 reconstructions beats AtlasNet's best of 1
- This is the cleanest empirical evidence for the **multi-modal generative** property of PolyGen: when ambiguity is high (e.g., back of object is occluded in image), multiple valid completions exist and PolyGen samples from the distribution while AtlasNet returns a single (chamfer-optimal) point

## Connections to our hypotheses (H1-H5)

### H1 — 2-stage (segmentation + generation) > end-to-end
**Partial support.** PolyGen is itself a 2-stage model — vertex model then face model. The decomposition is *strictly beneficial* (Table 1, faces-only NLL = 1.79; combined vertex+face = 4.26). The vertex model could plausibly be substituted for our **sub-task 1 (tooth detection / segmentation)** if we retrain it to output per-tooth *point sets* rather than whole-mesh vertices — a small adaptation. **Implication for our project:** PolyGen's 2-stage architecture is *direct architectural evidence* for H1, but the two stages are *mesh sub-structures* (vertices, faces), not *task sub-structures* (segmentation, generation). For H1 to hold in our context, we'd need a separate segmentation model.

### H2 — Diffusion on point clouds > mesh-based VAE for surface generation
**Mild contradiction / reformulation.** PolyGen is **neither** a diffusion model **nor** a VAE — it's an autoregressive Transformer. It produces meshes that are *qualitatively different* from both point-cloud diffusion outputs and VAE-decoded meshes. The point is: **the choice of generative family (diffusion vs VAE vs autoregressive) is orthogonal to the choice of representation (point cloud vs SDF vs mesh).** PolyGen sits in a 4th cell of the 2×2 grid: mesh × autoregressive. The right reformulation of H2 is: "**any** modern generative family (diffusion, VAE, autoregressive) on a 3D representation beats mesh-based VAE on an *intermediate* representation (point cloud, SDF, voxel) followed by a post-hoc mesh extractor" — and PolyGen is the most direct evidence: it produces the mesh directly with no post-hoc extraction, and the result is qualitatively better than AtlasNet (which is a mesh-deformed VAE) and the implicit-SDF-then-MC pipeline used by DeepSDF/DiGS (papers 002/003). **For our project:** the H2 question is "diffusion vs autoregressive on point cloud/SDF?"; PolyGen argues the *real* H2 question is "**autoregressive on mesh > all of the above**" — which is a *stronger* statement for our use case if inference cost is acceptable.

### H3 — Conditioning on opposing + adjacent teeth improves outer surface quality
**Strong support, with a caveat.** PolyGen's conditional generation story (Sec 2.5) is the cleanest H3 implementation in our reading list — the same shape, conditioned on class/image/voxels, and the per-step NLL drops with richer context (Table 3: 2.46 → 2.19 bits/vertex for the vertex model, 11% improvement). The mechanism is the standard one: **cross-attention from the decoder into a sequence of context embeddings**, computed by a domain-appropriate encoder. The caveat: **the face-model conditional performance does not improve** (1.79 → 1.81-1.82). The interpretation is that *face connectivity is mostly determined by vertex positions* — once the vertices are in the right place, the right face structure follows. **For our project, the implication is direct and important:** we should condition *vertex generation* on the partial arch (this is where the gains will be) and *face generation* should be left unconditional (or with a much weaker context). The voxel encoder (343 spatial embeddings for a 28³ input) is the closest analog to our setting — we'd want a similar encoder that takes the partial arch's point cloud and outputs spatial embeddings for cross-attention. The "voxel + pooling" ablation (2.28 vs 2.19) is a *warning*: global pooling loses ~4% of the conditional gain, so the spatial structure of the context matters.

### H4 — Implicit SDF > explicit mesh for high-quality surfaces
**Direct contradiction in spirit, with a subtle twist.** PolyGen's whole thesis is that **explicit mesh > implicit SDF** for graphics applications. The Figure 8 visual comparison (PolyGen chair vs Occupancy Networks chair post-processed to mesh) is the cleanest evidence: PolyGen produces a sparse, human-like mesh; ON produces a dense, blob-like over-tessellated mesh. The dental CAD community is *already* on PolyGen's side — 3Shape and exocad output explicit triangle/n-gon meshes, not implicit fields. **But for H4 in the *generative-model* context**, the question is what *intermediate representation* to diffuse over (or, in PolyGen's case, what to factorize an autoregressive model over). PolyGen argues for *no intermediate representation* — go straight from the data distribution to the mesh. **For our project:** if we follow PolyGen's path, our sub-task 4 (outer surface) could be a single Transformer that takes a partial arch's voxels (encoder output) and generates the missing tooth's vertices + faces as a single sequence. This *bypasses* DiGS (paper 003), FlexiCubes (paper 007), Diffusion-SDF (paper 004), and LION (paper 005) entirely. **The risk is the inference cost:** PolyGen's per-mesh generation is sequential (one coordinate at a time) — for a 256-coordinate vertex model + 800-token face model, that's ~1000 forward passes per tooth. At 18 blocks × 256-dim, each forward pass is ~50ms on a V100 → 50 seconds per tooth. **The mitigation** is the MADE variant (2.36× faster) or a distillation step.

### H5 — Synthetic data from existing CAD libraries can bootstrap training
**Strong support.** PolyGen is trained entirely on ShapeNet (synthetic CAD scans), 92.5% of which is augmented 50× via warping and decimation. No real-world data is used. The data augmentation is *aggressive* — 50 variants per mesh, with the decimation tolerance randomized to produce meshes of different sizes and connectivity. The chamfer reconstruction results on real-world TurboSquid meshes (Figures 6, 10) — PolyGen is trained on ShapeNet, evaluated on TurboSquid meshes that are clearly *not* in the training set, and the reconstructions are competitive with AtlasNet (which is trained on the same data). **This is the cleanest synthetic-to-real transfer evidence in our reading list** for autoregressive mesh generation. **For our project:** the implications are direct — if we can build a synthetic 10,000-tooth dataset (3DTeethSeg22's 1,800 scans × 5 augmentations × a few warps), we can train PolyGen-style vertex-then-face models on it and expect reasonable performance on real intra-oral scans. The 50× augmentation is a 50× effective dataset multiplier for free.

## Surprises / interesting things buried in section 4 / appendix

1. **The "valid predictions only" mask doesn't improve NLL much at inference (Table 1, -0.01 bits/vertex).** This is a strong indicator that the model *learns* the valid-output constraints well — the mask is mostly a safety net, not a real correction. The reverse is also true: masking *at training* hurts (-0.04 bits/vertex), because it over-constrains the model to ignore invalid outputs that *do* occur in the data (e.g., due to quantization collisions at the bin boundaries).

2. **The vertex model's discrete embedding is more important than the Transformer architecture (Table 1, 2.46 vs 2.56 bits/vertex = 0.10 drop when using continuous embeddings).** The 8-bit quantization isn't just for mesh size — it's also a *regularizer* that forces the model to reason about bin-level geometry rather than sub-bin precision. For our project, this suggests we should also quantize tooth-coordinate inputs/outputs to 8 bits when training our own PolyGen-style model.

3. **AtlasNet is a single-modal model trained to minimize chamfer; PolyGen is multi-modal and can produce 10 different plausible completions (Figure 9).** This is the **clinically important** property — for a real patient, a dentist wants to see 3-5 candidate crowns, not a single averaged-out "best guess". PolyGen's nucleus sampling with top-p=0.9 is a direct knob for this.

4. **Section 3.2 mentions but does not deeply analyze the augmentation "shapes"** — the 50 augmented copies are not equally valuable; the 1°-tolerance decimations produce 800-vertex meshes (richer) while the 20° decimations produce 50-vertex meshes (sparser). The model is exposed to a continuous spectrum of mesh resolutions during training, which may explain why it generalizes to TurboSquid meshes that have very different vertex counts.

5. **Draco at 8-bit gets 27.68 bits/vertex total** — this is a *lossless compressor* baseline. The fact that PolyGen gets 4.26 means it's *predicting the data distribution* so well that the residual entropy is only 4.26 bits per vertex. (Of course, this is in-distribution NLL — out-of-distribution shapes will have much higher loss.) The ratio 4.26/27.68 = 0.154 is a measure of how "predictable" the ShapeNet vertex distribution is. **For our project:** we should compute the equivalent Draco number for the 3DTeethSeg22 dataset to get a sense of how predictable tooth geometry is.

6. **The "diversity under nucleus sampling" finding (Section 3.4) is the most underappreciated result.** PolyGen samples are not only accurate, they're *diverse* — Fig 7 shows that the distribution of vertex counts, face counts, node degrees, and edge lengths closely matches the true ShapeNet distribution. This is the multi-modal property in action, and it's exactly what we want for a dental product: "give the dentist 5 crowns of different morphologies, let them pick".

7. **No "tongue" or "intra-oral scan" or "anatomical landmark" handling** — the conditioning context is purely *object-level* (the same chair from different views). For our use case, we need to encode the *anatomical context* (the partial arch's geometry, the adjacent teeth's cusps, the opposing dentition's fossae) — this is more than just "the chair from the side", it's "the chair with three legs, and we need to know which leg to add". The PolyGen image/voxel encoder is *not* designed for this — it's a global encoder. We'll need a *partial* encoder that respects the FDI-aligned neighborhood structure.

8. **The encoder–decoder asymmetric scaling is unusual:** vertex model 18 blocks, face model 12 blocks. The vertex sequence is typically longer (up to 800) than the face sequence (up to 2800 indices, but most faces are 3-4 vertices), but the *information density* of the face tokens is higher. The asymmetric depth is a manual design choice, not a systematic scaling law.

## Quote-worthy sentences

> "Existing learning-based approaches for object synthesis have avoided the challenges of working with 3D meshes, instead using alternative object representations that are more compatible with neural architectures and training approaches." (Abstract)

> "Human created meshes are compact, and reuse geometric primitives to efficiently represent real-world objects. Neural autoregressive models have demonstrated a remarkable capacity to model complex, high-dimensional data including images, text and raw audio waveforms. Inspired by these methods we present PolyGen, a neural generative model of meshes, that autoregressively estimates a joint distribution over mesh vertices and faces." (Sec 1)

> "The use of n-gons rather than triangles has two main advantages: The first is that it reduces the size of meshes, as flat surfaces can be specified with a reduced number of faces. Secondly, large polygons can be triangulated in many ways, and these triangulations can be inconsistent across examples. By modelling n-gons we factor out this triangulation variability." (Sec 2.1)

> "A caveat to this approach is that n-gons do not uniquely define a 3D surface when n is greater than 3, unless the vertices it references are planar." (Sec 2.1)

> "Surprisingly, using cross-attention in the face model harms performance, which we attribute to overfitting." (Sec 3.3)

> "This confirms our expectations, as voxels characterize the coarse shape unambiguously, whereas images can be ambiguous depending on the object pose and lighting. However the additional context does not lead to improvements for the face model, with all conditional face models performing slightly worse than the best unconditional model. This is likely because mesh faces are to a large extent determined by the input vertices, and the conditioning context provides relatively little additional information." (Sec 3.5)

> "We find that when making a single prediction, our model performs worse than AtlasNet. This is not unexpected, as AtlasNet optimizes the evaluation metric directly, whereas our model does not. When allowed to make 10 predictions, our model achieves slightly better performance than AtlasNet." (Sec 3.6)

> "We additionally note that the statistics of our mesh resemble human-created meshes to a greater extent." (Sec 3.4, after comparing to Occupancy Networks)

> "Although these are coarse descriptions of a 3D mesh, we find our model's samples to have a similar distribution for each mesh statistic." (Sec 3.4, on mesh-summary distributions under nucleus sampling)

> "In this work we opt to represent meshes using n-gons rather than triangles." (Sec 2.1)

## Code/data availability

- **Code:** https://github.com/google-deepmind/deepmind-research/tree/master/polygen (Apache 2.0) — `modules.py` has the Transformer blocks + pointer network; `data_utils.py` has the mesh preprocessing (sorting, quantization, decimation). Two Colabs (training from scratch on a toy dataset, sampling from a ShapeNet-pretrained checkpoint) are pre-built.
- **Pretrained checkpoints:** Google Cloud Storage bucket `deepmind-research-polygen` — class-conditional ShapeNet models with longer sequences than the paper. Direct link in the README.
- **Data:** ShapeNet Core v2 (Chang et al. 2015) — requires the standard ShapeNet license agreement. Rendered images are produced by the included Blender pipeline.
- **Differences from paper:** (1) global info (class label) added as the first sequence position rather than projected at each layer; (2) ReZero residual connections (Bachlechner et al. 2020) for faster training; (3) only shifting augmentations (no axis scaling or piecewise-linear warping) — found to be as effective as the full augmentation set in the paper.

## For our project

### Three takeaways

**(T1) The H2 question is the wrong H2 question.** Reformulate H2 as: "**any modern generative family (diffusion, VAE, autoregressive) on an explicit mesh representation beats every intermediate-representation-then-extraction pipeline**." PolyGen is the strongest direct evidence for this reformulation — the Figure 8 side-by-side with Occupancy Networks is a striking visual demonstration. **For our project, the implication is to seriously consider a PolyGen-style vertex+face autoregressive model as the v0 architecture**, *bypassing* the Diffusion-SDF → DiGS → FlexiCubes pipeline we've been building (papers 003, 004, 007). The "vertex model" could be conditioned on the partial arch (replacing sub-task 4's "completion" role), and the "face model" could be left unconditional (replacing the mesh-extraction role).

**(T2) n-gons are not free for dental.** The crown occlusal surface is *exactly* the high-curvature region where n-gons fail (non-planarity → triangulation artifacts). We should use triangles, not n-gons, for the *output* mesh — even if we train on n-gon-reduced meshes (a useful compression step). The export pipeline should re-triangulate n-gons via a *planarity-constrained* triangulation (e.g., ear-clipping with the vertex projected to the n-gon's best-fit plane) before 3D printing. **Practical:** write a post-processing step that converts n-gon PolyGen outputs to triangle meshes via trimesh's `triangulate_quadrify` or a custom ear-clipping with planarity check.

**(T3) Conditional generation is asymmetric across stages.** The PolyGen result that *only* the vertex model benefits from richer conditioning, while the face model does not, is a deep architectural lesson: **intermediate structural choices are mostly determined by global features; only the "fill-in" stage benefits from rich context.** For us: the partial arch (or its voxel encoding) should condition only on the *vertex* generation; the *face* generation can use a much smaller context (or none at all). This is a 30-50% reduction in conditioning compute vs. naively conditioning both stages.

### Concrete next steps

1. **Pilot PolyGen on a 1,000-tooth subset of 3DTeethSeg22 (week 3-4)**
   - Pre-process: sort vertices by `(z, y, x)`, 8-bit quantize, decimate to ≤400 vertices per tooth
   - Train vertex model (18 blocks, 256-dim) for 1M steps on 4× A100 — cost estimate: $500-1,500 on Lambda
   - Train face model (12 blocks, 256-dim) for 0.5M steps
   - **Class label** = FDI number (11-18, 21-28, 31-38, 41-48) — 32-way classification
   - Condition: leave-as-no-context for v0 pilot (just to confirm the architecture trains)
   - **Quantitative target:** Draco bits/vertex for the 3DTeethSeg22 tooth-only meshes, as the "achievable" lower bound

2. **Add partial-arch conditioning (week 5-6)**
   - Encode the partial arch (one or more missing teeth) as a 32³ voxel grid (the size of a single arch quarter)
   - Use PolyGen's 3D ResNet voxel encoder (Table 4b), output 7×7×7×256 = 343 spatial embeddings
   - Add cross-attention in the vertex model only (NOT the face model)
   - Train on (partial_arch, missing_tooth_mesh) pairs from 3DTeethSeg22
   - **Quantitative target:** per-tooth NLL 2.5 → 2.0 bits/vertex with conditioning (matching the image/voxel conditioning delta in Table 3)

3. **Compare head-to-head with PVD-AF-DiGS-FC (week 6-7)**
   - Generate the same 100 test missing teeth with both pipelines
   - Metrics: chamfer distance to ground-truth tooth mesh, surface smoothness, planarity, **occlusal-cusp preservation** (the big clinical question)
   - **Time-budget comparison:** PolyGen ~50s/tooth vs. PVD-AF-DiGS-FC ~5s/tooth (1.5s PVD + 0.5s DiGS lift + 1s FlexiCubes + post)
   - **Multi-modal comparison:** PolyGen can give 10 candidates; PVD-AF-DiGS-FC gives 1 (PVD is multi-modal but only if we run with full DDPM steps and pick the best)

4. **Adopt the v0 architecture as either PolyGen (if surface quality > speed) or PVD-AF-DiGS-FC (if speed > quality)**
   - For v0 product, PVD-AF-DiGS-FC is the right default (1-2 weeks to ship)
   - For v1 product, PolyGen is the right default (4-6 weeks, slower inference but better quality and multi-modal)
   - For v2 product, a hybrid: PolyGen for the *occlusal surface* (where n-gons are bad but multi-modality is critical for dentist choice) + FlexiCubes for the *intaglio surface* (where determinism and planarity matter)

5. **Compute and adopt Draco for the 3DTeethSeg22 tooth-only meshes** as a "what's the entropy floor" measurement
   - 8-bit quantization, highest compression setting
   - Report bits/vertex per tooth class (incisor/canine/premolar/molar)
   - This sets the lower bound for what any generative model can hope to achieve

6. **Decide: do we publish this as "n-gon crown generation" or "triangle crown generation"?**
   - The dental CAD community is split — 3Shape outputs n-gons (with quad-dominant meshes), exocad outputs triangles
   - PolyGen's n-gon output is *more compact* and *more readable* by dental software
   - The risk: many dental 3D printers require triangle STL files
   - **Recommendation:** train on n-gons for compactness, export as triangle meshes via a planarity-aware triangulation

### Open questions for HK

- **Inference cost vs. surface quality:** is 50s/tooth acceptable for a v1 product? (For comparison, a dental lab technician designs a crown in 15-30 minutes today, so 50s is 20-30× faster.) Or do we need a distilled/parallel version?
- **Class conditioning vs. partial-arch conditioning:** for v0, which is more important to validate first? I'd recommend class conditioning (32-class) before partial-arch conditioning, because the dataset is simpler and the validation is cleaner.
- **Should we also pilot the MADE vertex-model variant (2.36× faster training, 0.19 bits worse)?** This would let us iterate faster in the early weeks, with the base model as the v1 final.
- **Should we explore PolyGen's image encoder (256² → 16×16×256) for conditioning on a 2D rendering of the partial arch, in addition to the 3D voxel encoder?** This could let dentists use a smartphone photo of the patient's mouth as input (lower-fidelity but more accessible).

### Citations to follow up

- **MeshGPT (Siddiqui et al., 2023, arXiv:2311.15423) — paper 013 in our list** — the triangle-first autoregressive counterpart to PolyGen; together with PolyGen and MeshDiffusion (paper 014), completes the 2×2 grid
- **MeshDiffusion (Liu et al., 2023, arXiv:2303.08133) — paper 014 in our list** — the score-based counterpart to PolyGen
- **AtlasNet (Groueix et al., 2018, CVPR)** — the deformed-mesh VAE baseline PolyGen compares against
- **Occupancy Networks (Mescheder et al., 2019, CVPR)** — the implicit-SDF VAE PolyGen visually compares against (Figure 8)
- **DeepSDF (Park et al., 2019, CVPR) — paper 002 in our list** — the foundational implicit-SDF paper; PolyGen is the explicit-mesh alternative
- **Draco (Google)** — the lossless mesh compressor baseline; useful to compute the entropy floor of any mesh dataset
- **PointGrow (Sun et al., 2020, WACV)** — the autoregressive point-cloud generator; PolyGen's vertex model is a strictly better version (Transformer + 8-bit + n-gon support)
- **Polygon-RNN (Castrejón et al., 2017, Acuna et al., 2018)** — the 2D polygon autoregressive model; PolyGen's mesh pointer network is the 3D generalization
- **ReZero (Bachlechner et al., 2020)** — the residual-connection trick used in the official code for faster training

### Compute note

- Vertex model: 4× V100 × 1M steps × 16 batch = ~150 V100-hours
- Face model: 4× V100 × 0.5M steps × 16 batch = ~75 V100-hours
- **Total: ~225 V100-hours = ~$1,000-1,500 on Lambda for the full unconditional model on a single tooth class**
- For partial-arch conditioning pilot, add ~50% for the encoder training: ~$1,500-2,200 total
- For a full 4-class (incisor/canine/premolar/molar) v0 prototype: 4× the compute = **~$6,000-9,000 on Lambda**, or ~$1,500-2,200 if we use the MADE variant

### Note in `papers/015-polygen.md`. Next paper to read

The remaining seed-list items are:
- **3DTeethSeg** (Ben-Hamadou et al., original 2021 challenge, before 3DTeethSeg22 paper 001) — the predecessor dataset and challenge
- **MeshSegNet** (Lian et al., MICCAI 2019) — a tooth segmentation CNN
- **TS-MTL** (Liu et al., MICCAI 2021) — tooth segmentation multi-task learning
- **PointNet++ for dental meshes** (Qi et al., NeurIPS 2017) — the foundational dental mesh backbone
- **3D-Diffusion (Wu et al., 2023)** — "Diffusion Probabilistic Models for 3D Point Cloud Generation"; the original 3D point cloud diffusion paper, precursor to PVD (paper 012)
- **SDF-Diffusion** — could be the same as paper 004 (Diffusion-SDF) or a different paper; need to clarify
- **Occupancy Networks** (Mescheder et al., 2019) — the implicit-SDF VAE referenced in PolyGen's Figure 8
- **ConvONet** (Peng et al., ECCV 2020) — the locally-conditioned implicit-SDF alternative to DeepSDF
- **PCN** (Yuan et al., 2018) — the foundational point cloud completion network, precursor to PoinTr (paper 008)
- **3Shape, exocad** — closed-source commercial CAD systems; useful for studying output format
- **Tufts, OSF dental scans** — public dental scan datasets; relevant for our data acquisition plan

**Recommended next paper (016):** Either **Occupancy Networks** (the implicit-SDF VAE that PolyGen visually compares against, completing the "implicit vs explicit mesh" duality) or **3D-Diffusion (Wu et al., 2023)** (the foundational point cloud DDM, the precursor to PVD which is paper 012). The dental-specific ones (MeshSegNet, TS-MTL, 3DTeethSeg original) are also high-value since they directly address our project's sub-task 1.

**Recommendation: read Occupancy Networks next** — it's the implicit-SDF VAE that PolyGen explicitly compares against (Figure 8), it's only 10 pages, and it would close the "implicit-SDF VAE" gap in our reading list (we have DeepSDF paper 002, DiGS paper 003, Diffusion-SDF paper 004, LION paper 005, but not the original VAE). This would also let us re-evaluate H4 in light of PolyGen's explicit-mesh argument.
