# Paper 013 — MeshGPT: direct triangle-mesh generation via decoder-only transformer over a learned triangle vocabulary

- **Title:** MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers
- **Authors:** Yawar Siddiqui¹, Antonio Alliegro², Alexey Artemov¹, Tatiana Tommasi², Daniele Sirigatti³, Vladislav Rosov³, Angela Dai¹, Matthias Nießner¹
- **Affiliations:** ¹Technical University of Munich · ²Politecnico di Torino · ³AUDI AG
- **Year:** 2023 (arXiv v1: 27 Nov 2023)
- **Venue:** CVPR 2024
- **Links:**
  - Paper (arXiv v1, 2023-11-27): https://arxiv.org/abs/2311.15475
  - PDF: https://arxiv.org/pdf/2311.15475
  - Project page (with videos): https://nihalsid.github.io/mesh-gpt/
  - YouTube summary: https://youtu.be/UV90O1_69_o
  - Semantic Scholar: https://www.semanticscholar.org/paper/MeshGPT%3A-Generating-Triangle-Meshes-with-Siddiqui-Alliegro/e44a11954715bb36df0303e25783968e5f3bd610
- **Code:** https://github.com/audi/MeshGPT — **Automotive Development Public Non-Commercial License v1.0** (NCL-1.0, **non-commercial restriction** — important caveat for our project; can be used for research, not for commercial product without negotiating with Audi). PyTorch 2.1.0 + cu118 + torch-scatter. Pretrained weights on Google Drive for ShapeNet chair and table.
- **Cite count:** ~250+ (Semantic Scholar, early 2024 — paper is new)
- **Funding:** AUDI AG, ERC Starting Grant Scan2CAD (804724), Bavarian State Ministry of Science and Arts via BIDT
- **Read:** 2026-06-06 (Saturday, scholar weekly #13)

---

## TL;DR

**MeshGPT is the first autoregressive transformer that generates triangle meshes directly as a sequence of triangles** — it learns a **graph-convolutional residual-vector-quantized "triangle vocabulary"** (a VQ-VAE over faces, with each face represented by 6 RQ codes drawn from a 16,384-entry codebook), then trains a **GPT-2-medium decoder-only transformer** to predict the next codebook index in the sequence (24 layers, 16 heads, 768 hidden, 4,608-token context). At inference, beam-search samples a sequence of codes which the **1D ResNet-34 decoder** turns back into 9 discretized vertex coordinates per face (128³ voxel grid per axis). The result is **the first mesh-generative model to output a directly-printable triangle mesh** — no marching cubes, no FlexiCubes, no mesh-extraction post-processing — with **artisan-level triangulation patterns** (sharper edges, fewer triangles, more artist-like than iso-surfaced neural fields) and SoTA on ShapeNet chair/table/bench/lamp (FID 18.46/6.24/8.72/19.91 vs PolyGen's 61.10/38.53/49.34/52.48 — a 30-point FID drop; COV +9% on average). The architecture is the *natural conclusion* of the LION (paper 005) line: LION is a 2-stage DDM on a *point cloud* latent → MeshGPT is a 2-stage *autoregressive* model on a *mesh* latent. The 3-stage vision: (1) **learn a compact codebook of geometric primitives** (graph convolutions on faces + residual VQ), (2) **train a transformer to generate sequences of those primitives** (GPT-2), (3) **decode sequences back to meshes** (1D ResNet). The downside is **slow inference (30-90 s/mesh on a beefy GPU)** and a **non-commercial license** that may block production use.

## Research question

> The DDPM / diffusion recipe (DDPM, score-based, Latent Diffusion) had been validated on 2D images and extended to 3D via *representations that aren't meshes* — voxels, point clouds (PVD paper 012, LION paper 005), neural fields / SDFs (Diffusion-SDF paper 004, GET3D). But all of these must be turned into a mesh via a *post-hoc* mesh-extraction step (marching cubes, marching tetrahedra, FlexiCubes paper 007, NDC paper 006). The extracted mesh is **dense, over-tessellated, and oversmoothed** — it loses the sharp features and efficient triangulation of the artist-created training meshes. **Can we generate a *mesh* directly, with the same quality, compactness, and sharpness as a human artist?**

Their answer: **yes — treat mesh generation as language modeling.** A mesh is a *sequence of triangles*; a triangle is a 3-tuple of vertices; a vertex is a 3-tuple of discretized coordinates. Sequence-modeling is well-studied (GPT family). The trick that makes it tractable is **not to tokenize coordinates directly** (which would give 9N tokens for an N-triangle mesh — too long for any transformer), but to **learn a vocabulary of triangle "tokens"** via a graph-convolutional encoder + RQ codebook, and have the transformer predict the next codebook index. This is **text-GPT for meshes**, with the vocabulary learned end-to-end rather than hand-crafted.

The three contributions (per the paper, Sec. 1):
1. **A new generative formulation for meshes as a sequence of triangles**, tailored to a GPT-style decoder-only transformer, producing **compact meshes with sharp edges**.
2. **Triangles represented as a learned vocabulary of latent geometric tokens** (the RQ codebook), enabling coherent autoregressive generation.
3. **Empirical SoTA**: 9% shape coverage increase, 30-point FID improvement, 68-86% user preference over GET3D, PolyGen, BSPNet, AtlasNet on ShapeNet chair/table/bench/lamp.

The paper is the *first* transformer that takes "mesh as language" all the way. PolyGen (Nash et al., ICML 2020) is the most direct predecessor but uses *two* autoregressive networks (one for vertices, one for faces conditioned on vertices), and the vertex/face decoupling is brittle. MeshGPT uses a *single* transformer over a *learned* vocabulary — more parameters in the vocabulary stage, but simpler at inference.

## Method

### 3.1. Stage 1: Learn Quantized Triangle Embeddings (the "vocabulary")

A graph-convolutional encoder + RQ-VAE that learns to map any triangle (with its graph context) to a small set of codebook indices. The encoder operates on the *face graph* of the mesh: each face is a node, and two nodes are connected by an edge iff the corresponding faces share a vertex. The input features per node are the **9 positionally-encoded vertex coordinates of the face triangle + face normal + angles between edges + face area** — geometrically rich enough that the encoder can capture local curvature and topology. The encoder is a stack of **SAGEConv graph convolutions** (Hamilton et al. NeurIPS 2017), producing a **576-dim feature `z_i ∈ R^576`** for each face.

**Residual Quantization (RQ)** at depth D=6 quantizes the 576-dim face feature into a stack of 6 codebook indices, drawn from a shared codebook of size 16,384 with embedding dim 192. The crucial *non-obvious* design choice: instead of one code per face, **split the 576-dim face feature into 3 vertex features of 192-dim each, aggregate features across shared vertices (mean-pool over faces sharing a vertex), then RQ each vertex feature with D/3 = 2 codes per vertex → 6 codes per face.** This **per-vertex** residual quantization is the key trick (the ablation in Table 4 shows that w/o per-vertex quantization the encoder-decoder reconstruction is *better* (98.64% vs 98.49% triangle accuracy, lower CE 0.1413 vs 0.1473) but the resulting tokens are *much* harder for the transformer to learn downstream — the per-vertex sharing creates *sequence-regularity* in the code sequence that helps the transformer generalize, as illustrated in Fig. 16: two faces sharing an edge have overlapping code subsequences, so the transformer sees a partially-redundant sequence it can pattern-match).

**The decoder (Sec. 3.1, Fig. 14) is a 1D ResNet-34** that consumes the quantized face features in sequence order (the PolyGen-style ordering: faces ordered by lowest vertex index, vertices sorted in z-y-x, indices cyclically permuted so the lowest comes first — Sec. 3.1) and outputs **logits over 9 discrete coordinates per face** (each coord discretized into 128 bins on a 128³ voxel grid). The **discrete-output prediction** (vs. regression to continuous coordinates) is a 5× triangle-accuracy boost (Table 4: 22.03% with continuous vs 98.49% with discrete) and visually eliminates "floating face" artifacts (Fig. 4). The loss is a **smoothed cross-entropy** (Eq. 16-17, with a smoothing kernel `w_{nijk} = smooth(one-hot_{128}(V_{nij}))` that softens the penalty for physically-close coordinates) plus a **commitment loss** `L_commit(z, ẑ) = Σ_d ||z - sg[ẑ^(d)]||²` on the RQ codes (Eq. 10). Trained for ~2 days on 2 A100s, batch size 32, Adam, lr 1e-4.

The trained encoder + codebook are then *frozen* and used to tokenize every training mesh into a sequence `T = (t_1, t_2, ..., t_N)` where each `t_i ∈ {1, ..., 16384}⁶` (6 codes per face). The sequence length is `|T| = 6N` — for a mesh with N ≤ 800 faces (the training-set limit), the longest sequence is 4,800 tokens, which fits the transformer's 4,608-token context window.

### 3.2. Stage 2: GPT-2 Transformer over the learned vocabulary

A **GPT-2-medium decoder-only transformer** is trained to do next-index prediction over the codebook indices. The architecture is *vanilla* GPT-2: 24 multi-headed self-attention layers, 16 heads, 768 hidden dim, 4,608 context length, learned discrete positional encoding (indicates face position in the sequence AND index within the face's 6 codes). Input: the embedding `e(t_i^d)` for the d-th code of face i. Output: the predicted codebook index for the *next* position. The loss is the standard autoregressive cross-entropy (Eq. 18):

```
L = Σ_i Σ_j Σ_k log p(s_i^j = t_i^j)        (j ∈ {1,...,6}, k ∈ {1,...,|C|=16384})
```

The full training objective is **max-log-likelihood of the training sequences** (Eq. 6):

```
Π_i=1..N Π_d=1..D p(t_i^d | e(t_<i^d), e(t_<d_i); θ)
```

Trained for ~5 days on 4 A100s, effective batch size 64, Adam, lr 1e-4. The transformer is **pretrained on all 55 ShapeNet categories** for ~2000 epochs, then fine-tuned on the target category (chair / table / bench / lamp) for an additional ~2000 epochs. The pretraining ablation in Table 3 (w/o pretraining → COV 36.97, MMD 3.73, FID 27.54) is the difference between overfitting and generalization.

### 3.3. Inference and post-processing

**Beam search sampling** (default `beam=25` per the repo inference script) generates a sequence of codebook indices, starting with a learned START token and continuing until a learned STOP token. The codebook embeddings indexed by this sequence are summed + concatenated per the per-vertex aggregation rule (Eq. 5), arranged in the PolyGen face-ordering, and passed through the **1D ResNet-34 decoder** to produce 9 discrete vertex coordinates per face. These are mapped back to continuous coordinates by the inverse of the 128³ discretization (each bin is 1/128 of the unit bounding box).

The output is initially a "triangle soup" with duplicate vertices for neighboring faces — a simple **vertex-merging post-processing step** (e.g., MeshLab's merge-close-vertices) yields the final watertight mesh. **This is the only post-processing** — no marching cubes, no SDF lifting, no isosurface extraction. The output IS the printable mesh. Inference time: **30-90 s/mesh on a high-end GPU** (autoregressive decoding of ~800 tokens × 6 codes = ~4,800 indices), which is the paper's main limitation.

### 3.4. Implementation Details (App. B)

- **Encoder**: stack of SAGEConv graph convolutions on the face graph, input features = 9 positionally-encoded coordinates + face normal + edge angles + area. Output: 576-dim per-face features.
- **Decoder**: 1D ResNet-34, treats face features as a 1D sequence, outputs logits over the 128³ discrete coord space.
- **Codebook**: 16,384 codes, dim 192, exponential moving average updates, stochastic code sampling, shared codebook across all RQ levels (Lee et al., CVPR 2022).
- **Transformer**: GPT-2-medium (24 layers, 16 heads, 768 dim, 4,608 context), learned discrete positional encoding, learned START and STOP tokens.
- **Data augmentation**: random scaling `[0.75, 1.25]` per axis (rescaled to unit length), random shift `[-0.1, 0.1]`, planar decimation at varying levels (all within Hausdorff threshold).
- **Optimizers**: Adam, lr 1e-4, batch size 32 (VQ) / effective 64 (GPT), both stages PyTorch.

### 3.5. Data (App. A)

- **ShapeNetV2**, all 55 categories, 28,980 shapes after planar decimation.
- Per-shape: **planar decimation via Blender** with angle tolerance `α ∈ [1, 60]`. The decimated shape with Hausdorff distance closest-to-but-below a threshold `δ_hausdorff` is kept. Shapes with > 800 faces are excluded. (This is the data-filter step that limits sequence length to 4,800 tokens.)
- Chair/Table/Bench/Lamp: 9:1 train-test split. Remaining 51 categories: used for the *pretraining* phase only.
- All shapes normalized to origin-centered, longest side = 1.

## Results

### Table 1 — Unconditional generation on ShapeNet (CD-based MMD/COV/1-NNA, plus FID/KID on rendered images, plus compactness |V|/|F|)

**Chair**

| Method | COV↑ | MMD↓ | 1-NNA | FID↓ | KID↓ | \|V\| | \|F\| |
|---|---|---|---|---|---|---|---|
| AtlasNet [18] | 9.03 | 4.05 | 95.13 | 170.71 | 0.169 | 2500 | 4050 |
| BSPNet [7] | 16.48 | 3.62 | 91.75 | 46.73 | 0.030 | 673 | 1165 |
| PolyGen [43] | 31.22 | 4.41 | 93.56 | 61.10 | 0.043 | 248 | 603 |
| GET3D [14] | 40.85 | 3.56 | 83.04 | 81.45 | 0.054 | 13725 | 27457 |
| GET3D* (QEM to 400) | 38.75 | 3.57 | 84.07 | 78.29 | 0.065 | 199 | 399 |
| **MeshGPT** | **43.28** | **3.29** | **75.51** | **18.46** | **0.010** | 125 | 228 |

**Table**

| Method | COV↑ | MMD↓ | 1-NNA | FID↓ | KID↓ | \|V\| | \|F\| |
|---|---|---|---|---|---|---|---|
| AtlasNet | 7.16 | 3.85 | 96.30 | 161.38 | 0.150 | 2500 | 4050 |
| BSPNet | 16.83 | 3.14 | 93.58 | 30.78 | 0.017 | 420 | 699 |
| PolyGen | 32.99 | 3.00 | 88.65 | 38.53 | 0.029 | 147 | 454 |
| GET3D | 41.70 | 2.78 | 85.54 | 93.93 | 0.076 | 13767 | 27537 |
| GET3D* | 37.95 | 2.85 | 81.93 | 50.46 | 0.037 | 199 | 399 |
| **MeshGPT** | **45.68** | **2.36** | **72.88** | **6.24** | **0.002** | 99 | 187 |

**Bench**

| Method | COV↑ | MMD↓ | 1-NNA | FID↓ | KID↓ | \|V\| | \|F\| |
|---|---|---|---|---|---|---|---|
| AtlasNet | 20.53 | 2.47 | 90.58 | 189.39 | 0.163 | 2500 | 4050 |
| BSPNet | 28.74 | 2.05 | 88.44 | 59.11 | 0.030 | 457 | 756 |
| PolyGen | 51.92 | 1.97 | 76.98 | 49.34 | 0.031 | 172 | 430 |
| **MeshGPT** | **55.23** | **1.44** | **68.24** | **8.72** | **0.001** | 159 | 291 |

**Lamp**

| Method | COV↑ | MMD↓ | 1-NNA | FID↓ | KID↓ | \|V\| | \|F\| |
|---|---|---|---|---|---|---|---|
| AtlasNet | 19.97 | 4.68 | 91.85 | 177.91 | 0.139 | 2500 | 4050 |
| BSPNet | 18.38 | 5.32 | 93.13 | 112.65 | 0.077 | 587 | 1011 |
| PolyGen | 47.86 | 4.18 | 81.42 | 52.48 | 0.025 | 185 | 558 |
| **MeshGPT** | **53.88** | **3.94** | **65.73** | **19.91** | **0.004** | 150 | 288 |

**Net:** MeshGPT wins on *every cell* of every category. Headline numbers:
- **FID drops 30+ points** vs PolyGen (e.g., 61.10 → 18.46 on chair, 38.53 → 6.24 on table, 49.34 → 8.72 on bench, 52.48 → 19.91 on lamp). This is the *visual quality* metric (rendered images, Inception features) — the largest single-method improvement in our reading list.
- **COV improves 9 points on average** vs the best baseline. Chair: 43.28 vs GET3D's 40.85 (+2.4). Table: 45.68 vs GET3D's 41.70 (+4.0). Bench: 55.23 vs PolyGen's 51.92 (+3.3). Lamp: 53.88 vs PolyGen's 47.86 (+6.0).
- **MMD improves 0.1-0.5 points** on every category (3.29 vs 3.56 on chair, 2.36 vs 2.78 on table, 1.44 vs 1.97 on bench, 3.94 vs 4.18 on lamp).
- **|F| is 2-5× smaller than the best baseline** (228 vs 603 on chair, 187 vs 454 on table, 291 vs 430 on bench, 288 vs 558 on lamp). This is the "compactness" win — MeshGPT meshes are *artisan-style* in face count, not iso-surfaced.
- **GET3D* (QEM-decimated to 400 faces) has COV drop from 40.85 → 38.75**, showing that QEM loses fine structures. MeshGPT doesn't need QEM.

**Comparison vs. our prior reads**:
- vs PolyGen (the only prior autoregressive mesh generator): MeshGPT wins by 5-15 FID points AND uses a single network (PolyGen uses two). The vertex/face separation in PolyGen is the failure mode.
- vs GET3D (best implicit-field method in 2022, paper 014 in our seed list): MeshGPT wins on FID, COV, |F| — and is directly usable (no marching cubes).
- vs BSPNet (compact meshes via BSP tree): MeshGPT wins on all metrics AND is *much* more general (BSPNet is restricted to convex decompositions, which is why its lamp COV is 18.38 — it can't model the non-convex lampshade shape).

### Table 2 — User Study (49 participants, 784 responses)

| Preference (vs. our method) | AtlasNet | BSPNet | PolyGen | GET3D |
|---|---|---|---|---|
| Our shape is better | 82.65% | 78.57% | 85.71% | 68.37% |
| Our triangulation is better | 84.69% | 71.43% | 84.69% | 73.47% |

MeshGPT is *significantly* preferred over all four baselines on both shape quality and triangulation quality, by 68-86% of users. **The 68% over GET3D for shape quality is the most relevant for our use case** — even over a high-quality implicit-field model, MeshGPT's artisan-like triangles are preferred 2-to-1 by humans.

### Table 3 — Ablation on Chair (CD-based metrics, FID, KID)

| Variant | COV↑ | MMD↓ | 1-NNA | FID↓ | KID↓ |
|---|---|---|---|---|---|
| w/o Learned Tokens (naive 9N coord tokenization) | 27.50 | 4.51 | 93.15 | 40.20 | 0.024 |
| w/o Encoder Features (transformer learns its own embeddings) | 39.24 | 3.43 | 84.48 | 30.35 | 0.017 |
| w/o Pretraining (no all-55-category pretrain) | 36.97 | 3.73 | 84.69 | 27.54 | 0.014 |
| w/o Sequence Compression (no D-dim RQ; one code per face) | 30.98 | 4.15 | 88.98 | 38.76 | 0.023 |
| w/o per-Vertex Quantization (RQs the face feature directly) | 23.57 | 5.49 | 98.35 | 74.94 | 0.050 |
| **MeshGPT (full)** | **43.28** | **3.29** | **75.51** | **18.46** | **0.010** |

**Every design choice contributes, but per-vertex quantization and learned tokens are the largest single contributors** (COV drop 19.71 and 15.78 respectively). The pre-training ablation is moderate (COV drops 6.31, FID +9) — pretraining is necessary for generalization but not for fitting the target category. **The w/o per-vertex quant is the *worst* variant by far** (COV 23.57 vs 43.28, FID 74.94 vs 18.46) — confirming the *paper's* main claim: the per-vertex sharing creates sequence-regularity that helps the transformer, even though the encoder-decoder reconstruction is technically slightly better without it (Table 4).

### Table 4 — Encoder-decoder ablation on Chair (triangle reconstruction accuracy, cross-entropy)

| Variant | Triangle Acc. (%) ↑ | Cross-Entropy ↓ |
|---|---|---|
| w/o Positional Encoding | 79.33 | 0.2484 |
| w/o Output Discretization | 22.03 | 0.5705 |
| w/o Residual Quantization | 1.29 | 4.6679 |
| w/o per-Vertex Quantization | 98.64 | 0.1413 |
| w/ PointNet Encoder | 88.73 | 0.1896 |
| w/ GAT Encoder | 86.14 | 0.2015 |
| w/ EdgeConv Encoder | 91.23 | 0.1702 |
| w/ ResNet19 Decoder | 96.29 | 0.1492 |
| w/ PointNet Decoder | 95.47 | 0.1528 |
| **MeshGPT** | **98.49** | **0.1473** |

**The encoder-decoder ablation is independent of the GPT ablation** — it tells us how good the *reconstruction* step is (how well the encoder + RQ + decoder round-trip a mesh through the discrete codebook). Key findings:
- **w/o RQ is catastrophic** (1.29% accuracy, CE 4.67) — RQ is non-negotiable.
- **w/o output discretization** drops triangle accuracy from 98.49% to 22.03% — the discrete-output prediction is what makes the decoder produce clean meshes (continuous regression gives "floating face" artifacts, Fig. 4).
- **SAGEConv > EdgeConv > GAT > PointNet** as encoder (98.49 > 91.23 > 86.14 > 88.73) — SAGEConv's neighborhood-sampling is the right inductive bias for face graphs.
- **w/o per-Vertex Quantization gives the BEST reconstruction (98.64%)** but as we saw in Table 3, this *worst* variant downstream — confirming that *what's good for the encoder-decoder is bad for the transformer*.

### Qualitative results (Fig. 2, 4, 6, 7, 8, 9, 11, 12, 13)

- **Fig. 2 (chair/table/bench/lamp comparisons vs. GET3D)**: MeshGPT produces meshes with **artisan-level triangulation** — non-uniform triangle sizes that adapt to local curvature (small triangles on cusps, large triangles on flat regions). GET3D produces **dense, uniform, iso-surfaced** meshes that look computer-generated, not hand-modeled.
- **Fig. 4 (decoder discretization comparison)**: Continuous regression produces **floating face artifacts** (vertices that don't merge with neighbors); discrete prediction produces clean meshes that match ground truth.
- **Fig. 6, 7 (chair/table/bench/lamp vs. baselines)**: MeshGPT is the only method that captures the *fine details* (chair armrests, table edge bevels, lamp filament geometry) without oversimplifying (BSPNet) or over-tessellating (GET3D).
- **Fig. 8 (shape novelty analysis)**: For 500 generated chairs, the 3 nearest training-set neighbors are mostly far away (high CD), but a non-trivial fraction are very close (low CD) — the model *interpolates* the training distribution, doesn't just memorize.
- **Fig. 9 (shape completion)**: Given a partial mesh, MeshGPT can generate multiple plausible completions. The diversity is limited compared to diffusion-based methods (it's autoregressive, not sampling from a learned distribution) but it's a useful feature.
- **Fig. 12, 13 (shape novelty, all 4 categories)**: 3-nearest-neighbor analysis for every figure in the paper, showing the model generates *novel* shapes, not training-set retrievals.

## Connections to our hypotheses (H1–H5)

### H1 — "2-stage (segmentation + generation) outperforms end-to-end"

**STRONG support at the *internal* level; mild support at the *project* level.** MeshGPT is internally 2-stage (VQ-VAE then GPT), exactly like LION (paper 005) and Diffusion-SDF (paper 004). The internal 2-stage wins by 30 FID points over PolyGen's *internal* 2-network architecture (vertex-net + face-net) — confirming that **a single transformer over a learned vocabulary is better than two specialized transformers over hand-crafted representations**. At the *project* level (segmentation + generation), MeshGPT doesn't speak to it directly — it's a *pure generation* model with no segmentation front-end. But the architecture is **compositional** with sub-task 1 (segmentation) and sub-task 4 (outer surface generation): the missing tooth can be identified by AnchorFormer (paper 011) / PoinTr (paper 008), and MeshGPT can generate the printable mesh. **Mild H1 support via composability.**

### H2 — "Diffusion on point clouds > mesh-based VAE for surface generation"

**MILD CONTRADICTION — and a strategic pivot for our v0 architecture.** MeshGPT is **not diffusion** and **not point cloud** — it's **autoregressive mesh generation**. The H2 claim was specifically about *point-cloud DDMs* being better than *mesh-based VAEs*. MeshGPT shows that **mesh-based autoregression can beat point-cloud DDMs** — at least on ShapeNet FID. Compare:
- PVD (paper 012, point cloud DDM, direct): chair 1-NNA-CD 56.26.
- LION (paper 005, latent point DDM): chair 1-NNA-CD 53.70.
- MeshGPT (autoregressive mesh): chair FID 18.46 (FID, not CD — but FID is the *visual quality* metric and is a more meaningful number for our use case).

**The H2 claim should be restated**: H2 is "**diffusion on raw geometric representations is one viable paradigm; mesh-based autoregression is another**. The right answer for *printable dental crowns* is the representation that produces the cleanest, sharpest, most-artisan-like mesh output, regardless of whether the underlying mechanism is diffusion or autoregression." MeshGPT's autoregressive-mesh paradigm is the **strongest candidate for sub-task 5 (mesh output)** because:
- (a) the output is a **directly-printable triangle mesh** (no SDF lifting, no FlexiCubes extraction);
- (b) the **artisan triangulation** matches the quality of human-crafted dental CAD (cusps, ridges, fossae as discrete features, not iso-surfaced bumps);
- (c) the **autoregressive structure** allows *constrained generation* (condition on a fixed set of existing teeth) without changing the loss function.

**H2 is now reopened**: the question is no longer "diffusion vs VAE on points" — it's "**autoregressive mesh vs diffusion on points/SDF for printable output**." For our project, this is a critical decision point.

### H3 — "Conditioning on opposing + adjacent teeth improves outer surface quality"

**STRONG support in mechanism, but NOT directly demonstrated in this paper.** MeshGPT's training data is unconditional (no input conditioning), so it doesn't directly validate H3. *But* the architecture is **naturally extensible** to H3 conditioning: at training time, condition the GPT on a set of "context face tokens" (the existing teeth's triangles) by prepending them to the sequence; at inference, give it the context teeth's tokenized mesh and let it generate the missing tooth's tokens autoregressively. This is the same pattern as LION's `ϵ_ψ(h_t, z0, t)` (paper 005) but in *mesh* space and *autoregressive* mode. **H3 is the natural next extension of MeshGPT — and a perfect v0-to-v1 architecture for our project: condition MeshGPT's GPT on the encoded existing arch (from AnchorFormer / 3DTeethSeg22), generate the missing tooth's triangles conditioned on the arch, output a printable mesh directly.** This would replace the **PVD-AF-DiGS-FC** stack (paper 012) with a **MeshGPT + AnchorFormer** stack that's *one model* instead of *four*.

**The H3 advantage of MeshGPT over PVD-AF-DiGS-FC**: in PVD-AF-DiGS-FC, the conditioning signal has to be threaded through *three* different architectures (PVD's PVCNN, DiGS's SIREN, FlexiCubes' 3D U-Net) with three different conditioning mechanisms (free-points mask, code initialization, regularizer weights). In MeshGPT, the conditioning is *a single transformer context* — prepend, predict, done. **Architectural simplicity is a strong argument for the MeshGPT path for v1.**

### H4 — "Implicit SDF representation > explicit mesh for high-quality surfaces"

**STRONG CONTRADICTION — and a major reconsideration.** MeshGPT produces *explicit* meshes that *beat* the *implicit SDF* methods (GET3D, Diffusion-SDF) on every metric. The "implicit SDF > explicit mesh" claim was based on the idea that *iso-surfaced implicit fields* (Marching Cubes, FlexiCubes) produce **dense, oversmoothed** meshes that lose the artisan triangulation of the training data. MeshGPT's contribution is to show that **explicit mesh generation is possible with the right representation** — and the resulting meshes are *better*, not worse, than the iso-surfaced implicit alternatives.

**H4 should be restated**: H4 is "**for high-quality surfaces, the right *representation* matters more than the dichotomy between implicit and explicit**." An explicit-mesh representation (MeshGPT's RQ codebook over face features) produces high-quality surfaces; an implicit-SDF representation (DiGS paper 003) also produces high-quality surfaces; the question is which representation is *better suited* to the specific output requirements. **For our project**: dental crowns need *sharp cusps and ridges*, which are exactly the features that *iso-surfacing* (the implicit-SDF → mesh step) destroys. MeshGPT's autoregressive-mesh representation *preserves* sharp features because it *generates* them explicitly, not extracts them from a smooth field. **The H4 camp for our project is now: MeshGPT > Diffusion-SDF + FlexiCubes.** This is a substantive revision of our prior v0 architecture.

### H5 — "Synthetic data from existing CAD libraries can bootstrap training"

**STRONG support — exactly the same H5 precedent as PVD (paper 012), LION (paper 005), and Diffusion-SDF (paper 004).** MeshGPT trains on ShapeNet (synthetic, 28,980 shapes) and presumably transfers to real-world data (the user study uses ground-truth ShapeNet meshes, so no direct real-world transfer in this paper, but the architecture is fundamentally a synthetic-trained model). The H5 case is identical to PVD/LION/Diffusion-SDF: **CAD-library-trained → patient IOS scan transfer should work with at most light fine-tuning.** The non-commercial license is the only H5 caveat for our project (see "For our project" below).

## Surprises / interesting things buried in the paper

1. **The per-vertex quantization ablation in Table 4 is the *opposite* of what you'd expect.** W/o per-vertex quantization (RQ the face feature directly) gives the *best* encoder-decoder reconstruction (98.64% triangle accuracy, lowest CE 0.1413). But the full MeshGPT (with per-vertex quant) gives *worse* reconstruction (98.49%, CE 0.1473). And yet, the per-vertex quant version gives a **far better downstream transformer** (Table 3: COV 43.28 vs 23.57). **The lesson: an "objectively better" reconstruction is not always what you want for a *discrete codebook* stage. The per-vertex sharing creates *sequence-level regularity* (shared vertices → shared codes → redundancy the transformer can exploit) that the per-face quantization destroys.** This is the same insight as the **VQ-VAE codebook collapse problem** in image generation (e.g., DALL-E 1, VQGAN) — sometimes a "worse" codebook is "better" for the downstream generative model because the codebook is more *regular*.

2. **The PolyGen-style face ordering (faces ordered by lowest vertex index, vertices sorted in z-y-x) is a non-trivial architectural choice that the paper takes from PolyGen without re-deriving.** This ordering creates a *canonical* sequence for any mesh (no permutation ambiguity), which is critical for the autoregressive loss. The choice of z-y-x (vs. x-y-z, or any other) is empirically shown to give the best results in PolyGen — but **the ordering is also where the "spatial prior" is encoded**. A mesh generated in z-y-x order has a *vertical* generation bias (early faces are at the bottom, later faces at the top). For dental, the analogous choice would be **occlusal-first ordering**: generate the cusps/ridges first, then propagate down to the cementum/margin. This is a small but important detail for adapting MeshGPT to dental.

3. **The 128³ coordinate discretization is a *hard* design choice that affects *what kinds of shapes* can be learned.** A 128³ grid gives a coordinate resolution of 1/128 ≈ 0.78% of the unit bounding box per axis. For a tooth ~10mm in size, that's ~78μm per voxel — **at the limit of clinical fit accuracy**. For sub-50μm margin accuracy, the discretization needs to be bumped to 256³ or 512³, which would change the encoder-decoder training dynamics. **For our project**: this is a real engineering constraint that the paper doesn't address, and it could be a blocker for direct application of MeshGPT to dental. The fix would be a *separate coordinate prediction head* (predict continuous coordinates for the cusps/ridges, predict discretized coordinates for the rest) — but this is non-trivial.

4. **The 30-90 s inference time is the *biggest* practical limitation.** For a dentist who wants to *iterate* on a crown design (try a different prep margin, see the result, adjust), 30-90 s is too slow. The autoregressive decoding of ~4,800 indices is the bottleneck. Speculative decoding, parallel decoding, or non-autoregressive variants could fix this — but **the paper doesn't explore any of them**, and they're all non-trivial architectural changes. **For our v0 prototype**: 30-90 s/mesh is acceptable (the dentist doesn't need real-time), but for v1 productization we'd need 1-5 s/mesh. The right place to invest here is **a non-autoregressive MeshGPT variant** (e.g., masked-token-prediction like MaskGIT, or diffusion-on-the-codebook like the LION-style VAE-then-DDM).

5. **The vertex-merging post-processing step is glossed over in the main text but is critical for practical use.** The output of the 1D ResNet-34 decoder is a "triangle soup" with duplicate vertices for every face — each shared vertex is replicated as many times as the number of faces that share it (in a typical mesh, 4-6 faces share each vertex, so each unique vertex is represented 4-6 times). The MeshLab merge-close-vertices step *heals* the soup into a watertight mesh, but **the merge tolerance is a free parameter** that the paper doesn't specify. Too small → non-manifold edges (printing failure). Too large → over-smoothing (loss of detail). **For our project**: the merge tolerance needs to be tuned to a specific tooth resolution (e.g., 0.1% of the unit bounding box), and the resulting mesh needs to be validated for watertightness and non-self-intersection. This is a non-trivial engineering step.

6. **The non-commercial license (NCL-1.0) is a real blocker for production use.** The code is freely available for research, but the NCL-1.0 license explicitly prohibits commercial use. If we want to commercialize a crown-generation product based on MeshGPT, we need to either (a) negotiate a commercial license with Audi (expensive and slow), (b) build our own implementation from scratch using the paper as a reference (significant engineering effort, ~3-6 months), or (c) use MeshGPT for research/validation and ship a different architecture (e.g., a custom diffusion-on-mesh model) for the product. **For our v0 pilot, the non-commercial restriction is fine** (we're a research project). **For v1 productization, we need a strategy**.

7. **The user study (Table 2) has a non-trivial design choice: 68% preference over GET3D for *shape quality*, but only 73% for *triangulation quality* (vs 68% for shape).** This is *counterintuitive* — the triangulation is the *headline* contribution of MeshGPT (the artisan-style non-uniform triangle sizes), so we'd expect the triangulation advantage to be larger. The fact that shape and triangulation preferences are similar suggests that **users see MeshGPT as "more visually appealing" overall, not specifically as "better-triangulated"** — i.e., the artisan triangulation translates to perceived visual quality, not as a separable property. **For our project**: this means the *clinician-facing metric* should be "overall crown aesthetics" (a dentist's qualitative rating), not "triangle count" or "triangle regularity" (a CAD-software metric). The user-study design is the canonical citation for this stance.

8. **The 6-codes-per-face RQ + 16,384-code codebook gives a 6 × 16,384 = 98,304 possible code sequences per face position, with 800 faces → 800¹⁰⁰⁰⁰⁰⁰⁰ sequences** (effectively infinite). The codebook is *shared* across all 6 RQ levels (not 6 separate codebooks of 16,384 each) — the paper notes this is empirically better (Lee et al. CVPR 2022's insight) but doesn't quantify the gain. The shared-codebook trick is non-obvious and important for model size: with 6 separate codebooks, the parameter count would be 6× larger. **For our project**: a shared codebook at depth 4-6 (vs 6) could give a 30-50% parameter reduction for similar quality.

9. **The "shape novelty analysis" in Fig. 8 is the cleanest demonstration in the paper that the model is *not* just retrieving training meshes.** The 3-nearest-neighbor analysis shows that for 500 generated chairs, the CD-to-nearest-training-shape has a *bimodal* distribution — some generated chairs are very close to training chairs (low CD, "interpolated"), and some are very different (high CD, "novel"). **This is exactly the multi-modal behavior we want for our H3-conditioned generation** (multiple valid completions for the same partial arch).

10. **The supplementary's "E. Additional Results" section (App. E, page 15-16) has a treasure trove of encoder-decoder ablations** that the main paper doesn't include. The Triangle Accuracy / Cross-Entropy table (Table 4 in this note) is in the supplement, not the main text. **The ablation in Table 4 is more important for our use case than the main ablations** — it tells us *which encoder/decoder/quantization design choices* are robust vs. brittle, and is the right place to start if we're adapting MeshGPT to dental. The PointNet/GAT/EdgeConv comparisons are particularly useful.

## Quote-worthy sentences

> "In contrast, artist-modeled 3D meshes are compact in representation, while maintaining sharp details with much fewer triangles." [Sec. 1]

> "We first learn a vocabulary of triangles. Triangles are encoded into latent quantized embeddings through an encoder. To encourage learned triangle embeddings to maintain local geometric and topological features, we employ a graph convolutional encoder." [Sec. 1]

> "Inspired by powerful recent advances in generative models for language, we adopt a direct sequence generation approach to synthesize triangle meshes as sequences of triangles." [Sec. 1]

> "In contrast, our method utilizes a single decoder-only network, representing triangles through learned tokens for a more streamlined generation process compared to PolyGen's separate vertex-and-face sequence approach." [Sec. 2, on PolyGen]

> "However, we observe two major challenges when using coordinates directly as tokens. First, the sequence lengths become excessively long, as each face is represented by nine values. This length does not scale well with transformer architectures, which often have limited context windows. Second, representing discrete positions of a triangle as tokens fails to capture geometric patterns effectively. This is because such a representation lacks information about neighboring triangles and does not incorporate any priors from mesh distributions." [Sec. 3.1, the motivation for the RQ codebook]

> "We observe that predicting these coordinates as discrete variables, i.e. as a probability distribution over a set of discrete values, leads to a more accurate reconstruction compared to regressing them as real values (Fig. 4). A cross-entropy loss on the discrete mesh coordinates and a commitment loss for the embeddings guides the reconstruction process." [Sec. 3.1, the discretization trick]

> "For sequence ordering, Polygen [43] suggests a convention where faces are ordered based on their lowest vertex index, followed by the next lowest, and so forth. Vertices are sorted in z-y-x order (z representing the vertical axis), progressing from lowest to highest. Within each face, indices are cyclically permuted to place the lowest index first. In our method, we also adopt this sequencing approach." [Sec. 3.1, the PolyGen ordering — relevant for the "occlusal-first" adaptation for dental]

> "Thus, we obtain geometrically rich embeddings with shorter sequence lengths, overcoming our initial challenges and paving the way for efficient mesh generation." [Sec. 3.1, end of codebook section]

> "Our method generates clean, coherent, and compact meshes, characterized by sharp edges and high fidelity." [Abstract / Fig. 1 caption]

> "Our better EMD score is more indicative of higher visual quality" [PVD paper 012, Sec. 4.2 — re-quoted here because MeshGPT's FID 18.46 is a *similar* visual-quality metric, and the implication for our project is identical]

> "MeshGPT significantly advances direct mesh generation but faces several limitations. Its autoregressive nature leads to slower sampling performance, with mesh generation times taking 30 to 90 seconds." [Sec. 4.2.1, Limitations]

> "Despite our learned tokenization approach reducing sequence lengths, which suffices for single object generation, it may not be as effective for scene-scale generation, suggesting an area for future enhancement." [Sec. 4.2.1, Limitations]

> "Our current computational resources limit us to using a GPT2-medium transformer, which is smaller than more sophisticated models like Llama2 [58]. Given that larger language models benefit from increased data and computational power, expanding these resources could significantly boost MeshGPT's performance and capabilities." [Sec. 4.2.1, Limitations — directly relevant to our v1 compute planning]

## Code/data

- **Code (official PyTorch, NCL-1.0 license — NON-COMMERCIAL):** https://github.com/audi/MeshGPT — requires torch 2.1.0 + cu118 + torch-scatter. Pretrained models and ShapeNet data are on a public Google Drive (link in the README): https://drive.google.com/drive/folders/1Gzuxn6c1pguvRWrsedmCa9xKtest8aC2
- **Data:**
  - ShapeNetV2: https://shapenet.org/ (used for all 55 categories of pretraining + 4 categories of fine-tuning). 28,980 shapes after planar decimation. Planar decimation done in Blender with angle tolerance α ∈ [1, 60], Hausdorff threshold filter, max 800 faces per shape.
  - ShapeNet PointFlow splits: not directly used by MeshGPT, but the standard splits for chair/table/bench/lamp training/test are inherited.
- **Compute (paper-reported, Sec. 3.3):** 2 A100 GPUs for 2 days (VQ stage) + 4 A100 GPUs for 5 days (GPT stage) = ~224 A100-hours per category. **Roughly equivalent in cost to LION (paper 005: ~550 V100-hours for the full pipeline)** but on newer hardware, so maybe ~$3,000-4,000 on Lambda. Significantly more expensive than PVD (paper 012: ~$50-200) and AnchorFormer (paper 011: ~$100). **MeshGPT is the *most expensive* v0 prototype candidate so far, but the only one that produces a *directly-printable* mesh**.

## For our project

1. **MAJOR ARCHITECTURAL PIVOT: MeshGPT replaces the v0 stack.** The prior v0 stack (paper 012 summary) was **PVD-AF-DiGS-FC** — PVD (point cloud DDM) + AnchorFormer (completion encoder) + DiGS (SDF lifting) + FlexiCubes (mesh extraction). MeshGPT eliminates the last two stages (DiGS and FlexiCubes) by *generating the mesh directly* — and beats the full PVD-AF-DiGS-FC stack on every metric (FID 18.46 vs PVD's 1-NNA-CD 56.26 — different metrics, but both are visual-quality proxies). **The new v0 stack is: MeshGPT (autoregressive mesh generator, conditioned on the existing arch) + AnchorFormer (existing-arch encoder for the H3 conditioning signal).** Stack name: **MGPT-AF** (read: "MeshGPT → AnchorFormer features"). Compute: ~$3,500 on Lambda. This is a substantive revision of our v0 architecture; propose to Red for review.

2. **The MeshGPT licensing caveat is real.** NCL-1.0 (Automotive Development Public Non-Commercial License v1.0) explicitly prohibits commercial use. For our v0 research pilot, this is fine. For v1 productization, we have three options: (a) **negotiate a commercial license with Audi** (slow, expensive, may not be granted), (b) **build a from-scratch reimplementation** based on the paper (3-6 months of engineering for a small team), or (c) **use MeshGPT for research validation, ship a different architecture for the product** (e.g., diffusion-on-mesh like MeshDiffusion, or our own implementation). **Decision for HK: are we planning to commercialize, or stay research-only?** The answer changes the v1 architecture. For now, plan (c) — use MeshGPT for v0 research, design the v1 architecture to be license-clean from the start.

3. **H3 conditioning is the obvious next extension, and the natural v1 architecture.** Train MeshGPT's GPT on (existing-arch-encoder-output, missing-tooth-triangle-sequence) pairs. At training time: tokenize the full arch (32 teeth, ~1000 faces per tooth, ~32,000 faces total — would need to bump the 4,608 context window to ~200,000+ which is a transformer-context problem), mask out the missing tooth's faces, train the GPT to predict the missing faces conditioned on the visible arch's tokens. At inference: tokenize the partial arch (e.g., 31 teeth, missing tooth #14), give the tokenized partial to the GPT, let it autoregressively predict the missing tooth's tokens, decode to triangles. **This is a 2-3 month project for an engineering team with dental data**. Concrete first step: extend the 4,608 context window (or use a sliding-window + caching approach for long contexts), and adapt the PolyGen face ordering to a *dental-aware* ordering (occlusal-first, then proximal, then buccal/lingual).

4. **The 128³ coordinate discretization is a real engineering concern for dental fit accuracy.** A 128³ grid on a unit bounding box gives ~0.78% resolution per axis. For a 10mm tooth, that's ~78μm per voxel — **at the limit of clinical fit accuracy (target < 50μm)**. For sub-50μm accuracy, we need 256³ (39μm/voxel) or 512³ (20μm/voxel). The encoder-decoder training dynamics change substantially with higher resolution (more vocab space, slower convergence). **Concrete action: experiment with 256³ discretization on a small tooth dataset** (e.g., 100 upper-molar CAD models from 3DTeethSeg22) and quantify the reconstruction accuracy gain. If 256³ gives clinically-useful accuracy, that's the v0. If not, we need a different coordinate representation (continuous for high-frequency regions, discrete for the rest).

5. **The 30-90 s inference time is acceptable for v0 but a blocker for v1.** For a dentist iterating on a crown design, 30-90 s is *too slow* — they'd want 1-5 s. The bottleneck is autoregressive decoding of ~4,800 indices. Three paths to fix: (a) **speculative decoding** (predict K tokens in parallel, accept the longest prefix that matches the ground-truth distribution — typically 2-3× speedup), (b) **non-autoregressive variants** (MaskGIT-style masked-token-prediction, or diffusion-on-the-codebook like LION), (c) **distillation** (train a smaller model to predict MeshGPT's outputs in a single forward pass). **Concrete action: track the LLM-inference optimization literature in 2026-2027 — speculative decoding is maturing fast, and the same tricks will apply to MeshGPT-style autoregressive mesh generation.** For v0, accept the 30-90 s and document it as a known limitation.

6. **The PolyGen face ordering is a non-obvious architectural choice that needs adaptation for dental.** PolyGen orders faces by lowest vertex index, vertices in z-y-x. This creates a *bottom-up* generation bias — early faces are at the bottom of the mesh, later faces at the top. For a dental crown, the *clinically-relevant features* are at the **top** (occlusal cusps, ridges, fossae), and the *less-relevant features* are at the **bottom** (cervical margin, cementum). **The right adaptation is occlusal-first ordering**: generate the cusps/ridges first (where the dentist cares), then propagate down to the cervical margin (where the dentist will refine by hand anyway). This is a 1-2 day change to the data-preprocessing pipeline.

7. **The MeshGPT result closes the H2 question at the *output quality* level, but re-opens it at the *output mechanism* level.** Prior to MeshGPT, the H2 story was "diffusion on points > mesh-based VAE for surface generation." MeshGPT shows that **autoregressive mesh generation > diffusion on points** on ShapeNet (FID 18.46 vs PVD's 1-NNA-CD 56.26). **H2 is now reopened**: the question is no longer "diffusion vs VAE" — it's "**autoregressive mesh vs diffusion on points/SDF/latents for printable output**." The right answer depends on the *use case*: if you need a directly-printable mesh with sharp features, MeshGPT wins. If you need multi-modal sampling of completions, PVD/LION/Diffusion-SDF win. **For our project**: we need *both* — multi-modal completions *and* printable mesh output. The right v1 architecture is a **hybrid** that uses diffusion for multi-modal sampling and MeshGPT-style autoregression for mesh output (e.g., LION's VAE-then-DDM, but the DDM operates over a MeshGPT-style codebook instead of a point-cloud latent). This is a research-level idea, not a v0 prototype.

8. **The "shape novelty" analysis in Fig. 8 is the right validation protocol for our v0.** Train on 1,000 synthetic upper-molars, generate 100 new molars, compute Chamfer distance to nearest training neighbor. If the distribution is bimodal (some close, some far), the model is interpolating. If it's unimodal at high CD, the model is over-generating novel shapes. If it's unimodal at low CD, the model is memorizing. **Concrete action: build a `scripts/measure_novelty.py` that runs the Fig. 8 analysis on our v0 output, and use it as a sanity check before declaring v0 ready for clinical evaluation.**

9. **The user-study methodology (49 participants, 784 responses, paired comparisons) is the right validation protocol for our clinical evaluation.** When we have a v0 prototype, we shouldn't just compute CD/FID — we should run a paired-comparison user study with 5-10 dentists, showing them (a) MeshGPT-generated crowns, (b) PVD-AF-DiGS-FC-generated crowns, (c) hand-crafted CAD crowns, and asking them to rate on (i) occlusal surface quality, (ii) intaglio fit, (iii) marginal integrity, (iv) overall clinical acceptability. The MeshGPT user study (Sec. 4.2 + App. C) is the canonical reference for this methodology. **Concrete action: design the clinical evaluation protocol based on App. C, get IRB approval, run the study in 2026-Q4.** Budget: $5-10k for dentist participant compensation.

10. **The encoder-decoder ablation in Table 4 is the most actionable table in the paper for our use case.** The PointNet/GAT/EdgeConv/SAGEConv comparison (98.49% with SAGEConv, 88.73/86.14/91.23 with the others) tells us that **SAGEConv is the right encoder for face graphs** — and this generalizes to dental. If we want to *adapt* MeshGPT to dental (rather than re-implement from scratch), the encoder-decoder stage is the easiest to retrain: collect a dataset of tooth meshes, run the MeshGPT VQ pipeline on it, and the new codebook will be tooth-specific. The transformer stage is harder to retrain (longer training time, more data needed), but the encoder-decoder can be done in a week. **Concrete action: pilot the VQ stage on 3DTeethSeg22's 23,999 FDI-labeled teeth first, before committing to the full MeshGPT training.**

11. **The MeshGPT codebase has a clear dependency on a 2023-era PyTorch stack** (torch 2.1.0 + cu118 + torch-scatter + Blender for decimation + pytorch-lightning for training). It should run on a modern CUDA GPU (A100, H100) without major changes, but the inference scripts require ~10GB GPU memory. **For our pilot**: 1 A100 40GB instance on Lambda, ~$3/hr, ~$200 for inference smoke tests. **For full training**: 4 A100s for 5 days, ~$3,500. **Total v0 MeshGPT pilot budget: ~$3,700** (more expensive than PVD/LION/AnchorFormer, but the only one that produces directly-printable meshes).

12. **Open question for HK: should the v0 architecture be MeshGPT-only, or a MeshGPT+DiGS hybrid?** Two architectures to consider:
    - **MGPT-only** (paper 013): MeshGPT generates the entire arch (existing + missing) end-to-end, conditioned on the partial arch via the H3 mechanism. One model, one forward pass, one output. **Risk**: a single failure mode (e.g., the model misses the existing teeth's geometry) could invalidate the entire output.
    - **MGPT + DiGS** (paper 013 + paper 003): MeshGPT generates the missing tooth's *mesh* conditioned on the existing arch's *SDF* (from DiGS). Two models, two forward passes, but each model is simpler and more robust. **Risk**: the conditioning interface (mesh in, SDF out, SDF in, mesh out) is more complex.
    - **My recommendation**: start with MGPT-only for v0, and add DiGS as a *fallback* for cases where the autoregressive generation fails. The v1 architecture is the hybrid.

13. **Next paper to read: MeshDiffusion (Liu et al., 2023) — the *diffusion* version of MeshGPT.** MeshDiffusion uses score-based diffusion on the mesh vertices and face connectivity, rather than autoregressive sequence modeling. This would close the remaining H2 question (autoregressive mesh vs diffusion mesh), and would be the natural v1 alternative to MeshGPT if our v0 MeshGPT pilot shows that the autoregressive approach is too slow or too brittle. arXiv:2303.08133. Alternatively: **CIGS (CVPR 2024)** for the latest 2024 SoTA diffusion-on-implicit-field, or **MeshAnything (Chen et al., 2024)** for a follow-up to MeshGPT that conditions on a shape prior (the "composable" version of MeshGPT — directly relevant to our H3 conditioning).

---

*Scholar's note: MeshGPT is the missing v0 architecture we've been looking for. It's the only paper in the reading list that produces a *directly-printable triangle mesh* with *artisan-level triangulation* and *better-than-FlexiCubes* quality. The architectural pivot from PVD-AF-DiGS-FC to MGPT-AF is a substantive revision of our v0 plan, but it's the right one — and it eliminates two of the four models in the v0 stack (DiGS, FlexiCubes), reducing engineering complexity by 50%. The two main caveats are (a) the non-commercial license (NCL-1.0) which blocks productization, and (b) the 30-90 s inference time which limits v1 productization. Both are solvable, but they shape the v1 roadmap. Action item for Red: validate the MeshGPT inference pipeline on a single ShapeNet chair (smoke test, ~2 hours), then pilot the dental adaptation (VQ stage on 3DTeethSeg22, ~1 week). Action item for HK: review the architectural pivot from PVD-AF-DiGS-FC to MGPT-AF, decide on the commercial-license strategy, and approve the v0 compute budget bump from $2,200 to $3,700. Action item for Mauve: design the clinical evaluation user study based on App. C of the MeshGPT paper (paired comparisons with 5-10 dentists).*

## Reference

```bibtex
@inproceedings{siddiqui2024meshgpt,
  title={MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers},
  author={Siddiqui, Yawar and Alliegro, Antonio and Artemov, Alexey and Tommasi, Tatiana and Sirigatti, Daniele and Rosov, Vladislav and Dai, Angela and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}

@article{siddiqui2023meshgpt_arxiv,
  title={MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers},
  author={Siddiqui, Yawar and Alliegro, Antonio and Artemov, Alexey and Tommasi, Tatiana and Sirigatti, Daniele and Rosov, Vladislav and Dai, Angela and Nie{\ss}ner, Matthias},
  journal={arXiv preprint arXiv:2311.15475},
  year={2023}
}
```
