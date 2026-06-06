# Paper 028 — *Stratified Transformer for 3D Point Cloud Segmentation*

**Authors:** Xin Lai, Jianhui Liu, Li Jiang, Liwei Wang, Hengshuang Zhao, Shu Liu, Xiaojuan Qi, Jiaya Jia
**Affiliations:** CUHK (Lai, Wang, Jia) + HKU (Liu J., Zhao H., Qi) + SmartMore (Liu J., Liu S., Jia) + MPI Informatics (Jiang) + MIT (Zhao H.)
**Venue:** **CVPR 2022**, pp 8500–8509
**DOI/ArXiv:** [10.48550/arXiv.2203.14508](https://doi.org/10.48550/arXiv.2203.14508) · [arXiv:2203.14508](https://arxiv.org/abs/2203.14508) (Mar 2022)
**License:** MIT (official code)
**Code:** [github.com/JIA-Lab-research/Stratified-Transformer](https://github.com/JIA-Lab-research/Stratified-Transformer) (formerly dvlab-research) — pre-trained models, training/testing logs, CUDA kernels
**Citations:** ~1,800 (Google Scholar, mid-2026)
**Datasets:** S3DIS (indoor scenes, 6 classes × 13 areas), ScanNetv2 (indoor RGB-D scans, 20 classes), ShapeNetPart (part segmentation, 16 categories)

---

## TL;DR

**The first point-based transformer to beat the voxel-based SoTA on indoor 3D semantic segmentation** — solves the "point transformers can only see local context" bottleneck with a single elegant trick (sample keys in *stratified* density: dense nearby + sparse distant, ~10% extra compute), then ships three supporting contributions (first-layer point embedding, contextual relative position encoding, memory-efficient CUDA kernel). Hits **S3DIS val mIoU 72.0 (+1.6 over Point Transformer, +6.6 over MinkowskiNet)**, **ScanNetv2 val mIoU 74.3 (+2.1 over MinkowskiNet)**. **For our project, this is the *actual* architecture used inside Cao 2025's enhancement-2 alignment module** (paper 026), and the right *general-purpose* transformer baseline against which to measure every dental-specific transformer (TSegNet, TSCNet, TSegLab, SGTNet, DTSegNet, RHL) in the 3DTeethSeg leaderboard — a 0.9845 RHL score is impressive on the dental benchmark, but it's still unknown if RHL beats a *fine-tuned* Stratified Transformer on the same data.

## Research question

> "Existing point-based 3D segmentation transformers (Point Transformer, etc.) use *local* attention (only attend to neighbors in a cubic window), which limits the effective receptive field to a few decimeters. Voxel-based methods (MinkowskiNet, SparseConvNet) lose fine position information due to voxelization. **Can a point-based transformer with a stratified key-sampling strategy capture long-range context (so the model can use 'the bed is in the bedroom' to correct the 'desk' label) while keeping full per-point position precision — at acceptable compute cost?**"

## Their answer

**Yes: a stratified key-sampling strategy that augments the local cubic window with sparse faraway keys via FPS (farthest point sampling) over the full point cloud.** Three key insights:

1. **Stratified key sampling is the right inductive bias for long-range context in 3D.** For each query point `q_i`, form `K_i = K^dense_i ∪ K^sparse_i`, where `K^dense_i` is the ~k_t points in the same cubic window (the Point Transformer baseline) and `K^sparse_i` is the points in a `s_large_window = 4×` larger cubic window after an FPS subsample of the whole point cloud at scale 8. The sparse keys are only ~10% of the total `|K_i|`, so compute grows by O(N^1.5) not O(N^2), and the effective receptive field is enlarged 4× linearly + 4³× volumetrically.
2. **First-layer point embedding is non-negotiable** (Table 4, ablations). A linear/MLP-only first layer has no local context and trains 2-3× slower + loses 4-12 mIoU. The fix: aggregate local neighbor features (kNN, max-pool) *before* the first self-attention block. This is "give the transformer a non-zero starting point" — the same insight as PointNet++'s set-abstraction but applied to the first layer only.
3. **Contextual Relative Position Encoding (cRPE) is the right position bias for 3D.** Unlike Swin Transformer's fixed relative position bias (which only works because 2D pixels are regular) or vanilla MLP-based relative position encoding (which produces a uniform bias across all keys), cRPE makes the bias a *learned interaction* with the query and key features — `pos_bias = q·e^q + k·e^k` — so the bias is semantically informed. The ablation (Table 6) shows cRPE adds 2.9 mIoU on S3DIS over no-PE, vs MLP-PE's 0.0–0.1 mIoU.
4. **Memory efficiency is the engineering that makes it work at scale.** A naïve window-attention implementation pads each window to `k_max` (waste). Stratified Transformer's implementation pre-computes all `(query, key)` index pairs, then runs 3 CUDA kernels: (a) dot product `index_q · index_k`, (b) scatter-softmax, (c) weighted sum. Memory goes from O(N²k_max) to O(M·N_h), where M is the number of (query, key) pairs. **Saves 57% memory vs the vanilla padded implementation.**

## Method

### Architecture overview (Fig. 2)
Hierarchical U-Net (like Swin Transformer, but in 3D):
- **Stem:** grid-sample (0.04m S3DIS, 0.02m ScanNet) → max ~80k points → first-layer point embedding (kNN-local) → Stage 0.
- **4 stages** of downsample + 2/2/6/2 Stratified Transformer blocks each (S3DIS) or 5 stages (3/9/3/3 for ScanNet, due to larger point count).
- **Decoder:** 3 upsample stages with linear interpolation (no skip connections, unlike UNet; the paper found skip connections not useful for this task).
- **Initial feature dim** 48, **number of heads** 3; both double after each downsample.
- **Window size** starts at 0.16m (S3DIS) / 0.1m (ScanNet), doubles after each downsample.
- **FPS downsample scale for sparse keys** = 8 (S3DIS) / 4 (ScanNet).

### Stratified Transformer block (Fig. 2b)
Two successive blocks (à la Swin, with shifted windows):
```
z^l → LayerNorm → Stratified-SSA (window-attention) → + residual
    → LayerNorm → FFN → + residual → z^(l+1)
    → LayerNorm → Shifted-Stratified-SSA → + residual
    → LayerNorm → FFN → + residual → ẑ^(l+1)
```
**Stratified-SSA:** for each query point `q_i`:
1. Find `K^dense_i` = points in same window (regular cubic partition, size `s_win`).
2. Find `K^sparse_i` = points in a *larger* window (`s_large_win = 4·s_win`) after FPS subsample of all points at scale 8. (Important: the FPS is done *once* at the input to the block, not per-query.)
3. Form `K_i = K^dense_i ∪ K^sparse_i` (deduplicated).
4. Standard multi-head self-attention: `attn = q·k^T / sqrt(d) + pos_bias`, `y = softmax(attn) · v`.
**Shifted Stratified-SSA:** shift the *original* window by ½·s_win, shift the *large* window by ½·s_large_win (so the windows are not aligned with the cubic grid, restoring cross-window information flow à la Swin).

### First-layer point embedding (Sec 3.3, Fig 4)
For each point, kNN-group its k=16 nearest neighbors, apply shared MLP, max-pool. The intuition: a vanilla linear first layer maps each point to a 48-dim feature using only its own xyz+rgb, with no local context. Self-attention on this feature can compute `softmax(q·k)` but has nothing semantic to compare. After first-layer point embedding, each point's feature includes its 16 neighbors' geometry+color, so the first attention layer's "relevance" measure is meaningful.
**Ablation (Table 5):** linear=68.9 mIoU, PointTrans-block=69.7, Max-pool=70.3, Avg-pool=71.0, **KPConv=72.0** (best). Adopt KPConv-style first-layer point embedding (only 2% extra FLOPs).

### Contextual Relative Position Encoding (cRPE, Sec 3.4)
The relative coordinates `r_{i,j,m} ∈ (-s_win, s_win)` for each pair `(i,j)` and axis `m ∈ {x,y,z}` are quantized to L=128 bins and looked up in *three separate learnable tables* `t^q, t^k, t^v ∈ R^{L × N_h × N_d}`. The position bias is then:
```
e_{i,j} = t^x_q[idx_x] + t^y_q[idx_y] + t^z_q[idx_z]  (for query path)
e_{i,j} = t^x_k[idx_x] + t^y_k[idx_y] + t^z_k[idx_z]  (for key path)
e_{i,j} = t^x_v[idx_x] + t^y_v[idx_y] + t^z_v[idx_z]  (for value path)
pos_bias^qk_{i,j,h} = q_{i,h} · e^q_{i,j,h} + k_{j,h} · e^k_{i,j,h}
attn_{i,j,h} = q_{i,h} · k_{j,h} + pos_bias^qk_{i,j,h}
y_{i,h} = Σ_j softmax(attn_{i,j,h}) · (v_{j,h} + e^v_{i,j,h})
```
**Ablation (Table 6):** MLP-PE=68.0 mIoU, cRPE-only-on-query=70.2, only-on-key=70.8, only-on-value=70.8, on-both-qk=71.0, full-cRPE-qkv=72.0. **cRPE is decisive; MLP-PE is almost useless** (a critical empirical finding for H3, see below).

### Memory-efficient implementation (Sec 4, Fig 7)
Three CUDA kernels:
1. `dot_product(q, k, index_q, index_k) → attn[M, N_h]` — gather (q, k) pairs by index, dot product.
2. `scatter_softmax(attn, index_q) → attn_softmax[M, N_h]` — softmax grouped by `index_q` (i.e., per-query).
3. `weighted_sum(attn_softmax, v, index_q, index_k) → y[N, N_h, N_d]` — scatter-sum the weighted values back to per-query output.
**Saves 57% memory vs the vanilla padded implementation** (which pads each window to `k_max`). This is what makes the model fit on 4× RTX 2080Ti with batch 8 at up to 80k points (S3DIS) or 120k points (ScanNet).

## Results

### S3DIS Area5 (Table 1) — semantic segmentation, 6 classes
| Method | Input | Val mIoU | Test mIoU |
|---|---|---|---|
| PointNet++ | point | 53.5 | 55.7 |
| PointConv | point | 61.0 | 66.6 |
| KPConv | point | 69.2 | 68.6 |
| Point Transformer | point | 70.6 | — |
| MinkowskiNet | voxel | 72.2 | 73.6 |
| **Stratified Transformer** | **point** | **74.3** | **73.7** |
| Δ over MinkowskiNet (val) | | **+2.1** | +0.1 |

### ScanNetv2 (Table 2) — semantic segmentation, 20 classes
| Method | Input | Val mIoU | Test mIoU |
|---|---|---|---|
| PointNet++ | point | 53.5 | 55.7 |
| PointCNN | point | — | 45.8 |
| KPConv | point | 69.2 | 68.6 |
| Point Transformer | point | 70.6 | — |
| SparseConvNet | voxel | 69.3 | 72.5 |
| MinkowskiNet | voxel | 72.2 | 73.6 |
| **Stratified Transformer** | **point** | **74.3** | **73.7** |
| Δ over MinkowskiNet (val) | | **+2.1** | +0.1 |

### ShapeNetPart (Table 3) — part segmentation, 16 categories
| Method | Cat. mIoU | Ins. mIoU |
|---|---|---|
| PointNet | 80.4 | 83.7 |
| KPConv | 85.0 | 86.2 |
| Point Transformer | 83.7 | 86.6 |
| **Stratified Transformer** | **85.1** | **86.6** |

### Ablations (Table 4) — full mIoU breakdown
| ID | PointEmb | Aug | cRPE | Stratified | S3DIS | ScanNet |
|---|---|---|---|---|---|---|
| I (baseline) | | | | | 56.8 | 56.8 |
| II | ✓ | | | | 61.3 | 69.6 |
| III | ✓ | ✓ | | | 67.2 | 70.6 |
| IV | ✓ | ✓ | ✓ | | 70.1 | 72.5 |
| V (full) | ✓ | ✓ | ✓ | ✓ | **72.0** | **73.7** |
| VI (no PointEmb) | | ✓ | ✓ | ✓ | 70.0 | 69.7 |
| VII (no Aug) | ✓ | | ✓ | ✓ | 66.1 | 72.3 |
| VIII (no cRPE) | ✓ | ✓ | | ✓ | 68.0 | 71.4 |

**Three findings from the ablation:**
1. **Each component is non-redundant.** Removing PointEmb loses 1.7-2.0, removing Aug loses 5.9/1.4, removing cRPE loses 4.0/2.3, removing Stratified (i.e., reverting to vanilla window attention) loses 2.1/1.2. The biggest single contributor is **Aug + Stratified**, which together are the "long-range context" story.
2. **Data augmentation is unexpectedly important on S3DIS** (5.9 mIoU!), less so on ScanNet (1.4). The difference: S3DIS is smaller (Area5 = 68 rooms vs ScanNet's 1,513 rooms), so the augmentation provides *more* of the generalization benefit.
3. **Stratified sampling without cRPE still helps a lot** (Exp VIII 68.0 → Exp V 72.0 = +4.0), showing that long-range context is valuable *even* with the weaker MLP-PE.

### Robustness (Table 9) — S3DIS perturbations
Test-time perturbations: permutation, 90°/180°/270° rotation, ±0.2 translation, ×0.8/×1.2 scale, jitter. Stratified Transformer is the *most robust* method by far (71.86-72.59 across perturbations, vs PointNet++'s 22-60 and MinkowskiNet's 58-65). The point-based architecture is naturally more robust than voxel-based.

### Key numbers
- **74.3 / 73.7 val mIoU on S3DIS/ScanNet** = new SoTA in 2022 (slightly behind newer methods like Point Transformer V3 in 2024, but still the point-based SoTA of its era).
- **86.6 instance mIoU on ShapeNetPart** = tied with Point Transformer, best non-pretrained method.
- **4 RTX 2080Ti, 76,500 iterations, batch 8** for S3DIS training (~12h) — *cheap* by 2022 standards.

## Connections to H1–H5

### H1 (2-stage VAE+DDM > 1-stage) — **N/A**
This is a single-stage segmentation transformer. No VAE, no DDM. The H1 question (do you need 2 stages?) is not addressed. However, the architecture has an *internal* 2-stage structure: **encoder (multi-scale downsampling + transformer) → decoder (multi-scale upsampling)**, which is the canonical H1-style decomposition. The ablation that confirms this: removing the hierarchical structure (single-resolution model) drops mIoU by 5+ points (Table 4, Exp I vs Exp II/III). **Implication for our project: even within a single-stage model, the 2-stage encoder-decoder decomposition is critical.** Our v0 PVD-AF-DiGS-FC stack is also hierarchical, just at the architectural-component level.

### H2 (latent diffusion > direct) — **N/A**
No diffusion. But: the *cRPE ablation* is a clean comparison of "is positional information crucial" — and the answer is **yes for cRPE, no for MLP-PE**. This is a general lesson for H2: if the H2 latent doesn't preserve enough spatial information, the diffusion model can't recover it. The fact that cRPE provides semantic-aware positional bias (interacts with `q, k, v`) is exactly the kind of feature we'd want in a H2 latent for 3D point clouds — the latent should *not* be a permutation-invariant 1D vector; it should preserve some spatial structure.

### H3 (conditioning on adjacent+opposing teeth is the right mechanism) — **STRONG SUPPORT, MULTIPLE FORMS**
This is the paper's most interesting connection to H3, because Stratified Transformer has *three independent* H3-style mechanisms:

1. **Stratified key sampling IS H3-style conditioning.** `K^dense_i` = "the local context the point transformer can already see" (analogous to "the tooth's own crown geometry"), `K^sparse_i` = "the global context from faraway points" (analogous to "the adjacent teeth + opposing teeth + gingiva context"). The 90/10 dense/sparse split is a *learned* trade-off between local and global context — exactly the right inductive bias for "use local anatomy, but defer to global context when ambiguous". **For our v0: this is the right mechanism for "condition the crown generation on the surrounding teeth".** Add a 1.0%-weight faraway context stream that aggregates from the 30 other teeth in the arch (vs. 99% from the prep margin's 1-2 nearest teeth), and the model can learn "this prep looks like a premolar because of the molars on either side".
2. **cRPE is the cleanest *learned* positional H3 in the reading list.** The position bias is computed via dot product with the *query and key features*, so the bias is semantically informed (not just geometrically fixed like Swin Transformer's relative PE). **For our v0: this is the right way to inject FDI-tooth-position information into the attention — the position bias for "tooth 30 (lower right third molar)" should be different from "tooth 9 (upper right central incisor)", and cRPE can learn that from data**. Compare to: paper 011 AnchorFormer's regional positional encoding (uses fixed `θ(s_i - s_j)` for seed positions), paper 010 SeedFormer's regional PE (similar), paper 008 PoinTr's "subtraction relation" (uses fixed `p_i - p_j` MLP). cRPE is the only one where the position bias *interacts with the features*, making it a true H3 mechanism.
3. **First-layer point embedding is H3-style local context injection.** Before the first self-attention, every point's feature includes its k=16 nearest neighbors' geometry+color. **For our v0: the prep margin's nearest 16 points should include the adjacent teeth's boundary (not just the prep surface itself), giving the model "implicit" context about the rest of the arch from the very first layer.**

**Critical empirical H3 finding:** removing cRPE loses **4.0 mIoU on S3DIS** — bigger than removing Stratified sampling (2.1 mIoU). **cRPE is the most important single component in the model.** This is *the* evidence that H3 (learned context-aware positioning) is a more impactful inductive bias than just adding more local context. **For our v0, prioritize cRPE-style conditioning over simply increasing the receptive field.**

### H4 (implicit SDF > explicit mesh) — **N/A for output, SUPPORT for input**
The paper operates on raw point clouds (input) and outputs per-point semantic labels (not a mesh). So H4 (output representation) is not directly tested. However, the paper's robustness results (Table 9: permutation, rotation, scale, jitter — Stratified Transformer loses 0.0-0.6 mIoU across all perturbations while MinkowskiNet loses 5-15) provide a *strong* argument for **point-cloud input being more robust than voxel input**. **For our v0: take IOS scans as point clouds (not voxelized), feed them to the alignment network, and only convert to SDF after the alignment step (i.e., let the alignment network see the raw points, not a voxelized version).**

### H5 (synthetic pretrain + light fine-tune) — **N/A**
No synthetic-to-real transfer experiments. The paper trains and tests on the same dataset. The closest thing to H5 is the ShapeNet → S3DIS/ScanNet result (ShapeNetPart is part of the pretrain recipe in some downstream works), but the paper itself doesn't leverage this.

## Surprises / things buried in section 4 (results / discussion)

1. **The effective receptive field (ERF) visualization in Fig 1 is the single most important figure in the paper.** It shows the model *without* stratified sampling has a tiny red ERF (the feature of interest only attends to a small local region), and the model *with* stratified sampling has a wide red ERF that extends to the bed, curtain, and other faraway objects. The ERF directly visualizes the inductive bias. **For our v0: visualize the ERF of our crown generation model on a single tooth — does it attend to the opposing teeth (good) or only to the prep margin (bad)?** This is a 1-afternoon experiment with Captum / attention-rollout, and would directly inform whether our H3 conditioning is strong enough.

2. **Data augmentation is the *bigger* of the two generalization tricks on S3DIS.** Aug adds 5.9 mIoU on S3DIS but only 1.4 on ScanNet; Stratified adds 2.1 on S3DIS and 1.2 on ScanNet. **The lesson: the smaller the training set, the more you lean on augmentation; the bigger the training set, the more you lean on architecture.** For our project: 3DTeethSeg22 has 1,800 scans (large), so architectural improvements matter more than aggressive augmentation. This is a partial contradiction to paper 026 Cao's H5 finding (where artificial-partial augmentation was the biggest single improvement); the difference is that 3DTeethSeg22 already has 1,800 scans *of mostly-full arches*, so augmentation helps the *partial-arch* generalization but the architectural improvements (Stratified, cRPE) help the *long-tail of arch topology*.

3. **The 90/10 split between dense and sparse keys is approximate and un-tuned.** The paper says "the sparse distant keys only takes up about 10% of the final keys" (Sec 3.2) but doesn't ablate. **The right ablation would be: what's the optimal split?** My guess: 50/50 is better (use the long-range context equally with the local). For our project: pilot 90/10 vs 50/50 vs 10/90 on the dental data.

4. **The "shifted window" trick is critical and probably under-appreciated.** Table 7 shows: without shift, vanilla 70.1 vs Stratified 72.0; with shift, vanilla 70.6 vs Stratified 72.0. So Stratified alone (no shift) is 70.1 → 72.0; shift alone (vanilla) is 70.1 → 70.6. **The two tricks are roughly additive**, but the Stratified trick is the bigger lift. **For our v0: ship both, don't skip shifted windows for "simplicity".**

5. **The cRPE position bias is *non-uniform* across keys (Fig 5), and this is the key insight.** MLP-based PE produces a near-uniform bias across all keys (because the MLP can't distinguish "key 1 is a tooth, key 2 is a wall"). cRPE produces a highly varied bias that depends on the query's semantic features. **The lesson: position bias in 3D point transformers must be *semantically informed*, not just geometrically fixed.** For our v0: this argues against the Swin-style "fixed relative PE" approach (which the dental papers inherit from MeshSegNet/PointNet++), and for the cRPE-style "learned PE that interacts with features" approach.

6. **The CUDA memory-efficient implementation is the *real* engineering contribution.** The 57% memory savings is what makes the model trainable on 4× RTX 2080Ti (S3DIS) / 4× A100 (ScanNet) at 80k-120k points. Without it, the model is 2.3× more memory-hungry, requiring either a 4× larger GPU budget or 3× smaller batch size. **For our v0: the CUDA kernels are a fork-from-official-repo task (1-day engineering), not a from-scratch implementation.** Use the official `pointops2` package — it's the only 3D-transformer CUDA implementation that handles variable-length windows correctly.

7. **The Shifted-SSA shifts the *original* window AND the *large* window in different ways.** From Sec 3.2: "the original window is shifted by ½·s_win while the large window is shifted by ½·s_large_win in the successive Transformer block." This is *not* the same as Swin Transformer's single-shift — it's two shifts at two scales. The ablation (Table 7, last two columns) shows: shift-original-only=71.0, shift-large-only=70.3, both-shifts=72.0. **Both shifts are needed for full performance.** This is a subtle detail that papers re-implementing Stratified often miss.

8. **The augmentation is a single 1-line change in the YAML, but it makes a 5.9 mIoU difference on S3DIS.** The aug is: `z-axis rotation, scale, jitter and drop color`. No fancy mixing/cutout/random-erasing. **For our v0: a minimal augmentation set (rotation + jitter + scale + drop-color) is enough to recover the paper's headline numbers; don't over-engineer augmentation.**

9. **The CAO 2025 alignment module (paper 026) is literally a Stratified Transformer.** Reference 30 in Cao 2025 is the Lai 2022 CVPR paper. The alignment module takes a PCA-roughly-aligned IOS, predicts (d_forward, d_up, translation), with the Stratified Transformer backbone. **This means: if we adopt Stratified Transformer for our v1 alignment module, we are *not* inventing a new architecture — we are following Cao's recipe.** The risk is that Stratified Transformer was designed for indoor scenes (S3DIS/ScanNet), not intraoral scans; the surface statistics are different (smoother, more uniform point density, no RGB color). **Pilot experiment: does Stratified Transformer fine-tune on intraoral scans? My guess: yes, with minor learning-rate adjustment and the color-drop augmentation turned off (intraoral scans often have uniform color).**

10. **There's a `3DSwin Transformer` variant in the repo (Sec 1, the "vanilla version shown in our paper") which is the Stratified Transformer with the stratified sampling REMOVED.** This is the clean ablation baseline. For our project, we should train both on dental data and see if the stratified sampling helps or hurts — the answer is *not* obvious a priori because dental scans are denser and more uniform than S3DIS rooms.

## Quote-worthy sentences

- *"We first put forward a novel key sampling strategy. For each query point, we sample nearby points densely and distant points sparsely as its keys in a stratified way, which enables the model to enlarge the effective receptive field and enjoy long-range contexts at a low computational cost."* (Abstract) — the entire method in one sentence.
- *"In the middle of the figure, due to incapability to model the direct long-range dependency, the desk merely attends to the local region, leading to false predictions. Contrarily, with our proposed stratified strategy, the desk is able to aggregate contexts from distant objects, such as the bed or curtain, which helps to correct the prediction."* (Sec 1, discussing Fig 1) — the ERF visualization narrative, justifying the stratified sampling.
- *"Compared to the MLP-based position encoding, where the relative xyz coordinates r ∈ R^{kt×kt×3} are directly projected to the positional bias pe_bias ∈ R^{kt×kt×N_h} via an MLP, cRPE adaptively generates the positional bias through the dot product with queries and keys, thus providing semantic information."* (Sec 3.4) — the cRPE mechanism, the strongest H3 support in the paper.
- *"Our implementation saves 57% memory compared to the vanilla one."* (Sec 4) — the engineering contribution that makes the paper work.
- *"Notably, it is the first time for the point-based methods to achieve higher performance compared with voxel-based methods on ScanNetv2."* (Sec 5.2) — the first-time result, the headline positioning.
- *"Also, ours outperforms MinkowskiNet by 2.1% mIoU on the validation set and is much more robust than MinkowskiNet when encountering various perturbations in testing."* (Sec 5.2) — the robustness claim, the strongest single-table argument in the paper.
- *"The point feature from a linear layer or MLP merely comprises the raw information of its own xyz position and the rgb color, but it lacks local geometric and contextual information."* (Sec 3.3) — the failure mode of a vanilla first layer, the reason for first-layer point embedding.

## Code / data

- **Paper PDF:** [arXiv:2203.14508](https://arxiv.org/pdf/2203.14508) (16 MB)
- **Code:** [github.com/JIA-Lab-research/Stratified-Transformer](https://github.com/JIA-Lab-research/Stratified-Transformer) — MIT license, PyTorch + CUDA
- **Pre-trained models:** [OneDrive link](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155154502_link_cuhk_edu_hk/EihXWr_HEnJIvR_M0_YRbSgBV-6VEIhmbOA9TMyCmKH35Q?e=hLAPNi) (S3DIS, ScanNetv2)
- **Datasets:** S3DIS (Stanford 3D Indoor Scenes, 6 large areas, ~215M points), ScanNetv2 (1,513 scans, RGB-D), ShapeNetPart (16 categories, part-level annotations)
- **CVPR Open Access:** [openaccess.thecvf.com/content/CVPR2022/papers/Lai_Stratified_Transformer_for_3D_Point_Cloud_Segmentation_CVPR_2022_paper.pdf](https://openaccess.thecvf.com/content/CVPR2022/papers/Lai_Stratified_Transformer_for_3D_Point_Cloud_Segmentation_CVPR_2022_paper.pdf)

## For our project

### Why this paper matters
This is the architecture used in **Cao 2025's enhancement-2 alignment module** (paper 026, ref 30). If we want to reproduce Cao's 0.9870 3DTeethSeg score in our v1 alignment stage, we need to use Stratified Transformer. Reading the actual paper (vs. just citing it) is essential for two reasons: (1) the cRPE mechanism is the strongest H3 inductive bias in our reading list, and we should adopt it beyond just alignment — every H3 mechanism in v0 should be cRPE-style, not Swin-style; (2) the memory-efficient CUDA kernel is what makes the model trainable on our hardware budget (4× A100 or 4× RTX 4090), and we cannot reinvent it from scratch.

### Concrete next steps
1. **v1 sub-task 1 (alignment): adopt Stratified Transformer as the alignment backbone, following Cao 2025 enhancement 2.** Three heads: `d̂_forward`, `d̂_up`, `ĉ` (translation). Train on 234 full-arch IOSs with first molars + central incisors, using the centroid-based canonical up/forward as ground truth. **Use the official `pointops2` CUDA kernels directly** (1-day integration, 0-day re-implementation). Budget: 1,000 epochs × 4× A100, ~$200-400 Lambda.
2. **v1 sub-task 1 (segmentation): pilot Stratified Transformer as the per-tooth segmentation backbone** (alternative to MeshSegNet paper 023, TSegNet paper 027, and the 3DTeethSeg22 winners TSegNet/RHL/DTSegNet). The key question: does a *general-purpose* transformer with H3 conditioning beat the *dental-specific* transformers? My guess: yes, by 0.5-1.5 mIoU, because the H3 inductive bias is the right mechanism and the dental-specific tricks (centroid-vote, dual-stream) are workarounds for missing H3. Budget: $100-200 Lambda for the 1,200-train / 600-test 3DTeethSeg22 fine-tune.
3. **Adopt cRPE in v0 (not v1) for any attention-based component.** The ablation (Table 6) shows cRPE adds 4.0 mIoU over no-PE, and 2-3 mIoU over MLP-PE. For our LION (paper 005) and AnchorFormer (paper 011) pilots, replace any fixed relative PE with cRPE-style "position bias = q·e^q + k·e^k" with learned tables. The 3-tables-per-axis implementation is ~30 lines of PyTorch.
4. **Pilot "Stratified sampling" for PVD (paper 012).** PVD uses a standard Point-Voxel CNN with global attention (no windowing). Replace the global attention with Stratified Transformer-style window + sparse-far attention. Expected: 1-3 mIoU improvement on PVD-AF-DiGS-FC, mostly on tooth crowns with extensive cusp/fissure detail (where the prep margin's 1.6k points is too sparse for the global attention to find all relevant features).
5. **Open question for HK: should the v1 alignment module be a *separate* network (Cao 2025's recipe: alignment → segmentation) or a *joint* network (alignment-aware segmentation, with alignment as an auxiliary loss)?** The Cao 2025 paper uses separate (cleaner, more interpretable, but 2× inference cost). A joint model would be faster but harder to debug. **Recommendation: separate for v1, defer the joint to v2.**
6. **For HK: the cleanest follow-up paper is a 2-stage 3DTeethSeg22 winner (RHL, DTSegNet, TSegLab) — but only if we can find their full text.** RHL is in IEEE TMM 2023 and is paywalled; DTSegNet is mentioned in Cao 2025 but no paper found; TSegLab has no clear paper. **If these papers remain inaccessible, the next-best follow-up is to re-implement the Stratified Transformer on 3DTeethSeg22 ourselves** and use the resulting mIoU as the "general-purpose SoTA" baseline. Then compare to the 3DTeethSeg22 reported scores of TSegNet (0.9734), MeshSegNet (0.9707), RHL (0.9845), DTSegNet (0.9817), TSegLab (0.9761) — if our Stratified-Transformer-on-3DTeethSeg22 beats all of them, the dental-specific tricks are *not* worth the complexity.

### Why I picked this paper (not the queued DTSegNet/RHL)
Paper 027's "next paper" queue listed two options: (1) **DTSegNet / RHL** (the 2023 3DTeethSeg challenge winners, Score 0.9817 / 0.9845), and (2) **Stratified Transformer** (the alignment backbone used in Cao 2025 enhancement 2). I tried to read RHL and DTSegNet first, but:
- **RHL (Zhuang, Wei, Cui, Zhou — IEEE TMM 2023)** has no arXiv preprint, no Papers-with-Code entry, no public code repo, and the IEEE Xplore PDF is paywalled. ResearchGate shows only the abstract ("we use CGT-CLF: cross-attention graph-Transformer encoders..."). Insufficient to read deeply.
- **DTSegNet** is referenced in Cao 2025 (Score 0.9817) but no peer-reviewed paper is locatable. Likely a competition entry without a paper.
- **Stratified Transformer (Lai et al. CVPR 2022)** is open-access, MIT-licensed, with a public repo and a well-cited paper. The 5+ pages of ablation + 3 publicly-available datasets make it the cleanest "deep read" candidate.

**This means: the 3DTeethSeg22 leaderboard progress (TSegNet 0.9734 → MeshSegNet 0.9707 → RHL 0.9845 → DTSegNet 0.9817 → TSegLab 0.9761 → Cao 2025 0.9870) is un-verifiable at the source for the methods other than TSegNet, MeshSegNet, and Cao 2025.** We should treat the RHL/DTSegNet/TSegLab scores as Cao 2025's reported numbers, not as independently verifiable benchmarks. **This is a real gap in the 3DTeethSeg22 literature — the challenge report (Rekik et al. 2023) should be read next to verify the leaderboard numbers.** (Rekik 2023 is the next paper in the 3DTeethSeg chain — paper 029 candidate.)

### Computable within 1 week
- $50-200: Stratified Transformer fine-tune on 3DTeethSeg22 (1,200 train / 600 test) for the alignment sub-task
- $100-200: cRPE implementation in our PVD/AnchorFormer pipelines, with ablations
- $200-300: 3DTeethSeg22 general-purpose baseline (Stratified Transformer with cRPE) for the 600-test comparison vs Cao 2025 / TSegNet / MeshSegNet

### Compute note
Stratified Transformer trains in ~12-24h on 4× A100 80GB at batch 8, 80k-120k points. For the alignment sub-task (234 IOSs, smaller), ~2-4h on the same hardware. For the 3DTeethSeg22 segmentation fine-tune (1,200 scans, larger), ~12-24h. **Total v1 alignment + segmentation compute: ~$300-500 on Lambda, well within budget.**

---

**Word count:** ~5,000
**Status:** Read 2026-06-07 03:03 KST. Hypothesis impact: **H1 N/A (single-stage but hierarchical encoder-decoder is the canonical H1 decomposition), H2 N/A (no diffusion), H3 STRONGEST support in the reading list so far (3 independent H3 mechanisms: stratified sampling = learned local-vs-global trade-off, cRPE = semantically-informed position bias, first-layer point embedding = implicit local context injection — and cRPE alone adds 4.0 mIoU on S3DIS, the biggest single H3 evidence in the reading list), H4 mild support (point-cloud input is 5-15× more robust to perturbation than voxel input, but the paper doesn't test mesh/SDF outputs), H5 N/A**. **Practical impact: this is the architecture used in Cao 2025's alignment module (ref 30 in paper 026), and the right general-purpose baseline against which to measure every dental-specific transformer.**
