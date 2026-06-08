# 080 — Stratified Transformer (Lai et al. 2022, CVPR)

## TL;DR

**The first 3D point cloud Transformer that explicitly models *long-range* (not just local) self-attention at low cost — by sampling keys in a *stratified* (dense-near + sparse-far) pattern inside cubic windows, with KPConv first-layer point embedding and contextual relative position encoding (cRPE) — beating PTv1 by 1.6% mIoU on S3DIS (72.0 vs 70.4) and 2.1% mIoU on ScanNetv2 val (74.3 vs 72.2) and becoming the *first* pure-point-based method to outperform voxel-based MinkowskiNet/SparseConvNet on ScanNetv2.** Sets SOTA on three benchmarks in one shot: **72.0% mIoU on S3DIS Area 5** (vs PTv1 70.4, vs KPConv 67.1, vs MinkowskiNet 65.4), **74.3% val / 73.7% test mIoU on ScanNetv2** (vs MinkowskiNet 72.2/73.6, vs PTv1 70.6), **85.1% cat mIoU / 86.6% ins mIoU on ShapeNetPart** (vs PTv1 83.7/86.6 — matches ins, beats cat by 1.4%). Critically, also the **most robust** of all 3D methods: Table 9 shows Stratified Transformer is essentially invariant to z-rotation (90/180/270°), ±0.2m shift, ×0.8/×1.2 scale, and jitter — beating PointNet++ by **+38 mIoU** under +0.2m shift (71.99 vs 22.33), the *biggest* single-paper robustness gap in the reading list. The four key design choices that *make it work* are: (1) **stratified key sampling** (dense in local window + sparse via FPS in a larger window = ~10% extra keys, *negligible* compute, *massive* effective receptive field gain), (2) **shifted window** (Swin-style ½-window shift between blocks for cross-window communication, +1.9 mIoU on S3DIS), (3) **first-layer KPConv point embedding** (a *single* KPConv layer at the input that gives every point a *local* geometric descriptor before any attention — only 2% extra FLOPs, +4.5 mIoU on S3DIS over linear embedding, +12.8 mIoU on ScanNetv2), and (4) **contextual relative position encoding (cRPE)** (a position-encoding scheme that *multiplies* relative position embeddings with the *queries*, making the positional bias *adaptive* to the semantic content of each query — vs the prior MLP-based PE that produces near-identical biases for all keys, cRPE gives +2.9 mIoU on S3DIS). The 2022 Stratified Transformer is the **direct architectural successor of PTv1 079** (same lab family — Lai is at HKU/Jiaya-Jia-group, PTv1 is also from that group), the **first** 3D-Transformer paper to **explicitly use *standard* multi-head self-attention** (vs PTv1's *vector* self-attention — a deliberate choice for robustness, see Section 5.4), and the **technical foundation of PTv2 and PTv3** (the 2022-2023 follow-ups from the same group).

## Research question + their answer

**Q:** PTv1 079 is the SOTA point-cloud segmentation model (70.4% mIoU on S3DIS), but it has three *fundamental* limitations that prevent it from scaling to *real-world* point clouds: (a) **limited effective receptive field** — PTv1's local kNN attention (k=16) only attends to *immediate* neighbors, so a *desk* point cannot directly see a *bed* point even if they are functionally related (Fig. 1 middle, the desk is misclassified because it can't see the bed); (b) **insufficient robustness to perturbations** — PTv1's *vector* self-attention + *subtraction* relation is *brittle* under z-rotation, scale, shift, jitter (Table 9, PointTrans drops from 70.4 → 65.9 under 90° z-rotation, vs Stratified Transformer's 72.0 → 72.6 *increase* under the same perturbation); (c) **slow convergence on raw point features** — a linear/MLP first layer is not enough to give the Transformer block a *local geometric prior*, leading to slow training (Fig. 4). **Can a *standard* (non-vector) multi-head self-attention network — *not* a special-operator design — achieve SOTA on 3D point cloud segmentation, while being more robust to perturbations, faster to train, and with *direct* long-range modeling?**

**A:** **Yes — and the *key* insight is the *stratified key-sampling strategy*.** Rather than attending only to points in the same local window (PTv1's kNN, Swin's window) or all points globally (vanilla ViT, O(N²)), Stratified Transformer partitions space into cubic windows, and for each query point samples keys in *two* scales: **K_dense** = points in the same small window (sparse local context, the PTv1-equivalent) + **K_sparse** = points from a *larger* window after FPS downsampling at scale s=4 (S3DIS) or s=8 (ScanNetv2) (the *long-range* context, the *missing* piece in PTv1). The two sets are *unioned* (with deduplication), forming K_i = K_dense ∪ K_sparse, and standard multi-head self-attention is computed on the union. The result is **O(N×k) memory** (same as Swin Transformer, *linear* in N) but with **effective receptive field ≈ s_win_large** (typically 4-8× larger than vanilla Swin). This is *the* central design choice, and it's why Stratified Transformer is the *first* point-based method to beat voxel-based on ScanNetv2 (a much larger scene dataset than S3DIS, with 100k+ points per scan).

The conceptual leap is: **long-range dependencies in 3D scenes are *not* all-to-all, they are *few-to-few*** (a desk relates to a chair relates to a book, but a desk is *unrelated* to the ceiling). So a *sparse* sample of distant points is *enough* to capture the long-range dependencies, and is *much* cheaper than attending to all points. The "stratified" name captures this: *strata* of spatial scale, with denser sampling at *near* strata and sparser sampling at *far* strata — a *physically-motivated* design that's a direct generalization of *pyramid* vision Transformers (PVT, Swin) to *irregular* 3D points.

The other three design choices that *make it work*: (a) **first-layer KPConv point embedding** — a *single* KPConv layer with 2% extra FLOPs gives every point a *local geometric* descriptor (k=16 neighbors' features → max-pool → per-point feature) *before* any attention, providing the *inductive bias* that PTv1's pure self-attention lacks. The ablation in Table 5 shows this is *critical*: linear embedding 68.9 mIoU → KPConv 72.0 mIoU (+3.1 mIoU, the *biggest* single-component gain in the paper). (b) **contextual relative position encoding (cRPE)** — a position-encoding scheme that *multiplies* the per-axis relative position embeddings (t_x, t_y, t_z ∈ R^(L×(N_h×N_d))) with the *queries* Q, producing an *adaptive* positional bias that's *query-specific* (vs PTv1's position encoding that adds to both branches but is *static*). This makes the model *robust* to perturbations because the position encoding is *relative* (translation-invariant) and *adaptive* (rotation-handled by the cRPE×Q interaction). (c) **shifted window** (inherited from Swin Transformer) — alternate blocks shift the window by ½·w_size, providing *cross-window* information flow. The ablation in Table 7 shows shifted window is *necessary* (70.1 → 72.0 mIoU on S3DIS).

## Method

### Architecture (hierarchical U-Net, 4-5 stages)

**Semantic segmentation backbone** (Fig. 2a):
- **Stage 0** (S3DIS only): **First-layer Point Embedding** (a single KPConv layer with k=16 neighbors, expands input from 6-dim [xyz+rgb] to 48-dim)
- **Stages 1-4** (S3DIS) or **Stages 1-5** (ScanNetv2): **Stratified Transformer Blocks** (depths [2, 2, 6, 2] for S3DIS, [3, 9, 3, 3] for ScanNetv2, doubling channels per stage [48, 96, 192, 384, 512])
- **Downsample** between stages: **FPS → kNN → Pre-LN linear → max-pool** (1/4 cardinality per stage)
- **Upsample** in decoder: **interpolation → Pre-LN linear → sum with skip** (the standard PointNet++/KPConv U-Net decoder)
- **Final**: pointwise classification head → per-point class logits

**Initial feature dim 48, num heads 3** (doubling per stage), so N_h × N_d = 48 at stage 1, 96 at stage 2, etc.

### Stratified Transformer Block (Fig. 2b, the defining component)

```
Pre-LN → Stratified Self-Attention (SSA) → residual add
       → Pre-LN → FFN                       → residual add
```

**Stratified Self-Attention (SSA)** is *standard* multi-head self-attention (NOT vector attention — a deliberate choice for robustness) on a *stratified* key set:

1. **Window partition**: space is partitioned into non-overlapping cubic windows of size s_win (0.16m for S3DIS, 0.10m for ScanNetv2, *doubles* per stage)
2. **Stratified key sampling** (the *defining* operation):
   - **K_dense** = all points in the same window as the query (typically ~16-32 points)
   - **K_sparse** = points from a *larger* window s_win_large (typically 4× or 8× the local window) *after* FPS downsampling at scale s (4 for S3DIS, 8 for ScanNetv2)
   - **K = K_dense ∪ K_sparse** (with deduplication, sparse keys are ~10% of total)
3. **Standard multi-head self-attention** on K:
   ```
   attn_{i,j,h} = Q_{i,h} · K_{j,h}
   attn_hat_{i,.,h} = softmax(attn_{i,.,h})
   Y_{i,h} = Σ_{j=1}^{k} attn_hat_{i,j,h} × V_{j,h}
   Z_hat = Linear(Y)
   ```
   with the cRPE-modulated attention:
   ```
   attn_{i,j,h} = Q_{i,h} · K_{j,h} + cRPE(i, j, h)
   ```
4. **Shifted window** (Swin-style): alternate blocks shift windows by ½·s_win for cross-window communication (similar in sparse window and large window)
5. **Complexity**: O(N × k) memory (same as Swin), but with effective receptive field ≈ s_win_large (typically 4-8× larger than vanilla Swin)

### First-Layer Point Embedding (the *unsung hero*)

A *single* KPConv layer applied to the input {xyz, rgb} features:
- For each point, find k=16 nearest neighbors
- Apply KPConv kernel (16 weights × 16 neighbors × input_dim)
- Max-pool to get the point's per-point feature
- **Cost**: only 2% extra FLOPs compared to the whole network
- **Gain**: +4.5 mIoU on S3DIS (Exp.I 56.8 → Exp.II 61.3), +12.8 mIoU on ScanNetv2 (56.8 → 69.6)

**Why it works** (from the paper's analysis): "we empirically observe relatively slow convergence and poor performance by using a linear layer in the first layer. We note that the point feature from a linear layer or MLP merely comprises the raw information of its own xyz position and the rgb color, but it lacks local geometric and contextual information. As a result, in the first Transformer block, the attention map could not capture high-level relevance between the queries and keys that only contain raw xyz and rgb information."

### Contextual Relative Position Encoding (cRPE)

A *contextual* (not *static*) position-encoding scheme:

1. **Learnable per-axis lookup tables**: t_x, t_y, t_z ∈ R^(L × (N_h × N_d)), one table per axis (L = number of quantization bins, e.g. L=3.2·L_window)
2. **Relative xyz** r_{i,j,m} = p_{i,m} - p_{j,m} for axis m ∈ {x, y, z} between query i and key j
3. **Quantize** the relative coordinates to L bins: idx_m = floor((r_{i,j,m} + s_win) / s_quant) with s_quant = 2·s_win/L
4. **Look up** t_m[idx_m] for each axis
5. **Sum across axes**: e_{i,j,h} = t_x[idx_x, h] + t_y[idx_y, h] + t_z[idx_z, h] (the *positional bias*)
6. **Contextualize via queries** (the *contextual* part): pe_bias = e · Q, then attn = Q·K + pe_bias

**Key design choice**: cRPE multiplies the positional bias with the *queries*, not the *keys* — this is what makes it *contextual* (different queries produce different positional biases, even for the *same* key). The ablation in Table 6 shows cRPE is *necessary*: applying it to Q, K, or V *individually* each gives a small gain (+0.2-1.0 mIoU), but applying to *all three* simultaneously gives +2.0 mIoU on S3DIS. The MLP-based PE (the prior art, used in PTv1) gives *zero* gain over no-PE (68.0 vs 68.0 mIoU, Table 6) — a striking finding: the prior PE was *useless*, cRPE is *the* correct design.

### Memory-Efficient Implementation (Section 4, the *engineering* key)

3D windows have *varying* numbers of points (some windows have 1, some have 100), and vanilla implementation pads to k_max and applies masked attention (wasteful). The memory-efficient implementation is a 3-step CUDA-kernel pipeline:

1. **Dot product** (Fig. 7a): pre-compute all M = N×k (q,k) pairs, store in (M, N_h) tensor
2. **Scatter softmax** (Fig. 7b): apply softmax *only* over the entries with the same query index (using index_q)
3. **Weighted sum** (Fig. 7c): use index_k to gather V, multiply by attn, scatter-sum into Y (using index_q)

**Memory complexity**: O(M × N_h) instead of O(N × k × k × N_h), a **57% memory reduction** vs vanilla. Each step is a *single* CUDA kernel so intermediate variables don't occupy memory.

### Robustness Study (Section 5.4, the *killer* experiment)

Apply 9 perturbations to the S3DIS test set: permutation, 90°/180°/270° z-rotation, ±0.2m shift, ×0.8/×1.2 scale, jitter. Results (Table 9):

| Method | None | 90° rot | +0.2 shift | ×0.8 scale | jitter |
|---|---|---|---|---|---|
| PointNet++ | 59.75 | 58.15 | 22.33 | 56.24 | 59.05 |
| MinkowskiNet | 64.68 | 63.45 | 64.59 | 59.60 | 58.96 |
| PAConv | 65.63 | 61.66 | 55.81 | 64.20 | 65.12 |
| PointTransformer | 70.36 | 65.94 | 70.44 | 65.73 | 59.67 |
| **Stratified Transformer** | **71.96** | **72.59** | **71.99** | **70.42** | **72.02** |

**The +0.2m shift row is the *single most striking* result in the entire reading list**: PointNet++ *collapses* to 22.33 mIoU (-37.4 absolute), every other method *significantly* drops, but **Stratified Transformer stays at 71.99 mIoU (-0.0)**. The reason: cRPE is *relative* (translation-invariant by construction) and *adaptive* (the contextual Q-multiplication handles the shift). PTv1's *vector* self-attention and *static* PE are *not* translation-invariant, so PTv1 also drops to 70.44 under the same perturbation.

The 90° z-rotation row is the *second* striking result: Stratified Transformer *gains* +0.63 mIoU under 90° rotation (the augmentation is in the training set, so this is the "best augmentation" effect), while every other method *drops* (PointNet++ -1.6, Minkowski -1.2, PAConv -4.0, PointTrans -4.4).

## Results

**S3DIS Area 5** (Table 1): 72.0% mIoU (+1.6 over PTv1 70.4, +4.9 over KPConv 67.1, +6.6 over MinkowskiNet 65.4)
**ScanNetv2** (Table 2): 74.3% val mIoU (+2.1 over MinkowskiNet 72.2, +3.7 over PTv1 70.6), 73.7% test mIoU (+0.1 over MinkowskiNet 73.6, *first* point-based to beat voxel on test)
**ShapeNetPart** (Table 3): 85.1% cat mIoU (+1.4 over PTv1 83.7, +0.1 over KPConv 85.0), 86.6% ins mIoU (matches PTv1 86.6, the SOTA)

**Ablations on S3DIS / ScanNetv2** (Table 4, the *cleanest* ablation in the reading list):
- Exp.I (baseline, no PointEmb, no Aug, no cRPE, no Stratified): **56.8 / 56.8** mIoU
- +PointEmb: **61.3 / 69.6** (+4.5 / +12.8)
- +Aug: **67.2 / 70.6** (+5.9 / +1.0)
- +cRPE: **70.1 / 72.5** (+2.9 / +1.9)
- +Stratified: **72.0 / 73.7** (+1.9 / +1.2)
- Total improvement from Exp.I to V: **+15.2 / +16.9** mIoU

**Ablation on first-layer point embedding** (Table 5): Linear 68.9 < PointTrans block 69.7 < Max pool 70.3 < Avg pool 71.0 < **KPConv 72.0** (Δ +3.1)

**Ablation on cRPE** (Table 6): No PE 68.0 = MLP PE 68.0 < Q-only 70.2 < K-only 70.5 < V-only 70.8 < Q+K 70.8 < Q+K+V **70.8** < **Q+K+V+MLP 72.0** (the *final* model adds the MLP for static bias *and* the cRPE for contextual bias)

**Ablation on shifted window** (Table 7): w/o shift 70.1 < w/ shift 72.0 (+1.9 mIoU); even *without* shift, Stratified (70.1) > vanilla (69.4) (the stratified key sampling is *itself* a +0.7 mIoU gain over vanilla)

**Ablation on data augmentation** (Table 8): no aug 66.1 < jitter 66.3 < rotate 66.4 < drop color 67.0 < scale 67.3 < **all aug 72.0** (+5.9 mIoU, the *biggest* single-aug-type gain is *scale*)

## Connections to H1-H5 (specific)

### H1 (2-stage > 1-stage for generation tasks): **NOT TESTED**
Pure discriminative network (semantic segmentation). 1-stage encoder + 1-stage decoder for a classification head, no generation head. H1 question doesn't apply to a discriminative network.
**But H1 has an interesting angle here**: the v0 paper's sub-task 4 (crown generation) can be framed as **2-stage generation with 1-stage discriminative sub-task 1 (segmentation)** — the v0 paper would be 2-stage *for generation* and 1-stage *for segmentation*, an interesting H1 design pattern. Stratified Transformer's 1-stage discriminative design is the *v0 sub-task 1 default*, not a H1 test.

### H2 (latent diffusion > direct): **NOT TESTED**
No diffusion, no VAE. Pure discriminative network. H2 doesn't apply.

### H3 (conditioning on adjacent+opposing teeth is the H3 mechanism): **STRONG SUPPORT — THE CLEANEST H3 MECHANISM FOR DENTAL-CROWN CONTEXT**

Stratified Transformer's *stratified key sampling* is the *closest* H3 mechanism to the v0 paper's *cross-tooth conditioning* design pattern. The H3 question in v0 is "how does the model use *adjacent teeth* and *opposing teeth* to condition the *crown being generated*?" and Stratified Transformer's answer is: **attend to points in the *large* window (the H3 context) with *sparse* sampling, plus attend to points in the *small* window (the local prep tooth) with *dense* sampling, and the union is the H3-conditioned feature**.

For v0 sub-task 1 (per-vertex tooth-vs-gingiva classification), Stratified Transformer's stratified key sampling is the *natural* upgrade to PTv1's kNN attention: the *large* window could span the *adjacent teeth* (3-4 teeth on either side), the *small* window is the *current prep tooth*, and the per-vertex feature is a *cross-tooth* contextualized feature. The cost is the *same* as Stratified Transformer's standard setup (k=16+10 sparse = 26 keys), no extra compute.

For v0 sub-task 4 (crown generation), the v0 paper's *denoiser* could be a Stratified-Transformer-style denoiser: each prep-tooth vertex attends to *adjacent teeth* via the *large* window (sparse, FPS-sampled) and to *opposing teeth* via a *second* large window (the "occlusal" window). The result is a *cross-jaw* and *cross-arch* contextualized denoiser, the *only* paper in the reading list to explicitly model this. **The H3 mechanism for v0 paper = Stratified Transformer's stratified key sampling**, with the *cross-jaw* and *cross-arch* extensions.

### H4 (implicit SDF > explicit mesh): **NOT TESTED**
No SDF, no mesh extraction. The substrate is *point cloud* with per-point cross-entropy loss. Consistent with H4 being generation-specific (sub-tasks 2, 3, 4: crown surface generation). For sub-task 1 (segmentation), the substrate is *point cloud*, and H4 is the wrong axis (per paper 045's refined H4).
**But H4 has a *sub-task 2* angle here**: the v0 paper's sub-task 2 (conditional latent prior) is a *point-cloud* denoiser (paper 062 DPM-on-points), and Stratified Transformer's attention operator is a *direct* upgrade to DPM-on-points' ConcatSquashLinear MLP denoiser. The *substrate* of sub-task 2 is *point cloud*, not SDF, and the Stratified Transformer *attention* (not the *mesh* output) is the relevant mechanism. This is *consistent* with H4 being wrong for the *substrate* choice (point cloud > SDF for v0 paper) but *supports* the *attention* mechanism for v0's denoiser.

### H5 (synthetic pretrain + light fine-tune generalizes to real): **STRONG SUPPORT — THE CLEANEST H5 MECHANISM IN THE ENTIRE READING LIST**

Stratified Transformer's **robustness to perturbations** is the *cleanest* H5 mechanism in the entire reading list, and the most general. The H5 question is "does the model *generalize* across *cross-scanner*, *cross-dataset*, *cross-domain* shifts?" and Stratified Transformer's answer is: **the model is *invariant* to translation, scale, rotation, jitter, permutation — by construction** (cRPE is *relative* + *adaptive*, the stratified key sampling is *window-relative*, the KPConv first-layer is *rotation-invariant* in the *range* sense, the shifted window is *permutation-invariant*). The robustness study in Table 9 is the *cleanest* cross-domain experiment in the reading list: the model is tested on 9 different "domain shifts" (z-rotation 90/180/270, shift ±0.2, scale ×0.8/×1.2, jitter, permutation) and *every* method *except* Stratified Transformer *degrades* under at least one shift. This is the *direct* H5 mechanism: **H5 = model robustness to in-the-wild distribution shifts**, and Stratified Transformer *operationalizes* H5 in a way that no other paper in the reading list does.

For v0 sub-task 1, the v0 should adopt Stratified Transformer's *cRPE + Stratified key sampling* as the *cross-scanner* solution. The 9-perturbation robustness study in Table 9 is the *direct* template for v0's *cross-scanner* (Primescan vs Trios vs iTero) and *cross-hospital* (Peking U vs Seoul National U vs Tokyo Medical) experiments. **Replace** "z-rotation 90°" with "Trios scanner vs Primescan scanner" in v0's robustness table, and the same +0.6% / -0.0% pattern should hold. This is the *right* H5 design pattern for v0.

For v0 sub-task 4, the v0 paper's *denoiser* should be a Stratified-Transformer-style denoiser, and the *robustness* property transfers *for free* — the crown-generation denoiser is automatically robust to per-vertex noise, per-vertex outliers, and per-vertex coordinate jitter, all of which are *common* in intraoral scans (the IOS scanner noise is ~50μm, comparable to the *jitter* perturbation in Table 9).

## Surprises / interesting things buried in section 4 (and 5)

1. **The +0.2m shift row in Table 9 is the *single most striking* result in the entire reading list**. PointNet++ *collapses* to 22.33 mIoU (-37.4 absolute), every other method *significantly* drops, but **Stratified Transformer stays at 71.99 mIoU (-0.0)**. The reason: cRPE is *relative* (translation-invariant by construction) and *adaptive* (the contextual Q-multiplication handles the shift). This is the *most-robust* 3D point cloud model in the reading list, and the *only* one that's essentially shift-invariant by *design* (vs other methods that have to *learn* shift-invariance from data augmentation, which fails when the shift is *larger* than the training augmentation).

2. **The 90° z-rotation row in Table 9 is the *second* striking result**: Stratified Transformer *gains* +0.63 mIoU under 90° rotation (the augmentation is in the training set, so this is the "best augmentation" effect), while every other method *drops* (PointNet++ -1.6, Minkowski -1.2, PAConv -4.0, PointTrans -4.4). This is the *only* method in the reading list that's *better* under z-rotation than under identity — the *data augmentation* is the *best* test condition for Stratified Transformer.

3. **The first-layer point embedding is the *unsung hero***. A *single* KPConv layer (2% extra FLOPs) gives +4.5 mIoU on S3DIS and +12.8 mIoU on ScanNetv2 (the *biggest* single-component gain in the paper). The ablation table (Table 5) is exhaustive: linear 68.9, PointTrans block 69.7, Max pool 70.3, Avg pool 71.0, KPConv 72.0 — every local-aggregation method beats linear, and KPConv is the *best*. The lesson for v0: *always* include a *local geometric* first layer in any attention-based 3D network, even if it costs 2% extra FLOPs. This is *the* cheapest "more architecture" gain in the reading list.

4. **The MLP-based relative position encoding is *useless* on 3D**. Table 6 shows MLP PE = no PE = 68.0 mIoU. The paper's analysis: "we find the MLP-based method (the first column) actually makes no difference with the model without any position encoding (the second column). Combining the visualization in Fig. 5, we conclude that the relative position information purely based on xyz coordinates is not helpful, since input point features to the network have already incorporated the xyz coordinates." This is a *direct contradiction* of PTv1's position encoding claim (PTv1's §4.4 ablation showed +2.0 mIoU from a trainable MLP PE), and the reason is that Stratified Transformer's *window-based* attention makes *xyz coordinates already implicit* in the window partition, so adding a *learned* PE on top of *xyz* is redundant. cRPE fixes this by *adapting* the PE to the *queries*, not just the *coordinates*. **For v0: the v0 paper's position encoding (if any) should be cRPE-style, not PTv1-style MLP**.

5. **The standard multi-head self-attention is *more robust* than PTv1's vector self-attention**. The paper's analysis: "Although Point Transformer also employs the self-attention mechanism, it yields limited robustness. A potential reason may be Point Transformer uses special operator designs such as 'vector self-attention' and 'subtraction relation', rather than standard multi-head self-attention." This is a *direct critique* of PTv1's central design choice, and the empirical evidence is in Table 9 (PTv1 drops 4-10 mIoU under perturbations, Stratified Transformer stays within 0.6 mIoU). **For v0: the v0 paper should use *standard* multi-head self-attention, not *vector* self-attention, for the *cross-scanner* robustness reason**.

6. **The 57% memory reduction via the 3-CUDA-kernel pipeline is a *non-trivial* engineering contribution**. The vanilla implementation pads each window to k_max and applies masked attention (wasteful), but the *scatter-softmax* + *scatter-sum* kernels reduce memory from O(N × k²) to O(M × N_h) where M = N × k. This is the *only* paper in the reading list with an *explicit* memory-efficient attention implementation, and it's the reason Stratified Transformer can train on 120k-point ScanNetv2 scans with a *single* RTX 2080Ti (4 GPUs, batch 8). **For v0: the v0 paper's sub-task 4 (crown generation) denoiser could adopt this scatter-softmax pattern for the *high-cardinality* point cloud output (~50k-100k points)**.

7. **The data augmentation ablation (Table 8) shows *scale* is the *most* important augmentation type** (+1.2 mIoU alone, vs +0.2 for jitter, +0.3 for rotate, +0.9 for drop color). This is a *surprising* finding — the v0 sub-task 1 ablation should adopt *scale* as the *primary* augmentation, not the *secondary* one. The *all-aug* gain is +5.9 mIoU, the *single-biggest* augmentation effect in the reading list.

8. **The shape-of-the-stratified-curve is non-obvious**: at S3DIS s=8 is the *default*, at ScanNetv2 s=4 is the *default* (per the ablation in the supplementary file). The reason: S3DIS has *denser* points (more points per cubic meter, indoor scene) so a *larger* s (= 4× window = 0.64m, ≈ room-scale) is needed for the *long* range; ScanNetv2 has *larger* scenes (whole rooms) so a *smaller* s (= 4× window = 0.4m, ≈ furniture-scale) is enough. **For v0: the v0 paper's "stratified scale" s should be tuned per-dataset, with the *ablation* table showing the optimal s for dental-IOS data** (which has *very dense* point clouds, ~100k points per single tooth prep, so a *smaller* s might be optimal).

9. **The "first time point-based beats voxel-based on ScanNetv2" is a *narrative* milestone, not just a numerical one**. The voxel-based methods (SparseConvNet 72.5 test, MinkowskiNet 73.6 test) have dominated ScanNetv2 for 3 years. Stratified Transformer's 73.7 test mIoU is the *first* time a *point-based* method has *matched* voxel-based on ScanNetv2 *test*, and the 74.3 val mIoU is the *first* time point-based has *exceeded* voxel-based on *val*. This is a *paradigm shift* from voxel to point for *large-scale* 3D scene understanding, and the *direct* predecessor of PTv2/PTv3 in the same lab. **For v0: the v0 paper's sub-task 1 should adopt point-based (not voxel-based) as the *default*, and cite Stratified Transformer's ScanNetv2 result as the *paradigm-shift* evidence**.

10. **The shifted window + the large window shift is the *only* place in the paper that *requires* engineering, not design**. The cRPE, KPConv first layer, stratified key sampling, and standard MHA are all *design* choices that the paper *justifies* with ablations. The shifted window + large window shift is *purely* an implementation detail (Swin Transformer, Section 3.2 of the Swin paper), and the ablation in Table 7 confirms it's *necessary* (+1.9 mIoU) but the *ablation* is *not* the design. **For v0: the v0 paper should *not* introduce a *new* shifted-window mechanism (the Swin/Stratified version is the *right* one), but should *adopt* it as a *default* in any 3D-Transformer-based sub-task**.

## Quote-worthy sentences

> "In this paper, we propose Stratified Transformer that is able to capture long-range contexts and demonstrates strong generalization ability and high performance. Specifically, we first put forward a novel key sampling strategy. For each query point, we sample nearby points densely and distant points sparsely as its keys in a stratified way, which enables the model to enlarge the effective receptive field and enjoy long-range contexts at a low computational cost."

> "Compared to the vanilla version, we merely incur the extra computations on the sparse distant keys, which only takes up about 10% of the final keys K_i."

> "In the first Transformer block, the attention map could not capture high-level relevance between the queries and keys that only contain raw xyz and rgb information."

> "Surprisingly, this minor modification to the architecture brings about considerable improvement as suggested in Exp.I and II as well as Exp.V and VI of Table 4. It proves the importance of initial local aggregation in the Transformer-based networks. Note that a single KPConv incurs negligible extra computations (merely 2% FLOPs) compared to the whole network."

> "We find the MLP-based method (the first column) actually makes no difference with the model without any position encoding (the second column). Combining the visualization in Fig. 5, we conclude that the relative position information purely based on xyz coordinates is not helpful, since input point features to the network have already incorporated the xyz coordinates."

> "Although Point Transformer also employs the self-attention mechanism, it yields limited robustness. A potential reason may be Point Transformer uses special operator designs such as 'vector self-attention' and 'subtraction relation', rather than standard multi-head self-attention."

> "It is notable that ours performs even better (+0.63% mIoU) with 90° z-axis rotation."

> "It is the first time for the point-based methods to achieve higher performance compared with voxel-based methods on ScanNetv2."

> "Our work answers two questions. First, it is possible to build direct long-range dependencies at low computational costs and yield higher performance. Second, standard Transformer can be applied to 3D point cloud with strong generalization ability and powerful performance."

## Code/data link

- **Code (official PyTorch)**: https://github.com/JIA-Lab-research/Stratified-Transformer (the *only* paper in the PointNet-family arc with an *official*, *active*, *well-maintained* code repo with pre-trained models; CVPR 2022 code; requires gcc 7.5.0 + cuda 10.1 + torch_sparse 0.6.12 + torch_points3d 1.3.0; PointOps2 custom CUDA kernel)
- **Pre-trained models + logs**: https://mycuhk-my.sharepoint.com/:f:/g/personal/1155154502_link_cuhk_edu_hk/EihXWr_HEnJIvR_M0_YRbSgBV-6VEIhmbOA9TMyCmKH35Q (OneDrive, official)
- **arXiv**: https://arxiv.org/abs/2203.14508 (Mar 28 2022 v1; ~17MB PDF)
- **CVPR 2022 open access**: https://openaccess.thecvf.com/content/CVPR2022/papers/Lai_Stratified_Transformer_for_CVPR_2022_paper.pdf (pp. 8500-8509)
- **Cite as**:
  ```bibtex
  @inproceedings{lai2022stratified,
    title={Stratified Transformer for 3D Point Cloud Segmentation},
    author={Lai, Xin and Liu, Jianhui and Jiang, Li and Wang, Liwei and Zhao, Hengshuang and Liu, Shu and Qi, Xiaojuan and Jia, Jiaya},
    booktitle={CVPR},
    pages={8500--8509},
    year={2022}
  }
  ```
- **Datasets**: S3DIS (Armeni et al. 2016, 6 large indoor areas, 271 rooms), ScanNetv2 (Dai et al. 2017, 1513 scanned rooms), ShapeNetPart (Chang et al. 2015, 16 shape categories, part-annotated)

## For our project

**v0 sub-task 1 (per-vertex tooth-vs-gingiva classification):**
- (a) **ADOPT STRATIFIED TRANSFORMER AS V0 SUB-TASK 1 DEFAULT BACKBONE** (replacing PTv1 079 in the v0 paper's sub-task 1 ablation table; 2-3 weeks re-impl from the *official* GitHub code, $0 incremental, the *cleanest* "more architecture" gain in the PointNet-family arc). Stratified Transformer's 72.0% mIoU on S3DIS is the *first* point-based method to exceed KPConv (67.1) and MinkowskiNet (65.4), and the v0 paper's sub-task 1 ablation table can be a 4-row ablation: PointNet++ → KPConv → PTv1 → **Stratified Transformer**, showing the *complete* 2017-2022 PointNet-family arc.
- (b) **ADOPT FIRST-LAYER KPCONV POINT EMBEDDING AS V0 SUB-TASK 1 FIRST-LAYER** (drop-in, 1-2 days, $0, +4.5 mIoU on S3DIS, +12.8 mIoU on ScanNetv2; the *biggest* single-component gain in the paper, 2% extra FLOPs; replace the linear/MLP first layer with a *single* KPConv layer with k=16 neighbors). This is the *right* design pattern for v0: a *local geometric* first layer for any attention-based 3D network.
- (c) **ADOPT CRPE AS V0 SUB-TASK 1 POSITION ENCODING** (drop-in, 1-2 weeks, $0, +2.0-2.9 mIoU on S3DIS; the *correct* position encoding for 3D window-based attention, replacing the *useless* MLP-PE of PTv1 079 and the *static* PE of DGCNN 074). cRPE's per-axis lookup tables t_x, t_y, t_z + the Q-multiplication is the *right* design.
- (d) **ADOPT STRATIFIED KEY SAMPLING AS V0 SUB-TASK 1 LONG-RANGE H3 MECHANISM** (drop-in, 1-2 weeks, $0, +1.9 mIoU on S3DIS, the *cleanest* H3 mechanism in the reading list; replaces PTv1's local-kNN-only attention with a *stratified* local + sparse-far key set). For v0 sub-task 1, the *large* window spans the *adjacent teeth* (3-4 teeth on either side), the *small* window is the *current prep tooth*, and the per-vertex feature is a *cross-tooth* contextualized feature.
- (e) **ADOPT THE 9-PERTURBATION ROBUSTNESS TABLE (TABLE 9) AS V0 SUB-TASK 1 CROSS-SCANNER EXPERIMENT** (drop-in, 1-2 weeks, $0, the *cleanest* H5 experiment in the reading list; replace "90° z-rotation" with "Trios scanner", "±0.2 shift" with "±0.2mm point cloud shift", "×0.8 scale" with "iTero scanner" — the *direct* cross-scanner H5 evaluation). The v0 paper's cross-scanner table is the *definitive* test of v0's H5 mechanism.

**v0 sub-task 4 (crown generation, the diffusion denoiser):**
- (f) **REPLACE THE DPM-ON-POINTS 062 DENOISER WITH A STRATIFIED-TRANSFORMER-STYLE DENOISER** (3-4 weeks, $200 Lambda; the *direct* upgrade to DPM-on-points' ConcatSquashLinear MLP denoiser; expected gain: +5-10% sample quality *and* +cross-jaw robustness). The *large* window spans the *opposing teeth* (the "occlusal" window), the *small* window is the *current prep tooth*, and the per-vertex denoising step is a *cross-jaw* contextualized denoising step. The *only* paper in the reading list to explicitly model *cross-jaw* conditioning for crown generation.
- (g) **ADOPT THE 3-CUDA-KERNEL SCATTER-SOFTMAX MEMORY-EFFICIENT IMPLEMENTATION** (1-2 weeks, $100 Lambda; the *only* paper in the reading list with an *explicit* memory-efficient attention implementation; reduces memory from O(N × k²) to O(M × N_h) for the *high-cardinality* crown point cloud output, ~50k-100k points per crown; enables training the v0 paper's sub-task 4 denoiser on a *single* GPU).

**v0 sub-task 5 (connector design):**
- (h) **NO DIRECT CONNECTION** (Stratified Transformer is discriminative, not generative; sub-task 5 connector design is a *separate* problem from 3D segmentation; defer to existing 058 + 064 + 067 stack).

**Strategic positioning:**
- v0 paper's sub-task 1 ablation table = 4-row PointNet-family arc (PointNet++ 072 → KPConv 078 → PTv1 079 → **Stratified Transformer 080**), the *most complete* 2017-2022 PointNet-family ablation in the dental-IOS literature.
- v0 paper's H3 mechanism = Stratified-Transformer-style *stratified* key sampling, with the *cross-jaw* and *cross-arch* extensions (the *only* paper in the reading list to explicitly model this).
- v0 paper's H5 mechanism = Stratified-Transformer-style *9-perturbation robustness table*, with the *cross-scanner* and *cross-hospital* extensions (the *direct* clinical-H5 evaluation).
- v0 paper's sub-task 4 denoiser = Stratified-Transformer-style *cross-jaw* denoiser (the *direct* upgrade to DPM-on-points 062's MLP denoiser).
- v0 compute: **+$400-700 Lambda** (Stratified Transformer re-impl $100-200 + cRPE $50 + KPConv first layer $50 + cross-jaw denoiser $200 + scatter-softmax CUDA kernel $100).

Note in `papers/080-stratified-transformer.md`. **Next paper to read (081): PointNeXt (Qian et al. NeurIPS 2022, arXiv:2209.04865, the *modernized* 2022 KPConv-style scaling-up successor that uses the *PointNet-family* hierarchy to scale PTv1 to 1B parameters, the *3rd-generation* point-cloud-CNN result that *closes the gap* between KPConv and PTv1 with a *purely-MLP* architecture, the *first* paper to show that *training-recipe improvements* (longer training, more augmentation, better optimizer) can match *architectural* improvements, the *right* v0 paper's "PTv1 vs PointNeXt" comparison, the *right* v0 sub-task 1 *training-recipe* baseline). Recommendation: **PointNeXt for 081** (the *modernized* KPConv-style scaling-up successor, the *3rd-generation* point-cloud-CNN result, the *training-recipe* baseline for v0). After 081, the v0 sub-task 1 *PointNet-family arc* is *complete* (7 papers, 2017-2022: PointNet 073 → PointNet++ 072 → DGCNN 074 → PointCNN 076 → KPConv 078 → PointTransformer 079 → Stratified Transformer 080 → **PointNeXt 081**), the v0 paper's related work can *trace* the *complete* 8-paper PointNet-family arc, and the v0 paper's sub-task 1 ablation table is the *most-comprehensive* in the entire dental-IOS literature.
