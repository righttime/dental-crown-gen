# 010 — SeedFormer: Patch Seeds based Point Cloud Completion with Upsample Transformer

- **Title:** SeedFormer: Patch Seeds based Point Cloud Completion with Upsample Transformer
- **Authors:** Haoran Zhou¹, Yun Cao², Wenqing Chu², Junwei Zhu², Tong Lu¹*, Ying Tai², Chengjie Wang²
- **Affiliations:** ¹State Key Lab for Novel Software Technology, Nanjing University · ²Youtu Lab, Tencent
- **Year:** 2022 (arXiv:2207.10315 v1, 21 Jul 2022)
- **Venue:** ECCV 2022 (European Conference on Computer Vision, oral/poster)
- **Links:**
  - arXiv: https://arxiv.org/abs/2207.10315
  - ECCV: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630409.pdf
  - Code: https://github.com/hrzhou2/seedformer (MIT, PyTorch, pretrained weights released)
  - Project page: https://hrzhou2.github.io/publications/
- **Code/data:** PyTorch implementation released by authors. Pretrained on PCN, ShapeNet-55, ShapeNet-34, KITTI. 12.8 MB generator model. Test set outputs published.

---

## TL;DR

SeedFormer introduces **Patch Seeds** — a new shape representation that *preserves regional/local information* by storing features in 256 local seeds (instead of pooling to one global feature and losing everything) — and an **Upsample Transformer** that extends the transformer self-attention into the *basic operation* of point generation, so each new point is computed as a self-attention-weighted average of features in its kNN local neighborhood. Together they set a new SoTA on **PCN (CD-Avg 6.74 vs SnowflakeNet 7.21, vs PoinTr 8.38)**, on **ShapeNet-55 (CD-Avg 0.92 vs PoinTr 1.09, −15.6%)**, on **ShapeNet-34 unseen categories (CD-Avg 1.34 vs PoinTr 2.05, −34.6% on generalization)**, and on **KITTI real-scanned cars (Fidelity 0.151 vs GRNet 0.816, MMD 0.516)** — all with only **3.20M parameters** (vs SnowflakeNet 19.3M, GRNet 76.7M), making it the most parameter-efficient model in our reading list.

## Research question

> Can point cloud completion recover *fine-grained local geometric details* (sharp cusps, smooth planes, thin features) by replacing the lossy **global feature pooling** with a **set of regional features** ("Patch Seeds") that get *propagated* to subsequent upsample layers, AND by making each new point's position depend on its **local neighborhood** via transformer self-attention rather than on a 2D-grid folding or a point-wise deconvolution?

Their answer: yes, **if and only if** you (1) split the encoding into 128 patches with point-transformer + set-abstraction and *keep their features* (no max-pool), (2) generate 256 seed points via a *softmax-free* Upsample Transformer (the seed generator's "remove softmax" trick is what makes seed points spread to *unseen* regions, see Sec. 3.3 last paragraph), and (3) at every upsample layer, condition the new-point generation on the **interpolated seed features** of the parent (a "regional encoding" injected into the transformer as a positional encoding term, Eq. 6).

## Method

### Architecture overview (Figure 2)

1. **Encoder.** Two layers of **Set Abstraction** [Qi et al. 2017] + **Point Transformer** [Zhao et al. 2021] on the 2,048-point partial input → **128 patch centers** `P_p ∈ ℝ^(128×3)` and **patch features** `F_p ∈ ℝ^(128×256)`. No global max-pool — that's the whole point.
2. **Seed Generator.** `F = UpTrans(F_p, P_p)` produces 256 seed features `F ∈ ℝ^(256×128)`. A shared MLP on `F` produces seed coordinates `S ∈ ℝ^(256×3)`. **Coarse point cloud** `P_0` is built by **merging `S` and the original input** then FPS to 512 points.
3. **Upsample Layers.** Three upsample layers with rates `r_1=1, r_2=4, r_3=8` for PCN (so 512 → 512 → 2048 → 16,384). Each layer's `UpTrans` takes the current points, the *interpolated seed features* (Eq. 2: inverse-distance-weighted average of the 256 seed features per point), and the *previous layer's features as skip* (this is the "skip" in the upsample transformer).
4. **Output.** `P_3` is the final 16,384-point completion. Final coordinates = duplicated parent + MLP-predicted displacement offset.

### Patch Seeds — the key representation

The seed features `F` are **not pooled** — they are a `(N_s=256, C_s=128)` tensor that gets *propagated* to every subsequent upsample layer. At each layer, the previous layer's points are augmented with `sl_i = Σ_j 1/d_ij · f_j / Σ_j 1/d_ij` (Eq. 2) — an inverse-distance interpolation of the seed features onto every parent point. This is what gives the upsample layers *regional context* of the complete shape even though they only see the *partial* input through their own local features.

The **two seed groups trick** (Sec. 5, Fig. 6). Each patch center is split into *two* seeds by the seed generator (different self-attention weight groups in Eq. 3). For a symmetric input the two groups are spatially near-duplicates; for an asymmetric input one group stays near the seen patch (preserving structure) and the other infers the missing region. This is the **interpretability** hook for us — "color each completed point by its nearest seed" produces a clinically-meaningful decomposition.

### Upsample Transformer — the key operation

The new point is a self-attention-weighted average of features in the parent's kNN local field. Three equations do all the work:

- **Attention logits** (Eq. 3): `â_ijm = α_m(β(q_i^l) − γ(k_j^l) + δ)`, where `q_i` is the current point's query (concat of `P_l` and `s_l`), `k_j` is the previous layer's feature for point `j` (the "skip"), and `δ` is the **positional encoding** (Eq. 6): `δ = ρ(p_i − p_j) + θ(s_i − s_j)` — i.e. spatial relation *plus* seed-feature relation. The `m` index runs over `r_l` separate kernels (one per output child), each learning a different "shape characteristic" (sharp cusp, smooth plane, normal direction). This is a direct cousin of SnowflakeNet's per-point 1D deconvolution (paper 009) — both impose a "learned shape characteristic" inductive bias.
- **Softmax** (Eq. 4): standard softmax over the kNN neighborhood in *upsample layers*; **deliberately removed** in the *seed generator* because softmax's `[0,1]` normalization biases points toward staying inside the seen region, but the seed generator needs to *spread* to unseen regions. The ablation (Table 7) confirms: w/o softmax 6.74 vs w/ softmax 6.83.
- **Aggregation** (Eq. 5): `h_im = Σ_j a_ijm ∗ (ψ(v_j^l) + δ)` — elementwise product of attention weight and value (transformer-style), then summed over the kNN.

**Implementation details** (Sec. A supplement). 128 patches via SA(128) + PT(128) → SA(256, 128) + PT(256). Channel C=128 throughout the upsample layers. Adam, lr=0.001, decay×0.1 every 100 epochs, batch 48 on 2× TITAN Xp. PCN upsampling rates `(1, 4, 8)`, ShapeNet-55/34 rates `(1, 4, 4)`.

### Loss

Two terms (Eq. 7):
- `L_comp` = sum of CD losses at *every* output layer `P_l` plus the seed CD (CD between `S` and a same-size FPS-downsampled GT). Multi-resolution supervision.
- `L_part` = partial matching loss from Cycle4Completion (Wen 2021) on the final output to preserve the input's structure.

## Results

### PCN (Table 1) — 8 categories, partial-from-2.5D-depth, 2048 → 16384 points

| Method | CD-Avg (×10³) |
|---|---|
| PCN [43] | 9.64 |
| GRNet [40] | 8.83 |
| NSFA [44] | 8.06 |
| PoinTr [42] | 8.38 |
| SnowflakeNet [38] | 7.21 |
| **SeedFormer** | **6.74** (−6.5% over SnowflakeNet, −19.6% over PoinTr) |

Best on *all 8 categories* — plane (3.85), cabinet (9.05), car (8.06), chair (7.06), lamp (5.21), couch (8.85), table (6.05), boat (5.85).

### ShapeNet-55 (Table 2) — 55 categories, 25/50/75% partial ratios

| Method | CD-Avg | F1@1% |
|---|---|---|
| FoldingNet | 3.12 | 0.082 |
| GRNet | 1.97 | 0.238 |
| PoinTr | 1.09 | 0.464 |
| **SeedFormer** | **0.92** | **0.472** |

On "hard" (75% missing) specifically: **CD 1.49 vs PoinTr 1.79, −16.8%** — the largest gap is on the hardest cases, which is exactly the regime that matters for a missing-tooth completion (most of the tooth is missing).

### ShapeNet-34 (Table 3) — 21 unseen categories, generalization test

| Method | Seen CD | Seen F1 | **Unseen CD** | **Unseen F1** |
|---|---|---|---|---|
| GRNet | 1.74 | 0.251 | 2.99 | 0.216 |
| PoinTr | 1.23 | 0.421 | 2.05 | 0.384 |
| **SeedFormer** | **0.83** | **0.452** | **1.34** | **0.402** |

**−34.6% on unseen-category CD vs PoinTr.** This is the strongest generalization result in our reading list and is the cleanest empirical support for H5 we've seen (model trained on one distribution, generalizes to novel tooth morphologies).

### KITTI (Table 4) — real-scanned sparse LiDAR cars

| Method | Fidelity ↓ | MMD ↓ |
|---|---|---|
| PCN | 2.235 | 1.366 |
| FoldingNet | 7.467 | 0.537 |
| MSN | 0.434 | 2.259 |
| GRNet | 0.816 | 0.568 |
| **SeedFormer** | **0.151** | **0.516** |

Fidelity improved **5.4× over GRNet, 2.9× over MSN** — the only model that actually preserves the input structure on real, very-sparse scans. Fine-tuned from the PCN-pretrained model on ShapeNetCars (so a clear synthetic→real transfer precedent for H5).

### Ablations (Tables 5–8)

- **Patch Seeds vs global feature**: 6.74 vs 6.97 (the global feature loses 23% relative performance). Seed count: 256 best (128: 7.00, 512: 6.75 — flat above 256).
- **Generator design** (Table 6): Folding 6.93, Deconv 6.90, GraphConv 6.88, Point-wise Attn 6.85, **Upsample Transformer 6.74**. The "local aggregation" property is the consistent winner; full transformer is +0.11 over point-wise attention for +2.7× FLOPs (29.6G vs 7.9G). **The point-wise attention variant (3.12M params, 7.87G FLOPs, 6.85 CD) is the right v0 efficiency point.**
- **Softmax** in seed generator: w/o softmax 6.74 vs w/ softmax 6.83 vs scaled-softmax 6.80 vs log-softmax 6.80. Removing softmax wins.
- **Positional encoding**: none 6.88, positional 6.80, **positional + regional 6.74**, combined 6.78. The regional encoding from the seed features is doing real work (0.06 CD improvement).

### Complexity (Table 9)

| Method | Params | FLOPs | CD-Avg |
|---|---|---|---|
| GRNet | 76.71M | 25.88G | 8.83 |
| SnowflakeNet | 19.32M | 10.32G | 7.21 |
| **SeedFormer** | **3.20M** | 29.61G | **6.74** |
| point-wise attn (variant) | 3.12M | 7.87G | 6.85 |

**3.20M parameters = 23× smaller than GRNet, 6× smaller than SnowflakeNet** — the most parameter-efficient complete architecture in our reading list by a wide margin. The full Upsample Transformer's FLOPs (29.6G) are roughly on par with GRNet's; the point-wise attention variant drops FLOPs to 7.9G while still beating every prior method on CD.

## Connections to our hypotheses (H1–H5)

- **H1 (2-stage seg + gen > end-to-end)** — **MILD support.** SeedFormer is *itself* a 2-stage architecture (encoder → seed generator → upsample layers), and the ablation in Table 5 explicitly shows that a *global feature* at the seed-generator bottleneck loses 0.23 CD vs. the patch-seed alternative. So even within a single completion network, the *preserve local structure* trick outperforms the *collapse to global* trick. This is a fine-grained, not architectural, piece of H1 support. **Concretely:** SeedFormer's encoder can be replaced with our 3DTeethSeg22-trained FDI-segmentation head without retraining the upsample transformer — the architecture supports the 2-stage (segmentation then conditional generation) decomposition cleanly.

- **H2 (diffusion on point clouds > mesh-based VAE)** — **NO RELEVANT EVIDENCE (deterministic).** SeedFormer is a deterministic encoder-decoder with no stochastic component. It directly competes with the deterministic side of H2 (PCN, FoldingNet, SnowflakeNet, PoinTr) and wins, which is a baseline for "what non-diffusion completion looks like in 2022". **Concretely:** SeedFormer is the right *baseline* against which to compare any LION/Diffusion-SDF variant — if the diffusion model can't beat 6.74 on PCN, it isn't worth the $1,500 of Lambda compute to scale to teeth. **Caveat for the v0 prototype:** SeedFormer is the cheapest competitive completion model in our reading list and should be the v0 candidate alongside PoinTr (paper 008) and SnowflakeNet (paper 009).

- **H3 (conditioning on adjacent + opposing teeth improves outer surface quality)** — **STRONGEST ARCHITECTURAL SUPPORT YET.** This is the central contribution. Three concrete mechanisms, all directly mappable to our use case:
  1. **Patch Seeds as regional conditioning.** The seed features are not a single vector — they are a (256, 128) tensor, one row per *region* of the object. The upsample layers receive an *interpolated* regional feature at every point (Eq. 2). For us: each seed can be tied to a *specific tooth* (FDI 14, 15, 16, 17, 18, …) and the interpolation becomes a per-point "how much of each neighbor tooth's identity is influencing this new point". This is the cleanest implementation of H3 in our reading list — better than LION's `z_0`-conditioned AdaGN (paper 005, single global vector) and better than SnowflakeNet's skip-transformer (paper 009, only operates on parent-child neighbors).
  2. **Regional positional encoding** (Eq. 6). The transformer's `δ` includes `θ(s_i − s_j)` — the difference of *seed features*, not just positions. This is the right inductive bias for our use case: an "occlusal" seed and a "buccal" seed have different feature vectors, and new points on the occlusal surface should attend preferentially to other occlusal points. **This is the only architecture in our reading list that explicitly represents *which anatomical region* a new point belongs to.**
  3. **The kNN attention range.** The "k" in the local field controls how far local context reaches. For a tooth-sized object the default k (8) is too small; **for our v0 pilot bump k to 16–24** so the occlusal surface can attend across the entire prep.
  
  **Open question for HK:** the *seed-feature* interpolation in Eq. 2 is weighted by **inverse Euclidean distance** in 3D space. For our use case, this means an inner-surface point near the prep will get strong influence from the *adjacent* tooth's seed (because it's geometrically close) and weak influence from the *opposing* tooth's seed (because it's geometrically far). That's the wrong default for the *occlusal* surface, which should be dominated by opposing teeth. **Hypothesis test:** replace the inverse-distance kernel with a learned, FDI-aware kernel (or two kernels — "neighbor-driven" for inner surface, "opposing-driven" for outer surface) and ablate on 3DTeethSeg22 with one tooth masked.

- **H4 (implicit SDF > explicit mesh)** — **NO RELEVANT EVIDENCE (point cloud, not SDF).** SeedFormer is the *most explicit* representation in our reading list — a set of 3D points with no field, no surface, no normals. But it has a *clean role* as the **completion backbone upstream of DiGS** (paper 003): lift the predicted point cloud to an implicit SDF via DiGS, then extract a mesh via FlexiCubes (paper 007). The Patch Seeds representation is a particularly *good* input to DiGS because the seeds are *structured* (one per region) rather than uniformly sampled — DiGS's divergence penalty on the learned SDF field can be designed to respect the seed boundaries (e.g., the prep-margin seed is a *hard* boundary, the occlusal-surface seed is a *soft* surface).

- **H5 (synthetic data from existing CAD libraries can bootstrap training)** — **STRONGEST SUPPORT YET.** Three independent pieces of evidence:
  1. **ShapeNet-34 unseen categories (Table 3):** trained on 34 categories, tested on 21 *never seen during training* categories. SeedFormer is the only model in our reading list that explicitly evaluates this. CD-Avg 1.34 vs PoinTr 2.05 (a 34.6% gap) — the gap *widens* on unseen categories, which is the opposite of what naive over-parameterized transformers do. This is the cleanest H5 evidence we've collected.
  2. **KITTI real-scanned transfer (Table 4):** PCN-pretrained + ShapeNetCars fine-tune → real LiDAR cars, Fidelity 0.151 (5.4× better than GRNet). Synthetic → real transfer works, and works *better* than models that didn't pretrain on synthetic.
  3. **The two-seed-group interpretability** (Fig. 6, Sec. 5): the model *learns to decompose* unseen shapes into "preserved" + "inferred" regions, and this decomposition is consistent across categories. This is the H5-friendly property that we want: a model pretrained on a generic tooth CAD library should still decompose a *patient-specific* partial scan into the same "what we know" vs. "what we infer" split.
  
  **Concrete action for the v0 pipeline:** the H5 pilot becomes (a) train SeedFormer on 3DTeethSeg22 with one tooth *masked per arch* (synthetic missing-tooth dataset, no labeling cost), (b) fine-tune on a small set of patient-specific cases (10–50), (c) evaluate whether the patch-seed decomposition gives us a *trust* metric per completed tooth region (e.g., high-confidence if the seed is a "preserved" type, low-confidence if "inferred" — useful for the dentist's review step).

## Surprises / interesting things buried in section 4

1. **The softmax removal in the seed generator is the single biggest architectural trick for unseen regions.** The ablation (Table 7) is small (0.09 CD) but the *qualitative* effect (Fig. 3c) is that w/o softmax the seed generator produces *spread* points covering the unseen region; w/ softmax it produces *concentrated* points near the seen region. This is a general insight for any "fill the gap" task: **softmax constrains to the support of the input distribution, which is the opposite of what you want when generating content outside the support of the input.** Note for DiGS: SIREN activations don't have this problem (they're smooth, unbounded), but for the *upstream* completion model we should default to softmax-free attention for the *seed-generation* pass.

2. **The "regional encoding" `θ(s_i − s_j)` is what makes Patch Seeds different from regular point-wise features.** It is not in the original Point Transformer; the supplement confirms it's an addition specific to SeedFormer. **This is the right architectural idea for multi-region generation** (one seed per FDI position, the regional encoding injects *which* tooth the new point belongs to).

3. **The point-wise attention variant (3.12M params, 7.87G FLOPs, 6.85 CD) is the best compute-accuracy trade-off in our reading list.** For our v0 prototype on Lambda, this variant gives us sub-$30 training cost. The full Upsample Transformer variant gives us 0.11 extra CD at 4× the FLOPs — worth it for v1, not for v0.

4. **SeedFormer is trained on a single partial input (no data augmentation tricks like PoinTr's mix-up or PCN's random scaling discussed in the paper).** The 3.20M-param count being so low with no overfitting on ShapeNet-34 unseen suggests the architecture is genuinely *generalizing* rather than memorizing. This is the most encouraging sign for H5 in our reading list.

5. **The two-seed-group visualization (Sec. 5, Fig. 6) is the first *interpretable* completion result in our reading list.** Coloring each completed point by its nearest seed produces a clinically-meaningful decomposition. We can ship this as a "trust heatmap" to the dentist — a per-region confidence that comes for free from the model architecture.

6. **The KITTI fine-tune protocol (PCN-pretrain → ShapeNetCars fine-tune → KITTI inference) is exactly the v0 pipeline we want.** Same three-stage transfer: synthetic arch → synthetic tooth → patient scan.

## Quote-worthy sentences

- *"Patch Seeds, which not only captures general structures from partial inputs but also preserves regional information of local patterns."* (abstract)
- *"this global feature structure possesses two intrinsic drawbacks in its representation ability: (i) fine-grained details are easily lost in the pooling operations in the encoding phase and can hardly be recovered from a diluted global feature in the generation, and (ii) such a global feature is captured from a partial point cloud, thus representing only the 'incomplete' information of the seen part, and is contrary to the objective of generating the complete shape."* (Sec. 1)
- *"the standard transformer structure represents an intrinsic limitation that the softmax normalization explicitly produces attention weights within a specific range of (0, 1). This may limit the learning ability especially in the seed generator."* (Sec. 3.3)
- *"it is essential for the seed generator to produce seed points outside the local neighborhood."* (Sec. 3.3)
- *"Generates 3D points is a fundamental step for point cloud processing and can be generalized to wider research areas."* (Sec. 1)
- *"The idea of Patch Seeds is to capture both global shape structure and fine-grained local details by learning regional features which are stored in several local seeds."* (Sec. 6)

## Code/data link

- Code (MIT, PyTorch, pretrained models, test set outputs): https://github.com/hrzhou2/seedformer
- Datasets: PCN (Yuan 2018, https://www.merl.com/research/highlights/point-completion-network), ShapeNet-55/34 (Yu 2021, https://github.com/yuxumin/PoinTr), KITTI (Geiger 2012, http://www.cvlibs.net/datasets/kitti/)

## For our project — concrete next steps

**Adopt SeedFormer as the v0 completion backbone for sub-task 4 (outer surface).** Specifically:

1. **Reuse the Patch Seeds representation as our H3 conditioning mechanism.** Map 3DTeethSeg22's FDI labels to 32 seed slots (one per tooth position in the arch) and let the seed generator predict the missing tooth's seeds conditioned on the *present* teeth's seeds. The Patch Seeds become a *per-tooth* representation rather than per-region-of-object, but the architecture is identical.

2. **Use the point-wise attention variant (3.12M params, 7.87G FLOPs, 6.85 PCN CD) for the v0 prototype** on Lambda. Training budget: ~$30 for a single-class pilot, ~$100 for a 5-class pilot. Full Upsample Transformer is for v1.

3. **Replace inverse-distance interpolation (Eq. 2) with an FDI-aware kernel** for the inner vs. outer surface distinction. Inner-surface points get strong "neighbor tooth" influence (for the contact area); outer-surface points get strong "opposing tooth" influence (for the occlusal fit). Two-kernel ablation on 3DTeethSeg22 with one tooth masked.

4. **Remove softmax in our seed generator** (Sec. 3.3 trick). This is critical: our seeds need to *spread* to the missing region, not concentrate on the seen region.

5. **Lift SeedFormer's output to a DiGS SDF and extract a FlexiCubes mesh** for the printability check. The Patch Seeds output is a point cloud; for clinical use we need a watertight mesh, and our paper 003 (DiGS) + paper 007 (FlexiCubes) pipeline is the right post-processor.

6. **Compute budget for the v0 pilot:** ~$30–$100 on Lambda (vs LION's $1,500 or Diffusion-SDF's $2,000). This is the cheapest full v0 in our reading list and should be the *first* thing we ship.

7. **For H5 evidence on the v0 dataset:** the ShapeNet-34 unseen-category evaluation (Sec. 4.3) is a direct precedent for our "synthetic CAD pretrain → patient fine-tune → test" pipeline. The −34.6% gap over PoinTr on unseen categories is the strongest argument we have that synthetic pretraining will transfer to patient variability.

8. **Ship the "trust heatmap" UX:** color each completed point by its nearest seed. The two-seed-group decomposition (Sec. 5, Fig. 6) gives us a free confidence map for the dentist's review step.

**Compute note:** SeedFormer trains in 2× TITAN Xp GPUs with batch size 48. The Lambda A10 (24GB) or A100 (40/80GB) instances should handle this with batch 24–32. Estimate 6 hours for PCN, 12 hours for ShapeNet-55, 2 hours for a tooth-specific fine-tune.

**Next paper to read:** *AnchorFormer (CVPR 2023, arXiv:2303.04724)* — the direct successor to SeedFormer with discriminative-node completion. Or *PMP-Net++ (TPAMI 2023)* if we want the "multi-step point moving paths" extension of the existing approach. Or *SVDFormer (ICCV 2023)* for a 2023-era point of comparison with self-view augmentation. AnchorFormer is the highest-priority pick because it generalizes the Patch Seeds idea to *learned* anchor positions rather than fixed per-region seeds.
