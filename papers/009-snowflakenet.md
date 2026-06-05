# 009 — SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer

- **Title:** SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer
- **Authors:** Peng Xiang* (Tsinghua), Xin Wen* (JD.com), Yu-Shen Liu† (Tsinghua), Yan-Pei Cao (Kuaishou Y-tech), Pengfei Wan (Kuaishou), Wen Zheng (Kuaishou), Zhizhong Han (Wayne State) — *equal contribution; †corresponding
- **Affiliations:** School of Software, BNRist, Tsinghua University · Y-tech, Kuaishou Technology · Wayne State University · JD.com
- **Year:** 2021 (ICCV 2021, **Oral**; arXiv:2108.04444 v1 Aug 10 2021, v2 Oct 27 2021)
- **Journal extension:** "Snowflake Point Deconvolution for Point Cloud Completion and Generation with Skip-Transformer" — TPAMI 2023, vol. 45 no. 5, pp. 6320–6338 (arXiv:2202.09367, Feb 2022). The journal version extends SPD to **auto-encoding, unconditional generation, single-view reconstruction, and point cloud upsampling** in addition to completion.
- **Venue:** IEEE/CVF International Conference on Computer Vision (ICCV) 2021
- **Links:**
  - Paper: https://arxiv.org/abs/2108.04444
  - ICCV open-access: https://openaccess.thecvf.com/content/ICCV2021/papers/Xiang_SnowflakeNet_Point_Cloud_Completion_by_Snowflake_Point_Deconvolution_With_Skip-Transformer_ICCV_2021_paper.pdf
  - Official code: https://github.com/AllenXiangX/SnowflakeNet (MIT, PyTorch, pretrained weights released)
  - Jittor port: https://github.com/AllenXiangX/SPD_jittor (released Feb 2023)
- **Code/data:** PyTorch implementation released by the authors. Pretrained on PCN, Completion3D, ShapeNet-34/21, ShapeNetCars, KITTI. Completion uses PCN dataset; supplementary adds ShapeNet-34 unseen-class completion (PoinTr-style).

---

## TL;DR

SnowflakeNet reframes point-cloud completion as the **snowflake-like progressive splitting** of a coarse 512-point seed into a dense 16,384-point complete shape through a stack of three **Snowflake Point Deconvolution (SPD)** layers, each of which expands its parent points into r× child points via a learned point-wise 1D deconvolution. The key trick is a **skip-transformer** that lives *between* consecutive SPD layers: it takes the parent points' per-point feature as a **query** and the previous layer's displacement features (k-nearest-neighbor selected) as **keys/values**, so the *current* splitting pattern is informed by the *previous* splitting pattern in the same local patch. The result is the cleanest local-detail generation seen in the field: it sets SoTA on PCN (CD 7.21 vs NSFA's 8.06, **−10.5%**) and Completion3D (CD 7.60 vs PMP-Net's 9.23, **−17.3%**), and — critically for us — produces **sharper cusps, smoother flat planes, and more locally structured point arrangements** than the folding-based or vanilla coarse-to-fine baselines it was designed to fix.

## Research question

> Can the *local geometric structure* of a completed point cloud (sharp edges, smooth planes, thin tubes) be recovered by **explicitly structuring the point generation process** — rather than treating the decoder as an unconstrained point-wise MLP — and by **letting successive decoder layers condition on each other** rather than decoding each resolution independently?

Their answer: yes, but only if you (1) commit to a **rooted-tree growth model** where every child point has an identifiable parent and a small set of *learned shape characteristics* (kernels) that combine additively, (2) compute the displacement with a **per-point 1D deconvolution** (kernel size = stride = r_i) so the splitting respects local shape context, and (3) wire consecutive layers with a **skip-transformer that uses the previous layer's displacement features as attention context** — the "skip" being the connection between the *current* point feature and the *previous* layer's displacement feature, not a U-Net style encoder skip.

## Method

### Architecture overview (Figure 3)

1. **Feature extraction module.** Input partial point cloud `P` of size N×3 (PCN: N=2048; Completion3D: N=2048). Three layers of **set abstraction** from PointNet++ [41] (Qi 2017), each augmented with a **point-transformer** [60] block to inject local shape context. Output: a global **shape code** `f` of size 1×C (C=512).
2. **Seed generation module.** `f` is fed to a **point-wise splitting** operation to produce per-point features, then through an MLP to generate a coarse complete point cloud `P_c` of size N_c×3 (N_c=256). Concatenated with the input `P`, then **FPS**-downsampled to `P_0` of size N_0×3 (N_0=512). The seed is the "root" of every snowflake tree.
3. **Point generation module.** Three SPD layers with up-sampling factors **r_1=1, r_2=4, r_3=8** (so 512 → 512 → 2048 → 16,384). Each SPD is preceded by a **skip-transformer** that aggregates the previous SPD's displacement features (using kNN-selected neighbors) into a *shape context* feature used by the current SPD's per-point splitting.

### Snowflake Point Deconvolution (SPD) — the key operation

Goal: turn a set of N_{i-1} parent points with features `H_{i-1} = {h_j^{i-1}}` into a set of N_i = r_i · N_{i-1} child points with features, **respecting local structure**.

- For each parent point, run a small **PointNet** to get per-point feature `q_j^{i-1}`.
- Pass `q_j^{i-1}` through the **skip-transformer** to get shape-context feature `h_j^{i-1}` (this is the SPD's per-point feature, of dim C'=128). The skip-transformer pulls in the previous layer's displacement features for context.
- **Point-wise splitting**: each `h_j^{i-1}` is a vector of C' logits, where each logit `h_{j,m}^{i-1}` is interpreted as the *activation* of the m-th **learned kernel** `K_m ∈ ℝ^{r_i × C'}`. The k-th child point feature is the activation-weighted sum of kernel rows: `g_{j,k} = Σ_m h_{j,m}^{i-1} · k_{m,k}`. (See Eq. 2.) This is a **1D deconvolution** with kernel size = stride = r_i, applied independently per point.
- Concat with a duplicate of the parent's shape-context feature (preserves the parent's geometry in the child), feed the r_i × 2C' tensor to an MLP, and produce the **displacement feature** `K_i = {k_j^i}` (this is also the input to the next skip-transformer).
- The actual 3D displacement is `ΔP_i = tanh(MLP(K_i))` (Eq. 1), and the child point positions are `P_i = P̂_i + ΔP_i` where `P̂_i` is the r_i-times-duplicated parent.
- The `tanh` clamps the displacement to [-1, 1] (scaled by the bounding box), preventing degenerate splits.

**The kernel interpretation is the conceptual hook.** Section 3.2 phrases the splitting as: "the *m*-th kernel K_m indicates a certain shape characteristic (e.g., a sharp cusp direction, a smooth surface normal), and h_{j,m}^{i-1} is the activation of that characteristic at parent point j." The child point is a weighted combination of activated shape characteristics. This is the **structural inductive bias** the paper claims is missing from PCN/NSFA/SA-Net.

**Flexibility knob.** When r_i=1 (the first SPD), the layer is used to *rearrange* seed points to a better position rather than multiply them — the first SPD is essentially a refinement step. When r_i>1, the layer expands.

### Skip-Transformer (Figure 5, Eqs. 3–5)

The skip-transformer lives *between* SPD layers (not inside a single SPD). It takes:
- per-point feature `q_j^{i-1}` (the "query" — comes from the new SPD's PointNet)
- displacement feature `k_j^{i-1}` (the "key/value" — comes from the previous SPD's output)

Steps:
1. Concat `q_j^{i-1}` and `k_j^{i-1}`, pass through an MLP → value vector `v_j^{i-1}`.
2. For each parent point, find its **k-nearest neighbors** in coordinate space (k=8 in implementation). The kNN restricts attention to *local* patterns and is what makes the operation tractable.
3. Attention: `a_{j,l}^{i-1} = softmax_l(MLP(q_j^{i-1} ⊖ k_{j,l}^{i-1}))` where ⊖ is element-wise subtraction (Eq. 4). Note this is **not standard dot-product attention** — the relation is computed by an MLP on the difference, the same form as the "relation attention" in Set Transformer / ecto-architectures.
4. Shape context: `h_j^{i-1} = v_j^{i-1} ⊕ Σ_l a_{j,l}^{i-1} ⊙ v_{j,l}^{i-1}` (Eq. 5), a gated sum.

**For the first SPD** there's no previous displacement, so `q_j^0` is used as both query and key (the skip-transformer degenerates to a small self-attention on the seed features). This is the "bootstrapping" trick — it means the first SPD still has a local attention layer to organize the seed cloud before the growth pattern takes over.

### Training

- **Loss**: `L = L_completion + λ · L_preservation` (Eq. 6).
  - **L_completion**: sum of 4 **Chamfer Distances** between the predicted {P_c, P_1, P_2, P_3} and the corresponding GT point clouds (downsampled to matching density). The 4-term CD supervises the *growth trajectory*, not just the final output.
  - **L_preservation**: the unidirectional partial matching loss from [48] (Huang 2020, Point FractalNet) — penalizes cases where the output point cloud does not contain the input points, so the model can't "move" the existing teeth to make room for hallucinated ones.
  - λ is set such that L_preservation is ~1/10 the magnitude of L_completion.
- **Optim**: Adam (default settings), **300 epochs**, batch size 32 (PCN) or 24 (Completion3D), NVIDIA Titan Xp / V100 GPUs. LR scheduler: cosine decay with warmup.
- **Other details**: feature dim C=512, C'=128, k=8 for kNN, k=16 for set-abstraction neighbor selection.
- **Output resolution**: 16,384 points (PCN) or 2,048 (Completion3D — the GT there is 2,048).

### Datasets

- **PCN** (Yuan 2018, [57]): 8 categories, 28,974 train / 1,200 test. Complete shapes have 16,384 points sampled from the surface. Incomplete shapes are 8 back-projected partial views per complete shape. The "dense completion" benchmark — fine for evaluating *local* detail.
- **Completion3D** (Tchapmi 2019, [43]): 30,958 models / 8 categories, train 28,974 / val 800 / test 1,184. Both partial and complete are **2,048 points**. The "sparse completion" benchmark — coarser, harder for local detail.
- (TPAMI 2023 extension adds ShapeNet-34/21 unseen-class completion, ShapeNetCars, KITTI for fine-tuning, single-view reconstruction, and ShapeNet unconditional generation.)

## Results

### Table 1 — PCN (per-point L1 CD × 10³, lower is better)

| Method | Average | Plane | Cabinet | Car | Chair | Lamp | Couch | Table | Boat |
|---|---|---|---|---|---|---|---|---|---|
| FoldingNet (Yang 2018) | 14.31 | 9.49 | 15.80 | 12.61 | 15.55 | 16.41 | 15.97 | 13.65 | 14.99 |
| PCN (Yuan 2018) | 9.64 | 5.50 | 22.70 | 10.63 | 8.70 | 11.00 | 11.34 | 11.68 | 8.59 |
| AtlasNet (Groueix 2018) | 10.85 | 6.37 | 11.94 | 10.10 | 12.06 | 12.37 | 12.99 | 10.33 | 10.61 |
| GRNet (Xie 2020) | 8.83 | 6.45 | 10.37 | 9.45 | 9.41 | 7.96 | 10.51 | 8.44 | 8.04 |
| CDN (Wang 2021) | 8.51 | 4.79 | 9.97 | 8.31 | 9.49 | 8.94 | 10.69 | 7.81 | 8.05 |
| NSFA (Zhang 2020) | 8.06 | 4.76 | 10.18 | 8.63 | 8.53 | 7.03 | 10.53 | 7.35 | 7.48 |
| **SnowflakeNet (Ours)** | **7.21** | **4.29** | **9.16** | **8.08** | **7.89** | **6.07** | **9.23** | **6.55** | **6.40** |

**SnowflakeNet wins on every category** — even on the categories where folding-based methods (AtlasNet) and PCN/NSFA (coarse-to-fine) historically had categorical edges. The −10.5% over NSFA on average CD is the headline number. The visual results (Fig. 6) show: smoother car bodies, *much* cleaner chair backs (where CDN fails and others generate noise between the columns), and better-preserved table leg geometry.

**Note: PoinTr (paper 008) is *not* in this table** — they were concurrent at ICCV 2021. PoinTr reports on PCN separately: in the original paper (Table 2) PoinTr's PCN average CD is 8.38, **worse than SnowflakeNet's 7.21**. So SnowflakeNet beats PoinTr on PCN. (PoinTr's strength was the *diverse* ShapeNet-55 benchmark, not the toy 8-category PCN.)

### Table 2 — Completion3D (per-point L2 CD × 10⁴, lower is better)

| Method | Average | Plane | Cabinet | Car | Chair | Lamp | Couch | Table | Boat |
|---|---|---|---|---|---|---|---|---|---|
| FoldingNet | 19.07 | 12.83 | 23.01 | 14.88 | 25.69 | 21.79 | 21.31 | 20.71 | 11.51 |
| PCN | 18.22 | 9.79 | 22.70 | 12.43 | 25.14 | 22.72 | 20.26 | 20.27 | 11.73 |
| TopNet | 14.25 | 7.32 | 18.77 | 12.88 | 19.82 | 14.60 | 16.29 | 14.89 | 8.82 |
| SA-Net | 11.22 | 5.27 | 14.45 | 7.78 | 13.67 | 13.53 | 14.22 | 11.75 | 8.84 |
| GRNet | 10.64 | 6.13 | 16.90 | 8.27 | 12.23 | 10.22 | 14.93 | 10.08 | 5.86 |
| PMP-Net | 9.23 | 3.99 | 14.70 | 8.55 | 10.21 | 9.27 | 12.43 | 8.51 | 5.77 |
| **SnowflakeNet (Ours)** | **7.60** | **3.48** | **11.09** | **6.9** | **8.75** | **8.42** | **10.15** | **6.46** | **5.32** |

−17.3% over PMP-Net (the prior Completion3D SOTA). Wins on every category except Lamp (where SA-Net/GRNet are competitive at 10.22 / 13.53 — the Lamp category is the standard failure mode for all completion methods because the thin tubes are not in the partial input).

### Table 3 — Skip-Transformer ablation (Completion3D val)

| Variant | avg. | Couch | Chair | Car | Lamp |
|---|---|---|---|---|---|
| Self-att (replaces skip-XFMR with self-attention) | 8.89 | 6.04 | 10.9 | 9.42 | 9.12 |
| No-att (removes attention, just adds features) | 9.30 | 6.15 | 11.2 | 10.4 | 9.38 |
| No-connect (removes the skip entirely) | 9.39 | 6.17 | 11.3 | 10.5 | 9.51 |
| **Full (transformer-based skip)** | **8.48** | **5.89** | **10.6** | **9.32** | **8.12** |

Three findings: (a) the **skip connection alone** is worth ~0.91 CD (No-connect 9.39 → No-att 9.30 is small, but Full → No-att is 0.82), (b) **attention > no-attention** in the skip module (8.48 vs 9.30, ~0.82 CD), (c) **cross-layer transformer (Full) > within-layer self-attention (Self-att)** — 8.48 vs 8.89. The cross-layer information is the unique contribution; in-layer self-attention is similar to what PoinTr (paper 008) does. SnowflakeNet's win is essentially "use the previous layer's displacement, not just the current point's neighbors."

### Table 4 — Component ablation

| Variant | avg. CD | Notes |
|---|---|---|
| Folding-expansion (replace SPD with FoldingNet-style 2D grid) | 8.80 | The folding-based decoder is the dominant baseline approach. |
| E_PCN + SPD (PCN encoder + SnowflakeNet decoder) | 8.93 | A "swap the encoder" experiment — the decoder is what matters. |
| w/o partial matching loss | 8.50 | The preservation loss matters only slightly (8.48 → 8.50). |
| PCN-baseline (PCN trained from scratch on Completion3D) | 13.30 | Reference for the "no SPD" lower bound. |
| **Full** | **8.48** | |

Two takeaways: (1) the **point-wise splitting operation** beats folding (8.48 vs 8.80, 0.32 CD) — small but real; (2) the **encoder choice barely matters** (E_PCN + SPD 8.93 vs Full 8.48), confirming the paper's claim that SPD is the contribution.

## Connections to our hypotheses (H1–H5)

- **H1 (2-stage: segmentation + generation > end-to-end):** Mild support. SnowflakeNet is a *single-stage* completion model (one forward pass, partial → complete), but its internal organization is clearly 2-stage: **(a) feature extraction + seed generation** produces the global "sketch" of the missing part (the 512-point seed P_0), and **(b) the SPD stack** refines and densifies it. The architecture argument: separate "where is the missing volume" from "what does the surface look like" — and SnowflakeNet's 4-term CD loss supervises *both* the coarse and fine stages. **Takeaway:** H1's segmentation+generation split is consistent with SnowflakeNet's coarse+fine split, but the *segmentation* step in our pipeline (paper 001) provides what SnowflakeNet's feature extractor has to learn from scratch (where the missing region is). **For us:** keep the 3DTeethSeg22 segmentation front-end and run SnowflakeNet only on the masked region — don't try to teach one model to do both.

- **H2 (Diffusion on point clouds > mesh-based VAE):** Mild contradiction, mild support. SnowflakeNet is **deterministic, single-pass** (no diffusion, no VAE) — but the SPD + skip-transformer stack effectively *is* a hierarchical generative process: it decomposes the joint distribution of N points into a product of conditional distributions P(P_i | P_{i-1}). This is the *autoregressive* interpretation, not the *diffusion* interpretation, and the paper's results suggest that for **point cloud completion** (where the partial input fixes most of the global structure), the autoregressive + skip-XFMR path is stronger than the unconditional diffusion prior would be. **For us:** if we want **multi-modal completions** (paper 005 LION's selling point), keep a diffusion model in the toolbox — but for the v0 prototype where the dentist wants *one* best crown, SnowflakeNet's deterministic + locally-structured approach is the right call.

- **H3 (Conditioning on opposing + adjacent teeth improves outer surface quality):** Strong support. The skip-transformer's central purpose is exactly this: **the previous layer's displacement features are the "context" that conditions the current layer's splitting**. The "context" here is intra-shape local structure (parent-child in the snowflake tree), but the architecture is one attention head and one MLP from intra-shape context to **inter-tooth** context (the feature extractor already encodes the partial arch). **For us:** the natural extension is to inject **an inter-tooth attention** between (a) the SPD's "child" features for the missing tooth and (b) the partial scan's features for the **adjacent + opposing teeth** — the paper shows attention over kNN neighbors is sufficient for local structure; we'd add a second attention over the FDI-aligned neighbor teeth. The 16-dim FDI embedding suggested in the previous digest (paper 008) becomes a **per-tooth positional encoding** for this second attention.

- **H4 (Implicit SDF > explicit mesh for high-quality surfaces):** Refines. H4 has been holding up beautifully for the **field representation** (DeepSDF, DiGS, Diffusion-SDF) but this paper makes a strong complementary point: **for the *local* surface quality, the *generation* path matters as much as the *representation* path.** The implicit-SDF stack we've built (DiGS + Diffusion-SDF + FlexiCubes, papers 003/004/007) would inherit the same "unstructured local prediction" problem that this paper is solving for explicit point clouds — the DiGS SIREN output + FlexiCubes extractor can't produce a **locally structured** surface if the diffusion prior hallucinates the same unorganized local patterns. **For us:** the SPD + skip-transformer idea generalizes to **SDF fields** as a "neighborhood-conditioned SIREN update" — the SIREN's output at point x is the displacement, the skip-transformer pulls context from the previous layer's output in the same local region, and the per-point kernel interpretation carries over to per-3D-location kernels. Worth a note: this would be a *novel* paper-worthy contribution (SDF-SPD) and we should check whether any of the SPD follow-ups (e.g., "Decomposition of point cloud completion" or Wang 2022) have already done it. The TPAMI 2023 extension applies SPD to *generation* (unconditional) but not to *SDFs*.

- **H5 (Synthetic data can bootstrap training):** Strong support. All training is on PCN (synthetic ShapeNet renderings of 8 categories), with no real intra-oral scan data. The KITTI experiment in the TPAMI extension transfers synthetic → real LiDAR. **For us:** SnowflakeNet is the cheapest completion model in our reading list to *fine-tune on a small (100–500) dental dataset* — the encoder is the only thing that needs adaptation (the SPD decoder is shape-agnostic), and 300 epochs on 30k synthetic arches runs in ~6 hours on a single V100. So the synthetic-bootstrap → real-fine-tune path is **directly validated**.

## Surprises / interesting things buried in the paper

1. **The 4-term CD loss.** The model is supervised at *every* resolution (P_c, P_1, P_2, P_3). This is a stronger training signal than the standard "supervise the final output only" used in PCN/NSFA/SA-Net. The ablation isn't in the main paper, but the architecture obviously benefits from it — the 4 GT densities (downsampled to {256, 512, 2048, 16384}) are computed offline and compared at each level. **For us:** this is a free training-time regularization for *any* coarse-to-fine generation model. Apply it to LION (paper 005) and Diffusion-SDF (paper 004) — supervise the latent decoder at *multiple* diffusion timesteps, not just the final one.

2. **k=8 for kNN in the skip-transformer is suspiciously small.** A kNN of 8 in a 2048-point cloud is essentially nearest-neighbor. The paper doesn't ablate k, but the result is the model is *extremely* local — it doesn't see a large neighborhood context. For a tooth crown with ~32k points and an occlusal surface that's < 100 points across, k=8 means the model only sees 1 cusp's worth of context. **For us:** when adapting SPD to teeth, increase k to ~16–24 (or use radius-kNN at ~0.5mm).

3. **The kernel-interpretation in section 3.2 is mostly rhetoric.** The math is real (Eq. 2 is a 1D deconvolution with shared kernels), but the paper never visualizes what individual kernels learn. The TPAMI extension has *some* kernel analysis (appendix), but the ICCV paper leaves it as a metaphor. **For us:** worth visualizing for our teeth-trained kernels — do they learn "buccal cusp" / "lingual cusp" / "central fossa" primitives? If yes, that's a strong interpretability win for clinical adoption.

4. **The ablation of "PCN encoder + SPD decoder" is a clean experiment.** It shows that the decoder is the contribution, not the encoder. Implication: if we have a stronger encoder (e.g., a 3DTeethSeg22-trained Point Transformer, paper 001), we can drop it in front of SPD with essentially no friction. The current encoder is vanilla PointNet++ set abstraction + point-transformer, which is not the strongest 3D encoder in 2026.

5. **SnowflakeNet doesn't use PointNet++ for the point-wise MLP inside SPD — it uses a "basic PointNet"** (per-point MLP + max-pool). This is the "Set Transformer" / "point cloud → feature vector" primitive. It's an indicator that SPD doesn't need DGCNN-style local feature aggregation inside the splitting — the skip-transformer is doing that job externally.

6. **There's a 3DTeethSeg connection buried in the related work.** The PoinTr authors cite a paper by Cui et al. ("TSegNet" or similar) for the **partial matching loss** (reference [48] = "Unsupervised Learning of Fine Structure Generation for 3D Point Clouds by 2D Projection Matching", CVPR 2021, by Chen, Han, Liu, Zwicker). This is the same group as the 3DTeethSeg challenge (paper 001). So the loss we identified as "preservation" in paper 008 (PoinTr) is in fact borrowed from the same Tsinghua / dental-research cluster — there's a tight community here.

7. **No ShapeNet-55 numbers in the ICCV paper.** This is the surprising part — PoinTr's headline result was on ShapeNet-55/34, but SnowflakeNet stayed on PCN and Completion3D. The TPAMI 2023 extension adds ShapeNet-34/21 unseen-class completion (so the *diversity* story is told in the journal version). The ICCV paper is laser-focused on **local detail**, not **category diversity**. A clean sign that the two papers are attacking different axes: PoinTr = diversity, SnowflakeNet = local detail.

## Quote-worthy sentences

- *"The biggest problem is that these methods only focus on the expansion of point number and the reconstruction of global shape, while ignoring to preserve a well-structured generation process for points in local regions. This makes these methods difficult to capture local detailed geometries and structures of 3D shape."* (Section 2, page 4)
- *"Our insight of revealing detailed geometry is to introduce skip-transformer in SPD to learn point splitting patterns which can fit local regions the best."* (Section 1, page 3)
- *"Each kernel K_m indicates a certain shape characteristic, which describes the geometry and structure of 3D shape in local region. Correspondingly, every logit h_{j,m} indicates the activation status of the m-th shape characteristic."* (Section 3.2, page 5)
- *"SnowflakeNet can interpret the generation process of complete point cloud into an explicit and locally structured pattern."* (Section 1, page 3)
- *"SnowflakeNet models the generation of point clouds as the snowflake-like growth of points in 3D space, where the child points are progressively generated by splitting their parent points after each SPD."* (Abstract, page 1)

## Code / data availability

- **Code:** https://github.com/AllenXiangX/SnowflakeNet — MIT licensed, full PyTorch implementation with pretrained weights. The repo has separate folders for completion, generation, single-view reconstruction, and point cloud upsampling. Requires `torch==1.7.1+cu110` and the included `pointnet2_ops_lib` / `Chamfer3D` / `emd` C++ extensions.
- **Jittor port:** https://github.com/AllenXiangX/SPD_jittor (faster on Chinese hardware).
- **Pretrained weights:** Google Drive link in the README (also a Baidu Pan backup).
- **Datasets:** PCN (Yuan 2018) and Completion3D (Tchapmi 2019) — both derived from ShapeNet. The TPAMI 2023 extension adds ShapeNet-34/21, ShapeNetCars, KITTI fine-tuning scripts.

## For our project (concrete next steps)

1. **Adopt SPD (the operation) as the v0 generation backbone for sub-task 4 (outer surface).** It directly addresses the "cusps + fissures + sharp features" requirement that PoinTr (paper 008) does not. The decoder is shape-agnostic, so we can drop in a stronger 3DTeethSeg22-trained Point Transformer encoder. Compute estimate: 300 epochs × 30k synthetic arches × single V100 ≈ 5–6 hours, ~$15–20 on Lambda.

2. **Bump kNN-k to 16–24 for tooth data.** Default k=8 in SPD is calibrated for 2k-point completion; tooth crowns have ~32k points and an occlusal surface ~100 points across. Smaller k means a cusp sees only its immediate neighbor points and won't learn the global occlusal table pattern.

3. **Add a 4-term CD loss (or analogous multi-resolution supervision) to the v0 pipeline.** This is the "supervise every layer" trick that makes SPD train stably. The equivalent for implicit-SDF (Diffusion-SDF, paper 004) would be supervising the diffusion model at multiple timesteps, not just the final ε-prediction.

4. **Add an inter-tooth attention head to the skip-transformer.** SPD's existing skip-transformer is intra-shape (previous layer's displacement features). Add a second attention from each "child point feature for the missing tooth" to the **FDI-aligned adjacent + opposing teeth's point features** — the H3 conditioning mechanism. 16-dim FDI embedding (paper 008's suggestion) becomes a per-tooth positional encoding.

5. **Investigate the "kernels-as-shape-primitives" interpretation for clinical interpretability.** Visualize what the SPD kernels learn on tooth data. If they learn cusps / fossae / ridges, this is a strong argument for clinical adoption ("the model has learned the same morphological vocabulary as a dental anatomy textbook").

6. **Compute budget for v0 pilot on real teeth:** 6h training (synthetic arches) + 2h fine-tuning (100 real patient arches from 3DTeethSeg22) + 1h inference (FlexiCubes mesh extraction) ≈ 9h on V100 ≈ $30. Add another $30 for ablations. Total v0 pilot: **~$60 compute** — cheaper than every other architecture in our reading list. **Promote SnowflakeNet to a top-3 v0 candidate alongside LION (paper 005) and Diffusion-SDF (paper 004).**

7. **Open question for HK:** SnowflakeNet is the right pick for v0 if (a) we want fast iteration, (b) we don't need multi-modal completions, (c) we trust the *encoder* to find the missing region (which means the 3DTeethSeg22 segmentation front-end does the work). If we want diffusion sampling at inference (for the "give the dentist 3 options" UX), LION or Diffusion-SDF is the pick. **Recommendation: pilot both, in parallel, on the same 100-tooth subset. Total pilot: ~$150 compute, 1 week wall-clock.**

8. **Next paper to read:** **SeedFormer** (Zhou, Wang, Liu, Sun, Wang, Zhang — ICCV 2022, arXiv:2207.10399) — the direct successor to SnowflakeNet that adds a **patch-seed-based** strategy with **self-attention on patches** and **Voxel-Set-Attention** for capturing local context. SeedFormer's claim is the same "local detail" story but with a more scalable neighborhood (patches instead of kNN on raw points). Should be paper 010. Alternatively, **SVDFormer** (ICCV 2023) if we want a 2023-era point of comparison.

---

*Read 2026-06-06 by Scholar. Connections to H1–H5 incorporated. Diff with paper 008 (PoinTr): both are point cloud completion, but PoinTr is *diverse-category single-pass* (transformer, 1.09 CD-Avg on ShapeNet-55) and SnowflakeNet is *single-category locally-structured coarse-to-fine* (SPD, 7.21 on PCN — *better than PoinTr's 8.38 on the same benchmark*). They are complementary, not substitutes.*
