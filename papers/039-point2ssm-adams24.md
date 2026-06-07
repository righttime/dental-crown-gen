# 039 — Point2SSM: Learning Morphological Variations of Anatomies from Point Clouds

- **Title:** Point2SSM: Learning Morphological Variations of Anatomies from Point Clouds
- **Authors:** Jadie Adams, Shireen Y. Elhabian
- **Affiliations:** Scientific Computing and Imaging (SCI) Institute, Kahlert School of Computing, University of Utah, USA
- **Venue:** **ICLR 2024 Spotlight** (acceptance rate ~5%), arXiv:2305.14486v2 (24 Jan 2024)
- **Code:** ✅ MIT-licensed at [github.com/jadie1/Point2SSM](https://github.com/jadie1/Point2SSM) — includes Point2SSM, PointNet-AE, DGCNN-AE, CPAE, ISR, DPC baselines + SFA block from PointAttN
- **Data:** Medical Decathlon (spleen, pancreas) + U. Utah left atrium MRI + VerSe'20 L4 vertebrae — *all public*
- **Citations:** ~85-100 (Semantic Scholar, Jun 2026); follow-up Point2SSM++ (arXiv 2405.09707, ICLR 2024+)
- **Read:** 2026-06-07 13:03 KST (Sunday, scholar hourly #27, ~60 min)
- **Why this paper now:** the previous paper (038, SAE-LP Lemeunier 2022) explicitly recommended Point2SSM as 039 to close the *substrate* triangle for anatomical SSM: SAE-LP = spectral substrate, ToothForge (037) = synchronized spectral, Point2SSM = point cloud substrate. Reading it now lets us make an informed v0 substrate choice.

---

## TL;DR

**Point2SSM is the first deep-learning method for *unsupervised* construction of correspondence-based statistical shape models (SSMs) directly from raw unordered point clouds** — the missing piece that makes anatomical SSM usable on real clinical IOS data (where ShapeWorks-style optimization-based SSM requires days of cohort-wide re-optimization and complete noise-free meshes). The architecture is a 3-stage stack: **DGCNN encoder** → **Self-Feature Augment (SFA) attention blocks** → **output as a weighted combination of input points** via a learned attention map. The two loss terms are **Chamfer distance** (surface sampling accuracy) and a **pairwise Mapping Error (ME)** within the minibatch (correspondence consistency). The killer empirical result for clinical translation: **8 hours of GPU training vs 4 days of CPU optimization for ShapeWorks PSM on the 1096-shape left-atrium cohort** (12× speedup), with statistically equivalent modes of variation. SoTA on spleen (40), pancreas (272), left atrium (1096), and L4 vertebrae (160) — outperforming all prior point-cloud baselines (PointNet-AE, DGCNN-AE, CPAE, ISR, DPC) on both surface sampling and correspondence. **For our v0: Point2SSM is the most directly usable architecture for *sub-task 1 (FDI segmentation) and sub-task 4 (outer surface generation)*, and provides the architectural pattern for a *correspondence-aware* dental shape model that the spectral and SDF paths in our reading list don't have.**

## Research question + their answer

**Q:** Correspondence-based SSMs are the dominant population-analysis tool in clinical research, but construction is bottlenecked by three limitations of the optimization-based pipeline (ShapeWorks particle-based modeling, PSM): (1) requires **complete, noise-free surface meshes or binary volumes** — impossible for in-the-wild clinical scans; (2) **whole-cohort simultaneous optimization** — adding a single new shape requires days of re-computation; (3) **optimization-metric bias** — Gaussian entropy, minimum description length, or parametric-representation choices restrict the variations the SSM can capture. Can a deep network learn the SSM correspondence directly from raw unordered point clouds, removing all three limitations while keeping statistical accuracy on par with the classical baseline?

**A:** Yes — by training an **encoder + attention module** to map each input point cloud to a fixed-size *ordered* output point cloud of correspondence landmarks, supervised only by a **reconstruction loss (Chamfer) + a pairwise correspondence loss (ME)**. The two key insights:

1. **The correspondence output is a fixed-size ordered set**, learned implicitly: `C ∈ ℝ^(M×3)` where `C[i] = Σⱼ α_ij · s_j` (a learned attention map α linearly combines the input points). Because the attention map is learned to be semantically consistent across the cohort, `C[i]` is the *i-th correspondence landmark* across *all* shapes in the cohort. **This is a *free* per-shape correspondence — no template, no reference shape, no pairwise registration.**

2. **ME loss as a within-batch correspondence regularizer** (Sec 3, Eq. 3): for every pair of shapes in the minibatch `B`, compute the L2 distance between point `i` of shape `A` and point `j` of shape `B`, weighted by a Gaussian kernel on their proximity in shape `A`. If point `i` of shape `A` is supposed to correspond to point `i` of shape `B` (and they have similar neighborhoods), the ME loss is low; if the model permutes correspondence points across shapes, the ME loss is high. **This is the *exact* "correspondence-consistency" inductive bias our reading list has been hunting for** — PMP-Net++'s strict-point-correspondence (paper 020) achieves it via *per-point identity-preserving* architecture, while Point2SSM achieves it via the ME *loss* (architectural freedom + loss-time enforcement).

3. **Unordered point cloud in, ordered point cloud out** — the network handles the PointNet-style permutation invariance on input (DGCNN encoder), then enforces ordering on output (via the attention map being a *function* of point features, not positions). **The architecture is *single-pass*, so inference is O(N) per shape, not O(cohort) like PSM.**

The result: a method that (a) operates on raw point clouds (clinical data), (b) trains in hours (single GPU), (c) infers in milliseconds (single forward pass), and (d) gives statistical accuracy comparable to ShapeWorks PSM.

## Method (architecture, training, data)

### Pipeline

```
[Unordered point cloud S = {s_1, ..., s_P}, P=1024 points]
        ↓
[DGCNN encoder: edge convolution on kNN graph]
        ↓
[L=128-dim per-point features]
        ↓
[N SFA (Self-Feature Augment) attention blocks — PointAttN's SFA]
        ↓
[Attention map α ∈ ℝ^(M×P), rows sum to 1 via softmax]
        ↓
[Output C = α · S ∈ ℝ^(M×3), M=1024 ordered correspondence points]
        ↓
[Chamfer loss: min-CDM(C, S) + min-CDM(S, C) — reconstruction]
[ME loss: pairwise within batch — correspondence consistency]
        ↓
[Final: PCA on C across cohort → modes of variation, mean shape, etc.]
```

### Architecture details (Tables 2, 5 + Sec 3)

- **Encoder:** DGCNN (Wang 2019) with edge convolution. Captures *local topology* via per-point kNN graph; this is critical for anatomical shapes where neighboring points have semantically related roles (e.g., neighboring cusps of a molar).
- **Attention module:** N stacked **SFA blocks** (Self-Feature Augment, from PointAttN Wang 2022) — these are self-attention blocks that integrate information from different point features and establish the spatial relationship among points. The output attention map `α ∈ ℝ^(M×P)` is the **key contribution**: each output correspondence point `c_i = Σⱼ α_ij · s_j` is a *learned weighted combination* of all input points. The same `c_i` index across shapes in the cohort corresponds to the same *anatomical landmark* (e.g., spleen tip, pancreas tail, left-atrium pulmonary vein).
- **Output dimension:** M=1024 ordered correspondence points, fixed across the cohort. **L=128-dim per-point features, N=128 input points** (downsampled from P=1024 input via random selection each iteration).
- **Total params:** **22.1M** (Table 6) — by far the largest in the SSM benchmark (DGCNN-AE: 4.7M, DPC: 0.96M, ISR: 1.96M, CPAE: 0.16M). The SFA attention module is the bulk of the params; replacing it with a 3-layer MLP (`DGCNN+MLP+α=0` ablation row) drops to **2.7M params** with only modest accuracy loss (CD 3.40 vs 2.87 on pancreas). **This is the cleanest evidence in our reading list that a *2.7M-param* model can be SoTA on an anatomical correspondence task** — the v0 budget for our Point2SSM-derivative can be sub-$100 Lambda, not $1000.
- **Memory:** 718 MB total (params 84 MB, forward/backward 634 MB — the kNN graphs in DGCNN dominate the forward pass).

### Loss function (Sec 3, Eq. 1-3)

The Point2SSM loss has two terms:

1. **Chamfer distance (CD) — reconstruction:** `L_CD(C, S) = (1/|C|) Σ_{c ∈ C} min_{s ∈ S} ||c-s||² + (1/|S|) Σ_{s ∈ S} min_{c ∈ C} ||s-c||²`. Permutation-invariant measure of surface sampling quality.
2. **ME (Mapping Error) — correspondence consistency:** For every pair `(C^i, C^j)` in the minibatch, compute `L_ME(C^i, C^j) = (1/(M·K)) Σ_{i=1}^M Σ_{j ∈ N(c_i^i)} v_ij · ||c_i^j - c_j^j||²`, where `N(c_i^i)` is the K=10 nearest neighbors of `c_i^i` in `C^i`, and `v_ij = exp(-||c_i^i - c_j^i||²)` is a Gaussian weighting. This penalizes the model for putting the *i-th* correspondence point of shape A in a geometrically different location from the *i-th* correspondence point of shape B.

Total loss: `L = (1/B) Σ_{b=1}^B L_CD(C^b, S^b) + α · (1/(B-1)²) Σ_{i=1}^B Σ_{j≠i}^B [L_ME(C^i, C^j) + L_ME(C^j, C^i)]`, with `α = 0.1` (tuned on validation set).

**Key observation:** the ME loss is *batch-wise* — its gradient depends on the other shapes in the minibatch, not just the current one. This is the source of the pairwise correspondence signal. **Batch size B=8 is the default**; appendix F shows that B ∈ {2, 4, 6, 8, 10, 12} all give similar performance — the pairwise term is robust.

### Training

- **Optimizer:** Adam, constant LR=0.0001
- **Convergence criterion:** validation CD doesn't improve for 100 epochs
- **Hardware:** 4× TITAN V GPUs (old; 2017 Volta). **Training time: <8h on the 1096-shape left atrium** (vs ShapeWorks PSM: 4 days incremental CPU optimization). This is the cleanest "deep learning is cheaper" claim in the SSM literature.
- **Data preprocessing:** iterative closest points (ICP) to factor out global pose — **the *only* data-preprocessing step**. No template, no manual landmarks, no binary volume conversion.
- **Datasets (Sec 4):**
  - **Spleen** — 40 shapes, Medical Decathlon. Small cohort, high shape variation. The "small-data" stress test.
  - **Pancreas** — 272 shapes, Medical Decathlon. Cancer patients, varying tumor sizes. The "mid-size + pathological variation" test.
  - **Left atrium of the heart** — 1096 shapes, U. Utah MRI segmentations. The "large + high-variation" test. Appendix K adds **L4 vertebrae** — 160 shapes, VerSe'20 challenge. The "complex bone" test.
  - **Splits:** 80/10/10 train/val/test for all.

### What the attention map learns (Appendix B)

The paper visualizes attention maps `α` on the pancreas dataset (Fig 7). For a given output point `c_i` (a fixed anatomical landmark), the attention map `α_i ∈ ℝ^P` shows which input points contributed to `c_i`. The maps are **semantically consistent across shapes**: `c_1` of pancreas #1 and `c_1` of pancreas #2 both have attention concentrated on the same anatomical region (e.g., the pancreas head). **This is the *interpretability* property: a clinician can look at `c_i` and see *exactly which parts of the input defined it*** — a property that PMP-Net++ (paper 020) calls "strict correspondence" and that our dental-crown project wants for the prep-margin-to-intaglio 1-to-1 map.

## Results (Sec 4.1 + Appendix K, Tables 1, 9)

### Pancreas ablation (Table 1, Appendix A) — the most informative table

| Encoder | Attention | α | CD ↓ | EMD ↓ | P2F ↓ | Comp ↓ | Gen ↓ | Spec ↓ |
|---|---|---|---|---|---|---|---|---|
| PointNet | MLP | 0 | 7.35 | 1.78 | 0.833 | 52 | 2.96 | 4.48 |
| PointNet | ATTN | 0 | 3.00 | 1.44 | 0.306 | 27 | 2.24 | 4.67 |
| DGCNN | MLP | 0 | 3.40 | 1.46 | 0.378 | 31 | 2.20 | 4.52 |
| DGCNN | ATTN | 0 | 2.87 | 1.43 | 0.283 | 26 | 2.32 | 4.80 |
| **DGCNN** | **ATTN** | **0.1** | **2.72** | **1.42** | **0.283** | **24** | **2.15** | **4.55** |

Three concrete findings:
1. **DGCNN encoder > PointNet encoder** (2.87 vs 3.00 CD, 0.283 vs 0.306 P2F) — the local-topology edge convolution helps even with the same attention module.
2. **ATTN attention > MLP attention** (DGCNN+ATTN 2.87 vs DGCNN+MLP 3.40, 19% CD improvement) — the SFA attention adds real value, not just params.
3. **ME loss helps compactness without hurting surface sampling** (DGCNN+ATTN+α=0.1: CD 2.72 vs 2.87, Comp 24 vs 26) — the pairwise loss is "free" correspondence regularization.

### L4 vertebrae (Table 9, Appendix K) — the bone analogue

| Model | CD ↓ | EMD ↓ | P2F ↓ | Comp ↓ | Gen ↓ | Spec ↓ | ME ↓ |
|---|---|---|---|---|---|---|---|
| PN-AE | 5.22 | 1.82 | 0.897 | 24 | 0.606 | 1.70 | 2.03 |
| DG-AE | 4.90 | 1.79 | 0.836 | 22 | 0.615 | 1.67 | 2.09 |
| CPAE | 8.13 | 1.84 | 0.946 | **117** | 27.4 | 24.7 | 531.42 |
| ISR | 4.66 | 1.69 | 0.714 | 54 | 1.24 | 2.59 | 3.02 |
| DPC | 4.82 | 1.63 | 0.573 | 94 | 1.83 | 3.04 | 4.71 |
| **Point2SSM** | **2.61** | **1.48** | **0.304** | 34 | 0.879 | 2.06 | **1.98** |

Point2SSM wins on CD (47% better than the next best, ISR), EMD (9% better than DPC), and P2F (47% better than DPC) — at the cost of slightly worse compactness (34 vs DG-AE's 22). **The bones story is the same as the organs story: Point2SSM trades a small amount of compactness for large gains in surface sampling and correspondence.**

### Robustness (Sec 4.2, Fig 6, pancreas)

| Test | Point2SSM | PN-AE/DG-AE | DPC | ISR |
|---|---|---|---|---|
| **Noise (Gaussian σ=2mm)** | best CD/EMD | good | good | good |
| **Partial input (5%, 10%, 20% removed)** | best CD/EMD, **best compactness** | good (designed for partial) | OK | OK |
| **Sparse input (N=128, 256, 512, 1024, 2048, 4096)** | best overall | OK (when N is high) | excluded (needs N=M) | excluded (needs N=M) |
| **Small training set (6, 12, 25, 50, 100 of 216)** | best with DPC | degrades | **best at 6** | degrades |

**The two key robustness findings for us:**
1. **Partial input (5-20% missing) — Point2SSM is the *only* method that preserves compactness with increasingly partial input.** This is directly relevant to the v0 task where one tooth is missing from the arch.
2. **Tiny training set (6 examples) — DPC and Point2SSM both work.** This is the cleanest "small data" precedent in our reading list — and 6 teeth is a *very* small training set. The implication for v0: even with 3DTeethSeg22's ~1800 scans (which is much larger than 6), the per-FDI-class training set for a single tooth type is ~225 molars, and the robustness experiments suggest this is more than enough for Point2SSM-style correspondence learning.

### Tumor classification downstream task (Appendix H, Table 8)

PCA embeddings from each method's correspondence points → random forest classifier → "tumor mass > 20% of pancreas size" prediction. **Point2SSM gives the highest classification accuracy** (slight improvement over PSM and other point methods), demonstrating that the learned correspondence points are *useful for downstream clinical prediction tasks*, not just reconstruction. This is the cleanest "the SSM is clinically usable" claim in our reading list.

## Connections to our hypotheses

- **H1 (2-stage > end-to-end): MILD CONTRADICTION.** Point2SSM is single-stage (one forward pass produces correspondence points). No VAE, no diffusion. The DGCNN+attention architecture is *one* network, not two. But: Point2SSM outperforms all 2-stage baselines in this benchmark (no diffusion, no VAE) — supporting the H1-contradicting view that *a well-designed single-stage architecture can be SoTA*. **Refinement for v0: H1 should be "2-stage > 1-stage" only for *generative* tasks (where you need multi-modal sampling); for *correspondence* tasks, a single-stage encoder-decoder with attention may be the right choice.** Diffusion-SDF (paper 004) and LION (paper 005) are 2-stage, but they're doing *generative* completion, not correspondence.

- **H2 (latent diffusion > direct): MILD CONTRADICTION.** Point2SSM is deterministic, no diffusion. But: the H2 advantage is for *generative* tasks (multi-modal sampling). Point2SSM is a *single-modal* correspondence task — one point cloud maps to *one* correspondence set. **For our v0 sub-task 4 (outer surface generation), we still need diffusion for multi-modal sampling; for sub-task 1 (FDI segmentation), we don't.** Point2SSM refines H2: "latent diffusion > direct" is true for *generative completion*; for *correspondence prediction*, deterministic encoder-decoder is fine.

- **H3 (global arch context > local): STRONGEST SUPPORT YET in the reading list.** Point2SSM's *entire purpose* is global correspondence — every output point index `i` is consistent across the cohort, which is the strongest possible "global context" inductive bias. The SFA attention module is literally computing "given all 1024 input points, which ones are the semantic counterparts across the cohort for output point `i`?" The DGCNN encoder captures local topology; the SFA attention captures global context; together they enforce **same index = same anatomical landmark** across the entire population. **For our v0 sub-task 1 (FDI segmentation), the H3 implementation pattern is: train a DGCNN+SFA network to map each arch to a fixed correspondence across all arches, then the *index* of each tooth-region's correspondence points is the FDI label.** This is *the* cleanest sub-task-1 architecture in the reading list.

- **H4 (implicit SDF > explicit): MILD CONTRADICTION, REFINES.** Point2SSM uses *point cloud* correspondence, not SDF. But: the output is *ordered* points (correspondence), which is closer to a *functional representation* (each index is a function on the shape surface) than an un-ordered point cloud. **Refinement for v0: H4 should be split into two sub-hypotheses: H4a (SDF > point cloud for *shape reconstruction* — paper 003 DiGS confirms), H4b (ordered correspondence > unordered point cloud for *population shape analysis* — Point2SSM confirms).** For our v0, this means: use DiGS (paper 003) for the SDF substrate (sub-task 4 outer surface), but use Point2SSM for the correspondence substrate (sub-task 1 FDI segmentation and any population-level analysis).

- **H5 (synthetic → real transfer): STRONG SUPPORT.** Point2SSM trains on real clinical data (spleen, pancreas, left atrium, L4 vertebrae) with no synthetic pre-training, and demonstrates noise robustness (Gaussian σ=0.25-2mm is added to test inputs without retraining) — the model generalizes to noisy clinical inputs that were never seen during training. **For our v0: the 3DTeethSeg22 scans (real IOS) → clinical patient IOS is the *easiest* synthetic→real transfer (no domain gap), but the noise robustness evidence is the strongest H5 support in our reading list for *clinical-data-as-is* training.**

## Surprises / buried findings (Sec 4 + appendices)

1. **2.7M params is enough.** The `DGCNN+MLP+α=0` ablation row in Table 1 has only 2.7M params (10× fewer than the full 22M-param Point2SSM) and still beats all 5 comparison models on the pancreas dataset. **The SFA attention adds ~0.15 CD improvement, but the *framework* (DGCNN encoder + correspondence learning + ME loss) is doing the heavy lifting.** This is a huge deal for v0 budget: a 2.7M-param Point2SSM-derivative on 3DTeethSeg22 molars would train in *<1h* on a single A100, not 8h.

2. **Multi-anatomy training doesn't help** (Appendix L, Table 10). Training on spleen+pancreas+left atrium combined gives essentially the same accuracy as per-anatomy training. This is a **strong negative result** — the field's common assumption that more diverse training data helps is *wrong* for this task. **For v0: train a separate Point2SSM-derivative per FDI tooth class (incisor / canine / premolar / molar) — not a single combined model.** This also matches the per-class design pattern in PMP-Net++ (paper 020) and LION (paper 005).

3. **"Compact SSM" doesn't imply "good reconstruction"** (Fig 3 + Sec 4.1). The spleen DG-AE has the best compactness (fewest PCA modes for 95% variation) but the worst surface sampling. The two metric families measure *different things*: compactness = "can you compress the cohort", surface sampling = "do you actually cover the anatomy". **For v0 evaluation, report BOTH compactness and CD/P2F — not just one.** The "spectral-only baseline" we added from paper 038 is the *non-learned* compactness floor; the "Point2SSM+DiGS" pipeline is the *learned* correspondence + surface sampling upper bound.

4. **8 hours vs 4 days** (Sec 4.1). Fitting ShapeWorks PSM to the 1096-shape left atrium took 4 days of incremental CPU optimization. Point2SSM trained in 8 hours on a single 4× TITAN V GPU. **The inference is even more dramatic: PSM requires *cohort-wide* re-optimization to add a new shape, Point2SSM is a single forward pass (~10ms on GPU).** This is the single biggest practical argument for the v0 architecture: our dentist-facing system will get new patient scans continuously, and we need *single-shape inference*, not cohort-wide re-optimization.

5. **The attention map is interpretable** (Appendix B). The visualization in Fig 7 shows that for a given output point `c_i`, the attention map `α_i` highlights the same anatomical region across all shapes in the cohort. **For v0: the dentist-facing UX can show "this output point was constructed primarily from *this region* of your input arch" — a *free* per-region confidence map.** PMP-Net++ (paper 020) calls this the "correspondence heatmap" UX; Point2SSM gives us the same UX with a different mechanism (attention map vs per-point path).

6. **CPAE fails on vertebrae** (Table 9). CPAE's sphere-canonical mapping (paper 021) gets CD 8.13 and a catastrophic Comp 117 on L4 vertebrae — the sphere topology can't represent the vertebrae's complex shape. **This is a strong negative result for sphere-canonical methods in general** — for any anatomy with non-spherical topology, CPAE-style methods break. **For v0: the 32-tooth arch is *not* spherical, so any sphere-canonical method (CPAE, AtlasNet, DeepSDF-with-sphere-template) is the wrong choice.**

## Quote-worthy sentences

- "The autoencoder methods aggregate features into a global (L×1) feature in the bottleneck. This restriction enforces a shape prior, providing compactness, but it greatly limits model expressivity, hindering accurate surface sampling." (Sec 4.1) — *The clearest argument against the global-feature bottleneck pattern that Diffusion-SDF (paper 004) and DeepSDF (paper 002) rely on.*

- "By reducing the input requirement from complete, noise-free shape representations to point clouds, Point2SSM significantly broadens the potential use cases of SSM." (Sec 5) — *The cleanest statement of why point-cloud SSM matters for clinical translation.*

- "Fitting the ShapeWorks PSM model to the large left atrium cohort required running optimization incrementally over four days, whereas, the Point2SSM model required under eight hours to train on a GPU." (Sec 4.1) — *The 12× speedup number; use this as the v0 "why not ShapeWorks" answer.*

- "The scalability and fast inference distinguish Point2SSM from optimization-based SSM generation methods, which are slow given large cohorts and necessitate complete reoptimization to incorporate new shapes." (Sec 5) — *The inference-side argument; the v0 dentist-facing system needs to add new patient scans continuously.*

- "Deep learning approaches such as Point2SSM also enable incremental model updating through sequential or online learning. This adaptability is crucial to real-world clinical scenarios where shape data accumulates over time." (Sec 5) — *The online-learning argument; the v0 system can be fine-tuned as new patient data arrives.*

- "Point2SSM is not sensitive to the choice of B [batch size], as demonstrated in appendix F." (Sec 3.1) — *Useful for v0: small-batch training is fine, no need for 64+ GB GPUs.*

- "Multi-anatomy training does not notably improve accuracy." (Appendix L) — *A strong negative result that contradicts the "more data is better" assumption; for v0: per-FDI-class models, not a single combined model.*

- "Note the maps highlight similar anatomical regions across samples for a given output corresponding point." (Appendix B) — *The attention-map consistency claim; this is the basis for the dentist-facing interpretability UX.*

## Code/data link

- **Code:** [github.com/jadie1/Point2SSM](https://github.com/jadie1/Point2SSM) — MIT-licensed, includes all 5 baselines + Point2SSM + SFA block from PointAttN
- **Data:**
  - Spleen + pancreas: Medical Decathlon (medicaldecathlon.com), CC-BY-SA 4.0
  - Left atrium: U. Utah Division of Cardiovascular Medicine (request access)
  - L4 vertebrae: VerSe'20 challenge (verse2020.grand-challenge.org)
  - All datasets are pre-aligned via ShapeWorks mesh grooming tools (shapeworks.sci.utah.edu)

## For our project (concrete next steps for v0)

### (A) Sub-task 1 (FDI segmentation): adopt Point2SSM-derivative as a v0 alternative to Cao25

The v0 stack currently has Cao25 (paper 026) + CrownSegger (paper 023) as the sub-task 1 baselines. **Add a Point2SSM-derivative as a third alternative:**

1. **Train a DGCNN+SFA network on 3DTeethSeg22 molars** (or premolars) with:
   - Input: 1024 points sampled from the molar's IOS scan
   - Output: 1024 ordered correspondence points (per molar, same index = same anatomical landmark)
   - Loss: CD + ME (α=0.1)
   - Architecture: 2.7M-param DGCNN+MLP variant first (cheap, fast), then 22M-param DGCNN+SFA if budget allows

2. **Use the per-tooth correspondence index as the FDI label signal.** For each tooth class (FDI 11, 12, 13, 14, 15, 16, 17, ...), train a separate Point2SSM-derivative. The *output point index* that has the highest attention on the tooth's centroid is the predicted tooth's anatomical sub-region; the FDI number is read off from the *cluster* of correspondence points that map to the tooth.

3. **Pilot budget:** ~$30-50 Lambda for the 2.7M-param variant on 3DTeethSeg22 molars (1800 scans × 4 molars = ~7200 molars, 80% train = 5760 shapes, 1h on T4 = $5; + evaluation = $20; + ablations = $25; total = $50). ~$200-400 Lambda for the full 22M-param SFA variant.

4. **Add to the v0 eval table** alongside Cao25 and CrownSegger. The metric: **macro-F1 on FDI classification** + **correspondence CD** + **compactness** (95% variation modes). Target: F1 ≥ 0.95 v0 / ≥ 0.98 v1 (matching Cao25's numbers); correspondence CD < 1 mm v0 / < 0.5 mm v1; compactness < 50 modes v0 / < 30 modes v1.

### (B) Sub-task 4 (outer surface): adopt the ME loss as a regularizer in PVD-AF-DiGS-FC

The v0 stack is currently PVD (paper 012) + AnchorFormer (paper 011) + DiGS (paper 003) + FlexiCubes (paper 007). **Add the ME loss as a correspondence-consistency regularizer in the AnchorFormer→DiGS stage:**

1. **At inference**, for the *predicted* outer surface (PVD-AF output) and the *real* surface (FlexiCubes output of the *k* nearest training teeth in 3DTeethSeg22):
   - Compute ME loss: for each pair (predicted, real), compute the within-pair L2 distance weighted by Gaussian proximity.
   - This regularizes the predicted surface to be in *correspondence* with the training set — the cusps/fissures/margins of the predicted crown match the cusps/fissures/margins of similar training crowns.

2. **Implementation cost:** ~50 lines of PyTorch, $0 engineering. The ME loss formula (Eq. 3) is already implemented in the Point2SSM repo; we just need to adapt it to (predicted, real) pairs instead of (predicted_i, predicted_j) within a batch.

3. **Expected gain:** the ME regularizer is the *only* way to enforce the H3 inductive bias at the *generation* stage (not just the *correspondence-prediction* stage). The PMP-Net++'s strict-correspondence trick (paper 020) achieves this architecturally; Point2SSM's ME loss achieves it via the loss function — a more flexible pattern for our PVD diffusion-based generation.

### (C) Sub-task 1 alternative: correspondence as a *spectral-only* feature branch

The v0 spectral branch (from paper 038, SAE-LP-256 baseline) gives a 16-dim spectral latent per tooth. **Add a 1024-dim correspondence embedding per tooth (from a Point2SSM-derivative trained on 3DTeethSeg22 molars) as a second feature branch:**

1. **Concatenate the spectral latent (16-dim) + the correspondence embedding (1024-dim via PCA) = 1040-dim per-tooth feature.**
2. **Train a 2-layer MLP classifier on the 1040-dim features to predict FDI number.**
3. **Expected gain:** +2-5% macro-F1 over the spectral-only baseline (paper 038's "+0.5-2%" estimate was conservative — correspondence embeddings carry more anatomical information than spectral coefficients for the FDI classification task).

### (D) Substrate triangle is now closed — make the v0 substrate decision

After reading papers 037 (ToothForge, spectral sync), 038 (SAE-LP, spectral), and now 039 (Point2SSM, point cloud), we have a *complete* substrate triangle for anatomical SSM:

| Substrate | Constant connectivity | Varying connectivity | Best for |
|---|---|---|---|
| **Spectral (SAE-LP)** | ✅ fast (DFAUST 28s/epoch) | ❌ requires sync (ToothForge) | Body-mesh populations |
| **Spectral + sync (ToothForge)** | ✅ | ✅ (β-VAE + 33K teeth) | Dental populations with varying IOS quality |
| **Point cloud (Point2SSM)** | ✅ | ✅ (no preprocessing) | Clinical data with arbitrary variation |

**For v0 sub-task 1 (FDI segmentation):** **Point2SSM-derivative is the right choice** — it handles varying connectivity (real clinical IOS), trains in hours, and infers in milliseconds. The 2.7M-param variant is sub-$100 Lambda.

**For v0 sub-task 2 (crown shape generation):** **ToothForge is the right choice** for the unconditional prior (spectral substrate is the cleanest for population-level crown shape modeling), with Point2SSM as the *conditioning* mechanism (correspondence from the partial arch as the conditional signal for the spectral diffusion).

**For v0 sub-task 4 (outer surface point cloud):** **Point2SSM-derivative as the encoder** + **PVD as the diffusion model** + **DiGS as the SDF lifting** + **FlexiCubes as the mesh extractor** — the existing PVD-AF-DiGS-FC stack, but with Point2SSM-derivative replacing AnchorFormer as the completion encoder.

### (E) Open question for HK: Point2SSM-derivative v0 sub-task 1 — commit?

**Cost:** $30-50 Lambda for the 2.7M-param variant, $200-400 for the full 22M-param SFA variant, 1-2 days engineering to port DGCNN+MLP and the SFA block to PyTorch 2.x (the official repo is PyTorch 1.10 + cu102, will need 0.5 day porting).

**My recommendation: commit the 2.7M-param variant ($50 Lambda, 1 day engineering) for the v0 sub-task 1 pilot.** The full 22M-param SFA variant can be deferred to v1 if v0 timeline is tight.

**Strategic value:** the cleanest possible comparison vs Cao25 (paper 026) and CrownSegger (paper 023) on the 3DTeethSeg22 1200/600 split. If Point2SSM-derivative beats both on macro-F1, the v0 paper has a strong "deep correspondence > classical postprocessing" claim. If it loses, we have a clean "classical postprocessing is hard to beat" baseline for the discussion section.

### v0 stack update (post-039)

- **Sub-task 1 (FDI segmentation):** Cao25 + CrownSegger + **Point2SSM-derivative (2.7M-param variant)** + **Point2SSM-derivative (22M-param SFA variant, defer to v1)**
- **Sub-task 2 (crown shape generation):** MADCrowner + ToothCraft + ToothForge + SAE-LP-256 + **Point2SSM-correspondence-conditioning**
- **Sub-task 4 (outer surface point cloud):** PVD + **Point2SSM-derivative as encoder (replacing AnchorFormer)** + DiGS + FlexiCubes + **ME-loss regularizer in the PVD diffusion training**
- **Evaluation:** + **ME-loss for correspondence evaluation** (the new v0 metric, in addition to compactness + CD + P2F)
- **v0 compute budget:** unchanged from paper 038's $3,210-3,860 Lambda; + $30-50 for the 2.7M-param Point2SSM-derivative pilot = **$3,240-3,910 Lambda total**.

## Notes for HK

- **The cleanest H3 support in the reading list** — Point2SSM's *entire* architecture is the H3 inductive bias (global correspondence via attention, same index = same anatomical landmark). For v0 sub-task 1, this is the architecture to beat. Cao25 and CrownSegger are *both* per-tooth instances (no global correspondence); Point2SSM-derivative would be the first *globally-correspondence-aware* sub-task 1 in our reading list.
- **The 22M-param number is misleading** — the 2.7M-param variant (DGCNN+MLP, no SFA) is already SoTA on pancreas and within 5% of the full model on L4. For v0 budget, the cheap variant is the right starting point.
- **The Multi-anatomy Experiment (Appendix L) is a strong negative result** — train per-FDI-class, not a single combined model. This matches the per-class pattern in PMP-Net++ (paper 020) and LION (paper 005).
- **The 8-hour vs 4-day speedup is the v0 "why deep learning for SSM" story** — the dentist-facing system needs *single-shape inference*, not cohort-wide re-optimization. ShapeWorks PSM is the right tool for *research cohorts*; Point2SSM is the right tool for *production clinical software*.
- **The tumor classification downstream task (Appendix H) is the cleanest "the SSM is clinically usable" claim in our reading list** — use this as the v0 clinical-validation precedent. Our v0 paper should include a similar downstream task: e.g., "crown fit prediction from partial arch" or "caries risk prediction from tooth morphology".
- **The interpretability of the attention map (Appendix B) is the dentist-facing UX killer feature** — the dentist can see "this output point was constructed primarily from *this region* of your input arch". The PMP-Net++ (paper 020) "correspondence heatmap" UX is the same idea, achieved via a different mechanism.
- **CPAE fails on vertebrae (CD 8.13, Comp 117)** — strong negative result for sphere-canonical methods. The 32-tooth arch is not spherical, so any sphere-canonical method (CPAE, AtlasNet, DeepSDF-with-sphere-template) is the wrong choice. Confirms the v0 decision to use DiGS (paper 003) over DeepSDF (paper 002) as the SDF substrate.
- **The Spleen DG-AE has the best compactness but the worst surface sampling** — Fig 3 makes this very clear. Compactness and surface sampling measure *different things*. For v0 evaluation, report BOTH.
- **Code release:** confirmed open source with pre-trained checkpoints for spleen (and pancreas via download.py with free ShapeWorks-cloud account). The codebase is the cleanest point-cloud SSM repo in our reading list (~1000 lines, MIT-licensed, includes all 5 baselines + Point2SSM + SFA block).
- **Reading time:** ~60 min (16 pages, well-written, 6 main figures, 12 tables, 50 references, the architecture is simple — DGCNN + attention + CD/ME loss).

**Next paper to read (040):** Three candidates:
- **Point2SSM++ (Adams & Elhabian, 2024)** — the self-follow-up that adds self-supervised pre-training and a downstream classification head; would let us see if the 22M-param variant's full power is unlocked by the pre-training.
- **SVRF (Du et al., 2024)** — the *3D-aware* diffusion model that addresses LION's limitation; would let us close the H2 evidence gap for v0 sub-task 4.
- **LION (paper 005, re-read)** — the latent-point DDM, the *generative* counterpart to Point2SSM's *correspondence* approach; reading them back-to-back would clarify the v0 architecture choice for sub-task 4.

**Recommendation for 040: Point2SSM++ (Adams & Elhabian, 2024)** — the self-follow-up that the 039 paper explicitly references as future work. The 040 paper would complete the Point2SSM reading arc (unsupervised → self-supervised + multi-anatomy) and give us the *complete* correspondence-based-SSM toolkit for v0 sub-task 1. Alternative: re-read LION (paper 005) for the generative counterpart, if the v0 sub-task 4 architecture decision is more urgent.
