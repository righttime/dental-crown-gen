# 008 — PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers

- **Title:** PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers
- **Authors:** Xumin Yu*, Yongming Rao*, Ziyi Wang, Zuyan Liu, Jiwen Lu†, Jie Zhou (*equal contribution; †corresponding)
- **Affiliation:** Department of Automation, Tsinghua University + State Key Lab of Intelligent Technologies and Systems + Beijing National Research Center for Information Science and Technology
- **Year:** 2021 (ICCV 2021; arXiv:2108.08839 v1, 19 Aug 2021)
- **Venue:** IEEE/CVF International Conference on Computer Vision (ICCV) 2021
- **Links:**
  - Paper: https://arxiv.org/abs/2108.08839
  - Official code: https://github.com/yuxumin/PoinTr
- **Code/data:** PyTorch implementation released by the authors; experiments on PCN (8 cat), ShapeNet-55 (new, 55 cat), ShapeNet-34 (new, 34 seen + 21 unseen cat), and KITTI (real LiDAR cars).

---

## TL;DR

PoinTr reformulates **point cloud completion** as a **set-to-set translation** problem and solves it with a vanilla Transformer encoder–decoder, where the point cloud is represented as a sequence of "point proxies" (FPS-sampled centers + DGCNN local features + position embeddings). A **Geometry-Aware Transformer block** injects a kNN-based geometric inductive bias into self-attention, a **Query Generator** produces *dynamic* decoder queries from the encoder output, and **Multi-Scale Point Generation** (FoldingNet on the predicted centers) upsamples proxies to a full-resolution point cloud. The result is a clean single-pass model that beats GRNet by ~45% on ShapeNet-55 CD-Avg (1.09 vs 1.97) and roughly *doubles* F-Score@1% (0.464 vs 0.238), with strong zero-shot transfer from synthetic ShapeNet to real KITTI LiDAR scans.

## Research question

> Can a Transformer architecture — with no spatial inductive bias by default — match or beat the leading CNN / PointNet++ / folding-based methods on point cloud completion, and can it generalize across **diverse** categories, viewpoints, and incompleteness levels (rather than the 8-category PCN toy setting the field had settled into)?

Their answer: yes, but only if you (1) replace the per-point tokenization with **point proxies** (centers + local features + positions) so the sequence length stays manageable, (2) add a **geometry-aware block** that lets self-attention see local 3D structure, (3) use **dynamic queries** instead of learned fixed queries so the decoder is input-conditioned, and (4) train on a new diverse benchmark they introduce.

## Method

### Architecture overview
1. **Downsample + feature extract:** FPS the input partial point cloud (N=2048 points) to **N=128 center points**. For each center, run a hierarchical DGCNN to extract a local feature (channels: 3→8→32→64→64→128). Add a position embedding (an MLP on the 3D coordinates) to get the **point proxy** `F_i = F'_i + ϕ(q_i)`. The proxy carries both the local geometric pattern (`F'`) and the global position (`ϕ`).
2. **Transformer encoder (6 layers):** standard multi-head self-attention + FFN over the 128 proxies. Output: encoder memory `V = {V_1, ..., V_N}`.
3. **Query generator:** max-pool `V`, project to `M × 3`, reshape into coordinates `{c_1, ..., c_M}`. Concatenate the global feature with each `c_i` and run an MLP to produce **M dynamic query embeddings** `Q = {Q_1, ..., Q_M}`. (The "sketch" of the missing part.)
4. **Transformer decoder (8 layers):** self-attention on `Q` + cross-attention from `Q` to `V` + FFN. Output: **M predicted point proxies** `H = {H_1, ..., H_M}` representing the missing part.
5. **Multi-scale point generation:** reuse the `c_i` from the query generator as local centers. Apply a FoldingNet to each `H_i` to produce a set of neighboring points `P_i = f(H_i) + c_i`. Concatenate `P_i` across all `i` to get the predicted missing point cloud.
6. **Coarse + fine supervision:** loss is the sum of two Chamfer distances — one on the M coarse centers vs. the GT centers, one on the full predicted point cloud vs. the full GT point cloud. This forces the proxies to be *meaningful* (not just routed to a learned global feature).

### The Geometry-Aware Block
Vanilla self-attention: `softmax(QKᵀ/√d)V` — has no notion of "near" or "far" in 3D space. PoinTr's block does:
- Use the query coordinates `p_Q` to find kNN keys in coordinate space.
- Aggregate the kNN features with a linear layer + max-pool (the DGCNN trick).
- Concatenate the geometric feature with the semantic feature from self-attention, project back to the original dim.

This is a **plug-and-play** module — drop it into any transformer block. Ablation (Table 4) shows adding it to just the **first** encoder + first decoder block is *better* than adding it to all blocks (8.38 vs 8.44 CD-L1 on PCN), because more than one layer of geometric attention overfits.

### The Query Generator
A subtle but consequential design choice. The decoder queries are **not learned** (the standard BERT-style approach). Instead, they are **generated** from the encoder output. The intuition: the queries should reflect an *initial guess* of the missing shape, which obviously depends on the input. The output of the query generator is M=224 (PCN) or M=96 (ShapeNet-55/34) 3D coordinates — and these coordinates are the **local centers** of the FoldingNet head. So the "sketch of the missing shape" is literally a set of 3D points in space.

### Loss
Chamfer Distance on (a) the coarse centers and (b) the fine points:
```
J₀ = 1/n_C Σ_{c∈C} min_{g∈G} ||c−g||  +  1/n_G Σ_{g∈G} min_{c∈C} ||g−c||
J₁ = 1/n_P Σ_{p∈P} min_{g∈G} ||p−g||  +  1/n_G Σ_{g∈G} min_{p∈P} ||g−p||
J  = J₀ + J₁
```
No EMD (too expensive at O(N²)). CD has the standard permutation-invariance we need for unordered points.

### Training
- AdamW, initial LR 5e-4, weight decay 5e-4, batch size 54 (PCN) or 128 (ShapeNet-55/34), 300/200 epochs.
- LR decay 0.9 (PCN) or 0.76 (ShapeNet-55) per 20 epochs.
- N=128 input proxies, M=224 (PCN) or 96 (ShapeNet) output proxies.
- Encoder depth 6, decoder depth 8, k=16 for DGCNN, k=8 for geometry-aware block, 6 attention heads, hidden dim 384.
- 30.9M params, 10.41 GFLOPs.

### Datasets
- **PCN** (Yuan 2018): 8 categories, 28,974 train / 1,200 test. The "old" benchmark.
- **ShapeNet-55** (new): all 55 categories, 41,952 train / 10,518 test, 80-20 split. For each object: sample 8,192 GT points, randomly pick a viewpoint, remove the n=2048-6144 furthest points to get the partial, downsample remaining to 2,048. Three difficulty levels (simple/moderate/hard) by varying n.
- **ShapeNet-34** (new): 34 "seen" + 21 "unseen" categories. Train on 46,765 seen objects, test on 3,400 seen + 2,305 unseen objects. Measures **category-level generalization**.
- **KITTI** (real LiDAR cars): fine-tune the model on ShapeNetCars, evaluate on KITTI cars (no GT for KITTI, use Fidelity + MMD).

## Results

### ShapeNet-55 (Table 1, all 55 categories, mean over simple/mod/hard)
| Method | CD-S (×10³) | CD-M (×10³) | CD-H (×10³) | CD-Avg (×10³) | F1@1% |
|---|---|---|---|---|---|
| FoldingNet | 2.67 | 2.66 | 4.05 | 3.12 | 0.082 |
| PCN | 1.94 | 1.96 | 4.08 | 2.66 | 0.133 |
| TopNet | 2.26 | 2.16 | 4.30 | 2.91 | 0.126 |
| PFNet | 3.83 | 3.87 | 7.97 | 5.22 | 0.339 |
| GRNet | 1.35 | 1.71 | 2.85 | 1.97 | 0.238 |
| **PoinTr** | **0.58** | **0.88** | **1.79** | **1.09** | **0.464** |

PoinTr wins on every column. The 0.88-point F1@1% improvement over GRNet is the single most striking number in the paper — it means the predicted points are *much* closer to the GT surface (within 1% of the diagonal of the bounding box).

### ShapeNet-34 (Table 2) — 21 unseen categories
| Method | CD-S (×10³) | CD-M (×10³) | CD-H (×10³) | CD-Avg (×10³) | F1@1% |
|---|---|---|---|---|---|
| GRNet | 1.85 | 2.25 | 4.87 | 2.99 | 0.216 |
| **PoinTr** | **1.04** | **1.67** | **3.44** | **2.05** | **0.384** |

PoinTr degrades gracefully on unseen categories (CD-Avg goes 1.23 → 2.05, ~67% increase) while GRNet degrades much more (1.74 → 2.99, ~72% increase). On the hard setting, GRNet is 4.87 vs PoinTr's 3.44 — a 30%+ gap. **The transformer attention is more robust to novel categories than the CNN/gridding kernels of GRNet.**

### PCN (Table 3, the "old" benchmark)
| Method | CD-L1 (×10³) | Airplane | Cabinet | Car | Chair | Lamp | Sofa | Table | Watercraft |
|---|---|---|---|---|---|---|---|---|---|
| FoldingNet | 14.31 | 9.49 | 15.80 | 12.61 | 15.55 | 16.41 | 15.97 | 13.65 | 14.99 |
| PCN | 9.64 | 5.50 | 22.70 | 10.63 | 8.70 | 11.00 | 11.34 | 11.68 | 8.59 |
| TopNet | 12.15 | 7.61 | 13.31 | 10.90 | 13.82 | 14.44 | 14.78 | 11.22 | 11.12 |
| MSN | 10.0 | 5.6 | 11.9 | 10.3 | 10.2 | 10.7 | 11.6 | 9.6 | 9.9 |
| GRNet | 8.83 | 6.45 | 10.37 | 9.45 | 9.41 | 7.96 | 10.51 | 8.44 | 8.04 |
| PMP-Net | 8.73 | 5.65 | 11.24 | 9.64 | 9.51 | 6.95 | 10.83 | 8.72 | 7.25 |
| CRN | 8.51 | 4.79 | 9.97 | 8.31 | 9.49 | 8.94 | 10.69 | 7.81 | 8.05 |
| **PoinTr** | **8.38** | 4.75 | 10.47 | 8.68 | 9.39 | 7.75 | 10.93 | 7.78 | 7.29 |

PoinTr wins overall, with PMP-Net + CRN being its only serious competitors (the field was getting crowded). PCN's airplane result is *worse* than the 2018 PCN paper (22.70 vs 5.50) — the "PCN dataset" has known quirks (incomplete ground truth) that PoinTr happens to handle well.

### KITTI (Table 5) — real LiDAR cars
| Metric | AtlasNet | PCN | FoldingNet | TopNet | MSN | NSFA | PFNet | CRN | GRNet | PoinTr |
|---|---|---|---|---|---|---|---|---|---|---|
| Fidelity ↓ | 1.759 | 2.235 | 7.467 | 5.354 | 0.434 | 1.281 | 1.137 | 1.023 | 0.816 | **0.000** |
| MMD ↓ | 2.108 | 1.366 | 0.537 | 0.636 | 2.259 | 0.891 | 0.792 | 0.872 | 0.568 | **0.526** |

**0.000 Fidelity** is suspicious (likely a benchmark quirk where the predicted points coincide with the input scan), but the qualitative Figure 5 is convincing — PoinTr's cars have cleaner boundaries and more recognizable tires. **The big point: a model trained on synthetic ShapeNet cars transfers to real LiDAR with just fine-tuning.** This is the strongest H5 precedent we've seen.

### Complexity (Table 8)
| Model | Params | FLOPs | CD55 | CD34 |
|---|---|---|---|---|
| FoldingNet | 2.30M | 27.58G | 3.12 | 3.62 |
| PCN | 5.04M | 15.25G | 2.66 | 3.85 |
| TopNet | 5.76M | 6.72G | 2.91 | 3.50 |
| PFNet | 73.05M | 4.96G | 5.22 | 8.16 |
| GRNet | 73.15M | 40.44G | 1.97 | 2.99 |
| **PoinTr** | 30.9M | 10.41G | **1.07** | **2.05** |

Reasonable trade-off: 3rd smallest in params, 2nd smallest in FLOPs, but the lowest error. **Much more practical than GRNet (40G FLOPs vs 10G) for the same task.**

### Ablation (Table 4, PCN)
| Model | Query Gen | DGCNN | Geometry | CD-L1 | F1@1% |
|---|---|---|---|---|---|
| A (vanilla transformer) | – | – | – | 9.43 | 0.6782 |
| B | ✓ | – | – | 9.09 | 0.713 |
| C | ✓ | ✓ | – | 8.69 | 0.736 |
| D | ✓ | ✓ | all blocks | 8.44 | 0.741 |
| E (full) | ✓ | ✓ | 1st block only | **8.38** | **0.745** |

Each design choice helps. The geometry-aware block is most useful in the *first* block only — adding it everywhere overfits. The query generator alone gives +0.34 CD — surprisingly large for a "free" architectural change.

## Connections to our hypotheses

- **H1 (2-stage > end-to-end for missing-tooth detection):** **Mild support.** PoinTr is a single-stage model (encoder→decoder), but in our pipeline it plays the role of *one* of two stages: (stage 1) tooth-detection / missing-tooth-identification via 3DTeethSeg22 (paper 001), (stage 2) point-cloud completion via PoinTr conditioned on the partial arch. The fact that PoinTr's encoder-decoder cleanly separates "understanding the input" from "generating the output" means we can swap in a handcrafted or learned detector at the front without retraining. Also, the ShapeNet-34 unseen-category result (CD-Avg 2.05 vs 1.07 on seen cats) tells us the completion *itself* is a hard, separate problem from "is this tooth missing?" — a strong argument for the 2-stage decomposition.

- **H2 (diffusion > mesh-based VAE for surface generation):** **Inconclusive, mildly negative.** PoinTr is a **single-pass** transformer, not diffusion. It produces a point cloud in one forward pass, no iterative denoising, no multi-modal sampling. This is a *worse* fit for the "dentist wants N plausible crowns" use case than LION (paper 005) or Diffusion-SDF (paper 004) — you get one prediction, not a distribution. For our project, PoinTr is the **completion backbone** (sub-task 4: outer surface), but the **generative** step (sub-task 3/4: diverse plausible crown shapes) should still be LION or Diffusion-SDF. **Refine H2:** diffusion > single-pass transformer > mesh VAE — but the win of diffusion over single-pass is for *diversity* and *uncertainty quantification*, not raw accuracy.

- **H3 (conditioning on adjacent + opposing teeth improves outer surface):** **STRONG support, the cleanest implementation in the literature so far.** The Query Generator is *literally* the H3 conditioning mechanism: the encoder ingests the partial arch (which includes the adjacent + opposing teeth, by construction — they're the non-missing parts of the input), and the decoder's queries are **dynamically generated** from the encoder output. This is exactly the template we want for H3. In LION (paper 005) the conditioning is on `z0` via AdaGN, in Diffusion-SDF (paper 004) it's via cross-attention to a PointNet encoder, and in PoinTr it's via the query generator. **PoinTr's variant is the simplest and most direct for our use case** — and it's the one to clone for the v0 arch.

- **H4 (implicit SDF > explicit mesh for high-quality surfaces):** **Mild contradiction.** PoinTr produces an **explicit point cloud** (2,048–8,192 unordered points), not an implicit SDF. It's the most "explicit" representation in our reading list so far (more so than FlexiCubes, NDC, DiGS, DeepSDF). The output has no analytic gradient, no level-set semantics, no topological flexibility. For clinical fit, the point cloud would need to be lifted to an SDF (via DiGS, paper 003) and then to a mesh (via FlexiCubes, paper 007) for the sub-task 5 printability check. So **PoinTr is the H4-weakest link in the current pipeline**, but it has a defensible role as the **completion-from-partial-scan** stage, where its single-pass speed and robustness to noise beat DiGS's iterative optimization.

- **H5 (synthetic data can bootstrap training):** **STRONG support, perhaps the strongest single piece of evidence in the literature for our setting.** PoinTr trains on **synthetic ShapeNet only** and zero-shot transfers to real KITTI LiDAR scans with fine-tuning. The training data distribution is orders of magnitude cheaper to generate than the equivalent clinical IOS scans. For our project: a synthetic-to-real pipeline for dental crowns is **directly precedented** by PoinTr. We can bootstrap on 3DTeethSeg22 (which is *real* IOS, not synthetic) but the early-stage training of "give me a partial arch, I complete it" can be done on synthetic dental CAD repos. **Specifically:** generate 10,000 synthetic arches by (1) sampling 32 tooth positions on a Bézier curve (paper 001's arch prior), (2) sampling per-tooth shapes from a CAD library, (3) randomly masking out 1-3 teeth to simulate missing teeth, and (4) training PoinTr to complete the mask. This is **the H5 pilot experiment.**

## Surprises / things buried in the paper

1. **The geometry-aware block is most useful in the FIRST block only.** The ablation shows adding it to *all* blocks is *worse* than adding it to the first block (8.44 vs 8.38 CD-L1 on PCN). The interpretation: a single layer of geometric attention is enough to inject the inductive bias; more layers overfit. This is a useful rule of thumb for any future 3D transformer.

2. **The query generator alone gives +0.34 CD improvement** (model A → B in Table 4). Dynamic queries are nearly as important as the geometry-aware block. The "free" architectural change is the biggest.

3. **FoldingNet as a head is the *only* learnable decoder.** The actual point-cloud upsampling from M proxies to 2,048+ points is done by a *single* FoldingNet call per proxy — not a learned DGCNN, not a DNN, not a diffusion. The rest of the network is doing the heavy lifting of "where are the proxies"; FoldingNet is just a smooth 2D-grid-to-3D-surface deformation. **Implication:** we can swap FoldingNet for a more sophisticated decoder (e.g., a tiny diffusion model) later without retraining the encoder.

4. **PoinTr's CD is *significantly* better than its F-Score implies, and its F-Score is *significantly* better than its CD implies.** The CD on ShapeNet-55 is 0.58 (S) / 0.88 (M) / 1.79 (H) — small numbers. The F1@1% is 0.464 — close to perfect. Together they say: "the predicted points are very close to the GT surface, and the predicted set is very similar to the GT set." This is the right pair of metrics for clinical use.

5. **The KITTI Fidelity=0.000 result is a benchmark artifact**, not a real number. The standard KITTI eval protocol computes Fidelity as `CD(pred, GT_input)` but KITTI has no GT — they use the input as a proxy. A model that *exactly* copies the input gets 0. So 0.000 means "didn't deviate from the input car shape," which is a low bar. MMD (0.526 vs GRNet 0.568) is the more honest comparison.

6. **The paper trains on PCN with `k=16` for the DGCNN extractor and `k=8` for the geometry-aware block.** Two different k values. The reasoning (not in the paper, but conventional in the PointNet++ literature) is that the extractor needs to capture a richer local neighborhood (16 points), while the geometry block only needs to know the immediate neighbors (8 points) for local attention. This is a small detail that often gets copy-pasted wrong.

7. **The paper does NOT report runtime / inference latency.** Table 8 gives FLOPs (10.41G) and params (30.9M), which is enough to estimate GPU time, but no actual "ms per shape on V100" number. A quick back-of-envelope: 10.41 GFLOPs at 100 TFLOPs/s on a V100 = 0.1 ms of compute, plus memory IO = ~5-10 ms per shape. Single-pass, no iterative latent optimization = **fast at inference**, unlike DeepSDF (9.72 s) and DiGS (slow, iterative).

## Quote-worthy sentences

- (Sec. 3.1) *"Note that benefiting from the self-attention mechanism in transformers, the features learned by the transformer network are invariant to the order of point proxies, which is also the basis of using transformers to process point clouds."*
- (Sec. 3.4) *"We propose to use dynamic query embeddings, which makes our decoder more flexible and adjustable for different types of objects and their missing information."*
- (Sec. 1, related work) *"The bottleneck of such methods lies in the max-pooling operation in the encoding phase, where fine-grained information is lost and can hardly be recovered in the decoding phase."*
- (Sec. 4.1) *"Although the projection method proposed in [51] is a better approximation to real scans, our strategy is more flexible and efficient."*
- (Sec. 4.6) *"We find that only adding the geometric block to the first transformer block in both encoder and decoder can lead to a slightly better performance (model E), which indicates the role of geometric block is to introduce the inductive bias and a single layer is sufficient while adding more blocks may result in over-fitting."*

## Code & data

- Official PyTorch code: https://github.com/yuxumin/PoinTr
- Pretrained checkpoints on PCN and ShapeNet-55: provided in the repo.
- Datasets:
  - PCN: https://www.kaggle.com/datasets/pointcloudchallenge/pointcloudcompletion (or the original Yuan 2018 release)
  - ShapeNet-55/34: rendered on-the-fly from ShapeNetCore.v1 using the provided rendering script.
  - KITTI: http://www.cvlibs.net/datasets/kitti/
- License: code is MIT, data is the original ShapeNet / KITTI / PCN licenses.

## Follow-on work that directly applies to dental crowns

- **SnowflakeNet (2022)** — "Snowflake Point Deformation" with progressive refinement; the next iteration of point completion. Worth reading for the multi-stage decoder.
- **PMP-Net (CVPR 2021)** — "Point cloud completion by learning multi-step point moving paths" — predicts a path of point displacements instead of a final point cloud. More interpretable, may be relevant for the margin line in dental crowns.
- **PoinTr's later descendants** (2022-24) — many follow-ups; the field has largely converged on transformer + dynamic queries for completion.
- **3DTeethSeg (paper 001)** — the *dental* completion problem, on the *real* dataset we'll actually use. No published work to my knowledge does a transformer-based completion on 3DTeethSeg22 — this would be a clean first-author contribution.
- **Seedformer (ICCV 2022)** — successor to PoinTr with a "Patch Seed" representation; better for the local-detail preservation that crowns need (cusps, fissures).

## For our project

1. **Adopt PoinTr as the v0 outer-surface completion backbone for sub-task 4.** The H3 conditioning mechanism (query generator) is exactly what we need: encoder takes the partial arch (with the missing tooth location pre-masked), queries are dynamically generated, decoder produces the missing tooth's points. Concrete plan:
   - Pre-train PoinTr on a synthetic 10,000-arch dataset (sampled from a Bézier arch prior + CAD library per-tooth shapes + random masking) — **the H5 pilot experiment**.
   - Fine-tune on 3DTeethSeg22 with one tooth masked out per arch. The 3DTeethSeg22 dataset is 1,800 scans × 32 teeth = ~57,600 training teeth if we mask them one at a time.
   - The query generator should be conditioned on the FDI number of the missing tooth (concatenate a learned 16-dim class embedding to the global feature before generating queries) — this is a small extension of the architecture and gives us class-aware completion.

2. **PoinTr alone is not enough — we still need DiGS (paper 003) downstream.** PoinTr's point-cloud output has no analytic gradient and no level-set semantics. For the clinical fit check (sub-task 3 inner surface, sub-task 5 printability), we need a continuous SDF. The cleanest pipeline: **PoinTr (point cloud) → DiGS (SDF) → FlexiCubes (mesh) → trimesh (manifold repair)**. PoinTr is the missing-tooth generator, DiGS is the implicit-field refiner, FlexiCubes is the mesh extractor. All three were designed in 2021-23 and they compose naturally.

3. **Use the geometry-aware block trick everywhere.** It's a one-line DGCNN-style augmentation to any transformer block, and it consistently helps. Drop it into LION's PVCNN (paper 005) and Diffusion-SDF's attention blocks (paper 004) — both are 3D transformers and both could benefit.

4. **The ShapeNet-34 unseen-category result is the key H3 generalization argument.** When the dentist sees a "weird" anatomy (a tooth shape the model wasn't trained on), the completion should still work. PoinTr degrades 67% on unseen categories; GRNet degrades 72%. **This means transformer-based completion is more robust than CNN/gridding-based completion to patient variability.** Worth reporting in any clinical validation paper.

5. **Replace the FoldingNet head with a tiny DiGS head.** FoldingNet produces a smooth 2D-grid deformation — fine for the proxy, but for the final point cloud we want analytic normals and a continuous field. A 5-layer SIREN MLP (paper 003) conditioned on the proxy feature is a drop-in replacement that also gives us the implicit field for free.

6. **Critical open question for HK:** **do we want a single deterministic completion (PoinTr) or a *distribution* of completions (LION / Diffusion-SDF)?** The clinical UX difference is: "here's the crown" vs. "here are 5 crown options, pick one." PoinTr is fast and accurate; LION/Diffusion-SDF are slower but give the dentist choice. Both are valid; depends on the product. I'd lean **PoinTr for the v0 prototype** (fast iteration), **LION for the v1 product** (gives the dentist a choice). Worth a 30-min conversation.

7. **Read SnowflakeNet or Seedformer as paper 009.** Both are direct successors to PoinTr with better local detail (cusps, fissures, sharp features — all critical for the occlusal surface of a crown). The architectural lineage is: PCN (2018, FoldingNet) → GRNet (2020, gridding + CNN) → PoinTr (2021, transformer + dynamic queries) → SnowflakeNet/Seedformer (2022, progressive point deformation). Picking up at the latest generation for our pilot is the right call.

8. **Compute note for the pilot:** PoinTr is **much** cheaper to train than Diffusion-SDF (paper 004) or LION (paper 005). On a single V100: PCN trains in ~24h, ShapeNet-55 in ~48h. With 3DTeethSeg22 (smaller than ShapeNet-55), expect ~12-24h for the fine-tune. Total pilot budget: **~$50 on Lambda** for the PoinTr baseline. Compare to LION's ~$1,500 or Diffusion-SDF's ~$2,000. **This is the cheapest reasonable v0 we can build.**

---
*Scholar reading note — paper 008 of the dental-crown-gen survey. Built on the architectural pattern from papers 002-007. Strongly recommend this paper as the v0 prototype target.*
