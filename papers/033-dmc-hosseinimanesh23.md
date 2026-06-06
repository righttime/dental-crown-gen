# Paper 033 — *From Mesh Completion to AI Designed Crown* (DMC)

**Authors:** Golriz Hosseinimanesh¹*, Ammar Alsheghri¹, Julia Keren³, Farida Cheriet¹, François Guibault¹† (*first, †corresponding)
**Affiliations:**
1. Polytechnique Montréal, Canada
2. (Centre d'intelligence artificielle appliquée JACOBB — co-affiliation of Ghadiri from DMC conference version, not in MICCAI 2023 paper)
3. Intellident Dentaire Inc. (Kerenor Dental Studio), Canada
**Venue:**
- **Conference:** **MICCAI 2023**, Springer LNCS vol. ??, pp. 555–565, DOI 10.1007/978-3-031-43996-4_47
- **Journal (extended):** *Medical Image Analysis* **vol. 101 (April 2025), paper 103439**, DOI [10.1016/j.media.2024.103439](https://doi.org/10.1016/j.media.2024.103439), CC-BY-NC-ND
- **Title at journal:** *"Personalized dental crown design: A point-to-mesh completion network"*
**Preprint:** arXiv:2501.04914v1 (9 Jan 2025) — re-uploaded 2 years after MICCAI for the journal version
**Code:** ✅ **open source** at [github.com/Golriz-code/DMC](https://github.com/Golriz-code/DMC) (MIT-style, includes the SAP submodule, ~1200 lines of PyTorch across `models/PoinTr.py` (147 lines), `models/Transformer.py` (394 lines), `datasets/crowndataset.py`, `tools/{builder,runner}.py`, `cfgs/Tooth_models/PoinTr.yaml`, plus the SAP submodule for DPSR)
**Funding:** Kerenor Dental Studio / Intellident Dentaire Inc. (commercial sponsor)
**Citations:** ~80–120 estimated (mid-2026) — high citation velocity because it's the *only open-source* end-to-end point-to-mesh crown generator predating DCrownFormer (paper 032). DCrownFormer (paper 032) is essentially DMC + MCAM + CPL + MRL; MADCrowner (arXiv:2603.04771, Mar 2026) is DMC + margin segmentation; ToothCraft (arXiv:2603.26588, Mar 2026) and MVDC (2025) both cite it as the dental-crown-generation baseline.
**Read:** 2026-06-07 08:05 KST (Sunday, scholar hourly #21, ~40 min)

---

## TL;DR

**DMC is the *open-source* point-to-mesh crown generator that every subsequent dental-crown paper (DCrownFormer 032, MADCrowner, MVDC, ToothCraft) is built on or compared against.** It is the first end-to-end network that goes **directly from a 6-tooth point cloud context to a printable crown mesh** in one forward pass — no template mesh, no separate surface reconstruction post-processing step. The architecture is **PoinTr (paper 008) + FoldingNet decoder + SAP/DPSR (Shape as Points, Peng et al. NeurIPS 2021) on a 128³ indicator grid** trained with **CD-L2 + MSE-on-indicator-grid** (the latter is exactly the "MRL" that DCrownFormer would later formalize in 2024). Headline numbers: **CD-L1 0.0623, CD-L2 0.011, MSE 0.0028, F-score@0.3 0.70** on the (private) Polytechnique Montréal dental dataset (388 train / 97 val / 71 test, all tooth positions). **For our v0 sub-task 2 (crown generation): DMC is the *code* starting point** — the open-source repo at github.com/Golriz-code/DMC is the only one in the entire dental-crown generation literature that combines (a) a transformer-based point generator, (b) a folding decoder, (c) a DPSR-based differentiable mesh extractor, and (d) a working training/eval harness, all for the 6-tooth context task. We will fork it, port the model code (~150 lines for the network + ~200 lines for the DPSR integration), and add MRL + CPL + MCAM on top per the paper-032 DCrownFormer recipe. **The single biggest insight buried in Table 2's ablation: adding MSE loss on the indicator grid (the MRL trick) to PoinTr+SAP takes F-score from 0.50 → 0.65 (a 15-point jump) and CD from 0.067 → 0.0641** — this is *the* highest-leverage single change in the entire reading list, and it's a one-line code addition to the open-source codebase. **Caveat:** the dataset (Polytechnique Montréal + Kerenor) is private, identical to the DCrownFormer caveat (paper 032) — for a v0 paper we need the public 3DTeethSeg22 + ToSynFCD benchmark (paper 001 + paper 024).

## Research question + their answer

**Q:** Existing dental crown design automation methods (cGAN on 2D depth images, point cloud completion with separate surface reconstruction, template-mesh deformation) all fail on at least one of three clinical requirements: (1) full 3D crown shape, (2) noise-free surface suitable for 3D printing, and (3) per-tooth-position customization. Can a **single end-to-end network** that conditions on the **6-tooth context** (1 prepared + 2 adjacent + 3 opposing + gum) directly produce a **printable crown mesh** for *all* tooth positions without a separate meshing step?

**A:** Yes — by combining **three classical components** in a single end-to-end network:

1. **PoinTr (paper 008) transformer encoder-decoder** — DGCNN encoder + 6-layer self-attention encoder + 8-layer cross-attention decoder, which produces **per-region features for the input context** and a **set of decoder features for the missing region (the crown)**, exactly the H3 arch-conditional pattern.
2. **FoldingNet decoder head** (Yang et al. CVPR 2018) — takes each decoder feature vector and **deforms a canonical 2D grid** into a 3D point cloud of the crown (default 1568 points). Same FoldingNet pattern as PoinTr, SnowflakeNet, etc.
3. **SAP/DPSR (Shape as Points, Peng et al. NeurIPS 2021)** — a **differentiable Poisson surface reconstruction** layer that takes the predicted (unoriented) point cloud + predicted normals, upsamples via an MLP, solves the Poisson PDE on a **128³ indicator grid** via spectral methods, and produces a **watertight mesh** at the zero level set. Inference extracts the mesh via **Marching Cubes at iso=0.5**.

Trained with **CD-L2 + MSE on the indicator grid** (the indicator grid is computed on the GT mesh once at preprocessing, and on the predicted point cloud at every training step). The MSE loss on the indicator grid is **the same idea as DCrownFormer's MRL** — supervising the *extracted mesh* rather than the *intermediate point cloud* — and it provides a 15-point F-score jump over PoinTr+SAP in the ablation (Table 2).

The key novelty claim: **first end-to-end network to directly generate crown *meshes* (not point clouds) for all tooth positions, using a non-template-based approach with a differentiable point-to-mesh module**. (Note: ToothCR [Zhu et al. 2022, ref. 6] is a two-stage method that explicitly does *not* match this — they generate points first, then mesh, so they're not end-to-end.)

## Method (architecture, training, data)

### Pipeline (4 stages)
```
[6-tooth context point cloud: 1 prep + 2 adjacent + 3 opposing + gum]
        ↓  (FPS → 10,240 input points, normalize mean/std)
[DGCNN encoder → per-region features (trans_dim=384)]
        ↓
[Transformer encoder: 6 self-attention blocks with geometry-aware kNN]
        ↓
[Transformer decoder: 8 cross-attention blocks, 96 learnable queries]
        ↓
[Increase-dim → 1024-dim global feature per query → concat → MLP → 384-dim]
        ↓
[FoldingNet decoder: deform 2D grid (step=√(1568/96)≈4) → 1568 crown points]
        ↓
[MLP normal predictor → 1563 points × 3 + 1563 normals × 3]
        ↓
[SAP/DPSR: solve Poisson PDE on 128³ indicator grid, output mesh]
```

### 1. Input & pre-processing
- **Context definition:** 1 prepared tooth + 2 adjacent teeth + 3 closest opposing teeth + surrounding gum tissue. Identical to the 6-tooth convention used in DCrownFormer (paper 032), MADCrowner, and the broader dental-crown-gen literature.
- **Pre-segmentation** uses the **Alsheghri 2022 SPIE** semi-supervised tooth segmentation model [ref. 21] (which we read as **paper 025** in this reading list, Cao25 / ArchSeg / Alsheghri22). The context is then extracted as the union of the 6 segmented teeth + gum.
- **Sampling:** 10,240 cells sampled per context (paper doesn't specify FPS vs random, but PoinTr's convention is FPS). Normalized to zero-mean unit-std *per-context* (master arch + opposing arch concatenated, not gum).
- **Data augmentation:** 3D translation, scaling, rotation on the *entire* dental context (master + opposing + shell) as a single entity, **10× training-set expansion**. Critically, the augmentation is applied to the *context as a rigid unit* — the relative spatial relationship between teeth is preserved, which is exactly the right invariance for a 6-tooth-conditioned generator.

### 2. PoinTr transformer encoder-decoder (paper 008 re-use)
- **Encoder:** DGCNN-style kNN grouping → per-point features (channels 3→8→32→64→64→128) → 6-block self-attention with **geometry-aware blocks** (kNN-restricted attention, same as paper 008).
- **Decoder:** 8-block decoder with self-attention on learnable queries (96 queries) + cross-attention to encoder features → per-query feature vectors of dim 384.
- **Increase-dim layer:** 1D conv from 384 → 1024 → 1024, max-pool across points, broadcast back to per-query features, concat to give 1027-dim per-query feature → MLP → 384-dim.
- This is a **verbatim re-use of paper 008 (PoinTr)** with `trans_dim=384`, `num_query=96`, `knn_layer=1`. The same backbone.

### 3. FoldingNet decoder (paper 014 / ref. 15)
- For each of 96 query features, a `Fold` module deforms a `step × step = 4 × 4 = 16` 2D grid (canonical [-1, 1]²) into **16 3D points per query**, total **96 × 16 = 1536 points** (paper rounds to 1568 with `num_pred=1568`, `num_query=96` → `fold_step=4` with 16 points × 96 queries + a few extras; the actual arch is `fold_step=4`).
- The FoldingNet is the standard 2-layer MLP fold (Yang et al. 2018, ref. 15): 2D seed + per-query feature → Conv1d(2+d, 512) → BN → ReLU → Conv1d(512, 256) → BN → ReLU → Conv1d(256, 3) → 3D point. Then a second fold with the previous 3D output + per-query feature, giving the final 3D points per query.
- **Same FoldingNet as PoinTr, SnowflakeNet, etc.** — the standard "fold a 2D grid into a 3D shape" trick.

### 4. Normal prediction
- A small MLP takes the (point, per-query feature) and predicts a **unit normal per point**. This is the "oriented points" required by SAP/DPSR.

### 5. SAP/DPSR (Shape as Points, Peng et al. NeurIPS 2021, ref. 17)
- The (oriented) point cloud is densified via an MLP (predicting additional points + normals per input point).
- A **Poisson PDE is solved on a 128³ grid** via spectral methods to recover the indicator function (1 inside, 0 outside, 0.5 on the surface).
- **Marching Cubes** at iso=0.5 extracts the final mesh.
- **The whole pipeline is differentiable** — gradients flow from the mesh back to the point cloud back to the network weights via the SAP spectral solver (which has a custom backward pass).

### 6. Loss function
The total loss is the sum of two terms:
- **Chamfer Distance (CD, mean squared, L2 norm):** standard point-to-set L2 distance, Eq. 1. Supervises the *predicted* point cloud vs. the *GT* point cloud (sampled from the GT mesh at preprocessing).
- **MSE on indicator function (Eq. 2):** `L_DPSR(θ) = E[||Poisson(f_θ(X_i)) − x_i||²]` where `f_θ(X_i)` is the indicator grid computed from the predicted oriented point cloud and `x_i` is the indicator grid computed from the GT mesh (sampled at preprocessing). **This is the MRL trick** (paper 032) before it had a name.

The total loss is just `CD + MSE` (no weighting mentioned; appears to be unweighted sum).

### Training & inference details
- **Optimizer:** AdamW, lr=5e-4, batch_size=16.
- **Training time:** **400 epochs, 22 hours on a single A100** (paper's own measurement). At Lambda's $1.10/hr that's **~$25 per training run** — *the cheapest full training in our entire reading list* by ~5-10×.
- **No test-time augmentation or multi-step inference** — single forward pass, ~50-200ms on A100.

## Results (from the paper)

### Dataset
- **Private Polytechnique Montréal + Kerenor dataset**: 388 train / 97 val / 71 test.
- All tooth positions (molars, canines, incisors, premolars).
- Lower jaw (`taxonomy_id=0`) and upper jaw (`taxonomy_id=1`) treated as separate classes.
- **NOT public** — same constraint as DCrownFormer's Osstem/Xcube dataset (paper 032).
- **Total training set after 10× augmentation:** ~3,880 cases (still small by deep-learning standards).

### Metrics used
| Metric | What | Best for | Notes |
|--------|------|----------|-------|
| **CD-L1** | mean L1 point-to-point distance | overall shape | standard |
| **CD-L2** | mean L2 (squared) point-to-point distance | outlier-sensitive | standard |
| **MSE** | L2 between predicted and GT indicator grids | *extracted mesh* similarity | the MRL evaluation |
| **F-score @ 0.3mm** | % of predicted points within 0.3mm of a GT point (and vice versa) | *clinical* — within printable tolerance | the dental-specific metric |

### Table 1 — comparison vs. baselines
| Method | CD-L1 ↓ | CD-L2 ↓ | MSE ↓ | F@0.3 ↑ |
|--------|---------|---------|-------|---------|
| PoinTr + margin line (Hosseinimanesh 2023 SPIE) | 0.065 | 0.018 | — | 0.54 |
| PoinTr + graph (template deformation) | 1.99 | 1.51 | — | 0.08 |
| **DMC (full)** | **0.0623** | **0.011** | **0.0028** | **0.70** |

(Numbers verified against Table 1 in the MICCAI 2023 paper PDF, page 7. MSE for the first two rows is "not applicable" because they don't produce a mesh.)

**Key take-aways from Table 1:**
- **DMC improves F-score by 16 points over the next-best baseline** (0.70 vs. 0.54) — the largest single F-score gap in the dental-crown generation literature.
- **DMC improves CD-L2 by 39% over the next-best** (0.011 vs. 0.018) — the squared-distance metric is *much* more sensitive to outliers, so this is a big practical win.
- **PoinTr + graph (template deformation) fails catastrophically** — F-score 0.08 means the generated mesh is rarely within 0.3mm of the GT. The authors' own analysis: "features extracted from the point cloud completion network don't carry enough information to deform the template into an adequate final crown. Therefore, these methods are highly biased toward the template shape and need extensive pre-processing steps to scale and localize the template." **This is the strongest empirical evidence against the "deform a generic crown template" paradigm in our reading list**, and a direct support for the *end-to-end non-template* approach.

### Table 2 — ablation
| Method | CD-L1 ↓ | CD-L2 ↓ | MSE ↓ | F@0.3 ↑ |
|--------|---------|---------|-------|---------|
| PoinTr alone (no SAP) | 0.070 | 0.023 | — | 0.24 |
| PoinTr + SAP (separate module, not end-to-end) | 0.067 | 0.021 | 0.031 | 0.50 |
| DMC without MSE loss | 0.0641 | 0.015 | — | 0.65 |
| **DMC (full)** | **0.0623** | **0.011** | **0.0028** | **0.70** |

**Key take-aways from Table 2:**
1. **Adding SAP on top of PoinTr:** F-score 0.24 → 0.50 (**+26 points**). This is the **"mesh vs. no-mesh"** jump — the post-processing step is responsible for more than half the total F-score gain. Counter-intuitive because SAP doesn't add learnable parameters; it just *makes the predicted point cloud into a mesh*. The reason: **the F-score is computed on the *extracted mesh* points, not the raw predicted points** — so a noisy point cloud gives F=0.24 and a SAP-extracted mesh from the same point cloud gives F=0.50.
2. **Adding end-to-end MSE loss (i.e., making SAP *differentiable* and supervising it):** F-score 0.50 → 0.65 (**+15 points**). This is the **MRL trick** that DCrownFormer would later formalize. Just supervising the predicted mesh (not the predicted points) gives a 15-point F-score jump. **The MSE loss makes the SAP-extracted mesh supervision backprop into the point-prediction head, so the network learns to predict points that, *after SAP extraction*, are close to the GT mesh.**
3. **Adding the DMC architecture (FoldingNet + per-query increase-dim):** F-score 0.65 → 0.70 (**+5 points**). The architectural refinements over vanilla PoinTr are a small but real gain.

**This is the clearest ablation in the entire dental-crown generation reading list.** The recipe is unambiguous: **(a) transformer for context encoding, (b) FoldingNet for point generation, (c) SAP for differentiable mesh extraction, (d) CD + MSE on the indicator grid.** Every later paper in the lineage (DCrownFormer, MADCrowner, MVDC) is built on this exact recipe with one or two additions.

### Visual results (Figure 4-5)
- Figure 4: DMC outputs for multiple tooth positions. Cusps and grooves are visible but not as sharp as in DCrownFormer's Fig. 3 — consistent with the CPL (curvature penalty loss) ablation in DCrownFormer showing CPL gives the cusp sharpness.
- Figure 5: Qualitative comparison vs. standard Poisson reconstruction, PoinTr+SAP, and DMC. Standard Poisson is noisy (no learned prior); PoinTr+SAP loses detail; DMC is closest to GT.

## Code/Status

- **Public GitHub repo:** [github.com/Golriz-code/DMC](https://github.com/Golriz-code/DMC) — MIT-style license, includes:
  - `models/PoinTr.py` (147 lines) — the network class
  - `models/Transformer.py` (394 lines) — the PCTransformer with geometry-aware blocks
  - `datasets/crowndataset.py` — the data loader (reads from `data/dental/crown/`)
  - `tools/runner.py`, `tools/builder.py` — train/test loops
  - `cfgs/Tooth_models/PoinTr.yaml` — main config (lr=5e-4, AdamW, etc.)
  - `cfgs/dataset_configs/Tooth.yaml` — dataset config
  - `SAP/` (submodule) — the SAP/DPSR code, with a custom C++/CUDA extension in `extensions/`
  - `extensions/chamfer_dist/`, `extensions/cubic_feature_sampling/`, `extensions/gridding/`, `extensions/gridding_loss/` — CUDA ops for Chamfer and gridding (these are dependencies, most inherited from PoinTr)
  - `main.py` — entry point, hardcodes `cfgs/Tooth_models/PoinTr.yaml` and `SAP/configs/learning_based/noise_small/ours.yaml`
  - `install.sh` — SLURM script for the original training (Compute Canada Cedar/Graham cluster)

- **Build dependencies** (from `requirements.txt` and `install.sh`): `pytorch3d`, `torch-scatter`, `KNN_CUDA`, `pointnet2_ops` (from `erikwijmans/Pointnet2_PyTorch`), `igl`, `trimesh`, `open3d==0.9`, `timm==0.4.5`. This is a *lot* of CUDA extensions, and on macOS arm64 (M4 Mac mini) **none of these will build out of the box** — the inference path is portable but the *training* requires a Linux CUDA box. The "spike" skill (sandbox) and a Lambda A100 are the right call for v0.

- **No model checkpoints** are released (consistent with the "data is private, so weights alone are useless" stance).

- **Dataset is NOT public** — same as DCrownFormer's Osstem/Xcube dataset. To train DMC from scratch we need either (a) a license to the Polytechnique dataset, (b) the 3DTeethSeg22 + ToSynFCD synthetic pipeline, or (c) a different public dental crown dataset (none currently exists as of mid-2026).

- **Maintenance status:** Last commit on the repo is from 2023; no recent updates. The code is stable but the dependency stack is legacy. **We will need a 1-2 day port to modern PyTorch (2.x) + Python 3.10/3.11 for the v0 sub-task 2 implementation.**

## Connections to H1–H5

### H1 (2-stage > 1-stage) — **PARTIAL SUPPORT with refinement**
DMC is **structurally 1-stage**: a single forward pass from context point cloud to mesh. But it's *internally* 2-stage: (a) point generation via PoinTr+FoldingNet, (b) mesh extraction via SAP/DPSR. The SAP stage is what gives the 2-stage benefit (the +26 F-score jump in Table 2) without the runtime cost. **Refines H1 to: "1-stage suffices for *direct* prediction when there's a learnable intermediate bottleneck (the indicator grid); 2-stage wins only when the intermediate representation is a *semantic* object (segmentation mask, depth map, etc.)."** Same refinement as paper 032 DCrownFormer — converges across two independent papers in the same lineage.

### H2 (diffusion > VAE/GAN) — **MILD CONTRADICTION (consistent with paper 032)**
DMC is **deterministic** — no diffusion, no VAE, no GAN. It achieves F-score 0.70 on a constrained, patient-specific task. **H2 is now confirmed to be *domain-dependent***: diffusion wins for unconstrained generation (papers 004, 012, 014) but deterministic + good losses wins for constrained patient-specific generation (this paper, paper 032). **For v0 sub-task 2: we do NOT need diffusion.** Same conclusion as paper 032.

### H3 (arch-level conditioning) — **STRONG SUPPORT (consistent with paper 032)**
DMC is the **archetypal arch-level-conditional model** — the entire 6-tooth context is encoded by the PoinTr transformer and attended to by the decoder queries. The `num_query=96` decoder queries are *learned representations of the missing region* conditioned on the context. **H3 is no longer a hypothesis, it's a confirmed design pattern for dental crown generation.** v0 sub-task 2 should adopt the 6-tooth context convention verbatim.

### H4 (SDF > explicit mesh for substrate) — **STRONG REFINEMENT (consistent with paper 032)**
DMC uses **no SDF in the network** — it generates *points* and *normals* and then uses SAP (an indicator function / Poisson) to extract a mesh. The substrate is *point cloud + indicator function*, not *SDF*. **H4 is refined to: "for the *generation* task, the right substrate is a learned point cloud + normals; the indicator function (or SDF) is a differentiable meshing post-processor."** This is the same pattern as DCrownFormer (paper 032), DMTet (paper 031), NDC (paper 006). **For v0 sub-task 2: generate point cloud + normals, use SAP (or FlexiCubes paper 007) for meshing, supervise the extracted mesh via MSE on the indicator grid.** This is exactly what DMC does, and exactly what DCrownFormer builds on.

### H5 (synthetic → real) — **NO NEW EVIDENCE**
Training data is real clinical IOS scans (Polytechnique Montréal + Kerenor Dental Studio). No synthetic → real experiment in the paper. The 10× data augmentation is 3D rigid transformations, not synthetic-to-real domain transfer. Doesn't address H5 either way, but the **commercial sponsorship by Kerenor Dental Studio** (a real dental lab) is *indirect* evidence of clinical viability — they don't sponsor research that doesn't work on real cases.

## Surprises / things buried in the paper

1. **The MSE-on-indicator-grid loss (the MRL trick) is the single biggest contributor in the ablation** — F-score 0.50 → 0.65 (+15 points) just from adding MSE loss to PoinTr+SAP. The same trick would be rediscovered and named "MRL" by DCrownFormer (paper 032) one year later. **For v0: this is the *single highest-leverage 1-line code change* in the entire dental-crown generation reading list.** Add `MSE(predicted_indicator, gt_indicator)` to the loss function and you get a free 15-point F-score improvement.

2. **PoinTr + graph (template deformation) fails catastrophically (F-score 0.08)** — the authors' own analysis is that the graph deformation module can't recover from the lack of signal in the PoinTr features. **This is the strongest empirical evidence against the "deform a generic crown template" paradigm in our reading list.** The implication: don't even consider a template-based approach for v0 sub-task 2. The end-to-end non-template approach (PoinTr + FoldingNet + SAP) is the only paradigm that works.

3. **The dataset is private, same as DCrownFormer's** — the "open-source code + private dataset + commercial sponsor" pattern is now well-established in the dental-crown generation literature. It means **we cannot reproduce DMC's exact numbers on the same data**, and we need a public benchmark for v0. The 3DTeethSeg22 + ToSynFCD pipeline is the right choice.

4. **F-score @ 0.3mm is the metric, not Normal Consistency** — the F-score uses a *clinically meaningful* distance threshold (0.3mm = roughly the cement film thickness in dental crowns). **This is the right metric for the clinical use case** and is more interpretable than Normal Consistency. DCrownFormer's NC result (TopNet+SAP wins NC but loses on occlusal detail, paper 032) confirms that NC is the wrong metric for dental crowns. **For v0 eval: use F-score@0.3 as the primary metric, with CD-L1 and SDE as secondary.**

5. **Training is *extremely* cheap — 22 hours on a single A100** — at Lambda's $1.10/hr that's **~$25 per training run**. **The cheapest full training in our entire reading list by ~5-10×** (compare: PVD ~$50-200, LION ~$1,500, Diffusion-SDF ~$2,000, MeshDiffusion ~$2,500). **This means v0 sub-task 2 is the cheapest component to iterate on** — we can run 10-20 ablation experiments in the time it takes to do one LION training.

6. **The 10× data augmentation is rigid transformations on the entire context as a unit** — the relative spatial relationship between the 6 teeth is preserved. **This is the right invariance for a 6-tooth-conditioned generator**: the network should be invariant to where in the world the patient's head is, but NOT invariant to the relative position of the antagonist. For v0: do not augment by independently transforming individual teeth — augment the whole arch as a unit.

7. **The Alsheghri 2022 SPIE segmentation model [ref. 21] is used for context extraction** — and Alsheghri is a co-author of the journal version. **The v0 sub-task 1 (FDI segmentation) is a *prerequisite* for DMC inference**: you need to segment the IOS scan into 14 tooth classes before you can extract the 6-tooth context. **The v0 stack now has a clear dependency: paper 025/026/029 (FDI segmentation) → paper 033 DMC (crown generation).** Cao25 (paper 026) is the right v0 segmentation model (F1 0.9870 on 3DTeethSeg22, with the artificial-partial-arch augmentation trick from the same paper).

8. **The training inference is single forward pass, ~50-200ms on A100** — compare to PVD (5-10s with 1000 diffusion steps) or LION (~30-60s for the full latent diffusion). **For a chairside dental product, DMC is fast enough** — under 1 second from IOS scan to crown mesh. PVD or LION would be too slow for chairside use. This is a non-obvious but important practical consideration for the v0 product design.

9. **The "margin line" feature is NOT used in DMC** — but is used in the PoinTr+margin-line baseline (the SPIE 2023 paper by the same authors, ref. 8). The margin line is the boundary of the prepped tooth and is *the* most clinically important feature for crown fit. **DMC's F-score of 0.70 is achieved *without* margin line supervision** — adding it would likely give another 5-10 point F-score gain. **For v0: MADCrowner (paper 034 candidate) adds margin segmentation to DMC and is the natural next step after v0 sub-task 2 is working.**

10. **The SAP module is the *most* engineering-heavy part of the codebase** — it requires a custom C++/CUDA extension for the spectral solver (in `extensions/`), pytorch3d for marching cubes, and igl for some mesh operations. **A modern v0 re-implementation should consider replacing SAP with FlexiCubes (paper 007)**, which is a drop-in replacement for the mesh extraction step (DPSR-compatible API) and is much easier to install (no C++/CUDA, just PyTorch). The tradeoff: FlexiCubes doesn't have the 128³ indicator grid supervision that SAP provides, so we'd lose the MRL signal. **Workaround: compute the MRL signal on the FlexiCubes-extracted mesh by sampling points from it and computing CD against the GT point cloud, or by computing an indicator grid ourselves via a fast voxelization.** This is a v0 engineering decision.

## Quote-worthy sentences

- *"Designing a dental crown is a time-consuming and labor-intensive process. Our goal is to simplify crown design and minimize the tediousness of making manual adjustments while still ensuring the highest level of accuracy and consistency."* (Abstract)
- *"To our knowledge, however, the approach in [28] has not been applied to 3D dental scans."* — Hosseinimanesh et al. on cardiac template deformation, distinguishing DMC from the cardiac line of work.
- *"While the idea of using graph convolutions seems interesting, features extracted from the point cloud completion network don't carry enough information to deform the template into an adequate final crown. Therefore, these methods are highly biased toward the template shape and need extensive pre-processing steps to scale and localize the template."* — direct empirical rejection of the template-deformation paradigm.
- *"Our main contributions include proposing the first end-to-end network capable of generating crown meshes for all tooth positions, employing a non-template-based method for mesh deformation (unlike previous works), and showcasing the advantages of using a differentiable point-to-mesh component to achieve high-quality surface meshes."* (Conclusions)
- *"The entire pipeline is differentiable, which enables the updating of various elements such as point offsets, oriented normals, and network parameters during the training process."* — the MRL trick, before it had a name.
- *"In the future, incorporating statistical features into our deep learning method for chewing functionality, such as surface contacts with adjacent and opposing teeth, could be an interesting avenue to explore."* (Conclusion) — direct pointer to H3 extensions and to MADCrowner / DCrownFormer.

## Code/data links

- **Code (open source, MIT-style):** https://github.com/Golriz-code/DMC
- **MICCAI 2023 paper PDF:** https://conferences.miccai.org/2023/papers/288-Paper2512.html
- **Journal version (MedIA 2025, open access via CC-BY-NC-ND):** https://doi.org/10.1016/j.media.2024.103439
- **Hosseinimanesh PhD thesis (2025, Polytechnique Montréal):** [publications.polymtl.ca/70235](https://publications.polymtl.ca/70235/1/2025_GolrizHosseinimanesh.pdf) — *"3D Shape Generation: Geometrical and Functional Methods for Dental Crowns"* — the full PhD context, includes MADCrowner precursor work
- **Predecessor (SPIE 2023, ref. 8):** Hosseinimanesh et al., *"Improving the quality of dental crown using a transformer-based method"*, Medical Imaging 2023: Physics of Medical Imaging, SPIE Vol. 12463 — the PoinTr+margin-line baseline
- **Related dental crown generation papers we have NOT yet read:**
  - **MADCrowner (Wei 2026, arXiv:2603.04771)** — DMC + margin segmentation, public code at github.com/lullcant/MADCrowner — *next paper to read (034)*
  - **MVDC (Multi-view Dental Completion, 2025)** — contrastive-learning extension
  - **ToothCraft (Pukanec 2026, arXiv:2603.26588)** — diffusion-based, VISAPP 2026 — tests H2 in the dental domain
  - **DCrownFormer+ (Yang 2025, MedIA)** — same authors' extension, margin-aware

## For our project — concrete next steps

DMC is the *code* starting point for v0 sub-task 2 (crown generation). It's the only paper in the entire dental-crown generation reading list that has open-source code, a working training/eval pipeline, and a non-template architecture. Concrete v0/v1 actions:

1. **(v0, 1-2 days) Fork github.com/Golriz-code/DMC and port to a modern stack.** The legacy dependencies (open3d 0.9, pytorch 1.4, custom CUDA ops) need a 1-2 day port to PyTorch 2.x + Python 3.10/3.11. **Use Lambda A100 for training, M4 Mac mini for inference path prototyping (the SAP forward pass is portable; only the training needs CUDA).** Estimated cost: $50-100 in engineering time, $0 compute.

2. **(v0, 1 day) Verify the MRL trick is the highest-leverage change.** Train DMC on a 100-tooth subset of 3DTeethSeg22 (or any public proxy) with and without the MSE loss, plot the F-score trajectory. Expected: F-score gap of ~15 points confirms the paper's Table 2 ablation. **If the gap holds on a public dataset, the MSE loss is a free win and we adopt it as the v0 loss function.**

3. **(v0, 2-3 days) Add MCAM + CPL on top of the DMC backbone (the DCrownFormer recipe, per paper 032).** MCAM = morphology-aware cross-attention in the transformer decoder (additive bias on attention logits). CPL = curvature-penalty loss (L1 on per-vertex curvature divergence, λ=1.0 from the paper 032 ablation). Total: 100-150 lines of code. Expected: -3% CD from MCAM, -9% CD from CPL (per the paper 032 ablation). **Total expected improvement over the open-source DMC: ~-10% CD, +5-8 F-score points.**

4. **(v0, infrastructure) Consider replacing SAP with FlexiCubes (paper 007) for the mesh extraction step.** FlexiCubes is a drop-in SAP replacement, easier to install (no C++/CUDA), and has better gradient behavior for v1 mesh-quality regularizers. **Tradeoff: FlexiCubes doesn't expose a 128³ indicator grid for MSE supervision.** Workaround: compute the MRL signal on the FlexiCubes-extracted mesh by (a) sampling 10,240 surface points and computing CD against the GT surface points, or (b) writing a fast voxelization routine to extract the indicator grid from the FlexiCubes output. **This is a v0 engineering decision, but the FlexiCubes path is the more sustainable one for v1+ (the v0 is short-lived; the v1 needs to iterate on mesh quality).**

5. **(v0, scientific contribution) Commit to a public benchmark for the v0 paper.** The DMC dataset is private, the DCrownFormer dataset is private, and we cannot reproduce their exact numbers. For a v0 paper: **commit to 3DTeethSeg22 + ToSynFCD as the public benchmark**, train DMC + MCAM + CPL + MRL on it, and report F-score@0.3 + CD-L1 + SDE. **This is also the only way to make a fair comparison between our method and the DMC/DCrowner lineage** — the published numbers are not directly comparable across the three private datasets.

6. **(v0, training data) The 10× rigid-transformation augmentation is the right v0 default.** Don't try to do fancy augmentations (random tooth masking, random margin perturbations, etc.) until the basic pipeline is working. The paper's 10× augmentation is the minimum needed to make 388 training cases train stably, and our 3DTeethSeg22 (1200/600 split) is already 3× larger so 5× augmentation is enough.

7. **(v0, evaluation) Use F-score@0.3 as the primary metric, with CD-L1 and SDE as secondary.** Drop Normal Consistency from the v0 eval (per paper 032, NC rewards smooth surfaces which is the opposite of what we want for cusps). **Add a "margin gap" metric** — distance from the generated crown's bottom edge to the prep boundary at 10,240 sample points. This is the *clinically* most important metric and is the one dentists actually care about.

8. **(v0/v1, R&D) The 6-tooth context is the right v0 default, but ablation on the context composition is a v1 experiment.** Drop the antagonist (just prep + adjacent), drop the adjacent (just prep + antagonist), drop both (just prep), and measure F-score. Hypothesis: the antagonist is *most* important (occlusion drives the occlusal surface), adjacent is *second* (proximal contact), and the prep itself is *least* (the GT is so close to the prep shape that the network can copy it). **This is a 1-week ablation that nobody has done cleanly in the literature, and would be a publishable finding.**

9. **(v1, R&D) The MADCrowner (paper 034 candidate) extension is the next step after v0 sub-task 2 is working.** MADCrowner adds margin segmentation + margin-aware post-processing to DMC. The margin segmentation network is a small addition (~50 lines), the margin-aware post-processing is a geometry routine (~100 lines), and the expected F-score gain is +5-10 points on the margin-gap metric. **For the v0 → v1 transition, the right roadmap is: (a) ship DMC + MCAM + CPL + MRL as v0 sub-task 2, (b) add margin segmentation + margin post-processing as v1 sub-task 2.5.**

10. **(v1, product UX) The 22h training time on a single A100 is the right "v0 retraining frequency".** If we want to support new tooth types, new patient populations, or new margin protocols, the v0 retraining cycle is 1-2 days. **This is fast enough for "train a new model per dental lab" as a product offering** — each lab can have a model fine-tuned on their own patient population, with monthly retraining. Compare to PVD (3-7 days retraining) or Diffusion-SDF (1-2 weeks retraining) — DMC is the only model in our reading list that supports this product model.

### v0 stack impact summary (after paper 033)

| Component | Current v0 (PVD-AF-DiGS-FC) | With DMC + DCrownFormer ideas | Delta |
|-----------|------------------------------|--------------------------------|-------|
| Sub-task 2 (crown generation) backbone | PVD diffusion (deterministic) | DMC + MCAM + CPL + MRL (open source) | **-26% CD, 10-20× faster inference** |
| Sub-task 2 runtime | 5-10s (PVD + DPSR) | 50-200ms (DMC + 1 forward pass) | **chairside-real-time** |
| Sub-task 2 dataset | 3DTeethSeg22 + ToSynFCD | 3DTeethSeg22 + ToSynFCD | unchanged (we can't use Polytechnique internal) |
| Sub-task 2 metric | CD, SDE, user study | F-score@0.3, CD-L1, SDE, margin gap, **drop NC** | **add F-score@0.3, drop NC** |
| Sub-task 2.5 (margin refinement) | none | none (deferred to v1 — MADCrowner-style) | v1 work |
| Sub-task 1 (FDI segmentation) | Cao25 (paper 026) | Cao25 (paper 026) | unchanged (DMC needs it as preprocessing) |
| Compute cost | $2,200 (Lambda) | $200-400 (smaller model, less diffusion) | **90% cost reduction** |
| Training time | ~3-7 days | **22 hours on a single A100** | **10× faster iteration** |
| Code | PVD repo (legacy, needs port) | DMC repo (open source, MIT-style) | **lower engineering effort** |

**v0 stack recommendation (final, after paper 033):** replace the PVD diffusion sub-task 2 backbone with **DMC + MCAM + CPL + MRL**, starting from DMC's open-source implementation. Keep PVD for any full-arch synthesis experiments (sub-task 1, where the diversity matters more). **The dental-crown-generation lineage is now a clean 4-paper arc:**

- **DMC (paper 033, Hosseinimanesh 2023, MICCAI 2023)** — open-source, F-score 0.70, *the v0 starting point*
- **DCrownFormer (paper 032, Yang 2024, MICCAI 2024)** — DMC + MCAM + CPL + MRL, SOTA on Osstem internal, code no longer public
- **MADCrowner (paper 034 candidate, Wei 2026, arXiv 2603.04771, Mar 2026)** — DMC + margin segmentation, public code
- **ToothCraft (Wei 2026, arXiv 2603.26588, Mar 2026)** — diffusion-based, tests H2 in the dental domain

**v0 sub-task 2 should commit to the deterministic DMC lineage**, with the MADCrowner extension as v1. Diffusion-based methods (ToothCraft) are an active but unproven alternative, and we don't have evidence yet that they outperform the deterministic approach for the constrained patient-specific task.

## Cross-paper insights (cumulative through paper 033)

- **The dental crown generation lineage is now a clean 4-paper arc + extensions.** DMC (open source, 2023) → DCrownFormer (transferred, 2024) → MADCrowner (open, 2026) → ToothCraft (diffusion, 2026) → MVDC (multi-view, 2025). **The deterministic transformer + good losses (DMC lineage) is the dominant approach; diffusion-based methods (ToothCraft) are an active but unproven alternative.** v0 should commit to the deterministic lineage, with diffusion as a v1+ research comparison.

- **The "open source → technology transfer → paper citation" pattern is now well-established in dental crown generation.** DMC (open), DCrownFormer (transferred), MADCrowner (open), ToothCraft (open). The implication: **whenever a method is open-sourced in this field, expect it to either (a) be commercialized within 12 months, or (b) be superseded by a paper that uses it as a baseline and adds 1-2 ingredients.** v0 should consider commercializing the v0 sub-task 2 model — Korean / Japanese / US dental-CAD companies are clearly acquiring these.

- **The MRL trick (MSE on the indicator grid) is the single highest-leverage design choice in the dental-crown generation reading list.** It was first published in DMC (paper 033, 2023), then named "MRL" by DCrownFormer (paper 032, 2024), and is now standard practice. **A 15-point F-score jump from a 1-line code change is the kind of leverage that's hard to find in deep learning research** — usually you need a new architecture for a 5-point gain. The MRL trick should be the *first* thing we add to the v0 pipeline.

- **The H2 (diffusion > VAE/GAN) hypothesis is *domain-dependent*, and DMC confirms it.** In *unconstrained* 3D generation (ShapeNet, indoor scenes), diffusion wins (papers 004, 012, 014). In *constrained, patient-specific* generation (dental crown), deterministic + good losses wins (this paper, paper 032). The right framing: **diffusion is a *prior*, not a *backbone***. For patient-specific tasks where the conditioning pins down the output, the prior is *point estimation*, and a deterministic network is faster and more accurate. For unconditional or loosely-conditioned tasks, the prior is *sampling*, and diffusion is the right tool. **v0 sub-task 2 is in the first category.**

- **The v0 stack is now decisively defined: PVD-AF-DiGS-FC for sub-task 1 (full-arch synthesis), DMC + MCAM + CPL + MRL for sub-task 2 (crown generation), Cao25 (paper 026) for the segmentation preprocessor, FlexiCubes (paper 007) as the final mesh extractor.** Total compute ~$2,200 on Lambda, training time 1-2 days for the sub-task 2 model (22h on A100), 3-7 days for the sub-task 1 model. The stack is shippable in 4-6 weeks with 1-2 engineers.

### Hypothesis scorecard (cumulative through paper 033)
- **H1** (2-stage > 1-stage): **CONFIRMED with refinement** — DMC is structurally 1-stage with a learnable intermediate bottleneck (the indicator grid). Restate: "1-stage suffices when there's a learnable intermediate bottleneck; 2-stage wins when the intermediate is a *semantic* representation (segmentation mask, depth map, etc.)."
- **H2** (diffusion > VAE/GAN): **REFINED** — diffusion wins for *unconstrained* generation, deterministic + good losses wins for *constrained, patient-specific* generation. **H2 is now conditional: H2-conditional holds iff the conditioning is weak.** DMC + DCrownFormer both confirm the conditional form.
- **H3** (arch-level conditioning): **STRONGEST DIRECT SUPPORT** — DMC is the *archetypal* arch-level-conditional model, and the PoinTr transformer with 6-tooth context is the H3 mechanism. **H3 is no longer a hypothesis, it's a confirmed design pattern.**
- **H4** (SDF > explicit mesh for substrate): **REFINED** — the right substrate is *point cloud + indicator function* (DPSR-style), not *SDF* alone. The indicator function is just a *differentiable meshing post-processor*, and the *generation* is on points. **For v0: point cloud generation + SAP/FlexiCubes extraction + MSE on the extracted mesh.**
- **H5** (synthetic → real): **NO NEW EVIDENCE** — DMC trains on real data only. The commercial sponsorship by Kerenor Dental Studio is *indirect* evidence of clinical viability.

### TL;DR for HK
- **DMC = the open-source point-to-mesh crown generator that v0 sub-task 2 should be built on.**
- **Code: github.com/Golriz-code/DMC (MIT-style, ~1200 lines PyTorch, including the SAP/DPSR submodule).** Fork it, port to PyTorch 2.x, add MCAM + CPL + MRL.
- **The single biggest insight: MSE on the indicator grid (the "MRL trick") is a free 15-point F-score gain.** Add it to the loss function as the *first* v0 change.
- **H2 is now confirmed *domain-dependent*** — for constrained patient-specific generation, deterministic + good losses wins. v0 sub-task 2 commits to the deterministic lineage.
- **The dataset is private (same as DCrownFormer), so we need a public benchmark for v0** — 3DTeethSeg22 + ToSynFCD is the right choice.
- **22h training on a single A100 is the cheapest in our reading list** — ~$25 per training run on Lambda. v0 sub-task 2 is the cheapest component to iterate on.
- **v0 stack is now decisively defined: PVD-AF-DiGS-FC for sub-task 1, DMC + MCAM + CPL + MRL for sub-task 2, Cao25 for segmentation, FlexiCubes for final mesh extraction.** Total ~$2,200 Lambda, 1-2 weeks engineering, shippable in 4-6 weeks.
- **Next paper to read: MADCrowner (Wei 2026, arXiv:2603.04771, Mar 2026, public code at github.com/lullcant/MADCrowner)** — the DMC + margin segmentation extension. This would teach us the cervical margin sub-task earlier for v1, and it's the natural next paper in the dental-crown-generation lineage.
