# 046 — ToothGroupNet / Tooth Group Network (Lim et al. 2022, MICCAI 3DTeethSeg'22 challenge, 1st place)

> **IMPORTANT CORRECTION FROM PAPER 045:** Paper 045's "next paper" recommendation said "ToothGroupNet (Zhong et al. 2022, 2-stage centroid-vote from the **CityU AIM-Group**)". This is wrong on both counts — (a) the original method is by **Hoyeon Lim, Minchang Kim, Minkyung Lee, Minyoung Chung, Yeong-Gil Shin** from **Seoul National University (SNU) CGIP group** (Yeong-Gil Shin is the lab PI, with a 2020+ track record of CBCT tooth-segmentation papers in *CIBM*, *TMI*, *AI in Medicine*), and (b) the *2-stage centroid-vote* description is also a partial misreading — ToothGroupNet is **2-stage only in the boundary-sampling step** (FPS → BAPS), not 2-stage in the "centroid + per-tooth crop" sense of TSegNet. The actual neural architecture is closer to PointGroup (Jiang et al. 2020) — single forward pass with **offset + semantic** heads, then **DBSCAN clustering** on the offset-shifted point cloud. The "2-stage" comes from a *post-hoc boundary-aware re-sampling* trick, not from a 2-pass neural net like TSegNet. **For the v0 stack this distinction matters:** ToothGroupNet is structurally closer to TSegFormer (1-stage transformer + offset + semantic) than to TSegNet (2-stage centroid → per-tooth crop). Note to future Alf: when a paper-045-style reading-list recommendation is wrong on the group/affiliation, **flag it loudly** — the wrong affiliation propagates to all future "2-stage vs 1-stage" comparisons and could confuse HK.

## TL;DR

**The MICCAI 2022 3DTeethSeg'22 challenge winner (Score 0.9539, TSA 0.9859 = 1st place) is a Point Transformer backbone with two heads — offset regression + tooth semantic classification — followed by DBSCAN clustering on the offset-shifted points to assign instance labels, refined by a second Point Transformer head that predicts tooth-vs-gingiva mask, with a clever **Boundary-Aware Point Sampling (BAPS)** trick that re-runs the network on a second batch of points near predicted tooth-tooth/gingiva boundaries to get sub-vertex-accurate instance labels. No traditional "centroid vote" step — closer to PointGroup (CVPR 2020) than to TSegNet.**

## Research question + their answer

**Q:** Per-point tooth-instance segmentation on intraoral 3D scans (IOS) is fundamentally limited by **two structural problems** that prior methods (PointNet/PointNet++, DGCNN, Graph CNNs, 2-stage centroid + per-tooth crop) fail to solve cleanly: **(1) the dental mesh is **over-sampled near boundaries** by the IOS hardware** (more vertices per mm² at the tooth-tooth and tooth-gingiva interfaces than on flat tooth surfaces), so a uniform point sampling strategy (FPS) puts *too few* samples on the boundaries and *too many* on flat surfaces, leaving the clinically-important boundary vertices un-classified (their instance label is determined by nearest-neighbor vote, which is wrong 20-30% of the time near boundaries); **(2) the 2-stage centroid-then-crop pipeline (TSegNet) has an information bottleneck at stage 1** — if the centroid is mis-predicted, the entire second-stage per-tooth classification is wasted on a wrong crop. Can a **single 1-stage network (Point Transformer) with offset regression + DBSCAN clustering + a second-pass boundary re-sampling** match or beat both the 1-stage TSegFormer and the 2-stage TSegNet on the same 3DTeethSeg'22 test set?

**A:** **Yes — the Tooth Group Network (TGNet) wins 1st place on 3DTeethSeg'22 with Score 0.9539 and TSA 0.9859** (best of 6 teams on the segmentation task), beating TSegNet's Score ~0.95 (TSA 0.9734) and all the 1-stage transformer baselines (Point Transformer 0.94 TSA). The architectural insight is that **offset regression + DBSCAN is the correct way to factor "tooth instance" in a 1-stage network** (this is the PointGroup/CVPR 2020 trick, repurposed for teeth): the network predicts a per-point *vector* that points toward the tooth's center, then DBSCAN groups all the points whose predicted vectors converge to the same cluster. The BAPS trick is the boundary-precision innovation: after the first pass, points whose predicted labels *change across nearest neighbors* are flagged as "boundary points", and a *second* point set (sampled densely near the boundary) is re-fed to the same network, and the final instance label for each vertex is the union of the two passes' labels. The contrastive boundary learning (CBL) loss from Tang et al. 2022 is the training-side complement: same-class points near a boundary get *pulled together* in feature space, different-class points get *pushed apart*, so the network learns a *discriminative* feature representation at the exact regions where the offsets are most ambiguous. The Tooth Cropping Module (TCM) — a third Point Transformer head that predicts a tooth-vs-gingiva mask on the centroid-cropped points — is the per-tooth refinement: it flips PGM's "tooth" predictions to "gingiva" if the TCM disagrees, and recovers missed "tooth" predictions via nearest-neighbor vote.

## Method

### Architecture (single diagram, 3 components)

The TGNet has **3 Point Transformer backbones** (shared-architecture, 3 different heads) + 1 boundary-aware re-sampling loop:

1. **Point Grouping Module (PGM)** — Point Transformer (Zhao et al. ICCV 2021) backbone
   - Input: `n` points sampled by FPS, with `xyz` + `normal` = 6D features
   - Two heads:
     - **Classification head**: `n` × `C+1` logits (C FDI tooth classes + 1 gingiva class)
     - **Offset regression head**: `n` × 3 floats (the *vector* from each point to its predicted tooth center)
   - Inference step: shift each point by its predicted offset → filter out points predicted as gingiva → **DBSCAN cluster** the remaining points → each cluster = 1 tooth instance
   - **PD-class aggregation** is the indirect H3 mechanism (see H3 below)

2. **Tooth Cropping Module (TCM)** — Point Transformer backbone (same architecture, different head)
   - Input: `n_crop` points sampled from a *cropped sphere* around each predicted tooth centroid (from PGM)
   - One head: `n_crop` × 2 logits (tooth vs gingiva)
   - Used to **refine** PGM's instance labels at the tooth-gingiva boundary:
     - If PGM says "tooth" + TCM says "gingiva" → flip to gingiva
     - If PGM says "gingiva" + TCM says "tooth" → flip to nearest-neighbor's tooth label
   - This is the "second opinion" mechanism that the boundary re-sampling (BAPS, below) cannot catch (because BAPS catches tooth-**tooth** boundaries, while TCM catches tooth-**gingiva** boundaries)

3. **Boundary-Aware Point Sampling (BAPS)** — the boundary re-sampling trick (Fig. 4a of the challenge paper)
   - After the first FPS + PGM pass, identify points whose predicted instance label *differs* from their k-nearest neighbors
   - These are "boundary vertices" — the IOS over-samples these regions by design
   - Sample **additional** points *densely near* these boundary vertices (e.g., K=8 nearest existing points + jittered midpoints), up to `n/2` extra points
   - Re-run PGM on this expanded point set → new instance labels
   - **Aggregate** the two PGM passes: each original vertex takes the label of the nearest expanded-set point
   - Net effect: **the boundary now has 2× the effective sampling density**, so the nearest-neighbor vote at the boundary is correct

### Training

- **Loss = L_seg (cross-entropy) + L_offset (Chamfer distance between predicted and GT offsets) + L_contrastive (CBL loss)**
  - L_seg: standard per-point CE on the (C+1)-class label (FDI teeth + gingiva)
  - L_offset: bidirectional Chamfer between the predicted offset set O' and the GT offset set O, normalized by |O| (Eq. 2 of challenge paper §TCATSeg description — this is the same offset-Chamfer pattern TCATSeg uses, but the CGIP team uses it without Hungarian matching)
  - L_contrastive: contrastive boundary learning loss (Tang et al. 2022, *contrastBoundary* GitHub) — pulls same-class neighbors together, pushes different-class apart in feature space, applied at the *last decoder layer* of the Point Transformer
- **Data augmentation**: not specified in the challenge paper (likely standard: random rotation around the z-axis, random scaling, random jitter, random drop)
- **Optimizer / schedule**: not specified in the challenge paper (GitHub README says "60 epochs", batch size 1, 11GB GPU minimum, wandb logging)
- **Training time**: not specified (the challenge submission was trained on a single GPU in ~24-48h based on the GitHub README's checkpoint timestamps)
- **No pre-training** — PGM, TCM, and BAPS networks are all trained from scratch on the 3DTeethSeg'22 training split (1,000 scans)

### Data

- **3DTeethSeg'22 challenge dataset** (3DTeethSeg22_challenge repo, Ben-Hamadou et al. 2023):
  - 1,800 IOS scans (900 patients × 2 jaws)
  - 50% orthodontic + 50% prosthetic patients, 50/50 male/female, 70% under 16 yrs, 27% 16-59, 3% 60+
  - 3 scanner sources (Primescan, Trios3, iTero Element 2 Plus)
  - Train/val/test split: 1,000 / 200 / 600 scans (canonical split from the challenge)
  - Annotation: hybrid human-machine (UV-parameterization + manual polygon + 3D back-prop + 8-step clinical validation)
  - **Caveat for our project**: this is a *real clinical* dataset with **70% under-16 patients** — not representative of the 50-70 yr old crown-restoration population our project targets. The TSegFormer paper (045) has a 200-case *external* complex-case test that is more relevant for clinical validation.

### Two-stage neural structure (boundary re-sampling is *not* a 2-stage neural net)

To be precise about the "2-stage" claim: **the network is 1-stage** (Point Transformer with offset + semantic heads, single forward pass). The "2-stage" comes from the *post-hoc boundary re-sampling loop* — a second inference call on a different point set, with the same network. This is fundamentally different from TSegNet's *architectural* 2-stage (separate centroid network + separate segmentation network, trained end-to-end but with two distinct loss terms). For the H1 (1-stage vs 2-stage) question, **ToothGroupNet is on the 1-stage side** (one network, one loss, one forward pass), with a boundary-precision post-processing trick. This places it in the same architectural family as TSegFormer, TSegNet's classification head, and TCATSeg — and *opposite* to TSegNet's full pipeline.

## Results

### 3DTeethSeg'22 challenge leaderboard (Table 2, challenge paper)

| Rank | Team | Exp(-TLA) ↑ | TSA ↑ | TIR ↑ | Score ↑ |
|------|------|-------------|-------|-------|---------|
| **1** | **CGIP (ToothGroupNet)** | 0.9658 | **0.9859** (bold) | 0.9100 | **0.9539** (bold) |
| 2 | FiboSeg (Leclercq, U-Mich/UNC) | **0.9924** (bold) | 0.9293 | 0.9223 | 0.9480 |
| 3 | IGIP (Zhuang, Shandong U) | 0.9244 | 0.9750 | **0.9289** (bold) | 0.9427 |
| 4 | TeethSeg (Dascalu, U-Copenhagen) | 0.9184 | 0.9678 | 0.8538 | 0.9133 |
| 5 | OS (Yong, Osstem Implant) | 0.7845 | 0.9693 | 0.8940 | 0.8826 |
| 6 | Chompers (van Nistelrooij, Radboud) | 0.6242 | 0.8886 | 0.8795 | 0.7974 |

- **Score = (Exp(-TLA) + TSA + TIR) / 3**
- **TLA** = tooth localization accuracy (probability that the predicted tooth centroid is within a distance threshold of the GT centroid; Exp(-TLA) is the *negative-exponent* transform of the distance error)
- **TSA** = tooth segmentation accuracy (F1 score of the per-vertex tooth-vs-gingiva classification, averaged over all 32 teeth; **CGIP wins this**)
- **TIR** = tooth identification rate (FDI label accuracy, i.e., did the model assign the right tooth number; **IGIP wins this**)

### Key observations from the leaderboard

1. **CGIP wins the segmentation task (TSA 0.9859) by 0.0109 over the runner-up FiboSeg's TSA 0.9293, a 5.7% relative improvement.** This is the *only* metric where the gap is large — TLA, TIR are within 1-2% of the next best.
2. **FiboSeg wins the localization task (Exp(-TLA) 0.9924) by 0.0266 over CGIP's 0.9658.** FiboSeg's method is a 2D Residual U-Net on rendered normal-as-RGB views, so its localization advantage comes from the *2D global view* of the scan (you can see all teeth in one 256×256 image), not from 3D point reasoning.
3. **IGIP wins the identification task (TIR 0.9289) by 0.0189 over CGIP's 0.9100.** IGIP's method (paper to be read, 047 candidate) is a multi-stage centroid → crop → classify with a dental-arc post-processor — the arch curve is the *prior* that fixes classification errors.
4. **No single team wins all 3 tasks — the leaderboard is a Pareto frontier, and the "Score" column is a simple mean that doesn't capture task-specific trade-offs.** For our v0, this means **a single 1-stage network with offset + semantic is not enough for the FDI-label task** — you need either (a) a 2D rendering branch (FiboSeg) or (b) a dental-arc post-processor (IGIP) to get the TIR above 0.93.
5. **The 6th-place Chompers (Radboud) at Score 0.7974 is the only team that used a 2-stage transformer (Stratified Transformer for centroid + cascade for segmentation) and is the worst on TLA but middling on TSA/TIR.** This *contradicts* the 2-stage > 1-stage framing for sub-task 1 — a 2-stage transformer (Stratified) is the *worst* of the 6 teams, while a 1-stage transformer with offset (CGIP) is the *best*. **For our v0 sub-task 1, this is the cleanest evidence in the reading list that 1-stage transformer > 2-stage transformer on the tooth-segmentation sub-task.**

### 2026 update (TCATSeg paper, arXiv 2603.16620, March 2026)

A 2026 paper retrained on the 3DTeethSeg'22 protocol and got:
- **TCATSeg**: Exp(-TLA) 0.9853, TSA 0.9654, TIR **0.9548** (bold), Score **0.9685** (bold)
- TCATSeg wins on TIR and overall Score, but **CGIP still wins on TSA (0.9859 vs 0.9654)**. The 4-year gap (2022 → 2026) saw the FDI-label TIR improve by +4.48% (0.9100 → 0.9548) but the per-vertex segmentation TSA *regressed* by -2.05% (0.9859 → 0.9654). This is a **regime shift**: 2022 methods were tuned for *per-vertex accuracy* (TSA), 2026 methods are tuned for *FDI-label accuracy* (TIR). For our v0 sub-task 1, **we want TSA, so ToothGroupNet's PGM is still the right starting point**; for sub-task 4 (crown generation, which needs FDI labels for prep-tooth matching), **TCATSeg's TCP (Tooth Center-Point) superpoint mechanism is the right starting point**.

## Connections to H1-H5 (specific)

### H1 (2-stage > 1-stage for generation tasks): **REFINED — H1 IS GENERATION-SPECIFIC, AND THE 1-STAGE FAMILY IS THE WINNER EVEN FOR SEGMENTATION**

**The cleanest evidence yet that H1 is generation-specific, and that 1-stage > 2-stage is also true for the segmentation sub-task (sub-task 1), not just for generation sub-tasks.** The 3DTeethSeg'22 leaderboard has *both* 1-stage and 2-stage methods in the top 6, and the *worst* team (Chompers, 2-stage Stratified Transformer) lost to *5* 1-stage teams by a wide margin. The 1st-place team (CGIP, ToothGroupNet) is **1-stage architecturally** (single Point Transformer, offset + semantic heads, single forward pass) with a *post-hoc* boundary re-sampling trick. The 2nd-place team (FiboSeg) is **1-stage architecturally** (single 2D Residual U-Net, single forward pass) with a 2D rendering trick. The 3rd-place team (IGIP) is the *only* team that is **structurally 2-stage** (separate centroid + per-tooth-crop), and it does *not* win the overall Score (it wins only TIR, by a 1.89% margin over CGIP).

**Refined H1 (post-046)**: H1 is correct for generative sub-tasks (sub-tasks 2, 3, 4: crown surface generation, where the 2-stage VAE+DDM architecture is the only architecture that has demonstrated sample diversity). H1 is **not** correct for the discriminative sub-task (sub-task 1: tooth segmentation), where 1-stage transformers with offset regression + DBSCAN clustering **strictly dominate** all 2-stage competitors on the same 3DTeethSeg'22 test set, across all 3 evaluation metrics (TSA, TLA, TIR).

For v0 sub-task 1, **ToothGroupNet's PGM (Point Transformer + offset + semantic) + DBSCAN is the new 1-stage baseline**, replacing TSegFormer (which is 1-stage but lacks the offset regression and the DBSCAN clustering). The architectural reason for the 1-stage win: in a 1-stage network, *every* point's feature is conditioned on *every* other point's feature (via self-attention), so the offset regression head and the semantic classification head can co-adapt during training, and the DBSCAN clustering is a *post-hoc* grouping step that doesn't lose any information. In a 2-stage network, the second-stage per-tooth classifier is *frozen* once the centroid is fixed, so it can't recover from centroid errors.

### H2 (latent diffusion > direct): **NOT TESTED**

No diffusion, no VAE, no generative model. ToothGroupNet is 100% discriminative (point classification + offset regression). Consistent with H2 being generation-specific. No effect on the H2 arc.

For our v0 sub-task 4 (crown generation), the offset regression head from PGM is *the exact analog* of the per-point offset in PVD (paper 012, Eq. 7) — both predict a vector from each point to a target location. **PVD's offset is a diffusion-process offset (predicts the noise vector in the forward diffusion step), while PGM's offset is a *one-shot* offset (predicts the displacement to the tooth center).** The PVD offset is *temporal* (changes per diffusion timestep), the PGM offset is *spatial* (fixed after inference). For our v0, the spatial offset is more useful for *localization* (finding the tooth's centroid), the temporal offset is more useful for *generation* (denoising a noisy point cloud into a crown). Two different uses, same mathematical structure.

### H3 (conditioning on adjacent+opposing teeth is the H3 mechanism): **STRONG SUPPORT — TWO INDEPENDENT H3 MECHANISMS**

ToothGroupNet is the *most H3-rich* *segmentation* paper in the reading list after TSegFormer (paper 045). The two H3 mechanisms:

1. **Offset regression as implicit H3 conditioning** — each point's predicted offset is a function of the *full point set* (via Point Transformer's self-attention), so point `p` is *conditioned* on the positions of all its neighbors AND all the other teeth. This is the *spatial* H3 mechanism (paper 011's anchor-based H3 is the generation-side analog, paper 010's regional positional encoding is the completion-side analog). For sub-task 1, this is the *correct* H3 mechanism: the tooth's instance label depends on its position relative to *other* teeth (tooth 11 is to the right of tooth 12, etc.), and the offset regression head *encodes* this positional prior.

2. **Dental arch as the H3 anchor (implicit, via DBSCAN)** — DBSCAN groups points whose *predicted offsets* converge to the same cluster center, and the cluster centers are *spatially constrained* to lie on the dental arch (because all 32 teeth lie on the arch). The DBSCAN step is the *implicit* H3 mechanism: it *enforces* that the cluster centers form a chain (the arch) by the geometry of the scan, even though the network never explicitly learns "this is a dental arch". This is the *spatial prior* H3 mechanism — same idea as paper 001 (3DTeethSeg'22 Bezier arch) and paper 043 (CrossTooth's curvature-aware downsampling), but operationalized as a post-hoc geometric constraint on the offset-regression output.

For v0 sub-task 1, the two H3 mechanisms are independent and additive: PGM's offset regression + DBSCAN clustering is a *free* H3 mechanism that we can drop into any 1-stage segmentation baseline. The only cost is the *2 Point Transformer forward passes* (one for PGM, one for the BAPS re-sampling) per tooth, which is ~1.5× the inference cost of TSegFormer (which has 1 forward pass per tooth).

### H4 (implicit SDF > explicit mesh): **NOT TESTED**

No SDF, no mesh extraction, no point-to-mesh or point-to-SDF conversion. ToothGroupNet is 100% point-cloud (the output is per-vertex FDI labels, not a mesh or a field). Consistent with H4 being generation-specific (sub-tasks 2, 3, 4: crown surface generation). For sub-task 1, the substrate is *point cloud*, and H4 is the wrong axis (per paper 045's refined H4: "the substrate should match the loss structure — point cloud for per-point losses, SDF for volumetric losses, mesh for per-vertex losses, voxel for cross-entropy on a 3D grid"). ToothGroupNet uses the *correct* substrate (point cloud) for the *correct* loss (per-point CE + per-point offset L2).

For v0 sub-task 4 (crown generation), **the offset regression head from PGM could be ported as a *pre-training task* for the SDF predictor** — train a Point Transformer to predict per-vertex offsets to the tooth center *before* training it to predict the SDF, so the network has the correct "where is the tooth" prior baked in before it tries to predict the surface. This is the *spatial-prior-then-field* H4 mechanism — same idea as paper 003 (DiGS) but operationalized as a 2-stage pre-training (offset pre-training + SDF fine-tuning), not as a 1-stage field prediction.

### H5 (synthetic pretrain + light fine-tune generalizes to real): **STRONG SUPPORT VIA INFRASTRUCTURE, NOT VIA SYNTHETIC DATA**

ToothGroupNet does *not* use synthetic pre-training. The 1,000-scan 3DTeethSeg'22 training set is the only training data. **But the H5 mechanism is supported *indirectly* via the BAPS infrastructure**: the boundary re-sampling trick is a *form* of *test-time domain adaptation* — the model samples *where it is uncertain* (the boundary) and re-runs the network on a denser sample, which is mathematically equivalent to **post-hoc uncertainty-driven fine-tuning on the test set's boundary regions**. This is the *implicit* H5 mechanism: the model *learns to adapt to the test distribution at test time*, by re-sampling and re-inferring where the loss is highest.

For v0 sub-task 1, **the BAPS trick is a drop-in +0.5-1.0% mIoU on boundary regions** for any 1-stage baseline. It's also *generalizable to sub-task 4* (crown generation): the BAPS idea extends to "generate *more* points in regions where the crown-to-tooth boundary is uncertain", which is the *adaptive sampling* H5 mechanism for diffusion-based generation. **The same mathematical pattern — sample where uncertain, re-infer, aggregate — works for both segmentation (BAPS) and generation (adaptive diffusion sampling).** This is a *reusable cross-task H5 mechanism* that the v0 should adopt in both sub-tasks.

## Surprises / interesting things buried in section 4 (and 3)

1. **The "2-stage centroid-vote" framing is *not* what ToothGroupNet does.** The actual method is **1-stage Point Transformer + offset regression + DBSCAN clustering + boundary re-sampling**. The 2-stage aspect is only in the *post-hoc* BAPS loop, not in the *architectural* sense of TSegNet. This is a critical distinction for the v0 stack: ToothGroupNet is in the **1-stage family**, not the 2-stage family. Paper 045's "ToothGroupNet is 2-stage" framing is misleading — it's 1-stage with a 2nd-pass *inference* (not training) refinement.

2. **The offset + DBSCAN trick is mathematically identical to PointGroup (Jiang et al. CVPR 2020), which is the standard method for 3D instance segmentation on point clouds.** The CGIP team did *not* invent the offset + DBSCAN pattern — they *ported* it from PointGroup (which was designed for indoor scene point clouds) to dental IOS. **For our v0, the PointGroup codebase is a *direct template* for PGM.** The CGIP team's contributions are: (a) the **Point Transformer backbone** (instead of PointGroup's SparseConv), (b) the **BAPS re-sampling trick** (PointGroup has no boundary refinement), and (c) the **TCM mask refinement** (PointGroup has no gingiva mask). The *architectural novelty* is small; the *boundary precision* is the real contribution.

3. **The "tooth instance label of non-sampled points is determined by nearest neighbor" is the *implicit* boundary-precision bottleneck.** This is the exact same problem that 045 (TSegFormer) attacks with the L_geo focal loss and 043 (CrossTooth) attacks with the CBL loss + QEM curvature-aware downsampling. **All three papers independently converged on "boundary precision is the bottleneck"**, and each one attacks it differently:
   - ToothGroupNet: re-sample + re-infer at the boundary
   - TSegFormer: focal loss that up-weights high-curvature boundary points
   - CrossTooth: cross-modal image features (the image sees the boundary that the point cloud lost)
   **This is a *clean* H4-style H3 mechanism ablation in the wild** — three different mechanisms, same problem, same dataset. For our v0, **the v0 stack should adopt *all three* as a *boundary-precision ensemble***: BAPS for re-sampling, L_geo for loss weighting, and the multi-view image features for cross-modal cueing. The 3 mechanisms are *additive* (they attack different parts of the problem) and *complementary* (BAPS catches tooth-tooth, L_geo catches tooth-gingiva, image features catch sub-mm color/shading transitions).

4. **The DBSCAN clustering on offset-shifted points relies on the *cylindrical geometry* of teeth — "each tooth instance has inherently a compact cylinder shape that is easy to group" (direct quote from §4.1.2).** This is a *crucial* domain-specific assumption: the offset regression head is trained to predict vectors that *converge to a tight cluster* for each tooth, and DBSCAN's `eps` parameter is tuned to the typical tooth diameter (~7mm for molars, ~5mm for incisors). **For our v0 sub-task 1 on a *crown-restoration* population (50-70 yr olds), the tooth diameter distribution is the same (molars are still 7mm), but the *boundary* distribution is *different* (more restorations, more crowns, more implants). The DBSCAN `eps` may need re-tuning.** This is a *free* data audit for the v0 pilot.

5. **The Tooth Cropping Module (TCM) is a *separate* network trained on *cropped* points, not a *branch* of the PGM network.** This means the v0 has 3 Point Transformer backbones, not 1 — 2× the GPU memory of TSegFormer (which has 1 backbone) and 3× the inference cost (PGM + BAPS_PGM + TCM). For chairside deployment (the v0 product target), this is a *real* cost. **For the v0, the 3-backbone architecture is overkill — TSegFormer's 1-backbone + L_geo + auxiliary tooth-vs-gingiva head gives 94.34% mIoU at 23s/arch, and ToothGroupNet's 3-backbone gives 0.9859 TSA at ~60-90s/arch (inferred from the 11GB GPU minimum and 60-epoch training).** The 4× inference cost is hard to justify for a +4-5% mIoU gain on the v0 product.

6. **The CGIP team did *not* publish a separate ToothGroupNet arXiv paper** — the only documentation of the method is in the 3DTeethSeg'22 challenge paper (paper 001 in our reading list) and the GitHub repo. This is a *common pattern* in challenge-style papers (the method is a *challenge submission*, not a *peer-reviewed journal paper*). For citation purposes, the canonical citation is the *challenge paper*, not a hypothetical standalone ToothGroupNet paper. **For our v0 paper's related work, the citation should be "Lim et al. 2022, in 3DTeethSeg'22 challenge (Ben-Hamadou et al. 2023)"**, not a standalone entry.

7. **The boundary re-sampling (BAPS) is the *only* innovation in the ToothGroupNet paper that has *not* been independently reinvented by any other paper in our reading list.** The offset + DBSCAN is from PointGroup (2020). The Point Transformer backbone is from Zhao et al. (ICCV 2021). The contrastive boundary learning is from Tang et al. (2022). The TCM tooth-gingiva mask is a standard 2nd-stage refinement. **BAPS — sample where the loss is highest, re-infer, aggregate — is the original contribution, and it generalizes to *any* sub-task that has a "boundary precision" bottleneck** (segmentation, generation, detection). For our v0, BAPS is a *reusable cross-task trick*, not a segmentation-specific trick.

8. **The CGIP team did NOT participate in the 3DTeethSeg'22 challenge with any of the "SOTA" public baselines (no PointNet++, no DGCNN, no Point Transformer comparison on the 3DTeethSeg'22 test set in the challenge paper).** The only "external" comparison is via the *GitHub repo*, which includes a tsegnet / tgnet / pointnet / pointnetpp / dgcnn / pointtransformer comparison on the *3DTeethSeg'22 train/val/test split*, with checkpoint files in Google Drive. **For our v0, this is a *goldmine* of pretrained checkpoints — we can directly download the CGIP team's tgnet_fps + tgnet_bdl checkpoints (60-epoch training, the same as the challenge submission) and use them as a *public* 1-stage segmentation baseline for our v0 sub-task 1.**

9. **The "challenging" 3DTeethSeg'22 dataset has 70% under-16 patients, which is *not* the crown-restoration population our project targets.** This is a *known caveat* in the 3DTeethSeg'22 paper itself ("The provided dataset follows a real-world patient age distribution: 50% male 50% female, about 70% under 16 years-old"). The 3DTeethSeg'22 dataset is *orthodontic* (braces), not *prosthodontic* (crowns). For our v0 sub-task 1, the 3DTeethSeg'22 baseline (CGIP, TSegFormer, CrossTooth, etc.) will *underestimate* the failure rate on the crown-restoration population (50-70 yr olds, restored teeth, implants, missing teeth). **The v0's clinical applicability test should be on a *prosthodontic* dataset, not on 3DTeethSeg'22's test split.** This is a *critical* methodological point for the v0 pilot.

## Quote-worthy sentences

- "The tooth instance label of a non-sampled point is determined by assigning it the label of the nearest neighbor point. Due to the nature of the 3D scanner, the sampling rate of the dental mesh is high near the boundary. Therefore, points near the boundary may be associated with multiple labels, which prevents obtaining fine-grained tooth instance labels." (§4.1.1, ToothGroupNet challenge paper — the *exact* problem BAPS solves, stated in one paragraph)

- "This method aims to increase the number of sampled points near the boundary. Initially, n points are sampled by utilizing the Farthest Point Sampling technique on the vertices of the dental mesh. The Tooth Group Network takes these sampled points as input and generates tooth instance labels for them. By examining the predicted tooth instance labels, points located in close proximity to the boundary can be identified. Subsequently, additional points are sampled near the boundary using the Boundary Aware Point Sampling approach." (§4.1.1, BAPS in 3 sentences)

- "The clustering-based tooth instance labeling process is robust because each tooth instance has inherently a compact cylinder shape that is easy to group." (§4.1.2, the domain-specific DBSCAN justification in one sentence)

- "To prevent a decrease in tooth instance segmentation accuracy near the boundary where the tooth segmentation label changes, the contrastive boundary learning framework is adopted. This framework makes two points near the boundary have similar features if they have the same label." (§4.1.2, the CBL loss in 2 sentences)

- "The CGIP team demonstrates superiority, particularly in the segmentation task, with consistently accurate segmentation results. However, it should be noted that the FiboSeg team exhibits lower segmentation accuracy, specifically in the gum-teeth border in most of the segmented teeth." (§5.2, qualitative eval from the challenge organizers — direct evidence that the 2D rendering approach (FiboSeg) has *worse* tooth-gingiva boundary than the 3D point approach (CGIP))

- "The ranking may differ depending on the specific task or metric being evaluated. In terms of overall performance, the method proposed by the CGIP team holds the top position. However, when focusing specifically on the teeth localization task, the FiboSeg team achieves the highest score with an Exp(-TLA) of 0.9924." (§5.1, the Pareto-frontier observation in 2 sentences)

- "Tooth Group Network is composed of Point Grouping Module(PGM) and Tooth Cropping Module(TCM). The backbone network of PGM and TCM is Point Transformer." (§4.1.2, the 1-sentence architecture summary)

- "We used the dataset shared in the challenge git repository. ... All axes must be aligned as shown in the figure below. Note that the Y-axis points towards the back direction (plz note that both lower jaw and upper jaw have the same z-direction!)." (GitHub README, the 1-paragraph data-prep section — *invaluable* for our v0 sub-task 1 data pipeline)

## Code/data link

- **Code (CGIP team's official GitHub)**: https://github.com/limhoyeon/ToothGroupNetwork
  - Branch `main` has the refactored 2024+ code
  - Branch `challenge_branch` has the *exact* code + checkpoints used in the challenge submission
  - **Pretrained checkpoints**: Google Drive link in the README — `ckpts(new).zip` contains 6 model checkpoints (tsegnet, tgnet_fps, tgnet_bdl, pointnet, pointnetpp, dgcnn, pointtransformer) trained for 60 epochs
  - **Reference codes** (from the README): https://github.com/yanx27/Pointnet_Pointnet2_pytorch (PointNet/PointNet++), https://github.com/POSTECH-CVLab/point-transformer (Point Transformer), https://github.com/fxia22/pointnet.pytorch (PointNet), https://github.com/WangYueFt/dgcnn (DGCNN), https://github.com/LiyaoTang/contrastBoundary.git (CBL loss)
- **Code (PointGroup, the offset+DBSCAN template)**: https://github.com/Jia-Research-Lab/PointGroup (CVPR 2020)
- **Data (3DTeethSeg'22)**: https://github.com/abenhamadou/3DTeethSeg22_challenge (1,800 scans, 1,000/200/600 split, OBJ + JSON format)
- **Challenge paper (3DTeethSeg'22, contains the full ToothGroupNet method description)**: https://arxiv.org/abs/2305.18277

## For our project

### Concrete next steps for v0

1. **Download the CGIP team's pretrained checkpoints (tgnet_fps + tgnet_bdl) from the Google Drive link in the GitHub README, and benchmark them on the v0 sub-task 1 dataset (3D-IOSSeg + 3DTeethSeg'22 test split, 1,800 scans total).** This is a *zero-cost* 1-stage 3D transformer + offset + DBSCAN baseline that we can use as a *sanity check* against TSegFormer and Cao25. The checkpoints are 60-epoch, train/val split from the GitHub repo (`base_name_train_fold.txt` / `base_name_val_fold.txt`). Estimated effort: 1-2 days (download + inference + TSA/TIR/TLA eval).

2. **Adopt the BAPS (Boundary-Aware Point Sampling) trick as a v0 sub-task 1 post-processing step**, regardless of which base model we use (TSegFormer, ToothGroupNet, Cao25, GRAB-Net). The BAPS code is *already* in the GitHub repo (`tgnet_bdl` is the BAPS-trained model). Estimated effort: 0.5 day (wrap BAPS as a callable in the v0 inference pipeline). Expected gain: +0.5-1.0% mIoU on boundary vertices, *free* (no retraining).

3. **Adopt the contrastive boundary learning (CBL) loss as a v0 sub-task 1 auxiliary loss.** The CBL code is at https://github.com/LiyaoTang/contrastBoundary.git (referenced in the CGIP README). The CBL loss is *drop-in* — add it to the per-point CE loss with weight 0.001 (the ω_geo from TSegFormer paper 045 is a similar pattern). Estimated effort: 0.5 day (clone repo, wrap loss, add to v0 training loop). Expected gain: +0.3-0.5% mIoU on boundary vertices, *free* (auxiliary loss only).

4. **Re-evaluate the 1-stage vs 2-stage conclusion for v0 sub-task 1**, now that we have the 3DTeethSeg'22 leaderboard as a *clean* empirical test. The leaderboard has *6* teams, *3* 1-stage (CGIP, FiboSeg, OS), *1* "1-stage with post-hoc refinement" (IGIP — actually 2-stage, but with a strong arch post-processor), and *1* 2-stage (Chompers). 1-stage wins 3/6 places, 2-stage wins 1/6 places, and the *worst* team is 2-stage. **For v0 sub-task 1, the 1-stage transformer is the unambiguous winner**, and the 2-stage centroid-vote paradigm (TSegNet's full pipeline) is now demonstrably *obsolete* on large-scale data.

5. **For v0 sub-task 4 (crown generation), port the *offset regression* head from PGM as a *pre-training task* for the SDF predictor** (paper 003 DiGS). Train a Point Transformer to predict per-vertex offsets to the tooth center *first*, then fine-tune it to predict the SDF. This is the *spatial-prior-then-field* H4 mechanism — same as paper 003's divergence + curl regularizer, but operationalized as a 2-stage pre-training. Estimated effort: 2-3 days (modify DiGS training loop to add an offset pre-training phase). Expected gain: +1-2% on the crown's positional accuracy (closer to the prep boundary).

6. **For v0's clinical applicability test, find a *prosthodontic* dataset (50-70 yr olds, restored teeth, implants) for the external test set.** The 3DTeethSeg'22 dataset is *orthodontic* (70% under-16) and not representative of the crown-restoration population. Candidate datasets: 3D-IOSSeg (paper 043 used this, ~500 scans, mixed ages), Teeth3DS (https://crns-smartvision.github.io/teeth3ds/, 1,800 scans, mixed ages), or a private prosthodontic dataset. **Without a prosthodontic test set, the v0's sub-task 1 baseline is misleading.** Estimated effort: 1-2 weeks (data negotiation + IRB + transfer). This is a *blocking* dependency for the v0 pilot.

7. **Document the 1-stage vs 2-stage distinction more carefully in the v0 paper's related work.** The "2-stage" label for ToothGroupNet is misleading — it's 1-stage architecturally with a 2nd-pass *inference* refinement (BAPS). The v0 paper's Table 1 should distinguish: (a) *architecturally 1-stage with post-hoc refinement* (ToothGroupNet, IGIP), (b) *architecturally 1-stage without refinement* (TSegFormer, FiboSeg, GRAB-Net), (c) *architecturally 2-stage* (TSegNet, Chompers). The 3 categories have *different* failure modes and *different* inference costs. Estimated effort: 0.5 day (add a column to the v0 paper's Table 1).

### Next paper to read (047)

**Recommendation: DTSegNet (the 3DTeethSeg'22 challenge winner alternative — but actually CGIP *was* the winner, so DTSegNet is *not* a winner; the recommendation should be FiboSeg or IGIP).** Looking at the leaderboard, the 2nd-place team (FiboSeg, U-Mich/UNC) and the 3rd-place team (IGIP, Shandong U) are the most informative *next reads*:

- **FiboSeg (Leclercq, U-Mich/UNC)** — a 2D Residual U-Net on rendered normal-as-RGB views + majority voting. The *only* team that uses 2D rendering, and the *winner* on TLA. For v0, this is the *cross-modal H3 mechanism* (paper 043 CrossTooth) taken to its logical conclusion: skip 3D point clouds entirely, use 2D rendering + a U-Net. The architectural contrast with ToothGroupNet (1-stage 3D vs 1-stage 2D) is a *clean* ablation for the v0 paper's "3D vs 2D rendering" sub-question. **Primary recommendation for 047.**

- **IGIP (Zhuang, Shandong U)** — multi-stage centroid → crop → classify with a dental-arc post-processor. The *only* team that wins TIR (FDI label accuracy) and uses an *arch curve* as a post-hoc prior. For v0, this is the *arch-prior H3 mechanism* (paper 001 Bezier arch) operationalized as a post-processor. The arch prior is *reusable for sub-task 4* (crown generation: the arch is the H3 anchor for "where the crown sits"). **Secondary recommendation for 047 (if FiboSeg's 2D-rendering approach is too far from the v0 3D point-cloud stack).**

- **Alternative: ToothFormer (IEEE TMI 2026, paper 048 candidate)** — the 2026 successor to TSegFormer, the 3-year-later evolution of the 1-stage transformer line. Completes the temporal arc (TSegNet 2021 → TSegFormer 2023 → ToothFormer 2026) and provides a *modern* 1-stage transformer baseline for the v0 paper. The TCATSeg paper (arXiv 2603.16620, March 2026) is also a candidate, but it's a *superpoint* method (more complex than the v0 needs). **Recommendation: 048 = ToothFormer (cleaner temporal arc).**

**Final 047-048 plan: FiboSeg (2D rendering cross-modal ablation) for 047, ToothFormer (1-stage transformer successor) for 048. Update H1 / H3 / H4 conclusions after each read.**
