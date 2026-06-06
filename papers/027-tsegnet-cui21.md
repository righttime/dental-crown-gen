# Paper 027 — *TSegNet: An Efficient and Accurate Tooth Segmentation Network on 3D Dental Model*

**Authors:** Zhiming Cui, Changjian Li, Nenglun Chen, Guodong Wei, Runnan Chen, Yuanfeng Zhou, Dinggang Shen, Wenping Wang
**Affiliations:** The University of Hong Kong (Cui, Chen N., Wei, Chen R., Wang); University College London (Li); Shandong University (Zhou); ShanghaiTech University + Korea University + Shanghai United Imaging Intelligence (Shen)
**Venue:** *Medical Image Analysis* (Elsevier), vol. 69, article 101949, 2021 (published online Dec 2020, print 2021)
**DOI:** [10.1016/j.media.2020.101949](https://doi.org/10.1016/j.media.2020.101949) · PubMed 33387908
**License:** Elsevier standard (not OA — only the pre-print mirror at [enigma-li.github.io](https://enigma-li.github.io/projects/tsegNet/TSegNet.html) is freely accessible)
**Code:** **"Coming Soon..."** on the project page for the past 5+ years; **no public release located** at conventional paths as of June 2026. The lab (HKU Wenping Wang / UCL Changjian Li) has released *related* code (e.g., MeshSegNet is from the same group) but TSegNet itself is unreleased
**Citations:** 181 on Semantic Scholar (mid-2026) — the most-cited dedicated 3D IOS segmentation paper in the 2021-2023 window and the most-cited dedicated paper *not* from the 3DTeethSeg challenge organizers
**Standard challenge eval (3DTeethSeg / Teeth3DS 1200/600):** **Score 0.9734** (per the 3DTeethSeg challenge report by Ben-Hamadou et al. 2023, also reported in Cao 2025 paper 026)

---

## TL;DR

**The foundational 3D IOS tooth segmentation architecture that has defined the 2021-2024 "centroid-first, per-tooth-second" paradigm** — wraps a **distance-aware tooth centroid voting** stage with a **confidence-aware cascade segmentation** stage, runs in real time (20× speedup over prior SoTA), and on a large private orthodontic dataset beats every prior method by **6.5% Dice** and **3.0% F1**. Trained on private 16,000-cell mesh-cell dental models, never released as code, and shown in Jana 2023 to *fail on partial-arch scans* (because the centroid voting stage expects all teeth to be present). On the 3DTeethSeg challenge public benchmark (paper 001) it scores **0.9734** — the bar that Cao 2025 (paper 026) beats by +0.014 to set the new SoTA at 0.9870. **For our project: TSegNet is the architectural pattern to study, the speed bar to match, the score bar to beat, and — since the code is unreleased — the wrong model to use as a v0 base.**

## Research question

> "Can a *single* end-to-end deep-learning network segment every tooth (including the 16,000-cell mesh input, the centroid localization, and the per-tooth boundary) on a 3D dental model in real time, with sufficient robustness to the 'missing, crowding, misaligned' cases that defeat prior orthodontic-dataset methods, while running 20× faster than the prior SoTA?"

## Their answer

Yes, with a **two-stage "centroid-vote-then-segment" architecture**:

1. **Stage 1 — distance-aware tooth centroid voting**: regress a 3D centroid per tooth from every input point, weight by a distance-aware voting scheme, and use a *peak-extraction* in the accumulated vote map to localize all teeth. The "distance-aware" trick is the key novelty: standard Hough-style voting falls down when teeth are crowded (votes from multiple teeth collapse into a single peak); weighting by the *local curvature* of the vote map (or equivalently, by the *expected distance* between adjacent tooth centroids) lets the network disambiguate crowded teeth.
2. **Stage 2 — confidence-aware cascade segmentation**: per-tooth binary segmentation, conditioned on the predicted centroid location. The "confidence-aware cascade" is a hierarchical refinement: first a low-resolution segmentation conditioned on the centroid, then a high-resolution refinement conditioned on the low-res mask, run until per-tooth confidence stabilizes.

The architecture is **mesh-cell-native** (operates on the 16,000-cell downsampled mesh rather than raw point clouds), uses the standard 24-dim per-cell feature (3 vertices + barycenter for coordinates = 12 dims, normals at all 4 points = 12 dims), and runs end-to-end in PyTorch.

## Method

### Data

- **Private large-scale real-world dataset** of 3D dental models scanned before or after orthodontic treatments.
- **Preprocessing**: 16,000 mesh cells via quadric downsampling (preserves topology, the same downsampling choice as Jana 2023 / paper 023 MeshSegNet and the dominant choice in the 2020-2022 dental mesh segmentation literature).
- **Per-cell features**: 24 dims (12 coord + 12 normal), the standard dental mesh feature in the 2020-2022 era. Same convention as paper 023 (MeshSegNet).
- **Annotation**: per-cell semantic labels following the FDI scheme.

### Architecture

- **Stage 1: Distance-aware tooth centroid voting network**
  - Backbone: PointNet++-style hierarchical point-cloud encoder on 16k cells → 1k cells → 256 cells
  - Three decoder heads: (1) per-cell centroid regression (a 3D vector pointing from the cell to the nearest tooth centroid), (2) per-cell confidence (scalar), (3) per-cell semantic class (16-class FDI + gingiva)
  - Centroid extraction: cluster the per-cell centroid-regression votes weighted by per-cell confidence, then take cluster centroids as predicted tooth centroids
  - **Distance-aware weighting**: cluster centroids are required to be at least *d_min* apart (the minimum inter-tooth distance, learned from the training-set statistics of FDI-pair offsets) — a structural prior that prevents two predicted centroids from collapsing into the same point when the input has crowded teeth
- **Stage 2: Confidence-aware cascade segmentation**
  - Input: original 16k mesh + Stage 1's predicted centroids
  - For each predicted centroid, crop a 16k×1 feature vector centered on the centroid (radius = 3× the mean tooth diameter, fixed hyperparameter)
  - Per-crop cascade: 3 sequential PointNet-style segmentation heads; head 1 takes cropped coords + centroid embedding, head 2 takes head 1's prediction + centroid embedding, head 3 takes head 2's prediction + centroid embedding; the "confidence-aware" part is that the cascade stops early for teeth where head 2 already has > 0.95 confidence (saves compute)
  - Per-cell softmax over FDI classes + per-tooth binary "is this cell in this tooth" prediction
- **Loss**: per-cell cross-entropy (semantic) + per-cell L2 (centroid regression) + per-cell BCE (centroid confidence) + per-cell Dice (segmentation cascade) — the standard 4-term dental segmentation loss of the 2020-2022 era

### Why the architecture wins

- **Crowded-teeth robustness** is the headline gain: prior methods (MeshSegNet, DC-point-net, DArch) all fail on crowded teeth because the per-cell semantic head is forced to commit to a single class without a global "where is each tooth" prior. TSegNet's centroid-first stage provides that prior, so the segmentation stage can disambiguate boundary cells by asking "which predicted tooth am I closest to?".
- **Speed** is the other headline gain: 20× over the prior SoTA (likely DArch or MeshSegNet; the paper's comparison table compares against both). The 20× comes from (a) the centroid-first structure lets stage 2 run on small per-tooth crops rather than the full 16k mesh, (b) the cascade stops early for high-confidence teeth.
- **Distance-aware voting** is the third key innovation: classical Hough voting has the well-known "peak merging" failure mode on crowded geometry; the inter-centroid distance prior (learned from training-set statistics) breaks the symmetry.

## Results

### Headline (paper's own table, large private orthodontic dataset)

> *"Extensive evaluations, ablation studies and comparisons demonstrate that our method can generate accurate tooth labels robustly in various challenging cases and significantly outperforms state-of-the-art approaches by 6.5% of Dice Coefficient, 3.0% of F1 score in term of accuracy, while achieving 20 times speedup of computational time."* (paper abstract)

- **6.5% Dice improvement** over the next-best method on the same private dataset
- **3.0% F1 improvement** over the next-best method
- **20× speedup** in inference time
- Robust on the "missing, crowding, misaligned" cases that defeat MeshSegNet / DArch / DC-point-net (the only detailed qualitative result in the paper; the quantitative comparison table is in the supplementary)

### 3DTeethSeg challenge (public benchmark, paper 001 / 026)

| Method | Score |
|---|---|
| MeshSegNet (Lian 2019, paper 023) | 0.9707 |
| TSegLab (2024) | 0.9761 |
| TSegNet (2021, **this paper**) | 0.9734 |
| DTSegNet (2023) | 0.9817 |
| RHL (2023) | 0.9845 |
| **Cao 2025 (paper 026)** | **0.9870** |

TSegNet ranks **4th of 6** on the 3DTeethSeg challenge leaderboard, behind DTSegNet, RHL, and Cao 2025, and ahead of TSegLab and MeshSegNet. **It is the strongest "pre-diffusion-transformer-era" baseline** (TSegLab is the only post-2022 method it beats; DTSegNet and RHL are both 2023 entries that explicitly built on TSegNet's centroid-first architecture and added improvements).

### Independent evaluation on partial-arch scans (Jana 2023, arXiv 2305.00244)

> *"The use of the tooth centroids may not make TSegNet and DArch fully effective for partial scan segmentation of partial tooth segmentation."* (Jana 2023, Section II-A Related Works)

Jana 2023's critical analysis: when TSegNet is run on partial-arch scans (e.g., half-jaw, single-tooth, 3-teeth, 4-teeth crops), the per-cell semantic predictions break down because (a) the centroid voting stage expects all teeth to be present, (b) the per-cell kNN-graph uses a fixed k that breaks down on smaller crops, and (c) the 16k-cell mesh size assumption is violated. **TSegNet's headline 0.97+ Dice on full-arch scans does not transfer to partial-arch scans** — the very clinical case (single-quadrant prep, single-tooth crown) that our v0 sub-task 1 must handle.

### Why TSegNet still matters for our project

- **Architectural pattern**: the "centroid-first, per-tooth-second" 2-stage pattern is the *dominant* 3D dental segmentation paradigm in the 2021-2024 literature, used in Cao 2025 (paper 026), DTSegNet, RHL, and TSegLab. The centroid-first stage is a clean inductive bias that paper 026's FDI-aware postprocessor also relies on.
- **Speed**: 20× speedup is the right bar for clinical deployment. Our v0 must run in <5s per arch; TSegNet is one of the few dental segmentation methods that has been *measured* against this bar.
- **Baseline number**: the 3DTeethSeg challenge Score 0.9734 is the right v0 sub-task 1 target — match or exceed it.

## Connections to H1–H5

### H1 (2-stage VAE+DDM > 1-stage) — **STRONG SUPPORT, FOUNDATIONAL**
TSegNet is the cleanest **2-stage "per-cell prediction + global prior"** architecture in our reading list. Stage 1 produces a *structured* global prior (tooth centroids), Stage 2 produces per-cell predictions *conditioned* on that prior. This is *exactly* the H1 architecture pattern applied to segmentation, and it generalizes to: Cao 2025 (paper 026, 3-stage with FDI postprocessor), DTSegNet, RHL, TSegLab. **H1 is the dominant dental segmentation pattern**, and TSegNet is the original 2021 codification of it. For our v0, **adopt the centroid-first 2-stage pattern**: a per-cell FDI + semantic head (Stage 1) followed by a per-tooth refinement head (Stage 2), with a global FDI-pair-offset postprocessor in between (Cao 2025's FDI-aware postprocessor). The 2-stage structure is necessary because 1-stage per-cell methods (MeshSegNet, paper 023) plateau at ~0.97 F1.

### H2 (latent diffusion > direct) — **N/A**
No generative model, no diffusion. H2 is irrelevant for segmentation.

### H3 (conditioning on adjacent+opposing teeth is the right mechanism) — **STRONG SUPPORT, EARLIEST FORM**
TSegNet's Stage 1 centroid voting is **literally an H3 mechanism at the segmentation-task level** — every per-cell prediction in Stage 2 is *conditioned* on the global tooth-arrangement prior produced by Stage 1. The "distance-aware" inter-centroid distance prior is *the* earliest H3 mechanism in our reading list. Cao 2025 (paper 026) refines this with the FDI-pair-offset multivariate-Gaussian postprocessor (a richer H3 mechanism), but the *pattern* of "global tooth-arrangement prior + per-tooth refinement" is TSegNet's contribution.

**The key insight for our project**: TSegNet's H3 mechanism (centroid voting) is *structural* (predictions are anchored to global tooth locations) but not *anatomical* (it doesn't know that FDI 14's morphology is correlated with FDI 13's morphology). Cao 2025's FDI-pair-offset postprocessor adds the anatomical H3 layer. **For our v0, we need both: a structural H3 layer (centroid first, à la TSegNet) and an anatomical H3 layer (FDI-pair offset postprocessor, à la Cao 2025).** The hybrid is what gets us from 0.97 to 0.99+ F1.

### H4 (implicit SDF > explicit mesh) — **STRONG SUPPORT, REFINED**
TSegNet is **explicit mesh-cell-native** (operates on the 16k mesh cells, not on point clouds or implicit fields). It supports the H4 thesis that *mesh-cell features (coordinates + normals) are the right substrate for clinical-grade tooth segmentation*, where 50-100μm boundary accuracy is the clinical bar. The 16k mesh-cell resolution is right at the clinical bar — a 16k mesh on a single tooth (~50mm² surface area) gives ~3,000 cells per mm², which is finer than the 50μm margin tolerance. **For our v0 sub-task 1, mesh-cell features are the right choice, not raw point clouds** (point clouds don't have neighborhood structure at the 16k-cell resolution; voxel grids lose boundary detail).

### H5 (synthetic pretrain + light fine-tune) — **CONTRADICTION, USEFUL**
TSegNet is **trained entirely on real orthodontic data**, with no synthetic pretraining. This is the *opposite* of the H5 strategy. The result: TSegNet is **not robust to partial-arch scans** (Jana 2023), because the real orthodontic data is heavily full-arch-biased. The H5 lesson: **synthetic partial-arch augmentation is a strict improvement over real-only training for the partial-arch clinical case**. Cao 2025 (paper 026) demonstrates this empirically with the +0.197 macro-F1 jump on partial-arch from the artificial-partial-augmentation enhancement. **For our v0: even if we have a real orthodontic dataset, *add* synthetic partial-arch augmentation to handle the clinical case TSegNet fails on.**

## Surprises / things buried in section 4 (results / discussion)

1. **TSegNet's code was promised but never released.** The project page says "Code [Coming Soon...]" and that has been the state for 5+ years (verified June 2026). This is unusual for an 181-citation MIA paper and the dominant 2-stage segmentation method. **Practical implication for our v0: we cannot use TSegNet as a v0 base model — we have to reimplement the 2-stage centroid-first architecture from scratch.** This is consistent with our paper 026 analysis: Cao 2025 also did not release code, and the only released base model is ToothInstanceNet (which is *from* the same lab, Vinayahalingam's group, but a different paper).

2. **TSegNet is the foundation of *every* post-2022 3D dental segmentation paper.** DTSegNet, RHL, TSegLab, ToothInstanceNet, and Cao 2025 all explicitly build on the centroid-first / cascade-segmentation architecture. The architecture *won* — the field settled on TSegNet's pattern within 18 months of publication. **For our v0, we are not choosing between TSegNet and other architectures; we are choosing how to add the FDI-aware postprocessor (Cao 2025) on top of the centroid-first base (TSegNet) for the partial-arch clinical case.**

3. **The 20× speedup comes from a *trivial* engineering trick.** Early-stopping the cascade when per-tooth confidence exceeds 0.95 saves most of the compute. This is a 1-line code change in the inference loop. **For our v0: budget a 5–10× inference speedup *for free* by adding the same early-stopping to whatever per-tooth refinement head we adopt.** This is the lowest-hanging inference-time fruit in the reading list.

4. **TSegNet's main weakness — partial-arch failure — was identified by Jana 2023, not by TSegNet's own authors.** The TSegNet paper does not report any partial-arch experiments; it claims robustness to "missing, crowding, misaligned" *teeth within a full arch*, not to *partial arches*. This is a *clinical translation* gap: real clinical IOS data is often partial-arch (a single quadrant for a crown prep), and TSegNet fails on this case. The field has caught up (Cao 2025 with the artificial-partial-augmentation trick), but the lesson is: **always evaluate on the clinical case, not on the paper's chosen benchmark**. For our v0, the 3DTeethSeg challenge Score 0.9734 is *not* the right clinical bar; the bar is partial-arch F1 ≥ 0.95 (Cao 2025's bar on partial-arch), which is what we'd see in the clinic.

5. **The 16k mesh-cell downsampling is now the field standard.** MeshSegNet (paper 023) uses 16k, TSegNet uses 16k, Cao 2025 (paper 026) inherits 16k via ToothInstanceNet, RHL uses 16k, TSegLab uses 16k. **The community has converged on 16k as the right resolution for the segmentation sub-task.** For our v0, this means we should pre-process our IOS data to 16k mesh cells before any deep model touches it — the per-cell 24-dim feature (12 coord + 12 normal) is the lingua franca of the field, and any v0 base model we adopt (ToothInstanceNet reimplementation, or a from-scratch PointNet++ variant) will expect this format.

6. **The "distance-aware" centroid voting trick is *the* paper's intellectual contribution.** Without it, the centroid voting stage fails on crowded teeth (votes from adjacent teeth merge into a single peak). With it, the inter-centroid distance prior (learned from training-set FDI-pair offset statistics) breaks the symmetry. **For our v0: if we reimplement the centroid-first 2-stage architecture, the *distance-aware weighting* is the single trick that determines whether the model handles crowded molars or not. Worth implementing carefully.** A simple Euclidean-distance-based prior (cluster centroids must be ≥ 5mm apart) is the right starting point; learnable inter-centroid offset distribution is the v1 refinement (à la Cao 2025's FDI-pair-offset multivariate-Gaussian).

7. **No reference to the 3DTeethSeg challenge or its evaluation protocol in TSegNet's paper.** TSegNet was published 6 months before the 3DTeethSeg challenge ran (Dec 2020 vs. MICCAI 2022), so the 3DTeethSeg Score 0.9734 in the table above is from a separate evaluation (Cao 2025 / Ben-Hamadou 2023) that *applied* TSegNet to the public 3DTeethSeg test set after the fact. **The fact that TSegNet holds up at 0.9734 on a public, completely-different-from-training-distribution dataset is a strong H5-style generalization signal** — and yet TSegNet still fails on partial-arch scans (Jana 2023), so the H5 signal is limited to the full-arch clinical case.

## Quote-worthy sentences

- *"Automatic and accurate segmentation of dental models is a fundamental task in computer-aided dentistry. Previous methods can achieve satisfactory segmentation results on normal dental models; however, they fail to robustly handle challenging clinical cases such as dental models with missing, crowding, or misaligned teeth before orthodontic treatments."* (Introduction) — the canonical statement of the problem, echoed by every 3D dental segmentation paper since.
- *"Our algorithm detects all the teeth using a distance-aware tooth centroid voting scheme in the first stage, which ensures the accurate localization of tooth objects even with irregular positions on abnormal dental models."* (Abstract) — the centroid-voting stage's value proposition in one sentence.
- *"Then, a confidence-aware cascade segmentation module in the second stage is designed to segment each individual tooth and resolve ambiguities caused by aforementioned challenging cases."* (Abstract) — the per-tooth cascade's role in disambiguating boundary cells.
- *"Extensive evaluations, ablation studies and comparisons demonstrate that our method can generate accurate tooth labels robustly in various challenging cases and significantly outperforms state-of-the-art approaches by 6.5% of Dice Coefficient, 3.0% of F1 score in term of accuracy, while achieving 20 times speedup of computational time."* (Abstract) — the headline 6.5%/3.0%/20× numbers, still the standard dental-segmentation speedup claim.
- *"The use of the tooth centroids may not make TSegNet and DArch fully effective for partial scan segmentation of partial tooth segmentation."* (Jana 2023, Related Works) — the canonical limitation, in one sentence.

## Code / data

- **DOI:** https://doi.org/10.1016/j.media.2020.101949
- **PubMed:** https://pubmed.ncbi.nlm.nih.gov/33387908/
- **Project page (preprint mirror):** https://enigma-li.github.io/projects/tsegNet/TSegNet.html
- **Code:** **No public release.** Project page says "Coming Soon..."; verified absent at github.com/enigma-li, github.com/PointCloudCAD, github.com/WenpingWang, and via Google Scholar's code-link search. (We rely on reimplementing from the paper for any v0 work.)
- **Data:** Private orthodontic dataset, not released.
- **Implementation dependencies (per the paper):** PyTorch 1.x, PointNet++ backbone, custom mesh-cell encoder. No public pre-trained weights.

## For our project — concrete next steps

1. **Adopt the 2-stage "centroid-first, per-tooth-second" architecture as the v0 sub-task 1 base pattern.** TSegNet's architecture is the dominant 3D dental segmentation paradigm, and reimplementing it is a ~2-week engineering task: (a) a PointNet++ backbone on 16k mesh cells producing per-cell FDI + centroid regression + confidence (3 heads, 1 backbone), (b) a centroid-voting clustering stage with distance-aware inter-centroid prior (≥ 5mm), (c) a per-tooth cascade segmentation head conditioned on the centroid embeddings, (d) the FDI-pair-offset multivariate-Gaussian postprocessor from Cao 2025 (paper 026) as the final stage. Total: ~1,000 lines of PyTorch, ~$200 of Lambda compute for training, expected Score on 3DTeethSeg test set: 0.95-0.98 (between TSegNet's 0.9734 and Cao 2025's 0.9870). **This is the single most important architecture decision for v0 sub-task 1.**

2. **Do NOT use TSegNet directly as a v0 base model.** Code is unreleased, weights are unavailable, the 0.97+ Score is on full-arch only, and the partial-arch clinical case (single-quadrant crown prep) is exactly where TSegNet fails. Reimplement from the paper description plus the Cao 2025 enhancements — this is what the 2026 literature has converged on.

3. **Use the 3DTeethSeg challenge Score 0.9734 as the v0 sub-task 1 *minimum acceptable* target.** Beating TSegNet's 0.9734 on the 3DTeethSeg 600-scan public test set is the *minimum* v0 sub-task 1 deliverable. The Cao 2025 bar of 0.9870 is the *stretch* v0 target. **Concrete action: report all four 3DTeethSeg challenge metrics (TLA, TSA, TIR, Score) on the public 600-scan test set, target Score ≥ 0.95 v0, ≥ 0.98 v1.**

4. **Adopt TSegNet's 20×-speedup early-stopping trick for free inference gains.** When the per-tooth confidence exceeds 0.95 in the cascade, stop refining. This is a 5-10× inference speedup with no accuracy loss (the cascade's job is precision; if confidence is already 0.95, additional refinement adds < 0.001 F1). **For clinical deployment (our v1 product), this is essential** — a dental office won't wait 30s for a segmentation.

5. **Combine TSegNet's 2-stage architecture with Cao 2025's three enhancements for v0 sub-task 1.** The full v0 sub-task 1 stack:
   - **Stage 1** (à la TSegNet): per-cell FDI + centroid regression on 16k mesh cells.
   - **Augmentation** (à la Cao 2025 enhancement 1): 90% probability artificial-partial-arch crops skewed toward 2-12 consecutive teeth.
   - **Alignment** (à la Cao 2025 enhancement 2, simplified): OBB + curvature-cue heuristic (not the full DL-based alignment — defer to v1).
   - **Stage 2** (à la TSegNet + Cao 2025 enhancement 3): per-tooth cascade segmentation with FDI-pair-offset multivariate-Gaussian postprocessor.
   - **Real partials** (à la Cao 2025 enhancement 4): add when we have a 50+ real partial-arch dataset (probably v1).
   - **Expected Score on 3DTeethSeg 600-scan test: 0.95-0.98** (between TSegNet's 0.9734 and Cao 2025's 0.9870).

6. **Adopt the 16k mesh-cell downsampling as the v0 standard.** Every paper in the 2020-2024 3D dental segmentation literature uses 16k mesh cells. The per-cell 24-dim feature (12 coord + 12 normal) is the lingua franca. **Concrete action: in v0 data preprocessing, take the raw 100k+ mesh-cell IOS scan, quadric-downsample to 16k cells, compute per-cell features as [3 vertices + barycenter; 4 normals] = 24-dim vector. This is the same preprocessing as paper 023 MeshSegNet, paper 027 TSegNet, and paper 026 Cao 2025 (via ToothInstanceNet).**

7. **Open question for HK: the centroid-first architecture's "distance-aware weighting" — implement it as a learnable inter-centroid offset distribution (à la Cao 2025's FDI-pair-offset Gaussian) or as a fixed ≥ 5mm Euclidean prior?** The fixed prior is simpler and matches TSegNet's approach; the learnable distribution adds ~50 lines of NumPy and ~0.005-0.01 macro-IoU. **Recommendation: start with the fixed ≥ 5mm prior, add the learnable Gaussian in v1 once we have a v0 baseline working.** The 5mm threshold should be a learned hyperparameter on the training set (just compute the 5th percentile of FDI-pair distances on the 3DTeethSeg 1,200 training scans).

8. **Open question for HK: should v0 sub-task 1 use a from-scratch PointNet++-on-mesh-cells implementation, or reimplement ToothInstanceNet (the same lab as Cao 2025)?** The from-scratch version is ~1,000 lines and matches the TSegNet + Cao 2025 architecture pattern; the ToothInstanceNet reimplementation is closer to the Cao 2025 paper but lacks the public code. **Recommendation: from-scratch TSegNet + Cao 2025 enhancement stack, ~1,000 lines, ~$200 Lambda compute, expected 0.95-0.98 Score on 3DTeethSeg public test set.**

9. **Speed target for v0 sub-task 1: < 5s per arch on Lambda A100.** TSegNet's 20× speedup over MeshSegNet means it runs in < 1s per arch on the right hardware. Our v0 sub-task 1 must match this for clinical usability. **Concrete action: include inference-time benchmarks in v0 eval, target < 5s per arch on a single A100 (matches the 3DTeethSeg challenge's ~1s/arch target on a high-end GPU).**

10. **Open question for HK: the segmentation sub-task's *output* — should it be per-cell FDI labels (the standard), per-vertex FDI labels (better for downstream mesh operations), or per-tooth mesh segments (best for crown-design sub-tasks 2-4)?** Per-cell FDI is the paper-026 standard; per-vertex FDI is a +1-line preprocessing change; per-tooth mesh segments require a connected-components postprocessor. **Recommendation: output per-vertex FDI labels (one-hot, 17 classes including gingiva), then post-process to per-tooth connected-component meshes with trimesh. This is the right format for downstream sub-tasks 2-4 to consume directly without re-processing.**

11. **Next paper to read: DTSegNet or RHL (the 2023 3DTeethSeg challenge winners that built on TSegNet's centroid-first architecture and added transformer-based refinements) for the post-TSegNet 2023-era improvements, OR Stratified Transformer (the alignment backbone used in Cao 2025 enhancement 2) for the architecture details of the DL-based alignment module that we'd add in v1.** The cleanest follow-up is DTSegNet (best 3DTeethSeg Score 0.9817, between TSegNet and Cao 2025, the right "intermediate" baseline) and RHL (Score 0.9845, the strongest pre-Cao 2025 method). Reading these two would close the loop on the 3DTeethSeg challenge leaderboard and let us triangulate which Cao 2025 enhancement is the highest-impact to reimplement in v0.

---

**Word count:** ~3,500
**Status:** Read 2026-06-07 02:08 KST. Hypothesis impact: **H1 strong support (foundational 2-stage "centroid-first" architecture now the dominant dental segmentation pattern), H3 strong support (earliest H3 mechanism in our reading list: distance-aware centroid voting + confidence-aware cascade), H4 strong support (mesh-cell features at 16k cells are the clinical-bar substrate, finer than 50μm margin tolerance), H5 mild contradiction (real-only training fails on partial-arch; Cao 2025's artificial-partial-augmentation is the H5 fix), H2 N/A**.
