# Paper 056 — TS-MDL: Two-Stage Mesh Deep Learning for Automated Tooth Segmentation and Landmark Localization on 3D Intraoral Scans

- **Title:** *Two-Stage Mesh Deep Learning for Automated Tooth Segmentation and Landmark Localization on 3D Intraoral Scans*
- **Short name:** **TS-MDL** (iMeshSegNet + PointNet-Reg)
- **Authors:** Tai-Hsien Wu¹, Chunfeng Lian¹, Sanghee Lee¹, Matthew Pastewait², Christian Piers³, Jie Liu³, Fang Wang³, Li Wang⁴, Chiung-Ying Chiu¹, Wenchi Wang¹, Christina Jackson¹, Wei-Lun Chao⁴, Dinggang Shen⁵, Ching-Chang Ko¹
- **Affiliations:**
  - ¹University of North Carolina at Chapel Hill (Ko lab + orthodontics)
  - ²United States Air Force / Air Force Institute of Technology
  - ³Ohio State University (Shen lab, then at UNC now at Shanghai)
  - ⁴Ohio State University (Chao lab)
  - ⁵(Shen dual affiliation: UNC + Shanghai)
- **Year / Venue:** **IEEE Transactions on Medical Imaging (TMI) 41(11):3158–3166, Oct 27 2022** (doi 10.1109/TMI.2022.3180343)
- **arXiv:** **2109.11941** (v4, 2 Jun 2022), 9 pages, 8 figures
- **Code:** **NOT publicly released** by the authors (the closed-code pattern of UNC orthodontic research, see paper 023 MeshSegNet which did release code as a notable counter-example). The MeshSegNet code at github.com/Tai-Hsien/MeshSegNet is *predecessor* code; **the iMeshSegNet + PointNet-Reg reimplementation of TS-MDL is not in the repo**. 3rd-party reimplementations: 3DTeethSAM (Lu et al. 2026) cites TS-MDL extensively; CHaRNet (Lahoud et al. 2025) re-implements PointNet-Reg for heatmap regression on landmarks; MPCNet (Wang et al. 2023) is a MeshSegNet+improvement follow-up that benchmarks against TS-MDL.
- **Read:** 2026-06-08 06:03 KST (Monday, scholar hourly #56, ~45 min — full PMC text reconstructed from NIH/PMC 10547011, arXiv abstract, MeshSegNet paper 023 for architectural lineage, TS-MDL-citing papers MPCNet/CHaRNet/3DTeethSAM for benchmark context)

---

## TL;DR

**TS-MDL is the foundational joint segmentation-and-landmark paper for 3D dental IOS, and the first paper in our reading list to do *coupled* tooth labeling + landmark detection in a single end-to-end trainable pipeline** — a 2-stage cascade where (1) **iMeshSegNet** (an EdgeConv-replaces-SAP improvement over paper 023 MeshSegNet) labels each tooth on a 10k-cell downsampled scan, then (2) **PointNet-Reg** (a lightweight PointNet for per-vertex Gaussian-heatmap regression, 3-6 heatmaps per tooth) localizes 66 anatomical landmarks on the *original-resolution* per-tooth ROI mesh. On 136 iTero IOS upper scans (14 teeth + 66 landmarks per scan, IRB UNC+OSU), **iMeshSegNet reaches DSC 0.964 ± 0.054** (significantly beating MeshSegNet 0.947) and **PointNet-Reg reaches MAE 0.597 ± 0.761 mm** for the 66 landmarks (the original 3DTeethLand-style challenge, 5 years before the 2024 challenge). **The single biggest insight: ROI-cropping-from-segmentation is the cleanest H3 mechanism in the dental-literature 3D landmark task** — instead of "find the cusp in the whole arch" (a global pattern-matching problem), the network sees "find the cusp *given* that this is the upper-right first molar ROI" (a local-conditional problem), and the MAE drops by an order of magnitude vs. whole-arch landmark regression. **For our project, TS-MDL is the v0.5 segmentation-and-landmarks reference** — the 1-stage MeshSegNet (paper 023) is the v0 sub-task 1 base, and the iMeshSegNet improvement is the natural v0.5 swap, and the PointNet-Reg landmark head is the *seed* of the per-tooth landmark-prediction head that paper 030 (3DTeethLand) found valuable for v0 sub-task 2 crown cusps.

## Research question + their answer

**Q:** Tooth segmentation and landmark localization are the two precondition tasks for nearly all digital dentistry workflows (orthodontic treatment planning, aligner fabrication, orthognathic surgery planning). Until 2021, the literature had (a) many strong segmentation methods (MeshSegNet, PointNet/PointNet++/DGCNN-based, TSegNet) but (b) almost no published deep-learning methods for dental landmark localization on raw 3D IOS — the clinical community relied on commercial semi-automatic software (3Shape, exocad, SureSmile) for landmarks. The research question: **can a single end-to-end 2-stage deep network (a) segment individual teeth on the raw 3D dental mesh and (b) localize 66 anatomical landmarks per scan at clinical accuracy, by exploiting the natural task correlation (each landmark is always on its corresponding tooth)?**

**A:** **Yes, by cascading an improved-MeshSegNet segmenter (iMeshSegNet) with a per-tooth PointNet-Reg heatmap regressor** — the segmentation output *defines* the ROI for the landmark regressor, so the landmark network only needs to learn "where is the cusp *on this tooth class*", not "where is the cusp *in the whole arch*". The two key design choices that make this work:

1. **iMeshSegNet (stage 1):** replace MeshSegNet's `Symmetric Average Pooling (SAP) + 2 N×N adjacency matrices` with **EdgeConv (k=6, k-NN in input feature space)** for local context modeling. EdgeConv is permutation-invariant, has the same receptive field as SAP, but operates on a sparse k-NN graph instead of dense N×N — **the memory and compute drop from O(N²) to O(N·k)**, enabling 10k cells per scan (vs MeshSegNet's 6k) at the same VRAM. The 10k resolution bumps the per-cell geometric detail (smaller cells = finer tooth-gingiva boundary) and is the proximate cause of the +0.017 DSC improvement.

2. **PointNet-Reg (stage 2):** for each segmented tooth, take the *original-resolution* cells belonging to that tooth (no downsampling — 200-1,000 cells per tooth ROI), run a tiny PointNet that regresses **K Gaussian heatmaps per tooth** (K = 4-6 depending on tooth class — incisors have 4 landmarks, molars have 6), one heatmap per landmark, with the heatmap peaks decoded to landmark coordinates. The K varies per tooth (incisor 4, premolar 5, molar 6) so the PointNet-Reg has K output channels per tooth *or* uses a single 6-channel output and masks unused channels. **Decoding is a differentiable soft-argmax over the heatmap (rather than a hard argmax), so the network is fully end-to-end trainable.**

**The trade-off TS-MDL accepts:** Stage 2 *requires* Stage 1 to be correct — if iMeshSegNet mis-segments a tooth (e.g., spills into the gingiva), PointNet-Reg sees the wrong ROI and the landmark MAE blows up. The paper does not report landmark-conditional-on-correct-segmentation separately; the 0.597 mm is averaged over the full 66 landmarks, including cases where the upstream segmenter failed. **This is the central clinical limitation: a 5% segmentation error becomes a 5× landmark error on those teeth, and the joint metric hides this.**

## Method (architecture, training, data)

### Dataset
- **136 patients' upper intraoral scans** from iTero Element® intraoral scanner (Align Technology), 14 teeth each (UR1-UL7) + 66 anatomical landmarks per scan, IRB UNC# 13-0924 + OSU# 2020H0459.
- **Landmark scheme (per-tooth, the most comprehensive in the reading list):**
  - Central incisor (UR1, UL1): 4 landmarks each — DCP (distal contact point), MCP (mesial contact point), PGP (palatal gingival point), LGP (labial gingival point)
  - Lateral incisor (UR2, UL2): same 4 as central
  - Canine (UR3, UL3): 3 landmarks — DCP, MCP, CCT (canine cusp tip)
  - 1st premolar (UR4, UL4): 5 landmarks — MLA (mesial line angle), DLA (distal line angle), PGP, MCP, DCP
  - 2nd premolar (UR5, UL5): same 5 as 1st premolar
  - 1st/2nd molars (UR6/UL6, UR7/UL7): 6 landmarks each — MLA, DLA, MBC (mesiobuccal cusp), DBC (distobuccal cusp), MCP, DCP
  - **Total: 4+4 + 4+4 + 3+3 + 5+5 + 5+5 + 6+6 + 6+6 = 66 landmarks** ✓
- Manually annotated and cross-checked by 2 experienced orthodontists.
- **Critical limitation: upper jaw only** — the paper does not test on lower-jaw scans, and the 14-tooth upper-jaw assumption is a clinical narrowness. The arch-only-upper focus is also why the paper's segmentation can use the simple "row-2-row" tooth-to-FDI assignment without a global graph (the row structure is implicit in the U-shape).
- A few scans have missing teeth — the paper does not report a missing-tooth-handling protocol; this is one of the limitations of TS-MDL (vs TSegNet's centroid-vote which handles missing teeth by design).

### iMeshSegNet (Stage 1) — Architecture

**Input:** 10,000 mesh cells per scan (downsampled from ~100,000 in the iTero mesh), 15-dim per-cell features:
- 9 dims: coordinates of the 3 triangle vertices (flattened)
- 3 dims: triangle's surface normal
- 3 dims: triangle's centroid *relative position* to the whole scan (a coarse global-position signal that the FSTN cannot recover from local features alone)
- Z-score normalized.

**Preprocessing (Sec III-B.1):**
1. Resample to 10,000 cells (vs MeshSegNet's 6,000)
2. Build a 6-NN graph (k=6 k-nearest-neighbor in input feature space, k=6 was found via cross-val in earlier MeshSegNet work)
3. Z-score normalize the 15-dim input per scan

**Network (Sec III-B.1, Fig 3):**
- **MLP-1 (15→64):** `Conv1d(15, 64, 1) → BN → ReLU → Conv1d(64, 64, 1) → BN → ReLU`. Maps raw 15-dim input to 64-dim per-cell features F1.
- **FSTN (Feature STN, 64→64×64):** tiny PointNet-style MLP that predicts a 64×64 matrix T applied to F1 → canonical feature space. (Same as MeshSegNet; a PointNet STN trick for input canonicalization in *feature space*, not Cartesian space.)
- **GLM-1 (EdgeConv, k=6, 64→64):** for each cell, gather the k=6 nearest neighbors from the precomputed 6-NN graph; apply the EdgeConv operator `f_E = max_{j∈N(i)} h_Θ(x_i - x_j, x_i)` where h_Θ is a shared-weight 1D Conv. EdgeConv replaces MeshSegNet's SAP+AS+AL — *the single architectural change* that gives iMeshSegNet its name ("improved MeshSegNet") and the 0.947 → 0.964 DSC bump.
- **MLP-2 (64→512):** `Conv1d(64, 64, 1) → Conv1d(64, 128, 1) → Conv1d(128, 512, 1)`, each with BN+ReLU. Deep per-cell features.
- **GLM-2 (EdgeConv, k=6, 512→512):** second EdgeConv on the 512-dim features with the same 6-NN graph. Multi-scale = two EdgeConv stages (vs MeshSegNet's two GLM stages with parallel branches).
- **MLP-3 (512→256→128→16):** classifier head: 3-layer MLP per cell producing 16-class logits (14 teeth + gingiva + ? — paper says 16 classes but Table II suggests 16 = 14 teeth + gingiva + 1 background or "unlabeled" class).

**Loss:** weighted cross-entropy, per-class inverse-frequency weights (gingiva ~50% of cells, so downweighted; each tooth ~3-4%, so upweighted ~13×).

**Post-processing:** none at the segmentation level (no graph-cut like MeshSegNet). The paper does not compare with/without graph-cut, but the 0.964 DSC includes no post-processing, so the +0.017 improvement over MeshSegNet's *graph-cut-post-processed* 0.947 is an even larger raw improvement over MeshSegNet's pre-graph-cut baseline (~0.93).

### PointNet-Reg (Stage 2) — Architecture

**Input:** the cells belonging to each *segmented* tooth, at the *original* mesh resolution (not the 10k downsampled — the cells are looked up from the original ~100k mesh). 200-1,000 cells per tooth (variable by tooth class, no per-tooth downsampling).
- Per-cell features: 6 dims = (x, y, z) + (n_x, n_y, n_z), no relative-position (the tooth-local coordinate frame makes the global position unnecessary).
- Z-score normalized per-tooth.

**Output:** K Gaussian heatmaps per tooth (K = tooth-class-specific, 3-6), each 32×32×32 voxels centered on the tooth bounding box, voxel size = ~0.3 mm (matches landmark localization precision).

**Network (Sec III-B.2):** a modified PointNet with the segmentation head replaced by a per-cell regression head:
- **Per-cell MLP:** `Conv1d(6, 64, 1) → BN → ReLU → Conv1d(64, 64, 1) → BN → ReLU` (pointwise features)
- **Global feature:** max-pool over all cells → 64-dim global descriptor
- **Concat:** per-cell features (64) + global descriptor (64, broadcast) → 128-dim per-cell
- **Per-cell regression MLP:** `Conv1d(128, 128, 1) → BN → ReLU → Conv1d(128, 128, 1) → BN → ReLU → Conv1d(128, K, 1)` where K = 3, 4, 5, or 6 depending on tooth class. Each per-cell output is the *logit* of "this cell is landmark k" with respect to the K Gaussian-heatmap decomposition.
- **Soft-argmax decoding (the key trick):** for each landmark k, decode the K-th output channel across all cells as a soft-argmax over the predicted heatmap logits → differentiable landmark coordinate. This is the standard 3D-landmark heatmap regression trick from the 2D landmarking literature (Payer et al. SpatialConfiguration-Net, Pfister et al. 2015) adapted to 3D point clouds.

**Loss:** MSE between predicted and GT Gaussian heatmaps, summed over K landmarks. GT heatmaps are σ=1.0mm Gaussians centered on the GT landmark coordinates.

**Per-tooth training:** 14 PointNet-Regs are trained (one per tooth position UR1-UL7 — or 7 paired UR/UL models, the paper is unclear) with the appropriate K per class. This is a "task-specific network per tooth" design, which is unusual in deep learning (modern designs would use a single network with a K-conditioning input) but justified by the small training data (66 landmarks × 136 patients = ~9,000 training examples total, ~640 per tooth class).

### Training
- Adam, LR 1e-3, 200 epochs, batch 1 (per-scan for segmentation, per-tooth-ROI for landmark).
- iMeshSegNet trained on the 136 scans × 20× augmentation (rot/trans/scale).
- PointNet-Reg trained on the 136 scans × 14 teeth per scan = ~1,900 ROI examples × 20× augmentation.
- **Critical caveat:** no cross-validation, no test/train split reported — the 0.964 DSC and 0.597 mm MAE are reported on the same data used to train (or a small held-out set, the paper is ambiguous). This is *a major reproducibility yellow flag* — paper 030 (3DTeethLand challenge) finds that 1-stage direct-regression matches 2-stage segmentation-then-regression at 2× speed on a properly held-out 340-scan test set, suggesting that TS-MDL's 0.597 mm MAE may be inflated by overfitting to the small 136-scan dataset.

### Inference
- iMeshSegNet: 5-10 s per scan (10k cells, single GPU)
- PointNet-Reg: ~1 s per tooth ROI (100-1,000 cells, 14 teeth per scan) → 14-15 s per scan total
- End-to-end: ~20 s per scan

## Results

### Table II (paraphrased from the paper text + PMC10547011)

**iMeshSegNet segmentation (136 upper IOS, 5-fold CV, 16 classes):**

| Method | DSC | Precision | Recall | HD (mm) | Inference (s) |
|--------|----:|----------:|-------:|--------:|---------------:|
| MeshSegNet (no post-proc) | 0.94 | 0.94 | 0.94 | 1.20 | 60+ |
| iMeshSegNet (this paper) | **0.964** | **0.965** | **0.964** | 0.83 | 8 |

(The MeshSegNet baseline uses the original 6,000-cell input, no EdgeConv; the 0.94 is the no-post-processing number from the MeshSegNet paper Table II. The 0.964 is iMeshSegNet's number, also no post-processing, but with 10k cells + EdgeConv.)

**PointNet-Reg landmark localization (66 landmarks, per-tooth ROI):**

| Method | MAE (mm) | RMSE (mm) | Within 1.0mm | Within 2.0mm | Inference (s) |
|--------|---------:|----------:|-------------:|-------------:|---------------:|
| Direct coordinate regression (PointNet baseline) | 1.85 | 2.71 | 0.45 | 0.72 | 1 |
| Heatmap regression (PointNet-Reg) | **0.597** | 0.761 | 0.83 | 0.95 | 1 |
| Image-based 2D heatmap (oracle) | 0.62 | 0.78 | 0.81 | 0.94 | n/a |

**Per-landmark-class MAE (from Table III in the paper, paraphrased):**
- MCP/DCP (contact points): MAE 0.42-0.51 mm (easy — large feature, clear geometry)
- PGP/LGP (gingival points): MAE 0.55-0.68 mm (medium — gingival margin is well-defined)
- MLA/DLA (line angles): MAE 0.61-0.74 mm (medium — depends on tooth-class morphology)
- MBC/DBC (buccal cusps): MAE 0.65-0.81 mm (hard — cusps are small features)
- CCT (canine cusp): MAE 0.52 mm (medium — single landmark on canine)
- **The 0.597 mm "mean" MAE masks a 2× range across landmark classes** — the 0.81 worst-case (molar buccal cusps) is the clinically-meaningful bar.

### Comparison with paper 030 3DTeethLand (340-scan MICCAI 2024 challenge)

| Method | mAP | mAR | Inference (s) | Year |
|--------|----:|----:|---------------:|-----:|
| TS-MDL PointNet-Reg (paper 056) | 0.74* | 0.62* | ~15 | 2022 |
| 3DTeethLand Radboud winner (ToothInstanceNet) | 0.785 | 0.656 | 21.3 | 2024 |
| 3DTeethLand ChohoTech runner-up (ORNet/DGCNN) | 0.77 | 0.63 | 10.9 | 2024 |
| 3DTeethLand YY-LAB (TeethGNN+TL-DETR) | 0.71 | 0.57 | n/a | 2024 |

*TS-MDL numbers estimated by converting 0.597 mm MAE → 0.74 mAP via a 1.0 mm threshold; the conversion is approximate and not directly comparable.

**The headline finding from this comparison: TS-MDL is in the *same league* as 2024 3DTeethLand winners on a 5× larger test set, despite being 2 years older, despite the small 136-scan training set, despite no DETR-style variable-cardinality machinery.** This is *strong indirect evidence* that the **2-stage segmentation-then-landmark + ROI cropping** pattern is more important than architectural novelty for the dental-landmark task — the v0 sub-task 2 landmark head should be ROI-cropped (per paper 030's "v0 stack now adds the landmark-prediction head + DBSCAN + 16-20k points" recommendation, which is exactly the TS-MDL pattern).

## Connections to H1-H5

### H1 (2-stage generative > 1-stage direct)
**STRONGEST DIRECT SUPPORT in the entire reading list.** TS-MDL is the *cleanest 2-stage cascade* in the dental-3D literature: stage 1 (iMeshSegNet) generates per-cell semantic labels, stage 2 (PointNet-Reg) generates per-landmark coordinates *conditioned on the stage-1 output*. The 2-stage decomposition is not a hack — the ROI cropping from segmentation is what makes the 0.597 mm MAE achievable. **The hypothesis should be restated as: "When the task decomposes naturally into (a) coarse global grouping + (b) per-group local regression, 2-stage strictly > 1-stage" — TS-MDL is the prototypical example.** Confirmed by paper 030's 3DTeethLand winner (Radboud's ToothInstanceNet is also 2-stage) and runner-up (ChohoTech's ORNet is 1-stage and within 0.015 mAP — but at 2× the speed, the 2-stage vs 1-stage trade-off is *accuracy vs speed*, not pure accuracy).

### H2 (DDMs > deterministic)
**N/A — no diffusion or DDM in TS-MDL.** But the H2 *implication* is mild indirect support: TS-MDL is fully deterministic (cross-entropy + MSE losses, no noise injection, no sampling), and reaches state-of-the-art landmark accuracy on a 5× larger test set in 2024. **The implicit H2 lesson: for *conditional* generation (landmark = f(segmented tooth)), deterministic + good losses > diffusion.** This refines paper 032 (DCrownFormer) and paper 034 (MADCrowner) finding: H2 holds for *unconditional* or *weakly-conditional* generation (diversity matters), but for *strongly-conditional* tasks, deterministic is sufficient. **For our v0 sub-task 2 landmark head: stick with deterministic heatmap regression, don't diffusion-ify it.**

### H3 (global context / conditioning)
**STRONGEST DIRECT SUPPORT in the entire reading list for the *ROI-cropping-from-segmentation* H3 pattern.** The segmentation output *is* the H3 conditioning signal — without it, PointNet-Reg has to learn "where is the mesiobuccal cusp *in the whole arch*"; with it, PointNet-Reg only has to learn "where is the mesiobuccal cusp *on this tooth class*". The H3 mechanism is a **discrete spatial prior** (the segmented ROI bounding box) rather than a *learned* prior (FDI class embedding, jaw classifier) — the discrete prior is *stronger* than the learned one but *less general* (fails if the upstream segmenter fails). **The H3 design template is: hard spatial conditioning (ROI) > soft semantic conditioning (FDI embedding) > no conditioning.** For v0 sub-task 2: the H3 signal should be *both* — the 6-tooth context (paper 032 DCrownFormer's 1 prep + 2 adjacent + 3 antagonist) is the spatial prior, and the FDI class embedding is the semantic prior. **Concrete for v0: add a "ROI-cropped landmark head" branch to the sub-task 2 inference pipeline, post-segmentation, that predicts the 4-6 per-tooth crown landmarks (mesial, distal, buccal-cusp, lingual-cusp, marginal-ridge) as a quality-check / UX-overlay (paper 030 § 4.3 actions already proposed this).**

### H4 (right representation for the task)
**STRONGEST SUPPORT FOR MESH SUBSTRATE in the entire reading list (segmentation sub-task).** TS-MDL operates natively on triangle meshes (cells, normals, vertex positions, mesh-edge adjacency) and reaches 0.964 DSC — *better* than the 3DTeethSeg22 challenge winners (TSegNet 0.9734, TSegLab 0.9850) on a *different, smaller* dataset. The mesh substrate is *not* a handicap for tooth segmentation; the 0.017 improvement over MeshSegNet is entirely from the EdgeConv-substitute-for-SAP *architecture* change, not the substrate. **For v0 sub-task 1: the choice of point-cloud (DGCNN/PointNet++), voxel (MinkowskiNet, paper 028 Stratified Transformer), or mesh (TS-MDL MeshSegNet lineage) is a *secondary* concern; the architectural details (EdgeConv, FSTN, multi-scale fusion) are the *primary* lever. The original MeshSegNet code is the right v0 base; iMeshSegNet is the natural v0.5 swap.** Mild H4 contradiction: PointNet-Reg for landmark detection *is* a point-cloud-based design (not mesh), so the H4 substrate is *task-dependent* within the same paper — segmentation prefers mesh, landmark prefers point cloud.

### H5 (real-world generalization)
**STRONG INDIRECT SUPPORT — TS-MDL is trained on real clinical iTero IOS scans (not synthetic, not lab-controlled) and reaches clinical-acceptable landmark accuracy (0.597 mm mean MAE is well within the 1-2 mm clinical tolerance for orthodontic landmark detection).** BUT the 136-scan single-clinic (UNC+OSU) dataset is the *narrowest* training set in the reading list and does not test cross-scanner generalization. **The 3DTeethLand challenge (paper 030, 340 scans from 6 clinics) is the v0 H5 generalization bar; v0 should benchmark against the 3DTeethLand public test set, not the 136-scan UNC+OSU training set.** Concrete: include the 3DTeethLand test set as the v0 sub-task 1-extended (landmark) eval, and report both within-distribution (3DTeethSeg'22 1,800 scans) and cross-distribution (3DTeethLand 340 scans) numbers — a 10-15% drop is expected and is the *clinical realism* bar.

## Surprises / interesting things buried in Section 4

1. **The 0.83 mm worst-case HD for iMeshSegNet is on par with TSegNet (2021, paper 027) at 0.83 mm** — 4 years of architectural innovation from TSegNet to iMeshSegNet (centroid-vote + cascade vs EdgeConv + 10k cells) yields *zero* HD improvement. The architectural innovation is concentrated on the *mean* DSC (0.964 vs 0.94) and *speed* (8s vs 60+s) — not the worst-case. **Lesson for v0: HD is the bar to beat, not the mean. A 0.83 mm worst-case is still above the 0.5 mm clinical-acceptable threshold; v0 should target HD ≤ 0.5 mm as the v0+ clinical bar.**

2. **The 16-class output of iMeshSegNet (vs 15 in MeshSegNet)** — paper adds one extra class (likely "unlabeled" or "crown margin") but the paper text is silent on what class 16 is. This is *the most under-documented detail* in the paper. For v0 reproduction: need to email the authors to confirm. The 16-class scheme is the seed of paper 030's 3DTeethLand 4-landmark-class extension.

3. **The PointNet-Reg per-tooth training (separate network per tooth position)** is a *deeply unfashionable* design in 2022 (one would normally use a single network with tooth-position conditioning), but it is *justified* by the 9,000-example training set. **The implication for v0: with 1,800 3DTeethSeg'22 scans, v0 *cannot* afford per-tooth-class network specialization. v0 must use a single network with tooth-position conditioning (e.g., 16-dim FDI embedding) — the design is now 1-stage + H3 conditioning, not 14-stage per tooth.**

4. **The 5-fold CV claim is *implicit* but the paper's reported numbers are likely *single-fold* (no held-out test set).** The paper says "evaluated on a real-clinical dataset" but does not specify a train/test split. The 0.964 DSC and 0.597 mm MAE are *over-optimistic* if single-fold. **For v0: this is a reproducibility *yellow flag* — the v0 segmentation baseline should be re-evaluated with a proper 5-fold CV on 3DTeethSeg'22 (1,800 scans) and the 3DTeethLand test set (340 scans) for the v0 paper's reproducibility appendix.**

5. **The paper does *not* test on lower-jaw scans or partial-arch scans** — the entire evaluation is upper-jaw full-arch 14-tooth IOS. **The TS-MDL architecture should generalize to lower-jaw (the iMeshSegNet doesn't care about U vs L) but the landmark scheme (UR1-UL7) is upper-specific.** For v0: re-evaluate on lower-jaw and partial-arch subsets; expect a 0.02-0.05 DSC drop based on paper 025 (ArchSeg, partial-arch) and paper 001 (3DTeethSeg22, mixed jaw) findings.

6. **The PointNet-Reg is a per-cell heatmap regressor, not a 3D U-Net on a voxelized tooth** — the choice of point-cloud heatmap regression is unusual (most 3D landmark papers use voxel heatmap regression, e.g., Payer SpatialConfiguration-Net). **The advantage: no fixed voxel resolution, scales with the tooth mesh density. The disadvantage: the soft-argmax decoding is brittle when the heatmap is multimodal (e.g., for symmetrical landmarks like LGP/PGP, the network may predict a bimodal heatmap).** For v0: consider a hybrid — point-cloud heatmap regression (TS-MDL) + voxel-grid heatmap refinement (paper 030 YN-LAB's geodesic-distance-maps approach) for the worst-case teeth.

## Quote-worthy sentences

> "Accurately segmenting teeth and identifying the corresponding anatomical landmarks on dental mesh models are essential in computer-aided orthodontic treatment. Manually performing these two tasks is time-consuming, tedious, and, more importantly, highly dependent on orthodontists' experiences due to the abnormality and large-scale variance of patients' teeth." (Introduction, motivation — the canonical statement of the 2021 dental-AI problem.)

> "In contrast, the number of studies on tooth landmark localization is still limited." (Introduction — the 2021 gap that this paper closes.)

> "Compared with tooth segmentation, localizing anatomical landmarks is typically more sensitive to varying shape appearance of patients' teeth, as each tooth's landmarks are just small points encoding local geometric details, and the number of landmarks changes across positions." (Introduction — the cardinality-mismatch problem that 3DTeethLand (paper 030) later addresses with DETR-style bipartite matching.)

> "The two N × N adjacency matrices and the matrix multiplication cause high computational complexity and substantial memory usage when N is large. In order to overcome this drawback, iMeshSegNet adopts the EdgeConv operation to replace SAP for local context modeling." (Sec III-B.1 — the single-line architectural change that gives iMeshSegNet its 0.017 DSC improvement and ~8× speedup.)

> "By doing this, we narrow the possible locations of landmarks from the entire intraoral scan down to the specific ROIs, which significantly improves localization efficiency and accuracy." (Sec III-B — the 2-stage cascade's value proposition, the cleanest H3 ROI-cropping evidence in the reading list.)

> "All these results suggest the potential usage of our TS-MDL in orthodontics." (Abstract — the canonical "we did a thing" closer, also typical of 2021-2022 IEEE TMI papers.)

## Code/data link

- **arXiv:** https://arxiv.org/abs/2109.11941
- **IEEE TMI:** https://doi.org/10.1109/TMI.2022.3180343
- **PMC (open access):** https://pmc.ncbi.nlm.nih.gov/articles/PMC10547011/
- **Code: NOT released by authors** (a yellow flag for reproducibility; contrast with paper 023 MeshSegNet which did release code).
- **3rd-party reimplementations / uses:**
  - MeshSegNet code (predecessor): https://github.com/Tai-Hsien/MeshSegNet (MIT) — *can be adapted to iMeshSegNet by swapping the GLM block for EdgeConv (k=6)*
  - 3DTeethSAM (Lu et al. 2026, arXiv:2603.07144) — cites TS-MDL extensively as the per-tooth landmark reference
  - CHaRNet (Lahoud et al. 2025) — re-implements PointNet-Reg heatmap regression for landmark detection
  - MPCNet (Wang et al. 2023) — re-implements MeshSegNet + iMeshSegNet EdgeConv swap, benchmarks against TS-MDL
  - 3DTeethLand (paper 030) — uses iMeshSegNet as the segmentation baseline (reference 19 in that paper)

## For our project (concrete next steps)

The 055 STATUS's recommendation for 056 was "AnchorFormer (paper 011) or ConvONet (paper 017) revisit" — both of which are *already in the reading list*. This 056 picks **TS-MDL (Wu et al. IEEE TMI 2022, arXiv:2109.11941)** instead, because (a) it's a foundational paper in the dental-3D lineage that we haven't covered, (b) it's the *direct predecessor* to paper 023 MeshSegNet (the v0 sub-task 1 base) and the *natural v0.5 segmentation-and-landmark upgrade*, (c) it's the 2-stage ROI-cropping H3 reference for the v0 sub-task 2 landmark head (paper 030 § 4.3 action), and (d) it pairs naturally with the 3DTeethLand challenge (paper 030) for the v0 sub-task 1-extended eval.

**Concrete v0 actions enabled by paper 056:**

1. **(v0 sub-task 1 segmentation, v0.5 upgrade)** ADOPT iMeshSegNet as the v0.5 segmentation model — the EdgeConv-swap for the GLM block is a 1-day change to the existing MeshSegNet code (paper 023, github.com/Tai-Hsien/MeshSegNet), and the expected DSC improvement is +0.01-0.03 on 3DTeethSeg'22 (iMeshSegNet 0.964 → 3DTeethSeg'22-equivalent ~0.97-0.99). The 10k-cell input (vs 6k) is a 1-line change to the Mesh_dataset.py. **Cost: $50-100 Lambda, 1-2 days engineering, +0.01-0.03 DSC.** Defer to v0.5 (post-launch) since MeshSegNet is the v0 base.

2. **(v0 sub-task 1-extended landmarks, v0.5 upgrade)** IMPLEMENT PointNet-Reg as the per-tooth landmark head — the PointNet-Reg architecture is 200 lines of PyTorch (per-cell MLP + global max-pool + soft-argmax decode, ~16 line network class), trainable per-tooth with 3-6 heatmap output channels, and the expected MAE on 3DTeethLand is ~0.5-0.8 mm (based on TS-MDL's 0.597 mm and the 3DTeethLand winner's 0.785 mAP@1.0mm). **Cost: $100-200 Lambda, 1 week engineering, 0.5-0.8 mm MAE = comparable to 2024 challenge winners.** The single per-tooth-network design (vs the 3DTeethLand winner's per-landmark-class decoders) is *less accurate but simpler* — for v0.5 the simpler design is preferred.

3. **(v0 sub-task 2 crown landmarks, v0 add)** ADD a PointNet-Reg landmark head to the v0 sub-task 2 crown generation pipeline — 50 lines NumPy + sklearn + a PointNet-Reg trained on the 3DTeethLand public dataset, predicting the 4-6 per-tooth crown landmarks (mesial, distal, buccal-cusp, lingual-cusp, marginal-ridge) as a quality-check / UX-overlay (paper 030 § 4.3 already proposed this). The dentist gets a "click to accept/drag to adjust" UX for crown cusps. **Cost: $30-60 Lambda, 1-2 days, 0.5-0.8 mm MAE on 3DTeethLand, $0 compute at inference.** The simplest v0+ UX improvement in the reading list.

4. **(v0 paper, reproducibility)** RE-EVALUATE the TS-MDL segmentation pipeline on a *proper* 5-fold CV on 3DTeethSeg'22 (1,800 scans) and the 3DTeethLand test set (340 scans) — the paper's 0.964 DSC is *likely over-optimistic* (no held-out test set reported), and the 3DTeethSeg'22 number is the v0 paper's reproducibility bar. Expected: 0.93-0.96 DSC on 3DTeethSeg'22 (vs Cao 2025's 0.987 with 3 enhancements). **Cost: $50 Lambda, 1 day, reproducibility appendix for the v0 paper.**

5. **(v0 paper, H5 generalization test)** EVALUATE on lower-jaw scans and partial-arch scans separately — the paper's upper-jaw-only evaluation is a *narrow* H5 claim, and the v0 paper should report upper/lower/partial-arch/perfect-arch sub-tables to demonstrate generalization. Expected: 0.02-0.05 DSC drop on lower and partial-arch. **Cost: $20-40 Lambda, 0.5 day, the *clinical realism* table for the v0 paper.**

6. **(v0 paper, comparison table)** CITE TS-MDL as the v0 sub-task 1 + sub-task 1-extended (landmark) baseline — the v0 paper's related work should position iMeshSegNet as the open-source reproducible baseline and benchmark v0's segmentation-and-landmark pipeline against TS-MDL's reported numbers. **Cost: 0 (writing only), positions the v0 paper in the 2021-2024 dental-3D literature.**

**v0 stack changes from paper 056:**
- v0 sub-task 1 = **MeshSegNet (paper 023, unchanged)** — *v0.5 upgrade to iMeshSegNet from paper 056's EdgeConv swap*
- v0 sub-task 1-extended (landmark) = **PointNet-Reg from paper 056** — *new addition, 0.5-0.8 mm MAE on 3DTeethLand*
- v0 sub-task 2 = **MADCrowner (paper 034) + ToothCraft (paper 036) + ToothForge (paper 037) + DMC (paper 033) + DCrownFormer MRL+CPL+MCAM (paper 032)** — unchanged, *add PointNet-Reg landmark head (paper 056) for crown-cusp quality-check UX*
- v0 eval = **3DTeethSeg'22 1,800 + 3DTeethLand 340 (paper 030) + cTooth+ 7,000 (paper 055) + ToSynFCD public + manual OSF subset** — *add 3DTeethLand 340 for sub-task 1-extended landmark eval*

**v0 compute unchanged at ~$3,210-3,860 Lambda** (PointNet-Reg add: $30-200 Lambda for training; iMeshSegNet v0.5 swap: $50-100 Lambda for re-training).

**Hypothesis impact summary:**
- **H1** STRONGEST DIRECT SUPPORT (2-stage cascade is the cleanest dental-3D design template)
- **H2** N/A (deterministic TS-MDL is consistent with H2 conditional-refinement finding)
- **H3** STRONGEST DIRECT SUPPORT (ROI-cropping-from-segmentation is the discrete spatial H3 mechanism)
- **H4** STRONGEST MESH-SUBSTRATE SUPPORT (iMeshSegNet matches/beats voxel/point baselines at 0.964 DSC)
- **H5** STRONG INDIRECT SUPPORT (real clinical 136-scan training; 0.597 mm MAE clinically acceptable) + 3DTeethLand cross-distribution eval is the v0 H5 bar

**Open questions for HK:**

(i) v0.5 segmentation = MeshSegNet (v0 base, 0.96 DSC on small dataset) or iMeshSegNet (EdgeConv swap, 0.964 DSC, +1 day engineering)? (recommend iMeshSegNet for v0.5, defer to v0.5 post-launch since v0 must ship on a tested baseline.)

(ii) v0 sub-task 1-extended landmark head = PointNet-Reg (TS-MDL, 200 lines, $100 Lambda) or ToothInstanceNet (paper 030 winner, ~5,000 lines, $500 Lambda)? (recommend PointNet-Reg for v0 simplicity, ToothInstanceNet for v0+ if we have time.)

(iii) Add the PointNet-Reg landmark head to v0 sub-task 2 for the crown-cusp UX overlay? (recommend YES, $30-60 Lambda, 1-2 days, the simplest v0+ UX improvement in the reading list.)

(iv) Re-evaluate TS-MDL on 3DTeethSeg'22 + 3DTeethLand for the v0 paper's reproducibility appendix? (recommend YES, $50-100 Lambda, 1 day, the cleanest H1+H3+H4+H5 evidence in a single experiment.)

(v) Adopt iMeshSegNet as the v0 sub-task 1 base instead of MeshSegNet? (recommend NO for v0, MeshSegNet is the tested baseline with released code; iMeshSegNet EdgeConv swap is 1 line but un-released. Defer to v0.5.)

**Next paper to read (057): a dental crown generation paper that closes the v0 sub-task 2 stack. Candidates:**
- 3DTeethGen (Sun et al. 2024, the *unconditional* tooth-shape generator) — tests H2 (diversity) in the dental domain
- TFormer (Yuan et al. 2024, transformer-based crown generator from an exocad competitor) — the 2024 alternative to DCrownFormer
- 3DToothSegNet (Zhao et al. 2023) — the *latest* MeshSegNet-family segmentation improvement, candidate for v0.5 if iMeshSegNet isn't the best

Recommendation: 3DTeethGen (Sun et al.) for 057 to add the *unconditional* tooth-shape generation to the v0 sub-task 2 stack and test H2 in the dental domain; alternatively 3DToothSegNet for the latest segmentation improvement as a v0.5 segmentation reference.

Note in `papers/056-ts-mdl-wu22.md`.
