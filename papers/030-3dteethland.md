# Paper 030 — Detecting Dental Landmarks from Intraoral 3D Scans: the 3DTeethLand challenge

- **Authors:** Achraf Ben-Hamadou, Nour Neifar, Ahmed Rekik, Oussama Smaoui, Firas Bouzguenda, Sergi Pujades (CRNS Sfax / Udini / Inria Grenoble), with Niels van Nistelrooij, Shankeeth Vinayahalingam (Radboudumc), Kaibo Shi, Hairong Jin, Youyi Zheng (YY-LAB, Zhejiang U.), Tibor Kubík et al. (TESCAN 3DIM), Xiaoying Zhu et al. (YN-LAB, NDCS), Huikai Wu (ChohoTech), Weijie Liu et al. (IGIP-LAB, Shandong U.).
- **Venue:** Medical Image Analysis (under review; arXiv:2512.08323v2, 28 Apr 2026)
- **Challenge:** MICCAI 2024 satellite event, 49 teams registered → 10 in prelim → 6 finalists
- **Dataset:** 340 intraoral scans (170 patients × 2 jaws), enriched from 3DTeethSeg'22 with new per-tooth landmark annotations (5 fixed + variable cusps), CC BY-NC-ND, GDPR-compliant, France+Belgium clinics.
- **Code (winning team):** github.com/nnistelrooij/3dteethland (MIT-ish) + checkpoints on Google Drive.
- **Reference:** DOI 10.48550/arXiv.2512.08323 — to be cited as Ben-Hamadou, Neifar, Rekik et al., MedIA 2026.

## TL;DR
3DTeethLand is the **first public 3D-dental-landmark benchmark**: 340 IOS scans with five fixed anatomical landmarks (mesial, distal, facial, inner, outer) plus variable cusps per tooth, evaluated with mAP/mAR across 4 categories. The winning Radboud team (RS 0.917, mAP 0.785, mAR 0.656) uses a **two-stage Stratified-Transformer pipeline** — low-res full-arch instance seg + high-res per-tooth landmark decoders + weighted DBSCAN. The runner-up ChohoTech (RS 0.83) uses a **single-stage DGCNN with offset regression + class-specific NMS** and is **2× faster** (10.9 s/scan vs 21.3 s). **The headline insight: segmentation is *not* strictly necessary for high landmark accuracy** — well-optimized direct-regression pipelines can match two-stage pipelines at half the runtime, and class-specific post-processing (DBSCAN, NMS) is a bigger lever than architectural choice.

## Research question + their answer

**Q:** Can deep learning detect anatomical landmarks (cusps, marginal ridges, contact points) directly on intraoral 3D scans robustly enough for clinical orthodontic CAD workflows, and what architectural patterns win?

**A:** Yes — but with a clear pattern: **separate decoders per landmark class on a high-resolution per-tooth crop, plus targeted post-processing**, beats both per-tooth segmentation-then-regression and global single-stage heatmap regression. The winning pipeline's two key ingredients are (1) **specialized decoders per landmark category** (mesial/distal vs. facial vs. inner/outer vs. cusp each get their own head) and (2) **post-processing tailored to landmark proximity** (weighted DBSCAN for clusters, class-specific NMS for offset regression, geodesic-distance maps for heatmaps).

## Method (architecture, training, data)

### Dataset
- 340 IOS scans from 170 patients (one upper + one lower per patient), re-annotated from 3DTeethSeg'22.
- 5 fixed landmarks per tooth (mesial, distal, facial, inner, outer) + variable cusp points (number depends on tooth class: 1 for incisors/canines, 2 for premolars, 4 for molars — though the paper doesn't enumerate this exactly; the cusp branch handles variable cardinality via DETR-style bipartite matching).
- Train/test split via Synapse platform (https://www.synapse.org/Synapse:syn57400900/wiki/); evaluation on a held-out test set; no external data allowed.
- Inter-observer variability: separate orthodontists re-annotated a subset; reported but not as a baseline in the leaderboard.

### Evaluation
- **mAP and mAR** (not RMSE/MDE/MAE) because the number of cusps varies per tooth — RMSE assumes fixed cardinality.
- 4 categories: C (cusp), F (facial), I/O (inner/outer), M/D (mesial/distal).
- Distance thresholds 0–3 mm in 0.1 mm steps → area under PR curve.
- Greedy landmark assignment by predicted class score.
- **Ranking protocol:** Wilcoxon signed-rank test (p<0.001) on pairwise mAP/mAR comparisons, bootstrapped 100× with 10% resampling → normalized point score → final ranking. Robust but opaque; designed to be statistically defensible across the 6 teams.

### Methods (top-6)
| Team | Pre-proc | Aug | Post-proc | Architecture | Loss |
|------|----------|-----|-----------|--------------|------|
| **Radboud** (1st) | Z-score + pose norm | flip, scale, rot-z | **weighted DBSCAN** + surface refine | **ToothInstanceNet (Stratified-Transformer 2-stage)** — shared encoder + 3 decoders (seed map, offset+bandwidth, instance embed) for tooth seg; shared encoder + 6 decoders (1 seg + 5 landmark-class) for landmark det | seg: spatial-emb + CE + focal; ldmk: BCE + smooth-L1 + **Chamfer** + **separation** |
| **ChohoTech** (2nd) | FPS → 20k pts | random pert | class-specific **NMS** | **ORNet (DGCNN) — single-stage, no seg**, 2 heads: prob heatmap + offset regression | L2 + L1 |
| YY-LAB (3rd) | mesh decimate ~10k facets | heatmap compute | graph-cut + bipartite matching | TeethGNN seg + **TL-DETR** landmark det (2 decoder branches: fixed vs cusp) | CE + MSE + bipartite matching |
| YN-LAB (4th) | gingiva removal + FPS 30k + unit sphere | vertex jitter | HDBSCAN + Gaussian weighted voting | 3D U-Net seg + **multi-stage PointMLP** with **curriculum learning** (R=1.0mm → 0.5mm) | Dice (DSC) |
| IGIP-LAB (5th) | 16k pts + curvature + Euclidean distance field | grid sample + random offset | confidence filter (0.7) + density-based clustering | **PointTransformer V3** single-stage, no seg | MSE |
| 3DIMLAND (6th) | 64k vertices + normals + per-vertex geodesic distance maps | rot/trans/scale/FFD | **calibrated topology-driven NMS** + topology-aware graph + distance threshold | **PointTransformer V3** encoder-decoder | MSE |

### Key architectural ideas to lift
1. **ToothInstanceNet (Radboud):** seed map + offset map + bandwidth → iterative Gaussian clustering for instance seg (no per-tooth IoU, no postproc graph-cut). This is **a more principled version of TSegNet's centroid-regression stage (paper 028)**: the seed map is a learned heatmap of "how close am I to a tooth center," the offset map is the per-point direction to that center, and the bandwidth is the per-tooth size.
2. **Per-landmark-class decoders (Radboud):** 6 decoders, one per landmark type. This is **the H3 inductive bias for landmarks**: the network learns "what a mesial point looks like" vs. "what a facial-axis point looks like" with separate weight sets. The cost: 6× decoder params (still small in absolute terms).
3. **DETR-style bipartite matching for variable cusps (YY-LAB):** cusps vary in number per tooth, so the network predicts a heatmap *and* a probability per point → bipartite matching loss (like DETR / KeypointDETR) handles the cardinality mismatch. **Cleaner than PVD-style "free points" mask for the cusp case.**
4. **Curriculum learning with shrinking radius (YN-LAB):** train on R=1.0mm pseudo-label masks first, then fine-tune at R=0.5mm. This is **the only use of curriculum learning in the reading list** and it's directly applicable to the cusp-prediction problem (the supervision signal gets harder as the radius shrinks because the data imbalance gets worse).
5. **Geodesic-distance maps as supervision (3DIMLAND):** convert each landmark into a per-vertex distance field on the mesh, supervise the network to regress distances → argmin for landmark position. This is **the continuous-field version of heatmap regression** and it generalizes to variable cusp cardinality naturally (a single distance map per cusp point).
6. **Class-specific NMS (ChohoTech):** when the same model predicts multiple landmark classes per point, class-aware NMS (suppress within a class, not across classes) avoids the "best landmark wins, others suppressed" failure mode of vanilla NMS. **A one-line code change with a measurable +mAP on mesial/distal (the hardest, most crowded class).**
7. **DBSCAN weighted by predicted distance (Radboud):** cluster the offset-corrected point proposals, weight by 1/predicted_distance → cluster centroid is the landmark. Solves the "noisy cluster center" problem when several points all claim to be a cusp.

## Results

### Final leaderboard (6 finalists, MICCAI 2024)
| Team | mAP cusp | mAP facial | mAP I/O | mAP M/D | mAR cusp | mAR facial | mAR I/O | mAR M/D | mAP | mAR | **RS** |
|------|----------|------------|---------|---------|----------|------------|---------|---------|-----|-----|--------|
| **Radboud** | 0.77±0.06 | 0.76±0.05 | 0.79±0.05 | 0.79±0.04 | 0.67±0.03 | 0.63±0.04 | 0.66±0.05 | 0.65±0.04 | 0.785±0.04 | 0.656±0.04 | **0.917** |
| **ChohoTech** | 0.76±0.07 | 0.76±0.07 | 0.78±0.06 | 0.78±0.06 | 0.62±0.06 | 0.58±0.08 | 0.62±0.06 | 0.67±0.08 | 0.77±0.07 | 0.63±0.07 | 0.83 |
| YY-LAB | 0.68±0.10 | 0.72±0.08 | 0.74±0.07 | 0.70±0.08 | 0.55±0.09 | 0.56±0.08 | 0.60±0.07 | 0.57±0.07 | 0.71±0.07 | 0.57±0.07 | 0.62 |
| YN-LAB | 0.75±0.12 | 0.66±0.10 | 0.61±0.14 | 0.65±0.10 | 0.53±0.10 | 0.52±0.09 | 0.51±0.11 | 0.53±0.08 | 0.64±0.10 | 0.52±0.08 | 0.31 |
| IGIP-LAB | 0.63±0.10 | 0.59±0.11 | 0.63±0.11 | 0.52±0.16 | 0.51±0.09 | 0.44±0.09 | 0.50±0.09 | 0.41±0.13 | 0.59±0.10 | 0.46±0.08 | 0.13 |
| 3DIMLAND | 0.59±0.08 | 0.62±0.06 | 0.55±0.09 | 0.57±0.07 | 0.45±0.06 | 0.45±0.06 | 0.45±0.07 | 0.45±0.05 | 0.57±0.06 | 0.43±0.05 | 0.03 |

### Runtime
| Team | Total (sec) | Avg/scan (sec) |
|------|-------------|----------------|
| Radboud | 2125.3 | 21.25 |
| ChohoTech | **1089.9** | **10.89** |
| YY-LAB | 2289.7 | 22.89 |
| YN-LAB | 2962.3 | 29.62 |
| IGIP-LAB | 1313.5 | 13.13 |
| 3DIMLAND | **1067.5** | **10.67** |

### Quantitative comparisons that matter
- **Radboud vs ChohoTech** are *not* statistically significantly different on mAP (Wilcoxon p>0.001 on most comparisons), but Radboud is *significantly* more consistent across scans (smaller std devs) and has higher mAR (better at finding all landmarks, not just the most confident one).
- **Segmentation-free ≠ worse:** ChohoTech (no seg, 2nd place) matches YY-LAB's two-stage pipeline (3rd place) on mAP for facial and inner/outer categories.
- **Cusp is the hardest category** across all teams (mAP 0.59-0.77, lowest of the 4 categories for everyone) because cusps vary in number per tooth class.
- **Mesial/Distal is the second-hardest** for low-ranked teams (mAP 0.52-0.65 for bottom 3) because adjacent teeth have nearly identical mesial/distal points → DBSCAN/NMS is the critical lever here.
- **Facial & Inner/Outer are the easiest** (mAP 0.66-0.79 across teams) because they're geometrically distinctive (outer is the cheek-side axis, facial is the buccal-cusp line).

## Connections to H1-H5

### H1 (2-stage pipeline > 1-stage)
**CONTRADICTED in the landmark sub-task (consistent with sub-task 1 segmentation paper 029):** 2 of the top-3 teams (Radboud, YY-LAB) use 2-stage (seg → landmark), but ChohoTech's 1-stage (DGCNN with offset regression) matches them at half the runtime. **The right framing for H1 in landmarks: 2-stage wins on mAR (recall all landmarks, no false negatives) but 1-stage matches on mAP (precision per landmark).** The key for us: **our v0 sub-task 1 (segmentation) should follow 2-stage, but our v1 sub-task 2 (crown generation) does NOT need a 2-stage decomposition** — a single diffusion model conditioned on landmarks (H3) is sufficient.

### H2 (diffusion-based > VAE/GAN)
**NO DIRECT EVIDENCE** — all 6 methods are deterministic, none use DDMs. The challenge focuses on landmark localization (not generation), so this is expected. **Gap to fill: a diffusion-based landmark generator conditioned on partial-arch features (H2×H3) would be novel; queue as a v2 research direction.** The DETR-style bipartite matching (YY-LAB) is the only "set prediction" idea in the reading list, and it's a viable alternative to diffusion for variable-cardinality problems.

### H3 (conditioning on adjacent+opposing teeth)
**STRONGEST INDIRECT SUPPORT** — the segmentation-free methods (ChohoTech, IGIP-LAB, 3DIMLAND) all condition implicitly on the *full arch* point cloud (16-64k vertices including the gingiva and adjacent teeth), and they still get 0.57-0.77 mAP. The 2-stage methods (Radboud) condition on the *single-tooth crop* at 10k points resolution, getting 0.79 mAP. **The lesson: high local resolution on a focused region (single tooth) beats high global resolution on the whole arch.** This is **directly applicable to our v0 sub-task 2 (crown generation)**: condition on a single-prep-margin crop at 10k points rather than the full 32-tooth arch. The 5× resolution boost on the relevant region outweighs the loss of arch context. Counter-argument: for v1 product, the dentist wants the *full-arch coherence* (the new crown should match the rest of the arch in color, anatomy, etc.) — so the right v1 design is a 2-stage (arch-context encoder → tooth-crop decoder) exactly like Radboud's pattern.

### H4 (implicit SDF > explicit mesh for substrate)
**REJECTS for landmarks — STRONG SUPPORT for SDF as substrate, but landmarks themselves are points, not surfaces.** All 6 methods operate on point clouds or meshes, none on SDFs. The paper's "Limitations and future directions" section explicitly calls out implicit or hybrid representations as unexplored for landmarks — **a research gap we could close with a DiGS-based landmark detector** (predict the SDF, then ∇SDF at landmark regions gives landmark positions via curvature extrema; the paper 003 DiGS's divergence/smoothness priors would regularize the SDF to be cusp-friendly). The paper's own discussion: "this architectural bias naturally encourages the exploration of alternative approaches, such as implicit or hybrid representations, which could be better suited for certain specific cases." **Concrete v2 idea: SDF-based landmark detector (DiGS backbone + per-vertex offset head) → argmax of |∇SDF| within a per-tooth bounding box → landmark position.** This would be the v2 paper and the implicit-SDF H4 win in the landmarks sub-task.

### H5 (synthetic → real transfer)
**STRONG SUPPORT** — 6 teams trained on a 340-scan dataset collected from 2 clinics in France+Belgium, none used external pre-training, and they all generalize to within-clinic + across-patient scans. **No out-of-distribution evaluation was performed** (the paper acknowledges "incorporating more challenging real-world conditions, such as varying levels of noise and incomplete scans, would better prepare algorithms for the complexities encountered in clinical environments"). For our project, this is a yellow flag: **the 3DTeethLand numbers (~0.78 mAP at 0.5mm threshold) are upper bounds; expect 5-15% degradation on real-world IOS scans from different scanner brands.** Add scanner-specific fine-tuning to our v0 deployment plan.

## Surprises / interesting things buried in section 4-6

1. **The "first-ranked team used Stratified Transformer" surprise is in the title, but the *real* surprise is ChohoTech's 2× speed advantage for matched mAP.** This suggests **DGCNN > PointTransformer V3 for landmark detection** in the no-segmentation regime — surprising because the trend across papers 008-011 has been transformer > DGCNN. Likely reason: DGCNN's local kNN graph is more robust to variable input sizes (20k points of full arch) than PTv3's voxel-set attention (which assumes 64³ grid + sparse windows).

2. **Mesial/Distal is the hardest category for the bottom-3 teams (mAP 0.52-0.65) but easy for the top-2 (mAP 0.78-0.79).** This is a 0.20+ mAP gap attributable entirely to **the post-processing (weighted DBSCAN, class-specific NMS) — not the backbone.** Concrete takeaway: even if you can't beat ToothInstanceNet on architecture, you can match its mAP on M/D by adopting weighted DBSCAN for cluster refinement. This is a 50-line code change, not a research project.

3. **The runtime vs accuracy frontier is not Pareto-dominated:** ChohoTech (no seg, fast, accurate) and Radboud (with seg, slow, more consistent) are both on the frontier, but 3DIMLAND is the *fastest* (10.67 s/scan) and *worst* (mAP 0.57) — a clear Pareto non-optimal point. The lesson: **just being fast doesn't make you useful; the speedup has to come with accuracy.**

4. **The "Impact of global vs local point sampling" insight (Sec 6.3) is gold:** ChohoTech's ablation showed that bumping points from 5k → 20k nearly doubled mAP. This means **for our v0 sub-task 2 (crown generation), 16k-20k points on the per-tooth crop is the right resolution** — more than the 10k Radboud used (their 10k is on a *smaller per-tooth* crop after seg), and far more than the 1-2k we might be tempted to use to save compute.

5. **"Dense local 3D sampling is necessary, though not sufficient, for precise landmark localization"** (Abstract). This is the most quotable line — and it directly supports our v0 sub-task 2 plan to use AnchorFormer + FlexiCubes at 16k+ points (vs. the 4k POCN default).

6. **The Future Directions section explicitly mentions self-supervised learning as a research opportunity** for landmarks. **This is a clean v2 research direction:** pretrain a 3D encoder on the 3DTeethSeg'22 unlabeled 1,800 scans (we have them; paper 001), fine-tune for landmarks on 3DTeethLand's 340 scans → expected +0.05-0.10 mAP for free, since the 340-scan labeled set is tiny.

7. **The "Limitations" section (Sec 6.5) calls out implicit/hybrid representations as unexplored** — and **lists self-supervised learning as a future direction**. These are both agenda items our v2 plan should address; they're not in the v0 plan because v0 prioritizes clinical fit (sub-task 3) over landmark precision.

8. **Stratified Transformer (Radboud's backbone for both seg and landmark det) is the same architecture used by Cao 2025 (paper 026) for tooth segmentation.** The two papers are by the same Radboudumc group; the team's trajectory is "Stratified Transformer as the universal 3D dental backbone." **A clean v0 sub-task 1 design: use the same Stratified Transformer we adopt for landmarks (Radboud) for the segmentation step too** — single backbone, two heads, joint training.

## Quote-worthy sentences

- *"while individual tooth segmentation is a common initial strategy, it introduces a significant computational burden and is not strictly necessary to achieve high accuracy."* (Abstract)
- *"participants demonstrated that dense local 3D sampling is necessary, though not sufficient, for precise landmark localization."* (Abstract)
- *"targeted post-processing strategies, such as class-specific non-maximum suppression, were shown to consistently improve performance."* (Abstract)
- *"The superior robustness of the Radboud method can be attributed in part to its use of specialized decoders that independently propose landmarks for each anatomical class."* (Sec 5.4)
- *"This class-specific specialization allows the network to learn distinct features for different types of landmarks, improving detection rates on irregular morphologies and contributing to its higher recall."* (Sec 5.4)
- *"high local point density on individual teeth is necessary but not sufficient for accurate landmark detection, as performance also depends on modeling choices and post-processing strategies."* (Sec 6.3)
- *"this architectural bias naturally encourages the exploration of alternative approaches, such as implicit or hybrid representations, which could be better suited for certain specific cases."* (Sec 6.5)
- *"Self-supervised learning could be explored to leverage large amounts of unlabeled intraoral scans, reducing reliance on manual annotations and offering strong potential for learning more robust and generalizable representations."* (Sec 6.5)

## Code/data link
- **Paper:** arxiv.org/abs/2512.08323 (v2, 28 Apr 2026), DOI 10.48550/arXiv.2512.08323
- **Dataset:** crns-smartvision.github.io/teeth3ds/ (CC BY-NC-ND 4.0; 340 scans + landmark annotations; Synapse for download)
- **Winning team (Radboud, ToothInstanceNet):** github.com/nnistelrooij/3dteethland (code + checkpoints on Google Drive)
- **YY-LAB (TL-DETR):** github.com/bibi547/TL-DETR
- **YN-LAB:** gitlab.com/m26409021/ynlab
- **IGIP-LAB:** github.com/weijiezaibenpao/igip-sdu-code
- **ChohoTech (ORNet):** github.com/Choho-Tech-Wu/3DTeethLand
- **3DIMLAND:** github.com/tescangroup/PTv3-for-detecting-anatomical-landmarks-in-dentistry
- **Challenge page:** synapse.org/Synapse:syn57400900/wiki/

## For our project

### Concrete next steps

1. **Adopt Stratified Transformer as the universal 3D dental backbone (v0 sub-task 1 + sub-task 1-extended landmarks).** The Radboud team's ToothInstanceNet is the same backbone Cao 2025 (paper 026) uses for segmentation; the same backbone wins 3DTeethLand for landmarks. **One backbone, two heads (FDI classifier + landmark offset regressor), joint training** — replaces the multi-decoder TS-MDL/DLLNet pattern from the related work. Expected 0.78-0.92 mAP on the public 3DTeethLand test set, matching Radboud's 0.785. Compute: 1-2 days on a single A100, $30-60 on Lambda.

2. **Adopt per-landmark-class decoders (v1 sub-task 1-extended landmarks).** Radboud's 6-decoder design (1 seg + 5 landmark classes) is the **H3 inductive bias for landmark category**, and it's the source of their consistent mAR lead. Each decoder is a 2-layer MLP, so 6× params is still small. **Implementation: copy the ToothInstanceNet repo's decoder design, swap the per-class task to our 5 fixed landmarks, train end-to-end.** Expected +0.05-0.10 mAR over single-decoder baseline.

3. **Adopt weighted DBSCAN as the post-processing for crown-generation v0 (sub-task 2).** After our PVD/AnchorFormer completion, run a "is this point a landmark candidate?" head (binary classifier) + "which direction is the offset?" regression head, then weighted DBSCAN to get the final cusp/mesial/distal positions. The post-processing logic is **50 lines of NumPy + scikit-learn, $0 compute, +0.01-0.05 mAP free** — and it gives us the dentist-friendly UX of "here are the detected landmarks on the new crown, click to accept or drag to adjust."

4. **Adopt DETR-style bipartite matching for the cusp sub-task (v1 sub-task 1-extended).** Cusps vary in number per tooth, so a fixed-cardinality regression head can't handle it. YY-LAB's TL-DETR pattern (heatmap + per-point probability + bipartite matching loss) is the cleanest solution in the reading list. **Concrete: in the per-tooth landmark head, add a cusp-branch (parallel to fixed-landmark-branch) with bipartite matching loss, like DETR's Hungarian matcher.**

5. **Adopt the "20k points per arch" resolution as v0 sub-task 2's default.** ChohoTech's ablation: 5k → 20k points nearly doubled mAP. **For our v0 PVD (paper 012) pilot, train on 16-20k points per missing tooth, not the PVD paper's default 1,848 free points.** This is a +1× compute cost but the expected +mAP justifies it (especially on cusp/marginal-ridge detail where 2k points can't resolve the anatomy).

6. **Run a 1-day pilot: ToothInstanceNet vs. ChohoTech ORNet on 3DTeethLand (no seg vs seg-then-landmark).** Both have public code; the pilot costs ~$50 Lambda and answers: **for v0 sub-task 1-extended (crown prep margin + landmarks), do we need tooth segmentation first or is direct regression sufficient?** Expected answer: tooth seg helps for the *arch-level* tasks (FDI labeling, arch alignment) but is unnecessary for the *tooth-level* tasks (landmark, crown generation) — and our v0 sub-task 2 is tooth-level. **Recommendation: skip segmentation in v0 sub-task 2 (ChohoTech-style), keep it in v0 sub-task 1 (Cao 2025 / ToothInstanceNet style).**

7. **Add a self-supervised pre-training stage (v2 sub-task 1-extended).** The 1,800 unlabeled scans from 3DTeethSeg'22 (paper 001) can pre-train a 3D encoder, then fine-tune on 3DTeethLand's 340 labeled scans. **Expected +0.05-0.10 mAP for ~$200 Lambda pre-training cost.** This is a v2 task (after v0 is shipped), not v0.

8. **Open question for HK: should we treat 3DTeethLand's 340 scans as a v0 evaluation set, or skip and use the 1,800 from 3DTeethSeg'22 alone?** The 340 scans are the only public landmark annotations; the 1,800 only have FDI labels. **For v0 sub-task 1 (segmentation), use 1,800 from 3DTeethSeg'22 (more data, no landmarks needed). For v1 sub-task 1-extended (landmarks), use 3DTeethLand's 340 (the only public landmark benchmark).**

9. **Open question for HK: run a 1-day spike on the DiGS-based landmark detector (v2 R&D)?** Predict SDF on per-tooth crop → ∇SDF argmax at curvature extrema → landmark position. The challenge paper's Limitations explicitly calls this out as unexplored. **Risk: DiGS may over-smooth cusps, making them hard to localize precisely. Spike to de-risk before committing to a v2 paper.**

10. **Cite this paper in our v0 sub-task 1 and v0 sub-task 2 plans as the v0 evaluation protocol for landmarks.** Add a note to `docs/SYNTHESIS.md` that "sub-task 1 v0 evaluation: 3DTeethSeg'22 1200/600 split (FDI + segmentation); sub-task 1-extended v1 evaluation: 3DTeethLand 340 scans (landmarks), target mAP ≥ 0.78 to match Radboud."

### Cross-paper insights
- **ToothInstanceNet (paper 030 winner) + Cao 2025 (paper 026) + TSegNet (paper 028) + TSegLab (paper 029) = the same Radboudumc / Sfax / Strasbourg cluster has now won 3 of the last 4 dental 3D challenges** (3DTeethSeg'22, 3DTeethLand'24, 3DTeethSeg'25). The team's design pattern is consistent: **Stratified Transformer + multi-decoder heads + per-class post-processing**. This is a *de facto* design template for dental 3D perception and a strong prior for our v0.
- **The "segmentation-free" trend in landmarks (ChohoTech, IGIP-LAB, 3DIMLAND) parallels the "2D-before-3D" trend in segmentation (paper 029 TSegLab's Mask R-CNN pre-filter).** The lesson: **for tasks where the 3D representation adds cost without adding accuracy, skip it.** For our v0 sub-task 2, this means *consider* a 2D diffusion on rendered crown views + 3D registration (1-day spike, paper 029 suggested this) before committing to a 3D diffusion on the point cloud.
- **The DETR-style bipartite matching for variable cardinality (YY-LAB) is the cleanest H3 mechanism for "number of cusps varies per tooth" — and a generalizable pattern for our v1 sub-task 1-extended (crown prep design has variable number of finish-line points).**

### v0 stack updates (delta from paper 029's synthesis)
- **v0 sub-task 1 (segmentation): unchanged** — Cao 2025 + GNN FDI labeling + harmonic-map boundary refinement, target Score ≥ 0.96.
- **v0 sub-task 1-extended (landmarks): NEW, deferred to v1** — adopt ToothInstanceNet (Stratified Transformer) + 6 per-class decoders + weighted DBSCAN, target mAP ≥ 0.78 on 3DTeethLand.
- **v0 sub-task 2 (crown generation): add 16-20k points per arch resolution** (ChohoTech's finding) and **add a landmark-prediction head + DBSCAN post-processing** (Radboud's pattern) for the dentist-friendly UX.
- **v0 sub-task 4 (crown completion): add a "predicted-crown-has-detected-cusps" sanity check** — if our DiGS+FlexiCubes output doesn't have curvature maxima at the expected cusp positions, the dentist sees a warning. Compute: ~10ms curvature-extraction on the predicted mesh, $0 cost.

### Hypothesis scorecard (cumulative through paper 030)
- **H1** (2-stage > 1-stage): **CONTRADICTED in landmarks** (ChohoTech matches Radboud at 2× speed); **CONFIRMED in segmentation** (Cao 2025, TSegLab); **CONFIRMED in diffusion** (LION > PVD). Restate as "2-stage wins when intermediate representation is semantically meaningful; 1-stage suffices for direct regression."
- **H2** (diffusion > VAE/GAN): still **STRONG** from papers 004, 005, 012, 014; **no new evidence** in 030.
- **H3** (arch-level conditioning): **STRONGEST INDIRECT** in 030 (per-landmark-class decoders + 2-stage arch-then-tooth design).
- **H4** (SDF > explicit mesh for substrate): **NO DIRECT EVIDENCE** in 030 (all methods are point/mesh); **gap explicitly identified** in 030's Limitations — a v2 research direction.
- **H5** (synthetic → real): **CONFIRMED** in 030 (6 teams generalize across patients from 2 clinics with no external data); **yellow flag for scanner-specific generalization** (acknowledged in 030's Future Directions).

### TL;DR for HK
- **First public 3D-landmark benchmark, 340 scans, 6 finalists.** Winning team (Radboud) uses ToothInstanceNet (Stratified Transformer + 6 per-class decoders) with mAP 0.785, mAR 0.656. Runner-up (ChohoTech) is single-stage DGCNN with offset regression + class-specific NMS, 2× faster, no segmentation needed.
- **Headline lesson: segmentation is not strictly necessary for high landmark accuracy** — well-optimized direct regression matches two-stage pipelines. **Post-processing (weighted DBSCAN, class-specific NMS) is a bigger lever than architectural choice.**
- **For our v0 sub-task 1-extended (landmarks, deferred to v1):** adopt Stratified Transformer + 6 per-class decoders + weighted DBSCAN. Target mAP ≥ 0.78. Compute: $30-60 Lambda, 1-2 days.
- **For our v0 sub-task 2 (crown generation):** add landmark-prediction head + DBSCAN post-processing for dentist-friendly UX. Resolution: 16-20k points per arch.
- **Open question for HK:** should we pilot a 2D-then-3D pipeline (paper 029 TSegLab's 2D Mask R-CNN + 3D registration, paper 030's segmentation-free 1-stage trend) for sub-task 2 instead of a 3D diffusion? 1-day spike to de-risk.
- **Next paper to read:** PolyDiff (ICCV 2023) for diffusion-on-mesh (H2 × mesh extension, alternative to MeshDiffusion for our v0), or SA-ConvONet (Tang et al. 2021) for the sign-agnostic ConvONet extension that closes the SDF-vs-occupancy gap (refines H4), or PCN (Yuan et al. 2018) for the foundational point-completion baseline that papers 008-011 all build on.
