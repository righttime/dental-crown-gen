# 001 — 3DTeethSeg'22: 3D Teeth Scan Segmentation and Labeling Challenge

- **Title:** 3DTeethSeg'22: 3D Teeth Scan Segmentation and Labeling Challenge
- **Authors:** Achraf Ben-Hamadou, Oussama Smaoui, Ahmed Rekik, Sergi Pujades, Edmond Boyer, Hoyeon Lim, Minchang Kim, Minkyung Lee, Minyoung Chung, Yeong-Gil Shin, Mathieu Leclercu, Lucia Cevidanes, Juan Carlos Prieto, Shaojie Zhuang, Guangshun Wei, Zhiming Cui, Yuanfeng Zhou, Tudor Dascalu, Bulat Ibragimov, Tae-Hoon Yong, Hong-Gi Ahn, Wan Kim, Jae-Hwan Han, Byungsun Choi, Niels van Nistelrooij, Steven Kempers, Shankeeth Vinayahalingam, Julien Strippoli, Aurélien Thollot, Hugo Setbon, Cyril Trosset, Edouard Ladroit
- **Year:** 2023 (preprint; challenge held at MICCAI 2022)
- **Venue:** MICCAI 2022 Singapore — Satellite Event / Challenge track
- **Link:** https://arxiv.org/abs/2305.18277
- **Code/data:** https://github.com/abenhamadou/3DTeethSeg22_challenge (code + data via Figshare)

---

## TL;DR

A MICCAI 2022 challenge that released the largest public intra-oral 3D scan dataset to date (1,800 scans, 900 patients, 23,999 labeled teeth) and benchmarked 6 algorithms on **localization + segmentation + FDI labeling**; the winning stack (CGIP) hits TSA = 0.9859 / TIR = 0.9100 with a Point-Transformer + offset-clustering + boundary-aware sampling pipeline, but no method handles missing/damaged teeth or braces well.

## Research question

> Can a shared public benchmark + a held-out docker-evaluated test set push the field from handcrafted curvature/active-contour methods to learning-based 3D segmentation that is robust to real-world intra-oral scan variation?

Their answer: yes — but only on "clean" adult scans. The challenge deliberately limited scope to anatomical variation across subjects; braces, missing teeth, and damaged teeth are deferred to future editions.

## Method

### Dataset
- 1,800 scans from 900 patients (2 scans/patient: upper + lower jaw).
- GDPR-compliant; anonymized; sourced from clinics in France/Belgium.
- 50% orthodontic / 50% prosthetic; 50/50 M/F; ~70% under 16 y/o.
- Acquired with **Primescan, Trios3, iTero Element 2 Plus** (10–90 µm accuracy, 30–80 pts/mm²).
- Train: 1,200 scans / 16,004 teeth. Test: 600 scans / 7,995 teeth.
- Format: OBJ meshes + JSON per-vertex `labels` (FDI 1–32 + 0 for gingiva) and `instances`.

### Annotation pipeline (8 steps, human-machine hybrid)
1. Preprocessing + PCA-based pose normalization (align to occlusal plane)
2. Manual 3D cropping with a tight sphere per tooth
3. **Harmonic parameterization** (UV flatten via Eck et al. 1995) + curvature overlay — annotator works in 2D
4. Manual boundary annotation in UV space
5. Back-propagate boundaries to 3D
6. Extract crown meshes
7. **FDI labeling** (32 teeth + gingiva)
8. Clinical validation loop (returns to step 2/4/7 on issue)

### Evaluation metrics
- **TLA** (Teeth Localization Accuracy): `mean(Exp(-normalized_distance))`, with a 5× tooth-size penalty for missing/empty centroids.
- **TSA** (Teeth Segmentation Accuracy): F1 over per-tooth instance point clouds.
- **TIR** (Teeth Identification Rate): fraction of GT teeth whose nearest predicted centroid is < 0.5× tooth size away AND has the correct FDI label.
- **Global score** = average of the three.

### The 6 winning methods (one paragraph each)

| Team | Backbone | Strategy | Key trick |
|------|----------|----------|-----------|
| **CGIP** 🥇 | Point Transformer (Zhao 2021) | End-to-end on point cloud | Boundary-Aware Point Sampling + offset-based clustering (PointGroup-style) + Tooth Cropping Module to refine gingiva |
| **FiboSeg** 🥈 | Residual U-Net on 2D renders (Pytorch3D) | Multi-view 2D → back-project | Normals encoded as RGB + depth as 4th channel; weighted majority voting + island removal |
| **IGIP** 🥉 | PointNet++ for centroids + custom patch net | Multi-stage (separate→centroid→patch→classify) | **Dental arch curve fitting** on centroids for post-hoc label correction |
| **TeethSeg** | 3D U-Net (volumetric) + Random Walker | Voxelize → coarse seg → mesh refinement | Random Walker steered by local convexity at edges |
| **OS** | HRNet heatmap (2D top-down) + GC-Learn (Lian 2020) mesh seg | 2D centroid heatmap → crop → 3D seg | HRNet gives 16-class FDI in one shot; KNN upsample back to original mesh |
| **Champers** | Stratified Transformer (Lai 2022) | Two-stage: centroid then cascade seg | Normalized-Euclidean + Separation centroid loss; multi-crop proposal merging (IoU ≥ 0.35) |

## Results

| Team | Exp(-TLA) | TSA | TIR | Score |
|------|-----------|------|------|-------|
| **CGIP** | 0.9658 | **0.9859** | 0.9100 | **0.9539** |
| FiboSeg | **0.9924** | 0.9293 | 0.9223 | 0.9480 |
| IGIP | 0.9244 | 0.9750 | **0.9289** | 0.9427 |
| TeethSeg | 0.9184 | 0.9678 | 0.8538 | 0.9133 |
| OS | 0.7845 | 0.9693 | 0.8940 | 0.8826 |
| Champers | 0.6242 | 0.8886 | 0.8795 | 0.7974 |

- **Best localization:** FiboSeg (2D render approach is geometrically well-conditioned).
- **Best segmentation:** CGIP (boundary-aware sampling wins at tooth-gum edges).
- **Best labeling:** IGIP (shape+position concatenation + arch-curve post-processing).
- **Common failure modes:** missing-tooth detection (IGIP, OS), inaccurate tooth-boundary delineation (TeethSeg), and gum-teeth border quality (FiboSeg).

## Connections to our hypotheses

- **H1 (2-stage > end-to-end):** **Supports H1.** All 6 top methods are 2-stage (centroid detection → per-tooth crop segmentation). Even the "end-to-end" Point Transformer CGIP internally clusters offsets then re-runs a cropping module — a 2-stage structure. Pure single-pass end-to-end approaches (ToothNet, FeaStNet variants) are noted as prior-art but absent from winners. **Caveat:** the boundary-aware sampling trick shows the 2 stages can share a backbone, so "stage" is more about compute than architecture.

- **H2 (diffusion > mesh VAE for surface):** **Inconclusive here.** All methods are discriminative (segmentation), not generative (surface synthesis). This paper is about *finding* teeth, not *generating* crowns. Useful baseline infra (dataset, F1 metrics) but doesn't speak to diffusion.

- **H3 (conditioning on opposing + adjacent improves outer surface):** **Direct support, but inverted.** The IGIP team's "shape + position" concatenation for FDI labeling is a tiny analogue of "use the dental arch (adjacent context) to refine per-tooth labels." Their dental-arch-curve post-processing literally fits a Bezier through the centroids and uses it to fix classification errors. This is the exact same inductive bias we need for outer-surface generation: **the global dental arch is a strong prior on any single tooth's geometry**. Strongly supports conditioning on arch.

- **H4 (implicit SDF > explicit mesh):** **Mildly contradicts** the current SOTA. Every winning method works on explicit meshes/point clouds, not implicit SDFs. The reason is that segmentation/labeling only needs per-vertex classification, and explicit representations make boundary refinement (KNN upsample, Random Walker on the mesh) cheap. For our generation problem, SDF may still win — but this paper gives us no direct evidence. We should read DeepSDF/DiGS before concluding.

- **H5 (synthetic CAD can bootstrap training):** **No evidence either way**, but a useful signal: the annotation pipeline itself uses 3D *mesh* editing (UV flatten, manual boundary) rather than neural synthesis. Real clinical scans are still the gold standard; synthetic data bootstrapping is untested. Worth running our own experiment once we have a baseline.

## Surprises / things buried in section 4

1. **The UV-flatten annotation trick is genius.** They UV-map each tooth (Eck et al. 1995 harmonic parameterization) so the human annotator draws boundaries on a flat 2D image with curvature overlay. Then back-project. This is a *data labeling* technique, not a model — but it's probably the reason 23,999 teeth got labeled with high accuracy. Borrow this idea for our own annotation of any dental data we acquire.
2. **PointGroup-style offset clustering beats per-pixel classification for instance seg** — CGIP and OS both rely on learning per-point offsets to centroids then DBSCAN. This is the same trick as the original PointGroup (Jiang 2020) — instance segmentation is essentially "regress to center, then cluster." We can use this exact pattern for separating adjacent crowns during generation.
3. **FiboSeg's 2D multi-view approach wins localization despite losing segmentation** — suggests that 2D renders are a strong inductive bias when geometry is well-conditioned (centroids), but they bleed at fine boundaries. A **hybrid render-then-refine-on-3D** strategy could be a cheap win for us.
4. **DBSCAN density clustering appears 3 times** (CGIP, OS, Champers). It's the de-facto "tooth deduplicator" — every team uses it to merge multi-crop predictions. Robust, no training, deterministic.
5. **Boundary-Aware Point Sampling (CGIP)** — they run Farthest Point Sampling twice: once uniform, once concentrated at predicted tooth boundaries. This explicitly compensates for the IOS's over-sampling at edges. Pure engineering insight worth replicating.
6. **The dataset is "easy"** — 70% under 16, no braces, no damaged teeth in the test set. The challenge deliberately punted on the hard cases. **Real clinical scans will be much harder.** Our system must handle the cases the challenge didn't test.

## Quote-worthy sentences

> *"Teeth segmentation and labeling is difficult as a result of the inherent similarities between teeth shapes as well as their ambiguous positions on jaws."* (sec 1)

> *"Setting the optimal threshold value for surface curvature-based methods is not straightforward. Indeed, these methods are still sensitive to noise, and selecting the wrong threshold can systematically affect the segmentation accuracy."* (sec 1.3)

> *"The clustering-based tooth instance labeling process is robust because each tooth instance has inherently a compact cylinder shape that is easy to group."* (CGIP, sec 4.1.2)

> *"These results emphasize the diversity and strengths of the different methods, showcasing their effectiveness in specific aspects of the challenge."* (sec 5.1)

> *"Future directions could include the incorporation of more variabilities in the dataset, such as more challenging cases with missing or damaged teeth and ambiguous labeling scenarios."* (sec 6) — **this is literally our problem statement.**

## Code & data

- **Code:** https://github.com/abenhamadou/3DTeethSeg22_challenge (eval scripts, docker templates)
- **Data:** hosted on Figshare (link in repo README). 1,200 train + 600 test scans, OBJ + JSON, ~120 GB? (need to check)
- **Docker challenge format:** participants submitted a docker container, run on hidden test. This is the gold standard for preventing overfit-to-test — we should copy this for any internal benchmarks we run.

## For our project

Concrete next steps ordered by priority:

1. **Download the dataset this week.** Even just the train split (1,200 scans). This is the single biggest unblocker — it gives us a public, well-annotated source of *real* intra-oral scans with FDI labels. The README says it's GDPR-compliant and public.
2. **Use the FDI labels as ground-truth for sub-task 1 (tooth detection / "which tooth is missing?").** Train PointNet++ or Point Transformer on this directly. Estimated effort: 1–2 days. We now have a baseline for sub-task 1.
3. **Lift the UV-flatten annotation trick for our own data.** If/when we acquire private scans, use harmonic parameterization + manual 2D boundary drawing instead of annotating 3D meshes directly. Speedup will be 5–10×.
4. **Use the dental-arch-curve (Bezier) prior from IGIP for sub-task 4 (outer surface generation).** The arch is a strong global prior; we should encode it as a conditioning signal in any diffusion model. **Strong evidence for H3.**
5. **Adopt the PointGroup offset-clustering + DBSCAN pattern** to separate adjacent crown instances in our generated mesh. Don't reinvent instance separation; this is the standard recipe.
6. **Read the team's individual methods papers in detail** for the next few weekly digests: Cui 2020 (TSegNet) is the canonical 2-stage baseline; Lai 2022 (Stratified Transformer) is the architectural state-of-the-art.
7. **Open question for our architecture:** do we need to re-implement segmentation, or can we *assume* tooth instances are given (FDI labels + per-tooth masks) and only generate the missing tooth's mesh? If yes, our problem shrinks dramatically. Discuss with HK.

---
*Scholar 🦉 — 2026-06-05*
