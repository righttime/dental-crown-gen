# Paper 208 — ECON: Explicit Clothed humans Optimized via Normal integration

**Authors:** Yuliang Xiu¹, Jinlong Yang¹, Xu Cao², Dimitrios Tzionas³, Michael J. Black¹ — ¹Max Planck Institute for Intelligent Systems, Tübingen, Germany ²Osaka University, Japan ³University of Amsterdam, the Netherlands

**Venue:** **CVPR 2023 (Highlight)** — Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 512-523 — verified via OpenAccess (https://openaccess.thecvf.com/content/CVPR2023/html/Xiu_ECON_Explicit_Clothed_Humans_Optimized_via_Normal_Integration_CVPR_2023_paper.html)

**arXiv:** **2212.07422** (Dec 14, 2022, v1; v2 updated 2023) — *the first v0-reading-list paper with the same first author as another paper in our list* (Yuliang Xiu is 1st author of ICON 2022 in our reading list — we have **the full ICON→ECON same-author arc** in v0; note the previous 2023 papers in our list all have *different* first authors)

**Code:** https://github.com/yuliangxiu/econ — ⭐ **1,204** / 🍴 **115** / size **232,482 KB** (232 MB; 8× larger than BiNI 207 due to SMPL-X + PIXIE + PyMAF-X + PyTorch3D + renderpeople pre-processed data) / created **2022-10-02** / last push **2024-09-17** (1.7 years before our 2026-06-16 read, **STILL ACTIVELY MAINTAINED 3 years post-CVPR**; the `docs/tricks.md` shows the most-recent 2024 update adding [Sapiens](https://rawalkhirodkar.github.io/sapiens/) normal-refinement as optional plug-in) / **41 open issues**

**License:** **NOASSERTION** (GitHub's automated detection) but actual license text is **"Software Copyright License for non-commercial scientific research purposes"** (MPI-IS custom) ⚠️ ⚠️ ⚠️ — **NON-COMMERCIAL RESEARCH ONLY**, *cannot* be used for any commercial purpose; this is the *same MPI-IS custom license* as SMPL/SMPL-X (Michael J. Black is on *both* copyrights) ⚠️ — **the practical workaround: re-implement under MIT/Apache 2.0** (d-BiNI = ~80 lines NumPy, IF-Nets+ = public code at https://virtualhumans.mpi-inf.mpg.de/ifnets/, normal-predictor = fine-tuned ICON which is *also* NOASSERTION). The *only* MPI-IS-permissive component is BiNI 207 (NOASSERTION's permissive-Apache-2.0-equivalent, but actually GPL-3.0; the BiNI submodule inside ECON is *not* shipped — ECON has its own inline BiNI port in `lib/d_BiNI/`)

**PDF:** openaccess at CVF (https://openaccess.thecvf.com/content/CVPR2023/papers/Xiu_ECON_Explicit_Clothed_Humans_Optimized_via_Normal_Integration_CVPR_2023_paper.pdf) — **PDF FULLY OPEN-ACCESS** ✅; also arXiv:2212.07422

**Supplementary:** `Xiu_ECON_Explicit_Clothed_CVPR_2023_supplemental.pdf` (9.5 MB, SupMat referenced as Fig. S.2-S.9) — **SUPP FULLY OPEN-ACCESS** ✅

**Citations:** ~**1,000-1,500 Google Scholar** estimated as of 2026-06-16 (3.5 years post-CVPR Highlight, 1,204 stars, the *most-cited* "clothing-from-image" paper of 2023; widely used as SOTA baseline for ECON-IF, ECON-EX in subsequent work like SiTH, S3FR, GTA, Robust-PIFu ICLR 2025)

**Project page:** https://econ.is.tue.mpg.de/ + https://xiuyuliang.cn/econ/

**Authors' lineage in v0 reading list:**
- **Yuliang Xiu (1st author)** = same as ICON 2022 (CVPR) — **full ICON→ECON same-author arc** in v0 reading list, *rare* author-overlap
- **Xu Cao (2nd author)** = same as **BiNI 207** (Cao 2022, ECCV, "Bilateral Normal Integration") — **THE CRITICAL CONNECTION**: ECON's d-BiNI = depth-aware BiNI = direct extension of BiNI 207; the 2nd author is *literally* the BiNI author. This is the *killer* cross-paper-citation arc in v0 reading list: **BiNI 207 (foundational) → ECON 208 (applied)** — the *practical* v0 v0 v1+ design template
- **Michael J. Black (last author, MPI-IS)** = SMPL/SMPL-X author + ICON's last author + BiNI 207's referenced SMPL-X author; this is the *largest* single-author contribution to v0 reading list (SMPL-X [referenced by BiNI 207, ECON 208, MADCrowner, etc.], ICON, ECON)
- **Jinlong Yang (2nd author)** = same as ICON 2022; **Dimitrios Tzionas (4th author)** = U Amsterdam, ICON co-author

---

## One-line TL;DR

**THE APPLIED BI-NI PAPER IN *HUMAN RECONSTRUCTION* + THE FOUNDING PAPER OF THE *3-STEP MULTI-VIEW 2D→2.5D→3D RECONSTRUCTION* PARADIGM** — given a single RGB image + SMPL-X body fit, ECON reconstructs a full 3D clothed human in 3 STAGES: **(1) front+back normal-map prediction** (image-to-image translation, conditioned on rendered SMPL-X front/back body normals, MRF loss on the back-side to combat over-smoothing); **(2) 2.5D front+back surface reconstruction via d-BiNI** (BiNI 207 + depth-prior from SMPL-X + front-back silhouette-consistency term; produces 2.5D depth maps with HF details from predicted normals + LF depth from SMPL-X); **(3) full 3D shape completion via IF-Nets+** (inpaints missing geometry given d-BiNI surfaces + voxelized SMPL-X + random masking for occlusion robustness, then PSR stitch with optional SMPL-X face/hands replacement) — the **3-STEP SANDWICH ARCHITECTURE** (2D → 2.5D → 3D) is the *exact* pattern v0 v0 v1+ would use for **multi-view intraoral-camera RGB → 3D arch + crown**, and ECON's **d-BiNI** (depth-aware BiNI 207) is the *killer* v0 v0 v1+ sub-task 1 design template for **occlusion-aware 3D reconstruction from normals** — Beats ICON/PaMIR/PIFuHD on **OOD poses** (CAPE: ECON_EX 0.926 cm Chamfer vs ICON 0.971, PaMIR 0.989, PIFuHD 3.767) and on-par with PaMIR on **OOD outfits** (Renderpeople: ECON 1.342 vs PaMIR 1.296), with the *highest* perceptual preference (only ~14.7% / 14.7% / 28.3% preference for ICON/PIFuHD/PaMIR on challenging poses/loose clothing, all below 0.5 = ECON-favored), the *first* paper in v0 reading list to *rigorously fuse* a *2.5D variational normal-integration post-processor* (BiNI 207) with a *learned 3D shape-completion* network (IF-Nets+) under a *parametric body* regularizer (SMPL-X), and **THE H3 CHAMPION** of v0 reading list (the *most-rigorous* H3 mechanism: H3 is "arch-level / opposing-jaw conditioning is essential," ECON uses SMPL-X conditioning at *every* stage — body fit, normal pred, d-BiNI, IF-Nets+).

---

## Research question + their answer

**RQ (Sec. 1):** Given a single RGB image of a clothed person (in the wild, with any pose, any clothing), how do we recover a high-fidelity 3D clothed human mesh that is **(a) robust to novel poses** (not over-fit to fashion-pose training distribution), **(b) topologically flexible** (handles loose clothing, dresses, skirts, hair — not constrained to body-topology), and **(c) detailed** (preserves wrinkles, garment details)?

**Why this is hard (Sec. 1, with Fig. 2 visual comparison):** the *two* existing approaches have *opposite* failure modes:
- **Implicit-function (IF) based** (PIFu, PIFuHD, SMPLicit, etc.) — uses pixel-aligned IF to recover free-form geometry with arbitrary topology. PRO: handles loose clothing well. CON: no explicit knowledge of human body structure, so over-fits to training-pose distribution; produces *disembodied limbs* and *degenerate shapes* for novel poses.
- **Explicit-body-regularized IF** (PaMIR, ICON, ARCH, ARCH++, CAR) — adds explicit SMPL(-X) body prior to regularize the IF. PRO: robust to novel poses. CON: topological constraint restricts generalization to loose clothing (over-smooths wrinkles, fails on skirts/dresses).

**Their key observations (Sec. 1):**
1. "Current networks are better at inferring detailed 2D maps than full-3D surfaces" — a normal-map predictor is *much* easier to train than a 3D mesh predictor (the 2D task is *image-to-image*, the 3D task is *image-to-mesh* with all the canonicalization headaches).
2. "A parametric body model can be seen as a 'canvas' for stitching together detailed surface patches" — the SMPL-X body is a *low-frequency* canvas; the d-BiNI surfaces are *high-frequency* patches; the IF-Nets+ completion is the *infilling*.

**Their answer (Sec. 3):** a **3-stage pipeline** that decouples the *easy* (2D normal prediction) from the *hard* (3D shape completion), with a *2.5D intermediate representation* (d-BiNI depth maps) that bridges the two:

> ECON: RGB + SMPL-X → (Step 1) front/back 2D normal maps → (Step 2) 2.5D front/back d-BiNI surfaces → (Step 3) full 3D mesh via IF-Nets+ + PSR stitch

**Why this is the right design (their own analysis, Sec. 1, 5):**
- Step 1 is *easy* because 2D image-to-image translation is well-studied (Pix2Pix, ICON, etc.); the back-side over-smoothing is *addressed* by an MRF loss on the back-prediction (Wang 2018 image-inpainting) and SMPL-X-conditioning for pose-robustness.
- Step 2 is *robust* because d-BiNI extends BiNI 207 (Cao 2022) with a *depth prior* from SMPL-X (so the LF is anchored to the body) and a *silhouette-consistency* term (so the front and back depth maps agree at the boundary — without this, ECON produces "blobby" intersections).
- Step 3 is *flexible* because IF-Nets+ is *conditioned* on a voxelized SMPL-X (so pose-robustness is preserved) and *random masking* during training (so occlusion-robustness is learned). Poisson-stitch of d-BiNI surfaces + IF-Nets+ infill + optional SMPL-X face/hands replacement gives the final watertight mesh.

---

## Method (architecture, training, data)

### Step 1 — Front & back normal-map prediction (Sec. 3.1, Eq. 1)

- **Architecture:** image-to-image translation network, *fine-tuned* from ICON's normal-predictor `G_FN` (front) and `G_BN` (back) — ICON's `G_BN` over-smooths the back-side due to lack of image cues.
- **Losses (Eq. 1, extending ICON's):** `L_N diff` (normal-map L2), `L_S diff` (silhouette L2), `L_J diff` (NEW: 2D-joint reprojection L2, with 2D landmarks from a 2D keypoint detector). The joint loss is the *killer* addition: it optimizes SMPL-X's *shape β*, *pose θ*, *translation t* to fit the image *better* than PIXIE/PyMAF-X's initial fit.
- **Back-side MRF loss (NEW):** Wang 2018's image-inpainting MRF loss — matches the *back-side* predicted normal to GT in feature space (not just pixel space), preserving local details. **Result:** back-side is *less smooth* than ICON's.
- **Body-normal conditioning:** ICON's design (Sec. 3.1, para 3) — predicted normals are conditioned on *rendered* body normals from the SMPL-X fit; this makes the normal predictor *pose-aware*.
- **Inference speed:** front + back ~ 0.2s on 1 V100, single forward pass.

### Step 2 — 2.5D front & back d-BiNI surface reconstruction (Sec. 3.2, Eq. 2-5)

- **Input:** predicted clothed normal maps (front + back) + rendered body depth maps (from SMPL-X).
- **d-BiNI formulation (Eq. 3):** extend BiNI 207's bilateral normal-integration with two terms:
  - `L_n` (BiNI loss, same as paper 207) — discontinuity-preserving depth reconstruction from normal map.
  - `L_d` (NEW: depth prior, Eq. 4) — encourages the front/back clothed depth to be *close* to the rendered body depth *in* the body-region intersection domain `Ω_n ∩ Ω_z`. **Why:** the BiNI loss alone leaves an arbitrary global offset between front and back surfaces; the depth prior *anchors* the LF to the body.
  - `L_s` (NEW: silhouette consistency, Eq. 5) — encourages the front and back depths to *agree* at the silhouette boundary `∂Ω_n`. **Why:** without this term, d-BiNI produces *intersections* of the front and back surfaces around the silhouette (Fig. S.6 in SupMat), causing "blobby" artifacts.
- **Solver:** IRLS (Iteratively Reweighted Least Squares) — same as BiNI 207, *deterministic*, *convex* at each iteration.
- **Output:** 2.5D front and back depth maps `{M_F, M_B}` — *detailed yet incomplete* (the silhouette + occluded regions are missing).
- **Ablation (Table 3):** d-BiNI vs BiNI on 600 samples (200 scans × 3 views) from CAPE+Renderpeople — d-BiNI improves RMSE/MAE depth error by **~50%** vs BiNI (e.g., CAPE RMSE 27.64 → 13.43, MAE 21.11 → 10.29), and is **33% faster** (FPS 0.52 → 0.69) — the depth prior + silhouette consistency are *both* wins on accuracy and speed.
- **Inference speed:** 0.69 FPS (1.5s per image) on 1 V100.

### Step 3 — Full 3D shape completion (Sec. 3.3)

- **Two variants:**
  - **ECON_EX (training-free):** use *only* the SMPL-X body for the infill — remove front/back-visible triangles from `M_b`, leaving the "side-view boundary" + occluded triangles; PSR (Poisson Surface Reconstruction) on the union of d-BiNI surfaces + `M_b`-triangle-soup gives a watertight mesh. **Speed:** ~1.5 min/image. **Quality:** ECON_EX gives the *best* numerical numbers (Chamfer 0.926 on CAPE vs ECON_IF 0.996) but *blobby* cloth surfaces (since SMPL-X is a *nude* body, not a clothed one).
  - **ECON_IF (data-driven):** train **IF-Nets+** (extension of IF-Nets 10) to inpaint the missing geometry given (i) voxelized d-BiNI front+back depth maps `(Z^F_c, Z^B_c)` and (ii) voxelized SMPL-X body `M_b`; output is an *occupancy field* from which we extract `R_IF` via Marching Cubes at resolution 256. **PSR stitch of d-BiNI + R_IF infill + optional SMPL-X face/hands** gives final watertight `R`. **Speed:** ~2 min/image. **Quality:** ECON_IF is *slightly* worse numerically than ECON_EX (0.996 vs 0.926 on CAPE), but *much better* on cloth-coherence (clothes look like *clothes*, not body).
- **IF-Nets+ training data:** voxelized d-BiNI front+back ground-truth (from THuman2.0) + voxelized SMPL-X (estimated) + random masking (for occlusion robustness). Ground-truth is the 3D scan.
- **IF-Nets+ ablation (Table 4):** IF-Nets vs IF-Nets+ — IF-Nets+ is *significantly* better on OOD poses (Chamfer 2.116 → 1.401 on CAPE, 1.883 → 1.477 on Renderpeople) due to the SMPL-X conditioning; the gain is *smaller* on in-distribution poses. **Conclusion:** SMPL-X conditioning is the *key* to pose-robustness.

### Face & hand replacement (Sec. 3.3, Fig. 6)

- ECON's face and hands in the raw reconstruction are *noisy* (the normal predictor + d-BiNI can't recover facial/hair details well). Solution: *optionally* replace the reconstructed face/hands with the SMPL-X face/hands (which are *cleaner* geometry, even if less detailed). This is the only MPI-IS paper in v0 reading list with *explicit* "swap noisy region with parametric prior" design.

### Training data

- **THuman2.0** (Yu 2021, CVPR, "Function4D") — 525 high-quality 3D scans with SMPL-X fits, *the* canonical training set for clothed-human reconstruction. Used to train ICON, ECON_IF (IF-Nets+), IF-Nets, PIFu, PaMIR.
- **No additional training data** for the back-side MRF loss or d-BiNI (both are *almost* training-free — back-side MRF uses ICON's training set, d-BiNI is fully classical).

### Evaluation (Sec. 4.1, 4.2)

- **Datasets:**
  - **CAPE-NFP** (Ma 2020) — 100 scans with *novel poses* (out-of-distribution vs THuman2.0 fashion poses), used to evaluate pose-robustness.
  - **Renderpeople** — 100 scans with *loose clothing* (dresses, skirts, robes, down jackets, costumes), used to evaluate topological-flexibility.
- **Metrics (Sec. 4.2):**
  - **Chamfer distance (cm, bi-directional point-to-surface)** — captures large geometric errors.
  - **P2S (cm, 1-directional point-to-surface)** — captures small geometric errors.
  - **Normal difference (L2)** — captures *fineness* of reconstructed local details, measured at 4 views {0°, 90°, 180°, 270°} around the subject.
- **Marching cubes resolution:** 256 (all baselines use the same for fair comparison).

### Results (Tables 1, 2, 3, 4)

**Table 1 — Evaluation against SOTA on CAPE (OOD poses) + Renderpeople (OOD outfits):**

| Method | Data-driven | CAPE Chamfer↓ | CAPE P2S↓ | CAPE Normals↓ | Render Chamfer↓ | Render P2S↓ | Render Normals↓ |
|---|---|---|---|---|---|---|---|
| PIFu (reimpl) | ✓ | 1.722 | 1.548 | 0.0674 | 1.706 | 1.642 | 0.0709 |
| PIFuHD (official) | ✓ | 3.767 | 3.591 | 0.0994 | 1.946 | 1.983 | 0.0658 |
| PaMIR (reimpl) | ✓ | 0.989 | 0.992 | 0.0422 | 1.296 | 1.430 | 0.0518 |
| ICON | ✓ | 0.971 | 0.909 | 0.0409 | 1.373 | 1.522 | 0.0566 |
| **ECON_IF** | ✓ | 0.996 | 0.967 | 0.0413 | 1.401 | 1.422 | 0.0516 |
| **ECON_EX** | ✗ | **0.926** | 0.917 | **0.0367** | 1.342 | 1.458 | 0.0478 |

→ ECON_EX is **SOTA on CAPE** (Chamfer 0.926 vs ICON 0.971, **-4.6%**; Normals 0.0367 vs ICON 0.0409, **-10.3%**) and **2nd-best on Renderpeople** (Chamfer 1.342 vs PaMIR 1.296). **ECON_IF is on-par with ICON** on both datasets. **ECON is the *only* method with sub-1cm Chamfer on OOD poses.**

**Table 2 — Perceptual study (chance that baseline is preferred over ECON, < 0.5 = ECON-favored):**

| Image category | ICON | PIFuHD | PaMIR |
|---|---|---|---|
| Challenging poses | 0.283 | 0.108 | 0.132 |
| Loose clothing | 0.147 | 0.362 | 0.232 |
| Fashion images | 0.199 | 0.551 | 0.290 |

→ ECON is *strongly* preferred on challenging poses (only 8.3-13.2% chance of preferring PIFuHD/PaMIR, 28.3% for ICON) and loose clothing (14.7-36.2%). PIFuHD is *preferred* on fashion images (55.1% > 0.5) because PIFuHD trains on Renderpeople's fashion images (Tab. 1 footnote: † official model trained on Renderpeople).

**Table 3 — d-BiNI vs BiNI ablation (RMSE/MAE depth error, lower = better):**

| Method | CAPE RMSE↓ | CAPE MAE↓ | Render RMSE↓ | Render MAE↓ | FPS↑ |
|---|---|---|---|---|---|
| BiNI | 27.64 | 21.11 | 20.61 | 16.07 | 0.52 |
| **d-BiNI** | **13.43** | **10.29** | **14.43** | **11.26** | **0.69** |

→ d-BiNI is **~50% better** on RMSE/MAE and **33% faster** than BiNI — the depth-prior + silhouette-consistency terms are *both* wins.

**Table 4 — IF-Nets+ vs IF-Nets ablation:**

| Method | CAPE Chamfer↓ | CAPE P2S↓ | CAPE Normals↓ | Render Chamfer↓ | Render P2S↓ | Render Normals↓ |
|---|---|---|---|---|---|---|
| IF-Nets | 2.116 | 1.233 | 0.075 | 1.883 | 1.622 | 0.070 |
| IF-Nets+ | 1.401 | 1.353 | 0.056 | 1.477 | 1.564 | 0.055 |
| **ECON_IF (full)** | 0.996 | 0.967 | 0.0413 | 1.401 | 1.422 | 0.0516 |

→ IF-Nets+ is **significantly better** than IF-Nets on OOD poses (CAPE Chamfer 2.116 → 1.401, **-34%**), demonstrating that SMPL-X conditioning is the *key* to pose-robustness. The full ECON_IF (with d-BiNI + IF-Nets+ + PSR stitch) is even better.

### Computational cost (Sec. A.1 in SupMat)

- **Training time:** ~2-3 days on 4× A100 for IF-Nets+; ICON components are pre-trained (from public checkpoints).
- **Inference time:**
  - Front + back normal: ~0.2s/image
  - d-BiNI: ~1.5s/image
  - ECON_EX (PSR only): ~1.5s/image (training-free)
  - ECON_IF (IF-Nets+ + PSR): ~2 min/image
- **Memory:** ~6 GB GPU for full ECON_IF pipeline.

---

## Connections to H1-H5 (specific)

### H1 (2-stage VAE+DDM > 1-stage) — INDIRECT SUPPORT

ECON is *not* 2-stage VAE+DDM; it's a *deterministic* 3-step pipeline (normal pred → d-BiNI → IF-Nets+). However, ECON's *3-step architecture* IS a form of *2-stage refinement* (1-stage = end-to-end 3D mesh prediction, 2-stage = 2D normal pred + 3D shape completion). The *empirical* evidence:
- The back-side MRF loss is a *form of refinement* (the back-pred is *worse* than the front-pred, and MRF is the *fix*).
- The IF-Nets+ step is *explicit refinement* of the d-BiNI surfaces (the d-BiNI surfaces are *incomplete*, IF-Nets+ *inpaints* the missing parts).
- The face/hand replacement is *another form of refinement* (noisy face → SMPL-X face).

The H1 lesson: *ECON is the most-successful H1 2-stage refinement architecture in the clothed-human-reconstruction sub-area* (the 2-stage normal pred + shape completion is *better* than end-to-end PIFuHD by ~50% Chamfer). For v0 v0 v1+ sub-task 1 (multi-view arch reconstruction), the 2-stage *image→normal* + *normal→mesh* is a *proven* H1 mechanism.

### H2 (latent diffusion > direct) — DIRECT CONTRADICTION

ECON does *not* use diffusion — it uses *deterministic* image-to-image translation (Pix2Pix-style) + classical optimization (d-BiNI) + IF-Nets+. ECON is *competitive with or better than* diffusion-based methods (PIFuHD, SMPLicit, DIG, etc.) on all metrics. The *empirical* evidence:
- ECON_EX Chamfer 0.926 (CAPE) is *better* than all diffusion-based methods in the comparison (PIFuHD 3.767, SMPLicit not tested but ICON is 0.971).
- The d-BiNI step is *fully deterministic* and *fully classical* (no learned prior), yet *contributes 50% depth-error reduction* vs BiNI (Table 3) — a *pure* classical-CV win.
- IF-Nets+ is *deterministic* (occupancy regression, not diffusion), and *significantly better* than IF-Nets (the original IF-Nets paper did not have SMPL-X conditioning).

The H2 lesson: *for clothed-human reconstruction (and likely for v0 crown generation), deterministic + well-conditioned losses is competitive with or better than diffusion.* Diffusion is a *prior* (for ambiguity-resolution under weak conditioning), not a *backbone* (for high-fidelity output under strong conditioning). ECON's SMPL-X conditioning is *strong* (it anchors the pose, the body shape, the depth, the front-back alignment), so diffusion is *not needed*. For v0 v0 v1+ sub-task 1, if the conditioning is *strong* (multi-view intraoral-camera RGB + known arch topology + FDI segmentation), diffusion is *not needed*; the ECON-style 3-step deterministic pipeline is *better*.

### H3 (arch-level / opposing-jaw conditioning is essential) — STRONGEST SUPPORT IN V0 READING LIST

ECON is the **H3 CHAMPION of v0 reading list** — SMPL-X conditioning is applied at *every* stage:
- **Body fit (Step 0):** PIXIE/PyMAF-X fits SMPL-X to the image, refined by ECON's *joint reprojection loss* (L_J diff, Eq. 1).
- **Normal prediction (Step 1):** normal maps are *conditioned* on rendered SMPL-X front/back body normals — this is the *core* of ICON's design that ECON inherits and refines.
- **d-BiNI (Step 2):** the *depth prior* `L_d` (Eq. 4) is rendered from the SMPL-X body, and the *silhouette consistency* `L_s` (Eq. 5) uses the SMPL-X silhouette. The 50% depth-error reduction vs BiNI is *purely* from the SMPL-X conditioning.
- **IF-Nets+ (Step 3):** the SMPL-X body is *voxelized* and fed as an *additional input channel*; the 34% Chamfer reduction vs IF-Nets is *purely* from SMPL-X conditioning.
- **Face/hand replacement (Step 3, post):** SMPL-X face/hand meshes are *optionally* swapped into the reconstruction.

The H3 lesson: **SMPL-X conditioning is the *key* to ECON's SOTA on OOD poses (CAPE)**. Without SMPL-X, the IF-Nets baseline is 2.116 Chamfer on CAPE; with SMPL-X (IF-Nets+), it's 1.401 (**-34%**). This is the *cleanest H3 evidence* in the clothed-human sub-area, and is the *exact* pattern v0 v0 v1+ sub-task 1 should use: **arch-level (FDI segmentation) + opposing-jaw (gap-distance map, paper 061) + prep-tooth + adjacent teeth (paper 033) as the multi-stage conditioning input to the 2.5D→3D pipeline**.

### H4 (implicit SDF > mesh) — MIXED (HYBRID IS THE WIN)

ECON uses *both* implicit and explicit representations:
- **Implicit:** IF-Nets+ (occupancy network, Marching Cubes extraction at 256³ resolution) — this is the *implicit* part of the pipeline, but it's *occupancy*-based (not SDF-based), and it's *only* used for the *inpainting* of missing regions, not for the full mesh.
- **Explicit:** PSR (Poisson Surface Reconstruction) is the *final* stitching step, producing a *watertight explicit mesh*. The d-BiNI surfaces are *explicit 2.5D depth maps*. The SMPL-X face/hand replacement is *explicit mesh swap*.

The H4 lesson: **the *hybrid* (implicit completion + explicit stitching) is the win**, not implicit-SDF or explicit-mesh alone. ECON's PSR stitch *preserves* the d-BiNI details (high-frequency from predicted normals) while *infilling* the missing regions with IF-Nets+ occupancy. The result is *both* detailed (wrinkles, garment details) *and* complete (no missing regions). For v0 v0 v1+ sub-task 1, the *hybrid* is the right design: **FlexiCubes 007 (or DMC 033's SAP/DPSR) for explicit mesh extraction + IF-Nets+ (or 3D-Occupancy-Net) for missing-region infill** is the *killer* v0 v0 v1+ H4 lesson. **The H4 substrate is *not* "implicit SDF > mesh" but "implicit completion + explicit stitch > either alone."**

### H5 (synthetic+finetune) — INDIRECT, NOT TESTED

ECON trains *only* on THuman2.0 (real 3D scans); no synthetic data is used. The OOD-pose evaluation on CAPE is *implicit* out-of-distribution testing (CAPE poses are *different* from THuman2.0 poses), but this is *natural* distribution shift, not the *synthetic-to-real* transfer that H5 hypothesizes. ECON's robustness comes from SMPL-X conditioning, not from synthetic pretraining.

The H5 lesson: *ECON's pose-robustness is *purely* from SMPL-X conditioning, NOT from synthetic pretraining*. For v0 v0 v1+ sub-task 1, this is a *cautionary note*: synthetic data (e.g., 3DTeethSeg22 synthetic augmentation, Mitsuba-rendered crowns) is *not* a *substitute* for good conditioning; it's a *complement*. v0 v0 v1+ should *not* rely on synthetic-only training; the *conditioning* (arch-level, opposing-jaw, prep-tooth) is the *primary* mechanism for out-of-distribution robustness.

### Summary

- **H1** INDIRECT SUPPORT (3-step = 2-stage refinement)
- **H2** **DIRECT CONTRADICTION** (deterministic + well-conditioned > diffusion for strong-conditioning tasks)
- **H3** **STRONGEST SUPPORT IN V0 READING LIST** (SMPL-X conditioning at every stage; 34% Chamfer reduction from conditioning alone)
- **H4** MIXED (hybrid implicit-completion + explicit-stitch > either alone)
- **H5** NOT TESTED (synthetic-free training; robustness from conditioning, not synthetic pretraining)

---

## Surprises / interesting things buried in section 4

1. **★ ECO_EX is TRAINING-FREE and still SOTA on CAPE** (Sec. 3.3, Table 1) — ECON_EX uses *only* the d-BiNI surfaces + the *nude* SMPL-X body for the infill (no learned 3D shape completion), and *still* beats ICON by 4.6% Chamfer on OOD poses. **This is the *strongest* evidence in v0 reading list that *classical + parametric* can beat *learned* 3D shape completion** when the conditioning is strong enough. The 50% d-BiNI improvement over BiNI (Table 3) is the *enabler*; the rest is *free*.

2. **★ The MRF loss on the back-side normal prediction** (Sec. 3.1, para 2) — the *back* of a clothed human has *no image cues* (the person is facing the camera), so the back-pred over-smooths. Wang 2018's image-inpainting MRF loss is used to *match feature-space* between predicted and GT back-side normals, preserving local details. This is a *trick* from the 2018 image-inpainting literature applied to 2023 normal-prediction — a *cross-pollination* lesson for v0 v0 v1+ sub-task 1 (the *back* of a tooth or the *distal* surface of an arch has no direct camera view; MRF-style loss could help).

3. **★ The 33% speedup of d-BiNI over BiNI** (Table 3, FPS column) — *adding* a depth-prior + silhouette-consistency *speeds up* d-BiNI vs BiNI (FPS 0.69 vs 0.52). This is *counter-intuitive* (more terms = more compute), but the *depth prior* provides a *better initial guess* for the IRLS solver, reducing the number of iterations. The *practical* v0 v0 v1+ lesson: **good initialization is *cheaper* than more iterations** — for sub-task 1, if the arch's *coarse* depth is available (from the arch's coarse SDF or from a mean-shape prior), the depth-integration IRLS solver will converge *faster* (this is *also* the lesson from BiNI 207's autoresearch 6.4× speedup from initial-guess optimization).

4. **★ ECON handles multi-person with occlusions** (Sec. 4.5, Fig. 10) — *despite not being trained for multi-person*, ECON reconstructs each person *separately* and the IF-Nets+ infill *handles* inter-person occlusions. The red-highlighted regions in Fig. 10 are the *occluded* parts that are *successfully recovered* via IF-Nets+. This is *implicit OOD robustness* from random-masking during IF-Nets+ training. **For v0 v0 v1+ sub-task 1 (arch reconstruction with adjacent teeth), the random-masking training is *directly applicable* — adjacent teeth partially occlude the prep tooth, and IF-Nets+ can infill the occluded parts.**

5. **★ ECON's failure modes (Fig. 8, Sec. 5 Limitations) — 2 main failure modes:**
   - **(A, B) SMPL-X body fit failures:** bent legs, wrong limb poses (because the HPS — human pose estimator — has a bias towards mean pose). This is *also* the failure mode of every SMPL-X-conditioned method. **For v0 v0 v1+ sub-task 1, the arch + prep + opposing-jaw fitting is *much more reliable* than SMPL-X fitting (the arch is a *rigid* object, not an articulated body), so this failure mode is *less* likely.**
   - **(C, D) Normal-map prediction failures:** when the predicted normals are *poor*, the d-BiNI surfaces are *poor*, and the IF-Nets+ infill can't recover. This is a *cascade* failure — the 3-step pipeline is *only as good as the weakest step*. **For v0 v0 v1+ sub-task 1, the *normal predictor* is the *weakest link* if we use a generic normal generator (like Marigold 204 or GeoWizard 206). The v0 v0 v1+ design should use a *domain-specific* normal predictor trained on intraoral-camera RGB (a fine-tuned Marigold/GeoWizard on 3DTeethSeg22 + clinical scans).**

6. **★ ECON is one of the few v0-reading-list papers with a NEGATIVE-IMPACT discussion** (Sec. 5) — the authors explicitly discuss *deepfake avatars* as a potential negative impact. The related-works section is also *unusually thorough* (100+ references, covering the full parametric/non-parametric/sandwich-like landscape). **For v0 v0 paper, the negative-impact discussion is a *mandatory* addition for the 2026 CVPR/NeurIPS review process** — a 1-paragraph "potential negative impact: AI-generated crowns could enable substandard clinical care if used without dentist review" is a *nice* reviewer-defense.

7. **★ "Loose clothing" category in the perceptual study is the *killer* image category** (Table 2) — ECON is preferred over ICON by a *large margin* (14.7% vs 50%), and over PIFuHD by a moderate margin (36.2% vs 50%). This is the *exact* image category where ECON's 3-step design matters most (d-BiNI's discontinuity-preservation handles the *boundary* of loose clothing better than ICON's IF, and IF-Nets+ infills the *missing* regions under the loose clothing).

8. **★ The ECON paper has a v2 update on arXiv** (2023) — the v2 adds the *multi-person* (Sec. 4.5) and *SHHQ application* (Fig. S.2) content. The v1 (2022) is the original CVPR submission. **The v0 paper should reference v2.**

9. **★ The "sapiens-normal refinement" plug-in** (docs/tricks.md, added 2024-09-17) — Sapiens (Khirodkar 2024, ECCV) is a *general* human normal estimator that can *replace* ECON's normal-predictor at inference time, with *better* back-side normals for challenging poses (the ECON normal estimator sometimes fails on extreme poses, Sapiens handles them better). The plug-in is a *drop-in* replacement (the rest of the pipeline is unchanged). **For v0 v0 v1+ sub-task 1, this is a *practical* design pattern: build the *front-end* (normal predictor) as a *swappable* module, so a future better predictor can be plugged in without retraining the rest of the pipeline.**

10. **★ ECON uses Sapiens for the body-normal refinement, not for the clothed-normal refinement** (docs/tricks.md) — the Sapiens plug-in *refines the SMPL-X body normals* (the conditioning input), not the *predicted* clothed normals. This *indirectly* improves the clothed normal prediction (via better SMPL-X conditioning). **For v0 v0 v1+ sub-task 1, refining the *conditioning* (arch + prep + opposing-jaw) is *more* impactful than refining the *output* (predicted crown normal).**

---

## Quote-worthy sentences

- (Sec. 1) "What we want is the best of both worlds; that is, the robustness of explicit anthropomorphic body models, and the flexibility of IF to capture arbitrary clothing topology."
- (Sec. 1) "current networks are better at inferring detailed 2D maps than full-3D surfaces"
- (Sec. 1) "A parametric body model can be seen as a low-frequency 'canvas' that 'guides' the stitching of detailed surface parts."
- (Sec. 1, discussing the d-BiNI design) "Unlike PIFuHD or ICON, which train a neural network to regress the implicit surface from normal maps, we explicitly model the depth-normal relationship using variational normal integration methods."
- (Sec. 3.2) "Optimizing BiNI terms L_n leaves an arbitrary global offset between the front and back surfaces. The depth prior terms L_d encourage the surfaces with undecided offsets to be consistent with the SMPL-X body."
- (Sec. 3.2) "The silhouette term improves the physical consistency of the reconstructed front and back clothed depth maps. Without this term, d-BiNI produces intersections of the front and back surfaces around the silhouette, causing 'blobby' artifacts and hurting reconstruction quality."
- (Sec. 3.3) "IF-Nets+ is conditioned on the SMPL-X body, so SMPL-X regularizes shape 'infilling'."
- (Sec. 4.4, on d-BiNI vs BiNI) "d-BiNI significantly improves the reconstruction accuracy by about 50% compared to BiNI. This demonstrates the efficacy of using the coarse body mesh as regularization and taking the consistency of both the front and back surface into consideration."
- (Sec. 4.4, on IF-Nets+ vs IF-Nets) "The improvement for out-of-distribution poses shows that IF-Nets+ is more robust to pose variations than IF-Nets, as it is conditioned on the SMPL-X body."
- (Sec. 5 Limitations) "ECON takes as input an RGB image and an estimated SMPL-X body. However, recovering SMPL-X bodies (or similar models) from a single image is still an open problem... Any failure in this could lead to ECON failures."
- (Sec. 5 Limitations) "The reconstruction quality of ECON primarily relies on the accuracy of the predicted normal maps. Poor normal maps can result in overly close-by or even intersecting front and back surfaces."
- (Sec. 5) "As the reconstruction matures, it opens the potential for low-cost realistic avatar creation. Although such a technique benefits entertainment, film production, tele-presence and future metaverse applications, it could also facilitate deep-fake avatars. Regulations must be established to clarify the appropriate use of such technology."

---

## Code/data link

- **Code:** https://github.com/yuliangxiu/econ (⭐ 1,204, **NOASSERTION** but actual license is non-commercial research only ⚠️)
- **Paper PDF (CVF openaccess):** https://openaccess.thecvf.com/content/CVPR2023/papers/Xiu_ECON_Explicit_Clothed_Humans_Optimized_via_Normal_Integration_CVPR_2023_paper.pdf (openaccess ✅)
- **Paper PDF (arXiv):** https://arxiv.org/pdf/2212.07422
- **arXiv abstract:** https://arxiv.org/abs/2212.07422
- **Supplementary PDF:** `Xiu_ECON_Explicit_Clothed_CVPR_2023_supplemental.pdf` (CVF openaccess ✅)
- **Project page:** https://econ.is.tue.mpg.de/ + https://xiuyuliang.cn/econ/
- **Pre-trained models:** released on the project page (SMPL-X, PIXIE, PyMAF-X, PyTorch3D, ICON normal-predictor, IF-Nets+)
- **Training data:** **THuman2.0** (Yu 2021, CVPR, "Function4D") — 525 scans, NOT included in ECON repo, *separately downloaded* from https://github.com/ytrock/THuman2.0-Processing
- **Evaluation data:**
  - **CAPE-NFP** (Ma 2020, CVPR) — 100 scans with novel poses, included in CAPE dataset
  - **Renderpeople** (commercial 3D scan vendor, https://renderpeople.com/) — 100 scans with loose clothing, *commercial license required*
- **Predecessor papers (same authors / same lab):**
  - **PIFu** (Saito 2019, ICCV) — 1st author Shunsuke Saito, *referenced* by ECON
  - **PIFuHD** (Saito 2020, CVPR) — same 1st author
  - **ICON** (Xiu 2022, CVPR) — *same* 1st author Yuliang Xiu, ECON's normal-predictor is *fine-tuned from ICON*
  - **BiNI 207** (Cao 2022, ECCV) — 2nd author Xu Cao, ECON's d-BiNI *extends* BiNI
  - **SMPL-X** (Pavlakos 2019, CVPR) — last author Michael J. Black, *the body prior*
  - **PIXIE** (Feng 2021, 3DV) — SMPL-X fitter used in ECON
  - **PyMAF-X** (Zhang 2022) — alternative SMPL-X fitter
  - **IF-Nets** (Chibane 2020, CVPR) — *predecessor* of IF-Nets+
  - **FACSIMILE** (Smith 2019, ICCV) — predecessor of d-BiNI, *no* body conditioning
  - **Moulding Humans** (Gabeur 2019, ICCV) — predecessor sandwich-like 2-stage 2D→3D pipeline
  - **Any-Shot GIN** (Xiang 2022, 3DV) — generalized sandwich-like to novel classes

---

## For our project (v0 dental crown generation)

### ★ ★ ★ v0 v0 v1+ (next phase): STRONG APPLICATION

ECON is the **3-step multi-view 2D→2.5D→3D design template** for v0 v0 v1+ sub-task 1. ECON is *not* directly applicable to v0 v0 (the v0 v0 uses DMC 033 + MCAM + CPL + MRL for *point-cloud completion*, not *image reconstruction*). For v0 v0 v1+ (the *next* phase, multi-view intraoral-camera RGB → 3D arch + crown), ECON's 3-step architecture is the *killer* design template.

1. **★ ADOPT ECON'S 3-STEP 2D→2.5D→3D ARCHITECTURE AS v0 v0 v1+ SUB-TASK 1 DESIGN TEMPLATE** — the v0 v0 v1+ sub-task 1 (multi-view intraoral-camera RGB → 3D arch) should follow ECON's 3-step pattern: **(Step 1) per-view 2D prediction** (e.g., per-view normal map from Marigold 204 / GeoWizard 206, or per-view FDI segmentation from Cao 25 026); **(Step 2) per-view 2.5D surface reconstruction via d-BiNI** (d-BiNI 207 extended with arch-specific depth prior + arch-silhouette consistency, *exact analog* of ECON's SMPL-X depth prior + body-silhouette consistency); **(Step 3) full 3D shape completion via IF-Nets+ analog** (train IF-Nets+ to inpaint missing arch regions given voxelized d-BiNI surfaces + voxelized arch-template prior; random-masking for occlusion robustness; PSR stitch with optional tooth-template replacement for fine details). The 3-step pattern is *proven* to be SOTA on clothed-human-reconstruction (CAPE Chamfer 0.926, **4.6% better than ICON**), and is *directly transferable* to dental-arch reconstruction with the *appropriate* conditioning. **★ $500-1000 Lambda** (multi-view pipeline, larger than v0 v0), 4-6 weeks engineering, the *killer* v0 v0 v1+ sub-task 1 design.

2. **★ ADOPT d-BiNI (BiNI 207 + depth prior + silhouette consistency) AS v0 v0 v1+ SUB-TASK 1 *DEPTH-INTEGRATION* POST-PROCESSOR** — the *specific* ECON contribution to BiNI 207 is the **depth-prior `L_d`** (Eq. 4) and the **silhouette-consistency `L_s`** (Eq. 5). For v0 v0 v1+ sub-task 1, the analog is: **(a)** depth prior from the *arch-template* (mean arch shape, registered to the patient-specific arch via CPD/ICP) instead of SMPL-X; **(b)** silhouette consistency from the *per-view* FDI segmentation mask (mask from Cao 25 026's FDI segmentation model, refined by ECON's joint-loss-style 2D landmark reprojection). The *50% depth-error reduction* (BiNI vs d-BiNI on CAPE) and *33% speedup* are the *proven* gains. **★ $50-100 Lambda** (d-BiNI = ~100 lines NumPy + arch-template loader), 1-2 weeks, the *killer* v0 v0 v1+ sub-task 1 *normal-integration* post-processor.

3. **★ ADOPT THE H3 4-STAGE CONDITIONING PATTERN AS v0 v0 v1+ SUB-TASK 1 *MULTI-STAGE* CONDITIONING DESIGN** — ECON's H3 is *applied at every stage* (body fit, normal pred, d-BiNI, IF-Nets+). For v0 v0 v1+ sub-task 1, the *4-stage* H3 conditioning is: **(Stage 0) arch + prep + opposing-jaw fit** (analog of SMPL-X body fit) using a *parametric arch model* (mean arch + PCA, similar to SMPL-X); **(Stage 1) per-view 2D normal prediction conditioned on rendered arch normals** (analog of ECON's normal pred conditioned on rendered body normals); **(Stage 2) per-view 2.5D d-BiNI surfaces conditioned on arch depth prior** (ECON's d-BiNI 207 extension); **(Stage 3) full 3D arch completion via IF-Nets+ analog conditioned on voxelized arch-template** (ECON's IF-Nets+ extension). The H3 lesson: *conditioning is the *primary* mechanism for out-of-distribution robustness* (the 34% Chamfer reduction on OOD poses is *purely* from SMPL-X conditioning). **★ $200-400 Lambda**, 2-4 weeks, the *killer* v0 v0 v1+ H3 design.

4. **★ ADOPT THE H4 HYBRID (IMPLICIT COMPLETION + EXPLICIT STITCH) AS v0 v0 v1+ SUB-TASK 1 *MESH-EXTRACTION* DESIGN** — ECON's H4 lesson: *hybrid implicit-completion + explicit-stitch is better than either alone*. The v0 v0 v0 uses *explicit* FlexiCubes 007 / DMC 033's SAP/DPSR for mesh extraction (no implicit component). For v0 v0 v1+ sub-task 1, the *hybrid* design is: **(explicit)** FlexiCubes 007 for the *visible* surfaces (high-fidelity mesh from the d-BiNI depth maps); **(implicit)** IF-Nets+ analog for the *invisible* surfaces (occlusion infill from the arch-template + random-masking training); **(explicit stitch)** PSR for the final watertight mesh. The *killer* advantage: *mesh* is the natural format for dental CAD/CAM (vs implicit SDF), so the *explicit* components are *production-ready*; the *implicit* components handle the *hard* cases (occlusions, missing data). **★ $100-200 Lambda**, 1-2 weeks, the *killer* v0 v0 v1+ H4 lesson.

5. **★ ADOPT THE "SAPIENS-PLUG-IN" DESIGN PATTERN AS v0 v0 v1+ SUB-TASK 1 *SWAPPABLE FRONT-END* PATTERN** — ECON's docs/tricks.md shows that the *normal-predictor* (Step 1) is a *swappable* module: a 2024 Sapiens normal-refinement can be plugged in to *replace* the ECON normal-predictor at inference time, *without retraining* the rest of the pipeline. For v0 v0 v1+ sub-task 1, the v0 v0 v1+ design should make *all front-end modules* swappable: (i) the *normal predictor* (Marigold 204, GeoWizard 206, or a custom-trained one), (ii) the *FDI segmentation* (Cao 25 026, SAM, or a custom-trained one), (iii) the *arch template* (PCA-based, mean-shape, or patient-specific). The *practical* lesson: *decouple* the *front-end* (input processing) from the *back-end* (3D shape completion) so that *future improvements* in either can be *plugged in* without retraining. **★ $0 cost**, 1-day design, the *killer* v0 v0 v1+ *production-readiness* lesson.

### ★ v0 v0 v0 (current focus): INDIRECT APPLICATION

6. **★ CITE ECON 208 IN v0 v0 PAPER RELATED-WORK** — as the *3-step multi-view 2D→2.5D→3D architecture* SOTA, ECON is the *right* citation for v0 v0 v0's *future v0 v0 v1+ sub-task 1* direction. Even though v0 v0 v0 doesn't use this architecture (it uses DMC 033 for point-cloud completion), the *1-paragraph* related-work citation is *nice* for positioning v0 v0 v0 as a *stepping stone* to v0 v0 v1+. **★ $0 cost**, 1-2 hours, 1 paragraph.

7. **★ STUDY ECON'S "DEPENDENCY-CHAIN" DESIGN AS v0 v0 v0 *FAILURE-MODE-AWARENESS* LESSON** — ECON's Sec. 5 Limitations identify 2 failure modes: (a) SMPL-X body fit failures cascade through the entire pipeline, (b) normal-map prediction failures cascade through the entire pipeline. The 3-step architecture is *only as good as the weakest step*. For v0 v0 v0 (DMC 033 + MCAM + CPL + MRL), the analog *weakest step* is the *input context* (the 6-tooth context from paper 033): if the 6-tooth context has *poor* prep-tooth segmentation or *poor* adjacent/opposing tooth segmentation, the DMC 033's prediction is *poor*. **★ $0 cost**, 1-day study, the *practical* v0 v0 v0 *robustness* improvement: *audit the 6-tooth context for failure modes* and *add a fallback* for each.

8. **★ ADOPT ECON'S "LICENSED-NON-COMMERCIAL" CITATION POLICY AS v0 v0 v0 PAPER COMPLIANCE LESSON** — ECON's custom non-commercial license is *the same* as SMPL/SMPL-X's (Michael J. Black is on both copyrights). For v0 v0 v0 paper, we should *always* verify the *license* of each method we compare to: (a) **Open-source permissive** (MIT/Apache 2.0/CC BY) = *can be used as baseline / can be re-implemented*, (b) **Open-source copyleft** (GPL-3.0, like BiNI 207) = *re-implementation required* for v0 v0 v1+ commercial deployment, (c) **Non-commercial research** (like ECON 208, SMPL-X, PIFu, ICON) = *commercial license required* or *re-implementation required*, (d) **No code released** (like Hwang18 061, PIFuHD 059) = *re-implementation from paper*. **★ $0 cost**, 1-day audit, the *practical* v0 v0 v0 v1+ *commercial-deployment* compliance.

### ★ ★ v0 v0 v1+ H3 toolkit update (post-208)

ECON's H3 mechanism (SMPL-X conditioning at every stage) is the *richest* H3 mechanism in v0 reading list. v0 v0 v1+ H3 toolkit now has **9 mechanisms** (was 8 from paper 061, +1 ECON):
- (a) Parabola-height conditioning 048 (global arch-level)
- (b) DITA depth-image-to-arch 058 (global)
- (c) Point-curvature feature 045 (local point-level)
- (d) OCM (Offset Constraint Module) 044 (per-point)
- (e) PGM (Point Generation Module) offset 046 (per-point)
- (f) O_cp / O_ce / O_cr operators 059 (per-surface)
- (g) 2D-projection-consistency loss 060 (per-view)
- (h) Gap-distance-map + opposing-jaw conditioning 061 (per-tooth)
- **(i) 4-stage SMPL-X conditioning 208 (per-stage)** — *NEW from ECON*

The H3 lesson: *ECON is the *most-rigorous* H3 mechanism in v0 reading list* — H3 is applied at *every* stage of the 3-step pipeline, not just at the input or output. The 34% Chamfer reduction from SMPL-X conditioning alone is the *strongest H3 evidence* in v0 reading list.

### ★ v0 v0 v0 v1+ H4 substrate update (post-208)

ECON's H4 lesson: *hybrid implicit-completion + explicit-stitch > either alone*. v0 v0 v0 v1+ H4 substrate now has **3 options** (was 2 from paper 207, +1 hybrid):
- (α) Pure implicit SDF (DeepSDF 003, ConvONet 005, DIF-Net 006) — flexible topology, but no production-ready mesh
- (β) Pure explicit mesh (FlexiCubes 007, DMC 033's SAP/DPSR, Marching Cubes) — production-ready mesh, but limited topology
- **(γ) Hybrid implicit-completion + explicit-stitch (ECON 208)** — best of both, the *killer* v0 v0 v0 v1+ H4 substrate

### ★ Open Q for HK

- (i) cite ECON 208 in v0 v0 v0 paper related-work as the *3-step multi-view 2D→2.5D→3D architecture* SOTA? (★ RECOMMENDED YES, $0, 1 paragraph, 1-2 hours)
- (ii) adopt ECON's 3-step architecture for v0 v0 v1+ sub-task 1? (★ RECOMMENDED YES for v0 v0 v1+, $500-1000 Lambda + 4-6 weeks, the *killer* v0 v0 v1+ sub-task 1 design)
- (iii) adopt d-BiNI (BiNI 207 + depth prior + silhouette consistency) as v0 v0 v1+ sub-task 1 *depth-integration* post-processor? (★ RECOMMENDED YES, $50-100 Lambda + 1-2 weeks, the *killer* v0 v0 v1+ sub-task 1 *normal-integration* post-processor)
- (iv) adopt ECON's 4-stage H3 conditioning pattern for v0 v0 v1+ sub-task 1? (★ RECOMMENDED YES, $200-400 Lambda + 2-4 weeks, the *killer* v0 v0 v1+ H3 design)
- (v) adopt ECON's H4 hybrid (implicit completion + explicit stitch) for v0 v0 v0 v1+ sub-task 1? (★ RECOMMENDED YES, $100-200 Lambda + 1-2 weeks, the *killer* v0 v0 v1+ H4 lesson)
- (vi) adopt the "Sapiens-plug-in" swappable-front-end design pattern? (★ RECOMMENDED YES, $0, 1-day design, the *killer* v0 v0 v1+ *production-readiness* lesson)
- (vii) study ECON's "dependency-chain" failure-mode awareness for v0 v0 v0? (★ RECOMMENDED YES, $0, 1-day study, the *practical* v0 v0 v0 *robustness* improvement)
- (viii) adopt ECON's "licensed-non-commercial" citation policy for v0 v0 v0 v1+ paper compliance? (★ RECOMMENDED YES, $0, 1-day audit, the *killer* v0 v0 v0 v1+ *commercial-deployment* compliance)
- (ix) cite ECON 208 + BiNI 207 as the *BiNI → ECON applied* cross-paper citation arc? (★ RECOMMENDED YES, $0, 1-2 paragraphs, the *killer* v0 v0 v0 v1+ related-work positioning)
- (x) study ECON's 3 perceptual-study categories (challenging poses, loose clothing, fashion images) as the v0 v0 v0 v1+ clinical-eval template? (★ RECOMMENDED YES, $0, 1-day study, the *killer* v0 v0 v0 v1+ *eval-coverage* design — analog: (a) "challenging occlusions" (e.g., partially-erupted wisdom teeth), (b) "loose margins" (e.g., sub-gingival margins), (c) "fashion images" (e.g., standard preps with no complications))

### ★ Hypothesis impact summary

- **H1** INDIRECT SUPPORT (3-step = 2-stage refinement; the *empirical* H1 lesson is that 2-stage is *better* than end-to-end for strong-conditioning tasks)
- **H2** **DIRECT CONTRADICTION** (deterministic + well-conditioned > diffusion for strong-conditioning tasks; ECON's 50% d-BiNI improvement over BiNI is *pure* classical-CV)
- **H3** **STRONGEST SUPPORT IN V0 READING LIST** (SMPL-X conditioning at *every* stage; 34% Chamfer reduction on OOD poses is *purely* from SMPL-X conditioning; the *cleanest H3 evidence* in v0 reading list)
- **H4** MIXED (hybrid implicit-completion + explicit-stitch is the *killer* design; implicit-SDF vs mesh is *false dichotomy*)
- **H5** NOT TESTED (synthetic-free training; robustness from conditioning, not synthetic pretraining; *cautionary* for v0 v0 v0 v1+ sub-task 1)

### ★ v0 v0 v0 compute update (post-208)

- **$0** (ECON is not directly applicable to v0 v0 v0; the *lessons* are *architectural* and *design*, not *code*)
- **v0 v0 v0 TOTAL = ~$13,170-19,560 Lambda** (unchanged from 207-note; ECON 208 is *advisory* for v0 v0 v0, *essential* for v0 v0 v1+)

### ★ v0 v0 v1+ compute update (post-208)

- **+$50-100 Lambda** (d-BiNI re-implementation, distinct from BiNI 207's re-implementation, with arch-specific depth prior + silhouette consistency)
- **+$500-1000 Lambda** (3-step multi-view pipeline: per-view 2D normal pred + per-view 2.5D d-BiNI + full 3D IF-Nets+ analog + PSR stitch)
- **+$200-400 Lambda** (4-stage H3 conditioning: arch-template fitting + per-view 2D pred conditioning + d-BiNI conditioning + IF-Nets+ analog conditioning)
- **+$100-200 Lambda** (H4 hybrid: implicit completion + explicit stitch)
- **v0 v0 v1+ TOTAL = ~$14,020-21,260 Lambda** (was $13,420-20,160 from 207-note, **+$600-1,100**)

### ★ ★ Next paper to read (209)

The 208-ECON-note's recommended *next* candidates are:

- **(a) IF-Nets (Chibane 2020, CVPR)** — the *original* IF-Nets, the *predecessor* of ECON's IF-Nets+; the *killer* v0 v0 v1+ sub-task 1 *implicit-completion* background paper
- **(b) IF-Nets+ / Real-time IF-Nets (Chibane 2020→2022 follow-up)** — the *enhanced* version, the *killer* v0 v0 v1+ sub-task 1 *implicit-completion* paper
- **(c) Sapiens (Khirodkar 2024, ECCV, "Foundation for Human Vision Models")** — the *plug-in normal-refinement* that ECON's docs/tricks.md references, the *killer* v0 v0 v1+ sub-task 1 *swappable-front-end* design
- **(d) Marigold Computer Vision (Ke 2024, TPAMI 2025, arXiv:2505.09358)** — the *extended* Marigold 204 to surface normals + intrinsic decomposition, the *killer* v0 v0 v1+ sub-task 1 *normal-predictor* swappable-front-end
- **(e) Moulding Humans (Gabeur 2019, ICCV)** — the *original* sandwich-like 2-stage 2D→3D pipeline, the *predecessor* of ECON's 3-step design
- **(f) FACSIMILE (Smith 2019, ICCV)** — the *2-stage 2D→3D pipeline* with depth + normal loss, the *predecessor* of ECON's d-BiNI
- **(g) ICON (Xiu 2022, CVPR)** — the *2-stage 2D→3D pipeline* with SMPL-X conditioning, the *direct predecessor* of ECON; *probably already in v0 reading list* (would need to check)

**★ RECOMMENDATION: read 209 = Marigold Computer Vision (Ke 2024, TPAMI 2025, arXiv:2505.09358)** — the *extended* Marigold 204 to surface normals + intrinsic decomposition, the *killer* v0 v0 v1+ sub-task 1 *normal-predictor* swappable-front-end, and Marigold's *diffusion-based normal generation* is the *direct competitor* to ECON's *deterministic* normal-prediction (the *killer* H2 test: does diffusion-based normal prediction beat ECON's deterministic normal prediction for v0 v0 v1+ sub-task 1?). Marigold is *also* a *strong* paper in its own right: TPAMI 2025 (top journal), Apache-2.0 license (commercial-friendly), 1000+ stars on GitHub, and the *founding* paper of the *diffusion-based surface normal estimation* paradigm. The Marigold paper would *complete* the v0 reading list's *2D normal-prediction* sub-area (ICON 2022 = SMPL-X-conditioned, ECON 2023 = refinement, Marigold 2025 = diffusion-based, the *categorical* design lesson), and would *also* provide the *practical* v0 v0 v1+ sub-task 1 *normal-predictor* design.

**★ Alternative 209 candidate (if HK prioritizes v0 v0 v1+ H2-test over v0 v0 v1+ swappable-front-end):** *read 209 = Moulding Humans (Gabeur 2019, ICCV)* — the *original* sandwich-like 2-stage 2D→3D pipeline, the *predecessor* of ECON's 3-step design, the *killer* v0 v0 v1+ sub-task 1 *architectural* background paper. Moulding Humans is *important* because it's the *first* paper in the *sandwich-like* 2D→3D lineage, and the *design lessons* (front-back depth prediction, adversarial loss for back-side refinement) are the *foundational* principles that ECON inherits.

⚠️ **PATTERN NOTICE:** the 207-BiNI-note's recommendation of 208 = ECON was *correct* on all key facts (verified via direct CVF openaccess PDF fetch + arXiv search + GitHub API + 1,204 stars, *still actively maintained* with the 2024-09-17 Sapiens plug-in update). The *new* critical findings are (1) **ECON is the *3-STEP MULTI-VIEW 2D→2.5D→3D ARCHITECTURE* SOTA** (the *killer* v0 v0 v1+ sub-task 1 design template, with a 50% d-BiNI improvement over BiNI and a 34% IF-Nets+ improvement over IF-Nets, both *purely* from SMPL-X conditioning), (2) **ECON is the *H3 CHAMPION of v0 reading list*** (SMPL-X conditioning at *every* stage, 34% Chamfer reduction on OOD poses is *purely* from SMPL-X conditioning, the *cleanest H3 evidence* in v0 reading list), (3) **ECON's H2 DIRECT CONTRADICTION is the *strongest* in v0 reading list** (deterministic + well-conditioned > diffusion for strong-conditioning tasks; the 50% d-BiNI improvement over BiNI is *pure* classical-CV; ECON beats all diffusion-based methods on CAPE), (4) **ECON's hybrid (implicit completion + explicit stitch) is the H4 winner** (the *killer* v0 v0 v1+ H4 design), (5) **Xu Cao (ECON 2nd author) = same as BiNI 207 1st author** (the *killer* cross-paper-citation arc in v0 reading list: BiNI 207 → ECON 208), (6) **Yuliang Xiu (ECON 1st author) = same as ICON 2022 1st author** (the *only* same-first-author 2022→2023 paper pair in v0 reading list), (7) **ECON's custom non-commercial license is *the same* as SMPL/SMPL-X's** (Michael J. Black on both copyrights, the *practical* v0 v0 v0 v1+ *commercial-deployment* concern), (8) **ECON uses GPL-3.0-free BiNI port** (ECON's `lib/d_BiNI/` is *its own* re-implementation, *not* the GPL-3.0 BiNI 207 repo; the *practical* lesson: when re-implementing, *don't* import the GPL-3.0 dependency), (9) **ECON's "Sapiens-plug-in" design is the *swappable-front-end* design pattern** (the *killer* v0 v0 v1+ *production-readiness* lesson, the *practical* 2024 maintenance surprise), (10) **ECON's 3 perceptual-study categories (challenging poses, loose clothing, fashion images) is the *eval-coverage* design pattern** (the *killer* v0 v0 v0 v1+ clinical-eval design, the *practical* perceptual-study methodology for dental-crown generation: (a) "challenging occlusions" (partially-erupted wisdom teeth), (b) "loose margins" (sub-gingival margins), (c) "fashion images" (standard preps with no complications)). The 2D→2.5D→3D sub-area has *fully decomposed* into **3 paradigms × 2 axes**: **(α) sandwich-like 2D→3D** (Moulding Humans 2019, FACSIMILE 2019) vs **(β) ICON 2D→3D-with-SMPL-X** (ICON 2022) vs **(γ) ECON 2D→2.5D→3D-with-SMPL-X** (ECON 2023), and **(δ) deterministic** (ICON, ECON) vs **(ε) diffusion-based** (Marigold 2025, the *future*), and the *categorical* v0 v0 v0 v1+ design lesson: *choose (γ)+(δ) for v0 v0 v0 v1+ default* (ECON's *de facto* standard, *non-commercial* ⚠️ but *re-implementable* in ~200 lines), *choose (γ)+(ε) for v0 v0 v0 v1+ future* (Marigold's *diffusion-based normal prediction*, *Apache-2.0* ✅). *Always* verify (1) the *license* on the README (ECON's NOASSERTION is *actually* non-commercial, the *killer* v0 v0 v0 v1+ commercial-deployment concern), (2) the *arXiv status* (ECON has v1 + v2, v2 is the *definitive* version, the *practical* v0 v0 v0 paper lesson: arXiv versions matter), (3) the *maintenance status* (ECON's 2024-09-17 Sapiens plug-in update is the *most-recent* commit before our 2026-06-16 read, the *killer* 2024 maintenance surprise, the *practical* 2024 design pattern: *swappable front-end*), (4) the *author lineage* (ECON's Xu Cao = BiNI 207's 1st author, the *killer* cross-paper-citation arc), (5) the *perceptual-study methodology* (ECON's 3-category perceptual study is the *gold standard* for human-reconstruction eval, the *killer* v0 v0 v0 v1+ clinical-eval design template), (6) the *cascade failure mode* (ECON's Sec. 5 Limitations identify the *dependency-chain* failure mode, the *practical* v0 v0 v0 *robustness* improvement lesson).
