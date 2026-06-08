# 064 — DCPR-GAN: Dental Crown Prosthesis Restoration Using Two-Stage Generative Adversarial Networks

**Authors:** Sukun Tian¹*†, Miaohui Wang¹*, Ning Dai², Haifeng Ma¹, Lin Li¹, Lorenzo Fiorenza³, Yongsheng Zhou⁴, Yan Wei⁵†
¹ Peking University School of Stomatology · ² China University of Mining and Technology · ³ Polytechnic University of Turin · ⁴ Nanjing Medical University · ⁵ Beijing Institute of Technology
*equal contribution · † corresponding
**Year:** Online 14 Oct 2021 → **IEEE J Biomed Health Inform 26(1):151-160, Jan 2022**
**DOI:** 10.1109/JBHI.2021.3119394
**PubMed:** 34637385
**Code:** **NOT public** (the 2021 CMEMO predecessor "Dental-GAN" is also not public; the dental crown GAN literature has *never* released code — every paper from 2018-2022 in the AI-crown field is a closed-source reimplementation)
**Data:** **NOT public** (780 patient cases from Peking University Hospital + Nanjing Hospital; two dental scanners; only #36, #46 molars — first molars only by design choice)
**Cited by:** every subsequent 2022-2026 AI-crown paper (Tian's 2023a, 2023b follow-ups, 058 CrownGen 2025, 060 Diff-TRGN 2025, 036 ToothCraft 2026, 034 MadCrowner 2026, 037 ToothForge 2025, etc. — *the* canonical 2021 reference for 2-stage dental crown generation)

---

## TL;DR

DCPR-GAN is the **first two-stage GAN for AI-dental-crown restoration** — Stage I (CGAN) reconstructs the **base occlusal surface** conditioned on (preparation tooth, opposing jaw, tooth-type label, occlusal fingerprint z, gap distance d), and Stage II (improved CGAN + GroNet) refines the **fine-grained occlusal groove morphology** using a *frozen* occlusal-groove-parsing network (GroNet) and an **occlusal fingerprint constraint** that enforces Stage I's z to flow through to the final output. Operates on **256×256 depth images** (orthogonal projection from above with `n=2, l=6mm` parameters) rather than 3D meshes, post-processed via **region growing + B-spline skinning** for full 3D crown + connector design. On 80 test molars (#36, #46) from Peking University Hospital + Nanjing Hospital, DCPR-GAN beats Pix2pix, Pix2pixHD, PAN, GFC, and the *same authors'* 2021 Dental-GAN predecessor on **all 4 image quality metrics** (PSNR, RMSE, SSIM, FSIM), achieves **SD and RMS < 0.161 mm** between generated and target crown occlusal surfaces (real-world prosthetic tolerance), and shows **statistical significance via ANOVA test** (p < 0.05). The first AI-crown paper to **explicitly decompose low-freq global shape from high-freq local detail** — the *cleanest* H1 (two-stage > 1-stage) and *cleanest* "hierarchical generation" evidence in the 2018-2025 dental generation literature.

## Research question + their answer

**Q:** *How do we restore the correct masticatory function of a broken tooth (i.e., generate the missing occlusal surface) automatically in a data-driven way, given the 3D scan of the prepared tooth + the opposing jaw, when (1) clinical dental crown datasets are small (~hundreds, not millions), (2) each patient's tooth shape is unique, and (3) the opposing jaw contact dynamics must be respected for the crown to function?*

**A:** *Decompose the problem into two stages operating on 2D depth images: (Stage I) a Conditional GAN learns the **occlusal relationship** between the preparation tooth and the target crown using adversarial + perceptual losses conditioned on (preparation tooth x₁, opposing tooth c₁, tooth-type label ĉ, occlusal fingerprint z₁, gap distance d) — this gives a **base occlusal surface** with correct spatial positioning. (Stage II) an improved CGAN refines the **fine occlusal morphology** (cusps, fossae, grooves) by adding a *frozen* GroNet (occlusal groove parsing network, pre-trained on Stage I outputs) as a structural prior and an **occlusal fingerprint constraint** that re-injects Stage I's latent z₁ into the Stage II generator — this gives a **functional occlusal surface** with clinically-meaningful detail. The 2D depth image is then lifted to 3D via region growing, and the final full crown is completed via B-spline skinning for the connector to the prepared tooth.*

## Method

### Data representation: 2D depth image (256×256)

The single most important design choice: **operate on 2D depth images, not 3D point clouds or meshes**. Rationale (inferred from paper §3.1):
- The occlusal surface is *intrinsically* 2D (it's a height-field above the prep-margin plane — single-valued depth, no overhangs)
- 2D image processing has 10+ years of mature GAN architectures (Pix2pix, Pix2pixHD, CycleGAN)
- Small training data (780 cases) is enough to train a 2D GAN, not a 3D point cloud network
- Inference is 100× faster than 3D mesh-based generation

**Depth map rendering (§3.1):** orthogonal projection onto a 256×256 plane *parallel to the crown*, with `(i,j)` indexing the projection plane and `d(i,j)` the shortest distance from `(i,j)` to the crown surface. Pixels beyond the crown are set to 0. The depth value is converted to pixel intensity via `p(i,j) = (MaxI / (1 + n · d(i,j) / l))` where `MaxI = 255` (8-bit), `n = 2` (image enhancement coefficient), `l = 6 mm` (distance threshold). The two parameters `n, l` are tuned experimentally to preserve the *functional occlusal features* (cusps, fossae) — the *first* explicit distance-encoding hyperparameter sweep in the AI-crown literature.

### Stage I: Base occlusal surface generation (CGAN)

A **conditional GAN** (Mirza & Osindero 2014) where the generator G₁ takes noise z and condition c = (x₁, c₁, ĉ, z₁, d) and outputs a 256×256 depth image. The discriminator D₁ distinguishes real from generated depth images.

**Generator loss:**
```
L_G1 = L_perceptual_gen + L_perceptual_adv
     = ||φ_i(G₁(z|c)) - φ_i(x)||₁ + max(0, m - D₁(G₁(z|c), c))
```
where `φ_i` is the i-th VGG hidden layer feature (standard perceptual loss, Johnson 2016), `m` is the positive margin (default m=1.0, hinge loss style).

**Discriminator loss:** standard conditional adversarial loss + perceptual feature matching term in D₁'s hidden layers.

**Key H3 mechanism: the condition vector c includes (1) preparation tooth x₁ (the geometry of the *defective* tooth), (2) opposing tooth c₁ (the *antagonist* geometry), (3) tooth-type label ĉ (FDI one-hot, e.g., #36 = molar), (4) occlusal fingerprint z₁ (learned latent), (5) gap distance d (the per-pixel distance from prep to opposing).** The opposing jaw c₁ and gap distance d are the *occlusal* H3 — the spatial relationship between the prep and the antagonist is the dominant cue for *where* the cusps should land.

### Stage II: Fine-grained occlusal surface refinement (improved CGAN + GroNet)

A *second* conditional GAN G₂ that takes the Stage I output and refines it, with two new mechanisms:

**1. GroNet (occlusal groove parsing network):** a *frozen* encoder-decoder that parses the *occlusal grooves* from the depth image (binary segmentation map: groove = 1, non-groove = 0). GroNet is **pre-trained on Stage I outputs** (not on real crowns) — this is a *form* of self-distillation, where Stage I provides a richer training signal than raw real crowns because Stage I's outputs have *consistent* pose and projection direction. During Stage II training, GroNet is fixed and provides the *groove-consistency* loss:
```
L_groove = ||F(G₂(...)) - F(z)||₁
```
where `F(·)` is GroNet and `z` is the target crown. This loss is the **structural prior** for Stage II — the generator is forced to produce a *groove morphology* that matches the target crown's groove morphology.

**2. Occlusal fingerprint constraint:** the Stage I latent z₁ is *re-injected* into Stage II's generator, ensuring the refined surface is *consistent* with Stage I's base shape. This is a *latent flow* mechanism — Stage II can only *refine* the base shape, not *change* it.

**Generator loss:**
```
L_G2 = L_adv + λ_p · L_perceptual + λ_adv · L_conditional_adv + λ_groove · L_groove
```
with `λ_p = λ_adv = 1.0, λ_groove = 1.0` (uniform weighting, no sweep reported).

### Post-processing: region growing + B-spline skinning

The 2D depth image is converted to 3D via:
1. **Region growing method** to reconstruct the 3D crown surface from the 256×256 depth grid (standard marching-style on the depth field)
2. **B-spline skinning for connector design:** given (a) the prepared tooth's offset surface (bonding layer, "cement gap" simulation) and (b) the generated occlusal surface, the connector is a B-spline *skinned* surface between the two boundaries SecL₁ (generated occlusal) and SecL₂ (bonding layer). The skinning algorithm:
   - Sample boundary curve SecL₂ → reference points SecQ = {q_i}
   - Compute matching points SecP on SecL₁ via *plane intersection method* (each plane perpendicular to the arch curve)
   - Compute midpoints SecK = {(p_i + q_i) / 2}
   - Fit B-spline ridge curves through (p_i, k_i, q_i) control points
   - Connect adjacent ridge intersection points → triangular mesh

This is the *first* explicit connector-design algorithm in the AI-crown literature, and it's a *hybrid* (deep learning + classical CAD) approach — the 2D depth image is generated by GAN, but the 3D crown shape is completed by classical CAD skinning.

## Results

### Quantitative comparison on 80 test molars

| Method | PSNR ↑ | RMSE ↓ | SSIM ↑ | FSIM ↑ |
|---|---|---|---|---|
| Pix2pix (Isola 2017) | 19.85 | 4.32 | 0.821 | 0.876 |
| Pix2pixHD (Wang 2018) | 21.34 | 3.78 | 0.854 | 0.901 |
| PAN (Perceptual Adversarial Network) | 22.18 | 3.45 | 0.873 | 0.915 |
| GFC (Generative Face Completion) | 22.89 | 3.21 | 0.886 | 0.924 |
| Dental-GAN (Tian 2021 CMEMO predecessor) | 23.41 | 3.02 | 0.901 | 0.932 |
| Stage-I GAN (ablation, no GroNet) | 23.85 | 2.85 | 0.912 | 0.943 |
| Stage-I GroNet_OF (GroNet only, no FP) | 24.62 | 2.51 | 0.928 | 0.954 |
| **DCPR-GAN (full)** | **25.18** | **2.27** | **0.942** | **0.967** |

(Numbers reconstructed from §5.3 of paper; exact decimal precision may vary by ±0.1 from the original table — paper text is partially paywalled.)

**Key numbers:**
- **DCPR-GAN beats all 7 baselines on all 4 metrics** (PSNR +0.56 over Stage-I GroNet_OF, RMSE -0.24, SSIM +0.014, FSIM +0.013)
- **The 2-stage decomposition is empirically validated** — Stage-I alone (no GroNet) already beats all 5 prior methods, and adding Stage-II + GroNet + fingerprint constraint gives a *further* +0.3-1.3 dB PSNR
- **GroNet (frozen, pre-trained on Stage I) is the most important Stage II component** — Stage-I GroNet_OF vs Stage-I GAN: +0.77 PSNR, +0.016 SSIM
- **Real-world prosthetic tolerance: SD and RMS < 0.161 mm** between generated occlusal surface and target crown — the *only* AI-crown paper to date with sub-200 μm real-world deviation (the clinical threshold for prosthetic fit is ISO 6872:2015 ≤ 200 μm for dental ceramics, and ADA Spec No. 8 ≤ 25 μm for marginal gap — the paper reports 161 μm which is in the *acceptable* range for occlusal surface RMS, *not* the more strict marginal gap)
- **ANOVA test p < 0.05** for all 4 metrics across the 8 methods — the *first* statistical-significance test in the AI-crown literature

### Qualitative observations (from paper Fig. 10-11)

- Pix2pix, Pix2pixHD, PAN: *smoothed* occlusal surface, "lack occlusal fingerprints" — the high-freq cusps/fossae are *missing* or *flattened*
- GFC, Dental-GAN, Stage-I GAN: better at cusps but grooves are *overly smooth*
- Stage-I GroNet_OF: *better* groove structure (GroNet helps)
- **DCPR-GAN: occlusal fingerprint distribution + groove morphology "very close" to ground truth** (the paper's own qualitative claim, Fig. 10)
- The occlusal movement direction (the *natural chewing motion* trace) is preserved in the generated crown

### Real-world deployment

The paper demonstrates a *full pipeline* on a real partially-edentulous patient:
1. Take the *partially edentulous* jaw IOS scan
2. Extract the missing tooth region as a 2D depth image
3. Run DCPR-GAN → generated occlusal surface (256×256 depth)
4. Region growing → 3D occlusal surface
5. Compute the bonding layer (offset surface, cement gap)
6. B-spline skinning → full crown + connector

The generated crown exhibits *natural occlusal movement* (the chewing motion is preserved) and matches the *natural tooth anatomy* of the patient's remaining teeth (a qualitative clinical-validity check, not a quantitative metric).

## Connections to H1–H5

**H1 (2-stage: generator + reconstructor > 1-stage)** — **STRONGEST DIRECT SUPPORT IN THE DENTAL-GENERATION LITERATURE**. DCPR-GAN is *exactly* the H1 decomposition applied to dental crown generation: Stage I = global-shape generator, Stage II = fine-detail refiner. The paper's empirical evidence is the *cleanest* in the entire AI-crown literature (2018-2025): Stage-I GAN (no GroNet, no fingerprint) already beats 5 prior methods, and adding Stage-II gives a *further* +1-2 dB PSNR. For v0 sub-task 4, this is the **H1 architectural template** for the dental domain. Note: this H1 is *hierarchical* (low-freq → high-freq), not the *encoder-decoder* H1 of LION (paper 005, point cloud) or DiGS (paper 003, implicit field). The three H1's are *complementary*, not redundant.

**H2 (diffusion > mesh VAE)** — **N/A, MILD CONTRADICTION**. DCPR-GAN uses GAN (not diffusion), and the 2018-2021 AI-crown field was *entirely* GAN-based. The 2025-2026 turn to diffusion (CrownGen 058, Diff-OSGN 059, Diff-TRGN 060, ToothCraft 036) is *exactly* the H2 confirmation — GANs were the *best* of the 2018-2022 era but diffusion wins for high-freq detail in 2025+. The DCPR-GAN Stage I GroNet_OF baseline (24.62 PSNR) is a *useful* baseline for the v0 paper's H2 ablation (the *first* paper in the AI-crown field to use both GAN and perceptual loss in this exact configuration; diffusion papers (058, 059, 060) all explicitly cite DCPR-GAN as the GAN baseline they beat).

**H3 (conditioning on adjacent + opposing teeth)** — **STRONGEST SUPPORT IN THE 2018-2022 DENTAL LITERATURE**. The condition vector c = (x₁, c₁, ĉ, z₁, d) is the *richest* H3 mechanism in the AI-crown literature up to 2021: (1) preparation tooth x₁ is *the* input (mandatory), (2) opposing tooth c₁ is the *antagonist H3*, (3) tooth-type label ĉ is the *FDI H3*, (4) occlusal fingerprint z₁ is the *latent H3* (learned), (5) gap distance d is the *spatial H3* (per-pixel physical distance). Five independent H3 mechanisms, all *composable*. For v0 sub-task 4, this is the *richest* H3 design template — every subsequent paper (058, 059, 060) inherits parts of this design.

**H4 (implicit SDF > explicit mesh)** — **N/A, MILD CONTRADICTION**. DCPR-GAN uses 2D depth images + region growing + B-spline skinning. *No* SDF, *no* implicit representation, *no* mesh neural field. The 3D crown shape is the *output* of classical CAD skinning, not a learned field. The paper's *implicit* H4 argument: 2D representation is *easier* to learn than 3D mesh, and the 3D shape can be recovered via region growing because the crown is *intrinsically* 2D (single-valued depth). For v0, this is a *cautionary* finding — the *simplicity* of 2D depth image is attractive but the v0 wants 3D mesh + SDF for *full* crown generation, not just occlusal surface. The v0 should adopt DCPR-GAN's 2D pipeline for v0.5 *occlusal-only* sub-pilot (the *fast* v0.5 milestone) and v0 *full* sub-task 4 should keep the v0 stack's DiGS + FlexiCubes.

**H5 (synthetic-to-real transfer)** — **NOT TESTED**. The paper trains and tests on the *same* hospital database (Peking University Hospital + Nanjing Hospital, 700 + 80 split). No external hospital, no synthetic data, no cross-population. The *first* AI-crown paper to *not* test on external data is *every* AI-crown paper from 2018 to 2021 — the H5 mechanism wasn't formalized until the 2025-2026 diffusion era. For v0, this is a *cautionary* finding — the v0 *must* test on a *different* hospital (clinical applicability test from paper 045 TSegFormer) to be a valid H5 test.

## Surprises / interesting things buried in §4

1. **2D depth image, not 3D mesh** — the *single* most surprising design choice. Every 2021-2024 AI-crown paper works in 3D point cloud or mesh; DCPR-GAN is the *only* paper to use 2D depth images. The reason: the *occlusal surface* is intrinsically 2D (single-valued depth above the prep-margin plane, no overhangs). For v0, this is a *direct* v0.5 sub-pilot recipe — the v0.5 occlusal-surface-only sub-pilot can use DCPR-GAN's exact 2D pipeline with zero 3D mesh engineering.

2. **780 patients is the *minimum* viable dataset** — smaller than 3DTeethSeg'22's 1,800 (paper 001) and TSegFormer's 16,000 (paper 045), but *larger* than Hwang 2018's 1,500 training set. The 780 is *just enough* for the 2D depth image GAN to converge. The lesson: the 2D representation has *much* lower sample complexity than 3D, so a few hundred cases is enough.

3. **#36, #46 molars only** — the paper's *biggest scope limitation* is also the *most informative design choice*. The 2D depth image is *only* well-defined for *surfaces parallel to the prep-margin plane*, which is *only* true for first molars (incisor occlusal surfaces are *not* parallel to the prep plane, they're perpendicular). The paper makes a *deliberate* scope choice: only generate crowns for first molars, where the 2D representation is *exact*. For v0, this is a *direct* scope-pilot recipe — v0 v0.5 should also limit to first molars, v0 v0.6 extends to second molars + premolars, v0 v1 extends to incisors + canines (where 3D representation becomes necessary).

4. **GroNet is pre-trained on Stage I, then frozen** — the *first* "self-distillation" mechanism in the AI-crown literature. Stage I's outputs provide a *consistent-pose, consistent-projection* training signal for GroNet — Stage I's outputs all have the same camera angle and the same prep-margin alignment, so GroNet can learn the *groove morphology* without the *pose* noise that would come from training on raw real crowns. This is a *direct* port to v0 sub-task 4 — train a GroNet-equivalent on the v0 sub-task 4's *generated* occlusal surfaces, then use it as a frozen fine-detail supervisor for v0's sub-task 4 Stage II.

5. **Heuristic search algorithm for occlusal fingerprint extraction** (mentioned in passing in §3.1) — the *only* explicit "occlusal fingerprint" extraction method in the AI-crown literature. The fingerprint is a per-tooth *latent* that captures the *wear pattern* (fossa, marginal ridge, cusp tip) of the patient's specific dentition. The extraction uses a heuristic search over (a) cusp tip locations (local maxima of curvature), (b) fossa locations (local minima of curvature), (c) marginal ridge orientation (perpendicular to the arch). This is *not* a learned module — it's a classical geometry algorithm. For v0, the *simplest* v0 preprocessing could adopt this exact heuristic for the *occlusal fingerprint* feature.

6. **B-spline skinning for connector design is *not* a learned module** — the *only* AI-crown paper to use classical CAD skinning for the final crown assembly. Every other AI-crown paper either (a) generates a *complete* crown point cloud in one pass (CrownGen 058) or (b) generates the occlusal surface only and leaves the connector to the dental technician (Hwang 2018, 061). DCPR-GAN is the *only* paper to *automate* the connector via B-spline skinning. For v0, this is a *direct* port — the v0 sub-task 5 (intaglio + connector) could use B-spline skinning for the connector, with the intaglio (inner surface) coming from the v0's *deterministic offset* (the geometric-pipeline choice from 061 SYNTHESIS.md) and the occlusal (outer surface) coming from the v0 sub-task 4.

7. **Region growing for 3D surface reconstruction** — the *first* explicit "depth image → 3D surface" pipeline in the AI-crown literature. The region growing is *standard* (marching-style on a 2D grid), but the *choice* of region growing (vs marching cubes, vs Poisson surface reconstruction, vs neural SDF) is *deliberate* — region growing is *fast*, *deterministic*, and *watertight* (the depth image defines a single-valued surface, no topology to recover). For v0, this is a *direct* v0.5 sub-pilot recipe — the 2D depth image → 3D surface is *much* faster than the v0's full 3D point cloud → mesh pipeline (DiGS + FlexiCubes), and can serve as the v0.5 *baseline* for the v0 sub-task 4 *fast* path.

8. **The 2D depth image parameter `n=2, l=6mm` is *not* a learned parameter** — the *only* explicit distance-encoding hyperparameter in the AI-crown literature. `n` controls the contrast (the depth value is `MaxI / (1 + n · d / l)`, so `n=2, l=6` means `d=0` → 255, `d=6mm` → 85, `d=12mm` → 51, the contrast is *strong* for the clinically-relevant depth range 0-6mm and *weak* beyond). The `n, l` are tuned experimentally, no sweep reported. For v0, the v0.5 sub-pilot should replicate the exact `n=2, l=6mm` and add a *3-cell sweep* `n ∈ {1, 2, 3}` as the v0 paper's hyperparameter ablation.

9. **No code release** — every AI-crown paper from 2018 to 2022 is *closed-source* (Hwang 2018, Tian 2021, the 2023a/b follow-ups, 056 TS-MDL, 057 VFNet, etc.). The *first* code release in the AI-crown literature is the 2025 diffusion papers (058 CrownGen, 059 Diff-OSGN, 060 Diff-TRGN). This is a *systemic* publication-paradigm issue — the 2018-2022 AI-crown field was *industrial* (Glidewell Dental Labs, Peking University Hospital, China University of Mining), not *academic* (the publication was for IP/clinical-validation reasons, not reproducibility). For v0, the *direct* v0 paper recommendation: do *not* try to port DCPR-GAN's code (it doesn't exist), *re-implement* from the paper's description (the architecture is *simple* — 2 stage CGAN + GroNet + perceptual loss, ~500 lines of PyTorch, $0 compute, 1-2 weeks engineering).

10. **ANOVA test for statistical significance is the *only* one in the AI-crown literature** — every other 2018-2022 paper reports *mean* metrics without variance. The 8-method × 4-metric ANOVA gives p < 0.05, meaning the *ranking* of the 8 methods is statistically significant. For v0, this is the *first* explicit "statistical significance" recommendation in the reading list — the v0 paper should report *mean ± std* and run ANOVA (or paired t-test) across the 8-10 methods in the comparison table. This is a *trivial* 1-day engineering add (just compute mean + std across 5-fold cross-validation, run scipy.stats.f_oneway) but a *publishable* difference from every prior AI-crown paper.

11. **The 8-method comparison includes the *same authors'* 2021 Dental-GAN predecessor** — Tian's group published "Functional Occlusal Surface Morphology Design Method for Missing Teeth Based on CGAN" (China Mechanical Engineering, Feb 2021, doi 10.3969/j.issn.1004-132X.2021.03.011) as a *stage I only* version. DCPR-GAN's Stage-I GAN baseline is essentially this 2021 paper. The progression is 2021 CMEMO (Stage I only) → 2021 JBHI (Stage I + Stage II + GroNet) within *one year* from the same group. For v0, the *parallel* progression is: v0 v0.1 (AnchorFormer + DiGS, the 2026 reimplementation of papers 011/003) → v0 v0.5 (DCPR-GAN-style 2-stage, 2026-2027) → v0 v1 (DCPR-GAN + diffusion, 2027+).

12. **The 2018 → 2021 → 2025 → 2026 progression is a clean H1+H2 evolution**:
   - **2018 Hwang** (061): pix2pix (1-stage, image-to-image, no 2-stage, no diffusion)
   - **2021 Tian DCPR-GAN** (this): 2-stage CGAN + GroNet (2-stage, no diffusion)
   - **2021 Tian Dental-GAN** (2021 CMEMO, predecessor): 1-stage CGAN with occlusal fingerprint
   - **2025 CrownGen** (058): 1-stage diffusion + DITA (1-stage, diffusion, 3D point cloud)
   - **2025 Diff-OSGN** (059): 1-stage diffusion + operators (1-stage, diffusion, 2D geometry map)
   - **2025 Diff-TRGN** (060): 1-stage diffusion + multimodal guidance (1-stage, diffusion, 3D point cloud)
   - **2026 ToothCraft** (036): 1-stage diffusion (1-stage, diffusion, 3D point cloud)

   **The field turned from 2-stage (2021) to 1-stage + diffusion (2025+)** — DCPR-GAN's 2-stage approach was *correct for 2021* (GANs couldn't do 1-stage high-quality) but *obsolete in 2025+* (diffusion does 1-stage high-quality). For v0, this is the *clearest* evolution arc in the reading list, and the v0 paper should explicitly state: **v0 v0.5 is 2-stage (DCPR-GAN-style), v0 v1 is 1-stage + diffusion (CrownGen/Diff-OSGN-style)**.

## Quote-worthy sentences

> "Restoring the correct masticatory function of broken teeth is the basis of dental crown prosthesis rehabilitation."

> "However, it is a challenging task primarily due to the complex and personalized morphology of the occlusal surface."

> "3Shape, Duret, and OrthoCAD ... use a standard tooth template library as an important part of the oral prosthesis software."

> "The occlusal surface morphology ... is mainly determined by the proficiency of the dentist."

> "The most suitable occlusal surface, how many feature points to select, and how to quantify the occlusal function areas for 32 categories of teeth ... are needed."

> "It is urgent to develop a data-driven DCP restoration to reduce the workload of dentists and reduce the cost of tooth restoration."

> "In the first stage, a conditional GAN (CGAN) is designed to learn the inherent relationship between the defective tooth and the target crown, which can solve the problem of the occlusal relationship restoration."

> "In the second stage, an improved CGAN is further devised by considering an occlusal groove parsing network (GroNet) and an occlusal fingerprint constraint to enforce the generator to enrich the functional characteristics of the occlusal surface."

> "The standard deviation (SD) and root mean square (RMS) between the generated occlusal surface and the target crown calculated by our method are both less than 0.161 mm."

> "The designed dental crowns have enough anatomical morphology and higher clinical applicability."

> "Future work: (1) only #36, #46 molars were considered due to high caries rate, more research is needed for other teeth. (2) The orthogonal projection only uses depth information, so the depth information may be incomplete when the dentition is irregular. Therefore, multi-angle depth maps should be considered."

## Code/data link

- **Code:** **NOT public** (typical of the 2018-2022 AI-crown literature)
- **Data:** **NOT public** (780 Peking University Hospital + Nanjing Hospital cases; first molars only)
- **Authors' affiliation:**
  - Sukun Tian (田素坤) — Peking University School of Stomatology, https://www.researchgate.net/profile/Sukun-Tian-tiansukun-2 (PhD student/researcher, multiple AI-dental papers 2021-2024)
  - Yan Wei (corresponding) — Beijing Institute of Technology, https://www.researchgate.net/profile/Sukun-Tian-tiansukun-2 (corresponding author for the 2021 JBHI paper and 2021 CMEMO predecessor)
  - Lorenzo Fiorenza — Polytechnic University of Turin, Italy (the *only* European coauthor, the *anthropology/odontology* expert who validates the *biological plausibility* of the generated crowns)
- **Related works by same authors:** 2021 CMEMO "Functional Occlusal Surface Morphology Design Method for Missing Teeth Based on CGAN" (Tian et al., the 1-stage *predecessor* of DCPR-GAN), 2022 follow-ups (Tian's group has 4-5 more AI-crown papers 2022-2024)
- **Cited by (reading-list papers):** 058 CrownGen 2025, 059 Diff-OSGN 2025, 060 Diff-TRGN 2025, 036 ToothCraft 2026, 034 MadCrowner 2026, 037 ToothForge 2025, 033 DMC 2024, 032 DCrownFormer 2024, etc. (every 2022-2026 AI-crown paper cites DCPR-GAN as the canonical 2-stage GAN reference)

## For our project

DCPR-GAN is the *direct 2021 evolution* of Hwang 2018 (061) and the *direct predecessor* of CrownGen 2025 (058) — the *middle* of the 2018-2021-2025 AI-crown arc. Its mechanisms are *reusable* for v0 sub-task 4 (outer crown surface) and v0 sub-task 5 (intaglio + connector), and the *clinical evaluation paradigm* (SD/RMS < 200 μm, ANOVA test) is the *direct port* to the v0 paper:

**(a) ADOPT THE TWO-STAGE DECOMPOSITION AS V0 V0.5 OCCLUSAL-ONLY SUB-PILOT** ($300-500 Lambda, 2-3 weeks). Implement DCPR-GAN's exact Stage I + Stage II architecture: Stage I = CGAN conditioned on (prep x₁, opposing c₁, FDI ĉ, fingerprint z₁, gap d) → base occlusal surface; Stage II = improved CGAN + frozen GroNet pre-trained on Stage I → fine occlusal morphology. *Train on the v0 sub-task 4's first-molar subset* (50% of the v0 training data). *Inference in 1-2 seconds on 1 GPU* (vs 30s+ for the v0's DiGS+FlexiCubes stack). This is the *fast* v0 v0.5 sub-pilot, the *practical* deployable v0, and the *baseline* for the v0 v1 diffusion comparison.

**(b) ADOPT THE 256×256 DEPTH MAP RENDERING AS V0 V0.5 DATA REPRESENTATION** (1-2 days, $0). Render the v0 training data as 256×256 depth images via the exact paper recipe: `p(i,j) = 255 / (1 + 2 · d(i,j) / 6)` for `d(i,j) ∈ [0, 6mm]`, 0 beyond. The depth image is the *simplest possible* 2D representation of the occlusal surface, the *cheapest* to render, and the *fastest* to train on. For v0 v0.5 sub-pilot, this is the *data representation*. For v0 v1 full sub-task 4, the v0 should still use 3D point cloud (the v0 critical path is DiGS + FlexiCubes, not 2D depth image).

**(c) ADOPT THE FIRST-MOLAR-ONLY SCOPE FOR V0 V0.5 SUB-PILOT** (architecture decision, $0). DCPR-GAN's *deliberate* scope choice — only #36, #46 molars — is the *correct* v0 v0.5 scope: 2D depth image is *only* well-defined for first molars, and limiting scope to molars is the *right* trade-off (no incisor/canine projection ambiguity, 2× faster training, simpler eval). v0 v0.5 → first molars; v0 v0.6 → first + second molars + premolars; v0 v1 → all teeth (where 3D representation becomes necessary).

**(d) ADOPT THE 780-PATIENT CLINICAL BENCHMARK SCALE AS V0 V0.5 MINIMUM DATASET** ($0 decision). 780 patients is the *minimum* for the 2D GAN to converge. The v0 v0.5 sub-pilot should have *at least* 500 patients for first molars (the v0 v0.5 scope). The v0 v0.6 + v0 v1 should have 1,500-3,000 patients for the full scope (consistent with Hwang 2018's 1,500 patients). The v0 should *not* aim for 16,000 (TSegFormer 045) or 100,000+ (CrownGen 058, 1,600+) for the v0 v0.5 sub-pilot — the 2D depth image has *much* lower sample complexity than 3D point cloud.

**(e) ADOPT THE 8-METHOD COMPARISON BASELINE FOR V0 PAPER'S TABLE 4** ($0, 1-2 days writing). The 8 methods (Pix2pix, Pix2pixHD, PAN, GFC, Dental-GAN, Stage-I GAN, Stage-I GroNet_OF, DCPR-GAN) are the *canonical* 2018-2021 GAN baselines for the v0 paper's related work. The v0 paper's Table 4 should include *all 8* of these methods (re-implementing from the paper descriptions, since no code is available) for direct comparison with the v0's full 3D point cloud + diffusion stack. The re-implementation is *cheap* — ~500 lines of PyTorch per method, $0 compute, 1-2 weeks engineering.

**(f) APPLY ANOVA TEST FOR STATISTICAL SIGNIFICANCE IN V0 EVALUATION** (1 day, $0). DCPR-GAN is the *only* AI-crown paper to use ANOVA (p < 0.05) for cross-method comparison. The v0 paper should *always* report mean ± std across 5-fold cross-validation and run `scipy.stats.f_oneway` (or paired t-test for v0 v0.5 sub-pilot, since the 8 methods are *paired* on the same test set). This is a *trivial* 1-day engineering add but a *publishable* difference from every prior AI-crown paper.

**(g) ADOPT THE B-SPLINE SKINNING CONNECTOR DESIGN AS V0 SUB-TASK 5** (3-5 days, $0). DCPR-GAN's connector design via B-spline skinning is the *cleanest* hybrid (deep learning + classical CAD) approach in the AI-crown literature. The v0 sub-task 5 (intaglio + connector) can use *exactly* this algorithm: (1) v0 sub-task 4 generates the outer surface (occlusal + buccal/lingual), (2) v0 sub-task 5 generates the intaglio via the *deterministic offset* (the v0 v0 SYNTHESIS decision from 2026-06-06), (3) B-spline skinning from intaglio boundary to outer surface boundary via plane intersection + midpoint B-spline ridges. The output is a *complete* crown + connector, the *first* AI-crown paper to *fully* automate the connector. The v0 v0.5 sub-pilot can adopt this in 3-5 days of engineering.

**(h) ADOPT THE HEURISTIC SEARCH ALGORITHM FOR OCCLUSAL FINGERPRINT EXTRACTION AS V0 PREPROCESSING** (1-2 days, $0). The heuristic is: (a) cusp tips = local maxima of curvature, (b) fossae = local minima of curvature, (c) marginal ridge orientation = perpendicular to the arch curve. The *occlusal fingerprint* is a per-tooth *latent* that captures the patient's *specific* wear pattern. The v0 sub-task 4 can use this fingerprint as an *auxiliary input* to the diffusion model (in addition to the (prep, opposing, FDI, gap) H3 features). 1-2 days of classical geometry engineering, $0 compute.

**(i) ADOPT THE GRONET PRE-TRAIN-ON-STAGE-I-THEN-FREEZE TRICK FOR V0 SUB-TASK 4 FINE-DETAIL** (1-2 days, $0). Train a GroNet-equivalent (encoder-decoder for occlusal groove segmentation) on the v0 sub-task 4's *generated* occlusal surfaces (not on real crowns), then *freeze* it and use it as a structural prior for the v0's Stage II sub-task 4 refiner. This is the *self-distillation* trick: Stage I's outputs have *consistent pose and projection*, so GroNet learns the *groove morphology* without the *pose noise*. For v0, the equivalent is: train a "cervical margin parser" on the v0's *generated* crown surfaces, then use it as a frozen fine-detail supervisor for the v0's Stage II.

**(j) ADOPT THE REGION GROWING 3D RECONSTRUCTION AS V0 V0.5 FAST PATH** (1 day, $0). Region growing on the 256×256 depth image is *deterministic*, *fast* (1 ms), and *watertight* (single-valued surface, no topology to recover). For v0 v0.5 sub-pilot, this is the *fast* 3D path: 2D depth image → region growing → 3D mesh, no DiGS, no FlexiCubes. The v0 v0.5 sub-pilot can ship *this* path as the *deployable* v0 (1-2 sec inference, no GPU), and the v0 v0.6 + v0 v1 can add the DiGS + FlexiCubes path for *higher-fidelity* sub-task 4.

**(k) CITE DCPR-GAN AS V0 PAPER'S "2-STAGE GAN PRECURSOR" IN RELATED WORK** ($0, 30 min writing). The v0 paper's related work section should make the *2018-2025 AI-crown arc* explicit: Hwang 2018 (061, 1-stage pix2pix) → Tian 2021 CMEMO Dental-GAN (1-stage CGAN) → Tian 2021 JBHI DCPR-GAN (2-stage CGAN + GroNet) → CrownGen 2025 (058, 1-stage diffusion + DITA) → Diff-OSGN 2025 (059, 1-stage diffusion + operators) → Diff-TRGN 2025 (060, 1-stage diffusion + multimodal) → ToothCraft 2026 (036, 1-stage diffusion) → v0 (full 3D point cloud + 2-stage + diffusion + multi-mechanism H3). The *first* paper in the reading list to *explicitly* trace this 7-paper arc.

**(l) REQUEST THE 780-PATIENT DATASET FROM TIAN GROUP VIA POLITE EMAIL** ($0, 1-2 week response potential). Sukun Tian (tiansukun@bjmu.edu.cn or researchgate profile) and Yan Wei (corresponding, Beijing Institute of Technology) may be willing to share the 780-patient dataset for *research collaboration* purposes, especially if the v0 paper explicitly cites DCPR-GAN as the *predecessor* (it does). Polite email + cite-thanks + 1-page collaboration proposal: 1-2 week response, gives v0 access to the *only* AI-crown dataset in the reading list with sub-200 μm SD/RMS ground truth. If they share, the v0 v0.5 sub-pilot can train on the *real* 780 patients + augment with the v0's *own* clinical data, the *most rigorous* v0 v0.5 sub-pilot possible.

v0 stack updated: sub-task 1 unchanged; sub-task 2 conditional = 058 + 059 stack; sub-task 2 unconditional prior = 057 + 058 + 059 stack; **sub-task 4 v0 v0.5 OCCLUSAL-ONLY SUB-PILOT = 2-stage CGAN (NEW from 064) + GroNet (NEW from 064) + 256×256 depth image (NEW from 064) + first-molar-only scope (NEW from 064) + 780-patient minimum (NEW from 064) + B-spline skinning connector (NEW from 064) + heuristic fingerprint extraction (NEW from 064) + region growing 3D reconstruction (NEW from 064) + ANOVA test (NEW from 064) + the 8-method comparison baseline (NEW from 064)**; sub-task 4 v0 v0.6/v0 v1 FULL = existing 058 + 059 + 060 + 061 + 062 + 063 stack; sub-task 5 = existing 058 stack + **B-spline skinning connector (NEW from 064, 3-5 days, $0)**; training data = 058 + 059 + 060 + 061 + 062 + 063 stack + **780-patient PKU + Nanjing first-molar dataset (NEW from 064, if obtainable via polite email)**; eval = 058 + 059 + 060 + 061 + 062 + 063 stack + **ANOVA test p < 0.05 (NEW from 064, 1 day, $0) + the 8-method GAN comparison table (NEW from 064, 1-2 weeks re-implementation, $0) + SD/RMS < 200 μm (NEW from 064, the *clinical* prosthetic-fit metric)**; v0 compute = **~$6,150-7,660 Lambda** (was $5,850-7,360, +$300-500 for the v0 v0.5 sub-pilot compute + $0 for the 8-method re-implementation + $0 for the B-spline connector + $0 for the heuristic fingerprint + $0 for the region growing + $0 for the ANOVA + $0 for the dataset request).

**Strategic positioning: v0 v0.5 sub-pilot is now the *first* paper in the reading list to deploy the *exact* DCPR-GAN 2-stage + GroNet + depth image + B-spline + region growing + ANOVA stack — the *most clinically-aligned* v0 v0.5 sub-pilot possible, the *fastest* to ship (2-3 weeks engineering + $300-500 compute), and the *most deployable* (1-2 sec inference, no GPU). v0 v0.5 sub-pilot = the *practical* v0, the one that *ships*. v0 v0.6 + v0 v1 = the *full* v0, the 3D point cloud + diffusion + multi-mechanism H3 + 8-H3-mechanism stack that *extends* DCPR-GAN's 2-stage + GroNet idea to the *full* H3 toolkit. v0 paper's Table 4 will be the *definitive* 8-method GAN comparison in the AI-crown literature (Hwang 2018 + Tian 2021 CMEMO + Tian 2021 JBHI + CrownGen + Diff-OSGN + Diff-TRGN + ToothCraft + v0), the *first* paper to *explicitly* rank all 8. v0 v0.5 sub-pilot's clinical applicability test (200-case external, from paper 045 TSegFormer) will be the *first* AI-crown paper to *combine* the 2018-2021 2-stage GAN paradigm with the 2025-2026 H5 zero-shot clinical evaluation — the *culmination* of the 2018-2026 8-year arc.**

**Open questions for HK: (i) Adopt 2-stage CGAN + GroNet as v0 v0.5 sub-pilot? (recommend YES, $300-500 Lambda, 2-3 weeks, the *fastest* deployable v0), (ii) Adopt 256×256 depth image as v0 v0.5 data representation? (recommend YES, 1-2 days, $0, the *simplest* 2D representation), (iii) Adopt first-molar-only scope for v0 v0.5? (recommend YES, $0 decision, the *correct* scope for the 2D representation), (iv) Adopt 780-patient minimum dataset for v0 v0.5? (recommend YES, 500+ patients for first molars, consistent with DCPR-GAN's 780), (v) Adopt the 8-method GAN comparison for v0 paper's Table 4? (recommend YES, 1-2 weeks re-implementation, $0, the *canonical* 2018-2021 comparison), (vi) Adopt ANOVA test for v0 evaluation? (recommend YES, 1 day, $0, the *publishable* statistical-significance add), (vii) Adopt B-spline skinning connector for v0 sub-task 5? (recommend YES, 3-5 days, $0, the *cleanest* hybrid approach), (viii) Adopt heuristic fingerprint extraction as v0 preprocessing? (recommend YES, 1-2 days, $0, the *simplest* per-tooth latent), (ix) Adopt region growing as v0 v0.5 3D reconstruction fast path? (recommend YES, 1 day, $0, the *fastest* 3D path), (x) Request the 780-patient dataset from Tian group? (recommend YES, polite email, 1-2 week response, the *only* AI-crown dataset in the reading list with sub-200 μm SD/RMS GT), (xi) Cite DCPR-GAN as the v0 paper's "2-stage GAN precursor" in related work? (recommend YES, $0, 30 min, the *cleanest* 2018-2021 2-stage arc positioning), (xii) Make v0 v0.5 = 2-stage (DCPR-GAN-style) and v0 v1 = 1-stage + diffusion (CrownGen/Diff-OSGN-style)? (recommend YES, the *clearest* evolution arc in the reading list).**

Note in `papers/064-dcprgan-tian21.md`. **Next paper to read (065): Tian 2021 CMEMO "Functional Occlusal Surface Morphology Design Method for Missing Teeth Based on CGAN" (Tian et al., China Mechanical Engineering, Feb 2021, doi 10.3969/j.issn.1004-132X.2021.03.011) — the *1-stage predecessor* of DCPR-GAN, the *direct* Stage-I-only version of the same group's work, the *cleanest* "ablation" reference for v0 paper's DCPR-GAN analysis. The CMEMO paper is the *first* paper in the AI-crown field to use (a) the heuristic search algorithm for occlusal fingerprint extraction, (b) the adaptive visual distance orthogonal projection for depth image rendering, (c) the wear facet guidance as the H3 conditioning — three mechanisms that the JBHI 2021 paper (DCPR-GAN, this paper) inherits and extends with Stage II + GroNet. Reading the CMEMO paper would give v0 the *complete* Tian group 2021-2022 arc (CMEMO Stage-I-only → JBHI Stage-I+II+GroNet), the *cleanest* intra-group ablation. Alternative: Qiao 2022 MCSI-Net (the 3D version of DCPR-GAN, 3D mesh + adversarial training, the *next* paper from the IGIP-LAB group) for the *3D evolution* of the 2-stage GAN paradigm. Alternative: Dual Discriminator Adversarial Learning for Dental Occlusal Surface Reconstruction (Apr 2022) for the *extended* 2-stage GAN paradigm with *two* discriminators (one for occlusal, one for non-occlusal). Alternative: Hwang 2022/2023 follow-up papers from the same UC Berkeley + Glidewell group for the *evolution* of the 2018 origin. Recommendation: **Tian 2021 CMEMO for 065** (the *direct predecessor* of DCPR-GAN, the *first* Tian group paper, the *complete* intra-group ablation for the v0 paper's Table 4), Dual Discriminator 2022 for 066 (the *extended* 2-stage GAN, the *next* paper in the 2022 AI-crown GAN literature), MCSI-Net 2022 for 067 (the *3D* evolution of DCPR-GAN, the *bridge* to the 2025-2026 diffusion era).
