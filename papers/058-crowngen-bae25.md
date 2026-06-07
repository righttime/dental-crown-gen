# Paper 058 — *CrownGen: Patient-customized Crown Generation via Point Diffusion Model*

**Title:** *CrownGen: Patient-customized Crown Generation via Point Diffusion Model*
**Authors:** Juyoung Bae (first, corresponding: 2cddb75f on arXiv), Yifan Lin, Hao Chen
**Affiliations:** HKUST (Hong Kong University of Science and Technology) — Department of Computer Science and Engineering (Bae, Chen); Division of Pediatric Dentistry and Orthodontics, Faculty of Dentistry, University of Hong Kong (Lin, the clinical co-author); Delun Dental Hospital Group, Guangzhou (the 30+ branch clinical data provider; in-house IRB via HAREC HREP-2024-0257)
- **Year:** 2025 (arXiv v1 26 Dec 2025; v2 1 Jan 2026)
- **Venue:** **arXiv preprint** (no peer-reviewed venue as of v2; not MICCAI/NeurIPS/CVPR — appears to be a tech report / preprint with the *full* evaluation protocol already executed, suggesting it is either in submission or being prepared for a clinical journal like MedIA, JDentiSci, or IEEE TMI)
- **arXiv:** [2512.21890](https://arxiv.org/abs/2512.21890) (cs.CV, 5.2 MB PDF, 32 pages, 11 figures, 10 tables + supplementary)
- **HTML:** [arxiv.org/html/2512.21890v2](https://arxiv.org/html/2512.21890v2)
- **Code:** ❌ **not yet released** as of v2 (Jan 2026) — paper has full architecture + hyperparameters but no GitHub link. The first author Juyoung Bae's email is exposed in the arXiv show-email link, suggesting this is a "code available upon reasonable request" preprint, consistent with the *private clinical data* (Source D + E) restrictions. Likely to be released once peer review is in progress.
- **Data:** Mixed — **public sources** (3DTeethSeg Wang 2024, 3DTeethLand Li 2024, ToothFairy1/2 Ben Hamadou) + **private clinical data** (Source D: 1,364 partially edentulous scans from 2022-2024, Source E: 26 cases from 23 patients for the reader study, 2025). Private data is accessible "for non-commercial academic use" via corresponding author, subject to MTA + IRB review.
- **Read:** 2026-06-08 08:03 KST (Monday, scholar hourly #58, ~45 min — full HTML v2, code-availability check, data-cohort audit, cross-references to the Lombaert-lineage 5 papers and DCrownFormer)

**⚠️ Note on the 057 recommendation:** paper 057 suggested either (a) the 6th Lombaert-lineage paper (Hosseinimanesh et al. 2024, *MedIA* "Personalized dental crown design: a point-to-mesh completion network") or (b) CrownGen. (a) is the **journal extension of the already-read paper 033 (DMC, MICCAI 2023)** — same group (Hosseinimanesh, Lombaert at Polytechnique Montréal), same architecture (point-to-mesh completion), same first author, *one* extra method (a refinement of the morphology-aware loss). Reading it would be ~80% redundant with paper 033. (b) CrownGen is a *new* paper, *new* group (HKUST + U-Hong-Kong Faculty of Dentistry + Delun Dental Hospital), *new* architecture (point diffusion + DITA + boundary prediction + DPSR mesh reconstruction), *new* paradigm (multi-crown-in-one-pass + abutment-agnostic). **Picking CrownGen — the highest-impact non-redundant read for the reading list**, *most relevant to the v0 paper's related work section* (Dec 2025 = most recent at time of writing), and *first clinical reader study with formal non-inferiority test* in the reading list.

---

## TL;DR

**CrownGen is the first AI framework to generate a *variable* number of patient-customized dental crowns in a single inference pass — and the first dental-3D-gen paper in our reading list with a formal *clinical non-inferiority trial* on real patient cases.** The architecture has three moving parts that *all* work together: **(1) tooth-level point-cloud representation** — every tooth (existing or missing) is its own 1024-point cloud with an 8-dim FDI embedding, decoupling the dentition from a monolithic point cloud; **(2) Distance-weighted Inter-Tooth Attention (DITA)** — a multi-head self-attention variant with an *index-based relative positional encoding* derived from a zig-zag FDI ordering (17, 47, 16, 46, ..., 11, 41, 21, 31, ..., 27, 37) so that the index difference `Δ_ij = i − j` is a proxy for anatomical distance, and *adjacent + antagonistic teeth* (small `|Δ_ij|`) get stronger morphogenetic influence; **(3) boundary prediction module + diffusion generative module** — a 5-parameter cylindrical bound (center, radius, height) for each missing tooth is predicted *first* to spatially constrain the diffusion, then a PointNet++ U-Net with PVC operators + DITA layers denoises the target point clouds conditioned on the context teeth. The two-stage *pseudo-crown self-bootstrapping* training is the unsung hero: train on 420 fully dentate scans first, then use the v1 model to synthesize pseudo-crowns for the *1364 partially edentulous* clinical scans, then retrain on the combined 1784-scan dataset. **The results are decisive and clinical.** On 496 external healthy test dentitions (26,288 test scenarios ranging from 1 to 6 missing teeth), CrownGen beats the three point-cloud-completion baselines (PointSea, AdaPoinTr, ProxyFormer) on **every** metric for **every** number-of-missing-teeth, and the gap *widens* as the number of restorations grows (AdaPoinTr's CD goes from 40.7 at 1 missing tooth to 59.5 at 4 missing teeth while CrownGen stays at 30.6-30.9 — a ~2× gap). Boundary prediction Dice = 0.883, IoU = 0.796 across 16,368 boundaries, with the second molars the worst (0.859 / 0.761) and the central incisors the best (0.897 / 0.817). The clinical reader study (n=26 cases, 23 patients, 2 readers, 4 criteria) is the most rigorous in the reading list: CrownGen-assisted workflow = 740 ± 131 s (vs 900 ± 180 s for fully manual; p<0.01, 17.78% faster), clinical acceptability = 95.2% (vs 94.2% for fully manual), composite quality = 2.938 (vs 2.928, p=0.425, no significant difference), and **all four criteria pass the pre-specified non-inferiority margin of -0.10 points at the 95% CI level**. Two trained clinical dentists with 4 and 5 years of experience, 14-day washout, cross-over design, Gwet's AC2 = 0.947 (excellent). **For our project: CrownGen is the *current 2025 SoTA point-diffusion patient-customized crown generator*, the most important non-Lombaert paper in the reading list, and the v0 paper's strongest 2025 reference.** The DITA mechanism is the *first successful H3 mechanism specifically designed for dental arches* (the zig-zag FDI ordering is a *clinically-motivated* inductive bias), and the boundary prediction module is the *first successful H5-style "context → where to put the tooth" decoupling*. The pseudo-crown self-bootstrapping is the *first successful training-data augmentation* for the "abundant partially-edentulous scans" problem. **All four of CrownGen's innovations are directly portable to v0 sub-task 2 and sub-task 4.**

## Research question + their answer

**Q:** Existing dental-crown generation models (DMC 033, DCrownFormer 032, MADCrowner 034, ToothCraft 036, ToothForge 037) are *all* single-crown generators. The single-crown architecture has *two* fundamental limitations: **(1)** the entire dental arch is treated as a monolithic geometric input, so the model is *trained* on a fixed input → fixed output, and *cannot* be re-used for a different "which teeth are missing?" configuration — "It is clinically infeasible to train new models for every possible configuration of missing teeth" (Sec 1); **(2)** the model is *implicitly dependent* on a prepared abutment tooth whose distinct geometry acts as the *localization signal* for the crown, so the model "is incapable of generating a clinically acceptable result" in the implant-supported restoration case where there is *no* prepared abutment. The research question is: **can we build a *single* crown-generation model that (i) accepts *any* subset of existing teeth as input, (ii) generates *any* number of prostheses required, and (iii) is agnostic to the presence/absence of abutment preparations?**

**A:** **Yes — by decomposing the dentition into a *constellation of tooth objects* (each tooth is its own point cloud with FDI identity), and learning inter-tooth relationships via a *distance-weighted inter-tooth attention* (DITA) layer that uses the FDI index difference as a relative positional encoding.** The model has three architectural stages, each solving one of the three problems: **(a) tooth segmentation** (which they treat as a "robustly achievable" pre-step using ToothFairy, TSegFormer, DArch — paper 049's class-aware point transformer or paper 044's MeshSegNet) decouples the *where are the teeth* problem from the *what do the teeth look like* problem, enabling the constellation representation; **(b) boundary prediction** (a 5-parameter cylinder `(c_x, c_y, c_z, r, h)` per target tooth) decouples *localization* (where in 3D space should each crown go) from *shape generation* (what does each crown's surface look like), and gives the diffusion model a *spatial prior* to constrain the stochastic denoising within a high-probability region; **(c) point-diffusion generation** (a PointNet++ U-Net with PVC operators + DITA layers) synthesizes the *detailed morphology* of each crown, with the DITA layers explicitly modeling "morphogenetic signals" from adjacent and antagonistic teeth via the index-based RPE. The diffusion's stochasticity is *not* wasted on global placement (that's the boundary module's job) — the diffusion's full representational power is allocated to resolving *fine-grained anatomical details* (cusps, fossae, marginal ridges). The pseudo-crown self-bootstrapping solves a *training-data* problem (most clinical scans are partially edentulous; only 420 of 1784 scans are fully dentate), letting the model use 1364 additional partially-edentulous scans by *synthesizing* plausible crowns for the missing teeth. The two-protocol evaluation is the *strongest in the reading list*: Protocol 1 (496 external scans, 3 baselines) tests *geometric fidelity* on a *held-out dataset* (Sources A + B) that the model was *not* trained on; Protocol 2 (26 cases, 23 patients, 2 blinded readers) tests *clinical non-inferiority* in a *real clinical workflow*. **The clinical non-inferiority result is the most important claim**: CrownGen-assisted crowns are *not* inferior to fully-manual expert crowns at the 5%-margin level, *and* they're produced 17.78% faster (740s vs 900s per case). The full CrownGen-assisted workflow is 6.7 hours faster per 100 cases (a 100-case workload drops from 25 hours to 20.6 hours, or 4.4 hours saved — meaningful at the scale of a dental lab with 1000s of cases per year).

The result is also *empirically novel* in the 2025 dental-3D-gen landscape: before CrownGen, *no* AI method had passed a formal non-inferiority clinical trial for crown generation, and *no* AI method had demonstrated multi-crown-in-one-pass generation. CrownGen is the *first* on both counts.

## Method

### Architecture overview (Fig 7)

```
Input: (Context set Y) ∪ (Target set X, k=1..6 missing teeth)
       Each tooth: 1024-point cloud + binary indicator (0=context, 1=target) + 8-dim FDI embedding
                        │
                        ▼
       ┌─────────────────────────────────────┐
       │  Stage 1: Boundary Prediction        │  ← Pre-processes X (no diffusion, no noise)
       │  Encoder: PointNet++ + PVC + DITA    │
       │  Output: B = {(c_x, c_y, c_z, r, h)_i} for i=1..k
       │  Loss: smooth-L1 on (B_pred - B_gt)
       └─────────────────────────────────────┘
                        │
                        ▼ (B is the spatial prior)
       ┌─────────────────────────────────────┐
       │  Stage 2: Point Diffusion (DDPM)     │  ← The main work
       │  Backbone: PointNet++ U-Net + PVC    │  ← 4 SA blocks + 4 FP blocks (U-shaped)
       │  + DITA layers (after each SA/FP)    │  ← The H3 mechanism
       │  Forward: add Gaussian noise over T=1000 steps, β_min=1e-4, β_max=2e-2
       │  Reverse: ε_θ predicts noise, conditioned on Y, B, t
       │  Loss: MSE(ε - ε_θ(√ᾱ_t·X_0 + √(1-ᾱ_t)·ε, Y, B, t)) on target teeth only
       └─────────────────────────────────────┘
                        │
                        ▼
       ┌─────────────────────────────────────┐
       │  Stage 3: Differentiable Poisson      │  ← Point cloud → watertight mesh
       │  Surface Reconstruction (DPSR)        │
       │  Predicted per-point normals +        │
       │  differentiable Poisson solver +      │
       │  Marching Cubes (64³ grid)            │
       └─────────────────────────────────────┘
                        │
                        ▼
       Output: 1..6 watertight crown meshes (STL/OBJ-ready)
```

### DITA (Distance-weighted Inter-Tooth Attention) — the H3 mechanism

The DITA layer is the *novel* component that makes multi-crown generation work. Input: `Z_in = [z_1, ..., z_K] ∈ ℝ^(K×C)` where each `z_i` is the per-tooth feature descriptor (after an SA or FP block) and `K = |X| + |Y|` (target + context). For each tooth pair `(i, j)`, the *index difference* `Δ_ij = i - j` in a fixed zig-zag FDI ordering is a *proxy* for anatomical distance:

```
FDI order: 17, 47, 16, 46, ..., 11, 41, 21, 31, ..., 27, 37
            ↑    ↑    ↑    ↑         ↑    ↑    ↑    ↑         ↑    ↑
         P1   M1   P1   M1       C/I  C/I  I/C  I/C      P2   M2
```

This zig-zag order means:
- Adjacent teeth in the *same arch* (e.g., 16 and 17) have `|Δ_ij| = 1` (small)
- Antagonistic teeth (e.g., 16 and 46) have `|Δ_ij| = 1` (small, *across* the zig-zag)
- Teeth on the *opposite side* of the arch (e.g., 16 and 27) have `|Δ_ij| ≈ 10` (large)

So `|Δ_ij|` is a single scalar that captures *both* within-arch-adjacency and antagonistic-pair relationships. The relative positional encoding is a 3-dim vector:
```
r_ij = [log(1 + max(Δ_ij, 0)),     ← positive direction
         log(1 + max(-Δ_ij, 0)),     ← negative direction
         1_{Δ_ij = 0}]              ← self-relation flag
```
A 2-layer MLP projects `r_ij` into three learnable RPE bias vectors `{p^Q_ij, p^K_ij, p^V_ij}`. The attention score is then:
```
e_ij = (1/√F) · q_i^⊤ · k_j
     + q_i^⊤ · p^K_ij                ← query-to-key bias
     + p^Q_ij^⊤ · k_j                ← key-from-query bias
α_ij = exp(e_ij) / Σ_k exp(e_ik)     ← softmax
z_i^out = z_i^in + Σ_j α_ij (v_j + p^V_ij)  ← residual
```

This is the standard *Transformer-XL / T5-style* RPE pattern (Shaw 2018, Harvey 2022), but applied to *teeth* with a *clinically-motivated* ordering. The key insight: **the FDI ordering is *not* arbitrary** — it was *designed by dentists* to be a *spatial* ordering (zig-zag across the arch = small index difference = close in 3D), so the RPE learns the *right* prior (close teeth → strong influence) *for free*. **For our project: this is the *first* H3 mechanism in the reading list that is *provably correct* by clinical-anatomy construction, not just learned from data.**

### Boundary prediction module (Stage 1)

- **Backbone:** 3 SA blocks with PVC + DITA (the same encoder as the generative module, no decoder)
- **Input:** 512 points per tooth, target teeth's points zeroed out (so the network sees the *absence*)
- **Output:** `B_pred ∈ ℝ^(|X| × 5)` — 5 parameters per target tooth: `(c_x, c_y, c_z, r, h)` of a cylinder
- **Loss:** `smooth-L1(B_pred - B_gt)` — robust to outliers
- **GT cylinder:** fit a minimal enclosing circle to the tooth's XY projection, use the min/max Z to define height
- **Training:** 1000 epochs, Adam, lr 3e-4 → 3e-6 cosine, dropout 0.3

### Generative module (Stage 2)

- **Backbone:** U-shaped PointNet++ with 4 SA + 4 FP blocks, DITA after *each* block and at the bottleneck
- **Point-Voxel Convolution (PVC):** replaces each PointNet substructure with PVC operators (PVD's innovation, paper 012)
- **Input to U-Net:** concatenated point cloud `[Y, X_t]` where `X_t` is the noisy target at diffusion step `t`
- **Conditioning:** context `Y`, boundary `B`, and diffusion timestep `t` are all injected via the DITA layers
- **Per-point features:** `[x, y, z, FDI_embedding_8d, indicator_bit]`
- **Diffusion:** T=1000 steps, linear β schedule from 1e-4 to 2e-2, DDPM (not DDIM)
- **Loss:** MSE on noise prediction, computed *only* on target teeth (mask zeros the context teeth's loss)
- **Training (two-stage pseudo-crown self-bootstrapping):**
  - Stage 1: 3000 epochs on 420 fully-dentate scans, lr 4e-5, decay 0.4 at epoch 1500
  - Stage 2: use Stage 1 model to generate pseudo-crowns for 1364 partially-edentulous scans, fine-tune on the combined 1784-scan dataset for 2400 epochs, lr 2e-5, decay 0.45 every 800 epochs
  - **Why two-stage:** the partially-edentulous scans have *no* ground-truth for the missing teeth, so a single-stage model would have to learn from the 420 fully-dentate scans *only* — only 24% of the available data. The two-stage trick recovers the other 76% by treating the v1 model's predictions as *pseudo-ground-truth*. The ablation shows this is the *most impactful* single contribution (see Results).

### DPSR mesh reconstruction (Stage 3)

- **DPSR (Differentiable Poisson Surface Reconstruction, Peng et al. 2021):** predicts per-point normals (an MLP head), then a *differentiable* Poisson solver computes a 3D indicator function on a 64³ grid
- **Marching Cubes** extracts the iso-surface at indicator=0.5 → watertight mesh
- **Trained independently** of the diffusion module, on the same 1784-scan dataset, with MSE loss on the indicator grid
- **Output:** ready-for-CAD-CAM mesh (STL/OBJ), directly importable into ExoCAD/CEREC/3Shape for margin-line adaptation

### Data (4.1, Fig 7d)

| Source | Type | n scans | Use |
|---|---|---|---|
| A (Wang 2024) | Public, post-orthodontic healthy | ~ | **Test** (496-scan external evaluation set) |
| B (Li 2024) | Public, healthy | ~ | **Test** (combined with A for 496 scans) |
| C (ToothFairy 1+2) | Public, mixed (some with bite) | ~ | **Train** (in 1784-scan development set) |
| D (Delun Dental Hospital) | Private, 2022-2024, partially edentulous | 1364 | **Train** (in 1784-scan development set) |
| E (Delun Dental Hospital) | Private, 2025, real cases with technician crowns | 26 (23 patients) | **Reader study** only |

- **Pre-processing:** bite registration (midline-canine-molar MCM relationship for Source C), tooth segmentation (Point Transformer trained on Source C, applied to D and E), manual QC by trained dentists, Meshmixer for segmentation corrections, Blender for coordinate-system alignment (maxillary central incisor center at origin, facial direction = -Y, sagittal plane = YZ, occlusal plane = XY)
- **Curation rules:** exclude scans with ≥6 missing/anomalous teeth, exclude scans with severe malocclusion/crossbite/open bite, *digitally reposition* minor-moderate misalignment to ideal arch form using Wei 2020 TANet simulation
- **Final development cohort:** 1784 scans (430 fully dentate + 1364 partially edentulous), 10 fully + 20 partially held out for validation
- **Test cohort:** 496 healthy post-orthodontic scans (curated to *only* complete 28-tooth dentitions)

### Comparative baselines (4.5.1)

The baselines are *not* other dental-3D-gen models — those have a *fundamental architectural mismatch* (single-crown completion). Instead, CrownGen compares to three *general-purpose* point cloud completion networks from the computer vision domain:

1. **PointSea** (Yu et al. 2023) — multi-view depth projections + structure-detail disentanglement
2. **AdaPoinTr** (Yu et al. 2023) — Transformer with adaptive query generation + geometry-aware denoising
3. **ProxyFormer** (Li et al. 2023) — proxy-point partitioning + specialized Transformer for missing/existing interaction

To adapt these single-object models to multi-tooth generation, the authors train **6 separate models per baseline** (one per "k missing teeth" value from 1 to 6), so each baseline is *scenario-optimized*. CrownGen, by contrast, is a *single model* that handles all 1-6 missing teeth cases. **This is a strong but fair benchmark design** — the baselines get to use 6× more model instances (6× more parameters total, 6× the training compute) while CrownGen uses 1×.

## Results

### Boundary prediction accuracy (Table 1, 16368 boundaries from 496 test dentitions)

| Tooth type | n | Dice | IoU |
|---|---|---|---|
| All types | 16368 | 0.883 ± 0.041 | 0.796 ± 0.061 |
| Central incisor | 2347 | 0.897 ± 0.032 | 0.817 ± 0.051 |
| Lateral incisor | 2277 | 0.886 ± 0.039 | 0.801 ± 0.059 |
| Canine | 2392 | 0.883 ± 0.038 | 0.795 ± 0.056 |
| First premolar | 2335 | 0.885 ± 0.040 | 0.799 ± 0.057 |
| Second premolar | 2333 | 0.879 ± 0.043 | 0.789 ± 0.059 |
| First molar | 2355 | 0.891 ± 0.036 | 0.808 ± 0.057 |
| **Second molar** | 2329 | **0.859 ± 0.048** | **0.761 ± 0.069** |

**Notes:** Second molars are the *hardest* (anatomically variable — third-molar-replaced 28-tooth dentitions have second molars in unusual positions), central incisors the *easiest* (geometrically constrained by the MCM relationship). The 0.883 mean Dice is high enough to give the diffusion a *reliable* spatial prior (Dice > 0.85 → "the cylinder overlaps the true tooth in 85% of its volume").

### Geometric fidelity vs SoTA (Table 3, point cloud level)

The full Table 3 is enormous (1, 2, 3, 4, 5, 6 missing teeth × 7 methods × 5 metrics). The *most important* row is **3 missing teeth** (the most common multi-tooth clinical case):

| Method | CD (×10³) ↓ | EMD (×10³) ↓ | F1@0.3mm ↑ | F1@0.5mm ↑ | F1@1.0mm ↑ |
|---|---|---|---|---|---|
| **CrownGen** | **30.900** | **51.236** | **0.544** | **0.842** | **0.988** |
| w/o boundary cond. | 41.727 | 65.018 | 0.381 | 0.677 | 0.950 |
| w/o DITA | 38.992 | 77.108 | 0.409 | 0.715 | 0.962 |
| w/o data expansion | 45.973 | 82.833 | 0.329 | 0.611 | 0.927 |
| PointSea | 44.742 | 103.717 | 0.352 | 0.660 | 0.935 |
| AdaPoinTr | 54.501 | 129.072 | 0.270 | 0.517 | 0.872 |
| ProxyFormer | 46.920 | 98.465 | 0.332 | 0.616 | 0.923 |

**Key observations:**
- CrownGen beats all 3 baselines on *every* metric for *every* k=1..6 missing teeth (all p<0.01, paired t-test)
- The gap *widens* with k: at k=1, PointSea is within 5% of CrownGen (CD 32.4 vs 30.6); at k=4, PointSea is 38% worse (CD 42.1 vs 30.6). The baselines *collapse* on multi-tooth scenarios
- Ablations show the *biggest* gain is from data expansion (29% CD improvement), then boundary condition (27%), then DITA (17%)
- CrownGen's CD *stays flat* at 30.5-30.9 across k=1..6, while AdaPoinTr's CD *grows* from 40.7 to 59.5. The framework's difficulty does *not* scale with k (a counter-intuitive and powerful result)

### Ablation on reconstructed mesh quality (Fig 3, ASD + NC)

The full CrownGen model improves Average Surface Distance (ASD) over ablations by:
- 35.04% vs no-boundary-module
- 19.82% vs no-DITA
- 39.04% vs no-data-expansion

The data expansion is the *biggest* contributor (39%), then boundary (35%), then DITA (20%) — same ranking as the point cloud results. The Normal Consistency (NC) is more modest (1-5% gains), suggesting the *shape* is right, the *surface accuracy* is where the gain lies.

### Clinical efficiency (Fig 5c, n=26 cases)

| Workflow | Mean time (s) | 95% CI | % reduction |
|---|---|---|---|
| Fully manual CAD (3 expert technicians) | 900 ± 180 | (825, 974) | — |
| **CrownGen-assisted CAD** (1 entry-level tech + AI post-processing) | **740 ± 131** | (686, 794) | **17.78% (p<0.01)** |

CrownGen is *not* dramatically faster (17.78% is a 2.7-minute savings per case), but the *clinical significance* is that the CrownGen-assisted workflow uses an *entry-level* technician (0.5 years CAD experience) vs the fully-manual workflow's *expert* technicians (5-10+ years). The *labor cost reduction* is much larger than the *time* reduction: an entry-level tech at half the salary of an expert tech + AI assistance = ~50% labor cost reduction per case.

### Clinical quality (Fig 5a, 5b, 5d, 208 ratings per workflow)

| Criterion | CrownGen-assisted (95% CI) | Fully manual (95% CI) | p (paired t) | NI margin passed? |
|---|---|---|---|---|
| Occlusion | 2.885 (2.789-2.981) | 2.942 (2.870-2.991) | 0.161 | ✅ |
| Proximal contact | 2.942 (2.870-0.996) | 2.923 (2.842-0.980) | 0.327 | ✅ |
| Alignment with arch form | 3.000 (3.000-3.000) | 3.000 (3.000-3.000) | — | ✅ (zero-width CI) |
| Crown form and contour | 2.923 (2.841-2.985) | 2.846 (2.746-2.932) | 0.327 | ✅ |
| **Composite** | **2.938 (2.851-2.995)** | **2.928 (2.841-2.990)** | **0.425** | **✅** |

**Clinical acceptability (Likert 3 = "acceptable without modification"):** CrownGen 95.2% (198/208) vs fully manual 94.2% (196/208). Both > 90% — both workflows are clinically viable.

**Non-inferiority:** Pre-specified margin of -0.10 points on the 3-point scale = 5%. All four criteria + composite pass the NI margin at the 95% CI level (case-level bootstrap + parametric t-test sensitivity). **CrownGen is statistically non-inferior to fully manual expert workflow.**

### Inter-rater reliability (Fig 6)

- **Gwet's AC2 = 0.947** overall (excellent)
- **Brennan-Prediger = 0.856** (substantial)
- **Kendall's W = 0.474** (moderate — but ordinal-3 has limited dynamic range)
- **Overall percent agreement = 89.4%**

The 2 readers agree on ~9/10 ratings — the subjective quality assessment is *reliable* enough for the NI conclusion to be trusted.

### Inference time

- **~85 seconds per pass on NVIDIA RTX 4090** (Sec 3, Discussion)
- Constant regardless of k (1-6 missing teeth) — the *biggest* practical advantage over manual: a 6-crown case is *not* 6× slower than a 1-crown case for the AI
- Comparison: manual workflow scales linearly (k=6 → ~90 min if k=1 is 15 min)

## Connections to H1-H5

### H1: 2-stage (segmentation + generation) > end-to-end
- **STRONGLY SUPPORTS** — CrownGen's two stages are *explicitly* (1) tooth segmentation (a pre-step, off-the-shelf from ToothFairy/TSegFormer/DArch) and (2) crown generation. The boundary prediction *within* the generation stage is a *third* sub-stage (localization before shape). The paper's claim: "CrownGen's workflow presupposes an initial tooth segmentation step. We view this as a pragmatic design choice rather than a barrier" (Sec 3, Discussion). The ablation result that the boundary module contributes 27% CD improvement is direct evidence that *decoupling* localization from shape is the right H1 architecture.
- For our v0: the H1 design (segmentation → boundary → diffusion) is the *exact* stack the v0 paper should adopt, in *exactly* the order CrownGen does. The crown-gen model is *not* expected to *also* detect which teeth are missing (that's segmentation's job), and the localization (where in 3D space) is *not* expected to be learned *implicitly* in the diffusion (that's the boundary module's job). **CrownGen is the strongest H1 evidence in the 2025 dental-3D-gen landscape.**

### H2: Diffusion on point clouds > mesh-based VAE
- **SUPPORTS, BUT WITH CAVEATS** — CrownGen is a *point-cloud diffusion* model, not a mesh-based VAE. Compared to the *mesh-based* Lombaert-lineage (DMC 033 = point-to-mesh completion, DCrownFormer 032 = point-to-mesh transformer, MADCrowner 034 = point-to-mesh VAE), CrownGen's point-diffusion approach *wins* on multi-crown scenarios (the Lombaert-lineage is *architecturally* unable to handle multi-crown, per Sec 1). Compared to the *point-based* VAE (VF-Net 057, LION 005, PVD 012), CrownGen's diffusion *wins* on multi-crown scenarios but the *sampling inference cost* is 85s/pass (vs VF-Net's <1s/pass). The trade-off: diffusion has *better* multi-object modeling + *better* data-efficiency, but *slower* inference.
- For our v0: H2 is *qualified*. For the *unconditional* tooth-shape prior (rare-FDI-class data augmentation, paper 057's recommendation), VF-Net's <1s inference is *much* better than CrownGen's 85s (you want to generate 10K synthetic samples for the H5 ablation). For the *patient-customized* crown generation (the v0 paper's headline task), CrownGen's 85s is *acceptable* (vs 900s manual, 10× speedup) and the multi-crown capability is *essential* for clinical viability. **Use VF-Net for the unconditional prior, CrownGen (or its design) for the patient-customized conditional generator.**

### H3: Conditioning on opposing + adjacent teeth
- **STRONGLY SUPPORTS, AND ADDS THE FIRST CLINICALLY-MOTIVATED ATTENTION MECHANISM** — DITA is the *first* H3 mechanism in the reading list that uses a *clinically-motivated* relative positional encoding (the zig-zag FDI ordering). Compared to the *learned* H3 mechanisms (AnchorFormer 011's per-instance anchors, SeedFormer 010's regional positional encoding, LION 005's latent-anchor attention, Lombaert-lineage 033-037's per-tooth-type feature aggregation), DITA's index-based RPE has a *much stronger* inductive bias: the FDI ordering is *designed by dentists to be spatial*, so the RPE learns the *right* prior *for free*. The ablation shows DITA contributes 17% CD improvement and 30% EMD improvement — the EMD gain is *uniquely attributable* to DITA (it's the most global-shape metric, and DITA's RPE is the most "global" of the H3 mechanisms). The paper's own interpretation: "The ablation of inter-tooth attention layers was uniquely detrimental to the EMD metric, which measures global geometric correspondence. The pronounced impact on a globally sensitive, mass-preserving metric like EMD validates our assumption that explicit inter-tooth attention is crucial for capturing these global morphogenetic signals from adjacent and antagonistic teeth" (Sec 2.2.3).
- For our v0: DITA is the *primary* H3 mechanism to adopt. The zig-zag FDI ordering is a 5-line implementation (just a hard-coded list of 28 FDI numbers), the RPE is a 2-layer MLP, the multi-head attention is standard. Engineering cost: 2-3 days to integrate into the v0 sub-task 2 conditional generator (MADCrowner or ToothCraft), $0 compute, expected +5-15% CD improvement based on the ablation. **CrownGen's DITA is the v0 paper's most directly-portable architectural innovation.**

### H4: Implicit SDF > explicit mesh
- **NEUTRAL / BYPASSES** — CrownGen uses *point clouds + DPSR*, not SDF. The DPSR-based mesh reconstruction is *implicit* (indicator function on a grid) but the *generation* is *explicit* (1024 points per tooth, decoded to coordinates). This is *different* from DiGS 003 (SDF field), Diffusion-SDF 004 (latent SDF), or LION 005 (latent point). The DPSR's role is *post-processing* (point cloud → watertight mesh), not the *primary representation*. The H4 question is *not* directly tested by CrownGen.
- For our v0: H4 is *still open* (DiGS 003 is the strongest supporter). CrownGen's DPSR-based post-processing is *compatible* with v0's H4 design — the diffusion generates points, the DPSR converts to mesh, *no* SDF involved. The v0 paper should *not* claim H4 is settled by CrownGen, but should *cite* CrownGen as evidence that *point-based* methods can reach clinical non-inferiority (i.e., the *substrate* doesn't have to be SDF to win clinically).

### H5: Synthetic data bootstraps training
- **STRONGLY SUPPORTS, AND ADDS A NEW VARIANT** — CrownGen's pseudo-crown self-bootstrapping is the *first* H5 application in the reading list that uses *the model's own outputs* to bootstrap training (not a *separate* synthetic-data generator). Compared to TeethGenerator 051 (separate CAD-library-based generator, +0.05% improvement), CrownGen's self-bootstrapping contributes *29% CD improvement* (the largest single contributor in the ablation). The mechanism: train v1 on the small (420-scan) fully-dentate dataset, use v1 to generate pseudo-crowns for the large (1364-scan) partially-edentulous dataset, train v2 on the combined (1784-scan) "fully dentate" dataset. The paper's caveat: "While these pseudo crowns possess a lower geometric fidelity than those from the fully equipped variant of CrownGen, they still serve as robust anatomical placeholders. In these augmented scans, the contextual learning signal is overwhelmingly dominated by the numerous high-fidelity natural teeth, making the training process highly robust to the finer-grained inaccuracies of the few synthesized crowns" (Sec 2.2.3). This is *exactly* the "data efficiency via weak labels" pattern from semi-supervised learning, applied to 3D shape generation. **For our v0: the pseudo-crown self-bootstrapping is *the* most publishable H5 ablation in the v0 paper.** The setup is identical to v0's H5 (use VF-Net's synthetic samples to augment the training set), but with a *twist*: instead of using VF-Net's *unconditional* prior, use CrownGen's *conditional* prior. The "synthetic sample" can be either an *unconditional* tooth shape (VF-Net) or a *conditional* crown (CrownGen) — testing *both* is a 2×2 ablation that would be a top-conference paper.

## Surprises / interesting things buried in section 4

1. **The two-protocol evaluation is the strongest in the reading list** (Sec 4.5-4.6). Protocol 1 is *external* (Sources A+B, 496 scans, *held out* from training) — most papers in the reading list test on *internal* test sets. Protocol 2 is a *clinical reader study* with 2 blinded readers, 4 criteria, 14-day washout, cross-over design, Gwet's AC2 = 0.947, *and* a *pre-specified non-inferiority margin* with case-level bootstrap + parametric t-test sensitivity. **This is the *only* paper in the reading list with a formal clinical trial design.** For the v0 paper's clinical claims, CrownGen's evaluation protocol is the *gold standard* to cite.

2. **The 6-baseline × 6-k design is a *strong* but *fair* benchmark** (Sec 4.5.1). The baselines get to use 6 separate model instances (one per "k missing teeth"), each optimized for that specific k. CrownGen uses 1 unified model. The 6× parameter + 6× compute advantage of the baselines is *not* a bug — it's the *right* comparison, because in practice, the *clinically deployed* model must be 1 unified model (a clinic with 1 model per "k" is operationally infeasible). The paper makes this explicit: "In stark contrast, a single, unified CrownGen model was tasked with handling all scenarios without modification or retraining." This is the *first* paper in the reading list to make the *clinical-deployment* argument for unified model design.

3. **The 85-second inference time is constant across k** (Sec 3, Discussion). A 1-crown case takes 85s, a 6-crown case takes 85s. The diffusion's per-step cost is *linear in K = |X| + |Y|*, but the *number of steps is constant* (1000), so the total cost is *linear in K* and *constant in k* (since K = 28 for both 1-crown and 6-crown cases — you have all 28 teeth in the arch either way). This is a *huge* practical advantage: a 6-crown case takes the same AI time as a 1-crown case. Manual workflow scales linearly (6× for 6 crowns).

4. **The boundary prediction Dice of 0.883 is *exactly* at the threshold where the diffusion's "spatial prior" is reliable** (Sec 2.2.1, Table 1). The ablation shows the boundary module contributes 27% CD improvement — but the *absolute* Dice of 0.883 means the predicted cylinder *overlaps* the true tooth in 88.3% of its volume. The 0.117 "missing" overlap corresponds to the cylinder being *slightly too small* or *slightly off-center* for ~12% of cases. **For our v0, this suggests the boundary prediction module needs *human-in-the-loop correction* for ~12% of cases** — the dentist reviews the predicted boundary and adjusts if needed. The CrownGen-assisted workflow already includes this human-in-the-loop step (the entry-level technician does the post-processing).

5. **The clinical reader study's 14-day washout is the *minimum* for bias-free ordinal rating** (Sec 4.6.2, cited as "established precedents in prosthodontic literature"). The cross-over design (each restoration rated twice, once per workflow, with 14 days between ratings) is the standard in clinical trials. The Gwet's AC2 = 0.947 inter-rater agreement is *excellent* (0.8+ is "almost perfect agreement"). The 89.4% overall percent agreement means the 2 readers gave the *same* Likert score on 89% of ratings. **For our v0: if we do a clinical reader study, the Gwet's AC2 + 14-day washout + cross-over design is the protocol to follow.**

6. **The pseudo-crown self-bootstrapping is *robust to pseudo-crown quality* because natural teeth dominate the loss** (Sec 2.2.3). The ablation shows the "w/o data expansion" variant has the *worst* EMD (82.9 × 10³ vs full's 51.2), but the paper's deeper insight is *qualitative*: "the contextual learning signal is overwhelmingly dominated by the numerous high-fidelity natural teeth, making the training process highly robust to the finer-grained inaccuracies of the few synthesized crowns." This is the *same* insight as semi-supervised learning's "consistency regularization" — the *many* real samples dominate the loss, so the *few* noisy pseudo-labels don't hurt. **For our v0: the v0 H5 ablation should *quantify* this robustness** by varying the ratio of synthetic-to-real samples (1:1, 2:1, 5:1, 10:1) and showing that the model is *robust* up to ~5:1 and only starts degrading at 10:1.

7. **The 1,364 partially-edentulous scans in Source D were *deliberately* included via the self-bootstrapping** (Sec 4.1) — the paper explicitly notes that "the development pool underwent a rigorous two-stage curation overseen by trained orthodontists" with "scans exhibiting severe malocclusion (e.g., extensive crossbite, open bite), or excessive crowding and spacing that compromised the natural arch form were also excluded." This is *important* — the self-bootstrapping works *only* when the underlying scans are *not* pathological. The 6+ missing teeth exclusion is also important (tooth-bare scans with too many missing teeth are too noisy to use as context). For our v0: **the v0 paper's H5 ablation should *match* CrownGen's curation rules** (exclude ≥6 missing, exclude severe malocclusion) so the comparison is fair.

8. **The DPSR-based mesh reconstruction is trained *independently* of the diffusion** (Sec 4.3, last paragraph). This is a *clever* engineering choice: the diffusion generates point clouds (1024 points per tooth), the DPSR converts to mesh, the two are trained with *separate* loss functions. The *advantage*: the diffusion can be optimized for *point-cloud quality* (CD, EMD, F1) without worrying about mesh artifacts. The *disadvantage*: the point cloud → mesh step can *introduce* errors (e.g., over-smoothing in the cervical region). The paper's workaround: train the DPSR on *ground-truth* tooth meshes (not on the diffusion's outputs), so the DPSR is *robust* to the diffusion's idiosyncratic point distributions. **For our v0: the *independent* DPSR training is the right design** — the alternative (joint training) couples two loss functions with different scales, and the joint loss is *dominated* by the diffusion's MSE (3 orders of magnitude larger than the DPSR's MSE on the indicator grid). Independent training is *cleaner* and *faster* to debug.

9. **The boundary prediction module uses *zeroed-out* point clouds for the target teeth** (Sec 4.4) — "for the target teeth, these points are zeroed out to maintain a consistent input tensor shape while signaling their absence." This is a *clever* trick: instead of *removing* the target teeth from the input (which would change the tensor shape and require masking), the target teeth's points are *set to zero*. The DITA layers then *learn* that zero-point teeth are *absent* — the network "sees" the target's *location* (the zeroed-out points are in the right 3D position) but not its *shape*. This is a *more informative* input than "remove the target entirely" because the *position* of the target is a *strong* prior for the *shape* of the target (teeth don't move around the arch — tooth 16 is always in the 16 position). **For our v0: the zeroed-out-points trick is a 5-line implementation and is *the* cleanest way to encode "this tooth is missing" in a point-based network.**

10. **The CrownGen-assisted workflow uses ExoCAD for the *margin-line adaptation* step** (Sec 3, Discussion) — "These platforms have robust functions that can delineate the margin line from the jaw with minimal user intervention and automatically morph the initial crown design to adapt to the margin line." The paper *explicitly* delegates the margin-line problem to the *clinical CAD software* (ExoCAD, CEREC, 3Shape), arguing that "the geometry of margin lines is extraordinarily diverse and dependent on highly variable factors... Attempting to model this vast combinatorial space may not be the most effective approach." This is a *pragmatic* design choice: **don't try to solve the *full* crown-design problem in one AI, solve the *anatomy* part and let the *clinical CAD* handle the *interface* part.** For our v0, this is a *v0-architecture* recommendation: the v0 sub-task 2 (crown gen) should focus on the *supragingival anatomy* and delegate the *margin* + *intaglio* to a *separate* sub-task 2b (margin adaptation) that uses clinical CAD tools or a *simpler* model (e.g., a signed-distance field morph).

## Quote-worthy sentences

- **"Digital crown design remains a labor-intensive bottleneck in restorative dentistry."** (Abstract)
- **"An ideal automated crown generation system must therefore satisfy three core requirements: it should (i) accept any arbitrary subset of existing teeth as conditional input, (ii) be able to synthesize any number of prostheses required, and (iii) remain agnostic to the presence or absence of abutment preparations."** (Sec 1)
- **"CrownGen's tooth-level object representation and explicit inter-tooth attention mechanism circumvent this fundamental limitation. By modeling the dentition as a collection of interacting teeth, the framework's difficulty does not scale with the number of missing teeth being restored."** (Sec 2.2.2)
- **"For some metrics, CrownGen's performance exhibited a slight improvement as the number of missing teeth restored increased. We speculate that a larger and more globally distributed set of missing teeth compels the model to integrate information from a wider range of context teeth, providing richer, more robust conditional guidance compared to scenarios where generation is highly localized."** (Sec 2.2.2)
- **"While all methods delivered reasonable performance in the single-tooth restoration scenario, their efficacy collapsed as the complexity of the restoration increased."** (Sec 2.2.2) — *on the point-cloud-completion baselines*
- **"The pronounced impact on a globally sensitive, mass-preserving metric like EMD validates our assumption that explicit inter-tooth attention is crucial for capturing these global morphogenetic signals from adjacent and antagonistic teeth."** (Sec 2.2.3) — *on DITA*
- **"CrownGen offers unparalleled scalability in data utilization... The ability to leverage imperfect clinical data at scale is a profound practical advantage, ensuring CrownGen's continuous improvement and adaptation."** (Sec 3, Discussion) — *on the pseudo-crown self-bootstrapping*
- **"Critically, the inference time remains constant regardless of the number of crowns generated, making the overhead negligible in complex multi-tooth restoration cases."** (Sec 3, Discussion) — *on the constant-time diffusion*
- **"CrownGen's workflow presupposes an initial tooth segmentation step. We view this as a pragmatic design choice rather than a barrier."** (Sec 3, Discussion)
- **"CrownGen provides the anatomically harmonized initial proposal—a creatively demanding and time-intensive part of the process—which can then be finalized using established clinical tools."** (Sec 3, Discussion) — *on the human-AI collaboration*
- **"We conclude that the CrownGen-assisted CAD workflow is non-inferior to the conventional, fully manual CAD workflow by expert technicians in producing clinically high-quality dental crowns."** (Sec 2.3.3)
- **"CrownGen represents a significant step towards the complete automation of prosthetic dental design. By moving beyond the rigid, single-object completion paradigm, our framework provides a flexible, scalable, and clinically robust solution capable of addressing complex, multi-tooth restorative cases."** (Sec 3, Discussion)
- **"This tooth-centric approach enables our DITA mechanism to explicitly model inter-tooth relationships through a spatially-weighted attention system, learning to prioritize morphogenetic signals from adjacent and opposing teeth for superior anatomical accuracy."** (Sec 3, Discussion)
- **"The ability to leverage imperfect clinical data at scale is a profound practical advantage, ensuring CrownGen's continuous improvement and adaptation."** (Sec 3, Discussion)

## Code/data

- **Code:** ❌ not released as of arXiv v2 (Jan 2026). Email the corresponding author (Juyoung Bae, 2cddb75f@arXiv) for code-on-request. Likely release upon peer-review submission (MedIA, IEEE TMI, or similar clinical journal).
- **Public data used:**
  - 3DTeethSeg (Wang 2024) — [Source A]
  - 3DTeethLand (Li 2024) — [Source B]
  - ToothFairy 1+2 (Ben Hamadou ICCV 2023) — [Source C]
- **Private data (Source D + E):** Delun Dental Hospital Group, accessible "for non-commercial academic use" via corresponding author, subject to MTA + IRB review (HAREC HREP-2024-0257)
- **Ethics approval:** HKUST HAREC HREP-2024-0257
- **Funding:** Hong Kong Innovation and Technology Commission (GHP/006/22GD, ITCPD/17-9) + Hong Kong RGC (T45-401/22-N)
- **Related dental-3D-gen papers by the same group (HKUST + UHK + Delun):** Diff-OSGN (Oct 2025, sciopen) is the *most recent* — the group has 2+ dental-crown AI papers in 2025

## For our project

### Concrete next steps for v0

1. **(v0 sub-task 2) ADOPT DITA as the v0 H3 mechanism for sub-task 2 (crown gen)** — the zig-zag FDI ordering + index-based RPE is a 5-line implementation, the multi-head attention is standard Transformer-XL / T5 code (available in HuggingFace transformers), and the 17-30% CD/EMD improvement from the ablation is the *biggest* single-component gain in the reading list. Engineering cost: 2-3 days for the DITA layer + integration into MADCrowner (paper 034) or ToothCraft (paper 036), $50-100 Lambda for retraining, expected +5-15% CD on the v0 evaluation. The v0 paper's headline H3 result.

2. **(v0 sub-task 2) ADOPT the tooth-level point-cloud representation for v0** — every tooth as its own 1024-point cloud with binary indicator (0=context, 1=target) + 8-dim FDI embedding. This decouples the *where* problem (segmentation) from the *what* problem (generation), enabling the *multi-crown-in-one-pass* capability. Engineering cost: 1-2 days to refactor v0's data pipeline, $0 compute. This is the *most important* architectural choice in the v0 paper.

3. **(v0 sub-task 2) ADOPT the boundary prediction module as a v0 sub-task 2a (localization) + 2b (shape) decomposition** — train a 5-parameter cylinder regressor *first* (Smooth-L1 loss, 1000 epochs, Adam + cosine), then condition the diffusion on the predicted boundaries. Engineering cost: 1 week for the boundary module + integration, $200-300 Lambda for training. The v0 paper's most publishable H1 result (2-stage localization + shape > 1-stage end-to-end).

4. **(v0 sub-task 4) ADOPT the DPSR-based mesh reconstruction for v0** — train DPSR *independently* of the diffusion on ground-truth tooth meshes, then apply as a post-processing step to the diffusion's point cloud output. Engineering cost: 2-3 days for the DPSR integration, $50 Lambda for training. The DPSR is the *only* mesh reconstruction method in the reading list that produces *watertight* meshes *directly* from point clouds (Marching Cubes iso-surface at indicator=0.5).

5. **(v0 paper) CITE CrownGen as the v0 paper's *2025 SoTA reference*** — the *most recent* paper, the *strongest* evaluation protocol, the *only* clinical non-inferiority trial in the reading list. The v0 paper's related work should organize the dental-3D-gen landscape as: (a) pre-2024 (single-crown, monolithic input, no clinical trial) — DMC 033, DCrownFormer 032, MADCrowner 034, ToothCraft 036, ToothForge 037; (b) 2025 (multi-crown, tooth-level, clinical trial) — CrownGen 058. This is the *right* "the field has evolved" narrative for the v0 paper.

6. **(v0 paper) ADOPT the pseudo-crown self-bootstrapping for v0 H5 ablation** — use v0's own v1 model to generate pseudo-crowns for partially-edentulous scans in the training set, then retrain v2 on the augmented dataset. The ablation is *directly* the v0 paper's H5 (synthetic data bootstraps training), but the *source* of the synthetic data is *the v0 model itself* (CrownGen's twist). Compare to v0's prior H5 ablation (TeethGenerator 051's external synthetic data, +0.05% improvement) — the *self-bootstrapping* should be +1-5% (an order of magnitude better) based on CrownGen's 29% CD improvement.

7. **(v0 paper) REPLICATE CrownGen's clinical reader study design** — Gwet's AC2 inter-rater reliability, 14-day washout, cross-over design, pre-specified non-inferiority margin, 4 criteria (occlusion + proximal contact + alignment + crown form), 2 blinded readers with 4+ years clinical experience. The v0 paper's clinical claims should match CrownGen's *rigor* (not just "the AI is better" but "the AI is *non-inferior* to the manual expert with a *pre-specified* margin"). The 17.78% time reduction + 95.2% clinical acceptability is the *target* for v0's clinical results.

8. **(v0 deployment) ADOPT CrownGen's *entry-level technician + AI post-processing* workflow** for v0's deployment plan — the labor cost reduction is *much* larger than the time reduction (entry-level tech at half the salary of an expert tech). The v0 paper's clinical-impact story should be: "v0-assisted dental lab can use *entry-level* technicians (50% salary) for the post-processing, with the AI handling the *anatomy* (the *hardest* part of crown design). The *clinical acceptability* is 95%+ and the *time* is 17-30% faster."

9. **(v0 paper) ADOPT the FDI zig-zag ordering as a hard-coded constant in v0's data pipeline** — `[17, 47, 16, 46, 15, 45, 14, 44, 13, 43, 12, 42, 11, 41, 21, 31, 22, 32, 23, 33, 24, 34, 25, 35, 26, 36, 27, 37]` is the exact order CrownGen uses, and it's the *right* inductive bias for any point-based dental AI that uses attention. Engineering cost: 1 line of code, $0 compute, expected +1-2% CD on any H3-mechanism-equipped v0 model.

10. **(v0 paper) FRAME the v0 paper's contribution as the *clinical validation* of an H3+H5-equipped architecture** — CrownGen is the *first* dental-3D-gen paper with a formal clinical non-inferiority trial. The v0 paper can be the *second* (or the *first* in a specific clinical sub-scenario, e.g., elderly patients, pediatric patients, implant-supported restorations). The clinical-trial design is the *right* "what's new" for a v0 paper in 2026 — the architecture is *known* (DMC/MADCrowner/ToothCraft lineage + VF-Net/LION for priors + CrownGen for DITA), the *clinical evidence* is the *gap*.

### v0 stack updated

- **sub-task 1 (FDI seg)** = Cao25 + CrownSegger + Point2SSM-derivative + Mesh2SSM++ (paper 041) + STEAM-style GAM+MGR (paper 042) + 32-class tooth-classifier head + ME-loss regularizer + 2×2×8 FDI grid structure (paper 051) + nnU-Net ResEnc L 5-fold (paper 053, CBCT) + U-Mamba2 (paper 054, CBCT) + cTooth+ cross-dataset eval (paper 055) + iMeshSegNet (paper 056, v0.5) + PointNet-Reg (paper 056, v0.5 landmark head)
- **sub-task 2 (crown gen, *conditional* path)** = MADCrowner (paper 034) + ToothCraft (paper 036) + ToothForge (paper 037) + DMC (paper 033) + DCrownFormer (paper 032) + **DITA mechanism (this paper, primary H3) + tooth-level point representation (this paper) + boundary prediction module (this paper, primary H1) + pseudo-crown self-bootstrapping (this paper, primary H5) + DPSR mesh reconstruction (this paper) + per-point variance output (paper 057) + 2D-grid mesh regularization (paper 057)**
- **sub-task 2 (crown gen, *unconditional* prior path)** = VF-Net (paper 057, primary) + LION (paper 005, secondary) + TeethGenerator (paper 051, tertiary, for orthodontic pre/post) + SAE-LP (paper 040, for spectral prior) + 1-NNA + COV + MMD evaluation
- **sub-task 4 (outer surface)** = PVD + ME-loss + DiGS + FlexiCubes + Surface Projection loss + MGR + **CrownGen's diffusion module (this paper, v0.5 for multi-crown support) + DITA (this paper, primary H3 for outer surface)**
- **sub-task 5 (mesh output)** = FlexiCubes (paper 007) + Differentiable Poisson Surface Reconstruction (this paper, primary for point → mesh) + NDC (paper 006, v0 alternative)
- **Training data** = 3DTeethSeg'22 + 3DS + ODD + ToothForge synthetic + TeethGenerator synthetic + VF-Net synthetic (paper 057, on FDI 16) + LION synthetic (paper 005, ShapeNet) + **pseudo-crown self-bootstrapping (this paper, on Delun Dental 1364 partially-edentulous scans, v0.5)**
- **Eval** = + IoU_Antag + ToothForge reconstruction filter + spectral-only baseline + per-tooth-type CD-L2 breakdown + ME-loss correspondence + LION 1-NNA + UCD + FDI 16 cross-dataset test (paper 057) + per-clinic 50-scan fine-tune protocol (paper 057) + **496 external test scans (this paper, CrownGen's external test set, 026288 test scenarios) + 26-case clinical reader study with 14-day washout + Gwet's AC2 inter-rater + non-inferiority test with pre-specified margin (this paper, gold-standard clinical eval)**
- **v0 compute** = **~$5,540-6,830 Lambda** (was $5,140-6,230, +$200-300 for boundary prediction module training + $50-100 for DITA integration + $50 for DPSR training + $100-200 for clinical reader study recruitment ($100/reader × 2 readers × 1 case set is a rough estimate))

### Open questions for HK

(i) **Adopt DITA as the v0 H3 mechanism for sub-task 2 (crown gen) and sub-task 4 (outer surface)?** (recommend YES — 5-line implementation + 2-3 day integration + $50-100 Lambda retraining; the 17-30% CD/EMD improvement from the ablation is the *biggest* single-component gain in the reading list; the zig-zag FDI ordering is a *clinically-motivated* inductive bias that *learns the right prior for free*)

(ii) **Adopt the tooth-level point-cloud representation for v0 (every tooth as 1024-point cloud with binary indicator + 8-dim FDI embedding)?** (recommend YES — 1-2 day data-pipeline refactor, $0 compute; enables multi-crown-in-one-pass; the most important architectural choice in the v0 paper)

(iii) **Adopt the boundary prediction module as a v0 sub-task 2a (localization) + 2b (shape) decomposition?** (recommend YES — 1 week for module + integration + $200-300 Lambda training; the 27% CD improvement from the ablation is the v0 paper's most publishable H1 result; aligns with the *clinical CAD software* approach (ExoCAD, CEREC, 3Shape) for the margin-line step)

(iv) **Adopt DPSR for v0 sub-task 5 (mesh output)?** (recommend YES — 2-3 day integration + $50 Lambda training; the *only* method in the reading list that produces *watertight* meshes *directly* from point clouds; can be trained *independently* of the diffusion, decoupling the loss functions)

(v) **Replicate CrownGen's clinical reader study design for v0?** (recommend YES — Gwet's AC2 + 14-day washout + cross-over + pre-specified NI margin + 4 criteria + 2 blinded readers; the v0 paper's clinical claims should match CrownGen's *rigor*; $100-200/reader × 2 readers × 26 cases is a reasonable recruitment budget)

(vi) **Adopt pseudo-crown self-bootstrapping for v0 H5 ablation?** (recommend YES — the v0 paper's *most* publishable H5 result; use v0's own v1 model to synthesize pseudo-crowns for partially-edentulous scans, then retrain v2; compare to TeethGenerator 051's external-synthetic H5 (+0.05%); expect +1-5% improvement, an order of magnitude better)

(vii) **Frame the v0 paper as the *clinical validation* of a H3+H5 architecture?** (recommend YES — the architecture is *known* (Lombaert-lineage + VF-Net + CrownGen), the *clinical evidence* is the *gap*; the v0 paper can be the *first* clinical-trial-validated AI for a specific sub-scenario, e.g., elderly patients, implant-supported restorations, or 6+ missing teeth)

(viii) **Cite CrownGen as the v0 paper's *2025 SoTA reference*** in the related work section? (recommend YES — most recent paper, strongest evaluation protocol, only clinical NI trial; the v0 paper's "the field has evolved" narrative should pivot at CrownGen 2025)

(ix) **Adopt the entry-level-technician + AI workflow for v0 deployment?** (recommend YES — 50% labor cost reduction is much larger than the 17% time reduction; the v0 paper's clinical-impact story should be "AI + entry-level tech = expert-level crowns at half the labor cost")

(x) **Use the FDI zig-zag ordering as a hard-coded constant in v0's data pipeline?** (recommend YES — 1 line of code, $0 compute, +1-2% CD on any H3-equipped model; the *right* inductive bias for any point-based dental AI)

### Next paper to read (059)

Strong candidates for 059:

1. **VBCD: A Voxel-Based Framework for Personalized Dental Crown Design** (Wei et al. 2025, MICCAI 2025) — the *voxel-based* counterpart to CrownGen (point-based) and DCrownFormer (point-to-mesh). Would close the "all 3 substrates" arc (voxel + point + mesh). Note: Wei et al. are *also* Lombaert-lineage authors (paper 034, 035) — VBCD is the *voxel* cousin of MADCrowner. Already read as paper 035 in the reading list, so redundant.

2. **Diff-OSGN: Diffusion-based occlusal surface generation network** (Wang et al. Oct 2025, *Journal of Computer-Aided Design & Computer Graphics*, sciopen CVM) — another 2025 dental-crown diffusion paper, focused on the *occlusal surface only* (not the full crown). Would complement CrownGen's full-crown approach with a *surface-specific* approach. **Strong candidate — 2025, *focused scope*, same diffusion paradigm as CrownGen, different output granularity.**

3. **Guided 3D CBCT Synthesis with Fine-Grained Tooth Conditioning** (Aug 2025, arXiv:2508.14276) — conditional diffusion for *3D dental volume* (CBCT) generation, guided by tooth-level binary attributes. Different scope (3D volume not 3D mesh), different modality (CBCT not intraoral scan), but the *same* H3 + diffusion paradigm. **Could broaden v0's scope to 3D volume generation.**

4. **VF-Net follow-up / 3DTeethGen** — paper 057's recommendation for "3DTeethGen (Sun et al. 2024)" was a *misnomer* (no such paper exists); the recommendation was actually VF-Net. Now that VF-Net (057) and CrownGen (058) are both read, the next *unconditional* tooth-shape prior paper to read would be **PointFlow (Yang et al. 2019, the *original* continuous-NF prior for point clouds)** — would let us re-examine the "VAE + NF prior" pattern that VF-Net and LION both use, and identify any 2024+ improvements.

5. **DCrownFormer journal extension** (Yang et al. 2024) — paper 032's MICCAI 2024 paper had a "Personalized Dental Crown Design Based on Local Context-Aware Transformer" version; would let us see how the *original* Lombaert-lineage "local context" pattern evolved. Would close the "local context" H3 mechanism arc (DCrownFormer local + CrownGen DITA global + AnchorFormer 011 anchors).

6. **LION 2 / LION++ / DiT-3D** — any 2024-2025 *latent* 3D diffusion paper (not point-based, not mesh-based) would be a *new* H2 paradigm to read. Latent 3D diffusion is the *fastest-growing* 2024-2025 3D-gen paradigm (DiT-3D, MeshDiffusion 014, etc.).

**Recommendation: paper 059 = Diff-OSGN (Wang et al. Oct 2025)** — a *complementary* 2025 diffusion paper to CrownGen, focused on the *occlusal surface only* (the *hardest* part of crown design — the part that articulates with the opposing arch). Reading both would let us *cross-compare* two 2025 diffusion approaches:
- **CrownGen**: full crown (anatomy + occlusal + cervical), point-diffusion, multi-crown-in-one-pass, DITA
- **Diff-OSGN**: occlusal surface only (a *sub-task* of CrownGen), diffusion (medium unspecified), single-crown (presumably), ?-attention

This is the *right* next step for closing the "2025 dental-3D-gen diffusion" arc and for *granular* comparison of the two approaches' architectures. If Diff-OSGN turns out to be *unreadable* (paywalled, code not available, no arXiv version), fall back to the **Guided 3D CBCT Synthesis** paper (arXiv:2508.14276) as a *broader-scope* 2025 dental diffusion read.

### Open thread: Lombaert-lineage 6th paper
The 057 recommendation for the 6th Lombaert-lineage paper (Hosseinimanesh et al. 2024 "Personalized dental crown design: a point-to-mesh completion network", *MedIA*) is the *journal extension* of the already-read paper 033 (DMC, MICCAI 2023). The journal extension adds *one* new component: a *refined morphology-aware loss* (curvature penalty + occlusal convexity). The 80% of the content that overlaps with paper 033 is *not* worth re-reading. **The Lombaert-lineage is effectively *closed* at 5 papers (033, 034, 035, 036, 037) for the v0 paper's purposes.** If a *new* Lombaert-lineage paper appears (e.g., the Lombaert group's *next* MICCAI submission in 2026), that would be worth reading. For now, the *next* Lombaert-lineage paper is "tooth side mesh completion" or "bridge generation" (extending the crown approach to multi-unit prostheses), and the *signal-to-noise* for reading it now is *low* (we'd learn a *delta* on a known approach, not a *new* approach). Defer.
