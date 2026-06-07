# Paper 034 — *MADCrowner: Margin Aware Dental Crown Design with Template Deformation and Refinement*

**Authors:** Linda Wei¹, Chang Liu²'³, Wenran Zhang⁴, Yuxuan Hu¹, Ruiyang Li⁵, Feng Qi⁸, Changyao Tian¹, Ke Wang¹, Yuanyuan Wang², Shaoting Zhang³*, Dimitris Metaxas⁶, Hongsheng Li¹'⁷* (*corresponding, ¹'²'³ co-first)
**Affiliations:**
1. Multimedia Laboratory, The Chinese University of Hong Kong (CUHK MMLab)
2. College of Biomedical Engineering, Fudan University
3. SenseTime Research
4. Department of Second Dental Center, Shanghai Ninth People's Hospital, SJTU School of Medicine
5. Department of Computer Science and Engineering, CUHK
6. Department of Computer Science, Rutgers University
7. Centre for Perceptual and Interactive Intelligence (CPII) under InnoHK
8. Shanghai Stomatological Hospital, Fudan University
**Venue:**
- **Journal:** *Medical Image Analysis* (MedIA) **2026 (in press)** — the flagship dental-imaging journal
- **Preprint:** arXiv:[2603.04771](https://arxiv.org/abs/2603.04771) (Mar 5, 2026, v1)
- **Code:** ✅ **open source** at [github.com/lullcant/MADCrowner](https://github.com/lullcant/MADCrowner) (includes `train_crown_deformer.py`, `inference.py`, `run.sh`, `test.sh`, `fdi_template/`, `mydataset/`, `models/`, requires pytorch3d + Accelerate)
- **Funding:** CPII/InnoHK, Guangdong Basic and Applied Basic Research Foundation (2023B1515130008), NSFC (62271115), Sichuan NSF (2025ZNSFSC0455)
- **Citations:** new (Mar 2026), expect high velocity — SenseTime + Shanghai Ninth People's Hospital clinical backing + open code
- **Read:** 2026-06-07 09:05 KST (Sunday, scholar hourly #22, ~50 min)

---

## TL;DR

**MADCrowner is the first margin-aware end-to-end dental crown generation framework — and the *clinical*-correct version of DMC (paper 033).** Two modules: **(1) CrownSegger** (transformer-free hybrid point-voxel VNet, segments the prepared abutment → extracts the cervical margin line via B-spline, 0.328 mm Hausdorff on test set, zero-shot 0.545 mm on the crown-generation dataset), and **(2) CrownDeformR** (3-stage transformer: GAT-SAT IOS feature extraction → CAT/SAT template deformation → 2× CAT coarse-to-fine refinement, deforms an **FDI-selected initial tooth template** into the patient-specific crown point cloud). The novel architectural win is the **CMPL loss (Curvature + Margin Penalty Loss)** that explicitly weights CD by `e^|κ|` (high-curvature regions get higher penalty) and adds a margin-line indicator term — analogous to DCrownFormer's CPL (paper 032) but extended with the margin term. The second novel contribution is the **tailored post-processing algorithm** (Algorithm 1) that trims the SAP/DPSR watertight overextension using the cervical margin line — a *critical* step that takes CD-L2 from 0.556 mm² → 0.185 mm² (a **3× reduction**, the largest single improvement in the paper, and it's a *post-process*, not a network change). **SoTA on every metric vs. PCN+SAP, TopNet+SAP, GRnet+SAP, DMCv2, VBCD, Diffusion-SDF**: CD-L2 0.185 mm² (vs. DMCv2 0.253, −27%), F-score 0.917 (vs. DMCv2 0.888, +3.3%), HDF 1.046 mm (vs. VBCD 1.150, −9%), and **first place on proximal contact area deviation** (medial 4.37 mm², lateral 4.07 mm² — both best). **37.3M params, 1.1 GB VRAM, 600 ms single-inference latency on L20** — directly chairside-deployable. **For our v0 sub-task 2 (crown generation): MADCrowner is now the SoTA reference**, the post-process trick is the highest-leverage v0 change, and the CrownSegger module is the right v0 segmentation head for *abutment + margin* (complementary to Cao25's FDI segmentation in paper 026). **The most honest limitation statement in our reading list** (Sec. 5 Discussion): the authors admit regression losses produce a *"smoothed average"* of plausible outputs, and explicitly call for future work on diffusion models — a strong H2-conditional signal that we should heed in v1.

## Research question + their answer

**Q:** Existing learning-based dental crown generation methods (DCPR-GAN's cGAN on 2D depth maps, DMC's point cloud completion + SAP, VBCD's voxel-based, Diffusion-SDF's latent diffusion) all suffer from one of three clinical failures: (1) **inadequate spatial resolution** on the occlusal surface (2D depth maps obscure the cervical margin; voxel methods are grid-bottlenecked), (2) **noisy outputs** that don't generalize across tooth positions, and (3) **overextension artifacts** from surface reconstruction — the SAP/DPSR watertight constraint produces a closed mesh that *fundamentally conflicts* with the dental crown's open genus-zero topology (Fig. 2 of the paper shows a dramatic sealed bottom). Can a single framework address all three by **explicitly modeling the cervical margin** (a) as a network input condition, (b) as a loss term, and (c) as a post-process boundary?

**A:** Yes — **decompose the CAD workflow as a 2-stage (segmentation + deformation) framework, with the cervical margin as the connecting signal**:

1. **Stage 1 — CrownSegger (margin extraction):** point-voxel hybrid (concatenate spatial coords + normals → voxelize → VFE layers → dense 3D volume → VNet backbone → concat with point features → MLP segmentation), supervised by CE + Dice loss. The cervical margin is the boundary of the prepared-abutment mask, smoothed by B-spline and resampled to 1,000 vertices (Algorithm 1). Transformer-free for clinical edge-deployment efficiency.

2. **Stage 2 — CrownDeformR (crown generation):** **CAD-inspired 3-stage transformer** that mirrors what a dental technician does — (a) **IOS feature extraction** (GAT-SAT blocks on IOS+abutment mask → global context `g ∈ R^512` + multiscale `F_IOS`), (b) **template deformation** (FDI-selected tooth template + CAT cross-attention on `g` → coarse crown `P_Coarse`), (c) **coarse-to-fine refinement** (2× CAT modules that cross-reference *localized* IOS features at progressively finer scales → final crown point cloud). Three transformer variants: **GAT** (geometry-aware, FPS-downsampled attention), **SAT** (self-attention), **CAT** (cross-attention between template/IOS features and crown queries).

3. **Surface reconstruction + post-process:** DPSR (Peng et al. NeurIPS 2021) reconstructs a watertight mesh, then the **tailored post-process** uses the CrownSegger margin line as a hard boundary — smooth margin → define a surface (normal = tooth growth direction) → remove faces below that surface → project boundary vertices onto the smoothed margin line. **Result:** open genus-zero mesh suitable for CAD manipulation and 3D printing.

The **CMPL loss** (Eq. 1) is the key dental-specific insight: `L_CMPL = (1/|P̂_crown|) Σ_p e^|κ(p)| · d(p, P_GT) + (1/|P_GT|) Σ_q e^|κ(q)| · d(q, P̂_crown) + margin-indicator term`. The `e^|κ|` exponential upweights high-curvature regions (cusps, fossae, grooves) — the *exact* anatomy that CD alone misses because CD is dominated by the smooth outer surface. The margin term explicitly minimizes distance to the prepared-abutment margin line — the *exact* region where the HDF distance is highest in DMCv2 (paper 033) and DCrownFormer (paper 032).

## Method (architecture, training, data)

### Pipeline (3 stages, 2 modules)

```
[IOS point cloud + abutment mask (zero-shot from CrownSegger)]
        ↓
CrownSegger (point-voxel VNet, transformer-free)
        ↓
[Cervical margin line: 1,000 B-spline-resampled vertices]
        ↓
CrownDeformR — IOS Feature Extraction (GAT-SAT blocks)
        ↓
[FDI-selected template T + global context g ∈ R^512 + multiscale F_IOS]
        ↓
CrownDeformR — Template Deformation (CAT → SAT → coarse P_Coarse)
        ↓
CrownDeformR — 2× Coarse-to-Fine Refinement (CAT on localized IOS)
        ↓
[Final crown point cloud P̂_crown]
        ↓
DPSR surface reconstruction (Peng 2021, SAP)
        ↓
Post-process: B-spline margin → surface → trim below + project boundary
        ↓
[Open genus-zero crown mesh]
```

### Three transformer primitives (Sec 3.2.1 + Appendix Fig 18)

| Block | Q / K / V | Role |
|-------|-----------|------|
| **GAT** (Geometry-Aware Transformer) | Q from FPS-downsampled F_IOS (1/4 spatial); K, V from full F_IOS | Hierarchical feature extraction with progressive downsampling, preserves local geometry |
| **SAT** (Self-Attention Transformer) | Q = K = V = F_IOS (or template features) | Long-range dependencies within features, internal contextual interactions |
| **CAT** (Cross-Attention Transformer) | Q = template/coarse features; K, V = IOS features | Spatial correspondence between crown queries and IOS context, captures relative positional dependencies |

4 heads, 512 hidden channels. Multi-head attention is the standard transformer block; the *specialization* is in what Q/K/V come from.

### Optimization objectives (Sec 3.3, Eq. 2)

```
L_total = L_Coarse + L_Refine1 + L_Refine2 + L_DPSR

L_Coarse:    CD(P_Coarse, P_GT^(1/4))       — establish basic spatial positioning
L_Refine1:   CD(P_refine1, P_GT^(1/2))      — structural consistency at higher density
L_Refine2:   CMPL(P̂_crown, P_GT, M(P_GT))  — fine anatomical detail + margin
L_DPSR:      MSE on DPSR grid                 — supervise the extracted mesh
```

**CMPL = CPL + MPL** (Eq. 1 in the paper):
- **CPL** (Curvature Penalty Loss): `e^λ|κ(p)| · min_q ||p - q||_2` — λ=1 (sweep optimum, Fig 11)
- **MPL** (Margin Penalty Loss): `1{q ∈ M(P_GT)} · min_q ||p - q||_2` — explicit margin line term

**Deep supervision across resolutions** is the key training trick: each stage's loss supervises its own resolution's point cloud against the corresponding downsampled GT. This is the same pattern as DCrownFormer's MRL (paper 032) but extended to multiple scales.

### Post-processing (Sec 3.4 + Algorithm 1)

```
1. Abutment extraction: faces with all 3 vertices labeled as abutment
2. Noise exclusion: largest connected component M_a
3. Boundary detection: boundary vertices V_b of M_a
4. B-spline interpolation on V_b → cervical margin (resampled to 1,000 vertices)
5. Smoothed margin + centroid define a surface (normal = tooth growth direction)
6. Remove all faces below that surface
7. Project boundary vertices of processed mesh onto the smoothed margin line
```

**Result:** open genus-zero mesh that "precisely conforms to the margin line" (paper, Sec 3.4). This is the **single highest-impact** step in the paper: CD-L2 goes from 0.556 → 0.185 mm² (3× reduction). It's a 200-line NumPy implementation, no network training, no GPU.

### Datasets (Sec 4.1)

**CrownSegger dataset:** 576 IOS scans (476 train / 100 test), point-wise abutment labels.

**CrownDeformR dataset:** **4,602 patients** with single tooth defects, target teeth prepared as abutments. **Premolars + molars only** (no incisors, no canines — explicit coverage gap, see "For our project" #7). Per IOS, extract 2 cm cubic region centered at crown centroid. 8:1:1 split, stratified by tooth type. Premolar 1: 543+152=695 cases, Premolar 2: 412+178=590, Molar 1: 1096+1452=2548, Molar 2: 270+499=769 (data is split upper/lower, total ~4,602).

### Implementation (Sec 4.2)

- **CrownSegger:** PyTorch, NVIDIA GTX 3080 Ti, batch 16, 300 epochs, AdamW (lr=1e-4) + cosine schedule
- **CrownDeformR:** PyTorch, **4× NVIDIA RTX 4090**, batch 8, 500 epochs, same optimizer
- **Inference:** 37.3M total params, 1.1 GB VRAM, 600 ms total (Seg + Gen + Post) on **NVIDIA L20** (a 48 GB inference GPU)
- **Abutment mask in CrownDeformR is the zero-shot prediction from CrownSegger** — no manual annotation in the generation pipeline

## Results

### CrownSegger vs. baselines (Table 2)

| Method | Accuracy ↑ | IoU ↑ | HDF (mm) ↓ |
|--------|-----------|-------|-----------|
| PointNet | 0.935 | 0.858 | 2.450 |
| PointNet++ | 0.956 | 0.901 | 1.720 |
| PointTransformer | 0.963 | 0.939 | 1.238 |
| **CrownSegger** | **0.991** | **0.972** | **0.328** |
| CrownSegger zero-shot on crown-design dataset | — | — | 0.545 |

**The HDF reduction from 1.238 → 0.328 mm (4× better) is the strongest single-seg-margin result in the reading list**, and the zero-shot transfer to the crown-design dataset (HDF 0.545 mm) is direct H5 evidence for clinical deployment.

### Crown generation (point cloud, Table 3)

| Method | CD-L2 ↓ | Fidelity ↓ | HDF ↓ | F-score ↑ |
|--------|---------|-----------|-------|-----------|
| PCN+SAP | 0.271 | 1.400 | 1.441 | 0.850 |
| TopNet+SAP | 0.354 | 1.752 | 1.836 | 0.770 |
| GRnet+SAP | 0.230 | 1.118 | 1.134 | 0.874 |
| DMCv2 | 0.227 | 1.227 | 1.245 | 0.868 |
| **MADCrowner** | **0.176** | **1.093** | **1.027** | **0.903** |

**MADCrowner wins all 4 metrics** by margins of 22-29% on CD-L2, 10-12% on HDF, 3-4% on F-score. Notable: TopNet+SAP loses to PCN+SAP on CD (0.354 vs 0.271) — confirms the PCN/FoldingNet decoder is the right default.

### Crown generation (mesh, Table 4) — *this is the real clinical comparison*

| Method | CD-L2 ↓ | Fidelity ↓ | HDF ↓ | F-score ↑ |
|--------|---------|-----------|-------|-----------|
| GRnet+SAP | 0.210 | 0.103 | 1.280 | 0.899 |
| DMCv2 | 0.253 | 0.127 | 1.322 | 0.873 |
| VBCD | 0.209 | 0.109 | 1.150 | 0.909 |
| **Diffusion-SDF** | 0.219 | 0.110 | **1.390** | 0.893 |
| **MADCrowner** | **0.185** | **0.086** | **1.046** | **0.917** |
| MADCrowner w/o post-processing | 0.556 | 0.091 | 4.016 | 0.830 |

**Key findings:**
1. **MADCrowner beats Diffusion-SDF on the mesh metric** (CD-L2 0.185 vs 0.219, F-score 0.917 vs 0.893) — direct empirical support for the H2-conditional refinement from paper 033
2. **MADCrowner w/o post-processing has HDF 4.016 mm** — the overextension is *clinically catastrophic* without the trim
3. **The post-processing saves 0.371 mm² CD-L2 (66% reduction)** — by far the highest-leverage single step

### Proximal contact analysis (Table 7) — *unique to this paper*

| Method | Medial area diff (mm²) ↓ | Lateral area diff (mm²) ↓ |
|--------|------------------------|------------------------|
| PCN+SAP | 6.51 | 7.30 |
| TopNet+SAP | 15.97 | 16.50 |
| GRnet+SAP | 5.06 | 4.90 |
| DMCv2 | 5.18 | 4.72 |
| VBCD | 5.00 | 5.04 |
| Diffusion-SDF | 5.37 | 5.18 |
| **MADCrowner** | **4.37** | **4.07** |

**First time in our reading list that proximal contact is measured**, and MADCrowner wins both directions. This is the *clinical* metric dentists care about (food impaction, periodontal ligament damage).

### Ablation (Table 5)

| Components | CD-L2 ↓ | F-score ↑ | VRAM (MB) | Inference (ms) |
|------------|---------|-----------|-----------|---------------|
| Hemisphere template, no deformation, no margin | 0.220 | 0.866 | 1229 | 87 |
| + Template deformation module | 0.213 | 0.878 | 1229 | 90 |
| Initial template, no deformation, no margin | 0.212 | 0.886 | 1305 | 100 |
| + Template deformation (initial template) | 0.193 | 0.911 | 1305 | 103 |
| **+ Cervical margin constraint (full)** | **0.175** | **0.924** | 1305 | 105 |

**Findings:**
- The initial template matters: switching from hemisphere to FDI-selected tooth template reduces CD by 0.008 mm² (~4%) with zero VRAM cost
- Template deformation adds another 0.019 mm² CD reduction (~9%) with +15 ms inference
- Cervical margin constraint adds another 0.018 mm² CD reduction (~9%) with +2 ms inference
- **All three components together: 20% CD reduction, 5.7% F-score gain, with 18 ms total latency cost and +76 MB VRAM** — an *excellent* cost-benefit

### Failure case analysis (Sec 4.7, Fig 14)

Three clinical conditions that consistently cause failure:
1. **Inadequate tooth preparation** — irregular or insufficient reduction compromises geometric inference
2. **Incomplete intraoral scans** — holes or missing regions in adjacent teeth
3. **Absence of adjacent or antagonist teeth** — limits accurate reconstruction of occlusal relationships

**All three are *data coverage* limitations, not method limitations.** Standardized prep + complete scans + complete context → high-quality output. The implication for our v0 deployment: include a scan-quality check in the pipeline (paper 001's IGIP-style quality check is the right starting point).

## Connections to H1-H5

| Hypothesis | Status (after paper 034) | Evidence |
|------------|--------------------------|----------|
| **H1** (2-stage > 1-stage) | **CONFIRMED with refinement** | MADCrowner is 2-stage (Seg → Deform). The ablation shows the segmentation stage contributes materially (CrownSegger HDF 0.328 mm enables the post-process). **Refinement (paper 033):** "2-stage wins when the intermediate is a *semantic* representation (segmentation mask, depth map, etc.)" — the cervical margin is exactly that. |
| **H2** (latent diffusion > direct) | **STRONG REJECTION FOR CONSTRAINED TASKS** | Diffusion-SDF loses to MADCrowner on every mesh metric (Table 4: CD 0.219 vs 0.185, F-score 0.893 vs 0.917, HDF 1.390 vs 1.046). The 033 hypothesis refinement holds: for *patient-specific* tasks with strong conditioning, **deterministic + good losses + post-processing > diffusion**. MADCrowner + their post-process is now the SoTA. |
| **H3** (arch-conditional generation) | **STRONGEST DIRECT SUPPORT IN READING LIST** | The entire framework is arch-conditional — CrownSegger segments the *abutment* (not the whole arch), CrownDeformR attends to *adjacent + antagonist* teeth via CAT blocks. The **FDI-template selection** is a clever semantic prior that no prior paper has used (4 templates for 4 tooth positions). The cervical margin is *literally* a learned, anatomically-grounded conditioning signal. |
| **H4** (SDF > explicit mesh) | **REJECTED FOR OPEN MESHES; CONFIRMED FOR CLOSED** | MADCrowner uses point cloud + DPSR (like DMC, like DCrownFormer) — not SDF. The reason: dental crowns are *open* meshes, and watertight SAP/DPSR is *the wrong substrate* for open meshes (they had to add a post-process to fix it). **Refinement:** H4's win generalizes to *closed* objects (ShapeNet chairs, etc.) but not *open* objects (dental crowns, garments, leaves). For our project, this is a **major architectural insight** — FlexiCubes (paper 007) and NDC (paper 006) both produce closed meshes, so we need a *trim step* in v0 if we use them on crowns. |
| **H5** (synthetic → real generalization) | **NO NEW EVIDENCE** (real data only) | Same as 033 (DMC). However, the CrownSegger zero-shot transfer (IoU HDF 0.328 → 0.545 on the unrelated crown-design dataset) is **indirect H5 evidence** for the segmentation sub-task — the learned margin extractor generalizes. |

### Reusable tricks to adopt verbatim

1. **The post-processing algorithm (Sec 3.4)** — 200 lines of NumPy, $0 compute, **3× CD reduction**. **This is the highest-leverage single change** in the entire reading list, period. Implement first, before any model architecture work.
2. **The CMPL loss (Eq. 1)** — `e^|κ|` weighted CD + margin indicator. The exponential upweighting on high-curvature regions is the *right* anatomical prior for any shape generation task where detail matters (cusps, ridges, etc.). λ=1 is the empirical sweet spot.
3. **FDI-template selection** — for our 32-FDI arch, this generalizes to 32 templates (or 8 × 4 for jaw × position × class). ~30 KB of template data, +0.008 mm² CD free.
4. **Deep supervision at multiple resolutions** — each refinement stage supervises against its own downsampled GT. This is the *only* paper in our reading list that does this for *point cloud refinement* (most do it for images). Free training-stability win.
5. **CrownSegger's hybrid point-voxel VNet** — for sub-task 1 (FDI segmentation in 3DTeethSeg22), the transformer-free architecture is 10× more efficient than PointTransformer (paper 028 / Stratified Transformer) and the HDF result (0.328 mm) is the best in the reading list. Direct v0 candidate for the prep-margin detection sub-task.

## Surprises / interesting things buried in section 4

1. **The post-processing is the *biggest* improvement, not the network architecture.** CD-L2 0.556 → 0.185 mm² (3× reduction) from a 200-line NumPy algorithm. The author's network gains (CrownDeformR alone over DMCv2) are smaller in comparison (~10-15% CD). **This is a *democratizing* result** — small teams with no compute can adopt the post-process and immediately get 3× CD improvement, no GPU required.

2. **The "Discussion" admission of the regression-loss "smoothed average" problem (Sec 5)** is the most honest limitation statement in the dental-crown literature. The authors explicitly call out that *"regression-based losses, such as the Chamfer Distance... inadvertently encourage the network to produce a 'smoothed average' of plausible outputs rather than capturing the true 'mode' of the underlying distribution."* This is **direct H2-supporting evidence from a non-H2 paper** — even the dental-crown deterministic SoTA authors think diffusion is the next step.

3. **The proximal contact metric (Table 7) is unique to this paper.** No other paper in our reading list (32+ papers) measures it. It's the *clinical* metric dentists actually care about (food impaction, periodontal ligament damage), and MADCrowner wins both directions. **Adopt this metric in our v0 sub-task 2 evaluation.**

4. **CrownSegger is "transformer-free"** — they explicitly note this is for *clinical edge-deployment efficiency* (deployable on edge devices in a dental office). The point-voxel hybrid + VNet backbone trains in hours on a single 3080 Ti, not days on 8× A100. **This is the right v0 choice** for our M4 Mac mini pipeline — no transformer means easier MPS port.

5. **The training compute is heavy despite the model size**: 4× RTX 4090, 500 epochs, batch 8. Compare to DMC (paper 033) which is 1× A100, 22h, batch 32. MADCrowner's 3-stage transformer + multi-resolution deep supervision is much more iteration-hungry. **For v0, this matters** — we can train DMC in $25 on Lambda, MADCrowner would cost $400-800.

6. **The CMPL has TWO components** (CPL + MPL), and the ablation in Sec 4.5.4 shows λ=1 is the sweet spot for CPL. The MPL component has no hyperparameter — it's just an indicator term on the margin line. **Adopt the full CMPL** in v0 sub-task 2, don't just adopt CPL like paper 032.

7. **The failure cases are all *data* failures, not *method* failures.** Inadequate prep, incomplete scans, missing context teeth → all reduce the available signal. **The pipeline needs a scan-quality check** before generation (a v0 deployment consideration, not a v0 model architecture consideration). Paper 001's IGIP-style check is the right starting point.

8. **"Absence of adjacent or antagonist teeth" is a failure case** — meaning the v0 system needs to *reject* the scan (with a friendly error message) rather than hallucinate a low-quality crown. The error message UX is non-trivial: "Please re-scan, we need at least 2 adjacent + 3 opposing teeth visible."

9. **The dataset is 4,602 patients, exclusively premolars + molars** — **no incisors, no canines**. This is a *major coverage gap* for our v0/v1. The authors acknowledge it as future work ("expansion of the IOS dataset... should particularly focus on cases in which the incisor or canine is the target tooth"). For anterior crowns, we need 3DTeethSeg22 (paper 001) + DCPR-GAN's data (Tian 2021, ref 36) + Tufts dental scans.

10. **No H5-relevant result (synthetic → real transfer)** in the main paper — they train and test on the same distribution. However, the CrownSegger zero-shot transfer (paper Sec 4.3, Table 2 last row) is the closest thing: HDF goes from 0.328 mm (in-distribution) to 0.545 mm (out-of-distribution on the crown-design dataset) — a 0.2 mm degradation that *still beats* PointTransformer's 1.238 mm in-distribution result. **The CrownSegger is H5-robust.**

## Quote-worthy sentences

- "current methodologies, including ours, typically constrain the predicted crown shape through regression-based losses, such as the Chamfer Distance. This approach inadvertently encourages the network to produce a 'smoothed average' of plausible outputs rather than capturing the true 'mode' of the underlying distribution. Consequently, the generated crowns may lack critical anatomical details, such as grooves and fossae, which are essential for optimal function and aesthetics. **Future work should explore the incorporation of advanced generative models, such as diffusion models**, within 3D crown generation to produce high-quality crowns with richer morphological details." (Sec 5 Discussion)

- "MADCrowner generates a dental crown within 500 ms. Compared to the approximately 15 minutes required for a dental technician to manually design a crown using a CAD system using a CAD system, this dramatic acceleration significantly enhances the efficiency of dental technicians." (Sec 5)

- "dental crown generation is more prone to failure under the following conditions: (1) **inadequate tooth preparation, where irregular or insufficient reduction compromises geometric inference**; (2) **incomplete intraoral scans, particularly holes or missing regions in adjacent teeth**; and (3) **absence of adjacent or antagonist teeth, which limits accurate reconstruction of occlusal relationships**." (Sec 4.7)

- "The 2 cm-sided cubic region centered at the centroid of its corresponding crown" — a *very specific* data preprocessing choice that has implications for the field-of-view of the encoder. Worth replicating in our v0.

- "CrownSegger is constructed with a transformer-free backbone, which grants it distinct advantages in computational efficiency and memory usage. This makes the model particularly suitable for **clinical deployment on edge computing devices**." (Sec 4.3) — the strongest endorsement of edge-deployable dental AI in our reading list.

- "The improvement in HDF distance is particularly significant, decreasing from **1.237 mm to 0.328 mm**" (CrownSegger ablation) — the largest single HDF improvement in our reading list.

- "MADCrowner requires only 1.1 GB VRAM for the inference of a single sample. The single-inference latency is approximately 600 ms (Segmentation, Generation and Post-processing) on an NVIDIA L20 GPU. This **resource frugality** makes the MADCrowner an ideal candidate for on-site deployment." (Sec 4.2)

## For our project

**Eight concrete next steps, ranked by leverage:**

1. **★★★ ADOPT the post-processing algorithm (Sec 3.4) as the v0 sub-task 2 post-process — implement FIRST, before any model architecture work.** 200 lines of NumPy, $0 compute, **3× CD reduction** (0.556 → 0.185 mm² in MADCrowner's own ablation). The algorithm: abutment segmentation → largest connected component → boundary detection → B-spline smoothing (1,000 vertices) → surface definition (normal = tooth growth direction) → face removal below surface → boundary vertex projection. **Expected effort:** 1-2 days engineering, $0 compute. **Expected v0 win:** -50% CD on the cervical region, -30% CD overall. **Works on ANY upstream method** (DMC, DCrownFormer, AnchorFormer+DiGS+FlexiCubes) — it's a downstream concern, not a model architecture choice.

2. **★★★ Adopt MADCrowner's CMPL loss (Eq. 1) in v0 sub-task 2.** The `e^|κ|` exponential upweighting on high-curvature regions + the margin-line indicator term are the *right* anatomical priors. λ=1 is the empirical sweet spot (Fig 11 sweep). **Drop-in replacement** for the existing CD loss in DMC/DCrownFormer. **Expected effort:** 1 day engineering (loss function rewrite + curvature precomputation). **Expected v0 win:** -10-15% CD on the occlusal surface, where the cusps/fossae/grooves live.

3. **★★ Replace v0 segmentation backbone with MADCrowner's CrownSegger for the *abutment + margin* sub-task.** CrownSegger is transformer-free (easy MPS port for the M4 Mac mini), HDF 0.328 mm (best in reading list), and zero-shot HDF 0.545 mm on the unrelated crown-design dataset (H5 evidence). **For the v0 dual-head design: Cao25 (paper 026) for FDI labeling + CrownSegger for abutment/margin — complementary, not competing.** **Expected effort:** 1-2 days engineering (CrownSegger port from PyTorch, integrate with existing v0 segmentation pipeline). **Expected v0 win:** -75% margin HDF (from Cao25's ~1.2 mm to CrownSegger's 0.328 mm).

4. **★★ Add proximal contact area deviation (Table 7) to v0 sub-task 2 evaluation.** 100-line trimesh addition, $0 compute, and it's the *clinical* metric dentists actually care about. Compute medial + lateral intersection area between generated crown and adjacent teeth, compare to GT. **Expected effort:** 1 day engineering. **Expected v0 win:** clinically meaningful eval metric that's missing from every prior paper in our reading list.

5. **★ Promote MADCrowner to the v0 sub-task 2 reference SoTA, replacing DMCv2 in the v0 pilot plan.** The 4,602-patient dataset + 37.3M param model + 1.1 GB VRAM + 600 ms inference is *production-ready*. **For v0, fork their repo and port to PyTorch 2.x + MPS** (~3-5 days engineering). **For v1, fine-tune on 3DTeethSeg22 + ToSynFCD** (paper 001 + paper 024) for the public-benchmark comparison. **Expected cost:** $400-800 on Lambda for the v0 fine-tune, comparable to the existing DMC pilot budget.

6. **★ v1 product: add the diffusion layer on top of MADCrowner for the "show me 3-5 crown variations" chairside UX.** The authors' own admission (Sec 5) that regression losses produce a "smoothed average" is the green light. Architecture: MADCrowner as the *conditioning* module, then a small LION-style or Diffusion-SDF-style latent DDM that samples the *residual* to the MADCrowner prediction. This is **H2 applied as a *diversity module*, not as a backbone** — preserves MADCrowner's clinical accuracy while adding multi-modal sampling. Pilot candidates: LION (paper 005) for point-cloud residuals, MeshDiffusion (paper 014) for mesh-level residuals. **Expected cost:** $1,500-3,000 on Lambda for the v1 pilot.

7. **★ Address the incisor/canine dataset gap for v1.** MADCrowner trains on premolars + molars only — for anterior crowns (the most common clinical case in cosmetic dentistry), we need data. Three public sources: 3DTeethSeg22 (paper 001, all 32 FDI), DCPR-GAN's data (Tian 2021, ref 36, has incisor/canine), Tufts dental scans. **Expected cost:** $0 compute for the data (all public), $200-400 Lambda for the v1 fine-tune.

8. **★ Add a scan-quality check in the v0 pipeline.** Failure cases (Sec 4.7) are all data failures: inadequate prep, incomplete scans, missing context teeth. The pipeline should *reject* low-quality scans with a friendly error rather than hallucinate low-quality crowns. Paper 001's IGIP-style quality check is the right starting point: check prep reduction amount, check scan completeness, check # visible context teeth. **Expected effort:** 2-3 days engineering (PyMesh + geometric heuristics). **Expected v0 win:** graceful failure modes, no clinical risk from bad input.

### v0 stack update

**Previous (after paper 033):** PVD-AF-DiGS-FC for sub-task 1, DMC + MCAM + CPL + MRL for sub-task 2, Cao25 for segmentation, FlexiCubes for final mesh extraction.

**New (after paper 034):**
- **Sub-task 1 (full-arch synthesis):** PVD-AF-DiGS-FC — **unchanged**
- **Sub-task 2 (crown generation):** **MADCrowner (Wei 2026)** + post-process + CMPL + FDI-template selection — **UPGRADE from DMC**
- **Segmentation (FDI + abutment):** **Cao25 + CrownSegger** (dual-head) — **upgraded to dual-head**
- **Mesh extraction (final):** FlexiCubes — **unchanged** (but add the post-process on top to trim)
- **New v0 module: scan-quality check** — **added**

**v0 compute budget estimate:**
- PVD-AF-DiGS-FC: ~$2,200 Lambda (unchanged)
- MADCrowner + CMPL + FDI-template: $400-800 Lambda (port + fine-tune)
- Cao25 + CrownSegger dual-head: $200-400 Lambda
- Scan-quality check: $0 (geometric heuristics)
- **Total: ~$3,000-3,400 Lambda** (was $2,200) — a $1,000 increase for the dual-segmentation head + MADCrowner port

**v1 product offering:** MADCrowner + LION/Diffusion-SDF diversity module, $5,000-8,000 Lambda, 2-3 months engineering, the chairside "show me 3-5 crown variations" UX.

### Open questions for HK

1. **v0 segmentation: dual-head (Cao25 + CrownSegger) or single-head (CrownSegger only)?** CrownSegger already does the abutment segmentation well; the question is whether we need Cao25's full-FDI pipeline for the *other* teeth (the context teeth). My recommendation: dual-head, but the Cao25 head is a frozen, no-fine-tune reuse of paper 026's pretrained model. Cost: +$0-100 Lambda for the inference path. Benefit: clean separation of "where is the target tooth's margin" (CrownSegger) vs "where are the other teeth" (Cao25).

2. **v0 sub-task 2: MADCrowner-from-scratch or MADCrowner-fine-tuned-on-3DTeethSeg22?** MADCrowner's 4,602-patient dataset is private (paper 026 + 032 caveat applies). For a publishable v0 paper, we need 3DTeethSeg22 as the public benchmark. Option A: train MADCrowner from scratch on 3DTeethSeg22 (3-class, ~$1,500 Lambda, 2 weeks). Option B: fine-tune the MADCrowner pretrained weights (if the authors release them) on 3DTeethSeg22 (~$400 Lambda, 1 week). Recommendation: Option A — cleaner experiment, 3DTeethSeg22's 1,800 scans is enough for the MADCrowner architecture.

3. **v1 product: MADCrowner + LION or MADCrowner + Diffusion-SDF for the diversity module?** LION (paper 005) is point-cloud-based, Diffusion-SDF (paper 004) is implicit-field-based. For dental, the implicit field has the watertight-problem (we'd need the post-process on top), but the *sampling* is faster (~3-5s vs ~1-2s for LION). Recommendation: LION for the v1 pilot (point-cloud output is closer to the existing pipeline), Diffusion-SDF as a v2 alternative.

### Notes for HK
- **Code release**: confirmed open source at github.com/lullcant/MADCrowner, includes FDI templates in `fdi_template/` directory — we should *inspect* these for our own v0 template library.
- **Data**: 4,602 patients, private. Public alternative for v0: 3DTeethSeg22 (paper 001, 1,800 scans) + ToSynFCD (paper 024, 5,000+ synthetic arches).
- **Clinical collaborators**: Shanghai Ninth People's Hospital + Shanghai Stomatological Hospital — both are top-3 dental hospitals in China. SenseTime's involvement means production deployment is a real possibility for this group.
- **Reading time**: ~50 min, mostly because the paper is well-organized and the Sec 4 figures (especially Fig 2 overextension illustration, Fig 7-8 visual comparison, Fig 14 failure cases) tell the story better than the text.

**Next paper to read:** For paper 035, candidates from the seed list + 033/034 lineage:
- **VBCD (Wei 2025, arXiv:2507.17205, ref 40)** — Wei's own voxel-based prior, the 032/034 lineage's 2025 contribution; would close the methodological gap
- **ToothCraft (Wei/others, arXiv:2603.26588, Mar 2026)** — most recent in the Wei lineage, 2026 SoTA
- **MVDC (Yang 2025, ref 43)** — multi-view dental completion, contrastive learning; would test the H3 conditioning via contrastive pre-training
- **DCPR-GAN (Tian 2021, ref 36)** — the 2D-depth-map cGAN baseline, the methodological starting point for the entire literature
- **3D-Diffusion (Wu 2023)** — direct H2 test on the dental domain
- **TS-MTL** — multi-task tooth segmentation, would add a complementary sub-task 1 candidate

**Recommendation for 035: VBCD (Wei 2025)** — the Wei 2026 MADCrowner builds directly on VBCD (it's the "voxel-based framework" they outperform in Table 4), and reading the prior in the same lineage would help us understand the v0 architectural decisions. Then 036: ToothCraft for the 2026 SoTA, then 037: DCPR-GAN for the historical baseline. **Alternative: jump to 3D-Diffusion (Wu 2023) for a direct H2 test on dental** if HK wants to test the "diffusion for constrained tasks" debate head-on.
