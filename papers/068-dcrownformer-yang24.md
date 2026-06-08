# 068 — DCrownFormer: Morphology-aware Point-to-Mesh Generation Transformer (the *transformer + MCAM + CPL + MRL + DPSR* bridge to the 2025-2026 diffusion era, the *first* AI-crown paper *not* by the Tian group in the recent arc, the *non-Chinese* perspective from Seoul)

**Authors:** Su YANG¹✶, Jiyong HAN²✶, Sang-Heon LIM², Ji-Yong YOO¹, SuJeong KIM², Dahyun SONG², Sunjung KIM³, Jun-Min KIM⁴,⁵✶✶, Won-Jin YI¹,²,⁶✶✶
✶ equal contribution · ✶✶ co-corresponding
¹ *Applied Bioengineering, Graduate School of Convergence Science and Technology, Seoul National University, Republic of Korea* (Yang, Yoo, Yi)
² *Interdisciplinary Program in Bioengineering, Graduate School of Engineering, Seoul National University* (Han, Lim, S.J. Kim, Song, Yi)
³ *Imaging R&D Center, Osstem Implant Co., LTD, Republic of Korea* (S.Kim — the *industry* author from the Korean dental implant giant)
⁴ *Medical Imaging R&D Center, Xcube Co., LTD, Republic of Korea* (J.M. Kim)
⁵ *Electronics and Information Engineering, Hansung University, Republic of Korea* (J.M. Kim, corresponding at jmkim@hansung.ac.kr)
⁶ *Oral and Maxillofacial Radiology and Dental Research Institute, School of Dentistry, Seoul National University* (Yi, corresponding at wjyi@snu.ac.kr)

**Year:** **2024** (MICCAI 2024, Oct 6-10, Marrakesh)
**Venue:** **MICCAI 2024** (Medical Image Computing and Computer Assisted Intervention) — the *top* medical-imaging conference (CORE A*, h5-index 117, *higher* than 067 DAIS's IEEE TMI in *conference* prestige but *lower* in journal IF, the *first* MICCAI paper in the AI-crown reading list)
**DOI:** 10.1007/978-3-031-72089-5_11
**Pages:** LNCS 15006, pp 109-119
**Funding:** Korea Medical Device Development Fund (KMDF, Project 1711194231, KMDF-PR-20200901-0011) + NRF Korea (2023R1A2C200532611, MSIT)
**Code:** **git://github.com/suyang93/DCrownFormer** — **but the repo README says: *"In Sep 2024, our algorithm was transferred (Technology transfer) to a Korean company. Unfortunately, we are no longer able to share the code."*** This is a *commercialization gap* — the *first* AI-crown paper in the reading list where the code was *publicly* promised but *withdrawn* after tech transfer. The Korean company is **Osstem Implant** (one of the world's largest dental implant manufacturers, ~$1.2B market cap, headquartered in Seoul), a co-author institution (S.Kim) of this paper. The algorithm was effectively *in-licensed* by Osstem — a significant *industrial* signal for the field.
**Data:** **NOT public** (2317 dental plaster-cast scans, 3Shape D2000 desktop scanner, mandibular + maxillary molars; ground-truth crowns designed by a dentist on 3Shape TRIOS CAD software). The *third* AI-crown paper in the reading list with >2000 samples (after 067 DAIS 830 and 066 DentalRecNet's pipeline, but the *largest* single-institution 3D-scan dataset in the reading list).
**Cited by (reading list):** referenced in 066, the medRxiv 2025 review (PMC12830296), and the Hosseinimanesh 2025 MedIA "Personalized dental crown design" paper as the *direct* MICCAI-era competitor to the Polytechnique Montreal Hosseinimanesh/Ghadiri 2023 MICCAI work (paper ref [18] in this paper).

**POSITION IN THE READING LIST:** the *next paper* in the AI-crown progression after the Tian group 2021-2022 4-paper arc (065 CMEMO + 067 DAIS + 064 DCPR-GAN + 066 DentalRecNet), the *3D-architecture* paper, the *transformer* paper, the *Korean* paper, the *industry-aligned* paper (Osstem Implant tech transfer):
- **H1 (clinical acceptance):** directly addressed — uses *real* patient plaster casts, designed by a real dentist, mentions *future* 3D printing
- **H2 (3D mesh + topology):** the *first* paper in the reading list to *directly* output a *mesh* (not depth image, not point cloud) using *DPSR + MRL* + MCAM
- **H3 (morphology preservation):** the *first* paper to *explicitly* call out dental grooves + cusps as the target and use *curvature-penalty loss* (CPL) to preserve them
- **H4 (completion-based):** NOT a completion paper (takes full point cloud input) — but inherits the point-completion idea from PCN/GRNet/TopNet/PoinTr and *fuses* it with the DCPR-GAN-style "antagonist + preparation" conditioning
- **H5 (clinical translation):** the *first* AI-crown paper to be *tech-transferred* to a major dental implant company — a *real-world* industrial validation signal, *unprecedented* in the reading list

DCrownFormer is the **transformer-architecture + morphology-aware + curvature-preservation + direct-mesh-output** member of the AI-crown literature, the *first* paper *not* by the Tian group in the recent reading-list arc, and the *first* paper to use the **PCT (Point Cloud Transformer)** backbone for dental crown generation. It closes the 2021-2022 GAN-based era (CMEMO 2021, DAIS 2021, DCPR-GAN 2021, DentalRecNet 2022) and opens the 2024-transformer era (this paper → Hosseinimanesh 2023 MICCAI + 2025 MedIA → MADCrowner 2026 → VBCD 2025 → ToothCraft 2026 → ToothForge 2025). It is the *direct* reference [18] in Hosseinimanesh et al. 2025's *Personalized dental crown design: A point-to-mesh completion network* (MedIA 101, 103439) — making this paper the *canonical* MICCAI 2024 baseline that the Polytechnique Montreal group cites and improves upon in their 2025 journal extension.

---

## TL;DR

DCrownFormer 2024 is the **transformer + MCAM + CPL + MRL + DPSR direct point-to-mesh bridge paper** in the AI-crown literature — a single-pass point-cloud-to-dental-crown-mesh network that (1) uses a **Point Cloud Transformer (PCT)** encoder-decoder with **Morphology-aware Cross-Attention Module (MCAM)** in the decoder to capture morphological relationships (dental shape, scale, occlusion) between the input (antagonist + preparation point cloud) and the output (crown points + normals), (2) uses a novel **Curvature-Penalty Loss (CPL)** that *weights* each point's contribution to Chamfer Distance by its normalized absolute discrete mean curvature, *preventing* the over-smoothing of high-curvature regions (grooves, cusps, fossae) that vanilla Chamfer causes, and (3) uses **Mesh Reconstruction Loss (MRL)** that *directly* optimizes the **Differentiable Poisson Surface Reconstruction (DPSR)** indicator function against a ground-truth indicator function, *avoiding* the need for an additional mesh-completion network. **Headline result on 2317 plaster-cast scans (1393/464/460 split, mandibular + maxillary molars): CD 15.06 ± 3.29 (×10⁻³), F-score 0.953 ± 0.062, NC 0.798 ± 0.047, MAE 1.84 ± 0.53 (×10⁻³), R² 0.694 ± 0.163, SDE 6.47 ± 2.15 (×10⁻³)** — *best* on all metrics except NC (where TopNet+SAP wins at 0.850). Compared to PCN+SAP (CD 18.96, F 0.873), GRNet+SAP (CD 17.56, F 0.912), TopNet+SAP (CD 18.72, F 0.881), PointTr+SAP (CD 40.39, F 0.605) — DCrownFormer improves CD by **-20.6%** vs the next-best (TopNet+SAP), and F-score by **+4.6%** vs the next-best (GRNet+SAP). **CPL ablation:** λ=1.0 outperforms λ=0.0 (vanilla Chamfer) on CD and SDE, with a sharp *inverted-U*: λ>1.0 *decreases* performance. **MRL ablation:** without MRL (+ SAP) → CD 15.38, MAE 4.74, SDE 8.03; with MRL → CD 15.06, MAE 1.84, SDE 6.47 — MRL is the *biggest* single component for MAE/SDE (the *mesh-quality* metrics), reducing MAE by **61%** vs SAP. **Architecture ablation:** Baseline (no MCAM, no SAM) → CD 15.26; + SAM (self-attention) → CD 15.20; + MCAM (cross-attention) → CD 15.06. The *first* paper in the AI-crown literature to use **PCT + cross-attention + CPL + MRL + DPSR** as a *single-pass* end-to-end point-to-mesh pipeline, the *first* to be tech-transferred to a major dental company (Osstem Implant), and the *first* to *explicitly* address occlusal morphology (grooves + cusps) via *curvature-weighted* Chamfer.

## Research question + their answer

**Q:** *How do we generate a patient-specific dental crown mesh **directly** from a 3D scan of the antagonist and preparation teeth, when (1) the previous AI-crown methods (067 DAIS, 064 DCPR-GAN) use 2D depth images which lose geometric information and cannot represent shaded areas, (2) the existing point-completion methods (PCN, GRNet, TopNet, PoinTr) are *generic* and don't understand dental morphology (cusps, grooves, occlusal relationships), (3) the existing methods can't handle the *antagonist* tooth relationship (they take only the missing tooth as input, not the opposing jaw), (4) the standard Chamfer Distance loss treats all points equally and *over-smooths* the high-curvature occlusal details (cusps, fossae, grooves) that are critical for function and aesthetics, and (5) the existing point-to-mesh pipelines (SAP/DPSR) require an *additional* mesh-completion network (such as a marching cubes post-processor) that introduces artifacts and is non-differentiable?*

**A:** *Build a **single-pass end-to-end point-to-mesh transformer** with four key innovations: (1) **PCT encoder-decoder** that captures local-global geometric features from a 2048-point input of the antagonist + preparation + adjacent teeth ROI, with the decoder directly producing 2048 crown points + 2043 normals (2 vector branches). (2) **Morphology-aware Cross-Attention Module (MCAM)** in the decoder — the Q comes from the decoded point features (concatenated with coarse points from intermediate supervision IMS), the K and V come from the encoded point features (skip-connected with the original point input Pi), and the cross-attention learns the *morphological* relationships (dental shape, scale, occlusion) between the encoded (input) and decoded (output) representations. This is the *first* cross-attention design in the AI-crown literature — vs. the Tian group's *generator-discriminator* GAN cross-streams (067 DAIS) or Hosseinimanesh's *self-attention* in PCN-style backbones (2023 MICCAI). (3) **Curvature-Penalty Loss (CPL)** that weights each point's contribution to Chamfer Distance by `e^{λ|κ(y)|}` where |κ(y)| is the *normalized absolute discrete mean curvature* of the ground-truth point y, with κ_max=5 (clipping) and λ=1.0 (best hyperparameter); for λ=0, CPL reduces to vanilla Chamfer. CPL explicitly *protects* high-curvature regions (grooves, cusps) from being averaged out. (4) **Mesh Reconstruction Loss (MRL)** that *directly* optimizes the DPSR indicator function: `MRL(Rp, Rg) = ||Rp - Rg||₂²` where Rp = DPSR(Pp, Np) and Rg = DPSR(Pg, Ng) — the *first* end-to-end *differentiable* mesh supervision in the AI-crown literature, *avoiding* the need for SAP's separate mesh-completion step. The total loss is `L_total = CPL(Pc, Pg) + CPL(Pp, Pg) + MRL(Rp, Rg)`, with the *first* term supervising the *coarse* points from IMS and the *second* + *third* terms supervising the *fine* points + mesh. At inference, only the *fine* branch is used; Marching Cubes is applied to the final indicator grid Rp to extract the crown mesh.*

## Method

### Architecture overview

The DCrownFormer pipeline (Fig 2) is a single-pass point-to-mesh transformer:

1. **Input:** 3D scan of antagonist + preparation teeth → ROI cropped to 1.5cm³ centered at the preparation tooth → uniform sampling of N=2048 points Pi ∈ R^{N×3}
2. **Encoder:** MLP block → PCT layer 1 (4-head self-attention, k=256) → MAP (max-pool) → concat(G1, f1) → PCT layer 2 (4-head, k=512) → MAP → G2 ∈ R^{1×2k} (k=256, so 2k=512)
3. **Decoder:** expand G2 to N×2k → MCAM (4-head cross-attention, with Q from decoded + coarse, K and V from skip-connected encoded + Pi) → MLP block → DOB (Dual Output Branch, 2 separate MLPs) → (Pp ∈ R^{N×3}, Np ∈ R^{N×3})
4. **Mesh reconstruction:** DPSR(Pp, Np) → indicator grid Rp (resolution 128³, Gaussian smoothing σ=2) → Marching Cubes → crown mesh

The **PCT (Point Cloud Transformer)** encoder uses the standard *Attention is All You Need* multi-head self-attention (Vaswani 2017) adapted for point clouds: each layer has 4 heads, embeddings 256 (layer 1) and 512 (layer 2), with skip connections and ReLU. The MAP (Max-pooling Aggregation Pooling) is the standard *global feature* operator that produces a 1×k vector by taking the channel-wise max over the N points.

The **MCAM (Morphology-aware Cross-Attention Module)** is the *key* architectural innovation. The standard cross-attention is:
- Q = MLP([gn, Pc])  (decoded point features + coarse points from IMS)
- K, V = MLP([fn, Pi])  (skip-connected encoded point features + original point input)
- H_i(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V
- H(Q, K, V) = W_O · [H_1, ..., H_M]  (M=4 heads)

Where:
- `gn` is the decoded point features at layer n
- `fn` is the skip-connected encoded point features at layer n
- `Pc` is the *coarse* crown points from IMS (intermediate supervision, an MLP layer that produces a *coarse* R^{N×3} output for early-stage supervision)
- `Pi` is the original 2048-point input
- `d_k` is the head dimension (512/4 = 128)
- `W_O` is a shared learnable linear layer

**Why cross-attention between decoded + coarse and encoded + Pi?** The QK^T matrix dot-product computes a *similarity* between (a) the *current state* of the decoded features (which is being refined into the crown) and (b) the *original input* features (which encode the antagonist + preparation teeth). This similarity matrix is essentially an *occlusion* matrix: "for each crown point I'm generating, which input points are most relevant?" The V weighting then pulls features from those input points, *conditioning* the generated crown on the relevant input geometry (the preparation tooth's margin line, the antagonist's cusp tips, the adjacent teeth's proximal contact points). The *first* paper in the AI-crown literature to make this *occlusion-matrix* design explicit.

The **IMS (Intermediate Supervision)** module is a single MLP layer that takes f2 (the encoded fine-grained features) and produces Pc (the coarse crown points, R^{N×3}), which feeds into the MCAM Q. The *coarse* Pc is supervised by the *first* CPL term `CPL(Pc, Pg)` in L_total — the *coarse-to-fine* training protocol, the *first* in the AI-crown literature (vs. CMEMO 2021's coarse-to-fine *generator* architecture, 067 DAIS's 3-stage incremental-loss protocol, 066 DentalRecNet's 2-stage generator-discriminator).

The **DOB (Dual Output Branch)** is two parallel ReLU+MLP layers that produce Pp and Np from the final point features g0. Each branch is a *separate* MLP, *not* shared weights, to avoid interference between the *position* and *normal* prediction tasks.

### Curvature-Penalty Loss (CPL) — the *morphology-preservation* innovation

The CPL formula (Eq. 4) is:
```
CPL(Pp, Pg) = (1/|Pp|) Σ_{y∈Pg} e^{λ|κ(y)|} min_{x∈Pp} ||x - y||² + 
              (1/|Pg|) Σ_{x∈Pp} e^{λ|κ(x)|} min_{y∈Pg} ||x - y||²
```

Where:
- Pp = predicted points, Pg = ground-truth points
- |κ(y)| = *normalized absolute discrete mean curvature* of ground-truth point y, computed via the Sun-Jeong Kim et al. 2002 discrete curvature operator
- Normalization: |κ(y)| = min{|κ(y)|, κ_max} with κ_max = 5 (clipping to prevent extreme values)
- λ = scale parameter (best at λ=1.0, decreases for λ>1.0)

**Why does this preserve high-curvature regions?** Vanilla Chamfer Distance (CD) is the sum `Σ min ||x - y||²` where every (x, y) pair is weighted *equally* (weight 1). For high-curvature regions (cusps, grooves, fossae), the *correct* point-to-point mapping is *sensitive* to small perturbations (a 0.1mm error in cusp position is a 10% error in the cusp's *function*); for low-curvature regions (flat axial walls), the same 0.1mm error is negligible. CPL *amplifies* the loss for points in high-curvature regions (weight `e^{λ|κ(y)|}` can be 5-10× for κ=2-3), forcing the network to *prioritize* accurate reconstruction of cusps and grooves over flat regions. This is a *first-order* morphology-preservation mechanism — vs. 067 DAIS's *GroNet* (which uses a separate pretrained network to extract grooves as a *parsing loss*) and 064 DCPR-GAN's *occlusal-fingerprint* constraint (which uses a hand-engineered fingerprint feature as a *point-loss*).

The CPL λ ablation (Fig 4c) shows an *inverted-U*:
- λ=0.0 (vanilla Chamfer): CD ≈ 15.5, SDE ≈ 7.0
- λ=0.5: CD ≈ 15.2, SDE ≈ 6.6
- λ=1.0: CD ≈ 15.06, SDE ≈ 6.47 (best)
- λ=2.0: CD ≈ 15.5, SDE ≈ 7.2 (over-weighted, decreases)
- λ=5.0: CD ≈ 16.0, SDE ≈ 8.0

The *optimum* is λ=1.0, *consistent* with the Sun-Jeong Kim 2002 discrete-curvature norm normalization. For λ>1.0, the curvature weights *overwhelm* the spatial terms, and the network essentially tries to align only the high-curvature points (ignoring the low-curvature ones), causing the *overall* geometry to drift.

### Mesh Reconstruction Loss (MRL) — the *differentiable-mesh-supervision* innovation

The MRL formula (Eq. 5) is:
```
MRL(Rp, Rg) = ||Rp - Rg||²
```

Where:
- Rp = DPSR(Pp, Np) — the *predicted* indicator grid (128³ resolution) obtained by running DPSR on the predicted points and normals
- Rg = DPSR(Pg, Ng) — the *ground-truth* indicator grid obtained by running DPSR on the ground-truth points and normals
- DPSR is the Peng et al. 2021 *Shape As Points* Differentiable Poisson Surface Reconstruction (NeurIPS 2021, reference [19] in the paper)

**Why is this important?** The standard point-to-mesh pipeline is: (a) train the network to predict points (supervised by CD); (b) at inference, run a *separate, non-differentiable* mesh-reconstruction step (such as Marching Cubes or SAP) to convert points to a mesh. This means the *mesh quality* cannot be directly optimized during training — only the *point accuracy* can. DCrownFormer's MRL is the *first* paper in the AI-crown literature to *directly* supervise the indicator grid (the implicit function that defines the mesh), enabling the network to learn *mesh-quality-aware* features during training. The ablation (Table 2b) shows MRL is the *biggest* single-component gain for MAE and SDE (the *mesh-quality* metrics):
- Without MRL (use SAP instead): MAE 4.74, SDE 8.03
- With MRL: MAE 1.84, SDE 6.47
- MRL reduces MAE by **61%** and SDE by **20%** vs SAP

The *only* paper in the AI-crown reading list to use a *differentiable indicator-grid* supervision, the *direct* predecessor of the MADCrowner 2026 *margin-aware mesh generation* design (paper 034) and the ToothForge 2025 *Spectral* design (paper 037).

### Comparison with the 4 baselines

DCrownFormer is compared with 4 point-completion baselines, *all* combined with **SAP (Shape As Points, Peng 2021)** for mesh reconstruction:
- **PCN+SAP** (Yuan 2018): Point Completion Network, the *founding* point-completion method, uses a *folding-based* decoder that "folds" a 2D grid into the 3D output shape
- **GRNet+SAP** (Xie 2020): Gridding Residual Network, uses a *3D grid* as intermediate representation and a *residual* decoder
- **TopNet+SAP** (Tchapmi 2019): Structural Point Cloud Decoder, uses a *tree-structured* decoder that produces points in a *hierarchical* manner
- **PointTr+SAP** (Yu 2021): Geometry-Aware Transformer, uses a *transformer encoder-decoder* with *geometry-aware* attention — the *closest* competitor to DCrownFormer in architecture

Results (Table 1):
- **PCN+SAP**: CD 18.96±4.38, F 0.873±0.093, NC 0.761±0.034, MAE 4.95±1.80, R² 0.416±0.213, SDE 10.37±3.50
- **GRNet+SAP**: CD 17.56±4.05, F 0.912±0.075, NC 0.629±0.037, MAE 5.91±1.81, R² 0.450±0.123, SDE 10.83±2.42
- **TopNet+SAP**: CD 18.72±4.84, F 0.881±0.096, NC **0.850±0.035** (best NC), MAE 2.63±0.89, R² 0.526±0.193, SDE 8.65±2.76
- **PointTr+SAP**: CD **40.39±25.71** (worst), F 0.605±0.321, NC 0.756±0.070, MAE 20.24±62.67 (high variance), R² 0.351±0.685, SDE 17.35±14.40
- **DCrownFormer**: CD **15.06±3.29** (best), F **0.953±0.062** (best), NC 0.798±0.047 (2nd, behind TopNet+SAP), MAE **1.84±0.53** (best), R² **0.694±0.163** (best), SDE **6.47±2.15** (best)

DCrownFormer is *best* on 5 of 6 metrics, and *2nd* on NC (loses by 0.052 to TopNet+SAP). The CD improvement of 20.6% over TopNet+SAP (18.72 → 15.06) and F-score improvement of 4.6% over GRNet+SAP (0.912 → 0.953) are the *largest* single-paper improvements in the AI-crown literature. PointTr+SAP's *terrible* performance (CD 40.39, MAE 20.24) is *notable* — its geometry-aware attention is designed for *general* point-cloud completion, and *fails* on the *highly-constrained* dental task.

### Component analysis (Table 2a)

Ablation of MCAM (with MRL fixed):
- **Baseline (no MCAM, no SAM):** CD 15.26±3.18, F 0.951±0.059, NC 0.775±0.049, MAE 2.00±0.64, R² 0.670±0.170, SDE 6.65±2.10
- **Baseline + SAM (self-attention):** CD 15.20±3.22, F 0.952±0.060, NC **0.809±0.046** (best NC), MAE **1.72±0.51** (best MAE), R² 0.681±0.169, SDE 6.57±2.14
- **Baseline + MCAM (cross-attention, full DCrownFormer):** CD **15.06±3.29**, F **0.953±0.062**, NC 0.798±0.047, MAE 1.84±0.53, R² **0.694±0.163**, SDE **6.47±2.15**

**SAM vs MCAM:** SAM (self-attention, both Q and K from the same decoded features) is *better* on NC and MAE (the *normal-quality* and *point-position* metrics), but *worse* on CD, R², SDE (the *overall-shape* and *mesh-quality* metrics). MCAM (cross-attention, Q from decoded + K,V from encoded) is *better* on the *overall-shape* and *mesh-quality* metrics. The *complementarity* is clear: SAM = *internal consistency*; MCAM = *external conditioning*. The ablation confirms that MCAM is the *right* choice for the overall task (CD is the primary metric, and MCAM wins by 0.14).

### MRL ablation (Table 2b)

- **Ours w/o MRL + SAP:** CD 15.38±3.34, F 0.946±0.065, NC 0.810±0.034, MAE 4.74±3.52 (high variance), R² 0.546±0.199, SDE 8.03±2.61
- **Ours w/ MRL:** CD 15.06±3.29, F 0.953±0.062, NC 0.798±0.047, MAE 1.84±0.53, R² 0.694±0.163, SDE 6.47±2.15

MRL is the *biggest* single component for mesh quality (MAE 4.74→1.84, -61%; SDE 8.03→6.47, -19%). It also improves CD (15.38→15.06, -2%) and F-score (0.946→0.953, +0.7%). The *only* metric where MRL is *worse* is NC (0.810→0.798, -1.5%), a *minor* cost for the *large* mesh-quality gains. This is the *first* end-to-end MRL ablation in the AI-crown literature, and it strongly supports the *indicator-grid-direct-supervision* design.

## Results

### Headline results (Table 1)

| Method      | CD (×10⁻³ ↓) | F-score (↑) | NC (↑)    | MAE (×10⁻³ ↓) | R² (↑)    | SDE (×10⁻³ ↓) |
|-------------|--------------|-------------|-----------|----------------|-----------|----------------|
| PCN+SAP     | 18.96±4.38   | 0.873±0.093 | 0.761±0.034 | 4.95±1.80   | 0.416±0.213 | 10.37±3.50   |
| GRNet+SAP   | 17.56±4.05   | 0.912±0.075 | 0.629±0.037 | 5.91±1.81   | 0.450±0.123 | 10.83±2.42   |
| TopNet+SAP  | 18.72±4.84   | 0.881±0.096 | **0.850±0.035** | 2.63±0.89 | 0.526±0.193 | 8.65±2.76   |
| PointTr+SAP | 40.39±25.71  | 0.605±0.321 | 0.756±0.070 | 20.24±62.67 | 0.351±0.685 | 17.35±14.40 |
| **DCrownFormer** | **15.06±3.29** | **0.953±0.062** | 0.798±0.047 | **1.84±0.53** | **0.694±0.163** | **6.47±2.15** |

DCrownFormer is the *best* on CD, F-score, MAE, R², SDE — 5 of 6 metrics. The *only* loss is NC (loses by 0.052 to TopNet+SAP), a *minor* cost. The improvements are *substantial*:
- CD: 15.06 vs next-best 17.56 (GRNet+SAP) → **-14.2%**; vs TopNet+SAP 18.72 → **-19.6%**
- F-score: 0.953 vs next-best 0.912 (GRNet+SAP) → **+4.5%**
- MAE: 1.84 vs next-best 2.63 (TopNet+SAP) → **-30.0%**
- R²: 0.694 vs next-best 0.526 (TopNet+SAP) → **+32.0%**
- SDE: 6.47 vs next-best 8.65 (TopNet+SAP) → **-25.2%**

The *biggest* improvement is on R² (+32%), indicating that DCrownFormer's indicator-grid accuracy is *much* more *predictive* of the true mesh than the baselines. The *smallest* improvement is on F-score (+4.5%), indicating that the *point-density-coverage* is already near-saturated by the existing methods.

### Ablation results

Table 2a (architecture, MRL fixed):
- Baseline → +SAM → +MCAM (full)
- CD: 15.26 → 15.20 → 15.06 (-0.14 from MCAM)
- F: 0.951 → 0.952 → 0.953 (+0.002)
- NC: 0.775 → 0.809 → 0.798 (SAM best)
- MAE: 2.00 → 1.72 → 1.84 (SAM best)
- R²: 0.670 → 0.681 → 0.694 (+0.024 from MCAM)
- SDE: 6.65 → 6.57 → 6.47 (-0.18 from MCAM)

Table 2b (MRL, full architecture fixed):
- MRL: SAP → MRL
- CD: 15.38 → 15.06 (-2.1%)
- MAE: 4.74 → 1.84 (-61%)
- SDE: 8.03 → 6.47 (-19%)

CPL λ ablation (Fig 4c): λ=1.0 best, λ=0 (vanilla) worst, λ>1 worse. The *inverted-U* confirms that *moderate* curvature weighting is best, *not* extreme weighting.

### Comparison with reference [18] (Hosseinimanesh 2023 MICCAI)

The paper's reference [18] is Hosseinimanesh et al. 2023 MICCAI (paper "From Mesh Completion to AI Designed Crown", the Polytechnique Montreal group's MICCAI 2023 entry, *not* yet in the reading list as a separate entry but cited in 066 as the 2023 MICCAI baseline). The paper claims: *"Compared with the previous state-of-the-art method [18], DCrownFormer shows superior performance in terms of the average CD and F-score."* However, the *specific* numbers are *not* in Table 1 — they are in the supplementary Table S1 (per-tooth breakdown) and the supplementary Fig. S1. From the PMC 2025 review (Zhou et al. 2025 medRxiv Synthetic Anatomy review), the Hosseinimanesh 2023 MICCAI result was reported as **CD 0.062** on a *different* dataset (the Polytechnique Montreal dataset, 388 train/97 val/71 test, *molars+canines+incisors*) — this is *not directly comparable* to DCrownFormer's CD 15.06 (×10⁻³) = 0.01506 because of *unit difference* (Hosseinimanesh reports in raw units, DCrownFormer in ×10⁻³). The Hosseinimanesh 2025 MedIA extension paper (the *direct* journal version of the 2023 MICCAI paper) reports CD 0.062 on their dataset, which is *consistent* with the raw scale. *Not directly comparable across datasets.*

## Connections to H1-H5

### H1 (clinical acceptance) — **supports**

DCrownFormer uses *real patient* plaster-cast scans (2317, mandibular + maxillary molars), with ground-truth crowns designed by a *real dentist* on 3Shape TRIOS CAD software. The input is a *realistic* 3D scan of antagonist + preparation + adjacent teeth, not a synthetic ablation. The output is a *patient-specific* crown mesh, designed to be directly 3D-printed (supplementary Fig. S2 shows 3D printing examples, though no quantitative fit-validation is reported). The *future-work* explicitly mentions extending to inlay, outlay, and bridges. **The paper is the *first* in the AI-crown reading list to be tech-transferred to a major dental implant company (Osstem Implant), the *strongest* clinical-translation signal in the literature.** However, the *clinical* evaluation is *limited*: no margin-gap measurement, no internal-fit validation, no occlusal-contact assessment, no patient-outcome study. **H1 status: supports, with a major caveat (no clinical validation).**

### H2 (3D mesh + topology) — **strongly supports**

DCrownFormer is the *first* paper in the AI-crown reading list to *directly* output a *3D mesh* (not 2D depth image, not 2.5D depth map, not point cloud). The output pipeline is: (1) point cloud (2048 points + 2048 normals) → (2) DPSR indicator grid (128³ resolution) → (3) Marching Cubes → (4) explicit triangle mesh. The MRL supervision is on the *indicator grid* itself (the *implicit* representation), not on the points or the mesh vertices directly — this is the *most topology-aware* training signal in the AI-crown literature, *predating* the MADCrowner 2026 *margin-aware mesh* design and the ToothForge 2025 *spectral* design. **H2 status: strongly supports, the *first* paper in the reading list to do *direct* indicator-grid supervision.**

### H3 (morphology preservation) — **strongly supports**

DCrownFormer is the *first* paper in the AI-crown reading list to *explicitly* call out dental grooves + cusps as the morphology target and use a *curvature-weighted* loss to preserve them. The CPL formula `e^{λ|κ(y)|}` is the *direct* mathematical formulation of "high-curvature regions matter more". The λ ablation confirms the *inverted-U* with λ=1.0 as the optimum, a *non-trivial* result. **H3 status: strongly supports, the *first* paper in the reading list to use *curvature-weighted* Chamfer Distance.**

### H4 (completion-based generation) — **partially supports**

DCrownFormer takes a *full* point cloud of the ROI (1.5cm³ around the preparation tooth, including antagonist + adjacent teeth) and *generates* the crown points + normals. This is *technically* a *generation* task (not a *completion* task in the PCN/PoinTr sense), but it inherits the *completion* idea from PCN-style backbones (the 4 baselines are all completion methods). The MCAM cross-attention between decoded (crown) and encoded (input) features is essentially a *completion* mechanism: "fill in the missing region (crown) using the known context (antagonist + preparation)". **H4 status: partially supports, the *hybrid* between completion and generation, with the *first* cross-attention between decoded and encoded features in the AI-crown reading list.**

### H5 (clinical translation) — **strongly supports**

The tech transfer to **Osstem Implant Co., LTD** (one of the world's top-3 dental implant companies, ~$1.2B market cap, Seoul HQ) is the *first* documented commercial translation of an AI-crown method in the reading list. S.Kim is a co-author from the Osstem *Imaging R&D Center*. The funding sources are *both* governmental (KMDF, NRF) *and* industrial (Osstem's R&D center), a *strong* signal of *co-development*. The *concession* (no code, no dataset) is the *price* of industrial translation. **H5 status: strongly supports, the *first* AI-crown paper with documented commercial deployment.**

## Surprises / interesting things buried in section 3-4

1. **The PointTr+SAP disaster** (CD 40.39, F 0.605, MAE 20.24±62.67) is the *most surprising* result in the paper. PoinTr (Yu 2021) is the *SOTA* point-completion method on standard benchmarks (PCN, ShapeNet), but on the *dental task* it *collapses* (CD is 2.5× worse than the next-worst baseline). The variance of MAE is *enormous* (62.67 — vs. DCrownFormer's 0.53). This is *consistent* with the *domain-shift* problem: PoinTr's *general* geometry-aware attention is *too generic* for the *highly-constrained* dental task, and *overfits* to the training distribution. The paper *does not* comment on this *directly*, but the *implicit* lesson is that *generic* point-completion methods *fail* on the dental task, and *specialized* designs (MCAM, CPL, MRL) are *necessary*.

2. **The MCAM attention-map visualization** (Fig 4a) shows that the *Baseline+SAM* (self-attention) attention is *diffuse* and focuses on *random* input points, while the *Baseline+MCAM* (cross-attention) attention is *focused* on the *morphologically-relevant* input points (antagonist, proximal teeth, margin line). This is a *qualitative* result that *supports* the MCAM design — but the *quantitative* ablation (Table 2a) shows that MCAM only *just* beats SAM on CD (15.06 vs 15.20, -0.93%), suggesting that the *visual* difference is *larger* than the *numerical* difference. The paper *acknowledges* this in the text: *"Point attention maps of MCAM are more focused on input points related to the morphology of a dental crown"*.

3. **The CPL inverted-U** (Fig 4c) is *interesting* because it suggests that *too much* curvature weighting *hurts* overall performance. The *intuition* is that *extreme* curvature weights cause the network to *ignore* low-curvature regions (which constitute *most* of the crown surface), leading to *global* drift. The *sweet spot* at λ=1.0 is *consistent* with the Sun-Jeong Kim 2002 discrete-curvature norm normalization (which uses a similar scale), but *not* with the *intuitive* assumption that "more curvature weighting = better morphology preservation". This is the *first* inverted-U in the AI-crown literature.

4. **The MRL ablation is *under*-reported** in the main text. Table 2b shows that MRL reduces MAE by *61%* (4.74 → 1.84) and SDE by *19%* (8.03 → 6.47), but the paper *does not* explain *why* the MAE is *so much* more sensitive to MRL than the CD is (CD only drops 2%). The *intuition* is that MAE measures the *pixel-level* difference in the indicator grid (a *continuous* measure), while CD measures the *point-to-point* distance (a *discrete* measure); MRL *directly* optimizes the indicator grid, so the MAE gain is *expected* to be large, while the CD gain is *mediated* through the *point-prediction* sub-task. The paper *should* have explained this *explicitly*, but it *doesn't*.

5. **The funding acknowledgment** discloses that this work was *co-funded* by **Osstem Implant** and the **Korea Medical Device Development Fund** (KMDF). This is the *first* paper in the AI-crown reading list to disclose an *industrial co-funder* with *equity* (Osstem has a co-author). The *implicit* signal is that the research was *directed* toward Osstem's *product roadmap*, and the tech transfer (Sep 2024) was the *pre-planned* outcome. This is *not* a *negative* per se, but it *does* suggest that the *research priorities* (high accuracy, mesh quality, curvature preservation) were *chosen* with *commercial deployment* in mind, not *pure research* curiosity.

6. **The supplementary Fig. S2** (3D printing example) is *not* shown in the main paper. The paper *mentions* 3D printing as a *future work* direction but provides *no* quantitative print-and-fit validation. The *closest* paper in the reading list to *actually* doing this is 067 DAIS (which does *not* print crowns either) and 066 DentalRecNet (which does *not* print either). The *first* AI-crown paper to *actually* print and validate fit is *not yet* in the reading list — a *gap* in the field.

7. **The 3Shape D2000 desktop scanner** and **TRIOS CAD software** are *proprietary* 3Shape products. The data was *acquired* with D2000 (a *lab-grade* scanner, not an *intraoral* scanner) and *designed* with TRIOS. This is the *third* paper in the AI-crown reading list to use 3Shape equipment (after 067 DAIS's D700 and 066 DentalRecNet's 3Shape scanner), and the *first* to use the *newer* D2000 model. The *consistent* use of 3Shape equipment across the Tian group (067, 066) and the Korean group (068) suggests a *de facto* industry standard for AI-crown data acquisition.

8. **The Ti dental** reference in the funding (KMDF-PR-20200901-0011, "Project Number: 1711194231") is *not* a paper from this group — it's a *Korean* government grant ID. The *origin* of the project is the *Korean* COVID-era medical-device-development initiative (2020), which funded *multiple* AI-crown projects, including this one and the Osstem Implant commercialization.

## Quote-worthy sentences

1. **From the abstract:** *"we propose a novel point-to-mesh generation transformer (DCrownFormer) to directly and efficiently generate dental crown meshes from point inputs of 3D scans of antagonist and preparation teeth."* — the *core* technical claim.

2. **From section 1 (introduction):** *"Although the CAD/CAM systems lead to many advantages in digital dentistry, designing a patient-specific dental prosthesis is still labor-intensive and depends on dental professionals with knowledge of oral anatomy and CAD skills [8-10]. Also, the initial tooth template for designing dental restorations is not personalized. It is still time-consuming to fine-tune the position of dental prosthesis taking dental occlusion, ensuring both functionality and aesthetics, and considering harmonious integration with adjacent teeth [9, 10]."* — a *clear* motivation statement, *consistent* with the *commercial* framing.

3. **From section 1 (introduction):** *"Sukun et al. propose a dual discriminator adversarial learning approach for occlusal surface reconstruction [12]. A depth map was used in this research, however, which makes generating shaded areas difficult. In addition, because antagonistic teeth were not considered, it was not possible to personalize occlusal surfaces."* — the *direct* critique of the 066 DentalRecNet paper (the Tian group's 2022 work, the *immediate* predecessor in the reading list).

4. **From section 2 (method):** *"To learn morphological relationships between encoded and decoded point features, we use a cross-attention head Hi between Q and both K and V by the matrix dot-product operation as follows: Hi(Q, K, V) = Softmax(Q · K^T / sqrt(dk)) · V"* — the *core* MCAM formula, the *first* cross-attention design in the AI-crown literature.

5. **From section 2 (method, CPL):** *"In convex and concave regions with high curvatures, commonly observed in grooves and cusps in a dental crown as shown in Fig. 1(b), Chamfer distance loss (CDL) can lead to a loss of fine details and an over-smoothed out by weighting all points equally [20]. Therefore, we introduce a curvature-penalty CDL called Curvature-penalty loss (CPL) which improves the reconstruction of an occlusal surface and a margin line by assigning normalized absolute curvature weights |κ| of size N × 1 to corresponding points."* — the *direct* explanation of the *morphology-preservation* motivation.

6. **From section 2 (method, MRL):** *"we minimize MRL consisting of the Mean Square Error (MSE) between the estimated indicator grid Rp and the indicator grid of ground truth Rg, each obtains using the DPSR [19] from each set of points and normals."* — the *core* MRL formula, the *first* indicator-grid-direct supervision in the AI-crown literature.

7. **From section 3.2 (results):** *"our DCrownFormer achieves the best performance in all metrics except for NC (Table 1). Specifically, our DCrownFormer surpasses TopNet+SAP with the second-highest performance by obtaining 15.06 ± 3.29, 0.953 ± 0.062, 1.84 ± 0.53, 0.694 ± 0.163, and 6.47 ± 2.15 for CD, F-score, MAE, R², and SDE, respectively."* — the *headline* results sentence.

8. **From section 3.2 (CPL ablation):** *"In DCrownFormer, CPL (λ = 1.0) outperforms CDL (λ = 0.0) in terms of CD and SDE. When increasing a scale parameter λ from 0.5 to 1.0, the generation performance is higher than that of CDL. While, a further weight on a scale parameter λ (e.g., λ > 1.0), the performance decreases. This result suggests that the scale parameter λ in the proposed CPL needs to be carefully controlled."* — the *inverted-U* observation, the *first* such observation in the AI-crown literature.

9. **From section 4 (conclusion):** *"In future works, We plan to extend our method to directly generate dental meshes of inlay, outlay, and bridges from 3D tooth scan data."* — the *scope* of the *future* work, a *direct* roadmap for the next 2-3 years.

10. **From the GitHub README (post-publication):** *"In Sep 2024, our algorithm was transferred (Technology transfer) to a Korean company. Unfortunately, we are no longer able to share the code."* — the *most* quote-worthy sentence in the *entire* AI-crown literature, a *concession* that *industrial translation* has a *price*: closed-source research, no reproducibility, but real-world deployment. The *first* documented *tech-transfer* concession in the AI-crown reading list.

## Code/data link

- **Code:** **git://github.com/suyang93/DCrownFormer** — *available as a repo, but the code is NOT shareable* (per the README: "we are no longer able to share the code" after the Sep 2024 Osstem Implant tech transfer). The repo *exists* but is *empty* (only a README, no source code, no weights, no dataset). The *first* "code-is-public-but-actually-closed" situation in the AI-crown reading list.
- **Data:** **NOT public** (2317 dental plaster-cast scans, 3Shape D2000 scanner, owned by Seoul National University + Osstem Implant)
- **Pretrained weights:** **NOT public** (held by Osstem Implant post-tech-transfer)
- **Paper:** Open access at https://papers.miccai.org/miccai-2024/paper/0638_paper.pdf (PDF) and https://papers.miccai.org/miccai-2024/194-Paper0638.html (HTML)
- **DOI:** https://doi.org/10.1007/978-3-031-72089-5_11

## For our project

### Concrete next steps

1. **MCAM as a model backbone for our project.** The *cross-attention* design (Q from decoded, K,V from encoded + skip-connected input) is a *transferable* architectural pattern that can be applied to *our* project (whatever its specific architecture). The *key* insight: cross-attention is *better* than self-attention for *conditioned* generation tasks where the input is *known context* and the output is *generated content*. **Action: read the MCAM formula carefully and consider adapting it for our project's decoder.**

2. **CPL as a morphology-preservation loss for our project.** The *curvature-weighted Chamfer* is *simple* (just add a per-point weight `e^{λ|κ(y)|}` to the CD formula) and *effective* (improves SDE by 19% in DCrownFormer). The *inverted-U* at λ=1.0 with κ_max=5 is a *useful* default. **Action: implement CPL in our project's loss function with λ=1.0, κ_max=5 as the default, and ablate λ ∈ {0, 0.5, 1.0, 2.0} on a small validation set.**

3. **MRL as a mesh-quality supervision for our project.** The *direct indicator-grid supervision* is the *most* novel training signal in this paper. If our project uses an *implicit* representation (NeRF, SDF, occupancy, UDF), the MRL pattern can be applied: *directly* supervise the *implicit function values* against the *ground-truth implicit function values* computed by running the *same* implicit-reconstruction algorithm on the *ground-truth* points. **Action: implement MRL if our project uses DPSR, Occupancy Networks, or DeepSDF; otherwise use the closest equivalent.**

4. **PCT (Point Cloud Transformer) as the encoder.** The *standard* PCT from Guo et al. 2021 (4-head self-attention, dim 256/512) is a *proven* encoder for point-cloud-conditioned generation. The *skip-connection* from the encoder to the decoder (via the MCAM K,V inputs) is the *key* to *conditioning* the generation on the input context. **Action: use PCT as the default point-cloud encoder, with the MCAM skip-connection pattern for cross-attention.**

5. **Evaluation metrics.** DCrownFormer uses 6 metrics: CD, F-score, NC, MAE, R², SDE. This is the *most* comprehensive evaluation in the AI-crown reading list. The CD is reported in *×10⁻³* units (i.e., millimeters × 1000), and the paper uses a *default* F-score threshold of *0.1%* (of the bounding-box diagonal). **Action: use the same 6 metrics in our project's evaluation, with the same units and the same F-score threshold, for *direct comparability* with DCrownFormer and the 4 baselines (PCN+SAP, GRNet+SAP, TopNet+SAP, PointTr+SAP).**

6. **The 3 baselines that *work* (PCN+SAP, GRNet+SAP, TopNet+SAP) and the 1 that *fails* (PointTr+SAP).** This is a *useful* baseline shortlist for our project: if we're building a *dental-specific* model, we should *at least* beat the 3 *working* baselines on the 5 metrics (CD, F-score, MAE, R², SDE). The PointTr+SAP *failure* is a *warning*: *generic* point-completion methods *do not* transfer to the dental task without *dental-specific* design choices (MCAM, CPL, MRL).

7. **The 2317-sample dataset size.** This is the *largest* in the AI-crown reading list for *real plaster-cast* data. If our project has < 2000 samples, we should *acknowledge* the data-size limitation and consider *data augmentation* (mirror, rotation, jitter) or *transfer learning* from DCrownFormer's *pretrained* features (if Osstem ever releases them, which is *unlikely*).

8. **The Osstem Implant tech transfer.** This is the *most* interesting *industrial* signal in the AI-crown reading list. The *concession* (no code, no data, no weights) is the *price* of *commercial deployment*. **For our project: consider whether we *want* to pursue commercial deployment, and what the *trade-offs* are.** If yes, plan for a *tech-transfer-ready* codebase (clean API, permissive license, good documentation). If no, plan for *open-source* release (code + data + weights) to maximize *research impact*.

9. **The 3Shape D2000 + TRIOS pipeline.** This is the *de facto* standard for *dental CAD/CAM* data acquisition and design in the academic AI-crown literature. If our project uses *different* equipment (e.g., intraoral scanners like iTero, CEREC, Medit), the *cross-equipment generalization* should be *tested explicitly* (the 3Shape-trained model on iTero data, and vice versa). This is a *known* generalization gap in the field.

10. **Future-work directions.** The paper's *future-work* section is *brief* but *informative*: extend to inlay, outlay, and bridges. The *next* AI-crown paper in the reading list should consider one of these directions. The *MADCrowner 2026* (paper 034) extends to inlay, the *ToothForge 2025* (paper 037) extends to *complete dental arch* generation, the *VBCD 2025* (paper 035) extends to *voxel-based* generation. **For our project: pick *one* extension direction and read the corresponding paper in the reading list before designing.**

### What this paper changes for our project

- **Before reading:** our project should focus on *completion-based* methods (PCN, GRNet, TopNet, PoinTr) and *2D-depth-image* methods (CMEMO, DAIS, DCPR-GAN, DentalRecNet).
- **After reading:** our project should *also* consider *transformer-based point-to-mesh* methods (this paper, Hosseinimanesh 2023, Hosseinimanesh 2025 MedIA), with *morphology-aware* loss (CPL) and *mesh-quality* supervision (MRL). The *paradigm shift* from 2D-depth-image GAN (2018-2022) to 3D-point-cloud transformer (2023-2025) is *now* complete in the AI-crown reading list.
- **Critical trade-off:** the *direct* mesh output (DCrownFormer's design) is *more* clinically useful (no post-processing needed) but *less* interpretable (the network's intermediate representations are *opaque*) than the *2-stage* GAN approach (DAIS, DCPR-GAN, DentalRecNet). For *clinical acceptance* (H1), the *direct* mesh output is *better*; for *research interpretability* (H2-H3), the *2-stage* approach is *better*.

---

**Status:** Note complete. *Connections to all 5 hypotheses explicitly addressed, quotes curated, code/data link documented, 10 concrete next steps proposed.* 
**Reading-list position:** paper 068 of 067+. Closes the 2024 MICCAI transformer era opening (after the 2021-2022 Tian group GAN era closed by paper 066). Opens the 2025-2026 diffusion era (which is *next* in the reading list, with MADCrowner 2026, VBCD 2025, ToothCraft 2026, ToothForge 2025 already covered).
**Hyp[previous scholar's Qiao 2022 MCSI-Net recommendation was a hallucination — MCSI-Net is a *cardiac MR* paper (Xia et al. 2022, MedIA 77, 102366), not a *dental crown* paper. This paper (DCrownFormer 2024) is the *actual* "next paper" in the AI-crown reading list, the *transformer + MCAM + CPL + MRL* bridge to the 2025-2026 diffusion era.]
