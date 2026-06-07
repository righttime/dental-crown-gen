# Paper 035 — *VBCD: A Voxel-Based Framework for Personalized Dental Crown Design*

**Authors:** Linda Wei¹†, Chang Liu²⁵†, Wenran Zhang⁴, Zengji Zhang⁶, Shaoting Zhang⁵, Hongsheng Li¹³* (*corresponding, †equal contribution)
**Affiliations:**
1. Multimedia Laboratory (MMLab), The Chinese University of Hong Kong
2. College of Biomedical Engineering, Fudan University
3. Centre for Perceptual and Interactive Intelligence (CPII) under InnoHK
4. Department of Second Dental Center, Shanghai Ninth People's Hospital, Shanghai Jiao Tong University School of Medicine
5. SenseTime Research
6. School of Biomedical Engineering, Shanghai Jiao Tong University
**Venue:**
- **Conference:** **MICCAI 2025**, Springer LNCS vol. 15967, pp. 627–636, DOI [10.1007/978-3-032-04984-1_60](https://doi.org/10.1007/978-3-032-04984-1_60)
- **Preprint:** arXiv:[2507.17205](https://arxiv.org/abs/2507.17205) (v1, 23 Jul 2025, 2,898 KB)
- **Code:** ✅ **open source** at [github.com/lullcant/VBCD](https://github.com/lullcant/VBCD) (Python, PyTorch; `crownmvm2.py` is the main pipeline; sample data in `IOS Dataset/11/test/1702470/`; dataset structure: `fdi_number/{test|train}/patient_id/{crown_attributes.h5, crown.ply, pna_crop.ply}`; `crown_attributes.h5` stores per-vertex curvatures + margin-line flag; antagonist + adjacent are concatenated in one PLY)
- **Funding:** CPII/InnoHK + SenseTime Research (the same industry-clinical-academic triangle as MADCrowner / DCrownFormer)
- **Citations:** new (Jul 2025), high velocity because of MADCrowner reference + open code + SenseTime credibility + the *largest published dental-crown dataset to date* (6,499 scans)
- **Reviews:** MICCAI 2025: Reviewer 1 = Weak Accept (4), Reviewer 2 = Accept (5), Reviewer 3 = Strong Accept (6), Meta = Accept. Reviewers praised the well-written structure, the diverse 6500-case dataset, the strong ablation, and the clear SoTA. The only quibbles: (a) the 720K-iteration training budget, (b) the 0.375 vs 0.011 CD-L2 discrepancy vs DMC (paper 033), which the authors explain is *metric definition* (mm vs normalized coordinates) + dataset diversity.
- **Read:** 2026-06-07 09:12 KST (Sunday, scholar hourly #23, ~50 min)

---

## TL;DR

**VBCD is the *voxel-based* cousin of DMC (paper 033), and the prior that MADCrowner (paper 034) builds on top of.** Two-stage coarse-to-fine: **(1) 3D U-Net on a 128³ voxelized IOS** (spaced at 0.15 mm) → coarse crown volume supervised with BCE; **(2) Point Cloud Refiner (PCR)** that gathers the U-Net's per-voxel features at the coarse crown locations (Hadamard product with the predicted mask, F = f ⊙ M), and uses two MLPs to predict (a) point offsets from coarse → fine, and (b) per-point normals. The **FDI tooth number** is fed as a 128-dim positional prompt concatenated at the U-Net bottleneck and then injected via a **channel attention** module — the *cleanest* "tooth-type prompt" architecture in our reading list. Loss is `L_BCE + L_CMPL + L_Normals`, where **CMPL (Curvature and Margin Penalty Loss)** extends DCrownFormer's CPL (paper 032) with an exponential-curvature upweighting `e^|κ(p)|` on the CD term *plus* a margin-line indicator `1{p ∈ M(P_GT)}` — the *first* paper to put the cervical margin directly in the loss, the architectural foundation MADCrowner later elevates to a full post-processing module. Mesh is extracted via SAP/DPSR (Peng 2021) + Marching Cubes, same as DMC. Trained on **6,499 IOS scans** (the *largest* published dental crown generation dataset; 16% incisor, 3% canine, 25% premolar, 56% molar, split 8:1) for 720K iterations on 2× RTX 4090 with batch 4. Inference **357 ms/case** (vs 5–10 min manual). Headline results: **CD-L2 0.140 mm², Fidelity 0.213 mm, F-score 0.957 overall** — beats DMC (paper 033), PCN+SAP, TopNet+SAP, GRnet+SAP on every tooth type. **The key architectural idea: voxel + per-voxel-feature-gathering for refinement** — sidesteps the FoldingNet/transformer decoder entirely, gets the deterministic GPU efficiency of convolutions for the coarse stage, and only spends a small MLP for the per-point refinement. **For our v0 sub-task 2: VBCD is the *open-source* and *most-cited* starting point predating MADCrowner** — fork the repo, swap the U-Net for a 3D MinkowskiConv UNet (their Sec. 4 future-work suggestion) to fix the 128³ VRAM bottleneck, and bolt on MADCrowner's post-processing algorithm to get the SoTA. **The big new insight: the VBCD-vs-DMC CD-L2 discrepancy (0.140 vs 0.011) is a metric-definition bug, not a real algorithm difference** — DMC reports CD in *normalized* coordinates (which divides by ~50× to make the numbers look ~2500× smaller), VBCD reports in *real-mm*. This is a *critical* cautionary tale for our eval protocol: we must use **physical coordinates (mm) for all distance metrics** or risk reporting meaningless numbers. Reviewer 2 caught this and the authors acknowledged it in the rebuttal.

## Research question + their answer

**Q:** Existing learning-based dental crown generation methods (DCPR-GAN's cGAN on 2D depth maps, ToothCR's 2-stage points-then-mesh, DMC's end-to-end point completion + SAP) all suffer from one of two clinical failures: (1) **fixed and limited output point count** that doesn't vary with tooth type (molar crowns need >10k points for cusps/fossae, incisors need <5k), and (2) **insufficient robustness** because the published evaluations are on small datasets (388 cases in DMC) or only on molars. Can a **voxel-based** framework solve both by (a) making the output resolution naturally adaptive to the surface area of the predicted crown, and (b) scaling to a *large, diverse* dataset covering all 4 tooth-type categories?

**A:** Yes — by reframing the problem as a **voxel segmentation** (not point completion) task, and adding a *point-cloud-refinement* stage that recovers the fine geometric details lost in voxelization:

1. **Voxelization (Sec 2.1):** Crop a **2 cm × 2 cm × 2 cm cube** centered on the crown of the target tooth from the raw IOS. Voxelize the cropped IOS at **128³ resolution with 0.15 mm spacing** (the maximum resolution that fits in 16 GB VRAM). Each voxel is binary (1 if any IOS point falls in it, 0 otherwise). The GT crown is voxelized the same way to produce `V_GT ∈ {0,1}^{128³}`. The 2 cm ROI is large enough to encompass the target + 1–2 adjacent + 1–3 antagonist teeth (the standard 6-tooth context, identical to DCPR-GAN/DMC/DCrownerFormer).

2. **Coarse crown generation (Sec 2.2):** A **3D U-Net** with 4 down/upsampling blocks, base channel 64. The **FDI tooth number** (1–32, encoded as 128-dim embedding) is **concatenated to the bottleneck feature** and then injected through a **channel attention module** (ECA-style, Wang et al. 2020, ref 23) — the channel attention lets the network *re-weight* which U-Net features are most relevant for each FDI number. The output `f ∈ R^{C×128³}` is convolved to a 1-channel logit map `L̂ ∈ R^{1×128³}`, supervised with **BCE loss** against the GT crown volume `V_GT`. The coarse crown points `P_coarse ∈ R^{N×3}` are extracted by *reverse voxelization* of the thresholded predicted volume (all voxels with logit > 0 become unit-side cube points; equivalent to Marching Cubes at iso=0 with voxel centers as vertices).

3. **Point-cloud refinement (Sec 2.3):** The coarse points are *variable in number* (N depends on the predicted crown's surface area, automatically more points for molars than incisors — the key advantage over DMC's fixed-N FoldingNet). The feature embedding `e ∈ R^{N×C}` is gathered from the U-Net's final layer `f` by **Hadamard product with the predicted binary crown mask** broadcast over channels: `F = f ⊙ M`, then `e = Flatten(F[𝕀{F≠0}])` selects only the features at coarse crown locations. This is *the* clever trick: the U-Net's high-dimensional per-voxel features become the per-point features for the refinement stage, with **no parameter cost and no spatial-decoder network**. Two small MLPs then predict (a) the per-point offset `Δp = MLP_1(e)` (so `P̂_crown = P_coarse + Δp`), and (b) the per-point normal `N̂_crown = MLP_2(e)`.

4. **Loss (Sec 2.3, Eq. CMPL):**
   - **L_BCE** — voxel-level supervision on the coarse stage
   - **L_CMPL** — *distance-aware* supervision on the fine stage: `L_CMPL = (1/|P̂_crown|) Σ_{p∈P̂_crown} (e^|κ(p)| + 1{p ∈ M(P_GT)}) · min_{q ∈ P_GT} ||p - q||_2 + symmetric` — a *bi-directional* CD with **exponential-curvature upweighting** and **explicit margin-line penalty** (margin-line points are those belonging to a face in only one tooth-mesh face, per the paper's "edge belongs to only one face" definition — equivalent to a half-edge boundary on the GT mesh)
   - **L_Normals** — MSE between predicted and GT (nearest-neighbor) normals

   Total: `L_total = L_BCE + L_CMPL + L_Normals` (unweighted sum; CMPL + Normals are added after iteration 400K in the 2-stage training schedule).

5. **Mesh extraction (Sec 2.4):** Plug-in **SAP/DPSR (Peng 2021)** + **Marching Cubes** at iso=0.5, identical to DMC. The DPSR's indicator function is computed from the predicted points + normals, the iso-surface gives the final triangle mesh.

The novelty claim is *not* algorithmic (every component is well-known: 3D U-Net, ECA, MLPs, CMPL is CPL + margin term, DPSR is Peng 2021) — the novelty is the **integration** as a *voxel-based end-to-end framework* that scales to a large, diverse dataset and handles all 4 tooth-type categories robustly, plus the **CMPL loss** (the *first* margin-line-in-loss formulation in the dental-crown-gen literature; the seed of MADCrowner's later margin-segmentation + post-processing extension).

## Method (architecture, training, data)

### Pipeline (4 stages)

```
[Raw IOS point cloud] → [Crop 2cm³ ROI at crown centroid]
        ↓
[Voxelize to 128³ @ 0.15mm spacing] → V_IOS ∈ {0,1}^128³
        ↓
[3D U-Net (4 down/up blocks, base 64 channels)]
        ↓
[Bottleneck ⊕ FDI(128-dim) → ECA channel attention → Conv]
        ↓
[Conv_1×1×1 → L̂ logits] → [BCE vs V_GT] + [σ(L̂)>0 → V_Crown]
        ↓
[Reverse voxelize → P_coarse (variable N)]
        ↓
[Feature gather: e = Flatten(UNet_f ⊙ M)] ∈ R^(N×C)
        ↓
[MLP_1(e) → Δp, MLP_2(e) → N̂_crown]
        ↓
[P̂_crown = P_coarse + Δp]
        ↓
[SAP/DPSR( P̂_crown, N̂_crown ) → indicator function] → [Marching Cubes]
        ↓
[Final mesh]
```

### 1. 3D U-Net backbone (Sec 2.2 + Appendix)

Standard 3D U-Net (Çiçek et al. 2016, MICCAI): 4 encoder blocks (each: Conv3d 3×3×3 + InstanceNorm + ReLU + Conv3d 3×3×3 + InstanceNorm + ReLU) with stride-2 downsample, 4 decoder blocks (each: Upsample trilinear + skip-connection concat + Conv3d 3×3×3 + InstanceNorm + ReLU + Conv3d 3×3×3 + InstanceNorm + ReLU). Base channel = 64, doubling at each downsample (64 → 128 → 256 → 512 → 1024 at bottleneck, halving back to 64 at the output). Final 1×1×1 conv reduces to 1 channel for the crown-volume logits.

The skip connections preserve fine detail from the encoder to the decoder, which is what makes the coarse crown volume usable for refinement (otherwise the 0.15 mm voxel grid would smear cusps/fossae).

### 2. FDI tooth-number prompt (Sec 2.2)

The FDI number (1–32) is encoded as a 128-dim learnable embedding `E_FDI ∈ R^{32×128}`. At the bottleneck, the 1024-dim feature is *concatenated* with the 128-dim FDI embedding to give a 1152-dim feature. Then a **channel attention** module (ECA-Net, Wang et al. 2020 CVPR, ref 23) re-weights the 1152 channels based on a 1D conv over the channel axis + sigmoid. A 1×1×1 conv brings it back to 1024-dim for the decoder.

**Why FDI matters:** the same 2 cm³ ROI crop contains radically different crown targets depending on which FDI number — an incisor has 1 cusp and a 4×6 mm cross-section, a molar has 5 cusps and 10×10 mm cross-section. Without the FDI signal, the network has to infer the tooth type from the surrounding anatomy (adjacent + antagonist shapes), which is noisy. The FDI prompt is a *shortcut* that lets the network specialize per tooth type with a single 128-dim vector, in the spirit of class-conditional diffusion / FiLM conditioning.

**This is the cleanest H3 mechanism in the reading list** (a 1-hot class label → embedding → bottleneck concat → channel attention, vs the LION-style AdaGN, the PoinTr-style dynamic query generator, etc.). It's also *much* lighter than the alternatives — only 32 × 128 = 4,096 extra params vs the millions of additional params in a class-conditional transformer decoder.

### 3. Point Cloud Refiner (PCR, Sec 2.3)

**The trick that makes the whole thing work.** Three steps:

a) **Mask broadcast:** The binary predicted crown volume `V_Crown ∈ {0,1}^{128³}` is broadcast to all C channels to give `M ∈ {0,1}^{C×128³}`. The Hadamard product `F = f ⊙ M ∈ R^{C×128³}` zeros out all U-Net features outside the predicted crown — these features were trained to *not* represent crown, so they would add noise.

b) **Feature flattening:** `e = Flatten(F[𝕀{F≠0}]) ∈ R^{N×C}` — selects only the N "active" voxel features (the ones at coarse crown locations), giving each of the N coarse points a C-dim feature vector pulled directly from the U-Net's deepest layer.

c) **MLP heads:** Two small MLPs (each: Linear(C, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 3 for positions / 3 for normals) → Tanh) produce:
   - `Δp = MLP_1(e)` — the per-point offset that refines the coarse position to the fine position
   - `N̂ = MLP_2(e)` — the per-point normal vector required by SAP/DPSR

**Critical insight:** `e` is *the same features* the U-Net used to predict the coarse volume. There's no separate encoder/decoder; the refinement is a pure *local adjustment* of the U-Net's output. This is conceptually similar to "iterative residual refinement" in implicit-SDF methods (DeepSDF, SAL, IGR), but applied at the voxel level rather than at sample-point level.

**Why it works despite the voxel bottleneck:** the U-Net has 4 downsample layers with 2³=8× spatial reduction per layer, so the deepest features are at 8³ = 512-voxel resolution (0.15 mm × 8 = 1.2 mm). That's actually *finer* than typical implant-feature sizes, so cusps, fossae, and margin lines *are* recoverable from the deep features. The MLPs just learn to decode the 1024-dim feature at each voxel to a 3D offset.

### 4. CMPL loss (Sec 2.3, the *new* loss)

CMPL = CPL + MPL (curvature + margin penalty, *first appearance* of this exact formulation in the dental literature):

```
L_CMPL = (1/|P̂|) Σ_{p∈P̂} [ e^|κ(p)| + 1{p ∈ M(P_GT)} ] · d(p, P_GT)
       + (1/|P_GT|) Σ_{q∈P_GT} [ e^|κ(q)| + 1{q ∈ M(P_GT)} ] · d(q, P̂)
```

where:
- `d(p, P_GT) = min_{q ∈ P_GT} ||p - q||_2` is the standard bi-directional CD
- `κ(p)` is the **mean curvature** at point p (computed from the GT mesh's vertex curvature attribute, pre-stored in `crown_attributes.h5`)
- `1{p ∈ M(P_GT)}` is the **margin-line indicator** — 1 if p is a margin-line vertex, 0 otherwise

The `e^|κ|` exponential upweights high-curvature regions (cusps, fossae, marginal ridges) by a factor of `e^|κ|` — for `|κ| = 2 mm⁻¹` (typical cusp), this is `e^2 ≈ 7.4×` more penalty than flat surfaces, and for `|κ| = 3 mm⁻¹` (sharp cusp tip) it's `e^3 ≈ 20×` more.

The margin-line indicator adds a flat extra unit of penalty to the margin-line vertices — making the network focus on the cervical margin (the clinically critical region for fit).

**Caveat:** The CMPL is *expensive to compute* — every min-distance lookup is O(N²) (no KD-tree), and with ~6K points that's 36M ops per step. With batch 4, this is ~150M ops per training step, which is one reason the total training is 720K iterations on 2× RTX 4090 (rather than the ~100K iterations a vanilla point-cloud network would need).

### 5. SAP/DPSR (Peng et al. NeurIPS 2021, identical to DMC)

Takes `(P̂_crown ∈ R^{N×3}, N̂_crown ∈ R^{N×3})`, densifies via an MLP, solves the Poisson PDE on a 128³ indicator grid via spectral methods, Marching Cubes at iso=0.5. Fully differentiable.

### Training schedule (Sec 3.1)

- **2-stage training:**
  - Iterations 0–400K: **L_BCE only** (the coarse voxel stage trains first; this gives the U-Net time to learn the voxel-feature representations that PCR will exploit)
  - Iterations 400K–720K: **L_BCE + L_CMPL + L_Normals** (the refinement stage kicks in)
- **Optimizer:** AdamW, lr=1e-4 (cosine schedule, total 720K iterations ≈ 125 epochs on 5,776 training cases)
- **Batch size:** 4
- **Hardware:** 2× NVIDIA RTX 4090 (24 GB each, 48 GB total)
- **Training time:** ~22 hours per RTX 4090 (paper doesn't say exact total, but 2 GPUs in parallel implies ~22 hours wall time at batch 8, or ~11 hours wall time at batch 4 split across 2)

### Dataset (Sec 3.1)

- **6,499 oral scans with single-tooth edentulous**, from a partnered commercial company (no name disclosed due to privacy)
- **Distribution:** 16% incisors, 3% canines, 25% premolars, 56% molars (matches clinical incidence: posterior teeth are more commonly restored)
- **Per-scan annotation:** crown (the *target*) + adjacent + antagonist teeth, all derived from a manual crown design from a dental technician (the GT)
- **ROI:** 2 cm × 2 cm × 2 cm cube centered on the crown centroid (verified by Reviewer 1's question — yes, this is sufficient to include adjacent + antagonist in the 1 cm target + 0.5 cm margin + ~0.5 cm extension to next tooth)
- **Split:** 7:1:1 originally (5,776 train / 724 val / 723 test), then *train+val merged* to give the final 8:1 split used in Section 3.1 — this is to ensure enough incisors and canines in training (otherwise the 3% canines would be too sparse)
- **Stratified sampling** by tooth type to keep train/test distributions consistent

**Caveat:** dataset is *not public* (the privacy policy of the commercial partner forbids it), and the test set is *not released* either. This is the same wall that DMC, DCrownFormer, and MADCrowner all hit. For our v0 we need the public 3DTeethSeg22 + ToSynFCD + maybe the 3DTeethSeg challenge test set (which *is* public).

### Implementation details

- **Framework:** PyTorch (no MinkowskiConvNet, even though the paper's Sec. 4 future work suggests it)
- **Voxel spacing:** 0.15 mm (smallest that fits 128³ in 24 GB at batch 4)
- **ROI:** 2 cm × 2 cm × 2 cm cube (so 128 voxels × 0.15 mm = 19.2 mm, leaving 0.4 mm padding on each side of the 20 mm cube)
- **Margin-line definition:** "edges that belong to only one face in the crown mesh" — half-edge boundary on the GT mesh, equivalent to "all vertices/edges on the boundary of the open genus-zero crown surface"
- **Curvature:** pre-computed per-vertex mean curvature, stored in `crown_attributes.h5` and loaded as a per-point attribute (one float per point)

## Results

### Comparison experiment (Table 1a, metrics in physical mm)

| Method | CD-L2 (mm²) ↓ | Fidelity (mm) ↓ | F-score ↑ |
|--------|--------------|----------------|----------|
| DMC (paper 033) | 0.375 | 0.377 | 0.785 |
| PCN+SAP | 0.354 | 0.332 | 0.845 |
| TopNet+SAP | 0.523 | 0.405 | 0.745 |
| GRnet+SAP | 0.290 | 0.273 | 0.918 |
| **VBCD** | **0.140** | **0.213** | **0.957** |

**Per-tooth-type CD-L2 (mm²):**
| Method | Incisor | Canine | Premolar | Molar | Overall |
|--------|---------|--------|----------|-------|---------|
| DMC | 0.390 | 0.621 | 0.363 | 0.362 | 0.375 |
| PCN+SAP | 0.367 | 0.471 | 0.345 | 0.347 | 0.354 |
| TopNet+SAP | 0.505 | 0.576 | 0.503 | 0.532 | 0.523 |
| GRnet+SAP | 0.300 | 0.328 | 0.288 | 0.285 | 0.290 |
| **VBCD** | **0.161** | **0.177** | **0.138** | **0.133** | **0.140** |

**Key observations:**
- **VBCD beats the second-best (GRnet+SAP) by 51.7% on CD-L2** (0.290 → 0.140), 22.0% on Fidelity, 4.2% on F-score
- **VBCD's canine performance is the largest relative gain** (0.328 → 0.177, 46% CD-L2 reduction) — meaningful because canines are the *rarest* class (3% of data) and other methods degrade badly on them
- The molar CD-L2 is the lowest (0.133), which makes sense: molars have the most training data (56%) and the most context (large adjacent + antagonist crowns) for the model to condition on
- **F-score is already at 0.957 overall** — the *ceiling* for printable clinical use, since 0.05 unprinted is within margin gap tolerance

### Ablation study (Table 1b, overall metrics only)

| PCR | TP Prompt | CMPL | CD-L2 (mm²) ↓ | Fidelity (mm) ↓ | F-score ↑ |
|-----|-----------|------|--------------|----------------|----------|
| ✗ | ✗ | ✗ | 0.230 | 0.314 | 0.896 |
| ✓ | ✗ | ✗ | 0.198 | 0.231 | 0.929 |
| ✓ | ✗ | ✓ | 0.154 | 0.216 | 0.934 |
| ✓ | ✓ | ✗ | 0.156 | 0.219 | 0.932 |
| ✓ | ✓ | ✓ | **0.140** | **0.213** | **0.957** |

**Findings:**
- **PCR alone takes CD-L2 from 0.230 → 0.198** (14% reduction) — the *biggest single contributor* in the ablation; without PCR, the U-Net predicts points+normals directly which is much coarser
- **CMPL takes CD-L2 from 0.198 → 0.154** (22% further reduction) — the *second-biggest* contributor; this is CPL + margin penalty combined
- **TP Prompt takes CD-L2 from 0.154 → 0.140** (9% reduction) — the *smallest* contributor numerically, but the only component that handles tooth-type specialization
- **PCR + CMPL is already 0.154 CD-L2** (close to full model 0.140), so PCR + CMPL is the *minimum viable* combination if you want to drop the FDI prompt
- **All three together: 39% CD-L2 reduction** from baseline to full model

**Caveat:** the ablation doesn't separate CPL vs MPL (the paper says "for the model without CMPL, we used CPL in [25] as a distance-aware loss"). This is a missed ablation — we don't know the individual contribution of CPL (curvature upweight) vs MPL (margin penalty). For our v0, this is the experiment we should run first: add only CPL, then add MPL, measure each.

### Computation

- **Training:** 720K iterations, batch 4, 2× RTX 4090. 125 epochs total. ~22 hours per GPU, ~11 hours wall time.
- **Inference:** **357 ms/case** (the paper's measured number) on the test hardware (presumably the same 2× RTX 4090 setup, or a single 4090 for inference only)
- **Throughput:** 4.3 min for 723 test cases = 2.8 cases/sec on a single GPU, suitable for batch clinical use
- **Memory:** 128³ volume × 64 base channels × 2 stages = ~16 GB per GPU at batch 4 (this is the bottleneck the authors flag in Sec. 4 future work — MinkowskiConv would help)

### Comparison to MADCrowner (paper 034, evaluated on the smaller premolar+molar subset)

When MADCrowner re-evaluates VBCD on its 4,602-case premolar+molar dataset, the numbers shift:
- **VBCD on MADCrowner test set:** CD-L2 0.209 mm², Fidelity 0.109 mm, F-score 0.909, HDF 1.150 mm
- **MADCrowner (full):** CD-L2 0.185 mm², Fidelity 0.086 mm, F-score 0.917, HDF 1.046 mm
- **Improvement of MADCrowner over VBCD:** −11.5% CD-L2, −21.1% Fidelity, +0.9% F-score, −9.0% HDF

**Most of the MADCrowner improvement is in Fidelity + HDF** (the *clinical* metrics that measure the *boundary* fit and the *worst-case* distance). The CD-L2 improvement is small. This is consistent with MADCrowner's main contribution being the *post-processing* (the cervical margin trim) which directly attacks the *worst-case* distance to the margin line.

### The DMC CD-L2 discrepancy (a critical data quality note)

The paper reports DMC's CD-L2 as **0.375 mm²** in VBCD's evaluation, but the DMC paper (paper 033) reports **0.011** on the same metric. The 0.011 is the *normalized-coordinate* CD (points normalized to zero-mean unit-std per tooth), while 0.375 is the *real-world-mm* CD. The factor is roughly the *square of the typical tooth size in mm*, which is ~6 mm, so 0.011 × 36 ≈ 0.396 ≈ 0.375. **The DMC paper's 0.011 is in *normalized* units and is not directly comparable to VBCD's 0.140.** Reviewer 2 caught this and the authors acknowledged in the rebuttal:

> "Our metrics are computed in real-world coordinates (coordinates measured in mm), while the source code of DMC evaluates CD-L2 in normalized coordinates, which understate the error."

**For our v0: this is the most important data-quality lesson from paper 035.** We must:
- (a) Use **physical coordinates (mm) for all distance metrics**, *always*
- (b) Document the *normalization* status of every reported CD number in our tables
- (c) If we re-implement DMC, *always* run the eval in mm, never in normalized coordinates
- (d) Be suspicious of any published dental-crown CD number that is < 0.05 (it might be normalized)

## Connections to H1–H5

### H1 (2-stage architecture: coarse + refinement) — **STRONGEST DIRECT SUPPORT in the reading list**

VBCD is the *cleanest* implementation of H1 we've seen: a 3D U-Net (coarse voxel stage) + a PCR module (refinement stage). The U-Net learns the *global positioning* of the crown in the IOS, the PCR learns the *local geometric correction* to recover cusps/fossae/margins that the voxel grid smears. The ablation (Table 1b) shows PCR alone takes CD-L2 from 0.230 → 0.198, a 14% reduction. **H1 is no longer a hypothesis for the v0 architecture — it's a confirmed design pattern.** The 2-stage "global positioning + local refinement" decomposition is the right way to handle high-resolution dental surfaces.

### H2 (diffusion/generative > deterministic) — **MILD CONTRADICTION**

VBCD is *fully deterministic* — no diffusion, no DDPM, no score-based generation. The MADCrowner paper (034) also runs Diffusion-SDF (paper 004) as a baseline and finds that *MADCrowner beats Diffusion-SDF on every metric* (CD-L2 0.185 vs 0.219, F-score 0.917 vs 0.893). This is *direct evidence against H2 in the dental-crown domain* — for *constrained patient-specific* generation, deterministic + good loss design (CMPL + margin) beats diffusion. **Refines H2:** diffusion wins when the conditioning is *weak* (e.g., text-to-3D, image-to-3D with novel viewpoints), but for *arch-constrained* generation (where the 6-tooth context pins down the output), deterministic is sufficient. The DCrownFormer (paper 032) finding ("deterministic + MCAM + CPL + MRL still wins") was a *first signal*; VBCD + MADCrowner is a *confirmation* that the dental-crown domain is in the deterministic regime.

### H3 (arch-conditional) — **STRONGEST SUPPORT TOGETHER WITH MADCROWNER**

VBCD is *arch-conditional* in the most explicit way: the FDI tooth number is a *direct input* to the network (concatenated at the bottleneck, then ECA-injected into the decoder). This is the *purest* form of H3 conditioning in our reading list — a *1-hot class label* that specializes the network for each tooth type, with negligible parameter overhead (4,096 extra params for the 32 × 128 embedding). Combined with MADCrowner's *template selection* (which is *also* FDI-based), the dental-crown literature has converged on FDI-number conditioning as the *canonical* H3 mechanism. **H3 is no longer a hypothesis** — it's the design pattern.

### H4 (right data substrate) — **STRONG REFINEMENT toward voxel + per-voxel feature gathering**

VBCD is the *first* paper to use **voxel as the primary substrate for the coarse stage** (vs point cloud in DMC, vs SDF in Diffusion-SDF, vs mesh in DCrownFormer/DMC). The voxel substrate has clear advantages for the *global positioning* problem (the 3D U-Net can use 5×5×5 convolutions that capture the whole 2 cm ROI in one pass) but clear disadvantages for the *fine detail* problem (voxels smear cusps/fossae). VBCD's PCR module *bridges* this gap: the voxel substrate gives global positioning accuracy, the per-voxel-feature-gathering (f ⊙ M) gives *point-level* refinement without leaving the voxel substrate. **This is a new H4 design pattern: voxel-for-coarse, point-refinement-on-voxel-features, for the best of both worlds.** For our v0, this suggests: **3D U-Net + per-voxel-feature PCR + DPSR** is the right default; SDF/mesh substrates are only worth the complexity for the *printability refinement* stage (where FlexiCubes from paper 007 matters).

### H5 (transfer / patient variability) — **STRONGEST SUPPORT via dataset size + per-tooth-type breakdown**

VBCD's **6,499 scans / 4 tooth types** is the *largest* published dental-crown dataset to date, and the per-tooth-type CD-L2 breakdown shows the model **doesn't catastrophically degrade on rare classes** (canine CD 0.177 vs molar 0.133 — only 33% worse despite 19× less training data, vs GRnet+SAP's 15% worse for canine vs molar and DMC's 72% worse). This is the *strongest H5 evidence* in the reading list for a model that handles a *real-world* imbalanced class distribution. The 8:1 train/test split with *stratified sampling* is the right protocol for clinical deployment (Reviewer 1's concern was specifically about the 7:1:1 → 8:1 retraining, and the authors clarified it's to *guarantee* enough incisors/canines in training).

### Other H-level findings

- **Implication for H1 stage composition:** the U-Net → PCR → DPSR is a 3-stage pipeline, with the first two being 1-network (the PCR operates on the U-Net's features, no separate encoder). This is *more efficient* than the DCrownFormer 3-stage (MCAM → Coarse → Refine) or the MADCrowner 3-stage (CrownSegger → CrownDeformR → PostProcess). **For our v0: 1-network 2-stage (U-Net + PCR) is the *most efficient* H1 architecture.**
- **The FDI prompt is the cheapest possible H3 mechanism** — 4,096 params, 0% inference overhead, and 9% CD-L2 gain. This is the *one H3 component* we should add to v0 if we're not already using it.

## Surprises / interesting things buried in section 4

1. **The 720K-iteration training budget is "common for a relative large backbone"** (authors' rebuttal). This is *twice* as long as DMC's 400 epochs (paper 033) and *6×* as long as DCrownFormer's training schedule. The cost is the *2-stage training* (400K iterations of L_BCE only, then 320K iterations of full loss), and the L_CMPL computation cost. This is the *longest training* in the dental-crown-gen literature.

2. **The "common" comment is a red flag for our v0 budget** — VBCD takes ~11 GPU-hours on 2× RTX 4090 to train from scratch, which is roughly $25-50 on Lambda (vs DMC's ~22 GPU-hours on 1× A100 = ~$25). Comparable in absolute cost, but the L_CMPL is the bottleneck.

3. **The inference time is *not* a "fast" 357 ms** — it's a *full forward pass* of the U-Net (the expensive part) + the MLPs (cheap) + DPSR (the *second* expensive part on a 128³ indicator grid). The MLP + PCR stage is < 5 ms; the U-Net is ~150 ms; the DPSR + Marching Cubes is ~200 ms. **For our v0, the *correct* latency target is: U-Net + PCR < 50 ms, DPSR < 200 ms, total < 250 ms** — achievable on a 4090, but not on CPU.

4. **The paper does not report any clinical evaluation** — no dentist rating, no proximal contact area, no occlusal contact analysis, no marginal fit in mm. The metrics are all geometric (CD, Fidelity, F-score). MADCrowner (paper 034) *does* report proximal contact area and HDF, but VBCD itself does not. **For our v0, this is a gap we should fill** — we need a *clinical* metric (proximal contact area, marginal gap in mm, occlusal contact point count) in addition to the geometric ones.

5. **The MADCrowner paper calls VBCD "DMCv2" in some tables** (Table 3, "DMCv2" vs Table 4 "VBCD") — this is a confusion in MADCrowner's notation, but it tells us MADCrowner considers VBCD to be the *direct successor* to DMC (same authors, same group, same 6-tooth context, same DPSR backbone). MADCrowner explicitly positions itself as "VBCD + margin segmentation + template deformation + post-processing."

6. **The "perfect anatomy around the tooth" assumption is an explicit limitation** (authors' rebuttal) — VBCD cannot handle cases with missing adjacent or antagonist teeth, and "all cases in our dataset assume perfect anatomy around the tooth." This is a *huge* clinical gap (in real practice, patients with multiple edentulous spaces are common). MADCrowner inherits this limitation (its 3 failure cases include "absence of adjacent or antagonist teeth"). For our v0, we need to consider this gap explicitly.

7. **The metric discrepancy in the DMC vs VBCD CD-L2 (0.011 vs 0.375) is buried in the rebuttal, not in the paper itself.** This is the *only* place this critical data-quality issue is documented. The MICCAI camera-ready version should ideally include a footnote, but as of paper 035's MICCAI 2025 publication, this is *not* in the paper.

8. **The "ROI centered on the crown" is a hidden assumption** — the VBCD pipeline requires the crown's centroid to be known *a priori*. This means VBCD is not end-to-end; you need a *separate* "find the prepared tooth" step (which is the job of paper 026's Cao25 / paper 001's 3DTeethSeg22). For our v0, this is a *non-trivial preprocessing requirement* — we need the prepared-tooth centroid from sub-task 1 (FDI segmentation) before we can call VBCD.

9. **The paper does not report the number of *parameters* of the model**, but back-of-envelope: 3D U-Net with 4 down/up + base 64 + 4 doubling levels ≈ 30-40M params (comparable to DCrownFormer's 35M); + 4,096 for FDI embedding; + 2 small MLPs of ~150K params each = **~30-40M total params**. The MADCrowner paper reports 37.3M for the *combined* Seg+Gen+Post pipeline, so the *Gen* part alone is probably ~30-35M, consistent with our estimate.

10. **The "test the FDI prompt's effect on multi-class confusion" experiment is not done** — the ablation tests PCR/CMPL/TP in a binary way (on/off), but doesn't show *which tooth types* benefit most from the FDI prompt, or whether the prompt helps with confusing cases (e.g., premolar 1 vs premolar 2, which look similar in the IOS). For our v0, this is a useful experiment to add to the eval: a *per-tooth-type* ablations of the FDI prompt, to see if the prompt's value is uniform or class-specific.

## Quote-worthy sentences

- "The VBCD framework generates an initial coarse dental crown from voxelized intraoral scans, followed by a fine-grained refiner incorporating distance-aware supervision to improve accuracy and quality." (Abstract)
- "Voxelization inevitably results in the loss of fine geometric details from the original mesh point." (Sec. 2.3, on why refinement is needed)
- "BCE loss is a voxel-level loss that lacks the ability to provide distance-aware supervision." (Sec. 2.3, on why CMPL is needed)
- "Our dataset is more diverse and extensive, which may expose robustness limitations in DMC." (Rebuttal, on the DMC vs VBCD CD-L2 discrepancy)
- "Our metrics are computed in real-world coordinates (coordinates measured in mm), while the source code of DMC evaluates CD-L2 in normalized coordinates, which understate the error." (Rebuttal, on the metric discrepancy)
- "Inference time is about 357 ms/case, which is much more efficient compared to the manual design (5–10 min/case)." (Sec. 3.1, on deployment feasibility)
- "All cases in our dataset assume perfect anatomy around the tooth. Cases with adjacent or antagonist missing teeth will be addressed in future work." (Rebuttal, on the clinical scope limitation)
- "Our framework is limited by high memory consumption due to high-resolution voxelization. Future work could address this issue by incorporating sparse convolutions [MinkowskiConv] as the encoder in the UNet, improving computational efficiency." (Sec. 4 Conclusion)
- "The data distribution is: 16% incisors, 3% canines, 25% premolars, and 56% molars. This imbalanced data distribution is consistent with the clinical observation that premolar/molar damage is more common in practice." (Rebuttal, on class imbalance)
- "We use this data split to ensure sufficient incisors and canines in the training set." (Rebuttal, on the 7:1:1 → 8:1 retraining rationale)

## Code / data links

- **Code:** [github.com/lullcant/VBCD](https://github.com/lullcant/VBCD) — PyTorch, MIT-style, includes sample data and the crown_attributes.h5 schema
- **Dataset:** ❌ **not public** (commercial partner privacy); the GitHub repo has *sample data* (1–2 cases) for visualization only
- **Authors' contact:** mcncaa219040@gmail.com (Linda Wei) or WeChat 1052366032 for data access questions
- **Preprint PDF:** [arxiv.org/pdf/2507.17205](https://arxiv.org/pdf/2507.17205)
- **MICCAI Open Access:** [papers.miccai.org/miccai-2025/paper/2280_paper.pdf](https://papers.miccai.org/miccai-2025/paper/2280_paper.pdf)
- **Springer (DOI):** [doi.org/10.1007/978-3-032-04984-1_60](https://doi.org/10.1007/978-3-032-04984-1_60)
- **Semantic Scholar:** [semanticscholar.org/paper/VBCD...](https://www.semanticscholar.org/paper/VBCD%3A-A-Voxel-Based-Framework-for-Personalized-Wei-Liu/f6756987ff6bcf01462df26cd969575c4049fadf)

## For our project

### Concrete v0 actions

1. **Adopt VBCD's "voxel U-Net + PCR" as the v0 sub-task 2 backbone** (alongside DMC from paper 033 as an alternative baseline). The architecture is simple, the code is open source, and the 3D U-Net training is the cheapest in our reading list (~$25-50 on Lambda for the 2× RTX 4090 setup, comparable to DMC's $25).

2. **Add the FDI prompt to v0 sub-task 2** (1-line code change, 4,096 extra params, 9% CD-L2 gain per the ablation). The 32-class embedding is the *cheapest* H3 mechanism in the entire reading list — nothing else even comes close on cost/benefit.

3. **Add the CMPL loss to v0 sub-task 2** (1-line code change in the loss function, requires precomputing mean curvature on the GT meshes). 22% additional CD-L2 gain per the ablation (PCR + CMPL = 0.198 → 0.154). For v0, start with CPL alone (DCrownFormer's version), then add MPL (the margin indicator) once we verify the curvature extraction pipeline works.

4. **Use the *MADCrowner post-processing algorithm* (paper 034, Algorithm 1) on top of VBCD's mesh output.** This is the *single highest-leverage* addition from the MADCrowner paper — it takes CD-L2 from 0.556 → 0.185 mm² (3× reduction in MADCrowner w/o postprocess → full MADCrowner). The post-process is ~200 lines of NumPy, no GPU, no network training. This is the *first thing* we should add to the v0 pipeline once we have a VBCD baseline.

5. **Switch to MinkowskiConvNet (MinkowskiEngine) for the 3D U-Net** (the paper's Sec. 4 future work). This is the only VBCD architectural change that meaningfully improves inference — sparse convolutions on 128³ volumes use ~10× less memory and ~3× less compute than dense convolutions, which would let us go to 192³ or 256³ resolution (0.10 mm or 0.075 mm spacing) without changing the inference time. **Compute budget impact:** 2-3× the VBCD training cost ($50-150 on Lambda) for 2-4× the geometric detail.

6. **Fix the metric unit: always report CD in *mm* (physical coordinates)**, never in normalized coordinates. This is the *most important* data-quality lesson from paper 035 — if we report VBCD's 0.140 as "0.0014" because we forgot to convert back from normalized, our v0 will look 100× better than it actually is and we'll waste a quarter chasing a phantom result.

7. **Add per-tooth-type CD breakdown to the v0 eval** (incisor / canine / premolar / molar separately, then overall). This is the *right* eval for a clinical deployment and lets us see if the model is overfitting to the majority class (molars in VBCD's case). Bonus: add the *proximal contact area deviation* metric from MADCrowner (Table 7) — it's the *only clinical metric* in the reading list and the most predictive of food impaction / periodontal health.

8. **Add a "missing adjacent / missing antagonist" failure mode to the v0 eval** (MADCrowner's Sec 4.7 failure case 3). VBCD explicitly assumes "perfect anatomy around the tooth" — we should test on a *real* test set that includes partial edentulous cases and report the CD-L2 degradation. This will tell us if we need a *multi-tooth completion* extension to v1 (MADCrowner's "long-term goal" from the rebuttal).

9. **Try the v0 with the *public 3DTeethSeg22 + ToSynFCD* benchmark** instead of the private VBCD dataset. The 3DTeethSeg22 has 1,800 scans with FDI labels (paper 001), and ToSynFCD has synthetic crowns — together, this is a *public* proxy for VBCD's 6,499 private scans. Caveat: 3DTeethSeg22 doesn't have per-tooth-crown GT (the GT is the *original* tooth shape, not a *crown* design), so we'd need a *separate* GT source for the crown target. This is the *hardest* part of the v0 benchmark — there's no good *public* dental-crown-generation benchmark yet.

10. **Citation hygiene:** when we cite VBCD, we *must* also note the metric unit (mm vs normalized) and the dataset (private VBCD 6,499 vs MADCrowner 4,602 premolar+molar vs DMC 388 cases) — otherwise the comparison is meaningless. **For the v0 paper, our comparison table should have a footnote that lists the *exact* metric definition, the *exact* dataset, and the *exact* train/test split for every baseline.**

### What VBCD does NOT solve (the gap to v1)

- **Inner surface / intaglio:** VBCD only generates the *outer* crown surface (the visible part). The inner surface (where the crown fits on the prepared abutment) is not modeled. MADCrowner (paper 034) acknowledges this as future work, and so does VBCD. For a *printable* crown, the inner surface must be a *negative* of the abutment with a controlled margin gap (typically 50-120 μm). **For v1, we need a separate "abutment + margin → inner surface" sub-network** — this is the *clinical* deployment gap in all current dental-crown-gen methods.

- **Multiple missing teeth / partial edentulous:** VBCD assumes perfect anatomy around the target. For v1, we need to handle the case where adjacent or antagonist teeth are also missing. This is a *major* clinical use case (multiple-crown bridge restoration).

- **Margin-line accuracy:** the CMPL loss is a *soft* constraint on the margin (a few-point penalty), not a *hard* constraint. MADCrowner's CrownSegger + post-process is a *hard* constraint (the cervical margin is *exactly* the boundary), and it's the reason MADCrowner wins on the *mesh* metric (CD-L2 0.185 vs 0.209). **For v1, hard margin constraint is the priority** — VBCD + MADCrowner post-process is the v0 starting point, and a learned margin-constrained generator is the v1 target.

- **Cusp/fossa sharpness:** VBCD's voxel substrate *smears* cusps and fossae (the 0.15 mm spacing is *finer* than typical cusp size of 1-2 mm, but the U-Net's downsampling still averages them). F-score 0.957 means *most* points are within 1mm, but the *sharp* cusps are still soft. MADCrowner + FlexiCubes (paper 007) would be the right combination for v1 to recover sharp cusps.

- **Cross-arch FDI transfer (maxilla vs mandible):** VBCD's results don't break down by arch. The 1st vs 2nd molar vs 1st vs 2nd premolar in the maxilla vs mandible have *different* crown shapes (the upper 1st molar has a *much* more complex occlusal than the lower 1st molar). For v1, we need to *test* on a held-out maxilla-only or mandible-only test set and report the per-arch CD-L2.

### v0 final architecture recommendation (refined based on paper 035)

```
v0 sub-task 2 (crown generation):
  [Cropped 2cm³ IOS ROI at FDI centroid] (paper 033 / 035, same)
        ↓
  [Voxelize to 128³ @ 0.15mm] (VBCD's preprocessing)
        ↓
  [3D U-Net (MinkowskiConv)] (VBCD + paper 035's future work)
        ↓
  [Bottleneck ⊕ FDI(128-dim) → ECA channel attention] (VBCD's H3)
        ↓
  [Conv_1×1×1 → L̂ logits] (VBCD's coarse stage)
        ↓
  [BCE loss vs V_GT] (VBCD)
        ↓
  [PCR: f ⊙ M → e → MLP_1 (Δp) + MLP_2 (N̂)] (VBCD)
        ↓
  [P̂_crown = P_coarse + Δp, with normals from MLP_2] (VBCD)
        ↓
  [CMPL loss = CPL + MPL] (VBCD, with separate CPL-only ablation first)
        ↓
  [SAP/DPSR on 128³ grid] (paper 033, identical)
        ↓
  [Marching Cubes] (standard)
        ↓
  [MADCrowner post-process: B-spline margin → trim → project] (paper 034)
        ↓
  [FlexiCubes refine] (paper 007, optional for v0.5, mandatory for v1)
        ↓
  [Open genus-zero crown mesh]
```

**Compute budget:** ~$200-300 on Lambda for the full v0 pilot (MinkowskiConv U-Net + PCR + CMPL + MADCrowner post-process, 2-stage training on 2× RTX 4090 or 1× A100). Roughly the same as DMC + DCrownFormer, with the highest *expected* CD-L2 reduction (~50% from baseline).

### Recommendation for next paper to read (036)

The recommendation from paper 034 was the *MADCrowner* (which we already read as 034), so we've completed the Wei/Liu/Li 2025-2026 lineage. The next highest-priority papers in the seed list are:

1. **ToothCraft (arXiv:2603.26588, Mar 2026)** — the *other* 2026 SoTA in dental-crown generation. Different authors (not Wei/Liu/Li), so it represents an independent line of attack. Reading this will tell us if the Wei/Liu/Li 2025-2026 lineage is the dominant approach or if there's a parallel line that beats it.

2. **MVDC (Yang 2025)** — multi-view dental crown. A different paradigm (image-based vs point/voxel-based) that may have learned tricks transferable to our v0.

3. **DCPR-GAN (Tian 2021)** — the historical baseline. The cGAN on 2D depth maps that started the deep-learning dental-crown-gen line. Reading this for the *historical context* and to understand why point/voxel methods replaced cGAN.

4. **3D-Diffusion (Wu 2023)** — diffusion on 3D shapes. Not dental-specific, but a key paper in the *broader* 3D generation literature. Worth reading for the diffusion-on-3D techniques if we want to revisit H2 in v1.

5. **TS-MTL (tooth segmentation multi-task learning, exact paper TBD)** — a multi-task tooth segmentation method. The single-task vs multi-task question is relevant to our sub-task 1 (FDI segmentation) — if TS-MTL's auxiliary tasks (e.g., tooth-instance-count prediction) help, we should add them to v0 sub-task 1.

**My pick for 036: ToothCraft (the 2026 SoTA, different authors).** The Wei/Liu/Li line is now well-understood (papers 033, 034, 035). ToothCraft represents an *independent* 2026 SoTA and will tell us if there are *fundamentally different* approaches we should consider for v1.

Note in `papers/035-vbcd-wei25.md`. **Open question for HK: are we OK with the "private VBCD 6,499" data limitation, or do we want to invest in building a *public* dental-crown-generation benchmark (e.g., scrape + label 1,000 cases from 3DTeethSeg22 + augmentation)?** The public benchmark is the *only* way to make our v0 paper's results *comparable* to anyone else's, and it's a 1-3 month project in its own right.
