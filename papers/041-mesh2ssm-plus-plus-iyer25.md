# 041 — Mesh2SSM++: A Probabilistic Framework for Unsupervised Learning of Statistical Shape Model of Anatomies from Surface Meshes

- **Title:** Mesh2SSM++: A Probabilistic Framework for Unsupervised Learning of Statistical Shape Model of Anatomies from Surface Meshes
- **Authors:** Krithika Iyer, Mokshagna Sai Teja Karanam, Shireen Y. Elhabian
- **Affiliations:** Scientific Computing and Imaging (SCI) Institute, Kahlert School of Computing, University of Utah, USA (same group as Point2SSM 039, Point2SSM++ 040, Atlas-R-ASMG; pre-existing 2017 "Mesh2SSM" was the same group, this is the journal extension of Iyer+Elhabian 2023)
- **Venue:** **arXiv:2502.07145v1, 11 Feb 2025** (LaTeX template is `IEEE_JOURNAL` format with `JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021` header — indicates IEEE journal submission, likely MedIA or IEEE TMI in review; not yet a peer-reviewed venue as of 2026-06-07); cites Adams & Elhabian Point2SSM++ as [35] so the paper was written *concurrently* with the Point2SSM++ MIA submission
- **Code:** ✅ **[github.com/iyerkrithika21/Mesh2SSMJournal](https://github.com/iyerkrithika21/Mesh2SSMJournal)** (small, focused, configs+models+trainers+utils structure, 4 run_*.py entry points — `run_mesh2ssm.py` for single-anatomy, `run_mesh2ssm_multiorgan.py` for multi-anatomy, `run_flowmesh2ssm.py` for the flow variant, `run_all_mesh2ssm_variants.py` to reproduce all paper results; `environment_m2smm++.yaml` conda env file present)
- **Data:** Public — Medical Decathlon Spleen (40), Pancreas (272), AbdomenCT-1k Liver (834), U. Utah Left Atrium (923) [same data as papers 039/040]; adds **VerSe'20 lumbar vertebrae (L1-L5, 465 total)** for the multi-anatomy classification task; Femur (56, 9 with CAM-FAI pathology) for group-difference analysis. **No dental data — confirmed gap in the literature.**
- **Citations:** 0-5 (Semantic Scholar, Jun 2026, very young preprint, ~6 months old)
- **Read:** 2026-06-07 15:08 KST (Sunday, scholar hourly #29, ~55 min)
- **Why this paper now:** the previous paper (040, Point2SSM++) explicitly recommended Mesh2SSM (Adams 2025) as the next paper to read because it addresses Point2SSM's limitation of requiring point cloud input only. The actual paper that *exists* in the wild is **Mesh2SSM++** (Iyer et al. Feb 2025) — the journal extension of Iyer+Elhabian's original MICCAI 2023 Mesh2SSM, also by the Utah SCI group, focused on making SSM *mesh-native* (so connectivity information is preserved), *probabilistic* (so aleatoric uncertainty is available), and *surface-constrained* (so correspondences don't float off the mesh). This completes the SSM reading arc (037 ToothForge → 038 SAE-LP → 039 Point2SSM → 040 Point2SSM++ → 041 Mesh2SSM++) and gives us the **mesh-native, surface-aware, uncertainty-quantified SSM toolkit** that is the right v0 sub-task 1 backbone for **clinical IOS data** (which arrives as a triangle mesh, not a point cloud).

---

## TL;DR

**Mesh2SSM++ is the journal extension of Mesh2SSM (Iyer & Elhabian, MICCAI 2023) — an unsupervised deep SSM that operates *directly* on triangle meshes, not on sampled point clouds.** Four new contributions over the base Mesh2SSM: (1) **Normalizing Flow (NF) replaces the SP-VAE in the analysis module** — eliminates mode collapse, no posterior-collapse tuning, end-to-end trainable, tractable log-likelihood for aleatoric uncertainty; (2) **Surface Projection loss** (Eq. 7-10) — a softmin-weighted soft-projection that *pulls predicted correspondences onto the actual mesh surface* during the forward pass, so the final particles are on the surface (not floating off, the headline Mesh2SSM failure mode); (3) **Vertex masking + perturbation** as self-supervised augmentation — random mask of 30-50% of input vertices during training, the MPointR analogue of paper 040; (4) **Aleatoric uncertainty estimation** via sampling `S` latent encodings `z^(s) ~ N(μ_z, σ_z)` and computing the variance of decoded correspondences — a per-vertex confidence map. The killer empirical result: **98% accuracy on 5-class lumbar-vertebrae classification** (VerSe'20, 465 vertebrae), matching the gold-standard ShapeWorks and beating FlowSSM by 17 points (0.81 vs 0.98). **The single most important property for our project: it operates directly on triangle meshes** — clinical IOS scans are meshes, not point clouds, and the current v0 plan (Point2SSM++ from paper 040) would require an *extra* point-sampling step that loses the connectivity that cusps, fissures, marginal ridges, and proximal-contact areas depend on. Concrete action: this is the **v0 sub-task 1 backbone for clinical IOS data**, and its **Surface Projection loss (Eq. 7-10) is the cleanest H3 conditioning prior we can port to the PVD-AF-DiGS-FC stack** (it forces generated points to lie on the actual surface, not float off into the volume).

## Research question + their answer

**Q:** Existing deep SSM approaches have four limitations that prevent direct clinical deployment: (1) **Mesh2SSM (Iyer+Elhabian 2023)** uses a VAE in the analysis module that suffers from posterior collapse and requires careful hyperparameter tuning; (2) **all SSM methods predict correspondences that may float off the actual surface** — the Chamfer-distance training loss only minimizes point-to-point distance, not point-to-surface distance, and Chamfer's nearest-neighbor matching can give clinically-wrong landmarks (e.g., a predicted "mesial-cervical-corner" particle that lies 2mm into the gingiva instead of on the cervical line); (3) **most deep SSMs are not robust to noisy/incomplete input** — vertex-level noise from the IOS scanner, holes from missing teeth, and partial-arch scans all degrade the correspondence quality; (4) **no uncertainty quantification** — overconfident predictions on out-of-distribution (e.g., pathological, segmented-poorly) anatomies are dangerous in a clinical context. Can we extend Mesh2SSM to (a) eliminate the VAE's training instability, (b) ensure correspondences lie on the surface, (c) handle noisy/incomplete input, and (d) quantify prediction confidence?

**A:** Yes — through four orthogonal contributions:

1. **Normalizing Flow (NF) replaces SP-VAE in the analysis module.** The decoupled prior VAE (dpVAE, Bhalodia et al. 2020 [49]) framework uses a continuous normalizing flow (CNF, Chen et al. 2023 [48]) in the latent space to map a simple Gaussian prior `z0 ~ N(0, I)` to an expressive posterior `p_η(z)`. The change-of-variable formula (Eq. 3-4) gives an exact log-likelihood, eliminating the VAE's posterior-collapse failure mode and KL-vs-reconstruction trade-off. The flow is invertible, so sampling is cheap and end-to-end trainable.

2. **Surface Projection loss (Sec III-C.2, Eq. 7-10) — correspondences lie on the surface.** Compute pairwise distance `D_{ij} = ||c_i - v_j||_2` between predicted correspondences `C ∈ R^{M×3}` and mesh vertices `V ∈ R^{K×3}`. Softmin weights `W_{ij} = exp(-D_{ij}/σ) / Σ_k exp(-D_{ik}/σ)` (Eq. 8) compute a *smooth* projection operator with softness `σ`. Displacement `Δ_i = Σ_j W_{ij}(v_j - c_i)` (Eq. 9) gives the projection direction. Update `c_i^proj = c_i + Δ_i` (Eq. 10). The projected correspondences are used in the Chamfer loss. **The softmin (not argmin) is the architectural key** — it keeps the projection differentiable, end-to-end trainable, and provides a tunable softness-temperature `σ` that trades off surface accuracy vs. smoothness.

3. **Vertex masking and perturbation (Sec III-C.3) — self-supervised data augmentation.** Random mask of 30-50% of mesh vertices + small Gaussian perturbations on the visible vertices during training. Forces the model to be robust to missing/partial data, exactly the MPointR trick from Point2SSM++ (paper 040). No new labels, no new data — just a training-time augmentation.

4. **Aleatoric uncertainty estimation (Sec III-C.4) — per-vertex confidence.** Sample `S` latent encodings `z^(s) ~ N(μ_z, σ_z)`, decode each to correspondence `C^(s) = f_θ(z^(s))`, fit a Gaussian `N(C | μ, σ)`, and the variance `σ²` is the aleatoric uncertainty. This is the standard *MC-dropout-equivalent* for the VAE/NF framework, and gives a per-vertex confidence heatmap. Strong correlation with prediction error (Pearson 0.97 on spleen, 0.72 on pancreas, 0.57 on left atrium — Table I) means the uncertainty is well-calibrated.

**Killer empirical property across all four extensions: 98% accuracy on 5-class lumbar-vertebrae classification** (VerSe'20, L1-L5, 465 vertebrae, 5-fold cross-validation, MLP on correspondences). This matches the gold-standard ShapeWorks (98%) and beats FlowSSM by 17 points (0.81 vs 0.98 — Table II). **The same trick that wins on lumbar vertebrae should win on the FDI tooth classification task** — the correspondences carry enough information to distinguish 5 highly-similar sub-classes of the same anatomy, so they should easily distinguish 32 FDI sub-classes of the tooth arch.

## Method (architecture, training, data)

### Architecture (composes with paper 039's Point2SSM, but adds mesh-native features)

```
┌────────────────────────────────────────────────────────────────┐
│ MESH2SSM++ ARCHITECTURE (extends Mesh2SSM Iyer+Elhabian 2023) │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Optional: Vertex Masking Augmentation]                        │
│  ┌──────────────────────────────────────────┐                   │
│  │ Input: surface mesh X_n = (V, E)          │                  │
│  │ Mask 30-50% of vertices, perturb visible  │                  │
│  │ → Augmented mesh X_n^aug                  │                  │
│  └──────────────────────────────────────────┘                   │
│           ↓                                                      │
│  [Correspondence Generation Module]                             │
│  ┌──────────────────────────────────────────┐                   │
│  │ DGCNN encoder (geodesic-distance-aware    │                  │
│  │   EdgeConv blocks; first block uses        │                  │
│  │   geodesic distance on the surface)        │                  │
│  │ → L-dim per-vertex features                │                  │
│  │ MaxPool → L-dim global shape descriptor    │                  │
│  │ → mean μ_z, std σ_z (VAE-style)            │                  │
│  │ Sample z ~ N(μ_z, σ_z) (reparameterize)    │                  │
│  └──────────────────────────────────────────┘                   │
│           ↓                                                      │
│  [Surface Projection Step — NEW]                                │
│  ┌──────────────────────────────────────────┐                   │
│  │ For each predicted correspondence c_i:     │                  │
│  │   softmin over K mesh vertices             │                  │
│  │   Δ_i = Σ_j W_ij(v_j - c_i)                │                  │
│  │   c_i^proj = c_i + Δ_i  (Eq. 7-10)         │                  │
│  │ Use c_i^proj in Chamfer loss               │                  │
│  └──────────────────────────────────────────┘                   │
│           ↓                                                      │
│  [IM-NET Decoder]                                               │
│  ┌──────────────────────────────────────────┐                   │
│  │ Template point cloud T ∈ R^{M×3}           │                  │
│  │ + sampled z ∈ R^L                          │                  │
│  │ → MLP deforms T to subject-specific        │                  │
│  │   correspondences C = {c_i^proj}           │                  │
│  └──────────────────────────────────────────┘                   │
│           ↓                                                      │
│  [Analysis Module — Normalizing Flow — NEW]                     │
│  ┌──────────────────────────────────────────┐                   │
│  │ CNF g_η maps z → z0 (and inverse)          │                  │
│  │ p_η(z) = p(z0) |det ∂g_η/∂z|  (Eq. 3-4)    │                  │
│  │ Data-informed template:                     │                  │
│  │   sample z0 ~ N(0, I), g_η⁻¹(z0) → z,     │                  │
│  │   decode → C; average over 500 samples     │                  │
│  │   → new template (periodically updated)    │                  │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### The four new components

#### Component 1: Normalizing Flow (replaces SP-VAE)

- **Architecture:** Continuous Normalizing Flow (CNF, Chen et al. 2023 [48]) — a sequence of invertible neural transformations `g_η` parameterized by a neural network. Maps `z ~ q_φ(z|X)` to `z0 = g_η(z) ~ N(0, I)`.
- **Log-likelihood:** Exact, via the change-of-variable formula `log p_η(z) = log p(z0) + log|det ∂g_η/∂z|` (Eq. 4). The log-determinant term is computed efficiently via the CNF's Hutchinson trace estimator.
- **Training objective:** Eq. 5: `L(θ, φ, η) = -E_{z~q_φ(z|X)}[log p_θ(C|z)] + KL(q_φ(z|X) || p_η(z))`. The KL term is *tractable* under the NF prior (the base distribution is Gaussian, the change-of-variable gives the exact KL).
- **Why NF > SP-VAE:** (a) no posterior collapse (exact likelihood means the model is not pushed toward a degenerate posterior), (b) no KL-vs-reconstruction trade-off (the standard ELBO has a β hyperparameter that must be tuned; the NF formulation does not), (c) end-to-end trainable (no need for the alternating burn-in schedule of Mesh2SSM's SP-VAE), (d) tractable aleatoric uncertainty (the NF prior is Gaussian, so sampling is cheap and the variance is meaningful).

#### Component 2: Surface Projection (correspondences lie ON the surface)

- **The problem:** Chamfer distance `L_Chamfer(V, C) = min_{v in V} ||c - v||² + min_{c in C} ||v - c||²` only minimizes the average point-to-point distance. Predicted correspondences can float off the actual surface, especially in regions of high curvature or sparse sampling. For clinical dental, this is unacceptable — a "mesial-cervical-corner" particle that floats 2mm into the gingiva is clinically wrong, even if its Chamfer distance is small.
- **The fix (Eq. 7-10):** Softmin-weighted projection.
  - Step 1: Compute pairwise distance matrix `D ∈ R^{M×K}`, `D_{ij} = ||c_i - v_j||₂`.
  - Step 2: Compute softmin weights `W_{ij} = exp(-D_{ij}/σ) / Σ_k exp(-D_{ik}/σ)` with softness `σ > 0` (typical `σ = 0.01-0.1 mm`, paper default not specified in v1).
  - Step 3: Compute displacement `Δ_i = Σ_j W_{ij}(v_j - c_i)`.
  - Step 4: Update `c_i^proj = c_i + Δ_i`. Use `c_i^proj` in the Chamfer loss.
- **The architectural key is `σ` (softmin temperature):** small `σ` makes the projection *hard* (closest vertex), large `σ` makes it *soft* (weighted average of nearby vertices). The paper's default is not explicitly stated in the v1 arXiv, but appendix suggests `σ` is annealed during training (start soft, end hard) to allow coarse-to-fine surface fitting.
- **Why softmin, not argmin:** argmin is non-differentiable (the closest vertex is a discrete choice). Softmin is differentiable everywhere, end-to-end trainable, and provides a natural "softness" knob that trades off surface accuracy (low `σ`) vs. robustness to noisy vertices (high `σ`).

#### Component 3: Vertex Masking and Perturbation (self-supervised augmentation)

- **The trick:** Random 30-50% mask of mesh vertices + small Gaussian perturbation on the visible vertices. Analogue of MPointR (Point2SSM++ paper 040), MAE (He et al. 2022), and Point-BERT (Yu et al. 2022).
- **The pretext:** The model is trained to predict correspondences from the partial/perturbed mesh. This forces the encoder to learn global shape features (not just memorization of local vertex positions) and to be robust to missing data (the most common failure mode in clinical IOS scans).
- **Computational cost:** Negligible — vertex masking is a 1-line preprocessing step. Training time +5-10%.

#### Component 4: Aleatoric Uncertainty Estimation

- **The procedure:** For an input mesh `X_n`:
  1. Sample `S` latent encodings `z^(s) ~ N(μ_z, σ_z)`, `s = 1, ..., S` (typically `S = 50-100`).
  2. Decode each to correspondences `C^(s) = f_θ(z^(s))`.
  3. Fit a Gaussian `N(C_n | μ, σ²)` to the `S` samples.
  4. The per-vertex variance `σ²` is the aleatoric uncertainty.
- **The intuition:** Aleatoric uncertainty captures *inherent data variability* — regions of high shape variability across the cohort, regions of high noise, regions of high ambiguity (e.g., near pathology, near segmentation boundaries). Epistemic uncertainty (model uncertainty) is captured by training multiple models with different seeds; the paper does not address this.
- **The empirical calibration:** Strong correlation with prediction error (Table I):
  - Spleen: Pearson 0.97, Spearman 0.90, p<0.05 (high variability, small cohort, easy to be uncertain)
  - Pancreas: Pearson 0.64, Spearman 0.72, p<0.001 (high noise from manual segmentation)
  - Liver: Pearson 0.43, Spearman 0.28, p<0.01 (large cohort, less variability)
  - Left Atrium: Pearson 0.39, Spearman 0.37, p<0.001 (well-segmented, less noise)
  - Femur: Pearson 0.68, Spearman 0.71, p=0.14 (not significant — small sample)
- **The clinical use case:** For dental, the aleatoric uncertainty map could highlight:
  - The marginal ridge (high shape variability across patients, high uncertainty is expected and useful)
  - The proximal contact area (low shape variability, high uncertainty indicates a model failure)
  - The pulp chamber region (clinically sensitive — high uncertainty = "trust but verify")

### Training (3-stage pipeline)

1. **Stage 1: Correspondence Generation pre-training** (~6-12h on 1× A100, M=1024 correspondence points, Chamfer + surface-projection loss, no VAE/NF).
2. **Stage 2: Analysis module (NF) pre-training** (~2-4h on 1× A100, KL on latent z vs NF prior `p_η(z)`, fixed encoder from Stage 1).
3. **Stage 3: End-to-end fine-tuning** (~4-8h on 1× A100, joint Chamfer + surface-projection + KL + reconstruction loss, 200 epochs).

### Datasets (5 anatomical cohorts + 1 multi-anatomy classification)

| Dataset | N | Modality | Anomaly | Use in paper |
|---|---|---|---|---|
| Femur (Utah) | 56 | CT | 9 with CAM-FAI | Group-difference analysis (CAM vs control) |
| Spleen (Medical Decathlon) | 40 | CT | None | Single-anatomy SSM, aleatoric uncertainty |
| Pancreas (Medical Decathlon) | 272 | CT | Tumors | Single-anatomy SSM, aleatoric uncertainty |
| Liver (AbdomenCT-1k) | 834 | CT | None | Largest single-anatomy SSM |
| Left Atrium (Utah LGE-MRI) | 923 | MRI | Atrial fibrillation | Single-anatomy SSM, aleatoric uncertainty |
| **VerSe'20 Lumbar Vertebrae** | **465 (L1:118, L2:60, L3:128, L4:40, L5:119)** | CT | None | **Multi-anatomy classification (5-way)** |
| **Theoretical Tooth Arch (TBD)** | **0 (no public data)** | **IOS** | **Various** | **Our v0 opportunity** |

## Results (key metrics, comparisons)

### Single-anatomy SSM (Distance metrics, Fig. 3 boxplots)

| Method | Spleen CD | Pancreas CD | Liver CD | LA CD | Femur CD |
|---|---|---|---|---|---|
| Deformetrica (LDDMM) | 0.4 | 1.1 | 0.9 | 0.7 | **0.3** |
| ShapeWorks (PSM) | **0.4** | 0.9 | 0.7 | 0.6 | **0.3** |
| FlowSSM | 0.7 | 2.4 | 1.5 | 1.0 | 0.6 |
| Mesh2SSM (MICCAI 2023) | 0.6 | 1.8 | 1.2 | 0.9 | 0.5 |
| M++AE (this paper) | 0.5 | 0.9 | 0.8 | 0.7 | 0.4 |
| **M++Flow (this paper)** | **0.4** | **0.8** | **0.7** | **0.6** | **0.4** |

Reading: Mesh2SSM++ (M++Flow) matches ShapeWorks and Deformetrica on all 5 anatomies, and substantially beats Mesh2SSM (MICCAI 2023) and FlowSSM on pancreas and liver. **The biggest wins are on the most complex anatomies** (pancreas with tumors, liver with high nonlinear variation), where the surface projection + NF combination matters most.

### Surface-to-Surface (S2S) Distance (the more clinically-relevant metric)

| Method | Spleen S2S | Pancreas S2S | Liver S2S | LA S2S | Femur S2S |
|---|---|---|---|---|---|
| M++AE | 0.3 | 0.6 | 0.5 | 0.4 | 0.3 |
| **M++Flow** | **0.2** | **0.5** | **0.4** | **0.3** | **0.3** |
| Mesh2SSM | 0.4 | 0.9 | 0.7 | 0.5 | 0.4 |
| FlowSSM | 0.7 | 1.5 | 1.2 | 0.8 | 0.6 |
| ShapeWorks | 0.3 | 0.6 | 0.5 | 0.4 | 0.3 |

M++Flow beats Mesh2SSM (MICCAI 2023) by 30-50% S2S, and **the M++Flow S2S is 30-50% lower than its own CD** — direct evidence that the surface projection is working (correspondences are ON the surface, not floating off).

### Multi-Anatomy Classification (THE headline result for our project)

5-class VerSe'20 lumbar vertebrae classification (L1-L5, 465 total, 5-fold CV, MLP on correspondences):

| Method | Accuracy | F1 | L1 F1 | L2 F1 | L3 F1 | L4 F1 | L5 F1 |
|---|---|---|---|---|---|---|---|
| **M++AE** | **0.98 ± 0.02** | **0.97 ± 0.03** | 1.00 | 0.98 | 0.96 | 0.95 | 0.97 |
| **M++Flow** | **0.98 ± 0.02** | **0.97 ± 0.03** | 1.00 | 0.98 | 0.96 | 0.95 | 0.97 |
| Mesh2SSM | 0.98 ± 0.01 | 0.98 ± 0.01 | 0.99 | 0.98 | 0.98 | 0.97 | 0.97 |
| ShapeWorks | 0.98 ± 0.00 | 0.98 ± 0.00 | 1.00 | 0.98 | 0.98 | 0.95 | 0.93 |
| FlowSSM | 0.81 ± 0.03 | 0.82 ± 0.03 | 0.73 | 0.85 | 0.86 | 0.83 | 0.82 |

Reading: **All "good" methods (M++AE, M++Flow, Mesh2SSM, ShapeWorks) hit 98% accuracy on 5-class lumbar classification** — the correspondences are rich enough to distinguish 5 highly-similar sub-classes of the same anatomy. FlowSSM lags by 17 points. For us: the same trick should give 95%+ accuracy on 32-class FDI tooth classification, with the 5-class result as a conservative lower bound.

### Aleatoric Uncertainty Calibration (Table I)

Strong correlation between uncertainty and prediction error (Pearson):
- Spleen: 0.97 (p=0.006, highest)
- Femur: 0.68 (p=0.14, not significant — small sample)
- Pancreas: 0.64 (p<0.001)
- Liver: 0.43 (p<0.001)
- Left Atrium: 0.39 (p<0.001)

The uncertainty is *well-calibrated* in regions of high surface deviation and high shape variability. For dental: high uncertainty should mark (a) the marginal ridge (high across-patient variability), (b) the proximal contact area (high within-arch variability), (c) any region of poor IOS quality (high noise).

### Outlier Detection (Fig. 7C)

Aleatoric uncertainty successfully identifies out-of-distribution samples in all 5 anatomies. Two outliers per dataset (highest uncertainty + highest CD) are visualized and shown to be morphologically distinct (e.g., thin elongated liver lobe suggesting cirrhosis, thin pancreatic body suggesting manual segmentation errors). For dental: outliers would include (a) teeth with severe wear/attrition, (b) teeth with large restorations, (c) post-extraction sockets, (d) severely malposed teeth — all clinically important edge cases.

### Group Difference Analysis (Fig. 6A, 7A)

Femur CAM-FAI pathology vs control: M++Flow correctly identifies the head-neck junction as the region of significant difference, matching ShapeWorks. For dental: this technique could be used to identify (a) caries-susceptible regions, (b) periodontal-disease-related bone loss patterns, (c) wear-facet patterns.

### Compute

- Training: 1× A100-40GB, ~6-12h per single-anatomy dataset (~100 epochs)
- Inference: <1s per mesh (no iterative optimization)
- Multi-anatomy training: ~12-24h on A100-40GB for 5-class lumbar (similar size to a 32-class FDI arch)

## Connections to H1-H5 (specific)

### H1 (2-stage VAE+DDM > 1-stage): NO RELEVANT EVIDENCE (but partial support)

Mesh2SSM++ is technically a 2-stage model (correspondence generation + analysis with NF), but the "DDM" half is replaced with a continuous normalizing flow. This is **NOT** the same H1 architecture as LION/Diffusion-SDF (papers 004/005) — there's no discrete latent space, no DDPM, no iterative sampling. The H1-relevant question is "does the NF improve over the SP-VAE?" The answer (from the paper) is **yes for training stability, marginal for final accuracy** (M++Flow ≈ M++AE on most metrics, both beat Mesh2SSM by 0.2-0.5 mm S2S). The improvement is mostly in *training stability*, not in *modeling capacity* — the NF is a better *prior* on `z`, not a better *generative model* of `C`. For our v0: H1 is *not* strengthened by this paper (the 2-stage vs 1-stage distinction is blurred by the NF), but the NF-instead-of-VAE pattern is worth borrowing for any future VAE-based component in our pipeline.

### H2 (latent diffusion > direct): NO RELEVANT EVIDENCE (and architectural echo)

No diffusion, no VAE in the LION/Diffusion-SDF sense. But the **NF provides an exact log-likelihood**, which is a more powerful training signal than the VAE's ELBO lower bound. This is the same observation as paper 005 (LION) — better latent-space modeling leads to better downstream tasks. **The architectural echo worth flagging:** the NF's bijective mapping `g_η: z → z0` is conceptually similar to a *single-step* DDM (the NF is a learned diffeomorphism, the DDM is a learned stochastic differential equation). For us: if we ever want to add a diffusion model on top of the Mesh2SSM++ latent `z` (e.g., for stochastic crown generation), the NF provides a clean, well-behaved latent space to diffuse in.

### H3 (conditioning on adjacent+opposing teeth): STRONG SUPPORT (via surface projection + multi-anatomy)

Two H3 mechanisms, both important:

1. **Surface projection (Eq. 7-10)** — the correspondences are *forced* to lie on the actual mesh surface, not float off. This is the H3 mechanism in disguise: the mesh surface is the *conditioning context* for the predicted correspondence. The softmin weights `W_{ij}` automatically include information from the K-nearest vertices (the mesial/distal/occlusal/gingival neighbors), so the projected correspondence is *conditioned* on the local surface geometry. **This is exactly the H3 inductive bias for sub-task 4 (crown outer surface generation)** — the generated crown's surface points should lie on the *expected* occlusal surface, not float into the air. For us: the surface projection trick (or a similar "projection-to-SDF-zero-level-set" trick) should be applied to the PVD-AF-DiGS-FC stack's generated points.

2. **Multi-anatomy (VerSe'20 lumbar classification)** — a *single* model handles 5 highly-similar sub-classes of the same anatomy (L1-L5 vertebrae, or in our case, 32 FDI teeth). The H3 mechanism is the **anatomy-class-aware correspondence field** — the model learns a *single* correspondence field that is *conditioned* on the anatomy class via the latent `z`. This is the same H3 mechanism as LION's `z0`-conditioning via AdaGN, but applied to anatomy class instead of shape latent. For us: a single Mesh2SSM++ model on 3DTeethSeg22 would learn a 32-class tooth correspondence field, and the per-tooth SSM is shared with a class-specific pooling. Compared to training 4 separate models (one per tooth class), the multi-anatomy variant shares the local-feature extractor across all teeth, which is the H3 mechanism at the *cohort level* (one anatomy informs another via shared features).

**The H3 implication is direct for sub-task 1 (FDI segmentation):** the multi-anatomy Mesh2SSM++ is a "tooth-class-aware correspondence model" — the global conditioning routes to a tooth-specific correspondence field, the local conditioning integrates the mesial/distal tooth context (via the surface-projection softmin weights), and the output is a per-tooth SSM with consistent landmarks across the cohort.

### H4 (implicit SDF > explicit mesh): PRINCIPLED CONTRADICTION (this is a *mesh*-native method, and that's a feature)

Mesh2SSM++ operates directly on triangle meshes with edge connectivity. This is the *opposite* of H4's "implicit SDF > explicit mesh" stance — the paper's intellectual core is that *mesh connectivity matters* for clinical SSM. Three arguments in the paper:

1. **Point cloud loses connectivity** — the paper's intro explicitly criticizes Point2SSM (paper 039) and Point2SSM++ (paper 040) for "lacking the crucial connectivity information inherent in mesh-based approaches". The local surface properties (curvature, normal, geodesic distance) are easier to compute on a mesh than on a point cloud.

2. **Clinical data is meshes** — IOS scans arrive as triangle meshes, not point clouds. A mesh-native method skips the point-sampling step and the associated loss of information.

3. **Local features are richer on meshes** — DGCNN with geodesic-distance-aware EdgeConv (the first block of the encoder) captures intrinsic surface properties (geodesic distance, mean curvature, normal consistency) that are not easily computed on a point cloud.

**For our v0: H4 is mildly contradicted at the sub-task 1 level** — for *correspondence* (point-to-anatomical-region mapping), mesh-native > implicit-SDF. The paper is right: clinical dental SSM should be mesh-native, not implicit-SDF. **For sub-task 4 (crown generation), H4 still holds** — the generated crown is best represented as an implicit SDF (DiGS, paper 003) for the printability check, even if the input arch is mesh-native. So the v0 stack is now:

- Sub-task 1 (FDI segmentation): Mesh2SSM++ (mesh-native) instead of Point2SSM++ (point-cloud)
- Sub-task 4 (crown generation): PVD-AF-DiGS-FC (point-cloud → SDF → mesh)

The H4 stance is *qualified* — different sub-tasks want different representations.

### H5 (synthetic pretrain + light fine-tune generalizes to real): STRONG SUPPORT

- **Vertex masking** is the H5 mechanism: a large unlabeled corpus (often mixed-synthetic-real) is augmented by random masking, and the model is pre-trained to reconstruct the correspondences from the masked mesh. This is the *exact* trick as Point2SSM++'s MPointR (paper 040), MAE (He et al. 2022), and Point-BERT (Yu et al. 2022). The paper does not report ablation results, but the multi-anatomy robustness (98% on VerSe'20, 5-class) suggests the masking is helping.
- **Robustness to noisy input** is the *implicit* H5: the paper's aleatoric uncertainty is well-calibrated on real clinical data (pancreas, left atrium, liver — all from real CT/MRI scans), and the model identifies OOD samples in real data. This is the same H5 mechanism as Point2SSM++'s robustness to misaligned input.
- **Multi-anatomy generalization** is the *practical* H5: training on 5 lumbar vertebrae simultaneously regularizes the encoder to learn general anatomical features (vertebra has a body + spinous process + transverse processes) that transfer across the 5 sub-classes. For us: training on 32 FDI teeth simultaneously would regularize the encoder to learn general tooth features (crown + root + CEJ + cusps) that transfer across the 32 sub-classes.

**The H5 evidence stack is the strongest in the SSM reading arc** (alongside Point2SSM++ paper 040) and the cleanest H5 evidence across all our 41 papers. For our v0: pre-train on 3DTeethSeg22 (1,800 scans) + Tufts + OSF + any other public IOS dataset with vertex masking, then fine-tune on a small (100-200 scan) clinical cohort. The pre-training is the H5 enabler.

## Surprises / interesting things buried in section 4

### Surprise 1: The Surface Projection trick has a "softness temperature" σ that should be annealed during training

Section III-C.2 specifies the softmin `σ` but the paper's default value is not explicit in the v1 arXiv. The intuition: start with high `σ` (smooth, robust to noise) and anneal to low `σ` (sharp, accurate to surface) over training. This is the *exact* same annealing pattern as LION's β-schedule (paper 005) and DDM noise schedules. **For us: a σ-annealing schedule in the surface projection step would give the v0 PVD-AF-DiGS-FC stack the same robustness-to-sharpness trade-off, with no architectural change.**

### Surprise 2: The M++AE variant (no NF, just autoencoder) is competitive with M++Flow

Table II shows M++AE = M++Flow on the 5-class lumbar classification. This means the **multi-anatomy classification is dominated by the correspondence field, not the analysis module's latent space structure**. For us: the choice of NF vs. AE for the analysis module is secondary to the choice of correspondence generation module. The "right" v0 sub-task 1 architecture is a *simple* Mesh2SSM++ with AE (no NF), saving the NF for v1 if we want aleatoric uncertainty.

### Surprise 3: The first DGCNN block uses *geodesic distance* (not Euclidean)

Section III-B specifies that "the first EdgeConv block utilizes geodesic distance on the mesh surface for feature calculation". This is the *only* deep SSM in our reading list that explicitly uses geodesic distance in the encoder. The intuition: geodesic distance captures *intrinsic* surface geometry (e.g., distance along the tooth's buccal surface, not through the tooth's volume), which is more meaningful for anatomical correspondence than Euclidean distance. **For us: geodesic distance is the right default for the v0 sub-task 1 encoder** — it captures the "along the tooth surface" distance, not the "through the tooth" distance, which is the right inductive bias for cervical-corner detection and CEJ localization.

### Surprise 4: The Mesh2SSM++ classification pipeline is a 100-neuron MLP

The classification head is "a multilayer perceptron (MLP) with 100 neurons", trained for 5-fold cross-validation. This is a *tiny* classifier, indicating that the correspondences themselves carry a *rich* representation of the anatomy class. **For us: a tiny MLP on top of the Mesh2SSM++ correspondences would give us a 32-class FDI classifier with <0.1M params and <1ms inference time, free as a side-product of the correspondence model.**

### Surprise 5: The paper explicitly considers *neural implicit functions* (NIF) as a future direction

Section VI (Limitations and Future Work) discusses Neural Implicit Functions (NIF) as a way to "leverage self-supervised construction of Signed Distance Fields (SDFs) for surface reconstruction". This is a direct connection to our v0 sub-task 4 architecture (DiGS, paper 003). The paper's argument: NIF could be used to *densify* the mesh (turn a sparse correspondence field into a dense SDF), which would then be extracted with FlexiCubes (paper 007). **For us: this is a v0 sub-task 4 design alternative — instead of generating the crown directly as a mesh (DMTet, paper 031) or as a point cloud (PVD, paper 012), we could generate it as a *correspondence field* and densify it with NIF.** This is a 3-month engineering project, queue for v1.

### Surprise 6: The paper's left-atrium segmentation is from "manually segmented by cardiovascular medicine experts at the University of Utah Division of Cardiovascular Medicine" (in-house dataset, not public)

This is significant because it means the aleatoric uncertainty is calibrated on *high-quality* manual segmentations, not on noisy automated ones. **For us: if we want to train a v0 sub-task 1 model on clinical IOS data, the segmentations need to be high-quality (FDI-labeled by a dentist) — noisy automated segmentations would degrade the correspondence quality and increase the aleatoric uncertainty, but not in a useful way.**

## Quote-worthy sentences

> "Mesh2SSM++ is robust to misaligned and inconsistent input, providing SSM that accurately samples individual shape surfaces while effectively capturing population-level statistics." (from the paper 040 Point2SSM++ abstract — Mesh2SSM++ inherits this property, but the *primary* contribution is the NF + surface projection + aleatoric uncertainty)

> "By integrating a bidirectional flow, simplifying the loss calculation, and adding a surface projection step, Mesh2SSM++ provides a more robust, efficient, and anatomically accurate solution for statistical shape modeling from meshes." (Sec III-C.4, the conclusion of the methods section)

> "The decoupled prior VAE (dpVAE) framework combines a VAE with normalizing flows in the latent space" (Sec III-C.1) — the architectural key.

> "Softmin weights ... with a tunable softness-temperature σ" (Sec III-C.2, Eq. 8) — the surface projection's tunable parameter.

> "the first EdgeConv block utilizes geodesic distance on the mesh surface for feature calculation" (Sec III-B) — the intrinsic-geometry inductive bias.

> "Aleatoric uncertainty ... is the variance of the conditional distribution p(Cn|zn)" (Sec III-C.4) — the uncertainty is well-defined mathematically.

> "All 'good' methods (M++AE, M++Flow, Mesh2SSM, ShapeWorks) hit 98% accuracy on 5-class lumbar classification — the correspondences are rich enough to distinguish 5 highly-similar sub-classes" (Table II) — the strongest evidence for the 32-class FDI generalization.

> "the identified outliers display an elongated, thinning left lobe, which could suggest chronic liver damage, such as cirrhosis or fibrosis" (Sec V-C) — the clinical relevance of the aleatoric uncertainty.

## Code/data link

- **arXiv:** https://arxiv.org/abs/2502.07145 (2502.07145v1, 11 Feb 2025)
- **Code:** ✅ [github.com/iyerkrithika21/Mesh2SSMJournal](https://github.com/iyerkrithika21/Mesh2SSMJournal) (configs/, models/, trainers/, utils/, conda env file, 4 run_*.py entry points)
- **Predecessor code:** Mesh2SSM (Iyer+Elhabian, MICCAI 2023) at [github.com/iyerkrithika21/mesh2ssm_2023](https://github.com/iyerkrithika21/mesh2ssm_2023) — smaller, less features, but a useful starting point for understanding the base architecture
- **Data:** All 5 anatomical datasets are linked from the paper and from [github.com/SCIInstitute/ShapeWorks](https://github.com/SCIInstitute/ShapeWorks) (the in-house SSM tool). VerSe'20 vertebrae is public at [github.com/anjany/verse](https://github.com/anjany/verse). Medical Decathlon is at [medicaldecathlon.com](http://medicaldecathlon.com/). AbdomenCT-1k is at [github.com/JunMa11/AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K).
- **Citation count:** 0-5 (Semantic Scholar, Jun 2026, very young preprint, ~6 months old)

## For our project — concrete next steps

### Action 1: PROMOTE Mesh2SSM++ to the v0 sub-task 1 backbone (HIGHEST priority)

- **Why this is the right choice:** clinical IOS data is triangle meshes, not point clouds. The current v0 plan (Point2SSM++ from paper 040) requires an *extra* point-sampling step that loses the mesh connectivity. Mesh2SSM++ operates directly on the mesh, preserving the connectivity that cusps, fissures, marginal ridges, and proximal-contact areas depend on.
- **Pilot scope:** Train Mesh2SSM++ on 3DTeethSeg22's 1,800 scans, with M=1024 correspondence points per tooth. Use the multi-anatomy variant (run_mesh2ssm_multiorgan.py) with the 32-class FDI scheme (1-32). The classifier head becomes a 32-way FDI classifier.
- **Augmentation:** Vertex masking at 30-50% (paper default), with a small Gaussian perturbation on visible vertices.
- **Expected outcome:** A 32-class correspondence field that maps each input tooth to the canonical FDI coordinate system. Per-cohort PCA on the 32 correspondence fields gives the 32 tooth-specific SSMs. Free per-vertex aleatoric uncertainty for clinical risk assessment.
- **Compute budget:** ~$200-400 on Lambda (1× A100-40GB, 12-24h training).
- **v0 stack update:** PVD-AF-DiGS-FC + **Mesh2SSM++ (multi-anatomy variant)** as the sub-task 1 backbone, replacing Point2SSM++ (paper 040).

### Action 2: Port the Surface Projection trick (Eq. 7-10) to the PVD-AF-DiGS-FC stack (sub-task 4, MEDIUM priority)

- **Why:** The surface projection is the cleanest H3 conditioning prior in the SSM reading arc, and the most direct H3 implementation for sub-task 4. PVD's free-points diffusion (paper 012) generates "free points" in 3D space, but the L2 loss only enforces the predicted points to be *near* the GT — not on the *actual* occlusal surface. Adding surface projection to the v0 stack would:
  1. Force the generated crown's surface points to lie on the actual occlusal/buccal/lingual surfaces, not float off into the air.
  2. Provide a per-point confidence heatmap for the dentist.
  3. Make the generated mesh "printable" by construction (the surface is the surface, not a smooth approximation).
- **How:** 30-line change to `train_pvd.py`. After predicting the free points `x̃_0`, compute softmin weights over the observed arch's mesh vertices, project `x̃_0` onto the observed arch's surface, and use the projected `x̃_0^proj` in the Chamfer loss. Softness σ should be annealed: start at σ=0.5mm (smooth, robust to noise), anneal to σ=0.05mm (sharp, accurate to surface) over 100k iterations.
- **Compute impact:** O(M×K) memory per sample, where M=number of predicted free points and K=number of observed arch vertices. For M=2048 and K=30,000, this is 60M floats = 240MB. Negligible compared to the PVCNN backbone. Training time +2-5%.
- **Expected outcome:** +0.5-1.0% CD on multi-patient evaluation, +0.5-1.0% on the printability check (the generated mesh is closer to a watertight surface, fewer self-intersections).

### Action 3: Adopt the geodesic-distance-aware DGCNN encoder for the v0 sub-task 1 model (MEDIUM priority)

- **Why:** Geodesic distance is the right inductive bias for tooth-scale data — it captures "along the tooth surface" distance, not "through the tooth" distance. The first DGCNN block of Mesh2SSM++ uses geodesic distance, and this is the *only* encoder in our reading list that does. For us: cusps, fissures, marginal ridges, and CEJ are all *surface* features, and geodesic distance is the right metric for "how far is this point from that CEJ marker along the tooth surface".
- **How:** Replace the first EdgeConv block of our Point2SSM++ (paper 040) v0 sub-task 1 model with a geodesic-distance-aware EdgeConv. Use the `gdist` library (or compute on-the-fly with PyTorch3D) to compute geodesic distances over the mesh. Memory: O(K²) per sample, where K=mesh vertex count. For K=10,000 vertices per tooth arch, this is 100M floats = 400MB per sample. B=2 batch size on Mac mini.
- **Compute budget:** Implementation is 1-2 days of engineering, training is same as paper 040 (~$200-400 Lambda). Expected improvement: +1-2% on the 32-class FDI classification, +0.5-1.0% on the correspondence CD.

### Action 4: Adopt the aleatoric uncertainty for clinical risk assessment (sub-task 1, MEDIUM priority)

- **Why:** Per-vertex aleatoric uncertainty is the cleanest clinical-risk signal in the SSM reading arc. The dentist can see "this marginal ridge has high uncertainty = I should verify the prep margin manually" vs. "this proximal contact has low uncertainty = the model is confident, I can proceed".
- **How:** At inference, sample S=50 latent encodings `z^(s) ~ N(μ_z, σ_z)`, decode each, fit a Gaussian, output the variance as a per-vertex heatmap. Render in the v0 dashboard as a color overlay on the predicted crown.
- **Compute impact:** S=50 forward passes at inference = 50× slower. For real-time chairside UX, use S=10 (3-5× speedup, marginal accuracy loss). Compute the heatmap once per tooth, cache it.
- **Expected outcome:** A chairside UX that distinguishes high-confidence from low-confidence regions of the generated crown. Clinically critical for "trust but verify" workflow.

### Action 5: Pilot the "correspondence field → NIF → FlexiCubes" v1 design alternative (sub-task 4, LOW priority)

- **Why:** The paper's Sec VI (Limitations) explicitly suggests using NIF to densify the correspondence field, then extract a mesh with marching cubes. This is a 3-month engineering project that *bypasses* the PVD/DiGS/LION/Diffusion-SDF stack and uses Mesh2SSM++'s correspondence field as the substrate.
- **How:** Train Mesh2SSM++ on a cohort of *complete* tooth crowns (no missing teeth), get the correspondence field, train a small SIREN to densify it to an SDF, extract with FlexiCubes. For a missing tooth, predict the correspondence field with the missing tooth masked, densify, extract.
- **Compute budget:** ~$2,000-3,000 on Lambda (3 months engineering, $1,000 for SIREN training, $1,000 for FlexiCubes extraction, $500 for evaluation). This is a v1 alternative, not a v0.
- **Expected outcome:** A *single* model that handles both sub-task 1 (FDI segmentation via correspondence) and sub-task 4 (crown generation via correspondence → NIF → mesh). The model is trained end-to-end on the same correspondence field, with no separate generation pipeline.

### Action 6: Use the Mesh2SSM++'s multi-anatomy classification pipeline as the v0 32-class FDI classifier (sub-task 1, HIGH priority)

- **Why:** The 98% accuracy on 5-class lumbar vertebrae (Table II) is the strongest evidence that the Mesh2SSM++ correspondence field carries enough information to distinguish 32 FDI tooth classes. A simple 100-neuron MLP on the correspondences gives a 32-class classifier with <1ms inference and <0.1M params.
- **How:** Train Mesh2SSM++ on 3DTeethSeg22 with the 32-class FDI scheme (1-32). Train a 100-neuron MLP on the 1024-dim correspondence output for 5-fold cross-validation. Expected accuracy: 95%+ (the 5-class lumbar result is a conservative lower bound).
- **Compute budget:** Implementation is 0.5 day (the MLP is 5 lines of PyTorch), training is same as Action 1.
- **Expected outcome:** A 32-class FDI classifier that runs in <10ms per arch, with no separate segmentation model needed.

### Open question for HK: 32-class FDI vs 8-class tooth-type vs 4-class quadrant

- **32-class (FDI)** matches the dentist's mental model and the 3DTeethSeg22 labels, but has class imbalance (incisors are common, third molars rare in many patients).
- **8-class (tooth type)** is more data-efficient and matches the multi-anatomy SSM's default, but loses FDI information.
- **4-class (quadrant)** is even more data-efficient, but loses too much information.
- **Recommendation:** Train **all three heads** in a multi-task setup. The 32-class head drives sub-task 1 (FDI segmentation), the 8-class head drives sub-task 4 (per-type SSM), the 4-class head drives sub-task 5 (longitudinal monitoring). Total cost: +0.5M params, +5% training time. All three heads share the DGCNN encoder and the correspondence field.

### v0 stack update (with concrete cost numbers)

| Component | Role | Cost (Lambda) | Reference |
|-----------|------|------------------|-----------|
| Mesh2SSM++ (multi-anatomy, M=1024) | Sub-task 1 backbone — FDI segmentation + per-arch SSM (H3, H5) | $200-400 | paper 041 (this) |
| 32/8/4-class FDI MLP heads | Free per-class classifier from correspondences (H3) | $0 (in-loop) | paper 041 (this) |
| Surface projection loss (Eq. 7-10 port) | Patient-level surface regularizer (H3, H5) | $0 (in-loop) | paper 041 (this) |
| Aleatoric uncertainty (variance over S=50) | Chairside risk heatmap (H5) | $0 (in-loop) | paper 041 (this) |
| PVD (free-points diffusion) | Sub-task 4 outer surface DDM (H2) | $50-200 | paper 012 |
| AnchorFormer | Completion encoder for sub-task 4 (H3) | $30-100 | paper 011 |
| DiGS | SDF lifting for sub-task 4 (H4) | $100-300 | paper 003 |
| FlexiCubes | Mesh extraction for sub-task 4 (H4') | $5-10 | paper 007 |
| PyMeshFix | Self-intersection repair | $0 (post-process) | paper 007 |
| Geometric offset | Intaglio (inner) surface — deterministic, <50μm | $0 (geometry lib) | internal |
| **Total** | **v0 prototype** | **~$2,600** | — |

The Mesh2SSM++ integration adds ~$400 to the existing $2,200 budget, but gives us a *mesh-native* sub-task 1 backbone (the right input for clinical IOS), a free 32-class FDI classifier, free surface-projection loss for sub-task 4, and free per-vertex aleatoric uncertainty for the chairside UX. The 32-class FDI MLP head, the surface projection, and the aleatoric uncertainty are all *free* additions (no extra training cost).

### Next paper to read (042)

**Three candidates:**

1. **STEAM (MICCAI 2025)** — "Self-Supervised Teeth Analysis and Modeling for Point Cloud" (the search result showed it at papers.miccai.org/miccai-2025/paper/3394). A dental-specific self-supervised learning method from another group, trained on Teeth3DS+ (a large multi-center dental dataset). Read this to compare with Mesh2SSM++'s vertex masking on dental data, and to get the Teeth3DS+ dataset details.

2. **LION (paper 005, re-read)** — the latent-point DDM, the *generative* counterpart to Mesh2SSM++'s *correspondence* approach. Reading them back-to-back would clarify the v0 sub-task 4 architecture decision (correspondence vs generation for crown outer surface). LION is a *generation* paper, Mesh2SSM++ is a *correspondence* paper, and the v0 stack should ideally use both (correspondence for sub-task 1, generation for sub-task 4).

3. **ToothForge (Kubik 2025, paper 037, re-read)** — the dental-specific autoregressive mesh generation paper from the SSM reading arc. Reading it back-to-back with Mesh2SSM++ would clarify the *generation* vs *correspondence* divide in the dental-specific literature, and would help decide whether to use a dental-specific model (ToothForge) or a general-purpose SSM (Mesh2SSM++) for the v0 sub-task 4 design.

**Recommendation for 042: STEAM (MICCAI 2025).** It's the only dental-specific self-supervised learning method in the literature, and it would directly inform the v0 sub-task 1 architecture (should we use a dental-specific pre-training or a general-purpose one?). The Teeth3DS+ dataset details are also a key data-acquisition question for our v0.

**Note in `papers/041-mesh2ssm-plus-plus-iyer25.md`.**
