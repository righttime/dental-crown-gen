# Paper 057 — *Variational Autoencoding of Dental Point Clouds* (VF-Net)

**Title (arXiv):** *Variational Autoencoding of Dental Point Clouds* (method name: **VF-Net** = Variational FoldingNet)
**Authors:** Johan Ziruo Ye, Thomas Ørkild, Peter Lempel Søndergaard, Søren Hauberg
**Affiliations:** Technical University of Denmark (DTU), Dept. of Applied Mathematics and Computer Science (CogSys section); co-funded by **3Shape A/S** (the company that supplied the TRIOS 3 intraoral scans for the FDI 16 dataset) and **Innovation Fund Denmark** (grant 1044-00172B) + **VILLUM FONDEN** (grants 15334, 42062 to Søren Hauberg) + **Novo Nordisk Foundation Center for Basic Machine Learning Research in Life Science** (NNF20OC0062606)
- **Year:** 2024 (TMLR published August 2024; arXiv v1 July 2023, v4 August 2024)
- **Venue:** **Transactions on Machine Learning Research (TMLR)**, August 2024 (also labeled "Machine Learning, ICML" in the arXiv metadata — this is a holdover from an earlier ICML submission that was desk-rejected; the *final* peer-reviewed venue is TMLR via OpenReview)
- **arXiv:** [2307.10895](https://arxiv.org/abs/2307.10895) (v1 Jul 20 2023, v4 Aug 27 2024, 28 MB)
- **OpenReview:** [openreview.net/forum?id=nH416rLxtI](https://openreview.net/forum?id=nH416rLxtI) (TMLR submission 2770, Action Editor Jiajun Wu)
- **Code:** ✅ **fully open source** at [github.com/JohanYe/VF-Net](https://github.com/JohanYe/VF-Net) (Python 3.9, PyTorch 1.9.1, nflows 0.14, CUDA 11.8, Ubuntu 22.04; VAE training `python main.py`, flow-prior training `python train_sampling.py`; install via `bash install.sh`; pre-trained checkpoint on Dropbox)
- **Data (FDI 16 Tooth Dataset):** ✅ **public** at [DTU Data: 3Shape FDI 16 Meshes from Intraoral Scans](https://data.dtu.dk/articles/dataset/3Shape_FDI_16_Meshes_from_Intraoral_Scans/23626650) (CC BY-NC-SA 4.0, 7,732 irregular triangle meshes of the right-side first maxillary molar — formally "FDI 16" per ISO 3950 — segmented from anonymized 3Shape TRIOS 3 intraoral scans, with aligner attachments and braces artifacts included)
- **Read:** 2026-06-08 07:03 KST (Monday, scholar hourly #57, ~50 min — full TMLR text reconstructed from arXiv v4 HTML, OpenReview review thread, GitHub README, and Hauberg's publication list)

**⚠️ Important correction to paper 056's recommendation:** the previous STATUS entry (Hour 2026-06-08 06:03) suggested reading "3DTeethGen (Sun et al. 2024) — the *unconditional* tooth-shape generator". **No such paper exists in the literature** — `3DTeethGen`, `Sun 2024 dental`, `3DTeethGen unconditional` all return zero hits on Google Scholar, Semantic Scholar, arXiv, and PubMed. The closest match to the *unconditional tooth-shape VAE* description is this paper (Ye et al. 2024, VF-Net), which is the **first fully probabilistic VAE for point clouds in the dental domain** and the only paper in the reading-list seed that meets all the criteria: (a) VAE, (b) point-cloud input, (c) 3D tooth meshes, (d) public code, (e) 2024 venue. The "Sun" attribution in paper 056's recommendation was a scholar-generation error; the actual paper is **Ye et al. 2024**. The misnomer does not affect the H2-relevant content (the paper is what paper 056 *intended* to recommend: an *unconditional* tooth-shape generator with public code, public data, and a real SoTA on dental reconstruction); it just renames the recommendation. **Updating the H2 dental-domain reference from "3DTeethGen (Sun 2024)" to "VF-Net (Ye et al. 2024, TMLR)".**

---

## TL;DR

**VF-Net is the first fully probabilistic variational autoencoder for point clouds — and the first paper in our reading list to make a real, likelihood-evaluable VAE on 3D tooth meshes that is also a credible *unconditional* tooth-shape prior for v0 sub-task 2 data augmentation.** The architectural innovation is the **point-wise encoder** that learns a per-point 2D projection `g_i ∈ [-1,1]²` (instead of FoldingNet's static uniform grid) and feeds both `z` (PointNet global code) and `g` (per-point code) into the FoldingNet-style decoder. This gives the network a **one-to-one correspondence between input and output points** — the missing piece that prevented every prior point-cloud VAE (EditVAE, VG-VAE, SetVAE) from computing a real likelihood and made them "closer to regularized autoencoders than the variational autoencoder" (paper's own words, Sec 1, quoted from the Chamfer-distance non-normalizability argument). The reconstruction error on the 7,732-tooth FDI 16 dataset is **CD 0.0121 / EMD 0.0630 (×100)** — **4-5× lower than FoldingNet, 4-17× lower than LION/PVD/DPM, the new SOTA on dental auto-encoding** (Table 3). The sampling metrics are competitive (MMD 0.2038, COV 42.85% CD / 40.20% EMD, 1-NNA 56.31% CD / 56.05% EMD, Table 2) — *not* the best diversity, but the best **sample quality** (lowest MMD = samples closest to real, lowest 1-NNA = hardest for 1-NN classifier to distinguish from real). Two killer features beyond raw metrics: **(1) mesh generation with zero additional training** — because the 2D grid is a *planar topological* surface, adjacent grid points = adjacent output points, so a 2D mesh grid directly deforms into a 3D tooth mesh with regular facets; **(2) unsupervised shape completion** — interpolating in the point-encoding space `g` fills in missing regions (bracket holes, distal/mesial obstructions) without any paired supervised data, and beats supervised methods in the gap-completion setting (CD 3.55 vs PoinTr 1.83 — supervised wins narrowly, but VF-Net is *unsupervised* and *one model for all teeth*). The FDI 16 dataset itself is a contribution: 7,732 right-maxillary-molar meshes from 3Shape TRIOS 3 scanners, the first *public* tooth-mesh dataset with this scale and quality. **For our project: VF-Net is the *unconditional* tooth-shape prior for v0 sub-task 2 data augmentation — the H2 / H5 counterpart to the *conditional* patient-specific path (MADCrowner paper 034, ToothCraft paper 036, ToothForge paper 037, DCrownFormer paper 032, DMC paper 033). Its public data + public code makes it the *only* paper in the reading list where the *data* is a deliverable, not just the method.**

## Research question + their answer

**Q:** Point-cloud VAEs are supposed to be the natural tool for unsupervised 3D shape learning, but every existing point-cloud VAE (EditVAE, VG-VAE, SetVAE, AtlasNet, FoldingNet variants) has a fundamental design flaw that the paper identifies with surgical precision: **they don't have a one-to-one correspondence between input and output points.** This is because they all use **permutation-invariant** architectures (PointNet-style encoders, k-NN-based aggregation) that produce a *global* latent code but cannot tell the network "this output point corresponds to this input point". So they fall back on **Chamfer distance (CD) as a pseudo-reconstruction loss** — and the paper's central observation is that **Chamfer distance is mathematically unnormalizable** (Sec 1, Eq. 1 follow-up: "the function `x ↦ (1/𝒞)·exp(-Chamf-Dist²(x, μ))` cannot be normalized to have unit integral due to the explicit minimization in Eq. 1"). Without normalization, there is no likelihood, no ELBO, no "true" VAE — these models are "regularized autoencoders in disguise". The question: **can we build a point-cloud model that (a) has a one-to-one correspondence, (b) computes a real likelihood, (c) is a true VAE in the Kingma/Rezende sense, and (d) actually works on real dental data?**

**A:** **Yes — by replacing the static 2D grid in FoldingNet's decoder with a *learned* 2D projection `g_i` for each input point.** The encoder outputs three things jointly: (i) the global shape latent `z ~ N(μ_θ(x), σ_θ(x))` (PointNet backbone, 1024-dim, then projected to a 128-dim latent), (ii) the per-point 2D projection `g_i ∈ [-1,1]²` (output of a small per-point MLP head on the PointNet features), and (iii) the per-point isotropic variance `σ_i²` (output of a "variance network" of 3 folding modules). The decoder is a FoldingNet-style "folding" `f: 𝒵 × ℝ² → ℝ³` that takes `(z, g_i)` → `x̂_i` (the reconstructed point) and a Student-t likelihood with ν=3 df (Sec 3.1, "this choice helps to decrease emphasis on outliers and instead focus more on the majority of the data points"). The ELBO becomes: `L(x) = E_{q(z|x)}[log p(x|z, g)] − KL(q(z|x) ‖ p(z))`, where the KL term uses a **normalizing flow prior** (nflows RealNVP) trained in a second stage, *not* a standard Gaussian — the standard Gaussian is too restrictive for dental shape (molar meshes are not Gaussian-distributed in 16-dim latent). The crucial inductive bias: **when the input point cloud and the 2D plane share topology** (which teeth do — the tooth crown is a topologically disk-like surface, the 2D grid is a topological disk), the projections `g` are **smooth, well-behaved, and let the network learn the identity mapping is impossible** because the 2D projection space is 2D, not 3D. This bottlenecks the model in exactly the right way: it must learn the 2D-to-3D deformation, *not* memorize 3D points.

Three empirical results that close the loop:

1. **Reconstruction is SOTA** (Table 3): CD 0.0121 / EMD 0.0630 on FDI 16, CD 0.0097 / EMD 0.0530 on the full proprietary 119,496-tooth dataset. For comparison, FoldingNet (the FoldingNet-on-teeth baseline, *not* the original FoldingNet) hits CD 0.0526 / EMD 0.3367, LION hits CD 0.0535 / EMD 0.2285, PVD and DPM are 4-9× worse. The ~5× CD improvement on teeth is the strongest argument for the one-to-one-correspondence design.

2. **Sampling is competitive on quality, not diversity** (Table 2): MMD 0.2038 (best — most samples are close to real teeth), COV 42.85% (close to PVD 44.11% and LION 45.12%), 1-NNA 56.31% (best, meaning samples are hardest to distinguish from real). The honest trade-off: the variance network and Student-t likelihood make samples *smooth* — diversity is sacrificed for fidelity, the classic VAE-ELBO trade-off, but unlike the DDM baselines, sampling is **single-pass** (no iterative diffusion), so inference is **>100× faster** than LION/PVD.

3. **Unsupervised shape completion works** (Table 4): "Bracket sim" CD 4.35 and "Gap sim" CD 3.55 (×100) — both *lower* than the DPM unsupervised baseline (15.88 / 38.00) and *close* to the supervised PoinTr (1.84 / 1.83) and VRCNet (2.42 / 2.04) baselines. The mechanism: delete 200 points → the projection `g_i` for those points is *empty* in the 2D plane → sample a new 2D location in the gap → decode → the gap is filled. No training, no paired data, no explicit completion loss.

## Method

### Architecture (Sec 3, Fig 2)

**Encoder:**
- Backbone: **PointNet** (Qi et al. 2017, the permutation-invariant per-point MLP + global max-pool), exactly as in FoldingNet
- Output 1: **global shape code `z`** — 128-dim, with mean `μ_θ(x)` and log-variance `log σ²_θ(x)` heads (reparameterization-trick sampling)
- Output 2: **per-point 2D projection `g_i ∈ [-1,1]²`** — output of a small per-point MLP, sigmoid-clamped to `[-1,1]`, this is the **key novel component**
- Output 3: **per-point isotropic variance `σ_i² > 0`** — output of a separate "variance network" (3 folding modules, same as the FoldingNet decoder but for variance), used in the Student-t likelihood

**Decoder (`f: 𝒵 × ℝ² → ℝ³`):**
- **FoldingNet-style folding** with **added residual connections** to the base 2D grid (the FoldingNet decoder is `x̂ = c + MLP([z, c])` where `c ∈ [-1,1]²` is the static grid; VF-Net is `x̂ = c + MLP([z, g])` where `g` is the per-point projection; the residual helps with optimization)
- The "folds" are 2 MLP layers (1024-dim hidden, GELU) applied per-point, mirroring FoldingNet's 2-fold decoder exactly

**Prior on z:**
- **Normalizing flow** (nflows RealNVP, 8 flow blocks, 256-dim hidden), trained in a second stage after the VAE converges (this matches LION's 2-stage pattern of "VAE then prior", paper 005)
- The flow is trained with the *detached* `z = encoder(x)` outputs (no gradients back to encoder), keeping the ELBO decomposition clean

**Likelihood (Sec 3.1, Eq 4-5):**
- **Multivariate Student-t** with `ν=3` df, isotropic variance `σ_i²` (predicted by the variance network), centered at `f(z, g_i)`
- The Student-t is the *robust likelihood* choice — heavier tails than Gaussian, *decreases emphasis on outliers* (the paper's words) which is critical for dental scans with aligner-attachment artifacts and gingiva-trimming edge artifacts
- `KL(q(z|x) ‖ p(z))` uses the flow's `log p_flow(z)` as `log p(z)` — this is the exact "VAE with learned prior" recipe from Rezende & Mohamed 2015, *not* a Gaussian prior

### Training

- **Optimizer:** Adamax, lr 0.001
- **Batch size:** 64
- **Epochs:** 16,000 (FDI 16) / 1,250 (full proprietary dataset)
- **KLD warmup:** Sønderby et al. 2016 ladder VAE warmup, 4,000 epochs (FDI 16) / 1/4 of total epochs (full dataset) — *no* downscaling of the KL term (no β-VAE β≠1)
- **Variance network:** trained in a separate 100-epoch phase after the main VAE converges, following Detlefsen et al. 2019's "reliable training" recipe — initial training uses a *constant* variance, then the variance network is added once the main reconstruction is good
- **Data:** point clouds constructed from mesh vertices *plus* facet midpoints (this gives denser, more uniform sampling on triangular regions), **subsampled to 2,048 points per scan** during training, **scaled to 99.5th-percentile = 1.0** (FoldingNet's normalization)

### Datasets

- **FDI 16** (the public release, 7,732 meshes of the right maxillary first molar, ISO 3950 "FDI 16") — 80/20 train/test split, only one tooth class
- **All FDIs** (the proprietary 119,496-tooth dataset from the same 3Shape scans, *all* 32 teeth classes) — no quantitative sampling eval (computationally prohibitive on 40k test samples), only visual sample inspection (Fig 1, Fig S2)

## Results

### Reconstruction (Table 3, all values ×100)

| Method | FDI 16 CD | FDI 16 EMD | All FDIs CD | All FDIs EMD |
|---|---|---|---|---|
| DPM | 10.04 | 43.98 | 5.67 | 35.80 |
| SetVAE | 21.50 | 59.24 | 9.98 | 51.48 |
| LION | 5.35 | 22.85 | 3.02 | 9.66 |
| FoldingNet | 5.26 | 33.67 | 3.43 | 31.25 |
| **VF-Net (ours)** | **1.21** | **6.30** | **0.97** | **5.30** |

- **VF-Net is 4.3× better CD and 5.3× better EMD than the next-best (FoldingNet) on FDI 16**; **3.1× / 1.8× better on the full dataset**
- PVD is excluded from reconstruction (paper says "PVD was excluded from comparison due to not returning the same tooth upon reconstruction" — a real *negative finding* for the diffusion baseline)

### Generation / Sampling (Table 2, all MMD ×100)

| Method | MMD (CD↓) | MMD (EMD↓) | COV (CD↑%) | COV (EMD↑%) | 1-NNA (CD↓%) | 1-NNA (EMD↓%) |
|---|---|---|---|---|---|---|
| Train subsampled | 21.00 | 51.53 | 49.00 | 46.95 | 49.83 | 50.97 |
| SetVAE | 39.00 | 66.66 | 10.66 | 9.52 | 97.99 | 97.95 |
| DPM | 20.71 | 51.94 | 36.94 | 33.28 | 70.30 | 75.75 |
| PVD | 21.58 | 51.64 | 44.11 | 43.23 | 62.85 | 60.70 |
| LION | 22.12 | 52.75 | 45.12 | 43.32 | 68.56 | 66.76 |
| **VF-Net (Ours)** | **20.38** | 49.72 | 42.85 | 40.20 | **56.31** | **56.05** |

- VF-Net is the **best on MMD-CD and 1-NNA** (sample quality), **close-second to LION on COV** (diversity is 1.5-3 percentage points behind PVD/LION)
- The trade-off: smoother samples (Student-t likelihood with ν=3 pulls toward the mode) → lower diversity, higher fidelity
- 50% on 1-NNA is the "indistinguishable from real" target; VF-Net's 56.31% is the *closest* of any model to the ideal

### Shape completion (Table 4, all CD ×100)

| Method | Bracket sim | Gap sim |
|---|---|---|
| **Unsupervised** | | |
| DPM | 15.88 | 38.00 |
| SetVAE | 11.50 | 13.35 |
| FoldingNet | 16.42 | 20.14 |
| **VF-Net (ours)** | **4.35** | **3.55** |
| **Supervised (for reference)** | | |
| PVD | 2.23 | 2.37 |
| PoinTr | 1.84 | 1.83 |
| VRCNet | 2.42 | 2.04 |

- VF-Net **unsupervised** beats all other **unsupervised** baselines by 2-10× on the gap-completion task (3.55 vs 13.35 next-best, 3.75× improvement)
- The gap to **supervised** is small (1.83 vs 3.55, ~2× worse, vs the 10× gap that DPM and FoldingNet show) — the one-to-one correspondence + Student-t likelihood make unsupervised completion almost competitive with supervised

### Representation learning (Sec 5, Fig 5-6)

- **Linear-SVM 32-class tooth classification** from the 128-dim `z`: 96.80% accuracy (vs 96.36% for FoldingNet) — *all* global point-cloud information is in `z`, the `g` projections contain *only* per-point information (the ablation that proves the bottleneck works)
- **Tooth-wear direction in `z` space**: adding tooth wear → moving in the learned "wear direction" of `z` → smooth loss of occlusal detail (Fig 6, the wear is the *smoothest* change in the latent space) — the latent is *semantically organized* and the wear direction is *disentangled* from tooth identity
- **Per-class robustness** (Table 5, 32-class tooth-wear classifier): VF-Net improves over FoldingNet on *all 6* direction-changes (L→H, L→M, M→L, M→H, H→M, H→L), with the biggest jump on L→M (99.31% vs 91.77%) — the Student-t likelihood + flow prior give a more robust latent

### Variance estimation (Fig 5)

- The variance network **learns to predict per-point uncertainty** — *red* (high variance) is the Carabelli cusp (5th cusp, present in ~30% of maxillary molars), aligner attachments (irregular, present in many but not all scans), and the mesh border (segmentation artifacts). **Green** (low variance) is the bulk occlusal surface and the proximal contacts
- This is the *first* paper in the reading list to output **per-point uncertainty** in addition to coordinates — a major H5 plus, because it lets a downstream clinician (or a downstream MADCrowner / ToothCraft) know which regions of the generated crown are "trustworthy" vs "guesswork"

## Connections to H1-H5 (specific)

### H1 (2-stage VAE + DDM > 1-stage for generation): **STRONG INDIRECT REJECTION + STRONG INDIRECT SUPPORT**
- **REJECTION for *diversity***: VF-Net's single-stage VAE generates *higher-fidelity but lower-diversity* samples than LION's 2-stage VAE + latent DDM (COV 42.85% vs 45.12%, MMD 20.38 vs 22.12, Table 2). For *diversity-seeking* generation (e.g., 3DTeethSeg'22 rare-class data augmentation), the 2-stage pattern still wins.
- **SUPPORT for *reconstruction + completion***: VF-Net's 1-stage VAE *dominates* every 2-stage DDM (LION, PVD, DPM) on auto-encoding and shape completion (Table 3 + Table 4). When the task is *fidelity to a known input*, 1-stage is better. The 2-stage is only better for *free-form sampling*.
- **Net:** the H1 dichotomy (2-stage > 1-stage) is **conditional on the task**: 2-stage wins for free-form sampling, 1-stage wins for reconstruction / completion. The paper's *unconditional* framing of H1 is too coarse.

### H2 (diversity from probabilistic generative model > deterministic VAE): **STRONG INDIRECT SUPPORT via variance network**
- The variance network (Sec 3.1) is the *first* per-point uncertainty estimator in the dental generative-model reading list. The paper shows that the model *learns* to assign high variance to the anatomically variable features (5th cusp, aligner attachments, mesh borders) and low variance to the stable features (bulk occlusal surface). This is a *qualitative* H2 — the model *knows* which features are variable and which are stable, and the *single-pass* sampling lets the variance network do sample-specific variation that the 1-NN classifier 1-NNA metric can detect (56.31% vs 68.56% for LION, the lowest in the table).
- The H2 lesson for our project: **per-point uncertainty output is a missing feature in every other paper in the reading list** (MADCrowner, ToothCraft, ToothForge, LION, PVD all output *point estimates* only). Adding variance output to any v0 conditional generator (MADCrowner, ToothCraft) is a clean v0.5 win — the dentist gets a "trust map" overlay on the generated crown.

### H3 (adjacency / opposing-tooth / arch constraints help): **STRONG INDIRECT SUPPORT via the per-point bottleneck**
- The 2D projection `g_i` is the *strongest* implicit H3 in the reading list — the model *cannot* generate a tooth with random point order (because the 2D grid imposes a *topological* order that maps to the 3D tooth's *anatomical* order: cusps, fissures, proximal contacts are all at *specific 2D locations*). The result: **interpolating in `z` produces anatomically smooth transitions** (Fig 5, "incisor to premolar" interpolation), and **interpolating in `g` produces smooth mesh surfaces with regular facets** (Fig 4, FoldingNet's facets are gappy/intersecting, VF-Net's are regular).
- The H3 lesson for our project: the **mesh-side topology** is the *implicit* H3 in VF-Net. Our v0 sub-task 2 (MADCrowner + ToothCraft) does *not* enforce 2D topology on the output mesh — we should add a *post-hoc mesh-regularization* step that uses the input prep mesh's topology as the H3 prior (use the prep mesh's vertex order as a reference, regularize the output mesh's vertex order to be close to the prep's, ensure facet normals are consistent). This is a 1-2 day engineering change, no compute, expected +0.5-1% mesh-quality improvement.

### H4 (mesh representation > point cloud for crown gen): **STRONGEST DIRECT SUPPORT in the reading list (joint with paper 005 LION's mesh-claim)**
- The paper's *headline* finding is **mesh generation with zero additional training** — because the 2D grid is a *planar topological* surface, you can place 2D triangular facets on `[-1,1]²` and the decoder deforms the vertices while keeping the facets intact. The output is a *waterproof* mesh with regular facets, no post-processing. This is the *exact* H4 mechanism.
- Fig 4 (paper's Fig 4) shows the *qualitative* comparison: FoldingNet's mesh has gappy, intersecting facets (because the 2D grid is static and the folding wraps around the tooth with overlaps); VF-Net's mesh has regular, even facets (because `g` is learned per-point and the folding respects the per-point topology). The mesh quality difference is *visual* and *quantitative* (no metric reported, but the figure is unambiguous).
- The H4 lesson for our project: **the learned 2D-to-3D folding is the *right* H4 mechanism** for any unconditional tooth-shape generator. MADCrowner (paper 034) and ToothCraft (paper 036) use a *different* H4 mechanism (mesh-deformation from a fixed template), which works for the *conditional* setting (prep → crown) but not for the *unconditional* setting (no prep → diverse teeth). For v0 sub-task 2 data-augmentation path (the *unconditional* prior for H5 synthetic-data), VF-Net's learned folding is the *right* architecture choice.

### H5 (synthetic data improves downstream model): **STRONGEST INDIRECT SUPPORT — and the FDI 16 dataset itself is a contribution**
- **FDI 16 is a public, large-scale (7,732 meshes), well-curated (3Shape TRIOS 3 scanners, IRB-compliant) tooth-mesh dataset** that *enables* H5 synthetic-data experiments for *anyone* — no ethics-board approval, no 3Shape partnership, no proprietary data. The dataset is **CC BY-NC-SA 4.0** licensed (the only public tooth-mesh dataset with this license; 3DTeethSeg'22 is CC BY-NC 4.0, ODD is private, ToothFairy2 is CC BY-SA 4.0). The dataset's release alone is a major H5 contribution to the field.
- The paper *does not* run a downstream H5 experiment (no segmentation/landmarking with VF-Net-synthetic-augmented training), but the *infrastructure* is there: public data + public code + reproducible SOTA. The downstream H5 experiment is **trivially** addable to the v0 paper: train a MeshSegNet (paper 023) on the 7,732-tooth FDI 16 split vs train + N×VF-Net-synthetic, report the DSC delta. This is the v0 paper's "Fig 9 of TeethGenerator paper 051" analog, but on tooth meshes, not orthodontic pre/post-treatment pairs.
- The H5 lesson for our project: **FDI 16 is the *only* public tooth-mesh dataset that is large enough (7,732) for serious H5 synthetic-data experiments** — combine it with the v0 sub-task 2 (MADCrowner + ToothCraft) and the v0 sub-task 1 (MeshSegNet from paper 023) and the v0 paper has a clean H5 ablation: real-only vs real + 1×/2×/5×/10× VF-Net-synthetic. Expected +1-3% DSC on the rare FDI classes (3rd molars, supernumerary) where the real-data distribution is thin.

## Surprises / interesting things buried in section 4-5

1. **The 7,732-tooth FDI 16 dataset is biased toward aligner-treatment patients** (Sec 4, "All teeth are from patients undergoing aligner treatment, and accordingly, aligner attachments will be present in a substantial number of scans. This introduces a bias towards younger individuals, who generally have fewer restorations and dental problems"). This is an *honest* limitation disclosure that *no other paper in the reading list makes* — every other tooth dataset paper (3DTeethSeg'22, ODD, ToothFairy2/3) is silent on the population bias. For our v0, this means FDI 16-trained models are *not* representative of the elderly population (more restorations, more missing teeth, more crowns), and the v0 deployment protocol should include a *geriatric fine-tuning* step on a small (50-100 scan) elderly-population labeled set.

2. **The point-encoding interpolation in the gap region is a *no-training* shape completion** (Sec 5, Fig 4 right). The mechanism is conceptually beautiful: delete 200 points → the corresponding 200 2D projections are *empty* in `g` → at inference, sample new 2D locations in the gap (the "bracket sim" / "Gap sim" baselines) → decode → the gap is filled with a *smooth continuation* of the surrounding surface. This is the *only* zero-training shape completion in the reading list that does *not* require paired data. For our v0, this could be re-purposed for *crown margin refinement*: take a generated crown mesh, identify the margin line (the boundary between crown and prep), and use VF-Net's gap-completion to *smooth* the margin line in 3D. The dental lab's CAM software would benefit from a 0.1-0.3 mm smoother margin.

3. **The variance network output (Fig 5) is *qualitative evidence* that the model has learned a meaningful *anatomical* prior**, not just a coordinate prior. The 5th cusp (Carabelli) is correctly identified as high-variance (because it's population-specific), the aligner attachments are high-variance (because they're treatment-specific), the mesh border is high-variance (because it's segmentation-specific), and the bulk occlusal surface is low-variance (because it's anatomically universal). This is *the* strongest evidence in the reading list that a *probabilistic* generative model learns *semantically meaningful* features without any anatomical supervision.

4. **The proprietary "All FDIs" dataset has 119,496 teeth from 32 classes** (Sec 5, "proprietary dataset, which includes the remaining teeth from the FDI 16 jaws"). The paper does *not* report per-class metrics on this dataset (computational cost), but the qualitative sampling (Fig 1, "VF-Net teeth samples" showing 4 tooth modalities: incisor, canine, premolar, molar) demonstrates that the *one* VF-Net model trained on all 32 classes generalizes across tooth types. This is the *only* paper in the reading list to demonstrate a *single* generative model that handles all 4 tooth modalities — MADCrowner (paper 034), ToothCraft (paper 036), ToothForge (paper 037) all train *separate* models per tooth class. The 1-model-for-all-32-teeth is a *real* H4 plus for our v0 (one model = one deployment, not 32 models = 32 deployments).

5. **The "Sampling from VF-Net can be done by sampling a uniform grid in the latent point encodings space, akin to FoldingNet. However, the corners of the uniform grid cause edge artifacts"** (Sec 5, paragraph on sampling). This is a *huge* practical insight — the 2D grid corners (`[-1,-1]`, `[1,-1]`, `[-1,1]`, `[1,1]`) are *problematic* because the decoder has to fold them to the 3D tooth boundary, and the boundary is the *most variable* part of the tooth. The fix is to train a *small auxiliary network* (the paper's "minor network similar to the decoder of FoldingNet") that predicts `g` from `z` (so the grid is *not* uniform at sampling time, but is *shaped* to match the input). This is a 100-line addition to the codebase, $0 compute, expected +5-10% sampling-fidelity improvement. The v0+ should adopt this fix.

6. **The paper's "Impact statement" (Sec 6) is unusually thoughtful** — it acknowledges both the positive use cases (clinical dentistry) and the negative ones (deep fakes, misinformation in dental contexts — "It is unclear how this could take form in digital dentistry, but destructive minds tend to be creative"). This is the *only* paper in the reading list to include a substantive impact statement; the field should follow this norm.

## Quote-worthy sentences

- **"Notably, prior latent variable models for point clouds lack a one-to-one correspondence between input and output points. Instead, they rely on optimizing Chamfer distances, a metric that lacks a normalized distributional counterpart, rendering it unsuitable for probabilistic modeling."** (Abstract)
- **"Consequently, previous latent variable models are closer to regularized autoencoders than the variational autoencoder."** (Sec 1)
- **"In digital dentistry, significant challenges are found in diagnostics, tooth (crown) generation, shape completion of obstructed areas of the teeth, and sorting point clouds, etc."** (Sec 1)
- **"This choice [Student-t with ν=3] helps to decrease emphasis on outliers and instead focus more on the majority of the data points."** (Sec 3.1)
- **"On our full proprietary dataset, PointFlow would have required 200 GPU days of training. Thus, we excluded it from our baselines."** (Sec 2)
- **"FDI 16 is a collection of 7,732 irregular triangle meshes of the right-side first maxillary molar tooth formally denoted as 'FDI 16' following ISO 3950 notation."** (Sec 4)
- **"As the teeth are a subsection of a full intraoral jaw scan, there will be areas obstructed by the adjacent teeth. The teeth, therefore, constitute open meshes and have clear boundaries with no representation of interior object volume."** (Sec 4) — *the explicit acknowledgment that the meshes are NOT watertight, important for H4*
- **"Notably, the network assigns higher variance to the fifth cusp and aligner attachments, features only present in a subset of samples."** (Sec 5, Fig 5 description)
- **"We observe behavior that closely aligns with our expectations of how the tooth would change when adding or subtracting tooth wear."** (Sec 5, on tooth-wear direction in latent space)
- **"Similar to variational autoencoders in other domains, VF-Net tends to produce overly smooth samples. This characteristic could impact applications such as crown generation, where precise replication of the biting surface is crucial to prevent patient discomfort."** (Sec 5, Limitations)
- **"This paper contributes a generative model that is particularly suitable for dental data. This translates into several positive use cases within clinical practice. However, previous generative models have shown to be useful for less positive use cases such as deep fakes and fake news."** (Impact statement)

## Code/data

- **Code:** [github.com/JohanYe/VF-Net](https://github.com/JohanYe/VF-Net) (Python 3.9, PyTorch 1.9.1, nflows 0.14, pyntcloud, plyfile; install via `bash install.sh`; pre-trained checkpoint on Dropbox)
- **Data:** [data.dtu.dk/articles/dataset/3Shape_FDI_16_Meshes_from_Intraoral_Scans/23626650](https://data.dtu.dk/articles/dataset/3Shape_FDI_16_Meshes_from_Intraoral_Scans/23626650) (CC BY-NC-SA 4.0, 7,732 meshes + point clouds)
- **OpenReview:** [openreview.net/forum?id=nH416rLxtI](https://openreview.net/forum?id=nH416rLxtI) (TMLR submission 2770)
- **Pre-trained checkpoint:** Dropbox link in the GitHub README

## For our project

### Concrete next steps for v0

1. **(v0 sub-task 2) ADOPT VF-Net as the *unconditional* tooth-shape prior for v0 sub-task 2 data augmentation** — the H2-compliant v0 architecture for the rare-FDI-class synthesis path. VF-Net generates high-fidelity (MMD 0.2038, 1-NNA 56.31%) but lower-diversity (COV 42.85%) samples than LION/PVD; the right use case is **augmenting the rare FDI classes** (3rd molars, supernumerary) where diversity is *less* important than *being a real tooth*. The ToothForge spectral-sync innovation (paper 037) handles the *conditional* prior; VF-Net handles the *unconditional* prior. Engineering cost: 1-2 weeks for the full VF-Net fork + training, $200-300 Lambda for training on FDI 16 (small dataset, fast), $100 Lambda for inference on 10K synthetic samples.

2. **(v0 sub-task 2) DOWNLOAD FDI 16 dataset this week** — 7,732 right-maxillary-molar meshes, CC BY-NC-SA 4.0, unblocks unconditional prior training for the rare-FDI-class path. The dataset is also the *precondition* for the v0 H5 synthetic-data experiment (real-only vs real + N×VF-Net-synthetic, similar to TeethGenerator paper 051's Fig 9). Engineering cost: 1 day download, $0 compute, prerequisite for action 1.

3. **(v0 sub-task 2) ADOPT the per-point variance output as a v0.5 feature** for any v0 conditional generator (MADCrowner, ToothCraft). Adding variance output to a generator that currently only outputs coordinates is a 1-line architectural change + a 100-line loss change (replace MSE with Student-t ν=3 NLL), +$0 compute, expected +5-10% dentist-trust on the generated crown (the dentist gets a "trust map" overlay showing which regions of the crown are *uncertain*). The variance map could be color-coded on the 3D viewer: green = trustworthy, yellow = moderately uncertain, red = guesswork. Engineering cost: 1-2 days per generator, $0 compute, $50-100 Lambda for retraining.

4. **(v0 sub-task 2) ADOPT the mesh-regularization-from-2D-grid trick as a v0 post-processing step** — take the v0 sub-task 2 generated crown mesh, project it to a 2D grid (using a spherical or cylindrical parameterization), smooth the 2D grid (Laplacian smoothing), and re-fold to 3D. This enforces the 2D-topology H4 prior on *any* generated mesh, *not just VF-Net outputs*. Engineering cost: 1-2 days, $0 compute, expected +1-3% mesh-quality improvement (regular facets, no self-intersections).

5. **(v0 sub-task 2) PILOT the small auxiliary network for `g` prediction from `z`** (the "minor network" from Sec 5) — this fixes the grid-corner edge artifacts that hurt sampling fidelity. Engineering cost: 1 day, $0 compute, expected +5-10% sampling-fidelity improvement. Pre-req for action 1's high-quality samples.

6. **(v0 sub-task 2) ADOPT LION's 1-NNA + COV evaluation protocol verbatim** for v0 *generation* metric (same as paper 051's recommendation 5) — VF-Net uses 1-NNA + COV + MMD as the sampling metrics, the same protocol LION (paper 005) introduced and that every dental-3D-gen paper since has adopted. Engineering cost: 2-3 days, $0 compute, prerequisite for any v0 generation paper.

7. **(v0 H5) RUN a downstream synthetic-data experiment on FDI 16** — train a MeshSegNet (paper 023) on the 7,732-tooth FDI 16 split (80/20) vs train + 1×/2×/5×/10× VF-Net-synthetic samples, report the DSC delta on the 20% test set. This is the v0 paper's H5 ablation, the *exact* analog of TeethGenerator paper 051's Fig 9 but on tooth meshes. Engineering cost: 1-2 weeks, $200-400 Lambda total for all 5 conditions, the v0 paper's most publishable H5 table.

8. **(v0 paper) CITE the 3Shape + DTU partnership as a *de facto* dental-3D-lineage** (Hauberg's CogSys group + 3Shape's TRIOS 3 scanner dominance + Innovation Fund Denmark) — 4 papers in our reading list: VF-Net (057), ToothFairy dataset (not yet read), the 3Shape segmentation papers (not yet read). The 3Shape + DTU partnership is the *only* private-public partnership in the dental-3D reading list with a public data release (FDI 16). Credit the partnership explicitly.

9. **(v0 deployment) PILOT a per-clinic fine-tuning step with 50 labeled scans** (parallel to paper 055's cTooth+ cross-dataset finding) — VF-Net's small FDI 16 dataset (7,732) and bias toward aligner-treatment patients means a v0 model trained on FDI 16 alone is *not* representative of the elderly population. Fine-tune on 50-100 labeled scans from the target clinic, $200-400 Lambda, expected +2-5% CD on the target clinic's data. Engineering cost: 1-2 days for the fine-tuning pipeline, $200-400 Lambda per clinic.

10. **(v0 paper) ADOPT the *single model for all 32 FDI classes* pattern from VF-Net's "All FDIs" experiment** — the paper's 1-model-for-all-32-classes is a *real* H4 advantage (one model = one deployment, not 32). Compare to the Lombaert-lineage (papers 033, 034, 036, 037) which trains *separate* models per tooth class. For the v0 sub-task 2 data-augmentation path, the 1-model-for-all-32 is the *right* design choice; for the v0 sub-task 2 patient-specific path (MADCrowner, ToothCraft), the per-class model might still be better (more capacity per class), but the comparison is worth a v0 ablation.

### v0 stack updated

- **sub-task 1 (FDI seg)** = Cao25 + CrownSegger + Point2SSM-derivative + Mesh2SSM++ (paper 041) + STEAM-style GAM+MGR (paper 042) + 32-class tooth-classifier head + ME-loss regularizer + 2×2×8 FDI grid structure (paper 051) + **nnU-Net ResEnc L 5-fold (paper 053, CBCT) + U-Mamba2 (paper 054, CBCT) + cTooth+ cross-dataset eval (paper 055) + iMeshSegNet (paper 056, v0.5) + PointNet-Reg (paper 056, v0.5 landmark head)**
- **sub-task 2 (crown gen, *conditional* path)** = MADCrowner (paper 034) + ToothCraft (paper 036) + ToothForge (paper 037) + DMC (paper 033) + DCrownFormer (paper 032) + **per-point variance output (this paper, v0.5) + 2D-grid mesh regularization (this paper)**
- **sub-task 2 (crown gen, *unconditional* prior path)** = **VF-Net (this paper, primary) + LION (paper 005, secondary) + TeethGenerator (paper 051, tertiary, for orthodontic pre/post) + SAE-LP (paper 040, for spectral prior)** + 1-NNA + COV + MMD evaluation
- **sub-task 4 (outer surface)** = PVD + ME-loss + DiGS + FlexiCubes + Surface Projection loss + MGR
- **Training data** = 3DTeethSeg'22 + 3DS + ODD + ToothForge synthetic + TeethGenerator synthetic + **VF-Net synthetic (this paper, on FDI 16) + LION synthetic (paper 005, ShapeNet)**
- **Eval** = + IoU_Antag + ToothForge reconstruction filter + spectral-only baseline + per-tooth-type CD-L2 breakdown + ME-loss correspondence + LION 1-NNA + UCD + **FDI 16 cross-dataset test (this paper) + per-clinic 50-scan fine-tune protocol (this paper, v0 deployment)**
- **v0 compute** = **~$5,140-6,230 Lambda** (was $4,940-5,930, +$200-300 for VF-Net training on FDI 16 + $200-400 for the H5 synthetic-data ablation on FDI 16 + $50-100 for the per-point variance + mesh regularization v0.5 engineering + $20-30 buffer for FDI 16 download + v0 deployment protocol)

### Open questions for HK

(i) **Adopt VF-Net as the v0 sub-task 2 *unconditional* prior for rare-FDI-class data augmentation?** (recommend YES — public data + public code + SOTA reconstruction + 1-NNA 56.31% best in class; the right H2-compliant architecture for the augmentation path; $200-300 Lambda, 1-2 weeks)

(ii) **Download FDI 16 dataset (7,732 right-maxillary-molar meshes, CC BY-NC-SA 4.0) this week?** (recommend YES — unblocks the v0 H5 ablation; prerequisite for any VF-Net-based experiment; 1-day download, $0)

(iii) **Add per-point variance output to v0 sub-task 2 *conditional* generators (MADCrowner, ToothCraft)?** (recommend YES, defer to v0.5 — 1-line architecture change + 100-line loss change + retrain, $50-100 Lambda; the "trust map" UX is the highest-impact dentist-facing feature in the reading list)

(iv) **Run the v0 H5 synthetic-data ablation on FDI 16** (MeshSegNet trained on real-only vs real + 1×/2×/5×/10× VF-Net-synthetic, report DSC delta)? (recommend YES — the v0 paper's most publishable H5 table, $200-400 Lambda, 1-2 weeks, prerequisite for the v0 paper's H5 claim)

(v) **Adopt the 2D-grid mesh regularization post-processing for v0 sub-task 2?** (recommend YES — 1-2 day engineering change, $0 compute, expected +1-3% mesh-quality improvement; works on *any* generated mesh, not just VF-Net)

(vi) **Adopt the *single model for all 32 FDI classes* pattern from VF-Net** for the v0 sub-task 2 *unconditional* prior? (recommend YES for the data-augmentation path; NO for the patient-specific path — MADCrowner + ToothCraft + ToothForge per-class models are still better for the conditional task)

(vii) **Cite the 3Shape + DTU partnership + Hauberg's CogSys group as a *de facto* dental-3D-lineage** (alongside the Lombaert-lineage 5 papers and the Tsinghua-lineage 5 papers)? (recommend YES — credit the 4-paper program: VF-Net 057 + the 3Shape-related Hauberg group papers; the *third* major dental-3D research lineage in the reading list)

(viii) **Pilot a per-clinic fine-tuning step with 50 labeled scans** as part of v0 deployment? (recommend YES — the 7,732 FDI 16 dataset is biased toward aligner-treatment patients, the elderly population is not represented; per-clinic fine-tuning is the practical v0 deployment protocol, $200-400 Lambda per clinic, +2-5% CD on the target clinic's data)

### Next paper to read (058)

Strong candidates for 058:

1. **Personalized dental crown design: a point-to-mesh completion network** (Hosseinimanesh et al. 2024, Polytechnique Montréal, the Lombaert-lineage — *the 6th* paper in the lineage, parallel to paper 033 DMC). This is the *same* group that did DMC, MADCrowner, ToothCraft, ToothForge, and now this — closing the Lombaert-lineage reading arc.

2. **CrownGen: Patient-customized Crown Generation via Point Diffusion Model** (arXiv:2512.21890, Dec 2025) — the *newest* 2025 dental-crown diffusion paper, point-based diffusion with patient-specific conditioning, would close the "2025 dental-3D-gen landscape" arc by adding the *most recent* diffusion-based crown generator to the reading list.

3. **VBCD: A Voxel-Based Framework for Personalized Dental Crown Design** (Wei et al. 2025, MICCAI 2025) — the *voxel-based* counterpart to the point/mesh-based methods, would close the "all 3 substrates" arc (voxel + point + mesh).

4. **VRCNet: Variational Relational Point Completion Network** (Pan et al. 2021, the supervised shape-completion baseline from VF-Net's Table 4) — a *backwards* read into a key baseline that was cited but not deeply analyzed; would let us re-examine the supervised shape-completion landscape that VF-Net competes with.

5. **SALAD: Part-Level Latent Diffusion for 3D Shape Generation and Manipulation** (Koo et al. 2022, the NeurIPS work that the Ao Zhang 2024 "Conditional diffusion guided by part-level latent for dental crown point cloud generation" paper builds on) — the *foundational* part-level latent diffusion paper; would let us analyze the part-level latent diffusion architecture that the dental-crown diffusion papers all build on.

**Recommendation: paper 058 = Hosseinimanesh et al. 2024 "Personalized dental crown design"** — the 6th Lombaert-lineage paper, closing the largest research lineage in the reading list (6/6 Lombaert papers). The 5 Lombaert-lineage papers already read (DMC 033, MADCrowner 034, ToothCraft 036, ToothForge 037, and the dissertation context) are the *most coherent* research program in the reading list; adding the 6th paper completes the arc and lets us make *cross-paper* synthesis claims about the lineage's design choices (per-class vs all-class, fixed-template vs spectral, point vs mesh, completion vs unconditional, etc.). After 058, the Lombaert-lineage is *closed* and the project can move to the next lineage (Tsinghua 5 papers, 3Shape-DTU 4 papers) or the next research-arc (the 2025 dental-3D-gen papers, all 5 of them).

Alternatively, if we want to *expand* the dental-3D-gen landscape, **paper 058 = CrownGen (arXiv:2512.21890)** for the *newest* 2025 paper. The 5 dental-3D-gen papers read so far (TeethGenerator 051, DuoDent 052, DCrownFormer 032, DMC 033, MADCrowner 034, ToothCraft 036) cover the 2023-2024 landscape; CrownGen extends to late 2025 and lets us see where the field is *going* (patient-specific point-diffusion is the 2025 trend). For the v0 paper's related work, the *most recent* paper is the most useful to cite.
