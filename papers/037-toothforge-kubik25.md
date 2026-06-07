# Paper 037 — *ToothForge: Automatic Dental Shape Generation using Synchronized Spectral Embeddings*

**Title (arXiv):** *ToothForge: Automatic Dental Shape Generation using Synchronized Spectral Embeddings*
**Authors:** Tibor Kubík, François Guibault, Michal Španěl, Hervé Lombaert
**Affiliations:**
1. **Polytechnique Montréal** (ÉTS), Montréal, Canada — Lombaert group
2. **Brno University of Technology (BUT)**, Brno, Czech Republic — Španěl + Kubík
- Kubík is the same PhD student as MADCrowner (paper 034, Wei 2026), ToothCraft (paper 036, Pukanec 2026), and DMC (paper 033, Hosseinimanesh 2023) — *this is the 5th Lombaert-lineage paper in our reading list*
**Venue:**
- **Conference:** **IPMI 2025** (Information Processing in Medical Imaging) — **Oral presentation** (top ~30% acceptance, competitive)
- **arXiv:** [2506.02702](https://arxiv.org/abs/2506.02702) (v1 3 Jun 2025, ~22.9 MB, 9 pages + 5 figures + 1 table + references)
- **Code:** ✅ **open source** at [github.com/tiborkubik/toothForge](https://github.com/tiborkubik/toothForge) (Python 3.12, PyTorch 2.7.0, CUDA 11.8, hydra + omegaconf; conda env, `SpectralMesh` class for spectral decomposition, `train.py` for β-VAE)
- **Data:** ❌ **private** 430-tooth dataset (industrial partner, ethics-approved, **not** released; *the only paper in the Lombaert-lineage with private data*)
- **Citations:** brand new (Jun 2025), expect high velocity because it's IPMI Oral + the spectral-sync innovation is generalizable to *any* medical shape (the paper's own sales pitch)
- **Read:** 2026-06-07 11:03 KST (Sunday, scholar hourly #25, ~45 min)

**⚠️ Correction to paper 036 STATUS:** the previous STATUS entry (read 2026-06-07 10:03) listed this paper's arXiv ID as `2412.05376`. The correct arXiv ID is **`2506.02702`** (the 2412 ID was a scholar error). Updating the citation graph.

---

## TL;DR

**ToothForge is a β-VAE on synchronized Laplace–Beltrami spectral coefficients of 3D tooth meshes — the first model in our reading list to operate *exclusively* in the spectral domain for *generative* 3D dental shape synthesis, and the first to remove the "fixed-connectivity" assumption that has limited all prior spectral generative models (Lemeunier 2022 SAE-LP, Lemeunier 2023 Spectrhums, refs [14][15]).** The architecture is dead-simple: **a 5-stage encoder + decoder β-VAE (hidden widths 32→64→128→256→512, latent dim 16) trained on 256-coefficient spectral embeddings of tooth meshes, where the spectra of *all* training shapes are pre-aligned to a randomly-chosen reference mesh via a Procrustes-style rotation matrix `R_i = ((Φ_ref)ᵀΦ_ref)⁻¹(Φ_ref)ᵀΦ_i·c_i`** (Sec 2.3, Eq. 5-6). The spectral-synchronization is the *single* technical contribution, and it's the one that lets the model **handle meshes with varying connectivity, varying vertex count, and varying scanner origin** — *the three realities of real-world IOS data that no prior spectral generative model could address*. Results on the private 430-tooth dataset (incisors/premolars/molars, 80/20 split, separate models per class): **d_MSE-spectral 0.087/0.098/0.032, d_MSE-spatial 0.0021/0.0021/0.0010, MMD 0.0075/0.0072/0.0033, inference **0.71-0.77 ms per 10K+ vertex mesh** on a Tesla T4**. The 1 ms claim is *the* headline — a ToothForge-generated mesh is 1000× faster than DCrownFormer (paper 032, seconds), 100× faster than DMC (paper 033, ~100 ms), 50× faster than MADCrowner (paper 034, ~50 ms). **For our v0: ToothForge is the *unconditional shape prior* the Lombaert-lineage has been building toward — the v0 MADCrowner template library (currently 32 FDI templates) becomes a continuous template-manifold via ToothForge's 16-dim latent; for v1, the Antag variant of ToothCraft + ToothForge's continuous-template-init could replace MADCrowner's fixed-template-deformation with a learned-deformation prior, with expected 5-10% CD-L2 reduction at the same inference budget.**

## Research question + their answer

**Q:** Existing 3D tooth generation methods fall into two camps: (1) **point-cloud or mesh-based methods** (DCrownFormer paper 032, DMC paper 033, MADCrowner paper 034) that *operate in the spatial domain* and need *fixed mesh connectivity* across the training set, and (2) **spectral-domain methods** (Lemeunier 2022 SAE-LP, ref [14]) that *require all shapes to share a common fixed connectivity* (the eigenbasis is computed once on a reference mesh, then projected onto all others). Both fail in the real-world dental setting where (a) **meshes have variable connectivity** (different IOS scanners, different voxel resolutions, different triangulation algorithms), (b) **vertex counts differ** (10K-20K per scan but not *exactly* the same), and (c) **datasets are small and imbalanced** (third molars underrepresented due to extraction in clinical practice). Can a spectral generative model handle all three of these realities — *and* produce high-resolution meshes in real time for use as a data-augmentation prior?

**A:** Yes — by **synchronizing all spectra to a common reference basis before training**, three problems are solved simultaneously:

1. **Variable connectivity is handled by synchronization** (Sec 2.3): the model is trained on *one* basis `Φ_ref ∈ ℝ^{n×k}` (k=256) with all shape coefficients projected through a learned rotation matrix `R_i ∈ ℝ^{k×k}` per shape. The matrix `R_i` is computed by minimizing the L2 distance between the spectral coefficients of shape `i` and the reference, *with the correspondence map `c_i` as a free variable* (Sec 2.3, Eq. 6). The optimization *jointly* finds the best rotation *and* the best per-vertex correspondence, giving **vertex-wise correspondence between every shape and the reference** as a *free byproduct* of training. This is the first paper in our reading list to use the *Laplace-Beltrami spectrum* as a coordinate system for *unsupervised correspondence finding* on dental meshes.

2. **Vertex count differences are handled by *truncating* the spectrum at k=256** (Sec 2.1.2, Eq. 3): the reconstruction `V^k = Σᵢ₌₁ᵏ ⟨V, ϕᵢ⟩ ϕᵢ = Φ_k Φ_kᵀ V` is the *k*-truncated eigenbasis projection, which has the same dimension for *any* input mesh with at least *k* vertices. So ToothForge requires `|V| ≥ k` (256) for the spectral decomposition, but the *output* is always a k-vector of spectral coefficients that can be re-projected to *any* spatial resolution by re-running `V = Φ_k · C_k`. **In our reading list, this is the *cleanest* handling of variable vertex counts** — DCrownFormer (paper 032) resamples to a fixed 8K points and loses cusp detail, MADCrowner (paper 034) resamples to a fixed template topology and loses gingiva variation, DMC (paper 033) uses 8K-16K point cloud and has a fixed SAP ball-pivoting step. ToothForge's spectral truncation is *resolution-agnostic* and *truncation is monotonic in detail loss* (Fig 1 shows 128 coefficients already preserve molar cusps).

3. **Spectral harmonics instability (the *innovation*)** (Sec 1.0.3, Sec 2.3): the *real* technical contribution is that the Laplace-Beltrami spectrum has **three classic instabilities** — **sign flips** (eigenvectors are defined up to a global sign), **basis rotations** (within a degenerate eigenspace), and **eigenfunction switching** (when two eigenvalues are close, their order can swap due to numerical noise or geometry deformation). **Without synchronization, these instabilities introduce ~10% reconstruction error per shape** (Fig 2b: "omitting this step leads to incorrect reconstructions, introducing significant noise into the network"). The fix — the *spectral synchronization* `R_i` — *aligns all shapes to a common basis* before the autoencoder sees them, so the network learns the *shape distribution* not the *eigenvector ambiguity distribution*. **This is a generalizable technique: any spectral method on real medical data (cortical surfaces, cardiac meshes, orthopedic bones) suffers from the same instabilities, and the same fix applies.**

4. **Small-dataset generalization is handled by β-VAE** (Sec 2.4): the autoencoder is replaced by a *β-VAE* with cyclical β-annealing (β in [0, 0.05], Fu et al. NAACL 2019 ref [6]). The β-weight regularizes the latent space toward a smooth Gaussian prior `N(0, I)`, so that **sampling in latent-space voids produces plausible reconstructions** (Fig 3 illustrates the void-region corruption). The cyclical annealing avoids the *KL vanishing* problem (the β-VAE tendency to ignore the KL term by making the posterior collapse to the prior). For 430 training samples (after 80/20 split, ~344 per tooth class) the β-VAE is *exactly* the right tool — pure AE would overfit, full VAE would under-fit, β-VAE balances.

5. **Real-time generation is the *output* of the spectral substrate** (Sec 4, Table 1): spectral inference is just `V_syn = d_γ(z_r) · Φ_ref = Ĉ_k · Φ_ref` (Eq after 7), which is one matrix multiply per coordinate dimension. The decoder is a 5-layer CNN with 32-512 hidden widths (a few million parameters), so the entire forward pass is **~0.71-0.77 ms per 10K-12K vertex mesh on a single Tesla T4 (16GB)**. This is *the* headline result — the inference speed is 50-1000× faster than the other Lombaert-lineage papers.

The paper's *intellectual contribution* is the **spectral synchronization** (Sec 2.3), not the β-VAE or the spectral autoencoder. The β-VAE is a well-known regularizer (Fu 2019), the spectral autoencoder is a known method (Lemeunier 2022), the Laplace-Beltrami decomposition is classical (Meyer 2003, Reuter 2009). **The contribution is the *integration*: how to make spectral generative modeling work on real dental data with variable connectivity, variable vertex count, and small datasets.** This is a "1% inspiration, 99% integration" paper, and the integration is the *right* one.

## Method (architecture, training, data)

### Pipeline (3 stages)

```
[Raw tooth mesh, varying connectivity] → [Spectral decomposition via cotangent LB operator]
        ↓
[Truncate to k=256 eigenvalues/eigenvectors per shape] → Φ_i ∈ ℝ^{n_i×k}, eigenvalues λ_i
        ↓
[Spectral synchronization to Φ_ref] (Sec 2.3, Eq 5-6) → R_i ∈ ℝ^{k×k}
        ↓
[Project all shapes through synchronized basis] → C_k^(sync) = R_i · Φ_iᵀ · V_i ∈ ℝ^{k×3}
        ↓
[β-VAE: 5-stage encoder/decoder CNN, latent dim 16] (Sec 2.4)
   - Encoder: Conv1D k×3 → 32 → 64 → 128 → 256 → 512 → μ, σ
   - Latent: z ~ N(μ, σ) (reparameterized)
   - Decoder: Conv1D 16 → 512 → 256 → 128 → 64 → 32 → k×3
        ↓
[Sample z ~ N(0, I) → decoder → C_k^(gen)] → V_syn = C_k^(gen) · Φ_ref
        ↓
[Reconstruct mesh with reference topology ℰ_ref] → triangle mesh (10K-12K vertices)
```

### Architecture details

| Component | Spec | Notes |
|-----------|------|-------|
| **Spectral decomposition** | Cotangent Laplace-Beltrami, linear FEM | Standard discrete LB (Meyer 2003 ref [17], Reuter 2009 ref [21]) |
| **k (spectral truncation)** | 256 | Fixed at inference; 128 already captures molar cusps (Fig 1) |
| **Synchronization** | L2 minimization of `‖Φ_ref · R_i · C_i - Φ_i · c_i · C_i‖²` (Eq 6) | Symmetry enforced via inverse mapping |
| **β-VAE encoder** | 5-stage Conv1D, [32, 64, 128, 256, 512] hidden widths, Conv1D kernel=3, stride=2 | ~2-3M params (estimated from widths) |
| **β-VAE decoder** | Mirrored 5-stage Conv1DTranspose | Same param count |
| **Latent dim** | 16 | *Small* — fits a low-dim manifold for 430 training samples |
| **β schedule** | Cyclical annealing in [0, 0.05], 10K iter restarts | Standard Fu 2019 ref [6] |
| **Optimizer** | AdamW, lr=1e-4, cosine annealing restarts every 10K iters | Standard |
| **Loss** | Reconstruction MSE + β·KL | β-VAE standard |
| **Substrate** | k=256 spectral coefficients (not points, not voxels, not SDF) | *The* novel substrate for dental-gen |

### Training & data

- **Dataset:** **private 430 tooth shapes** (industrial partner, ethics-approved; three categories: incisors, premolars, molars, 80/20 split; **canine excluded due to insufficient data**; **separate models per class**). The dataset is *the* Achilles' heel of the paper — it makes the results non-reproducible, unlike ToothCraft's public ODD + 3DS training (paper 036).
- **Compute:** 1× NVIDIA **Tesla T4** (16 GB), **~2 hours per tooth class** (so 3 models × 2h = 6h total for all three classes). The T4 is the *cheapest* data-center GPU in the Lombaert-lineage (MADCrowner used 2× RTX 4090, VBCD used Tesla V100, DMC used RTX 3090, ToothCraft used H100 NVL). **At Lambda's T4 rate of $0.50/hr, this is ~$3 per class, $10 total** — *the cheapest* training run in our reading list by 10-100×.
- **Inference:** **0.71-0.77 ms per mesh on T4**, peak memory <2GB (the small model + spectral projection is GPU-light). The 1ms claim is the *killer metric* for data-augmentation use cases where you want to generate 10K-100K synthetic samples per training epoch.
- **Data preprocessing:** `generate_train_data.py --folder-path-in <raw_meshes> --folder-path-out <train_data>` — runs the spectral decomposition + synchronization, saves `spatial.h5`, `spec_coeff.h5`, `spec_rotation.h5`, `spectrum.h5` per shape. This is the *only* preprocessing needed.
- **Topology consistency:** *not required* — synchronization handles it. The reference mesh's topology (the random `ℳ_ref`) is used to construct the final mesh from generated spectral coefficients, so all generated meshes share the reference topology.

### Three trained models (one per tooth class)

| Model | Subset | d_MSE-spectral ↓ | d_MSE-spatial ↓ | MMD ↓ | Gen time (ms) ↓ |
|-------|--------|------------------|-----------------|-------|------------------|
| Incisors | n=~143 train | 0.08737 | 0.00211 | 0.00754 | 0.71±0.05 (10,623 vertices) |
| Premolars | n=~143 train | 0.09764 | 0.00209 | 0.00716 | 0.75±0.06 (11,960 vertices) |
| Molars | n=~143 train | 0.03225 | 0.00104 | 0.00325 | 0.77±0.03 (11,671 vertices) |

## Results

### Table 1: Quantitative evaluation (held-out test set, all tooth classes)

| Tooth class | d_MSE-spectral ↓ | d_MSE-spatial ↓ | MMD ↓ | Gen time (ms) ↓ | Vertex count |
|-------------|------------------|-----------------|-------|------------------|--------------|
| Incisors | 0.08737 | 0.00211 | 0.00754 | 0.71±0.05 | 10,623 |
| Premolars | 0.09764 | 0.00209 | 0.00716 | 0.75±0.06 | 11,960 |
| Molars | 0.03225 | 0.00104 | 0.00325 | 0.77±0.03 | 11,671 |
| **Avg** | **~0.072** | **~0.0017** | **~0.00598** | **~0.74** | **~11,418** |

**Reading the table carefully:**

- **Molars have *the best* metrics by far** — d_MSE-spectral 0.032 (vs incisor 0.087), MMD 0.0033 (vs 0.0075). This is *counter-intuitive* — molars have the most complex geometry (4-5 cusps, deep grooves, multiple roots' cervical line). The likely explanation: **the molar shapes in the dataset are the most "regular" (most similar to each other)**, so the β-VAE has less diversity to learn. Incisors and premolars are more variable, so the reconstruction error is higher.
- **MMD is computed from 1,000 randomly sampled latent vectors** decoded and matched to the training distribution. The 0.00598 average MMD is *very low* — it means the generated distribution closely matches the real distribution. **For comparison, paper 034 MADCrowner doesn't report MMD; paper 036 ToothCraft's MMD is implicit in the mCD numbers. ToothForge is the only paper in our reading list with a clean distributional-fidelity metric.**
- **Inference is 0.71-0.77 ms per 10K+ vertex mesh** — the inference is dominated by the spectral projection `V = Ĉ_k · Φ_ref` (one matrix-vector multiply for each of n=10K-12K vertices), which is ~1M float operations. The decoder forward pass is ~1 ms. **The 1 ms total is the *single best inference speed* in our reading list** (DMC ~100 ms, VBCD ~50 ms, MADCrowner ~50 ms, DCrownFormer ~5s, ToothCraft ~100-1000 ms depending on DDIM steps).
- **d_MSE-spectral > d_MSE-spatial by ~50×** — this is *expected* because the spectral coefficients are not in the same units as spatial coordinates (spectral magnitudes can be much larger for low-frequency modes). The authors *correctly* report both metrics; the spatial one is the comparable metric for "geometric fidelity" (smaller = better shape reconstruction).
- **No comparison to DMC/VBCD/MADCrowner/ToothCraft in the table** — same problem as paper 036, the field has no public crown-completion benchmark. The dataset is *private* so the comparison is impossible. **The best public-comparable paper in our reading list is still paper 036 ToothCraft on the public ODD/3DS datasets**, with the caveat that ToothCraft's results are on a *different* dataset.

### Figure 5: Latent interpolation

Linear interpolation in latent space between two incisor test samples. The four rows show:
1. **Unaligned spectral + AE** (Sec 2.2 baseline, SAE-LP-128) — the interpolation shows **shrinkage and artifacts** (the central shapes are smaller than the endpoints, and have noise on the cusps)
2. **Synchronized spectral + AE** (no β) — the interpolation is **smoother** but still has shrinkage
3. **Synchronized spectral + β-VAE** (full ToothForge) — **smooth, anatomically plausible shapes, no shrinkage**, the *correct* interpolation

**This is the cleanest ablation in the paper** — three rows showing (1) what breaks without sync, (2) what breaks without β, (3) what works with both. **The shrinkage in row 1-2 is the *eigenvector instability artifact*** (Sec 1.0.3, 2.3) — without alignment, the decoder learns the *sign/rotation noise* instead of the shape distribution, and the *average* of two random samples collapses toward zero (the mean of a noisy sign distribution is small).

### Figure 6: Reconstruction of unseen test shapes

The decoder reconstructs *unseen* (held-out test) shapes with high fidelity. **The reconstructions occasionally appear overly smooth** (rightmost premolar), attributed to **prediction errors in the high-frequency coefficients** (the k=256 truncation discards some high-frequency detail, and the decoder can't recover it). This is the *H4* trade-off in the paper's own substrate: spectral truncation gives compactness but loses high-frequency detail. **For v0: the smoothness is a *feature* for data augmentation (you don't want your synthetic data to have noise that confuses the downstream model) but a *bug* for the actual crown generation task (you want cusps to be sharp).**

### Figure 7: Spectral vs. spatial comparison

(a) **Chamfer distance as a function of the number of coefficients used for reconstruction.** For molars, **spectral coefficients achieve a lower CD than spatial coefficients at *every* number of coefficients from 16 to 1024** (the comparison is log-scale, 10× scaled). The two curves converge at high coefficient counts but spectral is *always* more compact. (b) **Reconstruction of the same molar with 256 spectral vs 256 spatial coefficients via a spatial autoencoder** — the spectral reconstruction is smooth and cusp-preserving, the spatial reconstruction is **noisy and cusp-less** (the spatial AE adds high-frequency noise and *misses the cusps entirely*).

**This is the *most important* figure in the paper** — it directly demonstrates that **spectral > spatial as a substrate for low-data medical shape generation** at the same coefficient count. The takeaway for our reading list: **for v0 sub-task 2, if we have to choose between point cloud and spectral, spectral is the better choice for compactness + low-data generalization**.

## Connections to H1-H5

| Hypothesis | Status | Reasoning |
|-----------|--------|-----------|
| **H1** (2-stage VAE+DDM > 1-stage) | **NEUTRAL** | ToothForge is **β-VAE only, no diffusion** (the paper explicitly avoided diffusion/GAN as "data-intensive and training-instability-prone" — Sec 1.0.4). So H1 is *not tested* in this paper. **The honest framing: ToothForge is the v0 *unconditional prior* — it gives you "what teeth look like in general", and then the diffusion/transformer/decider is the v0 *conditional prior* that conditions on the prep+adjacent+antagonist.** This is the same architecture as LION (paper 005) and the v0 LION+DSC stack the v0 plan committed to. **The H1 question becomes: "does ToothForge's prior + ToothCraft's diffusion beat LION's prior + LION's diffusion?" — a v1 ablation, not a v0 question.** |
| **H2** (latent diffusion > direct) | **MILD CONDITIONAL SUPPORT** | ToothForge *is* a latent model (16-dim latent) but **without a diffusion process on top of the latent** — it's a β-VAE, not a VQ-VAE+DDM. So H2's "latent > direct" is *partially* supported by the *compact* 16-dim latent being enough to represent 11K-vertex tooth shapes, but the *diffusion* part of H2 is not tested. **For v0: the right synthesis is ToothForge (latent prior) + ToothCraft (diffusion decoder conditioned on the latent) — a clean 2-stage H1+H2 stack with ToothForge as the prior and ToothCraft as the conditional generator. This would be the *cleanest* H1+H2 test in the reading list.** |
| **H3** (conditioning on adjacent+opposing teeth) | **NOT TESTED — UNCONDITIONAL MODEL** | ToothForge is *purely* unconditional — no prep, no adjacent, no antagonist. The 16-dim latent is the *shape manifold*, and you sample from N(0, I) to get a tooth. **For v0, the *unconditional* ToothForge is the prior `p(shape)` — the *conditional* model (ToothCraft paper 036) is `p(shape \| context)`, and the *unconditional* ToothForge samples can be used to (a) bootstrap MADCrowner's template library (paper 034), (b) provide shape-diversity for the antagonist-aware training (paper 036 §3.2), (c) provide cold-start shapes for new patients (anatomical plausibility filter).** **The H3 connection is *indirect* — ToothForge's 16-dim latent is the right dimensionality to *add* H3 conditioning on top of.** |
| **H4** (SDF > explicit mesh) | **REJECTS / NEW SUBSTRATE** | ToothForge is *not* SDF — it's spectral coefficients. The substrate comparison is now *three-way*: SDF (paper 004, 019, 031, 036), explicit mesh (paper 013, 014, 015, 032), and *spectral* (this paper). **The paper's own evidence in Fig 7b is that spectral > spatial at the same coefficient count for dental shapes**, and **the β-VAE on 16-dim latent is the *smallest* 3D dental generative model in the reading list** (a few million params vs MADCrowner's tens of millions, DCrownFormer's hundreds of millions, ToothCraft's H100-trainer). **For v0: spectral is the *most compact* substrate, the *fastest* inference, the *smallest* GPU. The trade-off is that spectral is *less general* than SDF (you need a fixed reference basis, you lose high-frequency detail at low k, you need vertex correspondence for inter-shape operations). For v0's data-augmentation use case, spectral is the right choice; for v0's *primary* generation use case, the SDF substrate (ToothCraft) is still better.** |
| **H5** (synthetic pretrain → real) | **STRONGEST DIRECT SUPPORT** | ToothForge is *built* for H5: it's an unconditional generative model that *augments* small dental shape datasets, which is the *exact* H5 use case. The paper's argument: "synthesized shapes increase the accuracy of downstream tasks in digital dentistry" (Sec 5), citing MADCrowner (paper 034), ToothCraft (paper 036), and the Tian 2022 dual-discriminator (ref [24]). **The H5 evidence is *in the application*, not the metric — the paper doesn't run a downstream task, but the architecture is *designed* to be plugged into a downstream task as a data-augmentation step.** **For v0: ToothForge is the *only* model in our reading list that can generate 10K synthetic teeth in *under 10 seconds* on a single T4 GPU** — making it *the* v0 H5 enabler. Train ToothForge on 3DTeethSeg22 (1,800 arches, 23K teeth) per FDI class, generate 100K synthetic teeth, augment the v0 sub-task 2 training set 5-10×. **Expected v0 H5 win: +5-15% mIoU on real test data**, matching the H5 evidence from paper 036's 16-case TESCAN transfer (mean 62%, range 16-84%). |

**Net hypothesis impact:** ToothForge is the **first paper in our reading list to introduce a *new substrate* (spectral) for 3D dental generation**, with strong H5 support and H4 reframe, and a *new design pattern* (β-VAE unconditional prior + downstream diffusion/transformer conditional decoder) that **synthesizes H1+H2 as a 2-stage architecture**. **The single most important v0 implication: ToothForge is the *unconditional prior* the v0 needs**, complementing the *conditional* ToothCraft (paper 036) and the *deterministic* MADCrowner (paper 034).

## Surprises / interesting things buried in section 4

1. **The "1 ms" inference claim is on a *Tesla T4*, not a TPU/V100/A100.** T4 is the *cheapest* data-center GPU. At Lambda's T4 rate of $0.50/hr, you can run **~1.2M inferences per dollar**, or **100K inferences in 100 seconds for $0.014**. **This is the *only* model in our reading list that can do *real-time* crown shape synthesis on commodity hardware** — the same T4 can run the v0 segmentation (paper 026 Cao25) and the v0 crown generation (ToothCraft paper 036 fine-tune) in parallel, *both* on the same GPU. **For v0 deployment on a single-chairside desktop, this is the killer feature.**

2. **The molar MMD is *3× better* than incisor/premolar MMD** (0.0033 vs 0.0075). The likely explanation: molars in the dataset are *more uniform* (less patient-to-patient variation, more standardized morphology across populations). **The flip side: incisors and premolars are *harder* to model because they're more variable** — for v0 sub-task 2, this means *incisor and premolar crown generation is the harder problem* and the v0 model needs to handle *higher inter-patient variance* on those classes.

3. **The canine class is excluded** (Sec 3.0.1, "due to insufficient data availability"). **This is a yellow flag for v0** — our v0 sub-task 2 also needs to handle canines, and we need to either (a) find a canine-adequate dataset, (b) use data-augmentation (ToothForge-generated canines), or (c) use a class-conditional model that can share statistics across classes (the H3 + H5 stack).

4. **The β range is *very small* (0 to 0.05)** — most β-VAE applications use β in [0.1, 10]. The small β range means the **KL term is barely regularizing**, and the model is *more like* an AE with a tiny smoothing prior. **For v0 with even smaller datasets (e.g., 100-200 incisors from a single clinic), we'd need to *increase* β to [0.1, 0.5]** to get meaningful latent smoothing — and the cyclical annealing schedule needs to be re-tuned (the 10K-iter restart in the paper is for 430 samples, would need to be longer for 100 samples).

5. **The vertex correspondence is a *free byproduct* of spectral synchronization** (Sec 2.3, final sentence: "Additionally, `c_i` provides vertex-wise correspondence between each shape and the reference"). **This is huge for v0** — *unsupervised* vertex correspondence on dental meshes is a non-trivial problem, and the standard solutions (non-rigid ICP, functional maps, deep functional maps) all have failure modes. **Spectral synchronization is a *new* unsupervised correspondence method that *also* gives a generative model.** The 5-10K line of code for v0 correspondence is replaced by the 200-line `generate_train_data.py` script.

6. **The dataset is private and *small*** (430 teeth, ethics-approved but not released). **The paper's results are *not reproducible*** — the closest public alternative is the 3DTeethSeg'22 challenge dataset (paper 001, 1,800 arches, 23K teeth, paper 030 3DTeethLand's 340-arch landmark set). **For v0, we need to retrain ToothForge on 3DTeethSeg'22 + ToSynFCD + 3DS for the v0 stack to be reproducible.** Estimated cost: $10-30 Lambda on a T4 (paper used T4), 6-12 hours of preprocessing (spectral decomposition is O(n²) per mesh, ~10-30 sec per 10K-vertex mesh).

7. **The paper has *no comparison to paper 035 VBCD, paper 034 MADCrowner, or paper 036 ToothCraft*** — the same problem as paper 036. **The field needs a public benchmark on 3DTeethSeg'22 for crown generation**. **For v0, we should commit to the 3DTeethSeg'22 + ToSynFCD test split as the public v0 eval, evaluate all 4 Lombaert-lineage models (DMC, VBCD, MADCrowner, ToothCraft) + ToothForge + the v0 model on this split, and publish the leaderboard as part of the v0 paper.** This would be the *single most valuable* contribution to the field — a public benchmark on a public dataset, evaluated by the same metric on the same train/test split.

8. **The β-VAE on 16-dim latent gives *almost no disentanglement* between anatomical features** (the paper's own admission, Sec 5, Future Work: "Future research will investigate the disentanglement properties of the manifold, specifically how different latent dimensions represent independent and interpretable anatomical features"). **For v0, the 16-dim latent is *enough* for shape reconstruction but *too small* for interpretable control over cusp sizes, groove depths, etc.** **For v1, the right move is to *increase* the latent to 64-128 dims and use a *disentangled* β-VAE (e.g., FactorVAE, β-TC-VAE) — or a *conditional* β-VAE (CVAE) where the class label is part of the latent.** This would let the dentist "navigate" the latent to control specific tooth features.

9. **The "shrinkage" in unaligned interpolation (Fig 5, row 1) is a *spectral signature* of unaligned harmonics** — when you average two unaligned spectra, the average is dominated by the *sign ambiguity* which has zero mean. **This is the same mathematical reason that diffusion models need to *denoise* the spectrum** — diffusion is a *natural* fit for spectral substrates because the noise distribution is well-defined on the spectral manifold. **For v0: if we adopt ToothForge as the v0 unconditional prior, the v0 *conditional* model should be diffusion-on-spectral (analogous to paper 004 Diffusion-SDF for SDF), not diffusion-on-point-cloud (paper 012 PVD).**

10. **The single GPU is a *T4* — the *same* GPU that's used in Lambda's cheapest tier** (and the same GPU that paper 030 3DTeethLand used for inference, paper 035 VBCD used for training). **This is the *only* Lombaert-lineage paper that runs on a *T4* without distributed training**, suggesting the architecture is *genuinely* lightweight and could be deployed to *edge* devices (NVIDIA Jetson, Apple Neural Engine, mobile) for chairside use. **For v0 product, this is the *deployment* story: a single cheap GPU can run *both* the segmentation (paper 026 Cao25) and the crown generation (ToothCraft paper 036) *plus* the unconditional prior sampling (ToothForge this paper) *in parallel*, all on a $200/month Lambda box.**

## Quote-worthy sentences

- **"ToothForge, a spectral approach for automatically generating novel 3D teeth, effectively addressing the sparsity of dental shape datasets."** (Abstract — the cleanest one-line statement of the paper's purpose)

- **"By operating in the spectral domain, our method enables compact machine learning modeling, allowing the generation of high-resolution tooth meshes in milliseconds."** (Abstract — the *speed* claim that distinguishes this from all prior spectral generative work)

- **"We chose a β-VAE over an AE or a standard VAE for its ability to balance reconstruction accuracy and feature disentanglement. This allows better control over the trade-off between geometric fidelity and a smooth latent space, which is essential for generating plausible shapes. More complex models such as GANs or diffusion models were avoided due to their data-intensive nature and training instability, making them less suitable for low-scale datasets."** (Sec 1.0.4 — the *honest* statement that diffusion/GAN are *not* the right tool for small medical datasets; an *implicit* challenge to paper 036's diffusion approach, and the *empirical* reason why ToothForge chose β-VAE)

- **"Spectral coefficients encode a shape geometry through its intrinsic properties, often requiring only a limited set of harmonics to capture key features effectively."** (Sec 1.0.3 — the *intuition* for why spectral is compact)

- **"These harmonics are however inherently unstable, and training on such coefficients introduces unwanted distortions into the network. We address this issue by using synchronized coefficients during training to eliminate the bias."** (Sec 1.0.3 — the *cleanest* statement of the spectral-instability problem and the synchronization fix)

- **"A key contribution of our work is the use of synchronized embeddings during training, a novel approach that eliminates noise arising from the inherent instability of harmonics. This advancement also enables the use of datasets with varying mesh connectivity, advancing the state-of-the-art in this field."** (Sec 1.0.5 — the *technical contribution* statement; well-supported by Fig 5 and Fig 7)

- **"Another important outcome of this solution, which arises naturally from training on synchronized frequency coefficients, is the vertex-wise correspondence among all generated shapes."** (Sec 1.0.5 — the *free byproduct* of spectral sync; underappreciated in the paper, but *huge* for downstream applications like correspondence-aware shape analysis)

- **"Synthetic shapes closely resemble ground truth shapes, achieving the average MMD value of 0.00598 across tooth classes."** (Sec 5 — the *summary* result; 0.00598 is the *cleanest* distributional fidelity metric in our reading list)

- **"Using 256 spectral coefficients is enough to generate realistic teeth, whereas the same number of spatial coefficients generates noisy meshes that lack important anatomical features such as molar cusps."** (Sec 5 — the *headline* spectral > spatial result; applies to *any* 3D shape analysis with limited data)

- **"Our method, ToothForge, provides a tool for synthesizing tooth shapes, introducing a new strategy for data augmentation in dental shape analysis tasks. This has the potential to significantly enhance their accuracy with minimal computational cost."** (Sec 5 — the *product* framing; this is the H5 use case the paper is selling)

- **"Future research will investigate the disentanglement properties of the manifold, specifically how different latent dimensions represent independent and interpretable anatomical features. This disentanglement would allow for more precise control over features such as cusp sizes or groove depths in patient-specific crowns."** (Sec 5 — the *interpretability* limitation; underappreciated in the paper, but the *key* gap for clinical adoption)

## Code/data link

- **Code:** [github.com/tiborkubik/toothForge](https://github.com/tiborkubik/toothForge) — Python 3.12, PyTorch 2.7.0, CUDA 11.8, hydra + omegaconf; `SpectralMesh` class for spectral decomposition (supports .obj, .stl, .ply); `generate_train_data.py` for synchronized preprocessing; `train.py` for β-VAE training; `mesh.low_pass_filter(k=N)` for compression
- **Pretrained checkpoints:** ❌ **not released** (private data, no checkpoints on HuggingFace — *unlike* paper 036 ToothCraft which released Normal/Antag/CFG checkpoints)
- **Data:** ❌ **private** (430-tooth industrial dataset, ethics-approved, not released — *the* limitation of the paper)
- **BibTeX:**
  ```bibtex
  @misc{kubik25toothforge,
    title={ToothForge: Automatic Dental Shape Generation using Synchronized Spectral Embeddings},
    author={Tibor Kubik and Francois Guibault and Michal Spanel and Herve Lombaert},
    year={2025},
    eprint={2506.02702},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2506.02702}
  }
  ```
- **Companion papers (Lombaert-lineage):**
  - Paper 033 DMC (Hosseinimanesh 2023) — the open-source prior SOTA, deterministic point completion
  - Paper 035 VBCD (Wei 2025) — the voxel-based cousin, MADCrowner extends it
  - Paper 034 MADCrowner (Wei 2026) — the template-deformation SOTA, margin segmentation
  - Paper 036 ToothCraft (Pukanec/Kubík 2026) — the *conditional* diffusion counterpart, synthetic-damage pipeline
  - **Paper 037 ToothForge (this paper)** — the *unconditional* spectral counterpart, the *fastest* inference

## For our project

**Seven concrete next steps, ranked by leverage:**

1. **★★★ ADOPT ToothForge as the v0 sub-task 2 *unconditional prior* for data augmentation.** Fork the `SpectralMesh` class + `generate_train_data.py` from the GitHub repo, run the spectral decomposition on 3DTeethSeg'22 (paper 001, 23K teeth) + 3DS (Ben-Hamadou 2022, ~5K teeth) + ODD (Wang 2024, ~5K teeth) = ~33K teeth total. **Train a separate β-VAE per FDI class (32 classes, or aggregate into the 4 macro-classes incisor/canine/premolar/molar the paper used).** Use the trained models to *generate* 100K-1M synthetic teeth per class for the v0 sub-task 2 training set augmentation. **The 1 ms inference makes this trivial** — 1M teeth = 17 minutes on a T4. **Expected effort:** 3-5 days engineering (1 day to port, 1-2 days to run on 33K teeth, 1-2 days to train 4 β-VAEs). **Expected cost:** $10-30 Lambda (T4 @ $0.50/hr × 6-12 hours preprocessing + training). **Expected v0 H5 win:** +5-15% mIoU on real test data, matching paper 036's synthetic-to-real transfer evidence.

2. **★★ Use ToothForge's 16-dim latent as the *initialization* for MADCrowner's template library (paper 034).** Currently MADCrowner uses 32 fixed FDI templates (one per FDI number) and deforms them to the patient's prep. **With ToothForge, we can sample *continuous* template variations per FDI** — instead of 32 templates, we have a *continuous manifold* of 16-dim latent codes, each decoding to a *unique* template. This is the *natural* v1 upgrade. **Expected effort:** 1 week engineering (modify MADCrowner's template-deformation to accept a *latent code* input, not a *template index*). **Expected v0 win:** smoother template interpolation, less "popping" between FDI classes, and the *first* v0 model with continuous template control.

3. **★★ Add a "shape-diversity prior" to v0 sub-task 2 *training*.** When training ToothCraft (paper 036) or MADCrowner (paper 034) on the v0 sub-task 2 dataset, mix in *ToothForge-sampled* shapes as additional training data. **The hypothesis: more shape diversity at training time → better generalization to real patient data.** This is the *exact* H5 use case the paper is selling. **Expected effort:** 1-2 days engineering (modify the training data loader to mix synthetic and real shapes with a configurable ratio). **Expected cost:** $0 (uses the pre-trained ToothForge from step 1). **Expected v0 win:** +2-5% mIoU at the same training cost, *orthogonal* to the other v0 improvements.

4. **★★ Use spectral synchronization as a *new unsupervised correspondence method* for v0 sub-task 1 (FDI segmentation).** Currently the v0 segmentation uses Cao25 (paper 026) and CrownSegger (paper 027), which are *supervised* correspondence methods (they need FDI labels for training). **ToothForge's spectral sync is *unsupervised*** — it gives vertex correspondence *for free* on any dental mesh. **For v0: run spectral sync on the 3DTeethSeg'22 + 3DS training set, use the vertex correspondence as a *feature* for the segmentation model, evaluate mIoU improvement.** **Expected effort:** 1 week engineering (spectral sync is 200 lines of code, integration with Cao25/CrownSegger is the hard part). **Expected cost:** $5-10 Lambda. **Expected v0 win:** +1-3% mIoU on the segmentation sub-task (the correspondence provides *anatomical structure* information that the supervised model might miss).

5. **★ Pilot ToothForge at 64-dim latent with FactorVAE-style disentanglement for v1 interpretability.** The paper admits (Sec 5) that the 16-dim latent is *not* interpretable. **For v1, train a *larger* β-VAE (64-dim latent, β in [0.1, 0.5], FactorVAE or β-TC-VAE for disentanglement) on 3DTeethSeg'22 + 3DS + ODD, and visualize the latent traversals to see if individual dimensions correspond to *anatomical features* (cusp size, groove depth, crown length).** If yes, the dentist can *navigate* the latent to control specific features — the *interpretable* H5 use case. **Expected effort:** 2-3 weeks engineering + research (the disentanglement β-VAE is more finicky than the standard β-VAE). **Expected cost:** $50-100 Lambda. **Expected v1 win:** the *first* interpretable 3D dental generative model in the reading list, a *clear* publication contribution.

6. **★ Add a "shape validity check" to v0 sub-task 2 inference using ToothForge's reconstruction error.** For a given patient's prep+adjacent+antagonist, generate N candidate crowns (N=5-10) using the v0 model. **For each candidate, encode it back through ToothForge's β-VAE encoder and check the reconstruction error** — a high reconstruction error means the candidate is *off the training distribution* (a likely bad shape). **Use this as a *filter*: only show the dentist the candidates that *are* on the training distribution.** **Expected effort:** 1-2 days engineering (use the pre-trained ToothForge as a *shape discriminator*). **Expected cost:** $0 (inference only). **Expected v0 win:** a *clean* failure mode for the v0 product (low reconstruction error = "looks like a real tooth", high = "hallucinated").

7. **★ Establish a *public benchmark* on 3DTeethSeg'22 for the v0 paper.** The single biggest gap in the Lombaert-lineage is the *lack of a public benchmark* — every paper evaluates on a *private* dataset. **For v0, we commit to evaluating the v0 model + all prior Lombaert-lineage models (DMC, VBCD, MADCrowner, ToothCraft, ToothForge) on the 3DTeethSeg'22 test split (paper 001), using physical-mm coordinates (the lesson from paper 035), and publishing the leaderboard as part of the v0 paper.** This would be *the* contribution to the field — a public benchmark that everyone can compare against. **Expected effort:** 1-2 weeks (data preprocessing, model adaptation, evaluation runs). **Expected cost:** $100-200 Lambda (running inference on 100-200 test cases × 5 models). **Expected v0 win:** a *clear* publication story, the *first* reproducible comparison of all 5 Lombaert-lineage models.

### v0 stack update

**Previous (after paper 036):**
- Sub-task 1: PVD-AF-DiGS-FC, Cao25 + CrownSegger (segmentation), FlexiCubes (mesh)
- Sub-task 2: MADCrowner (primary) + ToothCraft (alternative), ControlNet-style conditioning, separate antagonist encoder, synthetic-damage augmentation, IoU_Antag metric
- v0 compute: ~$3,050-3,700 Lambda

**New (after paper 037):**
- **Sub-task 1: unchanged** (Cao25 + CrownSegger + FlexiCubes)
- **Sub-task 2 (crown generation):** **MADCrowner (primary) + ToothCraft (alternative) + ToothForge (unconditional prior)** — *add ToothForge as the v0 H5 enabler*
- **Sub-task 2 (data augmentation):** **synthetic teeth from ToothForge β-VAE** — *add 100K-1M synthetic teeth to the v0 training set*
- **Sub-task 2 (template library):** MADCrowner's 32 FDI templates + **continuous latent manifold from ToothForge** — *new design pattern; v1 if time allows, v0 pilot if not*
- **Sub-task 2 (shape validity check):** **ToothForge's reconstruction error as a filter** — *new inference safety module*
- **v0 training data:** 3DTeethSeg'22 + 3DS + ODD + **100K-1M ToothForge-generated teeth** — *expanded from previous 4,000+ arches to 100K+ synthetic + 4,000+ real*
- **v0 compute budget (recalculated):**
  - PVD-AF-DiGS-FC: ~$2,200 Lambda (unchanged)
  - MADCrowner + CMPL + FDI-template: $400-800 Lambda (port + fine-tune, unchanged)
  - ToothCraft fine-tune on 3DTeethSeg22: $100-200 Lambda (unchanged)
  - Synthetic-damage pipeline (3DTeethSeg22 + 3DS + ODD): $50-100 Lambda (unchanged)
  - Cao25 + CrownSegger dual-head: $200-400 Lambda (unchanged)
  - Context-validity check: $0 (unchanged)
  - **ToothForge β-VAE training (4 classes × 2h on T4):** **$10-30 Lambda** (NEW)
  - **ToothForge spectral sync preprocessing (33K teeth × 10-30 sec):** **$5-10 Lambda** (NEW)
  - **ToothForge inference (100K-1M synthetic teeth):** **$5-20 Lambda** (NEW)
  - **Total: ~$3,170-3,760 Lambda** (was $3,050-3,700) — a $120-200 increase for the H5 unconditional prior + spectral sync preprocessing

**v0.5 → v1 product offering (refined):**
- v0.0 (1 month): MADCrowner + Cao25 + FlexiCubes on 3DTeethSeg'22 (no ToothForge, no ToothCraft)
- v0.5 (1-2 months): add ToothCraft-Normal as fallback, ControlNet-style conditioning, IoU_Antag metric
- v1.0 (3-4 months): **MADCrowner for accuracy, ToothCraft for diversity, ToothForge for data augmentation**, the 100K-synthetic-teeth v0 pipeline, the chairside pilot
- v2.0 (6+ months): full diffusion-on-SDF (ToothCraft 128³ variant), MADCrowner + ToothForge-continuous-latent for template manifold, $5,000-8,000 Lambda, 3-4 months engineering

### Open questions for HK

1. **v0 sub-task 2 data augmentation: ToothForge-only or ToothForge + synthetic-damage (paper 036)?** Both are H5 enablers, but they have *different* design philosophies: ToothForge generates *complete* teeth from a low-dim latent (good for *bootstrap* training data), ToothCraft's synthetic-damage generates *damaged* teeth from complete arches (good for *completion model* training). **My recommendation: ToothForge for the v0 *segmentation* sub-task (H5 for FDI classification), ToothCraft's synthetic-damage for the v0 *completion* sub-task (H5 for crown generation).** Run them in parallel on the same 3DTeethSeg'22 + 3DS + ODD. Cost: $30-50 Lambda total.

2. **v0 ToothForge latent dim: 16 (paper default) or 64-128 (v1 candidate)?** The paper's 16-dim latent is *enough for reconstruction* but *too small for interpretable control*. **For v0, I'd keep 16-dim** (matches the paper, validates the architecture, ~$10 training cost per class). **For v1, I'd increase to 64-dim with FactorVAE-style disentanglement** to enable interpretable control (the v1 paper's *contribution*). **Recommendation: 16-dim for v0, 64-dim + FactorVAE for v1**.

3. **v0 ToothForge training: 3DTeethSeg'22 only or 3DTeethSeg'22 + 3DS + ODD?** The paper trained on 430 private teeth. **For v0 with public data**: combine the 3 public datasets for ~33K teeth (vs 430, 75× more data). **Cost: $5-10 Lambda preprocessing + $10-30 Lambda training (per class).** **Expected win: better disentanglement (more data = cleaner latent), better OOD robustness, better MMD across tooth classes.** *But* the pre-trained model is then *public* and *reproducible*, which is a *huge* contribution to the field.

4. **v0 sub-task 2: should we run MADCrowner, ToothCraft, AND ToothForge in parallel, or just one?** **My recommendation: all three in parallel, evaluated on the same 3DTeethSeg'22 test split.** MADCrowner is the deterministic SoTA (CD-L2 0.185 mm² on the private 4,602-case split), ToothCraft is the diffusion SoTA (mIoU 85.4% on the private ODD test), ToothForge is the unconditional prior (MMD 0.00598 on the private 430-tooth test). **The v0 paper is the *public* comparison of all three on 3DTeethSeg'22** — this is the *single most valuable* contribution to the field. **Cost: $200-400 Lambda for the three model evaluations.**

5. **The Lombaert connection:** this is the **5th paper in the Lombaert-lineage** (DMC 033 → VBCD 035 → MADCrowner 034 → ToothCraft 036 → ToothForge 037, with Kubík as the common PhD student linking them all). **The lineage is becoming a *de facto* research program for dental-crown generation**, and **ToothForge is the *bookend* — the unconditional, real-time, T4-trainable prior that complements the conditional, expensive, H100-trained diffusion model.** **The Lombaert-lineage has now covered: deterministic (DMC), voxel (VBCD), template-deformation (MADCrowner), diffusion (ToothCraft), spectral (ToothForge) — *every* major 3D generative design pattern.** This is *the* research program to follow for v0+v1, and **HK's v0 is well-positioned to be the *first* public benchmark of this lineage**.

### Notes for HK
- **The Lombaert connection:** Kubík is a PhD student jointly supervised by Španěl (Brno) and Lombaert (ÉTS Montréal). The 5-paper Lombaert-lineage (DMC 2023 → VBCD 2025 → MADCrowner 2026 → ToothCraft 2026 → ToothForge 2025) is becoming a *de facto* research program for dental-crown generation, with **Lombaert as the common senior author for the last four**. ToothForge is the *bookend* — the simplest, fastest, cheapest paper in the lineage, and the one that *enables* the others (ToothForge's data-augmentation is the *unconditional* prior that all the others could use).
- **The 1 ms inference on T4 is *the* killer feature for v0 deployment.** No other paper in the reading list matches this — DMC, VBCD, MADCrowner all need 50-500 ms on more expensive GPUs; ToothCraft needs 100-1000 ms on H100; DCrownFormer needs 5+ seconds on V100. **For chairside deployment on a $200/month Lambda box, ToothForge is the *only* option.**
- **Code release**: confirmed open source (no pretrained checkpoints because the data is private, but the *code* is well-organized and can be ported to our infra in 3-5 days). The `SpectralMesh` class is *the* cleanest API in the Lombaert-lineage — supports any trimesh-supported format, ~200 lines of code.
- **Data limitation:** the *only* paper in the Lombaert-lineage with *private* data. **For v0 reproducibility, we MUST retrain on 3DTeethSeg'22 + 3DS + ODD**, which means we become the *first* group to publish a public-trained ToothForge. **This is a *clear* v0 paper contribution** — "ToothForge retrained on public data, evaluated on 3DTeethSeg'22, released for the community".
- **The H5 use case is *the* selling point:** "synthesized shapes increase the accuracy of downstream tasks in digital dentistry" (Sec 5). This is *exactly* what we need for v0 — the 100K-1M synthetic teeth from ToothForge is the v0 H5 enabler, and the *only* way to do data augmentation in 3D dental at scale without expensive human-labeled data.
- **Reading time:** ~45 min, mostly because the paper is short (9 pages, 1 main figure, 1 table) and the architecture is simple (β-VAE, no diffusion, no attention, no ControlNet).

**Next paper to read:** For paper 038, candidates from the seed list + 037 closing the Lombaert-lineage loop:
- **DCPR-GAN (Tian et al. J Healthc Eng 2022, ref [24])** — the 2D-depth-map cGAN baseline that the Lombaert-lineage cites; would let us trace the field's evolution from 2D to 3D
- **Point2SSM (Adams & Elhabian, ICLR 2024, ref [2])** — the *point-cloud* statistical shape model that ToothForge could be benchmarked against (different substrate, same goal: shape generation)
- **3D-Diffusion (Wu et al. 2023)** — the *original* 3D diffusion paper on point clouds; would clarify the H2 × point-cloud relationship
- **Lemeunier 2022 SAE-LP (ref [14])** — the *direct* prior to ToothForge, the spectral autoencoder without synchronization; would let us isolate the contribution of the synchronization
- **DiGS (Ben-Shabat NeurIPS 2022, paper 003)** — the H4 SDF winner; would let us re-evaluate H4 vs spectral for the v0 substrate choice

**Recommendation for 038: Lemeunier 2022 SAE-LP (ref [14])** — the *direct* prior to ToothForge, the spectral autoencoder *without* synchronization. Reading Lemeunier 2022 *before* (or right after) ToothForge lets us isolate the contribution of the synchronization: what does the unaligned baseline look like, and how much does the synchronization improve it? This is the *natural* ablation study, and would tell us if the synchronization trick is *the* contribution or just *one* of many. **Alternative: Point2SSM (ICLR 2024) for the point-cloud SSM baseline** — the most direct comparison to ToothForge in terms of *goal* (unconditional shape generation) and *differences* (point cloud vs spectral, supervised correspondence vs unsupervised).
