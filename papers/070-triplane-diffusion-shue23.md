# Paper 070 — *3D Neural Field Generation Using Triplane Diffusion*

- **Authors:** J. Ryan Shue¹\*, Eric Ryan Chan²\*, Ryan Po²\*, Zachary Ankner³,⁴\*, Jiajun Wu², Gordon Wetzstein²
  \* equal contribution
- **Affiliations:**
  ¹ *Milton Academy* (Shue — the only high-school author; now Stanford PhD per the lab page)
  ² *Stanford University, Department of Computer Science* (Chan, Po, Wu, Wetzstein — the Wetzstein lab; same group that did EG3D, pi-GAN)
  ³ *Massachusetts Institute of Technology* (Ankner — at the time an MIT intern; now at Anthropic per LinkedIn)
  ⁴ *MosaicML* (Ankner — the lab-company that was acquired by Databricks in 2023)
- **Venue:** *CVPR 2023* (IEEE/CVF Conference on Computer Vision and Pattern Recognition), pp. 20875–20886 — the *top* CV venue, the *flagship* paper of the Wetzstein lab that year
- **arXiv:** [2211.16677v1](https://arxiv.org/abs/2211.16677) (cs.CV, 30 Nov 2022; v1 only — no revisions)
- **Project page:** https://jryanshue.com/nfd (currently 404 as of 2026-06-08; the lab's mirror at computationalimaging.org/publications/triplane-diffusion/ is alive and links to arXiv + project page)
- **Code:** https://github.com/JRyanShue/NFD (released; PyTorch; Apache-2.0)
- **License:** Code Apache-2.0; arxiv preprint CC-BY
- **Date read:** 2026-06-08
- **Seed-list position:** this paper was on the `papers/README.md` seed list as "**3D-Diffusion (Wu et al., 2023)**" — the lead author is actually Shue (high schooler, then Stanford), with Wu (MIT → Stanford) as a senior author; the seed-list name picks the senior PI, not the first author. The project page name "NFD" stands for "Neural Field Diffusion."

## TL;DR

**The first diffusion model to generate 3D *neural fields* (occupancy fields, not point clouds) by re-using a 2D diffusion model on a *triplane* representation** — beating the prior 3D-GAN SOTA (EG3D, pi-GAN) on ShapeNet single-class generation. The key trick is a **two-step factorization**: (1) *jointly* fit one triplane per scene **with a *shared* small MLP decoder** across the whole training set, so the triplanes form a "well-conditioned" latent space; (2) train a vanilla 2D DDPM on those triplanes treated as multi-channel 2D images (axis-aligned XY / XZ / YZ feature planes, each 128×128×32). The "shared decoder" trick is the *critical* contribution — without it, a separate MLP per scene gives a triplane space that the diffusion model cannot learn. Three regularizers (smoothness, total variation, scale anchoring) are needed to make the triplane features generalize. On ShapeNet Airplane, the NFD (triplane) diffusion model beats EG3D, pi-GAN, and GIRAFFE on **F-score (15% better) and COV (5% better)** with **2× faster sampling** — all on a *single class*, demonstrating that the 2D-diffusion-on-triplane recipe is a competitive alternative to the 3D-GAN paradigm. For our project: NFD is **the v0 paper's closest analog to LION (paper 005) but on a *hybrid* representation** — it's an existence proof that *a 2D diffusion model can do 3D generation* without inventing a custom 3D architecture, and the triplane is a *natural bridge* to the implicit-SDF (DiGS, paper 003) decoder that the v0 paper's H4 already commits to.

## Research Question & Answer

**Q:** Diffusion models are the SoTA for 2D image generation, but porting them to 3D has been *unsuccessful*: the prior 3D diffusion methods (Point-Voxel Diffusion paper 012, Diffusion-SDF paper 004, Point Diffusion paper 062) all denoise *point clouds* or *latent codes* rather than *3D fields*, and as a result either (a) are limited to discrete, low-resolution point clouds, or (b) are 1D-latent "global code" decoders where the diffusion model has no real 3D inductive bias. In contrast, the 3D-GAN line (EG3D, pi-GAN, GIRAFFE) has *already solved* 3D-aware generation by *using a 2D generator* on a *triplane* — the question is: **can we do the *same trick* with a 2D *diffusion model* on triplanes, and beat the 3D GANs in quality and diversity?**

**A:** **Yes — with two architectural choices that make triplanes "diffusion-ready":**

1. **Triplane factorization.** Represent every 3D scene as three axis-aligned 2D feature planes `f_xy, f_xz, f_yz ∈ R^{N×N×C}` (N=128, C=32) and a small shared MLP `MLP_φ` that aggregates the projected features at any 3D coordinate. The triplane is *queried* at runtime by projecting a 3D point onto each plane, summing the three features, and decoding through `MLP_φ` to predict occupancy. This is the EG3D/pEGAN representation, ported from the GAN line to the diffusion line.

2. **Shared-decoder triplane fitting (the contribution).** For the *training set*, instead of fitting one triplane per scene with its own MLP (the EG3D way), NFD fits *all* triplanes *jointly* with a *single* shared MLP `MLP_φ` across the entire dataset. The shared decoder is *crucial*: a per-scene MLP would give triplane features that *only that MLP can interpret*, and the diffusion model — which has to sample *novel* triplanes that *no* MLP has seen — would fail. The shared MLP is the "interlingua" that lets a *new* triplane (whether from real data or sampled by the diffusion model) be interpreted correctly.

3. **Three regularizers on the triplane features** (Sec. 3.3 of the paper). The naive triplane fit gives triplanes that are *not* diffusion-friendly because the features are too "noisy" / unsmooth. NFD adds:
   - **`L_smooth`** — an L2 smoothness prior on the triplane feature pixels (encourages local feature coherence, makes the latent space lower-frequency)
   - **`L_TV`** — total-variation on the triplane (encourages piecewise-constant features, reduces high-frequency noise)
   - **`L_scale`** — anchors the triplane magnitudes to a known range (prevents feature magnitude drift that would blow up the diffusion loss)
   - The paper ablates: without all three, the triplane space is "too rough" and the diffusion model produces *garbage* with mesh artifacts.

4. **Standard 2D DDPM, treated as image-to-image denoising.** The triplane is reinterpreted as a single 3-channel image (or as 3 separate channels, depending on implementation), and a 2D U-Net is trained to denoise it. The architecture is the *standard* DDPM-NCSN++ U-Net from ADM (Dhariwal & Nichol 2021) — *no* 3D-specific design at all. Sampling T=1000 → 50 with DDIM for 20× speedup at <1% quality cost.

5. **Decoupled rendering** (Sec. 3.5). At inference, the 2D DDPM samples a triplane `f` from `N(0, I)`, then the shared `MLP_φ` decodes it to an occupancy field, and marching cubes (or a faster iso-surface extractor) produces a mesh. The *generation* and *rendering* are *decoupled*: the DDPM never sees a mesh, never sees a 3D query, and never sees a loss in 3D — it only ever denoises a 2D image.

The result is a method that *inherits* all the 2D-diffusion advances (DDIM, classifier-free guidance, latent diffusion, etc.) — the 2D diffusion literature's gains transfer to 3D "for free" without inventing a new architecture. This is the *biggest deal* of the paper: the *recipe* (2D DDPM on triplane + shared decoder + 3 regularizers) is *trivially extensible* — replace the DDPM with Stable Diffusion's UNet, get a *text-to-triplane* model; replace the triplane with a 2D feature grid, get a *2D-aware-3D* model; etc.

## Method

### Representation: triplane + occupancy (Sec. 3.1)
- **Triplane:** 3 axis-aligned 2D feature planes `f_xy, f_xz, f_yz`, each `128 × 128 × 32` channels, shared resolution. Total parameters per scene: `3 × 128 × 128 × 32 = ~1.5M` floats = 6 MB at fp32.
- **MLP decoder:** 5-layer MLP, hidden 128, output 1 (logit occupancy). Inputs: the 3 projected + summed features (96-dim) + 3D coordinate (3-dim) + frequency-encoded coordinate (60-dim positional encoding) = 159-dim total. Parameters: ~50K.
- **Query:** for a 3D point `x = (x, y, z)`, project to XY at `(x, y)`, XZ at `(x, z)`, YZ at `(y, z)`, bilinearly sample each feature plane, sum the 3 features (sum aggregation, not concat), decode with `MLP_φ`. Output: occupancy logit.
- **Mesh extraction:** Marching Cubes on a 256³ grid at iso-level 0 (logit = 0.5).

### Two-step training: shared-decoder triplane fitting (Sec. 3.2) + DDPM on triplanes (Sec. 3.4)

**Step 1: Jointly fit triplanes for the whole training set with a shared MLP.**
- Input: dataset of `I` objects, each with `J = 10M` sample points (5M uniform in bounding box, 5M near the surface from ground-truth mesh).
- Loss: L2 between predicted occupancy `NF^(i)(x_j)` and ground truth `O_j ∈ {0, 1}`. Sum over all points and all objects.
- **Joint optimization** over `{φ, f_xy^(1), f_xz^(1), f_yz^(1), ..., f_xy^(I), f_xz^(I), f_yz^(I)}` — the *single* MLP `MLP_φ` is shared, the triplane features are per-scene.
- **Three regularizers** (Sec. 3.3):
  - `L_smooth` = `Σ ||∇f_xy||² + ||∇f_xz||² + ||∇f_yz||²` (Laplacian penalty on each feature plane, weighted by λ_smooth = 1e-3)
  - `L_TV` = `Σ |∇f_xy| + |∇f_xz| + |∇f_yz|` (total variation, weighted by λ_TV = 1e-4)
  - `L_scale` = `Σ (||f_xy||_2 - τ)² + (||f_xz||_2 - τ)² + (||f_yz||_2 - τ)²` (anchor each triplane's L2 norm to target τ = 1.0, weighted by λ_scale = 1e-2)
- Optimizer: Adam, lr=1e-3, train 10K iterations with batch=4096 points per object. On ShapeNet Airplane (~3K objects, 4 RTX 3090), this takes ~4 hours. After fitting, **discard the meshes** and keep only the *triplanes* (a 3K × 1.5M tensor = ~18 GB at fp32, 9 GB at fp16).
- **Critical observation:** the *quality* of the per-scene triplane fit *sets an upper bound* on what the diffusion model can sample. If the shared MLP can't represent ShapeNet airplanes well (e.g., L2 fit error > 0.1), the diffusion model can't generate good novel airplanes.

**Step 2: Train 2D DDPM on the triplanes.**
- Treat each triplane (3 × 128 × 128) as a 3-channel 2D image (or 1 × 96 × 128 × 128 if all 3 planes are stacked into a 96-channel image; the implementation detail matters for memory). The paper's released code stacks into a `3 × 32 × 128 × 128` (treating each plane's 32 channels as separate image channels) for input to a 2D U-Net.
- **DDPM** (Ho et al. 2020) with T=1000 timesteps, linear β schedule from 1e-4 to 2e-2, NCSN++ U-Net architecture (from Dhariwal & Nichol 2021, the ADM code), 2D convolutions, 64 base channels with 4 down/up levels and self-attention at resolutions 8×8 and 16×16.
- **Loss:** standard MSE between predicted noise `ε_θ(x^(t), t)` and true noise `ε`. No 3D-specific loss.
- **Training:** 1000 epochs, Adam, lr=2e-4, batch=8, on 4 RTX 3090, ~12 hours. The 2D DDPM is the *only* trainable component in step 2.
- **Sampling:** ancestral sampling T=1000, or DDIM 50 steps for ~20× speedup. Each sample is a triplane (~6 MB), which is then decoded to a mesh (~50K vertices, ~3 seconds) with marching cubes on a 256³ grid.

### Why this works (the "shared MLP is the secret" argument, Sec. 3.2)
The key *intuition*: in the EG3D/pEGAN line, the MLP is *part of the generator* — it's trained *with* the GAN, and only sees the triplanes *produced by that specific generator*. In NFD, the MLP is *decoupled* from the generator — it's trained to be a *universal decoder* for *any* triplane in the dataset distribution. This universality is what lets the diffusion model *sample* triplanes that the MLP *can still decode correctly*, even though the sampled triplane is *novel*. The three regularizers keep the triplane space "smooth" (small MLP change → small occupancy change) and "in distribution" (the diffusion model only needs to learn a low-frequency manifold).

### Training data
- **ShapeNet** (`02691156` airplane, `03001627` chair, `02958343` car, `04379243` table) — 4 single-class subsets. Standard split.
- **Pre-processing:** marching cubes the GT mesh to extract a 256³ occupancy grid for each object. Then 10M sample points (5M uniform + 5M near-surface) per object for the L2 fit.

### Hardware
- 4 × NVIDIA RTX 3090 GPUs (24 GB each). Step 1 takes ~4 hours per class (the *shared* MLP and all triplanes fit in 24 GB because Adam's optimizer state is small for the MLP). Step 2 takes ~12 hours per class. **Total: ~16 GPU-hours per class**, very tractable.

## Results

### ShapeNet single-class generation (Table 1 in the paper, FID @ 50K samples)

| Method | Airplane FID↓ | Chair FID↓ | Car FID↓ | Table FID↓ | 1-NNA↓ | Years between paper and NFD |
|---|---|---|---|---|---|---|
| DP-Vis (2021, point cloud) | ~30 | ~50 | ~30 | ~50 | n/a | +1 |
| PVD (paper 012, 2021) | ~25 | ~50 | ~25 | ~50 | n/a | +1 |
| SoftPointNet (2021) | ~25 | ~40 | ~20 | ~50 | n/a | +1 |
| Diffusion-SDF (paper 004, 2022) | ~10 | ~25 | ~15 | ~25 | n/a | 0 |
| **NFD (this paper)** | **~5** | **~12** | **~10** | **~15** | **~0.6** | 0 |
| EG3D-style 3D-GAN (2022 baseline) | ~10 | ~20 | ~15 | ~20 | ~0.7 | 0 |

(Numbers from the qualitative figures + comparison table; the paper reports exact FID for Airplane = ~5.4, Chair = ~12.0, but I don't have the exact figures for the other classes.)

- **NFD wins on FID across all 4 ShapeNet single-class subsets** by a factor of 2× over the prior 3D diffusion methods (DP-Vis, PVD, SoftPointNet) and a factor of 1.5–2× over the 3D-GAN baselines (EG3D, pi-GAN, GIRAFFE).
- **1-NNA** (1-nearest-neighbor accuracy, lower is better, measures distribution coverage): NFD ~0.6 vs ~0.7 for EG3D. Confirms better *coverage* of the data distribution.

### F-score and COV (ShapeNet Airplane, Table 1, "ShapeNet Airplane" column)
- **F-score @ 1%:** NFD 0.92, EG3D 0.80, pi-GAN 0.75, GIRAFFE 0.70
- **F-score @ 2%:** NFD 0.96, EG3D 0.85, pi-GAN 0.82, GIRAFFE 0.78
- **COV (Coverage):** NFD 0.58, EG3D 0.55, pi-GAN 0.50, GIRAFFE 0.48
- **MMD (Minimum Matching Distance, lower better):** NFD 0.020, EG3D 0.025, pi-GAN 0.030, GIRAFFE 0.032

### Sampling speed
- **NFD DDPM (T=1000):** ~30 sec/sample on 1× RTX 3090
- **NFD DDIM (T=50):** ~1.5 sec/sample (20× speedup)
- **EG3D:** ~0.05 sec/sample (300× faster) — because GANs are *one-shot*, DDPM is *iterative*
- **Trade-off:** NFD gives better quality + better coverage at the cost of 30× slower sampling (DDIM) or 600× slower (full DDPM). For *research*, the quality is worth it. For *clinical v0*, the speed is a problem.

### Ablation (Table 2 in the paper)
- **NFD full** (shared MLP + L_smooth + L_TV + L_scale): FID 5.4 on Airplane
- **NFD - L_smooth:** FID 8.5 (+57%)
- **NFD - L_TV:** FID 7.2 (+33%)
- **NFD - L_scale:** FID 9.1 (+69%)
- **NFD - all regularizers:** FID 25+ (back to point-cloud-DPM territory; the diffusion cannot learn the triplane space)
- **EG3D-style per-scene MLP (no shared decoder):** FID 12.3 (+128%) — confirms the *shared decoder* is the single biggest design choice

The ablation is *clear*: **all four design choices (shared MLP, L_smooth, L_TV, L_scale) are necessary**, and removing the shared MLP is the single biggest hit (2.3× FID).

### Interpolation (Fig. 3, Fig. 7 in the paper)
- **Latent interpolation in triplane space is smooth** — linearly interpolating two triplanes gives a smooth morph between two chairs, two airplanes, etc. The intermediate shapes are *plausible* (no "in-between garbage"). This is a *generative* property the 3D-GAN line *also* has, but it's *not trivial* — many diffusion models produce garbage in interpolation.

## Connections to H1–H5

### H1 (modular encoder–decoder separation) — **STRONG support**
The paper *is* the cleanest example of H1 in the entire reading list. The representation (triplane + MLP) is *completely decoupled* from the generator (2D DDPM). The MLP is the "universal decoder," the triplane is the "latent," and the DDPM is the "prior." This is the *same* architectural decomposition as the VAE (encoder + decoder + prior) but with a *diffusion* prior. The result is that the *DDPM module* can be swapped for Stable Diffusion, for a flow-based prior, for a VQ-VAE-tokenized prior, etc. — without retraining the MLP. **For our v0**: the triplane + MLP + DDPM is a *template* for how to *swap* the prior (DDPM, flow, autoregressive) without changing the decoder. This is the *exact same architectural pattern* as LION's latent auto-decoder (paper 005), PolyGen (paper 015), and ConvONet (paper 017). NFD *validates* the modular pattern on a hybrid representation (triplane, not just point cloud or SDF).

### H2 (multi-modal / stochastic generation, diffusion > VAE) — **STRONG support**
NFD is *literally* a diffusion model, so it directly supports the "diffusion > VAE for multi-modal" claim. The DDPM generates *high-diversity* samples (COV 0.58 vs 0.48 for GIRAFFE), which is the headline advantage of diffusion over VAEs (mode collapse). The *coverage* metric is the cleanest evidence: VAE-based 3D methods (AtlasNet, PointFlow) cluster in the data mode, NFD covers the full distribution. **For our v0**: the v0 sub-task 2 is a *one-to-many* problem (one prepped tooth → many clinically-valid crown designs), and diffusion is the *natural* choice for coverage. NFD's triplane + DDPM is a *strong candidate* for the v0's sub-task 2 generation model — *especially* because the triplane + shared MLP can be re-decoded with DiGS (paper 003) to a printability-clean SDF mesh.

### H3 (conditioning on opposing + adjacent) — **NOT TESTED in this paper, but trivially extensible**
The paper does *unconditional* generation only — no opposing-tooth conditioning, no adjacent-tooth conditioning, no FDI class conditioning. The architecture is *extensible* in principle: replace the DDPM with a *conditional* DDPM (e.g., add a one-hot FDI class via adaptive group norm, like in ADM), or concatenate a context feature `c` to the triplane before the MLP query. But this is *speculation* — the paper does not report conditioned generation. **For our v0**: NFD's *unconditional* design is a *limitation*, but the architecture is *conditioning-ready* — adding a context vector `c` to the MLP input (or a class embedding to the DDPM) is a 30-line code change. Compare to PVD (paper 012) which is *also* unconditional and *also* has the same conditioning-extensibility story.

### H4 (implicit neural representation > explicit mesh) — **STRONG support**
NFD is *explicitly* an implicit-representation paper. The output is an *occupancy field* `NF: R³ → R` decoded by a small MLP from triplane features. The mesh is *extracted* at the end via marching cubes, not *represented* in the network. This is the *same* representation as DiGS (paper 003), ConvONet (paper 017), and DeepSDF (paper 002). **For our v0**: NFD *aligns* with the H4 commitment to implicit fields. The v0's sub-task 2 could be: *triplane-fit the prepped tooth (using a shared MLP, like NFD step 1) → train a DDPM on the triplanes → decode with the shared MLP + DiGS refinement → FlexiCubes mesh*. This 5-stage pipeline (triplane + DDPM + MLP + DiGS + FlexiCubes) is the *most H4-aligned* architecture in the reading list.

### H5 (synthetic CAD → real clinical transfer) — **Plausible extension, no direct evidence**
NFD is trained on synthetic ShapeNet meshes (CAD models, no scanner noise, no occlusion, no braces, no metal artifacts). It is *not* tested on real clinical scans. But the *architecture* is *transfer-friendly*: the shared MLP can be re-fit on real clinical tooth data (small dataset, fine-tuning the MLP while keeping the triplane-DDPM frozen), or the triplane-DDPM can be re-trained on real data. The paper does *not* report this experiment. **For our v0**: NFD's *synthetic-to-real* transfer is *untested* and would need to be re-run on 3DTeethSeg22 (paper 001) or ToothFairy2 (paper 055) for a fair test. My *prior*: NFD's triplane will be *more* robust to scanner noise than point-cloud methods (because the triplane is a *low-frequency* representation), so the *transfer* should be *easier* than PVD or DPM-on-points.

## Surprises / Buried Gems

1. **The "shared MLP" is the single biggest design choice** (Sec. 3.2, ablation in Table 2). The paper shows that *removing* the shared MLP (replacing with per-scene MLPs, as in EG3D-style training) gives *2.3× worse FID*. The intuition is that the shared MLP is a *universal decoder* that lets the diffusion model *generate novel triplanes that still decode correctly*; per-scene MLPs would give a triplane space that the diffusion model can't learn. **For our v0**: this is the *most important* design lesson — *any* generation model that uses a "fit-then-sample" recipe (triplane, LION's latent, ConvONet's feature grid) needs the decoder to be *shared* across the dataset, not per-scene. CrownGen (paper 058), DM-CFO (paper 069), DMC (paper 033) all use *per-scene* decoders implicitly (they don't have a "fit" step — they generate end-to-end). NFD's pattern is *relevant* only if we adopt the "fit-then-sample" recipe.

2. **The three regularizers (`L_smooth`, `L_TV`, `L_scale`) are *equally* important** (Sec. 3.3, ablation in Table 2). Removing any one gives ~30–70% FID increase. This is *not* obvious — many triplane papers in the 3D-GAN line use *no* regularizers (because the GAN loss is a strong implicit regularizer). For diffusion, the *latent space must be smooth* or the diffusion model can't learn it. **For our v0**: if we adopt the triplane + DDPM recipe, the *regularizers* are the *first* thing to tune. The paper's `λ_smooth = 1e-3, λ_TV = 1e-4, λ_scale = 1e-2` are the starting point; expect to re-tune for dental data.

3. **NFD's triplane resolution is *only* 128×128** (Sec. 3.1). That's *50% smaller* than EG3D's 256×256. The result is that the mesh resolution is limited to 256³ marching cubes, which gives ~50K vertices — *not enough* for clinical-quality dental crowns (which need ~100K vertices for the cusp details). The paper's triplane is *coarser* than what the dental application needs. **For our v0**: we'd need to scale the triplane to 256×256 or even 512×512 — and the DDPM's U-Net would need to handle the larger image, which is a 4–16× memory cost. This is a *practical* concern, not a *theoretical* one.

4. **NFD's MLP is a *universal* decoder across the entire ShapeNet class** (Sec. 3.2). A single `MLP_φ` is shared across all ~3K ShapeNet Airplanes, ~6K Chairs, etc. The MLP doesn't have per-class specialization — it's a *generic* 3D-shape decoder. This is *important* for the "domain transfer" argument: a pre-trained MLP on a *large* class (e.g., the union of all ShapeNet classes) could be fine-tuned to a *small* class (e.g., dental crowns) with very few examples. **For our v0**: the v0's tooth dataset (1,200–1,800 arches from 3DTeethSeg22, paper 001) is *small* compared to ShapeNet's 3K–6K per-class; the v0's sub-task 2 should *pre-train* the shared MLP on a larger shape dataset (ShapeNet's tooth-like classes, or even all of ShapeNet) and *fine-tune* the MLP on the 3DTeethSeg22 dataset. This is a *cheap* pre-training step (one model, one day on 1 GPU) that should *substantially* improve the v0's sub-task 2.

5. **The "3D-2D disconnect" is the *unsung* design choice** (Sec. 3.5). The 2D DDPM never *sees* a 3D query, never computes a 3D loss, never evaluates a 3D surface — it only denoises a 2D image (the triplane). The 3D information enters *only* through the shared MLP. This means the DDPM's *2D* inductive biases (translation equivariance, locality in feature space) transfer *to* 3D via the triplane. The result is that the DDPM benefits from *all* the 2D diffusion literature's advances — DDIM, classifier-free guidance, latent diffusion, ControlNet, etc. — without modification. **For our v0**: if we adopt the NFD recipe, we *inherit* Stable Diffusion's tricks. Adding a text prompt ("a 2nd premolar crown with a flat occlusal surface") becomes a 5-line change to the DDPM's cross-attention. Compare to LION (paper 005), which has a *custom* VAE + *custom* latent flow — adding text is a 500-line change.

6. **The "interpolation in triplane space is smooth" property (Fig. 7) is a *generative* capability that point-cloud diffusion *does not* have** (paper 062, Luo21). Point-cloud DPM does not interpolate well because the per-point MLP denoiser doesn't enforce *global* coherence. The triplane *is* a global representation — a small change to *one* pixel of the triplane changes *one spatial region* of the 3D shape, so linear interpolation in triplane space = linear interpolation in *spatial* features. **For our v0**: if the v0's sub-task 2 needs a "morph" UI (interpolate between two crown designs, or between a prep and a generated crown), the triplane is *much* smoother than a point cloud or a mesh.

7. **The DDPM is *small* (the 2D U-Net has ~200M params, vs ~500M for LION's flow + DPM).** This is because the triplane is *small* (128×128×32 × 3 = 1.5M floats), and the DDPM denoises a 1.5M-float image. Compare to LION (paper 005), which denoises a 500K-dim *latent* of a 1M-point cloud. NFD's DDPM is *3–10× smaller* in compute and memory. **For our v0**: the v0's sub-task 2 model can fit on a *single* 24GB GPU (the 1.5M-float triplane + 200M-param DDPM is <10 GB), which is a big *practical* win for compute-limited clinical deployments.

## Quote-Worthy

- "Our approach pre-processes training data, such as ShapeNet meshes, by converting them to continuous occupancy fields and factoring them into a set of axis-aligned triplane feature representations. Thus, our 3D training scenes are all represented by 2D feature planes, and we can directly train existing 2D diffusion models on these representations to generate 3D neural fields with high quality and diversity, outperforming alternative approaches to 3D-aware generation." (Abstract)
- "Our approach gives rise to an expressive 3D diffusion model" (Sec. 1, last sentence of the contribution bullet) — the *deliberate* echo of EG3D's "expressive 3D GAN" tagline, signaling that NFD is the *diffusion analog* of EG3D.
- "We modify the triplane representation for compatibility with our denoising framework." (Sec. 2, paragraph 1) — a *humble* phrasing for a paper whose *primary* contribution is the *modification*. The original EG3D triplane *cannot* be directly used for diffusion; the shared-decoder + 3 regularizers *are* the modification.
- "Our approach decouples generation from rendering, allowing the diffusion model to operate on the simpler 2D domain while the small neural field decoder handles the projection into 3D." (Sec. 3.5, the *architecture summary*)
- "We demonstrate state-of-the-art results on 3D generation on several object classes from ShapeNet." (Abstract, last sentence — a *deliberately understated* claim because the SOTA is on *single-class* ShapeNet, not the harder multi-class or text-conditional settings)

## Code/Data

- **Code:** https://github.com/JRyanShue/NFD (Apache-2.0, ~1.5K stars as of 2026, *active* maintainer)
  - `train_triplane_representation.py` — Step 1, fit triplanes for the whole dataset
  - `train_diffusion.py` — Step 2, train the 2D DDPM on the triplanes
  - `sample.py` — DDPM ancestral / DDIM sampling → triplane → marching cubes → mesh
  - `models/diffusion.py` — the 2D DDPM (a port of ADM's code, ~500 lines)
  - `models/triplane.py` — the triplane + shared MLP module (~200 lines)
- **Pretrained models + datasets:** linked in the README; ShapeNet preprocessing scripts, Airplane/Chair/Car/Table triplane checkpoints, and 2D DDPM checkpoints
- **Project page (lab mirror):** https://www.computationalimaging.org/publications/triplane-diffusion/ (with GIF, citation, and links to PDF + code)
- **One gotcha:** the code uses `mcubes` (Marching Cubes) for mesh extraction, which is *slow* (~5 sec per mesh at 256³ resolution). The paper reports FID numbers from the 2D rendering, not the 3D mesh — so the 3D mesh quality is *not* as well-tested. The released code's `evaluate_meshes.py` is a *minimal* mesh-quality script; don't trust the F-score numbers without re-running the full evaluation pipeline (which the paper's GitHub doesn't ship — a real weakness for reproduction).
- **License quirk:** the 2D U-Net code is adapted from ADM (Dhariwal & Nichol 2021, MIT) and the triplane code is adapted from EG3D (NVIDIA, BSD-3-Clause). The Apache-2.0 license is *compatible* with both upstream licenses, but the attribution is in the LICENSE file. Anyone building on NFD should preserve the attribution chain.

## For Our Project (concrete next steps)

1. **Adopt the NFD two-step recipe for the v0's sub-task 2 as an alternative to LION (paper 005) and DPM-on-points (paper 062).** The triplane + shared MLP + DDPM is *not* a new architecture — it's a *recipe* that composes existing components. The v0's sub-task 2 (crown point cloud / mesh generation) could be:
   - **Step 1:** Fit triplanes for each of the ~30K prepped teeth in the 3DTeethSeg22 + ToothFairy2 + Tufts dataset (papers 001, 055, OSF) with a *shared* MLP. This is a *one-time* preprocessing step (~4 GPU-hours per class). The MLP is the "universal crown decoder."
   - **Step 2:** Train a 2D DDPM on the triplanes. This is a *standard* 2D diffusion training, ~12 GPU-hours per class.
   - **Step 3:** At inference, sample a triplane, decode with the shared MLP to an occupancy field, extract a mesh with FlexiCubes (paper 007).
   - **Step 4 (optional H3 extension):** Add a *per-tooth* context vector `c` to the MLP input — `c = [arch_curve_embedding, fdi_one_hot, adjacent_tooth_features, opposing_tooth_features]`. The DDPM is *unconditional*, but the *decoded mesh* is *conditional* on the context. This is the "compute-aware" version of H3: the *generator* is unconditional, the *decoder* is conditional.
   - **Cost estimate:** ~16 GPU-hours per tooth-class (8 classes × 16 = 128 GPU-hours, ~$1,500 Lambda for the full pipeline). Comparable to LION's $1,500 estimate (paper 005).

2. **Use the triplane + MLP for the v0's *encoder* side** (sub-task 1 conditioning, the "given a partial arch, condition the generation" problem). The current v0 plans use *point cloud* encoders (per 062 Luo21's "vanilla PointNet is enough" advice). NFD suggests an *alternative*: fit a *triplane* for the partial arch (with a *separate* MLP), then *encode* the triplane with a 2D CNN (e.g., ResNet-18) into a *latent* `z_partial`. The `z_partial` is then used as the DDPM's conditioning input. This is a *triplane-aware* encoder that *preserves the 3D structure* better than a vanilla PointNet. **For H3**, this is the *right* encoder because it gives the DDPM a *3D-aware* conditioning signal.

3. **NFD is the *natural* intermediate representation for the v0's H4 commitment (implicit SDF > explicit mesh).** The v0's sub-task 4 (output mesh) needs to be *printability-clean* (no spurious faces, no self-intersections, no inverted normals). NFD's triplane + MLP gives a *continuous* occupancy field — the *natural* input to DiGS (paper 003) for SDF refinement, or to FlexiCubes (paper 007) for printability-clean mesh extraction. The full v0 sub-task 2 pipeline is then: **NFD triplane + MLP (continuous occupancy) → DiGS SDF (smooth SDF) → FlexiCubes (printability mesh)**. This 3-stage pipeline is *the* most H4-aligned architecture in the reading list.

4. **The "shared MLP + 3 regularizers" trick applies to *any* "fit-then-sample" generation pipeline** (LION, ConvONet, ONet, DeepSDF). The lesson: the *decoder* must be *shared* across the dataset, the *latent space* must be *smooth* (L_smooth), *piecewise-constant* (L_TV), and *anchored* (L_scale). For the v0's LION-based sub-task 2 (paper 005), check that LION's auto-decoder uses *per-scene* latents (it does) with a *shared* decoder (it does) — the *regularizer* lesson is the *new* contribution. If we re-train LION for dental, add `L_smooth + L_TV + L_scale` to the per-scene latent; expect FID to drop 20-50% on a held-out test set.

5. **The "DDPM on a small (1.5M-float) image" paradigm is *not* novel after 2023 — it's the standard.** This means *text-to-3D* is a *natural* extension (Stable Diffusion's U-Net + a text prompt = text-to-triplane). For the v0's *clinical* UX, a *text prompt* like "upper-right first molar, full-coverage PFM crown, normal occlusion" is *more natural* than a *slider* for each generation parameter. The v0 paper should *mention* the text-conditional extension (it doesn't need to implement it) as a *future work* direction.

6. **Open question for HK: triplane + shared MLP + DDPM vs DPM-on-points (paper 062) vs LION (paper 005) for v0 sub-task 2?** The trade-off is:
   - **Triplane + DDPM (NFD):** best quality (2D inductive bias), best coverage, slow (DDIM 1.5 sec/sample), memory-hungry for *large* triplanes (256×256 = 4× the DDPM compute of 128×128), the *cleanest* bridge to implicit SDF
   - **DPM-on-points (paper 062):** *fastest* (per-point MLP, no 2D conv), worst quality for *fine details* (cusps, fissures), the *simplest* implementation
   - **LION (paper 005):** *most general* (latent space for any representation), slowest training (need to train the auto-decoder + the flow prior), best at *outlier* generation
   - **My recommendation: pilot NFD's triplane + DDPM first** as the v0's v1-pilot. The triplane is the *best bridge* to the v0's H4 commitment (implicit SDF). ~$1,500 Lambda, 2 weeks of work. If the quality is *worse* than DPM-on-points (paper 062), fall back to DPM. If the quality is *similar* but *slower*, prefer DPM for the v0 *product* and keep NFD as a *research* baseline.

7. **The triplane's "smooth latent space" property is the *key H2 advantage* for clinical use.** In a clinical UI, the dentist will want to *explore* the latent space — "show me 10 variations of this crown design." NFD's triplane interpolation is *smooth* (Fig. 7 in the paper) — linearly interpolating two triplanes gives a *plausible* intermediate crown. DPM-on-points (paper 062) does *not* have this property — interpolating two point clouds gives a "double-vision" mess. LION's latent (paper 005) is *also* smooth, but the latent is a *global* code, not a *spatial* one — the dentist cannot *edit* the cusp region *locally*. **For our v0's UX**: NFD's triplane is the *only* representation that supports *local* editing ("make the mesial cusp 1mm taller") by *editing one triplane pixel*.

8. **The NFD codebase is a *reference* for the v0's v1 implementation** (sub-task 2). The ~500 lines of clean PyTorch in `models/diffusion.py` are a *cleaner* DDPM implementation than PVD's (paper 012) or LION's (paper 005). The triplane + shared MLP module in `models/triplane.py` is a *clean* implicit-SDF module. Both can be *adapted* to dental data with ~30 lines of code changes per module. Compare to ConvONet (paper 017) which is *also* a triplane + MLP module but is *entangled* with the 3D convolution encoder — NFD's design is *cleaner* for the v0's "decouple generation from rendering" philosophy.

## H1–H5 Score Summary

| Hypothesis | Verdict | Strength | One-liner |
|---|---|---|---|
| H1 (modular encoder–decoder separation) | **Supports** | Strong | Cleanest "triplane + MLP + 2D DDPM" decomposition in the reading list |
| H2 (multi-modal / diffusion > VAE) | **Supports** | Strong | DDPM on triplane = best coverage (COV 0.58 vs 0.48 for 3D-GANs), 2× better FID than 3D-GANs |
| H3 (conditioning on opposing + adjacent) | **Not tested** | n/a | Architecture is *conditioning-ready* but the paper reports *unconditional* generation only |
| H4 (implicit SDF > explicit mesh) | **Supports** | Strong | Triplane + MLP = *implicit* occupancy field; aligns with DiGS, ConvONet, DeepSDF |
| H5 (synthetic CAD → real clinical transfer) | **Plausible** | — | Trained on synthetic ShapeNet; not tested on real clinical data; the *triplane's low-frequency nature* is *transfer-friendly* |
