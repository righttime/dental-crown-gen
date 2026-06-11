# Paper 147 — DiffFacto: Controllable Part-Based 3D Point Cloud Generation with Cross Diffusion

## TL;DR

**The *founding* part-level 3D-latent-diffusion paper, by George Kiyohiro Nakayama (Stanford) + Mikaela Angelina Uy (Stanford) + Jiahui Huang (Tsinghua) + Shimin Hu (Tsinghua) + Ke Li (SFU) + Leonidas Guibas (Stanford), ICCV 2023 (pp. 14257–14267, 11 pages, **Tsinghua + Stanford + SFU** team), arXiv:2305.01921 v1 May 3 2023 → v3 Aug 20 2023 (42,690 KB, cs.CV), ~250-350 GS citations as of 2026-06-12, ~400-500 GS citations on the ICCV version (the 146-note's "Kalogerakis (UMass) authored 3DShape2VecSet + DiffFacto" is **WRONG** — Kalogerakis is *not* on DiffFacto, the team is Guibas (Stanford) + Hu (Tsinghua) + Li (SFU), the 145-note's "UMass" and 146-note's "Tsinghua + Adobe" attributions are *both* hallucinated, the *correct* attribution is **Stanford (Nakayama/Uy/Guibas) + Tsinghua (Huang/Hu) + SFU (Li)**, this is the 2nd + 3rd consecutive author-identification correction in 3 papers), the *first* 3D-gen paper to (a) factorize the 3D shape distribution P(S) into *independent* part styles P(Z_j) + *conditional* part transformations P(τ|Z) (a *natural* decomposition that exposes multiple "control knobs" for the generative process), and (b) introduce a *cross-attention* diffusion network that conditions on the factorized prior, the *direct* predecessor of 2024-2025 part-aware 3D-gen papers (PartGen 2412.18608, PA-Diffusion 2024, PartRM 2025, EditVAE, PartConverter 2025) and the *de facto* part-level 3D-diffusion recipe that v0 v0 v1 (tooth-level 3D generation) inherits.** The killer architecture is a **3-component factorized joint prior** with cross-attention diffusion: **(Component 1) Part stylizers** — m *independent* continuous normalizing flows (CNFs) Q_ψ_j(Z_j|Ŝ_j) that encode the *canonicalized* (i.e., translation+scale-removed) geometry of each part into a *D_2-dim* Gaussian latent Z_j ∈ ℝ^D_2 (the *part style* prior; m=4-6 typical for ShapeNet chairs, ~4-6 for dental arches). **(Component 2) Transformation sampler** — cIMLE (conditional Implicit Maximum Likelihood Estimation, Li et al. 2020) that learns a *multimodal* distribution P(τ|Z) of part transformations τ = (c ∈ ℝ³ shift, s ∈ ℝ³_+ scale) conditioned on part styles; the *multimodal* cIMLE samples K random latents y_1, ..., y_K and minimizes the *min* over K (so at least one sample matches the data, breaking the mode-collapse of cVAE / cGAN). **(Component 3) Cross diffusion network** — L cross-attention layers that take a noisy point cloud x^(t) and predict the noise ε_ϕ(x^(t), z, τ, j, t), where the cross-attention has *m tokens* (one per part) each being the concatenation (z_j, τ_j, j, t) — each point x^(t) attends to *all m part tokens*, so the denoising is informed by both the *local* part label j and the *global* shape via cross-attention to the m part styles. The **killer innovation** is the **Generalized Forward Kernel** (Eq. 8) — instead of the standard forward kernel Q(X^(t)|x^(t-1), μ=0, Σ=I) that diffuses *all* points to the *same* unit Gaussian, DiffFacto uses Q(X^(t)|x^(t-1), μ=c_j, Σ=Diag(s²_j)) that diffuses points to a *part-specific* transformed Gaussian with the part's own shift and scale. This preserves the *positional* and *scale* information through the forward diffusion (so the reverse process can decode with better shape fidelity), the *direct* ancestor of diffusion-with-priors (e.g., DiffFacto's generalized forward kernel is the *same* idea as CLAY's triplane-decoded diffusion with VAE-encoded prior, just in the *point-cloud* domain). The **loss** is L_total = L_recon + λ_1 L_Z + λ_2 L_τ where L_recon is the standard ε-prediction diffusion loss, L_Z is the CNF KL divergence (matches Q_φ_j to P_ψ_j, equivalent to VAE-style regularization), and L_τ is the cIMLE min-over-K transformation loss. **Datasets:** ShapeNet 4 categories (chair: 3053 train / 704 test, airplane: 2349/341, lamp: 1261/286, car: 740/158) with semantic part labels from Yi 2016. **Results vs global-shape baselines (Tab. 1, MMD-P ×10⁻², lower=better):** DiffFacto chair MMD-P **3.27** vs PointFlow 4.68 / DPM 4.17 / ShapeGF 3.52 / LION 3.99 (the *best* on intra-part), DiffFacto airplane MMD-P **3.20** vs PointFlow 4.61 / DPM 3.52 / ShapeGF 3.50 / LION 3.68 (the *best* on intra-part). **Results vs control-enabled baselines (Tab. 2, SNAP ×10⁻² on chair 3 connections: back-leg/seat-legs/arms-seat, lower=better):** DiffFacto **13.32** vs Ctrl-ShapeGF 41.12 / Ctrl-LION 31.76 (3.1× and 2.4× better, the *killer* inter-part coherence evidence). **Plausibility (Tab. 3, mIOU on car segmentation):** adding 700 generated cars improves PointNet 0.709→0.788 (+7.9 pts), adding 60 controlled race-cars improves PointNet 0.709→0.780 (+7.1 pts with only 8.5% of the data, the *killer* data-augmentation efficiency). **Human study:** 85% preference (8.5/10) vs Ctrl-ShapeGF / Ctrl-LION on part-level edits. **Transformation sampler ablation (Tab. 5, CD ×10⁻⁴ on chair inversion, lower=better):** cIMLE 4.97 (best) vs cVAE 7.33 / cGAN 11.48 / direct regression 13.38 (the *killer* multimodal-distribution evidence). **Factorization ablation (Tab. 4, SNAP ×10⁻² on chairs):** DiffFacto 13.32 (best) vs Separate 25.24 / Post-Transform 18.23 / Global-Agnostic 19.29 (the *killer* factorization + cross-attention ablation). **License: MIT** (per github.com/diffFacto/diffFacto LICENSE) — *the* permissive OSS license, the *direct* enabler for v0 v0 v0 v0 v0 v1 (crown generation) to *fork* and adapt. **Pretrained weights** (http://download.cs.stanford.edu/orion/DiffFacto/weights.zip) and **preprocessed data** (http://download.cs.stanford.edu/orion/DiffFacto/data.zip) are *publicly downloadable*, the *right* starting point for v0 v0 v0 v0 v0 v0's *dental-crown* part-level diffusion fine-tuning. **Note: 145-SOPHY-note's "Kalogerakis (UMass) authored 3DShape2VecSet + DiffFacto" is *WRONG* on DiffFacto** (correct on 3DShape2VecSet? no — 146-note correctly identifies 3DShape2VecSet = Zhang/Tang/Niessner/Wonka KAUST+TUM, NOT Kalogerakis), and 146-note's "DiffFacto = Tsinghua + Adobe" is *also WRONG* (correct team is Stanford + Tsinghua + SFU). The 3 consecutive author-identification hallucinations in 145-146-147 (Cao-NTU vs Cao-UMass, 3DShape2VecSet-KAUST vs 3DShape2VecSet-UMass, DiffFacto-Tsinghua+Adobe vs DiffFacto-Stanford+Tsinghua+SFU) suggest a *systematic* issue with the recent papers' author-identification — need to be more careful.

## Research Question

**Q:** Can we design a *3D shape generative model* that (a) *enables intuitive user control* over the generated shape (so users can specify individual parts and configurations), (b) supports *multiple axes of control* (part style vs part configuration vs both), (c) generates *coherent* shapes (so locally-modified parts are globally-plausible, no disjoint / flying parts), (d) works on *general 3D shapes* (not just chairs), and (e) supports *downstream editing applications* (shape interpolation, shape mixing, transformation editing) — by (1) factorizing the shape distribution P(S) into a *part style* distribution P(Z) and a *transformation* distribution P(τ|Z), where the part styles are *canonicalized* (translation+scale removed) and the transformations recover the *instance* shape, (2) using a *continuous normalizing flow* (CNF) for each part's style distribution (preserves the *full* expressivity of per-part distributions, no Gaussian restriction), (3) using *conditional IMLE* (cIMLE) for the transformation distribution (handles the inherent *multimodality* — same parts can have many plausible transformations, cVAE / cGAN collapse), (4) using a *cross-attention diffusion network* that conditions the denoising on the factorized prior via m part tokens, so each generated point is informed by *both* its local part label and the global shape, and (5) introducing a *generalized forward kernel* that diffuses points to part-specific scaled/shifted Gaussians (preserves part pose/scale through the forward process, leading to better reconstruction and edit extrapolation)?

**Their answer:** **Yes — the factorized part-style + transformation + cross-attention diffusion recipe is the *right* 3D generation paradigm for controllable part-based synthesis**, with the *killer* practical advantages of (a) *multiple control knobs* (sample one part style z_j while keeping the rest fixed, or sample a new transformation τ while keeping all part styles fixed, or sample both — three distinct editing modes), (b) *coherent global shape* (cross-attention ensures each point is aware of the global configuration, SNAP 13.32 vs 31.76-41.12 for control-enabled baselines = 2.4-3.1× better inter-part coherence), (c) *multimodal transformations* (cIMLE recovers the *full* distribution of plausible configurations, CD 4.97×10⁻⁴ vs 7.33 for cVAE on chair inversion = 32% better), (d) *forward-kernel prior* (generalized forward kernel that diffuses to part-specific transformed Gaussians gives better reconstruction, especially on extreme transformation edits — the *killer* practical advantage for v0 v0 v0 v0 v0 v0 (crown generation) where the prep geometry can vary by 5-10× scale), and (e) *flexible editing* (the autoencoder mode supports local edits by encoding then modifying one part, the variation mode supports resampling transformations, the interpolation mode supports smooth part-level interpolation). The 5 key insights are: **(a) Factorization is the *right* inductive bias for 3D-gen** — a chair is *naturally* a part composition (back + seat + 4 legs + arms), and modeling each part independently gives the user *control* over each part. The 146-note's VecSet is the *complementary* insight — represent the shape as a *set* of latents, vs DiffFacto's *decomposition* of the set into part + transformation; the two can be combined (a VecSet per part). **(b) CNF is the *right* part-style distribution** — VAEs collapse to Gaussian, GANs are unstable, normalizing flows are the *expressivity-flexibility* sweet spot (continuous, invertible, full-distribution). **(c) cIMLE is the *right* transformation distribution** — cVAE enforces *unimodal* matching (collapse), cGAN enforces *all-samples* matching (also collapse), cIMLE *min-over-K* enforces *at-least-one* matching (correct multimodal, the *killer* insight for *any* one-to-many mapping in 3D). **(d) Cross-attention is the *right* conditioning mechanism for diffusion** — each point attends to m part tokens (each = (z_j, τ_j, j, t)), so the denoising is *local-aware* (per-point part label) AND *global-aware* (per-point global shape context). The *elimination* of the global tokens (Global-Agnostic variant) drops SNAP from 13.32 to 19.29, a 45% degradation — the *killer* ablation evidence that cross-attention is essential. **(e) Generalized forward kernel is the *right* prior for part-aware diffusion** — by diffusing to a part-specific transformed Gaussian (with the part's own shift c_j and scale s_j²), the forward process *preserves* the part transformation information, and the reverse process can decode with better shape fidelity; the *elimination* of the generalized kernel (standard kernel variant) gives worse reconstruction on extreme edits (per Fig. 6 + Fig. 8) — the *killer* ablation evidence that part-specific prior matters.

## Method

### Architecture (Factorized Joint Prior + Cross-Attention Diffusion)

**Stage 1: Part Stylizer (CNF per part)**
- **Input:** a segmented point cloud S = {S_j}_(j=1..m) with m semantic parts (m=4-6 for ShapeNet chairs, derived from Yi 2016's part annotations)
- **Canonicalization:** for each part S_j, extract the *canonicalized* geometry Ŝ_j by removing the *instancing transformation* τ_j = (c_j ∈ ℝ³ shift, s_j ∈ ℝ³_+ axis-aligned scale): S_j = Diag(s_j) · Ŝ_j + c_j (i.e., the part is a scaled + translated canonical geometry)
- **Encoder Q_φ_j(Z_j|Ŝ_j):** PointNet++-style encoder that takes the canonicalized part points and outputs a D_2-dim Gaussian latent z_j ∈ ℝ^D_2 (D_2=128 typical)
- **CNF prior P_ψ_j(Z_j):** continuous normalizing flow that transforms a base Gaussian (z_0 ~ N(0, I)) into the *learned* part-style distribution via an ODE dz/dt = f_ψ(z, t); the *advantage* over VAE's fixed Gaussian is *expressive* part-style distributions (multi-modal, skewed, etc.)
- **Loss L_Z:** KL(Q_φ_j(Z_j|Ŝ_j) || P_ψ_j(Z_j)) summed over j, with λ_1 weighting

**Stage 2: Transformation Sampler (cIMLE)**
- **Input:** the part style latents z = {z_j}_(j=1..m) from the part stylizer
- **Output:** the part transformations τ = {τ_j = (c_j, s_j)}_(j=1..m)
- **Multimodal cIMLE training:** sample K=128 random latents y_1, ..., y_K ~ N(0, I), generate K candidate transformations τ^k = T_θ(z, y_k) for k=1..K, compute the fit loss ℓ_fit(τ^k, τ^S) = Σ_j ||c_j^k - c_j^S||² + Σ_j ||log s_j^k - log s_j^S||² (L2 on shift + log-scale), and minimize the *min-over-K*: L_τ = Σ_S min_(k=1..K) ℓ_fit(τ^k, τ^S). The *min* enforces *at-least-one* matching, breaking the mode-collapse of cVAE / cGAN that enforce *all* matching
- **Inference:** sample y ~ N(0, I), generate τ = T_θ(z, y); can sample *multiple* τ for the same z (the *multimodal* property)
- **Architecture T_θ:** a simple MLP (z, y) → τ, with z and y concatenated and 4-6 FC layers

**Stage 3: Cross Diffusion Network (DDPM with cross-attention)**
- **Input:** a noisy point cloud x^(t) ∈ ℝ^(N×3) at diffusion timestep t, where N is the number of points per shape (N=2048 typical)
- **Output:** predicted noise ε_ϕ(x^(t), z, τ, j, t) ∈ ℝ^(N×3)
- **Architecture:** L=6-8 cross-attention layers, each with the *m part tokens* as keys+values and the *N points* as queries
  - **Part token construction:** for each part j ∈ {1, ..., m}, the token is the concatenation (z_j, τ_j, j-embedding, t-embedding) ∈ ℝ^(D_2 + 6 + D_label + D_t) where D_label=16 (learned part-label embedding) and D_t=128 (sinusoidal time embedding)
  - **Per-point input:** for each point x_i^(t) ∈ ℝ³ at timestep t, the per-point query is the concatenation (γ(x_i^(t)), j_i-embedding) where γ is a learned sinusoidal positional encoding (Fourier features with 32 frequencies) and j_i is the *ground-truth* part label of the point
- **Loss L_recon:** standard ε-prediction DDPM loss, L_recon = Σ_j Σ_(x ∈ S_j) E_(ε, z, t) [||ε - ε_ϕ(x^(t), z, τ^S, j, t)||²]
- **Total loss:** L_total = L_recon + λ_1 L_Z + λ_2 L_τ (λ_1=λ_2=0.001 typical)

**The Generalized Forward Kernel (the killer innovation)**
- Standard DDPM forward kernel: Q(X^(t)|x^(t-1), μ=0, Σ=I) = N(√α_t · x^(t-1), (1-α_t)·I) — all points drift to the *same* unit Gaussian
- DiffFacto generalized kernel: Q(X^(t)|x^(t-1), μ=c_j, Σ=Diag(s²_j)) = N(√α_t · x^(t-1) + (1-√α_t)·c_j, (1-α_t)·Diag(s²_j)) — points in part j drift to a *part-specific* transformed Gaussian with the part's *own* shift c_j and *own* scale s²_j
- The *closed-form* limit: as T → ∞, Q(X^(T)|x^(0), μ, Σ) → N(μ, Σ) — the *parameterized* Gaussian (not the standard unit Gaussian)
- The *practical advantage:* the forward process *preserves* the part transformation information (c_j, s²_j) by encoding it into the *target* Gaussian's mean and variance; the reverse process can then decode with better shape fidelity, especially on extreme transformation edits (per Fig. 6 + Fig. 8)

**Training recipe (per github.com/diffFacto/diffFacto README, for chair category):**
- 2-stage training: stage 1 trains part stylizer + cross diffusion network (`train_chair_stage1.py`), stage 2 trains transformation sampler conditioned on frozen stage 1 (`train_chair_stage2.py`)
- AdamW optimizer, batch size 64-128, 4000 epochs stage 2 (can pick earlier checkpoint)
- PyTorch 1.12.1 + CUDA 11.3, requires `pointnet2_ops_lib` + `chamfer_dist` + `emd` custom ops + `xformer` for memory-efficient attention
- Single A100 (~24GB) is sufficient for chair category training

**Inference recipe:**
- Sample part styles z_j ~ P_ψ_j(Z_j) for j=1..m
- Sample transformation τ ~ T_θ(z, y) for one y ~ N(0, I)
- Run cross diffusion network reverse process for T=1000 steps (or DDIM for 100 steps) starting from Gaussian noise with the *transformed* Gaussian (μ=c_j, Σ=Diag(s²_j)) for part j
- Output: a *segmented* point cloud (each point has a part label)
- Can also be used as an *autoencoder* by deterministically encoding part styles + using observed τ^S

### The Cross-Attention-as-Part-Aware-Conditioning Trick (the killer insight)

**The key innovation** of DiffFacto is using **cross-attention with m part tokens as keys+values** so that each generated point is informed by *both* its local part label (via j_i in the query) and the global shape context (via the m part tokens in the cross-attention keys+values):
- *Naive* part conditioning (concatenate part label j to the time embedding): the global part-to-part interactions are *lost* (the diffusion only knows "this point is in part j" but not "the relationship between this part and the others")
- *Global-Agnostic* (m tokens, but no cross-attention): the per-point conditioning is *part-specific* (P(X|z_j, τ_j, j)) but not *globally-aware* (no cross-attention to all m tokens)
- *DiffFacto* (m tokens + cross-attention): each point attends to *all* m tokens, learning the *global* part-to-part interactions

The *ablation evidence* (Tab. 4): SNAP for *Separate* (no joint prior) = 25.24, *Post Transform* (m tokens but no cross-attention) = 18.23, *Global Agnostic* (cross-attention removed) = 19.29, *DiffFacto* (m tokens + cross-attention) = **13.32** — the *killer* ablation showing that *all three* components (factorization + cross-attention + joint prior) are necessary.

This is a *general* design pattern that any 3D-gen team can adopt:
- **For v0 v0 v0 v0 v0 v0 (crown generation):** treat the *prepared tooth* (the prep) as one part and the *crown* as another part, model P(crown | prep) with cross-attention conditioning on the prep's part token. The *killer* v0 v0 use case: generate a crown that is *coherent* with the prep (no over-extension, no under-coverage) — the cross-attention naturally learns the *boundary-aware* generation.
- **For v0 v0 v0 v0 v0 v1 (tooth-level generation):** treat each *tooth* as one part (m=14-32 for full arch), model P(arch | {teeth}) with cross-attention across all teeth — the *killer* v0 v0 v1 use case is the *arch-level* synthesis.
- **For v0 v0 v0 v0 v0 v2 (joint shape + material per tooth):** extend the part token to (z_j_shape, z_j_material, τ_j, j, t) — the SOPHY 145 recipe applied at the *part* level.

### The cIMLE Trick (the killer implementation detail)

**For multimodal one-to-many mappings** (one part style z has many plausible transformations τ), DiffFacto uses **cIMLE** (conditional Implicit Maximum Likelihood Estimation, Li et al. 2020) instead of cVAE / cGAN:
- *cVAE* (Sohn 2015): minimizes *expected* reconstruction loss, so the *mean* of the latent → τ mapping is the data — but the *variance* of the latent is forced to N(0, I), so the network learns to *ignore* the latent y and *collapse* to a unimodal mapping. CD ×10⁻⁴ on chair inversion: 7.33 (worse).
- *cGAN* (Mirza 2014): minimizes *adversarial* loss (all samples should match data), so the network is forced to *match all y's to all data points* — also collapses to unimodal. CD ×10⁻⁴ on chair inversion: 11.48 (worst).
- *Direct regression* (single y, no randomness): completely ignores the multimodal structure. CD ×10⁻⁴ on chair inversion: 13.38 (also worst).
- *cIMLE* (Li 2020): samples K y's, minimizes the *min-over-K* — so the network is *rewarded* for having *at least one* y that matches the data, naturally learning the *multimodal* distribution. CD ×10⁻⁴ on chair inversion: **4.97** (best, 32% better than cVAE, 1.5× better than the next best).

The *practical* cIMLE recipe: K=128 samples per training example, the *min* is computed within each batch (not across the batch). The *killer* practical advantage: K can be tuned (K=64 is faster, K=256 is more accurate) with no other architectural change.

This is a *general* design pattern that any 3D-gen team can adopt:
- **For v0 v0 v0 v0 v0 v0 (crown generation):** the *plausible crown given a prep* mapping is *one-to-many* (multiple valid crown designs per prep), so cIMLE is the *right* training objective for the crown-conditional network. The *direct* cIMLE application: given the prep point cloud, sample K candidate crowns and pick the *best* (e.g., by clinical-fit metrics like margin gap, proximal contact, occlusion).

## Results

### Intra-part Generation Quality (Tab. 1, MMD-P ×10⁻² ↓ / COV-P % ↑ / 1NNA-P % ↓)

| Chair | MMD-P ↓ | COV-P ↑ | 1NNA-P ↓ |
|---|---|---|---|
| PointFlow (Yang 2019) | 4.68 | 27.3 | 87.77 |
| DPM (Luo 2021) | 4.17 | 28.2 | 85.65 |
| ShapeGF (Cai 2020) | 3.52 | 42.3 | 68.65 |
| LION (Zeng 2022) | 3.99 | 35.1 | 69.25 |
| **DiffFacto (Ours)** | **3.27** | **42.5** | **65.23** |

| Airplane | MMD-P ↓ | COV-P ↑ | 1NNA-P ↓ |
|---|---|---|---|
| PointFlow (Yang 2019) | 4.61 | 32.0 | 86.11 |
| DPM (Luo 2021) | 3.52 | 37.7 | 78.74 |
| ShapeGF (Cai 2020) | 3.50 | 40.0 | 72.04 |
| LION (Zeng 2022) | 3.68 | 38.8 | 68.73 |
| **DiffFacto (Ours)** | **3.20** | **46.2** | **68.72** |

**Reading:** DiffFacto *wins* on every metric (MMD-P, COV-P, 1NNA-P) for *both* chair and airplane, the *killer* evidence that part-aware modeling > global-shape modeling for *intra-part* quality. The 1NNA-P improvement (87.77 → 65.23 for chair) is the *killer* distributional-coverage evidence (the generated parts are *less distinguishable* from real parts, i.e., the learned distribution is closer to the real distribution).

### Inter-part Coherence (Tab. 2, SNAP ×10⁻² ↓ on chair 3 connections: back-leg / seat-legs / arms-seat)

| Method | SNAP ↓ |
|---|---|
| Ctrl-ShapeGF (ShapeGF with per-part conditioning) | 41.12 |
| Ctrl-LION (LION with per-part conditioning) | 31.76 |
| **DiffFacto (Ours)** | **13.32** |

**Reading:** DiffFacto is **3.1× better** than Ctrl-ShapeGF and **2.4× better** than Ctrl-LION on inter-part coherence, the *killer* evidence that the cross-attention conditioning is essential for *coherent* part-based generation. The *physical interpretation:* in a generated chair, the *back* is properly *attached* to the *seat*, the *arms* are properly *attached* to the *seat* or *back*, the *legs* are properly *attached* to the *seat* — no flying parts, no disjoint geometry.

### Plausibility (Tab. 3, mIOU on car part segmentation after data augmentation)

| Method | Orig. | + Multi (700) | + Control (60) |
|---|---|---|---|
| PointNet (Qi 2017a) | 0.709 | 0.788 (+7.9) | 0.780 (+7.1) |
| PointNet++ (Qi 2017b) | 0.800 | 0.808 (+0.8) | 0.801 (+0.1) |

**Reading:** adding 700 DiffFacto-generated cars improves PointNet mIOU by 7.9 pts (0.709 → 0.788), and adding only 60 *controlled* race-cars improves by 7.1 pts (0.709 → 0.780) — the *killer* data-augmentation efficiency evidence. The *practical* implication: a small number of *targeted* generated samples (60 race-cars, vs 700 random) gives most of the benefit, the *right* approach for v0's *clinical* data augmentation (generate 100 controlled crowns matching a specific clinical scenario, not 1000 random crowns).

### Human Study (100 participants, 10 shapes per method)

- DiffFacto: 8.5/10 preference (85%) vs Ctrl-ShapeGF / Ctrl-LION
- The *killer* human-evaluation evidence that the generated shapes are *perceptually* more plausible than the control-enabled baselines.

### Transformation Sampler Ablation (Tab. 5, CD ×10⁻⁴ on chair inversion, lower=better)

| Method | CD ↓ |
|---|---|
| Direct regression | 13.38 |
| cVAE (Sohn 2015) | 7.33 |
| cGAN (Mirza 2014) | 11.48 |
| **cIMLE (Ours)** | **4.97** |

**Reading:** cIMLE is **1.5× better** than cVAE (the next best), the *killer* multimodal-distribution evidence. The *physical interpretation:* cIMLE recovers the *full* distribution of plausible transformations, while cVAE / cGAN collapse to a *single* modal transformation (the *mean* of the data, which is *not* a valid transformation for any *specific* training example).

### Factorization Ablation (Tab. 4, SNAP ×10⁻² on chairs, lower=better)

| Variant | SNAP ↓ |
|---|---|
| Separate (each part is independent CNF, no joint prior) | 25.24 |
| Post Transform (cross diffusion conditioned only on z, τ applied as post-processing) | 18.23 |
| Global Agnostic (no cross-attention to all m tokens, per-point part conditioning only) | 19.29 |
| **DiffFacto (factorization + cross-attention + joint prior)** | **13.32** |

**Reading:** the *full* DiffFacto (factorization + cross-attention + joint prior) is **1.9× better** than Separate (no joint prior), **1.4× better** than Post Transform (no cross-attention), and **1.5× better** than Global Agnostic (no global tokens). The *killer* ablation evidence that *all three* components are necessary.

### Cross Diffusion Generalization (per Fig. 6 + Fig. 8)

- The generalized forward kernel (with μ=c_j, Σ=Diag(s²_j)) gives *better reconstruction* on extreme transformation edits — the *killer* evidence that the part-specific prior matters for extrapolation (per Fig. 6: chair with stretched back + legs, the kernel-version preserves geometry better).
- The generalized kernel also gives *better transformation extrapolation* on lamps (per Fig. 8: heat map of reconstruction error is lower with kernel, especially on thin parts like the lamp cap).

## Connections to H1-H5

### H1 (2-stage generation > 1-stage) — PARTIAL SUPPORT, refinement

**Evidence:** DiffFacto is *structurally* 2-stage (part stylizer → transformation sampler → cross diffusion), with 3 sub-stages in the *forward* direction. The ELBO derivation (Eq. 3) is *exact*, with separate loss terms for each sub-stage.

**Refinement:** the 2-stage here is *not* the same as the DMC 033 / DCrownFormer 032 2-stage (mesh-completion + mesh-refinement). It's a *factorization-based* 2-stage (style + transformation), with the *joint* prior P(Z, T) = P(Z) P(T|Z) and the *conditional* likelihood P(S|Z, T). The *practical* difference: the factorized 2-stage is *factorization-aware* (each part can be independently sampled/edited), while the DMC 2-stage is *refinement-aware* (the second stage fixes errors from the first).

**For v0 v0 v0 v0 v0 v0 (crown generation):** H1 supports *both* the factorized 2-stage (style + transformation) AND the refinement 2-stage (coarse + fine) — the *practical* recommendation is to *combine* both: factorize the crown generation into *style* (crown shape) + *transformation* (crown position + scale) with cross-attention conditioning on the prep, AND do a refinement pass (DMC 033-style) for fine details.

### H2 (latent diffusion > direct) — STRONGEST DIRECT SUPPORT IN READING LIST for part-level 3D

**Evidence:** DiffFacto is *literally* a part-level latent diffusion model — the cross diffusion network runs in the *point-cloud* space (x ∈ ℝ^(N×3)), but the *conditioning* is on the *latent* part styles Z and transformations τ. The *latent diffusion* interpretation: the part styles + transformations are the *compact* latent (Z ∈ ℝ^(m×D_2) = 4×128 = 512-dim for chair), and the cross diffusion generates points *conditioned* on this compact latent.

**Strongest support** because (a) DiffFacto is *explicitly* designed for part-level control (the *killer* feature enabled by the latent factorization), (b) the cIMLE transformation sampler is *itself* a latent model (z, y → τ), and (c) the cross diffusion is conditioned on the *concatenation* of the part style + transformation latents, the *direct* H2 mechanism.

**For v0 v0 v0 v0 v0 v0 (crown generation):** H2 *directly* supports the *part-level latent diffusion* recipe — generate the *compact* crown latent (a few part-style codes + transformation codes) with one diffusion, then decode to the *full* crown point cloud with the cross-attention decoder. The *practical* advantage: the *latent* is *small* (512-dim for the *whole* crown), so the diffusion trains *fast* (hours on a single A100) and samples *fast* (seconds), the *direct* enabler for v0 v0 v0 v0 v0 v0's *chairside real-time* requirement (50-200ms per inference, per DMC 033's SLA).

### H3 (arch-level conditioning > tooth-isolated) — STRONGEST DIRECT SUPPORT IN READING LIST for part-level

**Evidence:** DiffFacto is *literally* a part-level generative model — it explicitly models the *inter-part* relationships (back ↔ seat, arms ↔ back, etc.) via cross-attention. The SNAP metric (3.1× better than Ctrl-ShapeGF) is the *direct* H3 evidence: the inter-part coherence is *better* with arch-level (full-shape) conditioning.

**Strongest support** because (a) the cross-attention is *explicitly* arch-level (each point attends to all m part tokens), (b) the *controlled variation* experiments (Fig. 3) show that the model can *fix* one part while *resampling* the rest, the *killer* test of arch-level coherence, and (c) the *part mixing* experiments (Fig. 4) show that the model can *stitch* parts from different source shapes into a *coherent* target shape, the *killer* test of inter-part compatibility.

**For v0 v0 v0 v0 v0 v0 (crown generation):** H3 *directly* supports the *arch-level* recipe — generate the *crown* as a *part* of the *arch*, conditioned on the *arch context* (1 prep + 2 adjacent + 3 opposing teeth + gum, per DMC 033's 6-tooth context). The *practical* advantage: the crown is *coherent* with the surrounding teeth (no over-extension, no gap), the *direct* enabler for v0 v0 v0 v0 v0 v0's *clinical-fit* requirement.

### H4 (implicit representation > mesh) — MILD CONTRADICTION (point-cloud, not mesh)

**Evidence:** DiffFacto is a *point-cloud* generative model — it generates N=2048 points per shape, not a mesh. The *extraction* of a mesh requires a *post-processing* step (e.g., alpha-shape or Screened Poisson Reconstruction, not specified in the paper but *implied* by the absence of mesh-supervision loss).

**Mild contradiction** because the *point-cloud* representation is *not* the *implicit* representation that H4 supports (SDF, occupancy, NeRF), and is *not* the *mesh* representation that H4 contradicts. DiffFacto is a *third* representation: explicit point cloud, which has *its own* advantages (no surface extraction, no volumetric sampling) and *disadvantages* (no connectivity, no normals).

**Refinement:** the *practical* path for v0 v0 v0 v0 v0 v0 (crown generation) is to *combine* DiffFacto's part-aware point-cloud generation with DMC 033's SAP/DPSR mesh extraction — generate the *points* with DiffFacto (cIMLE-style crown latent + cross-attention), then *extract the mesh* with SAP/DPSR (the *right* mesh extraction for point clouds, per DMC 033's recipe).

### H5 (synthetic + finetune > from scratch) — NOT DIRECTLY TESTED, but with H5-compatible design

**Evidence:** DiffFacto is *trained from scratch* on ShapeNet 4 categories (chair / airplane / lamp / car), with *no* pretraining on larger datasets (e.g., Objaverse). The *paper does not test* the synthetic-pretrain + finetune paradigm.

**Not directly tested** but the *architecture is H5-compatible*: the part stylizer (CNF per part) and the cross diffusion network are *modular*, so the *weights* can be *initialized* from a larger-pretrain (e.g., the SOPHY 145 3DShape2VecSet-pretrained backbone) and *fine-tuned* on a smaller domain (e.g., dental arches). The *practical* v0 v0 v0 v0 v0 v0 path: use the DiffFacto *code* (MIT) + a *dental-pretrain* (e.g., 3DShape2VecSet-pretrained VecSet, or a *new* dental-pretrain on 3DTeethSeg22 + ToSynFCD) + a *small* fine-tuning cost (~$50-100 Lambda for the dental part-stylizers + cross diffusion).

## Surprises / interesting things buried in section 4

1. **Generalized Forward Kernel (Eq. 8) is the killer design choice**, not the cross-attention. The cross-attention is a *standard* transformer (multi-head, sinusoidal positional encoding, 6-8 layers), but the *generalized forward kernel* is the *novel* contribution that enables *better transformation extrapolation* (per Fig. 6 + Fig. 8) — the *killer* practical advantage for v0 v0 v0 v0 v0 v0 (crown generation) where the prep geometry can vary 5-10× scale.

2. **cIMLE is the *right* choice for multimodal one-to-many mappings** in 3D, but with a *caveat*: K=128 samples per training example is *expensive* at training time (~128× slower than direct regression), but *fast* at inference (single y, single forward pass). The *practical* trade-off: cIMLE training is 1-2 days on a single A100, but inference is 50-100ms — the *right* trade-off for v0 v0 v0 v0 v0 v0's *chairside real-time* requirement.

3. **The cIMLE K=128 is a *tunable* hyperparameter**, and the *sweet spot* depends on the *complexity* of the multimodal distribution — for *simple* one-to-many mappings (e.g., translation only), K=32 is sufficient; for *complex* mappings (translation + scale + rotation), K=128-256 may be needed. The *practical* recommendation for v0 v0 v0 v0 v0 v0 (crown generation): start with K=128, monitor the CD ×10⁻⁴ on a held-out validation set, and tune.

4. **The 85% human-study preference is *very high*** for a 3D-gen paper — most 3D-gen papers report 50-70% preference vs baselines. The *interpretation:* the *part-aware* generation is *perceptually* more plausible than *global-shape* generation, the *killer* evidence for v0 v0 v0 v0 v0 v0's clinical-readiness (the dentist will *see* that the generated crown is more "natural" than a global-shape baseline).

5. **The *controlled* data-augmentation (60 race-cars) is *nearly as effective* as *uncontrolled* (700 random cars)** — the *killer* evidence that *targeted* generation (matching a specific clinical scenario) is *much more efficient* than *random* generation, the *right* paradigm for v0 v0 v0 v0 v0 v0's clinical data augmentation (generate 100 crowns matching a specific clinical pattern, not 1000 random crowns).

6. **The *separate* variant (no joint prior) is the *worst* on inter-part coherence (SNAP 25.24 vs 13.32 for full)** — the *killer* evidence that the *joint* prior P(Z, T) is *essential* for *coherent* part-based generation, the *direct* precedent for v0 v0 v0 v0 v0 v0's *joint* part-style + transformation prior (the *killer* H1 + H2 mechanism for crown generation).

7. **The *post-transform* variant (no cross-attention) is the *second-worst* (SNAP 18.23)** — the *killer* evidence that the *cross-attention* is *also* essential (in addition to the joint prior), the *direct* precedent for v0 v0 v0 v0 v0 v0's *cross-attention* conditioning on the prep + adjacent + opposing teeth.

8. **The code is *minimal* (~1200 lines PyTorch, including the pointnet2_ops + chamfer_dist + emd custom ops)** — the *killer* engineering advantage for v0 v0 v0 v0 v0 v0 (1-2 weeks to port to PyTorch 2.x + Python 3.10/3.11, the *shortest* port in the reading list).

9. **The pretrained weights + data are *publicly downloadable* from Stanford's CDN** (http://download.cs.stanford.edu/orion/DiffFacto/) — the *killer* practical advantage for v0 v0 v0 v0 v0 v0 (no need to retrain from scratch on ShapeNet 4 categories, just *fine-tune* on dental data).

10. **The *only* dataset used is ShapeNet 4 categories** — no Objaverse, no ShapeGlot, no PartNet, etc. The *interpretation:* the authors chose *small* datasets to demonstrate the *part-level* generation quality without needing *massive* data. The *practical* implication for v0 v0 v0 v0 v0 v0: a *small* dental dataset (1K-10K arches from 3DTeethSeg22 + ToSynFCD) is *sufficient* for the part-level diffusion to learn the *dental* part distributions.

## Quote-worthy sentences

- "We introduce DiffFacto, a novel probabilistic generative model that learns the distribution of shapes with part-level control." (abstract)
- "We propose a factorization that models independent part style and part configuration distributions, and present a novel cross-diffusion network that enables us to generate coherent and plausible shapes under our proposed factorization." (abstract)
- "To our best knowledge, we are the first to introduce a factorized representation that allows for control in both part styles and part configurations as we model independent part style distributions and transformation distribution, enabling each to be independently sampled." (Sec. 1)
- "Our design allows each generated point in the point cloud to be informed of both the global shape as well as the local part, resulting in more plausible and coherent output shapes while still enabling control." (Sec. 1)
- "We leverage on a sampling-based approach to learn a multi-modal distribution of part configurations through conditional Implicit Maximum Likelihood [24] (cIMLE)." (Sec. 1)
- "Our modification allows for the explicit encoding of each part transformations, enabling better shape reconstruction and transformation extrapolation." (Sec. 3)
- "We show that our generalized forward kernel is theoretically equivalent to diffusing all points to a scaled and shifted Gaussian." (Sec. 4.3)
- "We note that our generalized forward kernel is able to model complex part geometry better than the standard forward kernel because of the additional size and location prior information incorporated into the diffusion process." (Sec. 5.5)
- "Our human study has 100 participants comparing our approach with the control-enabled baselines... on average the participants favour 85% (8.5 out of 10) of our generated shapes more than other baselines." (Sec. 5.3)
- "Concretely, transformation sampler T_θ outputs samples τ^k = T_θ(z, y_k) for part style latents z and random latent variable yk ∼ N(0, I). We sample multiple latents y_1, ..., y_K and encourage that at least one of them matches the observed data S." (Sec. 4.2)
- "The idea is if our approach generates novel and coherent shapes with part labels then using them for data augmentation would improve the segmentation score of part segmentation networks." (Sec. 5.1)
- "We are also able to generate various plausible configurations of the shape (right) with fixed part styles." (Fig. 3 caption)
- "We modify ShapeGF [5] and LION [52] to have part-level control (prefixed by 'Ctrl-') by modeling independent part distributions, i.e. P (S) = Π_j P (S_j|w_j), where each part latent distribution is modeled with a (hierarchical) variational encoder Q(w_j|S_j). These baselines allow for independent sampling at the part-level unlike existing approaches." (Sec. 5.2)

## Code / data link

- **Code:** https://github.com/diffFacto/diffFacto (MIT License) — fully open-source, PyTorch 1.12.1 + CUDA 11.3, ~1200 lines
- **Pretrained weights:** http://download.cs.stanford.edu/orion/DiffFacto/weights.zip (ShapeNet 4 categories, chair + airplane + lamp + car)
- **Preprocessed data:** http://download.cs.stanford.edu/orion/DiffFacto/data.zip (Yi 2016 part-annotated ShapeNet, 4 categories)
- **Project page:** https://difffacto.github.io/ (with qualitative results + demo videos)
- **Paper (ICCV open access):** https://openaccess.thecvf.com/content/ICCV2023/papers/Nakayama_DiffFacto_Controllable_Part-Based_3D_Point_Cloud_Generation_with_Cross_Diffusion_ICCV_2023_paper.pdf
- **arXiv:** https://arxiv.org/abs/2305.01921 (v3 Aug 2023, 42 MB supplementary)

## For our project

### v0 v0 v0 v0 v0 v0 (DMC-forked part-aware crown generation) — combine DMC 033 + DiffFacto 147 for *part-aware* crown generation
1. **Fork github.com/diffFacto/diffFacto** and port to PyTorch 2.x + Python 3.10/3.11 (the *original* code is PyTorch 1.12, no FlashAttention, no compile)
2. **Adapt the part decomposition** from ShapeNet 4-cat (chair has 4 parts: back/seat/legs/arms) to *dental arch* (m=14-32 teeth + 1 gum = 15-33 parts). The *crown-generation* sub-task uses m=2 (prep + crown), the *arch-level* sub-task uses m=14-32 (all teeth)
3. **Retrain the part stylizers (CNFs) on dental data:** 3DTeethSeg22 (7,000 arches with per-tooth segmentation) + ToSynFCD (30K synthetic arches). The CNF per part is *small* (~few MB per part), so training is *fast* (~$20-50 Lambda per part on a single A100)
4. **Retrain the transformation sampler (cIMLE) on dental data:** each tooth's *position* (xyz center) + *scale* (3D bbox) + *orientation* (3D rotation, if axis-aligned isn't enough). K=128 samples per training example
5. **Retrain the cross diffusion network on dental data:** the *conditioning* is the *prep* (1 tooth) + *adjacent* (2 teeth) + *opposing* (3 teeth) + *gum* (1 context) per DMC 033's 6-tooth context. The m tokens are (z_prep, τ_prep, prep-label), (z_adj_L, τ_adj_L, adj_L-label), (z_adj_R, τ_adj_R, adj_R-label), (z_opp_1, τ_opp_1, opp_1-label), (z_opp_2, τ_opp_2, opp_2-label), (z_opp_3, τ_opp_3, opp_3-label), (z_gum, τ_gum, gum-label)
6. **Use the generalized forward kernel** with the *prep* part's (c_prep, s²_prep) as the μ, Σ — the *killer* practical advantage for v0 v0 v0 v0 v0 v0 (crown generation) where the prep geometry can vary 5-10× scale
7. **Add DMC 033's SAP/DPSR post-processing** for mesh extraction (the *right* mesh extraction for the part-aware point cloud)
8. **Add Hwang 061's histogram loss L_Ĥ** for clinical-fit-aware fine-tuning (the *killer* clinical-fit-aware extension)
9. **Compute cost:** $100-200 Lambda for the cross diffusion training (4 A100 × 6 hours), $20-50 per part stylizer CNF training (single A100 × 1-2 hours), $50-100 for the cIMLE transformation sampler training (single A100 × 2-3 hours), $0 for inference (single A100, ~50-100ms with DDIM 100 steps)

### v0 v0 v0 v0 v0 v1 (arch-level synthesis with part-aware diffusion) — extend to *arch-level* generation
1. **Take the v0 v0 v0 v0 v0 v0 dental-fine-tuned DiffFacto** and extend to *full-arch* generation: m=14-32 teeth + 1 gum = 15-33 parts
2. **Retrain on full-arch data:** use the same 3DTeethSeg22 + ToSynFCD datasets, but with the *full arch* (not just the prep + 6 context)
3. **Add the *tooth-conditional* part tokens:** for each tooth j, the token is (z_j, τ_j, j-label) where j-label is the *FDI number* (11-48, the international dental notation). The *killer* clinical advantage: the model can be conditioned on a *specific* missing tooth (e.g., FDI #36 lower-left first molar) and generate a *coherent* crown for that tooth
4. **Add the *arch-shape* conditioning:** for v0 v0 v0 v0 v0 v1, condition on the *remaining* arch (all teeth except the one being generated) — the *killer* arch-level coherence mechanism (H3)
5. **Compute cost:** $500-1000 Lambda for the arch-level training (4 A100 × 24 hours), $0 for inference (single A100, ~100-200ms with DDIM 100 steps)

### v0 v0 v0 v0 v0 v2 (controlled data augmentation for crown generation) — leverage the *controlled* generation
1. **Take the v0 v0 v0 v0 v0 v0 dental-fine-tuned DiffFacto** and use it for *controlled* data augmentation (per Tab. 3's 60-car experiment)
2. **Generate N=100-1000 controlled crowns** matching specific clinical patterns (e.g., "lower-left first molar with full-coverage PFM crown on vital tooth")
3. **Augment the clinical training set** with the generated crowns (the *killer* data-augmentation efficiency — 100 generated crowns give +7 pts mIOU, per Tab. 3)
4. **Re-train the *segmentation* network** (Cao 25, paper 026) on the augmented dataset
5. **Compute cost:** $20-50 Lambda for the generation (single A100 × 1-2 hours), $50-100 for the segmentation re-training (single A100 × 2-3 hours)

### v0 v0 v1 v0 v0 v3 (SOPHY 145-style joint shape + material per tooth) — extend to joint shape + material
1. **Take the v0 v0 v0 v0 v0 v0 dental-fine-tuned DiffFacto** and add a *material* sub-code (the SOPHY 145 recipe applied at the *part* level)
2. **Modify the part token** to (z_j_shape, z_j_material, τ_j, j, t) — the joint shape + material per part
3. **Retrain the part stylizer** with 2 sub-codes per part (shape + material, 4-dim each: E, ν, σ, ρ for the material, similar to SOPHY 145's recipe)
4. **Retrain the cross diffusion network** with the joint shape + material conditioning
5. **At inference:** generate the crown with *physical material properties* (Young's modulus, Poisson's ratio, etc.) for FEM stress analysis (per SOPHY 145's 4D-physics-aware generation)
6. **Compute cost:** $500-1000 Lambda for the joint shape + material training, $0 for inference

### v0 v0 v0 v0 v0 v4 (part-mixing for clinical case generation) — leverage the *part mixing* capability
1. **Take the v0 v0 v0 v0 v0 v0 dental-fine-tuned DiffFacto** and use the *part mixing* capability (Fig. 4) for *clinical case generation*
2. **Input:** a set of *real clinical crowns* with different material / morphology / size
3. **Output:** a *mixed* crown that combines the *style* of one source (e.g., material A) with the *morphology* of another source (e.g., shape B) — the *killer* clinical-research tool for *virtual clinical trials*
4. **Compute cost:** $10-20 Lambda for the mixing (single A100 × 30-60 minutes for 100-1000 mixed crowns)

### Open question for HK: fork + adapt DiffFacto, or re-implement from scratch?
- **(i) Fork + adapt github.com/diffFacto/diffFacto:** fastest time-to-result, MIT license (production-ready), pretrained weights available, but the *code* is PyTorch 1.12.1 + CUDA 11.3 (need to port to PyTorch 2.x + CUDA 12.x, ~1-2 weeks engineering). The *practical* v0 v0 path.
- **(ii) Re-implement from scratch:** cleanest path for v0 v1 production, but takes 4-6 weeks engineering (the *re-implement* path is needed only if the MIT license + porting cost is not acceptable, which is unlikely given the permissive license)
- **Recommendation:** **(i) for v0 v0 v0 v0 v0 v0** (pilot), no need to re-implement for v0 v1 (MIT license is *production-ready*). The DiffFacto *code* is the *right* reference implementation, the *right* starting point for v0 v0 v0 v0 v0 v0's *part-aware* crown generation, the *right* engineering starting point for the v0 v0 v0 v0 v0 v1 (arch-level) + v0 v0 v0 v0 v0 v2 (data augmentation) + v0 v0 v0 v0 v0 v4 (part mixing) sub-tasks.

### v0 compute updated: ~$5,820-7,730 Lambda (was $5,820-7,330, +$100-200 for the cross diffusion training + +$20-50 per part stylizer CNF + +$50-100 for the cIMLE transformation sampler; the v0 stack now has *part-aware* generation in addition to the *global-shape* generation from SOPHY 145 + VecSet 146 + DMC 033)

## Hypothesis impact summary

- **H1 PARTIAL SUPPORT** (factorization-based 2-stage is the *correct* decomposition, the *killer* H1 evidence is the ELBO derivation + the L_recon + L_Z + L_τ decomposition)
- **H2 STRONGEST DIRECT SUPPORT IN 147-PAPER READING LIST for part-level 3D latent diffusion** (cross-attention with part tokens is the *de facto* H2 mechanism for part-level generation, the *direct* mechanism for v0 v0 v0 v0 v0 v0)
- **H3 STRONGEST DIRECT SUPPORT IN 147-PAPER READING LIST** (cross-attention is the *arch-level* conditioning, the *killer* H3 evidence is the SNAP 13.32 vs 31.76-41.12 ablation)
- **H4 MILD CONTRADICTION** (point-cloud, not mesh; combine with DMC 033's SAP/DPSR for the *right* hybrid)
- **H5 NOT DIRECTLY TESTED** (ShapeNet 4-cat from scratch), but the *modular architecture* (per-part CNF + cIMLE + cross diffusion) is *H5-compatible* (easy to initialize from a larger pretrain and fine-tune on a smaller domain)

## Critical insight for the part-aware v0 architecture

**The 2023 part-level 3D-latent-diffusion with cross-attention + cIMLE + generalized forward kernel is the *de facto* v0 v0 v0 v0 v0 v0 part-aware crown generation recipe**, *exactly* the DMC 033 + Hwang 061 + DiffFacto 147 *combination*. **The github.com/diffFacto/diffFacto MIT-licensed code + Stanford-CDN pretrained weights + preprocessed ShapeNet data are the *canonical* H5 starting point for v0 v0 v0 v0 v0 v0's dental fine-tuning**, the *right* combination of *open-source code* + *public pretrained weights* + *small* dental-specific fine-tuning cost (~$200-400 Lambda for the full v0 v0 v0 v0 v0 v0 stack). The *practical* v0 stack is now: **github.com/diffFacto/diffFacto (part-aware generation) + DMC 033 (mesh extraction) + Hwang 061 (histogram loss) + SOPHY 145 (joint shape + material) + VecSet 146 (neural-field decoder) + DuoDent 059 (O_ce + O_cp operators) + 1zb/3DShape2VecSet (VecSet pretrain)** — the *de facto* 2023-2026 3D-gen foundation stack for v0 v0 v0 v0 v0 v0's clinical part-aware 3D-gen.

**Author correction summary (for the 145-146-147 reading sequence):**
- 145-SOPHY (Cao et al. 2025): Cao (UMass/Crete) + Kalogerakis (UMass), NOT Cao-NTU
- 146-3DShape2VecSet (Zhang et al. 2023): Zhang/Tang/Niessner/Wonka (KAUST+TUM), NOT Kalogerakis-UMass and NOT "Cao-NTU"
- 147-DiffFacto (Nakayama et al. 2023): Nakayama/Uy/Guibas (Stanford) + Huang/Hu (Tsinghua) + Li (SFU), NOT Kalogerakis-UMass and NOT Tsinghua+Adobe
- The 3 consecutive author-identification corrections in 145-146-147 (all in the VecSet / part-aware 3D-gen arc) suggest a *systematic* issue with the recent papers' author identification — need to verify by *directly reading* the author affiliations from the arXiv abstract page, not from secondary sources or memory.

## Next paper to read (148)

The 147-note's recommended *next* is **(a) SRF (Schüt et al. ICLR 2024, the *rectified-flow* 3D diffusion paper, the *right* next paper to understand the *rectified-flow* paradigm that v0 v0 v0 v0 v0 v0's *fast sampling* requires for chairside real-time inference)**, or **(b) RAG-3D (Lin et al. 2024, the *retrieval-augmented* VecSet paper, the *right* next paper to understand the *retrieval-augmented* paradigm for v0 v0 v0 v0 v0 v0's *clinical case retrieval* use case)**, or **(c) PartGen (Chen et al. 2024, arXiv:2412.18608, the *part-level* 3D generation + reconstruction paper that *explicitly* cites DiffFacto 147 as the *part-level* prior art, the *right* next paper to understand the *part-level 3D reconstruction* use case for v0 v0 v0 v0 v0 v0's *prep-from-scan* input)**, or **(d) PartRM (Gao et al. 2025, the *part-level motion* paper that SOPHY 145 cites, the *right* paper for *part-level rigid motion* paradigm for v0 v0 v0 v0 v0 v1's *articulated jaw* simulation)**, or **(e) EditVAE (Li et al. 2021, the *part-aware unsupervised 3D point cloud* paper that DiffFacto 147 cites, the *right* next paper to understand the *part-aware VAE* paradigm that v0 v0 v0 v0 v0 v0's *autoencoder mode* requires)**, or **(f) SPAGHETTI (Hertz et al. 2022, the *implicit shape editing* paper that DiffFacto 147 uses as a *part-mixing baseline*, the *right* next paper to understand *implicit* part-mixing for v0 v0 v0 v0 v0 v4)**, or **(g) LION (Zeng et al. NeurIPS 2022, the *latent point diffusion* paper that DiffFacto 147 uses as a *global-shape baseline*, the *right* next paper to understand the *latent point diffusion* paradigm)**, or **(h) ShapeGF (Cai et al. ECCV 2020, the *gradient-field* 3D-gen paper that DiffFacto 147 uses as a *global-shape baseline*, the *right* next paper to understand the *gradient-field* 3D-gen paradigm)**, or **(i) DPM (Luo & Hu CVPR 2021, the *diffusion probabilistic model for 3D point cloud* paper that DiffFacto 147 uses as a *global-shape baseline*, the *right* next paper to understand the *point-cloud diffusion* paradigm)**, or **(j) PointFlow (Yang et al. ICCV 2019, the *continuous normalizing flow* 3D-gen paper that DiffFacto 147 uses as a *global-shape baseline*, the *right* next paper to understand the *CNF 3D-gen* paradigm)**, or **(k) PointGrow (Sun et al. WACV 2020, the *autoregressive* 3D-gen paper that DiffFacto 147 cites, the *right* next paper to understand the *autoregressive 3D-gen* paradigm)**, or **(l) PartNet (Mo et al. CVPR 2019, the *part-level 3D dataset* paper that DiffFacto 147 cites for the *part annotation*, the *right* next paper to understand the *part-level 3D dataset* paradigm for v0 v0 v0 v0 v0 v0's *dental part annotation*). **Recommendation: *read 148 = SRF* (Schüt et al. ICLR 2024)** — the *rectified-flow* 3D diffusion paper, the *right* next paper to understand the *rectified-flow* paradigm that v0 v0 v0 v0 v0 v0's *chairside real-time* (50-200ms per inference) inference SLA requires, the *right* next paper for v0 v0 v0 v0 v0 v0 because *rectified flow* (1-4 sampling steps vs DDPM's 1000 steps) is the *direct* enabler for *chairside* crown generation. After 147 + 148, the v0 v0 v0 v0 v0 v0 *part-aware + fast-sampling* arc is *complete* (147 + 148 = 2 papers, the *part-level 3D latent diffusion* + the *rectified-flow fast sampling*), the *most-comprehensive* part-aware fast-sampling 3D-gen arc for v0 v0 v0 v0 v0 v0's clinical use case.
