# 005 — LION: Latent Point Diffusion Models for 3D Shape Generation

- **Title:** LION: Latent Point Diffusion Models for 3D Shape Generation
- **Authors:** Xiaohui Zeng, Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, Karsten Kreis (NVIDIA Toronto + U. of Toronto + Vector Institute)
- **Year:** 2022
- **Venue:** NeurIPS 2022
- **Links:**
  - Paper (arXiv): https://arxiv.org/abs/2210.06978
  - OpenReview: https://openreview.net/forum?id=tHK5ntjp-5K
  - Project page: https://research.nvidia.com/labs/toronto-ai/LION/
  - Code (official PyTorch): https://github.com/nv-tlabs/LION
- **Code/data:** Official implementation released by NVIDIA (Torch, Python 3, single GPU training). Baselines for context: PVD (Point-Voxel Diffusion) at https://github.com/alexzhou907/PVD, DPM (diffusion-point-cloud) at https://github.com/luost26/diffusion-point-cloud, Shape As Points (SAP) at https://github.com/SongyouPeng/SAP. Trained on ShapeNet (airplane 2,832 / chair 4,746 / car 5,248 train shapes), 13-class ShapeNet-vol (airplane/bench/cabinet/car/chair/display/lamp/loudspeaker/rifle/sofa/table/telephone/watercraft), and 55-class ShapeNet. Data: per-shape normalized to `[-1, 1]³` (vol variant) or globally normalized (PointFlow split).

---

## TL;DR

LION is the **first hierarchical latent diffusion model for 3D point clouds**, decomposing the generation problem into a *two-stage* pipeline — **(1) train a hierarchical point-cloud VAE** (PVCNN encoder/decoder + global shape latent `z0 ∈ ℝ^128` + point-structured latent `h0 ∈ ℝ^(3+Dh)×N` with `N=2048` latent points), then **(2) train two separate DDMs in those latent spaces** (a ResNet-FC DDM over `z0`, a conditional PVCNN DDM `ϵψ(h_t, z0, t)` over `h0` with shape-latent conditioning via adaptive GroupNorm). SoTA on ShapeNet airplane/chair/car single-class and 13-class benchmarks (Table 1–4), beating prior point-cloud DDMs (PVD, DPM) by 5–15 points 1-NNA, and uniquely enables **multimodal voxel-guided synthesis** and **multimodal denoising** by fine-tuning only the encoders to map perturbed inputs back into the latent space, then running a few **"diffuse-denoise"** steps in latent space — a trick that, for us, **is the architectural template for partial-scan → complete-crown completion** (H3). The natural "right answer" for H2 is now: do we generate SDFs (paper 004 / Diffusion-SDF) or points (paper 005 / LION)?

## Research question

> Can we build a DDM-based 3D shape generative model that **simultaneously achieves (i) high generation quality, (ii) flexibility for conditional/interactive use, and (iii) mesh output** — i.e., escapes the "either good samples or smooth meshes" trade-off that prior point-cloud DDMs (PVD, DPM) hit?

Their answer: **yes, by putting the diffusion model in a *learned latent* space rather than directly on point clouds, and *augmenting* (not replacing) the latent pipeline with a modern surface-reconstruction method at decode time.** This is the same insight that LDM (Rombach et al., 2021) brought to images, and the same "diffusion in latent space" pattern that Diffusion-SDF (paper 004) brought to neural SDFs — but with a *point-cloud-shaped* latent space that is "ideal for DDMs" (per the authors) and avoids the consistency-issues that paper 004 hit when diffusing in a 1-D latent that parameterizes a full neural field.

The two key claims are:
1. **Hierarchical VAE + latent DDMs > point cloud DDMs directly**, because the DDMs only have to learn the *mismatch* between actual encodings and a Gaussian prior (the "prior hole problem" from Vahdat et al. 2021), rather than the full point cloud distribution.
2. **Augmenting with SAP (Shape As Points) gives mesh output *for free***, because LION's point clouds can be turned into meshes with a fine-tuned SAP that has learned the *noise distribution* of LION's autoencoder.

## Method

### Stage 1: Hierarchical point-cloud VAE

- **Inputs:** point cloud `x ∈ ℝ^(3×N)` with `N = 2048` points.
- **Two latents:**
  - Global shape latent `z0 ∈ ℝ^128` (vector).
  - Point-structured latent `h0 ∈ ℝ^((3+Dh)×N)` — `N` latent points, each with xyz + `Dh` feature dims. `Dh = 1` by default (the appendix ablates this; more dims marginally help).
- **Encoders** (parametrized by `ϕ`):
  - `q_ϕ(z0 | x)` — predicts a factorial Gaussian (`μ_z`, `σ_z²`) over `z0` from the input.
  - `q_ϕ(h0 | x, z0)` — predicts a factorial Gaussian over `h0` per latent point, *conditioned on `z0`*.
- **Decoder** `p_ξ(x | h0, z0)` — a factorial Laplace with predicted means, fixed unit scale → **L1 reconstruction loss** between `x` and the decoded xyz. (Note: the decoder predicts the *xyz of the output points*, so `h0`'s xyz coordinates are the "smoothed positions" of the output, and the latent features modulate them.)
- **VAE training objective** (Eq. 5; the paper writes this as the *negative* ELBO since we minimize):
  ```
  L_VAE = -E_q[log p(x | h0, z0)] + λ_z · KL(q(z0|x) || N(0,I)) + λ_h · KL(q(h0|x,z0) || N(0,I))
  ```
  - **Crucial schedule**: `λ_z = λ_h` start at `1e-7`, anneal linearly to `0.5` over 8,000 epochs. This is *very* aggressive KL annealing — the early epochs are essentially an autoencoder (no regularization) and the regularization kicks in late. The paper says this is necessary because the L1 reconstruction is too strong a pressure to allow early KL; the model needs to "discover" a good encoder first, then have the prior pulled in.
  - Note: `λ = 1` would be a rigorous ELBO; the paper uses `λ = 0.5`, which is a "tighter" prior that the authors found empirically better.
- **Skip weight 0.01, variance offset 6.0** — two additional numerical-stability hacks the paper applies; not architectural, but worth noting if we reimplement.
- **Backbone: PVCNN** (Point-Voxel CNN, Zhou et al. ICCV 2021) for the encoders and decoder. Combines PointNet's set-processing efficiency with convolutions' spatial inductive bias. The PVCNN architecture is `SA1 → SA2 → SA3 → SA4 → (optional global attention) → FP4 → FP3 → FP2 → FP1` (4 set-abstraction + 4 feature-propagation stages, voxel grid sizes 32/16/8/4, groupers with 32 neighbors and radii 0.1/0.2/0.4/0.8, hidden dims 32/64/128/128). Full hyperparameter table in App. D (Table 9).

### Stage 2: Two latent DDMs

- **Global shape latent DDM** `ϵ_θ(z_t, t)`: small ResNet with fully-connected layers (implemented as 1×1 convs). Sinusoidal time embedding (dim 64) → MLP → repeat through 4 ResNet blocks. Operates on the 128-dim `z_t`.
- **Latent point DDM** `ϵ_ψ(h_t, z0, t)`: **conditional** DDM over the `N=2048` latent points. The conditioning on `z0` is implemented via **adaptive Group Normalization (AdaGN)** in the PVCNN layers — i.e., the scale/shift of every AdaGN layer is predicted from a projection of `z0`. This is the same trick that powers Stable Diffusion's text conditioning and is the architectural mechanism we'll want to copy for H3 (conditioning on adjacent + opposing teeth).
- **Score matching objectives** (Eq. 6, 7):
  ```
  L_SMz(θ) = E[ ||ϵ - ϵ_θ(z_t, t)||² ]      # shape DDM
  L_SМh(ψ) = E[ ||ϵ - ϵ_ψ(h_t, z0, t)||² ]  # point DDM, conditioned on z0
  ```
- **Mixed score parametrization** (Vahdat et al. 2021): the score models predict a *residual correction* to the analytic Gaussian score, not the noise `ϵ` directly. This is the "latent diffusion" trick that makes training the latent DDMs stable when the encodings are regularized to be near-Gaussian.
- **Diffusion process**: linear `β` schedule, `β_0 = 1e-4`, `β_T = 0.02`, `T = 1000` steps. Standard DDPM forward process (Eq. 1).
- **Training** (App. E.3, Table 10):
  - VAE: 8,000 epochs, Adam, lr 1e-3, batch 128, β₁=0.9, β₂=0.99.
  - Latent DDMs: 24,000 epochs, Adam, lr 2e-4, weight decay 3e-4, EMA decay 0.9999, 20-epoch warmup. Spectral normalization + gradient norm regularizer with weight 1e-2.
  - **Single-class training cost: ~550 GPU-hours** (~110h VAE + ~440h DDMs). Total project compute: ~340,000 V100 GPU-hours. **The single-class number is the relevant one for us** — and it's still prohibitive on anything smaller than a V100/A100. Mac mini M4 cannot do this in any reasonable time.

### Generation

- **Sample `z0` from the shape DDM** (ancestral sampling, Eq. 4).
- **Sample `h0` from the point DDM, conditioned on `z0`** — i.e., the point DDM learns a distribution over *latent points given a global shape code*.
- **Decode `h0` and `z0` → point cloud `x`** via the frozen VAE decoder.
- **Optional: reconstruct mesh** via SAP (fine-tuned on LION's noisy-decoded point clouds). Standard Marching Cubes would also work but produces noisier surfaces; SAP's "differentiable Poisson surface reconstruction" + learned refinement is better.

### "Diffuse-denoise" trick (Sec. 3.1)

This is the most important contribution for us. The trick:
1. Encode a *perturbed* input (voxelized, noisy, partial) using a **fine-tuned encoder** → get `(z0, h0)`.
2. **Diffuse** `(z0, h0)` for `τ` steps in the *prior* of the latent DDMs (forward process) → `(z_τ, h_τ)`.
3. **Denoise** `(z_τ, h_τ)` back to `(z0', h0')` using the *trained* latent DDMs (reverse process).
4. **Decode** `(z0', h0')` → cleaned point cloud.

By varying `τ` (1, 10, 50, 200, 1000), the user gets a **trade-off knob** between fidelity-to-input and output quality. The fine-tuning only touches the encoders (the DDMs are frozen). This is exactly the workflow we need: take a partial intra-oral scan → encode to latent → diffuse-denoise → decode to a complete crown.

The paper's voxel-guided synthesis experiments (Sec. 5.4, Fig. 12, 13) show LION cleanly beating PVD and DPM at this task — the latter two "perform very poorly for outlier noise or voxel inputs" because their DDMs operate on point clouds directly with no encoder to absorb the input distribution shift.

### Surface reconstruction (Sec. 3.1 + App. C.4)

- LION's outputs are point clouds. For mesh output, fine-tune **SAP (Shape As Points, Peng et al. NeurIPS 2021)** on data generated by LION's autoencoder. The fine-tuning data is produced by: encode clean shapes → diffuse-denoise in latent space (random number of steps in {20, 30, 35, 40, 50}) → decode → feed to SAP.
- This adjusts SAP to LION's *specific* point-noise distribution. Without this, SAP's output is visibly worse on LION's samples.
- For our use case, we don't need SAP — we need printable meshes, and Marching Cubes (after solving the SDF) or Poisson reconstruction (on the points) is sufficient. But the *pattern* (point cloud generative model + mesh reconstruction at decode time) is the right one.

## Results

### Table 1 — Single-class unconditional, ShapeNet PointFlow split (1-NNA↓, lower is better; 50% = identical distributions)

| Method | Airplane CD | Airplane EMD | Chair CD | Chair EMD | Car CD | Car EMD |
|---|---|---|---|---|---|---|
| r-GAN | 98.40 | 96.79 | 83.69 | 99.70 | 94.46 | 99.01 |
| l-GAN (EMD) | 89.49 | 76.91 | 71.90 | 64.65 | 71.16 | 66.19 |
| PointFlow | 75.68 | 70.74 | 62.84 | 60.57 | 58.10 | 56.25 |
| SetVAE | 76.54 | 67.65 | 58.84 | 60.57 | 59.94 | 59.94 |
| DPM | 76.42 | 86.91 | 60.05 | 74.77 | 68.89 | 79.97 |
| **PVD** | **73.82** | **64.81** | **56.26** | **53.32** | **54.55** | **53.83** |
| **LION** | **67.41** | **61.23** | **53.70** | **52.34** | **53.41** | **51.14** |

LION beats the previous best (PVD) by 6.4, 2.6, 1.1 points of 1-NNA-CD on airplane/chair/car. The biggest win is on airplane (the most topologically varied class).

### Table 4 — 13-class ShapeNet-vol, jointly trained, no class conditioning

| Method | CD | EMD |
|---|---|---|
| PointFlow | 63.25 | 66.05 |
| ShapeGF | 55.65 | 59.00 |
| DPF-Net | 67.10 | 64.75 |
| DPM | 62.30 | 86.50 |
| PVD | 58.65 | 57.85 |
| **LION** | **51.85** | **48.95** |

LION beats PVD by 6.8 points CD on the 13-class setting, with no class conditioning. This is the most impressive result — it means the hierarchical VAE is doing the multi-modal heavy lifting, not a class-conditional mechanism.

### Sampling speed

- **1000-step DDPM ancestral sampling: 27.12 s/shape** (on V100).
- **25-step DDIM sampling: 0.89 s/shape** — real-time. (App. F.9 confirms quality loss is small.)

### Voxel-guided synthesis (Sec. 5.4, Fig. 12-13)

- LION's 1-NNA-CD on voxel inputs: **~60-65%** at 10-50 diffuse-denoise steps, vs. PVD's **~80-90%** (worse) at the same step count.
- LION's voxel IOU (fidelity to input): **~80%** at 10-50 steps, vs. PVD's **~40-60%** (also worse).
- LION is the *only* method that both (a) respects the input voxel structure AND (b) generates high-quality outputs.

### Compute

- 340,000 V100 GPU-hours total.
- Per single-class model: ~550 GPU-hours.
- SAP fine-tuning: <1,000 epochs on the LION-generated point clouds.

## Connections to our hypotheses

### H1 (2-stage > end-to-end) — **STRONG SUPPORT**

LION is *literally* 2-stage: stage 1 trains the VAE alone, stage 2 trains the latent DDMs alone. The motivation in their Sec. 1 is precisely the H1 argument: "directly diffusing on point clouds is hard because the distribution is complex → regularize to Gaussian via VAE → then diffusion has an easier job." This is the cleanest possible empirical support for the 2-stage decomposition we're already using (3DTeethSeg22 for sub-task 1, then conditional generation for sub-tasks 3-4).

**Action:** Treat LION as the H1 proof-of-concept for the *generation* half of our pipeline. The 2-stage pattern is now triple-confirmed: DiGS (paper 003) for reconstruction, Diffusion-SDF (paper 004) for SDF generation, LION (paper 005) for point generation.

### H2 (diffusion on point clouds > mesh-based VAE for surface generation) — **REFRAME: H2 is about "diffusion > mesh VAE", not "point cloud > mesh"**

LION doesn't directly test H2's "point cloud > mesh" claim. What it *does* test is the "diffusion > mesh VAE" half. The relevant comparison is LION vs. SetVAE (a point-cloud VAE) and ShapeGF (a gradient-field VAE): LION beats both handily (e.g., chair CD 53.70 vs. SetVAE 58.84, ShapeGF 58.01). So the diffusion > VAE half of H2 is again supported.

But the "point cloud > mesh" half is *not* directly tested here — LION's outputs go through SAP for the mesh. The right read of LION + Diffusion-SDF together is: **both diffusion and point cloud representation work; the question is which combination**. And for our H2 to be more than just "diffusion > VAE", we should commit to: **(LION's point cloud DDM + Poisson/Marching Cubes) vs. (Diffusion-SDF's SDF DDM + Marching Cubes)** as the H2 comparison.

### H3 (conditioning on adjacent + opposing teeth improves outer surface) — **STRONGEST SUPPORT YET, ARCHITECTURALLY**

LION's point DDM is **`ϵ_ψ(h_t, z0, t)` — explicitly conditioned on the global shape latent `z0` via AdaGN**. This is the exact architectural mechanism we need for H3: condition the missing-tooth generation on a learned feature vector extracted from the adjacent + opposing teeth.

Two specific carries for us:
1. **The "diffuse-denoise" trick is the H3 inference algorithm.** Encode the observed (context teeth) point cloud → fine-tune the encoder to map it to a meaningful `z0` → diffuse-denoise with a `τ` that trades off input fidelity vs. output quality. This is *exactly* the partial-scan → complete-crown workflow. The 50-step diffuse-denoise in LION's voxel-guided synthesis is a proof-of-concept for the missing-tooth case (a missing tooth is "the most extreme voxel removal" of all — we just remove all the voxels of the target tooth and ask the model to fill them in).
2. **The 1-NNA results on the 13-class ShapeNet-vol setting are the cleanest evidence that a `z0`-style global feature can disambiguate across diverse categories** — our analog is: a `z0` extracted from the *arch* (not just the target tooth) can disambiguate across FDI positions 11-48.

**Action:** This is now the strongest candidate for our H3 implementation. Sketched: (a) train a VAE on complete arches (from 3DTeethSeg22), (b) the *encoder* of the VAE produces `z0` from the arch point cloud, (c) the *decoder* generates the target tooth's latent points `h0` from `z0`, (d) at inference, take a partial arch (one tooth missing) → encoder → `z0` → decoder → complete missing tooth. The "diffuse-denoise" handles the noise from the partial scan.

### H4 (implicit SDF > explicit mesh) — **WEAKENS / DOES NOT SUPPORT**

LION is *explicit* point cloud, not implicit SDF. The paper's mesh outputs go through SAP (differentiable Poisson), which is *closer* to implicit than marching cubes but is still fundamentally a points-to-mesh pipeline. The H4 camp is paper 004 (Diffusion-SDF) + paper 002 (DeepSDF) + paper 003 (DiGS).

**The H2×H4 vs. H2×H4' question.** This is the **core decision we need to make** in the next 2-3 weeks. Two viable paths:
- **Path A (paper 004's Diffusion-SDF):** generate SDFs directly. Pros: H4 is intuitive (the mesh is the zero-level-set of a learned continuous function), clinical fit metrics (margin gap) are easier to compute, Marching Cubes is well-validated for 3D printing. Cons: the diffusion model has to operate in a 1D latent that parameterizes a 9-layer MLP, end-to-end fine-tuning is expensive, latent space can be misspecified.
- **Path B (LION):** generate point clouds, reconstruct mesh via Poisson / Marching Cubes. Pros: simpler latent space (point-structured, not MLP-parameterized), SAP-style surface reconstruction is well-validated, the "diffuse-denoise" trick is perfect for H3. Cons: need a separate surface-reconstruction step, point cloud noise propagates to the mesh.

**Recommendation:** run a small pilot on both (a single tooth class, ~100 training meshes) and pick the one that produces better printable meshes in terms of the chamfer distance to the ground truth *and* the visual smoothness of the occlusal surface. This is a one-week experiment for Red, and the decision gates the next 6 months of work.

### H5 (synthetic data from CAD libraries bootstraps training) — **STRONG SUPPORT**

LION is trained *entirely* on synthetic ShapeNet CAD data and produces state-of-the-art generative samples. This is the strongest possible indirect support for H5: a clean experimental demonstration that synthetic CAD → high-quality 3D generative model is a viable pipeline. Our analog is exocad / 3Shape outputs (synthetic crowns) → trained diffusion model → real-crown-quality outputs.

**Action:** Plan data acquisition around synthetic CAD first (exocad has an SDK, 3Shape has an academic license tier, and the Tufts Dental Database has open meshes). Don't block on real IOS scans.

## Surprises and interesting things buried in the appendix

1. **The KL annealing schedule is wildly aggressive (1e-7 → 0.5 over 8,000 epochs).** This means the VAE is essentially a deterministic autoencoder for the first ~80% of training, and the prior is only enforced at the end. This is a "soft" way to avoid the posterior collapse that's common in VAE training, and it's a hack worth borrowing.

2. **The "extra dimensions of the latent points" ablation (App. F.1.3) shows that `Dh = 1` is essentially as good as `Dh = 32`** — most of the information is in the xyz coordinates of the latent points, and the per-point features are almost entirely redundant. This is a useful architectural simplification: we don't need to fiddle with `Dh`.

3. **The shape latent learns interpretable global categories** (Fig. 8 in the paper + App. F.3.2 t-SNE). Fixing `z0` and sampling different `h0` produces shapes that "look like the same kind of object" but with different details. This is *the* inductive bias we want for H3: a global arch context + a per-tooth detail.

4. **Sampling time has a 30× speedup (DDIM 25 steps, 0.89s vs. 27s) with minimal quality loss.** This matters for any user-facing product. If we ever want a "live design preview" UX (à chair CAD software), DDIM is what we run.

5. **The SAP fine-tuning is the difference between "nice meshes" and "terrible meshes"** (App. F.1.4). Without fine-tuning, SAP's output on LION samples has visible artifacts because LION's decoded point clouds have a different noise distribution than clean ShapeNet. This is a general lesson: **the surface-reconstruction step needs to be calibrated to the generative model's noise distribution**. For us: if we go with Path B (point cloud DDM), the Poisson reconstruction / Marching Cubes parameters need to be tuned on LION-style (or our DDM-style) noisy points, not on clean scans.

6. **The 55-class ShapeNet model** (App. F.3.2) — training on *all 55 ShapeNet categories* without class conditioning still produces recognizable shapes. The hierarchical VAE is doing the categorization implicitly. This is encouraging for our setting where the "category" (FDI position) is one of 32 options and we have relatively few examples per class.

7. **β schedule is linear from 1e-4 to 0.02, T=1000 steps.** Standard, but worth noting: this is the *same* schedule used in the original DDPM (Ho et al. 2020). The paper doesn't ablate this, so we should treat it as a sensible default.

8. **The encoder-fine-tuning for voxel-guided synthesis takes only ~130-470 epochs to converge** (App. E.4). This is much faster than training the latent DDMs (24,000 epochs). For us: if we adopt LION's approach, the per-task fine-tuning of the encoders is cheap; we don't have to retrain the DDMs for every new conditioning input type.

## Quote-worthy sentences

- **"By mapping point clouds into regularized latent spaces, the DDMs in latent space are effectively tasked with learning a smoothed distribution. This is easier than training on potentially complex point clouds directly, thereby improving expressivity."** (Sec. 1, paragraph on expressivity) — the core argument for why latent diffusion > direct diffusion on raw data, applicable to our SDF-vs-points choice.

- **"The shape latent variables capture global shape, while the latent points model details. We validate this by fixing the shape variable to different values and only sampling different latent points."** (project page) — the cleanest empirical statement of why hierarchical > monolithic latent space, and the strongest template for H3 (global arch → per-tooth details).

- **"Performing more diffuse-denoise steps means that more independent, novel shapes are generated. These will be cleaner and of higher quality, but also correspond less to the noisy or voxel inputs used for guidance."** (Sec. 5.4) — the explicit `τ` knob is the right interface for the dentist: "give me a crown that *respects* the prep margin" (low τ) vs. "give me a crown that *looks great*" (high τ).

- **"PVD and DPM perform acceptably for normal and uniform noise, which is similar to the noise injected during training of their DDMs, but perform very poorly for outlier noise or voxel inputs... It is LION's unique framework with additional fine-tuned encoders in its VAE and only latent DDMs that makes this possible."** (Sec. 5.4) — the strongest single sentence arguing for the encoder-fine-tuning pattern over direct DDM conditioning.

- **"LION can not directly generate textured shapes. A promising extension would be to include image-based training by incorporating neural or differentiable rendering."** (Sec. 6) — honest limitation. For us, this is fine (crowns don't need textures) but the *observation* that DDMs need their conditioning to be encoded by a *separate* network (not by the DDM directly) is the architectural lesson.

## Code/data link

- Official PyTorch code: https://github.com/nv-tlabs/LION (MIT-style license — used MIT-licensed baselines throughout).
- Datasets: ShapeNet (https://shapenet.org/terms), ShapeNet-vol splits as defined in Peng et al. (SAP, https://github.com/SongyouPeng/SAP), PointFlow splits (https://github.com/stevenygd/PointFlow).
- Compute: V100 NVIDIA GPUs. ~550 V100-hours per single-class model, ~340,000 total for the project.

## For our project

1. **The two-path decision.** Run a small pilot (1 week, ~50 V100-hours) on **both** Diffusion-SDF (paper 004) and LION (paper 005) for a single tooth class (e.g., upper first molar, FDI 16) with ~100 training meshes scraped from Tufts / exocad demo files. Compare (a) reconstruction chamfer distance to held-out ground truth, (b) visual smoothness of the occlusal surface, (c) the cost of a single forward pass (since inference runs on the M4 Mac mini). **This is the single most important decision for the next quarter** — the path we pick gates everything from data acquisition to the evaluation metric set. Lean toward LION unless the pilot shows Diffusion-SDF meaningfully better, because (i) the latent space is simpler, (ii) the "diffuse-denoise" trick is the right interface for H3, (iii) point cloud → Poisson reconstruction is well-validated for 3D printing.

2. **Adopt the hierarchical VAE + latent DDM pattern verbatim.** Even if we end up on Path A (SDF), the "first train a regularizing VAE, then train latent DDMs" pattern is the right way to make the diffusion step tractable. This is the architectural lesson from LION that paper 004 (Diffusion-SDF) *also* uses (their "SDF-VAE"), and that we should not try to skip.

3. **Use the diffuse-denoise trick for H3 inference.** The workflow: (a) train a VAE on complete arches (from 3DTeethSeg22), (b) at inference, take a partial arch (one tooth missing) → encode with a *fine-tuned* encoder (fine-tuned on partial arches) → get `(z0, h0)`, (c) diffuse-denoise for `τ` steps in the prior space, where `τ` is the dentist's "match the prep margin vs. ignore it" knob, (d) decode → final tooth. The 50-step diffuse-denoise in LION's voxel-guided synthesis is a direct proof-of-concept.

4. **Drop SAP from the plan; use Poisson surface reconstruction (Path B) or Marching Cubes (Path A).** SAP is great for artistic smooth surfaces; we need clinical-grade surfaces with known thickness and good margin behavior. Poisson / Marching Cubes are the right tools. But — *caveat from this paper* — calibrate the surface-reconstruction parameters on the *generative model's noise distribution*, not on clean scans. If we adopt Path B, run a parameter sweep on Poisson reconstruction depth / point density on a held-out set of LION-generated crowns (or our DDM-generated crowns).

5. **The shape latent (`z0`) is where we should put the global arch context.** Concretely: train a separate "arch encoder" that takes the arch point cloud (minus the target tooth) and outputs a `z0`. The arch encoder is *fine-tuned* for each FDI position, or shared across FDI positions with an additional "FDI embedding" concatenated to `z0`. This is a small extension of the LION pattern and a clean realization of H3.

6. **Compute budgeting.** Even the cheapest LION-class model needs ~110 V100-hours for the VAE alone. This is not happening on the M4 Mac mini. Plan: rent a single A100 on Lambda Labs for the pilot (~$1.50/hr × 100hr = $150), and a 4×A100 node for the full tooth-class training (~$6/hr × 1000hr = $6,000). The $6K is a small fraction of the cost of a single commercial exocad seat.

7. **Sampling time matters for the dentist UX.** DDIM 25 steps at 0.89s on V100 → roughly 10s on M4 (since M4 GPU is ~10× slower for diffusion forward passes). This is borderline acceptable for a "live preview" mode where the dentist drags a slider. If we want sub-second on M4, we need distillation (like consistency models) or aggressive DDIM (5-10 steps), which is a research sub-question for paper 006+.

8. **The KL annealing schedule (1e-7 → 0.5 over 8000 epochs) is worth adopting verbatim.** It avoids posterior collapse and is well-tuned for hierarchical VAE training. If we reimplement either path, copy this schedule.

9. **Add paper 006 to the queue: probably "3D Shape Generation with Score-Based Diffusion Models on Neural Fields" or "Neural Dual Contouring" for the mesh-extraction step.** The former tests if there's a way to *skip* the VAE+DDM two-step and go straight to neural field diffusion. The latter is about turning a point cloud / implicit representation into a clean printable mesh (the clinical-grade surface-reconstruction step that paper 005's SAP doesn't really solve). For HK's review, mark these as "read next, depth-first."

10. **Long-term: synthesize an entire clinical workflow as a "diffuse-denoise" pipeline.** Take a partial scan → fine-tuned arch encoder → `z0` → tooth-specific DDM (conditioned on `z0` and FDI position) → tooth latent `h0` (Path B) or SDF latent `z_sdf` (Path A) → decode + surface reconstruction → final crown. The diffuse-denoise step happens *between* the arch encoder and the tooth DDM, and the τ parameter is exposed to the dentist as "how much to respect the prep margin" vs. "how much to let the model hallucinate the occlusal surface." This is the **end-state architecture** we should be designing *toward*, even if we ship a single-tooth unconditional prototype first.

---

*Scholar's note: LION completes the trio of H2 papers (Diffusion-SDF, LION, future: PVD-extension / Point-Voxel Diffusion V2). It does not change the H4 story (DiGS + Diffusion-SDF are still the implicit-SDF camp), and it strongly supports H1 + H3 architecturally. The "diffuse-denoise" trick is the most important single architectural idea from this paper, and it generalizes to our problem in a way that the prior two papers didn't make obvious. Action item for Red: pilot the LION-vs-Diffusion-SDF decision in the next 1-2 weeks. Action item for HK: review the τ-as-dentist-knob idea; if you like it, that becomes a design pillar for the eventual UX.*
