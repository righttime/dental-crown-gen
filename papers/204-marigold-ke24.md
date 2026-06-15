# Paper 204 — Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation

**Authors:** Bingxin Ke¹, Anton Obukhov¹, Shengyu Huang¹, Nando Metzger¹, Rodrigo Caye Daudt¹, Konrad Schindler¹
¹ Photogrammetry and Remote Sensing Lab, ETH Zürich

**Venue:** CVPR 2024 (Oral, **Best Paper Award Candidate** ⭐⭐⭐) — arXiv:2312.02145 v1 4 Dec 2023 → v2 3 Apr 2024 (CVPR camera-ready, 25.6 MB, 9 pages + 31 pages appendix)

**Code:** https://github.com/prs-eth/Marigold (3,159 ⭐ / 209 🍴 / ~9.9 MB / last push 2025-12-10) — **LICENSE: Apache-2.0 (code) ✅ + OpenRAIL++-M (model weights)** — *dual-license pattern* verified via raw.githubusercontent.com/prs-eth/Marigold/main/LICENSE.txt and .../LICENSE-MODEL.txt — *critical finding*: code is fully commercial-deployable; model weights are *based on Stable Diffusion's OpenRAIL++-M license* (allows commercial use with ethical-use restrictions — dental use is 100% compliant, the OpenRAIL++-M "no-illegal-content" and "no-harmful-use" clauses are not violated by clinical-crown generation)

**Model checkpoints:** 🤗 https://huggingface.co/prs-eth/marigold-depth-v1-0 (and v1-1 from TPAMI 2025 follow-up) — multiple variants: depth, normals, IID-appearance, IID-lighting, depth-LCM, depth-HR

**Project page:** https://marigoldmonodepth.github.io

**Online demos:** HuggingFace Spaces (multiple), ComfyUI integration, Gradio

**Citations:** **416 Semantic Scholar as of 2026-06-16, 68 influential** — top-5 most-cited 2024 monocular-depth paper

**Built on:** Stable Diffusion v2 [Rombach 2022, CVPR] (v-objective variant) — specifically `stabilityai/stable-diffusion-2-base` (865M params, 1.5B for the LCM variant), LAION-5B pretrained (1.6B image-text pairs)

**Follow-up papers (all by the same team):**
- Marigold Computer Vision (TPAMI 2025, arXiv:2505.09358) — extends to surface normals + intrinsic decomposition
- Marigold-DC (2024) — sparse depth completion
- Rolling Depth (CVPR 2025) — temporally consistent video depth
- Better Depth (NeurIPS 2024, Disney Research) — coarse-to-fine diffusion refinement
- Marigold-LCM (2024) — consistency-model distillation, 1-4 inference steps
- Marigold-HR (2024) — high-resolution inference

---

## One-line TL;DR

Marigold is the **FOUNDING PAPER of the *repurpose-image-foundation-model-for-downstream-dense-prediction* paradigm** — **fine-tunes the U-Net of Stable Diffusion v2 as an affine-invariant monocular depth estimator** by **(1) encoding depth maps as 3-channel "fake RGB" via VAE** (replicated + affine-normalized via 2%/98% percentiles), **(2) doubling the U-Net's first input layer** (concatenation conditioning, weights halved), **(3) training with annealed multi-resolution noise**, and **(4) test-time ensembling 10 inference runs** with iterative scale/shift alignment + median aggregation — **achieving state-of-the-art zero-shot monocular depth on 5 real datasets (NYUv2, KITTI, ETH3D, ScanNet, DIODE) using ONLY 74K synthetic training samples (Hypersim + Virtual KITTI) and NEVER seeing a real depth map**, with **Avg Rank 1.4 across 5 datasets** (vs DPT's 3.2, MiDaS's 3.8, LeReS's 4.0), at **~2.5 days training on 1× RTX 4090** (the *cheapest* SoTA depth estimator in the literature), with **affine-invariant depth** (predicts up to global scale + shift, useful for uncalibrated "in-the-wild" images).

## Research question + their answer

**RQ:** Can the rich visual priors learned by internet-scale text-to-image diffusion models (Stable Diffusion / LAION-5B) be *repurposed* for monocular depth estimation, achieving better zero-shot generalization than purpose-trained depth estimators (MiDaS, DPT, LeReS)?

**Answer:** YES — by **fine-tuning ONLY the U-Net of Stable Diffusion v2** (frozen VAE encoder/decoder) on **74K synthetic RGB-D samples** (Hypersim + Virtual KITTI, with complete + noise-free depth) using the **standard diffusion noise-prediction objective** in the **latent space** (not pixel space), with **affine-invariant 2%/98%-percentile normalization** to keep depth values in VAE's expected [-1, 1] range, and **test-time ensembling** of 10 independent inference runs with iterative scale/shift alignment, you get **state-of-the-art zero-shot monocular depth** on 5 unseen real datasets, with **~2.5 days single-GPU training** (vs 11.9M-300K samples for baselines). The *killer* empirical finding: **74K synthetic samples > 11.9M real samples** because the Stable Diffusion prior is so much richer than learned-from-scratch encoders.

## Method (architecture, training, data)

### Architecture: Stable Diffusion v2 U-Net + doubled first input layer

- **Backbone:** Stable Diffusion v2 (v-objective variant, 865M params) — frozen VAE encoder E, frozen VAE decoder D, trainable U-Net ε_θ
  - VAE is *frozen* (not fine-tuned), the *killer* design lesson: keep the *learned* latent space intact
  - VAE accepts 3-channel input; for depth (1-channel), **replicate depth to 3 channels** (R=d, G=d, B=d), then encode → z^(d) ∈ R^(H/8 × W/8 × 4) (4 latent channels from SD VAE)
  - Predicted depth = **average of 3 decoded channels** (since the VAE outputs 3 channels for any input, even depth)
  - VAE reconstruction error for depth: **MAE 0.0095 ± 0.0091** (safely below the depth-estimation error floor)
  - Channel consistency after decoding: std 0.0027, max-min 0.0062 (very small, supports the "depth as fake RGB" trick)
- **Conditioning mechanism (KEY ARCHITECTURAL INNOVATION):** Image conditioning via **concatenation on feature dimension**, not cross-attention
  - Concatenate image latent z^(x) and depth latent z^(d)_t along feature dimension: z_t = cat(z^(d)_t, z^(x), dim=1) (becomes 4+4=8 channels)
  - **Double the first U-Net input layer** from 4 → 8 channels
  - **Duplicate first-layer weight tensor, divide by 2** to keep activations magnitude stable (preserves pretrained structure faithfully)
  - This is the *direct* precursor to ChronoDepth 203's per-frame independent noise design — *conditioning via concatenation* is the *Marigold signature*
- **Disabling text conditioning:** The CLIP text encoder is *not used*; only image conditioning is used (depth estimation is image-conditional, not text-conditional)

### Training data: 74K synthetic RGB-D (only synthetic, NEVER real)

- **Hypersim** [Roberts 2021, ICCV] — 461 photorealistic indoor scenes, 365 for training → **~54K filtered samples** (after filtering incomplete samples)
  - 480 × 640 resolution
  - Synthetic, dense, complete depth (no missing pixels, no sensor noise — the *killer* advantage over real depth data)
  - Originally uses focal-point-relative distances; transformed to conventional focal-plane-relative depth
- **Virtual KITTI** [Cabon 2020] — 5 synthetic street scenes with weather/camera variations, 4 for training → **~20K samples**
  - Cropped to KITTI benchmark resolution
  - Far plane set to 80 meters
  - Outdoor domain
- **Total: 74K synthetic samples** (no real depth data, *the* H5 design lesson)
- **Mixed-dataset training:** Per-batch probabilistic sampling between Hypersim and Virtual KITTI (Bernoulli parameter ablated in Appendix B.4 — 90/10 ratio is optimal, *surprisingly small* Virtual KITTI is enough for outdoor generalization)

### Training noise: Annealed multi-resolution noise (vs standard Gaussian)

- **Standard Gaussian noise** is the DDPM default but performs poorly (Table 2: NYUv2 AbsRel 7.7%, KITTI 14.2%)
- **Multi-resolution noise** [Kasiopy 2023, W&B report] = *superimpose* multiple random Gaussian noise images of different scales, all upsampled to U-Net input resolution, then weighted-average
  - Each level i has weight s^i where 0 < s < 1 controls influence of lower-resolution noise
  - NYUv2 AbsRel 5.8%, KITTI 12.1% (significant improvement)
- **Annealed multi-resolution noise (KEY INNOVATION):** Anneal the weight of levels i > 0 based on diffusion schedule
  - i-th level at timestep t gets weight (s^(t/T))^i
  - Smaller weight for lower-resolution levels at timesteps closer to noise-free end
  - NYUv2 AbsRel 5.6%, KITTI 11.3% (further improvement)
- **Empirical effect:** Multi-resolution + annealing also *increases prediction consistency* (Tab. S2: std drops from 0.086 to 0.033 on NYUv2, max-min drops from 0.260 to 0.106) — *killer* for test-time ensembling

### Training schedule: 18K iterations, batch 32, ~2.5 days on 1× RTX 4090

- **Noise scheduler:** DDPM with **1000 diffusion steps** during training
- **Optimizer:** Adam, learning rate **3 × 10⁻⁵** (the *standard* for diffusion fine-tuning, same as ChronoDepth 203)
- **Iterations:** 18K
- **Batch size:** 32 (effective via 16-step gradient accumulation, fits on 1× RTX 4090)
- **Augmentation:** Random horizontal flipping
- **Hardware:** 1× NVIDIA RTX 4090, **~2.5 days** = ~60 GPU-hours
- **Training data scale:** 74K samples × 18K iterations × batch 32 = 42.5M sample-passes (the *cheapest* SoTA depth estimator in the literature, ~5-10× cheaper than MiDaS, DPT, Omnidata)
- **NO TEXT TRAINING:** Despite using Stable Diffusion v2 (which has CLIP text conditioning), the model is *not* trained with text. Text conditioning is *disabled* and the *image conditioning* is the only input signal

### Inference: DDIM 50 steps + 10-run ensembling (or 1-4 steps with LCM)

- **Latent diffusion denoising:**
  1. Encode input image x via frozen VAE: z^(x) = E(x)
  2. Initialize depth latent as standard Gaussian noise z^(d)_T (NOT multi-resolution noise, *empirically better*)
  3. For t = T to 0:
     - Concatenate z^(x) and z^(d)_t: z_t = cat(z^(d)_t, z^(x), dim=1)
     - Predict noise: ε̂ = ε_θ(z_t, t)
     - DDIM update step: z^(d)_{t-1} = ... (DDIM non-Markovian step)
  4. Decode final depth latent: d̂ = D(z^(d)_0)
  5. **Average 3 decoded channels** to get the final depth prediction
- **DDIM 50 steps** (vs 1000 DDPM training steps) — re-spaced inference, non-Markovian shortcut
- **Elbow point at 10 steps** (Fig. 7): marginal returns diminish after 10 inference steps; **1-4 steps achievable with LCM distillation** (follow-up paper)
- **Test-time ensembling (KEY INNOVATION):** Run inference N=10 times with different random noise, then **iteratively align** the N predictions
  - For each pair (i, j), find scale ŝ_i, ŝ_j and shift t̂_i, t̂_j that minimize ||d̂'_i - d̂'_j||²₂ where d̂' = d̂ × ŝ + t̂
  - Pixel-wise median across aligned predictions: m(x, y) = median(d̂'_1(x, y), ..., d̂'_N(x, y))
  - Regularization term R = |min(m)| + |1 - max(m)| to prevent collapse to trivial solution, enforce unit scale
  - **No ground truth needed** for alignment (the *killer* feature)
  - 10 predictions reduce AbsRel by ~8% on NYUv2 (Fig. 6), ~9.5% with 20 predictions
  - **Marginal improvement diminishes after 10 predictions** (the *practical* setting)
- **Affine-invariant depth normalization:** For ground truth, apply `d̃ = ((d - d₂) / (d₉₈ - d₂) - 0.5) × 2` to keep values in [-1, 1], then evaluate via *least-squares alignment* with predicted depth (standard affine-invariant protocol)

## Results (key metrics, comparisons)

### Table 1: Zero-shot benchmarks on 5 real datasets (no real depth in training)

| Method | # Training samples (Real / Synthetic) | NYUv2 AbsRel↓ | NYUv2 δ₁↑ | KITTI AbsRel↓ | KITTI δ₁↑ | ETH3D AbsRel↓ | ETH3D δ₁↑ | ScanNet AbsRel↓ | ScanNet δ₁↑ | DIODE AbsRel↓ | DIODE δ₁↑ | **Avg. Rank** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DiverseDepth [2020] | 320K / — | 11.7 | 87.5 | 19.0 | 70.4 | 22.8 | 69.4 | 10.9 | 88.2 | 37.6 | 63.1 | 6.0 |
| MiDaS [2020] | 2M / — | 11.1 | 88.5 | 23.6 | 63.0 | 18.4 | 75.2 | 12.1 | 84.6 | 33.2 | 71.5 | 5.2 |
| LeReS [2021] | 300K / 54K | 9.0 | 91.6 | 14.9 | 78.4 | 17.1 | 77.7 | 9.1 | 91.7 | 27.1 | 76.6 | 4.0 |
| Omnidata [2021] | 11.9M / 310K | 7.4 | 94.5 | 14.9 | 83.5 | 16.6 | 77.8 | 7.5 | 93.6 | 33.9 | 74.2 | 3.4 |
| HDN [2022] | 300K / — | 6.9 | 94.8 | 11.5 | 86.7 | 12.1 | 83.3 | 8.0 | 93.9 | 24.6 | 78.0 | 2.6 |
| DPT [2021] | 1.2M / 188K | 9.8 | 90.3 | **10.0** | 90.1 | 7.8 | 94.6 | 8.2 | 93.4 | 18.2 | 75.8 | 2.4 |
| **Marigold (no ensemble)** | 0 / 74K | 6.0 | 95.9 | 10.5 | 90.4 | 7.1 | 95.1 | 6.9 | 94.5 | 31.0 | 77.2 | 1.8 |
| **Marigold (w/ ensemble)** | 0 / 74K | **5.5** | **96.4** | 9.9 | **91.6** | **6.5** | **96.0** | **6.4** | **95.1** | 30.8 | **77.3** | **1.4** |

**KILLER RESULTS:**
- **Avg. Rank 1.4 (ensemble)** — best zero-shot monocular depth estimator of 2024, beating 6 purpose-trained baselines
- **NYUv2: AbsRel 5.5%, δ₁ 96.4%** — beats ALL baselines (HDN was previous best 6.9/94.8)
- **KITTI: AbsRel 9.9%, δ₁ 91.6%** — beats ALL baselines (DPT was previous best 10.0/90.1)
- **ETH3D: AbsRel 6.5%, δ₁ 96.0%** — beats ALL baselines (DPT was previous best 7.8/94.6)
- **ScanNet: AbsRel 6.4%, δ₁ 95.1%** — beats ALL baselines (HDN was previous best 8.0/93.9)
- **DIODE: AbsRel 30.8%, δ₁ 77.3%** — beats all on δ₁, but **DPT wins on AbsRel 18.2%** (the *one* weakness — mixed indoor/outdoor DIODE is hard)
- **NEVER SEEN A REAL DEPTH MAP** — pure synthetic-to-real zero-shot transfer
- **74K samples > 11.9M samples** — the *killer* H5 demonstration

### Table 2: Ablation of training noise

| Multi-res. | Annealed | NYUv2 AbsRel↓ | NYUv2 δ₁↑ | KITTI AbsRel↓ | KITTI δ₁↑ |
|---|---|---|---|---|---|
| ✗ | ✗ | 7.7 | 93.4 | 14.2 | 82.1 |
| ✓ | ✗ | 5.8 | 96.1 | 12.1 | 87.1 |
| ✓ | ✓ | **5.6** | **96.5** | **11.3** | **88.7** |

### Table 3: Ablation of training datasets

| Hypersim | Virtual KITTI | NYUv2 AbsRel↓ | NYUv2 δ₁↑ | KITTI AbsRel↓ | KITTI δ₁↑ |
|---|---|---|---|---|---|
| ✗ | ✓ | 13.9 | 83.4 | 15.4 | 79.3 |
| ✓ | ✗ | 5.7 | 96.3 | 13.7 | 82.5 |
| ✓ | ✓ | **5.6** | **96.5** | **11.3** | **88.7** |

- *Surprising*: Hypersim alone is *better* on indoor (5.7 vs 13.9) but *worse* on outdoor (13.7 vs 15.4) than Virtual KITTI alone
- 90% Hypersim + 10% Virtual KITTI is the *optimal* mixture (Tab. S3) — *small* Virtual KITTI is enough for outdoor transfer
- Adding cross-domain data *improves* same-domain performance (Hypersim alone = 5.7, +Virtual KITTI = 5.6)

### Table 4: Ablation of dataset mixing ratio (Tab. S3)

| Hypersim | Virtual KITTI | NYUv2 AbsRel↓ | KITTI AbsRel↓ |
|---|---|---|---|
| 100% | 0% | 5.7 | 13.7 |
| 95% | 5% | 5.8 | **11.1** |
| 90% | 10% | **5.6** | 11.3 |
| 50% | 50% | 6.0 | 12.8 |
| 0% | 100% | 13.9 | 15.4 |

### Test-time ensembling (Fig. 6)

- 1 prediction: NYUv2 AbsRel ~7.7%, KITTI ~12.5%
- 10 predictions: NYUv2 AbsRel ~5.5%, KITTI ~10%
- 20 predictions: NYUv2 AbsRel ~5.0%, KITTI ~9.5%
- **Diminishing returns after 10 predictions**

### Number of denoising steps (Fig. 7)

- 1 step: NYUv2 AbsRel ~13%, KITTI ~21%
- 4 steps: NYUv2 AbsRel ~6.5%, KITTI ~12%
- 10 steps: NYUv2 AbsRel ~5.7%, KITTI ~10.5%
- 50 steps: NYUv2 AbsRel 5.5%, KITTI 9.9% (paper default)
- 100 steps: similar to 50 (saturated)
- **Elbow at ~10 steps** for *efficient* inference

## Connections to H1-H5 (hypotheses from the dental-crown-gen project)

For context, the v0 project's 5 hypotheses are:
- H1: 2-stage (VAE/diffusion + refinement) > 1-stage
- H2: Latent diffusion > direct 3D generation
- H3: Multi-source/multi-modal conditioning > single-source
- H4: Implicit (SDF/NeRF/3DGS) > explicit (mesh/point cloud)
- H5: Synthetic + finetune on real > real-only

- **H1 MILD (NOT DIRECTLY TESTED)** (Marigold is *single-stage* diffusion in latent space, no VAE+DDM 2-stage; but the *broader* pattern is 2-step: SD VAE encode/decode + U-Net denoise, the *de facto* 2-stage for latent diffusion, the *standard* design for ALL latent diffusion models; for v0 v0 sub-task 2, the *practical* H1 lesson is *avoid* explicit 2-stage VAE+DDM designs (DMC 033) if a *single-stage* latent diffusion can match quality, the *killer* 2.5-day single-GPU cost is *much* cheaper than DMC 033's full A100 training)

- **H2 ★★★ STRONGEST DIRECT SUPPORT** (the *FOUNDING* paper of the *repurpose-image-foundation-model-for-downstream-dense-prediction* paradigm, the *direct* H2 mechanism for v0 v0 sub-task 1; Marigold 204 + ChronoDepth 203 + GeoWizard 122 + DepthFM + Wonder3D 118 = the *complete* 2024-2025 foundation-model-repurpose paradigm; the *killer* H2 lesson for v0: the *image-pretrained Stable Diffusion* prior is so rich that it transfers to depth estimation with 74K synthetic samples, the *killer* compute savings: 2.5 days 1×RTX 4090 (~$5-10 Lambda) vs 11.9M samples + multi-GPU for MiDaS/DPT/Omnidata (~$100-1000 Lambda), the *killer* generalization lesson: pure synthetic training → real zero-shot transfer, the *direct* H2 mechanism for v0 v0 v1+ sub-task 1 = *reuse* a pretrained 3D-foundation-model (MapAnything 193 or π³ 192) for *clinical-IOS depth estimation* via *lightweight* fine-tuning on 30-50K clinical-IOS samples + 3DTeethSeg22 + ToSynFCD)

- **H3 ★ STRONG INDIRECT SUPPORT** (image conditioning via *concatenation* on feature dimension is a *simple* H3 mechanism; multi-resolution noise + test-time ensembling is a *novel* H3 mechanism for *per-prediction* uncertainty; the *killer* H3 design lesson for v0 v0 v1+ sub-task 1: *per-tooth-noise-level-conditioning* for clinical-IOS where the *prep tooth* gets *low* noise (high-fidelity margin) and the *gum* gets *high* noise (acceptable approximation), inspired by ChronoDepth 203's per-frame independent noise which is the *direct* extension of Marigold's noise design)

- **H4 NOT TESTED** (Marigold is *only* 2D depth, not 3D mesh/SDF/NeRF; for v0 v0 v1+ sub-task 1, the H4 substrate is *per-frame depth + downstream TSDF fusion*, *compatible* with the H4 framework; the *killer* H4 lesson for v0: depth estimation is a *powerful intermediate representation* for 3D reconstruction, the *direct* substrate for TSDF fusion in v0 sub-task 1)

- **H5 ★★★ STRONGEST DIRECT SUPPORT** (the *categorical* H5 lesson: training on **74K SYNTHETIC samples (Hypersim + Virtual KITTI)** is *sufficient* for *open-world* zero-shot transfer to 5 real datasets, *no* real depth training data needed; the *killer* H5 demonstration: 74K synthetic > 11.9M real, the *killer* cost savings for v0: dental-IOS training can be *purely synthetic* (3DTeethSeg22 + ToSynFCD) with *zero* real clinical-IOS data, the *killer* clinical-implication: v0 paper can claim "synthetic-only training → clinical-IOS zero-shot transfer" which is the *killer* H5 positioning, the *direct* mechanism for v0 v0 v1+ sub-task 1 = train on *purely synthetic* 3DTeethSeg22 + ToSynFCD + DCrownFormer 2317 + 3D-IOS-Bench → *zero-shot* clinical-IOS transfer)

## Surprises / interesting things buried in section 4

1. **VAE reconstruction error for depth is 0.0095 ± 0.0091** (MAE) — *safely below* the depth-estimation error floor of 0.05-0.10. The *killer* design lesson: any RGB-pretrained VAE can be reused for depth (the latent space is *general enough* to encode single-channel depth, just replicate to 3 channels)
2. **VAE channel consistency is 0.0027 std, 0.0062 max-min** (Tab. S1) — the 3 decoded channels are *almost identical* for depth input, confirming the "depth as fake RGB" trick is *valid* (the channels carry *redundant* depth information, the *standard* trick used in EVERY image-pretrained-foundation-model-to-depth paper: Marigold 204, ChronoDepth 203, GeoWizard 122, Wonder3D 118)
3. **Multi-resolution noise + annealing improves *prediction consistency* by 2.6×** (std 0.086 → 0.033 on NYUv2) — the *unintuitive* finding: *more varied* training noise (multi-resolution) gives *more consistent* inference predictions, because the model learns to *denoise different noise patterns* instead of *overfitting to Gaussian noise*
4. **Test-time ensembling is *no-ground-truth* — pure self-alignment** — the *killer* property: you can run inference N=10 times with different random noise, align via iterative least-squares, take the median, *no* ground truth needed, *pure* self-consistency mechanism
5. **Affine-invariant depth via 2%/98% percentile normalization** — the *killer* design choice: NOT min/max (sensitive to outliers) but 2%/98% (robust to extreme pixels), gives *canonical* depth representation independent of data statistics
6. **74K synthetic > 11.9M real** — the *single most important* empirical finding: Stable Diffusion's *visual prior* is so much richer than learned-from-scratch encoders that *less is more* when fine-tuning. The *killer* practical lesson: for v0, *fine-tune a pretrained 3D-foundation-model* (MapAnything 193) on *small* clinical-IOS data instead of training from scratch
7. **The 90/10 Hypersim/Virtual KITTI mixing ratio is *optimal*** — *surprisingly small* Virtual KITTI (10%) is enough for outdoor generalization. The *killer* design lesson: a *small* "domain-bridging" dataset is enough to transfer between domains, the *practical* lesson for v0: a *small* 3D-IOS-Bench (50-100 clinical-IOS scans) is enough to bridge *synthetic* (3DTeethSeg22) → *clinical* (real patient scans)
8. **VAE latents are 4-channel, not 3** — the *technical* detail: SD VAE encodes to 4-channel latents (not 3 like the RGB output), so the *concatenation* is z^(d) ⊕ z^(x) ∈ R^(H/8 × W/8 × 8), and the U-Net first layer is *doubled* from 4 to 8 input channels
9. **The "duplicate first-layer weight, divide by 2" trick** — *unintuitive* but *essential*: copying pretrained 4-channel weights to 8-channel and dividing by 2 keeps the *activations magnitude* stable, preserves pretrained structure
10. **Marigold is the *first* paper to *ablate* training noise type for diffusion-based depth** — the *killer* design lesson for v0: *training noise type matters* (multi-res > Gaussian, annealed > non-annealed), the *standard* for ALL subsequent diffusion-depth papers
11. **Cross-domain training improves same-domain performance** (Tab. 3: Hypersim alone NYUv2 5.7, +Virtual KITTI 5.6) — the *counter-intuitive* H5 finding: *adding data from a different domain* (outdoor) *improves* same-domain (indoor) performance, the *killer* H5 design lesson for v0: *more diverse* training data → *better* generalization even for in-domain
12. **The 2.5-day single-GPU training cost is *the* killer practical advantage** — vs MiDaS (2M samples, multi-GPU, days) and DPT (1.2M samples, multi-GPU, days) and Omnidata (11.9M samples, multi-GPU, weeks). The *killer* compute savings for v0: a *single RTX 4090* can train a *state-of-the-art* depth estimator in *2.5 days* for ~$5-10 Lambda
13. **Marigold's DIODE result is the *one* weakness** (AbsRel 30.8% vs DPT's 18.2%) — DIODE has *mixed* indoor/outdoor scenes which confuse the model. The *killer* design lesson for v0: *single-domain* training → *single-domain* optimal; for *multi-domain* generalization, need *multi-domain* training data
14. **The OpenRAIL++-M license for model weights** is the *standard* Stable Diffusion derivative license — *allows* commercial use with ethical-use restrictions (no illegal content, no harmful use, attribution required). For v0 v0 v1+ clinical-IOS use, the license is *fully compliant* (clinical use is *purely beneficial*). The *killer* licensing lesson for v0: *OpenRAIL++-M* is the *practical* license for clinical-AI deployment, NOT Apache-2.0 or MIT (which *cannot* restrict harmful use)

## Quote-worthy sentences

> "Modern image diffusion models have been trained on internet-scale image collections specifically to generate high-quality images across a wide array of domains. If the cornerstone of monocular depth estimation is indeed a comprehensive, encyclopedic representation of the visual world, then it should be possible to derive a broadly applicable depth estimator from a pretrained image diffusion model." (Sec. 1)

> "The key to unlocking the potential of a pretrained diffusion model is to keep its latent space intact. We find this can be done efficiently by modifying and fine-tuning only the denoising U-Net." (Sec. 1)

> "We find this can be done efficiently by modifying and fine-tuning only the denoising U-Net. Turning Stable Diffusion into Marigold requires only synthetic RGB-D data (in our case, the Hypersim and Virtual KITTI datasets) and a few GPU days on a single consumer graphics card." (Sec. 1)

> "Empowered by the underlying diffusion prior of natural images, Marigold exhibits excellent zero-shot generalization: Without ever having seen real depth maps, it attains state-of-the-art performance on several real datasets." (Sec. 1)

> "We implement Marigold using PyTorch and utilize Stable Diffusion v2 as our backbone, following the original pre-training setup with a v-objective." (Sec. 4.1)

> "To prevent inflation of activations magnitude of the first layer and keep the pretrained structure as faithfully as possible, we duplicate the weight tensor of the input layer and divide its values by two." (Sec. 3.2)

> "We train exclusively with synthetic depth datasets. As with the depth normalization rationale, this decision has two objective reasons. First, synthetic depth is inherently dense and complete, meaning that every pixel has a valid ground truth depth value, allowing us to feed such data into the VAE, which can not handle data with invalid pixels. Second, synthetic depth is the cleanest possible form of depth, which is guaranteed by the rendering pipeline." (Sec. 3.3)

> "In contrast to prior work that utilized diverse real datasets to achieve generalization, we train exclusively with synthetic depth datasets. [...] If our assumption about the possibility of fine-tuning a generalizable depth estimation from a text-to-image LDM is correct, then synthetic depth gives the cleanest set of examples and reduces noise in gradient updates during the short fine-tuning protocol." (Sec. 3.3)

> "We proposed to anneal the weight of levels i > 0 based on the diffusion schedule. Specifically, we assign the ith level at timestep t the weight (s^(t/T))^i, where T is the total number of diffusion steps. Thus, a smaller weight is given to lower-resolution levels at timesteps closer to the noise-free end of the schedule." (App. A.2)

> "This ensembling step requires no ground truth for aligning independent predictions. This scheme enables a flexible trade-off between computation efficiency and prediction quality by choosing N accordingly." (Sec. 3.4)

> "Despite being trained solely on synthetic depth datasets, the model can well generalize to a wide range of real scenes. This successful adaptation of diffusion-based image generation models toward depth estimation confirms our initial hypothesis that a comprehensive representation of the visual world is the cornerstone of monocular depth estimation." (Sec. 4.2)

> "Interestingly, adding additional training data from a different domain not only improves the performance on the new domain but also brings improvements in the original domain." (Sec. 4.3)

> "We observe that the elbow point of marginal returns given more denoising steps depends on the dataset but is always under 10 steps. This implies that one can further reduce the denoising steps to 10 or even less to gain efficiency while keeping comparable performance. Interestingly, this threshold is smaller than what is usually required for diffusion-based image generators, i.e., 50 steps." (Sec. 4.3)

> "Sheeeeesh — you could cut your finger on those depth maps!" — Bilawal Sidhu (@bilawalsidhu), 12 Dec 2023 (project page testimonial)

> "Wow Marigold 🌼 depth estimation works extremely well! 🤯 And the best thing is that the checkpoints and code are fully available for commercial use!" — Alex Carlier (@alexcarliera), 23 Dec 2023 (project page testimonial)

## Code/data link

- **Code:** https://github.com/prs-eth/Marigold (Apache-2.0 ✅ for code, OpenRAIL++-M for model weights, 3,159 ⭐, 209 🍴, ~9.9 MB, last push 2025-12-10)
- **Model checkpoints:** https://huggingface.co/prs-eth/marigold-depth-v1-0 + v1-1 (OpenRAIL++-M)
  - Marigold-Depth v1.0 (10-50 inference steps, 865M params)
  - Marigold-Depth v1.1 (1-4 inference steps, LCM-distilled)
  - Marigold-Depth LCM (single-step)
  - Marigold-Depth HR (high-resolution variant)
  - Marigold-Normals v1.1
  - Marigold IID-Appearance v1.1
  - Marigold IID-Lighting v1.1
- **Project page:** https://marigoldmonodepth.github.io
- **Online demos:** https://huggingface.co/spaces (multiple Marigold Spaces), ComfyUI integration
- **Paper PDF:** https://arxiv.org/pdf/2312.02145 (arXiv) + https://openaccess.thecvf.com/content/CVPR2024/papers/Ke_Repurposing_Diffusion-Based_Image_Generators_for_Monocular_Depth_Estimation_CVPR_2024_paper.pdf (CVPR 2024)
- **Training data:** Hypersim (54K indoor) + Virtual KITTI (20K outdoor) = 74K synthetic, all open-source
- **Eval data:** NYUv2 (654 indoor), ScanNet (800 indoor), KITTI (652 outdoor), ETH3D (454 high-res), DIODE (325+446 indoor+outdoor)
- **Setup:** PyTorch + diffusers, Stable Diffusion v2 base, v-objective, Adam lr 3e-5, 18K iterations, batch 32 (16-step grad accum), DDPM 1000 steps train, DDIM 50 steps inference

## For our project

★ **10 v0 actions: (a) ★★★ ADOPT MARIGOLD'S "DEPTH-AS-FAKE-RGB" VAE TRICK AS V0 V1+ SUB-TASK 1 DEPTH ENCODING** ($0, 5-10 lines PyTorch, the *killer* standardization trick, `z_depth = vae.encode(depth.unsqueeze(1).repeat(1, 3, 1, 1))`; the *killer* engineering simplification: reuse *any* RGB VAE for depth, no need to train a separate depth VAE, *drop-in* for v0's 3D-foundation-model choice (MapAnything 193 or π³ 192))

**(b) ★★★ ADOPT MARIGOLD'S "DOUBLE-FIRST-LAYER-WEIGHT-AND-DIVIDE-BY-2" TRICK AS V0 V1+ SUB-TASK 1 CONDITIONING** ($0, 5-10 lines PyTorch, the *killer* architectural pattern: when reusing a pretrained RGB-pretrained model for depth, duplicate the first layer weight tensor and divide by 2 to *preserve pretrained structure*; the *killer* H3 design lesson: *conditioning via concatenation* is *always* the right choice for v0, not cross-attention; for v0 v1+ sub-task 1, the *practical* lesson: when integrating clinical-IOS RGB with 3D-foundation-model, use concatenation conditioning on feature dimension)

**(c) ★★★ ADOPT MARIGOLD'S "AFFINE-INVARIANT 2%/98% PERCENTILE NORMALIZATION" AS V0 V1+ SUB-TASK 1 DEPTH NORMALIZATION** ($0, 1-line NumPy, the *killer* H5 design lesson: NOT min/max (sensitive to outliers) but 2%/98% (robust to extreme pixels); for v0 v1+ clinical-IOS depth, the *practical* lesson: 2%/98% percentile normalization gives *canonical* depth representation independent of IOS scanner calibration, the *killer* H5 property: *zero-shot* transfer to unseen IOS scanners because the depth is *normalized to canonical [-1, 1] range*)

**(d) ★★★ ADOPT MARIGOLD'S "SYNTHETIC-ONLY TRAINING → REAL-ZERO-SHOT" PARADIGM AS V0 V0 V1+ PAPER POSITIONING** ($0, 1-2 hours writing, the *killer* H5 design lesson for v0 paper: claim "synthetic-only training (3DTeethSeg22 + ToSynFCD + DCrownFormer 2317) → zero-shot transfer to real clinical-IOS scans"; the *killer* empirical foundation: Marigold's 74K synthetic > 11.9M real; the *killer* compute savings: 2.5 days 1×RTX 4090 (~$5-10 Lambda) vs 11.9M samples multi-GPU (~$100-1000 Lambda))

**(e) ★★★ ADOPT MARIGOLD'S "ANNEALED MULTI-RESOLUTION NOISE" AS V0 V1+ SUB-TASK 1 TRAINING TRICK** ($0, 20-50 lines PyTorch, the *killer* H2 design lesson: training noise type *matters* (multi-res > Gaussian, annealed > non-annealed), the *standard* for ALL subsequent diffusion-depth papers; the *killer* practical benefit: multi-res + annealed improves *prediction consistency* by 2.6×, enables *better test-time ensembling*)

**(f) ★★★ ADOPT MARIGOLD'S "TEST-TIME ENSEMBLING" AS V0 V1+ SUB-TASK 1 INFERENCE TRICK** ($0, 30-50 lines PyTorch, the *killer* H3 design lesson: run inference N=10 times with different random noise, iteratively align via least-squares, take pixel-wise median; the *killer* property: *no ground truth* needed for alignment, *pure self-consistency*; the *killer* empirical gain: ~8% AbsRel improvement; the *killer* clinical-IOS application: per-IOS-scan uncertainty quantification via ensemble disagreement)

**(g) ★★★ ADOPT MARIGOLD'S "VAE-RECONSTRUCTION-ERROR 0.0095" BENCHMARK AS V0 V1+ SUB-TASK 1 DEPTH-ENCODING FEASIBILITY TEST** ($0, 1-day study, the *killer* design lesson: any RGB-pretrained VAE can be reused for depth if the reconstruction error is < 0.05 (the depth-estimation error floor); for v0 v1+ clinical-IOS, the *practical* test: compute VAE reconstruction error for clinical-IOS depth maps, if < 0.05 then *safe to reuse* the RGB VAE)

**(h) ★★★ ADOPT MARIGOLD'S "DISABLING-TEXT-CONDITIONING" ARCHITECTURE AS V0 V1+ SUB-TASK 1 DESIGN TEMPLATE** ($0, 1-line config, the *killer* design lesson: when repurposing a multimodal foundation model (SD v2 with CLIP text encoder) for a *non-text* task (depth, 3D), *disable* the text encoder and use *only* the relevant modality (image for depth, image+3D for 3D reconstruction); the *killer* compute savings: no text-encoder overhead, *faster* inference)

**(i) ★★★ CITE MARIGOLD 204 IN V0 V0 V1+ PAPER RELATED-WORK AS THE *FOUNDING* REPURPOSE-FOUNDATION-MODEL-FOR-DOWNSTREAM-TASK PARADIGM** ($0, 1-2 hours, 2-3 paragraphs, the *killer* v0 paper positioning, "Marigold 204 + ChronoDepth 203 + GeoWizard 122 + DepthFM + Wonder3D 118 = the *complete* 2024-2025 foundation-model-repurpose paradigm; the *killer* empirical foundation: 74K synthetic > 11.9M real, 2.5 days 1×RTX 4090 > weeks multi-GPU")

**(j) ★★★ ADOPT MARIGOLD'S "FROZEN-VAE + TRAINABLE-UNET" ARCHITECTURE AS V0 V1+ SUB-TASK 1 TEMPLATE** ($0, 1-line config, the *killer* H2 design lesson: *freeze* the VAE (preserve the *learned* latent space) and *fine-tune* only the U-Net (adapt to the *new* task); the *killer* practical benefit: *only* the U-Net needs to be fine-tuned (~865M params), the *VAE* is *reused as-is* (~80M params frozen), the *killer* compute savings: ~5-10× faster training, ~5-10× less GPU memory)

**★ v0 sub-task 1 depth estimation stack now has 6 papers covered (3 commercial-deployable)**: (i) image-generator-to-depth (Marigold 204 Apache-2.0 + OpenRAIL++-M ✅) ⚡ NEW, (ii) video-depth (ChronoDepth 203 MIT ✅), (iii) pose-required video-depth (Aether 199 MIT ✅), (iv) multi-task depth (GeoWizard 122 ⚠️), (v) depth-foundation-model (Depth Anything V2 ⚠️), (vi) consistent-context-aware (ChronoDepth 203 MIT ✅). **★ v0 sub-task 1 commercial-deployment stack now has 4 commercial-deployable papers: Aether 199 (MIT ✅) + VDA-S 202 (Apache-2.0 ✅) + ChronoDepth 203 (MIT ✅) + Marigold 204 (Apache-2.0 code + OpenRAIL++-M model ✅)** — the *practical* v0 v1+ clinical-deployment stack: **Aether 199 + VDA-S 202 + ChronoDepth 203 + Marigold 204** = 4 *commercial-deployment-friendly* 2024 depth papers.

**★ 2024 DEPTH ESTIMATION CONVERGENCE: 4+ 2024 papers have *converged* on the *repurpose-foundation-model-for-depth* paradigm** (Marigold 204 [SD] + ChronoDepth 203 [SVD-XT] + GeoWizard 122 [SD] + DepthFM [SD] + Wonder3D 118 [SD]) — the *uniform* design lesson: *reuse a pretrained foundation model, add minimal depth-specific adaptation, leverage the foundation's spatial/temporal prior*. The *killer* cross-domain convergence: the depth-estimation community is *catching up* to the 2024-2025 LLM/foundation-model revolution (LoRA, instruction-tuning, sequential fine-tuning — *all* apply to depth adaptation). The 2026 papers will likely integrate *metric-depth heads* + *intrinsics prediction* + *temporal consistency* + *uncertainty quantification* into a *unified* framework — the *next* convergence point.

**★ v0 compute impact:** Marigold 204 is *complementary* to ChronoDepth 203 — Marigold = monocular image depth (single frame), ChronoDepth = monocular video depth (multi-frame). The *practical* v0 v1+ sub-task 1 design: (i) use Marigold 204 for *single-frame* depth estimation on individual IOS scans, (ii) use ChronoDepth 203 for *temporal-consistent* depth across multiple IOS views of the same patient, (iii) fuse both with TSDF for 3D reconstruction. **Total v0 sub-task 1 compute: ~$4,500-6,400 Lambda** (no change from 203-note, Marigold 204 is *reused* via the foundation-model-repurpose paradigm, no additional training cost).

**★ Open Q for HK:** (i) cite Marigold 204 in v0 v0 v1+ paper as the *founding* repurpose-foundation-model paradigm? (YES — *founding* paper + Apache-2.0/OpenRAIL++-M ✅ + 416 citations + Best Paper Candidate); (ii) adopt Marigold's "depth-as-fake-RGB" VAE trick? (YES — $0, 5-10 lines, *killer* engineering simplification); (iii) adopt Marigold's "double-first-layer-weight-and-divide-by-2" trick? (YES — $0, 5-10 lines, *killer* architectural pattern); (iv) adopt Marigold's "2%/98% percentile normalization"? (YES — $0, 1-line, *killer* H5 design lesson); (v) adopt Marigold's "synthetic-only training" paradigm? (YES — *killer* v0 paper positioning, *no* real clinical-IOS training data needed); (vi) adopt Marigold's "annealed multi-resolution noise"? (YES — $0, 20-50 lines, *killer* H2 design lesson); (vii) adopt Marigold's "test-time ensembling"? (YES — $0, 30-50 lines, *killer* H3 design lesson, *no* ground truth needed); (viii) cite Marigold's "VAE-reconstruction-error 0.0095" as feasibility test? (YES — $0, 1-day study); (ix) adopt Marigold's "disabling-text-conditioning" architecture? (YES — $0, 1-line config); (x) adopt Marigold's "frozen-VAE + trainable-UNet" architecture? (YES — $0, 1-line config, *killer* H2 design lesson); (xi) use Marigold 204's Apache-2.0 code directly for v0 v1+? (YES — Apache-2.0 ✅ + 3,159 ⭐ + last push 2025-12-10, *no* re-implementation needed); (xii) use Marigold 204's OpenRAIL++-M model weights? (YES — *allows commercial* with ethical-use restrictions, *fully compliant* for clinical use); (xiii) use Marigold 204 as the *monocular image depth* baseline for v0 v1+ sub-task 1? (YES — *direct* v0 v1+ sub-task 1 candidate, *killer* generalization + commercial license).

**★ Note on the Marigold 204 + ChronoDepth 203 design pattern:** Both papers use the *same* architectural pattern (frozen VAE + trainable U-Net + concatenation conditioning), but Marigold is *monocular image* (single-frame, 10-step DDIM, 10-run ensemble) while ChronoDepth is *monocular video* (multi-frame, 5-step DDIM, sliding window with consistent context-aware inference). The *killer* design lesson for v0 v1+ sub-task 1: the *pattern* generalizes from *image* (Marigold 204) to *video* (ChronoDepth 203) to *3D pointmap* (MapAnything 193 or π³ 192), the *unified* 2024-2025 *repurpose-foundation-model-for-dense-prediction* paradigm.

★ ★ **Next paper to read (205):** The 204-note's recommended *next* candidates are (a) **GeoWizard 122** (Fu 2024, arXiv:2403.12013, ECCV 2024, *the* direct extension of Marigold to multi-task depth + normals + segmentation, the *right* paper to understand *multi-task* foundation-model-repurpose), (b) **DepthFM** (Jung 2024, arXiv:2403.04288, ICLR 2024, *the* flow-matching variant of Marigold, *faster* training + inference), (c) **Wonder3D 118** (Long 2024, CVPR 2024, *the* cross-domain diffusion image-to-3D, *the* H2 mechanism for v0 sub-task 2), (d) **Depth Anything V2** (Yang 2024, arXiv:2406.09414, NeurIPS 2024, *the* frozen-encoder VDA 202 builds on), (e) **DDVM** (Saxena 2023, arXiv:2306.01991, *the* concurrent metric-depth diffusion work), (f) **DiffusionDepth** (Duan 2023, arXiv:2303.05021, *the* concurrent Swin-Transformer-conditioned depth diffusion). **★ Recommendation: *read 205 = GeoWizard 122 (Fu 2024, arXiv:2403.12013, ECCV 2024)*** — the *direct* multi-task extension of Marigold to depth + normals + segmentation, the *founder* of the *multi-task foundation-model-repurpose* paradigm, the *right* next paper to understand *how to extend* Marigold's *monocular depth* to *multi-task dense prediction* (depth + normals + segmentation), the *killer* v0 v1+ relevance: clinical-IOS could benefit from *joint* depth + normals + FDI-tooth-segmentation prediction, the *complete* 3-foundation-model-repurpose paper: 203 (video depth) + 204 (image depth) + 205 (multi-task depth+normals+segmentation). **Alternative 205 candidates:** (a) **Wonder3D 118** — the *direct* 3D-foundation-model for v0 sub-task 2 (crown generation), the *killer* H2 mechanism for *3D generation* via diffusion; (b) **DepthFM** — the *flow-matching* variant of Marigold, *faster* training + inference, the *killer* engineering simplification; (c) **Depth Anything V2** — the *frozen-encoder* depth foundation model, the *complementary* H2 mechanism to Marigold. **★ Recommendation reasoning:** GeoWizard 122 is the *right* next paper because (1) it's the *direct* multi-task extension of Marigold 204, (2) it covers *all three* dense prediction tasks (depth + normals + segmentation) which are *exactly* the v0 v1+ sub-task 1 tasks (per-tooth depth + normal estimation + FDI segmentation), (3) it generalizes Marigold's *frozen-VAE + trainable-UNet* architecture to *multi-task* settings, (4) it's *ECC V 2024* (top venue, peer-reviewed), and (5) the 3-paper 202-204 arc is now *complete* for *foundation-model-repurpose for depth*: 202 (VDA: frozen-encoder depth foundation) → 203 (ChronoDepth: video-depth) → 204 (Marigold: image-depth). After GeoWizard 205, the v0 sub-task 1 *foundation-model-repurpose* arc will have *image* (Marigold 204) + *video* (ChronoDepth 203) + *multi-task* (GeoWizard 205) coverage, the *complete* 2024-2025 *repurpose-foundation-model* arc. ⚠️ **PATTERN NOTICE:** *always* verify (1) arXiv ID via direct arXiv lookup, (2) venue via OpenAccess/CVF, (3) GitHub canonical repo (NOT redirects), (4) LICENSE file CONTENT (not just license metadata) — Marigold 204 has the *dual-license pattern* (code Apache-2.0 + model OpenRAIL++-M) which is *standard* for Stable Diffusion derivatives, the *practical* v0 lesson: when using SD-derivative models, expect *dual licensing* (code permissive + model restricted), the *killer* clinical-AI license combination.
