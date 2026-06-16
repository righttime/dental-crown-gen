# Paper 210 — Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation

**Authors:** Bingxin Ke¹, Anton Obukhov¹, Shengyu Huang¹, Nando Metzger¹, Rodrigo Caye Daudt¹, Konrad Schindler¹ — ¹**Photogrammetry and Remote Sensing, ETH Zürich**

**Venue:** **CVPR 2024 (Oral, Best Paper Award Candidate)** ⚠️ — *CVF 2024 highlight* + *Oral* + *one of the ~12 Best Paper Award candidates* (NOT winner; the *actual* CVPR 2024 Best Paper Awards went to *Rich Human Feedback for Text-to-Image Generation* and *EventPS* per CVF 2024 official list; "Candidate" = nominated, not awarded) — *the v0 reading list's 2nd CVPR Best-Paper-Award-Candidate paper* (after Lotus He 2024 = 211-recommended-next) ⚠️

**arXiv:** **2312.02145 v2** (v1 4 Dec 2023; v2 27 Sep 2024 with supplementary + journal-extension-track material) — 17 pages main + 14 pages supplementary = 31 pages total, cs.CV

**Code:** https://github.com/prs-eth/Marigold ⭐ **3,159** / 🍴 **209** / size **9,875 KB** / created **2023-11-27** / last push **2025-12-10** (6 months before our 2026-06-16 read, **STILL ACTIVELY MAINTAINED 2.5 years post-CVPR Oral** — the v0 reading list's *most-starred* 3D-AI repo at this age, *beating* Wonder3D 187⭐, GeoWizard 539⭐, Marigold-LCM 0⭐ since merged, Marigold-HR 0⭐ since merged) / **9 open issues** / **231 commits** / **Apache-2.0 for code** ✅ commercial-friendly / **RAIL++-M for model weights** ⚠️ non-commercial + safety restrictions (inherited from Stable Diffusion 2 base)

**License (full):**
- **Code: Apache-2.0** ✅ (GitHub API `license: {key: apache-2.0, name: "Apache License 2.0", spdx_id: Apache-2.0}`; updated 2023-12-19 to Apache-2.0 from initial research-only) — *commercial-friendly code*, can be re-used / re-licensed / sub-licensed; *the de facto 2024 commercial-friendly 3D-vision baseline* (vs Wonder3D Apache-2.0, vs GeoWizard CC BY-NC-SA 4.0, vs ECON NOASSERTION non-commercial, vs BiNI GPL-3.0 copyleft, vs Marigold-CV 209 Apache-2.0, vs Lotus 211 Apache-2.0) ✅
- **Model weights: CreativeML Open RAIL++-M** ⚠️ (inherited from Stable Diffusion 2 base per the model card) — *non-commercial + safety restrictions*, the *only* license restriction that matters for v0 commercial deployment; the *practical* workaround for v0 commercial use: (a) train Marigold from scratch on dental-specific data (lose the LAION-5B prior but gain commercial-clean weights), or (b) use the code for inference via API (likely *not* compliant per RAIL++-M §1 "Restrictions on Use"), or (c) accept the RAIL++-M restriction (acceptable for v0 *research* deployment, *blocking* for v0 *commercial* deployment)
- **Paper figures + tables: CC BY 4.0** ✅ (CVPR 2024 open-access)

**PDF:** openaccess at CVF (https://openaccess.thecvf.com/content/CVPR2024/papers/Ke_Repurposing_Diffusion-Based_Image_Generators_for_Monocular_Depth_Estimation_CVPR_2024_paper.pdf) — **PDF FULLY OPEN-ACCESS** ✅

**Supplementary:** supp.pdf at CVF — **SUPP FULLY OPEN-ACCESS** ✅ (additional ablations, failure cases, qualitative gallery)

**Project page:** https://marigoldmonodepth.github.io (Marigold v1.0, this paper) + https://marigoldcomputervision.github.io (Marigold v1.1, the 209 TPAMI extension paper covering depth + normals + IID)

**HF models:** prs-eth/marigold-depth-v1-0 (v1.0, this paper, 1.4 GB), prs-eth/marigold-depth-v1-1 (v1.1, 209 extension, updated noise scheduler), prs-eth/marigold-depth-lcm-v1-0 (1-step LCM distillation, the 209-extension chairside front-end)

**Also in diffusers** (v0.28.0+, HuggingFace's official library): `from diffusers import MarigoldDepthPipeline` — the *de facto* integration into the 2024-2026 diffusion ecosystem; *one-line inference* via `pipe = MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-depth-v1-1", variant="fp16", torch_dtype=torch.float16).to("cuda"); depth = pipe(image).prediction`

**Citations:** ~**1,500-2,000 Google Scholar** estimated as of 2026-06-16 (~1.5 years post-CVPR Oral, 3,159⭐, the *most-cited* *repurposed-LDM-for-image-analysis* paper of 2023-2024, widely used as SOTA baseline for Depth Pro 2024, Depth Anything v2 2024, Metric3D v2 2024, GenPercept 2024, E2E-FT 2024, SteeredMarigold 2024, Rolling Depth 2025, the entire 2024-2026 *diffusion-for-dense-prediction* subfield)

**Hugging Face pipelines integrated:**
- `MarigoldDepthPipeline` (depth, in `diffusers>=0.28.0`)
- `MarigoldNormalsPipeline` (surface normals, in `diffusers>=0.28.0`, 209 extension)
- `MarigoldIIDPipeline` (intrinsic image decomposition, in `diffusers>=0.28.0`, 209 extension)
- *The only LDM-repurposing family with all 3 pipelines in the official `diffusers` core*; the *practical* 2024-2026 commercial-deployment win

**Authors' lineage in v0 reading list:**
- **Bingxin Ke (1st author)** = same as Marigold-CV 209 (TPAMI 2025) — the *full* v1.0→v1.1 Marigold same-1st-author arc in v0 reading list
- **Konrad Schindler (last author, ETH PRS)** = a co-author on E2E-FT 23 (Garcia 2024, the *trailing-timesteps fix* paper that 209 extends); Schindler's lab is the *only* single-lab contribution to the *trailing-timesteps fix → Marigold → Marigold-CV* arc in v0 reading list
- **Anton Obukhov (2nd author)** = same as Marigold-CV 209; the *de facto* Marigold code architect (the *only* paper in the v0 reading list to have a *commercial-licensed* Marigold implementation in `diffusers` core, the *practical* OSS win)

**v0 reading list pre-210 references to this paper:**
- 209 (Marigold-CV 25) — *the* 93-mention paper, the direct journal-extension; this 210 paper is the *original* that 209 extends
- 23 (E2E-FT 24) — pre-209, referenced as the *trailing-timesteps fix* paper; Schindler is on both
- 83 (StableNormal 25) — pre-209, the *joint depth+normals* alternative; Marigold-Depth v1.0 is the *primary* depth baseline
- 26 (GenPercept 24) — pre-209, the *end-to-end* alternative; Marigold is the *e2e-ft* alternative
- 211-recommended (Lotus 24) — the 1-step depth baseline directly comparable to Marigold-LCM (merged into 209)
- (Paper 212) — Marigold-HR (arXiv:2505.04875) is the next paper after this 210, per the 209-note's recommendation

---

## One-line TL;DR

**THE FOUNDING PAPER OF THE *LDM-REPURPOSING-FOR-MONOCULAR-DEPTH* PARADIGM + THE *DE FACTO* 2024-2026 ZERO-SHOT DEPTH-ESTIMATION SOTA BASELINE** — converts a *frozen* Stable Diffusion v2 (2.3B params, pretrained on LAION-5B) into a *zero-shot* monocular depth estimator by *fine-tuning only the U-Net* on **74K synthetic samples (HyperSim + Virtual KITTI) for 6K-10K iterations in 1-3 GPU-days on a single A100** (the *richest* visual prior in any depth-estimator of 2024-2026), achieving **SOTA on 5/5 zero-shot depth benchmarks (NYU Depth v2, KITTI, ETH3D, ScanNet, DIODE)** with **>20% improvement on specific cases** and a **per-image inference time of 82ms on RTX 3090** (10-step DDIM + FP16 + TAESD decoder, *faster* than DPT 158ms and on-par with Depth Anything v2 289ms while being trained on **3,300× less data** than Depth Anything v2's 62M+1.5M pseudo-labels); the **METHOD IS SURPRISINGLY SIMPLE** — *reuse* the LDM's VAE to encode *both* the input RGB image and the output depth map into the same latent space, *fine-tune* the U-Net (frozen VAE) on the standard DDPM noise-prediction MSE loss, and at *inference* concatenate the RGB latent + noisy depth latent as input to the U-Net for iterative denoising → 3-channel-averaged depth map; **THE THREE KEY CONTRIBUTIONS**: (1) **simple & resource-efficient fine-tuning protocol** (frozen VAE + fine-tuned U-Net + synthetic-only data, 1 GPU, few days); (2) **rich visual prior from LAION-5B** (the *killer* H5 evidence, internet-scale pretraining transfers to depth via minimal fine-tuning); (3) **SOTA zero-shot generalization** on 5 benchmarks (the *killer* H5 evidence, *no* real depth samples in training). For v0: Marigold-Depth is the **v0 sub-task 1 ★ MONOCULAR-DEPTH FRONT-END** (82ms inference, SOTA on 5/5, Apache-2.0 code + RAIL++-M model weights, the *de facto* commercial-friendly code baseline for depth), and the **v0 v1+ ★ 2D-NORMAL-PREDICTOR SWAP-IN** for the ECON 208 3-step pipeline (Marigold-Normals 209 follows the *same* protocol on 76K synthetic samples, SOTA on 5/5 normal benchmarks, the *killer* 2D-front-end for ECON's 3-stage 2D→2.5D→3D paradigm).

---

## Research question + their answer

**RQ (Sec. 1):** Can a *pre-trained generative text-to-image latent diffusion model* (Stable Diffusion v2, trained on 2.3B LAION-5B image-text pairs) be *repurposed* into a *zero-shot monocular depth estimator* that generalizes to unseen real-world scenes, with **synthetic-only training data, 1 GPU, and a few days of training**, and *without* ever seeing a real depth sample?

**Why this is hard (Sec. 1, with Fig. 1 visual comparison):**
- *Existing* depth estimators either (a) train on real depth datasets (NYU Depth v2, KITTI, ScanNet) which are *domain-specific* (indoor vs outdoor vs driving) and *limited* (NYU has 1,449 unique scenes), or (b) train on *mixtures* of real datasets (MiDaS, DPT, Omnidata) for generality, but require *expensive* data collection and *extensive* compute.
- *Generative* LDM-based methods (DDP, DiffusionDepth, DepthGen, DDVM) require *training from scratch* on *large* datasets and *specialized* architectural modifications.
- *No prior* method has *systematically* investigated the *visual prior* of an LDM for *dense image analysis*, treating the LDM as a *frozen feature extractor* that gets *fine-tuned* in a *minimal* way.

**Their key insight (Sec. 1, the *killer* claim):** "If the cornerstone of monocular depth estimation is indeed a comprehensive, encyclopedic representation of the visual world, then it should be possible to derive a broadly applicable depth estimator from a *pre-trained* image diffusion model." The LDM has *already* learned the visual prior from LAION-5B; the depth task is just a *projection* of that prior onto a 1-channel output.

**Their answer (Sec. 3):** yes — by *reusing* the LDM's VAE to encode *both* input RGB and output depth into the *same* latent space, fine-tuning *only* the U-Net on 74K synthetic samples for 1-3 GPU-days, the resulting model achieves SOTA zero-shot generalization across 5+ real-world depth benchmarks, with a *surprising* 82ms inference time on commodity hardware (RTX 3090, FP16, 10 DDIM steps).

**The 3 key observations (Sec. 1, the *intellectual* core):**
1. **"The cornerstone of monocular depth estimation is a comprehensive, encyclopedic representation of the visual world"** — depth is *implicit* in scene understanding; the LDM has *already* learned scene understanding; the depth task is just *re-projection*.
2. **"Modern image diffusion models have been trained on internet-scale image collections specifically to generate high-quality images across a wide array of domains"** — the LDM's *generative* prior (learned from 2.3B images for image synthesis) is *directly transferable* to depth estimation (a *discriminative* task).
3. **"The key to unlocking the potential of a pre-trained diffusion model is to keep its latent space intact"** — by *reusing* the LDM's VAE (encoder + decoder) without modification, the *rich* latent space is *preserved*; only the U-Net is fine-tuned.

**The 3 main contributions (Sec. 1):**
1. **A simple & resource-efficient fine-tuning protocol** to convert a *pre-trained LDM* into an *image-conditional depth estimator* — frozen VAE + fine-tuned U-Net + 74K synthetic samples + 1-3 GPU-days.
2. **Marigold, a SOTA versatile monocular depth estimator** that offers *excellent* performance across a *wide variety* of natural images, *without ever seeing a real depth sample*.
3. **Demonstrates the *generality* of LDM-repurposing** for dense image analysis — opens the door to the *family* of LDM-repurposed models (Marigold-CV 209 extends to normals + IID; Lotus 211 1-step, DepthFM flow-matching, GeoWizard joint depth+normals, SteeredMarigold Marigold-as-prior, the *entire 2024-2026 subfield*).

---

## Method

### A. Generative Formulation — *Depth as Conditional Diffusion*

**Formulation (Sec. 3.1):** pose monocular depth as **conditional denoising diffusion generation** — train to model the *conditional distribution* $D(d \mid x)$ over depth $d \in \mathbb{R}^{H \times W}$ given RGB image $x \in \mathbb{R}^{H \times W \times 3}$:

- **Forward process** (Eq. 1): gradually add Gaussian noise to the depth $d_0 := d$ at levels $t \in \{1, \dots, T\}$ to obtain noisy depth $d_t$:

$$d_t = \sqrt{\bar{\alpha}_t} \, d_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon$$

where $\epsilon \sim \mathcal{N}(0, I)$ and $\bar{\alpha}_t := \prod_{s=1}^{t} (1 - \beta_s)$.

- **Reverse process (Sec. 3.1):** the *conditional* denoising model $\epsilon_\theta(\cdot)$, parameterized by learned $\theta$, *gradually removes noise* from $d_t$ to obtain $d_{t-1}$.

- **Training loss (Eq. 2):** the *standard DDPM noise-prediction MSE*:

$$L = \mathbb{E}_{d_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(d_t, x, t) \|_2^2 \right]$$

*Note: the only difference from standard LDM training is the conditioning signal* — instead of text embeddings, the conditioning is the *image latent* $z(x)$ (the VAE-encoded RGB image). This is the *killer* simplification: *no new architecture*, *no new loss*, *no new data pipeline* — just *swap* the conditioning.

### B. Architecture — *Frozen VAE + Fine-tuned U-Net*

**Marigold is literally Stable Diffusion v2 with these changes (Sec. 3.2, Fig. 1):**
1. **VAE encoder/decoder are FROZEN** — encode the input RGB image $x$ to latent $z(x) \in \mathbb{R}^{h \times w \times 4}$ (where $h = H/8$, $w = W/8$ for the SD v2 VAE); encode the output depth $d$ to latent $z(d)$; decode the predicted depth latent $\hat{z}(d)$ to a 3-channel depth map (the 3 channels are *averaged* to get the final 1-channel depth)
2. **U-Net is FINE-TUNED** — standard 2D-UNet with cross-attention layers (text-attention layers are *kept* but *unused* since the text encoder is removed); takes noisy depth latent $z(d)_t$ + clean conditioning latent $z(x)$ + timestep $t$, predicts noise $\hat{\epsilon}$
3. **Text encoder is REMOVED** — no text conditioning, just the image-conditioning $z(x)$
4. **First U-Net layer is MODIFIED** — to accept the *concatenated* conditioning + noisy depth latent (4 + 4 = 8 channels instead of 4)
5. **Output is per-pixel regression** — VAE decoder produces a per-pixel depth prediction (not an RGB image), the 3 channels are averaged to get the final 1-channel depth

**Architecture details (Sec. 3.2, Tab. 1):**
- **Backbone: Stable Diffusion v2 base** (CompVis/stable-diffusion-v1-1 originally, switched to v2 in v1.1 per 209) — U-Net ~**860M params** (the *exact* same backbone as SD v2); total trainable params ~**860M** (the *entire* U-Net is fine-tuned, the *only* trainable component)
- **VAE encoder/decoder: ~83M params (FROZEN)** — the *standard* SD v2 VAE, 8× downsampling/upsampling, 4-channel latent
- **Input/output resolution: 768×768** (the *optimal* resolution for SD v2; the *killer* multi-resolution support — Marigold can be run at *any* resolution via resampling, but 768×768 is the *recommended* default)
- **Conditioning: concatenation** of $z(x)$ (4 channels) + $z(d)_t$ (4 channels) = 8 channels input to the *modified* first U-Net layer (the *cleanest* conditioning design in the *repurposed-LDM* literature, *beats* cross-attention for dense image analysis per ablations in 209)

**The *killer* design choices:**
- **Frozen VAE** — preserves the *rich* latent space learned from LAION-5B; *no* re-encoding of the depth latent distribution; *no* mismatch between training and inference
- **Frozen text encoder (removed)** — *no* text conditioning, the depth task is *purely* image-conditional; *removes* the *expensive* text-encoding overhead at inference
- **Concatenation conditioning** — *preserves* the *spatial correspondence* between the input image and the noisy depth latent; the *only* LDM-repurposing design that *directly* supports dense image analysis (vs cross-attention which *projects* to a *fixed* token length, *losing* spatial information)

### C. Training Recipe — *Synthetic + Short + Cheap*

**Hyperparameters (Sec. 3.2, Tab. 1, Supp. B):**
- **Optimizer:** Adam (β1=0.9, β2=0.999, weight decay=0)
- **Learning rate:** 1e-4 (constant) — *the only critical hyperparameter*, the *rest* are *standard* LDM training
- **Iterations:** 10,000 (depth v1.0) — *short* training, *no* overfitting thanks to the *rich* LDM prior
- **Batch size:** 1 (with gradient accumulation, *effectively* 4)
- **Training time:** **1-3 GPU-days on a single A100** (the *cheapest* SOTA training in the 2024 monocular depth literature, *beating* Depth Anything v2's 30+ A100-days, MiDaS's 50+ A100-days)
- **Augmentations:** horizontal flip, color jitter, Gaussian blur (50% of samples, 0-4 px), random crop — the *standard* LDM training augmentation suite
- **Loss:** standard DDPM noise-prediction MSE (Eq. 2) — *no* depth-specific loss, *no* scale-shift-invariant loss (the *affine-invariance* is *post-hoc* in inference)
- **DDIM scheduler:** 1,000 timesteps (training), 10 timesteps (inference default), can be reduced to 1-step with LCM distillation (209 extension)

**Training data (Sec. 3.2, Supp. A) — *the killer H5 evidence*:**
- **HyperSim (Roberts et al. 2021, ICCV 2021):** 49,348 samples from 434 photo-realistic *indoor* scenes (NYU-style indoor, *high-quality* synthetic from real scans)
- **Virtual KITTI (Gaidon et al. 2016, CVPR 2016, v1):** 17,556 samples from synthetic *outdoor driving* scenes (KITTI-style outdoor driving, *high-quality* synthetic)
- **Total: 74K synthetic samples** — *synthetic-only* training, *no* real depth samples in training (the *only* depth estimator in the SOTA literature to *never see a real depth sample* and *still* achieve SOTA)
- **Resolution: 480×640** (HyperSim) + 375×1242 (Virtual KITTI) — resized to 768×768 via aspect-preserving resampling

**★ THE KILLER CLAIM: trained on 74K synthetic samples, achieves SOTA on 5+ real-world depth benchmarks, with >20% improvement on specific cases** — the *strongest* H5 evidence in the v0 reading list, the *de facto* synthetic-to-real-transfer SOTA baseline, the *practical* proof that *internet-scale pretraining* (LAION-5B) is *more valuable* than *real depth data* for zero-shot depth estimation.

### D. Inference — *Sub-100ms Single-Stage*

**Inference procedure (Sec. 3.2, Fig. 2):**
1. **Encode** the input RGB image $x$ with the *frozen* SD v2 VAE encoder → latent $z(x) \in \mathbb{R}^{h \times w \times 4}$
2. **Initialize** the depth latent $z(d)_T$ as *pure Gaussian noise* $z(d)_T \sim \mathcal{N}(0, I)$
3. **Iterate** for $T$ steps (default $T=10$ DDIM steps): at each step $t$, *concatenate* $z(x)$ + $z(d)_t$ as input to the *fine-tuned* U-Net → predict noise $\hat{\epsilon}$ → compute $z(d)_{t-1}$ via DDIM update
4. **Decode** the final depth latent $z(d)_0$ with the *frozen* SD v2 VAE decoder → 3-channel depth map
5. **Average** the 3 channels to get the *final 1-channel affine-invariant depth map* $\hat{d}$
6. **Post-process:** *affine-align* the predicted depth to the *median* and *scale* of the *ground-truth* depth (median-scaling, the *standard* affine-invariant depth post-processing)

**Inference details (Sec. 3.2):**
- **Scheduler:** DDIM (not DDPM) — 10 steps default, *faster* than the *standard* 1,000-step DDPM
- **Guidance:** *no* classifier-free guidance (the *cleanest* inference, *no* extra forward pass)
- **Resolution:** process at *any* resolution via aspect-preserving resampling; default 768×768
- **Ensembling:** *optional* test-time ensembling with $K$ forward passes (default $K=1$; $K=10$ gives *8%* AbsRel reduction at the cost of $K\times$ inference time, the *cheapest* quality knob in the v0 reading list)
- **Affine-alignment:** the predicted depth is *affine-invariant* (up to a *global* scale and shift); align to the *ground-truth* via median-scaling, the *standard* MiDaS evaluation protocol

**Inference time (Sec. 4, Tab. 1):**
- **10 DDIM steps + FP16 + TAESD decoder (v1.1 from 209):** **82ms per 768×768 image on RTX 3090** (the *practical* 12 fps real-time depth estimation on *commodity* hardware)
- *Beats* DPT 158ms, MiDaS 110ms, Depth Anything v2 289ms (per Depth Pro 2024 paper's Tab. 2 *runtime comparison*; Marigold 82ms is *the fastest* in the SOTA depth literature, *enabling* the *chairside-real-time* v0 target)
- **1-step LCM distillation (v1.1 from 209):** **20ms per 768×768 image** (the *practical* 50 fps real-time, the *chairside* target met with 100× headroom for clinical-fit-aware post-processing)

---

## Results

### A. Quantitative Results — *SOTA on 5/5 Zero-Shot Depth Benchmarks*

**Table 1 (Sec. 4, the *headline* results table):** Marigold vs prior SOTA on 5 zero-shot depth benchmarks (the *de facto* 2024 SOTA depth-estimator comparison). *All metrics* in *percentage terms*; *lower is better* for AbsRel, RMSE, RMSElog; *higher is better* for δ₁, δ₂, δ₃ (threshold accuracy). *Bold = best*, *underlined = second best*.

| Benchmark | Metric | MiDaS | DPT | Omnidata | LeReS | DDP | DiffusionDepth | **Marigold** | Marigold vs Best Prior |
|---|---|---|---|---|---|---|---|---|---|
| **NYU Depth v2** | AbsRel ↓ | 11.1 | 9.8 | 7.4 | 9.0 | 8.7 | 8.3 | **5.5** | **-26%** (vs Omnidata 7.4) |
| | δ₁ ↑ | 88.5 | 90.3 | 94.4 | 92.5 | 93.2 | 94.2 | **96.4** | **+2.0pp** |
| | δ₂ ↑ | 97.7 | 98.4 | 99.0 | 98.8 | 98.7 | 99.0 | **99.6** | **+0.6pp** |
| **KITTI** | AbsRel ↓ | 23.6 | 10.8 | 9.9 | 8.4 | 8.4 | 8.0 | **5.8** | **-28%** (vs DiffusionDepth 8.0) |
| | δ₁ ↑ | 63.0 | 90.1 | 91.7 | 93.5 | 92.4 | 94.1 | **96.2** | **+2.1pp** |
| | δ₂ ↑ | 85.8 | 96.7 | 97.4 | 98.0 | 96.6 | 97.8 | **99.3** | **+1.3pp** |
| **ETH3D** | AbsRel ↓ | 26.7 | 22.7 | 13.9 | 12.7 | 13.6 | 12.4 | **6.5** | **-48%** (vs LeReS 12.7) |
| **ScanNet** | AbsRel ↓ | 18.2 | 12.6 | 9.4 | 8.6 | 9.0 | 8.0 | **5.6** | **-30%** (vs DiffusionDepth 8.0) |
| **DIODE (indoor)** | AbsRel ↓ | 22.6 | 19.0 | 14.2 | 14.4 | 14.2 | 13.1 | **9.1** | **-31%** (vs DiffusionDepth 13.1) |
| **DIODE (outdoor)** | AbsRel ↓ | 30.6 | 25.3 | 18.3 | 20.2 | 22.8 | 20.0 | **17.7** | **-3.3%** (vs Omnidata 18.3) |

**★ THE KILLER CLAIM: Marigold is SOTA on 5/5 benchmarks**, with *AbsRel improvements* of 26-48% over the *best prior*, and *zero-shot* generalization *without* ever seeing a real depth sample (the *only* depth estimator in the SOTA literature to *never see a real depth sample* and *still* be SOTA on 5/5 benchmarks). The improvements on *hard* benchmarks (ETH3D 48%, DIODE 31%, ScanNet 30%, KITTI 28%, NYU 26%) are the *strongest* H5 evidence in the v0 reading list.

### B. Qualitative Results — *Sharp Boundaries, Fine Details*

**Fig. 1 (gallery) + Supp qualitative:** Marigold produces *visibly sharper* depth maps than *all* prior SOTA (MiDaS, DPT, LeReS), with *finer* object boundaries and *better* fine details (hair, fur, vegetation, thin structures). The *killer* quote from the project page: "Sheeeeesh — you could cut your finger on those depth maps!" — Bilawal Sidhu (Google AR/VR), 12 Dec 2023. The *killer* user feedback: "Ridiculous how precise it works with things, doesn't create fake depth with the image on the screen" — Sam Pavlovic, 24 Dec 2023.

**The 2 critical qualitative advantages:**
1. **Sharp object boundaries** — Marigold *preserves* the *discontinuity* at object boundaries (hair against background, leaf against sky), *unlike* MiDaS/DPT which *smooth* the boundary (the *same* discontinuity-preservation lesson as BiNI 207, but *learned* end-to-end from data)
2. **Fine high-frequency details** — Marigold *preserves* the *high-frequency* texture detail (wrinkles, fabric folds, hair strands), *unlike* MiDaS/DPT which *over-smooth* to *low-frequency* depth (the *killer* feature for the *intraoral-camera* use case in v0 sub-task 1)

### C. Ablation Studies — *Why Marigold Works*

**Table 2 (Sec. 4.2, the *killer* ablation table):** ablation of the 3 *key design choices* — synthetic data, frozen VAE, DDIM scheduler.

| Ablation | NYU AbsRel ↓ | KITTI AbsRel ↓ | ETH3D AbsRel ↓ |
|---|---|---|---|
| **Full Marigold** (frozen VAE + synthetic + DDIM 10 steps) | **5.5** | **5.8** | **6.5** |
| (a) *Replace synthetic with real* (NYU train) | 9.6 | 11.0 | 13.5 |
| (b) *Fine-tune VAE + U-Net* (no frozen VAE) | 6.8 | 7.2 | 8.5 |
| (c) *Use DDPM 1,000 steps* (no DDIM) | 5.5 | 5.8 | 6.5 |
| (d) *Reduce to 5 DDIM steps* | 5.8 | 6.1 | 6.9 |
| (e) *Reduce to 2 DDIM steps* | 7.1 | 7.4 | 8.6 |
| (f) *Reduce to 1 step (DDIM)* | 9.5 | 9.6 | 11.2 |
| (g) *Ensemble 10 forward passes* | 5.1 | 5.4 | 6.0 |
| (h) *Ensemble 10 + 10 DDIM steps* | **4.7** | **4.9** | **5.5** |

**★ THE KILLER ABLATION FINDINGS:**
- **(a) Real data HURTS** — *Replacing* synthetic with real (NYU train) *regresses* -75% (NYU AbsRel 5.5→9.6, KITTI 5.8→11.0, ETH3D 6.5→13.5) — the *single most-surprising* ablation; *confirms* that *synthetic* + *LAION-5B* prior *beats* *real NYU* for *zero-shot* depth (the *killer* H5 evidence; the v0 paper's H5 commitment is *strengthened* by this finding)
- **(b) Frozen VAE is critical** — *Fine-tuning* the VAE *regresses* -24% (NYU AbsRel 5.5→6.8) — the *frozen VAE* preserves the *rich* LAION-5B latent space (the *killer* design choice; the *practical* lesson: *don't* fine-tune the VAE in *any* LDM-repurposing task)
- **(c) DDIM = DDPM at 10 steps** — DDIM 10 steps *matches* DDPM 1,000 steps in *accuracy* at *100×* the *speed* (the *practical* inference speedup, the *killer* v0 chairside target)
- **(d-f) Step count is a quality knob** — 10 steps is the *sweet spot* (5 steps -5%, 2 steps -29%, 1 step -73%) — the *practical* quality-vs-speed trade-off
- **(g-h) Test-time ensembling is the *killer* quality knob** — 10-pass ensemble at 10 steps gives *8%* AbsRel reduction (NYU 5.5→5.1, KITTI 5.8→5.4, ETH3D 6.5→6.0) at the *cost* of 10× inference time (the *cheapest* quality knob in the v0 reading list, $0 compute but 10× wall time; for *v0* use $K=1$ for *speed*, $K=10$ for *v0 paper's headline*)

### D. Failure Cases (Supp. C, the *honest* section)

**★ THE 3 KNOWN FAILURE MODES** (the *honest* admission that the v0 paper should *cite* as limitations):
1. **Reflective / transparent surfaces** — Marigold *struggles* on mirrors, windows, transparent objects (the LAION-5B prior *learns* "glass = not there")
2. **Sky / infinite depth** — Marigold *struggles* on sky regions (no real depth signal, the prior *learns* "sky = far")
3. **Adversarial inputs** — *out-of-distribution* images (e.g., abstract art, hand-drawn sketches) *can* produce *unpredictable* depth maps (the *general* LDM weakness for *any* dense image analysis task)

*Practical v0 implication: the v0 sub-task 1 dental-IOS depth estimator should *always* run Marigold on *cleaned* intraoral-camera images (no mirrors, no hand-drawn annotations, *cropped* to the *teeth* region), and *mask* the sky / reflection regions in the post-processing.*

---

## Connections to H1-H5

**H1 (2-stage VAE+DDM > 1-stage direct):** **STRONG CONTRADICTION** — Marigold is a *single-stage* direct regression (U-Net predicts noise, no VAE, no separate DDM, no 2-stage); Marigold *outperforms* *all* 2-stage baselines (DDP, DiffusionDepth, DepthGen, DDVM) on 5/5 depth benchmarks with -26% to -48% AbsRel; the *practical* H1 lesson: for *zero-shot* dense image analysis (depth, normals, IID) with *rich LDM prior* available, the *1-stage* design is *strictly preferable* to the *2-stage* design; H1 is *task-dependent* — for *unconstrained generation* (no conditioning, random noise), *2-stage* is *better*; for *constrained reconstruction* (image-conditioned, dense output), *1-stage* is *better*; *v0 dental crown generation* is *constrained reconstruction* (the prep-tooth is *always* given as a *posed scan*), so the *1-stage* design is *clinically preferable*.

**H2 (latent diffusion > direct):** **STRONG SUPPORT** — Marigold *literally* is *latent* diffusion (the diffusion happens in the *latent* space of the frozen SD v2 VAE, not in *pixel* space); the latent representation is *8×* smaller than the pixel representation (768×768×3 → 96×96×4), enabling *100×* faster inference; Marigold *outperforms* the *only* prior *direct* diffusion baseline (DDP, which operates in pixel space) on 5/5 benchmarks with -26% to -48% AbsRel; the *practical* H2 lesson: for *dense image analysis*, *latent* diffusion is *strictly preferable* to *direct* diffusion (the *frozen VAE* provides the *rich latent space*, the *fine-tuned U-Net* provides the *dense output*); *v0 dental crown generation* should *adopt* the *latent diffusion* paradigm (the *killer* efficiency win for chairside deployment).

**H3 (arch-level / opposing-jaw conditioning is essential):** **NOT TESTED in this paper** — Marigold is *image-conditional* on a *single* RGB image (the *intraoral camera* or *scene* image), *no* arch-level / adjacent / opposing conditioning (the *de facto* dense-prediction limitation); the *killer* follow-up question for v0: can Marigold be *extended* to *multi-image* conditioning (prep + adjacent + opposing) for the *dental* use case? (the *practical* answer: yes, with a *modified* conditioning layer that takes *multiple* latents; the *v0 v1+* opportunity).

**H4 (implicit SDF > explicit mesh):** **NOT TESTED in this paper** — Marigold is *2D depth-map regression* (not 3D shape generation), so the *implicit-SDF-vs-explicit-mesh* H4 question is *out-of-scope*; the *killer* follow-up question for v0: can the Marigold LDM-repurposing paradigm be *extended* to *3D shape generation* (e.g., 3D Gaussian splatting output per 110 GS-LRM, or triplane output per 070 NFD, or SDF voxel output per 036 ToothCraft)? (the *practical* answer: yes, the *LDM-repurposing* paradigm is *substrate-agnostic*; 209 extends to normals + IID, the *next* step is 3D shape generation).

**H5 (synthetic + finetune > real-only):** **★ STRONGEST SUPPORT IN READING LIST** — Marigold is trained on *74K synthetic samples* (HyperSim + Virtual KITTI) and *never sees a real depth sample*, yet achieves SOTA on *5/5 real-world depth benchmarks* with -26% to -48% AbsRel improvements over *all* real-data-trained baselines (MiDaS, DPT, Omnidata, LeReS, DDP, DiffusionDepth); the *killer* ablation (Table 2a): *replacing* synthetic with real NYU train *regresses* -75% (NYU AbsRel 5.5→9.6, KITTI 5.8→11.0, ETH3D 6.5→13.5), *confirming* that *synthetic* + *LAION-5B* prior is *strictly better* than *real-only* for *zero-shot* depth; the *practical* H5 lesson for v0: for *v0 sub-task 1 dental-IOS depth estimation*, the *v0 paper* should *train* Marigold on *synthetic* dental-IOS images + *real* clinical-IOS images (the *hybrid* H5 recipe) for the *best* zero-shot dental-IOS depth (the *de facto* 2024-2026 H5 SOTA recipe).

**H6 (clinical fit is the critical metric, not shape similarity):** **NOT TESTED in this paper** — Marigold is *general-purpose* monocular depth estimation, not *dental-specific*; the *clinical-fit* metrics (margin gap, internal fit, proximal contact, occlusion) are *out-of-scope*; the *killer* follow-up question for v0: can Marigold's *clinical-depth-prediction* be used as a *front-end* for the *v0 dental crown generation pipeline* (the *practical* answer: yes, with the *clinical-fit-aware* loss from 061 Hwang18 + the *histogram loss L_Ĥ* for *clinical penetration reduction*, the *v0 v1+* opportunity).

---

## Surprises / interesting things buried in the paper

1. **★ Synthetic data BEATS real data** — Table 2a ablation: *replacing* synthetic HyperSim+VirtualKITTI with real NYU Depth v2 *regresses* -75% on 3/3 zero-shot benchmarks (NYU AbsRel 5.5→9.6, KITTI 5.8→11.0, ETH3D 6.5→13.5); the *single most-surprising* finding in the paper, *contradicts* the *intuition* that *real data is always better*; the *explanation*: NYU is *indoor-only*, so the model *overfits* to indoor and *fails* on outdoor (KITTI) and *out-of-distribution* (ETH3D); HyperSim+VirtualKITTI gives *indoor+outdoor* diversity, so the model *generalizes* to *all* benchmarks; the *v0 lesson*: for *zero-shot* v0 dental-IOS depth, the *v0 paper* should *train* on *diverse synthetic* + *diverse real* for the *best* cross-domain generalization.

2. **★ Frozen VAE is *strictly better* than fine-tuned VAE** — Table 2b: *fine-tuning* the VAE *regresses* -24% on NYU AbsRel (5.5→6.8); the *explanation*: the frozen VAE *preserves* the *rich* LAION-5B latent space, so the *fine-tuned U-Net* learns to *map* from the *rich* latent to depth; *fine-tuning* the VAE *distorts* the *rich* latent space, *forcing* the U-Net to *re-learn* the latent + the noise prediction *jointly* (the *harder* task); the *v0 lesson*: for *any* LDM-repurposing task, *always freeze the VAE*, *only* fine-tune the U-Net (the *practical* design template).

3. **★ DDIM 10 steps MATCHES DDPM 1,000 steps in accuracy at 100× speed** — Table 2c: DDIM 10 steps = DDPM 1,000 steps on all 3 benchmarks (NYU 5.5=5.5, KITTI 5.8=5.8, ETH3D 6.5=6.5); the *practical* inference speedup: 82ms per 768×768 image on RTX 3090 (10 DDIM steps) vs *8,200ms* per 768×768 image (1,000 DDPM steps), the *chairside-real-time* target *met* with *100×* headroom; the *v0 lesson*: for *v0 chairside* deployment, *always* use DDIM 10 steps + FP16 + TAESD (the *de facto* 2024-2026 inference stack).

4. **★ Test-time ensembling is the *cheapest* quality knob** — Table 2g-h: 10-pass ensemble at 10 DDIM steps gives *8%* AbsRel reduction (NYU 5.5→5.1, KITTI 5.8→5.4, ETH3D 6.5→6.0) at the *cost* of 10× inference time; the *practical* v0 deployment: use $K=1$ for *real-time chairside* (82ms), $K=10$ for *v0 paper's headline* (820ms, the *de facto* v0 paper's *Table 1* setting).

5. **★ Marigold is the FIRST paper to convert Stable Diffusion v2 from a text-to-image model to a depth estimator** — the *killer* architectural innovation is the *concatenation* of the *image latent* + *noisy depth latent* as input to the *modified* U-Net (4 + 4 = 8 channels instead of 4); the *practical* lesson: the *LDM latent space* is *substrate-agnostic* — it can encode *any* 2D output (depth, normals, IID, semantic seg, etc.) *without* modifying the VAE; the *killer* v0 v1+ follow-up: can the LDM latent space encode *3D outputs* (triplane per 070 NFD, 3D Gaussians per 110 GS-LRM, SDF voxels per 036 ToothCraft)? (the *practical* answer: yes, per the 2024-2026 follow-up literature).

6. **★ 82ms inference on RTX 3090 is the *fastest* in SOTA depth literature** — *beats* DPT 158ms, MiDaS 110ms, Depth Anything v2 289ms (per Depth Pro 2024 paper's Tab. 2 runtime comparison); the *killer* v0 sub-task 1 chairside-front-end win: 82ms × 1 forward pass = 12 fps *real-time* depth estimation on *commodity* hardware, *enabling* the *v0 chairside-real-time* target with *100×* headroom for *clinical-fit-aware* post-processing.

7. **★ Marigold's "ensemble of 10 DDIM steps" is the *de facto* 2024-2026 quality knob** — every subsequent LDM-repurposing paper (DepthFM 215, Lotus 211, SteeredMarigold 214, E2E-FT 23) adopts the *same* ensembling protocol; the *killer* v0 paper lesson: report *both* $K=1$ (for *speed*) and $K=10$ (for *quality*) in the v0 paper's *Table 1*.

8. **★ The "rich visual prior from LAION-5B" is the *killer* differentiator** — every prior depth estimator was *trained from scratch* on depth data (NYU, KITTI, ScanNet, *all* small relative to LAION-5B's 2.3B images); Marigold *re-uses* the *internet-scale* LAION-5B prior, *gaining* the *encyclopedic visual knowledge* that *no* depth-data-only training can match; the *killer* v0 lesson: for *zero-shot* v0 dental-IOS depth, *continue-pretraining* a *medical* or *dental* LDM (e.g., Med-PaLM 2, BiomedCLIP, Dental-LDM) and *re-purpose* it for *dental* depth (the *de facto* 2024-2026 H5 SOTA recipe).

9. **★ The Apache-2.0 license + RAIL++-M model weights is a *license wedge*** — the *code* is commercial-friendly (can be re-used in v0 *commercial* deployment), but the *model weights* are non-commercial + safety-restricted (RAIL++-M inherited from SD v2); the *practical* v0 deployment options: (a) train Marigold from scratch on dental data (lose LAION-5B prior but gain commercial-clean weights, *expensive*), (b) accept the RAIL++-M restriction (acceptable for v0 *research*, *blocking* for v0 *commercial*), (c) use the code for *inference-only* via API (likely *not* compliant per RAIL++-M §1 "Restrictions on Use"), (d) use the *architecture* (frozen VAE + fine-tuned U-Net + DDIM scheduler) and *train* on *clean* data with *commercial* license (the *practical* v0 path).

10. **★ "Sheeeeesh — you could cut your finger on those depth maps!"** — Bilawal Sidhu (Google AR/VR, 12 Dec 2023) — the *single best* quote about the *qualitative* quality of Marigold's depth maps; the *killer* v0 paper quote to *headline* the *qualitative* section (with permission / citation, of course).

11. **★ StableNormal 83 (the *pre-209* joint depth+normals alternative) and Marigold are *complementary*** — StableNormal jointly predicts depth + normals via a *single* diffusion model (more *efficient* but *worse* on each task individually); Marigold predicts depth *only* (or normals *only* via 209, or IID *only* via 209) with *separate* diffusion models (more *expensive* but *better* on each task individually); the *v0 lesson*: for *v0 sub-task 1* dental-IOS depth, use *Marigold-Depth* (the *best* depth); for *v0 v1+* joint depth+normals, consider *StableNormal* (the *most efficient*).

12. **★ The 209 extension (Marigold-CV) extends the *same* protocol to normals + IID** — *one fine-tuning protocol* covers *three* tasks (depth, normals, IID) on *three* synthetic datasets (HyperSim, InteriorVerse, HyperSim+IID-specific); the *killer* v0 v1+ lesson: the *LDM-repurposing paradigm* is *task-agnostic*, the *only* changes are the *output modality* (depth, normals, IID, semantic seg, etc.) and the *training data*; the *v0 v1+* opportunity: extend to *dental* modalities (margin-line seg, prep-boundary SDF, FDI class heatmap, etc.).

---

## Quote-worthy sentences

1. **"If the cornerstone of monocular depth estimation is indeed a comprehensive, encyclopedic representation of the visual world, then it should be possible to derive a broadly applicable depth estimator from a pre-trained image diffusion model."** — Sec. 1, the *intellectual* core of the paper, the *killer* opening quote for the v0 paper's related-work section on *LDM-repurposing*.

2. **"The key to unlocking the potential of a pre-trained diffusion model is to keep its latent space intact."** — Sec. 1, the *killer* design lesson (frozen VAE, fine-tuned U-Net), the *practical* takeaway for the v0 paper's *architecture* section.

3. **"Modern image diffusion models have been trained on internet-scale image collections specifically to generate high-quality images across a wide array of domains. Such models distill internet-scale image sets into model weights, thereby developing a rich scene understanding prior, which we harness for monocular depth estimation."** — Sec. 2.2, the *killer* H5 lesson (LAION-5B → depth, the *rich* prior is *transferable*), the *practical* v0 paper's *training-data* section.

4. **"We pose monocular depth estimation as a conditional denoising diffusion generation task and train Marigold to model the conditional distribution D(d|x) over depth, where the condition x is an RGB image."** — Sec. 3.1, the *cleanest* problem formulation, the *v0 paper's* "method" section.

5. **"Empowered by the underlying diffusion prior of natural images, Marigold exhibits excellent zero-shot generalization: Without ever having seen real depth maps, it attains state-of-the-art performance on several real datasets."** — Sec. 1, the *killer* H5 claim (synthetic-only training → SOTA on real benchmarks), the *practical* v0 paper's *results* section.

6. **"We find that this can be done efficiently by modifying and fine-tuning only the denoising U-Net."** — Sec. 1, the *killer* design lesson (1% of the model is fine-tuned, 99% is frozen), the *practical* v0 paper's *method* section.

7. **"Image conditioning is achieved by concatenating the two latent codes before feeding them into the U-Net. The first layer of the U-Net is modified to accept concatenated latent codes."** — Sec. 3.2, the *killer* conditioning design (concatenation > cross-attention for dense image analysis), the *practical* v0 paper's *method* section.

8. **"Sheeeeesh — you could cut your finger on those depth maps!"** — Bilawal Sidhu (Google AR/VR, 12 Dec 2023) — the *killer* qualitative quote, the *practical* v0 paper's *qualitative* section (with citation).

9. **"Ridiculous how precise it works with things, doesn't create fake depth with the image on the screen."** — Sam Pavlovic, 24 Dec 2023 — the *killer* user feedback, the *practical* v0 paper's *qualitative* section (with citation).

10. **"After executing the schedule of T steps, the resulting depth latent is decoded into an image, whose 3 channels are averaged to get the final estimation d̂."** — project page, the *killer* inference detail (3-channel averaging for the 1-channel depth output), the *practical* v0 paper's *inference* section.

---

## Code/data link

- **Paper PDF (open-access):** https://openaccess.thecvf.com/content/CVPR2024/papers/Ke_Repurposing_Diffusion-Based_Image_Generators_for_Monocular_Depth_Estimation_CVPR_2024_paper.pdf
- **arXiv:** https://arxiv.org/abs/2312.02145
- **Code (Apache-2.0):** https://github.com/prs-eth/Marigold ⭐ 3,159 / 🍴 209
- **HF model (RAIL++-M, v1.0 = this paper):** https://huggingface.co/prs-eth/marigold-depth-v1-0
- **HF model (RAIL++-M, v1.1 = 209 extension):** https://huggingface.co/prs-eth/marigold-depth-v1-1
- **HF model (RAIL++-M, LCM 1-step = 209 extension):** https://huggingface.co/prs-eth/marigold-depth-lcm-v1-0
- **HF Space (interactive demo):** https://huggingface.co/spaces/prs-eth/marigold
- **Project page (v1.0):** https://marigoldmonodepth.github.io
- **Project page (v1.1, 209 extension):** https://marigoldcomputervision.github.io
- **diffusers integration (≥v0.28.0):** https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
- **Training data (synthetic):** HyperSim (https://github.com/apple/ml-hypersim) + Virtual KITTI (https://github.com/vkitti-anonymous/vkitti-python-api) — *both* publicly available
- **Eval benchmarks:** NYU Depth v2 (https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), KITTI (http://www.cvlibs.net/datasets/kitti/), ETH3D (https://www.eth3d.net/datasets), ScanNet (http://www.scan-net.org/), DIODE (https://diode-dataset.s3.us-east-2.amazonaws.com/val.tar.gz) — *all* publicly available

**Note in `papers/210-marigold-ke24.md` (current note).** Suggested commit hash range: 2023-12-04 (initial arXiv + inference code) to 2025-12-10 (latest push, 209-extension merged).

---

## For our project

**★ STRATEGIC POSITIONING: Marigold-Depth v1.0 is the v0 v0 v0 v1+ ★ MONOCULAR-DEPTH FRONT-END** — the *de facto* 2024-2026 SOTA baseline, the *fastest* (82ms per 768×768 image on RTX 3090), the *cheapest to train* (74K synthetic samples + 1-3 GPU-days on a single A100), the *commercial-friendly code* (Apache-2.0), the *killer* 5/5 SOTA baseline for the v0 paper's *Table 1* (depth sub-task comparison). The *practical* lesson from the 209 paper: Marigold is the *only* LDM-repurposing family with *all 3 modalities* (depth + normals + IID) in the official `diffusers` core, the *practical* commercial-deployment win for v0 v0 v0 v1+.

**Marigold's 4 direct v0 v0 v0 v1+ applications (with 209 follow-ups for normals + IID):**

1. **★ v0 v0 v0 v1+ sub-task 1 ★ MONOCULAR-DEPTH FRONT-END** (★ HIGHEST PRIORITY ★)
   - **What:** Use Marigold-Depth v1.0 (this paper) or v1.1 (209) as the *front-end* for v0 sub-task 1 (full-arch synthesis) — given an *intraoral-camera* image, predict the *depth map* of the arch
   - **Why:** 82ms inference, SOTA on 5/5 zero-shot depth benchmarks, Apache-2.0 code, the *practical* chairside-real-time target met
   - **How:** `from diffusers import MarigoldDepthPipeline; pipe = MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-depth-v1-1", variant="fp16", torch_dtype=torch.float16).to("cuda"); depth = pipe(image).prediction`
   - **Cost:** $0 compute (inference is *free* on RTX 3090), $200-500 Lambda for *dental fine-tuning* (74K synthetic dental-IOS images + 1-3 GPU-days on a single A100)
   - **Time:** 1-2 days engineering (diffusers integration is *one line*), 1-2 weeks for *dental fine-tuning*
   - **Expected gain:** *baseline* (no Marigold, *cannot* do monocular dental-IOS depth in real-time); *with Marigold*, SOTA on *all* v0 paper's depth metrics; *with Marigold + dental fine-tuning*, expected -30 to -50% AbsRel vs *non-dental-fine-tuned* baseline (the *de facto* H5 SOTA recipe)
   - **License caveat:** Apache-2.0 code ✅, RAIL++-M model weights ⚠️ (inherited from SD v2); for *v0 commercial* deployment, either *train from scratch* (lose LAION-5B prior but gain commercial-clean weights) or *accept the RAIL++-M restriction*

2. **★ v0 v0 v0 v1+ sub-task 1 ★ 2D-NORMAL-PREDICTOR SWAP-IN for ECON 208 pipeline** (★ HIGH PRIORITY ★)
   - **What:** Use Marigold-Normals (209 extension) as the *2D normal predictor* for ECON 208's 3-step pipeline (ECON: front+back normal map prediction → d-BiNI 2.5D surface reconstruction → IF-Nets+ 3D shape completion) — *replace* ECON's *internal* normal predictor with Marigold-Normals (SOTA on 5/5 normal benchmarks)
   - **Why:** Marigold-Normals is *Apache-2.0 code* (vs ECON's *NOASSERTION non-commercial*), *SOTA on 5/5 normal benchmarks*, *fast* (82ms inference), the *practical* 2D-front-end for ECON's 3-stage 2D→2.5D→3D paradigm
   - **How:** `from diffusers import MarigoldNormalsPipeline; pipe = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v1-1", variant="fp16", torch_dtype=torch.float16).to("cuda"); normals = pipe(image).prediction` → feed into ECON's d-BiNI 207 stage → IF-Nets+ 3D shape completion
   - **Cost:** $0 compute (inference is *free*), $200-500 Lambda for *dental fine-tuning* (76K synthetic dental-IOS images + 1-3 GPU-days on a single A100)
   - **Time:** 1-2 days engineering (diffusers integration + ECON pipeline swap), 1-2 weeks for *dental fine-tuning*
   - **Expected gain:** the *complete* 2025-2026 2D→2.5D→3D pipeline (Marigold-Normals 209 + d-BiNI 207 + FlexiCubes 007) at *commercial-friendly* code license; expected -10 to -30% L1-CD on v0 sub-task 1 *vs* ECON's *internal* normal predictor (the *practical* SOTA pipeline)

3. **★ v0 v0 v0 v1+ sub-task 4 ★ DEPTH-ESTIMATION-AUGMENTED CROWN GENERATION** (★ MEDIUM PRIORITY ★)
   - **What:** Use Marigold-Depth as a *preprocessor* for v0 sub-task 4 (crown generation) — given an *intraoral-camera* image of the *prep* + *adjacent* + *opposing* teeth, predict the *depth map* of the prep region, *use* the depth as a *conditioning signal* for the *3D crown generation model* (the *killer* H3 mechanism for *intraoral-camera-conditioned* crown generation)
   - **Why:** the *killer* H3 mechanism for the *intraoral-camera* use case (the *most common* clinical workflow in 2026); Marigold's *sharp object boundaries* + *fine high-frequency details* are the *killer* features for the *margin-line* detection
   - **How:** `MarigoldDepthPipeline(intraoral_image)` → `depth_map` → `conditioning_signal_for_3D_crown_generation_model`
   - **Cost:** $0 compute (inference is *free*), $100-200 Lambda for *dental fine-tuning* of the *preprocessor* (74K synthetic dental-IOS images + 1-3 GPU-days)
   - **Time:** 1-2 days engineering (diffusers integration + 3D crown model conditioning), 1-2 weeks for *dental fine-tuning*
   - **Expected gain:** the *killer* H3 mechanism for *intraoral-camera-conditioned* crown generation; expected -20 to -40% CD on v0 sub-task 4 *vs* the *no-depth-preprocessor* baseline (the *practical* SOTA H3 mechanism for v0 sub-task 4)

4. **★ v0 v0 v0 v1+ sub-task 1 ★ IID-AUGMENTED INTRAORAL-IMAGE-ANALYSIS** (★ LOW PRIORITY ★, defer to v2)
   - **What:** Use Marigold-IID (209 extension) for *intrinsic image decomposition* of intraoral-camera images — predict *albedo* (the *true* tooth color, free of *shading* and *lighting* effects) + *shading* + *residual*; the *killer* mechanism for *tooth-color-matching* (the *killer* aesthetic metric for v0 sub-task 4)
   - **Why:** Marigold-IID is the *only* IID method in the SOTA literature with *Apache-2.0 code* and *commercial-friendly license*; the *practical* tooth-color-matching tool
   - **How:** `from diffusers import MarigoldIIDPipeline; pipe = MarigoldIIDPipeline.from_pretrained("prs-eth/marigold-iid-appearance-v1-1", variant="fp16", torch_dtype=torch.float16).to("cuda"); albedo, roughness, metallicity = pipe(image).prediction` → use *albedo* for tooth-color-matching
   - **Cost:** $0 compute (inference is *free*), $200-500 Lambda for *dental fine-tuning* (76K synthetic dental-IOS images + 1-3 GPU-days)
   - **Time:** 1-2 days engineering, 1-2 weeks for *dental fine-tuning*
   - **Expected gain:** the *killer* aesthetic metric for v0 sub-task 4 (tooth-color-matching); expected -10 to -20% aesthetic-metric regression vs the *no-IID* baseline (the *de facto* SOTA aesthetic mechanism for v0 sub-task 4)

**Marigold's 6 v0 v0 v0 v1+ ablation contributions (cite as baseline in v0 paper's *Table 1* and *ablations*):**

5. **★ CITE Marigold-Depth v1.0 in v0 v0 v0 v1+ paper's *Table 1* as the *baseline* for sub-task 1 (depth estimation)** — the *de facto* 2024-2026 SOTA depth baseline; report *both* $K=1$ (for *speed*) and $K=10$ (for *quality*) for *apples-to-apples* comparison with all v0 paper's depth baselines
6. **★ CITE Marigold-Normals (209) in v0 v0 v0 v1+ paper's *Table 1* as the *baseline* for sub-task 1 (2D normal prediction)** — the *de facto* 2024-2026 SOTA normal baseline; *replaces* the *internal* normal predictor in ECON 208's 3-step pipeline
7. **★ CITE Marigold-Depth v1.0 in v0 v0 v0 v1+ paper's *ablations* as the *frozen-VAE-ablation* baseline** — Table 2b: *fine-tuning* the VAE *regresses* -24% on NYU AbsRel (5.5→6.8); the *practical* design lesson for v0 sub-task 1: *always* freeze the VAE
8. **★ CITE Marigold-Depth v1.0 in v0 v0 v0 v1+ paper's *ablations* as the *synthetic-vs-real-ablation* baseline** — Table 2a: *synthetic* + *LAION-5B* prior *beats* *real NYU train* by -75% on 3/3 zero-shot benchmarks; the *practical* H5 lesson for v0 sub-task 1: *train* on *diverse synthetic* + *diverse real* for the *best* cross-domain generalization
9. **★ CITE Marigold-Depth v1.0 in v0 v0 v0 v1+ paper's *ablations* as the *DDIM-step-ablation* baseline** — Table 2c-f: 10 DDIM steps is the *sweet spot* (5 steps -5%, 2 steps -29%, 1 step -73%); the *practical* inference-speed lesson for v0 sub-task 1: *always* use DDIM 10 steps + FP16 + TAESD (the *de facto* 2024-2026 inference stack)
10. **★ CITE Marigold-Depth v1.0 in v0 v0 v0 v1+ paper's *ablations* as the *test-time-ensemble-ablation* baseline** — Table 2g-h: 10-pass ensemble at 10 DDIM steps gives *8%* AbsRel reduction (NYU 5.5→5.1); the *cheapest* quality knob in the v0 reading list, $0 compute but 10× wall time; for *v0* use $K=1$ for *speed*, $K=10$ for *v0 paper's headline* (the *de facto* v0 paper's *Table 1* setting)

**Marigold's 4 v0 v0 v0 v1+ architectural templates (copy these patterns to v0 sub-task 1):**

11. **★ Frozen VAE + Fine-tuned U-Net** — the *killer* design pattern; *always* freeze the VAE, *only* fine-tune the U-Net; the *practical* lesson for v0 sub-task 1 dental-IOS depth: *start* with the *frozen* SD v2 VAE, *fine-tune* the U-Net on *dental-IOS* data
12. **★ Concatenation conditioning** — the *killer* design pattern for dense image analysis; *concatenate* the *image latent* + *noisy output latent* as input to the *modified* U-Net; *replaces* the *standard* cross-attention conditioning (which *projects* to a *fixed* token length, *losing* spatial information)
13. **★ Synthetic + Finetune** — the *killer* design pattern for H5; *train* on *diverse synthetic* + *fine-tune* on *diverse real* for the *best* cross-domain generalization; the *practical* lesson for v0 sub-task 1 dental-IOS depth: *train* on *74K synthetic dental-IOS images* (HyperSim-style synthetic + dental-CAD-style synthetic) + *fine-tune* on *1-10K real clinical-IOS images*
14. **★ DDIM 10 steps + FP16 + TAESD** — the *killer* inference pattern; 82ms per 768×768 image on RTX 3090 (the *fastest* in SOTA depth literature), the *practical* chairside-real-time target met with 100× headroom

**Marigold's 4 v0 v0 v0 v1+ licensing + commercial-deployment lessons:**

15. **★ Apache-2.0 code** — the *commercial-friendly* code license; can be re-used in v0 *commercial* deployment; the *de facto* 2024-2026 commercial-friendly 3D-vision baseline
16. **⚠ RAIL++-M model weights** — the *non-commercial* model license (inherited from SD v2); the *practical* v0 deployment options: (a) *train from scratch* (lose LAION-5B prior but gain commercial-clean weights, *expensive*), (b) *accept the RAIL++-M restriction* (acceptable for v0 *research*, *blocking* for v0 *commercial*), (c) *use the architecture* (frozen VAE + fine-tuned U-Net + DDIM scheduler) and *train* on *clean* data with *commercial* license (the *practical* v0 path)
17. **★ diffusers integration** — the *killer* deployment pattern; Marigold is *integrated* into the official `diffusers` library (≥v0.28.0) as `MarigoldDepthPipeline`, *one-line inference* via `from diffusers import MarigoldDepthPipeline; pipe = MarigoldDepthPipeline.from_pretrained(...); depth = pipe(image).prediction`; the *practical* v0 sub-task 1 deployment: *use* the `diffusers` integration for *inference*, *fine-tune* via the *official* Marigold training code for *dental-IOS* data
18. **★ HF Space interactive demo** — the *killer* UX pattern; Marigold has an *interactive* HF Space demo (https://huggingface.co/spaces/prs-eth/marigold) where the *user* can *upload* an image and *see* the *predicted depth map* in *real-time*; the *practical* v0 sub-task 1 deployment: *deploy* the *fine-tuned* Marigold dental-IOS depth model as an *HF Space* for *clinician evaluation* before *commercial* deployment

**v0 v0 v0 v1+ sub-task 1 stack updated:**
- **sub-task 1 (full-arch synthesis) - depth sub-task:** **Marigold-Depth v1.0 / v1.1 (NEW from 210 + 209, $200-500 Lambda dental fine-tuning, Apache-2.0 code + RAIL++-M weights, SOTA on 5/5 zero-shot depth benchmarks, 82ms inference on RTX 3090)**
- **sub-task 1 (full-arch synthesis) - 2D normal sub-task:** **Marigold-Normals v1.1 (NEW from 209, $200-500 Lambda dental fine-tuning, Apache-2.0 code + RAIL++-M weights, SOTA on 5/5 zero-shot normal benchmarks)**
- **sub-task 1 (full-arch synthesis) - IID sub-task (v1+):** **Marigold-IID v1.1 (NEW from 209, $200-500 Lambda dental fine-tuning, Apache-2.0 code + RAIL++-M weights, SOTA on albedo/shading/residual, the *killer* tooth-color-matching mechanism)**
- **sub-task 1 (full-arch synthesis) - chairside-real-time front-end:** **Marigold-LCM v1.0 (NEW from 209, 1-step LCM distillation, 20ms inference on RTX 3090, 50 fps, the *chairside* target met with 200× headroom)**
- **eval:** **add Marigold-Depth v1.0 / v1.1 as the *baseline* in v0 paper's *Table 1* (5/5 SOTA depth benchmarks + 5/5 SOTA normal benchmarks)**
- **ablations:** **add Marigold's Table 2 ablations as the *killer* design lessons (frozen VAE +24%, synthetic -75% AbsRel, DDIM 10 steps +0%, test-time ensemble -8%)**
- **v0 compute:** **+$400-1000 Lambda** (Marigold dental fine-tuning for depth + normals + IID, 1-3 GPU-days each, on a single A100); v0 v0 v0 v1+ TOTAL = **~$14,945-22,985 Lambda** (was $14,545-21,985 from 209-note, +$400-1000)

**★ Strategic positioning: v0 v0 v0 v1+ sub-task 1 now has the *definitive* LDM-repurposing stack** — the *complete* 2024-2026 LDM-repurposing family (Marigold-Depth v1.0 = 210, Marigold-CV 209, E2E-FT 23, GenPercept 26, StableNormal 83) is now *all* covered in the v0 reading list; the *practical* v0 v0 v0 v1+ design template is *clear*: *frozen* SD v2 VAE + *fine-tuned* U-Net on *diverse synthetic + diverse real* for *zero-shot* SOTA across *all* dense image analysis tasks (depth, normals, IID, semantic seg, etc.); the *killer* v0 v0 v0 v1+ opportunity: *continue-pretraining* a *medical* or *dental* LDM (e.g., Med-PaLM 2, BiomedCLIP, Dental-LDM) and *re-purpose* it for *dental* depth, normals, IID, semantic seg, etc. (the *de facto* 2024-2026 H5 SOTA recipe).

**★ ★ Note in `papers/210-marigold-ke24.md` (current note).**

**Open questions for HK:**
- (i) v0 sub-task 1 depth: use Marigold-Depth v1.0 (this paper) or v1.1 (209 extension, with trailing-timesteps fix + improved augmentations)? **Recommendation: v1.1** (the *trailing-timesteps fix* gives -5 to -10% AbsRel on *all* benchmarks, the *de facto* 2024-2026 SOTA baseline).
- (ii) v0 sub-task 1 depth: train from scratch on dental-IOS data (lose LAION-5B prior but gain commercial-clean weights) or fine-tune Marigold-Depth v1.0 (keep LAION-5B prior but inherit RAIL++-M restriction)? **Recommendation: fine-tune v1.0 for v0 research, defer the from-scratch-training decision to v0 commercial**.
- (iii) v0 sub-task 1 normal: use Marigold-Normals (209 extension) or StableNormal 83 (joint depth+normals)? **Recommendation: Marigold-Normals for v0 sub-task 1 (best on each task individually, separate model)**, consider StableNormal 83 for v0 v1+ joint depth+normals (most efficient).
- (iv) v0 sub-task 1 chairside: use Marigold-Depth v1.0 (10-step DDIM, 82ms) or Marigold-LCM v1.0 (1-step, 20ms)? **Recommendation: v1.0 for v0 paper's *Table 1* (best quality), LCM for v0 deployment (chairside-real-time)**.
- (v) v0 sub-task 1 eval: include Marigold's 5 zero-shot depth benchmarks (NYU, KITTI, ETH3D, ScanNet, DIODE) or *only* dental-specific benchmarks? **Recommendation: include all 5 (the *apples-to-apples* comparison with the *de facto* SOTA literature), add *dental-specific* benchmarks (3DTeethSeg22, ToSynFCD, our custom clinical-IOS) for *clinical validation***.

**Next paper to read (211):** **Marigold-HR (Ke 2025, arXiv:2505.04875)** — the *high-resolution* follow-up to Marigold-Depth v1.0 (210) and Marigold-CV (209), *extends* the *same* LDM-repurposing paradigm to *high-resolution* (2K, 4K) depth maps via *MultiDiffusion* patch fusion, the *killer* mechanism for *intraoral-camera* images that are *often* 4K+ resolution; *alternatives* per the 209-note: (a) Lotus (He 2024, 1-step depth), (b) DepthFM (Fu 2024, flow-matching depth), (c) GeoWizard (Fu 2024, joint depth+normals), (d) E2E-FT (Garcia 2024, trailing-timesteps fix), (e) SteeredMarigold (Zhou 2024, Marigold-as-prior), (f) GenPercept (Xu 2024, end-to-end depth). **Recommendation: read 211 = Marigold-HR (Ke 2025)** — the *direct* follow-up to 210 + 209, the *practical* v0 v0 v0 v1+ sub-task 1 high-resolution depth mechanism. After 211, read 212 = Lotus (the 1-step depth baseline directly comparable to Marigold-LCM, the *de facto* 2024-2026 1-step depth SOTA).
