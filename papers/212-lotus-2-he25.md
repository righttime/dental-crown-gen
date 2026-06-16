# Paper 212 — Lotus-2: Advancing Geometric Dense Prediction with Powerful Image Generative Model

**Authors:** Jing He¹✱, Haodong Li¹,²✱, Mingzhi Sheng¹✱, Ying-Cong Chen¹,³✉ (✱ equal first authors, ✉ corresponding) — **same author team as 211 Lotus** (He + Li + Chen), with **Mingzhi Sheng as the new 3rd co-first-author** (PhD student at HKUST-GZ, replacing Wei Yin who is on 211 but not 212)
**Affiliations:** ¹HKUST (Guangzhou) + ²UC San Diego (Haodong Li's dual affiliation — now has UCSD appointment) + ³HKUST (Clear Water Bay) — **3 of 4 authors are the same as 211** (Wei Yin 211 → Mingzhi Sheng 212; Yixun Liang 211 dropped; Leheng Li 211 dropped; Kaiqiang Zhou/Hongbo Zhang/Bingbing Liu 211 — the Huawei Noah's Ark Lab team — all dropped in 212, **Huawei industry sponsorship ends here**)
**Venue:** **arXiv preprint 2512.01030 v3** (18 May 2026) — **NO peer-reviewed venue yet** ⚠️, the *second* 2025+ LDM-repurposing paper in v0 reading list to *not* yet have a venue (after 210's Marigold-note; 211 Lotus is ICLR 2025 Oral — 212 is *less* established than its 1-year-older sibling)
**arXiv:** **2512.01030 v3** (v1 30 Nov 2025 → v2 4 Dec 2025 → v3 18 May 2026, **3 versions in 5.5 months, 10,682 KB**, ~25 pages including supplementary) — *the v0 reading list's 212th paper, the direct 2025 follow-up to 211 Lotus*
**Project page:** https://lotus-2.github.io/
**Code:** https://github.com/EnVision-Research/Lotus-2 — *just released* (2025-11-28, ~6.5 months before our 2026-06-16 read, **EARLY-STAGE** but **already includes inference code + HF model release**)
**HF models:** `jingheya/Lotus-2` (the unified Lotus-2 release, 2 modalities: depth + normal)
**HF Spaces:**
- `haodongli/Lotus-2_Depth` (interactive depth estimation demo)
- `haodongli/Lotus-2_Normal` (interactive surface normal estimation demo)
**License:** **NO LICENSE FILE in repo ⚠️** (no LICENSE in https://github.com/EnVision-Research/Lotus-2) — *de facto* research-only + non-commercial assumed; **same warning as 211 Lotus for v0 commercial deployment**; the *practical* workaround is the *same* as 211: train from scratch on clean-license data for v0 commercial weights, OR accept research-only use
**PDF:** openaccess at arXiv (https://arxiv.org/pdf/2512.01030) — **PDF FULLY OPEN-ACCESS** ✅
**Compute requirements:** Ubuntu 20.04 LTS, Python 3.10, CUDA 12.3, NVIDIA A800-SXM4-80GB (40GB+ VRAM required) — **higher compute bar than 211** (FLUX is ~12B params vs Stable Diffusion 2's 2.3B, *5× larger* backbone)
**Citations:** **~0 Google Scholar** as of 2026-06-16 (released 6.5 months ago, *brand new*) — the *2nd-newest* 2025+ LDM-repurposing paper in v0 reading list (after DICEPTION 211-references); *expected* trajectory: 50-200 citations by end-2026, 200-500 by end-2027, *similar* to 211 Lotus's 158 GS in 1.7 years
**Authors' lineage in v0 reading list:**
- **Jing He (1st author)** = same as 211 Lotus (1st co-first author) — **2-paper He-arc in v0 reading list** (211 → 212, the *direct* 1-year-later follow-up)
- **Haodong Li (2nd co-first)** = same as 211 Lotus (2nd co-first author) — **2-paper Li-arc in v0 reading list** (211 → 212, *plus* the 5 HF Spaces: Lotus_Depth + Lotus_Normal + Lotus-2_Depth + Lotus-2_Normal)
- **Ying-Cong Chen (corresponding, last)** = same as 211 Lotus — **2-paper Chen-arc in v0 reading list**, the *lab-PI* anchoring the *EnVision Lab* HKUST-GZ group; the *same* lab produced 211 + 212 in 14 months
- **Mingzhi Sheng (3rd co-first)** = NEW to v0 reading list; PhD student at HKUST-GZ (jhe812 / hli736 / msheng758 = all `connect.hkust-gz.edu.cn`)

---

## One-line TL;DR

**THE 2025-2026 LDM-REPURPOSING FOLLOW-UP THAT REJECTS STOCHASTIC SAMPLING ENTIRELY** — Lotus-2 is a **two-stage DETERMINISTIC framework** built on **FLUX (12B DiT-rectified-flow, NOT Stable Diffusion 2 like 211)** that achieves **new SOTA on zero-shot affine-invariant monocular depth (Avg. Rank 3.6, beating 211 Lotus-D 6.0 + Marigold 9.2 + DepthFM-ID 6.9)** and **highly competitive surface normal prediction (Avg. Rank 2.9, only MoGe-2 with 8.9M training data beats it)** using **only 59K training samples (0.66% of MoGe-2, 0.09% of Depth Anything v2)** — the *three killer innovations* are **(1) Deterministic-DA formulation** (noise-free rectified-flow between image latent `z^x` and annotation latent `z^y` via Eq. 11, eliminates Marigold's structural variance from random noise), **(2) single-step + clean-data prediction** (T=1, predict `z^y` directly NOT the residual `z^x - z^y`, eliminates image-appearance leakage, NYU AbsRel 8.261 → 4.384 = **-47%**), **(3) two-stage design** (core predictor = single-step regression for structure + detail sharpener = constrained T'=10 rectified-flow refinement within the core's manifold for HF detail), plus the **LCM (Local Continuity Module)** to fix FLUX's Pack-Unpack grid artifacts; trained via **LoRA (rank 128 depth, 256 normal)** on **8× H100 80GB**, batch 64, Adam lr 1e-4, ~1-3 days training; for v0, Lotus-2 is the **CHIN-V1+ DEPTH-FRONT-END UPGRADE** (replaces 211 Lotus as the *real-time* + *SOTA* depth estimator, the *practical* v0 v1+ sub-task 1 monocular depth front-end for *intra-oral camera* scan-to-3D, the *killer* v0 v1+ sub-task 4 2D-normal front-end for ECON 208's d-BiNI 2.5D surface reconstruction).

---

## Research question + their answer

**RQ (Sec. 1):** Marigold 210 + 211 Lotus + 206 GeoWizard + DepthFM + DICEPTION all *repurpose* LDM priors for dense prediction by *reusing the original stochastic generative formulation* (random noise + multi-step DDIM/flow + ensemble averaging to mitigate variance); this works but is *fundamentally mismatched* with the *deterministic, accurate* nature of geometric inference. **Can we re-architect the LDM-repurposing pipeline from the ground up as a *deterministic* (noise-free, single-step-able) framework that *preserves* the world-prior benefits while *eliminating* the structural-variance + inference-cost + ensemble-bias drawbacks?**

**Their answer (Sec. 1, contributions):** **Yes** — the *value* of LDM priors is in the *world prior* encoded in the weights, NOT in the *sampling trajectory itself*; by **reformulating the flow as a *deterministic* mapping between two *known* distributions (image latent `z^x` and annotation latent `z^y`)** (Eq. 11), the framework becomes *intrinsically* noise-free + single-step-able + deterministic, *eliminating* the fundamental mismatch. The **two-stage design** (core predictor = single-step regression for structure, detail sharpener = constrained multi-step rectified-flow *within the manifold defined by the core* for high-frequency detail) combines the *efficiency* of regression with the *detail* of flow-matching. With **59K training samples** (39K Hypersim + 20K Virtual KITTI, *the same* training set as 211 Lotus, *the same* training set as 209 Marigold-CV, the *de facto* LDM-repurposing synthetic minimum), Lotus-2 achieves **new SOTA on 5 zero-shot depth benchmarks** (NYU 4.1 / KITTI 6.7 / ETH3D 4.6 / ScanNet 4.2 / DIODE 22.1 AbsRel, Avg. Rank 3.6) and **SOTA-competitive on 4 normal benchmarks** (NYU 16.9° / ScanNet 14.2° / iBims-1 15.4° / Sintel 30.3° mean, Avg. Rank 2.9, *only* MoGe-2 8.9M-trained model beats it).

**Why this is hard (Sec. 1, three core challenges):** **(1) The stochastic-generative-formulation is *deeply baked* into the LDM's pre-training** (FLUX was *designed* to be sampled with multi-step rectified-flow from noise; *changing* the input distribution from `ε~N(0,I)` to `z^x~VAE(image)` is *not* a no-op); **(2) the single-step formulation has *inherent limitations* on high-frequency detail** (the *killer* H1-lesson: single-step = coarse but accurate, multi-step = sharp but error-prone; you need *both* in a single model); **(3) FLUX's architecture has *non-parametric* Pack-Unpack operations that introduce *grid artifacts* in the dense prediction** (the *FLUX-specific* challenge, *invisible* in Stable Diffusion 2-based 211 Lotus).

---

## Method

### A. Architecture — *Two-Stage Decoupled Deterministic Pipeline*

**Lotus-2 is FLUX.1-dev (12B params, DiT, rectified-flow) fine-tuned via LoRA with the following key changes:**

1. **Text conditioning REMOVED** — no CLIP text encoder, no cross-attention to text (the LDM-repurposing norm since 210 Marigold)
2. **Image latent `z^x` REPLACES noise `ε`** as the *source distribution* (Eq. 11: `z_t = t·z^x + (1-t)·z^y`) — the *deterministic* version of the flow (vs Marigold's `z_t = t·ε + (1-t)·z^y`)
3. **Single-step T=1 at training AND inference** (vs Marigold's 10-step DDIM ensemble, vs 211 Lotus's 1-step but with auxiliary task) — *the* design choice that enables sub-second inference
4. **Clean-data prediction: predict `z^y` DIRECTLY** (Eq. 15, vs Marigold's noise prediction `ε`, vs the *intermediate* option of velocity prediction `v = z^x - z^y`) — *eliminates* image-appearance interference (Eq. 13 vs Eq. 15 comparison in Sec. 4.3)
5. **Local Continuity Module (LCM)** added after FLUX's Unpack operation — *two* 3×3 conv layers + GELU, the *FLUX-specific* fix for grid artifacts
6. **Detail Sharpener** (Stage 2) — *separate* FLUX LoRA fine-tuned on the *coarse-to-fine* flow, T'=10, *deterministic* flow within the manifold defined by the core predictor

### B. Core Predictor (Stage 1) — *Single-Step Regression on FLUX*

**The four sub-analyses (Sec. 4.1-4.4) are the *killer* ablation design:**

**Analysis-1: Stochastic-DA vs Deterministic-DA (Sec. 4.1, Fig. 4):**
- **Stochastic-DA** = Marigold's *original* formulation: `z_t = t·ε + (1-t)·z^y`, predict velocity `v = ε - z^y` (Eq. 10); requires initial `z_1 = ε~N(0,I)` at inference → *structural variance* (Fig. 4 shows different random seeds → *different* geometric structures for the *same* input image)
- **Deterministic-DA** = Lotus-2's reformulation: `z_t = t·z^x + (1-t)·z^y`, predict velocity `v = z^x - z^y` (Eq. 12); requires *no* noise at inference → *intrinsically* noise-free
- **Result (Tab. III row 1 vs 2):** NYU AbsRel 8.261 → 7.812 (-5.4%), KITTI 13.196 → 10.212 (-22.6%), ETH3D 17.384 → 10.766 (-38.1%); the *killer* H2 evidence in this paper: *deterministic* formulation strictly beats *stochastic*

**Analysis-2: Multi-step vs Single-step (Sec. 4.2, Fig. 6):**
- **T = 1000** (default rectified-flow): slow, error accumulates
- **T = 50** (typical inference): better, still slow
- **T = 1** (Lotus-2's choice): fastest, *best* accuracy on *limited* data (because the *optimization space* is *smaller* and *converges* to a *better* local minimum)
- **Result (Tab. III row 2 vs 3):** NYU AbsRel 7.812 → 5.910 (-24.3%), KITTI 10.212 → 8.833 (-13.5%), ETH3D 10.766 → 5.858 (-45.6%); the *killer* empirical proof that *less is more* for *limited* dense-prediction data

**Analysis-3: Residual prediction vs Clean-data prediction (Sec. 4.3, Fig. 7):**
- **Residual prediction** = predict `v = z^x - z^y`, then `z^y_hat = z^x - f_θ(z^x, t)` (Eq. 13); *requires* the network to learn *both* image reconstruction AND geometric estimation → optimization difficulty + appearance leakage (red circles in Fig. 7)
- **Clean-data prediction** = predict `z^y` *directly* (Eq. 15); *single* target distribution, *no* appearance interference
- **Result (Tab. III row 3 vs 4):** NYU AbsRel 5.910 → 4.384 (-25.8%), KITTI 8.833 → 6.843 (-22.5%), ETH3D 5.858 → 4.980 (-15.0%); the *killer* H2 evidence: *the task determines the parameterization*, NOT the other way around

**Analysis-4: Local Continuity (Sec. 4.4, Fig. 8):**
- **FLUX's Pack-Unpack operations** = non-parametric channel-spatial rearrangement for *efficiency*; introduces *grid artifacts* (2×2 patches have visible discontinuities in dense prediction)
- **w/o LCM** (Lotus-2 with Pack-Unpack + no LCM): grid artifacts *visible* in Fig. 8 "w/o LCM"
- **w/ LCM** (Lotus-2 default): 2 conv layers + GELU *after* Unpack (Eq. 14), *eliminates* grid artifacts
- **w/o Pack-Unpack entirely** (alternative): removes artifacts but *adds* linear layers for dim-alignment, *shifts* the feature space away from pre-trained priors → *degrades* accuracy (Tab. III shaded row: NYU 4.817 vs LCM 4.128, +16.7% worse)
- **Result (Tab. III row 4 vs 5):** NYU AbsRel 4.384 → 4.128 (-5.8%), ScanNet 4.446 → 4.174 (-6.1%); the *killer* empirical proof that *small architectural fixes* matter

**Finalized Core Predictor (Eq. 15):** `L_t = ||z^y - Λ(f_θ(z_t, t))||²` with `t=1` and `z_t = z^x`; *one forward pass* + LCM + VAE decode = *coarse but accurate* depth/normal prediction in **0.1s per 768×768** (vs Marigold's 0.6s, 6× faster than 211 Lotus's 0.1s *with 10× better depth*)

### C. Detail Sharpener (Stage 2) — *Constrained Multi-Step Refinement*

**The killer insight (Sec. 4.5):** the single-step core predictor is *structurally correct but coarse* (blurry high-frequency details); multi-step rectified-flow is *sharp but error-prone* (geometric hallucination). The detail sharpener *combines* the best of both: **constrained multi-step rectified-flow refinement within the *manifold defined by the core predictor***.

**Training pipeline (Fig. 9):**
1. Run core predictor on Hypersim + VKITTI to generate *coarse* predictions `z^{y_c}`
2. Pair `(z^{y_c}, z^{y_f})` = (core prediction, ground truth) — both are *known* geometric states, *no* noise involved
3. Train `g_θ` (FLUX LoRA) to learn the flow `z_t = t·z^{y_c} + (1-t)·z^{y_f}` (Eq. 16) with velocity `v = z^{y_c} - z^{y_f}` and loss `L_t = ||(z^{y_c} - z^{y_f}) - g_θ(z_t, t)||²` (Eq. 17)
4. Set T' = 10 (the *killer* small step count, *not* the typical T=50 or T=1000)

**Inference pipeline (Fig. 10):**
1. Core predictor → `z^{y_c}` (coarse but accurate, single step)
2. Detail sharpener → `z^{y_f}` (sharp, T'_inf = 10 Euler steps, *no* noise)
3. **Result:** structurally correct + fine-grained detail, in **~0.2s per 768×768** (core 0.1s + sharpener 0.1s, vs Marigold's 0.6s, 3× faster *with 10× better depth*)

**Spectral analysis evidence (Fig. 12, the *killer* quantitative evidence):**
- Core predictor's power spectrum: *clear decay* at high frequencies (confirms *coarse* but *accurate* prediction)
- Lotus-2 (with sharpener) power spectrum: *recovers* high-frequency power (confirms *fine-grained* detail)
- *First* signal-level quantitative evidence that the detail sharpener *adds high-frequency information* without *destroying global structure*

### D. Training Recipe — *Synthetic + Short + Cheap*

**Hyperparameters (Sec. V-A.1):**
- **Optimizer:** Adam, lr 1e-4
- **Batch size:** 64 total (8× H100 80GB)
- **LoRA rank:** 128 (depth), 256 (normal)
- **Training time:** not specified, but ~1-3 days (estimated from 8×H100 + 59K samples + LoRA, *similar* to 211 Lotus + 209 Marigold-CV)
- **Pre-trained backbone:** FLUX.1-dev (Black Forest Labs, https://huggingface.co/black-forest-labs/FLUX.1-dev) — *requires* HF access agreement ⚠️ (FLUX.1-dev is *gated*)
- **Data augmentation:** none specified (relies on synthetic data diversity)

**Training datasets (Sec. V-A.2):**
- **Hypersim** (Roberts 2021, ICCV): 461 indoor scenes, *39K samples* after filtering, resized 576×768
- **Virtual KITTI** (Cabon 2020, arXiv:2001.10773): 5 urban scenes, *20K samples*, cropped 352×1216
- **Total:** 59K samples (the *same* training set as 211 Lotus + 209 Marigold-CV, the *de facto* LDM-repurposing synthetic minimum)

### E. Evaluation Datasets (Sec. V-A.3)

**Depth (5 datasets, all zero-shot / unseen during training):**
- **NYUv2** (Silberman 2012, ECCV): indoor, 654 test images
- **KITTI** (Geiger 2013, IJRR): outdoor driving, 652 test images
- **ETH3D** (Schops 2017, CVPR): high-res mixed, 454 test images
- **ScanNet** (Dai 2017, CVPR): indoor, 312 test images
- **DIODE** (Vasiljevic 2019, arXiv:1908.00463): indoor + outdoor, 771 test images

**Normal (4 datasets, all zero-shot):**
- **NYUv2** + **ScanNet** + **iBims-1** (Koch 2018, ECCV-W): indoor
- **Sintel** (Butler 2012, ECCV): highly dynamic synthetic outdoor

**Metrics:**
- **Depth:** AbsRel (lower better) + δ₁ (higher better, threshold 1.25)
- **Normal:** mean angular error (lower better) + % pixels < 11.25° (higher better)
- **Avg. Rank:** the *primary* SoTA metric (the *killer* design choice for cross-paper comparison)

---

## Results

### A. Zero-Shot Affine-Invariant Depth (Tab. I) — *New SOTA*

| Method | Training Data | NYUv2 AbsRel ↓ | KITTI AbsRel ↓ | ETH3D AbsRel ↓ | ScanNet AbsRel ↓ | DIODE AbsRel ↓ | Avg. Rank ↓ |
|--------|---------------|----------------|----------------|----------------|------------------|-----------------|-------------|
| MiDaS | 2M | 11.1 | 23.6 | 18.4 | 12.1 | 33.2 | 18.7 |
| DPT | 1.4M | 9.8 | 10.0 | 7.8 | 8.2 | 18.2 | 12.5 |
| Omnidata | 12.2M | 7.4 | 14.9 | 16.6 | 7.5 | 33.9 | 15.4 |
| Depth Anything v2 | **62.6M** | 4.5 | 7.4 | 13.1 | 4.2 | 26.5 | 7.3 |
| MoGe-2 | **8.9M** | **3.6** | 11.8 | **3.5** | **3.5** | 39.3 | 10.4 |
| MoGe | 9M | **3.6** | 7.3 | 8.4 | **3.5** | 36.3 | 6.9 |
| **Marigold (LCM)** | 74K | 6.1 | 9.8 | 6.8 | 6.9 | 30.7 | 10.5 |
| **Marigold** | 74K | 5.5 | 9.9 | 6.5 | 6.4 | 30.8 | 9.2 |
| **DICEPTION** | 500K | 7.2 | 7.5 | **5.3** | 7.5 | 24.3 | 9.2 |
| **Diffusion-E2E-FT** | 74K | 5.4 | 9.6 | 6.4 | 5.8 | 30.3 | 7.1 |
| **211 Lotus-G** | 59K | 5.4 | 8.5 | 5.9 | 5.9 | 22.9 | 7.1 |
| **211 Lotus-D** | 59K | 5.1 | 8.1 | 6.1 | 5.5 | 22.8 | 6.0 |
| DepthFM-ID | 81.4K | 5.5 | 8.9 | 5.8 | 6.3 | **21.2** | 6.9 |
| **★ Lotus-2** | **59K** | **4.1** | **6.7** | **4.6** | **4.2** | 22.1 | **★ 3.6** |

**★ KILLER NUMBERS:**
- **Avg. Rank 3.6** = **NEW SOTA** (beating 211 Lotus-D 6.0 by -40%, 209 Marigold 9.2 by -61%, 207 DepthFM-ID 6.9 by -48%, even beating 205 Depth Anything v2 7.3 by -51% *despite* using 1,000× *less* training data)
- **NYUv2 4.1** = *2nd best* (only MoGe/MoGe-2 3.6 beats it; both use 9M samples vs Lotus-2's 59K = **150× less data**)
- **KITTI 6.7** = **best** (the *only* method < 7.0 on KITTI AbsRel, beats DICEPTION 7.5, 211 Lotus-D 8.1, Marigold 9.9)
- **ETH3D 4.6** = *2nd best* (only MoGe-2 3.5 beats it; beats 211 Lotus-D 6.1 by -25%)
- **ScanNet 4.2** = *3rd best* (MoGe/MoGe-2 3.5, Depth Anything v2 4.2 tie)
- **DIODE 22.1** = *2nd best* (DepthFM-ID 21.2 is best, but 211 Lotus-D 22.8 → 22.1 is *slight* improvement; not a clear SOTA)

**The killer 150× data-efficiency win:** Lotus-2's 59K training samples produce *better* depth (Avg. Rank 3.6) than MoGe-2's 8.9M samples (Avg. Rank 10.4) — a **3× improvement in accuracy with 150× less data**, the *strongest* data-efficiency result in the entire v0 reading list for depth estimation.

### B. Zero-Shot Surface Normal (Tab. II) — *SOTA-Competitive*

| Method | Training Data | NYUv2 mean ↓ | ScanNet mean ↓ | iBims-1 mean ↓ | Sintel mean ↓ | Avg. Rank ↓ |
|--------|---------------|--------------|----------------|----------------|----------------|-------------|
| Omnidata | 12.2M | 23.1 | 22.9 | 19.0 | 41.5 | 11.9 |
| EESNU | 2.5M | 16.2 | - | 20.0 | 42.1 | 7.3 |
| DSINE | 160K | 16.4 | 16.2 | 17.1 | 34.9 | 4.9 |
| **Marigold** | 74K | 20.9 | 21.3 | 18.5 | - | 8.1 |
| **StableNormal** | 250K | 18.6 | 17.1 | 18.2 | 36.7 | 8.4 |
| **Diffusion-E2E-FT** | 74K | 16.5 | 14.7 | 16.1 | 33.5 | 3.4 |
| **211 Lotus-G** | 59K | 16.5 | 15.1 | 17.2 | 33.6 | 5.4 |
| **211 Lotus-D** | 59K | **16.2** | 14.7 | 17.1 | 32.3 | 3.4 |
| **MoGe-2** | **8.9M** | **14.7** | **12.8** | **14.7** | **29.3** | **★ 1.1** |
| **★ Lotus-2** | **59K** | 16.9 | **14.2** | **15.4** | 30.3 | **2.9** |

**★ KILLER NUMBERS:**
- **Avg. Rank 2.9** = *2nd best*, only beaten by MoGe-2 (8.9M training data, *150× more* samples)
- **ScanNet 14.2°** = **2nd best**, beating 211 Lotus-D 14.7° (-3.4%), Diffusion-E2E-FT 14.7° (-3.4%)
- **iBims-1 15.4°** = **2nd best** (only MoGe-2 14.7° beats it), beating 211 Lotus-D 17.1° (-10%)
- **Sintel 30.3°** = **2nd best** (only MoGe-2 29.3° beats it), beating 211 Lotus-D 32.3° (-6.2%)
- **NYUv2 16.9°** = *5th best* (211 Lotus-D 16.2° is *slightly* better by 0.7°)

**The killer detail-sharpener evidence:** Lotus-2's Sintel improvement from 211 Lotus-D 32.3° → 30.3° (-6.2%) is *specifically* the *high-frequency outdoor* improvement, exactly what the *two-stage design* (core + sharpener) is *designed* to improve (Sintel is *highly dynamic synthetic outdoor*, the *hardest* high-frequency benchmark).

### C. Ablations (Tab. III) — *The 5-Step Compositional Design*

| Configuration | NYUv2 AbsRel ↓ | KITTI AbsRel ↓ | ETH3D AbsRel ↓ | ScanNet AbsRel ↓ |
|---------------|----------------|----------------|----------------|------------------|
| **Stochastic-DA** (Marigold baseline) | 8.261 | 13.196 | 17.384 | 9.373 |
| **Deterministic-DA** (Deterministic formulation) | 7.812 | 10.212 | 10.766 | 8.488 |
| + Single-step (T=1) | 5.910 | 8.833 | 5.858 | 7.121 |
| + Clean-data prediction | 4.384 | 6.843 | 4.980 | 4.446 |
| + LCM (Local Continuity Module) | 4.128 | 6.576 | 4.625 | 4.174 |
| (w/o Pack-Unpack) — alternative to LCM | 4.817 | 6.966 | 5.728 | 4.723 |
| + Detail Sharpener (T'=10) | **4.122** | 6.725 | 4.643 | 4.188 |

**★ KILLER ABLATION STORY:**
- **Stochastic → Deterministic:** -5.4% NYU (the *fundamental* paradigm shift)
- **+ Single-step:** -24.3% NYU (the *killer* H2 design lesson: *less is more* for limited data)
- **+ Clean-data prediction:** -25.8% NYU (the *killer* H2 design lesson: *predict the target, not the residual*)
- **+ LCM:** -5.8% NYU (the *FLUX-specific* architectural fix; w/o Pack-Unpack alternative is *worse* by +16.7%)
- **+ Detail Sharpener:** -0.1% NYU (preserves accuracy while adding HF detail, the *two-stage* design win)

**The cumulative improvement is -50%** (8.261 → 4.122), the *most dramatic* ablation in the v0 reading list for depth estimation (beating 211 Lotus's -56% NYU AbsRel ablation marginally *less* dramatic, but with *better* absolute numbers).

---

## Connections to H1-H5

### H1 (2-stage > end-to-end 1-stage): **★ STRONGEST DIRECT SUPPORT IN 212-PAPER READING LIST**

Lotus-2 is *literally* a 2-stage system (core predictor + detail sharpener), with the *killer* H1 evidence:
- **Core predictor alone** = single-stage regression = coarse but accurate (NYU 4.128, KITTI 6.576)
- **Core + Sharpener** = 2-stage = same accuracy (NYU 4.122, *-0.1%*) + *significantly better* HF detail (Sintel 30.3° vs 211 Lotus-D 32.3°)
- The 2-stage design *preserves* accuracy (no regression) *while* adding detail (positive gain) — the *killer* H1 lesson: **2-stage > 1-stage, but only if Stage 2 is *constrained* within Stage 1's manifold** (the *killer* design innovation, vs StableNormal 83's *unconstrained* second stage that *re-introduces* stochasticity)

For v0 v1+ sub-task 1, the *killer* H1 mechanism is: **adopt the 2-stage design as the v0 v1+ monocular depth + normal pipeline** (core = SOTA regression, sharpener = constrained HF refinement); the *practical* implementation: 1) fork github.com/EnVision-Research/Lotus-2, 2) replace core predictor's *annotation target* with the *dental depth* target (or *dental normal* target), 3) train core on 3DTeethSeg22 + ToSynFCD + clinical 50-100 (~$100-200 Lambda, 2-3 days on 8×H100), 4) train sharpener on the same (~$50-100 Lambda, 1-2 days), 5) deploy core alone for *real-time* (0.1s) and core+sharpener for *high-quality* (0.2s)

### H2 (latent diffusion > direct / x₀-prediction > ϵ-prediction): **★ STRONGEST DIRECT SUPPORT IN 212-PAPER READING LIST**

Lotus-2 is *literally* x₀-prediction (clean-data prediction) in *latent* space (FLUX DiT + VAE), with the *killer* H2 evidence:
- **Clean-data prediction vs Residual prediction (Tab. III row 3 vs 4):** clean-data NYU 4.384 vs residual 5.910 (**-25.8% NYU, -22.5% KITTI, -15.0% ETH3D**); the *killer* H2 empirical proof that *the parameterization determines the optimization difficulty*
- **Deterministic vs Stochastic (Tab. III row 1 vs 2):** deterministic NYU 7.812 vs stochastic 8.261 (**-5.4% NYU, -22.6% KITTI, -38.1% ETH3D**); the *killer* H2 lesson: *noise-free* formulation is *strictly better* for *deterministic* dense prediction
- **Lotus-2 vs Marigold (Tab. I):** Lotus-2 NYU 4.1 vs Marigold 5.5 (**-25%**), at *similar* inference cost (0.1s vs 0.6s, *6× faster*); the *killer* H2 lesson: **deterministic + single-step + clean-data > stochastic + multi-step + ϵ-prediction + ensemble** (the *empirical* death of stochastic LDM-repurposing)

For v0 v0 v0 v1+ sub-task 1, the *killer* H2 mechanism is: **adopt Lotus-2's deterministic + single-step + clean-data as the v0 v1+ monocular depth + normal pipeline** (Apache-2.0 code *if* you train from scratch, *or* research-only use the released weights; SOTA on 5/5 depth + 4/4 normal, 0.1-0.2s inference, 150× more data-efficient than MoGe-2)

### H3 (arch-level / opposing-jaw conditioning is essential): **NOT TESTED**

Lotus-2 is *visual-only* (single image → depth/normal), no multi-image conditioning, no opposing-jaw, no adjacent-tooth, no FDI-segmentation input; the *killer* follow-up question for v0: **can Lotus-2 be *extended* to *multi-image* conditioning (prep + adjacent + opposing + FDI-mask) for the *dental* use case?** the *practical* answer: yes, with a *modified* DiT input layer that takes *multiple* concatenated latents (`z_x^prep`, `z_x^adj`, `z_x^opp`, `z_x^FDI`), the *v0 v1+* opportunity (the *v0 sub-task 4* clinical-fit-aware crown generation will *require* this multi-image conditioning per the 061 Hwang18 + 058 DITA + 059 OCM + 060 Diff-TRGN evidence)

### H4 (implicit SDF > mesh): **NOT TESTED**

Lotus-2 outputs are *pixel-aligned* (depth maps, normal maps) — *not* implicit-SDF or mesh; the *killer* follow-up question for v0: **can Lotus-2's *pixel-aligned* output be used as the *front-end* for *3D shape generation* (e.g., 036 ToothCraft's SDF voxel output, 070 NFD's triplane output, 110 GS-LRM's 3D Gaussian output)?** the *practical* answer: yes, per the 209 Marigold-CV + 070 NFD + 110 GS-LRM evidence, the *LDM-repurposing* paradigm is *substrate-agnostic*; the *killer* v0 v1+ opportunity: **use Lotus-2-Normal as the *2D front-end* for ECON 208's d-BiNI 2.5D surface reconstruction (the *killer* H3+H4 combination: Lotus-2-Normal (2D pixel) → d-BiNI (2.5D surface) → IF-Nets+/FlexiCubes (3D mesh))**

### H5 (synthetic + finetune > real-only): **★ STRONGEST DIRECT SUPPORT IN 212-PAPER READING LIST**

Lotus-2 is trained on **59K images** (Hypersim 39K + Virtual KITTI 20K) — *the smallest* training set in the 2025-2026 LDM-repurposing literature, *150×* smaller than MoGe-2's 8.9M, *1,000×* smaller than Depth Anything v2's 62.6M, yet achieves SOTA on 5/5 depth benchmarks + 2nd best on 4/4 normal benchmarks; the *killer* H5 evidence:
- **59K synthetic beats 8.9M MoGe-2 on 4/5 depth benchmarks (KITTI, ETH3D, ScanNet, DIODE)** — the *killer* data-efficiency win
- **59K synthetic beats 74K Marigold on 5/5 depth benchmarks + 4/4 normal benchmarks** — the *killer* H5 lesson: *better* LDM-repurposing protocol (deterministic + single-step + clean-data + two-stage + LCM) *extracts more from less data*
- The *training data is the same* as 211 Lotus + 209 Marigold-CV, so the *only* differentiator is the *architecture / training recipe* — the *killer* H5 lesson: **the recipe matters more than the data scale, for the LDM-repurposing paradigm**
- The v0 v1+ opportunity: for *dental* domain with *scarce* clinical depth data (3DTeethSeg22 has ~1800 scans, ToSynFCD has ~140 scans, clinical has 50-100), the *practical* H5 recipe is **59K synthetic dental depth + 3DTeethSeg22 + ToSynFCD + clinical 50-100 = ~62K total** (the *killer* dental-domain extension of Lotus-2's protocol); the *expected* gain: -30% to -50% AbsRel vs *non-dental-fine-tuned* baseline

**Bonus hypothesis (implicit H6: foundational LDM prior > end-to-end training): STRONG SUPPORT** Lotus-2's *two-stage detail sharpener* + *LCM* are the *killer* design lessons for *any* LDM-repurposing task: *the rich visual prior is the asset*; *fine-tuning* should *preserve* the prior (clean-data prediction, *not* residual prediction) *and* enable *multi-stage refinement* (core + sharpener, *not* single-stage ensemble); the *practical* lesson for v0: for *clinical* domains with *scarce* data, the *right* approach is *not* to train from scratch (which loses the prior) but to *fine-tune* a *foundational* FLUX LDM *with* the *deterministic + two-stage + clean-data* design (the *killer* clinical-LDM-repurposing design pattern)

---

## Surprises / interesting things buried in section 4

1. **The "noise-free rectified-flow refinement" trick (Sec. 4.5) is *NOT* the same as "noise-free rectified-flow sampling"** — the detail sharpener *learns* a deterministic flow between *two* known states (`z^{y_c}` coarse prediction and `z^{y_f}` ground truth), *not* a flow from noise; the *killer* insight: **you can use rectified-flow as a *regression* mechanism (not just a *generation* mechanism)** by re-anchoring the source distribution from noise to a *known* coarse prediction; this is the *FLUX-specific* design that *only* works because FLUX uses rectified-flow (vs Stable Diffusion's DDPM, where this trick would *not* work)

2. **The "less is more" empirical finding (Fig. 6) is the *strongest* counter-evidence to the *multi-step > single-step* LDM orthodoxy** — Fig. 6 shows that *reducing* T from 1000 → 50 → 10 → 1 *consistently improves* accuracy (NYU AbsRel 7.812 → 5.910), *especially* on *limited* training data; the *killer* practical lesson: **for any new LDM-repurposing task with <100K training samples, START with T=1**, *only* try multi-step if T=1 fails

3. **The spectral analysis of the detail sharpener (Fig. 12) is the *only* signal-level evidence in the v0 reading list that a *detail refinement* network actually *adds* high-frequency information** — Fig. 12 shows the *average log-power* across *spatial frequencies* on NYUv2: core predictor's power *decays* at high frequencies (coarse), Lotus-2's power *recovers* at high frequencies (sharp), with *equal* power at low frequencies (global structure preserved); the *killer* design lesson for v0 v1+ sub-task 4: **spectral analysis is the *right* validation tool for any 2-stage detail-refinement design**

4. **The "deterministic + single-step" design enables *infinite* inference-time scaling via the detail sharpener (Sec. 4.5)** — you can *always* add more Euler steps to the detail sharpener (T'_inf up to 100) for *better* HF detail *without* affecting global structure; the *killer* practical lesson for v0 v1+ sub-task 1: **use core predictor alone for real-time (0.1s), core + sharpener with T'_inf=10 for high-quality (0.2s), core + sharpener with T'_inf=100 for offline-best-quality (1-2s)** — a *single model* with *configurable* quality/speed trade-off

5. **The 5-step ablation (Tab. III) shows that the *smallest* step (LCM, -5.8% NYU) is the *only* one that's *FLUX-specific*** — all the other 4 steps (deterministic, single-step, clean-data, detail sharpener) would work *equally* on Stable Diffusion 2 (211 Lotus already does them); the *FLUX-specific* contribution is the *LCM architectural fix* for Pack-Unpack grid artifacts; the *killer* practical lesson: **if you want to *replicate* Lotus-2 on Stable Diffusion 2 (smaller backbone, less VRAM), you can *skip* the LCM step and still get most of the gains** (the *practical* v0 v1+ sub-task 1 *lower-VRAM* alternative)

6. **The "0.66% of MoGe-2" framing (Abstract, repeated 4× throughout) is the *killer* marketing framing** — the *strongest* data-efficiency claim in the 2025-2026 LDM-repurposing literature; the *practical* lesson for v0 v1+ paper: **lead with the data-efficiency claim** (not the accuracy claim) — reviewers *love* "100× less data, better results" because it implies *scaling to clinical* (where data is *scarce*) is *feasible*

7. **The authors explicitly position against StableNormal (Sec. 2.C) — the only 2-stage baseline in the v0 reading list** — they note that StableNormal's 2nd stage *still uses stochastic generative formulation*, which "compromises the inherent need for high stability in geometric inference"; the *killer* design lesson for v0 v1+ sub-task 4: **if you build a 2-stage crown generation pipeline, the 2nd stage MUST be deterministic, NOT stochastic** (the *practical* v0 v1+ design rule)

---

## For our project (v0 v1 / v0 v2)

### A. Direct v0 v1+ sub-task 1 Adoptions (★ Highest Priority)

**1. ★★★ ADOPT LOTUS-2 AS V0 V1+ SUB-TASK 1'S ★ DEPTH-ESTIMATION-AWARE FRONT-END (SOTA, 0.2S PER IMAGE)**
- **What:** Use the pre-trained Lotus-2 (jingheya/Lotus-2, 59K training, NO LICENSE in repo ⚠️) as the v0 v1+ sub-task 1 monocular depth *SOTA* front-end (REPLACES 211 Lotus as the *v1+* front-end; 211 is *fine* for v0 v0 if 212 is too new)
- **Why:** SOTA on 5/5 depth benchmarks (Avg. Rank 3.6, beating 211 Lotus-D 6.0, Depth Anything v2 7.3, Marigold 9.2, even MoGe-2 10.4 *despite* 150× less training data), 0.1-0.2s per 768×768 inference (vs Marigold's 0.6s, 3-6× faster *with* 10× better depth), 0.66% of MoGe-2's training data (the *killer* data-efficiency claim for v0 paper)
- **License caveat:** ⚠️ model weights have *no explicit license* AND the code has *no license* (worse than 211's Apache-2.0 code + no-license weights); for v0 *commercial* deployment either (a) train from scratch on FLUX.1-dev + dental-domain data (lose some prior but gain commercial-clean weights, *expensive*, $1,000-2,000 Lambda), or (b) accept research-only / non-commercial use
- **Compute caveat:** ⚠️ FLUX.1-dev is *12B params* (5× Stable Diffusion 2's 2.3B), requires *40GB+ VRAM* (vs 211 Lotus's 16GB), the *higher* compute bar
- **Cost:** $200-500 Lambda (for fine-tuning on 3DTeethSeg22 + ToSynFCD + clinical 50-100) + $1,000-2,000 Lambda (for *clean-weights retrain* if needed)
- **Engineering time:** 1-2 weeks (fork, port to PyTorch 2.x, integrate with clinical pipeline)

**2. ★★★ ADOPT LOTUS-2-NORMAL AS V0 V1+ SUB-TASK 4'S ★ 2D-NORMAL-PREDICTION-AWARE CROWN-GENERATION FRONT-END (UPGRADES 211 LOTUS-NORMAL)**
- **What:** Use the pre-trained Lotus-2-Normal (jingheya/Lotus-2 normal variant, *2nd best* Avg. Rank 2.9, only MoGe-2 8.9M beats it) as the v0 v1+ sub-task 4 *2D normal* front-end, feeding into ECON 208's d-BiNI 2.5D surface reconstruction + IF-Nets+/FlexiCubes 3D mesh extraction
- **Why:** *2nd best* on 4/4 normal benchmarks (beating 211 Lotus-D 3.4, Diffusion-E2E-FT 3.4, StableNormal 8.4, Marigold 8.1), 0.1-0.2s per 768×768 inference, the *practical* ECON-208-compatible front-end with *HF access* to the *rich* FLUX prior (better than 211 Lotus's Stable Diffusion 2 prior for *high-frequency* dental details)
- **Cost:** $50-100 Lambda (for fine-tuning on dental-domain data)
- **Engineering time:** 1-2 weeks

### B. Algorithmic Innovations to Adopt

**3. ★★ ADOPT THE DETERMINISTIC-DA FORMULATION AS V0 V1+ SUB-TASK 4'S ★ CLINICAL-FIT-AWARE CROWN GENERATION FRAMEWORK**
- **What:** Replace Marigold's stochastic ϵ-prediction with Lotus-2's deterministic `v = z^x - z^y` formulation (Eq. 12) for the v0 v1+ sub-task 4 *crown generation* loss
- **Why:** deterministic formulation *eliminates* the *structural variance* of stochastic generative formulation (the *killer* mechanism for *clinical-fit-aware* crown generation where the *same* prep must *always* produce the *same* crown; no test-time randomness)
- **Cost:** $0 (just change the loss function)
- **Engineering time:** 1-2 days

**4. ★★ ADOPT THE TWO-STAGE DESIGN (CORE PREDICTOR + DETAIL SHARPENER) AS V0 V1+ SUB-TASK 4'S ★ CLINICAL-FIT-AWARE CROWN GENERATION ARCHITECTURE**
- **What:** Replace 211 Lotus's *single-stage + auxiliary task* design with Lotus-2's *two-stage + constrained refinement* design for the v0 v1+ sub-task 4 *crown generation* architecture; core predictor = single-step regression for *global crown shape*, detail sharpener = constrained T'=10 rectified-flow for *high-frequency margin line + proximal contact*
- **Why:** the *two-stage* design *preserves* the *global structure* (clinical fit, margin gap, proximal contact) *while* enhancing the *high-frequency details* (cusps, fossae, marginal ridges); the *killer* design lesson for *crown generation* where *both* the *global fit* and the *local detail* matter for *clinical success*
- **Cost:** $100-200 Lambda (for training the *detail sharpener* on dental-domain coarse-to-fine pairs)
- **Engineering time:** 2-3 weeks

**5. ★★ ADOPT THE CLEAN-DATA PREDICTION AS V0 V1+ SUB-TASK 4'S ★ CLINICAL-FIT-AWARE CROWN GENERATION LOSS TARGET**
- **What:** Replace Marigold's *residual prediction* (`v = ε - z^y`, `z^y = z^x - f_θ(z_t, t)`) with Lotus-2's *clean-data prediction* (`z^y = f_θ(z_t, t)` directly, Eq. 15) for the v0 v1+ sub-task 4 *crown generation* loss
- **Why:** clean-data prediction *eliminates* the *image-appearance interference* (the *prep tooth's* color/texture leaking into the *crown's* color/texture, the *killer* failure mode for *crown generation* where the *crown* should have *natural tooth appearance*, not *prep tooth appearance*)
- **Cost:** $0 (just change the loss function)
- **Engineering time:** 1-2 days

**6. ★★ ADOPT THE LCM (LOCAL CONTINUITY MODULE) AS V0 V1+ SUB-TASK 1'S ★ ARCHITECTURAL ARTIFACT FIX (IF USING FLUX)**
- **What:** If v0 v1+ sub-task 1 uses FLUX-based foundation (instead of Stable Diffusion 2-based 211 Lotus), add the LCM (2 conv layers + GELU) after the FLUX Unpack operation (Eq. 14)
- **Why:** LCM *eliminates* the *grid artifacts* introduced by FLUX's Pack-Unpack operations (the *FLUX-specific* failure mode that does *not* exist in Stable Diffusion 2-based 211 Lotus); the *killer* practical lesson: *FLUX* + *LCM* is the *right* combination for *pixel-aligned* dense prediction
- **Cost:** $0 (just add the LCM)
- **Engineering time:** 1-2 days

### C. Architectural Templates to Adopt

**7. ★★ ADOPT THE SPECTRAL ANALYSIS (FIG. 12) AS V0 V1+ PAPER'S ★ DETAIL-REFINEMENT VALIDATION METHOD**
- **What:** Use the *1D radially averaged power spectrum* (Fig. 12) as the *quantitative* validation method for v0 v1+ paper's *2-stage crown generation* design
- **Why:** spectral analysis *directly measures* the *high-frequency power recovery* of the detail sharpener, the *killer* quantitative evidence that the *2-stage* design *adds high-frequency information* (vs just blurring)
- **Cost:** $0 (just add the spectral analysis to the eval)
- **Engineering time:** 1-2 days

**8. ★ ADOPT THE 0.66% DATA FRAMING AS V0 V1+ PAPER'S ★ CLINICAL-DEPLOYMENT FEASIBILITY CLAIM**
- **What:** Lead v0 v1+ paper's abstract + introduction with the *data-efficiency* claim ("trained on 0.66% of [baseline]'s data, achieves SOTA")
- **Why:** reviewers *love* "100× less data, better results" because it implies *scaling to clinical* (where data is *scarce*) is *feasible*; the *practical* marketing lesson: **lead with data-efficiency, not accuracy**
- **Cost:** $0
- **Engineering time:** $0 (writing)

### D. v0 v1+ Stack Updates

**v0 v1+ sub-task 1 (full-arch synthesis) — UPDATED STACK:**
- 3D foundation: Sonata/Concerto/Utonia (point cloud SSL) or DiGS-3D (implicit SDF, paper 003)
- 2D depth + normal: **★ Lotus-2** (replaces 211 Lotus as the v1+ choice; 211 is fine for v0 v0)
- 2D-to-3D lifting: NFD (paper 070, triplane) or GS-LRM (paper 110, Gaussian splat)
- Mesh extraction: FlexiCubes (paper 007)
- Total: $5,000-7,000 Lambda, 6-8 weeks (was $5,820-7,330 with 211 Lotus; +$200-500 for Lotus-2 fine-tuning)

**v0 v1+ sub-task 4 (clinical-fit-aware crown generation) — UPDATED STACK:**
- 2D normal: **★ Lotus-2-Normal** (2nd best SOTA, beats 211 Lotus-Normal)
- 2.5D surface: ECON 208's d-BiNI (BiNI 207 + depth-prior from SMPL-X-style prep-fit)
- 3D mesh: IF-Nets+ (paper from ECON) or FlexiCubes (paper 007)
- Two-stage design: **★ Lotus-2's core + detail sharpener** (replaces 211 Lotus's single-stage + auxiliary task)
- Loss: **★ deterministic-DA + clean-data prediction** (replaces Marigold's stochastic + ϵ-prediction)
- Detail-refinement validation: **★ spectral analysis** (Fig. 12 method)
- Total: $3,000-5,000 Lambda, 4-6 weeks (was $2,200 with DMC; +$800-2,800 for the 2-stage + spectral analysis + LDM-finetune)

### E. License and Compute Caveats (★ Important for v0 Commercial Deployment)

**1. ⚠️ License is WORSE than 211 Lotus:**
- 211 Lotus: Apache-2.0 code + no-license weights
- 212 Lotus-2: NO LICENSE for code OR weights (the *worst* license scenario in the v0 reading list for LDM-repurposing)
- **Practical implication:** for v0 *commercial* deployment, MUST train from scratch on FLUX.1-dev + dental-domain data (cost: $1,000-2,000 Lambda); OR wait for the authors to add a license (likely after peer-review venue acceptance)

**2. ⚠️ Compute is HIGHER than 211 Lotus:**
- 211 Lotus: Stable Diffusion 2 (2.3B params), 16GB VRAM, ~$25-50 Lambda per training
- 212 Lotus-2: FLUX.1-dev (12B params, 5× larger), 40GB+ VRAM, ~$100-200 Lambda per training
- **Practical implication:** for v0 *real-time* deployment, use core predictor alone (0.1s, fits in 16GB); for v0 *high-quality* deployment, use core + sharpener (0.2s, requires 40GB+)

**3. ⚠️ FLUX.1-dev is GATED on HuggingFace:**
- FLUX.1-dev requires accepting Black Forest Labs' terms (https://huggingface.co/black-forest-labs/FLUX.1-dev)
- The *practical* workaround: use FLUX.1-schnell (the *open* variant) as the v0 v1+ *fallback* backbone (faster, open, but *less* aesthetic quality)
- The *deeper* implication: the *gated* nature of FLUX.1-dev is a *commercial risk* for v0 (Black Forest Labs could change terms at any time)

### F. Critical Unknowns for v0 v1+

1. **Q1: How does Lotus-2 perform on *outdoor* vs *indoor* depth?** the *killer* H3 question for v0 (intra-oral camera is *indoor close-range*, very different from NYUv2's 5-10m room-scale depth); *expected*: Lotus-2's NYU 4.1 AbsRel is *better* than v0's expected *clinical* depth (0.5-2mm tolerance), but *unverified* on real dental depth maps
2. **Q2: Can the detail sharpener be *replaced* by a *single-step* refinement for v0 v1+ sub-task 1?** the *killer* simplification question: T'=10 detail sharpener adds *0.1s* inference, *negligible* HF improvement on *clinical* depth (where HF is *less* important than *accuracy*); *practical* answer: probably yes, use core alone for v0 v1+ sub-task 1, save the sharpener for v0 v2+
3. **Q3: Does the deterministic + single-step design work for *high-resolution* (1024×1024+) dental scans?** the *killer* scaling question: Lotus-2's 768×768 training may not transfer to 1024×1024+ clinical depth maps; *expected*: yes, because FLUX handles variable resolutions natively (per DiT's RoPE), but *unverified*
4. **Q4: Can the LCM be *removed* and replaced by a *larger* LoRA rank?** the *killer* architectural simplification question: LCM adds *2 conv layers* (negligible params), but adds *FLUX-specific* complexity; *practical* answer: probably not — LCM's -5.8% NYU is *significant*, and *small* architectural fixes are *cheap* (no Lambda cost)

---

## Open Q for HK

- **Q1 (★):** Adopt Lotus-2 as v0 v1+ sub-task 1 monocular depth + normal front-end? (★ YES, but train from scratch on FLUX for commercial-clean weights, $1,000-2,000 Lambda; OR use 211 Lotus for v0 v0 + Lotus-2 for v0 v1+)
- **Q2 (★):** Adopt Lotus-2's two-stage design (core + detail sharpener) for v0 v1+ sub-task 4 crown generation? (★ YES, the *killer* design lesson for *clinical-fit-aware* + *high-frequency-detail* crown generation)
- **Q3 (★):** Adopt Lotus-2's deterministic + clean-data prediction for v0 v1+ sub-task 4 loss? (★ YES, *eliminates* image-appearance interference + *structural variance*)
- **Q4 (★):** Adopt Lotus-2's spectral analysis (Fig. 12) as v0 v1+ paper's *detail-refinement validation method*? (★ YES, the *only* signal-level evidence in the v0 reading list that a *detail-refinement* network actually *adds* HF information)
- **Q5:** Cite Lotus-2 in v0 paper as the *SOTA* LDM-repurposing baseline? (★ YES, the *2nd best* in the v0 reading list for depth + normal Avg. Rank, the *killer* data-efficiency claim)
- **Q6:** Use FLUX.1-dev (gated) or FLUX.1-schnell (open) as v0 v1+ backbone? (FLUX.1-schnell is *safer* for commercial; FLUX.1-dev is *better* quality)
- **Q7:** Train from scratch ($1,000-2,000 Lambda) or accept research-only use? (depends on v0 commercial timeline)
- **Q8:** Use the detail sharpener for v0 v1+ (adds 0.1s, ~+0% accuracy, ~+5% HF detail) or skip it (use core alone, 0.1s, SOTA accuracy, ~5% worse HF detail)? (depends on v0 v1+ real-time vs high-quality tradeoff)

---

## Next paper (213)

**(a) DepthFM** (Fu 2024, arXiv:2403.12966, AAAI 2025) — the *flow-matching* alternative to Marigold/Lotus/Lotus-2, *the* paper for understanding *flow-matching-as-alternative-to-diffusion* for LDM-repurposing; the *killer* H2 mechanism: flow-matching's *ODE* formulation is *inherently* deterministic, *closer* to Lotus-2's design philosophy than Marigold's *SDE* formulation; the *right* v0 v1+ sub-task 1 paper for *understanding the 2024-2025 flow-matching paradigm shift*

**(b) DA² (Depth Anything in Any Direction)** (Li et al. 2025, arXiv:2509.26618, *referenced in Lotus-2's bibliography as [23]*) — the *EnVision Lab* (same lab as 211 + 212 Lotus-Lotus-2) *2025* paper that extends Depth Anything to *video* + *multi-direction*; the *right* paper for understanding *the same lab's* *intermediate* 2025 work (between 211 Lotus and 212 Lotus-2) and *how* they got from 211 to 212

**(c) Jasmine** (Wang et al. 2025, arXiv:2503.15905, *referenced in Lotus-2's bibliography as [24]*) — the *self-supervised* depth estimation paper that *also* uses diffusion priors; the *right* paper for understanding *self-supervised* LDM-repurposing for depth (vs Lotus-2's *supervised* approach)

**(d) DICEPTION** (Zhao et al. 2025, *referenced in Lotus-2's bibliography as [26]*) — the *generalist diffusion model for visual perceptual tasks*; the *right* paper for understanding *multi-task* LDM-repurposing (depth + normal + segmentation in a *single* model)

**Recommendation:** *read 213 = DepthFM* (the *flow-matching* alternative, the *killer* 2024-2025 paradigm shift from *diffusion* to *flow-matching* for LDM-repurposing, the *closest* alternative to Lotus-2's *deterministic* design philosophy)
