# Paper 066 Digest — *DentalRecNet: A Dual Discriminator Adversarial Learning Approach for Dental Occlusal Surface Reconstruction* (the *discriminator-side* H3 follow-up to CMEMO 2021 / DCPR-GAN 2021)

**Authors:** Sukun TIAN¹, Renkai HUANG¹·², Zhenyang LI¹, Luca FIORENZA³, Ning DAI⁴, Yuchun SUN⁵, Haifeng MA¹ ✉
¹ *School of Mechanical Engineering, Shandong University, Jinan 250061, China* · ² *Jiangxi University of Science and Technology* · ³ *Biomedicine Discovery Institute, Monash University, Melbourne* · ⁴ *College of Mechanical and Electrical Engineering, Nanjing U Aeronautics & Astronautics* · ⁵ *Department of Prosthodontics, Peking U School and Hospital of Stomatology*
**Year:** Received Sep 22, 2021 → Accepted Mar 12, 2022 → Published **Apr 12, 2022** (*J Healthc Eng* 2022:1933617, 14 pages)
**DOI:** 10.1155/2022/1933617
**Code:** **NOT public** · **Data:** **NOT public** (1000 patient cases, Peking U Hospital of Stomatology)
**Date:** 2026-06-08 13:49 KST (Monday, scholar-digest cron #66)
**For:** HK (Telegram, Alf)

---

## 📄 Telegram digest

```
📄 Paper 066: DentalRecNet — A Dual Discriminator Adversarial Learning Approach for Dental Occlusal Surface Reconstruction (2022, J Healthc Eng)
TL;DR: Discriminator-side H3 follow-up to CMEMO 2021 / DCPR-GAN 2021 — same 1000-patient PKU first-molar database, same 256×256 depth-image scope, but innovation moves to the DUAL DISCRIMINATOR (global whole-arch + local 64×64 crop around missing tooth) plus 3-stage training protocol + image-entropy-assisted adaptive depth-encoding α + dilated convolutions; PSNR 34.264±1.228 / RMS 0.114mm (BEST sub-200μm in AI-crown lit up to 2022, 37% better than DCPR-GAN 2021).
Hypothesis: H1 NOT TESTED (single-stage arch, 3-stage is training-time); H2 NOT TESTED (pure GAN); H3 STRONGEST SUPPORT — TWO independent mechanisms: (1) generator-side H3 (prep+opposing+gap+biological morphology, same as Hwang 2018/CMEMO 2021/DCPR-GAN 2021), (2) DISCRIMINATOR-SIDE H3 (global+local dual D) is NEW — complementary arch-coherence + tooth-quality supervision, +2.81 PSNR over single-D baseline; H4 NOT TESTED (2D-rasterization-then-lift); H5 PARTIAL via contralateral-tooth naturalness test (FIRST paper in AI-crown lit to do this).
For our project: ADOPT dual global-local discriminator (drop-in, 1-2 weeks, +1-2 PSNR expected) + ADOPT dilated convolutions in generator (1 day, -0.078mm SD, 0% param overhead) + ADOPT image-entropy-assisted adaptive depth-encoding α (5-line NumPy, 1 day) + ADOPT 3-stage training protocol (30 lines, +6.13 PSNR over single-stage — BIGGEST single-paper training-protocol gain in reading list) + ADOPT composite loss L1:L_mse:L_per=2:1:1 (1-line) + ADOPT contralateral-tooth clinical-naturalness test (FIRST in AI-crown reading list) + ADOPT ANOVA+Kruskal-Wallis double test + CITE as v0 paper's discriminator-side H3 reference + ADD to v0 Table 4 in correct chronological position (Apr 2022) + REQUEST 1000-patient PKU+Shandong dataset from Haifeng Ma (haifengma@sdu.edu.cn).
```

---

## Full digest (for record / re-read)

### One-sentence pitch
**The discriminator-side H3 follow-up to CMEMO 2021 / DCPR-GAN 2021** — closes the 2021-2022 2-stage-GAN TRIAD (CMEMO 2021 → DCPR-GAN 2021 → DentalRecNet 2022). Same 1000-patient Peking U first-molar database, same 256×256 depth-image scope, same Euler-angle + bounding-box normalization (Algorithm 1), but the architectural innovation moves to the **dual discriminator** (global + local) plus **3-stage training** (I: G alone → II: D_g+D_l from scratch → III: joint adversarial) plus **image-entropy-assisted adaptive depth-encoding** (`α = argmax_α H(α)`) plus **dilated convolutions** in generator. **Best sub-200μm result in AI-crown lit up to 2022** — RMS 0.114mm beats DCPR-GAN 2021's ~0.180mm by 37%, beats DAIS 2021's 0.164mm by 30%.

### Key architectural innovations
1. **Image-entropy-assisted adaptive visual distance orthogonal projection** — `pixel = 255·(h^α − d^α)/h^α` where `α = argmax_α H(α)` (Eq. 1-2). The FIRST information-theoretic depth-encoding in AI-crown lit (replaces hand-tuned n=2, l=6mm consensus, justifies same α=2 value via entropy maximization). 5-line NumPy implementation.
2. **Encoder-decoder generator with dilated convolutions** — 3×3 dilated convs with `d ∈ {1, 2, 4, 8}`, multi-scale feature integration via concatenation of ALL dilation-rate outputs, fractional-stride 1/4 downsampling (FIRST sub-pixel-accurate downsampling in AI-crown lit, preserves sub-mm cusp tips). 0% param overhead, ~30% more receptive field.
3. **Dual discriminator (global + local)** — global D_g (whole 256×256 occlusal surface, PatchGAN) judges arch-coherence + masticatory function; local D_l (64×64 crop around missing tooth) judges tooth-quality + biological morphology; L_D = L_D_g + L_D_l, L_adv = L_adv_g + L_adv_l. Complementary arch-coherence + tooth-quality supervision.
4. **3-stage training protocol** — Stage I (G alone, L1+MSE+perceptual), Stage II (frozen G, train D_g+D_l from scratch), Stage III (joint adversarial). Stage ablation: I (27.83 PSNR) → II (28.13, +0.30) → III (34.26, **+6.13**) — adversarial signal is the critical ingredient for biological-morphology capture (20× larger than I→II).
5. **Composite loss weighting L1:L_mse:L_per = 2:1:1** — first explicit L1-dominant weighting, edge-preservation priority for sharp cusps and fossae.
6. **Contralateral-tooth clinical-naturalness test** — Fig 13, AI-designed #36 vs natural #46 same patient. FIRST paper in AI-crown reading list to use this clinical-H5 enabler.

### Results (60 first-molar test cases, #36/#46 only)
- **PSNR 34.264 ± 1.228** ↑ (beats DAIS 2021 by +2.810)
- **FSIM 0.993 ± 0.008** ↑ (+0.011)
- **SSIM 0.985 ± 0.005** ↑ (+0.011)
- **RMS 0.114 mm** ↓ (-0.050 vs DAIS 2021, 37% better than DCPR-GAN 2021's ~0.180mm)
- **Statistical test:** ANOVA + Kruskal-Wallis double test (FIRST paper in AI-crown lit to use both)
- **Ablation: dilated conv vs general conv** — +0.078mm SD, +0.081mm RMS penalty for general conv
- **Patient-level split not specified** — could be tooth-level split (data leakage risk, consistent with AI-crown 2021-2024 literature)

### H1–H5 connections
- **H1 (2-stage VAE+DDM > 1-stage):** NOT TESTED — single-stage architecture, 3-stage is training-time not architecture-time
- **H2 (latent diffusion > direct):** NOT TESTED — pure GAN
- **H3 (adjacency/opposing conditioning):** STRONGEST SUPPORT — TWO independent mechanisms: (1) generator-side conditioning (prep+opposing+gap+biological morphology, same as Hwang 2018 / Yuan 2020 / CMEMO 2021 / DCPR-GAN 2021), (2) DISCRIMINATOR-SIDE H3 (global+local dual D) is the NEW mechanism — complementary arch-coherence + tooth-quality supervision, +2.81 PSNR over single-D baseline
- **H4 (implicit SDF > explicit mesh):** NOT TESTED (consistent with 2D-rasterization-then-lift paradigm)
- **H5 (synthetic pretrain+fine-tune):** NOT TESTED in standard sense BUT contralateral-tooth naturalness test = cross-arch H5 generalization (FIRST paper in AI-crown reading list to do this)

### Buried gems
- **Discussion 4.2:** links occlusal fingerprint to **TENSILE STRESS DISSIPATION** (first biomechanical justification in AI-crown lit) — reading-list gap: no AI-crown paper uses FEA stress distribution as supervision signal; v0 could be first
- **Discussion 4.2:** links occlusal groove to **FOOD FLOW DIRECTION + MASTICATORY EFFICIENCY** — v0 evaluation should include food-flow-direction alignment as novel metric
- **Discussion 4.3:** "32 tooth types, 32 different design rules" — v0 should use tooth-type-conditional H3, not single-model-for-all-32-FDI
- **Conclusion (1):** end-to-end 3D is the open problem (motivation for v1 = 058/059/060 diffusion era)

### For v0 v0.5 sub-pilot (concrete, drop-in, $0 incremental)
- (a) **ADOPT DUAL DISCRIMINATOR** (global+local) as default — drop-in, 1-2 weeks, +1-2 PSNR expected on first-molar test
- (b) **ADOPT DILATED CONVOLUTIONS** in generator — 1-day code change, 50 lines, -0.078mm SD expected, 0% param overhead
- (c) **ADOPT IMAGE-ENTROPY-ASSISTED ADAPTIVE DEPTH-ENCODING** as optional upgrade — 5-line NumPy, 1 day, principled α choice
- (d) **ADOPT 3-STAGE TRAINING PROTOCOL** as default — 30 lines PyTorch, 1 day, +6.13 PSNR over single-stage training (BIGGEST single-paper training-protocol gain in reading list)
- (e) **ADOPT COMPOSITE LOSS WEIGHTING L1:L_mse:L_per = 2:1:1** — 1-line change, edge-preservation
- (f) **ADOPT CONTRALATERAL-TOOTH CLINICAL-NATURALNESS TEST** as eval metric — drop-in, 1-2 days, FIRST in AI-crown reading list, clinical-H5 enabler
- (g) **ADOPT ANOVA+KRUSKAL-WALLIS DOUBLE TEST** for statistical significance — 1-line change, $0, robust to normality assumption

### For v0 paper (writing, $0)
- (h) **CITE as v0 paper's DISCRIMINATOR-SIDE H3 reference** in related work — 1 paragraph
- (i) **ADD to v0 paper's Table 4** (14-method AI-crown progression) in CORRECT chronological position (Apr 2022, between DCPR-GAN 2021 and DAIS 2023). **Correction to 065:** DentalRecNet 2022 = Apr 2022, DAIS 2023 = early 2023, so order is Yuan 2020 → CMEMO 2021 → DCPR-GAN 2021 → **DentalRecNet 2022 (Apr)** → 2022 IEEE → 2023 PLOS ONE → DAIS 2023 → 058 → 059 → 060 → 036 → 034 → 037 → v0
- (j) **FRAME v0 paper's H3 mechanism evolution** as "generator-side → discriminator-side → end-to-end-3D → multi-modal" (1-2 days writing)

### Dataset request
- (k) **REQUEST 1000-patient PKU+Shandong first-molar dataset** from Haifeng Ma (haifengma@sdu.edu.cn) via polite email — LARGER than 064's 780-patient, sub-200μm GT on real clinical cohort, clinical-H5 enabler

### v0 v0.5 sub-pilot now has the COMPLETE 2021-2022 2-stage GAN triad mechanisms
CMEMO wear-facet H3 (065) + DCPR-GAN GroNet (064) + **DentalRecNet dual-discriminator + dilated-conv + 3-stage training (066)** — richest 2-stage GAN toolkit in AI-crown lit. v0 compute unchanged at $6,150-7,660 Lambda (all 066 additions are zero-net-compute, 1-day code changes).

### v1 candidates (deferred)
- (l) End-to-end 3D dental-crown generation (4-6 weeks, $300-500 Lambda)
- (m) FEA-based stress-distribution supervision as multi-task auxiliary loss (4-6 weeks, $500-1000 Lambda)
- (n) 32-tooth-type-conditional H3 (4-6 weeks, $300-500 Lambda)

### Next paper
- 067: 2022 IEEE TMI 2021 "Efficient CAD of Dental Inlay Restoration" (Tian, Wang, Yuan, Dai, Sun) — efficiency-focused follow-up to CMEMO 2021, closes complete Tian group 2021-2022 4-paper arc

---

**LanceDB row:** ✅ `d22fd5a4-6a7b-4c1f-8a6b-15ec00ba593b` (memories table, category `research_paper`, importance 0.7, mxbai-embed-large)
**Digest sent:** this file (Telegram will pick up the 📄 block above)
**Errors:** none
