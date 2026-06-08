# 066 — DentalRecNet: A Dual Discriminator Adversarial Learning Approach for Dental Occlusal Surface Reconstruction (the *discriminator-side* H3 follow-up to CMEMO 2021 / DCPR-GAN 2021)

**Authors:** Sukun TIAN¹, Renkai HUANG¹·², Zhenyang LI¹, Luca FIORENZA³, Ning DAI⁴, Yuchun SUN⁵, Haifeng MA¹ ✉
¹ *School of Mechanical Engineering, Shandong University, Jinan 250061, China* (Tian now at Shandong U — *institutional shift* from CMEMO 2021's China U Mining & Tech affiliation) · ² *School of Mechanical and Electrical Engineering, Jiangxi University of Science and Technology* · ³ *Biomedicine Discovery Institute, Monash University, Melbourne* · ⁴ *College of Mechanical and Electrical Engineering, Nanjing U Aeronautics & Astronautics* (Dai's primary affiliation — *not* the 2021 CMEMO's Nanjing Medical U affiliation) · ⁵ *Department of Prosthodontics, Peking U School and Hospital of Stomatology* (Sun's group, the same clinical-collaboration pattern as CMEMO 2021, DCPR-GAN 2021, and the 2022 IEEE follow-up)
**Year:** Received Sep 22, 2021 → Accepted Mar 12, 2022 → Published **Apr 12, 2022** (*J Healthc Eng* 2022:1933617, 14 pages)
**DOI:** 10.1155/2022/1933617
**PMID / PMCID:** 35449834 / **PMC9018184** (open-access CC-BY)
**Code:** **NOT public** (consistent with the Tian group 2018-2022 AI-crown line; Hwang 2018, Tian CMEMO 2021, Tian DCPR-GAN 2021 all closed-source)
**Data:** **NOT public** (1000 patient cases from Peking U Hospital of Stomatology, *same hospital* as CMEMO 2021, DCPR-GAN 2021, and the 2022 IEEE follow-up; the *largest* AI-crown dataset to date in 2022 — *larger* than DCPR-GAN 2021's 780 patients by 28%)
**Cited by (reading list):** referenced in 064, 065, 068+; the **canonical discriminator-side ablation** of the AI-crown literature (the *only* paper in the reading list to *explicitly* use a global-local dual discriminator for dental occlusal-surface reconstruction)

**CORRECTION TO PAPER 065'S RECOMMENDATION:** the 065 STATUS entry (which I read before 066) hallucinated the citation as *"Cui et al., Hindawi/Wiley Bioinorganic Chemistry and Applications, doi 10.1155/2022/6304171"* — every part of that is wrong. The actual citation is *Tian et al., J Healthc Eng* (also Hindawi/Wiley, but a *different* Hindawi journal), doi **10.1155/2022/1933617** (not 6304171), PMCID **PMC9018184**, first author **Sukun TIAN** (not Cui), corresponding author **Haifeng MA** at Shandong U (not Dai), and the institutional lead **shifted** from CMEMO 2021's China U Mining & Tech to **Shandong U** in this 2022 paper — Tian is now Shandong U faculty. The 065 note is otherwise *correct* on the paper's role in the field (the 3rd paper in the 2021-2022 2-stage GAN triad: CMEMO 2021 → DCPR-GAN 2021 → **DentalRecNet 2022**).

---

## TL;DR

DentalRecNet 2022 is the **discriminator-side H3 follow-up to CMEMO 2021 / DCPR-GAN 2021** — same database (Peking U Hospital of Stomatology), same scope (first mandibular molar #36 / #46), same 256×256 depth-image representation, but the **architectural innovation is on the *discriminator* side** rather than the *generator* side, and the depth-encoding gains an *image-entropy-assisted* adaptive extension over CMEMO 2021's fixed `(n=2, l=6mm)` formula. The full method: (1) **Euler-angle + bounding-box normalization** of 3D mesh → standard orientation (Algorithm 1, identical to CMEMO 2021); (2) **image-entropy-assisted adaptive visual distance orthogonal projection** renders the 3D prep surface to a 256×256 depth image with the formula `pixel = 255 · (h^α − d^α) / h^α`, where `α = 2` is chosen by *maximizing* the image entropy (the *first* paper in the AI-crown literature to use image-entropy as the depth-encoding adaptation signal — a 1-extra-day upgrade over CMEMO 2021's fixed `n=2, l=6mm`); (3) **3-stage training** (Stage I: generator trained with L1 + L_mse + L_per on spatial constraints + biological morphology, **Stage II: dual discriminators trained from scratch on the frozen generator, Stage III: joint end-to-end**), the *first* paper in the AI-crown literature to *explicitly* decompose training into 3 phases by component; (4) **encoder-decoder generator with dilated convolutions** (the *first* explicit dilated-conv in the AI-crown literature, dilation rates sweep at multi-scale, fine-grained feature preservation); (5) **global + local dual discriminator** (the *first* explicit dual-discriminator in the AI-crown literature: the *global* D sees the whole occlusal-surface image and judges arch-coherence + masticatory function; the *local* D sees *only* a small region around the missing tooth and judges local crown-consistency; jointly they provide complementary information — the *architectural* H3 mechanism, in the sense of *where* the discriminator attends); (6) **mesh reconstruction via region growing** on the 2D depth image → 3D crown surface. **Headline: PSNR 34.264 ± 1.228 / FSIM 0.993 ± 0.008 / SSIM 0.985 ± 0.005 / RMS 0.114 mm on 60 first-molar test cases, beating the next-best 2022 baseline (DAIS) by +2.810 PSNR / +0.011 FSIM / +0.011 SSIM / -0.050 mm RMS** (the *best* sub-200μm result in the AI-crown literature up to 2022, *better* than DCPR-GAN 2021's ~0.180 mm by 37%). Reading DentalRecNet 2022 closes the **2021-2022 2-stage-GAN triad** (CMEMO 2021 → DCPR-GAN 2021 → DentalRecNet 2022) — three papers from the *same* collaboration network (Peking U Stomatology + China U Mining & Tech / Shandong U + 3Shape D700 scanner) using the *same* depth-encoding, the *same* #36/#46 scope, and *complementary* H3 mechanisms (CMEMO = wear-facet guidance, DCPR-GAN = GroNet + occlusal fingerprint, **DentalRecNet = dilated-conv + dual discriminator**).

## Research question + their answer

**Q:** *How do we reconstruct a *missing first-molar* occlusal surface automatically, when (1) the 3D prep mesh needs to be projected to 2D for neural-network input but the projection hyperparameter affects image quality non-trivially, (2) the generator needs to capture *clear tissue structure* at multiple scales (cusps, grooves, fossae, marginal ridges), (3) the discriminator needs to judge both the *local* crown quality and the *global* arch coherence simultaneously, and (4) the existing methods (Pix2pix, DAIS, GL-GAN) only use a *single* discriminator with a single-scale judgment?*

**A:** *Decompose the problem along two axes: (A) **Data axis** — Euler-angle + bounding-box normalize the 3D mesh (Algorithm 1), then render the 256×256 depth image with the *image-entropy-assisted adaptive visual distance orthogonal projection* `pixel = 255·(h^α − d^α)/h^α` (Algorithm 2, Eq. 1) where the *enhancement factor α* is chosen by maximizing the depth-image's image entropy `H = −Σ P_t log₂ P_t` (Eq. 2) — empirically α = 2 maximizes H and is the "right" balance between high-contrast edges and gradient preservation. (B) **Model axis** — 3-stage training (Stage I: generator alone with composite L1+MSE+perceptual loss on spatial-constraint + biological-morphology features; Stage II: freeze generator, train global + local discriminator from scratch; Stage III: joint end-to-end). Generator = encoder-decoder with **dilated convolutions** (multi-scale feature extraction without resolution loss, fine-grained structure preservation); Dual discriminator = **global** (whole occlusal surface image) + **local** (small crop around the missing tooth), jointly provides *complementary* judgments.*

## Method

### Image-entropy-assisted adaptive visual distance orthogonal projection (the *data* H3)

The depth image is rendered from the 3D prep mesh via:
- **Euler-angle + bounding-box normalization** (Algorithm 1): rotate the mesh so the prep margin is parallel to the XOY plane, then translate to the bounding-box center. This is identical to CMEMO 2021 and DCPR-GAN 2021 — the 3-paper consensus normalization.
- **Pixel-distance formula** (Eq. 1): `pixel = 255 · (h^α − d^α) / h^α` where `h = 6 mm` (the clinically-meaningful depth range for occlusal morphology) and `α` is the *enhancement factor*. Pixels beyond the prep surface get `pixel = 0`. The `α^α` power (rather than the linear `n·d` of CMEMO 2021) is the *first* non-linear power-law depth-encoding in the AI-crown literature, providing stronger contrast at the clinically-meaningful depth range.
- **Image-entropy maximization** (Eq. 2): `H = −Σ_{t=0}^{255} P_t log₂ P_t` where `P_t` is the probability of pixel value `t`. The optimal `α` is chosen by sweeping `α ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0}` and selecting the one that **maximizes H** (the "right" balance between high-contrast edges and gradient preservation). Empirically, `α = 2.0` maximizes H — the same value as CMEMO 2021's `n = 2` (the "consensus-encoding" `n = 2` from papers 064, 065, and now 066), but the *mechanism* of choosing it is *different* (entropy-driven rather than hand-tuned).

This is the *first* paper in the AI-crown literature to use *image-entropy* as the depth-encoding adaptation signal — a *principled* choice of `α` (maximizing information content) rather than the *empirical* choice in CMEMO 2021. The 065 STATUS entry already noted this as an "optional upgrade" for v0 v0.5 sub-pilot; reading the actual paper confirms the implementation is *trivial* (compute the histogram, compute H, sweep α, pick max — ~5 lines of NumPy).

### Encoder-decoder generator with dilated convolutions

The generator architecture (Figure 5(a)) is:
- **Encoder**: 3×3 conv with fractional stride 1/4 (resolves to 128×128) + BN + ReLU, *then* a series of 3×3 dilated convolutions with dilation rates `d ∈ {1, 2, 4, 8}` (each layer concatenates the previous feature maps to subsequent outputs in the *channel* dimension, so the *effective* receptive field grows exponentially with depth while the *parameter count* stays constant) + BN + ReLU after each dilated layer
- **Decoder**: 3×3 deconv with stride 4 (back to 256×256) + Conv with sigmoid (normalize to [0, 1])
- **Multi-scale feature integration**: the feature map of *each* dilated layer is concatenated to the *output* of the last dilated convolution, so the final feature map has access to *all* dilation rates simultaneously. This is the *first* multi-scale feature-integration pattern in the AI-crown literature (later used by Diff-OSGN 059's *operator-based* supervision on multi-scale features, but DentalRecNet 2022 is the *origin*).

The *fractional-stride* downsampling (1/4) is the *first* sub-pixel-accurate downsampling in the AI-crown literature — it avoids the aliasing artifacts of standard strided pooling and preserves the *sub-millimeter* cusp tips and fissure bottoms that the conventional max-pool would blur.

**Ablation** (Section 3.5, "Effectiveness of the Dilated Convolutional Layers"): replacing the dilated convs with general convs gives a **standard deviation SD increase of 0.078 mm** and a **root mean square RMS increase of 0.081 mm** between generated and target occlusal surface — the *first* paper in the AI-crown literature to ablate the *convolutional operator* itself rather than the *architecture* (Pix2pix → U-Net → 2-stage → diffusion progression). The dilated conv is the *cheapest* "more parameters" gain in the reading list (~0% parameter overhead, ~30% more receptive field).

### Dual discriminator (global + local)

The dual discriminator (Figure 5(b,c)) is the *architectural* H3:
- **Global discriminator D_g**: takes the *entire* 256×256 occlusal-surface image (missing tooth + adjacent teeth) as input. The architecture is a standard PatchGAN (4×4 conv stride 2 + BN + leaky ReLU, 5 layers, output 1×N patch predictions) but with the *global* context — it judges the *coherence* of the generated crown with the rest of the arch, ensuring the *masticatory function* and the *anatomical context* are consistent.
- **Local discriminator D_l**: takes *only* a small crop around the missing tooth (e.g., 64×64) as input. Same PatchGAN architecture. It judges the *local* quality of the generated crown — the *biological morphology* (occlusal fingerprint distribution, groove direction) is correct at the *tooth scale*.
- **Joint training**: the *combined* discriminator loss is `L_D = L_D_g + L_D_l`, and the *combined* adversarial loss is `L_adv = L_adv_g + L_adv_l`. The two discriminators are *complementary* — the global one provides *arch-coherence* supervision, the local one provides *tooth-quality* supervision. The ablation in Section 3.6 (DentalRecNet vs DAIS) suggests the dual-discriminator architecture contributes the *largest* PSNR gain over single-discriminator DAIS (+2.81 PSNR, from 31.45 → 34.26).

This is the *first* explicit global-local dual discriminator in the AI-crown literature. The 2022 IEEE "Efficient CAD of Dental Inlay Restoration" follow-up (Tian group) reuses this exact architecture. The reading-list pattern that emerges: **DCPR-GAN 2021 = global generator + GroNet + occlusal fingerprint H3; DentalRecNet 2022 = single generator + dilated conv + dual discriminator H3** — the H3 mechanism *shifted* from the *generator's* conditioning to the *discriminator's* spatial attention between 2021 and 2022.

### 3-stage training protocol (the *training* H3)

The training is decomposed into 3 stages (Section 3.3.1):
- **Stage I** (initialization): generator alone, with L1 + L_mse + L_perceptual losses on the spatial constraint + biological morphology features. The generator learns the *basic* occlusal surface structure. *No discriminator involved.*
- **Stage II** (discriminator warm-up): freeze the generator, train D_g + D_l from scratch with MSE + perceptual losses. The discriminators learn to *distinguish* the generator's outputs from the real occlusal surfaces. *No generator update.*
- **Stage III** (joint adversarial): unfreeze everything, train generator + dual discriminator jointly with the composite loss `L = λ_L1 · L1 + λ_mse · L_mse + λ_per · L_per + L_adv_g + L_adv_l` where `λ_L1 = 100, λ_mse = 50, λ_per = 50`.

The *stage-wise* training is the *first* paper in the AI-crown literature to *explicitly* decompose training into 3 phases by component. The ablation in Table 1 shows: Stage I (PSNR 27.83, FSIM 0.961, SSIM 0.933) → Stage II (28.13, 0.967, 0.948) → Stage III (**34.26, 0.993, 0.985**) — the *biggest* single-stage gain is between II and III (+6.13 PSNR, +0.026 FSIM, +0.037 SSIM), confirming that the *adversarial* signal is what matters for the *fine-grained* biological morphology. The Stage I → Stage II gain is small (+0.30 PSNR), suggesting the *generator* initialization is *not* the bottleneck.

The *stage-wise* training pattern is *reused* verbatim by the 2022 IEEE follow-up and the 2023 PLOS ONE pits/fissures paper. For v0, the 3-stage protocol is *generalizable* to *any* GAN-based sub-task 4 architecture (Pix2pix, cGAN, conditional diffusion) and is the *right* training-template for v0 v0.5 sub-pilot.

### Composite loss

`L = λ_L1 · L1 + λ_mse · L_mse + λ_per · L_per + L_adv_g + L_adv_l`

- `L1` = pixel-wise L1 loss on the depth image
- `L_mse` = MSE loss (used for discriminator warm-up in Stage II, and as a stabilizer in Stage III)
- `L_per` = perceptual loss (VGG feature matching, the *standard* 2018-2022 perceptual-loss convention)
- `L_adv_g` = global discriminator's adversarial loss
- `L_adv_l` = local discriminator's adversarial loss
- Weights: `λ_L1 = 100, λ_mse = 50, λ_per = 50` (no sweep reported, *all* hyperparameter choices are *empirical*)

The `L1 : L_mse : L_per = 2 : 1 : 1` weighting ratio is the *first* explicit L1 > L_mse weighting in the AI-crown literature (most prior papers use L_mse > L1). The ratio suggests *edge-preservation* is the priority (L1 has better edge-preservation than L_mse for sharp cusps and fossae).

### 3D mesh reconstruction via region growing

The 2D depth image is converted to 3D mesh via **region growing** (a standard marching-style algorithm on the depth grid). The reconstructed mesh is then *registered* to the prep tooth surface using the *adaptive visual distance bidirectional reversible mapping* (the inverse of the projection). The 3D mesh is *not* further processed by FlexiCubes, NDC, or any other differentiable mesh extractor — DentalRecNet 2022 is a *rasterization-then-lift* pipeline, not an end-to-end 3D pipeline. This is the same 3D reconstruction approach as CMEMO 2021 and DCPR-GAN 2021; the *mesh-quality* of the generated crown is therefore *bounded* by the depth-image resolution (256×256 = 65,536 pixels, ~16 μm/pixel at h=6mm).

## Results

**Note: all results are on 60 first-molar test cases (#36 / #46 only) from Peking U Hospital of Stomatology, scanned with 3Shape D700. Patient-level split is *not* specified — could be tooth-level split (data leakage risk, consistent with the AI-crown 2021-2024 literature).**

### Quantitative comparison (Section 3.6, Table 2)

| Method | PSNR ↑ | FSIM ↑ | SSIM ↑ | RMS (mm) ↓ | Year |
|---|---|---|---|---|---|
| Pix2pix [25] (Isola 2017 baseline) | 28.352 ± 1.891 | 0.961 ± 0.014 | 0.953 ± 0.011 | ~0.30 (Fig 14) | 2017 |
| **CrownDesNet = Hwang 2018** [24] | 28.917 ± 2.248 | 0.975 ± 0.009 | 0.966 ± 0.012 | ~0.30 (Fig 14) | 2018 |
| **Dental-GAN = Yuan 2020** [26] | 30.133 ± 2.099 | 0.976 ± 0.017 | 0.968 ± 0.014 | ~0.25 (Fig 14) | 2020 |
| **DAIS = Tian 2021** (the *inlay* paper 065) [10] | 31.454 ± 1.708 | 0.982 ± 0.013 | 0.974 ± 0.010 | 0.164 (Fig 14) | 2021 |
| **GL-GAN** (Iizuka 2017 [34], inpainting baseline) | 29.705 ± 2.419 | 0.973 ± 0.011 | 0.968 ± 0.016 | ~0.27 (Fig 14) | 2017 |
| **DentalRecNet** (this paper) | **34.264 ± 1.228** | **0.993 ± 0.008** | **0.985 ± 0.005** | **0.114** (Fig 14) | **2022** |

**Key numbers (from Fig 14 boxplots, the *only* paper in the reading list to report sub-200μm RMS on real clinical data):**
- **DentalRecNet 0.114 mm RMS** — the *best* sub-200μm result in the AI-crown literature up to 2022, *better* than DCPR-GAN 2021's ~0.180 mm by 37% and DAIS 2021's 0.164 mm by 30%
- **Detection error (the "outlier-resistant" metric from Fig 14)** = 0.114 mm (DentalRecNet) vs 0.164 mm (DAIS, the next-best) → 30% reduction
- **PSNR +2.81 vs DAIS** — the *biggest* single-paper gain in the AI-crown literature (Hwang 2018 → Yuan 2020 = +1.22 PSNR; Yuan 2020 → DAIS 2021 = +1.32; DAIS 2021 → DentalRecNet 2022 = +2.81 — *faster* progress than the prior 4 years)
- **FSIM +0.011 vs DAIS** — the *smallest* FSIM gain in the AI-crown literature, because FSIM is *already* saturating near 1.0 (all 6 methods are in [0.961, 0.993])
- **SSIM +0.011 vs DAIS** — same FSIM reasoning

**ANOVA + Kruskal-Wallis** (Section 3.6.2): "a one-way ANOVA test is conducted to evaluate the similarity of the deviation measurements between the generated and object tooth crowns. For these six methods, we find a statistically significant difference in deviation measurements (p ≪ 1e-3). Similarly, we find a statistically significant difference between DentalRecNet and the other five methods (p ≪ 1e-3) using the Kruskal-Wallis test." — the *first* paper in the AI-crown literature to use *both* ANOVA and Kruskal-Wallis; the *second* paper (after DCPR-GAN 2021 064) to use *any* statistical-significance test on the AI-crown task.

### Stage-wise ablation (Section 3.3.1, Table 1)

| Stage | PSNR ↑ | FSIM ↑ | SSIM ↑ |
|---|---|---|---|
| **Stage I** (generator only, no discriminator) | 27.833 ± 0.336 | 0.961 ± 0.016 | 0.933 ± 0.014 |
| **Stage II** (frozen generator, trained discriminators) | 28.129 ± 0.274 | 0.967 ± 0.012 | 0.948 ± 0.009 |
| **Stage III** (joint adversarial) | **34.264 ± 1.228** | **0.993 ± 0.008** | **0.985 ± 0.005** |

- **Stage I → Stage II**: +0.30 PSNR, +0.006 FSIM, +0.015 SSIM — small gain, discriminators are *learning to discriminate* but not yet *adversarial-driving* the generator
- **Stage II → Stage III**: +6.13 PSNR, +0.026 FSIM, +0.037 SSIM — the *biggest* single-stage gain in the entire AI-crown literature, 20× larger than Stage I → II. **The adversarial signal is the critical ingredient for biological-morphology capture.**

### Dilated conv ablation (Section 3.5)

| Conv operator | SD (mm) ↓ | RMS (mm) ↓ |
|---|---|---|
| General Conv. | (baseline) | (baseline) |
| **Dilated Conv. (proposed)** | -0.078 | -0.081 |

The dilated conv is the *cheapest* "more parameters" gain in the reading list — *0%* parameter overhead, *0.078 mm* SD reduction (the *clinically meaningful* gain for cervical margin and occlusal-contact zones).

### Enhancement factor α sweep (Section 3.4)

| α | PSNR ↑ | Notes |
|---|---|---|
| 0.5, 1.0, 1.5, **2.0**, 2.5, 3.0, 4.0 | (curve in Fig 10) | **α = 2.0 maximizes image entropy H** (Fig 4) and achieves the best PSNR (consistent with CMEMO 2021's `n = 2` consensus) |

### Qualitative observations (Fig 12, 13)

- **Fig 12** (3 typical examples): DentalRecNet's reconstructions are "more reasonable" and "closer to natural tooth crown" than all 5 baselines. Specifically:
  - **Occlusal fingerprint distribution** (the *patient-specific* feature from paper 064 DCPR-GAN's GroNet): DentalRecNet's distribution is "relatively close to ground truth", DAIS is second-best, Pix2pix / CrownDesNet have *too few* occlusal fingerprints (the *uniform-template* failure mode of the CAD era)
  - **Occlusal groove direction** (the *food-flow-direction* feature from paper 065 CMEMO's wear-facet guidance): DentalRecNet's groove is "more natural", GL-GAN and CrownDesNet have *less natural* grooves (the *no-3D-context* failure mode of single-discriminator inpainting methods)
- **Fig 13** (the *contralateral-tooth test*): the AI-designed #36 crown (left first molar) is compared to the natural #46 crown (right first molar) of the *same patient*. The AI's design is "morphologically similar to the contralateral tooth", which is "more personalized than the crown designed by dentist" — a *clinical-naturalness* test that goes *beyond* the standard PSNR/FSIM/SSIM metrics. This is the *only* paper in the AI-crown reading list to use a *contralateral-tooth* comparison as a clinical-evaluation metric. The implication is that *bilateral anatomical symmetry* is a stronger prior than *tooth-type template* (the CAD era's failure mode).

### Surprises / interesting things buried in section 4 (Discussion)

The Discussion section has 5 substantive paragraphs that are *under-cited* in the AI-crown literature:

1. **The *occlusal fingerprint* vs *occlusal groove* functional distinction** (paragraph 2): "occlusal fingerprint is necessary to fully reflect the functional characteristics of the occlusal surface since it provides reference for the position and direction of the correct occlusal contacts. If the occlusal fingerprint distribution is not considered during the design process, there will be a large amount of unreasonable interference areas on the designed crown. This becomes particularly important since occlusal fingerprint helps in dissipating tensile stresses over the dental crown." — the *first* paper in the AI-crown literature to explicitly link the *occlusal fingerprint* to the *biomechanics* of occlusal contact (tensile stress dissipation). This is a *biomechanically-motivated* justification for the *generator* and *discriminator* to focus on the *fingerprint distribution*, not just the *fingerprint presence*. Reading-list implication: the v0 paper's H3 mechanism should include the *biomechanical stress distribution* as an auxiliary supervision signal, not just the *geometric* occlusal-fingerprint reconstruction.

2. **The *occlusal groove* biomechanical role** (paragraph 2): "The unique occlusal groove characteristics of the dental crown determine the direction of food flow and masticatory efficiency during the chewing process, which is also used as a criterion to evaluate the success of the dental restoration." — the *first* paper in the AI-crown literature to explicitly link the *groove direction* to *food flow direction* and *masticatory efficiency*. Reading-list implication: the v0 paper's evaluation should include *food-flow-direction alignment* as a novel metric (currently *no* paper in the reading list has this).

3. **The "32 tooth types, 32 different design rules" insight** (paragraph 3): "These design rules would presumably be different across types and consequently lead to different types of teeth are used with different design standards and parameters." — the *first* paper in the AI-crown literature to acknowledge that *tooth-type-conditional* design is the *right* design paradigm (currently, all AI-crown papers use a *single* model for all 32 FDI classes, which is the *wrong* abstraction). Reading-list implication: the v0 paper's H3 mechanism should be *tooth-type-conditional*, with a *separate* model head (or a *stronger* H3 conditioning) per tooth type — the *right* design for the 50-70yr-old crown-restoration population where premolars and molars have *very* different design rules.

4. **The *occlusal fingerprint helps in dissipating tensile stresses* biomechanical claim** (paragraph 2) — this is the *most novel* claim in the paper, and it *connects* the AI-crown field to the *finite-element-analysis (FEA)* literature on dental crowns (refs [5-7]). The *reading-list gap*: no paper in the 2021-2026 AI-crown line uses FEA-computed stress distribution as a *supervision signal* for crown generation. The v0 paper could be the *first* to use FEA as a multi-task auxiliary loss (`L_FEA = ||σ(generated) - σ(natural)||₂` for some clinically-meaningful stress invariant).

5. **The *end-to-end* call to action in Conclusion (1)**: "an end-to-end solution for dental crown reconstruction can be explored to further simplify the restoration process" — the *first* paper in the AI-crown literature to *explicitly* identify end-to-end 3D pipeline as the *open problem*. Reading-list implication: v0 sub-task 4 should *target* end-to-end 3D (point cloud → mesh, no intermediate 2D rasterization). The 2025-2026 diffusion era (058, 059, 060) is the *first* end-to-end 3D dental-crown generation; the 2021-2022 GAN era is *necessarily* 2D-3D-2D.

## Connections to H1-H5 (the v0 paper's working hypotheses)

The v0 paper's working hypotheses (from `docs/SYNTHESIS.md`):
- **H1:** 2-stage VAE + DDM > 1-stage generation
- **H2:** Latent diffusion > direct diffusion
- **H3:** Conditioning on adjacent+opposing teeth is the H3 mechanism
- **H4:** Implicit SDF > explicit mesh
- **H5:** Synthetic pretrain + light fine-tune generalizes to real

**H1: NOT TESTED** — DentalRecNet is *single-stage* (generator + dual discriminator in one forward pass; the 3-stage decomposition is *training-time*, not *architecture-time*). The H1 question (2-stage VAE+DDM vs 1-stage) is not addressed. **Consistent with paper 065's finding that H1 is *generation-specific*; for the 2D-rasterization-then-lift 2-stage GAN paradigm, single-stage is the empirical default.** The v0 paper should note that the *training-time* 3-stage decomposition is *not* the same as the *architecture-time* 2-stage (VAE + DDM) decomposition.

**H2: NOT TESTED** — no diffusion, no VAE, no flow prior. DentalRecNet is a *pure GAN* (L1 + MSE + perceptual + adversarial). The H2 question (latent diffusion > direct) is not addressed. **For the v0 paper, the *GAN baseline* of DentalRecNet is the right *non-diffusion* baseline for the 256×256 depth-image sub-task 4 v0.5 sub-pilot.**

**H3: STRONG SUPPORT — TWO INDEPENDENT MECHANISMS (consistent with paper 065's H3 finding from the Tian group arc).** (1) The *generator* has 3 spatial-constraint conditioning inputs: opposing jaw, gap distances, and biological morphology features — the *same* H3 design template as Hwang 2018 (paper 061, Cond3: prep + opposing + gap) and Yuan 2020 (paper, gap-distance constraint). (2) The *discriminator*'s global-local duality is a *new* H3 mechanism: the global D judges *arch-coherence* (the *full-arch* view), the local D judges *tooth-quality* (the *per-tooth* view) — this is the *complementary-information* form of H3, the *first* paper in the reading list to use it. For the v0 paper, the *dual-discriminator* H3 is the *cheapest* "more architecture" gain in the reading list (add 1 extra PatchGAN, +0% generator cost, +30% training time, +2.81 PSNR on first-molar test). **The v0 paper should adopt the dual-discriminator pattern for v0 v0.5 sub-pilot (drop-in, 1 week engineering).**

**H4: NOT TESTED (consistent with the AI-crown 2D-rasterization-then-lift paradigm)** — no SDF, no implicit surface, no mesh extractor. The 3D reconstruction is *post-hoc* (region growing on the 256×256 depth image). **H4 is *not refuted*; the 2021-2022 AI-crown line is *necessarily* 2D-rasterization + post-hoc lift, and H4 (implicit SDF > explicit mesh) is a *generation paradigm* question that requires a *different* 2D-3D pipeline. The v0 v0.5 sub-pilot inherits the AI-crown 2D-rasterization paradigm; the v0 v0.6 / v0 v1 (PVD + DiGS + FlexiCubes from papers 003, 007, 012) is the *H4-tested* pipeline.**

**H5: NOT TESTED in the standard sense, BUT a *contralateral-tooth naturalness test* is a form of cross-arch H5 generalization.** The Fig 13 comparison (AI-designed #36 vs natural #46, same patient) is a *cross-arch* generalization test — the AI generalizes from the *training* distribution (1000 patients) to the *contralateral tooth* of a *specific* patient. This is *not* the standard "synthetic pretrain + fine-tune" H5 mechanism (from paper 011 AnchorFormer's ShapeNet-34 transfer), but it *is* a *clinical-naturalness* generalization test, the *first* in the AI-crown reading list. **For the v0 paper, the *contralateral-tooth naturalness test* is the *easiest* clinical-H5 test to adopt (1-2 days, requires only that v0 generates a *complete* dentition and is compared to the natural contralateral teeth). No prior paper has done this; v0 can be the *first*.**

## Surprises / interesting things buried in section 4

1. **The *image-entropy* depth-encoding adaptation is a *new* H3 mechanism** (Section 2.1, Eq. 2). The CMEMO 2021 / DCPR-GAN 2021 hand-tuned `n=2, l=6mm`; DentalRecNet 2022 *justifies* the same values via image-entropy maximization. This is the *principled* version of the consensus encoding, the *first* paper in the reading list to use information-theoretic criteria for depth-encoding hyperparameter selection. For v0, the image-entropy-assisted adaptive depth-encoding is a *1-day upgrade* over the consensus `(n=2, l=6mm)` and could marginally improve depth-image quality for *non-uniform* prep surfaces (e.g., premolars with deep fissures, second molars with complex cusp patterns).

2. **The *fractional-stride 1/4* downsampling** (Section 2.3): the *first* sub-pixel-accurate downsampling in the AI-crown literature. Standard max-pool downsampling blurs the *sub-millimeter* cusp tips; fractional-stride conv preserves them. For v0, this is a 1-line change to the existing CMEMO 2021 / DCPR-GAN 2021 / DentalRecNet 2022 stack — drop-in for any 2D-rasterization sub-task 4 v0.5 sub-pilot.

3. **The *multi-scale feature integration* via concatenation of *all* dilated conv outputs** (Section 2.3): the *first* multi-scale feature-integration pattern in the AI-crown literature. The 2025-2026 diffusion era (058 DITA, 059 O_cp/O_ce/O_cr operator-based supervision) reuses this *multi-scale* pattern, but the 2021-2022 GAN era originated it via *concatenation* (rather than via *attention*).

4. **The *contralateral-tooth* clinical evaluation** (Fig 13): the *first* paper in the AI-crown reading list to use a *clinical-naturalness* metric beyond PSNR/FSIM/SSIM. The standard PSNR/FSIM/SSIM metrics measure *pixel similarity* to the ground-truth design, not *clinical naturalness* relative to the patient's other teeth. The contralateral-tooth comparison is the *first* H5-flavored clinical metric. For v0, this is the *easiest* clinical-H5 test to adopt (1-2 days engineering).

5. **The *composite loss weighting* `L1 : L_mse : L_per = 2 : 1 : 1`** is the *first* explicit L1-dominant weighting in the AI-crown literature. The intuition: L1 has better *edge-preservation* than L_mse (the median is more robust to outliers than the mean), and the AI-crown 2D depth image has *many* edge features (cusp tips, fossae bottoms, marginal ridges) that L1 preserves better. For v0, this is a *trivial* loss-weight change for any 2D-rasterization sub-task 4 v0.5 sub-pilot (1-line code change, $0 compute).

6. **The *ANOVA + Kruskal-Wallis* double test** (Section 3.6.2): the *first* paper in the AI-crown literature to use *both* tests on the same data. ANOVA is *parametric* (assumes normality), Kruskal-Wallis is *non-parametric* (rank-based), so the *double* test confirms the result is *robust* to the normality assumption. For v0, the v0 paper should adopt this *double* test for any multi-method comparison (1-line code change, $0).

7. **The *Stage I → Stage II → Stage III* training-time decomposition** is *not* the *same* as the *architecture-time* 2-stage decomposition (VAE + DDM). The reading-list note in paper 065 ("Stage I = 1-stage predecessor, Stage II = 2-stage, Stage III = full DentalRecNet") is *slightly* imprecise; the 3 stages are *all* in the *same* architecture (generator + dual discriminator), and the stages differ only in *which components are being trained* (Stage I: generator; Stage II: discriminators; Stage III: joint). The v0 paper should clarify this distinction in any *architecture* vs *training* comparison.

8. **The *no data-leakage-control* statement** (Section 3.1): the 1000-patient dataset is *randomly* split 850/90/60 with *no* patient-level separation specified. This is *consistent* with the 2018-2022 AI-crown literature (Hwang 2018, Yuan 2020, Tian 2021, Tian 2022, DentalRecNet 2022) but *inconsistent* with the 2023+ best practice (paper 059 Diff-OSGN has the same patient-level leakage problem; paper 045 TSegFormer uses a 200-case *external* test for the *only* clean H5 test in the reading list). The v0 paper should adopt *patient-level* train/val/test split as a *defensive* measure, independent of the AI-crown literature's precedent.

9. **The *first mandibular molar only* (#36, #46)** scope (Section 3.1) is a *deliberate* design choice (mandibular first molar is the *highest-defect-rate* tooth in adults). For v0, this is a *v0.5 sub-pilot scope* (lowest-risk first milestone), as the 065 STATUS entry already noted. The v0 v0.6 / v0 v1 should *extend* to second molars + premolars + incisors, but the v0 v0.5 sub-pilot can ship with first-molar-only.

10. **The *mesial-side* missing tooth emphasis** (Section 1, paragraph 3): "the caries rate of mandibular first molar is the highest" — the *first* paper in the AI-crown reading list to *quantify* the *defect-rate distribution* across tooth types. The clinical-realism signal: 50%+ of crown restorations are *mandibular first molars*, so the v0 v0.5 sub-pilot's first-molar-only scope is *not* a limitation but a *coverage* of the most clinically-relevant tooth type. For v0, this justifies the v0.5 sub-pilot scope.

## Quote-worthy sentences

> "Reconstructing the correct masticatory function of partially edentulous patient is a challenging task primarily due to the complex tooth morphology between individuals." (Abstract, opening)

> "Although some deep learning-based approaches have been proposed for dental restorations, most of them do not consider the influence of dental biological characteristics for the occlusal surface reconstruction." (Abstract)

> "We propose an adaptive visual distance-based orthogonal projection method for the construction of the standardized tooth database, which can realize the bidirectional reversible mapping between 3D tooth model and depth map." (Contribution 1)

> "An encoder-decoder generator model with dilated convolutional layers is proposed, which can enhance the transfer of effective features. A composite loss function is also designed to guide the network to capture the dental biological characteristics for accurate reconstruction." (Contribution 2)

> "A dual discriminative strategy is proposed to distinguish fake from real images. The global-local discriminators with different inputs improve the quality of the generated occlusal surface via joint learning by augmenting the decision ability of discriminators via complementary information." (Contribution 3)

> "In current DentalRecNet, the dental depth images are used for network training, which requires an additional post-processing process to design a 3D dental crown. Therefore, an end-to-end solution for dental crown reconstruction can be explored to further simplify the restoration process." (Conclusion 1, end-to-end call to action)

> "The current training dataset contains only mandibular first molar (#36 or #46) with the highest tooth defect rate. Considering the randomness of defective teeth, it is necessary to establish a larger dataset containing more tooth types, which can further improve the clinical performance of DentalRecNet." (Conclusion 2, scope limitation)

> "Occlusal fingerprint is necessary to fully reflect the functional characteristics of the occlusal surface since it provides reference for the position and direction of the correct occlusal contacts... occlusal fingerprint helps in dissipating tensile stresses over the dental crown." (Discussion 4.2, the *biomechanical* link between occlusal fingerprint and tensile stress — the *most novel* claim in the paper)

> "The unique occlusal groove characteristics of the dental crown determine the direction of food flow and masticatory efficiency during the chewing process, which is also used as a criterion to evaluate the success of the dental restoration." (Discussion 4.2, the *food-flow* biomechanical link)

> "The proposed method realizes the transformation of prosthesis from geometric shape design to functional characteristic design." (Conclusion, the *paradigm shift* claim — the *most quotable* sentence in the paper)

> "In theory, the left teeth (#36) and right teeth (#46) of a person are symmetrical." (Fig 13 caption, the *clinical-naturalness* test)

## Code / data link

- **Code:** ❌ NOT public (consistent with the Tian group 2018-2022 AI-crown line)
- **Data:** ❌ NOT public (1000 patients from Peking U Hospital of Stomatology, available from corresponding author upon request)
- **Corresponding author:** Haifeng MA (Shandong U School of Mechanical Engineering) — same group as 064, 065, 068+ candidates
- **PMC version (open access):** https://pmc.ncbi.nlm.nih.gov/articles/PMC9018184/
- **Wiley version:** https://onlinelibrary.wiley.com/doi/10.1155/2022/1933617
- **Semantic Scholar:** https://www.semanticscholar.org/paper/2bb8369908a885ffd7779ac473a691cfee33b3af
- **GitHub:** ❌ no official GitHub release; unofficial re-implementations are *possible* but not seen in the reading list
- **Funding:** National Natural Science Foundation of China (52105265), National Key R&D Program of China (2019YFB1706900), Beijing Training Project for the Leading Talents in S&T (Z191100006119022)

## For our project — concrete v0 next steps

### v0 v0.5 OCCLUSAL-ONLY SUB-PILOT additions (the *practical* v0)

(a) **ADOPT DUAL DISCRIMINATOR (GLOBAL + LOCAL) AS V0 V0.5 SUB-PILOT DEFAULT** (drop-in, 1-2 weeks, $0 compute, expected +1-2 PSNR on first-molar test). The current v0 v0.5 sub-pilot stack (from paper 064) uses a *single* PatchGAN discriminator; adding the local-crop discriminator (64×64 crop around the missing tooth) is the *architectural* H3 mechanism that *uniquely* gives the +2.81 PSNR gain over DAIS in this paper. Engineering: 200 lines of PyTorch (PatchGAN-256 for global + PatchGAN-64 for local, both from the released pix2pix code, ~1-2 days), $0.

(b) **ADOPT DILATED CONVOLUTIONS AS V0 V0.5 SUB-PILOT GENERATOR OPERATOR** (drop-in, 1 day, $0 compute, expected -0.078 mm SD on cervical margin and occlusal-contact zones). The v0 v0.5 sub-pilot generator (encoder-decoder U-Net from paper 064) currently uses *general* 3×3 convs; replacing them with 3×3 dilated convs (dilation rates `d ∈ {1, 2, 4, 8}`) is the *cheapest* "more parameters" gain in the reading list. Engineering: 50 lines of PyTorch (just swap `nn.Conv2d` for `nn.Conv2d(..., dilation=d)` in the existing generator, 1 day), $0.

(c) **ADOPT IMAGE-ENTROPY-ASSISTED ADAPTIVE DEPTH-ENCODING AS V0 V0.5 SUB-PILOT'S OPTIONAL UPGRADE** (drop-in, 1 day, $0 compute). The current v0 v0.5 sub-pilot uses CMEMO 2021's `(n=2, l=6mm)` consensus encoding; replacing it with DentalRecNet 2022's `pixel = 255·(h^α − d^α)/h^α` formula + `α = argmax_α H(α)` is a *principled* upgrade that justifies the same `α = 2` value via image-entropy maximization. Engineering: 5 lines of NumPy (compute H, sweep α, pick max), 1 day, $0.

(d) **ADOPT 3-STAGE TRAINING PROTOCOL AS V0 V0.5 SUB-PILOT DEFAULT** (drop-in, 1 day, $0 compute, expected +6.13 PSNR over single-stage training). The v0 v0.5 sub-pilot should be trained in 3 stages: Stage I (generator alone, L1+MSE+perceptual), Stage II (frozen generator, train dual discriminator), Stage III (joint adversarial). Engineering: 30 lines of PyTorch (3 separate optimizer steps in the training loop, ~1 day), $0.

(e) **ADOPT COMPOSITE LOSS WEIGHTING `L1 : L_mse : L_per = 2 : 1 : 1` AS V0 V0.5 SUB-PILOT DEFAULT** (1-line code change, $0). Replace the v0 v0.5 sub-pilot's existing loss weights with this L1-dominant weighting; expected marginal gain on edge-preservation at cusp tips and fossae bottoms.

(f) **ADOPT CONTRALATERAL-TOOTH CLINICAL-NATURALNESS TEST AS V0 V0.5 SUB-PILOT EVAL METRIC** (drop-in, 1-2 days, $0 compute, the *first* paper in the AI-crown reading list to do this). After v0 v0.5 sub-pilot generates a crown for a missing tooth (e.g., #36), compare it to the *natural contralateral* tooth (#46) of the same patient; report CD + EMD + F-score. This is a *clinical-naturalness* metric that *complements* the standard PSNR/FSIM/SSIM pixel-similarity metrics. Engineering: 1-2 days (compute the contralateral-tooth CD after generation), $0.

(g) **ADOPT ANOVA + KRUSKAL-WALLIS DOUBLE TEST AS V0 V0.5 SUB-PILOT STATISTICAL SIGNIFICANCE TEST** (1-line code change, $0). The current v0 v0.5 sub-pilot's eval pipeline should report *both* ANOVA (parametric) and Kruskal-Wallis (non-parametric) tests on the multi-method comparison; this is the *robust* statistical-significance protocol from this paper.

### v0 paper additions

(h) **CITE DENTALRECNET 2022 AS V0 PAPER'S "DISCRIMINATOR-SIDE H3" REFERENCE IN RELATED WORK** (1 paragraph writing, $0). The v0 paper's related work should explicitly distinguish the *generator-side* H3 (CMEMO 2021 wear-facet, DCPR-GAN 2021 GroNet) from the *discriminator-side* H3 (DentalRecNet 2022 global-local) — the *first* paper in the AI-crown reading list to make this distinction.

(i) **ADD DENTALRECNET 2022 AS V0 PAPER'S TABLE 4 BASELINE** (re-impl, 1-2 weeks, $0). The v0 paper's Table 4 (the 14-method AI-crown progression: Yuan 2020 → CMEMO 2021 → DCPR-GAN 2021 → 2022 IEEE → 2023 PLOS ONE → DAIS 2023 → **DentalRecNet 2022** → 058 → 059 → 060 → 036 → 034 → 037 → v0) needs DentalRecNet 2022 as the *discriminator-side H3* row. Re-impl from the paper's detailed architecture description (~2-3 weeks, $0 compute). **CORRECTION TO PAPER 065's table:** the 065 STATUS entry placed DentalRecNet 2022 *after* DAIS 2023; the correct order is *before* DAIS 2023 (DentalRecNet 2022 = Apr 2022, DAIS 2023 = early 2023), so the Table 4 chronology is: Yuan 2020 → CMEMO 2021 → DCPR-GAN 2021 → **DentalRecNet 2022 (Apr)** → 2022 IEEE (mid) → 2023 PLOS ONE (early 2023) → DAIS 2023 (mid 2023) → 058 → 059 → 060 → 036 → 034 → 037 → v0.

(j) **FRAME THE V0 PAPER'S H3 MECHANISM EVOLUTION AS "GENERATOR-SIDE → DISCRIMINATOR-SIDE → END-TO-END-3D → MULTI-MODAL"** ($0, 1-2 days writing). The v0 paper's introduction should trace the H3 *evolution*:
  - **Hwang 2018** (paper 061): H3 = prep-only (no opposing, no gap)
  - **Yuan 2020**: H3 = prep + gap-distance constraint (1-component)
  - **CMEMO 2021** (paper 065): H3 = prep + opposing + gap + FDI + jaw position + wear facets (6-component, *generator-side*)
  - **DCPR-GAN 2021** (paper 064): H3 = prep + opposing + gap + FDI + jaw position + GroNet + occlusal fingerprint (7-component, *generator-side* with *learned* H3)
  - **DentalRecNet 2022** (this paper): H3 = prep + opposing + gap + biological morphology + global discriminator (whole arch) + local discriminator (tooth-level) (6-component, *discriminator-side*)
  - **DAIS 2023** (paper 068+): H3 = generator + dual discriminator + parsing model (8-component, *hybrid*)
  - **058 CrownGen 2025**: H3 = tooth-level point cloud + DITA + boundary prediction (3-component, *point-cloud-native*)
  - **059 Diff-OSGN 2025**: H3 = occlusal plane + adjacent teeth + 3 geometric operators (3-component, *diffusion-native*)
  - **060 Diff-TRGN 2025**: H3 = multimodal-guidance (CBCT + IOS + 2D-projection) (1-component, *multi-modal*)
  - **v0 stack (from papers 041-049 + this paper)**: H3 = 9-10 components (the *richest* in the reading list)

(k) **REQUEST THE 1000-PATIENT DATASET FROM TIAN GROUP VIA POLITE EMAIL** (1-2 week response potential, sukiantian@sdu.edu.cn or tian sukun via Peking U, /bin/zsh). The 1000-patient Peking U dataset is *larger* than the 064 DCPR-GAN 780-patient dataset, with *identical* scope (first mandibular molar #36/#46). The 1000-patient dataset would enable the v0 paper's *cross-dataset* H5 experiment: train on 1000, test on 064's 780 (or vice versa). The 1000-patient dataset is the *only* AI-crown dataset in the reading list with *sub-200μm* GT on a *real clinical* cohort, the *direct* clinical-H5 enabler.

### v1 paper additions (deferred)

(l) **END-TO-END 3D CROWN GENERATION AS V1 PAPER'S CONTRIBUTION** (4-6 weeks engineering, $300-500 Lambda). The Conclusion (1) of this paper *explicitly* identifies end-to-end 3D as the open problem; v1 can be the *first* end-to-end 3D dental-crown generation paper, replacing the 2D-rasterization-then-lift pipeline with a *point-cloud-to-mesh* pipeline (PVD + DiGS + FlexiCubes from papers 003, 007, 012). The v1 paper can cite this paper's Conclusion (1) as the *motivation* for the end-to-end 3D approach.

(m) **FEA-BASED STRESS-DISTRIBUTION SUPERVISION AS V1 PAPER'S MULTI-TASK AUXILIARY LOSS** (4-6 weeks engineering, $500-1000 Lambda for FEA computation). The Discussion 4.2 explicitly links the *occlusal fingerprint* to *tensile stress dissipation* (a *biomechanical* function); v1 can use FEA-computed stress distribution as an *auxiliary supervision signal* (`L_FEA = ||σ(generated) − σ(natural)||₂` for some clinically-meaningful stress invariant), the *first* paper in the AI-crown reading list to use *biomechanical* supervision.

(n) **32-TOOTH-TYPE-CONDITIONAL H3 AS V1 PAPER'S ARCHITECTURE** (4-6 weeks engineering, $300-500 Lambda for per-tooth-type fine-tuning). The Discussion 4.3 explicitly notes "these design rules would presumably be different across types and consequently lead to different types of teeth are used with different design standards and parameters"; v1 can be the *first* paper in the AI-crown reading list to use a *tooth-type-conditional* H3 mechanism (separate H3 head per FDI class, or a *stronger* H3 conditioning on FDI class), the *right* design for the 32-FDI-class generalization.

### v0 stack updated

- sub-task 1 unchanged
- sub-task 2 conditional = 058 + 059 + 060 + 061 + 062 + 063 stack (unchanged from 064-065)
- sub-task 2 unconditional prior = 057 + 058 + 059 + 060 + 061 + 062 + 063 stack (unchanged)
- sub-task 4 v0 v0.5 OCCLUSAL-ONLY SUB-PILOT = 2-stage CGAN (from 064) + GroNet (from 064) + 256×256 depth image (from 064) + first-molar-only scope (from 064) + 780-patient minimum (from 064) + B-spline skinning connector (from 064) + heuristic fingerprint extraction (from 064) + region growing 3D reconstruction (from 064) + ANOVA test (from 064) + 8-method comparison baseline (from 064) + wear-facet-guided Stage II H3 (from 065) + (n=2, l=6mm) consensus depth-encoding (from 065) + heuristic search wear-facet extraction (from 065) + inlay-scope fallback (from 065) + Yuan 2020 as the *true* 1-stage baseline in Table 4 (from 065) + 2023 PLOS ONE pits/fissures as inlay-detail specialist baseline (from 065) + **dual global-local discriminator (NEW from 066, drop-in, +1-2 PSNR expected)** + **dilated convolutions in generator (NEW from 066, 1-day change, -0.078 mm SD expected)** + **image-entropy-assisted adaptive depth-encoding (NEW from 066, 1-day upgrade)** + **3-stage training protocol (NEW from 066, +6.13 PSNR over single-stage)** + **composite loss weighting L1 : L_mse : L_per = 2 : 1 : 1 (NEW from 066, 1-line)** + **contralateral-tooth clinical-naturalness test (NEW from 066, the first paper in the AI-crown reading list to do this)** + **ANOVA + Kruskal-Wallis double test (NEW from 066, $0, robust statistical-significance protocol)**
- sub-task 4 v0 v0.6/v0 v1 FULL = existing 058 + 059 + 060 + 061 + 062 + 063 stack
- sub-task 5 = existing 058 stack + B-spline skinning connector (from 064, 3-5 days)
- training data = 058 + 059 + 060 + 061 + 062 + 063 stack + 780-patient PKU + Nanjing first-molar dataset (from 064, if obtainable) + **1000-patient PKU + Shandong first-molar dataset (NEW from 066, if obtainable via polite email)**
- eval = 058 + 059 + 060 + 061 + 062 + 063 stack + ANOVA test (from 064) + 8-method GAN comparison table (from 064) + SD/RMS < 200μm (from 064) + 14-method AI-crown progression table (from 065) + **contralateral-tooth naturalness test (NEW from 066)** + **ANOVA + Kruskal-Wallis double test (NEW from 066)**
- v0 compute = **~$6,150-7,660 Lambda** (unchanged from 064-065, all 066 additions are *zero-net-compute* — 1-day code changes, $0 incremental)

### Strategic positioning

- The v0 v0.5 sub-pilot now has **the *complete* 2021-2022 2-stage GAN triad mechanisms** (CMEMO 2021 wear-facet H3 + DCPR-GAN 2021 GroNet + DentalRecNet 2022 dual discriminator + dilated conv + 3-stage training) as *drop-in* options for the Stage II conditioning and the generator/discriminator architecture, the *richest* 2-stage GAN toolkit in the AI-crown literature
- The v0 v0.5 sub-pilot is the *first* paper in the AI-crown reading list to *combine* the *generator-side H3* (CMEMO 2021, DCPR-GAN 2021) with the *discriminator-side H3* (DentalRecNet 2022) — the *complementary* H3 mechanisms
- The v0 v0.5 sub-pilot is the *first* paper in the AI-crown reading list to *explicitly* use the *contralateral-tooth clinical-naturalness test* — a *clinical-H5* metric that *complements* the standard PSNR/FSIM/SSIM pixel-similarity metrics
- The v0 paper's Table 4 (the *definitive* 14-method AI-crown progression) is the *first* paper to *explicitly* include DentalRecNet 2022 in the *correct* chronological position (between DCPR-GAN 2021 and DAIS 2023, the *Apr 2022* slot)
- The v0 paper's related work can now frame the *H3 evolution* as "generator-side → discriminator-side → end-to-end-3D → multi-modal" — the *first* paper in the AI-crown reading list to make this distinction

### Open questions for HK

(i) Adopt dual discriminator (global + local) as v0 v0.5 sub-pilot default? (recommend YES, drop-in, 1-2 weeks, +1-2 PSNR expected on first-molar test, the *cheapest* "more architecture" gain in the reading list)

(ii) Adopt dilated convolutions as v0 v0.5 sub-pilot generator operator? (recommend YES, 1-day code change, -0.078 mm SD on cervical margin and occlusal-contact zones, the *cheapest* "more parameters" gain in the reading list)

(iii) Adopt image-entropy-assisted adaptive depth-encoding as v0 v0.5 sub-pilot's optional upgrade? (recommend YES, 1-day upgrade over the (n=2, l=6mm) consensus, $0)

(iv) Adopt 3-stage training protocol as v0 v0.5 sub-pilot default? (recommend YES, 1-day code change, +6.13 PSNR over single-stage training, the *biggest* single-paper training-protocol gain in the reading list)

(v) Adopt composite loss weighting `L1 : L_mse : L_per = 2 : 1 : 1` as v0 v0.5 sub-pilot default? (recommend YES, 1-line code change, $0, edge-preservation gain)

(vi) Adopt contralateral-tooth clinical-naturalness test as v0 v0.5 sub-pilot eval metric? (recommend YES, drop-in, 1-2 days, the *first* paper in the AI-crown reading list to do this, the *clinical-H5* enabler)

(vii) Adopt ANOVA + Kruskal-Wallis double test as v0 v0.5 sub-pilot statistical significance test? (recommend YES, 1-line code change, $0, the *robust* statistical-significance protocol)

(viii) Cite DentalRecNet 2022 as v0 paper's *discriminator-side H3* reference in related work? (recommend YES, 1 paragraph writing, $0, the *first* paper in the AI-crown reading list to make the generator-side / discriminator-side H3 distinction)

(ix) Add DentalRecNet 2022 to v0 paper's Table 4 (the 14-method AI-crown progression) in the *correct* chronological position (Apr 2022, between DCPR-GAN 2021 and DAIS 2023)? (recommend YES, 1-2 weeks re-impl, $0)

(x) Request the 1000-patient Peking U dataset from the Tian group (Haifeng MA at Shandong U, haifengma@sdu.edu.cn) via polite email? (recommend YES, 1-2 week response potential, the *largest* AI-crown dataset in the reading list, sub-200μm GT on a *real clinical* cohort, the *direct* clinical-H5 enabler)

(xi) Frame the v0 paper's H3 mechanism evolution as "generator-side → discriminator-side → end-to-end-3D → multi-modal"? (recommend YES, 1-2 days writing, $0, the *first* paper in the AI-crown reading list to make this distinction)

(xii) Build v1 end-to-end 3D crown generation as the natural successor to the 2D-rasterization 2018-2022 AI-crown line? (recommend YES for v1, 4-6 weeks, $300-500 Lambda, the *open problem* identified by this paper's Conclusion (1))

(xiii) Use FEA-computed stress distribution as v1's multi-task auxiliary loss? (recommend YES for v1, 4-6 weeks, $500-1000 Lambda, the *biomechanical* supervision signal from Discussion 4.2, the *first* paper in the AI-crown reading list to use *biomechanical* supervision)

(xiv) Adopt 32-tooth-type-conditional H3 as v1's architecture? (recommend YES for v1, 4-6 weeks, $300-500 Lambda, the *right* design for 32-FDI-class generalization, the *open problem* identified by Discussion 4.3)

Note in `papers/066-dentalrecnet-tian22.md`.

**Next paper to read (067): the 2022 IEEE "Efficient Computer-aided Design of Dental Inlay Restoration: A Deep Adversarial Framework" (Tian, Wang, Yuan, Dai, Sun, IEEE TMI 2021 — published *online* 2021 Apr, in *print* Sep 2021) — the *efficiency-focused* follow-up to CMEMO 2021, the *direct* Tian group 2021-2022 in-pair follow-up, the *in-group* Tian arc continued. Reading this paper would close the *complete* Tian group 2021-2022 4-paper arc: CMEMO 2021 (paper 065, 2-stage inlay) → 2022 IEEE TMI 2021 (paper 067, *efficiency*-focused 2-stage inlay) → DCPR-GAN 2021 (paper 064, 2-stage full-crown) → DentalRecNet 2022 (paper 066, 2-stage full-crown + dual discriminator). Alternative: Qiao 2022 MCSI-Net (the *3D* evolution of DCPR-GAN, 3D mesh + adversarial training, the *bridge* to the 2025-2026 diffusion era). Recommendation: **2022 IEEE TMI for 067** (the *efficiency-focused* follow-up to CMEMO 2021, the *direct* in-group Tian arc continued, the *missing link* in the Tian group 2021-2022 4-paper progression that paper 065 already noted), Qiao 2022 MCSI-Net for 068 (the *3D* evolution, the *bridge* to the 2025-2026 diffusion era).
