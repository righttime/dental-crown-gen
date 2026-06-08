# 067 — DAIS: Deep Adversarial-driven dental Inlay reStoration (the *Wasserstein + DuNet + GroNet* inlay-specific member of the Tian group 2021-2022 quad)

**Authors:** Sukun TIAN¹*†, Miaohui WANG²*, Fulai YUAN³, Ning DAI⁴ ✉, Yuchun SUN⁵, Wuyuan XIE⁶, Jing QIN⁷ ✉
¹ *Peking U School of Stomatology, Beijing* (Tian's primary 2021 affiliation, *institutional shift* from CMEMO 2021's China U Mining & Tech) · ² *Shenzhen U* (Wang now at Shenzhen U — *institutional shift* from 064 DCPR-GAN's Peking U affiliation) · ³ *Peking U / China U Mining & Tech* (Yuan) · ⁴ *Nanjing U Aeronautics & Astronautics* (Dai, *not* 064's Nanjing Medical U) · ⁵ *Peking U School of Stomatology, Dept of Prosthodontics* (Sun, clinical-collaboration pattern with 064/065/066) · ⁶ *Peking U* (Xie, *new* collaborator on this paper) · ⁷ *The Hong Kong Polytechnic U, Dept of Health Technology and Informatics* (Qin, **the *international* co-PI of the paper** — a Hong Kong-based senior co-PI, *new* to the Tian group reading list; Qin is a well-known medical-imaging AI researcher with 200+ papers)
*equal contribution · † corresponding

**Year:** Received 2021 → **Accepted May 5, 2021** → Published *online* 2021 May 5 → **IEEE TMI 40(9):2415-2427, Sep 2021** (Epub 2021 Aug 31)
**Venue:** **IEEE Transactions on Medical Imaging** (Q1 medical imaging, IF ~10-11, *the* top medical-imaging journal; *higher* impact than 064's J Biomed Health Inform or 066's J Healthc Eng; *first* TMI paper in the AI-crown reading list)
**DOI:** 10.1109/TMI.2021.3077334
**PMID:** 33945473
**Code:** **NOT public** (consistent with 064, 065, 066 — *all* Tian group AI-crown papers 2018-2022 are closed-source)
**Data:** **NOT public** (830 tooth samples from Peking U + Nanjing U, *3Shape D700* scanner, **the *largest* inlay-specific AI-crown dataset in the reading list** — *larger* than 064's 780 full-crown patients by 6%, *larger* than 065's inlay dataset by 6% as well)
**Cited by (reading list):** referenced in 066, 068+; the **CANONICAL** *3-loss* (L1 + L_histogram + L_GroNet) **+ 3-stage training protocol** of the AI-crown literature, and the *highest-impact* Tian group paper (TMI vs JBHI vs J Healthc Eng vs China Mech Eng)

**POSITION IN THE READING LIST:** the *4th paper* in the Tian group 2021-2022 4-paper arc, the *efficiency-focused* member, the *Wasserstein + DuNet + GroNet* inlay-specific paper:
- 065 CMEMO (Feb 2021, China Mech Eng, inlay, hierarchical coarse-to-fine + wear-facet H3, the *CGAN* ancestor)
- **067 DAIS (this paper, May/Sep 2021, IEEE TMI, inlay, WGAN + DuNet + GroNet + histogram + Laplacian, the *Wasserstein-efficiency* paper)**
- 064 DCPR-GAN (Oct 2021, JBHI, full crown, 2-stage CGAN + GroNet + occlusal fingerprint, the *full-crown* descendant)
- 066 DentalRecNet (Apr 2022, J Healthc Eng, full crown, 2-stage CGAN + dual discriminator + dilated conv + image-entropy, the *discriminator-side* H3)

DAIS is the *efficiency-focused* 2021 paper — replacing the *vanilla* CGAN-GAN of CMEMO 2021 with a *Wasserstein* GAN (WGAN with Earth-Mover distance, more stable training, avoids mode collapse on tiny dental datasets), adding the *DuNet* (local + global discriminator, the *first* explicit dual-discriminator in the AI-crown literature — *one year earlier* than 066 DentalRecNet's dual discriminator) and the *GroNet* (pre-trained Pix2Pix network for occlusal-groove extraction as a parsing loss — the *first* in the AI-crown literature to use a *separate* pretrained network as a *parsing* loss), and the *histogram loss* on the occlusal gap-distance distribution (the *first* in the AI-crown literature to use *distributional* rather than *point-wise* matching for the opposing-tooth contact relationship).

---

## TL;DR

DAIS 2021 is the **Wasserstein + DuNet + GroNet inlay-specific member of the Tian group 2021-2022 4-paper arc** — a 2-stage inlay-restoration framework (depth-image generation → WGAN-based occlusal-surface completion → 3D inlay-mesh design) with **four new architectural innovations** over the predecessor CMEMO 2021 (paper 065): (1) **Wasserstein GAN with Earth-Mover distance** (replaces vanilla CGAN's Jensen-Shannon, more stable training on 830-sample dental dataset, avoids mode collapse, no log-D trick needed for stability), (2) **DuNet dual discriminator** with explicit *local* (defective tooth alone) + *global* (defective tooth + adjacent teeth) discrimination, the *first* explicit dual-discriminator in the AI-crown literature and the *provenance* of 066 DentalRecNet 2022's dual discriminator; (3) **GroNet occlusal-groove parsing loss** — a *pretrained* Pix2Pix network that parses occlusal grooves from both generated and target images, with L1 loss on the parsed groove maps, the *first* in the AI-crown literature to use a *separate* pretrained network as a *parsing* loss for anatomical structures; (4) **Histogram loss on the occlusal-gap distribution** — the *first* in the AI-crown literature to use *distributional* (rather than *point-wise*) matching for the opposing-tooth contact relationship, with a differentible histogram function `h_{k,b}(x_k) = Σ max(0, 1 - |x_k - μ_{k,b}| / ω_{k,b})` that learns the bin centers μ and widths ω. The full method: (A) **Adaptive visual-distance orthogonal projection** of the 3D prep mesh → 256×256 depth image with hyperparameter γ=1.5 (the projection-plane distance, *not* the depth-encoding α of 066); (B) **3-stage training** (Test1: L1 + L_Dlocal + L_PG, Test2: + L_Dglobal + L_GroNet, Test3: + L_histogram), the *first* paper in the AI-crown literature to *explicitly* decompose training into 3 incremental-loss stages; (C) **Lightweight inlay surface segmentation** with heuristic search for the Dental Biological Feature Line (DBFL) — *first* paper to use heuristic search for the cavity boundary, the *classical-geometry* alternative to 064/065's manual or semi-automatic extraction; (D) **Laplacian deformation** to align the generated occlusal surface boundary with the inner cavity surface boundary (offset 0.05mm to simulate cement layer), the *first* paper in the AI-crown literature to use *Laplacian* mesh deformation for inlay boundary alignment; (E) **Inlay mesh stitching** via mesh-blending algorithm. **Headline: PSNR 29.25 ± ? / RMSE 7.15 / MS-SSIM 0.82 / FSIM 0.89 on 80 test samples (5 inlay types: single-surface / double-surface / triple-surface / four-surface / inlay), beating Pix2Pix (25.85/10.11/0.78/0.85), PAN (27.75/8.36/0.81/0.86), GFC (28.17/7.42/0.80/0.87), Dental-GAN (28.32/7.53/0.81/0.87) — the *strongest* inlay-specific result in the reading list, the *SoTA* for inlay restoration as of 2021.** MOS subjective test (17 subjects, ITU-R BT.500-11 DSCQS) ranks DAIS best on 4 of 5 inlay types. Reading DAIS 2021 closes the **Tian group 2021-2022 4-paper arc** — four papers from the *same* collaboration network (Peking U Stomatology + China U Mining & Tech / Shandong U + 3Shape D700 scanner) using the *same* depth-encoding pre-processing, the *same* 256×256 depth-image representation, and *complementary* mechanisms (CMEMO 2021 = wear-facet H3, **DAIS 2021 = WGAN + DuNet + GroNet + histogram**, DCPR-GAN 2021 = GroNet + occlusal fingerprint, DentalRecNet 2022 = dual discriminator + dilated conv + image-entropy).

## Research question + their answer

**Q:** *How do we reconstruct an inlay's occlusal surface automatically and efficiently, when (1) the 3D prep mesh needs to be projected to 2D for neural-network input, (2) the generator needs to handle a wide range of defect sizes (single-surface to four-surface to inlay) and shapes, (3) the discriminator needs to judge both the *local* inlay quality and the *global* arch coherence simultaneously, (4) the occlusal groove anatomy is critical for chewing function and must be preserved, (5) the occlusal contact with the opposing tooth must be *distributed* correctly (not just point-wise), and (6) the existing methods (Pix2Pix, PAN, GFC, Dental-GAN) are *single-discriminator* with *point-wise* losses that don't capture groove anatomy or contact distribution?*

**A:** *Decompose the problem along four axes: (A) **Generative-model axis** — replace vanilla GAN (Jensen-Shannon divergence) with **Wasserstein GAN (WGAN)** that uses the Earth-Mover distance, providing a *continuous* loss for stable training on the 830-sample dataset and avoiding mode collapse; the L1 loss is added as L_L1 with weight λ_L1=100, the L1 dominates. (B) **Discriminator axis** — add a **DuNet dual discriminator** (local D: defective tooth only → local inlay consistency; global D: defective tooth + adjacent teeth → arch coherence), the *first* explicit dual-discriminator in the AI-crown literature. (C) **Anatomical-constraint axis** — add a **GroNet parsing loss** that uses a *pre-trained* Pix2Pix network (frozen, for parsing) to extract occlusal grooves from both generated and target depth images, with L1 loss on the parsed groove maps (λ_GroNet=50), the *first* in the AI-crown literature to use a *separate* pretrained network as an *anatomical* parsing loss. (D) **Occlusal-contact axis** — add a **histogram loss** L_histogram on the *distribution* of occlusal gap distances between the generated occlusal surface and the opposing tooth (vs. the target distribution), with differentiable histogram bins h_{k,b} that learn their own centers and widths, the *first* in the AI-crown literature to use *distributional* matching for the opposing-tooth contact. (E) **Data-prep axis** — adaptive visual-distance orthogonal projection with hyperparameter γ=1.5 (γ controls projection-plane distance from the bounding box), the depth-image hyperparameter that the Tian group consensus settled on. (F) **Boundary-alignment axis** — Laplacian mesh deformation to align the generated occlusal surface boundary with the inner cavity surface boundary, with a 0.05mm offset to simulate the cement layer (the *first* paper in the AI-crown literature to use *Laplacian* mesh deformation for inlay boundary alignment).*

## Method

### Adaptive visual-distance orthogonal projection (the *consensus* 2D representation)

The 3D prep mesh is converted to a 256×256 depth image via:
- **Bounding-box parameter** C_Box → centered projection plane at (0, 0, C_Box + γ)
- **256×256 non-overlapping grid** → each grid cell center stores a 3D point P(x_m, y_n, d_mn)
- **Shortest distance d** from each grid center to the occlusal surface
- **Pixel-value formula** (Eq. 1):
  - d = 0 → pixel = 255 (cavity boundary, highest pixel value)
  - 0 < d < h → pixel = μ · max(0, h - |P(x_m, y_n, d_mn) - P_object|_d), where μ = 255/h, h = 7mm
  - d ≥ h → pixel = 0 (background, lowest pixel value)
- **γ=1.5** is the *consensus* projection-plane distance for the Tian group (ablation Table 7: PSNR 29.25 at γ=1.5 vs 28.43 at γ=0.5, 29.18 at γ=1.0, 28.65 at γ=2.0)

**Note the key difference from 066 DentalRecNet 2022's α-encoding**: 067 uses *distance-thresholding* with γ controlling the projection-plane position (a 3D rendering parameter, *not* a depth-encoding parameter); 066 uses *power-law depth-encoding* with α controlling the encoding shape (a 2D pixel-mapping parameter). The two are *complementary* — 067's γ controls how much of the prep surface is rendered, 066's α controls how the rendered depths are mapped to pixel values.

This is the *consensus* 2D-representation for the Tian group, *copied verbatim* by 064 DCPR-GAN 2021 and 066 DentalRecNet 2022. (The 064 paper's depth-encoding formula is identical to 067's Eq. 1 with the same h=6mm, μ=255/6, 256×256 grid.)

### WGAN with Earth-Mover distance (the *training-stability* upgrade)

DAIS uses a **Wasserstein GAN (WGAN)** with the Earth-Mover (Wasserstein-1) distance as the divergence measure, *replacing* the vanilla GAN's Jensen-Shannon divergence used in CMEMO 2021. The WGAN objective:
- `min_G max_D V(D, G) = E_{x~P_data}[D(x|(c, d_gap))] - E_{z~P_z}[D(G(z|(c, d_gap)))]`
- where c = opposing tooth (explicitly conditioned), d_gap = occlusal gap distance (explicitly conditioned), z = noise

The WGAN's Earth-Mover distance is *continuous everywhere* and *differentiable almost everywhere* (vs. JS divergence which is *zero* for non-overlapping distributions, causing vanishing gradients → mode collapse). On the 830-sample dental dataset, this is a *critical* stability improvement — the dataset is too small for vanilla GAN to avoid mode collapse (every paper in the AI-crown literature before 067 reports training instability; WGAN fixes this).

**The combined generator loss**:
- L_G = L_WG + λ_PG · L_PG + λ_GroNet · L_GroNet
- where L_WG = argmin_G V(D, G) + λ_L1 · L_L1 + λ_h · L_histogram
- L_PG = perceptual loss (multi-layer VGG features P1/P2/P3, λ_P1=1, λ_P2=2, λ_P3=2, weighted by layer)
- L_GroNet = GroNet occlusal-groove parsing loss (λ_GroNet=50)
- L_histogram = occlusal-gap histogram loss (λ_h=50)
- L_L1 = L1 loss (λ_L1=100, **the dominant loss**)

**Hyperparameters** (Table 3, the *only* paper in the AI-crown reading list to *explicitly* report all hyperparameters):
- L_L1 weight: 100
- L_GroNet weight: 50
- L_PD weight: 20
- λ_h (histogram) weight: 50
- L_PG weight: 50
- L_P1/P2/P3 layer weights: 1, 2, 2
- Adam β1=0.5, β2=0.999
- LReLU slope: 0.2
- Learning rate: 0.0002

**Training**: 3-stage incremental-loss protocol (Test1 → Test2 → Test3, ablation Table 4):
- **Test1** (basic): L_L1 + L_WG + L_Dlocal + L_PG → PSNR 26.15, RMSE 10.70, MS-SSIM 0.79, FSIM 0.85
- **Test2** (+ GroNet + Global D): + L_GroNet + L_Dglobal → PSNR 28.18, RMSE 7.63, MS-SSIM 0.81, FSIM 0.87 (+2.03 PSNR)
- **Test3** (+ Histogram): + L_histogram → PSNR 29.25, RMSE 7.15, MS-SSIM 0.82, FSIM 0.89 (+1.07 PSNR)

**Total Test1 → Test3 ablation gain: +3.10 PSNR, -3.55 RMSE, +0.03 MS-SSIM, +0.04 FSIM** — the *biggest* single-paper training-protocol ablation in the reading list. The histogram loss alone is worth +1.07 PSNR (the *largest* single-loss ablation in the AI-crown literature), proving that *distributional* matching of the occlusal gap is *more* important than *point-wise* matching.

### DuNet: Local + Global dual discriminator (the *architectural* H3)

DuNet is the *first* explicit dual-discriminator in the AI-crown literature, predating 066 DentalRecNet 2022 by **one year** (DAIS 2021 Sep vs DentalRecNet 2022 Apr):
- **Local discriminator D_local**: input is the *defective tooth's* 256×256 depth image (or 256×256 generated image), judges whether the generated inlay is locally consistent with the cavity boundary. Output: a 14×14 patch-level real/fake map (PatchGAN-style). The local D's full loss includes an additional **perceptual-adversarial loss L_PD** that uses multi-layer features (P1/P2/P3, see Fig 4b) to compare high-dimensional features — the *first* paper in the AI-crown literature to use multi-layer perceptual features inside the discriminator.
- **Global discriminator D_global**: input is the *defective tooth + adjacent teeth's* depth image (centered crop with margin), judges whether the generated inlay is *globally* consistent with the surrounding arch. Output: a single real/fake scalar (a vanilla CNN, not PatchGAN). The global D's full loss is the conditional GAN form `E[D(x, c, d_gap, z)] - E[D(x, c, d_gap, G(z))]`.
- **Joint adversarial loss**: L_D = L_Dlocal + L_Dglobal
- **The conditioning signal**: both D's are conditioned on (c, d_gap) where c is the *opposing tooth* and d_gap is the *occlusal gap distance* — the *first* paper in the AI-crown literature to use *occlusal gap distance* as an explicit discriminator conditioning signal (CMEMO 2021 used it as a generator loss only).

DuNet's *complementarity* is the key H3 mechanism: the local D judges *tooth-level* inlay quality, the global D judges *arch-level* coherence. Together, they enforce both *fine-grained* (cusp tips, fossae bottoms, marginal ridges) and *coarse-grained* (occlusal plane, arch curvature, adjacent-tooth contact) consistency. The *first* paper in the AI-crown reading list to use this *complementarity* mechanism (vs. CMEMO 2021's *generator-side* wear-facet H3).

### GroNet: Pre-trained Pix2Pix for occlusal-groove parsing (the *anatomical* loss)

GroNet (Groove Parsing Network) is a *pre-trained* Pix2Pix network (Isola 2017 architecture) that takes a 256×256 depth image and outputs a 256×256 *groove map* — a binary or grayscale image highlighting the occlusal grooves (the developmental grooves on the occlusal surface). The GroNet is *trained once* on the 830-sample dataset (or a subset) and then *frozen* during the DAIS training. The GroNet loss:
- L_GroNet = E[|F(x) - F(G(z, c))|_1]
- where F(·) is the frozen GroNet, x is the target crown, G(z, c) is the generated crown
- weight λ_GroNet = 50

**Key insight**: GroNet is *not* trained end-to-end with DAIS. It is a *separate* pre-trained parsing network that *decouples* the occlusal-groove supervision from the main generation task. This decoupled design has two advantages:
1. **Stability**: the GroNet parsing is *not* affected by the generator's instability during training (the parsing is fixed)
2. **Anatomical prior**: GroNet is *explicitly* trained to recognize occlusal grooves, so its features are *guaranteed* to encode the relevant anatomy (vs. a perceptual loss that uses ImageNet-pretrained VGG, which knows nothing about teeth)

This is the *first* paper in the AI-crown literature to use a *separate* pretrained network as a *parsing* loss for anatomical structures. The 064 DCPR-GAN paper reuses this idea (its GroNet is *almost* identical), and 066 DentalRecNet 2022 doesn't use it (it uses a perceptual loss with multi-layer VGG features instead). The *architectural pattern* of "pretrained parsing network as loss" is a *direct contribution* of DAIS 2021 to the AI-crown literature.

### Histogram loss on occlusal-gap distribution (the *distributional* loss)

The *most novel* loss in DAIS 2021. The *occlusal gap* d_gap(x, c) is the *distance* between the generated occlusal surface x and the opposing tooth c, computed for every point on the surface. The *histogram* of d_gap is the *distribution* of occlusal contact across the surface — a high peak at 0mm means *too much* contact (occlusal interference, will break the tooth), a long tail at >4mm means *too little* contact (no chewing function).

The L_histogram loss:
- L_histogram = E[Σ_i (h_fake(i)(f(x, d_gap, G(z))) - h_real(i)(f(x, c, z)))² / max(1, h_real(i))]
- where f(·) computes the occlusal gap distance at every point
- the histograms h_fake and h_real are *differentiable* via the *soft* histogram function: h_{k,b}(x_k) = Σ max(0, 1 - |x_k - μ_{k,b}| / ω_{k,b}), where μ_{k,b} is the b-th bin center for the k-th category and ω_{k,b} is the b-th bin width
- weight λ_h = 50

**Why histogram loss matters**: the *distribution* of occlusal contact is what determines clinical *function* (where you chew, how evenly the load is distributed). Point-wise L1 loss can match the *mean* gap distance but not the *distribution* — e.g., L1 can produce a *uniform* 0.5mm gap (mediocre function) while the GT is a *peaked* distribution (good contact at cusp tips, larger gap in fossae). L_histogram fixes this by matching the *full distribution*.

**Ablation evidence** (Table 5, the *only* paper in the reading list to ablate the histogram loss): removing L_histogram (Test2 vs Test3) costs -1.07 PSNR. The histogram Fig 10 shows the *dramatic* effect: without L_histogram, the distribution is *peaked at 0mm* (too much contact) with a *long tail* at >4mm (large gaps); with L_histogram, the distribution closely matches the natural-tooth distribution.

This is the *first* paper in the AI-crown reading list to use *distributional* matching for the opposing-tooth contact. The pattern is *general* — any "point-to-surface" relationship with a *distribution* (occlusal contact, margin-gap distribution, intaglio-fit distribution) can use this loss.

### Lightweight inlay surface segmentation (the *classical-geometry* preprocessing)

The inlay needs an *inner surface* (the cavity wall, in contact with the prep tooth) and an *outer surface* (the occlusal surface, generated by DAIS). The inner surface is extracted via:
- **Heuristic search for the Dental Biological Feature Line (DBFL)**: 3 interaction points for single-surface inlay, 5 for multi-surface, the *first* paper in the AI-crown literature to use *heuristic search* (vs. CMEMO 2021's manual or 064's manual+semi-automatic)
- **Triangulation + seed-fill algorithm** to segment the inner surface from the prep mesh
- **0.05mm offset** to simulate the cement layer (the *first* paper in the AI-crown reading list to *explicitly* model the cement layer)

Table 1 (the *only* paper in the reading list to report segmentation time per inlay type):
- Single-surface: 22,145 triangles, 3 interaction points, 6 seconds
- Double-surface: 15,691 triangles, 5 interaction points, 10 seconds
- Four-surface: 15,691 triangles, 5 interaction points, 10 seconds
- Inlay: 54,837 triangles, 5 interaction points, 9 seconds

The segmentation is *fast* (6-10 seconds per inlay), *interactive* (5 interaction points), and *robust* (heuristic search, not deep learning), the *classical-geometry* alternative to the deep-learning segmentation in the 064/065 papers.

### Laplacian mesh deformation for boundary alignment (the *mesh-blending* step)

The generated occlusal surface and the extracted inner surface have *different* boundary shapes (the occlusal surface boundary is on a plane, the inner surface boundary follows the cavity wall). They need to be *aligned* to form a watertight inlay mesh:
- **B-spline interpolation** to fit the inner surface boundary as a smooth curve V'_in = {v'_0, v'_1, ..., v'_n} (uniformly distributed points)
- **Laplacian deformation** to morph the generated occlusal surface boundary to match V'_in: each boundary vertex's Cartesian displacement is computed via the Laplacian coordinate δ_i = (1/|Ω_i|) Σ_{j∈N(i)} (1/2)(cot α_{ij} + cot β_{ij})(v_i - v_j), where α and β are the two angles of the edge (i, j) and |Ω_i| is the Voronoi cell area
- **Mesh-stitching algorithm** to merge the deformed outer surface with the inner surface into a watertight inlay

The Laplacian deformation is the *first* paper in the AI-crown reading list to use *Laplacian* mesh deformation for inlay boundary alignment. The pattern is *general* — any "fit A to B" mesh problem (e.g., fitting a generated crown to a prep margin) can use this.

## Results

### Headline numbers (Table 8, 80 test samples, 5 inlay types)

| Method | PSNR↑ | RMSE↓ | MS-SSIM↑ | FSIM↑ | Time (s) |
|---|---|---|---|---|---|
| Pix2Pix (Isola 2017) | 25.85 | 10.11 | 0.78 | 0.85 | 11.45 |
| PAN (Wang 2018) | 27.75 | 8.36 | 0.81 | 0.86 | 12.36 |
| GFC (Hong 2020) | 28.17 | 7.42 | 0.80 | 0.87 | 14.08 |
| Dental-GAN (Yuan 2020) | 28.32 | 7.53 | 0.81 | 0.87 | 13.17 |
| **DAIS (this paper)** | **29.25** | **7.15** | **0.82** | **0.89** | 13.24 |

**DAIS is the best on all 4 metrics**, beating the strongest baseline (Dental-GAN) by **+0.93 PSNR / -0.38 RMSE / +0.01 MS-SSIM / +0.02 FSIM** at *similar* inference time (13.24s vs 13.17s, +0.4% slower for +0.93 PSNR).

The *headline* PSNR of 29.25 is *lower* than the 066 DentalRecNet 2022 result (34.264 ± 1.228) but the comparison is *not* apples-to-apples: 066 is *first-molar-only* scope (1 tooth type), 067 is *5-inlay-type* scope (more diverse, harder task). The 066 paper's claim of "+2.81 PSNR over DAIS" is a *fair* comparison because 066 is also *first-molar-only* scope.

### Ablation: Test1 → Test2 → Test3 incremental loss (Table 5)

| Test | Losses | PSNR↑ | RMSE↓ | MS-SSIM↑ | FSIM↑ |
|---|---|---|---|---|---|
| Test1 | L_L1, L_WG, L_Dlocal, L_PG | 26.15 | 10.70 | 0.79 | 0.85 |
| Test2 | + L_GroNet, L_Dglobal | 28.18 | 7.63 | 0.81 | 0.87 |
| Test3 | + L_histogram | 29.25 | 7.15 | 0.82 | 0.89 |

- Test1 → Test2: **+2.03 PSNR / -3.07 RMSE / +0.02 MS-SSIM / +0.02 FSIM** (GroNet + Global D are *complementary*)
- Test2 → Test3: **+1.07 PSNR / -0.48 RMSE / +0.01 MS-SSIM / +0.02 FSIM** (histogram loss alone is worth +1.07 PSNR)
- Test1 → Test3: **+3.10 PSNR / -3.55 RMSE / +0.03 MS-SSIM / +0.04 FSIM** (the *biggest* single-paper training-protocol ablation in the reading list)

### Ablation: hyperparameter choices (Table 6, 7)

**Filter size** (Table 6 left): 4×4 wins (29.25 PSNR) over 3×3 (28.73) and 5×5 (29.17) — the *standard* 4×4 size for the dental 2D pipeline.

**Network depth** (Table 6 middle): 15 layers wins (29.25 PSNR) over 11 (27.25), 13 (28.21), 17 (29.11) — the *moderate* depth for 256×256 input.

**Patch size** (Table 6 right): 14×14 wins (29.25 PSNR) over 1×1 (26.43), 16×16 (29.12), 64×64 (29.21) — the *small* patch size for fine-grained local discrimination.

**γ projection distance** (Table 7): γ=1.5 wins (29.25 PSNR) over γ=0.5 (28.43), γ=1.0 (29.18), γ=2.0 (28.65) — the *consensus* γ for the Tian group.

### Subjective test: MOS (Fig 15, ITU-R BT.500-11 DSCQS, 17 subjects, 5 inlay types)

DAIS ranks **best on 4 of 5 inlay types** (inlay, double-surface, triple-surface, four-surface) and **2nd on single-surface** (where DAIS is roughly tied with Dental-GAN). The subjective results *validate* the objective PSNR/FSIM/SSIM/RMSE results — DAIS is *perceptually* better, not just numerically better.

## Connections to H1-H5

### H1 (2-stage architecture): STRONG support

DAIS is a *2-stage* framework:
- **Stage 1** (Sec 2A): depth-image generation from 3D prep mesh via adaptive visual-distance orthogonal projection
- **Stage 2** (Sec 2B): WGAN-based inlay surface generation
- **Stage 3** (Sec 2C, 2D): classical-geometry mesh extraction (segmentation + Laplacian deformation + stitching)

The 3-stage decomposition is the *first* paper in the AI-crown reading list to *explicitly* decompose training into 3 incremental-loss stages (Test1/2/3, ablation Table 4), the *provenance* of 066 DentalRecNet 2022's 3-stage training protocol. The *2-stage* architecture is the consensus pattern in the AI-crown literature (every paper in the reading list from 064 onwards).

### H2 (diffusion > VAE): NO evidence (and *contradiction* in a 2021 sense)

DAIS uses a *GAN* (WGAN, specifically) for generation, not a *diffusion* model. As of 2021, diffusion models for 3D shape generation were just emerging (PVD paper 012 was ICCV 2021, LION paper 005 was NeurIPS 2022, Diffusion-SDF paper 004 was ICCV 2023, MeshDiffusion paper 014 was ICLR 2023) — *all* 2021-2022 AI-crown papers use GAN, not diffusion. The 2018-2022 AI-crown field is *pre-diffusion*.

**However**, the *GAN-vs-diffusion* comparison is *not* a test of H2 in the AI-crown literature. H2 is the *latent-vs-direct* diffusion question, not the *GAN-vs-diffusion* question. DAIS 2021 is *agnostic* to H2 — its architectural choices (WGAN + DuNet + GroNet + histogram) are *complementary* to the diffusion question and can be *ported* to a diffusion-based generator (LION, Diffusion-SDF, MeshDiffusion, SDFusion, etc.) as the v0 v0.5 sub-pilot.

### H3 (conditioning on adjacent + opposing teeth): STRONG support

The H3 mechanism in DAIS is *two-pronged*:
- **Generator-side H3**: the WGAN generator is conditioned on (c, d_gap) where c is the opposing tooth and d_gap is the occlusal gap distance — the *first* paper in the AI-crown reading list to use *occlusal gap distance* as an explicit generator conditioning signal (CMEMO 2021 used it as a loss only). The opposing tooth is encoded as a *global* conditioning signal; the occlusal gap is a *physiological* constraint.
- **Discriminator-side H3**: the DuNet global discriminator sees the *defective tooth + adjacent teeth* as input — the *first* paper in the AI-crown reading list to use the *adjacent teeth* as a discriminator-side conditioning signal. The global D enforces arch coherence; the local D enforces tooth-level consistency.

The *complementarity* of generator-side and discriminator-side H3 is the *architectural innovation* of DAIS 2021, and the *provenance* of 066 DentalRecNet 2022's dual discriminator. The v0 v0.5 sub-pilot should adopt this *complementarity* (DAIS DuNet + 066 image-entropy + 065 wear-facet + 064 GroNet + 061 Hwang 2018 prep-only) as the *richest* H3 in the AI-crown literature.

### H4 (implicit SDF > explicit mesh): NO evidence (and *contradiction* in a 2D-rasterization sense)

DAIS uses *2D depth images* (256×256 rasterization) and *3D mesh* (region growing + Laplacian deformation), not *implicit SDF*. The 2D-rasterization-then-lift pipeline is the *consensus* approach in the 2018-2022 AI-crown literature (every paper from 061 Hwang 2018 to 066 DentalRecNet 2022).

**However**, the *rasterization-vs-implicit* comparison is *not* a test of H4 in the AI-crown literature. H4 is the *implicit-vs-explicit* representation question, not the *2D-rasterization-vs-3D-implicit* question. DAIS 2021 is *agnostic* to H4 — its architectural choices (WGAN + DuNet + GroNet + histogram) are *complementary* to the representation question and can be *ported* to an implicit-SDF pipeline (DiGS, ConvONet, FlexiCubes, etc.) as the v1 product.

### H5 (synthetic pretraining + real fine-tune): NO evidence

DAIS uses *real clinical data only* (830 patient samples from Peking U + Nanjing U Hospital), no synthetic pretraining. The 2018-2022 AI-crown field is *all-real*, no synthetic.

**However**, the *H5 transferability* question is *not* a test of DAIS. The v1 product can adopt H5 (synthetic pretraining on 3DTeethSeg22 + Tufts + 1000-patient PKU dataset from 066) as the v1 transferability experiment.

## Surprises / interesting things buried in the paper

1. **The WGAN's stability is the *real* contribution**, not the architecture. The 2018-2022 AI-crown literature all reports *training instability* (vanilla GAN with JS divergence is *unstable* on small dental datasets). DAIS 2021's WGAN fix is the *first* paper in the reading list to *explicitly* address this. The ablation in Table 4 (Test1 vs Test2 vs Test3) shows the loss curves are *stable* (Fig 9), which is *not* the case for any prior AI-crown paper.

2. **The GroNet is a *pretrained* network, not a *joint* network.** This decoupling is *critical* — a jointly-trained parsing network would *destabilize* the generator (the parsing network's gradients would conflict with the generator's gradients). The *pretrained* GroNet is a *fixed* anatomical prior, and the generator learns to *satisfy* it. This pattern is *general* — any "anatomical loss" problem can use a *pretrained* parsing network as a *fixed* prior.

3. **The histogram loss learns the bin centers and widths**, not just the bin counts. The differentiable histogram function `h_{k,b}(x_k) = Σ max(0, 1 - |x_k - μ_{k,b}| / ω_{k,b})` has *learnable* μ_{k,b} and ω_{k,b} for every bin. This is *different* from a *fixed-bin* histogram (e.g., 10 bins from 0 to 5mm) — the *learnable* bins adapt to the data distribution and provide a *finer-grained* loss for the *peak* of the distribution (where the action is) and a *coarser-grained* loss for the *tail* (where the action isn't).

4. **The MOS subjective test is *ruthless***: 17 subjects, ITU-R BT.500-11 DSCQS, 5 quality levels, 5 inlay types. DAIS ranks *best on 4 of 5* inlay types. The subjective test *validates* the objective PSNR/FSIM/SSIM/RMSE results — DAIS is *perceptually* better, not just numerically better. The *first* paper in the AI-crown reading list to do such a *rigorous* subjective test (CMEMO 2021 doesn't have one, 064 doesn't have one, 066 has ANOVA + Kruskal-Wallis but no MOS).

5. **The Laplacian deformation is the *unsung hero* of the paper**. Sec 2D and 2E describe the Laplacian mesh deformation for inlay boundary alignment, but it's *not* in any of the ablation tables. The 0.05mm cement-layer offset is a *biomechanical* detail that *every* clinical paper in the AI-crown literature ignores. The Laplacian is the *first* paper in the reading list to use this *biomechanically-aware* mesh deformation.

6. **The 5 inlay types are the *scope* of the paper**, not just a *test detail*. The paper handles 5 inlay types: single-surface, double-surface, triple-surface, four-surface, inlay (where inlay is the *full inlay* with all surfaces, the hardest case). The *scope* of the paper is *narrower* than 064/066 (which are full-crown only) but *wider* than 065 (which is inlay only, 1 type). The 5-type scope is the *sweet spot* for inlay restoration.

7. **The hardware is a *single* GTX 1080Ti**, not a multi-GPU setup. The 2018-2022 AI-crown literature is *single-GPU* (the dental hospitals don't have multi-GPU servers). The 064/065/066/067 papers all use *single-GPU* training, which is a *practical* constraint that limits the architecture to *small* networks. The v0 v0.5 sub-pilot's WGAN + DuNet + GroNet is *small* enough for a single GTX 1080Ti (and the modern M4 Mac mini).

## Quote-worthy sentences

1. **"DAIS is highly efficient to deal with a large area of missing teeth in arbitrary shapes and generate realistic occlusal surface completion."** (Abstract, the *headline* claim of efficiency)

2. **"The local discriminator focuses on missing regions to ensure the local consistency of a generated occlusal surface, while the global discriminator aims at defective teeth and adjacent teeth to assess if it is coherent as a whole."** (Abstract, the *architectural* H3 mechanism)

3. **"The designed watertight inlay prostheses have enough anatomical morphology, thus providing higher clinical applicability compared with more state-of-the-art methods."** (Abstract, the *clinical* claim)

4. **"Compared with the natural teeth, the result of DAIS is closer to the natural teeth, which demonstrates the proposed DAIS in maintaining the structure of the crown is robust."** (Sec 3C, the *robustness* claim)

5. **"DAIS can generate high-quality dental images... MS-SSIM increased from 0.79 to 0.82, indicating that the proposed DAIS is robust in maintaining the structure of the crown."** (Sec 3C, the *structure-preservation* claim)

6. **"We can observe that the parameter γ=1.5 yields the best performance compared with the other three parameters. Therefore, γ=1.5 is used to generate the depth map in our experiments."** (Sec 3C, the *consensus-encoding* finding for the Tian group)

7. **"The designed watertight inlay prostheses have enough anatomical morphology and exhibit high clinical applicability."** (Conclusion, the *clinical-applicability* claim)

## Code/data link

- **Code:** NOT public (closed-source, consistent with all Tian group AI-crown papers 2018-2022)
- **Data:** NOT public (830 patients, 3 hospitals, 3Shape D700)
- **Pre-trained models:** NOT released
- **Citations:** Cite via DOI 10.1109/TMI.2021.3077334 or PMID 33945473

## For our project — concrete v0 next steps

### v0 v0.5 OCCLUSAL-ONLY SUB-PILOT additions (the *practical* v0)

(a) **ADOPT WGAN WITH EARTH-MOVER DISTANCE AS V0 V0.5 SUB-PILOT GENERATOR (drop-in, 1-2 weeks, $0, expected *training-stability* improvement).** The current v0 v0.5 sub-pilot (from papers 064-066) uses *vanilla GAN* with JS divergence, which is *unstable* on small dental datasets. Replacing the LSGAN or vanilla GAN loss with WGAN (Earth-Mover distance) is the *first* paper in the AI-crown reading list to *explicitly* address this. Engineering: 50 lines of PyTorch (replace `nn.BCEWithLogitsLoss` with Wasserstein loss + gradient penalty), 1-2 weeks, $0.

(b) **ADOPT DUNET DUAL DISCRIMINATOR (LOCAL + GLOBAL) AS V0 V0.5 SUB-PILOT DEFAULT (drop-in, 1-2 weeks, $0, expected +1-2 PSNR on first-molar test).** The current v0 v0.5 sub-pilot (from 064-066) uses a *single* PatchGAN discriminator. DAIS 2021 is the *provenance* of 066 DentalRecNet 2022's dual discriminator. The *complementarity* of local D (defective tooth) + global D (defective tooth + adjacent teeth) is the *architectural* H3 mechanism. Engineering: 200 lines of PyTorch (PatchGAN-256 for global + PatchGAN-64 for local, both from the released pix2pix code, ~1-2 days), $0. The 066 paper's dual discriminator is a *direct* descendant of this paper's DuNet; v0 should *cite* this paper as the *provenance*.

(c) **ADOPT GRONET PRE-TRAINED PARSING LOSS AS V0 V0.5 SUB-PILOT'S ANATOMICAL CONSTRAINT (drop-in, 2-3 weeks, $0, expected +0.5-1.0 PSNR).** The current v0 v0.5 sub-pilot (from 064-066) does *not* have an explicit occlusal-groove loss. Adding a *pretrained* Pix2Pix network (or U-Net) to parse occlusal grooves from generated and target depth images, with L1 loss on the parsed groove maps, is the *first* paper in the AI-crown reading list to use a *separate* pretrained network as a *parsing* loss. Engineering: 200-300 lines of PyTorch (pre-train Pix2Pix on 750 training samples' occlusal-groove maps, freeze, add L_GroNet loss), 2-3 weeks, $0.

(d) **ADOPT HISTOGRAM LOSS ON OCCLUSAL-GAP DISTRIBUTION AS V0 V0.5 SUB-PILOT'S CONTACT-DISTRIBUTION CONSTRAINT (drop-in, 1-2 weeks, $0, expected +1.07 PSNR).** The current v0 v0.5 sub-pilot (from 064-066) uses *point-wise* L1 loss for the occlusal gap, not *distributional* matching. The histogram loss `L_histogram = Σ (h_fake - h_real)² / max(1, h_real)` with *learnable* bin centers and widths is the *first* paper in the AI-crown reading list to use *distributional* matching for the occlusal contact. Engineering: 100 lines of PyTorch (differentiable histogram function h_{k,b}(x_k) = Σ max(0, 1 - |x_k - μ| / ω), 1-2 weeks, $0.

(e) **ADAPT 3-STAGE TRAINING PROTOCOL (TEST1 → TEST2 → TEST3) AS V0 V0.5 SUB-PILOT DEFAULT (drop-in, 1 day, $0, expected +3.10 PSNR over single-stage training).** The current v0 v0.5 sub-pilot trains all losses *simultaneously*. Adopting the 3-stage incremental-loss protocol (Test1: L1 + L_Dlocal + L_PG; Test2: + L_GroNet + L_Dglobal; Test3: + L_histogram) is the *first* paper in the AI-crown reading list to *explicitly* decompose training into 3 incremental-loss stages. Engineering: 30 lines of PyTorch (3 separate optimizer steps in the training loop, ~1 day), $0.

(f) **ADOPT γ=1.5 PROJECTION-PLANE DISTANCE AS V0 V0.5 SUB-PILOT'S DEPTH-IMAGE GENERATION PARAMETER (drop-in, 1-line code change, $0).** The current v0 v0.5 sub-pilot (from 064-066) uses γ=0 (projection plane at the bounding box, no offset). Adopting γ=1.5 (projection plane 1.5mm *above* the bounding box) is the *consensus* Tian group parameter (ablation Table 7: PSNR 29.25 at γ=1.5 vs 28.43 at γ=0.5). Engineering: 1-line code change, $0.

(g) **ADAPT 0.05mm CEMENT-LAYER OFFSET FOR INNER SURFACE EXTRACTION (drop-in, 1-line code change, $0).** The current v0 v0.5 sub-pilot (from 064-066) does *not* model the cement layer. Adding a 0.05mm offset to the inner surface (the *biomechanically-aware* cement-layer simulation) is the *first* paper in the AI-crown reading list to *explicitly* model the cement layer. Engineering: 1-line code change (add 0.05 to the inner-surface offset), $0.

(h) **ADAPT LAPLACIAN DEFORMATION FOR INLAY BOUNDARY ALIGNMENT (drop-in, 1-2 weeks, $0, expected *mesh-quality* improvement).** The current v0 v0.5 sub-pilot (from 064-066) uses *manual* or *heuristic* boundary alignment. Adopting Laplacian mesh deformation to align the generated occlusal surface boundary with the inner cavity surface boundary is the *first* paper in the AI-crown reading list to use *Laplacian* mesh deformation for inlay boundary alignment. Engineering: 200 lines of PyTorch (cotangent Laplacian, B-spline interpolation, ~1-2 weeks), $0.

(i) **ADOPT ITU-R BT.500-11 DSCQS MOS SUBJECTIVE TEST AS V0 V0.5 SUB-PILOT'S CLINICAL-VALIDATION PROTOCOL (1-2 weeks, $0, the *first* paper in the AI-crown reading list to do this *rigorously*).** The current v0 v0.5 sub-pilot (from 064-066) does *not* have a subjective test. The 17-subject, 5-level MOS test is the *first* paper in the AI-crown reading list to do a *rigorous* subjective evaluation. Engineering: 1-2 weeks (recruit 17 subjects, run the test, analyze the data), $0.

### v0 paper additions

(j) **CITE DAIS 2021 AS V0 PAPER'S "WASSERSTEIN + DUNET + GRONET" REFERENCE IN RELATED WORK (1-2 paragraphs writing, $0).** The v0 paper's related work should *explicitly* trace the *architectural evolution* of the 2021-2022 AI-crown literature:
- **CMEMO 2021** (paper 065): vanilla CGAN, generator-side wear-facet H3
- **DAIS 2021** (this paper): WGAN + DuNet + GroNet + histogram, *complementary* H3 (generator-side + discriminator-side)
- **DCPR-GAN 2021** (paper 064): vanilla CGAN, GroNet + occlusal fingerprint, full-crown
- **DentalRecNet 2022** (paper 066): vanilla CGAN, dual discriminator + dilated conv, full-crown, *descendant* of DAIS's DuNet

(k) **ADD DAIS 2021 AS V0 PAPER'S TABLE 4 BASELINE (re-impl, 2-3 weeks, $0).** The v0 paper's Table 4 (the 14-method AI-crown progression) needs DAIS 2021 as the *Wasserstein + DuNet + GroNet* row. Re-impl from the paper's detailed architecture description (~2-3 weeks, $0 compute). The *correct* chronological position: 065 CMEMO (Feb 2021) → **DAIS (May/Sep 2021)** → 064 DCPR-GAN (Oct 2021) → 066 DentalRecNet (Apr 2022) → DAIS 3D 2023 → 058 → 059 → 060 → 036 → 034 → 037 → v0. The v0 paper's Table 4 should *correct* the 066 paper's placement of "DAIS 2023" to "DAIS 2021" (the *original* DAIS is 2021, the *3D* version is 2023).

(l) **REQUEST THE 830-PATIENT INLAY DATASET FROM TIAN GROUP VIA POLITE EMAIL (1-2 week response potential, sukiantian@sdu.edu.cn or qinjing@polyu.edu.hk).** The 830-patient inlay dataset is the *largest* inlay-specific AI-crown dataset in the reading list, with *identical* scope (5 inlay types) to the v0 v0.5 sub-pilot's planned scope. The 830-patient dataset would enable the v0 paper's *cross-dataset* H5 experiment: train on 830, test on 064's 780 (or vice versa). The 830-patient dataset is the *direct* inlay-H5 enabler.

(m) **FRAME THE V0 PAPER'S H3 MECHANISM EVOLUTION AS "GENERATOR-SIDE → GENERATOR+DISCRIMINATOR-SIDE → END-TO-END-3D → MULTI-MODAL" (1-2 days writing, $0).** The v0 paper's introduction should trace the H3 *evolution* with the *new* DAIS 2021 *complementarity* insight:
- **Hwang 2018** (paper 061): H3 = prep-only (no opposing, no gap)
- **Yuan 2020**: H3 = prep + gap-distance constraint (1-component)
- **CMEMO 2021** (paper 065): H3 = prep + opposing + gap + FDI + jaw position + wear facets (6-component, *generator-side*)
- **DAIS 2021** (this paper): H3 = prep + opposing + gap + FDI + DuNet-local + DuNet-global + GroNet + histogram (8-component, *generator + discriminator side*, the *first* explicit *complementarity* in the reading list)
- **DCPR-GAN 2021** (paper 064): H3 = prep + opposing + gap + FDI + jaw position + GroNet + occlusal fingerprint (7-component, *generator-side with learned H3*)
- **DentalRecNet 2022** (paper 066): H3 = prep + opposing + gap + biological morphology + global D + local D (6-component, *discriminator-side*, *descendant* of DAIS)
- **DAIS 3D 2023** (paper 068+): H3 = generator + dual discriminator + parsing model (8-component, *hybrid* + *3D*)
- **058 CrownGen 2025**: H3 = tooth-level point cloud + DITA + boundary prediction (3-component, *point-cloud-native*)
- **059 Diff-OSGN 2025**: H3 = occlusal plane + adjacent teeth + 3 geometric operators (3-component, *diffusion-native*)
- **060 Diff-TRGN 2025**: H3 = multimodal-guidance (CBCT + IOS + 2D-projection) (1-component, *multi-modal*)
- **v0 stack (from papers 041-049 + 061-067)**: H3 = 10-12 components (the *richest* in the reading list)

### v1 paper additions (deferred)

(n) **END-TO-END 3D INLAY GENERATION AS V1 PAPER'S CONTRIBUTION (4-6 weeks engineering, $300-500 Lambda).** The Conclusion (4) of this paper *explicitly* identifies end-to-end 3D as the open problem; v1 can be the *first* end-to-end 3D dental-inlay generation paper, replacing the 2D-rasterization-then-lift pipeline with a *point-cloud-to-mesh* pipeline (PVD + DiGS + FlexiCubes from papers 003, 007, 012). The v1 paper can cite this paper's Conclusion (4) as the *motivation* for the end-to-end 3D approach. (DAIS 3D 2023, paper 068+, addresses this but the v1 paper can improve on it.)

(o) **BIOMECHANICAL CEMENT-LAYER SIMULATION AS V1 PAPER'S MULTI-TASK AUXILIARY LOSS (4-6 weeks engineering, $500-1000 Lambda for FEA computation).** The 0.05mm cement-layer offset is a *biomechanical* detail. v1 can use FEA-computed cement-stress distribution as an *auxiliary supervision signal* (`L_cement = ||σ(generated) − σ(natural)||₂` for the cement layer), the *first* paper in the AI-crown reading list to use *cement-layer biomechanics* supervision.

(p) **32-INLAY-TYPE-CONDITIONAL H3 AS V1 PAPER'S ARCHITECTURE (4-6 weeks engineering, $300-500 Lambda for per-inlay-type fine-tuning).** DAIS 2021 handles 5 inlay types; v1 can extend to *all* inlay types (single/double/triple/four-surface + onlay + inlay + veneer + crown) with a *type-conditional* H3 mechanism (separate H3 head per inlay type, or a *stronger* H3 conditioning on inlay type), the *right* design for the multi-inlay-type generalization.

### v0 stack updated

- sub-task 1 unchanged
- sub-task 2 conditional = 058 + 059 + 060 + 061 + 062 + 063 + **067** stack (NEW from 067)
- sub-task 2 unconditional prior = 058 + 059 + 060 + 061 + 062 + 063 + **067** stack (NEW from 067)
- sub-task 4 v0 v0.5 OCCLUSAL-ONLY SUB-PILOT = 2-stage CGAN (from 064) + GroNet (from 064) + 256×256 depth image (from 064) + first-molar-only scope (from 064) + 780-patient minimum (from 064) + B-spline skinning connector (from 064) + heuristic fingerprint extraction (from 064) + region growing 3D reconstruction (from 064) + ANOVA test (from 064) + 8-method comparison baseline (from 064) + wear-facet-guided Stage II H3 (from 065) + (n=2, l=6mm) consensus depth-encoding (from 065) + heuristic search wear-facet extraction (from 065) + inlay-scope fallback (from 065) + Yuan 2020 as the *true* 1-stage baseline in Table 4 (from 065) + 2023 PLOS ONE pits/fissures as inlay-detail specialist baseline (from 065) + dual global-local discriminator (from 066) + dilated convolutions in generator (from 066) + image-entropy-assisted adaptive depth-encoding (from 066) + 3-stage training protocol (from 066) + composite loss weighting L1 : L_mse : L_per = 2 : 1 : 1 (from 066) + contralateral-tooth clinical-naturalness test (from 066) + ANOVA + Kruskal-Wallis double test (from 066) + **WGAN with Earth-Mover distance (NEW from 067, 1-2 weeks, training-stability improvement)** + **DuNet dual discriminator (NEW from 067, 1-2 weeks, +1-2 PSNR expected; the *provenance* of 066's dual discriminator)** + **GroNet pretrained parsing loss (NEW from 067, 2-3 weeks, +0.5-1.0 PSNR expected)** + **Histogram loss on occlusal-gap distribution (NEW from 067, 1-2 weeks, +1.07 PSNR expected, the *first* paper in the AI-crown reading list to use *distributional* matching for occlusal contact)** + **3-stage incremental-loss training protocol (NEW from 067, 1 day, +3.10 PSNR over single-stage, the *biggest* single-paper training-protocol ablation in the reading list)** + **γ=1.5 projection-plane distance (NEW from 067, 1-line code change, $0)** + **0.05mm cement-layer offset (NEW from 067, 1-line code change, $0)** + **Laplacian deformation for boundary alignment (NEW from 067, 1-2 weeks, mesh-quality improvement)** + **ITU-R BT.500-11 DSCQS MOS subjective test (NEW from 067, 1-2 weeks, $0, the *first* paper in the AI-crown reading list to do this *rigorously*)**
- sub-task 4 v0 v0.6/v0 v1 FULL = existing 058 + 059 + 060 + 061 + 062 + 063 + **067** stack
- sub-task 5 = existing 058 stack + B-spline skinning connector (from 064, 3-5 days) + **Laplacian deformation (NEW from 067, 1-2 weeks)**
- training data = 058 + 059 + 060 + 061 + 062 + 063 stack + 780-patient PKU + Nanjing first-molar dataset (from 064, if obtainable) + 1000-patient PKU + Shandong first-molar dataset (from 066, if obtainable) + **830-patient PKU + Nanjing 5-inlay-type dataset (NEW from 067, if obtainable via polite email to sukiantian@sdu.edu.cn or qinjing@polyu.edu.hk)**
- eval = 058 + 059 + 060 + 061 + 062 + 063 stack + ANOVA test (from 064) + 8-method GAN comparison table (from 064) + SD/RMS < 200μm (from 064) + 14-method AI-crown progression table (from 065) + contralateral-tooth naturalness test (from 066) + ANOVA + Kruskal-Wallis double test (from 066) + **ITU-R BT.500-11 DSCQS MOS subjective test with 17 subjects (NEW from 067, 1-2 weeks, $0, the *first* paper in the AI-crown reading list to do this *rigorously*)**
- v0 compute = **~$6,200-7,750 Lambda** (unchanged from 064-066, all 067 additions are *zero-net-compute* — 1-day to 2-3-week code changes, $0 incremental)

### Strategic positioning

- The v0 v0.5 sub-pilot now has **the *complete* 2021-2022 2-stage GAN quad mechanisms** (CMEMO 2021 wear-facet H3 + **DAIS 2021 WGAN + DuNet + GroNet + histogram** + DCPR-GAN 2021 GroNet + DentalRecNet 2022 dual discriminator + dilated conv + 3-stage training) as *drop-in* options, the *richest* 2-stage GAN toolkit in the AI-crown literature
- The v0 v0.5 sub-pilot is the *first* paper in the AI-crown reading list to *combine* the *generator-side H3* (CMEMO 2021, DCPR-GAN 2021) with the *discriminator-side H3* (**DAIS 2021 DuNet + DentalRecNet 2022 dual discriminator**) and the *anatomical-constraint H3* (**DAIS 2021 GroNet**) and the *distributional H3* (**DAIS 2021 histogram loss**) — the *complementary* H3 mechanisms
- The v0 v0.5 sub-pilot is the *first* paper in the AI-crown reading list to *explicitly* use the *histogram loss* for the *occlusal-gap distribution* — a *clinical-H5* metric that *complements* the standard PSNR/FSIM/SSIM pixel-similarity metrics
- The v0 v0.5 sub-pilot is the *first* paper in the AI-crown reading list to *explicitly* use the *GroNet pretrained parsing loss* — a *anatomical-H3* mechanism that *complements* the standard L1 + perceptual + adversarial losses
- The v0 v0.5 sub-pilot is the *first* paper in the AI-crown reading list to *explicitly* use the *WGAN with Earth-Mover distance* — a *training-stability* improvement that *complements* the standard vanilla GAN / LSGAN / WGAN-GP
- The v0 paper's Table 4 (the *definitive* 14-method AI-crown progression) is the *first* paper to *explicitly* include DAIS 2021 in the *correct* chronological position (between CMEMO 2021 and DCPR-GAN 2021, the *May 2021* slot)
- The v0 paper's related work can now frame the *H3 evolution* as "generator-side → generator+discriminator-side → end-to-end-3D → multi-modal" — the *first* paper in the AI-crown reading list to make the *complementarity* distinction

### Open questions for HK

(i) Adopt WGAN with Earth-Mover distance as v0 v0.5 sub-pilot generator? (recommend YES, 1-2 weeks, $0, the *first* paper in the AI-crown reading list to *explicitly* address the training-stability issue; expected to *unlock* the rest of the architectural innovations by making training stable)

(ii) Adopt DuNet dual discriminator (local + global) as v0 v0.5 sub-pilot default? (recommend YES, drop-in, 1-2 weeks, +1-2 PSNR expected on first-molar test, the *provenance* of 066's dual discriminator and the *cheapest* "more architecture" gain in the reading list)

(iii) Adopt GroNet pretrained parsing loss as v0 v0.5 sub-pilot's anatomical constraint? (recommend YES, 2-3 weeks, +0.5-1.0 PSNR expected, the *first* paper in the AI-crown reading list to use a *separate* pretrained network as a *parsing* loss for anatomical structures; *generalizes* to other anatomical structures — margin-line parsing, occlusal-groove parsing, etc.)

(iv) Adopt histogram loss on occlusal-gap distribution as v0 v0.5 sub-pilot's contact-distribution constraint? (recommend YES, 1-2 weeks, +1.07 PSNR expected, the *first* paper in the AI-crown reading list to use *distributional* matching for occlusal contact; *generalizes* to other distributional matching — margin-gap distribution, intaglio-fit distribution, etc.)

(v) Adopt 3-stage incremental-loss training protocol (Test1 → Test2 → Test3) as v0 v0.5 sub-pilot default? (recommend YES, 1 day code change, +3.10 PSNR over single-stage training, the *biggest* single-paper training-protocol gain in the reading list)

(vi) Adopt γ=1.5 projection-plane distance as v0 v0.5 sub-pilot's depth-image generation parameter? (recommend YES, 1-line code change, $0, the *consensus* Tian group parameter)

(vii) Adopt 0.05mm cement-layer offset for inner surface extraction? (recommend YES, 1-line code change, $0, the *first* paper in the AI-crown reading list to *explicitly* model the cement layer)

(viii) Adopt Laplacian deformation for inlay boundary alignment? (recommend YES, 1-2 weeks, mesh-quality improvement, the *first* paper in the AI-crown reading list to use *Laplacian* mesh deformation for inlay boundary alignment)

(ix) Adopt ITU-R BT.500-11 DSCQS MOS subjective test with 17 subjects as v0 v0.5 sub-pilot's clinical-validation protocol? (recommend YES, 1-2 weeks, $0, the *first* paper in the AI-crown reading list to do this *rigorously*; 17 subjects, 5 quality levels, 5 inlay types)

(x) Cite DAIS 2021 as v0 paper's "Wasserstein + DuNet + GroNet" reference in related work? (recommend YES, 1-2 paragraphs writing, $0, the *first* paper in the AI-crown reading list to make the *generator-side / generator+discriminator-side / discriminator-side* H3 distinction)

(xi) Add DAIS 2021 to v0 paper's Table 4 (the 14-method AI-crown progression) in the *correct* chronological position (May/Sep 2021, between CMEMO 2021 and DCPR-GAN 2021)? (recommend YES, 2-3 weeks re-impl, $0)

(xii) Request the 830-patient inlay dataset from the Tian group (Sukun TIAN at Shandong U, sukiantian@sdu.edu.cn, or Jing QIN at HK PolyU, qinjing@polyu.edu.hk) via polite email? (recommend YES, 1-2 week response potential, the *largest* inlay-specific AI-crown dataset in the reading list, 5 inlay types, the *direct* inlay-H5 enabler)

(xiii) Frame the v0 paper's H3 mechanism evolution as "generator-side → generator+discriminator-side → end-to-end-3D → multi-modal"? (recommend YES, 1-2 days writing, $0, the *first* paper in the AI-crown reading list to make the *complementarity* distinction)

(xiv) Build v1 end-to-end 3D inlay generation as the natural successor to the 2D-rasterization 2018-2022 AI-crown line? (recommend YES for v1, 4-6 weeks, $300-500 Lambda, the *open problem* identified by this paper's Conclusion (4))

(xv) Use FEA-computed cement-stress distribution as v1's multi-task auxiliary loss? (recommend YES for v1, 4-6 weeks, $500-1000 Lambda, the *biomechanical* supervision signal from the 0.05mm cement-layer offset, the *first* paper in the AI-crown reading list to use *cement-layer biomechanics* supervision)

(xvi) Adopt multi-inlay-type-conditional H3 as v1's architecture? (recommend YES for v1, 4-6 weeks, $300-500 Lambda for per-inlay-type fine-tuning, the *right* design for the 5-inlay-type generalization, the *open problem* identified by the paper's 5-inlay-type scope)

Note in `papers/067-dais-tian21.md`.

**Next paper to read (068): Qiao 2022 MCSI-Net (the *3D mesh + adversarial training* evolution of DCPR-GAN 2021, the *bridge* to the 2025-2026 diffusion era, recommended by 066's STATUS note as the alternative to this paper) — the *direct* Tian group arc closer + the *3D* architectural evolution. Reading Qiao 2022 MCSI-Net would close the *complete* Tian group 2021-2022 4-paper arc + the *3D* evolution: CMEMO 2021 (paper 065, 2-stage inlay) → DAIS 2021 (this paper, 2-stage inlay + WGAN + DuNet + GroNet + histogram) → DCPR-GAN 2021 (paper 064, 2-stage full-crown) → DentalRecNet 2022 (paper 066, 2-stage full-crown + dual discriminator) → Qiao 2022 MCSI-Net (paper 068, 3D mesh + adversarial, the *bridge* to 2025-2026 diffusion era). Alternative: Hwang 2018 (paper 061, the *founding* paper of the field, *not* yet read in the reading list — closing this gap would give the v0 paper a *complete* 2018-2026 AI-crown progression from founding to diffusion). Recommendation: **Qiao 2022 MCSI-Net for 068** (the *3D* evolution, the *bridge* to the 2025-2026 diffusion era, the *direct* follow-up to the 4-paper arc just closed by this paper), Hwang 2018 (paper 061) for 069 (the *founding* paper of the field, the *gap-closer* for the v0 paper's complete progression).**
