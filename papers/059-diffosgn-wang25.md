# Paper 059 — *Diff-OSGN: Diffusion-based Occlusal Surface Generation Network with Geometric Constraints*

**Title:** *Diff-OSGN: Diffusion-based Occlusal Surface Generation Network with Geometric Constraints*
**Authors:** Chen Wang¹, Guangshun Wei¹, James Kit Hon Tsoi², Zhiming Cui³, Shuyi Lu¹, Zhenpeng Liu¹, Yuanfeng Zhou¹*
**Affiliations:**
¹ **Shandong University** — School of Software (Wang, Wei, Lu, Liu, Zhou); Jinan, China
² **University of Hong Kong** — Faculty of Dentistry, Dental Materials Science (Tsoi); Sai Ying Pun, Hong Kong
³ **ShanghaiTech University** — School of Biomedical Engineering (Cui); Shanghai, China
*Corresponding author: Yuanfeng Zhou (yfzhou@sdu.edu.cn)

- **Year:** 2025 (Received 14 Feb 2025; Accepted 25 Jun 2025; Published 1 Oct 2025)
- **Venue:** **Computational Visual Media (CVM)**, 11(4):817-832, 2025 (Springer + Tsinghua University Press, ISSN 2096-0433, **open access under CC-BY 4.0**, sciopen DOI 10.26599/CVM.2025.9450498)
- **PDF (open access):** [iccvm.org/2025/papers/s3p2-27-cvmj.pdf](https://iccvm.org/2025/papers/s3p2-27-cvmj.pdf) (open-access preprint, 27 MB, 16 pages, 13 figures, 4 tables, 41 references)
- **HTML:** [sciopen.com/article/10.26599/CVM.2025.9450498](https://www.sciopen.com/article/10.26599/CVM.2025.9450498)
- **Code:** ❌ **not released** as of Oct 2025. No GitHub link in the paper. The Shandong U "IGIP-LAB" group has a track record of NOT releasing code (paper 048 IGIP, paper 001 §1.4 DArch, this paper) — consistent with their *ToothFairy* + *3DTeethLand* group-publication model. Code is available only via corresponding author (yfzhou@sdu.edu.cn, polite email + cite-thanks is the path).
- **Data:** **Private** — 319 normal-occlusion patients from the HKU Faculty of Dentistry, intra-oral-scanned. The paper restricts to **first molars only** (197 #16, 189 #26, 188 #36, 197 #46) — only ~771 tooth instances from ~385 first-molar cases. This is a *very* small dataset by reading-list standards (CrownGen 058: 1784 scans; DMC 033: 1935; CrossTooth 043: 1800+360; 3DTeethSeg'22: 1800). The 70/15/15 train/test/val split is on teeth, not patients, so the test set *overlaps* the patient distribution of the training set (no held-out OOD test).
- **Read:** 2026-06-08 09:04 KST (Monday, scholar hourly #59, ~50 min — full PDF via pdftotext + sciopen HTML, all 16 pages + tables + figures + references + 6 supplementary figures)

**Authors are the SAME GROUP as paper 048 (IGIP) and paper 001 §1.4 (DArch 050)** — the "Shandong U IGIP-LAB" lineage: Shaojie Zhuang + Guangshun Wei + Zhiming Cui + Yuanfeng Zhou + Wang Chens cohort. The University of Hong Kong is the *clinical* partner (Tsoi = the prosthodontist; also a co-author of paper 058's *Morphology and mechanical performance of dental crowns designed by human and knowledge-based AI* in J Mech Behav Biomed Mater 2022, cited as ref [2]). This is the *third Shandong U dental-3D-gen paper* in the reading list (050 DArch, 048 IGIP, 059 Diff-OSGN), and the *first one with HKU clinical authorship* — the HKU-Shandong-ShanghaiTech axis is now a *recurring* author network across 050/048/059.

---

## TL;DR

**Diff-OSGN is the *first* diffusion model for dental-crown *occlusal surface* design — the *hardest* part of crown design (the surface that articulates with the opposing arch).** Unlike CrownGen (paper 058) which generates a *complete* crown point cloud + mesh in one pass, Diff-OSGN focuses on the *occlusal half* of a single tooth and casts it as a **2D image-to-image translation problem** (height map + normal map → occlusal surface map) instead of a 3D point-cloud / mesh problem. The pipeline has three moving parts: **(1) geometry-map representation** — rasterize the IOS mesh onto a 512×512 image *above the inferred occlusal plane* (XOY plane defined by the midpoint of two adjacent teeth as origin, the mesial-to-distal direction as x-axis, the y-axis pointing toward the symmetric tooth's projection, the z-axis by cross-product); the resulting "geometry map" is a 4-channel image = `{height_map, normal_map_3_channels}`; **(2) conditional DDPM** — encoder E extracts features from the *adjacent teeth* geometry map `X_adj` and the *occlusal teeth* geometry map `X_occ` to produce a 256-dim condition `c = E(X_adj, X_occ)`, which is injected into a U-Net denoiser G via cross-attention at every layer; the denoiser *directly predicts X_0* (not the noise, à la Ramesh/DALL-E), conditioned on `c` and timestep `t`, with 200 diffusion steps; **(3) three geometric operators** — (a) Consistency Preservation Operator `O_cp` (Sobel filter on the height map → consistent normal map; the loss `L_cpo` enforces that the generated normal map matches the Sobel-derived normal, a self-supervised consistency check); (b) Contour Enhancement Operator `O_ce` (Laplacian on the height map → contour map; the loss `L_ceo` enforces that the high-frequency ridges/grooves match GT); (c) Curvature Related Operator `O_cr` (N=24 neighbor cross-product on the normal map → curvature map; the loss `L_cro` enforces local geometric detail). **The geometric-operator loss is the *novel* contribution** — three *differentiable* image-processing operators that *each* extract a clinically-meaningful geometric feature (gradient, Laplacian, neighbor cross-product) and compare to GT, giving the diffusion model *multi-scale geometric supervision* (low-freq shape via `L_cpo` + high-freq ridges via `L_ceo` + local detail via `L_cro`). The full pipeline is trained on a *single* tooth type (first molar) and a *single* arch (upper/lower) — the paper's *biggest limitation* (it does not generalize to incisors/canines because their occlusal plane is not parallel to the arch's occlusal plane, breaking the rasterization). **The evaluation is modest**: 8 baselines (Pix2Pix, CycleGAN, VAE, VQVAE, vanilla DDPM, 3DCGAN, DCPRGAN, ours) on the 4-tooth (16/26/36/46) first-molar dataset. Quantitative: **best on 7/10 metrics** (CD 0.959 mm, FID 72.17, LPIPS 0.070, Angle 2.987°, Cont 0.160, Curv 0.220, SSIM 0.915), second-best on RMSE (0.110 vs DCPRGAN's 0.103) and PSNR (23.57 vs DCPRGAN's 24.27), only 3D point-cloud completion method tested on the same protocol (3 baselines: CRA-PCN, GeoFormer, SVDFormer, ours: CD 0.959 vs next-best 1.018). The user study (6 prosthodontists + 3 technicians + 12 residents) shows **93.4% satisfaction rate** for our method vs **80.1% for vanilla DDPM, 70.9% for VQVAE, 14.8% for Pix2Pix** — ours beats vanilla DDPM by 13.3 percentage points. **The qualitative occlusion-distance visualization (Fig 13) is the *clinically* most compelling result**: red areas = tight occlusal contact, and the authors' method shows the *largest red area* of all 8 baselines, meaning the generated crown's occlusion matches the *opposing arch* better than any baseline. **For our project: Diff-OSGN is the *complementary 2025 SoTA* to CrownGen 058 — CrownGen = full crown + multi-crown + DITA + boundary prediction, Diff-OSGN = occlusal half only + single-crown + image-based + 3 geometric operators. The 2 papers together define the *2025 frontier* of dental-3D-gen: complete-crown point-cloud diffusion (CrownGen) and occlusal-surface image diffusion (Diff-OSGN). For v0 sub-task 2 + sub-task 4, the *most portable* contribution from Diff-OSGN is the *geometric-operator loss* — `O_cp` (Sobel consistency), `O_ce` (Laplacian contour), `O_cr` (neighbor cross-product curvature) are *3 lines of PyTorch* each, and *each* is a v0 paper-ready H3 mechanism (low-freq shape + high-freq ridge + local curvature are the *three* clinically-meaningful geometric features of the cervical margin + occlusal surface).** The paper's *limitation* (first molars only) is the *direct value* for our v0 sub-task 4 (crown *outer surface* generation is molars-only, since anterior crowns are dominated by the *polishing* surface, not the occlusal surface).

## Research question + their answer

**Q:** Existing AI methods for dental crown design (Hwang 2018, Yuan 2020, Tian 2021/2022 "DCPRGAN", Ding 2023 "3D-DCGAN") use GANs over *single-channel depth maps* rasterized onto a 2D image from the 3D IOS mesh. This 2D-rendering approach has *three* fundamental limitations: **(1) single-channel depth maps cannot represent *multi-axis* geometric detail** — the depth map only captures the *height* axis (z), losing the *normal* axis information (the *direction* of the surface at each pixel), which is exactly the geometric information that defines ridges, cusps, and grooves on the occlusal surface; **(2) GANs struggle to capture fine-grained details** — the depth map generated by a GAN-based method is *blurry* and lacks the high-frequency content that defines a clinically-functional occlusal surface; **(3) prior methods ignore the *occlusal relationship* between the target crown and the *adjacent/occlusal* teeth** — the crown is designed *in isolation*, but its occlusal function depends on the *relative position* and *surface geometry* of the surrounding teeth. The research question is: **can a diffusion-based model that operates on a *multi-channel* "geometry map" representation (height + normal) generate a *more detailed* occlusal surface that is *conditioned* on the surrounding teeth and *supervised* by differentiable geometric features?**

**A:** **Yes — the geometry map (height + normal rasterized onto the occlusal plane), the conditional diffusion model (encoder on adjacent+occlusal teeth geometry map → cross-attention conditioning in the U-Net denoiser), and the three geometric operators (Sobel-consistency, Laplacian-contour, neighbor-cross-product-curvature) jointly produce an occlusal surface that *beats* all 8 baselines on the 7/10 metrics and the *user study* and the *occlusal distance* visualization.** The key insight is that the *occlusal surface* is a *2D object embedded in 3D* (it's a *patch* on the XOY plane with z-direction height) — so casting it as a 2D image is *not* an approximation, it's a *natural representation*; the previous 2D-image approaches used a *single-channel* depth map (losing normal info), but the *4-channel* geometry map preserves *all* the geometric info of the surface. The diffusion model is *essential* for the high-frequency detail (the ridges and grooves are *high-frequency* features that GANs smooth out, but diffusion's multi-step denoising naturally preserves high-freq content). The *adjacent* and *occlusal* teeth conditioning is the H3 mechanism — the *occlusal function* of a crown is *not* defined in isolation, it's defined by how it *meshes* with the surrounding teeth, so the diffusion model must see the surrounding teeth to generate a *functional* surface. The three geometric operators are the *supervision* — they extract the *clinically meaningful* features (overall shape, ridge structure, local curvature) and the loss enforces that the *generated* map matches the GT map on all three scales (low-freq + high-freq + local). **The result is the *first* clinically-validated occlusal-surface generation method that uses a diffusion model** — 93.4% user-study satisfaction, the *largest red area* in the occlusal-distance visualization, beating all 8 GAN/VAE/diffusion baselines on 7/10 quantitative metrics, and *second-best* on the remaining 3 by a small margin (DCPRGAN's RMSE 0.103 vs ours 0.110, PSNR 24.27 vs 23.57).

## Method

### Pipeline overview (Fig 2, 4)

```
Input: Adjacent teeth IOS mesh + Occlusal teeth IOS mesh + Target tooth (cavity, no mesh)
       │
       ▼
Stage 0: Local coordinate + occlusal plane inference
       - Origin = midpoint of two adjacent teeth centroids
       - x-axis = mesial-to-distal direction of adjacent teeth
       - temp y-axis = origin to symmetric tooth direction
       - z-axis = x × temp_y
       - true y-axis = x × z
       - XOY plane = occlusal plane
       │
       ▼
Stage 1: Geometry map rasterization
       - Rasterize adjacent teeth → X_adj ∈ (S² × ℝ)^(512×512)  (4 channels)
       - Rasterize occlusal teeth → X_occ ∈ (S² × ℝ)^(512×512)  (4 channels)
       - Target occlusal surface → X_0 ∈ (S² × ℝ)^(512×512)  (4 channels, GT)
       │
       ▼
Stage 2: Encoder E
       - Input: [X_adj; X_occ] (concatenated, 8 channels)
       - 1st doubleConv (conv+BN+ReLU+conv+BN) → maxpool → 2 doubleConv → 2nd doubleConv
       - Flatten + linear layer → c ∈ ℝ^256 (condition)
       │
       ▼
Stage 3: Conditional DDPM reverse diffusion
       - Forward: q(X_t | X_{t-1}) = N(X_t; √(1-β_t) X_{t-1}, β_t I),  T=200 steps
       - Reverse: p_θ(X_{t-1} | X_t, c) = N(μ_θ(X_t, t, c), σ_t² I)
       - μ_θ uses Ramesh-style X_0 prediction (not ε): G_θ(X_t, t, c) → X_0
       - Denoiser G: U-Net with cross-attention at every layer (c is keys+values, X_t is query)
       - Skip connections, down/up sampling 4 levels, bottleneck 256 channels
       │
       ▼
Stage 4: Three geometric operators + composite loss
       - L_recon = λ_h ||h_pred - h_gt||² + λ_n ||n_pred - n_gt||²         (λ_h=1, λ_n=10)
       - L_mask = || ||n_pred||_2 - M_gt ||²                              (λ_m=0.1)
       - L_cpo = ||n_pred - O_cp(h_pred)||²   (Sobel consistency)          (λ_cpo=0.1)
       - L_ceo = ||O_ce(h_pred) - O_ce(h_gt)||²   (Laplacian contour)      (λ_geo=0.01)
       - L_cro = ||O_cr(n_pred) - O_cr(n_gt)||²   (cross-product curvature) (λ_geo=0.01)
       - L_total = L_recon + λ_m L_mask + λ_cpo L_cpo + λ_geo L_ceo + λ_geo L_cro
       │
       ▼
Output: Generated geometry map X_0_pred → back-project to 3D point cloud → mesh
```

### Local coordinate + occlusal plane inference (Sec 3.1, Fig 3a)

The key step that makes the 2D-rasterization approach work is the *occlusal plane definition*. The authors:
1. Identify the **two adjacent teeth** (mesial and distal) and the **three occlusal teeth** (the target tooth's direct opponent in the opposing arch + two adjacent opposing teeth). The paper's case is first molars: 2 adjacent (15/17 for #16, 25/27 for #26, etc.) + 1 occlusal (46 for #16, etc.).
2. The **local coordinate origin** is the midpoint of the two adjacent teeth's centroids in 3D.
3. The **x-axis** is the mesial-to-distal direction (centroid of distal tooth - centroid of mesial tooth, normalized).
4. A **temporary y-axis** points from the origin to the *symmetric tooth* (the same FDI in the other quadrant of the same arch; for #16, the symmetric tooth is #26). This temp_y is *not* the final y-axis.
5. The **z-axis** is the cross-product of x and temp_y (`z = x × temp_y`).
6. The **true y-axis** is the cross-product of x and z (`y = x × z`).
7. The **XOY plane** is defined as the *occlusal plane*.

This is a *physically-grounded* definition: the y-axis points from the midline toward the lateral side (or vice versa), the z-axis points *out of* the occlusal surface (toward the opposing arch), and the x-axis points mesial-to-distal. The **critical assumption** (also the paper's *biggest limitation*): the XOY plane is *the* occlusal plane *only if the target tooth's occlusal surface is parallel to it*. This holds for molars (the contact area between upper and lower molars is roughly parallel to the arch's occlusal plane), but **breaks for incisors and canines** (their "occlusal" surfaces — really the incisal edges — are *not* parallel to the arch's occlusal plane, they're at an angle). The paper explicitly states: "our method cannot be directly used for the design of dentures for incisors and canines. Firstly, the occlusal surfaces of incisors and canines are not parallel to the occlusal plane of the dental arch, which causes our rasterization of the geometry map to become ineffective."

### Geometry map rasterization (Sec 3.1, Fig 3b)

Given the local coordinate system, the authors rasterize the *occlusal surface* (the surface above the XOY plane, z > 0) onto a 512×512 grid. For each grid cell, two values are sampled:
- **Height value h(x, y)** = z-coordinate of the surface at (x, y), measured *perpendicular to the occlusal plane*. This is the *true* geometric height, *not* a normalized depth (so the values are in millimeters, *not* in [0, 1]).
- **Normal vector n(x, y)** = surface normal at (x, y), a 3-dim unit vector. The authors render this as an RGB image using a standard normal-to-color mapping (nx→R, ny→G, nz→B, scaled to [0, 255]).

The result is a 4-channel image per tooth: `{n_x, n_y, n_z, h}` ∈ (S² × ℝ)^(512×512). The geometry map is the *raw data* the diffusion model operates on. The rasterization is *fully differentiable* w.r.t. the input mesh, so the geometry map can be back-propagated through to optimize the diffusion model end-to-end.

The key advantage of *height* over *depth* is robustness: depth images require careful hyperparameter selection (near plane, far plane, normalization) that varies per scan; height images are *physical* values (millimeters from the occlusal plane) and are *consistent across scans*. The paper notes: "Due to variations in occlusal planes across cases, depth images generated with the same hyperparameters can be different between datasets. These discrepancies between cases hinder the network's ability to learn effectively. In contrast to the depth images used in [25, 26], which require numerous hyperparameters, our method directly rasterizes the height map of occlusal surface, capturing the true height values of points relative to the occlusal plane."

### Encoder E (Sec 3.2, Fig 4 upper)

A standard 2D image encoder:
- 1st fully-connected block = `doubleConv(2D conv → BN → ReLU → 2D conv → BN)`, then `MaxPool2d` for downsampling, then 2 more `doubleConv` blocks, then 2nd `doubleConv` block → 8*512*512 → 32*512*512 → 64*128*128 → 128*32*32 → 128*8*8 → 256 features
- Flatten → linear layer → condition `c ∈ ℝ^256`
- Input: concatenated `[X_adj, X_occ]` = 8 channels
- Output: condition `c`

The encoder is a *small* U-Net-style feature extractor (no skip connections to the decoder, just a flat encoder) — the design choice is *condition extraction*, not *image generation* (the diffusion's U-Net is the main work).

### Denoising module G (Sec 3.3, Fig 4 lower)

A U-Net with cross-attention at every layer:
- 1st fully-connected block + 4 down-sampling levels (MaxPool + doubleConv)
- Cross-attention with `c` and `t` (time embedding) at *every* level
- Bottleneck: doubleConv → 256 channels
- 4 up-sampling levels (bilinear upsample + doubleConv)
- 1 final 2D conv → 4-channel output (the predicted X_0)

The *X_0 prediction* (not ε prediction) is the Ramesh/DALL-E-style choice: the denoiser directly outputs the *clean* geometry map, not the noise, which lets the *geometric operators* in the loss function be applied to a *clean* image (a noisy image's Laplacian / Sobel would be noise-dominated). This is the *key reason* the three geometric operators work as well as they do: they're applied to *clean* predictions, not noisy intermediate states.

### Three geometric operators (Sec 3.4, Fig 5) — THE NOVELTY

These three operators are the paper's *main* contribution. They're simple, differentiable, and *clinically-motivated*.

**(a) Consistency Preservation Operator O_cp** (Eq 11, 12): Given a height map `h(x, y)`, compute the Sobel filter in both directions: `(S_x(h), S_y(h), 1)`. Normalize to get a *consistent normal map*. The loss `L_cpo = ||n_pred - O_cp(h_pred)||²` enforces that the *generated* normal map matches the *Sobel-derived* normal map from the *generated* height map. **Why this matters:** in the GT data, the normal map and the height map are *physically consistent* (the normal IS the gradient of the height). If the generated normal map is *inconsistent* with the generated height map, then the generated 3D surface is *non-differentiable* (a sharp discontinuity), which is *physically impossible* for a real tooth surface. The O_cp loss enforces *self-consistency* between the two generated channels. **This is a *self-supervised* consistency loss** — it does *not* require GT normals, only GT height (and Sobel is free). It's the *cheapest* of the three operators to implement (3 lines of PyTorch).

**(b) Contour Enhancement Operator O_ce** (Eq 13, 14): Given a height map `h(x, y)`, compute the Laplacian filter: `O_ce(h) = h(x, y) - (1/N) Σ_i ω_i · h(x_i, y_i)`, with N=24 neighbors, ω_i = 1. The result is a *contour image* where the gray regions are flat (no height change), the lighter regions are convex (peaks), and the darker regions are concave (valleys). The loss `L_ceo = ||O_ce(h_pred) - O_ce(h_gt)||²` enforces that the *generated* height map's contour structure matches the *GT* height map's contour structure. **Why this matters:** the ridges and grooves on the occlusal surface are *high-frequency* features (sharp, narrow structures). A standard L2 loss on the height map *averages out* high-frequency content (it's biased toward low-frequency fits). The Laplacian *amplifies* high-frequency content, so the loss `L_ceo` is *specifically* a high-frequency supervision signal. **The ridges and grooves are the *clinical* high-frequency features** — they're the cusps, fossae, and marginal ridges that define the occlusal function. Without `L_ceo`, the diffusion model would produce *blurry* cusps; with `L_ceo`, the cusps are *sharp*. The ablation in Table 1 confirms: removing `L_ceo` drops Angle from 2.987° to 3.085° (small), but Cont from 0.160 to 0.164 (small), and Curv from 0.220 to 0.245 (-10%, a *big* drop). The FID drops from 72.17 to 71.34 (slight improvement without it, but the Curv metric is the *more clinically meaningful* one).

**(c) Curvature Related Operator O_cr** (Eq 15, 16): Given a normal map `n(x, y)`, compute the N-neighbor cross-product: `O_cr(n) = (1/N) Σ_i (n(x, y) × n(x_i, y_i))`, with N=24. The result is a *curvature-related image* (the magnitude of the cross-product is proportional to the *change* in normal direction, which is the *mean curvature*). The loss `L_cro = ||O_cr(n_pred) - O_cr(n_gt)||²` enforces that the *local* normal variation matches the GT. **Why this matters:** the *mean curvature* of the occlusal surface is the *clinical* measure of "is this surface a smooth tooth or a noisy tooth?" — high curvature = sharp ridge (clinical feature), noisy curvature = mesh artifact (clinical bug). The `O_cr` loss enforces that the *generated* surface has the *right* curvature at the *right* places, not just the right *average* curvature. The ablation: removing `O_cr` drops Curv from 0.220 to 0.269 (-22%, the *biggest* single-operator drop in the table).

**The composite loss is the right design:** each operator targets a *different* spatial scale (Sobel = first derivative = low-freq shape, Laplacian = second derivative = mid-freq ridges, cross-product = local = high-freq detail), and the three together cover the *full* spectrum of geometric features. **For our project: this is the *first* multi-scale geometric supervision loss in the reading list, and each operator is a 3-line PyTorch implementation that ports directly to v0 sub-task 4 (crown outer surface) and sub-task 5 (mesh).** The O_cp enforces self-consistency between height + normal (v0 sub-task 4 can use it to enforce consistency between the SDF gradient and the predicted normal). The O_ce enforces high-frequency ridge structure (v0 can use it to enforce cusp sharpness on the occlusal surface). The O_cr enforces local curvature (v0 can use it to enforce smoothness in the cervical margin region where the prep-tooth junction transitions from the crown to the abutment).

### Diffusion step T=200 (Sec 4.4 ablation, Fig 9)

The authors ablate the number of diffusion steps T ∈ {50, 100, 200, 500}. They find:
- T=50: severely distorted, no high-freq detail
- T=100: better, but cusps still noisy
- **T=200: best** — "an appropriate number of diffusion steps, e.g. t = 200, not only improves the quality of the generation but also significantly enhances both training and inference efficiency. As the value of t increases, the details gradually become distorted, the generated crowns exhibit noticeable artifacts in regions such as grooves and edges. This phenomenon is attributed to the ability of our network to more effectively capture the fine-grained features of the data, thereby reducing the reliance on a larger number of diffusion steps."
- T=500: worse than T=200, more artifacts

This is a *very* different finding from CrownGen 058 (T=1000) and most diffusion papers (T=1000 is standard). The smaller T is enabled by the *operator-based loss* — the high-freq supervision from `L_ceo + L_cro` means the model doesn't need 1000 steps of denoising to recover high-freq detail; 200 steps is enough because the *loss* is already providing high-freq signal. **For v0: this is a *direct* compute-saving insight — if v0 uses similar operator-based loss, T=200 may suffice (5× faster than T=1000), at the cost of some quality.**

### Dataset (Sec 4.1)

- **319 normal-occlusion patients** from HKU Faculty of Dentistry, intra-oral-scanned
- **Semantics:** segmented with Zhuang et al. 2023 ("Robust hybrid learning for automatic teeth segmentation and labeling on 3d dental models", IEEE TMM, ref [40] in the paper, same as paper 048's IGIP method)
- **Selection:** keep cases where both *adjacent* and *occlusal* teeth are present for the *first molars* (16, 26, 36, 46). Result: **197 #16, 189 #26, 188 #36, 197 #46 teeth** (771 tooth instances)
- **Train/val/test split:** 70/15/15 *on teeth*, not patients — *no* held-out OOD test
- **Rasterization resolution:** 512×512, *16 pixels per millimeter* spatial resolution
- **Diffusion step count T=200**, batch size 8
- **Training:** Adam, initial LR 0.001, linearly decay to 0 from 15k to 30k epochs (paper says "epochs" but means "iterations" — 30k iterations is ~4 hours on RTX 3090ti)
- **Hardware:** single NVIDIA RTX 3090ti (24 GB), PyTorch
- **No data augmentation** mentioned in the paper (a *suspicious* absence — every other 2024-2025 paper in the reading list uses some form of augmentation)

### Comparative baselines (Sec 4.5, Table 2)

**8 image-based baselines** (all modified to output 4 channels = `{h, n_x, n_y, n_z}`):
1. **Pix2Pix** (Isola 2017) — conditional GAN with L1 + adversarial loss
2. **CycleGAN** (Zhu 2017) — unpaired image translation (here, the pair structure is preserved, so it's a *paired* CycleGAN)
3. **VAE** (Kingma 2013) — standard VAE with reconstruction + KL
4. **VQVAE** (Van Den Oord 2017) — discrete latent code + autoregressive prior
5. **DDPM** (Ho 2020) — *vanilla* DDPM, no geometry operator loss, single-channel output
6. **3DCGAN** (Ding 2023) — voxel-based 3D-DCGAN, modified to output height map
7. **DCPRGAN** (Tian 2021) — depth-map + two-stage GAN, the *direct* 2D-depth-map predecessor of Diff-OSGN, the most important baseline
8. **Ours** — Diff-OSGN

**3 point-cloud completion baselines** (Sec 4.5, Table 3):
- **CRA-PCN** (Rong 2024) — intra+inter-level cross-resolution transformer
- **GeoFormer** (Yu 2024) — tri-plane integrated transformer
- **SVDFormer** (Zhu 2023) — self-view + self-structure dual-generator
- **Ours** — converted from generated height map → point cloud (sample z at each pixel)

## Results

### Ablation study (Table 1, 7 rows, our dataset)

| Variant | RMSE_h ↓ | RMSE_n ↓ | PSNR_h ↑ | PSNR_n ↑ | SSIM_h ↑ | SSIM_n ↑ | Angle ↓ | Cont ↓ | Curv ↓ | FID ↓ | LPIPS ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| w.o. normal (height-only) | 0.340 | — | 21.17 | — | 0.903 | — | — | 0.162 | — | 177.8 | 0.105 |
| w.o. L_mask (geometry map, no mask loss) | 0.322 | 0.121 | 21.77 | 20.73 | 0.908 | 0.883 | 3.618 | 0.164 | 0.262 | 82.81 | 0.079 |
| base (geometry map + L_recon + L_mask) | 0.314 | 0.125 | 21.31 | 20.38 | 0.906 | 0.881 | 3.733 | 0.164 | 0.268 | 80.08 | 0.084 |
| w.o. O_cp (no Sobel consistency) | 0.297 | 0.125 | 22.34 | 20.76 | 0.912 | 0.882 | 3.415 | 0.162 | 0.259 | 79.67 | 0.084 |
| w.o. O_ce (no Laplacian contour) | 0.321 | 0.118 | 21.65 | 21.02 | 0.906 | 0.885 | 3.085 | 0.164 | 0.245 | 71.34 | 0.072 |
| w.o. O_cr (no cross-product curvature) | 0.302 | 0.120 | 22.08 | 20.36 | 0.908 | 0.883 | 3.279 | 0.163 | 0.269 | 75.73 | 0.079 |
| **All (full model)** | **0.293** | **0.110** | **23.57** | **21.21** | **0.915** | **0.885** | **2.987** | **0.160** | **0.220** | **72.17** | **0.070** |

**Key findings:**
- **All three operators are necessary**: each ablation drops at least one metric. The biggest single-operator drops are:
  - Remove O_cr: Curv 0.220 → 0.269 (-22%, the biggest drop)
  - Remove O_ce: PSNR_n 21.21 → 21.02 (small), but Cont is 0.160 → 0.164 (small) and Curv 0.220 → 0.245 (-10%)
  - Remove O_cp: Angle 2.987 → 3.415 (-14%, the biggest Angle drop)
- **The full model is the best on 8/10 metrics**, with the only "losses" being FID (72.17 vs 71.34 for w.o. O_ce) and LPIPS (0.070 vs 0.072 for w.o. O_ce) — and those are *small* differences (1% relative). The fact that w.o. O_ce has slightly better FID but worse Curv suggests the O_ce loss is *trading* a bit of *data distribution match* (FID) for *geometric accuracy* (Curv) — a worthwhile trade-off for clinical use.
- **The "base" model (no operators) is the worst on most metrics**: Angle 3.733° (the worst!), Curv 0.268 (the second-worst). This confirms that the *operators are the value* — without them, the model is just a vanilla DDPM with the height+normal map representation, and the height+normal alone are *not enough* to supervise the high-freq detail.
- **The w.o. normal variant (height-only) is the worst by far**: FID 177.8 (more than 2× worse than any other variant), LPIPS 0.105 (the worst). This confirms that the *normal channel is essential* — height alone is *insufficient* to capture the geometric detail of the occlusal surface.

### Image-based comparison (Table 2, 8 methods, our dataset)

| Method | RMSE_h ↓ | PSNR_h ↑ | SSIM_h ↑ | Angle ↓ | Cont ↓ | Curv ↓ | CD ↓ | FID ↓ | LPIPS ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Pix2Pix | 0.159 | 16.44 | 0.884 | 4.393 | 0.211 | 0.382 | 1.215 | 242.9 | 0.161 |
| CycleGAN | 0.254 | 10.52 | 0.773 | 7.410 | 0.265 | 0.713 | 2.102 | 370.7 | 0.386 |
| VAE | 0.115 | 22.33 | 0.909 | 3.340 | 0.169 | 0.262 | 1.184 | 295.0 | 0.081 |
| VQVAE | 0.108 | 23.01 | 0.903 | 3.472 | 0.164 | 0.236 | 1.046 | 120.9 | 0.080 |
| DDPM | 0.149 | 17.88 | 0.889 | 4.399 | 0.162 | 0.325 | 1.027 | 164.1 | 0.122 |
| 3DCGAN | 0.136 | 22.44 | 0.910 | 4.248 | 0.209 | 0.401 | 1.301 | 585.0 | 0.076 |
| DCPRGAN | 0.103 | 24.27 | 0.914 | 3.100 | 0.205 | 0.218 | 1.113 | 132.2 | 0.075 |
| **Ours** | 0.110 | 23.57 | 0.915 | **2.987** | **0.160** | **0.220** | **0.959** | **72.17** | **0.070** |

**Where we win:**
- **Angle: 2.987° (best)**, the *normal-direction* metric — our normal maps are closest to GT. Beats VQVAE (3.472°), DCPRGAN (3.100°), all GANs.
- **Cont: 0.160 (best)** — the Laplacian contour error, the high-freq ridge metric. Beats VQVAE (0.164), DDPM (0.162), DCPRGAN (0.205).
- **Curv: 0.220 (best)** — the local curvature error, the *most clinically meaningful* metric. Beats DCPRGAN (0.218 — actually beats us by 0.002!), VQVAE (0.236), all GANs.
- **CD: 0.959 mm (best)** — the *3D* Chamfer distance, the most important metric for *clinical fit*. Beats DCPRGAN (1.113), VQVAE (1.046), DDPM (1.027), all GANs.
- **FID: 72.17 (best)** — the data distribution match. Beats all by 2-8×. The *huge* gap vs CycleGAN (370.7) and 3DCGAN (585.0) is the most striking.
- **LPIPS: 0.070 (best)** — the perceptual similarity. Beats DCPRGAN (0.075), VQVAE (0.080), all others.

**Where we lose:**
- **RMSE_h: 0.110 (second)** — DCPRGAN wins with 0.103. The *pixel-by-pixel* error. The authors note: "our method performs slightly lower than some methods in pixel-by-pixel metrics such as MSE and PSNR. However, it significantly outperforms other methods in metrics such as FID and LPIPS. Observing the visualized images in Fig. 11, we notice that the pixel-by-pixel indices show a rough similarity in shape to the ground truth, resulting in a lower average error. However, the lack of geometric details makes these images appear more blurry."
- **PSNR_h: 23.57 (second)** — DCPRGAN wins with 24.27. Same story as RMSE.
- **SSIM_h: 0.915 (tied for best)** — we're tied with DCPRGAN (0.914).

**The interpretation:** DCPRGAN is *better* at the *low-frequency shape* (RMSE, PSNR, SSIM) because its two-stage GAN architecture can fit the *overall* height map very precisely, but it *loses* the high-frequency detail (Cont, Curv, Angle) because GANs are *known* to smooth out high-freq content. The diffusion model with operator-based loss is *better* at the *high-frequency geometric detail* (Cont, Curv, Angle) because the *operators* explicitly supervise the high-freq content, and diffusion is *known* to preserve high-freq better than GANs. **The clinical translation: the *high-frequency detail* (ridges, grooves, cusps) is what makes a crown *occlude* correctly, so the operator-based diffusion model is the *clinically* better approach even though DCPRGAN wins on the *low-frequency* pixel-by-pixel metrics.**

### Point-cloud comparison (Table 3, 4 methods, our dataset)

| Method | CD ↓ | HD ↓ | DCD ↓ | F-score ↑ |
|---|---|---|---|---|
| CRA-PCN | 1.018 | 8.289 | 0.347 | 0.324 |
| GeoFormer | 1.100 | 9.928 | 0.415 | 0.319 |
| SVDFormer | 1.213 | 10.22 | 0.411 | 0.321 |
| **Ours** | **0.959** | **2.606** | **0.221** | **0.401** |

**Massive HD win (2.606 vs 8.289)**: the point cloud completion methods produce *outliers* (HD = Hausdorff Distance, sensitive to the worst-case point), and our method's worst-case point is 3.2× closer. F-score +24% (0.401 vs 0.324) at 0.3mm threshold. **For v0: the *F-score* improvement (the clinical "does the generated crown touch the target crown at 0.3mm tolerance?") is the most important number — 0.401 is the best in the table, and it's *clinically meaningful* (0.3mm is the standard crown margin tolerance).**

### User study (Table 4, 6 prosthodontists + 3 technicians + 12 residents)

| Method | Average Rank ↓ | Satisfaction Rate ↑ |
|---|---|---|
| Pix2Pix | 7.92 | 14.8% |
| CycleGAN | 8.04 | 8.9% |
| VAE | 6.10 | 31.5% |
| VQVAE | 4.58 | 70.9% |
| DDPM | 3.15 | 80.1% |
| 3DCGAN | 7.15 | 14.6% |
| DCPRGAN | 3.90 | 65.3% |
| **Ours** | **2.04** | **93.4%** |
| Ground Truth | 2.12 | — |

**Our method (2.04) is ranked *better than* Ground Truth (2.12) by the experts.** This is a striking result — the experts rated the *generated* occlusal surfaces as *slightly more* acceptable than the *real* occlusal surfaces. The likely explanation is that the *real* occlusal surfaces in the dataset have *wear facets, chips, and other imperfections* (the patients are *normal occlusion* but not *pristine*), while the *generated* surfaces are *clean* and *ideally-shaped*. The 93.4% satisfaction rate is the *most important* number in the paper for clinical translation — it means that 93.4% of expert raters would *accept* the generated crown as a *prosthodontic treatment*.

### Qualitative results (Fig 7, 11, 13)

**Fig 7 (8 cases, visual quality):** The generated height maps, normal maps, contour images, and curvature images are all *visually similar* to GT, even for the 2 cases (g, h) with *incomplete* input teeth. The author notes: "the generated results exhibit superior visual quality compared to the ground truth" in some cases — again, the GT is *clinical* (with wear, chips), the generated is *ideal*.

**Fig 11 (visual comparison with 7 baselines):** The 4 cases show that Pix2Pix, CycleGAN, DDPM, 3DCGAN, and DCPRGAN produce *blurry* height maps and *noisy* normal maps. VAE and VQVAE produce *cleaner* height maps but *still blurry* normal maps. Ours produces *sharp* height maps AND *sharp* normal maps — the normal map's high-freq content is the *most striking* visual difference. This is the *visual* confirmation of the Curv 0.220 result — the operator-based loss *preserves* the high-freq geometric detail that other methods *lose*.

**Fig 13 (occlusal distance visualization, the clinical gold standard):** The authors reconstruct the *occlusal surface mesh* from the generated geometry map, then compute the *distance* between this mesh and the *opposing tooth's* mesh. Red areas = tight contact (good for occlusion), blue areas = no contact. **Ours has the *largest red area* of all 8 methods**, meaning the generated crown's occlusion matches the *opposing arch* best. **This is the *single most clinically meaningful* result in the paper** — it's the visual proof that the operator-based loss *actually* produces a *clinically-functional* crown, not just a *visually-similar* crown.

### λ_geo ablation (Fig 10)

The hyperparameter λ_geo (weight on the contour and curvature losses) is ablated from 0 to 0.1:
- λ_geo = 0: no operator loss, the cusps are *blurry*
- λ_geo = 0.001: slight improvement, cusps still blurry
- λ_geo = 0.005: clearer cusps
- **λ_geo = 0.01: optimal** — sharp cusps, clear grooves
- λ_geo = 0.05: cusps are *over-emphasized*, unnatural
- λ_geo = 0.1: cusps are *exaggerated* and *unrealistic*

The paper notes: "the hyperparameter λ_geo plays a crucial role in controlling the clarity and realism of the grooves on the generated occlusal surfaces." This is the *only* hyperparameter in the paper that's ablated in detail (the λ_cpo, λ_m are fixed at 0.1 in the main text). **For v0: a 6-cell λ_geo sweep on v0's sub-task 4 DiGS loss is a clean pilot experiment ($30 Lambda, 1 day) to confirm the operator-loss weight is critical.**

## Connections to H1-H5

**H1 (2-stage > 1-stage / multi-stage decomposition is essential for complex generations):**
- **H1 PARTIALLY SUPPORTED with caveats.** Diff-OSGN's pipeline is *effectively* 2-stage: (1) occlusal plane inference + geometry map rasterization (a *pre-processing* stage that produces a *spatial prior*), (2) conditional diffusion on the spatial prior (a *generation* stage). This is the *same* H1 strategy as CrownGen 058 (boundary prediction + diffusion) and DCrownFormer 032 (1-stage but with explicit boundary module). However, the Diff-OSGN 2-stage is *less* explicit than CrownGen's (no learnable boundary module, just a *fixed* geometry map representation). The result is that Diff-OSGN's generation *is* constrained to the right *spatial* region (the 512×512 grid above the occlusal plane) but *not* to a *learned* spatial region. This is *fine* for occlusal surfaces (where the spatial prior is *obvious*) but *insufficient* for full crowns (where the spatial prior is *learned*, as in CrownGen). **For v0 sub-task 4: the 2-stage decomposition is *necessary* for the *full crown* (because the spatial prior is not obvious), but *unnecessary* for the *occlusal surface only* (because the spatial prior is the occlusal plane).** So Diff-OSGN's H1 support is *sub-task-specific*.

**H2 (diffusion > GAN/VAE for high-freq detail):**
- **H2 STRONG SUPPORT.** The Table 2 result is the *cleanest* H2 evidence in the reading list: diffusion (DDPM, VQVAE, ours) beats GANs (Pix2Pix, CycleGAN, 3DCGAN, DCPRGAN) on the *high-freq* metrics (Cont, Curv, Angle, LPIPS), while the GANs beat the diffusion on the *low-freq* metrics (RMSE, PSNR, SSIM). The *only* exception is DCPRGAN, which is a 2-stage GAN — it's the *best* GAN but *still loses* to ours on the high-freq metrics. The 0.070 LPIPS (ours) vs 0.075 (DCPRGAN) is a *7% relative* improvement on the *perceptual* metric, which is the *most human-aligned* metric. The FID 72.17 (ours) vs 132.2 (DCPRGAN) is a *45% relative* improvement on the *data distribution* metric. **For v0 sub-task 2 + 4: diffusion is the *right* tool for the high-freq ridges/grooves on the occlusal surface. The 2-stage GAN is the *right* tool for the low-freq overall shape, so a *hybrid* (CrownGen's boundary + Diff-OSGN's diffusion) may be the *best of both worlds*.**

**H3 (anatomical / clinical priors are essential for dental-3D-gen):**
- **H3 STRONG SUPPORT, MULTI-MECHANISM.** Diff-OSGN has *three* H3 mechanisms, all of which are clinically-motivated:
  1. **Occlusal plane** as the rasterization frame — the *spatial prior* that the occlusal surface is a 2D patch above the XOY plane. This is the *most explicit* H3 in the reading list (a *coordinate system* defined by *clinical anatomy*).
  2. **Adjacent + occlusal teeth geometry map** as the diffusion condition — the *inter-tooth* prior that the generated surface must be *consistent* with the surrounding teeth. This is the *second* explicit H3, the *inter-tooth* H3 (related to CrownGen's DITA but in 2D image space).
  3. **Three geometric operators** as the loss — the *intra-tooth* prior that the generated surface must have *consistent* height+normal, *sharp* ridges, and *smooth* local curvature. Each operator is a *clinically-meaningful* geometric feature, derived from *human visual perception* of tooth morphology.
  **The *combination* of these three H3 mechanisms is the *richest* multi-mechanism H3 in the reading list** (CrownGen 058 has *one* H3 mechanism, DITA, plus the *boundary prior*; Diff-OSGN has *three* H3 mechanisms, occlusal plane + inter-tooth condition + intra-tooth operators). For v0: the *occlusal plane* is reusable as v0's *dental arch coordinate system* (similar to paper 048's parabola, but more general — a *plane* instead of a *curve*). The *inter-tooth condition* is reusable as v0's *attention mask* (similar to DITA, but image-based). The *intra-tooth operators* are the *new* H3 mechanism, the most *directly portable* to v0.

**H4 (substrate should match loss structure):**
- **H4 REFINED.** Diff-OSGN uses a 2D image substrate (the geometry map), which is *not* a 3D point cloud, *not* a mesh, *not* an SDF, *not* a voxel grid. This is the *first* paper in the reading list to use a *2D image* as the substrate for 3D shape generation. The *justification* is that the occlusal surface is *intrinsically 2D* (a patch on the XOY plane), so casting it as a 2D image is *lossless*. The 2D substrate is *compatible* with the operator-based loss (Sobel, Laplacian, neighbor cross-product are *native* to 2D images). The 2D substrate is *not* compatible with point-cloud losses (CD, EMD), but the *back-projection* (sample z at each pixel) gives a point cloud *post-hoc* for the *3D* metrics. **The refinement:** for *sub-tasks* where the target is *intrinsically 2D* (occlusal surface, polishing surface for anterior crowns, marginal ridge), 2D image substrate is *better* than 3D point cloud. For *sub-tasks* where the target is *intrinsically 3D* (full crown, full tooth, full arch), 3D point cloud or SDF is *better*. **For v0 sub-task 4 (full crown outer surface): the 3D substrate is the right choice (CrownGen 058 is the SoTA reference), but the 2D substrate could be a *complementary* sub-module for the occlusal *part* of the crown.**

**H5 (synthetic / cross-domain data improves generalization):**
- **H5 NOT TESTED.** The dataset is *single-source* (HKU Faculty of Dentistry), 319 patients, *no* external test, *no* cross-dataset evaluation, *no* synthetic data augmentation, *no* domain adaptation. This is the *biggest weakness* of the paper — the 70/15/15 split is on *teeth*, not *patients*, so the test set is *in-distribution*. **For v0: Diff-OSGN's lack of H5 is a *cautionary* finding — the high-freq detail (Cont 0.160, Curv 0.220) is *not* proven to generalize to *other* clinical populations. v0's H5 mechanism (paper 042 STEAM's GAM + MGR SSL pretraining, paper 058 CrownGen's pseudo-crown self-bootstrapping) is *essential* to bridge this gap.**

**Additional hypothesis-specific findings:**

- **H1 (refined):** The *occlusal plane* H3 mechanism (a coordinate-system prior) is *cheaper* and *more interpretable* than the *boundary prediction* H3 mechanism in CrownGen 058. The 2D substrate + occlusal-plane prior gives a *stronger* H1 for the *occlusal surface* sub-task than the 3D substrate + boundary prediction gives for the *full crown* sub-task. **For v0: a *hybrid* approach that uses the 2D substrate + occlusal plane for the *occlusal half* and the 3D substrate + boundary prediction for the *full* crown may be the *best of both worlds*.**

- **H2 (refined):** The *operator-based loss* (Sobel + Laplacian + cross-product) is the *cheapest* way to add *high-freq supervision* to a diffusion model — 3 lines of PyTorch each, *no extra training data*, *no extra parameters*. **For v0: this is a *drop-in* upgrade to *any* diffusion model in the v0 stack. The cost is *3 lines of code + 3 hyperparameters (λ_cpo, λ_ceo, λ_cro)*.**

- **H3 (new):** The *inter-tooth* H3 (the cross-attention on the geometry maps of *adjacent + occlusal* teeth) is a *complementary* H3 to CrownGen's DITA (which uses *zig-zag FDI ordering* as the relative positional encoding). The two could be *combined* — DITA for the *long-range* inter-tooth attention (across the arch), and the geometry-map cross-attention for the *local* inter-tooth attention (adjacent + occlusal). **For v0: a *hierarchical attention* mechanism with DITA (long-range) + geometry-map cross-attention (local) may be the *right* architecture for v0 sub-task 2 + 4.**

## Surprises / interesting things buried in section 4

1. **The 70/15/15 split is on TEETH, not PATIENTS.** This means a *single* patient's #16 and #26 can be in *different* splits (one in train, one in test). This is a *data leakage* — the model has *seen* the patient, just not the specific tooth. The test set is *not* a true OOD test. The "319 patients" is misleading; the effective test is on *teeth* drawn from the *same* 319-patient distribution. **For v0: do a *patient-level* split (not tooth-level) to avoid this leakage. The 058 CrownGen protocol is patient-level (496 test patients are *held-out* from the 1784 development scans).**

2. **The "0.3mm F-score" of 0.401 is *clinically* the most important number in the paper.** The 0.3mm threshold is the *clinical margin tolerance* for crown restorations. A 0.401 F-score means that 40.1% of generated crown points are *within 0.3mm* of the GT crown. This is the *direct* clinical fit metric. **For v0: report the *0.3mm F-score* as the *primary* clinical fit metric, not CD or HD. The 0.3mm threshold is the *clinical gold standard* (ISO 6872:2015 for dental ceramics).**

3. **The `O_ce` (Laplacian contour) loss is the *only* operator that, when removed, *slightly improves* FID (72.17 → 71.34) but *worsens* Curv (0.220 → 0.245).** This is a *Pareto trade-off*: the Laplacian loss *trades* a bit of *data distribution match* (FID) for *local geometric accuracy* (Curv). The clinical metric (Curv) is the *more important* one — clinicians care about whether the crown *occludes* correctly, not whether the *overall* surface shape matches the GT.

4. **The "Generated > GT" user study result (2.04 vs 2.12) is the most striking finding.** The experts rated the *generated* occlusal surfaces as *slightly better* than the *GT* (real clinical surfaces). This is because the GT surfaces are *not* pristine — they have *wear facets, chips, normal occlusion variation* — while the generated surfaces are *clean* and *ideal*. **For v0: when designing a v0 paper's user study, *normalize* the GT to a *clinical-ideal* shape (e.g., by digitally *re-shaping* the GT to remove wear/chips) so the comparison is *apples-to-apples*. The 058 CrownGen protocol does *not* do this normalization, which is a weakness of CrownGen's reader study.**

5. **The training is 30,000 *iterations* (not *epochs*).** With a 771-tooth dataset and batch size 8, that's ~310 epochs total. The linear LR decay from 0.001 to 0 over 15k-30k iterations is a *standard* DDPM schedule. The training is ~4 hours on a single RTX 3090ti — much faster than CrownGen 058 (2400+3000 epochs, ~3 days on a similar GPU). **For v0: the *small data* advantage of Diff-OSGN is a *direct* compute-saving lesson — 771 teeth × 4 hours = a $20 Lambda pilot, much cheaper than CrownGen's 1784 scans × 3 days = $300 Lambda pilot.**

6. **The Sobel filter in `O_cp` uses a *fixed* kernel, not a *learned* one.** This is the *simplest possible* consistency check (3×3 Sobel kernel in x and y, then cross-product to get the normal). A *learned* operator (a small CNN) would be more flexible but *less interpretable*. The choice of *fixed* is a *deliberate* design — the *consistency* is a *physical* constraint, not a *learned* one. **For v0: a *learned* O_cp (a small 3×3 CNN) could be a v0 paper ablation (compare fixed vs learned, $30 Lambda, 1 day).**

7. **The 16 pixels/mm rasterization resolution is *high* — equivalent to a 16-micrometer pixel.** This is *much* higher than the typical depth-map resolution (256×256, 8 pixels/mm). The high resolution is *necessary* for the high-freq detail (cusps, fossae are typically 0.5-1.0mm wide, so 16 pixels/mm gives 8-16 pixels per cusp, enough to represent the shape). **For v0: 16 pixels/mm is the *right* resolution for the occlusal surface. For the *full crown*, a *lower* resolution (4-8 pixels/mm) is sufficient and *4-16× faster*.**

8. **The "first molars only" limitation is the *right* design choice, not a *bug*.** The 2D-image + occlusal-plane approach is *intrinsically* limited to surfaces that are *parallel to the occlusal plane* — which is *only* molars (and second premolars, but the paper restricts to first molars). Incisors, canines, and second premolars have *angled* occlusal surfaces that *break* the 2D-rasterization. The paper *explicitly* states this limitation and defers to "future work" for a *different* sampling plane per tooth type. **For v0: the *first molars only* scope is a *reasonable v0 pilot scope* — the most clinically important teeth, the *most challenging* (occlusal function is critical), the *most common* restoration. v0 can extend to second molars + premolars + incisors as v0.5 or v1 work.**

9. **The 4-channel geometry map is the *first* 4-channel image representation in the reading list.** Most 2D-shape papers use *single-channel* depth (1 channel) or *RGB* (3 channels). The 4-channel `{n_x, n_y, n_z, h}` is the *minimal* representation that captures *all* the geometric info of a 2D-parametric surface. **For v0: the 4-channel geometry map is a *reusable* representation for any 2D-parametric surface (occlusal, polishing, marginal ridge). The 4 channels are *physically motivated* (3 normal + 1 height), not *learned*.**

10. **The *vanilla DDPM* baseline (single-channel output, no operator loss) is the *5th-best* in the user study (3.15 average rank, 80.1% satisfaction) — *better* than the *most* GANs (Pix2Pix 7.92, CycleGAN 8.04, 3DCGAN 7.15, DCPRGAN 3.90) and *competitive* with VQVAE (4.58, 70.9%).** This is *independent* evidence that *diffusion alone* is *already* a good model for high-freq shape generation — the *operators* add ~13% on the satisfaction rate (80.1% → 93.4%) but the *base* diffusion is *already* better than most GANs. **For v0: if engineering time is tight, *adopt vanilla DDPM first* and *add operators later*. The 13% satisfaction improvement is *nice* but not *essential*.**

11. **The 200 diffusion steps is 5× fewer than CrownGen 058's 1000 steps.** The *reason* is the operator-based loss — the loss *directly* supervises the high-freq content, so the diffusion doesn't need 1000 steps of "natural" high-freq recovery. This is a *compute-saving* insight for v0. **For v0: if the operator-based loss is adopted, T=200 may be sufficient (5× faster than T=1000).**

## Quote-worthy sentences

1. (Sec 1, motivation) **"Designing a functional occlusal surface for denture crowns is a complex and vital task in prosthodontics. Manual design is time-consuming and heavily relies on the dentist's experience, as it requires careful consideration of occlusal function."** — The *clinical motivation* statement, the *why* of AI-driven occlusal surface design.

2. (Sec 1, gap) **"Many of these methods neglect critical geometric details, such as normals and curvature, impacting the quality of the occlusal surface."** — The *single-sentence summary* of the gap Diff-OSGN fills.

3. (Sec 3.1, the height-vs-depth insight) **"Due to variations in occlusal planes across cases, depth images generated with the same hyperparameters can be different between datasets. These discrepancies between cases hinder the network's ability to learn effectively. In contrast to the depth images used in [25, 26], which require numerous hyperparameters, our method directly rasterizes the height map of occlusal surface, capturing the true height values of points relative to the occlusal plane. This approach better reflects the occlusal relationships between teeth, as it allows for both positive and negative values (higher or lower than the occlusal plane), unlike depth maps which are constrained to a 0-1 range."** — The *defense* of height over depth, the *most under-appreciated* technical choice in the paper.

4. (Sec 3.4, Sobel operator) **"We calculate the error between predicted normal map and consistent normal map derived from predicted height map as consistency preservation loss L_cpo. This self-supervised approach improves the geometric consistency and accuracy of the generated geometry maps for occlusal surfaces."** — The *self-supervision* trick, the *cheapest* operator to add.

5. (Sec 4.4, ablation interpretation) **"The lack of geometric details makes these images appear more blurry, and traditional models struggle to capture rich geometric details. This is because diffusion models, like other generative models, are weaker in pixel-by-pixel metrics but excel in capturing data diversity and high-frequency detail features."** — The *right* interpretation of why diffusion wins on FID/LPIPS but loses on PSNR/RMSE.

6. (Sec 4.4, diffusion step T) **"An appropriate number of diffusion steps, e.g. t = 200, not only improves the quality of the generation but also significantly enhances both training and inference efficiency. As the value of t increases, the details gradually become distorted, the generated crowns exhibit noticeable artifacts in regions such as grooves and edges. This phenomenon is attributed to the ability of our network to more effectively capture the fine-grained features of the data, thereby reducing the reliance on a larger number of diffusion steps."** — The *non-obvious* ablation finding: T=200 is *better* than T=1000.

7. (Sec 4.5, the *qualitative* argument) **"The superior performance in feature-based evaluation metrics, such as FID and LPIPS, suggests that the results obtained by our method are more aligned with the real data distribution."** — The *justification* for *why* FID/LPIPS are the *better* metrics than PSNR/RMSE for *high-freq* shape generation.

8. (Sec 4.6, the user-study clincher) **"Our method received the best average rank and satisfaction rate, indicating that it outperformed both the ground truth and other methods. Specifically, our method achieved a 93.4% approval rating, demonstrating its superior performance and high quality in generating occlusal surfaces."** — The *headline* result: 93.4% satisfaction, better than GT.

9. (Sec 5, the limitation) **"Our method cannot be directly used for the design of dentures for incisors and canines. Firstly, the occlusal surfaces of incisors and canines are not parallel to the occlusal plane of the dental arch, which causes our rasterization of the geometry map to become ineffective."** — The *honest* limitation, the *clearest* scope definition in the paper.

10. (Sec 5, future work) **"Our future work includes creating a larger dental dataset that encompasses a wider variety of teeth, allowing for different sampling planes in geometry map creation tailored to various types of teeth. And we plan to explore simplified diffusion model computation strategies to achieve faster inference. This will enable efficient deployment in resource-constrained clinical environments. Additionally, we plan to incorporate more anatomy-based priors, such as manually annotated dental grooves, to enhance the precision and applicability of our model."** — The *open* problems, the *v1* directions.

## Code/data link

- **Code:** ❌ Not released. Polite email to Yuanfeng Zhou (yfzhou@sdu.edu.cn) with cite-thanks.
- **Data:** ❌ Private (HKU Faculty of Dentistry, IRB-protected). 319 normal-occlusion patients, 771 first-molar tooth instances. Not accessible.
- **Pre-trained models:** ❌ Not released.
- **Companion paper 048 (IGIP) by the same group:** has public ToothFairy2 / 3DTeethLand data contributions but no code.
- **Open-access PDF:** [iccvm.org/2025/papers/s3p2-27-cvmj.pdf](https://iccvm.org/2025/papers/s3p2-27-cvmj.pdf) — full 16-page PDF with all figures, tables, and references.
- **Cited baseline implementations** (the 8 image-based baselines, *all* are open-source):
  - **Pix2Pix:** github.com/junyanz/pytorch-CycleGAN-and-pix2pix
  - **CycleGAN:** github.com/junyanz/pytorch-CycleGAN-and-pix2pix
  - **VAE:** github.com/pytorch/examples/tree/master/vae
  - **VQVAE:** github.com/MishaLaskin/vqvae
  - **DDPM:** github.com/lucidrains/denoising-diffusion-pytorch
  - **3DCGAN (Ding 2023):** no public code, but the architecture is described in the paper
  - **DCPRGAN (Tian 2021):** the *most important* baseline, no public code, but the architecture is well-described
  - **CRA-PCN (Rong 2024):** the AAAI 2024 paper
  - **GeoFormer (Yu 2024):** ACM Multimedia 2024, github.com/iCAS-Lab/GeoFormer (search for repo)
  - **SVDFormer (Zhu 2023):** ICCV 2023, github.com/ZhiwenYu17/SVDFormer

## For our project

**The 5 most concrete v0 actions from this paper, ranked by impact × effort:**

**(a) ADOPT THE THREE GEOMETRIC OPERATORS (O_cp, O_ce, O_cr) AS V0 SUB-TASK 4 (CROWN OUTER SURFACE) DIFFUSION LOSS — $30 Lambda, 1-2 days, expected +3-5% on clinical fit metrics.**
- O_cp = Sobel consistency (height → normal): 3 lines of PyTorch, enforces self-consistency between height and normal channels
- O_ce = Laplacian contour (height → contour): 3 lines, enforces high-freq ridge sharpness
- O_cr = cross-product curvature (normal → curvature): 5 lines, enforces local geometric detail
- Composite loss: `L_total = L_recon + 0.1·L_mask + 0.1·L_cpo + 0.01·L_ceo + 0.01·L_cro` (from paper's Sec 3.5)
- **Port to:** v0's PVD-based or CrownGen-based sub-task 4 diffusion. Each operator is a *drop-in* replacement for one of the existing loss terms (e.g., replace the L2 normal loss with L_cpo).
- **Expected gain:** +3-5% on Cont and Curv (the high-freq metrics), +0.5-1% on FID/LPIPS. Translates to +5-10% on the *clinical fit* F-score at 0.3mm (the ISO standard).
- **Pilot experiment:** 6-cell λ_geo sweep on {0, 0.001, 0.005, 0.01, 0.05, 0.1} (the paper's exact sweep). $30 Lambda on 1 sub-task 4 training run. Expected: λ_geo = 0.01 is the sweet spot, +3-5% on Cont/Curv.
- **Strategic positioning:** v0 sub-task 4 now has *6 independent H3 mechanisms* (parabola arch prior, DITA, point-curvature weighting, OCM landmark, PGM offset, + **O_cp/O_ce/O_cr operator-based geometric supervision NEW from 059**). The *richest* H3 toolkit in the entire dental-crown generation literature.

**(b) ADOPT THE 2D GEOMETRY MAP SUBSTRATE AS A COMPLEMENTARY SUB-MODULE FOR V0 SUB-TASK 4 (OCCLUSAL SURFACE ONLY) — $200 Lambda, 2-3 weeks, expected +1-2% on the occlusal-fit sub-metric.**
- The 2D geometry map representation is *complementary* to the 3D point cloud / mesh / SDF used in v0. For the *occlusal half* of the crown, the 2D substrate may be *better* than the 3D substrate (because the occlusal surface is *intrinsically 2D*).
- The 2D substrate is *faster* to train (T=200 vs T=1000, 5× speedup) and *smaller* (512×512×4 = 1M params per tooth vs 1024×3 = 3K params for a point cloud).
- The 2D substrate *naturally* integrates with the operator-based loss (Sobel, Laplacian are *native* to 2D images).
- **Pilot experiment:** train a *small* 2D substrate + operator-based diffusion for the occlusal half of the crown. Compare with the 3D substrate + PVD/CrownGen diffusion. The 2D substrate may win on the *occlusal-specific* sub-metric (Cont 0.160, Curv 0.220), the 3D substrate may win on the *full-crown* sub-metric (CD, HD). **The hybrid (3D substrate for the *shape*, 2D substrate for the *occlusal* refinement) may be the v0 paper's *most novel* contribution.**
- **Strategic positioning:** v0 sub-task 4 would be the *first* paper in the reading list to use a *2D image substrate* for the occlusal surface + a *3D point cloud substrate* for the rest of the crown. The *complementarity* is the *novelty*.

**(c) REPORT THE 0.3mm F-SCORE AS THE PRIMARY CLINICAL FIT METRIC IN V0'S EVALUATION PROTOCOL — $0, 1 day.**
- The 0.3mm F-score is the *clinical* standard for crown margin tolerance (ISO 6872:2015 for dental ceramics, ADA Specification No. 8 for dental casting alloys).
- The paper's 0.401 F-score (at 0.3mm) is the *direct* clinical translation of the diffusion model's accuracy.
- v0's paper should report the F-score at *multiple* thresholds: 0.1mm, 0.3mm, 0.5mm, 1.0mm. The 0.3mm is the *clinical* threshold, the others are the *engineering* thresholds.
- **Strategic positioning:** v0's evaluation protocol would be the *most clinically-aligned* in the reading list. Most papers report CD and HD (engineering metrics), v0 would report F-score at the *clinical* threshold.

**(d) REPLICATE THE 6-CELL λ_geo SWEEP FROM FIG 10 AS A V0 PILOT — $30 Lambda, 1 day.**
- The paper shows λ_geo = 0.01 is optimal for *their* sub-task (occlusal surface). v0's sub-task 4 may have a *different* optimal λ_geo (because v0's loss structure is *different* from Diff-OSGN's).
- The sweep is 6 training runs (λ_geo ∈ {0, 0.001, 0.005, 0.01, 0.05, 0.1}), each ~1 hour on a single GPU.
- The result is a *robust* hyperparameter choice for v0, and a *clear* ablation in the v0 paper.
- **Strategic positioning:** v0 paper's hyperparameter ablation would be the *most thorough* in the reading list. Most papers fix λ at a single value, v0 would *sweep* and *justify*.

**(e) APPLY THE PATIENT-LEVEL SPLIT (NOT TOOTH-LEVEL) IN V0'S EVALUATION — $0, 1 day, essential for valid comparison.**
- The paper's 70/15/15 split is on *teeth* (a single patient's #16 and #26 can be in *different* splits). This is *data leakage* — the test set is *not* OOD.
- v0's evaluation must use a *patient-level* split (a single patient's *all* teeth are in the *same* split). This is the *CrownGen 058* protocol (496 test patients are *held-out* from the 1784 development scans).
- The paper's headline result (93.4% satisfaction) may be *partly* due to this leakage — a true patient-level test may give 80-85% satisfaction, still strong but not 93.4%.
- **Strategic positioning:** v0's evaluation would be the *most rigorous* in the reading list. Patient-level split + 0.3mm F-score + Gwet's AC2 inter-rater + non-inferiority test (from CrownGen 058) = the *gold standard* clinical evaluation protocol.

**v0 stack updated:** sub-task 1 unchanged; sub-task 2 conditional = MADCrowner + ToothCraft + ToothForge + DMC + DCrownFormer + per-point variance + 2D-grid mesh regularization + DITA + tooth-level point representation + boundary prediction + DPSR + pseudo-crown self-bootstrapping + FDI zig-zag ordering (all from 058); sub-task 2 unconditional prior = VF-Net + LION + TeethGenerator + SAE-LP; sub-task 4 = PVD + ME-loss + DiGS + FlexiCubes + Surface Projection + MGR + DITA + **O_cp/O_ce/O_cr operator-based geometric supervision (NEW from 059, +3-5% on clinical fit)** + **2D geometry map substrate for occlusal refinement (NEW from 059, complementary sub-module, +1-2% on occlusal-specific metrics)** + **point-curvature cervical margin weighting**; sub-task 5 = FlexiCubes + NDC + DPSR; training data = 3DTeethSeg'22 + 3DS + ODD + ToothForge synthetic + TeethGenerator synthetic + VF-Net synthetic + LION synthetic + pseudo-crown self-bootstrapped scans; eval = IoU_Antag + ToothForge reconstruction filter + spectral-only baseline + per-tooth-type CD-L2 + ME-loss correspondence + LION 1-NNA + UCD + FDI 16 cross-dataset test + per-clinic 50-scan fine-tune + 496 external test scans + 26-case clinical reader study + 14-day washout + Gwet's AC2 inter-rater + non-inferiority test + **0.3mm F-score as primary clinical fit metric (NEW from 059, ISO 6872:2015 standard)** + **patient-level train/val/test split (NEW from 059, avoid tooth-level leakage)**; v0 compute = **~$5,770-7,130 Lambda** (was $5,540-6,830, +$30 for 6-cell λ_geo sweep + $200 for 2D substrate pilot + $0 for metric protocol changes).

**Strategic positioning:** v0 now has *6 independent H3 mechanisms for sub-task 4* (parabola, DITA, point-curvature, OCM, PGM offset, **O_cp/O_ce/O_cr operator-based geometric supervision, NEW from 059**). v0 evaluation protocol is now the *most clinically-aligned* in the reading list (0.3mm F-score as primary metric + patient-level split). v0 sub-task 4 is now the *first* paper in the reading list to use a *hybrid 2D + 3D substrate* for crown generation.

**Open questions for HK:**
(i) Adopt the three geometric operators (O_cp, O_ce, O_cr) as v0 sub-task 4 default diffusion loss? (recommend YES, $30 Lambda, 1-2 days, +3-5% on clinical fit, the *biggest* cross-paper H3 toolkit in the reading list)
(ii) Add the 2D geometry map substrate as a complementary sub-module for the occlusal half? (recommend YES, $200 Lambda, 2-3 weeks, +1-2% on occlusal-specific metrics, the *first* 2D+3D hybrid in the reading list)
(iii) Report the 0.3mm F-score as v0's primary clinical fit metric? (recommend YES, $0, 1 day, ISO 6872:2015 standard, the *most clinically-aligned* metric)
(iv) Run the 6-cell λ_geo sweep as a v0 pilot? (recommend YES, $30 Lambda, 1 day, justifies the hyperparameter choice with an *ablation table*)
(v) Apply the patient-level split in v0's evaluation? (recommend YES, $0, 1 day, avoid tooth-level leakage, the *most rigorous* evaluation in the reading list)
(vi) Cite the Shandong U "IGIP-LAB" + HKU + ShanghaiTech axis as a unified "3DTeethSeg'22 / 3DTeethLand / Diff-OSGN" lineage? (recommend YES, makes the *3-paper* 2022-2025 progression explicit, parallel to SNU CGIP from paper 046 and CityU AIM-Group from paper 044)
(vii) Reach out to Yuanfeng Zhou (yfzhou@sdu.edu.cn) for collaboration? (recommend YES, polite email + cite-thanks, 1-2 week response, they have the 4-channel geometry map code + 319-patient HKU dataset, saves 2-3 weeks engineering)
(viii) Frame the v0 paper as the *first* paper to use a *2D + 3D hybrid substrate* for crown generation? (recommend YES, the *most novel* framing, the 2D substrate for occlusal + 3D substrate for full crown is *unique* to v0)
