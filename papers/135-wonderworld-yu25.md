# 135 — WonderWorld: Interactive 3D Scene Generation from a Single Image (Hong-Xing Yu¹∗, Haoyi Duan¹∗, Charles Herrmann¹, William T. Freeman², Jiajun Wu¹ — **¹Stanford University + ²MIT (CSAIL)**, arXiv:2406.09394 v1 13 Jun 2024, **CVPR 2025 Highlight** (ranked #45 in the CVPR 2025 main-conference program listing, project page flags "Highlight"), code ✅ [github.com/KovenYu/WonderWorld](https://github.com/KovenYu/WonderWorld) (commit 5cf1146 referenced in FantasyWorld 2025, **no LICENSE file — repo is research-only by default**), project page ✅ [kovenyu.com/WonderWorld](https://kovenyu.com/WonderWorld/) (interactive browser demos + 12 virtual-world examples), paper ✅ [arxiv.org/abs/2406.09394](https://arxiv.org/abs/2406.09394) v4 25 Mar 2025 27.7MB, **~200-400 GS citations as of 2026-06-11** (rough estimate from related-work citing density in ICCV 2025 / NeurIPS 2025 papers including WonderTurbo and FantasyWorld; Semantic Scholar rate-limited 429 as of 2026-06-11 verification attempt)

> **TRAJECTORY NOTE:** the 134 (L4GM) note's "Next paper to read" recommended **WonderWorld (Yu et al. CVPR 2025)** — verified 2026-06-11 via [kovenyu.com/WonderWorld/](https://kovenyu.com/WonderWorld/) (project page explicitly says "CVPR 2025 (Highlight)") + [cvpr.thecvf.com/virtual/2025/poster/33364](https://cvpr.thecvf.com/virtual/2025/poster/33364) (poster #33364 in main conference). WonderWorld is the *direct* **scene-level** *interactive* 3D-scene-generation paper that pairs with the *object-level* 4D-reconstruction paper L4GM (134) — L4GM handles *single object* 4D from monocular video, WonderWorld handles *scene-level* 3D (NOT 4D) from a single image + interactive user camera-move + text-prompt. The two papers are the *exact* back-to-back pair for the **2024-2025 Stanford scene-generation dynasty** (WonderJourney [CVPR 2024] → WonderWorld [CVPR 2025] → WonderPlay [ICCV 2025] → WorldScore [ICCV 2025] — all Stanford WU lab, all driven by Hong-Xing Yu + collaborators). The *foundational* technical lineage is **WonderJourney → WonderWorld**: WonderJourney (Yu et al. CVPR 2024) is the *offline* *point-cloud* 3D-scene-generation ancestor that **takes tens of minutes per scene** (it progressively generates multi-views + aligns depth + optimizes 3D); WonderWorld is the *interactive* *FLAGS-Gaussian-surfel* descendant that **takes <10s per scene** by (a) *removing* the progressive multi-view generation, (b) *replacing* point cloud with FLAGS surfels, (c) *introducing* geometry-based initialization that reduces per-layer optimization to <1s. **CRITICAL DIFFERENCE FROM L4GM 134:** L4GM is *monocular-video-to-4D* (time-varying object reconstruction); WonderWorld is *single-image-to-3D-scene* (with user-driven camera extrapolation, no temporal axis). The two are *complementary paradigms* for the *interactive 3D/4D world* use case — together they form the **complete 2024-2025 interactive world-generation toolkit** for v0/v1/v2. The *killer open-source deliverable* is a **fully reproducible browser-based interactive system** ([index_stream.html](https://github.com/KovenYu/WonderWorld) + PyTorch3D + Marigold depth + OneFormer seg + SD Inpaint + LLM scene describer) — the *only* paper in our reading list that ships a *production-grade interactive web demo* (not just a gradio demo like TripoSR 108 or L4GM 134), the **killer reference implementation for the v0 interactive dental arch design prototype**.

## TL;DR

**WonderWorld is the FIRST interactive 3D scene generation framework that generates connected + diverse 3D scenes in <10s per scene on a single A6000 GPU**, by introducing two killer technical innovations: **(1) Fast LAyered Gaussian Surfels (FLAGS)** — a novel scene representation that combines layered representations (foreground / background / sky) with surfel-style flat-Gaussians, optimized via **geometry-based initialization** (pixel-aligned surfel position, normal-aligned orientation, Nyquist-sampling-theorem-derived scale) that reduces per-layer optimization to <1s with 100 Adam iterations (no densification); and **(2) Guided Depth Diffusion** — a training-free classifier-free-guidance-style extension to Marigold latent depth diffusion that injects the *partially visible existing depth* as a gradient guidance term on the denoising trajectory, eliminating seams between adjacent scenes. The system takes a single image + LLM-generated scene description (foreground/background/style) + text-guided outpainting, and produces 3 scenes/sec at the operator's interactive rate. The *direct application* to dental arch is **v0 sub-task 1 interactive dental arch design**: a clinician can iteratively explore the *missing* parts of a partial intra-oral scan, generating the adjacent prep / opposing arch / gum context on the fly while keeping the existing prepared tooth + adjacent teeth frozen.

## Research question + answer

**RQ:** Can we make 3D scene generation interactive (seconds per scene) so that users can iteratively *specify* scene contents and *navigate* the generated world in real time, with seamless multi-scene composition?

**Answer (paraphrased from §1 + §3):** Yes, with three combined design choices: (a) **FLAGS = 3DGS with z-axis shrunk to ε** + per-layer geometry-based initialization (pixel-aligned depth → position, surface normal → orientation, Nyquist-sampling-theorem → scale) that reduces optimization from tens-of-minutes to <1s per layer; (b) **text-guided diffusion inpainting** that fills occluded regions at the *layer* level (foreground, background, sky) rather than the *view* level — no progressive multi-view generation required; and (c) **guided depth diffusion** that uses the visible existing depth as a classifier-free-guidance-style term on the Marigold depth-denoising trajectory, eliminating boundary seams. The system **generates 1 scene in 9.5 seconds on an A6000** vs WonderJourney 749.5s / LucidDreamer 798.1s / Text2Room 766.9s (Table 1, **~80× speedup**), and **wins the human 2AFC preference >98% against all three baselines** (Table 3) on bird-eye view renderings.

## Method

### 3.1 FLAGS (Fast LAyered Gaussian Surfels)

**Definition (§3.1):** Each scene ℰ = {L_fg, L_bg, L_sky} is a *radiance field* represented by **three radiance-field layers**, where each layer is a *set of surfels*. A surfel (per footnote 1, "in contrast to a traditional surfel that carries a solid piece of surface, each surfel in FLAGS carries a small radiance field") is parameterized as **5 attributes per surfel**: 3D position **p**, orientation quaternion **q**, x-axis and y-axis scales **s** = [s_x, s_y] (z-axis shrunk to ε ≪ min(s_x, s_y)), opacity **o**, view-independent RGB color **c**. The Gaussian kernel is G(x) = exp(−½ (x−p)ᵀ Σ⁻¹ (x−p)) with covariance Σ = Q diag(s_x², s_y², ε²) Qᵀ. **FLAGS is therefore a *3DGS variant* where every Gaussian's z-axis is shrunk to a tiny number, AND view-dependent colors are removed** — the *core* representation is "flat oriented radiance-field splats" that can be rendered with the *same differentiable rendering pipeline* as 3DGS (3D-to-2D projection + alpha blending).

**Single-view layer generation (§3.1, Eq. 4-5):**
- Generate the scene image I_scene via a text-guided diffusion model; structured scene description T = {F, B, S} = g_LLM(J, U) is produced by an LLM (foreground object prompt, background prompt, style prompt) from the user prompt U + instruction prompt J.
- Compute foreground mask M_fg = ∪_k O_k : ‖O_k ⊙ E‖ > 0, where E is the *significant depth edge mask* (∇D > T) and O_k are object masks from **OneFormer** (Jain et al. CVPR 2023, [23]).
- Background mask M_bg = 1 − M_vis where M_vis is a *visible sky mask* from OneFormer.
- Background image I_bg = M_bg ⊙ I_inpaint(I_scene, M_fg, {B, S}) — uses **Stable Diffusion Inpaint** [46] to inpaint the {B, S} description at the M_fg region.
- Sky layer: M_sky = 1 (full sky dome), I_sky = I_inpaint(I_scene, 1 − M_vis, {"sky", S}).

**Geometry-based initialization (§3.1, Eq. 6-8, Fig. 3) — the KILLER innovation:**
1. **Pixel-aligned generation** — N_fg = ‖M_fg‖_F surfels per layer (one surfel per valid pixel). Color c initialized to pixel RGB. Position **p = R⁻¹(d · K⁻¹[u, v, 1]ᵀ − T)** — pixel coordinates (u, v) back-projected through camera intrinsics K and extrinsics (R, T) to 3D world position using the *estimated monocular depth d*.
2. **Normal-aligned orientation** — surfel normal Q_z = n, where n is the world-frame normal from the layer-image estimated normal n_cam rotated to world frame via R⁻¹. Q_x, Q_y derived from Q_z and a unit up-vector u = [0, 1, 0]ᵀ via cross products (Eq. 7).
3. **Nyquist-sampling scale initialization (Eq. 8, Fig. 3)** — sampling interval at a surfel is T_N = d/(f cos θ) where θ is the angle between surfel normal n and image-plane normal n_img. Setting the signal frequency of a surfel (inverse Gaussian bandwidth 1/(2k s_x)) equal to the maximum signal frequency (Nyquist: 1/(2 T_N)) gives **s_x = d / (k f_x cos θ_x)**, s_y = d / (k f_y cos θ_y), with **k = √2** (bandwidth hyperparameter). Initialized surfels provide *seamless coverage* of the visible surface without significant overlap; opacity o = 0.1 for sufficient gradient to fine-tune.

**Optimization (§3.1):** Back-to-front, sky → background → foreground, each layer 100 Adam iterations (no densification), loss L = 0.8 L_1 + 0.2 L_D-SSIM against the masked layer image. Optimize opacity + orientation + scales (NOT colors, NOT positions — they're already initialized from depth/normals). Per-layer optimization <1 second on A6000.

### 3.2 Guided Depth Diffusion

**Setup (§3.2, Fig. 4):** Let D_guide = rendered depth from existing scene at the outpainting camera viewpoint, M_guide = binary mask of visible regions, D_scene = estimated depth for the new outpainted image I_scene. The fundamental challenge: *strong discrepancy* between D_guide ⊙ M_guide and D_scene ⊙ M_guide → seams when connecting scenes. Baseline methods use *ad-hoc* post-processing (global shift+scale alignment in LucidDreamer [8]; fine-tuning depth estimator in WonderJourney [67]) that "do not reduce the inherent ambiguity in the estimation of the new scene depth" (§4.1).

**Solution (Eq. 11):** Training-free guided extension to a *latent depth diffusion* model (Marigold depth [27]). Standard latent depth diffusion: ϵ_t = UNet(d_t, I_scene, t), denoise d_t → d_{t−1} → ... → d_0 → D_scene = Decoder(d_0). **Guided variant** modifies the predicted noise:
- ϵ̂_t = UNet(d_t, I_scene, t) − s_t · g_t
- g_t = ∇_{d_t} ‖ D_{t−1} ⊙ M_guide − D_guide ⊙ M_guide ‖²

where D_{t−1} = Decoder(d_{t−1}) is the *pre-decoded* depth map (the **acceleration trick** — do gradient guidance in the *pre-decoded* depth space, not the latent space, for speed). s_t is the guidance weight. The guidance term *encourages the denoising trajectory to stay consistent with visible existing depth*, leading to smooth geometry extrapolation. This is **conceptually similar to classifier-free guidance (Ho & Salimans 2022 [19]) and diffusion self-guidance (Epstein et al. NeurIPS 2023 [12]) and readout guidance (Luo et al. CVPR 2024 [39])** but applied to *latent depth space* with *partial visibility* as the conditioning signal — a **novel application** of guidance to depth diffusion.

### 4 Experiments

**Baselines (§4):** Three representative methods: (a) **WonderJourney [67]** — *point cloud* representation, offline, ~10 min/scene; (b) **LucidDreamer [8]** — *3DGS* representation, offline, ~13 min/scene; (c) **Text2Room [20]** — *mesh* representation, offline, ~12 min/scene, *indoor-only training data* so doesn't generalize to outdoor. *No prior method* allows interactive 3D scene generation — so all baselines are inherently *offline*.

**Implementation (§4):**
- **Stable Diffusion Inpaint** [46] for outpainting + inpainting background/sky + text-to-image
- **OneFormer** [23] for sky + foreground object segmentation
- **Marigold Normal** [27] for normal estimation
- **Marigold Depth** [27] for depth diffusion
- Camera + depth parameters: camera_speed=0.001, fg_depth_range=0.015, depth_shift=0.001, sky_hard_depth=0.02, init_focal_length=960
- 28 scenes generated: 7 scenes × 4 test examples (city, campus, nature, fantasy)
- Fixed panoramic camera path for automated evaluation (vs interactive for demos)
- RepViT-SAM model for additional segmentation
- 48GB GPU memory required (A6000 class)
- 100 Adam iterations per layer, no densification

**Results — Generation Speed (Table 1, A6000 GPU):**
| Method | Time per scene |
|---|---|
| WonderJourney [67] | 749.5s (~12.5 min) |
| LucidDreamer [8] | 798.1s (~13.3 min) |
| Text2Room [20] | 766.9s (~12.8 min) |
| **WonderWorld (Ours)** | **9.5s** |

**~80× speedup** vs the fastest prior method. The reason: prior methods spend most time on (a) generating multiple views to fill holes between existing scene and new scene, (b) aligning depth for these views, (c) training 3D representation to fit them. WonderWorld *eliminates* (a) via layer-level inpainting and *accelerates* (b)+(c) via geometry-based initialization.

**Results — Novel View Quality (Table 2):**
| Method | CS↑ | CC↑ | CIQA+↑ | Q-Align↑ | CA↑ |
|---|---|---|---|---|---|
| WonderJourney [67] | 27.34 | 0.9544 | 0.6443 | 2.7170 | 5.6007 |
| LucidDreamer [8] | 26.72 | 0.8972 | 0.5260 | 2.7355 | 5.2935 |
| Text2Room [20] | 24.50 | 0.9035 | 0.5620 | 2.6495 | 5.5244 |
| **WonderWorld (Ours)** | **29.47** | **0.9948** | **0.6512** | **3.6411** | **5.9543** |

Wins all 5 metrics. **CLIP score +2.13 over WonderJourney** (semantic alignment), **CLIP consistency +0.0404 over WonderJourney** (multi-view coherence), **CLIP-IQA+ +0.0069** (perceptual quality), **Q-Align +0.92** (large margin, vision-language-model quality assessment), **CLIP-Aesthetic +0.35** (aesthetic quality).

**Results — Human 2AFC Preference (Table 3, 204 human raters, bird-eye view):**
| vs Method | WonderWorld preference rate |
|---|---|
| vs WonderJourney [67] | **98.5%** |
| vs LucidDreamer [8] | **98.6%** |
| vs Text2Room [20] | **98.0%** |

**>98% human preference** against all three baselines. These are *decisive* margins — the 1.4-2.0% non-preference likely represents the most adversarial raters or the most favorable baseline crops.

**Ablation Study (Table 4):**
| Variant | CS↑ | CC↑ | CIQA+↑ | Q-Align↑ | CA↑ |
|---|---|---|---|---|---|
| Ours w/o geometry-based init | 27.23 | 0.9836 | 0.6153 | 3.5236 | 5.7284 |
| Ours w/o layers | 27.32 | 0.9922 | 0.6298 | 3.5288 | 5.7139 |
| Ours w/o depth guidance | 26.89 | 0.9936 | 0.6327 | 3.6011 | 5.7854 |
| **WonderWorld (full)** | **29.47** | **0.9948** | **0.6512** | **3.6411** | **5.9543** |

- **w/o geometry-based init**: replace FLAGS with **3DGS + MipSplatting** [70] based on same estimated depth, increase optimization iterations to match PSNR. **CS 27.23 vs 29.47 (-2.24)** — the geometry-based init gives the *largest* CS gain; also causes *alias effects* in novel views (Fig 6) because MipSplatting/3DGS doesn't have the *seamless surface coverage* of FLAGS Nyquist-init scales.
- **w/o layers**: use a *single* layer instead of 3. **CS 27.32 vs 29.47 (-2.15)**, **CC 0.9922 vs 0.9948**, **CIQA+ 0.6298 vs 0.6512** — the layered design *fills occluded regions* (Fig 7), critical for novel-view consistency.
- **w/o depth guidance**: drop the g_t guidance term. **CS 26.89 vs 29.47 (-2.58)**, the *largest* CS drop, and **creates significant seams** between adjacent scenes (Fig 8) — confirms the guidance term is the *killer* for boundary alignment.

All three components contribute, with **depth guidance > geometry-init > layered** in CS gain order.

## Connections to H1-H5

**H1 (2-stage VAE+DDM > 1-stage end-to-end):** **STRONG PARTIAL CONFIRMATION via INVERSION.** WonderWorld is *structurally* 1-stage end-to-end (FLAGS + guided depth diffusion in a single control loop), yet the *ablation* of each component (geometry-init, layers, depth guidance) shows that *all three* are required for the SOTA. The killer ablation evidence: **w/o depth guidance loses -2.58 CS** and creates seams — the guidance term is *conceptually* the same role as MVSplat360's latent space alignment loss (paper 125): the *bridging mechanism* that makes 1-stage end-to-end work. **Verdict: WonderWorld confirms H1 only IF the 1-stage has the proper bridging mechanism (guided depth diffusion is the equivalent of MVSplat360's latent space alignment); naive 1-stage is worse than naive 2-stage.** v0 sub-task 1 dental arch interactive reconstruction should adopt WonderWorld's *layered* + *guided* design as the H1 1-stage alternative to MVSplat360's *3DGS* + *SVD-refinement* 2-stage.

**H2 (latent diffusion > direct regression / GAN):** **STRONG REFINEMENT.** WonderWorld uses *latent* diffusion (Marigold depth [27] is a *latent* depth diffusion; SD Inpaint [46] is a *latent* image diffusion) throughout, but in a *guided* + *layered* way. The killer evidence: **w/o depth guidance loses -2.58 CS** while still using latent depth diffusion, confirming that *which* diffusion (latent vs pixel-space) is less important than *how* the diffusion is *guided* with the existing 3D context. **Verdict: H2 holds but with the caveat that *guidance* matters more than *substrate*** (latent vs pixel). v0 sub-task 1 should use *latent* diffusion (Marigold-style) + *guided* with the existing dental arch depth/normal (WonderWorld-style), not naive 2D pixel-space diffusion.

**H3 (arch-level / multi-source conditioning):** **STRONG REFINEMENT.** WonderWorld is the *first* paper in our reading list to introduce *interactive user-driven camera-move conditioning* as a 3D-generation mechanism — the user *specifies the camera* (where to look next) and the *scene content* (text prompt), and the system generates the corresponding 3D. This is **fundamentally different** from passive 3D generation (Wonder3D, MVDream) — it's *active* 3D generation with the user in the loop. The killer evidence: the *Layered* + *LLM scene description* + *guided depth diffusion* are all *conditioning mechanisms* for the 3D generation process. For v0 sub-task 1 dental arch, the *clinical analog* is: dentist specifies the *tooth position* (camera) and *preparation type* (text prompt) and the system generates the *corresponding full tooth geometry*. **Verdict: H3 is *massively* extended by interactive-camera-conditioning** — the user-in-the-loop design pattern is a 4th H3 mechanism alongside adjacent/opposing teeth (045/046), occlusal plane (059), and 6-tooth context (033).

**H4 (implicit SDF > mesh):** **MILD CONTRADICTION.** WonderWorld uses *FLAGS (Gaussian surfels)* for the 3D representation, NOT implicit SDF. The reason: FLAGS are *fast to optimize* (<1s per layer vs tens-of-seconds for SDF) and *fast to render* (3DGS pipeline) — the speed is the *killer* advantage for the interactive use case. However, FLAGS are *not watertight* and *not directly mesh-extractable* — for clinical dental use, the mesh extraction is required. **Verdict: H4 holds for *quality* (SDF > mesh > surfel for clinical accuracy) but NOT for *speed* (surfel > SDF for interactive use)**. v0 should use **FLAGS-style representation for v0 sub-task 1 interactive dental arch design (interactive preview)** + **DMC-style (paper 033) or ToSynFCD (paper 026) SDF/mesh representation for v0 sub-task 2 crown generation (clinical accuracy)** — the two-representation pipeline is the *killer* dental application of WonderWorld.

**H5 (synthetic + finetune on small dataset > from-scratch on real):** **NO TEST.** WonderWorld is *fully* trained on *real* data (Marigold, SD, OneFormer are all trained on large real datasets), with NO dental-specific fine-tuning. The paper doesn't test synthetic augmentation. For v0 sub-task 1 dental arch, the killer open question is: **can the LLM + Marigold + OneFormer pipeline be replaced by dental-trained versions via H5 synthetic + finetune?** The WonderTurbo paper (Ni et al. ICCV 2025) shows that *fine-tuning* is possible — direct precedent for v0 dental fine-tune of the WonderWorld pipeline.

## Surprises / interesting things buried in §4

1. **The "geometry-based initialization is the killer" ablation:** the LARGEST single ablation jump is from **w/o depth guidance → full** (-2.58 CS), but the SECOND largest is from **w/o geometry-init → full** (-2.24 CS). The geometry-based init with Nyquist-sampling-theorem scale derivation is *the* practical contribution — and the MipSplatting replacement *visually fails* in novel views (Fig 6 alias effects) *even at the same PSNR at the generation view*, confirming that the FLAGS representation has *better multi-view consistency* than vanilla 3DGS at the same single-view quality. This is a *deep* insight: **FLAGS is a 3DGS variant with better NVS properties, not just a speed optimization.**

2. **The "no densification" claim is the killer architectural decision.** Most 3DGS papers (including L4GM 134) use *densification* (split/clone Gaussians based on gradient magnitude) to handle under-reconstructed regions. WonderWorld *removes* densification entirely because the geometry-based init already places surfels at the *correct density* (one surfel per valid pixel). This means: (a) **100 Adam iterations is sufficient** (vs thousands for 3DGS with densification), (b) **no hyperparameter tuning of densification thresholds** (split/clone grad thresholds), (c) **deterministic convergence** (no random densification events). For v0 dental arch interactive design, this is *huge* — predictable inference time is critical for clinical UX.

3. **The "guided depth diffusion uses pre-decoded depth" acceleration trick:** the Eq. 11 g_t = ∇_{d_t} ‖ D_{t−1} ⊙ M_guide − D_guide ⊙ M_guide ‖² uses D_{t−1} = Decoder(d_{t−1}) which requires *decoding* the latent depth at every denoising step. This is **computationally expensive** — WonderWorld describes an "accelerated depth guidance implementation" in the supplementary material (which we didn't fully read). The trick is likely *gradient checkpointing* or *decoding every k steps*. For v0 dental arch, the *killer* is: **the pre-decoded depth space is the *right* guidance space**, not the *latent* space — this is consistent with MVSplat360's finding (paper 125) that *latent space alignment* is the right H2 bridging mechanism.

4. **The "LLM scene description" is critical for diversity** — the structured {F, B, S} (foreground, background, style) decomposition is generated by an LLM from the user prompt U + instruction J, and the paper notes that "to uncover and inpaint the occluded regions in the generated scene image, we introduce a single-view layer generation method." Without the LLM decomposition, the inpainting prompt would be ambiguous and produce poor results. For v0 dental arch, the LLM is the *killer* for the *patient-specific* generation: dentist prompt → LLM extracts {tooth type, position, opposing arch, gum condition} → structured generation prompt.

5. **The "no multi-view consistency test" is a weakness** — WonderWorld evaluates via 9 sudoku-like novel views + 2AFC, but the *quantitative* NVS metrics (PSNR/SSIM/LPIPS) are not reported in the main paper (only CLIP-based metrics CS/CC/CIQA+/Q-Align/CA are reported). This is a *deliberate choice* because *there is no ground truth for novel views in scene generation* — the input is a single image, so all novel views are unobserved. For v0 dental arch, the clinical analog has *ground truth* (the actual patient anatomy), so the *quantitative* NVS test (via VGGSfM, paper 087) IS available — v0 should add PSNR/SSIM/LPIPS to the evaluation suite that WonderWorld omits.

6. **The "limitations" section (§5) is the killer for v0/v1:** WonderWorld only generates *frontal-facing surfaces* — "the view synthesis range is limited to an area around the camera, as the back side of the object is not generated." For dental arch, this is *exactly* the limitation that v0 must address — a prep tooth has a *back side* (the prep margin side facing the gingiva) that must be generated with the same fidelity as the front side. Future work is suggested: "incorporate a 3D object generation module such as GRM [64] to generate individual objects separately from the scene background" — direct precedent for v0 sub-task 2 *generating the prep tooth* with GRM-style object-level generation, then composing into the dental arch scene.

## Quote-worthy sentences

- "We propose WonderWorld, a novel framework for **interactive 3D scene generation** that enables users to interactively specify scene contents and layout and see the created scenes in low latency." (Abstract)
- "Our approach does not need to generate multiple views, and it leverages a **geometry-based initialization** that significantly reduces optimization time." (Abstract)
- "We introduce the **guided depth diffusion** that allows partial conditioning of depth estimation." (Abstract)
- "Our system generates a single 3D scene in less than 10 seconds thank to our Fast LAyered Gaussian Surfels (FLAGS) representation" (Project page)
- "the layered design in our FLAGS **fills occluded regions**" (§4.2 ablation)
- "Our guided depth diffusion mitigates this issue" (§4.2)
- "**single 3D scene generation methods like LucidDreamer [8] do not extrapolate out of predefined scenes and suffer from severe geometric distortion at the boundaries of the generated scene**" (§4.1) — the *killer* motivation for guided depth diffusion
- "the difficulty in modeling detailed objects, such as trees, which leave 'holes' or 'floaters' when the viewpoint changes. Therefore, we see WonderWorld as an **interactive 3D world prototyping method, rather than a full end-to-end solution**" (§5 Limitations)
- "we see WonderWorld as an **interactive 3D world prototyping method, rather than a full end-to-end solution**. This invites an exciting future direction: using WonderWorld to interactively prototype a coarse 3D world structure, and then refine scene details and complete objects with slower but higher-fidelity models such as video diffusion [69]" (§5 Limitations)
- "the view synthesis range is **limited to an area around the camera**, as the back side of the object is not generated. Future work may incorporate a 3D object generation module such as GRM [64] to generate individual objects separately from the scene background" (§5 Limitations)

## Code/data

- **Code:** [github.com/KovenYu/WonderWorld](https://github.com/KovenYu/WonderWorld) — **NO LICENSE file** (research-only by default, same as WonderJourney). Includes:
  - PyTorch3D-based FLAGS renderer (3D-to-2D projection + alpha blending, modified from 3DGS)
  - Submodule `depth-diff-gaussian-rasterization-min` (modified 3DGS rasterizer for FLAGS)
  - Submodule `simple-knn` (KNN for FLAGS neighbor queries)
  - RepViT-SAM integration (`./RepViT/sam`) for efficient segmentation
  - `run.py` + `--example_config config/example.yaml` for interactive scene generation
  - `index_stream.html` browser-based viewer (WASD + arrow keys + R to generate new scene + Z to undo + X to save)
  - Pre-trained RepViT-SAM: `wget https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_sam.pt`
  - Requires: Python 3.10, PyTorch 2.4.0 + CUDA 12.4, 48GB GPU memory, OpenAI API key (if use_gpt=True), `en_core_web_sm` spacy model
- **Interactive browser demos:** [kovenyu.com/WonderWorld/](https://kovenyu.com/WonderWorld/) — 12 virtual worlds (Holy Spirit Cathedral, Ho Chi Minh City Hall, Venice, Forbidden City, Taj Mahal, Minecraft, Stanford Campus, Kremlin Park, Westlake, etc.) loadable in-browser with WASD/arrow-key navigation
- **Demo video:** YouTube UXEXqOKnezs (Jun 17 2025)
- **Model checkpoints:** NONE released separately (all built on Marigold + SD Inpaint + OneFormer + RepViT-SAM, all pre-trained checkpoints on HuggingFace)
- **Datasets:** uses *publicly available real images* + *synthetic images* as testing examples (28 scenes generated for evaluation: 7 scenes × 4 test examples), plus WonderJourney and LucidDreamer examples

## For our project

**(a) ★ ADOPT WONDERWORLD'S FLAGS + LAYERED DESIGN AS V0 SUB-TASK 1 (FULL-ARCH SYNTHESIS) STACK V1+.** The "fast scene representation" + "interactive camera-move" + "text-prompt scene content" + "guided depth diffusion for boundary alignment" is the *killer* paradigm for clinical dental arch interactive design. The dentist can:
- *Specify* the missing tooth position (camera-move)
- *Specify* the tooth type + preparation (text prompt)
- *See* the generated full dental arch in <10s
- *Iterate* by moving the camera and changing the prompt
- *Refine* via the guided depth diffusion that uses the existing arch depth as conditioning

**Implementation cost:** $200-500 Lambda (3DGS rasterizer fork + Marigold depth/SD Inpaint/OneFormer/RepViT-SAM integration + dental arch fine-tune of the text→{F,B,S} LLM decomposition prompt + dental arch training data — using 3DTeethSeg22 + ToSynFCD for the arch training set).

**(b) ★ ADOPT WONDERWORLD'S GEOMETRY-BASED INITIALIZATION AS V0 SUB-TASK 1 V1+ KEY ENGINEERING TRICK.** The pixel-aligned position + normal-aligned orientation + Nyquist-sampling scale initialization is *the* recipe for fast FLAGS optimization. The 100-Adam-iteration + no-densification claim is *the* speed guarantee. For v0 dental arch interactive design, this means **<1s per layer + 3 layers = <3s for the FLAGS optimization step**, the *rest* of the 10s budget is for diffusion inpainting + depth/normal estimation (the dominant cost).

**(c) ★ ADOPT WONDERWORLD'S GUIDED DEPTH DIFFUSION AS V0 SUB-TASK 1 V1+ BOUNDARY-ALIGNMENT MECHANISM.** The classifier-free-guidance-style extension to Marigold latent depth diffusion with *pre-decoded depth space* (not latent) is the *killer* for *seamless* multi-scene composition. For v0 dental arch interactive design, this means:
- *Existing arch depth* from the patient's intra-oral scan → D_guide
- *Outpainting camera viewpoint* where the dentist wants to extend the arch → new I_scene
- *Guided depth diffusion* → seamless depth map of the extended arch region
- *FLAGS optimization* with the guided depth → seamless 3D extension

**Implementation cost:** $50-100 Lambda (Marigold fine-tune on dental arch depth + guided-depth-diffusion extension to dental arch camera intrinsics, 10-20 lines of PyTorch, 1-2 weeks).

**(d) ★ ADOPT WONDERWORLD'S LAYERED DESIGN (FOREGROUND / BACKGROUND / SKY) AS V0 SUB-TASK 1 V1+ LAYER DECOMPOSITION.** For v0 dental arch interactive design, the *3-layer decomposition* maps naturally to:
- **Foreground layer (L_fg)** = the *teeth* (high-detail, multi-tooth context)
- **Background layer (L_bg)** = the *gum* + *alveolar bone* (low-detail, occluded surfaces)
- **Sky layer (L_sky)** = the *opposing arch* (occluded by the foreground, the "sky" relative to the prep tooth)

The layer-level inpainting is the *killer* for *occlusion handling* — the back side of the prep tooth, the distal interproximal surfaces, the sub-gingival margin — all are *occluded* in the input intra-oral scan and must be *inpainted* in the layer-level generation.

**(e) ★ ADOPT WONDERWORLD'S LLM SCENE DESCRIPTION (g_LLM) AS V0 SUB-TASK 1 V1+ PATIENT-PROMPT-INTERPRETER.** The LLM that decomposes the user prompt U + instruction J into {F, B, S} (foreground object prompt, background prompt, style prompt) is the *killer* for *patient-specific* generation. For v0 dental arch, the LLM extracts:
- **F** = tooth type (premolar, molar, canine, incisor) + preparation type (full crown, inlay, onlay, veneer) + material (zirconia, lithium disilicate, gold, PFM)
- **B** = gum condition (healthy, receded, inflamed) + bone level (normal, atrophied, augmented)
- **S** = occlusion (Angle Class I, II, III) + bite pattern (normal, deep, cross, open)

**Implementation cost:** $20-50 Lambda (GPT-4o-mini API for LLM decomposition + 100-200 example prompts for the dental prompt-decomposition prompt, 1 week).

**(f) ADOPT WONDERWORLD'S NO-DENSIFICATION + 100-ADAM-ITERATIONS AS V0 SUB-TASK 1 V1+ INFERENCE GUARANTEE.** The deterministic-convergence + predictable-inference-time claim is the *killer* for clinical UX. v0 dental arch interactive design needs *guaranteed* <10s per scene for clinical chairside use, and WonderWorld's no-densification + 100-iterations claim provides that guarantee.

**(g) CITE WONDERWORLD IN V0 PAPER §2 (RELATED WORK) + §3 (METHOD) + §4 (EVALUATION).** WonderWorld is the *de facto 2024-2025 reference* for interactive 3D scene generation, with the *de facto evaluation suite* (28 scenes × 5 metrics CS/CC/CIQA+/Q-Align/CA + 2AFC human study). For v0 dental arch paper:
- §2: cite as the *interactive 3D scene generation* baseline, parallel to our *interactive dental arch design* paradigm
- §3: cite FLAGS + guided depth diffusion as the *killer mechanisms* for our v0 sub-task 1
- §4: cite the 5-metric evaluation suite + 2AFC human study as the *de facto 2024-2025 evaluation protocol* for interactive 3D scene generation

**(h) FRAME WONDERWORLD'S LIMITATIONS AS V0 PAPER'S OPPORTUNITIES.** WonderWorld's §5 Limitations are the *killer* framing for v0 paper:
- "**frontal-facing surfaces only**" → v0 sub-task 1 dental arch must address the *back side* of the prep tooth (the sub-gingival margin side facing the gingiva)
- "**no detailed object modeling** (trees, floaters)" → v0 sub-task 2 crown generation must use *DMC-style (paper 033) high-fidelity point-to-mesh* for the prep tooth + adjacent teeth
- "**interactive prototyping, not full end-to-end**" → v0 paper's *killer positioning* is: v0 sub-task 1 = WonderWorld-style interactive dental arch prototype, v0 sub-task 2 = DMC-style high-fidelity crown generation, v0 sub-task 4 = WonderWorld-style refinement of the *complete* dental arch with the v0 sub-task 2 crown in place

**(i) USE WONDERWORLD'S INTERACTIVE BROWSER DEMO AS V0 PAPER'S CLINICAL DEMO TEMPLATE.** The `index_stream.html` browser-based interactive viewer (WASD + arrow keys + R to generate + Z to undo + X to save) is the *killer* reference implementation for the v0 paper's *clinical interactive demo*. The dentist can navigate the dental arch with WASD, specify a new tooth with R, undo with Z, and save the final arch with X — *exactly* the UX flow needed for clinical chairside use. **Implementation cost:** $0-50 Lambda (fork the index_stream.html + adapt to dental arch scene + add clinical UI elements like prep margin visualization, $0-50).

**Compute estimate:** v0 sub-task 1 V1+ with WonderWorld integration adds:
- Marigold depth/SD Inpaint/OneFormer/RepViT-SAM dental fine-tune: $200-300 Lambda
- Guided depth diffusion dental arch extension: $50-100 Lambda
- LLM scene description GPT-4o-mini API for 1000 test prompts: $10-20 Lambda
- Interactive browser demo fork: $0-50 Lambda
- Total V1+ add: $260-470 Lambda (the *biggest* v0 sub-task 1 V1+ addition so far)

**v0 compute: ~$9,730-12,100 Lambda** (was $9,470-11,630 from 127, +$260-470 for WonderWorld-inspired V1+ mechanisms). *All* under the *reimplementation* framework (WonderWorld is research-only, no commercial license, so v0 will *reimplement* the FLAGS + guided depth diffusion from scratch under MIT/Apache 2.0 — the *open-source* implementation at github.com/KovenYu/WonderWorld is the *reference* but not the *v0 codebase*).

**Strategic positioning:** WonderWorld is the **de facto 2024-2025 reference for interactive 3D scene generation** — the *first* paper to combine (a) layered scene representation, (b) geometry-based initialization with Nyquist-sampling scale derivation, (c) text-guided diffusion inpainting at the layer level, (d) guided depth diffusion for boundary alignment, and (e) interactive camera-move + text-prompt user control. The CVPR 2025 Highlight designation confirms its importance to the field. The *killer application* to dental arch is **interactive dental arch design** — the dentist specifies the *missing tooth position* (camera) and *preparation* (text) and sees the generated full arch in <10s. The *killer follow-up* to WonderWorld for v0 is **WonderTurbo (Ni et al. ICCV 2025, 0.72s/scene, the speed successor)** + **WonderPlay (Yu et al. ICCV 2025, dynamic 3D scene generation)** + **WorldScore (Duan et al. ICCV 2025, unified evaluation benchmark)** — the *complete* Stanford scene-generation dynasty for v0/v1/v2.

WonderWorld + WonderJourney (CVPR 2024, the offline ancestor) + WonderTurbo (ICCV 2025, the speed successor) + WonderPlay (ICCV 2025, the dynamic successor) + WorldScore (ICCV 2025, the eval benchmark) + MVSplat360 (NeurIPS 2024, paper 125, the feed-forward scene 3DGS companion) + Bolt3D (ICCV 2025, paper 116, the 6.25s scene generator) + L4GM (NeurIPS 2024, paper 134, the 4D object extension) = the **complete 2024-2025 interactive 3D/4D scene generation arc**, the *killer* reading list for v0 sub-task 1+ V1+.

**Note in `papers/135-wonderworld-yu25.md`.** **Next paper to read (136):** the *direct* *speed successor* to WonderWorld is **WonderTurbo (Ni et al. ICCV 2025, "WonderTurbo: Generating Interactive 3D World in 0.72 Seconds")** — generates interactive 3D scenes in **0.72s** per scene (vs WonderWorld's 9.5s, **~13× speedup**), the *killer* for real-time clinical chairside UX. After WonderTurbo, the *dynamic* successor is **WonderPlay (Yu et al. ICCV 2025, "WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions")** for v0 sub-task 1 V2 dynamic dental arch (chewing motion, jaw movement). After WonderPlay, the *eval benchmark* is **WorldScore (Duan et al. ICCV 2025, "WorldScore: A Unified Evaluation Benchmark for World Generation")** for v0 paper's eval protocol. **Recommendation: *read 136 = WonderTurbo (Ni et al. ICCV 2025, 0.72s/scene, the speed successor)*** — the *killer* for real-time clinical chairside UX, the *right* 136 to complete the Stanford scene-generation dynasty with the speed-focused paper.
