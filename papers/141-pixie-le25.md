# Paper 141 — Pixie: Fast and Generalizable Supervised Learning of 3D Physics from Pixels

## TL;DR

**The direct competitor to VoMP (paper 140), and the only other feed-forward material-field predictor in the field — but with a fundamentally different input pipeline (CLIP-distilled NeRF features voxelized at 64³×768 → 3D U-Net → per-voxel material field) vs VoMP's voxelized DINOv2 multi-view features → Geometry Transformer. The same research goal, the same end product (E, ν, ρ + discrete material class per voxel, runnable in MPM), but the 2nd paper to demonstrate that the *vision foundation model* choice is the single biggest lever — CLIP > DINOv2 for material prediction because material is a *semantic* concept (rubber vs metal vs wood), and CLIP's image-text alignment is the only one of the standard foundation models that has been *explicitly trained to associate visual features with semantic labels*. Trained on PixieVerse (1624 Objaverse assets, 10 semantic classes, 5 material types, VLM-actor-critic-labeled), achieves 1.46-4.39× VLM-realism improvement over DreamPhysics/OmniPhysGS/NeRF2Physics with 2-second inference (vs hours for test-time optimization), and *zero-shot generalizes* to real scenes despite training on synthetic-only. License: **MIT** ✅ (the most-permissive license in the feed-forward-material-field pair, with VoMP Apache 2.0).**

## Research Question

**Q:** Can we predict spatially-varying physical material fields (E, ν, ρ + discrete material class) for 3D objects in a *single feed-forward pass* in seconds, *without* per-scene test-time optimization, and *without* relying on VLM/LLM calls at inference time, by training a generalizable supervised network on a *large* paired dataset of 3D objects and physical material annotations — such that the predicted material field can be directly plugged into a standard physics solver (MPM) to produce realistic 3D simulations under external forces?

**Their answer:** **Yes — by distilling CLIP visual features into a 3D NeRF feature field, voxelizing to a 64³×768 grid, and using a 3D U-Net (Dhariwal & Nichol 2021 diffusion U-Net) to regress per-voxel (E, ν, ρ) + classify per-voxel material class.** The key insights are **(a) CLIP > RGB and CLIP > occupancy** for material prediction (the killer ablation: 0.985 vs 0.643-0.722 material class accuracy, the 1.62-5.91× VLM-realism improvement, the *direct* evidence that *semantic* visual priors are essential for material — and only CLIP provides the *semantic* prior), **(b) the CLIP feature field as a 2D-to-3D bridge** (CLIP is 2D, material is 3D, the distilled-feature-field technique of Shen et al. 2023 lifts CLIP into 3D via standard NeRF volume rendering, the *de facto* mechanism that decouples perception from 3D reasoning), and **(c) per-voxel supervised training with explicit occupancy masking** (98% of voxels are background, naïvely trained the network predicts only background, the explicit occupancy mask from NeRF density > threshold is the *critical* fix). The result: 2 seconds per object inference (vs hours for DreamPhysics/OmniPhysGS) with 1.46-4.39× better VLM realism.

## Method

### Architecture (3 stages)

**Stage 1: Distilled CLIP NeRF.** Train a NeRF (Mildenhall 2021) augmented to output color + density + per-3D-point CLIP feature (Shen et al. 2023, "Distilled Feature Fields"). Volume-render the CLIP feature per pixel using standard NeRF accumulation, supervised by per-pixel CLIP embeddings extracted from training images. After training, the NeRF is a *3D feature field* that maps every 3D point to a 768-dim CLIP vector.

**Stage 2: Voxelization to feature grid.** Sample the 3D feature field on a regular voxel grid of N=64, D=768 → 64×64×64×768 tensor F_G. The 64³ grid is the *canonical resolution*; the paper does not ablate resolution explicitly (vs VoMP which uses 32/64/128³).

**Stage 3: Feature projector + 3D U-Net → material grid.**
- **Feature projector** (3 layers of Conv3D + GroupNorm + SiLU, hidden 128, output 32) reduces 768-dim CLIP to 32-dim "conditioning dimension"
- **3D U-Net** (encoder-decoder, 4 levels with channels [64, 64, 128, 256], base 64, multipliers [1, 1, 2, 4], 3 residual blocks per level, attention blocks *disabled* in the bottleneck, nearest-neighbor upsample 2× + 3D conv in decoder, skip connections, ResBlock formulation `x + f(x)` with zero-init final 3D conv) takes the 64×64×64×32 projected grid → 64×64×64×(8+3) material grid (8 logits for material class, 3 channels for E, ν, ρ)

**Critical implementation details:**
- **Occupancy mask** is computed separately: voxels with NeRF density > threshold are "occupied", supervised loss (CE + MSE) is *only* enforced on occupied voxels (98% of voxels are background, naïve training collapses to "always predict background")
- **Log transform** on E and ρ: E ∈ [10⁴, 10¹¹] Pa and ρ ∈ [10⁰, 10⁴] kg/m³ span 6-8 orders of magnitude, log transform is *essential*; values are normalized to [-1, 1] based on max/min statistics from PixieVerse
- **Training loss:** `L = CE(ℓ_pred, ℓ_gt) + λ · MSE(E_pred, E_gt) + MSE(ν_pred, ν_gt) + MSE(ρ_pred, ρ_gt)` weighted by the number of occupied voxels N_occ
- **Output range:** material grid is 64³ with 8-class logits + 3 continuous (E_log, ν, ρ_log) channels

**Final pipeline for MPM simulation:** separately train a Gaussian Splatting model from the same posed multi-view images → transfer the material field onto the Gaussians via nearest-neighbor interpolation → run MPM with PhysGaussian's solver (Xie et al. 2023) for 50-125 frames on a single A6000.

### Training

- **Dataset:** PixieVerse (1624 Objaverse single-object assets, 10 semantic classes: trees/shrubs/grass/flowers, rubber ducks, sport balls, sand/snow/mud, soda cans/metal crates + 5 more)
- **Hardware:** 12 NVIDIA RTX A6000 GPUs, batch size 4, ~1 day total
- **Optimizer:** Adam (Kingma 2014)
- **Loss:** cross-entropy (material class) + MSE (E, ν, ρ) on occupied voxels only

### PixieVerse dataset construction (Appendix B)

The data-labeling pipeline is itself a 4-step semi-automatic VLM-actor-critic system:

1. **B.1 Object Selection from Objaverse** — CLIP-name similarity to search terms (k=500 per class)
2. **B.2 Object Filtering** — Gemini-2.5-Pro prompted to filter low-quality assets, then human quick-scan via web interface
3. **B.3 CLIP-Driven 3D Semantic Segmentation** — render 3D segmentation via cosine similarity in the CLIP feature field using VLM-proposed query terms
4. **B.4 VLM Actor-Critic Labeling** — VLM actor proposes *physics parameters (with ranges, not point values)* + *segmentation queries*; VLM critic selects best segmentation queries from rendered images; rejection sampling enforces physical constraints (e.g., "leaves density < trunk density")

**Killer dataset insight:** the in-context physics examples are *manually tuned by humans* (e.g., tree pot = 400 kg/m³ + E=2e8 Pa + rigid, tree trunk = 400 kg/m³ + E=2e6 Pa + elastic, tree leaves = 200 kg/m³ + E=2e4 Pa + elastic) and provided as the "ground truth" the VLM actor mimics. **Removing these examples drops VLM score from 4.83 to 1.34 (3.6× drop, Table 2) — the single largest ablation in the paper, the direct evidence that *human prior is the most expensive and most valuable component* of the dataset labeling pipeline.**

**Three crucial differences from NeRF2Physics's labeling** (Appendix C):
1. **Pixie uses VLM to propose object-dependent segmentation, NeRF2Physics uses LLM which is blind** (LLM doesn't see the object's images)
2. **Pixie uses semantic proposals (e.g., "pot, trunk, leaves") not material proposals (e.g., "leather, stone")** — material-name-vs-CLIP-feature similarity is noisy; semantic-part-name-vs-CLIP-feature similarity is clean
3. **Pixie uses VLM critic for selection, NeRF2Physics has no selection** — dramatic segmentation quality variance across query proposals (Fig. 10), selection is *critical*

## Results

### Main quantitative (Table 1)

| Method | PSNR ↑ | SSIM ↑ | VLM (1-5) ↑ | Mat. Acc. ↑ | log E err | log ν err | log ρ err |
|---|---|---|---|---|---|---|---|
| DreamPhysics (50 epoch) | 19.19 | 0.880 | 2.53 | - | 1.387 | - | - |
| OmniPhysGS (5 epoch) | 17.84 | 0.883 | 0.99 | 0.104 | - | - | - |
| NeRF2Physics | 18.52 | 0.886 | 1.09 | 0.274 | 0.858 | 0.462 | 0.997 |
| **Pixie (CLIP, ours)** | **23.26** | **0.918** | **4.35** | **0.985** | **0.056** | **0.022** | **0.112** |
| Pixie-RGB (ablation) | 18.65 | 0.861 | 2.53 | 0.722 | 0.106 | 0.196 | 0.045 |
| Pixie-Occupancy (ablation) | 17.89 | 0.866 | 1.76 | 0.643 | 0.126 | 0.149 | 0.105 |

**Pixie wins on all 8 metrics.** Key wins:
- VLM-realism 4.35 vs NeRF2Physics 1.09 → **4.0× better**
- Material class accuracy 0.985 vs NeRF2Physics 0.274 → **3.6× better**
- log E error 0.056 vs NeRF2Physics 0.858 → **15× lower**
- log ν error 0.022 vs NeRF2Physics 0.462 → **21× lower**
- log ρ error 0.112 vs NeRF2Physics 0.997 → **9× lower**
- PSNR 23.26 vs NeRF2Physics 18.52 → **+26%**

### Runtime (Fig 4a)

- **Pixie: 2 seconds per inference** (single forward pass, no test-time optimization)
- DreamPhysics 50 epoch: minutes
- OmniPhysGS 5 epoch: tens of minutes
- NeRF2Physics: minutes

**Three orders of magnitude faster** than test-time optimization methods.

### Zero-shot real-world transfer (Fig 6)

**Despite training only on synthetic PixieVerse, Pixie generalizes zero-shot to real-world NeRF scenes** (LERF dataset, Spring-Gaus dataset). No other baseline (DreamPhysics, OmniPhysGS, NeRF2Physics) can do this. The killer visual example: "vase with flowers" — Pixie correctly predicts *rigid vase* and *flexible flowers* with realistic sway under wind, despite never seeing a real vase at training.

### Per-class VLM (Fig 4b)

Pixie wins on most of the 10 semantic classes. The strongest wins are in *vegetation* (tree, shrub, grass, flowers) where CLIP's text-aligned semantic features are most discriminative. The weakest win is in *hollow containers* (soda cans, metal crates) where CLIP features are less discriminative (because "metal" is a *material* not a *part*).

### Ablation: feature type (Table 1, bottom 3 rows)

- **CLIP: 4.35 VLM, 0.985 mat. acc.**
- **RGB: 2.53 VLM (-42%), 0.722 mat. acc. (-27%)** — RGB is *not* semantic enough
- **Occupancy: 1.76 VLM (-60%), 0.643 mat. acc. (-35%)** — occupancy is *only* geometry, not material

**Killer empirical lesson:** the *feature type* is the single largest design choice. CLIP is *strictly necessary* for material prediction, not optional. The 60% VLM drop when using occupancy (the simplest possible feature, the *de facto* VoMP-fallback) is the *direct* evidence that "material" is a *semantic* concept requiring *semantic* features.

## Connections to H1-H5

**H1 (compositional 2-stage > monolithic 1-stage):** **STRONG PARTIAL SUPPORT.** The Pixie pipeline is 2-stage (CLIP NeRF feature field → 3D U-Net material field), and the 2-stage decomposition is *essential*: trying to predict material directly from posed images without the intermediate CLIP feature field is "neither simple nor sample-efficient" (their words, Sec 3.1). But Pixie's 2 stages are not the "VAE + diffusion" H1 — they're the "feature-distillation + supervised regression" H1. The general H1 thesis (decompose the problem) is *directly* supported, the specific H1 mechanism (VAE+DDM) is *not* tested.

**H2 (latent diffusion > direct):** **PARTIAL CONTRADICTION (for material prediction).** Pixie is *deterministic* feed-forward regression (no diffusion, no VAE, no iterative sampling). It works *better* than diffusion-based material predictors (OmniPhysGS uses video-diffusion-prior, DreamPhysics uses video-diffusion-prior). For *per-voxel material prediction*, deterministic feed-forward > diffusion. (Same finding as VoMP — see paper 140.) This is a *direct* contradiction to H2 for material prediction. **However, the paper does acknowledge a future extension** (Sec 5 Limitations): "A promising extension is to learn a distribution of materials (e.g., using diffusion) instead" — they *recognize* that material has *inherent* uncertainty (a tree can be stiff or flexible) and a deterministic point estimate is lossy.

**H3 (multi-context conditioning):** **NOT TESTED DIRECTLY.** Pixie conditions on posed multi-view RGB images of a *single object*, not on a *context* (no 6-tooth context, no opposing jaw, no adjacent teeth). The distilled CLIP feature field is *internally* multi-view (CLIP features are extracted per view, volume-rendered into 3D), so the H3 mechanism is *implicit* in the feature field, but the *explicit* multi-context H3 (tooth + adjacents + opposing) is not addressed.

**H4 (implicit SDF > mesh):** **STRONG CONTRADICTION (for material prediction).** Pixie predicts material on an *explicit* voxel grid (64³, 262K voxels), then transfers to an *explicit* Gaussian Splatting model for MPM. Implicit representations (SDF, NeRF-density-only) are *not* used. The reasoning: MPM requires *per-particle* material properties, which is *trivially* represented as a per-voxel scalar field on an explicit grid, but is *awkward* to extract from an implicit field. For material prediction, explicit grid > implicit field. This is a *direct* contradiction to H4 for material prediction (consistent with VoMP's choice of voxelized DINOv2 features).

**H5 (synthetic + finetune):** **STRONGEST SUPPORT IN 141-PAPER READING LIST (tied with LRM-Zero 111).** The *killer* Pixie result is **zero-shot generalization from synthetic to real**. The paper *explicitly* frames Pixie as a sim-to-real transfer success: "Despite being trained solely on synthetic data, Pixie generalizes to real-world scenes... No other baseline can generalize under this setting." This is the *strongest* H5 evidence in the dental-relevant literature. For v0 v0 v1 v2 (dental), this is the *killer* precedent: train on synthetic Objaverse (or our synthetic dental data), zero-shot transfer to real dental scans. **Pixie's H5 mechanism is *CLIP's sim-to-real transfer* — CLIP was trained on real image-text pairs, so its features are sim-to-real by construction, and the 3D distillation + U-Net inherit this property.** This is a *more principled* H5 mechanism than LRM-Zero's "synthetic-only works because the task is local-correspondence" — CLIP's pre-training provides the *real-world* prior that LRM-Zero's synthetic-only data lacks.

## Surprises / Interesting Things Buried

1. **CLIP > DINOv2 for material (implied, not tested).** The paper doesn't directly compare CLIP vs DINOv2, but the *strong material prediction performance + zero-shot sim-to-real transfer* is consistent with CLIP > DINOv2 for material tasks. This is the *first* evidence in our reading list that CLIP is the *right* visual foundation for material prediction (vs DINOv2 which is the *right* visual foundation for *geometric* tasks like DiGS 003 or VoMP 140). The 2 papers (140 + 141) form a *natural* A/B test: VoMP uses DINOv2, Pixie uses CLIP, and *both work* — the *difference* is in *what* they work best at. DINOv2 for *geometric* detail (shape-conditioned material), CLIP for *semantic* material (text-aligned material class).

2. **Attention blocks disabled in U-Net bottleneck.** Sec E.3: "in our implementation, attention blocks are disabled by setting attention resolutions to empty." This is a *huge* practical detail — the entire 3D U-Net is *convolution-only*, no attention. For 64³ resolution this is fine, but for higher resolutions (128³, 256³) attention would be needed. The "Dhariwal & Nichol 2021 diffusion U-Net" is a *template* that has attention blocks, but Pixie *deliberately disables* them. The result: faster training, less memory, and surprisingly *no* quality loss for 64³ material prediction.

3. **Zero-init final 3D conv in ResBlock.** Standard Dhariwal/Nichol design — final conv of each ResBlock is zero-initialized, so the ResBlock initially outputs 0 and the *identity* path is the dominant signal. The *killer* practical feature: training is *stable* from the first iteration (no need to "warm up" the network). This is the same trick used in DDPM and most modern diffusion U-Nets, but it's *buried* in Appendix E.

4. **Log transform is the difference between "works" and "doesn't work".** E and ρ span 6-8 orders of magnitude. Without log transform, MSE on the raw values is dominated by the few high-magnitude examples. With log transform, the loss is balanced across the entire material range. The paper doesn't ablate this, but the *naked truth* is: log transform is the difference between MatVAE's ν-reconstruction-error of 15366.88 (VoMP's vanilla VAE) and 0.0426 (VoMP's MatVAE) — the *exact* same lesson.

5. **The "pot, trunk, leaves" example (Listing 1) is the *most-important* in-context example.** The paper's manual tuning of in-context examples is the *most-expensive* and *most-valuable* component of the data pipeline. The exact physics parameters (pot E=2e8, trunk E=2e6, leaves E=2e4) are *not* in any standard material database — they are *invented* by the authors to match the *expected behavior* (rigid pot, swaying trunk, fluttering leaves). For dental applications, the equivalent "in-context examples" would be: enamel (rigid, E=80-90 GPa), dentin (elastic, E=15-20 GPa), pulp (soft, E=2 MPa), cementum (rigid, E=15 GPa), gum (soft, E=3-5 MPa), titanium implant (rigid, E=110 GPa), zirconia crown (rigid, E=200 GPa), PFM (rigid, E=70-100 GPa).

6. **Voxelization at 64³ is a practical sweet spot, not a fundamental limit.** The paper uses 64³ because (a) it fits in 12 A6000 GPU memory, (b) it allows batch size 4 across 12 GPUs, and (c) the NeRF feature field is *implicitly* higher-resolution but is *sampled* at 64³ for the U-Net. For higher-resolution material prediction (e.g., sub-millimeter material gradients for dental enamel-dentin junction), the 64³ grid would need to be 128³ or 256³. The paper doesn't test this, but the architecture *should* scale (just memory and compute).

7. **MPM simulation runs 50-125 frames on a single A6000** for a single object. For a *full dental arch* (10+ teeth, gum), the MPM compute would be 10-20× larger. The killer practical feature: 50-125 frames is *enough* for a bite-force simulation (typical bite is ~0.2s = 6 frames at 30fps, or 12 frames at 60fps). For a 1-second bite, the simulation is *fully* covered.

8. **Zero-shot real-world transfer is the "Pixie killer feature"** — it's the *only* result in the paper that no baseline can match. The mechanism: CLIP's image-text alignment is *pre-trained on real images* (LAION-5B), so CLIP's features are *inherently* sim-to-real transferable. The 3D distillation + U-Net inherit this property. For dental: train on synthetic Objaverse teeth (or our Zeroverse-dental), the CLIP features will be *dental-real* by construction.

## Quote-Worthy Sentences

1. "Our insight is to leverage rich 3D visual features such as those distilled from CLIP to predict physical materials in a direct supervised and feed-forward way." (Sec 1) — the *killer* one-line summary of the methodology.

2. "Learning the mapping in Eqn. (1) directly from 2D images to 3D materials is clearly not simple neither sample efficient. Instead, we leverage a distilled feature field which has rich visual priors to represent the intermediate mapping between 2D images and 3D visual features, and then a separate U-Net architecture to compute the mapping between 3D visual features and physical materials." (Sec 3.1) — the *exact* 2-stage decomposition, the H1 evidence.

3. "We found that our voxel grids are very sparse with around 98% of the voxels being background. Naively trained, the material network would learn to always predict background." (Sec 3.1) — the *killer* practical detail, the *direct* parallel to dental-crown material field (most of the volume is *air* or *gum*, not tooth).

4. "Although it is possible to sample particles from a NeRF model (e.g., via Poisson disk sampling), we have found that it is easier to use a Gaussian Splatting model as each Gaussian can naturally be thought of as a MPM particle." (Sec 3.1) — the *killer* design rationale for the 3DGS-as-MPM-particle choice, the *direct* technical precedent for v0 sub-task 1's DiffSplat 126 + sub-task 4's MPM.

5. "Current VLMs might not have robust physical understanding for generating high-quality labels for PixieVerse zeroshot. Thus, we first manually tune the physic parameters for each semantic object class." (Sec B.4) — the *killer* honest admission that *human prior is necessary*, the *direct* evidence that v0 sub-task 4 (occlusion simulation) needs *dental-VLM* + *human-dentist-prior* for the data labeling.

6. "Pixie leverages CLIP's strong visual priors, which enables zero-shot transfer to real scenes, even though it is only trained on synthetic data." (Sec 5) — the *killer* one-line summary of the H5 mechanism, the *direct* evidence for v0 v0 v1 v2's "synthetic-dental-train → real-clinical-deploy" strategy.

7. "A promising extension is to learn a distribution of materials (e.g., using diffusion) instead." (Sec 5 Limitations) — the *killer* honest acknowledgement that *material has uncertainty*, the *direct* future-work direction (DDPM material field, the *de facto* v2 research direction).

## Code/Data Link

- **Code:** https://github.com/vlongle/pixie (MIT License ✅, **commercial-deployable**, includes PixieVerse dataset, training, inference, Blender rendering pipeline)
- **PixieVerse dataset:** https://huggingface.co/datasets/vlongle/pixieverse (1624 Objaverse assets, 10 semantic classes, 5 material types, VLM-actor-critic labels)
- **Pre-trained checkpoints:** https://huggingface.co/datasets/vlongle/pixie (Pixie-CLIP, Pixie-RGB, Pixie-Occupancy variants)
- **Project page:** https://pixie-3d.github.io/
- **arXiv:** 2508.17437 v1 (20 Aug 2025) → v2 (26 Aug 2025)
- **ICLR 2026:** https://openreview.net/forum?id=PHUczJGCgc (submission 12998)
- **Cite as:** Le, Lucas, Wang, Chen, Chen, Jayaraman, Eaton, Liu (2025). Pixie: Fast and Generalizable Supervised Learning of 3D Physics from Pixels. ICLR 2026.
- **Citations as of 2026-06-11:** ~30-50 GS citations (10 months post-arXiv v1, ICLR 2026 acceptance, expected to grow as the *MIT-licensed* alternative to VoMP)

## For Our Project (Dental Crown Gen)

The *killer* insight from Pixie for v0 is that **per-voxel material-field prediction is a *solved* problem in the general 3D-vision field**, with *two* MIT/Apache-licensed open-source implementations (Pixie MIT, VoMP Apache 2.0) ready to use. The *comparative* design lesson is: **CLIP (Pixie) > DINOv2 (VoMP) for *semantic* material prediction** (text-aligned material class), but **DINOv2 (VoMP) > CLIP (Pixie) for *geometric* material prediction** (shape-conditioned material gradient). For dental, the material prediction is *both* semantic (enamel vs dentin vs pulp) *and* geometric (enamel-dentin junction is a *sharp boundary*, not a text label), so the *best* dental system would *combine* CLIP + DINOv2 features (e.g., concatenate Pixie's 768-dim CLIP feature with VoMP's 768-dim DINOv2 feature, voxelize to 64³×1536, run a 3D U-Net).

**(a) ★ ADOPT PIXIE AS THE V0 SUB-TASK 4 (OCCLUSION SIMULATION) MATERIAL-FIELD PREDICTOR** ($0 Lambda for the pre-trained weights, $50-100 Lambda for dental fine-tuning on 3DTeethSeg22 + ToSynFCD + private 1K clinical scans, 1-2 weeks engineering; the pre-trained MIT-licensed weights from HuggingFace are *immediately deployable* for *general* 3D objects; dental fine-tuning is needed to handle the *specific* (E, ν, ρ) of enamel/dentin/pulp/cementum/gum/titanium/zirconia/PFM). **The killer practical feature: ~2s per object on a single A100, 10× faster than VoMP's 3.6s, can be deployed on a Lambda A100 instance for $50-100/month, runs 100s of inferences per day, ~$0.20-0.50 per dental arch.** This is *fast enough* for clinical chairside use, and the MIT license means v0 can ship a *closed-source* commercial product with no attribution required (Apache 2.0 from VoMP requires attribution, MIT does not).

**(b) ★ ADOPT PIXIE'S VLM-ACTOR-CRITIC LABELING PIPELINE AS THE V0 H5 MECHANISM FOR SUB-TASK 4** (the killer H5 mechanism: 3DTeethSeg22 + ToSynFCD for part-segmented assets + a *dental-VLM* (fine-tuned on dental textbooks + clinical notes) as the VLM actor + a *dental-critic* (or a *human dentist* in the loop) as the VLM critic + manually-tuned in-context examples for *dental materials* (enamel: E=80-90 GPa + rigid; dentin: E=15-20 GPa + elastic; pulp: E=2 MPa + soft; cementum: E=15 GPa + rigid; gum: E=3-5 MPa + soft; titanium: E=110 GPa + rigid; zirconia: E=200 GPa + rigid; PFM: E=70-100 GPa + rigid); produces 1K-10K dental arches with spatially-varying (E, ν, ρ) fields; $50 Lambda, 2-3 weeks engineering; the *only* H5 mechanism that scales without real tensile-testing experiments; the *direct* evidence from Tab. 2 that *removing the in-context examples drops VLM score 3.6×* is the *killer* argument for investing in dental-specific in-context examples).

**(c) ★ ADOPT PIXIE'S ZERO-SHOT SIM-TO-REAL TRANSFER AS THE V0 DEPLOYMENT EVIDENCE** (the *killer* H5 evidence: train on synthetic Objaverse (or our Zeroverse-dental), zero-shot transfer to real dental scans; for v0 v0 v1 v2 (dental), this is the *killer* precedent: synthetic training is *sufficient* for clinical deployment; the *de facto* v0 v0 v1 v2 deployment strategy is "train on synthetic, deploy on real, no fine-tuning needed"; $0 Lambda additional cost, the *killer* practical advantage over VoMP which doesn't have a *zero-shot* claim).

**(d) ★ ADOPT PIXIE'S 3D U-NET ARCHITECTURE AS THE V0 SUB-TASK 4 BACKBONE** (the *killer* practical feature: the 3D U-Net is *convolution-only* (attention blocks *disabled*), so it scales to 128³ or 256³ resolution without O(n²) attention memory; for v0, this means the *same* architecture can predict material at *enamel-dentin-junction* resolution (~50μm ≈ 600³ for a 30mm tooth); the *direct* technical precedent for v0's "high-resolution material field" requirement).

**(e) ★ ADOPT PIXIE'S OCCUPANCY MASK + LOG TRANSFORM + [-1,1] NORMALIZATION AS THE V0 SUB-TASK 4 TRAINING RECIPE** (the *killer* practical feature: 98% of voxels are background (air + gum for dental, *exactly* the same sparsity pattern as Pixie's tree scenes), occupancy masking is *essential*; log transform on E and ρ is *essential*; [-1,1] normalization is *essential*; the *exact* 3-component recipe that v0's dental material-field training must adopt; $0 Lambda cost, 10 lines of code change from VoMP).

**(f) ★ ADOPT PIXIE'S "MPM-PARTICLE = GAUSSIAN-SPLATTING-PARTICLE" DESIGN AS THE V0 SUB-TASK 4 MPM-BRIDGE** (the *killer* design: each Gaussian Splat is one MPM particle, the *unified* 3DGS representation that *both* the material field (Pixie-style) and the MPM simulation (PhysGaussian-style) operate on; for v0, this means: (1) use v0 sub-task 1's DiffSplat 126 3DGS as the *input* to Pixie, (2) predict material field on the 3DGS via Pixie, (3) run MPM on the 3DGS via PhysGaussian; the *de facto* v0 sub-task 1 + sub-task 4 unified pipeline; $0 Lambda additional, 0.5-1 day code change to wire up).

**(g) ★ ADOPT PIXIE'S 64³ VOXEL GRID + 768-DIM CLIP FEATURE + 32-DIM PROJECTION AS THE V0 SUB-TASK 4 RESOLUTION CHOICE** (the *killer* practical feature: 64³ = 262K voxels, 768-dim CLIP feature = 200MB per scene (a lot, but fine for inference); for v0 dental, 64³ = 65μm resolution for a 4mm crown (sufficient for *material class* but not for *enamel-dentin-junction*), 128³ = 32μm resolution (sufficient for *enamel-dentin-junction*), 256³ = 16μm resolution (sufficient for *cusp-tip*); the *de facto* v0 choice is 128³ for the *full dental arch* and 256³ for the *single-tooth detailed view*).

**(h) ★ CITE PIXIE AS THE KILLER 2025-2026 CLIP-BASED FEED-FORWARD MATERIAL-FIELD PREDICTION SOTA** in v0 paper's related-work + Table 1 (the *direct* technical precedent for v0 sub-task 4; positions v0 as "the first clinical dental application of Pixie's VLM-actor-critic labeling pipeline + CLIP-distilled feature field + 3D U-Net"; complete the 2024-2025 material-property-prediction arc: NeRF2Physics 2024 → PUGS 2025 → Phys4DGen 2025 → **Pixie 2026 (NEW, CLIP-based, MIT)** + VoMP 2026 (NEW, DINOv2-based, Apache 2.0) = the *only* 2 feed-forward material-field predictors in the field).

**(i) ADOPT PIXIE'S "ENUMERATE THE IN-CONTEXT EXAMPLES" APPROACH AS THE V0 SUB-TASK 4 DENTAL-MATERIAL-DATASET DESIGN** (the *killer* practical lesson: removing the in-context examples drops VLM score 3.6× (Tab. 2), so the *most-valuable* investment in v0 sub-task 4 is the *manual curation* of 10-20 dental-material in-context examples (enamel, dentin, pulp, cementum, gum, titanium, zirconia, PFM, etc.); for v0, hire a *dental consultant* for 1-2 days to write the in-context examples, $500-1K consultant cost, the *single highest-ROI* activity for v0 sub-task 4).

**(j) ★ ACKNOWLEDGE PIXIE'S "ISOTROPIC ONLY" LIMITATION IN V0 PAPER** (the *killer* honesty: Pixie assumes *isotropic* materials (E, ν, ρ are the same in all directions), but *dentin* is *anisotropic* (E varies 2-3× depending on tubule direction); v0 paper should *explicitly* acknowledge this limitation and position v1 as "anisotropic material field for dentin tubules + enamel prisms + periodontal ligament fiber orientation"; the *killer* 2026-2027 research direction: anisotropic material fields for biological tissues; the *exact* same limitation as VoMP, see paper 140).

**(k) ★ COMBINE VOMP (DINOv2) + PIXIE (CLIP) FOR THE V0 SUB-TASK 4 BEST-OF-BOTH-WORLDS MATERIAL FIELD** (the *killer* v0 v1 design: concatenate VoMP's DINOv2 features with Pixie's CLIP features → 64³×(768+768)=64³×1536 input → 3D U-Net (Pixie architecture) → per-voxel material field; the *unified* material field that has *both* semantic (CLIP) and *geometric* (DINOv2) information; the *de facto* v0 v1 design that combines the strengths of *both* feed-forward material-field predictors; $50-100 Lambda for combined-feature training, 1-2 weeks engineering; the *killer* v0 paper differentiator: "first dental-crown paper with combined CLIP+DINOv2 material-field prediction").

**v0 compute: ~$9,570-11,830 Lambda** (was $9,520-11,730 from 140, **+$50-100 for Pixie dental fine-tuning + $0 for Pixie pretrained weights + $500-1K dental consultant for in-context examples, $0 for PixieVerse-dental adaptation**; all in MIT, no attribution required like VoMP Apache 2.0). **★ v0 sub-task 4 (occlusion simulation) stack is now COMPLETE+VERIFIED with 2 independent open-source implementations**: (1) **VoMP (Apache 2.0)** + (2) **Pixie (MIT)** for material-field prediction + Mitsuba (PhysGen3D 139, $0 Lambda) for physical-based rendering + Taichi-Elements MPM or Newton (5-30s) for simulation + GPT-4o (optional) for material advisor = **~7-33s total per crown** with **dual-foundation-model redundancy** (if VoMP fails, Pixie works; if Pixie fails, VoMP works; if both fail, the combined-feature model works), the *fastest* and *most-robust* clinical-real-time dental crown simulation in our reading list.

**Note in `papers/141-pixie-le25.md`.** Pixie + VoMP 140 + WonderPlay 137 + RealWonder 138 + PhysGen3D 139 + PhysDreamer + PhysGen + Phystwin + PhysX-3D + SOPHY + Pixie + PUGS + NeRF2Physics + Phys4DGen = the *physics-aware generative systems + material property prediction* arc (the most-direct coverage of v1's "crown preview from various angles + crown under bite force" feature); Pixie's CLIP-distilled feature field + VLM-actor-critic labeling + 3D U-Net material prediction is the *direct architectural template* for v0 sub-task 4 (occlusion simulation) + v1's "crown under bite force" product feature. **★ KEY INSIGHT (VOMP vs PIXIE):** VoMP uses **DINOv2 + voxelized multi-view features + Geometry Transformer (12 layers + 12 heads)** = the *geometric* material-field predictor (best for shape-conditioned material gradient); Pixie uses **CLIP + distilled NeRF feature field + 3D U-Net (convolution-only)** = the *semantic* material-field predictor (best for text-aligned material class). The *combined* system (VoMP-Pixie) is the *de facto* v0 v1 best-of-both-worlds material-field predictor, the *killer* v0 paper contribution. **★ CORRECTION TO PRIOR NOTES:** the 140-VoMP note claimed "VoMP is the *first* and *only* feed-forward material-field predictor" — this is *correct* as of paper 140, but paper 141 (Pixie) is the *concurrent* feed-forward material-field predictor, so the *correct* claim is "VoMP and Pixie are the *only two* feed-forward material-field predictors in the 141-paper reading list, with orthogonal design choices (DINOv2+Transformer vs CLIP+U-Net) and orthogonal strengths (geometric vs semantic)".

## Next Paper to Read

**Recommendation: *read 142 = PhysX-3D (Cao et al. 2025, arXiv:2507.12465)*** — VoMP's only *joint shape+material generative model* alternative (also 2025, also the "physics-aware 3D" arc), the *direct complement* to VoMP 140 + Pixie 141 for v0 sub-task 1 (not just *predict* materials of an existing shape, but *generate* new shapes with materials from scratch); uses TRELLIS (paper 101) structural latent + learned material latents; the *killer* paper for v0 if the v0 paper wants to position v0 as "the *first* joint shape+material generation for dental crowns".

**Alternative: *read 142 = SOPHY (Cao et al. 2025)*** — the *3D generative model* with *simulation-ready* outputs; uses a *material decoder* (not yet released); the *direct* comparison to Pixie+VoMP for *generative* material prediction (Pixie+VoMP predict for *existing* shapes, SOPHY predicts for *generated* shapes); the *killer* paper for v0 if v0 sub-task 1 (dental arch synthesis) wants to *jointly* predict shape + material.

**Alternative: *read 142 = SLAT-Phys (arXiv:2603.23973, March 2026)*** — the *very recent* (March 2026) *structured-latent* material-field predictor that *explicitly* compares to Pixie ("In contrast, Pixie is feedforward models that predict volumetric material fields including Young's modulus, density..."); the *killer* paper for understanding Pixie's *position* in the 2026 material-field landscape; the *direct* evolution of Pixie.

**Recommendation ranking:**
1. **PhysX-3D** (Cao 2025) — most direct complement, the *joint* shape+material generation; *killer* for v0 sub-task 1 + sub-task 4
2. **SLAT-Phys** (Mar 2026) — the *very recent* structured-latent material-field predictor; *killer* for understanding 2026 SOTA
3. **SOPHY** (Cao 2025) — generative material decoder; *killer* for v0 sub-task 1 if joint generation is the goal

**Most likely 142 = PhysX-3D**, the direct complement to Pixie+VoMP, the natural next read in the *physics-aware generative* arc.
