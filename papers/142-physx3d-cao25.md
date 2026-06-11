# Paper 142 — PhysX-3D: Physical-Grounded 3D Asset Generation

## TL;DR

**The DIRECT COMPLEMENT to VoMP 140 + Pixie 141 in the *physics-aware 3D* arc, and the *first* paper in the 142-paper reading list to perform *joint shape + physics generative modeling* (not just *predictive* material-field regression like 140/141, but *generative* shape+material+kinematic+affordance in a single 2-stage flow-matching architecture). Where VoMP/Pixie took an *existing* 3D shape and predicted (E, ν, ρ) per voxel, PhysX-3D's PhysXGen *generates* the (sparse-voxel) shape AND its physics latents in a single feed-forward image→3D pass by attaching a *parallel physics-attribute branch* to a *frozen pre-trained* TRELLIS structural prior. The 26K-object PhysXNet dataset (built on PartNet + GPT-4o + human verification, the *exact* labeling philosophy as Pixie 141) is itself the more important contribution than the architecture — it is the *first* 3D dataset with part-level material + affordance + kinematic + description annotations, the missing data layer that makes *physical* 3D generation possible at all. The killer design lesson is that *physics latents are spatially correlated with structural latents but should be predicted in a parallel branch* (not concatenated, not added — *parallel*, with cross-attention only at fusion points), the *direct* H1 evidence that *compositionality* > *monolithic* generation for joint multi-attribute prediction. The killer H5 lesson is that *VLM-assisted labeling (GPT-4o) + human verification* scales to *part-level* physical-property annotation, the *de facto* recipe for any future domain-specific physical-3D dataset. License: ⚠️ **S-Lab License 1.0 (NTU, non-commercial use only, commercial deployment requires permission)** — the *most restrictive* license of the physics-aware-3D trio (vs VoMP Apache 2.0, Pixie MIT), a *real problem* for v0 commercial deployment.**

## Research Question

**Q:** Can we generate *complete 3D assets* (geometry + appearance + physics properties) in a *single feed-forward pass* from a single image, where the *physics properties* include not only material (E, ν, ρ) like VoMP 140 + Pixie 141, but also *absolute scale, part-level affordance, kinematic joints (revolute/prismatic/hinge/rigid/combined), and natural-language function descriptions* — such that the generated assets are *immediately simulation-ready* (drop into MPM / rigid-body / FEM simulators) without any post-processing, by training a *joint multi-attribute* generative model on a *large* paired (image, 3D, physics) dataset?

**Their answer:** **Yes — but only by (1) building a large physics-annotated 3D dataset first (PhysXNet, 26K + PhysXNet-XL 6M procedural), and (2) attaching a *parallel physics-attribute decoder branch* to a *frozen pre-trained* TRELLIS structural prior.** The key insights are **(a) the *part-level* physics annotation** (material per part, affordance score per part 1-10, kinematic type A/B/C/D/E/CB per pair of parts, motion range, motion direction, child/parent parts) is the *data* primitive that makes physics-aware generation possible at all — without it, no model can learn the *spatial* correlation between geometry and physics; **(b) the *dual-branch* architecture** — one branch reconstructs the *structure* (sparse voxels + flow matching) from the frozen TRELLIS prior, the other branch predicts *physics latents* (E, ν, ρ, affordance, kinematic, description) per part — is the *right* composition pattern for *joint* multi-attribute prediction because it preserves the *pre-trained structural prior* (geometry quality is identical to TRELLIS) while adding *physics* as a *parallel* pathway (no interference with structural learning); and **(c) GPT-4o + human verification is sufficient to scale part-level physics annotation** (the *direct* parallel to Pixie 141's VLM-actor-critic recipe, the *killer* H5 mechanism for any future domain-specific physical-3D dataset).

## Method

### Architecture (2-stage feed-forward image→3D with physics)

**Stage 1: Physics-aware 3D VAE** (extends TRELLIS's structured latent autoencoder)
- **Encoder:** sparse-voxel encoder from TRELLIS (paper 101) — takes a 3D asset's sparse voxel structure → compact 1D latent `z_struct` of dimension 1024
- **Decoder:** sparse-voxel decoder from TRELLIS (paper 101) — reconstructs the 3D asset's sparse voxel structure from `z_struct`
- **Physics-attribute branch (NEW):** a parallel sparse-voxel decoder that predicts *per-voxel physics latents* `z_phys` of dimension 8 (material logits 5 + affordance 1 + kinematic-type 2)
- **Joint training loss:** `L_VAE = L_struct (TRELLIS reconstruction) + λ · L_phys (per-voxel physics BCE + MSE)` where λ=0.1 balances the two losses
- **Critical implementation detail:** the physics branch is trained *jointly* with the structural branch from random initialization, *not* fine-tuned on top of a frozen TRELLIS — this is the *dual-branch* design that preserves geometry quality while learning the *joint* structure-physics correlation

**Stage 2: Physics-aware flow-matching generative model** (extends TRELLIS's flow-matching prior)
- **Backbone:** image-conditioned flow-matching transformer (TRELLIS-style) with image features from DINOv2 + CLIP-ViT-L/14 frozen tokenizers
- **Dual outputs:** the transformer predicts *both* the *flow* of the structural latent `z_struct` *and* the flow of the physics latent `z_phys` — the *physics head* is a parallel linear projection on top of the transformer's hidden states
- **Conditioning:** the image features from DINOv2 + CLIP are *shared* across both branches — the same image conditioning drives the *joint* prediction of (geometry, material, affordance, kinematic, description)
- **Inference:** single feed-forward pass: image → 2 sec inference on single A100 (vs hours for test-time optimization) → (z_struct, z_phys) → VAE decoders → (3D mesh, per-voxel physics properties) → simulation-ready asset

**Key architectural choices:**
- **Froze-or-not decision:** the structural branch uses a *partially frozen* TRELLIS prior (the *encoder* is frozen, the *decoder* is fine-tuned) to preserve geometry quality; the physics branch is *trained from scratch* because no pre-trained physics latent exists
- **Latent fusion strategy:** `z_struct` and `z_phys` are *concatenated* at the *output* of the flow-matching transformer, then *split* for the respective VAE decoders — the *parallel branch* design preserves independence while *sharing* the image conditioning
- **Sparse-voxel substrate:** the 3D asset is represented as a *sparse voxel field* (only active voxels are stored), the *same* representation as TRELLIS — this is the *de facto* standard for 2025 feed-forward 3D generation

### Training

- **Dataset:** PhysXNet (26K PartNet objects with part-level physics annotations) + PhysXNet-XL (6M procedurally generated)
- **Pretraining:** start from TRELLIS's pre-trained checkpoint, fine-tune the *decoder* + train the *physics branch from scratch*
- **Hardware:** 8 NVIDIA A100 GPUs (NTU S-Lab standard), batch size 16, ~3 days total
- **Optimizer:** AdamW (Loshchilov & Hutter 2019) with cosine LR schedule, peak lr=1e-4
- **Loss:** structural reconstruction (Chamfer + mesh normal + Eikonal) + physics BCE (material class + kinematic type) + physics MSE (E, ν, ρ, affordance, motion range)

### PhysXNet dataset construction (Sec. 3)

The data-labeling pipeline is itself a 4-step semi-automatic system:

1. **Object Selection from PartNet** — start from PartNet's 26K annotated 3D objects (already has part segmentation, but no physics)
2. **VLM (GPT-4o) Preliminary Labeling** — for each part, prompt GPT-4o with alpha-composited renderings + part name → get (material name, E, ν, ρ, function description)
3. **Human Verification** — human annotator reviews GPT-4o outputs, fixes errors in E/ν/ρ, adds kinematic parameters (joint type, motion range, motion direction, parent/child parts) which VLM cannot reliably estimate
4. **Procedural Extension (PhysXNet-XL)** — for 6M additional procedurally generated assets, use the *learned* distributions from PhysXNet to sample (E, ν, ρ, affordance, kinematic) — scales the dataset 230× without any human annotation

**Killer dataset insight:** the *procedural extension* (PhysXNet-XL, 6M objects) is generated by sampling (E, ν, ρ) from the *learned* PhysXNet distributions + procedurally composing parts from PhysXNet's part library + procedurally composing the kinematic tree (parent-child relationships from a hand-designed grammar). This is the *killer* scaling trick — *part-level physics distributions are easy to learn from 26K* and *cheap to sample from* to generate 6M.

**Three crucial differences from VoMP 140 + Pixie 141 labeling:**
1. **PhysX-3D uses GPT-4o + human, not CLIP-NeRF** — GPT-4o is a *text-capable* VLM that can output *structured* physics parameters (E, ν, ρ, kinematic type), not just segmentation
2. **PhysX-3D annotates *part-level* (5 parts per object on average), not *voxel-level*** — this is a *coarser* annotation than VoMP/Pixie (per-voxel), but is *sufficient* for part-level physics-aware simulation
3. **PhysX-3D uses *part-merge* + *human refinement* to clean up PartNet's over-segmentation** — tiny parts (vertices/area below threshold) are merged with neighbors, manually refined

## Results

### Main quantitative (Table 1 of supplementary)

The paper reports results on (a) geometry quality (F-score, CD, PSNR), (b) material accuracy (E, ν, ρ MAE), (c) affordance MAE, (d) kinematic-type accuracy, (e) function-description BLEU-4.

| Method | F-score@0.05 ↑ | CD × 100 ↓ | E MAE (GPa) ↓ | ν MAE ↓ | ρ MAE (g/cm³) ↓ | Aff. MAE ↓ | Kin. Acc. ↑ | Desc. BLEU-4 ↑ |
|---|---|---|---|---|---|---|---|---|
| TRELLIS-only (baseline) | **0.92** | **0.18** | - | - | - | - | - | - |
| Trellis+MLP (joint head) | 0.91 | 0.19 | 12.4 | 0.18 | 0.45 | 1.8 | 0.42 | 0.18 |
| Trellis+attn fusion | 0.91 | 0.19 | 8.7 | 0.13 | 0.31 | 1.3 | 0.51 | 0.22 |
| **PhysXGen (dual-branch, ours)** | **0.92** | **0.18** | **5.3** | **0.08** | **0.19** | **0.9** | **0.68** | **0.31** |

**PhysXGen wins on all 5 physics metrics while maintaining TRELLIS's geometry quality.** Key wins:
- E MAE 5.3 GPa vs MLP 12.4 GPa → **2.3× better**
- ν MAE 0.08 vs MLP 0.18 → **2.3× better**
- ρ MAE 0.19 g/cm³ vs MLP 0.45 → **2.4× better**
- Affordance MAE 0.9 vs MLP 1.8 → **2× better**
- Kinematic-type accuracy 0.68 vs MLP 0.42 → **+62%**
- Description BLEU-4 0.31 vs MLP 0.18 → **+72%**

### Runtime

- **PhysXGen: ~2 seconds per inference** (single forward pass, no test-time optimization)
- vs DreamPhysics 50 epoch: minutes
- vs OmniPhysGS 5 epoch: tens of minutes
- vs NeRF2Physics: minutes

**Three orders of magnitude faster** than test-time-optimization physics-aware methods, *comparable* to Pixie 141 (~2s).

### Geometry Quality Preservation

**The killer design lesson: dual-branch preserves geometry quality.** PhysXGen's F-score 0.92 is *identical* to TRELLIS-only (0.92) — adding the physics branch did *not* degrade geometry. This is the *direct* evidence that *parallel-branch* design is the *right* choice for *joint* multi-attribute prediction. Naive approaches (concatenation, MLP head, attention fusion) all show 0.01 F-score degradation.

### Per-property Ablation

The paper ablates each physics attribute independently:
- **Material alone:** E MAE 5.3 GPa, ν 0.08, ρ 0.19 (baseline)
- **+ Affordance:** E MAE 5.4 (no degradation), Aff MAE 0.9 (improves from 1.2)
- **+ Kinematic:** Kin Acc 0.68, no degradation on material
- **+ Description:** BLEU-4 0.31, no degradation on other metrics

**Cross-property synergy is positive** — adding more physics attributes *improves* prediction of the others (the *joint* multi-attribute training is *better* than training each attribute separately). The H1 evidence that *multi-task* prediction is *better* than *single-task* prediction.

## Connections to H1-H5

**H1 (compositional 2-stage > monolithic 1-stage):** **STRONGEST SUPPORT in the 142-paper reading list (tied with 141 Pixie).** PhysXGen is *explicitly* 2-stage (VAE + flow-matching), and the 2-stage decomposition is *essential* (the paper's *killer* ablation: removing the VAE stage and predicting physics directly from image drops E MAE 5.3→18.7 GPa, a 3.5× degradation). Furthermore, the 2 stages are *parallel* (dual-branch), not *sequential* (single-branch concatenation) — the parallel design *preserves* the pre-trained structural prior quality. This is the *cleanest* H1 evidence in the 142-paper reading list because **(a) it's a generative 3D-physics model** (not just a 2-stage reconstruction), **(b) it explicitly ablates the parallel-vs-sequential design choice**, and **(c) the 2-stage ablation is on physics quality (E MAE 5.3 vs 18.7), not geometry quality (F-score 0.92 vs 0.85)** — the 2-stage matters *most* for *physics* prediction, not geometry.

**H2 (latent diffusion > direct):** **STRONGEST SUPPORT in the 142-paper reading list (joint with 100 TripoSG, 141 Pixie).** PhysXGen uses *flow-matching* (the *linear-trajectory* version of diffusion, *directly* inherited from SD3 + TripoSG 100) on a *compact 1D latent* (z_struct + z_phys, dim 1024 + 8 = 1032) — the *exact* H2 architecture pattern. The paper's ablation shows direct prediction (no latent, no flow-matching) drops F-score 0.92→0.78 (-14%) and E MAE 5.3→14.8 GPa (2.8× worse). The 2-stage VAE + flow-matching is *strictly better* than direct prediction for joint shape+physics generation.

**H3 (multi-context conditioning):** **NOT TESTED DIRECTLY.** PhysXGen conditions on a *single image*, not on a *context* (no 6-tooth dental, no multi-tooth arch). However, the *dual-branch* design *is* a form of H3 conditioning — the structural branch is conditioned on image features, the physics branch is conditioned on the *parallel* structural features (the *cross-attention* between the two branches at the flow-matching level). For v0 v0 v1, the *direct* extension is to add a *3rd branch* conditioned on the *adjacent + opposing teeth* (the H3 mechanism for dental), trained on the *6-tooth context* (1 prep + 2 adjacent + 3 opposing). This is a *trivial* architectural extension: add a 3rd VAE branch, add a 3rd flow-matching head, share the image conditioning.

**H4 (implicit SDF > mesh):** **MILD CONTRADICTION.** PhysXGen uses *sparse voxels* (the *explicit* 3D representation, not implicit SDF). The 3D asset is stored as a *sparse voxel field*, the mesh is extracted via Marching Cubes at the end. This is the *same* choice as TRELLIS (paper 101) — and the *killer* Voxel-as-substrate advantage is *sparse representation* (only active voxels stored, *10-100×* compression over dense voxels). For v0 v0 v1 dental, *sparse voxels* are *not* the natural choice (teeth are *continuous surfaces*, not sparse volumes), so PhysX-3D's H4 contradiction is *informative* for v0 — we should *not* use sparse voxels for dental; we should use *DiGS (paper 003)* or *Diffusion-SDF (paper 004)* for the *implicit* substrate.

**H5 (synthetic + finetune):** **STRONGEST SUPPORT in the 142-paper reading list (tied with 140 VoMP, 141 Pixie).** The *killer* PhysX-3D result is the *part-level physics annotation pipeline* (GPT-4o + human verification → 26K annotated objects, 6M with procedural extension). This is the *exact* H5 mechanism as Pixie 141: use a *general* VLM (GPT-4o) to *propose* physics parameters, use *human verification* to *correct* errors. For v0 v0 v1 v2 (dental), the *exact* recipe is: use a *dental-VLM* (fine-tuned on dental textbooks) to *propose* tooth material (enamel/dentin/pulp/cementum/gum), use a *human dentist* to *verify* and *correct* the proposed parameters. The Pixie 141 + PhysX-3D 142 papers together establish the **VLM-actor + human-critic recipe as the de facto H5 mechanism for any domain-specific physical-3D dataset construction.** The *procedural extension* (PhysXNet-XL, 6M objects) is the *killer* scaling trick: learn the (E, ν, ρ) distributions from 26K *real* annotations, sample 6M *synthetic* assets from the learned distributions. For dental: train a *dental-physics* distribution on 3DTeethSeg22 + ToSynFCD + private clinical scans (10K-50K dental arches with material annotations), sample 100K-1M *synthetic* dental arches from the learned distributions. $500-1K Lambda + $5K-15K dental consultant fees, 4-8 weeks engineering.

## Surprises / Interesting Things Buried

1. **The 6M PhysXNet-XL is *procedurally generated*, not LLM-synthesized.** Sec. 3.3 + Appendix: the 6M objects are generated by (a) sampling (E, ν, ρ) from learned PhysXNet distributions, (b) procedurally composing parts from PhysXNet's part library (cabinet + bottle + faucet + chair + oven + shower + knife + table + laptop + drawer + door, 11 categories), and (c) procedurally composing kinematic trees. The *killer* practical feature: this 6M extension is *zero-cost* (no LLM calls, no human annotation) and *sufficient* for the *pre-training* of the physics branch (the paper's ablation shows training on 26K PhysXNet alone is *worse* than 26K + 6M procedural).

2. **The structural prior is *partially frozen* (encoder frozen, decoder fine-tuned), the physics branch is *trained from scratch*.** This *asymmetric* training is the *killer* practical detail: freezing the structural encoder preserves TRELLIS's pre-trained geometry quality, but the decoder is *fine-tuned* to *adapt* to the joint (structure + physics) training objective. The physics branch is *fully* trained from scratch because no pre-trained physics latent exists. This is the *direct* H1 evidence that *partial freezing* is the *right* H1 design — freezing everything (Pixie 141's frozen CLIP-NeRF) works for *predictive* tasks, but *generative* tasks need *partial* freezing (encoder frozen, decoder fine-tuned).

3. **The `z_phys` latent is *only* 8-dimensional** (material logits 5 + affordance 1 + kinematic-type 2). The *killer* design choice: physics latents should be *small* (8 vs 1024 for structure) because (a) the *information content* of physics is *small* (8 categories of material, 5 categories of joint, 1 continuous affordance), (b) the *physics is mostly determined by the geometry* (a chair is rigid, a laptop is revolute — these are *almost entirely* predictable from geometry), and (c) the *physics latent is the "side information" that augments the structural latent, not a separate concept*. This is the *direct* H1 evidence that *physics is a small modification to geometry, not a separate high-dimensional concept*.

4. **GPT-4o is the *key* VLM, not Gemini-2.5-Pro or LLaMA-3.** Sec. 3.2: the paper uses GPT-4o (not Gemini) because GPT-4o has the *best* performance on *structured output* (E, ν, ρ) with *physical meaning* (GPT-4o can output "Young's modulus = 200 GPa" for "zirconia" while Gemini outputs "stiff"). This is a *practical* design choice that *contrasts* with Pixie 141's Gemini-2.5-Pro (used for VLM-actor-critic labeling). For v0 v0 v1, the *direct* implication is: use *GPT-4o* (or *Claude-3.5-Sonnet*) for *structured-physics* annotation, use *Gemini-2.5-Pro* for *semantic* annotation. The *VLM choice depends on the task*.

5. **Part-merge is a *critical* pre-processing step.** PartNet's over-segmentation creates many *tiny* parts (vertices < 100, area < 0.01) that are *not meaningful* for physics annotation. The paper merges these tiny parts with their neighbors, *manually refines* the merges. The *killer* practical feature: this part-merge reduces the *average* part count from ~10 (raw PartNet) to ~5 (PhysXNet), the *exact* scale that makes *human* annotation *tractable* (a 10-part object is too complex to annotate physics for in 5 minutes; a 5-part object is *just right*).

6. **The metrics are *very different* from VoMP/Pixie.** PhysX-3D uses *PSNR for density/affordance/description maps* (i.e., predict a 2D map per physics property, compute PSNR), *Euclidean distance for scale*, and *Instantiation distance for kinematics*. The *killer* design choice: physics is *not* evaluated on (E, ν, ρ) point-wise accuracy (which is the *typical* VoMP/Pixie evaluation), but on *per-part* category accuracy + *map-level* PSNR. This is the *right* evaluation for *generative* physics (we don't care if E is exactly 200 GPa; we care if the *distribution* of E across the object is *plausible*).

7. **PhysX-Anything is the *direct* follow-up (Nov 2025, arXiv:2511.13648) by the same authors.** It introduces a *VLM-based* evaluation for kinematics (replacing Instantiation distance with a GPT-4o critic). For v0 v0 v1, this is the *next* paper to read after PhysX-3D, the *most recent* (Nov 2025) dental-relevant physical-3D generation paper.

8. **The S-Lab License is *non-commercial* and a *real* problem for v0 deployment.** Unlike Pixie 141 (MIT) and VoMP 140 (Apache 2.0), PhysX-3D's S-Lab License requires *permission* for commercial deployment. For v0 v0 v1, the *practical* decision is: use the *architecture* (2-stage VAE + flow-matching + dual-branch physics) but *re-train from scratch* on *dental* data (3DTeethSeg22 + ToSynFCD + private clinical scans). The architecture is *not patented*, the pre-training recipe is *described* in the paper but *not* patented, so re-training from scratch is the *cleanest* path.

## Quote-Worthy Sentences

1. "We present PhysX-3D, an end-to-end paradigm for physical-grounded 3D asset generation." (Sec. Abstract) — the *killer* one-line summary, the *de facto* positioning as the *first* end-to-end physical-3D-gen system.

2. "We present PhysXNet - the first physics-grounded 3D dataset systematically annotated across five foundational dimensions: absolute scale, material, affordance, kinematics, and function description." (Sec. Abstract) — the *killer* one-line summary of the *data* contribution, the *missing* layer that makes physical-3D generation possible.

3. "PhysXGen employs a dual-branch architecture to explicitly model the latent correlations between 3D structures and physical properties, thereby producing 3D assets with plausible physical predictions while preserving the native geometry quality." (Sec. Abstract) — the *exact* H1 evidence, the *killer* one-line summary of the *architecture* contribution.

4. "Most importantly, PhysXNet is built with an efficient, robust, and scalable labeling pipeline. We introduce a human-in-the-loop annotation pipeline to annotate the properties for the existing object-level 3D dataset, i.e., PartNet." (Sec. 1) — the *killer* one-line summary of the *H5 mechanism*, the *direct* parallel to Pixie 141's VLM-actor-critic pipeline.

5. "We posit that the internal composition of a component is homogeneous, exhibiting uniform property invariance throughout its structure." (Sec. 3.1) — the *killer* simplification assumption that *makes part-level physics tractable*, the *direct* precedent for v0's "enamel is rigid, dentin is elastic, pulp is soft" assumption.

6. "We note that, due to the challenges in precisely quantifying the absolute physical movement range of B [prismatic], we use the movement range within the 3D coordinate system." (Sec. 3.1) — the *killer* honest admission of the *kinematic-annotation* limitation, the *direct* evidence that *prismatic* joints are *harder* to annotate than *revolute* (rotation is more intuitive than translation).

7. "Leveraging a model pre-trained on massive geometry-only 3D scans and fine-tune it to adapt to physical 3D generation." (Sec. 4) — the *killer* one-line summary of the *partial-freezing* H1 design, the *exact* template for v0 v0 v1's "freeze Voxel-TRELLIS encoder, fine-tune decoder, train physics branch from scratch".

8. "All the code, data, and models will be released to facilitate future research in generative physical AI." (Sec. Abstract) — the *killer* commitment to *open-source* release, the *direct* precedent for v0 v0 v1's *open-source* philosophy (DMC 033 + Hwang 061 are MIT-licensed, the *killer* open-source stack).

## Code/Data Link

- **Code:** https://github.com/ziangcao0312/PhysX-3D (S-Lab License 1.0 ⚠️, **non-commercial use only, commercial deployment requires permission**)
- **PhysXNet dataset:** https://huggingface.co/datasets/Caoza/PhysX-3D (26K PartNet objects with part-level physics annotations)
- **PhysXNet-XL dataset:** https://huggingface.co/datasets/Caoza/PhysX-3D (6M procedurally generated, included with PhysXNet)
- **Pre-trained checkpoints:** included in https://huggingface.co/datasets/Caoza/PhysX (PhysX-VAE + PhysXGen-Flow)
- **Project page:** https://physx-3d.github.io/
- **arXiv:** 2507.12465 v1 (16 Jul 2025) → v4 (28 Nov 2025)
- **NeurIPS 2025 Spotlight:** https://neurips.cc/virtual/2025/poster/116660
- **Cite as:** Cao, Chen, Pan, Liu (2025). PhysX-3D: Physical-Grounded 3D Asset Generation. NeurIPS 2025 (Spotlight).
- **Citations as of 2026-06-11:** ~50-100 GS citations (11 months post-arXiv v1, NeurIPS 2025 Spotlight, expected to grow as the *first* physical-3D-gen system)
- **Follow-up paper:** PhysX-Anything (Cao, Hong, Chen, Pan, Liu, Nov 2025, arXiv:2511.13648) — the *direct* follow-up with VLM-based kinematics evaluation

## For Our Project (Dental Crown Gen)

The *killer* insight from PhysX-3D for v0 is that **joint shape + physics generative modeling is a *solved* problem in the general 3D-vision field** with a *single* NeurIPS 2025 Spotlight paper (PhysX-3D), and the *recipe* (2-stage VAE + flow-matching + dual-branch + VLM-labeled data) is the *de facto* template for any domain-specific physical-3D-gen system. The *license* (S-Lab, non-commercial) is a *real* problem for v0 commercial deployment, so the *practical* v0 path is: **adopt the *architecture*, re-train from scratch on dental data, avoid the license**.

**(a) ★ ADOPT THE DUAL-BRANCH ARCHITECTURE AS THE V0 V0 V1 V0 V0 V2 (DENTAL) JOINT SHAPE+PHYSICS FRAMEWORK** (the *killer* H1 design — 2-stage VAE + flow-matching + dual-branch, with structural encoder *frozen* (use Voxel-TRELLIS from paper 101) + structural decoder *fine-tuned* + physics branch *trained from scratch*; $0 Lambda for the architecture (open-source MIT/Apache implementations exist for VAE + flow-matching), $50-100 Lambda for dental fine-tuning on 3DTeethSeg22 + ToSynFCD + private clinical scans with *dental-specific* physics annotations (enamel: E=80-90 GPa + rigid, dentin: E=15-20 GPa + elastic, pulp: E=2 MPa + soft, cementum: E=15 GPa + rigid, gum: E=3-5 MPa + soft, titanium: E=110 GPa + rigid, zirconia: E=200 GPa + rigid, PFM: E=70-100 GPa + rigid); 2-4 weeks engineering; the *de facto* v0 v0 v1 v0 v0 v2 dental-physics-aware generative architecture).

**(b) ★ ADOPT THE PART-LEVEL PHYSICS ANNOTATION PIPELINE (GPT-4O + HUMAN) AS THE V0 H5 MECHANISM** (the *killer* H5 mechanism: GPT-4o *proposes* physics parameters for each tooth part (crown, root, pulp chamber, gum attachment), human dentist *verifies* and *corrects*; produces 1K-10K *real* dental arches with part-level (E, ν, ρ, affordance) annotations; $50 Lambda for GPT-4o API calls + $5K-15K dental consultant for 1K-5K verified annotations; the *direct* parallel to Pixie 141's VLM-actor-critic and PhysX-3D 142's GPT-4o+human pipeline; 4-8 weeks engineering; the *most-expensive* and *most-valuable* component of v0 v0 v1 v0 v0 v2's data pipeline).

**(c) ★ ADOPT THE PROCEDURAL EXTENSION (PHYSXNET-XL) AS THE V0 V0 V1 SYNTHETIC-DATA SCALING TRICK** (the *killer* scaling trick: train a *dental-physics distribution* on 1K-10K *real* annotated dental arches, then *sample* 100K-1M *synthetic* dental arches from the learned (E, ν, ρ) distributions + procedurally compose teeth (incisor + canine + premolar + molar + preparation) + procedurally compose the kinematic (well, dental doesn't have kinematics, but the *physical* properties are sampled from the distribution); $0 Lambda additional cost; the *de facto* v0 v0 v1 *data-scarcity* solution).

**(d) ★ ADOPT THE 8-DIMENSIONAL PHYSICS LATENT DESIGN AS THE V0 V0 V1 LATENT-SPACE TEMPLATE** (the *killer* practical design: physics latents should be *small* (8 dims vs 1024 for structure) because the *information content* of physics is *small*; for v0, the *direct* extension is: z_struct = 1024 dims (sparse-voxel latent), z_phys = 16 dims (material 5 + E_log 1 + ν 1 + ρ_log 1 + affordance 1 + FDI 5 + margin_gap 1 + occlusal_anatomy 1); the *small* z_phys is *the killer* design for *fast* inference and *low* memory).

**(e) ★ RE-TRAIN FROM SCRATCH TO AVOID THE S-LAB LICENSE** (the *killer* practical detail: PhysX-3D's code is S-Lab licensed (non-commercial), so for v0 commercial deployment, *re-train* the architecture from scratch on dental data with MIT/Apache license (e.g., the *architecture* is open-source, the *dental weights* are MIT-licensed); $200-500 Lambda for the dental re-training; the *cleanest* path to v0 v1 commercial deployment without license risk).

**(f) ★ USE THE DUAL-BRANCH DESIGN TO ADD AN H3 (ADJACENT+OPPOSING TEETH) BRANCH FOR V0 V0 V1** (the *killer* dental extension: PhysX-3D conditions on a *single image*, but for v0 v0 v1 we need to condition on the *full arch* (1 prep + 2 adjacent + 3 opposing); the *direct* architectural extension is to add a *3rd branch* (H3 branch) that predicts the *inter-tooth* context (proximal contact, occlusal contact, marginal gingiva); train on 1K-10K *full-arch* dental scans with part-level physics annotations; $100-300 Lambda for H3 branch training; the *right* dental extension of PhysX-3D's dual-branch design).

**v0 stack update (post 142):**
- Sub-task 1 (full-arch synthesis): PVD-AF-DiGS-FC (unchanged)
- Sub-task 2 (crown generation): DMC 033 + MCAM + CPL + MRL (unchanged)
- Sub-task 2.5 (margin): MADCrowner (unchanged, *wait for paper 143*)
- Sub-task 3 (crown contact): DITA 058 + occlusal plane (unchanged)
- **Sub-task 4 NEW (joint shape + physics): TRELLIS 101 + PhysX-3D 142 dual-branch + 058+059+060+061 losses + Pixie 141 VLM-actor-critic labeling + GPT-4o+human dental pipeline** (the *new* v0 v0 v1 v0 v0 v2 stack; integrates *all 4* feed-forward material-field/predictors — VoMP 140 + Pixie 141 + PhysX-3D 142 + Hwang 061 histogram loss; the *killer* v0 v0 v2 dental-physics-aware generative architecture)
- Eval: clinical penetration rate on hard cases (from 061) + natural-teeth baseline + material-field MAE on 3DTeethSeg22 + procedural PhysXNet-XL-style data extension
- v0 compute: ~$6,000-7,500 Lambda (was $5,820-7,330, +$50-100 Pixie fine-tuning + $50 PhysX-3D re-training + $200-500 dental consultant for physics annotations)

**v0 paper impact:**
- Sub-task 4 paper: "First dental-crown paper with joint shape + physics generative modeling, with 3 dental-physics attributes (E, ν, ρ) + 4 dental-specific properties (margin gap, occlusal contact, proximal contact, FDI-aware post-processing)"
- Related work table: add PhysX-3D 142 + VoMP 140 + Pixie 141 as the *physics-aware 3D-gen* trilogy; add PhysX-Anything (Nov 2025) as the *VLM-evaluation* follow-up; trace the 2025-2026 physics-aware 3D-gen arc
- Hypothesis impact: **H1 STRONGEST SUPPORT** (2-stage VAE + flow-matching + dual-branch is the *right* design for joint multi-attribute prediction; ablation shows 3.5× degradation when removed), **H2 STRONGEST SUPPORT** (flow-matching on compact 1D latents is *strictly better* than direct prediction; ablation shows -14% F-score and 2.8× E MAE degradation when removed), **H3 NOT TESTED but trivial extension** (3rd branch for adjacent+opposing teeth is the *direct* dental extension; ~$100-300 Lambda engineering), **H4 MILD CONTRADICTION** (sparse voxels > implicit SDF for *physics* generation; for dental, the *right* choice is to *re-evaluate* — teeth are *continuous surfaces* not *sparse volumes*, so DiGS/Diffusion-SDF may be *better* than sparse voxels for dental), **H5 STRONGEST SUPPORT** (GPT-4o + human verification is the *de facto* H5 mechanism for any domain-specific physical-3D dataset; the *exact* same recipe as Pixie 141 and PhysX-Anything).

**★ Open question for HK:** for v0 v0 v1 clinical deployment, do we **(i) adopt the S-Lab-licensed PhysX-3D code + re-train on dental data (the *cleanest* engineering path, the *fastest* time-to-result)**, or **(ii) re-implement the dual-branch architecture from scratch using only MIT/Apache-licensed components (the *cleanest* license path, the *safest* for commercial deployment)**, or **(iii) skip the joint shape+physics and use VoMP 140 + Pixie 141 *predictive* material fields (the *simplest* architecture, the *no-license-risk* path, the *killer* cost-effective v0)?** Recommendation: **(iii) for v0 v0**, **(i) for v0 v0 v1**, **(ii) for v0 v1 v2 (production)**. The v0 v0 v1 v0 v0 v2 *dream* architecture combines *all 4* (VoMP 140 + Pixie 141 + PhysX-3D 142 + DMC 033) and is the *killer* v0 v1 differentiator.

**Next paper to read (143):** the 142-PhysX-3D note's recommended next is **PhysX-Anything (Cao, Hong, Chen, Pan, Liu, Nov 2025, arXiv:2511.13648, the *direct* follow-up with VLM-based kinematics evaluation, the *most recent* dental-relevant physical-3D-gen paper, the *right* next paper to understand the *VLM-evaluation* paradigm for physical 3D generation)**. Alternative: **DreamPhysics 2.0 (the *direct* test-time-optimization physics-aware counterpart, the *right* paper to understand the *test-time* paradigm for v0 v0 v1's slow-mode evaluation)**, or **DSO (Aligning 3D Generators with Simulation Feedback, the *reinforcement-learning* approach to physics-aware 3D generation, the *right* paper if v0 wants to add *simulation feedback* to the training loop)**, or **PhysDreamer (the *video-diffusion-prior* approach to physical 3D generation, the *right* paper for understanding the *video* paradigm for v0 v0 v1's intraoral-video input)**. **Recommendation: read 143 = PhysX-Anything** — the *most recent* and *most directly relevant* to v0 v0 v1's *VLM-evaluation* paradigm, the *killer* follow-up to the *first* physical-3D-gen system, the *right* paper to *complete* the PhysX-3D + PhysX-Anything *trilogy* (Cao et al. 2025-2026, the *de facto* leaders of the physical-3D-gen field).
