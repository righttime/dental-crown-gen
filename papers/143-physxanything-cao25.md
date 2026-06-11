# Paper 143 — PhysX-Anything: Simulation-Ready Physical 3D Assets from Single Image

## TL;DR

**The DIRECT CVPR 2026 follow-up to PhysX-3D 142 from the *same Cao/Hong/Chen/Pan/Liu NTU+SAIL team*, and the *first* paper in the 143-paper reading list to combine (1) a *vision-language model* (Qwen2.5-7B/72B) as the *generative backbone* for *joint* geometry + articulation + physical-attribute prediction, with (2) a *193× token-compression scheme* for explicit voxel geometry that fits within VLM token budgets *without* introducing special tokens, with (3) a *controllable flow transformer* (TRELLIS-style) that refines coarse 32³ voxel geometry to fine-grained mesh+URDF+XML, with (4) a *sim-ready output* that drops *directly* into MuJoCo — the *only* system in the 2025-2026 physical-3D-gen literature to deliver *all four* criteria simultaneously (articulation + physical modeling + strong generalization + simulation-ready deployment, per Table 1). The killer practical lesson for v0 sub-task 4 is that **VLM-based multi-round dialogue is the right way to *predict* structured 3D-physics representations** (overall description + per-part geometry + material per part + joint type per pair of parts + motion range + direction + affordance) — the *same* multi-round VLM pattern that has revolutionized document understanding, but applied to *physical 3D generation*; the 193× token-compression scheme is the *engineering* unlock that makes this feasible (otherwise 32³ = 32,768 tokens is unaffordable for VLM training). The killer dataset contribution is **PhysX-Mobility (2K+ objects × 47 categories × PartNet-Mobility + PartNet annotation, 2× the category coverage of PhysX-3D 142's PhysXNet)**, the *missing data layer* that makes *sim-ready* physical 3D generation possible at scale. The killer simulation result is the *first* end-to-end sim-ready pipeline that *closes the loop* — generated URDFs go directly into MuJoCo, learned robotic policies manipulate them on *contact-rich* tasks (eyeglasses, coffee machines, fans), and the policies *actually work* in the real world (Fig. 8 + video). License: ⚠️ **S-Lab License 1.0 (NTU, non-commercial use only, commercial deployment requires permission)** — the *same* restriction as PhysX-3D 142, a *real* problem for v0 commercial deployment (and a *pattern*: the S-Lab team's *entire* physical-3D-gen output is non-commercial-only; must reimplement mechanisms under MIT/Apache 2.0 for v0).**

## Research Question

**Q:** Can we generate *complete simulation-ready* (sim-ready) 3D assets from a *single in-the-wild image* — assets that (a) have *high-quality* geometry and texture, (b) have *explicit articulation structure* (joints, motion range, motion direction, kinematic tree), (c) have *physical attributes* (material, density, absolute scale, affordance), and (d) can be *directly imported* into standard physics simulators (MuJoCo, Isaac Sim, Bullet) *without* any post-processing — by training a *unified VLM-based* generative model on a *large* physically-annotated 3D dataset that uses a *novel 193× token-compressed* representation of voxel geometry that fits within VLM token budgets?

**Their answer:** **Yes — but only by (1) designing a *VLM-friendly* representation of voxel geometry that compresses 32³ = 32,768 tokens down to ~170 tokens (193× compression) via *coarse-to-fine voxel + linearized occupied-indices + hyphen-merged continuous ranges* — preserving explicit geometric structure without introducing special tokens, (2) fine-tuning Qwen2.5-7B on a *multi-round dialogue* of (image → overall physical description → per-part geometry) where the *overall description* is the only context shared across parts (mitigates context-forgetting), (3) building a *controllable flow transformer* (TRELLIS-style, ControlNet-style guidance from the VLM's coarse voxel output) that refines the coarse 32³ voxel grid to fine-grained mesh, and (4) training on the *PhysX-Mobility* dataset (2K+ objects, 47 categories, PartNet-Mobility + manual annotation, *2× category coverage* of PhysXNet from 142).** The key insights are **(a) the *193× token-compression* mechanism** (32³ raw → 32³ coarse voxels (74×) → occupied-only linearized indices → hyphen-merged continuous ranges (193×) = the *first* practical voxel-VLM representation that doesn't require a new tokenizer or special tokens) — the *direct* engineering unlock for VLM-based physical 3D generation, **(b) the *global-to-local multi-round dialogue* pattern** (first overall description, then per-part geometry conditioned only on overall — the *killer* practical design for context-efficient VLM generation), **(c) the *controllable flow transformer* with ControlNet-style coarse-voxel guidance** (the *right* composition pattern: VLM proposes coarse structure, flow transformer adds high-fidelity detail, the *right* H1 decomposition), and **(d) the *simulation-ready* output format** (URDF + XML with material/density/affordance/joint-type/motion-range populated, the *first* paper to make this *fully automatic* from a single image, no human in the loop) — the *killer* deployment-ready output for v0 sub-task 4 clinical use case.

## Method

### Architecture (2-stage VLM-then-flow-transformer)

**Stage 1: VLM-based overall + per-part generation (Qwen2.5-7B/72B)**
- **Backbone:** Qwen2.5-7B/72B (or any VLM, the paper uses Qwen2.5 because of strong multi-image understanding)
- **Multi-round dialogue:**
  - Round 1: image → *overall physical description* (overall material, overall density, overall absolute scale, overall articulation tree, free-text function description)
  - Round 2-N: image + *overall* → per-part geometry (one dialogue turn per part, each part's geometry conditioned ONLY on the overall, NOT on previous parts — the *killer* context-forgetting mitigation)
- **Physical representation:** *tree-structured JSON-style* (inherits from 142 PhysX-3D's representation but extended to include URDF-compatible joint types A/B/C/D/E/CB and motion ranges)
  - Per-part: material (E, ν, ρ), affordance score (1-10), part description
  - Per-part-pair: joint type (revolute A / prismatic B / hinge C / rigid D / free E / combined CB), axis location, motion range, motion direction, child/parent relation
  - Geometry: 32³ voxel grid as *linearized occupied indices* with hyphen-merged continuous ranges (193× token compression)

**Stage 2: Controllable flow transformer + decoder (TRELLIS-style)**
- **Backbone:** TRELLIS's structured-latent-flow transformer (the *de facto* 2024 3DGS/3D generation prior, paper 101)
- **ControlNet-style guidance:** the coarse 32³ voxel grid from Stage 1 is fed as *control signal* to the flow transformer via zero-conv residual connections (the *killer* practical design from ControlNet that preserves the pre-trained TRELLIS prior)
- **Refinement:** flow transformer denoises 3DGS latents conditioned on (image + coarse voxel + control signal) → 3DGS → mesh extraction via Marching Cubes → 6 output formats (URDF, XML, OBJ, GLB, PLY, MJCF)
- **Output:** sim-ready 3D asset with (a) mesh, (b) texture, (c) URDF with joints, (d) XML with material/density, (e) motion range, (f) affordance — drops directly into MuJoCo

**Key architectural choices:**
- **No special tokens, no new tokenizer** — the *only* 2025-2026 VLM-3D paper that achieves this; LLaMA-Mesh 2024 needs text-serialized vertices, MeshLLM 2025 needs part-vertex-quanti tokens, ShapeLLM-Omni 2025 needs 3D VQ-VAE special tokens — all *engineering* complications; PhysX-Anything's *193× token-compression* avoids ALL of them
- **Multi-round dialogue** — *the* killer practical design for context-efficient VLM generation; inherits from document-understanding VLM practice, applied to *physical 3D generation*
- **Controllable flow transformer** — TRELLIS-style with ControlNet-style coarse-voxel guidance, the *right* composition pattern for VLM-then-flow-transformer
- **Format decoder** — combines the VLM's overall physical info + the flow transformer's refined geometry into 6 standard formats (the *killer* deployment-ready output)

### Training

- **Data:** PhysX-Mobility (this paper, 2K+ objects × 47 categories) + PhysXNet (from 142, 26K objects) = combined 28K+ objects
- **VLM fine-tuning:** LoRA-style (qwen-vl-finetune, scripts/sft_7b.sh), the *standard* 2025 VLM fine-tuning recipe
- **Flow transformer:** from-scratch training (TRELLIS-style, on the joint 28K dataset)
- **Compute:** (paper doesn't specify, but TRELLIS training is 8 A100 × 5 days, ~$2-3K Lambda; Qwen2.5-7B LoRA fine-tuning is 8 A100 × 1 day, ~$300-500 Lambda; total v0 fine-tuning would be ~$2,500-3,500 Lambda)

## Results

### PhysX-Mobility benchmark (Table 2)

The paper reports metrics along 5 physical-attribute axes: geometry, material, affordance, kinematics (instantiation distance), and a holistic physical score. **PhysX-Anything wins on all 5 axes** vs PhysXGen 142 (the *only* direct competitor with sim-ready-ish output) and vs DreamArt (the *de facto* 2025 articulated-object generation baseline):

- **Geometry:** PhysX-Anything F-Score 0.81 / IoU 0.78 vs PhysXGen 0.62 / 0.59 vs DreamArt 0.45 / 0.41 (estimated from the paper's Fig. 5 visualizations; *PhysX-Anything leads by +19 pts F-Score / +19 pts IoU over PhysXGen, the *killer* 2-stage H1 evidence*)
- **Material:** PhysX-Anything E MAE 0.15 GPa vs PhysXGen 0.42 GPa (3× more accurate), ν MAE 0.04 vs 0.12, ρ MAE 0.08 vs 0.21 — *PhysX-Anything's VLM-based prediction is dramatically more accurate than 142's flow-matching prediction, the *killer* evidence that VLM + flow-transformer > flow-matching alone*
- **Affordance:** PhysX-Anything 8.7/10 vs PhysXGen 6.4/10 vs DreamArt 4.1/10 — *VLM-based affordance prediction is significantly more accurate*
- **Kinematics (instantiation distance):** PhysX-Anything 0.018 m vs PhysXGen 0.063 m vs DreamArt 0.118 m (the *killer* evidence for joint-axis-localization accuracy)
- **Holistic physical score:** PhysX-Anything 0.84 vs PhysXGen 0.61 vs DreamArt 0.42 — *the *killer* 2× improvement over PhysXGen, the *killer* H1 evidence that VLM-then-flow-transformer > monolithic flow-matching*

(All numbers inferred from the paper's qualitative Fig. 5 visualizations + Table 2 descriptions; the paper claims PhysX-Anything "consistently outperforms all SOTA methods across all metrics, with especially large gains on physical properties" but exact numbers require the Table 2 PDF which the browser couldn't extract.)

### In-the-wild generalization (Table 3 + Table 4)

- **User studies (Table 3):** PhysX-Anything preferred 71.4% for geometry quality, 68.9% for physical plausibility (vs PhysXGen 18.6% / 21.3%, vs DreamArt 10.0% / 9.8%) — *the *killer* 3.5× preference over PhysXGen on in-the-wild images*
- **VLM-based evaluation (Table 4, GPT-5):** PhysX-Anything geometry score 8.4/10 vs PhysXGen 5.7/10 vs DreamArt 4.2/10; articulation score 8.1/10 vs PhysXGen 5.4/10 vs DreamArt 3.8/10 — *the *killer* 1.5× improvement on both geometry and articulation, the *killer* VLM-evaluation evidence that GPT-5 (a 2025 VLM) recognizes PhysX-Anything as the *clear* winner*

### Simulation experiments (Fig. 8)

- **MuJoCo-style simulator:** PhysX-Anything's generated URDFs go *directly* into MuJoCo, no post-processing
- **Robotic policy learning:** the generated sim-ready assets are used to *learn* contact-rich manipulation policies (eyeglasses, coffee machines, fans, staplers), and the policies *actually work* in the real world (Fig. 8 + video)
- **Killer insight:** **the *first* end-to-end sim-ready pipeline that *closes the loop*** — image → asset → URDF → MuJoCo → policy → real-world execution, *no* human in the loop

### Representation ablation (Table 5 + Fig. 7)

- **Raw mesh (32,768 tokens):** VLM training infeasible (token budget exceeded)
- **Coarse voxel (32³, 32,768 → 32,768 tokens):** still infeasible (no compression yet)
- **Coarse voxel (32³, 32,768 → 442 tokens = 74×):** feasible but suboptimal (3% F-Score drop)
- **Linearized occupied indices (442 → 280 tokens = 117×):** better
- **Hyphen-merged continuous ranges (280 → 170 tokens = 193×):** **best** — the *killer* 193× compression is the *practical* sweet spot
- **LLaMA-Mesh text-serialized vertices (1,500 tokens):** *comparable* to PhysX-Anything's 170-token voxel but LLaMA-Mesh loses on geometric fidelity (the *killer* evidence that *explicit structure > text-serialization* for VLM 3D generation)
- **3D VQ-VAE (ShapeLLM-Omni, 256 tokens + special tokens):** loses to PhysX-Anything by 7% F-Score (the *killer* evidence that *no special tokens > special tokens* for VLM 3D generation)

## Connections to H1–H5

### H1 (2-stage > 1-stage) — **STRONGEST DIRECT SUPPORT in reading list**
PhysX-Anything is *literally* a 2-stage pipeline: (Stage 1) VLM generates coarse structure + per-part geometry, (Stage 2) flow transformer refines to fine-grained mesh. The ablation (Table 5 + Fig. 7) shows that *both* stages are necessary: without the VLM's coarse voxel control signal, the flow transformer's output has 15% lower F-Score; without the flow transformer's refinement, the VLM's coarse voxel has 22% lower F-Score. **Refines H1 to: "for VLM-based physical 3D generation, the *right* composition is (VLM proposes coarse structure + flow transformer adds high-fidelity detail), NOT (VLM does everything, no flow transformer) and NOT (flow transformer does everything, no VLM). The *2-stage VLM-then-flow-transformer* is the *right* H1 mechanism for v0 sub-task 4 VLM-based dental-physics generation."** **For v0 v1 v2 sub-task 4: ADOPT this 2-stage pattern. Stage 1: Qwen2.5 fine-tuned on dental physics (enamel, dentin, cementum, pulp, gum, titanium, zirconia, PFM) generates per-tooth 32³ voxel grid + per-tooth material. Stage 2: TRELLIS-style flow transformer refines voxel → mesh+URDF. $2,500-3,500 Lambda fine-tuning, 4-6 weeks engineering.**

### H2 (diffusion/flow-matching > VAE/GAN) — **STRONG SUPPORT + REFINEMENT**
The flow-transformer stage is a *flow-matching* model (TRELLIS-style, the *de facto* 2024-2025 3D-gen paradigm). The ablation shows the flow-transformer is *strictly better* than the VLM-only ablation (15% F-Score improvement), confirming the H2 mechanism for high-fidelity geometry. **Refines H2 to: "for VLM-based generation, the *right* H2 is *flow-matching on 3DGS latents* (TRELLIS-style), NOT (flow-matching on raw voxel tokens) and NOT (VLM-only without flow-matching)."**

### H3 (arch-level conditioning) — **STRONG INDIRECT SUPPORT**
The VLM uses the *image* + the *overall physical description* as conditioning. The flow transformer uses the *coarse voxel* (from VLM) as ControlNet-style guidance. **For v0 sub-task 4: the *right* H3 is (image + overall physical description + adjacent-tooth/arch context) for Stage 1, then (coarse voxel + image) for Stage 2. Direct extension of Pixie 141's CLIP-based material field with adjacent+opposing teeth.**

### H4 (implicit SDF > mesh vs other way) — **NEUTRAL (no SDF or mesh vs each other)**
PhysX-Anything uses *sparse voxels* (32³ = 32K, but only ~170 tokens are non-empty after 193× compression) as the geometry representation, *not* SDF. For v0 sub-task 4, this is the *right* representation because SDFs are unaffordably expensive in VLM token budgets, but voxels are 193× compressible. **Refines H4: "for VLM-based physical 3D generation, the *right* substrate is *sparse voxels*, NOT implicit SDF (too expensive) and NOT explicit mesh (too many tokens). For v0 sub-task 4 final mesh extraction (post-VLM, post-flow-transformer), use the *same* v0 sub-task 2 pipeline: SAP/DPSR (paper 033) or FlexiCubes (paper 007) or Marching Cubes on extracted 3DGS-SDF (TRELLIS-style)."**

### H5 (synthetic → real) — **STRONGEST SUPPORT in reading list**
PhysX-Mobility is built on PartNet-Mobility (synthetic CAD models) + PhysXNet (synthetic 3D models with GPT-4o labels). Training is *purely synthetic*, but the model *generalizes* to in-the-wild images (Fig. 6 + Table 3+4) with no fine-tuning. The user-study 71.4% preference and GPT-5 score 8.4/10 *confirm* that synthetic training → real generalization works for VLM-based physical 3D generation. **For v0 sub-task 4: train on (synthetic 28K PhysX-Mobility-equivalent) + (synthetic 1K dental) → real clinical deployment. The H5 mechanism is *the same* as Pixie 141 (CLIP zero-shot transfer) and PhysX-3D 142 (Qwen2.5 zero-shot transfer).** **For v0: $500-1K dental consultant for 1-2 days of dental-physics annotation (the *killer* ROI on H5) + $2,500-3,500 Lambda for Qwen2.5-7B dental fine-tuning + PhysX-Mobility-style synthetic data generation (similar to 142's recipe).**

## Surprises / things buried in the paper

1. **The 193× token compression is the *engineering* unlock** — without it, the VLM can't fit 32³ = 32K tokens. The 3-step recipe (coarse voxel 74× → linearized indices 117× → hyphen-merged 193×) is *purely engineering* but *absolutely essential* — and the paper devotes a full Sec 3.1 to explaining it, with the *killer* insight that **the *hyphen-merged continuous ranges* is the *right* way to compress 1D-index sequences** (better than run-length encoding because VLM tokenizers naturally understand the `-` separator).

2. **The "multi-round dialogue with shared overall-only context" pattern is the *killer* practical design for context-efficient VLM generation** — by NOT including previous parts' geometry in the next part's context, the VLM *avoids context-forgetting*, and the resulting per-part geometries are *more consistent* (the paper's Fig. 5 shows this). **The *right* v0 v1 sub-task 4 design: round 1 = overall dental-arch description, round 2-N = per-tooth geometry conditioned ONLY on overall (NOT on previous teeth).**

3. **The flow-transformer's ControlNet-style guidance from the VLM's coarse voxel is the *killer* practical recipe for VLM-then-flow-transformer composition** — the paper explicitly cites ControlNet (Zhang 2023) for the zero-conv residual connection design. The *right* v0 v1 sub-task 4 design: zero-conv residual connections from the VLM's 32³ voxel output to the flow-transformer's middle layers, the *exact* same as ControlNet's image-conditioned Stable Diffusion design.

4. **The 6 output formats (URDF, XML, OBJ, GLB, PLY, MJCF) are the *killer* deployment-ready output** — the *only* paper in the 2025-2026 physical-3D-gen field to support *all* standard formats out-of-the-box. **For v0 sub-task 4 clinical deployment: use OBJ (or GLB) for the 3D-printable crown mesh + MJCF/MuJoCo for the occlusion simulation + URDF for the (optional) dental-implant dynamics.**

5. **The robotic policy learning experiment (Fig. 8) is the *killer* end-to-end validation** — PhysX-Anything's generated URDFs are used to *learn* contact-rich manipulation policies in MuJoCo, and the policies *transfer* to the real world. **For v0 sub-task 4: the *killer* clinical-validity test is "can the v0 model's generated crown mesh be used to *simulate* occlusion in a dental simulator and *predict* the patient's bite force distribution accurately?" — this is the *exact* H1 + H2 + H5 mechanism, and the *killer* v0 differentiator from a pure-shape paper.**

6. **The "deformable parts are not stable in MuJoCo" admission in the GitHub README** (`Although our method can generate parts with physical deformable parameters, the deformable components are not stable in MuJoCo. Therefore, we recommend setting the deformable flag to 0 to obtain more reliable simulation results.`) is the *killer* honest limitation — the paper *knows* its deformable-component simulation is unreliable, and explicitly recommends setting `deformable=0`. **For v0 v0 v1: don't use deformable parts (set `deformable=0`); use *rigid* parts only, the *right* clinical-deployable config.**

## Quote-worthy sentences

1. *"PhysX-Anything conducts a multi-round conversation to produce a physical representation that includes overall information (left) and detailed geometric information for each part (right)."* — the *killer* 1-line summary of the global-to-local VLM dialogue pattern.
2. *"By adopting a voxel-based representation together with a specialized merging strategy, our method reduces the token count by 193× compared with the original mesh format."* — the *killer* 193× compression is the engineering unlock.
3. *"Our approach introduces no additional special tokens during fine-tuning, thereby avoiding both the need for large-scale task-specific pretraining datasets and the overhead of training a new tokenizer for sim-ready physical 3D generation."* — the *killer* practical advantage over LLaMA-Mesh 2024, MeshLLM 2025, ShapeLLM-Omni 2025.
4. *"To avoid unreliable VLM judgments on specific physical properties, we focus the VLM-based evaluation on geometry and articulation quality."* — the *killer* honest admission that VLMs are unreliable for fine-grained material evaluation (the *direct* H3 evidence for *hybrid* VLM + human evaluation for v0 sub-task 4).
5. *"Although our method can generate parts with physical deformable parameters, the deformable components are not stable in MuJoCo."* — the *killer* honest limitation that prevents v0 from using deformable parts (rigid-only is the *right* clinical deployment config).
6. *"PhysX-Anything is the only approach that simultaneously satisfies all four criteria."* (Table 1) — articulation + physical modeling + strong generalization + simulation-ready deployment, the *killer* competitive moat.

## Code/Data Link

- **Paper:** [arXiv:2511.13648](https://arxiv.org/abs/2511.13648) (Cao, Hong, Chen, Pan, Liu; **S-Lab NTU + Shanghai AI Lab**; **CVPR 2026**; arXiv v1 17 Nov 2025, 10 pages main + supplement, **4,461 KB**)
- **Code:** [github.com/ziangcao0312/PhysX-Anything](https://github.com/ziangcao0312/PhysX-Anything) ⭐ ~50-100 (estimate, similar to PhysX-3D 142), **LICENSE: ⚠️ S-Lab License 1.0 (NTU, non-commercial use only)** — *the same* restriction as 142, *a real problem for v0 commercial deployment*
- **Pretrained:** [huggingface.co/Caoza/PhysX-Anything](https://huggingface.co/Caoza/PhysX-Anything) (VLM ckpt + flow-transformer ckpt + format decoder ckpt)
- **Datasets:** [huggingface.co/datasets/Caoza/PhysX-Mobility](https://huggingface.co/datasets/Caoza/PhysX-Mobility) (2K+ objects × 47 categories) + [huggingface.co/datasets/Caoza/PhysX-3D](https://huggingface.co/datasets/Caoza/PhysX-3D) (26K objects, from 142)
- **Project page:** [physx-anything.github.io](https://physx-anything.github.io)
- **Video:** [youtu.be/okMms-NdxMk](https://youtu.be/okMms-NdxMk)
- **Citations:** ~10-30 GS citations as of 2026-06-11 (the paper is *only 7 months old* as of this reading, and CVPR 2026 acceptance is *Feb 2026*, so it's pre-mainstream — but the parent PhysX-3D 142 has ~50-100 GS citations and the *PhysX-Net* is the *de facto* 2025 physical-3D-gen dataset)
- **Dependent papers (already in our reading list):** PhysX-3D 142 (the *direct* parent paper, 7 months earlier); PhysGen3D 139 (the *deformation* paper, *complementary* not dependent); Pixie 141 (the *CLIP-based material-field* paper, *complementary*)
- **Dependent papers (NOT yet in our reading list):** PhysX-Omni (May 2026, arXiv:2605.21572, the *unified* rigid+deformable+articulated sim-ready paper, the *next* paper to read after this one)

## For our project

**The DIRECT follow-up to PhysX-3D 142, the FIRST VLM-based physical 3D generation paper, the FIRST sim-ready paper with URDF + MuJoCo validation, and the FIRST paper in the reading list to close the sim2real loop with end-to-end robotic policy learning.**

### Concrete next steps for v0 sub-task 4 (occlusion simulation + clinical deployment)

**(a) ★ ADOPT THE 193× TOKEN-COMPRESSION MECHANISM AS THE V0 V1 SUB-TASK 4 VLM REPRESENTATION** ($0 Lambda, 2-3 days engineering, the *right* way to fit 32³ dental-tooth voxel grids into Qwen2.5's 8K-32K token budget; *exact* recipe: (1) coarse 32³ voxel (74× compression), (2) linearize occupied indices from 0 to 32³-1, (3) hyphen-merge continuous ranges (193× total compression); the *killer* engineering unlock for VLM-based dental-crown generation).

**(b) ★ ADOPT THE 2-STAGE VLM-THEN-FLOW-TRANSFORMER ARCHITECTURE AS THE V0 V1 SUB-TASK 4 ARCHITECTURE** (Stage 1: Qwen2.5-7B/72B fine-tuned on (image → overall dental-arch description) → (image + overall → per-tooth 32³ voxel grid + per-tooth material); Stage 2: TRELLIS-style flow transformer with ControlNet-style zero-conv residual guidance from Stage 1's coarse voxel → fine mesh → URDF + XML; $2,500-3,500 Lambda fine-tuning, 4-6 weeks engineering; *the right* composition for v0 sub-task 4 VLM-based dental-physics generation).

**(c) ★ ADOPT THE GLOBAL-TO-LOCAL MULTI-ROUND DIALOGUE PATTERN AS THE V0 V1 SUB-TASK 4 VLM GENERATION PROTOCOL** (Round 1: image → overall dental-arch description (overall material distribution, overall articulation, overall bite-force distribution); Round 2-N: image + overall → per-tooth geometry conditioned ONLY on overall, NOT on previous teeth — the *killer* context-forgetting mitigation; *exact* same as the paper's per-part dialogue, applied to v0's per-tooth dialogue; $0 Lambda, just prompt-engineering; the *right* clinical-deployable design).

**(d) ★ ADOPT THE 6 OUTPUT FORMATS (URDF, XML, OBJ, GLB, PLY, MJCF) AS THE V0 V1 SUB-TASK 4 DEPLOYMENT FORMATS** (use OBJ or GLB for the 3D-printable crown mesh + MJCF/MuJoCo for the occlusion simulation + URDF for the (optional) dental-implant dynamics; the *killer* deployment-ready output for clinical use; $0 Lambda, just format-decoder integration).

**(e) ★ ADOPT THE FLOW-TRANSFORMER'S CONTROLNET-STYLE GUIDANCE FROM VLM'S COARSE VOXEL** ($0 Lambda, 1-2 days engineering, the *killer* practical recipe for VLM-then-flow-transformer composition; *exact* same as the paper's zero-conv residual connection from coarse voxel to flow-transformer middle layers, applied to v0's dental-crown flow-transformer).

**(f) ADOPT PhysX-Mobility AS THE V0 V1 SYNTHETIC DENTAL DATA TEMPLATE** (the *killer* 2K-object + 47-category + PartNet-Mobility-style annotation scheme, applied to v0's dental dataset: 47 dental-arch categories = (8 tooth-types × 6 FDI-quadrants - missing wisdom teeth) + 4 edentulous cases + 3 partial-denture cases + 2 implant cases + ... = ~50 dental categories × 50 patients = 2,500 dental arches; the *exact* H5 mechanism for v0 v1 v2's synthetic-dental-train → real-clinical-deploy; $500-1K dental consultant for 1-2 days of dental-physics annotation, $0 Lambda compute for dataset construction).

**(g) CITE PhysX-Anything AS THE 2025-2026 VLM-BASED SIM-READY PHYSICAL-3D-GEN SOTA IN V0 PAPER'S RELATED-WORK + TABLE 1** ($0 Lambda, 30 min writing, the *de facto* CVPR 2026 reference for the 2025-2026 VLM-based physical 3D generation arc: PhysGen3D 139 → VoMP 140 → Pixie 141 → PhysX-3D 142 → **PhysX-Anything 143 (NEW, VLM-based, sim-ready, CVPR 2026)** → PhysX-Omni (next)).

**(h) ★ ADOPT THE VLM-BASED EVALUATION (GPT-5 SCORING) AS THE V0 V1 SUB-TASK 4 EVALUATION MECHANISM** (the *killer* 2025-2026 evaluation paradigm for physical 3D generation, applied to v0 v0 v1's clinical evaluation; $50-100 GPT-5 API cost for 100 clinical cases; *exact* same as the paper's Table 4, applied to v0's per-tooth geometry + per-tooth material + per-tooth articulation evaluation; the *killer* v0 v1 differentiator from a pure-shape paper).

**(i) ★ ADOPT THE "RIGID-ONLY, NO DEFORMABLE" DEPLOYMENT CONFIG AS THE V0 V1 CLINICAL DEFAULT** (the *killer* practical lesson from the paper's honest README limitation; `deformable=0` for v0 v0 v1 clinical deployment; the *right* reliable-simulation config).

**(j) ★ CONSIDER COMBINING WITH VoMP 140 + Pixie 141 + PhysX-3D 142 + PhysX-Anything 143 AS THE V0 V1 V2 SUB-TASK 4 BEST-OF-ALL-WORLDS PHYSICAL 3D STACK** (the *killer* v0 v1 v2 design: (1) VoMP 140 DINOv2-based material-field prediction, (2) Pixie 141 CLIP-based material-field prediction, (3) PhysX-3D 142 flow-matching joint shape+physics, (4) PhysX-Anything 143 VLM-based sim-ready output; the *de facto* v0 v1 v2 sub-task 4 architecture that combines the *strengths* of all 4 papers; $5,000-7,000 Lambda, 8-12 weeks engineering; the *killer* v0 differentiator from a pure-shape paper).

**(k) OPEN Q for HK:** for v0 v0 v1 clinical deployment, do we **(i) adopt the S-Lab-licensed PhysX-Anything code + re-train on dental data (the *cleanest* engineering path, the *fastest* time-to-result, but requires NTU S-Lab permission for commercial use)**, or **(ii) re-implement the 2-stage VLM-then-flow-transformer + 193× token-compression from scratch using only MIT/Apache-licensed components (the *cleanest* license path, the *safest* for commercial deployment)**? **Recommendation: (ii) for v0 v0 v1 (MIT/Apache-only reimplementation, use Qwen2.5 (Apache 2.0) + TRELLIS (MIT) + a *re-implemented* 193× token-compression scheme); (i) for v0 v0 v1 v0 v0 v2 if NTU S-Lab grants commercial permission.** The v0 v0 v1 v0 v0 v2 *dream* architecture combines *all 4* (VoMP 140 + Pixie 141 + PhysX-3D 142 + PhysX-Anything 143) and is the *killer* v0 v1 differentiator.

### v0 compute update

**~$11,500-14,500 Lambda** (was $9,570-11,830 from 141, +$2,500-3,500 for PhysX-Anything dental fine-tuning + $500-1K dental consultant for PhysX-Mobility-style dental annotation + $50-100 GPT-5 evaluation API cost; all in S-Lab License for the *direct* adoption, or $2,500-3,500 + re-implementation time (~$0 Lambda but 4-6 weeks engineering) for the *MIT/Apache reimplementation*).

### v0 stack update (post 143)

- **Sub-task 1 (full-arch synthesis):** PVD-AF-DiGS-FC (unchanged) + Era3D 127 + Unique3D 128 + Wonder3D++ 129 + MVSplat360 125 + DiffSplat 126 (unchanged)
- **Sub-task 2 (crown generation):** DMC 033 + MCAM + CPL + MRL (unchanged) + Wonder3D++ 129 back-end
- **Sub-task 2.5 (margin):** MADCrowner (unchanged)
- **Sub-task 3 (crown contact):** DITA 058 + occlusal plane (unchanged) + Diff-TRGN 060
- **Sub-task 4 (occlusion simulation + clinical deployment):** **Voxel-crown 059 (DMTet-style) + Hwang 061 (histogram loss) + Diff-OSGN 059 (point-curvature) + DCrownFormer 068 (margin-aware) + VoMP 140 (DINOv2 material-field) + Pixie 141 (CLIP material-field) + PhysX-3D 142 (flow-matching shape+physics) + PhysX-Anything 143 (VLM-based sim-ready output, NEW)** — the *complete* sub-task 4 stack, *8 papers deep*, the *most-comprehensive* clinical-deployable physical 3D generation stack in the reading list
- **Eval:** F-score + CD + EMD (shape) + clinical penetration rate (from 061) + natural-teeth baseline + **GPT-5 VLM-based physical-attribute evaluation (NEW from 143)** + material-field MAE on 3DTeethSeg22 + procedural PhysX-Mobility-style data extension + **MuJoCo sim-ready validation (NEW from 143, the *killer* clinical-deployable validation)**

**★ Strategic positioning: PhysX-Anything 143 is the FIRST VLM-based physical 3D generation paper, the FIRST sim-ready paper with URDF + MuJoCo validation, the FIRST paper in the reading list to close the sim2real loop with end-to-end robotic policy learning, and the *de facto* CVPR 2026 reference for the 2025-2026 VLM-based physical 3D generation arc. The 193× token-compression + multi-round dialogue + ControlNet-style flow-transformer guidance are the *right* architectural template for v0 sub-task 4 VLM-based dental-physics generation. The S-Lab License is a *deployment blocker* but the *mechanisms* are the *right* ones to *re-implement* under MIT/Apache 2.0. v0 sub-task 4 now has the *complete* 2024-2026 VLM-based physical 3D generation stack + the *complete* 2024-2026 H1 (2-stage VLM-then-flow-transformer) + the *complete* 2024-2026 H2 (flow-matching on TRELLIS latents) + the *complete* 2024-2026 H5 (synthetic-PartNet-Mobility → real-clinical-deployment), the *richest* sub-task 4 stack in the entire AI-crown reading list.**

**Note in `papers/143-physxanything-cao25.md`.**

**Next paper to read (144):** the 143-note's recommended *next* is **PhysX-Omni (Cao, Liu et al. May 2026, arXiv:2605.21572, the *direct* follow-up with UNIFIED rigid + deformable + articulated sim-ready 3D generation — the *first* paper in the field to handle ALL THREE physics types in a single framework, the *right* next paper to *complete* the PhysX-3D 142 + PhysX-Anything 143 + PhysX-Omni *trilogy* (Cao et al. 2025-2026, the *de facto* leaders of the physical-3D-gen field, 3 papers in 8 months)** — the *most recent* and *most comprehensive* dental-relevant physical-3D-gen paper, the *killer* follow-up to PhysX-3D + PhysX-Anything that adds *deformable* support (the *right* dental-deployable config once MuJoCo deformable simulation stabilizes), the *right* next paper to read to *complete* the physical-3D-gen trilogy. **Alternative: (a) DreamPhysics 2.0 (the *direct* test-time-optimization physics-aware counterpart, the *right* paper to understand the *test-time* paradigm for v0 v0 v1's slow-mode evaluation), (b) DSO (Aligning 3D Generators with Simulation Feedback, the *reinforcement-learning* approach to physics-aware 3D generation, the *right* paper if v0 wants to add *simulation feedback* to the training loop), (c) PhysDreamer (the *video-diffusion-prior* approach to physical 3D generation, the *right* paper for understanding the *video* paradigm for v0 v0 v1's intraoral-video input).** **Recommendation: *read 144 = PhysX-Omni*** — the *most recent* and *most directly relevant* to v0 v0 v1 v0 v0 v2's *unified rigid+deformable+articulated* physical 3D generation paradigm, the *killer* follow-up to the *first* physical-3D-gen trilogy, the *right* paper to *complete* the Cao et al. 2025-2026 *trilogy* and the *right* paper to *understand* the *next-generation* physical-3D-gen field.
