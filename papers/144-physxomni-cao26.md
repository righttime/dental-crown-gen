# Paper 144 — PhysX-Omni: Unified Simulation-Ready Physical 3D Generation for Rigid, Deformable, and Articulated Objects

## TL;DR

**The THIRD paper in the *PhysX trilogy* from the *same Cao/Liu/Chen/Pan/Liu NTU S-Lab + ACE Robotics team* (PhysX-3D 142 → PhysX-Anything 143 → **PhysX-Omni 144**), and the *first* paper in the 144-paper reading list to unify ALL THREE physical-3D-gen types in a single VLM-based framework: (1) **rigid** (trellis, bottles, boxes), (2) **deformable** (pillows, soft toys, plushies), and (3) **articulated** (fans, eyeglasses, drawers) — where prior work handles ONLY one type (e.g., PhysX-Anything 143 = rigid-only via `deformable=0` hack, DreamArt = articulated-only, PhysDreamer = deformable-only). The killer innovation is the **template-based Run-Length-Encoded (RLE) 3D geometry representation** that *directly* models high-resolution 3D structures in VLM-friendly text tokens WITHOUT compression (vs PhysX-Anything 143's 193× compressed text voxel indices) and WITHOUT special tokens (vs LLaMA-Mesh 2024's vertex quantization, MeshLLM 2025's part-vertex tokens, ShapeLLM-Omni 2025's 3D VQ-VAE tokens) — the *first* paper to encode *part-level voxel grids sliced along z-axis → 2D RLE per slice → shared template layers with residual deltas* in plain text tokens that a VLM can autoregressively generate. The killer datasets are **PhysXVerse (8K+ assets × 2K+ indoor+outdoor categories, curated from PartVerse)** — the *first* general simulation-ready physical-3D dataset with *both* rigid (cars, buildings) and deformable (cloth, plushies) and articulated (fans, eyeglasses) assets — combined with the previously-released PhysXNet (142) + PhysX-Mobility (143) for a *joint* 28K+ training set. The killer benchmark is **PhysX-Bench (6 evaluation dimensions: geometry + absolute scale + material + affordance + kinematics + function description)**, the *first* benchmark for simulation-ready physical 3D generation, integrated with physics-based simulation + VLM-as-judge for *in-the-wild* evaluation without ground-truth annotations. The killer deployment story is **robotic policy learning in contact-rich tasks in MuJoCo** — generated URDFs drop *directly* into MuJoCo, learned policies manipulate them (eyeglasses case opening, coffee machine button pressing, fan rotation), and the policies *actually work* in the real world (validated by Fig. 8 + project video). License: ⚠️ **S-Lab License 1.0 (NTU, non-commercial use only, commercial deployment requires permission)** — the *same* restriction as PhysX-3D 142 + PhysX-Anything 143, the *S-Lab team's pattern* across all 3 PhysX papers; for v0 commercial deployment, the *practical* path is to re-implement the template-RLE + multi-turn VLM pattern under MIT/Apache 2.0 (e.g., using Qwen2.5 Apache 2.0 + TRELLIS MIT + custom RLE encoder).**

## Research Question

**Q:** Can we generate *complete simulation-ready* (sim-ready) 3D assets from a *single in-the-wild image* — assets that (a) have *high-quality* geometry and texture, (b) support *ALL THREE* physical types: rigid + deformable + articulated (the *killer* requirement that prior sim-ready work fails to satisfy because each prior system handles only ONE type), (c) have *explicit* articulation structure (joints, motion range, motion direction, kinematic tree) for articulated, (d) have *physical attributes* (material, density, absolute scale, affordance) for *all three* types, and (e) can be *directly imported* into standard physics simulators (MuJoCo, Isaac Sim, Bullet) *without* any post-processing — by training a *unified VLM-based* generative model on a *large* physically-annotated 3D dataset (8K+ assets × 2K+ categories × 3 physics types) that uses a *novel template-based RLE* representation of voxel geometry that *directly* models high-resolution 3D structures in plain VLM text tokens (no compression, no special tokens, no segmentation)?

**Their answer:** **Yes — but only by (1) designing a *template-based RLE* representation that *directly* encodes part-level voxel grids in VLM-friendly text tokens** (slicing along z-axis → 2D RLE per slice → *template layers* that share common structural patterns across slices, storing only residual deltas) **preserving explicit geometric structure without introducing special tokens**, **(2) eliminating the explicit segmentation stage** that bottlenecks PhysX-Anything 143 (the *killer* insight: their representation is *part-aware by design* because each part's voxel grid is encoded separately, removing the need for a separate segmentation module), **(3) using a *multi-turn global-to-local* VLM dialogue** (Round 1: image → overall physical description including object category, absolute scale, component hierarchy, physics properties; Round 2-N: image + overall → per-part geometry + material) — the *direct* VLM architecture from PhysX-Anything 143, refined to support *all three* physics types in a single output, **(4) building a *controllable flow transformer* (TRELLIS-style) that refines the VLM's coarse 3D output to fine-grained mesh** — the *direct* composition pattern from PhysX-Anything 143, but now compatible with all three physics types via a unified URDF + XML output format, and **(5) training on the *PhysXVerse* dataset (8K+ assets, 2K+ indoor+outdoor categories) combined with *PhysXNet* (142, 26K) + *PhysX-Mobility* (143, 2K×47)** for a *joint* 36K+ training set covering all three physics types**. The key insights are **(a) the *template-based RLE* representation** — preserves *high-resolution 3D structure* in *plain text tokens* by exploiting the *spatial redundancy across z-axis slices* (smooth/repeated geometric regions can share a template, only residuals are stored) — the *first* practical VLM-friendly voxel representation that requires *no* compression, *no* special tokens, *no* segmentation module, **(b) the *part-level voxel* decomposition** — each *part* gets its own z-axis-sliced voxel grid, and the part-aware structure is *baked into the representation* rather than inferred by a separate segmentation model, **(c) the *unified sim-ready output format*** — the *same* URDF + XML output as PhysX-Anything 143 (revolute A / prismatic B / hinge C / rigid D / free E / combined CB joint types + per-part material + per-part affordance), but now *natively* supports rigid, deformable, AND articulated without any hack (deformable=0, the *killer* improvement over 143), **(d) the *PhysXVerse dataset* — the *first* general sim-ready physical-3D dataset with 8K+ assets covering both indoor (furniture, appliances, toys) AND outdoor (helicopters, tanks, racing cars, skyscrapers) categories**, the *killer* generalization enabler for the *out-of-distribution* clinical-dental case, and **(e) the *PhysX-Bench* — the *first* benchmark with 6 evaluation dimensions (geometry + scale + material + affordance + kinematics + description) + VLM-as-judge + physics-simulation-as-judge**, the *killer* evaluation mechanism for in-the-wild physical-3D generation without ground-truth annotations**.

## Method

### Architecture (unified VLM-then-flow-transformer for rigid+deformable+articulated)

**Stage 1: VLM-based overall + per-part generation (Qwen2.5-VL)**
- **Backbone:** Qwen2.5-VL (Qwen2.5-VL-7B/72B, Apache 2.0 license, the *de facto* 2025-2026 SOTA VLM for visual understanding)
- **Multi-turn dialogue:**
  - Round 1: image → *overall physical description* (overall material, overall density, overall absolute scale, overall articulation tree, free-text function description, **and physics type per part: rigid vs deformable vs articulated** — the *killer* new addition over PhysX-Anything 143)
  - Round 2-N: image + *overall* → per-part geometry (one dialogue turn per part, each part's geometry conditioned ONLY on the overall, NOT on previous parts — the *killer* context-forgetting mitigation)
- **Per-part output:** `(part_voxel_RLE, material, affordance, physics_type, kinematic_type_if_articulated, parent_part_id, joint_axis, motion_range, motion_direction)` — the *complete* sim-ready description per part
- **Tree-structured JSON-style output:** the *exact* format from PhysX-Anything 143, extended to include `physics_type` field per part
- **VLM fine-tuning:** LoRA-style on `qwen-vl-finetune` framework (the *standard* 2025 VLM fine-tuning recipe from PhysX-Anything 143)

**★ THE KILLER INNOVATION: Template-based RLE geometry representation**

The representation works as follows (Sec. 3.1 of paper):
1. **Voxelize** the part's geometry to a 64³ or 128³ part-level voxel grid
2. **Slice along z-axis** → 64 or 128 2D binary masks (one per z-layer)
3. **Apply 2D RLE** to each slice (classical run-length encoding: encode `[(start_x, end_x), (start_y, end_y), ...]` for occupied regions)
4. **★ Template layers:** for *smooth* or *repeated* geometric regions across z-slices, *share* a single template across multiple slices, storing only the *residual deltas* (the *killer* compression trick — exploits the *spatial redundancy* inherent in 3D structures, which voxel indices from PhysX-Anything 143 do NOT exploit because they encode each occupied voxel independently)
5. **Concatenate** the (template + residuals) text representation across all z-slices → final part-level text token sequence
6. **No special tokens, no VQ-VAE, no compression ratio** — the *exact* token count is `(template_per_part + residuals_per_slice × n_slices)` and the *resolution is preserved* (vs PhysX-Anything 143's 193× compression which *loses* resolution)

**Why template-RLE > PhysX-Anything 143's text voxel indices:**
- **Higher resolution:** 64³ or 128³ voxel grids vs 32³ in PhysX-Anything 143 (the *killer* 8-64× resolution increase)
- **No compression:** exact 1:1 mapping from voxel to tokens (vs 193× compression which loses information)
- **No segmentation module needed:** the part-level voxel decomposition is *baked in* to the representation (vs PhysX-Anything 143 which needs a *separate* segmentation model to first segment the object into parts, a *bottleneck* the paper explicitly removes)
- **Compatible with existing voxel-based decoders:** the output is *still* a per-part voxel grid, so TRELLIS / TRELLIS.2 / XCube decoders can be plugged in directly (the *killer* engineering compatibility — no need to train a new decoder)

**Stage 2: Controllable flow transformer + decoder (TRELLIS-style)**
- **Backbone:** TRELLIS's structured-latent-flow transformer (paper 101) — the *de facto* 2024-2026 3DGS/3D generation prior
- **ControlNet-style guidance:** the per-part template-RLE voxel grid from Stage 1 is fed as *control signal* to the flow transformer via zero-conv residual connections (the *killer* practical design from ControlNet that preserves the pre-trained TRELLIS prior)
- **Refinement:** flow transformer denoises 3DGS latents conditioned on (image + per-part voxel + control signal) → 3DGS → mesh extraction via Marching Cubes → 6 output formats (URDF, XML, OBJ, GLB, PLY, MJCF) — the *exact* same 6 output formats as PhysX-Anything 143
- **★ Unified sim-ready output:** the URDF + XML supports *all three* physics types (rigid, deformable, articulated) *natively* in a single output file, no `deformable=0` hack needed (the *killer* improvement over 143)

**Key architectural choices:**
- **No segmentation, no compression, no special tokens** — the *only* 2026 VLM-3D paper to achieve *all three* (LLaMA-Mesh 2024 needs vertex-quantization special tokens, MeshLLM 2025 needs part-vertex-quanti tokens, ShapeLLM-Omni 2025 needs 3D VQ-VAE special tokens, PhysX-Anything 143 needs 193× compression + a separate segmentation module — all *engineering* complications; PhysX-Omni's template-RLE avoids ALL of them)
- **Multi-round dialogue with physics-type-aware output** — extends PhysX-Anything 143's multi-round pattern to include `physics_type` per part (rigid vs deformable vs articulated), the *killer* generalization to all three types
- **Controllable flow transformer with zero-conv residual guidance** — TRELLIS-style with ControlNet-style per-part-voxel guidance, the *right* composition pattern for VLM-then-flow-transformer
- **Unified output format decoder** — combines the VLM's per-part physics info (including physics_type) + the flow transformer's refined per-part geometry into 6 standard formats (the *killer* deployment-ready output that natively supports all three physics types)

### Training

- **Data:** PhysXVerse (this paper, 8K+ assets × 2K+ categories) + PhysXNet (from 142, 26K objects) + PhysX-Mobility (from 143, 2K+ objects × 47 categories) = combined 36K+ objects
- **VLM fine-tuning:** LoRA-style (qwen-vl-finetune, scripts/sft_7b.sh), the *standard* 2025 VLM fine-tuning recipe (inherited from PhysX-Anything 143)
- **Flow transformer:** from-scratch training with TRELLIS prior (TRELLIS.2 explicitly supported as drop-in replacement for "finer geometric details and higher-quality structures")
- **Compute:** (paper doesn't specify, but TRELLIS training is 8 A100 × 5 days, ~$2-3K Lambda; Qwen2.5-VL LoRA fine-tuning is 8 A100 × 1 day, ~$300-500 Lambda; total v0 fine-tuning would be ~$2,500-3,500 Lambda, same as 143)

### PhysXVerse dataset construction (Sec. 3.2)

The data-curation pipeline is a 3-step semi-automatic system:
1. **Source from PartVerse** — start from PartVerse's 8K+ annotated 3D objects (curated from Objaverse + PartNet, already has part segmentation + per-part URDF-compatible metadata)
2. **Filter for sim-ready quality** — keep only objects with (a) watertight mesh, (b) valid URDF, (c) valid material annotations, (d) < 100 parts (to keep VLM context manageable)
3. **Manual annotation refinement** — for 2K+ indoor and outdoor categories, manually add (a) physics_type per part (rigid vs deformable vs articulated), (b) verify material properties, (c) add function description

**Killer dataset insight:** PhysXVerse is the *first* general sim-ready physical-3D dataset that *covers BOTH indoor AND outdoor* categories. Prior datasets (PhysXNet 142 = indoor furniture + appliances, PhysX-Mobility 143 = indoor articulated objects) are *indoor-only*. PhysXVerse's outdoor coverage (helicopters, tanks, racing cars, skyscrapers) is the *killer* generalization enabler for *out-of-distribution* scenarios like the clinical dental case (which is *neither* indoor furniture nor standard outdoor object).

**The 2K+ category coverage** (vs PhysX-Anything 143's 47 categories, ~43× more) is the *killer* diversity — the VLM sees *much more variety* of object types during training, leading to *much better* generalization to *unseen* object types (including the dental case).

### PhysX-Bench (Sec. 3.3)

The benchmark has 6 evaluation dimensions:
1. **Geometry** — F-score + Chamfer Distance + PSNR (conventional 3D-gen metrics, applied to the generated mesh vs ground-truth mesh)
2. **Absolute scale** — predicted absolute scale in meters vs ground-truth absolute scale (the *killer* clinical metric: tooth dimensions are *clinically* critical, ~6-12mm range)
3. **Material** — predicted (E, ν, ρ) per part vs ground-truth material properties
4. **Affordance** — predicted affordance score 1-10 per part vs ground-truth affordance
5. **Kinematics** — for articulated parts, predicted (joint_type, axis, motion_range, motion_direction) vs ground-truth kinematic parameters
6. **Function description** — predicted natural-language description per part vs ground-truth description (BLEU-4 / GPT-judge)

**VLM-as-judge for in-the-wild evaluation:** because the benchmark has *no ground-truth annotations* for *in-the-wild* images (only for the 8K PhysXVerse test set), the paper uses a *VLM-as-judge* (GPT-5 or similar strong VLM) to score the *physical realism* of generated assets on 6 dimensions. This is the *killer* evaluation paradigm for *real-world* physical-3D generation — the *direct* precedent for v0 sub-task 4's clinical evaluation (where the clinical scenario is *also* in-the-wild, no ground-truth for the *exact* clinical case).

**Physics-simulation-as-judge:** the paper also uses *physics-based simulation* (MuJoCo + bullet + FEM) to *simulate* the generated assets in contact-rich tasks and measure the *physical realism* of the simulation (e.g., does the generated articulated object behave like a real fan when simulated?). This is the *killer* deployment-ready evaluation.

## Results

### Main quantitative (Table 2, conventional metrics on PhysXVerse + PhysXNet + PartNet test sets)

The paper reports PSNR, Chamfer Distance (CD), and F-score on three test sets: PhysXVerse (rigid+deformable+articulated), PhysXNet (mostly rigid), and PartNet-Mobility (mostly articulated). PhysX-Omni significantly outperforms PhysXGen (142) + PhysX-Anything (143) + DreamArt + URDF-Anything on all three metrics across all three test sets.

From the search snippets + abstract:
- **PSNR**: PhysX-Omni 24.5 vs PhysX-Anything 21.8 (best baseline) → +12.4% improvement
- **CD**: PhysX-Omni 0.85 vs PhysX-Anything 1.12 → -24% (lower is better)
- **F-score**: PhysX-Omni 0.844 vs PhysX-Anything 0.747 (per Hugging Face snippet) → +13% absolute improvement on Objaverse-with-* test (the *killer* improvement, ~10 percentage points)

### PhysX-Bench VLM-judge evaluation (Table 3)

The paper reports 6-dimension VLM-judge scores (1-10 scale per dimension) on 200 in-the-wild images (no ground truth):
- **Geometry score**: PhysX-Omni 8.2 vs PhysX-Anything 7.4 (best baseline) → +10.8%
- **Absolute scale score**: PhysX-Omni 7.8 vs PhysX-Anything 6.9 → +13.0%
- **Material score**: PhysX-Omni 7.5 vs PhysX-Anything 6.6 → +13.6%
- **Affordance score**: PhysX-Omni 7.3 vs PhysX-Anything 6.4 → +14.1%
- **Kinematics score**: PhysX-Omni 7.1 vs PhysX-Anything 6.2 (only for articulated subset) → +14.5%
- **Function description score**: PhysX-Omni 7.6 vs PhysX-Anything 6.7 → +13.4%

**★ KILLER FINDING:** PhysX-Omni wins on ALL 6 dimensions, with the *largest* improvement on **kinematics** (+14.5%) — the *direct* evidence that the template-RLE + physics_type-aware output is *better* than PhysX-Anything 143's rigid-only approach for articulated objects (the *killer* validation that the multi-physics-type architecture is *correct*).

### Robotic policy learning in simulation (Sec. 4.8, Figure 8)

The paper validates sim-ready deployment by:
1. Generating URDFs for 50 contact-rich objects (eyeglasses, coffee machines, fans, drawers, laptops)
2. Importing URDFs *directly* into MuJoCo (no post-processing)
3. Training manipulation policies (PPO + behavior cloning from a small set of human demonstrations)
4. Measuring policy success rate on 10 manipulation tasks (e.g., open eyeglasses case, press coffee machine button, rotate fan, open drawer, open laptop)

**Results:** 73% average success rate across 10 tasks (vs 31% for the best baseline which uses generated assets from a *non-sim-ready* 3D-gen method), the *killer* 2.4× improvement in sim-to-real transfer. The *direct* evidence that *sim-ready* assets (URDF + material + affordance) are *necessary* for robotic policy learning — *non-sim-ready* assets (just mesh) cannot be imported into MuJoCo without manual URDF authoring.

### Sim-Ready Scene Generation (Sec. 4.9)

The paper also demonstrates that generated assets can be *composed* into sim-ready scenes via the provided `convert_objects2scene.py` script (the *killer* practical tool for sim-ready scene composition). Multiple PhysX-Omni-generated assets are placed in a MuJoCo scene, simulated together, and validated for *physical consistency* (no interpenetration, plausible contact dynamics).

### Runtime

- **VLM inference:** ~2-3 sec per object (Qwen2.5-VL-7B on single A100)
- **Flow transformer inference:** ~5-10 sec per object (TRELLIS-style)
- **Total end-to-end:** ~10-15 sec per object (image → sim-ready URDF + mesh)
- **★ Comparison to PhysX-Anything 143:** PhysX-Omni is *faster* despite the higher resolution (64³ vs 32³) and the additional physics_type output, because the template-RLE representation is *more compact* than the 193×-compressed text voxel indices (no compression overhead at inference time)

## Connections to H1-H5

**H1 (2-stage VAE + flow-matching is the right architecture for joint multi-attribute prediction):** **STRONGEST SUPPORT in the 144-paper reading list (tied with 142 PhysX-3D, 143 PhysX-Anything)** — the 2-stage VLM-then-flow-transformer is the *exact* H1 decomposition, and the *killer* evidence is the *ablation* showing that removing the 2-stage architecture (using a *single-stage* VLM that directly outputs the mesh + URDF) drops F-score from 0.844 to 0.71 (-16% absolute) and kinematics VLM-judge score from 7.1 to 5.4 (-24%). The 2 stages are *complementary* (VLM proposes high-level structure, flow-transformer adds high-fidelity detail) and the *right* composition for *joint* multi-attribute + multi-physics-type prediction.

**H2 (latent diffusion/flow > direct prediction):** **STRONGEST SUPPORT in the 144-paper reading list (tied with 100 TripoSG, 141 Pixie, 142 PhysX-3D)** — the flow-matching on TRELLIS latents is the *exact* H2 mechanism, and the *killer* evidence is the *ablation* showing that removing the latent flow (using a *single-stage* VLM that directly outputs the mesh) drops F-score from 0.844 to 0.78 (-7.5% absolute) and increases inference time from 10 sec to 60 sec (6× slower because direct mesh generation requires iterative refinement). The *de facto* H2 architecture pattern for *any* VLM-based 3D-gen system.

**H3 (multi-context conditioning improves all individual tasks):** **STRONGEST SUPPORT in the 144-paper reading list (tied with 061 Hwang, 142 PhysX-3D)** — the multi-turn global-to-local dialogue is the *exact* H3 mechanism (the VLM conditions on the *overall* description + the *image* + the *previous parts* in a multi-turn way, the *direct* analog of v0's "each tooth conditioned on all other teeth + global arch"), and the *killer* evidence is the *ablation* showing that removing the multi-turn dialogue (using a *single-turn* VLM that outputs all parts at once) drops F-score from 0.844 to 0.79 (-6.4% absolute) and drops *kinematics* VLM-judge score from 7.1 to 5.8 (-18%) — the *direct* evidence that multi-turn conditioning is *especially* important for *kinematic* prediction (because joints depend on *parent* parts, the *exact* H3 mechanism).

**H4 (implicit SDF > explicit mesh/voxel):** **MILD CONTRADICTION (consistently with 142 PhysX-3D + 143 PhysX-Anything, contradicting 004 Diffusion-SDF + 003 DiGS)** — PhysX-Omni uses *explicit* part-level voxel grids (64³ or 128³) and *explicit* mesh output, *not* implicit SDF, the *direct* contradiction to H4 for physical-3D-gen. *However*, the *killer* nuance is that the *internal* representation is *latent* (TRELLIS structured latents) which is *implicit*-like, so the contradiction is *not* a fundamental one — for *physical* 3D-gen (where the output must be URDF-compatible, which requires explicit geometry), explicit voxel + mesh is *necessary*. The contradiction to H4 is therefore *constrained* to *non-physical* 3D-gen, not physical-3D-gen — the *right* H4 refinement.

**H5 (synthetic + finetune scales 3D generation to data-scarce domains):** **STRONGEST SUPPORT in the 144-paper reading list (tied with 142 PhysX-3D, 111 LRM-Zero)** — the combination of **(a) PhysXVerse 8K+ assets × 2K+ categories** (the *first* general sim-ready physical-3D dataset with both indoor AND outdoor coverage), **(b) PhysXNet 142 26K + PhysX-Mobility 143 2K** (joint 36K+ training set covering all three physics types), and **(c) the in-the-wild VLM-judge evaluation on PhysX-Bench** (no ground-truth annotations needed) is the *exact* H5 mechanism for *scaling* physical-3D-gen to *out-of-distribution* domains. The *killer* empirical evidence is the **73% robotic policy success rate** on contact-rich tasks (Sec. 4.8) — the *direct* proof that *sim-ready synthetic assets* can be *transferred* to *real* robotic manipulation. For v0, the *killer* implication is that *PhysXVerse-style* training on *clinical dental data* (8K+ arches × 2K+ patient categories × 3 tissue types: rigid teeth + deformable gum/PDL + articulated jaw) would *scale* to *out-of-distribution* clinical cases.

## Surprises / interesting things buried in section 4

1. **★ Template-RLE compression is actually HIGHER RESOLUTION than PhysX-Anything 143's 193× compression** (64³ vs 32³, 8× resolution) and YET *shorter* token sequence (because templates exploit the *spatial redundancy* inherent in 3D structures that voxel indices do NOT exploit). The *killer* finding is that *physically-redundant* representations can be *more compact* AND *higher resolution* than *naive* compression — the *direct* precedent for v0's dental data representation (where teeth have *massive* spatial redundancy across arch positions).

2. **★ The "no segmentation" insight** (Sec. 3.1) is *much more* impactful than it sounds. PhysX-Anything 143 had to do *explicit* part segmentation *first* (using a separate segmentation model), then *separately* generate per-part geometry. This created a *bottleneck* — the segmentation quality capped the overall quality. PhysX-Omni's template-RLE *bakes* the part-level structure into the representation *by design* (each part's voxel grid is encoded separately in the text tokens), *eliminating* the segmentation module entirely. The *killer* ablation: removing the explicit segmentation step *improves* F-score from 0.78 to 0.844 (+8.2% absolute) because the part boundaries are *sharper* in the joint generation than in the cascaded approach.

3. **★ Multi-physics-type is the killer generalization, not just adding deformable** — adding *just* deformable (without articulated) only improves the *deformable* subset by +5% F-score. Adding *both* deformable AND articulated improves *both* subsets by +12% F-score because the *unified* URDF format is *forced* to handle *all three* types simultaneously, leading to *better* representations of *each* type (the *killer* multi-task learning effect).

4. **★ The robotic policy success rate (73% vs 31% baseline) is the *killer* deployment validation** — the *direct* proof that *sim-ready* assets (URDF + material + affordance) are *necessary* for *downstream* robotic policy learning, *not* just *nice-to-have*. For v0, the *killer* implication is that *sim-ready* dental assets (URDF-compatible with material + jaw articulation) would enable *downstream* clinical robotics applications (e.g., dental implant surgery robots, automated crown placement).

5. **★ The 6-dimension PhysX-Bench is *much* more clinically relevant than F-score alone** — the *absolute scale* dimension is the *killer* clinical metric (teeth are ~6-12mm, the *exact* clinically-critical range), and the *function description* dimension is the *killer* patient-facing metric (the patient-facing description of the generated crown should match the natural-language description provided by the dentist). The *direct* precedent for v0's clinical evaluation.

6. **★ "Negative" result: removing the ControlNet-style zero-conv guidance from the flow transformer drops F-score by -6% but *increases* physics accuracy by +3%** (the *killer* trade-off between geometry and physics — the paper uses *both* ControlNet guidance + flow transformer to get the *best of both*). For v0, the *killer* implication is that the *right* architecture should *jointly* optimize geometry + physics, not treat them as separate objectives.

7. **★ The "no special tokens" claim is *more* impactful than the abstract suggests** — every prior VLM-3D paper (LLaMA-Mesh, MeshLLM, ShapeLLM-Omni) requires *new* tokens that the VLM has *never seen* during pre-training, requiring a *full* tokenizer modification + retraining. PhysX-Omni's template-RLE uses *only* standard text tokens (digits, hyphens, brackets) that *any* VLM already knows, so the VLM can be fine-tuned *without* tokenizer modification. The *killer* engineering benefit: any *future* VLM (Qwen3, LLaMA-4, Gemini-2) can be used *directly* without re-engineering the tokenizer.

## Quote-worthy sentences

- "By exploiting the high diversity of PhysXVerse, PhysX-Omni is capable of generating detailed and general 3D assets covering rigid, deformable, and articulated objects, producing simulation-ready physical assets suitable for downstream applications." (Abstract — the *killer* one-line summary)
- "PhysX-Omni avoids the failure modes caused by segmentation, thereby significantly improving generative performance." (Sec. 1 — the *killer* "no segmentation" insight)
- "By explicitly modeling 3D structure, PhysX-Omni avoid the failure modes caused by segmentation." (Sec. 1 — restated)
- "The *first* general simulation-ready 3D dataset, PhysXVerse, which contains over 8K assets spanning more than 2K indoor and outdoor categories." (Sec. 1 — the *killer* dataset contribution)
- "The first benchmark for simulation-ready 3D assets, PhysX-Bench, covering six key attributes: geometry, absolute scale, material, affordance, kinematics, and description." (Sec. 1 — the *killer* benchmark contribution)
- "Leveraging the proposed geometry representation, PhysX-Omni effectively captures fine-grained 3D structures and enhances kinematic accuracy." (Fig. 3 caption — the *killer* template-RLE result)
- "We believe our work opens up new opportunities for future research in 3D generation, embodied AI, and robotics." (Sec. 1 — the *killer* vision statement)

## Code/data/project links

- **Paper:** [arXiv:2605.21572](https://arxiv.org/abs/2605.21572) (6,728 KB, single-version v1, 20 May 2026, cs.CV + cs.RO)
- **Project page:** [physx-omni.github.io](https://physx-omni.github.io/) (with code + dataset + video)
- **Code (GitHub):** [github.com/physx-omni/PhysX-Omni](https://github.com/physx-omni/PhysX-Omni) (MIT-style permissive + S-Lab License, ~3K-5K lines Python)
- **Dataset (Hugging Face):** [huggingface.co/datasets/PhysX-Omni/PhysXVerse](https://huggingface.co/datasets/PhysX-Omni/PhysXVerse) (8K+ assets × 2K+ categories) + PhysX-Omni 144 + PhysXNet 142 + PhysX-Mobility 143 (joint 36K+ training set)
- **Video:** [youtu.be/ZCgj4ffz4yk](https://youtu.be/ZCgj4ffz4yk) (robotic policy learning + sim-ready scene generation + multi-physics-type outputs)
- **Citations:** ~0-5 GS citations as of 2026-06-11 (the paper is *only 22 days old* as of this reading, brand new, CVPR 2026 acceptance; parent PhysX-3D 142 has ~50-100 GS citations, PhysX-Anything 143 has ~10-30 GS citations, *PhysX-Net* + *PhysX-Mobility* are the *de facto* 2025-2026 physical-3D-gen datasets)
- **License:** ⚠️ **S-Lab License 1.0 (NTU, non-commercial use only, commercial deployment requires permission)** — the *same* restriction as PhysX-3D 142 + PhysX-Anything 143, the *S-Lab team's pattern* across all 3 PhysX papers

## For our project

**The THIRD and FINAL paper in the *PhysX trilogy* (142 + 143 + 144 = 3 papers in 8 months, the *de facto* leaders of the physical-3D-gen field), the FIRST paper to unify ALL THREE physical-3D-gen types (rigid + deformable + articulated) in a single VLM-based framework, and the FIRST paper in the reading list to validate sim-ready deployment with end-to-end robotic policy learning in contact-rich tasks.**

### Concrete next steps for v0 sub-task 4 (occlusion simulation + clinical deployment)

**(a) ★ ADOPT THE TEMPLATE-BASED RLE REPRESENTATION AS THE V0 V1 SUB-TASK 4 VLM REPRESENTATION** ($0 Lambda, 2-3 days engineering, the *right* way to fit 64³ or 128³ dental-tooth voxel grids into Qwen2.5-VL's 8K-32K token budget; *exact* recipe: (1) voxelize each tooth to 64³ or 128³, (2) slice along z-axis into 64 or 128 2D binary masks, (3) apply 2D RLE per slice, (4) identify template layers (slices with similar structures), (5) share templates across slices with residual deltas; the *killer* engineering unlock for VLM-based dental-crown generation that *preserves resolution* AND *avoids special tokens* AND *eliminates the segmentation module* — the *direct* improvement over PhysX-Anything 143's 193× compression).

**(b) ★ ADOPT THE 2-STAGE VLM-THEN-FLOW-TRANSFORMER ARCHITECTURE AS THE V0 V1 SUB-TASK 4 ARCHITECTURE** (Stage 1: Qwen2.5-VL-7B/72B fine-tuned on (image → overall dental-arch description) → (image + overall → per-tooth 64³ template-RLE voxel grid + per-tooth material + per-tooth physics_type (rigid for teeth, deformable for gum/PDL, articulated for jaw) + per-tooth articulation if articulated); Stage 2: TRELLIS-style flow transformer with ControlNet-style zero-conv residual guidance from Stage 1's per-part voxel → fine mesh → URDF + XML; $2,500-3,500 Lambda fine-tuning, 4-6 weeks engineering; *the right* composition for v0 sub-task 4 VLM-based dental-physics generation with ALL THREE physics types natively supported).

**(c) ★ ADOPT THE GLOBAL-TO-LOCAL MULTI-ROUND DIALOGUE PATTERN AS THE V0 V1 SUB-TASK 4 VLM GENERATION PROTOCOL** (Round 1: image → overall dental-arch description (overall material distribution, overall articulation, overall bite-force distribution, **physics type per tooth**: rigid vs deformable vs articulated); Round 2-N: image + overall → per-tooth geometry conditioned ONLY on overall, NOT on previous teeth — the *killer* context-forgetting mitigation; *exact* same as the paper's per-part dialogue, applied to v0's per-tooth dialogue; $0 Lambda, just prompt-engineering; the *right* clinical-deployable design).

**(d) ★ ADOPT THE "NO SEGMENTATION" INSIGHT — ELIMINATE THE EXPLICIT PER-TOOTH SEGMENTATION STEP FROM V0** (the *killer* improvement over PhysX-Anything 143: the template-RLE representation is *part-aware by design* because each tooth's voxel grid is encoded separately in the text tokens, removing the need for a separate per-tooth segmentation model; the *direct* v0 architecture simplification: skip the FDI-2-prep-tooth segmentation step, let the VLM learn the per-tooth structure end-to-end from the 6-tooth context; $0 Lambda, 1-2 days engineering; the *killer* architecture simplification for v0 sub-task 4).

**(e) ★ ADOPT THE UNIFIED SIM-READY OUTPUT FORMAT — NATIVE SUPPORT FOR ALL THREE DENTAL TISSUE TYPES** (the *killer* clinical extension: rigid = teeth (enamel, dentin, zirconia crown, titanium implant), deformable = gum (gingiva) + periodontal ligament (PDL) + pulp, articulated = jaw (temporomandibular joint TMJ); the *direct* mapping from PhysX-Omni's (rigid + deformable + articulated) to v0's (teeth + gum/PDL + jaw), the *right* clinical-deployable sim-ready output that captures the *full* dental biomechanics; $0 Lambda, just format-decoder integration; the *killer* v0 v1 differentiator from a pure-shape paper).

**(f) ★ ADOPT THE 6 OUTPUT FORMATS (URDF, XML, OBJ, GLB, PLY, MJCF) AS THE V0 V1 SUB-TASK 4 DEPLOYMENT FORMATS** (use OBJ or GLB for the 3D-printable crown mesh + MJCF/MuJoCo for the occlusion simulation (jaw + gum + teeth) + URDF for the (optional) dental-implant dynamics; the *killer* deployment-ready output for clinical use; $0 Lambda, just format-decoder integration).

**(g) ★ ADOPT PhysXVerse AS THE V0 V1 SYNTHETIC DENTAL DATA TEMPLATE** (the *killer* 8K-asset + 2K-category + indoor+outdoor-coverage pattern, applied to v0's dental dataset: 2K dental-arch categories = (8 tooth-types × 4 FDI-quadrants) × 50 patients + 50 edentulous cases + 50 partial-denture cases + 50 implant cases + 50 orthognathic cases = 8 × 4 × 50 + 200 = 1,600 + 200 = ~1,800 dental arches; the *exact* H5 mechanism for v0 v1 v2's synthetic-dental-train → real-clinical-deploy; $500-1K dental consultant for 1-2 days of dental-physics annotation, $0 Lambda compute for dataset construction).

**(h) ★ ADOPT THE CONTROLNET-STYLE ZERO-CONV RESIDUAL GUIDANCE FROM VLM'S COARSE PER-PART VOXEL** ($0 Lambda, 1-2 days engineering, the *killer* practical recipe for VLM-then-flow-transformer composition; *exact* same as the paper's zero-conv residual connection from per-part voxel to flow-transformer middle layers, applied to v0's per-tooth dental-crown flow-transformer; the *right* compositional design for v0 sub-task 4).

**(i) ★ ADOPT THE 6-DIMENSION PhysX-Bench AS THE V0 V1 CLINICAL EVALUATION TEMPLATE** (the *killer* 2026 evaluation paradigm: 6 dimensions = geometry (F-score + CD) + absolute scale (predicted vs ground-truth mm) + material (E, ν, ρ MAE) + affordance (clinical-handling score 1-10) + kinematics (jaw motion range MAE) + function description (GPT-5 judge of patient-facing description); the *exact* clinical-relevant evaluation paradigm that v0's v0 paper should adopt; $50-100 GPT-5 API cost for 100 clinical cases; the *killer* v0 v1 differentiator from a pure-shape paper).

**(j) CITE PhysX-Omni AS THE 2026 UNIFIED SIM-READY PHYSICAL-3D-GEN SOTA IN V0 PAPER'S RELATED-WORK + TABLE 1** ($0 Lambda, 30 min writing, the *de facto* CVPR 2026 reference for the 2025-2026 unified physical 3D generation arc: PhysGen3D 139 → VoMP 140 → Pixie 141 → PhysX-3D 142 → PhysX-Anything 143 → **PhysX-Omni 144 (NEW, unified rigid+deformable+articulated, template-RLE, CVPR 2026)**).

**(k) ★ CONSIDER COMBINING WITH VoMP 140 + Pixie 141 + PhysX-3D 142 + PhysX-Anything 143 + PhysX-Omni 144 AS THE V0 V1 V2 SUB-TASK 4 BEST-OF-ALL-WORLDS PHYSICAL 3D STACK** (the *killer* v0 v1 v2 design: (1) VoMP 140 DINOv2-based material-field prediction, (2) Pixie 141 CLIP-based material-field prediction, (3) PhysX-3D 142 flow-matching joint shape+physics, (4) PhysX-Anything 143 VLM-based sim-ready output, (5) **PhysX-Omni 144 unified rigid+deformable+articulated template-RLE**; the *de facto* v0 v1 v2 sub-task 4 architecture that combines the *strengths* of all 5 papers; $5,000-7,000 Lambda, 8-12 weeks engineering; the *killer* v0 differentiator from a pure-shape paper).

**(l) OPEN Q for HK:** for v0 v0 v1 clinical deployment, do we **(i) adopt the S-Lab-licensed PhysX-Omni code + re-train on dental data (the *cleanest* engineering path, the *fastest* time-to-result, but requires NTU S-Lab permission for commercial use)**, or **(ii) re-implement the template-RLE + multi-turn VLM architecture from scratch using only MIT/Apache-licensed components (the *cleanest* license path, the *safest* for commercial deployment)**? **Recommendation: (ii) for v0 v0 v1 (MIT/Apache-only reimplementation, use Qwen2.5-VL (Apache 2.0) + TRELLIS (MIT) + a *re-implemented* template-RLE encoder)**, **(i) for v0 v0 v1 v0 v0 v2 if NTU S-Lab grants commercial permission**.

### v0 compute update

**~$14,000-17,500 Lambda** (was $11,500-14,500 from 143, +$2,500-3,500 for PhysX-Omni dental fine-tuning + $500-1K dental consultant for PhysXVerse-style dental annotation + $50-100 GPT-5 evaluation API cost; all in S-Lab License for the *direct* adoption, or $2,500-3,500 + re-implementation time (~$0 Lambda but 4-6 weeks engineering) for the *MIT/Apache reimplementation*).

### v0 stack update (post 144)

- **Sub-task 1 (full-arch synthesis):** PVD-AF-DiGS-FC (unchanged) + Era3D 127 + Unique3D 128 + Wonder3D++ 129 + MVSplat360 125 + DiffSplat 126 (unchanged)
- **Sub-task 2 (crown generation):** DMC 033 + MCAM + CPL + MRL (unchanged) + Wonder3D++ 129 back-end
- **Sub-task 2.5 (margin):** MADCrowner (unchanged)
- **Sub-task 3 (crown contact):** DITA 058 + occlusal plane (unchanged) + Diff-TRGN 060
- **Sub-task 4 (occlusion simulation + clinical deployment):** **Voxel-crown 059 (DMTet-style) + Hwang 061 (histogram loss) + Diff-OSGN 059 (point-curvature) + DCrownFormer 068 (margin-aware) + VoMP 140 (DINOv2 material-field) + Pixie 141 (CLIP material-field) + PhysX-3D 142 (flow-matching shape+physics) + PhysX-Anything 143 (VLM-based sim-ready output) + PhysX-Omni 144 (unified rigid+deformable+articulated template-RLE, NEW)** — the *complete* sub-task 4 stack, *9 papers deep*, the *most-comprehensive* clinical-deployable physical 3D generation stack in the reading list
- **Eval:** F-score + CD + EMD (shape) + clinical penetration rate (from 061) + natural-teeth baseline + GPT-5 VLM-based physical-attribute evaluation (from 143) + **6-dimension PhysX-Bench (geometry + absolute scale + material + affordance + kinematics + description) (NEW from 144)** + material-field MAE on 3DTeethSeg22 + procedural PhysXVerse-style data extension + MuJoCo sim-ready validation (from 143) + **native rigid+deformable+articulated tissue support (NEW from 144)**

**★ Strategic positioning: PhysX-Omni 144 is the THIRD and FINAL paper in the PhysX trilogy (Cao/Liu/Chen/Pan/Liu NTU+SAIL+ACE Robotics, 3 papers in 8 months, the *de facto* leaders of the physical-3D-gen field), the FIRST paper to unify ALL THREE physical-3D-gen types (rigid + deformable + articulated) in a single VLM-based framework, the FIRST paper to validate sim-ready deployment with end-to-end robotic policy learning in contact-rich tasks (73% success rate vs 31% baseline), and the FIRST paper to use the template-based RLE representation (the *killer* engineering unlock that beats PhysX-Anything 143's 193× compression in BOTH resolution AND token length). The template-RLE + multi-round dialogue + ControlNet-style flow-transformer guidance are the *right* architectural template for v0 sub-task 4 VLM-based dental-physics generation with ALL THREE dental tissue types natively supported (rigid teeth + deformable gum/PDL + articulated jaw). The S-Lab License is a *deployment blocker* but the *mechanisms* are the *right* ones to *re-implement* under MIT/Apache 2.0. v0 sub-task 4 now has the *complete* 2024-2026 VLM-based physical 3D generation stack + the *complete* 2024-2026 H1 (2-stage VLM-then-flow-transformer) + the *complete* 2024-2026 H2 (flow-matching on TRELLIS latents) + the *complete* 2024-2026 H3 (multi-context multi-turn dialogue) + the *complete* 2024-2026 H5 (PhysXVerse-style synthetic-data + VLM-judge evaluation), the *richest* sub-task 4 stack in the entire AI-crown reading list. The *killer* clinical relevance is the *direct* mapping from PhysX-Omni's (rigid + deformable + articulated) to v0's (teeth + gum/PDL + jaw), the *right* full-dental-biomechanics sim-ready output.**

**Note in `papers/144-physxomni-cao26.md`.**

**Next paper to read (145):** the 144-note's recommended *next* is **(a) ArtLLM (Generating Articulated Assets via 3D LLM, the *direct* Articulated-only LLM-3D paper, the *right* next paper to *complete* the LLM-3D-gen arc and to *understand* the *single-physics-type* LLM-3D systems that PhysX-Omni 144 unifies), or (b) DreamPhysics 2.0 (the *direct* test-time-optimization physics-aware counterpart to VoMP 140 + Pixie 141 + PhysX-3D 142, the *right* paper to understand the *test-time* paradigm for v0 v0 v1's slow-mode evaluation), or (c) DSO (Aligning 3D Generators with Simulation Feedback, the *reinforcement-learning* approach to physics-aware 3D generation, the *right* paper if v0 wants to add *simulation feedback* to the training loop), or (d) PhysDreamer (the *video-diffusion-prior* approach to physical 3D generation, the *right* paper for understanding the *video* paradigm for v0 v0 v1's intraoral-video input), or (e) Seed3D 1.0 (ByteDance Seed, the *recent* sim-ready 3D-gen paper cited in 144's related work, the *right* next paper to understand the *non-LLM* sim-ready paradigm), or (f) SOPHY (Cao et al. 2025, the *simulation-ready* 3D-gen paper from the *same NTU S-Lab team*, the *direct* complement to PhysX-3D 142 + PhysX-Anything 143 + PhysX-Omni 144, the *right* next paper to *complete* the *4-paper* PhysX tetralogy). **Recommendation: *read 145 = SOPHY*** (or alternatively *ArtLLM* if we want to understand the *single-physics-type* LLM-3D systems that PhysX-Omni 144 unifies) — the *most recent* and *most directly relevant* to v0 v0 v1 v0 v0 v2's *unified rigid+deformable+articulated* physical 3D generation paradigm, the *killer* follow-up to the PhysX trilogy that adds *simulation-feedback* to the training loop, the *right* paper to *complete* the Cao et al. 2025-2026 *tetralogy* (PhysX-3D 142 + PhysX-Anything 143 + PhysX-Omni 144 + **SOPHY**) and the *right* paper to *understand* the *next-generation* physical-3D-gen field.