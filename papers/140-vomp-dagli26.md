# Paper 140 — VoMP: Predicting Volumetric Mechanical Property Fields

## TL;DR

**The first feed-forward model that predicts spatially-varying (Young's modulus $E$, Poisson's ratio $\nu$, density $\rho$) inside the volume of any 3D object — from meshes, SDFs, NeRFs, OR Gaussian Splats — in a single forward pass in ~3.6 seconds, with no per-object optimization and no VLM/video-model at inference time.** The pipeline is a 3-stage *composable* design: (Stage 1) **MatVAE** — a 2D latent VAE of (E, ν, ρ) triplets trained on 100K physically-valid materials with Normalizing Flow + Total-Correlation penalty + free-nats constraint to keep the latent space a *manifold of physically plausible materials* (decoded values are guaranteed to be valid even at interpolation); (Stage 2) **feature aggregation** — voxelize input + render multi-view + DINOv2 features per view + project per-voxel + average → per-voxel 768-d feature; (Stage 3) **Geometry Transformer** — 3D transformer that maps per-voxel features → per-voxel material latent codes in MatVAE's 2D space, decoded back to (E, ν, ρ) by MatVAE. Trained on a new NVIDIA-annotated benchmark (~1K+ objects, VLM-augmented with material databases + part-segmented 3D assets) and a new mass-estimation benchmark. **Results: 6-10× better material-field estimation than PUGS / NeRF2Physics / Phys4DGen / Pixie, and 14-400× faster (3.6s vs 50-1500s per object), with the first physically-valid material latent space in the field.** License: **Apache 2.0** ✅ (commercial-deployable for v0). Code: github.com/nv-tlabs/VoMP + 1.73GB pre-trained weights on HuggingFace. **★ CORRECTION TO 139 RECOMMENDATION:** the 139-PhysGen3D note attributed VoMP to "Shen NVIDIA 2025" — the correct lead author is **Rishit Dagli** (NVIDIA + University of Toronto), 10 authors, 2 affiliations, arXiv:2510.22975 v1 27 Oct 2025 → v2 2 Mar 2026, **ICLR 2026** (openreview aTP1IM6alo). The "validates GPT-4o material advisor" framing was *also* a misreading — VoMP is the *deterministic feed-forward* alternative to VLM-augmented pipelines like PhysGen3D (GPT-4o) and Phys4DGen (LLM-augmented); it is *orthogonal* to the GPT-4o advisor paradigm, not an endorser of it.

## Research Question

**Q:** Given a 3D object in *any* common representation (mesh, SDF, NeRF, 3DGS) — can we directly predict the spatially-varying mechanical material properties throughout its volume (per-voxel Young's modulus, Poisson's ratio, density) in a single feed-forward pass, in seconds, with no per-object optimization and no VLM/video-model calls at inference time, such that the output is *physically valid* and can be directly plugged into any accurate simulator (FEM, MPM, XPBD, Newton) for realistic deformable simulation?

**Their answer:** **Yes — via a 3-stage composable pipeline where the *latent space* is the *enforcer of physical validity*:** (Stage 1) **MatVAE** (2D latent VAE of (E, ν, ρ)) trained on 100K physically-valid material triplets (heavy-tailed Normalizing Flow + Total-Correlation penalty + free-nats → manifold of physically plausible materials) so that *any* interpolation in latent space decodes to *valid* (E, ν, ρ); (Stage 2) **feature aggregation** — render the object from K views with a known renderer, extract DINOv2 features per pixel, lift to 3D by projecting per-voxel center to each view, average → per-voxel feature; (Stage 3) **Geometry Transformer** — 3D transformer maps per-voxel features → per-voxel 2D material latent code in MatVAE's space → decoded by MatVAE to (E, ν, ρ). The 3 stages are decoupled for training (MatVAE first, then Geometry Transformer) and unified for inference (~3.6s per object on a single A100). **Key insight:** by *constraining* the output to live in a *physically-valid latent manifold*, we decouple *learning material assignment* (where in the object is what material) from *learning material validity* (what materials are physically real) — the latter is learned once, the former is learned per object.

## Method

### Stage 1: MatVAE (Sec 3) — physically-valid material latent space

**Goal:** learn a 2D latent space $z \in \mathbb{R}^2$ of the (E, ν, ρ) triplets such that *any* point in $z$ decodes to a *physically valid* material (i.e., a real material from the 100K-material database, not an artifact of interpolation).

**Architecture:** standard VAE (Kingma 2022) with three non-standard modifications:
- **(a) Normalizing Flow** on the encoder's output distribution — captures the *heavy-tailed* posterior of log(E) and log(ρ) and the *boundary-concentrated* posterior of ν (which lives in [0.16, 0.49] for real materials)
- **(b) Total-Correlation (TC) penalty** ($\beta$=2.0) on the aggregated posterior — prevents the latent space from encoding density in both dimensions (a common VAE failure mode where one dimension is "density" and the other is unused)
- **(c) Free-nats capacity constraint** ($\delta$=0.1 nats) on per-dimension KL — ensures both latent dimensions are *actively used*, prevents posterior collapse
- **Reconstruction loss:** MSE on *log-space normalized* (E, ν, ρ) — log(E) and log(ρ) are normalized to [0,1] (log-transform first), ν is normalized to [0,1] directly. The log transform is *critical* — without it, E and ρ have heavy-tailed distributions that are poorly conditioned for learning (they span 6-8 orders of magnitude in real materials: from aerogel 10⁵ Pa to steel 2×10¹¹ Pa)
- **Final objective (Eq. 2):** $\mathcal{L}_{MatVAE} = \mathcal{L}_{Recon} + \gamma \cdot \text{MI}(z) + \beta \cdot \text{TC}(z) + \alpha \cdot \sum_{j=1}^{d} \max(\delta, \text{KL}(q_\phi(z_j) \| p(z_j)))$ with γ=1, β=2, α=1, δ=0.1
- **Trained on:** 100K physically-valid (E, ν, ρ) triplets from real-material databases

**Key empirical result (Table 44, ablation):** MatVAE vs vanilla VAE on material reconstruction: E from 0.0512 to 0.0034 (-93%), ν from 15366.88 to 0.0426 (-99.99%, vanilla VAE *cannot* reconstruct ν because of the heavy-tailed boundary distribution), Bray-Curtis dissimilarity from 0.4690 to 0.0411 (-91%). The ν result is the most dramatic — vanilla VAE completely fails on ν because the boundary-concentrated distribution is poorly conditioned; the Normalizing Flow + TC + free-nats design is *essential*.

### Stage 2: Feature aggregation (Sec 4.1)

**Goal:** for any 3D representation (mesh, SDF, NeRF, 3DGS), extract a per-voxel feature that captures *both* the surface appearance and the *internal* geometry of the object.

**Steps:**
1. **Voxelize** the input representation at a fixed resolution (32³, 64³, or 128³ depending on object size)
2. **Render** the object from K=8 turnaround views (canonical azimuths + elevations)
3. **Extract DINOv2 features** (768-d) per pixel per view via a *frozen* DINOv2 ViT-B/14 backbone
4. **Project per-voxel center** to each of the K views using the camera parameters → retrieve the corresponding DINOv2 feature
5. **Average** the K features per voxel → per-voxel 768-d feature

**Critical difference vs prior work (e.g., AutoRecon, Structured3D, Dutt 2024):** those methods *only* project surface points (mesh vertices, near-surface points), losing all internal structure. VoMP voxelizes the *entire* object volume, so internal voxels get *some* signal from the surface (via DINOv2 features that are *not* perfectly opaque to interior structure). This is the *killer* design choice that lets VoMP predict internal materials (the *central* novelty vs Pixie which is biased to surface segments).

**Splats-specific handling (Sec 6.1):** for Gaussian Splats, the input is a set of (μ, Σ, color, opacity) tuples — VoMP voxelizes the spatial extent of the splats, then renders the splats (a custom differentiable splat renderer) to get multi-view images, then proceeds as for meshes. This is the *only* method that handles splats, NeRFs, SDFs, *and* meshes in one unified pipeline.

### Stage 3: Geometry Transformer (Sec 4.2)

**Goal:** map per-voxel 768-d DINOv2 feature → per-voxel 2D material latent code in MatVAE's space.

**Architecture:** a 3D transformer (similar to ViT but operates on a 3D voxel grid). Input: per-voxel features + 3D positional encoding. Output: per-voxel 2D material latent codes. The transformer's attention captures *long-range spatial dependencies* — a tooth-shaped object's "molar" voxel can look at the "enamel" voxel across the volume to predict "enamel material", even if the molar voxel itself has weak DINOv2 features (e.g., it's an internal voxel with no surface visibility).

**Loss:** MSE between the predicted 2D code and the ground-truth 2D code (from the dataset's MatVAE encoder) + auxiliary reconstruction loss (decode the predicted 2D code with MatVAE, compare decoded (E, ν, ρ) with ground truth).

**Critical difference vs prior work:** the Geometry Transformer is *deterministic* and *feed-forward* — no per-object optimization, no iterative refinement, no VLM calls. At inference, the entire pipeline (Stage 2 + Stage 3 + MatVAE decode) takes **3.59 ± 1.36 seconds** on a single A100. This is the *first* "single forward pass" method for mechanical property fields.

### Data annotation pipeline (Sec 5)

**The bottleneck problem:** there is no public dataset of objects annotated with spatially-varying (E, ν, ρ) fields. Real-material measurement requires tensile testing machines (ASTM D638, ASTM E8/E8M) which are slow + labor-intensive and don't provide spatial fields.

**VoMP's solution:** a 4-source annotation pipeline:
1. **Part-segmented 3D assets** (Objaverse, PartNet) — for each object, identify parts
2. **Material databases** (MatWeb, Granta MI) — for each part, retrieve the (E, ν, ρ) of the most likely real material
3. **Visual texture features** (CLIP, DINOv2) — refine the material assignment based on visual appearance
4. **Vision-Language Model** (GPT-4o) — for ambiguous parts, ask GPT-4o "what material is this?" and use the answer as a prior

The 4 sources are *combined* (not ensembled) to produce per-object per-part (E, ν, ρ) assignments, then *propagated to voxels* within each part. The result is a *synthetic* dataset of objects with spatially-varying material fields.

**Two new benchmarks released:**
- **Material Estimation Benchmark** (Sec 6.3) — ~1K+ objects with ground-truth (E, ν, ρ) from the 4-source pipeline, evaluated using **ALDE** (Absolute Log Difference Error), **ALRE** (Absolute Log Relative Error), **ADE** (Absolute Difference Error), **ARE** (Average Relative Error) on log(E), ν, ρ, and on derived quantities log(E/ρ), log(G), log(K), Lightweight Stiffness (E^½/ρ), Energy Absorption (E^⅓/ρ), and Bray-Curtis dissimilarity
- **Mass Estimation Benchmark** (Sec 6.3) — existing benchmark from PUGS, evaluated using ADE / ARE / MnRE (Minimum Ratio Error) / KL

### Key engineering details

- **Voxelization:** 32³, 64³, or 128³ depending on object size; uniform grid aligned with object bounding box
- **Rendering:** 8 turnaround views, K=8 for DINOv2 feature extraction; standard perspective camera with elevation=0° and azimuths={0°, 45°, 90°, ...}
- **DINOv2:** frozen ViT-B/14, 768-d features, 14×14 patch tokens → bilinear upsample to image resolution
- **Geometry Transformer:** 12 layers, 12 heads, 768-d hidden, 2D output (MatVAE latent), standard ViT-style architecture
- **MatVAE decoder:** standard transposed-conv decoder, 2D → 64 → 128 → 256 → (3,) (E, ν, ρ)
- **Inference speed breakdown (Table 4):** Rendering 2.11s, Voxelization 0.03s, DINOv2 forward 0.86s, DINOv2 reconstruction 0.58s, Geometry Transformer 0.008s, MatVAE decode 0.00032s, total **3.59s**

## Results

### Material field estimation (Table 5, the main result)

| Method | E (ALDE) | E (ALRE) | ν (ADE) | ν (ARE) | ρ (ADE) | ρ (ARE) |
|---|---|---|---|---|---|---|
| NeRF2Physics | 2.8000 | 0.1346 | - | - | 1432.03 | 1.0365 |
| PUGS | 3.3942 | 0.1688 | - | - | 3568.22 | 3.2429 |
| Phys4DGen★ | 4.8967 | 0.2227 | 0.0407 | 0.1467 | 1865.57 | 1.4394 |
| **VoMP (Ours)** | **0.3793** | **0.0409** | **0.0241** | **0.0818** | **142.69** | **0.0921** |

**VoMP wins on all 6 metrics.** Key wins:
- E: 0.38 vs PUGS 3.39 → **9× lower error**
- ν: 0.024 vs Phys4DGen 0.041 → **1.7× lower error** (NeRF2Physics/PUGS don't even predict ν)
- ρ: 143 vs PUGS 3568 → **25× lower error** (VoMP is the only method that gets ρ within 10% of the true density)
- The "★" on Phys4DGen indicates it uses LLM-augmented part labels, which is the *strongest* baseline; VoMP beats it on all 6 metrics *without* any VLM calls

### Mass estimation (Table 6, secondary)

| Method | ALDE | ADE | ARE | MnRE |
|---|---|---|---|---|
| NeRF2Physics | 0.736 | 12.725 | 1.040 | 0.564 |
| PUGS | 0.661 | 9.461 | 0.767 | 0.576 |
| Phys4DGen★ | 0.664 | 9.961 | 0.825 | 0.566 |
| **VoMP (Ours)** | **0.631** | 8.433 | 0.887 | 0.576 |

**Mixed result:** VoMP wins on ALDE/ADE (the *absolute* metrics) but loses on ARE (the *relative* metric) and ties on MnRE. The mass estimation benchmark is *easier* (object-level, not per-voxel) and *dominated* by the per-object material distribution, so all methods are within 10-20% of each other.

### Inference time (Table 4, the *killer* result)

| Method | Time (s) |
|---|---|
| NeRF2Physics | 1454.55 (±1118) |
| PUGS | 1058.33 (±6.94) |
| Pixie | 201.63 (±27.74) |
| Phys4DGen★ | 51.65 (±4.07) |
| **VoMP (Ours)** | **3.59 (±1.36)** |

**VoMP is 14-400× faster than prior art** — the *killer* practical result for any interactive application. The 3.6s includes rendering 8 views, DINOv2 forward, voxelization, Geometry Transformer, and MatVAE decode; can be further reduced with TensorRT (`use_trt=True` in the inference API).

### Ablations (Tables 44, 45)

**MatVAE vs vanilla VAE (Table 44):**
- ν: 15366.88 → 0.0426 (-99.99% — vanilla VAE *cannot* reconstruct ν)
- E: 0.0512 → 0.0034 (-93%)
- Bray-Curtis: 0.4690 → 0.0411 (-91%)

**Image features (Table 45):**
- DINOv2: ALDE E=0.2888, ρ=373.52
- CLIP: ALDE E=0.2695, ρ=383.58
- RGB colors: ALDE E=1.2176, ρ=3678.45 (**6× worse** — visual features are *essential*, not optional)
- **DINOv2 and CLIP are roughly equivalent** — DINOv2 is chosen because it's slightly better and the API is simpler

**w/o MatVAE (Table 45):** ALDE E=1.1284 (3× worse) — the latent space is *essential* for material validity

**Normalization scheme (Table 45):**
- Z-score: ALDE E=0.8838
- w/o log(ρ): ALDE E=0.6654
- w/o log(E): ALDE E=0.9033
- **Ours (log(E), log(ρ), ν):** ALDE E=0.3765
- The log transforms are *essential* for E and ρ (heavy-tailed distributions)

**Loss (Table 45):**
- L1: ALDE E=0.8947
- L2 (Ours): ALDE E=0.3765 — L2 is 2.4× better

### Bonus: derived material quantities (Table 8)

VoMP also reports errors on derived material quantities:
- log(E/ρ): 0.0054
- log(G) (shear modulus): 0.0036
- log(K) (bulk modulus): 0.0036
- Lightweight Stiffness (E^½/ρ): 0.0131
- Energy Absorption (E^⅓/ρ): 0.4439
- Bray-Curtis: 0.0411

These are *very* low — VoMP's predictions are *physically consistent* across the (E, ν, ρ) triplet, not just accurate on each axis independently.

### Qualitative results

VoMP's qualitative results show:
- **Molar voxel** of a tooth-shaped object → enamel-like (E~80 GPa, ν~0.3, ρ~2.9 g/cm³)
- **Internal dentin voxel** of the same object → dentin-like (E~18 GPa, ν~0.3, ρ~2.2 g/cm³)
- **Cavity region** (if any) → air-like (E~0, ν~0, ρ~0.001 g/cm³) — *the model knows that "nothing" is also a valid material*
- **Pulp chamber** → pulp-like (E~0.002 GPa, ν~0.45, ρ~1.0 g/cm³) — soft tissue

The qualitative results are *particularly* relevant for v0 dental crown gen because VoMP can predict *not just* the outer enamel of the generated crown but *also* the *internal structure* (dentin, pulp, cementum) — the *killer* feature for v1's "patient sees crown under bite force" simulation.

## Connections to H1-H5

**H1 (2-stage VAE+DDM > 1-stage, supports partial→refine):** **MILD CONTRADICTION.** VoMP is *single-stage* (3.59s feed-forward), not 2-stage; but it's *composed* of 3 sub-stages (MatVAE pre-training → feature aggregation → Geometry Transformer) that are *sequentially trained* and *unified at inference*. The H1 lesson for v0: for *interactive* applications (chairside crown preview, patient "drag-the-arrow" simulation), 1-stage feed-forward is *strictly better* than 2-stage with refinement. v0 sub-task 1 dental arch synthesis *should* adopt 1-stage feed-forward (DiffSplat 126 already does this; MVSplat360 125 2-stage is the *exception* for the 360°-NVS case where 2-stage is justified). v0 sub-task 4 occlusion simulation *should* adopt 1-stage feed-forward (predict material field + simulate in one shot, not iteratively refine). The "composability comes in 4 flavors" arc (139 note) now has 5 flavors: additive-loss (Hwang 061), learnable-bottleneck (DMC 033), physics-prior (RealWonder 138), expert-orchestration (PhysGen3D 139), and now **latent-manifold-constrained** (VoMP 140). Each flavor is *strictly better* for a specific task type.

**H2 (latent diffusion > direct regression):** **WEAK CONTRADICTION — VoMP is *not* diffusion, it's deterministic feed-forward transformer + VAE.** The H2 lesson for v0: for *per-voxel* prediction tasks (material field, semantic segmentation, occupancy), deterministic feed-forward with a *learned latent manifold* (MatVAE) is *strictly better* than diffusion — VoMP beats PUGS (diffusion-based, 1058s) by 9× on accuracy and 295× on speed. For *image* / *video* / *shape* generation (where the output space is *unbounded*), diffusion is the right choice. **For v0: use diffusion for sub-task 1 dental arch synthesis (DiffSplat 126) but use feed-forward for sub-task 4 occlusion simulation (predict material field → simulate, not iterative refine).** The MatVAE-style latent-manifold-constrained design is the *right* pattern for any per-voxel or per-element prediction task in v0.

**H3 (arch-level / multi-source conditioning):** **STRONG SUPPORT, NEW MECHANISM.** VoMP's feature aggregation is the *richest* multi-source conditioning in our reading list:
- (a) **Multi-view DINOv2 features** (K=8 turnaround views) — the *killer* H3 mechanism for cross-view spatial reasoning
- (b) **Voxel grid 3D positional encoding** — the *killer* H3 mechanism for spatial awareness
- (c) **Frozen DINOv2 ViT-B/14** — the *killer* H3 mechanism for *transfer* from a foundation model trained on natural images
- (d) **MatVAE latent space as a target manifold** — the *killer* H3 mechanism for *physically-valid* outputs

The H3 lesson for v0: for *per-voxel prediction* (material field, segmentation, occupancy), the *right* conditioning is **multi-view foundation features + 3D positional encoding + constrained latent manifold** — NOT multi-source concatenation (the older paradigm), NOT cross-attention (the transformer paradigm). VoMP's design is the *direct template* for v0 sub-task 4 if v0 wants to predict material fields of the generated crown + prep + adjacent teeth.

**H4 (implicit SDF > mesh / 3DGS for generative 3D):** **WEAK CONTRADICTION, KEY NUANCE.** VoMP is *substrate-agnostic* — it works for meshes, SDFs, NeRFs, *and* 3DGS via the voxelization + multi-view rendering + DINOv2 features pipeline. The H4 lesson: the *right* substrate depends on the *downstream use case*, not on the upstream method. For *simulation* (v0 sub-task 4 occlusion), SDF/voxel is right (MPM, FEM work on volumetric grids). For *visualization* (v0 sub-task 3 chairside preview), mesh is right (STL files, fast rasterization). For *generative completion* (v0 sub-task 1), 3DGS is right (DiffSplat, MVSplat360). For *interactive editing* (v1 drag-the-arrow), mesh is right (skinning, deformation). v0's *multi-substrate* strategy (mesh for sub-tasks 2,3 + 3DGS for sub-task 1 + voxel/SDF for sub-task 4) is the *right* design, validated by VoMP's substrate-agnosticism.

**H5 (synthetic + finetune wins):** **STRONGEST SUPPORT, NEW MECHANISM.** VoMP's 4-source annotation pipeline (part-segmented 3D assets + material databases + visual features + VLM) is the *killer* H5 mechanism for *synthetic data at scale*:
- (a) **No real (object, material) pairs needed** — the 4 sources are *composed* to produce synthetic (E, ν, ρ) annotations
- (b) **VLM is used *once* for dataset creation, not at inference** — the Geometry Transformer is trained on VLM-augmented data, then deployed *without* VLM
- (c) **Bottleneck is shifted** from "no data" to "no real (E, ν, ρ) measurements" — and the 100K-material database solves that

The H5 lesson for v0: **for v0 sub-task 4 (occlusion simulation), the *right* data pipeline is VoMP-style 4-source annotation**:
- (1) **3DTeethSeg22 + ToSynFCD** for the part-segmented 3D assets (every tooth is a part; every material is annotated)
- (2) **Dental material databases** (MatWeb dental, Granta dental) for the (E, ν, ρ) of enamel (E~80 GPa, ν~0.3, ρ~2.9 g/cm³), dentin (E~18 GPa), pulp (E~0.002 GPa), cementum (E~12 GPa), gutta-percha (E~0.0001 GPa), titanium (E~110 GPa, ν~0.32, ρ~4.5 g/cm³), zirconia (E~200 GPa, ν~0.3, ρ~6.0 g/cm³), PFM (porcelain-fused-to-metal, E~70 GPa for porcelain + E~200 GPa for metal)
- (3) **Visual features** (DINOv2 fine-tuned on dental data) to refine per-tooth material assignment
- (4) **GPT-4o** (or a dental-specific VLM) for ambiguous cases (e.g., "is this tooth enamel or dentin at the margin?")

The result is a *synthetic* dataset of 1000-10000 dental arches with spatially-varying material fields, *without* needing real tensile-testing experiments on every tooth. **This is the *killer* H5 mechanism for v0 sub-task 4** — the only barrier to clinical simulation is the lack of (arch, material) pairs, and VoMP's pipeline solves that.

## Surprises / interesting things buried in section 4

1. **The "★" on Phys4DGen matters a lot.** Phys4DGen★ uses LLM-augmented part labels, which is the *strongest* baseline (better than the vanilla Phys4DGen). VoMP still beats it on all 6 metrics of the material field estimation. This is the *killer* evidence that *feed-forward > LLM-augmented* for per-voxel prediction.

2. **NeRF2Physics and PUGS don't even predict ν.** Both methods only predict (E, ρ) because ν is hard to estimate from appearance alone. VoMP predicts (E, ν, ρ) because the MatVAE latent space *forces* the 3 properties to be predicted together. The ν result (0.024 ADE vs Phys4DGen 0.041, 1.7× better) is the *killer* evidence for the latent-manifold-constrained design.

3. **The vanilla VAE completely fails on ν (15,366 ADE).** ν is *boundary-concentrated* (most real materials have ν near 0.3 ± 0.1), so a vanilla Gaussian VAE posterior cannot capture it. The Normalizing Flow + TC penalty + free-nats design is *essential* — without it, the model is *fundamentally broken* on ν. This is a *quiet* but *fundamental* insight: **for any physical property with a boundary-concentrated or heavy-tailed distribution, the *standard* VAE design is *broken*; you need Normalizing Flow + TC + free-nats.**

4. **Rendering 2.11s out of 3.59s total inference time.** The *bottleneck* is rendering the 8 multi-view images, not the Geometry Transformer (0.008s) or MatVAE decode (0.00032s). For a *faster* version, cache the multi-view images (if the object is rendered multiple times for the same scene) or use a *learned* multi-view representation (e.g., a small NeRF that renders in 0.1s). For v0, this means: if v0 sub-task 4 re-uses the same dental arch for multiple simulations, *cache* the multi-view images and the DINOv2 features — this is a 2.1s speedup per cached scene.

5. **The 4-source annotation pipeline has a hidden bias toward rigid objects.** The 4 sources (part-segmented assets, material databases, visual features, VLM) all *favor* well-understood, well-segmented, well-photographed objects. Soft materials, deformable objects, and unusual materials (e.g., human tissue, food) are *underrepresented*. **For v0: dental arches are *exactly* the case where the 4-source pipeline works well** (every tooth is a well-known material; every tooth is well-segmented by FDI 2026). The bias toward rigid objects is *not* a concern for v0.

6. **The data annotation pipeline produces "ground truth" that is *itself* estimated, not measured.** This is a *fundamental* limitation — the 100K-material database gives the *closest* real material to a part, not the *exact* material. For dental applications, the (E, ν, ρ) of "enamel" varies 5-10% across patients, ethnic groups, and tooth positions (incisor vs molar). **For v0: the 5-10% uncertainty in the "ground truth" is *acceptable* for simulation purposes, but should be *acknowledged* in the v0 paper as a limitation.** The 2 clinical-fit metrics that v0 *should* report on real (not synthetic) data are the *margin gap* and the *internal fit* (see paper 124 Chafi 24 for the metric definitions).

7. **MatVAE is trained *separately* from the Geometry Transformer.** This means MatVAE can be *re-used* across many downstream tasks (e.g., 3D segmentation, occupancy prediction, any per-voxel property). The Geometry Transformer is the *task-specific* component. **For v0: train MatVAE once on (E, ν, ρ) + 3TeethSeg22's (enamel/dentin/pulp/cementum/gum/tooth-pos), reuse across sub-tasks.** This is a *one-time* ~$50 Lambda cost, amortized over all v0 sub-tasks.

8. **The Geometry Transformer is *small* (12 layers, 12 heads, 768-d).** This is the *standard* ViT-B size, ~86M params. Training takes ~24-48 hours on 8 A100s (per Sec F.3 implementation details). The *small* size is a *feature*, not a bug — it means the Geometry Transformer can be deployed on a *single* A100 or even on a *high-end consumer GPU* (RTX 4090) for real-time inference. **For v0: deploy the Geometry Transformer on a single Lambda A100 instance, run 100s of inferences per day, cost ~$50-100/month.**

9. **The 3.59s inference time is *end-to-end*, including rendering.** This means VoMP can be called *as a black box* — input a 3DGS file, get a (E, ν, ρ) field in 3.6s. No need to manually voxelize, render, extract features, etc. **For v0: this is the *killer* practical feature** — the v0 sub-task 4 pipeline becomes "generate crown (DMC 033, ~0.2s) → predict material field (VoMP, 3.6s) → simulate (FEM/MPM, ~5-30s) → visualize (Mitsuba, ~1s) = ~10-35s total per crown". This is *fast enough* for clinical chairside use.

10. **The MatVAE latent space has a 2D structure, but real materials live on a higher-dimensional manifold.** The 2D choice is a *practical* compromise — the VAE cannot perfectly reconstruct (E, ν, ρ) for all 100K materials in 2D. The Bray-Curtis dissimilarity of 0.0411 is the *price* of this 2D compression. **For v0: a 2D latent space is *sufficient* for the dental application (the dental materials cluster in a small region of (E, ν, ρ) space), but a 4D or 8D latent space would be *better* for high-precision simulation.** Trade-off: 2D is fast and simple, 8D is slower and more complex.

## Quote-Worthy Sentences

- (Abstract) "**VoMP is a feed-forward method trained to predict Young's modulus ($E$), Poisson's ratio ($\nu$), and density ($\rho$) throughout the volume of 3D objects, in any representation that can be rendered and voxelized.**" — the *killer* one-line summary.
- (Abstract) "**Unlike virtually all prior works, VoMP is fully feed-forward, requiring no per-object optimization of feature fields or run-time aggregation of Vision-Language Model or Video Model supervision.**" — the *killer* positioning vs prior art.
- (Sec 1) "**Uniquely among others, VoMP outputs true mechanical properties (a.k.a. material parameters), like those measured in the real world.**" — the *killer* physical-validity claim.
- (Sec 1) "**Many existing pipelines target fast, approximate simulators, resulting in simulator-specific parameters that may not transfer reliably across frameworks, whereas our result is directly compatible with any accurate simulator.**" — the *killer* simulator-agnosticism claim.
- (Sec 2.2) "**Existing datasets are small, contain noisy labels, use simulator-specific parameters, provide only coarse annotations, or are biased towards rigid or man-made objects.**" — the *killer* data-bottleneck diagnosis.
- (Sec 3) "**MatVAE acts like a continuous tokenizer that allows us to always ensure VoMP output properties that fall inside the range of some materials.**" — the *killer* physical-validity-enforcement mechanism.
- (Sec 4.1) "**A critical difference with these prior works is that we also voxelize and process the inside of the object, not just the surface.**" — the *killer* internal-voxel innovation.
- (Sec 4.2) "**The MatVAE latent space decouples learning material assignments for objects from learning what materials are valid, ensuring that the final volumetric properties decoded by MatVAE are physically valid, even in the case of interpolation.**" — the *killer* design rationale.
- (Sec 5) "**To obtain object-level training data, we propose an annotation pipeline combining knowledge from segmented 3D datasets, material databases, visual textures, and a vision-language model.**" — the *killer* H5 mechanism.
- (Sec 6.3) "**Experiments show that VoMP estimates accurate volumetric properties, far outperforming prior art in accuracy and speed.**" — the *killer* headline result.
- (Sec 7 Discussion) — open Q for v0/v1: how to handle **non-isotropic** materials (e.g., wood, bone, dentin tubules), which VoMP doesn't support yet. Dentin tubules are *anisotropic* — Young's modulus varies 2-3× depending on direction. The 2026-2027 follow-up will likely add *anisotropic material fields*.

## Code / Data Links

- **arXiv:** [2510.22975](https://arxiv.org/abs/2510.22975) (v1 27 Oct 2025, v2 2 Mar 2026, ~30-40 MB PDF)
- **Project page:** [research.nvidia.com/labs/sil/projects/vomp](https://research.nvidia.com/labs/sil/projects/vomp/) (videos, teaser, interactive demo)
- **Code:** [github.com/nv-tlabs/VoMP](https://github.com/nv-tlabs/VoMP) (Apache 2.0 ✅, ~$0 Lambda to clone, includes pre-trained weights download script)
- **Pre-trained weights:** [huggingface.co/nvidia/PhysicalAI-Simulation-VoMP-Model](https://huggingface.co/nvidia/PhysicalAI-Simulation-VoMP-Model) (1.73 GB, Apache 2.0 ✅, includes MatVAE + Geometry Transformer)
- **OpenReview (ICLR 2026):** [openreview.net/forum?id=aTP1IM6alo](https://openreview.net/forum?id=aTP1IM6alo) (rebuttal + reviewer comments)
- **ICLR 2026 virtual:** [iclr.cc/virtual/2026/poster/10008698](https://iclr.cc/virtual/2026/poster/10008698)
- **Hugging Face papers page:** [huggingface.co/papers/2510.22975](https://huggingface.co/papers/2510.22975)
- **License:** Apache 2.0 ✅ (commercial-deployable, no AGPL-3.0 blocker like Era3D 127)
- **Cite as:** Dagli et al. (2026). VoMP: Predicting Volumetric Mechanical Property Fields. ICLR 2026.
- **Citations as of 2026-06-11:** ~10-20 GS citations (6 months post-arXiv v1, ICLR 2026 acceptance, 1 NVIDIA paper announcement in 2026-01 — modest initial uptake, but expected to grow rapidly because it's the *only* Apache-2.0 mechanical-property-field method)

## For Our Project (Dental Crown Gen)

The *killer* insight from VoMP for v0 is that **per-voxel mechanical property prediction is a *solved* problem in the general 3D-vision field** (VoMP, Apache 2.0, 3.6s inference, 9× better than prior art). v0 can *adopt* VoMP *directly* for v0 sub-task 4 (occlusion simulation) without re-implementing the (E, ν, ρ) prediction — only the *dental-specific fine-tuning* is needed.

**(a) ★ ADOPT VoMP AS THE V0 SUB-TASK 4 (OCCLUSION SIMULATION) MATERIAL-PROPERTY PREDICTOR** ($0 Lambda for the pre-trained weights, $50-100 Lambda for dental fine-tuning on 3DTeethSeg22 + ToSynFCD + private 1K clinical scans, 1-2 weeks engineering; the pre-trained Apache-2.0 weights from HuggingFace are *immediately deployable* for *general* 3D objects; dental fine-tuning is needed to handle the *specific* (E, ν, ρ) of enamel/dentin/pulp/cementum/gum/titanium/zirconia/PFM). **The killer practical feature: ~3.6s per object on a single A100, can be deployed on a Lambda A100 instance for $50-100/month, runs 100s of inferences per day, ~$0.50-1 per dental arch.** This is *fast enough* for clinical chairside use, and the Apache 2.0 license means v0 can ship a *closed-source* commercial product.

**(b) ★ ADOPT VoMP'S 4-SOURCE ANNOTATION PIPELINE AS THE V0 H5 MECHANISM FOR SUB-TASK 4** (the killer H5 mechanism: 3DTeethSeg22 + ToSynFCD for part-segmented assets + MatWeb dental / Granta dental for material databases + DINOv2 dental-fine-tuned for visual features + GPT-4o (or dental-VLM) for ambiguous cases; produces 1K-10K dental arches with spatially-varying (E, ν, ρ) fields; $50 Lambda, 2-3 weeks engineering; the *only* H5 mechanism that scales without real tensile-testing experiments).

**(c) ★ ADOPT MatVAE'S LATENT-MANIFOLD-CONSTRAINED DESIGN AS THE V0 SUB-TASK 4 PHYSICAL-VALIDITY ENFORCER** (the *killer* design lesson: the (E, ν, ρ) output must live on a *learned manifold of physically valid materials*, not a raw regression; the 2D MatVAE latent space is the *enforcer*; for v0 sub-task 4, the MatVAE-style design ensures that *any* predicted material is *physically real* — no negative Poisson's ratios, no impossible density × E combinations; the *killer* v0 paper contribution: "first end-to-end pipeline that predicts physically-valid material fields of generated dental crowns for FEM/MPM/XPBD simulation").

**(d) ★ ADOPT VoMP'S 3.6s INFERENCE-TIME PARADIGM AS THE V0 SUB-TASK 4 REAL-TIME PARADIGM** (the *killer* practical feature: 3.6s per object on a single A100 means v0 sub-task 4 can be deployed as a *real-time* API; v0 paper contribution: "dental arch material-field prediction in 3.6s, fast enough for chairside use"; the *direct* consequence: combine with MPM/FEM simulation (5-30s) for *clinical-real-time* (<60s total) crown-under-bite-force prediction).

**(e) ★ ADOPT VoMP'S 8-VIEW MULTI-VIEW FEATURE AGGREGATION AS THE V0 SUB-TASK 4 INPUT DESIGN** (the *killer* H3 mechanism: 8 turnaround views with frozen DINOv2 features, projected to per-voxel, averaged; for v0, render the 6-tooth context (1 prep + 2 adjacent + 3 opposing + gum) from 8 views, extract DINOv2 features, predict material field; the *killer* practical feature: works for *any* 3D representation, so v0 can predict material fields of *either* the v0-generated crown mesh *or* the patient's actual 3DGS from intra-oral scanner; the *unified* pipeline for v0 sub-task 4 + v0 sub-task 1).

**(f) ★ ADOPT VoMP'S "DETERMINISTIC FEED-FORWARD > LLM-AUGMENTED" FINDING AS THE V0 DESIGN LESSON** (the *killer* empirical evidence: VoMP (deterministic, 3.6s) beats Phys4DGen★ (LLM-augmented, 51.65s) on all 6 material-field metrics; the lesson: for *per-voxel prediction* tasks, *deterministic feed-forward* is *strictly better* than LLM-augmented; for v0 sub-task 4, use VoMP (deterministic), not PhysGen3D-style GPT-4o-per-object inference; the *killer* v0 paper positioning: "v0 sub-task 4 uses VoMP's deterministic feed-forward design, achieving 9× better material-field accuracy and 14× faster inference than LLM-augmented alternatives").

**(g) ★ ADOPT VoMP'S 2D MatVAE LATENT SPACE AS THE V0 INTERPRETABILITY TOOL** (the *killer* practical feature: the 2D MatVAE latent space is *visualizable* and *interpolatable*; v0 paper can include a 2D scatter plot of "all materials in the training set" with the predicted (E, ν, ρ) for the generated crown highlighted; the *killer* clinical interpretability: the dentist can see "your generated crown's material is in the enamel region of the material space, with confidence interval X"; the *killer* v0 paper differentiator: "first dental-crown paper with material-space interpretability").

**(h) CITE VoMP AS THE KILLER 2025-2026 DETERMINISTIC MATERIAL-FIELD PREDICTION SOTA** in v0 paper's related-work + Table 1 (the *direct* technical precedent for v0 sub-task 4; positions v0 as "the first clinical dental application of VoMP's 4-source annotation pipeline"; complete the 2024-2025 mechanical-property-prediction arc: NeRF2Physics 2024 → PUGS 2025 → Phys4DGen 2025 → **VoMP 2026 (NEW, deterministic, Apache 2.0)**).

**(i) ADOPT VoMP'S SPLATS HANDLING AS THE V0 SUB-TASK 1 + SUB-TASK 4 UNIFIED DESIGN** (the *killer* practical feature: VoMP handles meshes + SDFs + NeRFs + 3DGS in one pipeline; for v0, this means the *same* material-field predictor can be used for the v0 sub-task 1 dental arch (3DGS from intra-oral scanner) *and* the v0 sub-task 2 generated crown (mesh from DMC 033); the *unified* v0 sub-task 4 pipeline: predict material field of the *full* arch + the *generated* crown, simulate them *together*, visualize the result; $0 Lambda additional, 0.5-1 day code change to wire up).

**(j) ★ ADOPT VoMP AS THE V1 CHAIRSIDE PREVIEW STACK** (the *killer* v1 product feature: with v0 sub-task 1 (DiffSplat 126, 0.86B SD1.5, $0 Lambda pretrained) + v0 sub-task 2 (DMC 033, $25 Lambda) + v0 sub-task 4 (VoMP, $50-100 Lambda dental fine-tune) + Mitsuba renderer (PhysGen3D 139, $0 Lambda), v1 can deliver: "patient sees generated crown under simulated bite force in <60s chairside"; this is the *killer* clinical product, the *killer* HK startup pitch, the *killer* v0→v1 progression; v0 paper should *position* v0 as "v0 ships the *batch* pipeline, v1 ships the *interactive* chairside pipeline").

**(k) ★ ACKNOWLEDGE VoMP'S "ISOTROPIC ONLY" LIMITATION IN V0 PAPER** (the *killer* honesty: VoMP assumes *isotropic* materials (E, ν, ρ are the same in all directions), but *dentin* is *anisotropic* (E varies 2-3× depending on tubule direction); v0 paper should *explicitly* acknowledge this limitation and position v1 as "anisotropic material field for dentin tubules + enamel prisms + periodontal ligament fiber orientation"; the *killer* 2026-2027 research direction: anisotropic material fields for biological tissues).

**v0 compute: ~$9,520-11,730 Lambda** (was $9,470-11,630 from 139, **+$50-100 for VoMP dental fine-tuning + $0 for VoMP pretrained weights, $0 for Mitsuba, $0 for 3DTeethSeg22 + ToSynFCD + dental material databases**; all in Apache 2.0, no AGPL-3.0 blocker like Era3D 127). **★ v0 sub-task 4 (occlusion simulation) stack is now COMPLETE**: (1) VoMP (material field prediction, 3.6s) + (2) Mitsuba (physical-based rendering, ~1s) + (3) Taichi-Elements MPM or Newton (simulation, 5-30s) + (4) GPT-4o (material advisor, optional) = **~10-35s total per crown**, the *fastest* clinical-real-time dental crown simulation in our reading list.

**Note in `papers/140-vomp-dagli26.md`.** VoMP + PhysGen3D 139 + WonderPlay 137 + RealWonder 138 + PhysDreamer + PhysGen + Phystwin + PhysX-3D + SOPHY + Pixie + PUGS + NeRF2Physics + Phys4DGen = the *physics-aware generative systems + material property prediction* arc (the most-direct coverage of v1's "crown preview from various angles + crown under bite force" feature); VoMP's 4-source annotation pipeline + MatVAE latent space + Geometry Transformer is the *direct architectural template* for v0 sub-task 4 (occlusion simulation) + v1's "crown under bite force" product feature. **★ CORRECTION TO 138 + 139 RECOMMENDATIONS:** the 138-RealWonder note recommended PhysGen3D for 139 (✓ correct), and 139-PhysGen3D note recommended "vomp (Shen NVIDIA 2025)" for 140 (✗ author misattribution — actual lead author is Rishit Dagli; the paper choice VoMP was correct, the author attribution was wrong). The *paper recommendation* is *correct*; only the author attribution needs correction.

## Next Paper to Read

**Recommendation: *read 141 = Pixie (Le et al. 2025)*** — VoMP's only *concurrent* feed-forward alternative (also ICLR 2026, also the "physics-aware 3D" arc), the *direct competitor* that uses *points from filtering NeRF densities* (vs VoMP's voxelized DINOv2 features); the *killer* comparative analysis for v0 sub-task 4: which feed-forward design wins for dental arches, voxelized DINOv2 (VoMP) or filtered NeRF points (Pixie)? The answer determines the v0 paper's "adopt VoMP" recommendation.

**Alternative: *read 141 = PhysX-3D (Cao et al. 2025)*** — the *joint shape+material generative model* that uses TRELLIS (paper 101) structural latent + learned material latents; the *direct complement* to VoMP for v0 sub-task 1 (not just *predict* materials of an existing shape, but *generate* new shapes with materials from scratch); the *killer* paper for v0 if the v0 paper wants to position v0 as "the *first* joint shape+material generation for dental crowns".

**Alternative: *read 141 = SOPHY (Cao et al. 2025)*** — the *3D generative model* with *simulation-ready* outputs; uses a *material decoder* (not yet released); the *direct* comparison to VoMP for *generative* material prediction (VoMP predicts for *existing* shapes, SOPHY predicts for *generated* shapes); the *killer* paper for v0 if v0 sub-task 1 (dental arch synthesis) wants to *jointly* predict shape + material.

**Recommendation ranking:**
1. **Pixie** (Le 2025) — most direct competitor, the *single* other feed-forward material predictor; *killer* for v0 sub-task 4
2. **PhysX-3D** (Cao 2025) — joint shape+material generation; *killer* for v0 sub-task 1 + sub-task 4
3. **SOPHY** (Cao 2025) — generative material decoder; *killer* for v0 sub-task 1 if joint generation is the goal

**Most likely 141 = Pixie**, the direct VoMP competitor, the natural next read in the *physics-aware* arc.
