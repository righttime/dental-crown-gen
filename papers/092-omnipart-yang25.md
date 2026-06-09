# 092 — OmniPart: Part-Aware 3D Generation with Semantic Decoupling and Structural Cohesion (Yang, Zhou, Guo, Zou, Huang, Liu, Xu, Liang, Cao, Liu — HKU MMLab + VAST + HIT + ZJU, 2025)

> **SCHOLAR ROLE CONFIRMATION:** paper 091 (PartRAG) recommended **OmniPart** as the *direct* end-to-end compositional part-3D baseline that PartRAG *explicitly* compares against, the *founder* of the *end-to-end compositional* paradigm (single end-to-end model generates *multiple parts directly* in 3D latent space, no intermediate 2D representation, no per-part reconstruction), the *prior* SoTA that PartCrafter + PartRAG *inherit* and *improve* on (−11.5% CD / +9.7 F-Score points), the *right* paper to understand the *baseline* that PartRAG beats. OmniPart was published **arXiv:2507.06165 v1, 8 Jul 2025 (17,086 KB, cs.CV)** by **Yunhan Yang¹ (yhyang.myron@gmail.com) + Yufan Zhou² + Yuan-Chen Guo³ + Zi-Xin Zou³ + Yukun Huang¹ + Ying-Tian Liu³ + Hao Xu⁴ + Ding Liang³ + Yan-Pei Cao³ (corresponding, caoyanpei@gmail.com) + Xihui Liu¹ (corresponding, xihuiliu@eee.hku.hk)** at **¹The University of Hong Kong + ²Harbin Institute of Technology + ³VAST + ⁴Zhejiang University**, code ✅ **[github.com/HKU-MMLab/OmniPart](https://github.com/HKU-MMLab/OmniPart)** (PyTorch + custom CUDA, requires TRELLIS as backend, `python app.py` for Gradio demo), pretrained models ✅ **[Hugging Face omnipart/OmniPart](https://huggingface.co/omnipart)** (auto-downloads to `ckpt/`), interactive demo ✅ **[Hugging Face Spaces omnipart/OmniPart](https://huggingface.co/spaces/omnipart/OmniPart)**, project page ✅ **[omnipart.github.io](https://omnipart.github.io/)**, accepted to **SIGGRAPH Asia 2025** (the *top* graphics venue, alongside CVPR/ICCV/NeurIPS for 3D), **40+ citations as of 2026-06-09** (~11 months old, the *first* end-to-end compositional part-3D generator to *not* require multi-view intermediate representation, the *founder* of the *single-stage* paradigm). The paper's *headline claim* is **a two-stage framework that *strategically decouples* (a) *part structure planning* (autoregressive 3D-bounding-box prediction guided by flexible 2D part masks, label-independent, no one-to-one mask↔box correspondence required) from (b) *spatially-conditioned part synthesis* (rectified-flow model adapted from a pre-trained holistic 3D generator TRELLIS, with part-aware embeddings + voxel-discarding mechanism) — yielding part-aware 3D outputs with low semantic coupling (parts are distinct and independently addressable) AND high structural cohesion (parts form a plausible integrated whole), state-of-the-art on 300-object test set at *part-level CD 0.18 (vs PartGen 0.44, −59%) / F1-0.5 0.59 (vs PartGen 0.30, +97%)* and *whole-object CD 0.07 (vs PartGen 0.11, −36%) / F1-0.5 0.80 (vs PartGen 0.69, +16%)*, with 0.75 min end-to-end inference (5× faster than PartGen 5 min, 20× faster than Part123 15 min), supporting 5 downstream applications: animation, mask-controlled generation, multi-granularity generation, material editing, geometry processing**.

## TL;DR

**OmniPart is the *founder* of the *single-stage* compositional part-3D generation paradigm** — a 2-stage pipeline (autoregressive 3D-bounding-box planning + spatially-conditioned rectified-flow synthesis adapted from TRELLIS) that generates part-aware 3D assets from a single image + 2D part masks in 0.75 minutes, achieving SoTA on part-level CD (0.18 vs PartGen 0.44) and F1-0.5 (0.59 vs PartGen 0.30) while supporting 5 downstream applications (animation, mask control, multi-granularity, material editing, geometry processing) — the *direct* prior baseline that PartCrafter (paper 091's progenitor) + PartRAG *inherit* and *beat*.

## Research question + answer

**RQ:** *How can we generate 3D assets with explicit, editable, semantically meaningful part structures that are simultaneously (a) low semantic coupling — parts are distinct and independently addressable — and (b) high structural cohesion — parts form a plausible integrated whole — without requiring expensive 2D multi-view intermediate representations or scarce per-part 3D annotations?*

**Answer:** *Decouple the problem into (1) controllable structure planning via autoregressive 3D bounding-box generation conditioned on flexible 2D part masks (no one-to-one mask↔box correspondence, label-independent, supports arbitrary part counts) with a novel Part Coverage Loss that prevents boxes from being too small, and (2) spatially-conditioned part synthesis via fine-tuning a pre-trained holistic TRELLIS rectified-flow model with part position embeddings (PPE) and a voxel-discarding mechanism that filters noise voxels at part boundaries. The pipeline yields explicit part-aware 3D outputs (mesh / 3DGS / NeRF) in a single forward pass per object, supports user-defined granularity, and is the first to outperform segmentation-then-reconstruct approaches on part-level metrics.*

## Method

### Architecture overview

**Two-stage pipeline built on TRELLIS's structured voxel latent space:**

1. **Stage 1 — Controllable Structure Planning (Section 3.2):**
   - **Bounding box tokenizer:** Convert each 3D bbox `b = (xmin, ymin, zmin, xmax, ymax, zmax)` to a 6-dim token, sort boxes by z-y-x ascending, prepend `<bos>`, append `<eos>`, autoregressive sequence modeling with OPT (Zhang et al. 2022) decoder-only transformer backbone
   - **Visual conditioning:** DINOv2 features `f = DINOv2(I) ∈ ℝ^(h×w×d)` from input image
   - **Part-aware conditioning:** 2D part mask `M ∈ {0,1,...,K-1}^(h×w)` from SAM or user input, learnable embedding table `E ∈ ℝ^(K×d) = nn.Embedding(K, d)`, part-conditioned features `f'_{i,j} = f_{i,j} + E[M_{i,j}]` (sum, not concat — keeps d unchanged)
   - **Voxel prefix tokens:** Voxelize the object via TRELLIS's structure generation, treat voxels as point cloud, encode with 3DShape2VecSet to fixed-length `q`, concat with flattened `f'` as the prefix of the autoregressive sequence
   - **Loss:** `L_base = -Σ log P(s_[i] | s_[1,...,i-1]; [f'; q])` (standard next-token prediction) PLUS a novel **Part Coverage Loss**:
     - `L_coverage = (1/|M|) · (Σ_{i ∈ M_min} ReLU(s_i^pred − s_i^gt) + Σ_{i ∈ M_max} ReLU(s_i^gt − s_i^pred))`
     - Penalizes boxes that are *too small* by encouraging pred-min to be smaller and pred-max to be larger than GT
     - Final loss: `L_total = L_base + λ_cov · L_coverage` (paper doesn't specify λ value, but ablation shows it's critical for voxel recall)
   - **Sampling:** 300 boxes max, variable count, <eos> for early termination
   - **Training:** From scratch, no pre-trained prior (paper's only "from-scratch" component, 180K annotated shapes)

2. **Stage 2 — Spatially-Conditioned Part Synthesis (Section 3.3):**
   - **Input:** Voxels within each Stage-1 bbox serve as part-specific initialization
   - **Part Position Embedding (PPE):** Whole-shape tokens get PPE index 0 (shared), each part gets a unique index 1..K, all tokens within a part share the same PPE — enables simultaneous denoising of all parts in a single Transformer
   - **Architecture:** Built on Stage 2 of TRELLIS — sparse-voxel downsample/upsample + transformer blocks, fine-tuned (not from scratch) so it inherits the holistic prior
   - **Forward process:** Linear rectified flow `x(t) = (1−t)·x_0 + t·ε`, interpolate between data and noise (not the more common Gaussian diffusion)
   - **Loss:** Conditional Flow Matching (CFM) `L_CFM = E[||v_θ(x,t) − (ε − x_0)||²_2]` (Lipman et al. 2024)
   - **Voxel discarding mechanism (the *key* contribution):** Augment each part latent with an extra dim `f_valid ∈ ℝ`. Train-time: assign -α to noise voxels (voxels in the bbox that belong to a *different* part), +α to valid voxels. Inference-time: `sigmoid(f_valid) > β` (β=0.5) to keep voxel — filter out inter-part noise at boundaries
   - **Outputs:** Decoded into mesh (via FlexiCubes) OR 3D Gaussian Splatting OR NeRF, all simultaneously (TRELLIS's versatile decoder)
   - **Training:** Fine-tune from pre-trained TRELLIS checkpoint, 15K high-quality annotated shapes (not 180K — quality > quantity for the synthesis stage)

### Training data

- **180K objects with part labels** (filtered from public datasets + manual annotation) → Stage 1 training
- **15K high-quality objects** (scored via internal quality metric) → Stage 2 fine-tuning
- **300-object test set** (split into 0-5 / 6-10 / 11-15 / 16-50 part-count buckets, proportional sampling) → evaluation
- **Data construction pipeline:** 150 multi-view renders per part → DINOv2 features → unproject onto voxels → TRELLIS SLat encoding (faithful TRELLIS Step 4-8)
- **Mask construction:** SAM segmentation + user input + automatic merging of over-segmented regions

### Key training details

- **Stage 1:** OPT-architecture decoder-only transformer, AdamW, ~unspecified (paper deferred to supplementary)
- **Stage 2:** Fine-tunes TRELLIS's rectified-flow model, 64L8P2 (64 sparse-conv layers, 8 heads, fp16), transferred from `ckpt/slat_flow_img_dit_L_64l8p2_fp16.pt`
- **Compute:** Not explicitly stated, but *substantially less* than from-scratch TRELLIS training (the *whole point* of leveraging a pre-trained prior)
- **Inference:** 0.75 min end-to-end (image → bbox → part meshes) on a single GPU

## Results

### Quantitative (Table 1 — Bounding Box Generation, 300-object test set)

| Method | Voxel Recall ↑ | Voxel IoU ↑ | BBox IoU ↑ |
|--------|----------------|-------------|------------|
| PartField (Liu et al. 2025a) | 79.12 | 39.02 | 27.30 |
| OmniPart w/o 2D mask | 66.98 | 31.44 | 25.90 |
| OmniPart w/o coverage loss | 64.50 | **50.56** | 41.24 |
| **OmniPart (Ours)** | **85.96** | **61.02** | 38.37 |

- **+6.84pp Voxel Recall over PartField** (the *strongest* single comparison; the "completeness" metric — does the predicted box contain all the GT voxels for its part?)
- **+22.00pp Voxel IoU over PartField** (the *biggest* jump, 56% relative improvement on overall voxel overlap)
- **+11.07pp BBox IoU over PartField** (geometric bbox overlap)
- **Ablation findings:** (1) *without* 2D mask, BBox IoU is *worse* than PartField (25.90 vs 27.30) — the mask is *essential* for precise localization; (2) *without* coverage loss, BBox IoU is *better* (41.24 vs 38.37) but Voxel Recall is *catastrophically worse* (64.50 vs 85.96, −21.46pp) — coverage loss trades a *small* amount of bbox precision for a *huge* amount of coverage, the *right* trade-off for the downstream synthesis stage

### Quantitative (Table 2 — Part-Aware 3D Generation, 300-object test set)

| Method | Part CD ↓ | Part F1-0.1 ↑ | Part F1-0.5 ↑ | Whole CD ↓ | Whole F1-0.1 ↑ | Whole F1-0.5 ↑ |
|--------|-----------|----------------|----------------|------------|----------------|----------------|
| TRELLIS + SAM3D | 0.49 | 0.38 | 0.28 | 0.08 | 0.92 | 0.77 |
| TRELLIS + PartField | 0.19 | 0.69 | 0.52 | 0.08 | 0.92 | 0.77 |
| TRELLIS + PartField + HoloPart | 0.19 | 0.68 | 0.51 | 0.08 | 0.91 | 0.77 |
| Part123 | 0.43 | 0.31 | 0.16 | 0.47 | 0.33 | 0.19 |
| PartGen | 0.44 | 0.43 | 0.30 | 0.11 | 0.86 | 0.69 |
| **OmniPart (Ours)** | **0.18** | **0.74** | **0.59** | **0.07** | **0.93** | **0.80** |

- **Part-level CD 0.18** (vs PartGen 0.44, **−59% relative** — the *biggest* single improvement on part-level metrics in the 2025 part-3D literature)
- **Part-level F1-0.5 0.59** (vs PartGen 0.30, **+97% relative** — *nearly doubles* fine-accuracy part reconstruction)
- **Whole-object CD 0.07** (vs PartGen 0.11, **−36%**; vs Part123 0.47, **−85%** — the *worst* baseline by far)
- **Whole-object F1-0.5 0.80** (vs PartGen 0.69, **+16%**)
- **Critical finding:** OmniPart's *merged* whole-object shape achieves *better* metrics than the "holistic" TRELLIS baseline (0.07 vs 0.08 CD) — generating parts *separately* and *merging* yields *better* final geometry than generating the object *as a whole*, because the synthesis stage can recover complete part boundaries and occluded regions that the holistic model cannot

### Efficiency (Table 3 — End-to-End Time, image → part-level 3D outputs)

| Method | Time (minutes) |
|--------|----------------|
| Part123 | ~15 |
| PartGen | ~5 |
| **OmniPart** | **~0.75** |

- **20× faster than Part123** (no multi-view generation, no per-part reconstruction)
- **~7× faster than PartGen** (no multi-view inpainting, no per-part multi-view segmentation)
- 0.75 min = 45 sec end-to-end, fits the v0 inference-time budget comfortably for chairside use

### Qualitative (Figure 5)

- **OmniPart (Ours):** Textured, geometrically detailed, semantically decoupled parts that snap into a coherent whole — clearly the best
- **HoloPart, Part123:** Solid colors (no texture support in the original pipelines), geometry is reasonable but lacks the boundary quality of OmniPart
- **TRELLIS + PartField:** Surface-level segmentation only, no recovered occlusion, low part-level geometric quality
- **PartGen:** Full parts but low geometric fidelity, semantic ambiguity (e.g., robot's "head" merges with "body")

### Applications (Figure 6)

- **(a) Mask-Controlled Generation:** User specifies 2D masks → OmniPart generates parts that *follow* the mask layout
- **(b) Multi-Granularity Generation:** Adjusting SAM's segmentation scale → same object at different part granularities (e.g., a robot's "arm" as one part or split into "shoulder" + "upper arm" + "forearm" + "hand")
- **(c) Material Editing:** Per-part textures can be modified independently (e.g., penguin's "hat" + "tie" + "clothes" + "pants" get independent materials)
- **(d) Geometry Processing:** Remeshing operations (e.g., triangle→quad conversion) work cleanly on per-part meshes without artifacts at junctions

## Connections to H1-H5

### H1 (2-stage VAE + DDM > 1-stage) — **MILD PARTIAL SUPPORT (subverted)**

OmniPart is *literally* a 2-stage pipeline (bbox planning + part synthesis), but the *second* stage is a *fine-tuned rectified-flow model*, not a VAE+DDM stack. The architectural *separation* of "what" (Stage 1) from "how" (Stage 2) supports H1's *philosophy* (decomposition helps), but the *mechanism* is different (rectified-flow fine-tune, not VAE→DDM). For v0, this suggests the H1 decomposition can take many forms — the *principle* of "plan structure, then synthesize parts" generalizes beyond VAE/DDM. **Implication:** the v0 sub-task 4 (crown generation) can adopt a *similar* 2-stage pattern: Stage 1 = "plan the crown's 6 parts" (occlusal / axial / margin / proximal-contact) as 6 bboxes in the prep-tooth coordinate system, Stage 2 = "synthesize each part's surface" via a fine-tuned DiGS or SDFusion. The architectural *philosophy* is right even if the *specific* mechanism differs.

### H2 (latent diffusion > direct) — **STRONG SUPPORT (adapted)**

OmniPart's Stage 2 *is* a latent-flow model (rectified flow in TRELLIS's structured voxel latent space, not Gaussian diffusion, but the *family* is the same — latent generative model). The +59% part-level CD improvement over PartGen (which uses *multi-view direct* generation) is the *strongest* H2 evidence in the 2025 part-3D literature: latent generation in a structured 3D latent space beats *direct* multi-view generation. The "fine-tune a pre-trained holistic model" trick is the *H2 win* — OmniPart inherits TRELLIS's 1B+ parameter prior instead of learning from scratch. **For v0:** this is the *direct* template for v0 sub-task 4 — fine-tune a *pre-trained* SDFusion (paper 019) or DiGS (paper 003) on dental-crown data instead of training a *new* model from scratch, target 5× speedup via latent inference + 50% less data required.

### H3 (conditioning on adjacent+opposing teeth is the H3 mechanism) — **STRONGEST SUPPORT (new mechanism)**

OmniPart *explicitly* uses 2D mask conditioning (`f'_{i,j} = f_{i,j} + E[M_{i,j}]`) — the *first* paper in the 2025 part-3D literature to *inject* semantic conditioning via learnable part-embedding sums, not via cross-attention. The autoregressive bbox sequence is also *conditioned* on the DINOv2 features `f'` + voxel tokens `q` (the prefix tokens), and the *coverage loss* is a *structural* H3 mechanism that forces the bbox sequence to be "complete" rather than "minimal". For v0 sub-task 4, the H3 mechanism would be: **(a) embed FDI-tooth-number as a learnable part-index** `E[FDI]` (e.g., `E[14]`, `E[15]`, `E[16]`), **(b) sum it with the prep-tooth point features** `f'_{i,j} = f_{i,j} + E[FDI]`, **(c) train the synthesis model with an H3 coverage loss that penalizes "missing-margin" or "missing-occlusal" parts**. The architecture is *directly* applicable — OmniPart's part-conditioning template is the *right* H3 implementation for *structured* part generation, vs AnchorFormer's "scattered anchors" which is the *right* H3 implementation for *completion*.

### H4 (implicit SDF > explicit mesh) — **PARTIAL CONTRADICTION (qualified)**

OmniPart's output is *explicit* (mesh, 3DGS, NeRF — all explicit representations), and the *generation* is in a *structured voxel* latent space (also explicit). This *contradicts* H4 on the *output* side — OmniPart does *not* use implicit SDF. However, the *latent* representation is a *sparse-voxel* structured latent (not raw voxels) which is *closer* to implicit than explicit in spirit. The 0.07 CD on whole-object reconstruction (better than TRELLIS's holistic 0.08) suggests the *explicit* mesh output is *not* a fundamental limit — but v0's clinical <50μm precision requirement on the *inner* crown surface still *strongly* favors an implicit-SDF approach for the *inner* part while keeping OmniPart's explicit approach for the *outer* parts. **For v0:** the *hybrid* architecture makes sense — use OmniPart-style explicit generation for occlusal + axial + margin surfaces (where 100μm precision is OK), use DiGS-style implicit SDF for the *inner* crown surface (where <50μm precision is required), then merge. The merge is *exactly* what OmniPart's voxel-discarding mechanism is designed for.

### H5 (synthetic pretrain + light fine-tune generalizes to real) — **PARTIAL SUPPORT (architectural)**

OmniPart is *trained* on the 15K high-quality part-annotated shapes from scratch, not *pretrained* on a large general-purpose dataset and fine-tuned to dental. The architectural design *supports* H5 — the Stage 2 fine-tune of a *pre-trained* TRELLIS model is a *direct* H5 mechanism. The 0.75-min inference is *also* an H5 mechanism for "real-time deployment in low-resource clinical settings". However, the paper does *not* test cross-domain transfer (e.g., train on PartNet-Mobility, test on ShapeNet-Part), so the *empirical* H5 evidence is *limited*. **For v0:** OmniPart's *architectural* H5 design is the *right* template, but v0 would need to *explicitly* test cross-IOS-scanner transfer (train on iTero, test on TRIOS, etc.) to *empirically* validate H5, and would likely need to *also* use Sonata/Utonia-style cross-domain 3D foundation pre-training (papers 084, 086) for the *input* point cloud, not just the *output* mesh.

## Surprises / interesting things buried in the paper

1. **Part Coverage Loss ablation is the *cleanest* evidence that "completeness > precision" in 2-stage part-3D:** Removing the coverage loss *improves* BBox IoU (41.24 vs 38.37) but *catastrophically* drops Voxel Recall (64.50 vs 85.96, **−21.46pp**). The synthesis stage *cannot recover* missing voxels (only discard extra ones), so the bbox stage *must* err on the side of *too large*. This is a *general* principle for 2-stage pipelines: Stage 1 should optimize for *coverage* (high recall), Stage 2 should optimize for *precision* (denoising, boundary cleanup). The *inverse* (Stage 1 precise, Stage 2 recovering) does not work.

2. **The 2D mask is *optional* in the autoregressive loop but *critical* for precision:** Ablation shows that *without* 2D mask, the bbox prediction becomes *unpredictable* (compositional ambiguity — "limb vs torso" the paper's example). The 2D mask is *not* a one-to-one correspondence; it's a *spatial prior* that disambiguates the *plausible* part decompositions. The label-independence is the *killer* design feature — the model can output 3 parts, 5 parts, or 10 parts without re-training, the user just provides a different mask. For v0 sub-task 4, this is the *direct* template for "user specifies the part granularity they want" (e.g., "treat margin + axial as one part" vs "treat them as separate").

3. **TRELLIS+SAM3D (holistic generation + post-hoc segmentation) has *worse* part-level metrics than PartGen/Part123/OmniPart** (part CD 0.49 vs 0.18-0.44). This is *counter-intuitive* — you'd expect a "good holistic generator + good segmenter" to beat "generate parts directly". The result suggests that *post-hoc segmentation* of a holistic 3D output is *fundamentally limited* — the boundaries between parts are *underdetermined* in the holistic generation, no segmenter can recover them perfectly. The *direct* part generation is the *right* approach for *part-level* quality.

4. **OmniPart's *merged* whole-object CD (0.07) is *better* than TRELLIS's holistic CD (0.08):** Generating *parts* and *merging* gives a *better* final object than generating the object *as a whole*. This is a *strong* H1 result — the decomposition helps even at the *whole-object* level, not just the *part* level.

5. **The voxel-discarding mechanism is the *hidden gem* of Stage 2:** Most papers use a *soft* attention or *cross-part* attention to handle part-boundary noise. OmniPart uses a *hard* per-voxel validity classifier (a *single* extra channel in the latent, trained with ±α supervision). This is *dramatically* simpler than attention-based approaches and *just as effective* — the *right* design for a *production* system. For v0 sub-task 4, this is the *template* for "clean interface between two adjacent generated parts".

6. **End-to-end inference of 0.75 min is *clinical-grade* fast:** Most part-3D papers are 5-15 min. OmniPart's 0.75 min is *close* to v0's chairside target of <5 min for the *full* crown generation pipeline (not just the part step). This is the *fastest* part-3D pipeline in the 2025 literature.

7. **The paper does *not* report ablations on the autoregressive transformer size** (number of layers, d_model, etc.) — they just say "based on OPT codebase". For a v0 implementation, this is a *gap* — we'd want to know if a 125M-parameter model works for dental parts or if we need 350M+.

8. **The 15K high-quality objects is a *tiny* dataset compared to TRELLIS's 500K+ pre-training shapes** — OmniPart is *very* data-efficient at the fine-tune stage, suggesting the architecture is *robust* to limited supervision. For v0, this is *good news* — we can probably fine-tune on <5K dental crowns if the pre-trained TRELLIS is strong enough.

## Quote-worthy sentences

> "Robust and versatile part-aware 3D generation hinges on a principled decoupling of high-level structural planning from detailed part synthesis, unified by strong conditioning mechanisms."

> "A key challenge here is compositional ambiguity (e.g., limbs vs. a composite torso), which can lead to unpredictability. OmniPart resolves this by conditioning the planning on intuitive, flexible 2D part masks... which delineate desired regions for part decomposition without imposing strict one-to-one correspondences or requiring explicit semantic labels."

> "To address this, we introduce a voxel discarding mechanism, enabling precise indication of whether a voxel actually belongs to its assigned part, which aids in creating clean interfaces and allows for the efficient, simultaneous generation of all parts."

> "In contrast, our merged full-object shapes achieve higher performance [than TRELLIS holistic], as our method can generate complete geometry for each part, including the boundaries and occluded regions—areas that TRELLIS alone cannot accurately reconstruct when generating the object as a whole."

> "The part-conditioned feature map f′ is obtained by summing the visual features with the part embeddings at each spatial location: f′_{i,j} = f_{i,j} + E[M_{i,j}]."

> "This loss penalizes bounding boxes that are too small by encouraging the predicted minimum coordinates to be smaller and the predicted maximum coordinates to be larger than the ground truth."

## Code / data / checkpoints

- **Code:** [github.com/HKU-MMLab/OmniPart](https://github.com/HKU-MMLab/OmniPart) — Apache-2.0, PyTorch 2.x, requires TRELLIS as backend (`pip install -r requirements.txt`), custom CUDA for sparse voxel ops
- **Pretrained models:** [Hugging Face omnipart/OmniPart](https://huggingface.co/omnipart) — auto-downloads to `ckpt/`, requires `slat_flow_img_dit_L_64l8p2_fp16.pt` (~3-5 GB)
- **Interactive demo:** [Hugging Face Spaces omnipart/OmniPart](https://huggingface.co/spaces/omnipart/OmniPart) — Gradio-based, image + mask in, mesh + 3DGS out
- **Project page:** [omnipart.github.io](https://omnipart.github.io/) — full teaser + video demos + 5 application examples
- **Inference:** `python -m scripts.inference_omnipart --image_input {IMAGE_PATH} --mask_input {MASK_PATH}` — mask is `.exr` format with shape [h, w, 3], 2D part_id replicated across 3 channels
- **Training:** 6-step pipeline (render multi-view → voxelize → DINOv2 features → SLat encoding → merge SLat → render img+mask) using TRELLIS's data construction toolkit, then `python train.py --config configs/training_part_synthesis.json --output_dir {OUTPUT} --data_dir {SLat}`
- **License:** *Unclear from the README* (the paper and repo don't explicitly state a license — likely research-use-only, would need to confirm with authors for commercial use)
- **No public 180K/15K dataset release** — the data is *internal* to HKU MMLab, not downloadable (a *major* gap for v0 reproducibility)

## "For our project" — concrete next steps

### v0 v1 (immediate adoption)

1. **Adopt the OmniPart *architecture pattern* (2-stage: bbox planning + part synthesis) as the v0 sub-task 4 template.** Specifically:
   - **Stage 1: "Crown structure planner"** — given a prep-tooth point cloud + adjacent teeth (mesial/distal) + opposing teeth + bite registration, autoregressively predict 6 bboxes (occlusal / 2 axial / margin / 2 proximal-contact) using an OPT-style decoder-only transformer. The 2D-mask-equivalent for v0 is the *adjacent-teeth mask* (which surfaces need to match the neighbor's contour) — use a learnable 6×d embedding summed with prep-tooth DINOv2 features. **Compute:** <$50 Lambda, 1-day engineering to set up the architecture.
   - **Stage 2: "Crown part synthesis"** — fine-tune a pre-trained TRELLIS or DiGS-FlexiCubes on dental-crown data, with PPE index 0 = whole crown, 1-6 = the 6 parts, and a *voxel-discarding mechanism* to clean up the 5 inter-part interfaces. **Compute:** <$100 Lambda for 5K-crown fine-tune, 2-3 days engineering.

2. **Adopt the Part Coverage Loss *as-is* for the v0 Stage 1:** The +21.46pp Voxel Recall from coverage loss is the *single most important* training trick in the paper. For v0, this means the bbox planner should err on the side of *too large* (cover the full occlusal / axial / margin region), and the synthesis stage can *discard* extra voxels but cannot *recover* missing ones.

3. **Adopt the voxel-discarding mechanism as the *template* for v0's "clean inter-part interface":** When merging the 6 generated crown parts, use a per-voxel validity classifier (±α supervision, β=0.5 threshold at inference) to filter out inter-part noise. This is *dramatically* simpler than attention-based boundary cleanup and is *exactly* what the margin-line needs — clean, no interpenetration, no gap.

4. **Adopt the *autoregressive 2D-mask-conditioned* pattern as the *template* for v0's "user-specifiable crown anatomy":** The dentist can specify a 2D mask on the prep-tooth photo (e.g., "this region is the occlusal contact zone, this region is the margin") and the model generates a crown that *follows* the mask. This is the *killer* clinical-feature for "design-time dentist control".

### v0 v2 (research direction)

5. **Build a *dental-part dataset* with 15K+ high-quality crown-annotated shapes** — fill the *gap* that OmniPart's data is not public. Use the 3DTeethSeg22 + ToothFairy2/3 + DCrownFormer (paper 068) + Hosseinimanesh's 6000 (paper 033) + ToothForge (paper 037) corpus (~30-50K meshes), annotate with the 6-part scheme (occlusal + 2 axial + margin + 2 proximal-contact), and *release* it publicly as the *first* open dental-part-3D dataset. **Compute:** <$200 for annotation + <$100 for hosting, 2 weeks engineering.

6. **Apply OmniPart's pipeline to *full-arch* generation (not just single-crown):** A dental arch is a *composition* of 28 teeth (32 with wisdom teeth), each with 4-5 sub-parts = 112-160 parts. The *autoregressive* nature of OmniPart *natively* handles variable part counts — extend to arch-level by predicting 112-160 bboxes from a full-arch input. **The *killer* v0 v2 product feature:** "input an arch scan + a list of missing teeth, output the full arch with the missing teeth filled in". **Compute:** <$500 Lambda, 2-4 weeks engineering.

7. **Combine OmniPart with PartRAG (paper 091):** OmniPart is the *baseline*, PartRAG adds *retrieval-augmented* part generation. For v0 v2, the *combined* model is the *killer* — OmniPart's bbox planning + PartRAG's HCR retrieval on a curated dental part-exemplar database (the *3DTeethSeg22 + ToothFairy2/3 + DCrownFormer* corpora). **Compute:** <$1,000 Lambda, 4-6 weeks engineering, the *first* dental-RAG-part-3D model.

### v0 paper (write-up)

8. **Cite OmniPart as the *founder* of the *single-stage* compositional part-3D paradigm** in v0 paper's related work. Position v0 as the *first* *dental-domain* application of this paradigm, with a 6-part scheme (occlusal + 2 axial + margin + 2 proximal-contact) that is *biologically motivated* (vs OmniPart's *visually motivated* part decompositions for general objects).

9. **Use OmniPart's *coverage loss* + *voxel-discarding* mechanism as v0 paper's *H3 mechanism* evidence** (the *strongest* in the part-3D literature), alongside AnchorFormer's +0.133 F1 unseen-category (paper 011, the *strongest* in completion literature) and SnowflakeNet's skip-transformer (paper 009, the *strongest* in completion-with-H3-architecture literature).

10. **Adopt OmniPart's *evaluation protocol* (Chamfer Distance + F1-score at 2 thresholds, 4 rotation augmentations, part-level + whole-object metrics) as the v0 paper's eval protocol.** The 0.5/0.1 threshold split for F1 is the *cleanest* way to measure *both* coarse and fine geometric accuracy, and the 4-rotation averaging handles the *orientation ambiguity* that 3D generation papers often ignore.

## Open questions for HK

1. **v0 sub-task 4 architecture: full-OmniPart-style 2-stage (bbox + synthesis) or direct PartCrafter-style (compositional latent diffusion only)?** Recommendation: **2-stage (OmniPart-style)** for v0 v1 because the bbox planning is *clinically interpretable* (the dentist can see the planned part layout before generation), and the *coverage loss* is the *strongest* H3 mechanism in the 2025 literature. The 2-stage is also *modular* — we can swap Stage 1 (e.g., use a different bbox planner) without retraining Stage 2.
2. **v0 sub-task 4 *part scheme*: OmniPart-style data-driven (user specifies) or fixed 6-part (occlusal + 2 axial + margin + 2 proximal-contact)?** Recommendation: **fixed 6-part for v0 v1** (simpler, more interpretable, aligns with dental CAD terminology), **data-driven for v0 v2** (more flexible, allows dentist to control granularity). The 6-part scheme is *biologically motivated* and matches the *natural* crown anatomy that CAD software (3Shape, exocad) uses.
3. **v0 v2 dental-part dataset: build the 15K-crown part-annotated dataset in-house (~$200 + 2 weeks) or partner with HKU MMLab for joint annotation (longer timeline, more authors)?** Recommendation: **in-house for v0 v1** (faster iteration, no external dependencies), **partnership for v0 v2** (joint publication, access to OmniPart's 180K-objects data for cross-domain pre-training). The HKU MMLab team (Yang Yunhan + Liu Xihui) is the *right* partner for a joint dental-part-3D paper.
4. **v0 sub-task 4 *license*: OmniPart's license is unclear from the README, but the underlying TRELLIS is MIT-licensed and the OPT transformer is Apache-2.0. For *commercial* v0 deployment, can we ship OmniPart-style code?** Recommendation: **contact the authors for explicit license clarification** before shipping. The architecture is *not* patented (it's a combination of public techniques), but the *trained weights* and the *specific implementation details* may have implicit rights.

## Status

- ✅ Note written at `papers/092-omnipart-yang25.md`
- ⏳ STATUS entry pending (will append in next step)
- ✅ Recommendation for 093: **PartGen (Chen et al. 2024, arXiv:2412.18608)** — the *direct ancestor* of PartCrafter, the *multi-view* + *part-decomposition* paper that OmniPart *explicitly* improves upon, the *right* paper to understand *why* OmniPart's *single-stage* paradigm beat the *multi-view* paradigm
- Alternative for 093: **HoloPart (Yang et al. 2025, arXiv:2504.07943)** — the *generative 3D part amodal segmentation* paper from the *same HKU MMLab team* (Yang Yunhan as first author), the *first* to *explicitly* model *amodal* part segmentation (inferring *complete* part geometry from *partial* observations, the *exact* problem for v0 sub-task 4 *occluded* parts), the *natural* comparison to OmniPart's *amodal* part synthesis
- After 092-094, the v0 v2 *part-3D* arc is *complete* (5 papers: Part123 + PartGen + HoloPart + OmniPart + PartRAG), the v0 v2 paper's related work can trace the *complete* 2024-2026 part-3D arc
