# Paper 151 — OctFusion: Octree-based Diffusion Models for 3D Shape Generation

**Authors:** Bojun Xiong¹, Si-Tong Wei¹, Xin-Yang Zheng², Yan-Pei Cao³, Zhouhui Lian¹, Peng-Shuai Wang¹ (¹Peking University, ²Tsinghua University, ³VAST)
**Venue:** Computer Graphics Forum (presented at SGP 2025), Volume 44 Issue 5 — Eurographics + Wiley
**arXiv:** 2408.14732 v1 27 Aug 2024 → v2 8 Jun 2025 (cs.CV + cs.GR)
**DOI:** 10.1111/cgf.70198
**License:** CC-BY (open access, Eurographics standard)
**Code:** https://github.com/octree-nn/octfusion
**Pretrained:** Google Drive + Baidu (VAE-ShapeNet-depth-8 + per-category df_steps-split.pth + df_steps-union.pth)
**~150-250 GS citations as of 2026-06-12** (≈10 months after v1, ~2 weeks after v2, SGP 2025 spotlight-by-acceptance)
**★ CORRECTION TO 150-NOTE:** the 150-note's "next paper" recommendation listed *"OctFusion (Hassan et al. CGF 2025)"* — the **author is Bojun Xiong** (NOT Hassan) and the venue is **CGF (presented at SGP 2025)**. The paper choice was correct, the attribution was wrong.

---

## One-line TL;DR

A 33M-parameter **octree-based latent VAE + unified weight-shared U-Net diffusion** that generates continuous, manifold, **1024³-effective-resolution** 3D shapes in **2.5s on a single 4090**, beating LAS-Diffusion and XCube on FID with **18× fewer parameters** than XCube and **~10× faster inference** than the cascaded-diffusion baseline.

## Research question + answer

**Q:** How to train a 3D diffusion model that (a) produces arbitrary-resolution, continuous, manifold, watertight meshes, (b) trains with a SINGLE network (not the 3-stage cascaded U-Net cascade used by LAS-Diffusion and XCube), and (c) generates in seconds (not the minutes-to-hours of DreamFusion/Magic3D-style per-shape 2D-distillation)?

**A:** Two coupled innovations — **(1) octree-based latent VAE representation** (DualOctreeGNN encoder → 3-dim latent feature per octree leaf node → MPU-blended SDF decoder, with loss = SDF regression + octree-splitting BCE + KL, trained 200 epochs on 2×A40) + **(2) unified multi-scale U-Net for diffusion** that processes octree nodes at ALL depths with a SINGLE network by **weight-sharing across levels** (F1 predicts 8-channel 0/1 split signals at depth 4→6, F2 predicts latent features at depth 8, the same U-Net weights are reused for all depths — vs LAS-Diffusion/XCube's 3-6 separate networks). The result is **33M params, 0.69 GB GPU memory, 48.2 ms inference, 1024³ effective resolution**.

## Method

**Architecture (1-stage 2-phase, or 2-stage depending on naming):**
- **VAE encoder**: DualOctreeGNN on octree depth 8 (resolution 256³), downsamples to depth 6 (64³), outputs 3-dim latent per leaf node.
- **VAE decoder**: shared MLP Φ_sdf(x, f_i) maps local coords + latent → local SDF; **multi-level partition-of-unity (MPU)** blends neighboring leaf-node SDFs into a global continuous SDF (Eq. 1, w_i(x) is locally-supported linear B-Spline → C^0 continuity guaranteed).
- **VAE loss**: L_VAE = L_sdf + L_octree + λL_KL, λ=0.1, L_sdf = (1/N)Σ[λ_s‖F_sdf(x)-D(x)‖² + ‖∇F_sdf(x)-∇D(x)‖²] with λ_s=200 (SDF regression + smoothness via gradient matching, DualOctreeGNN loss), L_octree = BCE on splitting status, L_KL = KL to standard Gaussian.

**Diffusion model (the core innovation):**
- Treat **octree splitting status as 0/1 continuous signals** + **leaf-node latent features as continuous signals**; add Gaussian noise to both (Eq. 4-5).
- **Unified U-Net** with **stages F1, F2** trained sequentially (NOT jointly — empirically produces better results). F1 is a CNN U-Net (3 levels 16³/8³/4³, channels 64/128/256) that predicts 8-channel 0/1 split signals at depth 4 to grow the octree to depth 6 (1 channel per child node). F2 is a DualOctreeGNN U-Net (2 levels 64³/32³, channels 128/256) that predicts clean latent features on the depth-6 octree, then F1 is reused (WEIGHT-SHARED) to grow to depth 8, then F2 to predict the depth-8 latent features.
- The KEY trick: **shallow levels are processed with the SAME U-Net weights as deep levels** (U-Net of F1 at depth 4 reuses F1's weights at depth 6). This avoids training multiple networks and dramatically reduces parameter count.
- **Sampling pipeline (Fig. 4):** Sample noise at depth 4 → F1 → split signals → grow octree to depth 6 → F2 → clean latents + F1 → split signals → grow to depth 8 → F2 → clean latents → VAE decoder → MPU-blended SDF → Marching Cubes → mesh.

**3-stage extension (deeper octree, Fig. 19):** F1 (split at depth 4), F2 (split at depth 6), F3 (latents at depth 8 → up to 1024³). Used on Objaverse.

**Conditioning extensions:**
- **Text**: CLIP text encoder + cross-attention injection. Trained on Text2Shape for chair/table.
- **Sketch**: view-aware local attention from LAS-Diffusion, aggregated to guide octree generation.
- **Category**: label embedding as conditional input to U-Net.
- **Texture**: SEPARATE color latent features c_i on each leaf node + second MLP Φ_color(x, c_i) decoded to color field; train a separate (non-unified) octree-based diffusion model for color latents conditioned on geometric latents. RGB assigned to each mesh vertex after MC.

**Training:**
- VAE: 2× A40 48G GPUs, 200 epochs, batch 8, AdamW, lr 10⁻³ → 10⁻⁵ linear, 200k points per mesh, 50k per iteration for SDF sampling.
- OctFusion: 4× 4090 GPUs, F1 trained 4000 epochs in <1 day + F2 trained 500 epochs in 2 days, AdamW, lr 10⁻⁴ fixed.
- Inference: 50 DDIM steps on 1× 4090 in 48.2 ms; Marching Cubes for mesh extraction.

**Data:**
- **ShapeNet** (5 categories: chair, table, airplane, car, rifle, following LAS-Diffusion split).
- **Objaverse** (10k high-quality meshes subset from LGM, depth 10 = 1024³ resolution).
- **Text2Shape** (chair + table captions, from SDFusion preprocessing).
- Repair non-watertight meshes via DualOctreeGNN, normalize to unit cube, convert to SDF via mesh2sdf (takes days).

## Results

**Quantitative — Shading-image FID (lower is better, Table 1):**

| Method | Chair | Airplane | Car | Table | Rifle |
|---|---|---|---|---|---|
| LAS-Diffusion† | 20.45 | 32.71 | 80.55 | 17.25 | 44.93 |
| XCube | 18.07 | 19.08 | 80.00 | N/A | N/A |
| **OctFusion†** | **16.15** | 24.29 | **78.00** | **17.19** | **30.56** |
| LAS-Diffusion‡ | 21.55 | 43.08 | 86.34 | 17.41 | 70.39 |
| **OctFusion‡** | 19.63 | **30.92** | 80.97 | 17.49 | **28.59** |

OctFusion wins on **3/5 unconditional (chair, car, table, rifle)** and **2/5 conditional (airplane, rifle)**. Improvements are largest in **complex-structure categories** (chair -4.30 FID, rifle -14.37 FID unconditional).

**Quantitative — COV/MMD/1-NNA (Table 2, vs same train/eval split):**
- COV(EMD) **53.17** (best, LAS-Diff 52.43)
- 1-NNA(EMD) **63.72** (best, LAS-Diff 65.15)
- MMD(CD) 13.78 (3rd, comparable to Wavelet-Diff 13.37)

**Quantitative — Textured mesh FID (Table 3, 5 categories, vs GET3D + DiffTF):**
- OctFusion chair **31.81** (GET3D 51.79, DiffTF 64.58) — **-39% vs GET3D**.
- OctFusion airplane **26.64** (DiffTF 90.48) — **-71% vs DiffTF**.
- OctFusion car **65.58** (GET3D 60.89, DiffTF 137.96).
- OctFusion table **43.87** (GET3D 59.41).
- OctFusion rifle **41.20** (GET3D N/A, DiffTF N/A).

**Efficiency (Table 4 + 5 — the killer practical advantage):**
- 33M params (vs LAS-Diffusion 57M, XCube 1.6B) — **18× smaller than XCube**.
- 0.69 GB GPU memory (vs LAS-Diffusion 1.06 GB, XCube 12.76 GB) — **18× less than XCube**.
- 48.2 ms inference on 4090 (vs LAS-Diffusion 66.1 ms, XCube 135.3 ms) — **2.8× faster than XCube**.
- 4096 node number at first stage (vs LAS-Diff 262,144) — **64× sparser**, only surface-intersecting voxels.
- Training: 4× 4090 3 days (vs LAS-Diff 8× V100 7 days, XCube 8× A100 4 days).

**Qualitative (Fig. 5-7):** OctFusion captures **fine details that LAS-Diffusion and XCube miss** — airplane propellers, chair fluting (swivel chairs), car wheel hubs, thin slats. No severe distortions/artifacts (vs IM-GAN, SDF-StyleGAN, MeshDiffusion).

**Diversity (Fig. 8):** Most generated chairs are significantly different from training set (CD ×10⁻³ histogram is broad, not concentrated at 0), so OctFusion is NOT memorizing.

**Objaverse generalization (Fig. 9):** Plausible 3D shapes on a far more complex distribution than single ShapeNet categories, demonstrating strong cross-domain transfer.

**Text-conditioned (Fig. 13):** Significantly higher quality than AutoSDF and SDFusion — generates "lots of drawers inside table" with delicate structure details, "a two-layer table" with structural layering.

**Sketch-conditioned (Fig. 15):** Better geometry quality and better sketch-matching than LAS-Diffusion — recovers wheel hubs, horizontal/vertical bars of chairs that LAS-Diffusion misses.

## Ablations

**1. Octree-based latent representation vs other octree/voxel (Section 4.2.1, Fig. 11):**
- **Completeness**: Octree leaf nodes form COMPLETE coverage of the 3D bounding volume (vs LAS-Diffusion's pruned volume shell + XCube's interior voxels → can have holes).
- **Continuity**: MPU blending guarantees continuous SDF (vs LAS-Diffusion/XCube's discrete voxels with NO continuity guarantee → truncated surfaces).
- **Efficiency**: Only surface-intersecting voxels → 4096 nodes at first stage vs LAS-Diff 262,144 (64× sparser).

**2. Unified U-Net (weight-sharing) vs cascaded (Section 4.2.2, Fig. 12):**
- V1 no sharing: FID 22.22 (vs OctFusion **16.15**) — **+38% worse FID**.
- V2 separate U-Nets per level: FID 16.00 (comparable) but **2× more params**, slower convergence, harder to train.
- The weight-sharing trick is what enables single-network training with competitive quality.

**3. Deeper octree (Section 4.2.3, Fig. 14):**
- 2 stages (depth 4 → 6 → 8) = default.
- 3 stages (depth 4 → 6 → 8 → 10) = more details (chair leg protrusions) but more compute.
- Authors chose 2 stages as default to balance quality + efficiency.

## Connections to H1-H5

**H1 (2-stage VAE+DDM > 1-stage): STRONG SUPPORT.** OctFusion is canonically 2-stage (VAE → diffusion on octree latent). The killer innovation is the **unified weight-shared U-Net** that processes BOTH stages of octree growth (depth 4 split + depth 6 split + depth 8 latents) with a SINGLE network — this is the *de facto* 2024-2025 H1 paradigm (vs LION 149's separate 2 stages, vs DMC 033's internal PoinTr→SAP). The **33M params** is concrete evidence that VAE+DDM scales — 18× smaller than XCube's cascaded scheme.

**H2 (latent diffusion > direct): STRONG SUPPORT.** Pure latent diffusion on octree-based features (8-channel 0/1 split signal + 3-dim latent per leaf). The killer empirical evidence is **1024³ effective resolution** with **33M params** — direct diffusion would be 18× larger. The 4.2.1 ablation explicitly argues for octree-based latent over plain octree (LAS-Diffusion) and sparse voxel (XCube) — completeness + continuity + efficiency.

**H3 (contextual conditioning): NOT TESTED in main paper, BUT supported in extensions.** Conditioning is limited to **category label + text (CLIP) + sketch (view-aware local attention)** — no arch-context or tooth-context. For v0, the missing piece is **6-tooth context** as H3 input. However, the architecture is H3-agnostic: condition can be added via cross-attention (text path) or AdaGN (category path) to F1+F2.

**H4 (implicit SDF > mesh): STRONG SUPPORT.** OctFusion's *defining* claim is **continuous + manifold** implicit SDF (via MPU-blended DualOctreeGNN decoder) over discrete-voxel representations. Section 4.2.1 is a head-to-head with LAS-Diffusion (discrete octree), XCube (sparse voxel), and Trellis (sparse voxel) — completeness + continuity + efficiency all favor OctFusion. The 33M-param VAE produces **arbitrary-resolution** SDFs (output at 1024³ effective, can mesh-extract at any resolution). This is the *killer* H4 substrate for v0's clinical-fit evaluation: continuous SDF → can compute margin gap to arbitrary precision.

**H5 (synthetic + finetune): NOT TESTED.** Only trained on real ShapeNet + Objaverse + Text2Shape — no synthetic pretraining. For v0, the missing piece is dental-specific finetuning, but the architecture is H5-friendly: the VAE can be reused, the U-Net fine-tunes cheaply (33M params), the 4090 inference is 48ms.

## Surprises / interesting things buried

1. **The 8-channel 0/1 split signal is brilliant.** Each octree node at depth 4 stores 8 binary values (one per child) instead of a single binary split — this lets F1 make per-child splitting decisions in ONE forward pass, instead of 8 sequential decisions. This is a subtle but killer architectural detail that contributes to the 2.5s inference.

2. **Sequential training (F1 then F2) beats joint training.** This is a surprising empirical finding (Sec 3.3.2) — most diffusion papers train all stages jointly with shared losses. The authors explicitly note that training F1 → fixing weights → training F2 produces "quantitatively better results" — likely because each stage's denoising task is then well-defined and not competing with the other stage's gradients.

3. **The 0.69 GB GPU memory is what enables single-4090 inference.** LAS-Diffusion needs 1.06 GB (volume shell), XCube needs 12.76 GB (interior voxels + cascades). OctFusion only needs 0.69 GB because it only stores surface-intersecting voxels (4096 at first stage) — **64× sparser than LAS-Diffusion**. This is the key to clinical deployment on edge devices.

4. **Texture generation is a SEPARATE diffusion model, NOT unified with geometry.** This is a practical engineering choice (texture is conditioned on geometry latent, so joint training is unstable), but it means texture is 2× the inference cost of geometry alone. For v0's color dental crown matching, the killer insight is that texture diffusion is a *downstream* stage that doesn't need to be unified.

5. **The arXiv v1 → v2 gap (Aug 2024 → Jun 2025) is 10 months.** This is unusually long for a paper — likely because of the CGF revision cycle. The v2 adds Objaverse experiments, texture generation, and sketch conditioning that weren't in v1. The v1 abstract was already strong.

## Quote-worthy sentences

> "Diffusion models have emerged as a popular method for 3D generation. However, it is still challenging for diffusion models to efficiently generate diverse and high-quality 3D shapes."

> "Our key observation is that the octree itself is hierarchical; when generating the deep octree nodes, the shallow nodes have already been generated, resulting in nested U-Net structures for different octree levels."

> "We propose a unified diffusion model for different octree levels, which reuses the trained weights for shallow octree levels nodes when denoising deep octree nodes."

> "Our OctFusion is currently lightweight and contains only 33.03M parameters. We expect that our OctFusion can be easily scaled up and greatly improved if more data and computational resources are available."

> "The leaf nodes of an octree form a complete coverage of the 3D bounding volume. We keep all octree leaf nodes in the latent space, which guarantees to contain the whole shape, whereas LAS-Diffusion, XCube and Trellis prune voxels and only keep a subvolume, which may lead to holes in the generated shapes."

> "We merge local implicit fields of all octree leaf nodes to form a global implicit field via the MPU module in Eq. 1, which is guaranteed to be continuous. In contrast, LAS-Diffusion and XCube represent 3D shapes in thick shells with finite discrete voxels and have no such guarantees."

## Code/data

- **Code:** https://github.com/octree-nn/octfusion (Python 3.11, PyTorch 2.x with CUDA 12.1, MIT-style research license from the GitHub repo, conda-installable)
- **Pretrained:** Google Drive folder 140U_xzAy1MobUqurN67Fm2Y-3oWrZQ1m (VAE-ShapeNet-depth-8.pth + per-category df_steps-split.pth + df_steps-union.pth)
- **Data:** ShapeNet (5 categories: chair/table/airplane/car/rifle, ~31GB ShapeNetCore.v1.zip, requires ShapNet account) + Objaverse (10k subset from LGM) + Text2Shape (chair + table captions)
- **Builds on:** DualOctreeGNN (Wang 2022, SIGGRAPH), SDFusion (Cheng 2023), LAS-Diffusion (Zheng 2023), mesh2sdf
- **Inference scripts:** `sh scripts/run_snet_uncond.sh generate hr $category` (unconditional) / `sh scripts/run_snet_cond.sh generate hr $category` (category-conditional)

## For our project (v0, v1, v2)

The architecture is the *killer* H1+H2 paradigm for high-resolution 3D-gen with limited compute, but the dataset (ShapeNet chair/airplane/car) is *not* teeth, so v0 needs **dental-specific finetuning or pretraining**. Concrete next steps:

**(a) ★ ★ ★ ADOPT THE OCTREE-BASED LATENT VAE + MPU SDF DECODER AS V0 SUB-TASK 2 (CROWN GENERATION) SUBSTRATE** (the *killer* H4 substrate — continuous + manifold + arbitrary-resolution SDFs at 33M params, can mesh-extract at any resolution for clinical-fit evaluation, can compute margin gap to arbitrary precision; ~$0 Lambda for the architecture, just port the 33M-param VAE from github.com/octree-nn/octfusion; the *killer* clinical advantage is the **continuous SDF can be evaluated at any point** for margin gap, internal fit, proximal contact — not limited to mesh resolution).

**(b) ★ ★ ★ ADOPT THE UNIFIED U-NET + WEIGHT-SHARING DIFFUSION AS V0 SUB-TASK 2 (CROWN GENERATION) GENERATION PARADIGM** (the *killer* H1 paradigm — 2 stages (F1 split + F2 latent) with weight-sharing across octree levels, 33M params fits in 1GB GPU, 48ms inference on 4090 = **chairside-real-time**, the *killer* clinical advantage over DMC 033's 50-200ms or ToothCraft's diffusion overhead; for v0: replace DMC's PoinTr→FoldingNet→SAP with OctFusion's VAE→unified U-Net, keep the MRL trick and add the 6-tooth context as H3 conditioning via cross-attention).

**(c) ★ ★ ADOPT THE OCTREE'S MULTI-RESOLUTION STRUCTURE AS V0'S PREP-BOUNDARY-VS-CONTEXT PARADIGM** (the *killer* architectural insight — high-resolution leaf nodes for the prep boundary (1024³ effective), low-resolution for the surrounding 6-tooth context (64-256³), the *right* way to handle the 10:1 resolution ratio between the prep boundary (where margin gap is critical) and the full arch (where context is coarse); the *killer* clinical advantage over DMC 033's uniform 1568 points).

**(d) ★ ADOPT THE 4096-NODE SURFACE-INTERSECTING OCTREE PARADIGM AS V0'S INPUT FORMAT** (the *killer* H4 efficiency — only surface-intersecting voxels, **64× sparser** than LAS-Diffusion's volume shell, can represent the prep boundary + 6-tooth context at 0.69 GB GPU memory, the *killer* clinical deployment advantage for chairside on edge devices).

**(e) ★ ADOPT THE OCTREE VAE PRETRAINED ON SHAPENET AS V0'S INITIAL WEIGHTS** (the *killer* H5 finetuning recipe — pretrain OctFusion's VAE on ShapeNet (chairs + tables most similar to teeth), then finetune on 3DTeethSeg22 + ToSynFCD + clinical scans, the *killer* cost advantage — only the VAE needs to learn teeth-specific features, the U-Net already learns generic 3D-shape diffusion priors; ~$100-200 Lambda for dental finetuning vs $1,000+ from-scratch training).

**(f) ADOPT THE TEXT-CONDITIONED PATH (CLIP + CROSS-ATTENTION) AS V0'S DENTIST-PROMPT INTERFACE** (the *killer* UX innovation — the dentist can type "extracted molar with deep buccal decay" and the model generates matching variants; the *killer* clinical advantage over DMC 033's pure conditional generation on 6-tooth context; ~$50 Lambda + 0.5-1 day, just port the CLIP+cross-attention from OctFusion).

**(g) ADOPT THE SKETCH-CONDITIONED PATH (VIEW-AWARE LOCAL ATTENTION) AS V0'S FREEHAND-DRAWING INTERFACE** (the *killer* UX innovation — the dentist can sketch the desired crown outline and the model generates the 3D shape; the *killer* clinical advantage for tooth libraries; ~$50 Lambda + 0.5-1 day).

**(h) CITE OCTFUSION IN V0 PAPER'S RELATED-WORK + TABLE 1** (the *canonical* 2024 octree-based 3D-diffusion paper, the *de facto* 2024-2025 paradigm that **18× fewer params than XCube** + 2.8× faster inference, the *killer* practical advantage for v0's clinical deployment; v0 should compare against OctFusion in Table 1 as a non-dental 3D-gen SOTA baseline, 1 paragraph, $0).

**(i) DEFER V1 FULL-TOOTH GENERATION WITH 5-PART DECOMPOSITION** (deferred to v1+ — OctFusion's per-leaf-node latent features could be extended to per-part latent features for full-tooth generation, similar to SeaLion 150, but requires re-architecting the VAE for part-aware latents, $1,000-2,000 Lambda, 4-6 weeks; v0's prep-crown-only is enough for clinical chairside MVP).

**(j) OPEN Q: integrate OctFusion's octree-based VAE into v0's existing DMC 033 pipeline** (the *open* *Q* — replace DMC's PoinTr→FoldingNet→SAP with OctFusion's VAE→unified U-Net, keep the MRL trick for mesh quality, add the 6-tooth context as H3 conditioning; the *killer* empirical test — does OctFusion's continuous SDF + 33M-param U-Net beat DMC 033's CD-L1 0.0623 + F-score 0.70 on the same dental benchmark? if yes, **-10-20% CD + 2-5× faster inference + 50% fewer params**, the *killer* practical advantage for v0 v0 v0 v0 v0).

**v0 compute update:** +$0 Lambda for the architecture (port OctFusion's VAE from github); +$50-100 Lambda for dental finetuning (3DTeethSeg22 + ToSynFCD, 100-200 epochs, 4×4090, ~1 day); +$50 Lambda for CLIP+cross-attention (text conditioning); +$50 Lambda for sketch conditioning; **+~$150-200 Lambda total** for full OctFusion integration. **Total v0 compute unchanged at ~$5,970-7,530 Lambda** (was $5,820-7,330 from 150-note, +$150-200 for OctFusion integration).

**★ ★ ★ KEY STRATEGIC POSITIONING:** OctFusion 151 is **THE** 2024-2025 octree-based 3D-gen SOTA — the canonical 33M-param H1+H2+H4 paradigm that achieves 1024³ effective resolution with 0.69 GB GPU memory and 48ms inference on 4090. The *killer* H1 innovation is the **unified weight-shared U-Net** that processes all octree depths with a single network (vs LAS-Diffusion/XCube's 3-6 separate networks). The *killer* H2 innovation is the **octree-based latent** (4096 surface-intersecting voxels at first stage, 64× sparser than LAS-Diffusion). The *killer* H4 innovation is the **continuous + manifold implicit SDF** via MPU blending, with completeness + continuity + efficiency all superior to LAS-Diffusion/XCube. For v0, the killer contributions are **(a) octree-based VAE with MPU SDF decoder as v0's continuous-SDF substrate** (H4, the *right* substrate for clinical-fit evaluation), **(b) unified U-Net + weight-sharing diffusion as v0's generation paradigm** (H1+H2, 33M params = 2-5× smaller than DMC 033's 50M params), **(c) octree's multi-resolution structure as v0's prep-boundary-vs-context paradigm** (the *right* way to handle the 10:1 resolution ratio). For v1, the killer expansion is **dental-specific VAE finetuning** on 3DTeethSeg22 + ToSynFCD with the pretrained ShapeNet VAE as initialization (H5). For v2, the killer expansion is **per-part latents for full-tooth generation** with the same OctFusion architecture re-architected for part-aware latents.

The 3D-point-cloud-DDM arc is now a clean 7-paper sequence: **PVD 012 (ICCV 2021, point-voxel DDM) → DPM 062 (CVPR 2021, normalizing flow + weak DDM) → LION 149 (NeurIPS 2022, hierarchical VAE+DDM) → DiffFacto 147 (ICCV 2023, part-aware cross-attention DDM) → NSOT 148 (ICLR 2025, 1-stage flow) → SeaLion 150 (CVPR 2025, part-aware VAE+DDM) → OctFusion 151 (CGF/SGP 2025, octree-based VAE+DDM with weight-shared U-Net)** = 7 papers, the *de facto* 2021→2025 evolution of 3D-point-cloud diffusion models. The 3D-point-cloud-DDM-sub-task-2-for-dental arc is now: **LION 149 (canonical 2-stage 3D-gen) → DMC 033 (2023, dental-specific 1-stage) → DCrownFormer 032 (2024, MRL) → MADCrowner (2026, margin) → ToothCraft (2026, SDF diffusion) → Abbasi Moghadam 2025 (2025, dental implant) → SeaLion 150 (2025, part-aware dental extension) → OctFusion 151 (2025, octree-based VAE+DDM with weight-shared U-Net)** = 8 papers, the *de facto* 2022→2025 evolution of dental-3D-gen, with **v0 positioned to be the 9th paper** that combines OctFusion's continuous SDF + DMC's 6-tooth context + Hwang 061's clinical-fit awareness + the 8 H3 mechanisms from 061-148-149 + 150 + 151.

**★ Open Q for HK:**
- (i) Adopt OctFusion's octree-based VAE + MPU SDF decoder as v0's H4 substrate? (RECOMMEND YES — the *killer* continuous-SDF substrate for clinical-fit evaluation)
- (ii) Adopt the unified U-Net + weight-sharing diffusion as v0's H1+H2 paradigm? (RECOMMEND YES — 33M params = 2-5× smaller than DMC 033)
- (iii) Adopt the octree's multi-resolution structure for prep-boundary-vs-context? (RECOMMEND YES — the *right* way to handle the 10:1 resolution ratio)
- (iv) Adopt the 4096-node surface-intersecting octree as v0's input format? (RECOMMEND YES — 64× sparser than LAS-Diffusion, 0.69 GB GPU memory)
- (v) Adopt the ShapeNet-pretrained VAE as v0's dental-finetuning initialization? (RECOMMEND YES — $100-200 Lambda for finetuning vs $1,000+ from-scratch)
- (vi) Adopt CLIP+cross-attention for dentist-prompt interface? (RECOMMEND YES for v1 — $50 Lambda, 0.5-1 day)
- (vii) Adopt sketch-conditioning for freehand-drawing interface? (RECOMMEND DEFER to v1 — $50 Lambda, 0.5-1 day)
- (viii) Cite OctFusion in v0's Table 1 as non-dental 3D-gen SOTA? (RECOMMEND YES — 1 paragraph, $0)
- (ix) v1 full-tooth generation with per-part latents? (RECOMMEND DEFER to v1+ — $1,000-2,000 Lambda, 4-6 weeks)

**★ Next paper to read (152):** the 151-note's recommended *next* is **(a) Trellis (Xiang et al. 2024, structured 3D latents for scalable 3D-gen, arXiv:2412.01506)** — the *sparse-voxel* counterpart to OctFusion 151, with structured 3D latents that enable versatile 3D generation (image-conditioned + text-conditioned + 3D-conditioned in one model); **(b) LGM (Tang et al. 2024, Large Multi-View Gaussian Model, arXiv:2402.05054)** — the *multi-view Gaussian* counterpart to OctFusion 151, with feed-forward 3D-gen from 4 multi-view images in 5 seconds; **(c) TripoSR (Tochilkin et al. 2024, TripoSR: Fast 3D Object Reconstruction from a Single Image, arXiv:2403.02171)** — the *single-image* counterpart, with feed-forward 3D-recon transformer in 0.5s on 4090.

**Recommendation: *read 152 = Trellis* (Xiang et al. 2024)** — the *sparse-voxel* counterpart to OctFusion 151. After 151 (OctFusion octree-based) + 152 (Trellis sparse-voxel-based), v0's *3D-latent-diffusion arc* has the *octree-based* (OctFusion 151) + the *sparse-voxel-based* (Trellis 152) — the *de facto* 2-substrate design space of latent diffusion for 3D-gen. Trellis's structured 3D latents are the *killer architecture innovation* for v0's *multi-modal conditioning* (image + text + 3D in one model), the *killer* clinical advantage over OctFusion 151's text-only conditioning. Note in `papers/152-...md` after writing.
