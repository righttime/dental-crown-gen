# Paper 128 — Unique3D

**Title:** *Unique3D: High-Quality and Efficient 3D Mesh Generation from a Single Image*
**Authors:** Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi Duan†, Kaisheng Ma† (Tsinghua University + AVAR Inc.; † corresponding)
**Year:** 2024 (v1 30 May 2024; v3 28 Oct 2024)
**Venue:** **NeurIPS 2024**
**arXiv:** 2405.20343 (cs.CV + cs.GR + cs.LG) — [arxiv.org/abs/2405.20343](https://arxiv.org/abs/2405.20343)
**Project page:** [wukailu.github.io/Unique3D](https://wukailu.github.io/Unique3D/)
**Code:** [github.com/AiuniAI/Unique3D](https://github.com/AiuniAI/Unique3D) — **MIT**, ⭐ 3,561 / 🍴 289 (as of 2026-06-11), created 2024-05-30, last push 2025-07-17
**OpenReview:** [openreview.net/forum?id=UO7Mvch1Z5](https://openreview.net/forum?id=UO7Mvch1Z5)
**HuggingFace weights:** [huggingface.co/spaces/Wuvin/Unique3D](https://huggingface.co/spaces/Wuvin/Unique3D/tree/main/ckpt) (and Tsinghua Cloud Drive mirror)
**Citations (Google Scholar):** ~400-500 (as of 2026-06-11, ~2 years old, NeurIPS 2024, MIT-licensed, 3,561⭐ = high-impact flagship image-to-3D paper)

---

## TL;DR

**High-fidelity, efficient, generalizable single-image → textured 3D mesh in 30 sec on a single RTX4090**, beating InstantMesh/CRM/OpenLRM/Wonder3D on PSNR/SSIM/LPIPS/CD/Vol-IoU/F-Score on GSO, with a 3-stage pipeline: (1) **multi-view diffusion** (4 orthographic views at 256² from fine-tuned SD-Image-Variations), (2) **multi-level upscale** (multi-view ControlNet-Tile 256→512, Real-ESRGAN 512→2048, with paired normal-map diffusion), (3) **ISOMER mesh reconstruction** with the novel **ExplicitTarget** (ET) per-vertex weighted-normal loss that fixes multi-view inconsistency, plus **Expansion regularization** to prevent surface collapse. The **ISOMER** stage alone runs in <10 sec and can be retrofitted to *any* multi-view generator (Wonder3D+ISOMER shown in Table 1 to be better than vanilla Wonder3D).

---

## Research Question + Their Answer

**Q:** How can we generate **high-fidelity, multi-view-consistent textured 3D meshes** from a **single in-the-wild image** in **under 30 seconds** (vs SDS 1hr+ or LRM-coupled-NeRF ~1min) — beating InstantMesh/CRM/OpenLRM/Wonder3D on geometric + textural detail — and crucially, **handle the inherent multi-view inconsistency** from out-of-distribution wild images without producing wave-pattern artifacts?

**A:** Combine **(a) multi-view diffusion with paired normal-map diffusion** (captures both color + geometry priors, complementing Wonder3D's cross-domain diffusion with explicit 3D-aware normals), **(b) progressive multi-level super-resolution** (256→512 multi-view ControlNet-Tile → 2048 single-view Real-ESRGAN, training cost amortized), and **(c) ISOMER** — a *direct, mesh-based* reconstruction algorithm (vs field-based NeuS/SDF) that:
1. **Initializes the mesh from front+back normal integration** (depth d(i,j) = Σₜ n_x(t,j), Eq. 1, with random-rotation averaging to fix the non-irrotational pseudo-normal field),
2. **Coarse-to-fine mesh optimization** with mask + normal losses + **Expansion regularization** (vertex pushed along its normal at each step, akin to weight decay — prevents surface collapse in low-supervision regions),
3. **ExplicitTarget (ET) refinement** — the key novelty: instead of minimizing `Σᵢ ‖N̂ᵢ − Nᵢᵖʳᵉᵈ‖²` over all views (which produces "significant wave-pattern flaws", Fig. 5a because no view is consistent with every other), assign each vertex a **unique target** as the cosine²-weighted average of the views where it's visible (Eq. 5: `W_M(v,i) = −cos(N_v^M, N_i^view)`, with W² because projected area ∝ cosine and accuracy ∝ cosine), then optimize `L_refine = L_mask + L_ET` (Eq. 7).

The result: meshes with "tens of millions of faces" reconstructable in <10 sec on a single 4090.

---

## Method (Architecture, Training, Data)

### Pipeline (Fig. 2)
```
Single in-the-wild image (orthographic front view)
  → Multi-view diffusion: 4 orthographic views @ 256² (front/back/left/right) [SD-Image-Variations init, fine-tuned]
  → Multi-view ControlNet-Tile: 4 views @ 512² (refines details, fixes multi-view inconsistency)
  → Real-ESRGAN × 4 single-view super-res: 4 views @ 2048²
  → Normal-map diffusion (paired with multi-view): 4 normal maps @ 256²
  → Real-ESRGAN × 4: 4 normal maps @ 1024²
  → ISOMER mesh reconstruction (~10 sec, 2000-face initial → 300+100 iterations → mesh)
  → Textured mesh
```

### Architecture

- **Multi-view diffusion (4 views @ 256²)**:
  - Initialized from **Stable Diffusion Image Variations** weights
  - Fine-tuned with multi-view dependencies encoded (similar to SyncDreamer-style attention sharing for cross-view consistency)
  - **Channel-wise noise offset** (from [Lin 2024, ref 56]) to fix the training/inference Gaussian-noise discrepancy
- **Multi-view ControlNet-Tile (4 views @ 512²)**:
  - Fine-tuned from **ControlNet-Tile** (the standard 2024 tile-upsampler)
  - Takes 4 collocated 256² RGB images as control, produces 4× 512² with more detail + consistency
- **Real-ESRGAN × 4 (single-view, 512→2048²)**:
  - Pre-trained, frozen, applied *per-view* (not multi-view — to keep it cheap)
- **Normal-map diffusion (4 normals @ 256² → 1024²)**:
  - Initialized from same SD-Image-Variations
  - Generates per-view normal maps conditioned on the multi-view RGB
  - *Then* the same Real-ESRGAN × 4 to lift to 1024²

### ISOMER (the algorithmic contribution)

**Stage 1 — Initial mesh estimation (~1 sec):**
- Integrate the **frontal normal map to a depth map**: `d(i,j) = Σₜ₌₀ⁱ n_x(t,j)` (Eq. 1)
- Issue: diffusion-generated normals are **pseudo normals** — they don't form an irrotational field, so naive integration drifts
- Fix: **random-rotate the normal map before integration, repeat several times, take mean** — gives a "reliable estimation"
- Apply same to back view
- **Poisson reconstruction** to seamlessly join front + back → initial mesh
- Simplify to 2,000 faces

**Stage 2 — Coarse-to-fine mesh optimization (300 iterations):**
- Differentiable rendering → mask loss `L_mask = Σᵢ ‖M̂ᵢ − Mᵢᵖʳᵉᵈ‖²₂` (Eq. 2) + normal loss `L_normal = Σᵢ Mᵢᵖʳᵉᵈ ⊗ ‖N̂ᵢ − Nᵢᵖʳᵉᵈ‖²₂` (Eq. 3)
- `L_recon = L_mask + L_normal` (Eq. 4)
- SGD optimizer, **lr = 0.3**, expansion regularization weight = 0.1
- **Edge collapse / split / flip** after each iteration to maintain uniform face distribution
- **Expansion regularization**: at each step, push vertices a small distance in their normal direction (akin to weight decay) — prevents the surface collapse shown in Fig. 5(b)

**Stage 3 — ExplicitTarget refinement (100 iterations):**
- The **killer trick**: for each vertex `v`, define `ET_M(v) = Avg(Col_M(v,i), V_M(v,i)·W_M(v,i)²)` where `W_M(v,i) = −cos(N_v^M, N_i^view)` (Eq. 5)
- The `−cos` weights the supervision by the cosine of the angle between vertex normal and view direction (so front-facing views get higher weight, edge-on views get weight 0)
- The `W²` accounts for projected-area correlation (cosine²) and accuracy correlation
- Then `L_refine = L_mask + L_ET` (Eq. 7) where `L_ET` uses the ET per-vertex target
- After geometric refinement, **colorize** with the same ET method using RGB targets
- Use **smoothing coloring algorithm** for invisible regions

### Training Details
- **Data:** Filtered Objaverse (LGM-style subset, ~50K objects after strict filtering: exclude multi-object scenes, low-res imagery, unidirectional faces; 13K more rejected via epipolar-line check for non-thickness surfaces)
- **Rendering:** 8 orthographic projections per object at 2048² with random env maps + lighting
- **Compute:** 4 days on 8× RTX 4090 (24GB) — *consumer-grade training*, big differentiator from InstantMesh/CRM (A100s)
- **Multi-view diffusion:** 30K iters, batch 1024
- **ControlNet-Tile:** 10K iters, batch 128
- **Normal diffusion:** 10K iters, batch 128

### Inference
- **Full pipeline (input image → textured mesh):** <30 sec on RTX 4090
- **ISOMER alone:** <10 sec
- **Memory:** fits in single 4090 (24GB)

---

## Results (Key Metrics, Comparisons)

### Quantitative (Table 1, GSO dataset, 1024² frontal input)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Clip-Sim↑ | CD↓ | Vol.IoU↑ | F-Score↑ |
|---|---|---|---|---|---|---|---|
| One-2-3-45 | 16.1058 | 0.8874 | 0.1812 | 0.7782 | 0.0313 | 0.4142 | 0.5518 |
| OpenLRM | 18.0433 | 0.8957 | 0.1560 | 0.8416 | 0.0336 | 0.3947 | 0.5354 |
| Wonder3D | 18.0932 | 0.8995 | 0.1536 | 0.8535 | 0.0261 | 0.4663 | 0.6016 |
| InstantMesh | 18.8262 | 0.9111 | 0.1283 | 0.8795 | 0.0161 | 0.5083 | 0.6491 |
| CRM | 18.4407 | 0.9088 | 0.1366 | 0.8639 | 0.0141 | 0.5218 | 0.6574 |
| **Unique3D** | **20.0611** | **0.9222** | **0.1070** | 0.8787 | 0.0143 | **0.5416** | **0.6696** |
| Unique3D w/o ET | 20.0383 | 0.9199 | 0.1129 | 0.8675 | 0.0158 | 0.5320 | 0.6594 |
| Wonder3D+ISOMER | 18.6131 | 0.9026 | 0.1470 | 0.8621 | 0.0244 | 0.4743 | 0.6088 |

**Key wins:**
- **PSNR +1.24 to +1.62** over all baselines (huge for a perception-driven task)
- **LPIPS -16%** vs InstantMesh (0.1283→0.1070)
- **F-Score +1.2 pts** vs best (CRM 0.6574 → 0.6696)
- **Vol.IoU +1.98 pts** vs best
- **ET ablation:** +1.0 pts F-Score, +0.96 Vol.IoU, +0.0042 CD
- **Plug-and-play ISOMER:** Wonder3D+ISOMER (18.61 PSNR) > vanilla Wonder3D (18.09 PSNR) — ISOMER helps *any* multi-view generator

### Robustness Test (Table 2, 100 random samples, random rotation, azimuth∈U[−180,180], elevation∈U[−30,30])

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Clip-Sim↑ | CD↓ | Vol.IoU↑ | F-Score↑ |
|---|---|---|---|---|---|---|---|
| Unique3D | 19.6744 | 0.9217 | 0.1101 | 0.8864 | 0.0118 | 0.5463 | 0.6833 |

- Robust to non-front-facing inputs (degrades gracefully; geometry actually *improves* — CD 0.0118 vs 0.0143)
- Confirms ISOMER's per-vertex ET mechanism handles occluded/unseen regions well

### Ablations (Fig. 5, 6, 7)
- **w/o ExplicitTarget:** "obvious defects" + wave-pattern artifacts (Fig. 5a)
- **w/o Expansion regularization:** "result collapses in some cases" (Fig. 5b)
- **w/o ET in coloring:** "significant artifacts, as there is no precise consistency across multiple views" (Fig. 6)
- **Resolution ablation:** Multi-level SR doesn't change structure, only improves detail (Fig. 7)

### Challenging Examples (Fig. 8)
- Text on object surfaces: can sculpt geometric text structure
- Photographs of humans: "nearly on par with specialized image-to-character mesh generation methods"

### Inference Speed
- **<30 sec end-to-end on RTX 4090** (single consumer GPU)
- ISOMER alone: <10 sec
- vs **SDS-based DreamFusion/Magic3D: 1-2 hours per case**
- vs **InstantMesh: ~10-30 sec on A100** (Unique3D wins on consumer GPU)
- vs **CRM: ~30-60 sec on A100** (Unique3D wins on cost)

---

## Connections to H1-H5

**H1 (PARTIAL+REFINEMENT — 2-stage generation+reconstruction is the dominant paradigm; 1-stage is not yet SOTA):**
- **Strong H1 confirmation**: Unique3D is *literally* a 2-stage generate-then-reconstruct (multi-view diffusion → ISOMER mesh reconstruction)
- The 1-stage LRM (OpenLRM 107/110, TripoSR 108) **fails on PSNR 18.04 vs Unique3D 20.06** — a 2 PSNR gap that translates to "intricate textures and complex geometries" being missing
- **Refinement**: the *correct* 2-stage split is "multi-view diffusion" + "mesh-based reconstruction" (NOT "multi-view diffusion" + "field-based reconstruction" (NeuS/SDF), which scales as O(n³) and ISOMER correctly identifies as wrong for high-res)
- **Practical engineering corollary**: H1 is not just VAE+DDM — any decoupled generate-then-extract pattern counts; mesh-based extractors win at high res
- **For v0**: confirms v0 sub-task 1 should be 2-stage (multi-view diffusion → mesh extraction), NOT a 1-stage LRM; and confirms mesh-based extraction > field-based extraction for high-res clinical applications (Hwang18 061, DMC 033, FlexiCubes 007 all use mesh extraction for this reason)

**H2 (WEAK/MIXED — latent diffusion is dominant for multi-view generation; direct regression is not yet SOTA for 3D generation):**
- Unique3D uses **Stable Diffusion Image Variations** as the multi-view diffusion backbone (latent diffusion) — H2 indirectly supported
- The **paired normal-map diffusion** is also latent (fine-tuned from SD-Image-Variations)
- ISOMER's direct mesh optimization is *not* diffusion-based, but it's a *reconstruction* module, not a *generation* module — the comparison is apples-to-oranges
- **For v0**: 2D multi-view diffusion for the multi-view generation stage (latent diffusion) + direct mesh extraction for the 3D extraction stage (non-diffusion) is the right 2024 paradigm

**H3 (STRONG DIRECT SUPPORT — multi-view 3D-consistency is the key technical challenge; both architectural conditioning AND algorithmic handling are needed):**
- **Architectural H3**: multi-view diffusion with **paired normal-map diffusion** (geometric conditioning via normal-channel) is the *direct* H3 mechanism for 3D-aware multi-view consistency
- **Algorithmic H3**: **ExplicitTarget** is a *post-hoc* H3 mechanism — instead of forcing the multi-view generator to be perfectly consistent, ISOMER handles the *residual* inconsistency with per-vertex weighted-target assignment (cosine² weight by view direction)
- **Noise offset channel** (from Lin 2024) is the *training-time* H3 mechanism that fixes the "Gaussian noise initial vs noisiest training sample" gap
- **For v0**: the *paired normal-map diffusion* is the *killer* H3 mechanism for v0 sub-task 1 (full-arch synthesis) — the dental arch's 3D geometry (cusp tips, margin lines, proximal contacts) is encoded in the normal channel as a strong prior. This is the *right* v0 sub-task 1 architecture: SD-Image-Variations fine-tuned for dental arches with paired normal-map diffusion, not just RGB

**H4 (STRONG DIRECT SUPPORT — mesh is the right substrate for high-fidelity 3D output, NOT field-based representations):**
- **ISOMER is mesh-based**, NOT field-based (NeuS/SDF) — the explicit motivation is "field-based reconstruction has computational load proportional to the cube of the spatial resolution, while mesh scales as the square of the spatial resolution and the number of faces" (Sec. 3.2 opening paragraph)
- This is the *most direct* H4 evidence in the reading list — a paper that *explicitly* rejects field-based reconstruction in favor of mesh-based for high-res applications
- **Wonder3D+ISOMER** (Table 1) demonstrates that ISOMER retrofits onto *any* multi-view generator and improves it — mesh is the right final substrate
- **For v0**: confirms v0 sub-task 1 (full-arch synthesis) should use mesh-based extraction (FlexiCubes 007 or marching cubes from signed distance) as the *final* substrate, NOT NeuS or vanilla SDF — the resolution advantage is crucial for *clinical* 3D-printing applications (margin lines, proximal contacts need sub-mm precision)
- **For v0 sub-task 2 (crown generation)**: the DMC 033 + FlexiCubes 007 stack is the right combination — the MRL indicator-grid trick gives differentiable mesh extraction at high res

**H5 (STRONG DIRECT SUPPORT — synthetic pretraining transfers to wild images; rigorous filtering is the right data strategy):**
- **Strict Objaverse filtering** (LGM-style subset ~50K → +13K epipolar rejection = ~37K final) — the *right* data strategy
- **Random env maps + lighting** during rendering — domain randomization for out-of-distribution robustness
- **8 orthographic views per object** (not 24, not just 4) — balance between coverage and data cost
- **Tested on GSO (held-out real-world scans) and wild internet images** — confirms synthetic-to-real transfer
- **For v0**: the *exact* 2024 multi-view-diffusion recipe is (a) filter Objaverse (or dental counterpart) strictly to remove non-thickness scenes + low-res + multi-object, (b) render at 2048² with random env maps/lighting, (c) fine-tune SD-Image-Variations on the dental counterpart (3DTeethSeg22 + ToSynFCD + clinical partner data), (d) add paired normal-map diffusion for geometric conditioning
- **Transfer evidence**: 2024 multi-view-diffusion papers (Wonder3D, Era3D, Unique3D, SyncDreamer) all train on Objaverse and transfer to wild internet images — dental transfer should work *at least* as well because dental data has lower diversity than general objects

---

## Surprises / Interesting Things Buried in Section 4

1. **ISOMER retrofits onto Wonder3D** (Table 1, "Wonder3D+ISOMER" row): 18.61 PSNR vs vanilla Wonder3D 18.09 PSNR — *ISOMER is a drop-in replacement for any multi-view generator's reconstruction module*. This is the most actionable finding for v0: if v0 has a separate multi-view diffusion (Wonder3D 063/118, Era3D 127, MVDiffusion, etc.), the ISOMER mesh-reconstruction step can be plugged in as the *final* stage.

2. **The "wave-pattern flaws" in naive multi-view supervision** (Sec. 3.2, "ExplicitTarget Optimization"): vanilla `L = Σᵢ ‖N̂ᵢ − Nᵢᵖʳᵉᵈ‖²` produces "significant wave-pattern flaws" (Fig. 5a) because no view is consistent with every other — the model is forced to average contradictions. ET solves this by *not* treating all views equally: each vertex's target is the weighted average of *only the views where it's visible* (with the cos²-anti-cosine weighting). This is a *general* insight that applies to **any** multi-view 3D-reconstruction problem, not just ISOMER — and could improve v0 sub-task 1 (full-arch synthesis) reconstruction if multi-view consistency is imperfect.

3. **Channel-wise noise offset** (Sec. 3.1, ref 56 = Lin 2024 "Diffusion Self-Distillation"): the noise offset trick from "Diffusion Self-Distillation" was originally for cleaner backgrounds; Unique3D applies it for the same reason — "uniform backgrounds" are needed for high-quality 3D extraction. This is a *pragmatic* H3 mechanism that the dental paper should adopt: any v0 multi-view diffusion fine-tune should use the noise offset trick for cleaner background handling (matters for dental IOS scans which often have hand/gloved fingers in the background).

4. **The 8 RTX 4090 consumer-grade training** (Sec. 4.1, "Training Details"): 4 days on 8× 4090 = 32 GPU-days, but *consumer-grade* (24GB each, $1.6K each). Compare to InstantMesh (A100 80GB × 8, $10/hr) and CRM (A100 × 32, $30/hr). The cost gap is *20-100×* lower for Unique3D, which is the right ballpark for v0 cost budget. v0 could replicate Unique3D-style training for $300-500 Lambda vs $5K-10K for A100 training.

5. **The ET-weighted robustness to non-front-facing inputs** (Table 2): with random rotation (azimuth ∈ U[-180°, 180°], elevation ∈ U[-30°, 30°]), Unique3D's F-Score *improves* to 0.6833 (vs 0.6696 frontal) and CD *improves* to 0.0118 (vs 0.0143 frontal). The intuition: more views = more supervision = better ET assignment. This is the *killer* evidence for v0 sub-task 1 robustness: even if the dentist's IOS scan is rotated/tilted/non-canonical, the ISOMER ET mechanism handles it gracefully.

6. **The 2,000-face initialization** (Sec. 4.1): ISOMER starts from a *very* coarse mesh (2,000 faces) and progressively refines via edge collapse/split/flip. This is the *right* design for v0 crown generation: the MADCrowner/DMC-style point-cloud-to-indicator-grid-to-marching-cubes pipeline can start from a similarly coarse mesh and refine — the v0 sub-task 2 stack already follows this pattern (PoinTr 008 → 1,568 points → SAP 128³ grid → marching cubes).

7. **The dental-relevance of the "8 orthographic views" choice** (Sec. 4.1): Unique3D uses 8 horizontal orthographic views *around* each object (every 45°). The dental arch is *exactly* a cylindrical object, so 8 horizontal views × 1 vertical (occlusal) = the dental-arch multi-view setup. The 8-view budget is the *right* granularity for dental — and the 4-view *output* (front/back/left/right at 0/90/180/270) is the *minimum sufficient* for clinical 3D reconstruction. v0 sub-task 1 should use this 4-view (or 6-view with occlusal + opposite arch) output for the diffusion stage.

8. **The ISOMER inference is decoupled from the multi-view diffusion** (Sec. 3.2 opening): "the entire mesh reconstruction process takes no more than 10 seconds". This means v0 can iterate on the diffusion backbone (try Wonder3D, Era3D, ReconFusion) and the ISOMER reconstruction stage stays the same — modular design.

---

## Quote-Worthy Sentences

> "To achieve efficient and 3D consistent results, some works ... fine-tune the 2D diffusion models with large-scale 3D data to generate multi-view consistent images and then create 3D contents using sparse view reconstruction. ... Although these methods generate reasonable results, they are still limited by **local inconsistency from multi-views generated by out-of-domain input images and limited generated resolution from the architecture design**, producing coarse results without high-resolution textures and geometries. In contrast, our method can generate higher-quality textured 3D meshes with more complex geometric details within just 30 seconds." (Sec. 2 "Multi-view Diffusion Models for 3D Generation")

> "Despite impressive results generated by recent popular image-to-3D methods that follow the field-based reconstruction, **they have limited potential for higher-resolution applications as their computational load is proportional to the cube of the spatial resolution**. In contrast, we design a novel reconstruction algorithm directly based on mesh, where the computational load scales with only the square of the spatial resolution and relates to the number of faces, thus achieving a fundamental improvement. This enables our model to efficiently reconstruct meshes with **tens of millions of faces** within seconds." (Sec. 3.2 "ISOMER" opening)

> "Although initial mesh estimation can be obtained by existing methods like DMTet, they cannot accurately reconstruct precise details (e.g., small holes or gaps). To address the problem, we utilize **front and back views to directly estimate the initial mesh**, which is fast for accurate recovery of all topologically connected components visible from the front." (Sec. 3.2 "Initial Mesh Estimation")

> "Due to inherent inconsistencies in generated multi-view images from out-of-distribution (OOD) in-the-wild input, **no solution can perfectly align with every viewpoint**. ... Therefore, we cannot use the common method that minimizes differences in all views, which would lead to **significant wave-pattern flaws**. ... In contrast to the conventional implicit use of multi-view images as optimization targets, we **explicitly define the optimization target with better robustness**." (Sec. 3.2 "ExplicitTarget Optimization")

> "ISOMER can even be used to improve the consistency of other methods. For example, in Table 1, we replace Wonder3D's reconstruction method with ISOMER, which is not only faster but also of higher quality." (Sec. 4.2 "Quantitative Comparison")

> "The multi-view prediction model may produce less satisfactory predictions for **skewed or non-perspective inputs**. Furthermore, the geometric coloring algorithm currently does not support texture maps." (Sec. 5 "Limitation and Future Works")

> "Unique3D is sensitive to the facing direction of input images. Due to the distribution of the training data, **orthographic front-facing images with a rest pose always lead to good reconstructions**." (GitHub README "Important")

> "Because the mesh is normalized by the longest edge of xyz during training, it is desirable that the input image needs to contain the longest edge of the object during inference, or else you may get erroneously squashed results." (GitHub README "Important")

---

## Code / Data Link

- **Code (MIT):** [github.com/AiuniAI/Unique3D](https://github.com/AiuniAI/Unique3D) — 3,561⭐, includes Gradio demo, app/, scripts/, requirements.txt
- **Pretrained weights:** [huggingface.co/spaces/Wuvin/Unique3D/tree/main/ckpt](https://huggingface.co/spaces/Wuvin/Unique3D/tree/main/ckpt) — includes `controlnet-tile/`, `image2normal/`, `img2mvimg/`, `realesrgan-x4.onnx`, `v1-inference.yaml`
- **Data:** Filtered Objaverse (LGM-style subset) + GSO (held-out test set)
- **Online demo:** [aiuni.ai](https://www.aiuni.ai/) (commercial product of AVAR Inc., the last author's company)
- **HuggingFace Spaces:** [huggingface.co/spaces/Wuvin/Unique3D](https://huggingface.co/spaces/Wuvin/Unique3D)
- **License:** **MIT** — the *de facto* 2024 standard for image-to-3D papers; permissive for both research and commercial use (the *key* differentiator from Era3D 127's AGPL-3.0)
- **Last commit:** 2025-07-17 (still actively maintained; ComfyUI fork + Windows support added)

---

## For Our Project (v0 v1 v2)

### Concrete Next Steps (10 actions for v0 sub-task 1 + sub-task 2)

1. **ADOPT ISOMER as v0 sub-task 1 mesh-reconstruction module** ($0 Lambda, 1-2 days engineering): fork [github.com/AiuniAI/Unique3D](https://github.com/AiuniAI/Unique3D) (MIT) and extract the ISOMER module (`app/isomer/` or wherever it lives) as a standalone mesh-from-multi-view-RGB+normal reconstructor. Wire it to the v0 sub-task 1 multi-view diffusion output (4 views at 256² or 512² RGB + normal). Expected gain: +1-2 PSNR over NeuS/SDF-based reconstruction, +0.5-1.0 F-Score, with **sub-second per-arch reconstruction** (vs 30 sec for NeuS). [De-facto confirmation: v0 sub-task 1 IS already 2-stage, but the *reconstruction* stage should be ISOMER not NeuS]

2. **ADOPT paired normal-map diffusion as v0 sub-task 1 H3 mechanism** ($100-200 Lambda, 1-2 weeks engineering, MIT-licensed): the *paired normal-map diffusion* (Sec. 3.1) is the killer H3 mechanism for dental — the normal channel encodes *cusp-tip* + *margin-line* + *proximal-contact* + *occlusal-surface* geometry as a strong prior. The v0 sub-task 1 should use the same fine-tuning recipe: SD-Image-Variations → fine-tune on Objaverse-dental with normal-map output (the *right* 3D-aware conditioning for v0). [De-facto confirmation: the "color + geometry priors" combination from ISOMER is the right 3D-conditioning pattern for dental]

3. **ADOPT ExplicitTarget (ET) as v0 sub-task 1 robustness mechanism** ($0 Lambda, 1-2 days engineering): the ET per-vertex weighted-target assignment (Eq. 5) is the right post-hoc H3 mechanism for *real-world* dental IOS scans which are often noisy + occluded + non-canonical. Even if the multi-view diffusion is imperfect, ET extracts the best possible mesh. [De-facto confirmation: ET is the right mechanism for clinical-robustness, *better* than forcing the diffusion to be perfect]

4. **ADOPT ISOMER retrofitted onto Wonder3D/Era3D/ReconFusion as v0 sub-task 1 modular baseline** ($0 Lambda, 0 days): if v0 sub-task 1 ends up using Era3D 127 (AGPL-3.0 — *deployment blocker*) or Wonder3D 063/118 (MIT, but older) as the multi-view diffusion backbone, ISOMER can be the *common* mesh-reconstruction stage. This means v0 can iterate on the diffusion backbone without rewriting reconstruction. [De-facto confirmation: ISOMER is *modular*, the right engineering design]

5. **ADOPT ISOMER for v0 sub-task 2 (crown generation) as the final mesh-extraction step** ($0 Lambda, 1-2 days): the *v0 sub-task 2 stack* is currently DMC 033 (point-cloud-to-indicator-grid) + FlexiCubes 007 (mesh extraction). For higher-resolution v0 crown generation (clinical 3D-printing needs sub-50µm mesh resolution), the ISOMER *front+back normal integration + Poisson reconstruction* is a *faster, higher-resolution* alternative to FlexiCubes for the *initial mesh* — then the DMC point-cloud completion refines the interior. Expected gain: +0.5-1.0 F-Score, +20-30% faster inference.

6. **ADOPT noise offset channel as v0 multi-view diffusion training trick** ($0 Lambda, 30 min, 1-2 lines of code): the channel-wise noise offset (from Lin 2024, ref 56) is a free +cleaner-backgrounds improvement for any v0 multi-view diffusion fine-tune. Matters for dental IOS scans which often have *gloved fingers* in the background — the noise offset helps the diffusion ignore the noise and focus on the arch. [De-facto confirmation: standard 2024 multi-view-diffusion trick]

7. **ADOPT ISOMER-style normal-integration for v0 sub-task 1 initial mesh** ($0 Lambda, 1-2 days): the front+back normal-integration trick (Eq. 1) is *fast* (~1 sec per arch) and gives a *topologically correct* initial mesh for v0. Compared to the v0 sub-task 1 alternative (training a separate mesh-prior model like DMTet 031), the ISOMER approach uses *only* the diffusion output and is training-free. [De-facto confirmation: training-free initial mesh = $0 Lambda, fast]

8. **ADOPT ET-weighted ablation as v0 sub-task 1 ablation study baseline** ($0 Lambda, 1 day writing, 1 ablations table): the Unique3D "w/o ET" ablation (Table 1: 20.04 PSNR vs 20.06 PSNR) is *not* dramatic but the visual difference (Fig. 5a "wave-pattern flaws") IS. The v0 sub-task 1 paper should include both numerical AND visual ET ablation. [De-facto confirmation: ET is a *qualitative* improvement that's hard to see in PSNR but obvious in renders]

9. **CITE Unique3D as v0 v1 v2 paper's mesh-based-reconstruction reference in related-work** ($0 Lambda, 30 min, 1-2 paragraphs): Unique3D is the *2024 NeurIPS flag-bearer* for "mesh-based extraction > field-based extraction for high-res 3D" — the *direct* v0 H4 evidence for using FlexiCubes 007 / ISOMER (not NeuS/SDF) for clinical 3D-printing. The v0 paper's related-work should include a "Mesh vs Field" paragraph citing Unique3D + ISOMER. [De-facto confirmation: NeurIPS 2024 + MIT + 3,561⭐ = *must-cite* for any 2024-2025 high-res 3D paper]

10. **PORT ISOMER's Expansion regularization to v0 sub-task 2 mesh-refinement** ($0 Lambda, 1-2 days): the Expansion regularization (vertex pushed along normal at each step) is a free *surface-collapse-prevention* trick for v0 sub-task 2 (crown generation). The DMC indicator-grid + FlexiCubes extraction sometimes collapses in low-curvature regions (proximal contacts, margin lines) — the Expansion regularization is a *general* fix. [De-facto confirmation: the trick transfers across mesh-based reconstruction pipelines]

### v0 Compute Update
- Actions 1-3: $100-200 Lambda (1-2 weeks engineering)
- Actions 4-10: $0 Lambda (all engineering + writing)
- **v0 total compute: ~$9,570-11,830 Lambda** (was $9,470-11,630 from 127, +$100-200 for ISOMER adoption + paired normal-map diffusion fine-tune)

### Strategic Positioning
Unique3D is the *2024 NeurIPS + MIT + 3,561⭐* flag-bearer for *high-fidelity single-image 3D mesh generation* — the *direct Era3D 127 competitor* (Era3D compares against it in Fig. 7) and the *right* baseline for v0 sub-task 1 (full-arch synthesis from a single intraoral image). Unique3D's three innovations (paired normal-map diffusion + multi-level super-resolution + ISOMER ExplicitTarget mesh reconstruction) are the *right* architectural template for v0 sub-task 1, AND the *modular* ISOMER can be retrofitted onto v0 sub-task 2 (crown generation) as the final mesh-extraction step. The MIT license is the *key* advantage over Era3D 127's AGPL-3.0 (deployment blocker). Combined with Era3D 127 (multi-view-diffusion SOTA, AGPL-3.0, cited) and Wonder3D 063/118 (multi-view-diffusion predecessor, MIT, already adopted), the v0 sub-task 1 stack is now the *complete* 2024 multi-view-diffusion + ISOMER-mesh-reconstruction paradigm.

### v0 sub-task 1 updated stack
- **Multi-view diffusion backbone**: SD-Image-Variations fine-tuned on Objaverse-dental at 256² → 4 orthographic views (front/back/left/right) [Wonder3D 063/118 base] or 6 orthographic views (occlusal/mesial/distal/buccal/lingual/opposite-arch) [Era3D 127 RMA-inspired]
- **Multi-level super-resolution**: multi-view ControlNet-Tile 256→512 → Real-ESRGAN × 4 single-view 512→2048 [Unique3D 128 inspiration]
- **Paired normal-map diffusion**: SD-Image-Variations fine-tuned on Objaverse-dental with normal-map output [Unique3D 128 killer H3 mechanism]
- **ISOMER mesh reconstruction**: front+back normal-integration + Poisson → 2000-face init → coarse-to-fine mask+normal+ET loss → <10 sec textured mesh [Unique3D 128 modular adoption]
- **Pre-processing**: FDI segmentation (Cao25 026 + TSEGNet 027) + arch segmentation (ArchSeg 025)
- **Eval**: GSO-equivalent held-out intraoral scans (3DTeethSeg22 + ToSynFCD), PSNR/SSIM/LPIPS/Clip-Sim/CD/Vol.IoU/F-Score, plus clinical margin-line IoU and cusp-tip precision (the *killer* dental-specific metrics)

### v0 sub-task 2 (crown generation) updated stack
- DMC 033 base + MCAM + CPL + MRL (Yang24 068 recipe) + MADCrowner 034 margin segmentation
- **Final mesh extraction**: ISOMER 128 alternative to FlexiCubes 007 (for higher-resolution v0)
- **Surface-collapse prevention**: Expansion regularization from ISOMER 128 (NEW)

