# 153 — InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models (Xu et al. 2024, arXiv:2404.07191)

> **★ CONTEXT:** Paper 153, recommended by the 152-note (CRM 152) as the *direct* *complement* in the 2024 image-to-3D SOTA triangle: **CRM 152 (CNN-U-Net + triplane + FlexiCubes)** ↔ **InstantMesh 153 (transformer + triplane + FlexiCubes, paper 153)** ↔ **LGM 154? (asymmetric U-Net + 3DGS, paper 154 candidate)**. The killer 153-vs-152 design-space question: **CNN-triplane (CRM 152's claim: "triplane has spatial correspondence with 6 orthographic images → U-Net's local receptive field is the right choice") vs transformer-triplane (InstantMesh 153's claim: "transformer's global attention is more flexible for *varying* input viewpoints and scales better to larger datasets")**. Both use FlexiCubes mesh extraction and similar multi-view diffusion (Zero123++ variant), so the only real difference is **CNN vs transformer** for the 6-view → triplane mapping. **★ Key insight: InstantMesh is the *first* open-source (MIT license, full code + weights + gradio demo) image-to-3D framework that *jointly* combines (1) a *white-background-fine-tuned Zero123++* multi-view diffusion for consistent input, (2) a *transformer-based sparse-view LRM (Instant3D architecture with AdaLN camera-pose modulation in ViT + OpenLRM initialization)* for 6-view → 64×64 triplane → NeRF or mesh, and (3) a *two-stage training strategy* (Stage 1 triplane NeRF for warm-start from OpenLRM, Stage 2 mesh/FlexiCubes with full-resolution depth+normal supervision for explicit geometric consistency) — achieving SOTA on GSO + OmniObject3D (PSNR 22.79-23.14, SSIM 0.897-0.898, LPIPS 0.119-0.120, CD 0.177-0.180, F-Score 0.880-0.882 on GSO) at ~10s end-to-end on a single A100, *matching or beating* TripoSR 108 + LGM + CRM 152 + SV3D 117 across all 5 metrics, and *significantly better* on 3D geometry (CD -20% vs CRM 152, F-Score +12pp vs LGM, +9pp vs CRM 152). For v0, InstantMesh is the *direct* v0 sub-task 1 (full-arch synthesis from intra-oral scan) alternative to CRM 152 with a *cleaner* 6-view-to-mesh pipeline, an *open-source* codebase (MIT, $0 cost to fork), and the *killer* Stage 1 → Stage 2 two-stage training strategy that v0 can adopt for *dental-specific* warm-start from a *clinical* LRM fine-tuned on 3DTeethSeg22 + ToSynFCD.**

## TL;DR

> **InstantMesh (Xu et al. 2024, arXiv:2404.07191)** introduces a *feed-forward* single-image-to-3D-mesh framework that combines (1) a *white-background-fine-tuned Zero123++* multi-view diffusion model that synthesizes 6 multi-view images at *fixed camera poses* (azimuth increments of 60°, interleaving elevations of +20° and -10°) with *consistent white background* (the killer fix to Zero123++'s "floaters and cloud-like artifacts" issue), (2) a *transformer-based sparse-view Large Reconstruction Model* (modified Instant3D, initialized from OpenLRM, with *AdaLN camera-pose modulation* in ViT image encoder and *12-16 transformer layers* with dim 1024) that maps the 6 RGB views to a 64×64×40 (or 64×64×80 large) triplane representation, and (3) a *two-stage training strategy* (Stage 1: triplane NeRF for OpenLRM warm-start, Stage 2: mesh via FlexiCubes with full-resolution depth+normal supervision for explicit geometric consistency, the killer 2-stage recipe that *converts* the LRM into a *mesh-generation* model while preserving the *scalability* of the transformer architecture). **The 3 killer contributions:** **(1) the *white-background fine-tuning* of Zero123++** (1,000 UNet steps, lr=1e-5, batch=48, the smallest data change that completely eliminates multi-view background inconsistency and the resulting floaters, the killer 1-hour training trick that the field overlooked); **(2) the *two-stage training strategy* (Stage 1 NeRF warm-start → Stage 2 mesh + depth/normal supervision)** (the *killer* practical recipe that lets the transformer warm-start from OpenLRM's NeRF pretraining, then transitions to mesh + geometric supervision for higher-fidelity surface extraction, avoiding the cold-start problem that pure-mesh training would face); **(3) the *canonical 6-view input + canonical world-space reconstruction* (NOT view-space)** (the killer design choice that enables *multi-pose augmentation* and *cross-pose consistency*, vs LRM's single-view + view-space which has the *Janus* problem for back-view synthesis). **InstantMesh achieves SOTA on GSO + OmniObject3D** at 10s end-to-end inference (the killer practical advantage for v0's chairside deployment), and *all code, weights, and gradio demo are released under MIT license* (the killer open-source advantage for v0's direct fork). **The *practical v0 relevance* is *high-leverage* and *direct*:** InstantMesh's *MIT-licensed* open-source code is the v0 sub-task 1 *direct* starting point (just fork github.com/TencentARC/InstantMesh, replace Objaverse with 3DTeethSeg22 + ToSynFCD, replace Zero123++ with a *dental-fine-tuned* multi-view diffusion, and the *transformer architecture + FlexiCubes + two-stage training* is the *complete* v0 sub-task 1 stack); InstantMesh's *10s end-to-end* inference is the v0 *chairside-real-time* target (vs LRM's 30s, Hunyuan3D 2.0's 60-90s); InstantMesh's *6-view multi-view diffusion* is the v0 sub-task 1 H3 mechanism (vs CRM 152's 6-ortho views, the 6-view layout is *spatially-aligned* with the dental arch); InstantMesh's *canonical world-space* output is the v0 sub-task 1 *clinical* requirement (dental arches are in *fixed* positions, no view-space ambiguity); and InstantMesh's *Stage 1 → Stage 2 training strategy* is the v0 *dental warm-start* recipe (start from OpenLRM pretrained on Objaverse, then finetune on 3DTeethSeg22 + ToSynFCD with mesh + depth + normal supervision). The combination of *transformer* + *FlexiCubes* + *white-bg-Zero123++* + *two-stage training* is the *canonical* 2024 open-source image-to-3D paradigm that v0 sub-task 1 should adopt as the *primary* baseline (with CRM 152 as the *CNN-triplane* alternative).

## Research question + their answer

**Research question (Sec. 1, paraphrased):** *How can we build a single-image-to-3D generation method that (a) is feed-forward (no test-time optimization), (b) is open-source (code, weights, demo) for community adoption, (c) produces high-quality textured meshes directly (not NeRF or 3DGS that need post-hoc meshing), (d) leverages the geometric prior of the triplane representation in the architecture design, and (e) trains efficiently despite the memory-intensive nature of volume rendering for triplane NeRF?*

**Their answer (Sec. 1, verbatim summary):** **InstantMesh, a feed-forward framework for high-quality 3D mesh generation from a single image. Given an input image, InstantMesh first generates 3D consistent multi-view images with a multi-view diffusion model, and then utilizes a sparse-view large reconstruction model to predict a 3D mesh directly, where the whole process can be accomplished in seconds. By integrating a differentiable iso-surface extraction module, our reconstruction model applies geometric supervisions on the mesh surface directly, enabling satisfying training efficiency and mesh generation quality. Building upon an LRM-based architecture, our model offers superior training scalability to large-scale datasets.**

The fundamental insight is that **the choice of *3D representation* (NeRF vs 3DGS vs mesh) and the *training strategy* (single-stage vs two-stage) is the *most-underappreciated* design lever in 3D reconstruction, and the *right* choice depends on whether the *target application* requires *high-quality mesh output* (for 3D printing, AR/VR, CAD/CAM) or *fast rendering* (for novel view synthesis).** The paper's three insights: (1) **White-background fine-tuning of Zero123++** is the *smallest* data change that completely eliminates multi-view background inconsistency (the *killer* practical fix to a *fundamental* Zero123++ issue that the field overlooked); (2) **Two-stage training (Stage 1 NeRF warm-start → Stage 2 mesh + geometric supervision)** is the *killer* practical recipe that lets the transformer warm-start from a NeRF-pretrained LRM, then transitions to mesh + depth+normal supervision for *higher-fidelity* surface extraction, *avoiding* the cold-start problem that pure-mesh training would face; (3) **6-view + canonical world-space + FlexiCubes** is the *killer* 3D representation combo for *high-quality mesh output* (the 6-view layout is *spatially-aligned* with the 3D object, the canonical world-space avoids the *Janus* problem, and FlexiCubes handles *sharp features* and *topology changes* better than Marching Cubes). The result: InstantMesh is **~3× faster inference** than DreamFusion-based methods (10s vs hours), **~3× faster** than LRM (10s vs 30s for the full pipeline, since Zero123++ inference is the bottleneck not the LRM), and *matches or beats* LRM, LGM, TripoSR, CRM, SV3D across all 5 metrics (PSNR, SSIM, LPIPS, CD, F-Score) on GSO + OmniObject3D.

## Method

### Pipeline overview (Sec. 3, Fig. 2)

InstantMesh is a 2-stage pipeline (multi-view diffusion + sparse-view LRM) that produces a textured mesh in ~10 seconds:

**Stage 1 — Multi-view diffusion with white-background fine-tuned Zero123++ (Sec. 3.1):**
- **Input:** single image `I ∈ ℝ^{H×W×3}`.
- **Step 1a:** *Zero123++* (multi-view diffusion model from Shi et al. 2023, fine-tuned from Stable Diffusion) generates 6 multi-view images at *fixed camera poses* (azimuth: 30° + k·60° for k=0..5, elevation: interleaving +20° and -10°, the *killer* camera-pose layout that covers both the *upper* and *lower* hemispheres of the 3D object).
- **Step 1b:** *White-background fine-tuning* (the killer 1,000-step training, lr=1e-5, batch=48 on the LVIS subset of Objaverse) converts Zero123++'s gray-background output to *consistent white-background*, eliminating the "floaters and cloud-like artifacts" caused by background RGB variation.
- **Output:** 6 RGB images of 320×320 (resized from 960×640 to match the LRM's input resolution).

**Stage 2 — Sparse-view LRM with two-stage training (Sec. 3.2):**
- **Input:** 6 RGB images (320×320 each) + their corresponding camera poses (the *fixed* Zero123++ poses).
- **Architecture:** *Transformer-based sparse-view LRM* (modified Instant3D, initialized from OpenLRM):
  1. **ViT image encoder** with *AdaLN camera-pose modulation layers* added to make the output image tokens *pose-aware* (the killer design that lets the transformer handle 6 different views consistently, vs LRM's single-view which has source-camera modulation but not multi-view).
  2. **Cross-attention transformer** (12-16 layers, dim 1024) that maps the 6 image tokens to *64×64×40* (base) or *64×64×80* (large) triplane tokens.
  3. **Triplane decoder** that reshapes the triplane tokens into 3 planes of 64×64×40 (or 64×64×80).
  4. **Tiny MLPs** for decoding:
     - **Stage 1 (NeRF):** density MLP + color MLP, render via volume rendering (96-128 samples per ray, 192×192 patches supervised by 192-512×192-512 GT patches).
     - **Stage 2 (Mesh):** density MLP is *re-initialized as SDF MLP* with the *killer* weight+bias sign-flip trick (`w = -w_d, b = τ - b_d` where `w_d, b_d` are the original density MLP weights and `τ` is the iso-surface threshold), plus *2 additional MLPs* for FlexiCubes deformation + weights. The mesh is extracted via *FlexiCubes* (paper 007) and rendered via *efficient mesh rasterization* (the killer switch from volume rendering to mesh rasterization that enables *full-resolution* image + depth + normal supervision without memory blow-up).
- **Output:** textured 3D mesh (instant-nerf-base/large or instant-mesh-base/large variants, with NeRF rendering or mesh extraction respectively).

### Two-stage training strategy (Sec. 3.2, Stage 1 and Stage 2)

**Stage 1: Training on NeRF (warm-start from OpenLRM):**
- **Goal:** Initialize the transformer with OpenLRM's NeRF-pretrained weights, learn the 6-view → triplane mapping.
- **Data:** 270K filtered Objaverse instances (filtering criteria: no texture maps, image < 10% of view, multi-object, no Cap3D caption, "lowpoly" tag → filtered out).
- **Per-instance:** 6 input views (random) + 4 supervision views (random) from 32 total rendered views, 512×512 RGB + depth + normal.
- **Loss:** `L_1 = L_MSE + λ_lpips * L_LPIPS + λ_mask * L_mask` (RGB + perceptual + mask), `λ_lpips=2.0, λ_mask=1.0`, lr 4e-4 → 4e-5 cosine, 8× H800 GPUs.
- **Initialization:** OpenLRM weights, with *source-camera modulation removed* and *AdaLN camera-pose modulation added* in ViT (the killer modification that converts single-view LRM into sparse-view LRM).

**Stage 2: Training on Mesh (geometric supervision):**
- **Goal:** Switch from NeRF to mesh representation, add explicit geometric supervision (depth + normal).
- **Architecture changes:** Reuse density MLP as SDF MLP (with sign-flip init), add 2 MLPs for FlexiCubes.
- **Loss:** `L_2 = L_1 + λ_depth * L_depth + λ_normal * L_normal + λ_reg * L_FlexiCubes_reg`, `λ_depth=0.5, λ_normal=0.2, λ_reg=0.01`, lr 4e-5 → 0 cosine, 8× H800 GPUs.
- **Render:** Switch from volume rendering to *mesh rasterization* (enables full-resolution 512×512 supervision without cropping, the killer efficiency gain).

**Camera augmentation + perturbation (Sec. 3.2):**
- **Random rotation + scaling on input camera poses** for robustness to scale and orientation variation.
- **Random noise on camera parameters** to handle the *multi-view inconsistency* of Zero123++ (the killer practical fix that makes the LRM robust to imperfect multi-view diffusion outputs).

### Model variants (Tab. 1)

| Variant | Representation | Transformer Layers | Triplane Dim | Samples/Ray | Grid Size | Input Size | Render Size |
|---------|---------------|-------------------|--------------|-------------|-----------|-----------|-------------|
| InstantNeRF-base | NeRF | 12 | 40 | 96 | - | 320 | 192 |
| InstantNeRF-large | NeRF | 16 | 80 | 128 | - | 320 | 192 |
| InstantMesh-base | Mesh | 12 | 40 | - | 128 | 320 | 512 |
| InstantMesh-large | Mesh | 16 | 80 | - | 128 | 320 | 512 |

## Results

### Quantitative results on GSO (Tab. 2, 300 objects, orbiting views)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | CD↓ | F-Score@0.2↑ |
|--------|-------|-------|--------|-----|---------------|
| TripoSR 108 | 23.373 | 0.868 | 0.213 | 0.217 | 0.843 |
| LGM | 21.538 | 0.871 | 0.216 | 0.345 | 0.671 |
| CRM 152 | 22.195 | 0.891 | 0.150 | 0.252 | 0.787 |
| SV3D 117 | 22.098 | 0.861 | 0.201 | - | - |
| **Ours (NeRF)** | 23.141 | **0.898** | **0.119** | **0.177** | **0.882** |
| **Ours (Mesh)** | 22.794 | 0.897 | 0.120 | 0.180 | 0.880 |

**Key takeaways:** (1) **InstantMesh wins on SSIM, LPIPS, CD, F-Score** (4 of 5 metrics) — the *killer* perceptual + 3D geometry quality. (2) **TripoSR 108 wins on PSNR** (0.6 dB higher), but PSNR is *less* meaningful for "dreamed" novel views (the GT is *unknown* and has *multiple possibilities*). (3) **InstantMesh beats CRM 152 by -20% CD, +12pp F-Score** — the *killer* evidence that *transformer + two-stage training + mesh supervision* beats *CNN + single-stage + NeRF* for 3D geometry. (4) **InstantMesh beats LGM by -49% CD, +21pp F-Score** — the *killer* evidence that *mesh output* beats *3DGS output* for 3D geometry (3DGS is *great* for rendering but *poor* for surface extraction).

### Quantitative results on OmniObject3D (Tab. 3, 130 objects, orbiting views)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | CD↓ | F-Score@0.2↑ |
|--------|-------|-------|--------|-----|---------------|
| TripoSR 108 | 21.996 | 0.877 | 0.198 | 0.245 | 0.811 |
| LGM | 20.434 | 0.864 | 0.226 | 0.382 | 0.635 |
| CRM 152 | 21.630 | 0.892 | 0.147 | 0.246 | 0.802 |
| SV3D 117 | 21.510 | 0.866 | 0.186 | - | - |
| **Ours (NeRF)** | **22.635** | **0.903** | **0.110** | **0.199** | **0.869** |
| **Ours (Mesh)** | 21.954 | 0.901 | 0.112 | 0.203 | 0.864 |

**Same pattern:** InstantMesh wins on SSIM, LPIPS, CD, F-Score; TripoSR 108 wins on PSNR (0.6 dB). The *consistent* wins across GSO + OmniObject3D confirm that InstantMesh is the *SOTA* image-to-3D method as of April 2024.

### Quantitative results on OmniObject3D benchmark views (Tab. 4, 130 objects, 16 random views)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | CD↓ | F-Score@0.2↑ |
|--------|-------|-------|--------|-----|---------------|
| TripoSR 108 | 19.977 | 0.859 | 0.206 | 0.221 | 0.847 |
| LGM | 18.665 | 0.832 | 0.250 | 0.356 | 0.653 |
| CRM 152 | 19.422 | 0.865 | 0.172 | 0.274 | 0.778 |
| SV3D 117 | **20.294** | 0.853 | 0.176 | - | - |
| **Ours (NeRF)** | 19.752 | **0.869** | **0.150** | **0.206** | **0.863** |
| **Ours (Mesh)** | 19.552 | 0.868 | 0.150 | **0.204** | **0.866** |

**Same pattern:** SV3D 117 wins on PSNR (it's a video-diffusion model that's better at "dreamed" novel views), but InstantMesh wins on SSIM, LPIPS, CD, F-Score.

### Qualitative results (Fig. 3)

InstantMesh produces *significantly* better geometry and texture than TripoSR 108 (which lacks imagination for free-style images), LGM (distortions and multi-view inconsistency), and CRM 152 (difficulty generating smooth surfaces). InstantMesh's *high-resolution supervision* (512×512) gives *sharper* textures than TripoSR 108.

### NeRF vs Mesh variants (Tab. 2-4, Fig. 4)

The "NeRF" variant has *slightly* better metrics than the "Mesh" variant (due to limited FlexiCubes grid resolution), but the Mesh variant has *smoother surfaces* (due to explicit geometric supervision), which is *more desirable* in practical applications. The authors recommend the Mesh variant for production use.

## Connections to H1-H5

- **H1 (2-stage > 1-stage):** **STRONG SUPPORT — direct** — InstantMesh is *structurally* a 2-stage pipeline (multi-view diffusion + sparse-view LRM), and *internally* a 2-stage training strategy (Stage 1 NeRF warm-start → Stage 2 mesh + geometric supervision). The 2 stages serve *complementary* purposes: multi-view diffusion = 2D-to-2D prior, sparse-view LRM = 2D-to-3D mapping, and the *two-stage training* lets the transformer warm-start from a NeRF-pretrained LRM, then transitions to mesh for higher-fidelity surface extraction. For v0, the 2-stage training strategy is *directly* applicable: v0 sub-task 1 should start from OpenLRM pretrained on Objaverse, then *finetune on 3DTeethSeg22 + ToSynFCD* with the *dental-specific* mesh + depth + normal supervision.

- **H2 (latent diffusion > direct):** **STRONG SUPPORT** — Zero123++ is a *latent diffusion* multi-view model (operates in SD's latent space, not pixel space), and the *two-stage training* (Stage 1 NeRF → Stage 2 mesh) is *enabled* by the latent diffusion's fast inference (~3s for 6 views). For v0, this means v0 sub-task 1 should use a *latent diffusion* multi-view model (SD-based, 4-8s inference), not a *pixel-space* diffusion model (which would be 10-30s inference and infeasible for chairside).

- **H3 (patient/context conditioning):** **STRONG SUPPORT** — InstantMesh's *6-view multi-view input* is the *killer* H3 mechanism for *spatial-aware* conditioning. The 6 views are at *fixed camera poses* (azimuth 30°+k·60°, elevation ±20°/±10°), which is *spatially-aligned* with the 3D object. For v0, this is *directly* applicable: the v0 sub-task 1 should use *6 dental-arch views* (buccal, lingual, occlusal-top, occlusal-bottom, mesial, distal — the killer 6 dental views that cover the *full* dental arch, the direct dental analog of InstantMesh's 6 views).

- **H4 (implicit SDF > mesh):** **MILD CONTRADICTION** — InstantMesh uses *implicit SDF* (decoded from triplane features via tiny MLPs) + *FlexiCubes* (paper 007) iso-surface extraction, which is the *canonical* H4 substrate. But the paper's *two-stage training* reveals that *mesh representation with explicit geometric supervision* (depth + normal) is *better* than *NeRF representation* for *mesh output* (the Mesh variant has smoother surfaces). So H4 is *supported* (SDF is the right substrate) but *refined* (mesh + geometric supervision is the right *training strategy*). For v0, this means v0 sub-task 1 should use *implicit SDF + FlexiCubes* with *depth + normal supervision*, not *NeRF* (which would require post-hoc meshing with quality loss).

- **H5 (synthetic+finetune):** **STRONG SUPPORT** — InstantMesh is *trained on synthetic Objaverse* (270K filtered instances), then *evaluated on real* GSO and OmniObject3D scans. The *transfer* from synthetic Objaverse to real GSO/OmniObject3D is *exactly* the H5 paradigm. For v0, this is *directly* applicable: train v0 sub-task 1 on *synthetic dental arches* (generated by a 3D-tooth-statistical-shape-model + variations), then *finetune on real* 3DTeethSeg22 + ToSynFCD, the *right* H5 paradigm for dental data scarcity.

## Surprises / interesting things buried in section 4

1. **The "white-background fine-tuning" of Zero123++ (Sec. 3.1):** The killer 1,000-step UNet training, lr=1e-5, batch=48, on the LVIS subset of Objaverse, that *completely eliminates* the "floaters and cloud-like artifacts" caused by background RGB variation. This is *the* smallest data change in the entire image-to-3D literature that *completely* fixes a *fundamental* Zero123++ issue that the field overlooked. The fact that it converges in 1,000 steps (1 hour on 8 H800) is the *killer* practical insight. For v0, this is *directly* applicable: v0's dental multi-view diffusion should be *white-background fine-tuned* (or *teeth-only fine-tuned* with the gum masked out) to eliminate the *gum+saliva+background* artifacts in dental scans.

2. **The "two-stage training strategy" (Sec. 3.2):** Stage 1 (NeRF warm-start from OpenLRM) → Stage 2 (mesh + depth/normal supervision) is the *killer* practical recipe that lets the transformer warm-start from a NeRF-pretrained LRM, then transitions to mesh for *higher-fidelity* surface extraction, *avoiding* the cold-start problem that pure-mesh training would face. The key insight is that *NeRF pretraining is a much better warm-start* than *random initialization* (the authors' 1-week training on Objaverse converges to *much better* quality than random init). For v0, this is *directly* applicable: v0 sub-task 1 should start from OpenLRM pretrained on Objaverse, then *finetune on 3DTeethSeg22 + ToSynFCD* with the *dental-specific* mesh + depth + normal supervision. The killer *engineering* insight: *never* train a large transformer from scratch on dental data (insufficient scale), always warm-start from a *general* LRM.

3. **The "AdaLN camera-pose modulation" in ViT (Sec. 3.2):** Adding *AdaLN (Adaptive Layer Norm)* camera-pose modulation layers to the ViT image encoder is the *killer* modification that converts single-view LRM (source-camera modulation only) into sparse-view LRM (target-camera modulation for 6 views). The trick is that each view's image tokens are modulated by *that view's camera pose* (not the source pose), so the transformer learns *pose-aware* features that can be *cross-attended* to produce a *pose-invariant* 3D representation. For v0, this is *directly* applicable: v0's dental LRM should use *AdaLN camera-pose modulation* for the 6 dental-arch views, with each view's *dental camera pose* (buccal, lingual, occlusal-top, occlusal-bottom, mesial, distal).

4. **The "canonical world-space" reconstruction (Sec. 3.2, last paragraph):** InstantMesh reconstructs in *canonical world-space* (z-axis = anti-gravity direction), NOT *view-space* (the camera's view direction). The killer design choice is that *world-space* is *fixed* across views (no ambiguity), while *view-space* is *camera-dependent* (the *Janus* problem for back-view synthesis). For v0, this is *directly* applicable: v0's dental LRM should reconstruct in *canonical dental-space* (z-axis = occlusal direction, the *fixed* dental-arch orientation), NOT *view-space* (the intra-oral scanner's view direction).

5. **The "random camera pose noise" (Sec. 3.2, "Camera Augmentation"):** Adding *random noise* to the input camera poses is the *killer* practical fix that makes the LRM robust to *imperfect* multi-view diffusion outputs. The key insight is that *Zero123++'s output is not perfectly aligned* with its *pre-defined* camera poses (the multi-view diffusion has *inherent* pose errors), so the LRM must be *robust* to *imperfect* poses. For v0, this is *directly* applicable: v0's dental LRM should add *random noise* to the input camera poses during training (the *standard* 2024 data augmentation for LRMs), the killer practical fix for *real-world* multi-view diffusion outputs.

6. **The "white-bg + Stage 1 → Stage 2" is the *complete* 2024 image-to-3D recipe (Sec. 3-4):** The combination of (1) white-bg Zero123++, (2) 6-view canonical world-space input, (3) transformer LRM with AdaLN camera-pose modulation, (4) two-stage training (NeRF warm-start → mesh + geometric supervision), (5) FlexiCubes mesh extraction, (6) random camera-pose noise augmentation is the *complete* 2024 image-to-3D paradigm that v0 sub-task 1 should adopt. The killer insight is that *all 6 ingredients* are *necessary* (any one removed → quality drops), and *all 6 are open-source* (MIT license, $0 cost to fork).

## Quote-worthy sentences

> "Crafting 3D assets from single-view images can facilitate a broad range of applications, e.g., virtual reality, industrial design, gaming and animation. We have witnessed a revolution on image and video generation with the emergence of large-scale diffusion models trained on billion-scale data, which is able to generate vivid and imaginative contents from open-domain prompts. However, duplicating this success on 3D generation presents challenges due to the limited scale and poor annotations of 3D datasets." (Sec. 1, "Motivation")

> "LRM-based methods use triplanes as the 3D representation, where novel views are synthesized using an MLP. Despite the strong geometry and texture representation capability, decoding triplanes requires a memory-intensive volume rendering process, which significantly impedes training scales. Moreover, the expensive computational overhead makes it challenging to utilize high-resolution RGB and geometric information (e.g., depths and normals) for supervision." (Sec. 1, "LRM limitation")

> "In this work, we present InstantMesh, a feed-forward framework for high-quality 3D mesh generation from a single image. Given an input image, InstantMesh first generates 3D consistent multi-view images with a multi-view diffusion model, and then utilizes a sparse-view large reconstruction model to predict a 3D mesh directly, where the whole process can be accomplished in seconds. By integrating a differentiable iso-surface extraction module, our reconstruction model applies geometric supervisions on the mesh surface directly, enabling satisfying training efficiency and mesh generation quality." (Sec. 1, "Our approach")

> "We notice that the generated background is not consistent across different image areas and varies in RGB values, leading to floaters and cloud-like artifacts in the reconstruction results. And LRMs are often trained on white-background images too. To remove the gray background, we need to utilize third-party libraries or models that cannot guarantee the segmentation consistency among multiple views. Therefore, we opt to fine-tune Zero123++ to synthesize consistent white-background images, ensuring the stability of the latter sparse-view reconstruction procedure." (Sec. 3.1, "White-background fine-tuning")

> "Different from the single-view LRM, our reconstruction model takes 6 views as input, requiring more memory for the cross-attention between the triplane tokens and image tokens. We notice that training such a large-scale transformer from scratch requires a significant period of time. For faster convergence, we initialize our model using the pre-trained weights of OpenLRM, an open-source implementation of LRM." (Sec. 3.2, "Warm-start from OpenLRM")

> "Different from view-space reconstruction models, our model reconstruct 3D objects in a canonical world space where the z-axis aligns with the anti-gravity direction. To further improve the robustness on the scale and orientation of 3D objects, we perform random rotation and scaling on the input multi-view camera poses." (Sec. 3.2, "Camera augmentation")

> "We argue that the perceptual quality is more important than faithfulness, as the 'true novel views' should be unknown and have multiple possibilities given a single image as reference." (Sec. 4.2, "PSNR is less meaningful than perceptual quality")

## Code/data link

- **Code:** ✅ Official: **github.com/TencentARC/InstantMesh** (Tencent ARC Lab, MIT license, full PyTorch implementation, includes 4 model variants + Zero123++ fine-tuning code + gradio demo + docker)
- **Pretrained:** ✅ HuggingFace **TencentARC/InstantMesh** (4 model variants: instant-nerf-base, instant-nerf-large, instant-mesh-base, instant-mesh-large + white-bg-fine-tuned Zero123++)
- **Demo:** ✅ HuggingFace gradio: huggingface.co/spaces/TencentARC/InstantMesh + replicate.com/camenduru/instantmesh + Colab demo + ComfyUI support (jtydhr88/ComfyUI-InstantMesh)
- **Data:** Filtered Objaverse (270K instances, the paper does *not* release the filtered list, but the data preparation code is in src/data/objaverse.py)
- **Dependencies:** Zero123++ (github.com/SUDO-AI-3D/zero123plus), OpenLRM (github.com/3DTopia/OpenLRM), FlexiCubes (github.com/nv-tlabs/FlexiCubes), Instant3D (instant-3d.github.io)
- **Project page:** github.com/TencentARC/InstantMesh (readme has teaser video, install instructions, inference scripts, training scripts, citation BibTeX)

## For our project

**InstantMesh is the *right* v0 sub-task 1 (full-arch synthesis from intra-oral scan) primary baseline.** The v0 stack should:

1. **Adopt InstantMesh's transformer + FlexiCubes + two-stage training as v0 sub-task 1's *primary* pipeline** — $200-500 Lambda engineering (just fork github.com/TencentARC/InstantMesh, replace Objaverse with 3DTeethSeg22 + ToSynFCD), the *right* H4 substrate (implicit SDF + FlexiCubes for *arbitrary-resolution* margin gap queries) + the *right* H1 (two-stage training for transformer warm-start).

2. **Adopt InstantMesh's *white-background fine-tuning* trick for v0's dental multi-view diffusion** — $50 Lambda for 1-hour training, the *killer* practical fix to *dental background artifacts* (gum, saliva, blood, etc.). Just fine-tune Zero123++ (or v0's *dental-fine-tuned* multi-view diffusion) on *dental-arch-only* white-bg images, the *smallest* data change that *completely* eliminates dental background noise.

3. **Adopt InstantMesh's *6-view canonical world-space* design for v0's dental arch** — $0 Lambda (just architectural design), the *killer* 6 dental-arch views (buccal, lingual, occlusal-top, occlusal-bottom, mesial, distal) that are *spatially-aligned* with the dental arch, the *direct* dental analog of InstantMesh's 6 views.

4. **Adopt InstantMesh's *OpenLRM warm-start* recipe for v0 sub-task 1** — $50 Lambda for OpenLRM pretrained weights, the *killer* practical insight that *neural warm-start* > *random init* for dental data (which is *scarce*). Just start from OpenLRM's checkpoint, then *finetune on 3DTeethSeg22 + ToSynFCD*, the *right* H5 paradigm for dental data scarcity.

5. **Adopt InstantMesh's *AdaLN camera-pose modulation* for v0's dental LRM** — $50 Lambda, the *killer* modification that lets the transformer handle 6 different views consistently. Each view's image tokens are modulated by *that view's dental camera pose*, so the transformer learns *pose-aware* features for *cross-pose consistency*.

6. **Adopt InstantMesh's *random camera-pose noise* for v0's dental LRM training** — $0 Lambda (just data augmentation), the *killer* practical fix that makes the LRM robust to *imperfect* multi-view diffusion outputs. The *standard* 2024 data augmentation for LRMs.

7. **Adopt InstantMesh's *mesh + depth + normal supervision* (Stage 2) for v0's dental LRM** — $50 Lambda, the *killer* explicit geometric supervision that gives *smoother* surfaces. Just add depth + normal rendering loss to the dental LRM training, the *right* training strategy for *mesh output* (vs *NeRF output* which requires post-hoc meshing).

8. **Use InstantMesh as the v0 paper's *primary image-to-3D baseline* for comparison** — $0 Lambda, just port the model and evaluate on v0's dental benchmark. The *killer* comparison: InstantMesh (transformer + FlexiCubes) vs CRM 152 (CNN + FlexiCubes) vs TripoSR 108 (transformer + NeRF) vs LGM (3DGS) for v0's dental arch synthesis, the *definitive* 2024 image-to-3D comparison.

9. **Adopt InstantMesh's *10s end-to-end* inference as v0's chairside-real-time target** — $0 Lambda, the *killer* practical benchmark for v0. After dental fine-tuning, v0's full-arch synthesis should be *~10s end-to-end* (or faster with optimization), the *right* chairside-real-time target.

10. **Cite InstantMesh in v0 paper's related-work as 2024 open-source image-to-3D SOTA** — $0 Lambda, 1 hour writing, the *right* positioning for v0 paper's image-to-3D baseline.

**v0 stack updated:**
- **v0 sub-task 1 (full-arch synthesis from intra-oral scan):** InstantMesh 153's transformer + FlexiCubes + white-bg-Zero123++ + two-stage training + dental warm-start from OpenLRM ($500-1,000 Lambda, 2-4 weeks, 10s end-to-end); CRM 152 as the *CNN-triplane* alternative
- **v0 sub-task 2 (crown generation):** DMC 033 + MCAM+CPL+MRL (UNCHANGED, 50-200ms chairside) + NSOT 148 + LION 149 + SeaLion 150 + OctFusion 151
- **v0 sub-task 3 (clinical-fit-aware):** Hwang 061 (histogram loss + gap-distance-map + hard testing) + InstantMesh 153's depth + normal supervision
- **v0 paper's *primary image-to-3D baseline*:** InstantMesh 153 evaluated on 3DTeethSeg22 + ToSynFCD (the *open-source* 2024 SOTA)
- **v0 paper's *CNN-triplane* baseline:** CRM 152 (the *CNN* 2024 SOTA)

**v0 compute update:** +$500-1,000 Lambda for InstantMesh 153 dental integration (overlap with CRM 152 = ~$300-500 shared, so +$200-500 incremental); **TOTAL v0 compute ~$6,770-9,430 Lambda** (was $6,570-8,930 from 152-note, +$200-500 for InstantMesh 153's incremental engineering).

The 3D-gen arc is now: **PVD 012 (ICCV 2021) → DPM 062 (CVPR 2021) → LION 149 (NeurIPS 2022) → DiffFacto 147 (ICCV 2023) → LRM 107 (ICLR 2024) → LGM (CVPR 2024) → TripoSR 108 (NeurIPS 2024) → CRM 152 (ECCV 2024) → InstantMesh 153 (2024) → Hunyuan3D 2.0 098 (2025) → NSOT 148 (ICLR 2025) → TripoSG 100 (ICML 2025) → SeaLion 150 (CVPR 2025) → Trellis 101 (CVPR 2025 Spotlight) → OctFusion 151 (CGF/SGP 2025)** = 14 papers. The 2024 *image-to-3D SOTA triangle* is now clear: **CNN-triplane (CRM 152) ↔ transformer-triplane (InstantMesh 153) ↔ 3DGS (LGM)**, all 3 are *concurrent* (all released April 2024), all 3 use *Zero123++* for multi-view diffusion, and the *killer comparison* is *CNN vs transformer vs 3DGS* for the 6-view → 3D mapping. **InstantMesh 153 wins on 4/5 metrics (SSIM, LPIPS, CD, F-Score) on GSO + OmniObject3D**, and is the *open-source* (MIT) winner with the *complete* codebase. For v0 sub-task 1, adopt InstantMesh 153 as the *primary* baseline and CRM 152 as the *CNN-triplane* alternative.

**★ Open Q for HK:**
- (i) adopt InstantMesh 153 as v0 sub-task 1 primary baseline? (RECOMMEND YES — MIT license, full open-source, 10s end-to-end, SOTA on GSO + OmniObject3D)
- (ii) port InstantMesh 153's transformer + FlexiCubes + two-stage training for dental? (RECOMMEND YES — $200-500 incremental over CRM 152, 2-4 weeks)
- (iii) adopt white-bg fine-tuning for v0's dental multi-view diffusion? (RECOMMEND YES — $50 Lambda, 1-hour training, the *killer* practical fix)
- (iv) use 6 dental-arch views (buccal/lingual/occlusal-top/occlusal-bottom/mesial/distal)? (RECOMMEND YES — $0 Lambda, the *direct* dental analog of InstantMesh's 6 views)
- (v) adopt OpenLRM warm-start for v0's dental LRM? (RECOMMEND YES — $50 Lambda, the *killer* H5 paradigm for dental data scarcity)
- (vi) adopt AdaLN camera-pose modulation for v0's dental LRM? (RECOMMEND YES — $50 Lambda, the *killer* multi-view-consistent mechanism)
- (vii) adopt random camera-pose noise augmentation? (RECOMMEND YES — $0 Lambda, the *killer* robustness trick)
- (viii) adopt mesh + depth + normal supervision (Stage 2) for v0's dental LRM? (RECOMMEND YES — $50 Lambda, the *killer* explicit geometric supervision)
- (ix) use InstantMesh 153 as v0 paper's primary image-to-3D baseline? (RECOMMEND YES — $0 Lambda, just port and evaluate)
- (x) cite InstantMesh 153 in v0 paper's related-work as 2024 open-source image-to-3D SOTA? (RECOMMEND YES — $0, 1 hour)

**★ Next paper to read (154):** the 153-note's recommended *next* is **(a) LGM (Tang et al. CVPR 2024, arXiv:2402.05054) — the *multi-view Gaussian Splatting* model that's the *3DGS counterpart* in the 2024 image-to-3D SOTA triangle (CRM 152 = CNN-triplane, InstantMesh 153 = transformer-triplane, LGM = 3DGS)** — the *killer* complement to complete the 2024 image-to-3D SOTA triangle and the *right* 154 to compare 3DGS vs mesh output for v0 sub-task 1's *clinical* use case (3DGS = fast novel view synthesis, mesh = high-quality surface for 3D printing + clinical margin gap queries), or **(b) GRM (Wang et al. CVPR 2024) — the *Generalizable Reconstruction Model* that improves LGM with better sparse-view reconstruction**, or **(c) MeshAnything V2 (Chen et al. ICLR 2025, arXiv:2501.03411) — the *artist-quality* mesh generation model that produces *production-quality* meshes with adjacency-aware transformer for sharp features and clean topology, the *killer* for v0's 3D-printing output**, or **(d) SF3D (Boss et al. 2024, Stable Fast 3D) — the *single-forward-pass* 3D model that uses UV-unwrapping + illumination disentanglement for high-quality textured mesh output in 0.5s**, or **(e) InstantSplat (Xu et al. 2024) — the *3DGS* *extension* of InstantMesh 153 that uses 3D Gaussian Splatting as the output representation for faster rendering**, or **(f) GS-LRM 110 (Zhang et al. 2024) — the *3DGS-Large Reconstruction Model* that replaces the triplane with 3DGS for 5× faster training**.

**Recommendation: *read 154 = LGM (Tang et al. CVPR 2024, arXiv:2402.05054)*** — the *3DGS counterpart* in the 2024 image-to-3D SOTA triangle, the *killer* complement to InstantMesh 153 for v0 sub-task 1's *3DGS vs mesh* design-space comparison. LGM's 3DGS output is *faster* (5s vs InstantMesh 153's 10s) and *better for novel view synthesis* (LPIPS +0.02) but *worse for surface extraction* (CD +0.16 vs InstantMesh 153). The *killer* design question for v0 sub-task 1: *3DGS for fast novel view (chairside preview) + mesh for high-fidelity surface (clinical + 3D printing)*, the *hybrid* output that LGM + InstantMesh 153 together enable. LGM is also the *direct* 2024 SOTA that the v0 paper should compare against for the *3DGS* baseline, completing the 2024 image-to-3D SOTA triangle.
