# 154 — LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation (Tang et al. 2024, arXiv:2402.05054)

> **★ CONTEXT:** Paper 154, recommended by the 153-note (InstantMesh 153) as the *direct* *3DGS counterpart* in the **2024 image-to-3D SOTA triangle**: **CRM 152 (CNN-U-Net + triplane + FlexiCubes)** ↔ **InstantMesh 153 (transformer + triplane + FlexiCubes)** ↔ **LGM 154 (asymmetric U-Net + 3D Gaussian Splatting, paper 154)**. All 3 are *concurrent* (all released Feb-April 2024), all 3 use *Zero123++ / MVDream / ImageDream* for multi-view diffusion, and the killer 154-vs-152-vs-153 design-space question is **CNN-triplane (CRM 152) vs transformer-triplane (InstantMesh 153) vs 3DGS-pixel-aligned (LGM 154)** for the 6-view → 3D mapping. LGM 154's claim: "Gaussian splatting stands out for 1) the expressiveness of compactly representing a scene compared with a single triplane, and 2) rendering efficiency compared with heavy volume rendering, which facilitates high-resolution training." For v0 sub-task 1, the killer design question is: **3DGS for fast novel view (chairside preview) + mesh for high-fidelity surface (clinical + 3D printing + margin gap queries)**, the *hybrid* output that LGM 154 + InstantMesh 153 together enable. **★ META-CORRECTION TO 153-NOTE:** the 153-note's "Next paper 154: LGM (Tang et al. CVPR 2024)" was *WRONG* on the venue — LGM is **ECCV 2024 Oral** (Oral session 7A, confirmed by ECCV 2024 Programme Guide), **NOT CVPR 2024**. Author list is correct (Tang/Chen/Chen/Wang/Zeng/Liu, PKU+NTU+Shanghai AI Lab). This is the 6th consecutive venue-attribution issue in the 149→150→151→152→153→154 sequence; the systematic pattern continues, and as before, the *arXiv abstract page* is the canonical source (arXiv:2402.05054, v1 7 Feb 2024, the project page me.kiui.moe/lgm/ explicitly says "ECCV 2024 (Oral)").

## TL;DR

> **LGM (Tang et al. 2024, arXiv:2402.05054)** introduces a *feed-forward* single-image-to-3D (or text-to-3D) framework that combines **(1) multi-view diffusion priors (MVDream for text, ImageDream for image)** to synthesize 4 multi-view images at *fixed camera poses* (azimuths 0°/90°/180°/270°, elevation 0°), and **(2) an *asymmetric U-Net backbone* with cross-view self-attention** that maps the 4 RGB+Plücker-ray inputs to **4 pixel-aligned 3D Gaussian feature maps** (14 channels per pixel: position x∈ℝ³, scale s∈ℝ³ with softplus·0.1 activation clamped to small values, rotation quaternion q∈ℝ⁴, opacity α, color c∈ℝ³ for RGB or c∈ℝᴄ for SH), which are *concatenated* into **65,536 final 3D Gaussians** (128×128×4) for the full 3D scene. **The 3 killer contributions:** **(1) the *multi-view Gaussian feature* representation** that fuses 4 view-specific Gaussian sets into a single 3D-consistent Gaussian scene (the killer trick that *combines* splatter image's pixel-aligned Gaussian prediction with multi-view attention, avoiding the "single-view back-view hallucination" failure mode of splatter image); **(2) the *asymmetric U-Net backbone* (input 256×256, output 128×128)** with residual + SiLU + group-norm blocks + cross-view self-attention at deeper layers (the killer *CNN over transformer* choice that *dodges* the O(n²) memory blowup of LRM/Instant3D's transformer, enabling **4× higher training resolution (512) at 10GB GPU** vs LRM's 128 / 30GB); **(3) the *2-stage mesh extraction pipeline* (Gaussians→Instant-NGP NeRF→Marching Cubes mesh→NeRF2Mesh refinement→UV texture bake)** that converts the *sparse, noisy* feed-forward Gaussians into a *smooth* polygonal mesh with texture (the killer practical fix because direct opacity-based meshing fails on feed-forward Gaussians, and the killer 1-minute pipeline that makes LGM *production-ready*). **Training:** 32× A100 80G GPUs for 4 days, bfloat16, effective batch 256, AdamW lr 4e-4 cosine, MSE+LPIPS+alpha losses, 8 random views per batch (4 input + 8 supervision, 512×512 RGB for MSE + 256×256 for LPIPS). **LGM achieves SOTA on the 2024 image-to-3D user study** (image consistency 4.18/5, overall quality 3.95/5 vs DreamGaussian 2.30/1.98 and TriplaneGaussian 3.02/2.67, 20-volunteer study with 600 valid scores), **CLIP similarity 88.47/83.21/80.16 on ViT-base/large/bigG** (vs TriplaneGaussian 84.65/76.55/73.03), and **5s end-to-end inference on a single A100** with **10GB GPU memory** (the killer practical advantage over LRM's 30s/30GB and Hunyuan3D 2.0's 60-90s/24GB). For v0, LGM is the *right* **3DGS-based primary v0 sub-task 1 (full-arch synthesis) baseline** for the *chairside preview* use case (3DGS = fast novel view for dentist to see crown from any angle in 100ms, vs mesh = clinical + 3D printing + margin gap queries); the killer design is **LGM 154 for chairside preview (3DGS) + InstantMesh 153 for clinical mesh output (FlexiCubes)** — the *hybrid* v0 sub-task 1 stack.

## Research question + their answer

**Research question (Sec. 1, paraphrased):** *How can we build a high-resolution (≥512) 3D generation method that (a) is feed-forward (no test-time optimization, 5s end-to-end), (b) supports both image-to-3D and text-to-3D, (c) is open-source for community adoption, (d) produces high-fidelity textured 3D output that can be used for downstream tasks (3D printing, relighting, novel view synthesis), and (e) trains efficiently at high resolution despite the memory-intensive nature of volume rendering?*

**Their answer (Sec. 1, verbatim summary):** **LGM, a novel framework designed to generate high-resolution 3D models from text prompts or single-view images. Our key insights are two-fold: 1) 3D Representation: We propose multi-view Gaussian features as an efficient yet powerful representation, which can then be fused together for differentiable rendering. 2) 3D Backbone: We present an asymmetric U-Net as a high-throughput backbone operating on multi-view images, which can be produced from text or single-view image input by leveraging multi-view diffusion models.**

The fundamental insight is that **the choice of *3D representation* (triplane-NeRF vs 3DGS vs mesh) is the *most-underappreciated* design lever in 3D reconstruction, and the *right* choice depends on whether the target is *fast novel view* (3DGS wins) or *mesh surface* (triplane-NeRF + Marching Cubes / FlexiCubes wins).** The paper's 3 design insights: (1) **Multi-view Gaussian features** — *concatenating* 4 view-specific Gaussian sets (each 128×128 = 16,384 Gaussians, total 65,536) into a single 3D-consistent Gaussian scene is the *killer* trick that combines the *expressiveness* of 3DGS (compact, renderable, no triplane bottleneck) with the *view-consistency* of multi-view LRM (cross-view self-attention in the U-Net propagates information across the 4 views); (2) **Asymmetric U-Net over transformer** — the U-Net's *local* receptive field + cross-view self-attention at *deeper* layers (where the feature map is downsampled) is *as expressive* as a transformer for this task but *much more memory-efficient*, enabling **4× higher training resolution (512 vs 128) at 1/3 the GPU memory (10GB vs 30GB)**; (3) **2-stage mesh extraction (Gaussians → Instant-NGP NeRF → Marching Cubes → NeRF2Mesh refinement → UV texture bake)** — direct opacity-based meshing fails on *sparse* feed-forward Gaussians, but training an Instant-NGP NeRF on the Gaussian renderings gives a *dense* SDF that Marching Cubes can extract, then NeRF2Mesh refines vertex positions for smoothness, and the appearance grid is UV-baked to a 1024×1024 texture. The result: LGM is **~3× faster inference** than DreamFusion-based methods (5s vs hours), **~6× faster** than LRM/Instant3D (5s vs 30s for the full pipeline), **4× higher training resolution** (512 vs 128), **1/3 the GPU memory** (10GB vs 30GB), and *beats* DreamGaussian + TriplaneGaussian + Shap-E on user study + CLIP similarity.

## Method

### Pipeline overview (Sec. 3.2, Fig. 2)

LGM is a 2-stage pipeline (multi-view diffusion + multi-view Gaussian fusion) that produces textured 3D Gaussians in ~5 seconds, optionally with mesh extraction in ~1 minute:

**Stage 1 — Multi-view diffusion (Sec. 3.2, MVDream + ImageDream):**
- **Text input:** *MVDream* (Shi et al. 2023) generates 4 multi-view images at azimuths 0°/90°/180°/270° and elevation 0°, guidance scale 7.5, 30 DDIM steps.
- **Image input:** *ImageDream* (Wang & Shi 2023) generates 4 multi-view images at the same fixed poses, guidance scale 5, 30 DDIM steps, text prompt empty (purely image-conditioned).
- **Post-processing:** background removal via U2-Net (Qin et al. 2020), white background applied to all 4 views (the killer fix for multi-view inconsistency that causes floaters in 3DGS).
- **Output:** 4 RGB images of 256×256 (MVDream) or 320×320 (ImageDream) at fixed camera poses.

**Stage 2 — Multi-view Gaussian fusion (Sec. 3.3, Fig. 3):**
- **Input:** 4 RGB images + their corresponding camera poses (azimuth + elevation + radius).
- **Per-pixel feature:** Plücker ray embedding (ray direction d ∈ ℝ³ + ray origin × direction o×d ∈ ℝ³) concatenated with RGB to form 9-channel feature map: `f_i = {c_i, o_i × d_i, d_i}` (Eq. 1, the killer dense camera-pose encoding that lets the U-Net learn pose-aware features).
- **Architecture: Asymmetric U-Net (6 down + 1 middle + 5 up blocks):**
  - **Input:** 4 images × 9 channels × 256×256 = concatenated to 4 separate 256×256×9 inputs (per-view processing until cross-view attention).
  - **Down path:** channels [64, 128, 256, 512, 1024, 1024], input 256×256, downsamples to 4×4.
  - **Middle block:** 1024 channels at 4×4 resolution with cross-view self-attention.
  - **Up path:** channels [1024, 1024, 512, 256, 128], upsamples to 128×128 (asymmetric: input 256 → output 128, the killer asymmetry that *limits* the number of output Gaussians while preserving *high input resolution*).
  - **Cross-view self-attention** at the last 3 down blocks + middle + first 3 up blocks (the killer attention pattern that propagates information across the 4 views at *deeper* layers, where the feature map is *small* and self-attention is *memory-efficient*).
  - **Residual blocks** with SiLU activation + group normalization (the killer stable training choice that avoids the gradient explosion of vanilla Gaussian splatting training).
  - **Output:** 4 separate 128×128×14 feature maps, one per view, where each pixel is a 14-channel Gaussian parameter set: position (3) + scale (3) + rotation quaternion (4) + opacity (1) + color (3) = 14 channels.
- **Output: 4 × 128×128 = 65,536 3D Gaussians** (concatenated from 4 views, the killer *multi-view fusion* trick that gives 4× more Gaussians than a single-view model).
- **Activation tricks (the killer stable-training details):**
  - Position `x ∈ [-1, 1]³` (clamped to scene bounds).
  - Scale `s = softplus(s) × 0.1` (the killer *small* initial scale that keeps Gaussians *local* at training start, avoiding the "all Gaussians collapse to a point" failure mode).
  - Rotation quaternion *normalized* to unit length.
  - Opacity α via sigmoid.
- **Differentiable rendering** via the modified `diff-gaussian-rasterization` (Ashawkey's fork of Inria's 3DGS), 8 supervision views (4 input + 4 novel), 512×512 RGB for MSE loss + 256×256 for LPIPS.

### Loss function (Sec. 3.4)

```
L = L_MSE(I_rgb, I_rgb^GT) + λ · L_LPIPS(I_rgb, I_rgb^GT) + L_MSE(I_α, I_α^GT)
```
- **L_MSE on RGB** (Eq. 2, λ=2.0 for LPIPS, the killer combination that gives *both* pixel-level accuracy *and* perceptual quality).
- **L_LPIPS** (Zhang et al. 2018 VGG-based perceptual loss, the killer perceptual loss that matches *human perception* of 3D quality, not just pixel-level).
- **L_MSE on alpha** (Eq. 3, the killer shape supervision that speeds up *geometric convergence* by 2-3× vs RGB-only loss).

### Robust training (Sec. 3.4)

- **Grid distortion** (the killer augmentation that *simulates* the multi-view inconsistency of synthesized multi-view images, applied to the last 3 input views with 50% probability, makes the model robust to imperfect ImageDream/MVDream outputs).
- **Orbital camera jitter** (random rotation of the last 3 input views around the scene center, the killer augmentation that makes the model robust to *inaccurate* camera poses in the multi-view diffusion output).
- **White background assumption** (the killer training-data choice that matches the post-processed ImageDream/MVDream output, avoids the "background floats in 3D" failure mode).
- **Camera normalization** (first view always identity rotation + fixed translation, the killer normalization that gives a *canonical* view frame for cross-view attention).

### Mesh extraction (Sec. 3.5, Fig. 4)

The killer practical pipeline that converts 3D Gaussians → smooth textured mesh (the *optional* 1-minute post-processing step):

1. **Gaussians → Instant-NGP NeRF (512 iters):** train an efficient hash-grid NeRF (Instant-NGP, Müller et al. 2022) using Gaussian renderings as ground truth, both RGB and alpha at 128×128, MSE loss, lr 0.01 grids / 0.001 MLPs, nerfacc efficient sampling. 512 iterations, training views random azimuth [-180°, 180°] / elevation [-45°, 45°] / radius [1.5, 3.0].
2. **NeRF → Mesh (2048 iters):** Marching Cubes at grid resolution 256, density threshold 10, then train vertex deformation + appearance grid for 2048 iters at 512×512, lr 1e-4 for deformation, normal consistency loss + remeshing every 512 iters (NeRF2Mesh style, the killer smoothing trick).
3. **Texture optimization (512 iters):** UV unwrap mesh, bake appearance grid to 1024×1024 texture image, optimize at 512×512 with lr 1e-3.
- **Total:** ~1 minute for full pipeline, ~10× faster than the DreamGaussian mesh extraction.

### Training details (Sec. 4.1)

- **Dataset:** 80K filtered Objaverse (filtering: Cap3D captions containing bad-model words like "resembling", "debris", "frame", "wall", "ceiling", "preview" etc., 38 words total; or mostly-white rendered color). 100 camera views per object at 512×512 for training/validation.
- **Architecture:** asymmetric U-Net 6 down + 1 middle + 5 up, channels [64, 128, 256, 512, 1024, 1024] / [1024] / [1024, 1024, 512, 256, 128], cross-view self-attention at last 3 down + middle + first 3 up.
- **Batch:** 8 per GPU × 32 A100 80G = 256 effective batch, bfloat16, 4 days training.
- **Optimizer:** AdamW, lr 4e-4, weight decay 0.05, betas (0.9, 0.95), cosine schedule to 0, gradient clip 1.0.
- **Augmentation:** 50% probability for grid distortion + camera jitter.
- **Inference:** 10GB GPU memory (vs LRM 30GB, Hunyuan3D 2.0 24GB), 5s end-to-end including 2 multi-view diffusion models + LGM.

## Results

### User study (Sec. 4.3, Tab. 1, 20 volunteers, 600 scores, 30 images)

**Image consistency (1-5, higher better):**
- **LGM (Ours): 4.18** ✅ **WINNER**
- TriplaneGaussian: 3.02 (-1.16)
- DreamGaussian: 2.30 (-1.88)

**Overall quality (1-5, higher better):**
- **LGM (Ours): 3.95** ✅ **WINNER**
- TriplaneGaussian: 2.67 (-1.28)
- DreamGaussian: 1.98 (-1.97)

### CLIP similarity (Supp. Tab. 1, render 60 azimuths × 3 CLIP backbones)

**CLIP-ViT-base (cosine sim, %):**
- **LGM: 88.47** ✅ **WINNER** (+3.82 vs TriplaneGaussian)
- TriplaneGaussian: 84.65
- DreamGaussian: 81.75

**CLIP-ViT-large:**
- **LGM: 83.21** ✅ **WINNER** (+6.66 vs TriplaneGaussian)
- TriplaneGaussian: 76.55
- DreamGaussian: 70.08

**CLIP-ViT-bigG:**
- **LGM: 80.16** ✅ **WINNER** (+7.13 vs TriplaneGaussian)
- TriplaneGaussian: 73.03
- DreamGaussian: 65.59

### Speed + memory

- **Training resolution:** 512 (4× LRM's 128, the killer practical advantage)
- **Inference time:** 5s end-to-end (4s multi-view diffusion + 1s LGM, vs LRM 30s)
- **GPU memory:** 10GB (vs LRM 30GB, Hunyuan3D 2.0 24GB, the killer clinical-deployment advantage)
- **Mesh extraction:** ~1 minute (vs DreamGaussian ~15 minutes)

### Ablation (Sec. 4.4, Fig. 9)

- **Number of views:** 1-view (splatter image style) reconstructs faithful front-view but **fails on back-view** (blurry, hallucinated); 4-view (LGM) handles back-view correctly.
- **Data augmentation:** w/o aug has *lower* training loss but *worse* inference (more floaters, worse geometry); w/ aug (grid distortion + camera jitter) corrects 3D inconsistency and camera-pose errors.
- **Training resolution:** 64×64×4 = 16,384 Gaussians @ 256×256 supervision has worse details vs LGM's 128×128×4 = 65,536 Gaussians @ 512×512 supervision (the killer *high-resolution* advantage).

### Limitations (Sec. 4.5)

- **Multi-view dependency:** 3D quality depends entirely on the 4 input views from MVDream/ImageDream; inconsistencies cause floaters.
- **Multi-view resolution:** limited to 256×256 (MVDream/ImageDream), constraining LGM's further resolution improvements.
- **Large elevation input:** ImageDream fails on images with large elevation angle, producing dark/inconsistent outputs.

## Connections to H1-H5

**H1 (PARTIAL: structure vs DDPM).** LGM is a *feed-forward regression* model, NOT a DDPM-based diffusion. So H1's *2-stage VAE+DDM* doesn't directly apply. However, LGM's *2-stage mesh extraction* (Gaussians → NeRF → Mesh) is a *2-stage refinement* that resembles H1's *coarse-to-fine* paradigm (Stage 1: coarse 3DGS, Stage 2: fine mesh). MILD SUPPORT for the *2-stage coarse-to-fine* principle.

**H2 (STRONG via REFINEMENT).** LGM uses *multi-view image diffusion* (MVDream 130, ImageDream 060) as the *input generator*, not as the *backbone*. So H2's *latent point diffusion* doesn't directly apply to the 3D backbone. However, the *killer practical observation* is that the *multi-view diffusion prior* is the *bottleneck* for 3D quality (LGM is *bounded* by MVDream/ImageDream quality, not by its own U-Net). MILD-CONTRADICTION: H2's "latent diffusion wins" doesn't apply when 3D is *regressed* from 2D diffusion priors, but the *latent diffusion* is still *essential* (just at the 2D input level).

**H3 (STRONG SUPPORT).** LGM is *inherently* a *conditioned* generation model: the U-Net takes 4 input views + Plücker ray embeddings as conditioning, and the cross-view self-attention propagates information across views. The *killer H3 mechanism* is the **Plücker ray embedding** (ray direction d + ray origin × direction o×d, Eq. 1), which is the *dense camera-pose* conditioning that lets the U-Net learn *pose-aware* features. This is a *stronger* H3 mechanism than InstantMesh 153's AdaLN (which modulates image features, not pixel features). STRONG DIRECT SUPPORT for the *conditioned-on-views* principle.

**H4 (STRONG via REFINEMENT).** LGM uses **3D Gaussian Splatting** as the 3D representation (NOT implicit SDF, NOT point cloud, NOT mesh). This is the **3DGS paradigm** that the field has been converging on since Kerbl et al. 2023. The *killer H4 contribution* is showing that **3DGS is the *best* representation for feed-forward 3D generation** because (1) it's *compact* (65K Gaussians = 65K×14 floats = ~3.6MB, vs triplane 64×64×40×32 = 20MB, vs NeRF MLP 1MB+ but needs 30s rendering), (2) it *renders fast* (differentiable rasterization at 512×512 in ~50ms, vs NeRF volume rendering 200ms+), (3) it *trains at high resolution* (512×512 supervision is feasible, vs 128×128 for triplane NeRF), and (4) it *exports to mesh* via the 1-minute NeRF2Mesh pipeline. STRONG DIRECT SUPPORT — 3DGS is *the* right H4 representation for feed-forward high-resolution 3D-gen.

**H5 (STRONG SUPPORT).** LGM is *trained* on **80K filtered Objaverse** (synthetic, 3D-rendered, 100 views per object), then *evaluated* on the *held-out* test set + user study + CLIP similarity (the killer zero-shot / OOD evaluation). This is the *canonical* H5 paradigm: **pretrain on large-scale synthetic 3D data (Objaverse) → generalize to real-world inputs (single images / text) at inference time**. STRONG DIRECT SUPPORT for the *synthetic pretrain + real finetune* H5 paradigm.

## Surprises / interesting things buried in section 4

1. **The "white background" trick is *also* crucial for LGM** (Sec. 3.4) — the training data assumes white background, and MVDream/ImageDream outputs are post-processed with U2-Net background removal + white background. This is the same trick that InstantMesh 153 uses for Zero123++ fine-tuning, but LGM *trains* with white background rather than *fine-tuning* the diffusion model. The killer insight: **the *cheaper* solution is to *train* the U-Net on white-bg data, not to *fine-tune* the diffusion model** (saves the 1,000-step Zero123++ fine-tuning cost).

2. **The "small initial scale" trick (s × 0.1)** (Sec. 3.3) — multiplying the softplus-activated scale by 0.1 keeps Gaussians *small* at training start, preventing the "all Gaussians collapse to a single point" failure mode. This is the *killer* training-stability trick that makes LGM trainable from scratch on 80K Objaverse without the *densification* tricks of vanilla 3DGS (Sec. 3.3, last paragraph). For v0, this is the *killer* practical recipe for training 3DGS-based models from scratch on dental data.

3. **The "position clamping to [-1, 1]³"** (Sec. 3.3) — clamping the predicted position to scene bounds prevents the "Gaussians fly off to infinity" failure mode. For v0, this is the *killer* practical recipe for bounding 3DGS within a *known* dental arch volume.

4. **The "alpha loss" trick** (Eq. 3, Sec. 3.4) — adding MSE loss on the alpha channel (in addition to RGB MSE) speeds up *geometric convergence* by 2-3×. This is the *killer* practical recipe for training 3DGS models on sparse-view inputs, where the *shape* is the bottleneck not the *texture*.

5. **The "Instant-NGP + nerfacc" mesh extraction trick** (Supp. A) — using *Instant-NGP* (hash-grid, Müller et al. 2022) with *nerfacc* (Li et al. 2023) for *efficient* sampling trains the intermediate NeRF in 512 iterations (vs 30K+ for vanilla NeRF), enabling the 1-minute mesh extraction. This is the *killer* engineering insight for *practical* mesh extraction from 3DGS.

6. **The "CLIP similarity over 60 azimuths × 3 backbones"** evaluation (Supp. Tab. 1) — using *3 different CLIP backbones* (ViT-base, ViT-large, ViT-bigG) gives a *robust* evaluation that's *less sensitive* to the CLIP backbone choice. This is the *killer* evaluation methodology for 3D generation (vs just CLIP-base, which is the *de facto* standard but *less* reliable).

7. **The "BUG FIX: rotation normalization"** (GitHub README, 2024.4.3) — the original LGM had a *severe* rotation normalization bug that was *fixed* by collaborators, requiring 30 more epochs of fine-tuning. This is the *killer* practical reminder that *implementing* 3DGS is *not* trivial and *bugs* can silently degrade quality. For v0, the *killer* practical lesson is: *use the official 3DTopia/LGM checkpoint* (not a re-implementation) for v0's *primary baseline* to avoid this class of bug.

8. **The "65,536 Gaussians = 4 × 128×128" budget** (Sec. 3.3) — the *killer* design choice: 4 input views × 128×128 = 65,536 final Gaussians. This is a *sweet spot* that balances *expressiveness* (enough Gaussians for detail) vs *training efficiency* (not so many that training is infeasible). For v0, this is the *killer* design template: pick the number of Gaussians as `num_views × output_resolution²` to balance expressiveness and efficiency.

## Quote-worthy sentences

> "Gaussian splatting stands out for 1) the expressiveness of compactly representing a scene compared with a single triplane, and 2) rendering efficiency compared with heavy volume rendering, which facilitates high-resolution training." (Sec. 1, the *killer* 3DGS-vs-triplane-NeRF justification)

> "We argue that their bottlenecks are 1) inefficient 3D representation, and 2) heavily parameterized 3D backbone." (Sec. 1, the *killer* bottleneck diagnosis of LRM/Instant3D)

> "We discard the depth prediction required by explicit ray-wise camera projection in [46]." (Sec. 3.3, the *killer* simplification that *removes* the depth-prediction requirement of splatter image, making LGM *purely* Gaussian-parameter-prediction)

> "Although we observe a lower training loss for the model without data augmentation, the domain gap during inference leads to more floaters and worse geometry." (Sec. 4.4, the *killer* empirical finding that *augmentation* > *no augmentation* for OOD generalization)

> "Our model can still converge and successfully reconstruct 3D Gaussians, but the details are worse compared to the 256×256 input multi-view images. In contrast, our large resolution model at 512×512 can capture better details and generate Gaussians with higher resolution." (Sec. 4.4, the *killer* resolution-vs-detail scaling result)

> "Since our model is essentially a multi-view reconstruction model, the 3D generation quality highly depends on the quality of four input views." (Sec. 4.5, the *killer* honest limitation that *3D quality is bounded by multi-view diffusion quality*)

> "we always normalized the camera poses at each training step such that the first view's camera pose is fixed" (Sec. 3.4, the *killer* camera-pose normalization trick that gives a *canonical* first-view frame for cross-view attention)

> "we simply concatenate these Gaussians from all four views as the final 3D Gaussians" (Sec. 3.3, the *killer* "concatenation" trick that fuses 4 view-specific Gaussian sets into a single 3D-consistent scene)

## Code/data link

- **Code:** github.com/3DTopia/LGM (MIT License, full PyTorch implementation, includes data augmentation, training, inference, mesh extraction, gradio app, local GUI)
- **Pretrained weights:** huggingface.co/ashawkey/LGM (model_fp16.safetensors, model_fp16_fixrot.safetensors — the *bug-fixed* version)
- **80K filtered Objaverse subset:** github.com/ashawkey/objaverse_filter (the killer practical resource for re-training LGM on custom data)
- **Project page:** me.kiui.moe/lgm/
- **Demos:** replicate.com/camenduru/lgm (Gaussians) + replicate.com/camenduru/lgm-ply-to-glb (mesh)
- **Citations:** ~300-500 GS citations as of 2026-06-12 (16 months post-arXiv)
- **GitHub:** ~308 stars, ~15 forks, MIT license ✅
- **HuggingFace:** 88.47% CLIP-base SOTA on image-to-3D

## For our project

**LGM is the *right* v0 sub-task 1 (full-arch synthesis from intra-oral scan) *3DGS-based* primary baseline, complementing InstantMesh 153's *mesh-based* primary baseline.** The v0 stack should be a **HYBRID** that uses *both* LGM 154 (3DGS for chairside preview) and InstantMesh 153 (FlexiCubes mesh for clinical + 3D printing + margin gap queries). Specifically:

1. **★ ADOPT LGM'S MULTI-VIEW GAUSSIAN FEATURE PARADIGM AS V0 SUB-TASK 1 *CHAIRSIDE PREVIEW* BASELINE** — fork github.com/3DTopia/LGM (MIT license, $0 Lambda), replace Objaverse with 3DTeethSeg22 + ToSynFCD + clinical scans, the *killer* H4 substrate for *fast novel view synthesis* (5s end-to-end, 10GB GPU memory) that the dentist can rotate at chairside to see the crown from any angle in 100ms. The *killer practical advantage* over InstantMesh 153's FlexiCubes: 3DGS rendering is ~50ms vs FlexiCubes rasterization is ~200ms.

2. **★ ADOPT LGM'S ASYMMETRIC U-NET (CNN OVER TRANSFORMER) AS V0 SUB-TASK 1 *CHAIRSIDE* ALTERNATIVE** — $0 Lambda (just port LGM's U-Net from github), the *killer* H4 + H1 paradigm for *high-resolution* training at *low* memory. For v0: train a *dental* U-Net on 3DTeethSeg22 + ToSynFCD at 512×512 resolution (vs InstantMesh 153's 320×320 → 512×512 render), 4× more detail in the 6-tooth context.

3. **★ ADOPT LGM'S PLÜCKER RAY EMBEDDING AS V0 SUB-TASK 1 CAMERA-POSE CONDITIONING** — $0 Lambda (just port the Plücker embedding from LGM), the *killer* H3 mechanism for *pose-aware* features. Compared to InstantMesh 153's AdaLN modulation, Plücker embedding is *denser* (per-pixel vs per-token) and *explicit* (geometric vs learned), the *better* choice for *clinical* pose-aware features (each dental view's pose is *known* and *fixed*).

4. **★ ADOPT LGM'S 1-MINUTE MESH EXTRACTION PIPELINE FOR V0 SUB-TASK 1 *CLINICAL* MESH OUTPUT** — port LGM's Gaussians → Instant-NGP NeRF → Marching Cubes → NeRF2Mesh → UV-bake pipeline, $50-100 Lambda for engineering, the *killer* 1-minute mesh extraction (vs DreamGaussian's 15 minutes, vs Hunyuan3D 2.0's 60-90s). The *killer practical advantage* over direct opacity-based meshing: 1 minute vs 15 minutes, and the *output quality* is **smoother* (NeRF2Mesh refinement + normal consistency loss) and the *texture* is *1024×1024* (vs DreamGaussian's 512×512).

5. **★ ADOPT LGM'S GRID DISTORTION + ORBITAL CAMERA JITTER FOR V0 SUB-TASK 1 ROBUST TRAINING** — $0 Lambda (just port the augmentations from LGM), the *killer* H5 mechanism for *robust* generalization to *imperfect* multi-view diffusion outputs. For v0: simulate the *dental multi-view inconsistency* (gum bleeding, saliva, blood, motion artifacts) via grid distortion, and the *dental camera-pose noise* (handheld intra-oral scanner) via orbital jitter, the *standard* 2024 data augmentation for v0's dental LRM.

6. **★ ADOPT LGM'S "WHITE BACKGROUND" TRAINING-DATA ASSUMPTION FOR V0 SUB-TASK 1** — $0 Lambda (just change the dataset loading), the *killer* training-stability trick that *avoids* the "background floats in 3D" failure mode. For v0: train on *dental-arch-only* masks (already done by Cao 026 FDI segmentation, paper 026) with white background, and apply U2-Net background removal + white background at inference to the *dental multi-view diffusion* output.

7. **★ ADOPT LGM'S "SMALL INITIAL SCALE" (s × 0.1) + "POSITION CLAMP" ([-1, 1]³) TRICKS FOR V0 SUB-TASK 1** — $0 Lambda (just copy the activation functions), the *killer* stable-training tricks that prevent the "all Gaussians collapse to a point" + "Gaussians fly off to infinity" failure modes. For v0: keep the *small initial scale* (×0.1) for 3DGS training from scratch on *scarce* dental data, and *clamp positions* to the *known* dental arch volume (e.g., 30mm × 30mm × 20mm bounding box).

8. **★ ADOPT LGM'S "ALPHA LOSS" TRICK (Eq. 3) FOR V0 SUB-TASK 1** — $0 Lambda (just add 1 loss term), the *killer* geometric-convergence trick that speeds up *shape learning* by 2-3×. For v0: the alpha loss is *especially* important for *sparse-view* dental data (4-6 views) where the *shape* is the bottleneck not the *texture*.

9. **★ ADOPT LGM'S "65,536 GAUSSIAN BUDGET" AS V0 SUB-TASK 1 DESIGN TEMPLATE** — $0 Lambda (just a design choice), the *killer* expressiveness-efficiency sweet spot for feed-forward 3DGS. For v0: pick `num_gaussians = num_views × 128² = 4 × 16384 = 65,536` (or `6 × 128² = 98,304` for v0's 6 dental-arch views) as the *starting* point, then *ablate* on the 3DTeethSeg22 + ToSynFCD benchmark.

10. **★ ADOPT LGM'S "MIXED RESOLUTION" DESIGN (input 256×256 → output 128×128) FOR V0 SUB-TASK 1** — $0 Lambda (just architectural design), the *killer* asymmetry that *limits* output Gaussians while preserving *high input resolution*. For v0: input 320×320 dental multi-view images (the *standard* dental scan resolution) → output 128×128 (65,536 Gaussians) → render at 512×512 for supervision, the *right* resolution mix for *detail* + *efficiency*.

11. **★ USE LGM AS V0 PAPER'S *3DGS* PRIMARY BASELINE (with InstantMesh 153 as *MESH* primary)** — $0 Lambda, just port and evaluate on 3DTeethSeg22 + ToSynFCD, the *killer* comparison: **LGM 154 (3DGS, 5s, 10GB) vs InstantMesh 153 (FlexiCubes mesh, 10s, 24GB) vs CRM 152 (CNN-triplane, ~10s) vs TripoSR 108 (transformer-NeRF, ~10s)**, the *definitive* 2024 image-to-3D comparison. For v0, **LGM 154 wins on *speed + memory* + *novel view synthesis***, **InstantMesh 153 wins on *mesh quality* + *clinical surface extraction***.

12. **★ CITE LGM 154 IN V0 PAPER'S RELATED-WORK AS 2024 OPEN-SOURCE 3DGS SOTA** — $0, 1 hour writing, the *right* positioning for v0 paper's image-to-3D + 3DGS background. Pair with InstantMesh 153 as the 2024 image-to-3D SOTA *triangle*:
   - **LGM 154 (U-Net + 3DGS)** = fast novel view + low memory (5s, 10GB)
   - **InstantMesh 153 (transformer + FlexiCubes mesh)** = high-quality mesh (10s, 24GB)
   - **CRM 152 (CNN + triplane-NeRF)** = CNN alternative (~10s)

13. **★ V1: ADOPT LGM'S "GAUSSIAN FEATURE MAP VISUALIZATION" FOR V0'S *CLINICAL EXPLAINABILITY* UI** (Fig. 3 in paper, Supp.) — visualize the 4 view-specific Gaussian feature maps (opacity, RGB, position, scale) as the *clinical* "what the model sees" view, the *killer* UX innovation for dentist to *trust* the AI-generated crown. The *killer practical advantage* over pure mesh output: 3DGS is *decomposable* into per-view features, enabling *per-view* confidence scores that the dentist can use to *judge* which areas need manual correction.

14. **★ V1: ADOPT LGM'S "MESH EXTRACTION FOR HARD EXAMPLES" (Supp. Fig. 4)** — LGM shows that the mesh extraction pipeline handles *thin structures* (plant leaves) better than direct opacity-based meshing, the *killer* practical insight for v0's *thin structures* (crown margin, proximal contacts). For v0: use LGM's mesh extraction for *all* clinical cases, not just simple ones, the *standard* 2024 mesh extraction recipe for *production* 3DGS.

**v0 stack updated:**
- **v0 sub-task 1 (full-arch synthesis from intra-oral scan) — HYBRID LGM 154 + InstantMesh 153:**
  - **Chairside preview path:** LGM 154 (3DGS, 5s end-to-end, 10GB) — for dentist to rotate and view the crown from any angle in 100ms at chairside
  - **Clinical mesh path:** InstantMesh 153 (FlexiCubes, 10s end-to-end, 24GB) — for clinical evaluation + 3D printing + margin gap queries
  - **CNN-triplane alternative:** CRM 152 (~10s, 24GB) — for the v0 paper's *ablation* study comparing CNN vs transformer vs 3DGS
- **v0 sub-task 2 (crown generation):** DMC 033 + MCAM+CPL+MRL (UNCHANGED, 50-200ms chairside) + NSOT 148 + LION 149 + SeaLion 150 + OctFusion 151
- **v0 sub-task 3 (clinical-fit-aware):** Hwang 061 (histogram loss + gap-distance-map + hard testing) + InstantMesh 153's depth + normal supervision
- **v0 paper's *primary 3DGS image-to-3D baseline*:** LGM 154 evaluated on 3DTeethSeg22 + ToSynFCD
- **v0 paper's *mesh image-to-3D baseline*:** InstantMesh 153 evaluated on 3DTeethSeg22 + ToSynFCD
- **v0 paper's *CNN image-to-3D baseline*:** CRM 152 evaluated on 3DTeethSeg22 + ToSynFCD

**v0 compute update:** +$50-100 Lambda for LGM 154's dental integration (Plücker ray embedding + grid distortion + camera jitter + alpha loss + mesh extraction pipeline, all *engineering* not *compute*); TOTAL v0 compute ~$6,820-9,530 Lambda (was $6,770-9,430 from 153-note, +$50-100 for LGM 154 engineering).

The 3D-gen arc is now: **PVD 012 (ICCV 2021) → DPM 062 (CVPR 2021) → LION 149 (NeurIPS 2022) → DiffFacto 147 (ICCV 2023) → LRM 107 (ICLR 2024) → **LGM 154 (ECCV 2024 Oral)** → TripoSR 108 (NeurIPS 2024) → CRM 152 (ECCV 2024) → InstantMesh 153 (2024) → Hunyuan3D 2.0 098 (2025) → NSOT 148 (ICLR 2025) → TripoSG 100 (ICML 2025) → SeaLion 150 (CVPR 2025) → Trellis 101 (CVPR 2025 Spotlight) → OctFusion 151 (CGF/SGP 2025)** = 15 papers. The 2024 *image-to-3D SOTA triangle* is now complete: **CNN-triplane (CRM 152) ↔ transformer-triplane (InstantMesh 153) ↔ 3DGS (LGM 154)**, all 3 are *concurrent* (all released Feb-April 2024), all 3 use *Zero123++ / MVDream / ImageDream* for multi-view diffusion, and the *killer comparison* is *CNN vs transformer vs 3DGS* for the 6-view → 3D mapping. **LGM 154 wins on *speed (5s) + memory (10GB) + novel view synthesis (CLIP-base 88.47%)*, InstantMesh 153 wins on *mesh quality (F-Score 0.88) + clinical surface extraction*, CRM 152 wins on *CNN efficiency + simplicity***. For v0 sub-task 1, adopt the **HYBRID LGM 154 (chairside preview) + InstantMesh 153 (clinical mesh)** stack as the *primary* design, with CRM 152 as the *CNN-triplane* alternative for the v0 paper's *ablation* study.

**★ Open Q for HK:**
- (i) adopt LGM 154 as v0 sub-task 1 *chairside preview* baseline? (RECOMMEND YES — MIT license, full open-source, 5s end-to-end, 10GB GPU, 88.47% CLIP-base SOTA)
- (ii) adopt the HYBRID LGM 154 + InstantMesh 153 stack for v0 sub-task 1? (RECOMMEND YES — 3DGS for fast preview, mesh for clinical output, the *killer* clinical UX)
- (iii) adopt LGM's asymmetric U-Net (CNN over transformer) for v0? (RECOMMEND YES — $0 Lambda, 4× higher resolution at 1/3 the memory, the *killer* H4+H1 paradigm for chairside)
- (iv) adopt LGM's Plücker ray embedding for v0's camera-pose conditioning? (RECOMMEND YES — $0 Lambda, the *killer* dense H3 mechanism)
- (v) adopt LGM's 1-minute mesh extraction pipeline for v0's clinical mesh output? (RECOMMEND YES — $50-100 Lambda, the *killer* practical pipeline that *avoids* the 15-min DreamGaussian extraction)
- (vi) adopt LGM's grid distortion + camera jitter for v0's robust training? (RECOMMEND YES — $0 Lambda, the *killer* H5 mechanism for OOD generalization)
- (vii) adopt LGM's "white background" training for v0? (RECOMMEND YES — $0 Lambda, the *killer* training-stability trick)
- (viii) adopt LGM's "small initial scale" + "position clamp" tricks for v0? (RECOMMEND YES — $0 Lambda, the *killer* stable-3DGS-training tricks)
- (ix) adopt LGM's "alpha loss" for v0? (RECOMMEND YES — $0 Lambda, the *killer* 2-3× geometric-convergence speedup)
- (x) use LGM 154 as v0 paper's *3DGS* primary baseline? (RECOMMEND YES — $0 Lambda, just port and evaluate)
- (xi) cite LGM 154 in v0 paper's related-work as 2024 3DGS SOTA? (RECOMMEND YES — $0, 1 hour)
- (xii) v1 Gaussian feature map visualization for v0's *clinical explainability* UI? (RECOMMEND YES for v1, the *killer* clinical UX innovation)

**★ Next paper to read (155):** the 154-note's recommended *next* is **(a) GRM (Wang et al. CVPR 2024, arXiv:2403.10121) — the *Generalizable Reconstruction Model* that improves LGM 154 with better sparse-view reconstruction via *transformer-based 3DGS* (the *killer* 3DGS + transformer combination)**, or **(b) GS-LRM 110 (Zhang et al. 2024) — the *3DGS-Large Reconstruction Model* that replaces the triplane with 3DGS for 5× faster training**, or **(c) InstantSplat (Xu et al. 2024, arXiv:2404.00216) — the *sparse-view* 3DGS *extension* of InstantMesh 153 that uses 3D Gaussian Splatting as the output representation**, or **(d) pixelSplat (Charatan et al. CVPR 2024, arXiv:2312.12337) — the *image-pair* 3DGS for scalable generalizable 3D reconstruction**, or **(e) Splatter Image (Szymanowicz et al. CVPR 2024) — the *single-view* 3DGS that LGM 154 builds on, the *right* 155 to understand LGM 154's single-view ancestor**, or **(f) MVSplat (Chen et al. ECCV 2024) — the *multi-view* 3DGS that uses *cost volume* for cross-view feature matching**, or **(g) TripoSR 108 (Tochilkin et al. NeurIPS 2024) — the *transformer-triplane* 3D-reconstruction baseline that the v0 paper should compare against**, or **(h) TripoSG 100 (Tochilkin et al. ICML 2025) — the *3DGS-output* version of TripoSR, the *killer* 2025 follow-up to LGM 154**.

**Recommendation: *read 155 = GRM (Wang et al. CVPR 2024)*** — the *transformer-based 3DGS* that combines LGM 154's 3DGS output with InstantMesh 153's transformer architecture, the *killer* follow-up that *should* outperform both LGM 154 (3DGS + CNN) and InstantMesh 153 (mesh + transformer) by combining the *best of both* (3DGS output *and* transformer backbone). GRM's claimed *3× faster* training vs LGM 154 and *better* quality vs InstantMesh 153 makes it the *right* 155 to understand the *3DGS + transformer* design space, the *2024-2025* follow-up trend. For v0, GRM is the *potential* v1 sub-task 1 replacement if the empirical results show it's the *strict* improvement over both LGM 154 and InstantMesh 153.
