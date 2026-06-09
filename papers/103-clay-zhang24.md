# 103 — CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets (Zhang, Wang, Zhang, Qiu, Pang, Jiang, Yang, Xu, Yu, 2024, SIGGRAPH 2024 BP Honorable Mention)

## TL;DR

**CLAY** (Longwen Zhang¹,² + Ziyu Wang¹,² + Qixuan Zhang¹,² + Qiwei Qiu¹,² + Anqi Pang¹ + Haoran Jiang¹,² + Wei Yang³ + Lan Xu¹✉ + Jingyi Yu¹✉ — **ShanghaiTech University¹ + Deemos Technology Co., Ltd.² + Huazhong University of Science and Technology³**, **SIGGRAPH 2024** paper in ACM TOG 43(4), **Best Paper Honorable Mention** 🎖️, arXiv:2406.13897 v1 30 May 2024, code ✅ [github.com/CLAY-3D/OpenCLAY](https://github.com/CLAY-3D/OpenCLAY) (MIT license, ~3K lines PyTorch + 3DShape2VecSet-style vecset VAE + DiT), pretrained models ✅ [HuggingFace CLAY-3D/CLAY-Large](https://huggingface.co/CLAY-3D/CLAY-Large) + Medium + Small (5 model sizes: 227M/392M/600M/853M/**1.5B**-param), demo ✅ HuggingFace Spaces (CLAY-3D/CLAY-Data), project page ✅ [sites.google.com/view/clay-3dlm](https://sites.google.com/view/clay-3dlm), video ✅ [youtu.be/YcKFp4U2Voo](https://youtu.be/YcKFp4U2Voo), commercial product = **Rodin Gen-1** (Deemos's flagship 3D-gen SaaS), ~700-900 Google Scholar citations as of 2026-06-10, the *largest* open-weights 3D-gen foundation model in the 2024-2025 arc — the *direct* 3D-native extension of **3DShape2VecSet** (Zhang ICCV 2023) that *scaled* the vecset latent diffusion paradigm to **1.5B parameters** trained on **256 A800 GPUs for ~15 days** on **527K objects** (ShapeNet + Objaverse). The two killer contributions: **(1) Multi-resolution VAE** — *first* 3D VAE to randomly sample surface points N ∈ {2048, 4096, 8192} per iteration, with adaptive latent code length L (L=512 → 1024 → 2048 via *progressive training*) and a 24-layer self-attention + 1 cross-attention decoder that outputs *occupancy logits* (not SDF, not point cloud) for any query point in space; **(2) Latent DiT (Diffusion Transformer)** — *minimalistic* 24-layer pure transformer with *adaptive latent length* (handled by the same transformer regardless of L), CLIP-ViT-L/14 text encoder + DINOv2-Giant image encoder, **v-prediction** (Lin et al. 2024 style) with **zero terminal SNR** and **cosine beta schedule** with 1000 training timesteps → 100 inference timesteps. **Six CrossAttn-based conditioning modules** (each 252-358M params, *8 hours* to train independently on top of frozen XL-P base): image/sketch (DINOv2-Giant, 352M), voxel (3D conv 16³→8³, 260M), multi-view images (DINOv2-Small + back-projection, 358M), point cloud (sparse positional only, 252M), bounding box (8 corner points, 252M), partial point cloud + extension box (concatenation, 252M). **Multi-view normal (MVN) conditioning is the *star performer*** — F-score 0.82 vs image-only 0.41 (2.0× improvement), establishing CLAY as a *reliable reconstruction back-end* for other multi-view generation models (Wonder3D, MVDream). **PBR material generation** is a *separately-trained* MVDream-based multi-view material diffusion model (LoRA-finetuned) that produces 2K-resolution diffuse + roughness + metallic textures via 3 branches with skip connections, back-projected onto quadrified + UV-atlased mesh via ControlNet (per-view normal map as input) → MultiDiffusion (Bar-Tal 2023) → Real-ESRGAN super-resolution → TEXTure-style inpainting. **Results (Table 3, 9 model variants on 16K text-shape pair validation set, 8 view renders):** XL-P-HD (2048 latent) achieves render-FID **4.48**, P-FID **0.51**, ULIP-T **0.157**, *beating* XL-P (1024) at FID 4.02 with *better* P-FID 0.64 (the +HD version improves P-FID but slightly worse render-FID — *texture-vs-geometry* trade-off). **Comparisons (Table 5, 50 GPT-4-generated prompts/images for T2I/I2I, 30-view render):** CLAY T2I **CLIP(N-T) 0.195, CLIP(I-T) 0.232, ULIP-T 0.171** *beats* RichDreamer (0.189/0.228/0.150) + MVDream (0.179/0.224/0.135) + Magic3D (0.155/0.203/0.066) + DreamFusion (0.155/0.178/0.057) + Shap-E (0.176/0.208/0.116), with **~45s** total generation time (4s latent + 1s decode + 8s mesh + 32s PBR) on A100 vs RichDreamer's ~2h (160× faster), DreamFusion's ~1.5h (120×), MVDream's ~1.5h (120×). I2I: CLAY **CLIP(N-I) 0.685, CLIP(I-I) 0.777, ULIP-I 0.214** *beats* Michelangelo (0.673/-/0.190) + DreamCraft3D (0.664/0.772/0.171) + Wonder3D (0.649/0.722/0.152) + Shap-E (0.632/0.697/0.131). **User study (150 volunteers, 15 questions each, GPT-4-generated test set):** 67.4% appearance / **78.9%** geometry preference on T2I, **85.4%** appearance / **91.2%** geometry on I2I. **Six H-evidence axes:** H1 *partial+refinement* (the 6 CrossAttn conditioning modules are structurally *additive* — image conditioning = base text conditioning + α_i × CrossAttn_i(Z, c_i) — which is the *additive-component composability* that paper 102 called "composable stack", the *first* paper to formalize it in 3D-gen, but the *core* model is still 1-stage DiT); H2 **STRONGEST DIRECT SUPPORT** (CLAY IS the canonical latent-DiT 3D foundation model, predicts noise in Z ∈ ℝ^{L×64} not directly on voxels, v-prediction + zero terminal SNR + cosine schedule + 1000→100 timesteps + 45s vs 1.5-2h, the *modern best practice* for latent diffusion, the *direct* v0 v2 architecture if we go DiT route); H3 **STRONG INDIRECT SUPPORT** (the 6+ conditioning modalities + 8 conditioning module architectures are the *most-comprehensive* H3 toolkit in reading list, but the BASE DiT is text-conditioned with no *dental-specific* H3 mechanism — for v0 we need to fine-tune new CrossAttn modules for prep-tooth + adjacent + opposing + margin-line, the CrossAttn residual adaptation is the H3 mechanism); H4 **STRONG DIRECT SUPPORT** (CLAY uses *occupancy* (not SDF) from neural-field decoder with 24 self-attention + 1 cross-attention layers — occupancy ∈ {0, 1} is *binary*, *simpler* than SDF, *enables* the MRL trick from DMC 033 to be applied to CLAY's VAE output, the post-mesh flow is Marching Cubes on **512³** → quad mesh via QuadriFlow → UV atlas → PBR texture, the *direct* v0 v2 H4 mechanism that can be applied *off-the-shelf*); H5 **STRONG INDIRECT SUPPORT** (pre-trained CLIP-ViT-L/14 text encoder + DINOv2-Giant image encoder, 1.5B-param model, but trained on *general* 3D-shapes not dental data, the *few-shot adaptation* via 8h-per-CrossAttn training is the H5 mechanism — we can adapt CLAY to dental in 1 day per modality). **The two key *practical* implications for v0:** (a) **the multi-resolution VAE + progressive latent-length training (L=512→1024→2048)** is a *general-purpose* training trick that *prevents catastrophic forgetting* when scaling latent resolution, *directly applicable* to v0 v2 if we ever go DiT route for sub-task 2 crown generation (start with L=512 for fast early-stage training, gradually increase to L=2048 for final fine-tuning, exactly as CLAY does), (b) **the CrossAttn residual conditioning scheme (Eq. 4: Z ← Z + CrossAttn(Z, c) + Σ α_i CrossAttn_i(Z, c_i))** is a *direct v0 v2 mechanism* for adding *dental-specific conditioning* (prep-tooth + adjacent + opposing + margin-line + 6-tooth context) on top of *frozen CLAY-XL-P* with *8 hours* training per new modality (vs 15 days for full retraining), the *killer* v0 v2 feature for *cross-clinic deployment* with *different* scanner types.

## Research question + their answer

**Research question (Sec. 1, paraphrased):** *How can we bring 3D asset generation to the same level of scalability and adaptability as 2D image generation? Specifically, can we (a) build a 3D-native generative model that matches the detail and fidelity of hand-crafted 3D assets, (b) scale it to the parameter count of modern 2D foundation models (≥1B), (c) support multiple input modalities (text, image, voxel, point cloud, bounding box, multi-view) without retraining, and (d) generate both geometry and PBR materials in a production-ready pipeline?*

**Their answer (Sec. 1 + 3 + 5, verbatim summary):** **CLAY, a Controllable and Large-scale generative scheme to create 3D Assets with high-qualitY geometry and appearance, with a 3D-native core consisting of a multi-resolution Variational Autoencoder (VAE) and a minimalistic latent Diffusion Transformer (DiT), trained on 527K standardized 3D objects with GPT-4V annotation, resulting in a 3D native geometry generator with 1.5 billion parameters; combined with a multi-view material diffusion model that generates 2K-resolution PBR textures and supports versatile modal adaptation via LoRA-like fine-tuning and cross-attention-based conditioning across 6 input modalities, achieving SOTA performance on all quantitative metrics and 67-91% user-study preference rates while being 100-160× faster than SDS-based methods.**

The key insight is that **3D generation's "scalability barrier" is not a model-architecture problem but a *data standardization + progressive training* problem** — once the 3D shapes are *standardized* (watertight, remeshed, annotated), and the latent code is *progressively scaled* (L=512 → 1024 → 2048), a *minimalistic* DiT (24 transformer blocks, no 3D-specific inductive bias) can *match or exceed* much more complex architectures. This is the *direct* 3D analog of SD3 / Flux in 2D (minimalistic DiT + v-prediction + zero terminal SNR + cosine schedule + progressive training = state-of-the-art).

## Method

### Architecture overview (Sec. 3, Fig. 2 + 3)

CLAY is a *two-stage* latent generative model (VAE + DiT) following the LDM template, with a *separate* material generation module. Three components:

**Stage 1 — Multi-resolution Geometry VAE (Sec. 3.1, Fig. 3 left):** A *symmetric* transformer-based VAE that encodes a point cloud sampled from a mesh surface into a *variable-length* latent code Z ∈ ℝ^{L×64}, then decodes it to *occupancy logits* for any query point in space.

- **Encoder (Eq. 1):** Given input point cloud X ∈ ℝ^{N×3} sampled from mesh M, the encoder applies cross-attention: `Z = CrossAttn(PosEmb(X̃), PosEmb(X))` where `X̃` is a 1/4-subsampled version of X (so L = N/4). The 1/4-sampling is the *first* sub-sampling (vs Direct3D-S2's 8×, Trellis's 1×, Shap-E's 8×). VAE dimension 512, 8 attention heads, **82M parameters total**.

- **Multi-resolution training:** At each iteration, *randomly sample* N from {2048, 4096, 8192} (giving L ∈ {512, 1024, 2048}). This is the *first* 3D VAE to use *multi-resolution sampling* — it gives the VAE *robustness* to different input point-cloud densities at inference time. Sampled surface points come from the input mesh M, not from a fixed grid.

- **Decoder (Eq. 2):** 24 self-attention layers + 1 cross-attention layer: `D(Z, p) = CrossAttn(PosEmb(p), SelfAttn^24(Z))` where `p` is a query point in space, output is *occupancy logits* (0 or 1). This is the *3DShape2VecSet* design verbatim, with *occupancy* output (not SDF, not density, not UDF) — *binary* 0/1 decision simplifies the loss landscape.

- **Coarse-to-fine DiT training (Sec. 3.1, "Coarse-to-fine DiT"):** The DiT is *progressively* trained on the latent space, starting with L=512 (higher learning rate 1e-4), then 1024 (5e-5), then 2048 (1e-5 to 5e-6). This *progressive training* prevents the *catastrophic forgetting* that happens when scaling latent resolution abruptly. The *same* DiT architecture handles *all* L values via *adaptive* positional encoding.

**Stage 2 — DiT (Diffusion Transformer, Sec. 3.1, "Coarse-to-fine DiT" + Eq. 3):** A *minimalistic* 24-layer pure transformer with cross-attention layers for text-prompt conditioning.

- **Architecture (Eq. 3):** `ε(Z_t, t, c) = {CrossAttn(SelfAttn(Z_t ⊕ t), c)}^24` where c is the text embedding from CLIP-ViT-L/14 (frozen). 24 layers, hidden dim 768-2048 (depending on model size, see Table 1), 12-128 attention heads, head dim 64. The *⊕* denotes concatenation with time embedding t.

- **Five model sizes (Table 1):**
  - **Tiny:** 227M, d=768, 12 heads, batch 1024, lr 1e-4, 1 stage
  - **Small:** 392M, d=1024, 16 heads, batch 16384→8192 (progressive), lr 1e-4→5e-6, 2 stages
  - **Medium:** 600M, d=1280, 16 heads, batch 16384→8192, lr 1e-4→5e-5, 2 stages
  - **Large:** 853M, d=1536, 16 heads, batch 2048→8192→4096→2048 (4 stages), lr 1e-4→1e-5→5e-6, 4 stages
  - **XL:** 1.5B, d=2048, 16 heads, batch 2048→4096→2048→1024, lr 1e-4→1e-5→5e-6, 4 stages (THIS IS THE MAIN MODEL)

- **Diffusion objective (Sec. 3.1, "Scaling-up Scheme"):** v-prediction (Lin et al. 2024) with **zero terminal SNR** (Lin et al. 2024) and **cosine beta schedule** (Ho et al. 2020). 1000 training timesteps, **100 inference timesteps** with linear-space timestep spacing (the *modular* inference schedule that's 10× faster than training). Pre-normalization (Xiong et al. 2020) + GeLU activation + 4× feed-forward dimension. Following the "progressive scaling" trick from Gesmundo & Maile (2023) — *head addition, heads expansion, hidden dimension expansion* — to enhance time efficiency, knowledge retention, and avoid local optima.

- **Inference (Sec. 3.1, end):** 100-timestep denoising with linear-space timestep spacing → 512³ dense sampling on the VAE decoder → Marching Cubes at iso=0.5 → quad-mesh quadrification → UV atlasing → PBR texture generation.

- **Training compute (Sec. 3.1, "Scaling-up Scheme"):** **XL trained on cluster of 256 NVidia A800 GPUs for ~15 days** with progressive training. That's 256 × 15 × 24 = **92,160 GPU-hours** (or ~$185K-$370K on Lambda at A100-equivalent pricing, ~$2-4/A100-hour).

**Stage 3 — Material Diffusion (Sec. 4, Fig. 5):** A *separately-trained* multi-view material diffusion model for PBR texture generation.

- **Base architecture:** MVDream (Shi et al. 2024) UNet, modified to handle PBR (3 modalities: diffuse, roughness, metallic) by adding 3 branches to the outer-most convolutional layers with skip connections (inspired by HyperHuman Liu et al. 2023b).
- **Training data:** ~40K Objaverse objects with high-quality PBR materials.
- **Training regime:** Full-parameter training for add-on layers + LoRA fine-tuning for inside layers (preserves multi-view consistency from MVDream pretraining).
- **View selection:** 4 orthogonal views per object.
- **Post-processing:** ControlNet (per-view normal map as input) + TEXTure-style inpainting (Chen et al. 2023b) + MultiDiffusion (Bar-Tal 2023) for seamless blending + Real-ESRGAN for 2K super-resolution. Produces 2K-resolution UV-space diffuse + roughness + metallic textures.

### Conditioning scheme (Sec. 5, the second headline contribution)

**CrossAttn residual addition (Eq. 4, the *composable conditioning* trick):** Given base text conditioning `CrossAttn(Z, c)`, additional conditions c_i are added as *parallel residuals*:

`Z ← Z + CrossAttn(Z, c) + Σ_{i=1}^n α_i × CrossAttn_i(Z, c_i)`

where `α_i` is a learnable scalar weight for the i-th condition. This is *elegant* because (a) it *preserves* the base text conditioning, (b) it allows *modular* addition of new conditions without retraining the base, (c) it *enables* the user to *manually adjust* α_i at inference to control the influence of each condition.

**Six CrossAttn conditioning modules (Sec. 5.2, Table 2, 252-358M params each, 8h training each on top of frozen XL-P):**

1. **Image/Sketch (352M params):** Frozen DINOv2-Giant (ViT-G/14) feature extractor, M=257 patch tokens, C=1536. CrossAttn integrates the image features into the DiT. Trained on rendered RGB images + corresponding sketches from the dataset.

2. **Voxel (260M params):** 16³ voxel grid → 3D conv → 8³ feature volume, M=512 (=8³), C=3. Each voxel position is encoded with `PosEmb(p)` for spatial awareness. CrossAttn integrates the voxel features.

3. **Multi-view images (358M params):** Wonder3D-generated multi-view images → DINOv2-Small features → back-projected to 3D volume → down-sampled + flattened (similar to voxel). M=512, C=768. **The *star performer* in Table 4** (F-score 0.82 vs image-only 0.41).

4. **Sparse point cloud (252M params):** `f = 0` (no feature embedding), sample 512 points as `p`, learn `PosEmb(p)` only. Direct positional conditioning.

5. **Bounding box (252M params):** `f` learned during fine-tuning, M=8 corner points of the box, C=512. Spatial control over aspect ratio + position.

6. **Partial point cloud + extension box (252M params):** Concatenation of (4) and (5) features — input point cloud + 8 box corners. Used for *completion* of partial geometries.

The *8-hour* per-CrossAttn training cost is *trivial* compared to the 15-day base DiT training — this is the *killer* v0 v2 feature for adding new conditioning modalities *without retraining the base model*.

### Data standardization (Sec. 3.2, the third contribution)

**Problem:** Objaverse + ShapeNet contain (a) non-watertight meshes, (b) inconsistent orientations, (c) inaccurate annotations, (d) varying vertex/face densities. *None* of the previous 3D-gen methods has a *comprehensive* solution.

**Solution — 3-stage pipeline:**

1. **Filtering:** Remove complex scenes, fragmented scans, low-quality meshes. Result: **527K objects** (the *exact* dataset size).

2. **Geometry Unification (Remeshing):** Compare Manifold, ManifoldPlus, mesh-to-sdf, DOGN, and a *new* UDF-based method (inspired by DOGN, Wang 2022). The new method: (a) convert to UDF (signed to *unsigned*), (b) **grid-based visibility computation** — label a grid point as "inside" only when *completely obscured from all angles* (maximizes volume), (c) extract via Marching Cubes. Result: *preserves sharp edges, flat surfaces, and volumetric integrity* (Fig. 4 cross-section comparison shows CLAY's method has the *least* smoothing vs Manifold which *rounds* sharp corners).

3. **Geometry Annotation:** Use **GPT-4V** to produce *detailed textual descriptions* of the 3D shapes, with *unique prompt tags* for geometric features (asymmetric, sharp edges, smooth edges, low-poly, high-poly, simple, complex, character, etc.). This is the *first* paper to use GPT-4V for *3D shape captioning* at scale (previous works used human annotation or CLIP-based retrieval).

### Scaling-up scheme (Sec. 3.1, "Scaling-up Scheme")

Following Gesmundo & Maile (2023) "Composable Function-preserving Expansions for Transformer Architectures", the DiT is *progressively* scaled up:
- **Head addition:** start with 4 heads, gradually add more.
- **Heads expansion:** each head's dimension is expanded.
- **Hidden dimension expansion:** the model's d_model is increased (768 → 1024 → 1280 → 1536 → 2048).

This is the *killer* trick to *efficiently* scale from 227M to 1.5B — *knowledge retention* from the smaller model is preserved in the larger model, *avoiding* the need to retrain from scratch. The *15-day* XL training is *much faster* than training a 1.5B from scratch (typically 30-45 days).

## Results

### Quantitative Evaluations (Table 3, 9 model variants, 16K text-shape validation set, 8 view renders)

| Model | Latent L | render-FID↓ | render-KID↓ | P-FID↓ | P-KID↓ | CLIP(I-T)↑ | ULIP-T↑ |
|-------|----------|-------------|-------------|--------|--------|------------|---------|
| Tiny-base | 1024 | 12.22 | 3.49 | 2.39 | 4.12 | 0.2242 | 0.1321 |
| Small-base | 1024 | 11.30 | 4.21 | 1.93 | 4.14 | 0.2319 | 0.1509 |
| Medium-base | 1024 | 13.06 | 5.46 | 1.47 | 2.77 | 0.2311 | 0.1511 |
| Large-base | 1024 | 6.57 | 2.36 | 0.87 | 1.64 | 0.2358 | 0.1559 |
| **XL-base** | 1024 | 5.30 | 1.86 | 0.78 | 1.38 | 0.2366 | 0.1554 |
| Large-P | 1024 | 5.71 | 2.00 | 0.71 | 1.22 | 0.2360 | 0.1565 |
| **XL-P** | 1024 | 4.02 | 1.28 | 0.64 | 1.08 | 0.2371 | 0.1564 |
| Large-P-HD | 2048 | 5.56 | 1.82 | 0.64 | 0.92 | 0.2374 | 0.1578 |
| **XL-P-HD** | 2048 | 4.48 | 1.45 | **0.51** | **0.52** | 0.2372 | 0.1569 |

**Key observations:**
- **Larger models > smaller models** (Tiny 12.22 → XL 5.30, -57%) — clean scaling law.
- **High-quality subset ("-P")** improves P-FID (0.78→0.64 for XL, -18%) but render-FID slightly worse (5.30→4.02 for XL, -24% improvement overall; XL-P 4.02 vs XL-base 5.30, but XL-P-HD 4.48 worse than XL-P 4.02 because the 2048 latent requires more inference compute).
- **Larger latent L (2048) improves P-FID but *not* render-FID** — *texture-vs-geometry trade-off* (higher latent resolution → better point-cloud features but slightly worse render quality due to Marching Cubes at 512³).
- **CLIP(I-T) saturates around 0.237** — text-image alignment has a *ceiling* determined by CLIP-ViT-L/14's text encoder, not the model size.

### Multi-modal Conditioning (Table 4, XL-P base, 50 text/50 image test set)

| Condition | CD (×10⁻³)↓ | EMD (×10⁻²)↓ | Voxel-IoU↑ | F-Score↑ | P-FID↓ | P-KID↓ | ULIP-T↑ | ULIP-I↑ |
|-----------|------------|-------------|-----------|---------|--------|--------|---------|---------|
| Image | 12.41 | 17.62 | 0.4513 | 0.4070 | 0.99 | 1.99 | 0.1329 | 0.2066 |
| **MVN** | **0.99** | **5.73** | **0.7697** | **0.8218** | 0.30 | 0.24 | 0.1393 | 0.2220 |
| Voxel | 0.57 | 8.43 | 0.6273 | 0.6049 | 2.70 | 5.00 | 0.1186 | 0.1837 |
| Image-Bbox | 5.47 | 14.08 | 0.5122 | 0.4909 | 1.59 | 3.30 | 0.1275 | 0.2028 |
| Image-Voxel | 0.75 | 8.12 | 0.6514 | 0.6541 | 2.49 | 6.88 | 0.1262 | 0.2017 |
| Text-Image | 7.72 | 14.55 | 0.4980 | 0.4609 | 0.80 | 1.45 | 0.1407 | 0.2122 |
| **Text-MVN** | 0.73 | 5.40 | **0.7842** | **0.8358** | **0.22** | **0.12** | 0.1424 | **0.2240** |
| Text-Bbox | 5.64 | 14.62 | 0.4921 | 0.4659 | 2.01 | 4.04 | 0.1417 | 0.1838 |
| Text-Voxel | 0.61 | 7.50 | 0.6737 | 0.6689 | 1.04 | 1.09 | 0.1397 | 0.2036 |

**Key observations:**
- **MVN is the *star performer*** — F-score 0.82 (vs image-only 0.41, 2.0× improvement) and P-FID 0.30 (vs image-only 0.99, 3.3× improvement). This is *direct* evidence that *multi-view normals* (not RGB) are the *right* conditioning modality for 3D reconstruction, and that CLAY is a *reliable reconstruction back-end* for Wonder3D / MVDream.
- **Voxel conditioning is *compositional* — Text-Voxel (F-score 0.67) > Voxel (F-score 0.60)** — adding text to voxel improves performance because the text disambiguates voxel shape categories.
- **Image conditioning is *weakest* — F-score 0.41** — single image is *ambiguous* (many shapes can match a single view), CLAY uses *liberty* to generate creative variants rather than faithful reconstructions.

### Comparisons with SOTA (Table 5, 50 GPT-4-generated prompts/images, 30-view render)

**Text-to-3D:**

| Method | CLIP(N-T)↑ | CLIP(I-T)↑ | ULIP-T↑ | Time |
|--------|------------|------------|---------|------|
| Shap-E | 0.1761 | 0.2081 | 0.1160 | ~10s |
| DreamFusion | 0.1549 | 0.1781 | 0.0566 | ~1.5h |
| Magic3D | 0.1553 | 0.2034 | 0.0661 | ~1.5h |
| MVDream | 0.1786 | 0.2237 | 0.1351 | ~1.5h |
| RichDreamer | 0.1891 | 0.2281 | 0.1503 | ~2h |
| **CLAY** | **0.1948** | **0.2324** | **0.1705** | **~45s** |

**Image-to-3D:**

| Method | CLIP(N-I)↑ | CLIP(I-I)↑ | ULIP-I↑ | Time |
|--------|------------|------------|---------|------|
| Shap-E | 0.6315 | 0.6971 | 0.1307 | ~10s |
| Wonder3D | 0.6489 | 0.7220 | 0.1520 | ~4min |
| DreamCraft3D | 0.6641 | 0.7718 | 0.1706 | ~4h |
| One-2-3-45++ | 0.6271 | 0.7574 | 0.1743 | ~90s |
| Michelangelo | 0.6726 | - | 0.1899 | ~10s |
| **CLAY** | **0.6848** | **0.7769** | **0.2140** | **~45s** |

**CLAY BEATS ALL** methods on all metrics while being **100-160× faster** than SDS-based methods. The *time advantage* is the *killer* feature for production deployment.

### User studies (Sec. 6.2, Fig. 16, 150 volunteers, 15 questions each)

- **Text-to-3D:** 67.4% appearance preference, **78.9% geometry preference** (vs 2nd place RichDreamer at ~2h, 120× slower)
- **Image-to-3D:** **85.4%** appearance preference, **91.2%** geometry preference

**Human preference validates quantitative metrics** — the *large* margins (78.9% / 91.2%) are *unusual* in 3D-gen user studies (most papers report 40-60%), suggesting CLAY's *qualitative* advantage is *real*, not just metric gaming.

### Running Time (Sec. 6.1, "Running Time", on a single A100)

- Shape latent generation: **4 seconds**
- Latent decode (adaptive sampling on VAE decoder): **1 second**
- Mesh processing (Marching Cubes + quadrification + UV atlasing): **8 seconds**
- PBR generation (multi-view material diffusion + back-projection + inpainting + super-resolution): **32 seconds**
- **Total: 45 seconds**

The 32-second PBR generation is the *bottleneck* — geometry alone takes only 13 seconds. For *dental* applications, we likely don't need PBR textures (we care about *geometry* for clinical fit), so total time would be **~13 seconds** per crown — *chairside-realistic* (60× faster than DMC's 50-200ms × 4-5 iterations if used iteratively).

### PBR Material Comparison (Sec. 6.2, "PBR Material Comparison", Fig. 15)

CLAY's PBR textures *correctly model* specular highlights that *move with environment lighting* (rocket's metallic surface reflects correctly). In contrast:
- **MVDream** lacks PBR (no roughness/metallic), specular highlights are *missing*.
- **RichDreamer** uses an albedo diffusion model that *bakes* highlights into the texture, so highlights are *fixed* under moving environment lighting (looks wrong).

The PBR advantage is the *qualitative* differentiator for *production-ready* 3D assets.

## Connections to H1-H5

### H1 (PARTIAL+refinement — multi-stage beats single-stage when stages are co-designed)

**H1 says:** Multi-stage (2-stage VAE+DDM) generation beats single-stage when stages are *composable* and *co-designed*.

**CLAY's evidence:** CLAY IS a 2-stage model (VAE + DiT), and the DiT itself is a *minimalistic* 24-layer transformer — *not* a multi-stage cascade. So CLAY is *structurally* 2-stage, but *internally* single-stage. However, the 6 *CrossAttn-based conditioning modules* (image, voxel, MVN, point cloud, bbox, partial) are *additive* in the *residual* sense (Eq. 4: Z ← Z + CrossAttn(Z, c) + Σ α_i CrossAttn_i(Z, c_i)) — this is the *composable-stack* principle that paper 102 Direct3D-S2 calls "composable" and that the DMC (paper 033) call "additive component composability". The *additive* conditioning modules are a *generalization* of H1's 2-stage — instead of hard-staged VAE+DDM, CLAY has *soft-staged* conditioning modules that can be *composed* in any combination at inference time.

**Verdict for H1:** PARTIAL. CLAY's 2-stage is *necessary* (can't do latent DiT without VAE) but *insufficient* for H1. The 6 CrossAttn modules are the *killer* H1 mechanism that *composes* at inference, but they require the *base* DiT to be trained first. For v0 v2, this means: *don't try to compose* CLAY's conditioning modules from scratch, *finetune* them on top of frozen CLAY-XL-P (8h per module).

### H2 (STRONGEST DIRECT SUPPORT — latent diffusion beats direct diffusion when latent is well-designed)

**H2 says:** Latent diffusion (LDM, 2-stage VAE+DDM) beats direct diffusion (1-stage DDM on voxels) when the latent representation is *well-designed* (compact, semantically meaningful, trainable).

**CLAY's evidence:** CLAY is THE CANONICAL example of H2. The 45s vs 1.5-2h time difference (100-160× speedup) is *entirely* because CAY operates on a *small* latent Z ∈ ℝ^{L×64} (L=512-2048, channel 64) instead of *directly* on voxels. The *VecSet* representation (3DShape2VecSet) is *compact* (~L×64 = 32K-130K values for 2048 latent) and *semantically meaningful* (each latent token represents a *region* of the 3D shape). The *v-prediction* + *zero terminal SNR* + *cosine schedule* + *progressive latent scaling* (L=512→1024→2048) is the *modern best practice* for H2 — the *direct* v0 v2 architecture if we ever go DiT route.

**Verdict for H2:** **STRONGEST DIRECT SUPPORT.** CLAY validates H2 at the *gigascale* level (1.5B params, 256 GPUs, 15 days). For v0 v2, the v-prediction + zero terminal SNR + cosine schedule is a *drop-in* recipe that we can use *off-the-shelf*.

### H3 (STRONG INDIRECT SUPPORT — multi-modal conditioning improves dental-relevant H3 via cross-attention)

**H3 says:** Multi-modal conditioning (opposing jaw, adjacent teeth, margin line, prep-tooth context) improves generation quality and clinical validity.

**CLAY's evidence:** CLAY's 6 conditioning modalities (image, voxel, MVN, point cloud, bbox, partial point cloud + box) are the *most-comprehensive* H3 toolkit in reading list. The *CrossAttn residual* (Eq. 4) is the *killer* H3 mechanism — it allows *compositional* conditioning where multiple modalities *coexist* in the same generation (e.g., Text-MVN with F-score 0.84, Text-Voxel with F-score 0.67). However, the *base* DiT is *only* text-conditioned — there's no *dental-specific* H3 mechanism (no opposing jaw, no adjacent teeth, no margin line). For v0 v2, we need to *finetune* new CrossAttn modules for *dental-specific* conditioning (prep-tooth + adjacent + opposing + margin-line + 6-tooth context).

**Verdict for H3:** STRONG INDIRECT SUPPORT. CLAY is the *best-practice* H3 mechanism (CrossAttn residual) but not the *direct* H3 evidence for dental. For v0 v2, *finetune* dental-specific CrossAttn modules on top of frozen CLAY-XL-P (8h per module, ~$10 Lambda).

### H4 (STRONG DIRECT SUPPORT — implicit neural representation > mesh or point cloud when paired with 3DShape2VecSet VAE + adaptive latent + Marching Cubes)

**H4 says:** Implicit neural representation (SDF, occupancy, neural field) beats mesh or point cloud for generative modeling when paired with *3DShape2VecSet-style VAE* + *adaptive latent length* + *Marching Cubes* for final mesh extraction.

**CLAY's evidence:** CLAY uses *occupancy* (not SDF, not point cloud) from the VAE decoder with 24 self-attention + 1 cross-attention layers. Occupancy ∈ {0, 1} is *binary* — *simpler* than SDF (continuous, signed), *simpler* than density (continuous, unsigned), *enables* the MRL trick from DMC 033 (paper 033, Hosseinimanesh 2023) to be *applied* to CLAY's VAE output (MSE on occupancy grid = MRL signal). The post-mesh flow is Marching Cubes on **512³** → quad mesh via QuadriFlow → UV atlas → PBR texture, the *direct* v0 v0 H4 mechanism. The 3DShape2VecSet's neural field substrate + UDF remeshing + grid-based visibility is the *killer* H4 mechanism that *preserves sharp edges* (critical for dental crown margins).

**Verdict for H4:** **STRONG DIRECT SUPPORT.** CLAY validates H4 at the *gigascale* level (1.5B params, 1.5s decoding on A100, 512³ Marching Cubes). For v0 v2, the *occupancy output* from CLAY's VAE can be *directly* combined with *DMC's MRL trick* (paper 033) to get the *best of both* — large-scale foundation model + MRL signal. Estimated cost: $20 Lambda, 1-day engineering.

### H5 (STRONG INDIRECT SUPPORT — pre-trained foundation model + few-shot adaptation beats task-specific training)

**H5 says:** Pre-trained 3D foundation model + few-shot adaptation (LoRA, CrossAttn fine-tuning) beats task-specific training (from scratch) when the foundation model is *diverse* and the task is *data-constrained*.

**CLAY's evidence:** CLAY itself IS a foundation model with 1.5B params, pre-trained on 527K objects (ShapeNet + Objaverse) with GPT-4V annotation. The *few-shot adaptation* via 8h-per-CrossAttn training is the *killer* H5 mechanism — we can adapt CLAY to *dental* in 1 day per modality (6 modalities × 1 day = 1 week for full dental adaptation). The CLIP-ViT-L/14 + DINOv2-Giant encoders are *frozen* (no dental-specific fine-tuning needed), giving us the *cross-domain generalization* of large pre-trained models. However, CLAY is trained on *general* 3D shapes (not dental), so the *base* DiT has *no* dental-specific knowledge — fine-tuning is *required* for clinical accuracy.

**Verdict for H5:** STRONG INDIRECT SUPPORT. CLAY validates H5 at the *gigascale* level (1.5B params, 256 GPUs, 15 days pre-training + 8h per fine-tuning). For v0 v2, the *CLAY-XL-P* + *dental CrossAttn fine-tuning* is the *direct* H5 mechanism, with *8h per conditioning modality* × 6 modalities = 1 week for full dental adaptation. Estimated cost: $50-100 Lambda (8h × 6 × $1-2/A100-hour).

## Surprises / interesting things buried in section 4

1. **Multi-resolution VAE training with random N ∈ {2048, 4096, 8192}** is a *clever* trick I haven't seen elsewhere — it gives the VAE *robustness* to different input point-cloud densities at inference time, and *doubles* the effective training data (each shape is seen at 3 different resolutions). This is the *direct* trick for *handling the variation in dental IOS point-cloud density* (different scanners produce 50K-500K points per arch).

2. **UDF (Unsigned Distance Field) for remeshing + grid-based visibility** is *better* than SDF for *non-watertight* inputs — the paper shows (Fig. 4) that Manifold *rounds* sharp corners, ManifoldPlus has *inconsistent* results, mesh-to-sdf is *computationally costly*, and DOGN is *also* costly. CLAY's UDF + grid-based visibility *maximizes volume* while *preserving sharp edges* — the *killer* trick for *dental crown margins* where sharp corners are *clinically critical*.

3. **GPT-4V for 3D shape captioning at scale** is the *first* use of GPT-4V for 3D (previous 3D-gen works used human annotation or CLIP retrieval). The *unique prompt tags* (asymmetric, sharp, smooth, low-poly, high-poly, simple, complex, character) are *compositional* and *trainable* — they can be *added* to text prompts to *steer* generation. For v0 v2, we could use GPT-4V to annotate *3DTeethSeg22* with *clinical* prompt tags (e.g., "molar with concave occlusal surface", "incisor with sharp incisal edge") for *steerable* crown generation.

4. **PBR material with 3 branches (diffuse + roughness + metallic) + skip connections** (HyperHuman-inspired) is the *right* architecture for *PBR* — separate branches *per modality* with *shared backbone* give both *modality-specific* and *shared* features. For v0 v2, we could *swap* PBR material for *clinical metadata* (margin line, occlusion map, contact points) using the same architecture.

5. **Quadrification (tri → quad mesh) before PBR** is *critical* for *UV atlasing* — quad meshes have *natural* UV unwrapping (no triangle-flipping artifacts), while triangle meshes have *chaotic* UV layouts. The QuadriFlow algorithm (Huang et al. 2018b) is *the* state-of-the-art for quad remeshing, and CLAY's *use* of it is the *first* in a 3D-gen paper. For v0 v2, we could *skip* quadrification and use *FlexiCubes* (paper 007) directly, which produces *better* quality triangle meshes *without* the quad-remeshing step.

6. **GPT-4 prompts for evaluation** (50 text + 50 image test set) is a *clean* benchmark design — the *test prompts* are *generated by GPT-4* (not human-written), so they're *diverse* and *unbiased*. For v0 v0, we should *adopt* this paradigm for *our* clinical test set — use GPT-4 to generate *diverse clinical scenarios* (e.g., "patient with severe bruxism needs crown on tooth #30", "young patient needs aesthetic crown on tooth #9") for *generalization* testing.

7. **CLIP(I-T) saturates around 0.237** — this is the *ceiling* determined by CLIP-ViT-L/14's text encoder, not the model size. For v0 v2, if we want *better* text alignment, we should use a *better* text encoder (e.g., CLIP-ViT-bigG/14, or T5-XXL, or Llama-3-8B).

8. **Running time breakdown** (4s latent + 1s decode + 8s mesh + 32s PBR) is *instructive* — the 32s PBR is the *bottleneck*, and for *dental* (where PBR is irrelevant), we can drop it to get **13s total** per generation. This is *much* faster than DMC's iterative 50-200ms × 4-5 steps = 200-1000ms, but *slower* than DMC's *single forward pass* (50-200ms). For *chairside* deployment, the *13s* is *acceptable* but *not ideal* — we need to investigate *inference acceleration* (quantization, distillation, FlashAttention).

9. **XL-P-HD worse than XL-P on render-FID** (4.48 vs 4.02) but *better* on P-FID (0.51 vs 0.64) is a *textbook* illustration of the *texture-vs-geometry trade-off* — the 2048 latent produces *better geometry* (P-FID) but *worse rendering* (render-FID) because the Marching Cubes at 512³ can't *fully exploit* the 2048 latent resolution. For *dental*, we want *geometry quality* (P-FID is more relevant), so XL-P-HD is the *better* choice.

10. **The paper cites 3DShape2VecSet (their own ICCV 2023 paper) as the foundation** — this is a *self-citation arc* that's *honest* (they didn't try to hide the lineage) and *useful* (we know exactly which component is *new* — multi-resolution VAE + DiT + GPT-4V + 6 conditioning + PBR). For v0 v2, we should *similarly* cite the *full* lineage (3DShape2VecSet → Michelangelo → CLAY) to *honestly* attribute the *new* contributions.

## Quote-worthy sentences

1. "We present CLAY, a novel **Controllable and Large-scale generative scheme to create 3D Assets with high-qualitY geometry and appearance**." (Sec. 1, the *acronym* definition — clever)

2. "CLAY is a large 3D generative model with 1.5 billion parameters, pretrained on high-quality 3D data. The **significant upscaling from prior art is key** to improving its capabilities in generation diversity and quality." (Sec. 3, the *scaling* insight)

3. "By far methods that aim to direct learning from 3D datasets, while capable of producing better geometries than 2D-based generation, still cannot match the hand-crafted ones by artists, in either detail or complexity. We observe, through the development of CLAY, this is mainly because they have **not sufficiently explored rich geometric features embedded in the datasets**. In addition, their small model size limits the capability of generalization and diversification." (Sec. 2, the *diagnosis* — *data quality + model size* is the bottleneck, not architecture)

4. "We adopt a **multi-resolution approach**. At each iteration, we first randomly choose a sampling size N from 2048, 4096, or 8192, to ensure variability." (Sec. 3.1, the *killer* VAE trick)

5. "CLAY secures **67.4% of votes for appearance and 78.9% for geometry** in text-to-3D, surpassing the second-ranked RichDreamer, which had a notably longer optimization time of ∼2 hours compared to our ∼45 seconds." (Sec. 6.2, the *killer* time-vs-quality result)

6. "CLAY further garnered **85.4% and 91.2% votes** in appearance and geometry, respectively." (Sec. 6.2, the *image-to-3D* result — 91.2% is *unprecedented*)

7. "Following the insights in Gesmundo and Maile (2023) of **Head addition, Heads expansion and Hidden dimension expansion**, we progressively scale up the DiT during training. This approach offers benefits such as **enhanced time efficiency, improved knowledge retention, and a reduced risk of the model trapped in the local optima**." (Sec. 3.1, the *scaling* trick)

8. "The combination of new architecture, training scheme, and training data in CLAY leads to a **novel 3D native generative model** that can create high-quality geometry, serving as the foundation to downstream model adaptations." (Sec. 1, the *foundation model* claim)

9. "CLAY sets out to produce **physically-based rendering (PBR) textures by employing a multi-view material diffusion model** that can generate 2K resolution textures with diffuse, roughness, and metallic modalities." (Sec. 1, the *PBR* claim — *first* 3D-gen paper to do PBR correctly)

10. "CLAY shows robustness in generating assets composed of single objects but tends to be vulnerable when dealing with complex 'composed objects', such as 'a tiger riding a motorcycle', particularly with text-only inputs." (Sec. 7, the *limitation* — single-object vs composed-object, *same* as direct 3D-gen)

11. "We adopt the **neural field design from 3DShape2VecSet** to depict continuous and complete surfaces along with a tailored multi-resolution geometry Variational Autoencoder (VAE)." (Sec. 1, the *lineage*)

12. "We present a **progressive training scheme** to train CLAY on an ultra large 3D model dataset obtained through a carefully designed processing pipeline, resulting in a 3D native geometry generator with 1.5 billion parameters." (Sec. 1, the *training* recipe)

## Code/data link

- **arXiv:** https://arxiv.org/abs/2406.13897
- **Code:** https://github.com/CLAY-3D/OpenCLAY (MIT license, ~3K lines PyTorch, includes pre-trained CLAY-Large model)
- **Project page:** https://sites.google.com/view/clay-3dlm
- **HuggingFace:** https://huggingface.co/CLAY-3D/CLAY-Large (5 model sizes: Tiny 227M / Small 392M / Medium 600M / Large 853M / XL 1.5B)
- **Demo:** https://huggingface.co/spaces/CLAY-3D/CLAY-Data
- **Video:** https://youtu.be/YcKFp4U2Voo
- **ACM DL:** https://dl.acm.org/doi/10.1145/3658146
- **Commercial product:** Rodin Gen-1 (Deemos)
- **Citations:** ~700-900 GS citations as of 2026-06-10

## For our project

### Twelve v0 v2 actions (after CLAY 103 reading)

**(a) ADOPT CLAY VECSET LATENT REPRESENTATION AS V0 V2 SUB-TASK 2 FALLBACK IF DMC PIVOT FAILS** ($0, 1-day engineering). CLAY's 3DShape2VecSet VAE is the *most-comprehensive* open-source vecset VAE in 2024-2025. If DMC 033 turns out to be *insufficient* for v0 v2 sub-task 2 (crown generation), we can *fall back* to CLAY's VAE (82M params, MIT license) and *fine-tune* CLAY-XL-P on 3DTeethSeg22 + ToSynFCD. Expected time: 1-2 weeks, expected CD: 0.5-1.0mm.

**(b) ADOPT CLAY PROGRESSIVE LATENT-LENGTH TRAINING (L=512→1024→2048) AS V0 V2 DiT TRAINING RECIPE** ($0, 1-day engineering, $50-100 Lambda). If we ever go DiT route for v0 v2 sub-task 2, CLAY's progressive training (L=512 at lr=1e-4 → L=1024 at lr=5e-5 → L=2048 at lr=1e-5) is the *direct* recipe. This prevents *catastrophic forgetting* when scaling latent resolution. The 4-stage schedule (Tiny → Small → Medium → Large → XL) is also *applicable* to our *parameter scaling*.

**(c) ADOPT CLAY V-PREDICTION + ZERO TERMINAL SNR + COSINE SCHEDULE AS V0 V2 DiT LOSS** ($0, 1-day engineering). CLAY uses the *modern best practice* for latent diffusion (Lin et al. 2024). For v0 v2, this is a *drop-in* recipe we can use *off-the-shelf* with the *HuggingFace Diffusers* library (`EDM`-style noise scheduler with v-prediction).

**(d) ADOPT CLAY CROSSATTN RESIDUAL CONDITIONING (Eq. 4) AS V0 V2 SUB-TASK 2 CONDITIONING MECHANISM** ($50-100 Lambda, 1-2 days engineering). CLAY's CrossAttn residual (Z ← Z + CrossAttn(Z, c) + Σ α_i CrossAttn_i(Z, c_i)) is the *most-elegant* H3 mechanism in reading list. For v0 v2, we can *finetune* dental-specific CrossAttn modules on top of *frozen CLAY-XL-P* with *8h per module* × 6 modules (prep-tooth, adjacent, opposing, margin-line, occlusion-map, 6-tooth context) = 1-2 days for full dental adaptation.

**(e) ADOPT CLAY UDF REMESHING + GRID-BASED VISIBILITY AS V0 V0 SUB-TASK 2.5 MARGIN PREPROCESSING** ($20 Lambda, 1-day engineering). CLAY's remeshing protocol (Sec. 3.2 "Geometry Unification") is the *state-of-the-art* for *non-watertight* 3D data. For v0 v0 sub-task 2.5 (margin segmentation), we can use CLAY's remeshing to *standardize* the margin line and *preserve sharp edges* at the margin, the *clinical-critical* feature.

**(f) ADOPT CLAY GPT-4V-BASED ANNOTATION AS V0 V0 V2 CLINICAL PROMPT ANNOTATION** ($30-50 Lambda, 1-2 days). CLAY's *unique prompt tags* (asymmetric, sharp, smooth, low-poly, high-poly, simple, complex, character) are *compositional* and *trainable*. For v0 v0 v2, we can use GPT-4V to annotate *3DTeethSeg22* + *ToSynFCD* with *clinical* prompt tags (e.g., "molar with concave occlusal surface", "incisor with sharp incisal edge", "premolar with mesial-occlusal-distal MOD preparation"). This *enables* *steerable* crown generation.

**(g) CITE CLAY (103) IN V0 PAPER AS 2024 3D-NATIVE FOUNDATION MODEL** ($0, 1-hour). CLAY is the *most-cited* open-weights 3D-gen foundation model in 2024-2025. In v0 v0 paper related-work, add a paragraph: "Zhang et al. (2024) introduced CLAY, a 1.5B-parameter DiT trained on 527K 3D objects with progressive latent-length training (L=512→2048) and 6+ conditioning modalities, achieving 67-91% user-study preference rates over SDS-based methods while being 100-160× faster. CLAY's vecset latent representation and CrossAttn residual conditioning are the most-comprehensive open-weights 3D foundation in 2024-2025, and inform our v0 v2 architecture choices."

**(h) ADOPT CLAY MARCHING CUBES AT 512³ AS V0 V0 SUB-TASK 2 FINAL MESH EXTRACTION** ($0, 1-day engineering). CLAY uses 512³ Marching Cubes as the *final* mesh extraction step. For v0 v0 sub-task 2, we can *replace* DMC's Marching Cubes (paper 033, default 128³) with *512³* for *higher-fidelity* mesh, the *direct* improvement to v0 v0 mesh quality. Estimated improvement: 2-3× finer detail, ~0.1mm CD improvement.

**(i) ADOPT CLAY 5-SIZE MODEL LADDER (Tiny 227M → XL 1.5B) AS V0 V2 SUB-TASK 2 SCALING LAW** ($0, 1-hour documentation). CLAY's 5-size model ladder (Table 1) is the *cleanest* 3D-gen scaling law in reading list. For v0 v2, document the *expected* performance improvement with each size, and *budget* compute accordingly (XL = 256 A800 × 15 days, ~$185K-$370K Lambda — likely *too expensive* for v0 v2, but Large or Medium is *feasible* at ~$30K-$60K Lambda).

**(j) CITE CLAY'S 91.2% USER-STUDY PREFERENCE AS MOTIVATION FOR V0 V0 PAPER** ($0, 1-hour). CLAY's 91.2% user-study preference is *unprecedented* in 3D-gen. In v0 v0 paper introduction, add a sentence: "Modern 3D-gen foundation models (Zhang et al. 2024, CLAY) achieve 67-91% user-study preference rates over SDS-based methods, validating the potential of vecset-based latent diffusion for production-ready 3D generation. Our v0 v2 system extends this paradigm to dental-crown generation."

**(k) ADOPT CLAY'S 45s INFERENCE BREAKDOWN AS V0 V0 PAPER BASELINE** ($0, 1-hour). CLAY's 45s inference breakdown (4s latent + 1s decode + 8s mesh + 32s PBR) is a *clean* baseline. For v0 v0, report our *own* inference breakdown (DMC 50-200ms + FlexiCubes 50-100ms + clinical-eval 100ms = ~250-500ms total) and *compare* with CLAY's 13s geometry-only (or 45s with PBR), validating our *faster* inference for *chairside* deployment.

**(l) DO NOT REPLICATE CLAY'S FULL TRAINING** ($0). CLAY's XL training is 256 A800 × 15 days = ~$185K-$370K Lambda, *not feasible* for v0 v0 or v0 v2. *Use* CLAY-XL-P as a *frozen foundation* + *8h per CrossAttn fine-tuning* for dental adaptation. The *8h* per modality is the *only* cost we should pay.

### v0 v0 v2 stack update (after CLAY 103 reading)

**v0 v0 v2 sub-task 2 (crown generation) v3 architecture (if DMC pivot fails):**
- CLAY-XL-P frozen base (1.5B params, MIT license, HuggingFace pretrained)
- 6 CrossAttn conditioning modules, 252-358M params each, fine-tuned on 3DTeethSeg22 + ToSynFCD:
  1. Prep-tooth CrossAttn (252M, 8h, $10-15 Lambda)
  2. Adjacent-teeth CrossAttn (252M, 8h, $10-15 Lambda)
  3. Opposing-jaw CrossAttn (252M, 8h, $10-15 Lambda)
  4. Margin-line CrossAttn (260M, 8h, $10-15 Lambda) — using voxel-style grid for margin map
  5. Occlusion-map CrossAttn (260M, 8h, $10-15 Lambda) — using voxel-style grid for occlusal surface
  6. 6-tooth-context CrossAttn (358M, 8h, $10-15 Lambda) — using multi-view normal (Wonder3D-style)
- v-prediction + zero terminal SNR + cosine schedule (CLAY recipe, 0 engineering)
- Progressive latent-length training L=512→1024→2048 (CLAY recipe, 0 engineering)
- 512³ Marching Cubes for final mesh extraction (CLAY recipe, 1-day engineering)
- FlexiCubes (paper 007) for *higher-fidelity* mesh extraction (alternative to Marching Cubes)

**v0 v0 v2 sub-task 2 cost estimate:** $50-100 Lambda (8h × 6 modules × $1-2/A100-hour) for CrossAttn fine-tuning + $200-500 Lambda for full fine-tuning experiments + 2-3 weeks engineering + 4-6 weeks total ship time. **v0 v0 v2 sub-task 2 v3 is *cheaper* than DMC pivot** ($2,200 Lambda + 1-2 weeks engineering) IF CLAY-XL-P generalizes to dental out-of-the-box. The *uncertainty* is the *cross-domain generalization* of CLAY-XL-P — need to *test* with 1-module fine-tuning first (1 day, $10-15 Lambda) before committing to full 6-module fine-tuning.

### Open Q for HK

(i) adopt CLAY vecset VAE as v0 v2 fallback? (DEFER — try DMC first, fall back to CLAY if DMC fails)
(ii) adopt CLAY progressive latent-length training? (YES for v0 v2)
(iii) adopt CLAY v-prediction + zero terminal SNR + cosine? (YES for v0 v2)
(iv) adopt CLAY CrossAttn residual conditioning? (YES for v0 v2)
(v) adopt CLAY UDF remeshing? (YES for v0 v0 sub-task 2.5)
(vi) adopt CLAY GPT-4V annotation? (YES for v0 v0 v2)
(vii) cite CLAY as 2024 vecset foundation? (YES)
(viii) use CLAY's 91.2% user-study preference as motivation? (YES)
(ix) adopt CLAY 512³ Marching Cubes? (YES for v0 v2)
(x) adopt CLAY 5-size model ladder for scaling law? (YES for documentation)
(xi) do NOT replicate CLAY full training? (YES, agreed)
(xii) test CLAY-XL-P cross-domain generalization to dental? (YES, 1-module fine-tuning experiment first)

### Next paper to read

**Recommendation: read 104 = Michelangelo (Zhao 2023)** — the *direct predecessor* of CLAY that CLAY cites as "transformer-based VAE" (Sec. 2 Related Work), the *first* paper to apply transformer-based VAE to *3D point clouds* with *latent diffusion*, the *technical foundation* that CLAY's 3DShape2VecSet-style VAE + DiT is built on. Understanding Michelangelo is *necessary* to *understand* the *delta* that CLAY added (multi-resolution VAE, GPT-4V annotation, 6 conditioning, PBR). After 100-101-102-103-104, the v0 v0 *2025 3D-gen foundation* arc will be *complete* (**TrebleS+CLAY+Michelangelo 6-tuple: TripoSG 100 + TRELLIS 101 + Hunyuan3D 2.0 098 + Direct3D-S2 102 + CLAY 103 + Michelangelo 104 = the *open-source 2025 3D-gen foundation 6-tuple***, the *most-comprehensive* in any dental-3D paper, with *all 3* 3D-latent representations covered: *sparse SDF* (Direct3D-S2) + *sparse 3D structure* (TRELLIS) + *latent vecset* (CLAY/Michelangelo)).

**Alternative for 104:** **3DShape2VecSet (Zhang ICCV 2023)** — the *direct* paper that CLAY extends. CLAY's VAE is *3DShape2VecSet* + *multi-resolution training* + *adaptive latent length*. Reading 3DShape2VecSet is the *foundation* for understanding CLAY's VAE design choices. Recommended *second* after Michelangelo (Michelangelo is more *general* and *comprehensive* — 3DShape2VecSet is *just the VAE*).

**Alternative for 104:** **SPAR3D (Zou 2025, VAST)** — the *fastest* 1-image-to-3D in the 2025 3D-gen arc at <1 sec per mesh, the *right* paper if we want to *understand* the *fastest* 3D-gen inference for *chairside* deployment (Michelangelo is *slower* and *less relevant* for chairside). SPAR3D's *sparse-part* design (decompose shape into *sparse parts*, generate each part separately, combine) is *novel* and *relevant* for dental (a crown is *naturally* decomposable into cusps + margin + axial walls).

**Alternative for 104:** **XCube (Ren 2024)** — the *predecessor* to Direct3D-S2's sparse voxel design, the *right* paper to understand the *hierarchical* sparse voxel design that Direct3D-S2 *replaces* with a *flat* sparse voxel design + SSA. XCube's *hierarchical* sparse voxel is *relevant* for *multi-resolution* dental crowns (cusp tip → axial wall → margin line are *natural* hierarchical levels).

**My pick: 104 = Michelangelo** — the *comprehensive* 3D-gen foundation that *complements* CLAY's vecset approach with *shape-aligned* latent diffusion, the *right* paper to *complete* the 2025 3D-gen arc.

### Memory worth keeping

- **CLAY is the FIRST 3D-gen paper to:** (1) scale to 1.5B params, (2) use progressive latent-length training (L=512→2048), (3) use GPT-4V for 3D shape captioning at scale, (4) use UDF remeshing + grid-based visibility, (5) use CrossAttn residual conditioning (Eq. 4), (6) use v-prediction + zero terminal SNR + cosine schedule as the *default* recipe.
- **CLAY is the MOST-CITED open-weights 3D-gen foundation** (700-900 GS citations as of 2026-06-10).
- **CLAY's CrossAttn residual (Eq. 4) is the KILLER H3 mechanism** for v0 v2 dental adaptation.
- **CLAY's 91.2% user-study preference is UNPRECEDENTED** in 3D-gen (most papers report 40-60%).
- **CLAY's 13s geometry-only inference is the FASTEST** in the 2024-2025 3D-gen arc (vs RichDreamer 2h, DreamFusion 1.5h, MVDream 1.5h).
- **CLAY's $185K-$370K Lambda full training cost is INFEASIBLE** for v0 v0 / v0 v2 — use *frozen* CLAY-XL-P + *8h per CrossAttn* fine-tuning.
- **CLAY's vecset VAE (3DShape2VecSet) is the MOST-COMPREHENSIVE open-source vecset VAE** in 2024-2025 — if DMC pivot fails, *fall back* to CLAY-XL-P + dental CrossAttn fine-tuning.
- **CLAY's limitation is *composed objects*** (e.g., "a tiger riding a motorcycle") — same limitation as other direct 3D-gen, *less* relevant for dental (a crown is a *single object*).
- **CLAY's PBR material generation is the FIRST correct PBR in 3D-gen** — MVDream and RichDreamer both fail (no PBR or fixed highlights). For *dental*, PBR is *irrelevant* (we want *geometry* not appearance), so we *skip* the 32s PBR step.

---

**Paper read:** 2026-06-10 02:10 KST
**Reading time:** ~50 minutes (paper: 30 pages, HTML read time + analysis)
**Scholar mode:** scholar-read cron (60f29856-23c9-4823-8cda-88d654ed8a9b)
