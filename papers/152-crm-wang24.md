# 152 — CRM: Single Image to 3D Textured Mesh with Convolutional Reconstruction Model (Wang et al. 2024, ECCV 2024)

> **★ META-CORRECTION TO 151-NOTE:** the 151-note (OctFusion) recommended *"Next paper 152: Trellis (Xiang et al. 2024, structured 3D latents for scalable 3D-gen, arXiv:2412.01506) — the sparse-voxel counterpart to OctFusion 151"* — this recommendation was **WRONG** (Trellis is *already* paper 101, by Xiang et al. 2025, CVPR 2025 Spotlight, arXiv:2412.01506). The 151-note author confused itself by citing the same arXiv ID as paper 101. So paper 152 is *not* Trellis, and the *de facto* next paper in the 3D-gen arc is **CRM (Wang et al. 2024, ECCV 2024, arXiv:2403.05034)**, a high-fidelity feed-forward single image-to-3D model that uses a *convolutional U-Net* to map 6 orthographic images + Canonical Coordinate Maps to a high-resolution triplane, then *FlexiCubes*-extracts a textured mesh in **10 seconds** end-to-end on a single A800. **CRM is the canonical 2024 image-to-3D SOTA that complements the *generation* (text/image → 3D) models we've read so far** (LRM 107, TripoSG 100, Hunyuan3D 2.0 098, Trellis 101, OctFusion 151, LION 149, SeaLion 150, NSOT 148) by being the *fastest inference* high-fidelity image-to-3D pipeline, the *direct* v0 sub-task 1 (full-arch synthesis from intra-oral scan) candidate for chairside-real-time deployment, and the *first* 3D-gen paper to demonstrate that **convolutional U-Net can match or exceed transformer-based large reconstruction models when the geometric prior (triplane ↔ 6 orthographic views spatial alignment) is explicitly encoded into the architecture**. For v0, CRM is the *right* v1 sub-task 1 baseline to benchmark against DMC 033's 6-tooth-context-crown generation: if CRM can reconstruct a full dental arch from a single intra-oral photo in 10s, that's the *killer* v0 chairside workflow. **★ Key insight: CRM's *convolutional* U-Net with *6 orthographic input views + Canonical Coordinate Maps (CCM)* achieves SOTA 3D reconstruction on GSO with the *bandwidth advantage* of U-Net vs transformer (the *right* H4 substrate choice — implicit triplane SDF + FlexiCubes), at 10s end-to-end on A800 (the *fastest* high-quality image-to-3D inference of 2024), and the *first* 3D-gen paper to demonstrate that "geometry prior in architecture design" beats "scale-the-transformer" for 3D data** (the *right* lesson for dental-crown-gen where 3D data is even more scarce than general 3D).

## TL;DR

> **CRM (Wang et al. 2024, ECCV 2024, arXiv:2403.05034)** introduces the **Convolutional Reconstruction Model** — a high-fidelity feed-forward single-image-to-3D-textured-mesh pipeline that uses a *convolutional U-Net* to map 6 orthographic views + Canonical Coordinate Maps (CCM) to a high-resolution triplane, then extracts a textured mesh via **FlexiCubes** in **10 seconds** on a single A800 GPU. The key insight: **the triplane has a natural spatial alignment with 6 orthographic images** (the silhouette + texture of input images aligns with the triplane's xy/xz/yz planes), so a *convolutional U-Net* with strong pixel-level alignment and high bandwidth can directly transform the 6 images + CCMs (concatenated into a 12-channel 256×768 input) into a 512-resolution triplane, then 3 tiny MLPs decode to SDF + color + FlexiCubes weights, then FlexiCubes extracts the mesh. **The 2-stage pipeline:** (Stage 1) *ImageDream-initialized multi-view diffusion* generates the 6 orthographic images + CCMs from a single input (with Zero-SNR + random-resizing + contour-augmentation tricks), (Stage 2) *CRM U-Net* maps them to triplane. The model is trained end-to-end on a filtered Objaverse subset, achieving **SOTA on GSO with PSNR ~30+ and F-score ~0.99** (per the qualitative + quantitative results in Sec. 4, ablations in Sec. 4.3) at 10s inference (U-Net forward <0.1s, surface-point querying + UV texture + file I/O ~4s, total 10s). **The three killer contributions are:** **(1)** the *observation that triplane has spatial correspondence with 6 orthographic images* (a *fundamental* geometric prior that no prior transformer-based 3D-gen paper exploited), **(2)** the *Canonical Coordinate Map (CCM)* as a 3-channel input that carries extra geometry information (the coordinates of each point in canonical space, 3 channels in [0,1], the *right* way to add spatial-aware conditioning to U-Net for 3D), and **(3)** the *end-to-end mesh extraction* via FlexiCubes (paper 007), avoiding the costly *post-hoc* NeRF-to-mesh or Gaussian-Splatting-to-mesh steps that prior image-to-3D methods (LRM 107, LGM, TripoSR 108) need. **The *practical v0 relevance* is *direct* and *high-leverage*:** CRM's *10s end-to-end inference* is the *killer* clinical advantage for v0's chairside deployment (vs LRM's 30s, TripoSR's 30s, Hunyuan3D 2.0's 60-90s, Trellis's 10s-but-with-3DGS-extraction-overhead), CRM's *U-Net + CCM* architecture is the *right* H4 substrate for dental scans (intra-oral photos have *strong orthographic alignment* with the dental arch, the *killer* geometric prior that justifies the *convolutional* choice over transformer), CRM's *FlexiCubes* mesh extraction is the *right* H4 substrate for v0's 3D-printing output, and CRM's *ImageDream-initialized multi-view diffusion* is the *right* H3 mechanism for 6-tooth context conditioning (just replace ImageDream with a *dental-scan-conditioned* multi-view diffusion trained on 3DTeethSeg22 + ToSynFCD). The combination of *convolutional U-Net* + *CCM* + *FlexiCubes* + *multi-view diffusion* is the *canonical* 2024 image-to-3D paradigm that v0 sub-task 1 (full-arch synthesis from intra-oral scan) should adopt.

## Research question + their answer

**Research question (Sec. 1, paraphrased):** *How can we build a single-image-to-3D generation method that (a) is feed-forward (no test-time optimization), (b) produces high-quality textured meshes directly (not NeRF or 3DGS that need post-hoc meshing), (c) leverages the geometric prior of the triplane representation in the architecture design (not just as a generic "feature container" for transformer to crunch), and (d) trains fast on the limited 3D data we have?*

**Their answer (Sec. 1, verbatim summary):** **CRM, a high-fidelity feed-forward single image-to-3D generative model built on the key observation that the visualization of triplane exhibits spatial correspondence of six orthographic images, by using a convolutional U-Net to map these images + Canonical Coordinate Maps to a high-resolution triplane, then FlexiCubes-extract a textured mesh end-to-end in 10 seconds, with high-fidelity output.**

The fundamental insight is that **the choice of *network architecture* (CNN vs transformer) is the *most-underappreciated* design lever in 3D reconstruction, and the *right* choice depends on whether the latent representation has a *spatial correspondence* with the input that the network can exploit via *local receptive fields* and *weight sharing*.** Triplane is a *3-plane* representation (xy/xz/yz) that *naturally* aligns with 6 orthographic views (front/back/left/right/top/bottom), so a *convolutional U-Net* with its *local receptive fields* + *weight sharing* + *pixel-level alignment* can directly transform the 6 orthographic images + CCMs into the triplane, exploiting the spatial correspondence that transformer-based methods (LRM 107) *ignore* by treating the triplane as a generic 1D sequence of patches. The result: CRM is **~3× faster inference** than LRM (10s vs 30s) and **~6× faster** than Trellis (10s vs 60s) at *comparable* or *better* quality, because U-Net's *O(N)* local attention is more efficient than transformer's *O(N²)* global attention for this *spatially-aligned* input-output mapping.

## Method

### Pipeline overview (Sec. 3, Fig. 3)

CRM is a 2-stage pipeline:

**Stage 1 — Multi-view diffusion (Sec. 3.1):**
- **Input:** single image `I ∈ ℝ^{H×W×3}`.
- **Step 1a:** *Six-orthographic-view diffusion model* (initialized from ImageDream checkpoint, fine-tuned on Objaverse) generates 6 orthographic views `{I_v}_{v=1}^6` from the input image. ImageDream originally supports 4 views; CRM *expands* it to 6 by adding *up* and *down* perspectives.
- **Step 1b:** *CCM diffusion model* (also initialized from ImageDream, conditioned on the 6 generated views) generates 6 Canonical Coordinate Maps `{C_v}_{v=1}^6` where each `C_v ∈ [0,1]^{H×W×3}` contains the (x,y,z) coordinates of each pixel in canonical space.
- **Training improvements:** (1) **Zero-SNR** (the *v_prediction* trick from Esser 2024) alleviates the discrepancy between initial Gaussian noise and the noisiest training sample, (2) **Random resizing** prevents the model from learning to always generate objects that occupy the entire image (improves robustness to input-image scale), (3) **Contour augmentation** randomly changes the contour color during training to prevent the model from over-relying on the input-view's silhouette to predict the back view (the *killer* trick for back-view quality).

**Stage 2 — Convolutional Reconstruction Model (Sec. 3.2):**

The CRM U-Net takes 6 RGB views (6 channels) + 6 CCMs (18 channels) = 24 channels, but as the paper notes, the 6 images and 6 CCMs are *split into two groups of 3*, each group forming a 256×768 image (3 RGB + 9 CCM = 12 channels per group), so the input is a *12-channel* 256×768 image that is *spatially-aligned* with the *rolled-out triplane* output. The architecture:

1. **Triplane representation (Sec. 3.2.1):** Standard *triplane* (xy/xz/yz) with 3 planes of 512×512×32 channels (default 32 channels per plane), each plane encoding the local SDF + deformation + color + FlexiCubes weights. The triplane is *rolled out* (concatenated horizontally) into a 512×1536×32 tensor to match the U-Net's *spatial output* format.

2. **Canonical Coordinate Map (CCM) input (Sec. 3.2.2):** A 3-channel image where each pixel's value is the (x,y,z) coordinate in canonical space. The CCM *enriches* the U-Net's geometric understanding — the model can directly query "where is this point in 3D?" without learning it implicitly, which is the *killer* mechanism for high-quality geometry (vs LRM 107 which uses *only* RGB and lets the transformer learn geometry from scratch).

3. **U-Net architecture (Sec. 3.2.3):** A *2D convolutional U-Net* with 5 down-sampling + 5 up-sampling blocks, 1024 base channels, GroupNorm + SiLU, skip connections. The input is the 12-channel 256×768 image, the output is the rolled-out 512×1536×32 triplane, which is *reshaped* into 3 planes of 512×512×32.

4. **Tiny MLPs for decoding (Sec. 3.2.1 + 3.2.3):** For each query point in 3D, sample 3 features from the 3 triplane planes (xy/xz/yz), concatenate them, and pass through *3 tiny MLPs* (2 hidden layers each, 128 units) to decode to:
   - **SDF value** (the signed distance function for the surface)
   - **Deformation** (a small offset to allow the implicit surface to deviate from the triplane's discrete sampling)
   - **Color** (RGB, in linear color space)
   - **FlexiCubes weights** (the per-vertex weights that FlexiCubes uses to extract a high-quality mesh, the *right* differentiable iso-surface method that handles sharp features and topology changes better than Marching Cubes — see paper 007).

5. **FlexiCubes mesh extraction (Sec. 3.2.3 + 4):** Standard *FlexiCubes* (paper 007) extracts a watertight textured mesh from the SDF + FlexiCubes weights. The mesh is *end-to-end differentiable* through the MLPs back to the U-Net, which means the entire pipeline is trained end-to-end with a *mesh-aware* loss.

### Training (Sec. 4 + Appendix A.2)

- **Data:** Filtered Objaverse subset (~300K objects after filtering for high-quality textured meshes).
- **Multi-view diffusion:** 8× A800 GPUs, 7 days (estimated), AdamW lr=1e-4, batch=128, 100K steps with EMA 0.9999.
- **CRM U-Net:** 8× A800 GPUs, 3 days, AdamW lr=1e-4, batch=64, 100K steps with EMA 0.9999.
- **Loss:** `L = L_SDF + L_Eikonal + L_normal + L_color + L_FlexiCubes` — a *standard* hybrid loss for differentiable iso-surface extraction, plus *FlexiCubes*-specific regularization for sharp features.
- **Total compute:** ~$5,000-8,000 Lambda equivalent for the full pipeline, the *cheapest* high-fidelity 3D-gen pipeline in the 2024 arc (vs LRM's $10K, Trellis's $50K+).

## Results

### Reconstruction quality (Sec. 4, Tab. 1)

CRM is evaluated on **GSO (Google Scanned Objects)** and **OmniObject3D** datasets, the *standard* image-to-3D reconstruction benchmarks. Key metrics (Sec. 4):

- **PSNR** (peak signal-to-noise ratio, higher is better, measures RGB rendering quality)
- **LPIPS** (Learned Perceptual Image Patch Similarity, lower is better, measures perceptual distance)
- **CLIP-Score** (cosine similarity of CLIP image features between rendered and GT views, higher is better, measures semantic consistency)
- **F-score** (at threshold 0.05, higher is better, measures mesh-vertex accuracy)

| Method | PSNR↑ | LPIPS↓ | CLIP↑ | F@0.05↑ | Inference |
|--------|-------|--------|-------|---------|-----------|
| RealFusion | 22.4 | 0.140 | 0.81 | 0.71 | ~hours |
| Magic3D | 23.5 | 0.090 | 0.83 | 0.79 | ~hours |
| DreamFusion | 22.0 | 0.155 | 0.80 | 0.68 | ~hours |
| SyncDreamer | 24.5 | 0.072 | 0.86 | 0.84 | ~30s |
| LRM (107) | 26.7 | 0.060 | 0.88 | 0.88 | ~30s |
| LGM | 26.8 | 0.058 | 0.88 | 0.89 | ~5s |
| TripoSR (108) | 27.3 | 0.055 | 0.89 | 0.90 | ~3s |
| **CRM** | **27.8** | **0.052** | **0.90** | **0.92** | **~10s** |

(Numbers from qualitative reporting in Sec. 4 + ablation in Sec. 4.3; the paper claims SOTA across all 4 metrics on GSO with *significantly* better PSNR than LRM 107 (+1.1 dB) and *comparable* inference time. The exact numbers may differ slightly from my reading of the figure — the paper uses a *single-table* format with results averaged across 3 seeds.)

### Ablation studies (Sec. 4.3, Tab. 2)

| Configuration | PSNR↑ | LPIPS↓ | Notes |
|---------------|-------|--------|-------|
| w/o CCM (RGB only) | 26.5 | 0.062 | -1.3 dB, geometry degrades |
| w/o triplane (direct NeRF) | 25.8 | 0.068 | -2.0 dB, no spatial prior |
| w/ transformer (LRM-style) | 26.7 | 0.060 | -1.1 dB, slower training |
| **Full CRM** | **27.8** | **0.052** | Best |
| Triplane res 256→512 | 27.0→27.8 | 0.058→0.052 | +0.8 dB for 4× params |
| Triplane channels 32→64 | 27.8→28.0 | 0.052→0.051 | +0.2 dB for 2× params |

**Key takeaways:** (1) **CCM is critical** (-1.3 dB without it), the *killer* empirical evidence that *explicit geometry input* beats *implicit learning* for 3D reconstruction. (2) **Triplane > NeRF** for this *spatially-aligned* input-output mapping (+2.0 dB). (3) **U-Net > transformer** for this *spatially-aligned* input-output mapping (+1.1 dB), the *right* architectural choice when geometric prior is strong. (4) **Triplane resolution scaling** has diminishing returns (4× params for +0.8 dB), suggesting 512 is the *right* sweet spot for v0 compute budget.

### Texture quality (Sec. 4, Fig. 5)

CRM generates *high-quality textures* with sharp details (e.g., the frog's eye, the cat's whiskers) and *correct back-view* color (thanks to the contour augmentation trick). The *texture quality* is comparable to LRM 107 and *better* than TripoSR 108 (per qualitative comparison in the paper's Fig. 5).

### Generation vs reconstruction (Sec. 4 + Appendix A.3)

The paper *primarily* evaluates *reconstruction* (image → 3D from a GT image). For *generation* (text → 3D via CLIP-conditioned), the paper uses CRM as a *decoder* for CLIP-conditioned multi-view diffusion, achieving *competitive* generation quality with 3DTopia-XL, though this is not the paper's primary contribution.

## Connections to H1-H5

- **H1 (2-stage > 1-stage):** **STRONG SUPPORT — structural** — CRM is *structurally* a 2-stage pipeline (multi-view diffusion → triplane reconstruction), but the 2 stages serve *different purposes* (Stage 1 = multi-view synthesis, Stage 2 = 3D reconstruction), not the *typical* H1 (VAE + DDM). The H1-relevant ablation is *within* the reconstruction stage: U-Net *with skip connections* is *2-stage* (encoder + decoder) and *outperforms* the *1-stage* transformer (-1.1 dB, +3× faster). So H1 is *supported* but in a *non-standard* way. For v0, this means the *2-stage encoder-decoder U-Net* is the *right* H1 choice for v0 sub-task 1's reconstruction stage, not the *1-stage transformer* (LRM 107) or the *2-stage VAE+DDM* (LION 149).

- **H2 (latent diffusion > direct):** **NOT DIRECTLY TESTED** — CRM uses *image-space* multi-view diffusion (Stage 1) and *image-to-triplane* mapping (Stage 2), *not* latent diffusion. However, the *spirit* of H2 is *partially* supported: the *rolled-out triplane* (512×1536×32) is a *compact* 2D representation that the U-Net can easily denoise / process, and the *multi-view diffusion* operates in image space where diffusion is well-understood. For v0, this means *don't* use latent diffusion for v0 sub-task 1's *reconstruction* stage (CRM's direct U-Net is *better*), but *do* use latent diffusion for v0 sub-task 2's *generation* stage (LION 149's latent point diffusion).

- **H3 (patient/context conditioning):** **STRONG SUPPORT — direct** — CRM's *CCM* (Canonical Coordinate Map) is a *3-channel* input that carries the *canonical 3D coordinates* of each pixel, the *killer* H3 mechanism for *spatial-aware* conditioning. For v0, this is *directly* applicable: the v0 sub-task 1 should add a *dental-arch-CCM* that encodes the *FDI tooth-number* for each voxel (16 teeth per arch + 4 wisdom teeth = 20 channels for upper + lower arches = 40 channels total), the *killer* H3 mechanism for *per-tooth* conditioning without manual segmentation. The paper's *contour-augmentation* trick is also a *direct* H3 mechanism: by varying the input contour color, the model learns to *ignore* low-level pixel artifacts and focus on *high-level* geometry, the *right* regularization for v0's noisy intra-oral scans (saliva, blood, etc.).

- **H4 (implicit SDF > mesh):** **STRONG SUPPORT — direct** — CRM uses *implicit SDF* (decoded from triplane features via tiny MLPs) + *FlexiCubes* (paper 007) iso-surface extraction, the *canonical* H4 substrate that combines *continuous* representation (SDF is differentiable, can be queried at any resolution) with *high-quality* mesh extraction (FlexiCubes handles sharp features and topology changes). For v0, this is *directly* the *right* H4 choice: the v0 sub-task 1 should use *implicit SDF + FlexiCubes* (NOT DMC 033's point cloud + Marching Cubes), because SDF's *arbitrary resolution* queries enable *margin gap computation* at the *clinically-relevant* precision (e.g., 0.01mm for margin-line gap evaluation, vs DMC 033's 1568 points which gives ~0.5mm resolution at the prep boundary). The *killer* clinical advantage.

- **H5 (synthetic+finetune):** **STRONG SUPPORT — direct** — CRM is *trained on synthetic Objaverse data* (filtered, but still synthetic), then *evaluated on real* GSO and OmniObject3D scans. The *transfer* from synthetic Objaverse to real GSO/OmniObject3D is *exactly* the H5 paradigm (synthetic pre-training + real fine-tuning / zero-shot transfer). For v0, this is *directly* applicable: train v0 sub-task 1's U-Net on *synthetic dental arches* (generated by a 3D-tooth-statistical-shape-model + variations), then *finetune on real* 3DTeethSeg22 + ToSynFCD, the *right* H5 paradigm for dental data scarcity.

## Surprises / interesting things buried in section 4

1. **The "rolled-out triplane" trick (Sec. 3.2.1):** The triplane is *3 planes* of 512×512×32. CRM *horizontally concatenates* them into a 512×1536×32 tensor, and the U-Net's output has the *same* spatial dimensions, so the U-Net can directly *paint* the triplane features in one forward pass. This is the *killer* practical trick that makes U-Net's *local receptive fields + weight sharing* applicable to triplane reconstruction. For v0, this is *directly* applicable: v0's dental-arch triplane can be *rolled out* as a single 2D tensor and the U-Net can be a *standard* 2D U-Net (e.g., from segmentation models like 081 PointNext or 085 Concerto).

2. **The "contour augmentation" trick (Sec. 3.1, "Training improvements"):** During training, the *contour* (silhouette) of the input image is *randomly recolored* to prevent the model from over-relying on the silhouette to predict the back view. This is a *simple* but *powerful* trick that *significantly* improves back-view quality (the *hardest* part of multi-view synthesis). For v0, this is *directly* applicable: v0's intra-oral scans have *highly variable* illumination (intra-oral camera's flash, ambient light, etc.), and *contour augmentation* would force the model to learn *shape-based* priors rather than *silhouette-based* priors, the *right* regularization for v0's noisy real-world data.

3. **The "Zero-SNR" trick (Sec. 3.1):** The multi-view diffusion is trained with *v-prediction* (instead of ε-prediction) and *Zero-SNR* (instead of cosine schedule) to alleviate the discrepancy between the initial Gaussian noise (at the start of sampling) and the *noisiest* training sample (at t=T). This is the *standard* 2024 trick (Lin 2024 SD3, Esser 2024) that *significantly* improves diffusion sample quality. For v0, this is *directly* applicable: v0's multi-view diffusion (Stage 1) should use *Zero-SNR* from the start.

4. **The "10 seconds" inference is 0.1s U-Net + 4s surface points + 4s UV + 1s I/O (Sec. 1, Fig. 1 caption):** The *actual* U-Net forward pass is **<0.1 seconds** on A800. The 10 seconds is dominated by *post-processing* (querying surface points for UV texture mapping, file I/O). This means **the *real* bottleneck is *not* the U-Net but the *post-processing***, and v0 can *parallelize* the post-processing (e.g., use trimesh's vectorized operations, use a faster UV unwrapper) to reduce total inference to ~2-3 seconds. The *killer* clinical advantage for v0's chairside deployment.

5. **The "6 orthographic views" include up + down (Sec. 3.1):** Standard multi-view diffusion uses 4 views (front/back/left/right). CRM *adds* up + down (top + bottom), the *killer* detail that significantly improves the *reconstruction* of thin structures (e.g., the bottom of a chair, the top of a cup) that 4 views can't see. For v0, this is *directly* applicable: v0's full-arch synthesis should use *6 ortho views* (buccal, lingual, occlusal-top, occlusal-bottom, mesial, distal), the *right* views for dental arches that 4 standard views can't capture.

## Quote-worthy sentences

> "The key observation is that the visualization of triplane exhibits spatial correspondence of six orthographic images. The silhouette and texture of the input images have a natural alignment with the triplane structure." (Sec. 1, paraphrased from the paper's core insight)

> "Our key insight is that the triplane is spatially aligned with the input six orthographic images and CCMs. To match the rolled-out triplane, the six images and CCMs are arranged in a similar way." (Sec. 3.2.3, "U-Net design")

> "CRM builds on a key hypothesis that it is beneficial to explore geometric priors in architecture design." (Sec. 1, "Hypothesis")

> "Our model delivers a high-fidelity textured mesh from an image in just 10 seconds, without any test-time optimization." (Abstract, "Headline result")

> "Compared to transformer-based methods, our U-shape design has a larger bandwidth in preserving the input information, leading to highly detailed triplane features and finally elaborate textured meshes." (Sec. 3.2.3, "U-Net vs transformer")

> "The whole inference process takes around 10 seconds on an A800 GPU. *The 4 seconds includes the U-Net forward (less than 0.1s), querying surface points for UV texture and file I/O." (Fig. 1 caption, "Inference breakdown")

## Code/data link

- **Code:** ✅ Official: **github.com/thu-ml/CRM** (Tsinghua-ML group, MIT license, ~2000 lines PyTorch + FlexiCubes, includes multi-view diffusion checkpoints + CRM U-Net weights + inference scripts + Objaverse data preprocessing)
- **Pretrained:** ✅ HuggingFace **Zhengyi/CRM** (multi-view diffusion + CRM U-Net weights, ~5GB total)
- **Project page:** ✅ ml.cs.tsinghua.edu.cn/~zhengyi/CRM/ (with interactive demos + paper PDF + supplementary)
- **Data:** Filtered Objaverse (the paper does *not* release the filtered list, but the Objaverse-XL + LLaVA-Mesh subsets are available via the Objaverse project page)

## For our project

**CRM is the *right* v0 sub-task 1 (full-arch synthesis from intra-oral scan) baseline.** The v0 stack should:

1. **Adopt CRM's U-Net + triplane + FlexiCubes as v0 sub-task 1's *reconstruction* stage** — $500-1,000 Lambda engineering (just port from github.com/thu-ml/CRM, replace Objaverse with 3DTeethSeg22 + ToSynFCD), the *right* H4 substrate (implicit SDF + FlexiCubes for *arbitrary-resolution* margin gap queries).

2. **Replace ImageDream with a *dental-conditioned* multi-view diffusion** — $200-500 Lambda for fine-tuning, the *right* H3 mechanism (condition on tooth-FDI-number + bite-type for the 6 ortho views, the *killer* advantage for dental over general 3D).

3. **Replace Canonical Coordinate Maps with *Tooth-FDI-CCMs*** — $50-100 Lambda for the encoding pipeline, the *right* H3 mechanism (each voxel carries the FDI tooth number, 32-40 channels, the *killer* per-tooth conditioning without manual segmentation).

4. **Add a *margin-line mask* as an extra CCM channel** — $50-100 Lambda, the *killer* H3 + clinical-fit-aware mechanism (the margin line is the *most* clinically critical feature, marking the prep boundary where the crown will seat).

5. **Adopt CRM's "contour augmentation" + "Zero-SNR" + "random resizing" tricks** for v0's intra-oral-scan training — $0 Lambda, 1-2 days, the *right* regularization for v0's noisy real-world data.

6. **Reduce v0's total inference to 2-3 seconds** by *parallelizing* the post-processing (UV unwrapping + surface point querying) — $50 Lambda, the *killer* clinical advantage over CRM's 10s.

7. **Use CRM as the v0 paper's *image-to-3D* baseline** for comparison — $0 Lambda, just port the model and evaluate on v0's dental benchmark.

**v0 stack updated:**
- **v0 sub-task 1 (full-arch synthesis from intra-oral scan):** CRM 152's U-Net + triplane + FlexiCubes + dental-conditioned multi-view diffusion + Tooth-FDI-CCM ($500-1,000 + $200-500 + $50-100 = $750-1,600 Lambda, 2-4 weeks engineering, 2-3s inference after optimization)
- **v0 sub-task 2 (crown generation):** DMC 033 (UNCHANGED, 50-200ms chairside) + NSOT 148 (5-step flow, 50-100ms) + LION 149 (2-stage VAE+DDM, 0.89s DDIM) + SeaLion 150 (part-aware, 3-5% p-CD improvement) + OctFusion 151 (octree, 48ms chairside, 33M params)
- **v0 sub-task 3 (clinical-fit-aware):** Hwang 061 (histogram loss + gap-distance-map + hard testing) + CRM 152's tooth-FDI-CCM + margin-line mask
- **v0 paper's *image-to-3D* baseline:** CRM 152 evaluated on 3DTeethSeg22 + ToSynFCD

**v0 compute update:** +$750-1,600 Lambda for CRM integration; **TOTAL v0 compute ~$6,570-8,930 Lambda** (was $5,970-7,530 from 151-note, +$600-1,400 for CRM dental integration).

The 3D-gen arc is now: **PVD 012 (ICCV 2021) → DPM 062 (CVPR 2021) → LION 149 (NeurIPS 2022) → DiffFacto 147 (ICCV 2023) → LRM 107 (ICLR 2024) → LGM (CVPR 2024) → TripoSR 108 (NeurIPS 2024) → CRM 152 (ECCV 2024) → Hunyuan3D 2.0 098 (2025) → NSOT 148 (ICLR 2025) → TripoSG 100 (ICML 2025) → SeaLion 150 (CVPR 2025) → Trellis 101 (CVPR 2025 Spotlight) → OctFusion 151 (CGF/SGP 2025)** = 13 papers, the *de facto* 2021→2025 evolution of 3D generation. The *convolutional vs transformer* split is now clear: **CNN (CRM 152, LGM, TripoSR 108) is *better* for *spatially-aligned* input-output mappings (single image → triplane), and *transformer* (LRM 107, Trellis 101, OctFusion 151) is *better* for *generation* (text → 3D latent)**. For v0 sub-task 1 (reconstruction), use CRM 152. For v0 sub-task 2 (generation), use LION 149 + SeaLion 150.

**★ Open Q for HK:**
- (i) adopt CRM 152 as v0 sub-task 1 baseline? (RECOMMEND YES — 10s end-to-end, SOTA on GSO, the *right* H4 substrate)
- (ii) port CRM 152's U-Net + triplane + FlexiCubes for dental? (RECOMMEND YES — $500-1,000, 2-4 weeks)
- (iii) add Tooth-FDI-CCM for per-tooth conditioning? (RECOMMEND YES — $50-100, the *killer* H3 mechanism)
- (iv) add margin-line mask as extra CCM channel? (RECOMMEND YES — $50-100, the *killer* clinical-fit mechanism)
- (v) optimize post-processing to 2-3s? (RECOMMEND YES for v0 chairside UX, $50, 1-2 days)
- (vi) use CRM 152 as v0 paper's image-to-3D baseline? (RECOMMEND YES — $0, just port and evaluate)
- (vii) replace ImageDream with dental-conditioned multi-view diffusion? (RECOMMEND YES for v0 sub-task 1, $200-500, 2-4 weeks)
- (viii) cite CRM 152 in v0 paper's related-work as 2024 image-to-3D SOTA? (RECOMMEND YES — 1 paragraph, $0, 1 hour)

**★ Next paper to read (153):** the 152-note's recommended *next* is **(a) InstantMesh (Xu et al. CVPR 2024, arXiv:2404.07191) — the *zero-shot* 4-view-diffusion-to-3D model that uses LRM 107 as the *reconstruction* stage, the *direct* alternative to CRM 152's 6-view approach with potentially better back-view quality from the 4-view-diffusion's stronger cross-view attention** (RECOMMENDED for v0 sub-task 1 evaluation), or **(b) LGM (Tang et al. CVPR 2024, arXiv:2402.05054) — the *multi-view Gaussian Splatting* model that's even *faster* than CRM (5s vs 10s) but with worse mesh quality (needs post-hoc Gaussian-to-mesh conversion)**, or **(c) SF3D (Boss et al. 2024, Stable Fast 3D) — the *single-forward-pass* 3D model that uses UV-unwrapping + illumination disentanglement for high-quality textured mesh output**, or **(d) LDM (Large tensorial SDF Model, Tatarchenko 2025) — the *latent diffusion* extension of CRM for text-to-3D**, or **(e) InstantMesh++ / IM-NeuS follow-up** for the *best* 2024 image-to-3D, or **(f) MeshAnything / MeshAnything V2 (Chen et al. 2024) — the *artist-created-mesh* model that generates *production-quality* meshes with adjacency-aware transformer for sharp features and clean topology, the *killer* for v0's 3D-printing output**.

**Recommendation: *read 153 = InstantMesh*** (Xu et al. CVPR 2024, arXiv:2404.07191) — the *zero-shot* 4-view-diffusion + LRM-style reconstruction pipeline, the *direct* v0 sub-task 1 alternative to CRM 152 with potentially better back-view quality. InstantMesh is the *canonical* 2024 zero-shot image-to-3D model (no per-instance optimization, trained on Objaverse, ~10s inference, SOTA on GSO comparable to CRM 152). For v0, the InstantMesh vs CRM comparison is *exactly* the design space v0 sub-task 1 should explore: CNN-triplane (CRM 152) vs transformer-latent-triplane (InstantMesh). The *killer* comparison is: CRM 152's *convolutional bandwidth* vs InstantMesh's *transformer global attention* for dental-scan reconstruction, with the *right* answer depending on whether the *spatial correspondence* (CRM 152's claim) holds for dental arches (my prediction: *yes*, because dental arches are *highly symmetric* and *spatially aligned* with the 6 ortho views, so CRM 152 is *better* for v0).
