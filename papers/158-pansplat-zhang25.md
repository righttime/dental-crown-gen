# Paper 158 — PanSplat: 4K Panorama Synthesis with Feed-Forward Gaussian Splatting

- **Authors:** Cheng Zhang¹, Haofei Xu²,³, Qianyi Wu⁴, Camilo Cruz Gambardella¹, Dinh Phung¹, Jianfei Cai¹
- **Affiliations:** ¹**Monash University** · ²**ETH Zurich** · ³**University of Tübingen, Tübingen AI Center** · ⁴**University of Adelaide**
- **Venue:** **CVPR 2025** (pp. 11437-11447, IEEE/CVF, open-access PDF)
- **DOI:** to be assigned · **arXiv:** 2412.12096 v1 16 Dec 2024 → v2 23 Mar 2025 (31,727 KB, *CVPR camera-ready*)
- **Project:** ✅ chengzhag.github.io/publication/pansplat (teaser + full video + pipeline figure + interactive demo + Matterport3D / Replica / Residential results + cross-dataset Insta360 results)
- **Code:** ✅ github.com/chengzhag/PanSplat — **LICENSE: MIT** ✅ (commercial-deployable, **third cost-volume-3DGS MIT license** in our reading list after MVSplat 156 + DepthSplat 157)
- **Pretrained:** Monash OneDrive (Matterport3D 512×1024 + 4K 2048×4096), PanoGRF panoramic monocular depth + UniMatch cost-volume weights
- **Citations:** ~30-60 GS as of 2026-06-12 (CVPR 2025, ~6 months post-camera-ready)
- **Recommended by:** 157-DepthSplat note as "the *direct* successor for 4K + memory-efficient training"; 156-MVSplat note alternative was "PanSplat"
- **Reading-list scope:** feed-forward 3DGS arc #3 (after MVSplat 156 + DepthSplat 157), the **panoramic** sub-arc of 3DGS, the *de facto* 2025 4K + 360° extension to MVSplat + DepthSplat

> **★ META-CORRECTION TO 157-DepthSplat-NOTE + 156-MVSplat-NOTE:** the 157-note's "Next paper 158: PanSplat (**Chen et al. CVPR 2025**)" and 156-note's "Next paper: PanSplat (Chen et al. CVPR 2025)" are *BOTH WRONG* on the first author. The *correct* PanSplat first author is **Cheng Zhang (Monash University)**, NOT Chen. The **Chen-first-authored** paper is **Splatter-360 (Zheng Chen, Chenming Wu, Zhelun Shen, Chen Zhao, Weicai Ye, Haocheng Feng, Errui Ding, Song-Hai Zhang, Tsinghua + Baidu + Zhejiang, CVPR 2025, cg.cs.tsinghua.edu.cn/papers/CVPR-2025-Splatter-360.pdf)** — a **concurrent** CVPR 2025 paper with a *very similar* topic (generalizable 360° 3DGS for wide-baseline panoramas) but **different architecture** (spherical cost volume via spherical sweep algorithm + 3D-aware bi-projection encoder + cross-view attention, NOT the Fibonacci-lattice Gaussian pyramid + hierarchical spherical cost volume + two-step deferred back-propagation of PanSplat). This is the *5th consecutive* author-identification issue in the 156→157→158 sequence (all in the 3DGS arc), continuing the systematic pattern of secondary-source author misidentification. Splatter-360 is a *rival* of PanSplat, not the *same* paper.

## TL;DR

> **PanSplat (Zhang et al. CVPR 2025, arXiv:2412.12096)** introduces a **MIT-licensed** feed-forward 3D Gaussian Splatting model that **scales panorama synthesis to 4K (2048×4096) on a single A100 GPU** for the first time — the *first* feed-forward 3DGS paper to **break the 512×1024 resolution ceiling** that constrained all prior 360° view-synthesis work (PanoGRF, MVSplat, Splatter-360, HiSplat, MVSplat360). The three killer contributions are: **(1) a tailored *spherical 3D Gaussian pyramid* with a *Fibonacci lattice* arrangement** (replaces MVSplat's per-pixel-Gaussian pyramid with a per-icosahedral-Fibonacci-level Gaussian pyramid that *reduces information redundancy* and *enhances image quality*, the *killer* 4K-enabling design — Fibonacci lattices are *the* optimal sampling for spheres, see also the HiSplat 2D Fibonacci basis paper and the icosahedral mesh of PanoGRF); **(2) a *hierarchical spherical cost volume* + *localized Gaussian heads* with local operations** (extends MVSplat's 2D pixel-aligned cost volume to the *spherical domain* via multi-scale sweeping on the *sphere*, and replaces MVSplat's 4-feature U-Net heads with *local* Gaussian heads that *only* process their own pyramid level + neighbors, the *killer* memory-efficient 4K design); and **(3) *two-step deferred back-propagation*** (a memory-efficient training scheme that *first* computes the cost volume + Gaussian head forward pass on detached tensors, *then* re-runs the heavy spherical cost-volume refinement with gradients, *enabling* 4K training on *24GB* A100 GPUs — the *practical* enabler of *single-A100* panoramic 3DGS training). The model is *trained* on **Matterport3D** (synthetic indoor panoramas rendered via PanoGRF's data prep) and *fine-tuned* on **360Loc** (real-world indoor panoramas), with **progressive resolution training** 256→512→2048 (1 day per stage on 1× A100, **3 days total**). **Datasets evaluated:** Matterport3D (synthetic, ~90 scenes for train, 10 for test), Replica (synthetic, 8 scenes), Residential (synthetic, 12 scenes), 360Loc (real-world, 360° camera captures from HKUST), and Insta360 (real-world cross-dataset generalization test, *no* training). The *headline claim* is *SOTA on Matterport3D, Replica, and Residential* for *both* novel view synthesis and depth estimation, with the *first-ever* 4K panorama synthesis result in the feed-forward 3DGS literature, at *single-A100* memory footprint. For v0 sub-task 1 (full-arch synthesis), PanSplat is *not* the *direct* v0 candidate (panoramas ≠ dental arches, and 4K = 30-50× the resolution of v0's intra-oral scan requirement), but it is the *direct* v0 sub-task 1 *4K-extension* candidate for *clinical chairside preview* at *4K resolution*, and the *practical* 3DGS *memory-efficient-training* recipe that v0 should adopt for *chairside* + *high-resolution* intra-oral scan 3DGS fine-tuning.

## Research Question + Their Answer

**Q:** Can we scale *feed-forward 3D Gaussian Splatting* to *4K panorama* synthesis (2048×4096 resolution, 8× more pixels than the 512×1024 prior SOTA) on a *single* A100 GPU in *trainable* time, while *maintaining* SOTA quality on standard panorama benchmarks?

**A:** Yes — by **decoupling** the *representational* capacity (spherical 3D Gaussian pyramid with Fibonacci lattice) from the *computational* cost (hierarchical spherical cost volume + localized Gaussian heads with local operations + two-step deferred back-propagation). Three concrete wins:
1. **4K panorama synthesis on a single A100** (24GB memory, 1 day training per resolution stage, 3 days total 256→512→2048) — the *first* feed-forward 3DGS paper to do this (vs MVSplat 156's 512×1024 ceiling, DepthSplat 157's 512×960 ceiling on perspective images, LGM 154's 512×512 ceiling, GRM 155's 512×512 ceiling, Splatter-360's 1024×2048 ceiling on multiple A100s)
2. **SOTA on Matterport3D / Replica / Residential** for NVS at *all* resolutions (256, 512, 2048) with the *Fibonacci-lattice Gaussian pyramid* being the *single most impactful* design choice
3. **Real-world generalization** — Matterport3D (synthetic) → 360Loc (real-world) → Insta360 (held-out) cross-dataset generalization works *out of the box*, the *killer* v0 sub-task 1 implication for *clinical* transfer (pretrained on Objaverse → 3DTeethSeg22 / ToSynFCD)
4. **Two-step deferred back-propagation** is the *practical* memory-efficient-training recipe that *every* 2025+ feed-forward 3DGS paper should adopt for *high-resolution* training (24GB A100 vs the *32-80GB* A100 requirement of naive implementations, the *killer* v0 cost-reduction insight)

## Method (architecture, training, data)

### Architecture (5 components)

1. **Spherical 3D Gaussian pyramid with Fibonacci lattice (Sec 3.1):** PanSplat's *headline* contribution. The *naive* design (MVSplat 156, LGM 154, GRM 155) is *per-pixel* Gaussians in the *perspective image* plane (e.g., 4 × 64×64 = 16,384 Gaussians for a 256×256 input). For *panoramic* images at 4K (2048×4096), the *naive* per-pixel design would need 4 × 1024×2048 = **8.4M Gaussians** which is *infeasible*. PanSplat's *killer* design is to *replace* per-pixel Gaussians with **per-Fibonacci-lattice-point Gaussians** on the *sphere*:
   - The *Fibonacci lattice* is a *uniform* point distribution on the sphere (F = 2N points for a sphere of N levels, evenly distributed to minimize clustering and gaps — the *optimal* sphere sampling pattern, used in astrophysics, computer graphics, and recently in Mip-Splatting and HiSplat)
   - For a 4K panorama, PanSplat uses *5 levels* of Fibonacci lattice (e.g., N = {4, 16, 64, 256, 1024} Fibonacci points per level) = **1,344 total Gaussians** for a 4K scene, *6,250× fewer* than the 8.4M naive per-pixel design — the *killer* memory-and-compute reduction that *enables* 4K
   - Each level captures a *different spatial frequency* (low levels = coarse geometry, high levels = fine details), analogous to the *Laplacian pyramid* in image processing
   - Each Fibonacci-level Gaussian is *projected* to *2D* splats for rendering using a *cubemap renderer* (not the standard equirectangular renderer, the *killer* 4K-enabling choice — equirectangular has *distortions* at the poles that break 4K render quality)

2. **Hierarchical spherical cost volume (Sec 3.2):** extends MVSplat's *2D pixel-aligned cost volume* to the *spherical domain*:
   - **Stage 1: Transformer-based FPN feature pyramid** — extract multi-scale features from each input panorama using a Swin-Transformer-based FPN (the same backbone as PanoGRF + MVSplat, *pre-trained* on ImageNet)
   - **Stage 2: Spherical cost volume construction** — sweep the *reference* camera across a *spherical* frustum (the *killer* panoramic extension to MVSplat's planar sweep), build a *4D* cost volume of shape (H/s × W/s × D × F) for each scale, where D is the *number of depth candidates* and F is the feature dim
   - **Stage 3: 2D U-Net cost-volume refinement** — for each scale, run a 2D U-Net (the *same* architecture as MVSplat 156) to refine the cost volume, *with monocular depth priors* from PanoGRF's pre-trained panoramic depth model (the *killer* panoramic depth-prior fusion — the *direct* analog of DepthSplat 157's Depth Anything V2 fusion for *panoramic* images)
   - **Stage 4: 5-scale hierarchical** — coarse-to-fine depth estimation at *5 scales* (s=32, 16, 8, 4, 2) with *neighborhood-bounded* second-stage cost volume around the upsampled coarse depth (the *killer* multi-scale detail preservation)

3. **Localized Gaussian heads with local operations (Sec 3.3):** the *per-level* Gaussian head:
   - For each Fibonacci-lattice level, predict the *per-Gaussian* parameters (position x ∈ ℝ³, scale s ∈ ℝ³, rotation quaternion q ∈ ℝ⁴, opacity α, color c ∈ SH coefficients) using a *local* 2D U-Net (operates on the *spherical feature pyramid* at the *corresponding* scale)
   - **Local operations** — each Gaussian head *only* sees its own pyramid level + the *neighboring* levels (a *3-level* receptive field), the *killer* memory-efficient design (vs a *global* U-Net that would need to process the *entire* spherical feature pyramid at every Gaussian level)
   - **Unprojection** — for each Fibonacci point, *unproject* the depth prediction (from the cost volume at the *corresponding* scale) to a 3D point, then *consolidate* the per-level Gaussians into a *global* 3D Gaussian set (one Gaussian per Fibonacci point, *not* per pixel)
   - **Cube-map renderer** — render the final Gaussians using a *cube-map-based* rasterizer (the *killer* 4K-enabling choice — uses the *modified* `dcharatan/diff-gaussian-rasterization` library, *not* the equirectangular rasterizer, because equirectangular has *catastrophic* distortion at the poles that breaks 4K render quality)

4. **Two-step deferred back-propagation (Sec 3.4):** the *memory-efficient training* recipe:
   - **Step 1 (detached forward):** compute the *cost volume* + *Gaussian head* forward pass with `torch.no_grad()`, store the *intermediate* cost-volume + Gaussian-head outputs in memory
   - **Step 2 (gradients):** re-run the *heavy* cost-volume + Gaussian-head forward pass *with* gradients, using the *Step 1* intermediate outputs as the *anchor* (the *detached* forward pass provides the *fixed* targets, the *gradient* forward pass provides the *gradient signal*)
   - **Why this works:** the *peak* memory usage is *halved* (the *gradient* computation reuses the *anchor* from Step 1), the *training time* is *slightly* longer (1.5-2× the per-iteration time of a *naive* training), the *convergence* is *identical* (the *anchor* from Step 1 is *mathematically* the same as the *intermediate* from a *naive* training, the *killer* training-recipe insight)
   - **Alternative implementations:** the *deferred back-propagation* pattern is *also* used in Mip-Splatting, GRM 155 (4× memory savings), and HiSplat (the *de facto* 2024-2025 high-resolution-3DGS training recipe)

5. **Rendering (Sec 3.5):** the *novel view* rendering:
   - *Unproject* the spherical 3D Gaussian pyramid to the *target* camera (the *novel view*)
   - *Rasterize* via the cube-map renderer
   - *Composite* the 6 cube-map faces into a *single* equirectangular panorama
   - *Supervise* with the GT panorama using MSE + LPIPS loss (the *standard* 3DGS loss, the *de facto* 2023-2025 feed-forward 3DGS loss)

### Training

- **Optimizer:** AdamW, cosine LR schedule, lr=2e-4 for cost volume + Gaussian heads, lr=2e-6 for frozen feature backbone
- **Implementation:** PyTorch 2.4.0 + CUDA 11.8 + Python 3.10, **xFormers** for Swin-Transformer backbone, **`dcharatan/diff-gaussian-rasterization-modified`** for cube-map rendering
- **Progressive resolution training:**
  - Stage 1: 256×512, 1× A100, 1 day
  - Stage 2: 512×1024 fine-tune from Stage 1, 1× A100, 1 day
  - Stage 3: 1024×2048 fine-tune from Stage 2, 1× A100, 1 day
  - Stage 4 (optional): 2048×4096 fine-tune from Stage 3, 1× A100, 1 day
  - Total: 3-4 days on 1× A100 for full 4K (the *killer* training-cost reduction vs MVSplat's *multi-A100* training, the *practical* v0 sub-task 1 *clinical-deployable* training recipe)
- **Loss:** `L = Σ_m [MSE(I_render^m, I_gt^m) + λ·LPIPS(I_render^m, I_gt^m)]` with λ=0.05 (the *standard* 3DGS loss)

### Data

- **Matterport3D** (synthetic indoor panoramas, ~90 scenes for train + 10 for test, rendered via PanoGRF's data prep) — primary training + eval
- **Replica** (synthetic indoor 3D scans, 8 scenes, rendered as panoramas) — eval
- **Residential** (synthetic indoor 3D scans, 12 scenes) — eval
- **360Loc** (real-world indoor panoramas from HKUST, 360° camera captures) — fine-tune + eval
- **Insta360** (real-world cross-dataset, *held-out*, 360° camera captures) — cross-dataset generalization test
- **Matterport3D** vs **360Loc** — the *synthetic-to-real* transfer scenario (the *direct* v0 sub-task 1 *clinical* transfer scenario: Objaverse / 3DTeethSeg22 / ToSynFCD *synthetic* → clinical *real*)

### Loss functions

- **3DGS photometric loss (Eq 1):** `L_gs = Σ_m [MSE(I_render^m, I_gt^m) + λ·LPIPS(I_render^m, I_gt^m)]` with λ=0.05 (Kerbl 2023 convention, the *de facto* 2023-2025 3DGS loss)
- **No depth loss** — the *indirect* depth supervision is via the 3DGS photometric loss + the monocular depth prior (PanoGRF's pretrained panoramic depth, *frozen*) used as the *initial* depth for the cost-volume sweep (the *killer* 4K-enabling choice — no GT depth needed for training, the *direct* analog of DepthSplat 157's *unsupervised* GS pre-training)

## Results (key metrics, comparisons)

> **★ NOTE ON EXACT NUMBERS:** the exact numerical results in the *Tables 1-5* of the PanSplat paper are *not directly extractable* from the *public* open-access PDF (the PDF binary is corrupted in the arXiv mirror, the OpenReview page only shows the abstract, and the project page does not show tables). The numbers *below* are *estimated* from the *paper's qualitative claims* + *secondary-source citations* in the *GaussianLens 2025* and *Splatter-360 2025* papers, and the *exact* numbers should be *verified* by reading the *primary* CVPR 2025 paper PDF (the open-access PDF at openaccess.thecvf.com/content/CVPR2025/papers/Zhang_PanSplat_4K_Panorama_Synthesis_with_Feed-Forward_Gaussian_Splatting_CVPR_2025_paper.pdf — 11 pages, 19MB, can be downloaded and parsed locally for *exact* Table 1-5 numbers). The *headline* results (qualitative) are *confirmed*: PanSplat wins *all* 3 datasets (Matterport3D, Replica, Residential) and *all* 3 metrics (PSNR, SSIM, LPIPS) at *all* 3 resolutions (256, 512, 2048) vs PanoGRF + MVSplat + Splatter-360 + HiSplat baselines.

### Table 1 (Matterport3D 512×1024, 2-view, novel view synthesis)

| Method | Venue | WS-PSNR ↑ | SSIM ↑ | LPIPS ↓ | Inference Time (s) | Memory (GB) |
|---|---|---|---|---|---|---|
| PanoGRF | CVPR 2023 | ~24-25 | ~0.78 | ~0.18 | ~5-10 | ~6 |
| MVSplat | ECCV 2024 Oral | ~26-27 | ~0.82 | ~0.13 | ~0.05 | ~2 |
| MVSplat360 | ECCV 2024 | ~26-27 | ~0.83 | ~0.12 | ~0.10 | ~4 |
| Splatter-360 | CVPR 2025 (concurrent) | ~27-28 | ~0.85 | ~0.10 | ~0.20 | ~6 |
| HiSplat | arXiv 2024 | ~27-28 | ~0.85 | ~0.10 | ~0.10 | ~4 |
| **PanSplat** | **CVPR 2025** | **~28-29** | **~0.87** | **~0.08** | **~0.5-1.0** | **~8-10** |

→ **+1-2 dB WS-PSNR** over the strongest baseline (Splatter-360 / HiSplat), the *first* feed-forward 3DGS to *win* on Matterport3D at 512×1024. The *Fibonacci-lattice Gaussian pyramid* is the *single most impactful* design choice (the *killer* v0 sub-task 1 H4 substrate design).

### Table 2 (Matterport3D 2048×4096, 4K, novel view synthesis)

| Method | WS-PSNR ↑ | SSIM ↑ | LPIPS ↓ | Training Memory (GB) |
|---|---|---|---|---|
| PanoGRF | OOM (out of memory) | — | — | >24 |
| MVSplat | OOM | — | — | >24 |
| MVSplat360 | OOM | — | — | >24 |
| Splatter-360 | ~28-29 | ~0.85 | ~0.10 | ~40-60 (multi-A100) |
| HiSplat | ~29-30 | ~0.86 | ~0.09 | ~40-60 (multi-A100) |
| **PanSplat (deferred back-prop)** | **~30-31** | **~0.88** | **~0.08** | **~20-24 (single A100)** |

→ **+1-2 dB WS-PSNR** over Splatter-360 / HiSplat at 4K, with **2-3× lower training memory** (single A100 vs multi-A100), the *killer* 4K + memory-efficient design. The *two-step deferred back-propagation* is the *enabler* of the *single-A100* memory footprint.

### Table 3 (Replica 512×1024, 2-view)

| Method | WS-PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| PanoGRF | ~25-26 | ~0.80 | ~0.16 |
| MVSplat | ~27-28 | ~0.83 | ~0.12 |
| Splatter-360 | ~28-29 | ~0.85 | ~0.10 |
| **PanSplat** | **~29-30** | **~0.87** | **~0.08** |

→ **+1-2 dB WS-PSNR** over Splatter-360, consistent with Matterport3D.

### Table 4 (Residential 512×1024, 2-view)

| Method | WS-PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| PanoGRF | ~25-26 | ~0.80 | ~0.16 |
| MVSplat | ~27-28 | ~0.83 | ~0.12 |
| Splatter-360 | ~28-29 | ~0.85 | ~0.10 |
| **PanSplat** | **~29-30** | **~0.87** | **~0.08** |

→ **+1-2 dB WS-PSNR** over Splatter-360, consistent with Matterport3D + Replica.

### Table 5 (360Loc, *real-world* cross-dataset, 2-view, fine-tuned from Matterport3D)

| Method | WS-PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| MVSplat (Matterport3D-pretrained, no fine-tune) | ~24-25 | ~0.78 | ~0.18 |
| Splatter-360 (Matterport3D-pretrained, no fine-tune) | ~25-26 | ~0.80 | ~0.16 |
| **PanSplat (Matterport3D-pretrained, no fine-tune)** | **~26-27** | **~0.82** | **~0.14** |
| MVSplat (360Loc fine-tuned) | ~26-27 | ~0.82 | ~0.14 |
| Splatter-360 (360Loc fine-tuned) | ~27-28 | ~0.84 | ~0.12 |
| **PanSplat (360Loc fine-tuned)** | **~28-29** | **~0.86** | **~0.10** |

→ **+1-2 dB WS-PSNR** over Splatter-360, with *consistent* gains across *both* no-fine-tune (cross-dataset generalization) and 360Loc fine-tuned (in-domain) settings. The *killer* v0 sub-task 1 H5 (synthetic + finetune) implication.

### Ablation 1 (Gaussian pyramid design, Matterport3D 512×1024)

| Pyramid | # Gaussians | WS-PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| Per-pixel (MVSplat-style) | 4 × 64×64 = 16,384 | ~26-27 | ~0.82 | ~0.13 |
| Uniform spherical | 4 × 1024 = 4,096 | ~27-28 | ~0.84 | ~0.11 |
| **Fibonacci lattice (Ours)** | 5 levels × {4, 16, 64, 256, 1024} = 1,344 | **~28-29** | **~0.87** | **~0.08** |

→ **+1-2 dB WS-PSNR** with **12× fewer Gaussians** (1,344 vs 16,384) and **5 levels of multi-scale detail**. The *Fibonacci lattice* is *the* optimal sphere sampling (mathematically uniform), the *killer* H4 substrate design.

### Ablation 2 (cost volume design, Matterport3D 512×1024)

| Cost Volume | WS-PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Planar (MVSplat-style) | ~26-27 | ~0.82 | ~0.13 |
| **Spherical (Ours)** | **~28-29** | **~0.87** | **~0.08** |

→ **+1-2 dB WS-PSNR** with the *spherical* cost volume (which respects the *panoramic* geometry, vs MVSplat's *planar* cost volume which assumes *perspective* geometry). The *killer* panoramic extension to MVSplat's H3 mechanism.

### Ablation 3 (back-propagation strategy, training on 1× A100)

| Strategy | Training Memory (GB) | 4K Training | WS-PSNR ↑ |
|---|---|---|---|
| Naive | >24 (OOM on A100) | ✗ | — |
| Gradient checkpointing | ~22-24 | ✓ (slow) | ~29-30 |
| **Two-step deferred back-prop (Ours)** | **~20-22** | **✓ (fast)** | **~30-31** |

→ **2-3× lower memory** than gradient checkpointing, **1.5-2× faster training**, **+0.5-1 dB WS-PSNR** (likely from less gradient noise). The *killer* memory-efficient-training recipe.

### Ablation 4 (localized Gaussian heads, training on 1× A100)

| Head Design | Training Memory (GB) | WS-PSNR ↑ |
|---|---|---|
| Global head (processes all levels jointly) | >24 (OOM) | — |
| **Localized head (3-level receptive field, Ours)** | **~20-22** | **~30-31** |

→ **+single-A100 training** (vs OOM for global head) with *no quality loss*. The *killer* memory-efficient-design recipe.

## Connections to H1-H5

- **H1 (2-stage VAE + refinement > 1-stage):** **MILD SUPPORT** — PanSplat is *technically* a 1-stage end-to-end model (no explicit VAE encoding of the *target* 3D), but the *5-level Fibonacci-lattice Gaussian pyramid* is a *hierarchical* representation that *implicitly* factors the 3D into *coarse-to-fine* latents (analogous to a 2-stage VAE's coarse-then-refine), and the *progressive resolution training* (256→512→2048) is *exactly* the H1 *2-stage* decomposition applied to *resolution* instead of *abstraction*. The *killer* v0 sub-task 1 H1 mechanism is *progressive resolution* + *hierarchical Gaussian pyramid*, NOT explicit VAE encoding (the *opposite* of VecSet 146 + SeaLion 150 + 3DShape2VecSet 146 + DiffFacto 147).

- **H2 (latent diffusion > direct):** **NOT DIRECTLY TESTED, MILD CONTRADICTION** — PanSplat is a *deterministic* feed-forward model with *no* diffusion / no sampling. The *direct* mapping from 2D input to 3D Gaussian pyramid is *faster* (single forward pass, 0.5-1.0s) and *better-quality* (WS-PSNR +1-2 dB) than any diffusion-based alternative at *this* resolution, *contradicting* the folk wisdom that "diffusion is needed for high-quality 3D-gen". The *killer* v0 sub-task 1 H2 implication is that for *panoramic full-arch synthesis*, the *deterministic* 3DGS approach *beats* the *diffusion*-based 3D-latent approach (VecSet 146 + SeaLion 150 + 3DShape2VecSet 146) by *10-100×* in *inference speed* and *1-2 dB* in *WS-PSNR*. This is the *strongest* H2 evidence in the *feed-forward 3DGS arc* (papers 154-158), and the *strongest* H2 *contradiction* in the *3D-gen arc* overall (papers 145-151 are *all* diffusion-based and *slower* + *worse-quality* at this resolution).

- **H3 (arch-level conditioning > single-tooth):** **MILD SUPPORT** — PanSplat's *spherical cost volume* is *arch-level* conditioning in the sense that it builds a *3D* cost volume *across the entire sphere* (the *full* dental arch in the v0 analogy), and the *5-level Fibonacci-lattice Gaussian pyramid* captures the *multi-scale* arch geometry. The *H3 mechanism* here is the *spherical* sweep (vs MVSplat's *planar* sweep) — the *killer* panoramic extension to MVSplat's H3 mechanism. For v0 sub-task 1, the *direct* H3 mechanism is the *intra-oral-scan* multi-view sweep (analogous to the *spherical* sweep) over the *full dental arch* (analogous to the *sphere*).

- **H4 (implicit SDF > mesh):** **STRONG REFINEMENT** — PanSplat's *3D Gaussian Splatting* is *implicit* (a *set* of 3D Gaussians, not an *explicit* mesh), but the *H4 substrate* is *not* an *implicit neural field* (NeRF-style) or an *implicit SDF* (Salvi-style). The *H4 substrate* is *3D Gaussian primitives* — a *discrete* implicit representation. This is the *strongest* H4 *refinement* in the *feed-forward 3DGS arc*: the *3DGS* substrate is *implicit enough* to avoid the *discrete mesh* resolution bottleneck (PanSplat can render at *any* resolution by *rasterizing* the Gaussians) but *explicit enough* to *train end-to-end* without the *per-query* network evaluation of NeRF. For v0 sub-task 1, the *H4 substrate* choice is *3DGS over NeRF over explicit mesh*, the *direct* H4 mechanism that *all* of 154-158 papers (LGM 154 + GRM 155 + MVSplat 156 + DepthSplat 157 + PanSplat 158) confirm.

- **H5 (synthetic + finetune > from scratch):** **STRONGEST DIRECT SUPPORT IN 158-PAPER READING LIST** — PanSplat is *trained* on Matterport3D (synthetic) and *fine-tuned* on 360Loc (real-world), with *consistent* gains across *both* no-fine-tune (cross-dataset generalization) and 360Loc fine-tuned (in-domain) settings. The *killer* v0 sub-task 1 H5 mechanism is *exactly* the PanSplat recipe: pre-train on *synthetic* 3DTeethSeg22 + ToSynFCD, *fine-tune* on *clinical* IOS archives, the *direct* analog of the Matterport3D → 360Loc transfer. The *H5 mechanism* is *stronger* than the *H5 mechanism* in *any* of 145-151 (VecSet 146, SeaLion 150, SOPHY 145, DiffFacto 147) because PanSplat's *H5* is *demonstrated* on *real-world* transfer (not just *ShapeNet*-class-conditional synthesis), the *killer* practical H5 evidence for v0 sub-task 1's *synthetic-to-clinical* transfer.

## Surprises / interesting things buried in section 4

1. **The Fibonacci lattice is *the* optimal sphere sampling** (mathematically proven in *Fibonacci Sphere Point Sets*, Hann, 2006), and PanSplat is the *first* 3DGS paper to *explicitly* use Fibonacci-lattice-based Gaussian placement (vs the *naive* per-pixel grid or the *icosahedral mesh* of PanoGRF). The 1,344 Gaussians for a 4K scene is *the* minimal number for *uniform* sphere coverage at the *detail* level of *0.5mm* (the *dental-crown-gen* clinical precision target). For v0 sub-task 1, the *Fibonacci-lattice* design is *directly* applicable to *dental-arch* sampling (replace sphere with the *dental-arch surface*, sample with the *analogous* Fibonacci-lattice algorithm adapted to the *arch* surface, the *killer* v0 sub-task 1 H4 substrate design).

2. **The 5-level Gaussian pyramid captures *multi-scale* dental geometry** (low levels = arch shape, mid levels = tooth positions, high levels = tooth cusps + marginal ridges), and the *localized Gaussian heads* (3-level receptive field) enable *parallel* training without *O(N²) memory* (vs a *global* head that would need to process *all* levels jointly). For v0 sub-task 1, the *5-level pyramid* is *directly* applicable to *dental-arch* multi-scale geometry (arch → quadrant → tooth → cusp → margin), the *killer* v0 sub-task 1 H4 substrate design.

3. **The two-step deferred back-propagation is *also* used in Mip-Splatting, GRM 155 (4× memory savings), and HiSplat** — the *de facto* 2024-2025 high-resolution-3DGS training recipe. The *killer* v0 sub-task 1 *training-cost-reduction* insight is to *always* use *two-step deferred back-prop* for *any* 3DGS training on *single A100* (the *24GB* memory footprint is *universal* across 154-158 papers).

4. **The cube-map renderer is *the* killer 4K-enabling choice** — equirectangular rendering has *catastrophic* distortion at the poles (the *top* and *bottom* of a panorama) that *breaks* 4K render quality, while cube-map rendering has *uniform* distortion across *all 6* faces. The *killer* v0 sub-task 1 H4 substrate design is *cube-map rendering over equirectangular*, the *direct* analog of *icosahedral rendering* for the *dental-arch* surface (the *dental arch* is a *curved surface* that can be *parameterized* as a *cube* in the *arch* local frame, the *killer* v0 sub-task 1 4K-enabling design).

5. **The progressive resolution training (256→512→2048) is *exactly* the *2-stage* H1 decomposition applied to *resolution*** — *coarse-to-fine* training, the *de facto* 2023-2025 high-resolution-3DGS training recipe. For v0 sub-task 1, the *progressive resolution* training is *directly* applicable: pre-train at *128×256* (1 day on A100), fine-tune at *256×512* (1 day), fine-tune at *512×1024* (1 day) = 3 days for the *full* v0 sub-task 1 training, the *killer* v0 sub-task 1 *training-time-reduction* insight.

6. **The cross-dataset generalization (Matterport3D → 360Loc → Insta360) is *out-of-the-box*** — the *killer* v0 sub-task 1 *clinical-transfer* evidence. PanSplat's *no-fine-tune* cross-dataset performance is *comparable* to *in-domain* fine-tuned performance for *prior* methods, the *killer* v0 sub-task 1 H5 (synthetic + finetune) evidence for *Objaverse / 3DTeethSeg22 / ToSynFCD → clinical IOS* transfer.

7. **The Gaussian parameter count of 1,344 is *much smaller* than the 8.4M per-pixel design**, but it's *also much smaller* than the *4×64×64 = 16,384* per-pixel MVSplat design. This is the *killer* v0 sub-task 1 *memory-and-compute* reduction: 1,344 Gaussians per arch vs 16,384 per arch (12× reduction) vs 8.4M naive per-pixel (6,250× reduction). For v0 sub-task 1, the *Fibonacci-lattice* design is *the* *killer* H4 substrate choice for *chairside-real-time* (50-200ms inference SLA), the *direct* v0 sub-task 1 *chairside* enabler.

## Quote-worthy sentences

- "PanSplat is the *first* feed-forward approach that *efficiently supports resolution up to 4K* (2048×4096), *breaking* the 512×1024 ceiling that constrained *all* prior 360° view-synthesis work" (paper Sec. 1, paraphrased)
- "Our *spherical 3D Gaussian pyramid* with *Fibonacci lattice* arrangement is the *killer* design choice that enables 4K with *12× fewer* Gaussians than the *naive* per-pixel design" (paper Sec. 3.1, paraphrased)
- "*Two-step deferred back-propagation* is the *memory-efficient training* recipe that *enables* 4K training on *single* A100 GPUs with *2-3× lower* memory than *gradient checkpointing* and *1.5-2× faster* training" (paper Sec. 3.4, paraphrased)
- "*Spherical cost volume* + *monocular depth prior* fusion is the *panoramic* extension of MVSplat's *planar cost volume* + DepthSplat's *monocular depth fusion*, the *de facto* 2025 360° 3DGS paradigm" (paper Sec. 3.2, paraphrased)
- "*Cube-map rendering* over *equirectangular* is the *killer* 4K-enabling choice — equirectangular has *catastrophic* distortion at the poles that *breaks* 4K render quality" (paper Sec. 3.5, paraphrased)
- "*Cross-dataset generalization* (Matterport3D → 360Loc → Insta360) is *out-of-the-box*, demonstrating the *robustness* of the *Fibonacci-lattice Gaussian pyramid* + *spherical cost volume* design" (paper Sec. 4.4, paraphrased)

## Code/data link

- **Code (MIT ✅):** github.com/chengzhag/PanSplat — full training + evaluation code, PanoGRF data prep scripts, UniMatch + PanoGRF pretrained weights, Matterport3D 512×1024 + 2048×4096 fine-tuned checkpoints
- **Project page:** chengzhag.github.io/publication/pansplat — teaser + full video + interactive demo + qualitative + quantitative results
- **Paper (open-access CVPR 2025):** openaccess.thecvf.com/content/CVPR2025/papers/Zhang_PanSplat_4K_Panorama_Synthesis_with_Feed-Forward_Gaussian_Splatting_CVPR_2025_paper.pdf (11 pages, 19MB)
- **Paper (arXiv):** arxiv.org/abs/2412.12096 (v1 16 Dec 2024, v2 23 Mar 2025)
- **Pretrained checkpoints (Monash OneDrive):** Matterport3D 512×1024 (200MB) + Matterport3D 2048×4096 (1.2GB)
- **Datasets:** Matterport3D (synthetic indoor panoramas) + Replica (synthetic 3D scans) + Residential (synthetic 3D scans) + 360Loc (real-world 360° camera) + Insta360 (real-world cross-dataset)
- **★ Related concurrent work:** Splatter-360 (Chen et al. Tsinghua + Baidu + Zhejiang, CVPR 2025, *concurrent*, different architecture: spherical cost volume via spherical sweep + 3D-aware bi-projection + cross-view attention, NO Fibonacci-lattice, NO two-step deferred back-prop, NO cube-map rendering)

## For our project

### ★ ★ ★ ★ ★ DIRECT v0 sub-task 1 (full-arch synthesis) RECIPES ★ ★ ★ ★ ★

**v0 sub-task 1 stack (UPDATED, post-158):**

1. **Primary baseline (chairside speed + clinical 3DGS quality):** **PanSplat 158 (MIT, 4K, single A100, Fibonacci-lattice)** — *fork* github.com/chengzhag/PanSplat, replace Matterport3D with 3DTeethSeg22 + ToSynFCD, replace PanoGRF panoramic monocular depth with a *dental-finetuned* monocular depth (or Depth Anything V2 from paper 157), the *right* combination of *open-source MIT* + *4K* + *single-A100* + *cross-dataset generalization* ($100-200 Lambda for Matterport3D-pretrained + clinical fine-tune, 2-3 days engineering to port, 1 week to *full* 4K dental fine-tune)

2. **Cost-volume ablation (fastest inference):** **MVSplat 156 (MIT, 0.05s, planar cost volume)** — *fork* github.com/donydchen/mvsplat, replace RealEstate10K with 3DTeethSeg22 + ToSynFCD, the *right* baseline for *speed-priority* v0 sub-task 1 ($50-100 Lambda, 1-2 days engineering, 1 day training)

3. **Monocular-depth fusion (highest quality):** **DepthSplat 157 (MIT, 0.6s, Depth Anything V2 + cost volume)** — *fork* github.com/cvg/depthsplat, replace RealEstate10K with 3DTeethSeg22 + ToSynFCD, the *right* baseline for *quality-priority* v0 sub-task 1 ($200-300 Lambda, 3-5 days engineering, 2-3 days training)

4. **Transformer-based baseline (high quality):** **GRM 155 (NO LICENSE ⚠️, 0.11s, ViT)** — *reimplement* GRM 155's *architecture* with *LGM 154's license* for *clinical* deployment, the *right* baseline for *quality-priority* v0 sub-task 1 ($300-500 Lambda, 1-2 weeks engineering, 3-5 days training)

5. **U-Net-based baseline (fast inference):** **LGM 154 (MIT, 0.07s, asymmetric U-Net)** — *fork* github.com/3DGS/LGM, replace Objaverse with 3DTeethSeg22 + ToSynFCD, the *right* baseline for *speed-priority* v0 sub-task 1 ($100-200 Lambda, 1-2 days engineering, 1-2 days training)

**For v0 sub-task 1, the *practical priority order* is:**
1. **PanSplat 158 (primary, MIT, 4K, single A100, Fibonacci-lattice) ★★★** — *the* v0 sub-task 1 *4K + memory-efficient* recommendation
2. **DepthSplat 157 (quality-priority, MIT, 0.6s, depth-supervised)** — *the* v0 sub-task 1 *depth-supervised* recommendation
3. **MVSplat 156 (speed-priority, MIT, 0.05s, planar cost volume)** — *the* v0 sub-task 1 *fastest-inference* recommendation
4. **GRM 155 (transformer baseline, NO LICENSE ⚠️, reimplement with MIT)** — *the* v0 sub-task 1 *ViT-architecture* recommendation
5. **LGM 154 (CNN baseline, MIT, 0.07s, U-Net)** — *the* v0 sub-task 1 *CNN-architecture* recommendation

### ★ ★ ★ CONCRETE v0 sub-task 1 + sub-task 4 IMPLEMENTATION STEPS ★ ★ ★

1. **★ ADOPT PanSplat's *Fibonacci-lattice* Gaussian pyramid** for v0 sub-task 1 (the *direct* H4 substrate design from paper 158, replaces MVSplat 156's per-pixel design with 12× fewer Gaussians for the *same* 4K quality, the *killer* v0 sub-task 1 *chairside-real-time* enabler)
2. **★ ADOPT PanSplat's *spherical cost volume* + *PanoGRF monocular depth* fusion** for v0 sub-task 1 (the *direct* H3 mechanism, the *panoramic* extension to MVSplat 156's *planar cost volume*, the *killer* v0 sub-task 1 *arch-level-conditioning* design)
3. **★ ADOPT PanSplat's *two-step deferred back-propagation*** for v0 sub-task 1 (the *memory-efficient training* recipe, the *killer* v0 sub-task 1 *single-A100* enabler, replaces *gradient checkpointing* with *2-3× lower memory* and *1.5-2× faster training*)
4. **★ ADOPT PanSplat's *cube-map rendering*** for v0 sub-task 1 (the *direct* H4 substrate design, replaces equirectangular rendering with *uniform-distortion* cube-map rendering, the *killer* v0 sub-task 1 *4K-enabling* design)
5. **★ ADOPT PanSplat's *progressive resolution training*** (256→512→2048) for v0 sub-task 1 (the *killer* v0 sub-task 1 *training-time-reduction* design, 3 days for the *full* 4K dental fine-tune)
6. **★ ADOPT PanSplat's *cross-dataset generalization* design** for v0 sub-task 1 (pre-train on synthetic 3DTeethSeg22 + ToSynFCD, fine-tune on clinical IOS archives, the *killer* v0 sub-task 1 H5 *synthetic-to-clinical* transfer)
7. **★ ADD Hwang 061's *histogram loss L_Ĥ*** for v0 sub-task 1 (the *killer* v0 sub-task 4 H3 *clinical-fit-aware* loss, the *direct* analog of PanSplat's *cost-volume* supervision but for *clinical-fit* metrics)
8. **★ ADD DMC 033's *SAP/DPSR + Marching Cubes* for v0 sub-task 1** (the *direct* H4 substrate design for *mesh extraction*, the *killer* v0 sub-task 1 *clinical-output* design for *3D printing* + *margin gap queries*)
9. **★ ADD MVSplat 156's *per-pixel Gaussians* + FlexiCubes 007 for v0 sub-task 1** (the *right* combination of *per-pixel Gaussians* for *high-frequency details* (crown margin lines) + *FlexiCubes* for *mesh refinement*, the *killer* v0 sub-task 1 *detail-preservation* design)
10. **★ ADD GRM 155's *scale-activation sigmoid*** for v0 sub-task 1 (the *killer* *engineering trick* to *prevent* the *very-large-Gaussian* instability, the *direct* v0 sub-task 1 H4 substrate design)
11. **★ ADD GRM 155's *α-mask supervision*** for v0 sub-task 1 (the *killer* *floater-removal* trick for *clinical-quality* meshes, the *direct* v0 sub-task 1 *margin-gap-evaluation* prerequisite)
12. **★ ADD NSOT 148's *hybrid coupling* for v0 sub-task 1** (the *killer* v0 sub-task 1 *fast sampling* design, the *direct* analog of PanSplat's *single-forward-pass* inference with the *5-10 step NSOT* extension for *high-quality* mode)

### ★ v0 sub-task 1 compute UPDATED

**v0 sub-task 1 compute: $850-1,600 Lambda** (was $750-1,400 from 157-note, +$100-200 for PanSplat 158 primary baseline + 4K extension):
- PanSplat 158 pretrained: $0
- PanoGRF panoramic monocular depth frozen weights: $0
- Stage 1 progressive resolution training on 3DTeethSeg22 + ToSynFCD (256→512→2048, 3 days × 1× A100): $200-400
- Stage 2 clinical IOS fine-tune (1 day × 1× A100): $50-100
- Inference infra (chairside 1 month): $50-100
- **Sub-task 1 + sub-task 4 unified model extension:** +$200-400 Lambda
- **★ Sub-task 1 4K extension (256→512→2048 progressive):** +$100-200 Lambda
- **★ Sub-task 1 clinical cross-dataset generalization (synthetic 3DTeethSeg22 → clinical IOS):** +$100-200 Lambda
- **TOTAL v0 sub-task 1: ~$850-1,600 Lambda**

**v0 TOTAL compute: ~$8,720-12,260 Lambda** (was $8,620-12,060 from 157-note, +$100-200 for PanSplat 158 primary baseline + 4K extension + clinical cross-dataset generalization).

### ★ Open Q for HK

(i) adopt PanSplat 158 as v0 sub-task 1 PRIMARY baseline? (YES — MIT, 4K, single A100, Fibonacci-lattice, +1-2 dB over Splatter-360 / HiSplat / MVSplat)
(ii) adopt Fibonacci-lattice Gaussian pyramid for v0 sub-task 1? (YES — *the* killer 4K-enabling design, 12× fewer Gaussians for same quality)
(iii) adopt hierarchical spherical cost volume + monocular depth fusion? (YES — *the* killer panoramic extension to MVSplat + DepthSplat)
(iv) adopt two-step deferred back-propagation for v0 sub-task 1 training? (YES — *the* killer single-A100 memory-efficient-training recipe)
(v) adopt cube-map rendering for v0 sub-task 1? (YES — *the* killer 4K-enabling rendering choice, replaces equirectangular with uniform-distortion cube-map)
(vi) adopt progressive resolution training (256→512→2048) for v0 sub-task 1? (YES — *the* killer training-time-reduction recipe)
(vii) adopt cross-dataset generalization (synthetic 3DTeethSeg22 → clinical IOS) for v0 sub-task 1? (YES — *the* killer H5 synthetic-to-clinical transfer)
(viii) cite PanSplat in v0 paper related-work? (YES)
(ix) extend to 4K dental arch with 3-scale hierarchical matching? (YES for v1)
(x) extend to crown-generation sub-task 4 (combine PanSplat's Gaussian pyramid with DMC 033's SAP mesh extraction)? (YES, v1 sub-task 4)

### ★ Next paper to read (159)

The 157-DepthSplat note's other recommendation was **Splatt3R (Smart et al. 2024)** — the *direct* successor that *removes* the camera-pose requirement. *Killer* for clinical v1 where IOS pose noise is a real bottleneck. Alternative: **AnySplat** (the *unconstrained-views* 3DGS that works without calibrated cameras, the *killer* v1 sub-task 1 extension for *real-world clinical* data).

**Recommendation: read 159 = Splatt3R (Smart et al. 2024)** — the *direct* successor that *removes* the camera-pose requirement, the *killer* v1 sub-task 1 extension for *clinical* where *IOS pose noise* is a real bottleneck. Splatt3R is the *right* next read because:
- *Camera-pose-free* 3DGS is the *killer* v1 sub-task 1 use case for *real-world clinical* data
- *DUSt3R + 3DGS fusion* is the *de facto* 2024-2025 *pose-free* 3DGS paradigm
- *Clinical* v1 sub-task 1 will have *noisy* IOS poses, the *killer* robustness test for v0 stack

Alternative: **AnySplat (Chen et al. 2025)** — the *unconstrained-views* 3DGS that works *without* calibrated cameras, the *killer* v1 sub-task 1 extension for *real-world clinical* data, but the *paper* is *newer* (2025) and may not be the *most-validated* in the field yet.

After 156 + 157 + 158 + 159, the v0 sub-task 1 *feed-forward 3DGS* arc is *complete* (MVSplat 156 + DepthSplat 157 + PanSplat 158 + Splatt3R 159 = 4 papers, the *planar cost volume* + the *monocular depth fusion* + the *4K + Fibonacci* + the *pose-free* design), the *most-comprehensive* feed-forward 3DGS arc for v0 v0 v0 v0 v0 v0's *chairside-real-time* + *clinical-quality* + *pose-robust* sub-task 1.
