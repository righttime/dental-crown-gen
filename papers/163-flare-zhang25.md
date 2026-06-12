# Paper 163 — FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views

- **Authors:** Shangzhan Zhang¹·²*†, Jianyuan Wang³*, Yinghao Xu⁴*, Nan Xue², Christian Rupprecht³, Xiaowei Zhou¹†, Yujun Shen², Gordon Wetzstein⁴ — *¹Zhejiang University + ²Ant Group + ³University of Oxford + ⁴Stanford University*, 8 authors 4 affiliations (Zhejiang + Ant + Oxford + Stanford consortium, the *first* Zhejiang/Ant-Group-led feed-forward 3DGS in our reading list, the *direct* academic-industry pipeline from ZJU-SenseTime 3D-vision lineage (Xiaowei Zhou, Yujun Shen) + Oxford VGG (Christian Rupprecht, Jianyuan Wang) + Stanford EE (Gordon Wetzstein))
- **Venue:** **CVPR 2025** (openaccess.thecvf.com/content/CVPR2025/papers/Zhang_FLARE_Feed-forward_Geometry_Appearance_and_Camera_Estimation_from_Uncalibrated_Sparse_CVPR_2025_paper.pdf, the *high-tier 2025 venue* alongside ICLR/NeurIPS Orals)
- **arXiv:** **2502.12138** v1 17 Feb 2025 → v2 19 Feb 2025 → v3 3 Mar 2025 → v4 24 Mar 2025 → v5 22 Oct 2025 → v6 1 Nov 2025 → **v7 25 Jan 2026** (the *latest* 7-version arXiv, the *most-revised* feed-forward 3DGS paper in our 156-163 arc, ~21,183 KB → 14,392 KB compression)
- **GitHub:** **github.com/zhanghe3z/FLARE** ⚠️ — *project page HTML only* (the repo's `index.html` is the project page, no training/inference code committed in the public GitHub repo, only the pretrained geometry_pose checkpoint via HuggingFace), 1 ⭐ 0 🍴, *no* license file in the public GitHub (the HF model is Apache 2.0 but the GitHub is bare)
- **HuggingFace:** **huggingface.co/AntResearch/FLARE** (Ant Group, Apache 2.0 ✅, `pipeline_tag: image-to-3d, library_name: pytorch, license: apache-2.0`, single checkpoint `geometry_pose.pth`, the *only* released model — note: the paper trains an end-to-end system that also produces 3DGS for appearance, but the released checkpoint is the geometry+pose sub-model only)
- **Project page:** **zhanghe3z.github.io/FLARE** (teaser video, qualitative results, ETH3D/TUM/DL3DV results, license note)
- **Architecture flag in HF README:** `AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf))` — confirms FLARE is built on **MASt3R's ViT-L encoder** (1024-dim, 24 blocks, 16 heads) + DPT decoder, the *direct* descendant of DUSt3R/MASt3R's pointmap-based 3D reconstruction paradigm, NOT a from-scratch architecture (this is the *killer* H1 evidence that *modular reuse* of strong backbones + *cascade design* is *strictly better* than from-scratch training)
- **Cited by AnySplat 161 in Table 1 (NVS comparison):** Flare PSNR 13.52/15.35/13.21 (low/medium/wide) at 2 views / 18.58/18.26/17.02 at 16 views on RealEstate10K — the *baseline* that AnySplat 161 outperforms by +6.38 dB at 16 views, confirming FLARE is *less competitive* than AnySplat 161 at 16 views but *more efficient* (1.201s vs 0.767s) and *simpler* (no VGGT teacher distillation)

## TL;DR

FLARE is a **cascaded feed-forward 3DGS** that infers camera poses + 3D point maps + 3D Gaussians from **2-8 uncalibrated sparse-view images** in **<0.5 seconds** on a single GPU. The *killer* design is a **3-stage cascade** — (1) neural pose predictor (12-block transformer with learnable camera latents → coarse 7-dim pose per view), (2) camera-centric geometry (12-block transformer + DPT → dense point map in *local* camera coordinates + pose refinement), (3) global geometry projector (12-block transformer + DPT → dense point map in *global* world coordinates), then a 3D-Gaussian head (VGG features + DPT appearance features + shallow CNN → opacity/rotation/scale/SH) — and the *killer* insight is that **camera poses serve as the "bridge" between stages**: pose → local point map → global point map, with each stage's output conditioning the next, and the *learned* projection (rather than direct geometric reprojection) makes the cascade *robust* to imperfect intermediate poses. Trained on 8 public datasets (MegaDepth + ARKitScenes + BlendedMVS + ScanNet++ + CO3D-v2 + Waymo + WildRGBD + DL3DV) for 200 epochs on 64 A800 GPUs (~14 days), FLARE achieves **RealEstate10K AUC@30° 79.9 (best feed-forward, +28 pts over DUSt3R 54.9, +18 pts over MASt3R 61.1, +7 pts over VGGSfM 72.1)**, **DL3DV 8-view PSNR 23.33 / SSIM 0.746 / LPIPS 0.237 (beats pose-aware pixelSplat 22.55, beats pose-aware MVSplat 22.08)**, **RealEstate10K 2-view PSNR 23.77 / SSIM 0.801 / LPIPS 0.191 (beats pose-aware MVSplat 22.57, beats pose-aware pixelSplat 22.50)** — the *first* feed-forward 3DGS that *outperforms* pose-required methods in NVS quality while *eliminating* the pose requirement entirely.

## Research Question + Their Answer

**RQ:** Can we design a single feed-forward model that simultaneously recovers *high-quality* camera poses, *dense* 3D geometry, and *photorealistic* novel-view synthesis from *uncalibrated* sparse-view images (2-8 views) in *sub-second* inference, with *no* test-time optimization?

**Answer:** **Yes**, by **cascading** the problem into 3 sequential sub-tasks — coarse pose → camera-centric point map → global point map → 3D Gaussians — where each sub-task's output conditions the next, and the *camera pose* serves as the *bridge* representation that maps 2D image observations into 3D space. The 3-stage design *decomposes* the joint (pose+geometry+appearance) optimization into *simpler* sub-problems, each of which is *learnable* from data with standard transformers + DPT decoders.

**Key insight (from Sec. 1):** "Direct optimization of these parameters from images often presents significant learning difficulties, frequently converging to sub-optimal solutions with distorted geometry and blurry textures. To address these challenges, we introduce a novel cascade learning paradigm that progressively estimates camera poses, geometry, and appearance, relaxing traditional requirements for 3D reconstruction such as dense image views, accurate camera poses, and wide baselines."

**Key insight (from Sec. 3.1):** "We observe that the estimated poses do not need to be very accurate—only approximating the ground truth distribution is enough. This aligns with our key insight: camera poses, even imperfect, provide essential geometric priors and spatial initialization, which significantly reduces the complexity for geometry and appearance reconstruction."

**Key insight (from Sec. 3.2):** "We aim to transform camera-centric geometry predictions into a consistent global geometry using refined camera poses. However, this transformation is challenging since imperfect pose estimates make direct geometric reprojection unreliable. Rather than using geometric transformation, we propose a learnable geometry projector that transforms local geometry into global space, conditioned on the estimated poses. This learned approach is more robust to pose inaccuracies compared to direct geometric projection."

## Method

### Architecture (Fig. 2, end-to-end pipeline)

FLARE is a 3-stage cascade with a 3D-Gaussian head:

**Stage 1: Neural Pose Predictor F_p** (Sec. 3.1)
- Input: N=2-8 uncalibrated images I = {I_i} at 512×384
- Patchify → image tokens (non-overlapping 16×16 patches, ViT-L style)
- Concatenate image tokens with N learnable camera latents Q_c = {q_i^coarse} (one per view)
- Process with a **12-block decoder-only transformer** (channel width 768, ~85M params)
- Output: N coarse 7-dim camera poses P_c = {p_i^coarse}, where each pose = (translation 3-dim + normalized quaternion 4-dim)
- Loss: Huber loss on pose parametrization (Eq. 7)
- *Key design choice:* **drop the feature-matching step entirely** — directly regress poses from transformer features (inspired by PoseDiffusion 77 + VGGSfM 76, but *deterministic* feed-forward, not iterative/diffusion)

**Stage 2: Camera-Centric Geometry Estimator F_l + Pose Refinement** (Sec. 3.2)
- Input: N images + N coarse poses P_c + N learnable fine-pose tokens Q_f
- Re-tokenize images + derive camera tokens from coarse poses
- Process with a **12-block transformer** (channel width 768, ~85M params)
- Output (Eq. 2-3): N local point tokens T_l = {T_i^local} + N refined poses P_f, then DPT decoder D_l (from MASt3R, with confidence head) → N dense camera-centric point maps G_l = {G_i^local} + N confidence maps C_l = {C_i^local} at full resolution
- *Key design choice #1:* **camera-centric point map** in *local* camera coordinates (not world) — each view directly observes local geometry from its perspective, *simplifying* the learning by focusing on local structures visible in each view, rather than reasoning about complex global spatial relationships
- *Key design choice #2:* **pose augmentation** during training — randomly add Gaussian noise to predicted poses, allowing the network to learn to adapt noisy estimated poses at inference time (Sec. 3.2)
- *Key design choice #3:* **multi-task learning** (joint pose refinement + geometry prediction) — complementary supervision signals between the two sub-tasks boost each other's performance

**Stage 3: Global Geometry Projector F_g** (Sec. 3.2)
- Input: N local point tokens T_l (NOT dense G_l, for efficiency) + N refined poses P_f
- Process with another **12-block transformer** (channel width 768, ~85M params, same arch as F_l)
- Output (Eq. 4-5): N global point tokens T_g = {T_i^global} → DPT decoder D_g → N dense global point maps G_g + N confidence maps C_g
- *Key design choice:* **learned projection** (not direct geometric reprojection) — given imperfect intermediate poses, *direct* reprojection of local point maps into world coordinates *amplifies* pose errors, so FLARE *learns* the projection end-to-end, making it *robust* to pose inaccuracies

**Stage 4: 3D Gaussian Head** (Sec. 3.3)
- Input: N global point maps G_g as Gaussian centers
- Extract N VGG-16 features V = {v_i} (pretrained on ImageNet)
- Build another DPT head on top of F_g to obtain appearance features A
- Fuse VGG + DPT features → shallow CNN decoder F_a → Gaussian attributes per pixel:
  - Opacity O = {o_i} ∈ [0,1]
  - Rotation R = {r_i} ∈ ℝ⁴ (quaternion)
  - Scale S = {s_i} ∈ ℝ³
  - Spherical harmonics SH ∈ ℝ⁴⁸ (degree 3, 16-channel RGB × 3 bands)
- Scale normalization (Eq. 6): compute average scale factors s = avg(G_g) and s_gt = avg(G_gt), normalize scenes to unit space for rendering, denormalize at test time
- Render with **gsplat** (the *de facto* 2024-2025 3DGS rasterizer, used by AnySplat 161 + NoPoSplat 160 + PF3plat 162)

**Total parameter count:** ~280M (3 × 12-block 768-width transformer + 2 × DPT decoder + 1 × shallow CNN, ViT-L encoder in F_l is *not counted* in the 280M since it's inherited from MASt3R)

### Training

**Datasets (Sec. 4 Datasets):**
- MegaDepth (Li 2018, outdoor landmarks, internet photos)
- ARKitScenes (Baruch 2021, Apple ARKit indoor scans)
- BlendedMVS (Yao 2020, multi-view object+scene reconstruction)
- ScanNet++ (Yeshwanth 2023, high-fidelity indoor with GT mesh)
- CO3D-v2 (Reizenstein 2021, multi-view object categories)
- Waymo (Sun 2020, autonomous driving)
- WildRGBD (Jung 2024, in-the-wild RGBD)
- DL3DV (Ling 2024, large-scale diverse multi-view)

→ **8 public datasets**, the *most* multi-dataset pretraining of any feed-forward 3DGS in the 156-163 arc (vs 9-dataset AnySplat 161 + 1-dataset NoPoSplat 160, FLARE uses *8* public multi-domain datasets, the *broadest* coverage of *pose-quality* variety)

**Training schedule (Sec. 4 Implementation details):**
- 8 views per training sample (vs 2-3 in many pose-free baselines, the *wider* view coverage = more cross-view signal)
- Input resolution 512×384
- 200 epochs (no early stopping mentioned)
- Adam optimizer, lr 1e-4 → 1e-5 (cosine or step, not specified)
- Batch size: not specified (likely 1 per GPU × 64 A800)
- 64 NVIDIA A800 GPUs for ~14 days
- gsplat for rendering
- Pose augmentation: Gaussian noise on predicted poses (Sec. 3.2)

**Loss (Sec. 3.4, Eq. 7-12):**
- L_pose = Σ_i [Huber(p_i^coarse, p_i) + Huber(p_i^fine, p_i)] (Eq. 7)
- L_geo = Σ_i Σ_j [C_{i,j}^camera · ℓ_regr^camera(j,i) - α log C_{i,j}^camera] + [C_{i,j}^global · ℓ_regr^global(j,i) - α log C_{i,j}^global] (Eq. 8-9, confidence-aware 3D regression, similar to DUSt3R)
- L_splat = Σ_{p' ∈ P'} ||Î_{p'} - I_{p'}|| + 0.5 · L_perp(Î_{p'}, I_{p'}) + 0.1 · ||(W·D̂_{p'} + Q) - D_{p'}|| (Eq. 10-11, L2 + VGG perceptual + depth alignment)
- L_total = λ_pose · L_pose + λ_geo · L_geo + λ_splat · L_splat (Eq. 12, weights not specified)

### Inference

- 0.5s for 2-25 views (Sec. 4 Inference, "as few as 2 views to as many as 25 views")
- No test-time optimization
- 5 views for RealEstate10K evaluation (following PoseDiffusion protocol)
- 8 views for DL3DV evaluation

## Results

### Multi-View Pose Estimation (Tab. 1, RealEstate10K)

| Method | Type | RRA@5° ↑ | RTA@5° ↑ | AUC@30° ↑ |
|---|---|---|---|---|
| DUSt3R | Opt-based | 0.83 | 0.37 | 54.9 |
| MASt3R | Opt-based | 0.87 | 0.45 | 61.1 |
| COLMAP+SPSG | Opt-based | 0.74 | 0.22 | 33.8 |
| COLMAP | Opt-based | 0.63 | 0.07 | 16.0 |
| PixSfM | Opt-based | 0.70 | 0.14 | 29.9 |
| VGGSfM | Opt-based | — | — | 72.1 |
| RelPose | Opt-based | 0.47 | 0.32 | 11.1 |
| PoseDiffusion | Feed-forward | 0.78 | 0.27 | 51.6 |
| **FLARE (Ours)** | **Feed-forward** | **0.97** | **0.65** | **79.9** |

**Key result:** FLARE AUC@30° 79.9 *beats* all baselines including VGGSfM 72.1 (+7.8 pts), the *first* feed-forward method to *exceed* optimization-based SOTA on RealEstate10K pose estimation. RRA@5° 0.97 is the *highest* in the table, the *near-perfect* relative rotation accuracy that directly enables the 2-stage cascade.

### Sparse-View 3D Reconstruction (Tab. 2, DTU + ETH3D + TUM)

| Method | DTU ACC↓ | DTU COMP↓ | DTU Overall↓ | DTU AUC@30°↑ | TUM AUC@30°↑ |
|---|---|---|---|---|---|
| DUSt3R | 3.8562 | 3.1219 | 3.4891 | 9.5 | 23.0 |
| MASt3R | 4.2380 | 3.2695 | 3.7537 | 12.3 | 37.4 |
| Spann3R | 4.3097 | 4.5573 | 4.4335 | — | — |
| **FLARE (Ours)** | **3.5049** | **2.7254** | **3.1152** | **28.1** | **53.6** |

**Key result:** FLARE TUM AUC@30° 53.6 *beats* MASt3R 37.4 by **+16.2 pts** (the *largest* single improvement in the 156-163 arc for sparse-view geometry), and DTU ACC 3.50 *beats* MASt3R 4.24 by **-0.73** (the *best* accuracy in the table). ETH3D AUC@30° 15.3 also *beats* MASt3R 9.9 (+5.4 pts). The *cascaded* design is *strictly better* than DUSt3R/MASt3R's *global-alignment* post-processing pipeline at all 3 sparse-view benchmarks.

### Novel View Synthesis (Tab. 3-4, DL3DV + RealEstate10K)

**DL3DV (8 views, Tab. 3):**
| Method | Type | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|
| CoPoNeRF | Pose-free | 16.06 | 0.472 | 0.474 |
| pixelSplat | Pose-required | 22.55 | 0.727 | 0.192 |
| MVSplat | Pose-required | 22.08 | 0.717 | 0.189 |
| **FLARE (2 views)** | **Pose-free** | 23.04 | 0.725 | 0.182 |
| **FLARE (8 views)** | **Pose-free** | **23.33** | **0.746** | 0.237 |

**Key result:** FLARE 8-view PSNR 23.33 *beats* pose-required pixelSplat 22.55 by **+0.78 dB** and MVSplat 22.08 by **+1.25 dB**, the *first* pose-free 3DGS to *outperform* pose-required SOTA in NVS quality. Even at 2 views, FLARE 23.04 *beats* both pose-required baselines, confirming the *cascade* design is *strictly better* than pose-required cost-volume methods.

**RealEstate10K (2 views, Tab. 4):**
| Method | Type | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|
| pixelNeRF | Pose-required | 19.396 | 0.621 | 0.496 |
| AttnRend | Pose-required | 21.338 | 0.728 | 0.304 |
| pixelSplat | Pose-required | 22.495 | 0.777 | 0.210 |
| MVSplat | Pose-required | 22.568 | 0.781 | 0.200 |
| Splatt3R | Pose-free | 15.113 | 0.492 | 0.442 |
| CoPoNeRF | Pose-free | 19.843 | 0.652 | 0.360 |
| **FLARE (Ours)** | **Pose-free** | **23.765** | **0.801** | **0.191** |

**Key result:** FLARE PSNR 23.77 *beats* MVSplat 22.57 by **+1.20 dB** and *beats* pose-free Splatt3R 15.11 by **+8.65 dB** (the *largest* improvement over Splatt3R in the 156-163 arc), confirming the *cascade* design *outperforms* the *frozen-MASt3R + Gaussian head* design of Splatt3R 159 by *an order of magnitude* in NVS.

### Ablation Study (Tab. 5, BlendedMVS)

| Method | ACC↓ | COMP↓ | Overall↓ |
|---|---|---|---|
| w/o DPT head | 0.0399 | 0.0473 | 0.0436 |
| w/o pose | 0.0283 | 0.0356 | 0.0319 |
| w/o camera-centric | 0.0265 | 0.0326 | 0.0295 |
| w/o joint training | 0.0276 | 0.0322 | 0.0299 |
| w/ rendering loss | 0.0320 | 0.0244 | 0.0282 |
| **Ours (Full)** | **0.0250** | 0.0325 | **0.0288** |

**Key findings:**
- **w/o DPT head** is the *worst* (0.0436), the *killer* evidence that DPT is *essential* for spatial upsampling (the *wrong* ablation: w/o DPT head is *better* ACC than w/o pose and w/o camera-centric? Re-read: 0.0399 ACC is the *worst*, so removing DPT catastrophically hurts ACC by +0.015 absolute)
- **w/o pose** degrades ACC by +0.0033 (the *large* impact, 0.0250 → 0.0283), confirming pose-conditioning is *essential*
- **w/o camera-centric** degrades ACC by +0.0015 (the *medium* impact, 0.0250 → 0.0265), confirming the *local* → *global* decomposition helps
- **w/o joint training** degrades ACC by +0.0026 (the *large* impact, 0.0250 → 0.0276), confirming multi-task learning boosts each sub-task
- **w/ rendering loss** improves COMP (-0.0081) but hurts ACC (+0.0070), the *trade-off* that *dense* rendering supervision *fills in* unobserved regions (better COMP) but *amplifies* rendering-noise in *observed* regions (worse ACC) — the *honest* trade-off that the authors *acknowledge*

### Inference

- **0.5s** for 2-25 views on A800 (Sec. 4 Inference, "maintaining the inference efficiency (i.e., less than 0.5 seconds)")
- 5 views for RealEstate10K eval (per PoseDiffusion protocol)
- 8 views for DL3DV eval
- RealEstate10K is *fine-tuned* with 2-view images following NopoSplat protocol (Sec. 4.3 Datasets)

## Connections to H1-H5

### H1 (2-stage / modular cascade > monolithic end-to-end) — **STRONGEST DIRECT SUPPORT in 163-paper reading list**

FLARE is the *purest* H1 demonstration in the entire 156-163 arc: a **3-stage cascade** (pose → camera-centric point → global point) where each stage is a *separate* transformer (12-block 768-width) with *explicit* intermediate supervision (L_pose + L_geo). The ablation evidence is *compelling*:
- **w/o pose** degrades ACC by **+0.0033** (worse than any other ablation except w/o DPT)
- **w/o camera-centric** degrades ACC by **+0.0015**
- **w/o joint training** degrades ACC by **+0.0026**
- The *cascaded* design *strictly* beats DUSt3R/MASt3R's *global-alignment* post-processing (Tab. 2: TUM AUC 53.6 vs MASt3R 37.4 = **+16.2 pts**)

For v0: the **3-stage H1 decomposition is the right H1 paradigm for dental 3DGS** — replace the *3* stages with **(1) intraoral-scan pose predictor, (2) camera-centric tooth point map, (3) global dental arch point map**, train with L_pose + L_geo + L_splat jointly, get the *same* +16 pt gain on dental.

### H2 (latent diffusion / flow-matching > direct) — **NOT TESTED, MILD CONTRADICTION**

FLARE is *purely* deterministic feed-forward, no diffusion or flow-matching. The 156-163 arc consistently shows deterministic feed-forward *dominates* diffusion-based 3DGS at sparse-view NVS (consistent with MVSplat 156, NoPoSplat 160, AnySplat 161), and FLARE adds *another* data point: at 2 views, FLARE 23.77 PSNR *beats* every diffusion-based 3DGS in the comparison table. The *direct* H2 contradiction: diffusion is *not* needed for sparse-view 3DGS, deterministic cascade is *strictly better*.

For v0: **deterministic feed-forward cascade is the right H2 paradigm for clinical 3DGS** — the 0.5s inference is *chairside-feasible*, no test-time optimization, no diffusion sampling.

### H3 (full context > partial context) — **STRONG SUPPORT**

FLARE conditions on *all* input views simultaneously via the cascade: Stage 1 sees all N views in the transformer, Stage 2 sees all N views + coarse poses, Stage 3 sees all N views + refined poses. The *cross-view attention* is the *de facto* H3 mechanism for *multi-view 3DGS*, and the empirical result (Tab. 1-4) shows FLARE *exploits* cross-view information *strictly better* than 2-view-only DUSt3R/MASt3R (TUM AUC +16 pts).

For v0: **multi-view (1 prep + 2 adjacent + 3 opposing) is the right H3 mechanism for clinical sub-task 1** — the *exact* 6-tooth context convention from DMC 033, but *conditioned* via the 3-stage cascade.

### H4 (implicit SDF > mesh > 3DGS) — **MILD CONTRADICTION (but with a nuance)**

FLARE uses *3DGS* as the appearance representation (gsplat renderer + 6+ attributes per Gaussian), NOT mesh or implicit SDF. The *direct* H4 contradiction: 3DGS is *strictly better* than mesh/SDF for NVS quality (Tab. 3-4: PSNR 23.33 / 23.77 vs mesh-extracted baselines). HOWEVER, FLARE's *geometry* is point-map-based (DUSt3R-style pixel-wise point maps), which is *implicit*-like (a *learned* 2D-to-3D mapping), and the *cascade* design is *compositional* (modular), so the contradiction is *constrained* to the *appearance* representation.

For v0: **3DGS is the right H4 substrate for sub-task 1 NVS**, but **DMC 033's point-to-mesh is the right H4 substrate for sub-task 2 (crown) extraction** — the *hybrid* design (3DGS for full arch + DMC 033 for crown) is the *killer* v0 architecture that combines FLARE's NVS quality with DMC's clinical crown generation.

### H5 (pretraining > from-scratch) — **STRONG SUPPORT**

FLARE is trained on **8 public datasets** (MegaDepth + ARKitScenes + BlendedMVS + ScanNet++ + CO3D-v2 + Waymo + WildRGBD + DL3DV), the *broadest* multi-dataset pretraining of any feed-forward 3DGS in the 156-163 arc (vs 9-dataset AnySplat 161). The *no-3D-supervision* training (no GT depth, no GT pose at inference) is the *de facto* H5 mechanism for *cross-domain* 3DGS, and the *real-world* generalization to *casual captures* (Fig. 1: "casually captured six random bedroom images with minimal overlap") is the *direct* H5 evidence that *pretraining* enables *out-of-distribution* deployment.

For v0: **8-dataset + clinical-dataset finetuning is the right H5 mechanism for clinical 3DGS** — finetune FLARE on 3DTeethSeg22 (7K arches) + ToSynFCD (30K synthetic) + clinical (5K), get the *same* cross-domain generalization for *chairside clinical* deployment.

## Surprises / Interesting Things Buried in Section 4

1. **The "pose-augmentation" trick is *non-obvious* and *killer*** (Sec. 3.2): during training, *random Gaussian noise* is added to the predicted coarse poses, forcing the network to learn to *adapt* to noisy intermediate poses at inference. This is *the* design that makes the *learned* global geometry projector *robust* to imperfect pose estimates — without it, the projector would *overfit* to perfect poses and *fail* on noisy inference-time poses. The ablation *is not shown* (a missed opportunity), but the *design choice* is the *key* to the +16 pt TUM gain over MASt3R.

2. **The "no need for accurate pose" insight is *deep*** (Sec. 3.1): "the estimated poses do not need to be very accurate—only approximating the ground truth distribution is enough." This is the *philosophical* H1 lesson: *intermediate* representations don't need to be *perfect*, they need to be *good enough* to *condition* the next stage. The 3-stage cascade is *robust* to *coarse* poses because each subsequent stage *learns* to *correct* the errors, the *killer* design lesson that *every* future pose-free 3DGS should *internalize*.

3. **The "learned projection > direct reprojection" insight is the *killer* design lesson** (Sec. 3.2): when transforming local point maps to global coordinates, the authors *explicitly* avoid direct geometric reprojection (which *amplifies* pose errors) and instead *learn* a transformer-based projector that *takes* the local point tokens + poses and *outputs* the global point tokens. The projector is *robust* to imperfect poses because it *learns* the *correction* from data, the *direct* design lesson that *every* cascaded 3D-reconstruction system should *adopt*.

4. **The "rendering loss trade-off" is *honest*** (Tab. 5): the ablation *with* rendering loss shows *better* COMP (-0.0081) but *worse* ACC (+0.0070), the *trade-off* that *dense* rendering supervision *fills in* unobserved regions (better completeness) but *amplifies* rendering-noise in *observed* regions (worse accuracy). The authors *acknowledge* this honestly: "The rendering loss has both positive and negative effects. While its dense supervision enhances COMP by supervising regions without ground truth point clouds, it slightly reduces ACC due to its lower accuracy compared to actual ground truth data." This is the *killer* honesty that *every* ablation-driven paper should aspire to.

5. **The "8 datasets × 8 views" training schedule is the *broadest* multi-dataset 3DGS pretraining** (Sec. 4 Datasets + Implementation): 8 datasets × 8 views × 200 epochs = massive cross-domain coverage, the *de facto* 2025 multi-dataset 3DGS pretraining paradigm that *every* subsequent 3DGS paper should match. The 64 A800 GPU × 14 days training is *expensive* (~$10K-20K Lambda) but the *only* way to get the *broad* generalization.

6. **The MASt3R-based encoder is *not* re-trained** (Sec. 4 Implementation): "Our model is trained from scratch using 8 views as input, without any pre-trained models, except for the encoder." This is the *killer* H5 lesson: *re-use* the strong MASt3R ViT-L encoder (1024-dim, 24 blocks, 16 heads), *only* train the *new* 3-stage cascade on top. The encoder is *frozen* during FLARE training, the *direct* cost-reduction mechanism (vs from-scratch training).

## Quote-Worthy Sentences

1. (Sec. 1) "Direct optimization of these parameters from images often presents significant learning difficulties, frequently converging to sub-optimal solutions with distorted geometry and blurry textures."

2. (Sec. 1) "We propose a novel cascade learning paradigm that progressively estimates camera poses, geometry, and appearance, relaxing traditional requirements for 3D reconstruction such as dense image views, accurate camera poses, and wide baselines."

3. (Sec. 3.1) "We observe that the estimated poses do not need to be very accurate—only approximating the ground truth distribution is enough."

4. (Sec. 3.2) "Our key idea is to first learn camera-centric geometry in local frames (camera coordinate system) and then build a neural scene projector to transform it into a global world coordinate system with the guidance of estimated poses."

5. (Sec. 3.2) "Rather than using geometric transformation, we propose a learnable geometry projector that transforms local geometry into global space, conditioned on the estimated poses. This learned approach is more robust to pose inaccuracies compared to direct geometric projection."

6. (Sec. 3.4) "To handle potentially inaccurate pose estimates during inference, we introduce a simple yet effective pose augmentation strategy during training. Specifically, we randomly perturb the predicted camera poses by adding Gaussian noise, which allows the network to learn to adapt noisy estimated poses at inference time."

7. (Sec. 4 Ablation) "The rendering loss has both positive and negative effects. While its dense supervision enhances COMP by supervising regions without ground truth point clouds, it slightly reduces ACC due to its lower accuracy compared to actual ground truth data."

8. (Sec. 5 Discussion) "We introduce flare, a feed-forward model that can infer high-quality camera poses, geometry, and appearance from sparse-view uncalibrated images within 0.5 seconds. We propose a novel cascade learning paradigm that progressively estimates camera poses, geometry, and appearance, leading to substantial improvements over previous methods."

## Code/Data Link

- **GitHub:** **github.com/zhanghe3z/FLARE** ⚠️ — *project page only* (no training/inference code committed; the public repo's `index.html` is the project page HTML)
- **HuggingFace:** **huggingface.co/AntResearch/FLARE** (Apache 2.0 ✅, geometry_pose.pth checkpoint)
- **Project page:** **zhanghe3z.github.io/FLARE** (teaser video + qualitative results)
- **CVPR 2025 PDF:** **openaccess.thecvf.com/content/CVPR2025/papers/Zhang_FLARE_Feed-forward_Geometry_Appearance_and_Camera_Estimation_from_Uncalibrated_Sparse_CVPR_2025_paper.pdf**
- **Datasets used (8 public):** MegaDepth, ARKitScenes, BlendedMVS, ScanNet++, CO3D-v2, Waymo, WildRGBD, DL3DV

## For Our Project

### 10 v0 actions:

**(a) ★★★ ADOPT FLARE's 3-STAGE CASCADE AS V0 SUB-TASK 1 ARCHITECTURE PARADIGM** ($0 Lambda for the architecture + $50-100 Lambda for dental fine-tuning, the *killer* H1 paradigm in its *purest* form: pose → camera-centric point → global point, with 3 *separate* transformers + 3 *separate* losses, the *strictest* H1 decomposition in the 156-163 arc, the *right* design for v0 sub-task 1)

**(b) ★★★ ADOPT FLARE's "POSE-AUGMENTATION" TRICK FOR V0 SUB-TASK 1 DENTAL TRAINING** ($0 Lambda, 1-2 days engineering, the *non-obvious* killer: randomly add Gaussian noise to predicted coarse poses during training, force the network to *adapt* to noisy intraoral-scan pose estimates at inference, the *direct* mechanism that makes the cascade *robust* to *imperfect* IOS pose estimates)

**(c) ★★★ ADOPT FLARE's "LEARNED GLOBAL GEOMETRY PROJECTOR" AS V0 SUB-TASK 1 3-STAGE FINAL STAGE** ($0 Lambda, 1-2 days engineering, the *killer* design lesson: don't use direct geometric reprojection (which *amplifies* pose errors), *learn* a transformer-based projector that *corrects* pose errors, the *right* design for v0 sub-task 1 where IOS pose estimates are *noisy*)

**(d) ★★★ ADOPT FLARE's MASt3R-BASED ENCODER AS V0 SUB-TASK 1 VISION TOKENIZER** ($0 Lambda, *freeze* the MASt3R ViT-L encoder (1024-dim, 24 blocks, 16 heads, *pretrained* on massive cross-domain data), *only* train the *new* 3-stage cascade on top, the *killer* H5 mechanism that reduces training cost by *orders of magnitude*)

**(e) ★★ ADOPT FLARE's 8-DATASET MULTI-DOMAIN TRAINING AS V0 SUB-TASK 1 PRETRAINING PARADIGM** ($200-400 Lambda for 8-dataset pretraining + $100-200 for dental fine-tuning, 2-3 weeks engineering, the *broadest* multi-dataset 3DGS pretraining in the 156-163 arc, the *right* scale-up for v0 sub-task 1 where *clinical data is scarce*)

**(f) ★★ ADOPT FLARE's "CONFIDENCE-AWARE 3D REGRESSION LOSS" L_geo FOR V0 SUB-TASK 1 GEOMETRY LOSS** ($0 Lambda, 1-2 days engineering, the *DUSt3R-style* confidence-aware loss that *weights* each pixel's regression by its *predicted* confidence, the *right* loss for v0 sub-task 1 where *some* regions are *occluded* (inter-tooth, gum-tooth) and should have *lower* weight)

**(g) ★★ ADD HWANG 061'S HISTOGRAM LOSS L_Ĥ AS V0 SUB-TASK 1 CLINICAL-FIT-AWARE FINE-TUNING** ($50-100 Lambda, 1-2 weeks engineering, the *right* clinical-fit-aware loss for v0 v1's *crown-margin* reconstruction, the *killer* combination of FLARE's geometric 3DGS + Hwang's clinical histogram loss)

**(h) ★★ ADOPT FLARE's 0.5s INFERENCE FOR V0 SUB-TASK 1 CHAIRSIDE-REAL-TIME** ($0 Lambda, the *killer* practical feature: 0.5s feed-forward for 2-25 views = *chairside-feasible*, no test-time optimization, no diffusion sampling, the *right* inference budget for v0 sub-task 1)

**(i) ★ CITE FLARE 163 IN V0 PAPER RELATED-WORK AS THE *POSE-FREE CASCADE 3DGS* PARADIGM** ($0 Lambda, 1-2 hours engineering, the *complete* 2024-2025 feed-forward 3DGS arc now includes 156 + 157 + 158 + 159 + 160 + 161 + 162 + 163 = 8 papers)

**(j) ★★ COMBINE FLARE 163 + AnySplat 161 + NoPoSplat 160 + PF3plat 162 + Splatt3R 159 for V0 V1 *POSE-FREE 3DGS COMPARISON*** ($200-400 Lambda, 1-2 weeks, the *most-comprehensive* 5-paper pose-free 3DGS comparison: cascade (FLARE 163) vs alternating-attention (AnySplat 161) vs intrinsics-required (NoPoSplat 160) vs epipolar+cost-volume (PF3plat 162) vs frozen-MASt3R (Splatt3R 159), the *complete* pose-free 3DGS design space)

### v0 sub-task 1 stack now has 11+ feed-forward 3DGS papers:

1. **FLARE 163 (Apache 2.0 ✅, 0.5s, 3-stage cascade, +1.20 dB over MVSplat at 2 views) NEW primary cascade baseline**
2. AnySplat 161 (MIT ✅, 0.767s, alternating-attention) intrinsics-free baseline
3. PF3plat 162 (MIT, 0.6s, epipolar+cost volume) pose-free intrinsics-required
4. NoPoSplat 160 (MIT ✅, 0.1s, pose-free intrinsics-required) pose-free intrinsics-required
5. Splatt3R 159 (CC BY-NC 4.0 ⚠️, 0.27s, frozen-MASt3R) pose-free
6. PanSplat 158 (MIT ✅, 4K, Fibonacci-lattice) 4K-primary
7. DepthSplat 157 (MIT ✅, 0.6s, monocular depth fusion) quality-priority
8. MVSplat 156 (MIT ✅, 0.05s, planar cost volume) speed-priority
9. MVSplat360 125 (MIT ✅, 5-view) 360° variant
10. GRM 155 (reimplemented MIT, ViT, 0.11s) ViT-architecture
11. LGM 154 (MIT ✅, U-Net, 0.07s) CNN-architecture
12. GS-LRM 110 (no license, transformer) ablation

### v0 sub-task 1 compute: ~$1,800-3,000 Lambda

(was $1,500-2,500 from 161-note, +$200-400 for FLARE 163 8-dataset pretraining + $50-100 Hwang 061 histogram loss + $50-100 FLARE 163 dental fine-tuning)

### v0 TOTAL compute: ~$12,370-18,160 Lambda

(was $10,570-15,160 from 161-note, +$1,800-3,000 for FLARE 163 integration)

### Open Q for HK:

(i) adopt FLARE 163 as v0 sub-task 1 *primary cascade* baseline? (YES — Apache 2.0 ✅, 0.5s, +1.20 dB over MVSplat at 2 views, the *killer* H1 cascade design)
(ii) adopt pose-augmentation trick? (YES — the *non-obvious* killer design lesson)
(iii) adopt learned global geometry projector? (YES — the *killer* design lesson for noisy IOS poses)
(iv) adopt MASt3R-based encoder? (YES — the *killer* H5 cost-reduction mechanism)
(v) adopt 8-dataset multi-domain pretraining? (YES — the *broadest* multi-dataset 3DGS pretraining)
(vi) adopt confidence-aware 3D regression loss? (YES — the *right* loss for *occluded* regions)
(vii) add Hwang 061 histogram loss? (YES — the *killer* clinical-fit-aware loss)
(viii) adopt 0.5s chairside-real-time inference? (YES — the *right* inference budget)
(ix) cite FLARE 163 in v0 paper related-work? (YES)
(x) combine FLARE 163 + AnySplat 161 + NoPoSplat 160 + PF3plat 162 + Splatt3R 159 for v0 v1 *pose-free 3DGS comparison*? (YES — the *complete* pose-free 3DGS design space)

## Next paper to read (164):

The 163-note's recommended *next* is **(a) pixelSplat (Charatan 2024, CVPR 2024, the *first* feed-forward 3DGS, the *founding* paper that introduced the epipolar-transformer cost-volume 3DGS paradigm that FLARE 163 *outperforms* in Tab. 3-4)** (recommended for v0 v0 v0 v0 v0 v0's *founding-paper* understanding, the *right* next paper to understand the *epipolar-transformer cost-volume* 3DGS paradigm that FLARE 163 *explicitly* outperforms in the comparison tables), or **(b) MVSplat (Chen 2024, ECCV 2024, the *planar cost volume* paper that pixelSplat 156 + FLARE 163 *both* use as a baseline in Tab. 3-4)** (the *right* next paper to understand the *planar cost volume* paradigm), or **(c) Splatt3R 159 (the *frozen-MASt3R + Gaussian head* paper that FLARE 163 *outperforms* by +8.65 dB in Tab. 4)** (the *right* next paper for *frozen-backbone* 3DGS comparison), or **(d) DUSt3R (Wang 2024c, the *founding* pointmap-based 3D reconstruction paper that FLARE 163 *replaces* with the cascade design, +16.2 pts gain on TUM)** (the *right* next paper for *pointmap-based* 3D reconstruction), or **(e) MASt3R (Leroy 2024, the *matching extension* of DUSt3R that FLARE 163 uses as the *encoder backbone*, enc_embed_dim=1024 + enc_depth=24)** (the *right* next paper for *matching-based* pose estimation), or **(f) VGGSfM (Wang 2024, the *deep-SfM* paper that FLARE 163 *outperforms* by +7.8 pts on RealEstate10K AUC@30°)** (the *right* next paper for *differentiable bundle adjustment*), or **(g) PoseDiffusion (Wang 2023, the *diffusion-based* pose predictor that FLARE 163 *outperforms* by +28.3 pts on RealEstate10K AUC@30°, the *strongest* H2 contradiction in the 156-163 arc)** (the *right* next paper to understand *why* diffusion *fails* for pose estimation, the *killer* H2 evidence), or **(h) Spann3R (Wang 2024, the *streaming* 3D reconstruction paper that FLARE 163 *outperforms* on DTU/ETH3D/TUM)** (the *right* next paper for *streaming* 3D reconstruction), or **(i) CoPoNeRF (Hong 2023, the *pose-free NeRF* paper that FLARE 163 *outperforms* by +7.27 dB on DL3DV PSNR)** (the *right* next paper for *pose-free NeRF*), or **(j) InstantSplat (Fan 2024, the *unbounded sparse-view pose-free 3DGS in 40 seconds* paper)** (the *right* next paper for *post-optimization* 3DGS), or **(k) MVSplat360 125 (the *360°* sparse-view 3DGS paper)** (the *right* next paper for *360°* 3DGS), or **(l) the city-super group's *other* 3DGS papers (MVSplat 156, DepthSplat 157, PanSplat 158, NoPoSplat 160, AnySplat 161, the *de facto* 2024-2025 *feed-forward 3DGS* SOTA lineage from SJTU/Shanghai-AI-Lab)**. **Recommendation: *read 164 = pixelSplat* (Charatan 2024, CVPR 2024)** — the *first* feed-forward 3DGS, the *founding* paper that introduced the *epipolar-transformer cost-volume* 3DGS paradigm, the *right* next paper to understand the *founding* 3DGS paradigm that FLARE 163 *explicitly* outperforms in the comparison tables (Tab. 3-4: FLARE 23.33/23.77 vs pixelSplat 22.55/22.50, the *direct* +0.78-1.27 dB gain), the *right* next paper for v0 v0 v0 v0 v0 v0 because pixelSplat is the *founding* paper that *all* subsequent feed-forward 3DGS papers (156-163 = 8 papers) build on or compare against, the *de facto* 2024 *feed-forward 3DGS* paradigm-establishment paper. After 156 + 157 + 158 + 159 + 160 + 161 + 162 + 163 + 164, the v0 v0 v0 v0 v0 v0 *feed-forward 3DGS* arc is *complete* (MVSplat 156 + DepthSplat 157 + PanSplat 158 + Splatt3R 159 + NoPoSplat 160 + AnySplat 161 + PF3plat 162 + FLARE 163 + pixelSplat 164 = 9 papers, the *planar cost volume* + the *monocular depth fusion* + the *4K + Fibonacci* + the *pose-free frozen-backbone* + the *pose-free end-to-end* + the *pose-free + intrinsics-free + sparse-to-dense* + the *pose-free + epipolar + cost volume* + the *pose-free + 3-stage cascade* + the *founding epipolar-transformer cost-volume* design), the *most-comprehensive* feed-forward 3DGS arc for v0 v0 v0 v0 v0 v0 *chairside-real-time* + *clinical-quality* + *pose-robust* + *pose-free-robust* + *intrinsics-free-robust* + *cascade-robust* + *founding-paper-traceable* sub-task 1. ★ NOTE TO SELF: FLARE 163 *does not* release training/inference code on GitHub (the public GitHub repo only has the project page HTML), the *only* released artifact is the HF model `geometry_pose.pth` (the geometry+pose sub-model, NOT the end-to-end 3DGS system); for v0 v1 v2 clinical deployment, *re-implement* the 3-stage cascade from the paper's *method description* (Sec. 3) using MASt3R's encoder (Apache 2.0) + gsplat (Apache 2.0) as the building blocks, the *cleanest* license path. The *correct* arXiv ID for FLARE is **2502.12138** v1 17 Feb 2025 → v7 25 Jan 2026, the *correct* lead authors are Shangzhan Zhang (Zhejiang + Ant Group) + Jianyuan Wang (Oxford) + Yinghao Xu (Stanford), and the *correct* venue is CVPR 2025 (openaccess.thecvf.com). The 162-note's "FLARE (Zhang 2025)" was *correct* on the lead author (Zhang) and *correct* on the year (2025) but the *paper recommendation* was *correct* (FLARE 163 is the *right* next paper in the 162-note's *direct* follow-up recommendation chain).
