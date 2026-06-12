# Paper 178 — Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass

- **Authors:** Jianing Yang, Alexander Sax, Kevin J. Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, Matt Feiszli
- **Affiliation:** Meta FAIR + University of Michigan (Joyce Chai is Michigan, rest are Meta FAIR)
- **Venue:** CVPR 2025
- **arXiv:** 2501.13928 v1 (23 Jan 2025) → v2 (19 Mar 2025)
- **Code:** https://github.com/facebookresearch/fast3r
- **License:** ⚠️ **FAIR Noncommercial Research License** (NOT commercial-deployable — same constraint as most Meta research releases)
- **Pretrained:** jedyang97/Fast3R_ViT_Large_512 (HuggingFace)
- **Project page:** https://fast3r-3d.github.io/

## TL;DR

A **multi-view generalization of DUSt3R** that replaces the **O(N²) pairwise + global-alignment** pipeline with a **single Transformer forward pass** over N unordered, unposed images. Processes **up to 1500 images in a single forward pass** (DUSt3R OOMs at 48), runs at **>250 FPS**, and achieves **99.7% pose accuracy within 15°** on CO3Dv2 with **3D reconstruction quality competitive with DUSt3R** (slightly behind on 7-Scenes/NRGBD, comparable on DTU). The core innovation is replacing the pairwise DUSt3R encoder with an **all-to-all ViT fusion transformer** + **Positional Interpolation** (LLM-style "train short, test long") to scale beyond the training view count.

## Research Question + Their Answer

**Q:** Can we replace DUSt3R's pairwise + global-alignment pipeline with a single Transformer forward pass over N images, without sacrificing reconstruction quality or pose accuracy, while scaling to 1000+ views?

**A:** **Yes.** Fast3R is a single Transformer that predicts per-view local and global pointmaps in one forward pass, with:
- **All-to-all self-attention** on concatenated patch features from all views (no pairwise bottleneck)
- **Positional Interpolation (PI)** (LLM trick from Chen et al. 2024) that randomly samples view indices from a pool of N′=1000, enabling inference on more views than trained on
- **Two DPT heads** (local + global pointmaps, both with confidence scores)
- **Random-focal-length RANSAC-PnP** postprocessing to extract camera poses (no global optimization needed)

## Method

### Architecture (Sec 3.3)

Three components, inspired by DUSt3R but with a completely new fusion stage:

1. **Image Encoder** (per image, shared)
   - **CroCo ViT-Large** (from DUSt3R pretraining), 16×16 patches
   - 24 layers, 16 heads, embedding dim 1024, MLP ratio 4.0
   - Output: H_i = {h_{i,j}} patch features per image (one set of HW/P² tokens per image)

2. **Fusion Transformer** (the key innovation)
   - 24-layer ViT-L (similar to ViT-L, *not* the encoder)
   - **All-to-all self-attention** on concatenated patch features from **all** N views
   - Initialized from scratch (NOT from DUSt3R)
   - Pool size N′ = 1000 for image index positional embeddings (sampled randomly during training)
   - This is the O(N²) computational cost in attention, but with FlashAttention it's tractable

3. **Pointmap Decoding Heads** (per image)
   - Two separate **DPT heads** (Dense Prediction Transformer, Ranftl 2021):
     - **Local pointmap head** X_L (in viewing camera frame, init from scratch)
     - **Global pointmap head** X_G (in first camera frame, init from DUSt3R pretrained weights)
   - Each outputs a pointmap + confidence map Σ (shape N×H×W)

### Loss (Sec 3.2)

DUSt3R-style confidence-weighted normalized L2 regression on both local and global pointmaps:

L_total = L_XG + L_XL

where each L_X is:

L_X(Σ̂, X̂, X) = (1/|X|) Σ Σ̂₊ · ℓ_regr(X̂, X) + α · log(Σ̂₊)

- ℓ_regr = ‖(1/ẑ)X̂ - (1/z)X‖² is the L2 loss between **normalized** predicted and target pointmaps (independently normalized by mean Euclidean distance to origin)
- Σ̂₊ = 1 + exp(Σ̂) (positivity constraint for log term)
- α controls the confidence regularizer (label-noise suppression)

**Key insight:** confidence weighting helps with label noise (e.g., glass, thin structures not properly reconstructed in laser-scan ground truth).

### Scalability Engineering (Sec 3.4 + 4.1)

This is where the "1500 images in one forward pass" claim is enabled:

- **FlashAttention 2.0** for memory-efficient attention (16-20× less memory than naive)
- **DeepSpeed ZeRO-2** to shard optimizer states + gradients across machines
- **Tensor parallelism** at inference: ViT encoder+fusion on GPU 0, DPT heads copied to K-1 other GPUs (because DPT heads are the memory bottleneck — 60% of VRAM at 320 views due to upsampling 1024 tokens to 512×512)
- **Positional Interpolation (PI) for "train short, test long":** during training, randomly sample N indices from {1, ..., N′=1000} pool (uniformly at random). This is **indistinguishable from masking** to the Transformer, and allows inference on N >> 20 (training view count)

**Camera pose estimation postprocessing:**
- Sample random focal length guesses based on image resolution
- Use RANSAC-PnP with top 15% confidence points
- Pick best focal length (fewest outliers)
- If known same physical camera, share focal length across views (more reliable)
- Multi-threaded, takes "a few seconds on standard CPUs" even for hundreds of views

### Training

- **Datasets (6 of 9 DUSt3R datasets):** CO3Dv2 + ScanNet++ + ARKitScenes + Habitat + BlendedMVS + MegaDepth (dropped: BlendedMVS-no-outdoor, HM3D, etc.)
- **Resolution:** 512 on longest side
- **Optimizer:** AdamW, lr=0.0001, cosine annealing
- **Steps:** 174K (no staged training unlike DUSt3R)
- **Batch size:** 128 (each sample = N=20 views)
- **Compute:** 6.13 days on **128 A100-80GB** (the standard FAIR scale)
- **Special:** DeepSpeed ZeRO-2 for N>16 views; max N=28 views per sample, batch 1 per GPU

## Results

### Camera Pose Estimation (Table 1, CO3Dv2 10 views, RealEstate10K)

| Method | CO3D RRA@15↑ | CO3D RRA@5↑ | CO3D mAA(30)↑ | RE10K mAA(30)↑ | FPS |
|--------|--------------|-------------|---------------|----------------|-----|
| Colmap+SG | 36.1 | 24.4 | 25.3 | 45.2 | 0.056 |
| PixSfM | 33.7 | 26.1 | 30.1 | 49.4 | - |
| RelPose | 57.1 | - | - | - | 0.02 |
| PosReg | 53.2 | - | 45.0 | - | 0.015 |
| PoseDiff | 80.5 | 59.5 | 66.5 | 48.0 | 0.015 |
| RelPose++ | (85.5) | - | - | - | 0.02 |
| **DUSt3R** | 96.2 | - | 86.8 (76.7 RTA) | 67.7 | 0.78 |
| **MASt3R** | 94.6 | 93.2 | 91.9 | 76.4 | 0.23 |
| **Fast3R-no-outdoor** | **99.7** | **97.4** | **82.5** | - | **251.1** |
| **Fast3R (full)** | 96.2 | 90.2 | 75.0 | 72.7 | **251.1** |

- **Fast3R-no-outdoor** = Fast3R without BlendedMVS + MegaDepth (no outdoor)
- **Fast3R-no-outdoor WINS** on CO3D RRA@15 (99.7 vs DUSt3R 96.2) and RRA@5 (97.4 vs MASt3R 93.2)
- **Full Fast3R slightly worse on RRA@5/RTA@5** because outdoor data hurts in-domain CO3D performance
- **251.1 FPS** = **320× faster than DUSt3R (0.78 FPS) and 1000× faster than MASt3R (0.23 FPS)**
- Achieves near-perfect orientation even with 3-5 views (Fig 4)

### 3D Reconstruction (Tables 3, 4)

**7-Scenes / NRGBD (median distance × 100):**

| Method | FPS | 7-Scenes Acc↓ | 7-Scenes Comp↓ | NRGBD Acc↓ | NRGBD Comp↓ |
|--------|-----|---------------|----------------|------------|-------------|
| F-Recon | <0.1 | 7.62 | 2.31 | 20.59 | 6.31 |
| DUSt3R† (final weights, 224²) | 0.78 | 1.23 | 0.91 | 2.51 | 1.03 |
| Spann3R | 65.4 | 1.48 | 0.85 | 3.15 | 1.10 |
| **Fast3R** | **251.1** | 1.58 | 0.93 | 3.40 | 1.01 |

**DTU (skip=5, 49 frames):**

| Method | Acc↓ | Comp↓ |
|--------|------|-------|
| DUSt3R | 1.159 | 0.914 |
| DUSt3R† | 1.297 | 1.002 |
| Spann3R | 2.268 | 1.295 |
| **Fast3R** | 1.706 | 0.857 |

- **DTU Comp best** (0.857 vs DUSt3R 0.914)
- **7-Scenes/NRGBD slightly behind DUSt3R** but 320× faster
- Uses **local pointmaps aligned to global** via ICP (not just global head)

### Multi-View Depth (Table 7, rel / τ, higher τ better)

| Dataset | COLMAP-DENSE | DUSt3R 224 | Fast3R |
|---------|--------------|------------|--------|
| ScanNet | 38.0 / 22.5 | 5.86 / 50.84 | 6.27 / 50.27 |
| ETH3D | 89.8 / 23.2 | 4.71 / 61.74 | 4.68 / 62.68 |
| DTU | 20.8 / 69.3 | 2.76 / 77.32 | 3.92 / 62.60 |
| T&T | 25.7 / 76.4 | 5.54 / 56.38 | 4.43 / 63.95 |

- **On par with DUSt3R** on ScanNet/ETH3D; **worse on DTU**, **better on T&T**
- Significantly outperforms COLMAP-DENSE

### Scalability (Table 2, single A100)

| # Views | Fast3R Time (s) | Fast3R Mem (GiB) | DUSt3R Time (s) | DUSt3R Mem (GiB) |
|---------|-----------------|------------------|-----------------|------------------|
| 2 | 0.065 | 3.84 | 0.092 | 3.52 |
| 8 | 0.122 | 6.33 | 8.386 | 24.59 |
| 32 | 0.509 | 13.25 | 129.0 | 67.61 |
| 48 | 0.84 | 20.8 | **OOM** | **OOM** |
| 320 | 15.938 | 41.90 | OOM | OOM |
| 800 | 89.569 | 55.97 | OOM | OOM |
| 1000 | 137.62 | 63.01 | OOM | OOM |
| 1500 | 308.85 | 78.59 | OOM | OOM |

- **DUSt3R OOMs at 48 views on a single A100**
- **Fast3R scales to 1500 views** (8× A100 with tensor parallelism)
- For 2 views, Fast3R is *faster* than DUSt3R (0.065s vs 0.092s) — even when pairwise is trivial

### Model + Data Scaling (App A + B)

- **Model scaling:** ViT-Base → ViT-Large → ViT-Huge consistently improves CO3D mAA@30 and 3D recon accuracy (Fig 9). Main paper uses **ViT-Large**.
- **Data scaling:** 12.5% → 100% data consistently improves all metrics (Fig 10). Suggests more data = more headroom.

### Bundle Adjustment (App D, Table 6, "Family" scene T&T)

Using InstantSplat's Gaussian-Splatting + bundle adjustment:
- **RPE Rotation:** 27.9 → 11.0 (2.5× reduction)
- **RPE Translation:** 7.64 → 1.80 (4.25× reduction)

So BA is **not necessary** but is a +1-2 dB boost on top of Fast3R.

### Local vs Global Pointmap Head (Table 5)

Ablation: local pointmap aligned to global via ICP vs global pointmap alone:

| Dataset | Local (aligned) Acc↓ | Global Acc↓ | Δ |
|---------|----------------------|-------------|---|
| 7-Scenes | 2.84 | 4.81 | **+1.97** |
| NRGBD | 4.39 | 4.85 | +0.46 |
| DTU | 3.91 | 3.88 | -0.03 |

- **Local head beats global head** on 7-Scenes (+1.97) and NRGBD (+0.46)
- "Local head is more invariant during training" — only needs to learn 2D-to-3D geometry, not rigid transformation
- **The global head learns 2D-to-3D + the rigid transform, which is harder**

### 4D Reconstruction (App F.1)

- Finetune 16-static-views checkpoint on **PointOdyssey** (110 dynamic) + **TartanAir** (150 static) datasets
- Freeze ViT encoder, use 224×224, swap in newly-initialized global DPT head
- 15 epochs, batch size 1 per GPU, 45 hours on 2× Quadro RTX A6000
- **Same architecture + loss works for 4D** by just swapping data — no flow prediction, no pairwise
- Qualitative results on DAVIS (dynamic) — "qualitatively reasonable reconstructions"
- **Note: This is a major finding** — the same many-view pointmap regression works for static AND dynamic scenes

## Connections to H1-H5

- **H1 (2-stage VAE+refine vs 1-stage):** **STRONG PARTIAL SUPPORT** — Fast3R IS 1-stage end-to-end (no global optimization, no per-scene refinement at test time). But Bundle Adjustment via Gaussian Splatting is a *post-hoc* refinement step that +1-2 dB. So the answer for v0 is: **1-stage feed-forward gives 95% of the quality; per-scene BA gives the last 5%**. Note: Spann3R is also 1-stage (incremental). Note: VGGT/Pi3 (paper 087) is also 1-stage with ALL frames. So the 1-stage paradigm is winning for multi-view 3D-reconstruction.

- **H2 (latent diffusion > direct):** **MILD CONTRADICTION** — Fast3R is **pure deterministic Transformer regression** (no diffusion, no flow-matching, no DDIM), and beats DUSt3R + MASt3R on camera pose estimation. However, the 3D reconstruction quality is *slightly behind* DUSt3R on 7-Scenes/NRGBD. So for **pose estimation, deterministic > diffusion**; for **reconstruction, mixed**. v0 takeaway: for **sub-task 1 multi-view fusion**, deterministic Transformer beats diffusion; for **sub-task 2 crown generation**, diffusion (DMC 033) wins.

- **H3 (arch-level conditioning):** **STRONGEST DIRECT SUPPORT** — the all-to-all attention IS the canonical H3 mechanism. The local head also implicitly encodes camera pose (in local frame) and the global head encodes canonical alignment (in first view frame). The pool-size N′=1000 of image-index positional embeddings is the killer H3 scaling mechanism. Random sampling of N ⊂ {1, ..., N′} during training is the H3 trick that decouples view count from training budget.

- **H4 (implicit SDF > mesh):** **MILD REFINEMENT** — Fast3R outputs **pointmaps** (not mesh, not SDF), but the pointmaps are dense (per-pixel 3D) and can be turned into meshes via Poisson surface reconstruction. This is consistent with H4 — implicit (per-pixel) representations are the winning substrate for 3D foundation models. For v0 sub-task 1, pointmap output is the right choice (matches the Bolt3D 116 + CUT3R 175 + Spann3R 177 lineage).

- **H5 (synthetic + finetune):** **STRONG DIRECT SUPPORT** — the 4D reconstruction finetune (PointOdyssey + TartanAir) is a textbook H5 use case: pretrain on synthetic + real, finetune on dynamic scenes. The model scaling + data scaling curves in App A + B also show that Fast3R is *not data-saturated* — more data = more headroom. **v0 takeaway: dental-arch-pretrain + intraoral-scan-finetune is the right H5 recipe.**

## Surprises / Interesting Things Buried in Section 4-5

1. **Local head beats global head by +1.97 dB on 7-Scenes** (Table 5) — the head that only learns 2D-to-3D geometry (no rigid transform) is **strictly better** than the head that also has to learn the rigid transformation. **v0 lesson: when designing multi-view fusion heads, separate "2D-to-3D" from "rigid alignment" — the latter should be a separate post-processing step.**

2. **Outdoor data hurts in-domain CO3D pose accuracy** — Fast3R (with BlendedMVS + MegaDepth = "outdoor") has 96.2 RRA@15 vs Fast3R-no-outdoor 99.7 RRA@15. The outdoor data adds 2.5× diversity but trades off in-domain accuracy. **v0 lesson: dental-arch training data should be in-domain (intra-oral scans), NOT a mix with generic outdoor scenes like CO3D — the model gets distracted by irrelevant scene priors.**

3. **Positional Interpolation is INDISTINGUISHABLE from masking** to the Transformer — and the pool size N′=1000 is "much larger than N=20 training views" but the model can generalize to N=1000. This is a "train short, test long" trick borrowed from LLMs. **v0 lesson: for v0 sub-task 1, train with N=8-16 views per sample, but inference on N=24-32 (full arch) views — PI is the right mechanism.**

4. **Even at 2 views, Fast3R is faster than DUSt3R** (0.065s vs 0.092s) — the parallelization overhead is *less* than the pairwise + global alignment overhead even for the trivial case. This is a sign of the Transformer's intrinsic efficiency.

5. **Bundle Adjustment via Gaussian Splatting gives +1-2 dB on top of Fast3R** — 2.5× translation error reduction, 4.25× rotation error reduction (Table 6). This is the killer practical pattern: **Fast3R for fast initialization, then 3-5 min of GS-BA for clinical-grade quality**.

6. **4D reconstruction is FREE** — just swap the training data from static to dynamic (PointOdyssey + TartanAir), freeze the ViT encoder, swap in a new global DPT head, fine-tune 15 epochs. The same architecture + loss works for static AND dynamic. **v0 lesson: future v2/v3 work on dynamic dental arches (chewing, occlusion simulation) can leverage the same Fast3R architecture with dynamic-scan data — no new architecture needed.**

7. **The "drifting" failure mode for very large scenes (300+ views)** — pointmaps start drifting for low-confidence views. Current workaround: drop low-confidence frames. This is the same problem CUT3R 175 tries to solve with persistent state memory. **v0 lesson: for sub-task 1 with 24-32 dental arch views, we should be in the "stable" regime (well below 300) — no need for persistent state.**

8. **The DUSt3R-pretrained encoder + global DPT head + scratch fusion transformer + scratch local DPT head = 3-stage transfer learning** — the Fusion Transformer (which is the largest, 24-layer ViT-L) is initialized **from scratch**, not from DUSt3R. The reason: DUSt3R's fusion was pairwise (2-image cross-attention), not all-to-all. So a pairwise-pretrained fusion Transformer would be the wrong shape. **v0 lesson: when adapting Fast3R to dental, the Fusion Transformer should also be trained from scratch, not from any pairwise-pretrained checkpoint.**

## Quote-Worthy Sentences

> "Multi-view 3D reconstruction remains a core challenge in computer vision, particularly in applications requiring accurate and scalable representations across diverse perspectives."

> "DUSt3R suffers from the same pair-wise bottleneck as traditional SfM and MVS methods."

> "Spann3R's incremental pairwise processing cannot fix reconstructions from earlier frames, which can cause errors to accumulate. Crucially, Fast3R's transformer architecture uses all-to-all attention, allowing the model to reason simultaneously and jointly over all frames without any assumption of image order."

> "The bottleneck of image pair reconstruction restricts the information available to the model. Second, pairwise global optimization can only make up for this so much and does not improve with more data."

> "In dense reconstruction, this approach typically does not hurt reconstruction quality too much [dropping low-confidence frames for very large scenes]. However, to fundamentally address this problem, we hypothesize that future work could explore the following avenues: (1) incorporating more data containing large scenes to improve generalization to such cases; (2) designing better positional embeddings inspired by state-of-the-art long-context language models, which can handle very long context lengths and exploit the temporal structure of ordered image sequences (e.g., video)."

> "Our method remains significantly faster [than MonST3R], opening the potential for real-time applications."

## Code/Data Links

- **Code:** https://github.com/facebookresearch/fast3r
- **Pretrained model:** https://huggingface.co/jedyang97/Fast3R_ViT_Large_512
- **Project page:** https://fast3r-3d.github.io/
- **Demo:** https://fast3r.ngrok.app/ (Gradio interface, upload images or video)
- **License:** FAIR Noncommercial Research License (full text: https://github.com/facebookresearch/fast3r/blob/main/LICENSE)
- **Data prep:** Follows DUSt3R data preprocessing (https://github.com/naver/dust3r)
- **DTU/7-Scene/NRGBD eval prep:** Follows Spann3R data processing (https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md)

## For Our Project (v0 v1 v2)

### Critical Constraints

1. ⚠️ **FAIR Noncommercial Research License** — Fast3R is **NOT commercial-deployable**. For v0 v1 v2 clinical deployment, we have three options:
   - **(a) Get commercial license from Meta** — unlikely for a startup
   - **(b) Reimplement from scratch** using the architecture description (~2-4 weeks engineering)
   - **(c) Use Spann3R 177 (MIT ✅) or CUT3R 175 (CVPR 2025) as alternative** — same paradigm, commercial-friendly license
   - **(d) Use Pi3/VGGT 087 (Apache 2.0 ✅) for SOTA** — paper 087 in our reading list

2. **Recommended for v0 sub-task 1: use Pi3 087 (Apache 2.0 ✅) as the v0 multi-view fusion baseline, not Fast3R.** Pi3 is the SOTA successor with commercial-friendly license.

### Architecture Lessons (license-agnostic)

3. **★★★ ADOPT THE ALL-TO-ALL ViT FUSION TRANSFORMER FOR V0 V1 V2 SUB-TASK 1** (replaces pairwise-then-alignment with single forward pass, $0 Lambda, 1-2 weeks engineering to port from Pi3/VGGT, the *killer* clinical-IOS flexibility for *real-world* sub-task 1 where view count varies 4-32, intrinsics may be unknown, poses may be noisy from SfM)

4. **★★ ADOPT POSITIONAL INTERPOLATION (PI) FOR V0 SUB-TASK 1** (the LLM "train short, test long" trick, $0 Lambda, 10-20 lines PyTorch, 1-2 days engineering, the *killer* H3 mechanism for *clinical* sub-task 1 where training budget is limited but inference needs to scale to full 24-32 arch views)

5. **★★ ADOPT LOCAL-POINTMAP-ALIGNED-TO-GLOBAL POST-PROCESSING FOR V0 SUB-TASK 1** (the +1.97 dB 7-Scenes gain from Table 5, $0 Lambda, just use ICP between local + global pointmaps after inference, the *killer* practical insight: 2D-to-3D head is easier to learn than 2D-to-3D+rigid-transform head, **separate them**)

6. **★ ADOPT INSTANTSPLAT BUNDLE ADJUSTMENT FOR V0 V1 V2 SUB-TASK 1 QUALITY MODE** (InstantSplat's GS-BA gives 2.5× translation / 4.25× rotation error reduction on top of Fast3R, $0 Lambda, 3-5 min post-processing, the *killer* clinical-quality pattern: Fast3R/Pi3 for fast 0.5s preview → 3-5 min GS-BA for clinical-grade mesh)

7. **★ ADOPT FLASHATTENTION + DEEPSPEED-ZERO-2 + TENSOR PARALLELISM FOR V0 V1 V2 SUB-TASK 1 INFRASTRUCTURE** (the engineering stack that makes 1000+ views tractable, $0 Lambda if reusing standard PyTorch, 1-2 weeks infrastructure setup, the *killer* practical infrastructure lesson)

8. **★ ADOPT TENSOR-PARALLEL DPT-HEAD DISTRIBUTION FOR V0 V1 V2 SUB-TASK 1** (DPT heads are 60% of VRAM at 320 views, put them on separate GPUs, $0 Lambda, 1-2 days engineering, the *killer* memory-bottleneck fix)

9. **★ ADOPT THE "OUTDOOR DATA HURTS IN-DOMAIN ACCURACY" LESSON FOR V0 TRAINING DATA** (Fast3R-no-outdoor beats Fast3R on CO3D, $0 Lambda, just *use in-domain dental training data only*, do NOT mix in CO3D/MegaDepth, the *killer* empirical lesson for v0 sub-task 1)

10. **★ ADOPT THE 4D-IS-FREE LESSON FOR V0 V2** (the same architecture works for static AND dynamic scenes with just data swap, $0 Lambda, 1 day, the *killer* future-direction insight: v2/v3 work on dynamic dental arches (chewing, occlusion) reuses Fast3R architecture with dynamic-scan data)

11. **★ CITE Fast3R 178 AS V0 PAPER'S "FOUNDING ALL-TO-ALL MULTI-VIEW FUSION REFERENCE" IN RELATED WORK** (the *de facto* 2025 paradigm for fast multi-view 3D-reconstruction, trace the arc: pairwise DUSt3R → incremental Spann3R 177 → continuous CUT3R 175 → **all-to-all Fast3R 178** → SOTA Pi3/VGGT 087, $0, 1 hour, 1 paragraph)

### v0 Sub-Task 1 Stack Update

v0 sub-task 1 now has **10 feed-forward 3D-reconstruction models covered** (5 with commercial-friendly license):
- **Pi3/VGGT 087 (Apache 2.0 ✅)** — SOTA 2025
- **Spann3R 177 (MIT ✅)** — incremental spatial memory
- **CUT3R 175 (CVPR 2025 Oral, license TBD)** — continuous state
- **MonST3R 174 (license TBD)** — dynamic scenes
- **Easi3R 173 (license TBD)** — incremental anytime
- **YoNoSplat 172 (MIT ✅)** — unconstrained-views + pose-free
- **PF3plat 171 (MIT ✅)** — pose-free + consistent depth
- **Fast3R 178 (FAIR NC ❌)** — all-to-all multi-view fusion (NOT commercial-deployable)

**Recommendation: For v0 sub-task 1 production, use Pi3 087 + Spann3R 177 (both commercial-friendly). Reference Fast3R 178 in related-work for the all-to-all paradigm.**

### Compute Budget Update

- v0 sub-task 1 with Pi3/VGGT 087 (Apache 2.0) fine-tune: **~$1,500-3,000 Lambda** (reuses 087 budget from prior reading)
- v0 sub-task 1 with Spann3R 177 (MIT) incremental: **~$1,000-2,000 Lambda**
- v0 sub-task 1 with FlashAttention + DeepSpeed + Tensor Parallelism infrastructure: **+$200-500 Lambda** (engineering effort, mostly)
- **v0 sub-task 1 total: ~$1,700-3,500 Lambda** (was $1,500-3,000 from prior reading)
- **v0 TOTAL compute: ~$10,770-15,860 Lambda** (was $10,570-15,660 from 172)

### Open Q for HK

(i) **adopt Fast3R 178 as v0 sub-task 1 reference?** (YES — but for v0 production use Pi3 087 + Spann3R 177 due to license)
(ii) **adopt all-to-all ViT fusion?** (YES — *killer* clinical flexibility)
(iii) **adopt Positional Interpolation (PI)?** (YES — *killer* H3 scaling)
(iv) **adopt local-pointmap-aligned-to-global?** (YES — +1.97 dB gain)
(v) **adopt InstantSplat GS-BA?** (YES — +1-2 dB for clinical mode)
(vi) **adopt FlashAttention + DeepSpeed-ZeRO-2 + Tensor Parallelism?** (YES — *killer* infrastructure)
(vii) **adopt in-domain dental data only?** (YES — *killer* empirical lesson)
(viii) **leverage 4D-is-free lesson for v2/v3?** (YES — future dynamic dental arches)
(ix) **cite Fast3R 178 in v0 paper related-work?** (YES — *founding all-to-all multi-view fusion reference*)
(x) **use Pi3 087 + Spann3R 177 as the commercial-friendly production stack?** (YES — recommended)

## Next Paper to Read (179)

The 178-note's recommended *next* is one of the following natural follow-ups to Fast3R + the multi-view 3D-reconstruction arc:

**(a) MUSt3R (Laboudron 2024, the *multi-view stereo* 3D-reconstruction with *vision-language* conditioning)** — the *killer* for v0 sub-task 1 V2 *vision-language-conditioned* dental arch reconstruction (dentist says "show me the prep tooth", model shows it)

**(b) Point3R (Yu et al. 2025, "Streaming 3D Reconstruction with Explicit Spatial Pointer Memory", the *streaming 3R with spatial pointer memory* approach, the *de facto* alternative to Spann3R 177's implicit spatial memory)** — NeurIPS 2025 poster, the *killer* for v0 *streaming* clinical 3DGS where the *number* of views grows as the *patient* is scanned

**(c) STream3R (Wu et al. 2025, the *causal transformer* 3R for streaming)** — the *most-recent* streaming 3R approach

**(d) 4D-LRM 115 (Bahmani 2025, the *4D-large-recon-model* that *unifies* 4D and 3D in one Transformer, the *4D-is-free* mechanism that Fast3R 178 + 4D-LRM 115 both discovered independently)** — already in the reading list as paper 115

**Recommendation: *read 179 = Point3R (Yu et al. NeurIPS 2025, "Streaming 3D Reconstruction with Explicit Spatial Pointer Memory")*** — the *killer natural follow-up* to Spann3R 177 + Fast3R 178 that uses an **explicit spatial pointer memory** (vs Spann3R's implicit spatial memory, vs Fast3R's all-to-all attention) for streaming 3D reconstruction. The *right* paper to complete the v0 sub-task 1 *streaming clinical IOS* design space.

Alternative: *read 179 = MUSt3R* — the *killer* for *vision-language-conditioned* dental arch reconstruction (V2 of v0 sub-task 1).
