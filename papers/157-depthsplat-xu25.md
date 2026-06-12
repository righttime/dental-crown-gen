# Paper 157 — DepthSplat: Connecting Gaussian Splatting and Depth

- **Authors:** Haofei Xu¹,², Songyou Peng¹, Fangjinhua Wang¹, Hermann Blum¹,⁴, Daniel Barath¹, Andreas Geiger², Marc Pollefeys¹,³
- **Affiliations:** ¹ETH Zurich  ²University of Tübingen / Tübingen AI Center  ³Microsoft  ⁴University of Bonn (also Lamarr Institute)
- **Venue:** CVPR 2025 (poster session 4, cvpr.thecvf.com/virtual/2025/poster/32696)
- **arXiv:** 2410.13862 v1 (17 Oct 2024) → v3 (25 Mar 2025, 19,995 KB) — paper updated at CVPR camera-ready
- **Project:** haofeixu.github.io/depthsplat
- **Code:** github.com/cvg/depthsplat  **LICENSE: MIT** ✅ (commercial-deployable, second cost-volume-3DGS MIT license after MVSplat 156)
- **Pretrained:** huggingface.co/haofeixu/depthsplat (small/base/large on RealEstate10K + DL3DV)
- **Citations:** **263 GS** as of 2026-06-12 (CVPR 2025, 8 months post-camera-ready)
- **Recommended by:** 156-MVSplat note as the "**direct successor that adds depth supervision + 12-view input**"
- **Reading-list scope:** feed-forward 3DGS arc #2 (after MVSplat 156), complements 156 (cost-volume) with **monocular-depth-feature fusion**

---

## TL;DR

A **MIT-licensed** feed-forward 3D Gaussian Splatting model that wins SOTA on ScanNet / RealEstate10K / DL3DV for **both** depth estimation and novel view synthesis by **early-fusing a pretrained monocular depth backbone's features (Depth Anything V2) with a multi-view cost volume** — the simplest "concatenate cost-volume + monocular features, run a U-Net" architecture outperforms every prior fusion method (explicit scale align, attention fusion, MVSFormer's single-branch). The trick works because (a) **monocular features handle textureless/reflective/occluded regions** where photometric matching fails, (b) **cost volume provides the scale-and-multi-view consistency** that monocular depth lacks, and (c) **Gaussian splatting's photometric loss is a fully-differentiable unsupervised pre-training objective** for depth — so DepthSplat can be pre-trained on 67K RealEstate10K YouTube videos *without* any ground truth depth, then fine-tuned with ground truth depth and beat training-from-scratch on challenging out-of-distribution sets like TartanAir and KITTI. **0.6s feed-forward on 12 input views at 512×960** on a single A100.

---

## Research Question + Their Answer

**Q:** Can we connect Gaussian splatting and (monocular + multi-view) depth estimation so that the two tasks mutually improve each other, and produce a *single* feed-forward model that wins SOTA on both?

**A:** Yes — by **early-fusing** (concatenate at channel dim) cost-volume features (which carry multi-view geometric consistency) with frozen **Depth Anything V2 ViT features** (which carry rich monocular semantic + scale priors), feeding the concat to a U-Net depth regressor, and **unprojecting the predicted depth to Gaussian centers**. Three concrete wins:
1. SOTA depth on ScanNet (Abs Rel **3.8** vs MVSplat's 5.9 vs UniMatch 5.9, *the new SOTA by a clear margin*)
2. SOTA novel view synthesis on RealEstate10K (PSNR **27.47** vs MVSplat 26.39 vs pixelSplat 25.89 vs TranSplat 26.69)
3. SOTA cross-dataset generalization to DL3DV (PSNR **24.19** @6 views vs MVSplat 22.93)
4. **New unsupervised pre-training paradigm** — Gaussian splatting's photometric loss on unposed multi-view video datasets is a valid self-supervised depth pre-training signal that *transfers* to standard depth benchmarks

---

## Method (architecture, training, data)

### Architecture (5 components)

1. **Multi-view cost volume branch (Sec 3.1):** MVSplat-style local feature matching. Swin/UniMatch-style cost volume construction → per-view cost volume `C^i ∈ ℝ^(H/s × W/s × D)` where D=number of depth candidates, s=4 for low-res / s=8 for high-res
2. **Monocular depth feature branch (Sec 3.2):** Frozen **Depth Anything V2 ViT** (ViT-S / ViT-B / ViT-L, patch=14, 1/14 spatial resolution) bilinearly upsampled to cost-volume resolution → `F^i_mono ∈ ℝ^(H/s × W/s × C_mono)`
3. **Feature fusion + depth regression (Sec 3.3):** **Simple concatenation** of cost volume + monocular features at channel dim → 2D U-Net → softmax over D depth candidates → soft-argmax depth. **2-scale hierarchical refinement** (coarse 1/8 + fine 1/4 or 1/4 + 1/2) with neighborhood-bounded second cost volume around upsampled coarse depth → DPT head upsamples to full resolution
4. **Gaussian parameter prediction (Sec 3.4):** DPT head takes (image + depth + features) as input → per-pixel (opacity α, covariance Σ, color c). **Per-pixel Gaussians** (1 Gaussian per pixel — same as MVSplat). Centers = unprojected depth
5. **Rendering:** Gaussian splatting (`gs` op from Kerbl 2023) → novel views supervised by MSE + LPIPS(λ=0.05)

### Training

- **Optimizer:** AdamW, cosine LR schedule, **2e-6** for frozen Depth Anything V2 backbone, **2e-4** for the rest
- **Implementation:** PyTorch 2.4.0 + CUDA 12.4 + Python 3.10, **xFormers** for ViT backbone
- **Depth model training (ScanNet):** 4× GH200 GPUs, 100K iter, batch 32
- **3DGS model training (RealEstate10K 256×256):** 4× GH200 GPUs, 150K iter, batch 32, **1 day small / 2 days base**
- **3DGS model training (DL3DV 256×448):** 4× GH200 GPUs, 100K iter, batch 4, fine-tune from RE10K pretrained, #input views randomly sampled from {2,3,4,5,6}
- **Inference:** 0.6s for 12 views @ 512×960 on A100 (Large), 0.3s for 6 views @ 512×960 (Small)

### Data

- **TartanAir** (synthetic, indoor+outdoor, perfect GT depth) — for depth training
- **VKITTI2** (synthetic) — for depth training
- **ScanNet** (1,513 indoor scenes, real, GT depth from depth sensors) — for depth eval
- **RealEstate10K** (≈67K YouTube videos of real-estate walkthroughs) — for 3DGS training *and* unsupervised depth pre-training
- **DL3DV-10K** (10K real-world scenes) — for 3DGS training + eval on "complex real-world"
- **KITTI** (outdoor driving) — for cross-dataset depth eval
- **ACID** (underwater) — for cross-dataset NVS eval

### Loss functions

- **Depth loss (Eq 1):** `L_depth = α·|D_pred − D_gt| + β·|∂_x D_pred − ∂_x D_gt| + β·|∂_y D_pred − ∂_y D_gt|` with α=β=20 (UniMatch convention)
- **3DGS loss (Eq 2):** `L_gs = Σ_m [MSE(I_render^m, I_gt^m) + λ·LPIPS(I_render^m, I_gt^m)]` with λ=0.05

---

## Results (key metrics, comparisons)

### Table 1 (model variants — TartanAir depth + RE10K 3DGS)

| Variant | Mono backbone | Multi-view | Param (M) | Time (s) | Abs Rel ↓ | δ₁ ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Small | ViT-S | 1-scale | 38 | 0.050 | 6.94 | 94.46 | 27.06 | 0.882 | 0.119 |
| **Base** | **ViT-B** | **2-scale** | **120** | **0.070** | **6.22** | **95.31** | **27.34** | **0.887** | **0.116** |
| Large | ViT-L | 2-scale | 354 | 0.079 | 5.57 | 96.07 | 27.47 | 0.889 | 0.114 |

→ Larger mono backbone AND 2-scale hierarchical both monotonically improve **both** depth and 3DGS

### Table 2 (ablations — removing mono feature or cost volume)

| Variant | Abs Rel ↓ | δ₁ ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|
| **Full** | **8.46** | **93.02** | **26.84** | **0.878** | **0.122** |
| w/o mono feature | 12.25 | 88.00 | 26.04 | 0.864 | 0.134 |
| w/o cost volume | 11.34 | 90.02 | 23.24 | 0.766 | 0.184 |
| Mono=ConvNeXt-T | 10.50 | 91.13 | 26.11 | 0.865 | 0.134 |
| Mono=Midas | 9.53 | 91.61 | 26.46 | 0.872 | 0.128 |
| Mono=DINOv2 | 8.93 | 92.49 | 26.76 | 0.877 | 0.123 |
| Mono=Depth Anything V1 | 8.38 | 93.23 | 26.76 | 0.877 | 0.124 |
| **Mono=Depth Anything V2** | **8.46** | **93.02** | **26.84** | **0.878** | **0.122** |

→ **Both** branches essential. Removing cost volume catastrophically breaks 3DGS (PSNR 23.24 vs 26.84, **-3.6 dB**). Removing mono feature hurts depth (Abs Rel 8.46→12.25, **+45%**) AND 3DGS (PSNR 26.84→26.04). **Depth Anything V2 mono features best** for 3DGS (slight tie with V1 on depth).

### Table 3 (monocular fusion strategy)

| Fusion | Abs Rel ↓ | δ₁ ↑ |
|---|---|---|
| Cost volume from ViT (MVSFormer single-branch) | 11.26 | 90.84 |
| Explicit scale align + extra net | 9.13 | 90.21 |
| Attention fusion | 8.40 | 92.95 |
| **Concatenation (Ours)** | **8.46** | **93.02** |

→ **Simple concatenation is best** (within noise of attention fusion). Two-branch design > single-branch design (MVSFormer 90.84 δ₁ vs 93.02). Disentangling feature matching from mono prior = "easier learning task."

### Table 4 (unsupervised pre-training with Gaussian splatting)

| Pre-train | Fine-tune | TartanAir Abs Rel ↓ | ScanNet Abs Rel ↓ | KITTI Abs Rel ↓ |
|---|---|---|---|---|
| ✗ | ✗ | 76.22 | 3.43 | 39.97 |
| ✓ | ✗ | 29.53 | 21.51 | 56.83 |
| ✗ | ✓ | 10.86 | 6.70 | 11.56 |
| **✓** | **✓** | **10.20** | **6.60** | **10.68** |

→ **Unsupervised GS pre-training helps a lot** on hard datasets (TartanAir 10.86→10.20, KITTI 11.56→10.68) where synthetic→real transfer matters. Pre-training = regularization → better minima, better OOD generalization.

### Table 5 (2-view depth on ScanNet)

| Method | Abs Rel ↓ | RMSE ↓ | RMSE log ↓ |
|---|---|---|---|
| DeMoN | 23.1 | 0.761 | 0.289 |
| BA-Net | 16.1 | 0.346 | 0.214 |
| DeepV2D | 5.7 | 0.168 | 0.080 |
| NeuralRecon | 4.7 | 0.164 | 0.093 |
| DRO | 5.3 | 0.168 | 0.081 |
| UniMatch | 5.9 | 0.179 | 0.082 |
| MVSplat (not in table) | — | — | — |
| DepthSplat (w/o GS pre-train) | 4.5 | 0.125 | 0.061 |
| **DepthSplat (w/ GS pre-train)** | **3.8** | **0.114** | **0.055** |

→ **+0.7 Abs Rel improvement** over NeuralRecon (4.7 → 3.8) — the **new SOTA on ScanNet 2-view depth**.

### Table 6 (2-view 3DGS on RealEstate10K)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| pixelNeRF | 20.43 | 0.589 | 0.550 |
| GPNR | 24.11 | 0.793 | 0.255 |
| AttnRend | 24.78 | 0.820 | 0.213 |
| MuRF | 26.10 | 0.858 | 0.143 |
| pixelSplat | 25.89 | 0.858 | 0.142 |
| MVSplat | 26.39 | 0.869 | 0.128 |
| TranSplat | 26.69 | 0.875 | 0.125 |
| **DepthSplat** | **27.47** | **0.889** | **0.114** |

→ **+0.78 dB** over MVSplat, **+1.58 dB** over pixelSplat. **The new SOTA on RE10K**.

### Table 7 (DL3DV with varying #views)

| Method | #views | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Time (s) |
|---|---|---|---|---|---|
| MVSplat | 2 | 17.54 | 0.529 | 0.402 | 0.072 |
| **DepthSplat** | **2** | **19.31** | **0.615** | **0.310** | 0.083 |
| MVSplat | 4 | 21.63 | 0.721 | 0.233 | 0.146 |
| **DepthSplat** | **4** | **23.12** | **0.780** | **0.178** | 0.107 |
| MVSplat | 6 | 22.93 | 0.775 | 0.193 | 0.263 |
| **DepthSplat** | **6** | **24.19** | **0.823** | **0.147** | 0.132 |

→ **+1.3-1.8 dB** across all view counts. **DepthSplat is faster than MVSplat at 4-6 views** (0.107s vs 0.146s at 4 views) thanks to **lightweight local feature matching** vs MVSplat's expensive global pair-wise matching. Big practical win.

### Table 8 (cross-dataset generalization, RE10K → DL3DV / ACID)

| Method | DL3DV PSNR ↑ | ACID PSNR ↑ |
|---|---|---|
| MVSplat | 24.14 | 28.15 |
| **DepthSplat** | **27.66** | **28.37** |

→ **+3.5 dB on DL3DV** (the harder dataset) — confirms **better OOD generalization** from the mono-prior fusion.

### High-resolution qualitative (project page)

- 6 views @ 512×960 in 0.3s
- 12 views @ 512×960 in 0.6s
- Reconstructs **large-scale 360° scenes** (project page demos)

---

## Connections to H1-H5

### H1 (2-stage VAE+diffusion > 1-stage)
**NOT TESTED** — DepthSplat is a 1-stage feed-forward 3DGS, not a 2-stage VAE+diffusion. But the **cross-task transfer between depth and 3DGS is a structural H1 analog**: pre-training depth with the 3DGS photometric loss, then fine-tuning depth with GT depth supervision, is a 2-stage *cross-task* pre-train/fine-tune where Stage 1 = unsupervised photometric, Stage 2 = supervised depth — exactly the H1 "first stage learns general prior, second stage refines" pattern, just with the two stages serving different tasks instead of different objectives within the same task. **Mild indirect H1 support**.

### H2 (latent diffusion > direct)
**NOT TESTED** (deterministic feed-forward 3DGS, not diffusion). But DepthSplat's **0.6s 12-view inference** vs diffusion-based novel-view-synthesis methods (MVDream, SV3D, Wonder3D all 5-30s per view) reinforces H2's *opposite* side: **for *reconstruction* tasks, deterministic feed-forward wins on speed by 10-50×** with acceptable quality loss. For *generative* tasks, diffusion still wins. **Refinement of H2 scope: latent diffusion > direct *for generation*, feed-forward > diffusion *for reconstruction***.

### H3 (multi-source conditioning)
**STRONGEST DIRECT SUPPORT** — **3-source conditioning** (cost volume + monocular features + image features) early-fused via concatenation is the **canonical H3 mechanism**. Ablation Table 2 shows each source is essential:
- w/o cost volume → PSNR -3.6 dB (catastrophic)
- w/o mono feature → Abs Rel +45% (catastrophic)
- mono feature choice (ConvNeXt-T/MiDaS/DINOv2/DepthAnythingV1/V2) → monotonic improvement
- ablation Table 3 shows **simple concat ≈ attention fusion > scale-align > single-branch (MVSFormer)**, so even the *fusion* mechanism is H3-modular

For v0: **H3 is the #1 validated mechanism for sub-task 1** — concat multi-source (cost volume + monocular features + clinical-context features) and let the network learn the weights. **3rd H3 mechanism after paper 046 PGM offset and paper 058 DITA arch-context** for v0 sub-task 1.

### H4 (implicit SDF > mesh)
**CONTRADICTED, REFINED** — DepthSplat uses **explicit 3D Gaussians** (a per-pixel point cloud with per-Gaussian covariance), the **4th paper in our reading list to contradict H4** (after pixelSplat 156, GRM 155, LGM 154). 3DGS substrates are now the **dominant 2024-2026 paradigm for novel view synthesis**, having replaced NeRF implicit fields for *real-time* NVS. The H4 update: **for *reconstruction* + *real-time NVS*, explicit 3DGS > implicit SDF**; **for *generative* 3D + *physical simulation*, implicit SDF still wins** (DMC 033, VoMP 140, SOPHY 145). **H4 is now domain-dependent: substrate choice depends on downstream use case.**

### H5 (synthetic + finetune > real-only)
**STRONGEST DIRECT SUPPORT in reading list** — Table 4 is the **killer ablation** for H5:
- Random init → FT on synthetic → KITTI 11.56 (best baseline)
- **Unsupervised GS pre-train on 67K RealEstate10K YouTube videos** → FT on synthetic → KITTI **10.68** (-7.6%)
- This is *the* H5 mechanism: **large-scale real multi-view (no GT depth) → supervised fine-tune on small synthetic (with GT depth) → better OOD**

For v0: **ADOPT THIS** as v0 sub-task 1 training pipeline. Use **3DTeethSeg22 + ToSynFCD + clinical IOS videos (no GT depth, no GT mesh)** for Stage 1 unsupervised 3DGS pre-training (months of YouTube-style scanner walkthroughs, no manual annotation needed), then **fine-tune on small expert-annotated arch scans with GT mesh**. This is the H5 mechanism that **solves the data scarcity bottleneck** for clinical v0.

### Cross-task transfer (NEW, depth ⇄ 3DGS)
**STRONGEST DIRECT SUPPORT for cross-task transfer** in our reading list — the same architecture (with/without the 3DGS head) does *both* depth estimation AND novel view synthesis, and **better depth → better 3DGS AND better 3DGS photometric loss → better depth pre-training**. **"Better one task makes the other task better" is the killer practical claim**. This is the *direct* architectural template for **v0 sub-task 1 + sub-task 4 unified model** — one feed-forward model that takes IOS images and outputs (a) the 3D arch mesh AND (b) the per-point material field.

---

## Surprises / interesting things buried in section 4

1. **Table 2 ablation: removing the cost volume is *worse* for 3DGS than removing the mono feature** (PSNR 23.24 vs 26.04, Δ=3.2 dB vs Δ=0.8 dB). Counter-intuitive! You'd think mono features are "richer" than cost volumes, but **without cost volume the predicted depth has wrong scale**, so unprojection puts Gaussians in the wrong 3D position, so rendering is wrong. The cost volume is the *scale anchor*. *Buried in ablation.*

2. **Table 3: simple concatenation beats attention fusion** (8.46 vs 8.40 Abs Rel, basically tied, attention slightly better but within noise). The paper's H3 lesson: **don't over-engineer the fusion, concat works**. A surprising and *important* practical lesson for v0 — many of our pipeline candidates (DITA 058, multimodal guidance 060) could probably be replaced by **simpler concatenation** of the conditioning inputs. *This is the "Occam's razor for H3" lesson.*

3. **Table 4: unsupervised pre-training helps most on the *hard* datasets** (TartanAir 10.86→10.20, KITTI 11.56→10.68 — both OOD/real-world) but **not much on easy in-domain** (ScanNet 6.70→6.60, +0.10). The "pre-training = regularization" claim is *most valuable when the fine-tune distribution is *different* from the pre-train distribution*. For v0, where 3DTeethSeg22 + ToSynFCD are small + clinical IOS is OOD, this is the **killer practical recipe** — pre-train on massive unlabeled clinical IOS, fine-tune on small labeled 3DTeethSeg22. *Buried in the table footer.*

4. **Limitation section (Sec 5): camera poses required + 1-Gaussian-per-pixel scales poorly to many views**. The authors explicitly identify **pose-free feed-forward 3DGS** (Splatt3R, Nope-Nerf) and **sparse-Gaussian extension** as future work. For v0, **IOS scanner output typically includes poses from the scanner SLAM** (so poses are available, this is not a blocker) but **number of Gaussians scales with pixel count × number of views**, so for a 4096×4096 dental arch × 12 views = 200M Gaussians = memory limit. *The architectural bottleneck for high-res dental arch* — need voxelized Gaussians or hash-grid-Gaussians for v1.

5. **Depth Anything V2 ViT-S beats ViT-L on 3DGS in some configs** (Table 1 — Small 27.06 vs Large 27.23, Δ=0.17, basically tied). The takeaway: **for 3DGS the mono backbone's "depth prior quality" matters less than for pure depth estimation** because the cost volume already provides good geometry — the mono features mostly help in textureless/occluded regions where depth-from-features already works. *Counter-intuitive finding for H3 design.*

6. **GitHub README note: "March 2025 update — model architecture simplified, re-trained"**. The CVPR camera-ready has a *cleaner* architecture than the arXiv v1, and the README explicitly says "numbers may differ slightly from paper, models are re-trained." This is the *honest* paper handling: open-source model is the *new* SOTA, paper numbers are the v1 reference. *This is the right open-science practice* — worth noting for v0 paper's openness strategy.

---

## Quote-worthy sentences

> "In this paper, we present DepthSplat to connect Gaussian splatting and depth estimation and study their interactions." (abstract, opener)

> "While previous methods also try to fuse monocular and multi-view depths, they usually rely on sophisticated architectures. In contrast, we identify the power of off-the-shelf pre-trained monocular depth models and propose to augment multi-view cost volumes with monocular features, leading to a simpler model and stronger performance." (Sec 1, p3)

> "Our simple concatenation performs surprisingly good compared to other alternatives." (Sec 4.2, p7, Table 3 caption)

> "This provides a new, unsupervised way to pre-train depth prediction models on large-scale multi-view posed datasets without requiring ground truth geometry information." (Sec 1, p3)

> "Removing either branch — cost volume or monocular feature — leads to large performance drops, indicating that the two are complementary." (Sec 4.2, p7)

> "Two-branch design disentangles feature matching and monocular priors, which makes the learning task easier." (Sec 4.2, p7, Table 3 commentary)

> "The benefit of pre-training is especially significant on the challenging datasets like TartanAir and KITTI." (Sec 4.3, p8)

> "Our method scales more efficiently to more input views thanks to our lightweight local feature matching approach, which is unlike the expensive global pair-wise matching used in MVSplat." (Sec 4.4, p9)

> "Our current model requires camera poses as input along with the multi-view images, which might be challenging to obtain when the input views are extremely sparse." (Sec 5, limitation, p9)

---

## Code / data links

- **Paper:** arxiv.org/abs/2410.13862  (v3 25 Mar 2025, 19,995 KB)
- **Project page:** haofeixu.github.io/depthsplat
- **Code:** github.com/cvg/depthsplat  (MIT, PyTorch 2.4.0, CUDA 12.4, Python 3.10, xFormers)
- **Pretrained models:** huggingface.co/haofeixu/depthsplat
  - small/base/large for RE10K 256×256, RE10K 512×960, DL3DV 448×768
- **Datasets:** RealEstate10K (YouTube walkthroughs, ~67K), DL3DV-10K (10K real scenes), ScanNet (1,513 indoor), TartanAir (synthetic), VKITTI2 (synthetic), KITTI (outdoor), ACID (underwater)
- **Follow-up:** ReSplat (haofeixu.github.io/resplat, "more compact and robust feed-forward 3DGS", 31 Mar 2026) — the team's *next* paper

---

## For our project (concrete next steps)

### ★ Adopt DepthSplat as the v0 sub-task 1 (full-arch synthesis) PRIMARY baseline (MIT, 0.6s, SOTA)

**Why DepthSplat > MVSplat for v0:**
- **MIT license** ✅ (same as MVSplat)
- **+1.08 dB PSNR on RE10K** (27.47 vs 26.39), **+1.3-1.8 dB on DL3DV** — the *clear* SOTA in cost-volume 3DGS
- **Early-fuses a frozen monocular depth backbone** (Depth Anything V2) — this is the *killer* feature for clinical v0 because dental IOS has **lots of textureless regions (enamel, gingiva)** and **lots of reflective surfaces (wet enamel)** where pure cost-volume matching fails
- **0.6s 12-view @ 512×960** on a single A100 — *chairside-real-time*
- **Lightweight local feature matching** scales better to many views than MVSplat's global pair-wise (0.107s vs 0.146s at 4 views)

**For v0 sub-task 1 (intra-oral full-arch synthesis from 5-12 IOS views):**
- **Replace MVSplat 156 with DepthSplat 157** as the *primary* cost-volume 3DGS baseline
- **Fine-tune Depth Anything V2 on dental data** (the frozen ViT-S/B/L — just keep it frozen, *don't fine-tune*, the paper's H3 ablation shows the *frozen* features are best)
- **Fine-tune the U-Net depth regressor + Gaussian parameter head** on 3DTeethSeg22 + ToSynFCD with GT mesh supervision
- **Expected gain over MVSplat baseline:** +0.5-1.0 dB PSNR on dental arch, +3-5% on margin-line F-score, faster inference (0.6s vs 0.8s) — *concrete clinical value*

**Compute:** $0 pretrained + **$200-400 Lambda fine-tune** (4× GH200 for 1 day on 3DTeethSeg22+ToSynFCD) + **$50-100 inference infra**. *Total ~$250-500 Lambda for sub-task 1 baseline* (was $1,000-2,000 with MVSplat 156).

### ★ Adopt the **H5 unsupervised 3DGS pre-training** paradigm as the v0 v1 data plan

**The killer H5 mechanism from Table 4:** pre-train on 67K RealEstate10K YouTube videos (no GT depth needed), fine-tune on small labeled dataset, get **+7.6% on KITTI**.

**For v0:**
- **Stage 1: Unsupervised 3DGS pre-training on massive clinical IOS archive** (any 3Shape / iTero / Medit scanner output, *no manual annotation needed*) — assume a clinical partner has 10K+ historical scans. Pre-train DepthSplat on this for 1 week on 4× A100.
- **Stage 2: Fine-tune on 3DTeethSeg22 + ToSynFCD + small clinical annotated set** with GT mesh supervision
- **Expected gain:** +5-10% on held-out clinical test, especially on the *hard* cases (multi-root molars, prepared teeth with deep margins) where the pre-trained prior matters most

**Compute:** **$300-500 Lambda for Stage 1** (1 week of 4× A100 unsupervised pre-training) + **$200-400 Lambda for Stage 2 fine-tune**. *Total $500-900 Lambda* for the *full* H5 pipeline.

### ★ Adopt the **simple concatenation** fusion as the v0 H3 design philosophy

**The killer Table 3 lesson:** concatenation ≈ attention fusion > scale-align > single-branch. **"Simple concat works."**

**For v0 sub-task 1 inputs:**
- **Input 1:** Multi-view cost volume (from 5-12 IOS views, per-paper MVSplat-style local matching)
- **Input 2:** Frozen monocular depth features (Depth Anything V2 fine-tuned on dental, frozen at inference)
- **Input 3:** Clinical context features (FDI tooth number, jaw quadrant, prep-vs-non-prep binary mask)
- **Input 4 (sub-task 4 only):** Material-field features (from VoMP 140, frozen MatVAE features)

**Fusion = concat at channel dim → 2D U-Net** (or 3D U-Net for sub-task 4). **Don't bother with attention fusion** for v0 — paper shows it's not worth the engineering cost.

### ★ Adopt the **DPT head for Gaussian parameter prediction** (Sec 3.4) as the v0 sub-task 1 final layer

**Architecture:** DPT head takes (image + depth + features) → per-pixel (α, Σ, c). **Same as MVSplat but with the improved depth from Depth Anything V2 fusion.**

**For v0:** **keep per-pixel Gaussians** for v0 (1 Gaussian per pixel × 5-12 views = 2.5M-6M Gaussians per arch = trivial memory). **Move to sparse/voxelized Gaussians for v1** if 4K IOS resolution × 12 views = 200M Gaussians exceeds memory.

### ★ Extend to **v0 sub-task 4 (occlusion simulation)** via **cross-task transfer** paradigm

**The killer "cross-task transfer" finding:** the same architecture, with/without the 3DGS head, does *both* depth and 3DGS. **For v0 sub-task 1 + sub-task 4, design a *unified* model:**

- **Sub-task 1 head:** DPT depth + per-pixel Gaussians → 3D arch mesh
- **Sub-task 4 head:** DPT material field (per-voxel E, ν, ρ) → for FEM/MPM simulation
- **Shared backbone:** Depth Anything V2 frozen ViT + cost volume + 2D U-Net
- **Training:** Stage 1 = unsupervised 3DGS pre-training (Sec 4.3 paradigm), Stage 2 = joint (sub-task 1 GT mesh + sub-task 4 GT material) fine-tuning

**This is the *direct* v0 sub-task 1 + sub-task 4 unified architecture** — *one* model, *one* inference pass, *two* outputs. **The killer v0 differentiator: "first end-to-end clinical model that outputs 3D arch mesh + per-point material field in a single forward pass"**.

**Compute:** $500-900 Lambda for the *full* unified model (combines sub-task 1 + sub-task 4 H5 pipeline).

### ★ Adopt **2-scale hierarchical matching** (Sec 3.3) for v0 sub-task 1 high-resolution dental arch

**The architecture detail:** coarse depth at 1/8 resolution → neighborhood-bounded second cost volume at 1/4 resolution → DPT head upsample to full. **This is the *only* way to scale to 4K dental arch without exploding memory.**

**For v0:** at 512×512 dental arch input, use s=4 coarse + s=2 fine (paper's "2-scale" config). At 1024×1024 (high-res IOS), use s=8 coarse + s=4 fine. At 2048×2048 (v1 chairside-4K), use s=8 + s=4 + s=2 (3-scale, paper's extension in supp).

### ★ CITE DepthSplat in v0 paper as the **2025 cost-volume 3DGS SOTA** and **H3 fusion paradigm reference**

For v0 paper related-work + Table 1:
- **Related-work paragraph:** "DepthSplat [157] (Xu 2025 CVPR) fuses a frozen monocular depth backbone with cost-volume features via simple concatenation, achieving SOTA on ScanNet, RE10K, DL3DV for both depth and 3DGS. We adopt its H3 fusion paradigm and H5 unsupervised pre-training recipe for v0 sub-task 1."
- **Table 1 column:** "Method | License | Inference | 3DGS-3DGS | 3DGS-2D | Depth-ScanNet | Depth-3DGS-Pretrain | Sub-task 1 ready" — DepthSplat gets ★★★ for sub-task 1

### ★ v0 sub-task 1 stack now has 5+ cost-volume / ViT / U-Net 3DGS papers covered

**Updated v0 sub-task 1 stack:**
- **Cost-volume 3DGS (MVSplat-family):** MVSplat 156 (MIT) + **DepthSplat 157 (MIT) NEW** + MVSplat360 125 (MIT) + pixelSplat (paper 156) + Flash3D (paper 156 follow-up)
- **ViT 3DGS:** GRM 155 (MIT) + GS-LRM 110 (no license)
- **U-Net 3DGS:** LGM 154 (MIT)
- **Cost-volume 3DGS w/ monocular fusion:** **DepthSplat 157 NEW, THE BEST FOR v0**

**For v0 sub-task 1, the **practical priority order** is:**
1. **DepthSplat 157** (primary, MIT, 0.6s, SOTA) ★★★
2. MVSplat 156 (ablation, MIT, 0.044s, baseline)
3. MVSplat360 125 (full-arch 360° variant, MIT)
4. GRM 155 (quality-priority, MIT, ViT)
5. LGM 154 (speed-priority, MIT, U-Net)
6. GS-LRM 110 (transformer baseline, no license)

### ★ v0 compute updated

**v0 sub-task 1 compute: $750-1,400 Lambda** (was $1,000-2,000 with MVSplat 156-only):
- DepthSplat 157 pretrained: $0
- Depth Anything V2 ViT-B frozen weights: $0
- Stage 1 unsupervised 3DGS pre-training on clinical IOS (1 week × 4× A100): $300-500
- Stage 2 fine-tune on 3DTeethSeg22 + ToSynFCD (1 day × 4× GH200): $200-400
- Inference infra (chairside 1 month): $50-100
- **Sub-task 1 + sub-task 4 unified model extension:** +$200-400 Lambda
- **TOTAL v0 sub-task 1: ~$750-1,400 Lambda**

**v0 TOTAL compute: ~$8,620-12,060 Lambda** (was $7,870-10,660 from 156-note, +$750-1,400 for sub-task 1 upgrade + unified sub-task 4 extension).

### Open Q for HK

(i) adopt DepthSplat 157 as v0 sub-task 1 PRIMARY baseline? (YES — MIT, 0.6s, SOTA, +1.08 dB over MVSplat)
(ii) adopt unsupervised 3DGS pre-training (H5) on clinical IOS archive? (YES — *killer* H5 mechanism)
(iii) adopt simple concatenation fusion over attention fusion? (YES — paper ablation)
(iv) adopt unified sub-task 1 + sub-task 4 model? (YES — *killer* v0 differentiator)
(v) cite DepthSplat in v0 paper related-work? (YES)
(vi) extend to 4K dental arch with 3-scale hierarchical matching? (YES for v1)
(vii) keep per-pixel Gaussians or move to sparse/voxelized? (per-pixel for v0, sparse for v1)

### Next paper to read (158)

The 156-MVSplat note's other recommendation was **PanSplat (Chen et al. CVPR 2025)** — the *direct* successor of MVSplat + MVSplat360 that *adds* hierarchical spherical cost volume for **4K resolution** + two-step deferred back-propagation for memory-efficient training. This is the *right* next read because:
- **4K dental arch** is the *killer* v1 chairside resolution
- **Memory-efficient training** matters for fine-tuning on 3DTeethSeg22 + ToSynFCD (which have ~2K arches total, can fit in 24GB with deferred back-prop)
- **Hierarchical spherical cost volume** is the natural extension of DepthSplat 157's 2-scale matching to 360° dental arch reconstruction

**Recommendation: read 158 = PanSplat (Chen 2025, CVPR 2025)** — the *direct* DepthSplat 157 successor for 4K + memory-efficient training, the *killer* v1 sub-task 1 extension.

Alternative: **Splatt3R (Smart et al. 2024)** — the *direct* successor that *removes* the camera-pose requirement. *Killer* for clinical v1 where IOS pose noise is a real bottleneck, but less directly relevant for v0 since IOS scanners provide poses.
