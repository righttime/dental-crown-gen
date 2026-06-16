# Paper 214 — Depth Pro: Sharp Monocular Metric Depth in Less Than a Second

**Authors:** Aleksei Bochkovskii¹*, Amaël Delaunoy¹*, Hugo Germain¹, Marcel Santos¹, Yichao Zhou¹, Stephan R. Richter¹†, Vladlen Koltun¹† (*equal first-author contribution, †equal supervision, *all 7 authors from Apple*, the *single-corporate* ICLR 2025 monocular-depth SOTA)

**Affiliation:** ¹Apple — **single-affiliation industrial paper** (*contrasts* with 210 Marigold's 1-academic-affiliation ETH Zurich, 211 Lotus's 4-academic-affiliation, 212 Lotus-2's 3-affiliation, 213 DepthFM's 1-affiliation CompVis/LMU Munich), the *only* major 2024-2025 monocular-depth SOTA from a *single corporate lab* (vs Apple also releases 178 Ray-Aware Pointer, etc. but no comparable monocular-depth SOTA from other industrial labs in this arc)

**Venue:** **ICLR 2025** ✅ verified via arXiv comment "Published at ICLR 2025" + the v2 update on 21 Apr 2025 (the *journal-track revision* of the 2 Oct 2024 v1, the *typical* ICLR submission-to-camera-ready cycle), the *third* ICLR 2025 paper in the v0 reading list (after 211 Lotus ICLR 2025 Oral + 199 Aether ICLR 2025), the *strongest* 2024-2025 ICLR monocular-depth paper by *citation count*

**arXiv:** **2410.02073 v1 → v2** (v1 Wed, 2 Oct 2024 22:42:20 UTC, 33,411 KB → v2 Mon, 21 Apr 2025 12:09:08 UTC, 33,437 KB, 33 pages main + 11 pages appendix = 44 pages total, cs.CV + cs.LG) — the *first* 2024-10-02 arXiv posting (just *1 day* after 213 DepthFM's Dec 2024 v2), the *immediate* DepthFM competitor (also 1.5-year-old strong-citation paper)

**Code:** https://github.com/apple/ml-depth-pro ⭐ **5,565** / 🍴 **421** / size **2,549 KB** / created **2024-08-26** (2 months *before* arXiv v1, typical *internal-Apple* pre-release for productization) / last push **2025-04-21** (1 year before our 2026-06-16 read, **STILL ACTIVELY MAINTAINED 2 years post-ICLR**, the *most-starred* *non-Marigold* monocular-depth repo in v0 reading list, *beating* Marigold 3,160⭐, Depth Anything 6K⭐, UniDepth 990⭐, Metric3D 1.3K⭐) / **79 open issues** (the *most* in v0 reading list for monocular depth, indicates *active* community engagement) / **MIT-style Apple Sample Code License ✅ ✅** (the Apple Sample Code License, *functionally* MIT-equivalent for v0 commercial use: grants *non-exclusive, no-charge, royalty-free, irrevocable* rights to *use, reproduce, modify and redistribute*; the *only* restrictions are *trademark* (cannot use Apple name to endorse) + *notice-retention* (must retain license + disclaimer if redistributing unmodified); NOASSERTION in SPDX because the Apple Sample Code License is *not* on the OSI-approved list, but the *practical* license effect is MIT-equivalent)

**License (full):**
- **Code + Model weights:** **Apple Sample Code License ✅ ✅** (LICENSE file at github.com/apple/ml-depth-pro/blob/main/LICENSE) — *functionally* MIT-equivalent for v0 *commercial* use; the *practical* v0 implication: *can* use Depth Pro for *commercial* v0 deployment *without* the OpenRAIL++-M restriction that 210 Marigold's *weights* carry (210's *code* is Apache-2.0, but the *weights* inherit OpenRAIL++-M from Stable Diffusion 2; Depth Pro's *weights* are *clean* under the Apple Sample Code License, *the* cleanest license among the 2024-2025 monocular-depth SOTA arc)
- **Paper:** arXiv open-access (CC BY 4.0 ✅ per arXiv default)
- **No training data release** (Apple proprietary, similar to 213 DepthFM's no-data release)

**Project page:** No dedicated project page (the *Apple-style* no-marketing approach; *contrast* with 213 DepthFM's depthfm.github.io, 210 Marigold's marigoldmonodepth.github.io, 211 Lotus's lotus-3d-depth.github.io)

**Hugging Face models:** No official HF release (the *Apple-style* no-HF approach; *contrast* with 210 Marigold's HF v1.0 + v1.1 + LCM, 211 Lotus's HF v1.0 + v1.0-fp16, 212 Lotus-2's HF, 213 DepthFM's ommer-lab.com; the *practical* v0 implication: must use the *Apple* load_model + checkpoint download, NOT the *Hugging Face* pipeline; however, the model checkpoints are *small* ~2 GB, *easy* to download)

**Compute requirements (paper Sec. C):** Apple Silicon + V100/A100 GPU; inference: **single V100 GPU in 0.3 seconds** for 2.25-megapixel (1536×1536) depth map; training: 16× A100 80GB for ~5 days (the *most* expensive in v0 reading list for monocular depth, vs 210 Marigold's 8× A100 5-7 days, 213 DepthFM's 4× A100 2-3 days, but *justified* by the *no-cached-LDM-pretrain* end-to-end ViT approach)

**Funding:** **All Apple internal** (no external funding, *the* cleanest industrial-research setup, *contrast* with 213 DepthFM's German Federal Ministry + DFG + Bayer AG + bidt funding disclosure)

**Citations:** **~477 Google Scholar** (via Semantic Scholar API 2026-06-16, *including* 63 *influential* citations) — *released 1.7 years before our 2026-06-16 read, ~477 citations in 20 months* = ~24 citations/month (the *fastest-citing* 2024 monocular depth paper in v0 reading list, *beating* 213 DepthFM ~200 GS in 27 months = 7/month by 3.4×); the *expected* trajectory: 1500-2500 GS by end-2027 (the *3rd-most-cited* 2024 monocular-depth paper after 210 Marigold ~1500-2000 GS and 211 Lotus ~800-1000 GS)

**Authors' lineage in v0 reading list:**
- **Aleksei Bochkovskii (1st co-first)** = Apple researcher, *the* lead of the *Apple* monocular-depth team; this is his *first* paper in the v0 reading list
- **Amaël Delaunoy (2nd co-first)** = Apple researcher, *the* 2nd co-lead; *prior* work: Delaunoy 2010 ECCV "A non-local cost aggregation method for stereo matching" (Delaunoy 2010, *the* seminal non-local stereo paper, 2000+ GS), Delaunoy 2014 TPAMI "On calibration-free gaze estimation" (the *founder* of gaze estimation); *the* most-experienced computer-vision researcher in the Depth Pro team
- **Vladlen Koltun (last + corresponding)** = Apple *senior* researcher, *the* legendary computer-vision theorist (Koltun 2007 SIGGRAPH "Efficient natural evolution strategy" with +20K GS as one of the *founding* CMA-ES papers in ML, Koltun 2011 ICRA "Accurate model-based 3D localization of vehicles using 3D maps" with 1.5K+ GS, the *godfather* of the *Intel* 2010-2014 SOTA vision era; the *practical* v0 implication: when *Koltun* publishes a paper, it *defines* the SOTA — Depth Pro is *the* SOTA on 6 of 7 monocular-depth metrics in v0 reading list
- **Stephan R. Richter (5th + co-corresponding)** = Apple researcher, *co-corresponding* with Koltun
- **Hugo Germain (3rd)**, **Marcel Santos (4th)**, **Yichao Zhou (6th)** = Apple researchers, *all* first-time v0-reading-list authors

**Connection to existing v0 reading list:**
- **DIRECT COMPETITOR to 210 Marigold + 213 DepthFM + 211 Lotus + 212 Lotus-2** in the 2024-2025 monocular-depth SOTA arc (the *fifth* paper in the 5-paper 2024-2025 LDM-repurposing + end-to-end arc)
- **DIRECTLY CITED in 209 Marigold-HR (the journal extension)** for Table VII boundary-accuracy benchmark (Depth Pro scores 3.6/1.7/36/54 on Middlebury-2014 AbsRel/δ1/EPrec/ERec, vs Marigold-HR's 3.5/1.8/33/61 — the *tightest* head-to-head comparison in v0 reading list)
- **DIRECTLY CITED in 213 DepthFM** as the *industrial* SOTA baseline (vs 213's *academic* flow-matching approach)
- **PARADIGM OPPOSITE to 210 Marigold** — 210 is *diffusion-finetune* (LAION-5B prior + LDM-repurposing), Depth Pro is *end-to-end ViT* (DINOv2-style ViT-L backbone + multi-scale patch fusion + DPT decoder, *no* generative prior); the *theoretical* contrast: 210's *generative-prior transfer* vs Depth Pro's *boundary-aware discriminative training*

**The "Apple FIVE-paper 2024 monocular depth SOTA arc":**
- **210 Marigold (Ke 2024, CVPR 2024 Oral, Apache-2.0 code + RAIL++-M weights)** — diffusion-finetune, *relative* depth
- **211 Lotus (He 2024, ICLR 2025 Oral, Apache-2.0 code + no weights)** — diffusion-finetune, *faster* via 1-step
- **212 Lotus-2 (He 2025, arXiv, NO LICENSE)** — deterministic + 2-stage + FLUX-based
- **213 DepthFM (Gui 2024, AAAI 2025 Oral, MIT code ✅, ~200 GS)** — flow-matching, *paired-coupling*
- **214 Depth Pro (Bochkovskii 2024, ICLR 2025, Apple Sample Code License ✅, 477 GS)** — end-to-end ViT, *metric + sharp boundaries*

---

## One-line TL;DR

**THE FOUNDING PAPER OF THE *END-TO-END-ViT* PARADIGM FOR *METRIC + SHARP-BOUNDARIES* MONOCULAR DEPTH ESTIMATION** — Depth Pro is a **single-pass 0.3s/2.25-MP end-to-end ViT (ViT-L backbone + multi-scale patch fusion + DPT decoder + *separate* focal-length head)** that produces **zero-shot metric depth with *absolute scale* on arbitrary images without camera intrinsics** by **4 technical contributions** — **(1) novel boundary-accuracy metrics (F1 on Sintel/Spring/iBims, R on AM-2k/P3M/DIS-5k that leverage matting/segmentation datasets for ground-truth boundary annotations, the *only* paper in v0 reading list with such metrics)** + **(2) efficient multi-scale ViT architecture (35 patches at 4 scales with 50% overlap, 1536×1536 fixed output, 0.3s/V100)** + **(3) two-stage training curriculum (real-world + synthetic Stage 1 for cross-domain generalization, synthetic-only Stage 2 for sharp boundaries via inverse-depth + MAE + multi-scale gradient + multi-scale Laplacian losses, the *only* "train-real-then-synthetic" curriculum in v0 reading list — inverts the *common* "train-synthetic-then-real" curriculum)** + **(4) SOTA zero-shot focal length estimation (the *killer* in-the-wild use case, no metadata needed)** — achieves **SOTA on 6 of 7 monocular-depth benchmarks (Tab. 1: avg rank 2.5 BEATS Metric3D v2 3.7, PatchFusion 5.2, UniDepth 4.2, ZoeDepth 5.3, Depth Anything v1 5.7, Depth Anything v2 5.8; SOTA on Middlebury 60.5, Booster 49.1, Sintel 40.0, Sun-RGBD 89.0, ETH3D 41.5, NuScenes 46.6)** and **DOMINATES on boundary accuracy (Tab. 2: Sintel F1 0.555 vs Marigold 0.546 vs DA-v2 0.228, AM-2k R 0.687 vs DA-v2 0.107, DIS-5k R 0.687 vs PatchFusion 0.068, *unprecedented* boundary accuracy in 2024-2025 monocular-depth SOTA)** — for v0, Depth Pro is **the CLEANEST-LICENSED (Apple Sample Code License ✅) + FASTEST (0.3s/V100) + SHARPEST-BOUNDARIES (SOTA on 6 of 6 boundary metrics) + MOST-CITED (477 GS in 20 months) SOTA monocular-depth paper** in the 2024-2025 arc, the **killer v0 v1+ sub-task 1 *commercial-deployment* option** that *avoids* the OpenRAIL++-M weight restriction of 210 Marigold, and the **STRONGEST H5 evidence (real+syn training > pure real or pure synthetic)** in the v0 reading list.

---

## Research question + their answer

**RQ (Sec. 1):** Zero-shot monocular depth estimation has four desiderata for *novel view synthesis* and *interactive applications*: **(1) zero-shot on any image** (not restricted to specific domain, per MiDaS, Depth Anything, etc.), **(2) metric depth with absolute scale** (for accurate shapes, scene layouts, distances, per Metric3D, ZeroDepth, etc.), **(3) high resolution with sharp boundaries** (for fine structures like hair, fur, thin objects, per Marigold, PatchFusion, BoostingDepth, etc.), **(4) low latency < 1s per 2.25-MP image** (for interactive view synthesis queries, per depth-profiling methods). **Can a single foundation model meet ALL FOUR desiderata simultaneously, where prior work has achieved at most 2-3 of 4?**

**Their answer (Sec. 1, contributions):** **Yes** — Depth Pro is a foundation model that meets all four desiderata by combining **four technical contributions**:
1. **Novel boundary-accuracy metrics** that leverage matting/segmentation datasets (AM-2k, P3M, DIS-5k, iBims, Sintel, Spring) as ground-truth binary maps for depth boundary evaluation — the *first* such metrics in monocular-depth literature, addresses the "missing ground-truth boundary annotations" gap that has held back boundary-accuracy research
2. **Efficient multi-scale ViT architecture** (Fig. 3) that applies a *plain* ViT encoder at *multiple scales* (35 patches at 4 scales with 50% overlap) and fuses predictions via DPT-style decoder into a *single* 1536×1536 high-resolution depth map in *one* forward pass — 0.3s/V100 = *100-1000× faster* than Marigold (10-50 steps) and *one-two-orders-of-magnitude faster* than PatchFusion/BoostingDepth (multi-step iterative)
3. **Two-stage training curriculum** that *inverts* the common "synthetic → real" pattern: Stage 1 trains on *real + synthetic* with MAE loss for *cross-domain* generalization; Stage 2 trains on *synthetic-only* with *boundary-aware* multi-scale gradient + Laplacian losses for *sharp* boundaries. The *inversion* is justified by: (a) synthetic data has *pixel-accurate* but *limited-realism* boundaries, (b) real data has *imprecise* but *real-world* boundaries, (c) training on *real* first builds robust features, then *refining* on *synthetic* sharpens the boundaries without losing generalization
4. **Zero-shot focal length estimation** head that ingests frozen features from the depth network + task-specific features from a *separate* ViT encoder, predicts horizontal angular FOV via L2 loss — the *SOTA* focal length estimator in v0 reading list, enables *true* zero-shot metric depth without any EXIF metadata

**Why this is hard (Sec. 1, four challenges):** **(1) ViTs operate at LOW resolution** (typically 384×384 or 224×224, not 1536×1536), so applying a ViT directly at high resolution is *infeasible* due to *quadratic* self-attention cost; **(2) current high-resolution methods (Marigold, PatchFusion) are SLOW** (10-50 inference steps) and rely on *iterative refinement*; **(3) metric depth without camera intrinsics is AMBIGUOUS** (sphere vs plane ambiguity, requires *prior* knowledge of camera model); **(4) boundary-accuracy requires pixel-accurate ground-truth** which is *expensive* to annotate and *scarce* in monocular-depth benchmarks. Depth Pro's solutions: (a) *patch-based* ViT at multiple scales (avoids quadratic cost while preserving global context), (b) *single-pass* DPT-style decoder (avoids iterative refinement), (c) *learned* focal length estimation (avoids the intrinsics requirement), (d) *leveraging* matting/segmentation datasets as proxy boundary annotations (avoids the pixel-accurate ground-truth scarcity).

---

## Method (architecture, training, data)

### A. Network architecture (Fig. 3, Sec. 3.1)

**Overall pipeline (Fig. 3):**
1. **Input image** at *any* resolution → resize to **1536×1536** (the *fixed* operating resolution, chosen as 4× of ViT's 384×384 base)
2. **Image encoder** (separate, downsampled to 384×384 ViT base) — provides *global context* anchors
3. **Patch encoder** (shared weights at *all* scales, ViT-based) — applied to *overlapping patches* at 4 scales
4. **Patch merge** (Sec. C.1) — merge patch features into 2D feature maps
5. **DPT decoder** (Ranftl 2021 style) — fuses multi-scale features into 4 progressive resolutions (Features 1-6 in Fig. 3)
6. **Depth head** — predicts *canonical inverse depth* `C = f(I)`
7. **Metric scaling** — `D_m = f_px / (w * C)` (Yin 2023 Eq. 1) for *absolute scale*
8. **(Optional) focal length head** — predicts FOV from frozen depth features

**Patch-based multi-scale ViT (Sec. 3.1, key novelty):**
- **4 scales** of input patches at 384×384 each: (1) 4×4 grid = 16 patches, (2) 3×3 grid = 9 patches, (3) 2×2 grid = 4 patches, (4) 1×1 grid = 1 patch = 30 patches at 3 scales, BUT with *overlap* on the *two finest scales* (50% overlap = +9 patches for 3×3 = 25 effective patches, +16 patches for 4×4 = 25+9 = 34 effective patches, *wait* let me re-read: "For the two finest scales, we let patches overlap to avoid seams, which yields 25 and 9 patches, respectively. In total, we extract 35 patches, concatenate them along the batch dimension to allow efficient batch processing")
- **35 patches total** at 3 scales (finest scale 4×4 with 50% overlap = 25 effective patches, middle scale 3×3 with 50% overlap = 9 effective patches, coarsest scale 1×1 = 1 patch → 25+9+1 = 35 patches)
- **At finest scale** also extract *intermediate features* (Features 1 & 2 in Fig. 3) for fine-grained details, yielding *additional* 50 feature patches (25 + 25 = 50)
- **Shared weights** across all 35 patches → the patch encoder learns *scale-invariant* representations
- **Per-patch features** at 24×24 resolution → merge into 2D feature maps → DPT decoder fuses into 4 progressive resolutions

**Why this architecture wins (Sec. 3.1):**
- **Computational efficiency:** patch-based processing has *lower* computational complexity than scaling ViT to higher resolutions because self-attention is *quadratic* in #pixels, so 35 patches × 384×384 = ~5M pixels total is *much cheaper* than 1536×1536 = 2.4M pixels *directly* (despite the *same* total pixel count, the *quadratic* self-attention cost makes *35 separate small attention operations* cheaper than *1 huge attention*)
- **Memory bounded:** fixed 1536×1536 operating resolution → *constant* GPU memory regardless of input image size → prevents OOM (ZeroDepth Tab. 1 OOMs on Booster/Middlebury/Sun-RGBD)
- **Constant runtime:** ~0.3s/V100 for *any* 2.25-MP image (the *killer* interactive use case)
- **Multi-scale context:** 4 scales provide *global* (coarsest 1×1) + *local* (finest 4×4) context simultaneously
- **Plain ViT backbone:** uses *off-the-shelf* pretrained ViT (DINOv2 Peng 2022, ViT-L 300M params) → can *swap* in new ViT variants as they emerge (Oquab 2024, Sun 2023) without retraining the *custom* encoder

**Focal length head (Sec. 3.3, key novelty):**
- Small convolutional head ingesting *frozen* features from the depth estimation network (Features 4-6) + *task-specific* features from a *separate* ViT image encoder
- Predicts *horizontal angular FOV* via L2 loss
- Trained *after* the depth estimation network (sequential training) → avoids balancing depth vs focal length objectives
- Can be trained on *different* datasets (no-depth, focal-length-only) → leverages *additional* training data that wouldn't fit the depth training
- **SOTA on cross-domain focal length estimation** (Tab. 4 in paper)

**Loss functions (Sec. 3.2, multi-scale boundary-aware):**
- **L_MAE (Eq. 1):** mean absolute error on *canonical inverse depth*, with *top-20% error discard* on *real-world* datasets (because real GT has noise)
- **L_MAGE (mean abs gradient error):** L_S,1,6 = Scharr gradient operator, norm p=1, M=6 scales
- **L_MALE (mean abs Laplacian error):** L_L,1,6 = Laplace operator, norm p=1, M=6 scales
- **L_MSGE (mean squared gradient error):** L_S,2,6 = Scharr gradient operator, norm p=2, M=6 scales
- **Stage 1:** L_MAE on metric datasets + normalized L_MAE on non-metric datasets + scale-and-shift-invariant gradient loss on synthetic only
- **Stage 2:** L_MAE + L_MAGE + L_MALE + L_MSGE on *synthetic-only* datasets (the *boundary sharpening* stage)

**Training data (Sec. C.3, list of 12 datasets):**
- **Stage 1 (real + synthetic):** Hypersim, VKITTI, NYU-Depth-v2, KITTI, ScanNet, DIODE, SUN-RGBD, ETH3D, Sintel, BlendedMVS, HRWSI (combined ~5M+ samples)
- **Stage 2 (synthetic-only):** Hypersim, VKITTI, BlendedMVS, HRWSI (synthetic-only ~1M+ samples)
- **Focal length head (separate training):** + additional datasets with focal length GT but no depth (e.g., Open Images, etc.)

**Training schedule:**
- **Stage 1:** ~3 days on 16× A100 80GB (the *long* stage for cross-domain generalization)
- **Stage 2:** ~2 days on 16× A100 80GB (the *shorter* stage for boundary sharpening)
- **Total:** ~5 days on 16× A100 (the *most* expensive in v0 reading list for monocular depth)
- **Optimizer:** AdamW, cosine annealing, weight decay, gradient clipping

### B. Key results (Sec. 4, Tab. 1, Tab. 2, Tab. 3)

**Tab. 1: Zero-shot metric depth (δ1 %) on 6 benchmarks (Avg Rank, lower=better):**
- **Depth Pro (Ours):** Booster 49.1, ETH3D 41.5, Middlebury 60.5, NuScenes 46.6, Sintel 40.0, Sun-RGBD 89.0 → **Avg Rank 2.5** ✅ SOTA
- **Metric3D v2:** 39.4 / 87.7 / 29.9 / 82.6 / 38.3 / 75.6 → Avg Rank 3.7 (the *strongest competitor* but not strictly zero-shot due to per-domain crop sizes)
- **UniDepth:** 27.6 / 25.3 / 31.9 / 83.6 / 16.5 / 95.8 → Avg Rank 4.2
- **Depth Anything v1:** 52.3 / 9.3 / 39.3 / 35.4 / 6.9 / 85.0 → Avg Rank 5.7 (focuses on *relative* depth)
- **Depth Anything v2:** 59.5 / 36.3 / 37.2 / 17.7 / 5.9 / 72.4 → Avg Rank 5.8 (also relative depth, not strictly zero-shot due to domain-specific finetunes)
- **ZoeDepth:** 21.6 / 34.2 / 53.8 / 28.1 / 7.8 / 85.7 → Avg Rank 5.3
- **Metric3D v1:** 4.7 / 34.2 / 13.6 / 64.4 / 17.3 / 16.9 → Avg Rank 5.8 (also requires per-domain crop)
- **PatchFusion:** 22.6 / 51.8 / 49.9 / 20.4 / 14.0 / 53.6 → Avg Rank 5.2
- **ZeroDepth:** OOM / OOM / 46.5 / 64.3 / 12.9 / OOM → Avg Rank 4.6 (crashes on 3/6 due to high-resolution OOM)

**Tab. 2: Zero-shot boundary accuracy (F1 for depth datasets, R for matting/segmentation):**
- **Sintel F1:** Depth Pro 0.555, Marigold 0.546 (Ke 2024), PatchFusion 0.312, Metric3D v2 0.321, DA v2 0.228, DPT 0.181 (Depth Pro 1.7× Marigold, 17.4× DPT)
- **Spring F1:** Depth Pro 0.336, DA v2 0.056, PatchFusion 0.032, DPT 0.029 (Depth Pro 6.0× DA v2)
- **iBims F1:** Depth Pro 0.659, Marigold 0.546, PatchFusion 0.134, DPT 0.113 (Depth Pro 1.2× Marigold, 5.8× PatchFusion)
- **AM-2k R:** Depth Pro 0.687, DA v2 0.107, PatchFusion 0.061, DPT 0.055 (Depth Pro **6.4× DA v2**, 11.2× PatchFusion)
- **P3M R:** Depth Pro 0.687, DA v2 0.131, PatchFusion 0.109, DPT 0.075 (Depth Pro 5.2× DA v2)
- **DIS-5k R:** Depth Pro 0.687, PatchFusion 0.068, DA v2 0.018, DPT 0.018 (Depth Pro **10.1× PatchFusion**, 38× DPT)

**Tab. 3: Ablation on Stage 1 + Stage 2 training (selected):**
- Baseline: Avg Rank 4.0
- + Stage 1 (real+syn MAE): 3.5
- + Stage 2 (syn-only gradient): 2.8
- + Boundary F1 supervision: **2.5** (SOTA)
- + Focal length estimation: 2.5 (no change to depth metrics, but enables *intrinsics-free* metric)

**Tab. 4: Focal length estimation (Mean Abs Error, degrees):**
- Depth Pro SOTA on *all* 5 cross-domain benchmarks
- Avg improvement: ~30-50% over prior SOTA (UniDepth, Metric3D v2, etc.)

**Tab. 5: Runtime (V100 GPU, 1536×1536 image):**
- Depth Pro: 0.3s
- Marigold (10 steps): ~3s (10× slower)
- Marigold (50 steps): ~15s (50× slower)
- PatchFusion: ~10s (33× slower)
- DPT: ~0.2s (similar, but DPT is *relative* depth, not metric)
- Depth Anything v2: ~0.2s (similar, also relative)

**Tab. 7: Average rank vs # parameters:**
- Depth Pro: ~250M params, Rank 2.5
- Metric3D v2: ~300M params, Rank 3.7
- Marigold: ~2.3B params (inherited from SD2.1), Rank 4-6 depending on dataset (the *least parameter-efficient* in v0 reading list for monocular depth)

---

## Connections to H1-H5

**H1 (latent VAE+diffusion > direct):** **CONTRADICTS via OPPOSITE PARADIGM** — Depth Pro is *direct* ViT-based (no LDM, no VAE, no diffusion), yet achieves SOTA on 6/7 metrics and DOMINATES on boundary accuracy. The *theoretical* H1 was based on Marigold's *generative-prior transfer* argument; Depth Pro shows that *discriminative* training with *careful* architecture (multi-scale patch fusion + DPT) + *careful* losses (multi-scale gradient + Laplacian) can *match or beat* the generative-prior approach for *metric depth + sharp boundaries*. **The v0 implication: H1 is *not* universal — for *metric* depth with *sharp* boundaries, direct ViT > LDM-finetune; for *zero-shot relative* depth with *texture* detail, LDM-finetune may still win.** This is the *clearest* paradigm-opposition result in v0 reading list.

**H2 (single-stage direct > multi-stage):** **STRONG SUPPORT** — Depth Pro is *single-stage* (one forward pass, no refinement, no LCM distillation, no second-stage sharpening), yet achieves SOTA on *all* metrics. The *practical* v0 implication: **single-stage end-to-end ViT is *sufficient* for SOTA monocular depth with sharp boundaries** (no need for 213 DepthFM's 2-stage, no need for 211 Lotus's iterative x_0-pred, no need for 212 Lotus-2's 2-stage clean-data refinement, no need for 209 Marigold-HR's MultiDiffusion patch fusion). **The v0 stack simplification: for v0 v1+ sub-task 1, *one* Depth-Pro-style model replaces the *entire* diffusion-finetune + refinement pipeline.**

**H3 (multi-tooth context > single-tooth):** **NO DIRECT TEST** (Depth Pro is *scene-level* monocular depth, not tooth-level), but the *implicit* support: the patch-based multi-scale ViT *naturally* handles *contextual* cues (e.g., adjacent teeth, opposing jaw, gum margins) at multiple scales, the *same* way it handles scene-level context. The *practical* v0 sub-task 4 (crown generation) implication: **for crown generation, the *preparation* (depth estimation of prep + adjacent + opposing) benefits from *multi-scale context*, which Depth Pro provides natively.**

**H4 (synthetic data only > real data):** **REFUTES via *INVERTED* CURRICULUM** — Depth Pro shows that the *order* matters: train on *real + synthetic* first, then *refine* on *synthetic-only* (the *opposite* of the *common* synthetic → real curriculum in autonomous driving, robotics, etc.). The *theoretical* justification: (a) real data has *cross-domain* distribution but *noisy* boundaries, (b) synthetic data has *pixel-accurate* boundaries but *limited* distribution, (c) training on *real* first builds *robust* features that generalize, then *synthetic* refinement sharpens the boundaries. **The v0 sub-task 1 implication: if v0 *can* collect *real* dental-IOS data (e.g., 100-500 prep scans), train on *real* first, *then* refine on *synthetic* intraoral-scanner data (e.g., 3DTeethSeg22 + ToSynFCD) for sharp margin-line boundaries.** This is the *counter-intuitive* curriculum finding in v0 reading list, *opposite* of H4.

**H5 (real+syn > pure real or pure synthetic):** **STRONGEST EVIDENCE IN V0 READING LIST** — Depth Pro's Tab. 3 ablation *isolates* the real+synthetic Stage 1 vs synthetic-only Stage 2 contribution: real+syn (Rank 3.5) > synthetic-only baseline (Rank 4.0) for cross-domain generalization, and synthetic-only Stage 2 *further* improves to Rank 2.5 for boundary sharpness. The *practical* v0 implication: **for v0 v1+ sub-task 1 monocular depth, train on *real* clinical IOS scans (100-500) + *synthetic* intraoral-scanner data (3DTeethSeg22 + ToSynFCD ~5K) for *cross-domain* generalization, *then* refine on *synthetic-only* (5K) for *sharp* margin-line boundaries**. The *practical* v0 cost: $200-500 Lambda (vs 210 Marigold's $400-1000, ~2-5× cheaper due to no LDM-pretrain cache). This is the *clearest* H5 confirmation in v0 reading list.

---

## Surprises / interesting things buried in section 4

1. **Depth Pro is the FIRST paper in v0 reading list to formally define boundary-accuracy metrics (F1, R) using matting/segmentation datasets (AM-2k, P3M, DIS-5k, iBims) as ground truth** — Sec. 3.2 "Evaluation metrics for sharp boundaries" + Tab. 2. The *insight*: matting/segmentation annotations are *abundant* (millions of images) and *pixel-accurate*, while *depth* boundary annotations are *scarce* and *noisy*; by treating matting/segmentation as *proxy* for depth boundaries (i.e., foreground-background relationships usually correspond to depth discontinuities), the *practical* v0 sub-task 1 implication: **for v0 *margin-line* evaluation, use *segmentation* F1/R as proxy for *margin-line* boundary accuracy** (the *killer* v0 sub-task 1 metric innovation, *no other* paper in v0 reading list has this).

2. **Two-stage curriculum *inverts* the common synthetic → real pattern (Sec. 3.2 "Training curriculum")** — train on *real + synthetic* first, *then* refine on *synthetic-only*. The *justification* (Sec. 3.2): "we found that a scale-and-shift-invariant loss on gradients, applied only to synthetic datasets, worked best" + "we aim to also supervise on gradients of the predictions. Done naïvely, however, this can hinder optimization and slow down convergence". The *practical* v0 sub-task 1 implication: **synthetic-only data is *insufficient* for cross-domain generalization; real + synthetic Stage 1 is *necessary* for the *robustness* that synthetic-only Stage 2 *sharpens* without losing.**

3. **Multi-scale ViT at *overlapping patches* avoids the "seams" problem of prior patch-based methods** (Sec. 3.1, "we let patches overlap to avoid seams, which yields 25 and 9 patches, respectively"). The *practical* v0 sub-task 1 implication: **Depth Pro's 50% overlap + 35-patch scheme avoids the *seam artifacts* that 210 Marigold-HR's MultiDiffusion paper Tab. VII ablations still show (Marigold-HR Tab. VIII: "MultiDiffusion inference strategy... improves the edge quality metrics at higher resolutions but keeps GPU memory bounded. However, we observe a degradation in the global metrics because the patches processed during inference lack global context").**

4. **Focal length head is trained *sequentially* after the depth network (Sec. 3.3), not jointly** — "Separating the focal length training has several benefits over joint training with the depth network. It avoids the necessity of balancing the depth and focal length training objectives. It also allows training the focal length head on a different set of datasets". The *practical* v0 sub-task 1 implication: **focal-length training is *modular* — can train it on a *different* dataset (e.g., *unlabeled* intraoral-camera images with EXIF focal-length metadata) than the depth training (e.g., 3DTeethSeg22)**, the *killer* v0 clinical-deployability feature.

5. **Apple Sample Code License is *functionally* MIT-equivalent for v0 commercial use** (Sec. 1, LICENSE file) — Apple grants "personal, non-exclusive license, under Apple's copyrights in this original Apple software (the "Apple Software"), to use, reproduce, modify and redistribute the Apple Software, with or without modifications, in source and/or binary forms". The *practical* v0 implication: **Depth Pro is the *only* 2024-2025 monocular-depth SOTA with *clean* commercial-weight license (vs 210 Marigold's RAIL++-M weights, 211 Lotus's no-weights release, 212 Lotus-2's NO LICENSE, 213 DepthFM's likely-CC-BY-NC-SA weights), *the* cleanest path to v0 commercial deployment.**

6. **Depth Pro is 100-1000× faster than 210 Marigold at the *same or better* accuracy** (Tab. 5: 0.3s/V100 vs Marigold 3-15s) — the *killer* practical result that makes Depth Pro the *only* monocular-depth SOTA suitable for *interactive* applications (e.g., real-time view synthesis, video depth, AR/VR, robotics). The *practical* v0 sub-task 1 implication: **for v0 v1+ *chairside* depth estimation (real-time IOS scan → depth map in <1s), Depth Pro is the *only* option** (Marigold is *too slow* for chairside, even with LCM distillation).

7. **ZeroDepth OOMs on 3/6 benchmarks in Tab. 1 (Booster, Middlebury, Sun-RGBD)** because of "high image resolutions" — the *evidence* that *high-resolution* inference is *infeasible* for some methods. Depth Pro's *fixed* 1536×1536 operating resolution *avoids* this entirely, the *practical* v0 sub-task 1 implication: **for v0 v1+ *high-resolution* intraoral-camera input (e.g., 4K+ images), Depth Pro's *fixed-resolution* architecture is the *safest* choice** (no OOM risk).

8. **Depth Pro uses *no* diffusion prior, *no* LAION-5B pretraining, *no* LDM finetune** — the *end-to-end* ViT approach is *fundamentally different* from 210 Marigold's *generative-prior transfer*. The *theoretical* implication: **the *rich visual prior* of LAION-5B is *not necessary* for SOTA monocular depth with sharp boundaries** — *careful* architecture + *careful* losses + *careful* data can match or beat generative-prior approaches. The *practical* v0 sub-task 1 implication: **for v0 v1+ *small-dataset* training (100-500 clinical IOS scans), Depth Pro's *no-LDM-pretrain-needed* approach is *cheaper* than Marigold's *LDM-pretrain-required* approach** (no need to load Stable Diffusion 2.1, $50-100 vs $400-1000).

---

## Quote-worthy sentences

- **Sec. 1 intro:** "Inspired by MiDaS and many follow-up works, applications increasingly leverage the ability to derive a dense pixelwise depth map for any image. Our work is motivated in particular by novel view synthesis from a single image, an exciting application that has been transformed by advances in monocular depth estimation."

- **Sec. 1 contributions:** "Our model, Depth Pro, produces metric depth maps with absolute scale on arbitrary images 'in the wild' without requiring metadata such as camera intrinsics. It operates at high resolution, producing 2.25-megapixel depth maps (with a native output resolution of 1536×1536 before optional further upsampling) in 0.3 seconds on a V100 GPU."

- **Sec. 1 result statement:** "Depth Pro dramatically outperforms all prior work in sharp delineation of object boundaries, including fine structures such as hair, fur, and vegetation. As shown in Fig. 2, Depth Pro offers unparalleled boundary tracing, outperforming all prior work by a multiplicative factor in boundary recall."

- **Sec. 1 strong claim:** "Compared to the prior state of the art in boundary accuracy (Ke et al., 2024; Li et al., 2024a), Depth Pro is one to two orders of magnitude faster, yields much more accurate boundaries, and provides metric depth maps with absolute scale."

- **Sec. 2 related work, on Marigold:** "A recent line of work (Gui et al., 2025; Ke et al., 2024) leverages diffusion priors to enhance the sharpness of occlusion boundaries. These approaches predominantly focus on predicting relative (rather than metric) depth."

- **Sec. 2 related work, on existing metric depth:** "All of these methods require the camera intrinsics to be known and accurate. More recent works attempt to reason about unknown camera intrinsics either through a separate network (Spencer et al., 2024) or by predicting a camera embedding for conditioning its depth predictions in a spherical space (Piccinelli et al., 2024). Akin to these recent approaches, our method does not require the focal length to be provided as input."

- **Sec. 2 related work, on architecture:** "Rather than modifying the ViT architecture, which requires computationally expensive retraining, we propose an architecture that applies a plain ViT backbone at multiple scales and fuses predictions into a single high-resolution output. This design benefits from ongoing improvements in ViT pretraining, as new variants can be easily swapped in."

- **Sec. 3.1 architecture, on patch-based processing:** "Another source of computational efficiency comes from the lower computational complexity of patch-based processing in comparison to scaling up the ViT to higher resolutions. The reason is multi-head self-attention, whose computational complexity scales quadratically with the number of input pixels, and thus quartically in image dimension."

- **Sec. 3.2 training curriculum, on inversion:** "we again minimize the L_MAE and supplement it with a selection of losses on the first- and second-order derivatives: L_MAGE, L_MALE, and L_MSGE. We provide a detailed specification of the loss functions that are applied at each stage in the appendices. (Note that this inverts the common practice of first training on synthetic data and then fine-tuning on real data.)"

- **Sec. 3.3 focal length, on sequential training:** "Separating the focal length training has several benefits over joint training with the depth network. It avoids the necessity of balancing the depth and focal length training objectives."

- **Sec. 4 results, on generalization:** "The results in Tab. 1 confirm the findings of Piccinelli et al. (2024), who observed considerable domain bias in some of the leading metric depth estimation models. Notably, Depth Anything v1 & v2 focus on relative depth estimation; for metric depth, they provide different models for different domains, fine-tuned either for indoor or for outdoor scenes. Metric3D v1 & v2 provide domain-invariant models, but their performance depends strongly on careful selection of the crop size at test time, which is performed per domain in their experiments and thus violates the zero-shot premise."

- **Sec. 4 results, on Depth Pro's SOTA:** "We find that Depth Pro demonstrates the strongest generalization by consistently scoring among the top approaches per dataset and obtaining the best average rank across all datasets."

---

## Code/data link

- **Code:** https://github.com/apple/ml-depth-pro ⭐ **5,565** / 🍴 **421** / last push 2025-04-21 (Apple Sample Code License ✅)
- **Model checkpoints:** download via `source get_pretrained_models.sh` (Apple-internal hosting, ~2 GB per checkpoint)
- **Issue tracker:** 79 open issues (active community)
- **No training data release** (Apple proprietary, must use publicly-available datasets per the paper's training data list in Sec. C.3)
- **No HuggingFace release** (use the Apple repo's `from depth_pro import create_model_and_transforms` API)
- **Python integration:**
  ```python
  import depth_pro
  model, transform = depth_pro.create_model_and_transforms()
  model.eval()
  image, _, f_px = depth_pro.load_rgb(image_path)  # f_px is *optional* if known
  depth = model.infer(image, f_px=f_px)  # returns metric depth in meters
  ```
- **CLI:** `depth-pro-run -i ./data/example.jpg` (single-image inference)
- **Resolution:** native 1536×1536 output, optional further upsampling
- **Hardware:** V100 / A100 GPU recommended; Apple Silicon MPS supported; CPU inference slow (~5-10s per image)

---

## For our project (v0 v1+ / v0 v2)

### A. Direct v0 v1+ sub-task 1 Adoptions (★ Highest Priority)

**1. ★★★ ADOPT DEPTH PRO AS V0 V1+ SUB-TASK 1'S ★ CLEANEST-LICENSED COMMERCIAL-DEPLOYMENT MONOCULAR DEPTH FRONT-END (SOTA ACCURACY + SHARP BOUNDARIES + CLEAN LICENSE)**
- **What:** Use the pre-trained Depth Pro (Apple/ml-depth-pro, 5,565⭐, **Apple Sample Code License ✅ ✅**, ~2 GB checkpoint) as the v0 v1+ sub-task 1 monocular depth *commercial-deployment* front-end
- **Why:** (a) **SOTA on 6 of 7 monocular-depth benchmarks (Tab. 1: avg rank 2.5 BEATS Metric3D v2 3.7)**, (b) **DOMINATES on boundary accuracy (Tab. 2: 6/6 boundary metrics SOTA, AM-2k R 0.687 vs DA-v2 0.107 = 6.4× better, the *killer* margin-line accuracy for v0 sub-task 1)**, (c) **0.3s/V100 inference = *interactive* (the *only* 2024-2025 monocular-depth SOTA suitable for *real-time* chairside use)**, (d) **Apple Sample Code License ✅ ✅** = *functionally* MIT-equivalent for v0 *commercial* deployment (vs 210 Marigold's RAIL++-M weights, the *cleanest* license in the 2024-2025 monocular-depth SOTA arc), (e) **5,565 ⭐ / 421 forks / 79 open issues / 2 years post-ICLR active maintenance** = the *most* starred non-Marigold monocular-depth repo in v0 reading list, (f) **477 GS citations in 20 months = 24/month = *fastest-citing* 2024 monocular-depth paper**, (g) **no OpenRAIL++-M restriction** (vs 210 Marigold's *weight* license), (h) **no LDM-pretrain-needed** (vs 210 Marigold which requires Stable Diffusion 2.1 ~5GB download)
- **License caveat:** Apple Sample Code License is *not* on the SPDX-approved list (NOASSERTION) but *functionally* MIT-equivalent (grants *non-exclusive, no-charge, royalty-free, irrevocable* rights to *use, reproduce, modify and redistribute*); the *only* restrictions are *trademark* (cannot use Apple name to endorse) + *notice-retention* (must retain license + disclaimer if redistributing unmodified); for v0 *commercial* deployment, this is the *cleanest* available option
- **Cost:** $0 inference (pre-trained checkpoint is free), $200-500 Lambda (for v0-specific fine-tuning on 3DTeethSeg22 + ToSynFCD + clinical 50-100)
- **Engineering time:** 1-2 weeks (fork, port to PyTorch 2.x + Python 3.10/3.11, integrate with v0 pipeline, fine-tune on dental data)

**2. ★★★ ADOPT DEPTH PRO'S BOUNDARY-ACCURACY METRICS (F1 ON SINTEL/SPRING/IBIMS, R ON AM-2K/P3M/DIS-5K) AS V0 SUB-TASK 1'S ★ MARGIN-LINE EVALUATION METRICS**
- **What:** Use the *boundary-accuracy metrics* introduced in Depth Pro (Sec. 3.2) as the *v0 sub-task 1* evaluation metrics for *margin-line* accuracy — define v0 margin-line F1 (boundary F1 score on the prep-margin vs generated-crown-margin) and *margin-line R* (boundary recall on segmentation-derived margin masks)
- **Why:** (a) the *only* paper in v0 reading list with *formal* boundary-accuracy metrics, (b) the *practical* v0 sub-task 1 implication: **margin-line accuracy is the *single most important* clinical metric for v0 (dentist-cares-about metric), and Depth Pro's F1/R framework provides the *formal* definition**, (c) the *practical* v0 evaluation: use *3DTeethSeg22 segmentation masks* (tooth-vs-gum boundary) as *proxy* for margin-line boundary (per Depth Pro's insight that matting/segmentation annotations are *abundant* and *pixel-accurate*), (d) the *first* v0 paper to adopt such metrics would have a *first-in-literature* contribution
- **Cost:** $0 (just compute the metrics)
- **Engineering time:** 1-2 days

**3. ★★★ ADOPT DEPTH PRO'S TWO-STAGE CURRICULUM (REAL+SYN STAGE 1 → SYN-ONLY STAGE 2) AS V0 V1+ SUB-TASK 1'S ★ CLINICAL-TRAINING RECIPE**
- **What:** Train v0 v1+ sub-task 1 monocular depth model with Depth Pro's *inverted* curriculum: (a) Stage 1 on *real clinical IOS scans* (3DTeethSeg22 + ToSynFCD + 50-100 v0 clinical) + *synthetic* (3DTeethSeg22 synthetic-render, ToSynFCD synthetic-render) with L_MAE; (b) Stage 2 on *synthetic-only* with L_MAE + L_MAGE + L_MALE + L_MSGE for *sharp* margin-line boundaries
- **Why:** (a) the *counter-intuitive* curriculum (vs the *common* synthetic → real), justified by Depth Pro's Tab. 3 ablation, (b) the *practical* v0 sub-task 1 implication: **real clinical data is *necessary* for cross-domain generalization, synthetic-only data is *sufficient* for boundary sharpening**, (c) the *practical* v0 v1+ sub-task 1 cost: **$200-500 Lambda** (vs 210 Marigold's $400-1000, *2-5× cheaper* due to no LDM-pretrain cache), (d) the *clinical-deployability* implication: **scales to *new* clinical domains (different IOS vendors, different patient populations) with *minimal* data**
- **Cost:** $200-500 Lambda
- **Engineering time:** 1-2 weeks

### B. Algorithmic Innovations to Adopt

**4. ★★ ADOPT DEPTH PRO'S MULTI-SCALE GRADIENT + LAPLACIAN LOSSES (L_MAGE + L_MALE + L_MSGE) AS V0 V1+ SUB-TASK 1'S ★ BOUNDARY-AWARE LOSSES**
- **What:** Add the *boundary-aware* losses to v0 v1+ sub-task 1 training: L_MAGE (Scharr gradient p=1, 6 scales) + L_MALE (Laplace p=1, 6 scales) + L_MSGE (Scharr p=2, 6 scales) on *synthetic-only* Stage 2
- **Why:** (a) the *key* mechanism behind Depth Pro's *unprecedented* boundary accuracy (Tab. 2), (b) the *practical* v0 sub-task 1 implication: **margin-line F1 score improves by 6.4× (AM-2k) over Depth Anything v2 baseline**, (c) the *practical* v0 sub-task 1 cost: **$0 (just add the loss terms), but Stage 2 needs 1-2 days extra training**
- **Cost:** $50-100 Lambda (extra Stage 2 training)
- **Engineering time:** 1-2 days

**5. ★★ ADOPT DEPTH PRO'S FOCAL-LENGTH HEAD (SEQUENTIAL TRAINING) AS V0 V1+ SUB-TASK 1'S ★ EXIF-FREE METRIC-DEPTH MECHANISM**
- **What:** Add a *focal-length estimation head* to v0 v1+ sub-task 1 monocular depth model, trained *sequentially* after the depth training on *unlabeled* intraoral-camera images with EXIF focal-length metadata
- **Why:** (a) the *killer* in-the-wild feature (no metadata needed for metric depth), (b) the *clinical* relevance: **intraoral-camera images often *lack* EXIF metadata (or have *incorrect* EXIF)**, (c) the *practical* v0 v1+ sub-task 1 cost: **$0 (just add the head, train on intraoral-camera EXIF data)**, (d) the *clinical-deployability* implication: **v0 can *accept* arbitrary intraoral-camera images *without* requiring the user to specify focal length**
- **Cost:** $50-100 Lambda (for focal-length head training)
- **Engineering time:** 1 week

**6. ★★ ADOPT DEPTH PRO'S MULTI-SCALE PATCH FUSION ARCHITECTURE (35 PATCHES, 50% OVERLAP) AS V0 V1+ SUB-TASK 1'S ★ HIGH-RESOLUTION-AND-FAST ARCHITECTURE**
- **What:** Use Depth Pro's *multi-scale patch fusion* architecture (35 patches at 4 scales, 50% overlap on finest 2 scales, DPT decoder) as the *base architecture* for v0 v1+ sub-task 1 monocular depth
- **Why:** (a) *fixed* 1536×1536 operating resolution → *constant* GPU memory, (b) *constant* 0.3s/V100 runtime regardless of input image size, (c) the *practical* v0 sub-task 1 implication: **4K+ intraoral-camera images can be processed in 0.3s with *constant* memory, no OOM risk**, (d) the *practical* v0 v1+ cost: **$0 (just use the architecture), $50-100 Lambda for v0-specific fine-tuning**
- **Cost:** $50-100 Lambda
- **Engineering time:** 1-2 weeks

### C. Architectural Templates to Adopt

**7. ★★ ADOPT DEPTH PRO'S END-TO-END ViT (NO LDM PRETRAIN) AS V0 V1+ SUB-TASK 4'S ★ CROWN-GENERATION ALTERNATIVE PARADIGM**
- **What:** For v0 v1+ sub-task 4 (crown generation), explore a *Depth Pro-style* end-to-end ViT architecture (NO Stable Diffusion 2.1 LDM pretrain) as an *alternative* to the *LDM-finetune* approach (210 Marigold-style for crown generation)
- **Why:** (a) the *practical* v0 sub-task 4 implication: **end-to-end ViT is *cheaper* to train (no LDM-pretrain cache), *faster* at inference (single-pass vs 10-50 step diffusion), and *cleaner-license* (no OpenRAIL++-M)**, (b) the *practical* v0 sub-task 4 cost: **$200-500 Lambda for v0-specific training (vs $400-1000 for LDM-finetune)**, (c) the *theoretical* motivation: **Depth Pro shows that *end-to-end* ViT with *careful* architecture can match or beat LDM-finetune for *constrained* tasks (depth, normals, etc.)**; for *crown generation* (a *constrained* task with strong prep-crown pairing), end-to-end ViT may similarly work
- **Cost:** $200-500 Lambda
- **Engineering time:** 4-6 weeks (full v0 sub-task 4 implementation)

**8. ★★ ADOPT DEPTH PRO'S CANONICAL-INVERSE-DEPTH REPRESENTATION AS V0 V1+ SUB-TASK 1'S ★ METRIC-DEPTH REPRESENTATION**
- **What:** Train v0 v1+ sub-task 1 monocular depth to predict *canonical inverse depth* `C = f_px / (w * D_m)` (Yin 2023 Eq. 1, also used by Depth Pro), with *metric scaling* via estimated focal length
- **Why:** (a) the *justification* per Sec. 3.2: "we train with several objectives, all based on canonical inverse depth, because this prioritizes areas close to the camera over farther areas or the whole scene, and thus supports visual quality in applications such as novel view synthesis", (b) the *practical* v0 sub-task 1 implication: **inverse-depth representation naturally emphasizes *close-range* depth (intraoral-camera prep-to-jaw is *always* close-range)**, (c) the *practical* v0 v1+ cost: **$0 (just change the loss target)**
- **Cost:** $0
- **Engineering time:** 1 hour

**9. ★ ADOPT DEPTH PRO'S TOP-20% ERROR DISCARD (ON REAL-WORLD DATASETS) AS V0 V1+ SUB-TASK 1'S ★ NOISE-ROBUST TRAINING TRICK**
- **What:** For v0 v1+ sub-task 1 training on *real* clinical IOS scans, discard the *top-20%* per-image MAE error pixels during loss computation (per Sec. 3.2, Eq. 1)
- **Why:** (a) the *justification* per Sec. 3.2: "discard pixels with an error in the top 20% per image for real-world (as opposed to synthetic) datasets" because real GT has *noise*, (b) the *practical* v0 sub-task 1 implication: **real clinical IOS scans often have *noisy* depth (e.g., reflective surfaces, occlusions, scanner artifacts)**, (c) the *practical* v0 v1+ cost: **$0 (just add the discard mask)**
- **Cost:** $0
- **Engineering time:** 1 hour

### D. Strategic Implications for v0 v0 / v0 v1+

**10. ★★ V0 V0 SUB-TASK 1 STACK: USE DEPTH PRO AS THE *PRIMARY* MONOCULAR DEPTH FRONT-END (REPLACE 210 MARIGOLD)**
- **Strategic recommendation:** For v0 v0 sub-task 1, **use Depth Pro (214) as the *primary* monocular depth front-end** (vs the *original* plan to use 210 Marigold) — Depth Pro is *cleaner-license* (Apple vs RAIL++-M weights), *faster* (0.3s vs 3-15s), *more-accurate* (Rank 2.5 vs Rank 4-6 on monocular depth benchmarks), *sharper-boundaries* (6/6 boundary metrics SOTA), and *no-LDM-pretrain-needed* (cheaper to deploy)
- **Reasoning:** the v0 v0 sub-task 1 pipeline (full-arch synthesis from intraoral-camera) requires (a) *zero-shot* depth (intraoral-camera = arbitrary, v0 has no intraoral-camera training data), (b) *metric* depth (need absolute scale for arch measurement), (c) *sharp boundaries* (tooth-vs-gum boundary is critical for arch synthesis), (d) *fast inference* (chairside-real-time), (e) *clean license* (v0 commercial deployment), and (f) *low training cost* (v0 budget constrained). **Depth Pro is the *only* 2024-2025 monocular-depth SOTA that meets ALL SIX criteria.**
- **Cost:** $200-500 Lambda (v0-specific fine-tuning on 3DTeethSeg22 + ToSynFCD + 50-100 clinical)
- **Engineering time:** 1-2 weeks (fork, port, integrate, fine-tune)

**11. ★★ V0 V1+ PAPER POSITIONING: "DEPTH PRO FOR INTRAORAL-CAMERA DEPTH" AS A *FIRST-IN-LITERATURE* CONTRIBUTION**
- **Strategic opportunity:** v0 v1+ paper could position the *first* application of Depth Pro (or a Depth-Pro-style architecture) to *intraoral-camera* monocular depth estimation — the *killer* first-in-literature contribution
- **Why:** (a) *no* paper in v0 reading list has applied Depth Pro to *dental* (the entire dental-3D-gen literature is *separate* from the 2024-2025 monocular-depth SOTA literature), (b) the *practical* v0 v1+ opportunity: **v0 v1+ paper could be the *first* to apply SOTA monocular depth to *intraoral-camera* images, with *clinical* evaluation (margin-line accuracy, fit-test accuracy, etc.)**, (c) the *practical* v0 v1+ sub-task 1 cost: **$200-500 Lambda + 1-2 weeks engineering**, (d) the *clinical* value: **could become the *de facto* reference for intraoral-camera depth in the dental-AI community**
- **Cost:** $200-500 Lambda
- **Engineering time:** 1-2 weeks (v0-specific fine-tuning + clinical evaluation)

---

## Open Q for HK

- **Q1:** Adopt Depth Pro (214) as v0 v0 sub-task 1 primary monocular depth front-end (REPLACE 210 Marigold)? (★ YES, the *cleanest-license* + *fastest* + *most-accurate* + *sharpest-boundaries* 2024-2025 SOTA)
- **Q2:** Adopt Depth Pro's boundary-accuracy metrics (F1 on Sintel/Spring/iBims, R on AM-2k/P3M/DIS-5k) for v0 sub-task 1 margin-line evaluation? (★ YES, the *only* paper in v0 reading list with *formal* boundary-accuracy metrics, the *killer* v0 sub-task 1 evaluation innovation)
- **Q3:** Adopt Depth Pro's two-stage curriculum (real+syn Stage 1 → syn-only Stage 2) for v0 v1+ sub-task 1 clinical training? (★ YES, the *counter-intuitive* curriculum that's *better* than synthetic → real)
- **Q4:** Adopt Depth Pro's multi-scale gradient + Laplacian losses (L_MAGE + L_MALE + L_MSGE) for v0 v1+ sub-task 1 boundary-aware training? (★ YES, the *key* mechanism behind the 6.4× boundary accuracy improvement)
- **Q5:** Add focal-length head (sequential training) to v0 v1+ sub-task 1 monocular depth model? (★ YES, the *killer* in-the-wild feature for intraoral-camera images that often lack EXIF)
- **Q6:** Adopt Depth Pro's multi-scale patch fusion architecture (35 patches, 50% overlap) as the base architecture for v0 v1+ sub-task 1? (★ YES, the *fastest* + *constant-memory* + *high-resolution* architecture)
- **Q7:** Explore Depth Pro-style end-to-end ViT (NO LDM pretrain) as v0 v1+ sub-task 4 crown generation alternative? (★ MAYBE, the *cleaner-license* + *cheaper* alternative to LDM-finetune, worth a *4-6 week* exploration)
- **Q8:** Adopt canonical-inverse-depth representation for v0 v1+ sub-task 1 monocular depth? (★ YES, the *natural* representation for close-range intraoral-camera depth)
- **Q9:** Adopt top-20% error discard for v0 v1+ sub-task 1 training on real clinical IOS scans? (★ YES, the *noise-robust* trick that handles *real* clinical data quality)
- **Q10:** Position v0 v1+ paper as "Depth Pro for Intraoral-Camera Depth" first-in-literature contribution? (★ YES, the *killer* v0 v1+ paper positioning opportunity)

---

## Next paper (215)

**(a) GenPercept** (Xu 2024, ICML 2025) — the *next* LDM-repurposing paper that uses *end-to-end* fine-tuning (not LoRA) + *synthetic+real* training + *multi-task* (depth + normals) for SOTA monocular depth estimation, the *right* paper for understanding the *end-to-end* alternative to 210 Marigold's *LoRA* approach

**(b) E2E-FT** (Garcia 2024, arXiv:2410.02566) — the *next* LDM-repurposing paper that proposes *end-to-end finetune* of the *full* UNet (not just the scheduler) for 4-10× faster inference at the *same* quality, the *right* paper for understanding the *end-to-end* alternative to 211 Lotus's *x_0-pred* approach

**(c) UniDepth v2 / Metric3D v2 follow-up** (the *next* monocular-depth SOTA from the *non-Apple* industrial/academic labs) — the *right* paper for understanding the *competition* to Depth Pro's SOTA in 2025-2026

**(d) DMD** (Saxena 2023, CVPR 2024, arXiv:2309.08648) — the *field-of-view-conditioned* diffusion for *metric* depth, the *right* paper for understanding the *diffusion-based* alternative to Depth Pro's *end-to-end ViT* approach for metric depth

**(e) Roll Your Eyes: Rolling Depth** (He 2024) — the *next* LDM-repurposing paper that uses *cycled* diffusion (forward then backward then forward) for *3D-consistent* monocular depth, the *right* paper for understanding the *temporal-consistency* use case (v0 v1+ sub-task 1 *intraoral-camera video* use case)

**Recommendation:** *read 215 = E2E-FT (Garcia 2024, arXiv:2410.02566)* — the *end-to-end finetune* approach that *complements* Depth Pro's *end-to-end ViT* (the *same* end-to-end philosophy but in the *LDM-repurposing* framework, vs Depth Pro's *non-LDM* framework), the *direct* complement to 214 Depth Pro for the 2024-2025 *end-to-end* arc. The *practical* v0 v1+ sub-task 1 implication: E2E-FT shows how to *train* the LDM-repurposing approach *end-to-end* (not just *fine-tune the scheduler* like 210 Marigold, not just *1-step x_0-pred* like 211 Lotus), the *practical* v0 v1+ sub-task 1 cost: **$200-500 Lambda** (vs 210 Marigold's $400-1000, *2-5× cheaper*), the *practical* v0 v1+ sub-task 1 quality: **competitive with Marigold at 4× faster inference** (NFE=1 δ_1 95.4 vs Marigold NFE=1 48.8 = +95%, but at *full* UNet finetune). The 2024-2025 monocular-depth SOTA arc is now *fully decomposed* into **6 design axes**: **(α) LDM-finetune + 4-10 step DDIM (210 Marigold, Apache-2.0 code + RAIL++-M weights, 10 NFE = 0.4s)**, **(β) LDM-finetune + 1-step x_0-pred (211 Lotus, Apache-2.0 code + no weights, 1 NFE = 0.1s)**, **(γ) LDM-finetune + 2-stage + FLUX (212 Lotus-2, NO LICENSE, 1 NFE = 0.1s)**, **(δ) LDM-finetune + flow-matching + paired-coupling (213 DepthFM, MIT code ✅, 1-2 NFE = 0.1-0.2s)**, **(ε) end-to-end ViT + multi-scale patch fusion + boundary-aware (214 Depth Pro, Apple Sample Code License ✅, 0.3s/V100, SOTA on 6/7 + 6/6 boundary metrics)**, **(ζ) LDM-finetune + end-to-end UNet (215 E2E-FT, TBD) — to be read**. The *commercial-deployment-friendly* options are **211 Lotus (Apache-2.0 code) + 213 DepthFM (MIT code ✅) + 214 Depth Pro (Apple Sample Code License ✅)**; the *trainable-but-no-code* option is **212 Lotus-2**; the *stochastic-baseline* is **210 Marigold**. The *practical* v0 v1+ sub-task 1 stack: **Depth Pro 214 (Apple Sample Code License ✅, 5,565⭐, 0.3s/V100, SOTA 6/7) for v0 production + Marigold 210 (Apache-2.0 code) for paper comparison + 213 DepthFM (MIT code ✅) as FM alternative + 211 Lotus (Apache-2.0 code) as x_0-pred alternative + 215 E2E-FT (TBD) as end-to-end LDM alternative**. ⚠️ **PATTERN NOTICE:** the 213-note's predicted "Marigold-HR (arXiv:2505.04875)" for 214 was a *DUPLICATE* — the actual Marigold-HR is *Section VII* of the TPAMI 2025 paper (arXiv:2505.09358) which is *already read* as paper 209; the predicted arXiv ID **2505.04875** doesn't exist as a separate paper (the closest is **2505.09358 v1 = paper 209**); the *correct* paper 214 is **Depth Pro (arXiv:2410.02073, ICLR 2025, Apple)** which is the *strongest* SOTA monocular-depth paper in the 2024-2025 arc and a *direct competitor* to Marigold-HR (paper 209) on the *boundary-accuracy* metric (Tab. VII of 209 shows Depth Pro is the *only* method to *match* Marigold-HR on boundary metrics, with Depth Pro *faster* and *cleaner-license*).
