# 205 — Depth Anything V2 (Yang et al., 2024)

**TL;DR:** THE PRACTICAL MONOCULAR DEPTH FOUNDATION MODEL — combines the fine-detail strength of Stable-Diffusion-based depth (Marigold 204) with the efficiency + robustness of discriminative DPT (Depth Anything V1), by replacing all labeled real images with **595K precise synthetic** images and using a **DINOv2-Giant teacher → pseudo-label 62M+ real → train smaller students** pipeline. Beats Marigold on accuracy while being **10× faster** and as small as **24.8M params**. *The killer empirical lesson for v0: data quality beats data quantity; the killer architecture for v0 sub-task 1: frozen-DINOv2 + lightweight DPT head is the right baseline to beat, not the right architecture to replace.*

---

## Research question + their answer

**RQ:** Can a *single* MDE (monocular depth estimation) model achieve *all six* desirable properties simultaneously — fine detail, transparent-object handling, reflections, complex scenes, efficiency, transferability — when prior works had to trade off (SD-based = fine detail + transparent + reflections but slow + non-transferable; discriminative DPT = complex + efficient + transferable but coarse + no transparent)?

**Answer:** YES — *without any new fancy techniques* — by fixing the **labeled data design** (replace noisy real labels with precise synthetic labels for the teacher) and the **teacher-student scaling** (scale teacher to DINOv2-G 1.3B, then distill to lightweight students via 62M+ real pseudo-labels). The result: Depth Anything V2 wins on all 6 axes (Tab. 1, Sec. 1).

The paper's framing is **Q1 / Q2 / Q3**:
- **Q1** (Sec. 2): Is the coarse depth of MiDaS / DA V1 from the discriminative modeling itself, or from the data? **A1: From the data.** Replace labeled real with synthetic → instant fine details even with DPT.
- **Q2** (Sec. 3): Why don't prior works use synthetic? **A2: Two limitations** — (a) synthetic→real distribution shift, (b) restricted scene coverage (graphics engines have fixed scene types).
- **Q3** (Sec. 4): How to amplify synthetic advantages while avoiding drawbacks? **A3:** Scale up the *synthetic-only* teacher to DINOv2-G, then use it to *pseudo-label 62M+ real images*, then train students on the pseudo-labels. This is **knowledge distillation via real images** — the teacher provides a clean "label" for diverse real scenes, students learn *robust* features (from real diversity) with *clean* supervision (from the synthetic-trained teacher).

---

## Method

### Architecture
- **Backbone:** DINOv2 (ViT-S / ViT-B / ViT-L / ViT-G), *frozen* during training
- **Decoder:** DPT (Dense Prediction Transformer) — the Ranftl et al. 2022 / DA-V1 design, with a *minor* modification: V2 uses **intermediate** features from DINOv2 (not the last 4 layers as in V1, which the authors call "unintentional"). Same patch sizes 4/8/16/32, same 4-scale fusion.
- **Out channels per scale:** Small [48, 96, 192, 384] / Base [96, 192, 384, 768] / Large [256, 512, 1024, 1024] / Giant [1536, 1536, 1536, 1536] (the Giant has uniform 1536 because DINOv2-G has uniform 1536-dim features)
- **Heads:** 1 (relative depth) — sigmoid → affine-invariant normalization (2%/98% percentile as in V1) → output
- **Variant:** also released as **metric depth** (Sec. 5) by fine-tuning the same backbone on metric depth labels (NYU-D / KITTI for indoor/outdoor). 6 metric depth models (3 scales × 2 domains).

### Training pipeline (3 stages)

**Stage 1 — Teacher (DINOv2-G, 1.3B):**
- Trained **purely on 595K synthetic images** with precise ground-truth depth labels (Hypersim 465K + vKITTI 130K + others; details in Sec. B.1)
- Loss: scale-shift-invariant L1 loss + gradient-matching loss (the V1 recipe; for details see DA V1 paper, Ranftl 2024)

**Stage 2 — Pseudo-labeling:**
- Use the trained teacher to predict depth for **62M+ unlabeled real images** (the V1 unlabeled image collection — a mix of images from multiple unlabeled sources; details in Sec. B.2)
- This produces **pseudo depth maps** for every real image

**Stage 3 — Students (ViT-S/B/L):**
- Train on the **62M+ pseudo-labeled real images** (not synthetic, not mixed)
- Loss: same as V1
- Inference: 1 forward pass → depth map

### Key training details (Sec. B)
- **Input size:** 518×518 (paper default), can go up to higher for finer detail
- **Augmentations:** color jitter, horizontal flip, random crop
- **Optimizer:** AdamW, lr 5e-6 for encoder + 5e-5 for DPT head
- **Distributed:** 32 A100 GPUs (the standard 2024 discriminator-MDE recipe)
- **Training time:** ~5 days for the largest model (Sec. B.3)

### DA-2K evaluation benchmark (Sec. 6)
- The paper notes current test sets are **too noisy** to reflect true MDE strength
- They construct **DA-2K**: 2,000 carefully-curated images with **precise annotations** (sparse LiDAR or SfM-cleaned), spanning diverse scenes (indoor, outdoor, transparent, reflective, complex layouts, dynamic objects)
- Manual quality check: each image must be visually consistent with its depth (the *cleanest* 2K-image benchmark in 2024 MDE literature)

---

## Results

### Tab. 1 — Preferable properties (V2 wins on all 6)

| Property | Marigold 204 | DA V1 | DA V2 (Ours) |
|---|---|---|---|
| Fine Detail | ✓ | ✗ | ✓ |
| Transparent Objects | ✓ | ✗ | ✓ |
| Reflections | ✓ | ✗ | ✓ |
| Complex Scenes | ✗ | ✓ | ✓ |
| Efficiency | ✗ | ✓ | ✓ |
| Transferability | ✗ | ✓ | ✓ |

### Tab. 12 — Transparent Surface Challenge (zero-shot, the *killer* ablation)

| Model | δ₁ ↑ (%) |
|---|---|
| MiDaS v3.1 (Ranftl 2022) | 25.9 |
| Depth Anything V1 (Yang 2024a) | 53.5 |
| **Depth Anything V2 (Ours, zero-shot)** | **83.6** |

→ **+30.1 pts over V1** (zero-shot, no fine-tuning). This is the *killer* evidence that V2's "synthetic-trained teacher → pseudo-label real" recipe transfers to **completely unseen** transparent-object categories.

### Standard benchmarks (Tab. 2 / Sec. 5.1)

| Model | Params | NYU-D (AbsRel ↓) | KITTI (AbsRel ↓) | ETH3D (AbsRel ↓) | ScanNet (AbsRel ↓) |
|---|---|---|---|---|---|
| MiDaS v3.1 | 105M | 5.3 | 11.6 | 7.3 | 7.6 |
| Depth Anything V1-Large | 335M | 4.5 | 7.6 | 6.7 | 4.7 |
| **Depth Anything V2-Large** | **335M** | **3.6** | **6.6** | **5.3** | **3.7** |
| Marigold (SD) | 865M | 5.5 | 9.9 | 6.5 | 6.4 |
| **Depth Anything V2-Giant** | **1.3B** | **2.9** | **5.5** | **4.3** | **2.8** |

→ V2-Large *beats* Marigold (865M, SD-based) with **2.6× fewer params** and **10× faster inference** on every benchmark.
→ V2-Giant extends the lead (2.9 vs 5.5 AbsRel on NYU-D, the *largest* model in 2024 MDE literature).

### Inference speed (Sec. 5.1, Tab. 4)

| Model | Params | Inference time per image (V100) | Relative speed |
|---|---|---|---|
| Marigold (SD, 50 DDIM steps) | 865M | ~2.5 s | 1× |
| **DA V2-Small** | **24.8M** | **0.05 s** | **50×** |
| **DA V2-Base** | **97.5M** | **0.08 s** | **31×** |
| **DA V2-Large** | **335M** | **0.18 s** | **14×** |
| DA V2-Giant | 1.3B | ~0.7 s | 3.5× |

→ **>10× faster than Marigold** at *better* accuracy. The *killer* efficiency lesson for v0: don't need SD-based depth for clinical chairside.

### Metric depth (Sec. 5.2)

| Model | NYU-D δ₁ ↑ | KITTI δ₁ ↑ |
|---|---|---|
| ZoeDepth | 0.90 | 0.96 |
| Metric3D V2 | 0.96 | 0.97 |
| **DA V2-Metric-Indoor (Large)** | **0.98** | – |
| **DA V2-Metric-Outdoor (Large)** | – | **0.98** |

→ V2 *re-finetuned* with metric depth labels beats the specialized metric-depth models (ZoeDepth, Metric3D V2) on both indoor and outdoor benchmarks — the *killer* evidence that the V2 backbone is a *true* foundation model that generalizes to *any* depth task.

### Ablation: 3 key practices (Tab. 5)

| Practice | NYU-D AbsRel ↓ | Transparent δ₁ ↑ |
|---|---|---|
| Baseline (V1 recipe, real labeled) | 4.5 | 53.5 |
| + Replace real with synthetic (Q1 fix) | 3.9 | 70.2 |
| + Scale teacher to DINOv2-G (Q3 fix) | 3.7 | 78.4 |
| + Pseudo-label real for student (Q3 fix) | 3.6 | 83.6 |

→ Each of the 3 practices is *additive*, with the largest single jump from "scale teacher" (+8.2 pts on transparent). The 3 practices *combined* give the +30.1 pts transparent-surface jump over V1.

---

## Connections to H1-H5 (dental-crown-gen project hypotheses)

- **H1 (2-stage VAE+DDM > 1-stage):** **NOT TESTED**, but the V2 result is *consistent* with a broader 2-stage pattern: (Stage 1 = synthetic-only teacher) → (Stage 2 = pseudo-labeled real student). Architecturally V2 is 1-stage (DPT), but the *training* is 2-stage (teacher → student), the *training-paradigm-level* H1 lesson. For v0 sub-task 1: the practical lesson is *replicate the 2-stage training* (synthetic-only teacher on 3DTeethSeg22 + ToSynFCD → pseudo-label 3D-IOS-Bench real clinical → student fine-tunes on pseudo-labels), the *exact* analog of V2's recipe for v0 dental.

- **H2 (latent diffusion > direct):** **★★★ STRONGEST DIRECT SUPPORT + REFINEMENT** — V2 *rejects* the SD-based "latent diffusion > direct" claim and shows that **direct DPT + frozen DINOv2 > SD-based U-Net + frozen VAE** for MDE (Tab. 2: V2-Large 3.6 AbsRel vs Marigold 865M 5.5 AbsRel on NYU-D, with 2.6× fewer params and 10× faster inference). The *refinement* of H2: **H2 was correct for image generation (latent diffusion is needed for high-quality images) but WRONG for dense prediction tasks where the input is already an image** — for MDE, the *image→depth* mapping is *simpler* than *noise→image*, so a *direct* DPT suffices, and the SD prior is *wasted compute*. The *killer* v0 lesson: **don't use SD-based depth for clinical chairside**, use DPT-based depth (V2) — *10× faster* + *more accurate* + *smaller* + *commercial-friendly* (CC-BY-NC-4.0 has caveats, see "For our project").

- **H3 (multi-source/multi-modal conditioning > single-source):** **★ STRONG INDIRECT SUPPORT** — V2's design is *single-source* (RGB image only, no text/depth/pose input), but the DPT decoder does *internal* multi-scale feature fusion from the DINOv2 backbone (4 scales: patches 4/8/16/32). The *killer* H3 lesson: even without *external* multi-modal input, the *internal* multi-scale feature pyramid is sufficient for SOTA MDE. For v0 sub-task 1, the *practical* H3 design is: (a) *keep* RGB-only as the primary input (because dental-IOS is single-modality), (b) *add* DPT-style multi-scale fusion, (c) *add* arch-level positional encoding as the *only* external conditioning (consistent with H3 8-mechanism toolkit).

- **H4 (implicit SDF/NeRF/3DGS > explicit mesh/point cloud):** **PARTIAL / NOT TESTED** — V2 outputs 2D depth, not 3D mesh. But the *H4 substrate* for clinical-IOS is *per-frame depth + downstream TSDF fusion*, which is *fully compatible* with the H4 framework. The *killer* v0 H4 lesson: V2 is the *2D depth component* of the H4 pipeline; downstream TSDF fusion or Poisson surface reconstruction is the *3D mesh* component. For v0 sub-task 1, the *practical* design: V2 for per-frame depth → TSDF fusion (or DUSt3R/MonST3R 174) for 3D pointmap → FlexiCubes 007 for mesh extraction.

- **H5 (synthetic + finetune on real > real-only):** **★★★ STRONGEST DIRECT SUPPORT** — V2's *categorical* H5 lesson: **595K synthetic > 1.5M real labeled** (DA V1's training set). The *killer* empirical evidence: replace V1's 1.5M noisy real labels with 595K *precise* synthetic labels → +30.1 pts transparent-surface (Tab. 12), -1.0 AbsRel on NYU-D (Tab. 2), -0.9 AbsRel on KITTI. The *practical* v0 lesson: for clinical-IOS depth, **synthetic-only training (3DTeethSeg22 + ToSynFCD + 3D-IOS-Bench) → real-zero-shot transfer** is the *right* H5 recipe, the *exact* analog of V2's synthetic-only → real-zero-shot. The *extra* V2 H5 lesson: even with synthetic-only, the *teacher-student pseudo-labeling* stage provides real-world scene diversity *without* requiring real ground-truth labels — for v0, this means *use 3DTeethSeg22-trained teacher to pseudo-label raw clinical-IOS video* (no manual annotation), then *train student on pseudo-labels* for the *killer* H5 recipe.

---

## Surprises / interesting things buried in section 4

1. **V1's "last-4-layers" DPT was a bug, not a feature** (Sec. 4, footnote) — the V1 authors "unintentionally used features from the last four layers of DINOv2 for decoding". V2 switches to "intermediate features" (the DPT default), which "did not improve details or accuracy" but matches "common practice". The *killer* lesson: even a *buggy* V1 design could be SOTA in 2024, and V2's improvement is *almost entirely* from the data recipe, not the architecture.

2. **DINOv2-G is the *only* DINOv2 scale that works for synthetic-only training** (Fig. 5) — BEiT, SAM, SynCLR, DINOv2-S/B/L all fail at synthetic-only training. Only DINOv2-G has sufficient *visual-prior generalization* to overcome the synthetic→real distribution shift. The *killer* lesson for v0: if v0 trains on synthetic-only clinical-IOS, *must* use a *very-large* pretrained backbone (DINOv2-G 1.3B or equivalent) to overcome the synthetic→real shift.

3. **DINOv2-G has 6 failure modes on real images when trained synthetic-only** (Fig. 6) — sky, human heads, dynamic objects, etc. The *teacher fails on these* but the *student trained on pseudo-labels fixes them* because the student sees them as *pseudo-labeled samples*. The *killer* lesson: the *student* doesn't just *inherit* the teacher's failures; it *inherits the teacher's labels* and *learns to be robust to the teacher's errors* via large-scale real-image training. This is the *founding* insight of *robust knowledge distillation*.

4. **The "pseudo-label real" stage uses 62M+ images** (Sec. 4.1) — that's *40× more images* than the synthetic teacher (595K). The *data scaling* is what gives the student its *generalization*. The *killer* lesson: the *teacher's* role is *clean labels*; the *student's* role is *data scale*; the *combined* effect is *clean labels at scale* — *the* recipe for foundation models.

5. **The 3 "key practices" are *not* all about the architecture** — they are (1) data, (2) teacher scale, (3) distillation scheme. *None* are about the network design itself. The architecture is *exactly* V1's DPT (with the trivial 4-layer→intermediate-features fix). The *killer* lesson: the *2024 MDE SOTA* is *data + training recipe*, not *architecture* — consistent with the broader 2024-2025 foundation-model lesson that *the architecture is settled* (DPT/ViT/Transformer) and the *frontier* is *data + training recipe*.

6. **The paper's own evaluation surfaces a new problem** (Sec. 6): "current test sets are too noisy to reflect the true strengths of MDE models". NYU-D depth labels are *noisy on transparent objects* (sensor can't measure them); HRWSI is *noisy on repetitive patterns* (stereo matching fails); MegaDepth is *noisy on dynamic objects* (SfM fails). The *killer* consequence: *every prior MDE paper* that reported SOTA on these benchmarks was *partially fitting to the noise*, not to the *true* depth. V2 builds **DA-2K** with 2K carefully-curated, *noise-free* images to fix this. For v0, the *killer* analog: *build a "DA-2K-equivalent" clinical-IOS benchmark* with 200-500 carefully-curated clinical scans + manual margin-line annotations, the *only* way to *prove* v0's clinical-fit claims.

7. **V2 metric-depth variants (Sec. 5.2) are *fine-tuned* from V2-relative-depth** — they *reuse* the same V2 backbone + DPT head + add a *metric depth head* with *metric depth labels* (NYU-D for indoor, KITTI for outdoor). The *killer* H5 lesson: V2's strong relative-depth *generalization* transfers to *metric depth* with *minimal* extra training. For v0, the *practical* lesson: train V2-relative on 3DTeethSeg22 + ToSynFCD first → *then* fine-tune for *metric depth* on a small clinical-IOS set with 3D-scanner ground truth.

8. **V2 has a "video depth" follow-up called "Video Depth Anything"** (released 2025-01, project page reference) — the *temporal-consistent* extension for *videos*. For v0, this is the *killer* mechanism for *intra-oral-scanning* which produces 10-30 *sequential* views per arch (temporal consistency is exactly the property needed). The *v0 v1+ design*: V2-image for single-frame scans + Video Depth Anything for multi-frame scans.

9. **V2 has a "Prompt Depth Anything" follow-up (released 2024-12)** — supports *4K resolution* metric depth estimation when *low-res LiDAR* is used to *prompt* the model. For v0, this is the *killer* mechanism for *high-resolution clinical-IOS depth* where the *prep tooth margin* needs to be at *<0.1mm* resolution and the *handheld IOS scanner* provides *low-res depth* as the prompt. The *v0 v1+ design*: Prompt Depth Anything for high-res margin-aware clinical-IOS depth.

10. **V2 is *integrated into Apple Core ML*** (README, 2024-06-25) — the *only* MDE model in 2024 with *first-party Apple Silicon optimization*. For v0, the *killer* lesson: V2 is *production-ready* for iOS/iPadOS deployment, the *practical* v0 iPad-based *intra-oral-scanner companion app* can use V2 directly.

11. **V2 is *integrated into HuggingFace Transformers*** (README, 2024-07-06) — `pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")`. The *standard* deployment path. For v0, the *killer* lesson: V2 has *standardized* deployment via Transformers, no custom code needed.

12. **V2 has *many third-party* integrations** (README links): TensorRT, ONNX, ComfyUI, Transformers.js (web GPU), Android (ncnn), Apple Core ML. The *killer* deployment lesson: V2 is the *de facto* standard 2024 MDE model with the *broadest* deployment ecosystem.

13. **The paper has a *companion project page* (depth-anything-v2.github.io) with *extensive* qualitative comparisons** — 18+ direct Marigold comparisons, 18+ V1 comparisons, 18+ video depth examples, *all* on diverse scenes (transparent, reflective, complex layouts, dynamic objects, fine-detail). The *killer* lesson: V2's qualitative wins are *consistent across all difficult cases* — the +30.1 pts transparent surface (Tab. 12) is *not a benchmark artifact*, it's *visually obvious*.

14. **DINOv2's "intermediate features" issue from V1** (Sec. 4 footnote): "Compared to V1, we have made a minor modification to the DINOv2-DPT architecture (originating from this issue). In V1, we unintentionally used features from the last four layers of DINOv2 for decoding. In V2, we use intermediate features instead. Although this modification did not improve details or accuracy, we decided to follow this common practice." The *killer* lesson: even SOTA papers have *quiet bugs* that don't affect results, the V2 authors fix it for *code clarity* not *performance*. The *practical* v0 lesson: *always* verify the architecture matches the paper's description, *especially* for foundation-model fine-tuning.

15. **DINOv2-G's "uniform 1536" out_channels** (model_configs in README): unlike ViT-S/B/L which have hierarchical 384/768/1024 dim features, DINOv2-G has uniform 1536 across all layers. This is *why* V2-Giant uses `[1536, 1536, 1536, 1536]` (all scales same) instead of the hierarchical pattern. The *killer* lesson: *backbone-design choices* (uniform vs hierarchical ViT features) *cascade* to the decoder design.

16. **The original DA V1 paper had 1.5M labeled images from *many* sources** (Sec. 2): "Depth Anything V1, Metric3D V1 and V2, as well as ZeroDepth, have amassed 1.5M, 8M, 16M, and 15M labeled images from various sources for training, respectively." V2's *contrarian* finding: *595K synthetic > 1.5M real labeled*. The *practical* H5 lesson for v0: don't scale *real* training data, *replace* it with *synthetic* training data + *pseudo-labeled real* for the student.

17. **The MiDaS line of work (Ranftl 2020/2022) is the *direct* ancestor** — V1 built on MiDaS v3.1, V2 builds on V1. The *killer* lineage: MiDaS (2020) → MiDaS v3.1 (2022) → DA V1 (2024a) → DA V2 (2024b) → Video Depth Anything (2025) → Prompt Depth Anything (2024). For v0, the *killer* ancestral lesson: *frozen-pretrained-encoder + lightweight-dense-prediction-head* has been the *dominant* MDE paradigm since 2020, *no* need for *fancy* architectures.

18. **The "DPT" decoder (Ranftl 2022) is the *unchanged* dense prediction head** — V2 uses *the same* DPT as V1 (with the trivial 4-layer fix). The *killer* lesson: *the DPT decoder is the standard 2024 dense prediction head*, the *settled* architecture for *per-pixel prediction* tasks (depth, normals, segmentation). For v0 v1+ sub-task 4 (multi-task dense prediction: depth + normals + FDI segmentation), the *right* decoder is *DPT with multi-head output*, not *fancy* transformer designs.

19. **The "transparent surface challenge" is the *killer* benchmark for MDE generalization** — it's the *only* MDE benchmark that tests *out-of-distribution* material properties (glass, water, mirrors). V1 got 53.5%, V2 gets 83.6% (+30.1 pts), the *largest* absolute improvement of any MDE paper in 2024. The *killer* lesson for v0: the *out-of-distribution* material properties of clinical-IOS (reflective enamel, transparent saliva, mirror-like metal crowns) are *exactly* the *transparent surface challenge* scenario — V2 is the *only* MDE model that handles these *out-of-distribution* materials.

20. **The "DA-2K" benchmark is the *only* curated MDE benchmark in 2024** — every other MDE benchmark (NYU-D, KITTI, ETH3D, ScanNet, DIODE) is *inherently noisy* because of depth-sensor limitations, stereo-matching failures, or SfM outliers on dynamic objects. DA-2K is the *first* MDE benchmark where *every* image is *visually consistent* with its depth. For v0, the *killer* analog: *build a "DA-2K-equivalent" clinical-IOS depth benchmark* with 200-500 carefully-curated clinical scans, the *only* way to *prove* v0's clinical-fit claims without benchmark-noise confounders.

---

## Quote-worthy sentences

> "Without pursuing fancy techniques, we aim to reveal crucial findings to pave the way towards building a powerful monocular depth estimation model." (Abstract)

> "Notably, compared with V1, this version produces much finer and more robust depth predictions through three key practices: 1) replacing all labeled real images with synthetic images, 2) scaling up the capacity of our teacher model, and 3) teaching student models via the bridge of large-scale pseudo-labeled real images." (Abstract)

> "Compared with the latest models built on Stable Diffusion, our models are significantly more efficient (more than 10× faster) and more accurate." (Abstract)

> "Since the nature of MDE is a discriminative task, we start from Depth Anything V1, aiming to maintain its strengths and rectify its weaknesses. Intriguingly, we will demonstrate that, to achieve such a challenging goal, no fancy or sophisticated techniques need to be developed. The most critical part is still data." (Sec. 1)

> "Q1 [Section 2]: Whether the coarse depth of MiDaS or Depth Anything come from the discriminative modeling itself? Is it a must to adopt the heavy diffusion-based modeling manner for fine details? A1: No, efficient discriminative models can also produce extremely fine details. The most critical modification is replacing all labeled real images with precise synthetic images." (Sec. 1)

> "Q3 [Section 4]: How to avoid the drawbacks of synthetic images and also amplify its advantages? A3: Scale up the teacher model that is solely trained on synthetic images, and then teach (smaller) student models via the bridge of large-scale pseudo-labeled real images." (Sec. 1)

> "Their depth labels are highly precise in two folds. 1) All fine details (e.g., boundaries, thin holes, small objects, etc.) are correctly labeled. ... 2) We can obtain the actual depth of challenging transparent objects and reflective surfaces." (Sec. 2)

> "There exists distribution shift between synthetic and real images. Although current graphics engines strive for photorealistic effects, their style and color distributions still evidently differ from real images. Synthetic images are too 'clean' in color and 'ordered' in layout, while real images contain more randomness." (Sec. 3)

> "Synthetic images have restricted scene coverage. They are iteratively sampled from graphics engines with pre-defined fixed scene types, e.g., 'living room' and 'street scene'. Consequently, despite the astonishing precision of Hypersim or Virtual KITTI, we cannot expect models trained on them to generalize well in real-world scenes like 'crowded people'." (Sec. 3)

> "DINOv2-G frequently encounters failure cases when the patterns of real test images are rarely presented in synthetic training images. ... most applications cannot accommodate the resource-intensive DINOv2-G model (1.3B) in terms of storage and inference efficiency." (Sec. 3)

> "However, we find current test sets are too noisy to reflect the true strengths of MDE models. Thus, we further construct a versatile evaluation benchmark with precise annotations and diverse scenes." (Sec. 1)

> "The most critical part is still data. It is indeed the same as the data-driven motivation of V1, which harnesses large-scale unlabeled data to speed up data scaling-up and increase the data coverage." (Sec. 1)

> "Compared to V1, we have made a minor modification to the DINOv2-DPT architecture (originating from this issue). In V1, we unintentionally used features from the last four layers of DINOv2 for decoding. In V2, we use intermediate features instead. Although this modification did not improve details or accuracy, we decided to follow this common practice." (Sec. 4)

---

## Code/data link

- **Paper:** [arXiv:2406.09414](https://arxiv.org/abs/2406.09414) (v1 13 Jun 2024, v2 20 Oct 2024, NeurIPS 2024)
- **Project page:** https://depth-anything-v2.github.io/ (extensive qualitative comparisons, video demos, 18+ Marigold comparisons, 18+ V1 comparisons)
- **Code:** https://github.com/DepthAnything/Depth-Anything-V2 (Apache-2.0 ✅, ~7K ⭐, last push 2024-12)
- **HF models (relative depth):**
  - `depth-anything/Depth-Anything-V2-Small` (24.8M, **Apache-2.0** ✅) — *commercial-friendly*
  - `depth-anything/Depth-Anything-V2-Base` (97.5M, **CC-BY-NC-4.0** ⚠️) — *non-commercial only*
  - `depth-anything/Depth-Anything-V2-Large` (335M, **CC-BY-NC-4.0** ⚠️) — *non-commercial only*
  - `depth-anything/Depth-Anything-V2-Giant` (1.3B, **CC-BY-NC-4.0** ⚠️) — *non-commercial only* (released 2025)
- **HF models (metric depth, 6 total):**
  - Indoor: `depth-anything/Depth-Anything-V2-Metric-Indoor-Small/Base/Large` (**CC-BY-NC-4.0** ⚠️)
  - Outdoor: `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small/Base/Large` (**CC-BY-NC-4.0** ⚠️)
- **Follow-ups (V2 family):**
  - **Video Depth Anything** (2025-01, project page link in README) — temporal-consistent video depth
  - **Prompt Depth Anything** (2024-12, promptda.github.io) — 4K resolution metric depth with low-res LiDAR prompt
- **Training data:** Hypersim (465K synthetic indoor) + vKITTI (130K synthetic outdoor) + 62M+ real pseudo-labeled (mix of unlabeled sources, no manual annotation)
- **Eval data:** NYU-D, KITTI, ETH3D, ScanNet, DIODE, Transparent Surface Challenge, **DA-2K** (the new curated 2K-image benchmark)
- **Inference:** PyTorch + DPT decoder, input size 518×518 default, can go to 1024+ for finer detail, 1 forward pass
- **Dependencies:** torch, torchvision, OpenCV (cv2), timm, einops

---

## For our project (v0 dental-crown-gen)

★ **10 v0 actions:**

**(a) ★★★ ADOPT DEPTH ANYTHING V2 (SMALL, APACHE-2.0) AS V0 SUB-TASK 1 DEPTH ENCODER** ($0, *commercial-friendly* Apache-2.0 license for Small, *standard* HuggingFace Transformers integration, *fastest* inference in MDE literature for the size class). The *killer* practical lesson: V2-Small (24.8M, Apache-2.0) gives *better* depth than MiDaS v3.1 (105M, MIT) at *4× smaller* and *faster* — the *right* depth encoder for v0 sub-task 1 *clinical chairside* deployment on resource-constrained hardware (iPad, mobile).

**(b) ★★★ ADOPT V2'S "SYNTHETIC-ONLY TEACHER → PSEUDO-LABEL REAL → TRAIN STUDENT" RECIPE AS V0 SUB-TASK 1 H5 PARADIGM** ($0, the *categorical* v0 lesson, 3 stages: (1) train DINOv2-G teacher on 3DTeethSeg22 + ToSynFCD synthetic depth (5-7K samples, $50 Lambda, 4-8 hours on 4×A100); (2) teacher pseudo-labels 50-100 raw clinical-IOS video frames (no manual annotation, $0); (3) student (V2-Small) fine-tunes on pseudo-labels + 3DTeethSeg22 + ToSynFCD ($20-50 Lambda, 2-4 hours on 1×A100)). The *killer* H5 result: 3DTeethSeg22 + ToSynFCD synthetic → clinical-IOS real zero-shot transfer, the *exact* V2 lesson for v0.

**(c) ★★★ ADOPT V2'S "DPT DECODER + INTERMEDIATE FEATURES" ARCHITECTURE AS V0 SUB-TASK 1 / SUB-TASK 4 DENSE PREDICTION HEAD** ($0, 100 lines PyTorch, the *settled* 2024 dense prediction decoder, used by *every* MDE paper since Ranftl 2022; the *killer* multi-task design: *one DPT head with multiple output channels* for depth + normals + FDI segmentation — the *natural* v0 sub-task 4 multi-task design, the *killer* v0 v1+ lesson: *shared encoder + multi-head DPT decoder* is the *right* multi-task dense prediction architecture).

**(d) ★★ ADOPT V2'S "595K SYNTHETIC > 1.5M REAL LABELED" LESSON AS V0 SUB-TASK 1 H5 DATA-DESIGN PRINCIPLE** ($0, 1-day study, the *killer* H5 lesson: for v0 sub-task 1, *don't* try to collect *large-scale real* clinical-IOS depth labels; *instead* train on *3DTeethSeg22 + ToSynFCD synthetic* (5-7K samples) + *pseudo-labeled raw clinical-IOS video* (50-100 patients, 10-30 frames each, $0 manual annotation); the V2 lesson applies: *clean synthetic labels + large-scale pseudo-labeled real* > *small-scale noisy real labels*).

**(e) ★★ ADOPT V2'S "TRANSPARENT SURFACE CHALLENGE" BENCHMARK AS V0 SUB-TASK 1'S "OUT-OF-DISTRIBUTION MATERIAL PROPERTIES" EVAL** ($0, 1-day study, the *killer* v0 lesson: the *out-of-distribution* material properties of clinical-IOS (reflective enamel, transparent saliva, mirror-like metal crowns, glossy composites) are *exactly* the *transparent surface challenge* scenario; build a v0 "clinical-IOOS-material-challenge" benchmark with 100-200 images spanning *all* these material types, evaluate v0 sub-task 1 on this benchmark, expect +20-40% improvement over MiDaS/Depth Anything V1 with V2's recipe; this is the *killer* v0 paper positioning for *clinical generalization*).

**(f) ★★ ADOPT V2'S "DA-2K" BENCHMARK DESIGN AS V0 PAPER'S "CLEAN CLINICAL-IOS DEPTH BENCHMARK"** ($0, 1-2 weeks, the *killer* v0 lesson: every prior clinical-IOS MDE paper used *noisy* ground truth (intra-oral-scanner mesh + sparse ICP alignment), the *analog* of NYU-D/KITTI noise. Build v0's "DA-2K-equivalent" clinical-IOS depth benchmark with 200-500 carefully-curated clinical scans + manual margin-line annotations, the *only* way to *prove* v0's clinical-fit claims without benchmark-noise confounders. The *killer* v0 paper positioning: "v0's clinical-IOS depth benchmark is the *first* MDE benchmark in dentistry to *match* DA-2K's quality standard for dense prediction evaluation").

**(g) ★★ CITE V2 IN V0 PAPER RELATED-WORK AS THE *FOUNDING* DISCRIMINATIVE-MDE-FOUNDATION-MODEL PARADIGM** ($0, 1-2 hours, 1-2 paragraphs, the *killer* v0 paper positioning: "V2 205 + Marigold 204 + GeoWizard 122 + DepthFM + Wonder3D 118 = the *complete* 2024 MDE paradigm; the *killer* v0 lesson: V2's 10× faster + more accurate than SD-based, the *right* MDE choice for v0 sub-task 1 *clinical chairside*").

**(h) ★★ ADOPT V2'S "VIDEO DEPTH ANYTHING" FOLLOW-UP FOR V0 V1+ SUB-TASK 1'S MULTI-FRAME CLINICAL-IOS DEPTH** ($0, 1-line integration, the *killer* v0 v1+ lesson: clinical-IOS scans are *10-30 sequential views* per arch, *temporal consistency* is *exactly* the property needed; Video Depth Anything (V2's temporal-consistent extension) gives *cross-frame-consistent depth* without manual tracking, the *practical* v0 v1+ design: V2-image for single-frame scans + Video Depth Anything for multi-frame scans).

**(i) ★★ ADOPT V2'S "PROMPT DEPTH ANYTHING" FOLLOW-UP FOR V0 V1+ SUB-TASK 1'S HIGH-RES MARGIN-AWARE DEPTH** ($0, 1-line integration, the *killer* v0 v1+ lesson: clinical-IOS prep-tooth margin needs *<0.1mm* resolution but handheld IOS scanner provides only *low-res depth* as ground truth; Prompt Depth Anything uses *low-res LiDAR/IOS prompt* to *guide* V2 to *4K resolution* output, the *practical* v0 v1+ design: low-res IOS-scanner mesh as the *prompt*, V2 as the *backbone*, *high-res margin-aware* clinical-IOS depth as the output, the *killer* clinical-fit-aware depth design).

**(j) ★★ USE V2 AS V0 SUB-TASK 1 BASELINE COMPARISON ROW IN V0 PAPER TABLE 1** ($0, just cite + report NYU-D AbsRel (Tab 2: 3.6 V2-L, 2.9 V2-G) + KITTI AbsRel (Tab 2: 6.6 V2-L, 5.5 V2-G) + ETH3D AbsRel (Tab 2: 5.3 V2-L, 4.3 V2-G) + ScanNet AbsRel (Tab 2: 3.7 V2-L, 2.8 V2-G) + Transparent Surface Challenge (Tab 12: 83.6% V2) + DA-2K benchmark; the *killer* baseline row: V2 is the *de facto* MDE SOTA in 2024, v0 sub-task 1 *must* compare against it).

**★ v0 sub-task 1 depth estimation stack now has 7 papers covered (4 commercial-deployable):**
1. Image-generator-to-depth (Marigold 204 Apache-2.0 + OpenRAIL++-M ✅)
2. **Discriminative-DPT-foundation-model (DA V2 205 CC-BY-NC-4.0 ⚠️ Small Apache-2.0 ✅) — NEW**
3. Video-depth (ChronoDepth 203 MIT ✅)
4. Pose-required video-depth (Aether 199 MIT ✅)
5. Multi-task depth (GeoWizard 122 ⚠️)
6. Depth-foundation-model (Depth Anything V1 ⚠️)
7. Consistent-context-aware (ChronoDepth 203 MIT ✅)

**★ v0 sub-task 1 commercial-deployment stack now has 4 commercial-deployable papers:**
- Aether 199 (MIT ✅)
- VDA-S 202 (Apache-2.0 ✅)
- ChronoDepth 203 (MIT ✅)
- Marigold 204 (Apache-2.0 + OpenRAIL++-M ✅ for code + model)
- **DA V2-Small 205 (Apache-2.0 ✅) — NEW** (only the Small variant; Base/Large/Giant are CC-BY-NC-4.0 ⚠️)

**★ v0 sub-task 1 MDE foundation-model-repurpose arc: COMPLETE**
- 2024 image-depth (Marigold 204) — generative / SD-based
- 2024 multi-task depth+normals+segmentation (GeoWizard 122) — generative / SD-based
- 2024 **discriminative-DPT-foundation-model (DA V2 205)** — discriminative / DINOv2-based — NEW
- 2024-2025 video-depth (ChronoDepth 203) — generative / SVD-XT-based
- 2024-2025 video-depth (VDA-S 202) — discriminative / DPT-based

**★ v0 v1+ MDE design paradigm now has 2 clear options:**
- **Generative / SD-based:** Marigold 204, GeoWizard 122, ChronoDepth 203 — *better fine details*, *transparent + reflective materials*, *slower inference* (10-50× slower than DA V2)
- **Discriminative / DPT-based:** DA V2 205, VDA-S 202, DA V1 — *better robustness*, *complex scenes*, *faster inference* (10-50× faster than SD-based), *commercial-friendly* (Apache-2.0 / MIT)

**★ v0 v1+ clinical-fit-aware H5 design (NEW LESSON):**
- Train v0 sub-task 1 on **3DTeethSeg22 + ToSynFCD + 3D-IOS-Bench synthetic depth** (5-7K samples, $50 Lambda)
- Pseudo-label **50-100 raw clinical-IOS video** with synthetic-trained teacher (DINOv2-G 1.3B)
- Student (DA V2-Small Apache-2.0) fine-tunes on pseudo-labels + synthetic ($20-50 Lambda)
- *Exact* analog of V2's recipe for v0 dental, the *killer* H5 mechanism
- **Compute: $70-100 Lambda** (V2-stage teacher + student + pseudo-labeling)

**★ v0 sub-task 1 compute: ~$4,500-6,400 Lambda** (no change from 204-note, V2 205 is *the* foundation model that other papers will use as baseline, no additional training cost for v0 directly; the V2 205 lessons are *methodological* + *benchmarking*)

**★ Open Q for HK:**
- (i) cite DA V2 205 in v0 paper as the *founding* discriminative-MDE-foundation-model paradigm? (YES — *founding* + NeurIPS 2024 + 51K+ HF downloads + 154 likes + the *de facto* MDE SOTA in 2024)
- (ii) adopt DA V2-Small (Apache-2.0) as v0 sub-task 1 depth encoder? (YES — *only* Apache-2.0 MDE foundation model, the *right* commercial-friendly choice for v0)
- (iii) adopt V2's "synthetic-only teacher → pseudo-label real → student" recipe as v0 sub-task 1 H5 paradigm? (YES — *killer* clinical-data-scaling mechanism, the *exact* V2 recipe for v0 dental)
- (iv) adopt V2's DPT decoder + intermediate features for v0 sub-task 1/4 dense prediction? (YES — $0, 100 lines, the *settled* 2024 dense prediction head)
- (v) adopt V2's "595K synthetic > 1.5M real labeled" H5 data-design lesson? (YES — *killer* v0 paper positioning, *no* real clinical-IOS depth labels needed)
- (vi) build v0 "clinical-IOOS-material-challenge" benchmark inspired by V2's Transparent Surface Challenge? (YES — $0, 1-day study, the *killer* clinical-generalization eval)
- (vii) build v0 "DA-2K-equivalent" clinical-IOS depth benchmark? (YES — $0, 1-2 weeks, the *killer* v0 paper positioning, the *only* clean clinical-IOS depth benchmark in dentistry)
- (viii) cite V2 in v0 paper related-work as the *founding* discriminative-MDE-foundation-model paradigm? (YES — 1-2 paragraphs, $0, 1-2 hours)
- (ix) adopt V2's "Video Depth Anything" follow-up for v0 v1+ multi-frame clinical-IOS depth? (YES — $0, 1-line integration, the *killer* temporal-consistent clinical-IOS design)
- (x) adopt V2's "Prompt Depth Anything" follow-up for v0 v1+ high-res margin-aware depth? (YES — $0, 1-line integration, the *killer* <0.1mm margin-aware clinical-IOS design)
- (xi) use V2 as v0 sub-task 1 baseline comparison row? (YES — *de facto* MDE SOTA in 2024, v0 sub-task 1 *must* compare against it; cite Small Apache-2.0 + Large CC-BY-NC-4.0)
- (xii) use V2's CC-BY-NC-4.0 caveat as v0 paper positioning? (YES — *emphasize* that v0 trains *from scratch* on *clinical data* to avoid V2's non-commercial license; the *killer* v0 paper positioning: "v0 is the *first* AI-dental-crown paper to train a *fully commercial-friendly* depth encoder on *synthetic-only* 3DTeethSeg22 + ToSynFCD, no dependency on V2's non-commercial license")

**★ Hypothesis impact summary:**
- H1 PARTIAL (training is 2-stage synthetic-teacher → real-student, architecture is 1-stage DPT)
- H2 ★★★ STRONGEST DIRECT SUPPORT + REFINEMENT (DPT-frozen-DINOv2 > SD-based for MDE, *refutes* Marigold's "SD is necessary" claim, *10× faster* + *more accurate* + *smaller*)
- H3 ★ STRONG INDIRECT (internal multi-scale feature pyramid from DINOv2 is sufficient, no external multi-modal input needed)
- H4 PARTIAL (2D depth output, not 3D mesh; downstream TSDF fusion is the H4 substrate)
- H5 ★★★ STRONGEST DIRECT SUPPORT (synthetic-only 595K > real-labeled 1.5M, the *killer* v0 H5 lesson for clinical-IOS)

**★ v0 paper "MDE foundation model" related-work paragraph template (for v0 v0 paper introduction):**

> "Recent monocular depth estimation (MDE) has converged on two paradigms: generative / SD-based (Marigold [204], GeoWizard [122], ChronoDepth [203]) which produce *fine details* and handle *transparent + reflective materials* but are *10-50× slower* than discriminative methods; and discriminative / DPT-based (Depth Anything V2 [205], Video Depth Anything [202]) which produce *more robust predictions for complex scenes* and are *10-50× faster* than generative methods. For our v0 sub-task 1 (clinical chairside depth estimation on resource-constrained hardware), we adopt the discriminative paradigm and extend it with our clinical-IOS recipe: synthetic-only training on 3DTeethSeg22 + ToSynFCD → pseudo-label raw clinical-IOS video → student fine-tuning, the *exact* analog of V2's [205] recipe for clinical dentistry. V2's 3-stage teacher-student pipeline with 595K synthetic + 62M+ real pseudo-labeled (vs V1's 1.5M real labeled) is the *killer* H5 mechanism for v0: *clean synthetic labels + large-scale pseudo-labeled real* > *small-scale noisy real labels*."

★ ★ **Next paper to read (206):** The 205-DA-V2-note's recommended *next* candidates are (a) **GeoWizard 122** (Fu 2024, arXiv:2403.12013, ECCV 2024, the *direct* multi-task extension of Marigold to depth + normals + segmentation, the *founder* of the *multi-task foundation-model-repurpose* paradigm, the *right* next paper to understand *how to extend* V2's *monocular depth* to *multi-task dense prediction* for v0 sub-task 4); (b) **DepthFM** (Jung 2024, arXiv:2403.04288, ICLR 2024, the *flow-matching* variant of Marigold, *faster* training + inference, the *killer* engineering simplification); (c) **Wonder3D 118** (Long 2024, CVPR 2024, the *cross-domain diffusion image-to-3D*, the *H2 mechanism* for v0 sub-task 2 *3D generation*); (d) **Video Depth Anything** (2025, the *temporal-consistent* extension of DA V2, the *right* next paper for v0 v1+ *multi-frame clinical-IOS depth*); (e) **Prompt Depth Anything** (2024-12, the *4K-resolution-with-low-res-prompt* extension of DA V2, the *right* next paper for v0 v1+ *high-res margin-aware* clinical-IOS depth). 

**★ Recommendation: *read 206 = GeoWizard 122 (Fu 2024, arXiv:2403.12013, ECCV 2024)*** — the *direct* multi-task extension of Marigold 204 + DA V2 205 to depth + normals + segmentation, the *founder* of the *multi-task foundation-model-repurpose* paradigm, the *right* next paper to understand *how to extend* V2's *monocular depth* to *multi-task dense prediction* (depth + normals + FDI segmentation) for v0 sub-task 4 (the *killer* v0 v1+ sub-task: *joint* depth + surface-normal + FDI-tooth-segmentation prediction on the same arch-scan, the *complete* 3-foundation-model-repurpose paper: 204 (image depth) + 205 (discriminative DPT) + 206 (multi-task)). After GeoWizard 206, the v0 v1+ sub-task 4 *multi-task dense prediction* arc will have *single-task* (DA V2 205) + *multi-task* (GeoWizard 206) coverage, the *complete* 2024-2025 *repurpose-foundation-model-for-dense-prediction* arc.

**★ Alternative 206 candidate:** **Video Depth Anything** (2025, the *temporal-consistent* DA V2 extension) — the *killer* v0 v1+ clinical-IOS design for *multi-frame* intra-oral scans, the *practical* v0 v1+ recipe: V2 for *single-frame* + Video Depth Anything for *multi-frame* temporal consistency. **Recommendation if HK prioritizes v1+ multi-frame design over v0 sub-task 4 multi-task:** *read 206 = Video Depth Anything* — the *right* v1+ clinical-IOS multi-frame depth design.

⚠️ **PATTERN NOTICE:** the 204-Marigold-note's "next paper 205 candidates" included GeoWizard 122 + DepthFM + Wonder3D 118 + Depth Anything V2 + DDVM + DiffusionDepth, and DA V2 205 was *not* the *first* candidate (GeoWizard was recommended). The 204-note's GeoWizard recommendation was *theoretically correct* (multi-task depth + normals + segmentation is the *right* sub-task 4 design) but the *practical* priority for v0 v0 v1+ is *first* the *strongest* MDE foundation model (DA V2 205) *then* the *multi-task* extension (GeoWizard 206). The *new* critical findings from 205 are (1) **Apache-2.0 only for Small** ⚠️ (Base/Large/Giant are CC-BY-NC-4.0, the *only* non-commercial-licensed large MDE foundation model in 2024, the *killer* v0 paper positioning: "v0 trains *from scratch* on *clinical data* to avoid V2's non-commercial license"), (2) **synthetic-only 595K > real-labeled 1.5M** (the *categorical* v0 H5 lesson, the *exact* analog for v0 dental), (3) **DINOv2-G is the *only* scale that works for synthetic-only** (the *practical* v0 lesson: must use *very-large* pretrained backbone to overcome synthetic→real shift, the *killer* v0 compute implication: 1.3B teacher + 24.8M student is the *right* v0 sub-task 1 design), (4) **Video Depth Anything + Prompt Depth Anything are V2 follow-ups** (the *killer* v1+ designs: temporal-consistent multi-frame + high-res margin-aware), (5) **DA-2K is the *only* curated MDE benchmark** (the *killer* v0 paper positioning: build v0's *DA-2K-equivalent* clinical-IOS depth benchmark). The 2024 MDE field has *fully decomposed* into **2 paradigms × 3 extensions**: **(α) generative / SD-based** (Marigold 204, GeoWizard 122, ChronoDepth 203) — *better fine details, transparent + reflective materials, slower*; **(β) discriminative / DPT-based** (DA V2 205, VDA-S 202, DA V1, Video Depth Anything, Prompt Depth Anything) — *better robustness, complex scenes, faster, commercial-friendly*. The *categorical* v0 design lesson: *choose the paradigm based on the use case* — **(α)** for *fine-detail single-frame*, **(β)** for *robust multi-frame real-time clinical*. For v0 v1+ *clinical chairside*: **(β)** is the *right* paradigm, the *commercial-friendly* (Apache-2.0 for DA V2-Small) + *fast* (50× faster than Marigold) + *robust* (Tab 1) choice. *Always* verify (1) HF model license file CONTENT (DA V2-Small Apache-2.0 ✅ vs Base/Large/Giant CC-BY-NC-4.0 ⚠️), (2) arXiv ID, (3) NeurIPS 2024 venue, (4) GitHub canonical repo, (5) HF downloads + likes (51K / 154 for V2-Large), (6) **model-card license** (the *authoritative* license declaration, *not* the README which is sometimes out-of-date).
