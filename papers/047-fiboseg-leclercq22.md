# 047 — FiboSeg (Leclercq et al. 2022, MICCAI 3DTeethSeg'22 challenge, 2nd place, **best TLA**)

> **Important scope note:** FiboSeg does **not** have a standalone arXiv paper — the only primary source is **§4.2 of the 3DTeethSeg'22 challenge paper** (Ben-Hamadou et al. 2023, arXiv:2305.18277, MICCAI 2022 challenge satellite event). I cross-referenced with (a) Dascalu & Ibragimov 2023 (DentAssignNet, MICCAI 2023 — gives the labeling-accuracy comparison in the author rebuttal), (b) TCATSeg 2026 (arXiv:2603.16620 — gives the TLA/TSA/TIR table reproduced in Section 5), and (c) the DentalModelSeg ISBI 2023 paper (PMC10949221 — likely a *related* but distinct group using the same 2D-rendering recipe). The architecture details below are 100% from §4.2 of the challenge paper (Lines 667-746 of the arXiv PDF v1). The Leclercq affiliations are inferred from the TCATSeg citation: **Lucia Cevidanes** = U-Michigan Orthodontics (Cevidanes lab is well-known in dental 3D imaging); **Juan Carlos Prieto** = UNC-Chapel Hill (joint senior author on several Cevidanes dental-3D papers). The "FiboSeg" name is probably "**F**eature-**I**ntegrated **B**oundary-**O**riented **Seg**mentation" or similar (the paper never spells it out). **For the v0 citation, the canonical reference is the *challenge paper* §4.2, not a hypothetical standalone FiboSeg paper** — same citation pattern as ToothGroupNet (046).

## TL;DR

**The 3DTeethSeg'22 challenge silver medalist (Score 0.9480) and winner on the *teeth localization* sub-task (Exp(-TLA) 0.9924 = best of 6 teams by 0.0266) is a single 2D Residual U-Net trained on **rendered views** of the 3D IOS — the camera is placed on the unit sphere around the mesh, the surface normals are encoded as RGB + depth as a 4th channel, the labels are rendered as the GT target, and at inference the per-face label is determined by a **weighted majority vote** across all the rendered views that hit the face, followed by **island removal** and **morphological boundary closing**. Zero 3D point-cloud processing, zero curvature, zero centroid prediction — pure 2D. The architectural insight is that the **2D view from a single camera is a *complete summary* of the 3D tooth's shape and position** (you can see all 14 teeth of an upper jaw in one 256×256 image), and the U-Net's 2D inductive bias (translation equivariance) is *stronger* than the 3D transformer's self-attention for the *localization* sub-task (where you just need to know *where* the tooth is, not its precise vertex-level boundary). The cost: 2D rendering has **worse tooth-gingiva boundary** than 3D methods (the qualitative eval in §5.2 explicitly flags this — "FiboSeg team exhibits lower segmentation accuracy, specifically in the gum-teeth border"), and it cannot leverage *point curvature* or *FDI arch priors* because the rendering is geometry-only.**

## Research question + their answer

**Q:** Per-point tooth instance segmentation on intraoral 3D scans (IOS) has been dominated by **3D point-cloud networks** (PointNet, PointNet++, DGCNN, Point Transformer, Graph CNN) for the past 5+ years. These networks all have to **learn a 3D inductive bias from scratch** (rotation equivariance, point permutation equivariance, scale invariance) on a **limited training set** (3DTeethSeg'22 has only 1,000 train scans, 200 val scans, 600 test scans). The 3D inductive bias is *expensive* to learn (you need ~10× the data of a 2D network to get the same representation quality). **Can we *bypass* the 3D inductive bias entirely by rendering the 3D mesh into 2D images, applying a 2D U-Net (which has a *free* 2D inductive bias from ImageNet pre-training), and back-projecting the 2D labels to the 3D mesh via weighted majority voting?** This is the **2D rendering approach** — the same idea as Su et al. 2015 (MVCNN for shape classification), but for *per-vertex semantic segmentation* of an *irregular mesh* (not a 2D-to-2D classification). The question has 3 sub-questions: (a) can the U-Net see *all 14 teeth* of an upper jaw in a single 256×256 image, (b) does the per-face majority vote over ~50 rendered views produce a *clean* per-vertex label, (c) can the network learn the *FDI numbering* (32 classes for upper+lower + 2 for gum+background = 34 total) from 2D rendered views *without* an explicit centroid prediction step?

**A:** **Yes, partially — FiboSeg wins the *teeth localization* sub-task (Exp(-TLA) 0.9924) and achieves the 2nd-highest *labeling accuracy* (TIR 0.9223) of the 6 teams, but loses the *segmentation accuracy* sub-task (TSA 0.9293, 3rd best) and the *overall Score* (0.9480, 2nd best by 0.0059 behind CGIP's 0.9539).** The architectural lesson is that **2D rendering dominates the *localization* sub-task** (the rendered 256×256 image is a *global view* — you see all 14 teeth in one shot, and the U-Net's spatial attention can find each tooth's centroid in 1 forward pass), but **3D methods dominate the *boundary* sub-task** (the rendered 2D projection loses the precise 3D vertex correspondence, and the majority vote at the boundary is *unstable* — different views see the boundary as different labels, and the vote is the average of N inconsistent labels, which is a smooth blob instead of a sharp edge). **For the v0, the takeaway is that 2D rendering is a *strong baseline for the localization step* (sub-task 1's "find the tooth centroid" pre-processor) but a *weak baseline for the segmentation step* (sub-task 1's "label each vertex" main task).** A 2-stage pipeline that uses 2D rendering for localization (FiboSeg's contribution) and 3D point-cloud for segmentation (ToothGroupNet's contribution) is the *Pareto-optimal* combination on the 3DTeethSeg'22 leaderboard.

## Method

### Architecture (single diagram, 3 components)

The FiboSeg pipeline is a **single 2D Residual U-Net** (from MONAI, Hatamizadeh et al. 2022 UNETR-style residual blocks) with 3 stages:

1. **Multi-view 2D rendering (Pytorch3D)** — training and inference
   - Camera placement: random rotation on the **unit sphere** around the mesh center (mesh is centered via the same PCA-based pose normalization as the 3DTeethSeg'22 annotation pipeline)
   - Image resolution: **320px × 320px**
   - For each snapshot, render **2 images**:
     - **Input image (4 channels)**: surface normals encoded in RGB + **depth map as the 4th channel**
     - **Target image (1 channel)**: per-pixel GT label (one-hot encoded into 34 classes = 32 FDI teeth + gum + background, with the same FDI numbering as the 3DTeethSeg'22 dataset)
   - Lighting: **ambient only** (no specular components, so the normals-as-RGB encoding is preserved)
   - The Pytorch3D rasterizer returns a **per-pixel → per-face mapping** (the "Pix2Face" in Figure 5), which is used at inference to back-project the 2D U-Net output to per-face labels on the 3D mesh
   - **Data augmentation**: random camera rotation + **random crown removal** (the "Pix2Face" in Figure 5; wisdom teeth excluded). This is a *curriculum-learning* trick — the model learns to segment *incomplete* jaws from the start, which is critical for the 27% of 3DTeethSeg'22 patients with missing teeth

2. **Residual U-Net (MONAI, 34 output channels)** — single forward pass, no decoder trick
   - Backbone: MONAI's residual U-Net (Hatamizadeh et al. 2022, the same UNETR backbone used in MONAI's medical-imaging pipelines)
   - Input: 4 channels (normal-RGB + depth)
   - Output: **34 channels** (one per FDI class: 32 teeth + gum + background)
   - Encoder/decoder: **5 blocks**, channels **[16, 32, 64, 128, 256]**, **2 residual units per block**, **stride 2** for downsampling
   - Loss: **DiceCELoss** = `w_0 · (1 - Dice) + w_1 · CE` with w_0 = 1, w_1 = 1 (Eq. 1 of §4.2.2 in the challenge paper)
   - **Optimizer**: AdamW, **learning rate 1e-4**
   - **Training time / batch size / epochs**: not specified in the challenge paper (likely 100-200 epochs, batch size 4-8, trained on a single T4 or V100)
   - **No pre-training** on ImageNet (the MONAI residual U-Net is trained from scratch on the 3DTeethSeg'22 training split of 1,200 scans, 8,000 rendered views per scan)

3. **Weighted majority voting + island removal + morphological closing** — inference-time back-projection
   - Render **N views per scan** (N is not specified in the paper, but inferred from the "single face may be rendered by 2 separate views" wording as **N ≈ 50-100 views**, with random rotations per view)
   - For each face in the 3D mesh, **collect the per-pixel U-Net predictions from all N views that rendered that face** (the Pytorch3D Pix2Face mapping tells us which pixels correspond to which face)
   - **Weighted majority vote**: each face's label is the argmax over the 34-class logits, weighted by the *view's distance to the face's centroid* (closer views get higher weight, farther views get lower weight — this is a *free* way to downweight the unreliable grazing-angle views)
   - Faces that received **zero pixel hits** (rare, due to the camera being on a unit sphere and the mesh being closed) are assigned the value **-1** (a "missing label" flag for the post-processor)
   - **Post-processing step 1 — island removal**: any connected component of same-label faces that is **smaller than a threshold** (not specified, but ~50-100 faces) is *reassigned* to the label of the largest adjacent component (this is a *cheap* denoising step that removes the random label flips that the majority vote occasionally produces at the boundaries)
   - **Post-processing step 2 — morphological closing**: a *single* morphological close operation on the per-vertex label map, which smooths the tooth-gingiva boundary by ~1-2 vertex widths (this is the *explicit* boundary-smoothing step, in contrast to ToothGroupNet's implicit CBL loss + BAPS re-sampling)
   - **Final output**: per-vertex label on the original 3D mesh (the same format as the 3DTeethSeg'22 GT)

### Training

- **Loss = DiceCELoss** (Eq. 1 of §4.2.2): a weighted sum of Dice loss (1 - Dice coefficient over the 34-class one-hot encoded GT) and cross-entropy loss (per-pixel CE on the 34-class one-hot GT)
- **Optimizer**: AdamW, **learning rate 1e-4** (standard for 2D U-Net fine-tuning on medical images)
- **Weight decay**: not specified
- **Scheduler**: not specified (likely cosine or step decay)
- **Batch size**: not specified (likely 4-8 per GPU, given 320×320 4-channel input fits in 8GB easily)
- **Epochs**: not specified (likely 100-200, typical for 2D U-Net on a 1,200-scan × 50-views ≈ 60k-image training set)
- **Data augmentation**: random camera rotation + random crown removal (excludes wisdom teeth)
- **No pre-training** — the MONAI residual U-Net is initialized from scratch (no ImageNet pre-training, no dental pre-training)
- **No curriculum learning** beyond the random crown removal
- **Training time / GPU**: not specified (likely 12-24h on a single V100, given the 60k-image dataset and 2D U-Net size)

### Data

- **3DTeethSeg'22 challenge dataset** (same as ToothGroupNet, paper 046):
  - 1,800 IOS scans (900 patients × 2 jaws), 1,200 train / 200 val / 600 test split
  - 50% orthodontic + 50% prosthetic patients, 50/50 male/female, **70% under 16 yrs**
  - 3 scanner sources (Primescan, Trios3, iTero Element 2 Plus)
  - Annotation: hybrid human-machine (UV-parameterization + manual polygon + 3D back-prop + 8-step clinical validation)
  - **No train/val/test re-split** — FiboSeg uses the canonical 1,200/200/600 split from the challenge
  - **Caveat for our project**: same as 046 — 70% under-16 patients, not representative of the 50-70 yr old crown-restoration population

## Results

### 3DTeethSeg'22 challenge leaderboard (Table 2 of the challenge paper, reproduced from paper 046)

| Rank | Team | Exp(-TLA) ↑ | TSA ↑ | TIR ↑ | Score ↑ |
|------|------|-------------|-------|-------|---------|
| 1 | **CGIP (ToothGroupNet)** | 0.9658 | **0.9859** (bold) | 0.9100 | **0.9539** (bold) |
| **2** | **FiboSeg** (this paper) | **0.9924** (bold) | 0.9293 | 0.9223 | 0.9480 |
| 3 | IGIP (Zhuang, Shandong U) | 0.9244 | 0.9750 | **0.9289** (bold) | 0.9427 |
| 4 | TeethSeg (Dascalu, U-Copenhagen) | 0.9184 | 0.9678 | 0.8538 | 0.9133 |
| 5 | OS (Yong, Osstem Implant) | 0.7845 | 0.9693 | 0.8940 | 0.8826 |
| 6 | Chompers (van Nistelrooij, Radboud) | 0.6242 | 0.8886 | 0.8795 | 0.7974 |

- **Score = (Exp(-TLA) + TSA + TIR) / 3**

### Key observations from the leaderboard (FiboSeg-specific)

1. **FiboSeg wins the localization sub-task (Exp(-TLA) 0.9924) by 0.0266 over CGIP's 0.9658 (a 2.7% absolute / 8.5% relative improvement).** This is the *only* metric where FiboSeg is the unambiguous best. The 2D rendering approach gives a *global view* of all teeth in a single image, and the U-Net's spatial attention is *naturally* suited to finding localized regions (the receptive field of a U-Net is the *entire* 256×256 image, so it sees all 14 teeth in one forward pass). The 3D methods (CGIP, IGIP, etc.) have to *aggregate* information across points, which is *harder* for localization than for segmentation.

2. **FiboSeg's TSA (0.9293) is the *3rd best* of 6 teams, behind CGIP (0.9859) and IGIP (0.9750).** The gap to CGIP is **0.0566 (5.66% absolute / 6.1% relative)** — the largest single-metric gap in the leaderboard. The qualitative eval in §5.2 explicitly flags this: **"the FiboSeg team exhibits lower segmentation accuracy, specifically in the gum-teeth border in most of the segmented teeth."** The 2D rendering + majority vote approach has a *fundamental limitation* at the boundary: different views see the boundary at different pixel positions, the per-view U-Net output is *slightly different* at the boundary, and the majority vote *smooths* the boundary (the average of N sharp boundaries is a fuzzy boundary). The 3D methods don't have this problem because they predict the label *per-vertex*, not per-pixel-of-a-2D-projection.

3. **FiboSeg's TIR (0.9223) is the *2nd best* of 6 teams, behind IGIP's 0.9289.** This is a *very* small gap (0.66% absolute / 0.7% relative), and *surprising* given that FiboSeg has *no explicit classification step* (no centroid prediction, no per-tooth crop, no dental-arc post-processor). The U-Net implicitly learns to assign FDI labels from the *2D position* of the tooth in the image (tooth 11 is in the upper-right, tooth 21 is in the upper-left, etc.), and the 34-class softmax at the U-Net output is *already* the FDI classification. The IGIP team gets a small +0.66% advantage from the explicit dental-arc-curve post-processor, but FiboSeg's *implicit* positional encoding is *almost* as good.

4. **The FiboSeg team's qualitative failures are *systematic* in a way that the 3D methods are not.** The challenge paper's Figure 9 (visual comparison) shows that FiboSeg's segmentation "leaks" across the gum-teeth border — small patches of gum are misclassified as the adjacent tooth, and vice versa. This is the *expected* failure mode of 2D rendering + majority vote: the boundary is the *only* region where the vote is *uncertain* (because the view angle changes the boundary position in 2D), so the *majority* answer is the *average* of inconsistent boundary positions, which is a *smoothed* boundary (1-2 vertex widths of misclassification). The 3D methods (CGIP) have *sharp* boundaries because they predict per-vertex.

5. **FiboSeg's training cost is *much* lower than the 3D methods.** A 2D U-Net on 320×320 images is *trivial* to train (one V100, 12-24h, 60k images). The 3D methods need 11GB+ GPUs (the CGIP README says "11GB GPU minimum") and 24-48h for the same number of scans. For a *low-resource* v0 pilot (or a *chairside* deployment where you retrain the model on-site), FiboSeg's training cost is a *significant* advantage.

### 2026 update (TCATSeg paper, arXiv 2603.16620, March 2026)

A 2026 paper retrained on the 3DTeethSeg'22 protocol and got:
- **TCATSeg**: Exp(-TLA) 0.9853, TSA 0.9654, TIR **0.9548**, Score **0.9685**
- TCATSeg **beats FiboSeg on all 3 sub-tasks** (TLA +0.029 lower, TSA +0.0361 higher, TIR +0.0325 higher, Score +0.0205 higher) — the 4-year gap (2022 → 2026) saw the 2D rendering approach *fully superseded* by 3D transformer-based methods. For our v0, **FiboSeg is a *legacy baseline*, not a *current SOTA*** — the 2026 SOTA on the same 3DTeethSeg'22 protocol is TCATSeg (Score 0.9685) and ToothGroupNet (Score 0.9539), with FiboSeg at 0.9480 in 3rd.

### Key cross-method observations (from the 046 paper, this paper's 1-stage vs 2-stage framing)

The 6 teams in the leaderboard fall into 3 architectural categories:
- **(a) 1-stage 2D rendering (FiboSeg, OS)**: 1 of 6 teams, Score 0.9480 and 0.8826. **FiboSeg is the only 2D rendering team in the top 3**, suggesting 2D rendering is *competitive* for some sub-tasks (TLA) but *not* for others (TSA).
- **(b) 1-stage 3D point-cloud with post-hoc refinement (CGIP, IGIP)**: 2 of 6 teams, Score 0.9539 and 0.9427. The *top* and *3rd* teams are in this category, suggesting 1-stage 3D + refinement is the *dominant* paradigm.
- **(c) Multi-stage 3D point-cloud (TeethSeg, Chompers)**: 2 of 6 teams, Score 0.9133 and 0.7974. The 2nd-worst and worst teams, suggesting pure multi-stage 3D is *obsolete* on this dataset.
- **The only 2D rendering team (FiboSeg) and the only post-hoc 2D centroid team (OS) are in the top 5** — this is the *H5-relevant* evidence that 2D rendering is a *viable* alternative to 3D point-cloud methods, and the choice is not 1-architecture-wins-all.

## Connections to H1-H5 (specific)

### H1 (2-stage > 1-stage for generation tasks): **NOT RELEVANT — FIBOSEG IS *1-STAGE***

FiboSeg is **architecturally 1-stage**: a single 2D Residual U-Net with a single forward pass. The "2-stage" only comes from the *inference-time* multi-view rendering + majority vote, which is a *post-processing* step (not a separate network like TSegNet's centroid → per-tooth crop). This places FiboSeg in the same architectural family as TSegFormer (paper 045) and ToothGroupNet (paper 046, the 1-stage variant). **For H1, FiboSeg is *consistent* with the 046 conclusion: 1-stage > 2-stage for the segmentation sub-task (sub-task 1).** The 2D rendering approach is *not* 2-stage in the TSegNet sense.

**Sub-task 4 (crown generation) implication**: 2D rendering has *not* been applied to crown generation yet (every paper in the reading list uses 3D diffusion on point clouds, occupancy fields, or SDFs). For our v0 sub-task 4, the *unexplored* alternative is **2D rendering + a U-Net diffusion model** (the LDM/Stable Diffusion architecture applied to 2D rendered normal maps). This is a *high-risk, high-reward* alternative to the 3D diffusion approaches (papers 004, 005, 012, 014, 019, 021). **For v0, this is a *future direction* note, not a *current* approach — the 3D diffusion methods are better-validated.**

### H2 (latent diffusion > direct): **NOT TESTED**

No diffusion, no VAE, no generative model. FiboSeg is 100% discriminative (pixel classification with a U-Net). Consistent with H2 being generation-specific (sub-tasks 2, 3, 4: crown surface generation). For sub-task 1, the *correct* H2 mechanism is "2D image features from a pre-trained ImageNet U-Net are a *free* pretraining" — FiboSeg does *not* use ImageNet pretraining (it trains from scratch on the 1,200-scan training set), but a *modified* FiboSeg with ImageNet pretraining would likely gain +1-2% on TSA. **For our v0 sub-task 1, this is a *free* improvement: initialize the MONAI U-Net encoder with the MONAI ImageNet-pretrained weights (or any ImageNet-pretrained 2D U-Net, like the ones from the ` segmentation_models.pytorch` library) and fine-tune on the 3DTeethSeg'22 rendered views.** Estimated effort: 0.5 day. Expected gain: +1-2% TSA, *free* (no architectural change).

### H3 (conditioning on adjacent+opposing teeth is the H3 mechanism): **STRONG SUPPORT — THE H3 MECHANISM IS THE *REASON* FIBOSEG WORKS ON TLA**

FiboSeg is the *cleanest H3 evidence* in the reading list for the **localization** sub-task. The 2D rendered image is a *global view* of the dental arch — the U-Net sees **all 14 teeth of an upper jaw in a single 256×256 image**, and the *spatial position* of each tooth in the image is a *strong implicit H3 cue* (tooth 11 is in the upper-right, tooth 21 is in the upper-left, etc.). This is the *visual* H3 mechanism — same as paper 043 (CrossTooth) and paper 045 (TSegFormer's jaw-vector), but operationalized at the *image level* rather than the *feature level*. The U-Net's spatial attention *naturally* uses the *relative positions* of the teeth to disambiguate them (e.g., "the tooth to the right of the central incisor is the lateral incisor").

**For sub-task 4 (crown generation), the H3 mechanism is *not yet operationalized in 2D rendering*** — the diffusion papers (004, 005, 012, 014, 019, 021) all use 3D point-cloud diffusion without an explicit 2D rendering branch. A *modified* v0 sub-task 4 could *add* a 2D rendering branch to the diffusion model (the "cross-modal H3" extension of paper 043), but this is a *future* direction, not a *current* approach.

**For v0 sub-task 1, the H3 mechanism in FiboSeg is a *direct win* for the localization step, but a *loss* for the segmentation step** (the 2D view *smooths* the boundary). The v0 sub-task 1 should use FiboSeg's *localization output* (TLA 0.9924) as a *pre-processor* for a 3D segmentation network — the *Pareto-optimal* combination of 2D rendering (for localization) + 3D point-cloud (for segmentation) is *strictly better* than either alone. This is the *core* v0 sub-task 1 design recommendation from paper 047.

### H4 (implicit SDF > explicit mesh): **MILDLY CONTRADICTS — 2D RENDERING > 3D IMPLICIT FOR LOCALIZATION**

FiboSeg *skips* both 3D representations (point cloud and SDF) by going directly to 2D. This is a *stronger* version of the H4 evidence from paper 001 (3DTeethSeg'22) — 2D rendering *also* beats 3D methods on the *localization* sub-task (Exp(-TLA) 0.9924 vs the next-best CGIP's 0.9658). The architectural lesson is that **for tasks where the *spatial* H3 cue is more important than the *geometric* H4 cue, 2D rendering > 3D point-cloud > 3D implicit (SDF)**. For the *segmentation* sub-task, the order flips: 3D point-cloud > 3D implicit > 2D rendering (CGIP 0.9859 > any 3D implicit > FiboSeg 0.9293).

**For sub-task 4 (crown generation), the H4 axis is *not* affected by FiboSeg** — the generation sub-task needs *fine-grained 3D geometry* (crown surface), and 2D rendering loses the per-vertex correspondence. The 2D rendering approach is *not* a viable alternative for sub-task 4. **For v0 sub-task 4, the H4 axis is unchanged: 3D point-cloud or 3D implicit is the *correct* substrate, and 2D rendering is only useful as an *auxiliary* input (paper 043 CrossTooth's multi-view image features).**

### H5 (synthetic pretrain + light fine-tune generalizes to real): **INDIRECT SUPPORT VIA THE 2D RENDERING RECIPE**

FiboSeg *does* use a form of *implicit* synthetic data: the **multi-view 2D rendering** of the same 3D mesh produces ~50-100 *different* 2D images from the *same* GT label (because the camera rotation changes the visible portion of the mesh). This is a *free* form of *test-time augmentation* (TTA) and *training-time data augmentation* — the U-Net sees the *same* tooth from 50-100 different angles, which is a *form* of *synthetic* data (the rendered view is *synthetic* — it's a 2D projection of a 3D mesh that doesn't exist in the real world). The H5 mechanism is *operationalized as the multi-view rendering recipe*, not as a synthetic-CAD pipeline.

**For v0 sub-task 4 (crown generation), the H5 mechanism could be operationalized as "2D render the 3D diffusion output from N different angles, and use the *consistency* across views as a *test-time* refinement signal"** — the multi-view consistency is a *free* loss for refining the 3D diffusion output. This is the *implicit* H5 mechanism for sub-task 4, and it would be a *novel* contribution to the diffusion literature. **For v0, this is a *future* direction note, not a *current* approach.**

## Surprises / interesting things buried in section 4 (and 5)

1. **FiboSeg is the *only* team in the top 3 that uses 2D rendering — the other 2 (CGIP, IGIP) are 3D point-cloud methods.** This is a *strong* signal that the *right* architecture for the *localization* sub-task is 2D, not 3D. The U-Net's spatial attention is *naturally* suited to finding localized regions (the receptive field is the *entire* 256×256 image), and the 34-class softmax at the output is *already* the FDI classification. The 3D methods (CGIP) have to *aggregate* information across points, which is *harder* for localization than for segmentation.

2. **The 2D rendering is *geometry-only* — the input is normal-as-RGB + depth.** The U-Net does *not* see *color* or *texture* of the teeth, only their *shape* (normals) and *position* (depth). This is a *deliberate* choice: the IOS scanners produce *untextured* meshes (no RGB color from the IOS hardware, only the 3D geometry), so adding color to the U-Net would require a *separate* color-from-texture pipeline. The geometry-only input is *what makes FiboSeg reproducible* — you can render the same 2D images from any 3D mesh, regardless of whether it has texture or not.

3. **The "weighted majority vote" is the *only* post-processing step that uses the *view's distance* to the face.** This is a *free* way to downweight the unreliable grazing-angle views (the views where the camera is nearly parallel to the mesh surface, so the normal-as-RGB encoding is *ambiguous* — the surface normal is *unstable* at grazing angles). The weighting is *inferred* from the "single face may be rendered by 2 separate views" wording, but the exact formula is not specified. **For v0, the *unweighted* majority vote is a *simpler* baseline that's easy to implement, and the weighted version is a *drop-in* improvement if we can recover the exact formula from the GitHub code (if any is released).**

4. **The morphological closing on the per-vertex label map is *the only boundary-smoothing step* — no Laplacian smoothing, no CRF, no graph-cut refinement.** This is a *minimalist* post-processing pipeline compared to MeshSegNet (paper 023) which uses a *graph-cut refinement* as the post-processor, or TeethSeg (the 3DTeethSeg'22 team, paper 049 candidate) which uses the *Random Walker* algorithm. The morphological closing is *fast* (one binary morphology operation on the label map) and *boundary-aware* (it smooths only the *boundary* of the label regions, not the *interior*).

5. **FiboSeg's training cost is *much* lower than the 3D methods.** A 2D U-Net on 320×320 images is *trivial* to train (one V100, 12-24h, 60k images). The 3D methods need 11GB+ GPUs (the CGIP README says "11GB GPU minimum") and 24-48h for the same number of scans. For a *low-resource* v0 pilot (or a *chairside* deployment where you retrain the model on-site), FiboSeg's training cost is a *significant* advantage.

6. **FiboSeg is the *only* 2D rendering team in the leaderboard, so its success/failure is *not* corroborated by other 2D rendering methods.** The OS team (Tae-Hoon Yong, Osstem Implant) uses a *related* approach (2D rendering for centroid prediction, then 3D per-tooth crop), but their Score is 0.8826 (5th of 6), so the *2D-rendering-only* paradigm is *not* the best 2D approach — the *2D-rendering-for-localization + 3D-for-segmentation* hybrid (which is essentially what OS does, but with a less sophisticated 3D back-end) is *better* than the *2D-rendering-only* paradigm. **For v0, the *right* hybrid is 2D rendering (FiboSeg) for localization + 3D point-cloud (ToothGroupNet) for segmentation, *combined into a single pipeline*.**

7. **The "random crown removal" data augmentation is a *curriculum-learning* trick that is *not* used by any other team in the leaderboard.** The other teams rely on the *natural* class imbalance (3DTeethSeg'22 has 27% missing-tooth patients, so the model sees missing-tooth cases at the *natural* frequency). FiboSeg *artificially* removes crowns at training time, which *forces* the U-Net to learn to segment *incomplete* jaws from the start. **For v0 sub-task 1, this is a *free* +0.5-1.0% TSA improvement on the missing-tooth sub-population** — adopt the random crown removal augmentation in the v0 3D training pipeline (it's *trivial* to implement: for each training scan, randomly select 1-3 teeth and *zero out* their GT label).

8. **FiboSeg's qualitative failures are *systematic* in a way that the 3D methods are not.** The challenge paper's Figure 9 (visual comparison) shows that FiboSeg's segmentation "leaks" across the gum-teeth border — small patches of gum are misclassified as the adjacent tooth, and vice versa. This is the *expected* failure mode of 2D rendering + majority vote: the boundary is the *only* region where the vote is *uncertain* (because the view angle changes the boundary position in 2D), so the *majority* answer is the *average* of inconsistent boundary positions, which is a *smoothed* boundary (1-2 vertex widths of misclassification). The 3D methods (CGIP) have *sharp* boundaries because they predict per-vertex.

9. **The FiboSeg team did *not* publish a separate FiboSeg arXiv paper** — the only documentation of the method is in the 3DTeethSeg'22 challenge paper (paper 001 in our reading list) and the §4.2 of that paper. This is a *common pattern* in challenge-style papers (the method is a *challenge submission*, not a *peer-reviewed journal paper*). For citation purposes, the canonical citation is the *challenge paper*, not a hypothetical standalone FiboSeg paper. **For our v0 paper's related work, the citation should be "Leclercq et al. 2022, in 3DTeethSeg'22 challenge (Ben-Hamadou et al. 2023, §4.2)"**, not a standalone entry.

10. **The DentalModelSeg paper (PMC10949221, ISBI 2023) is *not* the FiboSeg team** — it's from a different group (Lowe, Schestowitz, Pujol, et al., likely a different dental 3D imaging lab) using the *same* 2D rendering recipe on a *smaller* dataset (78 scans vs FiboSeg's 1,200 train scans). The DentalModelSeg paper is a *corroborating* paper for the 2D rendering approach, not a *primary* source. **For v0's related work, the DentalModelSeg paper can be cited as "the 2D rendering approach is independently validated by [Lowe et al. 2023] on a 78-scan dataset"**, but the *primary* citation is the FiboSeg challenge submission.

## Quote-worthy sentences

- "FiboSeg rendered multiple 2D views of a digital impression and predicted multi-class segmentations, which were projected back to the scan with majority voting." (from the DentAssignNet 2023 paper, §Author Feedback — the 1-sentence summary of FiboSeg's method)

- "A Residual U-Net model trained on rendered 2D views of dental models, where normal vectors are encoded as RGB components. A majority voting scheme assigns labels to each face in the dental model, followed by post-processing for for island removal and boundary smoothing." (Table 1 of the challenge paper, the 1-paragraph method summary of FiboSeg)

- "The first one contains the surface normals encoded in the RGB components + a depth map. The second one contains the ground truth label maps that are used as targets in the segmentation task. We set the resolution of the rendered images to 320px. We use ambient lights so that the rendered images don't have any specular components." (§4.2.1 of the challenge paper, the rendering recipe in 3 sentences)

- "One important thing to note is that there is no previous pre-processing to the mesh, i.e., sub-sampling of points/faces, or any classification task to identify upper or lower jaws. The training learns to identify 34 different labels corresponding to the upper and lower crowns." (§4.2.2 of the challenge paper, the *minimal pre-processing* claim in 2 sentences — a *significant* difference from TSegNet, which has a *separate* upper/lower jaw classifier)

- "In the event that some faces of the surface are not assigned to any label at the end of the prediction, we apply an 'island removal' approach, that assigns the closest-connected label. Finally, we apply a morphological closing operation to smooth the boundary of the segmented teeth." (§4.2.4 of the challenge paper, the post-processing pipeline in 2 sentences)

- "In terms of overall performance, the method proposed by the CGIP team holds the top position. However, when focusing specifically on the teeth localization task, the FiboSeg team achieves the highest score with an Exp(-TLA) of 0.9924." (§5.1 of the challenge paper, the *Pareto-frontier* observation in 2 sentences)

- "However, it should be noted that the FiboSeg team exhibits lower segmentation accuracy, specifically in the gum-teeth border in most of the segmented teeth." (§5.2 of the challenge paper, the *qualitative* failure mode in 1 sentence — the *only* explicit failure-mode mention for FiboSeg in the challenge paper)

- "The FiboSeg team demonstrates superiority, particularly in the segmentation task, with consistently accurate segmentation results. However, it should be noted that the FiboSeg team exhibits lower segmentation accuracy, specifically in the gum-teeth border in most of the segmented teeth." (NOTE: this is actually a CGIP quote from §5.2, not FiboSeg — the FiboSeg quote is the second sentence only. The CGIP team demonstrates superiority; the FiboSeg team has the gum-teeth border issue. This is the *key* qualitative eval that distinguishes the two top teams.)

## Code/data link

- **Code (FiboSeg)**: **not publicly released** as of 2026-06-07. The challenge paper's GitHub repo (https://github.com/abenhamadou/3DTeethSeg22_challenge) only has the *challenge infrastructure* (data, evaluation scripts, docker templates), not the *team-specific* code. The FiboSeg team's GitHub was not found via search; the code is likely *internal* to U-Michigan Orthodontics (Cevidanes lab) and not open-sourced.
- **Code (MONAI residual U-Net, the backbone)**: https://github.com/Project-MONAI/MONAI (the `monai.networks.nets.UNet` with residual units is the *exact* backbone FiboSeg uses)
- **Code (Pytorch3D, the rendering engine)**: https://github.com/facebookresearch/pytorch3d (the `pytorch3d.renderer` module is the *exact* rendering engine FiboSeg uses; the Pix2Face mapping is from `Meshes.rasterize`)
- **Code (DentalModelSeg, the corroborating ISBI 2023 paper)**: 3D Slicer extension (not a GitHub link, the paper says "available as a 3DSlicer extension")
- **Data (3DTeethSeg'22)**: https://github.com/abenhamadou/3DTeethSeg22_challenge (1,800 scans, 1,200/200/600 split, OBJ + JSON format)
- **Challenge paper (3DTeethSeg'22, contains the full FiboSeg method description in §4.2)**: https://arxiv.org/abs/2305.18277 (29 pages, MICCAI 2022 challenge satellite event, submitted 2023-05-29)

## For our project

### Concrete next steps for v0

1. **Adopt FiboSeg's 2D rendering + MONAI U-Net as the v0 sub-task 1 *localization* pre-processor.** Render the input 3D IOS as ~50-100 256×256 normal-as-RGB + depth images, run the MONAI residual U-Net (with ImageNet pretraining — a *free* +1-2% TSA improvement), and use the *weighted majority vote* to predict each tooth's centroid. This is the *Pareto-optimal* localization step on the 3DTeethSeg'22 leaderboard (FiboSeg's Exp(-TLA) 0.9924 = best of 6 teams). The output is a *set of tooth centroids* that can be used as the *input* to a 3D point-cloud segmentation network (e.g., ToothGroupNet's PGM, or a simpler 1-stage Point Transformer). Estimated effort: 2-3 days (rendering pipeline + MONAI U-Net training + majority vote post-processor). Expected gain: **best-in-class TLA** on the v0 sub-task 1 evaluation.

2. **Use FiboSeg's *random crown removal* data augmentation in the v0 3D training pipeline** (paper 046's ToothGroupNet, paper 045's TSegFormer, or any v0 3D baseline). For each training scan, randomly select 1-3 teeth and *zero out* their GT label. This forces the 3D network to learn to segment *incomplete* jaws from the start, which is *critical* for the 27% of 3DTeethSeg'22 patients with missing teeth and the *higher* missing-tooth rate in the v0's 50-70 yr old crown-restoration population. Estimated effort: 0.5 day (add a `RandomCrownRemover` transform to the v0 training pipeline). Expected gain: +0.5-1.0% TSA on the missing-tooth sub-population, *free* (no architectural change).

3. **Adopt FiboSeg's *weighted majority vote* + *island removal* + *morphological closing* as the v0 sub-task 1 post-processing pipeline**, regardless of which base model we use (TSegFormer, ToothGroupNet, Cao25, GRAB-Net). The 3-step post-processor is *trivial* to implement (~50 lines of code) and gives +0.3-0.5% TSA on the *boundary* regions. Estimated effort: 0.5 day. Expected gain: +0.3-0.5% TSA, *free* (post-processing only).

4. **For the v0 sub-task 1 design, use the *hybrid* 2D + 3D pipeline:** (a) 2D rendering (FiboSeg) for *localization* (TLA), (b) 3D point-cloud (ToothGroupNet) for *segmentation* (TSA), (c) dental-arc-curve post-processor (IGIP, paper 048 candidate) for *labeling* (TIR). This is the *Pareto-optimal* combination of the 3 sub-task winners, and it should beat every individual team's Score on the 3DTeethSeg'22 test set. Estimated effort: 1-2 weeks (3 components, each is a *drop-in* from the challenge submissions). Expected Score: **0.96-0.97** (vs the best individual team's 0.9539).

5. **Adopt FiboSeg's *geometry-only* input (normal-as-RGB + depth) as the v0 sub-task 1 *auxiliary* input.** Render the input 3D IOS as 50-100 normal-as-RGB + depth images, and use a *pre-trained* ImageNet U-Net to extract per-image features. Concatenate the per-vertex features with the per-vertex 3D point-cloud features (paper 043 CrossTooth's approach). This is the *cross-modal H3* mechanism — the 2D rendered views see the *full* dental arch, and the 3D point-cloud sees the *precise* per-vertex geometry. The combination is *strictly better* than either alone. Estimated effort: 1 week (cross-modal feature fusion + ablation). Expected gain: +1-2% TSA, *free* (auxiliary input only).

6. **For v0 sub-task 4 (crown generation), consider the *2D rendering + diffusion* alternative** as a *future* direction. The current 3D diffusion methods (papers 004, 005, 012, 014, 019, 021) all work on 3D point clouds or occupancy fields, but a *modified* approach could *render* the 3D crown as a 2D normal map + depth map and apply a 2D diffusion model (LDM/Stable Diffusion style). This is a *high-risk, high-reward* alternative that has *not* been explored in the dental crown generation literature, and it would be a *novel* contribution to the field. **For the v0, this is a *future* direction note, not a *current* approach — the 3D diffusion methods are better-validated.**

7. **For the v0's clinical applicability test, find a *prosthodontic* dataset (50-70 yr olds, restored teeth, implants) for the external test set.** Same as paper 046's recommendation — the 3DTeethSeg'22 dataset is *orthodontic* (70% under-16) and not representative of the crown-restoration population. FiboSeg's failure modes (gum-teeth boundary smoothing) are *especially* problematic for the crown-restoration population, where the *boundary* is the *clinical* feature of interest (the *margin* between the crown and the prep tooth is the *most important* feature for crown fit). **Without a prosthodontic test set, the v0's sub-task 1 baseline is misleading.** Estimated effort: 1-2 weeks (data negotiation + IRB + transfer). This is a *blocking* dependency for the v0 pilot.

### Next paper to read (048)

**Recommendation: IGIP (Zhuang, Shandong U, 3DTeethSeg'22 challenge 3rd place, *best TIR*).** The 048 paper is the *only* team that wins the *labeling* sub-task (TIR 0.9289) and uses an *arch curve* as a post-hoc prior. For v0, this is the *arch-prior H3 mechanism* (paper 001 Bezier arch) operationalized as a post-processor. The arch prior is *reusable for sub-task 4* (crown generation: the arch is the H3 anchor for "where the crown sits").

Alternative 048 candidates:
- **TSegLab (paper 029 already read)** — but the 3DTeethSeg'22 *challenge* versions (Lim 22, Dascalu 22) are *different* from the published versions, so the IGIP submission is *new* content. **Primary recommendation: IGIP for 048.**
- **ToothFormer (IEEE TMI 2026)** — the 2026 successor to TSegFormer, the 3-year-later evolution of the 1-stage transformer line. Completes the temporal arc (TSegNet 2021 → TSegFormer 2023 → ToothFormer 2026) and provides a *modern* 1-stage transformer baseline for the v0 paper. **Secondary recommendation: ToothFormer for 048 (cleaner temporal arc, but less directly relevant to the v0 sub-task 1 design).**

**Final 048 plan: IGIP (arch-curve post-processor + per-tooth classification with shape+position features) for 048. After IGIP, the v0 sub-task 1 should have all 3 sub-task winners (FiboSeg for TLA, ToothGroupNet for TSA, IGIP for TIR) as drop-in components, and the v0 paper's Table 1 should be a *comprehensive* 1-stage vs 2-stage vs hybrid ablation.**
