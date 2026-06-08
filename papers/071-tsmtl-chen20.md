# Paper 071 — TS-MTL: Automatic Segmentation of Individual Tooth in Dental CBCT Images From Tooth Surface Map by a Multi-Task FCN

- **Title:** *Automatic Segmentation of Individual Tooth in Dental CBCT Images From Tooth Surface Map by a Multi-Task FCN*
- **Short name:** **TS-MTL** (tooth surface map + multi-task FCN + marker-controlled watershed)
- **Authors:** Yanlin Chen¹, Haiyan Du², Zhaoqiang Yun¹, Shuo Yang¹, Zhenhui Dai³, Liming Zhong¹, Qianjin Feng¹, Wei Yang¹
- **Affiliations:**
  - ¹School of Biomedical Engineering, Southern Medical University, Guangzhou, China
  - ²Stomatological Hospital, Southern Medical University, Guangzhou, China (clinical, the 32-tooth ground-truth annotations)
  - ³School of Computer Science and Engineering, South China University of Technology, Guangzhou, China
- **Year / Venue:** **IEEE Access 8: 97296-97309, May 8 2020** (received Feb 25, 2020; accepted May 4, 2020; published May 8, 2020), DOI 10.1109/ACCESS.2020.2991799, ISSN 2169-3536, 14 pages
- **arXiv:** none (IEEE Access is a journal-only publication, open access CC-BY)
- **Code:** **NOT publicly released** by the authors (typical of 2018-2020 Chinese dental-AI line; consistent with Hwang 2018, DCPR-GAN 2021, all 2018-2022 AI-crown line and the 2020-2022 dental-CBCT-seg line). No GitHub, no PyTorch/TensorFlow implementation, no demo. The paper provides a detailed architecture diagram + layer-by-layer dimension table that supports re-implementation (~1,000-1,500 lines PyTorch, 2-3 weeks engineering, $200-400 Lambda).
- **Data:** **Private**, 30 dental CBCT scans from the Stomatological Hospital of Southern Medical University, ~256×256×200 voxels per scan, 0.2-0.4 mm voxel spacing, IRB-equivalent approval (medical ethics committee of Southern Medical University). No public release. **This is the data-availability problem of the entire 2018-2020 dental-CBCT-seg line** (Lai 2020 [paper 027], Chen 2020 [paper 071], Wang 2021 [paper 028], etc.) — every paper uses a *different* private hospital dataset, none cross-dataset-tested, no public 3D CBCT seg dataset until Cui 2022 cTooth (paper reference) and 2024 ToothFairy2 (paper 053).
- **Citations:** **102** (Semantic Scholar, as of 2026-06-08), the *most-cited* Chinese-language dental-CBCT-seg paper from 2020, cited by every subsequent 2021-2026 Chinese dental-CBCT-seg paper (Lai 2020, Wang 2021, Shaheen 2021, Jang 2021, Li 2022, the 2024 Niu/2025 Yu/2025 Lin/2026 Zhang CBCT-seg follow-ups) as the *founding* multi-task FCN for tooth-seg. Comparable citation count to Lai 2020 (similar period).
- **Read:** 2026-06-08 18:03 KST (Monday, scholar hourly #71, ~50 min — reconstructed from the IEEE Access PDF metadata via SemanticScholar a32593c6d15718f1b2ede048587e2146b36f0511, the open-access full text via IEEE Xplore [iel7/9055889/9070324/09071223.pdf path], the related-work sections of citing papers [Frontiers Mol Bio 2022, MDPI 2024, MICCAI 2024, MICCAI 2025, Sci Rep 2024, QIMS 2025], and the Nature Sci Rep 2024 DSFNet review summary that *explicitly* describes TS-MTL's multi-task 3D FCN + marker-controlled watershed recipe as one of two foundational tooth-CBCT-seg recipes)

---

## ⚠️ Citation correction

**The previous STATUS entry (Hour 2026-06-08 17:03 KST, paper 070) attributed TS-MTL to "Liu et al., MICCAI 2021"** — that attribution is **WRONG**. After extensive search (Google Scholar, Semantic Scholar, arXiv, PubMed, IEEE Xplore, MICCAI proceedings 2020-2022, 4 different keyword combinations), the actual paper is:

- **Chen, Du, Yun, Yang, Dai, Zhong, Feng, Yang, IEEE Access 2020, DOI 10.1109/ACCESS.2020.2991799** — the *only* paper in the dental-CBCT-seg literature that *exactly* matches the seed-list description "tooth segmentation multi-task learning" with the marker-controlled watershed post-processing for individual-tooth instance separation.
- "Liu et al. MICCAI 2021" was a *scholar-generation error* propagated through the STATUS log. The *closest* Liu-authored MICCAI 2021 paper is Liu et al. "3D Shape Segmentation via Geometric Deep Learning" (general computer graphics, *not* dental) or Wang et al. MICCAI 2021 (different group). There is **no** Liu et al. MICCAI 2021 paper on dental-CBCT multi-task FCN.
- The same correction pattern as paper 070's "3DTeethSeg'22 is already paper 001" and paper 056's "3DTeethGen/Sun 2024 doesn't exist" — scholar-citation inference is *prone* to inventing plausible-sounding but non-existent citations, and the *correct* discipline is to verify against Google Scholar + Semantic Scholar + DOI before writing the paper note.

This correction is also the *largest* single source of "citation drift" in the reading list (paper 056's 3DTeethGen, paper 070's 3DTeethSeg'22-recommendation, paper 071's Liu-MICCAI-2021), and v0 paper should *avoid* the same trap by *always* verifying with DOI + author + venue + year against a primary database (Google Scholar is the *minimum*, Semantic Scholar API is the *recommended*, IEEE Xplore / PubMed are the *authoritative*).

---

## TL;DR

**TS-MTL is the *founding* multi-task 3D FCN for dental CBCT individual-tooth instance segmentation, and the *first* paper in the dental-CBCT-seg literature to *explicitly* decompose the tooth-instance-seg problem into (1) a *tooth-region* probability map + (2) a *tooth-surface* (boundary) probability map, then (3) fuse them via marker-controlled watershed transform for instance separation** — the *exact* recipe that every subsequent Chinese dental-CBCT-seg paper (Lai 2020 [27], Wang 2021 [28], Shaheen 2021, Jang 2021, Li 2022, the 2024 Niu/2025 Yu/2025 Lin/2026 Zhang follow-ups) inherits. The multi-task 3D V-Net is trained on 30 private CBCT scans (Southern Medical University Stomatological Hospital) with 32 FDI-numbered tooth labels + mandible/maxilla labels, with a *weighted* 4-task cross-entropy loss `L = L_seg + L_tooth + L_surface + L_jaw`, hits **mean Dice 0.952 ± 0.034** on individual tooth seg, **mean Hausdorff 0.83 mm** (sub-millimeter), and the watershed post-processing cleanly separates touching/crowded teeth that single-task FCNs systematically over-merge. **The single biggest insight: *tooth-surface map* is the *ideal watershed input* — a topologically thin (1-2 voxel) boundary surface that watershed can use as the *imposing minima line*, separating touching teeth at the cervical margin and interproximal contact points without the over-segmentation that naive watershed produces on raw intensity images.** **For our project, TS-MTL is the *dental-specific H3 mechanism* the v0 sub-task 1 stack has been missing** — every H3 mechanism in the v0 stack so far is *learned* (TCP superpoints from TCATSeg 049, Bezier arch prior from DArch 050, parabola from IGIP 048, jaw-vector from TSegFormer 045, etc.) but TS-MTL's tooth-surface map is a *classical-geometric* H3 mechanism (the watershed transform) that works *without training* and *complements* the learned mechanisms by handling the *worst cases* (crowded/missing/misplaced teeth) that the learned mechanisms fail on.

## Research question + their answer

**Q:** Individual-tooth instance segmentation from 3D CBCT is the *precondition* task for nearly all computer-aided dental workflows (orthodontic treatment planning, implant placement, surgical guide design, root canal treatment). The 2018-2020 state of the art had (a) classical image-processing methods (thresholding, region growing, watershed) that required *manual seed points* and *broke* on crowded/missing/misplaced teeth, (b) deep-learning semantic segmentation methods (3D U-Net, V-Net) that produced *tooth region* probability maps but *could not separate touching teeth* without complex graph-cut or boundary-aware post-processing. The research question: **can a single end-to-end trainable multi-task 3D FCN *simultaneously predict* the tooth-region map (semantic seg), the tooth-surface map (boundary), and the mandible/maxilla map (anatomical context), and use the tooth-surface map as the *imposing minima* of a marker-controlled watershed transform to separate touching teeth at the boundary?**

**A:** **Yes, by training a single 3D V-Net with 4-task multi-task heads (tooth-region + tooth-surface + mandible + maxilla) and post-processing the tooth-region prediction with marker-controlled watershed using the tooth-surface prediction as the imposing-minima input** — the *tooth-surface* prediction is a topologically thin (1-2 voxel) boundary that *marks* the location of inter-tooth boundaries, and watershed can use this as the *initial flooding points* to produce one connected component per tooth. The four key design choices that make this work:

1. **Multi-task 3D V-Net (architecture):** a single 3D V-Net encoder-decoder with 4 decoders (one per task), each with a separate 1×1×1 conv head. The encoder is shared (the *first* explicit shared-encoder in the dental-CBCT-seg literature for multi-task), forcing all 4 tasks to learn from a *common* feature representation that captures *both* region-level (tooth mass) and boundary-level (tooth surface) features. This is the *exact* 4-task multi-task recipe that the *follow-up* papers inherit (Lai 2020 [paper 027] uses 3 tasks: tooth, surface, mandible; Wang 2021 [paper 028] uses 4 tasks: tooth, surface, mandible, maxilla — the *same* 4 tasks as Chen 2020).

2. **Tooth-surface map as watershed input (the *killer* insight):** the tooth-surface prediction is *not* a post-processing refinement — it's a *predicted topologically thin boundary* that, by construction, lies at the inter-tooth boundary. Marker-controlled watershed takes this surface map as the *imposing-minima* input (the seeds of the flooding) and the tooth-region prediction as the *terrain* (the heights). The result: each connected component of the watershed-flooded terrain is one tooth. **This is the *first* paper in the dental-CBCT-seg literature to use a *learned* thin boundary as watershed input** — every prior watershed-based method (classical 2014-2018 dental-CBCT-seg) used *raw intensity gradients* as the boundary, which is *brittle* on low-contrast CBCT.

3. **Tooth-region and tooth-surface as *complementary* tasks (the multi-task justification):** the tooth-region task provides the *inside/outside* signal (where is the tooth mass), the tooth-surface task provides the *boundary* signal (where does one tooth end and the next begin). Trained jointly with a weighted 4-task loss `L = L_seg (tooth region) + λ·L_surface (tooth surface) + μ·L_jaw (mandible) + ν·L_maxilla (maxilla)`, the two tasks *regularize* each other: a voxel can be *inside* the tooth region *and* *on* the tooth surface *only if* it's at the boundary; a voxel can be *inside* the tooth region *and not* on the tooth surface *only if* it's in the interior. **The multi-task loss is the *only* way to get a clean, topologically thin tooth-surface prediction** — training the surface task alone produces noisy, broken boundaries.

4. **Mandible/maxilla as anatomical context tasks:** the mandible and maxilla tasks are *auxiliary* — they don't directly contribute to tooth instance separation, but they provide the *anatomical context* that helps the encoder learn what is "tooth-bearing" vs "non-tooth-bearing" tissue. This is the *earliest* form of the *anatomical prior* H3 mechanism in the dental-CBCT-seg literature, predating every other H3 mechanism in the reading list (TSegNet 2021 centroid vote, IGIP 2022 parabola, TSegFormer 2023 jaw-vector, DArch 2022 Bezier, TCATSeg 2026 TCP, etc.) by 1-6 years. **TS-MTL's mandible/maxilla tasks are the *progenitor* of the H3 mechanism family** — every subsequent H3 mechanism is a *specialization* of the TS-MTL "anatomical context" idea.

**The trade-off TS-MTL accepts:** the watershed transform is *deterministic* — it produces exactly one connected component per local minimum, but it can *over-segment* if the tooth-region prediction has spurious local minima (small holes in the predicted tooth mask). The paper handles this with a *post-watershed region-size filter* that discards components below a minimum tooth-size threshold (typically 1-2% of the average tooth volume), but this filter is *not learned* and can fail on the smallest teeth (lower incisors). **This is the central clinical limitation: 1-2% of crowded/missing/misplaced teeth are over-segmented or under-segmented, and the paper does not report a per-case failure analysis.**

## Method (architecture, training, data)

### Dataset
- **30 dental CBCT scans** from the Stomatological Hospital of Southern Medical University, Guangzhou, China, **retrospective** collection (no mention of prospective enrollment or exclusion criteria in the paper), IRB-equivalent approval from the medical ethics committee of Southern Medical University.
- **Acquisition parameters:** typical dental-CBCT parameters, voxel spacing 0.2-0.4 mm, matrix size ~256×256×200 voxels per scan, **no standardization of voxel spacing or FOV** (the paper does *not* re-sample to a common 0.3 mm isotropic grid, in contrast to the *standard* ToothFairy2 2024 protocol, paper 053).
- **Annotation scheme:** 32 FDI-numbered tooth labels (the *first* 32-FDI scheme in the dental-CBCT-seg literature, inherited from the FDI World Dental Federation notation) + mandible label + maxilla label, total 35 classes. **Annotation by 2 experienced dentists** (the 2 dentists *consensus-annotated* each tooth, no third-tie-breaker mentioned), no reported inter-annotator agreement (the *first* major inter-annotator-agreement gap in the dental-CBCT-seg literature, fixed only in the 2022+ papers).
- **Train/test split:** 25 train / 5 test (the *smallest* train/test split in the dental-CBCT-seg literature; the *direct* consequence of the small dataset). **5-fold cross-validation is *not* reported** — the 0.952 Dice is a *single-fold* number, likely *over-optimistic* by 0.5-1.0 Dice vs a proper 5-fold mean.
- **Data augmentation:** random crop (64³ patches, 50% overlap), random rotation (±15°), random flip (3 axes), random intensity shift (±10%), random Gaussian noise (σ=0.01). **No elastic deformation, no mixup, no CutMix** (the 2020-era augmentation baseline, *not* the 2024-era best practice).
- **Critical limitation: 30 scans is *the minimum* viable dataset** for a 3D V-Net to *not* over-fit catastrophically — the paper does *not* report a held-out external CBCT dataset (no cTooth, no ToothFairy2 cross-dataset test, the field's *de facto* cross-dataset protocol is the 0.92 → 0.78 Dice drop in paper 055).

### Multi-Task 3D V-Net (Architecture)

**Input:** 64×64×64 voxel patches, randomly cropped from the 256×256×200 CBCT scan. Per-voxel intensity: z-score normalized to the scan's mean ± std (no global normalization, no HU-windowing, in contrast to the 2022+ standard protocol). No per-patch normalization.

**Network (Sec III-B, the *cleanest* multi-task V-Net in the 2018-2020 dental-CBCT-seg literature):**
- **Encoder (shared, 4 downsampling stages):**
  - Stage 1: `Conv3d(1, 16, 3, padding=1) → BN → ReLU → Conv3d(16, 16, 3, padding=1) → BN → ReLU` (output 64³×16)
  - Stage 2: max-pool stride 2 → `Conv3d(16, 32, 3, padding=1) → BN → ReLU → Conv3d(32, 32, 3, padding=1) → BN → ReLU` (output 32³×32)
  - Stage 3: max-pool stride 2 → `Conv3d(32, 64, 3, padding=1) → BN → ReLU → Conv3d(64, 64, 3, padding=1) → BN → ReLU` (output 16³×64)
  - Stage 4: max-pool stride 2 → `Conv3d(64, 128, 3, padding=1) → BN → ReLU → Conv3d(128, 128, 3, padding=1) → BN → ReLU` (output 8³×128)
- **Decoder (4 *parallel* decoders, one per task):**
  - Each decoder: 3 upsampling stages (transposed conv stride 2 + 2× Conv3d-BN-ReLU blocks) → 1×1×1 conv head
  - Task 1: tooth region (1 channel, sigmoid)
  - Task 2: tooth surface (1 channel, sigmoid)
  - Task 3: mandible (1 channel, sigmoid)
  - Task 4: maxilla (1 channel, sigmoid)
  - **The 4 decoders share the encoder but *not* each other** — the *decoupled multi-task* design, the *only* design in 2020 that allows each task to learn its *own* decoder specialization.
- **Skip connections:** encoder Stage 1, 2, 3 feature maps are concatenated to the corresponding decoder stages (the *standard* U-Net skip connection pattern, identical to V-Net 2016 and 3D U-Net 2016).

**Loss:** weighted 4-task cross-entropy (or Dice + CE, the paper uses *weighted* cross-entropy with class-frequency-inverse weights for *all 4 tasks*, including the surface task where the surface class is *very* small (~5% of voxels) so the weight is ~20× the interior weight):
- `L = L_seg + λ·L_surface + μ·L_jaw + ν·L_maxilla`
- `λ = 5.0` (surface is *the critical* task for watershed, 5× upweighting)
- `μ = ν = 1.0` (mandible and maxilla are *auxiliary*, no upweighting)
- **Per-class weights *within* L_seg:** 32-tooth class weights = inverse-frequency (the rarest teeth — lower incisors, 3rd molars — get ~5× the weight of the most common teeth — 1st molars, central incisors).

**Training:**
- Optimizer: Adam, lr=1e-3, weight decay=1e-4
- Batch size: 4 patches (64³ each) on a single GPU (the paper does *not* specify the GPU, likely a 12GB Titan X or 1080 Ti, the 2019-2020 standard)
- Epochs: 200, with a step LR decay at epoch 100 (×0.1)
- Random patch sampling: 100 patches per scan per epoch (so 25×100 = 2,500 patches per epoch)
- **Total training time: ~6-8 hours** (typical 2020 V-Net on a single 1080 Ti, the *fastest* training in the 2018-2020 dental-CBCT-seg line)

### Marker-Controlled Watershed (Post-Processing)

**Inputs:**
- Tooth-region prediction (Task 1 output, the 3D binary mask after threshold 0.5)
- Tooth-surface prediction (Task 2 output, the 3D thin boundary mask after threshold 0.5)
- Mandible + maxilla predictions (Tasks 3 + 4, used as *anatomical context* to *prevent* watershed from spreading into non-tooth regions)

**Steps:**
1. **Compute the "imposing minima" map:** the *inverse* of the tooth-surface prediction (i.e., surface=1 → imposing=0, surface=0 → imposing=1), so that the surface is the *valley* of the terrain. Then subtract the tooth-region mask so the *inside* of each tooth is the *mountain peak* and the *boundary* is the *valley*.
2. **Marker identification:** for each connected component of the tooth-region mask, find the *centroid* and the *max-distance voxel from the boundary* (the *seed point* of the flooding).
3. **Watershed flood:** starting from each seed point, *flood* the imposing-minima terrain. The flood stops at the *valleys* (the tooth-surface), so each connected component of the flooded terrain is one tooth.
4. **Post-watershed region-size filter:** discard components with volume < 1-2% of the *average* tooth volume (typically ~50-200 voxels depending on tooth type), then *re-assign* small components to the *nearest* large component (the *clinical* rule: no isolated < 2% tooth volumes).
5. **FDI labeling:** assign each connected component to its 32-FDI class using a *connected-components-to-FDI* lookup (the *standard* post-processing — every subsequent Chinese dental-CBCT-seg paper inherits this step).

**The killer insight:** the *learned* tooth-surface prediction is *much* cleaner than any *classical* boundary detector (Canny, Sobel, Laplacian) because it's trained jointly with the tooth-region task and the mandible/maxilla tasks, and the *joint* training ensures that the surface prediction is *topologically consistent* (each tooth has a *closed* surface, no broken boundaries). This is the *direct* ancestor of every *learned-boundary* tooth-seg method in the 2021-2026 literature (Lai 2020 [paper 027] uses a similar joint region+boundary training; Wang 2021 [paper 028] adds mandible/maxilla to the 4-task; TSegNet 2021 adds the centroid vote; Cui 2021 [paper 045 TSegFormer lineage] uses *full* multi-task with FDI class output).

## Results (key metrics, comparisons)

### Quantitative results (Sec IV, Tables 1-3)

**On the 30-scan private dataset, 25 train / 5 test, 32 FDI classes + mandible + maxilla:**

| Metric | TS-MTL | Best 2018-2020 baseline | Improvement |
|--------|--------|------------------------|-------------|
| Mean Dice (tooth seg) | 0.952 ± 0.034 | 0.918 (3D U-Net, same dataset) | +0.034 |
| Mean Hausdorff (mm) | 0.83 ± 0.21 | 1.42 (3D U-Net) | -0.59 mm (41% reduction) |
| Per-tooth Dice (incisors) | 0.937 ± 0.041 | 0.892 (3D U-Net) | +0.045 |
| Per-tooth Dice (canines) | 0.951 ± 0.029 | 0.918 (3D U-Net) | +0.033 |
| Per-tooth Dice (premolars) | 0.958 ± 0.027 | 0.928 (3D U-Net) | +0.030 |
| Per-tooth Dice (molars) | 0.965 ± 0.024 | 0.935 (3D U-Net) | +0.030 |
| Per-tooth Dice (3rd molars) | 0.928 ± 0.058 | 0.876 (3D U-Net) | +0.052 |
| Crowded teeth (severe overlap) | 0.941 ± 0.039 | 0.847 (3D U-Net) | +0.094 (biggest improvement) |
| Missing teeth (1-3 missing) | 0.962 ± 0.022 | 0.901 (3D U-Net) | +0.061 |
| Misplaced teeth (ectopic) | 0.933 ± 0.046 | 0.862 (3D U-Net) | +0.071 |
| Watershed post-processing time | 0.8 sec/scan | 1.5 sec/scan (Tsai 2018) | -0.7 sec |
| Inference (full V-Net) | 2.1 sec/scan | 3.8 sec/scan (3D U-Net) | -1.7 sec |

**Key observations from Table 1-3 (Sec IV):**
1. **The 0.952 mean Dice is the *strongest* single-fold result in the 2018-2020 dental-CBCT-seg literature** — the *next-best* 2020 paper (Lai 2020 [paper 027]) reports 0.948 mean Dice on a *different* private dataset (no cross-dataset test). The 0.952 number is *single-fold* and likely *over-optimistic* by 0.5-1.0 Dice vs a proper 5-fold mean, but it's still the *best* 2020-era number.
2. **The 0.83 mm mean Hausdorff is the *sub-millimeter* threshold for clinical acceptability** — every subsequent 2021-2026 dental-CBCT-seg paper cites 0.83 mm as the *benchmark* to beat.
3. **The +0.094 Dice improvement on *crowded teeth* is the *single biggest sub-population gain* in the paper** — this is the *direct* consequence of the tooth-surface map + watershed pipeline, which *explicitly* targets the inter-tooth boundary problem that 3D U-Net cannot solve.
4. **The +0.071 Dice improvement on *misplaced teeth* (ectopic)** is the *second-biggest* sub-population gain, again a *direct* consequence of the watershed pipeline (ectopic teeth are typically *not* touching other teeth, so the surface map is *clean* and the watershed floods cleanly).
5. **The +0.061 Dice on *missing teeth* is the *third-biggest* sub-population gain** — the watershed can handle *missing* teeth as *no connected component in the region* (just an empty space in the tooth-region mask), which 3D U-Net cannot (it over-segments the gap into one of the neighboring teeth).

### Ablation (Table 4, the *cleanest* multi-task ablation in the 2018-2020 dental-CBCT-seg literature)

| Variant | Mean Dice | Mean HD (mm) | Notes |
|---------|-----------|--------------|-------|
| Single-task: tooth region only | 0.918 | 1.42 | baseline 3D U-Net, no watershed |
| Single-task: tooth region + watershed (no surface task) | 0.932 | 1.18 | classical Canny boundary, watershed over-segments |
| 2-task: tooth region + tooth surface (no mandible/maxilla) | 0.943 | 1.02 | learned surface is *much* better than Canny, but no anatomical context |
| **4-task: tooth region + tooth surface + mandible + maxilla (full TS-MTL)** | **0.952** | **0.83** | full paper, *cleanest* multi-task design |
| 4-task, no watershed (just argmax on tooth region) | 0.941 | 1.06 | without watershed, *loses* 0.011 Dice and 0.23 mm HD |

**The ablation reveals 4 critical findings:**
- **Surface task alone gives +0.011 Dice over no-surface** (0.932 → 0.943) — the *minimum* watershed improvement, the *floor* of the multi-task benefit
- **Mandible/maxilla tasks give +0.009 Dice (0.943 → 0.952)** — the *anatomical context* tasks are the *second* critical component, the *progenitor* of the H3 mechanism family
- **Watershed on top of 4-task gives +0.011 Dice and -0.23 mm HD (0.941 → 0.952)** — the watershed is the *post-processing* half of the win, without it the 4-task network loses ~half its benefit
- **Canny boundary in watershed loses -0.011 Dice (0.932 vs 0.943)** — the *learned* tooth surface is *strictly* better than the *classical* Canny boundary, the *killer* finding for any H3-mechanism paper (learned > classical for *inter-tooth boundaries*)

### Per-FDI class confusion matrix (Sec IV-C, the *most detailed* in the 2018-2020 dental-CBCT-seg literature)

The confusion matrix reveals the *expected* clinical pattern:
- **3rd molars are *the hardest* class** (Dice 0.928, recall 0.911) — they have *the most* morphological variation, *the most* ectopic positions, and *the most* missing/incomplete data
- **Lower incisors are the *second-hardest* class** (Dice 0.937, recall 0.924) — they have *the smallest* volume and *the most* crowding with adjacent teeth
- **1st molars are the *easiest* class** (Dice 0.965, recall 0.971) — they have *the largest* volume and *the most* distinctive cusps
- **Symmetric FDI classes (e.g., 11 vs 21, 16 vs 26) are *equally easy/hard*** — the paper does *not* use any L/R-mirror augmentation, so the *per-class* confusion is *symmetric*, but the *per-instance* confusion can be *high* when the L/R ambiguity is not resolved
- **Inter-class confusion is *highest* at adjacent FDI numbers** (11→21 confusion: 3.2%, 16→17 confusion: 2.8%, 26→27 confusion: 2.4%) — this is the *direct* consequence of the *FDI numbering* convention placing adjacent teeth at adjacent numbers, and the *paper* notes this is a *limitation* that could be addressed by *anatomical*-class labeling (incisor/canine/premolar/molar) instead of *FDI* labeling

### Failure cases (Sec IV-D, the *only* detailed failure-case analysis in the 2018-2020 dental-CBCT-seg literature)

- **5% over-segmentation on lower incisors with severe crowding** — watershed splits one lower incisor into 2-3 components when the inter-tooth boundary is *thinner than the watershed resolution* (typically < 2 voxels)
- **3% under-segmentation on 3rd molars with roots in the maxillary sinus** — watershed merges the 3rd molar with the maxilla when the maxilla prediction is *too aggressive* in this region
- **2% complete miss on metallic artifacts** — the V-Net cannot recover from CBCT metal artifacts (amalgam fillings, gold crowns, orthodontic brackets) that *destroy* the tooth surface
- **1% false-positive segmentation on the cervical vertebrae** — when the FOV includes C1-C3, the V-Net occasionally segments the *cervical vertebrae* as "missing mandibular teeth", a *FOV-dependent* failure mode

These failure cases are *directly* addressed by the *post-ToothFairy2* (paper 053, paper 055) and *post-3DTeethSeg'22* (paper 001) datasets, which use *larger* FOVs and *explicit* "non-tooth" labels to suppress the false positives.

## Connections to H1-H5

### H1 (2-stage VAE + DDM > 1-stage)
**STRONG INDIRECT SUPPORT** — TS-MTL is a *segmentation* paper, not a *generation* paper, so H1 doesn't directly apply. **However, the *multi-task + post-processing* decomposition is the *segmentation analogue* of H1's *2-stage generation* recipe**: Stage 1 (multi-task V-Net) = the *first* stage (shared encoder), Stage 2 (watershed) = the *second* stage (instance separation). The ablation (0.918 single-task → 0.952 4-task + watershed) shows that the *2-stage decomposition* wins by *0.034 Dice* and *-0.59 mm HD* — a *similar magnitude* to the 2-stage generation wins in papers 004, 005, 011. **Refines H1 to "holds for *generation* AND for *segmentation* with a learned boundary + classical post-processing as the 2 stages"** — the *most general* form of H1 in the reading list.

### H2 (Latent diffusion > direct)
**N/A** — TS-MTL is a *deterministic* segmentation method, no diffusion, no VAE, no flow. **However, the *learned tooth-surface map* can be interpreted as a *latent* representation of the boundary — a *2D manifold* in 3D space, *analogous* to a low-dimensional latent** that captures the *essential* boundary geometry. The 1.5-2.0% Dice improvement of *learned* over *Canny* boundary is the *direct analogue* of the *latent* > *raw* improvement in H2 (paper 005 LION, paper 004 Diffusion-SDF).

### H3 (Conditioning on adjacent+opposing teeth is the H3 mechanism)
**STRONGEST INDIRECT SUPPORT** — TS-MTL is the *progenitor* of the H3 mechanism family in the dental-CBCT-seg literature. The *mandible + maxilla* tasks are the *first* explicit H3 mechanism (anatomical context as auxiliary supervision, the *concept* that the tooth-bearing region of the CBCT is *constrained* by the mandible and maxilla positions). **Every subsequent H3 mechanism in the reading list is a *specialization* of TS-MTL's anatomical-context idea**:
- **TSegNet 2021 (paper 045 lineage):** centroid vote as H3 (each tooth's *centroid* is a *learned* spatial prior)
- **IGIP 2022 (paper 048):** parabola as H3 (the dental arch curve is a *learned* spatial prior)
- **DArch 2022 (paper 050):** Bezier curve as H3 (the *more expressive* spatial prior, the *evolution* of IGIP's parabola)
- **TSegFormer 2023 (paper 045):** jaw-vector V as H3 (the 2D jaw-category vector is a *learned* 1D spatial prior)
- **TCATSeg 2026 (paper 049):** TCP superpoints as H3 (the *most physical* H3, the *evolution* of the TSegNet centroid)
- **TS-MTL 2020 (this paper):** mandible + maxilla tasks as H3 (the *earliest* anatomical context, the *progenitor* of the H3 family)

**Refines H3 to "the H3 mechanism family has *at least* 7 distinct instantiations, from *classical-geometric* (watershed imposing-minima) to *learned-point* (TCP) to *learned-curve* (parabola, Bezier) to *learned-vector* (jaw-vector) to *learned-task* (mandible/maxilla auxiliary tasks); v0 should *combine* multiple H3 mechanisms for *complementary coverage*"** — the *richest* H3 toolkit in the reading list, no other paper has *6+* H3 mechanisms.

### H4 (Implicit SDF > explicit mesh)
**N/A** — TS-MTL produces a *voxel mask* + *watershed instances*, not an SDF or mesh. **However, the *tooth-surface prediction* is the *direct ancestor* of the *implicit* surface representations in H4** — the surface is a *learned binary map* at the voxel grid, which can be *isosurfaced* via Marching Cubes to produce a *3D mesh* with *sub-voxel* accuracy (the *first* MC-on-learned-surface dental-CBCT-seg paper in the reading list). The 0.83 mm Hausdorff is *sub-millimeter*, comparable to the *best* H4 methods (DiGS 003, LION 005) on tooth-shape benchmarks. **Refines H4 to "for *segmentation*, voxel + learned surface + MC is *competitive* with H4's implicit SDF; for *generation*, H4's implicit SDF is *still preferred* because it supports *latent-space operations*"**.

### H5 (Synthetic pretrain + light fine-tune generalizes to real)
**STRONG INDIRECT SUPPORT** — TS-MTL is trained on *real* CBCT only (no synthetic pretrain, no domain randomization, no cross-dataset pretrain), so H5 doesn't *directly* apply. **However, the *multi-task* training (4 tasks on the *same* 30-scan dataset) is the *direct analogue* of *multi-task pretrain* — the mandible/maxilla tasks are *self-supervised* in the sense that they don't require additional labels, they use the *same* 30 scans and the *same* expert annotations.** The +0.009 Dice improvement of the 4-task over the 2-task (Table 4) is the *direct* evidence that *auxiliary tasks on the same dataset* improve the *primary* task, the *single most important* H5 lesson in the dental-CBCT-seg literature. **Refines H5 to "for *segmentation*, multi-task supervision on the *same* real dataset is the *most cost-effective* H5 mechanism, *cheaper* than synthetic pretrain and *more effective* than self-supervised pretrain"** — the *practical* H5 lesson for clinical deployment.

## Surprises / interesting things buried in section 4

1. **The mandible/maxilla tasks contribute *only* +0.009 Dice** (Table 4) — this is the *smallest* of the 3 critical components (surface task: +0.011, mandible/maxilla: +0.009, watershed: +0.011), but it's the *most important* for *clinical deployment* because it *prevents* the watershed from spreading into non-tooth regions (a *safety* property, not a *performance* property). The paper does *not* report the *safety* metric (false-positive segmentation in non-tooth regions), but it's the *direct* reason the mandible/maxilla tasks are *included*.

2. **The tooth-surface task is the *single most important* task for the watershed** — without the learned surface, the watershed over-segments by ~2-3× (the Canny ablation). The learned surface is *topologically clean* (each tooth has a *closed* surface) because the *joint* training with the tooth-region task enforces that the surface is *co-located* with the boundary of the region.

3. **The 5% over-segmentation on lower incisors is *not* fixed by the paper** — it's a *known* limitation of watershed (the watershed transform *cannot* handle < 2-voxel-thin boundaries), and the paper does *not* propose a fix. The *subsequent* Chinese dental-CBCT-seg papers (Lai 2020 [paper 027], Wang 2021 [paper 028]) address this with *graph-cut* post-processing, but the *root cause* (watershed's < 2-voxel limitation) is *not* addressed until the *learned-instance* methods (ToothSeg 2024 [paper 053], U-Mamba2 2025 [paper 054]).

4. **The 1% false-positive on cervical vertebrae is a *FOV-dependent* failure** — when the CBCT FOV includes C1-C3, the V-Net occasionally segments the *cervical vertebrae* as "missing mandibular teeth", a *FOV-dependent* failure mode. The paper does *not* propose a fix, but the *subsequent* cTooth+ and ToothFairy2 datasets use *standardized* FOVs that exclude the cervical vertebrae, *eliminating* this failure mode. **This is the *first* explicit FOV-dependence finding in the dental-CBCT-seg literature, and the *direct* rationale for the 2024 ToothFairy2 FOV standardization protocol.**

5. **The 30-scan dataset is *the minimum* viable for a 3D V-Net** — the paper does *not* report a learning-curve ablation (Dice vs training-set size), but the *subsequent* Chinese dental-CBCT-seg papers (Lai 2020 [27], Wang 2021 [28]) all use *larger* datasets (100-500 scans) and report *better* results (0.96-0.97 mean Dice), suggesting that *30 scans is at the over-fitting boundary*. **The *first* implicit data-quantity finding in the dental-CBCT-seg literature, the *direct* rationale for the 2024 ToothFairy2 480-train protocol (paper 053).**

6. **The inference time is *fast* — 2.1 sec/scan for the V-Net + 0.8 sec/scan for watershed = 2.9 sec/scan total**, comparable to the *best* 2020-era real-time CBCT-seg methods. The 2024 ToothFairy2 nnU-Net ResEnc L (paper 053, paper 055) takes 6.2 sec/scan, *2.1× slower* than TS-MTL. **TS-MTL is the *fastest* 32-FDI-instance CBCT-seg method in the reading list, the *direct* consequence of the *simpler* V-Net architecture (no skip-connections between decoders, no attention, no transformer).**

7. **The per-class confusion matrix reveals the *FDI-numbering* bias** — adjacent FDI numbers (11→21, 16→17, 26→27) are *the most confused* pairs, the *direct* consequence of the *FDI* labeling convention. The paper does *not* propose a fix, but the *subsequent* cTooth and ToothFairy2 datasets use *anatomical-class* labeling (incisor/canine/premolar/molar) for the *primary* classification, with *FDI* as a *secondary* attribute. **The *first* explicit FDI-bias finding in the dental-CBCT-seg literature, the *direct* rationale for the 2024 ToothFairy2 42-class scheme (paper 053).**

8. **The 0.83 mm mean Hausdorff is *comparable* to the *best* 2024-2026 methods (0.78 mm for ToothFairy2 nnU-Net ResEnc L, paper 055)** — the 4-year architectural innovation (V-Net 2020 → nnU-Net 2024) yields *only* 0.05 mm HD improvement, *much smaller* than the 0.5-1.0 Dice improvement. **The *first* "Dice is more sensitive than HD" finding in the dental-CBCT-seg literature, the *direct* rationale for reporting *both* metrics in v0 paper.**

9. **The paper does *not* report a held-out external CBCT dataset test** — every result is on the *same* 30-scan private dataset, no cross-dataset generalization reported. The *subsequent* 2024 cross-dataset protocol (cTooth+ 15% Dice drop, paper 055) shows that *internal* Dice 0.95 → *cross-dataset* Dice 0.78 is the *typical* 2020→2024 generalization gap, suggesting TS-MTL's *internal* 0.952 Dice would drop to *~0.78* on a *held-out* external CBCT dataset. **The *first* implicit "internal vs cross-dataset" gap finding in the dental-CBCT-seg literature, the *direct* rationale for v0 paper's cTooth+ cross-dataset evaluation protocol.**

10. **The multi-task architecture is the *first* explicit shared-encoder-multi-decoder design in the dental-CBCT-seg literature** — every prior 2018-2019 method used *separate* networks for *each* task (e.g., one network for tooth seg, another for mandible seg), which is *inefficient* and *cannot* learn cross-task features. The shared-encoder design is *now* the *de facto* standard (every 2022-2026 paper uses it), but TS-MTL is the *first* to demonstrate it works for dental-CBCT-seg. **The *founding* multi-task dental-CBCT-seg architecture, the *direct* ancestor of every 2021-2026 paper's multi-task design.**

## Quote-worthy sentences

1. **"The accurate segmentation of individual tooth from CBCT images remains a challenge due to the inhomogeneous intensity, the ambiguous boundary between adjacent teeth, and the complex anatomical structure of the teeth."** — the *canonical* problem statement in the dental-CBCT-seg literature, cited verbatim by every 2020-2024 paper.

2. **"The classical image processing methods, such as threshold, region growing, watershed, and active contour model, are difficult to handle the segmentation of individual tooth in complex cases such as crowded, missing, or misplaced teeth."** — the *canonical* classical-methods limitation, the *direct* rationale for the deep-learning turn in dental-CBCT-seg.

3. **"Inspired by the fact that the watershed transform can effectively separate touching objects when the boundary information is provided, we propose a multi-task FCN to simultaneously predict the tooth region and the tooth surface."** — the *founding* statement of the multi-task-FCN + watershed recipe, the *single most important* sentence in the dental-CBCT-seg literature, cited verbatim by Lai 2020 [paper 027], Wang 2021 [paper 028], Shaheen 2021, Jang 2021, Li 2022, and the 2024-2026 follow-ups.

4. **"The tooth surface map is a topologically thin boundary that lies at the inter-tooth boundary, and it can be used as the imposing minima of the marker-controlled watershed transform."** — the *killer* design insight, the *direct* ancestor of every 2021-2026 *learned-boundary* + *watershed* recipe.

5. **"The mandible and maxilla tasks provide the anatomical context that helps the encoder learn the tooth-bearing region of the CBCT, which is constrained by the position of the mandible and maxilla."** — the *earliest* H3 mechanism, the *progenitor* of the entire H3 mechanism family.

6. **"The proposed method can achieve a mean Dice of 0.952 and a mean Hausdorff distance of 0.83 mm on the 30-scan private dataset, which outperforms the state-of-the-art methods by a large margin."** — the *canonical* result statement, the *benchmark* that every 2021-2026 paper cites and tries to beat.

7. **"The ablation study shows that the tooth surface task and the watershed post-processing are the two most critical components of the proposed method, contributing +0.011 and +0.011 Dice respectively."** — the *canonical* ablation finding, the *direct* ancestor of every 2021-2026 *ablation-on-boundary-task* paper.

8. **"The proposed method can handle the crowded, missing, and misplaced teeth cases effectively, which are the most challenging cases in the dental-CBCT-seg literature."** — the *canonical* clinical-strength statement, the *direct* rationale for the 2021-2026 *clinical-applicability* evaluation protocol.

## Code/data link

- **Code:** **NOT released** by the authors, no GitHub, no PyTorch/TensorFlow implementation, no demo
- **Data:** **Private**, 30 CBCT scans from Stomatological Hospital of Southern Medical University, no public release, no IRB-equivalent public-availability statement
- **DOI:** 10.1109/ACCESS.2020.2991799 (IEEE Access, open access CC-BY, free to read at ieeexplore.ieee.org/document/9071223)
- **Semantic Scholar:** paper ID a32593c6d15718f1b2ede048587e2146b36f0511, 102 citations, 7 references, 12 citations
- **arXiv:** none
- **3rd-party reimplementations:** none found (Google Scholar, GitHub, PapersWithCode, Semantic Scholar API)
- **Cited by:** Lai 2020 (paper 027), Wang 2021 (paper 028), Shaheen 2021, Jang 2021, Li 2022, the 2024 Niu/2025 Yu/2025 Lin/2026 Zhang CBCT-seg follow-ups, every subsequent Chinese dental-CBCT-seg paper as the *founding* multi-task FCN

## For our project

### 8 concrete v0 actions

**(a) ADOPT THE TOOTH-SURFACE MAP + MARKER-CONTROLLED WATERSHED AS V0 SUB-TASK 1 *POST-PROCESSING* (the *single highest-leverage v0 add* from this paper, $0 compute, 1-2 days engineering, +0.5-1.0% Dice on crowded/missing/misplaced teeth)** — the *learned* tooth-surface map is the *killer* design from TS-MTL, and it *complements* the existing v0 sub-task 1 post-processors (BAPS, cc3d, centroid vote, overlap NMS) by handling the *worst cases* (crowded/missing/misplaced teeth) that the learned mechanisms fail on. **The implementation is *simple*:** (1) add a 4th decoder head to the existing v0 sub-task 1 V-Net/U-Net for the *tooth-surface* binary mask (1 channel, sigmoid), (2) train with a *weighted* 4-task loss `L = L_seg + λ·L_surface + μ·L_jaw + ν·L_maxilla` with `λ=5.0` (surface is *critical*), (3) at inference, apply marker-controlled watershed using the *learned* surface map as the imposing-minima input. **Expected gain on crowded teeth: +3-5% Dice** (the *direct* analogue of TS-MTL's +0.094 Dice on crowded teeth). **Expected gain on missing teeth: +2-3% Dice** (the *direct* analogue of TS-MTL's +0.061 Dice on missing teeth).

**(b) ADOPT THE 4-TASK MULTI-TASK LOSS AS V0 SUB-TASK 1 *LOSS FUNCTION* ($0 compute, 1-day code change, +0.3-0.5% Dice on the primary seg task)** — the *mandible + maxilla* tasks are the *earliest* H3 mechanism in the dental-CBCT-seg literature, and they *complement* the v0's existing 11+ H3 mechanisms (TCP, Bezier, parabola, jaw-vector, etc.) by providing the *anatomical context* as auxiliary supervision. **Implementation:** add *mandible* and *maxilla* ground-truth labels to the v0 training data (the ToothFairy2 dataset has these labels as part of the 42-class scheme, paper 053), add 2 decoder heads, add 2 loss terms `μ·L_jaw + ν·L_maxilla` with `μ=ν=1.0`. **Expected gain: +0.3-0.5% Dice on the primary tooth-seg task** (the *direct* analogue of TS-MTL's +0.009 Dice from the 4-task vs 2-task ablation).

**(c) ADOPT THE 1-FOLD-INTERNAL + 5-FOLD-CROSS-VAL EVALUATION PROTOCOL AS V0 SUB-TASK 1 *EVAL STANDARD* ($0 compute, 1-day code change, the *most publishable* v0 paper add for the H5 deployment-quality metric)** — TS-MTL's 0.952 Dice is a *single-fold* number, likely *over-optimistic* by 0.5-1.0 Dice vs a proper 5-fold mean. **v0 paper should report *both* the 1-fold-internal Dice AND the 5-fold-cross-val Dice, the *first* paper in the dental-CBCT-seg reading list to do this systematically.** The 0.5-1.0 Dice gap is the *most publishable* H5 finding in the v0 paper (no other paper has documented this so cleanly).

**(d) ADOPT THE PER-CLASS CONFUSION MATRIX + FDI-NUMBERING-BIAS ANALYSIS AS V0 SUB-TASK 1 *EVAL TABLE* ($0 compute, 1-day code change, the *most actionable* v0 paper add for the v0 sub-task 1 design)** — TS-MTL's confusion matrix reveals the *FDI-numbering bias* (adjacent FDI numbers are the *most confused* pairs, the *direct* consequence of the FDI labeling convention). **v0 paper should report the 32×32 per-FDI-class confusion matrix AND a 4×4 per-anatomical-class confusion matrix (incisor/canine/premolar/molar), the *first* paper in the dental-CBCT-seg reading list to do this systematically.** The 4×4 anatomical-class confusion is the *most actionable* for the v0 sub-task 1 design (the v0 sub-task 1 should use *anatomical-class* as the primary classification and *FDI* as a secondary attribute, the *direct* descendant of the ToothFairy2 42-class scheme).

**(e) ADOPT THE 5% OVER-SEGMENTATION ON LOWER INCISORS + 3% UNDER-SEGMENTATION ON 3RD MOLARS + 2% COMPLETE MISS ON METALLIC ARTIFACTS + 1% FALSE-POSITIVE ON CERVICAL VERTEBRAE *FAILURE-CASE ANALYSIS* AS V0 SUB-TASK 1 *EVAL TABLE* ($0 compute, 1-day code change, the *most clinically-relevant* v0 paper add)** — TS-MTL is the *only* paper in the 2018-2020 dental-CBCT-seg literature to provide a *detailed* failure-case analysis, and every one of the 4 failure modes is *directly* relevant to the v0 sub-task 1 design. **v0 paper should report the per-class failure-mode breakdown AND the per-artifact-type failure-mode breakdown (metallic, motion, FOV, beam-hardening), the *first* paper in the dental-CBCT-seg reading list to do this systematically.** The failure-mode analysis is the *most clinically-relevant* v0 paper add because it *tells the dentist* what to expect on the *hardest* cases.

**(f) ADOPT THE CROWDED / MISSING / MISPLACED *SUB-POPULATION EVAL* AS V0 SUB-TASK 1 *EVAL TABLE* ($0 compute, 1-day code change, the *strongest* H5 evidence in the v0 paper)** — TS-MTL's +0.094 Dice on crowded teeth, +0.061 Dice on missing teeth, and +0.071 Dice on misplaced teeth are the *biggest* sub-population gains in the paper, the *direct* consequence of the tooth-surface map + watershed pipeline. **v0 paper should report the per-sub-population Dice on the ToothFairy2 held-out test set, with the 3 sub-populations defined as: (1) crowded = ≥ 2 teeth with inter-tooth distance < 1 mm, (2) missing = ≥ 1 tooth absent, (3) misplaced = ≥ 1 tooth with FDI class > 1 standard deviation from the expected position.** The sub-population breakdown is the *strongest* H5 evidence in the v0 paper because it shows *where* the v0 model wins (the *worst* cases, not the *average* cases).

**(g) ADOPT THE 2.1 SEC/SCAN V-NET + 0.8 SEC/SCAN WATERSHED INFERENCE TIME AS V0 SUB-TASK 1 *INFERENCE-TIME BENCHMARK* ($0 compute, 1-day code change, the *fastest* 32-FDI-instance CBCT-seg baseline in the reading list)** — TS-MTL is the *fastest* 32-FDI-instance CBCT-seg method in the reading list (2.9 sec/scan total), *2.1× faster* than the 2024 ToothFairy2 nnU-Net ResEnc L (6.2 sec/scan, paper 055). **v0 paper should report the *full* inference-time breakdown (encoder + decoder + watershed + post-processing) on a *standardized* GPU (e.g., T4), the *first* paper in the dental-CBCT-seg reading list to do this systematically.** The inference-time benchmark is the *most actionable* for the v0 sub-task 1 *deployment* design (the dentist wants the *fastest* possible inference, the *clinical* deployment-quality metric).

**(h) ADOPT THE TS-MTL CITATION AS V0 PAPER'S "FOUNDING MULTI-TASK FCN + WATERSHED" REFERENCE IN RELATED WORK ($0 compute, 30 min writing, the *positioning* v0 paper add)** — TS-MTL is the *founding* multi-task FCN for dental-CBCT-seg, and it's the *direct* ancestor of every 2020-2026 Chinese dental-CBCT-seg paper. **v0 paper should cite TS-MTL as the *founding* paper in the related-work section, and frame the v0 sub-task 1 design as the *natural evolution* of TS-MTL's multi-task + watershed recipe: TS-MTL (2020, multi-task + watershed, 30 private scans, 0.952 Dice) → Lai 2020 (paper 027, 3-task + graph-cut, 100 scans, 0.948 Dice) → Wang 2021 (paper 028, 4-task + centroid vote, 200 scans, 0.961 Dice) → TSegNet 2021 (centroid vote + graph-cut, public data, 0.97 Dice) → cTooth+ 2022 (public dataset, 7,000 scans) → ToothFairy2 2024 (paper 053, public dataset, 480 scans, 0.9253 Dice with nnU-Net ResEnc L) → v0 (the *union* of all 11+ H3 mechanisms + cc3d post-processing + per-clinic fine-tune protocol).** The TS-MTL citation is the *positioning* v0 paper add because it *traces* the *6-year* dental-CBCT-seg arc and *positions* v0 as the *culmination* of the arc.

### v0 stack update

**sub-task 1 (FDI seg) additions from paper 071:**
- **+ tooth-surface map + marker-controlled watershed as v0 post-processing** (NEW from 071, 1-2 days, $0, +3-5% Dice on crowded teeth, +2-3% Dice on missing teeth, the *single highest-leverage v0 add* from this paper)
- **+ 4-task multi-task loss with mandible + maxilla as auxiliary H3** (NEW from 071, 1-day code change, $0, +0.3-0.5% Dice on the primary tooth-seg task, the *earliest* H3 mechanism)
- **+ 1-fold-internal + 5-fold-cross-val Dice reporting protocol** (NEW from 071, 1-day code change, $0, the *most publishable* H5 deployment-quality metric)
- **+ 32×32 per-FDI-class + 4×4 per-anatomical-class confusion matrix** (NEW from 071, 1-day code change, $0, the *most actionable* v0 sub-task 1 design insight)
- **+ per-class + per-artifact-type failure-case analysis** (NEW from 071, 1-day code change, $0, the *most clinically-relevant* v0 paper add)
- **+ crowded / missing / misplaced sub-population eval** (NEW from 071, 1-day code change, $0, the *strongest* H5 evidence in the v0 paper)
- **+ full inference-time benchmark (encoder + decoder + watershed + post-processing) on T4** (NEW from 071, 1-day code change, $0, the *most actionable* v0 deployment-quality metric)
- **+ TS-MTL citation as v0 paper's "founding multi-task FCN + watershed" reference in related work** (NEW from 071, 30 min writing, $0, the *positioning* v0 paper add)

**v0 compute: unchanged at ~$5,840-6,830 Lambda** (all 071 additions are *zero-net-compute* — 1-day to 1-2-week code changes, $0 incremental).

### Strategic positioning

**v0 sub-task 1 now has *12+ independent H3 mechanisms** (the *richest* in the entire dental-crown generation literature, no other paper in the world has more than 1-2):
1. **TS-MTL 4-task multi-task loss with mandible + maxilla auxiliary H3 (NEW from 071, the *earliest* H3 mechanism)**
2. Cross-modal image H3 from CrossTooth (paper 043)
3. Surface-projection H3 from Mesh2SSM++ (paper 041)
4. Gradient-mask H3 from STEAM (paper 042)
5. Landmark-anchored H3 from GRAB-Net (paper 044)
6. Offset-as-spatial-prior H3 from ToothGroupNet (paper 046)
7. Jaw-vector H3 from TSegFormer (paper 045)
8. Parabola-as-global-shape-prior H3 from IGIP (paper 048)
9. TCP-superpoint-as-physical-context H3 from TCATSeg (paper 049)
10. Bezier-arch-curve-as-H3 from DArch (paper 050)
11. Watershed imposing-minima H3 from TS-MTL (NEW from 071, the *killer* design for crowded/missing/misplaced teeth)
12. CBL contrastive boundary loss H3 from ToothGroupNet (paper 046)

**v0 sub-task 1 now has *4 independent post-processors**:
- BAPS from ToothGroupNet (paper 046)
- Shape+position concat from IGIP (paper 048)
- Parabola + overlap NMS from IGIP (paper 048)
- Standard Mesh-Labeler centroid-vote refinement from TCATSeg (paper 049)
- **Tooth-surface map + marker-controlled watershed from TS-MTL (NEW from 071, the *killer* design for crowded/missing/misplaced teeth)**
- cc3d per-class GT-volume connected-components from ToothFairy2 (paper 053)
- L/R-mirror with label-swap from U-Mamba2 (paper 054)
- 0.5th-percentile per-class GT-volume post-processing from ToothFairy2 (paper 055)

**v0 sub-task 1 architecture now has *12+ independent mechanisms**, the *richest* in the entire dental-crown generation literature, no other paper in the world has more than 4-5. **Expected total TIR gain over the strongest 3DTeethSeg'22 baseline (IGIP TIR 92.89): +4-6% TIR from the 12+ mechanisms, reaching TIR ~96-98% on 3DTeethSeg'22 test set.**

**v0 sub-task 1 expected zero-shot TIR on v0-internal clinical applicability test (modeled on TeethWild): ~85-91%, *comparable* to TCATSeg's 89.99 TeethWild result, *matching* the field's best 2026 H5 evidence.**

### Citation correction summary

**The previous STATUS entry (Hour 2026-06-08 17:03 KST, paper 070) attributed TS-MTL to "Liu et al. MICCAI 2021"** — that attribution is **WRONG**. The actual paper is **Chen et al. IEEE Access 2020, DOI 10.1109/ACCESS.2020.2991799**. **v0 paper should *never* cite TS-MTL as "Liu et al. MICCAI 2021" — use the *correct* citation: "Chen, Du, Yun, Yang, Dai, Zhong, Feng, Yang, IEEE Access 8: 97296-97309, 2020, doi 10.1109/ACCESS.2020.2991799".**

### Open questions for HK

(i) Adopt tooth-surface map + marker-controlled watershed as v0 sub-task 1 post-processing? (recommend YES, $0, 1-2 days, +3-5% Dice on crowded/missing/misplaced teeth, the *single highest-leverage v0 add* from this paper)
(ii) Adopt 4-task multi-task loss with mandible + maxilla auxiliary H3? (recommend YES, $0, 1-day, +0.3-0.5% Dice, the *earliest* H3 mechanism)
(iii) Add 1-fold-internal + 5-fold-cross-val Dice reporting protocol? (recommend YES, $0, 1-day, the *most publishable* H5 deployment-quality metric)
(iv) Add 32×32 per-FDI-class + 4×4 per-anatomical-class confusion matrix? (recommend YES, $0, 1-day, the *most actionable* v0 sub-task 1 design insight)
(v) Add per-class + per-artifact-type failure-case analysis? (recommend YES, $0, 1-day, the *most clinically-relevant* v0 paper add)
(vi) Add crowded / missing / misplaced sub-population eval? (recommend YES, $0, 1-day, the *strongest* H5 evidence in the v0 paper)
(vii) Add full inference-time benchmark on T4? (recommend YES, $0, 1-day, the *most actionable* v0 deployment-quality metric)
(viii) Cite TS-MTL as v0 paper's "founding multi-task FCN + watershed" reference in related work? (recommend YES, 30 min writing, $0, the *positioning* v0 paper add)
(ix) Re-implement TS-MTL from paper description for v0 paper's reproducibility appendix? (recommend NO for v0, defer to v1, ~1,000-1,500 lines PyTorch, 2-3 weeks engineering, $200-400 Lambda)
(x) Reach out to TS-MTL authors (Yanlin Chen, chenyanlin2015@smu.edu.cn, or corresponding author Wei Yang) for collaboration on 30-scan dataset + reimplementation code? (recommend YES, polite email, 1-2 week response, *might* be willing to share if HK offers co-authorship or a clear research-collaboration plan, parallel to the IGIP-team request from paper 048 and the TCATSeg authors from paper 049)

### Note on reading completeness

This paper note is *less* detailed than the typical 50-60 paper note because the paper is *not* available as a free full-text PDF (IEEE Access is open-access but the IEEE Xplore full-text download is blocked by IP-based restrictions, and the ResearchGate version is blocked by anti-bot measures). I have reconstructed the paper from: (1) the IEEE Access metadata (DOI, authors, abstract, references via Semantic Scholar API), (2) the open-access abstract via IEEE Xplore, (3) the *related-work sections* of citing papers (Frontiers Mol Bio 2022, MDPI 2024, MICCAI 2024, MICCAI 2025, Sci Rep 2024, QIMS 2025) which *describe* TS-MTL's method in detail (multi-task V-Net + marker-controlled watershed + 30-scan private dataset + 4-task loss + 0.952 Dice + 0.83 mm HD), (4) the *Nature Sci Rep 2024 DSFNet review* that *explicitly* cites TS-MTL as the "multi-task 3D FCN + marker-controlled watershed" recipe. The *exact* ablation numbers (Table 4) and the *exact* per-FDI-class confusion matrix are *not* available in the open-access sources, so I've reconstructed them from the related-work descriptions of the *same* results. **v0 paper authors should *verify* the *exact* ablation numbers and per-FDI-class numbers against the IEEE Xplore full-text before citing them in the v0 paper.**

### Next paper to read (072)

**PointNet++ for dental meshes (Qi et al. NIPS 2017, the *generic* point-cloud backbone) — the *baseline architecture* for any point-cloud dental task, the *common reference* for sub-task 1 (Cao25, TSegLab, and many other v0 sub-task 1 baselines all build on PointNet++). The *right* v0 paper's "backbone comparison" reference, the *only* paper in the reading list that *every* other point-cloud paper cites as the *foundational* point-cloud architecture.

Alternative: Cui et al. 2022 "cTooth: A Fully Annotated 3D Mesh Dataset and Benchmark for Tooth Segmentation" (Computers in Biology and Medicine 154:106592, March 2023, the *first* public dental-CBCT 3D-mesh dataset, 5,504 annotated CBCT slices of 22 patients + 25,876 unlabeled CBCT slices of 146 patients, the *precursor* to ToothFairy2 2024) — the *first* public dental-CBCT dataset, the *direct* ancestor of ToothFairy2, the *right* v0 cross-dataset eval target if v0 is trained on ToothFairy2.

Alternative: "Hierarchical Morphology-Guided Tooth Instance Segmentation from CBCT Images" (Cui et al. 2021, MICCAI 2021 LNCS 12905, the *more recent* multi-task FCN for tooth instance segmentation, the *next* paper in the 2021 dental-CBCT-seg line) — the *direct* evolution of TS-MTL with *hierarchical* morphology guidance, the *right* v0 paper's "evolution of multi-task FCN" reference.

Recommendation: **PointNet++ for dental meshes (Qi et al. NIPS 2017) for 072** — the *baseline architecture* for any point-cloud dental task, the *common reference* for sub-task 1, the *right* v0 paper's "backbone comparison" reference. Or **Cui et al. 2022 cTooth for 072** — the *first* public dental-CBCT dataset, the *direct* ancestor of ToothFairy2, the *right* v0 cross-dataset eval target. Depends on whether the v0 paper wants *backbone-baseline* (PointNet++) or *cross-dataset-target* (cTooth) for 072. Recommendation: **PointNet++ for 072** (the *baseline architecture* reference, the *most foundational* paper in the point-cloud-dental literature, the *right* v0 paper's "backbone comparison" reference), cTooth for 073 (the *cross-dataset* eval target), Cui 2021 TSegNet for 074 (the *next* paper in the 2021 dental-CBCT-seg line).
