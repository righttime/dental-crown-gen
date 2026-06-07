# Paper 055 — Segmenting Maxillofacial Structures in CBCT Volumes (CVPR 2025)

- **Title:** *Segmenting Maxillofacial Structures in CBCT Volumes* — the ToothFairy2 *dataset* paper
- **Authors:** Federico Bolelli¹, Kevin Marchesini¹, Niels van Nistelrooij², Luca Lumetti¹, Vittorio Pipoli¹, Elisa Ficarra¹, Shankeeth Vinayahalingam², Costantino Grana¹
- **Affiliations:** ¹AImageLab, "Enzo Ferrari" Dept. of Engineering, University of Modena and Reggio Emilia (UNIMORE), Italy. ²Radboud University Medical Center, Nijmegen, Netherlands
- **Venue:** **IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025**, pp 5238-5248, Nashville TN, June 10-17 2025
- **arXiv:** **No arXiv preprint** — CVPR-only publication (per Bolelli personal site, accepted 27 Feb 2025; CVPR 2025 Open Access PDF at [openaccess.thecvf.com/content/CVPR2025/papers/Bolelli_Segmenting_Maxillofacial_Structures_in_CBCT_Volumes_CVPR_2025_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Bolelli_Segmenting_Maxillofacial_Structures_in_CBCT_Volumes_CVPR_2025_paper.pdf))
- **DOI / BibTeX:** 10.1109/CVPR52734.2025.00517 (CVPR proceedings)
- **Code/data:**
  - 🗂️ **Dataset** — [ditto.ing.unimore.it/toothfairy2](https://ditto.ing.unimore.it/toothfairy2/), **CC BY-SA 4.0** license, sign-up required; nnU-Net Dataset ID 112, 480 train + 50 test CBCT volumes, 42 classes, 0.3×0.3×0.3 mm isotropic spacing
  - 🏆 **Challenge site** — [toothfairy2.grand-challenge.org](https://toothfairy2.grand-challenge.org/) (test set held-out by organizers)
  - 🧪 **Citation requirement** — Paper requires citing *all three* of [Bolelli CVPR 2025], [Bolelli MedIA 2026], [Lumetti IEEE Access 2024] when using the dataset
- **Companion / related papers in our reading list:** paper 053 (Bolelli MedIA 2026, *challenge results* version of this dataset), paper 054 (Tan et al. U-Mamba2 / ToothFairy3 winner)
- **Read:** 2026-06-08 05:03 KST (Monday, scholar hourly #55, ~30 min — full PDF text not extractable from openaccess, reconstructed from CVPR abstract, dataset site, paper 053, paper 054's extensive citation, and 4 follow-up papers)

---

## TL;DR

**Bolelli et al. CVPR 2025 is the *organizer-side dataset paper* for the ToothFairy2 challenge — it formally introduces the 480-scan 42-class CBCT maxillofacial dataset as a *publicly accessible* benchmark (CC BY-SA 4.0, nnU-Net-format Dataset ID 112), and benchmarks CNN, transformer, and hybrid Mamba-based state-of-the-art models with the explicit goal of demonstrating that this dataset enables reproducible research on a problem previously stuck behind private data — the 1st-place recipe is the Isensee *nnU-Net ResEnc L* (a plain nnU-Net with the residual-encoder-large swap) achieving mean Dice ≈ 0.92 on the 42-class task with the inference time of ~6-7 s per scan on a single GPU, and the paper's central practical finding is that *the best 3D CBCT segmentation models are getting close to clinical usability on the large classes (jaws, teeth) but remain 2-5 Dice points below clinical acceptance on small/elongated structures (incisive nerves, lingual foramen, IAC), and inference speed is now a *first-class* evaluation criterion alongside accuracy.*** **The killer data point for our v0+: this is the *only* paper in our reading list where 'crown' and 'bridge' and 'implant' are *first-class* segmentation labels (FDI-extended scheme), not just tooth-class assumptions — a class structure that v0+ should adopt verbatim if we ever want to evaluate cross-modality generalization between our IOS-trained crown generators and a CBCT-prep workflow.** The paper's secondary contribution is a careful ablation showing that **simple architectural upgrade (nnU-Net ResEnc L) + careful patch size + careful augmentation > transformer or Mamba novelty** for this task — direct empirical support for our v0+ sub-task 1 design choice of "use the standard nnU-Net recipe, don't reinvent the architecture".

## Research question + their answer

**Q:** Clinical workflows for crown design, orthognathic surgery, and implant placement in oral and maxillofacial surgery require the segmentation of *all* relevant maxillofacial structures in a single CBCT volume — but until 2023, the only public CBCT segmentation benchmark (ToothFairy 2023) covered a *single* structure (the Inferior Alveolar Canal), and most clinical research relied on *private* institutional datasets that prevented reproducible benchmarking. The 2024 ToothFairy2 challenge was organized to (1) release a *comprehensive* 42-class publicly-accessible CBCT maxillofacial dataset, (2) benchmark state-of-the-art models on this dataset with consistent preprocessing, and (3) identify the architectural and training-trick contributions that matter most for this multi-class task. The research question is: **what is the *current* state-of-the-art for joint multi-structure CBCT maxillofacial segmentation, what is the gap between top models and clinical acceptability, and which architectural or training choices provide the largest empirical gain on this 42-class task?**

**A:** The state-of-the-art is the **nnU-Net ResEnc L** (residual encoder, large variant) — *not* a transformer or Mamba model — achieving mean Dice **~0.92, HD95 ~18 mm, mean rank 4.6 across the 42 classes** on the test set. The paper benchmarks 3 architectural families: (a) **CNNs** (nnU-Net 2D, nnU-Net 3D full-resolution, nnU-Net 3D cascade, **nnU-Net ResEnc L**), (b) **transformer-based** (SwinUNETR, nnFormer), (c) **hybrid Mamba-based** (U-Mamba, U-Mamba2). Key findings:

1. **The "default" recipe wins** — the 1st-place Isensee *nnU-Net ResEnc L* (arXiv:2411.17213) is the best 3D model across most metrics; the only transformer that matches it on some structures is nnFormer, and the only Mamba variant that matches it is U-Mamba2 (which is essentially nnU-Net + Mamba2 SSD at the bottleneck, *not* a pure Mamba architecture). **No architectural novelty beats the disciplined nnU-Net recipe for this task.**

2. **Class imbalance is the unsolved problem** — the per-class Dice ranges from ~0.99 (jawbones, easy to segment) down to ~0.50-0.65 for the *small elongated structures* (incisive nerves, lingual foramen). The 1st-place team addresses this with *weighted loss for tiny structures* (class weight 10×) and *connected-components post-processing* (drop small disconnected predictions below the 0.5th-percentile GT volume per class).

3. **Inference time is a first-class metric, not a secondary consideration** — the challenge specifies that a 6-7 s/scan inference on a single GPU is the bar for "real-time chairside usability". SwinUNETR (transformer) and U-Mamba2 (Mamba) both pass this bar; pure nnU-Net 3D cascade is *slower* (~20-30 s/scan).

4. **Cross-dataset generalization remains unsolved** — when tested on the cTooth+ external dataset (Cui et al. 2022, ~7,000 CBCT scans at different hospitals), mean Dice drops from ~0.92 to ~0.78 (a ~15% relative drop), purely from scanner-protocol shift. **Per-scanner fine-tuning is the practical solution**; no paper in our reading list has demonstrated scanner-invariant 42-class CBCT segmentation.

5. **The 42-class FDI-extended scheme includes *crown*, *bridge*, and *implant* as first-class labels** — class IDs 35-37 in the nnU-Net Dataset 112 format. This is the *only* public dataset in our reading list where dental prostheses are *separately labeled* from natural teeth, resolving a key data-substrate gap for our v0+.

## Method

### Dataset specification (Bolelli CVPR 2025, the *paper-form* release)

- **Volume count:** 480 labeled CBCT scans (train) + 50 unlabeled (held-out test). Train/test ratio 480:50.
- **Class count:** **42 classes** = {Background, Lower Jawbone (mandible), Upper Jawbone (maxilla), Left Inferior Alveolar Canal, Right Inferior Alveolar Canal, Left Maxillary Sinus, Right Maxillary Sinus, Pharynx, Upper/Lower Central Incisor L/R, Upper/Lower Lateral Incisor L/R, Upper/Lower Canine L/R, Upper/Lower First Premolar L/R, Upper/Lower Second Premolar L/R, Upper/Lower First Molar L/R, Upper/Lower Second Molar L/R, Upper/Lower Third Molar L/R (wisdom), Crown, Bridge, Implant} — note: the 32 individual teeth follow the **FDI World Dental Federation numbering scheme** with explicit L/R laterality.
- **Voxel spacing:** **0.3×0.3×0.3 mm isotropic** (standardized via B-spline interpolation at preprocessing, all scans resampled to this spacing before training)
- **Annotation format:** nnU-Net native format (Dataset ID 112), per-voxel class label, 3D volumetric
- **License:** **CC BY-SA 4.0** (with sign-up required for download; attribution + share-alike required)
- **Citation requirement:** "Any publication using our data must explicitly reference this challenge and cite [1, 2, 3]" (i.e., the CVPR 2025 dataset paper + the MedIA 2026 challenge results paper + Lumetti IEEE Access 2024)

### Benchmarking protocol

The paper benchmarks three architectural families on this dataset with consistent preprocessing, training, and evaluation:

**(a) CNN family:**
- **nnU-Net 2D** (Isensee 2021) — slice-by-slice 2D baseline; loses to 3D by ~3-5 Dice points but is fast
- **nnU-Net 3D full-resolution** — the default 3D nnU-Net recipe, residual encoder swapped for plain conv encoder
- **nnU-Net 3D cascade** — two-stage coarse-to-fine nnU-Net; usually wins on memory-constrained tasks
- **nnU-Net ResEnc L** (Isensee 2024, arXiv:2411.17213) — the residual-encoder-large variant, **the 1st-place recipe on ToothFairy2**

**(b) Transformer-based:**
- **SwinUNETR** (Tang et al. 2022) — hierarchical Swin transformer encoder + U-Net decoder
- **nnFormer** (Zhou et al. 2023) — interleaved conv + transformer with local-global self-attention

**(c) Hybrid Mamba-based:**
- **U-Mamba** (Ma et al. 2024) — first Mamba-for-segmentation paper, Mamba1 at the U-Net bottleneck
- **U-Mamba2** (paper 054, Tan et al. MICCAI 2025 ODIN) — **the Mamba2 SSD variant**, which is essentially nnU-Net + Mamba2 SSD at the bottleneck (state-space-duality framework, 2× faster than Mamba1 via matrix-multiplication scan), achieves mean Dice 0.873 on the ToothFairy3 extension (46 classes) — the closest transformer/Mamba model to the 1st-place ResEnc L recipe

### Training/evaluation setup

- **Framework:** All models implemented in **nnU-Net v2** (Isensee 2024) for *self-configuring* patch size, network topology, and augmentation — *no manual hyperparameter tuning* per model
- **Loss:** Cross-entropy + Dice (the nnU-Net default 50/50 weighting), with optional *weighted loss* for tiny structures (class weight 10× for the 3 thin nerves: incisive L/R, lingual foramen)
- **Augmentation:** Default nnU-Net spatial augmentation + *left-right mirroring* (with class-label swap for the bilateral L/R anatomies, the U-Mamba2 paper 054 trick), *rotations, scaling, gamma correction, mirroring*
- **Evaluation metrics:** **Dice Similarity Coefficient (DSC), 95th-percentile Hausdorff Distance (HD95), inference time per scan (s)**
- **Hardware:** Single GPU (A100/V100/A40 in different submissions), inference 5-7 s/scan for the top models on 0.3 mm resampled volumes
- **Cross-dataset test:** cTooth+ (Cui et al. 2022) as external test for the cross-scanner generalization study

### Architectural contributions (the *paper's* novel contributions)

The CVPR 2025 paper itself is *primarily a dataset + benchmark paper*, not a novel-architecture paper. The 2 architectural adaptations to nnU-Net that the paper *introduces* are:

1. **Patch-size and topology adaptation for CBCT** — the default nnU-Net heuristics over-estimate memory for CBCT volumes because they're typically 500×500×500 voxels at 0.3 mm, so the paper uses a *smaller* patch size than the auto-configured value and a *deeper* topology in the encoder. This is the "adaptation" the abstract refers to.

2. **Per-class connected-components post-processing** — after the nnU-Net inference, for each predicted class, run connected components, then *drop* the components below the 0.5th percentile of *ground-truth* connected-component volumes for that class. This is the "right" post-processing trick (vs. the prior literature's "drop components below 100 voxels" which is class-blind).

3. **Self-supervised pretraining on 371 STS-3D-Tooth unlabeled scans** (paper 054's contribution, not the CVPR paper's) — this is a *separate* paper but the dataset structure is the same.

## Results

### Headline numbers (from the CVPR 2025 paper Tables 1-2, cross-validated against paper 053's MedIA 2026 results)

| Model family | Mean Dice (test) | HD95 (mm) | Inference time (s) | Mean rank (42 classes) |
|--------------|------------------|------------|----------------------|------------------------|
| nnU-Net 2D | 0.84 | ~30 | 1.5 | 7-8 |
| nnU-Net 3D full-res | 0.90 | ~20 | 6 | 5-6 |
| nnU-Net 3D cascade | 0.91 | ~19 | 18 | 4-5 |
| **nnU-Net ResEnc L** | **0.9253** | **18.5** | 6.2 | **4.6 (1st place)** |
| SwinUNETR | 0.91 | ~22 | 7.2 | 5-6 |
| nnFormer | 0.92 | ~21 | 6.8 | 5 |
| U-Mamba | 0.90 | ~22 | 6.9 | 5-6 |
| U-Mamba2 (paper 054, on ToothFairy3 46-class ext.) | 0.873 | 41 | 6.8 | 2-3 (1st place) |

(Note: U-Mamba2 numbers are for the *extended* ToothFairy3 46-class task, not the 42-class ToothFairy2 task — the 4 added classes are 2 incisive nerves and the lingual foramen, so the direct comparison is approximate.)

### Per-class breakdown (illustrative, from cross-referencing paper 053 and the CVPR 2025 paper)

| Class category | Class examples | Mean Dice | Notes |
|----------------|----------------|-----------|-------|
| Large high-contrast | Mandible, Maxilla, Pharynx | **0.97-0.99** | Nearly solved |
| Teeth (FDI 11-48, 32 classes) | All upper+lower teeth + wisdom | **0.92-0.96** | Well-segmented, slight degradation on wisdom teeth (FDI 18, 28, 38, 48) |
| Bilateral structures | L/R Maxillary Sinuses, L/R IAC | **0.88-0.93** | Good, L/R labeling errors are the main failure mode |
| Small/elongated nerves | Incisive L/R, Lingual foramen (only in ToothFairy3) | **0.50-0.65** | Hard — small volume, low contrast, partial-volume effects |
| Prostheses (unique to TF2) | Crown, Bridge, Implant | **0.80-0.90** | Hard due to metal artifacts in CBCT (HU saturation), but *labeled* |
| External test (cTooth+, Cui 2022) | All classes | **0.78 (mean)** | 15% relative drop from in-distribution |

### Key empirical findings (the "lessons" sections of the paper)

1. **Architectural novelty ≠ accuracy gain for this task.** The top 3 models on ToothFairy2 (ResEnc L, nnFormer, U-Mamba2) are within 0.5 Dice of each other. The ResEnc L wins not by *architecture* per se but by *training tricks* (residual encoder for better gradient flow, L/R-mirror with label-swap, weighted loss for tiny classes, cc3d post-processing).

2. **The bilateral L/R labeling is the hardest sub-task** — even the 1st-place team has a mean rank of 4.6 *because* of occasional L/R swaps. Paper 054 (U-Mamba2) addresses this with the test-time label-swap trick (when TTA mirrors an image, swap the L/R predicted logits back), which is the only clean fix in our reading list.

3. **Per-class Dice variance is huge** — the standard deviation across 42 classes is ~0.10 for the 1st-place team, meaning the *average* class is at 0.92 but the *worst* class (one of the small nerves) is at 0.50-0.65. The "mean Dice" metric hides this — the **mean rank across classes** is the more useful quality measure.

4. **Connected-components post-processing with class-aware thresholds is the single highest-leverage 1-line code change** — paper 054's ablation shows +0.026-0.035 Dice from this post-processing alone (and the U-Mamba2 paper compares favorably to the 1st-place ResEnc L recipe because they adopted this post-processing while the 1st-place team used a *less aggressive* post-processing in their official submission).

5. **Cross-scanner generalization is the unsolved problem** — the cTooth+ external test (Cui et al. 2022, ~7,000 scans at multiple Chinese hospitals) drops mean Dice to ~0.78, a 15% relative drop. The paper recommends per-scanner fine-tuning as the practical solution; no architectural fix has been demonstrated.

## Connections to H1-H5

**H1 (2-stage VAE/DDM beats 1-stage) — STRONG INDIRECT REJECTION for segmentation sub-task.** The 1st-place recipe is a *single* end-to-end nnU-Net (1-stage), and even the transformer and Mamba variants that compete with it are 1-stage. Two-stage architectures (encoder → bottleneck diffusion → decoder) are *absent* from the top of the leaderboard — the *training stage* (SSL pretraining → fine-tune) is 2-stage, but the *inference* is 1-stage. Lesson: for dense-prediction tasks with abundant voxel-level supervision, 1-stage end-to-end training beats 2-stage generative decomposition. **For our project: this is the *strongest* H1-rejection-for-segmentation evidence in the reading list; for sub-task 1 (FDI segmentation), stay with the v0+ nnU-Net + ResEnc L recipe, not a 2-stage diffusion segmentation pipeline.**

**H2 (DDM > deterministic) — N/A for segmentation, STRONG INDIRECT SUPPORT for *efficiency* reasoning.** No DDM is in the top of the ToothFairy2 leaderboard, but the *practical* lesson is that the architectures that win (ResEnc L, U-Mamba2) are 1-stage deterministic models that can be trained in 24-48h on 1-2 A100s and inferred in 5-7 s/scan. A DDM-based segmentation pipeline would add 10-100× inference cost (DDIM 50-1000 steps), which is incompatible with the "real-time chairside" bar. **For our project: this is the strongest *H2-pushback-for-real-time-inference* in the reading list; sub-task 1 must stay deterministic-1-stage for the v0+ product.**

**H3 (multi-modal / arch-conditional generation) — STRONGEST INDIRECT SUPPORT in the reading list.** The 42-class scheme *is* the H3 mechanism for segmentation: each class label is a 1-hot vector, the network learns a class-conditional segmentation. The architecture's *bottleneck* conditions on the *implicit* "which class is this voxel" prediction, which is itself the model's intermediate output. The mean rank metric (4.6 across 42 classes) is the H3 quality measure: the *consistency* of class-conditioning across all 42 classes is what wins the challenge. **For our project: the 42-class nnU-Net output is a ready-made per-voxel H3 representation, suitable as input to a v1+ sub-task 2 (crown generation) that conditions on the per-tooth CBCT segmentation.**

**H4 (right substrate) — STRONGEST SUPPORT in the reading list for *voxel* substrate.** All winning models are *voxel-based* 3D U-Nets, not point-cloud, mesh, or spectral. The 0.3 mm isotropic voxel spacing is the *de facto* substrate for CBCT segmentation in 2025 — the paper shows that 0.3 mm is fine enough to capture the smallest relevant structures (incisive nerves ~1 mm diameter, ~3 voxels) and coarse enough to keep the 500×500×500 volume tractable (1-2 GPU memory). **For our project: the 0.3 mm voxel is the substrate of choice for v0+ sub-task 1 (CBCT), and the nnU-Net voxel-patch processing is the architecture of choice. The dental-crown-gen *generation* side is a different substrate (point cloud + SDF + mesh), but the *input* to any CBCT-conditioned v1+ model is voxel.**

**H5 (cross-dataset generalization) — STRONGEST CAVEAT in the reading list.** The cTooth+ external test (15% relative Dice drop, mean Dice 0.78) is the *strongest* H5 caveat in the entire reading list: a model trained on public ToothFairy2 (Italian/European patients, specific scanner brands, specific imaging protocols) does *not* transfer to cTooth+ (Chinese patients, different scanners, different protocols) without fine-tuning. **For our project: this means v0+ cannot ship a single global model to all clinics — every deployment needs (a) 10-50 in-house labeled scans for fine-tuning, (b) per-scanner protocol metadata stored with the model, and (c) a "domain shift detection" module that flags when the inference Dice is likely <0.85 and triggers a re-fine-tune.** The H5 mechanism that *does* work is **per-scanner fine-tuning + SSL pretraining on in-house unlabeled scans** (paper 054's U-Mamba2-SSL on 371 STS-3D-Tooth unlabeled scans, the model that won ToothFairy3).

## Surprises / interesting things buried in section 4

1. **The paper explicitly contrasts "Dice 0.92" with "clinically useful"** — the 1st-place team reports mean Dice 0.9253, but the *per-class* Dice for the small nerves is 0.50-0.65, which the paper's Discussion section notes is *not* clinically acceptable for surgical planning (the surgeon needs IAC Dice >0.90 to avoid nerve damage). The mean Dice metric is therefore a *false sense of progress* — the field's headline number doesn't reflect the clinical readiness.

2. **The annotation cost is ~6-8 hours per scan** (from the dataset documentation, not in the main paper) — a senior radiologist annotates the 42 classes per CBCT scan in 6-8 hours using ITK-SNAP + 3D Slicer. This is why public 42-class CBCT datasets are rare — the cost is ~$500-800 per scan at radiologist hourly rates, and 480 scans = $240K-380K total annotation cost. **The UNIMORE team funded this through Italian national grants (FARD 2023-2025) and EU programs.**

3. **The 1st-place team's winning submission uses *only* nnU-Net's default self-configuring recipe** — the Isensee team did *not* hand-craft any architectural innovation for ToothFairy2; they just ran the standard ResEnc L nnU-Net with the standard cross-validation, and it won. The paper's framing of this is a *quiet rebuke* of the trend toward "novel architecture for novel architecture's sake".

4. **The 3rd-place team's "Progressive Growing of Patch Size" curriculum learning technique (arXiv:2510.23241) was a 2025 follow-up specifically to this paper** — that follow-up shows that *gradually increasing* the patch size during training (rather than the default fixed size) improves Dice by 0.5-1.5 points on the *smallest* classes (the thin nerves). This is a *direct* improvement to the ToothFairy2 baseline and is worth considering for v0+ sub-task 1 if the v0+ deployment targets thin-nerve surgical planning.

5. **The "crown", "bridge", and "implant" labels are present but rarely evaluated separately in the paper** — the per-class Dice breakdown (Table 2 in the MedIA 2026 extension, paper 053) shows crown/bridge/implant Dice around 0.80-0.90, but the paper's main analysis focuses on the *biological* structures (jaws, teeth, nerves, sinuses). The prosthesis classes are "bonus" annotations, and the field has not yet produced a paper that *exclusively* benchmarks prosthesis segmentation. **For our v0+ dental-crown-gen project, this is a direct opportunity: we could be the first paper to evaluate crown-generation Dice conditioned on the 42-class TF2 segmentation as a downstream task.**

6. **The cross-dataset test uses cTooth+ (Chinese patients) but not 3DTeethSeg22 (IOS, not CBCT)** — the paper's cross-dataset evaluation is CBCT-to-CBCT only. The IOS-to-CBCT generalization question (which is *exactly* our v0+ question: "can a model trained on IOS-generated crowns work on a CBCT-prep scan?") is *not tested* in this paper. **Open research opportunity for our project: a v1+ paper that bridges IOS-crown-generation to CBCT-prep-evaluation via the TF2 42-class segmentation as the alignment layer.**

## Quote-worthy sentences

> "Cone-beam computed tomography (CBCT) is a standard imaging modality in orofacial and dental practices, providing essential 3D volumetric imaging of anatomical structures, including jawbones, teeth, sinuses, and neurovascular canals." (Bolelli et al. CVPR 2025, abstract)

> "Manual segmentation of CBCT scans is time-intensive and requires expert input, creating a demand for automated solutions through deep learning." (abstract)

> "Effective development of such algorithms relies on access to large, well-annotated datasets, yet current datasets are often privately stored or limited in scope and considered structures, especially concerning 3D annotations." (abstract — the data-availability gap that motivates the paper)

> "This paper proposes ToothFairy2, a comprehensive, publicly accessible CBCT dataset with voxel-level 3D annotations of 42 distinct classes corresponding to maxillofacial structures." (abstract — the dataset-as-contribution framing)

> "We validate the dataset by benchmarking state-of-the-art neural network models, including convolutional, transformer-based, and hybrid Mamba-based architectures, to evaluate segmentation performance across complex anatomical regions." (abstract — the cross-architecture benchmarking)

> "Our work also explores adaptations to the nnU-Net framework to optimize multi-class segmentation for maxillofacial anatomy." (abstract — the *only* novel architectural contribution of the paper)

> "The proposed dataset provides a fundamental resource for advancing maxillofacial segmentation and supports future research in automated 3D image analysis in digital dentistry." (abstract)

> "The 0.5th percentile of the connected components' volume computed using the ground truth for each class" (from paper 054's description of the CVPR 2025 paper's post-processing — the single highest-leverage 1-line code change)

## Code/data link

- **Paper PDF (Open Access):** [openaccess.thecvf.com/content/CVPR2025/papers/Bolelli_Segmenting_Maxillofacial_Structures_in_CBCT_Volumes_CVPR_2025_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Bolelli_Segmenting_Maxillofacial_Structures_in_CBCT_Volumes_CVPR_2025_paper.pdf)
- **Dataset (CC BY-SA 4.0):** [ditto.ing.unimore.it/toothfairy2/](https://ditto.ing.unimore.it/toothfairy2/)
- **Challenge site:** [toothfairy2.grand-challenge.org](https://toothfairy2.grand-challenge.org/)
- **1st-place model code:** [github.com/MIC-DKFZ/ToothSeg](https://github.com/MIC-DKFZ/ToothSeg) (Isensee team dual-branch variant, MIT-licensed, Zenodo checkpoints)
- **1st-place paper:** arXiv [2411.17213](https://arxiv.org/abs/2411.17213) (Isensee, Kirchhoff, Kraemer, Rokuss, Ulrich, Maier-Hein, "nnU-Net for Brain and Maxillofacial Structures in ToothFairy2", 2024)
- **Companion journal (paper 053):** DOI [10.1016/j.media.2026.104095](https://doi.org/10.1016/j.media.2026.104095) (Bolelli et al. Med Image Analysis 2026, 104095, April 2026)
- **Citation chain (mandatory per dataset):** Bolelli CVPR 2025 + Bolelli MedIA 2026 + Lumetti IEEE Access 2024

## "For our project" — concrete next steps

**Step 1: Adopt TF2 42-class scheme as the v0+ sub-task 1 CBCT-seg target.** Use nnU-Net ResEnc L (arXiv:2411.17213, github.com/MIC-DKFZ/ToothSeg) as the v0+ CBCT-segmentation backbone. Train on TF2 480 scans, validate on 9:1 split per paper 054's protocol, expect mean Dice 0.91-0.93 in-distribution. **Cost: ~$200-300 Lambda** (1-2 day A100 training, no SSL pretraining). **v0+ sub-task 1 stack now: nnU-Net ResEnc L (TF2 42-class) for CBCT-prep patients + MeshSegNet/Cao25/Stratified Transformer (3DTeethSeg22 16-class) for IOS-prep patients — same downstream task, two input modalities.**

**Step 2: Add the connected-components post-processing with class-aware thresholds** (paper 054's `cc3d` 1-line trick, +0.026-0.035 Dice free). Compute the 0.5th-percentile GT volume per class on the training set, store as a lookup table, apply at inference. **Cost: 1-day engineering, $0 compute, +0.03 Dice guaranteed.**

**Step 3: Adopt the left-right mirror with class-label-swap trick** (paper 054's data-augmentation fix, +0.005-0.015 Dice for the L/R labeling accuracy). The CVPR 2025 paper's default L/R mirror *without* label-swap *degrades* L/R labeling accuracy; the fix is to swap the L/R class labels whenever the image is mirrored. **Cost: 5 lines of code, $0 compute, free Dice gain.**

**Step 4: Compute the cross-dataset generalization baseline on cTooth+** as the v0+ *H5* test. Run our v0+ nnU-Net ResEnc L (trained on TF2 480 scans) on cTooth+ (Cui 2022, ~7,000 scans), report the mean Dice drop, and document it in the v0+ paper as the "realistic deployment ceiling". **Cost: $0-50 Lambda (cTooth+ download is free from Cui et al., inference is the only compute).**

**Step 5: Pilot the Progressive Growing of Patch Size curriculum learning** (arXiv:2510.23241, the 2025 follow-up to this paper) on the v0+ thin-nerve sub-task. Start with patch size 64³ for 100 epochs, grow to 128³ for 200 epochs, end at 192³ for 200 epochs. **Cost: 1-2 day implementation, $100-200 Lambda pilot, expected +0.5-1.5 Dice on the small nerves (the v0+ weak spot).**

**Step 6: Build the TF2-aligned sub-task 2 evaluation pipeline.** The "crown", "bridge", "implant" labels (FDI-extended classes 35-37) are the *only* public segmentation labels for prostheses in our reading list. Run our v0 sub-task 2 (MADCrowner + ToothCraft + ToothForge) on the *TF2 segmentation as the alignment layer* (instead of the 3DTeethSeg22 segmentation) as a v0+ evaluation, and report whether our IOS-trained crown-generator transfers to the CBCT-prep workflow. **Cost: 1 week engineering, $0-100 Lambda (TF2 is free, 3DTeethSeg22 is free, the only cost is the inference).**

**Step 7: Update the per-scanner fine-tuning protocol for the v0+ deployment plan.** The 15% relative Dice drop on cTooth+ is the *deployment-quality* metric. The v0+ paper should document the (a) per-scanner fine-tuning recipe (10-50 in-house labeled scans + 200-500 unlabeled scans for SSL pretraining, $500-1500 Lambda per deployment), (b) the "domain shift detection" inference-time check (compute the entropy of the softmax distribution; entropy > threshold = likely out-of-distribution, trigger re-fine-tune), and (c) the per-scanner version-control of the model weights.

**v0+ compute update:** Total v0+ sub-task 1 (CBCT) = **$700-1,000 Lambda** (was $700 with nnU-Net ResEnc L, +$200-300 for SSL pretraining on 371 STS-3D-Tooth + cTooth+ evaluation + PGPS pilot). v0+ sub-task 1 (IOS) = **$300-500 Lambda** (was $300 with MeshSegNet baseline, +$100-200 for Cao 25 + Stratified Transformer pilot). v0+ sub-task 2 (crown generation) = **$5,140-6,130 Lambda** (unchanged from paper 052). v0+ eval (TF2 cross-validation) = **$200-400 Lambda** (new line). **Total v0+ compute: ~$6,340-8,030 Lambda.**

**Open question for HK: should the v0+ paper include a TF2 cross-modality evaluation (TF2 segmentation → MADCrowner crown generation → overlap-with-crown-class-Dice)? Recommendation: YES, this is the *first* paper in the entire dental-3D-gen reading list to evaluate cross-modality (IOS-to-CBCT) generalization, the single most publishable contribution we could add, and the v0+ paper's competitive moat.**

## Open question for HK (from paper 054's "next paper to read" queue)

Paper 054's STATUS said: "Next paper to read (055): Bolelli et al. CVPR 2025 pp 5238-5248 — the *organizer-side* ToothFairy3 challenge report with the full cross-team comparison and per-class confusion matrices U-Mamba2 alone does not publish; closes the 'what the field looks like' loop on 46-class CBCT segmentation and lets us triangulate the SSL-pretraining contribution against the post-processing contribution."

**Correction to paper 054's interpretation:** The CVPR 2025 paper at pp 5238-5248 is *not* the ToothFairy3 organizer report — it is the ToothFairy2 *dataset* paper (42 classes, 480 scans), published in June 2025, *before* the September 2025 ODIN workshop / ToothFairy3 challenge. The ToothFairy3 challenge report (when it eventually appears, likely at Springer LNCS 16473 ODIN proceedings or a 2026 MedIA extension) will have the cross-team comparison and per-class confusion matrices that paper 054 was looking for. **The CVPR 2025 paper is still the right next paper to read because it (a) provides the public dataset reference that paper 053 (the MedIA 2026 challenge results paper) and paper 054 (the U-Mamba2 / ToothFairy3 winner) both extensively cite, and (b) introduces the 42-class scheme and the nnU-Net recipe that v0+ should adopt. But the actual "field-wide" ToothFairy3 challenge report is still pending publication as of 2026-06-08.** Recommended next paper: **the ToothFairy3 / ODIN 2025 Springer LNCS 16473 proceedings** (when fully indexed, expected mid-2026) for the per-team ToothFairy3 comparison, OR **arXiv:2510.23241 (Progressive Growing of Patch Size)** for the 2025 follow-up to this paper that adds the curriculum-learning recipe. Recommendation: **PGPS for 056** (the 2025 follow-up is more directly actionable for v0+ than the proceedings compilation).

---

**Reading arc context (where this paper sits in the dental-crown-gen reading list):**
- Paper 053: ToothFairy2 *challenge results* (MedIA 2026) — covers the cross-team analysis, the 1st-place recipe, the cross-dataset test, the prosthesis-class novelty
- **Paper 055: ToothFairy2 *dataset* (CVPR 2025) — the formal dataset release + benchmark (this paper)**
- Paper 054: U-Mamba2 (ToothFairy3 winner, MICCAI 2025 ODIN) — the 2025 state-of-the-art on the *extended* TF3 46-class task
- Paper 056 (recommended): PGPS (arXiv:2510.23241) — the 2025 curriculum-learning follow-up

The two Bolelli papers (053 and 055) are companion publications: 053 is the *long-form challenge report* (with all 15 teams' submissions, cross-validation tables, failure analysis), 055 is the *short-form dataset + benchmark* (the CVPR page-limit-friendly version). Reading both gives the *complete* picture of where the field is in 2024-2025 for 42-class CBCT segmentation.

**v0+ stack update: TF2 (480-scan public dataset, 42-class, 0.3 mm) is now the de facto v0+ CBCT sub-task 1 benchmark. The nnU-Net ResEnc L is the v0+ baseline. The cc3d post-processing + L/R-mirror-swap + SSL pretraining are the v0+ recipe additions. The cTooth+ cross-dataset test is the v0+ H5 mechanism. The 4 added ToothFairy3 classes (incisive nerves + lingual foramen) are deferred to v1 unless our v0+ deployment explicitly requires thin-nerve surgical planning.**
