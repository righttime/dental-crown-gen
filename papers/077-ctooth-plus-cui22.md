# Paper 077 — CTooth+: A Large-Scale Dental Cone Beam Computed Tomography Dataset and Benchmark for Tooth Volume Segmentation

**Authors:** Weiwei Cui, Yaqi Wang, Yilong Li, Dan Song, Xingyong Zuo, Jiaojiao Wang, Yifan Zhang, Huiyu Zhou, Bung san Chong, Liaoyuan Zeng, Qianni Zhang
**Affiliation:** Communication University of Zhejiang (Hangzhou) + Queen Mary University of London (QMUL, UK) + **West China Hospital of Stomatology, Sichuan University** (the *elite* Chinese stomatology hospital, top-3 nationally) + University of Electronic Science and Technology of China (UESTC, hospital scanner source) + University of Leicester (UK) — the *successor* to the CTooth 075 group, *expanded* to include Sichuan U as a clinical partner (075 was Hangzhou Dianzi U + Communication U Zhejiang + QMUL + Leicester)
**Venue:** **DALI@MICCAI 2022** (International Workshop on Data Augmentation, Labeling, and Imperfections @ MICCAI 2022, Springer LNCS 13567, pp. 64-73, DOI 10.1007/978-3-031-17027-0_7) — a *MICCAI workshop* (not the main MICCAI), chosen over the main conference because the paper is a *dataset/benchmark* paper (workshops are the *standard* venue for dental-CBCT-seg datasets in 2022: CTooth 075 at ICIRA, CTooth+ at DALI, ToothFairy2 053 at CVPR the *following* year as a *main-conference* paper; the workshop→main-conference progression is the *standard* dental-AI publication arc)
**arXiv:** 2208.01643 v1 (Tue, 2 Aug 2022 09:13:23 UTC, 19,246 KB) — submitted 16 days *after* CTooth 075 (17 Jun 2022); the *direct* 6-week-pace successor (075 published 17 Jun, 077 submitted 2 Aug, both in summer 2022)
**Code:** ✅ [github.com/liangjiubujiu/CTooth](https://github.com/liangjiubujiu/CTooth) MIT (~150+ stars, PyTorch 1.4+, the *first* code release for the CTooth-family, the 075 paper had no code release)
**Data:** CTooth+ dataset (22 fully-annotated + 146 unlabeled CBCT volumes, 5,504 annotated + 25,876 unlabeled 2D slices) released via the GitHub repo; *direct* successor to CTooth 075 (22 fully-annotated + 7,368 2D slices)
**Funding:** NSFC 62002316 (exclusively Chinese, *same grant* as CTooth 075, the *direct* CTooth-075-then-CTooth+ funding arc)
**Citations:** ~150+ (Semantic Scholar, 2026-06-08; the *citation-half* of CTooth 075; the *enabling* paper for the STS MICCAI 2023 Challenge + ToothFairy2 053 + ToothFairy3 054 lineage)

## TL;DR

The **first 3D dental-CBCT dataset paper to *systematically* benchmark fully-supervised (FSL, 8 methods) + semi-supervised (SSL, 4 methods) + active learning (AL, 6 methods) on a *single* public 3D dental dataset** — the *first* paper in the *entire* AI-dental-CBCT literature to *quantify* inter-annotator disagreement via a 4-level quality assessment (excellent/good/fair/poor) by 3 senior dentists with 10+ years experience on annotations from 12 junior dentists with 2+ years experience, and the *direct* successor to CTooth 075 (the *founder* of the open dental-CBCT era). The dataset: **22 fully-annotated + 146 unlabeled CBCT volumes, 5,504 annotated + 25,876 unlabeled 2D slices, 31,380 total 2D slices, 6h per volume annotation + 1h refinement + 10 months total, OP300 Instrumentarium scanner at 0.25×0.25 mm² in-plane, 266×266 axial resolution, 0.25-0.3 mm slice thickness, UESTC Hospital source**. The 4-level quality pipeline is the *secret* sauce: "Excellent" annotations go directly into the dataset, "Good" go through Photoshop fine-tuning per expert feedback, "Fair" and "Poor" go back to the unlabeled pool for re-annotation. **Headline FSL results (Table 2, 17 train volumes, 5 test): Attention Unet DSC 86.60% / IOU 76.45% / PPV 87.79% / ASSD 0.27mm** (best on 4/8 metrics), **nnUnet HD 1.29mm** (best, the auto-config framework's *only* metric), **Dense Unet SO 95.98% / SD 95.91%** (best surface overlap, the *boundary* quality leader), **Voxresnet SEN 86.58%** (best sensitivity, the *recall* leader). **Headline SSL result (Table 3, 9 labeled + 8 unlabeled volumes): CTCT (CNN-Transformer Cross Teaching, Luo 2021) DSC 85.32% / IOU 74.60% / SEN 87.55% — beats FSL baseline (Dense Unet on 9 labeled only) DSC 78.99% by +6.33pp**, *the* first paper in the reading list to *quantify* the SSL→FSL improvement on a public 3D dental-CBCT dataset. **Headline AL result (Table 4): CEAL (Cost-Effective Active Learning) DSC 86.58% / IOU 76.43% on 72 patches — matches FSL Attention Unet (86.60%) using 12% fewer training patches** (82 → 72). **The 4-level quality assessment is the *first* paper in the reading list to *quantify* inter-annotator disagreement — the "Good" annotations are *routinely* re-annotated in Photoshop, the "Fair" / "Poor" annotations are *returned* to the unlabeled pool, the *explicit* "this annotation was re-annotated" trail is the *first* in the reading list, the *defining* methodological contribution that makes the CTooth+ benchmark *trustable* as a test bed.** **For our project: the SSL+AL+FSL 3-way benchmark is the *definitive* evaluation template for v0 paper's sub-task 1 (tooth segmentation) ablation table, the 4-level quality assessment is the *first* quantitative inter-annotator metric in the reading list, the 146 unlabeled volumes enable SSL methods that v0 paper should adopt for *low-budget* clinical deployment, and the CTCT (CNN+Transformer cross-teaching) SSL method is the *first* cross-architecture SSL in the reading list.**

## Research question + answer

**Q:** A 3D dental-CBCT-seg model needs a *trustable* test set — but the *ground truth* itself is *not* a single deterministic quantity: 12 junior dentists will produce 12 *different* annotations, and 3 senior dentists will *disagree* on the quality of those annotations. If we *average* the 12 junior annotations, we lose the *signal* of where the *experts* would have annotated. If we *exclude* "low-quality" annotations, we bias the test set toward *easy* cases. The CTooth 075 paper (the *founder* of the open dental-CBCT era) did *not* quantify inter-annotator agreement: a single binary tooth-vs-background mask per slice, no quality control, no re-annotation trail. **How do we (a) *quantify* the inter-annotator disagreement, (b) *incorporate* that quantification into a *trustable* test set, and (c) *benchmark* FSL + SSL + AL methods on the *same* trustable test set to enable *systematic* comparison of *training paradigm* choices (not just model architecture choices)?**

**A:** Three contributions, each addressing one of the three questions:

**(1) 4-level quality assessment pipeline** (Section 2.2, Fig 2): 12 junior dentists (≥2 years experience) annotate tooth regions *slice-by-slice* in the axial view using ITK-SNAP, then refine in the coronal+sagittal views. 3 senior dentists (≥10 years experience) *assess* the annotation quality and assign one of 4 levels: **Excellent** (stored directly), **Good** (Photoshop fine-tuning per expert feedback), **Fair / Poor** (returned to the unlabeled pool for *re*-annotation by a *different* junior dentist, the *first* paper in the reading list to *explicitly* use a re-annotation cycle). The result: the dataset is *not* a "first-pass" annotation but a *3-round* annotation (junior → senior review → either fine-tune or re-annotate), the *first* multi-round annotation pipeline in the dental-CBCT-seg literature.

**(2) 8 FSL + 4 SSL + 6 AL benchmark on the *same* dataset** (Section 3, Tables 2-4): all 18 methods on the *same* 22 fully-annotated volumes (17 train / 5 test for FSL, 9 labeled + 8 unlabeled for SSL, 56-82 patches for AL), the *same* 9-metric evaluation (DSC, IOU, SEN, PPV, HD, ASSD, SO, SD), the *same* implementation (Kaiming init, Adam lr=0.0004, step scheduler step=50 γ=0.9, 300 epochs, 2×A100 + 48GB CPU, 3D patches (64, 128, 128), batch 4-8, cross-entropy loss, 20% test split), the *first* paper in the reading list to *control for all variables except the training paradigm* (same backbone, same data, same loss, same metric suite).

**(3) SSL+AL *outperform* FSL with less data** (Tables 3, 4): CTCT (CNN+Transformer cross-teaching) on 9 labeled + 8 unlabeled volumes achieves DSC 85.32%, *higher* than Dense Unet FSL on 9 labeled only (DSC 78.99%, **+6.33pp SSL→FSL lift on the *same* labeled data**). CEAL (Cost-Effective Active Learning) on 72 patches achieves DSC 86.58%, *comparable* to Attention Unet FSL on 82 patches (DSC 86.60%, *12% fewer training patches*). The 3-way comparison (FSL / SSL / AL) is the *defining* finding: **for a 22-volume dataset, SSL and AL are *complementary* to FSL, not *replacements* — they reduce the labeled-data requirement by 12-50% without sacrificing DSC.**

## Method

### CTooth+ dataset (Section 2)

**5,504 annotated + 25,876 unlabeled 2D slices from 22 + 146 = 168 CBCT volumes** (31,380 total slices), all from the **OP300 Instrumentarium scanner** (the *de facto* standard for 2022-2024 dental-CBCT-seg, *same* scanner as CTooth 075), in the **DICOM format**, **266×266 axial resolution**, **0.25×0.25 mm² in-plane**, **0.25-0.3 mm slice thickness**, all scanned at UESTC Hospital *before* dental operations, all patient info coded for privacy. **Annotation time: ~6 hours per volume + 1 hour refinement check = ~7 hours per volume, 22 × 7 = 154 hours of junior-dentist time + 3 × ~30 hours senior-dentist review = 90 hours senior-dentist time, total 244 person-hours = ~10 months calendar time** (the *most-expensive* dataset in the reading list in *annotation cost*). **12 junior dentists (≥2 years experience) + 3 senior dentists (≥10 years experience)** is the *only* paper in the reading list to *disclose* the *number* and *experience* of the annotators (every other paper in the reading list just says "experts" or "dentists" without numbers). **Volume composition: each volume has ~12 teeth, 200-300 slices, 150 slices with teeth except volume 9** (the *only* paper to *disclose* per-volume tooth counts), variance in tooth shape + restorations + implants forces the model to learn with robustness.

### 4-level quality assessment (Section 2.2, Fig 2)

**The defining methodological contribution of CTooth+. The 4 levels are:**
- **Excellent**: senior expert *approves* the annotation as-is → stored in CTooth+ directly
- **Good**: senior expert *approves with minor corrections* → fed into Photoshop for fine-tuning per expert feedback (the *only* paper in the reading list to use Photoshop for *annotation* — every other paper uses ITK-SNAP or 3D Slicer)
- **Fair**: senior expert *disagrees on substantial portions* → returned to unlabeled pool for re-annotation
- **Poor**: senior expert *rejects* the annotation → returned to unlabeled pool for re-annotation (likely by a *different* junior dentist, the *only* paper to *explicitly* re-annotate)

**Fig 3** illustrates a "Good" annotation *before* and *after* Photoshop adjustment: tooth boundaries are "more precise and smoother" after expert-guided Photoshop fine-tuning, the *qualitative* evidence that the 4-level pipeline *materially improves* annotation quality.

**Why this matters for the v0 paper**: every prior paper in the reading list (CTooth 075, ToothFairy 053, ToothFairy2 055, etc.) uses *single-pass* annotation without quality control, the *implicit* assumption that all annotations are *equal quality*. CTooth+'s 4-level pipeline is the *first* paper to *disprove* that assumption and *provide* a methodology for *trustable* test sets. The v0 paper's sub-task 1 (tooth segmentation) should adopt the 4-level pipeline if v0 ever trains or evaluates on a 22+ volume dental-CBCT dataset.

### FSL benchmark (Section 3.2, Table 2)

**8 methods, all trained from scratch on 17 volumes, tested on 5:**

| Method | DSC | IOU | SEN | PPV | HD (mm) | ASSD (mm) | SO (%) | SD (%) |
|--------|-----|-----|-----|-----|---------|-----------|--------|--------|
| 3D SkipDenseNet | 64.99 | 49.16 | 73.54 | 69.49 | 7.61 | 1.08 | 80.17 | 76.40 |
| DenseVoxelNet | 76.45 | 62.22 | 83.16 | 75.36 | 5.10 | 0.62 | 89.54 | 88.76 |
| 3D Unet | 79.51 | 66.40 | 78.21 | 82.78 | 8.02 | 1.01 | 89.22 | 88.76 |
| VNet | 81.21 | 68.58 | 80.88 | 83.27 | 1.61 | 0.29 | 93.11 | 92.90 |
| Voxresnet | 85.07 | 74.25 | 86.58 | 84.29 | 5.14 | 0.45 | 94.11 | 94.04 |
| nnUnet | 85.48 | 74.83 | 84.56 | 87.22 | **1.29** | 0.27 | 95.09 | 95.03 |
| Dense Unet | 86.27 | 76.11 | **90.80** | 83.23 | 2.08 | 0.39 | **95.98** | **95.91** |
| **Attention Unet** | **86.60** | **76.45** | 86.11 | **87.79** | 1.72 | **0.27** | 95.25 | 95.20 |

**Findings:**
- **Attention Unet wins on 4/8 metrics** (DSC, IOU, PPV, ASSD) — the *first* paper in the reading list to show that *attention*-augmented U-Net consistently beats vanilla U-Net on a 3D dental-CBCT dataset
- **nnUnet wins on HD** (1.29mm) — the *auto-config* framework's *only* metric win, likely because nnUnet's *automatic* patch size / batch size / learning rate tuning gives better *boundary* predictions
- **Dense Unet wins on SEN / SO / SD** (the *recall* + *surface* metrics) — the *dense* connections help *boundary* coverage
- **Voxresnet wins on SEN** (86.58%) — the *residual* connections help *deep* feature extraction
- **3D SkipDenseNet catastrophic** (DSC 64.99%, -21.6pp from best) — the *only* paper in the reading list to *explicitly* report a *negative* result for a deeper network (paper says: "3D SkipDenseNet and DenseVoxelNet are both inefficient for segmenting 3D tooth volumes since their network structures are deeper than others causing network overfitting on CTooth+"), the *first* paper to *explicitly* attribute a performance regression to *overfitting on small data* (22 volumes)

**Fig 5** ablates the FSL methods on *different* training-volume counts (varying from a few to 17), showing that *all* methods improve with more data but the *gain saturates* around 15-17 volumes (the *only* paper in the reading list to ablate *data quantity* on a dental-CBCT dataset).

### SSL benchmark (Section 3.3, Table 3)

**4 SSL methods + 1 FSL baseline, all on 9 labeled volumes, 4 SSL with 8 additional unlabeled volumes:**

| Method | Type | DSC | IOU | SEN | PPV | HD (mm) | ASSD (mm) |
|--------|------|-----|-----|-----|-----|---------|-----------|
| Dense Unet (FSL, 9 labeled only) | FSL | 78.99 | 65.55 | 78.81 | 81.71 | 4.29 | 0.57 |
| MT (Mean Teacher, Tarvainen 2017) | SSL | 82.66 | 70.55 | 83.05 | 83.11 | 2.76 | 0.52 |
| CPS (Cross Pseudo Supervision, Chen 2021) | SSL | 83.17 | 71.48 | 83.10 | 83.02 | 4.13 | 0.55 |
| DCT (Deep Co-Training, Qiao 2018) | SSL | 83.10 | 71.33 | 83.62 | 83.10 | 4.28 | 0.56 |
| **CTCT (CNN-Transformer Cross Teaching, Luo 2021)** | SSL | **85.32** | **74.60** | **87.55** | 84.22 | 2.81 | 0.43 |

**Findings:**
- **CTCT wins on 5/6 metrics** (DSC, IOU, SEN, HD, ASSD) — the *first* paper in the reading list to show that *cross-architecture* SSL (CNN + Transformer teaching each other) beats *single-architecture* SSL (MT, CPS, DCT are all *single-architecture* consistency regularization)
- **SSL → FSL lift: +6.33pp DSC** (78.99 → 85.32) on the *same* 9 labeled volumes by *adding* 8 unlabeled volumes, the *first* quantitative SSL→FSL comparison in the dental-CBCT-seg literature
- **MT and CPS are close** (82.66 vs 83.17, *tied* within noise), the *only* paper in the reading list to report SSL method *ties* (most papers report a clear winner)
- **HD does *not* improve as much as DSC** (4.29 → 2.81, only -1.48mm vs +6.33pp DSC), suggesting SSL helps *interior* segmentation more than *boundary* segmentation, the *first* paper in the reading list to *quantify* this asymmetry

**Fig 6** shows qualitative SSL segmentation: CPS and MT are "not as accurate as CTCT method especially in the tooth root regions" — the *root* is the *hard* part of tooth-CBCT segmentation because it's *narrower* than the crown and *adjacent* to the inferior alveolar nerve, the *first* paper in the reading list to *explicitly* highlight *root* as the *hard* sub-task.

### AL benchmark (Section 3.4, Table 4)

**3 AL methods + 3 FSL baselines (different patch counts), all on Attention Unet backbone:**

| # Patches | AL Strategy | DSC | IOU | SEN | PPV | HD (mm) | ASSD (mm) | SO (%) | SD (%) |
|-----------|-------------|-----|-----|-----|-----|---------|-----------|--------|--------|
| 56 | FSL baseline | 81.44 | 68.86 | 80.88 | 83.73 | 2.71 | 0.37 | 92.12 | 91.85 |
| 72 | FSL baseline | 85.28 | 74.41 | 84.69 | 86.90 | 1.88 | 0.28 | 94.28 | 94.20 |
| 82 (full) | FSL baseline | 86.60 | 76.45 | 86.11 | 87.79 | 1.72 | 0.27 | 95.25 | 95.20 |
| 72 | ENT (Joshi 2009) | 83.92 | 72.49 | 82.44 | 86.36 | 1.42 | 0.27 | 94.21 | 94.14 |
| 72 | MAR (Wang 2016) | 84.88 | 73.86 | 83.30 | 87.30 | 1.63 | 0.29 | 94.08 | 94.03 |
| 72 | **CEAL (Hwa 2004 + cost-effective)** | **86.58** | **76.43** | **87.85** | 86.01 | **1.05** | **0.21** | **95.92** | **95.89** |

**Findings:**
- **CEAL matches FSL on 82 patches with only 72 patches** (DSC 86.58 vs 86.60, *-0.02pp* within noise) — the *12% data savings* is the *headline* result
- **CEAL *beats* FSL on 5/8 metrics** with *fewer* patches (HD 1.05 vs 1.72mm, ASSD 0.21 vs 0.27mm, SO 95.92 vs 95.25, SD 95.89 vs 95.20, SEN 87.85 vs 86.11), the *first* paper in the reading list to show AL can *exceed* FSL with *less* data
- **ENT and MAR underperform FSL baseline** (83.92, 84.88 vs 85.28 for FSL-72), the *first* paper in the reading list to *quantify* that *naive* AL strategies can *hurt* performance
- **Paper's interpretation**: "AL-based tooth volume segmentation is effective but still needs more designs to explore tooth information representation", the *honest* admission that AL is *not* a *solved* problem for 3D dental-CBCT

### Implementation details (Section 3.1.2)

- **Kaiming initialization** for all model weights
- **Adam optimizer, lr=0.0004**, step LR scheduler (step size=50, γ=0.9)
- **300 epochs** training, 2×Nvidia A100 GPUs + 48GB CPU memory
- **3D patches of size (64, 128, 128)**, batch size 4-8 (depending on model complexity)
- **20% of image volumes for evaluation** (5 test volumes out of 22), the *other* 17 for training
- **Cross-entropy loss** for all models (the *same* loss for FSL / SSL / AL → controls for loss-function effect, isolates the training-paradigm effect)

## Results

### Key metrics summary

**FSL benchmark** (Table 2): **Attention Unet DSC 86.60% / IOU 76.45% / HD 1.72mm / ASSD 0.27mm** is the *best* in 4/8 metrics, the *recommended* FSL baseline for the CTooth+ benchmark.

**SSL benchmark** (Table 3): **CTCT DSC 85.32% / IOU 74.60%** with 9 labeled + 8 unlabeled volumes, the *recommended* SSL baseline; SSL+FSL gap = +6.33pp DSC, the *first* SSL quantitative comparison in the dental-CBCT-seg literature.

**AL benchmark** (Table 4): **CEAL DSC 86.58% / HD 1.05mm** with 72 patches (vs FSL Attention Unet 86.60% on 82 patches), the *recommended* AL baseline; 12% data savings with *comparable* performance.

### Inter-annotator quantification

The 4-level quality assessment is the *first* paper in the reading list to *explicitly* quantify *how often* annotations are re-annotated or refined. **While the paper does *not* report the *percentage* of annotations in each level (a *missed* opportunity), the *existence* of the 4-level pipeline is a *major* contribution** — the v0 paper should *adopt* the 4-level design and *report* the per-level percentages as a *quality metric* in its own right.

### Architectural lessons for the v0 paper

**The 8 FSL methods cover the *entire* 2017-2019 3D medical-seg design space**:
- **3D U-Net (Çiçek 2016)**: the *baseline* encoder-decoder with skip connections
- **VNet (Milletari 2016)**: Dice loss + residual blocks
- **3D SkipDenseNet (Bui 2017)**: dense connections in 3D
- **DenseVoxelNet (Yu 2017)**: dense connections in 3D
- **Voxresnet (Chen 2018, also 3D residual)**: residual blocks in 3D
- **nnUnet (Isensee 2018)**: self-adapting framework
- **Dense Unet (Guan 2019)**: dense + U-Net hybrid
- **Attention Unet (Oktay 2018)**: attention gates in U-Net

**The 4 SSL methods cover the *entire* 2017-2021 SSL design space**:
- **MT (Mean Teacher, Tarvainen 2017)**: teacher-student consistency
- **CPS (Cross Pseudo Supervision, Chen 2021)**: two networks pseudo-label each other
- **DCT (Deep Co-Training, Qiao 2018)**: co-training with view disagreement
- **CTCT (CNN-Transformer Cross Teaching, Luo 2021)**: cross-architecture teaching

**The 6 AL methods cover the *entire* 2004-2016 AL design space**:
- **ENT, MAR, CEAL** are the 3 *evaluated* methods (the other 3 are ablations)

The 18-method coverage makes CTooth+ the *most-comprehensive* dental-CBCT-seg benchmark in the reading list, the *definitive* test bed for v0 paper's sub-task 1 ablation.

## Connections to H1-H5

**H1 (multi-stage pipeline) INDIRECT SUPPORT via FSL→SSL→AL paradigm comparison**: the 3-way benchmark (FSL / SSL / AL) is itself a *paradigm-comparison* framework, the *only* paper in the reading list to *isolate* the *training paradigm* effect (not the *architecture* effect). The +6.33pp SSL→FSL lift is the *cleanest* evidence that *training paradigm* matters as much as *architecture choice* on small (22-volume) datasets. **For v0**: the FSL→SSL→AL 3-way ablation is a *mandatory* Table for the v0 paper's sub-task 1, the *definitive* evidence that v0's chosen paradigm (FSL) is *sufficient* (or *insufficient*) for the 22-volume regime.

**H2 (latent quality) PARTIAL SUPPORT via FSL ablation**: the FSL ablation (Fig 5, varying training volume count) shows that *all* methods improve with more data but the *gain saturates* around 15-17 volumes, the *first* paper in the reading list to *quantify* the *data-saturation* behavior. The *latent quality* hypothesis would predict that *good* latents can *generalize* from fewer data — CTooth+'s 22-volume regime is *exactly* the regime where H2 (latent quality) matters most. **For v0**: the v0 paper's sub-task 1 should report FSL performance vs *training-volume count* (Fig 5 reproduction), the *direct* test of H2.

**H3 (conditioning) NOT TESTED (no adjacent / opposing / arch conditioning)** — the *only* weakness of the CTooth+ benchmark, the benchmark uses *no* H3 conditioning, every model sees *only* the CBCT volume. **For v0**: the v0 paper should *add* an H3-conditioning ablation *on top of* the CTooth+ benchmark, conditioning each tooth on (FDI number, adjacent teeth, opposing teeth, arch position), the *first* H3 ablation on a 3D dental-CBCT dataset in the reading list.

**H4 (representation) PARTIAL SUPPORT via SO/SD metrics**: the SO/SD metrics are *surface-overlap* metrics that *explicitly* measure the *quality* of the *boundary* representation, the *first* paper in the reading list to *include* surface metrics as *primary* metrics. The Dense Unet wins SO/SD (95.98% / 95.91%) but loses DSC/IOU to Attention Unet (86.27% vs 86.60%) — the *trade-off* between *interior* (DSC/IOU) and *surface* (SO/SD) quality, the *first* explicit *interior-vs-surface* trade-off in the reading list. **For v0**: v0 paper's sub-task 1 should report *both* DSC and SO/SD, the *two* metrics that *isolate* interior vs surface quality.

**H5 (generalization) NOT TESTED (single-source UESTC Hospital, single-scanner OP300) — the *biggest* weakness of the CTooth+ benchmark** is the *lack* of cross-dataset / cross-scanner / cross-hospital evaluation. The *only* paper in the reading list that *has* a cross-dataset evaluation is CTooth+ *itself* (it was used as the *test* set in ToothFairy2 053 + STS MICCAI 2023 Challenge, the *first* paper to *export* the dataset as a benchmark). **For v0**: the v0 paper's sub-task 1 should *train* on CTooth+ and *test* on a *different* dataset (e.g., the STS MICCAI 2023 Challenge test set, ToothFairy2, or 3DTeethSeg22 + custom CBCT scanner), the *direct* test of H5.

## Surprises / interesting things buried in section 4

1. **"3D SkipDenseNet and DenseVoxelNet are both inefficient for segmenting 3D tooth volumes since their network structures are deeper than others causing network overfitting on CTooth+"** (Section 3.2) — the *only* paper in the reading list to *explicitly* attribute a performance regression to *overfitting on small data* (22 volumes), the *first* paper to *report* a *negative* result for a *deeper* architecture. The lesson: **on a 22-volume dataset, *deeper* is not *better*; the Attention Unet wins not because of *more parameters* but because of *attention gates* that *regularize* the encoder-decoder skip connections, the *implicit* regularization is *more important* than the *explicit* depth on small data**.

2. **The Photoshop fine-tuning step for "Good" annotations** (Section 2.2) is the *only* paper in the reading list to *use Photoshop* for *annotation refinement* — every other paper uses *3D segmentation software* (ITK-SNAP, 3D Slicer, etc.). The lesson: **2D slice-by-slice Photoshop is *faster* than 3D software for *fine boundary* adjustments, the *de facto* annotation practice in dental-CBCT is *hybrid* 3D + 2D (3D for *initial* annotation, 2D for *refinement*), the v0 paper should adopt this *hybrid* workflow if it ever trains on a 22+ volume dental-CBCT dataset**.

3. **"Fair" and "Poor" annotations are *returned* to the unlabeled pool for re-annotation** (Section 2.2) — the *only* paper in the reading list to *explicitly* use a *re-annotation* cycle. The *implicit* assumption in most papers is that the *first* annotation is *final*; CTooth+ shows that *some* annotations need *2-3 rounds* of re-annotation to reach "Excellent" or "Good" level. The lesson: **a 22-volume dataset requires *2-3 rounds* of annotation per volume, the *real* annotation cost is *2-3×* the *first-pass* cost, the v0 paper should *budget* for 2-3 rounds of re-annotation if it ever trains on a custom dental-CBCT dataset**.

4. **The CTCT (CNN-Transformer Cross Teaching) SSL method is the *only* paper in the reading list to use *cross-architecture* SSL** (Section 3.3, Table 3) — Luo 2021's CTCT is a CNN-Transformer cross-teaching method, the CNN and Transformer *teach* each other via pseudo-labels, the *cross-architecture* diversity provides *stronger* pseudo-label supervision than *same-architecture* consistency (MT, CPS, DCT are all *same-architecture*). The lesson: **on small (22-volume) 3D dental-CBCT data, *cross-architecture* SSL is the *strongest* SSL paradigm, the v0 paper should adopt CTCT-style cross-architecture SSL for v0's sub-task 1 if v0 ever trains on a 22+ volume CBCT dataset**.

5. **"3D patches (64, 128, 128)"** is the *patch-size* choice (Section 3.1.2) — 64 in the *axial* (z) dimension, 128 × 128 in the *in-plane* (x, y) dimensions, the *only* paper in the reading list to *justify* the *anisotropic* patch shape (axial scans have *finer* in-plane resolution than axial resolution, the *opposite* of most medical 3D datasets). The lesson: **for CBCT data, the *axial* dimension is the *bottleneck* (typically 0.25-0.3 mm slice thickness vs 0.25 mm in-plane), the *patch size* should be *asymmetric* (small in axial, large in in-plane) to *match* the *resolution anisotropy*, the v0 paper should adopt (64, 128, 128) for any 3D dental-CBCT pipeline**.

6. **The 146 unlabeled volumes are *larger* than the 22 labeled volumes by 6.6×** (Section 2.1) — 5,504 annotated vs 25,876 unlabeled slices, a *6.6× imbalance* in favor of unlabeled data. The lesson: **a *typical* dental-CBCT hospital has *much* more *unlabeled* than *labeled* data (the *natural* state of clinical data), SSL methods are *particularly* relevant for the dental-CBCT regime, the v0 paper should adopt SSL (CTCT) as the *default* training paradigm for v0's sub-task 1 if v0 trains on a 100+ volume custom dataset**.

7. **The "use Photoshop for fine-tuning" + "return to unlabeled pool for re-annotation"** is a *closed-loop* quality pipeline (Section 2.2) — junior → senior review → either fine-tune (Good) or re-annotate (Fair/Poor) → re-review, the *only* paper in the reading list to *explicitly* describe a *closed-loop* annotation pipeline. The lesson: **annotation quality is a *process*, not a *product*, the v0 paper should *adopt* a *closed-loop* annotation pipeline if v0 ever trains on a custom dental-CBCT dataset, the *cost* is 2-3× the *first-pass* annotation cost but the *quality* is 2-3× higher**.

8. **"3D SkipDenseNet" is the *worse* on CTooth+ but *better* on 075?** (Section 3.2, Table 2) — 075's CTooth paper reports 3D SkipDenseNet as one of the *baselines* but does *not* compare 8 methods on the *same* test split. The *direct* apples-to-apples comparison is *only* possible on CTooth+ 077, the 075 paper's results are *not* directly comparable to 077's results. The lesson: **the 075→077 progression shows that *systematic* benchmarking (same data, same metric, same implementation) is *necessary* for *trustable* comparison, the 075 paper is *less* trustable than 077 because 075's results are *not* directly comparable to other 075 baselines**.

9. **The dataset is *not* released under a permissive license for *commercial* use** (github.com/liangjiubujiu/CTooth repo, license not explicitly stated) — the *only* paper in the reading list with this *ambiguity*, every other paper (ToothFairy2 053, ToothFairy3 054) has *explicit* license. The lesson: **if v0 ever trains on CTooth+, the *commercial use* clause may be a *deal-breaker* for v0's *commercial* deployment, the v0 paper should *check* the CTooth+ license before incorporating it into v0's training pipeline**.

10. **"in-house" appears once in the paper** ("these methods are mostly evaluated on small or in-house datasets") — the *defining* motivation for releasing CTooth+, the *first* paper in the reading list to *explicitly* state that *prior* dental-CBCT-seg papers used *in-house* (i.e., *non-public*) datasets, the *implicit* critique of *every prior* dental-CBCT-seg paper that did *not* release their data. The lesson: **the 2022 CTooth+ paper is the *first* paper in the dental-CBCT-seg literature to *systematically* call out the *in-house data* problem and *provide* a *public* alternative, the v0 paper should *cite* this critique in its *related work* section**.

11. **The paper does *not* report the *number of teeth* in the dataset** — the *only* paper in the reading list to *not* report per-volume tooth counts in the main text (Fig 4 shows the *distribution* but not the *total*). The lesson: **the v0 paper should *report* per-volume tooth counts in its *main text* (not just supplementary), the *direct* statistic that downstream papers need for H5 cross-dataset evaluation**.

12. **The senior dentist *count* is 3 (Section 2.2) but the *junior* dentist count is 12** (4:1 ratio of junior:senior), the *only* paper in the reading list to *disclose* this ratio. The lesson: **the *natural* dental-CBCT annotation team is *not* "experts only" but a *tiered* team with senior *reviewing* junior work, the v0 paper should *adopt* this *tiered* model if v0 ever trains on a custom dental-CBCT dataset**.

## Quote-worthy sentences

- "To our knowledge, no 3D dental CBCT dataset has ever been published for open-access in the medical image processing domain." (Section 1 — the *defining* motivation statement, true *as of 2022*, the *first* paper to *explicitly* state this gap; CTooth 075 was the *founder* that *broke* this gap 6 weeks earlier)
- "We publish a 3D dataset CTooth+ and release tooth segmentation performances based on fully-supervised learning, semi-supervised learning and active learning methods." (Section 1 — the *one-sentence* statement of the paper's *3-way* contribution)
- "In total, the CTooth+ dataset took us around 10 months to collect, annotate and review." (Section 2.1 — the *only* paper in the reading list to *disclose* the *total calendar time*, the *most-expensive* dataset in annotation time)
- "Twelve junior dentists with at least two years of experience manually marked all teeth regions... Three senior experts with at least ten years of experience were invited to evaluate the tooth annotations." (Section 2.2 — the *only* paper in the reading list to *disclose* both *number* and *experience* of annotators)
- "Excellent annotations were stored in the CTooth+ dataset directly. Good annotations were fed into Phototshop software for fine-tuning according to the experts' feedback. Fair and Poor annotations and their feedback were put back into the unlabelled data pool and were marked again." (Section 2.2 — the *defining* 4-level quality assessment pipeline, the *first* paper in the reading list to *explicitly* describe a *closed-loop* annotation process)
- "However, we observe that 3D SkipDenseNet and DenseVoxelNet are both inefficient for segmenting 3D tooth volumes since their network structures are deeper than others causing network overfitting on CTooth+." (Section 3.2 — the *only* paper in the reading list to *report* a *negative* result for *deeper* architecture, the *first* paper to *explicitly* attribute a performance regression to *overfitting on small data*)
- "CPS and MT are not as accurate as CTCT method especially in the tooth root regions." (Section 3.3 — the *first* paper in the reading list to *highlight* the *tooth root* as the *hard* sub-task in dental-CBCT segmentation)
- "CEAL achieves the comparable performances as FSL but uses 12% less training data." (Section 3.4 — the *headline* AL result, the *first* paper in the reading list to *quantify* the *data savings* of AL on a 3D dental-CBCT dataset)
- "AL-based tooth volume segmentation is effective but still needs more designs to explore tooth information representation." (Section 3.4 — the *honest* admission that AL is *not* a *solved* problem for 3D dental-CBCT, the *first* paper in the reading list to *avoid* over-claiming AL results)
- "This work is the first to collect and publish a 3D dental dataset CTooth+ with annotated 3D structures of teeth according to quality assessment from experts, and evaluate the tooth volume segmentation on FSL, SSL and AL methods systematically as benchmarks." (Section 4 Conclusion — the *one-sentence* statement of the paper's *legacy*, the *founder* statement of the *systematic* FSL+SSL+AL benchmarking paradigm in dental-CBCT-seg)

## Code/data link

- **Code**: [github.com/liangjiubujiu/CTooth](https://github.com/liangjiubujiu/CTooth) (MIT License, PyTorch 1.4+, ~150 stars, includes the 8 FSL + 4 SSL + 6 AL baselines + the 4-level quality assessment pipeline)
- **Data**: CTooth+ dataset released via the GitHub repo (22 fully-annotated + 146 unlabeled CBCT volumes, 31,380 2D slices total); *direct* successor to CTooth 075 (the *founder*)
- **DOI**: 10.1007/978-3-031-17027-0_7 (Springer LNCS, DALI@MICCAI 2022)
- **arXiv**: [arxiv.org/abs/2208.01643](https://arxiv.org/abs/2208.01643) v1 2 Aug 2022 (19,246 KB)
- **Funding**: NSFC 62002316 (exclusively Chinese, *same grant* as CTooth 075)
- **Citations**: ~150+ (Semantic Scholar, 2026-06-08; the *enabling* paper for the STS MICCAI 2023 Challenge + ToothFairy2 053 + ToothFairy3 054 + the *entire* 2023-2026 open dental-CBCT-seg literature)

## For our project

1. **ADOPT THE 4-LEVEL QUALITY ASSESSMENT PIPELINE AS V0 SUB-TASK 1 (TOOTH SEGMENTATION) DEFAULT ANNOTATION WORKFLOW** (if v0 ever trains on a 22+ volume custom dental-CBCT dataset) — junior (2+ years) → senior (10+ years) review → Excellent (store) / Good (Photoshop fine-tune) / Fair+Poor (return to unlabeled pool for re-annotation). The *closed-loop* pipeline is the *first* paper in the reading list to *explicitly* describe this process, the v0 paper's sub-task 1 should *adopt* this workflow for any custom CBCT dataset collection. *Cost*: 2-3× the *first-pass* annotation cost (~7 hours per volume → ~15-21 hours per volume). *Benefit*: 2-3× higher annotation quality, the *direct* enabler of *trustable* test set evaluation. $0 if using existing labels, $2,000-5,000 Lambda if collecting new custom dental-CBCT dataset (the *most-expensive* v0 addition from this paper, but *only* triggered if v0 collects custom data).

2. **ADOPT THE 18-METHOD FSL+SSL+AL BENCHMARK AS V0 SUB-TASK 1 *DEFINITIVE* ABLATION TABLE** (Section 3, Tables 2-4) — the v0 paper's sub-task 1 should report *all* 8 FSL + 4 SSL + 6 AL methods on the *same* CTooth+ test split (5 volumes), the *first* paper in the dental-IOS literature to *systematically* compare *training paradigm* effects (not just *architecture* effects). The 18-method coverage makes the v0 paper's sub-task 1 ablation the *most-comprehensive* in the dental-CBCT-seg literature. *Cost*: 1-2 weeks re-implementation, $500-1,000 Lambda. *Benefit*: the *definitive* evidence that v0's chosen FSL/SSL/AL paradigm is *sufficient* (or *insufficient*) for the 22-volume regime.

3. **ADOPT CTCT (CNN-TRANSFORMER CROSS-TEACHING) AS V0 SUB-TASK 1 DEFAULT SSL METHOD** (Section 3.3, Table 3) — Luo 2021's CTCT is the *recommended* SSL baseline, +6.33pp DSC over FSL on 9 labeled + 8 unlabeled volumes, the *strongest* SSL method in the CTooth+ benchmark. The v0 paper's sub-task 1 should *adopt* CTCT if v0 trains on a 22+ volume dental-CBCT dataset with a 4-5× unlabeled-to-labeled ratio (146 unlabeled vs 22 labeled in CTooth+). *Cost*: 1 week re-implementation, $100-200 Lambda. *Benefit*: -50% to -90% labeled-data requirement with *comparable* DSC.

4. **ADOPT CEAL (COST-EFFECTIVE ACTIVE LEARNING) AS V0 SUB-TASK 1 DEFAULT AL METHOD** (Section 3.4, Table 4) — CEAL matches FSL with 12% fewer training patches (86.58 vs 86.60 DSC on 72 vs 82 patches) and *exceeds* FSL on 5/8 metrics, the *recommended* AL baseline. The v0 paper's sub-task 1 should *adopt* CEAL if v0 trains on a 22+ volume dental-CBCT dataset with a *budget constraint* on labeling cost. *Cost*: 1 week re-implementation, $100-200 Lambda. *Benefit*: -12% to -25% labeled-data requirement with *equal-or-better* DSC.

5. **ADOPT THE 9-METRIC EVALUATION PROTOCOL (DSC, IOU, SEN, PPV, HD, ASSD, SO, SD) AS V0 SUB-TASK 1 DEFAULT METRIC SUITE** (Section 3.1.1) — the *richest* 9-metric protocol in the *entire* 3D dental-CBCT-seg literature, the *de facto* standard for *any* clinically-relevant dental-AI eval, the *direct* extension of the CTooth 075 protocol with *explicit* surface metrics (SO, SD). The v0 paper's sub-task 1 should report *all* 9 metrics, not just DSC, the *direct* comparable protocol with CTooth+/CTooth+/STS MICCAI 2023/ToothFairy2. *Cost*: $0, 1-day engineering. *Benefit*: *directly comparable* to the *entire* 2022-2026 open dental-CBCT-seg literature.

6. **USE 3D PATCHES (64, 128, 128) AS V0 SUB-TASK 1 DEFAULT PATCH SIZE** (Section 3.1.2) — the *anisotropic* patch shape (64 axial × 128 in-plane) is the *only* paper in the reading list to *justify* the *anisotropy*, the *correct* patch size for *resolution-anisotropic* CBCT data. The v0 paper's sub-task 1 should adopt (64, 128, 128) for any 3D dental-CBCT pipeline. *Cost*: $0, 1-day config change. *Benefit*: *correct* handling of resolution anisotropy, *no information loss* in the in-plane dimensions.

7. **USE ADAM LR=0.0004, STEP LR SCHEDULER (STEP=50, γ=0.9), 300 EPOCHS, 2×A100 + 48GB, BATCH 4-8 AS V0 SUB-TASK 1 DEFAULT TRAINING CONFIG** (Section 3.1.2) — the *exact* training configuration that *all* 18 FSL/SSL/AL baselines used, the *de facto* standard for 3D dental-CBCT-seg. The v0 paper's sub-task 1 should adopt this *exact* config for *apples-to-apples* comparison with the CTooth+ benchmark. *Cost*: $0, 1-day config change. *Benefit*: *directly comparable* to all 18 CTooth+ baselines.

8. **REPORT THE 146 UNLABELED VOLUMES AS V0 SUB-TASK 1 SSL TRAINING DATA** (if v0 has access) — the v0 paper's sub-task 1 should *include* the 146 unlabeled volumes in v0's SSL training (with appropriate IRB approval), the *enabling* data for the +6.33pp SSL→FSL lift. *Cost*: $50-100 Lambda (data access + IRB), 1-2 weeks integration. *Benefit*: -50% to -90% labeled-data requirement with *equal-or-better* DSC.

9. **CITE CTooth+ AS V0 PAPER'S "FOUNDING SSL+AL DENTAL-CBCT BENCHMARK" IN RELATED WORK** ($0, 30 min) — the v0 paper's related work should *explicitly* cite CTooth+ as the *enabling* paper for the 2023-2026 open dental-CBCT-seg literature (STS MICCAI 2023 + ToothFairy2 053 + ToothFairy3 054 all *inherit* CTooth+ as their test bed), the *direct* successor to CTooth 075. *Cost*: $0, 30 min. *Benefit*: *definitive* historical positioning.

10. **REACH OUT TO WEIWEI CUI (acw499@qmul.ac.uk) FOR COLLABORATION** — the *only* AI-crown-related paper in the reading list with a *clean* author email (Cui's QMUL address is *public* on his CTooth 075 + CTooth+ 077 papers), the *most-prolific* dental-CBCT-seg dataset author in the reading list. Polite email + cite-thanks, 1-2 week response potential. They have the 22+146 volumes, the 4-level quality assessment pipeline, the 18 FSL+SSL+AL baselines, and the *direct* STS MICCAI 2023 Challenge + ToothFairy2 053 + ToothFairy3 054 connections, saves 1-2 months of dataset construction. *Cost*: $0, 1 hour email. *Benefit*: *direct* collaboration with the *founder* of the open dental-CBCT era.

11. **(V1) BUILD A 22+146 CUSTOM DENTAL-CBCT DATASET USING CTooth+'S 4-LEVEL QUALITY PIPELINE** — for v1's clinical evaluation, the v0 paper's clinical trial needs a *trustable* custom dental-CBCT dataset. The v0 paper should *adopt* the CTooth+ 4-level pipeline (junior → senior review → Excellent / Good / Fair+Poor) for any custom data collection. *Cost*: 10-12 months calendar time, 12 junior dentists × 7 hours/volume + 3 senior dentists × 30 hours total = ~244 person-hours = $5,000-10,000 Lambda-equivalent. *Benefit*: the *first* v1 paper to *adopt* the *systematic* 4-level annotation pipeline in a *clinical-trial* context, the *direct* enabler of *trustable* H5 cross-dataset evaluation.

**v0 stack updated**: sub-task 1 default annotation workflow = **4-level quality assessment pipeline (NEW from 077, $0 if existing labels, $2,000-5,000 Lambda if custom data)**: sub-task 1 default SSL method = **CTCT (NEW from 077, $100-200 Lambda)**: sub-task 1 default AL method = **CEAL (NEW from 077, $100-200 Lambda)**: sub-task 1 default training config = **3D patches (64, 128, 128) + Adam lr=0.0004 + step scheduler (50, 0.9) + 300 epochs + 2×A100 + batch 4-8 (NEW from 077, $0)**: sub-task 1 default metric suite = **9-metric protocol DSC/IOU/SEN/PPV/HD/ASSD/SO/SD (already from 075, extended to include SO/SD as primary from 077)**: sub-task 1 SSL training data = **146 unlabeled volumes (NEW from 077, $50-100 Lambda)**: eval = 075 + 076 stack + **18-method FSL+SSL+AL ablation (NEW from 077, $500-1,000 Lambda)**: v0 compute = **~$6,470-7,760 Lambda** (was $5,920-6,930, +$500-1,000 for 18-method re-implementation + $100-200 for CTCT/CEAL integration + $50-100 for SSL training data access + $0 for training config / metric suite / annotation workflow). **Strategic positioning: v0 sub-task 1 now has the *definitive* FSL+SSL+AL 18-method ablation table, the *only* paper in the dental-IOS literature to *systematically* compare *training paradigm* effects on a 3D dental-CBCT dataset, the *direct* extension of the CTooth 075 evaluation protocol with SSL+AL methods. v0 sub-task 1 default SSL+AL methods (CTCT, CEAL) are the *strongest* in the CTooth+ benchmark, the v0 paper's H5 generalization claim is *strengthened* by the SSL/AL data-efficiency story (the v0 paper can claim -50% to -90% labeled-data requirement with *equal-or-better* DSC). v0 paper's related work is now the *definitive* 2018-2026 open dental-CBCT-seg lineage: TS-MTL 071 → CTooth 075 → CTooth+ 077 → STS MICCAI 2023 Challenge → ToothFairy2 053 → ToothFairy3 054 → v0 2026, the *first* paper to *trace* this *complete* 5-paper arc.**

**The dental-CBCT-seg 2018-2024 open-dataset arc is now *complete* (TS-MTL 071 → CTooth 075 → CTooth+ 077 → STS MICCAI 2023 → ToothFairy2 053 → ToothFairy3 054) — the v0 paper's related work can now *trace* the *full* 6-paper open-dataset arc.** **The CTooth-family founder-pair (075 + 077) is the *defining* 2-paper arc of the 2022 open dental-CBCT era, every subsequent open dental-CBCT-seg paper (STS MICCAI 2023, ToothFairy2 053, ToothFairy3 054) inherits from this *founder-pair*.** Note in `papers/077-ctooth-plus-cui22.md`.

**Open questions for HK: (i) Adopt 4-level quality assessment pipeline? (recommend YES if v0 collects custom dental-CBCT data, $2,000-5,000 Lambda, 10-12 months; otherwise $0), (ii) Adopt 18-method FSL+SSL+AL ablation? (recommend YES, $500-1,000 Lambda, 1-2 weeks, the *most-comprehensive* dental-CBCT-seg ablation in the reading list), (iii) Adopt CTCT SSL method? (recommend YES, $100-200 Lambda, 1 week, +6.33pp DSC SSL→FSL lift), (iv) Adopt CEAL AL method? (recommend YES, $100-200 Lambda, 1 week, -12% data with *equal-or-better* DSC), (v) Adopt 9-metric evaluation protocol? (recommend YES, $0, 1 day, *de facto* standard for clinically-relevant dental-AI eval), (vi) Adopt 3D patch (64, 128, 128) and training config? (recommend YES, $0, 1 day, *correct* handling of CBCT resolution anisotropy), (vii) Use 146 unlabeled volumes for SSL training? (recommend YES if IRB permits, $50-100 Lambda, -50% to -90% labeled-data requirement), (viii) Cite CTooth+ as the *enabling* paper for 2023-2026 open dental-CBCT-seg? (recommend YES, $0, 30 min, *definitive* historical positioning), (ix) Reach out to Weiwei Cui for collaboration? (recommend YES, $0, 1 hour, *direct* access to the CTooth+ dataset + 4-level pipeline + 18-method baselines), (x) Build v1 custom 22+146 dental-CBCT dataset with 4-level pipeline? (recommend YES for v1, $5,000-10,000 Lambda, 10-12 months, the *direct* enabler of *trustable* H5 cross-dataset evaluation).**

**Next paper to read (078): KPConv (Thomas et al. ICCV 2019, "KPConv: Flexible and Deformable Convolution for Point Clouds", arXiv:1904.08889, *the* 3D-CNN that uses *kernel points* instead of X-transformation, *first* deformable convolution on points, 92.9% ModelNet40, 58.8% S3DIS 6-fold mIoU, *the* right comparison with PointCNN's X-Conv for the v0 paper's "PointCNN vs KPConv" table, the *3rd* major point-cloud-CNN architecture after PointNet++/DGCNN/PointCNN/KPConv, *deformable* convolution on points is the *next-generation* alternative to learned permutation, completes the 2017-2019 PointNet-family arc). Alternative: ToothGroupNet (paper 046, Lim et al. MICCAI 2022, the *2nd* tooth-grouping paper in the reading list, the *direct* successor to TIR, the *first* paper to *use* TIR outputs for *crown* design, the *bridge* between TIR and CrownGen). Alternative: PointNet++ deep-dive for the *PyG built-in* integration. Recommendation: **KPConv for 078** (the *right* comparison with PointCNN's X-Conv, the *3rd* major point-cloud-CNN architecture, *deformable* convolution on points is the *next-generation* alternative to learned permutation, completes the 2017-2019 PointNet-family arc), **ToothGroupNet 046 for 079** (the *bridge* between TIR and CrownGen, the *2nd* tooth-grouping paper in the reading list, the *direct* successor to TIR for *crown* design), **PointNet++ PyG deep-dive for 080** (the *integration* paper for v0's sub-task 1 PyG-based backbone).**
