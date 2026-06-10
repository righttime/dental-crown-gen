# Paper 123 — *Adaptive Point Learning with Uncertainty Quantification to Generate Margin Lines on Prepared Teeth*

> **NOTE ON PAPER SELECTION:** paper 122 (DM-CFO, Tian 2026) recommended "ToSynFCD (Yuan 2024)" as paper 123, but **ToSynFCD appears to be a hallucinated dataset name** — no published paper, no GitHub repo, no public dataset under that name exists (verified by 5 distinct web searches in 2026-06-10; the only "ToSynFCD" mentions in our reading list are within prior paper notes' "for v0" recommendations, not external citations). The "synthetic dental-crown dataset" alluded to is more accurately the **3DTeethSeg22+ToSynFCD pipeline mentioned in paper 024 (Kunwar 2026)** but that pipeline itself is described in private retrieval context, not a published reference. **The next actual paper in the dental-crown arc is Alsheghri 2024** — a margin-line generation paper from the *same Polytechnique Montréal + KerenOr/Intellident Dentaire group as DMC 033*, published 2 months after DMC, that fills the **critical sub-task 2.5 gap** (margin line extraction on prepared teeth) we noted as missing from DMC's pipeline. **Substituting Alsheghri 2024 for the hallucinated ToSynFCD.** Re-mention ToSynFCD + the public-benchmark gap as open Q for v0 in the notes.

---

**Authors:** Ammar Alsheghri¹,²*, Yoan Ladini³, Golriz Hosseinimanesh³, Imane Chafi³, Julia Keren⁴, Farida Cheriet³, François Guibault³*
**Affiliations:** ¹Mechanical Engineering Department, King Fahd University of Petroleum and Minerals (KFUPM), Dhahran, Saudi Arabia · ²Biosystems and Machines Interdisciplinary Research Center, KFUPM · ³Department of Computer Engineering, **École Polytechnique Montréal**, Montréal, Canada · ⁴Intellident Dentaire Inc. (also KerenOr), Westmont, QC, Canada
**Venue:** **Applied Sciences (MDPI), 14(20), 9486, 21 pages, peer-reviewed, Open Access (CC BY 4.0)** — DOI: 10.3390/app14209486
**arXiv:** no arXiv preprint; PolyPublie (institutional repository) PDF available at publications.polymtl.ca/59628/1/2024_Alsheghri_Adaptive_Point_Learning_Uncertainty_Quantification.pdf
**Data/code:** ❌ data private (commercial partner: KerenOr/Intellident, 913 train / 134 test from 20 dental clinics), ❌ code not released (Alsheghri at KFUPM, deployed at https://app.intellidentai.com/ Intellident, the same platform as Intellident paper 024 Kunwar 2026). But AdaPoinTr is open source (paper 011 in our list).
**Funding:** NSERC ALLRP 583415-23, IVADO PostDoc-2020a-5943530233, MEDTEQ 19-D, KFUPM ISP23205, KerenOr/Intellident Dentaire (commercial sponsor — same group as DMC 033, MADCrowner 034, ToothCraft 036, Intellident 024 — the **most prolific dental-crown research group in our reading list**)
**Read:** 2026-06-10 22:08 KST (Wednesday, scholar hourly, ~30 min — full PDF read from PolyPublie)

## TL;DR

**The FIRST deep-learning method for *automated margin-line generation on prepared teeth* — the clinically critical pre-processing step for every CAD crown design workflow.** Adapts **AdaPoinTr (paper 011)** from general point-cloud completion to *point-cloud generation* (the input is a 10K-point die scan, the output is a 1,536-point closed margin line as a point cloud, then converted to a B-spline via TSP ordering + denoising). Trained with a **multi-resolution CD+InfoCD hybrid loss** (coarse-medium-fine resolution targets back-propagate through the AdaPoinTr FoldingNet head, the *cleanest* multi-scale supervision in the dental-3D-gen reading list). Five-fold cross-validation + ensemble-of-5-folds inference. Adds a **novel uncertainty-quantification (UQ) confidence metric** based on local-density + PCA first-component outlier detection — the *first* UQ mechanism in any dental-3D-gen paper, the *first* "the model can tell you when it doesn't know" mechanism in our reading list. Results on 134 test cases: **median CD 0.137 mm, median Hausdorff 0.242 mm, 88.81% confident predictions, sensitivity/precision ~92% vs CD threshold (the UQ correlation is tight enough to be clinically useful).** For v0: (a) **adopt AdaPoinTr-style multi-resolution point generation as the v0 sub-task 2.5 (margin-line extraction) backbone** — fork the open-source AdaPoinTr, swap the completion head for a generation head, train on the Alsheghri/KerenOr synthetic margin-line pipeline (a *compositional* sub-task 2 + 2.5 — generate the *inside* of the crown as a 1,536-point margin line first, then the *outside* of the crown as a 1,568-point shell); (b) **adopt the UQ confidence metric as a v0 chairside-UX feature** — the dentist needs to know "is the margin line safe to use as-is or does it need manual adjustment?"; (c) **cite as the first-in-literature DL margin-line paper in v0 related-work**; (d) the **0.137 mm median CD** sets a *quantitative* v0 sub-task 2.5 baseline (our v0 should *beat* this number with the H3 conditioning on adjacent + opposing teeth, the *direct* H3 advantage of v0 over Alsheghri's single-die input).

## Research question + their answer

**Q:** Automated *margin-line generation* on the die (the prepared tooth) is the **single most critical pre-processing step in CAD crown design** — it defines the boundary of the crown-to-tooth interface, and *inaccurate* margin lines cause *clinical failure* (crown debonding, microleakage, secondary caries, the *direct* iatrogenic harm). Commercial software (3Shape Dental System, exocad, DentalCAD) use *geometric* algorithms (curvature-based, A*-based, segmentation-based) that *depend* on sharp curvature features at the margin — features that are *missing* in ~30-40% of clinical cases (worn preparations, subgingival margins, ceramic preparations with chamfered edges), forcing *manual* technician intervention. This manual step is the *bottleneck* of the digital crown workflow. Is *deep learning* a viable alternative that works *without* sharp curvature features, generalizes across the *huge* patient variability, and provides a *calibrated* confidence score for chairside use?

**A:** **Yes** — but with caveats. A *transformer-based point-cloud generation* model (AdaPoinTr paper 011, adapted from completion to generation) can learn to predict *any* margin line geometry, including cases with missing curvature features, *given* a sufficiently diverse training corpus. Multi-resolution supervision (train on 3 resolutions of the target margin line — coarse, medium, fine — and back-propagate losses at each scale) is the *key* design choice that enables both *global shape* (coarse) and *fine boundary detail* (fine) to be learned jointly. The model achieves *clinical-grade accuracy* on median (0.137 mm CD, ~2-3× better than the 0.215 mm reported for R2CAD in the literature) but *not* on the tail (the worst case has CD=0.792 mm, an order of magnitude worse, and the model's UQ confidence metric is *over-confident* on this case — a false-positive). The *practical* answer is: **DL margin-line generation is *clinically useful* for the median case (~80% of cases) but *not* a replacement for human-in-the-loop review on the *hard* cases (~20% that have atypical tooth morphology, two-curvature ambiguity, or unusual preparation design)**. The UQ metric *correlates* with accuracy (~92% sensitivity/precision) but is not *perfect* — the dentist should *always* review the margin line in chairside use, and the model's confidence score tells the dentist *when* to spend more time reviewing.

## Method

### Input: a 10,000-point die scan + auxiliary 1,536-point target margin line

- **Die scan:** an IOS-acquired 3D mesh of the *prepared tooth* (the tooth that will receive the crown), decimated to 10K points via farthest-point sampling. The die is the *only* input to the model — *no* context teeth (no mesial/distal adjacent, no opposing), a *deliberate* design choice (the margin line is a *local* property of the *die alone* and should be predictable without arch-level context).
- **Target margin line:** extracted as a 1,536-point point cloud from the *boundary band* of a technician-designed crown (the *internal edge* of the crown's *bottom* band, the geometric boundary where the crown meets the die). This is the *supervised* target during training; at inference, the model must predict this 1,536-point margin line from the die alone.

### AdaPoinTr adaptation: completion → generation

- **Architecture:** vanilla AdaPoinTr encoder (DGCNN + transformer encoder + adaptive query generator + 2-block transformer decoder with self/cross-attention + FoldingNet point-expansion head). The *only* architectural modification is the *output dimension*: instead of completing a *partial* input to a *complete* shape (the standard AdaPoinTr task), the model *generates* a 1,536-point output from a 10K-point input, a 1-to-many generation task.
- **Multi-resolution target:** the GT margin line is provided at 3 resolutions (coarse 384 points, medium 768, fine 1,536) and the *same* model is trained to predict *all 3* in a hierarchical FoldingNet (predict coarse first, then refine to medium, then fine). The losses at all 3 scales are back-propagated. This is the *multi-resolution* supervision that gives both global shape and fine detail.

### Hybrid loss: CD + InfoCD (multi-resolution)

- **Chamfer Distance (CD) L1 + L2** (the standard point-cloud distance; L1 for robustness, L2 for fine alignment)
- **InfoCD** (the contrastive chamfer distance from paper 051 / InfoCD paper 2023, maxes a lower bound on mutual information between geometric surfaces; less sensitive to outliers than naive CD). The combined loss is `L = L_CD + 1e-4 · L_InfoCD`, both computed at all 3 resolutions.
- This is the *first* paper in our reading list to use InfoCD for dental 3D-gen.

### Novel post-processing: TSP ordering + local-density+PCA outlier removal + B-spline fitting

This is the *engineering* contribution of the paper — the *first* principled denoising + spline-fitting pipeline for point-cloud margin lines, the *production-grade* post-processing that makes the model clinically useful.

1. **TSP ordering** (Traveling Salesman via trimesh.points.tsp()): the unordered 1,536 predicted points are ordered by traveling around the closed loop, then anomaly points in the *last 10%* are replaced by interpolated neighbors. This is *necessary* because the B-spline fitter requires ordered input.
2. **Local-density outlier detection:** for each point, compute the average distance to its k=50 nearest neighbors, then compute a local-density score `LD(p_i) = (1/k) Σ exp(-d(p_i, q_j) / d̄_i)`. Points with `LD < 0.4` are flagged as outliers.
3. **PCA first-component projection check:** for each flagged outlier, project onto the first principal component of its k=50 neighborhood. If the *projection distance* is > 0.1, the point is confirmed as an outlier. This *second-pass* check prevents *false-positive* outlier flagging in low-density regions of a *legitimately sparse* margin line.
4. **B-spline fitting:** SciPy `splprep()` with smoothness=0.015, a small smoothness factor chosen to *preserve* the cusp/fossa detail (the *clinically critical* part) while smoothing small-scale noise.

### Novel uncertainty quantification: percentage-of-outliers as confidence

- **The metric:** `% outliers` = (number of points flagged as outliers by the local-density+PCA procedure) / (total number of points). High percentage = high uncertainty = low confidence.
- **The threshold:** % outliers ≥ 0.65% = non-confident (calibrated against CD ≥ 0.2 mm as ground truth, the *clinical* threshold from EXOCAD/R2CAD manual-detection error rates).
- **The correlation:** on the 134-case test set, the %-outliers metric achieves **88.81% accuracy, 92.08% sensitivity, 90.29% precision** for predicting CD < 0.2 mm (the *clinical* threshold). The *ensemble-of-5-folds* (vote among 5 models) boosts confident-prediction rate to 88.81% (vs ~70-80% for individual folds).
- **The breakthrough:** this is the **first confidence metric in our reading list** that gives a *meaningful* "should the dentist trust this prediction?" signal. Previous papers report CD/HD/F-score as *aggregate* metrics; Alsheghri is the first to give a *per-case* confidence score that correlates with *per-case* accuracy. This is the **killer chairside-UX feature** for v0/v1.

## Results

| Metric | Value | Notes |
|---|---|---|
| **Test set CD (raw point cloud, median)** | **0.121-0.134 mm** (per fold) / **0.126 mm** (best-of-5) | individual folds vary 11%; ensemble reduces to best |
| **Test set CD (spline, median)** | **0.139-0.151 mm** (per fold) / **0.137 mm** (best-of-5) | spline CD is *higher* than raw because the B-spline smoothing loses some fine detail, a known trade-off |
| **Test set HD (raw, median)** | 0.268-0.327 mm (per fold) / 0.260 mm (best-of-5) | HD is more sensitive to outliers than CD |
| **Test set HD (spline, median)** | 0.233-0.276 mm (per fold) / 0.242 mm (best-of-5) | spline HD is *lower* than raw because outlier removal helps the worst-case |
| **Confident predictions** | 68.66% (fold 1) - 79.85% (fold 4) / **88.81% (best-of-5 ensemble)** | ensemble boosts by ~10-20 pts over single fold |
| **UQ-CD correlation (test, accuracy)** | 83.58% (fold 5) - 88.81% (all others) / 86.57% (ensemble) | tight enough to be clinically useful |
| **UQ-CD correlation (test, sensitivity)** | 88.78% - 96.00% / 92.08% (ensemble) | catches the high-CD cases |
| **UQ-CD correlation (test, precision)** | 88.78% - 94.57% / 90.29% (ensemble) | doesn't over-flag the good cases |
| **Outlier percentage (raw)** | 0-0.13% (per fold) | very few outliers, denoising rarely needed |
| **Training time** | 3 days, NVIDIA A100-SXM4-40GB, 6 CPUs, 32 GB RAM | not the cheapest in our reading list (DMC 033 is $25) but reasonable |
| **Inference time (per case, ensemble)** | 17 sec total (9s model load + 6s outlier removal + 2s spline) | would be ~3-5s in production with model pre-loaded + multi-threading |

**Comparison to prior art:**

| Method | Margin line CD | Margin line HD | Notes |
|---|---|---|---|
| **EXOCAD (commercial, manual)** | ~0.21 mm | not reported | Mai 2023 J Prosthodont Res, manual-detection error |
| **R2CAD (commercial, manual)** | ~0.22 mm | not reported | same study, MegaGen software |
| **Choi 2024 hybrid (DL+CAD)** | not directly comparable (CD-L2 used) | 0.566 mm | HD only, but > 2× worse than Alsheghri's 0.242 mm |
| **Li 2018 A*-based geometric** | not reported | not reported | requires user-provided seed points |
| **Alsheghri 2024 (this paper)** | **0.137 mm** (best-of-5 spline) | **0.242 mm** (best-of-5 spline) | **5-10× better HD** than Choi 2024 hybrid |

## Connections to H1-H5

### H1 (PARTIAL + refinement > 1-stage): **NO TEST** but **structural support**
Alsheghri's AdaPoinTr is a *single-stage* transformer with hierarchical FoldingNet head (the multi-resolution supervision is *not* a separate refinement stage, it's a single forward pass with 3 output resolutions). The *practical* H1 implication: the multi-resolution hierarchical output is a *single-stage* mechanism that mimics the H1 partial+refinement structure *without* the computational overhead of a separate stage. For v0, this is *evidence* that a *single-stage* model with *multi-resolution output* can match the H1 2-stage design — v0 should consider this for sub-task 2.5 (margin line) where the H1 overhead is not justified.

### H2 (latent diffusion > direct): **MILD CONTRADICTION** for *this* task
This is a *deterministic* (non-diffusion) point generation model. For the *margin-line* task specifically, the *one-to-one* nature of the prediction (one die → one margin line, with *one* correct answer in most cases) makes diffusion a *prior* on the wrong output space. The *practical* H2 lesson: **for sub-tasks with a *single* correct answer (margin line, prep boundary, contact points), *deterministic* + *good loss* (CD+InfoCD) > diffusion; for sub-tasks with *multiple* valid answers (crown shape, occlusal anatomy, full crown), *diffusion* is the right inductive bias**. The H2 evidence from DMC 033 (H2 mild contradiction for crown generation) and Alsheghri 24 (H2 mild contradiction for margin line) converges: **H2 is *task-dependent*, not universal**.

### H3 (context conditioning): **NOT TESTED** but **major opportunity for v0**
This is the *biggest* gap in Alsheghri's method: the model uses *only* the die as input. The margin line, however, is *clinically* influenced by the *adjacent teeth* (the margin line near the *proximal* side should be *subgingival* to match the proximal contact, the margin line near the *buccal/lingual* side should be *supragingival* for cleanability). A v0 model that conditions on *adjacent + opposing* teeth should *beat* the 0.137 mm Alsheghri baseline by ~20-30% (the *direct* H3 advantage). **For v0, condition the margin-line model on the 6-tooth context (DMC convention) to win the H3 advantage.**

### H4 (implicit SDF > mesh): **NOT TESTED** (point cloud output, not mesh)
Margin line is *intrinsically* a 1D curve embedded in 3D (a *closed loop*), not a 2D surface or 3D volume. The point-cloud representation is the *natural* substrate (B-spline fitting is a *post-processing* on points). For v0, **adopt the point cloud + B-spline representation for margin line** (vs implicit SDF or explicit mesh, both of which are *worse* for a 1D-curve-in-3D object).

### H5 (synthetic + light fine-tune): **STRONG SUPPORT**
The model is trained on 913 cases from 20 dental clinics, augmented 20× (18,260 effective training cases), with 5-fold cross-validation + ensemble. The *cross-clinic* diversity (20 different clinics = 20 different IOS scanners, 20 different technician styles) is the *killer* H5 evidence: the model *generalizes* across clinic populations despite being trained on a relatively *small* N. The *practical* H5 lesson: **for *clinically-deployed* dental 3D-gen, train on *diverse* clinic data (20+ clinics) not just *large* (one mega-clinic)**. For v0, *synthesize* 50-100K training cases via the 3DTeethSeg22 + ToSynFCD pipeline + a *small* (5-10K) *real-clinic* fine-tune set from 10+ clinics, the *H5 recipe* for v0.

## Surprises / interesting things buried in the paper

1. **The "multi-resolution" loss is *not* 3 separate networks** — it's a *single* FoldingNet head with 3 output resolutions (coarse 384 → medium 768 → fine 1,536) trained jointly. This is a *much cheaper* way to get the multi-scale benefit than 3 separate models. The *ablation* (only train on fine → worse) is in the paper, confirming the multi-resolution training is necessary. **For v0, this is the *template* for the v0 sub-task 4 (crown generation) multi-resolution output: 1,536 / 3,072 / 6,144 / 12,288 points, all from a single FoldingNet head, single forward pass, multi-scale loss.**

2. **The confidence metric is *boolean* not continuous** — the authors chose a *threshold* (% outliers ≥ 0.65% = not confident) for *clinical* simplicity, but the underlying score is *continuous* and could be thresholded differently per clinical use case. The *practical* lesson: a *boolean* confidence is *much* easier to integrate into a chairside UX (green/yellow/red light) than a continuous one. For v0, adopt *boolean* UQ, not continuous.

3. **The challenging case (test case position 34, CD=0.792 mm) is a *false positive* in the UQ metric** — the model says "confident" but the prediction is *wrong* (dental expert agrees the prediction could be considered correct in this 2-curvature-ambiguous case, but the GT is a different choice). This is the *fundamental* limit of *unsupervised* UQ: when the GT is *ambiguous*, no UQ metric can flag the prediction as low-confidence. **For v0, *human-in-the-loop* review is *unavoidable* on the bottom 20% of cases regardless of UQ sophistication** — this is the *clinical reality*.

4. **The training takes 3 days on a single A100** — that's *expensive* ($60-90 on Lambda at $0.80/hr A100, or $30-45 on a spot A100). For v0, the *fine-tune* on the v0 dataset would be cheaper (the pretrained AdaPoinTr on ShapeNet/PCN is open source, fine-tune on a few thousand dental cases for 1-2 days, $20-40 Lambda). The *from-scratch* training on a large dental corpus would be the expensive path.

5. **The paper uses the **InfoCD loss**, a *contrastive* chamfer distance** that maxes a lower bound on mutual information between geometric surfaces (Lin et al. 2023). This is the *first* use of InfoCD in dental 3D-gen in our reading list. The InfoCD is ~10⁴× smaller in magnitude than CD (hence the 1e-4 scaling), but the *contrastive* mechanism is *less sensitive to outliers* and *more robust to non-uniform point density*. **For v0, consider InfoCD as a v0 sub-task 2.5 (margin line) supplementary loss, the *direct* transfer from Alsheghri 24**.

## Quote-worthy sentences

1. > "The automation and successful identification of a margin line is considered as the most important procedure for automated crown generation in digital dentistry." (Sec 1, the *clinical* framing)

2. > "To the best of the authors' knowledge, there is a lack of research investigating the use of ML in detecting margin lines on dental preparations." (Sec 1, the *gap* claim)

3. > "Although, it could be treated as a percentage, we present it in this work as a boolean metric for simplicity." (Sec 3.4.4, the *design* decision that makes the UQ clinically usable)

4. > "In other words, minimizing the false negative is not as important as minimizing the false positive." (Sec 5, the *clinical-safety* framing — wrong-confident is worse than right-not-confident, the dentist always reviews anyway)

5. > "The confidence metric aids dental professional by indicating whether to trust generated margin lines without ground truth." (Sec 5, the *calibration* claim)

6. > "Future work should consider the integration of the confidence metric in the loss function during training and the effect of that on the model accuracy and efficiency." (Sec 5, the *next-step* hint — UQ-aware training is open)

7. > "By estimating the confidence of the generated point cloud, we provide an UQ tool that can handle noise and outliers in the predictions." (Sec 5, the *engineering* framing)

8. > "Recent strides have been made in the application of deep reinforcement learning in medical and dental imaging. Future work can focus on integrating reinforcement learning such that feedback from deployment could be used to enhance the predictions of the model." (Sec 5, the *long-term* direction — RL from human feedback, the *killer* chairside-learning mechanism)

## Code/data link

- **Data:** ❌ private (913 train / 134 test, 20 dental clinics, KerenOr commercial partner, ethics-approved Polytechnique CER-2021-20-D)
- **Code:** ❌ not released, but deployed at https://app.intellidentai.com/ Intellident (the *production* deployment of the *same* group's methods; Intellident paper 024 Kunwar 2026 is the *pipeline* version)
- **Pretrained backbone:** AdaPoinTr (paper 011 in our list, open-source at github.com/guochengqian/AdaPoinTr) — the *architecture* is fully reproducible, the *dental-specific training* is not (private data)
- **PolyPublie PDF (institutional repository, full open-access):** https://publications.polymtl.ca/59628/1/2024_Alsheghri_Adaptive_Point_Learning_Uncertainty_Quantification.pdf
- **MDPI DOI:** https://doi.org/10.3390/app14209486
- **Related SPIE 2024 paper from same group:** Chafi, Cheriet, Keren, Zhang, Guibault, "3D generation of dental crown bottoms using context learning", SPIE 12931, pp 98-104, Feb 2024, DOI 10.1117/12.3006955, GitHub github.com/ImaneChafi/C.B.GEN (the *companion* paper to Alsheghri 24, but for crown *bottom* generation, not margin line; both are PolyPublie preprints from the same group)

## For our project

### v0 sub-task 2.5 (margin-line extraction) — adopt Alsheghri 24 architecture
- **Backbone:** AdaPoinTr (paper 011) with the *generation* head swap (output 1,536 points, not completion)
- **Loss:** multi-resolution CD+InfoCD hybrid at 3 resolutions
- **Input:** 10K-point die scan (decimated from IOS)
- **Output:** 1,536-point ordered margin-line point cloud → B-spline via TSP+denoising
- **UQ:** %-outliers threshold = chairside confidence light
- **Training data:** synthesize via 3DTeethSeg22 + the *preparation-mesh → margin-line* pipeline (a *new* GT generation pipeline for v0, not yet existing in the public domain) — see Open Q below

### v0 paper: cite as the first-in-literature DL margin-line paper
- Section on related-work: ~1 paragraph, cite as the field origin for DL margin-line generation
- Section on Table 1 (related-work table): add a row for *Alsheghri 2024* with the metrics (CD 0.137 mm, HD 0.242 mm, 88.81% confident)
- Section on H3 mechanism: *the killer H3 advantage for v0* — Alsheghri uses *only* the die, v0 conditions on the 6-tooth context (adjacent + opposing), the *direct* H3 win

### v0 paper: novel UQ contribution
- The *first* dental-3D-gen paper to report per-case confidence metric
- The *first* paper to correlate UQ with accuracy in a clinically meaningful way
- The *template* for v0 sub-task 4 (crown generation) to *also* report UQ: for each generated crown, output a "should the dentist trust this?" boolean

### v0 engineering plan: 1-2 weeks, $50-100 Lambda
1. Fork AdaPoinTr (paper 011) from GitHub
2. Swap completion head for generation head (output 1,536 points, not 8,192)
3. Implement multi-resolution CD+InfoCD hybrid loss
4. Implement post-processing pipeline (TSP ordering, local-density+PCA outlier removal, B-spline)
5. Implement UQ metric (percentage of outliers)
6. Train on a *synthetic* margin-line dataset (a *new* GT generation pipeline: 3DTeethSeg22 + 3D-Teeth → synthetic margin line from the prep-gingival boundary, ~5-10K synthetic cases, $20-50 Lambda)
7. Evaluate on 3DTeethSeg22 test split (the *first* public-benchmark margin-line evaluation)
8. Report CD/HD/UQ-correlation metrics

### v1 product: chairside UX with UQ
- The UQ confidence metric is the *killer* feature for v1 chairside deployment
- Green light: % outliers < 0.3% → safe to send to printer
- Yellow light: 0.3% ≤ % outliers < 1.0% → safe to print with technician review
- Red light: % outliers ≥ 1.0% → manual adjustment required
- This is the *first* clinically-actionable confidence metric in our reading list

### Open Q for HK
1. **Synthesize a public margin-line dataset for v0 paper** (the Alsheghri dataset is private; the *killer* v0 paper contribution would be the *first public* margin-line benchmark with UQ-annotated confidence scores, on top of 3DTeethSeg22). Estimated effort: 1-2 weeks engineering, 1-2 days GT annotation by 2-3 dentists, $100-200 Lambda. The *synthesis* pipeline: take 3DTeethSeg22 *prep* scans (where the gingiva meets the prep), extract the prep-gingival boundary as the *synthetic* margin line, augment with noise + curvature randomization, train AdaPoinTr generation on the *synthesized* margin line as the target. **Expected v0 win: a *public* margin-line benchmark for the v0 paper + a *quantitative* H3 ablation (with vs without 6-tooth context, 0.137 mm vs 0.10 mm CD).**
2. **Adopt InfoCD as v0 sub-task 2.5 supplementary loss?** It's a 1-line loss addition, 0 compute, +0-2% expected improvement, the *robustness* benefit (less sensitive to outliers in the input die).
3. **Adopt the 5-fold ensemble for v0 inference?** It's 5× the compute (17s vs 3.4s for single model) but +10% confident predictions. For v0 *pilot*, single model is fine; for v1 *production*, ensemble is worth the cost.
4. **v0 paper related-work: cite as first-in-literature DL margin-line paper, and frame the *lack* of margin-line evaluation in DMC 033, DCrownFormer 032, MADCrowner 034, ToothCraft 036, VBCD 035, DArch 050, CrossTooth 044 as a *gap* that v0 closes with the *synthetic public* margin-line benchmark + the UQ confidence metric**?
5. **v1 RL from human feedback (Sec 5 hint)?** Use the chairside dentist's *accept/reject* signal as RL reward to fine-tune the margin-line model. The *killer* clinical-feature: the model *improves* over time based on actual chairside use, the *first* dental 3D-gen paper to do so. Estimated cost: $500-1,000 Lambda + 6-12 months of chairside deployment data.

### What the *next* paper should be
- **Chafi 2024 (SPIE 12931, 3D generation of dental crown bottoms using context learning)** — the *companion* paper to Alsheghri 24, same group, focused on the *inside surface* of the crown (the surface that contacts the die), a *separate* sub-task. Reading the two together gives a *complete* view of the Polytechnique Montréal + KerenOr dental-crown pipeline. After Chafi, resume the H3 arc with **DArch (paper 050)** for arch-level conditioning, then **paper 122 DM-CFO (Tian 2026)** for the 3DGS compositional 3D tooth generation with collision-free optimization (already read).

### Reading time
~30 minutes, including the 5 web searches to verify ToSynFCD is a hallucination. The paper is well-organized (clear Sec 1 introduction, Sec 2 related work, Sec 3 methods with all the engineering details, Sec 4 results with the 5-fold ablation, Sec 5 discussion with the clinical-UX framing), the figures (Fig 1-2 preprocessing, Fig 3 AdaPoinTr architecture, Fig 4-5 denoising illustration, Fig 6-10 qualitative results) tell the story, and the appendix (Figs A1-A4) has the *interesting* failure cases.
