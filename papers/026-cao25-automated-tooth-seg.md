# Paper 026 — *Fully Automated Tooth Segmentation and Labeling for Both Full- and Partial-Arch Intraoral Scans Using Deep Learning*

**Authors:** Lingyun Cao, Niels van Nistelrooij, Shankeeth Vinayahalingam
**Affiliations:** Radboud University Medical Center (Radboudumc), Nijmegen, the Netherlands — Department of Oral and Maxillofacial Surgery (Cao, van Nistelrooij, Vinayahalingam); also affiliated with Ardim B.V. (van Nistelrooij + Vinayahalingam co-founders, AI-ultrasound for hip dysplasia — no role in study)
**Venue:** *International Dental Journal* (Elsevier, open-access) 75(5):100950, **published 2025-08-14**
**DOI:** [10.1016/j.identj.2025.100950](https://doi.org/10.1016/j.identj.2025.100950) · PMID 40815915 · PMCID PMC12392768
**License:** CC BY 4.0 (Elsevier OA)
**Code:** No dedicated repo located at conventional paths; ToothInstanceNet (the base model) has its own public code from the same lab
**Citations:** New (Aug 2025) — citation count not yet meaningful; directly cited by paper 025 (ArchSeg) in the *opposite* direction (Cao is the new SoTA, ArchSeg 025 is the prior SoTA it beats)
**Clinical context:** 600 IOSs (300 full + 300 partial) from 300 patients in a private clinic in **Hubei, China**, 3Shape Trios Move / 3Shape D500 scanners, PLY format, paired upper+lower scans per patient

---

## TL;DR

**The new SoTA in 2025 dental tooth segmentation + FDI labeling, and the direct full-automation successor to paper 025 (ArchSeg 2024)** — wraps ToothInstanceNet (the same lab's prior two-stage DentalNet+refinement model) with four targeted enhancements and posts a near-perfect F1 0.9908 on full-arch IOSs and F1 0.9884 on partial-arch IOSs, plus a **0.9870 overall score on the 3DTeethSeg challenge** (the public benchmark from paper 001), beating every prior method reported in Rekik et al. The big idea: **the right combination of 4 cheap tricks beats a new backbone** — artificial-partial-arch data augmentation, DL-based canonical-orientation alignment, FDI-pair-offset postprocessing, and 300 real partial-arch scans, each of which adds 0.005-0.05 to the right metric. Paper 025's manual two-label user input is fully replaced.

## Research question

> "Can a *single* deep-learning model perform tooth segmentation + FDI labeling on *both* full-arch and partial-arch intraoral scans at clinical-grade accuracy (F1 > 0.98) without any user-supplied labels, when every prior method either fails on partials (paper 023 MeshSegNet, paper 024 Kunwar, paper 025 ArchSeg) or requires manual inputs (paper 025 ArchSeg's two-label input)?"

## Their answer

**Yes, with four targeted enhancements layered on top of ToothInstanceNet** (the same lab's two-stage model from prior work — low-res instance segmentation + high-res single-tooth refinement):

1. **Artificial partial-arch data augmentation** (2-12 consecutive teeth cropped from full-arch scans via oriented bounding box, 90% probability, skewed distribution favoring fewer teeth) — trains the model to see partials even when only full-arch data is available.
2. **DL-based alignment module** (Stratified Transformer backbone, predicts canonical up + forward directions + a translation vector from a PCA-rough-aligned scan) — replaces paper 025's manual two-label registration with a learned pose normalizer that works on both full and partial arches, including single-quadrant partials.
3. **FDI-aware postprocessing** (multivariate-Gaussian distribution over FDI tooth-pair offsets, MAP assignment over the 16×16=256 possible tooth-pair candidates per arch) — enforces global anatomical consistency on top of the per-tooth softmax predictions, fixing "duplicate or skipped FDI labels".
4. **Real partial-arch IOSs in training** (300 real partials from 300 patients, 5-fold cross-validation) — closes the synthetic-to-real gap that augmentation alone cannot.

The ablation (5 models, each adding one enhancement) shows the *biggest single jump comes from artificial-partial augmentation* (Model 2 partial-arch macro-F1 0.7823 → 0.9788, a +0.197 leap), with each subsequent enhancement adding smaller but still meaningful gains.

## Method

### Data
- **600 IOSs from 300 patients** (300 full + 300 partial) at a private clinic in Hubei, China, scanned with 3Shape Trios Move or 3Shape D500, exported as PLY without color.
- **Inclusion:** paired upper+lower scans, diverse dental conditions (missing, prepared, implants, ortho appliances, residual roots, residual crowns, partially erupted).
- **Exclusion:** primary/mixed dentition, no paired upper+lower, poor-quality stitching, FDI-unlabelable scans, duplicate/longitudinal.
- **Annotation:** 2× by a single dentist (LC, 5y experience), 3-month interval, discrepancies resolved with a second reviewer (NvN) — 3DMedX (3DLab) point-wise annotation tool.
- **Teeth3DS / 3DTeethSeg challenge dataset** (paper 001, 1800 scans from 900 patients) used for external validation with 1200 train / 600 test split.

### Base model: ToothInstanceNet (Vinayahalingam et al., prior work)
- **Stage 1** (low-resolution): DentalNet-inspired instance segmentation + FDI label prediction on a downsampled point cloud.
- **Stage 2** (high-resolution): per-tooth crops → fine segmentation refinement on dense points.
- ToothInstanceNet on partial-arch (their own ablation, Model 1): macro-F1 0.7823, macro-IoU 0.6537 — i.e., *on partials, the base model is essentially broken*; this is the gap the four enhancements close.

### Enhancement 1: Artificial partial-arch IOS augmentation
- For each full-arch scan in training:
  1. Determine the FDI-ordered sequence of teeth
  2. Randomly select **2-12 consecutive teeth** (skewed distribution favoring fewer teeth)
  3. Crop the OBB around the selected teeth
  4. Keep the largest connected surface inside the OBB
  5. Translate to center at origin
- **Applied with 90% probability** during training.
- This is a *synthetic partial* but the topology and occlusal surface are preserved because the OBB cuts off cleanly at the OBB boundary.
- Reference 27 (Jana et al.) is the precedent — but Jana only *tested* on artificial partials, not *trained* on them. Cao *trains* on them, which is the H5 trick.

### Enhancement 2: DL-based alignment module
- **Problem:** PCA on partial arches fails (the per-tooth principal axes that PCA exploits are not stable when only 2-12 teeth are present).
- **Solution:** A learned alignment model based on **Stratified Transformer** (point transformer with stratified sampling for long-range context, reference 30).
- **Inputs:** a PCA-roughly-aligned IOS (gives a rough starting orientation, but not the canonical one).
- **Outputs (3 heads, all supervised):**
  - `d̂_forward ∈ R³` — canonical forward direction (e.g., posterior → anterior)
  - `d̂_up ∈ R³` — canonical up direction (e.g., occlusal → gingival)
  - `ĉ ∈ R³` — translation vector (needed for left-right disambiguation of partial-arch scans with only posterior teeth)
  - Auxiliary: `ŝ ∈ R^N` — binary point-wise tooth/gingiva segmentation (auxiliary supervision, helps the alignment features learn what is "tooth" vs "soft tissue")
- **Loss:** `L_align = L_orient + Smooth-L1(c', ĉ) + BCE(s', ŝ) + Dice(s', ŝ)`
  - `L_orient = 2 - d'_forward · d̂_forward / |d̂_forward| - d'_up · d̂_up / |d̂_up| + |d̂_forward · d̂_up| / (|d̂_forward| |d̂_up|)` — cosine angles + orthogonality constraint.
- **Training data for the alignment module:** 234 full-arch IOSs with both first molars and both central incisors (so the canonical up/forward can be derived from anatomical landmarks: 2 first-molar centroids + central-incisor centroid centroid → occlusal plane).
- **Inference:** apply the predicted translation in reverse, orthonormalize the up/forward/left-right basis, rotate the scan to canonical orientation before passing to ToothInstanceNet.
- **Why it's better than paper 025 ArchSeg:** ArchSeg's registration is a heuristic alignment that needs the user to supply the first/last FDI tooth labels. Cao's alignment is fully learned, takes only the scan, and handles the left-right disambiguation via the translation vector.
- **Why it's better than Zhuang et al. (ref 29):** Zhuang's method assumes *global anatomical completeness* (works on full arches only, can't tell left from right on a single posterior quadrant). Cao's translation head fixes this.

### Enhancement 3: FDI-aware postprocessing
- **Problem:** Per-tooth softmax predictions are *locally* correct (each tooth is confidently the right class), but the *global sequence* can have duplicates or skips (e.g., two teeth both predicted as FDI 16, or FDI 14 missing).
- **Solution:** Model the FDI-to-FDI offset as a multivariate Gaussian `p(o^[i,j]) ~ N(μ, Σ)`, where `o^[i,j]` is the relative offset between the i-th and j-th teeth in the predicted sequence.
- **Inference:** For each predicted tooth pair (i, j), compute the probability density of the offset for all 16×16 = 256 possible FDI candidates per arch, convert to negative-log costs, run **MAP** (i.e., dynamic programming / Viterbi) to find the globally consistent FDI labeling.
- **Effect:** Fixes duplicate/skipped labels → big jump in macro-IoU (Model 3 → Model 4 macro-IoU on partial: 0.9206 → 0.9280, +0.0074; the paper attributes this *entirely* to postprocessing since all other components are identical between the two models).

### Enhancement 4: Real partial-arch IOSs in training
- 300 real partial-arch IOSs from 300 patients, 5-fold cross-validation.
- The paper notes that *artificial* partial-arch augmentation (Enhancement 1) gives most of the gain (Model 2 vs Model 1: macro-F1 +0.197 on partials), but *real* partials still add a small but meaningful gain at the long tail (the "diverse dental conditions" like residual roots, residual crowns, partially erupted teeth that augmentation cannot synthesize).

### Training details
- **Architecture:** ToothInstanceNet for segmentation, Stratified Transformer for alignment, both implemented in PyTorch Lightning v2.3.3.
- **Alignment model:** batch size 8, base LR 0.001, 1000 epochs.
- **Segmentation model:** training params inherited from prior ToothInstanceNet paper.
- **5-fold cross-validation** on the 600 scans.
- **3DTeethSeg external test:** Teeth3DS 1200 train / 600 test, evaluated with TLA (teeth localization accuracy), TSA (F1 of all annotated/predicted tooth points), TIR (percentage of annotated teeth with correct FDI label).

## Results

### Internal data (5-fold CV, 600 IOSs)

**Table — Full-arch IOSs (higher is better; best in bold)**

| Model | F1-score | Tooth Dice | macro-F1 | macro-IoU |
|---|---|---|---|---|
| 1 (baseline ToothInstanceNet) | **0.9935** | 0.9816 | 0.9893 | 0.9331 |
| 2 (+ artificial partial-aug) | 0.9911 | 0.9806 | 0.9788 | 0.9053 |
| 3 (+ alignment) | 0.9899 | 0.9818 | 0.9908 | 0.9350 |
| 4 (+ FDI postprocessing) | 0.9902 | 0.9818 | 0.9926 | — |
| 5 (+ real partials) | 0.9908 | **0.9819** | **0.9940** | **0.9403** |

**Table — Partial-arch IOSs**

| Model | F1-score | Tooth Dice | macro-F1 | macro-IoU |
|---|---|---|---|---|
| 1 (baseline) | 0.9751 | 0.9810 | **0.7823** | 0.6537 |
| 2 (+ artificial partial-aug) | 0.9828 | 0.9812 | 0.9788 | — |
| 3 (+ alignment) | 0.9865 | 0.9856 | — | 0.9206 |
| 4 (+ FDI postprocessing) | 0.9865 | 0.9857 | — | — |
| 5 (+ real partials, **final**) | **0.9884** | **0.9862** | **0.9786** | **0.9280** |

**Three big findings:**
1. **The Model 1 → Model 2 jump on partial-arch is the entire paper's story in one number** — adding artificial-partial-augmentation alone takes partial-arch macro-F1 from 0.7823 to 0.9788 (+0.197, +25%). No other enhancement comes close to that magnitude.
2. **Adding artificial partials slightly *hurts* full-arch performance** (Model 1 F1 0.9935 → Model 2 F1 0.9911, −0.0024; macro-IoU 0.9331 → 0.9053, −0.0278). The paper's explanation: "the introduction of cropped scans shifted the model's focus towards learning localized features. Consequently, its capacity to make use of long-range spatial relationships was slightly reduced." This is a real, underappreciated trade-off — augmentation helps the missing distribution but can hurt the present distribution. The paper recovers it in later enhancements (alignment + postprocessing + real partials), but the recovered macro-IoU is 0.9403, only +0.0072 above the baseline.
3. **The final model is the best on every metric on partial-arch** and the best on 3/4 metrics on full-arch (the F1 leader stays Model 1, but the difference is 0.0027 — within noise).

### External data (3DTeethSeg / Teeth3DS, 600 test scans)

| Method | TLA | TSA | TIR | **Score** |
|---|---|---|---|---|
| TSegNet | — | — | — | 0.9734 |
| MeshSegNet | — | — | — | 0.9707 |
| RHL | — | — | — | 0.9845 |
| DTSegNet | — | — | — | 0.9817 |
| TSegLab | — | — | — | 0.9761 |
| **Cao 2025 (this paper)** | **0.9945** | **0.9862** | **0.9803** | **0.9870** |

Cao wins on TLA, TSA, TIR, and the average score — i.e., beats every prior SoTA on the public benchmark. Note: the individual SoTA numbers are from the 3DTeethSeg challenge report (Rekik et al. 2023, ref 36) and were reported *by the challenge*, so the comparison is fair.

**On the full Teeth3DS** (Cao's own eval, 1200 train / 600 test): precision 0.9933, sensitivity 0.9994, F1 0.9963, Tooth Dice 0.9823, macro-F1 0.9676, macro-IoU 0.9029. This is *higher* than the 5-fold CV F1 (0.9963 vs 0.9908), which is unusual — likely because the public test set is cleaner than their private data (less diverse conditions).

### Headline comparison to paper 025 (ArchSeg)

- **ArchSeg 025 (Alsheghri 2024):** tooth Dice 0.936 (mandible) / 0.948 (maxilla) on imperfect-arch test sets, **manual two-label user input** required.
- **Cao 2025 Model 5:** tooth Dice 0.9862 on partial-arch, **fully automatic**.
- **Improvement:** +0.038 to +0.050 in tooth Dice, plus the manual-label input is removed.

### Failure mode analysis (Section 4, Fig. 6-7)

The paper does a Kendall's τ-b correlation between dental conditions (missing teeth, prepared teeth, implants, ortho appliances, residual roots, residual crowns, partially erupted, total conditions) and prediction errors (FP, FN, wrong label, total errors). All four clinical conditions most strongly correlated with errors are:
- **Residual roots** (strongest signal)
- **Residual crowns**
- **Missing teeth**
- **Partially erupted teeth**

All p < 0.05 (some p < 0.001). **τ values are small** (the paper explicitly cautions "these associations should be interpreted as trends rather than strong predictors") but the *direction* is clear: these are the conditions that need manual review. The paper suggests a **"condition-aware flagging" UX** — if the system detects these conditions, route the scan to manual review.

## Connections to H1–H5

### H1 (2-stage VAE+DDM > 1-stage) — **STRONG SUPPORT, REFINED**
Cao's pipeline is **3-stage (alignment → instance segmentation + FDI labeling → FDI-aware postprocessing) + augmentation trick** and the ablation explicitly shows each stage adds value. The headline finding is that the *data augmentation* (artificial partials) is more impactful than any individual architectural change — a sharp rebuke to "throw a bigger model at it" thinking. H1's claim generalizes cleanly to segmentation: **decompose into modular stages, each of which can be independently improved**. The FDI-aware postprocessing being a dynamic-programming MAP over a Gaussian prior is a beautiful example of H1: the prior is cheap, the inference is exact, and the lift is free.

### H2 (latent diffusion > direct) — **N/A**
Segmentation only; no generative model, no diffusion. H2 doesn't apply.

### H3 (conditioning on adjacent+opposing teeth is the right mechanism) — **STRONG SUPPORT, NEW FORM**
This is the most interesting connection. Cao's **FDI-aware postprocessing** is **literally an H3 mechanism**: the global prior on the FDI tooth-pair offset distribution is a *learned* global context that constrains per-tooth local predictions. The 16×16 = 256 possible FDI pair candidates per arch form a *structured prior* over the global tooth arrangement, enforced via Viterbi-style MAP inference. This is the H3 mechanism for *segmentation* — for the segmentation sub-task, "conditioning on adjacent teeth" is not via cross-attention, it's via a *combinatorial dynamic-programming prior* over the FDI labeling.

**The artificial-partial-arch augmentation is also an H3 mechanism in disguise**: by training on partials (e.g., a 3-tooth quadrant), the model is implicitly learning to "use the local context of the visible teeth to predict the local context of the missing region" — the same inductive bias as LION's `z0` global latent (paper 005) and AnchorFormer's anchor scattering (paper 011), just expressed at the segmentation-task level rather than the generation-task level. **Implication: H3 generalizes across tasks, and the right H3 mechanism depends on what you're conditioning on (per-tooth FDI label vs. per-region geometry).**

### H4 (implicit SDF > explicit mesh) — **N/A**
Segmentation only; the input is a raw point cloud, the output is per-point class labels. H4 is about the *output representation*, which isn't an issue for segmentation.

### H5 (synthetic pretrain + light fine-tune) — **STRONGEST SUPPORT IN THE READING LIST SO FAR**
Cao is the *cleanest H5 paper in our reading list*. The artificial-partial-arch augmentation is *literally* the H5 recipe: synthesize the hard distribution from the easy distribution (crop partials from full-arch scans), train on the synthesized distribution, evaluate on real data. The 90% probability and skewed distribution (favoring fewer teeth) are the correct H5 details: high enough probability to dominate training, skewed distribution to match the clinical test distribution (where most partials are 2-4 teeth). The +0.197 macro-F1 jump from Model 1 to Model 2 on partial-arch is the **largest H5 evidence in the reading list** — bigger than PCN's KITTI 4× consistency improvement (paper 022), bigger than PVD's Redwood 3DScans transfer (paper 012), bigger than LION's 13-class ShapeNet-vol result (paper 005). **The lesson for our v0: if we can synthesize the hard distribution (crown-on-prepared-tooth-with-2-5-adjacent-context) from the easy distribution (full-arch scans from 3DTeethSeg22), we can use the same H5 recipe Cao used to dramatically improve the rare-class performance.**

## Surprises / things buried in section 4 (results / discussion)

1. **The biggest jump comes from data augmentation, not architecture.** Model 1 → Model 2 is +0.197 macro-F1 on partial-arch. Model 1 → Model 5 (everything) is +0.196 macro-F1. *All four architectural enhancements combined add zero net improvement on top of the augmentation* — the architecture is just preventing the augmentation from hurting the full-arch performance. This is a powerful anti-backbone-bias finding: **for the segmentation sub-task, data beats architecture by 10×**.

2. **The artificial-partial augmentation slightly *hurts* full-arch performance.** Model 1 full-arch F1 0.9935 → Model 2 full-arch F1 0.9911 (−0.0024); macro-IoU 0.9331 → 0.9053 (−0.0278). The paper's explanation: cropped scans teach the model to "focus on localized features", reducing its capacity to use long-range spatial relationships. **This is a real, underappreciated trade-off** in data-augmentation-driven learning. The paper recovers the full-arch performance with later enhancements (alignment + postprocessing + real partials), but the recovered macro-IoU is 0.9403, only +0.0072 above the baseline. **For our v0: we should monitor the *full-arch* performance even when the goal is *partial-arch* improvement.**

3. **The FDI-pair-offset postprocessing is a dynamic-programming MAP, not a neural network.** The 16×16 = 256 possible FDI tooth-pair candidates per arch, with a multivariate-Gaussian prior on the relative offsets, and Viterbi-style MAP inference. This is **classical AI (graph search + probabilistic graphical model)** applied to the *postprocessing* stage of a deep-learning pipeline. The lift is macro-IoU 0.9206 → 0.9280 on partial-arch, all attributed to the postprocessor. **This is a strong argument for hybrid pipelines: deep learning for the *per-tooth* prediction, classical combinatorial inference for the *global consistency* enforcement.** For our v0 crown-generation pipeline, the same pattern applies: PVD-AF-DiGS-FC for the per-tooth generation, then a classical combinatorial prior for "this crown must have FDI-#4 cusp morphology" enforced via H3-style conditioning.

4. **The Kendall's τ correlation is small but directionally consistent.** "Residual roots > residual crowns > missing teeth > partially erupted teeth" is the failure-mode ordering. **For our v0, the analogous failure modes will be: prepared teeth > teeth with buildups > teeth near wisdom teeth > teeth with crowns already present.** The 90%-probability artificial-partial augmentation should be *weighted* to over-sample these conditions, not just 2-12 consecutive teeth uniformly. The paper's uniform crop distribution is *the wrong default* if we want to be robust to clinical edge cases.

5. **The paper's "clinical translation" section under-promises.** The conclusion says "this work could aid clinicians in the *first step* of tooth identification" — i.e., they're framing their work as a building block, not a product. But the F1 0.9908 on full-arch and 0.9884 on partial-arch is *clinical-grade* — a dentist would need to manually review fewer than 1 in 100 teeth. This is *the bar to beat for our v0 segmentation stage*: at least 0.94 DSC on imperfect arches (paper 025's bar) but ideally 0.98+ (Cao's bar). **Our v0 sub-task 1 should target F1 ≥ 0.98 on imperfect arches, not 0.94.**

6. **No code release for the full Cao 2025 pipeline.** The ToothInstanceNet base is from a prior paper, and the alignment + postprocessing + augmentation code is described in the paper but not released. For our project, this means **we cannot simply use Cao's pipeline as-is** — we have to reimplement the four enhancements from scratch. The good news: each enhancement is *small* (~50-200 lines), the architecture is well-described, and PyTorch Lightning v2.3.3 is the only dependency. A clean reimplementation is a 2-3 day engineering task.

7. **The 3DTeethSeg score 0.9870 is the new SoTA bar.** Our v0 sub-task 1 (segmentation) needs to hit at least DSC 0.95 / F1 0.98 on the public Teeth3DS test set to be competitive. The 3DTeethSeg split is 1200 train / 600 test, so we can directly compare. **Practical implication: report our v0 segmentation metrics on the 600-scan public test set, not on a private split, so the numbers are directly comparable to Cao 2025, paper 025, and the 3DTeethSeg challenge leaders.**

8. **The left-right disambiguation via translation vector is a beautiful trick.** Partial-arch scans of *only posterior teeth* (e.g., a single quadrant) have no midline reference, so left/right is ambiguous. Cao's alignment module predicts *both* a rotation (up + forward directions) *and* a translation (c, the canonical position of the scan in the full arch), which together fully determine left/right. **For our v0: the same trick applies to crown generation — a single prepped tooth has no arch context, so the prep-margin geometry must be lifted to a canonical frame before generating the crown. Cao's translation head is the right inductive bias for "where in the arch is this scan?".**

## Quote-worthy sentences

- *"Partial-arch IOSs are commonly employed to generate digital designs for crowns, inlays, and bridges, offering accurate delineation of prepared teeth."* (Introduction) — the *exact* clinical use case our project targets.
- *"The introduction of cropped scans shifted the model's focus towards learning localized features. Consequently, its capacity to make use of long-range spatial relationships was slightly reduced."* (Discussion) — the underappreciated data-augmentation trade-off, in one sentence.
- *"Compared to the semiautomated transformer-based approach by Alsheghri et al, which was applied to artificially cropped partial-arch IOSs and achieved tooth Dice scores between 0.936 and 0.948, our fully automated Model 5 yielded a higher tooth Dice of 0.9862."* (Discussion) — the direct comparison to paper 025.
- *"While introducing artificial partial-arch IOSs and the alignment module significantly improved the model's performance on partial-arch IOSs, some errors persisted, particularly duplicate or skipped FDI labels. These were addressed in Model 4 by incorporating an FDI-aware postprocessing module, which leveraged both class probabilities and spatial relationships between teeth to enforce anatomical consistency."* (Discussion) — the FDI-pair-offset MAP in one sentence.
- *"The introduction of cropped scans shifted the model's focus towards learning localized features."* (Discussion) — the same idea again, important enough to repeat.
- *"Failure cases further showed the model's limitations when interpreting IOSs with complex dental conditions, including partially erupted teeth, prepared teeth, long bridge restorations, missing teeth, and irregular morphologies."* (Discussion) — the failure modes, ranked.

## Code / data

- **DOI:** https://doi.org/10.1016/j.identj.2025.100950
- **PMC full text:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12392768/ (CC BY 4.0)
- **PubMed:** https://pubmed.ncbi.nlm.nih.gov/40815915/
- **Code:** No dedicated repo for the full Cao 2025 pipeline. ToothInstanceNet (the base model) is from the same lab, and Vinayahalingam's lab has a track record of releasing code (e.g., https://github.com/vinayahalingam).
- **Data:** Internal (private clinic, Hubei, China). Not released publicly. The 3DTeethSeg / Teeth3DS public dataset (paper 001) was used for external validation and is the right benchmark for our v0 comparison.
- **Implementation dependencies:** PyTorch Lightning v2.3.3, scikit-learn v1.5.1.

## For our project — concrete next steps

1. **Adopt the artificial-partial-arch data augmentation as v0 sub-task 1's primary training trick.** The +0.197 macro-F1 jump on partial-arch is the largest H5 evidence in the reading list. **Concrete action:** in our v0 segmentation training loop, take the 3DTeethSeg22 1,800 full-arch scans, crop 2-12 consecutive teeth from each scan with 90% probability and a skewed distribution favoring fewer teeth, mix with the original full arches. This is a 1-day preprocessing step (similar OBB-based cropping to paper 025's preprocessing) that should be applied *before* the deep model touches the data. **Expected gain on partial-arch: +0.10-0.20 F1, similar to Cao's 025 → 026 jump.**

2. **Adopt the FDI-pair-offset multivariate-Gaussian postprocessor as v0 sub-task 1's final stage.** The 16×16=256 candidate per arch + multivariate-Gaussian prior + Viterbi MAP is a classical graphical-model postprocessor that adds macro-IoU +0.0074 on Cao's data. **Concrete action:** train the Gaussian on the FDI-to-FDI offsets from the 3DTeethSeg22 training set, implement the Viterbi MAP in ~50 lines of NumPy, and run after our sub-task 1 segmentation. This is a 1-day engineering task with $0 compute cost. **Expected gain: +0.005-0.01 macro-IoU, free.**

3. **Add a DL-based alignment module (Stratified Transformer) as a v1 sub-task 1 enhancement, not v0.** The alignment module is the most expensive of the four enhancements (Stratified Transformer is a 2022 transformer backbone, ~5-10M params) and the gain on top of artificial-partial augmentation is only +0.005-0.01. Defer to v1; v0 should use the simpler OBB + curvature-cue heuristic from paper 025 (reimplemented from scratch in 1 day). **For v1, the cleanest architecture is: ToothInstanceNet base + artificial-partial-augmentation + FDI-pair-offset postprocessor + Stratified-Transformer alignment, all trained end-to-end.**

4. **Adopt the 3DTeethSeg challenge split (1200 train / 600 test) as the v0 sub-task 1 evaluation protocol.** The challenge uses TLA, TSA, TIR, and the average Score — these are the *right* metrics for our segmentation stage, and reporting on the 600-scan public test set makes our numbers directly comparable to Cao 2025 (0.9870 Score), paper 025 (ArchSeg, did not evaluate on this), and the 3DTeethSeg challenge leaders. **Concrete action:** download the Teeth3DS dataset (paper 001 reference), use the 1200/600 split, report all four metrics, target Score ≥ 0.95 in v0 and Score ≥ 0.98 in v1.

5. **Reuse Cao's augmentation distribution as the v0 default, with one modification.** Cao uses uniform 2-12 consecutive teeth, with 90% probability, skewed toward fewer teeth. **For our project, we should also weight by FDI tooth type** (e.g., 2× over-sample molar-region partials, since molars are the most common crown site). The Kendall's τ analysis in the paper shows that *residual roots* and *residual crowns* are the highest-error conditions — neither is a normal clinical input, but *prepared teeth* are. **Add a 3rd condition to the augmentation: randomly trim a tooth's surface (simulating preparation) with 10% probability, so the model sees some "prepped" anatomy in training.**

6. **Treat the 0.9403 macro-IoU on full-arch as the v0 segmentation bar.** Cao's Model 5 hits 0.9403 on full-arch, 0.9280 on partial-arch. **Our v0 sub-task 1 (segmentation) should target macro-IoU ≥ 0.90 on imperfect-arch test sets and ≥ 0.94 on full-arch test sets** to be competitive. If we hit 0.98+ on both, we can skip the v1 architecture improvements and go straight to the generative sub-task 4.

7. **Open question for HK: should we use Cao's ToothInstanceNet, paper 024 Kunwar's DGCNN+MeshSegNet+Blender stack, or paper 025 ArchSeg's PTv2+graph-cut as the v0 base model?** ToothInstanceNet (Cao) is the most modern, has the best ablation table, and the lab has released the architecture. Paper 024 (Kunwar) is the only one with code + Blender postprocessing recipe released. Paper 025 (ArchSeg) has no code release. **Recommendation: pilot all three on the 3DTeethSeg 1200/600 split, pick the one with the highest Score, adopt as v0 base model.** This is a 1-week, $100 Lambda experiment.

8. **v0 segmentation stack candidate: Cao's pipeline, reimplemented in PyTorch 2.x.** Reimplement ToothInstanceNet base (2-stage DentalNet+refinement, ~300 lines), artificial-partial augmentation (1-day preprocessing), FDI-pair-offset postprocessor (1-day engineering, 50 lines of NumPy), and the OBB+curvature-cue heuristic alignment from paper 025 (1-day engineering, 100 lines). Skip the DL-based alignment module for v0. Total engineering: ~1 week. Total compute: ~$50-100 for a 1,800-scan training run. **Expected Score on 3DTeethSeg public test: 0.95-0.98.**

9. **Open question for HK: do we need *real* partial-arch IOSs in training, or is synthetic enough?** Cao's Model 5 (with real partials) outperforms Model 4 (without) by a small margin, but the gap is mostly on the long tail (residual roots, residual crowns, partially erupted). **For our v0, we likely don't have access to a 300-real-partial-arch dataset** — 3DTeethSeg22 is full-arch only. **Recommendation: skip real partials in v0, add a 50-real-partial-arch pilot in v1 if budget allows (the paper's Hubei clinic dataset is not publicly available).**

10. **Next paper to read: TSegNet (Rekik et al., the 3DTeethSeg challenge organizer) for the original 2022 baseline, OR a non-dental 3D segmentation SoTA (e.g., OneFormer3D or PointGroup) for the cross-domain best practice, OR Stratified Transformer (the alignment backbone) for the architecture details.** The cleanest follow-up is to read the ToothInstanceNet / DentalNet / Stratified Transformer trio (the three base models Cao built on) — these are the *building blocks* of the new SoTA, and understanding them is the prerequisite to reimplementing Cao's pipeline in v0. **Alternative: pivot to 3DTeethSeg challenge 2024 winners (if any new SoTA paper exists past Aug 2025) for a fresh comparison point.**

---

**Word count:** ~3,400
**Status:** Read 2026-06-07 01:03 KST. Hypothesis impact: **H1 strong support refined, H3 strong support new-form (DP-MAP over FDI pair offsets), H5 STRONGEST support in reading list (synthesize-partials-from-full-augs +0.197 macro-F1 on partial-arch), H2/H4 N/A**.
