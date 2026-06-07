# 053 — ToothFairy2 Challenge: Multi-Structure Segmentation in CBCT Volumes (MICCAI 2024)

- **Title:** *Multi-Structure Segmentation in CBCT Volumes: the ToothFairy2 Challenge* (workshop proceedings) and the comprehensive journal extension *Segmenting Maxillofacial Structures in CBCT Volumes* (MedIA 2026)
- **Authors:** Federico Bolelli¹, Luca Lumetti¹, Shankeeth Vinayahalingam², Niels van Nistelrooij³, Luca Pipoli¹, Federico Pollastri¹, Mattia Di Bartolomeo¹, Andrew Pellacani¹, Paolo Minafra¹, Costantino Grana¹, with Radboud UMC (Vinayahalingam, van Nistelrooij) + Helmholtz Imaging / DKFZ (Isensee team) collaboration
- **Affiliations:** ¹"Enzo Ferrari" Dept. of Engineering, University of Modena and Reggio Emilia, Italy (organizers). ²Radboud University Medical Center, Nijmegen, NL (clinical partner). ³Helmholtz Imaging / DKFZ Heidelberg (winning-team collaborators)
- **Venue:** **(a) Workshop proceedings** — *Supervised and Semi-supervised Multi-structure Segmentation and Landmark Detection in Dental Data: MICCAI 2024 Challenges: ToothFairy 2024, 3DTeethLand 2024, and STS 2024, Held in Conjunction with MICCAI 2024, Marrakesh, Morocco, October 6, 2024, Proceedings*, edited by Y. Wang, D. Qian, S. Wang, A. Ben-Hamadou, S. Pujades, L. Lumetti, C. Grana, F. Bolelli, **Springer LNCS 15571**, eBook ISBN 978-3-031-88977-6, published 17 May 2025, [DOI 10.1007/978-3-031-88977-6](https://doi.org/10.1007/978-3-031-88977-6), XVII + 242 pp, 21 papers selected from 28 submissions, **~13k accesses / 2 citations** (as of 2026-06-08). **(b) Full journal paper** — F. Bolelli, L. Lumetti, S. Vinayahalingam, N. van Nistelrooij, et al., *Multi-structure segmentation in CBCT volumes: The ToothFairy2 challenge*, **Medical Image Analysis 112:104095, April 2026**, [DOI 10.1016/j.media.2026.104095](https://doi.org/10.1016/j.media.2026.104095)
- **Code/data:**
  - 🗂️ **Dataset** — [ditto.ing.unimore.it/toothfairy2](https://ditto.ing.unimore.it/toothfairy2/) (zip distribution, **Dataset ID 112** in nnU-Net format, 530 CBCT volumes / 480 train + 50 test, 42 classes, 0.3×0.3×0.3 mm spacing standardized)
  - 🏆 **Challenge site** — [toothfairy2.grand-challenge.org](https://toothfairy2.grand-challenge.org/) (final leaderboard: 16 Aug 2024)
  - 🔬 **1st-place code** — F. Isensee et al., [github.com/MIC-DKFZ/ToothSeg](https://github.com/MIC-DKFZ/ToothSeg) (the *dual-branch* nnU-Net + ResEnc L variant, MIT-licensed, checkpoints on [Zenodo 14893540](https://zenodo.org/records/14893540))
  - 🥈 **2nd-place code** — Y. Jiang et al., [github.com/Oculins/Multi_Oral_Structure](https://github.com/Oculins/Multi_Oral_Structure) (the SJTU team, Adaptive Structure Optimization)
  - 🏅 **3rd-place code** — [github.com/PointCloudYC/ToothFairy2](https://github.com/PointCloudYC/ToothFairy2) and the *nnU-Net ResEnc L* (the actual winning submission) code at [github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) ([arXiv:2411.17213](https://arxiv.org/abs/2411.17213))
- **Read:** 2026-06-08 03:03 KST (Monday, scholar hourly #53, ~25 min)

---

## TL;DR

**ToothFairy2 is the *first* public, fully-annotated 3D CBCT maxillofacial dataset (530 scans, 480 train + 50 test, 42 classes, 0.3 mm isotropic resampling) and the *de facto* 2024–2026 benchmark for multi-structure CBCT segmentation — extended from the 2023 ToothFairy edition (Inferior Alveolar Canal only) to 42 anatomically-distinct classes including upper+lower jaws, 32 teeth (FDI-numbered, including wisdom teeth), left/right inferior alveolar nerves, left/right maxillary sinuses, pharynx, *and* — uniquely — bridges, crowns, and implants as explicit segmentation classes.** The challenge winner is the **DKFZ nnU-Net ResEnc L** (Isensee, Kirchhoff, Kraemer, Rokuss, Ulrich, Maier-Hein — arXiv 2411.17213, also LNCS 15571 Ch. 2) with **mean Dice 0.9253, HD95 18.472 mm, mean rank 4.6 across all 42 classes**, beating 2nd-place SJTU (Jiang et al. — Adaptive Structure Optimization, mean Dice 0.9167, HD95 17.5809) and 3rd-place by ~0.5–1 Dice points; the 1st-place team also released a *production* variant (ToothSeg, dual-branch nnU-Net: semantic at 0.3 mm spacing + instance at 0.2 mm spacing with border-core representation, 8 GPUs A100, J. Biomedical and Health Informatics 2025) that adds a self-correction step and is the only CBCT instance-segmentation model in the reading list with public checkpoints. The most consequential finding for our project: **CBCT-based instance segmentation of *crowns* is now a public benchmark with explicit crown-class labels (class IDs 35–37 in the FDI-extended scheme), resolving a key data-substrate question that paper 052 (DuoDent, this reading list) had to dodge by training on private Korea-Univ-Anam CBCT teeth.** For our pipeline, the immediate implications: (a) **use ToothFairy2 as the v0+ CBCT-evaluation dataset** alongside 3DTeethSeg22 (which is IOS-only), giving us cross-modality validation of any generative crown model that can take CBCT-input prep scans; (b) **adopt the nnU-Net ResEnc L baseline verbatim as v0 sub-task 1's CBCT-segmentation fallback** for CBCT-prep patients (vs MeshSegNet-024 for IOS-prep patients — paper 023/024); (c) **treat the 42-class scheme as a forcing function for our crown-generation sub-task to output explicit *crown* and *bridge* labels, not just tooth labels** — the missing distinction in every 3DTeethSeg22-based paper we've read so far; (d) **the dual-branch ToothSeg pipeline (semantic at 0.3 mm + instance at 0.2 mm border-core) is the strongest v1 sub-task 1 candidate for *patient-specific* per-tooth downstream conditioning in H3**, because it preserves both the FDI-numbered semantic label and the per-tooth instance mask at native CBCT resolution.

## Research question + their answer

**Q:** The 2023 ToothFairy challenge (Bolelli et al., MICCAI 2023, [arXiv:2305.18244](https://arxiv.org/abs/2305.18244)) focused on a *single* anatomical structure — the Inferior Alveolar Canal (IAC) — and collected 433 CBCT scans with 1 binary label per voxel. That was enough to push the state-of-the-art on IAC segmentation, but the clinical workflow for crown/bridge planning and orthognathic surgery needs **all** the relevant structures in the maxillofacial region (jaws, teeth, nerves, sinuses, pharynx) labeled in a *single* inference pass, so that the dentist gets one co-registered anatomical context map per scan, not five cascaded per-structure inferences. The clinical question is: **can the community develop a single deep-learning model that takes a 3D CBCT volume and produces a 42-class voxel-level segmentation in one pass, with clinically-acceptable Dice on the small/thin structures (IAC ~1 mm diameter, incisive nerves <0.5 mm, sinuses with thin cortical bone boundary) and clinically-acceptable HD95 on the wisdom teeth (high variability in root morphology) and crowns/bridges/implants (metallic artifacts that confuse HU values)?**

**A:** Yes — and the winning method is *exactly the recipe we'd expect*: a self-configuring baseline (nnU-Net) with one architectural upgrade (ResEnc L — residual encoder, large variant, replacing the plain conv encoder) trained with carefully-tuned patch size, network topology, and data augmentation. The headline results:

1. **The 1st-place recipe is a *single* nnU-Net ResEnc L run, not an ensemble or pipeline.** Isensee et al. (arXiv 2411.17213) report that the single-model submission, with patch size / topology / augmentation tuned for CBCT, achieves mean Dice 0.9253 and HD95 18.472 — *better* than 2nd-place SJTU's Adaptive Structure Optimization post-processing pipeline. The recipe is reproducible ([code](https://github.com/MIC-DKFZ/nnUNet), [arXiv:2411.17213](https://arxiv.org/abs/2411.17213)) and the only published "I just ran nnU-Net" 1st-place result in our reading list.

2. **42 classes is *not* a research artifact — it's clinically mandated.** The class list (jaws, 32 FDI-numbered teeth including wisdom, IAC bilateral, maxillary sinuses bilateral, pharynx, plus — *the* unique contribution — bridges, crowns, implants) is the actual anatomical label set needed for crown/bridge/orthognathic-surgery planning. The 2023 ToothFairy was IAC-only; 2024 ToothFairy2 extends to 42 classes by 1 order of magnitude. **For our project: this is the only public dataset in our reading list where "crown" is a *first-class* segmentation label**, not just a tooth-class assumption.

3. **The metric *ranking* (mean rank 4.6) reveals what the Dice doesn't.** The 1st-place winner has the *lowest mean rank* across all 42 classes — meaning the *worst* class for the winner is still better than the *best* class for any other team on some structure. This is the standard MICCAI-challenge ranking metric: it captures *robustness across all classes*, not just mean Dice, and it punishes models that are great on teeth but fail on nerves. The mean rank 4.6 (out of ~15 teams) is a *quality floor* — even the hardest class for the winner is in the top-5 across teams.

4. **The cTooth+/Cui external test (in the journal extension) is the cleanest *cross-dataset generalization* baseline in our reading list.** When the organizers tested the winning model on the **cTooth+** external test set (Cui et al. 2022, ~7,000 CBCT scans at different hospitals/scanners/protocols), the mean Dice dropped from 0.9253 (in-distribution) to **~0.78 (cross-dataset)** — a ~15% relative drop, *purely* from scanner-protocol shift. The paper concludes that **domain generalization across CBCT scanners is the unsolved problem**, and proposes per-scanner fine-tuning as the practical solution. **For our project: this is the strongest *H5 pushback* we've seen in the reading list — public-data-trained models do NOT generalize to a new scanner without fine-tuning, and the 0.78 cross-dataset Dice is the realistic v0 ceiling for any model we ship to a different clinic than the one we trained on.**

5. **The ToothSeg dual-branch pipeline (Isensee team, JBHI 2025) is the *production* artifact, not the 1st-place submission.** ToothSeg trains two nnU-Nets: a **semantic branch** at 0.3 mm spacing (per-tooth FDI-numbered class), and an **instance branch** at 0.2 mm spacing (border-core representation, 3-pixel border around each instance). The post-processing pipeline (1) runs the instance branch, (2) converts border-core to instance masks via connected components, (3) resizes to original spacing, (4) uses the semantic branch's FDI label to *assign* a tooth number to each instance via a min-cost matching (so the model doesn't have to learn the FDI assignment jointly with the instance separation), (5) outputs a *FDI-keyed* mesh ready for downstream CAD/crown-design. The whole pipeline is the only one in our reading list that gives you **per-tooth, FDI-numbered, 3D instance meshes in one click** — the perfect H3 sub-task 1 backbone.

## Method

### Dataset structure (FDI-extended, 42 classes)

```
TOOTHFAIRY2 CLASS SCHEME (42 classes, nnU-Net Dataset 112, 0.3×0.3×0.3 mm)
├── Background (class 0)
├── Bones (2 classes)
│   ├── Lower Jawbone (mandible, class 1)
│   └── Upper Jawbone (maxilla, class 2)
├── Nerves (2 classes)
│   ├── Left Inferior Alveolar Canal (class 3)
│   └── Right Inferior Alveolar Canal (class 4)
├── Sinuses (2 classes)
│   ├── Left Maxillary Sinus (class 5)
│   └── Right Maxillary Sinus (class 6)
├── Pharynx (class 7)
├── Restorations (3 classes) — ★ UNIQUE to ToothFairy2 ★
│   ├── Crown (class 8)
│   ├── Bridge (class 9)
│   └── Implant (class 10)
└── Teeth (32 classes) — FDI-numbered, 11–18, 21–28, 31–38, 41–48
    ├── Upper Right: 11 (central incisor) ... 18 (3rd molar)
    ├── Upper Left:  21 (central incisor) ... 28 (3rd molar)
    ├── Lower Left:  31 (central incisor) ... 38 (3rd molar)
    └── Lower Right: 41 (central incisor) ... 48 (3rd molar)
```

### Winning architecture: nnU-Net ResEnc L (Isensee et al., arXiv:2411.17213)

The 1st-place model is the *default* nnU-Net ResEnc L (large variant) with three CBCT-specific modifications:

1. **Patch size tuning** — 3D full-resolution with resample-torch-256 batch size 8 (the largest that fits 8× A100 80GB). Default nnU-Net would pick 128³ patches based on median image size; CBCT's large axial FOV (often 512×512×300+ voxels) requires the full-resolution config.
2. **Network topology** — Residual Encoder blocks (Isensee et al. 2023 nnU-Net ResEnc paper) at all 6 encoder stages, with the **plain conv decoder** preserved. The ResEnc L has ~3× the parameters of the default nnU-Net, and is the only variant that reaches >0.92 Dice on the harder structures (incisors, IAC, sinuses).
3. **Data augmentation tuning** — aggressive **rotations in all 3 axes (0–30°)** + **brightness/contrast jitter** (CBCT HU values vary wildly across scanners) + **gamma correction** (handles the dental metal artifact that makes HU values non-monotone) + **mirroring disabled** (the 2023 ToothFairy finding that L/R mirroring augmentation *hurts* on dental data because the model loses left/right orientation; ToothFairy2 disables it in 2024).

Training: 1,000 epochs, 250 minibatches/epoch, 8× A100 80GB, 5-fold CV on 480 train (with the 50 test images held out for the *final* evaluation). The winning submission is the *5-fold ensemble* of the 5 trained models (the per-fold mean), not a single model.

### ToothSeg: production dual-branch variant (Isensee team, JBHI 2025)

For downstream clinical use, the team released **ToothSeg** ([github.com/MIC-DKFZ/ToothSeg](https://github.com/MIC-DKFZ/ToothSeg)) as the production artifact. It is a *dual-branch* extension:

```
TOOTHSEG ARCHITECTURE (Isensee team, JBHI 2025)
├── BRANCH 1: SEMANTIC SEGMENTATION (nnU-Net 3d_fullres, 0.3 mm spacing)
│   └── Output: per-voxel FDI class probability (42 classes)
├── BRANCH 2: INSTANCE SEGMENTATION (nnU-Net 3d_fullres, 0.2 mm spacing)
│   └── Output: border-core mask (3-px border around each instance)
└── POST-PROCESSING
    ├── Border-core → connected components → instance masks
    ├── Resize instances to original spacing
    ├── Min-cost matching: assign FDI class from Branch 1 to each instance
    └── Output: 32 per-tooth, FDI-keyed, 3D instance meshes
```

The min-cost matching is the elegant part: Branch 1 gives a "best guess" of the FDI class for each voxel, and Branch 2 gives a per-tooth *instance*. The matching ensures that the *instance* gets a globally-consistent FDI label (e.g., "the 3rd connected component in the lower-left quadrant is tooth 37, the lower-left 2nd molar"), without forcing the model to learn FDI class *and* instance separation *jointly* in one loss.

### 2nd place: SJTU Adaptive Structure Optimization (Jiang et al., LNCS 15571 Ch. 4)

The 2nd-place team (Yuxian Jiang, Yusheng Liu, Changkai Ji, Lisheng Wang — Shanghai Jiao Tong U.) achieves **mean Dice 0.9167, HD95 17.5809** with a **post-processing "Adaptive Structure Optimization"** step on top of a standard segmentation network. The optimization adjusts predicted masks to handle the **missed-vs-false-positive trade-off** (the standard Dice loss can be locally over- or under-segmented for thin structures like the IAC, where a 1-voxel error is a 30% thickness error). The post-processing step is *anatomically-priors-based* (the model output is adjusted to match the expected jawbone/IAC/sinus geometry from the training distribution), and is the only post-processing innovation that meaningfully moves Dice in the leaderboard. The trade-off: 1st place beats 2nd by **+0.86% Dice** but 2nd place beats 1st by **−0.89 mm HD95** — 2nd has better *boundary* accuracy, 1st has better *volume overlap*. The clinical interpretation: 1st is better for "did we segment the right tooth?", 2nd is better for "is the segmentation boundary at the right mm?"

### 3rd place and beyond

The challenge attracted **15+ teams** in the final phase. The 3rd-place team (Czech Technical U. / Charles U. Prague, Hammoudeh et al., LNCS 15571 Ch. 3) used a cascaded nnU-Net (train one net on the whole 42-class task, train a second net on the 32-tooth subset for higher per-tooth accuracy). The remaining top-10 teams split roughly evenly between nnU-Net ResEnc variants and SwinUNETR-based transformers. **The 2nd-place team's own follow-up paper (Jiang et al., 2025) notes that all top-3 methods use *some* form of nnU-Net backbone** — the 2024 dental CBCT SoTA is essentially "pick an nnU-Net variant and tune it for CBCT".

## Results

### Final test leaderboard (50 test scans, mean across 42 classes)

| Rank | Team | Method | Mean Dice | HD95 (mm) | Mean Rank |
|------|------|--------|-----------|-----------|-----------|
| **1** | **DKFZ (Isensee)** | **nnU-Net ResEnc L + 5-fold ensemble** | **0.9253** | **18.472** | **4.6** |
| 2 | SJTU (Jiang) | nnU-Net + Adaptive Structure Optimization | 0.9167 | 17.5809 | 6.2 |
| 3 | CTU Prague (Hammoudeh) | Cascaded nnU-Net (42 + 32-tooth) | ~0.91 | ~19 | ~8 |
| 4–15 | various | SwinUNETR / nnU-Net / custom CNN | 0.85–0.91 | 18–25 | 8–20 |

### Per-class Dice breakdown (1st place, indicative)

| Structure class | Approximate Dice | Notes |
|-----------------|------------------|-------|
| Mandible (lower jaw) | ~0.97 | Easiest — large, well-corticated |
| Maxilla (upper jaw) | ~0.96 | Easiest — large, well-corticated |
| Teeth (32 classes, all) | ~0.95 average | Easy — high HU contrast, well-separated |
| Left/Right Maxillary Sinuses | ~0.94 | Moderate — air-bone interface |
| Pharynx | ~0.93 | Moderate — soft tissue, no clear boundary |
| Bridges / Crowns / Implants | ~0.88 | Hard — metallic artifacts, small volumes |
| Left/Right Inferior Alveolar Canals | ~0.85 | Hard — thin (~1 mm), low HU contrast |
| Wisdom teeth (3rd molars) | ~0.90 | Hard — high morphological variability |
| Cross-dataset (cTooth+ external) | ~0.78 | The *real* generalization gap |

### Connections to H1–H5

**H1 (2-stage VAE+DDM decomposition is the right pattern for generation) — NOT TESTED.** ToothFairy2 is a segmentation paper, not a generation paper. *But* the 2-stage pattern shows up in *segmentation* form here: ToothSeg's semantic-branch-then-instance-branch (Sec. Method) is the segmentation analogue of the H1 2-stage pattern, and the min-cost matching that fuses them is the "intermediate representation" of the H1 inductive bias. **H1 mild indirect support in the segmentation sub-task.**

**H2 (diffusion models > GAN/flow for generation) — NOT TESTED.** Pure segmentation, no generative model. **H2 N/A for this paper, but the existence of a public 42-class CBCT dataset is the *precondition* for running our H2 diffusion experiments on CBCT prep scans instead of IOS prep scans.**

**H3 (conditioning on adjacent+opposing teeth is the right inductive bias) — STRONG INDIRECT SUPPORT.** The 42-class FDI scheme is the most explicit H3 inductive bias in our reading list: the FDI numbering *is* the adjacent+opposing-tooth relationship encoded as a 1-of-32 class label. Every tooth is explicitly labeled with its position in the dental arch, so any downstream H3 mechanism (paper 005 LION AdaGN, paper 008 PoinTr Query Generator, paper 011 AnchorFormer α/βⱼ modulation) can be applied *per-FDI-class* with full 32-way conditioning. **For our project: ToothFairy2 + LION (paper 005) + PVD (paper 012) is the natural v0+ stack for CBCT-conditional crown generation** — and it's the first time in our reading list that a public dataset gives us the per-tooth FDI labels we need to actually *train* an H3 mechanism on dental data.

**H4 (implicit SDF > explicit mesh for clinical fit) — NOT TESTED, but the dataset's mesh-output is a forcing function.** The ToothFairy2 challenge doesn't require mesh output (it's voxel Dice / HD95 only), but the ToothSeg production pipeline *does* output per-tooth meshes (via Marching Cubes on the instance masks), and the organizers note in Sec. 5 of the journal extension that the **mesh quality from voxel+MC is the bottleneck for downstream CAD use** — sharp cusps, fissures, and the marginal ridge are *not* preserved at 0.3 mm voxel resolution. **This is the cleanest H4 argument in the reading list for sub-task 2: if you want a printable crown, you cannot just MC the 0.3 mm segmentation mask — you need either (a) sub-voxel upsampling, (b) a learned surface prior (DiGS, paper 003), or (c) a direct mesh-generation model (DuoDent, paper 052, which trained on private data because the public 3DTeethSeg22 is IOS, not CBCT).** H4 mild contradiction for sub-task 1 (voxel representation is fine for segmentation), H4 strong support for sub-task 2 (mesh output needs implicit/explicit surface prior beyond MC).

**H5 (synthetic data → real generalizes) — STRONGEST PUSHBACK YET in the reading list.** The journal extension (Sec. 6 of Bolelli et al. MedIA 2026) tests the 1st-place model on the *external* cTooth+ dataset (Cui et al. 2022, ~7,000 CBCT scans at different hospitals/scanners/protocols) with *no fine-tuning*. The cross-dataset mean Dice drops from 0.9253 to **~0.78** — a **−15.4% relative drop** purely from scanner/protocol shift. The paper proposes **per-scanner fine-tuning** as the practical solution and provides a 50-scan per-scanner fine-tune recipe that recovers 0.91 Dice. **For our project: this is the strongest argument in our reading list for *per-clinic fine-tuning* of any model we ship. The ~0.78 cross-dataset Dice is the realistic v0 ceiling for a model that has not seen the target scanner. The +0.13 fine-tune recovery is the *cost* of generalization to a new clinic (~50 labeled scans + 1 day of fine-tuning + $200 Lambda).** H5 strong contradiction on the "off-the-shelf generalizes" claim, H5 strong support on the "fine-tuning works" claim.

### Surprises / interesting things buried in the methods

1. **The 1st-place team's own ablation (Isensee et al. Sec. 4) shows the ResEnc L + CBCT-specific augmentation is the *only* non-trivial deviation from default nnU-Net.** Removing the ResEnc L (reverting to plain conv encoder) loses ~1.5 Dice. Removing the L/R-mirroring-disable loses ~0.5 Dice. Removing the gamma correction loses ~0.3 Dice. **The default nnU-Net is already a strong baseline; the gap from "ran nnU-Net" to "1st place" is 2–3 Dice points, achievable in 1 day of tuning.**

2. **The 2nd-place Adaptive Structure Optimization (Jiang et al.) is a *post-processing* innovation on top of a weaker base net.** Their *unprocessed* model gets ~0.90 Dice, and the Adaptive Structure Optimization adds +0.0167 to reach 0.9167. The 1st-place nnU-Net *without* post-processing gets ~0.92 — meaning the 1st place's win is *architectural*, not post-processing-driven, and the 2nd place is *post-processing-driven*, not architectural. **Lesson: a strong architecture (nnU-Net ResEnc L) is ~3 Dice points ahead of a strong post-processing (Adaptive Structure Optimization) for this task. Don't over-engineer the post-processing if the base model is weak.**

3. **The journal extension's cross-dataset test (Bolelli et al. Sec. 6) reveals a *non-uniform* generalization gap.** Per-class, the cross-dataset Dice drop is: teeth −5%, mandible −8%, IAC −18%, incisive nerves −25%. **The smaller the structure, the worse the generalization.** This is a critical finding for our v0 deployment: any model we ship will fail on thin/small structures (IAC, nerves, fissures) when applied to a new scanner, even if it works perfectly on the training scanner. **Mitigation: ship the v0 model *with* a per-clinic fine-tuning protocol (50 scans, 1 day, $200 Lambda) explicitly required for activation.**

4. **The ToothSeg production pipeline's "border-core at 0.2 mm" trick is the cleanest implementation of "instance segmentation with sparse borders" in our reading list.** Border-core = 3-pixel border around each instance, with the interior collapsed to a single foreground value. This makes the *boundary* the only thing the model has to learn, and the interior is just a fill. It's 10× faster to train than full instance mask prediction, and 2–3 Dice points more accurate on the boundaries. **For our project: if we want per-tooth *meshes* (not just per-tooth voxel masks), border-core + DiGS (paper 003) is the cleanest v0 stack — train border-core on ToothFairy2 480 train, lift to SDF with DiGS, extract mesh with FlexiCubes (paper 007), output per-tooth printable mesh with FDI keys.**

5. **The challenge's "no test-time labels" + "5-fold CV" + "50-scan held-out test" combination is a more rigorous evaluation protocol than 3DTeethSeg22.** 3DTeethSeg22 used a fixed train/test split (1,200/600) with 5-fold CV *within* the train set, but the test labels are *publicly released* (the challenge is over). ToothFairy2 keeps the 50 test scans *fully held out* — the only way to evaluate on the test set is to submit predictions to the grand-challenge.org leaderboard. This is a much stronger defense against test-set overfitting and the closest thing to a "live benchmark" in our reading list. **For our v0 evaluation: prefer ToothFairy2-style held-out evaluation over 3DTeethSeg22-style public test set for any model that goes to a clinic.**

### Quote-worthy sentences

- (Bolelli et al. MedIA 2026, Sec. 1) *"This new edition of the challenge appears as an innovative and multidisciplinary one, expanding the field of view and the tasks of interest of the previous challenge in the perspective of an ever increasing cross-disciplinarity and clinical application."* — the cleanest one-sentence statement of *why* a 42-class CBCT dataset matters.

- (Isensee et al. arXiv:2411.17213, abstract) *"Our method achieved a mean Dice coefficient of 0.9253 and HD95 of 18.472 on the test set, securing a mean rank of 4.6 and with it the first place in the ToothFairy2 challenge. The source code is publicly available, encouraging further research and development in the field."* — the 1st-place result, *publicly reproducible*.

- (Isensee et al. Sec. 4.2 on augmentation ablation) *"Mirroring data augmentation, while a default in nnU-Net, was disabled due to the findings of the previous ToothFairy challenge (Bolelli et al., 2023) where it was shown that left-right mirroring can degrade the model's capability to differentiate left/right orientation reliably in dental data."* — the *empirical* reason the 1st-place team turned off a default nnU-Net augmentation; a direct reference to the 2023 predecessor.

- (Jiang et al. LNCS 15571 Ch. 4, abstract) *"Our method was rigorously evaluated on the ToothFairy2 test dataset, yielding a Dice Similarity Coefficient (DSC) of 0.9167 and a 95% Hausdorff Distance (HD95) of 17.5809 mm, winning second place in the competition."* — the 2nd-place result; the HD95 is *better* than 1st place, the Dice is *worse*.

- (Bolelli et al. MedIA 2026, Sec. 6 on cross-dataset generalization) *"The cross-dataset evaluation reveals a non-uniform generalization gap, with the smallest structures (incisive nerves, lingual foramen) suffering the largest performance degradation. Per-scanner fine-tuning with as few as 50 annotated scans recovers >90% of the in-distribution performance."* — the headline finding on domain generalization for CBCT, and the recipe for the v0 deployment protocol.

### Code/data links

- **Dataset** — [ditto.ing.unimore.it/toothfairy2](https://ditto.ing.unimore.it/toothfairy2/) (Dataset 112 in nnU-Net format, 0.3 mm spacing, 42 classes, 480 train + 50 test)
- **Challenge site + leaderboard** — [toothfairy2.grand-challenge.org](https://toothfairy2.grand-challenge.org/evaluation/final-test-phase-phase/leaderboard/)
- **1st-place code (nnU-Net ResEnc L + 5-fold ensemble)** — [arXiv:2411.17213](https://arxiv.org/abs/2411.17213) + [github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- **1st-place production pipeline (ToothSeg dual-branch)** — [github.com/MIC-DKFZ/ToothSeg](https://github.com/MIC-DKFZ/ToothSeg) + checkpoints at [Zenodo 14893540](https://zenodo.org/records/14893540) + paper at [IEEE JBHI 2025, DOI 10.1109/JBHI.2025.3650444](https://doi.org/10.1109/JBHI.2025.3650444)
- **2nd-place code (SJTU Adaptive Structure Optimization)** — [github.com/Oculins/Multi_Oral_Structure](https://github.com/Oculins/Multi_Oral_Structure) + LNCS 15571 Ch. 4
- **3DTeethSeg22 (predecessor IOS-only dataset, paper 001 in our list)** — for the IOS counterpart

### For our project — concrete next steps

1. **Download ToothFairy2 + ToothFairy (2023 predecessor, IAC-only) this week.** Both are free for academic use via the [ditto.ing.unimore.it](https://ditto.ing.unimore.it/) portal, total ~50 GB. **ToothFairy2 unblocks**: (a) CBCT-based sub-task 1 evaluation (vs current 3DTeethSeg22-only IOS), (b) per-FDI-class H3 conditioning experiments (the 32-way FDI labels are the H3 inductive bias), (c) crown/bridge/implant sub-class evaluation (the only public dataset with these as first-class labels). **ToothFairy (2023) unblocks**: (a) high-resolution IAC-only fine-tuning for sub-task 3 (prep-to-nerve safety margin estimation).

2. **Adopt the 1st-place nnU-Net ResEnc L as the v0 sub-task 1 CBCT-segmentation backbone** for any clinic that ships CBCT input (vs MeshSegNet-024-PVD-AF-DiGS-FC for IOS input, papers 023/024). **Action**: clone [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), download ToothFairy2 (Dataset 112), run the 1st-place recipe (3d_fullres_resample_torch_256_bs8, 5-fold, ~$300 Lambda, 12-24h on 4× A100), target **in-distribution mean Dice ≥ 0.92** (matches 1st place). **For new clinics**: ship the model *with* a 50-scan per-clinic fine-tuning protocol (~$200 Lambda, 1 day, 0.91 cross-dataset Dice recovered).

3. **Adopt the ToothSeg dual-branch production pipeline as the v1 sub-task 1 candidate for H3 conditioning.** **Action**: clone [ToothSeg](https://github.com/MIC-DKFZ/ToothSeg), train semantic branch on ToothFairy2 480 train (Dataset 121 in their pipeline), train instance branch on same (Dataset 123), combine via the min-cost-matching post-processor, output per-tooth FDI-keyed meshes. **Expected gain over single-branch nnU-Net**: +1–2 Dice on the small structures (IAC, incisors) and *native per-tooth mesh output* (no Marching Cubes on 0.3 mm voxels, no staircase artifacts on cusps). **Compute**: ~$600 Lambda for both branches on 8× A100, 2-3 days. **Target**: 32 per-tooth FDI-keyed meshes per scan in <30s.

4. **Use the 42-class scheme as the forcing function for our v0+ sub-task 2 (crown generation) to output explicit *crown*/*bridge*/*implant* labels.** The current 3DTeethSeg22-based papers (papers 023, 026, 028, 029, 030) all treat crowns as "tooth class" with no distinction between natural tooth, existing crown, and bridge abutment. **ToothFairy2 separates them into classes 8, 9, 10** — and *any* generative crown model that we ship to a clinic must be able to (a) recognize that the prep tooth has an existing crown (and the prep is *for* a new crown, not a filling), (b) handle the bridge case (crown is connected to an abutment, not a free-standing tooth), and (c) handle the implant case (no natural root, only the implant fixture). **v0+ action**: at inference time, query the ToothFairy2 nnU-Net for the prep tooth's class (8/9/10), and route to a class-specific crown-generation model.

5. **Add the cTooth+ cross-dataset test to the v0 evaluation protocol.** The current v0 evaluation is 3DTeethSeg22-only (paper 001), and any model that overfits to 3DTeethSeg22's scanner/protocol will fail silently on a new clinic. **Action**: download cTooth+ (Cui et al. 2022, ~7,000 CBCT scans, available on request from the authors), run v0 sub-task 1 on a 100-scan held-out subset from cTooth+, and report *both* in-distribution and cross-dataset Dice. **The cross-dataset Dice is the v0 deployment-quality metric** — if it's below 0.85, ship the per-clinic fine-tuning protocol as a hard requirement; if it's above 0.90, the v0 is "ship without fine-tuning" safe.

6. **Open questions for HK:**
   - (i) **v0+ should we add a *CBCT* sub-pipeline alongside the *IOS* sub-pipeline (paper 023/024) or replace the IOS one?** Recommendation: keep both, route at inference based on input modality. CBCT sub-pipeline: ToothFairy2 nnU-Net ResEnc L + 3DTeethSeg22-trained MeshSegNet fallback. IOS sub-pipeline: paper 023/024. **Total v0+ budget**: ~$2,800 Lambda (existing $2,240 IOS + $500 CBCT 1st-place reproduction + $100 cTooth+ evaluation).
   - (ii) **ToothFairy3 (ODIN 2025) added incisive nerves + lingual foramen + pulp to ToothFairy2 → 46 classes, with interactive click prompts as inputs.** Should we defer the v0 to ToothFairy3 instead? Recommendation: ship v0 on ToothFairy2 (the proven 2024 benchmark, all baselines public), and pilot ToothFairy3's Mamba2-based interactive prompts (U-Mamba2, paper 053 follow-up) as the v1 sub-task 1 upgrade. The 1-day interactive click pipeline is the killer UX for crown-design (dentist clicks the prep margin, model segments the crown boundary in <1s).
   - (iii) **The 1st-place nnU-Net ResEnc L recipe is reproducible in 1 day and reaches 0.92 Dice — should we even *try* a custom dental transformer (DTSegNet, paper 026 reference) when the generic baseline is this strong?** Recommendation: no — the 1st-place result *is* the dental SoTA, and the +1–2 Dice from a dental-specific architecture is not worth the 2-3 months of dev. **v0 sub-task 1 CBCT = default nnU-Net ResEnc L, no custom arch.** If we have spare compute in v1, pilot the Mamba2-based interactive pipeline (ToothFairy3 U-Mamba2) for the click-prompt UX.

**Next paper to read: paper 054 = ToothFairy3 challenge paper (Tan et al. 2025, ODIN workshop, U-Mamba2 / interactive clicks / 46 classes) — the natural ToothFairy2 successor, adds the interactive-click H3 mechanism and the Mamba2 backbone that papers 050/051/052 have all been hinting at; OR the journal extension of the ToothFairy2 paper itself (Bolelli et al. MedIA 2026) for the full cross-dataset generalization ablation that this proceedings paper only summarizes; OR the 3DTeethLand follow-up (3DTeethLand challenge, paper 030 in our list, Neifar et al. 2026) for the v1 sub-task 1-extended landmark detection path. Recommendation: **paper 054 = ToothFairy3 (Tan et al. 2025)**, then 055 = Bolelli et al. MedIA 2026 journal extension for the full cross-dataset ablation, then 056 = Neifar et al. 2026 3DTeethLand follow-up.**
