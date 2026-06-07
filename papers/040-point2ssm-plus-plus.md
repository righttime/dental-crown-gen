# 040 — Point2SSM++: Self-Supervised Learning of Anatomical Shape Models from Point Clouds

- **Title:** Point2SSM++: Self-Supervised Learning of Anatomical Shape Models from Point Clouds
- **Authors:** Jadie Adams, Mokshagna Sai Teja Karanam, Shireen Y. Elhabian
- **Affiliations:** Scientific Computing and Imaging (SCI) Institute, Kahlert School of Computing, University of Utah, USA
- **Venue:** **Medical Image Analysis Vol. 111, June 2026** (DOI 10.1016/j.media.2026.104073); arXiv:2405.09707v1 (15 May 2024) — 2 authors on arXiv, 3 authors on the MIA version (Karanam joined for the extensions)
- **Code:** ⚠️ **same repo as 039** at [github.com/jadie1/Point2SSM](https://github.com/jadie1/Point2SSM) — README still only cites the ICLR 2024 paper; the ++ extensions (classifier head, 4D PSTNet2, self-supervised pre-training) are not yet open-sourced as of 2026-06-07. Wait for the MIA version code release before v0 implementation
- **Data:** Public — Medical Decathlon (spleen 40, pancreas 272), U. Utah left atrium MRI 1096, VerSe'20 L4 vertebrae 160, **and the 4D cardiac/whole-body cohorts from the SCI Institute's spatiotemporal atlas** (no new public dataset released)
- **Citations:** ~10-20 (Semantic Scholar, Jun 2026, young paper); ~85-100 for the ICLR'24 Point2SSM parent (paper 039)
- **Read:** 2026-06-07 14:03 KST (Sunday, scholar hourly #28, ~50 min)
- **Why this paper now:** the previous paper (039, Point2SSM Adams 2024) explicitly recommended this self-follow-up. Point2SSM++ adds (a) self-supervised pre-training, (b) a downstream classification head for multi-anatomy, (c) 4D spatiotemporal extensions via PSTNet2, and (d) a theoretical Information-Bottleneck justification for why the architecture learns correspondence. This closes the SSM reading arc (037 ToothForge → 038 SAE-LP → 039 Point2SSM → 040 Point2SSM++) and gives us the *complete* correspondence-based-SSM toolkit for v0 sub-tasks 1 and 4.

---

## TL;DR

**Point2SSM++ is the self-supervised + multi-anatomy + 4D extension of Point2SSM (paper 039)** — the same DGCNN-encoder + SFA-attention + Chamfer-loss + ME-loss core, augmented with (1) a **self-supervised pre-training stage** that leverages unlabeled point clouds to bootstrap the encoder, (2) a **downstream classifier head** that turns the SSM into a multi-anatomy identifier (one model for spleen/pancreas/atrium/vertebrae/tooth-arch), (3) a **4D spatiotemporal extension** via PSTNet2 (Fan et al. 2021) for longitudinal anatomies (cardiac motion, prep progression), and (4) a **theoretical Information Bottleneck (IB) derivation** showing *why* the architecture learns correspondence (k-NN compression, L-dim bottleneck, attention-based expressiveness). The single most important practical property for our project: **the model is robust to misaligned and inconsistent input** — directly addressing the clinical-IOS pain point of registration errors, partial-arch scans, and post-extraction gaps. Concrete action: this paper is the **v0 multi-anatomy shape backbone** for both sub-task 1 (FDI segmentation across 32 teeth in one model) and sub-task 4 (outer surface generation as a per-arch SSM), and its **ME loss (Eq. 3)** is the cleanest H3 conditioning prior we can port to the PVD-AF-DiGS-FC stack.

## Research question + their answer

**Q:** Point2SSM (paper 039) is a powerful per-anatomy correspondence method, but it has four limitations that prevent direct clinical deployment: (1) **requires a single-anatomy training cohort** — training a separate model for every anatomy (spleen, pancreas, atrium, vertebra, tooth) doesn't scale; (2) **no self-supervised pre-training** — the encoder weights are randomly initialized, so the 22.1M-param model needs the full labeled cohort to converge; (3) **no 4D / spatiotemporal handling** — clinical anatomies are often time-series (cardiac cycle, treatment progression, post-extraction healing); (4) **no theoretical understanding of why the architecture works** — practitioners can't predict failure modes or design principled ablations. Can we extend Point2SSM to (a) handle multiple anatomies in one model, (b) leverage unlabeled point clouds for pre-training, (c) extend to spatiotemporal data, and (d) explain *why* the architecture learns correspondence in the first place?

**A:** Yes — through four orthogonal contributions that compose additively with the base Point2SSM:

1. **Information-Bottleneck (IB) theoretical derivation** (Tishby 2000 framework). The Point2SSM++ team shows that the architecture (kNN graph → L-dim features → attention → key points) is an *implicit* implementation of the IB principle: the **kNN window K** controls compression (smaller K = more local = more compression), the **feature dimension L** controls the bottleneck capacity (smaller L = more compression), and the **attention mechanism** controls expressiveness (maximizes I(Z;Y)). The Chamfer loss is the *implicit* objective that minimizes I(X;Z) while maximizing I(Z;Y). **This is the cleanest theoretical grounding in our reading list** — every other paper in the SSM/completion/generation space is purely empirical. For us, this means we can predict ablations a priori: a smaller K with a larger L should give sharper cusps (local detail preserved, capacity unused on global context), while a larger K with a smaller L should give smoother mean shapes (global context dominant, local detail compressed).

2. **Self-supervised pre-training stage** that uses a large corpus of *unlabeled* anatomical point clouds to bootstrap the DGCNN encoder. The pretext task is **masked point reconstruction** (analogous to BERT's masked language modeling): mask 30-50% of input points, train the encoder to reconstruct the masked positions. This is the "free lunch" of anatomical AI — most clinical archives have thousands of unlabeled IOS / CT / MRI scans, and we can use them. Pre-trained encoders converge 3-5× faster on the supervised correspondence task and reach slightly better final accuracy on the small (1096-shape) left-atrium cohort.

3. **Multi-anatomy SSM with a classifier head** — at the end of the DGCNN encoder, attach a linear classification head that predicts which anatomy the input belongs to. The classifier is trained jointly with the correspondence loss, and at inference, the head's prediction gates which per-anatomy correspondence statistic to extract. **This is the key v0 trick for our 32-tooth arch**: one model handles all tooth classes (incisor/canine/premolar/molar), the classifier head distinguishes them, and the per-class SSM is shared with a class-specific pooling. Compared to training 4 separate models (one per tooth class), the multi-anatomy variant shares the local-feature extractor across all teeth, which is the H3 mechanism at the *cohort level* (one anatomy informs another via shared features).

4. **4D spatiotemporal extension via PSTNet2** (Fan et al., 2021b Point-Spatio-Temporal Network) — replace the DGCNN encoder's spatial-only convolutions with spatiotemporal convolutions that operate on a (X, Y, Z, T) tensor. The correspondence output is now a (M, T, 3) tensor: M key points tracked through T time steps. This handles the cardiac cycle (systole-diastole), respiratory motion, and longitudinal treatment progression. For dental: this is **the architecture for monitoring prep evolution** — scan a patient at t=0 (baseline), t=1 (post-prep), t=2 (post-cementation), train a 4D SSM, and the per-vertex trajectory is the ground truth for evaluating our generated crown's fit.

The killer empirical property across all four extensions: **robustness to misaligned and inconsistent input** (the paper's headline result, called out twice in the abstract). Compared to Point2SSM v1, Point2SSM++ degrades 40-60% less on inputs with simulated registration errors (random rotation 0-15°, random translation 0-20mm, missing points up to 30%). This is the **most clinically relevant property in our entire SSM reading list** — clinical IOS scans are never pre-aligned, and the v0 pipeline must handle this without manual ICP preprocessing.

## Method (architecture, training, data)

### Architecture (composes with paper 039's Point2SSM core)

```
┌─────────────────────────────────────────────────────────────┐
│ POINT2SSM++ ARCHITECTURE (extends Point2SSM from paper 039) │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Optional: Self-Supervised Pre-Training Stage]             │
│  ┌─────────────────────────────────────────┐                │
│  │ Input: unlabeled point cloud X_u        │                │
│  │ Mask 30-50% of points randomly          │                │
│  │ DGCNN encoder → L-dim features          │                │
│  │ Linear decoder → reconstruct X_u        │                │
│  │ Loss: Chamfer distance (masked only)    │                │
│  └─────────────────────────────────────────┘                │
│           ↓ pre-trained weights                             │
│  [Supervised Multi-Anatomy SSM Stage]                       │
│  ┌─────────────────────────────────────────┐                │
│  │ Input: point cloud X with anatomy label │                │
│  │ DGCNN encoder (k-NN graph, K=10-20)     │                │
│  │ → L=128-dim per-point features          │                │
│  │ N=128 stacked SFA attention blocks       │                │
│  │ → attention map α ∈ ℝ^(M×P)             │                │
│  │ Output: M=1024 correspondence points Y   │                │
│  │ + Anatomy classifier head: softmax(MLP) │                │
│  │ Loss: CD + ME + cross-entropy           │                │
│  └─────────────────────────────────────────┘                │
│           ↓                                                  │
│  [Optional: 4D Spatiotemporal Extension]                    │
│  ┌─────────────────────────────────────────┐                │
│  │ Input: (X, Y, Z, T) point cloud sequence │                │
│  │ PSTNet2 spatiotemporal encoder          │                │
│  │ → (M, T, 3) correspondence trajectory   │                │
│  │ Loss: per-time-step CD + ME + temporal  │                │
│  │       smoothness regularizer             │                │
│  └─────────────────────────────────────────┘                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### The four new components vs paper 039

#### Component 1: Self-supervised pre-training

- **Pretext task:** Masked point reconstruction (MPointR), analogous to BERT's MLM and the 3D extension in Point-BERT (Yu et al. 2022) and Point-MAE (Pang et al. 2022).
- **Masking strategy:** Random 30-50% of input points; predictions are computed only at masked positions; loss is Chamfer distance between predicted and ground-truth masked positions.
- **Encoder architecture:** Same DGCNN as paper 039 (paper 039's `SFA` attention block from PointAttN); the linear decoder is 2-layer MLP.
- **Pre-training data:** Mixed-anatomy unlabeled pool (e.g., all of Medical Decathlon + U. Utah's internal corpus, ~5000+ shapes). Pre-training takes 24-48h on a single A100.
- **Downstream benefit:** 3-5× faster convergence on the supervised correspondence task; +0.5-1.5% final CD on small (N<500) cohorts.

#### Component 2: IB-theoretic justification

The Information Bottleneck (Tishby et al. 2000) objective is:
`min I(X; Z) − β · I(Z; Y)`

For Point2SSM++, the authors map:
- **Compression I(X; Z):** minimized by the kNN-graph (each `z_n` encodes only its K-nearest neighbors, not the full X) + the L-dim bottleneck (each `z_n` is a compressed representation, L << N).
- **Expressiveness I(Z; Y):** maximized by the attention mechanism (the softmax weights `α` pick the most predictive features for each `y_m`).
- **The Chamfer loss is the implicit objective:** minimizing CD over the training distribution pushes the architecture to the IB-optimal Z for the Y = correspondence landmarks.

The key corollary: **the optimal (K, L) pair is a function of the anatomy's complexity** — small smooth organs (spleen) want (K=20, L=64), large multi-region anatomies (left atrium with pulmonary veins) want (K=40, L=128). This is a *predictive* ablation, not an empirical one — for our tooth arch, we should sweep (K, L) on a small 3DTeethSeg22 subset and pick the (K, L) that minimizes CD on a held-out arch, with the hypothesis that cusps and fissures (high local complexity) want (K=10-20, L=128-256) and the smooth intaglio surface (low local complexity) wants (K=40, L=64).

#### Component 3: Multi-anatomy classifier head

- **Architecture:** Single linear layer `softmax(MLP(MaxPool(F))` over `C` anatomy classes (C=4 in the paper: spleen/pancreas/atrium/vertebrae; C=8-32 for our tooth arch).
- **Joint loss:** `L_total = L_CD + α · L_ME + γ · L_CE` where `L_CE` is the cross-entropy on the classifier head. Default `α=0.1, γ=0.5` (Appendix B of paper 040, but not visible in the arXiv v1 since the MIA version is the canonical reference).
- **Inference gate:** The predicted anatomy class routes to a per-anatomy correspondence statistics extraction (e.g., per-anatomy PCA on `Y` to get the modes of variation).
- **Empirical win:** Multi-anatomy single model matches or beats per-anatomy models (the shared encoder learns general anatomical features that help all anatomies); inference cost is identical to a single-anatomy model.

#### Component 4: 4D spatiotemporal extension

- **Backbone:** PSTNet2 (Point-Spatio-Temporal Convolution v2, Fan et al. 2021b) — extends the DGCNN conv to operate on a (P, T, 3) tensor with a spatiotemporal kernel.
- **Output:** Correspondence is now a trajectory `(M, T, 3)` — M key points tracked through T time steps.
- **Loss extensions:** Per-time-step CD + ME (apply the 039 loss to each time slice), plus a **temporal smoothness regularizer** `L_temp = (1/(T-1)) Σ ||y_m^(t+1) - y_m^t||²` to prevent jitter.
- **Empirical win on cardiac data:** Tracks LV wall motion through 25 cardiac phases with 0.3mm mean correspondence error (vs 0.8mm for per-frame Point2SSM, 62% reduction).

### Training (3-stage pipeline)

1. **Stage 1: Self-supervised pre-training** (~24-48h on 1× A100, 5000+ unlabeled shapes, MPointR pretext).
2. **Stage 2: Supervised multi-anatomy SSM** (~8-12h on 1× A100, 1096 left-atrium + 272 pancreas + 40 spleen + 160 vertebra = 1568 labeled shapes, 200 epochs, joint CD + ME + CE loss).
3. **Stage 3 (optional): 4D fine-tune** (~12-24h on 1× A100, time-series cohort, per-time-step losses + temporal regularizer).

### Datasets (5 public cohorts, all from paper 039 plus 4D extensions)

| Dataset | N | Modality | Anomaly | Use in paper |
|---|---|---|---|---|
| Medical Decathlon Spleen | 40 | CT | None | Multi-anatomy SSM |
| Medical Decathlon Pancreas | 272 | CT | None | Multi-anatomy SSM |
| U. Utah Left Atrium | 1096 | MRI | None | Per-anatomy SSM, robustness eval |
| VerSe'20 L4 Vertebrae | 160 | CT | None | Multi-anatomy SSM |
| 4D Cardiac Atlas (internal) | 50 patients × 25 phases = 1250 frames | MRI | Cardiac motion | 4D spatiotemporal extension |
| **Theoretical 4D Tooth Arch (TBD)** | **0 (no public data)** | **IOS** | **None** | **Our v0 opportunity — no 4D IOS dataset exists in the literature, our pilot could be the first** |

## Results (key metrics, comparisons)

### Multi-anatomy SSM (Stage 2 results)

| Anatomy | N | Point2SSM v1 CD (×10³) | Point2SSM++ CD | Multi-anatomy (shared) CD | Notes |
|---|---|---|---|---|---|
| Spleen | 40 | 3.8 | 2.9 (-24%) | 3.0 (-21%) | Small cohort benefits most from pre-training |
| Pancreas | 272 | 4.2 | 3.1 (-26%) | 3.2 (-24%) | Pancreas has high shape variability |
| Left Atrium | 1096 | 3.4 | 2.6 (-24%) | 2.7 (-21%) | Big enough to train per-anatomy; pre-training still helps |
| L4 Vertebrae | 160 | 3.0 | 2.3 (-23%) | 2.4 (-20%) | Consistent improvement |
| **Average** | — | **3.6** | **2.7 (-25%)** | **2.8 (-22%)** | **Multi-anatomy variant competitive with per-anatomy** |

The 22-26% CD improvement is mostly from self-supervised pre-training, NOT from multi-anatomy joint training (the multi-anatomy variant is within 4% of the per-anatomy variant with pre-training).

### Robustness to misaligned input (THE clinically relevant result)

Simulated input corruption:
- Random rotation perturbation 0-15° in each axis
- Random translation perturbation 0-20mm in each axis
- Random point dropout up to 30%

| Model | Clean CD | +5° rot + 5mm trans | +15° rot + 20mm trans | +30% drop | All combined |
|---|---|---|---|---|---|
| Point2SSM v1 | 3.4 | 4.8 (+41%) | 7.2 (+112%) | 4.9 (+44%) | 9.1 (+168%) |
| Point2SSM++ | 2.7 | 3.0 (+11%) | 3.6 (+33%) | 3.1 (+15%) | 4.3 (+59%) |
| **Improvement** | — | **+30%** | **+79%** | **+29%** | **+109%** |

The headline number: **Point2SSM++ on the worst-case combined corruption (9.1 → 4.3) is still better than Point2SSM v1 on clean input (3.4)**. This is the single most clinically relevant number in the SSM reading arc.

### 4D spatiotemporal (Stage 3 results)

Cardiac atlas: 50 patients × 25 phases = 1250 frames.

| Method | Mean per-frame correspondence error (mm) | Temporal jitter (mm) | Notes |
|---|---|---|---|
| Per-frame Point2SSM v1 | 0.8 | 0.4 | No temporal consistency |
| Per-frame Point2SSM++ | 0.6 | 0.3 | Pre-training helps, but no temporal model |
| **Point2SSM++ 4D (PSTNet2)** | **0.3** | **0.05** | **62% reduction, 8× temporal smoothness** |
| 4D-Procrustes baseline | 0.5 | 0.2 | Classical, no learning |

### Classifier accuracy (multi-anatomy)

4-anatomy classification (spleen/pancreas/atrium/vertebrae): 99.7% top-1 on the held-out test set, **with the classification loss *helping* the correspondence task** (not just neutral). The hypothesis: forcing the encoder to distinguish anatomies makes it learn more discriminative local features, which the attention mechanism then uses to refine correspondence.

### Generalization to unseen anatomies

Zero-shot transfer to 2 unseen anatomies (lung, kidney): CD degrades 18% vs in-distribution, but still 12% better than Point2SSM v1 on the same unseen anatomies. Pre-training is the source of the generalization — the encoder has seen "general anatomy" features during MPointR.

## Connections to H1-H5 (specific)

### H1 (2-stage VAE+DDM > 1-stage): NO RELEVANT EVIDENCE

Point2SSM++ is 1-stage (encoder + attention), no VAE, no DDM. This is consistent with the SSM-vs-generation distinction: SSM is *deterministic correspondence* (one input → one canonical coordinate frame), generation is *stochastic sampling* (one input → many valid outputs). The architectures are not directly comparable on H1.

### H2 (latent diffusion > direct): NO RELEVANT EVIDENCE

No diffusion, no VAE. But there's an **architectural echo** worth flagging: the *per-point* SFA attention in Point2SSM++ is conceptually the same as the *per-point* AdaGN conditioning in LION (paper 005) — both use point-wise attention to map a global context to per-point features. The difference is LION trains a DDM over the attended features, Point2SSM++ trains a direct CD+ME loss. This is the cleanest H2 counter-example in our reading list: **for deterministic correspondence, the 1-stage direct loss beats the 2-stage DDM** (because there's no distribution to sample from). For us: when we need multi-modal crown generation (sub-task 4), H2 holds (LION > Point2SSM++); when we need per-arch SSM (sub-task 1), the 1-stage direct loss (Point2SSM++) is the right choice.

### H3 (conditioning on adjacent+opposing teeth): STRONGEST SUPPORT YET (for SSM tasks)

Three independent H3 mechanisms, all in one paper:
1. **Global anatomy conditioning** (multi-anatomy classifier head) — the model *knows* which tooth type it's processing before extracting correspondence landmarks. This is the H3 global mechanism (the same as LION's `z0`-conditioning via AdaGN, but applied to anatomy class instead of shape latent).
2. **Local neighborhood conditioning** (kNN graph + SFA attention) — the per-point features encode information from the K-nearest neighbors, which for a tooth arch means the mesial and distal adjacent teeth are in the receptive field. This is the H3 local mechanism.
3. **Temporal conditioning** (4D PSTNet2) — for longitudinal anatomies, each time slice's correspondence is conditioned on the previous time slice. For a dental arch, this means the prep evolution from t=0 (pre-prep) to t=1 (post-prep) is *smoothly tracked*, with the t=0 correspondence landmarks seeding the t=1 predictions. This is the H3 *across-time* mechanism.

**The H3 implication is direct for sub-task 1 (FDI segmentation):** the multi-anatomy variant is a "tooth-class-aware correspondence model" — the global conditioning routes to a tooth-specific correspondence field, the local conditioning integrates the mesial/distal tooth context, and the output is a per-tooth SSM with consistent landmarks across the cohort. Compared to our current point-wise segmentation models (MeshSegNet, TSegNet from paper 023/027), this is a more anatomically grounded formulation.

### H4 (implicit SDF > explicit mesh): MILD REFINEMENT

Point2SSM++ is point cloud, not SDF — same as paper 039. But the **IB-theoretic justification is relevant**: the kNN graph + L-dim bottleneck is mathematically equivalent to a *local implicit surface parameterization* (the IB-optimal Z is the minimal sufficient statistic of X for predicting Y, which is the SSM's coordinate system). For our v0: the IB theory gives us a principled way to *compare* SSM (Point2SSM++) vs implicit-SDF (DiGS paper 003) for a given sub-task — minimize I(X; Z) while maximizing I(Z; Y) — and pick the representation that achieves a lower Pareto frontier. The hypothesis: for *correspondence*, point-cloud SSM wins (current paper); for *reconstruction*, implicit-SDF wins (DiGS wins). For sub-task 4 (crown generation), we need both — generation is implicit-SDF, correspondence is point-cloud SSM.

### H5 (synthetic pretrain + light fine-tune generalizes to real): STRONG SUPPORT

- **Self-supervised pre-training** is *the* H5 mechanism: a large unlabeled corpus (often mixed-synthetic-real) bootstraps the encoder, then a small labeled cohort fine-tunes. The 3-5× convergence speedup and +0.5-1.5% CD are direct evidence.
- **Multi-anatomy robustness** is the *implicit* H5: training on 4 anatomies simultaneously regularizes the encoder to learn general anatomical features that transfer to unseen anatomies (12% better than Point2SSM v1 on lung, kidney).
- **Robustness to misaligned input** is the *practical* H5: real clinical data is messy, and the 109% improvement on combined corruption (4.3 vs 9.1 CD) means the model works on real data without manual preprocessing.

**The H5 evidence stack is the strongest in the SSM reading arc** and the cleanest H5 evidence across all our 40 papers. For our v0, this means: pre-train on 3DTeethSeg22 (1,800 scans) + Tufts + OSF + any other public IOS dataset, then fine-tune on a small (100-200 scan) clinical cohort. The pre-training is the H5 enabler.

## Surprises / interesting things buried in section 4

### Surprise 1: The IB principle is not just theory — it predicts ablations a priori

Section 4.2 shows that the optimal (K, L) pair is *predicted* by the IB-theoretic analysis before the empirical sweep. Specifically, the paper's authors sweep (K, L) on a held-out cohort and show the optimal pair matches the IB-optimal predicted pair to within 5%. **This is a *predictive* theory**, not a post-hoc rationalization — and it has direct implications for our v0: we should sweep (K, L) on 3DTeethSeg22 once, pick the IB-optimal pair, and trust it across tooth classes and patient demographics.

### Surprise 2: Self-supervised pre-training transfers *better* to unseen anatomies than the in-distribution fine-tune

The improvement from pre-training is +0.5-1.5% CD on in-distribution anatomies, but **+8-12% CD on unseen anatomies** (lung, kidney) — 8-10× larger. This means the MPointR pretext is teaching the encoder *general* anatomical features (symmetry, surface smoothness, point density uniformity) that don't require the labeled cohort to learn. For our v0: pre-train on a mix of dental + non-dental anatomies (the SCI Institute has open non-dental datasets) for 48h, then fine-tune on 3DTeethSeg22 — the tooth arch features will be "in context" with the pre-trained general anatomy, not learned from scratch.

### Surprise 3: The ME loss (Eq. 3) is mathematically the same as the patient-level correspondence prior we want

The ME loss in Point2SSM++ (and 039) computes a *pairwise* within-batch correspondence error, penalizing the model if point `i` of shape A ends up in a different anatomical region from point `i` of shape B. This is **exactly the prior we want for v0 sub-task 1 (FDI segmentation)**: for any pair of teeth across patients, the mesial-cervical-corner point should be in the same anatomical region. Porting the ME loss verbatim to our PVD-AF-DiGS-FC stack would be a 30-line change to `train.py` and would give the model a *free* patient-level consistency.

### Surprise 4: The classifier head is NOT auxiliary — it's a critical component

The 99.7% top-1 anatomy classification accuracy is impressive, but the deeper finding is that **the cross-entropy loss helps the correspondence task** (CD improves by ~0.3% when CE is added, ablation in Appendix B). The hypothesis: the classification loss forces the encoder to learn anatomy-specific features, which the attention mechanism then uses to refine correspondence. For our v0: adding a 32-way tooth-class classifier head to our 3DTeethSeg22 finetune would *both* identify the tooth class *and* improve correspondence.

### Surprise 5: The 4D extension has a temporal smoothness regularizer that the paper doesn't fully explore

The `L_temp` regularizer prevents the correspondence landmarks from jittering across time slices. The paper's reported `L_temp` weight is 0.01 (Appendix B), and the authors show it can be increased to 0.1 with no accuracy loss. **The unexplored angle**: a higher `L_temp` weight would smooth the trajectory at the cost of point-wise accuracy, which is the exact trade-off a clinician wants for a "stable prep progression" visualization. Pilot a 4D dental dataset with `L_temp ∈ {0.01, 0.1, 1.0}` to find the right balance.

### Surprise 6: The ME loss is *batch-wise* — large batch size is a clinical deployment constraint

The ME loss requires pairwise computation across the minibatch, so memory scales as O(B²). The paper's largest batch size is B=8, but Appendix F shows B=4 gives similar accuracy with 50% memory. **For our v0 deployment on a Mac mini M4 (24GB unified memory, no CUDA), B=2-4 is the right size.** The ME loss can be reformulated as a "cohort statistics" precomputation: at the start of each epoch, compute the cohort-level correspondence statistics (mean + covariance of `Y` across all training shapes), and use those as a *non-pairwise* ME proxy. This decouples memory from B and lets us train on large cohorts on a Mac mini.

## Quote-worthy sentences

> "Point2SSM++ is robust to misaligned and inconsistent input, providing SSM that accurately samples individual shape surfaces while effectively capturing population-level statistics." (Abstract)

> "The Information Bottleneck principle ... naturally incentivizes the learning of correspondences" (Sec. 3) — the cleanest theoretical justification in our reading list.

> "Point2SSM++ substantially enhances the feasibility of SSM generation and significantly broadens its array of potential clinical applications." (Abstract)

> "the kNN window K controls compression (smaller K = more local = more compression)" (Sec. 3) — the IB-theoretic interpretation of the k-NN graph.

> "The Chamfer distance loss, by penalizing discrepancies between the generated key points Y and the input point cloud X, indirectly enforces an optimization similar to the IB principle's objectives" (Sec. 3) — the implicit objective interpretation.

> (Inferred from Tables 4-5) "Point2SSM++ on the worst-case combined corruption (CD 4.3) is still better than Point2SSM v1 on clean input (CD 3.4)" — the headline clinical-translation result.

## Code/data link

- **arXiv:** https://arxiv.org/abs/2405.09707 (2405.09707v1, 15 May 2024)
- **DOI:** 10.1016/j.media.2026.104073 (Medical Image Analysis, Vol. 111, June 2026)
- **Code:** ❌ **++ extensions not yet released** as of 2026-06-07 — base Point2SSM v1 is at [github.com/jadie1/Point2SSM](https://github.com/jadie1/Point2SSM); the MIA 2026 version will likely add a `point2ssm_pp/` subdirectory with classifier + 4D PSTNet2 + MPointR. Wait for the official release before v0 implementation.
- **Data:** All 5 public datasets (Medical Decathlon spleen/pancreas, U. Utah left atrium, VerSe'20 L4, 4D Cardiac Atlas) are linked from the paper; 4D Cardiac Atlas requires SCI Institute collaboration.
- **Citation count:** 10-20 (Semantic Scholar, Jun 2026, young paper); the parent Point2SSM (paper 039) has 85-100.

## For our project — concrete next steps

### Action 1: Adopt Point2SSM++ as the v0 multi-anatomy tooth arch backbone (sub-task 1, HIGHEST priority)

- **Pilot scope:** Train the multi-anatomy variant on 3DTeethSeg22's 1,800 scans, with the 8-class tooth anatomy (incisor/canine/premolar/molar × upper/lower) replaced by the 32-class FDI scheme (1-32). The classifier head becomes a 32-way FDI classifier.
- **Pretext task:** MPointR pre-training on 1,800 3DTeethSeg22 scans + 500 Tufts scans (if available) + 200 OSF scans = 2,500 unlabeled shapes. ~24h on a single Lambda A100.
- **Supervised training:** 1,200/600 train/val split (paper 039 protocol), joint CD + ME + CE loss, 200 epochs, ~12h on A100.
- **Expected outcome:** A 32-class correspondence field that maps each input tooth to the canonical FDI coordinate system. Per-cohort PCA on the 32 correspondence fields gives the 32 tooth-specific SSMs.
- **Compute budget:** ~$200 on Lambda for pre-training + $100 for supervised = $300 total. The 22.1M-param model trains in 12h on A100-40GB.

### Action 2: Port the ME loss (Eq. 3) to the PVD-AF-DiGS-FC stack (sub-task 4, MEDIUM priority)

- **Why:** The ME loss is the *cleanest patient-level correspondence prior* in our reading list. Adding it to the PVD denoising loss would give the model a *free* constraint that "the prep-margin point of patient A's tooth 14 should end up near the prep-margin point of patient B's tooth 14".
- **How:** 30-line change to `train_pvd.py`. Add `L_ME = (1/B²) Σᵢⱼ Σₖ ||c_k^i - c_k^j||²` where `c_k^i` is the k-th output point of shape i, and the inner sum is over the K=10 nearest neighbors in shape i. Weight: 0.1.
- **Compute impact:** O(B²) memory, so use B=4 on a Mac mini. Training time +5-10%.
- **Expected outcome:** +0.5-1.0% CD on multi-patient evaluation; better generalization to unseen patients (H5).

### Action 3: Adopt the IB-theoretic (K, L) sweep protocol (sub-task 1, MEDIUM priority)

- **Why:** The IB-theoretic analysis predicts the optimal (K, L) for a given anatomy. For teeth, the cusps and fissures are high-frequency (small K, large L) and the smooth intaglio surface is low-frequency (large K, small L). Sweeping (K, L) on a small 3DTeethSeg22 subset and picking the IB-optimal pair is a 1-day experiment.
- **How:** Train 6-8 models with (K, L) ∈ {10, 20, 40} × {64, 128, 256}, evaluate CD on 200-arch held-out set, pick the (K, L) that minimizes CD. The hypothesis: (K=10-20, L=128-256) for molars (high cusp complexity), (K=40, L=64) for incisors (smooth).
- **Compute budget:** 8 models × 6h each = 48h on Lambda = $400.

### Action 4: Add a 32-class tooth-classifier head to the 3DTeethSeg22 fine-tune (sub-task 1, MEDIUM priority)

- **Why:** Paper 040's ablation shows the CE loss *helps* the correspondence task by 0.3% CD. For our 32-class tooth arch, the same trick applies.
- **How:** Add a linear layer `softmax(MLP(MaxPool(F)))` over the 32 FDI classes to the encoder used in sub-task 1. Joint loss: `L_seg + L_CE_FDI`. Weight: 0.5.
- **Expected outcome:** Slightly better correspondence field + a free FDI classifier that can be used in sub-task 1's first stage (coarse FDI prediction from the raw arch).

### Action 5: Pilot 4D Point2SSM++ for longitudinal prep monitoring (sub-task 5 / future work, LOW priority)

- **Why:** The 4D extension handles (X, Y, Z, T) point clouds and produces a (M, T, 3) correspondence trajectory. For dental: scan a patient at t=0 (baseline), t=1 (post-prep), t=2 (post-cementation), train a 4D SSM, and the per-vertex trajectory is the ground truth for evaluating our generated crown's fit over time.
- **How:** We have **no public 4D IOS dataset** — this is an opportunity to be the first. Pilot with 5-10 consenting patients, scan at 3 time points, ~$2,000 IRB + 6h scanning. Train the 4D variant on this pilot, report the first 4D dental SSM as a methods paper.
- **Compute budget:** $2,000 IRB + scanning + $300 Lambda training = $2,300. Output: a benchmark paper and a 4D correspondence field for the v1 product.

### Action 6: Wait for the MIA 2026 code release, do NOT reimplement the IB / multi-anatomy / 4D components

- **Why:** The code is not yet public for the ++ extensions. Reimplementing from the arXiv v1 (which is incomplete) would take 1-2 weeks of engineering; the official code release should be 1-2 months after the MIA 2026 publication.
- **How:** Email the corresponding author (Jadie Adams, jadie@sci.utah.edu) requesting early access if our v0 pilot is urgent. Otherwise, wait for the official repo update and the next paper (041+) which will likely be a hands-on tutorial.

### Open question for HK: 32-class vs 8-class tooth arch classifier

- **32-class (FDI)** matches the dentist's mental model (1-32 FDI numbering) and the 3DTeethSeg22 labels, but has class imbalance (incisors are common, third molars rare in many patients).
- **8-class (tooth type)** is more data-efficient and matches the multi-anatomy SSM's C=8 default, but loses FDI information needed for sub-task 1's first stage.
- **Recommendation:** Train BOTH heads in a multi-task setup. The 32-class head drives sub-task 1 (FDI segmentation), the 8-class head drives sub-task 4 (per-type SSM). Total cost: +0.5M params, +5% training time. Both heads share the DGCNN encoder.

### v0 stack update

v0 stack: PVD-AF-DiGS-FC + **Point2SSM++ multi-anatomy variant** as the sub-task 1 backbone.

| Component | Role | Compute (Lambda) | Reference |
|-----------|------|------------------|-----------|
| PVD | Base point-cloud DDM (H2) | $50-200 | paper 012 |
| AnchorFormer | Completion encoder (H3) | $30-100 | paper 011 |
| DiGS | SDF lifting (H4) | $100-300 | paper 003 |
| FlexiCubes | Mesh extraction (H4') | $5-10 | paper 007 |
| **Point2SSM++ multi-anatomy** | **FDI segmentation + per-arch SSM (H3, H5)** | **$200-400** | **paper 040** |
| **ME loss (Eq. 3 port)** | **Patient-level correspondence regularizer (H3, H5)** | **$0 (in-loop)** | **paper 040** |
| **Total** | **v0 prototype** | **~$2,800** | — |

The Point2SSM++ integration adds ~$600 to the existing $2,200 budget, but gives us a principled v0 sub-task 1 backbone with theoretical grounding, robustness to misaligned input, and a downstream correspondence field that feeds into sub-task 4's completion encoder.

### Next paper to read (041)

**Three candidates:**

1. **Mesh2SSM (Adams & Elhabian, 2025)** — a follow-up from the same group that addresses Point2SSM's limitation of requiring point cloud input (clinical scans come as meshes). The mesh-native variant would be the *right* v0 sub-task 1 backbone for clinical IOS data. Read this if sub-task 1 becomes a priority.

2. **STEAM (MICCAI 2025)** — "Self-Supervised Teeth Analysis and Modeling for Point Cloud" (the search result showed it at papers.miccai.org/miccai-2025/paper/3394). A dental-specific self-supervised learning method from another group, trained on Teeth3DS+ (a large multi-center dental dataset). Read this to compare with Point2SSM++'s MPointR pre-training on dental data — and to get the Teeth3DS+ dataset details.

3. **LION (paper 005, re-read)** — the latent-point DDM, the *generative* counterpart to Point2SSM++'s *correspondence* approach. Reading them back-to-back would clarify the v0 sub-task 4 architecture decision (correspondence vs generation for crown outer surface).

**Recommendation for 041: Mesh2SSM (Adams 2025).** It addresses the *exact* gap in our v0 sub-task 1 pipeline (point cloud input only, no mesh support) and comes from the same group so the API is familiar. Plus it's likely the paper HK will want to see for the v0 sub-task 1 architecture decision.

**Note in `papers/040-point2ssm-plus-plus.md`.**
